# 方案：线程角色编译期定死 —— T_HMX 专属线程 + HVX 池双执行器架构

> 基于 hexagon-mlir@`hmx` 分支 HEAD `a5761822`（2026-10-01）+ d9 调研（LLVM/IREE/Triton/Glow/ORT 先例；d9 调研报告未入库，关键先例与 file:line 已内联本文）。
> 按本仓 §6.2 模板组织：契约 → 决策点 → 代码 → FileCheck → host → 设备 A/B → 记录。
> 本文所有本仓引用带 file:line；未读/未测处显式标 [未验证]。

> ## ⚠️ 全文约定：S1/S2/S3 比值一律是「**我们 ÷ llama.cpp**」
>
> **>1 = 我们慢，<1 = 我们快。** 依据 `hexagon-mlir/ROADMAP.md:50`（「S2/S3 已快过手写」）
> 与 `docs/hmx/hmx-next-round-plan.md:304-307`（表头「vs llama」列的 52.25/56.71/15.41 是
> **llama 的时间**；验算 59.5/52.25=1.139、50.5/56.71=0.890、9/15.41=0.584）。
>
> **本约定是 2026-10-01 晚补加的** —— 原稿未写约定，而主笔在审阅时正是在这里读反了方向
> （把 0.58× 当成"我们慢 1.7 倍"，据此以为 S3 是主要战场；实际 S3 是我们**快** 1.7 倍）。
> **任何按「ratio < 1 = 我们慢」来读的读者都会得出与 §1.2 相反的结论。**
>
> ⚠️ **0.92 与实测表的 0.89 并存，无人裁决 —— 标「待权威源裁决」**。
> 两套数：`ROADMAP.md:50` = 1.13/0.92；`docs/hmx/hmx-next-round-plan.md:306` 实测表 = 1.14/0.89。
> **本文引用时抄 `ROADMAP.md:50` 那一套，但这是权宜之计，不是裁决。**
> ⚠️ **`ROADMAP.md:50` 自称的权威源 `docs/state/STATE-OF-PLAY.md:559` 引用链已断**
> —— 该行讲的是 `FastInversePass` 除法反号，与本比值无关。
> ⇒ **待办：指定一个权威源，或重测一次同构建 S1/S2/S3，消灭这组分叉。**

---

## 0. 一句话

把「这段代码跑在哪种引擎线程上」从**运行时事实**变成**编译期事实**：
编译器把每个 kernel 切成 HMX-role 与 HVX-role 两类区域（判据=硬件能力，零成本模型），
运行时只提供两个执行器——**一条进程级专属 HMX 线程（ensure 一次、锁跨算子持有、永不接触 HVX）**
+ **现有 HVX 线程池**——两线程间用 SPSC 环接成软件流水，
让 **HVX 侧 `pack_act(i+1)` / `unpack(i)`（+ `hmx.stage` DMA 提交）**
与 **HMX 侧 `mma(i+1)` / `bias_load` / `acc_read`** 真正重叠。
（**这是全文唯一权威的重叠表述**；§2 给出引擎归属边界的定义，§5 给出验收。
原稿 §0/§2/§5 三处互相矛盾，已按 2026-10-02 决策统一为此式。）

---

## 1. 对现有架构的判定

### 1.1 保留（这些是对的，方案全部复用）

| 保留项 | 证据 | 为什么对 |
|---|---|---|
| 逐 op 归属的能力谓词（shape/dtype/对齐/VTCM 预算 + reason code + remark） | `MatmulToHmxPass.cpp`（ROADMAP §1.2） | 与 Triton `getMMAVersionSafe`（AccelerateMatmul.cpp:42-83）同型，是"能力驱动自动切分"的正确形态；M3.1 退役结论（不建成本模型）继续成立 |
| `hmx.stage/await` 值边 + DMA staging tile 环 | ROADMAP §4.3、"机制就绪 stage/await 值边 ✅" | 跨线程流水的现成 IR 契约——stage/await 本来就是"提交 DMA/等待完成"的值边，今天在同一线程交错，明天天然跨线程。⚠️ **但它不是 §3.1 的判据刀口**（那是 DMA 语义，serial 路径上不存在）——**刀口是引擎归属边界**，见 §2 |
| `OpTrait::HmxDmaOnly` 结构性极性 | HmxToLLVMPass.cpp:101-110（"every *unmarked* dialect op counts as engine until proven otherwise…错并到 HVX 侧=挂死，错并到 HMX 侧=多付一对锁"） | 本方案把这个极性从"要不要锁"推广为"跑哪条线程"，继承同一 fail-safe 方向 |
| 单 HMX 资源 + "引擎不让出线程"前提 | ROADMAP M3.3、§2.4 | 硬约束（qurt_hmx.h:184 [未验证：SDK 头，不在本仓]，实测 0.998），方案的出发点 |
| 测量与门纪律 | 同构建双指纹、max(3×CV,15%)、owner+退出条件、证据指针 | 方案分阶段全按此执行 |

### 1.2 推翻（三处，都给证据）

**推翻一：per-kernel ensure/unlock 不应是终态锁粒度。**
- 现状：`HmxToLLVMPass.cpp:76-79` 定义 ensure/unlock 叶子；`:590-591` "The lock is one pairing per kernel: one ensure at entry, one unlock before each return"；`HexagonCAPI.cpp:205-219` 实现（线程局部 `tHoldsHmxLock` + `HAP_compute_res_hmx_lock`）。
- 三个问题：
  1. **失效模式零容忍**：你们自己的注释（HmxToLLVMPass.cpp:96）——漏一次 unlock，下一个线程在 `HAP_compute_res_hmx_lock` 上**永久挂死**。持锁人是"任意执行该 kernel 的线程"。
  2. **税**：每 kernel 一次锁往返；NON_SHARED unlock 还清 accumulator（HexagonCAPI.cpp:212-213）。属 M3.2 要削的 per-launch 固定税。
   3. **拓扑病根**：async 池里任何线程都可能既跑 linalg fallback（HVX）又跑 HMX 段——正是 LLVM PR #222340 TTI 注释描述的上下文饥饿拓扑
      （**2026-10-02 已 backport 进本仓**，见 §5.0）：`llvm_triton/llvm-project/llvm/lib/Target/Hexagon/HexagonTargetTransformInfo.cpp:453-462`
      （`areInlineCompatible` 定义起于 `llvm_triton/llvm-project/llvm/lib/Target/Hexagon/HexagonTargetTransformInfo.cpp:451`）原文：
      *"The hardware provides a fixed number of HVX contexts. Software that mixes the two engines dedicates some threads to HVX, and those threads hold the contexts for as long as they run. A thread dedicated to HMX needs no context at all, until HVX code reaches it."*
- **修正**：锁的所有权 = 一条专属 HMX 线程 × 整个会话。T_HMX 在首次 HMX launch 时 ensure 一次、持有；其余线程永不触碰 HMX 锁。竞态为零，"挂死"失效类整体消失，锁往返每会话 2 次。per-kernel 配对降级为 legacy 路径（A/B 与回滚需要它）。

**推翻二："线程路径已四次证伪"不应遮蔽 M3.3——四次证伪的是工作切分，不是角色切分。**
- ROADMAP §2.4：grid 并行最终在 runtime HMX lock 处串行化。那是 N 线程抢 1 个 HMX，必然串行。
- 角色切分 = 1 条 HMX 线程干 HMX、N 条 HVX 线程干其余：锁无竞争，per-thread 互斥约束**由构造满足**。
- llama.cpp 手写对照组就是这个拓扑（专属 HMX 线程 + HVX worker + 256 深环 + 软件流水）。**S2/S3 已快过它 0.92×/0.58×**（同一次实测里分别快 8% / 快 1.7 倍）说明你们 leaf 质量已经够，**跨线程不是这两个形状落后的原因**；**唯一还输的是 S1，输 13%（1.13×，2026-09-21 跨构建读数，仅作方向参考）**。本方案就是把这 13% 的机制编进编译器。

  ⚠️ **但"输的恰是它的跨线程流水"这一步是推断，不是实测。** 可隐藏量是**时间**，不是指令数：
  S1 的 LWP 分区（`docs/hmx/hmx-next-round-plan.md:61-70`，`5cea8231`/125 µs 那次）为
  **`unpack` 37.4% > engine 35.2% > `pack_act` 9.8% > residual 12.8% > `pack_weight` 4.7%**。
  ⇒ **要吃掉的 13% 落在 `unpack`（37.4%）上，不是 `pack`（14.5%）** ⇒ 天花板差 2.6 倍。
  ⚠️ 那次 LWP 是 125 µs 的构建，本文引用的 59.5 µs 是另一次（WR 开），**跨构建**，
  ⇒ **上界只作方向参考，落地前需同构建 A/B 重测**。

**推翻三：角色只有 op 粒度、没有区域载体——"半 HMX kernel"上报缺陷的根因。**
- M1.1 实测（ROADMAP §M1.1 注）：43 臂中 8 个静默回退 HVX（6 个有精确 reason code 但没人看到）；"vtcm-budget 产出半 HMX kernel（3 个链式 matmul 收 2 拒 1，报告成功、三分之一在 HVX、无任何标记）"。
- 修正：角色升为区域/函数级 IR 事实（§3），launcher 必打每 kernel 角色判定（"有人读"这一环接上 `tools/hexmlir/manifest_verdict.py`）。

顺带（非主线，按你们 §2.1 已给选项）：`enableHexKL` 三选一推荐 ① 删除。

---

## 2. 目标架构

```
                compile time                            runtime
Triton → TTIR → Linalg
        → MatmulToHmxPass            （逐op能力归属，不动）
        → 【新】ThreadRolePartition  （判据=纯能力，零成本模型）
              HMX-role  = 非HmxDmaOnly 的 hmx op + 纯标量胶水闭包
              HVX-role  = 其余一切（linalg/vector/scf/stage/await/DMA）
              混合且切不开 ⇒ 保持今日单线程语义 + reason code
        → 单角色 kernel ──────────► 今日路径，零变化（保 S3 类短 kernel）
        → 双角色 kernel ─► HMX 段 outline 成 @k.hmx_N
                            │（LLVM fn attr "hexagon_hmx"）
                            ▼
  T_HVX 池（现有 HexagonThreadPool）◄──SPSC环──► T_HMX（进程单例线程）
  · linalg/epilogue/softmax 链        (VTCM tile   · 只跑 hexagon_hmx 函数
  · hmx.stage = DMA 提交（引擎无关）    描述符+事件字) · mma / bias_load / acc_read
  · pack_act(i+1)  ‖  unpack(i)                    · ensure 一次，锁跨算子
                                            · 永不 acquire HVX 上下文（编译期保证）
```

> ## ✅ 决策已定（2026-10-02）：**方案 A** —— `pack`/`unpack` 归 T_HVX
>
> **原稿的矛盾已解除。** 原稿在 T_HMX 侧同时列了 `pack`/`unpack`、又声明 T_HMX
> 「永不 acquire HVX 上下文」，两句不能同时成立。**证据（可复算）**：
> `bin/runtime/hmx/src/HMXLayout.c` 里**引擎 intrinsic 零命中**
> （`grep -cE 'Q6_mx|mxmem|mxclracc|Q6_bias|Q6_activation|Q6_weight' HMXLayout.c` = 0），
> 而**含 `Q6_*` 的有 34 行 / 17 个唯一符号，全部是 HVX 向量 intrinsic**
> （`Q6_vmem_QRIV` / `Q6_vscatter_RMVhV` / `Q6_W_vdeal_VVR` / `Q6_V_vror_VR` …）
> ⇒ **`pack_act` 与 `unpack_acc` 是纯 HVX 代码。**
> 对照：引擎侧 5 个唯一符号全在 `HMXAPI.c`
> （`Q6_mxmem_AR_after_hf` / `Q6_mxclracc_hf` / `Q6_bias_mxmem2_A` /
> `Q6_activation_hf_mxmem_RR_deep` / `Q6_weight_hf_mxmem_RR`）。
> **现按方案 A 定稿。**
>
> **三条理由（记录在案，供日后回溯）：**
>
> 1. **选 B 会让上游 `hexagon_hmx` 语义失效。** 该属性的前提是**函数内没有 HVX 代码**
>    （`llvm_triton/llvm-project/llvm/test/CodeGen/Hexagon/hmx-attr-no-autohvx.ll:2` 原文
>    "so no HVX unit is acquired on the HMX thread"——**2026-10-02 已随 backport 进本仓，
>    逐字核对通过**）。选 B 等于放弃「**线程契约由编译器静态保证**」这个方案核心，
>    退化成「把单线程的锁挪进一条线程」，**收益趋零**。
> 2. **量化上天花板差 ~3.6×。** S1 的 LWP 口径（`docs/hmx/hmx-next-round-plan.md:61-70`，
>    跨构建 `5cea8231`/125 µs，**仅方向参考**）：`engine` 35.2% · `pack_act` 9.8% ·
>    `pack_weight` 4.7% · `unpack` 37.4% · `residual` 12.8%。
>    **方案 A** ⇒ T_HMX 串行链只剩 **35.2%**，`pack_act` 9.8% + `unpack` 37.4% **全部**变成 HVX 侧可重叠对象。
>    **方案 B** ⇒ T_HMX 链 ≈ **35.2 + 47.2 = 82%**，跨线程只剩 `residual` 可动。
> 3. **A 顺带化解 §6.10 的「pack 还是 unpack」两难** —— 两个都归 HVX 侧，不必二选一。
>
> **代价与对冲见 §4.5**：必须新增 `HmxLayoutHvx` trait，
> **否则 T_HVX 上的 pack 会去抢 HMX 锁 ⇒ 永久挂死。**

> ### 重叠哪一半 —— 唯一权威表述
>
> **§0 / §2 / §5 三处原稿互相矛盾**（`:15` 说 pack；`:70` 把 pack+unpack 同时列 T_HMX；
> `:71-72` 又同时列 T_HVX；`:163` 说 pack）。**现按决策 3 统一为：**
>
> ```
>   HVX 侧:   pack_act(i+1)  ‖  unpack(i)
>   HMX 侧:                    mma(i+1) / bias_load / acc_read
> ```
>
> **重叠对象 = 引擎归属边界**（`pack_act`/`unpack_acc`/`hmx.stage` 纯 HVX
> ⇔ `hmx.mma`/`acc_read`/`bias_load` 纯引擎）。**这条边界就是 SPSC 环的刀口。**
>
> ⚠️ **它与 §3.1 原先写的 stage/await 值边不是同一条线** —— 后者是 DMA 语义
> （DDR→VTCM 传输），**serial 路径上没有 `hmx.stage`/`hmx.await` 可切**
> （FA 的 QK 就在那条路上：`logs/real-shapes-2026-09-29/attn_qk_d128.manifest.json`
> Kt=4 → `serial:shallow-k`；`attn_qk_d256` Kt=8 同样）。**判据已按此改写，见 §3.1。**
>
> **可藏量口径**：`enableWeightResident` 默认开 ⇒ `pack_weight` ≈ 0
> ⇒ 可重叠对象 ≈ `pack_act` 9.8% + `unpack` 37.4% = **47.2%**（跨构建 LWP，**仅方向参考**）。

四个关键选择：

1. **T_HMX 的"纯度"由编译器保证，不靠纪律**：所有 T_HMX 上跑的函数带 LLVM fn 属性
   `"hexagon_hmx"`（上游 PR #222340，2026-09-17 合并，commit `2e055b8de1a1`，Qualcomm 参与）。
   上游 TTI 保证：`useHVX() = … && !IsHMX`（该函数永不auto-HVX）+ `areInlineCompatible()`
   双向拒绝跨角色内联。语义与测试见 llvm_triton/llvm-project/llvm/test/CodeGen/Hexagon/hmx-attr-no-autohvx.ll:1-2 与 hmx-attr-inline-compat.ll:1-41（**2026-10-02 已随 backport 进本仓**）
   （"so no HVX unit is acquired on the HMX thread"）——**这正是本方案线程契约的官方表达**。
2. **HVX 侧复用现有 async 底座**：`FormAsyncThreadsPass.cpp:10` "lowering virtual-threads to
   async.execute"（现限 rank-1 forall）→ `bin/runtime/multithreading/HexagonThreadPool`。
   上游 async dialect 无 affinity（d9 已查证），所以角色属性由**自己的** async→runtime
   lowering 消费（路由到哪个执行器），不动上游 async。
3. **桥 = SPSC 环**：生产者（HVX 池线程）写 tile 描述符，消费者（T_HMX）取。事件 ABI 抄
   HmxToLLVMPass 已有的 64-bit 不透明事件字 wire 纪律（`readEventWord` 一节）。环深由
   HmxPartitionPass 的 tile 环推导，不拍脑袋。
4. **拓扑惰性**：manifest 判定 single-role ⇒ 不建线程、不付任何手付开销。短 kernel 零回退。

---

## 3. 编译期角色切分：判据 / 载体 / 检查 / 可观测

### 3.1 判据（两级，**正交，不是互斥枚举**）

> ⚠️ **原稿把两级字段写进了一张表，表头却说"每个一个 canonical reason code"（暗示互斥）。
> 一个 dual-role kernel 会同时持有区域级 `role-hmx` 与 kernel 级 `role-split-ok` ⇒ 原表不自洽。**
> 现拆为两级。

**① 区域级（每个 region 一条）**

| 判定 | 规则（纯能力，零成本模型） | reason code |
|---|---|---|
| region→HMX | 含**非** `HmxDmaOnly` 的引擎 op（mma/acc_read/bias_load），且区域内其余 op 为标量胶水 | `role-hmx` |
| region→HVX | 无引擎 op | `role-hvx` |

**② kernel 级（每个 kernel 一条）**

| 判定 | 规则 | reason code |
|---|---|---|
| 混合可切 | 引擎段与向量段之间有**引擎归属边界**可下刀 | `role-split-ok` |
| 混合不可切（依赖交织） | 引擎 op 与向量 op 依赖交织、无刀口 | `role-mixed-irreducible` → 单线程语义（=今日行为） |
| 混合不可切（**无刀口**） | **tile 循环体内根本没有 pack** ⇒ 无可重叠对象 | **`role-split-nopack`** ⛔ 新增 |
| 混合但环放不下 | 切得开，但 SPSC 环的内存预算不够 | **`role-split-nobudget`** ⛔ 新增 ⇒ 退回单线程。现成形态见下 |
| 全 kernel 单角色·引擎 | — | `topology-single-role-hmx`（要 T_HMX 线程） |
| 全 kernel 单角色·向量 | — | `topology-single-role-hvx`（**不建线程**） |

**⚠️ 三个新增/修订分支的证据：**

- **`role-split-nopack`**：`HmxPartitionPass.cpp:679-706` 的 `emitSerialTileLoop` 函数体只有
  `AccClearOp` → `emitMmaKLoop` → `AccReadOp`，**循环体内无 pack、无 stage**。
  pack 在**整数组 prologue**（`MatmulToHmxPass.cpp:1139-1163`）。
  manifest 交叉验证：`logs/real-shapes-2026-09-29/s1_anchor.manifest.json` 的
  `pack_act_sites: 1`（整数组循环）vs `s2_anchor.manifest.json` 的 `pack_act_sites: 2`
  （稳态 + peeled，per-m-tile，来自 `HmxPartitionPass.cpp:1492` `emitPackAct`）。
  ⇒ **这不是 `role-mixed-irreducible`（依赖交织），是"没有刀口"。**
- **`role-split-nobudget`**：现成形态在 `HmxPartitionPass.cpp:1626-1631`
  （`fits(2)/fits(1)/fits(0)` → `budgetDepth==0` 时 decline）。
  ⚠️ **§4.3 要造的是第二个环（SPSC），它的内存预算目前无任何判据。**
- **单角色要拆两个值**：单角色且全 HVX **不建线程**；单角色且是引擎 **要 T_HMX 线程**。
  运行时后果完全不同，原稿一个 `topology-single-role` 表达不了。

> ⚠️ **刀口不止一种，原稿只写了一种**：§3.1 原来的刀口是 **stage/await 值边**，
> 而那是 **DMA 语义**（`hmx.stage`/`hmx.await` = DDR→VTCM 传输）。
> **但跨线程重叠需要的是引擎归属边界**（`pack`/`unpack` vs `mma`/`acc_read`），两者不是同一条线。
> **serial 路径上没有 `hmx.stage`/`hmx.await` 可切** ⇒ 原判据在 serial 形状上恒不成立。
> 实证：FA 的 QK 落在这条 —— `logs/real-shapes-2026-09-29/attn_qk_d128.manifest.json`
> Kt=4 → `serial:shallow-k`；`attn_qk_d256` Kt=8 同样。
> ⇒ **判据需要第二种刀口，或显式承认 S4a/S4b 在 QK 那一侧不成立**（见 §5 S4a/S4b 行）。

与 M3.1 退役结论的关系：**判据里没有任何"哪个更快"**——MatmulToHmx 继续回答"能不能上 HMX"，
本 pass 只回答"在哪个线程跑"，两者正交。

### 3.2 IR 载体

- `hex.thread_role<"hmx"|"hvx">` unit attr，挂 async.execute 与被 outline 的函数；
- LLVM 侧映射为 fn 属性 `"hexagon_hmx"`（hmx）/无属性（hvx）；
- 选 attr 不选新 op 的理由：IREE 的教训是 affinity 载体设计了但 no-op 两年
  （StreamInterfaces.td:28 "Today all affinities are no-op'ed"，你们已查证）——
  我们的 attr 从第一天就有消费者（自家 AsyncRuntime 路由 + launcher 打印 + manifest）。

### 3.3 静态检查（每条配 FileCheck，先写测试再改 pass）

1. `hex.thread_role<"hmx">` 函数体内不得出现超宽 vector 类型（verifier 硬错；
   对应上游 `useHVX` 门控语义）。
2. 跨角色禁止 inline/clone 合并：LLVM 层由上游 TTI 管（双向）；自家 clone/桥消除 pass
   （`preferCloneToConsumers` 等）加同极性检查——沿用 HmxToLLVMPass.cpp:101-110 的失效方向：
   错并到 HVX 侧=挂死，错并到 HMX 侧=多付一对锁，所以 fail-safe 方向是"未证明即 HMX"。
3. 分区聚合收敛：一个 async.execute 单角色（IREE `Partition::verify` 形态；其 joinAND/joinOR
   "divergent affinities not yet implemented" assert 正是缺这个检查的教训）。
4. VTCM 所有权跨线程交接必须显式：tile 所有权随事件字移交，禁止两线程同时写同一 tile
   （静态侧查 def-use，动态侧靠环协议）。

### 3.4 可观测（把 M1.1 的"没人读"一起修掉）

- manifest 增字段：`topology`（single-role / dual-role / legacy）、每 region 的 role+code、
  outline 段数、环深；
- launcher 每次编译后打印 per-kernel 角色判定行（复用 `manifest_verdict.py` 格式），
  Triton 路径默认可见；
- 任何回退/降级 emitRemark（Triton AccelerateMatmul.cpp:67-76 同型——降级必发声）。
  反面教材是 Edge TPU：自动切分+静默回退=性能悬崖+无人知晓（arXiv:2102.10423；
  你们 43 臂里 8 个静默回退已在同一条路上）。

---

## 4. 运行时改动清单（最小集）

1. **角色执行器**：HexagonThreadPool 加"绑定线程执行器"概念（IREE `deferred_work_queue`
   的 `bind_to_thread` 同型抽象，接口 4 个：bind/submit/drain/join）。提交任意线程、
   执行固定线程。
2. **锁所有权迁移**：`EnsureHmxLockForThisThread`（HexagonAPI.cpp:283-291）只在 T_HMX
   生命周期调用一次；可选 idle>N ms 释放策略（默认不释放——单实例部署你们已拍板，
   workspace-resident 同一先例）。per-kernel 配对保留为 legacy 路径，由 manifest 的
   topology 字段区分，供 A/B 与回滚。
3. **SPSC 环**：新原语（tile 描述符 + 两个原子索引 + 事件字），acquire/release 语义按
   Hexagon 内存模型钉住 [未验证：fence 选型]。host 单元测试 + 吞吐探针先行。
4. **accumulator 纪律**：锁跨算子 ⇒ accumulator 跨 kernel 持久。你们 lowering 已显式
   `acc_clear`（`HmxToLLVMPass.cpp:1196` 映射 `hmx.acc_clear` → `hmx_acc_clear_f16()`，测试有钉）——
   补一条不变量：任何 `hmx.mma` 前必须可见 acc 初始化（FileCheck 钉）。
   ⚠️ 现状这条保证**只存在于一条注释里**：`bin/runtime/src/HexagonCAPI.cpp:212-213`
   「A NON_SHARED unlock clears the accumulators per qurt_hmx.h, so it must come after the last
   read, **which the compiler's position guarantees**」。锁跨算子持有 ⇒ 这个 unlock 清零不再发生
   ⇒ **注释里的保证失效，必须变成测试。**
   **动作（已定，2026-10-02）**：把这条从注释升为 **FileCheck 测试** ——
   ① 正向：任何 `hmx.mma` 支配链上必须可见 `hmx.acc_clear`；
   ② **反向（今天的关键缺口）**：跨 kernel 场景下**不得**依赖 unlock 清零，
   即 `enableThreadRolePartition=ON` 时 acc 的初值必须由显式 `acc_clear` 提供。
   ⚠️ 这是 EdgeTPU「保证只存在于注释里」教训的**编译期版本**。

5. ✅ **引擎归属的第三类标记 `HmxLayoutHvx`**（**缺了会永久挂死；设计已定，2026-10-02**）
   **问题**：`issuesHmxEngineLeaves`（`HmxToLLVMPass.cpp:527-536`）把**任何**非 `HmxDmaOnly`
   的 hmx op 当引擎 op。而 `pack_act`（`HmxOps.td:156-158`）、`pack_weight`（`:228-230`）、
   `unpack_acc`（`:288-290`）**三者都没有 `HmxDmaOnly` trait，也都没有 `HmxEngineResource` effect**
   （全 `.td` 只有 2 个 `HmxDmaOnly` 载体：`stage:570` / `await:623`）
   ⇒ **今天 pack/unpack 被算成引擎 op。**
   **若 pack/unpack 移到 T_HVX 而不新增「要 HVX、不要引擎」的第三类标记，
   T_HVX 上的 pack 会去抢 HMX 锁 ⇒ 永久挂死**——正是 `HmxToLLVMPass.cpp:96-99` 警告的失效模式。
   ⚠️ **§1.1 说"继承同一 fail-safe 方向"，但这不是继承，是要新增一个 trait。**

   **设计（已定）：**
   - **新增 `NativeOpTrait` `HmxLayoutHvx`**（语义："HMX 布局代码，由 HVX 实现"），
     定义在 `HmxDialect.h`（与 `HmxDmaOnly:160` 并列）。
   - **加在恰好三个 op 上**：`pack_act` · `pack_weight` · `unpack_acc`。
   - **`issuesHmxEngineLeaves` 判据改为**：
     `无 HmxDmaOnly 且 无 HmxLayoutHvx ⇒ 引擎`（即两个 trait 都是"排除引擎"的标记）。

   **失效方向重审计（两个方向都不静默）：**

   | 误标 | 后果 | 可诊断性 |
   |---|---|---|
   | 布局 op 误标为**引擎** | 多付一对 ensure/unlock，函数自己会还 | 与今日行为完全相同，**无害** |
   | 引擎 op 误标为**布局** | 未持锁发引擎指令 ⇒ **设备 abort** | **响亮失败，好诊断** |

   ⇒ **fail-safe 极性仍是「未证明即引擎」**（`HmxToLLVMPass.cpp:106-109` 的原意不变），
   逐 op 显式授予豁免。**新的错误方向比旧的方向更容易发现，不是更难。**

   **再加一道双向一致性 lit**：trait 分类 ↔ 叶子符号分类两张表互钉 ——
   引擎叶取自 `HMXAPI.c:32,37,43,53-54`（5 个唯一符号），
   布局叶取自 `HMXLayout.c`（34 行含 `Q6_*` / 17 个唯一符号，**引擎 intrinsic 零命中**），
   **任一边漂移即变红**。
   ⚠️ **配对决策仍由 trait 做，不违反「不许按 callee 前缀决定配对」的现行规则**
   （`HmxToLLVMPass.cpp:113-115` 明确禁止按前缀判断）——**这张 lit 是审计表，不是配对表。**

   **Legacy 路径零影响**：trait 只改「算不算引擎」；legacy 下 pack 照旧被锁，无害。

6. ⛔ **host→device 的分流通道不存在**
   §4.2 说 legacy 路径「由 manifest 的 `topology` 字段区分」。但**设备 runtime 今天完全不读 manifest**：
   `bin/runtime/src/HexagonAPI.cpp` 与 `HexagonCAPI.cpp` 里 `grep manifest` 与
   `grep topology` **均 0 命中**。manifest 只到 host launcher
   （`backend/triton_hexagon_launcher.py:557-568` → `backend/utils.py:1658` `enforce_hmx_launch_contract`）。
   ⇒ 要按 `topology` 在设备侧分流，**需要一条新的 launch 通道**
   （launch 参数 / per-kernel 符号 / 环境变量）。**manifest 字段本身不够。**
   **默认候选（已定方向，三选一留给 S2 设计）：per-kernel 符号方案** ——
   HMX 段函数保留 `.hmx_section` 类稳定后缀，运行时 `dlsym` 探测分流。
   **理由**：比 launch 参数**侵扰小**（不改现有 launch ABI）、比环境变量**可组合**
   （可逐 kernel 判定，不受进程环境污染）。

---

## 5. 分阶段落地（每阶段可单 commit 回滚；门 `enableThreadRolePartition` 默认 OFF）

| 阶段 | 内容 | 验收 | 备注 |
|---|---|---|---|
| **S0**（~半天） | 查 pinned llvm_triton/llvm-project 是否含 `2e055b8de1a1`；无则 backport TTI 两个 hunk（~30 行）。HMX leaf 函数带 `hexagon_hmx` 编译通过 | lit 全绿 | 零行为变化 |
| **S1**（host 全验） | ThreadRolePartition pass + attr + verifier + manifest 字段；默认只 emit 单角色 | FileCheck 全套（成功/拒绝/mixed-irreducible/半HMX→PARTIAL+dual-role）；零行为变化 | 纯编译期 |
| **S2**（运行时底座） | 角色执行器 + T_HMX + SPSC 环 + 锁迁移（legacy 共存）；4 个探针：环吞吐、锁长持、DMA 跨线程等待、VTCM 跨线程 alloc/free | host 单元测试 + 探针报告；不跑真 kernel | [未验证]×4 见 §6 |
| **S2.5**（⛔ 新增前置） | **给 S1-class 引入 per-tile pack**：把整数组 prologue 的 pack（`MatmulToHmxPass.cpp:1139-1163`）折进 tile 循环，让 S1 形状**有可重叠对象** | 纯 host：manifest `pack_act_sites` 从 1 变 2（对齐 `s2_anchor`）；lit 全绿 | ⛔ **S3 的硬前置**。不做这步，S3 在 S1-class 上按 §3.1 自己的判据就是 no-op（`role-split-nopack`） |
| **S3**（首个双线程 kernel） | **S2-class** matmul（`256×64×2048`，Kt=64，**唯一已有 per-tile pack 的稳态形态**）：<br>**HVX 侧 `pack_act(i+1)` ‖ `unpack(i)`；HMX 侧 `mma(i+1)`/`bias_load`/`acc_read`** | ① 同构建双指纹 A/B，**`N ≥ 1000`**（once_share 1.70%），≥ max(3×CV,15%)；判决**四选一**见 §5.1.3<br>② ⭐ **LWP 归因探针臂：显式输出「跨线程相对单线程已有 37% 重叠的净增量」**<br>③ ⛔ **先过 §5.1.7 的 reject 判据**（`budgetDepth` 下降 ⇒ 不进 A/B） | ⚠️ 不能是 S1-class（见 S2.5，S1 的 A/B 待 S2.5 后补）。⛔ **不得以 47.2% 为预期**——S2-class 已有一笔 **1.92× staging 重叠**入账（`hmx-perf-findings-2026-09-27.md:183`/`:293`），真实上限更低（§5.1.5②）。⛔ **47.2% 与 37% 不许相减**（§5.1.5①）。⚠️ **净增量 < 门 ⇒ 默认保持 OFF + 负结果收档** |
| **S4a**（⛔ 由 S4 拆出 · 观测台架，**无 FA 性能门**） | 搭 LWP 归因的重叠率**观测台架**（只测不承诺）；量 M3.2（锁持有后 per-launch 固定税降幅） | **LWP 归因的重叠率报告**（不设 FA 性能门）；<br>**只有 M3.2 减税那项**套 `max(3×CV,15%)` + §5.1 的 N 规则 | ✅ **纯拓扑过门不可达已接受**（引擎份额 0.6% ≪ 15%），本阶段改为先把测量能力建起来 |
| **S4b**（⛔ 由 S4 拆出 · 真收益） | **组合机制**：FA 的 15% = **softmax 链去串行化（43.7%，M4.1 工作面）+ 本拓扑提供并行底座** | 组合门：softmax 侧与拓扑侧**合并**计 ≥ max(3×CV,15%)<br>⛔ **必须单独列交互项** `A_both − max(A_topo, A_softmax)` | ⛔ **拓扑单独份额 ≤ 0.6% 写死在本文档里，不再宣称独立功劳。** ⛔ **合并门在数学上不可证伪拓扑**（§5.1.6）。⚠️ QK 落 `serial:shallow-k`（`attn_qk_d128` Kt=4）⇒ 重叠主体是 softmax 链，QK 走 serial 不影响；但 §3.1 的 stage/await 刀口在 QK 上恒不成立 |
| **S5**（收口） | S3/S4b 过门 ⇒ 报用户批准翻默认；per-kernel 配对降级 legacy-only；经验推上游（hexagon 侧 RFC / async affinity） | 门数字 + 契约评审 | 翻默认须用户批准（你们规则） |

**依赖：`S0 → S1 → S2 → S2.5 → S3 → S4a → S4b → S5`**

⚠️ **原稿"S1 与 S2 文件面不重叠，可并行"已撤回**，有两处真实的**写-读**依赖：

| 依赖 | 证据 |
|---|---|
| 门 `enableThreadRolePartition` 需照 `enableHmxPipelineDepth` 的现成**三处**接线 | `include/hexagon/Dialect/Hmx/Transforms/Passes.td:134`（`Option<"pipelineDepth", "pipeline-depth", "int64_t", /*default=*/"0">`）· `lib/Target/Linalg_MLLVMIR/MLLVMIRTranslation.cpp:154-155` · `backend/hexagon_options.py:157`（`enableHmxPipelineDepth: int = 0`）。**S1 与 S2 都碰这三处** |
| manifest `topology` 字段：**S1 写**（§3.4）、**S2 的 legacy 分支读**（§4.2） | ⇒ **S2 的第 2 项不可能在 S1 之前完成** |

⇒ **S1 与 S2 只在「探针 / 单元测试」这部分文件面确实不重叠、可并行。**

✅ **§2 的引擎归属硬矛盾已解决**（2026-10-02 定为**方案 A**）。
⇒ **`S2` 开工前的前置只剩一件：§4.5 的 `HmxLayoutHvx` trait 必须先落地**，
否则 T_HVX 上的 pack 会去抢 HMX 锁 ⇒ **永久挂死**（`HmxToLLVMPass.cpp:96-99` 的失效模式）。

---

## 5.0 ✅ S0 已完成：LLVM `hexagon_hmx` 已 backport（2026-10-02），附一处**机制限制**

**做了什么**：把上游 PR #222340（`2e055b8de1a1`，2026-09-17，Qualcomm）逐字 apply 进
`llvm_triton/llvm-project`。**用的就是上游 diff 本身**（`curl .../pull/222340.diff`），
不是凭记忆重写。`patch -p1` **5 个 hunk 全部命中**，只偏 `-8`（.cpp）与 `-1`（.h）行。

| 文件 | 改动 |
|---|---|
| `.../HexagonTargetTransformInfo.cpp:55` | `useHVX()` 加 `&& !IsHMX` |
| `.../HexagonTargetTransformInfo.cpp:451-468` | 新增 `areInlineCompatible`（含那段上下文饥饿注释） |
| `.../HexagonTargetTransformInfo.h:44` | 新增成员 `const bool IsHMX;` |
| `.../HexagonTargetTransformInfo.h:60` | 构造里初始化 `IsHMX(F.hasFnAttribute("hexagon_hmx"))` |
| `.../HexagonTargetTransformInfo.h:194` | 声明 `areInlineCompatible` |
| `llvm/test/CodeGen/Hexagon/hmx-attr-inline-compat.ll` | 新增（上游） |
| `llvm/test/CodeGen/Hexagon/hmx-attr-no-autohvx.ll` | 新增（上游） |

**patch 已落盘**：`tools/hexmlir/llvm-hexagon-hmx-attr.patch`（181 行，头部写明出处与应用方式）。
⚠️ **`llvm_triton/llvm-project` 是 tarball 不是 git 仓 ⇒ 这个 patch 是唯一的持久化凭据，
丢了就只剩重下一次上游 diff。** 已做 **revert → 逐字节回 baseline → reapply** 往返验证。

### 零行为变化：已证

本仓源码里 `hexagon_hmx` **零命中** ⇒ 没有任何函数带该属性 ⇒ `IsHMX` 恒 `false`
⇒ `useHVX()` 与 `areInlineCompatible` 都退化成原行为。**今天不重建就完全不生效。**

### ⛔ 一处机制限制（本节最重要的产出）

上游两个测试在我们树上**一个过一个挂**，查清了原因，**不是 patch 的错**：

| 测试 | 结果 | 原因 |
|---|---|---|
| `hmx-attr-no-autohvx.ll` | ✅ **过** | `!IsHMX` 生效，有属性的函数不再被 loop-vectorize 变成 `<32 x i32>` |
| `hmx-attr-inline-compat.ll` | ❌ **挂** | 它的两个 helper 都是 **`alwaysinline`**，而**我们 pinned LLVM 的 `AlwaysInliner` 绕过兼容性检查** |

**根因（本仓可查）**：`llvm/lib/Analysis/InlineCost.cpp:3224` 的注释写明
「Never inline functions with conflicting attributes (**unless callee has always-inline attribute**)」，
其上 `:3215-3222` 的早返回块在 `isInlineViable` 成功时直接 `return`，
**跳过了 `:3227` 的 `functionsHaveCompatibleAttributes`**（该函数在 `:3088-3089` 才问
`TTI.areInlineCompatible`，默认 `IgnoreTTIInlineCompatible=false`，即检查是开的）。

**实测确认不是「钩子没接」**：把两个 helper 的 `alwaysinline` 去掉、走普通 `-passes=inline`
（高阈值）后，**两个跨角色 call 都被拦住了**（各留 1 个 `call void`）。
⇒ **`areInlineCompatible` 对普通内联有效，只对 `alwaysinline` 无效。**

**对本方案的三条影响：**

1. **§2 的「双向拒绝跨角色内联」今天只对普通内联成立。** 上游那句「这正是本方案线程契约的
   官方表达」要打折读。
2. **承重的是 §3.3 第 2 条那个 pass 级检查**（自家 clone/桥消除 pass 加同极性检查），
   TTI 钩子是普通内联路径上的第二道防线。**不要把安全性押在 TTI 钩子 alone。**
3. 若确实需要 `alwaysinline` 也被拦，**要再 backport 一处 inliner 改动**——
   **超出 S0 范围，未做**，登记在此以免日后当成已覆盖。


### ✅ 那个问号已查实（2026-10-02）：HMX 段**不带** `alwaysinline`

对 `logs/fa_pure.ll`（212 个 `define`）逐个解析 attribute group：

| | 数量 | 是谁 |
|---|---|---|
| 带 `alwaysinline` | **23** | **全部是 `_hexagon_runtime_*` 数学叶**（`acos/asin/atan/ceil/cos/exp/floor/pow/rsqrt/sin/sqrt/tan/tanh` 的 `__vf`/`__vhf`） |
| 不带 | 189 | 含 `@attention_fwd_kernel`、6 个 `@async_execute_fn*`、2 个 `@hexagon_runtime_hmx_{ensure,unlock}_dsp` |
| **HMX/matmul/pack/unpack 里带 `alwaysinline` 的** | **0** | — |

⇒ **`alwaysinline` 那条缝对本方案不构成风险**：那些数学叶是 HVX 侧代码，
而 HMX 段（`mma`/`acc_read`/`bias_load`）不需要超越函数 ⇒ 不会出现在 HMX 段的调用面上。
⇒ **选项 1 成立**：承重的是 §3.3 第 2 条的 pass 级检查，TTI 钩子是普通内联路径的第二道防线。
⚠️ 若将来 HMX 段需要超越函数（S1 那种带激活的形状**要重新查**），
本条结论作废，须回头补 inliner 的 backport。

本仓流水线自己就跑 AlwaysInliner（`lib/Target/HEX_LLVMIR/LLVMIRTranslation.cpp:51`
`createAlwaysInlinerLegacyPass`），所以「有没有 `alwaysinline`」是每个形状都要重问的问题，
不是一次性结论。

### 未做（明确登记）

- ⛔ **没有重建 `install/` 树**（`build/install/lib/cmake/llvm`，hexagon-mlir 链的就是它）。
  ⇒ **patch 目前只对 `build/bin/opt` 生效**（已重建，4 步，1 分钟）。
  ⇒ **要让 hexagon-mlir 真正用上，必须重建 install 树，那会改 `libtriton.so`、作废设备锚点。**
  **这一步须用户批窗口。**
- ⛔ 没重建 `libtriton.so`。设备锚点仍是
  `ce26015e8efb75cc047515000c8ad70f` / `97af133e81fbc361bca3be10164b7bc8`。

### ✅ 2026-10-02 夜：f16 除法那条（项目已知最大单点）的两块基石都到位了

**与本方案无关，但它是项目当前已知最大的单点，登记在此备查：**
`fdiv <64 x half>` → **194 次 libcall**；`fdiv <64 x float>` → **2 次**。
根因是 LLVM InstCombine 的 binop 收窄把 `<64 x float>` 变成 `<64 x half>`
（HVX 没有向量 f16 除法 ⇒ 192 次 libcall）。

**今天补上了它的一块基石**：上游 \`320a8a4db872\`（PR #202489，2026-07-03）已 backport
（patch 落 \`tools/hexmlir/llvm-hexagon-fdiv-ninf-narrowing.patch\`，4/4 hunk @ offset −113，往返验证过）。
它**保留收窄、只修 miscompile**：收窄后的 binop 在更小类型里重算，
可能在宽运算有限处溢出成 inf，\`ninf\` 被原样拷过去就会产生 poison。
实测行为：\`fdiv nnan ninf <4 x float>(fpext, fpext)\` → 收窄成 \`fdiv nnan <4 x half>\`（**\`ninf\` 被清**）；
而 fptrunc 也带 \`ninf\` 时 → \`fdiv nnan ninf <4 x half>\`（**保留**）。
**收窄没被取消，只丢了不安全的那个标志。** LLVM InstCombine 全套 1804 个测试 0 失败。

⚠️ **这也是「不给 fdiv 加 \`nnan ninf\`」的独立理由**：在未打这个补丁的基线上，
\`fdiv nnan ninf <64 x float>\` 会被收窄成 \`fdiv nnan ninf <64 x half>\`
⇒ **正是那个 poison-on-overflow miscompile。**

**⛔ 未做**：两个 patch 都**只对 \`build/bin/opt\` 生效**（各自重建过 opt）。
**要让 hexagon-mlir 吃到，必须重建 \`build/install\` 树，那会改 \`libtriton.so\`、作废设备锚点**
⇒ 归 S0b 那条，需用户批窗口。

### ⛔⛔ 2026-10-02 夜：同构建锚点重测完成，**本方案的立项理由被削弱**

真机、`ITERS=1000`、`REPS=3`、指纹 `ce26015e` 前后一致、null 臂三个形状全 0.0%。
报告：`docs/results/phase0-1-same-build-anchor-2026-10-02.md`。

| shape | 旧锚点（`388b6a2e`, 09-21） | **新锚点（`ce26015e`）** | 变化 |
|---|---:|---:|---:|
| S1 | 69 µs | **55 µs** | **−20.3%** |
| S2 | 85 µs | **42 µs** | **−50.6%** |
| S3 | 13 µs | **10 µs** | **−23.1%** |

**若 llama 的 52.25/56.71/15.41 仍成立**（⚠️ 跨构建，不可引用）：

| shape | 新比值 | 原记录 | 含义 |
|---|---:|---:|---|
| **S1** | **1.053** | 1.13 | **还慢，但只慢 5.3%（原 13%）** |
| S2 | 0.741 | 0.92 | 快 26%（原 8%） |
| S3 | 0.649 | 0.58 | 快 35%（原 73%） |

> ⛔ **这张表整列是 `N=1000` 口径（含一次性成本 `B/N`），而 llama 那一列是稳态口径
> （24576~81920 runs）。** 逐行的口径修正见本节末尾的勘误：
>
> | shape | 本表（`N=1000`） | **稳态 `A`** | llama | 稳态比值 |
> |---|---:|---:|---:|---:|
> | **S1** | 1.053 | **52.87 µs** | 52.25 | **1.012**（持平） |
> | S1 + `enableWorkspaceResident` | — | **47.89 µs** | 52.25 | **0.917**（我们快 8.3%） |
> | S2 | 0.741 | 38.84 | 56.71 | 0.685 |
> | S3 | 0.649 | 7.89 | 15.41 | 0.512 |
>
> ⚠️ 上面的 `52.87` 用的是**更正后的 `B ≈ 2 700 µs`**；用旧的 2 105 是 52.89
> ⇒ **差异 < 0.002 µs，比值不变。**
> ⇒ **⇒ 「S1 还慢 5.3%」这句话本身要按 1.012（持平）读，配 `ws` 则是 0.917（我们快）。**

#### ⇒ 三条后果

1. **§1.2「推翻二」的全部动机是「S1 输 13%」。若那是 5%，可摘的果子小 2.6 倍。**
   S3 的验收写的是「吃掉 S1 差距的**全部或大部**」⇒ **目标本身变小了。**
2. **§5.1.5② 的判断只会更糟**（S2-class 自己已有一笔 1.92× staging 重叠，
   而 S3 首发形状就是 S2-class）⇒ **真实上限远低于 47.2%。**
3. **⇒ `87ae8df9`（Phase 0.4+0.5：S1 对象码计数 + 真机 PMU）从「重要」升为「决定性」。**
   **在知道「S1 那 5% 里跨线程机制占多少」之前，本方案不该开工。**

#### ⚠️ 两条限制，必须一起说

- **分辨率**：`perf` 的量化步长是 **1.0 µs**（全部取值只有
  `{6,10,34,38,41,42,43,51,55}`）⇒ S3 的 −40% 实际是 **−40% ± 7%**；
  S1/S2 分辨率 ±0.9% / ±1.2%，**那两个结论稳**。
- **S3 的 weight-resident 效应与旧记录差 15 个百分点**：本次 −40.0%，
  `AGENTS.md:131` 记 −25.0%（S1/S2 都复现良好，差 1.2 / 0.8 个百分点）
  ⇒ **S3 上还有别的变量没控住，不要把 −40% 当定论。**

#### ⛔ 未做（下一步的第一优先）

**没重测 llama.cpp 那一侧** ⇒ 上面的比值是**条件句**。
**且 depth 3/1/2 三臂没测**（kStage 地板那条）—— 已在同一次窗口补跑，结果见
`logs/phase0-1-anchor-2026-10-02/depth_ab_N1000.log`。

---

## 5.1 ⛔ 验收判据：必须带 N，且判决分三类（2026-10-02 定）

> **这一节是对 S3 / S4b 两行的前置修正。原表只写「≥ max(3×CV,15%)」，漏了量纲。**

### 5.1.1 为什么：那条判据量的不是性能

稳态口径下（`docs/hmx/hmx-next-round-plan.md:226`）：

```
Perf(N) = A + B/N          A = 真稳态边际 · B ≈ 1470 µs = 每调用一次性 bring-up
```

⇒ **对 `Perf(N)` 施加固定百分比，测的东西取决于 N**：
N 小 ⇒ 百分比里 `B/N` 占比大，实际在测「一次性开销变小了」；N 大 ⇒ 在测「边际变小了」。
**两个不同的物理量被同一个数字回答。**

已实测的后果：RoPE trig 那次效应 **27.76%**（> 15% 地板，效应本身够大），
但 sd 4.99 ⇒ **CV 17.96%** ⇒ 阈值 `max(53.9,15) = 53.9%` ⇒ 三趟全 NOT-PROVEN。
`docs/hmx/rope-trig-share-device-result-2026-10-01.md:56`：**加样本救不回来**
（CV 的分子 sd 估的是散布不是标准误）。

### 5.1.2 ⛔ 硬规则：`once_share ≤ 2%` 才许下判决

```
once_share(N) = (B/N) / (A + B/N) ≤ 0.02   ⇔   N ≥ 50·B/A
```

按本方案三个形状的 `A`（`hmx-next-round-plan.md:268`，S1 69 / S2 85 / S3 13 µs）：

| 形状 | A | N≥300 | **N≥1000** | N≥3000 |
|---|---:|---:|---:|---:|
| S1-class | 69 | 6.63% | **2.09%** | 0.71% |
| **S2-class**（S3 首发形状） | 85 | 5.45% | **1.70%** ✅ | 0.57% |
| S3-class | 13 | 27.37% | 10.16% | 3.63% |

⇒ **S3 用 `N ≥ 1000`**（S2-class，once_share 1.70%）。
⚠️ `hmx-next-round-plan.md:240` 建议的 `ITERS≥300` **不够**（5.45%）⇒ 此处上调。
⚠️ **S3-class 要 2% 需 N ≥ 5654**——若将来在 S3-class 上做，门槛另算。

**`once_share > 2%` 时唯一允许的判决是 `PROVEN (wrong quantity)`。**

### 5.1.2 的勘误（2026-10-02，真机实测）：**`N ≥ 1000` 这个结论不成立**

**上面那张表（含 `:546` 唯一标 ✅ 的 S2-class 1.70%）按本节自己的定义被实测否掉。原文保留。**

**做法**：`Perf:` 随 `ITERS` 下降 ⇒ 用两点解 `t(N) = A + B/N`
（`B=(t1−t2)/(1/N1−1/N2)`、`A=t1−B/N1`），第三点独立校验，最大偏差 **0.42 µs**。

| 形状 | A 稳态 | B 一次性 | `once_share@1000` | `@20000` | **合规需 N ≥** |
|---|---:|---:|---:|---:|---:|
| S3-class 128×128×128 | 7.89 µs | 2105 µs | **21.05%** | 1.316% | **13 333** |
| 256×512×64 | 16.84 | 3158 | **15.79%** | 0.929% | **9 375** |
| 256×512×256 | 18.84 | 3158 | **14.35%** | 0.831% | **8 380** |
| **S2-class 256×64×2048** | 38.84 | 3158 | **7.52%** | 0.405% | **4 065** |
| S1 / 1024×512×64 | 52.89 | 2105 | **3.83%** | 0.199% | **1 990** |

⇒ ⛔ **五个形状在 N=1000 下全部超标。**
⇒ ⛔ **`:546` 的 S2-class「1.70% ✅」实测 7.52%，超标 3.8 倍。**
⇒ ✅ **N=20000 全部合规（≤1.32%）。**

**根因**：这张表给三个 class **共用同一个 `B = 1470 µs`**（三档验算吻合到 0.01%）
⇒ **那是假设，不是逐形状实测。**
而实测里 **A 缩小（优化见效）而 B 相对变大** ⇒ `B/A` 从 ~17 涨到 ~81。

⇒ ⭐ **修正后的规则：`N ≥ 50·B/A` 不是常数，每个形状类、每个构建都要重算。**
**`:549` 那句「⇒ S3 用 `N ≥ 1000`」按实测应是 **`N ≥ 4065`**（S2-class）。**

**⚠️ 但「比值型」结论不受影响**：同一形状同一 N 下比两臂时 `B` 是共模、会完全抵消
（`base − ws = A_base − A_ws`）。⇒ 今晚的 `ws` 效应、`staging` 效应、
`kStageMinKTiles=32` 钉值**都是同形状同 N 的臂间比较，依然有效**。
⇒ ⛔ **要重跑的是「绝对水平」类陈述，不是「效应量」类。**

**⚠️ 连带发现：`docs/README.md:37` 与本文件的比值表把「我们」那一列和
llama 那一列放在不同口径下比** —— llama 侧用 24576~81920 runs（`B/N` 可忽略 ⇒ 稳态），
我们用 N=1000（含 `B/N`）。**稳态口径下 S1 是 52.89/52.25 = 1.012×，
而 N=1000 口径下是 1.053×。**
⇒ ⚠️ **「S1 是唯一真差距（1.14×）」可能是口径伪像** ——
但 llama 侧仍是 2026-09 的数（本机跑不了，bench 在手机 Termux 上）⇒ **只是条件性线索，不是判决。**

> **⚠️ 追加勘误（2026-10-02 06:45）：上表里那个 `B = 2 105 µs` 是量化的产物，真实值 ≈2 700 µs。**
>
> 九个形状的 `B` 只取 **2105 / 3158 / 4211** 三个值，而它们**都是 1053 的整数倍** ——
> 1053 µs 正是「`Δ = t(1000) − t(20000) = 1 µs`」对应的量子
> （`1 / (1/1000 − 1/20000) = 1053`）⇒ **那不是三个物理值，只是 `Δ` 被量化成 2、3、4。**
>
> `B` 只出现在 `B/N` 里 ⇒ **N 越小分辨率越高**。把 S1 的 `t(N)`（N=1…20000，已扫）
> 换区间重拟合：
>
> | 拟合区间 | `B` | 最大残差 |
> |---|---:|---:|
> | N=1..30 | **2 665** | 9.6% |
> | N=1..3000 | **2 683** | 11.7% |
> | 全部 10 点 | **2 684** | 10.4% |
> | N≥100 | **2 827** | 0.9% |
> | N≥1000 | **1 992** | **0.5%** ← 残差最小，**却恰是量子档** |
>
> ⇒ ⭐ **九档里九档指向 2 650~2 830；唯一给出 1 992 的那档残差最小（0.5%），**
> **最容易被当成「更准的估计」，实际是最粗的。**
> ⇒ **⇒ `B ≈ 2 700 µs（±100）`；`once_share@1000 = 4.80%`；`N ≥ 2 519`。**
> ⇒ 上文那三行「S1-class 合规需 N ≥ 1 990」应读作 **`N ≥ 2 500 ~ 2 700`**。
> ⇒ ✅ **超标的方向与倍数不变**（S2-class 的 `1.70% ✅` 实测 7.52%）。
>
> ⭐ **顺带一条可迁移的教训：拟合残差小 ≠ 参数可信；在有量化噪声的数据上，
> 最紧的那一段往往正好落在量子上。**
>
> ⛔ **另：上表「五个形状的 A / B」那一列的 `B` 值同样受此影响** ⇒ 五个形状的
> `once_share` 与门槛都要按上面这个 `B` 重算一遍才能引用。

> **🟣 勘误七（2026-10-02 13:11–13:13）：`:465` 那句「两个 patch 都只对 `build/bin/opt` 生效」**
> **已过时 —— install 树已刷新，`libtriton.so` 已重链，设备锚点已作废并更换。**
>
> **做了什么（用户 2026-10-02 12:20 批准 S0b）：**
>
> | 步 | 命令 | 结果 |
> |---|---|---|
> | 0 | 备份 `libtriton.so` + `libhmxapi.a` 到 `logs/anchor-backup-2026-10-02/` | ✅ 两个 md5 都等于旧锚点 |
> | 1 | `ninja install`（`llvm_triton/build`） | ✅ RC=0 |
> | 2 | `ninja`（`triton/build/cmake.linux-x86_64-cpython-3.10`） | ✅ RC=0，58/58 步 |
>
> **⇒ 设备锚点：`ce26015e8efb75cc047515000c8ad70f` → `20342db00d287f29a22490762ca38066`**
> （`libhmxapi.a` **未变**，仍 `97af133e81fbc361bca3be10164b7bc8` —— 只重编了 host 侧）
> ⇒ **旧锚点可一行回滚**：`cp logs/anchor-backup-2026-10-02/libtriton.so.ce26015e
>   hexagon-mlir/triton/python/triton/_C/libtriton.so`
>
> ### ⭐ 真正卡住的不是重建，是 `clang++` 缺 `libtinfo.so.5`
>
> 第 2 步第一次跑 **61 步里 30 步 FAILED**，全部同一个错：
> `HOST_TOOLCHAIN/bin/clang++: error while loading shared libraries: libtinfo.so.5`。
> ⇒ **解法一行**：`LD_LIBRARY_PATH=<workspace>/HOST_TOOLCHAIN/libtinfo5/lib/x86_64-linux-gnu`
> （`.deb` 早就解包在那里了，只是没在库路径里）⇒ 之后 **58/58 全过、40 秒**。
> ⇒ ⭐ **⇒ 本项目第一次在本机重建 `libtriton.so` 成功。**
>
> ### ⚠️ 我自己两处报错，都已更正（append-only 记在这里）
>
> 1. **「`ninja install` 什么都没装」——错。** 我查了
>    `libLLVMAggressiveInstCombine.a`（Sep 16，**本来就不该变**）就下了结论。
>    实际 `install/lib/libLLVMInstCombine.a` **就是 2026-10-02 13:09** ⇒ **装成功了。**
>    ⇒ **「哪些库该变」这件事本身要先查清，不能只看目录 mtime。**
> 2. **「patch 在不在新 `.so` 里」——我没能验证。** 判别式两次都没触发收窄路径：
>    第一次 IR 写错（操作数不是 `fpext`），第二次形式对但 InstCombine 不收窄。
>    ⇒ 读了 patch 源码才明白：它在 **`InstCombineCasts.cpp` 的 `narrowBinOp`**，
>    **只在路径上存在 `fptrunc`（`FPT`）时才生效** ⇒ 我两个用例都够不着。
>    **⇒ 结论是「未验证」，不是「在」也不是「不在」。**
>
> ### ⛔ 因此 S0b 的准确状态是「做完了，但没验完」
>
> - ✅ install 树与 `libtriton.so` 现在**互相一致**，且都含 10-02 13:09 重编的 `libLLVMInstCombine.a`
> - ✅ 门全绿（`split_patch` 507 hunks 一致、`doctor` **17 ok / 0 fail**）
> - ✅ 新 `.so` 能加载、能编译（host `warmup` 探针通过）
> - ⛔ **`llvm-hexagon-fdiv-ninf-narrowing.patch` 的行为未在真机或 `opt` 上验证**
> - ⛔ **「f16 除法 194 次 libcall」这个项目当前最大单点，尚未重测**
>   ⇒ 那是 S0b 立项要解决的问题，**现在才第一次具备可测条件**
>
> 📄 `docs/results/s0b-rebuild-2026-10-02.md`
> 📄 新锚点基线：`logs/baseline-s0b-2026-10-02/newanchor-N1000.log`
>
> #### ⛔ 勘误七之更正（2026-10-02 14:22）：**新 `.so` 在 S1 上回归，已回滚**
>
> 勘误七写的是「锚点已换成 `20342db0`」。**那条只持续了约 1 小时，现已回滚。**
>
> | 形状 | 新 `.so`（`20342db0`） | 旧 `.so`（`ce26015e`，已恢复） |
> |---|---|---|
> | `S3_class` 128×128×128 | ✅ `perf=8.0` | ✅ `perf=8.0` |
> | `S2_class` 256×64×2048 | ✅ `perf=41.0` | ✅（基线 39） |
> | **`A_1024x512x64` = S1** | ⛔ **rc=1，零输出** | ✅ `perf=51.0` |
>
> **失败形态（这是最值得记的一条）**：
>
> - **host 侧编译通过**（`kern.warmup(...)` 返回正常）
> - **设备启动 rc=1，`stdout`/`stderr` 一个字都没有** —— 连那条
>   `tl.make_block_ptr` 弃用警告都没出现
> - ⇒ **与 N 无关**（N=10 与 N=1000 都失败）、**与形状顺序无关**
> - ⇒ **只有 S1 挂，S2/S3 正常** ⇒ 不是「`.so` 整体不可用」，是**特定形状的回归**
>
> **⛔ 根因未查**（时间不够）。⚠️ 错误被完全吞掉这一点本身值得单独查：
> `shape_pair.py` 在 launch 失败时没有把子进程输出带出来。
>
> ### ⇒ S0b 的净结果要重写
>
> | | 状态 |
> |---|---|
> | ✅ **重建能力已解锁** | `libtinfo.so.5` 的 `LD_LIBRARY_PATH` 修复 ⇒ **本机第一次能重编 `libtriton.so`**，全程 2 分钟 |
> | ✅ **重建流程已可复现** | 7 步见 `docs/results/s0b-rebuild-2026-10-02.md` §7 |
> | ✅ **回滚已验证** | 备份 md5 相符；回滚后 S1 `perf=51.0`、`rel=2.2046e-04`、指纹前后一致 |
> | ⛔ **重建产物不可用** | `20342db0` 在 S1 上回归，已移到 `logs/anchor-backup-2026-10-02/libtriton.so.20342db0.REGRESSES-S1` 作物证 |
> | ⛔ **fdiv patch 仍未验证** | 「重建成功」**不等于**「patch 生效」，这两件事我分开失败了 |
> | ⛔ **f16 除法 194 次 libcall 未重测** | 那才是 S0b 立项要解决的，现在仍未解决 |
>
> #### 🟧 勘误七之更正之五（2026-10-02 16:35）：**两个构建生成的设备指令流逐字节相同 —— 整条 LLVM/后端链路全部出局**
>
> 更正之四把机制候选换成「`a576182` 的 K-fusion」。**那个候选现在也出局了**，而且是被
> **产物**出局的，不是被推理出局的。
>
> **两个构建各自的真机对象码都在，可以直接对拍：**
>
> | 来源 | 由哪个构建产出 |
> |---|---|
> | `exp/hmx/phase0_4_s1_objectcode/*.o`（mtime 10-02 04:11–04:29） | **`ce26015e`**（好的那个） |
> | `logs/.triton-cache/1790919750-679996768/*/matmul.o` | **`20342db0`**（重建那个；缓存键 = `%Y-%s` of the `.so`，`env.sh:64-72`） |
>
> **归一化后的指令流 md5：**
>
> ```
> S1  OLD=14613f87e8f5   NEW=14613f87e8f5   -> IDENTICAL
> S2  OLD=a9e0415c97e8   NEW=a9e0415c97e8   -> IDENTICAL
> S3  OLD=0a9a9ff1c099   NEW=0a9a9ff1c099   -> IDENTICAL
> ```
>
> ⇒ ⭐⭐⭐ **⇒ 两个构建生成的设备指令流逐字节相同。**
> ⇒ **⇒ 因此：任何 codegen 变化（LLVM 指令选择、调度、内联，或后端 pass）
> 都不可能是 13:14 那次失败的原因 —— 因为它没有改变任何一条指令。**
> ⇒ **⇒ 更正之二/三/四依次排除的「脏缓存」「hmx-attr」「`a576182` K-fusion」，
> 现在全部被同一条证据一次性关掉了。**
>
> ### ⇒ 连 `a576182` 为什么无害，也一并解释了
>
> 它的标题是 "Fuse the K traversal into one HMX mma **via the croutons-per-mma option**"，
> 而**产物逐字节不变** ⇒ **那个选项默认关闭** ⇒ 它确实被链进了 `.so`（`.o` 时间戳 13:11:18 可证），
> **但对默认路径零影响**。这与本仓已有的记录一致：**新 pass 及其 lit 全部默认关 ⇒ 产物不变**。
>
> ### ⇒ 而 `hmx-attr` 为什么无害，本仓**自己的记录早就写过**
>
> `hexagon-mlir-local.patch:64792`：
> **「本仓源码里 `hexagon_hmx` 零命中 ⇒ 没有任何函数带该属性 ⇒ `IsHMX` 恒 `false`
> ⇒ 今天不重建就完全不生效。」**
> ⇒ 独立复核：`hexagon_hmx` 在 `logs/codegen-dumps/` 全部 60 个 `.mlir` 里**零命中**。
> ⇒ ⭐ **⇒ 重建激活了那条代码路径，但没给它任何可作用的对象。**
> ⇒ ⚠️ **⇒ 而这条记录本来就在仓里。我做 S0b 之前没有先读它。**
>
> ### ⇒ 于是只剩一个量在 S1 上是全矩阵里独有的
>
> | 维度 | S1 | S2 | S3 |
> |---|---:|---:|---:|
> | 输出字节 `2MN` | **1,048,576（正好 1 MiB）** | 32,768 | 32,768 |
> | `Mt` | **32** | 8 | 4 |
> | FLOP | 67,108,864 | 67,108,864（**与 S1 并列**） | 4,194,304 |
> | VTCM 峰值 / 8 MiB 预算 | 14.8% | **16.0%（更高却正常）** | 1.2% |
> | `pipeline.reason` | `shallow-k` | `None` | **`shallow-k`（与 S1 相同）** |
>
> - ⛔ **「S1 超过某个 VTCM 预算」被实测否掉** —— S2 的占比更高且正常。
> - ⛔ **`shallow-k` 不是判别式** —— S3 同样是 `shallow-k` 且正常。
> - ⛔ **FLOP 不区分 S1 与 S2** —— 两者**完全相等**。
> - ⚠️ **唯一真正在阈值上的量**：`MemoryOffsetsPass` 的 `bufferSize` 默认
>   **1048576**（`Passes.td:380-382`），检查是 `totalRequiredSize > bufferSize`
>   ⇒ `signalPassFailure()`（`MemoryOffsetsPass.cpp:250-257`）。
>   **S1 的输出正好是 1,048,576** ⇒ **恰好等于阈值，不大于它** ⇒ 通过。
>   ⚠️ **但这是 host 编译期检查，而 host 编译现在成功 ⇒ 不能解释设备端的 rc=1。**
>
> ### ⇒ 结论（这一条我认为是最终的了）
>
> **13:14 那次 S1 失败不是任何一个构建的性质。** 两个构建生成同一条指令流、
> 都在 9/9 上通过（`rel` 全 2.2e-04）、S1 单独跑也通过。
> ⇒ **它是一次环境/瞬态事件**，而**唯一让我为它做过重大决定的，是那条空日志。**
>
> ⇒ ⭐⭐⭐ **⇒ 方法上最终的一条：功能测试与产物对拍回答的是两个不同问题，
> 而我先做了功能测试就下了结论。**
> 「9/9 通过」说明不了「代码有没有变」；**「指令流 md5 相同」才说明得了。**
> ⇒ 而**早一步做产物对拍，就能省掉回滚、以及随后三条越来越长的勘误更正。**
>
> ⇒ 📄 `docs/results/s1-regression-not-reproducible-2026-10-02.md` §10
>

> #### 🟦 勘误七之更正之四（2026-10-02 16:25）：**真正的差异不是 LLVM 补丁，是 `a576182`（K-fusion）——而它一直躺在源码里没被编译**
>
> 更正之二与之三都把差异归到 LLVM 侧。**那是不完整的：`libtriton.so` 里还链进了后端自己的改动，
> 而那些改动比 LLVM 补丁大得多。**
>
> ### 三条硬证据（本机复核，非转述）
>
> | | |
> |---|---|
> | 工作 `.so`（`ce26015e`）构建 | **2026-10-01 14:18** |
> | `a576182` 落地 | **2026-10-01 22:54** —— **晚 8.5 小时** |
> | `HmxPartitionPass.cpp.o`（`obj.HmxTransforms.dir`）重建 | **2026-10-02 13:11:18**（我那次 `ninja`） |
>
> ⇒ ⭐⭐⭐ **⇒ `ce26015e` 从来不含 `a576182`。今天的重建是它第一次被编进 `.so`。**
>
> `a576182` = **"Fuse the K traversal into one HMX mma via the croutons-per-mma option"**，
> 改动面：`HmxPartitionPass.cpp` **388 行** · `LinalgToLLVMPass.cpp` · `MLLVMIRTranslation.cpp` ·
> `hexagon_options.py` 74 行 · 新增两个 lit（`mma-deep-croutons.mlir` 与
> **`mma-deep-croutons-reject.mlir`**）。
>
> ⚠️ **⚠️ ⇒ 这才是「只有 S1 挂」最合理的机制候选：**
> K-fusion **按 `kt` 与 croutons 批大小分支**，且**改 packet 数与 VTCM 占用** ——
> 而 K = 64（S1）/ 128（S3）/ 2048（S2）走的是不同路径。
> ⇒ **⚠️ 那个 `*-reject.mlir` 的存在说明作者自己预期过边界情况。**
>
> ### 四个 LLVM 补丁的处境（一条被证否、三条出局）
>
> | 补丁 | 判决 | 依据 |
> |---|---|---|
> | **`hmx-attr`** | ⛔ **可证是 no-op** | `hexagon_hmx` 在**后端源码里零命中** ⇒ `IsHMX` 恒 false；且 `BaseT::areInlineCompatible` 本来就落到 `BasicTTIImpl.h:396` 的等价检查 |
> | **`fdiv-ninf`** | ⛔ 对 f16 matmul **不可达** | 只清 `ninf` 旗标（IR 更保守），且只在「`fptrunc` 喂 binop」的路径上，f16 matmul 不走 |
> | `ps-aligna` / `fmaxnum-nnan` | ⛔ **出局** | 全部 obj 是 09-23，**已含在工作 `.so` 里**，不在 delta 内 |
>
> ⚠️ **顺带一条仍然重要的提醒**：`ps-aligna` 是四个里**唯一有硬失败模式**的
> （它修的是被破坏的 callee-saved AP，`needsAligna` 对变长对象触发，**确实与代码大小相关**）。
> **⇒ 若哪天它进了 delta，它要排第一。**
>
> ### ⛔ 一处两个独立调查互相矛盾，**必须记作未决**
>
> | | 「fdiv patch 在不在 `20342db0` 里」 |
> |---|---|
> | 二进制取证（**实测**：`.text` 仅 +192 B、438,124 个符号名只差 1 个） | ⛔ **不在** |
> | patch 审计（**从 ninja 日志重建**） | ✅ 在，但 inert |
>
> ⇒ **⇒ 我不替它们裁决。** 实测优先，但 `.text` 总量约束很强（+192 B 装不下
> `InstCombineCasts.cpp` 的改动）⇒ **最可能是「日志口径与链接口径不同」**，
> 例如 cmake 那棵树链的是自己那份 LLVM 归档。**这一条要查清才能算 S0b 的账。**
>
> ### ⇒ 于是 S0b 的账是这样记的
>
> | | |
> |---|---|
> | ✅ **重建能力解锁** | 2 分钟，已验证 |
> | ⚠️ **重建悄悄带进了一个从未编译过的后端改动** | `a576182` K-fusion，8.5 小时的源码-二进制漂移 |
> | ⛔ **f16 除法那个单点仍然没动** | 取决于上面那条矛盾怎么解 |
> | ⛔ **13:14 那次 S1 失败仍未定** | 已被证明**不是** `.so` 的性质（同一 `.so` 现在 9/9），但机制候选从「LLVM 补丁」换成了「`a576182` 的 K-fusion」 |
>
> ### ⇒ ⭐ 下一件该做的实验（比之前任何提案都便宜）
>
> **`20342db0` 上跑 9 形状 × N=1000。**
>
> 理由：9/9 的通过是在 **N=10** 测的，而**原始失败第一次出现在 N=1000**；
> 且 `a576182` 改的是 **VTCM 占用** ⇒ **N 相关的内存行为是剩下的主要未排除变量**。
> ⇒ 这一轮同时也是「换回 `20342db0` 之前必须做的那份同口径基线」。
>
> ⇒ 📄 `docs/results/s1-regression-not-reproducible-2026-10-02.md` §9
> ⇒ 📄 物证：`logs/anchor-backup-2026-10-02/libtriton.so.20342db0.9of9-OK`
>

> #### 🟥 勘误七之更正之三（2026-10-02 16:10）：**更正之二里「产物可用、回滚不必要」这句又要收窄**
>
> 更正之二说「不是代码回归、两个 `.so` 读数逐位相同」。
> **「读数相同」是对的**（那是 15:39–15:42 三次 S1 实测）；
> **但由此推出「产物可用」是错的** —— 我拿**一个形状**的读数去否定一个**内联行为**的改变。
>
> **二进制取证（`nm` / `objdump` 逐符号，两个 `.so` 都在 `logs/anchor-backup-2026-10-02/`）**
> 证明两个 `.so` **确实不同，且差异恰好只有一处**：
>
> | 证据 | `ce26015e` | `20342db0` |
> |---|---|---|
> | 字符串字面量 `hexagon_hmx` 出现次数 | **0** | **1** |
> | `llvm::HexagonTTIImpl::areInlineCompatible` | **不存在** | **存在**（234 B） |
> | `BasicTTIImplBase<HexagonTTIImpl>::areInlineCompatible` | 存在（166 B） | **不存在** |
> | `useHVX` 指令 | 27 B | 32 B，末尾 `cmpb $0x0,0x20(%rdi); sete %al` ← **就是 patch 的 `&& !IsHMX`** |
> | `.text` 增量 | — | **仅 +192 B** |
>
> ⇒ ⭐⭐⭐ **⇒ 差异 = `llvm-hexagon-hmx-attr.patch`，且它做的是
> `areInlineCompatible`：当两个函数的 `hexagon_hmx` 属性不同时返回 false。**
> ⇒ **这是内联决策的改变，而内联决策是形状相关的。**
> ⚠️ **⇒ 所以「S1 读数相同」只证明 S1 这个形状的最终代码没变，
> 不证明其它形状不变，更不证明原来 13:14 的失败不是它造成的。**
> ⇒ **必须以 9 形状复验为准**，不能以单形状读数下结论。
>
> ### ⚠️ 同时查实一件对 S0b 更要紧的事：**fdiv patch 仍然没进 `.so`**
>
> `.text` 只涨 192 B，且 438,124 个符号名里**只多/少了那 1 个**
> ⇒ **`llvm-hexagon-fdiv-ninf-narrowing.patch`（在 `libLLVMInstCombine.a` @ 10-02 13:09）
> 并没有被链进 `libtriton.so`。**
> ⇒ ⭐ **⇒ 我 13:0x 写的「install 树已刷新 ⇒ patch 已进 install 树的库」是对的，
> 但「⇒ 所以进了 `.so`」是错的 —— 库换了，`.so` 并没有真的重链进去那一部分。**
> ⇒ **⇒ 「f16 除法 194 次 libcall」这个项目当前最大单点，仍然完全未测。**
>
> ### ⇒ 三条结论按证据强度重排
>
> | 结论 | 强度 |
> |---|---|
> | `.so` 与旧 `.so` **不同**，差异 = hmx-attr patch 的 `areInlineCompatible` | ✅ **字节级证明** |
> | 该 patch **不在**旧 `.so` 里（`hexagon_hmx` 字面量 0 次） | ✅ **字节级证明** |
> | fdiv patch **不在**新 `.so` 里 | ✅ `.text` 仅 +192 B + 符号集只差 1 个 |
> | Hexagon CodeGen 本身**没变** | ✅ `createPassConfig` 逐指令相同，只有 6 B 位移因符号移动而变 |
> | hmx-attr patch **是否造成** 13:14 那次 S1 失败 | ⛔ **未定**（需 9 形状复验 + 一个不含该 patch 的链接） |
> | 13:14 那次失败本身的原因 | ⛔ **仍未定**（更正之二 §4 的结论不变） |
>
> ⇒ 📄 `docs/results/s1-regression-not-reproducible-2026-10-02.md`（含取证细节）
> ⇒ ⭐ **⇒ 下一件事不是「换回 `.so`」，是「把 9 形状复验跑完」；
> 换回之前必须先回答「hmx-attr patch 改变了哪些形状的代码」。**
>

> #### 🟩 勘误七之更正之二（2026-10-02 15:40–16:0x）：**「S1 回归」不可复现 —— 那不是代码回归，回滚是不必要的**
>
> 上面那条更正（14:22）说 `20342db0` 在 S1 上回归、产物不可用。**那个结论是错的。**
>
> **受控 A/B（两个 `.so` 都在 `logs/anchor-backup-2026-10-02/`，可直接对拍）**：
>
> | 跑 | `.so` | 缓存 | S1 结果 |
> |---|---|---|---|
> | 15:39 | `ce26015e`（好） | 各自命名空间 | ✅ `perf=55.0` `rel=2.2046e-04` |
> | 15:40 | **`20342db0`（曾判「坏」）** | 各自命名空间 | ✅ **`perf=55.0` `rel=2.2046e-04`** |
> | 15:42 | **`20342db0`** | **全新空目录 `/tmp/opencode/coldcache-badso`** | ✅ **`perf=55.0`** |
>
> ⇒ ⭐⭐⭐ **⇒ 三个假设被逐一排除：**
>
> 1. **不是代码回归** —— 同一形状、同一设备、同一脚本，两个 `.so` 读数**逐位相同**。
> 2. **不是脏 Triton 缓存** —— `env.sh:60` 的缓存键**包含后端库自身身份**，
>    命名规则实测为 `<epoch>-<libtriton.so 字节数>`
>    （`1790917894-679996768` = 坏那个；`1790918392-679997440`、`1790919630-679997440` = 好的）
>    ⇒ **两个 `.so` 各有独立命名空间**；且**连全新空缓存目录也照样成功**。
> 3. **不是「第一次跑必然失败」** —— 13:14 / 13:17 / 13:19 三次全败、13:40 起全成，
>    两次之间**唯一变化的是时间与缓存内容，不是任何输入**。
>
> ### ⇒ 那 13:14 的失败到底是什么？——**查不出来；现有证据不足以定论**
>
> 时间线（用缓存目录名的 epoch 前缀钉死，非推测）：
> `13:11:34` 坏 `.so` 命名空间建立 → `13:14:34` 失败跑落盘（指纹 `20342db0`）
> → `13:17:11` 该命名空间被写 → `13:19:52` 回滚。
>
> ⚠️ **可排除**：输入（`torch.manual_seed(0)`，`shape_pair.py:195`）、
> 设备（`S2`/`S3` 同时段正常）、缓存（上面第 2 条）、代码（第 1 条）。
> ⚠️ **不可排除**：13:11–13:19 之间**是否有别的东西在写那个 `.so` 或占用设备**
> —— 那段时间我在跑 S0b 的门与 `ninja` 收尾，**我没有当时的完整进程/设备日志**。
> ⇒ ⭐ **⇒ 按项目自己的纪律（`AGENTS.md`：可复算证据），这一条应记作
> 「原因未确定」，而不是「已定位为 fdiv patch / hmx-attr patch / 冷缓存」。**
>
> ### ⇒ 由此产生的三个更正
>
> **① S0b 其实成功了。** `20342db0` 可用 ⇒ **回滚是不必要的**，
> `libtriton.so` 可以换回 `20342db0`（⚠️ 换回前先跑一遍 9 形状复验）。
>
> **② 「重建成功」与「patch 生效」仍要分开验。** 复验只证明**产物能跑**，
> **不证明 fdiv patch 的行为** —— 判别式至今仍未跑通
> （它在 `InstCombineCasts.cpp` 的 `narrowBinOp`，只在路径上有 `fptrunc` 时才走到）。
> ⇒ **但它现在确实在 install 树的库里（`libLLVMInstCombine.a` @ 10-02 13:09），
> 所以「patch 有没有进 `.so`」已从「不知道」变成「几乎必然进了」，
> 只差一个能触发 `narrowBinOp` 的用例。**
>
> **③ 真正值得记住的教训比原来那条更强：**
> ⭐⭐⭐ **我因为一次「不可复现的失败」做了回滚，并把「产物不可用」写进了勘误。**
> 而触发回滚的那次失败**日志里一个字都没有** —— 根因是
> `shape_pair.py:205` 把 launch 的 stdout 吞进 buffer、`:241` 的 assert 直接死掉、
> **buffer 从未打印**。
> ⇒ ⭐ **静默失败不只是「查起来麻烦」，它会让人做出错误的重大决定。**
> ⇒ ✅ 已修：同时捕获 stdout+stderr，失败时打印 `LAUNCHFAIL` / `NOPERF` 与两个流的内容。
> ⇒ 📄 `docs/results/s1-regression-not-reproducible-2026-10-02.md`
>

> ⇒ ⭐⭐⭐ **⇒ 下一个人该做的第一件事不是「再重建一次」，**
> **而是查清两件独立的事：**
> **① `20342db0` 为什么在 S1 上回归（先让失败说话，见下）；**
> **② `llvm-hexagon-fdiv-ninf-narrowing.patch` 的行为到底进没进那个 `.so`。**
> **在 ① 之前不要把新 `.so` 当成可用的基线。**
>
> ### ⚠️ 顺带一条：`shape_pair.py` 的失败不可见，是今晚第 N 次同一形状的坑
>
> 今晚已经踩了两次「静默失败」：① 插桩里 `\n` 只写一个反斜杠 ⇒ 编译失败**无报错**；
> ② 探针在 `/tmp` 被清 ⇒ 结论无法复现。
> **这次是第三次：设备 launch 失败，日志里什么都没有。**
> ⇒ 已在 `exp/hmx/shape_attribution/dump_generated.py` 修过一次（让静默失败出声），
> **但 `shape_pair.py` 这条路径还没修** ⇒ 下一个碰到设备侧失败的人会再栽一次。
>
> **🔵 勘误六（2026-10-02 12:35）：本文件第一条方案（T_HMX 专属线程 + HVX 池）
> 的收益估算，缺一块它现在才拿到的输入。**
>
> 勘误五找到：**staging 成本是被 VTCM 预算卡的离散决策**
> （`HmxPartitionPass.cpp:1651-1653`），且 `scratchBytes ∝ Kt`、`room` 里的 `actBytes ∝ Mt`。
>
> ⇒ ⭐⭐⭐ **⇒ 这正是本方案要动的那一个量。**
> 今天 `T_HMX` 与 HVX 池线程共享同一份 VTCM 预算与同一个 HMX 资源；
> 拆成专属线程后 **HMX 侧不再与 HVX 竞争** ⇒ `budgetDepth` 可能从 1 升到 2
> ⇒ 而 **`S2`（`Kt=64`）的 39 µs 里有 19.7 µs 落在那一项上**
> ⇒ **「拆线程能救回多少」这个卖点，现在第一次有了可算的形式。**
>
> ⚠️ **但要定量，缺一块：哪些形状现在落在 `budgetDepth = 1`。**
>
> - ⛔ **本构建的 envelope 是 v1** ⇒ `metadata["hmx_manifest"]` 实跑 15 个形状全是 `None`
>   （`utils.py:1881` 明写 v1 无 manifest；决策本该写进 `hmx.kernel_manifest`，
>   见 `HmxPartitionPass.cpp:296-316`）
> - ⛔ **手工复现不做**：要经 `vtcmBytesCommitted`（`MatmulToHmxPass.cpp:993-1032`）
>   与 `HmxTarget::planBridge` 的 M-blocking 及权重驻留路径
>   ⇒ **一个复现错的机制解释比没有解释更糟**
>
> ⇒ **⇒ 因此本方案现在能做的是设计评审，**不能**有可信的收益估计。**
> 本文件 `:69` 那句「输的恰是它的跨线程流水这一步是推断，不是实测」**至今仍然成立**。
>
> **解锁顺序**（两条都在 `:6` 的 S0b 那个窗口之后）：
>
> ```
> S0b 重建后端（会改 libtriton.so，作废设备锚点）
>   └─ 建一个 v2/v3 envelope 的版本 ⇒ 读出逐形状 budgetDepth
>        └─ 才能给 T_HMX 分线程定量：拆线程救回多少 µs
> ```
>
> ⭐ **顺带一条与本方案无关但同源的**：`Kt ≥ kStageMinKTiles` 那个门（`603163c`）
> **已被留出数据独立确认**（`Kt=16` 那点判别力 4.5 倍，勘误四 §3），
> ⇒ **S2/S3 走 staged 路径这件事不再是「按推理定的」，而是有真机证据的。**
>
> 📄 `docs/results/output-term-is-rows-not-bytes-2026-10-02.md` §4.2–§4.3
> 📄 `exp/hmx/shape_attribution/dump_pipeline_decision.py`（读 `metadata["hmx_manifest"]` 的尝试；
> **保留，因为它记录了「v1 envelope 读不到」这个事实本身**）

> **🟢 勘误五（2026-10-02 09:51）：勘误四里那个 `g = 0.30 µs/Kt-tile` 系数
> **不是 Kt 的函数**，机制是一个被 VTCM 预算卡的**离散决策**。**
>
> **触发点**：`H7_kt128` = 128×128×4096（`Kt=128`、张量 2^19）实测 **44.0**，
> 而带 `Kt²` 的模型预测 **63.5** ⇒ **高出 19.5 µs，把 `Kt²` 直接否掉。**
>
> **但它指出了正确的结构**：把「staged 成本」=（实测 − 不含任何 `Kt` 项的模型）单列，
> 同为 `Kt=128` 的两点：
>
> | 形状 | `Mt` | `(M/32)·Kt` | staged 成本 |
> |---|---:|---:|---:|
> | `H5_bigM256` | 8 | 1024 | **39.09** |
> | `H7_kt128` | **4** | 512 | **14.56** |
>
> ⇒ **按 `Kt` 算差 2.7 倍；按 `(M/32)·Kt` 算只差 1.3 倍**
> ⇒ ⭐ **⇒ 它是 `(M-tile 数) × (K-tile 数)` 的函数，不是 `Kt` 的函数。**
>
> **代码给出了为什么会跳**（`HmxPartitionPass.cpp`）：
>
> ```cpp
> int64_t scratchBytes = Kt * layout::kCroutonBytes;                        // :1619  ∝ Kt
> int64_t room         = vtcmBudget - vtcmBytesCommitted(func) + actBytes; // :1644  actBytes ∝ Mt
> auto fits = [&](int64_t d) { return scratchBytes + d * ringBytes <= room; };   // :1651
> int64_t budgetDepth = fits(2) ? 2 : (fits(1) ? 1 : 0);                   // :1652
> ```
>
> ⇒ ⭐⭐⭐ **⇒ 流水深度被 VTCM 预算卡住，按 2 / 1 / 0 **跳变** ⇒ 关系里有间断
> ⇒ ⇒ **任何多项式项都补不上，这不是拟合能力不足。**
>
> ⚠️ **⇒ 勘误四那句「`g = 0.30 µs/Kt-tile` 是唯一被留出验证过的 staging 独有成本」
> 要改读法：它的**形式**（staging 独有、与 `K` 深度相关）被验证了，
> 但**它的量纲不是 `Kt`** ⇒ 那个系数是拟合产物，不是测量值。
>
> ⛔ **逐形状的深度未证实**：这个构建的 envelope 是 **v1**
> ⇒ `metadata["hmx_manifest"]` 实跑 15 个形状全是 `None`（`utils.py:1881`：v1 无 manifest）；
> 手工复现要经 `vtcmBytesCommitted`（`MatmulToHmxPass.cpp:993-1020`）与 `planBridge` 的
> M-blocking 与权重驻留路径 ⇒ **不做无法验证的复现。**
>
> **⚠️ 另一条实验硬约束（值得写进任何后续实验设计）**：
> **所有 matmul 维度必须是 2 的幂**（`tl.make_block_ptr` 要求，否则
> `arange's range must be a power of 2`）⇒ **张量大小也只能是 2 的幂**
> ⇒ **`2^19` 与 `2^20` 之间没有任何可用尺寸 ⇒ 模型在张量到 `2^20` 处的边界无法二分。**
>
> 📄 `docs/results/output-term-is-rows-not-bytes-2026-10-02.md` §4.1–§4.3（285 行）
> 📄 `exp/hmx/shape_attribution/README.md`（2 的幂约束 + 可直接跑的形状检查清单）

> **🟢 勘误四（2026-10-02 09:25）：「S1 受限于输出读出」方向对，但**量的形式错了** ——
> 那个量是**输出行数 `M`**，不是输出字节数。**
>
> **判决来自留出验证，不是拟合优度**：12 个干净 `A`（`B` 已按勘误三消除），
> 只用 9 个拟合，在 3 个从未参与拟合的形状上测。
>
> | 第二项 | 拟合集最大绝对残差 | **留出集最大绝对残差** |
> |---|---:|---:|
> | **输出字节** `2·M·N` | 1.6 µs | **21.0 µs** |
> | ⭐ **输出行数 `M`** | 1.6 µs | **9.8 µs** |
>
> ⇒ **拟合集分不出两者；留出集差 2.1 倍。**
>
> **决定性的一对**：`256×128×1024` 与 `128×256×1024` 的 **M·N、输出字节、FLOP、Kt
> 四个量全同**，只把 M 与 N 互换 ⇒ **28.0 vs 21.0 µs（差 33%）** ⇒ 字节数解释不了。
>
> **模型**（拟合集 ≤1.6 µs，留出集 ≤9.8 µs）：
>
> ```
> t ≈ F + s·M + f·FLOP + g·Kt·[Kt ≥ 32]
>   F = 3.11 µs   s = 0.0431 µs/行   f = 0.0740 µs/MFLOP   g = 0.300 µs/Kt-tile
> ```
>
> ⚠️ **`f` 折合 13.5 TFLOP/s > HMX 硬件量级（~4）⇒ `f` 不是吞吐，不要这样引用。**
> ⚠️ **一个留出点（`H3` = 512×128×2048，最大张量正好在 Triton 上限）仍差 −13% ⇒ 模型不完整。**
>
> ### ✅ 一条被**确认**而非推翻的 pin
>
> **`H1_gate16` = 256×256×512 的 `Kt = 16`，正好落在 `Kt=8`（非 staged）与 `Kt=32`（staged）
> 之间的空档** ⇒ 唯一能说清门在哪的形状。实测 16.0 µs：
> 「无 `Kt` 项」预测 14.7（差 **8.2%**）vs「有 `Kt` 项」预测 21.9（差 37.1%）
> ⇒ ⭐ **`kStageMinKTiles = 32`（`603163c`）被留出数据独立确认，判别力 4.5 倍。**
>
> ### ⭐ 最有价值的新信息：`g = 0.30 µs/Kt-tile`
>
> 它是**目前唯一被留出验证过的、staging 独有的成本**：
>
> - **`S2`（`Kt=64`）的 39 µs 里有 19.2 µs（49%）落在这一项上**
> - `S3_class`（`Kt=4`）几乎不落在这一项上 ⇒ **S2/S3 那 5 倍差主要就是这一项**
>
> ⇒ **⇒ 该被攻击的是 staging 的 per-tile 成本，不是输出字节，也不是 FLOP。**
> ⇒ **⇒ 这改变了 S 系列的优先级排序依据**，但**不改变**「S1 是唯一真差距」那个结论
> ——后者是比值型，而勘误三已证明比值型完全不受影响。
>
> 📄 `docs/results/output-term-is-rows-not-bytes-2026-10-02.md`（158 行，13 项断言 + 系数独立重算）
> 📄 `docs/state/CURRENT.md` §3.14

> **🔴🔴 勘误三（2026-10-02 08:42）：上面两份勘误的「超标」结论本身被推翻了 ——
> 超标的是测量 harness，不是后端。**
>
> **`t(N) = A + B/N` 里的 `B ≈ 2,700 µs` 不是后端的性质，而是 wrapper 的缺陷：
> `hexagon_launcher_base.py` 每次 launch 的计时循环之前没有预热调用。**
>
> **直接证据**（逐次 pcycle + qtimer 插桩，S1 = 1024×512×64，`N ∈ {10,30,60}`）：
>
> | | iteration 0 | 稳态 | 比值 |
> |---|---:|---:|---:|
> | pcycles | 2,539,310 / 2,501,152 / 2,662,809 | ~112,000 | **22~24×** |
> | qtimer | 2,704 / 2,711 / 2,851 µs | ~53 µs | **50~54×** |
> | 隐含频率 | **0.92~0.94 GHz** | **2.08~2.14 GHz** | 0.44× |
>
> ⇒ **`B` 在 `N` 上恒定**（真的一次性），且**频率是双峰的、没有爬坡**。
>
> **机制**：`HexagonAPI::AcquireResources()`（`HexagonAPI.h:52` → `:72`）在
> **每次 launch** 里调 `initialize_and_acquire_hmx()`（`HexagonAPI.cpp:220`），
> 它做 `HAP_power_set(HMX_v2, set_clock=TRUE, target_corner=VCORNER_MAX,
> perf_mode=CLK_PERF_HIGH)`（`:231/:234/:235/:238`）与
> **`HAP_compute_res_acquire(..., 100000)`（`:261`，阻塞最长 100 ms）**。
> ⇒ **HMX 上电与拉频每次 launch 重做一遍，kernel 的第一次调用正好落在这次 bring-up 里。**
>
> **两个旁证**：
> - `WARMAB_WARMUP=1` **消不掉** `B`（2,880 vs 2,624 µs）——它是**另一次独立 launch**。
> - `B` **不随 kernel 工作量走**：四个形状 FLOP 跨 8×、输出跨 32×，
>   而 `Δ`pcycles 只跨 3×；且**第一次调用越久、隐含频率越高**（0.87 → 1.41 GHz）
>   ⇒ 这是时钟在第一次调用*期间*仍在爬，不是额外计算。
>
> **修法（一行）**：在 `benchmark_time_and_pcycles` **之前**加一次被丢弃的
> `{function_call}`。实测同一形状：
>
> | | 修之前 | **修之后** |
> |---|---:|---:|
> | `t(N=3)` | 928 µs | **57 µs** |
> | `B` | 2,624 ~ 2,880 µs | **15 µs** |
> | `A` | 53.4 µs | **52.0 µs** |
> | **`once_share@1000`** | 4.7 ~ 5.2% | **0.029%** |
>
> ⇒ ⭐ **`once_share` 降到门槛的 1/69；`N=3` 就已合规。**
> ⇒ ⇒ **本节 `N ≥ 1000` 的门槛在预热后不再是约束**（门槛本身是否还该保留，另议）。
>
> ⚠️ **但「`A` 几乎不动（53.4 → 52.0）」这句话同时意味着：
> 前面所有 `N=1000` 的**稳态**读数本来就是对的，错的只是那个一次项。
> ⇒ 比值型结论不受影响；受影响的只是「绝对水平」类陈述。**
>
> ⚠️ **未签认**：该改动在 `qcom_hexagon_backend/backend/hexagon_launcher_base.py`，
> **尚未提交**。它是 Python codegen 模板（生成的 C++ 每次 launch 在设备上现编），
> **不进 `libtriton.so` ⇒ 不作废设备锚点**（加改动后的 5 次设备跑指纹 5/5 全一致）。
> ⇒ **但它改变 `Perf` 的语义 ⇒ 仓库与本文件里所有历史 `Perf` 数字都是「含冷启动」口径。
> 是否采纳、以及是否重跑历史基线，需用户签认。**
>
> 🔴 **⇒ 因此：勘误一与勘误二的「五形状在 `N=1000` 全部超标 3.8~7.5×
> 「`S2-class` 的 `1.70% ✅` 实测 7.52%」这些「超标」判定，
> 测的是 harness 缺陷，不是后端。原文按 append-only 保留，但不再成立。**
>
> 📄 `docs/results/b-is-per-launch-hmx-warmup-2026-10-02.md`（204 行）
> 📄 `docs/analysis/b-is-nearly-constant-2026-10-02.md`（那份「近似常数」的结论方向对、归因错）
> 📄 `docs/analysis/b-not-idle-dependent-2026-10-02.md`（那条排除仍然成立，且现在有了解释）

📄 `docs/results/once-share-compliance-2026-10-02.md`（引用逐条复验）
· `docs/results/s1-readout-bound-2026-10-02.md`
· 日志 `logs/shape-attribution-2026-10-02/`（5 份，指纹 md5 出现 10 次全一致）

---

### 5.1.3 判决词表：**三类**，不是两类

| 判决 | 含义 |
|---|---|
| `PROVEN` | 效应 ≥ 阈值，且 `once_share ≤ 2%` |
| `NOT-PROVEN (effect below floor)` | 效应本身 < 15%（**机制没用**） |
| `NOT-PROVEN (noise floor above effect)` | 效应 > 15% 但阈值被 3×CV 顶高（**尺子不够细**） |
| `PROVEN (wrong quantity)` | 测出的差异落在 `B/N` 上，不是 `A` 上（**量纲错了**） |

**第三类已经真实发生过**（`STATE-OF-PLAY.md §5.6` 撤回的两条结论就是这个）。

⚠️ **前两类不是划分**——效应 27% > 15% 且阈值 54% 时，两条描述**同时成立**。

### 5.1.4 ⚠️ 测量手段：先排斜坡，再谈样本量

RoPE 那 9 个样本是**单调斜坡不是噪声**（`32.57 / 27.24 / 23.29`，`retries:0`）
⇒ `3×CV` 惩罚的是斜坡，不是噪声。
**先做这三件（零机制改动），再考虑多采 N**：
1. **A/B 交错**（不是「跑完 A 再跑 B」）
2. **丢 1 趟暖机**
3. **取中位数**（不是取均值）

### 5.1.5 ⛔ 两个不许做的算术

**① 47.2% 与 37% 分母不同、构建不同，不许相减。**
47.2% 来自 `5cea8231`/125 µs；37% 来自 `docs/hmx/hmx-hvx-co-scheduling.md:104-117`
（真实核比各部分上界之和低 **≥**37%，下界，且混着 LWP 自身 1.80× 扰动）。
⇒ **净增量必须是实测 treat/base 比值；37% 只进 interpretation，不进算术。**

**② ⚠️ S2-class 已经有一笔 1.92× staging 重叠入账了。**
`docs/hmx/hmx-perf-findings-2026-09-27.md:183`：**S2 256×64×2048（Kt=64）= 1.92×（重叠赢）**；
`:293`「我们在 S2 上的 1.92× 来自 **staging 重叠（Kt≥32）**」。
**这正是 S3 的首发形状。** 而 47.2% 来自 staging 之前的构建
⇒ **真实可藏上限 < 47.2%，S3 不得以 47.2% 为预期。**

### 5.1.6 ⛔ S4b 的组合门在数学上不可证伪拓扑

拓扑单独贡献 `min(m, s) ≤ 0.6%` ⇒ **softmax 侧 + 拓扑侧的合并门恒等于 softmax 单独过门**，
拓扑既白嫖又无法被证伪。
⇒ **唯一信号是交互项** `A_both − max(A_topo, A_softmax)`，**它被合并门吃掉了**。
⇒ S4b 报告时**必须单独列交互项**，不许只报合并值。

### 5.1.7 ⛔ 环吃 VTCM 预算 ⇒ 这是 **reject 判据**，不是性能判据

线程创建 / ensure 属一次性（`§4.2`，进程单例）⇒ 进 `B`，不是 `A`。
**真正的边际风险不是环的建立成本，是环吃 VTCM 预算压低 `budgetDepth`**：
`HmxPartitionPass.cpp:1626-1631` 的 `fits(d) = scratchBytes + d*ringBytes <= room`
—— 加第二个环会让 `fits(2)` 可能变 `fits(1)`。
⇒ **新增判据：若 role-split 使 `budgetDepth` 下降，本 kernel 直接 reject**
（落 `role-split-nobudget`），**不进入 A/B**。**这是正确性/可行性判据，不是性能判据。**

---

## 5.2 ⛔ 设备窗口：S3 / S4a / S4b 的硬前置

**原稿全文搜「run_tests.sh lock / 用户授权 / 设备窗口」= 0 命中。补上：**

1. ⛔ **要用户签认**（设备窗口）。**S3 / S4a / S4b 三条都要。**
2. ⛔ 必须走
   `bash tools/run_tests.sh lock bash -c 'source tools/hexmlir/env.sh && exec <cmd>'`
3. ⛔ **禁止重建**——会作废设备锚点。
4. ⚠️ **每样本前后各盖一次指纹**（`libtriton.so` + `libhmxapi.a` 的 md5）。
5. ⚠️ **`kernel.warmup(...)` = 只编译；`kernel[grid](...)` = 真启动。**
   曾把编译写成启动，无意触碰设备（已披露两起，见 `docs/state/CURRENT.md`）。

**锚点（2026-10-02 复核，未变）**：
`libtriton.so ce26015e8efb75cc047515000c8ad70f`（2026-10-01 14:18）
· `libhmxapi.a 97af133e81fbc361bca3be10164b7bc8`（2026-10-01 12:46）

---

## 5.3 ⚠️ 「`pack_weight ≈ 0`」的适用边界（此前未划界）

**可藏量 47.2% 压在这个假设上**：`enableWeightResident` 默认开
（`backend/hexagon_options.py:105` `enableWeightResident: bool = True`）⇒ `pack_weight ≈ 0`。

⚠️ **但「默认开」不等于「对每个 kernel 都成立」**，本方案从未划界：

| 情况 | `pack_weight` | 证据 |
|---|---|---|
| 权重是入口块参数 + host 预排 | **≈ 0** | `AGENTS.md:131` 第 ⑤ 条（matmul 生效：S1 −8.5% / S2 −19.8% / S3 −25.0%） |
| **FA 的 K site** | **每趟 2 次** | 源是 `memref.alloc`（`tl.trans` 物化转置后的临时缓冲），两条源视图匹配器都返回空（`WeightResidentPass.cpp:1803-1834`） |
| **FA 的 V site** | **每趟 2 次** | 源确是入口块参数的视图，但是 `[1024,64]` **行块（K-block）**，匹配器只认「更宽权重的 N 列块」（`:954` 的 `n<=bn`） |

⇒ **本方案的 47.2% 只对「权重走 `argument-slot` + host 预排」的 kernel 成立。**
S3 首发形状（S2-class matmul）在这一类里 ✅；
**FA 不在这一类里**（每趟 4 次 `pack_weight`，见板上 `bf04bf2b`）。

**⇒ S4a / S4b（FA）不得直接套用 47.2% 这个口径。**

---

## 5.4 ⛔ 独立审核推翻了我的 trait 设计（2026-10-02）——**动代码前先读这一节**

派了独立 agent 攻「加 `HmxLayoutHvx` trait」这个判断，**它推翻了我三条**。我逐条复算，全部成立。

### ① 「零行为变化」是错的：我只扫了 fixture，没扫 lit

`test/Conversion/HmxToLLVM/` 下**有 32 个函数只含布局 op、不含引擎 op**
（`hmx-tail-leaves.mlir` 6 个 · `hmx-to-llvm-diagnostic-ntile-reject.mlir` 6 个 ·
`hmx-to-llvm.mlir` 5 个（`@bridge` / `@bridge_offset` / `@bridge_ranged` / `@bridge_f32_source`）·
`pack-strided-src.mlir` 4 个 · `hmx-to-llvm-fused-tail.mlir` 3 个 · 其余 8 个）。
**它们今天都拿到 `ensure_dsp`/`unlock_dsp`，且 CHECK 行钉住了。**
⇒ 加 trait ⇒ 这些 CHECK 全红 ⇒ **S1 的「lit 全绿」门当场失败。**
⚠️ 我第一次的扫描脚本 regex 写成 `func `（没匹配 `func.func`），得出「0 个」的假零结论。
**假零比假正更危险——它会让人直接否掉正确的东西。**

### ② 最深的一条：**一个 bit 不能同时对两个方向 fail-safe**

| 判据 | 默认极性 | 忘标记的后果 |
|---|---|---|
| ensure/unlock（`issuesHmxEngineLeaves`） | **未标记 ⇒ 引擎** | 多付一次锁，函数自己会还 ⇒ 无害 |
| 线程角色归属（本方案 §3） | **未标记 ⇒ HVX** | 布局 op 被切进 HMX-role、落到 T_HMX、**去抢 HVX context ⇒ 挂死且无诊断** |

⇒ 这正是 `HexagonTargetTransformInfo.cpp:453-458` 描述的死锁，也正是 §2 方案 A 第 1 条理由要保住的东西。
⇒ **一个 trait 不能同时承担这两个判据。** 需要**两个判据、两个相反的默认极性**：
- `HmxLayoutHvx`（负向豁免，默认引擎）⇒ 解决 ensure/unlock
- **新的正向 trait**（如 `HmxEngineIns`，默认 HVX）⇒ 解决线程角色
  ⇒ 忘标记时退化成「HVX 线程偶尔抢 HMX 锁」= **有竞争、无死锁**

### ③ `verifyHmxLeafCallers` 必然被触发，而 roadmap 全文没提它

`HmxToLLVMPass.cpp:559` 会检查「调了 `hmx_` 前缀叶子的函数必须在 `engineKernels` 里」。
pack/unpack 降到的正是 `hmx_pack_act_f16` / `hmx_unpack_acc_f32`（`HmxExternalFnNames.cpp:24-36`）⇒ 带前缀。
⇒ §2 的目标态**必然产生 pack-only 函数，必然编译失败**。

### ④ 为什么不能复用已有的 `HmxEngineResource` effect

它的极性是「**动了那块唯一的共享硬件状态**」——
`HmxDialect.h:73-76`：「the engine's accumulator / bias-register state **and its staging pipeline**」。
⇒ `stage`/`await` 带它**不是 bug**，它们驱动 staging pipeline。
⇒ 想拿它当判据就必须手工剔掉这两个 = **engine 白名单**，
而 `HmxToLLVMPass.cpp:113-115` **明令禁止**（「omission hangs the device」）。

### ⑤ ⛔ 结构性发现：**有���个决策点，方案的切分点写空了**

- `createHmxPartitionPass` 在 `LinalgToLLVMPass.cpp:536`
- `createHmxToLLVMPass` 在 `LinalgToLLVMPass.cpp:640`（**在后**）
- 而 `HmxToLLVMPass.cpp:1999` 有 `addIllegalDialect<HmxDialect>()`
  ⇒ **在 HmxToLLVM 那个决策点上，hmx op 必须已经全没了**

⇒ **`hmx.matmul` 与 `hmx.alloc_crouton` 在第一个决策点存在、在第二个不存在。**
⇒ **两处的「引擎 op 集合」不同**，而 §2 的图把 ThreadRolePartition 画在 MatmulToHmxPass 之后、
**没写它在 hmx-partition 之前还是之后**。§3.1 的判据表也没提这两个 op。
⇒ **ThreadRolePartition 必须在 hmx-partition 之前切**，判据要覆盖
`matmul` / `alloc_crouton` 这两个在第二点不存在的 op。

### ⑥ 我说错的两处事实

- **`hmx.matmul` 不是容器 op**，是**无 region 的叶子 op**
  （`HmxOps.td:47-49` 只有 `[DestinationStyleOpInterface, MemoryEffects<...>]`，无 region）。
- 「三个 op」确实漏了 **`unpack_acc_f32`（`HmxOps.td:349`）**，它与 `unpack_acc` 状态相同、
  同样降到 `HMXLayout.c`。
  ⚠️ 独立审核给的更硬的理由：`HmxToLLVMPass.cpp:2037-2039` 把 `LowerUnpackAcc` 与
  `LowerUnpackAccF32` 注册成**两个独立 pattern**，漏标不会有任何编译期信号。

### 建议的修法（**待用户签认，未动代码**）

1. `HmxLayoutHvx` 打**四个** op：`pack_act` · `pack_weight` · `unpack_acc` · `unpack_acc_f32`。
2. **另加一个正向 trait** 给线程角色判据用，默认极性相反。
3. `verifyHmxLeafCallers`（`:559`）改成**只查 `HMXAPI.c` 的 5 个引擎符号**
   （`HmxExternalFnNames.cpp:20-24`）⇒ 既解掉破面，又**把这个 check 变强**。
   现有注释 `:554-558` 自己就写明该前缀「**for verification only, never for the decision**」，
   所以**不违反** `:113-115`。
4. 上述 1 与 3 落地时，那 32 个函数的 CHECK 行会变——**那是修一个今天就存在的多余锁，不是回归**，
   必须逐个列出并在 commit message 里说明。

---

## 6. 风险与未验证（预登记，S2 探针优先级从上到下）

1. [未验证] pinned LLVM 是否已含 PR #222340（llvm_triton 子仓在 Linux 侧，本仓不可见）。
2. [未验证] DMA 事件跨线程等待语义（UserDMA 描述符由谁 poll、能否在另一线程 await）。
3. [未验证] VTCMPool 并发 alloc/free 真实覆盖（`bin/runtime/include/VTCMPool.h:15` 有 `#include <mutex>`，`:444` 有 `mutable std::mutex mutex_`，但所有权交接语义需探针）。
4. [未验证] `HAP_compute_res_hmx_lock` 长期持有与其他进程/驱动的交互（探针：独占 N 分钟 + 释放重取）。
5. [未验证] accumulator 跨 kernel 持久的正确性前提（acc_clear 显式化不变量是否处处成立）。
6. [未验证] Hexagon 内存序下环索引的 fence 选型。
7. 已知约束：单 HMX 线程使"多实例并发 HMX"更不可能——与既有 single-instance 立场一致（workspace-resident grid>1 硬拒同款契约）。
8. ✅ **[已决 2026-10-02] S4 门不可达 —— 接受物理，拆成 S4a / S4b。**
   **原发现（保留在案）**：**原 S4** 的机制是「softmax 链（HVX）‖ QK·PV（HMX）」
   ⇒ **可隐藏量 = HMX 引擎在 FA 时间里的份额**，而该份额 LWP **两轮实测为 0.6%**
   （`docs/history/hmx/fa-time-attribution-2026-09-21.md:97` `hmx.acc_clear`/`mma`/`acc_read`
   合计 0.6%；`:101`「引擎彻底无关（0.6%）——第二轮再次确认」；`docs/state/STATE-OF-PLAY.md:600` 复述）。
   项目门是 `max(3×CV,15%)`（`ROADMAP.md:46`）⇒ **0.6% ≪ 15%，纯拓扑收益路径在物理上不可达该门。**
   **决策（门和机制描述都改）**：
   - **S4a（观测台架，无 FA 性能门）**：验收改为 **LWP 归因的重叠率报告**，
     **只有 M3.2 减税那项**套 `max(3×CV,15%)`。
   - **S4b（真收益）**：组合门 = **softmax 链去串行化（43.7%，M4.1 工作面）+ 本拓扑提供并行底座**。
     **拓扑单独份额 ≤ 0.6% 写死在本文档里，不再宣称独立功劳。**
   ⇒ 43.7% 的来源：`fa-time-attribution:222`，`maxnumf` 归约的 running-max 依赖。
9. ✅ **[已决 2026-10-02] S3 目标形状不匹配 —— 采信 S2.5 前置 + 首发改 S2-class。**
   **原发现（保留在案）**：S3 原定打「S1-class matmul」，但 `HmxPartitionPass.cpp:679-706`
   的 `emitSerialTileLoop` **循环体内无 pack**（详见 §3.1 `role-split-nopack`）
   ⇒ **「第 i+1 块 pack」在 S1-class 上没有对象。**
   ⚠️ 另注（一个曾被推翻的推论，保留以免重犯）：`HmxPartitionPass.cpp:1589` 的 `shallow-k` 判定
   **只管 `emitStageLoop` 的 DMA staging 环**（DDR→VTCM 传输），**管不到 pack‖mma**；
   且 `ROADMAP.md:113` 记载 `pipeline-depth=1/2` 可绕过它
   ⇒ **「K=1024 的门让 S1 摊不上跨线程环」这个推理不成立。真正的障碍是 S1 的 tile 循环里没有 pack。**
   **决策**：S2.5 升为**硬前置**；S3 首发形状 = **S2-class**（`pack_act_sites` 1→2 对齐 `s2_anchor`）；
   **S1-class 的 A/B 待 S2.5 完成后再补。**
10. ✅ **[已决 2026-10-02] 「pack 还是 unpack」两难已消解 —— 方案 A 下两者都归 HVX 侧。**
    **原发现（保留在案）**：S1 的 LWP 分区（`hmx-next-round-plan.md:61-70`）：**`unpack` 37.4%** vs
    **`pack_act` 9.8% + `pack_weight` 4.7% = 14.5%**（`exp/hmx/leaf_bw_probe/RESULTS.md:89` 独立复核 14.5%）。
    原稿 §0/§2/§5 三处互相矛盾（`:15` pack · `:70` pack+unpack 列 T_HMX · `:163` pack），
    而唯一有数的那个（§5 S3）选了天花板低 2.6 倍的那一半。
    **决策（§2 方案 A）**：`pack_act` 与 `unpack_acc` **都归 T_HVX**
    ⇒ **两难自动消失，不必二选一**；可重叠对象 ≈ `pack_act` 9.8% + `unpack` 37.4% = **47.2%**
    （`pack_weight` 已被 `enableWeightResident` 默认开消掉）。
    ⚠️ **仍是跨构建 LWP（`5cea8231`/125 µs），仅方向参考；S3 须同构建重测 A/B 才能定。**
11. ✅ **[已决 2026-10-02] 净增量 —— 定为 S3 的第一等验收输出，不作为立项前置。**
    **原发现（保留在案）**：`docs/hmx/hmx-hvx-co-scheduling.md:104-117` 实测
    **真实核比它自己各部分的上界之和还低 37%** ⇒ **单线程内已经在重叠**。
    **决策**：**开工前无法知道净增量，把它当前提等于自我否决；S3 的 A/B 两臂本身就是测量。**
    ⇒ **S3 新增一条 LWP 归因探针臂**，显式输出「**跨线程相对单线程已有 37% 重叠的净增量**」。
    ⇒ **若净增量 < 门 ⇒ 默认保持 OFF + 负结果收档**（正落在 NOT-PROVEN 框架内）。

---

## 7. 为什么这是对的（外部三条锚 + 内部三条锚）

外部：
1. **LLVM PR #222340（2026-09-17）钦点了这个拓扑**（TTI 注释原文描述"dedicates some threads
   to HVX… A thread dedicated to HMX needs no context at all"），并把"约束端"（属性+静态强制）
   捐进了上游；**"推导端"（自动判定哪段代码落哪个线程）没有任何系统做过**（d9 调研结论）——
   你们的 MatmulToHmx 判定就是现成的判据源，缺的只是把 op 级归属升为 region 级线程角色。
2. **llama.cpp 用这个拓扑赢你们 S1 13%**；FA 的 softmax 断点（M4.1 动机）在双线程下变天然重叠。
3. IREE `bind_to_thread`/`deferred_work_queue` 给了运行时抽象样板；Edge TPU/TF soft placement
   给了"自动切分必须响亮"的反面教材。

内部：
1. 四次线程证伪不覆盖角色切分（锁无竞争）；
2. M3.1 退役支持零成本模型判据（本方案判据=能力+依赖）；
3. M3.2（固定税）与 M3.3（共调度）正是本方案的两个收益面；stage/await 与 HmxDmaOnly 极性是现成地基。
