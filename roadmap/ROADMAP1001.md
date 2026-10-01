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
