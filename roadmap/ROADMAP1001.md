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
> ⚠️ **0.92 与实测表的 0.89 并存且无人裁决**（`ROADMAP.md:50` = 1.13/0.92；
> `hmx-next-round-plan.md:306` 实测表 = 1.14/0.89）。本文抄 `ROADMAP.md:50` 那一套。
> ⚠️ **`ROADMAP.md:50` 自称的权威源 `docs/state/STATE-OF-PLAY.md:559` 引用链已断**
> —— 该行讲的是 `FastInversePass` 除法反号，与本比值无关。

---

## 0. 一句话

把「这段代码跑在哪种引擎线程上」从**运行时事实**变成**编译期事实**：
编译器把每个 kernel 切成 HMX-role 与 HVX-role 两类区域（判据=硬件能力，零成本模型），
运行时只提供两个执行器——**一条进程级专属 HMX 线程（ensure 一次、锁跨算子持有、永不接触 HVX）**
+ **现有 HVX 线程池**——两线程间用 SPSC 环接成软件流水，
让「第 i+1 块的 stage/pack（HVX 侧 DMA+向量）」与「第 i 块的 HMX mma」真正重叠。

---

## 1. 对现有架构的判定

### 1.1 保留（这些是对的，方案全部复用）

| 保留项 | 证据 | 为什么对 |
|---|---|---|
| 逐 op 归属的能力谓词（shape/dtype/对齐/VTCM 预算 + reason code + remark） | `MatmulToHmxPass.cpp`（ROADMAP §1.2） | 与 Triton `getMMAVersionSafe`（AccelerateMatmul.cpp:42-83）同型，是"能力驱动自动切分"的正确形态；M3.1 退役结论（不建成本模型）继续成立 |
| `hmx.stage/await` 值边 + DMA staging tile 环 | ROADMAP §4.3、"机制就绪 stage/await 值边 ✅" | 跨线程流水的现成 IR 契约——stage/await 本来就是"提交 DMA/等待完成"的值边，今天在同一线程交错，明天天然跨线程 |
| `OpTrait::HmxDmaOnly` 结构性极性 | HmxToLLVMPass.cpp:101-110（"every *unmarked* dialect op counts as engine until proven otherwise…错并到 HVX 侧=挂死，错并到 HMX 侧=多付一对锁"） | 本方案把这个极性从"要不要锁"推广为"跑哪条线程"，继承同一 fail-safe 方向 |
| 单 HMX 资源 + "引擎不让出线程"前提 | ROADMAP M3.3、§2.4 | 硬约束（qurt_hmx.h:184 [未验证：SDK 头，不在本仓]，实测 0.998），方案的出发点 |
| 测量与门纪律 | 同构建双指纹、max(3×CV,15%)、owner+退出条件、证据指针 | 方案分阶段全按此执行 |

### 1.2 推翻（三处，都给证据）

**推翻一：per-kernel ensure/unlock 不应是终态锁粒度。**
- 现状：`HmxToLLVMPass.cpp:76-79` 定义 ensure/unlock 叶子；`:590-591` "The lock is one pairing per kernel: one ensure at entry, one unlock before each return"；`HexagonCAPI.cpp:205-219` 实现（线程局部 `tHoldsHmxLock` + `HAP_compute_res_hmx_lock`）。
- 三个问题：
  1. **失效模式零容忍**：你们自己的注释（HmxToLLVMPass.cpp:96）——漏一次 unlock，下一个线程在 `HAP_compute_res_hmx_lock` 上**永久挂死**。持锁人是"任意执行该 kernel 的线程"。
  2. **税**：每 kernel 一次锁往返；NON_SHARED unlock 还清 accumulator（HexagonCAPI.cpp:212-213）。属 M3.2 要削的 per-launch 固定税。
  3. **拓扑病根**：async 池里任何线程都可能既跑 linalg fallback（HVX）又跑 HMX 段——正是 LLVM PR #222340 TTI 注释（HexagonTargetTransformInfo.cpp:460-470 [未验证：本仓不可见]（本地该文件仅 449 行，全文 grep 无此内容））描述的上下文饥饿拓扑：*"The hardware provides a fixed number of HVX contexts. Software that mixes the two engines dedicates some threads to HVX, and those threads hold the contexts for as long as they run. A thread dedicated to HMX needs no context at all, until HVX code reaches it."*
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
  · hmx.stage = DMA 提交（引擎无关）    描述符+事件字) · mma/bias/acc_read  ⚠️见下
  · pack/unpack 归属未定 ⛔                        · ensure 一次，锁跨算子
```

> ## ⛔⛔ 本图有一处**未决硬矛盾**，S2 开工前必须先解决
>
> **原稿在 T_HMX 侧同时列了 `pack`/`unpack`，又同时声明 T_HMX「永不 acquire HVX 上下文」。这两句不能同时成立。**
>
> **证据**（已核）：`bin/runtime/hmx/src/HMXLayout.c` 里**引擎 intrinsic 零命中**
> （`Q6_mx*`/`mxmem`/`mxclracc`/`Q6_bias*`/`Q6_activation*`/`Q6_weight*` 全部 0），
> **31 处是 HVX intrinsic**（`Q6_vmem_QRIV`/`Q6_vscatter_RMVhV`/`Q6_W_vdeal_VVR` …）。
> 引擎指令全在 `bin/runtime/hmx/src/HMXAPI.c:32,37,43,53-54`。
> ⇒ **`pack_act` 与 `unpack_acc` 是纯 HVX 代码。**
>
> **所以二选一，且两侧不可兼得**：
>
> | | 主张 | 代价 |
> |---|---|---|
> | **方案 A** | `pack`/`unpack` 归 **T_HVX** ⇒ T_HMX 只做 `mma/bias/acc_read` | 与 §1.1 的 `OpTrait::HmxDmaOnly` 极性**方向相反**（今天未标记 = 算引擎 op）⇒ **必须新增 trait**，否则 T_HVX 上的 pack 去抢 HMX 锁 ⇒ **永久挂死**（`HmxToLLVMPass.cpp:96-99` 的失效模式）。见 §4.5 |
> | **方案 B** | `pack`/`unpack` 留 **T_HMX** ⇒ T_HMX **需要** HVX 上下文 | ⚠️ **上游 `hexagon_hmx` 的语义失效**：其前提是函数内**没有** HVX 代码（`hmx-attr-no-autohvx.ll` 原文 "so no HVX unit is acquired on the HMX thread" [未验证：本仓不可见]）。且 T_HMX 线程的"纯度"保证被打破，`hexagon_hmx` 不再是它的静态契约 |
>
> ⚠️ **原稿 §2.1 引用上游那句「这正是本方案线程契约的官方表达」，在方案 B 下是错的。**
> **⇒ 这是本文档最硬的一处内部矛盾，且不在 §6 的风险表里。**

> ### 重叠哪一半 —— 唯一权威表述
>
> **§0 / §2 / §5 三处原稿互相矛盾**（`:15` 说 pack；`:70` 把 pack+unpack 同时列 T_HMX；
> `:71-72` 又同时列 T_HVX；`:163` 说 pack）。**以本节为准。**
>
> **可重叠的对象按引擎归属分两类**：
> - **纯 HVX 代码**（`pack_act` / `unpack_acc` / `hmx.stage`）⇒ 只能在 T_HVX 侧
> - **纯引擎代码**（`hmx.mma` / `acc_read` / `bias_load`）⇒ 只能在 T_HMX 侧
>
> **⇒ 跨线程 SPSC 环的天然刀口，就是这条引擎归属边界。**
> **⇒ 而 `§3.1` 现在的刀口写的是 stage/await 值边** —— 那是 DMA 语义，不是引擎归属，
> **两者不是同一条线**（serial 路径上没有 `hmx.stage`/`hmx.await` 可切，见 §3.1 补注）。

四个关键选择：

1. **T_HMX 的"纯度"由编译器保证，不靠纪律**：所有 T_HMX 上跑的函数带 LLVM fn 属性
   `"hexagon_hmx"`（上游 PR #222340，2026-09-17 合并，commit `2e055b8de1a1`，Qualcomm 参与）。
   上游 TTI 保证：`useHVX() = … && !IsHMX`（该函数永不auto-HVX）+ `areInlineCompatible()`
   双向拒绝跨角色内联。语义与测试见 llvm/test/CodeGen/Hexagon/hmx-attr-no-autohvx.ll:1-2 [未验证：本仓不可见]
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
> ⇒ **判据需要第二种刀口，或显式承认 S4 在 QK 那一侧不成立**（见 §5 S4 备注）。

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

5. ⛔⛔ **引擎归属的第三类标记**（**缺了会永久挂死**）
   `issuesHmxEngineLeaves`（`HmxToLLVMPass.cpp:527-536`）把**任何**非 `HmxDmaOnly` 的 hmx op
   当引擎 op。而 `pack_act`（`HmxOps.td:156-158`）、`pack_weight`（`:228-230`）、
   `unpack_acc`（`:288-290`）**三者都没有 `HmxDmaOnly` trait，也都没有 `HmxEngineResource` effect**
   （全 `.td` 只有 2 个 `HmxDmaOnly` 载体：`stage:570` / `await:623`）
   ⇒ **今天 pack/unpack 被算成引擎 op。**
   **若 pack/unpack 移到 T_HVX 而不新增「要 HVX、不要引擎」的第三类标记，
   T_HVX 上的 pack 会去抢 HMX 锁 ⇒ 永久挂死**——正是 `HmxToLLVMPass.cpp:96-99` 警告的失效模式。
   ⚠️ **§1.1 说"继承同一 fail-safe 方向"，但这不是继承，是要新增一个 trait。**
   ⇒ 归主笔待决（§2 的方案 A/B 取决于此）。

6. ⛔ **host→device 的分流通道不存在**
   §4.2 说 legacy 路径「由 manifest 的 `topology` 字段区分」。但**设备 runtime 今天完全不读 manifest**：
   `bin/runtime/src/HexagonAPI.cpp` 与 `HexagonCAPI.cpp` 里 `grep manifest` 与
   `grep topology` **均 0 命中**。manifest 只到 host launcher
   （`backend/triton_hexagon_launcher.py:557-568` → `utils.py:1658` `enforce_hmx_launch_contract`）。
   ⇒ 要按 `topology` 在设备侧分流，**需要一条新的 launch 通道**
   （launch 参数 / per-kernel 符号 / 环境变量）。**manifest 字段本身不够。**

---

## 5. 分阶段落地（每阶段可单 commit 回滚；门 `enableThreadRolePartition` 默认 OFF）

| 阶段 | 内容 | 验收 | 备注 |
|---|---|---|---|
| **S0**（~半天） | 查 pinned llvm_triton/llvm-project 是否含 `2e055b8de1a1`；无则 backport TTI 两个 hunk（~30 行）。HMX leaf 函数带 `hexagon_hmx` 编译通过 | lit 全绿 | 零行为变化 |
| **S1**（host 全验） | ThreadRolePartition pass + attr + verifier + manifest 字段；默认只 emit 单角色 | FileCheck 全套（成功/拒绝/mixed-irreducible/半HMX→PARTIAL+dual-role）；零行为变化 | 纯编译期 |
| **S2**（运行时底座） | 角色执行器 + T_HMX + SPSC 环 + 锁迁移（legacy 共存）；4 个探针：环吞吐、锁长持、DMA 跨线程等待、VTCM 跨线程 alloc/free | host 单元测试 + 探针报告；不跑真 kernel | [未验证]×4 见 §6 |
| **S2.5**（⛔ 新增前置） | **给 S1-class 引入 per-tile pack**：把整数组 prologue 的 pack（`MatmulToHmxPass.cpp:1139-1163`）折进 tile 循环，让 S1 形状**有可重叠对象** | 纯 host：manifest `pack_act_sites` 从 1 变 2（对齐 `s2_anchor`）；lit 全绿 | ⛔ **S3 的硬前置**。不做这步，S3 在 S1-class 上按 §3.1 自己的判据就是 no-op（`role-split-nopack`） |
| **S3**（首个双线程 kernel） | **S2-class** matmul（`256×64×2048`，Kt=64，**唯一已有 per-tile pack 的形态**）：HVX pack 第 i+1 块 ‖ T_HMX mma 第 i 块 | 同构建双指纹 A/B ≥ max(3×CV,15%)；NOT-PROVEN 允许 | ⚠️ **不能是 S1-class**（见 S2.5）。⚠️ 且见 §6.10：重叠 pack 还是 unpack 未决，**天花板差 2.6 倍**，本阶段须先定 |
| **S4**（FA 重叠 + 减税） | softmax 链（HVX）‖ QK·PV（HMX）；顺带量 M3.2（锁持有后 per-launch 固定税降幅） | ⛔ **原验收门「FA 稳态 17.4ms 上 A/B，max(3×CV,15%)」物理不可达** —— 引擎份额实测 0.6%（§6.8）⇒ **验收门或机制描述必须二选一改** | ⚠️ QK 落 `serial:shallow-k`（`attn_qk_d128` Kt=4）⇒ **重叠主体是 softmax 链，QK 走 serial 不影响**；但 §3.1 的 stage/await 刀口在 QK 上恒不成立 |
| **S5**（收口） | S3/S4 过门 ⇒ 报用户批准翻默认；per-kernel 配对降级 legacy-only；经验推上游（hexagon 侧 RFC / async affinity） | 门数字 + 契约评审 | 翻默认须用户批准（你们规则） |

**依赖：`S0 → S1 → S2 → S2.5 → S3 → S4 → S5`**

⚠️ **原稿"S1 与 S2 文件面不重叠，可并行"不成立**，有两处真实的**写-读**依赖：

| 依赖 | 证据 |
|---|---|
| 门 `enableThreadRolePartition` 需照 `enableHmxPipelineDepth` 的现成**三处**接线 | `include/hexagon/Dialect/Hmx/Transforms/Passes.td:134`（`Option<"pipelineDepth", "pipeline-depth", "int64_t", /*default=*/"0">`）· `lib/Target/Linalg_MLLVMIR/MLLVMIRTranslation.cpp:154-155` · `backend/hexagon_options.py:157`（`enableHmxPipelineDepth: int = 0`）。**S1 与 S2 都碰这三处** |
| manifest `topology` 字段：**S1 写**（§3.4）、**S2 的 legacy 分支读**（§4.2） | ⇒ **S2 的第 2 项不可能在 S1 之前完成** |

⇒ **S1 与 S2 只在「探针 / 单元测试」这部分文件面确实不重叠、可并行。**

⚠️ **S1/S2 开工前必须先解决 §2 的引擎归属硬矛盾**（`pack`/`unpack` 归 T_HVX 还是留 T_HMX）——
它决定 §4.5 要不要新增 trait，而**选错就是永久挂死**。

---

## 6. 风险与未验证（预登记，S2 探针优先级从上到下）

1. [未验证] pinned LLVM 是否已含 PR #222340（llvm_triton 子仓在 Linux 侧，本仓不可见）。
2. [未验证] DMA 事件跨线程等待语义（UserDMA 描述符由谁 poll、能否在另一线程 await）。
3. [未验证] VTCMPool 并发 alloc/free 真实覆盖（VTCMPool.h:15 有 `<mutex>`，但所有权交接语义需探针）。
4. [未验证] `HAP_compute_res_hmx_lock` 长期持有与其他进程/驱动的交互（探针：独占 N 分钟 + 释放重取）。
5. [未验证] accumulator 跨 kernel 持久的正确性前提（acc_clear 显式化不变量是否处处成立）。
6. [未验证] Hexagon 内存序下环索引的 fence 选型。
7. 已知约束：单 HMX 线程使"多实例并发 HMX"更不可能——与既有 single-instance 立场一致（workspace-resident grid>1 硬拒同款契约）。
8. ⛔⛔ **[未验证 · 门不可达] S4 的验收门与机制上限矛盾。**
   S4 的机制是「softmax 链（HVX）‖ QK·PV（HMX）」⇒ **可隐藏量 = HMX 引擎在 FA 时间里的份额**，
   而该份额 LWP **两轮实测为 0.6%**（`docs/history/hmx/fa-time-attribution-2026-09-21.md:97`
   `hmx.acc_clear`/`mma`/`acc_read` 合计 0.6%；`:101`「引擎彻底无关（0.6%）——第二轮再次确认」；
   `docs/state/STATE-OF-PLAY.md:600` 复述）。
   项目门是 `max(3×CV,15%)`（`ROADMAP.md:48`）⇒ **0.6% ≪ 15%，纯拓扑收益路径在物理上不可达该门。**
   而 §5 S4 同一格写「**不动融合白名单，纯拓扑收益**」——引擎只有 0.6%，纯拓扑收益上限就是 0.6%。
   **这是方案内部的第二处硬矛盾。**
   ⇒ 若 S4 要过门，必须**同时解决 softmax 链本身的串行化**
   （`fa-time-attribution:222` 记 softmax 链 **43.7%**，`maxnumf` 归约的 running-max 依赖），
   那是 **M4.1 / `docs/hmx/fa-softmax-serialization-plan.md` 的工作面**，
   **与"不动融合白名单、纯拓扑收益"不是同一件事**。
   ⇒ **S4 的验收门或机制描述必须二选一改。**
9. ⛔ **[未验证] S3 的目标形状与机制不匹配。**
   S3 原定打「S1-class matmul」，但 `HmxPartitionPass.cpp:679-706` 的 `emitSerialTileLoop`
   **循环体内无 pack**（详见 §3.1 `role-split-nopack`）⇒ **「第 i+1 块 pack」在 S1-class 上没有对象**。
   ⚠️ 另注：`HmxPartitionPass.cpp:1589` 的 `shallow-k` 判定**只管 `emitStageLoop` 的 DMA staging 环**
   （DDR→VTCM 传输），**管不到 pack‖mma**；且 `ROADMAP.md:113` 记载 `pipeline-depth=1/2` 可绕过它
   ⇒ **「K=1024 的门让 S1 摊不上跨线程环」这个推理不成立**（原 §3.1 讨论曾据此推论，已删）。
   **真正的障碍是上面那条：S1 的 tile 循环里没有 pack。**
10. ⛔ **[未验证] pack 还是 unpack —— 天花板差 2.6 倍。**
    S1 的 LWP 分区（`hmx-next-round-plan.md:61-70`）：**`unpack` 37.4%** vs
    **`pack_act` 9.8% + `pack_weight` 4.7% = 14.5%**（`exp/hmx/leaf_bw_probe/RESULTS.md:89` 独立复核 14.5%）。
    **§0/§2/§5 原稿三处互相矛盾**（`:15` pack · `:70` pack+unpack 列 T_HMX · `:163` pack），
    而**唯一有数的那个（§5 S3）选了天花板低 2.6 倍的那一半**。
    ⚠️ 另注：**`pack_weight` 那一档已被 `enableWeightResident`（默认开）消掉**
    ⇒ 真正可重叠的只有 `pack_act` ≈ 9.8%，**比 7.25 µs 的差距还小** ⇒ **S3 的目标可能不可达**。
    ⇒ **需同构建重测 A/B 才能定，跨构建数不可用于决策。**
11. ⛔ **[未验证] 单线程内可能已经有可观重叠。**
    `docs/hmx/hmx-hvx-co-scheduling.md:104-116` 实测：**真实核比它自己各部分的上界之和还低 37%**
    ⇒ **单线程内已经在重叠**。这对本方案是双向的：
    **支持** = 跨线程只是把已有重叠做得更彻底；
    **反对** = **那 37% 已经被吃掉了，跨线程的增量空间要在这 37% 之外算。**
    ⇒ **立项前必须先答：跨线程相对"单线程内已有重叠"的净增量是多少。**

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
