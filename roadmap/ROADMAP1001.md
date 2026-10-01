# 方案：线程角色编译期定死 —— T_HMX 专属线程 + HVX 池双执行器架构

> 基于 hexagon-mlir@`hmx` 分支 HEAD `a5761822`（2026-10-01）+ d9 调研（LLVM/IREE/Triton/Glow/ORT 先例；d9 调研报告未入库，关键先例与 file:line 已内联本文）。
> 按本仓 §6.2 模板组织：契约 → 决策点 → 代码 → FileCheck → host → 设备 A/B → 记录。
> 本文所有本仓引用带 file:line；未读/未测处显式标 [未验证]。

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
| 逐 op 归属的能力谓词（shape/dtype/对齐/VTCM 预算 + reason code + remark） | `MatmulToHmxPass.cpp`（ROADMAP §1.2） | 与 Triton `getMMAVersionSafe`（AccelerateMatmul.cpp:44-85）同型，是"能力驱动自动切分"的正确形态；M3.1 退役结论（不建成本模型）继续成立 |
| `hmx.stage/await` 值边 + DMA staging tile 环 | ROADMAP §4.3、"机制就绪 stage/await 值边 ✅" | 跨线程流水的现成 IR 契约——stage/await 本来就是"提交 DMA/等待完成"的值边，今天在同一线程交错，明天天然跨线程 |
| `OpTrait::HmxDmaOnly` 结构性极性 | HmxToLLVMPass.cpp:90-113（"every *unmarked* dialect op counts as engine until proven otherwise…错并到 HVX 侧=挂死，错并到 HMX 侧=多付一对锁"） | 本方案把这个极性从"要不要锁"推广为"跑哪条线程"，继承同一 fail-safe 方向 |
| 单 HMX 资源 + "引擎不让出线程"前提 | ROADMAP M3.3、§2.4 | 硬约束（qurt_hmx.h:184，实测 0.998），方案的出发点 |
| 测量与门纪律 | 同构建双指纹、max(3×CV,15%)、owner+退出条件、证据指针 | 方案分阶段全按此执行 |

### 1.2 推翻（三处，都给证据）

**推翻一：per-kernel ensure/unlock 不应是终态锁粒度。**
- 现状：`HmxToLLVMPass.cpp:76-79` 定义 ensure/unlock 叶子；`:590-591` "The lock is one pairing per kernel: one ensure at entry, one unlock before each return"；`HexagonCAPI.cpp:205-219` 实现（线程局部 `tHoldsHmxLock` + `HAP_compute_res_hmx_lock`）。
- 三个问题：
  1. **失效模式零容忍**：你们自己的注释（HmxToLLVMPass.cpp:96）——漏一次 unlock，下一个线程在 `HAP_compute_res_hmx_lock` 上**永久挂死**。持锁人是"任意执行该 kernel 的线程"。
  2. **税**：每 kernel 一次锁往返；NON_SHARED unlock 还清 accumulator（HexagonCAPI.cpp:212-213）。属 M3.2 要削的 per-launch 固定税。
  3. **拓扑病根**：async 池里任何线程都可能既跑 linalg fallback（HVX）又跑 HMX 段——正是 LLVM PR #222340 TTI 注释（HexagonTargetTransformInfo.cpp:460-470）描述的上下文饥饿拓扑：*"The hardware provides a fixed number of HVX contexts. Software that mixes the two engines dedicates some threads to HVX, and those threads hold the contexts for as long as they run. A thread dedicated to HMX needs no context at all, until HVX code reaches it."*
- **修正**：锁的所有权 = 一条专属 HMX 线程 × 整个会话。T_HMX 在首次 HMX launch 时 ensure 一次、持有；其余线程永不触碰 HMX 锁。竞态为零，"挂死"失效类整体消失，锁往返每会话 2 次。per-kernel 配对降级为 legacy 路径（A/B 与回滚需要它）。

**推翻二："线程路径已四次证伪"不应遮蔽 M3.3——四次证伪的是工作切分，不是角色切分。**
- ROADMAP §2.4：grid 并行最终在 runtime HMX lock 处串行化。那是 N 线程抢 1 个 HMX，必然串行。
- 角色切分 = 1 条 HMX 线程干 HMX、N 条 HVX 线程干其余：锁无竞争，per-thread 互斥约束**由构造满足**。
- llama.cpp 手写对照组就是这个拓扑（专属 HMX 线程 + HVX worker + 256 深环 + 软件流水），S2/S3 已输给它 0.92×/0.58× 说明你们 leaf 质量已够；**S1 还输 13%（1.13×，2026-09-21 跨构建读数，仅作方向参考），输的恰是它的跨线程流水**（第 i-1 块读出 ‖ 第 i 块 matmul）。本方案就是把这 13% 的机制编进编译器。

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
  · hmx.stage = DMA 提交（引擎无关）    描述符+事件字) · pack/mma/bias/acc_read/unpack
  · 第 i+1 块 stage/pack                            · ensure 一次，锁跨算子
  · 第 i-1 块 unpack 消费                            · 永不 acquire HVX 上下文（编译期保证）
```

四个关键选择：

1. **T_HMX 的"纯度"由编译器保证，不靠纪律**：所有 T_HMX 上跑的函数带 LLVM fn 属性
   `"hexagon_hmx"`（上游 PR #222340，2026-09-17 合并，commit `2e055b8de1a1`，Qualcomm 参与）。
   上游 TTI 保证：`useHVX() = … && !IsHMX`（该函数永不auto-HVX）+ `areInlineCompatible()`
   双向拒绝跨角色内联。语义与测试见 llvm/test/CodeGen/Hexagon/hmx-attr-no-autohvx.ll:1-2
   （"so no HVX unit is acquired on the HMX thread"）——**这正是本方案线程契约的官方表达**。
2. **HVX 侧复用现有 async 底座**：`FormAsyncThreadsPass.cpp:13` "lowering virtual-threads to
   async.execute"（现限 rank-1 forall）→ `bin/runtime/multithreading/HexagonThreadPool`。
   上游 async dialect 无 affinity（d9 已查证），所以角色属性由**自己的** async→runtime
   lowering 消费（路由到哪个执行器），不动上游 async。
3. **桥 = SPSC 环**：生产者（HVX 池线程）写 tile 描述符，消费者（T_HMX）取。事件 ABI 抄
   HmxToLLVMPass 已有的 64-bit 不透明事件字 wire 纪律（`readEventWord` 一节）。环深由
   HmxPartitionPass 的 tile 环推导，不拍脑袋。
4. **拓扑惰性**：manifest 判定 single-role ⇒ 不建线程、不付任何手付开销。短 kernel 零回退。

---

## 3. 编译期角色切分：判据 / 载体 / 检查 / 可观测

### 3.1 判据（决策点全枚举，每个一个 canonical reason code）

| 判定 | 规则（纯能力，零成本模型） | reason code |
|---|---|---|
| region→HMX | 含非 `HmxDmaOnly` 的 hmx op，且区域内其余 op 为标量胶水 | `role-hmx` |
| region→HVX | 无 hmx 引擎 op | `role-hvx` |
| 混合可切 | stage/await 值边能把引擎段与向量段分开 | `role-split-ok` |
| 混合不可切 | 引擎 op 与向量 op 依赖交织、无值边可下刀 | `role-mixed-irreducible` → 单线程语义（=今日行为） |
| 全 kernel 单角色 | | `topology-single-role`（不建线程） |

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
   （`preferCloneToConsumers` 等）加同极性检查——沿用 HmxToLLVMPass.cpp:99-106 的失效方向：
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
- 任何回退/降级 emitRemark（Triton AccelerateMatmul.cpp:69-82 同型——降级必发声）。
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
   `acc_clear`（测试有钉）——补一条不变量：任何 `hmx.mma` 前必须可见 acc 初始化（FileCheck 钉）。

---

## 5. 分阶段落地（每阶段可单 commit 回滚；门 `enableThreadRolePartition` 默认 OFF）

| 阶段 | 内容 | 验收 | 备注 |
|---|---|---|---|
| **S0**（~半天） | 查 pinned llvm_triton/llvm-project 是否含 `2e055b8de1a1`；无则 backport TTI 两个 hunk（~30 行）。HMX leaf 函数带 `hexagon_hmx` 编译通过 | lit 全绿 | 零行为变化 |
| **S1**（host 全验） | ThreadRolePartition pass + attr + verifier + manifest 字段；默认只 emit 单角色 | FileCheck 全套（成功/拒绝/mixed-irreducible/半HMX→PARTIAL+dual-role）；零行为变化 | 纯编译期 |
| **S2**（运行时底座） | 角色执行器 + T_HMX + SPSC 环 + 锁迁移（legacy 共存）；4 个探针：环吞吐、锁长持、DMA 跨线程等待、VTCM 跨线程 alloc/free | host 单元测试 + 探针报告；不跑真 kernel | [未验证]×4 见 §6 |
| **S3**（首个双线程 kernel） | S1-class matmul：HVX stage/pack 第 i+1 块 ‖ T_HMX mma 第 i 块（llama.cpp 读出/矩阵乘重叠的编译器版） | 同构建双指纹 A/B ≥ max(3×CV,15%)；目标吃掉 S1 对手写 13% 差距的全部或大部；NOT-PROVEN 允许 | 短 kernel 手付开销是主要风险 |
| **S4**（FA 重叠 + 减税） | softmax 链（HVX）‖ QK·PV（HMX）——不动融合白名单，纯拓扑收益；顺带量 M3.2（锁持有后 per-launch 固定税降幅） | FA 稳态 17.4ms 基线上 A/B | softmax 链数据依赖紧，环深要小 |
| **S5**（收口） | S3/S4 过门 ⇒ 报用户批准翻默认；per-kernel 配对降级 legacy-only；经验推上游（hexagon 侧 RFC / async affinity） | 门数字 + 契约评审 | 翻默认须用户批准（你们规则） |

依赖：S0→S1→S2→S3→S4 可并行处：S1 与 S2 文件面不重叠，可并行。

---

## 6. 风险与未验证（预登记，S2 探针优先级从上到下）

1. [未验证] pinned LLVM 是否已含 PR #222340（llvm_triton 子仓在 Linux 侧，本仓不可见）。
2. [未验证] DMA 事件跨线程等待语义（UserDMA 描述符由谁 poll、能否在另一线程 await）。
3. [未验证] VTCMPool 并发 alloc/free 真实覆盖（VTCMPool.h:15 有 `<mutex>`，但所有权交接语义需探针）。
4. [未验证] `HAP_compute_res_hmx_lock` 长期持有与其他进程/驱动的交互（探针：独占 N 分钟 + 释放重取）。
5. [未验证] accumulator 跨 kernel 持久的正确性前提（acc_clear 显式化不变量是否处处成立）。
6. [未验证] Hexagon 内存序下环索引的 fence 选型。
7. 已知约束：单 HMX 线程使"多实例并发 HMX"更不可能——与既有 single-instance 立场一致（workspace-resident grid>1 硬拒同款契约）。

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
