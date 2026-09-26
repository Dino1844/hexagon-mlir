# Hexagon-MLIR — HMX 后端路线图（Roadmap）

> 本文是本 fork（`hmx` 分支）的路线图：现状定位与后续工作方向。上游 [README.md](README.md) 保持原样。
> 当前仓库没有提交 `docs/state/STATE-OF-PLAY.md`；因此本文件只把已在源码、测试、文档和提交记录中可复核的事实作为基线。后续 agent 必须先更新本文件中的状态和证据，再实现对应工作项。
> agent 修改/测试规则见工作区的会话级 `AGENTS.md`。

---

## 0. 一句话定位

> **我们做成的是一个相当完整的 "HMX-aware Triton/MLIR backend"，
> 还不是 "任意 Triton 自动变成高性能 HMX kernel" 的通用编译器。**

——第三方评审（2026-09）的结论，本路线图以它为基线：

| 维度 | 评分 | 含义 |
|---|---|---|
| HMX 路径工程完成度 | **7.5 / 10** | vertical slice 扎实：闭环、可退让、面向真实设备，明显超过 demo/prototype |
| 通用 Triton 自动优化/融合 | **4.5 / 10** | 支持一个受约束的 Triton 子集；适合的 matmul 自动走 HMX，其余走 HVX/Linalg fallback |

**评审对方向的判断**：路线正确；下一阶段核心**不是继续堆 HMX leaf**，而是补齐
"前端支持矩阵 + cost model + 动态 shape/tail + 可观测 fallback + 跨算子融合"。
本路线图即按此展开。

---

## 1. 现状：已完成的 vertical slice

```
Triton → TTIR → triton-shared / Linalg
       → matmul-to-hmx（引擎归属 + crouton 布局进类型 + VTCM 预算）
       → bufferize → weight-resident / hmx-partition（tile 环 + DMA staging 软件流水）
       → hmx-workspace-resident（opt-in）→ hexagonmem（space-1 → runtime VTCM 池）
       → hmx-to-llvm（llvm.call + 引擎 ensure/unlock 配对）
       → SDK clang -mhmx 编译的 runtime leaves（libhmxapi.a）
       → host launcher / weight prepack 契约
```

评审认可的四个优点（浓缩）：

1. **完整闭环而非单点 lowering**：方言、归属、partition、常驻、lowering、runtime、host 预排七件一体；
2. **不强行替换**：HMX 逐 op 判定（VTCM allocator / shape / dtype / budget），失败保留原 `linalg.matmul` 并发 remark 说明原因；
3. **真实设备问题都处理了**：HMX lock（单资源串行化语义明确成文）、VTCM 常驻、DMA staging、权重预排、锁的 ensure/unlock 配对；
4. **面向设备的真优化**而非 IR 形式变换：M-blocking、activation staging、weight resident、fused tail、crouton 布局传播。

关键实测（同构建真机，判据 `max(3×CV, 15%)`；数字随轮次更新）：

| 项 | 数字 |
|---|---|
| 稳态 vs llama.cpp 手写（WR 默认开） | **S1 1.13× / S2 0.92× / S3 0.58×**（S2/S3 已快过手写） |
| FlashAttention | 稳态 **32.4 → 17.4 ms（1.86×，f32 激活 ABI）**；仓库 FA 测试 **18312 → 8184 µs**（NUM_THREADS 4→1）；评审引述 39.4 → 18.1 ms 未复测 |
| 常用算子 | `vec_add` 快手写 4.8×、`matmul` 1.15×、softmax/rms_norm 见 `docs/results/op-steady-state-2026-09-21.md` |
| host 门 | full manual lit **240：239 过 / 0 失败 / 1 skip**（`vector_size.mlir` 的 `REQUIRES`；P1.5 alignment/resident-v2 slice 后重跑） |
| 仓库状态 | inner `HEAD=26bbba0`（P1.3c 已提交并 push 到 `fork/hmx`）；当前 P1.5 census/liveness slice 未提交，按要求未重生成 stored patch |

---

## 2. 为什么还不能"随便写 Triton 就自动高效"（五条边界）

1. **HMX 适用面窄**：rank-2、静态、M/N/K 32 对齐、f16/f32、accumulator 可证明为空、VTCM 放得下（`MatmulToHmxPass` 合法性谓词 + `HmxTarget` 能力表）。非对齐 / 动态 shape / 非标 contraction / int8·bf16·fp8 / 复杂 mask·padding / 多 matmul 预算不足 ⇒ **编译成功但退回 HVX/Linalg，性能不等于 HMX**。且 f32 经 pack 量化到 f16、读出再 widen——是**相对误差预算语义，不是 bit-exact fp32**。
2. **融合是保守白名单**：仅 all-parallel、单入单出、全 f16、add/sub/mul/div、常量 splat、单一使用者；reduction / mask·cmpi·select / exp·log·sqrt / 类型转换 / max·min / 多用户 producer 一律不进（有设备测量依据：mask/phi/exp 链搬进 crouton 序可能慢 ~10×）。⇒ FA 中只有 QK·PV 两个 dot 有机会走 HMX，softmax 链仍是断点，做不到整块 attention 融合。
3. **前端仍是实验性入口**：实际走 `triton-shared-opt --triton-to-linalg-experimental`；已有 v1 文档支持矩阵（`docs/codegen/triton-support-matrix.md`），但前端自动查询、按 shape 的策略选择、cost model、HMX/HVX/HexKL 统一决策器仍未接入（`compiler.py` 自注需重构为动态 pass pipeline）。
4. **单资源 ≠ 并行扩展**：Triton grid 并行 ≠ HMX engine 并行；多线程最终在 runtime HMX lock 处串行化（线程路径已四次证伪），workspace-resident 明确要求 single-instance（grid>1 需自担风险，opt-in）。
5. **成熟度**：R1（maxnum legalize）等上游 LLVM Hexagon 后端 RA bug 待修 ⇒ 默认 OFF；triage env 门保留待退役；full manual lit 当前 0 红、1 个 `REQUIRES` skip；`docs/`、`tools/`、`AGENTS.md` 在工作区侧无版本控制。

---

## 3. 路线图（按优先级分四阶段）

### 阶段一 · 可观测与收口（P0 — 小步、先做）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M1.1 | **决策系统 / kernel manifest**（评审差距①） | 每个 matmul 报告：是否走 HMX、为何没走、VTCM 用量、是否 blocking、pack/unpack 次数。现有 remark/warning 收敛为**结构化 manifest + 诊断接口**；先保留现有 remark/warning 作为兼容输出 | 标准批可一键打印 manifest；字段与 pass 内判定一一对应；manifest 缺失或字段不一致使测试失败 |
| M1.2 | **Triton 支持矩阵**（差距②） | 明确四档：可编译 / 可 HMX / 仅 HVX / 拒绝；落成文档 + 测试矩阵，替代"散落在 pass 条件里" | 矩阵每格有对应测试；新 kernel 能 5 分钟查到归宿 |
| M1.3 | **记档项按触发执行** | 先建立缺失状态文档，记录布局生命周期、BufferManager 五入口、预算归属、resident 内容校验的现状和触发条件；再按文档中的触发条件实施，不把未存在的 `STATE §7.8/§7.9` 当作已批准设计 | 每个记档项都有源码证据、设计决议、lit 锁和设备验证记录 |
| M1.4 | **清理与退役** | 先复现并分类 `return_alloc_from_loop`；只有确认是既存且与本线无关时才 XFAIL。对 `HEXAGON_EPI_LOG`、`HEXAGON_PTR_LOG`、`HEXAGON_ASM_DUMP` 等环境门逐项标明用途、默认值、移除条件 | `run_lit_all` 的结果可复现；每个红测/临时门都有 issue、owner 和退出条件 |

### 阶段二 · 形状与类型扩展（P1 — 解锁可用面）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M2.1 | **形状语义与 specialization 前置设计** | 先定义 Triton/TTIR/Linalg 中的运行时 shape、边界 mask、padding、返回布局和 launcher 参数契约；明确 compile-time specialization 与 runtime dispatch 的责任边界 | 设计文档包含 IR 示例、非法输入诊断、缓存 key 影响、正确性 oracle 和 fallback 选择表；没有该契约不得开始尾块实现 |
| M2.2 | **tail 与 padding 机制** | 在独立的 HMX tail IR 形式和 runtime ABI 上实现非 32 对齐边界；完整 tile 继续走现有 `MatmulToHmxPass`，无法证明安全时保留 Linalg/HVX | 非 32 对齐 shape sweep 正确性全过；每个尾块路径都有 FileCheck 和设备用例；无按 kernel 名称的特判 |
| M2.3 | **动态 shape specialization**（差距③） | 在 M2.1/M2.2 之后实现四路选择：完整 HMX tile / HMX+tail / HVX fallback / 混合 kernel；选择结果写入 manifest，并纳入 Triton cache key | 动态 shape sweep 正确性全过；混合路径相对纯 fallback 的性能门槛和失败回退规则写入测试；编译失败与运行时回退可区分 |
| M2.4 | **dtype 能力契约表** | 按 (dtype 对, 累加/读出位宽, 每操作数行走轴) 键化的 `HmxTarget` 契约；int8/bf16/fp8 **先 PRM/引擎能力调研再落** | 第二个 dtype 至少一条真机 A/B；"加能力只改一处"成立；不支持 dtype 必须有稳定诊断 |
| M2.5 | **大 shape 常态化** | `planBridge` 的 M 分块已落；补**全 kernel VTCM 记账**，覆盖 resident weight、activation ring、accumulator、status、workspace 和其他 space-1 分配的生命周期 | 预算判定对整池成立；大 shape 进入标准批；manifest 能解释每一项 VTCM 占用 |

> **P1.5 evidence slice（2026-09-26，commit `270f135`）**：内部 `hmx-vtcm-accounting` diagnostic pass 与 `hmx.diagnostic_vtcm_liveness` 已提交；它们仍只在显式 marker 下运行，不比较 VTCM budget、不写 manifest v2、不被 launcher 消费。当前可证明范围包括 canonical source identity、structured liveness、views/aliases、DPS aliases、resident provenance 与 fail-closed unsupported classes。runtime site/grid/content/full-occupancy 等轴仍显式 `not-proven`。

> **M2.5 版本决策（2026-09-25）**：v2 `hex.hmx.kernel_manifest/v2` 继续逐字保持 bridge-only；v3 是新的 record-only wire schema，禁止从 v2 自动转换，需 coordinated envelope/cache/launcher migration，但不授权 production budget、`hmx-tail` 或 grid 决策；v4 延后到 full-kernel runtime evidence 完成后，才评审 production admission/budget gating。v3 不阻断已有 bridge-budgeted `full-hmx`，但阻断依赖 full-kernel facts 的 tail/cost-model 决策。当前已有两次 gate-ON、remote-attested process-aggregate probe capture（每次 `6 pass / 10 not-proven / 0 fail`，新 cache phases 与 `RESULT: PASS`），以及当前 gate-OFF build 上 direct tail `95/95` + resident `1/1`；一般 identity/content/allocator/full-occupancy 轴仍不是 v3 admission 证据。正式契约见 [`docs/hmx/hmx-v3-manifest-decision.md`](../docs/hmx/hmx-v3-manifest-decision.md)。

> **P1.5 promotion evidence（2026-09-26）**：commit `270f135`；manual lit `284 passed / 0 failed / 1 skipped`（285 tests），host/source matrix `13 pass / 14 not-proven / 0 fail`，probe/evidence host contracts `50 + 19` tests 通过。两次 gate-ON probe capture 均有四artifact manifest 与 remote attestation `match=true`，每次 `6 pass / 10 not-proven / 0 fail`；当前 gate-OFF `libtriton.so=68b0a38c…`、`linalg-hexagon-opt=6b7df7c7…`、`libhmxapi.a=71c20e3c…` 上 direct tail `95/95`、resident `1/1`，一次 launch transient 恢复。独立 reviewer 已给出 **R-A/R-B scoped PROMOTE**；用户已批准启动 R-C v3 record-only migration。`observed_high_water`、`resident`、完整 allocator/full-occupancy 等未证明轴仍不得改写为 complete，production `hmx-tail`/v4 仍未授权。

### M2.5-R：Evidence-before-manifest（当前唯一主线）

**研究结论**：先冻结事实语义和证据，不先冻结 manifest wire schema。v3 只能在
identity、liveness、allocator 和 tail oracle 都闭合后生成；v4 再晚一个阶段。

#### R-A：Identity / Allocator 轨道

- 冻结 `build_id / function_id / allocation_site_id / resident_scope_id`；
  不记录 raw address。
- 建立 static site ↔ runtime event 的 join；若 join 不可证明，则 v3 明确
  aggregate-only，不生成 per-site observed。
  - **2026-09-26 已闭合**：离线 join（`exp/hmx/vtcm_accounting_probe/site_join.py`）的**过度拒绝已修**——
    按 report window 分窗，故意的小缓冲 `-2` probe 不再被当成截断报告（25 项契约测试全绿）；并**造出缺的那份
    设备证据**：单站点、由编译器 `site_scope_enter` bracket 注册的 capture（`site_join_anchor.mlir` +
    `run_probe.py` 的 `VTCM_SITE_JOIN=1` 模式），gate-ON 上机后
    `{"status":"joined","windows_joined":[1,2],"windows_excluded":[],"bound_events":2}`，
    文档 `logs/hmx/site_join_result.json`。`mode=diagnostic-only`、`observed_device=false`、
    `performance_claimed=false`：它是**离线对账**，不是设备观测，也没有授权任何 admission。
- 完成 resident key/content/process identity：同 key 不同 bytes/alignment/content、
  多 module/process、cold/warm reuse 必须有明确接受或 fail-closed 结论。
- 建立 requested / allocator-aligned / charged / resident 的单位和关系模型，
  覆盖 128/256/2048 alignment、split/coalesce、free-cache 和 fragmentation。

**出口**：同一 build/scope 的 host/runtime evidence 可复核，且每个未证明轴都有
机器可读的 `not-proven/incomplete`；不要求先写 v3。

#### R-B：Liveness / Tail 轨道

- 将 structured-SCF 子集扩展到所有 function blocks、nested region、alias/view、
  pointer escape、call、async/DMA、dynamic extent，并对未支持形状 fail closed。
- 真实 production-shaped tail fixture：`grid=1`、M/N/K 单轴和组合 tail、
  zero-pad、residual store、workspace/weight resident。
- 先完成 correctness oracle 和边界矩阵；性能 A/B 只能在 correctness 闭合后开始。

**出口**：host lit + device correctness 全部通过，且 tail 不依赖静默 HVX retry。

#### R-C：Wire / Migration 轨道（依赖 A+B）

- 先实现独立 internal facts/sidecar；不让 launcher 消费 diagnostic probe。
- A/B 出口后，才实现 v3 producer、strict validator、Python consumer、envelope、
  cache token 和 stale-cache 负例；禁止 `v2_to_v3()`。
- v3 仍只记录，不授权 budget/tail/grid；v4 admission 另立评审。

#### 并行与禁止项

- A 与 B 在 fact contract 冻结后并行；C 必须等 A/B evidence。
- 不新增 backend option，不向 v2 填 peak/workspace/observed 字段。
- 不把 process high-water 归因到单 function/site，不用 `not-proven` 代替 0。
- 不在 v3 record-only 上直接打开 production `hmx-tail`。

### 阶段三 · 统一决策与代价模型（P2 — 差距④）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M3.1 | **统一 cost model** | 用 M/N/K × dtype × VTCM 余量 × 线程数 × pack 成本，选择 HMX / HVX / HexKL / DMA / fusion——替代"pass 顺序 + 硬编码阈值" | 同构建 A/B：选择器 ≥ 现状启发式；误判可从 manifest 追溯 |
| M3.2 | **削 per-launch 固定税**（STATE 优先级①） | bring-up ≈1.5 ms 对短工作负载致命 ⇒ 会话/op-batch 级常驻与复用 | `Perf(N)=A+B/N` 的 B 显著下降；S3 类短形状提升 > max(3×CV,15%) |
| M3.3 | **HVX×HMX 共调度** | 按 `hmx-hvx-co-scheduling` 的机制草案实现（引擎不让出线程是前提约束） | FA/混合 kernel 中两引擎重叠有实测收益才合入 |

### 阶段四 · 跨算子融合（P3 — 差距⑤，最高价值最难）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M4.1 | **reduction / normalization / attention 建模为 pipeline** | softmax、layernorm、mask、exp、reduction 从"断点"变成显式值边流水（复用 `hmx.stage/await` 调度契约与 pipeliner） | 单算子（softmax/rms_norm）不回退；FA 中断点可见地减少 |
| M4.2 | **融合白名单按机制理由渐进扩** | 每扩一类（先 reduction 族？exp 族？）必须：真机 A/B + 机制理由，**禁止按 shape/kernel 特判**（AGENTS §7.6） | 每步过双评审 + 同构建 A/B ≥ 判据 |
| M4.3 | **attention block 级融合** | QK·softmax·PV 一条 pipeline；FA 转置融合的 24-bit 2D DMA 前置已验证 | FA 稳态目标另定（当前 17.4 ms 为基线） |

### 贯穿 · 前端与工程基线

- **前端**：`triton-to-linalg-experimental` → 支持矩阵 + 明确诊断；`compiler.py` pass pipeline 动态化（其自注 TODO）；
- **上机纪律**：同构建 A/B、`libtriton.so` 与 `libhmxapi.a` **双指纹**、设备锁、判据 `max(3×CV,15%)`；
- **流程**：每个行为改动过 **architecture-review + ai-slop-cleaner 双评审**（先例：`b947063`），writer/reviewer 分离；
- **补丁**：P0.2 基线 stored patch 已校验；P0.3 commit `ca679fd` 已 push，但 stored patch 仍未重生成。当前差异以 doctor 输出为准，不得为消除门红灯自动改 patch。

---

## 4. 明确不做（非目标）

- **不继续堆 HMX leaf**（除非 PRM 出新引擎能力）；
- **不做** per-operand effects / 显式 `!hmx.acc` / 更多流水 stage（已拍板延后，STATE §8 ⑤）；
- **不做多 HMX 单元并行**（硬件单资源；线程化路径已四次证伪）；
- **不重写、不投资 hexkl**（ADR-001：零投资、不动上游，仅在触碰 `PreprocessWeightsForHMXPass` 时重估）；
- **不追 bit-exact fp32**（f32↔f16 量化语义已接受，以相对误差预算验收）；
- **不把 workspace-resident 默认打开**（single-instance 约束，grid>1 有覆盖风险）。

---

## 5. 一页速查

| 阶段 | 主题 | 解锁能力 | 状态 |
|---|---|---|---|
| P0 | manifest + 支持矩阵 + 记档收口 + 清理 | 可解释、可维护、门真绿 | P0.2 manifest 已完成；P0.3 支持矩阵已在 `ca679fd` 提交并 push |
| P1 | 动态 shape/tail + dtype 契约 + 大 shape 记账 | 任意 shape、更多 dtype 可用 | 部分已落（M 分块 ✅） |
| P2 | cost model + 固定税 + 共调度 | 决策不再靠 pass 顺序与硬阈值 | 规划中 |
| P3 | reduction/attention 融合 | 从"dot 走 HMX"到"block 走 HMX" | 机制就绪（stage/await 值边 ✅），待实现 |

> **当前状态（2026-09-26 晚，N1/N2/N3 收口后）**：P0.2/P0.3 与 P1.2/P1.3c 已分别提交并 push；P1.5 evidence slice 在 `270f135`，R-C v3 record-only migration 在 `098b2a0`（`fork/hmx`）。当前 manual lit **`301 passed / 0 failed / 1 skipped`**，边界矩阵 **29 格 `14 pass / 15 not-proven / 0 fail`**（`device_evidence_status=current`、`device_evidence_promotable=true`），probe host 测试 **108 passed**、runtime 源码契约 **9/9 PASS**；一份 remote-attested 设备 capture `logs/hmx/n123-a3-device-2026-09-26.log`（overlay `current`/`promotable`、0 validation error）导入 **13 recorded / 5 pass** 设备观测。R-A/R-B 已 scoped PROMOTE；v3 record-only 的**可达性**已定为永久设计边界（§6.5 勘误 7）；v4 与 production `hmx-tail` 仍未授权。**⚠️ 当前工作树有 15 个修改文件（+2214/−59）+ 14 个新文件未提交**（用户 2026-09-26 拍板"先不提交"），`tools/hexmlir/hexagon-mlir-local.patch` 尚未随之重生成。

---

## 6. 后续 agent 执行合同

下面的顺序是实现顺序，不是建议列表。每一项完成后才能进入下一项；同一阶段内可并行的任务必须拥有互不重叠的文件范围。

### 6.1 开始前的固定基线

1. 记录 `git rev-parse HEAD`、`git status --short`、`triton`/`triton_shared` 子仓库版本、`libtriton.so` 和 `libhmxapi.a` 指纹。
2. 运行一次 host lit、Python smoke tests，并保存失败清单；不得把当前失败误归因于新改动。
3. 用源码确认实际 pipeline：
   - 前端：`qcom_hexagon_backend/backend/compiler.py`；
   - 主 pipeline：`qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp`；
   - HMX 归属：`MatmulToHmxPass.cpp`；
   - tile/staging：`HmxPartitionPass.cpp`；
   - resident：`WeightResidentPass.cpp`、`HmxWorkspaceResidentPass.cpp`；
   - runtime lowering：`HmxToLLVMPass.cpp`、`HexagonMemToLLVMPass.cpp`；
   - host contract：`triton_qcom_hexagon_backend_api.cc`、`hmx_weight_prepack.py`、`triton_hexagon_launcher.py`。
4. 为本轮建立 `docs/state/STATE-OF-PLAY.md`，只记录已验证事实、证据路径、未决问题和触发条件。该文件建成后，ROADMAP 的状态引用才能恢复为权威引用。

### 6.2 每个工作项的实现模板

每项必须按以下顺序提交，避免“先写代码再猜测验收”：

1. **契约**：写输入 IR、输出 IR、属性/metadata、错误/回退语义和缓存影响。
2. **决策点**：列出所有拒绝原因及其优先级；每个原因只能有一个 canonical code，文本诊断从 code 渲染。
3. **源码改动**：优先复用现有 `HmxTarget`、VTCM 记账、weight-prepack 和 launcher 机制；禁止复制一套并行规则。
4. **静态测试**：先加最小 FileCheck/验证器测试，再改 pass；覆盖成功、拒绝、预算边界、dtype、布局和错误路径。
5. **host 测试**：覆盖 metadata/manifest、cache key、launcher 参数排列、prepack canary 和 fallback。
6. **设备测试**：同构建、双指纹、固定线程/频率/输入；至少 warmup + 重复采样，按 `max(3×CV, 15%)` 判定。
7. **审查记录**：记录变更文件、删除的旧逻辑、测试命令、设备结果、未测试项和回滚方式；更新 `STATE-OF-PLAY.md` 与本表状态。

### 6.3 依赖图和并行边界

```text
P0.1 基线与状态文档
  ├─ P0.2 manifest/诊断契约 ──┐
  ├─ P0.3 Triton 支持矩阵 ────┼─ P0.4 清理与退役
  └─ P0.5 BufferManager/布局记档 ┘
                 │
                 ▼
P1.1 shape/tail ABI 设计 → P1.2 tail IR/runtime → P1.3 dynamic specialization
                 ├──────────────► P1.4 dtype contract
                 └──────────────► P1.5 full-kernel VTCM accounting
                                  │
                                  ▼
                         P2.1 unified cost model
                                  ├─ P2.2 launch-tax amortization
                                  └─ P2.3 HVX×HMX scheduling
                                  │
                                  ▼
                         P3.1 reduction pipeline
                                  → P3.2 whitelist expansion
                                  → P3.3 attention block fusion
```

允许并行：P0.2 与 P0.3；P1.4 与 P1.5；P2.2 与 P2.3。禁止并行：manifest 与 cost model、shape ABI 与 tail implementation、tail implementation 与 dynamic specialization、VTCM accounting 与任何依赖其预算的 cost model。

### 6.4 统一验收门

- **正确性**：编译期 verifier、FileCheck、Python reference 对比和设备结果全部通过；f32 输入走 HMX 时使用相对误差预算，不宣称 bit-exact。
- **回退**：任何 HMX 不可证明安全的输入必须保留可执行 HVX/Linalg 路径，并在 manifest 中给出稳定原因码。
- **可观测**：manifest、remark、launcher metadata 和 runtime 日志中的 shape/dtype/VTCM/路径字段互相一致。
- **性能**：只比较同一构建和同一输入；报告中位数、CV、线程数、warmup、重复次数和完整环境指纹。
- **回滚**：每项改动可由单个 commit 或明确的文件集合撤销；不得依赖未提交的生成物或本地环境变量。

### 6.5 当前 Roadmap 的勘误

1. `docs/state/STATE-OF-PLAY.md` 原先被引用但未提交，已改为先建立状态文档。
2. M2 的“任意 shape 四路”原先跳过了 shape ABI 和 tail IR 依赖，已拆成设计、tail、specialization 三步。
3. “清理后 lit 必须全绿”改为先记录基线、分类既存失败、为临时门设置退出条件，避免把未知失败伪装成完成。
4. `hmx.stage/await` 已存在于 HMX dialect、partition pass 和对应 FileCheck 测试中；P3 可以复用该机制，但仍需为 reduction/normalization 定义新的数据依赖和正确性契约。
5. 当前 `compiler.py` 仍明确标注 pass pipeline 动态化为 TODO；因此支持矩阵和 cost model 应先提供外部诊断/manifest，再逐步把 pipeline 选择从硬编码迁移出去。
6. **2026-09-26 两处旧结论作废**：① 曾被当作"两个独立既有缺陷"的
   **free-cache 计数自相矛盾**与 **frame `cleared owner with bound events` 拒真 capture**，
   实为**同一根因**（runtime bitcode 构建缺头文件依赖 ⇒ `AccountingSnapshot` 跨 TU 两种布局）；
   构建修好后两症状同时消失，**不要**再按"记账语义/设计规则问题"重推。② 真根因是
   **裸相对 depfile 路径**（ninja 按 build root 解析 `DEPFILE` ⇒ `deps not found` ⇒ 丢掉全部头依赖），
   `-MMD -MF`+`DEPFILE` 只是必要条件；不变式由 `bin/runtime/test/test_runtime_depfile_contract.py` 守。
   另：`site_join.py` 原先无条件拒绝一切含 truncation 记录的 transcript，属 fail-closed 规则的**过度拒绝**，已按窗口收窄。
7. **2026-09-26 两条已拍板、不再作为待办**（详见 `docs/state/STATE-OF-PLAY.md` §6 决策 13/14/15）：
   `requireAllocationResult` 在真实 VTCM OOM 时 **abort 整个 PD** 为最终生产语义；
   v3 record marker 位于所有 cache-key 输入下游 ⇒ **v3 只能经 direct backend binding / MLIR fixture 到达**，
   普通 Triton 编译永远进不去，此为**永久设计边界**（cacheable 路径必须 loud reject），不再投入使其可达。

---

## 附 · 评审勘误（2026-09-23，本 fork 事实更新）

1. 评审时"工作树不干净、仍有未提交的 crash-triage 改动"→ **P0.2 已解决**：`b947063` 之后又有 P0 manifest commit `4410d12`；当前 P0.3 仍按用户要求未提交，且已完成独立复审。
2. 评审引"184 个 lit"→ 当前 P0.3 full manual lit 为 **198（197 过 / 0 失败 / 1 `REQUIRES` skip）**。
3. 上游 README 的 "Matrix Processing (experimental) via HexKL" → 本 fork 主线是 **`hmx` 方言**；hexkl 路径依 ADR-001 保持 inert。
4. FlashAttention 39.4 → 18.1 ms 为评审引述，未本轮复测；本页表内数字以我方同构建 A/B 记录为准。

---

*上游项目介绍与文档（User Guide / Tutorials / Developer Guide / FAQ）见 [README.md](README.md)。*
