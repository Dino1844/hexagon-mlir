# Hexagon-MLIR — HMX 后端路线图（Roadmap）

> 本文是本 fork（`hmx` 分支）的路线图：现状定位与后续工作方向。上游 [README.md](README.md) 保持原样。
> 事实与勘误以工作区 `docs/state/STATE-OF-PLAY.md` 为唯一权威入口；
> agent 修改/测试规则见工作区 `AGENTS.md`。

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
| host 门 | 手动 lit **191：189 过 / 1 既存红（顺序敏感，与本线正交）/ 1 skip** |
| 仓库状态 | `b947063`（2026-09-23，双评审收口后提交并推送 fork `hmx`）；补丁 157 files / 267 hunks 无损 |

---

## 2. 为什么还不能"随便写 Triton 就自动高效"（五条边界）

1. **HMX 适用面窄**：rank-2、静态、M/N/K 32 对齐、f16/f32、accumulator 可证明为空、VTCM 放得下（`MatmulToHmxPass` 合法性谓词 + `HmxTarget` 能力表）。非对齐 / 动态 shape / 非标 contraction / int8·bf16·fp8 / 复杂 mask·padding / 多 matmul 预算不足 ⇒ **编译成功但退回 HVX/Linalg，性能不等于 HMX**。且 f32 经 pack 量化到 f16、读出再 widen——是**相对误差预算语义，不是 bit-exact fp32**。
2. **融合是保守白名单**：仅 all-parallel、单入单出、全 f16、add/sub/mul/div、常量 splat、单一使用者；reduction / mask·cmpi·select / exp·log·sqrt / 类型转换 / max·min / 多用户 producer 一律不进（有设备测量依据：mask/phi/exp 链搬进 crouton 序可能慢 ~10×）。⇒ FA 中只有 QK·PV 两个 dot 有机会走 HMX，softmax 链仍是断点，做不到整块 attention 融合。
3. **前端仍是实验性入口**：实际走 `triton-shared-opt --triton-to-linalg-experimental`；缺支持矩阵、不支持语法的诊断、按 shape 的策略选择、cost model、HMX/HVX/HexKL 统一决策器（`compiler.py` 自注需重构为动态 pass pipeline）。
4. **单资源 ≠ 并行扩展**：Triton grid 并行 ≠ HMX engine 并行；多线程最终在 runtime HMX lock 处串行化（线程路径已四次证伪），workspace-resident 明确要求 single-instance（grid>1 需自担风险，opt-in）。
5. **成熟度**：R1（maxnum legalize）等上游 LLVM Hexagon 后端 RA bug 待修 ⇒ 默认 OFF；triage env 门保留待退役；1 条既存 lit 红；`docs/`、`tools/`、`AGENTS.md` 在工作区侧无版本控制。

---

## 3. 路线图（按优先级分四阶段）

### 阶段一 · 可观测与收口（P0 — 小步、先做）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M1.1 | **决策系统 / kernel manifest**（评审差距①） | 每个 matmul 报告：是否走 HMX、为何没走、VTCM 用量、是否 blocking、pack/unpack 次数。现有 remark/warning 收敛为**结构化 manifest + 诊断接口** | 标准批可一键打印 manifest；字段与 pass 内判定一一对应 |
| M1.2 | **Triton 支持矩阵**（差距②） | 明确四档：可编译 / 可 HMX / 仅 HVX / 拒绝；落成文档 + 测试矩阵，替代"散落在 pass 条件里" | 矩阵每格有对应测试；新 kernel 能 5 分钟查到归宿 |
| M1.3 | **记档项按触发执行** | `STATE §7.8` 布局生命周期（`drop-encodings` 双路 + 死校验 + stride 启发式，已拍板"先不动"，触发=第二布局需求或收口宣言）；`STATE §7.9` alloc 失败链（重试耗尽点带诊断终止 vs lowering 判空，**先读全 BufferManager 五入口**）+ 预算记账收敛 + resident 内容 debug 校验 | 各自独立一轮：设计→lit 锁→可行的设备验证 |
| M1.4 | **清理与退役** | 既存红 `return_alloc_from_loop`（XFAIL 或修）；triage env 门按期退役（`HEXAGON_EPI_LOG` 自标 TEMP）；既存红线移出豁免清单 | `run_lit_all` 真·全绿；无未记档 env 门 |

### 阶段二 · 形状与类型扩展（P1 — 解锁可用面）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M2.1 | **动态 shape specialization**（差距③） | 任意 shape 分四路：完整 HMX tile / HMX+tail / HVX fallback / 混合 kernel（compile 或 runtime specialization） | 形状 sweep（含非 32 对齐）正确性全过、混合路径性能不低于纯 fallback |
| M2.2 | **tail 与 padding 机制** | 非 32 对齐边界的 HMX 尾块处理（与 M2.1 同一机制，不加 shape 特判） | 同上 |
| M2.3 | **dtype 能力契约表** | 按 (dtype 对, 累加/读出位宽, 每操作数行走轴) 键化的 `HmxTarget` 契约；int8/bf16/fp8 **先 PRM/引擎能力调研再落** | 第二个 dtype 至少一条真机 A/B；"加能力只改一处"成立 |
| M2.4 | **大 shape 常态化** | `planBridge` M 分块已落（4096² 不再整 op 拒绝）→ 补**全 kernel VTCM 记账**（归属期盲区收口）+ 大 shape 真机用例 | 预算判定对整池成立；大 shape 进标准批 |

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
- **补丁**：改工作树后 `split_patch.py` 重生成（计数 157 files / 267 hunks 无损为当前基线）。

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
| P0 | manifest + 支持矩阵 + 记档收口 + 清理 | 可解释、可维护、门真绿 | 规划中 |
| P1 | 动态 shape/tail + dtype 契约 + 大 shape 记账 | 任意 shape、更多 dtype 可用 | 部分已落（M 分块 ✅） |
| P2 | cost model + 固定税 + 共调度 | 决策不再靠 pass 顺序与硬阈值 | 规划中 |
| P3 | reduction/attention 融合 | 从"dot 走 HMX"到"block 走 HMX" | 机制就绪（stage/await 值边 ✅），待实现 |

---

## 附 · 评审勘误（2026-09-23，本 fork 事实更新）

1. 评审时"工作树不干净、仍有未提交的 crash-triage 改动"→ **已解决**：`b947063` 全部提交，且经 architecture-review + ai-slop-cleaner 双评审收口（2 Critical + 4 Major + 全部 Minor 修复后入库）。
2. 评审引"184 个 lit"→ 现为 **191（189 过 / 1 既存红 / 1 skip）**。
3. 上游 README 的 "Matrix Processing (experimental) via HexKL" → 本 fork 主线是 **`hmx` 方言**；hexkl 路径依 ADR-001 保持 inert。
4. FlashAttention 39.4 → 18.1 ms 为评审引述，未本轮复测；本页表内数字以我方同构建 A/B 记录为准。

---

*上游项目介绍与文档（User Guide / Tutorials / Developer Guide / FAQ）见 [README.md](README.md)。*
