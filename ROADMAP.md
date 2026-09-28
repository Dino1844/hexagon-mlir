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
| host 门 | **本页不复述数字**——lit / host / probe / 边界矩阵的实测值与其复算命令只在 `docs/state/STATE-OF-PLAY.md §4.1` 写一次，本页只引用 |
| 仓库状态 | **本页不复述 HEAD、工作树形状与 patch 计数**——理由见 §5 补丁条的"为什么不能复述"。事实只有三处：`git rev-parse HEAD`、`git status --short`、`tools/run_tests.sh doctor` 的 `local patch matches branch diff vs main` 那一行。**本页只写要求，不写读数**：① 工作树除未跟踪 `logs/` 外干净；② 四份 stored patch 与 `git diff main` **逐字节**一致（不是 hunk 数相等就算）；③ 三份按用途之和 == 全量（无损） |

---

## 2. 为什么还不能"随便写 Triton 就自动高效"（五条边界）

1. **HMX 适用面窄**：rank-2、静态、M/N/K 32 对齐、f16/f32、accumulator 可证明为空、VTCM 放得下（`MatmulToHmxPass` 合法性谓词 + `HmxTarget` 能力表）。非对齐 / 动态 shape / 非标 contraction / int8·bf16·fp8 / 复杂 mask·padding / 多 matmul 预算不足 ⇒ **编译成功但退回 HVX/Linalg，性能不等于 HMX**。且 f32 经 pack 量化到 f16、读出再 widen——是**相对误差预算语义，不是 bit-exact fp32**。
2. **融合是保守白名单**：仅 all-parallel、单入单出、全 f16、add/sub/mul/div、常量 splat、单一使用者；reduction / mask·cmpi·select / exp·log·sqrt / 类型转换 / max·min / 多用户 producer 一律不进（有设备测量依据：mask/phi/exp 链搬进 crouton 序可能慢 ~10×）。⇒ FA 中只有 QK·PV 两个 dot 有机会走 HMX，softmax 链仍是断点，做不到整块 attention 融合。
3. **前端仍是实验性入口**：实际走 `triton-shared-opt --triton-to-linalg-experimental`；已有 v1 文档支持矩阵（`docs/codegen/triton-support-matrix.md`），但前端自动查询、按 shape 的策略选择、cost model、HMX/HVX/HexKL 统一决策器仍未接入（`compiler.py` 自注需重构为动态 pass pipeline）。
4. **单资源 ≠ 并行扩展**：Triton grid 并行 ≠ HMX engine 并行；多线程最终在 runtime HMX lock 处串行化（线程路径已四次证伪），workspace-resident 明确要求 single-instance（grid>1 需自担风险，opt-in）。
5. **成熟度**：R1（maxnum legalize）等上游 LLVM Hexagon 后端 RA bug 待修 ⇒ 默认 OFF；临时门/红测的完整清单见 **§2.1**；
   full manual lit 当前 **0 红 / 1 个 skip**（活的数字与复算命令见 `docs/state/STATE-OF-PLAY.md §4.1`）；`docs/`、`tools/`、`AGENTS.md` 在工作区侧无版本控制。

### 2.1 红测与临时门的 inventory（M1.4 的交付物，只登记一次）

> **规则**：任何红测或临时门必须在此有 **owner + 退出条件**；没有就不许加。红/绿复算：`docs/state/STATE-OF-PLAY.md §4.1`。

| 项 | 现状 | owner | 退出条件 |
|---|---|---|---|
| `qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vector_size.mlir` — **唯一的 lit skip** | `// REQUIRES: do-not-run-because-flaky-test-that-needs-being-investigated`。**上游自带**：`git diff main -- <file>` 为空，`git log main -- <file>` 只有上游的初始 commit，本地从未改过 ⇒ **本 fork 无 owner**。manual runner 不解析 `REQUIRES`，按**文件**记为 skip | **无（上游）** | 接受现状。触发条件：上游把它恢复为可跑且通过 ⇒ 删掉 `REQUIRES` 行并把本行移出 inventory；本 fork 若要在该文件上做工作 ⇒ **先给上游开 issue 并在此登记 owner**，不得单方面删 `REQUIRES` |
| `enableMaxnumLegalize`（R1）+ 3 个子旋钮 | 默认 **OFF**，等上游 LLVM Hexagon RA bug 修好；真机在 hexmem 路径 + ≥[512,128] maxnum tile 上 5/5 崩（本仓侧根因与已 cherry-pick 的上游 PR #204660 见 `docs/hmx/fa-crash-resolved.md`） | 本仓（等上游） | 上游 RA 修复后上机 A/B ⇒ 改默认 |
| `enableVectorRowReduce`（R2，vror butterfly） | 默认 OFF；host 查证其 pattern 在 FA/softmax 生产管线里**零命中**（+7% 属噪声，已 revert） | 本仓 | 若将来 FA/softmax 管线出现独立行归约再评估，否则按"不达标即关闭"保持关闭 |
| `HEXAGON_EPI_LOG` | env 门，默认关（`python/triton_qcom_hexagon_backend_api.cc`） | 本仓 | FA/rowmax 调查收口后删除 |
| `HEXAGON_PTR_LOG` | env 门，默认关；**在 LLIR 里插 `hexagon_runtime_dbg_log_ptr` 调用**，用来命名 FA 崩溃背后那个坏指针 | 本仓 | 同上。⚠️ 配对的 runtime 导出 `hexagon_runtime_dbg_log_ptr`（`bin/runtime/src/HexagonCAPI.cpp`）**自身无 env 门**（直接写 `rt_trc.txt`），但只有本门打开、LLIR 里插了调用才会被调到 |
| `HEXAGON_ASM_DUMP` / `HEXAGON_ASM_TO_OBJ` / `HEXAGON_ASM_DUMP_FILE` | env 门，默认关，**dump 汇编**（纯诊断件，与 HMX workarounds 无关） | 本仓（长期） | 无退出计划：只观测、默认关 |
| `HEXMLIR_RUNTIME_TRACE` | **编译期宏，不是 env 门**（`bin/runtime/{src/HexagonCAPI.cpp,multithreading/AsyncRuntime.cpp}` 的 `#ifdef`），默认不定义 | 本仓 | 同上 |
| `HEXMLIR_RUNTIME_DEBUG` | **CMake 选项**（`bin/runtime/CMakeLists.txt`，默认 OFF），恢复 `VTCMPool` 的 per-alloc/free 不变量扫描与日志 | 本仓（长期） | 无退出计划：默认关、行为不变，是诊断件不是 workaround |
| 已删除、不要重建的插桩 | `FA_O_OVERRIDE`（把编译出的 kernel `.o` 换成任意文件——**它在 launch 契约已强制之后替换 kernel 对象，是交付物里的活洞，已删**）、`FA_PIN_VREG`/`FA_GUARD`/`FA_PTR_LOG` env 门、barrier 插桩（emitter 与 runtime stub 均无） | — | 复活任一项都要先在本表登记 owner 与理由 |

---

## 3. 路线图（按优先级分四阶段）

### 阶段一 · 可观测与收口（P0 — 小步、先做）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M1.1 | **决策系统 / kernel manifest**（评审差距①） | 每个 matmul 报告：是否走 HMX、为何没走、VTCM 用量、是否 blocking、pack/unpack 次数。现有 remark/warning 收敛为**结构化 manifest + 诊断接口**；先保留现有 remark/warning 作为兼容输出 | 标准批可一键打印 manifest；字段与 pass 内判定一一对应；manifest 缺失或字段不一致使测试失败 |
| M1.2 | **Triton 支持矩阵**（差距②） | 明确四档：可编译 / 可 HMX / 仅 HVX / 拒绝；落成文档 + 测试矩阵，替代"散落在 pass 条件里" | 矩阵每格有对应测试；新 kernel 能 5 分钟查到归宿 |
| M1.3 | **记档项按触发执行** | 先建立缺失状态文档，记录布局生命周期、BufferManager 五入口、预算归属、resident 内容校验的现状和触发条件；再按文档中的触发条件实施，不把未存在的 `STATE §7.8/§7.9` 当作已批准设计 | 每个记档项都有源码证据、设计决议、lit 锁和设备验证记录 |
| M1.4 | **清理与退役** | 先复现并分类 `return_alloc_from_loop`；只有确认是既存且与本线无关时才 XFAIL。对 `HEXAGON_EPI_LOG`、`HEXAGON_PTR_LOG`、`HEXAGON_ASM_DUMP` 等环境门逐项标明用途、默认值、移除条件 | `run_lit_all` 的结果可复现；**每个红测/临时门都在 §2.1 的 inventory 里有 owner 和退出条件** |

### 阶段二 · 形状与类型扩展（P1 — 解锁可用面）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M2.1 | **形状语义与 specialization 前置设计** | 先定义 Triton/TTIR/Linalg 中的运行时 shape、边界 mask、padding、返回布局和 launcher 参数契约；明确 compile-time specialization 与 runtime dispatch 的责任边界 | 设计文档包含 IR 示例、非法输入诊断、缓存 key 影响、正确性 oracle 和 fallback 选择表；没有该契约不得开始尾块实现 |
| M2.2 | **tail 与 padding 机制** | 在独立的 HMX tail IR 形式和 runtime ABI 上实现非 32 对齐边界；完整 tile 继续走现有 `MatmulToHmxPass`，无法证明安全时保留 Linalg/HVX | 非 32 对齐 shape sweep 正确性全过；每个尾块路径都有 FileCheck 和设备用例；无按 kernel 名称的特判 |
| M2.3 | **动态 shape specialization**（差距③） | 在 M2.1/M2.2 之后实现四路选择：完整 HMX tile / HMX+tail / HVX fallback / 混合 kernel；选择结果写入 manifest，并纳入 Triton cache key | 动态 shape sweep 正确性全过；混合路径相对纯 fallback 的性能门槛和失败回退规则写入测试；编译失败与运行时回退可区分 |
| M2.4 | **dtype 能力契约表** | 按 (dtype 对, 累加/读出位宽, 每操作数行走轴) 键化的 `HmxTarget` 契约；int8/bf16/fp8 **先 PRM/引擎能力调研再落**。**2026-09-28 调研已落**：`docs/hmx/dtype-capability-research-2026-09-28.md`——f16 有规格；fp8 在 v79 有 operand 无读出（`cvt.f8` 需 v81）；**bf16 非操作数类型**；**int8（u8×s8）是唯一 v79 可闭环且高价值的首个候选**（先做设备 go/no-go 探针） | 第二个 dtype 至少一条真机 A/B；"加能力只改一处"成立；不支持 dtype 必须有稳定诊断 |
| M2.5 | **大 shape 常态化** | `planBridge` 的 M 分块已落；补**全 kernel VTCM 记账**，覆盖 resident weight、activation ring、accumulator、status、workspace 和其他 space-1 分配的生命周期 | 预算判定对整池成立；大 shape 进入标准批；manifest 能解释每一项 VTCM 占用 |

> **P1.5 evidence slice（2026-09-26，commit `270f135`）**：内部 `hmx-vtcm-accounting` diagnostic pass 与 `hmx.diagnostic_vtcm_liveness` 已提交；它们仍只在显式 marker 下运行，不比较 VTCM budget、不写 manifest v2、不被 launcher 消费。当前可证明范围包括 canonical source identity、structured liveness、views/aliases、DPS aliases、resident provenance 与 fail-closed unsupported classes。runtime site/grid/content/full-occupancy 等轴仍显式 `not-proven`。

> **M2.5 版本决策（2026-09-25）**：v2 `hex.hmx.kernel_manifest/v2` 继续逐字保持 bridge-only；v3 是新的 record-only wire schema，禁止从 v2 自动转换，需 coordinated envelope/cache/launcher migration，但不授权 production budget、`hmx-tail` 或 grid 决策；v4 延后到 full-kernel runtime evidence 完成后，才评审 production admission/budget gating。v3 不阻断已有 bridge-budgeted `full-hmx`，但阻断依赖 full-kernel facts 的 tail/cost-model 决策。当前已有两次 gate-ON、remote-attested process-aggregate probe capture（每次 `6 pass / 10 not-proven / 0 fail`，新 cache phases 与 `RESULT: PASS`），以及当前 gate-OFF build 上 direct tail `95/95` + resident `1/1`；一般 identity/content/allocator/full-occupancy 轴仍不是 v3 admission 证据。正式契约见 [`docs/hmx/hmx-v3-manifest-decision.md`](../docs/hmx/hmx-v3-manifest-decision.md)。

> **P1.5 promotion evidence（2026-09-26）**：commit `270f135`；**该轮当时的** manual lit `284 passed / 0 failed / 1 skipped`（285 tests）——⚠️ 这是历史快照，**活的门数字只看 `docs/state/STATE-OF-PLAY.md §4.1`**（此后 `09de68d` 等 commit 又增了 lit 文件，今天不是 284）。同一轮的 host/source matrix `13 pass / 14 not-proven / 0 fail`、probe/evidence host contracts `50 + 19` tests 通过，同理是快照。两次 gate-ON probe capture 均有四artifact manifest 与 remote attestation `match=true`，每次 `6 pass / 10 not-proven / 0 fail`；该轮 gate-OFF `libtriton.so=68b0a38c…`、`linalg-hexagon-opt=6b7df7c7…`、`libhmxapi.a=71c20e3c…` 上 direct tail `95/95`、resident `1/1`，一次 launch transient 恢复。独立 reviewer 已给出 **R-A/R-B scoped PROMOTE**；用户已批准启动 R-C v3 record-only migration。`observed_high_water`、`resident`、完整 allocator/full-occupancy 等未证明轴仍不得改写为 complete，production `hmx-tail`/v4 仍未授权。

### M2.5-R：Evidence-before-manifest（**manifest/v3 这条线的主线，不是全项目唯一主线**）

**研究结论**：先冻结事实语义和证据，不先冻结 manifest wire schema。v3 只能在
identity、liveness、allocator 和 tail oracle 都闭合后生成；v4 再晚一个阶段。

> **范围（2026-09-27 修正）**：本页原先称本节为"当前唯一主线"。**该说法不成立。**
> M3.1（统一 cost model，见下）有**独立的门与独立的证据链**，两条线各自可推进、
> 互不阻塞；把 M2.5-R 写成唯一主线会让 M3.1 的产出无处安放。
> 归并视图见 `docs/state/ROADMAP-SYNTHESIS-2026-09-27.md`。

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
| M3.1 | ~~**统一 cost model**~~ → **已退役（2026-09-28）** | 不建"HMX vs HVX 选择器"：引擎选择 = **能力谓词 + VTCM 预算**（`HmxTarget::queryContraction` + `planBridge`），能上就上 HMX，否则保持 IR 不动；**0 行生产代码改动** | **出口条件作废**。代码归档 `exp/_history/`（未删除），结案/负结果见 `docs/history/hmx/m3.1-cost-model-closure-2026-09-28.md`，方案见 `docs/state/M3.1-RETIREMENT-PLAN-2026-09-28.md`。⚠️ 勘误："S1 1.60× 残差"是时钟伪影（用了没测过的 1385；真值 ≈2.08 GHz），真值 S1 闭合 1.1% |
| M3.2 | **削 per-launch 固定税**（STATE 优先级①） | bring-up ≈1.5 ms 对短工作负载致命 ⇒ 会话/op-batch 级常驻与复用 | `Perf(N)=A+B/N` 的 B 显著下降；S3 类短形状提升 > max(3×CV,15%) |
| M3.3 | **HVX×HMX 共调度** | 按 `hmx-hvx-co-scheduling` 的机制草案实现（引擎不让出线程是前提约束） | FA/混合 kernel 中两引擎重叠有实测收益才合入 |
| M3.4 | **HMX 引擎指令形状：group call 深度** | 我们的 codegen 每 crouton 发一次 `hmx_mma`（`n_croutons=1`）走**延迟**路径；参照实现（llama.cpp）沿**输入通道 K** 发 deep group 走**吞吐**路径。叶子级延迟/吞吐相差 3.25×（`n=1` 25.76 vs `n=16` 7.74 pcyc/crouton，`logs/hmx/engine-throughput-2026-09-27.log`）| ✅ **2026-09-28 结案：原"硬前提"是误标。** group 轴是 **K 不是 N**——一次 group 只产出一个 32×32 tile，"从 group 里挑 N-tile j"不存在（`docs/hmx/group-call-readout-semantics.md`，K 轴由 PRM §4.2 + 生产源码 + llama.cpp 三处固定）。深度上限 = Kt；**S1 的 Kt=2** ⇒ 端到端 9.4% 在 S1 不可实现。且 K-deep **已实现并同构建真机 A/B**（S1 −6.35% / S2 −6.04% / S3 −1.67%，均未过 `max(3×CV,15%)`）⇒ 按"不达标即关闭"删除。**若为 K 重形状（K≥1024）重做，须在当前 `libtriton.so`+`libhmxapi.a` 双指纹内重跑 A/B，验收按 K 重形状定，不得挂 S1** |

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
- **补丁**：P0.2 基线 stored patch 已校验；P0.3 commit `ca679fd` 已 push。四份 stored patch 由**同一个脚本**产出（`split_patch.py` 同时写全量 + 三份拆分，不再有手工维护的第二份），全部对 `main` 取 diff（**不是** `HEAD`——工作已 commit 在 `hmx` 分支上，裸 `git diff` 只会给出工作树增量，patch 会静默丢掉整个改动集）。
  **本页不写读数，只写要求与复算命令**，三条要求各自有命令：

  | 要求 | 复算命令 |
  |---|---|
  | 四份与 `git diff main` 逐字节一致 | `tools/run_tests.sh doctor` → `local patch matches branch diff vs main` 须为 **OK** |
  | 三份之和 == 全量（无损拆分） | `python3 tools/hexmlir/split_patch.py` 末行 `total: N hunks (full patch has N hunks)`；不等即打印 `SPLIT IS NOT LOSSLESS` 并**非零退出** |
  | hunk 数可加总 | `grep -c '^@@ ' tools/hexmlir/hexagon-mlir-local.patch` |

  > **为什么这一页不能写读数**：写在 tracked 文件里的 HEAD、hunk 数、门结果是**派生态**，而**写下它的那次 commit 本身就会让它失效**。`054b24c` 正是活样本：它把 HEAD 从 `26bbba0` 改成 `09de68d`、并断言 doctor 的 patch 行为 OK，而该 commit 自己（19:52）落在最后一次 patch 重生成（19:05）之后，断言当场变假。所以本页只留**要求**（可长期成立）与**命令**（随时可复算），读数一律现查现引。
  > ⚠️ **重生成的触发条件是"刚 commit 过"，不是"工作树脏"**——恰恰相反，commit 落地那一刻工作树是**干净**的，任何"脏了就重生成"的直觉都会漏掉这一次。动了 `hexagon-mlir/` 就重跑 `split_patch.py`；**先改文档/代码，最后一条命令再 regenerates patch**。
  > ⚠️ 三份是**按 hunk 切**的，同一文件会出现多个 `diff --git` header 块，所以 `grep -c '^diff --git'` 不是文件数；**只有 hunk 数可加总**。
  > ⚠️ `doctor` 的**总计数随环境变**（手机不可达时 `phone reachable` FAIL）——只引用 patch 那一行，不要引用 `15/0` 这类总数。
  不得为消除门红灯自动改 patch。

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
| P2 | ~~cost model~~（M3.1 已退役）+ 固定税 + 共调度 | 决策 = 能力+预算（不再是"pass 顺序 + 硬阈值"）；剩余价值在**固定税**与**共调度** | M3.1 **退役**（2026-09-28，见 `docs/state/M3.1-RETIREMENT-PLAN-2026-09-28.md`）；**M3.2（削固定税）/ M3.3（共调度）是 P2 剩余、未动**；M3.4 前提未闭合 |
| P3 | reduction/attention 融合 | 从"dot 走 HMX"到"block 走 HMX" | 机制就绪（stage/await 值边 ✅），待实现 |

> **当前状态**：P0.2/P0.3 与 P1.2/P1.3c 已分别提交并 push；P1.5 evidence slice 在 `270f135`，R-C v3 record-only migration 在 `098b2a0`（`fork/hmx`）。
> **HEAD、工作树形状、patch 计数、门数字一律现查**（`git rev-parse HEAD` / `git status --short` / `doctor` / `docs/state/STATE-OF-PLAY.md §4.1`）——本页只写**要求**，理由见 §5 补丁条"为什么这一页不能写读数"。
> **门数字（manual lit / 边界矩阵 / probe host 测试 / runtime 源码契约）一律见 `docs/state/STATE-OF-PLAY.md §4.1`，本页不复述**——
> 本页历史段落里出现的旧数字都是**当时那轮的快照**，不是现状。边界矩阵的格数与 `not-proven` 计数见 §4.1；**"设备证据永不改写 `declared_status`"是设计**（这一条不随轮次变，故写在这里）。
> 一份 remote-attested 设备 capture `logs/hmx/n123-a3-device-2026-09-26.log`（overlay `current`/`promotable`、0 validation error）导入 **13 recorded / 5 pass** 设备观测。
> R-A/R-B 已 scoped PROMOTE；v3 record-only 的**可达性**已定为永久设计边界（§6.5 勘误 7）；v4 与 production `hmx-tail` 仍未授权。

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

允许并行：P0.2 与 P0.3；P1.4 与 P1.5；P2.2 与 P2.3。禁止并行：shape ABI 与 tail implementation、tail implementation 与 dynamic specialization。（cost model 相关禁止项随 M3.1 退役于 2026-09-28 删除。）

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

1. 评审时"工作树不干净、仍有未提交的 crash-triage 改动"→ **已解决且已过期**：P0 manifest commit `4410d12` 之后，
   P0.3 也已提交（`ca679fd`）；工作树自那以后保持干净（现查 `git status --short`，见顶部表与 §5）。原文"当前 P0.3 仍按用户要求未提交"作废。
2. 评审引"184 个 lit"→ **2026-09-23 当时的** P0.3 full manual lit 为 198（197 过 / 0 失败 / 1 `REQUIRES` skip）。
   ⚠️ 这是快照，**不要引用**——本页所有 lit 数字都只是历史，活的数字见 `docs/state/STATE-OF-PLAY.md §4.1`。
3. 上游 README 的 "Matrix Processing (experimental) via HexKL" → 本 fork 主线是 **`hmx` 方言**；hexkl 路径依 ADR-001 保持 inert。
4. FlashAttention 39.4 → 18.1 ms 为评审引述，未本轮复测；本页表内数字以我方同构建 A/B 记录为准。

---

*上游项目介绍与文档（User Guide / Tutorials / Developer Guide / FAQ）见 [README.md](README.md)。*
