# Hexagon-MLIR — HMX 后端路线图（Roadmap）

> 本文是本 fork（`hmx` 分支）的路线图：现状定位与后续工作方向。上游 [README.md](README.md) 保持原样。
> agent 修改/测试规则、以及「我想知道 X → 读哪里」的路由表见工作区的会话级 `AGENTS.md`（§0）。
>
> **状态：现行。**
>
> ⚠️ **本文件里最权威的两处，不要跳过**：
> **① §2.1 = 加常量 / 加门 / 加特判的唯一登记处**（owner + 退出条件 + 代码出处 + A/B 证据，见该节开头的维护规则）；
> **② §1.1 = 与手写对照的当前口径**（比值一律 `我们/llama` + 中文方向；旧的 `1.13/0.92/0.58` 与 `1.14/0.89/0.58` **已作废**）。
>
> ⚠️ **本文件不写派生数字**（字段总数、行号、hunk 数、门的结果）——只写**复算命令**，理由见 §5。
> ⚠️ **"某旋钮默认开还是关"只认代码** `qcom_hexagon_backend/backend/hexagon_options.py`；本文件里的默认值是快照。
>
> 📌 **指针层**（只回答"去哪读"）：每个旋钮 / 门现在什么状态、能不能动 → [`docs/state/KNOBS.md`](../docs/state/KNOBS.md)；
> 结论能不能引用 / 被谁推翻了 → [`docs/state/CLAIMS.md`](../docs/state/CLAIMS.md)；
> 性能现状 → [`docs/state/PERF.md`](../docs/state/PERF.md)；术语 → [`docs/GLOSSARY.md`](../docs/GLOSSARY.md)。
> **§2.1 的治理登记权不受影响**：owner / 退出条件只在这里记，`KNOBS.md` 只做指针。
> **本文件是内容层**，不遵守指针层的形状规则（`AGENTS.md §6.1` 第 4/5 条）。

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
       → hmx-workspace-resident（默认 ON）→ hexagonmem（space-1 → runtime VTCM 池）
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
| 稳态 vs llama.cpp 手写（**比值一律写 `我们/llama`**） | **当前权威口径（2026-10-04 收敛后）见下方「与手写对照」。** 旧的 `1.13/0.92/0.58`（ROADMAP 与 `docs/README.md` 的 `1.14/0.89/0.58`）**全部作废** |
| FlashAttention | 稳态 **32.4 → 17.4 ms（1.86×，f32 激活 ABI）**；仓库 FA 测试 **18312 → 8184 µs**（NUM_THREADS 4→1）；评审引述 39.4 → 18.1 ms 未复测 |
| 常用算子 | softmax/rms_norm 等见 `docs/results/op-steady-state-2026-09-21.md`。⚠️ 原文这里还写着 `vec_add` 快手写 4.8× / `matmul` 1.15×，**两个都留作历史快照**：① 它们不写"我们/llama"方向（违反本节硬规则）；② `matmul` 那条与 §1.1 的 S1 口径**对不上**（这里 1.15× 像"我们慢 15%"，那里默认面 S1 是慢 16%、组合面是快 1.41×）⇒ **未经复核不得引用**，复核前请用 §1.1 |
| host 门 | **本页不复述数字，只给复算命令**（lit 全量 = `bash tools/hexmlir/run_lit_all.sh`；环境/设备/构建健康 = `tools/run_tests.sh doctor`；路由表见 `AGENTS.md §0`）。⚠️ 「把数字集中写在一处」这个做法本身已被判为**单点失效**（那一份快照表自己标了"是某轮的快照"）⇒ **只留命令** |
| 仓库状态 | **本页不复述 HEAD、工作树形状与 patch 计数**——理由见 §5 补丁条的"为什么不能复述"。事实只有三处：`git rev-parse HEAD`、`git status --short`、`tools/run_tests.sh doctor` 的 `local patch matches branch diff vs main` 那一行。**本页只写要求，不写读数**：① 工作树除未跟踪 `logs/` 外干净；② 四份 stored patch 与 `git diff main` **逐字节**一致（不是 hunk 数相等就算）；③ 三份按用途之和 == 全量（无损） |

### 1.1 与 llama.cpp 手写对照（当前权威口径，2026-10-04）

**比值写法（本页硬规则）**：一律写 **`我们/llama`** 并配中文方向（"我们慢 19~21%" / "我们快 1.20×"）。
⚠️ **`docs/results/gap-table-2026-10-04.md` 的散文原写"S1 慢 16%"，与它表内两数算出的 19~21% 冲突** ——
差异来自分母用了我们自己（`(61.5−51.42)/61.5 = 16%`）⇒「llama 比我们快 16%」成立，
「**我们比 llama 慢 16%**」不成立。**这是"分母反转"坑的第三次出现，这次伪装成百分比。**
`llama/我们` 这种分母反转的写法是 2026-10-01 读反三次的根源（见下方勘误块）。

llama 侧分母 = **51.42 / 53.95 / 14.34 µs**（同二进制 3 rep 中位数，rep 间 <1%）。

| 形状 | llama µs | 我们·零旋钮默认 | 比（我们/llama） | 我们·组合旋钮 | 比（我们/llama） |
|---|---:|---:|---|---:|---|
| S1 1024×512×64 | 51.42 | 61-62 | **慢 1.19–1.21×（19~21%）** ⚠️ | 36-37 | **快 1.39–1.43×** |
| S2 256×64×2048 | 53.95 | 45 | **快 1.20×** | 35 | **快 1.54×** |
| S3 128×128×128 | 14.34 | 14 | **快 1.03×** | 6 | **快 2.39×** |

- 出处：`docs/results/gap-table-2026-10-04.md §3`。
- **结论先行**：**"随手给一个 matmul"（零旋钮默认）在 S1 上还慢 1.19–1.21×（19~21%）**；把旋钮收敛成 auto 之后
  零旋钮默认面变成 **S1 38 / S2 32 / S3 3**，三个形状**全部快过手写**
  （`docs/results/auto-convergence-2026-10-04.md`）。

> **📌 上表"我们"两列已被本次取代（2026-10-08，Phase 0.1 冻结构建重测）**
>
> 冻结构建 `libtriton.so 78865e23309d282c41c974462ef9ded8` / `libhmxapi.a f42384f2c7e176e6a0dbb538bdffcb78`，
> 同一仪器（`s23_sweep.py` 的 def/g4dw 臂，iters=1000/20000，逐 launch 前后指纹一致）重测：
>
> | 形状 | 零旋钮 def | 组合 g4dw | 锚点 base 口径 |
> |---|---:|---:|---:|
> | S1 1024×512×64 | **38 / 37**（N=1000/20000） | **35 / 34** | 37 |
> | S2 256×64×2048 | **32 / 33** | **31 / 34** | 32–34 |
> | S3 128×128×128 | **3 / 3** | **6 / 6** | 3 |
>
> 零旋钮面与 2026-10-04 终验**逐位一致**（默认面未再漂）。出处：`docs/results/phase01-anchor-2026-10-08.md`
> （原始 JSON：`logs/phase01-anchor-2026-10-08/`）。
> ⛔ **llama 列仍是 2026-09 的数 ⇒ 上表两个"比（我们/llama）"列全部待 llama 侧同窗口重测，
> 在那之前不得引用任何方向判词**（llama 侧重测是另一张卡）。


> **⚠️ 勘误（2026-10-06）：本行原先是一次"裁决"，现整体作废。**
>
> **原写**：「2026-10-01 已按权威源 `docs/state/STATE-OF-PLAY.md:559` 判定本行为准、
> `docs/README.md` 为错并更正」。**该裁决不成立**，两条独立理由：
>
> 1. **它没有执行。** 被判为"错"的 `docs/README.md:38` **至今仍是 59.5 / 52.25 / 15.41
>    与 `1.14× / 0.89× / 0.58×`**，一个字都没改。一次"已更正"的裁决留下了一个未改的
>    被裁决方，等于没有裁决。
> 2. **它援引的权威源不含被裁决的内容。** 我复核了 `docs/state/STATE-OF-PLAY.md:559`：
>    那一行讲的是 `drop-encodings` 默认 true ⇒ `verifyCroutonDirection` 恒在开头
>    `if (!encoding) return success();` 处早退成死校验（`FastInversePass` 那条在 `:556`）
>    ——**与 S1/S2/S3 的比值毫无关系**。援引一个谈另一件事的行当依据，是
>    `§2.1a` 里点名的失败模式（"能解析、但指错地方的指针，比没有指针更阴"）。
>
> **⇒ 本处原为一次裁决，因援引的权威源不含被裁决内容而作废。**
> 替代口径就是上面的表，证据是 `docs/results/gap-table-2026-10-04.md §3`。
> 旧口径 `1.13/0.92/0.58` 与 `1.14/0.89/0.58` **一律作废，不得再被引用**。
> ⚠️ `docs/README.md:38` 那份也仍是旧值——**它需要单独的一次勘误**，本页无权改它
> （见 §6.5 勘误 8）。

---

## 2. 为什么还不能"随便写 Triton 就自动高效"（五条边界）

1. **HMX 适用面窄**：rank-2、静态、M/N/K 32 对齐、f16/f32、accumulator 可证明为空、VTCM 放得下（`MatmulToHmxPass` 合法性谓词 + `HmxTarget` 能力表）。非对齐 / 动态 shape / 非标 contraction / int8·bf16·fp8 / 复杂 mask·padding / 多 matmul 预算不足 ⇒ **编译成功但退回 HVX/Linalg，性能不等于 HMX**。且 f32 经 pack 量化到 f16、读出再 widen——是**相对误差预算语义，不是 bit-exact fp32**。
2. **融合是保守白名单**：仅 all-parallel、单入单出、全 f16、add/sub/mul/div、常量 splat、单一使用者；reduction / mask·cmpi·select / exp·log·sqrt / 类型转换 / max·min / 多用户 producer 一律不进（有设备测量依据：mask/phi/exp 链搬进 crouton 序可能慢 ~10×）。⇒ FA 中只有 QK·PV 两个 dot 有机会走 HMX，softmax 链仍是断点，做不到整块 attention 融合。
3. **前端仍是实验性入口**：实际走 `triton-shared-opt --triton-to-linalg-experimental`；已有 v1 文档支持矩阵（`docs/codegen/triton-support-matrix.md`），但前端自动查询、按 shape 的策略选择、cost model、HMX/HVX/HexKL 统一决策器仍未接入（`compiler.py` 自注需重构为动态 pass pipeline）。
4. **单资源 ≠ 并行扩展**：Triton grid 并行 ≠ HMX engine 并行；多线程最终在 runtime HMX lock 处串行化（线程路径已四次证伪），workspace-resident 曾要求 single-instance（grid>1 需自担风险、opt-in），**该限制 2026-10-04 已解除**（运行时按 flat program id 分槽 ⇒ 并发实例各拿各的缓冲），今天它默认 ON。
5. **成熟度**：R1（maxnum legalize）等上游 LLVM Hexagon 后端 RA bug 待修 ⇒ 默认 OFF；临时门/红测的完整清单见 **§2.1**；
   full manual lit 的红/绿读数**本页不写**——复算 `bash tools/hexmlir/run_lit_all.sh`（该脚本打印 FAIL/SKIP 与汇总、失败非零退出，可直接做门）；`docs/`、`tools/`、`AGENTS.md` 在工作区侧无版本控制。

> **本节是 owner + 退出条件的唯一登记处**（项目规则，`AGENTS.md §7.8`：
> 「加常量 / 加门 / 加特判是最后手段，且必须在这里登记 owner + 退出条件」；
> 总判据是「加一个新能力的改动文件数应是 O(1)」，成本口径见
> `docs/codegen/capability-file-tax-2026-09-30.md`——**真正的成本是与后续集中化提交的重叠**，
> 行为修复才是 O(1）。⚠️ 同条规则的告警：**删闸门不是清理，是翻默认**）。

### 2.1 红测与临时门的 inventory（M1.4 的交付物，只登记一次）

> **规则**：任何红测或临时门必须在此有 **owner + 退出条件**；没有就不许加。
> 红/绿复算：`bash tools/hexmlir/run_lit_all.sh`（本页不留读数）。

#### 2.1 本节的维护规则（**加了新东西就得照这四条走**）

1. **新增常量 / 门 / 特判，必须在本节登记四项**：
   **owner + 退出条件 + 代码出处（`file:line` 或字段名）+ A/B 证据**。
   **四项缺一不许加进树。**
2. **派生数字一律不写死**：字段总数、文件计数、行号、hunk 数、门的结果 ——
   **写成复算命令**（`tools/run_tests.sh census` / `bash tools/hexmlir/run_lit_all.sh` /
   `tools/run_tests.sh doctor` / `grep -c '^@@ ' tools/hexmlir/hexagon-mlir-local.patch`）。
   理由见本节那条"集中写在一处"已经腐烂两轮的真实教训。
   ⚠️ **"某旋钮默认什么"只认 `hexagon_options.py`**；本表里的默认值是**快照**，复核请现查代码。
3. **登记不是保守，登记错了是投毒。** 本节的每一条都会被后续 agent 当依据执行。
   ⇒ **写"默认 OFF / 必须 OFF / 不许翻"之前，先 `grep -n` 核一次代码**。
   本节历史上有一条整条写反的登记（`enableWorkspaceResident`：默认值、退出条件、
   grid>1 契约三处全反），差点让后续 agent 回滚已验证的收益 —— 已重写，教训留在那一行里。
   同源的两条：**"删闸门"不是清理，是翻默认**；**"援引一个权威源"也不是裁决**，
   除非那个源**真的包含被裁决的内容**。
4. **没测过的旋钮，退出条件写"待 A/B"**，不许写"已证无效"。
   本节有过一次**反向伪造**：`AGENTS.md` 写「实测对 `linalg.pack` 路线无效，已记录」，
   而那份记录**全库不存在**（`knob-audit-2026-09-29-revised.md §0`）。

5. **比值一律 `我们/llama` + 中文方向**（`AGENTS.md §6.1` 第 2 条）。
   `llama/我们` 这种分母反转的写法是 2026-10-01 读反三次的根源。

6. **`file:line` 只在"当次核对"的意义上有效；字段名 / 符号名才是稳定指针。**
   ⚠️ **2026-10-07 的现场证据**：本节 §2.1b 那一批 `hexagon_options.py:NNN` 在**同一天**就全错了 ——
   修 `~6.3` 那段注释时在 `:195` 往下加了 6 行，把后面所有字段行号推走 6 行
   （`enableWorkspaceResident` 196→202、`enableHmxVectorReadout` 214→220、`hmxReadoutBatch` 220→226、
   `hmxReadoutDeferredDrain` 236→242）。**刚写对的行号当场变错。**
   ⇒ **本表所有 `file:NNN` 都只能当"核对时的读数"**，复核一律
   `grep -n '<字段名或符号名>' <file>`；**只写字段名的条目才可长期依赖。** 与规则 2 同源。

7. **指针层是 `docs/state/KNOBS.md`**（指针层/内容层的划分见 `AGENTS.md §6.1` 第 4 条）。
   它只列「这个旋钮管什么 + 一行现状 + 去哪看治理信息」，
   **治理信息（owner / 退出条件 / A/B 证据）只在本节** —— 本节仍是唯一登记处，
   `KNOBS.md` 不复制退出条件全文（复制即分叉）。
   **两处不一致时以本节 + 代码为准**；`KNOBS.md` 的默认值列是**快照**，
   复核一律 `grep -n '<字段名>' qcom_hexagon_backend/backend/hexagon_options.py`。

#### 2.1a 证据指针纪律（2026-09-30 起，与 §2.1 的 inventory 同等级）

> ⚠️ 本节自己的 owner / 退出条件登记在 §2.1 表格**末尾** —— §2.1 开头那条规则要求
> 「任何红测或临时门必须在此有 owner + 退出条件」，本节不例外。

> **规则**：**能影响决策的陈述**（数字、倍数、"更快"、"已证否"、"严格更优"）必须带
> **可复算的证据指针** = 一个日志文件/目录 **或** `file:line` **或** 一条能跑出该数字的命令。
> 给不出指针的，**显式标 `[假设]`**。
>
> **"文档里写过"不算指针**，"另一份文档说"也不算 —— 必须能落到 `logs/` / `exp/` / 源码行 / 命令。
>
> **已证否的陈述不许删，只加勘误块**（保留原文 + 说明哪里错 + 正确值 + 出处），
> 写法照 `docs/hmx/dma-overlap-reconsideration-2026-09-29.md §3`。
>
> **复算命令本身也会过期** —— 给了命令不等于命令还成立。
> 已知踩坑：`docs/codegen/switch-census.md §0` 与 `docs/state/STATE-OF-PLAY.md:159`
> 各自抄了一个"字段总数"，彼此已经不一致（一个 57、一个 63），而该节自带的复算命令
> 又给出第三个值。⇒ **本页不写任何总数。字段数一律现查：`tools/run_tests.sh census`。**
> 这条踩坑的教训不是"数字写错了"，是**手抄的派生数字必然腐烂**：它已经腐烂过两轮
> （`LWPloopDepth`、`vector_length`/R1 三子旋钮/`enableSeedLayoutConversions` 各删一批）。
>
> **一个能解析、但指错地方的指针，比没有指针更阴** —— 读的人会以为核对过了。
> 实例：审计文档曾把"7000×"的出处指成给出 18× 的那一行。
>
> 扫描器 `tools/hexmlir/scan_claims.py`（**刻意不以非零退出**：有无指针是待分诊的债，
> 不是构建失败；挂成硬门只会在它变吵的那一刻变得不可行动）。
> 审计记录与欠账清单：`docs/codegen/claim-audit-2026-09-29.md`。

| 项 | 现状 | owner | 退出条件 |
|---|---|---|---|
| ⚠️ **`test_flash_attention.py` 里那行 `tl.trans` 是承重的**（2026-10-01 实测） | 旧的 TODO 说它是绕开 "block ptr transpose creation failure"——**该说法已过期**：pinned Triton 3.7.0 @ `a9ced83` 下**转置的 K block pointer 完全能降**（无 `tt.trans`、无 `linalg.transpose`，只是一个 `strides: [1, 64]` 的列主序视图）。真正的阻塞在**本仓**：`hmx.pack_weight` 的源契约只收行主序，列主序 K 视图被 `HmxOps.cpp` 的 `verifyPackSource` 拒。🚨 **静默错值陷阱**：**只删 `tl.trans` 这一行、不动 block pointer，会零诊断通过并安静地算成 `qkᵀ`**——它的 ttsharedir 与正确的转置写法**只差两个步长值**（该为 `[1,64]` 却是 `[64,1]`），其余完全相同；**只因为 `BLOCK_N == BLOCK_DMODEL == 64` 才通过类型检查**。⇒ 已在该文件就地写下警告（**纯注释改动、零代码改动**）。**任何「顺手清理这行」的动作前必须先读那段注释** | 本仓（已完成） | **不用做**：这是**记录一条已知的承重项 + 堵一个坑**，不是待办。真正的修法（让 pack 叶子能读 K 连续源）见 `docs/hmx/fa-transpose-copy-design-2026-09-30.md` 附录 A2 记录的**三条已被推翻的主张**——尤其「上游免费折叠调早 24 行即可」是假的，而且强行调早更糟（门不认转置访问 ⇒ 拒收 ⇒ 整个矩阵乘掉回 HVX，参照 M<32 测量里 HVX 慢 **4.35×**） |
| `test/Dialect/Hmx/Transforms/tail-weight-resident.mlir` — **M<32 尾路径接受 weight-resident 权重**（2026-09-30 加红测钉「拒绝」；**2026-10-01 已翻转并实现**，文件从 `-reject` 改名为无后缀） | ✅ **已实现，待设备 A/B。** 根因：`findPackBridge(rhs, isWeight=true)` 找不到 `hmx.pack_weight` 就返回 null，而**权重常驻时按定义就没有 pack_weight** ⇒ `emitDiagnosticInputBridges` 把两者混为一谈。`enableWeightResident` 默认 True（`qcom_hexagon_backend/backend/hexagon_options.py:111`）⇒ **修好之前，默认配置下整条尾路径不可达**（这是本条的**根因陈述，指修复前的状态**；⚠️ 不要把它读成"今天尾路径仍不可达"——修好之后它正是**默认可达**的那条路径，本条的工作就是让它可达）。**修法（与本行原预测不同，如实记录）**：新增 `anyWeightPackWrites(rhs)` 探针区分两种 null —— **没有任何 PackWeightOp 写它 ⇒ 已常驻 ⇒ 接受**，并跳过权重侧的 4 项「行主序源」检查（dominance / `verifyPackCoverage` / `verifyDiagnosticRowMajor` / `verifyDiagnosticMatrixShape`）**且不发 `emitWeight`**；**有 PackWeightOp 但 bridge 不合规 ⇒ 仍拒绝，行为不变**。⚠️ **原预测的第 ② 处（「把 `verifyPackCoverage` 从 op 条数改成 tile 覆盖集合」）**不必要且已不做**：常驻权重一个 pack 都没有，coverage 在那里是空集而非缺口。测试用 `CHECK-DAG`×4 钉 4 个 `hmx.mma` + `acc_clear`/`acc_read`，`CHECK-NOT: hmx.pack_weight`，并用单条正则钉角块的 `n_tile=1, valid_cols=1, valid_rows=1`（整块 tile 的 unpack **合法地没有** bounds，先踩了这个坑） | 本仓 | **退出条件剩一半**：同构建双指纹设备 A/B ≥ `max(3×CV,15%)`（**需用户批窗口**）。已达成部分：lit 与 host 门当时全绿（**门怎么跑见 `AGENTS.md §0` 路由表：lit 全量 = `bash tools/hexmlir/run_lit_all.sh`，宿主单测走 pytest；本页不留读数**，当轮的数是"删 1 加 1 ⇒ 与基线同数"）、**6 个 tail 拒绝测试逐条复跑全过**（`tail-bridge-loop-reject` / `tail-external-writer-reject` / `tail-source-dominance-reject` / `tail-marker-invalid{,-partition}` / `tail-plan-{invalid-,}partition-reject`）⇒ 拒绝路径未被削弱。⚠️ 上机前须确认既有正确性证据未损：`logs/hmx/promotion-tail-final.json`（direct 95/95 + resident 1/1）。⚠️ 原行提醒的 `rowStride` 静默错值（`HmxToLLVMPass.cpp:757-771` 对 `strided<[1,64]>` 返回常量 1）**本次未碰** |
| `test/Conversion/HmxToLLVM/unpack-dst-non-contiguous-reject.mlir` — **非内连续的解包目的地响亮失败**（2026-10-01 新增） | 钉住**新增的闸**：`rowStride`（`HmxToLLVMPass.cpp`）在**静态内 stride ≠ 1** 时报 `HMX leaf requires an inner-contiguous buffer` 而不是照着声明的 stride 算。**为什么加**：改之前它把 `strided<[1,32]>`（列主序）的声明 stride(rank-2)=1 直接当 `row_stride` 传给叶子，而叶子是**行主序写** ⇒ 写到 `i+j`（即元素 `(i+j,0)`）而不是 `(i,j)`。**已实测**：改前 `rc=0`、stderr 零行、`llvm.call @hmx_unpack_acc_f16(..., %28, ...)` 而 `%28 = constant(1 : i32)`；改后同输入报错。**这比性能问题严重——它是静默错值。** **范围（为什么不夸大）**：只拦**静态**非内连续。**动态 stride 保持原样**（读不出来，仍退回 `width`），那是既有行为，改它是另一个决定。pack **源**本来就有闸（`bridgeCanExpress` `MatmulToHmxPass.cpp:251` + `verifyDiagnosticRowMajor` `HmxPartitionPass.cpp:1012/1015`），本条补的是**普通整块路径上没闸**的目的地（诊断尾路径 `:1175` 已有） | 本仓 | **退出条件：跑满一次设备窗口的 FA/矩阵乘后确认没有新的拒绝**。在此之前 ⚠️ **它可能把某个今天能编译（但算错）的东西变成编译不过**——那正是目的，但要有人知道。⚠️ 反向守卫同文件第二个函数：行主序 `strided<[64,1]>`（stride(0)>width，N-split 存储）**必须继续通过**，否则就把 N-split 存储一起杀了 |

| `qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vector_size.mlir` — **唯一的 lit skip** | `// REQUIRES: do-not-run-because-flaky-test-that-needs-being-investigated`。**上游自带**：`git diff main -- <file>` 为空，`git log main -- <file>` 只有上游的初始 commit，本地从未改过 ⇒ **本 fork 无 owner**。manual runner 不解析 `REQUIRES`，按**文件**记为 skip | **无（上游）** | 接受现状。触发条件：上游把它恢复为可跑且通过 ⇒ 删掉 `REQUIRES` 行并把本行移出 inventory；本 fork 若要在该文件上做工作 ⇒ **先给上游开 issue 并在此登记 owner**，不得单方面删 `REQUIRES` |
| `isCheapF16Elementwise` / `chainIsCheapF16Only`（`MatmulToHmxPass.cpp:1782` / `:1913`）— **epilogue/链式融合白名单**（**2026-10-01 补登记**） | 它**不是 epilogue 融合机制**，尽管 §3 闸 4 这么读。`FoldElementwiseIntoLayout` 是 `OpRewritePattern<hmx::PackActOp>`（`:2206`）⇒ 它实现的是 **matmul→elementwise→matmul 的桥消除**；**`→ store` 的 epilogue 根本没有 `pack_act` 可锚 ⇒ `+bias` 不融合是设计，不是缺陷**。⚠️ **2026-10-01 实测：它从未产出过一个能编译的 kernel。** 仓库 lit 正例 `@cheap_scale_chain` 在**单 pass** 下确实折叠（2 `hmx.matmul` + **1** `pack_act` + **1** `unpack_acc`），但同一 IR 过 `linalg-to-llvm` 立刻 `error: 'hmx.mma' op act must be in VTCM (memory space 1)`（机制：`findActivationBridge` 返回 null → `NoRowMajorBridge` → 无 staging 环 → verifier 拒绝）。**该 lit 的 RUN 行只跑单 pass，所以这条断裂整个测试套件看不见** ⇒ 已新增 `cheap-scale-chain-folded-e2e-reject.mlir` 把它钉成红断言。⚠️ 另两条窄门：`:1939` 要求每节点 `hasOneUse`，而 **Triton 的 `linalg.generic` 恒为 in-place DPS** ⇒ **凡 ≥2 个 map 的廉价链一律被否决**（手写 MLIR 用 fresh `tensor.empty` outs 才能折）；`:1922-1945` 要求叶子是 read-out / splat 常量 / crouton 值，rank-1 bias 广播是**硬否决** | 本仓 | **退出条件 = 要么让它产出可编译的 e2e，要么删掉它。** **在它产出可编译的 e2e 之前，「白名单折叠成功」不构成 epilogue 融合可用的证据。** ⚠️ `:2159-2160` 的 "Verified on device ... folded vs unfolded" **属于 `FoldChainedPack`（`:2164`）**，**不是**这个白名单 ⇒ **它没有任何真机验证记录** |
| `enableHexKL`（`hexagon_options.py:65`）— **一个「设了就必失败」的公开选项**（**2026-10-01 补登记**） | `LinalgToLLVMPass.cpp:107-113` 的守卫**无条件**：`if (enableHexKL) { emitError(...); signalPassFailure(); }` ⇒ **`enableHexKL=True` 今天在任何输入上都不可能编译成功**。而选项面仍挂着它，注释还承诺 `use HexKL to lower matmul and convolutions` ⇒ **用户写进 `triton.Config` 就会拿到硬编译错误**。⚠️ 它的两个 lit 用例（`test_hexkl_macro_matmul.py:42,62` 显式设 `True`）**今天必失败且不在任何门里** | 本仓 | **退出条件 = 用户决定三选一**：① 从选项面删掉它（连带删两个用例）· ② 把守卫收窄到「契约确实激活时」（⚠️ 生产里契约是激活的 ⇒ **对生产用户无变化**，只影响测试与手写 memref IR）· ③ 标 `xfail` 并**加进门**。**我的推荐是 ①**：③ 单独用会把一个真缺陷固化成「预期失败」。⚠️ 无论选哪个，「两个用例今天必失败」这件事应该进门——现在没人知道。**（2026-10-10 已随票 21 关闭：用户拍板执行选项 ①——`enableHexKL` 选项、硬拒守卫、lowering 分支、HexKL dialect/pass/runtime/tests/docs/ci 全部出树，本条登记的退出条件已执行完毕）** |
| `enableDoubleBuffering`（`hexagon_options.py:61`）— **默认 OFF，但代码已写好并接线**（**2026-10-01 补登记**） | 消费点 `LinalgToLLVMPass.cpp:476,491`；运行时是真描述符链并返回 token（`bin/runtime/UserDMA/UserDMA.cc:49`）。⚠️ **它不是「已证否」**：`docs/analysis/gap-casual-op-…md` §6 曾把它写成「被测量关闭」，**那是错的**——`dma-overlap-reconsideration-2026-09-29.md §1` 明写「**没有一条覆盖『DMA 引擎做搬运、与之重叠』这个命题**」，§4.2 明写「**要实现的东西已经写好并接了线，门一直关着**」。⇒ **它是「从未测量」。** 标量线程那个问题**已答**（`logs/hmx/async_probe_batch3_20260919-125656_verdict.txt` 最后一行 `verdict (pcycles N=32 (B-A)/H median=0.998): SYNCHRONOUS`，读法契约见 `§3`：`B−A == H` ⇒ 同步） | 本仓 | **退出条件 = 一次设备 A/B，不是代码改动。** **前置**：① 先修引用它的那行（`gap-casual-op` 勘误 **E1** 已修）· ② 同构建双指纹（⚠️ 今晚后端重建过多次，现无可比锚点）。**在它测过之前，任何人不得引用「DMA 重叠已证否」** |
| `enableBufferization`（`hexagon_options.py:58`）— **注释只写「Used to disable for some dma testing」，文档对 HMX 一字未提**（**2026-10-01 补登记**） | ⚠️ **2026-10-01 实测更正**：它**不是静默开关**。`LinalgToLLVMPass.cpp:122-124` 在 module 仍含张量 linalg op 时 **emitError** ⇒ **Triton 的张量路径上它是编译错误**（实测 `RuntimeError`，文案 `the no-bufferization path accepts manually managed memrefs only`）。它另有 `:300` `recordOnly = !enableBufferization`，**那一条只对手写 memref IR 静默**。⇒ **它不是一个可观测性问题** ⚠️ **同时更正一处流传的说法**：「`scratch>0` 与它都静默、只修一个等于没修」**前提有误**——两者在 `LinalgToLLVMPass.cpp:302` 上确实是同一个合取，**但只有一个是静默的**（`scratch>0`；而那个覆盖是 `compiler.py:285-299` **有意为之且带理由**的，缺的只是一句 `warnings.warn`，同后端 `triton_hexagon_launcher.py:685-689` 已有完全同型先例） | 本仓 | **退出条件 = 那句 `warnings.warn` 落地**（4 行，形状见 `triton_hexagon_launcher.py:685-689`）**或**用户决定不做。⚠️ **在此之前不要把这两条当成必须捆绑处理** |
| `kStageMinKTiles = 32`（`HmxPartitionPass.cpp:184`）— **决定是否走 staged 路径的默认值** （**2026-10-01 补登记**：此前未登记，**违反本节规则**） | 它是 §8 排序里 D6 的对象，也是**一道可绕过的默认值**：`:1411-1416` 里显式 `pipeline-depth=1/2` 会**跳过它** ⇒ **它不是一道闸，是一个缺省值**。⚠️ 它自己的 traceability block（`:129-167`）已经把结论写死了：**机制本身在仓内已被证否**（`exp/hmx/leaf_bw_probe/RESULTS.md`：正确的谓词是「源是否 L2-cold」，即 activation 字节 vs L2 容量，而不是 `Kt >= 32` 这个近似）；同 block 的 `measurement` 记了 S2 1.94× / S1 flat / S3 −10%，而 `workload representativeness` **明写 NOT ESTABLISHED**（`bench_ops.py` 把 S1 称作「S1 anchor shape」是循环论证，且三者的权重只占 VTCM 的 0–3%，**没有施加这道门本来要权衡的驻留压力**）。⚠️ 今晚实测补一条**当前读数**：43 臂里**只有 2 个 arm 真的上了 staged**（K 循环 bk=1024 与整块 k=4096），其余 K 循环臂（bk=128 / bk=256 / FFN bk128）全部 `serial:shallow-k` ⇒ **在真实 FFN 形状上这道门基本总是关着的** | 本仓 | **退出条件 = 删掉它，或把机制换成 L2-cold 谓词**——**不是再调一次数字**。同 block 的原话：「在同一个机制上重调数值是白费功夫」。⚠️ 在那之前**不要**把「调 Kt 阈值」当成提速手段；⚠️ 也不要把它当闸门引用（它可被显式 `pipeline-depth` 绕过，且 43 臂实测里 41 个 arm 压根碰不到它） |
| ⚠️ **测试面缺陷**：`hmx-layout-propagation.mlir` 的 `@cheap_scale_chain` 是**绿的，但它钉住的 IR 在 e2e 编译不过** （2026-10-01 实测） | `:24` 的 RUN 行是 `-pass-pipeline='builtin.module(func.func(matmul-to-hmx))'`——**只跑那一个 pass，从不做 e2e**。单 pass 下折叠确实发生（2 `hmx.matmul` + **1** `pack_act` + **1** `unpack_acc`，桥被消掉）；换成 `linalg-to-llvm` 立刻 `error: 'hmx.mma' op act must be in VTCM (memory space 1)`。**⇒ `FoldElementwiseIntoLayout`（`MatmulToHmxPass.cpp:2206`）至今从未产出过一个能编译的 kernel。** ⚠️ 另：`:2159-2160` 那句 "Verified on device ... folded vs unfolded" **属于它上面的 `FoldChainedPack`（`:2164`）**，**不是**元素 wise 折叠 ⇒ **元素 wise 折叠没有任何真机验证记录**。详见 `docs/hmx/epilogue-fusion-what-actually-exists-2026-10-01.md` | 本仓 | **✅ 2026-10-01 已达成一半**：新增 `test/Dialect/Hmx/Transforms/cheap-scale-chain-folded-e2e-reject.mlir`，**把折叠后的 IR 原样签入并用 `not ... linalg-to-llvm` 钉住那条 VTCM 错误** ⇒ 断裂现在是**测试套件里的一条红断言**，不再靠后人偶然发现。**为什么签入而不加 RUN 行**：① 本项目的门是 `run_lit_manual.sh`，它**跳过需要 `%t` 的文件** ⇒ 现场算中间产物的测试会被 SKIP，**而 SKIP 的测试作为证据等于零**；② 整文件跑 e2e 会先因 `resident-prepack is limited to one function` 失败（多函数），在原测试上加 `not` 会**因错误的原因变绿，比没有更糟**。**剩下的退出条件**：有人修好 e2e 断裂时，**连同删掉这个文件**，并在提交信息里说明。 ⚠️ 在那之前「白名单折叠成功」**不构成** epilogue 融合可用的证据。**在它变红之前，不要把「白名单折叠成功」当成 epilogue 融合可用的证据。** ⚠️ 顺带记两条窄门（不是 bug，是设计/取舍）：① 锚点是 `hmx.pack_act` ⇒ **`→ store` 的 epilogue 设计上就不在射程**，`+bias` 不融合是**设计**不是缺陷；② `:1939` 要求每节点 `hasOneUse`，而 **Triton 的 `linalg.generic` 恒为 in-place DPS** ⇒ **凡 ≥2 个 map 的廉价链一律被否决**，手写 MLIR 用 fresh `tensor.empty` outs 才能折 |
| `enableMaxnumLegalize`（R1，**原"+3 个子旋钮"已作废，见下方「R1 子旋钮待办（已关闭）」**） | 默认 **OFF**（`hexagon_options.py:133`）。⚠️ **本行原文的等上游理由是错的，2026-09-30 更正**（见下方「R1/R2 前提复核」）。原写「等上游 LLVM Hexagon RA bug 修好」并把它算作已由 `tools/hexmlir/llvm-hexagon-ps-aligna.patch` 解决——**但那个 patch 只碰栈/帧对齐**：`HexagonFrameLowering`、`HexagonISelDAGToDAG`、`HexagonRegisterInfo`、`HexagonVExtract` + 3 个测试，**RA / scheduler / callingconv / prologepilog 一个都没有**。真机在 hexmem 路径 + ≥[512,128] maxnum tile 上曾 5/5 崩（`docs/hmx/fa-crash-resolved.md`）。**✅ 2026-09-30 晚：两个前置条件都已查清并通过** —— ①「等上游」是假的（见下方复核节）；② **pattern 在线**：四份真机 FA dump 都有 **33/33/17/33 处** `<32 x float>` 上的 `llvm.maxnum.v32f32`，恰好满足 `isHvxVectorMaxnum`（要 VectorType + f32/f16）。⚠️ 本行此前一度写成「pattern 可能不在线」，那是**在一份 LLIR 上 grep `maxnumf` 打错了 token**（那是 MLIR 的写法；LLIR 里是 `llvm.maxnum`），错了一整天没被发现 （`docs/codegen/r1r2-pattern-presence-2026-09-30.md §2.1`）。⚠️ **同文 §7.1 又更正了那个「在线」的口径**：`33/33/17/33` 是 **LLIR**（R1 之后才展开的形态）的静态处数；**R1 在自己流水线位置上直接看到的是 4 处**向量 `arith.maxnumf`（`tools/hexmlir/dump_ir_stage.sh` 单节观测），而 §2.1 写的「60 处」是**对 `-mlir-print-ir-after-all` 全日志 `grep -c`** 的计数伪影（同一处 op × 含它的 dump 节数）。**✅ 2026-09-30 晚：上机 A/B 已跑完（`docs/codegen/r1-ab-2026-09-30.md`）** —— 当年 5/5 崩的形状**不再复现**（9 次 R1-ON 全部 PASS，`rel` 7e-5~9e-5）；性能 −4.9%（BN=128/grid=1，3/3 轮同号、差值 CV 1.5%）、≈0（grid=2）、反慢 1.1%（BN=64），**三个配置都低于 `max(3×CV,15%)`** ⇒ **保持 OFF，记负结果**。机制：R1 只改逐元素 maxnum，**够不着行归约**（那条链末端是 `vector.reduce.fmax`），所以量级本来就小。⚠️ 「R1-ON 会破坏行归约蝶形」**已查清为探针伪影**：那是**漏了 `fast` 旗标**的 `.ll` 探针得出的；真机 LLIR 里是 `call fast <32 x float> @llvm.maxnum.v32f32`，带 `fast` 时两个 intrinsic 都落成 `vmax`（真机对象码 R1-OFF 36 条 `vmax`、0 条 `sfmax`）⇒ **该说法不成立**（`docs/hmx/maxnum-maximum-selectability-recheck-2026-09-30.md`，含两轮假结论的复算）⇒ **R1 保持 OFF 的理由只剩"收益不达标"**（本行 −4.9% < 15% 判据）。另：⚠️ 本行原写「崩不再复现」时把 6 次失败说成"都与被测旋钮无关"，**过强**——R1-ON 在 BN=64 档 2/5 失败、同配置 OFF 0/5，不显著 ⇒ 只能写「**未复现**」（分档见 `docs/codegen/r1-ab-2026-09-30.md §4`）| 本仓 | **退出条件已满足**（A/B 完成，结论=保持 OFF）。**不删旋钮**：它是设备 A/B 开关（消费者 `exp/hmx/op_bench/fa_ablate.py` 的 `FA_MAXNUM`），且删闸门会**翻默认**（`isHvxVectorMaxnum` 匹配任何 `vector<*>` 的 `arith.maxnumf`，无形状前提 ⇒ 去掉闸门 = 全量改写）。**三个子旋钮已于 2026-09-30 删除，不要重建**。下次触发：随「FA 行归约向量化」落地后重测一次，届时若仍无效则连 pass 一起删 |
| `enableVectorRowReduce`（R2，vror butterfly） | 默认 OFF。⚠️ **「生产管线零命中」已被证伪（2026-09-30）**：谓词 `matchVectorRowReduce`（`lib/Conversion/LinalgToLLVM/Common.cpp:78-129`）接受**带 reduction iterator 的 `linalg.generic`**，而 generic 路径比那次测量**新**（`Common.cpp`/`VectorRowReducePass.cpp` 是 09-24，测量是 09-22）。现测生产 FA：`vector.reduce.fmax` 调用点 **4 → 0**、`vror` **0 → 20**（`docs/hmx/row-reduce-vector-domain-plan-2026-09-29.md:52`）⇒ **它今天就命中**。**✅ 2026-09-30 晚：上机 A/B 已跑完**（`docs/codegen/r1-ab-2026-09-30.md §3.2`，BN=128/grid=1、固定 R1=OFF、3 个健康轮）：R2=OFF 均值 9769 µs、R2=ON 均值 9631 µs，**配对差 +1.4% 而差值 CV 8.2%**（只有 1/3 轮同号）⇒ **纯噪声，保持 OFF，记负结果** | 本仓 | **退出条件已满足**（A/B 完成，结论=保持 OFF）。**不删旋钮**，理由同 R1 行（它是 A/B 开关，且删闸门会**同时**翻两个默认：去掉闸门还会把 `vectorizeOpts.skipVectorRowReduce` 置 false（`LinalgToLLVMPass.cpp:423`），把那 4 个归约从 vectorizer 手里带走 ⇒ **R1/R2 是一对，不能各自单独删**）。下次触发：同 R1 行（行归约向量化落地后重测） |
| `HEXAGON_EPI_LOG` | env 门，默认关（`python/triton_qcom_hexagon_backend_api.cc`） | 本仓 | FA/rowmax 调查收口后删除 |
| `HEXAGON_PTR_LOG` | env 门，默认关；**在 LLIR 里插 `hexagon_runtime_dbg_log_ptr` 调用**，用来命名 FA 崩溃背后那个坏指针 | 本仓 | 同上。⚠️ 配对的 runtime 导出 `hexagon_runtime_dbg_log_ptr`（`bin/runtime/src/HexagonCAPI.cpp`）**自身无 env 门**（直接写 `rt_trc.txt`），但只有本门打开、LLIR 里插了调用才会被调到 |
| `HEXAGON_ASM_DUMP` / `HEXAGON_ASM_TO_OBJ` / `HEXAGON_ASM_DUMP_FILE` | env 门，默认关，**dump 汇编**（纯诊断件，与 HMX workarounds 无关） | 本仓（长期） | 无退出计划：只观测、默认关 |
| `HEXMLIR_RUNTIME_TRACE` | **编译期宏，不是 env 门**（`bin/runtime/{src/HexagonCAPI.cpp,multithreading/AsyncRuntime.cpp}` 的 `#ifdef`），默认不定义 | 本仓 | 同上 |
| `HEXMLIR_RUNTIME_DEBUG` | **CMake 选项**（`bin/runtime/CMakeLists.txt`，默认 OFF），恢复 `VTCMPool` 的 per-alloc/free 不变量扫描与日志 | 本仓（长期） | 无退出计划：默认关、行为不变，是诊断件不是 workaround |
| 已删除、不要重建的插桩 | `FA_O_OVERRIDE`（把编译出的 kernel `.o` 换成任意文件——**它在 launch 契约已强制之后替换 kernel 对象，是交付物里的活洞，已删**）、`FA_PIN_VREG`/`FA_GUARD`/`FA_PTR_LOG` env 门、barrier 插桩（emitter 与 runtime stub 均无） | — | 复活任一项都要先在本表登记 owner 与理由 |
| **§2.1a 证据指针纪律本身** | 已生效。扫描器 `tools/hexmlir/scan_claims.py` 已入库，刻意非零退出。分诊欠账见 `docs/codegen/claim-audit-2026-09-29.md §7.3` 排序短名单 | 本仓 | 无退出计划（永久规则）。触发条件：全库无指针 claim 分诊完 ⇒ 把该文档 §7 收缩为一句归档说明 |
| **过期数字 `~6.3 us per alloc/free pair`（同一处事实被抄了 3 份）** | ⚠️ **原文写「全树唯一一处」是错的**——实测 **3 处**：`lib/Dialect/Hmx/Transforms/HmxWorkspaceResidentPass.cpp`、`backend/hexagon_options.py`（那段 `~6.3` 注释）、`include/hexagon/Dialect/Hmx/Transforms/Passes.td`。真值 **1.12 / 1.51 / 1.69 µs**（`docs/hmx/m3.2-device-result-2026-09-29.md §4`）。⚠️ 另注：`hexagon_options.py` 那份还顺带引用了 `gap-table 2026-10-04` 与 "iters=1000" ⇒ **三处必须同批改，只改一处会留下两处继续说错话** | 本仓 | ✅ **已修（2026-10-07，三处同批全改）**：都改成 1.12/1.51/1.69 µs 并注明原值 `~6.3` 有误 + 依据。**同一条里的 `size-independent` 同样无证据**（三点散布 1.5×、对数取自 09-28 构建）⇒ 三处一并降级为「未经检验」，不许当事实。⚠️ 改完**必须**重生成 `python3 tools/hexmlir/split_patch.py`（末行 `total: N hunks (full patch has N hunks)`），否则 `tools/run_tests.sh doctor` 的 `local patch matches branch diff vs main` 会 FAIL |
| `enableRowReduceGroupStore`（RGS，行归约留在向量域） | **默认 OFF**（2026-09-30 落地 Step 1，**真机一步没走**）。pass 挂在 `LinalgToLLVMPass.cpp:442`，即 `:427` 向量化之后、`:444` canonicalize 之前、**`:431` AddFastMath 之前**——最后这条顺序是承重的：蝶形的 5 个折叠只有在带 `fastmath<nnan>` 时才落成 5 条 `vmax`（无旗标实测 `vmax=0` / `valign=31`）。只改写 `maxnumf`；`addf` **显式拒绝**（会把 `max(max(c0,m),c1)` 重结合成 `max(m,max(c0,c1))`，数值有变、无人要求）。host 证据：lit 全绿（**复算 `bash tools/hexmlir/run_lit_all.sh`，读数按 `AGENTS.md §0` 路由表现查**；新文件含 f32 5 步 + f16 6 步 + 4 类拒绝路径，每条带 `expected-remark`；`arith.select` 两臂按**形态**钉住，变异测试证明对调会被抓）。独立复核提出的 5 个「合法 IR 被静默改错」反例（非单位 stride / chunk offset 被忽略 / acc 来自别的张量 / 行循环 yield 别的值 / 归约结果无人消费）**现已全部改为拒绝 + remark** | 本仓 | **Step 2/3 未做**：① 机器码判据（归约区 `memw`=0、`vmax`/`vror` 计数、每 32 行一次 128 B 分组 store；`allocframe` 增量**不预注册**）② 设备 A/B：**先过 F1 门**（`nomax` 天花板若已 <15% 就不开工），生产 FA 形状、同构建交错、双指纹、`rel` 对拍。设计/施工单：`docs/hmx/row-reduce-vector-domain-design-2026-09-30.md` + `docs/hmx/row-reduce-group-store-spec-2026-09-30.md`（后者记了施工单里 3 处被代码否掉的假设） |
| **`HmxCroutonLayout.h:46` 的裸字面量 `2`** | `kCroutonBytes = kCroutonElements * 2;` —— 那个 `2` 是**元素字节数**，但**没有名字**，所以任何绑定都锚不到它。同一事实另有两个**具名**副本：`HmxTarget::croutonElemBytes`（`HmxTarget.h:248`）与 Python 的 `hmx_weight_prepack._CROUTON_ELEM_BYTES`，而 `test_crouton_size_agreement.py`（2026-09-30 加）**只能绑后面两个**。⚠️ **不要绑 `kCroutonHalf`**：它也是 2，但那是「一个 pair 的两半」，**是不同的 fact**，绑它等于断言巧合 | 本仓 | 命名它（`kCroutonElemBytes`）并让 `kCroutonBytes` 引用之，再把该测试的元素大小锚点从 `HmxTarget::croutonElemBytes` 移到布局头 ⇒ 头与 host 才真正绑上。**需要一次 C++ 改动 ⇒ 必须增量编译验证，且同批重生成 `split_patch.py`** |
| `enableWorkspaceResident`（per-launch VTCM 工作区常驻） | ✅ **默认 ON**（`qcom_hexagon_backend/backend/hexagon_options.py:202` = `True`；`.td` 同步 `include/hexagon/Conversion/LinalgToLLVM/Passes.td:216-217` = `true`），**2026-10-04 翻**。**它确实有效**：2026-09-30 晚复算（已构建的 `linalg-hexagon-opt`）OFF = 17646 B / `hexagon_runtime_{alloc,free}_1d_dsp` 出现 **8** 次；ON = 17241 B / 出现 **0** 次 / `workspace_resident` **4** 处 ⇒ **产物完全不同**（⚠️ 由此推翻 `docs/codegen/knob-fork-classification-2026-09-30.md §1.3`「该形状上门是惰性的」——**那是错的**，该门不惰性）。真机 A/B：**S1 +5.71% / S2 +15.07% / S3 +16.38%**（`PerfPcycles`，5 组交错，`retries=0`，`docs/hmx/m3.2-device-result-2026-09-29.md:43-48`，日志 `logs/m3_2_2026-09-29/ab_steady.log`）；2026-10-04 收敛轮另测 S1 −17%（叠加在 readout 拆分之上）、S3 14→6 µs（`docs/results/gap-table-2026-10-04.md`、`docs/results/auto-convergence-2026-10-04.md`）。**为什么翻默认（2026-10-04）**：常驻条目改成**按调用方 flat program id 分槽**（`hexagon_runtime_workspace_resident_v2_dsp` 第 4 参 instance，由 lowering 从尾部 program-info pack 算 `pid_X*np_Y*np_Z + pid_Y*np_Z + pid_Z`；`VtcmPool::Resident` 的 slot 是一等字段，权重恒 slot 0）⇒ **grid>1 的并发实例各拿各的缓冲，single-instance 限制解除** | 本仓 | **已退役**：默认 ON 即终态，退出条件 = 无。⚠️ **本条曾经整条写反，已按 2026-10-04 的事实重写**：原文写「默认 OFF」「无退出计划：默认 OFF 是终态」「⛔ 翻默认需要用户单独批准，不在 agent 工作面内」「⛔ 不许删闸门，因为 `triton_hexagon_launcher.py` 对 `prod(grid)>1` 硬 `ValueError`」。**三处都与代码相反**：`hexagon_options.py:202` 是 `True`；那个 launcher 守卫**现在只是注释**（`triton_hexagon_launcher.py:658-669`，原文自己写着「旧拒绝会挡掉所有默认配置的启动，现在什么都不需要了」）。⚠️ **为什么当时判定"不能翻"的那个理由已被机制消除**：`HmxWorkspaceResidentPass.cpp` 的匹配条件确实是"函数里有任一 Hmx 方言 op"（无形状前提、无预算前提），但**翻默认的危险从来不在匹配条件，而在 single-instance 假设**——分槽把那个假设拿掉了。设备门：`test_softmax/... grid=4` 数值 + `mha_fa grid=4` bench（`docs/results/auto-convergence-2026-10-04.md`）。⚠️ **教训（与 §2.1a 的删除拦截同源）**：**"删闸门"不是清理，是翻默认**；反过来**登记一个过期默认值也不是保守，是投毒**——本行让后续 agent 差点回滚 2026-10-04 已验证的收益 |
| `enableWeightResident`（host 预排权重） | ✅ 默认 ON（`hexagon_options.py:111` = `True`；`.td` `Passes.td:170-171` = `true`）。2026-09-21 起默认开（host 预排 ⇒ 设备侧 `pack_weight` 归零）：稳态 S1 −8.5% / S2 −19.8% / S3 −25.0%（`hexagon_options.py:110` 注释自记）。**⚠️ 有一个已知独立回归，与本行无关但必须知道**：`bench_ops` 的 `mha_fa`（hexmem OFF 的非 HMX 路径）17811 → 30857 µs，纯 HEAD 可复现（看板在查） | 本仓 | 无退出计划（默认 ON 已端到端验证过）。⚠️ **它与 `enableWorkspaceResident` 是两件独立的事**，两者都默认 ON；历史上本文件把两者混成一对矛盾条款（一个说"默认不可达"、一个说"必须默认 OFF"），**该矛盾已按代码现状消解** |
| `addFastMath` | **C++ 侧默认 `true`，无 Python 字段** ⇒ 不属于 A1 step 4 的代码工作面。位置 `LinalgToLLVMPass.cpp:430`；`.td` 默认在 `include/hexagon/Conversion/LinalgToLLVM/Passes.td:54-56`；`backend/hexagon_options.py` 里 `grep -c "^    addFastMath"` = **0**，`lib/Target/Linalg_MLLVMIR/MLLVMIRTranslation.cpp` 也不读它。**是真决策点、不是死代码**：实测 `add-fastmath=false` → 17607 B vs 默认 17646 B，`fast` 旗标 1 处 → 0 处。已由 `qcom_hexagon_backend/test/test_option_surface_agreement.py:123-126` 的 `PINNED_DEFAULTS` 钉住默认值与理由 | 本仓（长期） | 无退出计划：它是 §2 那 13 个「够不到的决策点」之一（值被 `.td` 钉死、用户够不到），钉在 `PINNED_DEFAULTS` 里就是它的登记。**触发条件**：若将来给它加 Python 字段（= 把它暴露成可调项），必须先回答「fast-math 旗标是对输入域的断言、不是提示」，并同步更新 `PINNED_DEFAULTS`（那个测试会因 `.td` 默认变化而失败） |
| `enableConvTiling` | 默认 OFF（`backend/hexagon_options.py:60`），消费点 `LinalgToLLVMPass.cpp:317`。分类表标 **UNSURED**，2026-09-30 晚把**理由从猜变成查实**（结论不变）：pass 只匹配 `linalg::Conv2DNhwcFhwcOp`（`lib/Transforms/ConvTilingPass.cpp:204`），而全仓唯一的**生产者**是 `lib/Transforms/MatmulToConvPass.cpp:79`（其余 4 处命中全是消费者），那个 pass 只在 `LinalgToLLVMPass.cpp:310` 发、被 `:309` 的 `if (enableMatmulToConv && enableSeedLayoutConversions)` 包着，**`enableMatmulToConv` 没有 Python 字段**（`Passes.td:29-31` 默认 `false`）⇒ **从本后端到不了**。实测 `enable-conv-tiling=true` 在 matmul 上产物与默认**逐字节相同** | 本仓 | **无退出计划，直到 conv 路径拿到真实流量。** 退出条件：**给 conv 路径开工时**（那时 `enableMatmulToConv` 可能变成可达，`enableConvTiling` 的 UNSURED 才需要重评）。在那之前**不许**以「产物相同」为理由删它——那只能证明这条路径不命中，不能证明收益为零。`[未验证]`：「`triton_shared` 的 TableGen 有无间接路径」只做到 grep 级（`grep -rln "conv" triton_shared/` 的命中全是 `convertLhsToF32` 之类无关词），未逐条追完 `TritonArithToLinalg` 的全部 pattern |
| `enableHVXInlining`（A/B 前不许碰） | 默认 OFF（`backend/hexagon_options.py:86`），消费点**在 `api.cc` 的 LLVM 链接期**：`python/triton_qcom_hexagon_backend_api.cc:968-969` 与 `:1064-1065`，都调 `cond_run_inliner`（`lib/Target/HEX_LLVMIR/LLVMIRTranslation.cpp:44-54`，跑 `createAlwaysInlinerLegacyPass`）。**是活的**（现行审计 §6 第 1 条已把「死字段」翻掉）。⚠️ **本组最大的「测试配置伪影」风险：15+ 处测试/脚手架显式打开它** —— `hexagon-mlir/test/python/triton/test_flash_attention.py:209`、`test_softmax.py:88`、`tools/hexmlir/dump_codegen.py:132,173`、`exp/hmx/op_bench/fa_ablate.py:368,477`、`exp/hmx/tiny_matmul/*` ⇒ **改默认会同时改掉所有基线数字** | 本仓 | **退出条件 = 跑完那一次上机 A/B，不是代码改动。** 该 A/B 已在 `docs/codegen/measurement-config-audit-2026-09-29.md` 登记、board 上标着「至今未跑」。⛔ **A/B 之前不许碰这个默认值**（理由：它同时是 15+ 处基线的输入，改它 = 同时改掉对照组）。A/B 出来之前，任何「它大概是死代码/大概是严格更优」的推断都按 `knob-audit-2026-09-29-revised.md §6` 的方法规则处理：**先 grep 全仓**（含 `python/`、`bin/`、`test/`）并打 `git show main:` 基线 |
| `enableSplitReduceGeneric`（A/B 前不许碰） | 默认 OFF（`backend/hexagon_options.py:75`），消费点 `LinalgToLLVMPass.cpp:377`。**谓词在 pass 内、不在旋钮里**（`lib/Conversion/LinalgToLLVM/SplitReduceGenericPass.cpp:79-93`）：generic 体里 yield 了 `MaxNumF/MaximumF/MinNumF/MinimumF/MaxSI/MaxUI/MinSI/MinUI` 就 `notifyMatchFailure("max/min combining reductions do not vectorise after the split")`。⇒ **删闸门本身安全**（pass 自带拒绝路径），**但删闸门 = 默认 OFF→ON = 一次行为变更**。⚠️ 那个常被引用的 **8.5×**（softmax 1089 → 128 µs，`AGENTS.md:117`、`docs/state/STATE-OF-PLAY.md:363-364`）是**「pass 拒绝拆分」换来的，不是「拆分」的收益**；不拒绝的后果记在同文件 `:69-72`（128 element f16 max：78 → 941 条指令） | 本仓 | **退出条件 = 一次覆盖 add 型与非归约 kernel 的上机 A/B，不是代码改动。** 建议臂：`rms_norm`（add 型，正收益）+ `vec_add` / `silu`（无归约，验证无害）+ `softmax`（max 型，验证 pass 的拒绝路径仍生效）。⛔ **不许先删闸门再测**——删了就再也拿不到 OFF 对照臂。⚠️ add 型的正收益目前只有**指令数**（1239 → 686，`AGENTS.md:117`），**缺上机 A/B** ⇒ 引用它时必须说清这一点 |
| `enableVectorization`（A/B 前不许碰，且**不能与 R2 分开动**） | 默认 **ON**（`backend/hexagon_options.py:76`），**两个**消费点：`LinalgToLLVMPass.cpp:388`（`HexagonTilingPass`）与 `:421`（`HexagonVectorizationPass`）。实测 `=false` 仍合法（rc=0，17013 B vs 17646 B）。⚠️ **`:421` 的 `if (enableVectorization)` 块里有 `vectorizeOpts.skipVectorRowReduce = enableVectorRowReduce;`（`:423`）** ⇒ **它与 `enableVectorRowReduce` 共用一个 option struct，删 `:421` 的闸门会同时删掉这行** ⇒ R2 的 skip 语义被静默清掉。⚠️ 顺带勘误一条被引用的数字：分类表写「实测 `=false` 仍合法、`mma`=3」——「仍合法」成立，但本形状上 `mma` **恒为 2 不变**（K=256 只切 2 个 K-tile，`mma` 数与向量化无关） | 本仓 | **退出条件 = 与 `enableVectorRowReduce` 一起做一个联合决定，不是代码改动。** 两条必须同进同退（理由见上列 `:423`）。⛔ **不许单独动 `:421`。** 由于 R2 已在上一两行结案（A/B 完成、结论=保持 OFF），本行的实际处置是：**先只做登记**（把「`:421`+`:423` 是一个不可分割单元」写进本表），联合决定等 R2 的下次触发条件（随 FA 行归约向量化落地后重测）一起做 |
| **R1/R2 两行已结案 —— step 4 不得重开** | 上两行（`enableMaxnumLegalize` / `enableVectorRowReduce`）的退出条件**已满足**：上机 A/B 跑完，结论=**保持 OFF**（`docs/codegen/r1-ab-2026-09-30.md:145-149`；R1 −4.9% / ≈0 / 反慢 1.1%，R2 +1.4% 而差值 CV 6.4%，三个配置全部低于 `max(3×CV, 15%)`） | 本仓 | **无退出计划：已结案。** ⛔ **A1 step 4（S 组）不得重开这两条。** 它们已经是「默认 OFF 的已测项」，不是「待处理项」；S 组的 27 个数字里含它们只是因为本表按**决策点**计数，不按**待办**计数。下次触发：随「FA 行归约向量化」落地后**重测一次**，届时若仍无效则连 pass 一起删 |
| **`backend/compiler.py` 的 `scratch>0` 静默改写（独立缺陷，需用户）** | `backend/compiler.py:292-299`：`scratch > 0` 时用 `dataclasses.replace` **静默改写 4 个 flag** —— `enableMultiThreading=False`、`enableConvertToHexagonmem=False`、`enableVTCMTiling=True`、`enableThreadedDispatch=True`，**用户完全不被告知**。⚠️ 其中 `enableConvertToHexagonmem=False` **正是那个会干净拒绝全部 HMX 的开关**（`reason=vtcm-allocator-disabled`，见 §1.2 / `docs/codegen/knob-fork-classification-2026-09-30.md §1.2`）。它确实进了编译 key（`backend/hexagon_options.py:187`）⇒ A/B 本身可靠，但「**用户设的值 ≠ 被编译的值**」，违反项目硬规则「不许静默回退」 | **用户**（不是本仓 agent） | **无 agent 退出计划：这是需要用户决定的契约变更，不在 A1 的工作面内。** 三个选项：(a) 改成 loud 拒绝（`scratch>0` 与用户显式设的 flag 冲突时报错）；(b) 保留静默但**在编译日志里逐条打印**改写了什么；(c) 删掉这个隐式耦合、让 `scratch>0` 不再改写任何 flag。⚠️ **本行只是登记，不代表已批准任何一项**（`AGENTS.md`：未经用户批准不得自动变更契约）。**分类表 §3 已定性它是「独立缺陷，不是冗余旋钮」** ⇒ **不许把它当成 S 组旋钮顺手改掉** |
| **HMX v3 record（`HmxRecordV3.cpp` / `HmxRecordV3.h`，marker 门控的 record-only 诊断记录）— 冻结**（2026-10-07 用户裁决） | **冻结规则：v3 的授权消费者落地之前，不加新字段、不加新序列化事实。** 背景：v2 manifest / VtcmAccounting 记账 / v3 record 三套并行记录系统里，v3 今天没有授权消费者（marker 门控、record-only；`backend/driver.py` 的 `hmx_record_diagnostic()` 只是只读诊断口，不驱动任何决策）；`HmxRecordV3.h` 自我声明 "never a converter of the serialized v2 contract" ⇒ 每个新事实要经两个 skeleton builder 各写一遍。现有字段的行为有锁（`test_hmx_record_v3.py` 钉「closed schema + grants nothing + v2 边界不变」）。冻结 = 止住平行生长，**不动任何现有字段** | 本仓（裁决：用户） | **退出条件 = 二选一**：① v3 的授权消费者落地 ⇒ 解冻，并重评三套是否收敛为「一套事实收集 + 多个 serializer 视图」（⚠️ 收敛方案违反 `HmxRecordV3.h` 的 "never a converter" 不变式 ⇒ 按项目规则需用户先拍板）；② 用户决定删除 v3。**触发条件：任何人想给 v3 加字段/加 schema 项时，先回本条**——冻结期内加字段 = 违反本节规则 1（四项缺一不许加） |
| `enableEarlyUnpack`（**T11 重排探针，临时门**，2026-10-07 登记；✅ **已执行退出条件并删除（2026-10-08）**） | **默认 OFF**（`backend/hexagon_options.py` 字段 `enableEarlyUnpack`；`.td` flag `early-unpack`（HmxPartition）与 `enable-early-unpack`（LinalgToLLVM）两级接线，`MLLVMIRTranslation.cpp` 容错读）。**做什么**：`HmxPartitionPass.cpp` 的 `emitStageLoop` 把 hoisted read-out 的发射从 tile 循环**迭代尾巴**（最后一个 `acc_read` 之后，引擎已排空、unpack 纯串行）挪到 **N 循环内 `mma` 之后、`acc_read` 之前**，读**上一个 tile 的 AR 行**——那是引擎在执行本 tile 的 mma 链、发起线程空闲的唯一窗口，正是探针要测的「引擎缝里的净增量」。`m==0` 无前 tile ⇒ `scf.if` 守卫跳过（`max(m-1,0)` 会对行 0 解包两次、其中一次读未写的 AR）；最后一行 `Mt-1` 的 read-out 发射在环后（pipeliner 之前钉好插入点，落在 peeled epilogue 之后）。OFF = **同一条发射代码路径**（`if` 只包住挪动的那一小段，OFF 臂字节同一性由构建验证）。⚠️ **与 `enableHmxVectorReadout`（默认 ON）的交互，测量前必读**：挪进 N 循环的 read-out 不再是 m-tile loop body 的直属 op ⇒ `hmx-vector-readout` 对该函数 decline、readout 留在本线程 inline ⇒ **探针 A/B 两臂都必须显式 `enableHmxVectorReadout=False`**，否则两臂同时差「线程位置」与「发射位置」两件事。`test_option_surface_agreement.py` 的 Hmx option 字段计数随本旋钮 +1（已在该测试注明「临时探针旋钮，T11 结束后删除」） | **T11 战役卡**（`roadmap/ROADMAP1001.md` §9.3 T11；细则与防复活记录 `docs/architecture/hmx-coscheduling-followups-2026-10-07.md` §2 P0 / §5） | **T11 探针测量结束即删除本旋钮与本次发射顺序分支**：Python 字段、两级 `.td` option、`emitTileCompute`/`emitStageLoop` 的 early 分支、`test_option_surface_agreement.py` 计数回落、lit 的 EARLY 臂，一并删。判决按 `ROADMAP1001.md` §5.1 的 N 规则，无论正负都落盘；**探针不付费 ⇒ W4/W5（引擎完成 token）永久砍掉**。**已执行（2026-10-08）**：判决 **NOT-PROVEN——效应 < 15% 材料性地板**（S2-class 256×64×2048，Δ̂=−1.00 µs/−3.1%，N=1000 与 N=20000 两个保守区间均排除 −4.8 µs 地板；pcyc 互证 ~−1.2 µs、11/11 session 同号——「~1 µs 改善为真」可能性高但未过门，引用须注明；判据 = `docs/analysis/criterion-paired-se-v2-2026-10-07.md`，原始数据 `logs/t11-ab-2026-10-08/`）。旋钮与发射分支已按本条删除（Python 字段 / 两级 .td / early 分支 / 计数回落 / EARLY lit 臂）；**W4/W5 按本条永久砍掉** |

#### 2.1b 2026-09/10 新增旋钮与决策常数（**2026-10-06 补登记，此前 9 项零登记 ⇒ 违反本节规则**）

> **为什么单列一小节**：下面 9 个字段**在代码里真实存在**（逐个 `grep` 核过 `hexagon_options.py`），
> 而在本表出现之前**一个都没登记** —— 按 §2.1 开头那条规则，"没有 owner + 退出条件就不许加"。
> **默认值一律现查代码**（`AGENTS.md §0` 路由表：「某旋钮默认开还是关 → 读代码，别读文档」）。
> 下表的 `file:line` 是**核对时的读数**，文档互引的行号会漂——**复核请 `grep -n` 现查**。

| 旋钮 / 常数 | 默认值（核实出处） | A/B 状态 | owner | 退出条件 |
|---|---|---|---|---|
| `enableHmxVectorReadout`（accumulator 读出搬到第二线程，按批交接） | **ON** · `hexagon_options.py:220` · `.td` `Passes.td:191-192` = `true` | ✅ **已 A/B 过并据此翻默认**：OFF→G4 在 S1 是 **−24%**（`docs/results/auto-convergence-2026-10-04.md`、`docs/results/readout-multi-matmul-2026-10-04.md`）。降不下原因 ⇒ **带 remark 拒**（FA/KDA 对象普查：readout 符号 0） | 本仓 | 无退出计划（默认 ON 已端到端验证）。⚠️ 它的前提是"pass 只能降不下能安全取的结构"——若将来谓词放宽，必须重跑一次 FA/KDA 回归 |
| `hmxReadoutBatch`（G，一次交接命名的 AR 行数） | **4** · `hexagon_options.py:220` · `.td` `Passes.td:200-201` = `"4"` | ✅ **已标定**：{1,2,4,8} 里 4 最优（G=1 输给交接成本；G=4 1.39×、G=8 1.51×，均记在 `hexagon_options.py:202-206`）。⚠️ **越界不夹取、直接报错**（`HmxVectorReadoutPass.cpp:781`），因为夹取会"看起来被满足了" | 本仓 | 无退出计划（标定集是 {1,2,4,8}，含 G=1 与更大的 G）。下次触发：换引擎/换交接机制时重标 |
| `hmxReadoutDeferredDrain`（把出口 drain 推给下一次 `configure()` 屏障，让尾部读出与函数尾巴重叠） | **OFF**（opt-in）· `hexagon_options.py:242` · `.td` `Passes.td:206-215` = `false` | ⚠️ **测过但没过判据**：单独贡献 **−6.4%**（47→44 µs，两次一致，iters=1000），低于 `max(3×CV,15%)`。**而且 grid>1 不安全**：wrapper 的 program 迭代之间没有屏障 ⇒ ring 可能溢出 trap。多 readout 循环的函数也会带 remark 拒 | 本仓 | **退出条件 = 做成 grid-safe**（wrapper 是知道 grid 的那层 ⇒ 需要 per-iteration wrapper drain），再重测是否过判据。⚠️ 在那之前它必须保持 opt-in |
| `enableL2Prefetch`（向量化切片循环里发 `llvm.hexagon.Y5.l2fetch`） | **ON** · `hexagon_options.py:101` · `.td` `Passes.td:160-166` = `true`（⚠️ 那条 `.td` 说明文字还留着翻默认前的 "Off by default until the device A/B approves"，**说明与默认不一致，属待修**） | ✅ **已 A/B 过并据此翻默认**（`docs/results/l2-prefetch-2026-10-04.md`）：平铺 DDR-bound 形状 **5.5×**、嵌套形态 1.62×、近 L2 形状 −33%、其余 7 臂中性、数值全对 | 本仓 | 无退出计划。⚠️ **它的 2 KiB 块 / 8 KiB 距离 / 每 16 向量是标定常数**，出处 `exp/hmx/streaming_bw_probe/RESULTS.md`；⚠️ **控制字布局是 load-bearing**（V79 PRM 形式 `(stride<<32)\|(width<<16)\|height`，任一子字段 0 = 取消该线程全部未决预取）——**改它之前先读那条结果文档的踩坑记录** |
| `enableHmxPipelineDepth`（`hmx-partition` staging 环深：0=auto / 1=串行不流水 / 2~3=交给 pipeliner） | **0 = auto**（代码符号 `enableHmxPipelineDepth: int = 0`，**按符号名 grep，别按行号**） | ⚠️ **`auto` 那条机制门已被判死**：门判的是**单个 `MatmulOp` 的归约深度**，而真实 kernel 切块步长是 32~512 ⇒ 真实形状永远在门下（`docs/results/t-hmx-staging-gate-dead-2026-10-02.md`）⇒ `auto` 实际退化为串行。显式 2/3 可绕过该门 | 本仓 | **退出条件 = 要么把门换成 L2-cold 谓词，要么删掉 `auto` 让显式值成为唯一入口** —— 与 `kStageMinKTiles` 那条是**同一个决定**（见 `docs/state/CLAIMS.md` 待裁决区）。⚠️ **2026-10-07 补登记：本节此前完全没有它**，违反本节规则 1（四项缺一不许加进树） |
| `hmxCroutonsPerMma`（一次 `hmx.mma` 走几个 K crouton） | **0 = 硬件上限 32** · `hexagon_options.py:184` · `.td` `Passes.td:185-190` = `"0"` | ✅ **已 A/B，判决 NOT-PROVEN**：三臂 A=32 / B=1 / N=null（`docs/hmx/ncroutons-k-fusion-2026-10-01.md §3.6`）。⚠️ K-deep 那一版已按"不达标即关闭"删除，**本项是 Option 化后重新留的 A/B 面，不是恢复 K-deep** | 本仓 | **退出条件 = 在 K 重形状（K≥1024）上重跑一次同构建 A/B**；仍不达标则连 Option 一起删（⚠️ 删 Option ≠ 删上游代码，`n_croutons ≤ 32` 的通用安全守卫**保留**）。⚠️ 越界是**报错不是夹取**（32 是硬件位域上限，夹取会"看起来被满足了"） |
| `enableThreadRolePartition`（编译期给每个 region 定线程角色并写进 manifest） | **OFF** · `hexagon_options.py:170` · `.td` `Passes.td:176-179` = `false` | ❌ **从未上机 A/B**（它**不改运行时行为**，只记决策 ⇒ 今天的"收益"恒为 0；A/B 要等它真的 outline/搬代码） | 本仓 | **退出条件 = S2（真正切分/搬代码）落地并上机 A/B**。在那之前**只做登记**，不许当提速手段引用。⚠️ 它的顺序是承重的：pass 挂在 `LinalgToLLVMPass.cpp:562`（**在 hmx-partition 之后**）——`LinalgToLLVMPass.cpp:553-561` 的注释记了为什么"看起来更自然"的顺序反而没有信息量 |
| `forceHVXCroutonization`（强制走 HVX crouton 化 pass） | **OFF** · `hexagon_options.py:254` · `.td` `Passes.td:167-169` = `false`。消费点 `LinalgToLLVMPass.cpp:355` | ❌ **从未测量**。⚠️ **勘误**：`AGENTS.md` 曾写「实测它们对 `linalg.pack` 路线无效，**已记录**」——**那份记录不存在**（`knob-audit-2026-09-29-revised.md §0` 已指出）。⇒ 它是**"接出来待 A/B"，不是"已证无效"** | 本仓 | **退出条件 = 一次覆盖 crouton 布局转换的 A/B**，或降为 pass 选项（它只是我们给**上游** `ForceHVXCroutonPass` 接出的 A/B 面；旋钮是我们加的，被门住的是上游 pass ⇒ **删旋钮 ≠ 删代码**） |
| `extendPackUpperFrontier` / `extendPackLowerFrontier`（上游 `HexagonExtendPackPass` 的两个 frontier） | **ON / ON** · `hexagon_options.py:252-253` · `.td` `Passes.td:369-384`（`upper`/`lower` 默认 `"true"`）。消费点 `LinalgToLLVMPass.cpp:147-148`（+ `:149` 的 `parallelsOnly`） | ❌ **从未测量**（同上：**"已记录无效"是假的**）。⚠️ 它们**默认开**且门住的是**上游 pass**，所以这不是"我们的默认值"，是"我们把上游的默认值暴露出来" | 本仓 | **退出条件 = 补一次 A/B，或按审计建议降为 pass 选项**。⚠️ **不许在没测之前写"无效"**——那正是本表 2026-10-01 犯过的错 |
| **决策常数 `stagedReadoutMTiles = 2 × hmxReadoutBatch`**（`auto` staging 的 readout 通道门：m-tile 数 ≥ 2×批 才放行 staging） | 由 `LinalgToLLVMPass.cpp:549-550` 计算（`enableHmxVectorReadout ? 2 * hmxReadoutBatch : 0`），消费在 `HmxPartitionPass.cpp:1899`（`Mt < stagedReadoutMTiles` ⇒ 拒绝）与 `:2267-2274`（`readoutChannelFor`） | ⚠️ **机制已测，边界本身没测**：`2` 不是自由常数（它来自"readout 的第一批无法与任何东西重叠，**少于两批纯属交接开销**"这个管线填充机制），但**边界 `2G=8` 本身未测**——`Mt=5..31` 没有形状（`docs/results/auto-convergence-2026-10-04.md:53` 原文：「**边界 2G=8 本身未测**，由填充机制定位——照实登记在 pass 注释里」）。实测只覆盖两端：S1（Mt=32）47→37、S3（Mt=4）串行赢 3→7 | 本仓 | **退出条件 = 造出 `Mt` 落在 5..31 的形状重测边界**。⚠️ 在那之前引用它只能写"机制有依据、边界未测"，不许写"标定过"。⚠️ **它不是改 `kStageMinKTiles` 的标定**（那条传输通道照旧，新通道是并集——`auto-convergence` 明确说这是为了避开"用同样锚点重标定"那个历史错误） |

#### R1/R2 前提复核（2026-09-30）——为什么上面两行被改

原两行的问题不是措辞，是**它们被当成依据引用，而内容已经不成立**：

1. **R1 的「等上游 RA 修复」是假的。** 已 cherry-pick 的
   `tools/hexmlir/llvm-hexagon-ps-aligna.patch` 逐文件核对只含
   `HexagonFrameLowering` / `HexagonISelDAGToDAG` / `HexagonRegisterInfo` /
   `HexagonVExtract` + 3 个测试——**没有 RA、scheduler、callingconv、prologepilog**。
   它修的是**栈/帧对齐**（AP 欠对齐），不是寄存器分配。
   ⚠️ 仓内还有一处**未解决的自相矛盾**：`fa-crash-resolved.md:53` 认为
   那三个"bug"很可能只是同一个 AP 欠对齐的下游症状——**若如此，这个 patch 确实修好了它**，
   只不过机制不是 RA。**没有修好之后 R1-ON 的上机 A/B 记录**，
   所以退出条件（"上游 RA 修复后上机 A/B ⇒ 改默认"）**未满足**。
   ⇒ 待办：跑那一次 A/B。它是**唯一**能同时决定 R1 和 R2 的测量。

2. **R2 的「生产管线零命中」已过期 —— 而且不只是"过期"，是设计如此。**
   谓词 `matchVectorRowReduce`（`lib/Conversion/LinalgToLLVM/Common.cpp:78`）
   的注释原文写着：
   *"The op reaches us either still as a `linalg.reduce` or, since
   LinalgGeneralize rewrites every reduce to a generic earlier in the pipeline,
   as a `linalg.generic` carrying a reduction iterator. Both are accepted."*
   ⇒ **generic 形式就是当前流水线实际喂进来的形式**，谓词就是为它写的。
   现测生产 FA：`vector.reduce.fmax` 调用点 **4 → 0**、`vror` **0 → 20**
   （`docs/hmx/row-reduce-vector-domain-plan-2026-09-29.md:52`）。
   旧测量（09-22）早于 generic 路径（09-24）。

3. **两个旋钮都不冗余 —— 删掉它们是行为变更，不是清理。**
   - R1 的 `isHvxVectorMaxnum` 匹配任何 `vector<*>` 的 `arith.maxnumf`，无形状前提；
   - R2 除匹配外还牵动 `skipVectorRowReduce`（`LinalgToLLVMPass.cpp:423`），
     而 **R1 消费 R2 的产物**（`:579-581`）⇒ 两者是一对。
   ⇒ **不许把它们当"旋钮冗余"顺手删掉。**

#### R1 子旋钮待办（**已关闭，2026-10-06**）

原文第 4 条的标题是「三个子旋钮**等 R1 定案后一并处理**」。**这条待办已关闭**：R1 早已定案
（上机 A/B 完成、结论=保持 OFF），而那三个旋钮在 **2026-09-30 就已从树上删掉**。

- **核实（`grep` 确认）**：`FA_FIXUP` / `FA_MAXNUM_N` / `FA_MAXNUM_SKIP` /
  `enableMaxnumLegalizeFixup` / `enableMaxnumLegalizeSel` / `enableMaxnumLegalizeSkip`
  在 `hexagon-mlir/` 下**零命中**，唯一的残留是 `backend/hexagon_options.py:144-155`
  那段**解释它们为何被删的注释**。`exp/hmx/op_bench/fa_ablate.py:52,54,55` 仍有
  `FA_MAXNUM*` 的说明文字（工作区侧脚手架，不在本树）。
- **删除理由（勿改写、勿"恢复"）**：① `Fixup` **是语义开关不是调试件**——关掉它发裸
  `maximumf`、**丢掉严格 maxnum 的 NaN 语义**，那是"另一个、错误的答案"，正确性坑不该
  出现在用户可见的选项面上；② `Sel`/`Skip` 是逐位点二分，服务的那个崩溃**已归因**到 LLVM
  Hexagon 后端的 AP 欠对齐（`docs/hmx/fa-crash-resolved.md` + 已 cherry-pick 的
  `tools/hexmlir/llvm-hexagon-ps-aligna.patch`），而 R1 仍欠的那个问题"值不值"是**整旋钮
  ON/OFF A/B**，不是逐点二分。
- **⚠️ 删 Python 字段时必须同时删 C++ 的 `arch_kwargs.at("…")`** ——那份 dict 由
  `HexagonOptions().__dict__` 构造，`at()` 缺键会**抛**，不是静默取默认。
  `qcom_hexagon_backend/test/test_arch_kwargs_contract.py` 就是钉这一对的。
- **剩下的**：只有 `enableMaxnumLegalize` 一个开关（默认 OFF，见 §2.1 该行），它是设备 A/B 开关，
  **不删**（删闸门 = 翻默认，见 §2.1a）。

复算命令见 `docs/codegen/knob-audit-2026-09-29-revised.md`。

---

## 3. 路线图（按优先级分四阶段）

### 阶段一 · 可观测与收口（P0 — 小步、先做）

| # | 工作项 | 内容 | 验收 |
|---|---|---|---|
| M1.1 | **决策系统 / kernel manifest**（评审差距①） | 每个 matmul 报告：是否走 HMX、为何没走、VTCM 用量、是否 blocking、pack/unpack 次数。现有 remark/warning 收敛为**结构化 manifest + 诊断接口**；先保留现有 remark/warning 作为兼容输出 | 标准批可一键打印 manifest；字段与 pass 内判定一一对应；manifest 缺失或字段不一致使测试失败 |
> **⚠️ M1.1 的实测状态（2026-10-01）——「manifest 已经有了，缺的是有人读它」**
>
> **manifest 侧已经完成**：`execution.plan` / `reason` / `pipeline.*` / `bridge_counts.*` 齐全，
> host 侧有读取器。**但 Triton 路径上没有任何东西会把它印出来。**
> 实测 43 臂（host-only，单构建逐臂盖指纹）：41 个含 `tl.dot` 的算子里
> **31 到 HMX / 8 静默掉回 HVX（19.5%）/ 2 响亮硬错误**；
> ⚠️ **勘误（`docs/state/CURRENT.md:187`）：「19.5% 静悄悄掉回」这个口径要改** —— 已归档数据里
> 掉回只有 **4.1%**，而 `pipeline selected=serial` 是 **75.7%**；且 `serial` **也带原因码**
> （例 `shallow-k`），对 `Kt=2` **可能是正确决策** ⇒ **别假定它是缺陷**。方向仍一致，
> 绝对数以那份归档表为准。**⇒ 本页那个 19.5% 只能当"当时那轮 43 臂里的计数"读，不是现状。**
> **8 个静默里 6 个在 manifest 里有精确 reason code——数据是对的，只是没人看到。**
>
> **机制（三步，均本机复核）**：`mlir/lib/IR/Diagnostics.cpp:258-262` 先遍历 handler、都没接住才走
> 「只打印 `Error`」的默认路径 ⇒ **阈值只在没有 handler 时才生效**；
> Triton 确实装了 handler（`triton/python/src/ir.cc:107`），但它是**按值返回的局部对象**、
> 基类 `ScopedDiagnosticHandler` **析构即注销**（`Diagnostics.cpp:432`），
> 只活 `:168`/`:594`/`:1976` 三处；
> 而 HMX 归属跑在 `backend/compiler.py:199` 的 `translate_linalg_to_obj` 里 —— **不在那三处**。
>
> **⛔ `MLIR_ENABLE_DIAGNOSTICS` 是诱饵，不要去调它。** 它真实存在（`ir.cc:108-131`）且看起来正是
> 那个旋钮，但它抬高的是**一个当时并未装在编译 context 上的** handler 的阈值：
> 本机四组设置（不设 / `warnings` / `remarks` / `warnings,remarks`）实测 **MLIR 诊断行数全为 0**。
> 复现：`PROBE_M=16 .venv/bin/python exp/hmx/diag_visibility/probe.py`
> （**两行输出就是全部证据**：`manifest: hvx / tile-alignment` + `MLIR diagnostic lines: 0`）。
>
> **⚠️ 本条自己的勘误 E2 已被 E12 推翻**（`docs/analysis/gap-casual-op-hmx-hvx-fusion-2026-10-01.md`）：
> E2 写「没有阈值 setter ⇒ handler 办法是死路」，**三句全错**。
> **教训**：「我们没有某个 setter」这类**否定判断必须去读被依赖那个库的源码**，
> 不能靠本仓调用点推断——MLIR 源码就在 `llvm_triton/llvm-project/` 里。
>
> **已落地的部分**：`tools/hexmlir/manifest_verdict.py`（每 matmul 一行 + 模块级
> `ALL n / ⚠️ PARTIAL only k of n / NO site on HMX`），`codegen.sh` 在 JSON 之前先打它。
> **⚠️ 它顺手暴露了一个新问题**：`vtcm-budget` 会产出**「半 HMX」kernel**
> （3 个链式 matmul，1 个被拒 2 个收下，**报告成功、三分之一在 HVX、原本无任何标记**）——
> 「我拿到 HMX 了吗」这个问题对这种 kernel 的答案是**错的**。
>
> **剩下的（需用户决定）**：把 handler 装到 `translate_linalg_to_obj` 那层（**噪声是真的**：
> 所有 Triton 编译都会开始出 warning，含与 HMX 无关的 kernel），还是只做 manifest 汇总
> （信息更少，但 `triton_hexagon_launcher.py:571` 已有先例）。详见
> `docs/codegen/why-hmx-refusals-are-invisible-2026-10-01.md`。

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

> **P1.5 promotion evidence（2026-09-26）**：commit `270f135`；**该轮当时的** manual lit `284 passed / 0 failed / 1 skipped`（285 tests）——⚠️ 这是历史快照，**活的门数字现查**（`bash tools/hexmlir/run_lit_all.sh`；此后 `09de68d` 等 commit 又增了 lit 文件，今天不是 284）。同一轮的 host/source matrix `13 pass / 14 not-proven / 0 fail`、probe/evidence host contracts `50 + 19` tests 通过，同理是快照。两次 gate-ON probe capture 均有四artifact manifest 与 remote attestation `match=true`，每次 `6 pass / 10 not-proven / 0 fail`；该轮 gate-OFF `libtriton.so=68b0a38c…`、`linalg-hexagon-opt=6b7df7c7…`、`libhmxapi.a=71c20e3c…` 上 direct tail `95/95`、resident `1/1`，一次 launch transient 恢复。独立 reviewer 已给出 **R-A/R-B scoped PROMOTE**；用户已批准启动 R-C v3 record-only migration。`observed_high_water`、`resident`、完整 allocator/full-occupancy 等未证明轴仍不得改写为 complete，production `hmx-tail`/v4 仍未授权。

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
- ~~**不把 workspace-resident 默认打开**~~（**这条已作废**：2026-10-04 翻了默认，它今天**默认 ON**。
  当时写下的 single-instance 约束已被"按 flat program id 分槽"解除，见 §2.1 该行。
  保留原文是因为它记录了**当时的理由**——而那个理由今天已不成立，不删是为了不掩盖这次反转）。

---

## 5. 一页速查

| 阶段 | 主题 | 解锁能力 | 状态 |
|---|---|---|---|
| P0 | manifest + 支持矩阵 + 记档收口 + 清理 | 可解释、可维护、门真绿 | P0.2 manifest 已完成；P0.3 支持矩阵已在 `ca679fd` 提交并 push |
| P1 | 动态 shape/tail + dtype 契约 + 大 shape 记账 | 任意 shape、更多 dtype 可用 | 部分已落（M 分块 ✅） |
| P2 | ~~cost model~~（M3.1 已退役）+ 固定税 + 共调度 | 决策 = 能力+预算（不再是"pass 顺序 + 硬阈值"）；剩余价值在**固定税**与**共调度** | M3.1 **退役**（2026-09-28，见 `docs/state/M3.1-RETIREMENT-PLAN-2026-09-28.md`）；**M3.2（削固定税）/ M3.3（共调度）是 P2 剩余、未动**；M3.4 前提未闭合 |
| P3 | reduction/attention 融合 | 从"dot 走 HMX"到"block 走 HMX" | 机制就绪（stage/await 值边 ✅），待实现 |

> **当前状态**：P0.2/P0.3 与 P1.2/P1.3c 已分别提交并 push；P1.5 evidence slice 在 `270f135`，R-C v3 record-only migration 在 `098b2a0`（`fork/hmx`）。
> **HEAD、工作树形状、patch 计数、门数字一律现查**（`git rev-parse HEAD` / `git status --short` / `tools/run_tests.sh doctor` / `bash tools/hexmlir/run_lit_all.sh`）——本页只写**要求**，理由见 §5 补丁条"为什么这一页不能写读数"。
> **门数字（manual lit / 边界矩阵 / probe host 测试 / runtime 源码契约）本页一律不复述，只给复算命令**——
> ⚠️ 曾经的做法是"集中在 `docs/state/STATE-OF-PLAY.md §4.1 写一次"，**已判为单点失效**（那一节自称单一出处，
> 却带着一份自标"某轮快照"的表，见 `AGENTS.md §2` 的现状注记）⇒ **改成只留命令**；
> 本页历史段落里出现的旧数字都是**当时那轮的快照**，不是现状。边界矩阵的格数与 `not-proven` 计数按 `docs/README.md` 的目录指引现查；**"设备证据永不改写 `declared_status`"是设计**（这一条不随轮次变，故写在这里）。
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
8. **2026-10-06：§1 那次"比值裁决"整体作废**（详见 §1.1 的勘误块）。**遗留一个跨文件的欠账**：
   `docs/README.md:36-40` 仍写着 S1 59.5 / S2 50.5 / S3 9 vs llama 52.25 / 56.71 / 15.41 与
   `1.14× / 0.89× / 0.58×`——**本页无权改它**（工作区侧文件）。
   ⇒ 需在 `docs/README.md` 就地加勘误块，指向 `docs/results/gap-table-2026-10-04.md §3`。
   **在那之前，"两套比值并存、无人裁决"这个状态仍然成立**（`docs/state/CURRENT.md:159` 记的就是它）。

---

## 附 · 评审勘误（2026-09-23，本 fork 事实更新）

1. 评审时"工作树不干净、仍有未提交的 crash-triage 改动"→ **已解决且已过期**：P0 manifest commit `4410d12` 之后，
   P0.3 也已提交（`ca679fd`）；工作树自那以后保持干净（现查 `git status --short`，见顶部表与 §5）。原文"当前 P0.3 仍按用户要求未提交"作废。
2. 评审引"184 个 lit"→ **2026-09-23 当时的** P0.3 full manual lit 为 198（197 过 / 0 失败 / 1 `REQUIRES` skip）。
   ⚠️ 这是快照，**不要引用**——本页所有 lit 数字都只是历史，活的数字现查 `bash tools/hexmlir/run_lit_all.sh`。
3. 上游 README 的 "Matrix Processing (experimental) via HexKL" → 本 fork 主线是 **`hmx` 方言**；hexkl 路径依 ADR-001 曾保持 inert（**2026-10-10 已随票 21 退役：dialect/pass/runtime/tests/docs/ci 全部出树**）。
4. FlashAttention 39.4 → 18.1 ms 为评审引述，未本轮复测；本页表内数字以我方同构建 A/B 记录为准。

---

*上游项目介绍与文档（User Guide / Tutorials / Developer Guide / FAQ）见 [README.md](README.md)。*
