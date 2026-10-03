# ERRATA —— ROADMAP1001 的勘误记录

**这份文档是什么**：它保存 ROADMAP1001 历史上被推翻的判断——原文照抄、推翻它的证据、以及修正现在落在 ROADMAP 哪一节。

**它不是路标。** 要读当前结论，读 `ROADMAP1001.md`；那份文档不依赖本文即可读完。

**约定**：原文一律逐字保留，不做美化。一条勘误被自己的后一条推翻时，两条都留（勘误七有七层更正，全部在第一部分）。

**组织**：勘误原本散落在 ROADMAP 各处，与正文混排。2026-10-02 全部移入本文，ROADMAP 正文改为直接陈述当前事实。移入时的分组：

- **第一部分 · 项目级测量与构建**（与 T_HMX 方案无关，但决定了本文里哪些数字可用）
- **第二部分 · T_HMX 收益链**（直接作用于 ROADMAP 的立项理由）

---

## 索引

| 编号 | 主题 | 被推翻的断言 | 现在落在 |
|---|---|---|---|
| [一](#一) | `N ≥ 1000` 门槛 | 五个形状在 N=1000 全部超标 3.8–7.5× | ROADMAP §5.1.2 |
| [二](#二) | 同上（续） | 「S2-class 1.70% ✅」 | ROADMAP §5.1.2 |
| [三](#三) | `B ≈ 2,700 µs` 的归因 | 超标是后端的性质 | ROADMAP §5.1.2 |
| [四](#四) | 「S1 受限于输出读出」 | 那个量是**输出字节** | ROADMAP §5.1.8 |
| [五](#五) | `g = 0.30 µs/Kt-tile` | 系数是 Kt 的函数 | ROADMAP §5.1.8 |
| [六](#六) | T_HMX 收益估算 | 缺一块输入（逐形状 budgetDepth） | ROADMAP §5.1.7 |
| [七](#七) | S0b 重建 / S1 回归 / f16 除法 | 七层更正，见下 | ROADMAP §5.2 |
| [八](#八) | f16 除法 194 次 libcall | 这是项目当前最大单点 | ROADMAP §5.0 |
| [九](#九) | staging 门的前提 | T_HMX 的 staging 前提成立 | ROADMAP §5.1.7 |

---

# 第一部分 · 项目级测量与构建

> 这一组与 T_HMX 方案无关，但它们决定了 ROADMAP 里哪些数字能用。
> 核心结论：**`B ≈ 2,700 µs` 是 harness 缺陷，不是后端性质**；修掉之后 ROADMAP 的 `N ≥ 1000` 门槛不再是约束。

---

## 三

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
> **机制**：`HexagonAPI::AcquireResources()`（`HexagonAPI.h:52` → `HexagonAPI.h:72`）在
> **每次 launch** 里调 `initialize_and_acquire_hmx()`（`HexagonAPI.cpp:220`），
> 它做 `HAP_power_set(HMX_v2, set_clock=TRUE, target_corner=VCORNER_MAX,
> perf_mode=CLK_PERF_HIGH)`（`HexagonAPI.cpp:231/234/235/238`）与
> **`HAP_compute_res_acquire(..., 100000)`（`HexagonAPI.cpp:261`，阻塞最长 100 ms）**。
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
> 前面所有 `N=1000` 的**稳态**读数本来就是对的，错的只是那个一次项。**
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

### 勘误三的追加修正（2026-10-02 06:45）

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
> ⇒ ⭐ **九档里九档指向 2 650~2 830；唯一给出 1 992 的那档残差最小（0.5%），
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

---

## 一 · 二

> **勘误一（2026-10-02 06:0x）：五个形状在 `N=1000` 下全部超标，`N ≥ 1000` 这条规则不成立。**
>
> **做法**：`Perf:` 随 `ITERS` 下降 ⇒ 用两点解 `t(N) = A + B/N`
> （`B=(t1−t2)/(1/N1−1/N2)`、`A=t1−B/N1`），第三点独立校验，最大偏差 **0.42 µs**。
>
> | 形状 | A 稳态 | B 一次性 | `once_share@1000` | `@20000` | **合规需 N ≥** |
> |---|---:|---:|---:|---:|---:|
> | S3-class 128×128×128 | 7.89 µs | 2105 µs | **21.05%** | 1.316% | **13 333** |
> | 256×512×64 | 16.84 | 3158 | **15.79%** | 0.929% | **9 375** |
> | 256×512×256 | 18.84 | 3158 | **14.35%** | 0.831% | **8 380** |
> | **S2-class 256×64×2048** | 38.84 | 3158 | **7.52%** | 0.405% | **4 065** |
> | S1 / 1024×512×64 | 52.89 | 2105 | **3.83%** | 0.199% | **1 990** |
>
> ⇒ ⛔ **五个形状在 N=1000 下全部超标。**
> ⇒ ⛔ **旧版 §5.1.2 那张表里唯一标 ✅ 的「S2-class 1.70%」实测 7.52%，超标 3.8 倍。**
> ⇒ ✅ **N=20000 全部合规（≤1.32%）。**
>
> **根因**：这张表给三个 class **共用同一个 `B = 1470 µs`**（三档验算吻合到 0.01%）
> ⇒ **那是假设，不是逐形状实测。**
> 而实测里 **A 缩小（优化见效）而 B 相对变大** ⇒ `B/A` 从 ~17 涨到 ~81。
>
> ⇒ ⭐ **修正后的规则：`N ≥ 50·B/A` 不是常数，每个形状类、每个构建都要重算。**
> **旧版 §5.1.2 那句「⇒ S3 用 `N ≥ 1000`」按实测应是 **`N ≥ 4065`**（S2-class）。**
>
> **⚠️ 但「比值型」结论不受影响**：同一形状同一 N 下比两臂时 `B` 是共模、会完全抵消
> （`base − ws = A_base − A_ws`）。⇒ 今晚的 `ws` 效应、`staging` 效应、
> `kStageMinKTiles=32` 钉值**都是同形状同 N 的臂间比较，依然有效**。
> ⇒ ⛔ **要重跑的是「绝对水平」类陈述，不是「效应量」类。**
>
> **⚠️ 连带发现**：`docs/README.md:37` 与 ROADMAP 的比值表把「我们」那一列和
> llama 那一列放在不同口径下比 —— llama 侧用 24576~81920 runs（`B/N` 可忽略 ⇒ 稳态），
> 我们用 N=1000（含 `B/N`）。**稳态口径下 S1 是 52.89/52.25 = 1.012×，
> 而 N=1000 口径下是 1.053×。**
> ⇒ ⚠️ **「S1 是唯一真差距（1.14×）」可能是口径伪像** ——
> 但 llama 侧仍是 2026-09 的数（本机跑不了，bench 在手机 Termux 上）⇒ **只是条件性线索，不是判决。**
>
> ⇒ 📄 `docs/results/once-share-compliance-2026-10-02.md`（引用逐条复验）
> ⇒ · `docs/results/s1-readout-bound-2026-10-02.md`
> ⇒ · 日志 `logs/shape-attribution-2026-10-02/`（5 份，指纹 md5 出现 10 次全一致）

**⇒ 一与二已被勘误三整体推翻**（超标测的是 harness 缺陷）。原文保留在此仅为记录。

---

## 七

> **🟣 勘误七（2026-10-02 13:11–13:13）：旧版 §5.0 那句「两个 patch 都只对 `build/bin/opt` 生效」
> 已过时 —— install 树已刷新，`libtriton.so` 已重链，设备锚点已作废并更换。**
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

### 勘误七之更正（14:22）：新 `.so` 在 S1 上回归，已回滚

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

### 勘误七之更正之二（15:40–16:0x）：「S1 回归」不可复现

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
> `shape_pair.py:205` 把 launch 的 stdout 吞进 buffer、`shape_pair.py:241` 的 assert 直接死掉、
> **buffer 从未打印**。
> ⇒ ⭐ **静默失败不只是「查起来麻烦」，它会让人做出错误的重大决定。**
> ⇒ ✅ 已修：同时捕获 stdout+stderr，失败时打印 `LAUNCHFAIL` / `NOPERF` 与两个流的内容。
> ⇒ 📄 `docs/results/s1-regression-not-reproducible-2026-10-02.md`

### 勘误七之更正之三（16:10）：「产物可用」这句要收窄

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
> ⇒ **必须以 9 形状复验为准**，不能以单形状读数下结论。**
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

### 勘误七之更正之四（16:25）：真正的差异是 `a576182`（K-fusion）

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

### 勘误七之更正之五（16:35）：两个构建的设备指令流逐字节相同

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

### 勘误七之更正之六（17:00–17:20）：fdiv patch 确实在 `.so` 里

> 更正之三写「fdiv patch 仍然没进 `.so`」，依据是二进制 Agent 量的 `.text` 只涨 192 B。
> **那个界不足以否定归档成员级的证据。以归档链路为准：**
>
> | 环节 | 证据 |
> |---|---|
> | 源码含 patch | `InstCombineCasts.cpp:2136/2137/2165/2178`，`NarrowFMF` 命中 **5** 处 |
> | 源文件被改 | **2026-10-02 02:30:59** |
> | `InstCombineCasts.cpp.o` 重编 | **2026-10-02 13:09:00**（在 patch 之后） |
> | `libLLVMInstCombine.a` 重打包 | **2026-10-02 13:09:00** |
> | `libtriton.so` 链的就是它 | `triton/build/…/build.ninja:363`，`LINK_LIBRARIES` 里是**绝对路径** `llvm_triton/build/install/lib/libLLVMInstCombine.a` |
> | `20342db0` 链接时刻 | **13:11:34**（归档之后 2.5 分钟） |
> | `ce26015e` 链接时刻 | **2026-10-01 14:18**（比 patch 落地早 **23 小时**） |
>
> ⇒ ⭐⭐⭐ **⇒ patch 在 `20342db0` 里，不在 `ce26015e` 里。**
> ⇒ ⚠️ **⇒ `.text` 只涨 192 B 与此不矛盾**：patch 落在 `narrowBinOp`（一个 421 KB 目标文件里的
> `static` 成员）内，只改动 5 条指令、且发生在已存在的符号里 ⇒ **不产生新符号名，
> 也几乎不改变总 `.text` 尺寸**。**我用总量界去否定成员级证据，是错的推断方向。**
>
> ### ⭐ 而且 patch 的行为已实测正确 —— 但我之前那个判别式本身是错的
>
> 旧版 §5.0 记的判别式用 **`fdiv`**。**而 `fdiv` 根本不会触发这条路径**：
> `narrowBinOp`（`InstCombineCasts.cpp:841`）的 opcode 列表是 `and/or/xor/add/sub/mul`（整数 `InstCombineCasts.cpp:856-862`）
> 与 FP 的 **`FAdd` / `FSub` / `FMul`**（`InstCombineCasts.cpp:2129` 起）⇒ **`Fdiv` 不在其中。**
> ⇒ ⚠️ **⇒ 这就是为什么我前面两次"判别式没反应"——不是 patch 没生效，是用例选错了 opcode。**
>
> **正确的触发条件**（`InstCombineCasts.cpp:851`）：`match(Trunc.getOperand(0), m_OneUse(m_BinOp(BinOp)))`
> ⇒ **`fptrunc` 的操作数必须只有一个 use，且那个 use 是一个 binop。**
>
> **实测（`fadd`，BO 带 `ninf` 而 `FPT` 不带）：**
>
> ```
> define <4 x half> @narrowed(<4 x half> %h) {
>   %e = fpext <4 x half> %h to <4 x float>
>   %b = fadd nnan ninf <4 x float> %e, %e
>   %t = fptrunc <4 x float> %b to <4 x half>       ; ← 故意不带 ninf
>   ret <4 x half> %t
> }
> ```
>
> | opt | `@narrowed` 的结果 | 判决 |
> |---|---|---|
> | `install/bin/opt`（13:09 重装） | `fadd nnan <4 x half>` | ✅ **`ninf` 被清 ⇒ patch 生效** |
> | `build/bin/opt` | `fadd nnan <4 x half>` | ✅ 同上 |
> | 对照组 `@control`（`FPT` 也带 `ninf`） | `fadd nnan ninf <4 x half>` | ✅ **`ninf` 保留**，正是 patch 的第二条 |
>
> ⇒ ⭐⭐⭐ **⇒ patch 的两条语义分支都实测正确。**
> ⇒ 复现：`llvm_triton/build/install/bin/opt -passes=instcombine -S` + 上面的 IR。
>
> ### ⇒ 于是 S0b 的账终于平了
>
> | | 状态 |
> |---|---|
> | ✅ 重建能力 | 解锁，2 分钟 |
> | ✅ `hmx-attr` 落地 | 是，但**可证 no-op**（`hexagon_hmx` 后端零命中，`IsHMX` 恒 false） |
> | ✅ **fdiv patch 落地且行为正确** | **本条** |
> | ✅ `a576182` K-fusion 落地 | 是，但**默认关**（`croutons-per-mma` 选项），产物逐字节不变 |
> | ⛔ `f16` 除法 194 次 libcall **未重测** | **现在才第一次具备可测条件** |
> | ⛔ 13:14 的 S1 失败 | 仍是环境/瞬态，非任何构建的性质 |
>
> ⇒ ⭐ **⇒ 下一件事很具体：在 `20342db0` 上重测 `fdiv <64 x half>` 的 libcall 计数。**
> **预期：仍 > 2**（因为 patch 只修 miscompile，**不取消收窄**）
> ⇒ **若仍是 194 左右 ⇒ 「方案 A 改 LLVM」这条路已被证明只能修对、不能提速，
> 「方案 B（InstCombine 之后插一个 pass 扩回 f32）」才是提速的那条。**
> ⇒ 那正是 `ROADMAP` 里 `:dad3f375`（方案 A）与 `:2f1ffef4`（方案 B）的分野，
> **而这个分野至今没有证据。**

### 勘误七之更正之七（17:25）：`f16` 除法那个单点，`matmul` 路径上根本不存在

> 更正之六说「现在才第一次具备可测条件」。**条件具备了，我一测，问题本身就不在这条路径上。**
>
> **实测（两个 `.so` 各编一次 `matmul` 的 `llir`，逐行对比）**：
>
> | | `ce26015e`（无 patch） | `20342db0`（有 patch） |
> |---|---|---|
> | `matmul_kernel` 函数体 | **160 行** | **160 行** |
> | 两个函数体逐字节比较 | — | ✅ **完全相同** |
> | 全文 `fdiv` 出现 | 10 | 10 |
> | `llvm.fdiv` | 0 | 0 |
> | `sqrt` | 0 | 0 |
> | 数学 libcall 的 `declare`（`sqrt`/`sin`/`cos`/`expf`/`logf`/`powf`/`__*f16`） | **0** | **0** |
>
> ⇒ ⭐⭐⭐ **⇒ `matmul` 路径上没有任何 f16 除法 libcall —— 无论有没有 patch。**
> ⇒ **⇒ 所以「194 次 libcall vs 2 次」那个数字，不是 `matmul` 的问题。**
>
> ⚠️ **⚠️ ⇒ 而旧版 §5.0 那个数字的来源，本文件至今没有记录。**
> **它在旧版 §5.0 里被当作已知事实引用（拿它论证「不给 fdiv 加 `nnan ninf`」），
> 但：它量的是哪个 kernel？哪个形状？哪条路径？本文查不到出处。**
> ⇒ ⭐ **⇒ 按项目自己的纪律，这一项应当记作「出处不明」，
> 而不是「已证存在、只等重测」。我此前两轮都按后者处理，两轮都错。**
>
> ### ⇒ 三个可辩护的结论
>
> **① `hmx-attr` 与 `a576182` 落地但对 `matmul` 产物零影响** —— 已由本条与
> 勘误七之更正之五的两组独立测量确认（`matmul_kernel` 逐字节相同 + 三形状指令流 md5 相同）。
>
> **② `fdiv` patch 落地且行为正确**（更正之六的 `narrowed` / `control` 两条分支实测）。
> **但它在 `matmul` 上无事可做。**
>
> **③ 「方案 A 改 LLVM」vs「方案 B 插 pass 扩回 f32」这个分野，目前无法用证据裁决** ——
> 因为**被比较的那个现象本身没能在 `matmul` 上复现**。
> ⇒ ⇒ **要推进这件事，第一步不是选 A 或 B，而是先找到「194 次」的那个 kernel。**
> ⇒ 候选：`exp` / `log` / `rsqrt` / `gelu` / `silu` / `softmax` 这类含超越函数的算子
> （`hexmlir-all` 里就有 `test_gelu.py` / `test_silu.py` / `test_softmax.py`）。
> ⇒ ⚠️ **⇒ 而今晚的 `hexmlir-all` 7/7 恰好证明这些算子当前都是绿的**
> ⇒ **⇒ 即「f16 除法」在真机上并没有造成正确性问题，只可能造成性能问题。**
> ⇒ ⇒ **那就要问：它到底在哪慢？有没有人测过含除法算子的 Perf？**
> ⇒ 📄 本条的复现：`OUT_DIR=<d> .venv/bin/python tools/hexmlir/dump_codegen.py matmul llir`
> ⇒ ⚠️ **该脚本的 artifact 只能是 `llir` / `manifest` / `o` / `ttsharedir`（写 `ttir` 会被拒）**
> ——我为此浪费了三轮，路径也猜错过一次。**它的 usage 在 `dump_codegen.sh:14`。**

---

## 八

> **⚠️ 勘误八（2026-10-02 17:30）：上面这个数字在当前构建上复现不出来，且其机制在本流水线上没有入口。**
>
> 被纠正的原断言（ROADMAP §5.0）：**`fdiv <64 x half>` → 194 次 libcall；`fdiv <64 x float>` → 2 次。**
>
> 全仓搜「194」：**本仓没有更早的出处、没有量它的脚本、没有它对应的那份 IR**
> （`docs/` 里其余 5 处命中全部是 2026-10-02 我自己写的）。
>
> **用当前构建（含 patch 的 `20342db0`）扫遍 `dump_codegen.py` 支持的全部 7 个算子**：
> `matmul` / `silu` / `softmax` / `rms_norm` / `vec_add` 的 **`x half` 均为 0、数学 libcall 均为 0**；
> `flash_attention` 有 8 处 `x half`，**逐处看全是 `load <64 x half>` + `fpext` 到 f32**
> （输入布局转换，不是算术）；全篇**没有一个数学函数被 call**。
>
> **机制上它也走不通**：`narrowBinOp` 的入口守卫是
> `match(Trunc.getOperand(0), m_OneUse(m_BinOp(BinOp)))` ⇒ **必须有 `fptrunc`**，
> 而 HVX→HMX 这条流水线上 `f16` 只出现在**输入侧**、算术全在 f32 ⇒ **永不收窄**。
> 且 opcode 列表里**没有 `Fdiv`**（只有 `and/or/xor/add/sub/mul` 与 `FAdd/FSub/FMul`）。
>
> ⇒ **⇒ 这一项应当记作「出处不明」，不是「已证存在、只等重测」。**
> ⇒ ⇒ **「方案 A 改 LLVM」vs「方案 B 插 pass 扩回 f32」这个分野目前无法用证据裁决**，
> 因为**被比较的现象本身没能在任何可达 kernel 上复现**。
> ⇒ ⇒ **下一步不是选 A 或 B，而是先找到「194 次」的那个 kernel**
> ——候选是 `exp` / `log` / `rsqrt` / `gelu`，**它们不在 `dump_codegen.py` 的支持列表里，从没被扫过**。
> ⇒ ⚠️ 反向证据：今晚 `hexmlir-all` **7/7**（含 `test_gelu` / `test_silu` / `test_softmax`）
> ⇒ **⇒ 它在真机上没有造成正确性问题，只可能造成性能问题；而含除法算子的 Perf 从未有过对照。**
>
> 📄 `docs/results/f16-division-194-unreproducible-2026-10-02.md`（含逐算子数据与复现命令）

**⇒ 补充（勘误七之更正之七）**：两个独立调查（`matmul` 的 `llir` 逐字节对比 + 全算子扫描）
都指向同一结论，且进一步定位到机制层——`Fdiv` 不在 `narrowBinOp` 的 opcode 列表里。

---

# 第二部分 · T_HMX 收益链

> 这一组直接作用于 ROADMAP 的立项理由。

---

## 四

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

**⚠️ 勘误四的两处结论后来被收窄**：

1. **那个「被留出数据独立确认」的 pin，其形状表含合成条目** ⇒ 见勘误九。
2. **`g = 0.30 µs/Kt-tile` 的量纲不是 `Kt`** ⇒ 见勘误五。

---

## 五

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
> int64_t scratchBytes = Kt * layout::kCroutonBytes;                        // ∝ Kt
> int64_t room         = vtcmBudget - vtcmBytesCommitted(func) + actBytes; // actBytes ∝ Mt
> auto fits = [&](int64_t d) { return scratchBytes + d * ringBytes <= room; };
> int64_t budgetDepth = fits(2) ? 2 : (fits(1) ? 1 : 0);
> ```
>
> ⇒ ⭐⭐⭐ **⇒ 流水深度被 VTCM 预算卡住，按 2 / 1 / 0 **跳变** ⇒ 关系里有间断
> ⇒ ⇒ **任何多项式项都补不上，这不是拟合能力不足。**
>
> ⚠️ **⇒ 勘误四那句「`g = 0.30 µs/Kt-tile` 是唯一被留出验证过的 staging 独有成本」
> 要改读法：它的**形式**（staging 独有、与 `K` 深度相关）被验证了，
> 但**它的量纲不是 `Kt`** ⇒ 那个系数是拟合产物，不是测量值。**
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

---

## 六

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
> ROADMAP §1.2 那句「输的恰是它的跨线程流水这一步是推断，不是实测」**至今仍然成立**。
>
> **解锁顺序**：
>
> ```
> 建一个能写出 pipeline 决策的 envelope 版本 ⇒ 读出逐形状 budgetDepth
>   └─ 才能给 T_HMX 分线程定量：拆线程救回多少 µs
> ```
>
> ⭐ **顺带一条与本方案无关但同源的**：`Kt ≥ kStageMinKTiles` 那个门（`603163c`）
> **已被留出数据独立确认**（`Kt=16` 那点判别力 4.5 倍，勘误四 §3），
> ⇒ **S2/S3 走 staged 路径这件事不再是「按推理定的」，而是有真机证据的。**
>
> 📄 `docs/results/output-term-is-rows-not-bytes-2026-10-02.md` §4.2–§4.3
> 📄 `exp/hmx/shape_attribution/dump_pipeline_decision.py`（读 `metadata["hmx_manifest"]` 的尝试；
> **保留，因为它记录了「v1 envelope 读不到」这个事实本身**）

**⇒ 勘误六要求的那个输入已经拿到了（`budget_depth` 进了 manifest，见 commit `2c3ebb9`），
而读数把它自己否掉了** ⇒ 见勘误九。

---

## 九

> **⛔ 勘误九（2026-10-02 16:50）：勘误六末尾那条「已被留出数据独立确认」的三条里，
> 第一条的前提被推翻，而被钉住的那个常量从未在真实算子上生效过。**
>
> **① `kStageMinKTiles = 32` 的溯源表把一个合成形状当成了 real。**
> `HmxPartitionPass.cpp:198-201` 写「one real shape sits exactly ON the boundary
> (**FA PV, 1024x128x1024, Kt=32**) … PV's K is seq (large, allowed)」。
> **而仓库里真实的 FA kernel 不是这个形状**：`test_flash_attention.py:105` 的
> `acc = tl.dot(p, v, acc)`，`p=(BLOCK_M, BLOCK_N)`、`v=(BLOCK_N, BLOCK_DMODEL)`
> ⇒ **op 的 K = `BLOCK_N` = 64**；kv 的循环在 dot **外面**，seq 是循环 trip count、不是 op 的 K。
> ⇒ **判据（实测）**：FA 在 `D_HEAD`/`BLOCK_N` = 64/64、128/64、128/128、256/128 四个配置下，
> 两个 matmul **全部 `shallow-k` / `budget_depth=0`**。若 PV 的 op-K 真是 1024，它会 staged。
> ⇒ **⇒ 溯源表里唯一那个「正好在门槛上」的条目是合成的**；`HmxPartitionPass.cpp:202` 自己也写了
> 「Nothing real lands strictly inside Kt 9..31」——真实算子全在 `Kt ≤ 8` 那一侧。
>
> **② 门槛判的是 op-K（切块步长），不是矩阵 K；且改谓词救不了。**
> `HmxPartitionPass.cpp:381` `getTileShape()` 读 `lhs.getDimSize(1)`（activation crouton 阵列的 K），
> `HmxPartitionPass.cpp:431-433` 除以 `kTileEdge=32` ⇒ `kTiles` = **这个 op** 的 K/32 ⇒ 门槛 = op-K ≥ 1024。
> 实测：固定逻辑 K=14336，只改 K 循环步长 ⇒ `BK`=32/64/128/256/512 全 `shallow-k`，
> `BK`=1024/2048 才 `staged`。**判定完全跟着 `BK` 走。**
> ⇒ **原因**：staging 环预取的是同一输出 tile 的下一个 K-tile，只能跨**同一个 op 内部**预取；
> 而 Triton 的 `for k0 in range(0, K, BK)` 降在 op **外面** ⇒ 每个 op 的 K 就是 `BK`。
> ⇒ ⛔ **「改读逻辑 K」这个修法不存在**：manifest 里 `matmul.logical.k.value` 同样是 per-op
> （逻辑 K=14336 时它报 64 或 1024，从不报 14336）⇒ **IR 里根本没有更大的 K**。
> 要让门槛看见矩阵 K，必须让一个 `MatmulOp` 跨越整个 K 循环 —— 那是 lowering 设计变更。
>
> **③ ⛔ 决定性：VTCM 预算从来不是约束（host-only，已测实）。**
> `HmxPartitionPass.cpp:1681` `budgetDepth = fits(2) ? 2 : (fits(1) ? 1 : 0)` ⇒ **`budget_depth==2` 在定义上就等于
> 「预算装得下 depth 2」**。统计全部 **103** 份 manifest 的 `pipeline.reason`：
> `vtcm-budget` 出现 **0 次**；凡是放行的（那 2 个 `tile-count`）`budget_depth` 都是 2。
> ⇒ 且 ring 本身极便宜（`HmxPartitionPass.cpp:1647-1672` 原式 `slotBytes=32·K·2`、`ringBytes=slot+4`、
> `scratchBytes=Kt·2048`，预算 8 MiB @ `HmxTarget.h:118`）：
> **op-K=1024（门槛正上方）时 depth 2 只占预算 2.34%**；即使 op-K=16384 也只有 37.5%。
> ⇒ 算术自检：同式复算 FA 两个 matmul = 3.22% / 合计 6.45%，与此前记录的 3.2% / 6.4% 一致。
> ⇒ **depth 的真正上限是几何不是钱**（`HmxPartitionPass.cpp:1697-1700` ring 深度追上 tile 数即无稳态；`HmxPartitionPass.cpp:1688` 硬夹 2）。
>
> **⇒ 对 ROADMAP 的影响**：勘误六那条「先读出逐形状 `budgetDepth`，
> 才能给 T_HMX 分线程定量」的前置，**其输入量恒为 2** ⇒ **可隐藏的 µs 计算不出非零值**。
> ⇒ **失效条件**：出现 `BK ≥ 1024` 的 `tl.dot`、或 K 循环被改进 op 内部、或 `defaultVtcmBudget`
> 大幅下调、或出现单 kernel 多 matmul（`vtcmBytesCommitted` 会吃掉 `room`）——任一条成立则本勘误作废、必须重测。
>
> 📄 `docs/results/t-hmx-staging-gate-dead-2026-10-02.md`（含逐档数据、traceback 与复现命令）
> 📄 顺带：`exp/hmx/accumulator_budget_headroom.py:90-94` 是**同一处 2^20 误读**的第二个副本
> （「caps EVERY tensor's numel at 2^20」）⇒ 该脚本的 "decisive question" 段结论需重算

### ⚠️ 勘误九自身的一处越界（2026-10-02 17:0x，尚未写入 ROADMAP）

勘误九**原文**最后写的是：

> ⇒ ⭐ **建议把 T_HMX 从「暂停等数据」改判为「关闭」**，理由按强弱排序：③ > ② > 深度封顶。

**⇒ 这条建议已撤回。** 它推不出来：

| | 对象 | 收益来源 | 勘误九的测量结论 |
|---|---|---|---|
| **T_HMX** | **两个引擎之间** | 让 HMX 和 HVX 同时满负荷 | ⛔ **没测这个** |
| **staging** | **HMX 内部，搬运 vs 计算** | 内存挤时互相盖时间 | ✅ 无意义 |

勘误九测的是右列，结论只能否定右列。**左列（T_HMX 的引擎级 co-scheduling）不在测量范围内。**

⇒ **勘误九成立的范围**：**「用 `budgetDepth` 给 T_HMX 定量」这条路死了**（③ ② 都成立）。
⇒ **勘误九不成立的范围**：**T_HMX 本身没有被证伪**。
⇒ **T_HMX 的判决仍挂在 ROADMAP §6 第 11 条那个已预登记的净增量门上。**
---

## 勘误十（2026-10-03 下午）— T_HMX 首次得到**可信的负结果**，以及它推翻的五个假设

本条只追加，不改动上面任何既有文本。

### 1. 先修两个把测量变成噪声的故障（都不是功能问题）

**(a) 设备侧 DSP 进程泄漏。** 每次启动失败都会留下一个 `./run_main_on_hexagon`，
占住一个 unsigned PD domain。domain 数量有限，所以**第一次失败之后所有测量都是假的**，
症状是 `Error -2147482611: Failed to call main() on DSP`，与「没有加速」无法区分。

- 14:08 抓到 4 个残留（PID 18570/20514/23092/24543），杀掉后同一 kernel 立刻
  `Perf:62.000000 / PerfPcycles:132604`（对照 11:40 基线 61 µs / 129,972 ⇒ 无回归）。
- **`ps -A` 输出是 `PID ? ELAPSED CMD`，PID 在 `$1`；`$2` 是 `?`。**
  `readout_sweep2.py:reap_dsp` 最初取 `$2`，因此**静默什么都没杀**，白费 40 分钟。

**(b) Triton 后端用 `sys.exit(1)` 报设备失败。**
`triton/backends/qcom_hexagon_backend/hexagon_executor.py:752` ——
这是 `BaseException`，会**穿透 `contextlib.redirect_stdout` 直接杀掉进程**。
未加保护时扫描脚本只打印表头然后 `rc=1` 退出，一行结果都没有。
这解释了此前多次「OFF 臂没打出 Perf 就退出」。

### 2. 可信的负结果（1024×512×64 fp16, depth=2, `tools/run_tests.sh lock`）

| arm | Perf µs | PerfPcycles | read-out 符号 |
|---|---|---|---|
| OFF | 64 | 135742 | 0 |
| ON G=1 | 2090 | 1332669 | **5** |
| ON G=2 | 2085 | 1344103 | **5** |

- **符号 = 5 ⇒ 改写在对象里。** 这是今天第一个可信的负结果。
- **G=1 与 G=2 几乎相同（2090 / 2085）⇒ 完全平坦。**
  批量模型 `24300 ns / handoff_ns` 预言 G=8 → 1.51×；实测 **0.03×**。
- Perf 对 ITERS **平坦**（1/10/100/1000 → 2062/2306/2217/2093 µs）⇒ 是**每 iteration** 成本。
- **G=4 挂死**：14:21 起未完成，直到 15:14 被超时杀掉（单臂 >53 min）。

### 3. 读出工作量是对的，多出来的 cycle 不在读出上

记账文件 `/data/data/com.termux/files/home/csm/op/hmx_readout_acct.txt`
相邻两条 `drain` 记录做差：批数 +16、pcyc +51132 ⇒ **3196 pcyc/batch**，
一次 launch 的读出 = 16×3196 = 51132 pcyc = OFF 臂的 **37.7%**，
与原始测得的 **39.79%** 同量级 ⇒ **`__hmx_readout` 本身没有变慢。**

引擎侧多烧 1332669 − 135742 = **1196927 pcyc/iteration**。
若每 iteration 16 次 publish ⇒ **~74800 pcyc/publish**，
而 publish 全部内容只是 24 字节拷贝 + 2 次原子操作 + 1 次 futex wake。

### 4. 隐含频率暴露了真正的形态

| | Perf µs | pcyc | 隐含频率 |
|---|---|---|---|
| OFF | 64 | 135742 | **2.12 GHz** |
| ON | 2093 | 1332669 | **0.64 GHz** |

32× 墙钟里只有 9.8× 是真实 cycle，**约 1465 µs 是纯等待**。
⇒ 瓶颈**不是读出算力，是等另一个线程**。

### 5. 本条推翻的五个假设（全部保留记录）

1. **「grid 形状的 row 链带 `program_id` 项，折叠不掉」** — 错。
   两种形状 row 链**完全相同**（都是 `%iv + 0`），`program_id` 只出现在目标偏移里。
2. **「grid 形状生成不了、符号=0」** — 测量为真但归因错。
   `HmxPartitionPass.cpp` 的 late-dst hoist 在 **12:02:53** 生效
   （缓存对象：11:46 / 12:00 为 0 符号，12:02:53 起为 5）；早前扫描跑在 hoist 之前。
3. **`tailStart` 的 `+1` 少一行 AR** — 是被**种进树里的变异**（早前 agent 做变异测试时留下），
   不是发布缺陷。`.bak`/`.good` 均为正确公式且无 MUTATION 注释。
4. **「publish 的开销是 `qurt_futex_wake`」** — 错。
   `llama.cpp/ggml/src/ggml-hexagon/htp/hmx-queue.h:87-88` **完全一样**：
   每次 push 都 `atomic_fetch_add(&seqn,1)` + `qurt_futex_wake(&seqn,1)`，
   且 `HMX_QUEUE_POLL_COUNT` 在 v79 上**也是 1**（`:20-24`）。
5. **「线程优先级 / HVX 上下文取得方式有差异」** — 也没有。
   本实现 `HmxVectorExecutor.cpp:473-482` 与 `hmx-queue.c:131-138` 逐行对应；
   `:345` 同样取 HVX 锁，`:416` 同样在空闲时 `qurt_hvx_unlock()`。

⇒ **结构性的关键差异只剩一处**，且尚未测量：
llama.cpp 的 `matmul-ops.c:2601-2623` 是 `push(i)/pop(i-1)` 的**单深流水**，
生产者恒定只领先一格，消费者因此**几乎永不 park**，futex 等待路径走不到；
本实现的消费者会 park 进 futex 并付调度延迟。

### 6. 尚未测量的下一步（不得再凭推测动手）

**给执行器加两个计数器并在设备上打印：消费者 park 次数、引擎 futex 等待次数。**
这直接判定「futex 路径到底有没有被走到」。在此之前不得声称已定位。

### 7. 门与基线（未提交）

`ninja` rc=0 · lit **328/0/1** · pytest **202** · object gate **8 passed**
（门已重写为读**实际发出**的 `rowStart`/`rowCount`，不再按期望公式重算；
变异验证：`tailStart` 改回 `+1` 时 **1 failed / 7 passed**，
报错点名丢失的行 `[28]`，而所有符号检查仍绿 —— 只有新读取器看得见）。

**四块改动全部未提交**：S1（629 行）/ 抬 unpack（130 行）/ S2 运行时+ABI（~1900 行）/ readout pass。

---

## 勘误十一（2026-10-03 19:00–20:10）— §6 那个 0.03× 是**缓存假象**，真实结果是 **1.27×–1.33×**

本条追加，不改动上面任何既有文本。**勘误十 §2 的负结果已被本条推翻**；
本条只保留勘误十里仍然成立的部分（两个故障、隐含频率分析、五个被推翻的假设）。

### 1. 根因：Triton 缓存不按运行时库做键，陈旧 `.so` 被当成命中

`readout_sweep2.py` 的每臂缓存隔离是**无效的**：

```python
os.environ["TRITON_CACHE_DIR"] = cache    # 在 import triton 之后设置 ⇒ 不生效
```

triton 在 import 时就解析了缓存目录，所以所有臂共用同一个默认缓存。
更关键的是 **triton 的缓存键覆盖 kernel 源码与编译选项，但不覆盖
`libhexagon_mlir_async_runtime.a`** —— 改了 `HmxVectorExecutor.cpp` 并重新 ninja
之后，缓存命中仍会发给你一个**链接着旧运行时**的二进制。

**受控验证**（同 kernel、同选项、同 iters=1，只差缓存冷热）：

| 缓存 | Perf µs | pcycles | 对象符号 |
|---|---|---|---|
| 默认（陈旧 runtime） | 2062 | 1332669 | 5 |
| 全新（当前 runtime） | **49** | 103030 | 5 |

⇒ 勘误十 §2 的 2090 µs、32× 变慢、G 平坦、G=4 挂死，**全部是这一个原因**，
不是交接代价、不是 futex、不是 HVX 锁、也不是优先级。

**已修**（`exp/hmx/t3_overlap_ab/readout_sweep2.py`）：
`TRITON_CACHE_DIR` 在 import triton **之前**用 `setdefault` 设置；
并新增 `_runtime_stamp()`，把 `libhexagon_mlir_async_runtime.a` 的 **mtime**
放进缓存路径 ⇒ **运行时一改就自动冷缓存**，不必靠人记得清目录。

### 2. 修好之后的真实测量

`1024×512×64 fp16, depth=2, iters=1`，`llvm-nm` 逐臂确认符号：

| arm | Perf µs | pcycles | read-out 符号 | |
|---|---|---|---|---|
| OFF | 61–63 | ~130000 | 0 | 基线 |
| ON G=1 | **48** | 102528 | **5** | **+21.3% (1.27×)** |
| ON G=2 | **46** | 97237 | **5** | **+24.6% (1.33×)** |

两个 ON 臂的对象里都有 5 个 read-out 符号 ⇒ **不是"读出没发生"**。

**T_HMX 方向被证伪了，批量模型的形状也被证实了**：G=2 > G=1，
正是 `24300 ns / handoff_ns` 预言的单调上升。G=4/8 尚未测完（见 §4）。

### 3. 编译从来不是瓶颈（实测拆解）

| 阶段 | 耗时 |
|---|---|
| import 模块 + triton + MLIR 插件 | 1.3 s |
| `warmup` → `.o` 冷编译 | **5.7 s** |
| `warmup` → `.o` 缓存命中 | **0.00 s** |
| **kernel launch（缓存命中也是）** | **20 s** |
| 宿主 `torch.matmul` 参考比较 | 0.03 s |
| `adb shell` 往返 | 0.32 s |
| `reap_dsp`（一次 ps） | 0.44 s |

⇒ **每臂硬下限 20 s，全在 `hexagon_executor.py:583-589`**：
每次 launch **无条件 `adb push`** 输入张量 + `run_main_on_hexagon` + 全部 .so，
**没有哈希比对**。3 MB 输入走 `adb reverse` 隧道，每次重来。
**这是 Triton 传输层的问题，不是本项目的问题，也不是扫描脚本的问题。**

### 4. 扫描脚本已加固（后续 agent 直接用，不要重写）

`exp/hmx/t3_overlap_ab/readout_sweep2.py` + 新增 `exp/hmx/t3_overlap_ab/sweep_readout.sh`：

- **`--arm off|g<N>`：一臂一进程**，每臂独立超时（`PER_ARM_TIMEOUT`，默认 420 s）。
  理由实测：单进程跑全部臂时 G=4 挂了 53 分钟，把已完成的臂一起赔进去。
- 结果**增量追加**到 `/tmp/opencode/readout_results.tsv`，挂掉不丢已完成结果。
- 臂间清 `run_main_on_hexagon` 残留（否则下一次 launch 报
  `Failed to call main() on DSP`，与"没加速"无法区分）。
- `readout_sweep2.py` 顶部注释记录了缓存键这个坑，`sweep_readout.sh` 顶部记录了
  一臂一进程的由来。**改这个脚本前先读那两段。**

### 5. 交给下一个 agent 的下一步（按顺序）

1. **补齐 G=4/8/16/32**：`ARMS="off g1 g2 g4 g8" bash exp/hmx/t3_overlap_ab/sweep_readout.sh 1`。
   批量模型预言 G=8 → **1.51×（40.3 µs）**。已测的 G=1 1.27×、G=2 1.33×
   已落在模型的形状上，**G=4/8 是判定模型成立与否的决定性数据**。
   缓存现在按 runtime mtime 分目录，**必须确认缓存路径里出现了新的时间戳目录**，
   否则测的还是陈旧二进制。
2. **撤掉诊断计数器**：`HmxVectorExecutor.cpp` 里新增的
   `nPark/nHvxLock/nHvxUnlock/nPublish`（结构体约 :124-131，
   `publish` 入口、`vectorThreadEntry` 的 park/hvx 点、`reportReadoutAccounting`）
   **在定位完成前保留**；一旦 G 曲线确认，要么留着当长期诊断（开销是 4 个 relaxed
   原子加，实测未改变结果），要么连同两条 `hmxExecTrace("mech*")` 一起删。
   **它们没有让 ON 从 1.27× 掉回 0.03×，也不是原因。**
3. **不要再查 futex / HVX 锁 / 优先级**：勘误十 §5 已经用 llama.cpp 源码逐条排除，
   三处实现都是对的。唯一仍成立的机制线索是 §4 里那条 —
   `matmul-ops.c:2601-2623` 的 `push(i)/pop(i-1)` 单深流水让消费者几乎永不 park，
   而本实现会 park。**但那只在 park 计数证明确有大量 park 时才值得动手，现在没有该计数。**
4. **不要碰设备**除非走 `tools/run_tests.sh lock`；`adb` 在 `tools/hexmlir/adb`，不在 PATH。
5. **四块改动全部未提交**（用户明确指示过两次"先不提交"）：
   S1 ThreadRolePartition（629 行）/ 抬 unpack（130 行）/ S2 运行时+ABI（~1900 行）/ readout pass。
   门：ninja 0 · lit **328/0/1** · pytest **202** · object gate **8 passed**。

### 6. 今天的方法论收获（比任何单个数字都值钱）

**六次拿间接信号当结论，六次被推翻**，但这次终于收敛到一个可复跑的判据：

| # | 错误结论 | 被什么推翻 |
|---|---|---|
| 1 | 输出正确 ⇒ 执行器跑过 | 输出来自没被改掉的代码 |
| 2 | 符号消失 ⇒ visibility 丢弃 | `sym_visibility` 到不了 LLVM IR |
| 3 | 门 7 passed ⇒ 功能实现 | 门只测整矩阵 kernel |
| 4 | row 链带 `program_id` 项 | 两种形状 row 链完全相同 |
| 5 | publish 贵在 `qurt_futex_wake` | llama.cpp `hmx-queue.h:87-88` 一模一样 |
| 6 | 线程优先级 / HVX 取得方式有差异 | 本实现 `:473-482` 与 `hmx-queue.c:131-138` 逐行对应 |

**唯一正确的是受控实验：只改一个变量，其余全同。** 而本条最关键的一课是：
**"受控"还不够，两个臂必须真的用了同一份被测物。** 本条 §1 里两个臂的
kernel 源码、选项、iters 全同、符号都是 5，唯一差别是**缓存里那份 `.so` 链的运行时不同**。
⇒ **动手前的第一个实验应该是：确认被测二进制里含你要测的东西，
并让它的失效条件覆盖你会改的一切。** 这就是 `_runtime_stamp()` 存在的理由。

---

## 勘误十二（2026-10-03 23:00–23:45）— G≥4 "灾难性变慢" 是 **drain 屏障的过冲死锁**；"318 ms/iter" 从未存在过；批量模型否证

本条追加，不改动上面任何既有文本。**勘误十 §2 / 勘误十一遗留的
"G=4 每 iteration ~318 ms、G=8 ≥420 s 超时" 之谜在本条收口。**

### 1. "318 ms/iteration" 是把"被外层杀掉的挂死"硬除出来的虚构数字

iters 语义（代码链：`hexagon_options.py:51` → `compiler.py:185` → `driver.py:157` →
`hexagon_launcher_base.py:124-126` → `hexagon_benchmark.h:57-71`）：**一次 launch 在设备上
把 kernel 完整执行 N+1 次**（+1 是 `hexagon_launcher_base.py:122` 丢弃的暖机调用），
Perf = 每次调用的平均。所以：

- **G=4/G=8 在 iters=1 时也没落结果行** ⇒ 第一次 launch 就没能完成 ⇒ 不是慢，是死锁；
- 53 分钟 = 挂死被外层超时杀掉（勘误十 §2 的 14:21 事件本身就是这么记录的），
  "53 min / 10000 iters = 318 ms/iter" 没有任何一次完成测量作支撑；
- 318 ms 恰 ≈ 8 × 39.75 ms 曾被当成"~40 ms 系统节律救活"的证据——机制上不成立：
  过冲楔死连虚假唤醒都救不了（生产者卡在 drain 里不再 publish，消费者追平后 park，
  谁也不会再动）。[此前的 40 ms 节律假说随之作废]

### 2. 根因：drainLocked 的屏障是**模 16 环位置相等**，消费者跑过头一格即永久楔死

每调用 publish 数 = `floor(31/G)+1` = **32/16/8/4**（G=1/2/4/8），环容量 16
（`HmxVectorExecutor.cpp:72`）。旧 `drainLocked`（已删）逐槽等
`idxRead == next`：`processAvailable` 背靠背 retire 多批中间不 park（每批 ~0.76 µs/行），
消费者越过 `next` 位置后，模 16 相等永远无法满足；生产者卡在 drain ⇒ 不再 publish，
消费者追平后 park 在 seqn ⇒ **双侧永久 park**。probe3 在设备上两次实测同形状楔死
（`exp/hmx/s2_handoff/probe3.cpp:289-311`），probe 当时改成了单调计数器——
**但 shipped `HmxVectorExecutor.cpp` 一直是环相等版**。

**G=1/2 为什么"幸免"**：32/16 ≡ 0（mod 16）⇒ `idxDrain == target` 恒成立 ⇒
drain 是**数学空操作** ⇒ 内核返回时尾批读出还在飞。⇒ **勘误十一 §2 的 G=1/G=2 数字
（49/47 µs）是无屏障数字**：正确性靠 "host 往返（ms 级）≫ 在飞读出（≤24 µs）"
的时序运气，不是正确性保证；且它与大 iters 下的行为不可比。

### 3. 修复（只改 `.cpp`，冻结的 `HmxVectorExecutor.h` 未动）

照 probe3 设备验证过的形状（`probe3.cpp:223-244, 289-311`）：

1. **单调屏障**：新增 `publishedTotal`（publish 按 accepted 累加）；drain 等
   `retireSeqn == publishedTotal`（两个都是全量单调计数，永不越过）；删除 `idxDrain`。
2. **消费者 park 前 re-load**：旧注释声称 futex 值检查能关丢失唤醒窗口——
   probe3 设备日志证明不能；加显式 re-load（窗口不再依赖 `qurt_futex_wait` 的内部实现）。

干预性确认：修复前 G=4 iters=1 挂死（120 s 超时杀，23:23 诊断扫描）；修复后（archive
md5 `71d6460d197007c6a988f3d4b3209a84`，旧 `109c522bc13cfe1fed8c05f30bc14274`）
G=4 iters=1 = 49 µs 数值对，且 iters=1000（= 每 launch 1001 次跨活屏障）零楔死。

### 4. 修复后的完整曲线（iters=1000，数值全对，`/tmp/opencode/readout_results.tsv`）

| arm | Perf µs | pcyc | 符号 |
|---|---|---|---|
| OFF | 63 | 133703 | 0 |
| G=1 | 49 | 104927 | 5 |
| G=2 | 49 | 104136 | 5 |
| G=4 | 48 | 102458 | 5 |
| G=8 | 49 | 104567 | 5 |
| G=16 | 53 | 113962 | 5 |
| G=32 | 64 | 135566 | 5 |

（G=16/32 为首次实测，此前从未跑成过。23:36 那次 g16 "挂死" 是伪影：sweep 父进程
被外层 shell 超时杀掉后，孤儿臂的输出拉取挂在半死的 ssh ControlMaster 上——设备上
输出张量其实已全部写出（内核跑完了全部 1000 次迭代）；干净重跑 53 µs 通过。）

**曲线形状的物理解释**：收益 = 被重叠掉的读出比例。总读出 ≈ 24 µs（32 行 × 0.76 µs），
G 决定尾批行数 = 32 mod G（不重叠、drain 干等的部分）：G≤8 时尾批 ≤8 行，
几乎全部读出被重叠（49 µs 地板）；G=16 尾批 16 行，重叠一半（63 − 24×16/32 ≈ 51，
实测 53）；G=32 整批 32 行全在尾部，零重叠（≈ OFF，实测 64）。
**G 不是"批量越大越好"的旋钮，是"愿意把多少读出推迟到不重叠的尾部"的旋钮**；
本形状最优区间是 G∈[1,8] 的平地板。

**批量模型否证**：交接的 "G=8 → 1.51×" 不成立。实测 G∈{1,2,4,8} 全部 ~48–49 µs
（vs OFF 63 µs，**1.29×**，iters=1000）：**收益来自拆分本身，批量 G 不改变收益**
（唤醒/发布开销在此形状下本就不是大头）。注意 G=1/2 现在付的是**真屏障**
（旧 49/47 是空操作数字），仍 49 µs ⇒ 屏障成本 ≤2 µs。

### 5. 两个附带发现

1. **accounting 文件对 FastRPC-launched kernel 是死通道**：`hmxExecTrace` 的
   `fopen` 在 launch 上下文静默失败（同 printf 被吞一类）。对照实验：g1 完成、数值对、
   `HMX_EXEC_ACCT=1` 已传，`hmx_readout_acct.txt` 仍空。⇒ ① 诊断挂死不能靠它
   （本条用屏障形状 + 设备目录时间戳定位）；② `HmxVectorExecutor.cpp` 里
   "3×684 µs 记账 I/O 解释 2090 µs 回归" 的旧注释是错的（写入从未发生，684 µs
   是 probe 进程里测的），已改；勘误十一 §1 的陈旧缓存归因不受影响。
2. **对象码"错译"假说被 ISA 手册否证**：G4/G8 的 `{ call publish; memw(rowStart);
   memw(rowCount) }` 同 packet 是合法调度——V75 PRM §3.3.2："packets execute to
   completion – including updating all registers and memory – before the next
   packet begins"，call 目标就是下一 packet。同模式在数值全对的 OFF/G1/G2 对象里
   各 208 处。AGENTS.md 第 7/9 条（仪器无分辨力/探针旗标）的又一例：
   把 packet 内的指令序读成串行序，差点又立一个 miscompile 假说。

### 6. 测量基建（本条顺手修掉的三样）

1. **`HEXAGON_FAST_LAUNCH=1`**（默认关，仅 sweep 用）：稳定设备目录
   （`create_timestamped_folder` 去时间戳）+ adb shim 按 md5 只推差异文件
   + executor 不删设备库/目录。实测：输入/skel/libc++ 只推一次，其后每臂只推
   变化的 `libmatmul_kernel.so`（~1 MB），8.8 MB/launch → ~1 MB/launch。
   （libc++.so.1 一个文件 4.5 MB，此前交接文档只算了 3 MB 输入。）
2. **runtime 戳真正进了缓存路径**：勘误十一 §1 声称已修的 `_runtime_stamp()`
   实际从未进 `TRITON_CACHE_DIR` 的值（import 时 setdefault + 从未生效的 makedirs）；
   现在在 import triton 之前拼进路径。
3. **sweep 两个自伤 bug**：`rc=${pipestatus[0]...}` 是 zsh 惯用法（bash 是
   `PIPESTATUS`），HUNG 行与 acct 拉取从未执行过；python stdout 块缓冲把被杀臂的
   诊断输出全部吞掉（现在 `PYTHONUNBUFFERED=1`）。

## 勘误十三（2026-10-04）— 读出拆分六项收口：交接开销机制上消光，组合最优 61→36 µs（1.69×）；两处旧结论修正

上一节遗留的六个"没做"全部做完（六份报告在 `exp/hmx/{t6_live_counters,t3_engine_split,t4_shapes,t5_hvx_contention,t1_deferred_drain,t2_handoff}/REPORT.md`，
设计文档 `docs/hmx/deferred-drain-design-2026-10-04.md`）。除注明外全部
iters=1000、1024×512×64 f16 d2 grid=1、数值全对（TOL 5e-2）。

### 0. 定稿数字（父 agent 最终确认扫描，archive `a903487c…` 全程一致）

| 臂 | µs | pcycles | 符号 | vs OFF |
|---|---:|---:|---:|---|
| OFF | 61 | 129634 | 0 | 1× |
| G=4（拆分） | 45 | 96745 | 5 | 1.36× |
| G=4+WSR+延迟drain+T2运行时 | **36** | 77201 | 4 | **1.69×** |

### 1. 勘误十二 §5.1 修正："fopen 静默失败"是错误推断

perf.txt 由 wrapper 模板 C 代码用**绝对路径** fopen 写出且每次都回得来
（`hexagon_launcher_base.py:146`）——同一 launch 上下文文件 I/O 本来就是通的。
T6 的判定臂三证互印：DSP 进程 `getenv("HMX_EXEC_ACCT")` → `(null)`；**同一进程**
对 accounting 路径 fopen 探针**成功**；Perf 不含任何记账写成本。**真因：env 止步于
CPU 侧进程，不跨 FastRPC。** 其操作性结论（该通道对 launched kernel 不可用）不变，
死因改写如上。⇒ 所有"设备侧 getenv 控制开关"的方案同病：launch 上下文里 env 通道
是死的，别再设计依赖它的东西。

### 2. T6 活体计数器通道（新基建，零扰动）

`HmxVectorExecutor.cpp` 加有界诊断环（256 条，单调序号索引，relaxed 纯诊断，
无同槽竞争——数据环背压 15 < 256）+ 新导出 `hexagon_runtime_hmx_exec_dump(path)`
（visibility default，不进冻结头）；wrapper 模板在**计时区外** weak 调用它写进
perf.txt（`hexagon_launcher_base.py`）。零扰动验证：OFF 61 / G4 48（基线内），
vec_add（weak→null 路径）PASS。**从此交接机制不必曲线反推**：交接延迟、park 次数、
末次 drain 等待全部直读。

### 3. T3 引擎侧分解（63 µs OFF = mma 24.3 + unpack 22.4 + VTCM alloc/free 8.8 + pack_act 3.5 + dma ~3 + 杂项 1）

- **搬 pack_act 不值**：上限 5.5%（3.5/63），离 15% 判据差 3 倍；扣交接后净收益 0-2 µs。
- **新杠杆：每次 launch 的 VTCM alloc/free ≈8.2-8.8 µs（13.9%），几乎全在 1 MB AR 数组**——
  比 pack 大 2.4 倍。已被 T1 的 WSR 组合顺手拿掉（见 §5）。
- LWP 对本内核无分辨力（单一 do-while 直线循环，LWP pass 只括 scf.for ⇒ 1 个 region），
  改用 probe 隔离法（对象反汇编逐参数解码 + 独立进程复刻计时，总体闭合差 2.0%）。
- **读出总量口径修正**：T3 的 22.4 是 probe 形态；T4 用 T6 dump 直读 = **24.8 µs**
  （行数×0.77 @N=512，与 K/depth 无关；0.43 @N=256）。以后以 dump 直读为准。

### 4. T4 多形状验证 + T5 双单元（模型与硬件边界）

- **模型结构项跨 6 形状成立**（24/24 ON 臂吻合），两补丁：①净收益要减**残差
  H ≈ 0.27 µs/AR行 @G=4**（六形状一致；机制未分解）；②**depth=1 时"尾批"要重定义**——
  最后一个环内发布若恰落在最终迭代（G 整除 upper），其 G 行同样全暴露
  （实测 73 vs 天真预测 42.8，修正后 63.8 ✓）。**G=4-8 是全形状共同最优**，净收益
  12.8-26.5%。
- **双 HVX 单元真并行**（T5）：两线程 100%+100% 满负荷并发吞吐 1.969/2.0，
  两窗口各自满频 2112 pcyc/µs（非分时）；`qurt_hvx_lock+unlock` 一对 372 pcyc；
  内核内每批读出比隔离慢 ~9%（**大头不是 HVX 单元**，疑访存干扰 [假设]）。
  `HmxVectorExecutor.cpp` 的 "OPEN HAZARD"（单单元即挂死）就此关闭。
  **墙是访存（unpack 单独已 ~100 GB/s），不是 HVX 单元**。

### 5. T1 延迟 drain（默认关原型）+ T2 交接削减（机制消光）

- **T1**：`hmxReadoutDeferredDrain`（默认关）内核出口不发 drain；下一次调用的
  configure drain（已有）成为入口屏障；wrapper 循环后 weak drain 兜底末次调用。
  **AR 生命期暗坑**（延迟后消费者读已释放内存）由 **workspace-resident 覆盖 AR**
  解掉——对象级证据：WSR 开时出口 `free_1d_dsp`×4 全部消失变 resident。**两机制
  部分冗余**（各删 alloc/free 的一半）：G4 47 → +延迟 44 → +WSR 40 → 三开 39。
  iters=1 单调用语义 PASS。风险表（barrier 是移动非删除、grid>1 契约外、ring 压力）
  见 `docs/hmx/deferred-drain-design-2026-10-04.md`。
- **T2**：consumer **持锁跨有界自旋**（只在真正 futex park 前释放，第三方最多等
  ~10.5 µs 窗口）+ **kPollCount 1→2000**（实测标定：11.12 pcyc/poll 迭代，反汇编
  对形；2000×5.27ns≈10.5 µs 盖住 G=8 批间隙+drain 等待；偏离 llama.cpp v79=1 的
  理由=T5 双单元真并行+第二核空闲，其 v79=1 是省电端选择）。机制前后（dump 直读，
  G=4）：交接延迟 **0.50→0.04 µs**、nPark **7950→1002**（残余=调用间空隙）、
  nHvxLock **7994→1001**、末次 drain 等待 **3.74→2.54 µs**（残余=尾批真实读出）。
  端到端每臂 −2~3 µs（**低于 15% 判据，如实声明**：主判据是机制数字；端到端是辅证）。

### 6. 待用户拍板

1. **组合默认开否**：G=4+WSR+延迟drain（36-37 µs，vs OFF −41%）——三个开关的组合
   最优已实测过门，但 grid=1 契约边界（WSR grid>1 已知失败、延迟drain grid>1 契约外）。
2. T2 自旋的**功耗/热未测**；≥3 并发 HVX 使用者未测（现持锁窗口 ≤10.5 µs 有界）。
3. 看板验收：T1-T6 与 S2 待移 Done（WIP 已满，等这批验收放行）。
4. 验收后：重跑 `python3 tools/hexmlir/split_patch.py`，再由用户 commit
   （工作树含 S1/S2 未提交改动 + 本轮全部改动，agent 不代 commit）。

### 7. 指纹与方法

- 最终树指纹：`libtriton.so` `10a4186768b6af45195353f4d0f931f5`、
  `libhmxapi.a` `97af133e…`、`libhexagon_mlir_async_runtime.a`
  `a903487c5f6b19a05f2f3b02bbefcbd7`（T2 后；T2 前为 `d4cf4ebc…`，T1/T4/T5 期间）。
- 每臂一进程 + 超时 + 指纹前后核对的模式沿用勘误十二 §6；所有百分比 iters=1000。
- 流程披露（T2）：一次 iters=1 复测直接调 python 未包设备锁（当时无并发），
  已记录于 `exp/hmx/t2_handoff/REPORT.md` §6.7。
