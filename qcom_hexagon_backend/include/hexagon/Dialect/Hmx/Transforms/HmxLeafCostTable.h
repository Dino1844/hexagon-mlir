//===-- HmxLeafCostTable.h - measured HMX leaf prices (DIAGNOSTIC ONLY) ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// WHAT THIS IS (W1, docs/architecture/hmx-coscheduling-followups-2026-10-07.md
// §2 W1, adversarial-review revision): the measured per-leaf pcyc table and
// the diagnostic queries over it. It is deliberately a SEPARATE cost
// structure: it does NOT go into HmxTarget.h, which is the engine's
// capability CONTRACT (what the hardware takes) -- mixing build-specific
// measured numbers into a contract file makes the contract drift with every
// rebuild. Nothing in the compiler pipeline includes this header yet; the
// first intended consumer is W2's bounded enumeration selector and the
// manifest `modeled_ii` diagnostic (see THE MANIFEST SPEC below), both of
// which are post-unfreeze work.
//
// W1 CLOSE-OUT (2026-10-09): the DERIVED section below modeled_ii carries
// the co-scheduling review's S1 default-face chain decomposition and the
// World A/B bounds -- the R1 window's reading frame. Those rows are
// inferences composed from the measured cells above, never measured slots;
// their structures are deliberately separate (see that section's header).
//
// === THE FOUR CONTRACT NOTES (read before using any number here) ===
//
// 1. UPPER-BOUND ACCOUNTING. Every modeled total below is a SUM OF
//    INDEPENDENT leaf prices -- no overlap credit anywhere. The real
//    single-threaded kernel already overlaps its regions by a measured >=37%
//    (the real S1 core ran 68,262 pcyc against a 109,201 pcyc three-region
//    sum; logs/hmx/lwp-region-attribution-2026-09-27.log,
//    docs/hmx/hmx-hvx-co-scheduling.md "37%" section). It is
//    FORBIDDEN to subtract one arm's modeled total from another to derive
//    a net cross-thread gain; cross-thread predictions must use measured
//    treat/base ratios. (ROADMAP1001 §5.1.5 禁算术的延伸：
//    禁止从表直接减出净增量。47.2% 与 37% 不许相减 -- the same rule,
//    extended to this table.)
//    The query API is shaped to enforce this: it exposes arm totals and
//    mechanism-condition verdicts, never a "net gain" function.
//
// 2. UNITS ARE pcyc (the C15 processor-cycle counter), never us. The LWP
//    instrumented clock runs ~1.10 GHz while the steady state runs 2.08+ GHz
//    (AGENTS.md §2 power-voting entry), so microsecond figures do not
//    transfer between the two; cycle counts of throughput-bound leaf work
//    do (validated: the standalone leaf prices at 2.11 GHz match the
//    in-kernel LWP prices within 7%, logs/hmx/lwp-region-attribution-2026-
//    09-27.log "WHAT THIS SETTLES"). Whole-kernel cycle counts at different
//    clocks are NOT comparable (memory latency becomes more cycles at a
//    higher clock); per-leaf prices are approximately clock-insensitive.
//    Arm anchors measured in us are same-build A/B pairs only -- their RATIO
//    is the datum, and the test converts them with the steady clock only for
//    order-of-magnitude sanity, cross-build, never as an A/B.
//
// 3. RECALIBRATION TRIGGERS. A price cell is valid for the build fingerprint
//    in its citation. Re-measure a cell when (a) the leaf symbol family it
//    prices changes (HMXLayout.c / HMXAPI.c), or (b) before any decision
//    consumes it on a different build fingerprint than the citation's. The
//    standalone leaf prices were measured on the 958cc99c lineage
//    (2026-09-21); the current libhmxapi.a f42384f2 has had no leaf-level
//    re-measurement -- treat them as lineage prices, not current-build
//    prices, until re-measured.
//
// 4. DIAGNOSTIC ONLY. No pass may branch on these numbers until W2 passes
//    its own gate. The one thing this table is allowed to answer today is
//    "does the model reproduce the two measured flip points" -- see
//    test_hmx_leaf_cost_table.py, which is the acceptance test W1 named.
//
// === THE TWO FLIP POINTS (the acceptance anchors) ===
//
// Flip 1 -- serial vs staged, the transfer channel. The mechanism the
// measurements actually support is ACTIVATION SOURCE HEAT, not K depth: the
// serial ring packs straight from DDR (priced hot ~91-104 pcyc/unit when the
// activation fits in L2, cold ~392 pcyc/unit when it does not), while the
// staged ring DMA-copies the tile into a VTCM slot and packs from VTCM
// (~99.6 pcyc/unit) behind the engine. Staging pays when the serial source
// is COLD (the price drop must amortize the ring's fixed overhead, measured
// +9 us / S1 and +6 us / S3 on hot shapes), and loses when it is HOT.
// Measured endpoints (docs/results/kstage-floor-pinned-2026-10-02.md, build
// ce26015e, N=1000, 3 reps): S1 1024x512x64 (Kt=2, act 128 KiB = hot) serial
// 55 us vs staged 64 us -- serial wins; S3 128^3 (Kt=4, act 32 KiB = hot)
// 10 vs 16 -- serial wins; S2 256x64x2048 (Kt=64, act 1 MiB = cold) 87 vs
// 41 -- staged wins. The pass's kStageMinKTiles=32 floor flips at K=1024
// exactly as a SELECTION matter (256x256x512 -> shallow-k, 256x256x1024 ->
// staged; docs/results/t-hmx-staging-gate-dead-2026-10-02.md §4.2); its
// timing at the boundary is unmeasured, and this table's heat band
// (128 KiB, 1 MiB) contains that point (act 512 KiB), so the model returns
// "unverified" there on purpose -- the same status the pass's own comment
// gives the Kt 5..63 band.
//
// Flip 2 -- serial ring vs staged ring + vector read-out split, the read-out
// channel. The split moves the unpack loop to a consumer vector thread in
// batches of G m-tiles; the FIRST batch cannot overlap anything (pipeline
// fill), so fewer than two batches (Mt < 2*G) is pure handoff cost, and the
// hidden unpack must exceed the split's protocol cost. Measured endpoints
// (docs/results/auto-convergence-2026-10-04.md §3, build 35c61fe4 ->
// 123bd15f / libhmxapi f42384f2, iters=1000, def vs def+depth2): S1
// (Mt=32, G=4 -> 8 batches) serial ring 47 us vs staged+readout 37 us --
// readout wins; S3 (Mt=4, G=4 -> 1 batch) 3 vs 7 -- the serial ring wins.
// No shape between Mt=5 and Mt=31 has ever been run (HmxPartitionPass.cpp's
// read-out batch comment); the model flags that whole band unverified.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXLEAFCOSTTABLE_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXLEAFCOSTTABLE_H

#include <cstdint>

namespace mlir {
namespace hmx {
namespace leafcost {

//===----------------------------------------------------------------------===//
// Provenance registry. Every price cell cites one of these by id; a cell
// without a citation is a test failure (test_hmx_leaf_cost_table.py).
//===----------------------------------------------------------------------===//

enum class Src {
  LeafBwR3,      // standalone leaves, pack-after-vscatter build
  LeafBwR2,      // standalone leaves, unpack register-inner build
  LeafBwR1,      // standalone leaves, pre-optimization build (lineage only)
  LwpS1_2709,    // in-kernel LWP regions, S1, pre-K-fusion build
  LwpT1T2_0210,  // in-kernel LWP regions, S1/S2/S3, post-K-fusion build
  KstageFloor,   // serial-vs-staged arm pairs (flip 1)
  AutoConv,      // serial-ring-vs-staged+readout arm pairs (flip 2)
  GapOpSide,     // readout protocol live-counter facts
  GateDead,      // the K=1024 selection flip (BK sweep)
  FeedMt,        // DMA/feed sustained rate (negative-result entry)
  FeedBwL2,      // the L2 boundary the heat band leans on
  WarmupB,       // the once-per-launch warm-up term
  T1Pending,     // reserved: the 2026-10-08 campaign's LWP partition rerun
  T1Replica0810, // in-kernel LWP regions, 10-02-config replica arm (readout/WSR/L2 off)
  T1Default0810, // in-kernel LWP regions, default-face backfill (readout ON)
  ArchReview1009, // DERIVED rows only, never a measured slot: the review's
                  // S1 chain decomposition + World A/B bounds (2026-10-09)
};

struct SourceRef {
  Src id;
  const char *citation;
};

constexpr SourceRef kSources[] = {
    {Src::LeafBwR3,
     "exp/hmx/leaf_bw_probe/RESULTS.md round 3 (2026-09-21): standalone leaf "
     "calls, HAP pcycles at a power-voted 2.11 GHz, libhmxapi.a 958cc99c-lineage"},
    {Src::LeafBwR2,
     "exp/hmx/leaf_bw_probe/RESULTS.md round 2 (2026-09-21): bulk + "
     "register-inner (vdeal) unpack, libhmxapi.a 721102c8-lineage, same counter"},
    {Src::LeafBwR1,
     "exp/hmx/leaf_bw_probe/RESULTS.md round 1 (2026-09-21): pre-optimization "
     "leaves, libhmxapi.a 894ce6ae / libtriton 03726d09 -- LINEAGE ONLY, "
     "superseded by rounds 2-3"},
    {Src::LwpS1_2709,
     "logs/hmx/lwp-region-attribution-2026-09-27.log: S1 1024x512x64 WR=1 "
     "grid=1, C15 pcyc, median over 50 invocations, LWP-perturbed ~1.10 GHz "
     "clock (cycle counts only), libtriton d0f41c9c / libhmxapi 97af133e; "
     "pre-K-fusion engine accounting (1024 mma for Kt=2)"},
    {Src::LwpT1T2_0210,
     "logs/t1-partition-2026-10-02/lwp_{S1,S2,S3}_{wr0,wr1}_20261002_184534"
     ".json + docs/results/t1-t2-partition-remeasured-2026-10-02.md §2: "
     "in-kernel LWP region medians, C15 pcyc, libtriton 068bac60 / libhmxapi "
     "97af133e; post-K-fusion engine accounting (mma per tile = ceil(Kt/32)); "
     "region-id mapping cross-confirmed wr1 vs wr0 and against the 2026-09-27 "
     "infodump (id2=pack_act, id3=engine[wr1]/pack_weight[wr0], id4=unpack"
     "[wr1]/engine[wr0], id5=unpack[wr0])"},
    {Src::KstageFloor,
     "docs/results/kstage-floor-pinned-2026-10-02.md + logs/phase0-1-anchor-"
     "2026-10-02/depth_ab_N1000.log: depth3(=serial plain loop) vs depth2 "
     "(=staged) arm pairs, N=1000, 3 identical reps, warm_ab config=base, us "
     "(same-build A/B; the ratio is the datum), libtriton ce26015e"},
    {Src::AutoConv,
     "docs/results/auto-convergence-2026-10-04.md §3 + terminal table: def "
     "(serial ring) vs def+depth2 (staged ring + vector readout) arm pairs, "
     "iters=1000, us (same-build A/B), libtriton 35c61fe4 -> 123bd15f / "
     "libhmxapi f42384f2"},
    {Src::GapOpSide,
     "exp/hmx/gap_table/op_side/REPORT.md §4 + rows.tsv (2026-10-04): live-"
     "counter readout facts on the G4 combo arm, iters=1000, per-run "
     "PerfPcycles (78592 pcyc / 37 us = 2.124 GHz), libtriton 10a41867 / "
     "libhmxapi 97af133e"},
    {Src::GateDead,
     "docs/results/t-hmx-staging-gate-dead-2026-10-02.md §4.2: BK sweep "
     "selection flip at K=1024 (256x256x512 -> shallow-k; 256x256x1024 -> "
     "staged) -- SELECTION observations, no timing at the boundary; "
     "host-only warmup, build ce26015e-era"},
    {Src::FeedMt,
     "exp/hmx/leaf_bw_probe feed_mt arms + docs/hmx/hmx-next-round-plan.md "
     "§12.5 + AGENTS.md negative-results entry (2026-09-21): 1T feed loop "
     "sustains 96.4 GB/s rd+wr at 2.11 GHz; 2T is 1.57x SLOWER, 4T 1.50x -- "
     "the negative result that makes this the single-thread rate"},
    {Src::FeedBwL2,
     "exp/hmx/leaf_bw_probe/RESULTS.md uncertainties section: the feed_bw "
     "L2 boundary is ~1 MiB (indirect inference; the cold/hot pack split at "
     "S2's 1 MiB activation is the direct in-kernel confirmation)"},
    {Src::WarmupB,
     "docs/results/b-is-per-launch-hmx-warmup-2026-10-02.md: the once-per-"
     "launch warm-up term B ~= 15 us after the discarded-call fix (was "
     "2,624-2,880 us); once_share@1000 <= 0.106%, build ce26015e"},
    {Src::T1Pending,
     "RESERVED for logs/t1-lwp-partition-2026-10-08/ (the 2026-10-08 "
     "campaign's LWP partition rerun on the frozen build libtriton 78865e23 "
     "/ libhmxapi f42384f2). [2026-10-08 filled: 7 of the 8 reserved slots "
     "now cite T1Replica0810 / T1Default0810, each carrying its own run's "
     "fingerprint. The one cell still citing this source is the S2 unpack "
     "slot: no clean S2 unpack region exists in either arm (fused into the "
     "staged ring on the replica arm, outlined-and-LWP-skipped on the "
     "default face). Value 0 = not yet measured; the 2026-10-02 cells stay "
     "for lineage, replacing nothing]"},
    {Src::T1Replica0810,
     "logs/t1-lwp-partition-2026-10-08/ lwpreplica_{S1,S3}_wr1_20261008_015029"
     ".json (region means: pcycles / iter_count over the LWP collector's 100 "
     "records) + docs/results/t1-lwp-partition-2026-10-08.md section 2b: "
     "in-kernel LWP regions, replica of the 2026-10-02 table's exact config "
     "(readout=0 WSR=0 L2=0, WR=1), frozen build libtriton 78865e23 / "
     "libhmxapi f42384f2"},
    {Src::T1Default0810,
     "logs/t1-lwp-partition-2026-10-08/ default-face_lwp_S2_wr1_20261008_"
     "135510.json + results_default-face.json + docs/results/"
     "t1-lwp-partition-2026-10-08.md section 7: in-kernel LWP fused "
     "staged-ring region on the DEFAULT surface (readout=1 WSR=1 L2=1, "
     "WR=1), post-fix build libtriton fa79e610 (commit f16a7b0; LWP skips "
     "the readout-outlined consumer, so the fused producer region carries "
     "the GROUP handoff func.call and holds no unpack work) / libhmxapi "
     "f42384f2"},
    {Src::ArchReview1009,
     "docs/architecture/hmx-coscheduling-architecture-review-2026-10-09.md "
     "§2.2 (S1 main-chain decomposition) + §2.3 (World A/B bounds), "
     "2026-10-09: DERIVED, cross-config, never a measured slot -- the "
     "review maps the replica arm's per-leaf LWP regions (docs/results/"
     "t1-lwp-partition-2026-10-08.md §2b/§7c, build libtriton 78865e23 / "
     "libhmxapi f42384f2) onto the default face (§7a, build libtriton "
     "fa79e610 / libhmxapi f42384f2, 82,425 pcyc/iter); us figures use "
     "the default-face run's OWN implied clock 2.147 GHz; the 38 us "
     "baseline is the measured default-face S1 (§7a, N=1000/20000 ladder "
     "38/37)"},
};

//===----------------------------------------------------------------------===//
// The leaf families, mirroring the actual symbol families in
// bin/runtime/hmx/src/HMXLayout.c (pack/unpack) and HMXAPI.c (engine-side).
// A family covers all its entry points (single and _bulk): the per-unit
// prices below already abstract the call granularity, because bulk amortizes
// only the per-call fixed part (measured: fixed ~18 cyc, marginal ~84-86
// cyc/row-pair, LeafBwR2).
//===----------------------------------------------------------------------===//

enum class Leaf {
  // HVX-side layout leaves (HMXLayout.c)
  PackActF16,       // hmx_pack_act_f16 / _bulk
  PackActF32,       // hmx_pack_act_f32 / _bulk (f32 activation ABI)
  PackWeightF16,    // hmx_pack_weight_f16 / _bulk
  PackWeightF32,    // hmx_pack_weight_f32 / _bulk
  PackWeightF16T,   // hmx_pack_weight_f16_T / _bulk (transposed source)
  PackWeightF32T,   // hmx_pack_weight_f32_T / _bulk
  UnpackAccF16,     // hmx_unpack_acc_f16 / _bulk
  UnpackAccF32,     // hmx_unpack_acc_f32 / _bulk
  // tail (peeled-edge) variants of every family above
  PackActTailF16,
  PackActTailF32,
  PackWeightTailF16,
  PackWeightTailF32,
  PackWeightTailF16T,
  PackWeightTailF32T,
  UnpackAccTailF16,
  UnpackAccTailF32,
  // engine-side leaves (HMXAPI.c)
  BiasInitUnit,     // hmx_bias_init_unit_f16 (HVX-side identity bias setup)
  BiasLoad,         // hmx_bias_load_f16
  AccClear,         // hmx_acc_clear_f16 (Q6_mxclracc_hf)
  AccRead,          // hmx_acc_store_f16 (fused convert+clear read-out)
  MmaF16,           // hmx_mma_f16
  // the staging channel (hmx.stage / hmx.await -> hexagon_runtime_dma2d_start
  // / dma_wait, the 24-bit 2D UserDMA descriptors)
  DmaStage,
  DmaAwait,
};

enum class Family {
  None, // a standalone probe condition, not an in-kernel shape
  S1,   // 1024x512x64 f16 (Mt=32, Nt=16, Kt=2)
  S2,   // 256x64x2048 f16 (Mt=8, Nt=2, Kt=64)
  S3,   // 128x128x128 f16 (Mt=4, Nt=4, Kt=4)
};

enum class Cond {
  SrcVtcm,           // standalone: source already in VTCM
  SrcDdrL2Hot,       // standalone: DDR source, L2-hot
  SrcDdrColdStream,  // standalone: DDR source, 2 MiB streaming window (cold)
  SrcVtcmStrided,    // standalone: VTCM source, strided (N-split view)
  SrcDdrStridedCold, // standalone: DDR source, strided + cold
  InKernelWR1,       // in-kernel LWP region, weight resident (device pack off)
  InKernelWR0,       // in-kernel LWP region, weight packed on device
  FusedStagedRing,   // in-kernel FUSED region: stage+await+pack_act+engine
  EngineResidual,    // the small independent engine loop of a staged kernel
  Historical,        // superseded build lineage (kept, never current)
};

//===----------------------------------------------------------------------===//
// The price table. perUnitPcyc is the price of ONE unit of the family's
// natural unit: one packed crouton-column (2 KiB) for pack, one engine call
// for the blended engine price, one unpacked crouton (2 KiB) for unpack.
// regionPcyc is the whole-loop LWP median the per-unit price was derived
// over (in-kernel cells only). A cell with perUnitPcyc == 0 AND regionPcyc
// == 0 under Src::T1Pending is a reserved slot, not a zero price.
//===----------------------------------------------------------------------===//

struct LeafPriceEntry {
  Leaf leaf;
  Family family;
  Cond cond;
  double perUnitPcyc;
  int64_t unitCount;
  int64_t regionPcyc;
  Src src;
};

constexpr LeafPriceEntry kLeafPrices[] = {
    // ---- standalone leaves (Family::None), 2026-09-21 lineage ----
    //        leaf                family     cond                  perUnit unitCt region  src
    {Leaf::PackActF16,        Family::None, Cond::SrcVtcm,          99.57,     0,      0, Src::LeafBwR3},
    {Leaf::PackActF16,        Family::None, Cond::SrcDdrL2Hot,      91.34,     0,      0, Src::LeafBwR3},
    {Leaf::PackActF16,        Family::None, Cond::SrcDdrColdStream, 391.85,    0,      0, Src::LeafBwR3},
    {Leaf::PackActF16,        Family::None, Cond::SrcVtcmStrided,   194.49,    0,      0, Src::LeafBwR3},
    {Leaf::PackActF16,        Family::None, Cond::SrcDdrStridedCold, 937.75,   0,      0, Src::LeafBwR3},
    {Leaf::UnpackAccF16,      Family::None, Cond::SrcVtcm,          101.0,     0,      0, Src::LeafBwR2},
    // lineage only: pre-vscatter pack / pre-vdeal unpack, superseded
    {Leaf::PackActF16,        Family::None, Cond::Historical,       226.56,    0,      0, Src::LeafBwR1},
    {Leaf::UnpackAccF16,      Family::None, Cond::Historical,       109.59,    0,      0, Src::LeafBwR1},
    {Leaf::UnpackAccF16,      Family::None, Cond::Historical,       109.95,    0,      0, Src::LeafBwR1},

    // ---- in-kernel, S1 1024x512x64 (post-K-fusion, 2026-10-02) ----
    {Leaf::PackActF16,        Family::S1,   Cond::InKernelWR1,      103.6,    64,   6628, Src::LwpT1T2_0210},
    {Leaf::PackActF16,        Family::S1,   Cond::InKernelWR0,      103.1,    64,   6601, Src::LwpT1T2_0210},
    {Leaf::PackWeightF16,     Family::S1,   Cond::InKernelWR0,      187.8,    32,   6008, Src::LwpT1T2_0210},
    // blended engine region: 2048 calls = 1536 setup (3/output tile) + 512
    // mma (Kt=2 -> 1 batched mma per tile); the per-op split is NOT measured
    {Leaf::MmaF16,            Family::S1,   Cond::InKernelWR1,       22.2,  2048,  45445, Src::LwpT1T2_0210},
    {Leaf::MmaF16,            Family::S1,   Cond::InKernelWR0,       22.0,  2048,  45062, Src::LwpT1T2_0210},
    {Leaf::UnpackAccF16,      Family::S1,   Cond::InKernelWR1,       86.0,   512,  44045, Src::LwpT1T2_0210},
    {Leaf::UnpackAccF16,      Family::S1,   Cond::InKernelWR0,       87.1,   512,  44612, Src::LwpT1T2_0210},

    // ---- in-kernel, S2 256x64x2048 (staged shape; WR=1) ----
    // The staged ring fuses stage+await+pack_act+engine into ONE LWP region
    // -- no per-leaf decomposition exists. The unpack region and its price
    // are cross-confirmed by the WR=0 run (3071 vs 3063 pcyc).
    {Leaf::PackActF16,        Family::S2,   Cond::FusedStagedRing,     0.0,     0,  50834, Src::LwpT1T2_0210},
    {Leaf::MmaF16,            Family::S2,   Cond::EngineResidual,      0.0,     0,    315, Src::LwpT1T2_0210},
    {Leaf::UnpackAccF16,      Family::S2,   Cond::InKernelWR1,      191.9,    16,   3071, Src::LwpT1T2_0210},
    // S2 WR=0: region-id pairing in the raw JSON is ambiguous for the fused
    // and pack_weight loops -- carry the doc's percentages only, no absolute
    // price cell here (docs/results/t1-t2-partition-remeasured-2026-10-02.md
    // §2: fused 46.60% / pack_weight 20.43% / engine 0.30% / unpack 2.81%).

    // ---- in-kernel, S3 128x128x128 (post-K-fusion, 2026-10-02) ----
    {Leaf::PackActF16,        Family::S3,   Cond::InKernelWR1,       96.1,    16,   1538, Src::LwpT1T2_0210},
    {Leaf::PackActF16,        Family::S3,   Cond::InKernelWR0,       95.9,    16,   1534, Src::LwpT1T2_0210},
    {Leaf::PackWeightF16,     Family::S3,   Cond::InKernelWR0,      179.9,    16,   2878, Src::LwpT1T2_0210},
    {Leaf::MmaF16,            Family::S3,   Cond::InKernelWR1,       26.4,    64,   1692, Src::LwpT1T2_0210},
    {Leaf::MmaF16,            Family::S3,   Cond::InKernelWR0,       26.6,    64,   1704, Src::LwpT1T2_0210},
    {Leaf::UnpackAccF16,      Family::S3,   Cond::InKernelWR1,      124.2,    16,   1987, Src::LwpT1T2_0210},
    {Leaf::UnpackAccF16,      Family::S3,   Cond::InKernelWR0,      123.6,    16,   1977, Src::LwpT1T2_0210},

    // ---- lineage: S1 pre-K-fusion (2026-09-27) ----
    // 2560 calls = 1536 setup + 1024 mma (one mma per (m,n,k)); the engine
    // region shrank 57856 -> 45445 pcyc when K-fusion (2026-10-01) halved
    // the mma count. Both cells kept: they are the evidence for the
    // engine-call unit formula below.
    {Leaf::PackActF16,        Family::S1,   Cond::Historical,       103.3,    64,   6614, Src::LwpS1_2709},
    {Leaf::MmaF16,            Family::S1,   Cond::Historical,        22.6,  2560,  57856, Src::LwpS1_2709},
    {Leaf::UnpackAccF16,      Family::S1,   Cond::Historical,        87.4,   512,  44731, Src::LwpS1_2709},

    // ---- the 2026-10-08 campaign's LWP partition rerun (filled 2026-10-08) ----
    // Two arms in logs/t1-lwp-partition-2026-10-08/: the 10-02-config replica
    // (readout=0 WSR=0 L2=0, frozen build 78865e23) carries the clean WR=1
    // regions for S1/S3; the post-f16a7b0 default-face backfill (readout=1
    // WSR=1 L2=1, build fa79e610) carries the S2 fused region -- its ops hold
    // no unpack_acc (readout outlined the consumer), so it matches the
    // FusedStagedRing cond exactly. The S2 unpack slot stays pending: neither
    // arm yields a clean S2 unpack region (fused on the replica,
    // outlined-and-skipped on the default face). Region means = pcycles /
    // iter_count over 100 collector records.
    //        leaf                family     cond                  perUnit unitCt region  src
    {Leaf::PackActF16,        Family::S1,   Cond::InKernelWR1,      114.3,    64,   7312, Src::T1Replica0810},
    {Leaf::MmaF16,            Family::S1,   Cond::InKernelWR1,       22.0,  2048,  45116, Src::T1Replica0810},
    {Leaf::UnpackAccF16,      Family::S1,   Cond::InKernelWR1,       87.3,   512,  44675, Src::T1Replica0810},
    {Leaf::PackActF16,        Family::S2,   Cond::FusedStagedRing,     0.0,     0,  51235, Src::T1Default0810},
    {Leaf::UnpackAccF16,      Family::S2,   Cond::InKernelWR1,         0.0,     0,      0, Src::T1Pending},
    {Leaf::PackActF16,        Family::S3,   Cond::InKernelWR1,      103.9,    16,   1663, Src::T1Replica0810},
    {Leaf::MmaF16,            Family::S3,   Cond::InKernelWR1,       26.3,    64,   1684, Src::T1Replica0810},
    {Leaf::UnpackAccF16,      Family::S3,   Cond::InKernelWR1,      126.0,    16,   2016, Src::T1Replica0810},
};

//===----------------------------------------------------------------------===//
// The empty cells, enumerated instead of left silent. "无数据" is a legal
// table state; a missing row is not. Adding a measurement for one of these
// families means moving it from this list into kLeafPrices with a citation.
//===----------------------------------------------------------------------===//

struct NoDataEntry {
  Leaf leaf;
  const char *note;
};

constexpr NoDataEntry kNoData[] = {
    {Leaf::PackActF32,
     "FA/KDA run the f32 activation ABI, but no leaf-level or LWP-region "
     "measurement of hmx_pack_act_f32 exists"},
    {Leaf::PackWeightF32, "no measurement of hmx_pack_weight_f32 anywhere"},
    {Leaf::PackWeightF16T,
     "the _T (transposed-source) weight pack variants have no standalone or "
     "in-kernel cell; the in-kernel WR=0 cells above are the non-T family"},
    {Leaf::PackWeightF32T, "no measurement"},
    {Leaf::PackActTailF16,
     "the peeled-edge tail leaves (hmx_pack_*_tail_*) have never been "
     "benchmarked or LWP-attributed"},
    {Leaf::PackActTailF32, "no measurement"},
    {Leaf::PackWeightTailF16, "no measurement"},
    {Leaf::PackWeightTailF32, "no measurement"},
    {Leaf::PackWeightTailF16T, "no measurement"},
    {Leaf::PackWeightTailF32T, "no measurement"},
    {Leaf::UnpackAccF32,
     "no leaf-level or in-kernel measurement of hmx_unpack_acc_f32 (the f32 "
     "read-out path used by FA-style f32 outputs)"},
    {Leaf::UnpackAccTailF16, "no measurement"},
    {Leaf::UnpackAccTailF32, "no measurement"},
    {Leaf::BiasInitUnit,
     "only the BLENDED in-kernel engine-region price exists; the per-op "
     "split (bias init / load / clear / read / mma) was never separated"},
    {Leaf::BiasLoad, "blended only, see the engine-region cells"},
    {Leaf::AccClear, "blended only, see the engine-region cells"},
    {Leaf::AccRead, "blended only, see the engine-region cells"},
    {Leaf::MmaF16,
     "standalone (engine-only tight loop) mma price is a KNOWN GAP -- the "
     "gap table marks the theoretical engine peak as 'standard missing'; "
     "only the blended in-kernel region price exists"},
    {Leaf::DmaStage,
     "no per-call fixed cost for hmx.stage exists; only the sustained feed "
     "rate (kDmaRdWrBytesPerPcyc below) and the measured staged-ring deltas"},
    {Leaf::DmaAwait,
     "no per-call cost; the await's cost is inside the fused staged-ring "
     "region and the measured ring fixed overhead"},
};

//===----------------------------------------------------------------------===//
// Measured pipeline-arm anchors. us figures are same-build A/B pairs (the
// ratio is the datum); pcyc is carried only when the source log recorded
// PerfPcycles for that arm.
//===----------------------------------------------------------------------===//

enum class Arm {
  SerialPlain,     // pipeline-depth=3: the plain tile loop, no ring
  StagedNoReadout, // pipeline-depth=2: staged ring, read-out in-thread
  SerialRingDef,   // zero-knob defaults, read-out declined (serial ring)
  StagedReadout,   // staged ring + vector read-out split (def+depth2)
  ComboG4Dw,       // readout G4 + deferred drain + workspace resident
};

struct ArmAnchor {
  Family family;
  Arm arm;
  int64_t us;
  int64_t pcyc;
  Src src;
};

constexpr ArmAnchor kArmAnchors[] = {
    // flip 1 endpoints (kstage-floor-pinned, build ce26015e, N=1000, 3 reps)
    {Family::S1, Arm::SerialPlain,     55,     0, Src::KstageFloor},
    {Family::S1, Arm::StagedNoReadout, 64,     0, Src::KstageFloor},
    {Family::S2, Arm::SerialPlain,     87,     0, Src::KstageFloor},
    {Family::S2, Arm::StagedNoReadout, 41,     0, Src::KstageFloor},
    {Family::S3, Arm::SerialPlain,     10,     0, Src::KstageFloor},
    {Family::S3, Arm::StagedNoReadout, 16,     0, Src::KstageFloor},
    // flip 2 endpoints (auto-convergence, build 35c61fe4 -> 123bd15f)
    {Family::S1, Arm::SerialRingDef,   47,     0, Src::AutoConv},
    {Family::S1, Arm::StagedReadout,   37, 78592, Src::GapOpSide},
    {Family::S3, Arm::SerialRingDef,    3,     0, Src::AutoConv},
    {Family::S3, Arm::StagedReadout,    7,     0, Src::AutoConv},
    // context anchors (gap table, build 10a41867, iters=1000)
    {Family::S1, Arm::ComboG4Dw,       37, 78592, Src::GapOpSide},
};

enum class Proto {
  ConsumerBusyPerBatchUs,   // median busy time of the consumer per batch
  ConsumerBusyPerBatchPcyc, // the same, in the run's own pcyc
  PublishToStartUs,         // publish -> consumer starts working
  LastDrainDeferredUs,      // last drain wait, deferred-drain on
  LastDrainNonDeferredUs,   // last drain wait, deferred-drain off (the def arm)
  OncePerLaunchWarmupUs,    // the harness warm-up term (context, not a leaf)
};

struct ProtocolFact {
  Proto fact;
  double value;
  Src src;
};

constexpr ProtocolFact kReadoutProtocol[] = {
    {Proto::ConsumerBusyPerBatchUs,   2.667, Src::GapOpSide},
    {Proto::ConsumerBusyPerBatchPcyc, 5676,  Src::GapOpSide},
    {Proto::PublishToStartUs,         0.034, Src::GapOpSide},
    {Proto::LastDrainDeferredUs,      0.08,  Src::GapOpSide},
    {Proto::LastDrainNonDeferredUs,   2.54,  Src::GapOpSide},
    {Proto::OncePerLaunchWarmupUs,    15.0,  Src::WarmupB},
};

//===----------------------------------------------------------------------===//
// The measured flip endpoints -- the acceptance anchors. `measured` is what
// the device said; `serialUs`/`otherUs` are the two arms' times. The verdict
// queries below must reproduce every one of these, and must flag everything
// else unverified.
//===----------------------------------------------------------------------===//

enum class Flip { StagedVsSerial, ReadoutVsSerialRing };
enum class Winner { Serial, Staged, Readout };

struct FlipAnchor {
  Flip flip;
  Family family;
  int64_t m, n, k;
  int64_t batch; // read-out batch G; 0 for the transfer channel
  Winner measured;
  int64_t serialUs;
  int64_t otherUs;
  Src src;
};

constexpr FlipAnchor kFlipAnchors[] = {
    {Flip::StagedVsSerial,      Family::S1, 1024,  512,   64, 0, Winner::Serial, 55, 64, Src::KstageFloor},
    {Flip::StagedVsSerial,      Family::S2,  256,   64, 2048, 0, Winner::Staged, 87, 41, Src::KstageFloor},
    {Flip::StagedVsSerial,      Family::S3,  128,  128,  128, 0, Winner::Serial, 10, 16, Src::KstageFloor},
    {Flip::ReadoutVsSerialRing, Family::S1, 1024,  512,   64, 4, Winner::Readout, 47, 37, Src::AutoConv},
    {Flip::ReadoutVsSerialRing, Family::S3,  128,  128,  128, 4, Winner::Serial,   3,  7, Src::AutoConv},
};

//===----------------------------------------------------------------------===//
// Model scalars. Each carries the traceability triple the project requires
// (ROADMAP §2.1a); the test enforces their presence.
//===----------------------------------------------------------------------===//

// TRACEABILITY: kDmaRdWrBytesPerPcyc
//   mechanism: the sustained rd+wr byte rate of the single-thread DDR->VTCM
//     feed path, converted to bytes per C15 cycle at the measured 2.11 GHz
//     clock. A copy of B bytes costs 2*B of that budget (read plus write),
//     so dmaCopyPcyc(B) = 2*B / kDmaRdWrBytesPerPcyc. This prices the
//     hmx.stage DMA inside the staged ring.
//   measurement: feed_mt 1T = 96.4 GB/s rd+wr at 2.11 GHz
//     (exp/hmx/leaf_bw_probe feed_mt arms, 2026-09-21, libhmxapi 7c470647
//     lineage; the multi-thread arms are 1.5-1.6x SLOWER, so 1T is the
//     ceiling, not a minimum). 96.4 / 2.11 = 45.7.
//   shape set: the probe's own feed shapes (bulk pack rows); applied here to
//     the staged ring's per-m-tile slabs, which have not been timed as bare
//     DMA copies -- a calibration gap, covered by the fused-region
//     consistency check in the test.
//   workload representativeness: PARTIAL. The rate is a standalone feed-loop
//     rate; in-kernel the DMA shares the fabric with the engine. The S2
//     fused-region cross-check lands within 15%, which is the current
//     evidence that the number is not wildly off in situ.
constexpr double kDmaRdWrBytesPerPcyc = 45.7;

// TRACEABILITY: kHotActBytes
//   mechanism: the largest activation footprint measured to pack at the HOT
//     price. Below it the serial ring's pack reads L2-hot DDR and staging
//     has no source-heat gain to offer (the staged VTCM-source price, 99.57,
//     is the same as the hot DDR price), so the ring's fixed overhead makes
//     staging a pure loss.
//   measurement: S1 (1024x512x64, act 128 KiB) packs at 103.6 pcyc/unit
//     in-kernel (LwpT1T2_0210) against 91.3 standalone-hot -- hot; S3 (act
//     32 KiB) at 96.1 -- hot. No shape between 128 KiB and 1 MiB has a
//     timed serial-vs-staged pair.
//   shape set: S1 and S3 only; everything above 128 KiB up to 1 MiB is the
//     UNVERIFIED heat band (this includes the gate's own K=1024 flip shape,
//     256x256x1024 at act 512 KiB, whose selection was observed but whose
//     timing was not).
//   workload representativeness: the band boundary is an open hole, not a
//     tuned threshold -- do not read 128 KiB as "L2 size".
constexpr int64_t kHotActBytes = 131072;

// TRACEABILITY: kColdActBytes
//   mechanism: the smallest activation footprint measured to pack at the
//     COLD price. At or above it the serial ring's pack streams from DDR at
//     ~392 pcyc/unit while the staged ring's DMA + VTCM-source pack runs at
//     ~100 pcyc/unit behind the engine, which is the entire mechanism by
//     which staging wins.
//   measurement: S2 (256x64x2048, act 1 MiB) -- serial arm 87 us vs staged
//     41 us (kstage-floor-pinned, build ce26015e); the cold standalone price
//     391.85 pcyc/unit reproduces the serial arm within 12% upper-bound.
//     The feed_bw L2 boundary ~1 MiB (FeedBwL2) agrees.
//   shape set: ONE shape (S2). The band (128 KiB, 1 MiB) is unmeasured on
//     both sides.
//   workload representativeness: single-ended; the FFN/PV shapes the
//     backend targets have not been swept across this boundary.
constexpr int64_t kColdActBytes = 1048576;

// TRACEABILITY: kRingFixedOverheadPcyc
//   mechanism: the staged ring's fixed cost that the leaf prices do not
//     cover -- ring slot allocation, the issue/await protocol, and the fill
//     and drain edges. On a HOT shape (no source-heat gain) this is the whole
//     difference between the two arms, so the hot anchors measure it
//     directly; a COLD shape stages only when its heat gain exceeds it.
//   measurement: S1 hot staged-serial delta = +9 us, S3 = +6 us, same build
//     ce26015e at the ~2.12 GHz steady clock (kstage-floor-pinned) => 19.1k
//     and 12.7k pcyc. The larger (S1's) is carried as the bound.
//   shape set: the two hot anchors; the per-shape spread (12.7k-19.1k) is
//     itself unexplained -- treat the value as an order, not a constant.
//   workload representativeness: two shapes; the term is known FIXED-ish
//     (per launch, not per unit: S1 spread it over 64 pack units, S3 over
//     16, and the two totals agree better than the per-unit figures would).
constexpr int64_t kRingFixedOverheadPcyc = 19000;

// TRACEABILITY: kReadoutFixedPcyc
//   mechanism: the read-out split's protocol cost that the leaf prices do
//     not cover -- per-batch publish/handoff, the un-overlapped consumer
//     tail, and the final drain wait. The split only pays when the unpack it
//     hides exceeds this.
//   measurement: DERIVED, single-ended, from the S1 positive endpoint: the
//     arm's own recorded 78,592 pcyc (37 us, PerfPcycles, the GapOpSide G4
//     combo arm; the flip-2 A/B's defdd arm read the same 37 us on its own
//     build 123bd15f) minus the modeled S1 staged producer (~52.0k pcyc,
//     from this table's prices) = ~26.6k. The S1 endpoint is therefore the
//     CALIBRATION point, not an independent check; S3 (Mt=4, one batch,
//     pure handoff) is the independent negative check.
//   shape set: S1 with batch G=4 (8 batches). The protocol facts it rests
//     on (consumer busy 2.667 us/batch, publish->start 0.034 us, last drain
//     2.54 us non-deferred) are the GapOpSide G4-combo arm.
//   workload representativeness: one endpoint; no shape between Mt=5 and
//     Mt=31 has ever been run (the pass's own read-out batch comment says
//     so), so every verdict in that band is flagged unverified.
constexpr int64_t kReadoutFixedPcyc = 26600;

// TRACEABILITY: kSteadyGhz
//   mechanism: the steady-state core clock of the power-voted harness runs,
//     used ONLY to convert us arm anchors into pcyc for order-of-magnitude
//     sanity checks. It is never an A/B instrument: same-build arm pairs are
//     compared as ratios, and each real measurement's own PerfPcycles (when
//     recorded) outrank it.
//   measurement: gap table rows.tsv matmul arms 2.12-2.15 GHz
//     (libtriton 10a41867, iters=1000); t1-partition perf logs 2.12-2.21
//     (libtriton 068bac60). The LWP-instrumented clock is ~1.10 GHz and is
//     NEVER converted with this constant (contract note 2).
//   shape set: matmul-class kernels under the power-voted harness.
//   workload representativeness: good for matmul-class steady state; the
//     known slow-window phenomena (AGENTS §5) sit outside it.
constexpr double kSteadyGhz = 2.12;

// TRACEABILITY: kReadoutBatch
//   mechanism: the read-out GROUP batch G -- the number of m-tiles the
//     producer publishes per batch. The flip-2 mechanism condition is
//     expressed in batches: the first batch cannot overlap anything, so
//     fewer than two batches is pure handoff. NOT a new tuning constant: it
//     mirrors hmxReadoutBatch's default (hexagon_options.py), and the pass
//     wires stagedReadoutMTiles = 2 x batch from the same value.
//   measurement: calibrated as part of the G4 combo arm (GapOpSide); the
//     flip-2 endpoints (S1 Mt=32, S3 Mt=4) were measured with G=4.
//   shape set: the flip-2 anchors; other batch values have no endpoints.
//   workload representativeness: single value, default-owned elsewhere --
//     this header only mirrors it for the mechanism condition.
constexpr int64_t kReadoutBatch = 4;

//===----------------------------------------------------------------------===//
// Family price bundles -- the per-family inputs the queries consume. S2's
// packActHotPerUnit has no in-kernel serial cell (the S2 LWP run was
// staged), so the standalone hot price stands in, and S2's engine price is
// borrowed from S1 (both noted; the engine is a small term for S2, ~1.8k of
// ~205k pcyc serial). S2's staged producer is the MEASURED fused region --
// no modeled stand-in.
//===----------------------------------------------------------------------===//

struct FamilyPrices {
  Family family;
  int64_t m, n, k;             // the family's anchor grid
  double packActHotPerUnit;    // serial-ring pack price when the source is hot
  double packActVtcmPerUnit;   // the staged ring's pack price (VTCM source)
  double packActColdPerUnit;   // serial-ring pack price when the source is cold
  double enginePerCall;        // blended in-kernel engine price
  double unpackPerUnit;        // in-kernel unpack price
  int64_t stagedProducerPcyc;  // >0 = measured fused region; 0 = model it
  Src src;
};

constexpr FamilyPrices kFamilyPrices[] = {
    //        family      m     n     k     hot   vtcm   cold  engine unpack  fused  src
    {Family::S1,       1024,  512,   64,  103.6,  99.57, 391.85,  22.2,  86.0,      0, Src::LwpT1T2_0210},
    {Family::S2,        256,   64, 2048,  91.34,  99.57, 391.85,  22.2, 191.9,  50834, Src::LwpT1T2_0210},
    {Family::S3,        128,  128,  128,   96.1,  99.57, 391.85,  26.4, 124.2,      0, Src::LwpT1T2_0210},
};

//===----------------------------------------------------------------------===//
// Queries. Unit counts first, then the arm models, then the two flip
// verdicts, then modeled_ii. All arithmetic is deliberately trivial (sums,
// max, one division) so the Python mirror in test_hmx_leaf_cost_table.py
// stays line-for-line comparable. 32 is layout::kTileEdge
// (HmxCroutonLayout.h); it is written as a literal to keep this header
// parseable standalone.
//===----------------------------------------------------------------------===//

struct Grid {
  int64_t m = 0, n = 0, k = 0;
};

inline int64_t tiles(int64_t extent) { return extent / 32; }

/// One pack unit per (m-tile, k-tile) crouton-column; one unpack unit per
/// (m-tile, n-tile) output crouton. Engine calls: 3 setup + ceil(Kt/32)
/// batched mma per output tile (the K-fusion accounting; the 09-27 lineage
/// row in kLeafPrices is the pre-fusion count, kept as its evidence).
inline int64_t packActUnitCount(Grid g) { return tiles(g.m) * tiles(g.k); }
inline int64_t unpackUnitCount(Grid g) { return tiles(g.m) * tiles(g.n); }
inline int64_t engineCallCount(Grid g) {
  return tiles(g.m) * tiles(g.n) * (3 + (tiles(g.k) + 31) / 32);
}
inline int64_t actSrcBytes(Grid g) { return g.m * g.k * 2; }

/// A DMA copy of `bytes` costs 2*bytes of the rd+wr budget.
inline int64_t dmaCopyPcyc(int64_t bytes) {
  return (int64_t)((2.0 * (double)bytes) / kDmaRdWrBytesPerPcyc);
}

enum class Heat { Hot, Cold, Band };

inline Heat activationHeat(Grid g) {
  if (actSrcBytes(g) <= kHotActBytes)
    return Heat::Hot;
  if (actSrcBytes(g) >= kColdActBytes)
    return Heat::Cold;
  return Heat::Band; // the unmeasured (128 KiB, 1 MiB) band
}

inline const FamilyPrices *familyPrices(Family f) {
  for (const auto &fp : kFamilyPrices)
    if (fp.family == f)
      return &fp;
  return nullptr;
}

/// The serial arm's producer side (pack + engine), no overlap credit.
inline int64_t modeledSerialProducer(Grid g, const FamilyPrices &fp,
                                      Heat heat) {
  double pack = heat == Heat::Cold ? fp.packActColdPerUnit
                                   : fp.packActHotPerUnit;
  return (int64_t)(pack * (double)packActUnitCount(g) +
                   fp.enginePerCall * (double)engineCallCount(g));
}

/// The staged ring's producer side: one fill DMA, then per m-tile
/// max(DMA, VTCM-pack + engine) -- the DMA is hidden whenever it is the
/// smaller term. S2 carries the MEASURED fused region instead of the model.
inline int64_t modeledStagedProducer(Grid g, const FamilyPrices &fp) {
  if (fp.stagedProducerPcyc > 0)
    return fp.stagedProducerPcyc;
  int64_t perTile = (int64_t)(
      fp.packActVtcmPerUnit * (double)tiles(g.k) +
      fp.enginePerCall *
          (double)(tiles(g.n) * (3 + (tiles(g.k) + 31) / 32)));
  int64_t dma = dmaCopyPcyc(32 * g.k * 2);
  return dma + tiles(g.m) * (perTile > dma ? perTile : dma);
}

inline int64_t modeledUnpack(Grid g, const FamilyPrices &fp) {
  return (int64_t)(fp.unpackPerUnit * (double)unpackUnitCount(g));
}

/// The serial arm's whole modeled region sum (upper bound; contract note 1).
inline int64_t modeledSerialTotal(Grid g, const FamilyPrices &fp, Heat heat) {
  return modeledSerialProducer(g, fp, heat) + modeledUnpack(g, fp);
}

/// Whether (flip, grid, batch, wins) is one of the MEASURED endpoints.
inline bool isMeasuredEndpoint(Flip flip, Grid g, int64_t batch, bool wins) {
  for (const auto &a : kFlipAnchors) {
    if (a.flip != flip || a.m != g.m || a.n != g.n || a.k != g.k)
      continue;
    if (flip == Flip::ReadoutVsSerialRing && a.batch != batch)
      continue;
    if ((a.measured != Winner::Serial) == wins)
      return true;
  }
  return false;
}

struct FlipVerdict {
  bool wins;      // the model's pick
  bool verified;  // true ONLY at a measured endpoint
  const char *note;
};

/// Flip 1: does the staged ring beat the serial plain loop on this grid?
///
/// Hot: no -- the pack price is unchanged (VTCM source ~= hot DDR source)
/// and the ring's fixed overhead is a pure loss (measured at both hot
/// anchors). Cold: yes when the heat gain (cold - VTCM price, over all pack
/// units) exceeds the ring's fixed overhead. Band: no verdict.
inline FlipVerdict stagedBeatsSerial(Grid g, const FamilyPrices &fp) {
  Heat heat = activationHeat(g);
  if (heat == Heat::Band)
    return {false, false,
            "activation bytes fall in the unmeasured heat band "
            "(128 KiB, 1 MiB) -- no verdict"};
  if (heat == Heat::Hot) {
    bool verified = isMeasuredEndpoint(Flip::StagedVsSerial, g, 0, false);
    return {false, verified,
            "hot source: staged pays the same pack plus the ring's fixed "
            "overhead (kRingFixedOverheadPcyc) -- serial wins"};
  }
  int64_t gain = (int64_t)((fp.packActColdPerUnit - fp.packActVtcmPerUnit) *
                           (double)packActUnitCount(g));
  if (gain <= kRingFixedOverheadPcyc)
    return {false, false,
            "cold source but too few pack units to amortize the ring's "
            "fixed overhead -- no measured endpoint"};
  bool verified = isMeasuredEndpoint(Flip::StagedVsSerial, g, 0, true);
  return {true, verified,
          "cold source: the pack price drop (cold -> VTCM) exceeds the "
          "ring's fixed overhead -- staged wins"};
}

/// Flip 2: does the staged ring + vector read-out split beat the serial
/// ring? The split hides the unpack behind the producer only from the
/// second batch on, and only pays when the hidden unpack exceeds the
/// split's protocol cost.
inline FlipVerdict readoutBeatsSerialRing(Grid g, const FamilyPrices &fp,
                                           int64_t batch) {
  int64_t batches = (tiles(g.m) + batch - 1) / batch;
  if (batches < 2) {
    bool verified = isMeasuredEndpoint(Flip::ReadoutVsSerialRing, g, batch,
                                       false);
    return {false, verified,
            "fewer than two batches: the first batch cannot overlap "
            "anything, the split is pure handoff -- serial wins"};
  }
  int64_t hidden = modeledUnpack(g, fp);
  if (hidden <= kReadoutFixedPcyc)
    return {false, false,
            "hidden unpack at or below the split's protocol cost "
            "(kReadoutFixedPcyc) -- no measured endpoint"};
  bool verified =
      isMeasuredEndpoint(Flip::ReadoutVsSerialRing, g, batch, true);
  return {true, verified,
          "two or more batches and the hidden unpack exceeds the split's "
          "protocol cost -- read-out wins"};
}

//===----------------------------------------------------------------------===//
// modeled_ii -- the per-region table-lookup value the manifest spec below
// publishes. Region totals in pcyc, upper-bound caliber (contract note 1).
//===----------------------------------------------------------------------===//

struct ModeledII {
  int64_t packActPcyc;
  int64_t enginePcyc;
  int64_t unpackPcyc;
  int64_t totalPcyc;
  const char *caliber; // "upper-bound-leaf-sum"
};

inline ModeledII modeledII(Grid g, const FamilyPrices &fp) {
  int64_t pack = (int64_t)(fp.packActHotPerUnit * (double)packActUnitCount(g));
  int64_t engine = (int64_t)(fp.enginePerCall * (double)engineCallCount(g));
  int64_t unpack = modeledUnpack(g, fp);
  return {pack, engine, unpack, pack + engine + unpack,
          "upper-bound-leaf-sum"};
}

//===----------------------------------------------------------------------===//
// DERIVED entries -- the R1 window's reading frame (W1 close-out,
// 2026-10-09). NOT measured slots: every row below is an INFERENCE the
// co-scheduling architecture review composed FROM measured cells, carried
// in its own structures so a derived row can never be mistakable for a
// kLeafPrices / kArmAnchors measurement. Nothing computes with these yet;
// their one consumer is the R1 window (review §5) -- the experiment that
// decides which world S1 lives in. Until it runs, every number here is a
// conditional bound, not a prediction.
//
// Contract note 1 is unchanged and still governs: nothing below is an
// arm-total subtraction, and the subtraction the note forbids stays
// forbidden. These rows are mechanism bounds under a physics question the
// existing data cannot answer (see kWorldBounds).
//
// mechanism: the two sub-sections below -- the replica-to-default mapping
//   of the chain decomposition, and the engine-wait vs issue-saturated
//   question the R1 window adjudicates.
// measurement: none of these numbers is one. The inputs are the
//   T1Replica0810 / T1Default0810 cells above and the review's arithmetic
//   over them; Src::ArchReview1009 carries the full evidence chain.
// shape set: S1 1024x512x64 WR=1 default face ONLY -- the one shape the
//   review decomposed; no other family has a derived row.
// workload representativeness: the review's own caveat -- World A/B is
//   UNDECIDED, so both bounds below are conditional, not predicted.
//===----------------------------------------------------------------------===//

// --- The S1 default-face main-chain decomposition (review §2.2) ---
//
// The default face's S1 producer is ONE fused LWP region plus a small
// independent engine residual and a function-level residual (76.41% +
// 2.06% + 21.54% of 82,425 pcyc/iter, T1 §7a, build fa79e610) -- no clean
// per-leaf region exists on that face. The review mapped the REPLICA
// arm's per-leaf regions onto it instead: the leaf set is a shape
// property (form-independent), but the mapping is CROSS-CONFIG, hence the
// inferred caliber on every row. "glue" is the fused region's remainder
// once pack_act and engine are placed (stage/await + the readout GROUP
// handoff + loop glue); "residual" is the function-level share (entry and
// exit, everything outside the pipeline regions); "unpack_concurrent" is
// the vector thread's side and is NOT part of the main-chain sum.
//
// The review's closure check: fused_total + residual = 29.3 + 8.3 = 37.6
// us ~= the measured 38 us default face (fa79e610, N=1000/20000 ladder
// 38/37); the gap is the independent engine residual region (~1.7k pcyc,
// 2.06%), which the five main-chain rows deliberately leave outside the
// fused total. Every us figure below uses the default-face run's OWN
// implied clock (2.147 GHz) -- never kSteadyGhz, never an A/B instrument
// (contract note 2).
struct ChainRow {
  const char *component; // "pack_act" | "engine" | "glue" | "fused_total"
                         // | "residual" | "unpack_concurrent"
  int64_t pcyc;
  double us;             // at the default-face run's own 2.147 GHz
  const char *caliber;   // "derived-inferred" -- never a measured slot
  Src src;               // the derivation's provenance (the review)
};

constexpr ChainRow kS1DefaultChain[] = {
    {"pack_act", 7312, 3.4, "derived-inferred", Src::ArchReview1009},
    {"engine", 45116, 21.0, "derived-inferred", Src::ArchReview1009},
    {"glue", 10532, 4.9, "derived-inferred", Src::ArchReview1009},
    {"fused_total", 62960, 29.3, "derived-inferred", Src::ArchReview1009},
    {"residual", 17756, 8.3, "derived-inferred", Src::ArchReview1009},
    // concurrent vector-thread side (replica-arm caliber), NOT a
    // main-chain row: the third input of the World B three-thread bound.
    {"unpack_concurrent", 44675, 20.8, "derived-inferred",
     Src::ArchReview1009},
};

// --- The two-world bounds (review §2.3) -- the undecided question ---
//
// The ~21 us engine share of the S1 main chain is one of two PHYSICS, and
// the existing data cannot tell them apart: the v81 PRM does not say
// whether issue continues while the engine is busy (review §2.4), so both
// worlds are consistent with every measurement so far.
//
//   World A (engine-wait): acc_read stalls the CPU while the engine
//     computes; the window holds ~21 us of CPU idle, more than the 8.3 us
//     of pack+DMA+glue. A SINGLE-THREADED rearrangement (T11-style pack
//     hoisting / ping-pong scratch) reaches the bound -- no thread
//     topology. Bound ~= engine + residual ~= 29.3 us, -22% vs 38.
//   World B (issue-saturated): 22 pcyc/leaf ~= the pure issue stake; the
//     CPU never idles, there is no window to fill. Only moving the engine
//     issue stream to T_HMX cuts the main chain: the 3-thread bound ~=
//     max(engine 21.0, unpack 20.8, main chain 16.6) ~= 21-26 us,
//     -30~-45% vs 38.
//
// Evidence leaned both ways (toward A: S2's readout on/off arms measured
// equal, A=C=32.0 us -- unpack issue once hid inside an engine busy
// window; toward B: 22 pcyc/leaf sits near the pure-issue estimate, and
// S1 carries ~2x the engine leaves per unit of engine work that S2 does
// -- the issue-dense vs engine-waiting shape split, review §2.2). THE
// R1 WINDOW DECIDED IT (docs/results/s1-window-2026-10-09.md,
// logs/s1-window-2026-10-09/): World B. The rearrangement probe did not
// gain -- it measured +1.00 us, CI [+0.32, +1.68], time and pcyc rising
// together (a THIRD signature: inserting work into the engine window
// COSTS cycles -- guard overhead plus issue-stream perturbation; the
// window is occupied by engine feeding, not empty). World A is refuted
// and its carrier is dead; the thread topology is the only carrier
// left. The four-arm also re-priced the chain: readout deferral +10.0,
// the true depth-1 debt ~12 us (P-D -- the review's ~3 us premise was
// falsified), mechanism net -4.0 (D-B).
struct WorldBoundRow {
  const char *world;   // "A" (engine-wait) | "B" (issue-saturated)
  double baselineUs;   // the measured default-face S1 anchor (T1 §7a: 38)
  double boundUsLo;    // the bound's range in us (lo == hi: a point bound)
  double boundUsHi;
  const char *carrier; // the mechanism that reaches the bound
  const char *caliber; // names the world's R1-verdict status and the
                       // adjudicating measurement
  Src src;
};

constexpr WorldBoundRow kWorldBounds[] = {
    {"A", 38.0, 29.3, 29.3,
     "single-threaded rearrangement (T11-style pack hoisting / ping-pong "
     "scratch) hides pack+DMA+glue inside the engine window -- no thread "
     "topology",
     "derived-inferred-world-REFUTED: R1 "
     "(docs/results/s1-window-2026-10-09.md) measured the rearrangement "
     "at +1.00 us, CI [+0.32, +1.68] -- inserting work into the engine "
     "window costs cycles; no single-thread window exists, this carrier "
     "is dead",
     Src::ArchReview1009},
    {"B", 38.0, 21.0, 26.0,
     "thread topology only: the engine issue stream moves to T_HMX; "
     "3-thread bound = max(engine, unpack, main chain)",
     "derived-inferred-world-CONFIRMED: R1 "
     "(docs/results/s1-window-2026-10-09.md) -- no single-thread window; "
     "four-arm: readout deferral +10.0 us, depth-1 debt ~12 us (P-D, the "
     "review's ~3 us premise falsified), mechanism net -4.0 us (D-B, the "
     "topology cuts at matched depth)",
     Src::ArchReview1009},
};

//===----------------------------------------------------------------------===//
// THE MANIFEST `modeled_ii` SPEC -- NOT IMPLEMENTED (contract wall)
//
// The decision rule W1 was given: add `modeled_ii` to the v2 manifest if the
// schema admits additive diagnostic fields, otherwise implement table +
// queries + tests only and write the wiring as a spec. The wall is real and
// double:
//
//   1. `hex.hmx.kernel_manifest/v2` is a FROZEN CLOSED boundary, enforced
//      twice and pinned by a third literal:
//        - the producer refuses unknown fields (HmxManifest.cpp's
//          `allowedFields`);
//        - the consumer accepts a record only when its key set EQUALS the
//          allowed set (backend/utils.py `_validate_record` via
//          `_require_exact_fields`);
//        - test_hmx_manifest_field_agreement.py freezes the per-plan field
//          sets (BASE_FIELDS / HMX_FIELDS / TAIL_ONLY_FIELDS) and says in
//          its own docstring: "Adding a field to the frozen v2 boundary is
//          not something this test can authorise."
//      `plan_fingerprint` additionally hashes the whole record, so a new
//      field changes every record's fingerprint, not just the schema.
//
//   2. The additive path, `hex.hmx.kernel_manifest/v3` (HmxRecordV3), is
//      FROZEN BY USER RULING (2026-10-07, ROADMAP §2.1): "v3 的授权消费者
//      落地之前，不加新字段、不加新序列化事实" -- no new fields until an
//      authorized consumer lands (or the user deletes v3). The task order
//      for W1 explicitly forbids touching it.
//
// THE WIRING, for whoever executes this when a boundary opens (v3 unfreeze
// or a deliberate, user-authorized v2 boundary change):
//
//   value: modeledII(grid, familyPrices(family)) -- {pack_act_pcyc,
//     engine_pcyc, unpack_pcyc, total_pcyc} plus "caliber":
//     "upper-bound-leaf-sum" and the table's identity (file + the price
//     citations' build fingerprints), so a reader can tell which build the
//     numbers describe.
//   producer: HmxPartitionPass, at the site that already calls
//     setHmxManifestPipelineDecision -- the grid and the family prices are
//     both in hand there; the field is per-record (one matmul = one record).
//   semantics: DIAGNOSTIC ONLY -- identical to this header's contract note
//     4. No pass, launcher, or validator may branch on it; consumers
//     subtract nothing from it (contract note 1).
//   validator: an OPTIONAL additive field -- absent means "written before
//     the field existed", exactly the precedent `budget_depth` set
//     (2026-10-02). Present-but-malformed (negative, missing keys, unknown
//     caliber) is an error, not a repair.
//   tests: the manifest contract tests gain (a) absent-field validity,
//     (b) malformed-field rejection, (c) a producer round-trip through the
//     real C++ -> Python boundary (the test_hmx_manifest_metadata.py
//     style), and (d) this file's flip-reproduction test keeps guarding the
//     numbers themselves.
//===----------------------------------------------------------------------===//

} // namespace leafcost
} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXLEAFCOSTTABLE_H
