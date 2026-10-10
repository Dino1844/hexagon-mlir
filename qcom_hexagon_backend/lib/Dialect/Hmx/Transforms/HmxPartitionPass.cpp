//===-- HmxPartitionPass.cpp - hmx.matmul to the tile loop ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The tile level of the HMX path: an `hmx.matmul` on crouton arrays becomes
// bias init -> (m, n) tiles -> K loop of mmas -> read-out.
//
// VTCM ownership (docs/hmx/hmx-system-design.md §10.12): the crouton arrays
// stay ordinary memory-space-1 buffers, and this pass deliberately does not
// carve a region out for them. The VTCM the kernel runs in is the one the
// runtime already owns, and the existing space-1 machinery
// (convert-to-hexagonmem -> the runtime's VTCM pool) is what places them there,
// together with every other VTCM buffer of the kernel. Acquiring a region of
// our own instead -- which is what this pass used to do -- takes megabytes away
// from that pool, the pool then fails its own "at least 1 MB available" check,
// and the DSP process dies.
//
// Staged tile loop: the activation bridge is a loop of HVX packs from a
// row-major DDR source, and on a single thread an HVX pack cannot overlap the
// (synchronous) HMX engine -- only the DMA engine can (measured: probe (B-A)/H
// = 0.998, exp/hmx/async_probe). So when the activation behind an `hmx.matmul`
// is that bridge loop, this pass stages one 32 x K activation tile at a time
// into a static ring of VTCM slots with `hmx.stage`, and re-hosts the pack to
// read the awaited slot (VTCM -> crouton) instead of DDR.
//
// The tile loop is not hand-pipelined. The pass emits the serial source form --
// one iteration issues tile m's transfer with `hmx.stage`, awaits it and
// computes it -- and at depth 2 hands that loop to the SCF software pipeliner
// (`scf::pipelineForLoop`) with a two-stage schedule: issue in stage 0, await
// and compute in stage 1. The pipeliner generates the prologue, the steady
// kernel and the peeled epilogue, and versions the cross-stage values -- the
// token and the slot select -- as its own iter_args, so the ring rotation is
// the pipeliner's value versioning rather than hand-written rotation. What
// makes the two static slots alternate is a parity select over the tile index,
// a stage-0 op the pipeliner re-evaluates with the shifted induction variable
// in the kernel's issue part. Depth 1 is the same source loop left unpipelined:
// issue and await stay adjacent, one slot suffices, and there is nothing to
// overlap. Either way the weight stays whole and resident, and the accumulator
// chain is untouched: one clear -> K mmas -> fused read per (m, n) tile, never
// two overlapping chains.
//
// Staging is profitable only when K is deep enough to hide the DMA engine's
// fixed per-transfer cost, so `auto` declines a shape with `Kt <
// kStageMinKTiles` (32) and keeps the plain tile loop. An explicit
// `pipeline-depth=1/2` bypasses the floor -- those are the A/B arms.
//
// The pack destination is one crouton-row scratch, not the whole activation
// array: the pack and its mmas are in the same iteration, so exactly one
// crouton row is live at a time and a single statically-allocated `memref<1 x
// Kt x 16 x 32 x 2 x f16, 1>` suffices. The pack's result *is* that buffer, so
// the mmas consume the packed croutons by address -- no extra copy between the
// two -- and there is no reason to keep a second in-flight row. The activation
// array itself is dead in this path and is retired whole (bridge loop, pack
// writers, deallocation and allocation), so only the row-major staging ring and
// the scratch occupy VTCM on top of the weight and the accumulator. A staged
// loop therefore never carries a full second copy of the activation.
//
// The unstaged serial path folds its whole-array bridge into the m-tile loop
// the same way when the bridge is the canonical pack shape
// (`tryFoldSerialActivationBridge`): one pack per m-tile from a per-tile view
// of the bridge's source into the same one-row scratch, the mmas reading the
// scratch, and the array retired with the bridge. No ring and no pipeliner --
// the serial source loop is the staged form's compute half with the DMA
// stage/await replaced by the source view. A serial shape the fold declines
// (an outer loop re-executes the matmul, the bridge is not canonical, the
// array has another reader) keeps the plain tile loop and the whole array,
// exactly as before the fold existed.
//
// The last tiles are the pipeliner's peeled epilogue, which only awaits and
// computes -- it stages nothing. That is what keeps every `hmx.stage` row in
// range without a guard or a token sentinel: the kernel stops one iteration
// short per pipeline stage, so no issue ever runs past tile Mt-1.
// `hmx.stage`/`hmx.await` therefore lower to one unconditional DMA runtime call
// each -- a sentinel is not an option because the runtime's real tokens start
// at 0.
//
// The ring is static -- `depth` slots and `depth` status words, allocated once
// outside the loop and released once after the epilogue -- so the loop body
// allocates nothing. When the pipeline declines for any reason, the plain tile
// loop is emitted and a remark names the reason.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDType.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxTarget.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

// The one VTCM byte ledger: this pass's budget reads go through it rather than
// through a private walk (see the header for why the private walks went).
#include "HmxVtcmLedger.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h" // mlir::isPure, for the read-out hoist
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>

#define DEBUG_TYPE "hmx-partition"

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_HMXPARTITION
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

/// `pipeline-depth` selecting the unstaged serial tile loop: no staging
/// rewrite, the activation bridge and its array kept as ordinary memory. It is
/// the A/B arm that reproduces the pre-pipeline codegen; the staged emitter
/// never sees this value.
///
// NOT-A-DECISION: an enum sentinel for the `pipeline-depth` knob, not a
// decision constant. It carries no profitability meaning, so demanding a
// traceability triple of it would be noise. It was previously listed as one;
// see `docs/codegen/constants-traceability-2026-09-30.md` C8.
constexpr int64_t kSerialPipelineDepth = 3;

// TRACEABILITY: kStageMinKTiles
//   mechanism: The staged path can pay for the ring at any K; the overlap only
//     wins when the transfer is large next to the DMA engine's fixed
//     per-transfer cost (issue + wait + scratch addressing). One staged tile is
//     32 * K * 2 bytes with K = 32 * Kt, i.e. 2 KiB * Kt, so the transfer grows
//     linearly in Kt while the fixed cost does not.
//   measurement: warm A/B in ONE build, `auto` staging vs the
//     `pipeline-depth=3` serial arm; drivers under exp/hmx. S2 (Kt=64):
//     297 -> 153 us (1.94x). S1 (Kt=2): 125.5 -> 126.5 us (flat). S3: 60.5 ->
//     67 us (-10%). Only `auto` is gated; 1/2 are the A/B arms, 3 never reaches
//     the staged emitter.
//   shape set: S1/S2/S3, defined in exp/hmx/op_bench/bench_ops.py. **These are
//     the project's own anchor shapes, not a real workload set.** They are not
//     traced to any llama.cpp per-op bench entry.
//   workload representativeness: **NOT ESTABLISHED.** bench_ops.py calls S1
//     "the S1 anchor shape", which is circular. Their weights are 0-3% of VTCM,
//     so they do not exercise the residency pressure the gate is supposed to
//     trade against.
//   known-bad: the mechanism itself is refuted in-repo. exp/hmx/leaf_bw_probe/
//     RESULTS.md says the right predicate is "is the source L2-cold" (activation
//     bytes vs L2 capacity), not a Kt >= 32 approximation. **Re-tuning the
//     number on the same mechanism would be wasted work**; the fix is to replace
//     the scalar with that capability query.
//   S3 identity: a 3-file vote, NOT resolved. Three files define S3 as
//     128x128x128 (= Kt=4) -- exp/hmx/throughput_probe/host_residual_structure.py:42,
//     exp/hmx/deep_k_shapes/probe_deep_k.py:38, and
//     docs/hmx/gap-vs-llama-plain-2026-09-29.md:11. The lone outlier is
//     exp/hmx/op_bench/bench_ops.py:291 (1024x64x256 = Kt=8), which by that
//     count is mislabelled.
//     **But no log recorded which shape the -10% was actually taken on.** A
//     3-1 file vote is evidence, not proof, and it was deciding what a
//     calibration point MEANS -- so the earlier phrasing "RESOLVED" (and the
//     "8x below the floor, not 4x" inference that came with it) was withdrawn.
//
//     SETTLED 2026-10-02, same-build on device, N=1000, 3 reps, fingerprints
//     identical before and after (ce26015e8efb75cc047515000c8ad70f). Arm
//     enableHmxPipelineDepth=3 is the plain tile loop, =2 is staged with
//     overlap, config=base; log at
//     logs/phase0-1-anchor-2026-10-02/depth_ab_N1000.log, writeup at
//     docs/results/kstage-floor-pinned-2026-10-02.md:
//
//       shape              Kt   vs floor 32   depth3      depth2    depth2 vs depth3
//       S1 1024x512x64      2      0.06x        55 us       64 us      +16.4%  (loses)
//       S3 128^3            4      0.12x        10 us       16 us      +60.0%  (loses)
//       S2 256x64x2048     64      2.00x        87 us       41 us      -52.9%  (wins)
//
//     So the negative point IS 8x below the floor -- the withdrawn inference
//     was right -- and `auto` picks the faster arm on all three.
//
//     The old number was wrong because of the iteration count, not the noise.
//     That run used N=30, where once_share is 79% for S3 (ROADMAP1001.md 5.1.2):
//     it mostly measured the per-invocation bring-up, which staging does not
//     touch, so it diluted the mechanism to +10.8% where the mechanism is
//     +60.0%. Same dilution on the winning side: S2 read 1.94x then, 2.12x now.
//     A verdict drawn at N=30 is a verdict about the wrong quantity.
//
//     Still unpinned: the band just above the floor. Nothing is measured
//     between Kt 5 and 63, so 32 is a conservative pick inside (4, 64), not a
//     searched optimum. depth=1 (staged serial) is also still unmeasured.
// Flip point, re-derived and confirmed by host codegen: Kt is read off the
// crouton array (actType.getDimSize(1)) and K == Kt * layout::kTileEdge is
// required just above, so the floor of 32 flips at **K = 1024 exactly**.
// Observed: 256x256x512 -> reason "shallow-k"; 256x256x1024 -> staged, depth 2.
//
// Real-workload proximity -- RETRACTED 2026-10-02, see below. The constant
// stays at 32; what was wrong is the evidence that was offered for it.
//
// The earlier note here claimed a 2026-09-30 survey over
// logs/real-shapes-2026-09-29/ found "one real shape sitting exactly ON the
// boundary (FA PV, 1024x128x1024, Kt=32)" plus one at Kt=8, and that "the same
// attention op straddles the gate -- QK^T's K is head_dim (small, refused)
// while PV's K is seq (large, allowed)". That survey does not support the
// claim. Its attn_pv_s1024/s4096 entries were produced by
// exp/hmx/tmp-archive-2026-10-02/sweep.sh:14, which passes MATMUL_M/N/K to
// `dump_codegen.py matmul` -- a synthetic probe whose block_shape is the whole
// matrix. So those entries are single-tile matmuls of shape 1024x128x1024 and
// 256x128x4096, wearing an "attn_pv" name. No attention kernel was compiled.
//
// Compiling this repo's actual FA kernel
// (test/python/triton/test_flash_attention.py) gives, for both of its matmuls:
//
//     m=1024  n=64  k=64  Kt=2  selected=serial  reason=shallow-k
//
// The n and k do not match the survey's 128 / 1024, so the Kt=32 row is not
// this workload. `test_flash_attention.py:105` is `tl.dot(p, v, acc)` with
// p=(BLOCK_M, BLOCK_N) and v=(BLOCK_N, BLOCK_DMODEL), so the op's K is
// BLOCK_N; the kv loop is outside the dot, which makes seq a loop trip count
// rather than an op dimension. Same result at D_HEAD/BLOCK_N of 64/64,
// 128/64, 128/128 and 256/128: all shallow-k, budget_depth 0. A real op-K of
// 1024 would have staged.
//
// The gate also follows the K-loop step, not the matrix K. Holding logical
// K=14336 and varying only BK: BK 32..512 are all shallow-k, 1024 and 2048
// are staged. That is structural -- the ring prefetches the next K-tile of the
// same output tile, so it can only stage across K-tiles inside one op, and
// `for k0 in range(0, K, BK)` lowers outside it. Reading the logical K instead
// is not a predicate fix either: matmul.logical.k.value in the manifest is
// per-op too (64 or 1024 for logical 14336), so no larger K exists in this IR.
//
// What this leaves: kStageMinKTiles=32 is still a defensible conservative
// pick inside (4, 64) with no measurement between Kt 5 and 63, and the flip at
// K=1024 above is confirmed. What it does NOT have is a real workload sitting
// anywhere near it -- every measured operator lands at Kt <= 8. Re-derive the
// proximity evidence from compiled real kernels before citing it.
// See docs/results/t-hmx-staging-gate-dead-2026-10-02.md and
// docs/results/b3-voided-baselines-2026-10-02.md.
constexpr int64_t kStageMinKTiles = 32;

struct PipelineDecision {
  int64_t requestedDepth = 0;
  bool staged = false;
  int64_t depth = 0;
  PipelineReason reason = PipelineReason::None;
  int64_t neededBytes = 0;
  /// Bytes of VTCM the budget had room for. Only meaningful for
  /// PipelineReason::VtcmBudget; zero for every other reason.
  int64_t freeBytes = 0;
  /// K tiles, the staging unit the floor is expressed in. Only meaningful for
  /// PipelineReason::ShallowK; zero for every other reason.
  ///
  /// 2026-10-02: this used to be carried in `freeBytes`, because
  /// declineStageLoop took a single positional `detail`. That made one field
  /// mean bytes in one arm and a tile count in another, and the ShallowK remark
  /// duly printed "Kt <bytes>". The printed number was still right -- nothing
  /// read the field except that one remark -- but any future consumer that read
  /// freeBytes without switching on `reason` first would have been wrong in a
  /// way that compiles cleanly. recordPipelineDecision now rejects the
  /// mismatched combinations so the union cannot be reintroduced.
  int64_t kTiles = 0;
  int64_t budgetDepth = 0;
};

bool hasPipelineRemark(PipelineReason reason) {
  return reason != PipelineReason::None &&
         reason != PipelineReason::SerialRequested &&
         reason != PipelineReason::TileCount;
}

InFlightDiagnostic &renderPipelineRemark(InFlightDiagnostic &diag,
                                         const PipelineDecision &decision) {
  switch (decision.reason) {
  case PipelineReason::NoRowMajorBridge:
    return diag
           << "HMX pipeline not applied: the activation is not a "
              "row-major staging bridge (nothing to re-host onto a VTCM "
              "slot), so the tile loop reads the crouton array where it is";
  case PipelineReason::ExtraActivationReader:
    return diag << "HMX pipeline not applied: the activation array has a "
                   "reader the staging rewrite does not own";
  case PipelineReason::InvalidStagingGeometry:
    return diag << "HMX pipeline not applied: the activation staging geometry "
                   "does not describe rank-5 f16 crouton arrays with a static "
                   "row-major f16/f32 source";
  case PipelineReason::EmptyStagingGrid:
    return diag << "HMX pipeline not applied: the activation staging geometry "
                   "has an empty crouton grid";
  case PipelineReason::StagingGridMismatch:
    return diag << "HMX pipeline not applied: the source, activation and "
                   "weight grids disagree on the staging tile shape";
  case PipelineReason::ShallowK:
    return diag << "HMX pipeline not applied: Kt " << decision.kTiles
                << " is below the staging floor of " << kStageMinKTiles
                << " -- one transfer is too small to hide the DMA engine's "
                   "fixed cost behind the tile's compute";
  case PipelineReason::VtcmBudget:
    if (!decision.staged)
      return diag
             << "HMX pipeline not applied: activation staging needs "
             << decision.neededBytes
             << " bytes of VTCM (one crouton scratch plus the serial ring), "
                "only "
             << decision.freeBytes << " are free";
    if (decision.requestedDepth <= 0)
      return diag << "HMX pipeline not applied at depth 2: double-buffered "
                     "activation staging needs "
                  << decision.neededBytes << " bytes of VTCM, only "
                  << decision.freeBytes << " are free; using the serial ring";
    return diag << "HMX pipeline depth " << decision.requestedDepth
                << " requested, but only " << decision.freeBytes
                << " bytes of VTCM are free after the crouton scratch: a depth-"
                << decision.budgetDepth << " ring needs "
                << decision.neededBytes << " bytes; using depth "
                << decision.budgetDepth;
  case PipelineReason::PipelinerFailed:
    return diag
           << "HMX pipeline not applied at depth 2: the SCF loop "
              "pipeliner declined the schedule; the staged loop runs serially";
  case PipelineReason::TailPeeledEdge:
    return diag << "HMX tail plan lowered as full + peeled serial regions; "
                   "activation staging is not applied to tail plans";
  case PipelineReason::None:
  case PipelineReason::SerialRequested:
  case PipelineReason::TileCount:
    llvm_unreachable("silent pipeline decision rendered as a diagnostic");
  }
  llvm_unreachable("unknown HMX pipeline diagnostic");
}

LogicalResult readDecisionId(Operation *op, std::optional<int64_t> &id) {
  id.reset();
  Attribute raw = op->getAttr(kHmxDecisionIdAttr);
  if (!raw)
    return success();
  auto integer = dyn_cast<IntegerAttr>(raw);
  if (!integer || !integer.getType().isSignlessInteger(64)) {
    op->emitError("hmx.matmul has an invalid hmx.decision_id");
    return failure();
  }
  id = integer.getInt();
  return success();
}

LogicalResult recordPipelineDecision(MatmulOp op,
                                     const PipelineDecision &decision) {
  // 2026-10-02: kTiles is meaningful for exactly one reason. freeBytes is not
  // constrained, because the staged path sets it to the room it measured for
  // every decision it records, not only for the VtcmBudget decline -- it is
  // "bytes considered", not "bytes that caused a downgrade". kTiles used to
  // ride in that same field, which is how the ShallowK remark came to print a
  // tile count labelled as bytes. The check below is what stops the union from
  // being reintroduced.
  if (decision.kTiles != 0 && decision.reason != PipelineReason::ShallowK) {
    op.emitError("hmx pipeline decision carries a K-tile count for the wrong "
                 "reason");
    return failure();
  }
  std::optional<int64_t> id;
  if (failed(readDecisionId(op.getOperation(), id)))
    return failure();
  if (id) {
    auto func = op->getParentOfType<func::FuncOp>();
    auto module = op->getParentOfType<ModuleOp>();
    if (!func || !module) {
      op.emitError("hmx.matmul is not inside a function and module");
      return failure();
    }
    if (!module->hasAttr(kHmxManifestAttr)) {
      op.emitError("hmx.matmul has a decision id but no module manifest");
      return failure();
    }
    if (failed(setHmxManifestPipelineDecision(
            module, func.getName(), *id, decision.requestedDepth,
            decision.staged ? kHmxPipelineStaged : kHmxPipelineSerial,
            decision.depth, pipelineReasonCode(decision.reason),
            decision.budgetDepth)))
      return failure();
  }
  if (hasPipelineRemark(decision.reason)) {
    InFlightDiagnostic diag = op.emitRemark();
    renderPipelineRemark(diag, decision);
  }
  return success();
}

LogicalResult declineStageLoop(MatmulOp op, int64_t requestedDepth,
                               PipelineReason reason, int64_t freeBytes = 0,
                               int64_t neededBytes = 0, int64_t budgetDepth = 0,
                               int64_t kTiles = 0) {
  PipelineDecision decision;
  decision.requestedDepth = requestedDepth;
  decision.staged = false;
  decision.depth = 0;
  decision.reason = reason;
  decision.freeBytes = freeBytes;
  decision.kTiles = kTiles;
  decision.neededBytes = neededBytes;
  decision.budgetDepth = budgetDepth;
  return recordPipelineDecision(op, decision);
}

/// The tile sizes of a matmul, read off the crouton arrays' leading dimensions.
struct TileShape {
  int64_t m, n, k;
};

std::optional<TileShape> getTileShape(MatmulOp op) {
  auto lhs = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhs = dyn_cast<MemRefType>(op.getRhs().getType());
  auto out = dyn_cast<MemRefType>(op.getOuts().getType());
  if (!lhs || !rhs || !out)
    return std::nullopt;
  if (lhs.getRank() != 5 || rhs.getRank() != 5 || out.getRank() != 5)
    return std::nullopt;
  return TileShape{lhs.getDimSize(0), weightNTiles(rhs), lhs.getDimSize(1)};
}

struct TailGrid {
  int64_t mTiles = 0, nTiles = 0, kTiles = 0;
  int64_t mFullTiles = 0, nFullTiles = 0, kFullTiles = 0;
  int64_t mTail = 0, nTail = 0, kTail = 0;
  int64_t mLogical = 0, nLogical = 0, kLogical = 0;
};

/// Validate the complete static tail contract at the point where the physical
/// grid is about to be walked. The op verifier already checks the rank-5 grid;
/// this second check is intentionally about the plan arithmetic and keeps a
/// hand-authored or stale attribute from turning into a different region walk.
LogicalResult readTailGrid(MatmulOp op, const TileShape &shape,
                           TailPlanAttr plan, TailGrid &grid) {
  ArrayRef<int64_t> logical = plan.getLogical();
  ArrayRef<int64_t> padded = plan.getPadded();
  ArrayRef<int64_t> full = plan.getFull();
  ArrayRef<int64_t> tail = plan.getTail();
  if (logical.size() != 3 || padded.size() != 3 || full.size() != 3 ||
      tail.size() != 3)
    return op.emitError("tail_plan logical/padded/full/tail must each have 3 "
                        "dimensions");
  if (plan.getKPolicy() != "zero-pad-both-operands" ||
      plan.getMnPolicy() != "padded-edge-tile-bounded-store")
    return op.emitError("tail_plan has an unsupported padding or edge policy");

  auto checkSplit = [&](int64_t value, int64_t expectedFull,
                        int64_t expectedPadded, int64_t expectedTail,
                        StringRef name) -> LogicalResult {
    int64_t fullValue = 0, paddedValue = 0, tailValue = 0;
    if (!HmxTarget::splitExtent(value, fullValue, paddedValue, tailValue))
      return op.emitError() << "tail_plan " << name
                            << " is not a representable positive extent";
    if (paddedValue != expectedPadded || fullValue != expectedFull ||
        tailValue != expectedTail)
      return op.emitError() << "tail_plan " << name << " [full, padded, tail] "
                            << "does not match logical extent " << value;
    return success();
  };

  if (failed(checkSplit(logical[0], full[0], padded[0], tail[0], "M")) ||
      failed(checkSplit(logical[1], full[1], padded[1], tail[1], "N")) ||
      failed(checkSplit(logical[2], full[2], padded[2], tail[2], "K")))
    return failure();

  if (tail[0] == 0 && tail[1] == 0 && tail[2] == 0)
    return op.emitError("tail_plan has no peeled edge");

  if (padded[0] / layout::kTileEdge != shape.m ||
      padded[1] / layout::kTileEdge != shape.n ||
      padded[2] / layout::kTileEdge != shape.k)
    return op.emitError("tail_plan padded shape does not match the matmul "
                        "crouton grid");

  grid = TailGrid{padded[0] / layout::kTileEdge,
                  padded[1] / layout::kTileEdge,
                  padded[2] / layout::kTileEdge,
                  full[0] / layout::kTileEdge,
                  full[1] / layout::kTileEdge,
                  full[2] / layout::kTileEdge,
                  tail[0],
                  tail[1],
                  tail[2],
                  logical[0],
                  logical[1],
                  logical[2]};
  return success();
}

/// The activation bridge behind an `hmx.matmul`: the pack(s) that fill the
/// activation crouton array from one row-major source. That source is the DDR
/// side the tile loop stages; without it (a chained read-out -- already a
/// crouton in VTCM -- or a plain allocation) there is nothing to stage and the
/// plain tile loop is the right form.
///
/// Two shapes reach here. After bufferization the bridge carries the array as a
/// loop iter_arg (the `matmul-to-hmx` tensor form); canonicalization folds that
/// DPS write into a straight in-place pack loop on the allocation. Both are the
/// same bridge and both are staged.
///
/// This is a finder, not a validator: the bridge is emitted by `matmul-to-hmx`,
/// so the one structural fact the tile loop needs is that every producer of the
/// array shares one source and one enclosing loop. The old body-shape whitelist
/// is gone with the hand-written pipeline -- with one exception, `loop` below:
/// claiming a loop to ERASE is not a decline but an ownership claim, and the
/// only loop this finder may claim is the bridge's own (see
/// `isBridgePackLoop`).
struct ActivationBridge {
  Value src;                    // row-major source (the DDR side to stage)
  Value buffer;                 // the crouton buffer the array is built on
  scf::ForOp loop;              // enclosing pack loop; null when unlooped
  SmallVector<PackActOp> packs; // every pack that writes the array
};

/// Every `hmx.pack_act` that writes `array` as its destination.
static SmallVector<PackActOp> packWriters(Value array) {
  SmallVector<PackActOp> packs;
  for (OpOperand &u : array.getUses())
    if (auto p = dyn_cast<PackActOp>(u.getOwner()))
      if (p.getDst() == array)
        packs.push_back(p);
  return packs;
}

/// Whether `loop` IS the bridge's own pack loop -- whether it exists only to
/// pack `act`, so retiring the bridge may retire the loop along with it.
///
/// This is an identification test, NOT a validator: a loop that fails it is not
/// declined, it is simply not claimed, and its packs are retired one by one
/// instead (the `!bridge->loop` arm of `emitStageLoop`'s cleanup). No shape
/// stops staging because of this.
///
/// What it rules out is claiming a loop the bridge merely SITS inside.
/// `matmul-to-hmx` wraps its pack in a loop over the bridge's outer tiles
/// (`packCroutonsWithLeaves`), but canonicalization folds the single-tile case
/// (`Mt == 1`) into a bare `hmx.pack_act`, and what then encloses the pack is
/// the kernel's own loop -- the m/n/k loop the matmul runs in. The staged
/// emitter builds its tile loop *inside* wherever the matmul sits and afterwards
/// erases `bridge->loop` to retire the bridge; claiming the kernel's loop erases
/// the kernel, the freshly built tile loop and the matmul's result write with
/// it, and leaves behind a manifest that records a staged plan over an empty
/// span (no pack, no mma, no output store). That is the silent-decay bug this
/// test guards: a matmul under an N loop alone, `Mt == 1`.
///
/// The rule is `findPackBridge`'s, for the same reason on the serial arm: the
/// body holds the bridge's packs and the index arithmetic feeding them, and
/// anything else -- the kernel's fills and copies, a second bridge, the matmul
/// itself -- makes it somebody else's loop, which is not this finder's to
/// erase. A pack of another array counts as somebody else's.
static bool isBridgePackLoop(scf::ForOp loop, Value act) {
  for (Operation &inner : loop.getBody()->without_terminator()) {
    if (auto pack = dyn_cast<PackActOp>(&inner)) {
      if (pack.getDst() != act)
        return false;
      continue;
    }
    if (inner.getName().getDialectNamespace() == "arith")
      continue;
    return false;
  }
  return true;
}

std::optional<ActivationBridge> findActivationBridge(Value act) {
  SmallVector<PackActOp> packs;
  Value buffer = act;
  scf::ForOp loop;
  if (act.getDefiningOp<memref::AllocOp>()) {
    // Canonicalized form: the packs write the allocation in place.
    packs = packWriters(act);
    if (!packs.empty()) {
      // The packs' nearest enclosing loop is the bridge's own loop only when
      // the bridge still has one; a pack canonicalization left bare sits
      // inside the kernel's loop, which is not ours to erase (see
      // `isBridgePackLoop`). Unclaimed = unlooped bridge: same retirement
      // path as a pack at the top level, packs erased individually.
      scf::ForOp enclosing = packs.front()->getParentOfType<scf::ForOp>();
      if (enclosing && isBridgePackLoop(enclosing, act))
        loop = enclosing;
    }
  } else if (auto carried = act.getDefiningOp<scf::ForOp>()) {
    // Carried form: the array is the pack loop's iter_arg, so the buffer is the
    // iter_arg's initial value (the allocation) and the producers write the
    // region iter_arg.
    loop = carried;
    if (loop.getNumResults() != 1 || loop.getResult(0) != act ||
        loop.getInitArgs().empty())
      return std::nullopt;
    buffer = loop.getInitArgs()[0];
    for (Operation &inner : loop.getBody()->without_terminator())
      if (auto p = dyn_cast<PackActOp>(inner))
        if (p.getDst() == loop.getRegionIterArg(0))
          packs.push_back(p);
  } else {
    return std::nullopt;
  }
  if (packs.empty())
    return std::nullopt;

  // The bridge is one source packed by every writer; an unrolled body just has
  // more writers of the same source. Writers of different sources into one
  // array are not the bridge shape, and writers sitting in different loops are
  // not one bridge this finder can retire as a unit -- the nearest enclosing
  // loop of the first writer is the candidate every writer must share.
  Value src = packs.front().getSrc();
  scf::ForOp enclosing = packs.front()->getParentOfType<scf::ForOp>();
  for (PackActOp p : packs)
    if (p.getSrc() != src || p->getParentOfType<scf::ForOp>() != enclosing)
      return std::nullopt;
  return ActivationBridge{src, buffer, loop, packs};
}

/// One `hmx.pack_act`: the 32x32 block at (`row`, `col`) of `src` into crouton
/// (`row`, `col`) of `dst`. The memref form returns its destination (interface
/// plan §9.3), so the written buffer is an SSA value; the tile loop does not
/// need to read it back (the pack and its mma are in the same iteration), but
/// the result is what makes the dependency visible to an external scheduler.
static Value emitPackAct(IRRewriter &rewriter, Location loc, Value dst,
                         Value src, Value row, Value col,
                         std::optional<int64_t> decisionId,
                         IntegerAttr count = {}) {
  auto pack =
      PackActOp::create(rewriter, loc, TypeRange{dst.getType()}, dst, src, row,
                        col, count, IntegerAttr(), IntegerAttr());
  // Standalone hand-written hmx.matmul has no manifest decision to carry;
  // production/full-pipeline IR always supplies the explicit id.
  if (decisionId)
    pack->setAttr(kHmxDecisionIdAttr, rewriter.getI64IntegerAttr(*decisionId));
  return pack->getResult(0);
}

/// One `hmx.stage`: start the DMA of activation tile `row` from the row-major
/// DDR `src` into VTCM slot `dst`, recording completion in `status`. Returns
/// the token that `emitAwait` waits on -- an i32 register, the ring's only
/// handle. The op issues its transfer unconditionally; keeping every `row` in
/// range is the tile loop's job, done by construction (see the pass header).
static Value emitStage(IRRewriter &rewriter, Location loc, Value src, Value row,
                       Value dst, Value status) {
  return StageOp::create(rewriter, loc, TypeRange{rewriter.getI32Type()},
                         ValueRange{src, row, dst, status})
      ->getResult(0);
}

/// Wait for `token`'s transfer and hand back the now-ready slot. The returned
/// memref is `dst` itself (the op returns what it waited on), which is what the
/// pack consumes: the wait -> pack edge is an SSA value edge, not an effect.
static Value emitAwait(IRRewriter &rewriter, Location loc, Value token,
                       Value dst) {
  return AwaitOp::create(rewriter, loc, TypeRange{dst.getType()},
                         ValueRange{token, dst})
      ->getResult(0);
}

// The engine's activation repeat count `Rt[dC]` is five bits, so ONE `hmx.mma`
// covers at most 32 K croutons = 1024 input channels (V81 PRM 4.2.1: "up to 32
// croutons (1024 input channels)", hmx_prm_v81.txt:1593-1594). That is the
// verifier's `n_croutons <= 32` (HmxOps.cpp:410-411) -- a hardware bound, not a
// tunable -- and it is the same bound llama.cpp asserts before its dot chunk
// (`__builtin_assume(n_dot_tiles <= 32)`, hmx-mm-kernels-tiled.h:609).
//
// NOT-A-DECISION. This is the widest value the field can legally carry, read
// off the hardware encoding rather than chosen: the runtime already computed it
// (`HMXAPI.c:51 act_rt = 2047 | ((n_croutons-1) << 11)`, and `HMXAPI.c:53`
// `wt_rt = ((kBlockPairs * n_croutons - 1) << 7) | 0x7f`), so a smaller value
// here would leave the widest encoding unused, not encode a different choice.
// The encoder has no mechanism for "batch less than the hardware maximum", so
// there is nothing here to calibrate: lowering it cannot select a different
// code path, only a slower one. See docs/hmx/ncroutons-k-fusion-2026-10-01.md.
//
// `croutons-per-mma` is an Option, and the reason above is why that does not
// make 32 a decision: the option's DEFAULT is this same hardware maximum, so the
// emitted code is unchanged. The option exists to make "batch less than the
// maximum" *expressible*, which is what lets the batching be A/B'd inside one
// build instead of by editing a constant and rebuilding. The domain is
// {0} u [1, 32] (0 = this maximum), and out-of-domain values are an error, not
// a clamp.
static constexpr int64_t kMaxCroutonsPerMma = 32;

/// Resolve the `croutons-per-mma` option to the batch size the emitter uses.
/// 0 means the hardware maximum, exactly as `vtcm-budget`'s 0 means the device
/// default -- so a caller that does not pass the option, one that passes 0, and
/// one that passes 32 all ask for the same thing. The caller is responsible for
/// having rejected the out-of-domain values (see `runOnOperation`); this only
/// maps 0.
///
/// The parameter is named `requested` because the resolved value is not what was
/// asked for whenever 0 is passed, and the distinction is the whole reason the
/// two spellings are not merged.
static int64_t resolveCroutonsPerMma(int64_t requested) {
  return requested == 0 ? kMaxCroutonsPerMma : requested;
}

/// Emit the K traversal of one output tile as `hmx.mma`s, batching K croutons
/// through the engine's repeat count instead of issuing one instruction per
/// crouton.
///
/// `act`/`wt` are the whole crouton arrays and `m`/`n` the tile indices. `kt` is
/// K in croutons -- the activation array's second grid dim -- and it is a
/// compile-time constant (the last paragraph says why the integer may be
/// passed). `batch` is the resolved `croutons-per-mma` value: K croutons per
/// `hmx.mma`, in [1, 32]. The prose below is written for the default
/// `batch = 32`; the same three cases hold for any `batch`, with 32 read as
/// "the batch".
///
/// The batching is sound because the op's contract is "starting at position
/// (m, k) ... and walking forward from there" (HmxOps.td:474-476) and the count
/// travels ONLY in `n_croutons`: the address `croutonAddr` builds is a function
/// of `k` alone, `base + (row*rowStride + col) * 2048`, and
/// `test/Conversion/HmxToLLVM/mma-deep-croutons.mlir` pins that the whole
/// address chain is identical for `n_croutons = 1` and `n_croutons = 32` at the
/// same Kt, with the count reaching the leaf as the third argument and nothing
/// else. So a run of `c` croutons is one mma at `k` with `n_croutons = c`, and K
/// is walked as
///
///     floor(Kt / 32) batches of 32, then -- only when Kt % 32 != 0 -- one tail
///     batch of Kt % 32 at k = floor(Kt / 32) * 32.
///
/// That is llama.cpp's shape (`n_loops = n_dot_tiles / 32`, then
/// `rem = n_dot_tiles % 32` handled separately, hmx-mm-kernels-tiled.h:657-668)
/// and the tail is NOT optional: at `Kt % 32 == 0` there is no tail at all, and
/// emitting one anyway would be a zero-count mma the verifier rejects. The Kt
/// values that separate those two cases -- 31, 32, 33, 64, 65 -- are exactly the
/// ones `mma-deep-croutons.mlir` covers on the lowering side and
/// `hmx-partition-deep-crontons.mlir` covers here.
///
/// Every caller's `kt` was already this value, materialised as an
/// `arith.constant` for the loop bound: `shape.k = lhs.getDimSize(1)` for the
/// serial arm, `grid.kTiles` for a peeled region (equal to the same dim, which
/// `readTailGrid` rejects a disagreement with), and `Kt` for the staged scratch.
/// Passing the integer instead of the `Value` therefore changes the loop, not
/// its extent.
static void emitMmaKLoop(IRRewriter &rewriter, Location loc, Value act,
                         Value wt, Value m, Value n, int64_t kt, int64_t batch) {
  // K = 0 emitted nothing before this change (the loop was zero-trip), so keep
  // that: emitting a zero-count mma instead would turn a degenerate shape into a
  // verifier error, which is a new failure mode rather than a fix.
  if (kt < 1)
    return;
  // The whole K in one instruction: no loop and no tail. This is the shallow-K
  // case (the S1/S3 anchors have Kt = 2..8), where the one-per-cronton form
  // spent a `Kt`-iteration software loop and `Kt` issue slots on a single
  // engine packet's worth of work. A K no deeper than one batch needs no
  // batching either, and there the count is the shape, not the batch.
  if (kt <= batch) {
    auto kOnly = arith::ConstantIndexOp::create(rewriter, loc, 0);
    MmaOp::create(rewriter, loc, act, wt, m, n, kOnly,
                  rewriter.getI32IntegerAttr(kt));
    return;
  }

  int64_t batches = kt / batch;
  int64_t rem = kt % batch;
  auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto batchStep = arith::ConstantIndexOp::create(rewriter, loc, batch);
  auto batchEnd =
      arith::ConstantIndexOp::create(rewriter, loc, batches * batch);
  // The induction variable IS the crouton index: stepping it by the batch size
  // walks the batches without a multiply.
  auto kLoop =
      scf::ForOp::create(rewriter, loc, zero, batchEnd, batchStep, ValueRange{});
  rewriter.setInsertionPointToStart(kLoop.getBody());
  MmaOp::create(rewriter, loc, act, wt, m, n, kLoop.getInductionVar(),
                rewriter.getI32IntegerAttr(batch));
  rewriter.setInsertionPointAfter(kLoop);

  if (rem == 0)
    return;
  auto kTail =
      arith::ConstantIndexOp::create(rewriter, loc, batches * batch);
  MmaOp::create(rewriter, loc, act, wt, m, n, kTail,
                rewriter.getI32IntegerAttr(rem));
}

/// The serial tile loop: bias clear, the (m, n) tiles, one K loop of mmas and a
/// fused read-out per tile. This is the form whenever the pipeline does not
/// apply, including the `pipeline-depth=3` arm -- and, when the activation
/// behind the matmul is the canonical whole-array pack bridge and the fold
/// applies (`tryFoldSerialActivationBridge`), it is the inner half of the
/// folded form instead: the pack has already moved into the m-tile loop and
/// this emitter is not reached.
///
/// It touches only the matmul's own operands: `act`, `wt` and the output are
/// consumed where they are, so a bridge that fills `act` (and the array itself)
/// survives untouched. That is what makes the declined serial arm work for a
/// bridge shape -- the pack loop stays and the mma reads the whole array, with
/// nothing staged. There is no structural assumption here beyond the tile
/// shapes the caller already resolved.
static void emitSerialTileLoop(IRRewriter &rewriter, Location opLoc, Value bias,
                               MatmulOp op, TileShape shape, int64_t batch) {
  Value act = op.getLhs();
  Value wt = op.getRhs();
  Value ar = op.getOuts();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto zero = arith::ConstantIndexOp::create(rewriter, opLoc, 0);
  auto step = arith::ConstantIndexOp::create(rewriter, opLoc, 1);
  auto mM = arith::ConstantIndexOp::create(rewriter, opLoc, shape.m);
  auto nN = arith::ConstantIndexOp::create(rewriter, opLoc, shape.n);

  auto mLoop =
      scf::ForOp::create(rewriter, opLoc, zero, mM, step, ValueRange{});
  rewriter.setInsertionPointToStart(mLoop.getBody());
  Value m = mLoop.getInductionVar();

  auto nLoop =
      scf::ForOp::create(rewriter, opLoc, zero, nN, step, ValueRange{});
  rewriter.setInsertionPointToStart(nLoop.getBody());
  AccClearOp::create(rewriter, opLoc);
  emitMmaKLoop(rewriter, opLoc, act, wt, m, nLoop.getInductionVar(), shape.k,
               batch);
  AccReadOp::create(rewriter, opLoc, bias, ar, m, nLoop.getInductionVar(),
                    rewriter.getI32IntegerAttr(0));
}

struct PackBridge {
  Value source;
  Value buffer;
  scf::ForOp loop;
  SmallVector<Operation *> ops;
};

struct UnpackBridge {
  Value destination;
  Value residual;
  bool fused = false;
  scf::ForOp loop;
  Value result;
  SmallVector<Operation *> ops;
};

/// The diagnostic tail slice owns direct bridges and the canonical single-loop
/// DPS forms emitted by the current pack/unpack bridges. A loop with any
/// other body operation, mixed destinations, or multiple loops is rejected
/// rather than silently erased.
static std::optional<PackBridge> findPackBridge(Value array, bool isWeight) {
  PackBridge bridge;
  bridge.buffer = array;
  bool carriedArray = array.getDefiningOp<scf::ForOp>() != nullptr;

  auto addPack = [&](Operation *owner) -> bool {
    if (isWeight ? !isa<PackWeightOp>(owner) : !isa<PackActOp>(owner))
      return false;
    if (owner->getParentOfType<scf::ForOp>() != bridge.loop)
      return false;
    Value source = isWeight ? cast<PackWeightOp>(owner).getSrc()
                            : cast<PackActOp>(owner).getSrc();
    Value destination = isWeight ? cast<PackWeightOp>(owner).getDst()
                                 : cast<PackActOp>(owner).getDst();
    Value expectedDestination =
        bridge.loop && carriedArray ? bridge.loop.getRegionIterArg(0) : array;
    if (destination != expectedDestination)
      return false;
    if (!bridge.source)
      bridge.source = source;
    if (bridge.source != source)
      return false;
    bridge.ops.push_back(owner);
    return true;
  };

  if (auto loop = array.getDefiningOp<scf::ForOp>()) {
    if (loop.getNumResults() != 1 || loop.getResult(0) != array ||
        loop.getNumRegionIterArgs() != 1 || loop.getInitArgs().empty())
      return std::nullopt;
    bridge.loop = loop;
    bridge.buffer = loop.getInitArgs()[0];
    for (Operation &inner : loop.getBody()->without_terminator())
      if (isa<PackActOp, PackWeightOp>(&inner))
        if (!addPack(&inner))
          return std::nullopt;
  } else {
    for (OpOperand &use : array.getUses()) {
      Operation *owner = use.getOwner();
      if (isWeight ? !isa<PackWeightOp>(owner) : !isa<PackActOp>(owner))
        continue;
      if (owner->getParentOfType<scf::ForOp>()) {
        if (bridge.loop && bridge.loop != owner->getParentOfType<scf::ForOp>())
          return std::nullopt;
        bridge.loop = owner->getParentOfType<scf::ForOp>();
      }
      if (!addPack(owner))
        return std::nullopt;
    }
  }

  if (!bridge.source || bridge.ops.empty())
    return std::nullopt;
  for (Operation *op : bridge.ops)
    if (op->getParentOfType<scf::ForOp>() != bridge.loop)
      return std::nullopt;
  if (bridge.loop) {
    for (Operation &inner : bridge.loop.getBody()->without_terminator())
      if (!isa<PackActOp, PackWeightOp>(&inner) &&
          inner.getName().getDialectNamespace() != "arith")
        return std::nullopt;
  }
  return bridge;
}

static std::optional<int64_t> constantIndexValue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  if (!constant)
    return std::nullopt;
  auto integer = constant->getAttrOfType<IntegerAttr>("value");
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}

static LogicalResult verifyPackCoverage(Operation *anchor,
                                        ArrayRef<Operation *> ops,
                                        bool isWeight, int64_t outerTiles,
                                        int64_t kTiles) {
  if (outerTiles < 0 || kTiles < 0 ||
      (kTiles != 0 &&
       outerTiles > std::numeric_limits<int64_t>::max() / kTiles))
    return anchor->emitError()
           << "diagnostic tail bridge grid is not representable";
  int64_t expected = outerTiles * kTiles;
  if (static_cast<int64_t>(ops.size()) != expected)
    return anchor->emitError()
           << "diagnostic tail bridge must contain one pack per physical tile";
  SmallVector<char> seen(ops.size(), false);
  for (Operation *op : ops) {
    int64_t first = 0, second = 0;
    if (isWeight) {
      auto pack = cast<PackWeightOp>(op);
      auto k = constantIndexValue(pack.getKTile());
      auto n = constantIndexValue(pack.getNTile());
      if (!k || !n)
        return anchor->emitError()
               << "diagnostic tail weight bridge indices must be constant";
      first = *n;
      second = *k;
    } else {
      auto pack = cast<PackActOp>(op);
      auto m = constantIndexValue(pack.getRow());
      auto k = constantIndexValue(pack.getCol());
      if (!m || !k)
        return anchor->emitError()
               << "diagnostic tail activation bridge indices must be constant";
      first = *m;
      second = *k;
    }
    if (first < 0 || first >= outerTiles || second < 0 || second >= kTiles)
      return anchor->emitError()
             << "diagnostic tail bridge index is outside the padded grid";
    int64_t slot = first * kTiles + second;
    if (seen[slot])
      return anchor->emitError()
             << "diagnostic tail bridge contains a duplicate tile";
    seen[slot] = true;
  }
  return success();
}

static std::optional<UnpackBridge> findUnpackBridge(MatmulOp matmul) {
  UnpackBridge bridge;
  scf::ForOp commonLoop;
  for (OpOperand &use : matmul.getOuts().getUses()) {
    Operation *owner = use.getOwner();
    if (owner == matmul.getOperation() || isa<memref::DeallocOp>(owner))
      continue;
    bool fused = isa<UnpackAccF32Op>(owner);
    if (!fused && !isa<UnpackAccOp>(owner))
      return std::nullopt;

    if (auto loop = owner->getParentOfType<scf::ForOp>()) {
      if (commonLoop && commonLoop != loop)
        return std::nullopt;
      commonLoop = loop;
    }

    Value destination;
    Value residual;
    if (auto f16Unpack = dyn_cast<UnpackAccOp>(owner)) {
      destination = f16Unpack.getDst();
    } else {
      auto f32Unpack = cast<UnpackAccF32Op>(owner);
      destination = f32Unpack.getDst();
      residual = f32Unpack.getResidual();
    }
    if (!bridge.destination) {
      bridge.destination = destination;
      bridge.residual = residual;
      bridge.fused = fused;
    } else if (bridge.destination != destination ||
               bridge.residual != residual || bridge.fused != fused) {
      return std::nullopt;
    }
    bridge.ops.push_back(owner);
  }
  if (!bridge.destination || bridge.ops.empty())
    return std::nullopt;

  if (commonLoop) {
    if (commonLoop.getNumResults() == 0) {
      if (commonLoop.getNumRegionIterArgs() != 0)
        return std::nullopt;
    } else if (commonLoop.getNumResults() != 1 ||
               commonLoop.getNumRegionIterArgs() != 1 ||
               commonLoop.getInitArgs().empty() ||
               bridge.destination != commonLoop.getRegionIterArg(0)) {
      return std::nullopt;
    }
    for (Operation &inner : commonLoop.getBody()->without_terminator())
      if (!isa<UnpackAccOp, UnpackAccF32Op>(&inner) &&
          inner.getName().getDialectNamespace() != "arith")
        return std::nullopt;
    bridge.loop = commonLoop;
    if (commonLoop.getNumResults() != 0) {
      bridge.destination = commonLoop.getInitArgs()[0];
      bridge.result = commonLoop.getResult(0);
    }
  }
  return bridge;
}

/// NOT-A-DECISION: a safety bound on the pure-definition walk below, for the
/// reason `kFoldDepth` in HmxVectorReadoutPass.cpp gives. The chains that matter
/// in production are one or two ops -- a Triton block-pointer offset is
/// `arith.muli` under `memref.reinterpret_cast` -- so 16 is an order of magnitude
/// of headroom, and a deeper chain declines rather than relocating an unbounded
/// amount of IR above the matmul.
constexpr unsigned kHoistDepth = 16;

/// The ops that would have to move above `anchor` for `value` to be defined
/// there, in dependency order, or "cannot be made available".
///
/// WHY THIS EXISTS. A `memref.reinterpret_cast` is `Pure`
/// (mlir/Dialect/MemRef/IR/MemRefOps.td:1490), so it can always be evaluated
/// earlier. Nothing forces it to be *evaluated* earlier, and no canonicalizer
/// runs between `convert-bufferization-to-memref` and this pass (see the pass
/// order in LinalgToLLVMPass.cpp:502-537), so where the op was materialised is
/// where it stays. A destination whose offset is the constant 0 happens to be
/// materialised in the entry block, which is why the whole-matrix matmul hoists;
/// a destination offset by `tl.program_id` is materialised where
/// `materialize_in_destination` put it, *after* the matmul, which is why the
/// grid-tiled matmul of the same 1024x512x64 shape did not. Measured 2026-10-03:
/// the only IR difference between the two was that destination's position, and
/// moving it above the matmul is what turned the vector read-out on for the
/// grid shape.
///
/// `out` receives the ops in post-order -- an op always follows every op it
/// reads -- so splicing them in list order above `anchor` is correct.
/// `dominance` MUST be built over the IR as it stands now. It is the only thing
/// that distinguishes "already available" from "must be moved", and reading it
/// after a relocation would answer for the wrong block.
static bool collectHoistableDefs(Value value, Operation *anchor,
                                 DominanceInfo &dominance,
                                 SmallVectorImpl<Operation *> &out,
                                 unsigned depth = 0) {
  // Already available: nothing to move, and this is the whole reason the
  // dominance test comes FIRST rather than only at the leaves. A function
  // argument is a block argument with no defining op, and it is the operand
  // every production destination is ultimately built from; treating "no
  // defining op" as a decline before asking about dominance rejects exactly the
  // case this function exists for.
  if (dominance.dominates(value, anchor))
    return true;
  if (depth > kHoistDepth)
    return false;
  Operation *def = value.getDefiningOp();
  // Available by neither route and with no definition to move: an induction
  // variable or a loop-carried value, whose region would have to move instead.
  if (!def)
    return false;
  // Only same-block definitions are movable. A definition in an ENCLOSING block
  // already dominates, so it is answered above; a definition in a NESTED region
  // belongs to an op this pass does not own and must not be dragged out of it.
  if (def->getParentOp() != anchor->getParentOp())
    return false;
  // Pure, speculatable, and region-free. `mlir::isPure` does not look inside
  // regions, so the region check is not redundant with it: an op whose trait says
  // pure but whose body has effects would pass the trait and fail here.
  if (!mlir::isPure(def) || def->getNumRegions() != 0)
    return false;
  for (Value operand : def->getOperands())
    if (!collectHoistableDefs(operand, anchor, dominance, out, depth + 1))
      return false;
  // Two operands can share a chain. Moving an op twice is harmless, but this list
  // is read as the set of ops to move, so it is kept a set.
  if (!llvm::is_contained(out, def))
    out.push_back(def);
  return true;
}

/// The accumulator read-out a staged tile loop may absorb into its M
/// iterations: the one unpack loop that walks AR row `m` for `m` in [0, Mt).
struct RowUnpack {
  UnpackAccOp op;
  scf::ForOp loop;
};

/// Find the read-out that `emitStageLoop` can hoist, or nothing.
///
/// The read-out is a strictly serial tail in the unstaged shape: the matmul
/// retires, then a second loop walks every AR row and unpacks it. LWP measured
/// that tail at 39.79% of a 1024x512x64 kernel at depth 2 -- ~24 us of vector
/// work that overlaps nothing, because there is nothing left to overlap with.
///
/// Moving it inside the M loop is safe by construction, not by scheduling luck:
///
///   * `hmx.mma` writes no memory at all -- its accumulator is the implicit
///     engine register on `Hmx_EngineResource` (HmxOps.td:498-502). The
///     accumulator's *memory* image is written only by `hmx.acc_read`, at AR
///     (row m, col n), and `unpack_acc` reads AR (row m, col c). `croutonAddr`
///     is `base + (row * rowStride + col) * 2048` (HmxToLLVMPass.cpp:731-761),
///     so two different `m` are byte-disjoint croutons.
///   * `hmx.unpack_acc` carries only `MemRead, MemWrite` on the default
///     resource (HmxOps.td:290-293) -- no `Hmx_EngineResource` effect -- so it
///     neither conflicts with nor has to be ordered against `acc_clear`, `mma`
///     or `acc_read` on the implicit register.
///   * Within one iteration the unpack is emitted *after* the inner N loop,
///     which is where the `acc_read`s that fill AR row `m` live. So the read
///     follows the write in the same block, with no cross-iteration edge.
///
/// Row coverage then needs no new argument: the hoisted unpack sits in the same
/// iteration body as the `acc_read` it reads, so it inherits exactly the rows
/// the pipeliner already guarantees `acc_read` covers -- the kernel's
/// `m` in [0, Mt-1) and, at depth 2, the peeled epilogue's last row. Every AR
/// row is unpacked once and once only.
///
/// Nothing here is required. Anything other than this one shape leaves the
/// original loop exactly where it was, which is why every condition below is a
/// decline rather than a repair: `findUnpackBridge` already accepts several
/// read-out forms, and reinterpreting a form it did not recognise as this one
/// would move an op whose coverage nobody has checked.
///
/// The ONE thing this function does move is the pure chain that computes a
/// captured value (see `collectHoistableDefs`). That is not a repair of the
/// read-out's shape; it is a relocation of descriptor arithmetic the anchor
/// could have used all along, and it is what the production grid shape needs.
/// Everything is decided before anything moves: the walk below only appends to
/// `hoistable`, and the move happens once every other condition has passed, so a
/// shape that declines for any other reason leaves the IR byte-identical.
static std::optional<RowUnpack> findRowUnpack(IRRewriter &rewriter, MatmulOp matmul,
                                              Value ar, int64_t Mt,
                                              Operation *anchor) {
  auto bridge = findUnpackBridge(matmul);
  // No loop means the read-out is a bare op or a carried (DPS) loop; both are
  // forms this pass does not rewrite. The fused f32 residual form is the
  // diagnostic tail ABI, whose read-out is emitted by the diagnostic emitters
  // and must stay where they put it.
  if (!bridge || !bridge->loop || bridge->fused)
    return std::nullopt;
  scf::ForOp loop = bridge->loop;
  // The loop must carry no result and no region argument: the hoisted op is a
  // clone that outlives the loop, so a threaded value would have nowhere to go.
  if (loop.getNumResults() != 0 || loop.getNumRegionIterArgs() != 0)
    return std::nullopt;
  // Exactly one unpack, so "one per row" is a statement about one op and not
  // about a body this function has not fully understood.
  if (bridge->ops.size() != 1 || !isa<UnpackAccOp>(bridge->ops.front()))
    return std::nullopt;
  auto unpack = cast<UnpackAccOp>(bridge->ops.front());

  // It must read the accumulator this matmul writes, and the induction variable
  // must be the row -- that identity is the whole hoist. Reject any other use
  // of the induction variable, so the clone's remaining operands are provably
  // loop-invariant.
  Value iv = loop.getInductionVar();
  if (unpack.getSrc() != ar || unpack.getRow() != iv)
    return std::nullopt;
  // `row` and `col` are the only operands that could be the induction variable;
  // `col` must not be, or the clone would keep a use of a value that does not
  // exist inside the tile loop.
  if (unpack.getCol() == iv)
    return std::nullopt;

  // The traversal must be exactly [0, Mt) step 1, which is what makes
  // "unpack row m" mean "unpack the row `acc_read` wrote for tile m". A bound
  // the tile loop cannot match is a mismatch to decline, not to clamp.
  std::optional<int64_t> lower = constantIndexValue(loop.getLowerBound());
  std::optional<int64_t> upper = constantIndexValue(loop.getUpperBound());
  std::optional<int64_t> step = constantIndexValue(loop.getStep());
  if (!lower || !upper || !step || *lower != 0 || *upper != Mt || *step != 1)
    return std::nullopt;

  // The clone is emitted at the matmul, which in production is *before* the
  // read-out loop, so everything it captures must be available there.
  //
  // "AVAILABLE", not merely "already defined there": a destination materialised
  // *after* the matmul but out of pure, region-free ops is MOVED above it
  // rather than declined. That is the production grid case and it is not a
  // repair -- `memref.reinterpret_cast` is `Pure`, so the clone computes exactly
  // the same descriptor either way; only its position in the block changes. The
  // two shapes are otherwise identical: measured 2026-10-03, the whole-matrix
  // and grid-tiled forms of the same 1024x512x64 matmul differ in the emitted
  // read-out by nothing except where the destination's `reinterpret_cast` sat,
  // and that difference alone decided whether `hmx-vector-readout` had anything
  // to rewrite (the grid form kept a separate read-out loop, whose loop carries
  // no `acc_read`, which that pass requires in the loop it attaches to).
  //
  // A value that cannot be moved -- a block argument, anything defined inside a
  // nested region, anything with a side effect -- is still a decline, and it
  // declines with the IR untouched.
  DominanceInfo dominance(anchor);
  SmallVector<Operation *> hoistable;
  for (Value operand : unpack->getOperands())
    if (operand != iv &&
        !collectHoistableDefs(operand, anchor, dominance, hoistable))
      return std::nullopt;

  // Every condition has passed, so relocating the captured chain cannot
  // invalidate a decision this function made: nothing is rebuilt or re-checked,
  // two pure ops change position, and `hoistable` is in dependency order. The
  // DominanceInfo above is not consulted again -- it was built for the
  // pre-move IR and would answer for the wrong block.
  for (Operation *def : hoistable)
    rewriter.moveOpBefore(def, anchor);
  LLVM_DEBUG({
    for (Operation *def : hoistable)
      llvm::dbgs() << "hmx-partition: hoisting the read-out's "
                   << def->getName().getStringRef()
                   << " above the matmul so the read-out can ride the tile "
                      "loop\n";
  });

  return RowUnpack{unpack, loop};
}

/// Emit one rectangular tile region. The region is deliberately a plain SCF
/// loop nest: a tail edge is outside the ordinary activation staging pipeline,
/// so it must not be accidentally handed to `scf::pipelineForLoop`.
static void emitTileRegion(IRRewriter &rewriter, Location loc, Value bias,
                           Value act, Value wt, Value ar, int64_t mBegin,
                           int64_t mEnd, int64_t nBegin, int64_t nEnd,
                           int64_t kEnd, int64_t batch, Operation *&cursor) {
  if (mBegin >= mEnd || nBegin >= nEnd || kEnd <= 0)
    return;

  // `cursor` is the previously emitted region. Insert after it so the
  // fixed full/M-edge/N-edge/corner order is preserved; inserting at the
  // operation itself reverses every subsequent region and can make unpack
  // observe an accumulator before its acc_read.
  rewriter.setInsertionPointAfter(cursor);
  auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto mStart = arith::ConstantIndexOp::create(rewriter, loc, mBegin);
  auto mStop = arith::ConstantIndexOp::create(rewriter, loc, mEnd);
  auto nStart = arith::ConstantIndexOp::create(rewriter, loc, nBegin);
  auto nStop = arith::ConstantIndexOp::create(rewriter, loc, nEnd);

  auto mLoop =
      scf::ForOp::create(rewriter, loc, mStart, mStop, step, ValueRange{});
  rewriter.setInsertionPointToStart(mLoop.getBody());
  auto nLoop =
      scf::ForOp::create(rewriter, loc, nStart, nStop, step, ValueRange{});
  rewriter.setInsertionPointToStart(nLoop.getBody());
  AccClearOp::create(rewriter, loc);
  emitMmaKLoop(rewriter, loc, act, wt, mLoop.getInductionVar(),
               nLoop.getInductionVar(), kEnd, batch);
  AccReadOp::create(rewriter, loc, bias, ar, mLoop.getInductionVar(),
                    nLoop.getInductionVar(), rewriter.getI32IntegerAttr(0));
  rewriter.setInsertionPointAfter(mLoop);
  cursor = mLoop.getOperation();
}

/// The fixed physical partition for a tail plan: the full M/N rectangle, then
/// the three disjoint peeled rectangles (M edge, N edge, and their corner).
/// K is always walked through the padded grid; the K-edge lanes are already
/// zero-filled by the pack contract and are not a separate M/N region.
static Operation *emitPeeledEdgeTileLoop(IRRewriter &rewriter, Location loc,
                                         Value bias, MatmulOp op,
                                         const TailGrid &grid, int64_t batch) {
  Operation *cursor = op.getOperation();
  auto emit = [&](int64_t mBegin, int64_t mEnd, int64_t nBegin, int64_t nEnd) {
    emitTileRegion(rewriter, loc, bias, op.getLhs(), op.getRhs(), op.getOuts(),
                   mBegin, mEnd, nBegin, nEnd, grid.kTiles, batch, cursor);
  };

  // The full rectangle is the ordinary fast region. It is still serial in this
  // diagnostic slice; staging and its budget/pipeliner contract remain a
  // separate optimization, not a property of the edge ABI.
  emit(0, grid.mFullTiles, 0, grid.nFullTiles);
  if (grid.mFullTiles < grid.mTiles)
    emit(grid.mFullTiles, grid.mTiles, 0, grid.nFullTiles);
  if (grid.nFullTiles < grid.nTiles) {
    emit(0, grid.mFullTiles, grid.nFullTiles, grid.nTiles);
    if (grid.mFullTiles < grid.mTiles)
      emit(grid.mFullTiles, grid.mTiles, grid.nFullTiles, grid.nTiles);
  }
  return cursor;
}

static LogicalResult verifyDiagnosticRowMajor(Operation *anchor, Value value,
                                              StringRef name) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 2)
    return anchor->emitError()
           << name << " tail bridge requires a static rank-2 memref";
  SmallVector<int64_t, 2> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != 2 || ShapedType::isDynamic(strides[0]) ||
      ShapedType::isDynamic(strides[1]) || strides[1] != 1 ||
      strides[0] < type.getDimSize(1))
    return anchor->emitError()
           << name << " tail bridge requires a static row-major stride";
  return success();
}

static LogicalResult verifyDiagnosticMatrixShape(Operation *anchor, Value value,
                                                 int64_t rows, int64_t cols,
                                                 int64_t paddedRows,
                                                 int64_t paddedCols,
                                                 StringRef name) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type ||
      (type.getDimSize(0) != rows && type.getDimSize(0) != paddedRows) ||
      (type.getDimSize(1) != cols && type.getDimSize(1) != paddedCols))
    return anchor->emitError()
           << name << " shape does not match tail_plan " << "logical [" << rows
           << ", " << cols << "] or padded [" << paddedRows << ", "
           << paddedCols << "]";
  return success();
}

static bool diagnosticValueDominates(const DominanceInfo &dominance,
                                     Value value, Operation *insertionPoint) {
  if (auto blockArgument = dyn_cast<BlockArgument>(value))
    return blockArgument.getOwner()->findAncestorOpInBlock(*insertionPoint);
  Operation *definition = value.getDefiningOp();
  return definition && dominance.properlyDominates(definition, insertionPoint);
}

/// Move a diagnostic insertion cursor after any same-block definitions it uses.
/// A value that is not already available at the cursor is not silently
/// captured: either its definition is in this block and can be moved past, or
/// the diagnostic bridge is rejected as non-dominating.
static LogicalResult
placeDiagnosticCursorAfterValues(Operation *&cursor, ArrayRef<Value> values,
                                 const DominanceInfo &dominance,
                                 Operation *anchor, StringRef name) {
  for (Value value : values) {
    if (!value)
      continue;
    Operation *definition = value.getDefiningOp();
    if (!definition) {
      if (!diagnosticValueDominates(dominance, value, cursor))
        return anchor->emitError()
               << name
               << " value does not dominate the diagnostic insertion point";
      continue;
    }
    if (definition->getBlock() == cursor->getBlock()) {
      if (!definition->isBeforeInBlock(cursor))
        cursor = definition;
      continue;
    }
    if (!diagnosticValueDominates(dominance, value, cursor))
      return anchor->emitError()
             << name
             << " value is defined outside the dominating diagnostic block";
  }
  return success();
}

static void setDiagnosticDecisionId(Operation *op,
                                    std::optional<int64_t> decisionId) {
  if (decisionId)
    op->setAttr(
        kHmxDecisionIdAttr,
        IntegerAttr::get(IntegerType::get(op->getContext(), 64), *decisionId));
}

/// Rebuild the direct input bridges as four homogeneous regions per operand.
/// Full regions use the existing pack op without valid attrs; only the peeled
/// rectangle receives the explicit static valid extents.
static LogicalResult emitDiagnosticInputBridges(
    IRRewriter &rewriter, Location loc, MatmulOp op, const TailGrid &grid,
    std::optional<int64_t> decisionId, Operation *&cursor,
    SmallVectorImpl<Operation *> &retired) {
  auto actBridge = findPackBridge(op.getLhs(), /*isWeight=*/false);
  auto weightBridge = findPackBridge(op.getRhs(), /*isWeight=*/true);

  // A WEIGHT-RESIDENT operand arrives already in crouton form, so no
  // `hmx.pack_weight` writes it and `findPackBridge` necessarily returns null.
  // That null used to be indistinguishable from "the weight bridge is
  // malformed", which made the whole diagnostic tail path unreachable for the
  // default configuration -- `enableWeightResident` is True by default
  // (hexagon_options.py), so this was not a corner case.
  //
  // The two are told apart by asking whether ANY pack writes the array at all:
  //   * no PackWeightOp writes it  => the array's contents come from outside the
  //     pack path, i.e. it is already resident => nothing to check about a
  //     row-major SOURCE, because there is no source.
  //   * a PackWeightOp writes it but findPackBridge said no => mixed
  //     destinations / several loops / a non-arith loop body => refuse, exactly
  //     as before. This is why the test is "any writer" and not "bridge null".
  //
  // ⚠️ THIS IS A PROXY INFERENCE, and it is only sound because of a pass-order
  // invariant that used to be unwritten (2026-10-01, independent audit). Stating
  // it here so the next reader does not have to rediscover it:
  //
  //   In the PRODUCTION pipeline, `MatmulToHmxPass` always emits the weight
  //   packs (its :1436 / :1587 / :1631 construction sites), and
  //   `WeightResidentPass` runs IMMEDIATELY before this pass
  //   (LinalgToLLVMPass.cpp:509-512 then :527-530) -- so by the time we get
  //   here, "no PackWeightOp" really does mean "WeightResidentPass replaced them
  //   with a resident array", and the other meaning cannot occur.
  //
  //   That invariant is NOT available to a STANDALONE `hmx-partition`
  //   invocation, which HmxPartitionPass.cpp:1751 explicitly supports and which
  //   is how every lit test in this directory runs. There the audit MEASURED two
  //   accepting counterexamples: an rhs that is an untouched `memref.alloc`
  //   (engine reads uninitialised memory), and an rhs filled by a row-major
  //   `hmx.unpack_acc` (a crouton-order mixup, the mirror image of the
  //   HmxToLLVMPass `rowStride` hole -- and `rowStride` cannot see it, because it
  //   only sees rank-2 leaf buffers). Both lowered to four real `hmx.mma`.
  //
  // ⇒ The standalone form is a "no pack means resident" CLAIM, not a proof. The
  //   cheap strengthening (require the rhs to be a block argument, an alloc, or
  //   a reinterpret_cast -- i.e. something WeightResidentPass could have
  //   produced) is deliberately NOT done here: it would need its own
  //   counterexample sweep, and a wrong tightening would re-break the default
  //   configuration this change exists to unlock. If you tighten it, tighten it
  //   with a lit case per shape you exclude.
  auto anyWeightPackWrites = [&](Value array) {
    for (OpOperand &use : array.getUses())
      if (isa<PackWeightOp>(use.getOwner()))
        return true;
    return false;
  };
  const bool weightResident = !weightBridge && !anyWeightPackWrites(op.getRhs());
  if (weightResident) {
    PackBridge resident;
    resident.buffer = op.getRhs();
    weightBridge = resident;
  }
  if (!actBridge || !weightBridge)
    return op.emitError("diagnostic tail partition requires direct activation "
                        "and weight pack bridges");
  DominanceInfo dominance(op.getOperation());
  if (!diagnosticValueDominates(dominance, actBridge->source,
                                op.getOperation()) ||
      (!weightResident &&
       !diagnosticValueDominates(dominance, weightBridge->source,
                                 op.getOperation())))
    return op.emitError("diagnostic tail partition requires pack sources that "
                        "dominate the matmul");
  // Coverage is counted over the packs a bridge owns. A resident weight owns
  // none, and that is correct rather than a gap: there is nothing to cover,
  // because the engine reads the crouton the caller already handed it.
  if ((!actBridge->loop &&
       failed(verifyPackCoverage(op.getOperation(), actBridge->ops,
                                 /*isWeight=*/false, grid.mTiles,
                                 grid.kTiles))) ||
      (!weightResident && !weightBridge->loop &&
       failed(verifyPackCoverage(op.getOperation(), weightBridge->ops,
                                 /*isWeight=*/true, grid.nTiles, grid.kTiles))))
    return failure();
  // Likewise the two source checks are about a ROW-MAJOR source, which a
  // resident weight by definition does not have.
  if (failed(verifyDiagnosticRowMajor(op.getOperation(), actBridge->source,
                                      "activation source")) ||
      (!weightResident &&
       (failed(verifyDiagnosticRowMajor(op.getOperation(), weightBridge->source,
                                        "weight source")) ||
        failed(verifyDiagnosticMatrixShape(
            op.getOperation(), weightBridge->source, grid.kLogical,
            grid.nLogical, grid.kTiles * layout::kTileEdge,
            grid.nTiles * layout::kTileEdge, "weight source")))) ||
      failed(verifyDiagnosticMatrixShape(
          op.getOperation(), actBridge->source, grid.mLogical, grid.kLogical,
          grid.mTiles * layout::kTileEdge, grid.kTiles * layout::kTileEdge,
          "activation source")))
    return failure();

  Value actValue = actBridge->buffer;
  Value weightValue = weightBridge->buffer;
  auto checkArrayUsers = [&](Value array, PackBridge &bridge,
                             bool isWeight) -> LogicalResult {
    for (OpOperand &use : array.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == op.getOperation() || isa<memref::DeallocOp>(owner) ||
          (bridge.loop && owner == bridge.loop.getOperation()))
        continue;
      bool ownedPack =
          (isWeight ? isa<PackWeightOp>(owner) : isa<PackActOp>(owner)) &&
          (isWeight ? cast<PackWeightOp>(owner).getDst() == array
                    : cast<PackActOp>(owner).getDst() == array) &&
          llvm::is_contained(bridge.ops, owner) &&
          (!bridge.loop || owner->getParentOfType<scf::ForOp>() == bridge.loop);
      if (ownedPack)
        continue;
      return op.emitError("diagnostic tail partition found an unowned "
                          "activation/weight bridge user");
    }
    // A carried bridge writes its region argument, but its init buffer must
    // not have a second writer outside the canonical loop. Otherwise that
    // writer could overwrite the array after the rebuilt engine read.
    if (bridge.buffer != array) {
      for (OpOperand &use : bridge.buffer.getUses()) {
        Operation *owner = use.getOwner();
        if ((bridge.loop && owner == bridge.loop.getOperation()) ||
            isa<memref::DeallocOp>(owner))
          continue;
        return op.emitError("diagnostic tail partition found an external "
                            "writer of a carried bridge buffer");
      }
    }
    return success();
  };
  if (failed(checkArrayUsers(op.getLhs(), *actBridge, /*isWeight=*/false)) ||
      failed(checkArrayUsers(op.getRhs(), *weightBridge, /*isWeight=*/true)))
    return failure();

  auto emitAct = [&](int64_t mBegin, int64_t mEnd, int64_t kBegin, int64_t kEnd,
                     std::optional<int64_t> validRows,
                     std::optional<int64_t> validCols) {
    if (mBegin >= mEnd || kBegin >= kEnd)
      return;
    rewriter.setInsertionPoint(cursor);
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto mStart = arith::ConstantIndexOp::create(rewriter, loc, mBegin);
    auto mStop = arith::ConstantIndexOp::create(rewriter, loc, mEnd);
    auto kStart = arith::ConstantIndexOp::create(rewriter, loc, kBegin);
    auto kStop = arith::ConstantIndexOp::create(rewriter, loc, kEnd);
    auto mLoop =
        scf::ForOp::create(rewriter, loc, mStart, mStop, step, ValueRange{});
    rewriter.setInsertionPointToStart(mLoop.getBody());
    auto kLoop =
        scf::ForOp::create(rewriter, loc, kStart, kStop, step, ValueRange{});
    rewriter.setInsertionPointToStart(kLoop.getBody());
    auto pack = PackActOp::create(
        rewriter, loc, TypeRange{actValue.getType()}, actValue,
        actBridge->source, mLoop.getInductionVar(), kLoop.getInductionVar(),
        IntegerAttr(),
        validRows ? rewriter.getI64IntegerAttr(*validRows) : IntegerAttr(),
        validCols ? rewriter.getI64IntegerAttr(*validCols) : IntegerAttr());
    setDiagnosticDecisionId(pack.getOperation(), decisionId);
    rewriter.setInsertionPointAfter(mLoop);
    cursor = mLoop.getOperation();
  };

  auto emitWeight = [&](int64_t nBegin, int64_t nEnd, int64_t kBegin,
                        int64_t kEnd, std::optional<int64_t> validRows,
                        std::optional<int64_t> validCols) {
    if (nBegin >= nEnd || kBegin >= kEnd)
      return;
    rewriter.setInsertionPoint(cursor);
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto nStart = arith::ConstantIndexOp::create(rewriter, loc, nBegin);
    auto nStop = arith::ConstantIndexOp::create(rewriter, loc, nEnd);
    auto kStart = arith::ConstantIndexOp::create(rewriter, loc, kBegin);
    auto kStop = arith::ConstantIndexOp::create(rewriter, loc, kEnd);
    auto nLoop =
        scf::ForOp::create(rewriter, loc, nStart, nStop, step, ValueRange{});
    rewriter.setInsertionPointToStart(nLoop.getBody());
    auto kLoop =
        scf::ForOp::create(rewriter, loc, kStart, kStop, step, ValueRange{});
    rewriter.setInsertionPointToStart(kLoop.getBody());
    auto pack = PackWeightOp::create(
        rewriter, loc, TypeRange{weightValue.getType()}, weightValue,
        weightBridge->source, kLoop.getInductionVar(), nLoop.getInductionVar(),
        IntegerAttr(),
        validRows ? rewriter.getI64IntegerAttr(*validRows) : IntegerAttr(),
        validCols ? rewriter.getI64IntegerAttr(*validCols) : IntegerAttr(),
        // Rebuilt, not re-decided: the marker rides along from the bridge's
        // own packs, so a rebuild can never silently drop the orientation.
        cast<PackWeightOp>(weightBridge->ops.front()).getSrcTransposedAttr());
    setDiagnosticDecisionId(pack.getOperation(), decisionId);
    rewriter.setInsertionPointAfter(nLoop);
    cursor = nLoop.getOperation();
  };

  const std::optional<int64_t> noValid;
  const std::optional<int64_t> full = layout::kTileEdge;
  const std::optional<int64_t> validM = grid.mTail;
  const std::optional<int64_t> validN = grid.nTail;
  const std::optional<int64_t> validK = grid.kTail;
  emitAct(0, grid.mFullTiles, 0, grid.kFullTiles, noValid, noValid);
  if (grid.mFullTiles < grid.mTiles)
    emitAct(grid.mFullTiles, grid.mTiles, 0, grid.kFullTiles, validM, full);
  if (grid.kFullTiles < grid.kTiles) {
    emitAct(0, grid.mFullTiles, grid.kFullTiles, grid.kTiles, full, validK);
    if (grid.mFullTiles < grid.mTiles)
      emitAct(grid.mFullTiles, grid.mTiles, grid.kFullTiles, grid.kTiles,
              validM, validK);
  }
  // A resident weight is already crouton: there is no row-major source to pack
  // FROM, so emitting packs for it would need a source value that does not
  // exist. Skipping is the whole point of the resident path.
  if (!weightResident) {
    emitWeight(0, grid.nFullTiles, 0, grid.kFullTiles, noValid, noValid);
    if (grid.nFullTiles < grid.nTiles)
      emitWeight(grid.nFullTiles, grid.nTiles, 0, grid.kFullTiles, full,
                 validN);
    if (grid.kFullTiles < grid.kTiles) {
      emitWeight(0, grid.nFullTiles, grid.kFullTiles, grid.kTiles, validK,
                 full);
      if (grid.nFullTiles < grid.nTiles)
        emitWeight(grid.nFullTiles, grid.nTiles, grid.kFullTiles, grid.kTiles,
                   validK, validN);
    }
  }

  auto retirePackBridge = [&](PackBridge &bridge) {
    if (bridge.loop) {
      if (bridge.loop.getNumResults() != 0)
        bridge.loop.getResult(0).replaceAllUsesWith(bridge.buffer);
      retired.push_back(bridge.loop.getOperation());
    } else {
      retired.append(bridge.ops.begin(), bridge.ops.end());
    }
  };
  retirePackBridge(*actBridge);
  retirePackBridge(*weightBridge);
  return success();
}

/// Rebuild the direct output bridge after the peeled tile regions. Full M/N
/// tiles keep the ranged row-pair read-out; every edge rectangle uses one
/// explicit single-pair op per row-pair so the valid M/N extents are carried
/// all the way to the leaf.
static LogicalResult emitDiagnosticOutputBridge(
    IRRewriter &rewriter, Location loc, MatmulOp op, const TailGrid &grid,
    UnpackBridge &bridge, std::optional<int64_t> decisionId, Operation *&cursor,
    SmallVectorImpl<Operation *> &retired) {
  if (failed(verifyDiagnosticRowMajor(op.getOperation(), bridge.destination,
                                      "unpack destination")) ||
      failed(verifyDiagnosticMatrixShape(
          op.getOperation(), bridge.destination, grid.mLogical, grid.nLogical,
          grid.mTiles * layout::kTileEdge, grid.nTiles * layout::kTileEdge,
          "unpack destination")))
    return failure();
  if (bridge.residual &&
      (failed(verifyDiagnosticRowMajor(op.getOperation(), bridge.residual,
                                       "unpack residual")) ||
       failed(verifyDiagnosticMatrixShape(
           op.getOperation(), bridge.residual, grid.mLogical, grid.nLogical,
           grid.mTiles * layout::kTileEdge, grid.nTiles * layout::kTileEdge,
           "unpack residual"))))
    return failure();

  // The epilogue allocation is allowed to follow the matmul in the
  // bufferized producer form. Do not insert the rebuilt read-out before that
  // allocation (or before a fused residual): otherwise the new loop would
  // capture a non-dominating memref. Definitions from another region are only
  // accepted when the dominance proof already holds.
  DominanceInfo dominance(op.getOperation());
  SmallVector<Value> outputValues{bridge.destination};
  if (bridge.residual)
    outputValues.push_back(bridge.residual);
  if (failed(placeDiagnosticCursorAfterValues(cursor, outputValues, dominance,
                                              op.getOperation(),
                                              "unpack destination/residual")))
    return failure();

  const std::optional<int64_t> validM = grid.mTail;
  const std::optional<int64_t> validN = grid.nTail;
  const std::optional<int64_t> noValid;
  const std::optional<int64_t> full = layout::kTileEdge;

  auto emitOp = [&](Value source, Value destination, Value residual, Value row,
                    Value col, IntegerAttr count, std::optional<int64_t> rows,
                    std::optional<int64_t> cols,
                    std::optional<int64_t> nTile) -> Operation * {
    IntegerAttr validRows =
        rows ? rewriter.getI64IntegerAttr(*rows) : IntegerAttr();
    IntegerAttr validCols =
        cols ? rewriter.getI64IntegerAttr(*cols) : IntegerAttr();
    IntegerAttr nTileAttr =
        nTile ? rewriter.getI64IntegerAttr(*nTile) : IntegerAttr();
    Operation *created = nullptr;
    if (bridge.fused) {
      auto unpack = UnpackAccF32Op::create(
          rewriter, loc, TypeRange{destination.getType()}, source, destination,
          row, col, residual, count, validRows, validCols, nTileAttr);
      created = unpack.getOperation();
    } else {
      auto unpack = UnpackAccOp::create(
          rewriter, loc, TypeRange{destination.getType()}, source, destination,
          row, col, count, validRows, validCols, nTileAttr);
      created = unpack.getOperation();
    }
    setDiagnosticDecisionId(created, decisionId);
    return created;
  };

  auto emitFull = [&](int64_t mBegin, int64_t mEnd, int64_t nBegin,
                      int64_t nEnd) -> LogicalResult {
    if (mBegin >= mEnd || nBegin >= nEnd)
      return success();
    rewriter.setInsertionPointAfter(cursor);
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto mStart = arith::ConstantIndexOp::create(rewriter, loc, mBegin);
    auto mStop = arith::ConstantIndexOp::create(rewriter, loc, mEnd);
    auto mLoop =
        scf::ForOp::create(rewriter, loc, mStart, mStop, step, ValueRange{});
    rewriter.setInsertionPointToStart(mLoop.getBody());
    Value mTile = mLoop.getInductionVar();
    Value zeroCol = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (int64_t nTile = nBegin; nTile < nEnd; ++nTile) {
      if (bridge.fused) {
        // Keep diagnostic full f32 read-out on the bounds-safe row-pair leaf.
        // It makes this correctness path independent of destination alignment
        // and row-stride choices; this is diagnostic-only and does not admit a
        // production tail fast path.
        auto pairStop =
            arith::ConstantIndexOp::create(rewriter, loc, layout::kCroutonPair);
        auto pairLoop = scf::ForOp::create(
            rewriter, loc, arith::ConstantIndexOp::create(rewriter, loc, 0),
            pairStop, step, ValueRange{});
        rewriter.setInsertionPointToStart(pairLoop.getBody());
        emitOp(op.getOuts(), bridge.destination, bridge.residual, mTile,
               pairLoop.getInductionVar(), IntegerAttr(), full, full, nTile);
        rewriter.setInsertionPointAfter(pairLoop.getOperation());
      } else {
        Operation *created =
            emitOp(op.getOuts(), bridge.destination, bridge.residual, mTile,
                   zeroCol, rewriter.getI64IntegerAttr(layout::kCroutonPair),
                   noValid, noValid, nTile);
        rewriter.setInsertionPointAfter(created);
      }
    }
    rewriter.setInsertionPointAfter(mLoop);
    cursor = mLoop.getOperation();
    return success();
  };

  auto emitEdge = [&](int64_t mBegin, int64_t mEnd, int64_t nBegin,
                      int64_t nEnd, std::optional<int64_t> rows,
                      std::optional<int64_t> cols) -> LogicalResult {
    if (mBegin >= mEnd || nBegin >= nEnd)
      return success();
    if (!cols || *cols <= 0 || *cols > layout::kTileEdge)
      return op.emitError("diagnostic tail output edge requires valid columns");
    rewriter.setInsertionPointAfter(cursor);
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto mStart = arith::ConstantIndexOp::create(rewriter, loc, mBegin);
    auto mStop = arith::ConstantIndexOp::create(rewriter, loc, mEnd);
    auto pairStop =
        arith::ConstantIndexOp::create(rewriter, loc, layout::kCroutonPair);
    auto mLoop =
        scf::ForOp::create(rewriter, loc, mStart, mStop, step, ValueRange{});
    rewriter.setInsertionPointToStart(mLoop.getBody());
    Value mTile = mLoop.getInductionVar();
    for (int64_t nTile = nBegin; nTile < nEnd; ++nTile) {
      auto pairLoop = scf::ForOp::create(
          rewriter, loc, arith::ConstantIndexOp::create(rewriter, loc, 0),
          pairStop, step, ValueRange{});
      rewriter.setInsertionPointToStart(pairLoop.getBody());
      emitOp(op.getOuts(), bridge.destination, bridge.residual, mTile,
             pairLoop.getInductionVar(), IntegerAttr(), rows, cols, nTile);
      rewriter.setInsertionPointAfter(pairLoop.getOperation());
    }
    rewriter.setInsertionPointAfter(mLoop);
    cursor = mLoop.getOperation();
    return success();
  };

  if (failed(emitFull(0, grid.mFullTiles, 0, grid.nFullTiles)))
    return failure();
  if (grid.mFullTiles < grid.mTiles &&
      failed(emitEdge(grid.mFullTiles, grid.mTiles, 0, grid.nFullTiles, validM,
                      full)))
    return failure();
  if (grid.nFullTiles < grid.nTiles) {
    if (failed(emitEdge(0, grid.mFullTiles, grid.nFullTiles, grid.nTiles, full,
                        validN)))
      return failure();
    if (grid.mFullTiles < grid.mTiles &&
        failed(emitEdge(grid.mFullTiles, grid.mTiles, grid.nFullTiles,
                        grid.nTiles, validM, validN)))
      return failure();
  }

  if (bridge.loop) {
    if (bridge.result)
      bridge.result.replaceAllUsesWith(bridge.destination);
    retired.push_back(bridge.loop.getOperation());
  } else {
    retired.append(bridge.ops.begin(), bridge.ops.end());
  }
  return success();
}

/// The compute half of one tile: pack this tile's 32 x K source rows into the
/// one crouton-row scratch, then one (acc_clear, K mmas, acc_read) per N tile.
/// This is the compute part of both source loops' bodies: the staged loop,
/// where the pipeliner clones it into the kernel and the peeled epilogue (the
/// caller has already emitted the await and passes the awaited slot as
/// `packSrc`), and the folded unstaged serial loop, where `packSrc` is the
/// per-tile view of the bridge's row-major source.
///
/// The scratch is always crouton row 0: `packSrc` holds exactly the 32 source
/// rows of tile `m` (a slot the DMA filled, or a view at that row), and the
/// pack and its mma are in the same iteration, so the crouton row index the
/// engine reads is 0 regardless of which output tile this iteration computes.
/// The output tile row `m` is still what `acc_read` writes. Rewriting the whole
/// scratch every iteration costs nothing extra -- the pack has to write every
/// crouton of the tile anyway, and no other iteration's data is live in it --
/// and it is what lets one buffer (not the whole array) hold the activation.
static void emitTileCompute(IRRewriter &rewriter, Location loc, Value bias,
                            Value scratch, Value wt, Value ar, Value m,
                            Value packSrc, Value c0, Value c1, int64_t Kt,
                            int64_t batch, Value cNt,
                            std::optional<int64_t> decisionId) {
  // The pack's destination is the scratch itself: its crouton grid is one row
  // tall, so the destination crouton is (0, k) while the source block is
  // (row 0, column k) of `packSrc`. The scratch's contiguous axis is K,
  // so one ranged pack covers the whole K run -- the same shape the row-major
  // bridge uses (`packCroutonsWithLeaves`).
  auto scratchType = cast<MemRefType>(scratch.getType());
  emitPackAct(rewriter, loc, scratch, packSrc, c0, c0, decisionId,
              rewriter.getI64IntegerAttr(scratchType.getDimSize(1)));

  auto nLoop = scf::ForOp::create(rewriter, loc, c0, cNt, c1, ValueRange{});
  rewriter.setInsertionPointToStart(nLoop.getBody());
  AccClearOp::create(rewriter, loc);
  emitMmaKLoop(rewriter, loc, scratch, wt, c0, nLoop.getInductionVar(), Kt,
               batch);
  AccReadOp::create(rewriter, loc, bias, ar, m, nLoop.getInductionVar(),
                    rewriter.getI32IntegerAttr(0));
  rewriter.setInsertionPointAfter(nLoop);
}

/// Try to fold the whole-array activation pack bridge in front of an unstaged
/// serial matmul into the m-tile loop itself: one pack per m-tile, reading a
/// per-tile view of the bridge's row-major SOURCE into a one-row crouton
/// scratch, the mmas reading that scratch, and the whole-array bridge -- its
/// pack loop, its packs, the array allocation and its deallocations -- retired
/// the way the staged path retires it.
///
/// WHY. The unstaged serial tile loop is emitted with the bridge in front of
/// it: the packs fill the whole activation array before the first mma, so
/// producer layout work and engine work never share an iteration, and
/// thread-role-partition's verdict for the kernel is `role-split-nopack` --
/// there is no per-tile producer stream to hand to a second thread. Folding
/// the pack into the m-tile loop is what gives the serial form that stream:
/// the pack sits directly in the m-loop with the engine ops one loop deeper
/// in the n-loop it drives, the same iteration shape every staged arm's
/// compute half already has, minus the DMA stage/await. No ring, no
/// pipeliner, no hand-written peel: the serial source loop IS the folded
/// form.
///
/// WHAT MAKES IT SOUND. The pack's contract maps crouton (row, col) of `dst`
/// to block (row, col) of `src` with one coordinate pair, so a one-row
/// scratch must be packed from a source that holds exactly one tile's rows.
/// The per-tile view (a `memref.subview` at element row m * 32) is that
/// source: pure descriptor arithmetic, no data movement, and the strided
/// source the pack leaf already supports. The ranged pack with count = Kt is
/// the same leaf call the bridge itself makes, so the pack volume is
/// unchanged -- the bridge's Mt row-packs become the loop's Mt row-packs --
/// and the scratch is one crouton row where the array was Mt of them.
///
/// WHEN IT DECLINES. The fold re-executes the pack with the tile loop, so a
/// matmul an outer loop re-executes (flash attention's per-chunk Q@K and P@V,
/// an m-blocked contraction's per-block matmuls) must keep its bridge: a
/// hoisted bridge amortizes its packs across the outer iterations, and
/// folding it would re-pay them per iteration. That is the same condition the
/// staged read-out channel uses (`readoutChannelFor`). The bridge must also
/// be the canonical single-source, packs-plus-arith shape `findPackBridge`
/// accepts, with no reader of the array besides this matmul, the bridge's own
/// packs and the deallocations, and a source that describes this matmul's
/// crouton grid. A shape that declines keeps today's serial form exactly, and
/// every decline that had a bridge behind it names its reason in a remark --
/// only the no-bridge-at-all form stays silent, because there is nothing
/// being declined there (a chained read-out, a bare allocation: normal
/// serial shapes, not refused ones).
static LogicalResult tryFoldSerialActivationBridge(
    IRRewriter &rewriter, Location opLoc, Value bias, MatmulOp op,
    const TileShape &shape, int64_t batch, func::FuncOp func,
    int64_t vtcmBudget, bool &folded) {
  folded = false;
  Value act = op.getLhs();

  auto anyActPackWrites = [&](Value array) {
    for (OpOperand &use : array.getUses())
      if (isa<PackActOp>(use.getOwner()))
        return true;
    return false;
  };
  auto bridge = findPackBridge(act, /*isWeight=*/false);
  if (!bridge) {
    if (anyActPackWrites(act))
      op.emitRemark("HMX serial pack fold not applied: the activation "
                    "bridge is not the canonical single-source pack loop, so "
                    "the whole-array pack stays in front of the tile loop");
    return success();
  }

  // The array is retired with the bridge, so the only readers allowed are the
  // bridge's own packs, this matmul, and its deallocations -- the same
  // ownership the staged rewrite demands.
  for (OpOperand &u : act.getUses()) {
    Operation *owner = u.getOwner();
    if (owner == op.getOperation() ||
        (isa<PackActOp>(owner) && cast<PackActOp>(owner).getDst() == act) ||
        isa<memref::DeallocOp, scf::YieldOp>(owner))
      continue;
    op.emitRemark("HMX serial pack fold not applied: the activation array "
                  "has a reader the fold does not own");
    return success();
  }
  // A carried bridge writes its region argument; the init buffer underneath
  // must not have a second user either, or erasing the bridge would orphan a
  // reader of a buffer that no longer exists.
  if (bridge->buffer != act) {
    for (OpOperand &u : bridge->buffer.getUses()) {
      Operation *owner = u.getOwner();
      if (owner == bridge->loop.getOperation() ||
          isa<memref::DeallocOp>(owner))
        continue;
      op.emitRemark("HMX serial pack fold not applied: the bridge's "
                    "activation buffer has an external user");
      return success();
    }
  }

  // The source must be the static rank-2 row-major matrix the bridge's own
  // packs address, sized to this matmul's crouton grid, and the weight must
  // carry the same K run the scratch will hold.
  auto srcType = dyn_cast<MemRefType>(bridge->source.getType());
  auto wtType = dyn_cast<MemRefType>(op.getRhs().getType());
  if (!srcType || srcType.getRank() != 2 || !srcType.hasStaticShape() ||
      !dtype::isAdmittedFloat(srcType.getElementType()) || !wtType ||
      srcType.getDimSize(0) != shape.m * layout::kTileEdge ||
      srcType.getDimSize(1) != shape.k * layout::kTileEdge ||
      weightKTiles(wtType) != shape.k) {
    op.emitRemark("HMX serial pack fold not applied: the activation "
                  "bridge's source does not describe this matmul's crouton "
                  "grid");
    return success();
  }

  // Tail valid extents select the bounds-safe per-block leaf; the folded
  // ranged pack does not model them, so a bridge that carries them keeps its
  // whole-array form.
  for (Operation *packOp : bridge->ops) {
    auto pack = cast<PackActOp>(packOp);
    if (pack.getValidRows() || pack.getValidCols()) {
      op.emitRemark("HMX serial pack fold not applied: the bridge packs "
                    "carry tail valid extents the folded ranged pack does "
                    "not model");
      return success();
    }
  }

  // The fold re-executes the pack with the tile loop, so a matmul an outer
  // loop re-executes keeps its (possibly hoisted, amortized) bridge.
  for (Operation *ancestor = op->getParentOp(); ancestor != nullptr;
       ancestor = ancestor->getParentOp())
    if (isa<scf::ForOp, scf::ForallOp>(ancestor)) {
      op.emitRemark("HMX serial pack fold not applied: an outer loop "
                    "re-executes this matmul, so the whole-array bridge "
                    "amortizes its packs across iterations and folding would "
                    "re-pay them per iteration");
      return success();
    }

  // The array must be a plain allocation this pass can retire alongside the
  // bridge; anything else (an argument, a view) is not the fold's to free.
  auto arrayAlloc = bridge->buffer.getDefiningOp<memref::AllocOp>();
  if (!arrayAlloc) {
    op.emitRemark("HMX serial pack fold not applied: the activation array "
                  "is not a memref allocation the fold can retire");
    return success();
  }

  // Budget, staged-path style: the whole array comes back with the bridge,
  // and the scratch is one crouton row -- Kt of the array's Mt rows -- so a
  // bridge that fit always leaves room for the fold. Computing it rather than
  // assuming it keeps that a fact about this IR: the scratch is charged
  // against the same room the staged path would compute.
  auto actType = cast<MemRefType>(act.getType());
  int64_t actBytes = actType.getNumElements() * 2;
  int64_t scratchBytes = shape.k * layout::kCroutonBytes;
  int64_t room = hmx::vtcm::roomBytes(func, vtcmBudget, actBytes);
  if (scratchBytes > room) {
    op.emitRemark("HMX serial pack fold not applied: the crouton-row "
                  "scratch needs ")
           << scratchBytes << " bytes of VTCM, only " << room << " are free";
    return success();
  }

  std::optional<int64_t> decisionId;
  if (failed(readDecisionId(op.getOperation(), decisionId)))
    return failure();

  // The folded serial source loop: pack(m) at the top of the m-tile
  // iteration, the engine ops in the n-loop it drives. The scratch is one
  // crouton row, re-filled (not re-allocated) every tile; the per-tile view
  // replaces the DMA slot the staged form packs from.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  int64_t K = shape.k * layout::kTileEdge;
  auto scratchType = MemRefType::get(
      {1, shape.k, layout::kCroutonPair, layout::kCroutonCol,
       layout::kCroutonHalf},
      dtype::croutonElementType(rewriter.getContext()), AffineMap{},
      hexagon::VTCM_ADDRESS_SPACE);
  Value scratch =
      memref::AllocOp::create(rewriter, opLoc, scratchType, ValueRange{});
  auto c0 = arith::ConstantIndexOp::create(rewriter, opLoc, 0);
  auto c1 = arith::ConstantIndexOp::create(rewriter, opLoc, 1);
  auto cTileEdge =
      arith::ConstantIndexOp::create(rewriter, opLoc, layout::kTileEdge);
  auto cNt = arith::ConstantIndexOp::create(rewriter, opLoc, shape.n);
  auto cMt = arith::ConstantIndexOp::create(rewriter, opLoc, shape.m);

  auto mLoop = scf::ForOp::create(rewriter, opLoc, c0, cMt, c1, ValueRange{});
  rewriter.setInsertionPointToStart(mLoop.getBody());
  Value m = mLoop.getInductionVar();
  Value row = arith::MulIOp::create(rewriter, opLoc, m, cTileEdge);
  SmallVector<OpFoldResult> tileOffsets{row, rewriter.getIndexAttr(0)};
  SmallVector<OpFoldResult> tileSizes{rewriter.getIndexAttr(layout::kTileEdge),
                                      rewriter.getIndexAttr(K)};
  SmallVector<OpFoldResult> tileStrides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
  Value tile = memref::SubViewOp::create(rewriter, opLoc, bridge->source,
                                         tileOffsets, tileSizes, tileStrides);
  emitTileCompute(rewriter, opLoc, bias, scratch, op.getRhs(), op.getOuts(),
                  m, tile, c0, c1, shape.k, batch, cNt, decisionId);
  rewriter.setInsertionPointAfter(mLoop);
  memref::DeallocOp::create(rewriter, opLoc, scratch);

  // Retire the bridge, staged-path style. The packs' DPS results redirect to
  // the buffers they wrote (the carried loop's yield folds to its own iter
  // arg), then the matmul and the array's deallocations go, then the bridge
  // loop or the packs, and finally the array allocation underneath -- each
  // erased only once nothing uses it.
  for (Operation *packOp : bridge->ops)
    if (packOp->getNumResults() != 0)
      packOp->getResult(0)
          .replaceAllUsesWith(cast<PackActOp>(packOp).getDst());

  SmallVector<memref::DeallocOp> actDeallocs;
  for (Operation *user : act.getUsers())
    if (auto d = dyn_cast<memref::DeallocOp>(user))
      actDeallocs.push_back(d);

  rewriter.eraseOp(op);
  for (memref::DeallocOp d : actDeallocs)
    rewriter.eraseOp(d);
  if (bridge->loop) {
    rewriter.eraseOp(bridge->loop);
  } else {
    for (Operation *packOp : bridge->ops)
      rewriter.eraseOp(packOp);
  }

  SmallVector<memref::DeallocOp> bufferDeallocs;
  for (Operation *user : bridge->buffer.getUsers())
    if (auto d = dyn_cast<memref::DeallocOp>(user))
      bufferDeallocs.push_back(d);
  for (memref::DeallocOp d : bufferDeallocs)
    rewriter.eraseOp(d);
  if (arrayAlloc->use_empty())
    rewriter.eraseOp(arrayAlloc);

  folded = true;
  return success();
}

/// The staged tile loop. Returns false (and leaves the IR alone) when the
/// activation is not a bridge with a row-major source, when `auto` sees a K
/// extent below the staging floor (`kStageMinKTiles`), when the shape does not
/// fit the staging geometry, or when even the serial ring does not fit the VTCM
/// budget; the caller then emits the plain tile loop. Every decline names its
/// reason in a remark -- the pipeline is an optimisation, never a precondition,
/// and a silent fallback is an unobservable one.
///
/// The loop is the interface contract of
/// `docs/hmx/hmx-scheduling-interface-plan.md` §9.4 in its pipelined form: the
/// pass emits the serial source loop (issue -> await -> compute per tile) and,
/// at depth 2, hands it to `scf::pipelineForLoop` with the schedule "issue in
/// stage 0, await and compute in stage 1". The pipeliner generates the
/// prologue, the steady kernel and the peeled epilogue and versions the
/// cross-stage (token, slot) pair as iter_args; the depth-2 overlap -- tile
/// m+1's transfer in flight during tile m's compute -- is the pipeliner's
/// kernel `issue(m+1) await(m) compute(m)`, whose DMA/await/compute order is
/// identical to the hand-written ring it replaced. `depth` is 2 when the double
/// ring fits and `Mt > depth`, and 1 otherwise: the same source loop left
/// unpipelined (issue and await adjacent, one slot, no overlap).
///
/// `requestedDepth` is the pass's `pipeline-depth` knob: 0 lets the budget and
/// the tile count pick the deepest ring that fits, 1 forces the serial ring,
/// and 2 asks for the double ring (narrowed to the deepest ring that fits, with
/// a remark, when the budget cannot pay for it). The unstaged arm (3) never
/// reaches this emitter -- the caller handles it before asking for staging.
/// Whatever the knob says, the pass never exceeds the budget: an impossible
/// request becomes a remark plus the best depth that fits, never a silent
/// over-commit.
static LogicalResult emitStageLoop(IRRewriter &rewriter, Location opLoc,
                                   Value bias, MatmulOp op, func::FuncOp func,
                                   int64_t vtcmBudget, int64_t requestedDepth,
                                   int64_t batch, int64_t stagedReadoutMTiles,
                                   bool &staged) {
  staged = false;
  std::optional<int64_t> decisionId;
  if (failed(readDecisionId(op.getOperation(), decisionId)))
    return failure();
  Value act = op.getLhs();
  Value wt = op.getRhs();
  Value ar = op.getOuts();

  auto bridge = findActivationBridge(act);
  if (!bridge)
    return declineStageLoop(op, requestedDepth,
                            PipelineReason::NoRowMajorBridge);

  // The array is refilled by the tile loop, so the only other readers allowed
  // are the bridge's own packs, this matmul, and its deallocation. A second
  // reader would observe the array after the tile loop rewrote it.
  for (OpOperand &u : act.getUses()) {
    Operation *owner = u.getOwner();
    if (owner == op.getOperation() ||
        (isa<PackActOp>(owner) && cast<PackActOp>(owner).getDst() == act) ||
        isa<memref::DeallocOp, scf::YieldOp>(owner))
      continue;
    return declineStageLoop(op, requestedDepth,
                            PipelineReason::ExtraActivationReader);
  }

  auto actType = dyn_cast<MemRefType>(bridge->buffer.getType());
  auto wtType = dyn_cast<MemRefType>(wt.getType());
  auto srcType = dyn_cast<MemRefType>(bridge->src.getType());
  if (!actType || !wtType || !srcType || srcType.getRank() != 2 ||
      !srcType.hasStaticShape() ||
      !(dtype::isAdmittedFloat(srcType.getElementType())))
    return declineStageLoop(op, requestedDepth,
                            PipelineReason::InvalidStagingGeometry);

  int64_t Mt = actType.getDimSize(0), Kt = actType.getDimSize(1);
  int64_t Nt = weightNTiles(wtType);
  int64_t K = srcType.getDimSize(1);
  if (Mt < 1 || Kt < 1)
    return declineStageLoop(op, requestedDepth,
                            PipelineReason::EmptyStagingGrid);
  if (srcType.getDimSize(0) != Mt * layout::kTileEdge ||
      K != Kt * layout::kTileEdge || weightKTiles(wtType) != Kt)
    return declineStageLoop(op, requestedDepth,
                            PipelineReason::StagingGridMismatch);

  // `auto` declines a shallow-K shape on BOTH channels before it gives up:
  //
  //   transfer channel: below `kStageMinKTiles` one staged transfer is too
  //   small to hide the DMA engine's fixed cost behind the tile's compute
  //   (the threshold's mechanism and the device numbers are on the constant).
  //
  //   read-out channel (stagedReadoutMTiles, wired by the pipeline to
  //   2 x hmxReadoutBatch when the read-out split is enabled, and honoured
  //   only for a single-matmul function -- a multi-matmul function is
  //   declined whole by hmx-vector-readout, see the caller): the staged
  //   loop is also what the read-out split's m-tile loop attaches to, and
  //   its first batch cannot overlap anything (pipeline fill), so fewer
  //   than two batches -- Mt < 2 x G -- is pure handoff cost. The floor is
  //   twice the calibrated batch, not a new tuned constant: it is where
  //   overlap becomes possible at all.
  //   measurement: def-vs-def+depth2 sweep, one build, iters=1000
  //   (exp/hmx/gap_table/op_side/s23_sweep.py, 2026-10-04): Mt=32 (S1
  //   1024x512x64) 47 -> 37 us with the read-out fired; Mt=4 (S3
  //   128x128x128) 3 -> 7 us, the serial ring wins. The boundary itself
  //   (2 x G = 8) is positioned by the fill mechanism, not measured -- no
  //   shape between Mt=5 and Mt=31 has been run.
  //
  // An explicit `pipeline-depth=1/2` skips both floors -- those are the A/B
  // arms -- and `3` never reaches this emitter. A closed read-out channel
  // (stagedReadoutMTiles <= 0) declines exactly as before: the floor is a
  // second chance to stage, never a licence to stage everything.
  if (requestedDepth <= 0 && Kt < kStageMinKTiles &&
      (stagedReadoutMTiles <= 0 || Mt < stagedReadoutMTiles))
    return declineStageLoop(op, requestedDepth, PipelineReason::ShallowK, 0, 0, 0,
                           Kt);

  // The staged loop allocates, on top of the arrays the attribution already
  // counted: one crouton-row scratch (the pack destination, reused by every
  // tile) and `depth` staging slots (one 32 x K f16 tile each) plus `depth`
  // status words. The whole activation array is retired by this path, so its
  // bytes come back to the room -- not counting them would turn a ring that
  // fits into a spurious fallback.
  int64_t actBytes = actType.getNumElements() * 2;
  int64_t scratchBytes = Kt * layout::kCroutonBytes;
  int64_t srcElemBytes = srcType.getElementTypeBitWidth() / 8;
  int64_t slotBytes = layout::kTileEdge * K * srcElemBytes;
  // TRACEABILITY: statusBytes
  //   ⚠️ NOT GATE-VISIBLE, deliberately. `statusBytes` is a non-constexpr local
  //   and the depth cap is a bare `2` at three sites, so
  //   test_hmx_constant_traceability.py cannot see either. This block is
  //   documentation only. That is stated here so no reader assumes a machine
  //   check is standing behind it -- promoting these to named constants purely
  //   to satisfy the gate would be churn, not rigour.
  //   mechanism: one i32 completion token per ring slot, so the size is the
  //     token size and not a tunable. The depth cap of 2 is structural, not
  //     measured: the pipeliner issues one tile per stage, so a third slot
  //     would be a buffer nothing writes. `fits()` then derives the affordable
  //     depth from the remaining VTCM rather than from a constant.
  //   measurement: none required -- the size is the token size and the cap of 2
  //     is structural (the pipeliner issues one tile per stage, so a third slot
  //     is a buffer nothing writes). `fits()` derives the affordable depth from
  //     remaining VTCM rather than from a constant.
  //   shape set: n/a (not shape dependent).
  //   workload representativeness: n/a.
  //   The value that *is* measured, the K floor above, is a separate constant
  //     with its own traceability block.
  int64_t statusBytes = 4;
  int64_t ringBytes = slotBytes + statusBytes;
  int64_t room = hmx::vtcm::roomBytes(func, vtcmBudget, actBytes);

  // The deepest ring the budget can pay for, `scratch + depth * ring`: 2, or 1
  // when only the serial ring fits. 0 means not even that fits, so the plain
  // tile loop runs. The K floor above is about profitability, not footprint:
  // what a shape pays for here is the ring, not a second copy of the whole
  // activation.
  auto fits = [&](int64_t d) { return scratchBytes + d * ringBytes <= room; };
  int64_t budgetDepth = fits(2) ? 2 : (fits(1) ? 1 : 0);
  if (budgetDepth == 0)
    return declineStageLoop(op, requestedDepth, PipelineReason::VtcmBudget,
                            room, scratchBytes + ringBytes, budgetDepth);

  // The knob selects a depth, then the budget caps it. `requestedDepth` <= 0 is
  // auto; anything above 2 is clamped to the deepest ring that exists.
  int64_t depth =
      requestedDepth <= 0 ? budgetDepth : std::min<int64_t>(requestedDepth, 2);
  PipelineReason manifestReason = PipelineReason::None;
  int64_t manifestNeededBytes = 0;
  if (depth > budgetDepth) {
    manifestReason = PipelineReason::VtcmBudget;
    manifestNeededBytes = scratchBytes + budgetDepth * ringBytes;
    depth = budgetDepth;
  }
  // A ring at least as deep as the tile count has no steady state worth a
  // second slot -- the extra transfer would overlap nothing -- so geometry caps
  // it too. (The pipeliner's prologue could not stage past the end anyway: it
  // issues exactly one tile per pipeline stage, and the kernel stops that many
  // iterations short.)
  if (Mt <= depth) {
    LLVM_DEBUG(llvm::dbgs()
               << "hmx-partition: ring depth capped by tile count Mt=" << Mt
               << " (wanted " << depth << ")\n");
    if (depth == 2)
      manifestReason = PipelineReason::TileCount;
    depth = 1;
  }
  // Auto narrowing from the double ring to the serial one is still a staged
  // loop, but the overlap it was chosen for is gone.
  if (requestedDepth <= 0 && budgetDepth == 1 && Mt > 1) {
    manifestReason = PipelineReason::VtcmBudget;
    manifestNeededBytes = scratchBytes + 2 * ringBytes;
  }
  LLVM_DEBUG(llvm::dbgs() << "hmx-partition: activation staging depth " << depth
                          << " (requested " << requestedDepth
                          << ", budget depth " << budgetDepth << ", room "
                          << room << " B, scratch " << scratchBytes
                          << " B, ring/depth " << ringBytes << " B)\n");

  // The accumulator read-out, if it has the one shape this loop can absorb (see
  // `findRowUnpack`). Resolved HERE, after every `declineStageLoop` above and
  // before the first op this function creates, for two reasons that pull in
  // opposite directions and both matter:
  //
  //   * It relocates the pure chain that computes the read-out's destination
  //     (see `collectHoistableDefs`). On a shape that then declines for some
  //     other reason that move would be IR churn with no reader, so it must not
  //     happen before the declines.
  //   * It reads dominance with a `DominanceInfo` of its own, which would answer
  //     for the wrong block if built over the IR this function is midway through
  //     constructing.
  //
  // So: after the declines, before any creation.
  std::optional<RowUnpack> rowUnpack =
      findRowUnpack(rewriter, op, ar, Mt, op.getOperation());
  LLVM_DEBUG({
    if (rowUnpack)
      llvm::dbgs() << "hmx-partition: hoisting the accumulator read-out into "
                      "the tile loop (Mt="
                   << Mt << ")\n";
    else
      llvm::dbgs() << "hmx-partition: accumulator read-out not hoistable; the "
                      "separate read-out loop stays\n";
  });

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  Location loc = opLoc;

  auto slotType =
      MemRefType::get({layout::kTileEdge, K}, srcType.getElementType(),
                      AffineMap{}, hexagon::VTCM_ADDRESS_SPACE);
  auto statusType = MemRefType::get({1}, rewriter.getI32Type());
  // One crouton row: the pack destination, re-filled (not re-allocated) every
  // tile. It replaces the whole activation array, so the staged path pays one
  // row of croutons instead of `Mt` of them. The pack's DPS result is this same
  // buffer, so the mma consumes the packed croutons by address -- no copy
  // between pack and mma, and no second row in flight.
  auto scratchType = MemRefType::get(
      {1, Kt, layout::kCroutonPair, layout::kCroutonCol, layout::kCroutonHalf},
      dtype::croutonElementType(rewriter.getContext()), AffineMap{},
      hexagon::VTCM_ADDRESS_SPACE);

  // The static ring: `depth` (slot, status) pairs, one per in-flight tile, plus
  // the one scratch. All allocated once here and released once after the
  // epilogue; the loop body allocates nothing.
  Value scratch =
      memref::AllocOp::create(rewriter, loc, scratchType, ValueRange{});
  SmallVector<Value> slots, statuses;
  for (int64_t i = 0; i < depth; ++i) {
    slots.push_back(memref::AllocOp::create(rewriter, loc, slotType,
                                            ValueRange{},
                                            rewriter.getI64IntegerAttr(128)));
    statuses.push_back(memref::AllocOp::create(rewriter, loc, statusType,
                                               ValueRange{},
                                               rewriter.getI64IntegerAttr(4)));
  }

  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto cTileEdge =
      arith::ConstantIndexOp::create(rewriter, loc, layout::kTileEdge);
  auto cNt = arith::ConstantIndexOp::create(rewriter, loc, Nt);
  auto cMt = arith::ConstantIndexOp::create(rewriter, loc, Mt);

  // The serial source loop: iteration m issues tile m's transfer, awaits it and
  // computes it. Row is an element row offset, so tile m begins at source row
  // m * 32. On its own this is the depth-1 staged loop -- issue and await
  // adjacent, nothing in flight during the compute -- and at depth 1 it is
  // emitted as-is. At depth 2 it is the schedule input of the SCF pipeliner,
  // which splits it into "issue" (stage 0) and "await + compute" (stage 1) and
  // generates the prologue, the steady kernel and the peeled epilogue itself.
  //
  // The slot of tile m is slot(m mod 2) at depth 2, picked by a parity select
  // over the tile index. The select is a stage-0 op: the pipeliner re-evaluates
  // it with the shifted induction variable in the kernel's issue part (tile
  // m+1's transfer lands in slot((m+1) mod 2), never in the slot tile m is
  // being read from) and versions its result across stages -- the alternation
  // of the static slots is carried by the pipeliner's iter_args, not by
  // hand-written rotation. The status select needs no versioning: only the
  // stage op reads it, in the same stage.
  auto mLoop = scf::ForOp::create(rewriter, loc, c0, cMt, c1, ValueRange{});
  Block *mBody = mLoop.getBody();
  rewriter.setInsertionPointToStart(mBody);
  Value m = mLoop.getInductionVar();

  Value slotSel = slots[0];
  Value statusSel = statuses[0];
  if (depth == 2) {
    auto parity = arith::AndIOp::create(rewriter, loc, m, c1);
    auto odd = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                     parity, c0);
    slotSel = arith::SelectOp::create(rewriter, loc, odd, slots[1], slots[0])
                  .getResult();
    statusSel =
        arith::SelectOp::create(rewriter, loc, odd, statuses[1], statuses[0])
            .getResult();
  }

  // Issue -> await: the pack reads the slot the await handed back, so the
  // transfer -> compute dependency is an SSA edge, not an effect -- and the
  // token is `hmx.stage`'s result, which is exactly what the pipeliner versions
  // across the two stages.
  Value row = arith::MulIOp::create(rewriter, loc, m, cTileEdge);
  Value token = emitStage(rewriter, loc, bridge->src, row, slotSel, statusSel);
  Value ready = emitAwait(rewriter, loc, token, slotSel);
  emitTileCompute(rewriter, loc, bias, scratch, wt, ar, m, ready, c0, c1, Kt,
                  batch, cNt, decisionId);

  // The read-out rides at the tail of the iteration, after the inner N loop
  // whose `acc_read`s filled AR row `m` -- `emitTileCompute` leaves the
  // insertion point exactly there. The op is cloned, not rebuilt: the clone
  // keeps `count`, `valid_rows`/`valid_cols` and `hmx.decision_id` verbatim,
  // and only the row is remapped, so the attribution record this site carries
  // survives the move without a renumbering.
  if (rowUnpack) {
    IRMapping remap;
    remap.map(rowUnpack->op.getRow(), m);
    rewriter.clone(*rowUnpack->op.getOperation(), remap);
  }

  // The pipeliner owns the schedule from here: everything through the
  // `hmx.stage` is stage 0, the await and the compute are stage 1. With the
  // epilogue peeled it emits the prologue (the issue of tile 0), the kernel
  // `issue(m+1) await(m) compute(m)` over tiles 0 .. Mt-2, and an epilogue that
  // only awaits and computes tile Mt-1 -- the same DMA/await/compute order the
  // hand-written ring produced, with the (token, slot) rotation done by value
  // versioning. A static-bound loop is never predicated, so a failure here can
  // only happen before the pipeliner touches the IR: the source loop above is
  // intact and serially correct (issue -> await -> compute), so it stands as
  // the serial staged loop.
  if (depth == 2) {
    Operation *stageOp = token.getDefiningOp();
    scf::PipeliningOption options;
    options.peelEpilogue = true;
    options.getScheduleFn =
        [stageOp](scf::ForOp forOp,
                  std::vector<std::pair<Operation *, unsigned>> &schedule) {
          bool issue = true;
          for (Operation &bodyOp : forOp.getBody()->without_terminator()) {
            schedule.emplace_back(&bodyOp, issue ? 0u : 1u);
            if (&bodyOp == stageOp)
              issue = false;
          }
        };
    // The pipeliner emits the prologue and the kernel at the rewriter's
    // current insertion point, so it must be before the loop, not inside the
    // body this emitter just filled.
    rewriter.setInsertionPoint(mLoop);
    bool modifiedIR = false;
    if (failed(scf::pipelineForLoop(rewriter, mLoop, options, &modifiedIR))) {
      assert(!modifiedIR && "pipelining failed after rewriting the IR");
      // The source loop is intact: keep it as the serial staged loop, with
      // everything after it following the loop as usual.
      rewriter.setInsertionPointAfter(mLoop);
      manifestReason = PipelineReason::PipelinerFailed;
    }
  } else {
    // Depth 1 is the source loop as emitted: move the insertion point back out
    // of the body the compute left it in, so what follows follows the loop.
    rewriter.setInsertionPointAfter(mLoop);
  }

  // The separate read-out loop has done its work: every row it used to walk is
  // unpacked by the iteration that wrote it. Retire it here, where the
  // insertion point already sits after the last tile loop -- so nothing this
  // loop owned (its constants, its result) is still live at the erasure.
  if (rowUnpack)
    rewriter.eraseOp(rowUnpack->loop);

  PipelineDecision decision;
  decision.requestedDepth = requestedDepth;
  // Reaching this point means the staged source loop was emitted. A failed
  // SCF pipeliner leaves that loop serial, but it is still staged: the manifest
  // reason, not `pipeline_selected`, records the lost overlap.
  decision.staged = true;
  decision.depth = depth;
  decision.reason = manifestReason;
  decision.neededBytes = manifestNeededBytes;
  decision.freeBytes = room;
  decision.budgetDepth = budgetDepth;
  if (failed(recordPipelineDecision(op, decision)))
    return failure();
  staged = true;

  // The ring and the scratch live exactly as long as the tile loop. The
  // insertion point is after the pipelined epilogue (or after the serial loop
  // at depth 1), so neither is freed before its last reader.
  for (Value slot : slots)
    memref::DeallocOp::create(rewriter, loc, slot);
  for (Value status : statuses)
    memref::DeallocOp::create(rewriter, loc, status);
  memref::DeallocOp::create(rewriter, loc, scratch);

  // Retire the bridge, the matmul and the activation array. The array is dead
  // in this path: the tile loop writes the scratch instead, so neither the
  // bridge loop that filled it nor the array itself survives. Collect the
  // array's deallocations before its bridge loop is erased -- in the carried
  // form the loop is what defines the array -- then drop the bridge and the
  // matmul, then the deallocations, and finally the allocation underneath. The
  // earlier reader check guarantees nothing else observes the array.
  SmallVector<memref::DeallocOp> actDeallocs;
  for (Operation *user : act.getUsers())
    if (auto d = dyn_cast<memref::DeallocOp>(user))
      actDeallocs.push_back(d);

  rewriter.eraseOp(op);
  if (bridge->loop) {
    rewriter.eraseOp(bridge->loop);
  } else {
    for (PackActOp p : bridge->packs)
      rewriter.eraseOp(p);
  }
  for (memref::DeallocOp d : actDeallocs)
    rewriter.eraseOp(d);

  // The allocation is `act` itself in the canonicalized form and the bridge
  // loop's init arg in the carried one; either way it is unreferenced now. Drop
  // its deallocation (if the array value was not the thing deallocated) and
  // then the allocation, so the activation stops occupying VTCM.
  if (auto alloc = bridge->buffer.getDefiningOp<memref::AllocOp>()) {
    SmallVector<memref::DeallocOp> bufferDeallocs;
    for (Operation *user : bridge->buffer.getUsers())
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        bufferDeallocs.push_back(d);
    for (memref::DeallocOp d : bufferDeallocs)
      rewriter.eraseOp(d);
    if (alloc->use_empty())
      rewriter.eraseOp(alloc);
  }
  return success();
}

struct HmxPartitionPass
    : public mlir::hmx::impl::HmxPartitionBase<HmxPartitionPass> {
  using HmxPartitionBase::HmxPartitionBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HmxDialect, arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    // The manifest and its bridge-site totals are module state, while this
    // interface pass may run concurrently for sibling functions.
    std::lock_guard<std::mutex> manifestGuard(hmxModuleStateMutex());

    func::FuncOp func = cast<func::FuncOp>(getOperation());
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module) {
      func.emitError("hmx-partition requires a builtin.module parent");
      return signalPassFailure();
    }
    // A standalone hmx-partition invocation may lower hand-written hmx.matmul
    // IR that has not gone through matmul-to-hmx.  Such IR has no attribution
    // record to update; production/full-pipeline IR always carries the module
    // manifest created by matmul-to-hmx.
    const bool hasManifest = module->hasAttr(kHmxManifestAttr);
    if (hasManifest && (failed(ensureHmxManifest(module)) ||
                        failed(restoreHmxManifestDecisionIds(module, func)) ||
                        failed(refreshHmxManifestBridgeCounts(module))))
      return signalPassFailure();

    SmallVector<MatmulOp> matmuls;
    func.walk([&](MatmulOp op) { matmuls.push_back(op); });
    if (matmuls.empty())
      return;

    // The read-out channel of `auto` staging pays only when the tile loop it
    // stages executes ONCE per launch. The staged ring's fixed cost -- the
    // peeled prologue and epilogue, the scratch -- is per EXECUTION of the
    // tile loop, and a matmul sitting inside an outer loop pays it per
    // iteration of that loop. Measured both ways on shallow-K shapes (Kt=2,
    // iters=1000, this build, 2026-10-04): a once-per-launch matmul is the
    // channel's win (S1 61 -> 38 us), while flash attention's per-chunk Q@K
    // and P@V -- the same shape, executed 16 times per launch -- pay the
    // ring's fixed cost twice per chunk (staged ring + read-out 5466 us vs
    // unstaged serial 5230 us; the read-out itself is neutral there, 5466 vs
    // 5555 with the ring held fixed). The channel therefore opens only for a
    // matmul that no outer loop re-executes; the single-matmul gate this
    // replaces was a proxy for exactly this condition, and removing it
    // without a replacement cost flash attention +5%.
    auto readoutChannelFor = [&](MatmulOp op) -> int64_t {
      if (this->stagedReadoutMTiles <= 0)
        return 0;
      for (Operation *ancestor = op->getParentOp(); ancestor != nullptr;
           ancestor = ancestor->getParentOp())
        if (isa<scf::ForOp, scf::ForallOp>(ancestor))
          return 0;
      return this->stagedReadoutMTiles;
    };

    // The engine's budget, with the one field a caller may narrow: 0 means the
    // device default (see HmxTarget).
    const int64_t vtcmBudget = this->vtcmBudgetBytes > 0
                                   ? this->vtcmBudgetBytes
                                   : HmxTarget::defaultVtcmBudget;

    // The K batch per `hmx.mma`, resolved the same way: 0 is the hardware
    // maximum, so "not passed", "passed 0" and "passed 32" are one request.
    //
    // Rejected rather than clamped, and this is the only knob here that is.
    // `pipeline-depth` clamps because anything past its widest ring means "as
    // deep as you can", so the clamp agrees with the request. Here 32 is a
    // hardware bound (`Rt[dC]` is five bits) and a value above it is not a
    // deeper request but an unencodable one: clamping 64 to 32 would report
    // success for a batch the caller never asked for, which is the failure mode
    // `AGENTS.md` rule 8 exists to prevent -- a knob that looks effective and is
    // not. A negative value has no meaning either, so it is in the same class.
    //
    // Checked here, after the empty-matmul return above, because this is an
    // interface pass: a function the pass does not touch must not fail a build
    // over a configuration mistake that cannot affect its output.
    if (this->croutonsPerMma < 0 ||
        this->croutonsPerMma > kMaxCroutonsPerMma) {
      // `.getValue()` not the implicit conversion: streaming `Pass::Option`
      // directly prints nothing, which would ship a diagnostic that names the
      // domain but not the offending value.
      func.emitError(
          "hmx-partition croutons-per-mma must be 0 (the hardware maximum, "
          "32) or in [1, 32], but got ")
          << this->croutonsPerMma.getValue();
      return signalPassFailure();
    }
    const int64_t batch = resolveCroutonsPerMma(this->croutonsPerMma);

    Location loc = func.getLoc();
    IRRewriter rewriter(func.getContext());
    rewriter.setInsertionPointToStart(&func.getBody().front());

    // The identity conversion state is kernel-level setup, so it is created
    // once for the whole function. It is an ordinary space-1 block: the VTCM
    // machinery below places it next to the crouton arrays.
    auto biasType = MemRefType::get({layout::kConvStateBytes}, rewriter.getI8Type(),
                                    AffineMap{}, hexagon::VTCM_ADDRESS_SPACE);
    auto bias = memref::AllocOp::create(
        rewriter, loc, biasType, ValueRange{},
        rewriter.getI64IntegerAttr(layout::kConvStateBytes));
    BiasInitOp::create(rewriter, loc, bias);

    for (MatmulOp op : matmuls) {
      auto shape = getTileShape(op);
      if (!shape) {
        op.emitError("hmx.matmul must operate on 5D crouton arrays here");
        return signalPassFailure();
      }
      if (op->hasAttr(kHmxDiagnosticTailAttr) &&
          !isHmxDiagnosticTailMarker(op.getOperation())) {
        op.emitError("hmx.diagnostic_tail_partition must be a unit attribute");
        return signalPassFailure();
      }
      if (auto tailPlan = op.getTailPlanAttr()) {
        if (!isHmxDiagnosticTailMarker(op.getOperation())) {
          op.emitError("tail_plan is not yet supported by hmx-partition");
          return signalPassFailure();
        }
        if (failed(markHmxDiagnosticTailModule(module)))
          return signalPassFailure();
        TailGrid grid;
        if (failed(readTailGrid(op, *shape, tailPlan, grid)))
          return signalPassFailure();
        std::optional<int64_t> decisionId;
        if (failed(readDecisionId(op.getOperation(), decisionId)))
          return signalPassFailure();
        auto unpackBridge = findUnpackBridge(op);
        if (!unpackBridge) {
          op.emitError("diagnostic tail partition requires a direct unpack "
                       "bridge");
          return signalPassFailure();
        }
        SmallVector<Operation *> retired;
        Operation *cursor = op.getOperation();
        if (failed(emitDiagnosticInputBridges(rewriter, op.getLoc(), op, grid,
                                              decisionId, cursor, retired)))
          return signalPassFailure();
        cursor = emitPeeledEdgeTileLoop(rewriter, op.getLoc(), bias, op, grid,
                                       batch);
        if (failed(emitDiagnosticOutputBridge(rewriter, op.getLoc(), op, grid,
                                              *unpackBridge, decisionId, cursor,
                                              retired)))
          return signalPassFailure();

        // The rebuilt memref DPS ops write the same buffers as the direct
        // bridges they replace. Redirect any old result uses before erasing the
        // old sites; a tensor-shaped result would not be the memref contract
        // this diagnostic slice accepts.
        for (Operation *old : retired) {
          if (isa<scf::ForOp>(old) || old->getNumResults() == 0)
            continue;
          Value replacement;
          if (isa<PackActOp, PackWeightOp>(old))
            replacement = isa<PackActOp>(old) ? op.getLhs() : op.getRhs();
          else
            replacement = unpackBridge->destination;
          if (old->getResult(0).getType() != replacement.getType()) {
            op.emitError("diagnostic tail bridge result is not a memref DPS "
                         "value");
            return signalPassFailure();
          }
          old->getResult(0).replaceAllUsesWith(replacement);
        }
        for (Operation *old : retired)
          rewriter.eraseOp(old);

        PipelineDecision decision;
        decision.requestedDepth = this->pipelineDepth;
        decision.reason = PipelineReason::TailPeeledEdge;
        if (failed(recordPipelineDecision(op, decision)))
          return signalPassFailure();
        rewriter.eraseOp(op);
        continue;
      }
      // Single-op grid self-consistency: each dot keeps its own grid -- grids
      // are never unified across dots. A mismatch is a loud failure, never a
      // silently wrong loop nest. (The op verifier checks the same contract on
      // construction; this is the lowering half, guarding whatever reaches
      // this pass.)
      auto lhs = cast<MemRefType>(op.getLhs().getType());
      auto rhs = cast<MemRefType>(op.getRhs().getType());
      auto out = cast<MemRefType>(op.getOuts().getType());
      if (lhs.getDimSize(1) != rhs.getDimSize(1) ||
          out.getDimSize(0) != lhs.getDimSize(0) ||
          out.getDimSize(1) != rhs.getDimSize(0)) {
        op.emitError("hmx.matmul tile grids disagree: lhs [")
            << lhs.getDimSize(0) << ", " << lhs.getDimSize(1) << "] rhs ["
            << rhs.getDimSize(0) << ", " << rhs.getDimSize(1) << "] out ["
            << out.getDimSize(0) << ", " << out.getDimSize(1) << "]";
        return signalPassFailure();
      }

      // The matmul itself is replaced by the loop that walks its crouton
      // arrays.
      Location opLoc = op.getLoc();
      // A `#hmx.tail_plan` is not a production request in this pass. Only the
      // explicit `hmx.diagnostic_tail_partition` unit marker, preserved from
      // the diagnostic producer (or set on a direct test bridge), reaches the
      // diagnostic full + peeled serial emitter; every ordinary tail plan still
      // fails closed.
      // The diagnostic emitter rebuilds pack/unpack sites with explicit valid
      // extents, but it does not claim the looped bridge, VTCM-accounting, or
      // launcher gates.
      //
      // `pipeline-depth=3` is the unstaged arm: skip the staging rewrite
      // outright and emit the plain tile loop, which leaves the activation
      // bridge and its array in place. It is chosen here rather than by asking
      // the staged emitter to decline, so a bridge shape is preserved --
      // letting the staged loop run and then undoing it would already have
      // retired `act`.
      if (this->pipelineDepth == kSerialPipelineDepth) {
        PipelineDecision serial;
        serial.requestedDepth = this->pipelineDepth;
        serial.reason = PipelineReason::SerialRequested;
        if (failed(recordPipelineDecision(op, serial)))
          return signalPassFailure();
      } else {
        bool staged = false;
        if (failed(emitStageLoop(rewriter, opLoc, bias, op, func, vtcmBudget,
                                 this->pipelineDepth, batch,
                                 readoutChannelFor(op), staged)))
          return signalPassFailure();
        if (staged)
          continue;
      }
      // The serial tile loop. First try to fold the whole-array activation
      // bridge into the m-tile loop (per-tile pack from the bridge's source,
      // one crouton-row scratch, the bridge retired): that is the serial
      // source-ring form, and it is what gives the unstaged shape a per-tile
      // producer stream. A matmul the fold declines -- no canonical bridge, a
      // reader it does not own, an outer loop re-executing the matmul -- keeps
      // today's plain tile loop, reading the whole array where it is.
      bool folded = false;
      if (failed(tryFoldSerialActivationBridge(rewriter, opLoc, bias, op,
                                               *shape, batch, func, vtcmBudget,
                                               folded)))
        return signalPassFailure();
      if (!folded) {
        emitSerialTileLoop(rewriter, opLoc, bias, op, *shape, batch);
        rewriter.eraseOp(op);
      }
    }

    // The conversion-state block is kernel-level setup: it is filled once at
    // entry and read by every tile's read-out, so it is released on the way
    // out, after the last tile loop that reads it -- every exit gets its own
    // deallocation, and the entry block defines the buffer so it dominates all
    // of them. The tile loops were inserted before their `hmx.matmul`, which
    // the erase above removed, so no reader follows a return. Without this
    // deallocation the 256 B state leaks on every launch -- it is the one VTCM
    // allocation this pass makes with no partner -- and the runtime's pool
    // never gets it back.
    SmallVector<func::ReturnOp> returns;
    func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (func::ReturnOp ret : returns) {
      rewriter.setInsertionPoint(ret);
      memref::DeallocOp::create(rewriter, loc, bias);
    }

    // Weight residency and this pass may both have retired bridge writers. The
    // final count is therefore a walk of the post-partition IR, never the count
    // observed when attribution began.
    if (hasManifest && failed(refreshHmxManifestBridgeCounts(module)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hmx::createHmxPartitionPass(
    const mlir::hmx::HmxPartitionOptions &options) {
  return std::make_unique<HmxPartitionPass>(options);
}
