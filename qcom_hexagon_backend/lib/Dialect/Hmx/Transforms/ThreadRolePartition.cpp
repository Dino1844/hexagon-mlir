//===- ThreadRolePartition.cpp - Decide which thread each region belongs on ---===//
//
// The runtime has two engines and one lock. Today a kernel's HMX work and its
// HVX work run on whichever worker thread the grid scheduler picked, in program
// order, so the two engines take turns. This pass makes the *role* a
// compile-time fact instead: it decides, per region and per kernel, whether a
// split is even possible, and records that verdict in the manifest.
//
// It records only. Nothing is outlined, no attribute is attached to guide a
// later lowering, no SPSC ring is built, and no thread is created -- see "WHAT
// THIS PASS DOES NOT DO" in the Passes.td description. That is deliberate: the
// decision is the part that can be wrong, and a wrong decision that is only
// recorded costs nothing. A later stage consumes the manifest field this pass
// writes.
//
// The decision that matters is not "does this kernel touch both engines" --
// almost every one does. It is whether the two are *interleaved closely enough
// to hand work across*: a pack for tile i+1 has to be produced while tile i
// computes. That is only observable once hmx-partition has built the tile loop
// and the staging ring, which is why this pass runs after hmx-partition (see
// LinalgToLLVMPass.cpp). Before it, every kernel looks identical: one hmx.matmul
// with two independent pack loops in front of it.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxReadoutHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxRoleHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxTarget.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"

#include <mutex>

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_THREADROLEPARTITION
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

/// Does this op have to run where the HMX resource is held?
///
/// Read off the positive `HmxEngineIns` marker, never off "is this an hmx op
/// that lacks some other marker". The two predicates want opposite defaults and
/// one bit cannot serve both:
///
///   ensure/unlock wants unmarked == engine, because forgetting a marker there
///   costs one harmless lock round-trip while excluding a real engine op parks
///   the next thread in HAP_compute_res_hmx_lock forever.
///
///   thread role wants unmarked == vector, because sending an unmarked LAYOUT op
///   to the HMX thread parks it on a thread that holds no VTCM, with no
///   diagnostic.
///
/// `HmxEngineIns` is written for the second. `HmxLayoutHvx` and `HmxDmaOnly` are
/// the negative exemptions the first one uses, and neither is consulted here.
bool mustRunOnEngineThread(Operation *op) {
  return op->hasTrait<OpTrait::HmxEngineIns>();
}

/// Is this op layout work that *produces* engine input?
///
/// The producer/consumer split is the whole verdict, and `HmxLayoutHvx` does
/// not make it: that trait covers `pack_act`, `pack_weight`, `unpack_acc` and
/// `unpack_acc_f32` alike. A pack runs before the engine op it feeds and can be
/// hoisted onto the other thread ahead of time; an unpack consumes what the
/// engine op just wrote and cannot. So the predicate is the concrete op, not
/// the trait.
bool isPackWork(Operation *op) {
  return isa<PackActOp, PackWeightOp>(op);
}

bool isUnpackWork(Operation *op) {
  return isa<UnpackAccOp, UnpackAccF32Op>(op);
}

/// One region's verdict. "Region" here is the unit a split would cut along: a
/// block with an owner -- a loop body, or the function body. Walking ops instead
/// would classify per-op, and a tile loop that both packs and issues HMX is
/// precisely the case the kernel-level verdict has to recognise.
struct RegionVerdict {
  bool engine = false;
  bool pack = false;
  bool unpack = false;
};

/// What the kernel as a whole looks like, and the three co-location facts the
/// verdict is decided from. The co-location flags are what distinguish the
/// verdicts: a pack in its own loop that ran to completion long before the
/// engine work is not the same thing as a pack in the same loop as an HMX
/// instruction, and only the second is a stream to overlap.
struct KernelVerdict {
  RegionVerdict total;
  bool packBesideEngine = false;
  bool unpackBesideEngine = false;
  int64_t regions = 0;
};

bool isRegionRoot(Operation *op) {
  return isa<func::FuncOp, scf::ForOp, scf::IfOp, scf::WhileOp>(op);
}

/// Whether any engine op lives anywhere inside `root`'s subtree.
///
/// The beside-flags' same-iteration half (see `classifyRegions`): engine ops
/// nested inside a loop that also holds a pack directly are in that pack's
/// iteration stream, not in a disjoint region of the kernel. Only loops ask
/// this question -- the function body has no iterations, and its subtree is
/// the whole kernel.
bool holdsEngineWork(Operation *root) {
  bool engine = false;
  root->walk([&](Operation *op) { engine |= mustRunOnEngineThread(op); });
  return engine;
}

/// Attribute every op to the region root that most tightly encloses it, then
/// reduce the regions to kernel-level facts.
///
/// The nearest-enclosing-root rule is the attribution half, and it is what
/// keeps a pack in its own loop in its own loop. Taking each region's full
/// subtree instead -- which is the obvious reading of "what is in this loop"
/// -- makes the function body contain every pack and every engine op in the
/// kernel, so `packBesideEngine` is true for all HMX kernels and every mixed
/// kernel comes out `role-split-ok`. That was measured: the verdict did not
/// move when pipeline-depth went 1 -> 2, the very switch that decides whether
/// a pack sits inside the tile loop. Under the nearest-root rule the body
/// holds only the ops written directly in it.
///
/// The beside-flags need a second half, added 2026-10-08 for S2.5: a
/// same-iteration test. No emitter puts a pack and an engine op in the SAME
/// region -- hmx-partition's emitters always nest the engine ops one loop
/// deeper (the (m, n) tile loops hold acc_clear/mma/acc_read while the pack
/// sits directly in the m-tile loop around them, in the staged ring today and
/// in any per-tile serial form S2.5 builds), so nearest-root alone makes
/// `packBesideEngine` false for every mixed kernel and `role-split-ok`
/// unreachable on production IR -- including the staged s2_anchor form, the
/// one shape ROADMAP1001 section 3.1 names as the canonical split. The
/// producer-stream question is not "same region" but "same iteration": a pack
/// held directly by a LOOP whose body carries engine work executes once per
/// iteration alongside that work, which is exactly the object
/// "pack_act(i+1) || mma(i)" needs. So a pack- or unpack-holding region whose
/// root is a loop also counts as beside engine when engine work lives
/// anywhere in that loop's body. Two boundaries keep this from collapsing
/// back into the subtree bug: the function body never asks (it has no
/// iterations, and its subtree is the whole kernel), and the pack's own loop
/// must be its nearest root -- a pack loop nested inside a bigger loop that
/// also holds the engine work (the per-chunk attention bridge, the serial
/// whole-array bridge) still runs to completion within each outer iteration
/// and stays `role-split-nopack`.
void classifyRegions(func::FuncOp fn, KernelVerdict &out) {
  llvm::DenseMap<Operation *, RegionVerdict> byRoot;
  llvm::SmallVector<Operation *> roots;
  fn.walk([&](Operation *op) {
    bool engine = mustRunOnEngineThread(op);
    bool pack = isPackWork(op);
    bool unpack = isUnpackWork(op);
    if (!engine && !pack && !unpack)
      return;
    Operation *root = op;
    while (Operation *parent = root->getParentOp()) {
      if (isRegionRoot(parent)) {
        root = parent;
        break;
      }
      root = parent;
    }
    if (!byRoot.count(root)) {
      roots.push_back(root);
      byRoot[root] = RegionVerdict();
    }
    RegionVerdict &region = byRoot[root];
    region.engine |= engine;
    region.pack |= pack;
    region.unpack |= unpack;
  });

  out.regions = roots.size();
  for (Operation *root : roots) {
    const RegionVerdict &region = byRoot[root];
    out.total.engine |= region.engine;
    out.total.pack |= region.pack;
    out.total.unpack |= region.unpack;
    if (region.engine && region.pack)
      out.packBesideEngine = true;
    if (region.engine && region.unpack)
      out.unpackBesideEngine = true;
    // The same-iteration half: a pack or unpack held DIRECTLY by a loop whose
    // body carries engine work anywhere beneath it. Only loops ask, and only
    // the nearest root counts -- both boundaries are documented above.
    if (!region.engine && (region.pack || region.unpack) &&
        isa<scf::ForOp, scf::WhileOp>(root) && holdsEngineWork(root)) {
      if (region.pack)
        out.packBesideEngine = true;
      if (region.unpack)
        out.unpackBesideEngine = true;
    }
  }
}

constexpr StringLiteral kSingleRoleEngine = "topology-single-role-hmx";
constexpr StringLiteral kSingleRoleVector = "topology-single-role-hvx";
constexpr StringLiteral kSplitOk = "role-split-ok";
constexpr StringLiteral kMixedIrreducible = "role-mixed-irreducible";
constexpr StringLiteral kSplitNoPack = "role-split-nopack";

StringRef decideTopology(const KernelVerdict &v) {
  if (!v.total.engine)
    return kSingleRoleVector; // layout only, or nothing at all: HVX either way
  if (!v.total.pack && !v.total.unpack)
    return kSingleRoleEngine; // engine only, and an HMX thread is required
  // Mixed. Ordered by how much there is to hand across:
  //
  //  1. A pack runs inside the same loop as the engine work -- directly in
  //     the loop, with the engine ops possibly one loop deeper (every emitter
  //     nests them there; see `classifyRegions`). That is a per-tile producer
  //     stream: while tile i is in the matrix engine, the other thread can be
  //     building tile i+1. This is the only shape where a thread split pays.
  //  2. Layout work is co-located but is only unpack. It consumes what the
  //     engine op just wrote, so it cannot start early; there is no producer
  //     stream, only a dependency.
  //  3. Layout work exists but in disjoint regions. The pack loop ran to
  //     completion before the engine work began, so there is no per-tile
  //     granularity to stream at all.
  if (v.packBesideEngine)
    return kSplitOk;
  if (v.unpackBesideEngine)
    return kMixedIrreducible;
  return kSplitNoPack;
}

//===----------------------------------------------------------------------===//
// THE SPLIT (emission; ROADMAP1001 sections 2, 3.2, 4.1/4.3)
//===----------------------------------------------------------------------===//
//
// Everything above this banner decides and records. Everything below it is
// what a `role-split-ok` verdict turns into when the pass runs (the pass is
// only scheduled when enableThreadRolePartition is on, so the OFF arm never
// reaches any of it). The producer keeps the tile loop's pack (and the
// staged arm's stage/await: the DMA is engine-independent), the engine
// n-loop moves wholesale into a private work function, and the two sides
// meet through the role executor's ring: the producer submits one
// descriptor per m-tile, the bound thread runs the work function once per
// descriptor, and the kernel drains before it returns.
//
// THE SHAPE, AND WHY THE SCRATCH BECOMES THE WHOLE ARRAY AGAIN
// -----------------------------------------------------------
// hmx-partition's emitters give the split its input: the serial source ring
// (the folded serial loop, or the staged loop at depth 1), whose m-tile
// iteration is
//
//     [stage(m), await(m)]           staged arm only; producer-side DMA
//     hmx.pack_act(scratch, ..., 0, 0, count=Kt)   the producer work
//     for n in 0..Nt:                the engine half -- THIS moves out
//       hmx.acc_clear; hmx.mma(scratch, wt, 0, n, ...); hmx.acc_read(...)
//     [hmx.unpack_acc(...)]          staged arm's hoisted read-out, if any
//
// The one-row scratch is the single-thread assumption: the pack of tile m+1
// may overwrite it only because the engine work of tile m already ran, in
// program order, inside the same iteration. Split the engine half onto
// another thread and that assumption is gone: the pack of m+1 would race
// the engine reads of m in the same row.
//
// The first form removes the assumption structurally rather than policing
// it: the scratch becomes the full [Mt, Kt] crouton array -- the same shape
// the whole-array activation bridge used before hmx-partition retired it --
// and every tile packs into and computes from its OWN row (hmx.mma addresses
// the whole array by row index natively; hmx.pack_act does NOT -- its row
// operand names the SOURCE block as well, so the pack reaches its row
// through a one-row view of the array instead. See the emission note at the
// pack: moving the row operand would move the source with it). The ring
// depth is then the tile count itself, derived from the tile-ring geometry
// exactly as HmxSpscRing.h demands ("the compiler side derives it from the
// tile-ring geometry of the kernel"): a ring Mt deep never fills (at most
// Mt descriptors exist), so the producer never waits, never reuses a row,
// and the only barrier is the exit drain.
//
// THE VTCM QUESTION (ROADMAP1001 section 5.1.7, verified here)
// -----------------------------------------------------------
// The ring's own storage is `capacity x 24` bytes of DDR, malloc'd by the
// executor (HmxRoleExecutor.cpp bind) -- it costs no VTCM. What costs VTCM
// is the rotating scratch, and the check below is the honest one: the
// single-row scratch retires (its bytes come back), the [Mt, Kt] rows move
// in, and the split is declined as `role-split-nobudget` when the rest of
// the kernel's commitments leave no room. The STAGING ring's budgetDepth is
// decided inside hmx-partition, which runs BEFORE this pass, so this split
// cannot retroactively lower it: the manifest's budget_depth field and the
// fits() decision it records describe the staging ring and stay untouched.
// (With the pipeline forcing the serial source ring while this pass is
// enabled -- see LinalgToLLVMPass.cpp -- the staging ring is the depth-1
// serial ring on every dual-role kernel anyway.)
//
// THE TWO SIDES OF THE DATA EDGE (ROADMAP1001 section 3.3 item 3)
// ---------------------------------------------------------------
// The ring carries the WORK ITEMS (descriptors naming tile rows); the
// buffers the work closes over (rows, wt, bias, ar) are explicit function
// parameters of the outlined work function -- the same split of labour the
// read-out handoff uses, and the reason its record carries memref types.
// A tile row is owned by the producer until its descriptor is published and
// by the consumer until it retires (HmxSpscRing.h's ownership model); with
// one row per tile the two never collide. `ar` rows are written by the
// consumer's acc_read and read by the producer's read-out, so any read-out
// that sat INSIDE the m-loop (the staged arm's hoisted unpack) moves after
// the exit drain -- the only producer-side point where every tile's engine
// work is provably done.

/// TRACEABILITY: kRoleSubmitBatch
///   mechanism: how many descriptors one submit call carries. The handoff
///     cost this amortizes is the WAKE, ~900 ns end to end on this part
///     (exp/hmx/s2_handoff/probe2.cpp, quoted in
///     bin/runtime/include/HmxVectorExecutor.h:12-13): one qurt_futex_wake
///     per submit call covers the whole batch, so the per-tile handoff cost
///     falls as 900/batch ns. The read-out split amortizes the same cost by
///     batching G ROWS per descriptor (its other half); this is the
///     descriptor-count half of the same lesson.
///   measurement: not re-measured for the engine section. 4 is the read-out
///     batch's calibrated default (hmx-readout-batch, measured 1.39x at
///     G=4); the engine section's work per tile is far above the read-out's
///     760 ns, so the amortization is more comfortable here, not less. The
///     S3 device window owns the real number; changing it here is a
///     one-constant edit, not a redesign.
///   shape set: n/a (a calling-cost amortization, not a shape threshold).
///   workload representativeness: the S2-class anchor (256x64x2048, Mt=8)
///     submits 2 batches of 4.
constexpr int64_t kRoleSubmitBatch = 4;

/// One candidate m-tile loop, with everything the split needs from it.
struct RoleSplitMatch {
  scf::ForOp mLoop;
  PackActOp pack;         ///< the producer work, directly in the m-loop body
  scf::ForOp nLoop;       ///< the engine half, directly in the m-loop body
  Value scratch;          ///< the pack's destination (one crouton row)
  memref::AllocOp scratchAlloc;
  Value wt;               ///< shared buffers the engine half closes over
  Value bias;
  Value ar;
  SmallVector<Operation *> unpacks; ///< in-loop read-outs to move past the drain
  SmallVector<Operation *> biasInits; ///< function-level engine preamble to move
  int64_t mt = 0;         ///< tile count: the ring depth and the scratch rows
};

/// Is `op` one of the m-loop body's direct children?
static bool directlyIn(Operation *op, scf::ForOp loop) {
  return op->getParentOp() == loop.getOperation();
}

/// The one m-tile loop this pass knows how to split, or nothing.
///
/// EVERY requirement is a decline and never a repair (the read-out pass's
/// rule, for the same reason: a half-split kernel is the worst outcome
/// available). The shape matched is exactly what hmx-partition's emitters
/// produce for the serial source ring -- the folded serial loop and the
/// depth-1 staged loop -- plus the structural facts the emission relies on.
/// Anything else keeps today's single-thread form with a remark naming the
/// reason.
static std::optional<RoleSplitMatch> matchSplitLoop(func::FuncOp fn) {
  llvm::SmallVector<scf::ForOp> candidates;
  fn.walk([&](scf::ForOp loop) {
    // The pack and the engine n-loop are DIRECT children: the producer
    // stream is "pack beside engine in the same iteration", which the
    // classification already established -- this re-checks it structurally.
    bool pack = false, engine = false;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (isa<PackActOp>(op))
        pack = true;
      if (auto inner = dyn_cast<scf::ForOp>(op)) {
        bool holds = false;
        inner->walk([&](Operation *o) { holds |= mustRunOnEngineThread(o); });
        engine |= holds;
      }
    }
    if (pack && engine)
      candidates.push_back(loop);
  });
  if (candidates.empty()) {
    fn.emitRemark("thread-role split not applied: no tile loop holds a pack "
                  "beside engine work (the serial source ring is what the "
                  "split transforms)");
    return std::nullopt;
  }
  if (candidates.size() > 1) {
    fn.emitRemark("thread-role split not applied: ")
        << candidates.size()
        << " tile loops hold a pack beside engine work; the first form "
           "transforms exactly one serial source ring per kernel";
    return std::nullopt;
  }

  RoleSplitMatch m;
  m.mLoop = candidates.front();

  // Static 0..Mt trip: the descriptor math (rowStart = m, slot = m) and the
  // ring depth (= Mt) are derived from the induction variable counting
  // tiles from zero.
  auto lb = m.mLoop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub = m.mLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = m.mLoop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb || !ub || !step || lb.value() != 0 || step.value() != 1) {
    fn.emitRemark("thread-role split not applied: the tile loop is not the "
                  "static 0..Mt form the emitters produce");
    return std::nullopt;
  }
  m.mt = ub.value();

  // One pack, one engine n-loop, directly in the body.
  for (Operation &op : m.mLoop.getBody()->without_terminator()) {
    if (isa<PackActOp>(op)) {
      if (m.pack) {
        fn.emitRemark("thread-role split not applied: more than one pack in "
                      "the tile loop body");
        return std::nullopt;
      }
      m.pack = cast<PackActOp>(op);
    }
    if (auto inner = dyn_cast<scf::ForOp>(op)) {
      bool holds = false;
      inner->walk([&](Operation *o) { holds |= mustRunOnEngineThread(o); });
      if (holds) {
        if (m.nLoop) {
          fn.emitRemark("thread-role split not applied: more than one engine "
                        "loop in the tile loop body");
          return std::nullopt;
        }
        m.nLoop = inner;
      }
    }
    if (isa<UnpackAccOp, UnpackAccF32Op>(op))
      m.unpacks.push_back(&op);
  }
  if (!m.pack || !m.nLoop) {
    fn.emitRemark("thread-role split not applied: the tile loop body is not "
                  "the pack-plus-engine-loop form");
    return std::nullopt;
  }

  // The single-row scratch: the pack writes crouton row 0 of it, which is
  // the single-thread assumption this split removes. A pack addressing any
  // other row is a form this pass did not make and does not claim.
  auto packRow = m.pack.getRow().getDefiningOp<arith::ConstantIndexOp>();
  if (!packRow || packRow.value() != 0) {
    fn.emitRemark("thread-role split not applied: the pack does not address "
                  "crouton row 0 of a one-row scratch");
    return std::nullopt;
  }
  m.scratch = m.pack.getDst();
  m.scratchAlloc = m.scratch.getDefiningOp<memref::AllocOp>();
  if (!m.scratchAlloc || m.mLoop->isAncestor(m.scratchAlloc)) {
    fn.emitRemark("thread-role split not applied: the scratch is not an "
                  "allocation outside the tile loop");
    return std::nullopt;
  }
  auto scratchType = dyn_cast<MemRefType>(m.scratch.getType());
  if (!scratchType || scratchType.getRank() != 5 ||
      scratchType.getDimSize(0) != 1 || !scratchType.hasStaticShape() ||
      !scratchType.getLayout().isIdentity()) {
    fn.emitRemark("thread-role split not applied: the scratch is not the "
                  "static identity-layout one-row crouton array");
    return std::nullopt;
  }

  // The scratch's users must be exactly the pack, the engine loop and its
  // deallocation: a fourth reader observes the row this split hands to the
  // consumer, and this pass does not claim to reason about it.
  for (OpOperand &use : m.scratch.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == m.pack || owner == m.nLoop ||
        m.nLoop->isAncestor(owner) || isa<memref::DeallocOp>(owner))
      continue;
    fn.emitRemark("thread-role split not applied: the scratch has a reader "
                  "besides the pack, the engine loop and its deallocation");
    return std::nullopt;
  }

  // The shared buffers, read off the engine ops themselves (never off
  // operand positions this pass guesses): every mma's weight, every
  // acc_read's bias and AR. A disagreement is a form this pass has no
  // single-function story for.
  Value wt, bias, ar;
  bool disagree = false;
  m.nLoop->walk([&](Operation *o) {
    if (auto mma = dyn_cast<MmaOp>(o)) {
      if (wt && wt != mma.getWt())
        disagree = true;
      wt = mma.getWt();
    }
    if (auto read = dyn_cast<AccReadOp>(o)) {
      if (bias && bias != read.getBias())
        disagree = true;
      if (ar && ar != read.getDst())
        disagree = true;
      bias = read.getBias();
      ar = read.getDst();
    }
  });
  if (disagree || !wt || !bias || !ar) {
    fn.emitRemark("thread-role split not applied: the engine loop's mmas or "
                  "acc_reads do not agree on one weight, bias and AR");
    return std::nullopt;
  }
  m.wt = wt;
  m.bias = bias;
  m.ar = ar;
  // The shared buffers must be defined outside the m-loop: they become the
  // work function's parameters, so a value local to one iteration cannot
  // stand in for them.
  for (Value v : {m.wt, m.bias, m.ar}) {
    Operation *def = v.getDefiningOp();
    if (def && m.mLoop->isAncestor(def)) {
      fn.emitRemark("thread-role split not applied: an engine operand is "
                    "defined inside the tile loop");
      return std::nullopt;
    }
  }

  // No engine work outside the matched loop EXCEPT the one form the emitters
  // actually produce: `hmx.bias_init`, which runs once at function entry.
  // bias_init IS engine work (it loads the conversion state the acc_reads
  // use, one of the five engine leaves), so on the split it must move into
  // the section with the rest -- the producer thread must never issue an
  // engine instruction, or its per-kernel ensure blocks forever on the
  // bound thread's lifetime lock. Re-initialising the state per section
  // call is redundant but correct (same block, same register set) and costs
  // one 256-byte load against a tile of engine work. The depth-2 pipelined
  // form peels an epilogue holding a second engine run and a second matmul
  // holds another whole loop; neither is a bias_init, so both still decline
  // here.
  bool engineOutside = false;
  fn.walk([&](Operation *op) {
    if (!mustRunOnEngineThread(op) || m.mLoop->isAncestor(op))
      return;
    if (auto init = dyn_cast<BiasInitOp>(op)) {
      m.biasInits.push_back(op);
      return;
    }
    engineOutside = true;
  });
  if (engineOutside) {
    fn.emitRemark("thread-role split not applied: engine work outside the "
                  "matched tile loop (the pipelined depth-2 form, or a "
                  "second matmul); the first form transforms exactly one "
                  "serial source ring");
    return std::nullopt;
  }
  for (Operation *init : m.biasInits) {
    if (cast<BiasInitOp>(init).getBias() != m.bias) {
      fn.emitRemark("thread-role split not applied: the bias init's "
                    "conversion state is not the block the acc_reads use");
      return std::nullopt;
    }
  }

  // The read-outs to move: their non-induction operands must be defined
  // outside the m-loop, or the post-drain position cannot see them.
  for (Operation *unpack : m.unpacks) {
    for (Value operand : unpack->getOperands()) {
      if (operand == m.mLoop.getInductionVar())
        continue;
      Operation *def = operand.getDefiningOp();
      if (def && m.mLoop->isAncestor(def)) {
        fn.emitRemark("thread-role split not applied: an in-loop read-out "
                      "depends on a value computed inside the tile loop, so "
                      "it cannot move past the exit drain");
        return std::nullopt;
      }
    }
  }

  // Every value the engine loop uses from outside must be one this pass
  // knows how to rebuild or hand over: the scratch, the induction, the
  // shared buffers, or a constant (re-created in the work function).
  /// Collected and validated by `collectEngineOutsideUses` during the
  /// emission; the check lives there so the walk and the use of the walk
  /// cannot drift apart.
  return m;
}

/// VTCM bytes this function's own allocations already hold. The
/// post-bufferization image (memref.alloc in the VTCM space, plus the
/// resident-weight declaration), same as hmx-partition's own version -- the
/// two passes run at the same pipeline point and must see the same number
/// or this pass's budget check is fiction.
static int64_t roleVtcmBytesCommitted(func::FuncOp fn) {
  int64_t bytes = 0;
  fn.walk([&](memref::AllocOp alloc) {
    auto type = dyn_cast<MemRefType>(alloc.getType());
    if (!type || !type.hasStaticShape() ||
        type.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
      return;
    Type elem = type.getElementType();
    if (!elem.isIntOrFloat())
      return;
    bytes += type.getNumElements() * (elem.getIntOrFloatBitWidth() / 8);
  });
  if (auto module = fn->getParentOfType<ModuleOp>())
    if (auto resident =
            module->getAttrOfType<IntegerAttr>("hmx.weight_resident_bytes"))
      bytes += resident.getInt();
  return bytes;
}

/// Declare (once) and look up a private runtime function. NO mutex: this
/// pass's runOnOperation holds hmxModuleStateMutex for its whole run (the
/// manifest write it exists for), so every module mutation below is already
/// serialized -- a nested lock_guard on the same non-recursive mutex is the
/// deadlock, not the safety.
static func::FuncOp declareRoleRuntime(ModuleOp module, StringRef name,
                                       ArrayRef<Type> params,
                                       ArrayRef<Type> results = {}) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto fn = func::FuncOp::create(builder, builder.getUnknownLoc(), name,
                                 builder.getFunctionType(params, results));
  fn.setPrivate();
  return fn;
}

/// One word of the per-launch buffer table (HmxRoleHandoff.h): an internal,
/// zero-initialised i32 global holding one buffer's address. Same no-mutex
/// rule as declareRoleRuntime.
static LLVM::GlobalOp getOrCreateRoleBufferWord(ModuleOp module,
                                                StringRef name) {
  if (auto existing = module.lookupSymbol<LLVM::GlobalOp>(name))
    return existing;
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  return LLVM::GlobalOp::create(
      builder, module.getLoc(), builder.getI32Type(),
      /*isConstant=*/false, LLVM::Linkage::Internal, name,
      builder.getIntegerAttr(builder.getI32Type(), 0),
      /*alignment=*/0, /*addrSpace=*/0, /*dsoLocal=*/false,
      /*thread_local=*/false);
}

/// Store one buffer's address into its table word. The store is an LLVM op
/// in a `func.func` body on purpose: the word is an llvm.mlir.global (it has
/// to be, for the LLVM-dialect entry point to address it after
/// convert-func-to-llvm), and pre-lowered LLVM ops inside a func body are
/// left untouched by that conversion -- the same mechanism
/// HexagonLWPInstrumentation's stores rely on.
static void storeRoleBufferWord(OpBuilder &b, Location loc, ModuleOp module,
                                StringRef name, Value buffer) {
  LLVM::GlobalOp word = getOrCreateRoleBufferWord(module, name);
  Value addrIdx =
      memref::ExtractAlignedPointerAsIndexOp::create(b, loc, buffer);
  Value addr = arith::IndexCastOp::create(b, loc, b.getI32Type(), addrIdx);
  Value slot = LLVM::AddressOfOp::create(b, loc, word);
  LLVM::StoreOp::create(b, loc, addr, slot);
}

/// The producer-side submit, with the short return the ABI makes the
/// caller's problem.
///
/// A short return means the ring was full and the tail groups were NOT
/// taken: ignoring it would be silent corruption (HmxRoleExecutor.h's
/// submit contract), so the recovery is drain-then-resubmit -- drain is the
/// one barrier the frozen ABI offers, it retires EVERYTHING in flight, and
/// an empty ring always has room for the remainder (the ring is Mt deep and
/// at most Mt descriptors exist, so this path is unreachable in the first
/// form -- the check is emitted anyway because the ABI demands it and a
/// future shallower ring makes it live; what the drain costs when it IS
/// live is the S3 probe's question, by design).
static void emitRoleSubmit(OpBuilder &b, Location loc, func::FuncOp submitFn,
                           func::FuncOp drainFn, Value descBuf,
                           Value countIdx, Value countI32) {
  Type i32 = b.getI32Type();
  Type index = b.getIndexType();
  Value addrIdx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc,
                                                                 descBuf);
  Value addr = arith::IndexCastOp::create(b, loc, i32, addrIdx);
  Value accepted =
      func::CallOp::create(b, loc, submitFn, ValueRange{addr, countI32})
          .getResult(0);
  // NE, not EQ: the recovery is for the SHORT return (groups the ring did
  // not take). The first emission tested EQ, so the common all-accepted path
  // drained after every batch -- serializing the pipeline the split exists
  // to overlap -- while a real short return (unreachable with a ring Mt deep)
  // would have dropped its tail silently. Found in the S3 root-cause hunt.
  Value shortReturn = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                            accepted, countI32);
  scf::IfOp recovery = scf::IfOp::create(b, loc, shortReturn,
                                         /*withElseRegion=*/false);
  OpBuilder inner(b.getContext());
  inner.setInsertionPoint(recovery.getThenRegion().front().getTerminator());
  func::CallOp::create(inner, loc, drainFn, ValueRange{});
  Value acceptedIdx =
      arith::IndexCastOp::create(inner, loc, index, accepted);
  Value cWords = arith::ConstantIndexOp::create(
      inner, loc, kHmxRoleDescWords);
  Value offWords = arith::MulIOp::create(inner, loc, acceptedIdx, cWords);
  Value rem = arith::SubIOp::create(inner, loc, countIdx, acceptedIdx);
  Value remWords = arith::MulIOp::create(inner, loc, rem, cWords);
  Value one = arith::ConstantIndexOp::create(inner, loc, 1);
  Value tailBuf = memref::SubViewOp::create(
      inner, loc, descBuf, SmallVector<OpFoldResult>{offWords},
      SmallVector<OpFoldResult>{remWords},
      SmallVector<OpFoldResult>{one});
  Value tailIdx =
      memref::ExtractAlignedPointerAsIndexOp::create(inner, loc, tailBuf);
  Value tailAddr = arith::IndexCastOp::create(inner, loc, i32, tailIdx);
  Value remI32 = arith::IndexCastOp::create(inner, loc, i32, rem);
  func::CallOp::create(inner, loc, submitFn,
                       ValueRange{tailAddr, remI32});
}

/// Fill one descriptor word in the submit buffer.
static void storeRoleDescWord(OpBuilder &b, Location loc, Value descBuf,
                              Value base /*index of word 0 of this descriptor*/,
                              int64_t field, Value word /*i32*/) {
  Value at = arith::ConstantIndexOp::create(b, loc, field);
  Value idx = arith::AddIOp::create(b, loc, base, at);
  memref::StoreOp::create(b, loc, word, descBuf, ValueRange{idx});
}

/// Record the handoff for HmxToLLVMPass: the one place the outlined work
/// function's identity, the ring depth and the four buffer types survive
/// the conversions between here and there.
static void recordRoleHandoff(ModuleOp module, func::FuncOp engine,
                              func::FuncOp work, int64_t depth,
                              MemRefType rowsType, MemRefType wtType,
                              MemRefType biasType, MemRefType arType) {
  // No mutex -- see declareRoleRuntime: the pass holds it for its whole run.
  OpBuilder builder(module.getContext());
  auto existing = module->getAttrOfType<ArrayAttr>(kHmxRoleHandoffsAttr);
  SmallVector<Attribute> records;
  if (existing)
    records.assign(existing.getValue().begin(), existing.getValue().end());
  records.push_back(builder.getDictionaryAttr(
      SmallVector<NamedAttribute>{
          builder.getNamedAttr(kHmxRoleEngineField,
                               builder.getStringAttr(engine.getName())),
          builder.getNamedAttr(kHmxRoleWorkField,
                               builder.getStringAttr(work.getName())),
          builder.getNamedAttr(kHmxRoleDepthField,
                               builder.getI64IntegerAttr(depth)),
          builder.getNamedAttr(kHmxRoleRowsField, TypeAttr::get(rowsType)),
          builder.getNamedAttr(kHmxRoleWtField, TypeAttr::get(wtType)),
          builder.getNamedAttr(kHmxRoleBiasField, TypeAttr::get(biasType)),
          builder.getNamedAttr(kHmxRoleArField, TypeAttr::get(arType))}));
  module->setAttr(kHmxRoleHandoffsAttr, builder.getArrayAttr(records));
}

/// Build the work function the bound thread runs, and rewrite the tile
/// loop into the producer side. Returns failure only on IR the matcher
/// should not have produced (a hard error, not a decline).
static LogicalResult emitRoleSplit(func::FuncOp fn, RoleSplitMatch &m) {
  ModuleOp module = fn->getParentOfType<ModuleOp>();
  OpBuilder b(fn.getContext());
  Location loc = m.pack.getLoc();
  Type i32 = b.getI32Type();
  Type index = b.getIndexType();

  auto scratchType = cast<MemRefType>(m.scratch.getType());
  auto rowsType = MemRefType::get(
      SmallVector<int64_t>{m.mt, scratchType.getDimSize(1),
                           scratchType.getDimSize(2),
                           scratchType.getDimSize(3),
                           scratchType.getDimSize(4)},
      scratchType.getElementType(), AffineMap{},
      scratchType.getMemorySpace());

  // ---- budget: the rotating rows against the room the scratch retires ----
  int64_t rowBytes =
      scratchType.getNumElements() * (scratchType.getElementTypeBitWidth() / 8);
  int64_t budget = HmxTarget().vtcmBudget;
  int64_t committed = roleVtcmBytesCommitted(fn);
  int64_t room = budget - committed + rowBytes;
  int64_t need = m.mt * rowBytes;
  if (need > room) {
    fn.emitRemark("thread-role split not applied: the ")
        << need << " byte rotating scratch does not fit the " << room
        << " bytes left after the kernel's commitments (role-split-nobudget);"
           " the kernel keeps the single-thread form";
    return success();
  }

  // ---- validate the engine loop's outside uses before touching anything ----
  // Each must be the scratch, the induction, a shared buffer, or a constant
  // this pass can re-create; anything else is a form the matcher over-
  // claimed and the emission refuses loudly.
  SmallVector<Value> outsideUses;
  for (Operation &op : m.nLoop.getBody()->getOperations()) {
    op.walk([&](Operation *o) {
      for (Value operand : o->getOperands()) {
        // Block arguments of the engine loop itself (its induction var):
        // cloned with the loop.
        if (!operand.getDefiningOp() &&
            m.nLoop->isAncestor(operand.getParentBlock()->getParentOp()))
          continue;
        if (operand.getDefiningOp() &&
            m.nLoop->isAncestor(operand.getDefiningOp()))
          continue; // defined inside the engine loop: cloned with it
        if (!llvm::is_contained(outsideUses, operand))
          outsideUses.push_back(operand);
      }
    });
  }
  for (Value value : outsideUses) {
    if (value == m.scratch || value == m.wt || value == m.bias ||
        value == m.ar || value == m.mLoop.getInductionVar())
      continue;
    if (isa_and_nonnull<arith::ConstantOp>(value.getDefiningOp()))
      continue;
    return fn.emitError()
           << "thread-role split: the engine loop uses a value that is "
              "neither the scratch, a shared buffer, the tile index nor a "
              "constant; the matcher over-claimed this shape";
  }

  // ---- the work function ------------------------------------------------
  std::string workName = kHmxRoleWorkFnPrefix.str();
  func::FuncOp work;
  {
    // No mutex here either (see declareRoleRuntime): the pass's own guard
    // covers the probe-then-create window. The builder's insertion point is
    // the module body's END -- a builder without one builds a detached op
    // that never lands in the module, and everything below (the clones into
    // its body, the handoff record naming it) would describe a function no
    // pass can reach.
    OpBuilder mb(module.getContext());
    mb.setInsertionPointToEnd(module.getBody());
    for (unsigned n = 1; module.lookupSymbol(workName); ++n)
      workName = (Twine(kHmxRoleWorkFnPrefix) + "_" + Twine(n)).str();
    work = func::FuncOp::create(
        mb, mb.getUnknownLoc(), workName,
        mb.getFunctionType({rowsType, m.wt.getType(), m.bias.getType(),
                            m.ar.getType(), index, index},
                           {}));
  }
  work.setPrivate();
  // The role marker (ROADMAP1001 section 3.2): names this function's
  // REGION for every reader between here and the conversion -- this pass's
  // own bystander checks, the LWP instrumentation's skip. The LLVM-side
  // thread contract ("hexagon_hmx") is set by HmxToLLVMPass on the
  // llvm.func, from the handoff record, because attributes do not survive
  // every conversion in between.
  work->setAttr(kHmxThreadRoleAttr,
                StringAttr::get(b.getContext(), kHmxThreadRoleHmx));
  Block *workEntry = work.addEntryBlock();
  Value rowsParam = workEntry->getArgument(0);
  Value wtParam = workEntry->getArgument(1);
  Value biasParam = workEntry->getArgument(2);
  Value arParam = workEntry->getArgument(3);
  Value m0Param = workEntry->getArgument(4);
  Value countParam = workEntry->getArgument(5);

  OpBuilder wb(b.getContext());
  wb.setInsertionPointToEnd(workEntry);
  Value wc0 = arith::ConstantIndexOp::create(wb, loc, 0);
  Value wc1 = arith::ConstantIndexOp::create(wb, loc, 1);
  // The engine loop's own bounds, re-created here (the originals live in
  // the producer's iteration and cannot be cloned across).
  Value nLb = m.nLoop.getLowerBound();
  Value nUb = m.nLoop.getUpperBound();
  Value nStep = m.nLoop.getStep();
  auto rebuildConstant = [&](Value orig) -> Value {
    // Every constant kind arith has re-creates through the generic form:
    // the value attribute is copied verbatim, so index/int/float all round-
    // trip (only index constants exist in the shapes this pass matches, but
    // the generic form costs nothing and declines nothing).
    auto c = dyn_cast<arith::ConstantOp>(orig.getDefiningOp());
    if (!c)
      return Value();
    return arith::ConstantOp::create(wb, loc, c.getType(), c.getValue());
  };
  Value nLbNew = rebuildConstant(nLb);
  Value nUbNew = rebuildConstant(nUb);
  Value nStepNew = rebuildConstant(nStep);
  if (!nLbNew || !nUbNew || !nStepNew)
    return m.nLoop.emitError()
           << "thread-role split: the engine loop's bounds are not "
              "constants, which the matcher did not establish";

  // for i in 0..count: m = m0 + i; the engine loop, cloned, on row m.
  // The bias init goes first: it is engine work (the producer must never
  // issue it -- see the matcher's note), and the conversion state it loads
  // is what this call's acc_reads read. One init per section call is
  // redundant across calls but correct (same block, same register set).
  for (Operation *init : m.biasInits) {
    IRMapping initMap;
    initMap.map(m.bias, biasParam);
    wb.clone(*init, initMap);
  }
  scf::ForOp perTile = scf::ForOp::create(wb, loc, wc0, countParam, wc1,
                                           ValueRange{});
  OpBuilder tileB(b.getContext());
  tileB.setInsertionPoint(perTile.getBody()->getTerminator());
  Value mVal = arith::AddIOp::create(tileB, loc, m0Param,
                                     perTile.getInductionVar());

  IRMapping map;
  map.map(m.scratch, rowsParam);
  map.map(m.wt, wtParam);
  map.map(m.bias, biasParam);
  map.map(m.ar, arParam);
  map.map(m.mLoop.getInductionVar(), mVal);
  map.map(nLb, nLbNew);
  map.map(nUb, nUbNew);
  map.map(nStep, nStepNew);
  for (Value value : outsideUses) {
    if (value == m.scratch || value == m.wt || value == m.bias ||
        value == m.ar || value == m.mLoop.getInductionVar() ||
        value == nLb || value == nUb || value == nStep)
      continue;
    // A constant: re-create it in the work function so the clone does not
    // reach back into the producer.
    Value fresh = rebuildConstant(value);
    assert(fresh && "validated above");
    map.map(value, fresh);
  }
  Operation *clonedEngine = tileB.clone(*m.nLoop.getOperation(), map);
  // The crouton row: in the source form every mma reads row 0 because the
  // scratch IS one row; in the work function the row is the tile index.
  // Patch exactly the mma's `m` operand (operand 2 of (act, wt, m, n, k),
  // HmxOps.td) where it holds the cloned engine loop's lower bound -- the
  // value the m-body's row-0 constant became -- and nothing else: the other
  // operands that can hold a zero constant (the k base of a shallow-K mma)
  // are constants created INSIDE the engine loop, cloned as fresh ops, and
  // never equal to the mapped bound.
  clonedEngine->walk([&](Operation *o) {
    if (auto mma = dyn_cast<MmaOp>(o))
      if (mma.getM() == nLbNew)
        mma.setOperand(2, mVal);
  });
  func::ReturnOp::create(wb, loc);

  // ---- the producer side -------------------------------------------------
  func::FuncOp submitFn =
      declareRoleRuntime(module, kHmxRoleSubmitFn, {i32, i32}, {i32});
  func::FuncOp drainFn = declareRoleRuntime(module, kHmxRoleDrainFn, {});

  // The rotating rows replace the one-row scratch, at the scratch's own
  // position; the submit buffer is a stack object for the whole loop
  // (drain runs before the function returns, so the frame outlives every
  // descriptor the runtime holds -- the read-out split's descriptor does
  // the same for the same reason).
  b.setInsertionPoint(m.scratchAlloc);
  Value rows = memref::AllocOp::create(b, loc, rowsType, ValueRange{});
  Value descBuf = memref::AllocaOp::create(
      b, loc,
      MemRefType::get({kRoleSubmitBatch * kHmxRoleDescWords}, i32));

  // The per-launch buffer table, written before anything can read it: the
  // entry point rebuilds the work function's memref arguments from these
  // four words (HmxRoleHandoff.h's table comment is the whole story). This
  // is the producer's first act with the buffers, ahead of the tile loop
  // and therefore ahead of the first submit that could hand a descriptor to
  // the consumer.
  storeRoleBufferWord(b, loc, module, kHmxRoleRowsGlobal, rows);
  storeRoleBufferWord(b, loc, module, kHmxRoleWtGlobal, m.wt);
  storeRoleBufferWord(b, loc, module, kHmxRoleBiasGlobal, m.bias);
  storeRoleBufferWord(b, loc, module, kHmxRoleArGlobal, m.ar);

  // The old scratch's deallocation becomes the rows' -- AFTER the exit
  // drain below, which is emitted after the loop; find the dealloc now,
  // erase it, and re-create it at the end.
  SmallVector<memref::DeallocOp> scratchDeallocs;
  for (Operation *user : m.scratch.getUsers())
    if (auto d = dyn_cast<memref::DeallocOp>(user))
      scratchDeallocs.push_back(d);
  for (memref::DeallocOp d : scratchDeallocs) {
    b.setInsertionPoint(d);
    memref::DeallocOp::create(b, loc, rows);
    d.erase();
  }
  // The allocation itself is erased only at the end, with the pack and the
  // engine loop that still reference it -- an op destroyed with live uses
  // is a hard error, and the order here is the whole difference.

  // In the loop body: the pack moves to row m of the rows array, the
  // descriptor is filled at slot m mod batch, and every batch-th iteration
  // submits.
  //
  // THE PACK'S ROW IS A VIEW, NOT A ROW OPERAND. The pack's (row, col)
  // operands name the source block's position AND the destination crouton's
  // position with the same values (HmxOps.td: "Packs the 32x32 fp16 block
  // at (row, col) of the row-major src into crouton (row, col) of the ...
  // dst"), and every emitter before this pass kept the two equal by
  // construction -- the staged ring packs staging tile (0, 0) into crouton
  // (0, 0) of a one-row scratch. Redirecting the destination to row m by
  // moving the ROW OPERAND would move the SOURCE with it: the leaf reads
  // source rows [row*32, +32) (HMXLayout.c hmx__pack_32x32 pads
  // out-of-range source rows with zero), and the staging tile holds exactly
  // one tile's 32 rows, so every tile past the first would read out of
  // range and pack zeros. Measured on the device: the first dual-role
  // launch returned tile 0 correct and tiles 1..7 exactly zero
  // (logs/s3-rootcause-2026-10-08). The one-row view of `rows` at row m
  // keeps the operands at their source meaning -- crouton (0, 0) of the
  // m-th row view IS row m of the rows array (croutonAddr addresses a
  // sliced array at its real position, HmxToLLVMPass.cpp) -- while the
  // source stays staging tile (0, 0).
  b.setInsertionPoint(m.nLoop);
  Value mInd = m.mLoop.getInductionVar();
  SmallVector<OpFoldResult> rowOffs{mInd, b.getIndexAttr(0), b.getIndexAttr(0),
                                    b.getIndexAttr(0), b.getIndexAttr(0)};
  SmallVector<OpFoldResult> rowSizes{
      b.getIndexAttr(1), b.getIndexAttr(scratchType.getDimSize(1)),
      b.getIndexAttr(scratchType.getDimSize(2)),
      b.getIndexAttr(scratchType.getDimSize(3)),
      b.getIndexAttr(scratchType.getDimSize(4))};
  SmallVector<OpFoldResult> rowStrides{b.getIndexAttr(1), b.getIndexAttr(1),
                                       b.getIndexAttr(1), b.getIndexAttr(1),
                                       b.getIndexAttr(1)};
  Value rowView = memref::SubViewOp::create(b, loc, rows, rowOffs, rowSizes,
                                            rowStrides);
  PackActOp pack = PackActOp::create(
      b, loc, TypeRange{rowView.getType()}, rowView, m.pack.getSrc(),
      m.pack.getRow(), m.pack.getCol(), m.pack.getCountAttr(),
      m.pack.getValidRowsAttr(), m.pack.getValidColsAttr());
  for (const auto &named : m.pack->getAttrs())
    if (named.getName() != "count" && named.getName() != "valid_rows" &&
        named.getName() != "valid_cols")
      pack->setAttr(named.getName(), named.getValue());
  m.pack.erase();

  // The descriptor fields this pass owns: the crouton row is the tile index
  // (the rows array is one row per tile, so the row IS the identity), and
  // the work function re-derives it from rowStart -- which is why the
  // descriptor carries both and the entry point forwards only rowStart and
  // rowCount (the slot word is written because the ABI has the field, and
  // left to the runtime uninterpreted, exactly like the read-out batch's
  // informational words).
  Value cBatch = arith::ConstantIndexOp::create(b, loc, kRoleSubmitBatch);
  Value mMod = arith::RemSIOp::create(b, loc, mInd, cBatch);
  Value cWords = arith::ConstantIndexOp::create(b, loc, kHmxRoleDescWords);
  Value descBase = arith::MulIOp::create(b, loc, mMod, cWords);
  storeRoleDescWord(b, loc, descBuf, descBase, kHmxRoleDescSlot,
                    arith::IndexCastOp::create(b, loc, i32, mInd));
  storeRoleDescWord(b, loc, descBuf, descBase, kHmxRoleDescRowStart,
                    arith::IndexCastOp::create(b, loc, i32, mInd));
  storeRoleDescWord(
      b, loc, descBuf, descBase, kHmxRoleDescRowCount,
      arith::ConstantIntOp::create(b, loc, 1, 32));
  storeRoleDescWord(
      b, loc, descBuf, descBase, kHmxRoleDescPad,
      arith::ConstantIntOp::create(b, loc, 0, 32));
  storeRoleDescWord(
      b, loc, descBuf, descBase, kHmxRoleDescEventLo,
      arith::ConstantIntOp::create(b, loc, 0, 32));
  storeRoleDescWord(
      b, loc, descBuf, descBase, kHmxRoleDescEventHi,
      arith::ConstantIntOp::create(b, loc, 0, 32));

  Value cOne = arith::ConstantIndexOp::create(b, loc, 1);
  Value cZero = arith::ConstantIndexOp::create(b, loc, 0);
  Value completed = arith::AddIOp::create(b, loc, mInd, cOne);
  Value boundary = arith::RemSIOp::create(b, loc, completed, cBatch);
  Value atBoundary = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                           boundary, cZero);
  scf::IfOp submitIf =
      scf::IfOp::create(b, loc, atBoundary, /*withElseRegion=*/false);
  OpBuilder sb(b.getContext());
  sb.setInsertionPoint(submitIf.getThenRegion().front().getTerminator());
  emitRoleSubmit(sb, loc, submitFn, drainFn, descBuf, cBatch,
                 arith::ConstantIntOp::create(sb, loc, kRoleSubmitBatch, 32));

  // ---- after the loop: the tail batch, the barrier, the read-outs ------
  // The in-loop read-outs are MOVED, not dropped: each is cloned into a
  // fresh loop after the exit drain (the only producer-side point where
  // every tile's engine work is provably done -- the consumer owns `ar`
  // rows until then), and the originals are erased only once every clone
  // is in place. The constants below are created AT this position: reusing
  // the boundary math's constants would cross the m-loop's region boundary
  // (they were created inside it), and a use of them out here is exactly
  // the non-dominating operand the verifier rejects.
  b.setInsertionPointAfter(m.mLoop);
  Value cZero2 = arith::ConstantIndexOp::create(b, loc, 0);
  Value cOne2 = arith::ConstantIndexOp::create(b, loc, 1);
  int64_t fullBatches = m.mt / kRoleSubmitBatch;
  int64_t tailCount = m.mt - fullBatches * kRoleSubmitBatch;
  if (tailCount > 0) {
    Value cTail = arith::ConstantIndexOp::create(b, loc, tailCount);
    emitRoleSubmit(b, loc, submitFn, drainFn, descBuf, cTail,
                   arith::ConstantIntOp::create(b, loc, tailCount, 32));
  }
  // The exit drain: the kernel does not return until every group's engine
  // work has landed in `ar`, because `ar` is the caller's memory.
  func::CallOp::create(b, loc, drainFn, ValueRange{});
  if (!m.unpacks.empty()) {
    Value cMt = arith::ConstantIndexOp::create(b, loc, m.mt);
    scf::ForOp readoutLoop =
        scf::ForOp::create(b, loc, cZero2, cMt, cOne2, ValueRange{});
    OpBuilder rb(b.getContext());
    rb.setInsertionPoint(readoutLoop.getBody()->getTerminator());
    for (Operation *unpack : m.unpacks) {
      IRMapping remap;
      remap.map(mInd, readoutLoop.getInductionVar());
      rb.clone(*unpack, remap);
    }
  }

  // The engine loop and the in-loop read-outs leave the producer body only
  // now, after everything that cloned from them is in place: the former
  // lives in the work function, the latter past the exit drain. The bias
  // inits left with the work function too -- erased last because the work
  // function cloned them.
  m.nLoop.erase();
  for (Operation *unpack : m.unpacks)
    unpack->erase();
  for (Operation *init : m.biasInits)
    init->erase();
  // Now nothing references the one-row scratch: the pack was re-created on
  // the rows, the engine loop lives in the work function, the dealloc was
  // replaced above.
  m.scratchAlloc.erase();

  recordRoleHandoff(module, fn, work, m.mt, rowsType,
                    cast<MemRefType>(m.wt.getType()),
                    cast<MemRefType>(m.bias.getType()),
                    cast<MemRefType>(m.ar.getType()));

  // ---- static checks, on the committed IR (ROADMAP1001 section 3.3) ------
  // Hard errors, not remarks: a split that left engine work behind or took
  // wide vectors along is a bug in this pass, and shipping it silently
  // wrong is the one outcome worse than a red build. The checks run only
  // when the split committed, which is why they live here.
  bool engineLeft = false;
  fn.walk([&](Operation *op) {
    if (mustRunOnEngineThread(op))
      engineLeft = true;
  });
  if (engineLeft)
    return fn.emitError("thread-role split: engine work remained in the "
                        "producer after the split");
  bool wideVector = false;
  work.walk([&](Operation *op) {
    if (op->getDialect() &&
        op->getDialect()->getNamespace() ==
            vector::VectorDialect::getDialectNamespace())
      wideVector = true;
  });
  if (wideVector)
    return work.emitError("thread-role split: the outlined engine section "
                          "holds vector-dialect work; the bound thread must "
                          "not touch HVX (the upstream hexagon_hmx "
                          "contract)");
  return success();
}

struct ThreadRolePartitionPass
    : public mlir::hmx::impl::ThreadRolePartitionBase<ThreadRolePartitionPass> {
  using ThreadRolePartitionBase::ThreadRolePartitionBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HmxDialect, func::FuncDialect, scf::SCFDialect,
                    arith::ArithDialect, memref::MemRefDialect,
                    vector::VectorDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The manifest is module state and an interface pass may run concurrently for
    // sibling functions, so the write is serialised on the same mutex
    // hmx-partition uses (HmxPartitionPass.cpp's runOnOperation).
    std::lock_guard<std::mutex> manifestGuard(hmxModuleStateMutex());

    func::FuncOp fn = cast<func::FuncOp>(getOperation());
    ModuleOp module = fn->getParentOfType<ModuleOp>();
    if (!module) {
      fn.emitError("thread-role-partition requires a builtin.module parent");
      return signalPassFailure();
    }

    // The outlined vector read-out is the executor's WORK FUNCTION, not a kernel,
    // and it is not this pass's business for two independent reasons.
    //
    // One: there is no decision left to make. `__hmx_readout` holds the
    // `hmx.unpack_acc` the read-out split moved off the engine thread, and it
    // runs on the vector thread BY CONSTRUCTION -- the resident executor is the
    // only thing that can reach it, via the `configure` handoff. Classifying its
    // regions would produce a verdict describing a placement that was never a
    // choice, which is the same thing the `verdict.regions == 0` return below
    // exists to avoid.
    //
    // Two: it is what makes the multi-kernel check below fire. That check exists
    // because `topology` is module-level; `__hmx_readout` is not a second kernel,
    // so counting it makes the module look like two HMX kernels and turns a
    // correct compile into "found at least one besides this one".
    //
    // It only exists at all when `enable-hmx-vector-readout` is on (default off),
    // and the marker is the same attribute HmxToLLVMPass's handoff wiring and the
    // manifest bridge recount key on, so "is this the executor's work function"
    // has one answer in the codebase rather than a name convention.
    if (fn->hasAttr(kHmxReadoutOutlinedAttr))
      return;

    // The role split's outlined engine section is the same kind of
    // bystander, for the same two reasons: its placement is not a decision
    // (it runs on the bound thread BY CONSTRUCTION -- only the executor
    // reaches it, through the channel's bind), and classifying it would
    // make the module look like two HMX kernels. The marker is the
    // `hex.thread_role` region attribute the emission set (ROADMAP1001
    // section 3.2), which is what every other bystander check keys on too.
    if (fn->hasAttr(kHmxThreadRoleAttr))
      return;

    // A kernel with no HMX work has no role decision to make, and writing
    // "single-role-hvx" for it would read as a placement that was chosen rather
    // than one that was unnecessary.
    KernelVerdict verdict;
    classifyRegions(fn, verdict);
    if (verdict.regions == 0)
      return;

    // `topology` is a module-level field, and this pass is an interface pass, so
    // a module holding two HMX kernels would record whichever one ran last. That
    // is worse than an error: the reader would get a confident verdict about a
    // kernel nobody asked about. A module with one HMX kernel plus non-HMX
    // helpers is fine -- those return above and never reach this check. The
    // executor's outlined read-out is the third kind of bystander and is excluded
    // by the same marker, for the reasons above.
    unsigned hmxFunctions = 0;
    for (auto other : module.getOps<func::FuncOp>()) {
      if (other == fn || other->hasAttr(kHmxReadoutOutlinedAttr))
        continue;
      KernelVerdict probe;
      classifyRegions(other, probe);
      if (probe.regions)
        ++hmxFunctions;
    }
    if (hmxFunctions) {
      fn.emitError("thread-role-partition: the module's manifest topology field "
                   "is module-level and cannot describe more than one HMX "
                   "kernel; found at least one besides this one");
      return signalPassFailure();
    }

    StringRef topology = decideTopology(verdict);

    // The split itself. `role-split-ok` is the verdict that a per-tile
    // producer stream exists; the emission below turns it into the two-role
    // form (outlined engine section + submit/drain producer), and EVERY
    // decline inside names its reason in a remark -- the split is never
    // silently skipped, and a declined kernel keeps today's single-thread
    // IR with the topology still recorded below. The static checks run on
    // the committed IR inside the emitter, where the outlined function is
    // at hand.
    if (topology == kSplitOk) {
      if (std::optional<RoleSplitMatch> m = matchSplitLoop(fn)) {
        if (failed(emitRoleSplit(fn, *m)))
          return signalPassFailure();
      }
    }

    if (failed(setHmxManifestTopology(module, topology, verdict.regions)))
      signalPassFailure();
  }
};

} // namespace
