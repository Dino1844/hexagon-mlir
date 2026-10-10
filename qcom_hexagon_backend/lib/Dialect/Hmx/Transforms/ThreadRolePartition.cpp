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
#include "hexagon/Dialect/Hmx/Transforms/HmxIndexFold.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxReadoutHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxRoleHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxTarget.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

// The one VTCM byte ledger: this pass's budget read goes through it rather than
// through a private copy of hmx-partition's walk (see the header for why).
#include "HmxVtcmLedger.h"

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
///
/// THE TWO FORMS (R3, 2026-10-09). The serial source ring (the folded serial
/// loop, or the staged loop at depth 1) is the form the first split handled:
///
///     for m in 0..Mt:                       no iter_args
///       [stage(m), await(m)]                staged arm only
///       pack_act(m) -> scratch              the producer work
///       for n...: acc_clear/mma/acc_read    the engine half, ONE nest
///       [unpack_acc(m)]                     the in-loop read-out, if any
///
/// The pipelined depth-2 form (what `scf::pipelineForLoop` turns that into)
/// adds a prologue, a peeled epilogue, and a (token, slot) iter-arg pair the
/// steady loop carries:
///
///     <parity select>, stage(0)                        prologue: issue tile 0
///     for m in 0..Mt-1 iter_args(token, slot):         upper is Mt-1
///       <parity select for m+1>, stage(m+1)            issue NEXT tile
///       await(token, slot)                             await THIS tile
///       pack_act(m) -> scratch
///       for n...: acc_clear/mma/acc_read(m)            steady engine nest
///       [unpack_acc(m) | publish-if]                   read-out
///       yield(next token, next slot)
///     <folded Mt-1 row>
///     await(last token, last slot)                     epilogue: tile Mt-1
///     pack_act(Mt-1) -> scratch
///     for n...: acc_clear/mma/acc_read(Mt-1)           peeled engine nest
///     [unpack_acc(Mt-1) | tail publish]
///
/// Both engine nests outline into the SAME work function: the steady nest is
/// the clone source (its row becomes the work function's per-tile row), the
/// peeled nest is verified structurally equivalent and simply dropped -- the
/// peeled tile is the LAST DESCRIPTOR (rowStart = Mt-1), and the section's
/// (rows, wt, bias, ar, m0, count) signature already covers it. That is the
/// "peel 尾的 tile 就是最后几个 descriptor" property the merged R2+R3 plan
/// rests on.
/// One in-loop publish boundary of the read-out split (R2): the boundary
/// block itself plus the descriptor facts its publish hands the vector
/// thread, extracted once at match time. The emission erases the block and
/// MOVES its stores and call into a lagged boundary block one submit batch
/// later, so the facts extracted here are the ones the emission cannot
/// re-derive: the read-out batch descriptor and its address (the clamped
/// tail-site publish re-creates a whole call on them).
struct PublishSite {
  scf::IfOp ifOp;  ///< the boundary block (guarded `(m+1) % G == 0`)
  Value readoutBuf;  ///< the read-out batch descriptor (memref<6xi32>)
  Value readoutAddr; ///< its i32 base address (the publish call's operand 0)
};

struct RoleSplitMatch {
  scf::ForOp mLoop;
  PackActOp pack;         ///< the producer work, directly in the m-loop body
  scf::ForOp nLoop;       ///< the engine half, directly in the m-loop body
  Value scratch;          ///< the pack's destination (one crouton row)
  memref::AllocOp scratchAlloc;
  Value wt;               ///< shared buffers the engine half closes over
  Value bias;
  Value ar;
  Value rowValue;         ///< the steady nest's row expression (iv + 0)
  SmallVector<Operation *> unpacks; ///< in-loop read-outs (readout-OFF form)
  SmallVector<Operation *> biasInits; ///< function-level engine preamble to move
  int64_t mt = 0;         ///< tile count: the ring depth and the scratch rows
  int64_t upper = 0;      ///< folded loop upper: Mt (serial), Mt-1 (pipelined)

  // ---- the read-out coexistence state (R2; see collectReadoutState) ----
  /// The read-out split (hmx-vector-readout) runs BEFORE this pass in the
  /// production pipeline, so the in-loop read-outs arrive as publishes to the
  /// vector executor rather than as `hmx.unpack_acc` ops. The two forms are
  /// mutually exclusive and shape everything the emission does with the
  /// read-out: publishes are re-emitted one submit batch LATE (the zero-lag
  /// form serialized the producer behind the engine -- see the emission
  /// comment); unpacks move past the exit drain (the first form's behavior).
  SmallVector<PublishSite> publishIfs; ///< in-loop publish boundaries (R2)
  SmallVector<func::CallOp> tailPublishes; ///< post-loop publish(es)
  int64_t readoutGroup = 0; ///< G: the publish boundary's group size
  bool readoutOn = false;

  // ---- the pipelined (depth-2) form ----
  bool pipelined = false;
  AwaitOp epilogueAwait;   ///< the peeled epilogue's await (producer side)
  PackActOp epiloguePack;  ///< the peeled epilogue's pack
  scf::ForOp epilogueNest; ///< the peeled epilogue's engine nest
  Operation *epilogueUnpack = nullptr; ///< its read-out (readout-OFF form)
};

/// Is `op` one of the m-loop body's direct children?
static bool directlyIn(Operation *op, scf::ForOp loop) {
  return op->getParentOp() == loop.getOperation();
}

/// Does this loop directly hold engine work anywhere beneath it?
static bool holdsEngineWork(scf::ForOp loop) {
  bool holds = false;
  loop->walk([&](Operation *o) { holds |= mustRunOnEngineThread(o); });
  return holds;
}

/// The two read-out ops' shared accessors, spelled once each so the f32
/// residual form's extra operand cannot shift an index under a caller. Both
/// ops expose the same four facts (AR source, destination, row, column); the
/// accessors keep that spelled as facts rather than as operand positions.
static Value unpackSrc(Operation *op) {
  if (auto u = dyn_cast<UnpackAccOp>(op))
    return u.getSrc();
  return cast<UnpackAccF32Op>(op).getSrc();
}
static Value unpackDst(Operation *op) {
  if (auto u = dyn_cast<UnpackAccOp>(op))
    return u.getDst();
  return cast<UnpackAccF32Op>(op).getDst();
}
static Value unpackRow(Operation *op) {
  if (auto u = dyn_cast<UnpackAccOp>(op))
    return u.getRow();
  return cast<UnpackAccF32Op>(op).getRow();
}
static Value unpackCol(Operation *op) {
  if (auto u = dyn_cast<UnpackAccOp>(op))
    return u.getCol();
  return cast<UnpackAccF32Op>(op).getCol();
}

/// Structural equivalence of the steady engine nest and the peeled epilogue's
/// engine nest, modulo the row value.
///
/// The pipeliner produces the epilogue by CLONING the source loop body, so the
/// two nests are the same computation on different row expressions by
/// construction -- but "by construction" is exactly what this pass does not
/// trust: a future pipeliner change would leave the two silently different,
/// and the emission below clones only the STEADY nest and covers the peeled
/// tile with a descriptor. That coverage argument is only sound if the two
/// nests really are the same work, so the check is structural and complete
/// rather than spot: same op sequence, same attributes, and every operand
/// either the same SSA value (the shared buffers), the row pair
/// (`rowA`/`rowB`), a constant of the same folded value, or a block argument
/// of the correspondingly-positioned region (the nests' own inner induction
/// variables).
///
/// A constant can be spelled by different SSA ops and still be the same
/// number (the pipeliner re-materialises constants per clone), which is why
/// the constant case folds rather than compares. Everything else is SSA
/// identity: the shared buffers (`scratch`, `wt`, `bias`, `ar`) and the loop
/// bounds are defined outside both nests and are the same values in both, and
/// any operand that is neither is a shape this check does not understand, so
/// it declines.
static bool nestEquivalent(Operation *a, Operation *b, Value rowA, Value rowB,
                           DenseMap<Value, Value> &argMap) {
  if (a->getName() != b->getName())
    return false;
  if (a->getNumOperands() != b->getNumOperands() ||
      a->getNumResults() != b->getNumResults() ||
      a->getNumRegions() != b->getNumRegions())
    return false;
  if (a->getAttrs() != b->getAttrs())
    return false;
  for (auto [oa, ob] : llvm::zip(a->getOperands(), b->getOperands())) {
    if (oa == rowA && ob == rowB)
      continue;
    if (oa == ob)
      continue; // the same SSA value: a shared buffer, a shared constant
    // Constants re-materialised per clone: same folded number, different op.
    std::optional<int64_t> fa = foldIndex(oa);
    std::optional<int64_t> fb = foldIndex(ob);
    if (fa && fb && *fa == *fb)
      continue;
    // Block arguments of correspondingly-positioned regions: the nests' own
    // inner-loop induction variables (and any iter_args).
    auto it = argMap.find(oa);
    if (it != argMap.end() && it->second == ob)
      continue;
    return false;
  }
  for (auto [ra, rb] : llvm::zip(a->getRegions(), b->getRegions())) {
    if (ra.getBlocks().size() != rb.getBlocks().size())
      return false;
    for (auto [ba, bb] : llvm::zip(ra.getBlocks(), rb.getBlocks())) {
      if (ba.getArguments().size() != bb.getArguments().size())
        return false;
      for (auto [va, vb] : llvm::zip(ba.getArguments(), bb.getArguments()))
        argMap[va] = vb;
      if (ba.getOperations().size() != bb.getOperations().size())
        return false;
      for (auto [oa, ob] : llvm::zip(ba.getOperations(), bb.getOperations()))
        if (!nestEquivalent(&oa, &ob, rowA, rowB, argMap))
          return false;
    }
  }
  return true;
}

/// The descriptor buffer a published batch address is built from, or a null
/// Value for a form this pass does not know.
///
/// The read-out split publishes an i32 ADDRESS (the runtime ABI's
/// `void *ar, *dst`), so `buildDescriptor` emits the aligned pointer PLUS the
/// buffer's own descriptor offset, element-scaled -- the same rule `asAddress`
/// applies on the LLVM side, and the reason a view into the middle of another
/// buffer (a grid-tiled matmul's span) is published pre-offset rather than as
/// the bare aligned pointer every span shares. The address therefore reads
/// the descriptor through its two metadata queries, and the emission needs
/// the BUFFER those name. So the query is: pure arithmetic and casts, the two
/// metadata queries, and nothing else -- the queries name the buffer (and
/// must agree on it), the arithmetic is walked, and an impure op, a region,
/// or a second buffer is a form this pass did not make. The bare
/// `index_cast(extract)` spelling this pass matched before walks the same
/// way, so both spellings are accepted.
static Value readoutDescriptorBuffer(Value published) {
  Value buffer;
  SmallVector<Value> worklist{published};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *def = value.getDefiningOp();
    if (!def)
      continue; // a function argument or a result of something uninteresting
    if (isa<memref::ExtractAlignedPointerAsIndexOp,
            memref::ExtractStridedMetadataOp>(def)) {
      Value named = def->getOperand(0);
      // Two names in one expression: the address depends on two buffers, so
      // which one the descriptor is would be a guess.
      if (buffer && buffer != named)
        return {};
      buffer = named;
      continue;
    }
    // `mlir::isPure` does not look inside regions, so a region-bearing
    // "pure" op can still have effects; the region check is why that
    // distinction matters here.
    if (!mlir::isPure(def) || def->getNumRegions() != 0)
      return {};
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return buffer;
}

/// The read-out coexistence state of a matched tile loop (R2).
///
/// The production pipeline runs `hmx-vector-readout` BEFORE this pass, so by
/// the time the split looks at the loop, the read-out is either
///   * MOVED ONTO the vector executor: the in-loop `hmx.unpack_acc` ops are
///     gone and their positions hold boundary-guarded `exec_publish` calls,
///     with a tail publish after the loop; or
///   * STILL INLINE (`hmx.unpack_acc` ops in the body): the read-out split is
///     off, or it declined this shape for its own reasons.
/// The two are mutually exclusive; a loop holding both is a form this pass
/// did not make and declines.
///
/// WHY DETECTION IS EMISSION-SAFE (the part that makes the coexistence sound)
/// ---------------------------------------------------------------------
/// With publishes present, the in-loop read-out reads `ar` rows that the
/// BOUND THREAD writes, not this loop's body -- the read-out split matched
/// when its own proof (an in-body `acc_read`) was still there, and this pass
/// is about to move that `acc_read` onto T_HMX. The order the two passes
/// established is what carries the proof across that move: each publish
/// names the group the read-out it replaced covered, and the split re-emits
/// it one submit batch LATE with a `wait_retired` barrier AHEAD of it, so
/// the composed order at the lagged boundary is submit(batch) ...
/// wait_retired(previous batch's end + 1) ... publish(previous batch's
/// group), and "publish only after the engine sections of every tile the
/// publish names have RETIRED" is re-established at a finer grain than the
/// exit drain the first form used. The lag is a scheduling fact, not a
/// correctness one: the wait still precedes its own publish, and the groups
/// published are exactly the groups the read-out split published. The
/// emission's comment carries the full argument, including why the lag
/// exists at all.
static LogicalResult collectReadoutState(func::FuncOp fn, RoleSplitMatch &m) {
  // In-loop: publish-if blocks and/or unpacks, as DIRECT children of the body.
  SmallVector<scf::IfOp> publishIfs;
  for (Operation &op : m.mLoop.getBody()->without_terminator()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool publish = false;
      ifOp->walk([&](Operation *o) {
        if (auto call = dyn_cast<func::CallOp>(o))
          publish |= call.getCallee() == kHmxExecPublishFn;
      });
      if (publish)
        publishIfs.push_back(ifOp);
      continue;
    }
    if (isa<UnpackAccOp, UnpackAccF32Op>(op))
      m.unpacks.push_back(&op);
  }
  if (!publishIfs.empty() && !m.unpacks.empty()) {
    fn.emitRemark("thread-role split not applied: the tile loop holds both a "
                  "vector-executor publish and an inline read-out; the two "
                  "read-out forms are mutually exclusive and this shape is "
                  "neither");
    return failure();
  }

  // Post-loop: the tail publish (a plain call after the loop -- the read-out
  // split anchors it at the peeled row's read-out or right after the loop).
  // Collected by walking the function and excluding the loop's own subtree:
  // every publish this pass did not find in the body IS a tail publish, and
  // there is at most one of those per read-out loop (the read-out split's
  // rewriteLoop emits exactly one).
  fn.walk([&](func::CallOp call) {
    if (call.getCallee() == kHmxExecPublishFn && !m.mLoop->isAncestor(call))
      m.tailPublishes.push_back(call);
  });

  m.readoutOn = !publishIfs.empty() || !m.tailPublishes.empty();
  if (m.readoutOn &&
      (!m.unpacks.empty() ||
       (m.pipelined && m.epilogueUnpack != nullptr))) {
    fn.emitRemark("thread-role split not applied: the kernel holds a "
                  "vector-executor publish and an inline read-out at once; "
                  "the two read-out forms are mutually exclusive");
    return failure();
  }

  // THE COVERAGE GATE (re-derived for the lagged emission, 2026-10-09). The
  // original gate was a deadlock check: a wait ahead of a publish at its OWN
  // boundary could target tiles the covering submit had not handed the ring
  // yet, so the submit batch had to divide G. The lagged emission publishes
  // one submit batch LATE, and its wait's target was necessarily submitted
  // iterations earlier -- tiles 0..m are all in the ring once the producer
  // reaches iteration m + batch, because the last of them rides the submit
  // at or before iteration m + batch - 1 -- so THAT deadlock is structurally
  // gone: the lag is strictly safer than the gate's original demand.
  //
  // The divisibility is KEPT, for the shape property it still buys: with the
  // submit batch dividing G, every publish boundary is also a submit
  // boundary, so each lagged publish fires exactly at the NEXT submit
  // boundary after its group (the `submit(batch k+1) -> wait_retired(batch
  // k's end + 1) -> publish(batch k's group)` position structure), and at
  // most ONE live group's lag falls past the loop (the second-to-last lags
  // to `(K-1)G + batch <= KG <= upper`, inside) -- which is what lets the
  // after-loop emission place that one group, with its wait, at the single
  // tail-submit boundary. A G finer than the submit batch would scatter
  // publishes across non-boundary iterations and could clamp several groups
  // past the loop at once -- a form this emission's position argument does
  // not describe, so it declines loudly and the kernel keeps the
  // single-thread pipeline with its read-out intact.
  if (!publishIfs.empty()) {
    Value iv = m.mLoop.getInductionVar();
    for (scf::IfOp ifOp : publishIfs) {
      auto cmp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
      if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq) {
        fn.emitRemark("thread-role split not applied: a publish boundary "
                      "condition is not the equality the read-out split "
                      "emits");
        return failure();
      }
      Value rem = nullptr, zero = nullptr;
      if (dyn_cast_or_null<arith::RemSIOp>(cmp.getLhs().getDefiningOp())) {
        rem = cmp.getLhs();
        zero = cmp.getRhs();
      } else if (dyn_cast_or_null<arith::RemSIOp>(cmp.getRhs().getDefiningOp())) {
        rem = cmp.getRhs();
        zero = cmp.getLhs();
      } else {
        fn.emitRemark("thread-role split not applied: a publish boundary "
                      "condition has no remainder term (the read-out "
                      "split's `(m+1) %% G` form)");
        return failure();
      }
      auto remsi = cast<arith::RemSIOp>(rem.getDefiningOp());
      auto addi = dyn_cast_or_null<arith::AddIOp>(remsi.getLhs().getDefiningOp());
      std::optional<int64_t> group = foldIndex(remsi.getRhs());
      std::optional<int64_t> one = addi ? foldIndex(addi.getRhs()) : std::nullopt;
      std::optional<int64_t> zeroV = foldIndex(zero);
      if (!addi || addi.getLhs() != iv || !group || *group < 1 || !one ||
          !zeroV || *one != 1 || *zeroV != 0) {
        fn.emitRemark("thread-role split not applied: a publish boundary "
                      "condition is not the `(m+1) %% G == 0` form the "
                      "read-out split emits");
        return failure();
      }
      // One group size for every boundary: the read-out split runs with one
      // `hmx-readout-batch` per function, and the lagged emission below
      // guards every re-emitted publish with the one G it parsed -- two
      // different Gs in one loop is a form this pass did not make.
      if (m.readoutGroup == 0) {
        m.readoutGroup = *group;
      } else if (*group != m.readoutGroup) {
        fn.emitRemark("thread-role split not applied: the tile loop's publish "
                      "boundaries do not share one group size (the read-out "
                      "split emits one `hmx-readout-batch` per function)");
        return failure();
      }
      if (*group % kRoleSubmitBatch != 0) {
        fn.emitRemark()
            << "thread-role split not applied: the read-out publishes every "
            << *group << " tiles but the role submit batches every "
            << kRoleSubmitBatch
            << "; the lagged publish rides the submit boundary after its "
               "group, which only exists when every publish boundary is a "
               "submit boundary. The kernel keeps the single-thread "
               "pipeline with its read-out intact";
        return failure();
      }
    }

    // The site facts, extracted once here (never re-parsed in the emission):
    // the read-out batch descriptor a publish hands the vector thread, and
    // the rowStart arithmetic the lagged emission patches. The emission
    // MOVES each boundary block's stores and call into the lagged block and
    // re-bases one operand of the rowStart subtraction, so what it patches
    // is verified here -- where a surprise is a loud decline instead of a
    // silent coverage change.
    for (scf::IfOp ifOp : publishIfs) {
      func::CallOp publish;
      unsigned publishCalls = 0;
      ifOp->walk([&](func::CallOp call) {
        if (call.getCallee() == kHmxExecPublishFn) {
          publish = call;
          ++publishCalls;
        }
      });
      if (publishCalls != 1) {
        fn.emitRemark("thread-role split not applied: a publish boundary "
                      "block does not hold exactly one publish call (the "
                      "read-out split emits one per boundary)");
        return failure();
      }
      // The descriptor address: the read-out split's own spelling
      // (buildDescriptor), which reached across in emitPublish as a bare
      // `index_cast(extract_aligned_pointer_as_index(buffer))`. It now carries
      // the buffer's descriptor offset inside -- a batch word is an i32
      // ADDRESS, so a view into the middle of another buffer is published
      // pre-offset (HmxVectorReadoutPass.cpp: the 2048^3-B multi-span fix),
      // and the expression is `index_cast(addi(extract(buf),
      // muli(strided_metadata(buf) offset, bytes)))`. What the emission needs
      // is the BUFFER, so the arithmetic is walked rather than spelled out:
      // pure ops only, and the one `extract_aligned_pointer_as_index` in the
      // expression names it. Two extracts, an impure op, or none at all is an
      // unknown form -- the same decline, for the same reason (a surprise here
      // would silently point the lagged publish at the wrong descriptor).
      Value readoutBuf = readoutDescriptorBuffer(publish.getOperand(0));
      if (!readoutBuf) {
        fn.emitRemark("thread-role split not applied: a publish call's "
                      "descriptor address is not the read-out split's "
                      "extract-and-cast form");
        return failure();
      }
      for (Value v : {readoutBuf, publish.getOperand(0)}) {
        Operation *def = v.getDefiningOp();
        if (def && m.mLoop->isAncestor(def)) {
          fn.emitRemark("thread-role split not applied: a publish descriptor "
                        "is built inside the tile loop (the read-out split "
                        "builds it before the loop so it dominates every "
                        "publish)");
          return failure();
        }
      }
      // The per-item calling discipline: one descriptor per publish call.
      // The count operand is an i32 constant, read directly -- foldIndex
      // folds index-typed chains only.
      auto count = publish.getOperand(1).getDefiningOp<arith::ConstantOp>();
      auto countAttr =
          count ? dyn_cast_or_null<IntegerAttr>(count.getValue()) : nullptr;
      if (!countAttr || countAttr.getInt() != 1) {
        fn.emitRemark("thread-role split not applied: a publish call does not "
                      "carry exactly one batch (the read-out split's "
                      "per-item calling discipline)");
        return failure();
      }
      // The rowStart store: index_cast(subi(iv, G-1)) into the descriptor's
      // rowStart field -- the arithmetic the lagged emission re-bases from
      // the induction to `iv - batch`.
      memref::StoreOp rowStartStore;
      for (Operation &op : ifOp.getThenRegion().front()) {
        auto store = dyn_cast<memref::StoreOp>(op);
        if (!store || store.getMemRef() != readoutBuf)
          continue;
        auto at = store.getIndices()[0].getDefiningOp<arith::ConstantIndexOp>();
        if (at && at.value() == kHmxReadoutRowStart) {
          rowStartStore = store;
          break;
        }
      }
      auto rowCast = rowStartStore
                         ? dyn_cast_or_null<arith::IndexCastOp>(
                               rowStartStore.getValueToStore().getDefiningOp())
                         : nullptr;
      auto rowSub = rowCast ? dyn_cast_or_null<arith::SubIOp>(
                                  rowCast.getIn().getDefiningOp())
                            : nullptr;
      if (!rowSub || rowSub.getLhs() != iv ||
          foldIndex(rowSub.getRhs()).value_or(-1) != m.readoutGroup - 1) {
        fn.emitRemark("thread-role split not applied: a publish's rowStart is "
                      "not the read-out split's `iv - (G-1)` form, so the "
                      "lagged re-emission cannot re-base it");
        return failure();
      }
      m.publishIfs.push_back(PublishSite{ifOp, readoutBuf,
                                         publish.getOperand(0)});
    }
  }
  return success();
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
    // The read-outs are collected by `collectReadoutState` below -- the one
    // collector, so the same op can never enter `m.unpacks` twice (a doubled
    // entry would dangle after the first erase).
  }
  if (!m.pack || !m.nLoop) {
    fn.emitRemark("thread-role split not applied: the tile loop body is not "
                  "the pack-plus-engine-loop form");
    return std::nullopt;
  }

  // Static 0..upper step-1 trip, with the upper FOLDED rather than read off
  // a bare constant: the pipelined steady loop's upper is `subi Mt, 1` (the
  // pipeliner's arithmetic), the serial loop's is the constant Mt itself.
  // The descriptor math (rowStart = m, slot = m) and the ring depth (= Mt)
  // are derived from the induction variable counting tiles from zero.
  auto lb = m.mLoop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = m.mLoop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb || !step || lb.value() != 0 || step.value() != 1) {
    fn.emitRemark("thread-role split not applied: the tile loop is not the "
                  "static 0..upper step-1 form the emitters produce");
    return std::nullopt;
  }
  std::optional<int64_t> upper = foldIndex(m.mLoop.getUpperBound());
  if (!upper || *upper < 1) {
    fn.emitRemark("thread-role split not applied: the tile loop's upper "
                  "bound does not fold to a constant");
    return std::nullopt;
  }
  m.upper = *upper;

  // THE FORM DISCRIMINATOR: the pipeliner's steady loop carries its (token,
  // slot) pair as iter_args; the serial source loop carries nothing. One bit,
  // and it is structural -- not a guess about which emitter ran.
  m.pipelined = m.mLoop.getNumRegionIterArgs() != 0;

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

  // The steady nest's ROW: every acc_read in the nest names the same row
  // expression, and that expression is the induction (serial) or the
  // pipeliner's `iv + 0` shift of it (pipelined). The work function's clone
  // maps THIS value -- not the induction -- to its per-tile row, which is
  // why the offset must be exactly zero: any other row arithmetic is a
  // shape whose per-tile meaning this pass cannot restate.
  Value row;
  m.nLoop->walk([&](Operation *o) {
    if (auto read = dyn_cast<AccReadOp>(o)) {
      if (row && row != read.getM())
        disagree = true;
      row = read.getM();
    }
  });
  if (disagree || !row ||
      inductionOffset(row, m.mLoop.getInductionVar()).value_or(1) != 0) {
    fn.emitRemark("thread-role split not applied: the engine loop's "
                  "acc_reads do not name the tile loop's own row");
    return std::nullopt;
  }
  m.rowValue = row;

  // The tile count: the AR array's own row dimension (the acc_reads write
  // exactly those rows), cross-checked against the loop upper -- Mt for the
  // serial source loop, Mt-1 for the pipelined steady loop (the pipeliner
  // peels the last tile into the epilogue checked below). A disagreement is
  // a form the emitters do not produce.
  auto arType = dyn_cast<MemRefType>(m.ar.getType());
  if (!arType || !arType.hasStaticShape() || arType.getRank() < 1) {
    fn.emitRemark("thread-role split not applied: the accumulator array is "
                  "not a static crouton array");
    return std::nullopt;
  }
  m.mt = arType.getDimSize(0);
  if (m.mt < 1 || m.upper != (m.pipelined ? m.mt - 1 : m.mt)) {
    fn.emitRemark("thread-role split not applied: the tile loop's upper "
                  "bound does not match the accumulator's row count (the "
                  "form the emitters produce is Mt, or Mt-1 with a peeled "
                  "epilogue)");
    return std::nullopt;
  }

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

  // ---- the pipelined form's prologue and peeled epilogue ----------------
  //
  // Everything here is OUTSIDE the steady loop and unique per function: the
  // prologue is the one StageOp before the loop (it issues tile 0 through
  // the parity slot select), the epilogue is the await + pack + engine nest
  // (+ read-out) for the peeled last tile. The prologue stays producer-side
  // untouched (the DMA is engine-independent); the epilogue's pack moves to
  // the rotating rows like the steady pack, and its engine nest is verified
  // equivalent to the steady nest and covered by the tail descriptor.
  AwaitOp epilogueAwait;
  PackActOp epiloguePack;
  scf::ForOp epilogueNest;
  StageOp prologueStage;
  size_t stagesOutside = 0, awaitsOutside = 0, packsOutside = 0,
         nestsOutside = 0;
  fn.walk([&](Operation *op) {
    if (m.mLoop->isAncestor(op))
      return;
    if (auto stage = dyn_cast<StageOp>(op)) {
      ++stagesOutside;
      prologueStage = stage;
    }
    if (auto await = dyn_cast<AwaitOp>(op)) {
      ++awaitsOutside;
      epilogueAwait = await;
    }
    if (auto pack = dyn_cast<PackActOp>(op)) {
      ++packsOutside;
      epiloguePack = pack;
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      // Directly in the function body: the pipeliner emits the peeled
      // epilogue's nest as a sibling of the steady loop, and a nest nested
      // deeper (inside some other region) is not that.
      if (loop != m.mLoop && loop->getParentOp() == fn &&
          holdsEngineWork(loop)) {
        ++nestsOutside;
        epilogueNest = loop;
      }
    }
    if (isa<UnpackAccOp, UnpackAccF32Op>(op))
      m.epilogueUnpack = op;
  });
  if (m.pipelined) {
    if (stagesOutside != 1 || awaitsOutside != 1 || packsOutside != 1 ||
        nestsOutside != 1 || !prologueStage || !epilogueAwait ||
        !epiloguePack || !epilogueNest) {
      fn.emitRemark("thread-role split not applied: the pipelined form's "
                    "prologue/epilogue is not the one-issue, one-await, "
                    "one-pack, one-engine-nest shape the pipeliner emits");
      return std::nullopt;
    }
    // The prologue issues tile 0: its source row folds to 0 (the pipeliner
    // spells it `muli 0, 32`).
    if (foldIndex(prologueStage->getOperand(1)).value_or(1) != 0) {
      fn.emitRemark("thread-role split not applied: the pipelined prologue "
                    "does not issue tile 0");
      return std::nullopt;
    }
    // The epilogue's pack writes the SAME one-row scratch (its row operand
    // is 0 for the same reason the steady pack's is).
    if (epiloguePack.getDst() != m.scratch) {
      fn.emitRemark("thread-role split not applied: the peeled epilogue's "
                    "pack does not write the tile loop's scratch");
      return std::nullopt;
    }
    auto epRow = epiloguePack.getRow().getDefiningOp<arith::ConstantIndexOp>();
    if (!epRow || epRow.value() != 0) {
      fn.emitRemark("thread-role split not applied: the peeled epilogue's "
                    "pack does not address crouton row 0 of the scratch");
      return std::nullopt;
    }
    // The epilogue's nest: same shared buffers, and its row is the folded
    // LAST tile (the pipeliner's peeled-row chain).
    Value epRowValue;
    bool epDisagree = false;
    epilogueNest->walk([&](Operation *o) {
      if (auto mma = dyn_cast<MmaOp>(o)) {
        if (mma.getWt() != m.wt)
          epDisagree = true;
      }
      if (auto read = dyn_cast<AccReadOp>(o)) {
        if (read.getBias() != m.bias || read.getDst() != m.ar)
          epDisagree = true;
        if (epRowValue && epRowValue != read.getM())
          epDisagree = true;
        epRowValue = read.getM();
      }
    });
    if (epDisagree || !epRowValue ||
        foldIndex(epRowValue).value_or(-1) != m.mt - 1) {
      fn.emitRemark("thread-role split not applied: the peeled epilogue's "
                    "engine nest does not read the shared buffers at the "
                    "last tile's row");
      return std::nullopt;
    }
    // The two nests are the SAME work on different rows -- the coverage
    // argument for covering the peeled tile with a descriptor rests on it,
    // so it is verified structurally, not assumed from the pipeliner.
    DenseMap<Value, Value> argMap;
    if (!nestEquivalent(m.nLoop.getOperation(), epilogueNest.getOperation(),
                        m.rowValue, epRowValue, argMap)) {
      fn.emitRemark("thread-role split not applied: the peeled epilogue's "
                    "engine nest is not the steady nest on the last tile's "
                    "row, so one outlined section cannot cover both");
      return std::nullopt;
    }
    m.epilogueAwait = epilogueAwait;
    m.epiloguePack = epiloguePack;
    m.epilogueNest = epilogueNest;
  } else {
    // The serial source loop has no prologue and no epilogue; anything of
    // these kinds outside it is a second matmul's work or a form this pass
    // does not know, and the engine-outside check below declines it.
    (void)stagesOutside;
    (void)awaitsOutside;
    (void)packsOutside;
    (void)nestsOutside;
    m.epilogueUnpack = nullptr;
  }

  // No engine work outside the matched loop EXCEPT the forms the emitters
  // actually produce: `hmx.bias_init`, which runs once at function entry
  // (it moves into the section with the rest -- the producer thread must
  // never issue an engine instruction, or its per-kernel ensure blocks
  // forever on the bound thread's lifetime lock), and -- in the pipelined
  // form -- the peeled epilogue's engine nest, which the emission outlines
  // with the steady nest into the same section.
  bool engineOutside = false;
  fn.walk([&](Operation *op) {
    if (!mustRunOnEngineThread(op) || m.mLoop->isAncestor(op))
      return;
    if (m.pipelined && m.epilogueNest &&
        m.epilogueNest->isAncestor(op))
      return;
    if (auto init = dyn_cast<BiasInitOp>(op)) {
      m.biasInits.push_back(op);
      return;
    }
    engineOutside = true;
  });
  if (engineOutside) {
    fn.emitRemark("thread-role split not applied: engine work outside the "
                  "matched tile loop and its peeled epilogue (a second "
                  "matmul); the split transforms exactly one source ring "
                  "per kernel");
    return std::nullopt;
  }
  for (Operation *init : m.biasInits) {
    if (cast<BiasInitOp>(init).getBias() != m.bias) {
      fn.emitRemark("thread-role split not applied: the bias init's "
                    "conversion state is not the block the acc_reads use");
      return std::nullopt;
    }
  }

  // The scratch's users must be exactly the packs, the engine nests and its
  // deallocation: a further reader observes rows this split hands to the
  // consumer, and this pass does not claim to reason about it.
  for (OpOperand &use : m.scratch.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == m.pack || owner == m.nLoop ||
        m.nLoop->isAncestor(owner) || isa<memref::DeallocOp>(owner))
      continue;
    if (m.pipelined &&
        (owner == m.epiloguePack || m.epilogueNest->isAncestor(owner)))
      continue;
    fn.emitRemark("thread-role split not applied: the scratch has a reader "
                  "besides the packs, the engine nests and its deallocation");
    return std::nullopt;
  }

  // The read-out coexistence state: publishes (the read-out split ran) or
  // inline unpacks (it did not). Collected and validated by
  // `collectReadoutState`, whose comment is the soundness argument.
  if (failed(collectReadoutState(fn, m))) {
    return std::nullopt;
  }

  // The inline read-outs to move past the exit drain (readout-OFF form):
  // each one's ROW must be the loop's own row arithmetic (the move rewrites
  // it to the read-out loop's induction), and every other operand must be
  // defined outside the m-loop or the post-drain position cannot see it.
  if (!m.readoutOn) {
    for (Operation *unpack : m.unpacks) {
      Value rowOperand = unpackRow(unpack);
      if (inductionOffset(rowOperand, m.mLoop.getInductionVar()).value_or(1) != 0) {
        fn.emitRemark("thread-role split not applied: an in-loop read-out "
                      "does not read the tile loop's own row, so it cannot "
                      "move past the exit drain");
        return std::nullopt;
      }
      for (Value operand : unpack->getOperands()) {
        if (operand == rowOperand)
          continue;
        Operation *def = operand.getDefiningOp();
        if (def && m.mLoop->isAncestor(def)) {
          fn.emitRemark("thread-role split not applied: an in-loop read-out "
                        "depends on a value computed inside the tile loop, "
                        "so it cannot move past the exit drain");
          return std::nullopt;
        }
      }
    }
    // The pipelined form's PEELED read-out (readout-OFF): it must be the
    // steady body's read-out on the last tile's row -- same destination,
    // same count, same crouton column, same attributes -- because the
    // emission drops it and the moved read-out loop's last iteration covers
    // that row instead. Anything else is a shape whose coverage this pass
    // cannot restate.
    if (m.pipelined) {
      if (m.unpacks.empty() != (m.epilogueUnpack == nullptr)) {
        fn.emitRemark("thread-role split not applied: the pipelined form "
                      "has a read-out in the loop or in the epilogue but "
                      "not both; the coverage argument needs the pair");
        return std::nullopt;
      }
      if (m.epilogueUnpack) {
        if (foldIndex(unpackRow(m.epilogueUnpack)).value_or(-1) != m.mt - 1) {
          fn.emitRemark("thread-role split not applied: the peeled "
                        "epilogue's read-out does not read the last tile's "
                        "row");
          return std::nullopt;
        }
        Operation *steady = m.unpacks.front();
        if (steady->getName() != m.epilogueUnpack->getName() ||
            steady->getAttrs() != m.epilogueUnpack->getAttrs() ||
            unpackSrc(steady) != unpackSrc(m.epilogueUnpack) ||
            unpackDst(steady) != unpackDst(m.epilogueUnpack) ||
            unpackCol(steady) != unpackCol(m.epilogueUnpack) ||
            unpackRow(steady) == unpackRow(m.epilogueUnpack)) {
          fn.emitRemark("thread-role split not applied: the peeled "
                        "epilogue's read-out is not the steady read-out on "
                        "the last row (same destination and column, only "
                        "the row differs), so dropping it would lose or "
                        "duplicate a row");
          return std::nullopt;
        }
      }
    }
  }

  // Every value the engine loop uses from outside must be one this pass
  // knows how to rebuild or hand over: the scratch, the row, the shared
  // buffers, or a constant (re-created in the work function).
  /// Collected and validated during the emission; the check lives there so
  /// the walk and the use of the walk cannot drift apart.
  return m;
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

/// Erase `value`'s defining op once nothing uses it, cascading up its
/// operands -- the read-out pass's own helper (HmxVectorReadoutPass.cpp),
/// spelled here again because it is file-static there. Used to take out a
/// replaced publish boundary's guard arithmetic, whose pieces (the induction
/// arithmetic especially) may be shared with ops that must survive.
/// `isOpTriviallyDead`, never "would be dead if unused": the second says yes
/// for a value some other op still reads, and erasing that destroys a live
/// op.
static void eraseDeadDefiners(Value value, unsigned depth = 0) {
  if (depth > 8 || !value)
    return;
  Operation *def = value.getDefiningOp();
  if (!def || !mlir::isOpTriviallyDead(def))
    return;
  SmallVector<Value> operands(def->getOperands());
  def->erase();
  for (Value operand : operands)
    eraseDeadDefiners(operand, depth + 1);
}

/// One publish of a FIXED group (rowStart/rowCount as constants): the
/// read-out split's own store-and-call shape on a matched site's descriptor.
///
/// The in-loop lagged publish MOVES the read-out split's boundary block
/// instead of calling this (see the emission) -- whatever that pass emits
/// per publish rides along, and only one operand is re-based. This helper is
/// for the one position with no block to move: the CLAMPED group, whose
/// lagged position is the tail-submit boundary past the loop, so its publish
/// is re-created from the site facts the matcher extracted and verified.
static void emitFixedPublish(OpBuilder &b, Location loc, ModuleOp module,
                             const PublishSite &site, int64_t rowStart,
                             int64_t rowCount) {
  Type i32 = b.getI32Type();
  auto store = [&](int64_t field, Value word) {
    Value at = arith::ConstantIndexOp::create(b, loc, field);
    memref::StoreOp::create(b, loc, word, site.readoutBuf, ValueRange{at});
  };
  store(kHmxReadoutRowStart,
        arith::IndexCastOp::create(
            b, loc, i32, arith::ConstantIndexOp::create(b, loc, rowStart)));
  store(kHmxReadoutRowCount,
        arith::ConstantIntOp::create(b, loc, rowCount, 32));
  func::CallOp::create(
      b, loc, module.lookupSymbol<func::FuncOp>(kHmxExecPublishFn),
      ValueRange{site.readoutAddr,
                 arith::ConstantIntOp::create(b, loc, 1, 32)});
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
  int64_t room = hmx::vtcm::roomBytes(fn, budget, rowBytes);
  int64_t need = m.mt * rowBytes;
  if (need > room) {
    fn.emitRemark("thread-role split not applied: the ")
        << need << " byte rotating scratch does not fit the " << room
        << " bytes left after the kernel's commitments (role-split-nobudget);"
           " the kernel keeps the single-thread form";
    return success();
  }

  // ---- validate the engine loop's outside uses before touching anything ----
  // Each must be the scratch, the row (the tile-index fact the matcher
  // verified -- the induction itself in the serial form, the pipeliner's
  // `iv + 0` shift of it in the pipelined form), a shared buffer, or a
  // constant this pass can re-create; anything else is a form the matcher
  // over-claimed and the emission refuses loudly.
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
        value == m.ar || value == m.rowValue)
      continue;
    if (isa_and_nonnull<arith::ConstantOp>(value.getDefiningOp()))
      continue;
    return fn.emitError()
           << "thread-role split: the engine loop uses a value that is "
              "neither the scratch, a shared buffer, the row nor a "
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
  // THE ROW, not the induction: in the serial form the row IS the induction
  // (same SSA value), and in the pipelined form the pipeliner spells the row
  // as `addi iv, (muli 1, 0)` -- a distinct SSA value the induction never
  // appears as inside the nest. Mapping the row covers both, and the
  // outside-uses validation above already refused any other induction use.
  map.map(m.rowValue, mVal);
  map.map(nLb, nLbNew);
  map.map(nUb, nUbNew);
  map.map(nStep, nStepNew);
  for (Value value : outsideUses) {
    if (value == m.scratch || value == m.wt || value == m.bias ||
        value == m.ar || value == m.rowValue ||
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
  // The granular barrier (HmxRoleExecutor.h's wait_retired entry; the ABI
  // header's comment is the contract). Declared unconditionally so the
  // spelling exists in exactly one place for every arm of this pass, the
  // same rule as submit/drain; only the read-out coexistence path below
  // emits a call to it.
  func::FuncOp waitFn =
      declareRoleRuntime(module, kHmxRoleWaitRetiredFn, {i32});

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

  // ---- the pipelined form's peeled epilogue ------------------------------
  // The epilogue's pack moves to row Mt-1 of the rotating rows exactly the
  // way the steady pack moved to row m (through a one-row VIEW: the same
  // source-meaning argument as the steady pack's comment above), and the
  // peeled tile's descriptor is filled at the pack's position so the tail
  // batch below carries it. The epilogue's await stays producer-side (the
  // DMA is engine-independent), and its engine nest is NOT re-emitted here:
  // the matcher verified it is the steady nest on the last row, so the tail
  // descriptor's rowStart = Mt-1 runs it through the same outlined section.
  //
  // The builder anchors at the epilogue's ENGINE NEST, not at the pack it
  // replaces: the anchor op must outlive the emission (the pack is erased
  // mid-block and the steady emission anchors at its nest for exactly this
  // reason).
  if (m.pipelined) {
    b.setInsertionPoint(m.epilogueNest);
    Value cLast = arith::ConstantIndexOp::create(b, loc, m.mt - 1);
    SmallVector<OpFoldResult> epOffs{cLast, b.getIndexAttr(0),
                                     b.getIndexAttr(0), b.getIndexAttr(0),
                                     b.getIndexAttr(0)};
    Value epView = memref::SubViewOp::create(b, loc, rows, epOffs, rowSizes,
                                              rowStrides);
    PackActOp epPack = PackActOp::create(
        b, loc, TypeRange{epView.getType()}, epView, m.epiloguePack.getSrc(),
        m.epiloguePack.getRow(), m.epiloguePack.getCol(),
        m.epiloguePack.getCountAttr(), m.epiloguePack.getValidRowsAttr(),
        m.epiloguePack.getValidColsAttr());
    for (const auto &named : m.epiloguePack->getAttrs())
      if (named.getName() != "count" && named.getName() != "valid_rows" &&
          named.getName() != "valid_cols")
        epPack->setAttr(named.getName(), named.getValue());
    m.epiloguePack.erase();

    // The peeled tile's descriptor: slot (Mt-1) mod batch -- the tail batch
    // occupies slots 0..tailCount-1 (tailStart is a multiple of the batch),
    // and Mt-1 is its last entry. The constants are created AT this
    // position: the steady body's `cWords` lives inside the m-loop's region
    // and a use of it out here is the non-dominating operand the verifier
    // rejects.
    Value epMod = arith::ConstantIndexOp::create(
        b, loc, (m.mt - 1) % kRoleSubmitBatch);
    Value epWords = arith::ConstantIndexOp::create(b, loc, kHmxRoleDescWords);
    Value epBase = arith::MulIOp::create(b, loc, epMod, epWords);
    storeRoleDescWord(b, loc, descBuf, epBase, kHmxRoleDescSlot,
                      arith::ConstantIntOp::create(b, loc, m.mt - 1, 32));
    storeRoleDescWord(b, loc, descBuf, epBase, kHmxRoleDescRowStart,
                      arith::ConstantIntOp::create(b, loc, m.mt - 1, 32));
    storeRoleDescWord(
        b, loc, descBuf, epBase, kHmxRoleDescRowCount,
        arith::ConstantIntOp::create(b, loc, 1, 32));
    storeRoleDescWord(
        b, loc, descBuf, epBase, kHmxRoleDescPad,
        arith::ConstantIntOp::create(b, loc, 0, 32));
    storeRoleDescWord(
        b, loc, descBuf, epBase, kHmxRoleDescEventLo,
        arith::ConstantIntOp::create(b, loc, 0, 32));
    storeRoleDescWord(
        b, loc, descBuf, epBase, kHmxRoleDescEventHi,
        arith::ConstantIntOp::create(b, loc, 0, 32));
  }

  // ---- the read-out coexistence, LAGGED one batch (R2; the mechanism this
  // half exists for) ---------------------------------------------------------
  //
  // THE ZERO-LAG LESSON (2026-10-09, the merged R2+R3 A/B): the first
  // coexistence emission kept the read-out split's publish at its OWN
  // boundary and inserted the wait ahead of it, so one boundary iteration
  // read submit(batch) -> wait_retired(m+1) -> publish(group ending at m) --
  // the wait targeted the batch the SAME iteration had just submitted. The
  // producer parked until the engine chewed exactly that batch, then
  // published, then packed the next batch while the engine idled: producer
  // and engine serialized per batch, both engines took turns idling, and
  // every batch paid the wait round-trip. The S1-class A/B measured it at
  // +10.0 us / +27% against the default face (docs/results/
  // r2r3-ab-2026-10-09.md), and the arithmetic closed the case: producer
  // 8.3 + engine 21 (serialized) + ~9 wait round-trip + residual 8.3 ~= the
  // measured 47 us. THE RULE: a publish must lag the submit front --
  // waiting on the batch just submitted is putting the producer in line
  // behind the engine.
  //
  // THE LAGGED FORM (llama.cpp's `pop C_{i-1}` shape, the co-scheduling
  // review's section 4f): the publish of the group ending at tile m fires
  // at iteration m + kRoleSubmitBatch -- the NEXT submit boundary -- and
  // that iteration reads submit(batch k+1) -> wait_retired(m+1) ->
  // publish(batch k's group). The wait's target was submitted a full batch
  // of iterations earlier, so the engine has been chewing it the whole time
  // the producer packed and submitted the next batch: the target is met or
  // nearly met (`target <= retired returns immediately`, HmxRoleExecutor.h),
  // the producer stays a batch ahead, and the engine no longer idles
  // between batches. The vector thread's unpack of group k still overlaps
  // group k+1's engine work -- the coexistence's whole point -- without the
  // producer and the engine taking turns.
  //
  // The guard is the read-out split's own boundary test evaluated one batch
  // late (`(m-b+1) % G == 0`), plus `m-b+1 >= G`, which excludes the one
  // spurious firing at m = batch-1 (where `m-b+1 = 0` satisfies the
  // remainder test but would name a group ending before row 0). That
  // exclusion is also the FIRST group's edge case: the first boundary
  // publishes nothing, and group 0's publish fires at the second boundary --
  // the normal lag structure, not a special case. The published ops are the
  // boundary block's own stores and call, MOVED into the lagged block with
  // the rowStart arithmetic re-based from the induction to `m - batch`; the
  // coverage is therefore the read-out split's own -- every group it
  // published is published exactly once, one batch later. The LAST live
  // group's lagged position falls past the loop when `K*G + batch > upper`
  // (K the live group count): its publish moves to the first submit
  // boundary after the loop -- the tail submit -- and keeps its wait (see
  // the after-loop emission); `G >= batch` (the coverage gate) is what
  // makes at most ONE group fall out. The tail publish (the rows no in-loop
  // group named) needs no wait at all any more: it sits behind the exit
  // drain, which is the all-groups proof the wait used to stand in for.
  if (m.readoutOn) {
    for (PublishSite &site : m.publishIfs) {
      scf::IfOp ifOp = site.ifOp;
      // The lagged guard, at the boundary block's own position -- which is
      // after the submit-if the emission created at the engine nest's old
      // position, so the lagged publish reads submit-first, wait-second,
      // publish-third inside one iteration.
      b.setInsertionPoint(ifOp);
      Value cBatchL = arith::ConstantIndexOp::create(b, loc,
                                                     kRoleSubmitBatch);
      Value mPrev = arith::SubIOp::create(b, loc, mInd, cBatchL);
      Value cOneL = arith::ConstantIndexOp::create(b, loc, 1);
      Value completed = arith::AddIOp::create(b, loc, mPrev, cOneL);
      Value cGroupL = arith::ConstantIndexOp::create(b, loc,
                                                     m.readoutGroup);
      Value remainder = arith::RemSIOp::create(b, loc, completed, cGroupL);
      Value cZeroL = arith::ConstantIndexOp::create(b, loc, 0);
      Value atBoundary = arith::CmpIOp::create(
          b, loc, arith::CmpIPredicate::eq, remainder, cZeroL);
      Value isRealBoundary = arith::CmpIOp::create(
          b, loc, arith::CmpIPredicate::sge, completed, cGroupL);
      Value laggedBoundary =
          arith::AndIOp::create(b, loc, atBoundary, isRealBoundary);
      scf::IfOp laggedIf = scf::IfOp::create(b, loc, laggedBoundary,
                                             /*withElseRegion=*/false);
      // The wait first, at the top of the lagged block: it is the publish's
      // proof (the group's tiles have all RETIRED -- retire happens only
      // after the bound section returns), and the publish is what it
      // unblocks. The target is `mPrev + 1`: the tiles of the group the
      // block publishes end at mPrev, and retire-count mPrev+1 covers
      // exactly those.
      OpBuilder pb(b.getContext());
      pb.setInsertionPointToStart(&laggedIf.getThenRegion().front());
      Value waitTarget = arith::IndexCastOp::create(pb, loc, i32, completed);
      func::CallOp::create(pb, loc, waitFn, ValueRange{waitTarget});
      // The rowStart subtraction the matcher verified (`iv - (G-1)`) is the
      // one operand the lag re-bases; everything else in the block moves
      // untouched.
      arith::SubIOp rowSub;
      for (Operation &op : ifOp.getThenRegion().front()) {
        auto sub = dyn_cast<arith::SubIOp>(op);
        if (sub && sub.getLhs() == mInd &&
            foldIndex(sub.getRhs()).value_or(-1) == m.readoutGroup - 1) {
          rowSub = sub;
          break;
        }
      }
      assert(rowSub && "rowStart form validated at match time");
      // The boundary block's own ops -- the read-out split's stores and
      // call -- MOVE into the lagged block (not re-created), so whatever
      // that pass emits per publish rides along; only the rowStart's
      // induction operand is re-based to one batch back.
      Block &from = ifOp.getThenRegion().front();
      Block &to = laggedIf.getThenRegion().front();
      for (Operation &op :
           llvm::make_early_inc_range(from.getOperations())) {
        if (op.hasTrait<OpTrait::IsTerminator>())
          continue;
        op.moveBefore(to.getTerminator());
      }
      rowSub.setOperand(0, mPrev);
      // The boundary block and its guard arithmetic leave; the guard's
      // pieces die with the block unless something else reads them, which
      // the dead-definer walk checks rather than assumes.
      Value oldCondition = ifOp.getCondition();
      ifOp.erase();
      eraseDeadDefiners(oldCondition);
    }
  }

  // ---- after the loop: the tail batch, the barrier, the read-outs ------
  // The emission position differs by form, and the difference is
  // load-bearing: the pipelined form's tail batch carries the peeled
  // epilogue's tile, so it is emitted at the epilogue's engine nest (the
  // position the nest is about to leave) -- after the epilogue pack and
  // its descriptor fill, before the read-out split's tail publish. The
  // serial form has no epilogue, and the position is right after the loop.
  if (m.pipelined)
    b.setInsertionPoint(m.epilogueNest);
  else
    b.setInsertionPointAfter(m.mLoop);
  // The constants below are created AT this position: reusing the boundary
  // math's constants would cross the m-loop's region boundary (they were
  // created inside it), and a use of them out here is exactly the
  // non-dominating operand the verifier rejects.
  Value cZero2 = arith::ConstantIndexOp::create(b, loc, 0);
  Value cOne2 = arith::ConstantIndexOp::create(b, loc, 1);
  // The tail batch is the tiles the loop's own batch boundaries did not
  // reach: floor(upper / batch) * batch .. Mt-1. The SERIAL loop's upper is
  // Mt (every tile passes through the loop); the PIPELINED loop's upper is
  // Mt-1 (the pipeliner peeled the last tile into the epilogue), so the
  // tail here also carries the peeled tile.
  int64_t fullBatches = m.upper / kRoleSubmitBatch;
  int64_t tailCount = m.mt - fullBatches * kRoleSubmitBatch;
  if (tailCount > 0) {
    Value cTail = arith::ConstantIndexOp::create(b, loc, tailCount);
    emitRoleSubmit(b, loc, submitFn, drainFn, descBuf, cTail,
                   arith::ConstantIntOp::create(b, loc, tailCount, 32));
  }
  // THE CLAMPED GROUP (readout on): the last live in-loop group's publish,
  // whose lagged position fell past the loop (`K*G + batch > upper`, K the
  // live group count -- at most one group can fall out, the coverage gate's
  // `G >= batch` is what guarantees it). Its lag target is the first submit
  // boundary after the loop -- this tail submit -- so it publishes here,
  // AFTER the submit and BEFORE the drain, keeping its wait: the vector
  // thread chews this group while the engine finishes the tail batch, which
  // is the same one-batch-lead the in-loop lagged publishes keep.
  int64_t liveGroups =
      m.publishIfs.empty() ? 0 : m.upper / m.readoutGroup;
  bool clampedGroup =
      !m.publishIfs.empty() && liveGroups >= 1 &&
      liveGroups * m.readoutGroup + kRoleSubmitBatch > m.upper;
  if (clampedGroup && tailCount > 0) {
    for (const PublishSite &site : m.publishIfs) {
      Value target = arith::ConstantIntOp::create(
          b, loc, liveGroups * m.readoutGroup, 32);
      func::CallOp::create(b, loc, waitFn, ValueRange{target});
      emitFixedPublish(b, loc, module, site,
                       liveGroups * m.readoutGroup - m.readoutGroup,
                       m.readoutGroup);
    }
  }
  // The exit drain: the kernel does not return until every group's engine
  // work has landed in `ar`, because `ar` is the caller's memory. It is the
  // all-groups proof the tail publish's wait used to stand in for: the tail
  // publish (and, in the serial no-tail-batch case below, the clamped
  // group's publish) sits behind it with no wait of its own.
  func::CallOp::create(b, loc, drainFn, ValueRange{});
  // The clamped group when there is NO tail batch to lag to (the serial
  // form with `Mt % batch == 0`): its lagged position does not exist, so
  // this publish IS the last batch's, and the drain just proved everything
  // retired -- it publishes here with no wait. (There is no tail publish in
  // this case either: `Mt % batch == 0` with the gate's `batch | G` forces
  // `Mt % G == 0`, so the read-out split's tail range is empty.)
  if (clampedGroup && tailCount == 0) {
    for (const PublishSite &site : m.publishIfs)
      emitFixedPublish(b, loc, module, site,
                       liveGroups * m.readoutGroup - m.readoutGroup,
                       m.readoutGroup);
  }
  // The inline read-outs are MOVED, not dropped: each is cloned into a
  // fresh loop after the exit drain (the only producer-side point where
  // every tile's engine work is provably done -- the consumer owns `ar`
  // rows until then), and the originals are erased only once every clone
  // is in place. The clone maps the read-out's ROW -- the same tile-index
  // fact the work function's clone maps, for the same reason -- to the
  // read-out loop's induction, so one clone per read-out covers every row.
  // The pipelined form's PEELED read-out is not cloned: the matcher
  // verified it is the steady read-out on the last row, so the loop's last
  // iteration covers that row and the original is simply erased below.
  if (!m.unpacks.empty()) {
    Value cMt = arith::ConstantIndexOp::create(b, loc, m.mt);
    scf::ForOp readoutLoop =
        scf::ForOp::create(b, loc, cZero2, cMt, cOne2, ValueRange{});
    OpBuilder rb(b.getContext());
    rb.setInsertionPoint(readoutLoop.getBody()->getTerminator());
    for (Operation *unpack : m.unpacks) {
      IRMapping remap;
      remap.map(unpackRow(unpack), readoutLoop.getInductionVar());
      rb.clone(*unpack, remap);
    }
  }

  // The engine loops and the read-outs leave the producer body only now,
  // after everything that cloned from them is in place: the steady nest
  // lives in the work function, the peeled nest was verified equivalent and
  // is covered by the tail descriptor, the inline read-outs live past the
  // exit drain. The bias inits left with the work function too -- erased
  // last because the work function cloned them.
  m.nLoop.erase();
  if (m.pipelined)
    m.epilogueNest.erase();
  for (Operation *unpack : m.unpacks)
    unpack->erase();
  if (m.pipelined && m.epilogueUnpack)
    m.epilogueUnpack->erase();
  for (Operation *init : m.biasInits)
    init->erase();

  // Now nothing references the one-row scratch: the packs were re-created
  // on the rows, the engine loops live in the work function, the dealloc
  // was replaced above.
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
      } else {
      }
    } else {
    }

    if (failed(setHmxManifestTopology(module, topology, verdict.regions)))
      signalPassFailure();
  }
};

} // namespace
