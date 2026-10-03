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

#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxReadoutHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

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

/// Attribute every op to the region root that most tightly encloses it, then
/// reduce the regions to kernel-level facts.
///
/// The nearest-enclosing-root rule is the whole correctness of this function.
/// Taking each region's full subtree instead -- which is the obvious reading of
/// "what is in this loop" -- makes the function body contain every pack and
/// every engine op in the kernel, so `packBesideEngine` is true for all HMX
/// kernels and every mixed kernel comes out `role-split-ok`. That was measured:
/// the verdict did not move when pipeline-depth went 1 -> 2, the very switch
/// that decides whether a pack sits inside the tile loop. Under the nearest-root
/// rule the body holds only the ops written directly in it, and a pack in its
/// own loop stays in its own loop.
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
  //  1. A pack shares a loop with an engine op. That is a per-tile producer
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

struct ThreadRolePartitionPass
    : public mlir::hmx::impl::ThreadRolePartitionBase<ThreadRolePartitionPass> {
  using ThreadRolePartitionBase::ThreadRolePartitionBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HmxDialect, func::FuncDialect, scf::SCFDialect>();
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
    if (failed(setHmxManifestTopology(module, topology, verdict.regions)))
      signalPassFailure();
  }
};

} // namespace
