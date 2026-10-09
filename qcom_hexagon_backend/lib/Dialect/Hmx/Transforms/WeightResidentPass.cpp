//===-- WeightResidentPass.cpp - weights become resident VTCM -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Two sources of an HMX weight, one residency mechanism.
//
//   * A weight that is constant at compile time reaches the tile level as the
//     prepacked constant of `matmul-to-hmx`: one-shot bufferization turns it into
//     a `memref.global` plus a `memref.get_global`, and copy canonicalization
//     forwards that global straight into `hmx.matmul` -- the engine, however,
//     reads its weight from VTCM, so the tile level rejects a DDR operand.
//     This pass gives such an operand a VTCM buffer and records where its
//     contents come from. It deliberately does not emit the copy: the buffer's
//     lowering calls the runtime, which allocates the buffer on the first
//     launch, copies the weight in once, pins it against the per-launch
//     deallocation and returns the same address forever after.
//
//   * A *runtime* weight (a function argument) has no compile-time image, so
//     `matmul-to-hmx` bridges it with a `hmx.pack_weight` loop -- paid on every
//     launch. When the `prepackRuntimeWeights` option is on, the host pre-packer
//     (the launcher) writes that argument's bytes already in crouton order; the
//     kernel then only has to get those bytes into VTCM, which is exactly the
//     runtime's one-copy resident entry -- byte-identical to the constant path's
//     copy, so the same residency mechanism covers both. This pass replaces the
//     pack loop with the resident declaration and publishes the weight's slot and
//     layout on the module so the host packs the same permutation the compiler
//     would have (`hmx.weight_prepack`).
//
//     The option is off by default: dropping the pack is only correct when the
//     caller honours the published prepack contract (the launcher does; a raw
//     caller passing row-major bytes does not), so the default keeps the IR
//     byte-identical to before.
//
//   * An N-slice of a runtime weight: a decode kernel splits N across programs,
//     so the bridge packs one column block of a wider `[K, N]` weight. The
//     view's row stride is the whole N, which is what proves the argument's
//     bytes are the whole weight's bytes -- the resident therefore holds the
//     whole `[K, N]` and each `hmx.matmul` reads its block through a
//     `memref.subview`. A view whose whole shape cannot be pinned (an
//     offset that is not provably tile-aligned, or a non-row-major/derived
//     source) keeps the per-launch bridge rather than guessing.
//
// Residency is declared once, in the module attribute
// `hmx.weight_resident_bytes` (the aggregate of the per-buffer byte counts the
// pass computes -- VTCM buffers only, because that aggregate is the pool
// footprint the other budget readers compare against), and the runtime receives
// the same number on the lowering's call, so the declared footprint and what the
// device reserves cannot drift apart.
//
// The two static VTCM budgets are checked by two different passes, at the two
// points where the information exists, and they are not the same question:
//
//   * `matmul-to-hmx` admits one contraction. Its budget is that contraction's
//     own working set plus the crouton arrays earlier attributions in the same
//     function committed. It cannot include a resident total, because this pass
//     runs after it -- it needs the `hmx.matmul` that pass creates -- so the
//     ordering makes that combination impossible rather than merely late.
//   * This pass creates a buffer that occupies VTCM for the whole kernel, so it
//     checks the persistent total here, before it commits. What happens on a
//     refusal depends on which of the two sources above it is, because only one
//     of them has somewhere else to go:
//
//       - A runtime weight that does not fit the pool is placed in the
//         permanent **DDR mirror** instead (`hmx.weight_resident`'s `location`
//         key): the host's pre-pack image is copied into a process-lifetime
//         DDR buffer once, and the pack bridge becomes one contiguous fetch of
//         the block the engine is about to read -- no VTCM, no device-side
//         permutation, and the capacity number picks the placement rather than
//         deciding whether the weight is resident at all. A weight whose
//         *view* cannot be pinned in the first place (a non-pinnable offset,
//         a K-block source) still keeps the per-launch bridge: nothing about
//         its bytes is proven, so there is no image to mirror.
//       - A constant weight has no such route: the resident VTCM buffer is the
//         only form its operand can legally take, because `hmx.mma` requires
//         its weight in VTCM and the operand it replaces is a DDR
//         `memref.get_global`. Refusing to pin it is reported as an error
//         instead, because the alternative is a module that fails verification
//         later with a message about memory space rather than about the pool.
//
//     In neither case is the contraction refused: `matmul-to-hmx` admitted it
//     before this pass ran, and nothing here revisits that decision.
//
// An earlier design had `matmul-to-hmx` read this attribute too and called it
// "their single source of truth". That read was dead on the production path, so
// the guarantee it was cited for did not exist; and the weight's crouton array
// is an `hmx.alloc_crouton` at that point, so it would also have counted the
// same weight twice once the attribute did exist.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDType.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxTarget.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

// The one VTCM byte ledger: this pass's transient read goes through it rather
// than through a private copy of hmx-partition's walk (see the header for why).
#include "HmxVtcmLedger.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>

#define DEBUG_TYPE "weight-resident"

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_WEIGHTRESIDENT
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

/// Per-buffer residency record on the `hexagonmem.alloc`: where the contents
/// come from and how many bytes. The constant path names a global symbol; the
/// runtime path names the function argument slot. The lowering reads this
/// dictionary; the module attribute below is its aggregate for the static
/// budget readers.
constexpr StringLiteral kResidentAttr = kHmxWeightResidentAttr;
constexpr const char *kResidentKeyGlobal = "global";
constexpr const char *kResidentKeyAddress = "address";
constexpr const char *kResidentKeyBytes = "bytes";

/// Aggregate of every resident buffer in the module, in bytes.
constexpr const char *kResidentBytesAttr = "hmx.weight_resident_bytes";

/// The descriptor's placement fact, read fail-closed: absent means VTCM (the
/// only placement before DDR residency existed, and what keeps a VTCM
/// resident's IR byte-identical), `ddr` means the permanent DDR mirror, and
/// anything else is a producer/consumer disagreement rather than a placement.
/// Only the VTCM total is counted in `hmx.weight_resident_bytes`: that
/// aggregate is the persistent footprint of the *pool* the budget readers
/// check, and a DDR mirror does not come out of it.
static FailureOr<bool> residentIsDdr(DictionaryAttr resident,
                                     Operation *anchor) {
  auto location = resident.getAs<StringAttr>(kHmxWeightResidentLocationKey);
  if (!location)
    return false;
  if (location.getValue() == kHmxWeightResidentLocationDdr)
    return true;
  if (location.getValue() == kHmxWeightResidentLocationVtcm)
    return false;
  return anchor->emitError("resident weight location must be \"vtcm\" or "
                           "\"ddr\"");
}

/// JSON array of the runtime weights the host must pre-pack: each entry names
/// the function, the argument slot, the logical shape and the crouton shape.
constexpr StringLiteral kPrepackAttr = kHmxWeightPrepackAttr;

/// The module attribute carrying the permutation the host must apply.
constexpr StringLiteral kPrepackLayoutAttr = kHmxWeightPrepackLayoutAttr;

/// The crouton permutation the compiler applies, as a JSON coefficient map. A
/// weight grid is [Nt, Kt, 16, 32, 2] with logical [N, K], so a physical index
/// (d0=n_tile, d1=k_tile, d2=j, d3=c, d4=h) maps to logical
/// (tile*d1 + half*d2 + d4, tile*d0 + d3). Published so the host packer derives
/// the permutation from compiler metadata instead of re-deriving it, and can
/// fail loudly if the two ever disagree; built from the one tile-edge constant
/// so it cannot drift from the layout.
static std::string prepackLayoutJson() {
  return std::string("{\"ndims\":5,\"results\":[[[1,") +
         std::to_string(hmx::layout::kTileEdge) + "],[2," +
         std::to_string(hmx::layout::kCroutonHalf) + "],[4,1]],[[0," +
         std::to_string(hmx::layout::kTileEdge) + "],[3,1]]]}";
}

/// The prepacked constant a value is loaded from, or null. Only the direct
/// `memref.get_global` form is handled: that is what bufferization produces for
/// the constant fast path of `matmul-to-hmx`.
memref::GlobalOp prepackedSource(Value v) {
  auto getGlobal = v.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return {};
  return SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      getGlobal, getGlobal.getNameAttr());
}

static int64_t byteSize(MemRefType type) {
  return type.getNumElements() * (type.getElementTypeBitWidth() / 8);
}

static std::optional<int64_t> checkedByteSize(MemRefType type) {
  if (!type.hasStaticShape())
    return std::nullopt;
  Type element = type.getElementType();
  if (!element.isIntOrFloat())
    return std::nullopt;
  int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
  int64_t elements = type.getNumElements();
  if (elementBytes <= 0 || element.getIntOrFloatBitWidth() % 8 != 0 ||
      elements <= 0 ||
      elements > std::numeric_limits<int64_t>::max() / elementBytes)
    return std::nullopt;
  int64_t bytes = elements * elementBytes;
  if (bytes > std::numeric_limits<uint32_t>::max())
    return std::nullopt;
  return bytes;
}

/// The weight record's own fields: the identity key, the runtime key
/// kind, the source descriptors, and the content/address-reuse claims
/// that depend on the source.  The shared field skeleton (and the
/// buffer-safety checks this pass audits with) lives in
/// `HmxResidentContract.h`; only what is unique to a weight stays here.
static DictionaryAttr
makeWeightProvenance(MLIRContext *context, StringRef principal,
                     StringRef function, StringRef site, int64_t slot,
                     int64_t bytes, int64_t alignment, uint64_t identityKey,
                     StringRef runtimeKeyKind, StringRef source,
                     StringRef contentStatus, StringRef contentIdentity,
                     DictionaryAttr sourceView = {}) {
  NamedAttrList fields;
  appendResidentProvenanceCore(fields, context, "weight",
                               "weight-resident", principal, function, site,
                               slot, bytes, alignment, contentStatus);
  fields.append("identity_key",
                IntegerAttr::get(IntegerType::get(context, 64), identityKey));
  fields.append("runtime_key_kind", StringAttr::get(context, runtimeKeyKind));
  fields.append("source", StringAttr::get(context, source));
  if (sourceView)
    fields.append("source_view", sourceView);
  fields.append("content_identity", StringAttr::get(context, contentIdentity));
  fields.append("address_reuse_status",
                StringAttr::get(context, runtimeKeyKind == "argument-address"
                                             ? kHmxResidentNotProven
                                             : "immutable-source"));
  return DictionaryAttr::get(context, fields);
}

/// Validate the exact source type of a compile-time weight.  The resident
/// buffer is a copy, so a shape or element-type mismatch would make the
/// device-side descriptor disagree with the source even when the byte count
/// happened to match.
static bool validGlobalWeightSource(memref::GlobalOp source,
                                    MemRefType residentType) {
  if (!source || !source.getConstant())
    return false;
  MemRefType sourceType = source.getType();
  return sourceType.getShape() == residentType.getShape() &&
         sourceType.getElementType() == residentType.getElementType() &&
         dtype::isCroutonElement(sourceType.getElementType()) &&
         sourceType.getLayout().isIdentity();
}

struct WeightPack {
  memref::AllocOp array;
  SmallVector<PackWeightOp> packs;
  SmallVector<scf::ForOp> loops;
};

/// Every `hmx.pack_weight` that writes `array` as its destination.
static SmallVector<PackWeightOp> packWeightWriters(Value array) {
  SmallVector<PackWeightOp> packs;
  for (OpOperand &u : array.getUses())
    if (auto p = dyn_cast<PackWeightOp>(u.getOwner()))
      if (p.getDst() == array)
        packs.push_back(p);
  return packs;
}

/// True when `loop`'s body is the weight bridge and nothing else: index
/// arithmetic and pack writes. Erasing such a loop removes exactly the bridge.
static bool isWeightBridgeLoop(scf::ForOp loop) {
  for (Operation &inner : loop.getBody()->without_terminator())
    if (!isa<arith::ConstantIndexOp, arith::DivUIOp, arith::RemUIOp,
             arith::AddIOp, arith::MulIOp, arith::IndexCastOp, PackWeightOp>(
            inner))
      return false;
  return true;
}

/// The runtime-weight bridge behind `v`, or nullopt. Both bufferization shapes
/// are handled: the pack loop writes the allocation in place (the common
/// canonicalized form), or the allocation is the loop's carried init.
static std::optional<WeightPack> findWeightPack(Value v) {
  WeightPack pack;
  if (auto alloc = v.getDefiningOp<memref::AllocOp>()) {
    pack.array = alloc;
    pack.packs = packWeightWriters(v);
  } else if (auto loop = v.getDefiningOp<scf::ForOp>()) {
    if (loop.getNumResults() != 1 || loop.getResult(0) != v ||
        loop.getInitArgs().empty())
      return std::nullopt;
    pack.array = loop.getInitArgs()[0].getDefiningOp<memref::AllocOp>();
    if (!pack.array)
      return std::nullopt;
    for (Operation &inner : loop.getBody()->without_terminator())
      if (auto p = dyn_cast<PackWeightOp>(inner))
        if (p.getDst() == loop.getRegionIterArg(0))
          pack.packs.push_back(p);
    if (!isWeightBridgeLoop(loop))
      return std::nullopt;
    pack.loops.push_back(loop);
  } else {
    return std::nullopt;
  }
  if (!pack.array || pack.packs.empty())
    return std::nullopt;
  // The alloc form keeps its writers inside the same bridge loop; record it so
  // it can be erased with the bridge.
  if (pack.loops.empty()) {
    llvm::DenseSet<scf::ForOp> seen;
    for (PackWeightOp p : pack.packs) {
      scf::ForOp loop = p->getParentOfType<scf::ForOp>();
      if (!loop || !isWeightBridgeLoop(loop))
        return std::nullopt;
      if (seen.insert(loop).second)
        pack.loops.push_back(loop);
    }
  }
  return pack;
}

static std::string jsonArray(ArrayRef<int64_t> values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      out += ",";
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

/// Append one prepack entry (a JSON object) to the module's JSON array.
static void appendPrepackEntry(ModuleOp module, StringRef entry) {
  auto existing = module->getAttrOfType<StringAttr>(kPrepackAttr);
  if (!existing) {
    module->setAttr(kPrepackAttr,
                    StringAttr::get(module.getContext(), "[" + entry.str() + "]"));
    return;
  }
  std::string merged = existing.getValue().str();
  // Strip the closing bracket, add the separator only when non-empty, re-close.
  if (!merged.empty() && merged.back() == ']')
    merged.pop_back();
  if (merged.size() > 1)
    merged += ",";
  merged += entry.str();
  merged += "]";
  module->setAttr(kPrepackAttr, StringAttr::get(module.getContext(), merged));
}

/// The module's declared resident footprint, read under the same rules the
/// writer applies: absent means nothing is resident yet (zero), present means a
/// non-negative i64. A declared value that is negative or not an i64 is
/// *rejected* rather than read as a smaller number, so a guard built on this can
/// never be satisfied by a wrapped or corrupt aggregate. Shared with
/// `addCheckedResidentBytes` on purpose -- the guard and the writer must not be
/// able to disagree about what the aggregate means.
static FailureOr<int64_t> residentBytesDeclared(ModuleOp module) {
  auto declared = module->getAttrOfType<IntegerAttr>(kResidentBytesAttr);
  if (!declared)
    return int64_t{0};
  if (!declared.getType().isInteger(64) || declared.getInt() < 0)
    return module.emitError("resident byte aggregate is missing or malformed");
  return declared.getInt();
}

/// `a + b`, rejecting an overflow rather than wrapping into a smaller number.
/// The resident footprint is an exact requested-byte descriptor, so every sum
/// that feeds the budget is checked: a wrapped total is a total that would
/// *admit* something the pool cannot hold.
static FailureOr<int64_t> checkedResidentAdd(ModuleOp module, int64_t a,
                                             int64_t b) {
  if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b)
    return module.emitError("VTCM byte total overflows int64");
  return a + b;
}

static void addCompatibilityResidentBytes(ModuleOp module, int64_t bytes) {
  int64_t total = bytes;
  if (auto existing = module->getAttrOfType<IntegerAttr>(kResidentBytesAttr))
    total += existing.getInt();
  module->setAttr(
      kResidentBytesAttr,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), total));
}

/// Add `bytes` to the module's resident footprint.  The aggregate is an exact
/// requested-byte descriptor, not a content hash; malformed/overflowing state
/// is rejected instead of wrapping into a smaller number.
static LogicalResult addCheckedResidentBytes(ModuleOp module, int64_t bytes) {
  if (bytes < 0)
    return module.emitError("resident byte aggregate cannot be negative");
  FailureOr<int64_t> existing = residentBytesDeclared(module);
  if (failed(existing))
    return failure();
  FailureOr<int64_t> total = checkedResidentAdd(module, bytes, *existing);
  if (failed(total))
    return failure();
  module->setAttr(
      kResidentBytesAttr,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), *total));
  return success();
}

static LogicalResult verifyResidentByteAggregate(ModuleOp module) {
  auto declared = module->getAttrOfType<IntegerAttr>(kResidentBytesAttr);
  if (!declared || !declared.getType().isInteger(64) || declared.getInt() < 0)
    return module.emitError("resident byte aggregate is missing or malformed");
  int64_t sum = 0;
  LogicalResult result = success();
  module.walk([&](Operation *operation) {
    auto resident = operation->getAttrOfType<DictionaryAttr>(kResidentAttr);
    if (!resident)
      return;
    // A DDR mirror is resident but not *VTCM*-resident, so it is deliberately
    // not part of this aggregate -- the sum is the pool footprint the budget
    // readers compare against, and adding DDR bytes to it would make every
    // later admission refuse.
    FailureOr<bool> ddr = residentIsDdr(resident, operation);
    if (failed(ddr)) {
      result = failure();
      return;
    }
    if (*ddr)
      return;
    auto bytes = resident.getAs<IntegerAttr>(kResidentKeyBytes);
    if (!bytes || bytes.getInt() < 0 ||
        sum > std::numeric_limits<int64_t>::max() - bytes.getInt()) {
      result = module.emitError("resident byte aggregate does not close");
      return;
    }
    sum += bytes.getInt();
  });
  if (failed(result))
    return failure();
  if (sum != declared.getInt())
    return module.emitError(
        "resident byte aggregate disagrees with resident descriptors");
  return success();
}

/// The answer to the one question this pass has to ask before it commits a
/// resident buffer: does the persistent footprint still fit the pool?
///
/// The transient half comes from the shared VTCM ledger (`hmx::vtcm`), computed
/// fresh at each admission rather than cached, because the answer moves under
/// the loop: a runtime weight that becomes resident has its pack array erased,
/// so a cached total would keep charging for a buffer that is gone and would
/// refuse later weights for a transient that no longer exists.
///
/// `hexagonmem` VTCM is a single pool of `HmxTarget::defaultVtcmBudget` bytes,
/// shared by the crouton arrays, the accumulators and every resident weight, and
/// the device's `requireAllocationResult` is fail-closed -- an allocation that
/// does not fit aborts the whole PD at load time. So a residency whose overrun
/// is visible right here used to compile cleanly and abort the process on the
/// device. Deciding here is what turns that into a compile-time outcome.
///
/// The total is the three sets of bytes that are all live at once once the
/// buffer exists: the function's transient VTCM allocations, the resident
/// footprint committed so far, and the buffer being committed.
struct ResidentVtcmAdmission {
  int64_t transient = 0; ///< VTCM bytes the function's own allocations hold.
  int64_t resident = 0;  ///< Resident bytes committed before this buffer.
  int64_t requested = 0; ///< The buffer about to be committed.
  int64_t total = 0;     ///< transient + resident + requested.
  int64_t budget = 0;    ///< `HmxTarget::defaultVtcmBudget`.
  bool admitted = false; ///< `total < budget`; see the boundary note below.
};

/// The only place the resident capacity question is answered. Both creation
/// sites ask it, so the boundary, the budget and the arithmetic cannot differ
/// between a constant and a runtime weight.
///
/// The budget is `HmxTarget::defaultVtcmBudget`, the same constant the other
/// static readers fall back to (`HmxPartitionPass` uses it whenever its own
/// `vtcm-budget` option is unset). This pass has no budget option to narrow it
/// with, and adding one is deliberately out of scope: a third way to spell the
/// pool size is one more thing that can disagree with the device.
///
/// Two properties are the reason the guard lives here and not upstream:
///
///   * It is *conservative*, on purpose. The transient walk counts a runtime
///     weight's own pack array, which dies the moment the resident copy is what
///     the engine reads. So this can decline a residency that would in fact
///     have fitted. That is the safe direction to be wrong in: the cost is
///     losing the prepack optimisation for that one weight, never losing HMX.
///   * It can only decline *residency*. The contraction was admitted by
///     `matmul-to-hmx` before this pass ran, and nothing here revisits that
///     decision, so a residency refusal can never travel backwards and refuse a
///     contraction that was already accepted.
static FailureOr<ResidentVtcmAdmission>
admitResidentVtcm(func::FuncOp func, ModuleOp module, int64_t addedBytes,
                  int64_t bytes) {
  // A negative request is malformed rather than small, and it would make every
  // sum below look like it fits, so it is rejected before it is used.
  if (bytes < 0)
    return module.emitError("resident weight byte count is negative");
  ResidentVtcmAdmission admission;
  admission.budget = HmxTarget::defaultVtcmBudget;
  admission.requested = bytes;
  admission.transient =
      hmx::vtcm::transientBytes(func, hmx::vtcm::Population::Memref);
  // `hmx.weight_resident_bytes` holds what *earlier* functions committed -- this
  // pass is a per-function pass that writes the attribute only when it ends --
  // so it and `addedBytes` are disjoint and sum to the footprint so far.
  FailureOr<int64_t> declared = residentBytesDeclared(module);
  if (failed(declared))
    return failure();
  FailureOr<int64_t> resident = checkedResidentAdd(module, *declared, addedBytes);
  if (failed(resident))
    return failure();
  admission.resident = *resident;
  FailureOr<int64_t> withTransient =
      checkedResidentAdd(module, admission.resident, admission.transient);
  if (failed(withTransient))
    return failure();
  FailureOr<int64_t> total =
      checkedResidentAdd(module, *withTransient, admission.requested);
  if (failed(total))
    return failure();
  admission.total = *total;
  // Strict `<`, the same boundary `HmxTarget::planBridge` refuses at
  // (`footprint >= room` refuses): the budget is a pool size, so a total exactly
  // equal to it leaves no room for the allocation's own 128 B alignment.
  admission.admitted = admission.total < admission.budget;
  return admission;
}

/// The placement decision, spelled once so a reader can tell a weight that ran
/// out of pool from one that was never eligible. `slot` is the residency key
/// the descriptors use, so the message points at the thing to look for in the
/// IR. All four numbers stay in it: they are what says *why* the buffer could
/// not be VTCM, and the last term (the buffer being placed) is the one a reader
/// cannot reconstruct from the manifest.
static InFlightDiagnostic describeVtcmDdrPlacement(MatmulOp op, int64_t slot,
                                                   const ResidentVtcmAdmission &a) {
  InFlightDiagnostic diag = op.emitRemark();
  diag << "resident weight for argument slot " << slot
       << " does not fit the persistent VTCM pool: " << a.requested
       << " bytes would make the persistent VTCM total " << a.total
       << " bytes, over the " << a.budget << " byte budget (" << "transient "
       << a.transient << " + resident " << a.resident << " + this buffer); "
       << "placed in the DDR permanent region instead and its pack bridge "
       << "becomes a contiguous fetch";
  return diag;
}

/// Every way the runtime-weight path can decline a weight, reported where the
/// decline happens instead of being dropped. The pass had two silent `continue`s
/// in this path (the view matcher's, and the affine-proof one inside it), which
/// is how a bench could pay a per-launch pack with no diagnostic anywhere in the
/// pipeline to explain it -- see docs/hmx/pack-redundancy-fix-plan-2026-10-09.md
/// §2.2. A remark rather than an error even under the strict resident contract:
/// a weight this pass does not model is a correct kernel with an unclaimed
/// optimisation, not a malformed module. `slot` is unknown before the argument
/// has resolved, so the message drops the slot rather than naming the wrong one.
static InFlightDiagnostic describeWeightDecline(MatmulOp op,
                                                std::optional<int64_t> slot,
                                                StringRef reason) {
  InFlightDiagnostic diag = op.emitRemark();
  diag << "resident weight";
  if (slot)
    diag << " for argument slot " << *slot;
  diag << " declined: " << reason << "; the weight keeps its per-launch pack "
       << "bridge";
  return diag;
}

/// Follow a chain of layout-only views to the underlying function argument.
/// A view qualifies only when it provably covers the whole argument: offset 0
/// and dense row-major strides for its static shape, so the host pre-pack of
/// the whole argument feeds exactly what the bridge would read. The source may
/// be:
///
///   * a ranked memref -- it must have the same static shape and element type
///     as the view, so the view is the whole argument; or
///   * an unranked memref (`memref<*xf16>`) -- how a real Triton kernel receives
///     a weight. There is no static source shape to compare, so the view's
///     static shape *is* the published contract shape. This is only sound
///     because offset 0 + dense strides pin the view to the buffer base; a view
///     that merely asserted a shape over a larger buffer would still be rejected
///     by the stride check unless it covered the buffer densely.
///
/// Anything else returns null and keeps the old bridge: guessing would silently
/// compute on the wrong data. In particular a nonzero *or dynamic* offset is a
/// real slice, so it keeps the per-launch bridge (see the note next to the
/// offset check). Real kernels wrap pack sources in `reinterpret_cast` with a
/// dense-equivalent strided layout, which the bare-BlockArgument check below
/// would otherwise miss entirely.
static BlockArgument underlyingDenseArgument(Value v) {
  for (int depth = 0; depth < 8; ++depth) {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return arg;
    auto reinterpret = v.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpret)
      return {};
    auto viewType = dyn_cast<MemRefType>(v.getType());
    if (!viewType || !viewType.hasStaticShape())
      return {};
    Type srcType = reinterpret.getSource().getType();
    if (auto baseType = dyn_cast<MemRefType>(srcType)) {
      if (!baseType.hasStaticShape() ||
          baseType.getElementType() != viewType.getElementType() ||
          baseType.getShape() != viewType.getShape())
        return {};
    } else if (auto baseType = dyn_cast<UnrankedMemRefType>(srcType)) {
      if (baseType.getElementType() != viewType.getElementType())
        return {};
    } else {
      return {};
    }
    // The view must start at the buffer base. A nonzero *static* offset is a
    // real slice. A *dynamic* offset is the reason P2 does not fire on a real
    // decode kernel: the source view is `offset: [%n]` (the program-id N
    // offset), so the argument's bytes are not the weight's bytes. Pre-packing
    // the whole argument there would silently compute on the wrong data, so it
    // must keep the old bridge. Lifting that is a contract decision, not a
    // local one; the candidates are to hand the kernel the whole weight (no
    // N-offset view), to loop over N inside the kernel, or to carry the offset
    // in the host prepack contract. None is chosen here. Check both the op
    // operands and the view layout so an omitted-operand form cannot slip
    // through.
    for (int64_t off : reinterpret.getStaticOffsets())
      if (off != 0)
        return {};
    if (auto strided = dyn_cast<StridedLayoutAttr>(viewType.getLayout()))
      if (strided.getOffset() != 0)
        return {};
    // Dense row-major strides for the (static) shape.
    int64_t rank = viewType.getRank();
    SmallVector<int64_t> dense(rank, 0);
    int64_t stride = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
      dense[i] = stride;
      stride *= viewType.getDimSize(i);
    }
    if (auto strided = dyn_cast<StridedLayoutAttr>(viewType.getLayout())) {
      auto strides = strided.getStrides();
      for (int64_t i = 0; i < rank; ++i) {
        if (strides[i] == ShapedType::kDynamic || strides[i] != dense[i])
          return {};
      }
    } else if (!viewType.getLayout().isIdentity()) {
      return {};
    }
    v = reinterpret.getSource();
  }
  return {};
}

/// The host contract addresses tensor arguments by their tensor ordinal, not by
/// the raw MLIR argument number (scalars may precede tensors). Keep that ABI in
/// one place so compiler and launcher cannot silently drift.
static std::optional<int64_t> tensorArgumentSlot(func::FuncOp func,
                                                 BlockArgument target) {
  int64_t slot = 0;
  for (BlockArgument arg : func.getArguments()) {
    if (!isa<RankedTensorType, MemRefType, UnrankedMemRefType>(arg.getType()))
      continue;
    if (arg == target)
      return slot;
    ++slot;
  }
  return std::nullopt;
}

/// True when `v` is provably a multiple of `factor`, for the small affine forms
/// an N-block offset takes (`pid * BN`, sums of such, and the induction variable
/// of a static-step loop). The subview that reads one N block of the resident
/// weight is indexed in croutons, so the block offset has to land on a tile
/// edge; anything not provably a multiple keeps the per-launch bridge rather
/// than truncating a division.
///
/// This is a *proof*, not a pattern match: `false` means "not established" and
/// only ever costs the optimisation, while `true` has to be true for every
/// value the offset can take. That asymmetry is why the `mul` case is an `||`
/// (either factor already carries the product) while the `add` case and the
/// loop case below are `&&`.
///
/// Loop case (A2): `scf.for %iv = lb to ub step st` runs `%iv = lb + k * st`
/// for every `k >= 0`, so every value it takes is a multiple of `factor` exactly
/// when both `lb` and `st` are -- sign is irrelevant, because a multiple of
/// `factor` stays one under negation. That covers the form a real kernel's N
/// block actually takes (the inner loop's own induction variable, usually
/// behind an `index_cast`), which the affine forms above never see. The
/// companion guard in the caller (`offset` provably a multiple of `n` is a K
/// offset, so it is refused) keeps the same property: it also needs both terms,
/// so a loop whose step is smaller than N cannot prove itself a row offset and
/// is still accepted as the column offset it is.
static bool isMultipleOf(Value v, int64_t factor, int depth = 0) {
  if (depth > 8)
    return false;
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() % factor == 0;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt() % factor == 0;
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return isMultipleOf(cast.getIn(), factor, depth + 1);
  if (auto cast = v.getDefiningOp<arith::IndexCastUIOp>())
    return isMultipleOf(cast.getIn(), factor, depth + 1);
  if (auto add = v.getDefiningOp<arith::AddIOp>())
    return isMultipleOf(add.getLhs(), factor, depth + 1) &&
           isMultipleOf(add.getRhs(), factor, depth + 1);
  if (auto mul = v.getDefiningOp<arith::MulIOp>())
    return isMultipleOf(mul.getLhs(), factor, depth + 1) ||
           isMultipleOf(mul.getRhs(), factor, depth + 1);
  if (auto arg = dyn_cast<BlockArgument>(v))
    if (arg.getArgNumber() == 0)
      if (auto loop =
              dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp()))
        return isMultipleOf(loop.getLowerBound(), factor, depth + 1) &&
               isMultipleOf(loop.getStep(), factor, depth + 1);
  return false;
}

/// An N-slice of a wider runtime weight (B2): the pack bridge covers one N block
/// of a `[K, N]` matrix. The whole N is the view's row stride, so the argument's
/// bytes *are* the whole weight's bytes and one resident copy serves every
/// program.
struct WeightSlice {
  BlockArgument arg;
  int64_t n = 0;          // whole logical columns (N)
  Value dynamicOffset;    // N block offset in elements, when dynamic
  std::optional<int64_t> staticOffset; // N block offset in elements, when static
  // The entry-argument view this slice was proven against. It is the view's own
  // geometry, not the bridge source's: a tail guard merges its arms into a
  // result type whose strides and offset are both dynamic, while the contract
  // describes the concrete `[K, N]` view behind it.
  memref::ReinterpretCastOp view;
};

/// Capture the semantic source view that authorizes an address-keyed resident.
/// The pack bridge is erased after the first lowering, so later IR cannot
/// recover the original whole-object or static-N-slice descriptor (and an
/// unranked argument type alone says nothing about its original view).
/// Persist that fact next to the resident instead of treating every later
/// entry-argument pointer as interchangeable.
static std::optional<DictionaryAttr>
makeRuntimeSourceView(MLIRContext *context, Value source,
                      BlockArgument argument, const WeightSlice *slice) {
  // A tail guard's merged result type is `strided<[?,?], offset:?>` -- both
  // dynamic, because two arms of different provenance meet there. The contract
  // describes the view behind it, which is where the whole-N row stride and the
  // concrete offset live, so take the geometry from there when there is one.
  Value geometry =
      slice && slice->view ? slice->view->getResult(0) : source;
  auto type = dyn_cast<MemRefType>(geometry.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
      !dtype::isAdmittedFloat(type.getElementType()))
    return std::nullopt;

  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  SmallVector<int64_t> strides;
  if (auto strided = dyn_cast<StridedLayoutAttr>(type.getLayout())) {
    strides.assign(strided.getStrides().begin(), strided.getStrides().end());
  } else if (type.getLayout().isIdentity()) {
    strides.resize(type.getRank());
    int64_t stride = 1;
    for (int64_t i = type.getRank() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= type.getDimSize(i);
    }
  } else {
    return std::nullopt;
  }

  StringRef argumentKind =
      isa<UnrankedMemRefType>(argument.getType())
          ? "unranked-memref"
          : (isa<MemRefType>(argument.getType()) ? "ranked-memref"
                                                 : "unknown-memref");
  StringRef viewKind = "whole-argument";
  int64_t offset = 0;
  int64_t wholeN = shape[1];
  if (slice) {
    if (!slice->staticOffset)
      return std::nullopt;
    viewKind = "static-n-slice";
    offset = *slice->staticOffset;
    wholeN = slice->n;
  }

  Builder builder(context);
  NamedAttrList fields;
  fields.append("argument_kind", builder.getStringAttr(argumentKind));
  fields.append("offset", builder.getI64IntegerAttr(offset));
  fields.append("shape", builder.getI64ArrayAttr(shape));
  fields.append("strides", builder.getI64ArrayAttr(strides));
  fields.append("view_kind", builder.getStringAttr(viewKind));
  fields.append("whole_n", builder.getI64IntegerAttr(wholeN));
  return DictionaryAttr::get(context, fields);
}

/// Validate a descriptor left by an earlier lowering. This is intentionally a
/// closed schema: adding an unrecognized field would otherwise let two
/// spellings of the same contract evade same-slot comparison.
static LogicalResult validateRuntimeSourceView(DictionaryAttr sourceView,
                                               MemRefType residentType,
                                               BlockArgument argument,
                                               Operation *anchor) {
  auto reject = [&]() {
    return anchor->emitError(
        "strict resident weight source-view descriptor does not match its "
        "source");
  };
  if (!sourceView || sourceView.size() != 6)
    return reject();

  auto viewKind = sourceView.getAs<StringAttr>("view_kind");
  auto argumentKind = sourceView.getAs<StringAttr>("argument_kind");
  auto offsetAttr = sourceView.getAs<IntegerAttr>("offset");
  auto wholeNAtt = sourceView.getAs<IntegerAttr>("whole_n");
  auto shapeAttr = sourceView.getAs<ArrayAttr>("shape");
  auto stridesAttr = sourceView.getAs<ArrayAttr>("strides");
  if (!viewKind || !argumentKind || !offsetAttr || !wholeNAtt || !shapeAttr ||
      !stridesAttr || shapeAttr.size() != 2 || stridesAttr.size() != 2)
    return reject();
  auto shape0 = dyn_cast<IntegerAttr>(shapeAttr.getValue()[0]);
  auto shape1 = dyn_cast<IntegerAttr>(shapeAttr.getValue()[1]);
  auto stride0 = dyn_cast<IntegerAttr>(stridesAttr.getValue()[0]);
  auto stride1 = dyn_cast<IntegerAttr>(stridesAttr.getValue()[1]);
  if (!shape0 || !shape1 || !stride0 || !stride1 ||
      !offsetAttr.getType().isInteger(64) ||
      !wholeNAtt.getType().isInteger(64) || !shape0.getType().isInteger(64) ||
      !shape1.getType().isInteger(64) || !stride0.getType().isInteger(64) ||
      !stride1.getType().isInteger(64))
    return reject();

  int64_t k = shape0.getInt();
  int64_t n = shape1.getInt();
  int64_t rowStride = stride0.getInt();
  int64_t columnStride = stride1.getInt();
  int64_t offset = offsetAttr.getInt();
  int64_t wholeN = wholeNAtt.getInt();
  if (residentType.getRank() != 5 || residentType.getDimSize(2) != 16 ||
      residentType.getDimSize(3) != 32 || residentType.getDimSize(4) != 2)
    return reject();
  int64_t residentK = hmx::weightKTiles(residentType) * hmx::layout::kTileEdge;
  int64_t residentN = hmx::weightNTiles(residentType) * hmx::layout::kTileEdge;
  if (k <= 0 || n <= 0 || rowStride <= 0 || columnStride != 1 || offset < 0 ||
      wholeN <= 0 || wholeN % hmx::layout::kTileEdge != 0 || k != residentK)
    return reject();

  StringRef expectedArgumentKind =
      isa<UnrankedMemRefType>(argument.getType())
          ? "unranked-memref"
          : (isa<MemRefType>(argument.getType()) ? "ranked-memref"
                                                 : "unknown-memref");
  if (argumentKind.getValue() != expectedArgumentKind)
    return reject();
  if (auto ranked = dyn_cast<MemRefType>(argument.getType())) {
    if (!ranked.hasStaticShape() || ranked.getRank() != 2 ||
        !dtype::isAdmittedFloat(ranked.getElementType()) ||
        !ranked.getLayout().isIdentity() ||
        ranked.getShape() != ArrayRef<int64_t>{k, wholeN})
      return reject();
  } else if (auto unranked = dyn_cast<UnrankedMemRefType>(argument.getType())) {
    if (!dtype::isAdmittedFloat(unranked.getElementType()))
      return reject();
  } else {
    return reject();
  }

  StringRef kind = viewKind.getValue();
  if (kind == "whole-argument") {
    if (offset != 0 || n != residentN || n != wholeN || rowStride != wholeN)
      return reject();
  } else if (kind == "static-n-slice") {
    if (isa<UnrankedMemRefType>(argument.getType()) || residentN != wholeN ||
        n >= wholeN || n % hmx::layout::kTileEdge != 0 || wholeN % n != 0 ||
        rowStride != wholeN || offset % hmx::layout::kTileEdge != 0 ||
        offset > wholeN - n)
      return reject();
  } else {
    return reject();
  }
  return success();
}

/// Peel the `memref.cast`s a tail guard inserts to merge two differently-typed
/// arms into the one result type `scf.if` requires.
static Value peelViewCasts(Value v) {
  for (int depth = 0; depth < 4; ++depth) {
    auto cast = v.getDefiningOp<memref::CastOp>();
    if (!cast)
      break;
    v = cast.getSource();
  }
  return v;
}

/// The buffer a `memref.copy` writes into: the value itself, or the view of a
/// buffer it is a subview of.
static Value copyTargetBuffer(Value target) {
  if (auto sub = target.getDefiningOp<memref::SubViewOp>())
    return sub.getSource();
  return target;
}

/// True when `sub` selects a *prefix* of `view`: offset 0 on every dimension,
/// unit strides, every row, and a column extent the view itself bounds (a
/// dynamic extent is admitted because it is the guard's own clamp of
/// `min(n0 + BN, N) - n0`, which is bounded by the view by construction, while
/// a static one is range-checked here).
///
/// "Prefix" is the load-bearing half. The arm a guard like this replaces is the
/// arm that reads real bytes for exactly these columns and something else for
/// the rest; see the acceptance note on `underlyingSliceArgument` for why the
/// rest may be anything at all.
static bool isViewPrefix(memref::SubViewOp sub, memref::ReinterpretCastOp view) {
  auto viewType = dyn_cast<MemRefType>(view.getResult().getType());
  if (!viewType || !viewType.hasStaticShape())
    return false;
  int64_t rank = viewType.getRank();
  if (sub.getStaticOffsets().size() != static_cast<size_t>(rank) ||
      sub.getStaticStrides().size() != static_cast<size_t>(rank) ||
      sub.getStaticSizes().size() != static_cast<size_t>(rank))
    return false;
  for (int64_t offset : sub.getStaticOffsets())
    if (offset != 0) // a dynamic slot prints as kDynamic, which is not 0
      return false;
  for (int64_t stride : sub.getStaticStrides())
    if (stride != 1)
      return false;
  // Every row of the view is in bounds no matter where the N block starts, so
  // a gather that skips rows would differ from the resident in columns the
  // kernel does store. Require them all.
  if (sub.getStaticSizes().front() != viewType.getDimSize(0))
    return false;
  for (int64_t dim = 1; dim < rank; ++dim) {
    int64_t size = sub.getStaticSizes()[dim];
    if (ShapedType::isDynamic(size))
      continue;
    if (size <= 0 || size > viewType.getDimSize(dim))
      return false;
  }
  return true;
}

/// What one arm of a tail guard reads: the entry-argument view the arm reads,
/// or null when the arm cannot be shown to read one.
static memref::ReinterpretCastOp armView(Block &arm, Value yielded) {
  Value inner = peelViewCasts(yielded);
  if (auto direct = inner.getDefiningOp<memref::ReinterpretCastOp>())
    return direct;
  // The gather arm: a fresh dense buffer this arm allocates, zeroes and then
  // fills by copying a prefix of an entry-argument view into it.
  auto buffer = inner.getDefiningOp<memref::AllocOp>();
  if (!buffer)
    return nullptr;
  memref::ReinterpretCastOp view = nullptr;
  for (Operation &op : arm.without_terminator()) {
    auto copy = dyn_cast<memref::CopyOp>(op);
    if (!copy)
      continue;
    if (copyTargetBuffer(copy.getTarget()) != buffer.getResult())
      continue;
    auto source = copy.getSource().getDefiningOp<memref::SubViewOp>();
    if (!source)
      return nullptr;
    auto sourceView =
        dyn_cast_or_null<memref::ReinterpretCastOp>(source.getSource()
                                                        .getDefiningOp());
    if (!sourceView)
      return nullptr;
    if (!isViewPrefix(source, sourceView))
      return nullptr;
    if (view && view != sourceView)
      return nullptr;
    view = sourceView;
  }
  return view;
}

/// The entry-argument N view a pack source reads through, or null.
///
/// Two shapes reach here (A1 of docs/hmx/pack-redundancy-fix-plan-2026-10-09.md):
///
///   * the bare `reinterpret_cast` of the argument; and
///   * a **tail-guarded** view: a real kernel guards the last N block with
///     `if NN - n0 < BN`, so the masked load becomes an `scf.if` whose arms
///     `memref.cast` into one merged type -- the direct arm yields the view
///     itself, the gather arm yields a dense buffer it filled from a prefix of
///     that view. The pack source is then the `scf.if`'s result, and the bare
///     `getDefiningOp<ReinterpretCastOp>()` misses it.
///
/// Both arms must resolve to the *same* `reinterpret_cast` operation: two views
/// of the same argument at different offsets are not one N block, and a guard
/// over an internal buffer is not an entry-argument view at all.
static memref::ReinterpretCastOp resolvePackSourceView(Value v,
                                                       std::string *reason,
                                                       int depth = 0) {
  auto decline = [&](const char *message) {
    if (reason && reason->empty())
      *reason = message;
    return nullptr;
  };
  if (auto view = v.getDefiningOp<memref::ReinterpretCastOp>())
    return view;
  auto guard = depth < 2 ? v.getDefiningOp<scf::IfOp>() : nullptr;
  if (!guard || guard->getNumResults() != 1 || guard->getResult(0) != v)
    return decline(
        "the pack source is neither an entry-argument view nor a tail guard "
        "over one");
  auto &thenBlock = guard.getThenRegion().front();
  auto &elseBlock = guard.getElseRegion().front();
  auto yieldThen = dyn_cast<scf::YieldOp>(thenBlock.getTerminator());
  auto yieldElse = dyn_cast<scf::YieldOp>(elseBlock.getTerminator());
  if (!yieldThen || !yieldElse || yieldThen.getNumOperands() != 1 ||
      yieldElse.getNumOperands() != 1)
    return decline("the tail guard over the pack source does not yield one view");
  memref::ReinterpretCastOp thenView = armView(thenBlock, yieldThen.getOperand(0));
  memref::ReinterpretCastOp elseView = armView(elseBlock, yieldElse.getOperand(0));
  if (!thenView || !elseView || thenView != elseView)
    return decline(
        "the tail guard's arms do not all read the same entry-argument N view");
  return thenView;
}

/// Match the bridge source as `reinterpret_cast(arg, offset=[n0], sizes=[K, BN],
/// strides=[N, 1])` over an entry argument: one N block of a `[K, N]` weight,
/// either directly or behind the tail guard `resolvePackSourceView` describes.
/// Returns null for anything whose whole weight cannot be pinned -- not a 2D
/// row-major view, a non-argument source, a dynamic/unaligned offset, a view the
/// crouton grid disagrees with, or a block that does not tile N. The caller then
/// keeps the old per-launch bridge: guessing would silently pack the wrong bytes.
/// `reason` (when given) names the first check that refused, so the caller can
/// report the decline instead of dropping it silently.
///
/// **Why the guarded form may be matched (A1, the correctness argument).**
/// Replacing the bridge makes the resident serve bytes the gather arm did not:
/// the guard zeroes (in practice: leaves unpadded) the columns of the last tile
/// that run past `NN`, and the resident, built from the argument, holds the
/// argument's real bytes for those same columns. The two images therefore
/// differ only in the columns `n0 + c >= NN`. Each of those columns feeds
/// exactly one output column -- `acc[m, c]` sums `A[m, k] * B[k, c]`, so no
/// reduction ever mixes weight columns -- and those output columns are past the
/// logical N. A kernel that stores them would store past the end of the output
/// row, which is an out-of-bounds write into the next row; a kernel that is
/// correct masks them off (the bench kernel does: its store carries the same
/// guard its load does). So the differing bytes are written nowhere a later
/// read can observe, and the gather arm's fill value -- zero or otherwise --
/// cannot matter. The argument assumes the kernel stores what it computes only
/// for in-range columns, which is the same assumption the unguarded form of
/// this matcher already relies on when it hands the resident the whole slice.
///
/// What must *not* be relaxed for this to hold: both arms have to read the same
/// view (`resolvePackSourceView`), and the gather has to cover every row of it
/// (`isViewPrefix`) -- a row the gather skipped would differ in columns the
/// kernel does store.
static std::optional<WeightSlice> underlyingSliceArgument(Value v,
                                                          MemRefType crouton,
                                                          std::string *reason =
                                                              nullptr) {
  auto decline = [&](const char *message) -> std::optional<WeightSlice> {
    if (reason && reason->empty())
      *reason = message;
    return std::nullopt;
  };
  memref::ReinterpretCastOp reinterpret = resolvePackSourceView(v, reason);
  if (!reinterpret)
    return std::nullopt;
  auto srcType = dyn_cast<MemRefType>(v.getType());
  if (!srcType || srcType.getRank() != 2 || !srcType.hasStaticShape())
    return decline("the pack source is not a static 2D view");
  auto viewType = dyn_cast<MemRefType>(reinterpret.getResult().getType());
  if (!viewType || viewType.getRank() != 2 || !viewType.hasStaticShape())
    return decline("the entry-argument view is not a static 2D view");
  // A guard merges its arms into one result type; the bridge packs whatever
  // that type says, so it has to be the view's geometry (same shape, same
  // element type) for the proven slice to describe the bytes actually read.
  if (srcType.getShape() != viewType.getShape() ||
      srcType.getElementType() != viewType.getElementType())
    return decline("the pack source type does not match the entry-argument "
                   "view it reads");
  // The pre-pack contract is a function-argument contract: the source must be
  // the entry argument itself, not an internal buffer.
  auto arg = dyn_cast<BlockArgument>(reinterpret.getSource());
  if (!arg)
    return decline("the pack source view does not start at a function argument");
  // A row-major view with unit inner stride: the underlying matrix is [K, N]
  // with N = stride(0), and the view is one N block of it.
  auto strided = dyn_cast<StridedLayoutAttr>(viewType.getLayout());
  if (!strided)
    return decline("the entry-argument view is not strided row-major");
  auto strides = strided.getStrides();
  if (strides.size() != 2 || strides[1] != 1 ||
      ShapedType::isDynamic(strides[0]) || strides[0] <= 0)
    return decline("the entry-argument view is not a row-major [K, N] N block "
                   "(unit inner stride, whole-N row stride)");
  int64_t n = strides[0];
  int64_t k = viewType.getDimSize(0);
  int64_t bn = viewType.getDimSize(1);
  // The crouton bridge already fixed the tile grid; the view has to describe the
  // same [K, BN] block, the whole grid has to be whole croutons, and the block
  // has to tile N exactly so every program's slice lies inside the resident.
  // A weight grid is [Nt, Kt, ...]: dim0 is N, dim1 is K.
  if (crouton.getRank() != 5 ||
      hmx::weightKTiles(crouton) * hmx::layout::kTileEdge != k ||
      hmx::weightNTiles(crouton) * hmx::layout::kTileEdge != bn)
    return decline("the entry-argument view does not describe the same [K, BN] "
                   "block as the crouton grid");
  // Strictly narrower than the whole N: the model is "the view is *one* N block
  // of a wider weight", so a view as wide as the whole N is a block of nothing
  // (there are no other blocks for an offset to select) and any offset into it
  // is not an N offset -- it is the K-block case the offset check below rejects.
  // One N block means at least two.
  if (n % hmx::layout::kTileEdge != 0 || bn % hmx::layout::kTileEdge != 0 || n <= bn ||
      n % bn != 0)
    return decline("the N block does not tile the weight's N into whole 32-wide "
                   "blocks");
  // The offset is the descriptor's element offset (a one-element list), i.e. the
  // N block this program owns. A static offset is checked directly; a dynamic
  // one has to be provably tile-aligned (`pid * BN`, or a static-step loop's
  // induction variable -- see `isMultipleOf`) *and* provably a column offset: in
  // a row-major [K, N] matrix every whole number of rows is a multiple of N, so
  // an offset that is provably a multiple of N is a row (K) offset -- a
  // loop-varying block of an activation consumed as a weight, whose resident
  // holds one block while the offset walks past its end. A column offset is
  // never such a multiple (offset 0 is the dense path's business).
  if (reinterpret.getStaticOffsets().size() != 1)
    return decline("the entry-argument view does not name one element offset");
  WeightSlice slice;
  slice.arg = arg;
  slice.n = n;
  slice.view = reinterpret;
  int64_t staticOffset = reinterpret.getStaticOffsets().front();
  if (staticOffset != ShapedType::kDynamic) {
    if (staticOffset < 0 || staticOffset % hmx::layout::kTileEdge != 0 ||
        staticOffset + bn > n)
      return decline("the static N offset is negative, off the 32-element tile "
                     "edge, or runs past the weight's N");
    slice.staticOffset = staticOffset;
  } else {
    if (reinterpret.getOffsets().size() != 1)
      return decline("the entry-argument view does not name one element offset");
    Value offset = reinterpret.getOffsets().front();
    if (!isMultipleOf(offset, hmx::layout::kTileEdge))
      return decline("the N offset is not provably a multiple of 32, so the "
                     "resident's crouton subview cannot be indexed exactly");
    if (isMultipleOf(offset, n))
      return decline("the offset is provably a multiple of N, so it selects "
                     "rows (a K block), not an N block");
    slice.dynamicOffset = offset;
  }
  return slice;
}

static bool looksLikeEntryWeightSource(Value value) {
  if (isa<BlockArgument>(value))
    return true;
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>())
    return looksLikeEntryWeightSource(reinterpret.getSource());
  return false;
}

/// The strict runtime-source proof is intentionally narrower than the existing
/// matcher. A host prepack contract may describe a ranked or unranked entry
/// memref, but it must be an admitted float (f16, or f32 which the host packs
/// to the same fp16 crouton the device pack would build), have an exact static
/// view, and use either a zero/dense view or a statically fixed N slice.  A
/// dynamic offset is not a content identity: it can select a different object
/// on each invocation.
static LogicalResult
validateStrictRuntimeSource(Value src, MemRefType crouton, func::FuncOp func,
                            Operation *anchor, int64_t *slotOut,
                            DictionaryAttr *sourceViewOut = nullptr) {
  auto srcMemref = dyn_cast<MemRefType>(src.getType());
  if (!srcMemref || !dtype::isAdmittedFloat(srcMemref.getElementType()))
    return anchor->emitError(
        "strict resident runtime weight source must be an f16/f32 memref view");

  BlockArgument arg;
  std::optional<WeightSlice> slice = underlyingSliceArgument(src, crouton);
  if (BlockArgument dense = underlyingDenseArgument(src)) {
    arg = dense;
  } else if (slice) {
    if (slice->dynamicOffset)
      return anchor->emitError(
          "strict resident runtime weight source has a dynamic offset");
    arg = slice->arg;
  } else {
    return anchor->emitError(
        "strict resident runtime weight source is not a whole entry-argument "
        "view or a static N slice");
  }
  if (arg.getOwner() != &func.getBody().front())
    return anchor->emitError(
        "strict resident runtime weight source is not an entry argument");
  if (slice) {
    // The view, not the pack source: a tail guard's merged result type has a
    // dynamic layout offset even when the view it merges has a concrete one.
    auto viewType = cast<MemRefType>(slice->view.getResult().getType());
    auto strided = cast<StridedLayoutAttr>(viewType.getLayout());
    int64_t layoutOffset = strided.getOffset();
    if (!slice->staticOffset || ShapedType::isDynamic(layoutOffset) ||
        layoutOffset != *slice->staticOffset)
      return anchor->emitError(
          "strict resident runtime weight source offset disagrees with its "
          "view layout");
  }

  // The view matcher proves how the pack bridge reads the source.  A ranked
  // entry argument must additionally have an identity layout and a matching
  // static shape.  The established unranked ABI is narrower: only its
  // zero-offset, dense, static *view* is accepted, while content/shape
  // identity remains explicitly not-proven in the resident record.
  auto argumentType = dyn_cast<MemRefType>(arg.getType());
  auto unrankedArgument = dyn_cast<UnrankedMemRefType>(arg.getType());
  if (argumentType) {
    if (!argumentType.hasStaticShape() || argumentType.getRank() != 2 ||
        !dtype::isAdmittedFloat(argumentType.getElementType()) ||
        !argumentType.getLayout().isIdentity())
      return anchor->emitError(
          "strict resident runtime weight argument has no exact ranked "
          "row-major shape/stride");
    if (slice) {
      if (srcMemref.getRank() != 2 ||
          argumentType.getDimSize(0) != srcMemref.getDimSize(0) ||
          argumentType.getDimSize(1) != slice->n)
        return anchor->emitError(
            "strict resident runtime weight slice does not match its entry "
            "argument shape");
    } else if (srcMemref.getRank() != 2 ||
               argumentType.getShape() != srcMemref.getShape()) {
      return anchor->emitError(
          "strict resident runtime weight view does not match its entry "
          "argument shape");
    }
  } else if (!unrankedArgument || slice || !srcMemref ||
             !dtype::isAdmittedFloat(srcMemref.getElementType()) ||
             srcMemref.getRank() != 2) {
    return anchor->emitError(
        "strict resident runtime weight argument has no exact ranked "
        "row-major shape/stride");
  } else if (crouton.getRank() != 5 ||
             srcMemref.getDimSize(0) !=
                 hmx::weightKTiles(crouton) * hmx::layout::kTileEdge ||
             srcMemref.getDimSize(1) !=
                 hmx::weightNTiles(crouton) * hmx::layout::kTileEdge) {
    return anchor->emitError(
        "strict resident runtime weight view does not match its entry "
        "argument shape");
  }

  std::optional<int64_t> slot = tensorArgumentSlot(func, arg);
  if (!slot)
    return anchor->emitError(
        "strict resident runtime weight argument has no tensor slot");
  std::optional<DictionaryAttr> sourceView = makeRuntimeSourceView(
      src.getContext(), src, arg, slice ? &*slice : nullptr);
  if (!sourceView)
    return anchor->emitError(
        "strict resident runtime weight source view cannot be canonicalized");
  if (slotOut)
    *slotOut = *slot;
  if (sourceViewOut)
    *sourceViewOut = *sourceView;
  return success();
}

static LogicalResult
validateStrictWeightPack(WeightPack &pack, MemRefType crouton,
                         func::FuncOp func, Operation *anchor, int64_t *slotOut,
                         DictionaryAttr *sourceViewOut) {
  if (pack.packs.empty())
    // Wording matters here (2026-10-01). "has no pack" reads like a claim that
    // the IR is malformed, and the reader then goes looking for a producer bug.
    // Under `hasStrictResidentContract` the truth is narrower: strict mode is
    // documented as "the explicit request to fail closed on identity/descriptor
    // evidence" (HmxResidentContract.h:102-104), so a missing pack is the
    // pass refusing to proceed without that evidence. A legitimately
    // weight-RESIDENT operand also has no pack -- that is what residency means
    // -- and conflating the two is the same mistake
    // HmxPartitionPass.cpp:emitDiagnosticInputBridges made until 2026-10-01
    // (see ROADMAP.md §2.1, the tail-weight-resident row). Say which one this
    // is. Unreachable today: the only caller passes a `pack` that came from
    // `findWeightPack`, so it is non-empty by construction; kept as a defensive
    // assertion, not as a live gate.
    return anchor->emitError(
        "strict resident contract: the weight bridge carries no pack, so this "
        "pass cannot establish that the weight is resident; refusing to "
        "proceed. NOTE a genuinely weight-resident operand also has no pack, so "
        "this is not by itself evidence of a malformed producer");
  Value source = pack.packs.front().getSrc();
  for (PackWeightOp writer : pack.packs)
    if (writer.getSrc() != source)
      return anchor->emitError(
          "strict resident weight bridge has multiple source views");
  unsigned directDeallocs = 0;
  bool aliasDealloc = false;
  if (failed(validateResidentDeallocs(
          pack.array.getResult(), anchor, directDeallocs, aliasDealloc,
          ResidentDeallocForm::MemrefAndHexagonmem,
          "resident weight bridge")))
    return failure();
  if (directDeallocs != 1)
    return anchor->emitError(
        "strict resident weight bridge must have one direct deallocation");
  return validateStrictRuntimeSource(source, crouton, func, anchor, slotOut,
                                     sourceViewOut);
}

struct ExistingWeightResident {
  DictionaryAttr resident;
  FlatSymbolRefAttr global;
  int64_t slot = -1;
  DictionaryAttr sourceView;
  std::string principal;
  std::string function;
  std::string site;
  uint64_t identityKey = 0;
  int64_t alignment = 0;
};

/// Validate resident records which were produced by an earlier invocation or
/// supplied by a hand-written pipeline.  This keeps a repeated pass from
/// silently accepting a changed source/descriptor.
static LogicalResult
validateExistingWeightResident(Operation *operation,
                               ExistingWeightResident *existingOut = nullptr) {
  auto parentFunction = operation->getParentOfType<func::FuncOp>();
  Operation *anchor =
      parentFunction ? parentFunction.getOperation() : operation;
  auto resident = operation->getAttrOfType<DictionaryAttr>(kResidentAttr);
  if (!resident)
    return success();
  if (operation->getNumResults() != 1)
    return anchor->emitError(
        "strict resident weight must have exactly one allocation result");

  auto alloc = dyn_cast<hexagonmem::AllocOp>(operation);
  auto type = dyn_cast<MemRefType>(operation->getResult(0).getType());
  auto declaredBytes = resident.getAs<IntegerAttr>(kResidentKeyBytes);
  std::optional<int64_t> bytes = type ? checkedByteSize(type) : std::nullopt;
  if (!alloc || !bytes || !declaredBytes || declaredBytes.getInt() != *bytes ||
      !isSupportedResidentAlignment(residentAlignment(alloc)))
    return anchor->emitError(
        "strict resident weight has an inexact byte/alignment descriptor");

  auto provenance =
      operation->getAttrOfType<DictionaryAttr>(kHmxResidentProvenanceAttr);
  auto parentModule = operation->getParentOfType<ModuleOp>();
  auto siteName =
      provenance ? provenance.getAs<StringAttr>("site") : StringAttr();
  if (!provenance || !parentFunction || !parentModule || !siteName ||
      !isCanonicalResidentSite(siteName.getValue()))
    return anchor->emitError(
        "strict resident weight has no principal/provenance record");
  StringRef site = siteName.getValue();

  std::string principal = residentPrincipalName(parentModule);
  uint64_t functionId =
      residentFunctionIdentity(principal, parentFunction.getSymName());
  if (functionId == 0)
    return anchor->emitError(
        "strict resident weight function identity is zero");

  unsigned directDeallocs = 0;
  bool aliasDealloc = false;
  // The allocation here is already a `hexagonmem.alloc`; an earlier
  // lowering may have left its release in either op form.
  if (failed(validateResidentDeallocs(
          alloc.getResult(), operation, directDeallocs, aliasDealloc,
          ResidentDeallocForm::MemrefAndHexagonmem,
          "resident weight bridge")) ||
      directDeallocs != 0)
    return anchor->emitError(
        "strict resident weight has an alias or direct deallocation");

  auto globalRef = resident.getAs<FlatSymbolRefAttr>(kResidentKeyGlobal);
  auto address = resident.get(kResidentKeyAddress);
  if (globalRef && address)
    return anchor->emitError(
        "strict resident weight has both global and address provenance");

  DictionaryAttr expectedResident;
  DictionaryAttr expectedProvenance;
  DictionaryAttr runtimeSourceView;
  MLIRContext *context = operation->getContext();
  if (globalRef) {
    auto source = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        operation, globalRef.getAttr());
    if (!validGlobalWeightSource(source, type))
      return anchor->emitError(
          "strict resident weight global source is not an exact constant");
    uint64_t siteId = residentSiteIdentity(
        principal, parentFunction.getSymName(), site, "weight-resident",
        /*slot=*/0);
    if (siteId == 0)
      return anchor->emitError("strict resident weight site identity is zero");
    expectedResident = DictionaryAttr::get(
        context, {NamedAttribute(StringAttr::get(context, kResidentKeyGlobal),
                                 globalRef),
                  NamedAttribute(StringAttr::get(context, kResidentKeyBytes),
                                 IntegerAttr::get(IntegerType::get(context, 64),
                                                  *bytes))});
    expectedProvenance = makeWeightProvenance(
        context, principal, parentFunction.getSymName(), site, /*slot=*/0,
        *bytes, residentAlignment(alloc), siteId, "global-address",
        "global:" + globalRef.getValue().str(), kHmxResidentImmutableGlobal,
        "compile-time-symbol");
  } else if (address) {
    if (!isa<UnitAttr>(address))
      return anchor->emitError(
          "strict resident weight address marker must be a unit attribute");
    if (alloc.getDynamicSizes().size() != 1)
      return anchor->emitError(
          "strict runtime resident weight has an unsupported source operand");
    auto extract = alloc.getDynamicSizes()[0]
                       .getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>();
    BlockArgument argument;
    BaseMemRefType sourceType;
    if (extract) {
      argument = dyn_cast<BlockArgument>(extract.getSource());
      sourceType = dyn_cast<BaseMemRefType>(extract.getSource().getType());
    }
    if (!argument)
      return anchor->emitError(
          "strict runtime resident weight is not an entry f16/f32 argument");
    auto argumentType = dyn_cast<MemRefType>(argument.getType());
    bool unrankedArgument = isa<UnrankedMemRefType>(argument.getType());
    if (type.getRank() != 5 || type.getDimSize(2) != 16 ||
        type.getDimSize(3) != 32 || type.getDimSize(4) != 2)
      return anchor->emitError(
          "strict runtime resident weight has an invalid crouton type");
    const int64_t logicalK = hmx::weightKTiles(type) * hmx::layout::kTileEdge;
    const int64_t logicalN = hmx::weightNTiles(type) * hmx::layout::kTileEdge;
    if (!sourceType || !dtype::isAdmittedFloat(sourceType.getElementType()) ||
        argument.getOwner() != &parentFunction.getBody().front())
      return anchor->emitError(
          "strict runtime resident weight is not an entry f16/f32 argument");
    if (argumentType) {
      if (!argumentType.hasStaticShape() || argumentType.getRank() != 2 ||
          !dtype::isAdmittedFloat(argumentType.getElementType()) ||
          !argumentType.getLayout().isIdentity() ||
          argumentType.getDimSize(0) != logicalK ||
          argumentType.getDimSize(1) != logicalN)
        return anchor->emitError(
            "strict runtime resident weight argument has no exact ranked "
            "row-major shape/stride");
    } else if (!unrankedArgument) {
      return anchor->emitError(
          "strict runtime resident weight argument has no exact ranked "
          "row-major shape/stride");
    }
    std::optional<int64_t> slot = tensorArgumentSlot(parentFunction, argument);
    if (!slot)
      return anchor->emitError("strict resident weight has no tensor slot");
    runtimeSourceView = provenance.getAs<DictionaryAttr>("source_view");
    if (failed(validateRuntimeSourceView(runtimeSourceView, type, argument,
                                         anchor)))
      return failure();
    uint64_t siteId = residentSiteIdentity(
        principal, parentFunction.getSymName(), site, "weight-resident", *slot);
    if (siteId == 0)
      return anchor->emitError("strict resident weight site identity is zero");
    // Rebuilt from the allocation's own memory space rather than echoed from
    // the descriptor: a `location` the IR claims and the placement disagree
    // about fails the equality below instead of being read twice. VTCM is
    // spelled by absence, so a resident the producer labelled `vtcm` in the
    // IR is rejected as non-canonical.
    NamedAttrList residentFields;
    residentFields.append(kResidentKeyAddress, address);
    residentFields.append(
        kResidentKeyBytes,
        IntegerAttr::get(IntegerType::get(context, 64), *bytes));
    if (!hexagon::isInVTCMAddressSpace(type))
      residentFields.append(
          StringAttr::get(context, kHmxWeightResidentLocationKey),
          StringAttr::get(context, kHmxWeightResidentLocationDdr));
    expectedResident = residentFields.getDictionary(context);
    expectedProvenance = makeWeightProvenance(
        context, principal, parentFunction.getSymName(), site, *slot, *bytes,
        residentAlignment(alloc), siteId, "argument-address",
        "entry-argument-slot:" + std::to_string(*slot), kHmxResidentNotProven,
        "not-carried-to-device", runtimeSourceView);
  } else {
    return anchor->emitError(
        "strict resident weight has neither global nor address provenance");
  }

  if (resident != expectedResident)
    return anchor->emitError(
        "strict resident weight descriptor does not match its source");
  if (provenance != expectedProvenance)
    return anchor->emitError(
        "strict resident weight provenance does not match its source");
  if (existingOut) {
    auto slot = provenance.getAs<IntegerAttr>("slot");
    auto identity = provenance.getAs<IntegerAttr>("identity_key");
    existingOut->resident = resident;
    existingOut->global = globalRef;
    existingOut->slot = slot.getInt();
    existingOut->sourceView = runtimeSourceView;
    existingOut->principal = principal;
    existingOut->function = parentFunction.getSymName().str();
    existingOut->site = site.str();
    existingOut->identityKey = static_cast<uint64_t>(identity.getInt());
    existingOut->alignment = residentAlignment(alloc);
  }
  return success();
}

static std::string constantResidentIdentityOwner(StringRef principal,
                                                 StringRef function,
                                                 StringRef site,
                                                 StringRef global) {
  std::string owner = principal.str();
  owner += "\\x1f";
  owner += function.str();
  owner += "\\x1f";
  owner += site.str();
  owner += "\\x1fglobal:";
  owner += global.str();
  owner += "\\x1f0";
  return owner;
}

/// Preflight only the evidence-sensitive cases. The compatibility path still
/// keeps its device-pack fallback for an unmarked module; with the diagnostic
/// marker an unsupported runtime source is an error rather than a fallback.
static LogicalResult strictWeightPreflight(ModuleOp module,
                                           bool prepackRuntimeWeights) {
  if (!isValidResidentDiagnosticMarker(module))
    return module.emitError(
        "hmx.diagnostic_vtcm_accounting must be a unit attribute");
  if (!isValidResidentIdentityMarker(module))
    return module.emitError(
        "hmx.diagnostic_vtcm_identity must be a unit attribute");

  LogicalResult result = success();
  struct GlobalDescriptor {
    int64_t bytes;
    int64_t alignment;
  };

  std::map<uint64_t, std::string> identityOwners;
  std::map<std::string, GlobalDescriptor> globalOwners;
  std::map<std::string, ExistingWeightResident> existingGlobalBySymbol;
  std::map<int64_t, DictionaryAttr> runtimeViewBySlot;
  std::set<std::string> residentFunctions;

  // Establish existing ownership first. A repeated or partially lowered module
  // must be compared against these records before any new resident is emitted.
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    SmallVector<Operation *> records;
    function.walk([&](Operation *operation) {
      if (operation->hasAttr(kResidentAttr))
        records.push_back(operation);
    });
    for (Operation *operation : records) {
      ExistingWeightResident record;
      if (failed(validateExistingWeightResident(operation, &record))) {
        result = failure();
        continue;
      }
      residentFunctions.insert(record.function);

      if (record.global) {
        std::string symbol = record.global.getValue().str();
        GlobalDescriptor descriptor{
            record.resident.getAs<IntegerAttr>(kResidentKeyBytes).getInt(),
            record.alignment};
        auto [descriptorIt, descriptorInserted] =
            globalOwners.emplace(symbol, descriptor);
        if (!descriptorInserted) {
          if (descriptorIt->second.bytes != descriptor.bytes ||
              descriptorIt->second.alignment != descriptor.alignment)
            operation->emitError(
                "strict resident weight global has conflicting descriptors");
          else
            operation->emitError(
                "strict pre-existing resident records duplicate one global");
          result = failure();
        }
        existingGlobalBySymbol.emplace(symbol, record);

        std::string owner = constantResidentIdentityOwner(
            record.principal, record.function, record.site, symbol);
        auto [ownerIt, ownerInserted] =
            identityOwners.emplace(record.identityKey, owner);
        if (!ownerInserted && ownerIt->second != owner) {
          operation->emitError("strict resident weight identity key collision");
          result = failure();
        }
        continue;
      }

      auto [slotIt, slotInserted] =
          runtimeViewBySlot.emplace(record.slot, record.sourceView);
      if (!slotInserted && slotIt->second != record.sourceView) {
        operation->emitError(
            "strict pre-existing runtime weight records conflict for one slot");
        result = failure();
      } else if (!slotInserted) {
        operation->emitError(
            "strict pre-existing runtime weight records duplicate one slot");
        result = failure();
      }
    }
  }
  if (failed(result))
    return failure();

  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    function.walk([&](MatmulOp op) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.getOperandNumber() >= 2)
          continue;
        MemRefType type = dyn_cast<MemRefType>(operand.get().getType());
        if (!type)
          continue;

        if (auto source = prepackedSource(operand.get())) {
          if (!validGlobalWeightSource(source, type)) {
            op.emitError("strict resident weight global source is not exact");
            result = failure();
            continue;
          }
          std::optional<int64_t> bytes = checkedByteSize(type);
          if (!bytes) {
            op.emitError(
                "strict resident weight byte size is not representable");
            result = failure();
            continue;
          }

          std::string symbol = source.getSymName().str();
          int64_t alignment = 128;
          auto existing = existingGlobalBySymbol.find(symbol);
          if (existing != existingGlobalBySymbol.end()) {
            alignment = existing->second.alignment;
            if (alignment != 128) {
              op.emitError(
                  "strict resident weight global has a non-producer alignment");
              result = failure();
            }
          }
          GlobalDescriptor descriptor{*bytes, alignment};
          auto [descriptorIt, descriptorInserted] =
              globalOwners.emplace(symbol, descriptor);
          if (!descriptorInserted &&
              (descriptorIt->second.bytes != descriptor.bytes ||
               descriptorIt->second.alignment != descriptor.alignment)) {
            op.emitError(
                "strict resident weight global has conflicting descriptors");
            result = failure();
          }

          std::optional<std::string> location = residentSite(op.getOperation());
          std::string principal = residentPrincipalName(module);
          if (!location ||
              residentFunctionIdentity(principal, function.getSymName()) == 0) {
            op.emitError("strict resident weight has no stable "
                         "principal/function/site identity");
            result = failure();
            continue;
          }
          uint64_t identityKey = residentSiteIdentity(
              principal, function.getSymName(), *location, "weight-resident",
              /*slot=*/0);
          if (identityKey == 0) {
            op.emitError("strict resident weight site identity is zero");
            result = failure();
            continue;
          }
          std::string owner = constantResidentIdentityOwner(
              principal, function.getSymName(), *location, symbol);
          auto [ownerIt, ownerInserted] =
              identityOwners.emplace(identityKey, owner);
          if (!ownerInserted && ownerIt->second != owner) {
            op.emitError("strict resident weight identity key collision");
            result = failure();
          }
          residentFunctions.insert(function.getSymName().str());
          continue;
        }

        if (!prepackRuntimeWeights || operand.getOperandNumber() != 1)
          continue;
        if (auto defining = operand.get().getDefiningOp())
          if (defining->hasAttr(kResidentAttr))
            continue;
        std::optional<WeightPack> pack = findWeightPack(operand.get());
        if (!pack) {
          if (looksLikeEntryWeightSource(operand.get())) {
            // Same wording correction as validateStrictWeightPack above
            // (2026-10-01). `findWeightPack` returning null has several causes
            // and "no pack bridge" names only one of them. The common cause in
            // production shapes is the mundane one: the source is not a view
            // this pass knows how to make resident. Flash attention's V is
            // exactly that -- a row block (K-block) of a wider matrix, which
            // `looksLikeEntryWeightSource` cannot tell apart from an entry
            // weight, and which the N-slice matcher rejects by design
            // (WeightResidentPass.cpp:954, and the comment at :959-964). So this
            // error, if strict mode were ever enabled in production, would fire
            // on a kernel that is merely NOT-RESIDENT rather than malformed.
            // Say so, or the reader will go hunting for a producer bug that
            // does not exist -- the same dead end HmxPartitionPass.cpp led
            // someone down today (ROADMAP.md §2.1, tail-weight-resident row).
            op.emitError(
                "strict resident contract: this runtime weight source has no "
                "pack bridge this pass can use, so residency cannot be "
                "established; refusing to proceed. That is the expected "
                "outcome for any weight whose source view this pass does not "
                "model (e.g. a K-block row slice), NOT evidence that the "
                "producer is malformed");
            result = failure();
          }
          continue;
        }

        // A transposed pack source is outside the pre-pack contract: the host
        // pre-packs a [K, N] row-major argument (backend/hmx_weight_prepack.py)
        // while `underlyingDenseArgument`/`underlyingSliceArgument` reason
        // about the source's own geometry -- a [N, K] transpose input could be
        // misread as a view this pass models and pre-pack the wrong bytes.
        // Strict mode fails closed on it.
        if (llvm::any_of(pack->packs,
                         [](PackWeightOp p) { return p.getSrcTransposed().value_or(false); })) {
          op.emitError("strict resident contract: a transposed pack source is "
                       "not in the pre-pack contract, so residency cannot be "
                       "established; refusing to proceed");
          result = failure();
          continue;
        }

        int64_t slot = 0;
        DictionaryAttr sourceView;
        if (failed(validateStrictWeightPack(*pack, type, function,
                                            op.getOperation(), &slot,
                                            &sourceView))) {
          result = failure();
          continue;
        }
        auto [viewIt, viewInserted] =
            runtimeViewBySlot.emplace(slot, sourceView);
        if (!viewInserted && viewIt->second != sourceView) {
          op.emitError(
              "strict resident runtime weight slot has conflicting source "
              "views");
          result = failure();
          continue;
        }

        std::optional<std::string> location = residentSite(op.getOperation());
        if (!location) {
          op.emitError(
              "strict resident weight has no stable allocation-site location");
          result = failure();
          continue;
        }
        std::string principal = residentPrincipalName(module);
        if (residentFunctionIdentity(principal, function.getSymName()) == 0) {
          op.emitError("strict resident weight function identity is zero");
          result = failure();
          continue;
        }
        std::string site = *location;
        uint64_t identityKey =
            residentSiteIdentity(principal, function.getSymName().str(), site,
                                 "weight-resident", slot);
        if (identityKey == 0) {
          op.emitError("strict resident weight site identity is zero");
          result = failure();
          continue;
        }
        std::string identity = principal + "\\x1f" +
                               function.getSymName().str() + "\\x1f" + site +
                               "\\x1f" + std::to_string(slot);
        auto [identityIt, identityInserted] =
            identityOwners.emplace(identityKey, identity);
        if (!identityInserted && identityIt->second != identity) {
          op.emitError("strict resident weight identity key collision");
          result = failure();
        }
        residentFunctions.insert(function.getSymName().str());
      }
    });
  }
  if (failed(result))
    return failure();
  if (residentFunctions.size() > 1)
    return module.emitError(
        "strict resident weight provenance is ambiguous across functions");
  return success();
}

struct WeightResidentPass
    : public mlir::hmx::impl::WeightResidentBase<WeightResidentPass> {
  using WeightResidentBase::WeightResidentBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, hexagonmem::HexagonMemDialect,
                    hmx::HmxDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    // Resident prepack declarations and aggregate bytes are module state; the
    // nested function pass may otherwise race sibling functions.
    std::lock_guard<std::mutex> manifestGuard(hmxModuleStateMutex());

    func::FuncOp func = cast<func::FuncOp>(getOperation());
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return;
    if (!isValidResidentDiagnosticMarker(module)) {
      module.emitError(
          "hmx.diagnostic_vtcm_accounting must be a unit attribute");
      return signalPassFailure();
    }
    if (!isValidResidentIdentityMarker(module)) {
      module.emitError("hmx.diagnostic_vtcm_identity must be a unit attribute");
      return signalPassFailure();
    }
    // The raw census marker remains observational.  Only the paired static
    // identity marker opts into source-site/descriptor rejection; this keeps
    // existing diagnostic pipeline output and production key selection
    // unchanged.
    const bool strictResidentContract = hasStrictResidentContract(module);
    if (strictResidentContract &&
        failed(strictWeightPreflight(module, prepackRuntimeWeights)))
      return signalPassFailure();

    SmallVector<MatmulOp> matmuls;
    func.walk([&](MatmulOp op) { matmuls.push_back(op); });
    if (matmuls.empty())
      return;

    IRRewriter rewriter(func.getContext());
    // One resident buffer per distinct constant, no matter how many matmuls
    // read it (the runtime keys on the same source address anyway).
    llvm::DenseMap<Attribute, Value> residentBySource;
    // One resident buffer per runtime weight slot. Keep the canonical source
    // view as well: the same slot reached through two different descriptors is
    // ambiguous even when the resulting resident type has the same shape.
    llvm::DenseMap<int64_t, Value> residentBySlot;
    llvm::DenseMap<int64_t, DictionaryAttr> runtimeViewBySlot;
    int64_t addedBytes = 0;

    if (strictResidentContract) {
      func.walk([&](Operation *operation) {
        if (!operation->hasAttr(kResidentAttr))
          return;
        ExistingWeightResident record;
        if (failed(validateExistingWeightResident(operation, &record)))
          return signalPassFailure();
        if (record.global) {
          residentBySource.insert({record.global, operation->getResult(0)});
        } else {
          residentBySlot.insert({record.slot, operation->getResult(0)});
          runtimeViewBySlot.insert({record.slot, record.sourceView});
        }
      });
    }

    for (MatmulOp op : matmuls) {
      for (OpOperand &operand : op->getOpOperands()) {
        MemRefType type = dyn_cast<MemRefType>(operand.get().getType());
        if (!type)
          continue;
        // Only the inputs carry a prepacked weight; the output is written.
        if (operand.getOperandNumber() >= 2)
          continue;

        // --- Constant path: the prepacked `memref.global`. ---
        if (memref::GlobalOp source = prepackedSource(operand.get())) {
          if (strictResidentContract &&
              !validGlobalWeightSource(source, type)) {
            op.emitError(
                "strict resident weight global source is not an exact constant");
            return signalPassFailure();
          }
          int64_t bytes = byteSize(type);
          std::optional<std::string> strictSite;
          uint64_t identityKey = 0;
          if (strictResidentContract) {
            std::optional<int64_t> checkedBytes = checkedByteSize(type);
            strictSite = residentSite(op.getOperation());
            std::string principal = residentPrincipalName(module);
            if (!checkedBytes || !strictSite ||
                residentFunctionIdentity(principal, func.getSymName()) == 0) {
              op.emitError(
                  "strict resident weight has no exact principal/function/site "
                  "identity");
              return signalPassFailure();
            }
            bytes = *checkedBytes;
            identityKey = residentSiteIdentity(principal, func.getSymName(),
                                               *strictSite, "weight-resident",
                                               /*slot=*/0);
            if (identityKey == 0) {
              op.emitError("strict resident weight site identity is zero");
              return signalPassFailure();
            }
          }
          // The constant has to keep existing until the lowering reads its
          // address, so make it a public symbol rather than letting symbol DCE
          // drop an operand-less private global.
          source->setAttr(SymbolTable::getVisibilityAttrName(),
                          rewriter.getStringAttr("public"));

          FlatSymbolRefAttr symbol =
              FlatSymbolRefAttr::get(source.getSymNameAttr());
          Value resident = residentBySource.lookup(symbol);
          if (resident && strictResidentContract) {
            Operation *residentOp = resident.getDefiningOp();
            if (!residentOp ||
                failed(validateExistingWeightResident(residentOp))) {
              op.emitError("strict constant resident descriptor changed");
              return signalPassFailure();
            }
          }
          if (!resident) {
            // --- Capacity gate. ---
            //
            // Unlike a runtime weight, a constant has nowhere to fall back to:
            // its only legal form *is* the resident VTCM buffer, because the
            // operand it replaces is a DDR `memref.get_global` and `hmx.mma`
            // requires its weight in VTCM. Leaving it there is not a degraded
            // kernel, it is a module that fails verification three passes later
            // with `'hmx.mma' op wt must be in VTCM` -- a message that names
            // neither the budget nor the residency. Inventing a per-launch copy
            // here would be a new mechanism, so the overrun is reported at the
            // point that knows the numbers and compilation stops. The
            // contraction is still not what is being refused: nothing below
            // revisits the `hmx.matmul`, and `matmul-to-hmx` already admitted
            // it upstream.
            FailureOr<ResidentVtcmAdmission> admission =
                admitResidentVtcm(func, module, addedBytes, bytes);
            if (failed(admission))
              return signalPassFailure();
            if (!admission->admitted) {
              op.emitError()
                  << "resident weight @"
                  << source.getSymName() << " needs " << admission->requested
                  << " bytes but the persistent VTCM total would be "
                  << admission->total << " bytes, over the "
                  << admission->budget << " byte budget (transient "
                  << admission->transient << " + resident "
                  << admission->resident
                  << " + this buffer); a constant weight has no per-launch "
                     "fallback, so the module cannot be lowered";
              return signalPassFailure();
            }
            auto vtcmType =
                MemRefType::get(type.getShape(), type.getElementType(),
                                AffineMap{}, hexagon::VTCM_ADDRESS_SPACE);
            rewriter.setInsertionPoint(op);
            auto alloc = hexagonmem::AllocOp::create(
                rewriter, op.getLoc(), vtcmType, ValueRange{},
                rewriter.getI64IntegerAttr(128));
            alloc->setAttr(
                kResidentAttr,
                rewriter.getDictionaryAttr(
                    {rewriter.getNamedAttr(
                         kResidentKeyGlobal,
                         FlatSymbolRefAttr::get(source.getSymNameAttr())),
                     rewriter.getNamedAttr(
                         kResidentKeyBytes,
                         rewriter.getI64IntegerAttr(bytes))}));
            if (strictResidentContract) {
              DictionaryAttr expected = makeWeightProvenance(
                  func.getContext(), residentPrincipalName(module),
                  func.getSymName(), *strictSite,
                  /*slot=*/0, bytes, /*alignment=*/128, identityKey,
                  "global-address", "global:" + source.getSymName().str(),
                  kHmxResidentImmutableGlobal, "compile-time-symbol");
              if (failed(setOrValidateResidentProvenance(alloc.getOperation(),
                                                         expected,
                                                         "resident weight")))
                return signalPassFailure();
            }
            resident = alloc.getResult();
            residentBySource.insert({symbol, resident});
            if (strictResidentContract &&
                bytes > std::numeric_limits<int64_t>::max() - addedBytes) {
              op.emitError("resident byte aggregate overflows int64");
              return signalPassFailure();
            }
            addedBytes += bytes;
            LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] resident " << symbol
                                    << " (" << bytes << " bytes)\n");
          }

          rewriter.modifyOpInPlace(op, [&]() { operand.set(resident); });
          continue;
        }

        // --- Runtime path: the host pre-packs a function argument. ---
        // Only the weight (rhs) is a resident candidate; the activation (lhs) is
        // packed fresh every launch and stays on the existing bridge.
        if (!prepackRuntimeWeights || operand.getOperandNumber() != 1)
          continue;

        std::optional<WeightPack> pack = findWeightPack(operand.get());
        if (!pack) {
          // Not every rhs is a decline: an operand that never was a bridge (a
          // buffer the kernel built itself) is this pass's business only when it
          // looks like an entry weight, which is the same test the strict walk
          // uses to decide whether "no bridge" is worth reporting at all.
          if (!strictResidentContract && looksLikeEntryWeightSource(operand.get()))
            describeWeightDecline(
                op, std::nullopt,
                "this runtime weight has no pack bridge the pass can model (its "
                "source is not a view of a function argument)");
          continue;
        }
        // Same transposed-source refusal as the strict walk above, minus the
        // error: decline, and the per-launch `_T` bridge keeps the kernel
        // correct.
        if (llvm::any_of(pack->packs,
                         [](PackWeightOp p) { return p.getSrcTransposed().value_or(false); })) {
          if (!strictResidentContract)
            describeWeightDecline(
                op, std::nullopt,
                "the pack source is transposed, which the host pre-pack "
                "contract does not cover");
          continue;
        }
        // The bridge must pack one runtime input, not an internal buffer. It
        // usually reads layout-only views (`reinterpret_cast` from
        // bufferization); resolve those to the entry argument and to the *whole*
        // weight the resident buffer has to hold.
        //
        // Two forms reach here. A dense view covers the whole argument, so the
        // resident is the argument's own crouton array. An N-slice (B2) is one
        // column block of a wider `[K, N]` weight -- the view's row stride is the
        // whole N -- so the argument's bytes *are* the whole weight's bytes and a
        // single resident copy serves every program; the matmul reads its block
        // through a subview. A view that proves neither keeps the per-launch
        // bridge: pre-packing the whole argument for a partial view would
        // silently compute on the wrong bytes.
        Value src = pack->packs.front().getSrc();
        if (llvm::any_of(pack->packs,
                         [&](PackWeightOp p) { return p.getSrc() != src; })) {
          if (strictResidentContract) {
            op.emitError(
                "strict resident weight bridge has multiple source views");
            return signalPassFailure();
          }
          describeWeightDecline(
              op, std::nullopt,
              "the weight bridge packs more than one source view, so one "
              "resident cannot describe what it reads");
          continue;
        }
        // The pre-pack contract is defined on the crouton image: the host
        // produces the bytes this device-side pack would have written. For an
        // fp16 source that is a permutation of its own bytes. For an fp32
        // source the host quantises to the crouton's fp16 with the same
        // round-to-nearest conversion and the same permutation as the f32 pack
        // leaf, so the resident holds the device pack's image for every
        // non-NaN weight (NaN payloads are the engine's own and are documented
        // in backend/hmx_weight_prepack.py). The contract's `dtype` names the
        // source so the host validates the argument it was handed. Any other
        // element type keeps its per-launch bridge (see hmx.pack_weight).
        auto srcMemref = dyn_cast<MemRefType>(src.getType());

        BlockArgument arg;
        MemRefType residentType;
        int64_t logicalN = hmx::weightNTiles(type) * hmx::layout::kTileEdge;
        std::optional<WeightSlice> slice;
        // Why the slice matcher said no, kept for the decline remark: without
        // it every refusal in this path looks identical from the outside, and
        // the two that were silent until 2026-10-09 (a tail-guarded source, and
        // a loop induction variable) are indistinguishable from "never
        // eligible" in the pipeline's output.
        std::string sliceDecline;
        bool matched = false;
        if (BlockArgument denseArg = underlyingDenseArgument(src)) {
          arg = denseArg;
          residentType = MemRefType::get(type.getShape(), type.getElementType(),
                                         AffineMap{},
                                         hexagon::VTCM_ADDRESS_SPACE);
          matched = true;
        } else if ((slice = underlyingSliceArgument(src, type, &sliceDecline))) {
          if (strictResidentContract && slice->dynamicOffset) {
            op.emitError(
                "strict resident runtime weight source has a dynamic offset");
            return signalPassFailure();
          }
          arg = slice->arg;
          logicalN = slice->n;
          SmallVector<int64_t> wholeShape(type.getShape().begin(),
                                          type.getShape().end());
          wholeShape[0] = slice->n / hmx::layout::kTileEdge;
          residentType = MemRefType::get(wholeShape, type.getElementType(),
                                         AffineMap{},
                                         hexagon::VTCM_ADDRESS_SPACE);
          matched = true;
        }
        if (!matched) {
          if (strictResidentContract) {
            op.emitError(
                "strict resident runtime weight source is not an exact entry "
                "view");
            return signalPassFailure();
          }
          describeWeightDecline(
              op, std::nullopt,
              sliceDecline.empty()
                  ? "the pack source does not cover a whole function argument "
                    "and is not a whole-N slice of one"
                  : StringRef(sliceDecline));
          continue;
        }
        if (arg.getOwner() != &func.getBody().front()) {
          if (strictResidentContract) {
            op.emitError(
                "strict resident runtime weight source is not an entry argument");
            return signalPassFailure();
          }
          describeWeightDecline(
              op, std::nullopt,
              "the pack source view does not start at a function entry "
              "argument, so there is no host object to pre-pack");
          continue;
        }
        std::optional<int64_t> tensorSlot = tensorArgumentSlot(func, arg);
        if (!tensorSlot) {
          op.emitError("resident weight argument is not a tensor argument");
          return signalPassFailure();
        }
        int64_t slot = *tensorSlot;
        if (module->hasAttr("hmx.kernel_manifest")) {
          auto decisionId = op->getAttrOfType<IntegerAttr>(kHmxDecisionIdAttr);
          if (!decisionId ||
              failed(bindHmxManifestWeightSlot(
                  module, func.getSymName(), decisionId.getInt(), slot)))
            return signalPassFailure();
        }
        // The manifest records the argument binding even when the source is
        // not an admitted float; the policy pass then gives that slot the
        // canonical device-pack reason.
        if (!srcMemref || !dtype::isAdmittedFloat(srcMemref.getElementType())) {
          if (strictResidentContract) {
            op.emitError(
                "strict resident runtime weight source must be f16/f32");
            return signalPassFailure();
          }
          describeWeightDecline(
              op, slot,
              "the pack source element type is not one the host pre-pack "
              "contract can reproduce (f16/f32 crouton image)");
          continue;
        }

        if (strictResidentContract) {
          unsigned directDeallocs = 0;
          bool aliasDealloc = false;
          if (failed(validateResidentDeallocs(
                  pack->array.getResult(), op.getOperation(), directDeallocs,
                  aliasDealloc, ResidentDeallocForm::MemrefAndHexagonmem,
                  "resident weight bridge")))
            return signalPassFailure();
          if (directDeallocs != 1) {
            op.emitError("strict resident weight bridge must have one direct "
                         "deallocation");
            return signalPassFailure();
          }
        }
        DictionaryAttr strictSourceView;
        if (strictResidentContract) {
          std::optional<DictionaryAttr> sourceView = makeRuntimeSourceView(
              func.getContext(), src, arg, slice ? &*slice : nullptr);
          if (!sourceView) {
            op.emitError("strict resident runtime weight source view cannot be "
                         "canonicalized");
            return signalPassFailure();
          }
          strictSourceView = *sourceView;
        }

        Value resident = residentBySlot.lookup(slot);
        if (strictResidentContract && resident) {
          DictionaryAttr priorSourceView = runtimeViewBySlot.lookup(slot);
          if (!priorSourceView || priorSourceView != strictSourceView) {
            op.emitError(
                "strict resident runtime weight slot has conflicting source "
                "views");
            return signalPassFailure();
          }
        }
        // One resident per slot: if the same argument resolved to a different
        // whole shape here the two would alias, so keep this op's bridge.
        if (resident && cast<MemRefType>(resident.getType()) != residentType) {
          if (strictResidentContract) {
            op.emitError(
                "strict resident runtime weight slot has conflicting shapes");
            return signalPassFailure();
          }
          describeWeightDecline(
              op, slot,
              "this slot already has a resident of a different shape, and one "
              "slot is one buffer");
          continue;
        }
        if (!resident) {
          int64_t bytes = byteSize(residentType);
          std::optional<std::string> strictSite;
          uint64_t identityKey = 0;
          if (strictResidentContract) {
            std::optional<int64_t> checkedBytes = checkedByteSize(residentType);
            strictSite = residentSite(op.getOperation());
            std::string principal = residentPrincipalName(module);
            if (!checkedBytes || !strictSite ||
                residentFunctionIdentity(principal, func.getSymName()) == 0) {
              op.emitError(
                  "strict resident weight has no exact principal/function/site "
                  "identity");
              return signalPassFailure();
            }
            bytes = *checkedBytes;
            identityKey =
                residentSiteIdentity(principal, func.getSymName(), *strictSite,
                                     "weight-resident", slot);
            if (identityKey == 0) {
              op.emitError("strict resident weight site identity is zero");
              return signalPassFailure();
            }
          }
          // --- Placement gate: VTCM if it fits, the DDR mirror if it does not. ---
          //
          // The pool is a single 8 MiB of `HmxTarget::defaultVtcmBudget`,
          // shared with the kernel's own crouton arrays and accumulators, and
          // the device's `requireAllocationResult` is fail-closed -- an
          // allocation that does not fit aborts the whole PD at load time.
          // Deciding here is what turns that into a compile-time outcome.
          //
          // What happens next depends on which of the two sources this is,
          // because only one of them has somewhere else to go:
          //
          //   * A runtime weight has a second home: the permanent DDR mirror
          //     the host's pre-pack image is copied into once, from which the
          //     kernel then takes each block it needs by a contiguous fetch.
          //     It costs a copy the engine waits for instead of the strided
          //     per-launch pack the bridge paid, and it costs no VTCM at all,
          //     so a weight the pool cannot hold is still worth making
          //     resident. That is the placement below: the capacity number
          //     chooses *where* the weight lives, not whether it lives.
          //   * A constant has nowhere to go (see the gate above): its only
          //     legal form *is* the resident VTCM buffer, so an overrun there
          //     is reported as an error instead.
          //
          // A remark rather than an error even under the strict resident
          // contract, which turns every *other* decline in this pass into a
          // failure. Those declines mean the pass cannot certify a provenance
          // it was asked to certify; running out of pool is a legitimate
          // outcome of a correct pipeline, and a legitimate outcome must not
          // stop a kernel that is otherwise fine.
          FailureOr<ResidentVtcmAdmission> admission =
              admitResidentVtcm(func, module, addedBytes, bytes);
          if (failed(admission))
            return signalPassFailure();
          const bool ddr = !admission->admitted;
          if (ddr)
            describeVtcmDdrPlacement(op, slot, *admission);
          // Where the buffer is *declared*. The engine operand is the matmul's
          // own position; a DDR mirror has a second reader, the fetch that
          // replaces the pack, and that one sits where the pack loop sits --
          // earlier in the block, because the pack fed the matmul. Declaring it
          // there is what lets the fetch reference it without a use-before-def.
          Operation *residentAt = op.getOperation();
          if (ddr)
            residentAt = pack->loops.front();
          rewriter.setInsertionPoint(residentAt);
          // The residency key: the argument's aligned pointer. It is passed as
          // an operand (the allocation itself stays static; the verifier allows
          // this one resident-only operand) so the lowering reads it after the
          // function conversion, when a block argument is no longer a single
          // descriptor. Read it off the argument itself: the view proved
          // identical above, and the argument survives lowering more robustly.
          Value address = memref::ExtractAlignedPointerAsIndexOp::create(
              rewriter, op.getLoc(), arg);
          // The buffer's own placement. A DDR resident is an ordinary DDR
          // allocation (no memory space attribute) carrying the location fact;
          // a VTCM resident is byte-for-byte what it was before placement
          // existed, descriptor included.
          MemRefType bufferType =
              ddr ? MemRefType::get(residentType.getShape(),
                                   residentType.getElementType(), AffineMap{})
                  : residentType;
          auto alloc = hexagonmem::AllocOp::create(rewriter, op.getLoc(),
                                                   bufferType,
                                                   ValueRange{address},
                                                   rewriter.getI64IntegerAttr(128));
          NamedAttrList residentFields;
          residentFields.append(kResidentKeyAddress, rewriter.getUnitAttr());
          residentFields.append(kResidentKeyBytes,
                                rewriter.getI64IntegerAttr(bytes));
          if (ddr)
            residentFields.append(
                StringAttr::get(rewriter.getContext(),
                                kHmxWeightResidentLocationKey),
                StringAttr::get(rewriter.getContext(),
                                kHmxWeightResidentLocationDdr));
          alloc->setAttr(kResidentAttr,
                         residentFields.getDictionary(rewriter.getContext()));
          if (strictResidentContract) {
            DictionaryAttr expected = makeWeightProvenance(
                func.getContext(), residentPrincipalName(module),
                func.getSymName(), *strictSite, slot, bytes,
                /*alignment=*/128, identityKey, "argument-address",
                "entry-argument-slot:" + std::to_string(slot),
                kHmxResidentNotProven, "not-carried-to-device",
                strictSourceView);
            if (failed(setOrValidateResidentProvenance(alloc.getOperation(),
                                                       expected,
                                                       "resident weight")))
              return signalPassFailure();
            if (failed(validateExistingWeightResident(alloc.getOperation())))
              return signalPassFailure();
          }
          resident = alloc.getResult();
          residentBySlot.insert({slot, resident});
          if (strictResidentContract)
            runtimeViewBySlot[slot] = strictSourceView;
          // The module aggregate is the *pool* footprint the budget readers
          // compare against; a DDR mirror does not come out of it, so only a
          // VTCM placement is added.
          if (!ddr) {
            if (strictResidentContract &&
                bytes > std::numeric_limits<int64_t>::max() - addedBytes) {
              op.emitError("resident byte aggregate overflows int64");
              return signalPassFailure();
            }
            addedBytes += bytes;
          }

          // Publish the pre-pack contract: the host packs the whole argument,
          // and the permutation comes from the compiler's own layout map. The
          // logical shape is reconstructed from the crouton grid -- the whole
          // `[K, N]` for an N-slice, the argument exactly for a dense weight.
          // A weight grid is [Nt, Kt, ...], so K is `weightKTiles` and N is
          // `logicalN`.
          SmallVector<int64_t> logical{
              hmx::weightKTiles(residentType) * hmx::layout::kTileEdge,
              logicalN};
          // `dtype` names the *source* argument's element type, which is what
          // the host validates the runtime tensor against; the packed image is
          // always the fp16 crouton (`crouton`). The two coincide for an f16
          // weight and differ for an f32 one, where the host quantises.
          std::string sourceDtype =
              dtype::isF32(srcMemref.getElementType()) ? "f32" : "f16";
          // `location` is a fact about this weight, not a host instruction --
          // the image bytes are the same either way, and the host writes them
          // the same way -- but it is the field that lets a reader of the
          // contract tell a pool-resident weight from a mirror one without
          // going back to the IR.
          std::string entry =
              "{\"func\":\"" + func.getSymName().str() + "\",\"slot\":" +
              std::to_string(slot) + ",\"shape\":" + jsonArray(logical) +
              ",\"crouton\":" + jsonArray(residentType.getShape()) +
              ",\"dtype\":\"" + sourceDtype + "\",\"location\":\"" +
              (ddr ? kHmxWeightResidentLocationDdr.str()
                   : kHmxWeightResidentLocationVtcm.str()) +
              "\"}";
          appendPrepackEntry(module, entry);
          if (!module->getAttrOfType<StringAttr>(kPrepackLayoutAttr))
            module->setAttr(kPrepackLayoutAttr,
                            rewriter.getStringAttr(prepackLayoutJson()));
          LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] resident runtime slot "
                                  << slot << " (" << bytes << " bytes, "
                                  << (ddr ? "ddr" : "vtcm") << ")\n");
        }

        // Where this slot's buffer lives was decided when it was created;
        // every later consumer of the same slot follows that decision, because
        // one slot is one buffer.
        const bool ddrResident =
            !hexagon::isInVTCMAddressSpace(cast<MemRefType>(resident.getType()));
        // The block this program reads: the resident itself for a dense whole
        // argument, or a subview over the crouton grid's N tiles for an
        // N-slice. A weight grid is [Nt, Kt, ...] (dim0 = N, dim1 = K), so the
        // N block offset lands on dim0. The insertion point must already be
        // where this value has to be defined: at the matmul for the engine
        // operand, at the bridge for the fetch.
        auto blockView = [&](Value base) -> Value {
          if (!slice)
            return base;
          // The block's N offset in croutons: a static offset was proven
          // tile-aligned when it matched, a dynamic one is divided (exactly, by
          // the same proof).
          Value n0Crouton;
          if (slice->dynamicOffset) {
            Value edge = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                        hmx::layout::kTileEdge);
            n0Crouton = arith::DivUIOp::create(rewriter, op.getLoc(),
                                               slice->dynamicOffset, edge);
          } else {
            n0Crouton = arith::ConstantIndexOp::create(
                rewriter, op.getLoc(), *slice->staticOffset / hmx::layout::kTileEdge);
          }
          SmallVector<OpFoldResult> offsets{n0Crouton, rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(0)};
          SmallVector<OpFoldResult> sizes{
              rewriter.getIndexAttr(type.getDimSize(0)),
              rewriter.getIndexAttr(type.getDimSize(1)),
              rewriter.getIndexAttr(type.getDimSize(2)),
              rewriter.getIndexAttr(type.getDimSize(3)),
              rewriter.getIndexAttr(type.getDimSize(4))};
          SmallVector<OpFoldResult> strides(5, rewriter.getIndexAttr(1));
          return memref::SubViewOp::create(rewriter, op.getLoc(), base,
                                           offsets, sizes, strides);
        };

        if (ddrResident) {
          // The mirror lives in DDR, the engine reads VTCM, and a crouton grid
          // row (a whole range of N tiles) is contiguous in the mirror because
          // N is its outermost dimension. So the bridge's pack becomes one
          // contiguous copy at the pack's own program point: same writes, same
          // place in the loop, no strided scatter and no device-side
          // permutation -- the host already permuted the bytes. Blocking, per
          // §7-5 of the plan: an asynchronous stage would *replace* this copy,
          // not sit beside it.
          Operation *at = pack->loops.front();
          rewriter.setInsertionPoint(at);
          Value source = blockView(resident);
          memref::CopyOp::create(rewriter, op.getLoc(), source,
                                 pack->array.getResult());
          // The matmul keeps reading the buffer the bridge used to fill; for
          // the carried-loop form of the bridge that is the loop's init value,
          // so it has to be re-pointed before the loop goes away.
          rewriter.modifyOpInPlace(op, [&]() {
            operand.set(pack->array.getResult());
          });
        } else {
          rewriter.setInsertionPoint(op);
          rewriter.modifyOpInPlace(op, [&]() { operand.set(blockView(resident)); });
        }

        // The bridge is now dead: drop the pack loop and, for a weight whose
        // resident took the bridge's place in the engine's operand, the
        // crouton array and its deallocation too. Everything was verified to
        // be the bridge before this point, so nothing else can observe those.
        // A DDR-resident weight keeps its array -- the fetch copies *into* it
        // every iteration -- so it keeps the deallocation that pairs with it
        // as well. The pack ops go with their loop when it is the bridge and
        // nothing else.
        for (scf::ForOp loop : pack->loops)
          if (loop.use_empty())
            rewriter.eraseOp(loop);
        // The view the bridge read goes with it when it is dead, and a
        // tail-guarded one must go: its arms allocate and copy a dense staging
        // buffer, so leaving it behind would pay a masked gather per (m, n) for
        // a value nothing reads any more -- the exact cost this pass exists to
        // remove. A bare `reinterpret_cast` is pure and cheap, so it is left to
        // the canonicalizer rather than erased here.
        if (auto guard = src.getDefiningOp<scf::IfOp>(); guard && guard->use_empty())
          rewriter.eraseOp(guard);
        if (ddrResident)
          continue;
        SmallVector<memref::DeallocOp> deallocs;
        for (Operation *user : pack->array->getUsers())
          if (auto d = dyn_cast<memref::DeallocOp>(user))
            deallocs.push_back(d);
        for (memref::DeallocOp d : deallocs)
          rewriter.eraseOp(d);
        if (pack->array->use_empty())
          rewriter.eraseOp(pack->array);
      }
    }

    if (addedBytes) {
      if (strictResidentContract) {
        if (failed(addCheckedResidentBytes(module, addedBytes)))
          return signalPassFailure();
      } else {
        addCompatibilityResidentBytes(module, addedBytes);
      }
    }
    if (strictResidentContract &&
        (addedBytes || module->hasAttr(kResidentBytesAttr)) &&
        failed(verifyResidentByteAggregate(module)))
      return signalPassFailure();
    if (module->hasAttr("hmx.kernel_manifest") &&
        failed(reconcileHmxManifestWeightPolicies(module, prepackRuntimeWeights)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hmx::createWeightResidentPass(
    const mlir::hmx::WeightResidentOptions &options) {
  return std::make_unique<WeightResidentPass>(options);
}
