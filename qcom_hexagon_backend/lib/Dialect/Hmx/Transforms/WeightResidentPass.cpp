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
// pass computes). Both static VTCM-budget readers (`matmul-to-hmx`,
// `hmx-partition`) consult it, and the runtime receives the same number on the
// lowering's call, so the static budget and the resident footprint cannot
// drift apart.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

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
constexpr const char *kResidentAttr = "hmx.weight_resident";
constexpr const char *kResidentKeyGlobal = "global";
constexpr const char *kResidentKeyAddress = "address";
constexpr const char *kResidentKeyBytes = "bytes";

/// Aggregate of every resident buffer in the module, in bytes.
constexpr const char *kResidentBytesAttr = "hmx.weight_resident_bytes";

/// JSON array of the runtime weights the host must pre-pack: each entry names
/// the function, the argument slot, the logical shape and the crouton shape.
constexpr const char *kPrepackAttr = "hmx.weight_prepack";

/// The module attribute carrying the permutation the host must apply.
constexpr const char *kPrepackLayoutAttr = "hmx.weight_prepack_layout";

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

static int64_t residentAlignment(hexagonmem::AllocOp alloc) {
  if (auto attr = alloc->getAttrOfType<IntegerAttr>("alignment"))
    return attr.getInt();
  return 128;
}

static bool isResidentAliasLike(Operation *op) {
  return isa<memref::AssumeAlignmentOp, memref::CastOp, memref::SubViewOp,
             memref::ReinterpretCastOp, memref::MemorySpaceCastOp,
             memref::TransposeOp, memref::ViewOp, memref::ExpandShapeOp,
             memref::CollapseShapeOp, memref::ReshapeOp>(op);
}

/// Validate a transient pack array before its bridge is removed.  A direct
/// dealloc is unambiguous; a dealloc through a view/cast is not evidence that
/// the principal array can be dropped safely.
static LogicalResult validateResidentDealloc(Value value, Operation *anchor,
                                             unsigned &directCount,
                                             bool &aliasDealloc) {
  SmallVector<Value> worklist{value};
  SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (isa<CallOpInterface, func::ReturnOp, cf::BranchOp, cf::CondBranchOp,
              cf::SwitchOp, scf::YieldOp,
              memref::ExtractAlignedPointerAsIndexOp>(owner))
        return anchor->emitError(
            "resident weight bridge value escapes through a call/control-flow/"
            "pointer operation");
      if (isa<memref::DeallocOp, hexagonmem::DeallocOp>(owner)) {
        if (current == value)
          ++directCount;
        else
          aliasDealloc = true;
        continue;
      }
      if (isResidentAliasLike(owner))
        for (Value result : owner->getResults())
          worklist.push_back(result);
    }
  }
  if (aliasDealloc)
    return anchor->emitError("resident weight bridge has a deallocation "
                             "through a memref alias/view");
  return success();
}

static DictionaryAttr
makeWeightProvenance(MLIRContext *context, StringRef principal,
                     StringRef function, StringRef site, int64_t slot,
                     int64_t bytes, int64_t alignment, uint64_t identityKey,
                     StringRef runtimeKeyKind, StringRef source,
                     StringRef contentStatus, StringRef contentIdentity,
                     DictionaryAttr sourceView = {}) {
  NamedAttrList fields;
  fields.append("schema", StringAttr::get(context, kHmxResidentKeySchema));
  fields.append("key_namespace",
                StringAttr::get(context, kHmxResidentKeyNamespace));
  fields.append("kind", StringAttr::get(context, "weight"));
  fields.append("identity_key",
                IntegerAttr::get(IntegerType::get(context, 64), identityKey));
  fields.append("runtime_key_kind", StringAttr::get(context, runtimeKeyKind));
  fields.append("bytes",
                IntegerAttr::get(IntegerType::get(context, 64), bytes));
  fields.append("alignment",
                IntegerAttr::get(IntegerType::get(context, 64), alignment));
  fields.append("module", StringAttr::get(context, principal));
  fields.append("principal_status",
                StringAttr::get(context, principal == "<anonymous-principal>"
                                             ? kHmxResidentNotProven
                                             : "module-symbol"));
  fields.append("function", StringAttr::get(context, function));
  fields.append("function_id", IntegerAttr::get(IntegerType::get(context, 64),
                                                residentFunctionIdentity(
                                                    principal, function)));
  fields.append("role", StringAttr::get(context, "weight-resident"));
  fields.append("site", StringAttr::get(context, site));
  fields.append("site_id", IntegerAttr::get(
                               IntegerType::get(context, 64),
                               residentSiteIdentity(principal, function, site,
                                                    "weight-resident", slot)));
  fields.append("slot", IntegerAttr::get(IntegerType::get(context, 64), slot));
  fields.append("source", StringAttr::get(context, source));
  if (sourceView)
    fields.append("source_view", sourceView);
  fields.append("scope", StringAttr::get(context, kHmxResidentScope));
  fields.append("launch_status",
                StringAttr::get(context, kHmxResidentNotProven));
  fields.append("content_status", StringAttr::get(context, contentStatus));
  fields.append("content_identity", StringAttr::get(context, contentIdentity));
  fields.append("address_reuse_status",
                StringAttr::get(context, runtimeKeyKind == "argument-address"
                                             ? kHmxResidentNotProven
                                             : "immutable-source"));
  fields.append("reuse_status", StringAttr::get(context, "process-resident"));
  fields.append("descriptor_status", StringAttr::get(context, "checked"));
  return DictionaryAttr::get(context, fields);
}

static LogicalResult setOrValidateWeightProvenance(Operation *operation,
                                                   DictionaryAttr expected) {
  auto existing =
      operation->getAttrOfType<DictionaryAttr>(kHmxResidentProvenanceAttr);
  if (!existing) {
    operation->setAttr(kHmxResidentProvenanceAttr, expected);
    return success();
  }
  if (existing != expected)
    return operation->emitError(
        "resident weight provenance changed across repeated lowering");
  return success();
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
         sourceType.getElementType().isF16() &&
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
  int64_t total = bytes;
  if (auto existing = module->getAttrOfType<IntegerAttr>(kResidentBytesAttr)) {
    if (!existing.getType().isInteger(64) || existing.getInt() < 0 ||
        existing.getInt() > std::numeric_limits<int64_t>::max() - total)
      return module.emitError("resident byte aggregate overflows int64");
    total += existing.getInt();
  }
  module->setAttr(
      kResidentBytesAttr,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), total));
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
/// a program-id offset takes (`pid * BN`, sums of such). The subview that reads
/// one N block of the resident weight is indexed in croutons, so the block
/// offset has to land on a tile edge; anything not provably a multiple keeps the
/// per-launch bridge rather than truncating a division.
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
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
      !type.getElementType().isF16())
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
        !ranked.getElementType().isF16() || !ranked.getLayout().isIdentity() ||
        ranked.getShape() != ArrayRef<int64_t>{k, wholeN})
      return reject();
  } else if (auto unranked = dyn_cast<UnrankedMemRefType>(argument.getType())) {
    if (!unranked.getElementType().isF16())
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

/// Match the bridge source as `reinterpret_cast(arg, offset=[n0], sizes=[K, BN],
/// strides=[N, 1])` over an entry argument: one N block of a `[K, N]` weight.
/// Returns null for anything whose whole weight cannot be pinned -- not a 2D
/// row-major view, a non-argument source, a dynamic/unaligned offset, a view the
/// crouton grid disagrees with, or a block that does not tile N. The caller then
/// keeps the old per-launch bridge: guessing would silently pack the wrong bytes.
static std::optional<WeightSlice> underlyingSliceArgument(Value v,
                                                          MemRefType crouton) {
  auto reinterpret = v.getDefiningOp<memref::ReinterpretCastOp>();
  if (!reinterpret)
    return std::nullopt;
  auto viewType = dyn_cast<MemRefType>(v.getType());
  if (!viewType || viewType.getRank() != 2 || !viewType.hasStaticShape())
    return std::nullopt;
  // The pre-pack contract is a function-argument contract: the source must be
  // the entry argument itself, not an internal buffer.
  auto arg = dyn_cast<BlockArgument>(reinterpret.getSource());
  if (!arg)
    return std::nullopt;
  // A row-major view with unit inner stride: the underlying matrix is [K, N]
  // with N = stride(0), and the view is one N block of it.
  auto strided = dyn_cast<StridedLayoutAttr>(viewType.getLayout());
  if (!strided)
    return std::nullopt;
  auto strides = strided.getStrides();
  if (strides.size() != 2 || strides[1] != 1 ||
      ShapedType::isDynamic(strides[0]) || strides[0] <= 0)
    return std::nullopt;
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
    return std::nullopt;
  // Strictly narrower than the whole N: the model is "the view is *one* N block
  // of a wider weight", so a view as wide as the whole N is a block of nothing
  // (there are no other blocks for an offset to select) and any offset into it
  // is not an N offset -- it is the K-block case the offset check below rejects.
  // One N block means at least two.
  if (n % hmx::layout::kTileEdge != 0 || bn % hmx::layout::kTileEdge != 0 || n <= bn ||
      n % bn != 0)
    return std::nullopt;
  // The offset is the descriptor's element offset (a one-element list), i.e. the
  // N block this program owns. A static offset is checked directly; a dynamic
  // one has to be provably tile-aligned (`pid * BN`) *and* provably a column
  // offset: in a row-major [K, N] matrix every whole number of rows is a
  // multiple of N, so an offset that is provably a multiple of N is a row (K)
  // offset -- a loop-varying block of an activation consumed as a weight, whose
  // resident holds one block while the offset walks past its end. A column
  // offset is never such a multiple (offset 0 is the dense path's business).
  if (reinterpret.getStaticOffsets().size() != 1)
    return std::nullopt;
  WeightSlice slice;
  slice.arg = arg;
  slice.n = n;
  int64_t staticOffset = reinterpret.getStaticOffsets().front();
  if (staticOffset != ShapedType::kDynamic) {
    if (staticOffset < 0 || staticOffset % hmx::layout::kTileEdge != 0 ||
        staticOffset + bn > n)
      return std::nullopt;
    slice.staticOffset = staticOffset;
  } else {
    if (reinterpret.getOffsets().size() != 1)
      return std::nullopt;
    Value offset = reinterpret.getOffsets().front();
    if (!isMultipleOf(offset, hmx::layout::kTileEdge) || isMultipleOf(offset, n))
      return std::nullopt;
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
/// memref, but it must be f16, have an exact static view, and use either a
/// zero/dense view or a statically fixed N slice.  A dynamic offset is not a
/// content identity: it can select a different object on each invocation.
static LogicalResult
validateStrictRuntimeSource(Value src, MemRefType crouton, func::FuncOp func,
                            Operation *anchor, int64_t *slotOut,
                            DictionaryAttr *sourceViewOut = nullptr) {
  auto srcMemref = dyn_cast<MemRefType>(src.getType());
  if (!srcMemref || !srcMemref.getElementType().isF16())
    return anchor->emitError(
        "strict resident runtime weight source must be an f16 memref view");

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
    auto strided = cast<StridedLayoutAttr>(srcMemref.getLayout());
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
        !argumentType.getElementType().isF16() ||
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
             !srcMemref.getElementType().isF16() || srcMemref.getRank() != 2) {
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
    return anchor->emitError("strict resident weight bridge has no pack");
  Value source = pack.packs.front().getSrc();
  for (PackWeightOp writer : pack.packs)
    if (writer.getSrc() != source)
      return anchor->emitError(
          "strict resident weight bridge has multiple source views");
  unsigned directDeallocs = 0;
  bool aliasDealloc = false;
  if (failed(validateResidentDealloc(pack.array.getResult(), anchor,
                                     directDeallocs, aliasDealloc)))
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
  if (failed(validateResidentDealloc(alloc.getResult(), operation,
                                     directDeallocs, aliasDealloc)) ||
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
          "strict runtime resident weight is not an entry f16 argument");
    auto argumentType = dyn_cast<MemRefType>(argument.getType());
    bool unrankedArgument = isa<UnrankedMemRefType>(argument.getType());
    if (type.getRank() != 5 || type.getDimSize(2) != 16 ||
        type.getDimSize(3) != 32 || type.getDimSize(4) != 2)
      return anchor->emitError(
          "strict runtime resident weight has an invalid crouton type");
    const int64_t logicalK = hmx::weightKTiles(type) * hmx::layout::kTileEdge;
    const int64_t logicalN = hmx::weightNTiles(type) * hmx::layout::kTileEdge;
    if (!sourceType || !sourceType.getElementType().isF16() ||
        argument.getOwner() != &parentFunction.getBody().front())
      return anchor->emitError(
          "strict runtime resident weight is not an entry f16 argument");
    if (argumentType) {
      if (!argumentType.hasStaticShape() || argumentType.getRank() != 2 ||
          !argumentType.getElementType().isF16() ||
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
    expectedResident = DictionaryAttr::get(
        context,
        {NamedAttribute(StringAttr::get(context, kResidentKeyAddress), address),
         NamedAttribute(
             StringAttr::get(context, kResidentKeyBytes),
             IntegerAttr::get(IntegerType::get(context, 64), *bytes))});
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
            op.emitError(
                "strict resident runtime weight source has no pack bridge");
            result = failure();
          }
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
              if (failed(setOrValidateWeightProvenance(alloc.getOperation(),
                                                       expected)))
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
        if (!pack)
          continue;
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
          continue;
        }
        // The pre-pack contract is defined on an fp16 weight: the host permutes
        // the weight's own bytes, so the resident holds exactly what this
        // device-side pack would have written. A wider weight would need the
        // host to quantise it to the same fp16 the crouton holds, which is a
        // different contract -- leave such a weight on the per-launch bridge,
        // whose pack does quantise (see hmx.pack_weight).
        auto srcMemref = dyn_cast<MemRefType>(src.getType());

        BlockArgument arg;
        MemRefType residentType;
        int64_t logicalN = hmx::weightNTiles(type) * hmx::layout::kTileEdge;
        std::optional<WeightSlice> slice;
        if (BlockArgument denseArg = underlyingDenseArgument(src)) {
          arg = denseArg;
          residentType = MemRefType::get(type.getShape(), type.getElementType(),
                                         AffineMap{},
                                         hexagon::VTCM_ADDRESS_SPACE);
        } else if ((slice = underlyingSliceArgument(src, type))) {
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
        } else {
          if (strictResidentContract) {
            op.emitError(
                "strict resident runtime weight source is not an exact entry "
                "view");
            return signalPassFailure();
          }
          continue;
        }
        if (arg.getOwner() != &func.getBody().front()) {
          if (strictResidentContract) {
            op.emitError(
                "strict resident runtime weight source is not an entry argument");
            return signalPassFailure();
          }
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
        // The manifest records the argument binding even when the source is f32
        // or otherwise not eligible for resident prepack; the policy pass then
        // gives that slot the canonical device-pack reason.
        if (!srcMemref || !srcMemref.getElementType().isF16()) {
          if (strictResidentContract) {
            op.emitError("strict resident runtime weight source must be f16");
            return signalPassFailure();
          }
          continue;
        }

        if (strictResidentContract) {
          unsigned directDeallocs = 0;
          bool aliasDealloc = false;
          if (failed(validateResidentDealloc(pack->array.getResult(),
                                             op.getOperation(), directDeallocs,
                                             aliasDealloc)))
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
          rewriter.setInsertionPoint(op);
          // The residency key: the argument's aligned pointer. It is passed as
          // an operand (the allocation itself stays static; the verifier allows
          // this one resident-only operand) so the lowering reads it after the
          // function conversion, when a block argument is no longer a single
          // descriptor. Read it off the argument itself: the view proved
          // identical above, and the argument survives lowering more robustly.
          Value address = memref::ExtractAlignedPointerAsIndexOp::create(
              rewriter, op.getLoc(), arg);
          auto alloc = hexagonmem::AllocOp::create(
              rewriter, op.getLoc(), residentType, ValueRange{address},
              rewriter.getI64IntegerAttr(128));
          alloc->setAttr(
              kResidentAttr,
              rewriter.getDictionaryAttr(
                  {rewriter.getNamedAttr(kResidentKeyAddress,
                                         rewriter.getUnitAttr()),
                   rewriter.getNamedAttr(kResidentKeyBytes,
                                         rewriter.getI64IntegerAttr(bytes))}));
          if (strictResidentContract) {
            DictionaryAttr expected = makeWeightProvenance(
                func.getContext(), residentPrincipalName(module),
                func.getSymName(), *strictSite, slot, bytes,
                /*alignment=*/128, identityKey, "argument-address",
                "entry-argument-slot:" + std::to_string(slot),
                kHmxResidentNotProven, "not-carried-to-device",
                strictSourceView);
            if (failed(setOrValidateWeightProvenance(alloc.getOperation(),
                                                     expected)))
              return signalPassFailure();
            if (failed(validateExistingWeightResident(alloc.getOperation())))
              return signalPassFailure();
          }
          resident = alloc.getResult();
          residentBySlot.insert({slot, resident});
          if (strictResidentContract)
            runtimeViewBySlot[slot] = strictSourceView;
          if (strictResidentContract &&
              bytes > std::numeric_limits<int64_t>::max() - addedBytes) {
            op.emitError("resident byte aggregate overflows int64");
            return signalPassFailure();
          }
          addedBytes += bytes;

          // Publish the pre-pack contract: the host packs the whole argument,
          // and the permutation comes from the compiler's own layout map. The
          // logical shape is reconstructed from the crouton grid -- the whole
          // `[K, N]` for an N-slice, the argument exactly for a dense weight.
          // A weight grid is [Nt, Kt, ...], so K is `weightKTiles` and N is
          // `logicalN`.
          SmallVector<int64_t> logical{
              hmx::weightKTiles(residentType) * hmx::layout::kTileEdge,
              logicalN};
          std::string entry =
              "{\"func\":\"" + func.getSymName().str() + "\",\"slot\":" +
              std::to_string(slot) + ",\"shape\":" + jsonArray(logical) +
              ",\"crouton\":" + jsonArray(residentType.getShape()) +
              ",\"dtype\":\"f16\"}";
          appendPrepackEntry(module, entry);
          if (!module->getAttrOfType<StringAttr>(kPrepackLayoutAttr))
            module->setAttr(kPrepackLayoutAttr,
                            rewriter.getStringAttr(prepackLayoutJson()));
          LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] resident runtime slot "
                                  << slot << " (" << bytes << " bytes)\n");
        }

        // The engine reads the block this program owns: the resident itself for a
        // dense whole argument, or a subview over the crouton grid's N tiles for
        // an N-slice. A weight grid is [Nt, Kt, ...] (dim0 = N, dim1 = K), so the
        // N block offset lands on dim0.
        Value rhs = resident;
        if (slice) {
          rewriter.setInsertionPoint(op);
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
          rhs = memref::SubViewOp::create(rewriter, op.getLoc(), resident,
                                          offsets, sizes, strides);
        }
        rewriter.modifyOpInPlace(op, [&]() { operand.set(rhs); });

        // The bridge is now dead: drop the pack loop, the crouton array and its
        // deallocation. Everything was verified to be the bridge before this
        // point, so nothing else can observe the buffer. The pack ops go with
        // their loop when it is the bridge and nothing else.
        for (scf::ForOp loop : pack->loops)
          if (loop.use_empty())
            rewriter.eraseOp(loop);
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
