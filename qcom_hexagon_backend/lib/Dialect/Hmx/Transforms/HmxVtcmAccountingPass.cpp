//===-- HmxVtcmAccountingPass.cpp - diagnostic VTCM allocation census -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This pass is intentionally an internal diagnostic. It publishes a raw
// allocation-site census, a compiler-static site-identity sidecar, and, under
// a second explicit marker, a separate structured allocator-event result. The
// latter proves only a narrow subset:
// single-block function bodies with recursively balanced single-block
// scf.if/scf.for/scf.execute_region regions, statically sized allocations,
// direct deallocation, and reviewed synchronous operations. A finite-CFG
// scf.while is handled by the before/after fixpoint; unsupported while/CFG
// shapes, carried or escaping allocation values, calls, DMA/unknown effects,
// and incomplete site coverage fail closed. None of these results claims runtime
// capacity, changes manifest v2, launcher behavior, allocation placement, or
// production tail selection.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Crouton/IR/CroutonDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxVtcmAccounting.h"
#include "hexagon/Dialect/Hmx/Transforms/Passes.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_HMXVTCMACCOUNTING
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

constexpr StringLiteral kWeightResidentAttr = "hmx.weight_resident";
constexpr StringLiteral kWorkspaceResidentAttr = "hmx.workspace_resident";
constexpr StringLiteral kWeightResidentBytesAttr = "hmx.weight_resident_bytes";

constexpr StringLiteral kKeyKind = "kind";
constexpr StringLiteral kKeyStatus = "status";
constexpr StringLiteral kKeyAllocationSites = "allocation_sites";
constexpr StringLiteral kKeyUnknownAllocations = "unknown_allocations";
constexpr StringLiteral kKeyTransientBytes = "transient_bytes";
constexpr StringLiteral kKeyWorkspaceResidentBytes = "workspace_resident_bytes";
constexpr StringLiteral kKeyWeightResidentBytes = "weight_resident_bytes";
constexpr StringLiteral kKeyResidentSiteSumBytes = "resident_site_sum_bytes";
constexpr StringLiteral kKeyRawSiteSumBytes = "raw_site_sum_bytes";
constexpr StringLiteral kKeyPeakStatus = "peak_status";
constexpr StringLiteral kKeyExternalScratch = "external_scratch";
constexpr StringLiteral kKeyExternalVtcm = "external_vtcm";
constexpr StringLiteral kKindAllocationSiteCensus = "allocation-site-census";
constexpr StringLiteral kLivenessKindStructuredAllocatorEvents =
    "structured-allocator-events-v1";
/// Published control-flow subset.  The analysis is a finite worklist fixpoint
/// over the whole function CFG, so a multi-block body, a plain-CFG cycle, a
/// multi-block region, and `scf.while` are all covered; only a cycle that
/// carries a live allocation across an iteration is rejected.
constexpr StringLiteral kLivenessControlFlow =
    "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region";
/// Explicit, machine-readable admission policy per class.  Each string states
/// what a reader may conclude; a class with no reviewed summary stays closed.
constexpr StringLiteral kLivenessAliasPolicy =
    "reviewed-alias-lattice-unique-owner-required";
constexpr StringLiteral kLivenessCallPolicy = "no-summary-incomplete";
constexpr StringLiteral kLivenessAsyncPolicy = "no-summary-incomplete";
constexpr StringLiteral kLivenessExtentPolicy =
    "static-shape-or-proven-constant-upper-bound";
constexpr StringLiteral kLivenessJoinPolicy = "union-join-upper-bound";
constexpr StringLiteral kPeakSiteIdCanonical = "canonical-static-site-identity";
constexpr StringLiteral kStatusNotProven = "not-proven";
constexpr StringLiteral kPeakStatusNotProven = "not-proven";
constexpr StringLiteral kPeakStatusStructured = "structured-upper-bound";
constexpr StringLiteral kStatusComplete = "complete";
constexpr StringLiteral kStatusIncomplete = "incomplete";
constexpr StringLiteral kStatusNotApplicable = "not-applicable";
constexpr StringLiteral kIdentityKind = "static-vtcm-site-identity-v1";
constexpr StringLiteral kIdentityScope = "per-function-static-allocation-site";
constexpr StringLiteral kIdentitySource = "compiler-static";
constexpr StringLiteral kEvidenceContextKind = "vtcm-evidence-context-v1";
constexpr StringLiteral kEvidenceContextStatusIncomplete = "incomplete";
constexpr StringLiteral kEvidenceContextScope =
    "one-immutable-principal-module-function-canonical-site-grid1-single-invocation";
constexpr StringLiteral kEvidenceContextNotProven = "not-proven";
constexpr StringLiteral kEvidenceContextAggregate = "aggregate";
constexpr StringLiteral kRuntimeJoinNotIntegrated = "not-integrated";
constexpr StringLiteral kBuildIdExternallySupplied = "externally-supplied";
constexpr StringLiteral kBuildIdNotPresent = "not-present";
constexpr StringLiteral kBuildIdMalformed = "malformed-or-ambiguous";
constexpr StringLiteral kIdentityReasonMissingModule =
    "missing explicit module symbol";
constexpr StringLiteral kIdentityReasonMissingFunction =
    "missing or ambiguous function scope";
constexpr StringLiteral kIdentityReasonUnknownSource =
    "unknown source location";
constexpr StringLiteral kIdentityReasonDuplicateSite =
    "duplicate or ambiguous allocation-site identity";
constexpr StringLiteral kIdentityReasonFunctionCollision =
    "function identity hash collision";
constexpr StringLiteral kIdentityReasonSiteCollision =
    "allocation-site identity hash collision";
constexpr StringLiteral kIdentityReasonZeroHash =
    "static identity hash produced zero";
constexpr StringLiteral kIdentityReasonUnknownRole =
    "allocation role is not reviewed for identity";
constexpr StringLiteral kIdentityReasonResidentSeparate =
    "resident provenance is a separate runtime contract";
constexpr StringLiteral kIdentityReasonResidentProvenanceMissing =
    "resident provenance is missing or malformed";
constexpr StringLiteral kIdentityReasonResidentSource =
    "resident source descriptor is not statically proven";

enum class StaticIdentityStatus : uint8_t {
  kComplete,
  kIncomplete,
  kNotProven,
};

static StaticIdentityStatus mergeIdentityStatus(StaticIdentityStatus lhs,
                                                StaticIdentityStatus rhs) {
  if (lhs == StaticIdentityStatus::kNotProven ||
      rhs == StaticIdentityStatus::kNotProven)
    return StaticIdentityStatus::kNotProven;
  if (lhs == StaticIdentityStatus::kIncomplete ||
      rhs == StaticIdentityStatus::kIncomplete)
    return StaticIdentityStatus::kIncomplete;
  return StaticIdentityStatus::kComplete;
}

static StringRef staticIdentityStatus(StaticIdentityStatus status) {
  switch (status) {
  case StaticIdentityStatus::kComplete:
    return kStatusComplete;
  case StaticIdentityStatus::kIncomplete:
    return kStatusIncomplete;
  case StaticIdentityStatus::kNotProven:
    return kStatusNotProven;
  }
  llvm_unreachable("unknown static identity status");
}

// The runtime's crouton descriptor names one 2 KiB HMX tile; keep this in
// sync with CROUTON_SIZE rather than inferring bytes from the tensor shape.
constexpr int64_t kCroutonBytes = 2048;

static IntegerAttr u64Attribute(MLIRContext *context, uint64_t value) {
  return IntegerAttr::get(IntegerType::get(context, 64), APInt(64, value));
}

struct SizeResult {
  bool isVtcm = false;
  bool known = true;
  int64_t bytes = 0;
  /// True when `known` came from a dynamic-typed memref whose every extent was
  /// resolved through a proven constant upper bound rather than from a
  /// statically shaped type.  The byte figure is exact either way (a constant
  /// bound on an extent *is* that extent), but the distinction is published so
  /// a reader never mistakes a dynamically typed allocation for a static one.
  bool constantBoundedExtent = false;
};

struct AllocationSite {
  unsigned id = 0;
  Operation *operation = nullptr;
  Value value;
  int64_t bytes = 0;
  bool resident = false;
  bool constantBoundedExtent = false;
};

/// A source fact is intentionally independent of the operation's traversal
/// position.  A FusedLoc with more than one file location is ambiguous and is
/// rejected rather than selecting the first component.
struct StaticSourceFact {
  std::string canonical;
  std::string file;
  int64_t line = 0;
  int64_t column = 0;
};

/// A static site is collected independently of the raw census' ordinal IDs.
/// The role is a reviewed semantic classification (allocation op plus
/// resident marker); resident slots are read from the checked provenance
/// descriptor. There is deliberately no fallback to an allocation ordinal.
struct StaticAllocationSite {
  // These are transient IR handles used to attach facts. They are never
  // hashed, serialized, or used as identity components.
  Operation *operation = nullptr;
  Operation *function = nullptr;
  bool functionAmbiguous = false;
  std::string role;
  bool weightResident = false;
  bool workspaceResident = false;
  bool bytesKnown = false;
  int64_t bytes = 0;
  /// True when the size was proven through constant bounds on a dynamic-typed
  /// memref rather than from a statically shaped type.  Carried into the
  /// sidecar so a reader can tell an exact static size from an exact size that
  /// rests on a dynamic extent.
  bool constantBoundedExtent = false;
  /// Resident identity facts come from hmx.resident_provenance.  They are
  /// deliberately kept separate from the allocator census: a raw resident
  /// marker without a checked provenance record remains countable, but never
  /// receives a complete static site identity.
  int64_t slot = 0;
  bool slotProven = true;
  bool provenanceProven = true;
  std::string provenanceReason;
};

struct StaticFunctionIdentity {
  Operation *operation = nullptr;
  std::string symbol;
  uint64_t functionId = 0;
  bool functionIdValid = false;
  StaticIdentityStatus status = StaticIdentityStatus::kNotProven;
  bool ambiguous = false;
  std::string reason;
  SmallVector<unsigned> siteIndices;
};

struct StaticSiteIdentity {
  const StaticAllocationSite *site = nullptr;
  std::string principal;
  std::string function;
  std::optional<StaticSourceFact> source;
  std::string role;
  int64_t slot = 0;
  uint64_t functionId = 0;
  uint64_t siteId = 0;
  HmxDiagnosticEventToken eventToken{};
  bool functionIdValid = false;
  bool siteIdValid = false;
  StaticIdentityStatus status = StaticIdentityStatus::kComplete;
  std::string reason;
};

struct StaticIdentityResult {
  StaticIdentityStatus status = StaticIdentityStatus::kComplete;
  std::string principal;
  std::string module;
  std::string principalStatus;
  uint64_t scopeId = 0;
  Attribute buildId;
  std::string buildIdStatus;
  std::string buildIdSource;
  SmallVector<StaticFunctionIdentity> functions;
  SmallVector<StaticSiteIdentity, 0> sites;
  std::set<std::string> reasons;
};

struct Accounting {
  int64_t allocationSites = 0;
  int64_t unknownAllocations = 0;
  int64_t transientBytes = 0;
  int64_t workspaceResidentBytes = 0;
  int64_t weightResidentBytes = 0;
  int64_t declaredWeightResidentBytes = 0;
  SmallVector<AllocationSite> sites;
  std::unordered_set<uint64_t> workspaceKeys;
  bool hasDeclaredWeightBytes = false;
  bool externalScratch = false;
  std::string externalVtcm;
  bool complete = true;
};

struct FunctionLivenessFacts {
  std::string symbol;
  bool applicable = true;
  bool complete = true;
  bool coverageComplete = true;
  std::string reason;
  int64_t allocationSites = 0;
  int64_t deallocationSites = 0;
  int64_t transientPeakBytes = 0;
  int64_t workspaceResidentBytes = 0;
  int64_t weightResidentBytes = 0;
  int64_t modeledRequestedPeakBytes = 0;
  /// Canonical static site IDs of the sites charged at the peak.  These come
  /// from the versioned `hmx.vtcm-static-identity/v1` site identity, never
  /// from a census walk ordinal; when a peak site has no provable canonical
  /// identity the vector is withheld and `peakSiteIdsProven` stays false.
  SmallVector<int64_t> peakSiteIds;
  bool peakSiteIdsProven = false;
  /// Number of sites charged at the peak, published even when the identities
  /// themselves are not provable so the record never degrades to an
  /// unexplained empty list.
  int64_t peakSiteCount = 0;
  /// Sites whose requested size came from a constant bound on a dynamic extent
  /// rather than from a statically shaped type.
  int64_t constantBoundedSites = 0;
  /// Fixpoint evidence: worklist iterations consumed and blocks whose exit
  /// state had to be revised after the first computation.
  int64_t fixpointRounds = 0;
  int64_t revisedBlocks = 0;
  std::set<unsigned> siteIds;
};

struct LivenessAccounting {
  bool complete = true;
  bool coverageComplete = true;
  std::string reason;
  // FunctionLivenessFacts is large; keep the module record out of line.
  SmallVector<FunctionLivenessFacts, 0> functions;
};

static bool addBytes(int64_t &lhs, int64_t rhs) {
  if (rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs)
    return false;
  lhs += rhs;
  return true;
}

static bool getMemRefMemorySpace(BaseMemRefType type, int &memorySpace) {
  Attribute attr = type.getMemorySpace();
  if (!attr) {
    memorySpace = hexagon::DEFAULT_DDR_ADDRESS_SPACE;
    return true;
  }
  auto integer = dyn_cast<IntegerAttr>(attr);
  if (!integer)
    return false;
  memorySpace = static_cast<int>(integer.getInt());
  return true;
}

/// Element count for a VTCM memref.
///
/// Static dimensions are taken verbatim.  A dynamic dimension is admitted only
/// when it has a compile-time constant bound, so the product is still an exact
/// requested-byte figure; a bound that is merely "small in practice" (a block
/// argument, a clamped value, a shape read from a runtime buffer) is rejected so
/// the census reports the allocation as unproven instead of reporting a guessed
/// or zero byte count.  `extraOperandIsAddress` marks the reviewed resident
/// form where an operand is a provenance address rather than an extent, so it
/// must not be mistaken for a dynamic size.
static SizeResult getMemRefSize(BaseMemRefType type,
                                ValueRange dynamicSizes = {},
                                bool extraOperandIsAddress = false) {
  int memorySpace = hexagon::DEFAULT_DDR_ADDRESS_SPACE;
  if (!getMemRefMemorySpace(type, memorySpace))
    return {true, false, 0};
  if (memorySpace != hexagon::VTCM_ADDRESS_SPACE)
    return {};
  if (!type.hasRank())
    return {true, false, 0};
  Type element = type.getElementType();
  if (!element.isIntOrFloat() || isa<VectorType>(element) ||
      element.getIntOrFloatBitWidth() <= 0 ||
      element.getIntOrFloatBitWidth() % 8 != 0)
    return {true, false, 0};
  bool constantBoundedExtent = false;
  if (type.hasStaticShape()) {
    if (!dynamicSizes.empty() && !extraOperandIsAddress)
      return {true, false, 0};
  } else {
    // Every dynamic dimension needs its own proven operand, in declaration
    // order.  A mismatched count means the operands are not extents at all.
    if (dynamicSizes.size() != type.getNumDynamicDims())
      return {true, false, 0};
    constantBoundedExtent = true;
    for (Value size : dynamicSizes) {
      auto constant = size.getDefiningOp<arith::ConstantOp>();
      if (!constant)
        return {true, false, 0};
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      if (!integer)
        return {true, false, 0};
      Type constantType = integer.getType();
      if (!constantType.isIndex() && !constantType.isInteger())
        return {true, false, 0};
      if (constantType.isInteger() && constantType.getIntOrFloatBitWidth() > 64)
        return {true, false, 0};
      if (integer.getValue().getSExtValue() <= 0)
        return {true, false, 0};
    }
  }
  int64_t elements = 1;
  unsigned dynamicIndex = 0;
  for (int64_t extent : type.getShape()) {
    int64_t resolved = extent;
    if (ShapedType::isDynamic(extent)) {
      // Resolve this dynamic dimension from the next proven constant operand.
      auto constant =
          cast<arith::ConstantOp>(dynamicSizes[dynamicIndex++].getDefiningOp());
      resolved = cast<IntegerAttr>(constant.getValue()).getInt();
    }
    if (resolved <= 0)
      return {true, false, 0};
    if (elements > std::numeric_limits<int64_t>::max() / resolved)
      return {true, false, 0};
    elements *= resolved;
  }
  if (elements == 0)
    return {true, false, 0};
  int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
  if (elements > std::numeric_limits<int64_t>::max() / elementBytes)
    return {true, false, 0};
  return {true, true, elements * elementBytes, constantBoundedExtent};
}

static SizeResult getCroutonSize(crouton::CroutonType type) {
  if (!type.getVtcm().getValue())
    return {};
  int64_t elements = type.getNumElements();
  if (elements <= 0 ||
      elements > std::numeric_limits<int64_t>::max() / kCroutonBytes)
    return {true, false, 0};
  return {true, true, elements * kCroutonBytes};
}

static SizeResult getAllocationSize(Operation *operation) {
  if (auto alloc = dyn_cast<bufferization::AllocTensorOp>(operation)) {
    Attribute memorySpace = alloc.getMemorySpaceAttr();
    if (!memorySpace)
      return {};
    auto integerSpace = dyn_cast<IntegerAttr>(memorySpace);
    if (!integerSpace)
      return {true, false, 0};
    if (integerSpace.getInt() == hexagon::VTCM_ADDRESS_SPACE)
      return {true, false, 0};
    return {};
  }
  if (auto alloc = dyn_cast<memref::AllocOp>(operation)) {
    auto type = dyn_cast<BaseMemRefType>(alloc.getResult().getType());
    if (!type)
      return {};
    return getMemRefSize(type, alloc.getDynamicSizes());
  }
  if (auto alloc = dyn_cast<memref::AllocaOp>(operation)) {
    auto type = dyn_cast<BaseMemRefType>(alloc.getResult().getType());
    if (!type)
      return {};
    int memorySpace = hexagon::DEFAULT_DDR_ADDRESS_SPACE;
    if (!getMemRefMemorySpace(type, memorySpace))
      return {true, false, 0};
    if (memorySpace == hexagon::VTCM_ADDRESS_SPACE)
      return {true, false, 0};
    return {};
  }
  if (auto alloc = dyn_cast<hexagonmem::AllocOp>(operation)) {
    Type type = alloc.getResult().getType();
    // A runtime weight-resident allocation carries an extra address operand
    // next to a statically shaped result.  That operand is provenance for the
    // resident key, not an extent, so the byte figure still comes from the
    // type alone.
    bool extraOperandIsAddress = !alloc.getDynamicSizes().empty() &&
                                 operation->hasAttr(kWeightResidentAttr);
    SizeResult result;
    if (auto memrefType = dyn_cast<BaseMemRefType>(type))
      result =
          getMemRefSize(memrefType, alloc.getDynamicSizes(), extraOperandIsAddress);
    else if (auto croutonType = dyn_cast<crouton::CroutonType>(type))
      result = getCroutonSize(croutonType);
    else
      return {true, false, 0};
    if (!result.isVtcm)
      return {};
    if (!alloc.getDynamicSizes().empty() && !extraOperandIsAddress)
      return {true, false, 0};
    return result;
  }
  if (auto alloc = dyn_cast<AllocCroutonOp>(operation)) {
    auto type = cast<RankedTensorType>(alloc.getResult().getType());
    if (!type.hasStaticShape())
      return {true, false, 0};
    int64_t elements = type.getNumElements();
    if (elements <= 0 || elements > std::numeric_limits<int64_t>::max() / 2)
      return {true, false, 0};
    return {true, true, elements * 2};
  }
  return {};
}

static bool isSupportedAllocationOperation(Operation *operation) {
  return isa<bufferization::AllocTensorOp, memref::AllocOp, memref::AllocaOp,
             hexagonmem::AllocOp, AllocCroutonOp>(operation);
}

static bool isVTCMIdentityCandidate(Operation *operation) {
  if (!isSupportedAllocationOperation(operation) ||
      operation->getNumResults() == 0)
    return false;
  SizeResult size = getAllocationSize(operation);
  // A resident marker on a non-VTCM backing is a raw-census error, not a
  // VTCM allocator site. Do not give that malformed operation a plausible
  // join key in the static sidecar.
  return size.isVtcm;
}

static std::string getStaticAllocationRole(Operation *operation,
                                           bool weightResident,
                                           bool workspaceResident) {
  if (weightResident && workspaceResident)
    return "ambiguous-resident";
  if (weightResident)
    return "weight-resident";
  if (workspaceResident)
    return "workspace-resident";
  if (isa<AllocCroutonOp>(operation))
    return "crouton";
  if (isa<hexagonmem::AllocOp>(operation))
    return "transient-hexagonmem";
  if (isa<memref::AllocOp>(operation))
    return "transient-memref";
  if (isa<memref::AllocaOp>(operation))
    return "transient-alloca";
  if (isa<bufferization::AllocTensorOp>(operation))
    return "transient-tensor";
  return "unclassified-allocation";
}

static bool isReviewedStaticRole(StringRef role) {
  return role == "transient-memref" || role == "transient-hexagonmem" ||
         role == "crouton" || role == "weight-resident" ||
         role == "workspace-resident";
}

static Operation *getUniqueFunctionScope(Operation *operation,
                                         bool &ambiguous) {
  Operation *found = nullptr;
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (!isa<func::FuncOp>(parent))
      continue;
    if (found) {
      ambiguous = true;
      return nullptr;
    }
    found = parent;
  }
  return found;
}

static std::optional<StaticSourceFact> getStaticSourceFact(Location location) {
  // Reuse the resident contract's source walker so the accounting sidecar and
  // strict resident producers accept exactly the same debug-location shapes.
  // In particular, a fused location may contain a string label plus one file
  // location, but two file locations remain ambiguous.
  auto canonical = residentSite(location);
  if (!canonical)
    return std::nullopt;

  StringRef site = *canonical;
  if (!site.consume_front("file:"))
    return std::nullopt;
  size_t columnSeparator = site.rfind(':');
  if (columnSeparator == StringRef::npos || columnSeparator == 0)
    return std::nullopt;
  StringRef beforeColumn = site.substr(0, columnSeparator);
  size_t lineSeparator = beforeColumn.rfind(':');
  if (lineSeparator == StringRef::npos || lineSeparator == 0)
    return std::nullopt;

  StaticSourceFact fact;
  fact.file = beforeColumn.substr(0, lineSeparator).str();
  StringRef lineText = beforeColumn.substr(lineSeparator + 1);
  StringRef columnText = site.substr(columnSeparator + 1);
  unsigned long long line = 0;
  unsigned long long column = 0;
  bool parsed =
      !fact.file.empty() && !lineText.getAsInteger(10, line) &&
      !columnText.getAsInteger(10, column) && line != 0 && column != 0 &&
      line <= static_cast<unsigned long long>(
                  std::numeric_limits<int64_t>::max()) &&
      column <=
          static_cast<unsigned long long>(std::numeric_limits<int64_t>::max());
  if (!parsed)
    return std::nullopt;
  fact.line = static_cast<int64_t>(line);
  fact.column = static_cast<int64_t>(column);
  fact.canonical = *canonical;
  return fact;
}

static std::string makeStaticIdentityTuple(StringRef principal,
                                           StringRef function, StringRef source,
                                           StringRef role, int64_t slot) {
  std::string result;
  auto append = [&](StringRef part) {
    result += std::to_string(part.size());
    result += ':';
    result += part.str();
    result += '|';
  };
  append(principal);
  append(function);
  append(source);
  append(role);
  append(std::to_string(slot));
  return result;
}

struct ExplicitBuildIdentity {
  Attribute value;
  std::string source;
  bool present = false;
  bool valid = false;
};

static bool isExplicitBuildId(Attribute attribute) {
  if (!attribute)
    return false;
  // Opaque strings are not treated as build IDs; the runtime ABI carries a
  // 128-bit value, and accepting a host hash-like string would be ambiguous.
  if (auto integer = dyn_cast<IntegerAttr>(attribute))
    return integer.getType().isSignlessInteger(128);
  auto dictionary = dyn_cast<DictionaryAttr>(attribute);
  if (!dictionary || dictionary.size() != 2)
    return false;
  auto low = dictionary.getAs<IntegerAttr>("low");
  auto high = dictionary.getAs<IntegerAttr>("high");
  return low && high && low.getType().isSignlessInteger(64) &&
         high.getType().isSignlessInteger(64);
}

static ExplicitBuildIdentity getExplicitBuildIdentity(ModuleOp module) {
  ExplicitBuildIdentity result;
  SmallVector<std::pair<StringRef, Attribute>> found;
  for (StringRef name : {kHmxVtcmBuildIdAttr, kHmxVtcmBuildIdInputAttr}) {
    if (Attribute attribute = module->getAttr(name))
      found.push_back({name, attribute});
  }
  if (found.empty())
    return result;
  result.present = true;
  if (found.size() != 1)
    return result;
  result.value = found.front().second;
  result.source = found.front().first.str();
  result.valid = isExplicitBuildId(result.value);
  return result;
}

static bool residentHasInvalidUse(Value value) {
  SmallVector<Value> worklist{value};
  SmallPtrSet<Operation *, 32> visited;
  for (size_t index = 0; index < worklist.size(); ++index) {
    for (OpOperand &use : worklist[index].getUses()) {
      Operation *owner = use.getOwner();
      if (isa<memref::DeallocOp, hexagonmem::DeallocOp, func::ReturnOp,
              func::CallOp, func::CallIndirectOp, cf::BranchOp,
              cf::CondBranchOp, cf::SwitchOp, scf::YieldOp>(owner) ||
          isa<CallOpInterface>(owner))
        return true;
      if (isa<scf::ForOp>(owner) && use.getOperandNumber() >= 3) {
        // The first three operands are bounds/step.  Any later operand is an
        // iter_arg and therefore a region-carried storage value.
        return true;
      }
      if (isa<scf::WhileOp, scf::ConditionOp>(owner))
        return true;
      // A VTCM memref operand of any region-bearing operation is a structured
      // value transfer.  Plain captures used inside an execute-region are
      // visited as their concrete users and remain safe.
      if (owner->getNumRegions() != 0)
        return true;
      if (!visited.insert(owner).second)
        continue;
      for (Value result : owner->getResults())
        worklist.push_back(result);
    }
  }
  return false;
}

static bool readU64(DictionaryAttr dict, StringRef name, uint64_t &value);

static bool validGlobalSource(Operation *operation,
                              FlatSymbolRefAttr globalRef) {
  auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      operation, globalRef.getAttr());
  if (!global || !global.getConstant())
    return false;
  auto residentType = dyn_cast<MemRefType>(operation->getResult(0).getType());
  if (!residentType)
    return false;
  MemRefType globalType = global.getType();
  return globalType.hasStaticShape() && globalType.getLayout().isIdentity() &&
         globalType.getElementType().isF16() && residentType.hasStaticShape() &&
         residentType.getLayout().isIdentity() &&
         residentType.getElementType().isF16() &&
         globalType.getShape() == residentType.getShape() &&
         globalType.getElementType() == residentType.getElementType();
}

static bool validAddressSource(Operation *operation,
                               hexagonmem::AllocOp alloc) {
  if (alloc.getDynamicSizes().size() != 1)
    return false;
  Value address = alloc.getDynamicSizes().front();
  auto extract =
      address.getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>();
  if (!extract)
    return false;
  Value source = extract.getSource();
  auto argument = dyn_cast<BlockArgument>(source);
  if (!argument || !argument.getOwner()->isEntryBlock())
    return false;
  auto function = operation->getParentOfType<func::FuncOp>();
  if (!function || argument.getOwner() != &function.getBody().front() ||
      extract->getParentOfType<func::FuncOp>() != function)
    return false;
  auto sourceType = dyn_cast<BaseMemRefType>(source.getType());
  return sourceType && sourceType.getElementType().isF16();
}

static bool validResidentTag(Operation *operation, StringRef attrName,
                             int64_t bytes, bool weight) {
  // Resident storage is a process-lifetime floor.  Requiring an entry-block
  // allocation keeps a conditional/loop-local allocation from being counted
  // as if it were established on every invocation.
  auto function = operation->getParentOfType<func::FuncOp>();
  if (!function || function.getBody().empty() ||
      operation->getBlock() != &function.getBody().front())
    return false;

  auto dict = operation->getAttrOfType<DictionaryAttr>(attrName);
  if (!dict)
    return false;
  auto declaredBytes = dict.getAs<IntegerAttr>("bytes");
  if (!declaredBytes || !declaredBytes.getType().isSignlessInteger(64) ||
      declaredBytes.getValue().isNegative() || declaredBytes.getInt() != bytes)
    return false;

  if (weight) {
    Attribute globalAttr = dict.get("global");
    Attribute addressAttr = dict.get("address");
    bool global = globalAttr && isa<FlatSymbolRefAttr>(globalAttr);
    bool address = addressAttr && isa<UnitAttr>(addressAttr);
    if ((globalAttr != nullptr) != global ||
        (addressAttr != nullptr) != address)
      return false;
    if (global == address || !isa<hexagonmem::AllocOp>(operation))
      return false;
    auto alloc = cast<hexagonmem::AllocOp>(operation);
    size_t operands = alloc.getDynamicSizes().size();
    if ((global && operands != 0) || (address && operands != 1))
      return false;
    if (global &&
        !validGlobalSource(operation, cast<FlatSymbolRefAttr>(globalAttr)))
      return false;
    if (address && !validAddressSource(operation, alloc))
      return false;
  } else {
    uint64_t key = 0;
    if (!readU64(dict, "key", key))
      return false;
    // The census runs after the optional hexagonmem conversion, so accept the
    // same static workspace descriptor in either allocator representation.
    bool validAllocation = false;
    if (auto alloc = dyn_cast<memref::AllocOp>(operation))
      validAllocation = alloc.getDynamicSizes().empty();
    else if (auto alloc = dyn_cast<hexagonmem::AllocOp>(operation))
      validAllocation = alloc.getDynamicSizes().empty();
    if (!validAllocation)
      return false;
  }
  return !residentHasInvalidUse(operation->getResult(0));
}

struct ResidentProvenanceFacts {
  bool valid = false;
  int64_t slot = 0;
  StringRef reason;
};

static bool readNonNegativeI64(DictionaryAttr dict, StringRef name,
                               int64_t &value) {
  auto attr = dict.getAs<IntegerAttr>(name);
  if (!attr || !attr.getType().isSignlessInteger(64) ||
      attr.getValue().isNegative())
    return false;
  value = attr.getValue().getSExtValue();
  return true;
}

static bool readU64(DictionaryAttr dict, StringRef name, uint64_t &value) {
  auto attr = dict.getAs<IntegerAttr>(name);
  if (!attr || !attr.getType().isSignlessInteger(64))
    return false;
  // These fields are unsigned bit patterns.  Do not route them through
  // IntegerAttr::getInt(), whose signed interpretation would make values with
  // bit 63 set depend on a signed/unsigned conversion.
  value = attr.getValue().getZExtValue();
  return true;
}

static bool exactResidentType(MemRefType type) {
  return type && type.hasStaticShape() && type.getLayout().isIdentity() &&
         type.getElementType().isF16();
}

static bool validRuntimeWeightSourceView(Operation *operation,
                                         MemRefType residentType,
                                         DictionaryAttr sourceView,
                                         int64_t expectedSlot,
                                         StringRef &reason) {
  auto reject = [&](StringRef message) {
    reason = message;
    return false;
  };
  if (!sourceView || sourceView.size() != 6)
    return reject("runtime weight source_view is missing or malformed");
  auto viewKind = sourceView.getAs<StringAttr>("view_kind");
  auto argumentKind = sourceView.getAs<StringAttr>("argument_kind");
  auto offsetAttr = sourceView.getAs<IntegerAttr>("offset");
  auto wholeNAtt = sourceView.getAs<IntegerAttr>("whole_n");
  auto shapeAttr = sourceView.getAs<ArrayAttr>("shape");
  auto stridesAttr = sourceView.getAs<ArrayAttr>("strides");
  if (!viewKind || !argumentKind || !offsetAttr || !wholeNAtt || !shapeAttr ||
      !stridesAttr || shapeAttr.size() != 2 || stridesAttr.size() != 2)
    return reject("runtime weight source_view fields are incomplete");
  auto shape0 = dyn_cast<IntegerAttr>(shapeAttr.getValue()[0]);
  auto shape1 = dyn_cast<IntegerAttr>(shapeAttr.getValue()[1]);
  auto stride0 = dyn_cast<IntegerAttr>(stridesAttr.getValue()[0]);
  auto stride1 = dyn_cast<IntegerAttr>(stridesAttr.getValue()[1]);
  if (!shape0 || !shape1 || !stride0 || !stride1 ||
      !offsetAttr.getType().isInteger(64) ||
      !wholeNAtt.getType().isInteger(64) || !shape0.getType().isInteger(64) ||
      !shape1.getType().isInteger(64) || !stride0.getType().isInteger(64) ||
      !stride1.getType().isInteger(64))
    return reject("runtime weight source_view integer fields are malformed");
  int64_t k = shape0.getValue().getSExtValue();
  int64_t n = shape1.getValue().getSExtValue();
  int64_t rowStride = stride0.getValue().getSExtValue();
  int64_t columnStride = stride1.getValue().getSExtValue();
  int64_t offset = offsetAttr.getValue().getSExtValue();
  int64_t wholeN = wholeNAtt.getValue().getSExtValue();
  if (ShapedType::isDynamic(k) || ShapedType::isDynamic(n) ||
      ShapedType::isDynamic(rowStride) || ShapedType::isDynamic(columnStride) ||
      ShapedType::isDynamic(offset) || ShapedType::isDynamic(wholeN))
    return reject("runtime weight source_view contains a dynamic extent");
  if (residentType.getRank() != 5 || residentType.getDimSize(2) != 16 ||
      residentType.getDimSize(3) != layout::kTileEdge ||
      residentType.getDimSize(4) != 2)
    return reject("runtime weight destination is not the static HMX layout");
  int64_t residentK = hmx::weightKTiles(residentType) * layout::kTileEdge;
  int64_t residentN = hmx::weightNTiles(residentType) * layout::kTileEdge;
  if (k <= 0 || n <= 0 || rowStride <= 0 || columnStride != 1 || offset < 0 ||
      wholeN <= 0 || wholeN % layout::kTileEdge != 0 || k != residentK)
    return reject("runtime weight source_view shape/stride is not exact");

  auto alloc = dyn_cast<hexagonmem::AllocOp>(operation);
  if (!alloc || alloc.getDynamicSizes().size() != 1)
    return reject("runtime weight allocation has no single address source");
  auto extract = alloc.getDynamicSizes().front().getDefiningOp<
      memref::ExtractAlignedPointerAsIndexOp>();
  if (!extract)
    return reject("runtime weight address is not an aligned argument pointer");
  Value source = extract.getSource();
  auto argument = dyn_cast<BlockArgument>(source);
  auto function = operation->getParentOfType<func::FuncOp>();
  if (!argument || !function || !argument.getOwner()->isEntryBlock() ||
      argument.getOwner() != &function.getBody().front() ||
      extract->getParentOfType<func::FuncOp>() != function)
    return reject("runtime weight source is not an entry argument");
  int64_t argumentSlot = 0;
  bool foundArgumentSlot = false;
  for (BlockArgument candidate : function.getArguments()) {
    if (!isa<RankedTensorType, MemRefType, UnrankedMemRefType>(
            candidate.getType()))
      continue;
    if (candidate == argument) {
      foundArgumentSlot = true;
      break;
    }
    ++argumentSlot;
  }
  if (!foundArgumentSlot || argumentSlot != expectedSlot)
    return reject("runtime weight provenance slot disagrees with source argument");
  StringRef expectedArgumentKind =
      isa<UnrankedMemRefType>(argument.getType())
          ? "unranked-memref"
          : (isa<MemRefType>(argument.getType()) ? "ranked-memref"
                                                 : "unknown-memref");
  if (argumentKind.getValue() != expectedArgumentKind)
    return reject("runtime weight source_view argument kind disagrees");
  if (auto ranked = dyn_cast<MemRefType>(argument.getType())) {
    if (!ranked.hasStaticShape() || ranked.getRank() != 2 ||
        !ranked.getElementType().isF16() || !ranked.getLayout().isIdentity() ||
        ranked.getShape() != ArrayRef<int64_t>{k, wholeN})
      return reject("ranked runtime weight argument shape/layout is not exact");
  } else if (auto unranked = dyn_cast<UnrankedMemRefType>(argument.getType())) {
    if (!unranked.getElementType().isF16())
      return reject("unranked runtime weight argument element type is not f16");
  } else {
    return reject("runtime weight argument is not a memref");
  }

  StringRef kind = viewKind.getValue();
  if (kind == "whole-argument") {
    if (offset != 0 || n != residentN || n != wholeN || rowStride != wholeN)
      return reject("whole-argument runtime weight source_view is not dense");
  } else if (kind == "static-n-slice") {
    if (isa<UnrankedMemRefType>(argument.getType()) || residentN != wholeN ||
        n >= wholeN || n % layout::kTileEdge != 0 || wholeN % n != 0 ||
        rowStride != wholeN || offset % layout::kTileEdge != 0 ||
        offset > wholeN - n)
      return reject("static runtime weight source slice is not exact");
  } else {
    return reject("runtime weight source_view kind is unknown");
  }
  return true;
}

static ResidentProvenanceFacts
validateResidentProvenance(Operation *operation, ModuleOp module,
                           StringRef role, int64_t bytes) {
  ResidentProvenanceFacts result;
  auto provenance = operation->getAttrOfType<DictionaryAttr>(
      kHmxResidentProvenanceAttr);
  if (!provenance) {
    result.reason = kIdentityReasonResidentProvenanceMissing;
    return result;
  }
  auto function = operation->getParentOfType<func::FuncOp>();
  auto source = getStaticSourceFact(operation->getLoc());
  auto residentType = dyn_cast<MemRefType>(operation->getResult(0).getType());
  auto kind = provenance.getAs<StringAttr>("kind");
  auto provenanceRole = provenance.getAs<StringAttr>("role");
  auto functionName = provenance.getAs<StringAttr>("function");
  auto siteName = provenance.getAs<StringAttr>("site");
  auto moduleName = provenance.getAs<StringAttr>("module");
  auto descriptorStatus = provenance.getAs<StringAttr>("descriptor_status");
  int64_t declaredBytes = 0;
  int64_t alignment = 0;
  if (!function || !source || !residentType || !exactResidentType(residentType) ||
      !kind || kind.getValue() != (role == "weight-resident" ? "weight"
                                                               : "workspace") ||
      !provenanceRole || provenanceRole.getValue() != role ||
      !functionName || functionName.getValue() != function.getSymName() ||
      !siteName || siteName.getValue() != source->canonical ||
      !moduleName || moduleName.getValue() != residentPrincipalName(module) ||
      !descriptorStatus || descriptorStatus.getValue() != "checked" ||
      !readNonNegativeI64(provenance, "bytes", declaredBytes) ||
      declaredBytes != bytes ||
      !readNonNegativeI64(provenance, "alignment", alignment) ||
      !isSupportedResidentAlignment(alignment) ||
      !readNonNegativeI64(provenance, "slot", result.slot)) {
    result.reason = kIdentityReasonResidentProvenanceMissing;
    return result;
  }

  uint64_t expectedFunctionId =
      residentFunctionIdentity(residentPrincipalName(module), functionName.getValue());
  uint64_t expectedSiteId = residentSiteIdentity(
      residentPrincipalName(module), functionName.getValue(), siteName.getValue(),
      role, result.slot);
  uint64_t declaredFunctionId = 0;
  uint64_t declaredSiteId = 0;
  if (!readU64(provenance, "function_id", declaredFunctionId) ||
      !readU64(provenance, "site_id", declaredSiteId) ||
      declaredFunctionId != expectedFunctionId || declaredSiteId != expectedSiteId) {
    result.reason = kIdentityReasonResidentProvenanceMissing;
    return result;
  }

  auto residentAttrName = role == "weight-resident" ? kWeightResidentAttr
                                                     : kWorkspaceResidentAttr;
  auto resident = operation->getAttrOfType<DictionaryAttr>(residentAttrName);
  if (!resident) {
    result.reason = kIdentityReasonResidentProvenanceMissing;
    return result;
  }
  if (role == "workspace-resident") {
    uint64_t declaredKey = 0;
    uint64_t residentKey = 0;
    if (provenance.get("source_view") ||
        !readU64(provenance, "key", declaredKey) ||
        !readU64(resident, "key", residentKey) || declaredKey != residentKey ||
        (declaredKey & kHmxWorkspaceResidentKeyTag) == 0) {
      result.reason = kIdentityReasonResidentSource;
      return result;
    }
  } else {
    auto global = resident.getAs<FlatSymbolRefAttr>("global");
    auto address = resident.get("address");
    auto runtimeKind = provenance.getAs<StringAttr>("runtime_key_kind");
    auto source = provenance.getAs<StringAttr>("source");
    if (!runtimeKind || !source || (global != nullptr) == (address != nullptr))
      result.reason = kIdentityReasonResidentSource;
    else if (global) {
      const std::string expectedSource = "global:" + global.getValue().str();
      if (runtimeKind.getValue() != "global-address" ||
          provenance.get("source_view") || source.getValue() != expectedSource ||
          !validGlobalSource(operation, global))
        result.reason = kIdentityReasonResidentSource;
    } else if (!isa<UnitAttr>(address) ||
               runtimeKind.getValue() != "argument-address" ||
               source.getValue() !=
                   ("entry-argument-slot:" + std::to_string(result.slot)) ||
               !validRuntimeWeightSourceView(
                   operation, residentType,
                   provenance.getAs<DictionaryAttr>("source_view"), result.slot,
                   result.reason)) {
      result.reason = kIdentityReasonResidentSource;
    }
  }
  if (!result.reason.empty())
    return result;
  result.valid = true;
  return result;
}

static void recordAllocation(Operation *operation, Accounting &accounting) {
  bool weightResident = operation->hasAttr(kWeightResidentAttr);
  bool workspaceResident = operation->hasAttr(kWorkspaceResidentAttr);
  SizeResult size = getAllocationSize(operation);
  if (!size.isVtcm) {
    if (weightResident || workspaceResident) {
      ++accounting.unknownAllocations;
      accounting.complete = false;
    }
    return;
  }
  ++accounting.allocationSites;
  if (!size.known) {
    ++accounting.unknownAllocations;
    accounting.complete = false;
    return;
  }

  if (weightResident && workspaceResident) {
    accounting.complete = false;
    return;
  }
  if (weightResident &&
      !validResidentTag(operation, kWeightResidentAttr, size.bytes, true)) {
    accounting.complete = false;
    return;
  }
  if (workspaceResident &&
      !validResidentTag(operation, kWorkspaceResidentAttr, size.bytes, false)) {
    accounting.complete = false;
    return;
  }
  if (workspaceResident) {
    auto dict =
        cast<DictionaryAttr>(operation->getAttr(kWorkspaceResidentAttr));
    uint64_t key = 0;
    if (!readU64(dict, "key", key) || (key & kHmxWorkspaceResidentKeyTag) == 0 ||
        !accounting.workspaceKeys.insert(key).second) {
      accounting.complete = false;
      return;
    }
  }
  accounting.sites.push_back({static_cast<unsigned>(accounting.sites.size()),
                              operation, operation->getResult(0), size.bytes,
                              weightResident || workspaceResident,
                              size.constantBoundedExtent});
  if (weightResident) {
    if (!addBytes(accounting.weightResidentBytes, size.bytes))
      accounting.complete = false;
    return;
  }
  if (workspaceResident) {
    if (!addBytes(accounting.workspaceResidentBytes, size.bytes))
      accounting.complete = false;
    return;
  }
  if (!addBytes(accounting.transientBytes, size.bytes))
    accounting.complete = false;
}

/// Compiler-static provenance of one SSA value relative to a tracked VTCM
/// allocation site.
///
/// The lattice is deliberately three-valued and ordered
/// `kNone < kUnique < kAmbiguous`:
///
///  * `kNone`     - the value is not derived from a tracked site.
///  * `kUnique`   - exactly one reviewed path attributes the value to one
///                  site, so that site owns the value's storage lifetime.
///  * `kAmbiguous` - two different sites, or a site and a non-site, reach the
///                  same value.  The lifetime owner is then unproven and the
///                  analysis fails closed instead of picking a winner.
///
/// The order is total and monotone, so the fixpoint terminates by construction.
enum class ProvenanceKind : uint8_t { kNone, kUnique, kAmbiguous };

struct ValueProvenance {
  ProvenanceKind kind = ProvenanceKind::kNone;
  unsigned site = 0;
};

static ValueProvenance joinProvenance(const ValueProvenance &lhs,
                                      const ValueProvenance &rhs) {
  if (lhs.kind == ProvenanceKind::kAmbiguous ||
      rhs.kind == ProvenanceKind::kAmbiguous)
    return {ProvenanceKind::kAmbiguous, 0};
  if (lhs.kind == ProvenanceKind::kNone)
    return rhs;
  if (rhs.kind == ProvenanceKind::kNone)
    return lhs;
  if (lhs.site == rhs.site)
    return lhs;
  return {ProvenanceKind::kAmbiguous, 0};
}

static unsigned getProvenanceRank(ProvenanceKind kind) {
  switch (kind) {
  case ProvenanceKind::kNone:
    return 0;
  case ProvenanceKind::kUnique:
    return 1;
  case ProvenanceKind::kAmbiguous:
    return 2;
  }
  return 2;
}

static bool isProvenanceGrowth(const ValueProvenance &oldValue,
                               const ValueProvenance &updated) {
  if (getProvenanceRank(updated.kind) < getProvenanceRank(oldValue.kind))
    return false;
  if (oldValue.kind == ProvenanceKind::kUnique &&
      updated.kind == ProvenanceKind::kUnique)
    return oldValue.site == updated.site;
  return true;
}

struct LivenessMaps {
  DenseMap<Operation *, unsigned> siteByOperation;
  DenseMap<Value, unsigned> siteByValue;
  DenseMap<unsigned, const AllocationSite *> siteById;
  std::set<unsigned> siteIds;
};

/// Active-site state at one program point.  `bytes` is always re-derived from
/// `active`, so a state can never disagree with the site set it claims.
struct FlowState {
  std::set<unsigned> active;
  int64_t bytes = 0;

  bool operator==(const FlowState &other) const {
    return active == other.active && bytes == other.bytes;
  }
  bool operator!=(const FlowState &other) const { return !(*this == other); }
};

struct FlowResult {
  bool complete = true;
  std::string reason;
  int64_t peakTransientBytes = 0;
  std::set<unsigned> peakSiteIds;
  int64_t deallocationSites = 0;
  std::set<unsigned> visitedSiteIds;
  /// Worklist iterations consumed.  A non-zero value is the fixpoint evidence
  /// that some entry state needed more than one round (any cycle, or any
  /// merged predecessor state).
  int64_t fixpointRounds = 0;
  /// Blocks whose exit state changed after their first computation, i.e. the
  /// blocks a cycle or a path merge forced the analysis to re-examine.
  int64_t revisedBlocks = 0;

  void fail(StringRef failureReason) {
    if (complete) {
      complete = false;
      reason = failureReason.str();
    }
  }
};

static bool refreshStateBytes(FlowState &state, const LivenessMaps &maps) {
  int64_t total = 0;
  for (unsigned id : state.active) {
    auto found = maps.siteById.find(id);
    if (found == maps.siteById.end())
      return false;
    if (!addBytes(total, found->second->bytes))
      return false;
  }
  state.bytes = total;
  return true;
}

/// Union-join `incoming` into `state`.  The lattice is the power set of active
/// sites ordered by inclusion, so a join can only grow; the boolean reports
/// whether anything changed so the caller can decide whether to re-run a
/// dependent block.
static bool joinState(FlowState &state, const FlowState &incoming,
                      const LivenessMaps &maps, FlowResult &flow) {
  bool grew = false;
  for (unsigned id : incoming.active) {
    if (state.active.insert(id).second)
      grew = true;
  }
  if (!refreshStateBytes(state, maps)) {
    flow.fail("integer-overflow");
    return false;
  }
  return grew;
}

static bool isVtcmType(Type type) {
  if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
    int memorySpace = hexagon::DEFAULT_DDR_ADDRESS_SPACE;
    if (!getMemRefMemorySpace(memrefType, memorySpace))
      return true;
    return memorySpace == hexagon::VTCM_ADDRESS_SPACE;
  }
  if (auto croutonType = dyn_cast<crouton::CroutonType>(type))
    return croutonType.getVtcm().getValue();
  return false;
}

static bool isVtcmValue(Value value) { return isVtcmType(value.getType()); }

/// A view, cast, or reshape is an alias of its principal allocation: using it
/// is lifetime-equivalent to using the principal, and releasing it is not a
/// balanced pair with the principal (that stays a negative case).
static bool isKnownAliasProducer(Operation *operation) {
  return isa<memref::AssumeAlignmentOp, memref::CastOp, memref::CollapseShapeOp,
             memref::ExpandShapeOp, memref::MemorySpaceCastOp,
             memref::ReinterpretCastOp, memref::ReshapeOp, memref::TransposeOp,
             memref::ViewOp, memref::SubViewOp>(operation);
}

/// Reviewed destination-style producers: the single result is an alias of a
/// named destination operand rather than a fresh buffer.  Only the
/// single-result form is reviewed, so a future multi-result op cannot
/// silently alias every result to one operand.
static bool getSynchronousResultAlias(Operation *operation, Value &source) {
  if (operation->getNumResults() != 1)
    return false;
  if (auto matmul = dyn_cast<hmx::MatmulOp>(operation)) {
    if (operation->getNumOperands() < 3)
      return false;
    source = matmul.getOuts();
    return true;
  }
  if (auto pack = dyn_cast<hmx::PackActOp>(operation)) {
    if (operation->getNumOperands() < 4)
      return false;
    source = pack.getDst();
    return true;
  }
  if (auto pack = dyn_cast<hmx::PackWeightOp>(operation)) {
    if (operation->getNumOperands() < 4)
      return false;
    source = pack.getDst();
    return true;
  }
  if (auto unpack = dyn_cast<hmx::UnpackAccOp>(operation)) {
    if (operation->getNumOperands() < 4)
      return false;
    source = unpack.getDst();
    return true;
  }
  if (auto unpack = dyn_cast<hmx::UnpackAccF32Op>(operation)) {
    if (operation->getNumOperands() < 4)
      return false;
    source = unpack.getDst();
    return true;
  }
  return false;
}

static bool isKnownSynchronousUse(Operation *operation) {
  return isa<memref::CopyOp, memref::DimOp, memref::ExtractStridedMetadataOp,
             memref::LoadOp, memref::RankOp, memref::StoreOp, hmx::AccClearOp,
             hmx::AccReadOp, hmx::BiasInitOp, hmx::MatmulOp, hmx::MmaOp,
             hmx::PackActOp, hmx::PackWeightOp, hmx::UnpackAccOp,
             hmx::UnpackAccF32Op>(operation);
}

static bool isKnownAllocationOperation(Operation *operation) {
  return isa<bufferization::AllocTensorOp, hexagonmem::AllocOp,
             memref::AllocOp, memref::AllocaOp, AllocCroutonOp>(operation);
}

/// The regions of a reviewed structured operation whose terminator operands
/// become that operation's results. `scf.while` deliberately names only the
/// after region: its before region ends in `scf.condition`, which feeds the
/// after region's block arguments rather than the while's results. Liveness
/// handles the before condition as a separate exit below; it must not be
/// added here or result provenance would treat `scf.condition` as a yield.
static void getResultRegions(Operation *operation,
                             SmallVectorImpl<Region *> &regions) {
  if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
    regions.push_back(&ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
      regions.push_back(&ifOp.getElseRegion());
    return;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
    regions.push_back(&forOp.getRegion());
    return;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
    regions.push_back(&whileOp.getAfter());
    return;
  }
  if (auto execute = dyn_cast<scf::ExecuteRegionOp>(operation))
    regions.push_back(&execute.getRegion());
}

static bool isReviewedStructuredOperation(Operation *operation) {
  SmallVector<Region *> regions;
  getResultRegions(operation, regions);
  return !regions.empty();
}

/// Proof budget for both fixpoints.  This is a termination guarantee, not a
/// tuned threshold: a function needing more rounds than this is reported as not
/// proven rather than allowed to keep refining.
static constexpr unsigned kMaxProvenanceRounds = 64;

/// Computes, for every value in one function, which tracked allocation site
/// owns its storage.  This is a monotone fixpoint over the reviewed alias
/// lattice, deliberately kept separate from the byte dataflow so alias facts
/// are frozen before any program point is evaluated.
class ProvenanceAnalysis {
public:
  ProvenanceAnalysis(const LivenessMaps &maps, FlowResult &flow)
      : maps(maps), flow(flow) {}

  bool run(func::FuncOp function) {
    for (unsigned id : maps.siteIds) {
      const AllocationSite *site = maps.siteById.lookup(id);
      if (site)
        record(site->value, {ProvenanceKind::kUnique, id});
    }
    unsigned rounds = 0;
    bool changed = true;
    while (changed) {
      if (++rounds > kMaxProvenanceRounds) {
        flow.fail("alias provenance dataflow did not converge");
        return false;
      }
      changed = false;
      function.walk([&](Block *block) {
        computeBlockArguments(block, changed);
        for (Operation &operation : *block)
          computeResults(&operation, changed);
      });
    }
    return validate(function);
  }

  ValueProvenance lookup(Value value) const {
    auto found = values.find(value);
    return found == values.end() ? ValueProvenance{} : found->second;
  }

  bool hasProvenance(Value value) const {
    return lookup(value).kind != ProvenanceKind::kNone;
  }

private:
  bool record(Value value, ValueProvenance candidate) {
    auto found = values.find(value);
    if (found == values.end()) {
      values.try_emplace(value, candidate);
      return true;
    }
    ValueProvenance merged = joinProvenance(found->second, candidate);
    if (!isProvenanceGrowth(found->second, merged)) {
      flow.fail("alias provenance lattice is not monotone");
      return false;
    }
    if (merged.kind == found->second.kind && merged.site == found->second.site)
      return false;
    found->second = merged;
    return true;
  }

  void computeBlockArguments(Block *block, bool &changed) {
    Region *region = block->getParent();
    Operation *owner = region->getParentOp();
    if (block == &region->front()) {
      if (block->getNumArguments() == 0)
        return;
      if (!owner || isa<func::FuncOp>(owner)) {
        // Function arguments never own tracked storage: a VTCM-typed argument
        // is rejected by the caller before this analysis runs.
        return;
      }
      SmallVector<ValueProvenance> incoming;
      if (!getRegionEntryProvenance(owner, region, incoming)) {
        flow.fail("region block argument has no reviewed incoming value");
        return;
      }
      for (auto [index, argument] : llvm::enumerate(block->getArguments())) {
        if (index >= incoming.size()) {
          flow.fail("region block argument has no reviewed incoming value");
          return;
        }
        if (record(argument, incoming[index]))
          changed = true;
      }
      return;
    }
    for (BlockArgument argument : block->getArguments()) {
      ValueProvenance merged;
      for (Block *predecessor : block->getPredecessors()) {
        Operation *terminator = predecessor->getTerminator();
        auto branch = dyn_cast_or_null<BranchOpInterface>(terminator);
        if (!branch)
          continue;
        for (auto [index, successor] : llvm::enumerate(terminator->getSuccessors())) {
          if (successor != block)
            continue;
          SuccessorOperands operands = branch.getSuccessorOperands(index);
          OperandRange forwarded = operands.getForwardedOperands();
          if (argument.getArgNumber() >= forwarded.size())
            continue;
          merged = joinProvenance(
              merged, lookup(forwarded[argument.getArgNumber()]));
        }
      }
      if (record(argument, merged))
        changed = true;
    }
  }

  /// Provenance of the block arguments of a reviewed region entry block.
  bool getRegionEntryProvenance(Operation *owner, Region *region,
                                SmallVectorImpl<ValueProvenance> &out) {
    if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
      if (&forOp.getRegion() != region)
        return false;
      // `scf.for`'s body block arguments are the induction variable followed by
      // the loop-carried values.  The induction variable is an index SSA value
      // with no storage owner, so it maps to `kNone` rather than to an
      // init operand.
      out.push_back(ValueProvenance());
      for (Value value : forOp.getInitArgs())
        out.push_back(lookup(value));
      return true;
    }
    auto whileOp = cast<scf::WhileOp>(owner);
    if (&whileOp.getBefore() == region) {
      for (Value value : whileOp.getInits())
        out.push_back(lookup(value));
      return true;
    }
    if (&whileOp.getAfter() != region)
      return false;
    // The after region's block arguments receive whatever the before region
    // hands over through `scf.condition`.  Every condition that can reach this
    // region contributes, so a mismatch between iterations stays ambiguous
    // instead of being resolved by iteration order.
    bool sawCondition = false;
    unsigned width = 0;
    for (Block &block : whileOp.getBefore()) {
      auto condition = dyn_cast_or_null<scf::ConditionOp>(block.getTerminator());
      if (!condition)
        continue;
      // `scf.condition` carries the predicate as operand zero; only the
      // remaining operands feed the after region's block arguments.  Treating
      // the predicate as the first carried value silently loses the first
      // VTCM argument on a while whose after-block argument is otherwise
      // unused.
      if (condition.getNumOperands() == 0)
        return false;
      const unsigned carriedWidth = condition.getNumOperands() - 1;
      if (sawCondition && carriedWidth != width)
        return false;
      width = carriedWidth;
      for (unsigned index = 0; index < width; ++index) {
        ValueProvenance candidate = lookup(condition.getOperand(index + 1));
        if (!sawCondition) {
          out.resize(width);
          out[index] = candidate;
        } else {
          out[index] = joinProvenance(out[index], candidate);
        }
      }
      sawCondition = true;
    }
    if (!sawCondition || out.size() != width)
      return false;
    return true;
  }

  void computeResults(Operation *operation, bool &changed) {
    if (operation->getNumResults() == 0)
      return;
    std::optional<ValueProvenance> candidate = provenanceOfResults(operation);
    if (!candidate) {
      flow.fail("result has no reviewed alias provenance");
      return;
    }
    for (Value result : operation->getResults())
      if (record(result, *candidate))
        changed = true;
  }

  std::optional<ValueProvenance> provenanceOfResults(Operation *operation) {
    auto site = maps.siteByOperation.find(operation);
    if (site != maps.siteByOperation.end())
      return ValueProvenance{ProvenanceKind::kUnique, site->second};
    if (isKnownAliasProducer(operation)) {
      if (operation->getNumOperands() == 0)
        return std::nullopt;
      return lookup(operation->getOperand(0));
    }
    Value source;
    if (getSynchronousResultAlias(operation, source))
      return lookup(source);
    if (operation->getNumRegions() == 0)
      return ValueProvenance{};
    if (!isReviewedStructuredOperation(operation))
      return std::nullopt;
    SmallVector<Region *> regions;
    getResultRegions(operation, regions);
    SmallVector<ValueProvenance> merged(operation->getNumResults());
    bool sawYield = false;
    for (Region *region : regions) {
      if (region->empty())
        continue;
      for (Block &block : *region) {
        auto yield = dyn_cast_or_null<scf::YieldOp>(block.getTerminator());
        if (!yield)
          return std::nullopt;
        if (yield.getNumOperands() != operation->getNumResults())
          return std::nullopt;
        sawYield = true;
        for (unsigned index = 0; index < merged.size(); ++index)
          merged[index] =
              joinProvenance(merged[index], lookup(yield.getOperand(index)));
      }
    }
    if (!sawYield)
      return operation->getNumResults() == 0 ? std::optional<ValueProvenance>(ValueProvenance{})
                                            : std::nullopt;
    for (const ValueProvenance &provenance : merged)
      if (provenance.kind == ProvenanceKind::kAmbiguous) {
        // A region-carried value must have the same owner on every path.  One
        // path reaching a site and another reaching a different site (or no
        // site) is a lifetime fact this analysis cannot attribute, so it is
        // reported rather than resolved.
        flow.fail("region-carried value has ambiguous alias provenance");
        return std::nullopt;
      }
    return merged.empty() ? ValueProvenance{} : merged.front();
  }

  /// A VTCM-typed value with no reviewed owner is reported against the class
  /// that could have produced one, so the failure names the real blocker (an
  /// unmodeled completion or an unmodeled callee) instead of the generic
  /// untracked-storage label.  Naming the class is what makes the record
  /// machine-readable: a reader can tell "this shape is closed" from "this
  /// storage was never attributed".
  void reportUntrackedOwner(Value value) {
    if (Operation *defining = value.getDefiningOp()) {
      if (isa<hmx::StageOp, hmx::AwaitOp, memref::DmaStartOp>(defining)) {
        flow.fail("asynchronous transfer is unsupported");
        return;
      }
      if (isa<CallOpInterface>(defining)) {
        flow.fail("call is unsupported");
        return;
      }
    }
    flow.fail("untracked VTCM value or alias");
  }

  /// A complete function needs every tracked-storage value to have exactly one
  /// owner.  A VTCM-typed value with no owner is storage this analysis never
  /// sees; an ambiguous owner is a value whose uses may or may not reach a
  /// tracked site.
  bool validate(func::FuncOp function) {
    auto check = [&](Value value) {
      ValueProvenance provenance = lookup(value);
      if (provenance.kind == ProvenanceKind::kAmbiguous) {
        flow.fail("ambiguous alias provenance");
        return;
      }
      if (isVtcmValue(value) && provenance.kind != ProvenanceKind::kUnique)
        reportUntrackedOwner(value);
    };
    function.walk([&](Block *block) {
      for (BlockArgument argument : block->getArguments())
        check(argument);
      for (Operation &operation : *block) {
        for (Value operand : operation.getOperands())
          check(operand);
        for (Value result : operation.getResults())
          check(result);
      }
    });
    return flow.complete;
  }

  const LivenessMaps &maps;
  FlowResult &flow;
  DenseMap<Value, ValueProvenance> values;
};

static bool isResidentSite(const LivenessMaps &maps, unsigned site) {
  auto found = maps.siteById.find(site);
  return found != maps.siteById.end() && found->second->resident;
}

/// Byte-level liveness over one function.
///
/// The engine is a finite worklist fixpoint: each block is analyzed from a
/// state joined over its predecessors, and a block is re-analyzed when a back
/// edge or a late-arriving predecessor grows that join.  A loop therefore
/// becomes exact rather than being rejected, provided its body returns the
/// entry state.  A loop that carries an allocation across iterations, or a
/// deallocation that only some path can reach, is reported instead of being
/// approximated.
class LivenessFlow {
public:
  LivenessFlow(func::FuncOp function, const LivenessMaps &maps,
               const ProvenanceAnalysis &provenance, FlowResult &flow)
      : function(function), maps(maps), provenance(provenance), flow(flow) {}

  bool run() {
    if (function.getBody().empty())
      return true;
    unsigned blockCount = 0;
    function.walk([&](Block *block) { ++blockCount; });
    // Each block's exit state can only change when its entry join grows, which
    // is bounded by the number of active sites it can hold.  The budget is a
    // termination guarantee scaled by the CFG size, not a tuned threshold.
    maxRounds = 8 * (maps.siteIds.size() + 1) * (blockCount + 1) + 64;
    // The function entry state is empty: a resident allocation is never
    // released inside the function, so it is reported through the resident byte
    // floor rather than being tracked as a transient site that can never reach
    // a balanced pair.
    regionEntry[&function.getBody()] = FlowState();
    enqueue(&function.getBody().front());
    while (!worklist.empty()) {
      if (++flow.fixpointRounds > maxRounds) {
        flow.fail("byte liveness dataflow did not converge");
        return false;
      }
      Block *block = worklist.pop_back_val();
      // Clear the pending mark on pop: a block that was deferred for want of a
      // predecessor's exit state must be enqueueable again when that state
      // arrives, which is exactly how a cycle converges.
      pendingBlocks.erase(block);
      if (!isReady(block))
        continue;
      waiting = false;
      processBlock(block);
      if (!flow.complete)
        return false;
    }
    if (waiting) {
      flow.fail("structured region has no proven yield exit");
      return false;
    }
    // Soundness guard for the optimistic seeding above: every block that
    // control flow can actually reach from the entry must have been analyzed to
    // completion.  An unreachable block is allowed to stay unresolved, because
    // it can never contribute a program point; its allocation sites are still
    // caught by the coverage check.
    for (Block *block : computeReachableBlocks()) {
      if (!blockExitState.count(block)) {
        flow.fail("control-flow dataflow did not reach a proven exit state");
        return false;
      }
    }
    // A region that was entered but never produced a yield state has no proven
    // exit, so its containing operation was never evaluated.
    bool unresolvedRegion = false;
    function.walk([&](Region *region) {
      if (regionEntry.count(region) && !regionExit.count(region))
        unresolvedRegion = true;
    });
    if (unresolvedRegion) {
      flow.fail("structured region has no proven yield exit");
      return false;
    }
    if (!regionExit.count(&function.getBody())) {
      flow.fail("function has no proven return exit");
      return false;
    }
    return flow.complete;
  }

  const std::set<unsigned> &peakSiteIds() const { return flow.peakSiteIds; }
  int64_t peakBytes() const { return flow.peakTransientBytes; }

private:
  void enqueue(Block *block) {
    if (block && pendingBlocks.insert(block).second)
      worklist.push_back(block);
  }

  /// Optimistic readiness.  A cyclic CFG cannot be evaluated by waiting for
  /// *every* predecessor, because a loop header's only other predecessor is
  /// reached through the header itself.  A block is therefore analyzed as soon
  /// as it has one resolved predecessor and re-analyzed whenever another one
  /// appears.  The post-drain reachability check is what makes this sound: it
  /// requires every structurally reachable block to have been analyzed, so no
  /// block can keep an under-approximated entry state.
  bool isReady(Block *block) const {
    if (!regionEntry.count(block->getParent()))
      return false;
    if (block->getPredecessors().empty())
      return true;
    for (Block *predecessor : block->getPredecessors())
      if (blockExitState.count(predecessor))
        return true;
    return false;
  }

  void enqueueOwner(Region *region) {
    Operation *owner = region->getParentOp();
    if (owner && owner->getBlock())
      enqueue(owner->getBlock());
  }

  /// Blocks control flow can reach from the function entry: successors of a
  /// block plus the entry block of every region a reachable operation owns.
  SmallVector<Block *> computeReachableBlocks() {
    SmallVector<Block *> reachable;
    SmallPtrSet<Block *, 16> seen;
    SmallVector<Block *> worklist;
    if (function.getBody().empty())
      return reachable;
    worklist.push_back(&function.getBody().front());
    while (!worklist.empty()) {
      Block *block = worklist.pop_back_val();
      if (!seen.insert(block).second)
        continue;
      reachable.push_back(block);
      for (Operation &operation : *block) {
        for (Region &region : operation.getRegions()) {
          if (!region.empty())
            worklist.push_back(&region.front());
        }
      }
      for (Block *successor : block->getSuccessors())
        worklist.push_back(successor);
    }
    return reachable;
  }

  void updatePeak(const FlowState &state) {
    if (state.bytes <= flow.peakTransientBytes)
      return;
    flow.peakTransientBytes = state.bytes;
    flow.peakSiteIds = state.active;
  }

  void processBlock(Block *block) {
    FlowState state;
    FlowResult scratch;
    // Join over the predecessors that have resolved so far.  A predecessor that
    // resolves later re-enqueues this block, so the final recorded state always
    // includes every predecessor of a reachable block; the post-drain
    // reachability check is what turns that into a guarantee.
    for (Block *predecessor : block->getPredecessors()) {
      auto found = blockExitState.find(predecessor);
      if (found == blockExitState.end())
        continue;
      joinState(state, found->second, maps, scratch);
    }
    // Only a region's entry block inherits the state pushed in by the
    // containing operation; a block reached through a back edge inherits only
    // from its predecessors.
    if (block == &block->getParent()->front())
      joinState(state, regionEntry[block->getParent()], maps, scratch);
    if (!refreshStateBytes(state, maps)) {
      flow.fail("integer-overflow");
      return;
    }
    if (!transferBlock(block, state))
      return;
    auto previous = blockExitState.find(block);
    if (previous != blockExitState.end() && previous->second == state)
      return;
    if (previous != blockExitState.end())
      ++flow.revisedBlocks;
    blockExitState[block] = state;
    for (Block *successor : block->getSuccessors())
      enqueue(successor);
    enqueueOwner(block->getParent());
  }

  /// Transfer one block's operations onto the incoming state.  Returns false
  /// when the block cannot be proven; the reason is recorded on the flow, and
  /// `waiting` marks a block that is deferred until a nested region resolves.
  bool transferBlock(Block *block, FlowState &state) {
    for (Operation &operation : *block) {
      Operation *op = &operation;

      auto siteIt = maps.siteByOperation.find(op);
      if (siteIt != maps.siteByOperation.end()) {
        const AllocationSite &site = *maps.siteById.lookup(siteIt->second);
        flow.visitedSiteIds.insert(site.id);
        if (!site.resident) {
          if (!state.active.insert(site.id).second) {
            // The site is already charged on another path reaching this
            // program point, so this is not a single balanced pair.
            flow.fail("allocation site is live across a back edge or a "
                      "path merge");
            return false;
          }
          retiredSites.erase(site.id);
          if (!refreshStateBytes(state, maps)) {
            flow.fail("integer-overflow");
            return false;
          }
          updatePeak(state);
        }
        continue;
      }

      if (isa<memref::DeallocOp, hexagonmem::DeallocOp>(op)) {
        Value target = op->getOperand(0);
        auto deallocSite = maps.siteByValue.find(target);
        if (deallocSite == maps.siteByValue.end()) {
          // Releasing a view or cast is not a balanced pair with the principal
          // allocation, and releasing untracked VTCM storage has no site to
          // charge.  Both stay closed rather than being guessed.
          if (provenance.hasProvenance(target) || isVtcmValue(target)) {
            flow.fail("ambiguous or aliased deallocation origin");
            return false;
          }
          continue;
        }
        const AllocationSite &site = *maps.siteById.lookup(deallocSite->second);
        if (site.resident) {
          flow.fail("resident allocation has a deallocation");
          return false;
        }
        if (!state.active.erase(site.id)) {
          flow.fail(retiredSites.count(site.id)
                        ? "duplicate deallocation"
                        : "deallocation without an active allocation");
          return false;
        }
        retiredSites.insert(site.id);
        if (deallocOperations.insert(op).second)
          ++flow.deallocationSites;
        if (!refreshStateBytes(state, maps)) {
          flow.fail("integer-overflow");
          return false;
        }
        continue;
      }

      // Every use of an allocation-derived value must happen while its site is
      // active.  This precedes the class-specific escape rules so a use after
      // release is reported as a lifetime error rather than as whatever the
      // consuming operation happens to be.
      for (Value operand : op->getOperands()) {
        ValueProvenance fact = provenance.lookup(operand);
        if (fact.kind == ProvenanceKind::kAmbiguous) {
          flow.fail("ambiguous alias provenance");
          return false;
        }
        if (fact.kind != ProvenanceKind::kUnique)
          continue;
        if (state.active.count(fact.site) || isResidentSite(maps, fact.site))
          continue;
        flow.fail("allocation-derived value used after deallocation");
        return false;
      }

      if (isa<func::ReturnOp>(op)) {
        // A VTCM value returned to a caller outlives this frame, so its
        // requested bytes cannot be attributed to this invocation.
        for (Value operand : op->getOperands()) {
          if (provenance.hasProvenance(operand) || isVtcmValue(operand)) {
            flow.fail("VTCM allocation escapes through function return");
            return false;
          }
        }
        if (!state.active.empty()) {
          flow.fail("missing deallocation at function return");
          return false;
        }
        return publishRegionExit(block->getParent(), state);
      }

      if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        // Yielding a derived handle keeps the *owning* site alive; it does not
        // transfer ownership, so the carried value is re-attributed by the
        // provenance lattice rather than charged as a new allocation.  A
        // yielded VTCM value with no owner is storage this analysis cannot see.
        for (Value operand : yield.getOperands()) {
          ValueProvenance fact = provenance.lookup(operand);
          if (fact.kind == ProvenanceKind::kAmbiguous) {
            flow.fail("ambiguous alias provenance");
            return false;
          }
          if (fact.kind == ProvenanceKind::kUnique)
            continue;
          if (isVtcmValue(operand)) {
            flow.fail("region-carried VTCM value is unsupported");
            return false;
          }
        }
        return publishRegionExit(block->getParent(), state);
      }

      if (isa<scf::ConditionOp>(op)) {
        // `scf.while`'s before region exits through a condition, not a yield.
        // Its exit state feeds the after region through ordinary CFG edges; the
        // after region is the one whose yield ends the loop.
        return publishRegionExit(block->getParent(), state);
      }

      if (op->getNumRegions() != 0) {
        if (!isReviewedStructuredOperation(op)) {
          flow.fail(op->hasTrait<OpTrait::IsTerminator>() ||
                            op->getNumSuccessors() != 0
                        ? "plain CFG or while edge is unsupported"
                        : "structured-region operation is unsupported");
          return false;
        }
        for (Region &region : op->getRegions()) {
          if (region.empty())
            continue;
          auto entry = regionEntry.find(&region);
          bool changed = false;
          if (entry == regionEntry.end()) {
            entry = regionEntry.insert({&region, state}).first;
            changed = true;
          } else {
            FlowResult scratch;
            changed = joinState(entry->second, state, maps, scratch);
          }
          // `scf.while` executes its before region before the after region.
          // Seed the after entry with the enclosing state as a conservative
          // zero-trip bound, then join the before condition exits whenever
          // they become available.  The owner block is re-enqueued when a
          // before exit changes, so this is a real fixpoint rather than a
          // one-time optimistic seed.
          if (auto whileOp = dyn_cast<scf::WhileOp>(op);
              whileOp && &region == &whileOp.getAfter()) {
            auto beforeExit = regionExit.find(&whileOp.getBefore());
            if (beforeExit != regionExit.end()) {
              FlowResult scratch;
              changed |= joinState(entry->second, beforeExit->second, maps,
                                   scratch);
            }
          }
          if (changed)
            enqueue(&region.front());
        }
        if (!transferStructuredPostState(op, state))
          return false;
        updatePeak(state);
        continue;
      }

      if (isa<CallOpInterface>(op)) {
        // A call may retain, alias, or release storage this analysis cannot
        // see, and there is no reviewed call summary yet, so the contract
        // stays closed instead of assuming a call is harmless.
        flow.fail("call is unsupported");
        return false;
      }
      if (isa<hmx::StageOp, hmx::AwaitOp, memref::DmaStartOp>(op)) {
        // Completion is not modeled, so a release after the transfer could
        // race the engine.  Only an explicit completion summary admits this.
        flow.fail("asynchronous transfer is unsupported");
        return false;
      }

      if (auto extract = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(op)) {
        if (provenance.hasProvenance(extract.getSource())) {
          flow.fail("VTCM allocation pointer extraction is unsupported");
          return false;
        }
      }
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (provenance.hasProvenance(store.getValue())) {
          flow.fail("VTCM allocation value is stored or aliased");
          return false;
        }
      }

      if (op->hasTrait<OpTrait::IsTerminator>() && op->getNumSuccessors() != 0) {
        // Plain CFG edges are ordinary predecessors already represented in the
        // worklist, so the edge itself carries no extra state.
        continue;
      }
      if (isKnownAliasProducer(op) || isKnownSynchronousUse(op) ||
          isKnownAllocationOperation(op))
        continue;
      if (!isMemoryEffectFree(op)) {
        flow.fail("unknown side effect or asynchronous operation");
        return false;
      }
    }
    return true;
  }

  /// Record the state at a region's exit terminator.  A region with more than
  /// one exit contributes the join, which is the same sound upper bound used
  /// everywhere else in this analysis.
  bool publishRegionExit(Region *region, const FlowState &state) {
    auto exit = regionExit.find(region);
    if (exit == regionExit.end())
      regionExit[region] = state;
    else
      joinState(exit->second, state, maps, flow);
    return true;
  }

  /// State immediately after a reviewed structured operation.  A loop body may
  /// run zero times and an untaken conditional arm runs zero times, so those
  /// paths contribute the entry state.  Without that, a deallocation reachable
  /// only from inside a loop would be reported as a balanced pair even though
  /// a zero-trip loop never performs it.
  bool transferStructuredPostState(Operation *op, FlowState &state) {
    SmallVector<Region *> resultRegions;
    getResultRegions(op, resultRegions);
    bool contributesEntryState = isa<scf::ForOp, scf::WhileOp>(op);
    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      contributesEntryState = ifOp.getElseRegion().empty();
    FlowState merged;
    FlowResult scratch;
    bool sawRegion = false;
    // A while has two relevant exits: the before region's condition (which
    // can leave the loop without entering the body) and the after region's
    // yield.  The old result-region-only path dropped the former, allowing a
    // carried VTCM value to disappear on the zero-trip/condition-false path.
    // Wait until both nested regions have published an exit before publishing
    // the enclosing block's state.
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      auto beforeExit = regionExit.find(&whileOp.getBefore());
      if (beforeExit == regionExit.end()) {
        waiting = true;
        return false;
      }
      joinState(merged, beforeExit->second, maps, scratch);
      sawRegion = true;
    }
    for (Region *region : resultRegions) {
      if (region->empty()) {
        // An empty region body is a legal no-op, not a missing exit.
        sawRegion = true;
        continue;
      }
      auto exit = regionExit.find(region);
      if (exit == regionExit.end()) {
        // The containing block is deferred until the region resolves; it is
        // never reported as balanced on the strength of an unresolved arm.
        waiting = true;
        return false;
      }
      joinState(merged, exit->second, maps, scratch);
      sawRegion = true;
    }
    if (contributesEntryState) {
      // Zero-trip path: a loop body may never run and an arm without else may
      // never be taken, so the entry state is a live alternative.
      joinState(merged, state, maps, scratch);
    }
    if (!sawRegion) {
      flow.fail("structured region has no reviewed exit");
      return false;
    }
    state = merged;
    return true;
  }

  func::FuncOp function;
  const LivenessMaps &maps;
  const ProvenanceAnalysis &provenance;
  FlowResult &flow;
  SmallVector<Block *> worklist;
  DenseSet<Block *> pendingBlocks;
  DenseMap<Block *, FlowState> blockExitState;
  DenseMap<Region *, FlowState> regionEntry;
  DenseMap<Region *, FlowState> regionExit;
  std::set<unsigned> retiredSites;
  SmallPtrSet<Operation *, 8> deallocOperations;
  bool waiting = false;
  unsigned maxRounds = 0;
};

static bool hasCompleteSiteCoverage(const LivenessMaps &maps,
                                    const FlowResult &flow) {
  return maps.siteIds == flow.visitedSiteIds;
}

static void markFunctionUnsupported(FunctionLivenessFacts &facts,
                                    const LivenessMaps &maps,
                                    const FlowResult &flow, StringRef reason) {
  facts.complete = false;
  facts.coverageComplete = hasCompleteSiteCoverage(maps, flow);
  if (facts.reason.empty())
    facts.reason = reason.str();
}

static FunctionLivenessFacts
analyzeFunctionLiveness(func::FuncOp function,
                        ArrayRef<AllocationSite> allSites,
                        const DenseMap<Operation *, uint64_t> &canonicalSites) {
  FunctionLivenessFacts facts;
  facts.symbol = function.getSymName().str();
  LivenessMaps maps;
  FlowResult flow;
  bool hasPreBufferizationCrouton = false;
  bool mappingFailed = false;

  for (const AllocationSite &site : allSites) {
    if (site.operation->getParentOfType<func::FuncOp>() != function)
      continue;
    ++facts.allocationSites;
    if (!maps.siteIds.insert(site.id).second ||
        !maps.siteByOperation.try_emplace(site.operation, site.id).second ||
        !maps.siteByValue.try_emplace(site.value, site.id).second ||
        !maps.siteById.try_emplace(site.id, &site).second) {
      mappingFailed = true;
      continue;
    }
    facts.siteIds.insert(site.id);
    hasPreBufferizationCrouton |= isa<AllocCroutonOp>(site.operation);
    if (site.constantBoundedExtent)
      ++facts.constantBoundedSites;
    if (!site.resident)
      continue;
    bool ok = site.operation->hasAttr(kWeightResidentAttr)
                  ? addBytes(facts.weightResidentBytes, site.bytes)
                  : addBytes(facts.workspaceResidentBytes, site.bytes);
    if (!ok)
      mappingFailed = true;
  }

  if (mappingFailed) {
    markFunctionUnsupported(facts, maps, flow, "invalid allocation-site map");
    return facts;
  }
  for (Type resultType : function.getResultTypes()) {
    if (isVtcmType(resultType)) {
      markFunctionUnsupported(facts, maps, flow,
                              "function returns an untracked VTCM value");
      return facts;
    }
  }
  for (BlockArgument argument : function.getArguments()) {
    if (isVtcmValue(argument)) {
      markFunctionUnsupported(facts, maps, flow,
                              "function accepts an untracked VTCM value");
      return facts;
    }
  }
  if (function.getBody().empty()) {
    if (!facts.siteIds.empty())
      markFunctionUnsupported(facts, maps, flow,
                              "allocation site is outside a function body");
    return facts;
  }
  if (hasPreBufferizationCrouton) {
    markFunctionUnsupported(facts, maps, flow,
                            "pre-bufferization crouton allocation");
    return facts;
  }

  // Alias facts are frozen before any program point is evaluated, so a
  // path-specific producer discovered later cannot retroactively change an
  // already-published peak.
  ProvenanceAnalysis provenance(maps, flow);
  if (!provenance.run(function)) {
    facts.complete = false;
    facts.coverageComplete = hasCompleteSiteCoverage(maps, flow);
    facts.reason = flow.reason.empty() ? "alias provenance is not proven"
                                       : flow.reason;
    return facts;
  }

  LivenessFlow engine(function, maps, provenance, flow);
  engine.run();
  facts.coverageComplete = hasCompleteSiteCoverage(maps, flow);
  if (!flow.complete || !facts.coverageComplete) {
    facts.complete = false;
    if (facts.reason.empty()) {
      if (!flow.reason.empty())
        facts.reason = flow.reason;
      else if (!facts.coverageComplete)
        facts.reason = "an allocation site was not reached";
      else
        facts.reason = "missing deallocation";
    }
    return facts;
  }

  facts.deallocationSites = flow.deallocationSites;
  facts.transientPeakBytes = engine.peakBytes();
  facts.fixpointRounds = flow.fixpointRounds;
  facts.revisedBlocks = flow.revisedBlocks;
  facts.peakSiteCount = static_cast<int64_t>(engine.peakSiteIds().size());
  // Peak attribution uses the canonical static site identity, never a census
  // ordinal: a walk-order index is not an identity, so publishing it as a site
  // ID would make the record unstable under any reordering of the module.
  bool allCanonical = true;
  for (unsigned id : engine.peakSiteIds()) {
    const AllocationSite *site = maps.siteById.lookup(id);
    auto canonical = canonicalSites.find(site->operation);
    if (canonical == canonicalSites.end()) {
      allCanonical = false;
      break;
    }
    facts.peakSiteIds.push_back(static_cast<int64_t>(canonical->second));
  }
  facts.peakSiteIdsProven = allCanonical;
  llvm::sort(facts.peakSiteIds);
  facts.peakSiteIds.erase(
      std::unique(facts.peakSiteIds.begin(), facts.peakSiteIds.end()),
      facts.peakSiteIds.end());
  if (!addBytes(facts.modeledRequestedPeakBytes, facts.transientPeakBytes) ||
      !addBytes(facts.modeledRequestedPeakBytes,
                facts.workspaceResidentBytes) ||
      !addBytes(facts.modeledRequestedPeakBytes, facts.weightResidentBytes)) {
    facts.complete = false;
    facts.coverageComplete = false;
    facts.reason = "integer-overflow";
  }
  assert(!facts.complete || facts.coverageComplete);
  return facts;
}

static void setLivenessFailure(LivenessAccounting &result, StringRef reason) {
  result.complete = false;
  if (result.reason.empty())
    result.reason = reason.str();
}

static LivenessAccounting
analyzeModuleLiveness(ModuleOp module, ArrayRef<AllocationSite> allSites,
                      bool rawComplete,
                      const DenseMap<Operation *, uint64_t> &canonicalSites) {
  LivenessAccounting result;
  result.complete = rawComplete;
  result.coverageComplete = rawComplete;

  module.walk([&](memref::GlobalOp global) {
    if (isVtcmType(global.getType()))
      setLivenessFailure(result, "VTCM global storage is unsupported");
  });

  std::set<unsigned> claimedSiteIds;
  module.walk([&](func::FuncOp function) {
    if (function.isDeclaration()) {
      FunctionLivenessFacts facts;
      facts.symbol = function.getSymName().str();
      facts.applicable = false;
      facts.coverageComplete = true;
      facts.reason = "external declaration has no body to analyze";
      result.functions.push_back(std::move(facts));
      return;
    }
    FunctionLivenessFacts facts =
        analyzeFunctionLiveness(function, allSites, canonicalSites);
    claimedSiteIds.insert(facts.siteIds.begin(), facts.siteIds.end());
    if (!facts.coverageComplete)
      result.coverageComplete = false;
    if (!facts.complete)
      setLivenessFailure(result, facts.reason);
    result.functions.push_back(std::move(facts));
  });

  std::set<unsigned> expectedSiteIds;
  for (const AllocationSite &site : allSites)
    expectedSiteIds.insert(site.id);
  if (claimedSiteIds != expectedSiteIds) {
    result.coverageComplete = false;
    setLivenessFailure(result, "allocation site is outside function coverage");
  }
  if (!rawComplete && result.reason.empty())
    result.reason = "raw census is incomplete";
  assert(!result.complete ||
         (result.coverageComplete && claimedSiteIds == expectedSiteIds));
  return result;
}
static DictionaryAttr buildLivenessAttr(ModuleOp module,
                                        const LivenessAccounting &liveness) {
  MLIRContext *context = module.getContext();
  auto i64 = IntegerType::get(context, 64);
  SmallVector<Attribute> functionAttrs;
  for (const FunctionLivenessFacts &facts : liveness.functions) {
    NamedAttrList fields;
    fields.append("symbol", StringAttr::get(context, facts.symbol));
    StringRef functionStatus =
        !facts.applicable
            ? kStatusNotApplicable
            : (facts.complete ? kStatusComplete : kStatusIncomplete);
    StringRef coverageStatus =
        !facts.applicable
            ? kStatusNotApplicable
            : (facts.coverageComplete ? kStatusComplete : kStatusIncomplete);
    fields.append("status", StringAttr::get(context, functionStatus));
    fields.append("allocation_sites",
                  IntegerAttr::get(i64, facts.allocationSites));
    fields.append("allocation_site_coverage",
                  StringAttr::get(context, coverageStatus));
    if (facts.applicable && facts.complete) {
      fields.append("peak_status",
                    StringAttr::get(context, kPeakStatusStructured));
      fields.append("deallocation_sites",
                    IntegerAttr::get(i64, facts.deallocationSites));
      fields.append("transient_requested_peak_bytes",
                    IntegerAttr::get(i64, facts.transientPeakBytes));
      fields.append("workspace_resident_requested_bytes",
                    IntegerAttr::get(i64, facts.workspaceResidentBytes));
      fields.append("weight_resident_requested_bytes",
                    IntegerAttr::get(i64, facts.weightResidentBytes));
      fields.append("modeled_requested_peak_bytes",
                    IntegerAttr::get(i64, facts.modeledRequestedPeakBytes));
      // Peak site attribution is the canonical static site identity.  When a
      // peak site has no provable canonical identity the IDs are withheld
      // rather than filled with census ordinals, which are walk order and not
      // identity.  The peak *bytes* stay proven either way: they do not depend
      // on identity, and the byte figure is explicitly an upper bound.
      fields.append("peak_site_count",
                    IntegerAttr::get(i64, facts.peakSiteCount));
      if (facts.peakSiteIdsProven) {
        fields.append("peak_site_id_status",
                      StringAttr::get(context, kPeakSiteIdCanonical));
        fields.append("peak_site_ids",
                      DenseI64ArrayAttr::get(context, facts.peakSiteIds));
      } else {
        fields.append("peak_site_id_status",
                      StringAttr::get(context, kStatusNotProven));
      }
      fields.append("constant_bounded_extent_sites",
                    IntegerAttr::get(i64, facts.constantBoundedSites));
      fields.append("fixpoint_rounds",
                    IntegerAttr::get(i64, facts.fixpointRounds));
      fields.append("revised_blocks",
                    IntegerAttr::get(i64, facts.revisedBlocks));
    }
    if (!facts.reason.empty())
      fields.append("reason", StringAttr::get(context, facts.reason));
    functionAttrs.push_back(DictionaryAttr::get(context, fields));
  }

  size_t definitionCount = 0;
  size_t externalDeclarationCount = 0;
  for (const FunctionLivenessFacts &facts : liveness.functions) {
    if (facts.applicable)
      ++definitionCount;
    else
      ++externalDeclarationCount;
  }

  NamedAttrList fields;
  fields.append(
      "kind", StringAttr::get(context, kLivenessKindStructuredAllocatorEvents));
  fields.append("status", StringAttr::get(context, liveness.complete
                                                       ? kStatusComplete
                                                       : kStatusIncomplete));
  fields.append("unit", StringAttr::get(context, "requested-bytes"));
  fields.append("scope",
                StringAttr::get(context, "per-function-single-invocation"));
  fields.append("function_coverage",
                StringAttr::get(context, "definitions-only"));
  fields.append("definition_count", IntegerAttr::get(i64, definitionCount));
  fields.append("external_declaration_count",
                IntegerAttr::get(i64, externalDeclarationCount));
  fields.append("control_flow", StringAttr::get(context, kLivenessControlFlow));
  fields.append("alias_policy", StringAttr::get(context, kLivenessAliasPolicy));
  fields.append("call_policy", StringAttr::get(context, kLivenessCallPolicy));
  fields.append("async_policy", StringAttr::get(context, kLivenessAsyncPolicy));
  fields.append("extent_policy", StringAttr::get(context, kLivenessExtentPolicy));
  fields.append("join_policy", StringAttr::get(context, kLivenessJoinPolicy));
  fields.append("allocation_site_coverage",
                StringAttr::get(context, liveness.coverageComplete
                                             ? kStatusComplete
                                             : kStatusIncomplete));
  fields.append("allocator_peak_status",
                StringAttr::get(context, kStatusNotProven));
  fields.append("grid_status", StringAttr::get(context, kStatusNotProven));
  fields.append("resident_runtime_state",
                StringAttr::get(context, kStatusNotProven));
  fields.append("fragmentation_status",
                StringAttr::get(context, kStatusNotProven));
  fields.append("free_cache_retention_status",
                StringAttr::get(context, kStatusNotProven));
  fields.append("functions", ArrayAttr::get(context, functionAttrs));
  if (!liveness.reason.empty())
    fields.append("reason", StringAttr::get(context, liveness.reason));
  return DictionaryAttr::get(context, fields);
}

static void addIdentityReason(StaticSiteIdentity &site, StringRef reason) {
  if (site.reason.find(reason.str()) != std::string::npos)
    return;
  if (!site.reason.empty())
    site.reason += "; ";
  site.reason += reason.str();
}

static void addFunctionIdentityReason(StaticFunctionIdentity &function,
                                      StringRef reason) {
  if (function.reason.find(reason.str()) != std::string::npos)
    return;
  if (!function.reason.empty())
    function.reason += "; ";
  function.reason += reason.str();
}

static void invalidateSiteIdentity(StaticSiteIdentity &site, StringRef reason,
                                   StaticIdentityStatus status) {
  site.siteIdValid = false;
  site.status = mergeIdentityStatus(site.status, status);
  addIdentityReason(site, reason);
}

static StaticIdentityResult
buildStaticIdentity(ModuleOp module,
                    ArrayRef<StaticAllocationSite> candidates) {
  StaticIdentityResult result;

  result.principal = residentPrincipalName(module);
  if (result.principal != "<anonymous-principal>") {
    result.module = result.principal;
    result.principalStatus = "module-symbol";
  } else {
    result.principalStatus = kStatusNotProven;
    result.status = StaticIdentityStatus::kNotProven;
    result.reasons.insert(kIdentityReasonMissingModule.str());
  }
  if (result.principalStatus == "module-symbol" && !result.module.empty()) {
    result.scopeId = diagnosticScopeIdentity(result.principal, result.module);
    if (result.scopeId == 0) {
      result.status = StaticIdentityStatus::kIncomplete;
      result.reasons.insert(kIdentityReasonZeroHash.str());
    }
  }

  ExplicitBuildIdentity build = getExplicitBuildIdentity(module);
  if (!build.present) {
    result.buildIdStatus = kStatusNotProven;
    result.buildIdSource = kBuildIdNotPresent;
  } else if (!build.valid) {
    result.buildIdStatus = kStatusNotProven;
    result.buildIdSource = kBuildIdMalformed;
  } else {
    result.buildId = build.value;
    result.buildIdStatus = kBuildIdExternallySupplied;
    result.buildIdSource = build.source;
  }

  std::vector<func::FuncOp> functions;
  module.walk([&](func::FuncOp function) { functions.push_back(function); });
  std::sort(functions.begin(), functions.end(),
            [](func::FuncOp lhs, func::FuncOp rhs) {
              return lhs.getSymName().str() < rhs.getSymName().str();
            });

  DenseMap<Operation *, unsigned> functionIndex;
  std::map<std::string, SmallVector<unsigned>> symbolOwners;
  for (unsigned index = 0; index < functions.size(); ++index) {
    func::FuncOp function = functions[index];
    StaticFunctionIdentity facts;
    facts.operation = function.getOperation();
    facts.symbol = function.getSymName().str();
    symbolOwners[facts.symbol].push_back(index);
    result.functions.push_back(std::move(facts));
    functionIndex[function.getOperation()] = index;
  }
  for (const auto &entry : symbolOwners) {
    if (entry.second.size() < 2)
      continue;
    for (unsigned index : entry.second) {
      result.functions[index].ambiguous = true;
      result.functions[index].functionIdValid = false;
      result.functions[index].status = StaticIdentityStatus::kNotProven;
      addFunctionIdentityReason(result.functions[index],
                                kIdentityReasonMissingFunction);
    }
  }

  for (StaticFunctionIdentity &function : result.functions) {
    if (function.ambiguous)
      continue;
    if (function.symbol.empty()) {
      function.functionIdValid = false;
      function.status = StaticIdentityStatus::kNotProven;
      addFunctionIdentityReason(function, kIdentityReasonMissingFunction);
      continue;
    }
    if (result.principal.empty()) {
      function.functionIdValid = false;
      function.status = StaticIdentityStatus::kNotProven;
      addFunctionIdentityReason(function, kIdentityReasonMissingModule);
      continue;
    }
    // Reuse the versioned resident-key identity vocabulary. No resident
    // key/content provenance is read here, and accounting-schema revisions do
    // not alter this hash domain.
    function.functionId =
        residentFunctionIdentity(result.principal, function.symbol);
    if (function.functionId == 0) {
      function.functionIdValid = false;
      function.status = StaticIdentityStatus::kIncomplete;
      addFunctionIdentityReason(function, kIdentityReasonZeroHash);
      continue;
    }
    function.functionIdValid = true;
    function.status = StaticIdentityStatus::kComplete;
  }

  std::map<uint64_t, SmallVector<unsigned>> functionIdOwners;
  for (unsigned index = 0; index < result.functions.size(); ++index)
    if (result.functions[index].functionIdValid)
      functionIdOwners[result.functions[index].functionId].push_back(index);
  for (const auto &entry : functionIdOwners) {
    if (entry.second.size() < 2)
      continue;
    for (unsigned index : entry.second) {
      result.functions[index].functionIdValid = false;
      result.functions[index].status = StaticIdentityStatus::kIncomplete;
      addFunctionIdentityReason(result.functions[index],
                                kIdentityReasonFunctionCollision);
    }
  }

  // Sort the static facts before assigning records. This is presentation
  // determinism only; IDs below never use the resulting index.
  SmallVector<unsigned> candidateOrder;
  for (unsigned index = 0; index < candidates.size(); ++index)
    candidateOrder.push_back(index);
  auto sourceFor = [&](unsigned index) -> std::string {
    if (!candidates[index].operation)
      return {};
    auto source = getStaticSourceFact(candidates[index].operation->getLoc());
    return source ? source->canonical : std::string();
  };
  auto functionNameFor = [&](unsigned index) -> std::string {
    if (!candidates[index].function)
      return {};
    if (auto function = dyn_cast<func::FuncOp>(candidates[index].function))
      return function.getSymName().str();
    return {};
  };
  std::sort(candidateOrder.begin(), candidateOrder.end(),
            [&](unsigned lhs, unsigned rhs) {
              if (candidates[lhs].role != candidates[rhs].role)
                return candidates[lhs].role < candidates[rhs].role;
              if (functionNameFor(lhs) != functionNameFor(rhs))
                return functionNameFor(lhs) < functionNameFor(rhs);
              return sourceFor(lhs) < sourceFor(rhs);
            });

  StaticFunctionIdentity *unresolvedFunction = nullptr;
  auto getUnresolvedFunction = [&]() -> StaticFunctionIdentity & {
    if (!unresolvedFunction) {
      StaticFunctionIdentity facts;
      facts.symbol = "<unresolved-function-scope>";
      facts.functionIdValid = false;
      addFunctionIdentityReason(facts, kIdentityReasonMissingFunction);
      result.functions.push_back(std::move(facts));
      unresolvedFunction = &result.functions.back();
    }
    return *unresolvedFunction;
  };

  for (unsigned candidateIndex : candidateOrder) {
    const StaticAllocationSite &candidate = candidates[candidateIndex];
    StaticSiteIdentity site;
    site.site = &candidate;
    site.role = candidate.role;
    site.slot = candidate.slot;
    if (result.principal.empty()) {
      site.status = StaticIdentityStatus::kNotProven;
      addIdentityReason(site, kIdentityReasonMissingModule);
    }

    StaticFunctionIdentity *siteFunction = nullptr;
    if (candidate.functionAmbiguous || !candidate.function) {
      siteFunction = &getUnresolvedFunction();
      site.status = StaticIdentityStatus::kNotProven;
      addIdentityReason(site, kIdentityReasonMissingFunction);
    } else {
      auto functionIt = functionIndex.find(candidate.function);
      if (functionIt == functionIndex.end()) {
        siteFunction = &getUnresolvedFunction();
        site.status = StaticIdentityStatus::kNotProven;
        addIdentityReason(site, kIdentityReasonMissingFunction);
      } else {
        siteFunction = &result.functions[functionIt->second];
        site.function = siteFunction->symbol;
        site.functionId = siteFunction->functionId;
        site.functionIdValid = siteFunction->functionIdValid;
        if (!site.functionIdValid) {
          site.status = mergeIdentityStatus(site.status, siteFunction->status);
          if (!siteFunction->reason.empty())
            addIdentityReason(site, siteFunction->reason);
        }
      }
    }
    if (siteFunction)
      siteFunction->siteIndices.push_back(result.sites.size());

    if (candidate.operation) {
      site.source = getStaticSourceFact(candidate.operation->getLoc());
      if (!site.source) {
        site.status = StaticIdentityStatus::kNotProven;
        addIdentityReason(site, kIdentityReasonUnknownSource);
      }
    } else {
      site.status = StaticIdentityStatus::kNotProven;
      addIdentityReason(site, kIdentityReasonUnknownSource);
    }

    if (!isReviewedStaticRole(site.role)) {
      site.status = StaticIdentityStatus::kNotProven;
      addIdentityReason(site, kIdentityReasonUnknownRole);
    } else if ((candidate.weightResident || candidate.workspaceResident) &&
               (!candidate.provenanceProven || !candidate.slotProven)) {
      site.status = StaticIdentityStatus::kNotProven;
      if (candidate.provenanceReason.empty())
        addIdentityReason(site, kIdentityReasonResidentProvenanceMissing);
      else
        addIdentityReason(site, candidate.provenanceReason);
    } else if (site.functionIdValid && site.source && !site.function.empty() &&
               !result.principal.empty() && result.scopeId != 0) {
      site.siteId =
          residentSiteIdentity(result.principal, site.function,
                               site.source->canonical, site.role, site.slot);
      site.eventToken = diagnosticEventToken(
          result.principal, result.module, site.function, site.source->canonical,
          /*invocationId=*/1);
      if (site.siteId == 0 || site.eventToken.isZero()) {
        invalidateSiteIdentity(site, kIdentityReasonZeroHash,
                               StaticIdentityStatus::kIncomplete);
      } else {
        site.siteIdValid = true;
        site.status = StaticIdentityStatus::kComplete;
      }
    }
    result.sites.push_back(std::move(site));
  }

  // A repeated source/role/slot tuple is not made unique by a traversal
  // ordinal. A different reviewed role at the same source is a distinct
  // semantic site; an unreviewed role never receives an ID. Likewise, a hash
  // collision is not evidence of identity.
  std::map<std::string, SmallVector<unsigned>> tupleOwners;
  std::map<uint64_t, SmallVector<unsigned>> siteIdOwners;
  std::map<std::pair<uint64_t, uint64_t>, SmallVector<unsigned>> tokenOwners;
  for (unsigned index = 0; index < result.sites.size(); ++index) {
    StaticSiteIdentity &site = result.sites[index];
    if (!site.siteIdValid)
      continue;
    std::string tuple =
        makeStaticIdentityTuple(result.principal, site.function,
                                site.source->canonical, site.role, site.slot);
    tupleOwners[tuple].push_back(index);
    siteIdOwners[site.siteId].push_back(index);
    tokenOwners[{site.eventToken.low, site.eventToken.high}].push_back(index);
  }
  for (const auto &entry : tupleOwners) {
    if (entry.second.size() < 2)
      continue;
    for (unsigned index : entry.second)
      invalidateSiteIdentity(result.sites[index], kIdentityReasonDuplicateSite,
                             StaticIdentityStatus::kIncomplete);
  }
  for (const auto &entry : siteIdOwners) {
    if (entry.second.size() < 2)
      continue;
    bool sameTuple = true;
    std::string firstTuple;
    for (unsigned index : entry.second) {
      StaticSiteIdentity &site = result.sites[index];
      std::string tuple =
          makeStaticIdentityTuple(result.principal, site.function,
                                  site.source->canonical, site.role, site.slot);
      if (firstTuple.empty())
        firstTuple = tuple;
      else if (firstTuple != tuple)
        sameTuple = false;
    }
    for (unsigned index : entry.second)
      invalidateSiteIdentity(result.sites[index],
                             sameTuple ? kIdentityReasonDuplicateSite
                                       : kIdentityReasonSiteCollision,
                             StaticIdentityStatus::kIncomplete);
  }

  for (const auto &entry : tokenOwners) {
    if (entry.second.size() < 2)
      continue;
    for (unsigned index : entry.second)
      invalidateSiteIdentity(result.sites[index],
                             kIdentityReasonSiteCollision,
                             StaticIdentityStatus::kIncomplete);
  }

  for (const StaticFunctionIdentity &function : result.functions) {
    result.status = mergeIdentityStatus(result.status, function.status);
    if (!function.reason.empty())
      result.reasons.insert(function.reason);
  }
  for (const StaticSiteIdentity &site : result.sites) {
    result.status = mergeIdentityStatus(result.status, site.status);
    if (!site.reason.empty())
      result.reasons.insert(site.reason);
  }
  return result;
}

static DictionaryAttr
buildStaticIdentityAttr(ModuleOp module, const StaticIdentityResult &identity) {
  MLIRContext *context = module.getContext();
  auto i64 = IntegerType::get(context, 64);
  SmallVector<Attribute> functionAttrs;
  for (const StaticFunctionIdentity &function : identity.functions) {
    StaticIdentityStatus functionStatus = function.status;
    for (unsigned siteIndex : function.siteIndices)
      functionStatus =
          mergeIdentityStatus(functionStatus, identity.sites[siteIndex].status);
    NamedAttrList fields;
    fields.append("symbol", StringAttr::get(context, function.symbol));
    fields.append("function", StringAttr::get(context, function.symbol));
    if (function.functionIdValid)
      fields.append("function_id", u64Attribute(context, function.functionId));
    fields.append(
        "identity_status",
        StringAttr::get(context, staticIdentityStatus(functionStatus)));
    fields.append("function_scope",
                  StringAttr::get(context, function.operation
                                               ? "single-function"
                                               : "unresolved"));
    if (!function.reason.empty())
      fields.append("reason", StringAttr::get(context, function.reason));
    SmallVector<Attribute> siteAttrs;
    for (unsigned siteIndex : function.siteIndices) {
      const StaticSiteIdentity &site = identity.sites[siteIndex];
      NamedAttrList siteFields;
      if (site.siteIdValid) {
        siteFields.append("site_id", u64Attribute(context, site.siteId));
        siteFields.append(
            "event_token_bits",
            IntegerAttr::get(i64, 128));
        siteFields.append("event_token_low",
                          u64Attribute(context, site.eventToken.low));
        siteFields.append("event_token_high",
                          u64Attribute(context, site.eventToken.high));
        siteFields.append("event_token_basis",
                          StringAttr::get(context, kHmxDiagnosticEventTokenSchema));
      }
      if (site.functionIdValid)
        siteFields.append("function_id", u64Attribute(context, site.functionId));
      siteFields.append(
          "identity_status",
          StringAttr::get(context, staticIdentityStatus(site.status)));
      if (!identity.principal.empty()) {
        siteFields.append("principal",
                          StringAttr::get(context, identity.principal));
        siteFields.append("principal_status",
                          StringAttr::get(context, identity.principalStatus));
      }
      if (!site.function.empty())
        siteFields.append("function", StringAttr::get(context, site.function));
      siteFields.append("role", StringAttr::get(context, site.role));
      siteFields.append("role_status",
                        StringAttr::get(context, isReviewedStaticRole(site.role)
                                                     ? "reviewed"
                                                     : kStatusNotProven));
      siteFields.append("slot", IntegerAttr::get(i64, site.slot));
      siteFields.append(
          "slot_status",
          StringAttr::get(context, site.site->slotProven ? "provenance"
                                                          : kStatusNotProven));
      if (site.source) {
        siteFields.append("source",
                          StringAttr::get(context, site.source->canonical));
        siteFields.append("source_file",
                          StringAttr::get(context, site.source->file));
        siteFields.append("source_line",
                          IntegerAttr::get(i64, site.source->line));
        siteFields.append("source_column",
                          IntegerAttr::get(i64, site.source->column));
      } else {
        siteFields.append("source_status",
                          StringAttr::get(context, kStatusNotProven));
      }
      siteFields.append("size_status",
                        StringAttr::get(context, site.site->bytesKnown
                                                     ? kStatusComplete
                                                     : kStatusNotProven));
      // Distinguish an exact size read off a statically shaped type from an
      // exact size that rests on a constant bound for a dynamic extent.  Both
      // figures are exact; only the second needed an extent proof, and an
      // unproven extent never reaches this record as a zero.
      StringRef extentStatus = kStatusNotProven;
      if (site.site->bytesKnown)
        extentStatus = site.site->constantBoundedExtent
                           ? StringRef("constant-upper-bound")
                           : StringRef("static-shape");
      siteFields.append("extent_status",
                        StringAttr::get(context, extentStatus));
      if (site.site->bytesKnown)
        siteFields.append("requested_bytes",
                          IntegerAttr::get(i64, site.site->bytes));
      siteFields.append("static_fact", StringAttr::get(context, "true"));
      siteFields.append("runtime_observation_status",
                        StringAttr::get(context, kRuntimeJoinNotIntegrated));
      siteFields.append(
          "resident_provenance_status",
          StringAttr::get(context,
                          (site.site->weightResident ||
                           site.site->workspaceResident)
                              ? (site.site->provenanceProven ? "checked"
                                                             : kStatusNotProven)
                              : "none"));
      if (!site.reason.empty())
        siteFields.append("reason", StringAttr::get(context, site.reason));
      siteAttrs.push_back(DictionaryAttr::get(context, siteFields));
    }
    fields.append("sites", ArrayAttr::get(context, siteAttrs));
    functionAttrs.push_back(DictionaryAttr::get(context, fields));
  }

  NamedAttrList fields;
  fields.append("kind", StringAttr::get(context, kIdentityKind));
  fields.append("schema", StringAttr::get(context, kHmxVtcmIdentitySchema));
  fields.append("id_schema", StringAttr::get(context, kHmxResidentKeySchema));
  StringRef moduleStatus = staticIdentityStatus(identity.status);
  fields.append("status", StringAttr::get(context, moduleStatus));
  fields.append("identity_status", StringAttr::get(context, moduleStatus));
  fields.append("scope", StringAttr::get(context, kIdentityScope));
  fields.append("function_id_basis",
                StringAttr::get(context, "module-principal+function-symbol"));
  fields.append(
      "site_id_basis",
      StringAttr::get(context,
                      "module-principal+function-symbol+source+role+slot"));
  fields.append(
      "liveness_site_id_status",
      StringAttr::get(context, "separate-traversal-index-not-join-key"));
  unsigned realFunctionCount = 0;
  unsigned unresolvedFunctionCount = 0;
  for (const StaticFunctionIdentity &function : identity.functions) {
    if (function.operation)
      ++realFunctionCount;
    else
      ++unresolvedFunctionCount;
  }
  StringRef multiFunctionStatus = kStatusNotProven;
  if (unresolvedFunctionCount)
    multiFunctionStatus = kStatusIncomplete;
  else if (realFunctionCount < 2)
    multiFunctionStatus = "not-applicable";
  else if (identity.principalStatus == "module-symbol")
    multiFunctionStatus = "per-function-separated";
  fields.append("multi_function_status",
                StringAttr::get(context, multiFunctionStatus));
  fields.append("source", StringAttr::get(context, kIdentitySource));
  fields.append("runtime_join_status",
                StringAttr::get(context, kRuntimeJoinNotIntegrated));
  fields.append("join_status", StringAttr::get(context, kStatusNotProven));
  fields.append("build_id_status",
                StringAttr::get(context, identity.buildIdStatus));
  fields.append("build_id_source",
                StringAttr::get(context, identity.buildIdSource));
  if (identity.buildId)
    fields.append("build_id", identity.buildId);
  fields.append("resident_scope_id_status",
                StringAttr::get(context, kStatusNotProven));
  fields.append("resident_provenance",
                StringAttr::get(context, "separate-from-allocator-site"));
  fields.append("principal_status",
                StringAttr::get(context, identity.principalStatus));
  if (!identity.principal.empty()) {
    fields.append("principal", StringAttr::get(context, identity.principal));
    fields.append("module", StringAttr::get(context, identity.module));
  }
  if (identity.scopeId != 0)
    fields.append("scope_id", u64Attribute(context, identity.scopeId));
  fields.append("scope_id_status",
                StringAttr::get(context, identity.scopeId == 0
                                               ? kStatusNotProven
                                               : "compiler-static"));
  fields.append("function_count",
                IntegerAttr::get(i64, identity.functions.size()));
  fields.append("unresolved_function_count",
                IntegerAttr::get(i64, unresolvedFunctionCount));
  fields.append("site_count", IntegerAttr::get(i64, identity.sites.size()));
  fields.append("functions", ArrayAttr::get(context, functionAttrs));
  if (!identity.reasons.empty()) {
    SmallVector<Attribute> reasons;
    for (const std::string &reason : identity.reasons)
      reasons.push_back(StringAttr::get(context, reason));
    fields.append("reasons", ArrayAttr::get(context, reasons));
  }
  return DictionaryAttr::get(context, fields);
}

static bool hasUnsupportedEventContextOperation(ModuleOp module) {
  bool unsupported = false;
  module.walk([&](Operation *operation) {
    if (auto function = dyn_cast<func::FuncOp>(operation);
        function && function->hasAttr("async")) {
      unsupported = true;
      return;
    }
    if (isa<CallOpInterface>(operation) || isa<scf::ParallelOp>(operation) ||
        isa<memref::ExtractAlignedPointerAsIndexOp>(operation)) {
      unsupported = true;
      return;
    }
    Dialect *dialect = operation->getDialect();
    if (dialect && (dialect->getNamespace() == "async" ||
                    dialect->getNamespace() == "gpu")) {
      unsupported = true;
      return;
    }
  });
  if (auto grid = module->getAttrOfType<IntegerAttr>("hmx.kernel_vtcm_grid"))
    unsupported |= grid.getInt() != 1;
  return unsupported;
}

static DictionaryAttr
buildEventContextAttr(ModuleOp module, const StaticIdentityResult &identity,
                      const Accounting &accounting) {
  MLIRContext *context = module.getContext();
  auto i64 = IntegerType::get(context, 64);

  const StaticSiteIdentity *selected = nullptr;
  unsigned realFunctions = 0;
  for (const StaticFunctionIdentity &function : identity.functions)
    if (function.operation)
      ++realFunctions;
  unsigned validSites = 0;
  for (const StaticSiteIdentity &site : identity.sites) {
    if (!site.siteIdValid || site.eventToken.isZero())
      continue;
    ++validSites;
    selected = &site;
  }

  auto gridAttr = module->getAttrOfType<IntegerAttr>("hmx.kernel_vtcm_grid");
  const bool explicitGridOne = gridAttr && gridAttr.getInt() == 1;
  std::string reason;
  bool eligible = identity.status == StaticIdentityStatus::kComplete &&
                  identity.principalStatus == "module-symbol" &&
                  identity.scopeId != 0 && identity.buildId &&
                  accounting.complete && realFunctions == 1 && validSites == 1 &&
                  explicitGridOne &&
                  !hasUnsupportedEventContextOperation(module);
  if (!eligible) {
    if (identity.status != StaticIdentityStatus::kComplete)
      reason = "static identity is not complete";
    else if (identity.principalStatus != "module-symbol" || identity.scopeId == 0)
      reason = "module/principal scope is not canonical";
    else if (!identity.buildId)
      reason = "explicit build identity is required for an event token";
    else if (!accounting.complete)
      reason = "allocation accounting is incomplete";
    else if (!explicitGridOne)
      reason = "explicit hmx.kernel_vtcm_grid=1 is required";
    else if (realFunctions != 1 || validSites != 1)
      reason = "event context requires one function and one canonical site";
    else
      reason = "call, async, dynamic-grid, or pointer-escape scope is unsupported";
  }

  NamedAttrList fields;
  fields.append("kind", StringAttr::get(context, "vtcm-event-context"));
  fields.append("schema", StringAttr::get(context, kHmxVtcmEventContextSchema));
  fields.append("status", StringAttr::get(context, eligible ? "complete"
                                                           : "not-proven"));
  fields.append("eligible", BoolAttr::get(context, eligible));
  fields.append("mode", StringAttr::get(context, "diagnostic-only"));
  fields.append("function",
                StringAttr::get(context, selected ? selected->function : ""));
  fields.append("grid_product", IntegerAttr::get(i64, 1));
  fields.append("invocation_id", IntegerAttr::get(i64, 1));
  fields.append("immutable", IntegerAttr::get(i64, eligible ? 1 : 0));
  fields.append("token_basis",
                StringAttr::get(context, kHmxDiagnosticEventTokenSchema));
  fields.append("delayed_cache_owner_status",
                StringAttr::get(context, "aggregate"));
  fields.append("grid_scope_status", StringAttr::get(context, "not-proven"));
  fields.append("runtime_event_join_status",
                StringAttr::get(context, "not-proven"));
  fields.append("performance_claimed", IntegerAttr::get(i64, 0));
  if (eligible && selected) {
    fields.append("accounting_scope_id", u64Attribute(context, identity.scopeId));
    fields.append("function_id", u64Attribute(context, selected->functionId));
    fields.append("allocation_site_id", u64Attribute(context, selected->siteId));
    fields.append("token_bits", IntegerAttr::get(i64, 128));
    fields.append("token_low",
                  u64Attribute(context, selected->eventToken.low));
    fields.append("token_high",
                  u64Attribute(context, selected->eventToken.high));
  } else {
    fields.append("reason", StringAttr::get(context, reason));
  }
  return DictionaryAttr::get(context, fields);
}

static DictionaryAttr
buildEvidenceContextAttr(ModuleOp module, const StaticIdentityResult &identity,
                         const Accounting &accounting) {
  MLIRContext *context = module.getContext();
  auto i64 = IntegerType::get(context, 64);

  unsigned realFunctions = 0;
  unsigned realSites = 0;
  for (const StaticFunctionIdentity &function : identity.functions)
    if (function.operation)
      ++realFunctions;
  for (const StaticSiteIdentity &site : identity.sites)
    if (site.siteIdValid)
      ++realSites;

  NamedAttrList scopeFields;
  scopeFields.append("principal", StringAttr::get(context, identity.principal));
  scopeFields.append("principal_status",
                     StringAttr::get(context, identity.principalStatus));
  scopeFields.append("function_count",
                     IntegerAttr::get(i64, realFunctions));
  scopeFields.append("canonical_site_count",
                     IntegerAttr::get(i64, realSites));
  scopeFields.append("grid_product", IntegerAttr::get(i64, 1));
  scopeFields.append("invocation_count", IntegerAttr::get(i64, 1));
  scopeFields.append("immutable", IntegerAttr::get(i64, 1));
  scopeFields.append(
      "status",
      StringAttr::get(context, realFunctions == 1 && realSites == 1
                                   ? "declared-static-scope"
                                   : kEvidenceContextNotProven));
  scopeFields.append(
      "reason",
      StringAttr::get(
          context,
          realFunctions == 1 && realSites == 1
              ? "static sidecar selects one function and one canonical site"
              : "one-function/one-site scope is not selected by this module"));
  NamedAttrList scopeAttr;
  scopeAttr.append("scope", StringAttr::get(context, kEvidenceContextScope));
  scopeAttr.append("fields", DictionaryAttr::get(context, scopeFields));

  NamedAttrList frameFields;
  frameFields.append("status", StringAttr::get(context, kEvidenceContextNotProven));
  frameFields.append("scope", StringAttr::get(context, kEvidenceContextScope));
  frameFields.append("event_owner_status",
                     StringAttr::get(context, kEvidenceContextAggregate));
  frameFields.append("delayed_cache_owner_status",
                     StringAttr::get(context, kEvidenceContextAggregate));
  frameFields.append(
      "reason",
      StringAttr::get(context,
                       "runtime v1 has no trustworthy per-event owner propagation; delayed cache events remain aggregate"));
  NamedAttrList frameAttr;
  frameAttr.append("kind", StringAttr::get(context, "probe-only-frame-contract"));
  frameAttr.append("fields", DictionaryAttr::get(context, frameFields));

  NamedAttrList identityFields;
  identityFields.append(
      "build_id_status", StringAttr::get(context, identity.buildIdStatus));
  identityFields.append(
      "function_id_status", StringAttr::get(context, kEvidenceContextNotProven));
  identityFields.append(
      "allocation_site_id_status",
      StringAttr::get(context, kEvidenceContextNotProven));
  identityFields.append("resident_scope_binding",
                        StringAttr::get(context, kEvidenceContextNotProven));
  identityFields.append("content_identity_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  identityFields.append("workspace_overwrite_proof_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  identityFields.append("global_symbol_identity_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  identityFields.append("packed_weight_source_view_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  identityFields.append("packed_weight_digest_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  identityFields.append(
      "reason",
      StringAttr::get(context,
                       "static identity is not a runtime event join; resident scope/object/content descriptors are separate"));
  if (identity.buildId)
    identityFields.append("build_id", identity.buildId);
  NamedAttrList identityAttr;
  identityAttr.append("kind", StringAttr::get(context, "identity-binding-contract"));
  identityAttr.append("fields", DictionaryAttr::get(context, identityFields));

  auto unit = [&](StringRef name, StringRef basis, StringRef status,
                  int64_t declared = -1) {
    NamedAttrList fields;
    fields.append("unit", StringAttr::get(context, "bytes"));
    fields.append("basis", StringAttr::get(context, basis));
    fields.append("status", StringAttr::get(context, status));
    if (declared >= 0)
      fields.append("declared_bytes", IntegerAttr::get(i64, declared));
    return DictionaryAttr::get(context, fields);
  };
  int64_t rawDeclared = -1;
  if (accounting.complete)
    rawDeclared = accounting.transientBytes + accounting.workspaceResidentBytes +
                  accounting.weightResidentBytes;
  NamedAttrList allocatorFields;
  allocatorFields.append(
      "raw_requested",
      unit("raw_requested", "compiler-requested", "declared-module-sum",
           rawDeclared));
  allocatorFields.append(
      "allocator_input",
      unit("allocator_input", "runtime-vtcm-entry", kEvidenceContextNotProven));
  allocatorFields.append(
      "size_aligned",
      unit("size_aligned", "runtime-size-rounding", kEvidenceContextNotProven));
  allocatorFields.append(
      "charged",
      unit("charged", "runtime-pool-block-length", kEvidenceContextNotProven));
  allocatorFields.append(
      "resident",
      unit("resident", "runtime-resident-block-charge", kEvidenceContextNotProven));
  allocatorFields.append(
      "observed",
      unit("observed", "runtime-process-high-water", kEvidenceContextNotProven));
  NamedAttrList allocatorAttr;
  allocatorAttr.append("units", DictionaryAttr::get(context, allocatorFields));
  allocatorAttr.append("unit_order",
                       StringAttr::get(context,
                                        "raw_requested<=allocator_input<=size_aligned<=charged"));
  allocatorAttr.append("pool_cache_combined_status",
                       StringAttr::get(context, kEvidenceContextNotProven));
  allocatorAttr.append("pool_cache_combined_reason",
                       StringAttr::get(context,
                                        "pool high-water and BufferManager retention are separate ledgers; full occupancy is not proven"));
  allocatorAttr.append("full_kernel_occupancy_status",
                       StringAttr::get(context, kEvidenceContextNotProven));

  NamedAttrList residentFields;
  residentFields.append("scope_binding_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("object_descriptor_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("content_descriptor_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("workspace_overwrite_proof_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("global_symbol_identity_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("packed_weight_source_view_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("packed_weight_digest_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("address_reuse_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("content_mutation_status",
                        StringAttr::get(context, kEvidenceContextNotProven));
  residentFields.append("reason",
                        StringAttr::get(context,
                                         "resident scope must bind object/content descriptors; source-specific overwrite/global/packed proofs are absent"));
  NamedAttrList residentAttr;
  residentAttr.append("contract", DictionaryAttr::get(context, residentFields));

  NamedAttrList evidenceFields;
  auto evidenceAxis = [&](StringRef axis, StringRef status, StringRef reason) {
    NamedAttrList fields;
    fields.append("status", StringAttr::get(context, status));
    fields.append("reason", StringAttr::get(context, reason));
    evidenceFields.append(axis, DictionaryAttr::get(context, fields));
  };
  evidenceAxis("cold_warm", kEvidenceContextNotProven,
               "same-scope process reset/join protocol is absent");
  evidenceAxis("free_cache", "observed-only",
               "cache transitions are aggregate and owner is not preserved by v1");
  evidenceAxis("fragmentation", "observed-only",
               "fixed free-list observations are not a general allocator model");
  evidenceAxis("content_mutation", kEvidenceContextNotProven,
               "resident reuse does not compare content");
  evidenceAxis("address_reuse", kEvidenceContextNotProven,
               "numeric source-address reuse is not distinguishable without content");
  evidenceAxis("multi_module", kEvidenceContextNotProven,
               "v1 has no module ownership contract");
  evidenceAxis("multi_process", kEvidenceContextNotProven,
               "v1 has no process reset/join contract");
  evidenceAxis("grid", kEvidenceContextNotProven,
               "v1 has no launch-instance identity");
  evidenceAxis("allocator_model", kEvidenceContextNotProven,
               "header/split/coalesce/retry model is not closed");
  NamedAttrList evidenceAttr;
  evidenceAttr.append("axes", DictionaryAttr::get(context, evidenceFields));

  SmallVector<Attribute> missing;
  for (StringRef invariant : {
           "runtime per-event owner frame is not proven",
           "delayed cache events remain aggregate",
           "resident scope/object/content binding is not proven",
           "workspace overwrite/global-symbol/packed-source proof is absent",
           "pool and BufferManager combined high-water is not a full occupancy proof"})
    missing.push_back(StringAttr::get(context, invariant));

  NamedAttrList fields;
  fields.append("kind", StringAttr::get(context, kEvidenceContextKind));
  fields.append("schema", StringAttr::get(context, kHmxVtcmEvidenceContextSchema));
  fields.append("status",
                StringAttr::get(context, kEvidenceContextStatusIncomplete));
  fields.append("mode", StringAttr::get(context, "diagnostic-only"));
  fields.append("performance_claimed", IntegerAttr::get(i64, 0));
  fields.append("scope", DictionaryAttr::get(context, scopeAttr));
  fields.append("frame", DictionaryAttr::get(context, frameAttr));
  fields.append("identity", DictionaryAttr::get(context, identityAttr));
  fields.append("allocator", DictionaryAttr::get(context, allocatorAttr));
  fields.append("resident", DictionaryAttr::get(context, residentAttr));
  fields.append("evidence", DictionaryAttr::get(context, evidenceAttr));
  fields.append("missing_invariants", ArrayAttr::get(context, missing));
  return DictionaryAttr::get(context, fields);
}

struct HmxVtcmAccountingPass
    : public mlir::hmx::impl::HmxVtcmAccountingBase<HmxVtcmAccountingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<HmxDialect, hexagonmem::HexagonMemDialect,
                crouton::CroutonDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect, cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    bool accountingMarker = isHmxDiagnosticVtcmAccountingMarker(module);
    bool livenessMarker = isHmxDiagnosticVtcmLivenessMarker(module);
    bool identityMarker = isHmxDiagnosticVtcmIdentityMarker(module);
    bool evidenceContextMarker =
        isHmxDiagnosticVtcmEvidenceContextMarker(module);
    if (module->hasAttr(kHmxDiagnosticVtcmAccountingAttr) &&
        !accountingMarker) {
      module.emitError(
          "hmx.diagnostic_vtcm_accounting must be a unit attribute");
      return signalPassFailure();
    }
    if (module->hasAttr(kHmxDiagnosticVtcmLivenessAttr) && !livenessMarker) {
      module.emitError("hmx.diagnostic_vtcm_liveness must be a unit attribute");
      return signalPassFailure();
    }
    if (livenessMarker && !accountingMarker) {
      module.emitError("hmx.diagnostic_vtcm_liveness requires "
                       "hmx.diagnostic_vtcm_accounting");
      return signalPassFailure();
    }
    if (module->hasAttr(kHmxDiagnosticVtcmIdentityAttr) && !identityMarker) {
      module.emitError("hmx.diagnostic_vtcm_identity must be a unit attribute");
      return signalPassFailure();
    }
    if (identityMarker && !accountingMarker) {
      module.emitError("hmx.diagnostic_vtcm_identity requires "
                       "hmx.diagnostic_vtcm_accounting");
      return signalPassFailure();
    }
    if (module->hasAttr(kHmxDiagnosticVtcmEvidenceContextAttr) &&
        !evidenceContextMarker) {
      module.emitError(
          "hmx.diagnostic_vtcm_evidence_context must be a unit attribute");
      return signalPassFailure();
    }
    if (evidenceContextMarker && !identityMarker) {
      module.emitError(
          "hmx.diagnostic_vtcm_evidence_context requires "
          "hmx.diagnostic_vtcm_identity");
      return signalPassFailure();
    }
    if (evidenceContextMarker && !accountingMarker) {
      module.emitError(
          "hmx.diagnostic_vtcm_evidence_context requires "
          "hmx.diagnostic_vtcm_accounting");
      return signalPassFailure();
    }
    if (!accountingMarker)
      return;

    Accounting accounting;
    module.walk([&](memref::GlobalOp global) {
      if (isVtcmType(global.getType())) {
        accounting.externalVtcm = "global";
        accounting.complete = false;
      }
    });
    module.walk([&](func::FuncOp function) {
      for (BlockArgument argument : function.getArguments()) {
        if (!isVtcmType(argument.getType()))
          continue;
        if (!accounting.externalVtcm.empty())
          accounting.externalVtcm += "+";
        accounting.externalVtcm += "function-argument";
        accounting.complete = false;
      }
    });
    SmallVector<StaticAllocationSite> identityCandidates;
    module.walk([&](func::FuncOp function) {
      for (BlockArgument argument : function.getArguments()) {
        if (function.getArgAttr(argument.getArgNumber(), "hexagon.scratch")) {
          accounting.externalScratch = true;
          accounting.complete = false;
        }
      }
    });
    module.walk([&](Operation *operation) {
      bool supportedAllocation =
          isa<bufferization::AllocTensorOp, memref::AllocOp, memref::AllocaOp,
              hexagonmem::AllocOp, AllocCroutonOp>(operation);
      bool hasResidentTag = operation->hasAttr(kWeightResidentAttr) ||
                            operation->hasAttr(kWorkspaceResidentAttr);
      if (hasResidentTag && !supportedAllocation) {
        ++accounting.unknownAllocations;
        accounting.complete = false;
      }
      if (supportedAllocation) {
        if (isVTCMIdentityCandidate(operation)) {
          StaticAllocationSite candidate;
          candidate.operation = operation;
          candidate.weightResident = operation->hasAttr(kWeightResidentAttr);
          candidate.workspaceResident =
              operation->hasAttr(kWorkspaceResidentAttr);
          // Resident provenance is consulted only for the strict identity
          // sidecar. The raw census remains a separate byte ledger; a missing
          // or malformed descriptor is represented as not-proven rather than
          // manufacturing a slot or source join.
          candidate.role = getStaticAllocationRole(
              operation, candidate.weightResident, candidate.workspaceResident);
          candidate.functionAmbiguous = false;
          candidate.function =
              getUniqueFunctionScope(operation, candidate.functionAmbiguous);
          SizeResult identitySize = getAllocationSize(operation);
          candidate.bytesKnown = identitySize.known && identitySize.bytes >= 0;
          candidate.bytes = identitySize.bytes;
          candidate.constantBoundedExtent = identitySize.constantBoundedExtent;
          if (candidate.weightResident || candidate.workspaceResident) {
            ResidentProvenanceFacts provenance = validateResidentProvenance(
                operation, module, candidate.role, candidate.bytes);
            candidate.slot = provenance.slot;
            candidate.slotProven = provenance.valid;
            candidate.provenanceProven = provenance.valid;
            if (!provenance.valid)
              candidate.provenanceReason = provenance.reason.str();
          }
          identityCandidates.push_back(std::move(candidate));
        }
        recordAllocation(operation, accounting);
      }
    });

    if (Attribute declaredAttr = module->getAttr(kWeightResidentBytesAttr)) {
      auto declared = dyn_cast<IntegerAttr>(declaredAttr);
      if (!declared || !declared.getType().isSignlessInteger(64) ||
          declared.getValue().isNegative()) {
        accounting.complete = false;
      } else {
        accounting.hasDeclaredWeightBytes = true;
        accounting.declaredWeightResidentBytes = declared.getInt();
        if (accounting.declaredWeightResidentBytes !=
            accounting.weightResidentBytes)
          accounting.complete = false;
      }
    }
    if (accounting.weightResidentBytes != 0 &&
        !accounting.hasDeclaredWeightBytes)
      accounting.complete = false;

    int64_t residentBytes = 0;
    if (!addBytes(residentBytes, accounting.workspaceResidentBytes) ||
        !addBytes(residentBytes, accounting.weightResidentBytes))
      accounting.complete = false;
    // `resident_site_sum_bytes` is a discovered allocation-site fact.  The
    // module aggregate is only a cross-check: never turn a declaration delta
    // into a resident site.
    int64_t rawSiteSum = accounting.transientBytes;
    if (!addBytes(rawSiteSum, residentBytes))
      accounting.complete = false;

    StaticIdentityResult identity;
    // The census marker owns the additive sidecar. The separate identity
    // marker only changes whether incomplete identity facts fail the pass.
    identity = buildStaticIdentity(module, identityCandidates);

    // Canonical static site identity is resolved before liveness so peak
    // attribution can reference site identity rather than a census walk
    // ordinal.  A site without a provable canonical identity is simply absent
    // from this map; liveness then withholds the ID list instead of inventing
    // one.  The two sidecars stay independent: neither is a manifest field.
    DenseMap<Operation *, uint64_t> canonicalSites;
    for (const StaticSiteIdentity &site : identity.sites)
      if (site.siteIdValid && site.site && site.site->operation)
        canonicalSites[site.site->operation] = site.siteId;

    LivenessAccounting liveness;
    if (livenessMarker)
      liveness = analyzeModuleLiveness(module, accounting.sites,
                                        accounting.complete, canonicalSites);

    MLIRContext *context = module.getContext();
    auto i64 = IntegerType::get(context, 64);
    NamedAttrList fields;
    fields.append(kKeyKind,
                  StringAttr::get(context, kKindAllocationSiteCensus));
    fields.append(kKeyStatus,
                  StringAttr::get(context, accounting.complete
                                               ? kStatusComplete
                                               : kStatusIncomplete));
    fields.append(kKeyAllocationSites,
                  IntegerAttr::get(i64, accounting.allocationSites));
    fields.append(kKeyUnknownAllocations,
                  IntegerAttr::get(i64, accounting.unknownAllocations));
    fields.append(kKeyTransientBytes,
                  IntegerAttr::get(i64, accounting.transientBytes));
    fields.append(kKeyWorkspaceResidentBytes,
                  IntegerAttr::get(i64, accounting.workspaceResidentBytes));
    fields.append(kKeyWeightResidentBytes,
                  IntegerAttr::get(i64, accounting.weightResidentBytes));
    fields.append(kKeyResidentSiteSumBytes,
                  IntegerAttr::get(i64, residentBytes));
    fields.append(kKeyRawSiteSumBytes, IntegerAttr::get(i64, rawSiteSum));
    fields.append(kKeyPeakStatus,
                  StringAttr::get(context, kPeakStatusNotProven));
    fields.append(kKeyExternalScratch,
                  StringAttr::get(context, accounting.externalScratch
                                               ? "unsupported"
                                               : "none"));
    fields.append(kKeyExternalVtcm,
                  StringAttr::get(context, accounting.externalVtcm.empty()
                                               ? "none"
                                               : accounting.externalVtcm));
    module->setAttr(kHmxVtcmAccountingAttr,
                    DictionaryAttr::get(context, fields));
    if (livenessMarker)
      module->setAttr(kHmxVtcmLivenessAttr,
                      buildLivenessAttr(module, liveness));
    module->setAttr(kHmxVtcmIdentityAttr,
                    buildStaticIdentityAttr(module, identity));
    if (evidenceContextMarker) {
      module->setAttr(kHmxVtcmEvidenceContextAttr,
                      buildEvidenceContextAttr(module, identity, accounting));
      module->setAttr(kHmxVtcmEventContextAttr,
                      buildEventContextAttr(module, identity, accounting));
    }

    // Preserve the diagnostic priority of the former two branches with one
    // accounting failure path: an ordinary census failure wins, while an
    // external-VTCM fact keeps the more specific structured-liveness reason.
    bool reportLivenessFirst =
        livenessMarker && !liveness.complete &&
        (!accounting.complete ? !accounting.externalVtcm.empty() : true);
    if (reportLivenessFirst) {
      module.emitError()
          << "HMX VTCM structured liveness is not proven for this IR: "
          << (liveness.reason.empty() ? "unsupported-analysis-shape"
                                      : liveness.reason)
          << (liveness.coverageComplete
                  ? ""
                  : "; allocation-site coverage incomplete");
      return signalPassFailure();
    }
    if (!accounting.complete) {
      module.emitError()
          << "HMX VTCM accounting is incomplete: static allocation bytes or "
             "resident-byte provenance could not be proven";
      return signalPassFailure();
    }
    if (identityMarker && identity.status != StaticIdentityStatus::kComplete) {
      std::string reason;
      for (const std::string &part : identity.reasons) {
        if (!reason.empty())
          reason += "; ";
        reason += part;
      }
      if (reason.empty())
        reason = "missing or ambiguous static facts";
      module.emitError() << "HMX VTCM static identity is not proven; "
                         << reason;
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hmx::createHmxVtcmAccountingPass() {
  return std::make_unique<HmxVtcmAccountingPass>();
}
