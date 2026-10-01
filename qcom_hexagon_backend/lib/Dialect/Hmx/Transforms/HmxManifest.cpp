//===-- HmxManifest.cpp - structured HMX manifest -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxTarget.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::hmx;
namespace json = llvm::json;

std::mutex &mlir::hmx::hmxModuleStateMutex() {
  static std::mutex mutex;
  return mutex;
}

namespace {

constexpr StringLiteral kManifestAttr = "hmx.kernel_manifest";
// This is the only versioned name in this file.  It is a wire protocol
// identity, not a C++ API name.
constexpr StringLiteral kManifestSchema = "hex.hmx.kernel_manifest/v2";

constexpr StringLiteral kKeySchema = "schema";
constexpr StringLiteral kKeyMatmuls = "matmuls";
constexpr StringLiteral kKeyFunction = "function";
constexpr StringLiteral kKeyId = "id";
constexpr StringLiteral kKeyPlan = "plan";
constexpr StringLiteral kKeyReason = "reason";
constexpr StringLiteral kKeyShapeState = "shape_state";
constexpr StringLiteral kKeyLogical = "logical";
constexpr StringLiteral kKeyDtypes = "dtypes";
constexpr StringLiteral kKeyLhsElem = "lhs";
constexpr StringLiteral kKeyRhsElem = "rhs";
constexpr StringLiteral kKeyOutElem = "out";
constexpr StringLiteral kKeyCrouton = "crouton";
constexpr StringLiteral kKeyM = "m";
constexpr StringLiteral kKeyN = "n";
constexpr StringLiteral kKeyK = "k";
constexpr StringLiteral kKeySlot = "slot";

constexpr StringLiteral kKeyPadded = "padded";
constexpr StringLiteral kKeyFull = "full";
constexpr StringLiteral kKeyTail = "tail";
constexpr StringLiteral kKeyTailPolicy = "tail_policy";
constexpr StringLiteral kKeyKPolicy = "k";
constexpr StringLiteral kKeyMnPolicy = "mn";
constexpr StringLiteral kKeyLayout = "layout";

constexpr StringLiteral kKeyWorkspaceClass = "workspace_class";
constexpr StringLiteral kKeyGridPolicy = "grid_policy";
constexpr StringLiteral kKeyVtcmAccounting = "vtcm_accounting";
constexpr StringLiteral kKeyVtcmBudgetBytes = "vtcm_budget_bytes";
constexpr StringLiteral kKeyVtcmBeforeBytes = "vtcm_before_bytes";
constexpr StringLiteral kKeyVtcmBridgePeakBytes = "vtcm_bridge_peak_bytes";

constexpr StringLiteral kKeyExecution = "execution";
constexpr StringLiteral kKeyBlocking = "blocking";
constexpr StringLiteral kKeyBlockM = "block_m";
constexpr StringLiteral kKeyPipeline = "pipeline";
constexpr StringLiteral kKeyPipelineRequested = "requested";
constexpr StringLiteral kKeyPipelineSelected = "selected";
constexpr StringLiteral kKeyPipelineDepth = "depth";
constexpr StringLiteral kKeyPipelineReason = "reason";
constexpr StringLiteral kKeyBridgeCounts = "bridge_counts";
constexpr StringLiteral kKeyPackActSites = "pack_act_sites";
constexpr StringLiteral kKeyPackWeightSites = "pack_weight_sites";
constexpr StringLiteral kKeyUnpackSites = "unpack_sites";
constexpr StringLiteral kKeyCountSemantics = "count_semantics";
constexpr StringLiteral kCountSemantics = "ir_sites";

constexpr StringLiteral kKeyWeightBinding = "weight_binding";
constexpr StringLiteral kKeyKind = "kind";
constexpr StringLiteral kKeyPolicyRef = "policy_ref";
constexpr StringLiteral kKeyPolicy = "policy";
constexpr StringLiteral kKeyConsumers = "consumers";
constexpr StringLiteral kKeyWeightPolicies = "weight_policies";
constexpr StringLiteral kKeyPlanFingerprint = "plan_fingerprint";
constexpr StringLiteral kFingerprintPrefix = "sha256:";

constexpr StringLiteral kPlanFullHMX = kHmxPlanFullHMX;
constexpr StringLiteral kPlanHMXTail = kHmxPlanHMXTail;
constexpr StringLiteral kPlanHVX = kHmxPlanHVX;

constexpr StringLiteral kWorkspaceRuntimeInternal = kHmxWorkspaceRuntimeInternal;
constexpr StringLiteral kWorkspaceResidentSingleInstance =
    kHmxWorkspaceResidentSingleInstance;
constexpr StringLiteral kGridSingleInstance = kHmxGridSingleInstance;
constexpr StringLiteral kGridLegacyRuntime = kHmxGridLegacyRuntime;
constexpr StringLiteral kBridgeOnlyAccounting = kHmxVtcmAccountingBridgeOnly;
constexpr StringLiteral kRowMajorInnerContiguous =
    kHmxLayoutRowMajorInnerContiguous;
constexpr StringLiteral kTailKPolicy = kHmxTailKPolicy;
constexpr StringLiteral kTailMNPolicy = kHmxTailMNPolicy;
// The minimum-rows threshold is NOT repeated here. It used to be, as a private
// `kMinimumHmxRows = 4`. That was one more copy of a number whose other homes
// are HmxTarget::minRows and the hand-referenced llama.cpp spelling -- copies
// with no comment saying which was authoritative, which is how a value drifts
// without anyone noticing. The manifest validates the same
// predicate the attribution applies, so it reads the same constant.

constexpr StringLiteral kWeightResidentPrepack = kHmxWeightResidentPrepack;
constexpr StringLiteral kWeightDevicePack = kHmxWeightDevicePack;
constexpr StringLiteral kWeightEligibleAlignedF16 = kHmxWeightEligibleAlignedF16;
constexpr StringLiteral kWeightEligibleQuantizedF32 =
    kHmxWeightEligibleQuantizedF32;
constexpr StringLiteral kWeightEligibleB2NSlice = kHmxWeightEligibleB2NSlice;
constexpr StringLiteral kWeightTailConsumer = kHmxWeightTailConsumer;
constexpr StringLiteral kWeightF32Source = kHmxWeightF32Source;
constexpr StringLiteral kWeightUnprovenOffset = kHmxWeightUnprovenOffset;
constexpr StringLiteral kWeightIncompatibleConsumers =
    kHmxWeightIncompatibleConsumers;
constexpr StringLiteral kWeightPrepackDisabled = kHmxWeightPrepackDisabled;

bool isCanonicalPipelineReason(StringRef reason) {
  static constexpr PipelineReason kReasons[] = {
      PipelineReason::SerialRequested,       PipelineReason::NoRowMajorBridge,
      PipelineReason::ExtraActivationReader, PipelineReason::InvalidStagingGeometry,
      PipelineReason::EmptyStagingGrid,      PipelineReason::StagingGridMismatch,
      PipelineReason::ShallowK,              PipelineReason::VtcmBudget,
      PipelineReason::TileCount,             PipelineReason::PipelinerFailed,
      PipelineReason::TailPeeledEdge,
  };
  for (PipelineReason candidate : kReasons)
    if (reason == pipelineReasonCode(candidate))
      return true;
  return false;
}

bool isCanonicalWorkspaceClass(StringRef value) {
  return value == kWorkspaceRuntimeInternal ||
         value == kWorkspaceResidentSingleInstance;
}

bool isCanonicalGridPolicy(StringRef value) {
  return value == kGridSingleInstance || value == kGridLegacyRuntime;
}

bool isCanonicalWeightReason(StringRef policy, StringRef reason) {
  if (policy == kWeightResidentPrepack)
    return reason == kWeightEligibleAlignedF16 ||
           reason == kWeightEligibleQuantizedF32 ||
           reason == kWeightEligibleB2NSlice;
  if (policy == kWeightDevicePack)
    return reason == kWeightTailConsumer || reason == kWeightF32Source ||
           reason == kWeightUnprovenOffset ||
           reason == kWeightIncompatibleConsumers ||
           reason == kWeightPrepackDisabled;
  return false;
}

bool isI64(Attribute attr) {
  auto integer = dyn_cast_or_null<IntegerAttr>(attr);
  return integer && integer.getType().isSignlessInteger(64);
}

bool hasI64(DictionaryAttr dict, StringRef name) {
  return dict && isI64(dict.get(name));
}

StringAttr stringField(DictionaryAttr dict, StringRef name) {
  return dict ? dyn_cast_or_null<StringAttr>(dict.get(name)) : StringAttr();
}

DictionaryAttr dictionaryField(DictionaryAttr dict, StringRef name) {
  return dict ? dyn_cast_or_null<DictionaryAttr>(dict.get(name))
              : DictionaryAttr();
}

ArrayAttr arrayField(DictionaryAttr dict, StringRef name) {
  return dict ? dyn_cast_or_null<ArrayAttr>(dict.get(name)) : ArrayAttr();
}

bool isNone(Attribute attr) { return attr && isa<UnitAttr>(attr); }

bool isNonEmptyString(StringRef value) { return !value.trim().empty(); }

LogicalResult emitManifestError(ModuleOp module, bool report, Twine message) {
  if (report)
    module.emitError() << message;
  return failure();
}

enum class PlanKind { FullHMX, HMXTail, HVX };
enum class ShapeState { Static, PartiallyDynamic, Dynamic, Unavailable };

struct LogicalInfo {
  bool present = false;
  ShapeState state = ShapeState::Unavailable;
  std::optional<std::array<int64_t, 3>> staticShape;
};

struct BridgeCounts {
  int64_t packAct = 0;
  int64_t packWeight = 0;
  int64_t unpack = 0;
};

struct WeightBindingRef {
  StringRef function;
  int64_t slot = 0;
};

struct RecordSummary {
  PlanKind plan = PlanKind::HVX;
  std::optional<BridgeCounts> bridgeCounts;
  std::optional<WeightBindingRef> weightBinding;
};

std::optional<PlanKind> parsePlan(StringRef value) {
  if (value == kPlanFullHMX)
    return PlanKind::FullHMX;
  if (value == kPlanHMXTail)
    return PlanKind::HMXTail;
  if (value == kPlanHVX)
    return PlanKind::HVX;
  return std::nullopt;
}

std::optional<ShapeState> parseShapeState(StringRef value) {
  if (value == kHmxShapeStateStatic)
    return ShapeState::Static;
  if (value == kHmxShapeStatePartiallyDynamic)
    return ShapeState::PartiallyDynamic;
  if (value == kHmxShapeStateDynamic)
    return ShapeState::Dynamic;
  if (value == kHmxShapeStateUnavailable)
    return ShapeState::Unavailable;
  return std::nullopt;
}

bool isHmxPlan(PlanKind plan) { return plan != PlanKind::HVX; }

bool isFingerprintText(StringRef value) {
  if (!value.consume_front(kFingerprintPrefix) || value.size() != 64)
    return false;
  for (char c : value) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return false;
  }
  return true;
}

/// Reject the fields from the superseded flat engine contract explicitly.  A
/// semantic record must not be accepted merely because it also happens to
/// contain a complete new record.
LogicalResult rejectUnknownFields(ModuleOp module, DictionaryAttr dict,
                                  ArrayRef<StringRef> allowed, bool report,
                                  StringRef what) {
  for (NamedAttribute field : dict) {
    if (!llvm::is_contained(allowed, field.getName().strref()))
      return emitManifestError(module, report,
                               Twine("HMX manifest ") + what +
                                   " has unknown field '" +
                                   field.getName().strref() + "'");
  }
  return success();
}

LogicalResult rejectLegacyRecordFields(ModuleOp module, DictionaryAttr record,
                                       bool report) {
  static constexpr StringLiteral fields[] = {
      "engine",
      "m",
      "n",
      "k",
      "lhs_elem",
      "rhs_elem",
      "out_elem",
      "vtcm_budget",
      "vtcm_before",
      "vtcm_peak",
      "blocking",
      "block_m",
      "pipeline_requested",
      "pipeline_selected",
      "pipeline_depth",
      "pipeline_reason",
      "pack_act_sites",
      "pack_weight_sites",
      "unpack_sites",
      "count_semantics",
      "workspace",
      "workspace_bytes",
      "workspace_alignment",
      "vtcm",
      "static-source",
      "upstream-specialized",
  };
  for (StringRef field : fields) {
    if (record.get(field))
      return emitManifestError(
          module, report,
          Twine("HMX manifest record contains legacy field '") + field + "'");
  }
  return success();
}

LogicalResult validateDtypes(ModuleOp module, DictionaryAttr record,
                             PlanKind plan, bool report, bool requireFields) {
  Attribute raw = record.get(kKeyDtypes);
  if (!raw) {
    if (requireFields)
      return emitManifestError(module, report,
                               "HMX manifest record has no dtypes");
    return success();
  }
  auto dtypes = dyn_cast<DictionaryAttr>(raw);
  if (!dtypes)
    return emitManifestError(module, report,
                             "HMX manifest record dtypes must be an object");
  SmallVector<StringRef, 4> allowedDtypes = {kKeyLhsElem, kKeyRhsElem,
                                             kKeyOutElem};
  if (isHmxPlan(plan))
    allowedDtypes.push_back(kKeyCrouton);
  if (failed(
          rejectUnknownFields(module, dtypes, allowedDtypes, report, "dtypes")))
    return failure();
  for (StringRef field : {kKeyLhsElem, kKeyRhsElem, kKeyOutElem}) {
    StringAttr value = stringField(dtypes, field);
    if (!value || !isNonEmptyString(value.getValue()))
      return emitManifestError(
          module, report,
          Twine("HMX manifest record has no valid dtypes.") + field.str());
  }
  if (isHmxPlan(plan)) {
    for (StringRef field : {kKeyLhsElem, kKeyRhsElem, kKeyOutElem}) {
      StringRef value = stringField(dtypes, field).getValue();
      if (value != kHmxDTypeF16 && value != kHmxDTypeF32)
        return emitManifestError(
            module, report,
            Twine("HMX manifest dtype must be f16 or f32: ") + field.str());
    }
    StringAttr crouton = stringField(dtypes, kKeyCrouton);
    if (!crouton || crouton.getValue() != kHmxDTypeF16)
      return emitManifestError(module, report,
                               "HMX manifest crouton dtype must be f16");
  } else if (dtypes.get(kKeyCrouton)) {
    return emitManifestError(
        module, report, "HVX manifest record must not carry a crouton dtype");
  }
  return success();
}

LogicalResult parseLogicalShape(ModuleOp module, DictionaryAttr record,
                                bool report, bool requireFields,
                                LogicalInfo &result) {
  Attribute logical = record.get(kKeyLogical);
  StringAttr stateAttr = stringField(record, kKeyShapeState);
  if (!logical && !stateAttr) {
    if (requireFields)
      return emitManifestError(module, report,
                               "HMX manifest record has no logical shape");
    return success();
  }

  if (logical && isNone(logical)) {
    if (stateAttr && stateAttr.getValue() != kHmxShapeStateUnavailable)
      return emitManifestError(module, report,
                               "HMX manifest unavailable logical shape "
                               "requires shape_state=unavailable");
    if (requireFields && !stateAttr)
      return emitManifestError(module, report,
                               "HMX manifest record has no shape_state");
    result =
        LogicalInfo{/*present=*/true, ShapeState::Unavailable, std::nullopt};
    return success();
  }

  if (!logical) {
    // A later pass may add the tagged shape after the initial attribution.  Do
    // not manufacture one here, but do reject a present, contradictory state
    // in the final phase below.
    if (stateAttr && !parseShapeState(stateAttr.getValue()))
      return emitManifestError(module, report,
                               "HMX manifest has an invalid shape_state");
    if (requireFields)
      return emitManifestError(module, report,
                               "HMX manifest record has no logical shape");
    result =
        LogicalInfo{/*present=*/false, ShapeState::Unavailable, std::nullopt};
    return success();
  }

  auto dimensions = dyn_cast<DictionaryAttr>(logical);
  if (!dimensions)
    return emitManifestError(
        module, report, "HMX manifest logical shape must be an object or null");
  if (failed(rejectUnknownFields(
          module, dimensions,
          {StringRef(kKeyM), StringRef(kKeyN), StringRef(kKeyK)}, report,
          "logical shape")))
    return failure();
  if (requireFields && !stateAttr)
    return emitManifestError(module, report,
                             "HMX manifest record has no shape_state");

  SmallVector<int64_t, 3> staticValues;
  unsigned staticCount = 0;
  unsigned dynamicCount = 0;
  for (StringRef axis : {kKeyM, kKeyN, kKeyK}) {
    auto dimension = dyn_cast_or_null<DictionaryAttr>(dimensions.get(axis));
    if (!dimension)
      return emitManifestError(
          module, report,
          Twine("HMX manifest logical shape has no dimension '") + axis + "'");
    StringAttr kind = stringField(dimension, "kind");
    if (!kind)
      return emitManifestError(module, report,
                               "HMX manifest dimension has no kind");
    if (kind.getValue() == kHmxShapeStateStatic) {
      if (failed(rejectUnknownFields(module, dimension,
                                     {StringRef("kind"), StringRef("value")},
                                     report, "dimension")))
        return failure();
      if (!hasI64(dimension, "value") || dimension.get("symbol"))
        return emitManifestError(module, report,
                                 "HMX manifest static dimension is malformed");
      int64_t value = cast<IntegerAttr>(dimension.get("value")).getInt();
      if (value <= 0)
        return emitManifestError(
            module, report, "HMX manifest static dimension must be positive");
      staticValues.push_back(value);
      ++staticCount;
    } else if (kind.getValue() == kHmxShapeStateDynamic) {
      if (failed(rejectUnknownFields(module, dimension,
                                     {StringRef("kind"), StringRef("symbol")},
                                     report, "dimension")))
        return failure();
      StringAttr symbol = stringField(dimension, "symbol");
      if (!symbol || symbol.getValue() != axis || dimension.get("value"))
        return emitManifestError(module, report,
                                 "HMX manifest dynamic dimension is malformed");
      ++dynamicCount;
    } else {
      return emitManifestError(
          module, report,
          "HMX manifest dimension kind must be static or dynamic");
    }
  }

  ShapeState derived = staticCount == 3
                           ? ShapeState::Static
                           : (dynamicCount == 3 ? ShapeState::Dynamic
                                                : ShapeState::PartiallyDynamic);
  if (stateAttr) {
    auto declared = parseShapeState(stateAttr.getValue());
    if (!declared)
      return emitManifestError(module, report,
                               "HMX manifest has an invalid shape_state");
    if (*declared != derived)
      return emitManifestError(
          module, report,
          "HMX manifest shape_state disagrees with tagged dimensions");
    result.state = *declared;
  } else {
    result.state = derived;
  }
  result.present = true;
  if (result.state == ShapeState::Static)
    result.staticShape = std::array<int64_t, 3>{
        staticValues[0], staticValues[1], staticValues[2]};
  return success();
}

LogicalResult readShapeVector(ModuleOp module, DictionaryAttr record,
                              StringRef field, bool allowZero,
                              std::array<int64_t, 3> &values, bool report) {
  auto shape = dictionaryField(record, field);
  if (!shape)
    return emitManifestError(module, report,
                             Twine("HMX manifest record has no ") + field +
                                 " shape object");
  if (failed(rejectUnknownFields(
          module, shape, {StringRef(kKeyM), StringRef(kKeyN), StringRef(kKeyK)},
          report, "physical shape")))
    return failure();
  for (StringRef axis : {kKeyM, kKeyN, kKeyK}) {
    if (!hasI64(shape, axis))
      return emitManifestError(module, report,
                               Twine("HMX manifest ") + field +
                                   " shape has no integer " + axis);
    int64_t value = cast<IntegerAttr>(shape.get(axis)).getInt();
    if (value < (allowZero ? 0 : 1))
      return emitManifestError(module, report,
                               Twine("HMX manifest ") + field +
                                   " shape has an invalid extent");
    values[static_cast<size_t>(axis == kKeyM   ? 0
                               : axis == kKeyN ? 1
                                               : 2)] = value;
  }
  return success();
}

LogicalResult validatePhysicalShape(ModuleOp module, DictionaryAttr record,
                                    PlanKind plan, const LogicalInfo &logical,
                                    bool report, bool requireFields) {
  bool hasPadded = record.get(kKeyPadded) != nullptr;
  bool hasFull = record.get(kKeyFull) != nullptr;
  bool hasTail = record.get(kKeyTail) != nullptr;
  if (!hasPadded && !hasFull && !hasTail) {
    if (requireFields)
      return emitManifestError(module, report,
                               "HMX manifest record has no physical shape");
    return success();
  }
  if (!hasPadded || !hasFull || !hasTail)
    return emitManifestError(
        module, report, "HMX manifest has a partial padded/full/tail shape");

  std::array<int64_t, 3> padded{}, full{}, tail{};
  if (failed(
          readShapeVector(module, record, kKeyPadded, false, padded, report)) ||
      failed(readShapeVector(module, record, kKeyFull, true, full, report)) ||
      failed(readShapeVector(module, record, kKeyTail, true, tail, report)))
    return failure();

  if (logical.state != ShapeState::Static) {
    if (requireFields)
      return emitManifestError(module, report,
                               "HMX manifest requires a static logical shape");
    return success();
  }
  const auto &values = *logical.staticShape;
  for (size_t i = 0; i < 3; ++i) {
    if (values[i] > std::numeric_limits<int64_t>::max() - 31)
      return emitManifestError(module, report,
                               "HMX manifest logical shape overflows padding");
    int64_t expectedFull = (values[i] / 32) * 32;
    int64_t expectedTail = values[i] - expectedFull;
    int64_t expectedPadded = expectedFull + (expectedTail ? 32 : 0);
    if (full[i] % 32 != 0 || padded[i] % 32 != 0 || tail[i] > 31 ||
        full[i] != expectedFull || tail[i] != expectedTail ||
        padded[i] != expectedPadded)
      return emitManifestError(
          module, report,
          "HMX manifest physical shape disagrees with tagged logical shape");
  }
  if (plan == PlanKind::FullHMX) {
    for (int64_t value : tail)
      if (value != 0)
        return emitManifestError(module, report,
                                 "full-hmx record has a nonzero tail");
    if (record.get(kKeyTailPolicy))
      return emitManifestError(module, report,
                               "full-hmx record must not carry tail_policy");
  } else {
    bool hasNonzeroTail = false;
    for (int64_t value : tail)
      hasNonzeroTail |= value != 0;
    if (!hasNonzeroTail)
      return emitManifestError(module, report,
                               "hmx-tail record has no tail extent");
  }
  return success();
}

LogicalResult validateTailPolicy(ModuleOp module, DictionaryAttr record,
                                 PlanKind plan, bool report,
                                 bool requireFields) {
  Attribute raw = record.get(kKeyTailPolicy);
  if (!raw) {
    if (plan == PlanKind::HMXTail && requireFields)
      return emitManifestError(module, report,
                               "hmx-tail record has no tail_policy");
    return success();
  }
  if (plan != PlanKind::HMXTail)
    return emitManifestError(module, report,
                             "only hmx-tail records may carry tail_policy");
  auto policy = dyn_cast<DictionaryAttr>(raw);
  if (!policy)
    return emitManifestError(module, report,
                             "HMX manifest tail_policy must be an object");
  if (failed(rejectUnknownFields(
          module, policy, {StringRef(kKeyKPolicy), StringRef(kKeyMnPolicy)},
          report, "tail policy")))
    return failure();
  StringAttr k = stringField(policy, kKeyKPolicy);
  StringAttr mn = stringField(policy, kKeyMnPolicy);
  if (!k || !mn || k.getValue() != kTailKPolicy ||
      mn.getValue() != kTailMNPolicy)
    return emitManifestError(module, report,
                             "HMX manifest has an invalid tail policy");
  return success();
}

LogicalResult validatePipeline(ModuleOp module, DictionaryAttr pipeline,
                               bool report) {
  if (failed(rejectUnknownFields(
          module, pipeline,
          {StringRef(kKeyPipelineRequested), StringRef(kKeyPipelineSelected),
           StringRef(kKeyPipelineDepth), StringRef(kKeyPipelineReason)},
          report, "pipeline")))
    return failure();
  if (!hasI64(pipeline, kKeyPipelineRequested) ||
      !hasI64(pipeline, kKeyPipelineDepth) ||
      !stringField(pipeline, kKeyPipelineSelected))
    return emitManifestError(module, report,
                             "HMX manifest has a partial pipeline decision");
  StringAttr selected = stringField(pipeline, kKeyPipelineSelected);
  if (selected.getValue() != kHmxPipelineSerial &&
      selected.getValue() != kHmxPipelineStaged)
    return emitManifestError(module, report,
                             "HMX manifest has an invalid pipeline selection");
  int64_t requested =
      cast<IntegerAttr>(pipeline.get(kKeyPipelineRequested)).getInt();
  int64_t depth = cast<IntegerAttr>(pipeline.get(kKeyPipelineDepth)).getInt();
  if (requested < 0 || depth < 0 ||
      (selected.getValue() == kHmxPipelineSerial && depth != 0) ||
      (selected.getValue() == kHmxPipelineStaged && depth == 0))
    return emitManifestError(module, report,
                             "HMX manifest has an invalid pipeline depth");
  if (pipeline.get(kKeyPipelineReason) &&
      (!stringField(pipeline, kKeyPipelineReason) ||
       !isCanonicalPipelineReason(
           stringField(pipeline, kKeyPipelineReason).getValue())))
    return emitManifestError(module, report,
                             "HMX manifest has an invalid pipeline reason");
  return success();
}

LogicalResult validateBridgeCounts(ModuleOp module, DictionaryAttr bridge,
                                   bool report, BridgeCounts &counts) {
  if (failed(rejectUnknownFields(
          module, bridge,
          {StringRef(kKeyPackActSites), StringRef(kKeyPackWeightSites),
           StringRef(kKeyUnpackSites), StringRef(kKeyCountSemantics)},
          report, "bridge counts")))
    return failure();
  if (!hasI64(bridge, kKeyPackActSites) ||
      !hasI64(bridge, kKeyPackWeightSites) ||
      !hasI64(bridge, kKeyUnpackSites) ||
      !stringField(bridge, kKeyCountSemantics))
    return emitManifestError(module, report,
                             "HMX manifest has incomplete bridge counts");
  if (stringField(bridge, kKeyCountSemantics).getValue() != kCountSemantics)
    return emitManifestError(module, report,
                             "HMX manifest has invalid count semantics");
  counts.packAct = cast<IntegerAttr>(bridge.get(kKeyPackActSites)).getInt();
  counts.packWeight =
      cast<IntegerAttr>(bridge.get(kKeyPackWeightSites)).getInt();
  counts.unpack = cast<IntegerAttr>(bridge.get(kKeyUnpackSites)).getInt();
  if (counts.packAct < 0 || counts.packWeight < 0 || counts.unpack < 0)
    return emitManifestError(module, report,
                             "HMX manifest has a negative bridge count");
  return success();
}

LogicalResult validateExecution(ModuleOp module, DictionaryAttr record,
                                const LogicalInfo &logical, bool report,
                                bool requireFields,
                                std::optional<BridgeCounts> &counts) {
  Attribute raw = record.get(kKeyExecution);
  if (!raw) {
    if (requireFields)
      return emitManifestError(module, report,
                               "HMX manifest record has no execution facts");
    return success();
  }
  auto execution = dyn_cast<DictionaryAttr>(raw);
  if (!execution)
    return emitManifestError(module, report,
                             "HMX manifest execution must be an object");
  if (failed(rejectUnknownFields(
          module, execution,
          {StringRef(kKeyBlocking), StringRef(kKeyBlockM),
           StringRef(kKeyPipeline), StringRef(kKeyBridgeCounts)},
          report, "execution")))
    return failure();

  bool hasBlocking = execution.get(kKeyBlocking) != nullptr;
  bool hasBlockM = execution.get(kKeyBlockM) != nullptr;
  if (hasBlocking != hasBlockM)
    return emitManifestError(module, report,
                             "HMX manifest has a partial blocking decision");
  if (hasBlocking) {
    StringAttr blocking = stringField(execution, kKeyBlocking);
    if (!blocking || (blocking.getValue() != kHmxBlockingWhole &&
                      blocking.getValue() != kHmxBlockingMBlocked))
      return emitManifestError(module, report,
                               "HMX manifest has invalid execution blocking");
    if (!hasI64(execution, kKeyBlockM))
      return emitManifestError(module, report,
                               "HMX manifest has no execution block_m");
    int64_t blockM = cast<IntegerAttr>(execution.get(kKeyBlockM)).getInt();
    if (blockM <= 0)
      return emitManifestError(
          module, report, "HMX manifest execution block_m must be positive");
    if (logical.state == ShapeState::Static && logical.staticShape) {
      int64_t m = (*logical.staticShape)[0];
      if ((blocking.getValue() == kHmxBlockingWhole && blockM != m) ||
          (blocking.getValue() == kHmxBlockingMBlocked &&
           (blockM >= m || m % blockM != 0)))
        return emitManifestError(
            module, report,
            "HMX manifest execution block_m is inconsistent with logical M");
    }
  }

  if (execution.get(kKeyPipeline)) {
    auto pipeline = dyn_cast<DictionaryAttr>(execution.get(kKeyPipeline));
    if (!pipeline || failed(validatePipeline(module, pipeline, report)))
      return failure();
  }
  if (execution.get(kKeyBridgeCounts)) {
    auto bridge = dyn_cast<DictionaryAttr>(execution.get(kKeyBridgeCounts));
    BridgeCounts parsed;
    if (!bridge || failed(validateBridgeCounts(module, bridge, report, parsed)))
      return failure();
    counts = parsed;
  }

  if (requireFields && (!hasBlocking || !execution.get(kKeyPipeline) ||
                        !execution.get(kKeyBridgeCounts)))
    return emitManifestError(module, report,
                             "HMX manifest execution facts are incomplete");
  return success();
}

LogicalResult validateWorkspaceAndVtcm(ModuleOp module, DictionaryAttr record,
                                       PlanKind plan, bool report,
                                       bool requireFields) {
  bool hasWorkspace = record.get(kKeyWorkspaceClass) != nullptr;
  bool hasGrid = record.get(kKeyGridPolicy) != nullptr;
  bool hasAccounting = record.get(kKeyVtcmAccounting) != nullptr;
  if (hasWorkspace || hasGrid || hasAccounting) {
    if (hasWorkspace) {
      StringAttr value = stringField(record, kKeyWorkspaceClass);
      if (!value || !isCanonicalWorkspaceClass(value.getValue()))
        return emitManifestError(module, report,
                                 "HMX manifest has an invalid workspace_class");
    }
    if (hasGrid) {
      StringAttr value = stringField(record, kKeyGridPolicy);
      if (!value || !isCanonicalGridPolicy(value.getValue()))
        return emitManifestError(module, report,
                                 "HMX manifest has an invalid grid_policy");
    }
    if (hasAccounting && (!stringField(record, kKeyVtcmAccounting) ||
                          stringField(record, kKeyVtcmAccounting).getValue() !=
                              kBridgeOnlyAccounting))
      return emitManifestError(
          module, report, "HMX manifest vtcm_accounting must be bridge-only");
    if (plan == PlanKind::HMXTail && hasGrid &&
        stringField(record, kKeyGridPolicy).getValue() != kGridSingleInstance)
      return emitManifestError(module, report,
                               "hmx-tail requires single-instance grid policy");
    if (hasWorkspace && hasGrid) {
      StringRef workspace = stringField(record, kKeyWorkspaceClass).getValue();
      StringRef grid = stringField(record, kKeyGridPolicy).getValue();
      if (plan == PlanKind::FullHMX && workspace == kWorkspaceRuntimeInternal &&
          grid != kGridLegacyRuntime)
        return emitManifestError(
            module, report,
            "full-hmx runtime-internal workspace requires legacy-runtime grid");
      if (workspace == kWorkspaceResidentSingleInstance &&
          grid != kGridSingleInstance)
        return emitManifestError(
            module, report,
            "resident-single-instance workspace requires single-instance grid");
    }
  }

  bool hasBudget = record.get(kKeyVtcmBudgetBytes) != nullptr;
  bool hasBefore = record.get(kKeyVtcmBeforeBytes) != nullptr;
  bool hasPeak = record.get(kKeyVtcmBridgePeakBytes) != nullptr;
  if (hasBudget || hasBefore || hasPeak) {
    if (!hasBudget || !hasBefore || !hasPeak)
      return emitManifestError(
          module, report, "HMX manifest has partial bridge-only VTCM facts");
    if (!hasI64(record, kKeyVtcmBudgetBytes) ||
        !hasI64(record, kKeyVtcmBeforeBytes) ||
        !hasI64(record, kKeyVtcmBridgePeakBytes))
      return emitManifestError(
          module, report, "HMX manifest has invalid bridge-only VTCM facts");
    int64_t budget =
        cast<IntegerAttr>(record.get(kKeyVtcmBudgetBytes)).getInt();
    int64_t before =
        cast<IntegerAttr>(record.get(kKeyVtcmBeforeBytes)).getInt();
    int64_t peak =
        cast<IntegerAttr>(record.get(kKeyVtcmBridgePeakBytes)).getInt();
    if (budget < 0 || before < 0 || peak < before || peak > budget)
      return emitManifestError(module, report,
                               "HMX manifest has an invalid bridge VTCM peak");
  }
  if (requireFields && (!hasWorkspace || !hasGrid || !hasAccounting ||
                        !hasBudget || !hasBefore || !hasPeak))
    return emitManifestError(
        module, report, "HMX manifest workspace/VTCM facts are incomplete");
  return success();
}

LogicalResult validateWeightBinding(ModuleOp module, DictionaryAttr record,
                                    bool report, bool requireFields,
                                    std::optional<WeightBindingRef> &binding) {
  Attribute raw = record.get(kKeyWeightBinding);
  if (!raw) {
    if (requireFields)
      return emitManifestError(module, report,
                               "HMX manifest record has no weight_binding");
    return success();
  }
  auto value = dyn_cast<DictionaryAttr>(raw);
  if (!value)
    return emitManifestError(module, report,
                             "HMX manifest weight_binding must be an object");
  StringAttr kind = stringField(value, kKeyKind);
  if (!kind)
    return emitManifestError(module, report,
                             "HMX manifest weight_binding has no kind");
  if (kind.getValue() == kHmxWeightKindArgumentSlot) {
    if (failed(rejectUnknownFields(
            module, value, {StringRef(kKeyKind), StringRef(kKeyPolicyRef)},
            report, "weight binding")))
      return failure();
    auto ref = dictionaryField(value, kKeyPolicyRef);
    if (ref && failed(rejectUnknownFields(
                   module, ref, {StringRef(kKeyFunction), StringRef(kKeySlot)},
                   report, "weight policy reference")))
      return failure();
    StringAttr function = ref ? stringField(ref, kKeyFunction) : StringAttr();
    if (!ref || !function || !isNonEmptyString(function.getValue()) ||
        !hasI64(ref, kKeySlot))
      return emitManifestError(
          module, report, "HMX manifest argument weight binding is malformed");
    int64_t slot = cast<IntegerAttr>(ref.get(kKeySlot)).getInt();
    if (slot < 0)
      return emitManifestError(module, report,
                               "HMX manifest argument weight slot is negative");
    binding = WeightBindingRef{function.getValue(), slot};
  } else if (kind.getValue() == kHmxWeightKindCompileTimeConstant ||
             kind.getValue() == kHmxWeightKindInternalValue) {
    if (failed(rejectUnknownFields(module, value, {StringRef(kKeyKind)}, report,
                                   "weight binding")))
      return failure();
    if (value.get(kKeyPolicyRef))
      return emitManifestError(
          module, report,
          "HMX manifest non-argument weight binding has a policy_ref");
  } else {
    return emitManifestError(module, report,
                             "HMX manifest has an invalid weight_binding kind");
  }
  return success();
}

LogicalResult validateRecord(ModuleOp module, DictionaryAttr record,
                             bool report, bool requireFinal,
                             bool checkFingerprint, RecordSummary &summary) {
  if (failed(rejectLegacyRecordFields(module, record, report)))
    return failure();

  StringAttr function = stringField(record, kKeyFunction);
  if (!function || !isNonEmptyString(function.getValue()))
    return emitManifestError(module, report,
                             "HMX manifest record has no function name");
  auto id = dyn_cast_or_null<IntegerAttr>(record.get(kKeyId));
  if (!id || !id.getType().isSignlessInteger(64) || id.getInt() < 0)
    return emitManifestError(
        module, report, "HMX manifest record has no valid function-local id");

  StringAttr planAttr = stringField(record, kKeyPlan);
  if (!planAttr)
    return emitManifestError(module, report,
                             "HMX manifest record has no semantic plan");
  std::optional<PlanKind> plan = parsePlan(planAttr.getValue());
  if (!plan)
    return emitManifestError(
        module, report, "HMX manifest record has an invalid semantic plan");
  SmallVector<StringRef, 24> allowedFields = {
      kKeyFunction,   kKeyId,      kKeyPlan,   kKeyReason,
      kKeyShapeState, kKeyLogical, kKeyDtypes, kKeyPlanFingerprint};
  if (*plan != PlanKind::HVX) {
    for (StringRef field :
         {kKeyPadded, kKeyFull, kKeyTail, kKeyLayout, kKeyWorkspaceClass,
          kKeyGridPolicy, kKeyVtcmAccounting, kKeyVtcmBudgetBytes,
          kKeyVtcmBeforeBytes, kKeyVtcmBridgePeakBytes, kKeyExecution,
          kKeyWeightBinding})
      allowedFields.push_back(field);
    if (*plan == PlanKind::HMXTail)
      allowedFields.push_back(kKeyTailPolicy);
  }
  if (failed(
          rejectUnknownFields(module, record, allowedFields, report, "record")))
    return failure();
  StringAttr reason = stringField(record, kKeyReason);
  if (!reason || !isCanonicalHmxMatmulReason(reason.getValue()))
    return emitManifestError(module, report,
                             "HMX manifest record has no canonical reason");
  if ((*plan == PlanKind::FullHMX &&
       reason.getValue() != kHmxReasonSelectedAligned) ||
      (*plan == PlanKind::HMXTail &&
       reason.getValue() != kHmxReasonSelectedTail) ||
      (*plan == PlanKind::HVX &&
       (reason.getValue() == kHmxReasonSelectedAligned ||
        reason.getValue() == kHmxReasonSelectedTail)))
    return emitManifestError(
        module, report,
        "HMX manifest semantic plan and reason disagree on attribution");

  if (record.get(kKeyShapeState) && !stringField(record, kKeyShapeState))
    return emitManifestError(module, report,
                             "HMX manifest shape_state must be a string");
  LogicalInfo logical;
  if (failed(parseLogicalShape(module, record, report, requireFinal, logical)))
    return failure();
  if (requireFinal && logical.state == ShapeState::Unavailable &&
      *plan != PlanKind::HVX)
    return emitManifestError(
        module, report, "HMX manifest plan requires a tagged logical shape");
  if (logical.present && logical.state == ShapeState::Unavailable &&
      reason.getValue() != kHmxReasonLibraryCall &&
      reason.getValue() != kHmxReasonNonRank2 &&
      reason.getValue() != kHmxReasonUnsupportedLayout)
    return emitManifestError(
        module, report,
        "only library-call, non-rank-2 and unsupported-layout may have "
        "unavailable logical shape");
  if (logical.present && reason.getValue() == kHmxReasonDynamicShape &&
      logical.state == ShapeState::Static)
    return emitManifestError(
        module, report, "dynamic-shape reason requires a dynamic dimension");
  if (failed(validateDtypes(module, record, *plan, report, requireFinal)))
    return failure();

  if (isHmxPlan(*plan)) {
    if (logical.state != ShapeState::Unavailable &&
        logical.state != ShapeState::Static)
      return emitManifestError(
          module, report, "HMX plans require all-static logical dimensions");
    if (logical.staticShape &&
        (*logical.staticShape)[0] <= HmxTarget::minRows)
      return emitManifestError(module, report,
                               "HMX plans require logical M > " +
                                   llvm::Twine(HmxTarget::minRows));
    if (failed(validatePhysicalShape(module, record, *plan, logical, report,
                                     requireFinal)) ||
        failed(
            validateTailPolicy(module, record, *plan, report, requireFinal)) ||
        failed(validateWorkspaceAndVtcm(module, record, *plan, report,
                                        requireFinal)) ||
        failed(validateExecution(module, record, logical, report, requireFinal,
                                 summary.bridgeCounts)) ||
        failed(validateWeightBinding(module, record, report, requireFinal,
                                     summary.weightBinding)))
      return failure();
    if (record.get(kKeyLayout)) {
      StringAttr layout = stringField(record, kKeyLayout);
      if (!layout || layout.getValue() != kRowMajorInnerContiguous)
        return emitManifestError(module, report,
                                 "HMX manifest has an invalid layout");
    } else if (requireFinal) {
      return emitManifestError(module, report,
                               "HMX manifest record has no layout");
    }
  } else {
    static constexpr StringLiteral hmxOnlyFields[] = {
        kKeyPadded,
        kKeyFull,
        kKeyTail,
        kKeyTailPolicy,
        kKeyLayout,
        kKeyWorkspaceClass,
        kKeyGridPolicy,
        kKeyVtcmAccounting,
        kKeyVtcmBudgetBytes,
        kKeyVtcmBeforeBytes,
        kKeyVtcmBridgePeakBytes,
        kKeyExecution,
        kKeyWeightBinding,
        kKeyPlanFingerprint,
    };
    for (StringRef field : hmxOnlyFields) {
      // The fingerprint is a final record identity, not an HMX execution fact;
      // it is therefore intentionally allowed below for HVX records.
      if (field == kKeyPlanFingerprint)
        continue;
      if (record.get(field))
        return emitManifestError(
            module, report,
            Twine("HVX manifest record carries HMX-only field '") + field +
                "'");
    }
  }

  if (record.get(kKeyPlanFingerprint)) {
    StringAttr fingerprint = stringField(record, kKeyPlanFingerprint);
    if (!fingerprint || !isFingerprintText(fingerprint.getValue()))
      return emitManifestError(module, report,
                               "HMX manifest has an invalid plan_fingerprint");
  } else if (requireFinal && checkFingerprint) {
    return emitManifestError(module, report,
                             "HMX manifest record has no plan_fingerprint");
  }

  summary.plan = *plan;
  return success();
}

struct WeightPolicySummary {
  DictionaryAttr value;
  StringRef function;
  int64_t slot = 0;
  StringRef policy;
};

using WeightPolicyMap =
    std::map<std::pair<std::string, int64_t>, WeightPolicySummary>;

LogicalResult validateWeightPolicies(ModuleOp module, DictionaryAttr manifest,
                                     bool report, bool requireFields,
                                     WeightPolicyMap &policies) {
  Attribute raw = manifest.get(kKeyWeightPolicies);
  if (!raw) {
    if (requireFields)
      return emitManifestError(module, report,
                               "HMX manifest has no weight_policies");
    return success();
  }
  auto array = dyn_cast<ArrayAttr>(raw);
  if (!array)
    return emitManifestError(module, report,
                             "HMX manifest weight_policies must be an array");
  std::set<std::string> residentFunctions;
  for (Attribute item : array) {
    auto policyAttr = dyn_cast<DictionaryAttr>(item);
    if (!policyAttr)
      return emitManifestError(module, report,
                               "HMX manifest weight policy must be an object");
    if (failed(
            rejectUnknownFields(module, policyAttr,
                                {StringRef(kKeyFunction), StringRef(kKeySlot),
                                 StringRef(kKeyPolicy), StringRef(kKeyReason),
                                 StringRef(kKeyConsumers)},
                                report, "weight policy")))
      return failure();
    StringAttr function = stringField(policyAttr, kKeyFunction);
    if (!function || !isNonEmptyString(function.getValue()) ||
        !hasI64(policyAttr, kKeySlot))
      return emitManifestError(
          module, report, "HMX manifest weight policy has no function/slot");
    int64_t slot = cast<IntegerAttr>(policyAttr.get(kKeySlot)).getInt();
    if (slot < 0)
      return emitManifestError(module, report,
                               "HMX manifest weight policy slot is negative");
    StringAttr policy = stringField(policyAttr, kKeyPolicy);
    StringAttr reason = stringField(policyAttr, kKeyReason);
    if (!policy || !reason ||
        !isCanonicalWeightReason(policy.getValue(), reason.getValue()))
      return emitManifestError(
          module, report, "HMX manifest has an invalid weight policy/reason");
    auto consumers = arrayField(policyAttr, kKeyConsumers);
    if (!consumers)
      return emitManifestError(module, report,
                               "HMX manifest weight policy has no consumers");
    std::set<int64_t> consumerSet;
    for (Attribute consumer : consumers) {
      if (!isI64(consumer) || cast<IntegerAttr>(consumer).getInt() < 0 ||
          !consumerSet.insert(cast<IntegerAttr>(consumer).getInt()).second)
        return emitManifestError(
            module, report, "HMX manifest weight policy has invalid consumers");
    }
    auto key = std::make_pair(function.getValue().str(), slot);
    if (!policies
             .emplace(key, WeightPolicySummary{policyAttr, function.getValue(),
                                               slot, policy.getValue()})
             .second)
      return emitManifestError(
          module, report, "HMX manifest has duplicate weight policy slots");
    if (policy.getValue() == kWeightResidentPrepack)
      residentFunctions.insert(function.getValue().str());
  }
  if (residentFunctions.size() > 1)
    return emitManifestError(module, report,
                             "resident-prepack is limited to one function");
  return success();
}

std::optional<DictionaryAttr> findWeightPolicy(const WeightPolicyMap &policies,
                                               StringRef function,
                                               int64_t slot) {
  auto it = policies.find({function.str(), slot});
  if (it == policies.end())
    return std::nullopt;
  return it->second.value;
}

std::optional<BridgeCounts> readBridgeCounts(const DictionaryAttr &record) {
  auto execution = dictionaryField(record, kKeyExecution);
  if (!execution)
    return std::nullopt;
  auto bridge = dictionaryField(execution, kKeyBridgeCounts);
  if (!bridge)
    return std::nullopt;
  if (!hasI64(bridge, kKeyPackActSites) ||
      !hasI64(bridge, kKeyPackWeightSites) ||
      !hasI64(bridge, kKeyUnpackSites) ||
      !stringField(bridge, kKeyCountSemantics) ||
      stringField(bridge, kKeyCountSemantics).getValue() != kCountSemantics)
    return std::nullopt;
  return BridgeCounts{
      cast<IntegerAttr>(bridge.get(kKeyPackActSites)).getInt(),
      cast<IntegerAttr>(bridge.get(kKeyPackWeightSites)).getInt(),
      cast<IntegerAttr>(bridge.get(kKeyUnpackSites)).getInt()};
}

void setTopBridgeCounts(ModuleOp module, NamedAttrList &fields,
                        ArrayAttr matmuls) {
  MLIRContext *ctx = module.getContext();
  int64_t packAct = 0;
  int64_t packWeight = 0;
  int64_t unpack = 0;
  for (Attribute item : matmuls) {
    auto record = cast<DictionaryAttr>(item);
    StringAttr plan = stringField(record, kKeyPlan);
    if (!plan || parsePlan(plan.getValue()) == PlanKind::HVX)
      continue;
    std::optional<BridgeCounts> counts = readBridgeCounts(record);
    if (!counts)
      continue;
    packAct += counts->packAct;
    packWeight += counts->packWeight;
    unpack += counts->unpack;
  }
  auto i64 = IntegerType::get(ctx, 64);
  fields.set(kKeyPackActSites, IntegerAttr::get(i64, packAct));
  fields.set(kKeyPackWeightSites, IntegerAttr::get(i64, packWeight));
  fields.set(kKeyUnpackSites, IntegerAttr::get(i64, unpack));
}

void clearFingerprints(MLIRContext *ctx, SmallVectorImpl<Attribute> &entries) {
  for (Attribute &item : entries) {
    auto record = cast<DictionaryAttr>(item);
    if (!record.get(kKeyPlanFingerprint))
      continue;
    NamedAttrList fields(record);
    fields.erase(kKeyPlanFingerprint);
    item = fields.getDictionary(ctx);
  }
}

std::string canonicalJson(Attribute attr);
void appendCanonicalJson(llvm::raw_ostream &stream, Attribute attr);

void appendCanonicalString(llvm::raw_ostream &stream, StringRef value) {
  stream << json::Value(value);
}

void appendCanonicalJson(llvm::raw_ostream &stream, Attribute attr) {
  if (!attr || isNone(attr)) {
    stream << "null";
    return;
  }
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    stream << integer.getInt();
    return;
  }
  if (auto boolean = dyn_cast<BoolAttr>(attr)) {
    stream << (boolean.getValue() ? "true" : "false");
    return;
  }
  if (auto string = dyn_cast<StringAttr>(attr)) {
    appendCanonicalString(stream, string.getValue());
    return;
  }
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    stream << '[';
    for (auto [index, item] : llvm::enumerate(array)) {
      if (index)
        stream << ',';
      appendCanonicalJson(stream, item);
    }
    stream << ']';
    return;
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attr)) {
    SmallVector<NamedAttribute> fields(dictionary.begin(), dictionary.end());
    llvm::sort(fields, [](NamedAttribute lhs, NamedAttribute rhs) {
      return lhs.getName().strref() < rhs.getName().strref();
    });
    stream << '{';
    for (auto [index, field] : llvm::enumerate(fields)) {
      if (index)
        stream << ',';
      appendCanonicalString(stream, field.getName().strref());
      stream << ':';
      appendCanonicalJson(stream, field.getValue());
    }
    stream << '}';
    return;
  }
  // Manifest attributes are JSON-compatible by construction.  Keep a stable
  // representation if a future pass accidentally supplies another attribute;
  // final validation will reject it before this helper is used for a digest.
  stream << "null";
}

std::string canonicalJson(Attribute attr) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  appendCanonicalJson(stream, attr);
  stream.flush();
  return result;
}

std::string computePlanFingerprint(DictionaryAttr record,
                                   const WeightPolicyMap &policies) {
  MLIRContext *ctx = record.getContext();
  NamedAttrList payload;
  payload.set(kKeySchema, StringAttr::get(ctx, kManifestSchema));
  for (NamedAttribute field : record) {
    if (field.getName() == kKeyPlanFingerprint)
      continue;
    payload.set(field.getName(), field.getValue());
  }
  if (dictionaryField(record, kKeyWeightBinding)) {
    auto binding = dictionaryField(record, kKeyWeightBinding);
    StringAttr kind = stringField(binding, kKeyKind);
    if (kind && kind.getValue() == kHmxWeightKindArgumentSlot) {
      auto ref = dictionaryField(binding, kKeyPolicyRef);
      if (ref && stringField(ref, kKeyFunction) && hasI64(ref, kKeySlot)) {
        std::optional<DictionaryAttr> policy = findWeightPolicy(
            policies, stringField(ref, kKeyFunction).getValue(),
            cast<IntegerAttr>(ref.get(kKeySlot)).getInt());
        if (policy)
          payload.set("weight_policy", *policy);
      }
    }
  }
  std::string canonical = canonicalJson(payload.getDictionary(ctx));
  llvm::SHA256 hasher;
  hasher.update(canonical);
  std::array<uint8_t, 32> digest = hasher.final();
  return (Twine(kFingerprintPrefix) +
          llvm::toHex(llvm::ArrayRef<uint8_t>(digest.data(), digest.size()),
                      true))
      .str();
}

/// Read and validate the module attribute.  The default is deliberately a
/// producer phase: later HMX passes may not have filled pipeline, workspace,
/// weight, or fingerprint fields yet.  Publication uses requireFinal=true;
/// only that mode requires the complete executable contract and verifies the
/// fingerprint against the canonical semantic value.
FailureOr<DictionaryAttr> readManifest(ModuleOp module, bool reportErrors,
                                       bool createIfMissing,
                                       bool requireFinal = false,
                                       bool checkFingerprints = true) {
  Attribute raw = module->getAttr(kManifestAttr);
  if (!raw) {
    if (!createIfMissing)
      return emitManifestError(module, reportErrors,
                               "hmx.kernel_manifest is missing");
    MLIRContext *ctx = module.getContext();
    NamedAttrList fields;
    fields.append(kKeySchema, StringAttr::get(ctx, kManifestSchema));
    fields.append(kKeyMatmuls, ArrayAttr::get(ctx, {}));
    fields.append(kKeyWeightPolicies, ArrayAttr::get(ctx, {}));
    auto i64 = IntegerType::get(ctx, 64);
    fields.append(kKeyPackActSites, IntegerAttr::get(i64, 0));
    fields.append(kKeyPackWeightSites, IntegerAttr::get(i64, 0));
    fields.append(kKeyUnpackSites, IntegerAttr::get(i64, 0));
    fields.append(kKeyCountSemantics, StringAttr::get(ctx, kCountSemantics));
    module->setAttr(kManifestAttr, fields.getDictionary(ctx));
    raw = module->getAttr(kManifestAttr);
  }

  auto manifest = dyn_cast<DictionaryAttr>(raw);
  if (!manifest)
    return emitManifestError(module, reportErrors,
                             "hmx.kernel_manifest must be a DictionaryAttr");
  if (failed(rejectUnknownFields(
          module, manifest,
          {StringRef(kKeySchema), StringRef(kKeyMatmuls),
           StringRef(kKeyWeightPolicies), StringRef(kKeyPackActSites),
           StringRef(kKeyPackWeightSites), StringRef(kKeyUnpackSites),
           StringRef(kKeyCountSemantics)},
          reportErrors, "manifest")))
    return failure();
  auto schema = stringField(manifest, kKeySchema);
  if (!schema || schema.getValue() != kManifestSchema)
    return emitManifestError(module, reportErrors,
                             "hmx.kernel_manifest has an unsupported schema");
  auto matmuls = arrayField(manifest, kKeyMatmuls);
  if (!matmuls)
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest.matmuls must be an ArrayAttr");

  WeightPolicyMap policies;
  if (failed(validateWeightPolicies(module, manifest, reportErrors, false,
                                    policies)))
    return failure();
  if (requireFinal && !manifest.get(kKeyWeightPolicies))
    return emitManifestError(module, reportErrors,
                             "hmx.kernel_manifest has no weight_policies");

  std::set<std::pair<std::string, int64_t>> recordKeys;
  SmallVector<RecordSummary> summaries;
  summaries.reserve(matmuls.size());
  int64_t packActTotal = 0;
  int64_t packWeightTotal = 0;
  int64_t unpackTotal = 0;
  for (Attribute item : matmuls) {
    auto record = dyn_cast<DictionaryAttr>(item);
    if (!record)
      return emitManifestError(
          module, reportErrors,
          "hmx.kernel_manifest.matmuls must contain dictionaries");
    RecordSummary summary;
    if (failed(validateRecord(module, record, reportErrors, requireFinal,
                              checkFingerprints, summary)))
      return failure();
    StringAttr function = stringField(record, kKeyFunction);
    int64_t id = cast<IntegerAttr>(record.get(kKeyId)).getInt();
    if (!recordKeys.emplace(function.getValue().str(), id).second)
      return emitManifestError(
          module, reportErrors,
          "hmx.kernel_manifest contains duplicate function-local records");
    summaries.push_back(summary);
    if (summary.bridgeCounts) {
      packActTotal += summary.bridgeCounts->packAct;
      packWeightTotal += summary.bridgeCounts->packWeight;
      unpackTotal += summary.bridgeCounts->unpack;
    }
  }

  for (StringRef field :
       {kKeyPackActSites, kKeyPackWeightSites, kKeyUnpackSites}) {
    if (manifest.get(field) &&
        (!hasI64(manifest, field) ||
         cast<IntegerAttr>(manifest.get(field)).getInt() < 0))
      return emitManifestError(
          module, reportErrors,
          "hmx.kernel_manifest has an invalid bridge count");
  }
  if (requireFinal && (!hasI64(manifest, kKeyPackActSites) ||
                       !hasI64(manifest, kKeyPackWeightSites) ||
                       !hasI64(manifest, kKeyUnpackSites)))
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest has incomplete bridge counts");
  if (hasI64(manifest, kKeyPackActSites) &&
      cast<IntegerAttr>(manifest.get(kKeyPackActSites)).getInt() !=
          packActTotal)
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest bridge totals disagree with matmul records");
  if (hasI64(manifest, kKeyPackWeightSites) &&
      cast<IntegerAttr>(manifest.get(kKeyPackWeightSites)).getInt() !=
          packWeightTotal)
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest bridge totals disagree with matmul records");
  if (hasI64(manifest, kKeyUnpackSites) &&
      cast<IntegerAttr>(manifest.get(kKeyUnpackSites)).getInt() != unpackTotal)
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest bridge totals disagree with matmul records");
  if (manifest.get(kKeyCountSemantics) &&
      (!stringField(manifest, kKeyCountSemantics) ||
       stringField(manifest, kKeyCountSemantics).getValue() != kCountSemantics))
    return emitManifestError(module, reportErrors,
                             "hmx.kernel_manifest has invalid count semantics");
  if (requireFinal && !stringField(manifest, kKeyCountSemantics))
    return emitManifestError(module, reportErrors,
                             "hmx.kernel_manifest has no count semantics");

  if (requireFinal) {
    std::set<std::pair<std::string, int64_t>> referencedPolicies;
    for (size_t index = 0; index < matmuls.size(); ++index) {
      auto record = cast<DictionaryAttr>(matmuls[index]);
      const RecordSummary &summary = summaries[index];
      if (summary.weightBinding) {
        auto policy =
            findWeightPolicy(policies, summary.weightBinding->function,
                             summary.weightBinding->slot);
        if (!policy)
          return emitManifestError(
              module, reportErrors,
              "HMX argument weight binding has no matching weight policy");
        referencedPolicies.emplace(summary.weightBinding->function.str(),
                                   summary.weightBinding->slot);
        StringAttr recordFunction = stringField(record, kKeyFunction);
        if (recordFunction.getValue() != summary.weightBinding->function)
          return emitManifestError(
              module, reportErrors,
              "HMX weight binding function disagrees with its record");
        auto consumers = arrayField(*policy, kKeyConsumers);
        bool found = false;
        for (Attribute consumer : consumers) {
          if (isI64(consumer) &&
              cast<IntegerAttr>(consumer).getInt() ==
                  cast<IntegerAttr>(record.get(kKeyId)).getInt()) {
            found = true;
            break;
          }
        }
        if (!found)
          return emitManifestError(
              module, reportErrors,
              "HMX weight policy consumers do not name the bound record");
      }
    }
    for (const auto &entry : policies) {
      const WeightPolicySummary &summary = entry.second;
      auto consumers = arrayField(summary.value, kKeyConsumers);
      for (Attribute consumer : consumers) {
        if (!isI64(consumer))
          return emitManifestError(module, reportErrors,
                                   "HMX weight policy has an invalid consumer");
        int64_t id = cast<IntegerAttr>(consumer).getInt();
        auto recordIt = recordKeys.find({summary.function.str(), id});
        if (recordIt == recordKeys.end())
          return emitManifestError(module, reportErrors,
                                   "HMX weight policy names an unknown record");
        size_t index = 0;
        for (; index < matmuls.size(); ++index) {
          auto candidate = cast<DictionaryAttr>(matmuls[index]);
          StringAttr function = stringField(candidate, kKeyFunction);
          if (function.getValue() == summary.function &&
              cast<IntegerAttr>(candidate.get(kKeyId)).getInt() == id)
            break;
        }
        if (index == matmuls.size() || summaries[index].plan == PlanKind::HVX)
          return emitManifestError(
              module, reportErrors,
              "HMX weight policy names a non-HMX consumer");
        if (!summaries[index].weightBinding ||
            summaries[index].weightBinding->function != summary.function ||
            summaries[index].weightBinding->slot != summary.slot)
          return emitManifestError(
              module, reportErrors,
              "HMX weight policy consumer is bound to a different slot");
        if (summary.policy == kWeightResidentPrepack) {
          auto consumer = cast<DictionaryAttr>(matmuls[index]);
          StringAttr dtypesValue =
              stringField(dictionaryField(consumer, kKeyDtypes), kKeyRhsElem);
          auto logical = dictionaryField(consumer, kKeyLogical);
          bool aligned = true;
          if (logical) {
            for (StringRef axis : {kKeyM, kKeyN, kKeyK}) {
              auto dimension =
                  dyn_cast_or_null<DictionaryAttr>(logical.get(axis));
              if (!dimension || !hasI64(dimension, "value") ||
                  cast<IntegerAttr>(dimension.get("value")).getInt() % 32 != 0)
                aligned = false;
            }
          }
          // A resident-prepack consumer's weight is either exactly fp16 (the
          // host permutes its bytes) or fp32 (the host quantises with the same
          // conversion the device pack would run). The reason has to name the
          // same dtype the record declares: `eligible-quantized-f32` exists so
          // a manifest cannot claim an exact fp16 image for an fp32 weight.
          bool admittedF16 =
              dtypesValue && dtypesValue.getValue() == kHmxDTypeF16;
          bool admittedF32 =
              dtypesValue && dtypesValue.getValue() == kHmxDTypeF32;
          StringAttr reasonAttr = stringField(summary.value, kKeyReason);
          StringRef reason =
              reasonAttr ? reasonAttr.getValue() : StringRef();
          bool dtypeMatchesReason =
              (reason != kWeightEligibleAlignedF16 || admittedF16) &&
              (reason != kWeightEligibleQuantizedF32 || admittedF32);
          if (summaries[index].plan != PlanKind::FullHMX || !aligned ||
              (!admittedF16 && !admittedF32) || !dtypeMatchesReason)
            return emitManifestError(
                module, reportErrors,
                "resident-prepack is only valid for aligned f16/f32 HMX "
                "consumers whose policy reason matches the weight dtype");
        }
      }
    }
    for (const auto &entry : policies) {
      if (!referencedPolicies.count(entry.first))
        return emitManifestError(
            module, reportErrors,
            "HMX weight policy is not referenced by a final record");
    }

    if (checkFingerprints) {
      for (size_t index = 0; index < matmuls.size(); ++index) {
        auto record = cast<DictionaryAttr>(matmuls[index]);
        StringAttr fingerprint = stringField(record, kKeyPlanFingerprint);
        if (!fingerprint)
          return emitManifestError(
              module, reportErrors,
              "HMX manifest record has no plan_fingerprint");
        if (fingerprint.getValue() != computePlanFingerprint(record, policies))
          return emitManifestError(
              module, reportErrors,
              "HMX manifest plan_fingerprint disagrees with final plan");
      }
    }
  }
  return manifest;
}

LogicalResult writeManifest(ModuleOp module, NamedAttrList &fields) {
  MLIRContext *ctx = module.getContext();
  Attribute previous = module->getAttr(kManifestAttr);
  module->setAttr(kManifestAttr, fields.getDictionary(ctx));
  if (failed(readManifest(module, /*reportErrors=*/true,
                          /*createIfMissing=*/true))) {
    if (previous)
      module->setAttr(kManifestAttr, previous);
    else
      module->removeAttr(kManifestAttr);
    return failure();
  }
  return success();
}

std::optional<std::size_t> findRecord(ArrayAttr matmuls, StringRef functionName,
                                      int64_t id) {
  for (auto [index, item] : llvm::enumerate(matmuls)) {
    auto record = dyn_cast<DictionaryAttr>(item);
    if (!record)
      continue;
    StringAttr function = stringField(record, kKeyFunction);
    auto recordId = dyn_cast_or_null<IntegerAttr>(record.get(kKeyId));
    if (function && recordId && function.getValue() == functionName &&
        recordId.getInt() == id)
      return index;
  }
  return std::nullopt;
}

json::Value attrToJson(Attribute attr) {
  if (!attr || isNone(attr))
    return json::Value(nullptr);
  if (auto integer = dyn_cast<IntegerAttr>(attr))
    return integer.getInt();
  if (auto boolean = dyn_cast<BoolAttr>(attr))
    return boolean.getValue();
  if (auto string = dyn_cast<StringAttr>(attr))
    return string.getValue();
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    json::Array result;
    for (Attribute item : array)
      result.push_back(attrToJson(item));
    return json::Value(std::move(result));
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attr)) {
    json::Object object;
    for (NamedAttribute field : dictionary)
      object[field.getName().strref()] = attrToJson(field.getValue());
    return json::Value(std::move(object));
  }
  return json::Value(nullptr);
}

} // namespace

LogicalResult mlir::hmx::ensureHmxManifest(ModuleOp module) {
  return readManifest(module, /*reportErrors=*/true,
                      /*createIfMissing=*/true);
}

LogicalResult
mlir::hmx::addOrReplaceHmxManifestRecords(ModuleOp module,
                                          ArrayRef<DictionaryAttr> records) {
  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(current))
    return failure();

  MLIRContext *ctx = module.getContext();
  ArrayAttr currentMatmuls = arrayField(*current, kKeyMatmuls);
  SmallVector<Attribute> entries(currentMatmuls.begin(), currentMatmuls.end());
  for (DictionaryAttr record : records) {
    RecordSummary summary;
    if (failed(validateRecord(module, record, /*report=*/true,
                              /*requireFinal=*/false,
                              /*checkFingerprints=*/false, summary)))
      return failure();
    StringAttr function = stringField(record, kKeyFunction);
    int64_t id = cast<IntegerAttr>(record.get(kKeyId)).getInt();
    std::optional<std::size_t> existing =
        findRecord(ArrayAttr::get(ctx, entries), function.getValue(), id);
    if (existing) {
      NamedAttrList replacement(record);
      replacement.erase(kKeyPlanFingerprint);
      entries[*existing] = replacement.getDictionary(ctx);
    } else {
      NamedAttrList insertion(record);
      insertion.erase(kKeyPlanFingerprint);
      entries.push_back(insertion.getDictionary(ctx));
    }
  }

  std::sort(entries.begin(), entries.end(), [](Attribute lhs, Attribute rhs) {
    auto left = cast<DictionaryAttr>(lhs);
    auto right = cast<DictionaryAttr>(rhs);
    StringAttr leftFunction = stringField(left, kKeyFunction);
    StringAttr rightFunction = stringField(right, kKeyFunction);
    if (leftFunction.getValue() != rightFunction.getValue())
      return leftFunction.getValue() < rightFunction.getValue();
    return cast<IntegerAttr>(left.get(kKeyId)).getInt() <
           cast<IntegerAttr>(right.get(kKeyId)).getInt();
  });

  NamedAttrList fields(*current);
  ArrayAttr sorted = ArrayAttr::get(ctx, entries);
  fields.set(kKeyMatmuls, sorted);
  setTopBridgeCounts(module, fields, sorted);
  if (!fields.get(kKeyCountSemantics))
    fields.set(kKeyCountSemantics, StringAttr::get(ctx, kCountSemantics));
  return writeManifest(module, fields);
}

LogicalResult mlir::hmx::setHmxManifestPipelineDecision(
    ModuleOp module, StringRef functionName, int64_t id, int64_t requested,
    StringRef selected, int64_t depth, StringRef reason) {
  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(current))
    return failure();
  if (selected != kHmxPipelineSerial && selected != kHmxPipelineStaged)
    return emitManifestError(module, true,
                             "invalid HMX pipeline selection in manifest");
  if (requested < 0 || depth < 0 ||
      (selected == kHmxPipelineSerial && depth != 0) ||
      (selected == kHmxPipelineStaged && depth == 0))
    return emitManifestError(module, true,
                             "invalid HMX pipeline depth in manifest");
  if (!reason.empty() && !isCanonicalPipelineReason(reason))
    return emitManifestError(module, true,
                             "invalid HMX pipeline reason in manifest");

  MLIRContext *ctx = module.getContext();
  ArrayAttr matmuls = arrayField(*current, kKeyMatmuls);
  std::optional<std::size_t> index = findRecord(matmuls, functionName, id);
  if (!index)
    return emitManifestError(module, true,
                             "HMX matmul decision id has no manifest record");
  auto target = cast<DictionaryAttr>(matmuls[*index]);
  StringAttr plan = stringField(target, kKeyPlan);
  if (!plan || parsePlan(plan.getValue()) == PlanKind::HVX)
    return emitManifestError(module, true,
                             "pipeline decision refers to a non-HMX record");

  SmallVector<Attribute> entries(matmuls.begin(), matmuls.end());
  NamedAttrList record(target);
  DictionaryAttr targetExecution = dictionaryField(target, kKeyExecution);
  NamedAttrList execution(targetExecution ? targetExecution
                                          : DictionaryAttr::get(ctx, {}));
  NamedAttrList pipeline;
  if (DictionaryAttr old = dictionaryField(targetExecution, kKeyPipeline))
    pipeline = old;
  auto i64 = IntegerType::get(ctx, 64);
  pipeline.set(kKeyPipelineRequested, IntegerAttr::get(i64, requested));
  pipeline.set(kKeyPipelineSelected, StringAttr::get(ctx, selected));
  pipeline.set(kKeyPipelineDepth, IntegerAttr::get(i64, depth));
  if (!reason.empty())
    pipeline.set(kKeyPipelineReason, StringAttr::get(ctx, reason));
  else
    pipeline.erase(kKeyPipelineReason);
  execution.set(kKeyPipeline, pipeline.getDictionary(ctx));
  record.set(kKeyExecution, execution.getDictionary(ctx));
  record.erase(kKeyPlanFingerprint);
  entries[*index] = record.getDictionary(ctx);

  NamedAttrList fields(*current);
  fields.set(kKeyMatmuls, ArrayAttr::get(ctx, entries));
  return writeManifest(module, fields);
}

LogicalResult mlir::hmx::bindHmxManifestWeightSlot(ModuleOp module,
                                                   StringRef function,
                                                   int64_t recordId,
                                                   int64_t slot) {
  if (!isNonEmptyString(function) || recordId < 0 || slot < 0)
    return emitManifestError(module, true, "invalid HMX weight-slot binding");
  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(current))
    return failure();

  ArrayAttr matmuls = arrayField(*current, kKeyMatmuls);
  std::optional<std::size_t> index = findRecord(matmuls, function, recordId);
  if (!index)
    return emitManifestError(module, true,
                             "weight-slot binding has no manifest record");
  auto record = cast<DictionaryAttr>(matmuls[*index]);
  StringAttr plan = stringField(record, kKeyPlan);
  if (!plan || parsePlan(plan.getValue()) == PlanKind::HVX)
    return emitManifestError(module, true,
                             "weight-slot binding refers to an HVX record");

  MLIRContext *ctx = module.getContext();
  NamedAttrList reference;
  reference.append(kKeyFunction, StringAttr::get(ctx, function));
  reference.append(kKeySlot, IntegerAttr::get(IntegerType::get(ctx, 64), slot));
  NamedAttrList binding;
  binding.append(kKeyKind, StringAttr::get(ctx, kHmxWeightKindArgumentSlot));
  binding.append(kKeyPolicyRef, reference.getDictionary(ctx));
  NamedAttrList fields(record);
  fields.set(kKeyWeightBinding, binding.getDictionary(ctx));
  fields.erase(kKeyPlanFingerprint);

  SmallVector<Attribute> entries(matmuls.begin(), matmuls.end());
  entries[*index] = fields.getDictionary(ctx);
  NamedAttrList manifest(*current);
  manifest.set(kKeyMatmuls, ArrayAttr::get(ctx, entries));
  return writeManifest(module, manifest);
}

LogicalResult mlir::hmx::setHmxManifestWeightPolicy(
    ModuleOp module, StringRef function, int64_t slot, StringRef policyName,
    StringRef reason, ArrayRef<int64_t> consumers) {
  if (!isNonEmptyString(function) || slot < 0)
    return emitManifestError(module, true,
                             "invalid HMX weight policy function/slot");
  if (!isCanonicalWeightReason(policyName, reason))
    return emitManifestError(module, true, "invalid HMX weight policy/reason");
  std::set<int64_t> uniqueConsumers;
  for (int64_t consumer : consumers) {
    if (consumer < 0 || !uniqueConsumers.insert(consumer).second)
      return emitManifestError(module, true,
                               "invalid HMX weight policy consumers");
  }

  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(current))
    return failure();
  MLIRContext *ctx = module.getContext();
  ArrayAttr oldPolicies = arrayField(*current, kKeyWeightPolicies);
  SmallVector<Attribute> entries;
  if (oldPolicies)
    entries.append(oldPolicies.begin(), oldPolicies.end());
  std::optional<std::size_t> existing;
  for (auto [index, item] : llvm::enumerate(entries)) {
    auto policy = cast<DictionaryAttr>(item);
    StringAttr oldFunction = stringField(policy, kKeyFunction);
    int64_t oldSlot = cast<IntegerAttr>(policy.get(kKeySlot)).getInt();
    if (oldFunction.getValue() == function && oldSlot == slot) {
      existing = index;
      break;
    }
  }

  auto i64 = IntegerType::get(ctx, 64);
  SmallVector<Attribute> consumerAttrs;
  for (int64_t consumer : uniqueConsumers)
    consumerAttrs.push_back(IntegerAttr::get(i64, consumer));
  NamedAttrList policyFields;
  policyFields.set(kKeyFunction, StringAttr::get(ctx, function));
  policyFields.set(kKeySlot, IntegerAttr::get(i64, slot));
  policyFields.set(kKeyPolicy, StringAttr::get(ctx, policyName));
  policyFields.set(kKeyReason, StringAttr::get(ctx, reason));
  policyFields.set(kKeyConsumers, ArrayAttr::get(ctx, consumerAttrs));
  DictionaryAttr policy = policyFields.getDictionary(ctx);
  if (existing)
    entries[*existing] = policy;
  else
    entries.push_back(policy);

  std::sort(entries.begin(), entries.end(), [](Attribute lhs, Attribute rhs) {
    auto left = cast<DictionaryAttr>(lhs);
    auto right = cast<DictionaryAttr>(rhs);
    StringAttr leftFunction = stringField(left, kKeyFunction);
    StringAttr rightFunction = stringField(right, kKeyFunction);
    if (leftFunction.getValue() != rightFunction.getValue())
      return leftFunction.getValue() < rightFunction.getValue();
    return cast<IntegerAttr>(left.get(kKeySlot)).getInt() <
           cast<IntegerAttr>(right.get(kKeySlot)).getInt();
  });

  if (policyName == kWeightResidentPrepack) {
    std::set<std::string> residentFunctions;
    for (Attribute item : entries) {
      auto value = cast<DictionaryAttr>(item);
      if (stringField(value, kKeyPolicy).getValue() == kWeightResidentPrepack)
        residentFunctions.insert(
            stringField(value, kKeyFunction).getValue().str());
    }
    if (residentFunctions.size() > 1)
      return emitManifestError(module, true,
                               "resident-prepack is limited to one function");
  }

  SmallVector<Attribute> records;
  ArrayAttr matmuls = arrayField(*current, kKeyMatmuls);
  records.append(matmuls.begin(), matmuls.end());
  clearFingerprints(ctx, records);
  NamedAttrList fields(*current);
  fields.set(kKeyWeightPolicies, ArrayAttr::get(ctx, entries));
  fields.set(kKeyMatmuls, ArrayAttr::get(ctx, records));
  return writeManifest(module, fields);
}

LogicalResult
mlir::hmx::reconcileHmxManifestWeightPolicies(ModuleOp module,
                                              bool prepackRuntimeWeights) {
  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(current))
    return failure();

  struct PrepackInfo {
    std::optional<int64_t> logicalN;
    // The source dtype of the contract's crouton image: an f32 source is
    // quantised by the host, which the policy reason distinguishes.
    bool quantizedF32 = false;
  };
  std::map<std::string, PrepackInfo> prepackedSlots;
  if (auto prepack =
          module->getAttrOfType<StringAttr>(kHmxWeightPrepackAttr)) {
    auto parsed = json::parse(prepack.getValue());
    if (!parsed)
      return emitManifestError(module, true,
                               "hmx.weight_prepack is not valid JSON");
    const json::Array *entries = parsed->getAsArray();
    if (!entries)
      return emitManifestError(module, true,
                               "hmx.weight_prepack must be a JSON array");
    for (const json::Value &value : *entries) {
      const json::Object *object = value.getAsObject();
      if (!object)
        return emitManifestError(module, true,
                                 "hmx.weight_prepack entry must be an object");
      auto function = object->getString("func");
      auto slot = object->getInteger("slot");
      if (!function || !slot || *slot < 0)
        return emitManifestError(
            module, true, "hmx.weight_prepack entry has no function/slot");
      PrepackInfo info;
      if (const json::Array *shape = object->getArray("shape")) {
        if (shape->size() != 2)
          return emitManifestError(
              module, true, "hmx.weight_prepack entry has an invalid shape");
        auto n = (*shape)[1].getAsInteger();
        if (!n)
          return emitManifestError(
              module, true, "hmx.weight_prepack entry has an invalid shape");
        info.logicalN = *n;
      }
      if (auto sourceDtype = object->getString("dtype"))
        info.quantizedF32 = *sourceDtype == kHmxDTypeF32;
      std::string key = (*function).str() + "\x1f" + std::to_string(*slot);
      if (!prepackedSlots.emplace(std::move(key), info).second)
        return emitManifestError(
            module, true, "hmx.weight_prepack has duplicate function/slot");
    }
  }

  using SlotKey = std::pair<std::string, int64_t>;
  struct SlotInfo {
    SmallVector<int64_t> consumers;
    bool hasTail = false;
    bool hasF32 = false;
    bool hasNonStatic = false;
    bool hasShapeConflict = false;
    std::optional<std::array<int64_t, 3>> logical;
  };
  std::map<SlotKey, SlotInfo> slots;
  for (Attribute item : arrayField(*current, kKeyMatmuls)) {
    auto record = cast<DictionaryAttr>(item);
    StringAttr plan = stringField(record, kKeyPlan);
    if (!plan || parsePlan(plan.getValue()) == PlanKind::HVX)
      continue;
    DictionaryAttr binding = dictionaryField(record, kKeyWeightBinding);
    if (!binding ||
        stringField(binding, kKeyKind).getValue() != kHmxWeightKindArgumentSlot)
      continue;
    DictionaryAttr reference = dictionaryField(binding, kKeyPolicyRef);
    StringAttr function = stringField(reference, kKeyFunction);
    if (!function || !hasI64(reference, kKeySlot))
      return emitManifestError(
          module, true, "HMX manifest weight binding has no policy reference");
    int64_t slot = cast<IntegerAttr>(reference.get(kKeySlot)).getInt();
    if (slot < 0)
      return emitManifestError(
          module, true, "HMX manifest weight binding has a negative slot");
    SlotKey key{function.getValue().str(), slot};
    SlotInfo &info = slots[key];
    info.consumers.push_back(cast<IntegerAttr>(record.get(kKeyId)).getInt());
    if (parsePlan(plan.getValue()) == PlanKind::HMXTail)
      info.hasTail = true;
    DictionaryAttr dtypes = dictionaryField(record, kKeyDtypes);
    StringAttr rhs = stringField(dtypes, kKeyRhsElem);
    if (rhs && rhs.getValue() != kHmxDTypeF16)
      info.hasF32 = true;
    StringAttr state = stringField(record, kKeyShapeState);
    if (!state || state.getValue() != kHmxShapeStateStatic)
      info.hasNonStatic = true;
    std::array<int64_t, 3> shape{};
    if (auto logical = dictionaryField(record, kKeyLogical)) {
      bool valid = true;
      unsigned axis = 0;
      for (StringRef name : {kKeyM, kKeyN, kKeyK}) {
        auto dimension = dyn_cast_or_null<DictionaryAttr>(logical.get(name));
        if (!dimension || !hasI64(dimension, "value")) {
          valid = false;
          break;
        }
        shape[axis++] = cast<IntegerAttr>(dimension.get("value")).getInt();
      }
      if (valid) {
        if (info.logical && *info.logical != shape)
          info.hasShapeConflict = true;
        info.logical = shape;
      }
    }
  }

  for (auto &[key, info] : slots) {
    llvm::sort(info.consumers);
    info.consumers.erase(
        std::unique(info.consumers.begin(), info.consumers.end()),
        info.consumers.end());
    std::string lookup = key.first + "\x1f" + std::to_string(key.second);
    auto prepacked = prepackedSlots.find(lookup);
    StringRef policy = kWeightDevicePack;
    StringRef reason = kWeightPrepackDisabled;
    if (prepacked != prepackedSlots.end()) {
      policy = kWeightResidentPrepack;
      bool b2 = prepacked->second.logicalN && info.logical &&
                *prepacked->second.logicalN != (*info.logical)[1];
      reason = b2 ? kWeightEligibleB2NSlice
                  : (prepacked->second.quantizedF32
                         ? kWeightEligibleQuantizedF32
                         : kWeightEligibleAlignedF16);
    } else if (prepackRuntimeWeights) {
      if (info.hasShapeConflict)
        reason = kWeightIncompatibleConsumers;
      else if (info.hasTail)
        reason = kWeightTailConsumer;
      else if (info.hasF32)
        reason = kWeightF32Source;
      else
        reason = kWeightUnprovenOffset;
    }
    if (failed(setHmxManifestWeightPolicy(module, key.first, key.second, policy,
                                          reason, info.consumers)))
      return failure();
  }

  // A rerun after a record changed must not leave a policy for a slot that no
  // longer has an HMX consumer.  Such an orphan is not a harmless cache entry:
  // final validation would reject it, and retaining it would make the next
  // fingerprint depend on stale classification state.
  FailureOr<DictionaryAttr> latest =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(latest))
    return failure();
  ArrayAttr oldPolicies = arrayField(*latest, kKeyWeightPolicies);
  if (!oldPolicies)
    return success();
  SmallVector<Attribute> kept;
  for (Attribute item : oldPolicies) {
    auto policy = cast<DictionaryAttr>(item);
    SlotKey key{stringField(policy, kKeyFunction).getValue().str(),
                cast<IntegerAttr>(policy.get(kKeySlot)).getInt()};
    if (slots.count(key))
      kept.push_back(item);
  }
  if (kept.size() != oldPolicies.size()) {
    MLIRContext *ctx = module.getContext();
    SmallVector<Attribute> records;
    ArrayAttr matmuls = arrayField(*latest, kKeyMatmuls);
    records.append(matmuls.begin(), matmuls.end());
    clearFingerprints(ctx, records);
    NamedAttrList fields(*latest);
    fields.set(kKeyWeightPolicies, ArrayAttr::get(ctx, kept));
    fields.set(kKeyMatmuls, ArrayAttr::get(ctx, records));
    return writeManifest(module, fields);
  }
  return success();
}

LogicalResult mlir::hmx::setHmxManifestWorkspaceClass(ModuleOp module,
                                                      StringRef function,
                                                      StringRef workspaceClass,
                                                      StringRef gridPolicy) {
  if (!isNonEmptyString(function) ||
      !isCanonicalWorkspaceClass(workspaceClass) ||
      !isCanonicalGridPolicy(gridPolicy))
    return emitManifestError(module, true,
                             "invalid HMX workspace/grid classification");
  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(current))
    return failure();

  MLIRContext *ctx = module.getContext();
  ArrayAttr matmuls = arrayField(*current, kKeyMatmuls);
  SmallVector<Attribute> entries(matmuls.begin(), matmuls.end());
  for (auto [index, item] : llvm::enumerate(entries)) {
    auto record = cast<DictionaryAttr>(item);
    StringAttr recordFunction = stringField(record, kKeyFunction);
    if (recordFunction.getValue() != function)
      continue;
    StringAttr plan = stringField(record, kKeyPlan);
    if (!plan || parsePlan(plan.getValue()) == PlanKind::HVX)
      continue;
    if (parsePlan(plan.getValue()) == PlanKind::HMXTail &&
        gridPolicy != kGridSingleInstance)
      return emitManifestError(
          module, true, "hmx-tail requires a single-instance grid policy");
    NamedAttrList fields(record);
    fields.set(kKeyWorkspaceClass, StringAttr::get(ctx, workspaceClass));
    fields.set(kKeyGridPolicy, StringAttr::get(ctx, gridPolicy));
    fields.set(kKeyVtcmAccounting, StringAttr::get(ctx, kBridgeOnlyAccounting));
    fields.erase(kKeyPlanFingerprint);
    entries[index] = fields.getDictionary(ctx);
  }

  NamedAttrList fields(*current);
  fields.set(kKeyMatmuls, ArrayAttr::get(ctx, entries));
  return writeManifest(module, fields);
}

LogicalResult mlir::hmx::restoreHmxManifestDecisionIds(ModuleOp module,
                                                       func::FuncOp func) {
  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(current))
    return failure();

  SmallVector<int64_t> selectedIds;
  for (Attribute item : arrayField(*current, kKeyMatmuls)) {
    auto record = cast<DictionaryAttr>(item);
    StringAttr recordFunction = stringField(record, kKeyFunction);
    StringAttr plan = stringField(record, kKeyPlan);
    if (recordFunction.getValue() == func.getName() && plan &&
        parsePlan(plan.getValue()) != PlanKind::HVX)
      selectedIds.push_back(cast<IntegerAttr>(record.get(kKeyId)).getInt());
  }

  SmallVector<MatmulOp> matmuls;
  func.walk([&](MatmulOp op) { matmuls.push_back(op); });
  if (selectedIds.empty()) {
    if (matmuls.empty())
      return success();
    return emitManifestError(module, true,
                             "hmx.matmul has no selected HMX manifest record");
  }
  if (selectedIds.size() != matmuls.size())
    return emitManifestError(
        module, true,
        "hmx.matmul count disagrees with selected HMX manifest records");

  std::set<int64_t> assigned;
  for (MatmulOp op : matmuls) {
    Attribute rawId = op->getAttr(kHmxDecisionIdAttr);
    if (!rawId) {
      op.emitError("hmx.matmul has no explicit hmx.decision_id");
      return failure();
    }
    auto id = dyn_cast<IntegerAttr>(rawId);
    if (!id || !id.getType().isSignlessInteger(64)) {
      op.emitError("hmx.matmul has an invalid hmx.decision_id");
      return failure();
    }
    int64_t value = id.getInt();
    if (!llvm::is_contained(selectedIds, value) ||
        !assigned.insert(value).second) {
      op.emitError("hmx.matmul decision id disagrees with the manifest");
      return failure();
    }
  }
  return success();
}

LogicalResult mlir::hmx::refreshHmxManifestBridgeCounts(ModuleOp module) {
  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true);
  if (failed(current))
    return failure();

  struct Counts {
    int64_t packAct = 0;
    int64_t packWeight = 0;
    int64_t unpack = 0;
  };
  MLIRContext *ctx = module.getContext();
  ArrayAttr matmuls = arrayField(*current, kKeyMatmuls);
  SmallVector<Counts> perRecord(matmuls.size());
  Counts total;
  LogicalResult status = success();

  auto recordIndex = [&](Operation *op) -> std::optional<size_t> {
    if (failed(status))
      return std::nullopt;
    auto function = op->getParentOfType<func::FuncOp>();
    if (!function) {
      op->emitError("HMX bridge operation is not inside a function");
      status = failure();
      return std::nullopt;
    }
    Attribute raw = op->getAttr(kHmxDecisionIdAttr);
    if (!raw) {
      op->emitError("HMX bridge operation has no explicit hmx.decision_id");
      status = failure();
      return std::nullopt;
    }
    auto id = dyn_cast<IntegerAttr>(raw);
    if (!id || !id.getType().isSignlessInteger(64)) {
      op->emitError("HMX bridge operation has an invalid hmx.decision_id");
      status = failure();
      return std::nullopt;
    }
    std::optional<size_t> index =
        findRecord(matmuls, function.getName(), id.getInt());
    if (!index) {
      op->emitError("HMX bridge decision id has no manifest record");
      status = failure();
      return std::nullopt;
    }
    auto record = cast<DictionaryAttr>(matmuls[*index]);
    StringAttr plan = stringField(record, kKeyPlan);
    if (!plan || parsePlan(plan.getValue()) == PlanKind::HVX) {
      op->emitError("HMX bridge decision id refers to a non-HMX record");
      status = failure();
      return std::nullopt;
    }
    return index;
  };
  auto add = [&](Operation *op, Counts &counts, int64_t Counts::*field) {
    if (failed(status))
      return;
    std::optional<size_t> index = recordIndex(op);
    if (failed(status))
      return;
    ++(counts.*field);
    if (index)
      ++(perRecord[*index].*field);
  };
  module.walk(
      [&](PackActOp op) { add(op.getOperation(), total, &Counts::packAct); });
  module.walk([&](PackWeightOp op) {
    add(op.getOperation(), total, &Counts::packWeight);
  });
  module.walk(
      [&](UnpackAccOp op) { add(op.getOperation(), total, &Counts::unpack); });
  module.walk([&](UnpackAccF32Op op) {
    add(op.getOperation(), total, &Counts::unpack);
  });
  if (failed(status))
    return failure();

  SmallVector<Attribute> entries(matmuls.begin(), matmuls.end());
  auto i64 = IntegerType::get(ctx, 64);
  for (auto [index, item] : llvm::enumerate(entries)) {
    auto record = cast<DictionaryAttr>(item);
    StringAttr plan = stringField(record, kKeyPlan);
    if (!plan || parsePlan(plan.getValue()) == PlanKind::HVX)
      continue;
    Counts counts = perRecord[index];
    NamedAttrList fields(record);
    DictionaryAttr execution = dictionaryField(record, kKeyExecution);
    NamedAttrList executionFields(execution ? execution
                                            : DictionaryAttr::get(ctx, {}));
    NamedAttrList bridge;
    bridge.set(kKeyPackActSites, IntegerAttr::get(i64, counts.packAct));
    bridge.set(kKeyPackWeightSites, IntegerAttr::get(i64, counts.packWeight));
    bridge.set(kKeyUnpackSites, IntegerAttr::get(i64, counts.unpack));
    bridge.set(kKeyCountSemantics, StringAttr::get(ctx, kCountSemantics));
    executionFields.set(kKeyBridgeCounts, bridge.getDictionary(ctx));
    fields.set(kKeyExecution, executionFields.getDictionary(ctx));
    fields.erase(kKeyPlanFingerprint);
    entries[index] = fields.getDictionary(ctx);
  }

  NamedAttrList fields(*current);
  fields.set(kKeyMatmuls, ArrayAttr::get(ctx, entries));
  fields.set(kKeyPackActSites, IntegerAttr::get(i64, total.packAct));
  fields.set(kKeyPackWeightSites, IntegerAttr::get(i64, total.packWeight));
  fields.set(kKeyUnpackSites, IntegerAttr::get(i64, total.unpack));
  fields.set(kKeyCountSemantics, StringAttr::get(ctx, kCountSemantics));
  return writeManifest(module, fields);
}

LogicalResult mlir::hmx::finalizeHmxManifest(ModuleOp module) {
  Attribute previous = module->getAttr(kManifestAttr);
  FailureOr<DictionaryAttr> current =
      readManifest(module, /*reportErrors=*/true, /*createIfMissing=*/true,
                   /*requireFinal=*/true, /*checkFingerprints=*/false);
  if (failed(current))
    return failure();

  MLIRContext *ctx = module.getContext();
  ArrayAttr matmuls = arrayField(*current, kKeyMatmuls);
  WeightPolicyMap policies;
  if (failed(validateWeightPolicies(module, *current, /*report=*/true,
                                    /*requireFields=*/true, policies)))
    return failure();
  SmallVector<Attribute> entries(matmuls.begin(), matmuls.end());
  for (Attribute &item : entries) {
    auto record = cast<DictionaryAttr>(item);
    NamedAttrList fields(record);
    fields.set(kKeyPlanFingerprint,
               StringAttr::get(ctx, computePlanFingerprint(record, policies)));
    item = fields.getDictionary(ctx);
  }

  NamedAttrList fields(*current);
  fields.set(kKeyMatmuls, ArrayAttr::get(ctx, entries));
  module->setAttr(kManifestAttr, fields.getDictionary(ctx));
  if (failed(readManifest(module, /*reportErrors=*/true,
                          /*createIfMissing=*/false,
                          /*requireFinal=*/true, /*checkFingerprints=*/true))) {
    if (previous)
      module->setAttr(kManifestAttr, previous);
    else
      module->removeAttr(kManifestAttr);
    return failure();
  }
  return success();
}

std::string mlir::hmx::serializeHmxManifestJson(ModuleOp module) {
  // Serialization is a publication boundary.  Do not expose an intermediate
  // module attribute merely because its attribution records are already valid.
  if (!module->getAttr(kManifestAttr))
    return "{}";

  if (failed(finalizeHmxManifest(module)))
    return "{}";

  FailureOr<DictionaryAttr> manifest =
      readManifest(module, /*reportErrors=*/false, /*createIfMissing=*/false,
                   /*requireFinal=*/true, /*checkFingerprints=*/true);
  if (failed(manifest))
    return "{}";

  json::Object root;
  for (NamedAttribute field : *manifest)
    root[field.getName().strref()] = attrToJson(field.getValue());
  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << json::Value(std::move(root));
  stream.flush();
  return output;
}
