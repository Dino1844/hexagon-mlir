//===-- HmxRecordV3.cpp - record-only HMX kernel record (v3) ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Producer, validator and serializer for `hex.hmx.kernel_manifest/v3`.
//
// Three properties are structural here rather than by convention:
//
//  1. This file never reads `hmx.kernel_manifest`.  The v3 document is built
//     from the same compile-time attribution facts the v2 record builder reads
//     and from the P1.5 diagnostic sidecars.  There is no `v2_to_v3` path, and
//     no field is copied or renamed out of the serialized v2 contract.
//  2. `null` means unknown.  A number is written only where the evidence proves
//     it; "not proven" is never turned into `0`.
//  3. Validation is closed: every object has an exact field set, every enum is
//     enumerated, every unit is pinned, and anything else fails closed.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hmx/Transforms/HmxRecordV3.h"

#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxVtcmAccounting.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
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

namespace {

//===----------------------------------------------------------------------===//
// Wire vocabulary
//===----------------------------------------------------------------------===//

constexpr StringLiteral kKeySchema = "schema";
constexpr StringLiteral kKeyRecordMode = "record_mode";
constexpr StringLiteral kKeyAdmission = "admission";
constexpr StringLiteral kKeyRecords = "records";

constexpr StringLiteral kKeyFunction = "function";
constexpr StringLiteral kKeyId = "id";
constexpr StringLiteral kKeyPlan = "plan";
constexpr StringLiteral kKeyShape = "shape";
constexpr StringLiteral kKeyShapeState = "state";
constexpr StringLiteral kKeyLogical = "logical";
constexpr StringLiteral kKeySpecialization = "specialization";
constexpr StringLiteral kKeyScope = "scope";
constexpr StringLiteral kKeyInvocations = "invocations";
constexpr StringLiteral kKeyGrid = "grid";
constexpr StringLiteral kKeyGridPolicy = "policy";
constexpr StringLiteral kKeyGridRequiredProduct = "required_product";
constexpr StringLiteral kKeyResident = "resident";
constexpr StringLiteral kKeyResources = "resources";
constexpr StringLiteral kKeyRequested = "requested";
constexpr StringLiteral kKeyAllocatorAligned = "allocator_aligned";
constexpr StringLiteral kKeyObservedHighWater = "observed_high_water";
constexpr StringLiteral kKeyUnit = "unit";
constexpr StringLiteral kKeyBasis = "basis";
constexpr StringLiteral kKeyStatus = "status";
constexpr StringLiteral kKeyTransientRequestedPeakBytes =
    "transient_requested_peak_bytes";
constexpr StringLiteral kKeyResidentRequestedBytes = "resident_requested_bytes";
constexpr StringLiteral kKeyModeledRequestedPeakBytes =
    "modeled_requested_peak_bytes";
constexpr StringLiteral kKeyTransientAlignedPeakBytes =
    "transient_aligned_peak_bytes";
constexpr StringLiteral kKeyResidentAlignedBytes = "resident_aligned_bytes";
constexpr StringLiteral kKeyModeledAlignedPeakBytes =
    "modeled_aligned_peak_bytes";
constexpr StringLiteral kKeyValueBytes = "value_bytes";
constexpr StringLiteral kKeySource = "source";
constexpr StringLiteral kKeyProofs = "proofs";
constexpr StringLiteral kKeyLiveness = "liveness";
constexpr StringLiteral kKeyAllocator = "allocator";
constexpr StringLiteral kKeyGridProof = "grid";
constexpr StringLiteral kKeyResidentProof = "resident";
constexpr StringLiteral kKeyFallback = "fallback";
constexpr StringLiteral kKeyOnMalformedRecord = "on_malformed_record";
constexpr StringLiteral kKeyRecordFingerprint = "record_fingerprint";
constexpr StringLiteral kFingerprintPrefix = "sha256:";

constexpr StringLiteral kAxisM = "m";
constexpr StringLiteral kAxisN = "n";
constexpr StringLiteral kAxisK = "k";
constexpr StringLiteral kAxisValue = "value";
constexpr StringLiteral kAxisSymbol = "symbol";
constexpr StringLiteral kAxisKind = "kind";

// The plan vocabulary is the shared wire spelling from HmxManifest.h, used here
// for correlation only: a v3 record publishes which plan the current compile
// produced, it does not select one, and it grants no execution authority.
// `hmx-tail` in particular remains a diagnostic/test slice.
constexpr StringLiteral kPlanFullHMX = kHmxPlanFullHMX;
constexpr StringLiteral kPlanHMXTail = kHmxPlanHMXTail;
constexpr StringLiteral kPlanHVX = kHmxPlanHVX;

constexpr StringLiteral kShapeStateStatic = kHmxShapeStateStatic;
constexpr StringLiteral kShapeStatePartiallyDynamic =
    kHmxShapeStatePartiallyDynamic;
constexpr StringLiteral kShapeStateDynamic = kHmxShapeStateDynamic;

/// `upstream-static` means the specialization happened before the object was
/// built; `upstream-only` means it did not and the record keeps the dynamic
/// axes.  v3 introduces no runtime dispatcher of its own.
constexpr StringLiteral kSpecializationUpstreamStatic = "upstream-static";
constexpr StringLiteral kSpecializationUpstreamOnly = "upstream-only";

// A dimension `kind` shares the static/dynamic tokens with shape_state and must
// agree with it, so both names resolve to the same wire vocabulary.
constexpr StringLiteral kKindStatic = kHmxShapeStateStatic;
constexpr StringLiteral kKindDynamic = kHmxShapeStateDynamic;

/// Versioned mapping from the P1.5 structured liveness sidecar onto the v3
/// `liveness` proof.  The basis names the sidecar fact that was consumed, so a
/// reader can tell a proven axis from an assumed one.  The other three axes have
/// no accepted evidence source yet and stay `not-proven` unless a future sidecar
/// publishes an explicitly versioned key for them.
constexpr StringLiteral kLivenessBasisStructuredRequestedUpperBound =
    "structured-requested-upper-bound";

/// Sidecar facts this mapping is allowed to read.  Anything else is refused.
constexpr StringLiteral kSidecarUnitRequestedBytes = "requested-bytes";
constexpr StringLiteral kSidecarScopePerFunctionSingleInvocation =
    "per-function-single-invocation";
constexpr StringLiteral kSidecarPeakStatusStructuredUpperBound =
    "structured-upper-bound";
/// The sidecar that publishes the three module-level axis statuses.  Naming it
/// is what distinguishes "this sidecar says complete" from "nobody said".
constexpr StringLiteral kSidecarLivenessSchema = "structured-allocator-events-v1";

// NOT-A-DECISION: the int64 saturation bound used by addBytes' overflow
// check. It is a property of the type, not a tunable: any other value
// would be a bug rather than a policy.
constexpr int64_t kMaxI64 = std::numeric_limits<int64_t>::max();

bool isCanonicalPlan(StringRef value) {
  return value == kPlanFullHMX || value == kPlanHMXTail || value == kPlanHVX;
}

bool isCanonicalShapeState(StringRef value) {
  return value == kShapeStateStatic || value == kShapeStatePartiallyDynamic ||
         value == kShapeStateDynamic;
}

bool isCanonicalProofStatus(StringRef value) {
  return value == kHmxProofStatusComplete || value == kHmxProofStatusIncomplete ||
         value == kHmxProofStatusNotProven;
}

//===----------------------------------------------------------------------===//
// Small typed readers
//===----------------------------------------------------------------------===//

bool isI64(Attribute attr) {
  auto integer = dyn_cast_or_null<IntegerAttr>(attr);
  return integer && integer.getType().isSignlessInteger(64);
}

/// A missing field and an explicit `null` are the same wire value here: both
/// mean "no evidence".  They are also both written, so a consumer never has to
/// distinguish an absent key from a null one.
bool isNone(Attribute attr) { return !attr || isa<UnitAttr>(attr); }

bool isNonEmptyString(StringRef value) { return !value.trim().empty(); }

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

std::optional<int64_t> readNonnegativeI64(DictionaryAttr dict, StringRef name) {
  if (!isI64(dict.get(name)))
    return std::nullopt;
  int64_t value = cast<IntegerAttr>(dict.get(name)).getInt();
  if (value < 0)
    return std::nullopt;
  return value;
}

/// Exact-field rejection.  Unlike a tolerant reader this never invents a default
/// and never ignores an extra key: a v3 record with an unknown field is a
/// producer bug or a forged document, and both fail closed.
LogicalResult rejectUnknownFields(DictionaryAttr dict,
                                  ArrayRef<StringRef> allowed) {
  for (NamedAttribute field : dict)
    if (!llvm::is_contained(allowed, field.getName().strref()))
      return failure();
  return success();
}

/// A capacity quantity: either a proven non-negative count or unknown.  `0` is
/// a real observation and stays `0`.
bool isByteQuantity(Attribute attr) {
  if (isNone(attr))
    return true;
  return isI64(attr) && cast<IntegerAttr>(attr).getInt() >= 0;
}

bool addBytes(int64_t &total, int64_t value) {
  if (value < 0 || total > kMaxI64 - value)
    return false;
  total += value;
  return true;
}

//===----------------------------------------------------------------------===//
// Canonical JSON (deliberately not shared with the v2 fingerprint)
//===----------------------------------------------------------------------===//
//
// The v2 digest is frozen: refactoring its canonicalizer would silently change
// every published v2 fingerprint.  The v3 digest therefore owns a second
// implementation rather than sharing one, so no future change to either
// canonicalizer can move the other's bytes.  The two encodings are deliberately
// identical for JSON-compatible attributes, which is what lets the Python
// consumer recompute the digest.

void appendCanonicalJson(llvm::raw_ostream &stream, Attribute attr) {
  if (isNone(attr)) {
    stream << "null";
    return;
  }
  if (auto integer = dyn_cast<IntegerAttr>(attr)) {
    stream << integer.getInt();
    return;
  }
  if (auto boolean = dyn_cast<BoolAttr>(attr))
    return void(stream << (boolean.getValue() ? "true" : "false"));
  if (auto string = dyn_cast<StringAttr>(attr))
    return void(stream << json::Value(string.getValue()));
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    stream << '[';
    for (auto [index, item] : llvm::enumerate(array)) {
      if (index)
        stream << ',';
      appendCanonicalJson(stream, item);
    }
    return void(stream << ']');
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
      stream << json::Value(field.getName().strref()) << ':';
      appendCanonicalJson(stream, field.getValue());
    }
    return void(stream << '}');
  }
  // Validation rejects any non-JSON attribute before a digest is taken; keep a
  // stable representation if a future producer adds one.
  stream << "null";
}

std::string computeRecordFingerprint(DictionaryAttr record) {
  MLIRContext *ctx = record.getContext();
  NamedAttrList payload;
  // The schema is part of the digest, so a record can never be replayed under a
  // different wire name.
  payload.set(kKeySchema, StringAttr::get(ctx, kHmxRecordV3Schema));
  for (NamedAttribute field : record) {
    if (field.getName() == kKeyRecordFingerprint)
      continue;
    payload.set(field.getName(), field.getValue());
  }
  std::string canonical;
  llvm::raw_string_ostream stream(canonical);
  appendCanonicalJson(stream, payload.getDictionary(ctx));
  stream.flush();
  llvm::SHA256 hasher;
  hasher.update(canonical);
  std::array<uint8_t, 32> digest = hasher.final();
  return (Twine(kFingerprintPrefix) +
          llvm::toHex(llvm::ArrayRef<uint8_t>(digest.data(), digest.size()), true))
      .str();
}

bool isFingerprintText(StringRef value) {
  if (!value.consume_front(kFingerprintPrefix) || value.size() != 64)
    return false;
  for (char c : value)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Producers
//===----------------------------------------------------------------------===//

DictionaryAttr makeAxis(MLIRContext *ctx, int64_t extent, StringRef symbol) {
  NamedAttrList axis;
  if (extent > 0) {
    axis.append(kAxisKind, StringAttr::get(ctx, kKindStatic));
    axis.append(kAxisValue,
                IntegerAttr::get(IntegerType::get(ctx, 64), extent));
  } else {
    axis.append(kAxisKind, StringAttr::get(ctx, kKindDynamic));
    axis.append(kAxisSymbol, StringAttr::get(ctx, symbol));
  }
  return axis.getDictionary(ctx);
}

/// Unknown-capacity block.  `unit` and `basis` are fixed per block, the status
/// is `not-proven`, and every quantity is `null`.  A later phase may only
/// replace these with a *proven* value; it may never fill a default.
DictionaryAttr makeUnknownResourceBlock(MLIRContext *ctx, StringRef unit,
                                        StringRef basis,
                                        ArrayRef<StringRef> quantityNames) {
  NamedAttrList block;
  block.append(kKeyUnit, StringAttr::get(ctx, unit));
  block.append(kKeyBasis, StringAttr::get(ctx, basis));
  block.append(kKeyStatus, StringAttr::get(ctx, kHmxProofStatusNotProven));
  for (StringRef name : quantityNames)
    block.append(name, UnitAttr::get(ctx));
  return block.getDictionary(ctx);
}

DictionaryAttr makeProof(MLIRContext *ctx, StringRef status, StringRef basis) {
  NamedAttrList proof;
  proof.append(kKeyStatus, StringAttr::get(ctx, status));
  proof.append(kKeyBasis, basis.empty()
                              ? Attribute(UnitAttr::get(ctx))
                              : Attribute(StringAttr::get(ctx, basis)));
  return proof.getDictionary(ctx);
}

DictionaryAttr makeUnknownProof(MLIRContext *ctx) {
  return makeProof(ctx, kHmxProofStatusNotProven, StringRef());
}

DictionaryAttr makeProofs(MLIRContext *ctx) {
  NamedAttrList proofs;
  proofs.append(kKeyLiveness, makeUnknownProof(ctx));
  proofs.append(kKeyAllocator, makeUnknownProof(ctx));
  proofs.append(kKeyGridProof, makeUnknownProof(ctx));
  proofs.append(kKeyResidentProof, makeUnknownProof(ctx));
  return proofs.getDictionary(ctx);
}

DictionaryAttr makeResources(MLIRContext *ctx) {
  NamedAttrList resources;
  resources.append(kKeyRequested,
                   makeUnknownResourceBlock(
                       ctx, kHmxUnitBytes, kHmxBasisCompileTimeRequested,
                       {kKeyTransientRequestedPeakBytes,
                        kKeyResidentRequestedBytes,
                        kKeyModeledRequestedPeakBytes}));
  resources.append(kKeyAllocatorAligned,
                   makeUnknownResourceBlock(
                       ctx, kHmxUnitBytes, kHmxBasisAllocatorModel,
                       {kKeyTransientAlignedPeakBytes,
                        kKeyResidentAlignedBytes, kKeyModeledAlignedPeakBytes}));
  // The observation block always names its scope and leaves both the value and
  // the source unknown.  The only captured evidence is a process aggregate, and
  // promoting it to a per-record value is exactly what this schema forbids.
  NamedAttrList observed;
  observed.append(kKeyUnit, StringAttr::get(ctx, kHmxUnitBytes));
  observed.append(kKeyBasis, StringAttr::get(ctx, kHmxBasisRuntimeObservation));
  observed.append(kKeyStatus, StringAttr::get(ctx, kHmxProofStatusNotProven));
  observed.append(kKeyValueBytes, UnitAttr::get(ctx));
  observed.append(kKeyScope,
                  StringAttr::get(ctx, kHmxObservationScopeProcessHighWater));
  observed.append(kKeySource, UnitAttr::get(ctx));
  resources.append(kKeyObservedHighWater, observed.getDictionary(ctx));
  return resources.getDictionary(ctx);
}

DictionaryAttr makeScope(MLIRContext *ctx, StringRef function) {
  NamedAttrList grid;
  grid.append(kKeyGridPolicy,
              StringAttr::get(ctx, kHmxGridPolicySingleInstance));
  grid.append(kKeyGridRequiredProduct,
              IntegerAttr::get(IntegerType::get(ctx, 64),
                               kHmxGridRequiredProduct));
  NamedAttrList scope;
  scope.append(kKeyFunction, StringAttr::get(ctx, function));
  scope.append(kKeyInvocations,
               IntegerAttr::get(IntegerType::get(ctx, 64), kHmxScopeInvocations));
  scope.append(kKeyGrid, grid.getDictionary(ctx));
  scope.append(kKeyResident,
               StringAttr::get(ctx, kHmxResidentScopeProcessFloor));
  return scope.getDictionary(ctx);
}

DictionaryAttr makeFallback(MLIRContext *ctx) {
  NamedAttrList fallback;
  fallback.append(kKeyOnMalformedRecord,
                  StringAttr::get(ctx, kHmxFallbackRejectV3Record));
  return fallback.getDictionary(ctx);
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

LogicalResult validateAxis(DictionaryAttr axis, StringRef name) {
  if (!axis)
    return failure();
  if (failed(rejectUnknownFields(
          axis, {StringRef(kAxisKind), StringRef(kAxisValue),
                 StringRef(kAxisSymbol)})))
    return failure();
  StringAttr kind = stringField(axis, kAxisKind);
  if (!kind)
    return failure();
  if (kind.getValue() == kKindStatic) {
    if (axis.get(kAxisSymbol))
      return failure();
    // A static extent is a proven compile-time fact, so zero is not one of
    // them: a zero extent is a malformed record, not an empty matrix.
    std::optional<int64_t> value = readNonnegativeI64(axis, kAxisValue);
    return value && *value >= 1 ? success() : failure();
  }
  if (kind.getValue() == kKindDynamic) {
    // A dynamic axis names its own axis and carries no runtime guess: no value,
    // no pointer, and no extent sampled from a grid, a stride or an address.
    if (axis.get(kAxisValue))
      return failure();
    StringAttr symbol = stringField(axis, kAxisSymbol);
    return symbol && symbol.getValue() == name ? success() : failure();
  }
  return failure();
}

/// The declared state must be the unique derivation of the three tagged axes; a
/// producer cannot claim `static` while a dynamic axis is present, and the
/// specialization policy follows from the same three axes.
LogicalResult validateShape(DictionaryAttr shape) {
  if (failed(rejectUnknownFields(shape,
                                 {StringRef(kKeyShapeState),
                                  StringRef(kKeyLogical),
                                  StringRef(kKeySpecialization)})))
    return failure();
  StringAttr state = stringField(shape, kKeyShapeState);
  StringAttr specialization = stringField(shape, kKeySpecialization);
  if (!state || !isCanonicalShapeState(state.getValue()) || !specialization)
    return failure();
  DictionaryAttr logical = dictionaryField(shape, kKeyLogical);
  if (!logical ||
      failed(rejectUnknownFields(logical,
                                 {StringRef(kAxisM), StringRef(kAxisN),
                                  StringRef(kAxisK)})))
    return failure();
  unsigned staticCount = 0;
  for (StringRef axis : {kAxisM, kAxisN, kAxisK}) {
    DictionaryAttr entry = dictionaryField(logical, axis);
    if (failed(validateAxis(entry, axis)))
      return failure();
    if (stringField(entry, kAxisKind).getValue() == kKindStatic)
      ++staticCount;
  }
  StringRef derived = staticCount == 3   ? kShapeStateStatic
                      : staticCount == 0 ? kShapeStateDynamic
                                         : kShapeStatePartiallyDynamic;
  StringRef expectedSpecialization =
      staticCount == 3 ? kSpecializationUpstreamStatic
                       : kSpecializationUpstreamOnly;
  return state.getValue() == derived &&
                 specialization.getValue() == expectedSpecialization
             ? success()
             : failure();
}

LogicalResult validateScope(DictionaryAttr scope, StringRef function) {
  if (failed(rejectUnknownFields(scope,
                                 {StringRef(kKeyFunction),
                                  StringRef(kKeyInvocations), StringRef(kKeyGrid),
                                  StringRef(kKeyResident)})))
    return failure();
  StringAttr owner = stringField(scope, kKeyFunction);
  if (!owner || !isNonEmptyString(owner.getValue()) ||
      owner.getValue() != function)
    return failure();
  std::optional<int64_t> invocations =
      readNonnegativeI64(scope, kKeyInvocations);
  if (!invocations || *invocations != kHmxScopeInvocations)
    return failure();
  DictionaryAttr grid = dictionaryField(scope, kKeyGrid);
  if (!grid ||
      failed(rejectUnknownFields(grid,
                                 {StringRef(kKeyGridPolicy),
                                  StringRef(kKeyGridRequiredProduct)})))
    return failure();
  StringAttr policy = stringField(grid, kKeyGridPolicy);
  if (!policy || policy.getValue() != kHmxGridPolicySingleInstance)
    return failure();
  std::optional<int64_t> product =
      readNonnegativeI64(grid, kKeyGridRequiredProduct);
  if (!product || *product != kHmxGridRequiredProduct)
    return failure();
  StringAttr resident = stringField(scope, kKeyResident);
  return resident && resident.getValue() == kHmxResidentScopeProcessFloor
             ? success()
             : failure();
}

/// One capacity block.  The unit and basis are pinned per block, every quantity
/// is either a proven non-negative count or `null`, and a `not-proven` block may
/// not carry a number at all.  That last rule is the machine form of "never turn
/// not-proven into 0".
LogicalResult validateResourceBlock(DictionaryAttr block, StringRef unit,
                                    StringRef basis,
                                    ArrayRef<StringRef> quantityNames) {
  if (!block)
    return failure();
  SmallVector<StringRef> allowed = {StringRef(kKeyUnit), StringRef(kKeyBasis),
                                    StringRef(kKeyStatus)};
  for (StringRef quantity : quantityNames)
    allowed.push_back(quantity);
  if (failed(rejectUnknownFields(block, allowed)))
    return failure();
  StringAttr actualUnit = stringField(block, kKeyUnit);
  StringAttr actualBasis = stringField(block, kKeyBasis);
  StringAttr status = stringField(block, kKeyStatus);
  if (!actualUnit || actualUnit.getValue() != unit)
    return failure();
  if (!actualBasis || actualBasis.getValue() != basis)
    return failure();
  if (!status || !isCanonicalProofStatus(status.getValue()))
    return failure();
  bool unknownOnly = status.getValue() == kHmxProofStatusNotProven;
  for (StringRef quantity : quantityNames) {
    if (!isByteQuantity(block.get(quantity)))
      return failure();
    if (unknownOnly && !isNone(block.get(quantity)))
      return failure();
  }
  return success();
}

LogicalResult validateResources(DictionaryAttr resources) {
  if (failed(rejectUnknownFields(resources,
                                 {StringRef(kKeyRequested),
                                  StringRef(kKeyAllocatorAligned),
                                  StringRef(kKeyObservedHighWater)})))
    return failure();
  if (failed(validateResourceBlock(
          dictionaryField(resources, kKeyRequested), kHmxUnitBytes,
          kHmxBasisCompileTimeRequested,
          {StringRef(kKeyTransientRequestedPeakBytes),
           StringRef(kKeyResidentRequestedBytes),
           StringRef(kKeyModeledRequestedPeakBytes)})))
    return failure();
  if (failed(validateResourceBlock(
          dictionaryField(resources, kKeyAllocatorAligned), kHmxUnitBytes,
          kHmxBasisAllocatorModel,
          {StringRef(kKeyTransientAlignedPeakBytes),
           StringRef(kKeyResidentAlignedBytes),
           StringRef(kKeyModeledAlignedPeakBytes)})))
    return failure();

  DictionaryAttr observed = dictionaryField(resources, kKeyObservedHighWater);
  if (!observed ||
      failed(rejectUnknownFields(observed,
                                 {StringRef(kKeyUnit), StringRef(kKeyBasis),
                                  StringRef(kKeyStatus),
                                  StringRef(kKeyValueBytes),
                                  StringRef(kKeyScope), StringRef(kKeySource)})))
    return failure();
  StringAttr unit = stringField(observed, kKeyUnit);
  StringAttr basis = stringField(observed, kKeyBasis);
  StringAttr status = stringField(observed, kKeyStatus);
  StringAttr scope = stringField(observed, kKeyScope);
  if (!unit || unit.getValue() != kHmxUnitBytes)
    return failure();
  if (!basis || basis.getValue() != kHmxBasisRuntimeObservation)
    return failure();
  if (!status || !isCanonicalProofStatus(status.getValue()))
    return failure();
  // The only permitted observation scope.  A per-record high-water mark has no
  // accepted join key yet, so naming a narrower scope would be a claim the
  // schema cannot support.
  if (!scope || scope.getValue() != kHmxObservationScopeProcessHighWater)
    return failure();
  if (!isByteQuantity(observed.get(kKeyValueBytes)))
    return failure();
  if (status.getValue() == kHmxProofStatusNotProven &&
      !isNone(observed.get(kKeyValueBytes)))
    return failure();
  // The source is what makes an observation attributable.  A process aggregate
  // has no per-record source, so the field stays unknown.
  Attribute source = observed.get(kKeySource);
  if (!isNone(source) &&
      !isNonEmptyString(dyn_cast_or_null<StringAttr>(source).getValue()))
    return failure();
  return success();
}

LogicalResult validateProof(DictionaryAttr proof) {
  if (!proof || failed(rejectUnknownFields(
                       proof, {StringRef(kKeyStatus), StringRef(kKeyBasis)})))
    return failure();
  StringAttr status = stringField(proof, kKeyStatus);
  if (!status || !isCanonicalProofStatus(status.getValue()))
    return failure();
  Attribute basis = proof.get(kKeyBasis);
  if (isNone(basis))
    return status.getValue() == kHmxProofStatusComplete ? failure() : success();
  StringAttr basisText = dyn_cast_or_null<StringAttr>(basis);
  if (!basisText || !isNonEmptyString(basisText.getValue()))
    return failure();
  return success();
}

LogicalResult validateProofs(DictionaryAttr proofs) {
  if (failed(rejectUnknownFields(proofs,
                                 {StringRef(kKeyLiveness),
                                  StringRef(kKeyAllocator),
                                  StringRef(kKeyGridProof),
                                  StringRef(kKeyResidentProof)})))
    return failure();
  for (StringRef axis : {kKeyLiveness, kKeyAllocator, kKeyGridProof,
                         kKeyResidentProof})
    if (failed(validateProof(dictionaryField(proofs, axis))))
      return failure();
  return success();
}

LogicalResult validateFallback(DictionaryAttr fallback) {
  if (failed(rejectUnknownFields(
          fallback, {StringRef(kKeyOnMalformedRecord)})))
    return failure();
  StringAttr malformed = stringField(fallback, kKeyOnMalformedRecord);
  return malformed &&
                 malformed.getValue() == kHmxFallbackRejectV3Record
             ? success()
             : failure();
}

/// A published record.  Every quantity is checked, the shape is re-derived, and
/// the fingerprint is recomputed: a record whose digest disagrees with its
/// content is rejected instead of being trusted as an identity.
LogicalResult validatePublishedRecord(DictionaryAttr record) {
  if (failed(rejectUnknownFields(record,
                                 {StringRef(kKeyFunction), StringRef(kKeyId),
                                  StringRef(kKeyPlan), StringRef(kKeyShape),
                                  StringRef(kKeyScope), StringRef(kKeyResources),
                                  StringRef(kKeyProofs), StringRef(kKeyFallback),
                                  StringRef(kKeyRecordFingerprint)})))
    return failure();
  StringAttr function = stringField(record, kKeyFunction);
  if (!function || !isNonEmptyString(function.getValue()))
    return failure();
  std::optional<int64_t> id = readNonnegativeI64(record, kKeyId);
  if (!id)
    return failure();
  StringAttr plan = stringField(record, kKeyPlan);
  if (!plan || !isCanonicalPlan(plan.getValue()))
    return failure();
  if (failed(validateShape(dictionaryField(record, kKeyShape))) ||
      failed(validateScope(dictionaryField(record, kKeyScope),
                           function.getValue())) ||
      failed(validateResources(dictionaryField(record, kKeyResources))) ||
      failed(validateProofs(dictionaryField(record, kKeyProofs))) ||
      failed(validateFallback(dictionaryField(record, kKeyFallback))))
    return failure();
  StringAttr fingerprint = stringField(record, kKeyRecordFingerprint);
  if (!fingerprint || !isFingerprintText(fingerprint.getValue()))
    return failure();
  return fingerprint.getValue() == computeRecordFingerprint(record) ? success()
                                                                    : failure();
}

/// A record skeleton is the same contract minus the digest, which is computed
/// only after the P1.5 sidecars have been folded in.
LogicalResult validateRecordSkeleton(DictionaryAttr record) {
  if (failed(rejectUnknownFields(record,
                                 {StringRef(kKeyFunction), StringRef(kKeyId),
                                  StringRef(kKeyPlan), StringRef(kKeyShape),
                                  StringRef(kKeyScope), StringRef(kKeyResources),
                                  StringRef(kKeyProofs),
                                  StringRef(kKeyFallback)})))
    return failure();
  StringAttr function = stringField(record, kKeyFunction);
  if (!function || !isNonEmptyString(function.getValue()))
    return failure();
  if (!readNonnegativeI64(record, kKeyId))
    return failure();
  StringAttr plan = stringField(record, kKeyPlan);
  if (!plan || !isCanonicalPlan(plan.getValue()))
    return failure();
  if (failed(validateShape(dictionaryField(record, kKeyShape))) ||
      failed(validateScope(dictionaryField(record, kKeyScope),
                           function.getValue())) ||
      failed(validateResources(dictionaryField(record, kKeyResources))) ||
      failed(validateProofs(dictionaryField(record, kKeyProofs))) ||
      failed(validateFallback(dictionaryField(record, kKeyFallback))))
    return failure();
  return success();
}

/// The published document: a closed top-level shape with unique function-local
/// record ids.
LogicalResult validateDocument(DictionaryAttr document) {
  if (failed(rejectUnknownFields(document,
                                 {StringRef(kKeySchema),
                                  StringRef(kKeyRecordMode),
                                  StringRef(kKeyAdmission),
                                  StringRef(kKeyRecords)})))
    return failure();
  StringAttr schema = stringField(document, kKeySchema);
  StringAttr mode = stringField(document, kKeyRecordMode);
  StringAttr admission = stringField(document, kKeyAdmission);
  if (!schema || schema.getValue() != kHmxRecordV3Schema)
    return failure();
  if (!mode || mode.getValue() != kHmxRecordModeRecordOnly)
    return failure();
  if (!admission || admission.getValue() != kHmxRecordAdmissionNotAuthorized)
    return failure();
  ArrayAttr records = arrayField(document, kKeyRecords);
  if (!records)
    return failure();
  std::set<std::pair<std::string, int64_t>> seen;
  for (Attribute item : records) {
    auto record = dyn_cast<DictionaryAttr>(item);
    if (!record || failed(validatePublishedRecord(record)))
      return failure();
    if (!seen.emplace(stringField(record, kKeyFunction).getValue().str(),
                      *readNonnegativeI64(record, kKeyId))
             .second)
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// P1.5 sidecar enrichment
//===----------------------------------------------------------------------===//

/// The closed reading of one P1.5 per-function liveness fact.
///
/// Only the requested-byte triple is carried, because the join below accepts a
/// function *only* when its structured liveness status is `complete`; the
/// liveness axis is therefore `complete` by construction here and is not stored
/// as a second, always-equal copy of that fact.  The three module-level axes
/// are read only once the join has succeeded, so nothing is read that the
/// record will not publish.
///
/// The *absence* of evidence is represented by the `std::nullopt` the join
/// returns, never by a default-constructed value: a zeroed struct would publish
/// `complete` with three zero bytes, which is precisely the "not-proven became
/// 0" failure this schema exists to prevent.
struct LivenessEvidence {
  int64_t transientPeak = 0;
  int64_t residentRequested = 0;
  int64_t modeledPeak = 0;
  StringRef allocatorStatus = kHmxProofStatusNotProven;
  StringRef gridStatus = kHmxProofStatusNotProven;
  StringRef residentStatus = kHmxProofStatusNotProven;
};

StringRef readStatusOrNotProven(DictionaryAttr dict, StringRef name) {
  StringAttr value = stringField(dict, name);
  if (!value || !isCanonicalProofStatus(value.getValue()))
    return kHmxProofStatusNotProven;
  return value.getValue();
}

/// The basis an axis read from a sidecar carries.  An unproven axis names the
/// sidecar it was *not* proven by, which is to say nothing: `null` is the only
/// honest basis for a gap.
StringRef sidecarBasis(StringRef status) {
  return status == kHmxProofStatusComplete ? StringRef(kSidecarLivenessSchema)
                                          : StringRef();
}

/// Join the P1.5 sidecars to one function.
///
/// The join is deliberately narrow, and every requirement is checked before any
/// fact is read: an exact symbol match, a `complete` structured liveness status
/// with a `structured-upper-bound` peak, and a consistent
/// `modeled = transient + resident` identity.  A function that misses any of them
/// yields a zero-valued `LivenessEvidence`, which `finalizeHmxRecordV3` treats as
/// "no evidence" and leaves the record `not-proven`.  Both census markers must
/// therefore be present for a record to carry a requested-byte number: the
/// accounting sidecar proves the census is complete, the liveness sidecar
/// supplies the per-function bound, and neither substitutes for the other.
std::optional<LivenessEvidence> readFunctionEvidence(ModuleOp module,
                                                     StringRef function) {
  LivenessEvidence evidence;
  auto accounting =
      dyn_cast_or_null<DictionaryAttr>(module->getAttr(kHmxVtcmAccountingAttr));
  auto liveness =
      dyn_cast_or_null<DictionaryAttr>(module->getAttr(kHmxVtcmLivenessAttr));
  if (!accounting || !liveness)
    return std::nullopt;

  // The sidecar must still describe the unit and scope this schema publishes.  A
  // unit or scope change is a new sidecar schema, not a v3 fact.
  StringAttr unit = stringField(liveness, "unit");
  StringAttr scope = stringField(liveness, "scope");
  if (!unit || unit.getValue() != kSidecarUnitRequestedBytes)
    return std::nullopt;
  if (!scope || scope.getValue() != kSidecarScopePerFunctionSingleInvocation)
    return std::nullopt;
  if (readStatusOrNotProven(accounting, "status") != kHmxProofStatusComplete)
    return std::nullopt;

  ArrayAttr functions = arrayField(liveness, "functions");
  if (!functions)
    return std::nullopt;
  DictionaryAttr match;
  for (Attribute item : functions) {
    auto entry = dyn_cast<DictionaryAttr>(item);
    if (!entry)
      return std::nullopt;
    StringAttr symbol = stringField(entry, "symbol");
    if (!symbol)
      return std::nullopt;
    if (symbol.getValue() == function) {
      // Two entries for one symbol would make the join ambiguous; refuse it
      // rather than picking one.
      if (match)
        return std::nullopt;
      match = entry;
    }
  }
  if (!match)
    return std::nullopt;

  if (readStatusOrNotProven(match, "status") != kHmxProofStatusComplete)
    return std::nullopt;
  StringAttr peakStatus = stringField(match, "peak_status");
  if (!peakStatus || peakStatus.getValue() != kSidecarPeakStatusStructuredUpperBound)
    return std::nullopt;

  std::optional<int64_t> transient =
      readNonnegativeI64(match, "transient_requested_peak_bytes");
  std::optional<int64_t> workspace =
      readNonnegativeI64(match, "workspace_resident_requested_bytes");
  std::optional<int64_t> weight =
      readNonnegativeI64(match, "weight_resident_requested_bytes");
  std::optional<int64_t> modeled =
      readNonnegativeI64(match, "modeled_requested_peak_bytes");
  if (!transient || !workspace || !weight || !modeled)
    return std::nullopt;

  // The process floor is a separate layer from the transient peak.  Keep them
  // separate in the record and only require that they still add up to the model
  // bound the sidecar published.
  int64_t resident = 0;
  if (!addBytes(resident, *workspace) || !addBytes(resident, *weight))
    return std::nullopt;
  if (*modeled != *transient + resident)
    return std::nullopt;

  // The three module-level axes are read here, after the join succeeded, at the
  // sidecar's own published values.  They are `not-proven` today; a sidecar that
  // proves one of them flows through this mapping without a schema change, and
  // no axis may promote another.
  evidence.allocatorStatus =
      readStatusOrNotProven(liveness, "allocator_peak_status");
  evidence.gridStatus = readStatusOrNotProven(liveness, "grid_status");
  evidence.residentStatus =
      readStatusOrNotProven(liveness, "resident_runtime_state");
  evidence.transientPeak = *transient;
  evidence.residentRequested = resident;
  evidence.modeledPeak = *modeled;
  return evidence;
}

/// Publish the proven requested-byte facts onto one record.  Only `requested` and
/// `liveness` move: `allocator_aligned` and `observed_high_water` have no
/// accepted evidence source and stay `not-proven` with `null` values.
void applyEvidence(MLIRContext *ctx, NamedAttrList &record,
                   const std::optional<LivenessEvidence> &evidence) {
  if (!evidence)
    return;
  const LivenessEvidence &proven = *evidence;
  auto i64 = IntegerType::get(ctx, 64);
  NamedAttrList resources(
      dictionaryField(record.getDictionary(ctx), kKeyResources));
  NamedAttrList requested(
      dictionaryField(resources.getDictionary(ctx), kKeyRequested));
  requested.set(kKeyStatus, StringAttr::get(ctx, kHmxProofStatusComplete));
  requested.set(kKeyTransientRequestedPeakBytes,
                IntegerAttr::get(i64, proven.transientPeak));
  requested.set(kKeyResidentRequestedBytes,
                IntegerAttr::get(i64, proven.residentRequested));
  requested.set(kKeyModeledRequestedPeakBytes,
                IntegerAttr::get(i64, proven.modeledPeak));
  resources.set(kKeyRequested, requested.getDictionary(ctx));
  record.set(kKeyResources, resources.getDictionary(ctx));

  // `liveness` names the specific fact it consumed; the three module-level axes
  // name the sidecar that published their status, and carry no basis while they
  // are unproven.  A `complete` axis without a basis is exactly the "masks a
  // gap" case the four-axis contract exists to prevent.
  NamedAttrList proofs(
      dictionaryField(record.getDictionary(ctx), kKeyProofs));
  proofs.set(kKeyLiveness, makeProof(ctx, kHmxProofStatusComplete,
                                     kLivenessBasisStructuredRequestedUpperBound));
  proofs.set(kKeyAllocator, makeProof(ctx, proven.allocatorStatus,
                                     sidecarBasis(proven.allocatorStatus)));
  proofs.set(kKeyGridProof,
             makeProof(ctx, proven.gridStatus,
                       sidecarBasis(proven.gridStatus)));
  proofs.set(kKeyResidentProof,
             makeProof(ctx, proven.residentStatus,
                       sidecarBasis(proven.residentStatus)));
  record.set(kKeyProofs, proofs.getDictionary(ctx));
}

DictionaryAttr makeDocument(MLIRContext *ctx, ArrayRef<Attribute> records) {
  NamedAttrList document;
  document.append(kKeySchema, StringAttr::get(ctx, kHmxRecordV3Schema));
  document.append(kKeyRecordMode,
                  StringAttr::get(ctx, kHmxRecordModeRecordOnly));
  document.append(kKeyAdmission,
                  StringAttr::get(ctx, kHmxRecordAdmissionNotAuthorized));
  document.append(kKeyRecords, ArrayAttr::get(ctx, records));
  return document.getDictionary(ctx);
}

json::Value attrToJson(Attribute attr) {
  if (isNone(attr))
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

//===----------------------------------------------------------------------===//
// Public producer API
//===----------------------------------------------------------------------===//

Attribute mlir::hmx::buildHmxRecordV3Skeleton(MLIRContext *context,
                                              StringRef function, int64_t id,
                                              StringRef plan,
                                              ArrayRef<int64_t> dims) {
  // `dims` holds the logical extents: positive is a proven static extent,
  // negative is dynamic.  A zero extent is neither, and it must be refused here
  // rather than fall through to the `> 0` count below -- counting a zero as
  // dynamic would publish "unknown" for a fact the compile inputs do carry.
  if (dims.size() != 3 || id < 0 || !isNonEmptyString(function) ||
      !isCanonicalPlan(plan))
    return {};
  for (int64_t extent : dims)
    if (extent == 0)
      return {};
  unsigned staticCount = 0;
  for (int64_t extent : dims)
    staticCount += extent > 0;
  StringRef state = staticCount == 3   ? kShapeStateStatic
                    : staticCount == 0 ? kShapeStateDynamic
                                       : kShapeStatePartiallyDynamic;
  StringRef specialization = staticCount == 3 ? kSpecializationUpstreamStatic
                                              : kSpecializationUpstreamOnly;

  NamedAttrList logical;
  logical.append(kAxisM, makeAxis(context, dims[0], kAxisM));
  logical.append(kAxisN, makeAxis(context, dims[1], kAxisN));
  logical.append(kAxisK, makeAxis(context, dims[2], kAxisK));
  NamedAttrList shape;
  shape.append(kKeyShapeState, StringAttr::get(context, state));
  shape.append(kKeyLogical, logical.getDictionary(context));
  shape.append(kKeySpecialization, StringAttr::get(context, specialization));

  NamedAttrList record;
  record.append(kKeyFunction, StringAttr::get(context, function));
  record.append(kKeyId, IntegerAttr::get(IntegerType::get(context, 64), id));
  record.append(kKeyPlan, StringAttr::get(context, plan));
  record.append(kKeyShape, shape.getDictionary(context));
  record.append(kKeyScope, makeScope(context, function));
  record.append(kKeyResources, makeResources(context));
  record.append(kKeyProofs, makeProofs(context));
  record.append(kKeyFallback, makeFallback(context));
  return record.getDictionary(context);
}

LogicalResult mlir::hmx::isHmxRecordV3Requested(ModuleOp module,
                                                bool &requested) {
  requested = false;
  Attribute marker = module->getAttr(kHmxDiagnosticRecordV3Attr);
  if (!marker)
    return success();
  if (!isa<UnitAttr>(marker)) {
    module.emitError() << kHmxDiagnosticRecordV3Attr
                       << " must be a unit attribute";
    return failure();
  }
  requested = true;
  return success();
}

LogicalResult mlir::hmx::publishHmxRecordV3(ModuleOp module,
                                            ArrayRef<DictionaryAttr> records) {
  bool requested = false;
  if (failed(isHmxRecordV3Requested(module, requested)))
    return failure();
  if (!requested)
    return success();

  MLIRContext *ctx = module.getContext();
  SmallVector<Attribute> entries;
  if (Attribute raw = module->getAttr(kHmxRecordV3Attr)) {
    auto staged = dyn_cast<DictionaryAttr>(raw);
    if (!staged) {
      module.emitError() << kHmxRecordV3Attr
                         << " is not a well-formed staging dictionary";
      return failure();
    }
    // A finalized document is a published contract.  Rewriting its records from
    // a rerun would invalidate the fingerprints somebody may already have
    // stored, so refuse instead of silently restaging.  The fixed protocol
    // constants are what distinguishes a finalized document from a staging
    // dictionary; any other extra key is simply not a staging dictionary.
    if (staged.get(kKeyRecordMode) || staged.get(kKeyAdmission)) {
      module.emitError() << kHmxRecordV3Attr
                         << " is already a finalized v3 document";
      return failure();
    }
    if (failed(rejectUnknownFields(staged, {StringRef(kKeyRecords)}))) {
      module.emitError() << kHmxRecordV3Attr
                         << " is not a well-formed staging dictionary";
      return failure();
    }
    ArrayAttr previous = arrayField(staged, kKeyRecords);
    if (!previous) {
      module.emitError() << kHmxRecordV3Attr << " has no records array";
      return failure();
    }
    entries.assign(previous.begin(), previous.end());
  }

  // A rerun replaces the record for the same function-local id instead of
  // appending a second representation of one fact.
  auto findRecord = [&](StringRef function, int64_t id)
      -> std::optional<std::size_t> {
    for (auto [index, item] : llvm::enumerate(entries)) {
      auto record = dyn_cast<DictionaryAttr>(item);
      if (!record)
        continue;
      StringAttr owner = stringField(record, kKeyFunction);
      std::optional<int64_t> recordId = readNonnegativeI64(record, kKeyId);
      if (owner && recordId && owner.getValue() == function && *recordId == id)
        return index;
    }
    return std::nullopt;
  };

  std::set<std::pair<std::string, int64_t>> seen;
  for (DictionaryAttr record : records) {
    if (failed(validateRecordSkeleton(record))) {
      module.emitError() << "refusing to publish a malformed v3 record skeleton";
      return failure();
    }
    StringAttr function = stringField(record, kKeyFunction);
    int64_t id = *readNonnegativeI64(record, kKeyId);
    if (!seen.emplace(function.getValue().str(), id).second) {
      module.emitError() << "duplicate v3 record for function "
                         << function.getValue() << " id " << id;
      return failure();
    }
    if (std::optional<std::size_t> existing = findRecord(function.getValue(), id))
      entries[*existing] = record;
    else
      entries.push_back(record);
  }

  // Canonical order, so the published bytes do not depend on the order the
  // attribution driver happened to visit functions.
  llvm::sort(entries, [](Attribute lhs, Attribute rhs) {
    auto left = cast<DictionaryAttr>(lhs);
    auto right = cast<DictionaryAttr>(rhs);
    StringRef leftFunction = stringField(left, kKeyFunction).getValue();
    StringRef rightFunction = stringField(right, kKeyFunction).getValue();
    if (leftFunction != rightFunction)
      return leftFunction < rightFunction;
    return *readNonnegativeI64(left, kKeyId) < *readNonnegativeI64(right, kKeyId);
  });

  NamedAttrList staged;
  staged.append(kKeyRecords, ArrayAttr::get(ctx, entries));
  module->setAttr(kHmxRecordV3Attr, staged.getDictionary(ctx));
  return success();
}

LogicalResult mlir::hmx::finalizeHmxRecordV3(ModuleOp module) {
  bool requested = false;
  if (failed(isHmxRecordV3Requested(module, requested)))
    return failure();
  if (!requested) {
    // A record document without the mode marker is a forged or stale artifact.
    // Leaving it in place would let a later reader treat it as a proven record.
    if (module->getAttr(kHmxRecordV3Attr)) {
      module.emitError() << kHmxRecordV3Attr << " is present without "
                         << kHmxDiagnosticRecordV3Attr;
      return failure();
    }
    return success();
  }

  auto staged =
      dyn_cast_or_null<DictionaryAttr>(module->getAttr(kHmxRecordV3Attr));
  ArrayAttr skeletons = arrayField(staged, kKeyRecords);
  if (!skeletons) {
    module.emitError() << kHmxDiagnosticRecordV3Attr
                       << " is set but no v3 record was published";
    return failure();
  }

  MLIRContext *ctx = module.getContext();
  SmallVector<Attribute> records(skeletons.begin(), skeletons.end());
  // Per-function evidence is attributable to a record only when the function has
  // exactly one record; a shared function-level number would otherwise be
  // silently spread over several records.
  std::map<std::string, unsigned> perFunction;
  for (Attribute item : records)
    ++perFunction[stringField(cast<DictionaryAttr>(item), kKeyFunction)
                      .getValue()
                      .str()];
  for (Attribute &item : records) {
    auto record = cast<DictionaryAttr>(item);
    StringRef function = stringField(record, kKeyFunction).getValue();
    NamedAttrList enriched(record);
    if (perFunction[function.str()] == 1)
      applyEvidence(ctx, enriched, readFunctionEvidence(module, function));
    // The digest covers the *final* content, so it is recomputed for every
    // record regardless of whether evidence was folded in.  Leaving a record
    // without one is not an option: a skeleton is not a publishable record, and
    // a function with several matmuls would otherwise fail the document
    // validation instead of publishing its facts honestly unproven.
    enriched.set(kKeyRecordFingerprint,
                 StringAttr::get(ctx,
                                 computeRecordFingerprint(
                                     enriched.getDictionary(ctx))));
    item = enriched.getDictionary(ctx);
  }

  DictionaryAttr document = makeDocument(ctx, records);
  if (failed(validateDocument(document))) {
    module.emitError() << "the HMX v3 record document failed its own "
                          "validation; refusing to publish a record-only "
                          "contract";
    return failure();
  }
  module->setAttr(kHmxRecordV3Attr, document);
  return success();
}

FailureOr<std::string> mlir::hmx::serializeHmxRecordV3Json(ModuleOp module) {
  bool requested = false;
  if (failed(isHmxRecordV3Requested(module, requested)))
    return failure();
  // A record document without the mode marker is a forged or stale artifact.
  if (!requested && module->getAttr(kHmxRecordV3Attr)) {
    module.emitError() << kHmxRecordV3Attr << " is present without "
                       << kHmxDiagnosticRecordV3Attr;
    return failure();
  }
  if (!requested)
    return failure();

  // Publication is a read: the document was finalized and self-validated by
  // `hmx-v3-record`, and re-validating here keeps a hand-edited module attribute
  // from reaching the wire as if it were a proven record.
  auto document =
      dyn_cast_or_null<DictionaryAttr>(module->getAttr(kHmxRecordV3Attr));
  if (!document || failed(validateDocument(document))) {
    module.emitError() << "the HMX v3 record document is not a valid "
                          "record-only contract";
    return failure();
  }

  json::Object root;
  for (NamedAttribute field : document)
    root[field.getName().strref()] = attrToJson(field.getValue());
  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << json::Value(std::move(root));
  stream.flush();
  return output;
}
