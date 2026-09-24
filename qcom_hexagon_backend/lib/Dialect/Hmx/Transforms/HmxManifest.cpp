//===-- HmxManifest.cpp - structured HMX manifest -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <set>

using namespace mlir;
using namespace mlir::hmx;
namespace json = llvm::json;

std::mutex &mlir::hmx::hmxModuleStateMutex() {
  static std::mutex mutex;
  return mutex;
}

namespace {
constexpr StringLiteral kManifestAttr = "hmx.kernel_manifest";
constexpr StringLiteral kManifestSchema = "hex.hmx.kernel_manifest/v1";
constexpr StringLiteral kKeySchema = "schema";
constexpr StringLiteral kKeyMatmuls = "matmuls";
constexpr StringLiteral kKeyFunction = "function";
constexpr StringLiteral kKeyId = "id";
constexpr StringLiteral kKeyEngine = "engine";
constexpr StringLiteral kKeyReason = "reason";
constexpr StringLiteral kKeyM = "m";
constexpr StringLiteral kKeyN = "n";
constexpr StringLiteral kKeyK = "k";
constexpr StringLiteral kKeyLhsElem = "lhs_elem";
constexpr StringLiteral kKeyRhsElem = "rhs_elem";
constexpr StringLiteral kKeyOutElem = "out_elem";
constexpr StringLiteral kKeyVtcmBudget = "vtcm_budget";
constexpr StringLiteral kKeyVtcmBefore = "vtcm_before";
constexpr StringLiteral kKeyVtcmPeak = "vtcm_peak";
constexpr StringLiteral kKeyBlocking = "blocking";
constexpr StringLiteral kKeyBlockM = "block_m";
constexpr StringLiteral kKeyPipelineRequested = "pipeline_requested";
constexpr StringLiteral kKeyPipelineSelected = "pipeline_selected";
constexpr StringLiteral kKeyPipelineDepth = "pipeline_depth";
constexpr StringLiteral kKeyPipelineReason = "pipeline_reason";
constexpr StringLiteral kKeyPackActSites = "pack_act_sites";
constexpr StringLiteral kKeyPackWeightSites = "pack_weight_sites";
constexpr StringLiteral kKeyUnpackSites = "unpack_sites";
constexpr StringLiteral kKeyCountSemantics = "count_semantics";
constexpr StringLiteral kCountSemantics = "ir_sites";

bool isCanonicalPipelineReason(StringRef reason) {
  return reason == "serial-requested" || reason == "no-row-major-bridge" ||
         reason == "extra-activation-reader" ||
         reason == "invalid-staging-geometry" ||
         reason == "empty-staging-grid" || reason == "staging-grid-mismatch" ||
         reason == "shallow-k" || reason == "vtcm-budget" ||
         reason == "tile-count" || reason == "pipeliner-failed";
}

bool isI64(Attribute attr) {
  auto integer = dyn_cast<IntegerAttr>(attr);
  return integer && integer.getType().isSignlessInteger(64);
}

bool hasI64(DictionaryAttr dict, StringRef name) {
  Attribute attr = dict.get(name);
  return attr && isI64(attr);
}

std::optional<int64_t> i64Value(DictionaryAttr dict, StringRef name) {
  if (!hasI64(dict, name))
    return std::nullopt;
  return cast<IntegerAttr>(dict.get(name)).getInt();
}

LogicalResult emitManifestError(ModuleOp module, bool report, Twine message) {
  if (report)
    module.emitError() << message;
  return failure();
}

/// Validate one v1 matmul record. The optional contract and pipeline groups are
/// all-or-nothing: a partially populated record is a malformed contract, not a
/// field the consumer should guess.
LogicalResult validateRecord(ModuleOp module, DictionaryAttr record,
                             bool report) {
  auto stringField = [&](StringRef name) -> StringAttr {
    return dyn_cast_or_null<StringAttr>(record.get(name));
  };
  auto id = dyn_cast_or_null<IntegerAttr>(record.get(kKeyId));
  if (!stringField(kKeyFunction) ||
      stringField(kKeyFunction).getValue().empty())
    return emitManifestError(module, report,
                             "HMX manifest record has no function name");
  if (!id || !id.getType().isSignlessInteger(64) || id.getInt() < 0)
    return emitManifestError(
        module, report, "HMX manifest record has no valid function-local id");
  StringAttr engine = stringField(kKeyEngine);
  if (!engine || (engine.getValue() != "hmx" && engine.getValue() != "hvx"))
    return emitManifestError(module, report,
                             "HMX manifest record has no valid engine");
  StringAttr reason = stringField(kKeyReason);
  if (!reason || !isCanonicalHmxMatmulReason(reason.getValue()))
    return emitManifestError(module, report,
                             "HMX manifest record has no canonical reason");
  if ((engine.getValue().str() == "hmx") !=
      (reason.getValue().str() == kHmxReasonSelected))
    return emitManifestError(
        module, report,
        "HMX manifest engine and reason disagree on attribution");

  bool hasContractField = record.get(kKeyM) || record.get(kKeyN) ||
                          record.get(kKeyK) || record.get(kKeyLhsElem) ||
                          record.get(kKeyRhsElem) || record.get(kKeyOutElem);
  bool selected = engine.getValue() == "hmx";
  if (selected && !hasContractField)
    return emitManifestError(module, report,
                             "selected HMX record has no matmul contract");
  if (hasContractField) {
    if (!hasI64(record, kKeyM) || !hasI64(record, kKeyN) ||
        !hasI64(record, kKeyK) || !stringField(kKeyLhsElem) ||
        !stringField(kKeyRhsElem) || !stringField(kKeyOutElem))
      return emitManifestError(module, report,
                               "HMX manifest record has a partial contract");
    if (cast<IntegerAttr>(record.get(kKeyM)).getInt() <= 0 ||
        cast<IntegerAttr>(record.get(kKeyN)).getInt() <= 0 ||
        cast<IntegerAttr>(record.get(kKeyK)).getInt() <= 0)
      return emitManifestError(
          module, report, "HMX manifest record has invalid matmul dimensions");
    for (StringRef field : {kKeyLhsElem, kKeyRhsElem, kKeyOutElem}) {
      StringRef element = stringField(field).getValue();
      if (element.empty() || (selected && element != "f16" && element != "f32"))
        return emitManifestError(
            module, report, "HMX manifest record has invalid element type");
    }
  }

  if (selected) {
    if (!hasI64(record, kKeyVtcmBudget) || !hasI64(record, kKeyVtcmBefore) ||
        !hasI64(record, kKeyVtcmPeak) || !hasI64(record, kKeyBlockM) ||
        !hasI64(record, kKeyPackActSites) ||
        !hasI64(record, kKeyPackWeightSites) ||
        !hasI64(record, kKeyUnpackSites) || !stringField(kKeyBlocking) ||
        !stringField(kKeyCountSemantics))
      return emitManifestError(module, report,
                               "selected HMX manifest record is incomplete");
    StringRef blocking = stringField(kKeyBlocking).getValue();
    if (blocking != "whole" && blocking != "m_blocked")
      return emitManifestError(
          module, report, "selected HMX manifest record has invalid blocking");
    if (stringField(kKeyCountSemantics).getValue() != kCountSemantics)
      return emitManifestError(
          module, report,
          "selected HMX manifest record has invalid count semantics");
    int64_t budget = cast<IntegerAttr>(record.get(kKeyVtcmBudget)).getInt();
    int64_t before = cast<IntegerAttr>(record.get(kKeyVtcmBefore)).getInt();
    int64_t peak = cast<IntegerAttr>(record.get(kKeyVtcmPeak)).getInt();
    int64_t blockM = cast<IntegerAttr>(record.get(kKeyBlockM)).getInt();
    if (budget < 0 || before < 0 || peak < before || peak > budget)
      return emitManifestError(
          module, report, "selected HMX manifest record has invalid VTCM peak");
    if (blockM <= 0)
      return emitManifestError(
          module, report, "selected HMX manifest record has invalid block_m");
    int64_t m = cast<IntegerAttr>(record.get(kKeyM)).getInt();
    if ((blocking == "whole" && blockM != m) ||
        (blocking == "m_blocked" && (blockM >= m || m % blockM != 0)))
      return emitManifestError(
          module, report,
          "selected HMX manifest record has inconsistent blocking");
    for (StringRef field :
         {kKeyPackActSites, kKeyPackWeightSites, kKeyUnpackSites})
      if (cast<IntegerAttr>(record.get(field)).getInt() < 0)
        return emitManifestError(
            module, report,
            "selected HMX manifest record has invalid bridge count");
  }
  if (!selected) {
    for (StringRef field :
         {kKeyVtcmBudget, kKeyVtcmBefore, kKeyVtcmPeak, kKeyBlocking,
          kKeyBlockM, kKeyPipelineRequested, kKeyPipelineSelected,
          kKeyPipelineDepth, kKeyPipelineReason, kKeyCountSemantics,
          kKeyPackActSites, kKeyPackWeightSites, kKeyUnpackSites})
      if (record.get(field))
        return emitManifestError(
            module, report,
            "HVX manifest record must not carry HMX-only fields");
  }

  bool hasRequested = record.get(kKeyPipelineRequested) != nullptr;
  bool hasSelected = record.get(kKeyPipelineSelected) != nullptr;
  bool hasDepth = record.get(kKeyPipelineDepth) != nullptr;
  bool hasPipelineReason = record.get(kKeyPipelineReason) != nullptr;
  if (hasRequested || hasSelected || hasDepth || hasPipelineReason) {
    if (!hasI64(record, kKeyPipelineRequested) ||
        !hasI64(record, kKeyPipelineDepth) ||
        !stringField(kKeyPipelineSelected))
      return emitManifestError(
          module, report,
          "HMX manifest record has a partial pipeline decision");
    StringRef selected = stringField(kKeyPipelineSelected).getValue();
    if (selected != "serial" && selected != "staged")
      return emitManifestError(
          module, report, "HMX manifest record has invalid pipeline selection");
    int64_t requested =
        cast<IntegerAttr>(record.get(kKeyPipelineRequested)).getInt();
    int64_t depth = cast<IntegerAttr>(record.get(kKeyPipelineDepth)).getInt();
    if (requested < 0 || depth < 0 || (selected == "serial" && depth != 0) ||
        (selected == "staged" && depth == 0))
      return emitManifestError(
          module, report, "HMX manifest record has invalid pipeline depth");
    if (record.get(kKeyPipelineReason) &&
        (!stringField(kKeyPipelineReason) ||
         !isCanonicalPipelineReason(
             stringField(kKeyPipelineReason).getValue())))
      return emitManifestError(
          module, report, "HMX manifest record has an invalid pipeline reason");
  }

  return success();
}

FailureOr<DictionaryAttr> readManifest(ModuleOp module, bool reportErrors,
                                       bool createIfMissing) {
  Attribute raw = module->getAttr(kManifestAttr);
  if (!raw) {
    if (!createIfMissing)
      return emitManifestError(module, reportErrors,
                               "hmx.kernel_manifest is missing");
    MLIRContext *ctx = module.getContext();
    NamedAttrList fields;
    fields.append(kKeySchema, StringAttr::get(ctx, kManifestSchema));
    fields.append(kKeyMatmuls, ArrayAttr::get(ctx, {}));
    fields.append(kKeyPackActSites,
                  IntegerAttr::get(IntegerType::get(ctx, 64), 0));
    fields.append(kKeyPackWeightSites,
                  IntegerAttr::get(IntegerType::get(ctx, 64), 0));
    fields.append(kKeyUnpackSites,
                  IntegerAttr::get(IntegerType::get(ctx, 64), 0));
    fields.append(kKeyCountSemantics, StringAttr::get(ctx, kCountSemantics));
    module->setAttr(kManifestAttr, fields.getDictionary(ctx));
    return module->getAttrOfType<DictionaryAttr>(kManifestAttr);
  }

  auto manifest = dyn_cast<DictionaryAttr>(raw);
  if (!manifest)
    return emitManifestError(module, reportErrors,
                             "hmx.kernel_manifest must be a DictionaryAttr");
  auto schema = dyn_cast_or_null<StringAttr>(manifest.get(kKeySchema));
  if (!schema || schema.getValue() != kManifestSchema)
    return emitManifestError(module, reportErrors,
                             "hmx.kernel_manifest has an unsupported schema");
  auto matmuls = dyn_cast_or_null<ArrayAttr>(manifest.get(kKeyMatmuls));
  if (!matmuls)
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest.matmuls must be an ArrayAttr");
  std::set<std::pair<std::string, int64_t>> recordKeys;
  int64_t packActTotal = 0;
  int64_t packWeightTotal = 0;
  int64_t unpackTotal = 0;
  for (Attribute item : matmuls) {
    auto record = dyn_cast<DictionaryAttr>(item);
    if (!record)
      return emitManifestError(
          module, reportErrors,
          "hmx.kernel_manifest.matmuls must contain dictionaries");
    if (failed(validateRecord(module, record, reportErrors)))
      return failure();
    auto function = cast<StringAttr>(record.get(kKeyFunction)).getValue();
    int64_t id = cast<IntegerAttr>(record.get(kKeyId)).getInt();
    if (!recordKeys.emplace(function.str(), id).second)
      return emitManifestError(
          module, reportErrors,
          "hmx.kernel_manifest contains duplicate function-local records");
    bool selected =
        cast<StringAttr>(record.get(kKeyEngine)).getValue() == "hmx";
    if (!selected &&
        (record.get(kKeyPackActSites) || record.get(kKeyPackWeightSites) ||
         record.get(kKeyUnpackSites)))
      return emitManifestError(
          module, reportErrors,
          "HVX manifest record must not carry HMX bridge counts");
    if (selected) {
      packActTotal += *i64Value(record, kKeyPackActSites);
      packWeightTotal += *i64Value(record, kKeyPackWeightSites);
      unpackTotal += *i64Value(record, kKeyUnpackSites);
    }
  }
  if (!hasI64(manifest, kKeyPackActSites.str()) ||
      !hasI64(manifest, kKeyPackWeightSites.str()) ||
      !hasI64(manifest, kKeyUnpackSites.str()))
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest has incomplete bridge counts");
  for (StringRef field :
       {kKeyPackActSites, kKeyPackWeightSites, kKeyUnpackSites})
    if (cast<IntegerAttr>(manifest.get(field)).getInt() < 0)
      return emitManifestError(
          module, reportErrors,
          "hmx.kernel_manifest has a negative bridge count");
  int64_t topPackAct = *i64Value(manifest, kKeyPackActSites);
  int64_t topPackWeight = *i64Value(manifest, kKeyPackWeightSites);
  int64_t topUnpack = *i64Value(manifest, kKeyUnpackSites);
  if (topPackAct != packActTotal || topPackWeight != packWeightTotal ||
      topUnpack != unpackTotal)
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest bridge totals disagree with matmul records: "
        "top=(" +
            std::to_string(topPackAct) + "," + std::to_string(topPackWeight) +
            "," + std::to_string(topUnpack) + "), records=(" +
            std::to_string(packActTotal) + "," +
            std::to_string(packWeightTotal) + "," +
            std::to_string(unpackTotal) + ")");
  auto semantics =
      dyn_cast_or_null<StringAttr>(manifest.get(kKeyCountSemantics));
  if (!semantics || semantics.getValue() != kCountSemantics)
    return emitManifestError(
        module, reportErrors,
        "hmx.kernel_manifest.count_semantics must be \"ir_sites\"");
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
    auto function = dyn_cast_or_null<StringAttr>(record.get(kKeyFunction));
    auto recordId = dyn_cast_or_null<IntegerAttr>(record.get(kKeyId));
    if (function && recordId && function.getValue() == functionName &&
        recordId.getInt() == id)
      return index;
  }
  return std::nullopt;
}

json::Value attrToJson(Attribute attr) {
  if (auto integer = dyn_cast<IntegerAttr>(attr))
    return integer.getInt();
  if (auto string = dyn_cast<StringAttr>(attr))
    return string.getValue();
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    json::Value result = json::Array{};
    for (Attribute item : array)
      result.getAsArray()->push_back(attrToJson(item));
    return result;
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attr)) {
    json::Object object;
    for (NamedAttribute field : dictionary)
      object[field.getName().strref()] = attrToJson(field.getValue());
    return json::Value(std::move(object));
  }
  return nullptr;
}

} // namespace

LogicalResult mlir::hmx::ensureHmxManifest(ModuleOp module) {
  if (failed(readManifest(module, /*reportErrors=*/true,
                          /*createIfMissing=*/true)))
    return failure();
  return success();
}

LogicalResult
mlir::hmx::addOrReplaceHmxManifestRecords(ModuleOp module,
                                          ArrayRef<DictionaryAttr> records) {
  FailureOr<DictionaryAttr> current = readManifest(module, true, true);
  if (failed(current))
    return failure();

  MLIRContext *ctx = module.getContext();
  ArrayAttr currentMatmuls = cast<ArrayAttr>(current->get(kKeyMatmuls));
  SmallVector<Attribute> entries(currentMatmuls.begin(), currentMatmuls.end());
  for (DictionaryAttr record : records) {
    if (failed(validateRecord(module, record, true)))
      return failure();
    auto function = cast<StringAttr>(record.get(kKeyFunction)).getValue();
    int64_t id = cast<IntegerAttr>(record.get(kKeyId)).getInt();
    std::optional<std::size_t> existing =
        findRecord(ArrayAttr::get(ctx, entries), function, id);
    if (existing)
      entries[*existing] = record;
    else
      entries.push_back(record);
  }

  std::sort(entries.begin(), entries.end(), [](Attribute lhs, Attribute rhs) {
    auto left = cast<DictionaryAttr>(lhs);
    auto right = cast<DictionaryAttr>(rhs);
    StringRef leftFunction =
        cast<StringAttr>(left.get(kKeyFunction)).getValue();
    StringRef rightFunction =
        cast<StringAttr>(right.get(kKeyFunction)).getValue();
    if (leftFunction != rightFunction)
      return leftFunction < rightFunction;
    return cast<IntegerAttr>(left.get(kKeyId)).getInt() <
           cast<IntegerAttr>(right.get(kKeyId)).getInt();
  });

  int64_t packActTotal = 0;
  int64_t packWeightTotal = 0;
  int64_t unpackTotal = 0;
  for (Attribute item : entries) {
    auto record = cast<DictionaryAttr>(item);
    if (cast<StringAttr>(record.get(kKeyEngine)).getValue() != "hmx")
      continue;
    packActTotal += cast<IntegerAttr>(record.get(kKeyPackActSites)).getInt();
    packWeightTotal +=
        cast<IntegerAttr>(record.get(kKeyPackWeightSites)).getInt();
    unpackTotal += cast<IntegerAttr>(record.get(kKeyUnpackSites)).getInt();
  }

  NamedAttrList fields(*current);
  fields.set(kKeyMatmuls, ArrayAttr::get(ctx, entries));
  auto i64 = IntegerType::get(ctx, 64);
  fields.set(kKeyPackActSites, IntegerAttr::get(i64, packActTotal));
  fields.set(kKeyPackWeightSites, IntegerAttr::get(i64, packWeightTotal));
  fields.set(kKeyUnpackSites, IntegerAttr::get(i64, unpackTotal));
  return writeManifest(module, fields);
}

LogicalResult mlir::hmx::setHmxManifestPipelineDecision(
    ModuleOp module, StringRef functionName, int64_t id, int64_t requested,
    StringRef selected, int64_t depth, StringRef reason) {
  FailureOr<DictionaryAttr> current = readManifest(module, true, true);
  if (failed(current))
    return failure();
  if (selected != "serial" && selected != "staged")
    return emitManifestError(module, true,
                             "invalid HMX pipeline selection in manifest");
  if (requested < 0 || depth < 0 || (selected == "serial" && depth != 0) ||
      (selected == "staged" && depth == 0))
    return emitManifestError(module, true,
                             "invalid HMX pipeline depth in manifest");
  if (!reason.empty() && !isCanonicalPipelineReason(reason))
    return emitManifestError(module, true,
                             "invalid HMX pipeline reason in manifest");

  MLIRContext *ctx = module.getContext();
  ArrayAttr matmuls = cast<ArrayAttr>(current->get(kKeyMatmuls));
  std::optional<std::size_t> index = findRecord(matmuls, functionName, id);
  if (!index)
    return emitManifestError(module, true,
                             "HMX matmul decision id has no manifest record");
  auto target = cast<DictionaryAttr>(matmuls[*index]);
  if (cast<StringAttr>(target.get(kKeyEngine)).getValue() != "hmx")
    return emitManifestError(module, true,
                             "pipeline decision refers to a non-HMX record");
  SmallVector<Attribute> entries(matmuls.begin(), matmuls.end());
  NamedAttrList record(cast<DictionaryAttr>(entries[*index]));
  auto i64 = IntegerType::get(ctx, 64);
  record.set(kKeyPipelineRequested, IntegerAttr::get(i64, requested));
  record.set(kKeyPipelineSelected, StringAttr::get(ctx, selected));
  record.set(kKeyPipelineDepth, IntegerAttr::get(i64, depth));
  if (!reason.empty())
    record.set(kKeyPipelineReason, StringAttr::get(ctx, reason));
  else
    record.erase(kKeyPipelineReason);
  entries[*index] = record.getDictionary(ctx);

  NamedAttrList fields(*current);
  fields.set(kKeyMatmuls, ArrayAttr::get(ctx, entries));
  return writeManifest(module, fields);
}

LogicalResult mlir::hmx::restoreHmxManifestDecisionIds(ModuleOp module,
                                                       func::FuncOp func) {
  FailureOr<DictionaryAttr> current = readManifest(module, true, true);
  if (failed(current))
    return failure();

  SmallVector<int64_t> selectedIds;
  for (Attribute item : cast<ArrayAttr>(current->get(kKeyMatmuls))) {
    auto record = cast<DictionaryAttr>(item);
    auto recordFunction = cast<StringAttr>(record.get(kKeyFunction)).getValue();
    auto engine = cast<StringAttr>(record.get(kKeyEngine)).getValue();
    if (recordFunction == func.getName() && engine.str() == "hmx")
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

  llvm::SmallDenseSet<int64_t, 4> assigned;
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
  FailureOr<DictionaryAttr> current = readManifest(module, true, true);
  if (failed(current))
    return failure();

  struct Counts {
    int64_t packAct = 0;
    int64_t packWeight = 0;
    int64_t unpack = 0;
  };
  MLIRContext *ctx = module.getContext();
  ArrayAttr matmuls = cast<ArrayAttr>(current->get(kKeyMatmuls));
  SmallVector<Counts> perRecord(matmuls.size());
  Counts total;
  LogicalResult status = success();

  // A manifest-backed bridge site is part of the decision record, not an
  // anonymous implementation detail. Reject missing, malformed, or unknown ids
  // instead of silently attributing the count to no record.
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
    if (cast<StringAttr>(record.get(kKeyEngine)).getValue() != "hmx") {
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
    if (cast<StringAttr>(record.get(kKeyEngine)).getValue() != "hmx")
      continue;
    Counts counts = perRecord[index];
    NamedAttrList fields(record);
    fields.set(kKeyPackActSites, IntegerAttr::get(i64, counts.packAct));
    fields.set(kKeyPackWeightSites, IntegerAttr::get(i64, counts.packWeight));
    fields.set(kKeyUnpackSites, IntegerAttr::get(i64, counts.unpack));
    entries[index] = fields.getDictionary(ctx);
  }

  NamedAttrList fields(*current);
  fields.set(kKeyMatmuls, ArrayAttr::get(ctx, entries));
  fields.set(kKeyPackActSites, IntegerAttr::get(i64, total.packAct));
  fields.set(kKeyPackWeightSites, IntegerAttr::get(i64, total.packWeight));
  fields.set(kKeyUnpackSites, IntegerAttr::get(i64, total.unpack));
  return writeManifest(module, fields);
}

std::string mlir::hmx::serializeHmxManifestJson(ModuleOp module) {
  // Attribution is a publication prerequisite. A standalone transform may
  // still lower hand-written HMX IR, but that IR cannot be exported as a valid
  // translation envelope without the module publisher.
  if (!module->getAttr(kManifestAttr))
    return "{}";

  FailureOr<DictionaryAttr> manifest = readManifest(module, false, false);
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
