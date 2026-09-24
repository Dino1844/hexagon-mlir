//===-- HmxManifest.h - structured HMX attribution manifest ----*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The v1 HMX manifest is module metadata, not an IR rewrite: it records the
// engine decision for every original linalg.matmul and lets later HMX passes
// update the same decision as they lower it.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXMANIFEST_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXMANIFEST_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <mutex>
#include <string>

namespace mlir {
namespace hmx {

/// The explicit operation attribute carrying a function-local HMX decision id.
/// It is shared by attribution, bufferization, and the tile-level pass so the
/// hand-off never depends on location metadata.
inline constexpr StringLiteral kHmxDecisionIdAttr = "hmx.decision_id";

/// Canonical v1 attribution reason vocabulary. Keep this as the single C++
/// vocabulary shared by the producer and manifest validator; Python's
/// transport validator mirrors the wire names for its independent boundary.
inline constexpr StringLiteral kHmxReasonSelected = "selected";
inline constexpr StringLiteral kHmxReasonLibraryCall = "library-call";
inline constexpr StringLiteral kHmxReasonVtcmAllocatorDisabled =
    "vtcm-allocator-disabled";
inline constexpr StringLiteral kHmxReasonNonRank2 = "non-rank-2";
inline constexpr StringLiteral kHmxReasonDynamicShape = "dynamic-shape";
inline constexpr StringLiteral kHmxReasonUnsupportedDType = "unsupported-dtype";
inline constexpr StringLiteral kHmxReasonMinRows = "min-rows";
inline constexpr StringLiteral kHmxReasonTileAlignment = "tile-alignment";
inline constexpr StringLiteral kHmxReasonVtcmBudget = "vtcm-budget";

inline bool isCanonicalHmxMatmulReason(StringRef reason) {
  return reason == kHmxReasonSelected || reason == kHmxReasonLibraryCall ||
         reason == kHmxReasonVtcmAllocatorDisabled ||
         reason == kHmxReasonNonRank2 || reason == kHmxReasonDynamicShape ||
         reason == kHmxReasonUnsupportedDType || reason == kHmxReasonMinRows ||
         reason == kHmxReasonTileAlignment || reason == kHmxReasonVtcmBudget;
}

/// Shared lock for passes that read or write module-level HMX state while
/// nested under independently scheduled func.func operations.
std::mutex &hmxModuleStateMutex();

/// Create `hmx.kernel_manifest` when absent, or validate the existing v1 value.
/// A malformed or schema-incompatible manifest is an error rather than a
/// silently replaced value: downstream metadata must describe this module, not
/// a best-effort reconstruction of it.
LogicalResult ensureHmxManifest(ModuleOp module);

/// Add records, replacing an existing record with the same function-local id.
/// Replacement makes this operation idempotent without introducing a second
/// representation of one decision.
LogicalResult addOrReplaceHmxManifestRecords(ModuleOp module,
                                             ArrayRef<DictionaryAttr> records);

/// Update the pipeline fields of one attributed HMX record. `reason` is omitted
/// when empty.
LogicalResult setHmxManifestPipelineDecision(ModuleOp module,
                                             StringRef functionName, int64_t id,
                                             int64_t requested,
                                             StringRef selected, int64_t depth,
                                             StringRef reason = {});

/// Verify the explicit decision-id hand-off after tensor-form HMX operations
/// have been rebuilt in memref form. A manifest-backed function must contain
/// exactly one attributed `hmx.matmul` for each selected record; missing,
/// duplicate, or unknown ids are errors.
LogicalResult restoreHmxManifestDecisionIds(ModuleOp module, func::FuncOp func);

/// Recount bridge operations in the current module. These are static IR sites,
/// not launch-time call counts; the manifest publishes that distinction beside
/// the numbers. Calling this after residency and partition rewrites keeps the
/// count aligned with the final bridge IR.
LogicalResult refreshHmxManifestBridgeCounts(ModuleOp module);

/// Serialize the module's valid v1 manifest as a top-level JSON object.
std::string serializeHmxManifestJson(ModuleOp module);

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXMANIFEST_H
