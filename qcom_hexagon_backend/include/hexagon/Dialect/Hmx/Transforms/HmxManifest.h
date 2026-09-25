//===-- HmxManifest.h - structured HMX attribution manifest ----*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The HMX manifest is module metadata, not an IR rewrite: it records the
// semantic plan for every original linalg.matmul and lets later HMX passes
// update the same decision as they lower it.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXMANIFEST_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXMANIFEST_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
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

/// Internal, test-only IR marker. It is deliberately not a backend option:
/// the normal attribution path never sets it, so production tail selection
/// remains closed while diagnostic producer/consumer tests can exercise the
/// full hmx-tail contract end to end. The source linalg/hmx matmul carries the
/// operation marker; the enclosing module receives the same UnitAttr as durable
/// provenance for the n_tile attributes emitted after that matmul is erased.
inline constexpr StringLiteral kHmxDiagnosticTailAttr =
    "hmx.diagnostic_tail_partition";

/// Internal diagnostic output-bridge coordinate. It is an op attribute, not a
/// backend option: when present, HmxToLLVM selects one physical N output tile
/// from the original ranked descriptors. It is absent on the regular path.
inline constexpr StringLiteral kHmxDiagnosticNTileAttr = "n_tile";

/// The diagnostic guard is deliberately presence-and-type checked. A marker
/// carrying a string/integer/other payload is malformed test IR, not an
/// implicit opt-in.
inline bool isHmxDiagnosticTailMarker(Operation *op) {
  return op && op->hasAttr(kHmxDiagnosticTailAttr) &&
         isa<UnitAttr>(op->getAttr(kHmxDiagnosticTailAttr));
}

/// Publish the operation marker's module-scoped provenance. Reject a malformed
/// existing value rather than replacing it, so a manually malformed diagnostic
/// module cannot be silently repaired into an authorized lowering input.
inline LogicalResult markHmxDiagnosticTailModule(ModuleOp module) {
  if (Attribute marker = module->getAttr(kHmxDiagnosticTailAttr)) {
    if (!isa<UnitAttr>(marker))
      return module.emitError(
          "hmx.diagnostic_tail_partition must be a unit attribute");
    return success();
  }
  module->setAttr(kHmxDiagnosticTailAttr, UnitAttr::get(module.getContext()));
  return success();
}

/// Canonical semantic plan vocabulary. The plan is the executable shape
/// decision; a reason explains why that plan was selected or refused.
inline constexpr StringLiteral kHmxPlanFullHMX = "full-hmx";
inline constexpr StringLiteral kHmxPlanHMXTail = "hmx-tail";
inline constexpr StringLiteral kHmxPlanHVX = "hvx";

/// Canonical attribution reason vocabulary shared by the producer and the
/// manifest validator. Keep the wire names centralized; Python mirrors this
/// independent boundary.
inline constexpr StringLiteral kHmxReasonSelectedAligned = "selected-aligned";
inline constexpr StringLiteral kHmxReasonSelectedTail = "selected-tail";
inline constexpr StringLiteral kHmxReasonLibraryCall = "library-call";
inline constexpr StringLiteral kHmxReasonVtcmAllocatorDisabled =
    "vtcm-allocator-disabled";
inline constexpr StringLiteral kHmxReasonNonRank2 = "non-rank-2";
inline constexpr StringLiteral kHmxReasonDynamicShape = "dynamic-shape";
inline constexpr StringLiteral kHmxReasonUnsupportedDType = "unsupported-dtype";
inline constexpr StringLiteral kHmxReasonMinRows = "min-rows";
inline constexpr StringLiteral kHmxReasonTileAlignment = "tile-alignment";
inline constexpr StringLiteral kHmxReasonUnsupportedLayout =
    "unsupported-layout";
inline constexpr StringLiteral kHmxReasonVtcmBudget = "vtcm-budget";

inline bool isCanonicalHmxMatmulReason(StringRef reason) {
  return reason == kHmxReasonSelectedAligned ||
         reason == kHmxReasonSelectedTail || reason == kHmxReasonLibraryCall ||
         reason == kHmxReasonVtcmAllocatorDisabled ||
         reason == kHmxReasonNonRank2 || reason == kHmxReasonDynamicShape ||
         reason == kHmxReasonUnsupportedDType || reason == kHmxReasonMinRows ||
         reason == kHmxReasonTileAlignment ||
         reason == kHmxReasonUnsupportedLayout ||
         reason == kHmxReasonVtcmBudget;
}

/// Shared lock for passes that read or write module-level HMX state while
/// nested under independently scheduled func.func operations.
std::mutex &hmxModuleStateMutex();

/// Create `hmx.kernel_manifest` when absent, or validate the existing semantic
/// value. A malformed or schema-incompatible manifest is an error rather than a
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

/// Rebind one record to the function argument slot proven by the resident
/// weight pass. This is needed when attribution saw a layout-only view rather
/// than the entry argument itself.
LogicalResult bindHmxManifestWeightSlot(ModuleOp module, StringRef function,
                                        int64_t recordId, int64_t slot);

/// Publish the final policy for one function argument slot. This is called
/// after the resident-weight pass has classified all consumers, so the manifest
/// never advertises a partial first-consumer decision.
LogicalResult setHmxManifestWeightPolicy(ModuleOp module, StringRef function,
                                         int64_t slot, StringRef policy,
                                         StringRef reason,
                                         ArrayRef<int64_t> consumers);

/// Reconcile the final slot policies against the module's prepack declarations
/// after the resident-weight pass has finished its classify/rewrite phase.
LogicalResult reconcileHmxManifestWeightPolicies(ModuleOp module,
                                                 bool prepackRuntimeWeights);

/// Update the workspace/grid facts after the workspace-residency pass has made
/// its final decision for a function.
LogicalResult setHmxManifestWorkspaceClass(ModuleOp module, StringRef function,
                                           StringRef workspaceClass,
                                           StringRef gridPolicy);

/// Complete final-only fields (pipeline presence, weight references, and
/// fingerprints) and validate the semantic manifest before serialization.
LogicalResult finalizeHmxManifest(ModuleOp module);

/// Serialize the module's finalized semantic manifest as a top-level JSON
/// object.
std::string serializeHmxManifestJson(ModuleOp module);

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXMANIFEST_H
