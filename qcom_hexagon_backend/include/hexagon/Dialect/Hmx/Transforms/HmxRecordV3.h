//===-- HmxRecordV3.h - record-only HMX kernel record (v3) -------------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// `hex.hmx.kernel_manifest/v3` is a *record-only* wire schema.  It publishes the
// compile-time facts and the four independent proof statuses that
// `hex.hmx.kernel_manifest/v2` deliberately does not carry, and it authorizes
// nothing: `record_mode` is fixed to `record-only` and `admission` to
// `not-authorized`.  It is a second producer over the *same* compile inputs and
// the P1.5 diagnostic sidecars; it is never a converter of the serialized v2
// contract, and it never reads the v2 manifest.
//
// The v2 manifest keeps its own schema, serializer, validator, fingerprint and
// launcher semantics.  Nothing in this header may be used to select, re-select
// or budget an HMX/tail plan, to default a resident workspace, or to constrain a
// launch grid.  `docs/hmx/hmx-v3-manifest-decision.md` is the protocol contract.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXRECORDV3_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXRECORDV3_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include <string>

namespace mlir {
namespace hmx {

//===----------------------------------------------------------------------===//
// Internal mode marker (never a backend option)
//===----------------------------------------------------------------------===//

/// Internal, record-only diagnostic marker.  It is deliberately a module
/// attribute rather than a backend option, so the production `full-hmx` path
/// cannot be switched to the v3 envelope from a user-facing knob.  A
/// v3-capable consumer must be a separate envelope identity, not a flag on the
/// v2 one.  The marker is presence- *and* type checked: a marker carrying any
/// other payload is malformed IR, never an implicit opt-in.
///
/// The marker selects the *envelope and the records*, not the evidence.  The
/// requested-byte numbers a record can publish come from the P1.5 census
/// (`hmx.kernel_vtcm_accounting`) and structured liveness
/// (`hmx.kernel_vtcm_live_range`) sidecars, which are themselves produced by
/// separate internal markers and are absent in an ordinary compilation.  A marked
/// module with no sidecars therefore publishes records that are entirely
/// `not-proven` -- which is the normal case, and the reason a record can never
/// be read as an admission.  Both sidecars are required together: the census
/// proves the allocation set is complete, the liveness sidecar supplies the
/// per-function bound, and neither substitutes for the other.
inline constexpr StringLiteral kHmxDiagnosticRecordV3Attr =
    "hmx.diagnostic_v3_record";

/// The module attribute holding the unfinalized v3 record document.  It is
/// published by attribution from compile-time facts and enriched with the P1.5
/// diagnostic sidecars at the serialization boundary; it is not a v2 manifest
/// field and the launcher never reads it.
inline constexpr StringLiteral kHmxRecordV3Attr = "hmx.kernel_record/v3";

/// The only versioned record name in this file.  Like the v2 schema marker it is
/// a wire protocol identity, not a C++ API name.  It is deliberately *not* an
/// upgrade of `hex.hmx.kernel_manifest/v2`: the two schemas are unrelated
/// contracts and neither is derived from the other.
inline constexpr StringLiteral kHmxRecordV3Schema =
    "hex.hmx.kernel_manifest/v3";

/// Transport envelopes.  `v1` is the default and unchanged: it carries only the
/// v2 execution manifest.  `v2` is the coordinated record-only migration: it
/// carries the byte-identical v2 execution manifest plus a separate `hmx_record`
/// v3 child.  Neither envelope is an extension of the other, and no consumer
/// may "try v2 and then guess v3" -- the schema is an explicit identity.
inline constexpr StringLiteral kHmxTranslationV1Schema =
    "hex.hmx.translation/v1";
inline constexpr StringLiteral kHmxTranslationV2Schema =
    "hex.hmx.translation/v2";

//===----------------------------------------------------------------------===//
// Fixed protocol constants
//===----------------------------------------------------------------------===//

/// Neither value is a runtime switch.
inline constexpr StringLiteral kHmxRecordModeRecordOnly = "record-only";
inline constexpr StringLiteral kHmxRecordAdmissionNotAuthorized =
    "not-authorized";

/// The three proof axes that have no accepted evidence yet, and the axis that
/// can be closed by the P1.5 structured liveness analysis.  They are stored and
/// validated independently: a `complete` on one axis never promotes another.
inline constexpr StringLiteral kHmxProofStatusComplete = "complete";
inline constexpr StringLiteral kHmxProofStatusIncomplete = "incomplete";
inline constexpr StringLiteral kHmxProofStatusNotProven = "not-proven";

/// Capacity facts keep three separate units/bases.  Renaming one into another
/// (or collapsing them into a single `peak_bytes`) is forbidden.
inline constexpr StringLiteral kHmxUnitBytes = "bytes";
inline constexpr StringLiteral kHmxBasisCompileTimeRequested =
    "compile-time-requested";
inline constexpr StringLiteral kHmxBasisAllocatorModel = "allocator-model";
inline constexpr StringLiteral kHmxBasisRuntimeObservation =
    "runtime-observation";

/// The only observation scope a v3 record may declare.  The captured device
/// evidence is a process aggregate, so it can never be published as a
/// per-function value: see `docs/hmx/hmx-v3-manifest-decision.md` 3.3.
inline constexpr StringLiteral kHmxObservationScopeProcessHighWater =
    "process-high-water";

/// The fixed analysis scope of the first v3 revision.  `grid` is the *applicability
/// scope* of the record, not a runtime grid the launcher may enforce.
inline constexpr StringLiteral kHmxGridPolicySingleInstance =
    "single-instance";
inline constexpr int64_t kHmxGridRequiredProduct = 1;
inline constexpr int64_t kHmxScopeInvocations = 1;
inline constexpr StringLiteral kHmxResidentScopeProcessFloor = "process-floor";

/// The record's one declared failure behavior, and the only fallback value this
/// revision publishes.
///
/// The decision doc's illustrative shape also listed `on_unproven_proof` and
/// `on_descriptor_mismatch`.  A record-only document makes no such decision --
/// it never re-selects a plan and never resolves a descriptor -- so those two
/// fields were constants with no reader, and a field that only validates itself
/// advertises a capability the record does not have.  They are deliberately
/// absent rather than reserved.  `on_malformed_record` is live: the consumer
/// reads it before rejecting, so a record that declared different semantics
/// would be refused rather than acted on.
inline constexpr StringLiteral kHmxFallbackRejectV3Record =
    "reject-v3-record";

//===----------------------------------------------------------------------===//
// Producer / publication
//===----------------------------------------------------------------------===//

/// Build one v3 record skeleton from compile-time attribution facts.  This is a
/// second producer over the same inputs the v2 record builder reads, never a
/// projection of an existing v2 record.  `dims` holds the logical M/N/K extents
/// with a negative value meaning "dynamic"; the shape state and the upstream
/// specialization policy are derived from those axes rather than accepted from
/// the caller, so the two can never disagree.
///
/// Returns a null attribute when the facts carry no three-axis logical shape,
/// or when an extent is zero.  A v3 record cannot honestly describe a shape the
/// compile inputs do not carry, so no record is published rather than one with a
/// guessed axis.  (A `library_call` matmul *does* get a record when its operand
/// types are shaped: the record publishes the compile-time facts, not a reason.)
Attribute buildHmxRecordV3Skeleton(MLIRContext *context, StringRef function,
                                   int64_t id, StringRef plan,
                                   ArrayRef<int64_t> dims);

/// Publish the record skeletons for one function, replacing any record with the
/// same function-local id.  Inert unless the module carries the internal
/// record-mode marker.
LogicalResult publishHmxRecordV3(ModuleOp module,
                                 ArrayRef<DictionaryAttr> records);

/// True when the module requests the record-only v3 mode.  A malformed marker
/// value is reported and rejected rather than treated as "off".
LogicalResult isHmxRecordV3Requested(ModuleOp module, bool &requested);

/// Fold the P1.5 diagnostic sidecars into the published records, recompute every
/// record fingerprint, and validate the whole document.  Runs after the
/// accounting sidecars exist; fails the module when a record-only document
/// cannot be published, because a partially proven record is still a contract.
LogicalResult finalizeHmxRecordV3(ModuleOp module);

/// Serialize the finalized v3 record document as a top-level JSON object.  This
/// is a pure publication boundary: it validates the document and returns it, and
/// it neither enriches nor repairs anything.  Failure means the module did not
/// finalize a document; the caller must then reject the compilation rather than
/// fall back to the v2 envelope, since a silent schema downgrade is exactly what
/// this schema forbids.  An empty `records` array is a legitimate "nothing
/// publishable this run" result.
FailureOr<std::string> serializeHmxRecordV3Json(ModuleOp module);

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXRECORDV3_H
