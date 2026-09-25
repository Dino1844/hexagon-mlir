//===-- HmxVtcmAccounting.h - diagnostic VTCM accounting contract -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This header holds the internal VTCM/residency contract shared by the
// workspace producer and the diagnostic accounting pass. The raw census,
// optional structured live-range result, and optional static identity sidecar
// are internal diagnostics: they do not add a field to the v2 HMX manifest and
// are not consumed by the launcher. A later kernel-wide budget schema may
// promote the facts only after their lifetime semantics are proved.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXVTCMACCOUNTING_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXVTCMACCOUNTING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hmx {

/// Internal module marker for the diagnostic accounting pass. It is not a
/// backend option and is never set by ordinary attribution.
inline constexpr StringLiteral kHmxDiagnosticVtcmAccountingAttr =
    "hmx.diagnostic_vtcm_accounting";

/// Optional second marker for the structured live-range probe. It is still an
/// internal diagnostic guard, not a backend option.
inline constexpr StringLiteral kHmxDiagnosticVtcmLivenessAttr =
    "hmx.diagnostic_vtcm_liveness";

/// Optional marker for strict validation of the conservative static
/// site-identity sidecar. This is an internal diagnostic guard, not a backend
/// option. The sidecar is emitted with the marker-gated census; this marker
/// additionally makes missing/ambiguous identity facts fail closed.
inline constexpr StringLiteral kHmxDiagnosticVtcmIdentityAttr =
    "hmx.diagnostic_vtcm_identity";

/// Optional marker for the versioned diagnostic evidence context.  This is
/// deliberately separate from the static identity marker: it records the
/// smallest observation scope and the exact runtime join/refusal boundaries,
/// without changing the static sidecar or any v2 manifest.
inline constexpr StringLiteral kHmxDiagnosticVtcmEvidenceContextAttr =
    "hmx.diagnostic_vtcm_evidence_context";

/// Internal, non-wire static site identity. A future launcher/runtime context
/// ABI may consume this dictionary, but no current consumer is authorized to
/// treat it as a manifest v2 field or as an observed allocation record.
inline constexpr StringLiteral kHmxVtcmIdentityAttr =
    "hmx.kernel_vtcm_identity";

/// Serialization schema for the accounting sidecar itself. Resident IDs use
/// the independent `hmx.resident-key/fnv1a64/v1` hash domain declared by
/// HmxResidentContract.h; changing this accounting schema must not change IDs.
inline constexpr StringLiteral kHmxVtcmIdentitySchema =
    "hmx.vtcm-static-identity/v1";

/// Internal evidence-context sidecar schema.  It is not a v2/v3 wire schema;
/// the only consumer today is the opt-in accounting probe.
inline constexpr StringLiteral kHmxVtcmEvidenceContextAttr =
    "hmx.kernel_vtcm_evidence_context";
inline constexpr StringLiteral kHmxVtcmEvidenceContextSchema =
    "hmx.vtcm-evidence-context/v1";

/// Internal compiler-to-runtime event-token contract. This is separate from
/// both the v1 process context and the HMX manifest. It is emitted only for a
/// marker-gated, one-function/one-site/grid=1 scope; the runtime treats the
/// token as opaque and never derives it from an address.
inline constexpr StringLiteral kHmxVtcmEventContextAttr =
    "hmx.kernel_vtcm_event_context";
inline constexpr StringLiteral kHmxVtcmEventContextSchema =
    "hmx.vtcm-event-context/v1";

/// These are optional, explicitly supplied module attributes. They are never
/// synthesized from IR contents, host prepack data, a pointer, or a build
/// path. Only a 128-bit integer or a {low, high} pair of i64 values is
/// accepted; opaque/hash-like strings remain not-proven. If absent, the
/// sidecar reports build identity as not-proven.
inline constexpr StringLiteral kHmxVtcmBuildIdAttr = "hmx.build_id";
inline constexpr StringLiteral kHmxVtcmBuildIdInputAttr = "build_id";

/// Internal, non-wire accounting result. The dictionary is intentionally not a
/// manifest record: v2 remains bridge-only until a separately reviewed schema
/// can carry kernel-wide peak semantics.
inline constexpr StringLiteral kHmxVtcmAccountingAttr =
    "hmx.kernel_vtcm_accounting";

/// Optional structured allocator-event result. It is separate from the raw
/// census and remains an internal diagnostic only. A function publishes
/// deallocation and peak fields only when its status is `complete`; an
/// incomplete function retains the discovered allocation-site count and reason
/// without turning unsupported lifetime facts into zero-valued claims.
///
/// Published contract of the liveness sidecar:
///
///  * Scope is one function, one invocation. A proven peak is a compiler-static
///    upper bound over the requested-byte model only.
///  * Peak site attribution uses the canonical site identity of
///    `hmx.vtcm-static-identity/v1`, never a census walk ordinal. When a peak
///    site has no provable canonical identity the ID list is withheld and
///    `peak_site_id_status` reads `not-proven`; a site count is still published
///    so the record never degrades into an unexplained empty list.
///  * Alias ownership is a reviewed three-valued lattice. A value must have
///    exactly one owning site; ambiguity is reported, never resolved.
///  * Axes with no proof stay `not-proven` by construction: allocator high
///    water, grid, resident runtime state, fragmentation, and free-cache
///    retention. Nothing here observes the runtime allocator or a launch grid.
inline constexpr StringLiteral kHmxVtcmLivenessAttr =
    "hmx.kernel_vtcm_live_range";

inline bool isHmxDiagnosticVtcmAccountingMarker(ModuleOp module) {
  return module && module->hasAttr(kHmxDiagnosticVtcmAccountingAttr) &&
         isa<UnitAttr>(module->getAttr(kHmxDiagnosticVtcmAccountingAttr));
}

inline bool isHmxDiagnosticVtcmLivenessMarker(ModuleOp module) {
  return module && module->hasAttr(kHmxDiagnosticVtcmLivenessAttr) &&
         isa<UnitAttr>(module->getAttr(kHmxDiagnosticVtcmLivenessAttr));
}

inline bool isHmxDiagnosticVtcmIdentityMarker(ModuleOp module) {
  return module && module->hasAttr(kHmxDiagnosticVtcmIdentityAttr) &&
         isa<UnitAttr>(module->getAttr(kHmxDiagnosticVtcmIdentityAttr));
}

inline bool isHmxDiagnosticVtcmEvidenceContextMarker(ModuleOp module) {
  return module && module->hasAttr(kHmxDiagnosticVtcmEvidenceContextAttr) &&
         isa<UnitAttr>(module->getAttr(kHmxDiagnosticVtcmEvidenceContextAttr));
}

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXVTCMACCOUNTING_H
