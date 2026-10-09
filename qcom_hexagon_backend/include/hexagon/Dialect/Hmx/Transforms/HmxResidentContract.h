//===-- HmxResidentContract.h - marker-gated resident identity contract --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The resident lowering ABI carries a 64-bit runtime key, not a string
// identity.  This header defines the small, deterministic identity vocabulary
// used by the *diagnostic* resident contract.  The ordinary production path
// keeps its existing key bytes; the marker-gated path may publish these facts
// and reject an unprovable identity, but must not silently turn a host-side
// prepack hash into a device-side content identity.
//
// Besides the identity vocabulary, this header is the one home of the
// buffer-safety checks the *weight* and *workspace* residency passes share:
// the alias-op list, the allocation-alignment read, the dealloc/escape walk
// and the provenance record skeleton.  The two passes drive one runtime
// residency mechanism, so the predicates that mechanism rests on must not
// drift between them; their genuine differences (which dealloc op forms
// count as owner-visible releases, which fields each record adds) are
// explicit parameters here, never unions.  `HmxVtcmAccountingPass` keeps
// its own, strictly narrower escape analysis and is deliberately not a
// consumer of these helpers.
//
// A principal is one immutable object in one process.  The process scope is
// supplied by the launcher/runtime contract; an anonymous MLIR module is
// therefore represented as "<anonymous-principal>" and is not evidence that
// multiple objects can safely share a process.  Function and source-location
// components make allocation sites stable without using their traversal order.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXRESIDENTCONTRACT_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXRESIDENTCONTRACT_H

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxVtcmAccounting.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace hmx {

/// Per-allocation evidence.  The lowering contract remains in
/// `hmx.workspace_resident` / `hmx.weight_resident`; this sidecar is not read
/// by the runtime ABI and therefore cannot be mistaken for a device hash.
inline constexpr StringLiteral kHmxResidentProvenanceAttr =
    "hmx.resident_provenance";

/// The resident/prepack marker strings.  They are wire spellings on the
/// op/module attributes shared by the producer passes, the HexagonMem
/// conversion, and the host-facing API; one home keeps a one-sided rename from
/// silently dropping the contract.
inline constexpr StringLiteral kHmxWeightResidentAttr = "hmx.weight_resident";
inline constexpr StringLiteral kHmxWorkspaceResidentAttr =
    "hmx.workspace_resident";
inline constexpr StringLiteral kHmxWeightPrepackAttr = "hmx.weight_prepack";
inline constexpr StringLiteral kHmxWeightPrepackLayoutAttr =
    "hmx.weight_prepack_layout";

/// Where a resident weight's buffer lives. The fact rides three places that
/// must agree: the `hmx.weight_resident` descriptor (IR), the `location` field
/// of each `hmx.weight_prepack` entry (host contract), and the manifest policy
/// name (`resident-prepack` / `resident-prepack-ddr`).
///
/// The descriptor spells VTCM by *absence*: VTCM is what a resident was before
/// there was anywhere else to put one, and leaving it off keeps every existing
/// VTCM resident's IR byte-identical. Only a DDR resident carries the key. The
/// contract, which is a host-facing fact sheet rather than IR, always carries
/// the explicit value.
inline constexpr StringLiteral kHmxWeightResidentLocationKey = "location";
inline constexpr StringLiteral kHmxWeightResidentLocationVtcm = "vtcm";
inline constexpr StringLiteral kHmxWeightResidentLocationDdr = "ddr";

/// Stable hash domain for compiler-derived resident identities.  Keep this
/// independent from `hmx.kernel_vtcm_identity`'s accounting schema: evolving
/// the diagnostic census must not silently redefine an existing resident key.
inline constexpr StringLiteral kHmxResidentKeySchema =
    "hmx.resident-key/fnv1a64/v1";
inline constexpr StringLiteral kHmxResidentKeyNamespace = "hmx.resident/v1";
inline constexpr StringLiteral kHmxResidentScope =
    "one-immutable-principal/process";
inline constexpr StringLiteral kHmxResidentNotProven = "not-proven";
inline constexpr StringLiteral kHmxResidentImmutableGlobal =
    "immutable-compile-time-global";

/// Workspace keys use the high bit as their namespace.  Weight residency is
/// passed to the device as a source address and is namespaced by the runtime
/// resident kind; its stable identity is recorded separately in the sidecar.
inline constexpr uint64_t kHmxWorkspaceResidentKeyTag = uint64_t(1) << 63;

/// Hash domain of the workspace compatibility key (function symbol name plus
/// the allocation's index within that function).  It deliberately does not take
/// the module principal: the key must be the same for the same kernel
/// regardless of which module name the compile observed.  Keeping its own
/// schema keeps it out of `residentSiteIdentity`'s domain, where adding the
/// function name alone would be a different (much shorter) message.
inline constexpr StringLiteral kHmxWorkspaceResidentKeySchema =
    "hmx.workspace-key/fnv1a64/v1";

/// Keep producer descriptors within the VTCM allocator's supported alignment
/// range (nonzero powers of two up to the 2048-byte quantum).
inline bool isSupportedResidentAlignment(int64_t alignment) {
  return alignment > 0 && alignment <= 2048 &&
         (alignment & (alignment - 1)) == 0;
}

inline bool hasResidentDiagnosticMarker(ModuleOp module) {
  return module && module->hasAttr(kHmxDiagnosticVtcmAccountingAttr);
}

inline bool isValidResidentDiagnosticMarker(ModuleOp module) {
  if (!hasResidentDiagnosticMarker(module))
    return true;
  return isa<UnitAttr>(module->getAttr(kHmxDiagnosticVtcmAccountingAttr));
}

inline bool isValidResidentIdentityMarker(ModuleOp module) {
  if (!module || !module->hasAttr(kHmxDiagnosticVtcmIdentityAttr))
    return true;
  return hasResidentDiagnosticMarker(module) &&
         isa<UnitAttr>(module->getAttr(kHmxDiagnosticVtcmIdentityAttr));
}

/// Strict producer checks are opt-in through the existing static-identity
/// marker as well as the raw census marker.  The census marker alone must not
/// change production allocation selection; the identity marker is the
/// explicit request to fail closed on identity/descriptor evidence.
inline bool hasStrictResidentContract(ModuleOp module) {
  return hasResidentDiagnosticMarker(module) &&
         isHmxDiagnosticVtcmIdentityMarker(module);
}

/// Return the module symbol when one exists.  Anonymous modules are valid only
/// under the externally supplied one-principal/process scope; this function
/// intentionally does not invent a build ID from IR contents.
inline std::string residentPrincipalName(ModuleOp module) {
  if (auto name =
          module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
    if (!name.getValue().empty() && name.getValue() != "<anonymous-principal>")
      return name.getValue().str();
  }
  return "<anonymous-principal>";
}

/// A source-location based site is stable across reordering of independent
/// allocations.  Unknown locations are rejected by the strict producer rather
/// than falling back to an ordinal, because an ordinal is exactly the
/// repeated-allocation-order ambiguity this contract is meant to expose.  The
/// `slot` component is supplied by the checked resident provenance; it is not
/// inferred from traversal order and is not silently defaulted for a resident
/// source.
inline bool isResidentLocationMetadata(Location location) {
  // A string label in a fused location is metadata only when it wraps an
  // unknown location.  Do not treat a named callsite or a named location with
  // a real child as harmless metadata: it may carry a second source fact.
  while (auto name = dyn_cast<NameLoc>(location))
    location = name.getChildLoc();
  return isa<UnknownLoc>(location);
}

inline std::optional<std::string> residentSite(Location location,
                                               unsigned depth = 0) {
  if (depth > 8)
    return std::nullopt;
  if (auto file = dyn_cast<FileLineColLoc>(location)) {
    if (file.getFilename().getValue().empty() || file.getLine() == 0 ||
        file.getColumn() == 0)
      return std::nullopt;
    std::string result = "file:";
    result += file.getFilename().getValue().str();
    result += ":" + std::to_string(file.getLine());
    result += ":" + std::to_string(file.getColumn());
    return result;
  }
  if (auto name = dyn_cast<NameLoc>(location)) {
    if (isResidentLocationMetadata(location))
      return std::nullopt;
    return residentSite(name.getChildLoc(), depth + 1);
  }
  if (auto fused = dyn_cast<FusedLoc>(location)) {
    std::optional<std::string> selected;
    unsigned sourceCount = 0;
    for (Location nested : fused.getLocations()) {
      if (auto nestedSite = residentSite(nested, depth + 1)) {
        // Even identical file facts are two source components; selecting one
        // would make the identity depend on an arbitrary fused-location slot.
        if (++sourceCount != 1)
          return std::nullopt;
        selected = std::move(*nestedSite);
        continue;
      }
      // A NameLoc/UnknownLoc pair is a debug label, not a source location.
      // Every other unproven component makes the fused location ambiguous.
      if (!isResidentLocationMetadata(nested))
        return std::nullopt;
    }
    return selected;
  }
  return std::nullopt;
}

inline std::optional<std::string> residentSite(Operation *op) {
  return op ? residentSite(op->getLoc()) : std::nullopt;
}

/// Validate the serialized form of a producer-emitted site. Existing resident
/// records use this persisted value as their allocation-site identity: MLIR's
/// textual form may elide an operation's original file location, so recomputing
/// it after a print/parse roundtrip would turn a stable record into a false
/// mismatch.
inline bool isCanonicalResidentSite(StringRef site) {
  constexpr StringLiteral kPrefix = "file:";
  if (!site.consume_front(kPrefix))
    return false;
  size_t columnSeparator = site.rfind(':');
  if (columnSeparator == StringRef::npos || columnSeparator == 0)
    return false;
  StringRef beforeColumn = site.substr(0, columnSeparator);
  size_t lineSeparator = beforeColumn.rfind(':');
  if (lineSeparator == StringRef::npos || lineSeparator == 0)
    return false;
  unsigned long long line = 0;
  unsigned long long column = 0;
  StringRef lineText = beforeColumn.substr(lineSeparator + 1);
  StringRef columnText = site.substr(columnSeparator + 1);
  return !lineText.getAsInteger(10, line) &&
         !columnText.getAsInteger(10, column) && line != 0 && column != 0 &&
         line <= static_cast<unsigned long long>(
                     std::numeric_limits<int64_t>::max()) &&
         column <= static_cast<unsigned long long>(
                       std::numeric_limits<int64_t>::max());
}

/// Standard FNV-1a-64 parameters. `hashResidentPart` adds a fixed-width
/// little-endian length before each component, so this is a versioned,
/// length-delimited FNV-1a domain rather than a raw-string hash.
inline constexpr uint64_t kHmxResidentHashOffset = 14695981039346656037ull;
inline constexpr uint64_t kHmxResidentHashPrime = 1099511628211ull;

inline void hashResidentPart(uint64_t &state, StringRef value) {
  // FNV-1a over a length-delimited component.  Length delimiting avoids
  // ambiguous concatenations such as principal "ab" + function "c" versus
  // principal "a" + function "bc".
  uint64_t length = value.size();
  for (unsigned byte = 0; byte != 8; ++byte) {
    state ^= length & 0xff;
    state *= kHmxResidentHashPrime;
    length >>= 8;
  }
  for (char ch : value) {
    state ^= static_cast<uint8_t>(ch);
    state *= kHmxResidentHashPrime;
  }
}

/// Opaque token carried by the diagnostic event-context ABI. It is a pair of
/// hash words, not a pointer or an address-bearing object. The compiler
/// publishes the canonical tuple separately; the runtime compares the pair
/// exactly and never interprets it.
struct HmxDiagnosticEventToken {
  uint64_t low{0};
  uint64_t high{0};

  bool isZero() const { return low == 0 && high == 0; }

  bool operator==(const HmxDiagnosticEventToken &other) const {
    return low == other.low && high == other.high;
  }
};

inline constexpr StringLiteral kHmxDiagnosticEventTokenSchema =
    "hmx.vtcm-event-token/fnv1a128/v1";

/// Schema of the per-*site* token, which is a different claim from the frame
/// token above and must never be substituted for it.
///
/// The frame token (`kHmxDiagnosticEventTokenSchema`) names the observation
/// frame: one principal, one module, one function, one source site, one
/// invocation.  It deliberately excludes role and slot, because the frame is
/// about the observation and not about an individual allocation inside it.
///
/// The site token names one canonical allocation site, so it must include the
/// two facts that distinguish two allocations which share a source location:
/// the reviewed semantic role and the checked resident slot.  Two allocations
/// at the same file:line:col with different roles or different resident slots
/// are different sites, and a token that ignored those components would give
/// them one identity.
///
/// A distinct schema string is what makes the two tokens distinguishable at
/// every consumer: a site token is not a frame token that happens to have more
/// inputs, and a reader that is handed one where the other is expected can say
/// so instead of comparing two unrelated words.
inline constexpr StringLiteral kHmxDiagnosticSiteTokenSchema =
    "hmx.vtcm-site-token/fnv1a128/v1";

/// Per-op stamp naming the canonical site of one pool-backed allocation, and
/// the module-level table of every stamped site.  Both are internal diagnostic
/// records, not a manifest field and not a lowering contract on their own.
inline constexpr StringLiteral kHmxVtcmSiteScopeAttr = "hmx.vtcm_site_scope";
inline constexpr StringLiteral kHmxVtcmSiteScopesAttr =
    "hmx.kernel_vtcm_site_scopes";
inline constexpr StringLiteral kHmxVtcmSiteScopesSchema =
    "hmx.vtcm-site-scopes/v1";

/// Closed key set of an *eligible* per-op site stamp.  The set is exact: the
/// lowering pass turns these words into runtime ABI constants, so a missing
/// field would have to be defaulted and an extra field would be a promotion
/// claim this ABI does not make.
inline ArrayRef<StringLiteral> hmxDiagnosticSiteScopeKeys() {
  static constexpr StringLiteral kKeys[] = {
      "kind",
      "schema",
      "status",
      "role",
      "slot",
      "source",
      "function",
      "function_id",
      "site_id",
      "accounting_scope_id",
      "token_basis",
      "token_bits",
      "token_low",
      "token_high",
      "build_id_low",
      "build_id_high",
      "invocation_id",
      "grid_product",
  };
  return kKeys;
}

/// Closed key set of the module-level site table.  `eligible` and `reason` are
/// present on both shapes so a reader never has to infer which record it holds:
/// the eligible shape carries the per-site array, the ineligible one carries a
/// reason instead.
inline ArrayRef<StringLiteral> hmxDiagnosticSiteScopesEligibleKeys() {
  static constexpr StringLiteral kKeys[] = {
      "kind",
      "schema",
      "status",
      "eligible",
      "mode",
      "principal",
      "module",
      "function",
      "function_id",
      "accounting_scope_id",
      "build_id_low",
      "build_id_high",
      "grid_product",
      "invocation_id",
      "immutable",
      "token_basis",
      "site_count",
      "sites",
      "performance_claimed",
  };
  return kKeys;
}

inline ArrayRef<StringLiteral> hmxDiagnosticSiteScopesIneligibleKeys() {
  static constexpr StringLiteral kKeys[] = {
      "kind",
      "schema",
      "status",
      "eligible",
      "mode",
      "principal",
      "module",
      "function",
      "function_id",
      "accounting_scope_id",
      "build_id_low",
      "build_id_high",
      "grid_product",
      "invocation_id",
      "immutable",
      "token_basis",
      "site_count",
      "performance_claimed",
      "reason",
  };
  return kKeys;
}

/// Closed key set of one entry in the `sites` array of an eligible table.  The
/// `requested_bytes` field is the compiler's static request, not an observation;
/// it is carried so a reader can tell a site whose size was proven from one
/// whose extent only had a constant bound.
inline ArrayRef<StringLiteral> hmxDiagnosticSiteScopesEntryKeys() {
  static constexpr StringLiteral kKeys[] = {
      "role",
      "slot",
      "source",
      "function",
      "function_id",
      "site_id",
      "token_basis",
      "token_bits",
      "token_low",
      "token_high",
      "requested_bytes",
      "constant_bounded_extent",
  };
  return kKeys;
}

/// Wire kind of the `hmx.kernel_vtcm_event_context` sidecar. The producer
/// (HmxVtcmAccountingPass) and the consumer (HmxToLLVMPass) must agree on this
/// spelling exactly; a dictionary of any other kind is not an event-context
/// record and must not be read as one.
inline constexpr StringLiteral kHmxDiagnosticEventContextKind =
    "vtcm-event-context";

/// Closed key set of an *eligible* event-context record: the scope vocabulary
/// plus the canonical function/site identity the token was derived from. The
/// set is exact on purpose. A missing or extra key is a schema violation, not an
/// ignorable extra, so a stale or legacy spelling such as the v1 process
/// `scope_id` -- or a promotion field such as `promotable` that this ABI never
/// claims -- is rejected instead of being read as "the field I wanted was
/// absent".
inline ArrayRef<StringLiteral> hmxDiagnosticEventContextEligibleKeys() {
  static constexpr StringLiteral kKeys[] = {
      "kind",
      "schema",
      "status",
      "eligible",
      "mode",
      "function",
      "grid_product",
      "invocation_id",
      "immutable",
      "token_basis",
      "delayed_cache_owner_status",
      "grid_scope_status",
      "runtime_event_join_status",
      "performance_claimed",
      "accounting_scope_id",
      "function_id",
      "allocation_site_id",
      "token_bits",
      "token_low",
      "token_high",
  };
  return kKeys;
}

/// Closed key set of an *ineligible* record: the same scope vocabulary with a
/// `reason` in place of the identity fields. A negative record is still a typed
/// statement ("this pass retained a refusal"), so it is validated against its
/// own closed set instead of being accepted as a dictionary of unknown shape.
inline ArrayRef<StringLiteral> hmxDiagnosticEventContextIneligibleKeys() {
  static constexpr StringLiteral kKeys[] = {
      "kind",
      "schema",
      "status",
      "eligible",
      "mode",
      "function",
      "grid_product",
      "invocation_id",
      "immutable",
      "token_basis",
      "delayed_cache_owner_status",
      "grid_scope_status",
      "runtime_event_join_status",
      "performance_claimed",
      "reason",
  };
  return kKeys;
}

/// Exact set equality against a closed key set. A duplicate key cannot reach
/// here (DictionaryAttr is uniqued), so size plus membership is equality.
inline bool hasExactHmxDiagnosticEventContextKeys(
    DictionaryAttr attr, ArrayRef<StringLiteral> expected) {
  if (!attr || attr.size() != expected.size())
    return false;
  for (StringLiteral key : expected)
    if (!attr.get(key))
      return false;
  return true;
}

/// Runtime event-context ABI: entry/leave symbols, version, and the two flag
/// bits the compiler is allowed to set. These are the compiler's half of the
/// contract; the runtime's half lives in bin/runtime/include/VTCMPool.h and
/// bin/runtime/include/HexagonCAPI.h, and
/// bin/runtime/test/test_vtcm_event_abi_constants_contract.py pins the two
/// spellings to each other. Adding a bit here without a runtime meaning would
/// let a kernel claim a scope the device does not implement.
inline constexpr StringLiteral kHmxDiagnosticEventContextEnterFn =
    "hexagon_runtime_vtcm_accounting_event_context_enter_v1_dsp";
inline constexpr StringLiteral kHmxDiagnosticEventContextLeaveFn =
    "hexagon_runtime_vtcm_accounting_event_context_leave_v1_dsp";
inline constexpr uint32_t kHmxDiagnosticEventContextAbiVersion = 1;
inline constexpr uint32_t kHmxDiagnosticEventContextFlagSingleInvocation =
    1u << 0;
inline constexpr uint32_t kHmxDiagnosticEventContextFlagGridOne = 1u << 1;

/// The scope word is deliberately independent from the resident-v2 scope pair.
/// It identifies the static principal/module observation scope only; it does
/// not join a resident object or prove anything about a process reset.
inline uint64_t diagnosticScopeIdentity(StringRef principal,
                                        StringRef module) {
  uint64_t state = kHmxResidentHashOffset;
  hashResidentPart(state, kHmxDiagnosticEventTokenSchema);
  hashResidentPart(state, "scope");
  hashResidentPart(state, principal);
  hashResidentPart(state, module);
  return state;
}

/// Derive the opaque 128-bit event token from the canonical static scope and
/// the one explicitly supported invocation ordinal. Two independent FNV
/// states make the pair an actual 128-bit token while retaining the repository's
/// simple, length-delimited hash vocabulary. The caller still performs
/// collision checks and refuses a zero token; this helper is not a runtime
/// identity oracle.
inline HmxDiagnosticEventToken diagnosticEventToken(
    StringRef principal, StringRef module, StringRef function, StringRef site,
    uint64_t invocationId) {
  uint64_t low = kHmxResidentHashOffset;
  uint64_t high = kHmxResidentHashOffset ^ 0x9e3779b97f4a7c15ull;
  auto add = [&](StringRef value) {
    hashResidentPart(low, value);
    hashResidentPart(high, value);
  };
  add(kHmxDiagnosticEventTokenSchema);
  add("scope");
  add(principal);
  add(module);
  add("function");
  add(function);
  add("site");
  add(site);
  add("invocation");
  add(std::to_string(invocationId));
  return {low, high};
}

/// Derive the opaque 128-bit *site* token from the canonical site tuple plus
/// the one explicitly supported invocation ordinal.
///
/// This is deliberately a different function from `diagnosticEventToken`, not a
/// wider version of it.  The frame token answers "which observation is this";
/// the site token answers "which allocation site inside that observation is
/// this".  Mixing them would make a frame token look like a site token (or the
/// reverse) at a consumer that compares the pair, so role and slot are added
/// here and the schema string differs.
///
/// The hash construction is the same length-delimited FNV-1a double-state
/// scheme the rest of this header uses, so the two tokens have the same
/// collision properties and the same domain separation.  As with the frame
/// token, the caller performs the collision check and refuses a zero token;
/// this helper is not a runtime identity oracle.
inline HmxDiagnosticEventToken diagnosticSiteToken(
    StringRef principal, StringRef module, StringRef function, StringRef site,
    StringRef role, int64_t slot, uint64_t invocationId) {
  uint64_t low = kHmxResidentHashOffset;
  uint64_t high = kHmxResidentHashOffset ^ 0x9e3779b97f4a7c15ull;
  auto add = [&](StringRef value) {
    hashResidentPart(low, value);
    hashResidentPart(high, value);
  };
  add(kHmxDiagnosticSiteTokenSchema);
  add("scope");
  add(principal);
  add(module);
  add("function");
  add(function);
  add("site");
  add(site);
  add("role");
  add(role);
  add("slot");
  add(std::to_string(slot));
  add("invocation");
  add(std::to_string(invocationId));
  return {low, high};
}

/// Runtime site-scope ABI: entry/leave symbols, version, and the two flag bits
/// the compiler is allowed to set. These are the compiler's half; the runtime's
/// half is `kAccountingSiteScopeVersion` and the two `kAccountingSiteScope*`
/// flag bits in bin/runtime/include/VTCMPool.h, pinned to each other by
/// bin/runtime/test/test_vtcm_site_scope_contract.py. Adding a bit here without a
/// runtime meaning would let a kernel claim a scope the device does not
/// implement.
inline constexpr StringLiteral kHmxDiagnosticSiteScopeEnterFn =
    "hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp";
inline constexpr StringLiteral kHmxDiagnosticSiteScopeLeaveFn =
    "hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp";
inline constexpr uint32_t kHmxDiagnosticSiteScopeAbiVersion = 1;
inline constexpr uint32_t kHmxDiagnosticSiteScopeFlagSingleInvocation = 1u << 0;
inline constexpr uint32_t kHmxDiagnosticSiteScopeFlagGridOne = 1u << 1;

/// True when `attr` is exactly the per-op stamp shape the lowering consumes. A
/// stamp missing a word would have to be defaulted, and an extra word would be a
/// promotion claim this ABI does not make, so both are schema violations rather
/// than ignorable extras. The generic exact-key helper is reused on purpose: the
/// check is the same shape as the frame context's, and a second implementation
/// would be a second thing to keep correct.
inline bool hasExactHmxDiagnosticSiteScopeKeys(DictionaryAttr attr,
                                               ArrayRef<StringLiteral> expected) {
  return hasExactHmxDiagnosticEventContextKeys(attr, expected);
}

inline uint64_t residentFunctionIdentity(StringRef principal,
                                         StringRef function) {
  uint64_t state = kHmxResidentHashOffset;
  hashResidentPart(state, kHmxResidentKeySchema);
  hashResidentPart(state, "function");
  hashResidentPart(state, principal);
  hashResidentPart(state, function);
  return state;
}

/// This tuple is also published by hmx-vtcm-accounting's static identity
/// sidecar.  Both surfaces hash the independent resident-key schema above, so
/// a future site-to-resident join does not depend on the accounting sidecar
/// schema or pretend that runtime observation is already integrated.
inline uint64_t residentSiteIdentity(StringRef principal, StringRef function,
                                     StringRef site, StringRef role,
                                     int64_t slot) {
  uint64_t state = kHmxResidentHashOffset;
  hashResidentPart(state, kHmxResidentKeySchema);
  hashResidentPart(state, "allocation-site");
  hashResidentPart(state, principal);
  hashResidentPart(state, function);
  hashResidentPart(state, site);
  hashResidentPart(state, role);
  hashResidentPart(state, std::to_string(slot));
  return state;
}

inline uint64_t stableWorkspaceResidentKey(StringRef principal,
                                           StringRef function, StringRef site,
                                           int64_t slot) {
  uint64_t hash = residentSiteIdentity(principal, function, site,
                                       "workspace-resident", slot);
  return kHmxWorkspaceResidentKeyTag | (hash & ~kHmxWorkspaceResidentKeyTag);
}

/// The alias ops a resident candidate's value may flow through while still
/// naming the same underlying storage: layout-only views and casts.  The
/// weight and the workspace residency passes walk the same list on purpose:
/// both ask the same question -- "is this use a new name for the principal
/// allocation, or a new object?" -- about the same runtime residency
/// mechanism, and a list that drifted between the two would let a use be a
/// tracked alias on one path and an invisible one on the other.
///
/// `HmxVtcmAccountingPass` is deliberately not a consumer: its escape
/// analysis is strictly narrower (it rejects every dealloc, every
/// region-bearing transfer and every raw-pointer extraction outright), and
/// sharing this list would silently loosen it.
inline bool isResidentAliasLike(Operation *op) {
  return isa<memref::AssumeAlignmentOp, memref::CastOp, memref::SubViewOp,
             memref::ReinterpretCastOp, memref::MemorySpaceCastOp,
             memref::TransposeOp, memref::ViewOp, memref::ExpandShapeOp,
             memref::CollapseShapeOp, memref::ReshapeOp>(op);
}

/// The alignment a resident allocation's descriptor records.  Both producer
/// forms (`memref.alloc` before conversion, `hexagonmem.alloc` after) carry
/// an optional `alignment` attribute, and `ConvertToHexagonmem` gives an
/// omitted memref alignment the same 128-byte default as `hexagonmem.alloc`,
/// so the default is spelled out here: the recorded descriptor is the
/// descriptor the runtime receives, not an assumption about an unrelated
/// MLIR default.
inline int64_t residentAlignment(Operation *alloc) {
  if (auto attr = alloc->getAttrOfType<IntegerAttr>("alignment"))
    return attr.getInt();
  return 128;
}

/// Which deallocation op forms the shared dealloc walk reports as
/// owner-visible releases.  The two residency passes genuinely differ here,
/// and the difference is an explicit parameter rather than a union: the
/// workspace pass runs before `convert-to-hexagonmem`, so every release it
/// audits is still a `memref.dealloc`, while the weight path also validates
/// records left by an earlier lowering, where the allocation and its release
/// have already taken the `hexagonmem` form.  A union would silently widen
/// the workspace side: a hand-written `hexagonmem.dealloc` would start
/// counting as a direct release instead of an unrecognized use.
enum class ResidentDeallocForm {
  MemrefOnly,
  MemrefAndHexagonmem,
};

/// Audit every release of a resident candidate's value before anything is
/// erased.  A direct dealloc of the principal value is the one form the
/// residency passes may transfer or drop; a dealloc reached through an alias
/// is ambiguous, because the runtime resident table is keyed by the principal
/// allocation and not by an arbitrary alias descriptor; and an escape through
/// a call, control flow or a raw pointer extraction is a use no residency
/// pass can reason about.  `subject` names the resident in the diagnostics
/// ("resident workspace", "resident weight bridge") so the shared walk keeps
/// each pass's wording.
inline LogicalResult
validateResidentDeallocs(Value value, Operation *anchor, unsigned &directCount,
                         bool &aliasDealloc, ResidentDeallocForm deallocForm,
                         StringRef subject) {
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
            subject +
            " value escapes through a call/control-flow/pointer operation");
      if (isa<memref::DeallocOp>(owner) ||
          (deallocForm == ResidentDeallocForm::MemrefAndHexagonmem &&
           isa<hexagonmem::DeallocOp>(owner))) {
        if (current == value)
          ++directCount;
        else
          aliasDealloc = true;
        continue;
      }
      if (isResidentAliasLike(owner)) {
        for (Value result : owner->getResults())
          worklist.push_back(result);
      }
    }
  }
  if (aliasDealloc)
    return anchor->emitError(
        subject + " has a deallocation through a memref alias/view");
  return success();
}

/// Append the fields every resident provenance record shares.  The two
/// producers build one diagnostic vocabulary over one residency mechanism,
/// so the shared skeleton lives with it; each pass's own fields (the
/// workspace key, the weight identity/runtime-key/source descriptors) are
/// appended by the passes themselves and stay with them.
///
/// `kind` and `role` carry the vocabulary difference between the two
/// records ("workspace"/"workspace-resident" versus
/// "weight"/"weight-resident"); `role` also feeds `site_id`, so the two
/// cannot drift apart at a call site.  `contentStatus` is a parameter
/// because the two passes make different claims about content: a workspace
/// is refilled every launch, while a weight's content status depends on its
/// source.
///
/// `DictionaryAttr` canonicalizes its entries by sorting them, so the append
/// order here reaches neither the printed record nor the rerun comparison.
inline void appendResidentProvenanceCore(NamedAttrList &fields,
                                         MLIRContext *context, StringRef kind,
                                         StringRef role, StringRef principal,
                                         StringRef function, StringRef site,
                                         int64_t slot, int64_t bytes,
                                         int64_t alignment,
                                         StringRef contentStatus) {
  fields.append("schema", StringAttr::get(context, kHmxResidentKeySchema));
  fields.append("key_namespace",
                StringAttr::get(context, kHmxResidentKeyNamespace));
  fields.append("kind", StringAttr::get(context, kind));
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
  fields.append("function_id",
                IntegerAttr::get(IntegerType::get(context, 64),
                                 residentFunctionIdentity(principal,
                                                          function)));
  fields.append("role", StringAttr::get(context, role));
  fields.append("site", StringAttr::get(context, site));
  fields.append("site_id",
                IntegerAttr::get(IntegerType::get(context, 64),
                                 residentSiteIdentity(principal, function, site,
                                                      role, slot)));
  fields.append("slot", IntegerAttr::get(IntegerType::get(context, 64), slot));
  fields.append("scope", StringAttr::get(context, kHmxResidentScope));
  fields.append("launch_status",
                StringAttr::get(context, kHmxResidentNotProven));
  fields.append("content_status", StringAttr::get(context, contentStatus));
  fields.append("reuse_status", StringAttr::get(context, "process-resident"));
  fields.append("descriptor_status", StringAttr::get(context, "checked"));
}

/// Publish or re-check a provenance record.  A repeated lowering must produce
/// the same evidence it published the first time; a changed record is a
/// changed identity claim, not a refresh.  `subject` names the resident in
/// the diagnostic so the shared check keeps each pass's wording.
inline LogicalResult setOrValidateResidentProvenance(Operation *operation,
                                                     DictionaryAttr expected,
                                                     StringRef subject) {
  auto existing =
      operation->getAttrOfType<DictionaryAttr>(kHmxResidentProvenanceAttr);
  if (!existing) {
    operation->setAttr(kHmxResidentProvenanceAttr, expected);
    return success();
  }
  if (existing != expected)
    return operation->emitError(subject +
                                " provenance changed across repeated lowering");
  return success();
}

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXRESIDENTCONTRACT_H
