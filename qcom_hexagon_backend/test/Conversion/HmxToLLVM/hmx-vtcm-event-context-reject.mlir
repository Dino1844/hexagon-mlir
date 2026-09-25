//===- hmx-vtcm-event-context-reject.mlir - event ABI refuses ambiguity --===//
//
// The event sidecar has an explicit accounting-scope field and fixed-width
// opaque words. The old generic scope_id spelling or a narrowed/ambiguous
// integer type must not be silently reinterpreted. The sidecar is also a
// closed schema: only an absent sidecar may mean "no context", and only the
// exact refusal spelling may mean "not eligible".
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(hmx-to-llvm)'
//===----------------------------------------------------------------------===//

// A present sidecar of the wrong kind of attribute is a malformed record, not
// an absent one: reading it as a dictionary would lower the kernel with no
// event context while the producer clearly published something.
// expected-error @+1 {{diagnostic VTCM event-context sidecar must be a dictionary}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = "vtcm-event-context"} {
  llvm.func @kernel() { llvm.return }
}

// -----

// A record with no typed eligibility flag cannot be read either way.
// expected-error @+1 {{diagnostic VTCM event-context sidecar requires a BoolAttr 'eligible'}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  kind = "vtcm-event-context",
  schema = "hmx.vtcm-event-context/v1",
  status = "not-proven"
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// An integer is not a BoolAttr even though it converts to one.
// expected-error @+1 {{diagnostic VTCM event-context sidecar requires a BoolAttr 'eligible'}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  eligible = 1 : i64,
  kind = "vtcm-event-context",
  schema = "hmx.vtcm-event-context/v1",
  status = "not-proven"
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// A record of another kind is not an event-context record.
// expected-error @+1 {{diagnostic VTCM event-context sidecar requires kind = "vtcm-event-context"}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  eligible = true,
  kind = "vtcm-evidence-context",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  mode = "diagnostic-only",
  function = "kernel",
  grid_product = 1 : i64,
  invocation_id = 1 : i64,
  immutable = 1 : i64,
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  delayed_cache_owner_status = "aggregate",
  grid_scope_status = "not-proven",
  runtime_event_join_status = "not-proven",
  performance_claimed = 0 : i64,
  accounting_scope_id = 303 : i64,
  function_id = 101 : i64,
  allocation_site_id = 202 : i64,
  token_bits = 128 : i64,
  token_low = 404 : i64,
  token_high = 405 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// The ineligible spelling with eligible=true is not a valid eligible record:
// the identity fields are missing and a reason is not one of them.
// expected-error @+1 {{malformed eligible hmx diagnostic VTCM event-context sidecar}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  eligible = true,
  kind = "vtcm-event-context",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  mode = "diagnostic-only",
  function = "kernel",
  grid_product = 1 : i64,
  invocation_id = 1 : i64,
  immutable = 0 : i64,
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  delayed_cache_owner_status = "aggregate",
  grid_scope_status = "not-proven",
  runtime_event_join_status = "not-proven",
  performance_claimed = 0 : i64,
  reason = "static identity is not complete"
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// A promotion claim is not part of this ABI, and the closed key set rejects it
// instead of ignoring it.
// expected-error @+1 {{malformed eligible hmx diagnostic VTCM event-context sidecar}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  eligible = true,
  kind = "vtcm-event-context",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  mode = "diagnostic-only",
  function = "kernel",
  grid_product = 1 : i64,
  invocation_id = 1 : i64,
  immutable = 1 : i64,
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  delayed_cache_owner_status = "aggregate",
  grid_scope_status = "not-proven",
  runtime_event_join_status = "not-proven",
  performance_claimed = 0 : i64,
  promotable = 0 : i64,
  accounting_scope_id = 303 : i64,
  function_id = 101 : i64,
  allocation_site_id = 202 : i64,
  token_bits = 128 : i64,
  token_low = 404 : i64,
  token_high = 405 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// A refused record still has to be the exact refusal spelling: an ineligible
// sidecar that carries a token word is not a record this pass can interpret.
// expected-error @+1 {{malformed ineligible hmx diagnostic VTCM event-context sidecar}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  eligible = false,
  kind = "vtcm-event-context",
  schema = "hmx.vtcm-event-context/v1",
  status = "not-proven",
  mode = "diagnostic-only",
  function = "",
  grid_product = 1 : i64,
  invocation_id = 1 : i64,
  immutable = 0 : i64,
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  delayed_cache_owner_status = "aggregate",
  grid_scope_status = "not-proven",
  runtime_event_join_status = "not-proven",
  performance_claimed = 0 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// The exact ineligible spelling is a legitimate refusal: it is evidence that
// the producer declined, so no event context is emitted and nothing is wrong.
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  eligible = false,
  kind = "vtcm-event-context",
  schema = "hmx.vtcm-event-context/v1",
  status = "not-proven",
  mode = "diagnostic-only",
  function = "",
  grid_product = 1 : i64,
  invocation_id = 1 : i64,
  immutable = 0 : i64,
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  delayed_cache_owner_status = "aggregate",
  grid_scope_status = "not-proven",
  runtime_event_join_status = "not-proven",
  performance_claimed = 0 : i64,
  reason = "static identity is not complete"
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// The old generic scope_id spelling is not the event wire name.
// expected-error @+1 {{malformed eligible hmx diagnostic VTCM event-context sidecar}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  scope_id = 303 : i64,
  allocation_site_id = 202 : i64,
  eligible = true,
  function = "kernel",
  function_id = 101 : i64,
  grid_product = 1 : i64,
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}

// -----

// expected-error @+1 {{malformed eligible hmx diagnostic VTCM event-context sidecar}}
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
  accounting_scope_id = 303 : i32,
  allocation_site_id = 202 : i32,
  eligible = true,
  function = "kernel",
  function_id = 101 : i32,
  grid_product = 1 : i64,
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() { llvm.return }
}
