//===- hmx-vtcm-event-context-identity.mlir - token joins static identity ---===//
//
// The event token is derived from the canonical function/site/scope facts in
// hmx.kernel_vtcm_identity. When both sidecars are present the two must agree
// exactly, otherwise the device would compare opaque words that no compiler
// fact backs. This checks the accepting path; the refusals are in
// hmx-vtcm-event-context-reject.mlir.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: llvm.func @kernel
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_event_context_enter_v1_dsp
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_event_context_leave_v1_dsp
// CHECK-NOT: 0x
// CHECK-NOT: ptr=
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_identity = {
  functions = [{
    function = "kernel",
    function_id = 101 : i64,
    sites = [{
      event_token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
      event_token_bits = 128 : i64,
      event_token_high = 405 : i64,
      event_token_low = 404 : i64,
      function = "kernel",
      function_id = 101 : i64,
      identity_status = "complete",
      site_id = 202 : i64
    }]
  }],
  kind = "static-vtcm-site-identity-v1",
  schema = "hmx.vtcm-static-identity/v1",
  scope_id = 303 : i64,
  status = "complete"
},
                    hmx.kernel_vtcm_event_context = {
  accounting_scope_id = 303 : i64,
  allocation_site_id = 202 : i64,
  delayed_cache_owner_status = "aggregate",
  eligible = true,
  function = "kernel",
  function_id = 101 : i64,
  grid_product = 1 : i64,
  grid_scope_status = "not-proven",
  immutable = 1 : i64,
  invocation_id = 1 : i64,
  kind = "vtcm-event-context",
  mode = "diagnostic-only",
  performance_claimed = 0 : i64,
  runtime_event_join_status = "not-proven",
  schema = "hmx.vtcm-event-context/v1",
  status = "complete",
  token_basis = "hmx.vtcm-event-token/fnv1a128/v1",
  token_bits = 128 : i64,
  token_high = 405 : i64,
  token_low = 404 : i64
}} {
  llvm.func @kernel() {
    llvm.return
  }
}
