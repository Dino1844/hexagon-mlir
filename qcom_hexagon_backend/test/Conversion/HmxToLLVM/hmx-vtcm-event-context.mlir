//===- hmx-vtcm-event-context.mlir - diagnostic event-token lowering ------===//
//
// The event-context call is a marker-gated diagnostic side effect. This focused
// test uses an already-LLVM function so it checks the opaque token ABI without
// running the full Triton pipeline or changing the v2 manifest.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: llvm.func @kernel
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_event_context_enter_v1_dsp
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_event_context_leave_v1_dsp
// CHECK-NOT: 0x
// CHECK-NOT: ptr=
// CHECK: llvm.return
module attributes {hmx.diagnostic_vtcm_evidence_context,
                    hmx.kernel_vtcm_event_context = {
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
  accounting_scope_id = 303 : i64,
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
