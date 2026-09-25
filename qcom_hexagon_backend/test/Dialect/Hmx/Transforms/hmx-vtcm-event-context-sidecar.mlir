//===- hmx-vtcm-event-context-sidecar.mlir - static token contract -------===//
//
// The marker-gated accounting pass may publish an opaque event token only for
// its smallest static scope. This is a compiler sidecar, not a manifest field
// and not a runtime observation.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_event_context = {
// CHECK-DAG: kind = "vtcm-event-context"
// CHECK-DAG: schema = "hmx.vtcm-event-context/v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: eligible = true
// CHECK-DAG: function = "kernel"
// CHECK-DAG: grid_product = 1 : i64
// CHECK-DAG: invocation_id = 1 : i64
// CHECK-DAG: immutable = 1 : i64
// CHECK-DAG: token_basis = "hmx.vtcm-event-token/fnv1a128/v1"
// CHECK-DAG: token_bits = 128 : i64
// CHECK-DAG: allocation_site_id = {{-?[0-9]+}} : i64
// CHECK-DAG: accounting_scope_id = {{-?[0-9]+}} : i64
// CHECK-DAG: function_id = {{-?[0-9]+}} : i64
// CHECK-DAG: token_low = {{-?[0-9]+}} : i64
// CHECK-DAG: token_high = {{-?[0-9]+}} : i64
// CHECK-NOT: hex.hmx.kernel_manifest/v2
// CHECK-NOT: 0x
// CHECK-NOT: pointer
// CHECK-NOT: address=
module @event_context_module attributes {
    hmx.build_id = {low = 11 : i64, high = 22 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context} {
  func.func @kernel() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("event":10:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}
