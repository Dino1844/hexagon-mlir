//===- hmx-vtcm-event-context-reject.mlir - no partial event join --------===//
//
// More than one canonical site is outside the smallest event-token scope. The
// pass must publish a negative sidecar and never lower a partial context call.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_event_context = {
// CHECK-DAG: eligible = false
// CHECK-DAG: status = "not-proven"
// CHECK-DAG: reason = "event context requires one function and one canonical site"
// CHECK-NOT: hex.hmx.kernel_manifest/v2
module @event_reject attributes {
    hmx.build_id = {low = 11 : i64, high = 22 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context} {
  func.func @kernel() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("reject":10:1)
    %b = memref.alloc() : memref<8x8xf16, 1> loc("reject":11:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    memref.dealloc %b : memref<8x8xf16, 1>
    return
  }
}
