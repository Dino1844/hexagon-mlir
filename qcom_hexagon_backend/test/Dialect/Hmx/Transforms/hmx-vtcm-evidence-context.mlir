//===- hmx-vtcm-evidence-context.mlir - diagnostic context contract --------===//
//
// The evidence context is an internal, marker-gated sidecar.  It describes the
// smallest possible static scope and explicitly refuses a runtime per-site
// join.  It is not a manifest field and does not change allocation behavior.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_evidence_context = {
// CHECK-DAG: kind = "vtcm-evidence-context-v1"
// CHECK-DAG: schema = "hmx.vtcm-evidence-context/v1"
// CHECK-DAG: status = "incomplete"
// CHECK-DAG: mode = "diagnostic-only"
// CHECK-DAG: performance_claimed = 0 : i64
// CHECK-DAG: scope = "one-immutable-principal-module-function-canonical-site-grid1-single-invocation"
// CHECK-DAG: event_owner_status = "aggregate"
// CHECK-DAG: delayed_cache_owner_status = "aggregate"
// CHECK-DAG: function_id_status = "not-proven"
// CHECK-DAG: allocation_site_id_status = "not-proven"
// CHECK-DAG: resident_scope_binding = "not-proven"
// CHECK-DAG: content_identity_status = "not-proven"
// CHECK-DAG: workspace_overwrite_proof_status = "not-proven"
// CHECK-DAG: global_symbol_identity_status = "not-proven"
// CHECK-DAG: packed_weight_source_view_status = "not-proven"
// CHECK-DAG: packed_weight_digest_status = "not-proven"
// CHECK-DAG: pool_cache_combined_status = "not-proven"
// CHECK-DAG: pool high-water and BufferManager retention are separate ledgers
// CHECK-NOT: 0x
// CHECK-NOT: pointer
// CHECK-NOT: address
module @evidence_scope attributes {
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context} {
  func.func @kernel() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("evidence":10:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{hmx.diagnostic_vtcm_evidence_context requires hmx.diagnostic_vtcm_identity}}
module @missing_identity attributes {
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_evidence_context} {
  func.func @kernel() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("evidence":20:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}
