//===- hmx-vtcm-liveness.mlir - structured linear live upper bound -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: kind = "allocation-site-census"
// CHECK-DAG: peak_status = "not-proven"
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region"
// CHECK-DAG: unit = "requested-bytes"
// CHECK-DAG: functions = [{
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 10240 : i64
// CHECK-DAG: peak_site_count = 2 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "linear_liveness"
// CHECK-DAG: transient_requested_peak_bytes = 10240 : i64
// CHECK-DAG: weight_resident_requested_bytes = 0 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 0 : i64
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @linear_liveness_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @linear_liveness() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    %b = memref.alloc() : memref<32x32xf16, 1>
    memref.dealloc %a : memref<64x64xf16, 1>
    memref.dealloc %b : memref<32x32xf16, 1>
    return
  }
}
