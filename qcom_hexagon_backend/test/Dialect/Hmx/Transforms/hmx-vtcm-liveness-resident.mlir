//===- hmx-vtcm-liveness-resident.mlir - resident live floor ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: functions = [{
// CHECK-DAG: symbol = "resident_floor"
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: deallocation_sites = 0 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 1024 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 1024 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @resident_floor() {
    %workspace = memref.alloc() {hmx.workspace_resident = {key = -2 : i64, bytes = 1024 : i64}}
        : memref<32x16xf16, 1>
    return
  }
}
