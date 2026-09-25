//===- hmx-vtcm-accounting.mlir - diagnostic VTCM inventory ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The marker is internal diagnostic IR, not a backend option. The result is
// a raw allocation-site census; it is deliberately not a live-range peak,
// manifest v2 field, or launcher contract.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: kind = "allocation-site-census"
// CHECK-DAG: status = "complete"
// CHECK-DAG: peak_status = "not-proven"
// CHECK-DAG: external_scratch = "none"
// CHECK-DAG: allocation_sites = 4 : i64
// CHECK-DAG: unknown_allocations = 0 : i64
// CHECK-DAG: transient_bytes = 10240 : i64
// CHECK-DAG: workspace_resident_bytes = 1024 : i64
// CHECK-DAG: weight_resident_bytes = 4096 : i64
// CHECK-DAG: resident_site_sum_bytes = 5120 : i64
// CHECK-DAG: raw_site_sum_bytes = 15360 : i64
// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: build_id_status = "not-proven"
// CHECK-DAG: runtime_join_status = "not-integrated"
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.weight_resident_bytes = 4096 : i64} {
  memref.global "private" constant @weight : memref<64x32xf16> = dense<1.000000e+00> {alignment = 128 : i64}
  func.func @account() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    %b = memref.alloc() : memref<32x32xf16, 1>
    %workspace = memref.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 1024 : i64}}
        : memref<32x16xf16, 1>
    %weight = hexagonmem.alloc() {hmx.weight_resident = {global = @weight, bytes = 4096 : i64}}
        : memref<64x32xf16, 1>
    memref.dealloc %a : memref<64x64xf16, 1>
    memref.dealloc %b : memref<32x32xf16, 1>
    return
  }
}
