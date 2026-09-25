//===- hmx-vtcm-liveness-supported.mlir - proven liveness subset ----------===//
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
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_sites = 4 : i64
// CHECK-DAG: unknown_allocations = 0 : i64
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region"
// CHECK-DAG: symbol = "nested_scf"
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: allocation_sites = 3 : i64
// CHECK-DAG: deallocation_sites = 3 : i64
// CHECK-DAG: transient_requested_peak_bytes = 10240 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 10240 : i64
// CHECK-DAG: symbol = "synchronous_alias_use"
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: deallocation_sites = 1 : i64
// CHECK-DAG: transient_requested_peak_bytes = 512 : i64
// CHECK-DAG: peak_site_count = 1 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @supported_subset_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @nested_scf() {
    %outer = memref.alloc() : memref<64x64xf16, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cond = arith.constant true
    scf.for %i = %c0 to %c2 step %c1 {
      scf.if %cond {
        %then = memref.alloc() : memref<32x32xf16, 1>
        memref.dealloc %then : memref<32x32xf16, 1>
      } else {
        %else = memref.alloc() : memref<16x16xf16, 1>
        memref.dealloc %else : memref<16x16xf16, 1>
      }
      scf.yield
    }
    memref.dealloc %outer : memref<64x64xf16, 1>
    return
  }

  func.func @synchronous_alias_use() {
    %value = memref.alloc() : memref<16x16xf16, 1>
    %scratch = memref.alloc() : memref<4xf16>
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    memref.store %one, %value[%zero, %zero] : memref<16x16xf16, 1>
    %loaded = memref.load %value[%zero, %zero] : memref<16x16xf16, 1>
    %alias = memref.cast %value : memref<16x16xf16, 1> to memref<16x16xf16, 1>
    memref.store %loaded, %alias[%zero, %zero] : memref<16x16xf16, 1>
    memref.store %loaded, %scratch[%zero] : memref<4xf16>
    memref.dealloc %value : memref<16x16xf16, 1>
    memref.dealloc %scratch : memref<4xf16>
    return
  }
}
