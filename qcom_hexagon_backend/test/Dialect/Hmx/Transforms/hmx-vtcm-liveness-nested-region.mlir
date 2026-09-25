//===- hmx-vtcm-liveness-nested-region.mlir - execute-region nesting -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A single-block scf.execute_region has the same sequential lifetime semantics
// as a nested scf.if/scf.for body when it carries no VTCM values.  This
// fixture keeps that boundary explicit: the region owns a temporary alias and
// releases it before the enclosing loop continues.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "nested_execute_region"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: transient_requested_peak_bytes = 8704 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 8704 : i64
// CHECK-DAG: peak_site_count = 2 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// CHECK-DAG: control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region"
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @nested_region_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @nested_execute_region() {
    %outer = memref.alloc() : memref<64x64xf16, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.execute_region {
        %inner = memref.alloc() : memref<16x16xf16, 1>
        %alias = memref.cast %inner : memref<16x16xf16, 1>
            to memref<16x16xf16, 1>
        %zero = arith.constant 0 : index
        %value = memref.load %alias[%zero, %zero] : memref<16x16xf16, 1>
        memref.dealloc %inner : memref<16x16xf16, 1>
        scf.yield
      }
      scf.yield
    }
    memref.dealloc %outer : memref<64x64xf16, 1>
    return
  }
}
