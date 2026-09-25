//===- hmx-vtcm-liveness-aliases.mlir - view alias liveness ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A view is an alias of the principal allocation.  Reading through the view
// before releasing the principal is lifetime-safe; releasing the view itself
// remains an explicit negative case in hmx-vtcm-liveness-rejects.mlir.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "view_alias"
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
module @view_alias_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @view_alias() {
    %a = memref.alloc() : memref<512xi8, 1>
    %c0 = arith.constant 0 : index
    %view = memref.view %a[%c0][]
        : memref<512xi8, 1> to memref<256xi8, 1>
    %value = memref.load %view[%c0] : memref<256xi8, 1>
    memref.dealloc %a : memref<512xi8, 1>
    return
  }
}
