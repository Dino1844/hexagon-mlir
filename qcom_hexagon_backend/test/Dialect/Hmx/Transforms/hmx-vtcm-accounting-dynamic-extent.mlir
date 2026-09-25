//===- hmx-vtcm-accounting-dynamic-extent.mlir - proven extent bound -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A dynamic-typed memref is admitted only when every dynamic extent has a
// compile-time constant bound, because a constant bound on an extent *is* that
// extent and the byte product is therefore exact.  `extent_status` records
// which proof was used so a reader never mistakes a dynamic type carrying a
// constant bound for a statically shaped allocation.
//
// The negative counterparts (a runtime argument, a computed cap, a non-positive
// constant) are in hmx-vtcm-accounting-reject.mlir.
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The liveness record still proves the balanced pair, and the identity sidecar
// marks the size as resting on a constant upper bound.
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: unknown_allocations = 0 : i64
// CHECK-DAG: transient_bytes = 1024 : i64
// CHECK-DAG: extent_status = "constant-upper-bound"
// CHECK-DAG: extent_status = "static-shape"
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: extent_policy = "static-shape-or-proven-constant-upper-bound"
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "constant_bounded_extent"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: constant_bounded_extent_sites = 1 : i64
// CHECK-DAG: transient_requested_peak_bytes = 1024 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 1024 : i64
// CHECK-DAG: peak_site_count = 2 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @constant_bounded_extent() {
    %rows = arith.constant 16 : index
    %dynamic = memref.alloc(%rows) : memref<?x16xf16, 1>
    %static = memref.alloc() : memref<16x16xf16, 1>
    memref.dealloc %dynamic : memref<?x16xf16, 1>
    memref.dealloc %static : memref<16x16xf16, 1>
    return
  }
}
