//===- hmx-vtcm-accounting-identity-multi-function.mlir - scoped IDs -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A module may contain several functions, but each site is joined only within
// its explicit function symbol and the module principal.  The sidecar marks
// this as per-function separation; it does not aggregate functions into one
// process identity.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: multi_function_status = "per-function-separated"
// CHECK-DAG: principal = "multi_module"
// CHECK-DAG: function = "first"
// CHECK-DAG: function_id = -{{[1-9][0-9]*}} : i64
// CHECK-DAG: function = "second"
// CHECK-DAG: function_id = -{{[1-9][0-9]*}} : i64
// CHECK-DAG: runtime_join_status = "not-integrated"
// CHECK-DAG: join_status = "not-proven"
// CHECK-DAG: resident_scope_id_status = "not-proven"
// CHECK-DAG: resident_provenance = "separate-from-allocator-site"
module @multi_module attributes {hmx.diagnostic_vtcm_accounting,
                                   hmx.diagnostic_vtcm_identity} {
  func.func @first() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("multi":10:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }

  func.func @second() {
    %b = memref.alloc() : memref<8x8xf16, 1> loc("multi":10:1)
    memref.dealloc %b : memref<8x8xf16, 1>
    return
  }
}
