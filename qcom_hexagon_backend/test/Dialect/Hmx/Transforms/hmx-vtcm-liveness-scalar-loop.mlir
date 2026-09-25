//===- hmx-vtcm-liveness-scalar-loop.mlir - non-VTCM loop carry ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Region-carried scalar/index values are unrelated to VTCM allocation
// lifetime.  The structured analyzer may admit them; a memref/VTCM value in
// the same position remains fail-closed in the reject fixture.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "scalar_loop_carry"
// CHECK-DAG: allocation_sites = 0 : i64
// CHECK-DAG: deallocation_sites = 0 : i64
// CHECK-DAG: transient_requested_peak_bytes = 0 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @scalar_loop_carry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = scf.for %i = %c0 to %c2 step %c1
        iter_args(%carry = %c0) -> (index) {
      scf.yield %carry : index
    }
    return
  }
}
