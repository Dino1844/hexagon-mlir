//===- hmx-vtcm-liveness-structured.mlir - SCF event analysis -----------===//
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
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: functions = [{
// CHECK-DAG: symbol = "structured_events"
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: transient_requested_peak_bytes = 10240 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 10240 : i64
// CHECK-DAG: deallocation_sites = 3 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @structured_events() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    %cond = arith.constant true
    scf.if %cond {
      %b = memref.alloc() : memref<32x32xf16, 1>
      memref.dealloc %b : memref<32x32xf16, 1>
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %c = memref.alloc() : memref<16x16xf16, 1>
      memref.dealloc %c : memref<16x16xf16, 1>
      scf.yield
    }
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }
}
