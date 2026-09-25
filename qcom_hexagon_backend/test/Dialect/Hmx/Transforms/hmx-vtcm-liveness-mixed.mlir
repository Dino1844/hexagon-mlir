//===- hmx-vtcm-liveness-mixed.mlir - transient plus resident floor --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The modeled requested bound keeps the transient live range separate from
// the process-resident floor.  This is not an allocator/charged or observed
// pool value.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "mixed_resident_floor"
// CHECK-DAG: transient_requested_peak_bytes = 8192 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 1024 : i64
// CHECK-DAG: weight_resident_requested_bytes = 0 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 9216 : i64
// CHECK-DAG: allocator_peak_status = "not-proven"
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @mixed_resident_floor() {
    %transient = memref.alloc() : memref<64x64xf16, 1>
    %workspace = memref.alloc()
        {hmx.workspace_resident = {key = -9 : i64, bytes = 1024 : i64}}
        : memref<32x16xf16, 1>
    memref.dealloc %transient : memref<64x64xf16, 1>
    return
  }
}
