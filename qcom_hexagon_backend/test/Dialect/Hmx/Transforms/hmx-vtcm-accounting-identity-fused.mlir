//===- hmx-vtcm-accounting-identity-fused.mlir - fused source identity -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A debug string in a fused location is metadata, not a second source fact.
// When there is exactly one file/line/column component, the compiler can
// preserve it as the stable site identity.  Two file components remain
// ambiguous and are covered by the negative fixture.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: source = "file:fused:12:1"
// CHECK-DAG: source_file = "fused"
// CHECK-DAG: source_line = 12 : i64
// CHECK-DAG: source_column = 1 : i64
module @fused_identity attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity} {
  func.func @fused_source() {
    %a = memref.alloc() : memref<16x16xf16, 1>
        loc(fused["allocation", "fused":12:1])
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}
