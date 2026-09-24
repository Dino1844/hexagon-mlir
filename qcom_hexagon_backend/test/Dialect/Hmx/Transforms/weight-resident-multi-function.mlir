//===- weight-resident-multi-function.mlir - module state is aggregated ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Two sibling functions exercise the nested-pass module-state update. The
// resident byte aggregate must be the sum of both function-local contracts.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' | FileCheck %s
// CHECK: hmx.weight_prepack =
// CHECK: hmx.weight_resident_bytes = 16384 : i64
// CHECK-LABEL: func.func @weight_a
// CHECK: hmx.weight_resident
// CHECK-LABEL: func.func @weight_b
// CHECK: hmx.weight_resident
module {
  func.func @weight_a(%a: memref<64x64xf16>, %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %aa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%aa, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%ar : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %aa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
    return
  }

  func.func @weight_b(%a: memref<64x64xf16>, %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %aa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%aa, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%ar : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %aa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
    return
  }
}
