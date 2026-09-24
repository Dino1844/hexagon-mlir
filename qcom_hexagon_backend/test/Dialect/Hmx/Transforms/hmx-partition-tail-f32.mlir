//===- hmx-partition-tail-f32.mlir - f32 tail + residual bridge -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition),hmx-to-llvm)' | FileCheck %s --check-prefix=LLVM
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tail_f32
// CHECK-DAG: hmx.pack_act
// CHECK-DAG: hmx.pack_weight
// CHECK-DAG: hmx.unpack_acc_f32
// CHECK-DAG: valid_rows = 1 : i64
// CHECK-DAG: valid_cols = 1 : i64
// CHECK-NOT: hmx.matmul
// LLVM-DAG: llvm.func @hmx_pack_act_tail_f32(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVM-DAG: llvm.func @hmx_pack_weight_tail_f32(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVM-DAG: llvm.func @hmx_unpack_acc_tail_f32(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVM-DAG: llvm.call @hmx_pack_act_tail_f32
// LLVM-DAG: llvm.call @hmx_pack_weight_tail_f32
// LLVM-DAG: llvm.call @hmx_unpack_acc_tail_f32
func.func @tail_f32(%a: memref<33x33xf32>, %w: memref<33x33xf32>,
                    %out: memref<33x33xf32>, %res: memref<33x33xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.pack_act ins(%a, %c0, %c0 : memref<33x33xf32>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c0, %c1 : memref<33x33xf32>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c0 : memref<33x33xf32>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c1 : memref<33x33xf32>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c0, %c0 : memref<33x33xf32>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c0, %c1 : memref<33x33xf32>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c1, %c0 : memref<33x33xf32>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c1, %c1 : memref<33x33xf32>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.matmul ins(%act, %wt : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}
  hmx.unpack_acc_f32 ins(%ar, %c0, %c0, %res : memref<2x2x16x32x2xf16, 1>, memref<33x33xf32>)
      outs(%out : memref<33x33xf32>)
  hmx.unpack_acc_f32 ins(%ar, %c0, %c1, %res : memref<2x2x16x32x2xf16, 1>, memref<33x33xf32>)
      outs(%out : memref<33x33xf32>)
  hmx.unpack_acc_f32 ins(%ar, %c1, %c0, %res : memref<2x2x16x32x2xf16, 1>, memref<33x33xf32>)
      outs(%out : memref<33x33xf32>)
  hmx.unpack_acc_f32 ins(%ar, %c1, %c1, %res : memref<2x2x16x32x2xf16, 1>, memref<33x33xf32>)
      outs(%out : memref<33x33xf32>)
  return
}
