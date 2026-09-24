//===- hmx-partition-tail-looped.mlir - looped bridge diagnostic path ---===//
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

// CHECK-LABEL: func.func @tail_looped
// CHECK-DAG: hmx.pack_act
// CHECK-DAG: hmx.pack_weight
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.unpack_acc
// CHECK-DAG: valid_rows = 1 : i64
// CHECK-DAG: valid_cols = 1 : i64
// CHECK-NOT: hmx.matmul
// LLVM-DAG: llvm.call @hmx_pack_act_tail_f16
// LLVM-DAG: llvm.call @hmx_pack_weight_tail_f16
// LLVM-DAG: llvm.call @hmx_unpack_acc_tail_f16
func.func @tail_looped(%a: memref<33x33xf16>, %w: memref<33x33xf16>,
                        %out: memref<33x33xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>

  scf.for %i = %c0 to %c4 step %c1 {
    %m = arith.divui %i, %c2 : index
    %k = arith.remui %i, %c2 : index
    hmx.pack_act ins(%a, %m, %k : memref<33x33xf16>)
        outs(%act : memref<2x2x16x32x2xf16, 1>)
  }
  scf.for %i = %c0 to %c4 step %c1 {
    %k = arith.divui %i, %c2 : index
    %n = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %k, %n : memref<33x33xf16>)
        outs(%wt : memref<2x2x16x32x2xf16, 1>)
  }
  hmx.matmul ins(%act, %wt : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}
  scf.for %i = %c0 to %c2 step %c1 {
    %m = arith.divui %i, %c1 : index
    %n = arith.remui %i, %c1 : index
    hmx.unpack_acc ins(%ar, %m, %n : memref<2x2x16x32x2xf16, 1>)
        outs(%out : memref<33x33xf16>)
  }
  return
}
