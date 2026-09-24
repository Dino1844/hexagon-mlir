//===- hmx-partition-tail-serial.mlir - full + peeled edge lowering -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This is the diagnostic peeled-edge slice. It starts from direct memref
// bridges, rebuilds their pack/unpack sites with explicit valid extents, and
// emits the full M/N rectangle plus the three disjoint edge rectangles. The
// production tail attribution, looped bridge scheduling, and launcher gates
// remain separate closed paths.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition),hmx-to-llvm)' | FileCheck %s --check-prefix=LLVM
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tail_partition
// CHECK: hmx.bias_init
// CHECK-DAG: hmx.pack_act
// CHECK-DAG: hmx.pack_act
// CHECK-DAG: hmx.pack_act
// CHECK-DAG: hmx.pack_act
// CHECK-DAG: hmx.pack_weight
// CHECK-DAG: hmx.pack_weight
// CHECK-DAG: hmx.pack_weight
// CHECK-DAG: hmx.pack_weight
// Four accumulator regions, each with one K-loop mma and one read-out.
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.acc_read
// CHECK-DAG: hmx.acc_read
// CHECK-DAG: hmx.acc_read
// CHECK-DAG: hmx.acc_read
// CHECK-DAG: hmx.mma
// CHECK-DAG: hmx.mma
// CHECK-DAG: hmx.mma
// CHECK-DAG: hmx.mma
// CHECK-DAG: hmx.unpack_acc
// CHECK-NOT: hmx.matmul

// LLVM-DAG: llvm.func @hmx_pack_act_tail_f16(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVM-DAG: llvm.func @hmx_pack_weight_tail_f16(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVM-DAG: llvm.func @hmx_unpack_acc_tail_f16(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// LLVM-DAG: llvm.call @hmx_pack_act_tail_f16
// LLVM-DAG: llvm.call @hmx_pack_weight_tail_f16
// LLVM-DAG: llvm.call @hmx_unpack_acc_tail_f16
func.func @tail_partition(%a: memref<33x33xf16>, %w: memref<33x33xf16>,
                          %out: memref<33x33xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>

  // Direct source bridges are retired and rebuilt by hmx-partition.
  hmx.pack_act ins(%a, %c0, %c0 : memref<33x33xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c0, %c1 : memref<33x33xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c0 : memref<33x33xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c1 : memref<33x33xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c0, %c0 : memref<33x33xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c0, %c1 : memref<33x33xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c1, %c0 : memref<33x33xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c1, %c1 : memref<33x33xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)

  hmx.matmul ins(%act, %wt : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}

  // Direct output bridge; the pass replaces these sites with full/edge loops.
  hmx.unpack_acc ins(%ar, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  hmx.unpack_acc ins(%ar, %c0, %c1 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  hmx.unpack_acc ins(%ar, %c1, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  hmx.unpack_acc ins(%ar, %c1, %c1 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  return
}
