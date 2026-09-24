//===- matmul-to-hmx-diagnostic-tail-f32.mlir - producer f32 tail ABI ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The same internal producer guard is used for a wider source/result. The
// engine still reads f16 croutons; the f32 result is widened after bounded
// f16 read-out, so this fixture guards both producer contracts.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx),one-shot-bufferize{bufferize-function-boundaries},func.func(hmx-partition),hmx-to-llvm)' | FileCheck %s --check-prefix=E2E
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @diagnostic_tail_f32
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK-COUNT-2: tensor.pad
// CHECK: hmx.pack_act
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK-SAME: hmx.diagnostic_tail_partition
// CHECK-SAME: tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1]
// CHECK: hmx.unpack_acc
// CHECK: arith.extf
// E2E-DAG: llvm.call @hmx_pack_act_tail_f32
// E2E-DAG: llvm.call @hmx_pack_weight_tail_f32
// E2E-DAG: llvm.call @hmx_unpack_acc_tail_f16
// E2E-DAG: reason = "tail-peeled-edge"
// E2E-NOT: hmx.matmul
func.func @diagnostic_tail_f32(%a: tensor<33x33xf32>,
                               %b: tensor<33x33xf32>) -> tensor<33x33xf32> {
  %c = tensor.empty() : tensor<33x33xf32>
  %0 = linalg.matmul ins(%a, %b : tensor<33x33xf32>, tensor<33x33xf32>)
      outs(%c : tensor<33x33xf32>) {hmx.diagnostic_tail_partition}
      -> tensor<33x33xf32>
  return %0 : tensor<33x33xf32>
}
