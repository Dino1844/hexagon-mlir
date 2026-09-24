//===- matmul-to-hmx-diagnostic-tail-axes.mlir - each tail axis ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx),one-shot-bufferize{bufferize-function-boundaries},func.func(hmx-partition),hmx-to-llvm)' | FileCheck %s --check-prefix=E2E
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tail_m_only
// CHECK: hmx.matmul
// CHECK-SAME: tail_plan = #hmx.tail_plan<logical = [33, 64, 64], padded = [64, 64, 64], full = [32, 64, 64], tail = [1, 0, 0]
// CHECK-LABEL: func.func @tail_n_only
// CHECK: hmx.matmul
// CHECK-SAME: tail_plan = #hmx.tail_plan<logical = [64, 33, 64], padded = [64, 64, 64], full = [64, 32, 64], tail = [0, 1, 0]
// CHECK-LABEL: func.func @tail_k_only
// CHECK: hmx.matmul
// CHECK-SAME: tail_plan = #hmx.tail_plan<logical = [64, 64, 33], padded = [64, 64, 64], full = [64, 64, 32], tail = [0, 0, 1]
// E2E-DAG: llvm.call @hmx_pack_act_tail_f16
// E2E-DAG: llvm.call @hmx_pack_weight_tail_f16
// E2E-DAG: llvm.call @hmx_unpack_acc_tail_f16

func.func @tail_m_only(%a: tensor<33x64xf16>, %b: tensor<64x64xf16>) -> tensor<33x64xf16> {
  %c = tensor.empty() : tensor<33x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<33x64xf16>, tensor<64x64xf16>)
      outs(%c : tensor<33x64xf16>) {hmx.diagnostic_tail_partition}
      -> tensor<33x64xf16>
  return %0 : tensor<33x64xf16>
}

func.func @tail_n_only(%a: tensor<64x64xf16>, %b: tensor<64x33xf16>) -> tensor<64x33xf16> {
  %c = tensor.empty() : tensor<64x33xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x33xf16>)
      outs(%c : tensor<64x33xf16>) {hmx.diagnostic_tail_partition}
      -> tensor<64x33xf16>
  return %0 : tensor<64x33xf16>
}

func.func @tail_k_only(%a: tensor<64x33xf16>, %b: tensor<33x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x33xf16>, tensor<33x64xf16>)
      outs(%c : tensor<64x64xf16>) {hmx.diagnostic_tail_partition}
      -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
