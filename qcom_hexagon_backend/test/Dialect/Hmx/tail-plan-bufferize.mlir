//===- tail-plan-bufferize.mlir - tail plan survives bufferization -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries})' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tail_plan_buffers(
// CHECK-SAME: %arg0: memref<3x3x16x32x2xf16
// CHECK-SAME: %arg1: memref<2x3x16x32x2xf16
// CHECK-SAME: %arg2: memref<3x2x16x32x2xf16
// CHECK: hmx.matmul ins({{.*}}memref<3x3x16x32x2xf16
// CHECK-SAME: tail_plan = #hmx.tail_plan<logical = [65, 47, 70], padded = [96, 64, 96], full = [64, 32, 64], tail = [1, 15, 6], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">
func.func @tail_plan_buffers(%a: tensor<3x3x16x32x2xf16>,
                             %b: tensor<2x3x16x32x2xf16>,
                             %c: tensor<3x2x16x32x2xf16>) -> tensor<3x2x16x32x2xf16> {
  %0 = hmx.matmul ins(%a, %b : tensor<3x3x16x32x2xf16>, tensor<2x3x16x32x2xf16>)
      outs(%c : tensor<3x2x16x32x2xf16>)
      {tail_plan = #hmx.tail_plan<logical = [65, 47, 70], padded = [96, 64, 96], full = [64, 32, 64], tail = [1, 15, 6], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">}
      -> tensor<3x2x16x32x2xf16>
  return %0 : tensor<3x2x16x32x2xf16>
}
