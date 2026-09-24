//===- tail-plan.mlir - HMX static tail plan contract -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tail_plan
// CHECK: hmx.matmul
// CHECK-SAME: tail_plan = #hmx.tail_plan<logical = [65, 47, 70], padded = [96, 64, 96], full = [64, 32, 64], tail = [1, 15, 6], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">
func.func @tail_plan(%lhs: tensor<3x3x16x32x2xf16>,
                     %rhs: tensor<2x3x16x32x2xf16>,
                     %out: tensor<3x2x16x32x2xf16>) -> tensor<3x2x16x32x2xf16> {
  %result = hmx.matmul ins(%lhs, %rhs : tensor<3x3x16x32x2xf16>, tensor<2x3x16x32x2xf16>)
      outs(%out : tensor<3x2x16x32x2xf16>)
      {tail_plan = #hmx.tail_plan<logical = [65, 47, 70], padded = [96, 64, 96], full = [64, 32, 64], tail = [1, 15, 6], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">}
      -> tensor<3x2x16x32x2xf16>
  return %result : tensor<3x2x16x32x2xf16>
}

// -----

// A fully aligned plan has no peeled edge and still uses the same attribute.
// CHECK-LABEL: func.func @aligned_plan
// CHECK: hmx.matmul
// CHECK-SAME: tail_plan = #hmx.tail_plan<logical = [64, 32, 64], padded = [64, 32, 64], full = [64, 32, 64], tail = [0, 0, 0], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">
func.func @aligned_plan(%lhs: tensor<2x2x16x32x2xf16>,
                        %rhs: tensor<1x2x16x32x2xf16>,
                        %out: tensor<2x1x16x32x2xf16>) -> tensor<2x1x16x32x2xf16> {
  %result = hmx.matmul ins(%lhs, %rhs : tensor<2x2x16x32x2xf16>, tensor<1x2x16x32x2xf16>)
      outs(%out : tensor<2x1x16x32x2xf16>)
      {tail_plan = #hmx.tail_plan<logical = [64, 32, 64], padded = [64, 32, 64], full = [64, 32, 64], tail = [0, 0, 0], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">}
      -> tensor<2x1x16x32x2xf16>
  return %result : tensor<2x1x16x32x2xf16>
}
