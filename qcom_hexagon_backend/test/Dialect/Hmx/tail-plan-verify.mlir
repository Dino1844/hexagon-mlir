//===- tail-plan-verify.mlir - HMX tail plan verifier negatives ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

// expected-error @+1 {{padded extent must be align_up(logical, 32) at axis 0, expected 96, got 64}}
module attributes {test = #hmx.tail_plan<logical = [65, 64, 64], padded = [64, 64, 64], full = [64, 64, 64], tail = [1, 0, 0], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">} {}

// -----

// expected-error @+1 {{full extent must be a multiple of 32 at axis 1, got 17}}
module attributes {test = #hmx.tail_plan<logical = [65, 47, 70], padded = [96, 64, 96], full = [64, 17, 64], tail = [1, 30, 6], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">} {}

// -----

// expected-error @+1 {{tail extent must be less than 32 at axis 2, got 38}}
module attributes {test = #hmx.tail_plan<logical = [64, 64, 70], padded = [64, 64, 96], full = [64, 64, 32], tail = [0, 0, 38], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">} {}

// -----

// expected-error @+1 {{mn_policy must be 'padded-edge-tile-bounded-store'}}
module attributes {test = #hmx.tail_plan<logical = [64, 64, 64], padded = [64, 64, 64], full = [64, 64, 64], tail = [0, 0, 0], k_policy = "zero-pad-both-operands", mn_policy = "silent-store">} {}

// -----

// The physical grid remains authoritative after encoding erasure.
func.func @bad_unencoded_grid(%a: tensor<3x2x16x32x2xf16>,
                              %b: tensor<2x2x16x32x2xf16>,
                              %c: tensor<3x2x16x32x2xf16>) -> tensor<3x2x16x32x2xf16> {
  // expected-error @+1 {{tail_plan lhs physical grid [3, 2] does not match padded HMX grid [3, 3]}}
  %0 = hmx.matmul ins(%a, %b : tensor<3x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>)
      outs(%c : tensor<3x2x16x32x2xf16>)
      {tail_plan = #hmx.tail_plan<logical = [65, 47, 70], padded = [96, 64, 96], full = [64, 32, 64], tail = [1, 15, 6], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">}
      -> tensor<3x2x16x32x2xf16>
  return %0 : tensor<3x2x16x32x2xf16>
}

// -----

// A plan may not validate only a subset of its encoded operands.
func.func @bad_plan_partial_encoding(%a: tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>,
                                    %b: tensor<2x2x16x32x2xf16>,
                                    %c: tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>)
    -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>> {
  // expected-error @+1 {{tail_plan requires all three operands to carry a crouton logical layout when any one does}}
  %0 = hmx.matmul ins(%a, %b : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>, tensor<2x2x16x32x2xf16>)
      outs(%c : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>)
      {tail_plan = #hmx.tail_plan<logical = [64, 64, 64], padded = [64, 64, 64], full = [64, 64, 64], tail = [0, 0, 0], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">}
      -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
  return %0 : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
}
