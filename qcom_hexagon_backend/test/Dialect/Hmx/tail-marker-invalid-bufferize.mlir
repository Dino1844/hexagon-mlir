//===- tail-marker-invalid-bufferize.mlir - marker survives strictly -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries})' -verify-diagnostics
//===----------------------------------------------------------------------===//

func.func @invalid_bufferize_marker(%a: tensor<2x2x16x32x2xf16>,
                                    %b: tensor<2x2x16x32x2xf16>)
    -> tensor<2x2x16x32x2xf16> {
  %r = tensor.empty() : tensor<2x2x16x32x2xf16>
  // expected-error @+2 {{hmx.diagnostic_tail_partition must be a unit attribute}}
  // expected-error @+1 {{failed to bufferize op}}
  %0 = hmx.matmul ins(%a, %b : tensor<2x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>)
      outs(%r : tensor<2x2x16x32x2xf16>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition = "bad"}
      -> tensor<2x2x16x32x2xf16>
  return %0 : tensor<2x2x16x32x2xf16>
}
