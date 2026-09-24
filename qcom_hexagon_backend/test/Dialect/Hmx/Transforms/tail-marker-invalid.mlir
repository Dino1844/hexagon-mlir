//===- tail-marker-invalid.mlir - diagnostic marker type gate ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -verify-diagnostics
//===----------------------------------------------------------------------===//

func.func @invalid_tail_marker(%a: tensor<33x33xf16>,
                                %b: tensor<33x33xf16>) -> tensor<33x33xf16> {
  %c = tensor.empty() : tensor<33x33xf16>
  // expected-error @+1 {{hmx.diagnostic_tail_partition must be a unit attribute}}
  %0 = linalg.matmul ins(%a, %b : tensor<33x33xf16>, tensor<33x33xf16>)
      outs(%c : tensor<33x33xf16>)
      {hmx.diagnostic_tail_partition = "not-a-unit-marker"}
      -> tensor<33x33xf16>
  return %0 : tensor<33x33xf16>
}
