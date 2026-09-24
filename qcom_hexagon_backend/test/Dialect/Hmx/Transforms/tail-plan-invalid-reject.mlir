//===- tail-plan-invalid-reject.mlir - stale plan arithmetic fails closed --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' -verify-diagnostics
//===----------------------------------------------------------------------===//

func.func @reject_invalid_tail_plan(%a: memref<2x2x16x32x2xf16, 1>,
                                     %w: memref<2x2x16x32x2xf16, 1>,
                                     %r: memref<2x2x16x32x2xf16, 1>) {
  // expected-error @+3 {{logical must equal full + tail at axis 0}}
  hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%r : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [0, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}
  return
}
