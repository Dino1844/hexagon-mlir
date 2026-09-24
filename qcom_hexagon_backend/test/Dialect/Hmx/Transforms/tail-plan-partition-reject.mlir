//===- tail-plan-partition-reject.mlir - no silent tail-plan drop -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' -verify-diagnostics
//===----------------------------------------------------------------------===//

func.func @reject_tail_plan() {
  %a = memref.alloc() : memref<3x3x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x3x16x32x2xf16, 1>
  %r = memref.alloc() : memref<3x2x16x32x2xf16, 1>
  // expected-error @+1 {{tail_plan is not yet supported by hmx-partition}}
  hmx.matmul ins(%a, %w : memref<3x3x16x32x2xf16, 1>, memref<2x3x16x32x2xf16, 1>)
      outs(%r : memref<3x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [65, 47, 70], padded = [96, 64, 96], full = [64, 32, 64], tail = [1, 15, 6], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">}
  return
}
