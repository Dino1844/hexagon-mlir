//===- tail-source-dominance-reject.mlir - no non-dominating bridge use ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' -verify-diagnostics
//===----------------------------------------------------------------------===//

func.func @tail_source_after_matmul() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %out = memref.alloc() : memref<33x33xf16>

  // expected-error @+1 {{diagnostic tail partition requires pack sources that dominate the matmul}}
  hmx.matmul ins(%act, %wt : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}

  // The row-major source is defined only after the matmul. The diagnostic
  // rewrite must reject it rather than capture it in a newly inserted pack.
  %a = memref.alloc() : memref<64x64xf16>
  %w = memref.alloc() : memref<64x64xf16>
  hmx.pack_act ins(%a, %c0, %c0 : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c0, %c1 : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c0 : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c1 : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c0, %c0 : memref<64x64xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c0, %c1 : memref<64x64xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c1, %c0 : memref<64x64xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c1, %c1 : memref<64x64xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.unpack_acc ins(%ar, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  return
}
