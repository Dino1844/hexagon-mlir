//===- tail-bridge-loop-reject.mlir - looped tail bridges fail closed ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' -verify-diagnostics
//===----------------------------------------------------------------------===//

func.func private @opaque()

func.func @reject_looped_bridge(%a: memref<33x33xf16>,
                                %w: memref<33x33xf16>,
                                %out: memref<33x33xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c2 step %c1 {
    func.call @opaque() : () -> ()
    hmx.pack_act ins(%a, %c0, %c0 : memref<33x33xf16>)
        outs(%act : memref<2x2x16x32x2xf16, 1>)
  }
  scf.for %j = %c0 to %c2 step %c1 {
    hmx.pack_weight ins(%w, %j, %j : memref<33x33xf16>)
        outs(%wt : memref<2x2x16x32x2xf16, 1>)
  }
  // expected-error @+1 {{diagnostic tail partition requires direct activation and weight pack bridges}}
  hmx.matmul ins(%act, %wt : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}
  hmx.unpack_acc ins(%ar, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  return
}
