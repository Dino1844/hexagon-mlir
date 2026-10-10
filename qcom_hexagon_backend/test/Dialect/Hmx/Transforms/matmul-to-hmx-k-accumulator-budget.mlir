//===- matmul-to-hmx-k-accumulator-budget.mlir - the accumulator's budget ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The resident accumulator is charged the room beside the engine's read-out,
// on top of whatever the function already holds: the read-out (f16, m*n*2) plus
// the census term, against the budget
// (docs/hmx/kchunk-accumulator-residency-2026-10-10.md section 3). It is an
// f32 matrix, so it is twice the read-out's bytes; when the room does not cover
// it the pass declines -- loudly, with the two numbers -- and the contraction
// keeps the epilogue it had. No shape is forced into the residency and no block
// is silently mis-charged.
//
// Here a 64x64x32 contraction: its bridge is 16384 bytes, so the whole-contraction
// attribution succeeds at the 16385-byte budget, but the accumulator needs 16384
// bytes against the 8193 left beside the read-out -- 16385 - 8192 - 0 - 0.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=16385}))' -verify-diagnostics -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1) -> (d0, d1)>

// The decline says what it needs and what was free, so the number is
// reproducible from the shape rather than being a tuned threshold.
module {
func.func @k_loop_tight_budget(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
  %z = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<64x64xf32>
  %zero = linalg.fill ins(%z : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %acc = scf.for %k = %c0 to %c64 step %c32 iter_args(%carried = %zero) -> (tensor<64x64xf32>) {
    %as = tensor.extract_slice %a[0, %k] [64, 32] [1, 1] : tensor<64x64xf16> to tensor<64x32xf16>
    %bs = tensor.extract_slice %b[%k, 0] [32, 64] [1, 1] : tensor<64x64xf16> to tensor<32x64xf16>
    // expected-remark @+1 {{K-loop f32 accumulator not made VTCM-resident: it needs 16384 bytes and only 8193 are free beside the engine's read-out (vtcmBudget=16385 bytes)}}
    %mm = linalg.matmul ins(%as, %bs : tensor<64x32xf16>, tensor<32x64xf16>)
                        outs(%zero : tensor<64x64xf32>) -> tensor<64x64xf32>
    %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%carried, %mm : tensor<64x64xf32>, tensor<64x64xf32>) outs(%carried : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in2: f32, %out: f32):
      %s = arith.addf %in, %in2 : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    scf.yield %sum : tensor<64x64xf32>
  }
  return %acc : tensor<64x64xf32>
}
}

// -----

// The refusal is a decline, not a removal: the contraction is still attributed
// (the bridge fits) and the epilogue is the old one -- no resident array, and
// the fp16 read-out still travels through DDR.
// CHECK-LABEL: func.func @k_loop_tight_budget
// CHECK-DAG: hmx.matmul
// CHECK-DAG: hmx.unpack_acc ins(
// CHECK-DAG: arith.extf
// CHECK-DAG: arith.addf
// CHECK-NOT: bufferization.alloc_tensor
// CHECK-NOT: hmx.unpack_acc_f32
