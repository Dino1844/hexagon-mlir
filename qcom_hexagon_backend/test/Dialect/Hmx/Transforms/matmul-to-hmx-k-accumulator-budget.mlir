//===- matmul-to-hmx-k-accumulator-budget.mlir - the accumulator's budget ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The resident accumulator is charged the *whole* footprint this function
// would hold with it resident, priced against the same pool the bridge itself
// was planned against: the library's committed bytes, the accumulators this
// pass already made resident, and this attribution's own three arrays -- the
// activation croutons, the weight croutons and the engine's read-out, all live
// beside the accumulator at once (docs/hmx/kchunk-accumulator-residency-2026-10-10.md
// section 3). Those three are exactly the terms `planBridge` reserved, so an
// admission that ignored them would book the pool twice over: the bridge would
// hold its arrays and the accumulator would believe the rest is empty. They
// are counted whole, never as per-tile slots, because the state a host check
// must price is the degraded one -- when activation staging never happens the
// bridge arrays are whole arrays resident for the whole loop.
//
// The admission is a strict `<` on the sum, the boundary `planBridge` refuses
// at, so an exact fit is a refusal. A refusal is a decline: the contraction is
// still attributed (the bridge fits on its own plan) and the epilogue is the
// one it always had -- no resident array, and the fp16 read-out still travels
// through DDR.
//
// Every number below is recomputable from the shape it names.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=32768}))' -verify-diagnostics -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1) -> (d0, d1)>

// Over the line. A 64x64x64 contraction: activation croutons 64*64*2 = 8192,
// weight croutons 64*64*2 = 8192, read-out 64*64*2 = 8192 and the accumulator
// 64*64*4 = 16384 sum to 40960 against the 32768-byte budget. The bridge's
// own plan is the first three of those (24576), so it fits and the residency
// is refused; the remark names every term, so the sum is checkable rather
// than tuned.
module {
func.func @k_loop_over_line(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
  %z = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<64x64xf32>
  %zero = linalg.fill ins(%z : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %acc = scf.for %k = %c0 to %c64 step %c64 iter_args(%carried = %zero) -> (tensor<64x64xf32>) {
    %as = tensor.extract_slice %a[0, %k] [64, 64] [1, 1] : tensor<64x64xf16> to tensor<64x64xf16>
    %bs = tensor.extract_slice %b[%k, 0] [64, 64] [1, 1] : tensor<64x64xf16> to tensor<64x64xf16>
    // expected-remark @+1 {{K-loop f32 accumulator not made VTCM-resident: the function would hold 40960 bytes of VTCM against the 32768-byte budget (committed 0, resident accumulators 0, activation croutons 8192, weight croutons 8192, read-out 8192, this accumulator 16384)}}
    %mm = linalg.matmul ins(%as, %bs : tensor<64x64xf16>, tensor<64x64xf16>)
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

// The refusal is a decline, not a removal: the contraction is still attributed
// (its 24576-byte bridge fits the 32768-byte budget) and the epilogue is the
// old one -- no resident array, and the fp16 read-out still travels through DDR
// before the widening and the add.
// CHECK-LABEL: func.func @k_loop_over_line
// CHECK-DAG: hmx.matmul
// CHECK-DAG: hmx.unpack_acc ins(
// CHECK-DAG: arith.extf
// CHECK-DAG: arith.addf
// CHECK-NOT: bufferization.alloc_tensor
// CHECK-NOT: hmx.unpack_acc_f32

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// An exact fit is still a refusal: the boundary is the strict `<`, so a
// footprint that equals the budget does not pass. 32-wide K segments of the
// same 64x64 tile: 4096 activation + 4096 weight + 8192 read-out + 16384
// accumulator = 32768 against 32768.
module {
func.func @k_loop_exact_fit(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf32> {
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
    // expected-remark @+1 {{K-loop f32 accumulator not made VTCM-resident: the function would hold 32768 bytes of VTCM against the 32768-byte budget (committed 0, resident accumulators 0, activation croutons 4096, weight croutons 4096, read-out 8192, this accumulator 16384)}}
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

// CHECK-LABEL: func.func @k_loop_exact_fit
// CHECK-DAG: hmx.matmul
// CHECK-NOT: bufferization.alloc_tensor
// CHECK-NOT: hmx.unpack_acc_f32

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// The same budget admits a smaller footprint, with the terms weighted as they
// are rather than by shape: a 32x64 tile on 32-wide segments holds 2048
// activation + 4096 weight + 4096 read-out + 8192 accumulator = 18432, which
// fits, so the residency stands. The array is the loop's own iter arg and the
// fused leaf writes it in place with the carried value as residual -- no fp16
// image, no widening, no separate add.
module {
func.func @k_loop_one_block_under(%a: tensor<32x64xf16>, %b: tensor<64x64xf16>) -> tensor<32x64xf32> {
  %z = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<32x64xf32>
  %zero = linalg.fill ins(%z : f32) outs(%e : tensor<32x64xf32>) -> tensor<32x64xf32>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %acc = scf.for %k = %c0 to %c64 step %c32 iter_args(%carried = %zero) -> (tensor<32x64xf32>) {
    %as = tensor.extract_slice %a[0, %k] [32, 32] [1, 1] : tensor<32x64xf16> to tensor<32x32xf16>
    %bs = tensor.extract_slice %b[%k, 0] [32, 64] [1, 1] : tensor<64x64xf16> to tensor<32x64xf16>
    %mm = linalg.matmul ins(%as, %bs : tensor<32x32xf16>, tensor<32x64xf16>)
                        outs(%zero : tensor<32x64xf32>) -> tensor<32x64xf32>
    %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%carried, %mm : tensor<32x64xf32>, tensor<32x64xf32>) outs(%carried : tensor<32x64xf32>) {
    ^bb0(%in: f32, %in2: f32, %out: f32):
      %s = arith.addf %in, %in2 : f32
      linalg.yield %s : f32
    } -> tensor<32x64xf32>
    scf.yield %sum : tensor<32x64xf32>
  }
  return %acc : tensor<32x64xf32>
}
}

// CHECK-LABEL: func.func @k_loop_one_block_under
// CHECK-DAG: bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<32x64xf32>
// CHECK-DAG: hmx.unpack_acc_f32 ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[acc:[a-z0-9_]+]] : tensor<1x2x16x32x2xf16>, tensor<32x64xf32>) outs(%[[acc]] : tensor<32x64xf32>)
// CHECK-NOT: hmx.unpack_acc ins(
// CHECK-NOT: arith.extf
