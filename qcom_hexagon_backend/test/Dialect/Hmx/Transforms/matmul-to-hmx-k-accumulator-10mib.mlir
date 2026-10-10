//===- matmul-to-hmx-k-accumulator-10mib.mlir - the device-default budget ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The 1024³ tile of a K-looped contraction, against the device's own budget
// (no `vtcm-budget` option here, so the default is what production reads):
// activation croutons 1024*1024*2 = 2 MiB, weight croutons the same, the
// engine's read-out the same and the f32 accumulator 1024*1024*4 = 4 MiB.
// That is 10 MiB + 256 B of simultaneous residency against an 8 MiB pool --
// the state the loader refuses on device
// (docs/hmx/kchunk-depth-exchange-preanalysis-2026-10-10.md section 0.2: every
// form requesting ~10 MiB dies loading, while the whole-K forms at <= 6.25 MiB
// run). The residency is therefore refused here, and the contraction keeps the
// epilogue it had before the residency existed -- the f32 accumulator in DDR,
// read and written once per K segment.
//
// The second case is the control: the same kernel at half the tile asks for
// 2.5 MiB and keeps its resident accumulator. What decides is the sum, so no
// shape is ever the thing being special-cased.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -verify-diagnostics -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1) -> (d0, d1)>

// The 10 MiB class: 1024-row tiles, K walked in 1024-wide segments. The
// bridge itself still fits (its plan is 6 MiB), so the contraction is
// attributed and only the residency is refused.
module {
func.func @k_loop_1024_tile(%a: tensor<1024x2048xf16>, %b: tensor<2048x1024xf16>) -> tensor<1024x1024xf32> {
  %z = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<1024x1024xf32>
  %zero = linalg.fill ins(%z : f32) outs(%e : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %acc = scf.for %k = %c0 to %c2048 step %c1024 iter_args(%carried = %zero) -> (tensor<1024x1024xf32>) {
    %as = tensor.extract_slice %a[0, %k] [1024, 1024] [1, 1] : tensor<1024x2048xf16> to tensor<1024x1024xf16>
    %bs = tensor.extract_slice %b[%k, 0] [1024, 1024] [1, 1] : tensor<2048x1024xf16> to tensor<1024x1024xf16>
    // expected-remark @+1 {{K-loop f32 accumulator not made VTCM-resident: the function would hold 10485760 bytes of VTCM against the 8388608-byte budget (committed 0, resident accumulators 0, activation croutons 2097152, weight croutons 2097152, read-out 2097152, this accumulator 4194304)}}
    %mm = linalg.matmul ins(%as, %bs : tensor<1024x1024xf16>, tensor<1024x1024xf16>)
                        outs(%zero : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%carried, %mm : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%carried : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in2: f32, %out: f32):
      %s = arith.addf %in, %in2 : f32
      linalg.yield %s : f32
    } -> tensor<1024x1024xf32>
    scf.yield %sum : tensor<1024x1024xf32>
  }
  return %acc : tensor<1024x1024xf32>
}
}

// The default budget is the device's own, and the epilogue is the pre-residency
// one: the fp16 read-out into a DDR image, widened, then added into the
// accumulator the loop still carries.
// CHECK-LABEL: func.func @k_loop_1024_tile
// CHECK-DAG: vtcm_budget_bytes = 8388608 : i64
// CHECK-DAG: hmx.matmul
// CHECK-DAG: hmx.unpack_acc ins(
// CHECK-DAG: arith.extf
// CHECK-DAG: arith.addf
// CHECK-NOT: bufferization.alloc_tensor
// CHECK-NOT: hmx.unpack_acc_f32

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// The control: the same kernel at half the tile -- 512 rows, 512-wide
// segments -- holds 524288 activation + 524288 weight + 524288 read-out +
// 1048576 accumulator = 2621440 bytes, well inside the pool, so the residency
// stands. The shape did not change the answer; the sum did.
module {
func.func @k_loop_512_tile(%a: tensor<512x1024xf16>, %b: tensor<1024x512xf16>) -> tensor<512x512xf32> {
  %z = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<512x512xf32>
  %zero = linalg.fill ins(%z : f32) outs(%e : tensor<512x512xf32>) -> tensor<512x512xf32>
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %c1 = arith.constant 1 : index
  %acc = scf.for %k = %c0 to %c1024 step %c512 iter_args(%carried = %zero) -> (tensor<512x512xf32>) {
    %as = tensor.extract_slice %a[0, %k] [512, 512] [1, 1] : tensor<512x1024xf16> to tensor<512x512xf16>
    %bs = tensor.extract_slice %b[%k, 0] [512, 512] [1, 1] : tensor<1024x512xf16> to tensor<512x512xf16>
    %mm = linalg.matmul ins(%as, %bs : tensor<512x512xf16>, tensor<512x512xf16>)
                        outs(%zero : tensor<512x512xf32>) -> tensor<512x512xf32>
    %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%carried, %mm : tensor<512x512xf32>, tensor<512x512xf32>) outs(%carried : tensor<512x512xf32>) {
    ^bb0(%in: f32, %in2: f32, %out: f32):
      %s = arith.addf %in, %in2 : f32
      linalg.yield %s : f32
    } -> tensor<512x512xf32>
    scf.yield %sum : tensor<512x512xf32>
  }
  return %acc : tensor<512x512xf32>
}
}

// CHECK-LABEL: func.func @k_loop_512_tile
// CHECK-DAG: bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<512x512xf32>
// CHECK-DAG: hmx.unpack_acc_f32 ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[acc:[a-z0-9_]+]] : tensor<16x16x16x32x2xf16>, tensor<512x512xf32>) outs(%[[acc]] : tensor<512x512xf32>)
// CHECK-NOT: hmx.unpack_acc ins(
// CHECK-NOT: arith.extf
