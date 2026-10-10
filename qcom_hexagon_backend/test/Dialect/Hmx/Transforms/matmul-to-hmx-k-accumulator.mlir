//===- matmul-to-hmx-k-accumulator.mlir - the K-loop resident accumulator ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A contraction inside a K loop carries its f32 accumulator across the loop's
// segments. Without this rewrite every segment pays for the read-out as a DDR
// round trip: unpack the fp16 image, widen it, read-modify-write the
// accumulator and copy it back (docs/hmx/kchunk-accumulator-residency-2026-10-10.md
// section 2). With it the accumulator is a VTCM array the loop carries, the
// fused leaf adds the read-out into it in place, and only the loop's final
// value is read out by the post-loop consumers.
//
// Two spellings of the same region are covered: the accumulate form (the
// contraction's init is the loop-invariant zero-fill and a separate add
// generic accumulates into the carried value -- what Triton emits when it
// hoists `tl.zeros` above the loop) and the region form (the contraction's own
// init is the carried value). A contraction with no K loop around it keeps the
// epilogue it had.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' | FileCheck %s
//===----------------------------------------------------------------------===//

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @k_loop_accumulate
// The accumulator is a VTCM array beside the engine's read-out: the stock
// memory-space allocation, not the dialect's crouton allocation (which is an
// f16 crouton array's allocation point).
// CHECK-DAG: bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<64x64xf32>
// The loop starts from nothing, so the array is zero-filled at loop entry
// (write-only, where the copy it replaces also read the matrix).
// CHECK-DAG: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// The loop now carries that array.
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<64x64xf32>)
// The fused leaf writes it in place: the residual IS the destination, which is
// the contract that makes one buffer serve every segment.
// CHECK-DAG: hmx.unpack_acc_f32 ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[acc:[a-z0-9_]+]] : tensor<2x2x16x32x2xf16>, tensor<64x64xf32>) outs(%[[acc]] : tensor<64x64xf32>)
// Nothing else per segment: no fp16 image, no widen, no separate add (the
// accumulate the loop carried is the leaf's residual now).
// CHECK-NOT: hmx.unpack_acc ins(
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.addf
func.func @k_loop_accumulate(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                             %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
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

// -----

// The region form: the contraction's own init is the carried accumulator, so
// there is no separate add to fold and the C term is the loop's init value.
// Same resident form, same in-place leaf.
// CHECK-LABEL: func.func @k_loop_region
// CHECK-DAG: bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<64x64xf32>
// CHECK-DAG: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-DAG: hmx.unpack_acc_f32 ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[acc:[a-z0-9_]+]] : tensor<2x2x16x32x2xf16>, tensor<64x64xf32>) outs(%[[acc]] : tensor<64x64xf32>)
// CHECK-NOT: hmx.unpack_acc ins(
// CHECK-NOT: arith.extf
func.func @k_loop_region(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                         %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
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
    %mm = linalg.matmul ins(%as, %bs : tensor<64x32xf16>, tensor<32x64xf16>)
                        outs(%carried : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %mm : tensor<64x64xf32>
  }
  return %acc : tensor<64x64xf32>
}

// -----

// No K loop around the contraction: the epilogue is the one it always had.
// The resident accumulator is a K-loop capability only -- an accumulator is a
// value a loop carries, so there is nothing to make resident without one.
// CHECK-LABEL: func.func @no_k_loop
// CHECK-DAG: hmx.unpack_acc ins(
// CHECK-DAG: arith.extf
// CHECK-DAG: arith.addf
// CHECK-NOT: hmx.unpack_acc_f32
// CHECK-NOT: bufferization.alloc_tensor
func.func @no_k_loop(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                     %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %e = tensor.empty() : tensor<64x64xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%c, %mm : tensor<64x64xf32>, tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in2: f32, %out: f32):
    %s = arith.addf %in, %in2 : f32
    linalg.yield %s : f32
  } -> tensor<64x64xf32>
  return %sum : tensor<64x64xf32>
}
