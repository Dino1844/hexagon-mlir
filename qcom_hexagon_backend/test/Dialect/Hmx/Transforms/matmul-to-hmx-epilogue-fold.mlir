//===- matmul-to-hmx-epilogue-fold.mlir - the store eats the fp16 tile -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// An f32 accumulator whose only consumer chain narrows it to f16 for the store
// is an identity round trip: the read-out already produced the f16 image, so
// widening it and narrowing it back moves three times the tile through DDR for
// nothing (measured: 2048^3-B, docs/analysis/residual-9ms-attribution-
// 2026-10-09.md section 3.1). The epilogue therefore hands the fp16 tile
// straight to the boundary and drops both casts.
//
// The gate is a mechanism, not a shape list: every use of the result must be a
// pure conversion link (a float-cast `linalg.generic`, or the `tensor.extract_slice`
// a masked store addresses) reaching a `materialize_in_destination`, and there
// must be no residual, because the C term is added after the read-out and
// adding it to the fp16 tile instead of the widened image changes the rounding.
// Each refusal below keeps the widening epilogue byte for byte.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' | FileCheck %s
//===----------------------------------------------------------------------===//

// A store of the narrowed image: the tile is the store's operand, and neither
// the widening nor the narrowing `linalg.generic` survives.
// CHECK-LABEL: func.func @narrow_store(
// CHECK: hmx.unpack_acc
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
func.func @narrow_store(%a: memref<32x128xf16>, %b: memref<128x128xf16>,
                        %c: memref<32x128xf16>) {
  %cf = arith.constant 0.000000e+00 : f32
  %at = bufferization.to_tensor %a restrict writable
      : memref<32x128xf16> to tensor<32x128xf16>
  %wt = bufferization.to_tensor %b restrict writable
      : memref<128x128xf16> to tensor<128x128xf16>
  %z = tensor.empty() : tensor<32x128xf32>
  %init = linalg.fill ins(%cf : f32) outs(%z : tensor<32x128xf32>)
      -> tensor<32x128xf32>
  %m = linalg.matmul ins(%at, %wt : tensor<32x128xf16>, tensor<128x128xf16>)
      outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
  %ot = tensor.empty() : tensor<32x128xf16>
  %r = linalg.generic {
         indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}
       ins(%m : tensor<32x128xf32>) outs(%ot : tensor<32x128xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<32x128xf16>
  bufferization.materialize_in_destination %r in writable %c
      : (tensor<32x128xf16>, memref<32x128xf16>) -> ()
  return
}

// A masked store addresses a sub-region: the slice is a link, so the tile
// reaches the store through it and the casts still go.
// CHECK-LABEL: func.func @masked_store(
// CHECK: hmx.unpack_acc
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: tensor.extract_slice
// CHECK: bufferization.materialize_in_destination
func.func @masked_store(%a: memref<32x128xf16>, %b: memref<128x128xf16>,
                       %c: memref<32x128xf16>, %rows: index) {
  %cf = arith.constant 0.000000e+00 : f32
  %at = bufferization.to_tensor %a restrict writable
      : memref<32x128xf16> to tensor<32x128xf16>
  %wt = bufferization.to_tensor %b restrict writable
      : memref<128x128xf16> to tensor<128x128xf16>
  %z = tensor.empty() : tensor<32x128xf32>
  %init = linalg.fill ins(%cf : f32) outs(%z : tensor<32x128xf32>)
      -> tensor<32x128xf32>
  %m = linalg.matmul ins(%at, %wt : tensor<32x128xf16>, tensor<128x128xf16>)
      outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
  %ot = tensor.empty() : tensor<32x128xf16>
  %r = linalg.generic {
         indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}
       ins(%m : tensor<32x128xf32>) outs(%ot : tensor<32x128xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<32x128xf16>
  %slice = tensor.extract_slice %r[0, 0] [%rows, 128] [1, 1]
      : tensor<32x128xf16> to tensor<?x128xf16>
  %dst = memref.subview %c[0, 0] [%rows, 128] [1, 1]
      : memref<32x128xf16> to memref<?x128xf16, strided<[128, 1]>>
  bufferization.materialize_in_destination %slice in writable %dst
      : (tensor<?x128xf16>, memref<?x128xf16, strided<[128, 1]>>) -> ()
  return
}

// A residual is added after the read-out, so it keeps the wide form: folding
// it into the fp16 tile would change the rounding.
// CHECK-LABEL: func.func @residual_add(
// CHECK: hmx.unpack_acc
// CHECK: arith.extf
// CHECK: arith.addf
// CHECK: arith.truncf
func.func @residual_add(%a: memref<32x128xf16>, %b: memref<128x128xf16>,
                        %c: memref<32x128xf16>, %bias: tensor<32x128xf32>) {
  %at = bufferization.to_tensor %a restrict writable
      : memref<32x128xf16> to tensor<32x128xf16>
  %wt = bufferization.to_tensor %b restrict writable
      : memref<128x128xf16> to tensor<128x128xf16>
  %m = linalg.matmul ins(%at, %wt : tensor<32x128xf16>, tensor<128x128xf16>)
      outs(%bias : tensor<32x128xf32>) -> tensor<32x128xf32>
  %ot = tensor.empty() : tensor<32x128xf16>
  %r = linalg.generic {
         indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}
       ins(%m : tensor<32x128xf32>) outs(%ot : tensor<32x128xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<32x128xf16>
  bufferization.materialize_in_destination %r in writable %c
      : (tensor<32x128xf16>, memref<32x128xf16>) -> ()
  return
}

// A wider boundary reads the f32 image itself, so the widening is not dead:
// this is the store to an f32 output.
// CHECK-LABEL: func.func @wide_store(
// CHECK: hmx.unpack_acc
// CHECK: arith.extf
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: arith.truncf
func.func @wide_store(%a: memref<32x128xf16>, %b: memref<128x128xf16>,
                      %c: memref<32x128xf32>) {
  %cf = arith.constant 0.000000e+00 : f32
  %at = bufferization.to_tensor %a restrict writable
      : memref<32x128xf16> to tensor<32x128xf16>
  %wt = bufferization.to_tensor %b restrict writable
      : memref<128x128xf16> to tensor<128x128xf16>
  %z = tensor.empty() : tensor<32x128xf32>
  %init = linalg.fill ins(%cf : f32) outs(%z : tensor<32x128xf32>)
      -> tensor<32x128xf32>
  %m = linalg.matmul ins(%at, %wt : tensor<32x128xf16>, tensor<128x128xf16>)
      outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
  bufferization.materialize_in_destination %m in writable %c
      : (tensor<32x128xf32>, memref<32x128xf32>) -> ()
  return
}

// A consumer that computes on the result is not a link: what it produces is no
// longer the engine's fp16 image of the accumulator, so the widening stays.
// CHECK-LABEL: func.func @computed_consumer(
// CHECK: hmx.unpack_acc
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: arith.truncf
func.func @computed_consumer(%a: memref<32x128xf16>, %b: memref<128x128xf16>,
                             %c: memref<32x128xf16>, %scale: tensor<32x128xf32>) {
  %cf = arith.constant 0.000000e+00 : f32
  %at = bufferization.to_tensor %a restrict writable
      : memref<32x128xf16> to tensor<32x128xf16>
  %wt = bufferization.to_tensor %b restrict writable
      : memref<128x128xf16> to tensor<128x128xf16>
  %z = tensor.empty() : tensor<32x128xf32>
  %init = linalg.fill ins(%cf : f32) outs(%z : tensor<32x128xf32>)
      -> tensor<32x128xf32>
  %m = linalg.matmul ins(%at, %wt : tensor<32x128xf16>, tensor<128x128xf16>)
      outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
  %scaled = linalg.generic {
                indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                 affine_map<(d0, d1) -> (d0, d1)>,
                                 affine_map<(d0, d1) -> (d0, d1)>],
                iterator_types = ["parallel", "parallel"]}
              ins(%m, %scale : tensor<32x128xf32>, tensor<32x128xf32>)
              outs(%z : tensor<32x128xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %p = arith.mulf %in, %s : f32
    linalg.yield %p : f32
  } -> tensor<32x128xf32>
  %ot = tensor.empty() : tensor<32x128xf16>
  %r = linalg.generic {
         indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}
       ins(%scaled : tensor<32x128xf32>) outs(%ot : tensor<32x128xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<32x128xf16>
  bufferization.materialize_in_destination %r in writable %c
      : (tensor<32x128xf16>, memref<32x128xf16>) -> ()
  return
}
