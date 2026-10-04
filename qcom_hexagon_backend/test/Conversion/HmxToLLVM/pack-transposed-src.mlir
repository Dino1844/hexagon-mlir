//===- pack-transposed-src.mlir - the _T leaf family for K^T sources ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A `src_transposed` `hmx.pack_weight` reads the [N, K] transpose input
// directly: the leaf family is the `_T` twins (HMXAPI.h) and the source
// extents swap -- K is dim 1, the contiguous crouton-pair axis, and N dim 0.
// `src_stride` keeps the source's own row stride contract: the K width for a
// dense source, wider for a strided view.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// Dense [N, K] = [64, 128]: k = 128 (dim 1), n = 64 (dim 0), and the dense row
// stride is the K width, so the 5th argument reuses the k constant.
// pack_weight is HmxLayoutHvx, so this function touches no engine instruction
// and takes no lock.
// CHECK-LABEL: func.func @pack_weight_transposed_dense
// CHECK-NOT: llvm.call @hexagon_runtime_hmx_ensure_dsp
// CHECK-NOT: llvm.call @hexagon_runtime_hmx_unlock_dsp
// CHECK: %[[K:.*]] = llvm.mlir.constant(128 : i32)
// CHECK: %[[N:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: llvm.call @hmx_pack_weight_f16_T({{.*}}, {{.*}}, %[[K]], %[[N]], %[[K]], {{.*}}, {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @pack_weight_transposed_dense(%wsrc: memref<64x128xf16>,
                                        %wt: memref<2x4x16x32x2xf16, 1>,
                                        %kt: index, %nt: index) {
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x128xf16>)
      outs(%wt : memref<2x4x16x32x2xf16, 1>) {src_transposed}
  return
}

// -----

// Strided [N, K] view (an N block of a wider Wᵀ store): rows are 512 elements
// apart while only 128 wide, so `src_stride` (the 5th argument) must be 512,
// not 128.
// CHECK-LABEL: func.func @pack_weight_transposed_strided
// CHECK: %[[K:.*]] = llvm.mlir.constant(128 : i32)
// CHECK: %[[N:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[S:.*]] = llvm.mlir.constant(512 : i32)
// CHECK: llvm.call @hmx_pack_weight_f16_T({{.*}}, {{.*}}, %[[K]], %[[N]], %[[S]], {{.*}}, {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @pack_weight_transposed_strided(%wsrc: memref<64x128xf16, strided<[512, 1]>>,
                                          %wt: memref<2x4x16x32x2xf16, 1>,
                                          %kt: index, %nt: index) {
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x128xf16, strided<[512, 1]>>)
      outs(%wt : memref<2x4x16x32x2xf16, 1>) {src_transposed}
  return
}

// -----

// The ranged form: one `_T_bulk` call covers `count` consecutive K tiles, the
// transposed twin of the row-major bulk leaf. The 5th argument is again the
// dense K width (the k constant, 128).
// CHECK-LABEL: func.func @pack_weight_transposed_bulk
// CHECK: %[[K:.*]] = llvm.mlir.constant(128 : i32)
// CHECK: %[[N:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: llvm.call @hmx_pack_weight_f32_T_bulk({{.*}}, {{.*}}, %[[K]], %[[N]], %[[K]], {{.*}}, {{.*}}, {{.*}}) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @pack_weight_transposed_bulk(%wsrc: memref<64x128xf32>,
                                       %wt: memref<2x4x16x32x2xf16, 1>,
                                       %kt: index, %nt: index) {
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x128xf32>)
      outs(%wt : memref<2x4x16x32x2xf16, 1>) {count = 2 : i64, src_transposed}
  return
}

// -----

// The f32 single-tile `_T` twin.
// CHECK-LABEL: func.func @pack_weight_transposed_f32
// CHECK: %[[K:.*]] = llvm.mlir.constant(128 : i32)
// CHECK: %[[N:.*]] = llvm.mlir.constant(64 : i32)
// CHECK: llvm.call @hmx_pack_weight_f32_T({{.*}}, {{.*}}, %[[K]], %[[N]], %[[K]], {{.*}}, {{.*}}) : (i32, i32, i32, i32, i32, i32, i32) -> ()
func.func @pack_weight_transposed_f32(%wsrc: memref<64x128xf32>,
                                      %wt: memref<2x4x16x32x2xf16, 1>,
                                      %kt: index, %nt: index) {
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<64x128xf32>)
      outs(%wt : memref<2x4x16x32x2xf16, 1>) {src_transposed}
  return
}
