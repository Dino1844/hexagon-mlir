//===- l2-prefetch.mlir - l2fetch insertion for streaming slice loops -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The positive arms of hexagon-l2-prefetch on the vectorizer's slice-loop
// form. The decline arms are in l2-prefetch-decline.mlir.
//
// The two shapes here are the two the pipeline actually produces, taken from
// the post-ConvertBufferizationToMemRef dump of each:
//
//   * @flat_two_streams is the flat elementwise kernel (vec_add/silu/gelu):
//     the subviews slice the function-argument views directly, one vector
//     per iteration.
//   * @grid_strided_slice is the grid-strided kernel (the gap-table ceiling
//     arm): the loop reads a slice whose base offset is a runtime value, so
//     the fetch address must add the runtime offset, not a constant.
//
// The constants in the CHECK lines are the calibration results pinned in the
// pass source (exp/hmx/streaming_bw_probe): distance 8192 B, block 2048 B,
// control word (2048<<32) | (2048<<16) | 1 = 8796227239937 -- the 64-bit Rtt
// layout stride[47:32] | width[31:16] | height[15:0] (V79 PRM; a zero
// subfield cancels prefetches instead of fetching). A change to any of
// them should turn a CHECK here, which is the point of checking them.
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-l2-prefetch))' | FileCheck %s

module {

// CHECK-LABEL: func.func @flat_two_streams
func.func @flat_two_streams(%x: memref<131072xf32, strided<[1]>>, %y: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  // One hoisted metadata pair per stream, before the loop.
  // CHECK: memref.extract_strided_metadata %{{.+}} : memref<131072xf32, strided<[1]>>
  // CHECK: memref.extract_aligned_pointer_as_index
  // CHECK: memref.extract_strided_metadata %{{.+}} : memref<131072xf32, strided<[1]>>
  // CHECK: memref.extract_aligned_pointer_as_index
  // The guard: one fetch point per 512 elements (16 iterations of 32), and
  // room for the whole fetch block 2560 elements (80 iterations) ahead.
  // CHECK: scf.for
  // CHECK: arith.constant 512 : index
  // CHECK: arith.remsi
  // CHECK: arith.constant 2560 : index
  // CHECK: scf.if
  // The fetch: 8192 B ahead (64 vectors), one 2048 B block, width = stride =
  // 2048 and height 1 in the control word.
  // CHECK: arith.constant 4 : i64
  // CHECK: arith.constant 8192 : i64
  // CHECK: arith.constant 8796227239937 : i64
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // The store side is out of scope: exactly two streams, not three.
  // CHECK-NOT: llvm.call @llvm.hexagon.Y5.l2fetch
  scf.for %iv = %c0 to %c131072 step %c32 {
    %sx = memref.subview %x[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %sy = memref.subview %y[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %so = memref.subview %o[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %vx = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
    %vy = vector.transfer_read %sy[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
    %s = arith.addf %vx, %vy : vector<32xf32>
    vector.transfer_write %s, %so[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
  }
  return
}

// The grid-strided slice: the loop's stream starts at a runtime offset (a
// program's slice of the whole tensor), so the fetch address must add the
// extract_strided_metadata offset at runtime. f16 halves the element count:
// period 1024, room 5120, element size 2.
// CHECK-LABEL: func.func @grid_strided_slice
func.func @grid_strided_slice(%x: memref<16777216xf16>, %y: memref<16777216xf16>, %o: memref<16777216xf16>, %base: index) {
  %cst = arith.constant 0.000000e+00 : f16
  %c8192 = arith.constant 8192 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %sx0 = memref.reinterpret_cast %x to offset: [%base], sizes: [8192], strides: [1] : memref<16777216xf16> to memref<8192xf16, strided<[1], offset: ?>>
  %sy0 = memref.reinterpret_cast %y to offset: [%base], sizes: [8192], strides: [1] : memref<16777216xf16> to memref<8192xf16, strided<[1], offset: ?>>
  %so0 = memref.reinterpret_cast %o to offset: [%base], sizes: [8192], strides: [1] : memref<16777216xf16> to memref<8192xf16, strided<[1], offset: ?>>
  // The pass inserts ops, and the printer drops parsed SSA names for a
  // function it touched, so the CHECKs pin shapes and constants rather than
  // names: two metadata pairs (one per stream, f16), then the guard.
  // CHECK: memref.extract_strided_metadata %{{.+}} : memref<8192xf16, strided<[1], offset: ?>> -> memref<f16>, index, index, index
  // CHECK: memref.extract_strided_metadata %{{.+}} : memref<8192xf16, strided<[1], offset: ?>> -> memref<f16>, index, index, index
  // CHECK: scf.for
  // CHECK: arith.constant 1024 : index
  // CHECK: arith.constant 5120 : index
  // CHECK: scf.if
  // CHECK: arith.constant 2 : i64
  // CHECK: arith.constant 8192 : i64
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // CHECK-NOT: llvm.call @llvm.hexagon.Y5.l2fetch
  scf.for %iv = %c0 to %c8192 step %c64 {
    %sx = memref.subview %sx0[%iv] [64] [1] : memref<8192xf16, strided<[1], offset: ?>> to memref<64xf16, strided<[1], offset: ?>>
    %sy = memref.subview %sy0[%iv] [64] [1] : memref<8192xf16, strided<[1], offset: ?>> to memref<64xf16, strided<[1], offset: ?>>
    %so = memref.subview %so0[%iv] [64] [1] : memref<8192xf16, strided<[1], offset: ?>> to memref<64xf16, strided<[1], offset: ?>>
    %vx = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
    %vy = vector.transfer_read %sy[%c0], %cst {in_bounds = [true]} : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
    %s = arith.addf %vx, %vy : vector<64xf16>
    vector.transfer_write %s, %so[%c0] {in_bounds = [true]} : vector<64xf16>, memref<64xf16, strided<[1], offset: ?>>
  }
  return
}

// The intrinsic declaration lands at the end of the module, once, private:
// it is not part of the kernel's ABI.
// CHECK: llvm.func @llvm.hexagon.Y5.l2fetch(!llvm.ptr, i64) attributes {sym_visibility = "private"}

} // module
