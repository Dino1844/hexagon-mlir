//===- l2-prefetch.mlir - l2fetch insertion for streaming slice loops -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The positive arms of hexagon-l2-prefetch on the vectorizer's slice-loop
// forms. The decline arms are in l2-prefetch-decline.mlir.
//
// The shapes here are the ones the pipeline actually produces, taken from
// the post-ConvertBufferizationToMemRef dump of each:
//
//   * @flat_two_streams is the flat elementwise kernel (vec_add/silu/gelu):
//     the subviews slice the function-argument views directly, one vector
//     per iteration.
//   * @grid_strided_slice is the grid-strided kernel (the gap-table ceiling
//     arm): the loop reads a slice whose base offset is a runtime value, so
//     the fetch address must add the runtime offset, not a constant.
//   * @row_slice / @row_slice_unrolled are the 2-D row slices (linattn's
//     masked loop, softmax's row loops): the loop walks the contiguous last
//     dimension of a dense source while the row offset is a runtime value
//     (an enclosing loop's IV in the real kernels).
//   * @row_walk is the whole-row walk (linattn's output norm): the IV is the
//     row index and the slice spans the full row, so the stream advances
//     one row stride per copy.
//   * @reduction_input is a reduction loop's input stream (linattn's
//     rowsum): the accumulator is a VTCM scalar the pass never touches, the
//     input is a DDR stream it covers.
//   * @flat_unrolled is the flat form under the loop unroller: two copies of
//     half a step each tile the step, so one stream serves both.
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

// The 2-D row slice: the loop walks the contiguous (last) dimension of a
// dense rank-2 source; the row offset is a runtime value (an enclosing
// loop's IV in the real kernels). The hoisted row term is one muli by the
// row stride; the guard's room is measured against the source's full linear
// extent (131072 f16 elements = 256x512), because the stream crosses the
// inner loop's rows. One stream, one fetch: the store-side slice of the
// output says nothing.
// CHECK-LABEL: func.func @row_slice
func.func @row_slice(%x: memref<256x512xf16, strided<[512, 1], offset: ?>>, %o: memref<256x512xf16, strided<[512, 1], offset: ?>>, %row: index) {
  %cst = arith.constant 0.000000e+00 : f16
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: memref.extract_strided_metadata %{{.+}} : memref<256x512xf16, strided<[512, 1], offset: ?>>
  // CHECK: memref.extract_aligned_pointer_as_index
  // The row term of the base position: row * row-stride, hoisted.
  // CHECK: arith.constant 512 : index
  // CHECK: arith.muli
  // The window-entry guard: fire at the first stream position inside each
  // 1024-element window (the position row*512 + iv lands in
  // [k*1024, k*1024+128) once per window -- every other row for a runtime
  // row), room for distance + block (5120) inside the 131072-element extent.
  // CHECK: scf.for
  // CHECK: arith.constant 1024 : index
  // CHECK: arith.remsi
  // CHECK: arith.constant 128 : index
  // CHECK: arith.cmpi slt
  // CHECK: arith.constant 131072 : index
  // CHECK: arith.constant 5120 : index
  // CHECK: scf.if
  // The fetch: (source offset + row*512 + iv) elements * 2 B + 8192 B.
  // CHECK: arith.constant 2 : i64
  // CHECK: arith.constant 8192 : i64
  // CHECK: arith.constant 8796227239937 : i64
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // CHECK-NOT: llvm.call @llvm.hexagon.Y5.l2fetch
  scf.for %iv = %c0 to %c512 step %c128 {
    %sx = memref.subview %x[%row, %iv] [1, 128] [1, 1] : memref<256x512xf16, strided<[512, 1], offset: ?>> to memref<128xf16, strided<[1], offset: ?>>
    %so = memref.subview %o[%row, %iv] [1, 128] [1, 1] : memref<256x512xf16, strided<[512, 1], offset: ?>> to memref<128xf16, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[1], offset: ?>>, vector<128xf16>
    vector.transfer_write %v, %so[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, strided<[1], offset: ?>>
  }
  return
}

// The unrolled row slice (linattn's masked loop verbatim): two copies of
// half the step each, at %iv and %iv + 128. They tile the step, so they are
// ONE stream and ONE fetch covers both -- the fetch targets the stream
// position, not any single copy.
// CHECK-LABEL: func.func @row_slice_unrolled
func.func @row_slice_unrolled(%x: memref<256x512xf16, strided<[512, 1], offset: ?>>, %o: memref<256x512xf16, strided<[512, 1], offset: ?>>, %row: index) {
  %cst = arith.constant 0.000000e+00 : f16
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: arith.muli
  // CHECK: scf.for
  // CHECK: arith.constant 1024 : index
  // CHECK: arith.remsi
  // CHECK: arith.constant 256 : index
  // CHECK: arith.cmpi slt
  // CHECK: arith.constant 131072 : index
  // CHECK: arith.constant 5120 : index
  // CHECK: scf.if
  // CHECK: arith.constant 8796227239937 : i64
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // CHECK-NOT: llvm.call @llvm.hexagon.Y5.l2fetch
  scf.for %iv = %c0 to %c512 step %c256 {
    %sx0 = memref.subview %x[%row, %iv] [1, 128] [1, 1] : memref<256x512xf16, strided<[512, 1], offset: ?>> to memref<128xf16, strided<[1], offset: ?>>
    %iv128 = arith.addi %iv, %c128 : index
    %sx1 = memref.subview %x[%row, %iv128] [1, 128] [1, 1] : memref<256x512xf16, strided<[512, 1], offset: ?>> to memref<128xf16, strided<[1], offset: ?>>
    %so = memref.subview %o[%row, %iv] [1, 256] [1, 1] : memref<256x512xf16, strided<[512, 1], offset: ?>> to memref<256xf16, strided<[1], offset: ?>>
    %v0 = vector.transfer_read %sx0[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[1], offset: ?>>, vector<128xf16>
    %v1 = vector.transfer_read %sx1[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[1], offset: ?>>, vector<128xf16>
    %s = arith.addf %v0, %v1 fastmath<fast> : vector<128xf16>
    vector.transfer_write %s, %so[%c0] {in_bounds = [true]} : vector<128xf16>, memref<256xf16, strided<[1], offset: ?>>
  }
  return
}

// The whole-row walk (linattn's output norm): the IV is the row index and
// each slice spans the full 64-element row, so sigma (the stream advance per
// IV unit) is the row stride 64. The unrolled body (step 2, copies at %iv
// and %iv + 1) tiles the step. The window-entry guard fires when iv*64
// first enters each 1024-element window (iv % 16 in {0, 1}: every 16th IV,
// every 8th iteration of this step-2 loop) and the fetch address scales the
// IV by 64; the extent is the whole 512x64 source.
// CHECK-LABEL: func.func @row_walk
func.func @row_walk(%x: memref<512x64xf16>, %o: memref<512x64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: scf.for
  // The stream position: iv * 64.
  // CHECK: arith.constant 64 : index
  // CHECK: arith.muli
  // The window-entry guard: remsi(lin, 1024) < 128 (one fire per 1024
  // elements of advance), then the room term against the 32768-element
  // extent (5120 = distance + block).
  // CHECK: arith.constant 1024 : index
  // CHECK: arith.remsi
  // CHECK: arith.constant 128 : index
  // CHECK: arith.cmpi slt
  // CHECK: arith.constant 32768 : index
  // CHECK: arith.constant 5120 : index
  // CHECK: scf.if
  // CHECK: arith.constant 8796227239937 : i64
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // CHECK-NOT: llvm.call @llvm.hexagon.Y5.l2fetch
  scf.for %iv = %c0 to %c512 step %c2 {
    %sx0 = memref.subview %x[%iv, 0] [1, 64] [1, 1] : memref<512x64xf16> to memref<64xf16, strided<[1], offset: ?>>
    %iv1 = arith.addi %iv, %c1 : index
    %sx1 = memref.subview %x[%iv1, 0] [1, 64] [1, 1] : memref<512x64xf16> to memref<64xf16, strided<[1], offset: ?>>
    %so0 = memref.subview %o[%iv, 0] [1, 64] [1, 1] : memref<512x64xf16> to memref<64xf16, strided<[1], offset: ?>>
    %so1 = memref.subview %o[%iv1, 0] [1, 64] [1, 1] : memref<512x64xf16> to memref<64xf16, strided<[1], offset: ?>>
    %v0 = vector.transfer_read %sx0[%c0], %cst {in_bounds = [true]} : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
    %v1 = vector.transfer_read %sx1[%c0], %cst {in_bounds = [true]} : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
    vector.transfer_write %v0, %so0[%c0] {in_bounds = [true]} : vector<64xf16>, memref<64xf16, strided<[1], offset: ?>>
    vector.transfer_write %v1, %so1[%c0] {in_bounds = [true]} : vector<64xf16>, memref<64xf16, strided<[1], offset: ?>>
  }
  return
}

// A reduction loop's input stream (linattn's rowsum): the accumulator is a
// VTCM scalar consumed by memref.load/store, which is not a vector stream
// and stays untouched; the input is a DDR stream that gets the fetch. f32
// doubles the element count of every guard constant.
// CHECK-LABEL: func.func @reduction_input
func.func @reduction_input(%x: memref<512x512xf32>, %acc: memref<512xf32, 1>, %row: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  // CHECK: arith.muli
  // CHECK: scf.for
  // CHECK: arith.constant 512 : index
  // CHECK: arith.remsi
  // CHECK: arith.constant 64 : index
  // CHECK: arith.cmpi slt
  // CHECK: arith.constant 262144 : index
  // CHECK: arith.constant 2560 : index
  // CHECK: scf.if
  // CHECK: arith.constant 4 : i64
  // CHECK: arith.constant 8192 : i64
  // CHECK: arith.constant 8796227239937 : i64
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // CHECK-NOT: llvm.call @llvm.hexagon.Y5.l2fetch
  scf.for %iv = %c0 to %c512 step %c64 {
    %sx0 = memref.subview %x[%row, %iv] [1, 32] [1, 1] : memref<512x512xf32> to memref<32xf32, strided<[1], offset: ?>>
    %iv32 = arith.addi %iv, %c32 : index
    %sx1 = memref.subview %x[%row, %iv32] [1, 32] [1, 1] : memref<512x512xf32> to memref<32xf32, strided<[1], offset: ?>>
    %sa = memref.subview %acc[%row] [1] [1] : memref<512xf32, 1> to memref<f32, strided<[], offset: ?>, 1>
    %v0 = vector.transfer_read %sx0[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
    %z0 = memref.load %sa[] : memref<f32, strided<[], offset: ?>, 1>
    %r0 = vector.reduction <add>, %v0, %z0 : vector<32xf32> into f32
    memref.store %r0, %sa[] : memref<f32, strided<[], offset: ?>, 1>
    %v1 = vector.transfer_read %sx1[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
    %z1 = memref.load %sa[] : memref<f32, strided<[], offset: ?>, 1>
    %r1 = vector.reduction <add>, %v1, %z1 : vector<32xf32> into f32
    memref.store %r1, %sa[] : memref<f32, strided<[], offset: ?>, 1>
  }
  return
}

// The flat form under the loop unroller: two copies of 32 elements tile a
// step of 64. Not the flat form's single copy, so the linear emitter takes
// it: same address arithmetic, but the room guard measures the source's
// extent (131072) instead of the loop bound. One stream, one fetch.
// CHECK-LABEL: func.func @flat_unrolled
func.func @flat_unrolled(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  // CHECK: scf.for
  // CHECK: arith.constant 512 : index
  // CHECK: arith.remsi
  // CHECK: arith.constant 64 : index
  // CHECK: arith.cmpi slt
  // CHECK: arith.constant 131072 : index
  // CHECK: arith.constant 2560 : index
  // CHECK: scf.if
  // CHECK: arith.constant 8796227239937 : i64
  // CHECK: llvm.call @llvm.hexagon.Y5.l2fetch
  // CHECK-NOT: llvm.call @llvm.hexagon.Y5.l2fetch
  scf.for %iv = %c0 to %c131072 step %c64 {
    %sx0 = memref.subview %x[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %iv32 = arith.addi %iv, %c32 : index
    %sx1 = memref.subview %x[%iv32] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %v0 = vector.transfer_read %sx0[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
    %v1 = vector.transfer_read %sx1[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
    %s = arith.addf %v0, %v1 fastmath<fast> : vector<32xf32>
    vector.transfer_write %s, %sx0[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
  }
  return
}

// The intrinsic declaration lands at the end of the module, once, private:
// it is not part of the kernel's ABI.
// CHECK: llvm.func @llvm.hexagon.Y5.l2fetch(!llvm.ptr, i64) attributes {sym_visibility = "private"}

} // module
