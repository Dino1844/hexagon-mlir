//===- l2-prefetch-decline.mlir - every near miss says why ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The decline arms of hexagon-l2-prefetch. Every loop that has the streaming
// shape (a slice a vector read consumes) and then fails a condition explains
// itself with a remark; nothing declines silently here. The deliberate
// silences are pinned too: the store side (a slice no read touches) and the
// scalar reduction accumulator (memref.load, not a vector stream) are out of
// scope, and @store_side_is_silent / @reduction_accumulator_is_silent hold
// them to that.
//
// The 2026-10-05 generalization (row slices, row walks, unroll tiling) moved
// three declines from the per-slice checks to the stream-level ones, so their
// reasons changed wording: "size is not the step" is now "size does not divide
// the step" (an unrolled body can legally have size != step when the copies
// tile it), "source stride is not statically 1" became a check on the slice
// result (a rank-reduced row slice carries the row stride in its result, so
// the result is where unit stride lives), and "vector is not one step wide"
// became "not the slice width" (the vector matches its slice, and the slices
// -- not the vectors -- must tile the step).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-l2-prefetch))' -verify-diagnostics -split-input-file

// The slice is two steps wide but the loop advances one slice per iteration
// and no second copy fills the rest: 32 % 64 != 0, so no tiling exists.
func.func @size_does_not_divide_step(%x: memref<128xf32, strided<[1]>>, %o: memref<128xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  // expected-remark @+1 {{hexagon-l2-prefetch declined: slice size does not divide the loop step}}
  scf.for %iv = %c0 to %c128 step %c32 {
    %sx = memref.subview %x[%iv] [64] [1] : memref<128xf32, strided<[1]>> to memref<64xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>>, vector<64xf32>
  }
  return
}

// -----

// An unrolled body missing its second copy: the step is two slices wide but
// only one is read, so the stream the loop demands has a gap every other
// slice and the calibrated contiguous-block fetch would cover bytes nothing
// demands.
func.func @missing_copy(%x: memref<128xf32, strided<[1]>>, %o: memref<128xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  // expected-remark @+1 {{hexagon-l2-prefetch declined: copies of this source do not tile the loop step}}
  scf.for %iv = %c0 to %c128 step %c64 {
    %sx = memref.subview %x[%iv] [32] [1] : memref<128xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
  }
  return
}

// -----

// A strided slice: the stream is not contiguous, and the calibrated fetch
// (contiguous 2 KiB blocks) would pull the wrong lines. The stride a
// rank-reduced row slice carries lives in its result type, so that is where
// the check fires.
func.func @stride_not_one(%x: memref<256xf32, strided<[2]>>, %o: memref<256xf32, strided<[2]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  scf.for %iv = %c0 to %c128 step %c32 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: slice result is not unit-stride}}
    %sx = memref.subview %x[%iv] [32] [1] : memref<256xf32, strided<[2]>> to memref<32xf32, strided<[2], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[2], offset: ?>>, vector<32xf32>
  }
  return
}

// -----

// A column slice of a row-major source: unit stride in the source's dim 0
// would be needed, but the slice's surviving dim has the row stride, so the
// result is strided and the demand pattern is not contiguous.
func.func @column_slice(%x: memref<64x64xf16, strided<[64, 1]>>, %o: memref<64x64xf16, strided<[64, 1]>>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c64 step %c1 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: slice result is not unit-stride}}
    %sx = memref.subview %x[0, %iv] [64, 1] [1, 1] : memref<64x64xf16, strided<[64, 1]>> to memref<64xf16, strided<[64], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<64xf16, strided<[64], offset: ?>>, vector<64xf16>
  }
  return
}

// -----

// A masked read: the mask changes what the iteration actually reads, and the
// probe never calibrated a fetch pattern for it.
func.func @masked_read(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %m = vector.constant_mask [32] : vector<32xi1>
  scf.for %iv = %c0 to %c131072 step %c32 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: transfer_read is masked}}
    %sx = memref.subview %x[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst, %m {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
  }
  return
}

// -----

// The read vector is narrower than its slice: the loop is not the
// one-slice-per-copy form the tiling predicate is derived from.
func.func @vector_not_slice_wide(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  scf.for %iv = %c0 to %c131072 step %c32 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: transfer vector is not the slice width}}
    %sx = memref.subview %x[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<16xf32>
  }
  return
}

// -----

// A dynamic step: the fetch period and the distance in iterations are element
// counts derived from the step, so it must be a positive constant.
func.func @dynamic_step(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>, %step: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  // expected-remark @+1 {{hexagon-l2-prefetch declined: step is not a positive constant}}
  scf.for %iv = %c0 to %c131072 step %step {
    %sx = memref.subview %x[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
  }
  return
}

// -----

// The store side: a slice no read consumes is out of scope, not a near miss.
// No expected-remark here -- if one fires, -verify-diagnostics fails this
// test, which is the pin.
func.func @store_side_is_silent(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %z = arith.constant dense<0.000000e+00> : vector<32xf32>
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  scf.for %iv = %c0 to %c131072 step %c32 {
    %so = memref.subview %o[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    vector.transfer_write %z, %so[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
  }
  return
}

// -----

// A VTCM source: it has the full streaming shape, but VTCM is not behind the
// L2-miss path this pass works around -- fetching it is wasted fetch slots at
// best. Declined by remark, not silently, because it is a near miss.
func.func @vtcm_source(%x: memref<131072xf32, strided<[1]>, 1>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  scf.for %iv = %c0 to %c131072 step %c32 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: source is not in the default (DDR) address space}}
    %sx = memref.subview %x[%iv] [32] [1] : memref<131072xf32, strided<[1]>, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
  }
  return
}

// -----

// A padded (non-dense) source: the row stride (512) exceeds the row length
// (128), so a linear fetch could straddle the gap bytes between the
// subview's rows, which nothing proves backed. Dense row-major only.
func.func @nondense_source(%x: memref<256x128xf32, strided<[512, 1]>>, %row: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  scf.for %iv = %c0 to %c128 step %c128 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: source is not a dense row-major memref}}
    %sx = memref.subview %x[%row, %iv] [1, 128] [1, 1] : memref<256x128xf32, strided<[512, 1]>> to memref<128xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<128xf32, strided<[1], offset: ?>>, vector<128xf32>
  }
  return
}

// -----

// A partial row walk: the loop walks rows (the IV is the row index) but the
// slice covers only half of each 64-element row, so the stream advances 64
// elements per copy while the reads cover 32 -- a strided demand the
// calibrated contiguous fetch does not model.
func.func @partial_row_walk(%x: memref<512x64xf16, strided<[64, 1]>>, %o: memref<512x64xf16, strided<[64, 1]>>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-remark @+1 {{hexagon-l2-prefetch declined: slice width does not cover the loop's per-iteration advance}}
  scf.for %iv = %c0 to %c512 step %c1 {
    %sx = memref.subview %x[%iv, 0] [1, 32] [1, 1] : memref<512x64xf16, strided<[64, 1]>> to memref<32xf16, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf16, strided<[1], offset: ?>>, vector<32xf16>
  }
  return
}

// -----

// The induction variable in two dimensions at once: a diagonal stream, which
// a single linear fetch cannot track.
func.func @two_iv_dims(%x: memref<256x512xf16, strided<[512, 1]>>, %o: memref<256x512xf16, strided<[512, 1]>>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  scf.for %iv = %c0 to %c512 step %c128 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: offset carries the induction variable in more than one dimension}}
    %sx = memref.subview %x[%iv, %iv] [1, 128] [1, 1] : memref<256x512xf16, strided<[512, 1]>> to memref<128xf16, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[1], offset: ?>>, vector<128xf16>
  }
  return
}

// -----

// The induction variable scaled instead of offset: the stream advances a
// multiple of the IV per iteration, which the address arithmetic does not
// emit.
func.func @scaled_offset(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c2 = arith.constant 2 : index
  scf.for %iv = %c0 to %c131072 step %c32 {
    %iv2 = arith.muli %iv, %c2 : index
    // expected-remark @+1 {{hexagon-l2-prefetch declined: slice offset is not the induction variable plus a constant}}
    %sx = memref.subview %x[%iv2] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
  }
  return
}

// -----

// A slice whose offsets never mention the IV: the loop re-reads one position,
// which is not a stream at all.
func.func @offset_invariant(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  scf.for %iv = %c0 to %c131072 step %c32 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: slice offset does not depend on the induction variable}}
    %sx = memref.subview %x[%c64] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
  }
  return
}

// -----

// A stream shorter than the fetch distance plus block: the room guard could
// never fire inside it, so the whole emitter would be dead arithmetic. This
// source is 64x64 f16 = 8 KiB, below the 8 KiB distance + 2 KiB block.
func.func @stream_shorter_than_distance(%x: memref<64x64xf16, strided<[64, 1]>>, %row: index) {
  %cst = arith.constant 0.000000e+00 : f16
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  // expected-remark @+1 {{hexagon-l2-prefetch declined: stream is shorter than the fetch distance plus block}}
  scf.for %iv = %c0 to %c64 step %c64 {
    %sx = memref.subview %x[%row, %iv] [1, 64] [1, 1] : memref<64x64xf16, strided<[64, 1]>> to memref<64xf16, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
  }
  return
}

// -----

// Three copies of 32 with a step of 96: the tiling is complete, but the
// per-iteration advance (96 * 4 B) divides neither the 2 KiB block nor the
// 8 KiB distance, so the calibrated constants have no whole-iteration
// meaning in this loop.
func.func @advance_does_not_divide(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c96 = arith.constant 96 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  // expected-remark @+1 {{hexagon-l2-prefetch declined: iteration advance does not divide the fetch block or distance}}
  scf.for %iv = %c0 to %c131072 step %c96 {
    %sx0 = memref.subview %x[%iv] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %iv32 = arith.addi %iv, %c32 : index
    %sx1 = memref.subview %x[%iv32] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %iv64 = arith.addi %iv, %c64 : index
    %sx2 = memref.subview %x[%iv64] [32] [1] : memref<131072xf32, strided<[1]>> to memref<32xf32, strided<[1], offset: ?>>
    %v0 = vector.transfer_read %sx0[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
    %v1 = vector.transfer_read %sx1[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
    %v2 = vector.transfer_read %sx2[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
  }
  return
}

// -----

// The reduction accumulator silence: the input stream declines (strided
// column slice), and the accumulator -- a scalar memref.load/store of a VTCM
// subview -- says nothing, because a scalar load is not a vector stream.
// Only the input's remark is expected; any remark on the accumulator fails
// this test.
func.func @reduction_accumulator_is_silent(%x: memref<512x512xf32>, %acc: memref<512xf32, 1>, %row: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  scf.for %iv = %c0 to %c512 step %c64 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: slice result is not unit-stride}}
    %sx = memref.subview %x[%row, 0] [512, 1] [1, 1] : memref<512x512xf32> to memref<512xf32, strided<[512], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<512xf32, strided<[512], offset: ?>>, vector<512xf32>
    %sa = memref.subview %acc[%row] [1] [1] : memref<512xf32, 1> to memref<f32, strided<[], offset: ?>, 1>
    %z = memref.load %sa[] : memref<f32, strided<[], offset: ?>, 1>
    %r = vector.reduction <add>, %v, %z : vector<512xf32> into f32
    memref.store %r, %sa[] : memref<f32, strided<[], offset: ?>, 1>
  }
  return
}
