//===- l2-prefetch-decline.mlir - every near miss says why ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The decline arms of hexagon-l2-prefetch. Every loop that has the streaming
// shape (a slice at the induction variable that a read consumes) and then
// fails a condition explains itself with a remark; nothing declines silently
// here. The one deliberate silence is the store side: a slice no read touches
// is out of scope, and @store_side_is_silent pins that no remark fires for it.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-l2-prefetch))' -verify-diagnostics -split-input-file

// The slice is two steps wide but the loop advances one vector per iteration:
// the fetch period would not line up with the reads.
func.func @size_not_step(%x: memref<128xf32, strided<[1]>>, %o: memref<128xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  scf.for %iv = %c0 to %c128 step %c32 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: subview size is not the loop step}}
    %sx = memref.subview %x[%iv] [64] [1] : memref<128xf32, strided<[1]>> to memref<64xf32, strided<[1], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>>, vector<64xf32>
  }
  return
}

// -----

// A strided slice: the stream is not contiguous, and the calibrated fetch
// (contiguous 2 KiB blocks) would pull the wrong lines.
func.func @stride_not_one(%x: memref<256xf32, strided<[2]>>, %o: memref<256xf32, strided<[2]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  scf.for %iv = %c0 to %c128 step %c32 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: source stride is not statically 1}}
    %sx = memref.subview %x[%iv] [32] [1] : memref<256xf32, strided<[2]>> to memref<32xf32, strided<[2], offset: ?>>
    %v = vector.transfer_read %sx[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[2], offset: ?>>, vector<32xf32>
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

// The read vector is not one step wide: the loop is not the one-vector-per-
// iteration form the period arithmetic is derived from.
func.func @vector_not_step_wide(%x: memref<131072xf32, strided<[1]>>, %o: memref<131072xf32, strided<[1]>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c131072 = arith.constant 131072 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  scf.for %iv = %c0 to %c131072 step %c32 {
    // expected-remark @+1 {{hexagon-l2-prefetch declined: transfer vector is not one step wide}}
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
