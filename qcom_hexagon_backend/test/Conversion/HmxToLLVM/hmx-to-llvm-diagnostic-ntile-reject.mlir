//===- hmx-to-llvm-diagnostic-ntile-reject.mlir - n_tile provenance -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// n_tile is semantic diagnostic input. It cannot authorize itself merely by
// appearing on an operation, and neither its integer contract nor the module
// marker may be malformed.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(hmx-to-llvm)'
//===----------------------------------------------------------------------===//

module {
  func.func @unmarked_ntile(
      %src: memref<2x2x16x32x2xf16, 1>, %dst: memref<64x33xf16>) {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{diagnostic n_tile requires module marker hmx.diagnostic_tail_partition}}
    hmx.unpack_acc ins(%src, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
        outs(%dst : memref<64x33xf16>) {count = 16 : i64, n_tile = 0 : i64}
    return
  }
}

// -----

// expected-error @+1 {{hmx.diagnostic_tail_partition must be a unit attribute}}
module attributes {hmx.diagnostic_tail_partition = "bad"} {
  func.func @malformed_marker(
      %src: memref<2x2x16x32x2xf16, 1>, %dst: memref<64x33xf16>) {
    %c0 = arith.constant 0 : index
    hmx.unpack_acc ins(%src, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
        outs(%dst : memref<64x33xf16>) {count = 16 : i64, n_tile = 0 : i64}
    return
  }
}

// -----

module attributes {hmx.diagnostic_tail_partition} {
  func.func @negative_ntile(
      %src: memref<2x2x16x32x2xf16, 1>, %dst: memref<64x33xf16>) {
    %c0 = arith.constant 0 : index
    // expected-error @+2 {{failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
    hmx.unpack_acc ins(%src, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
        outs(%dst : memref<64x33xf16>) {count = 16 : i64, n_tile = -1 : i64}
    return
  }
}

// -----

module attributes {hmx.diagnostic_tail_partition} {
  func.func @wrong_operation(
      %src: memref<64x64xf16>, %dst: memref<2x2x16x32x2xf16, 1>) {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{diagnostic n_tile is only valid on hmx.unpack_acc and hmx.unpack_acc_f32}}
    hmx.pack_act ins(%src, %c0, %c0 : memref<64x64xf16>)
        outs(%dst : memref<2x2x16x32x2xf16, 1>) {n_tile = 0 : i64}
    return
  }
}

// -----

module attributes {hmx.diagnostic_tail_partition} {
  func.func @oversized_ntile(
      %src: memref<2x2x16x32x2xf16, 1>, %dst: memref<64x33xf32>) {
    %c0 = arith.constant 0 : index
    // One past INT32_MAX / (32 columns * 4-byte f32), before any lowering.
    // expected-error @+1 {{diagnostic n_tile exceeds the signed i32 read-out ABI limit}}
    hmx.unpack_acc_f32 ins(%src, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
        outs(%dst : memref<64x33xf32>) {count = 16 : i64, n_tile = 16777216 : i64}
    return
  }
}

// -----

// A partial f32 N tile must use the explicit bounds-safe leaf ABI. Without
// valid extents, the ordinary leaf would be free to read/write a full tile.
module attributes {hmx.diagnostic_tail_partition} {
  func.func @partial_f32_ntile(
      %src: memref<2x2x16x32x2xf16, 1>, %dst: memref<64x33xf32>) {
    %c0 = arith.constant 0 : index
    // expected-error @+2 {{diagnostic f32 n_tile selecting a partial tile requires bounds-safe valid_rows and valid_cols}}
    // expected-error @+1 {{failed to legalize operation 'hmx.unpack_acc_f32'}}
    hmx.unpack_acc_f32 ins(%src, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
        outs(%dst : memref<64x33xf32>) {count = 16 : i64, n_tile = 1 : i64}
    return
  }
}
