//===- unpack-dst-non-contiguous-reject.mlir - a transposed unpack dest fails closed --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-to-llvm)' -verify-diagnostics
//===----------------------------------------------------------------------===//
//
// A NON-inner-contiguous UNPACK DESTINATION used to lower silently, with the
// declared stride(rank-2) handed straight to the leaf as `row_stride`.
//
// `strided<[1, 32]>` is COLUMN-major: element (i,j) sits at i + 32*j. The
// unpack leaf is a row-major writer, so `row_stride = 1` makes it store at
// i + j -- element (i+j, 0) where it meant (i, j). Verified on the previous
// build: the emitted call was
//     llvm.call @hmx_unpack_acc_f16(..., %28, ...)   // %28 = constant(1 : i32)
// with no diagnostic anywhere. A silent wrong-value store, not a crash.
//
// The pack SOURCE was never exposed to this: `bridgeCanExpress`
// (MatmulToHmxPass.cpp:251) refuses transposed linalg indexing maps, and
// `verifyDiagnosticRowMajor` (HmxPartitionPass.cpp:1012/1015) demands
// strides[1] == 1. The destination was checked only on the diagnostic TAIL path
// (HmxPartitionPass.cpp:1175) -- the ordinary full-tile path had no gate.
//
// This test pins the ordinary path's new gate. The pack-source sibling with
// the same intent is `hmx-tail-layout-reject.mlir`; the tail-path destination
// check is `hmx-partition-tail-residual-dominance.mlir`'s neighbourhood.

func.func @column_major_unpack_destination(
    %src: memref<1x1x16x32x2xf16, 1>,
    %dst: memref<32x32xf16, strided<[1, 32]>>,
    %row: index, %col: index) {
  // expected-error @+2 {{HMX leaf requires an inner-contiguous buffer; got inner stride 32 on a rank-2 memref}}
  // expected-error @+1 {{failed to legalize operation 'hmx.unpack_acc'}}
  hmx.unpack_acc ins(%src, %row, %col : memref<1x1x16x32x2xf16, 1>)
      outs(%dst : memref<32x32xf16, strided<[1, 32]>>)
  return
}

// -----
//
// A row-major strided destination with stride(0) > width is the N-split store
// case rowStride() exists for: that stride is REAL and must keep flowing
// through, not be rejected along with the transposed one.
//
// -split-input-file IS LOAD-BEARING HERE, and an independent audit on
// 2026-10-01 caught the version that did not have it. Dialect conversion ABORTS
// at the first legalization failure: with both functions in ONE module, the
// transposed one fails, so the ops after it are never converted at all -- and
// the positive function then asserts nothing. Measured: making the positive
// function transposed as well still PASSED. Only the split marker above gives
// it teeth. (Suspect the same shape anywhere a -verify-diagnostics test has a
// negative case before a positive one.)
func.func @n_split_row_major_destination_is_still_accepted(
    %src: memref<1x2x16x32x2xf16, 1>,
    %dst: memref<32x64xf16, strided<[64, 1]>>,
    %row: index, %col: index) {
  hmx.unpack_acc ins(%src, %row, %col : memref<1x2x16x32x2xf16, 1>)
      outs(%dst : memref<32x64xf16, strided<[64, 1]>>)
  return
}
