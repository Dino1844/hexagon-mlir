//===- hmx-tail-leaf-verify.mlir - bounds-safe leaf attr negatives ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

// expected-error @+4 {{valid_rows and valid_cols must be provided together}}
func.func @missing_pair(%src: memref<16x16xf16>,
                         %dst: memref<1x1x16x32x2xf16, 1>,
                         %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<16x16xf16>)
      outs(%dst : memref<1x1x16x32x2xf16, 1>) {valid_rows = 16 : i64}
  return
}

// -----

// expected-error @+4 {{valid_cols must be in [1, 32]}}
func.func @too_wide(%src: memref<16x33xf16>,
                    %dst: memref<1x2x16x32x2xf16, 1>,
                    %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<16x33xf16>)
      outs(%dst : memref<1x2x16x32x2xf16, 1>)
      {valid_rows = 16 : i64, valid_cols = 33 : i64}
  return
}

// -----

// expected-error @+4 {{bounds-safe tail leaves require count=1, got 2}}
func.func @bulk_tail(%src: memref<16x64xf16>,
                      %dst: memref<1x2x16x32x2xf16, 1>,
                      %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<16x64xf16>)
      outs(%dst : memref<1x2x16x32x2xf16, 1>)
      {count = 2 : i64, valid_rows = 16 : i64, valid_cols = 32 : i64}
  return
}
