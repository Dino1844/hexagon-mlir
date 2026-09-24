//===- hmx-tail-layout-reject.mlir - unknown tail strides fail closed ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' -verify-diagnostics
//===----------------------------------------------------------------------===//

// expected-error @+5 {{pack source tail leaf requires static strides}}
// expected-error @+4 {{failed to legalize operation 'hmx.pack_act'}}
func.func @dynamic_source_stride(
    %src: memref<16x17xf16, strided<[?, 1], offset: ?>>,
    %dst: memref<1x1x16x32x2xf16, 1>, %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<16x17xf16, strided<[?, 1], offset: ?>>)
      outs(%dst : memref<1x1x16x32x2xf16, 1>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
  return
}
