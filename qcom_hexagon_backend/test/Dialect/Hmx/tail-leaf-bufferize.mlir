//===- tail-leaf-bufferize.mlir - explicit tail extents survive DPS --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries})' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tail_leaf_attrs_bufferize
// CHECK: hmx.pack_act {{.*}} {valid_cols = 17 : i64, valid_rows = 16 : i64}
// CHECK: hmx.unpack_acc {{.*}} {valid_cols = 17 : i64, valid_rows = 16 : i64}
// CHECK-NOT: tensor<
func.func @tail_leaf_attrs_bufferize(
    %src: tensor<16x17xf16>,
    %act: tensor<1x1x16x32x2xf16>,
    %dst: tensor<16x17xf16>) -> tensor<16x17xf16> {
  %c0 = arith.constant 0 : index
  %p = hmx.pack_act ins(%src, %c0, %c0 : tensor<16x17xf16>)
      outs(%act : tensor<1x1x16x32x2xf16>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
      -> tensor<1x1x16x32x2xf16>
  %u = hmx.unpack_acc ins(%p, %c0, %c0 : tensor<1x1x16x32x2xf16>)
      outs(%dst : tensor<16x17xf16>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
      -> tensor<16x17xf16>
  return %u : tensor<16x17xf16>
}
