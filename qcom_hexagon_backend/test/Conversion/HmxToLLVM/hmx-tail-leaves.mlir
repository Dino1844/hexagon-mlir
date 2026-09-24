//===- hmx-tail-leaves.mlir - explicit bounds-safe leaf ABI ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// These are diagnostic/test IR only. A valid extent explicitly selects a
// bounds-safe leaf; ordinary full-tile operations continue to use the existing
// fast leaves. HmxPartition still rejects a tail_plan until peeled-edge
// ownership is implemented.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-DAG: llvm.func @hmx_pack_act_tail_f16(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK-DAG: llvm.func @hmx_pack_weight_tail_f16(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK-DAG: llvm.func @hmx_pack_act_tail_f32(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK-DAG: llvm.func @hmx_pack_weight_tail_f32(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK-DAG: llvm.func @hmx_unpack_acc_tail_f16(i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK-DAG: llvm.func @hmx_unpack_acc_tail_f32(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

// CHECK-LABEL: func.func @tail_pack_f16
// CHECK: llvm.call @hmx_pack_act_tail_f16
func.func @tail_pack_f16(%src: memref<16x17xf16>,
                           %dst: memref<1x1x16x32x2xf16, 1>,
                           %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<16x17xf16>)
      outs(%dst : memref<1x1x16x32x2xf16, 1>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
  return
}

// CHECK-LABEL: func.func @tail_pack_f32
// CHECK: llvm.call @hmx_pack_act_tail_f32
func.func @tail_pack_f32(%src: memref<16x17xf32>,
                           %dst: memref<1x1x16x32x2xf16, 1>,
                           %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<16x17xf32>)
      outs(%dst : memref<1x1x16x32x2xf16, 1>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
  return
}

// CHECK-LABEL: func.func @tail_weight_f16
// CHECK: llvm.call @hmx_pack_weight_tail_f16
func.func @tail_weight_f16(%src: memref<16x17xf16>,
                             %dst: memref<1x1x16x32x2xf16, 1>,
                             %k: index, %n: index) {
  hmx.pack_weight ins(%src, %k, %n : memref<16x17xf16>)
      outs(%dst : memref<1x1x16x32x2xf16, 1>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
  return
}

// CHECK-LABEL: func.func @tail_weight_f32
// CHECK: llvm.call @hmx_pack_weight_tail_f32
func.func @tail_weight_f32(%src: memref<16x17xf32>,
                             %dst: memref<1x1x16x32x2xf16, 1>,
                             %k: index, %n: index) {
  hmx.pack_weight ins(%src, %k, %n : memref<16x17xf32>)
      outs(%dst : memref<1x1x16x32x2xf16, 1>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
  return
}

// CHECK-LABEL: func.func @tail_unpack_f16
// CHECK: llvm.call @hmx_unpack_acc_tail_f16
func.func @tail_unpack_f16(%src: memref<1x1x16x32x2xf16, 1>,
                             %dst: memref<16x17xf16>,
                             %row: index, %col: index) {
  hmx.unpack_acc ins(%src, %row, %col : memref<1x1x16x32x2xf16, 1>)
      outs(%dst : memref<16x17xf16>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
  return
}

// CHECK-LABEL: func.func @tail_unpack_f32
// CHECK: llvm.call @hmx_unpack_acc_tail_f32
func.func @tail_unpack_f32(%src: memref<1x1x16x32x2xf16, 1>,
                             %dst: memref<16x17xf32>,
                             %res: memref<16x17xf32>,
                             %row: index, %col: index) {
  hmx.unpack_acc_f32 ins(%src, %row, %col, %res : memref<1x1x16x32x2xf16, 1>, memref<16x17xf32>)
      outs(%dst : memref<16x17xf32>)
      {valid_rows = 16 : i64, valid_cols = 17 : i64}
  return
}
