//===- pack-dynamic-stride.mlir - runtime stride for erased pack sources --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A pack source (or unpack destination) whose static stride was erased to
// `strided<[?, ?]>` still has a true row stride in its RUNTIME memref
// descriptor, and the leaf needs that number: it walks rows at
// `src + (row * src_stride + col) * 2`.
//
// How the type gets erased: bufferization's branch merge (dense edge-temp
// gather vs raw strided view) casts both into `strided<[?, ?], offset: ?>`.
// The old lowering substituted the view WIDTH for the dynamic stride -- right
// for a dense temp, a silent shear for the raw view, whose rows are the whole
// N apart while the width is one N-tile. Measured on the device object code
// 2026-10-09 (pack_weight called with src = raw B + n0, src_stride = the slice
// width, true stride = the whole N; the leaf multiplies rows by that stride in
// libhmxapi.a): the engine multiplied out a sheared weight. See the rowStride
// TRACEABILITY note in HmxToLLVMPass.cpp.
//
// This file pins the replacement on all three stride-bearing leaves: the call
// argument after cols/n must be the truncation of the descriptor's
// strides[0] (`extractvalue [4, 0]` of the packed memref struct, the same
// read asAddress performs), never a width constant. The static contract (real
// stride as a constant) stays pinned by pack-strided-src.mlir; the dense
// fallback stays pinned by every dense test.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @pack_weight_dynamic_stride
// CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<2048x512xf16, strided<[?, ?], offset: ?>> to !llvm.struct
// CHECK: %[[ST:.*]] = llvm.extractvalue %[[SRC]][4, 0]
// CHECK: %[[STI:.*]] = llvm.trunc %[[ST]]
// CHECK: llvm.call @hmx_pack_weight_f16({{.*}}, {{.*}}, {{.*}}, %{{.*}}, %[[STI]],
func.func @pack_weight_dynamic_stride(%wsrc: memref<2048x512xf16, strided<[?, ?], offset: ?>>,
                                      %wt: memref<16x64x16x32x2xf16, 1>,
                                      %kt: index, %nt: index) {
  hmx.pack_weight ins(%wsrc, %kt, %nt : memref<2048x512xf16, strided<[?, ?], offset: ?>>)
      outs(%wt : memref<16x64x16x32x2xf16, 1>)
  return
}

// -----

// The activation source takes the same path (second pack rowStride site).
// CHECK-LABEL: func.func @pack_act_dynamic_stride
// CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<512x2048xf16, strided<[?, ?], offset: ?>> to !llvm.struct
// CHECK: %[[ST:.*]] = llvm.extractvalue %[[SRC]][4, 0]
// CHECK: %[[STI:.*]] = llvm.trunc %[[ST]]
// CHECK: llvm.call @hmx_pack_act_f16({{.*}}, {{.*}}, {{.*}}, %{{.*}}, %[[STI]],
func.func @pack_act_dynamic_stride(%asrc: memref<512x2048xf16, strided<[?, ?], offset: ?>>,
                                   %at: memref<1x64x16x32x2xf16, 1>,
                                   %row: index, %col: index) {
  hmx.pack_act ins(%asrc, %row, %col : memref<512x2048xf16, strided<[?, ?], offset: ?>>)
      outs(%at : memref<1x64x16x32x2xf16, 1>)
  return
}

// -----

// And the unpack destination side (dst_stride is the store-side twin of
// src_stride): a dynamic destination stride must come from the descriptor too,
// or the store shears exactly like the load does.
// CHECK-LABEL: func.func @unpack_dst_dynamic_stride
// CHECK: %[[DST:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<512x2048xf16, strided<[?, ?], offset: ?>> to !llvm.struct
// CHECK: %[[ST:.*]] = llvm.extractvalue %[[DST]][4, 0]
// CHECK: %[[STI:.*]] = llvm.trunc %[[ST]]
// CHECK: llvm.call @hmx_unpack_acc_f16({{.*}}, {{.*}}, {{.*}}, %{{.*}}, %[[STI]],
func.func @unpack_dst_dynamic_stride(%ar: memref<16x16x16x32x2xf16, 1>,
                                     %dst: memref<512x2048xf16, strided<[?, ?], offset: ?>>,
                                     %row: index, %col: index) {
  hmx.unpack_acc ins(%ar, %row, %col : memref<16x16x16x32x2xf16, 1>)
      outs(%dst : memref<512x2048xf16, strided<[?, ?], offset: ?>>)
  return
}
