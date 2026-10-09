//===- stage-dynamic-stride.mlir - the staged stride is read at runtime ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
// The stage counterpart of pack-dynamic-stride.mlir. Bufferization's branch
// merge (the dense edge-temp gather vs the raw strided view) casts both into
// `strided<[?, ?], offset: ?>`, so the static type has LOST the row distance.
// The runtime descriptor still carries it, and the 2D descriptor's srcStride
// must be that value: `extractvalue [4, 0]` of the packed memref struct, the
// same read `asAddress` already performs, times the element size.
//
// Substituting the tile width here would describe a rectangle whose rows are
// the width apart over a source whose rows are wider -- the same shear as the
// 1D bug, reached through the erased-stride path (the 2026-10-09 stride fix
// already corrected the ADDRESS this way; this pins the descriptor alongside
// it).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @stage_dynamic_stride
// CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<64x1024xf16, strided<[?, ?], offset: ?>> to !llvm.struct
// CHECK: %[[ROW:.*]] = llvm.trunc %{{.*}} : i64 to i32
// CHECK: %[[COLS:.*]] = llvm.mlir.constant(1024 : i32)
// CHECK: %[[ST:.*]] = llvm.extractvalue %[[SRC]][4, 0]
// CHECK: %[[STI:.*]] = llvm.trunc %[[ST]] : i64 to i32
// CHECK: %[[W_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// width = srcCols * elemBytes.
// CHECK: %[[WIDTH:.*]] = llvm.mul %[[COLS]], %[[W_ESZ]] : i32
// height = one crouton tile.
// CHECK: %[[HEIGHT:.*]] = llvm.mlir.constant(32 : i32)
// srcStride = the descriptor's stride * elemBytes -- never a width constant.
// CHECK: %[[S_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[STRIDE_BYTES:.*]] = llvm.mul %[[STI]], %[[S_ESZ]] : i32
// The row offset follows the same runtime stride.
// CHECK: %[[ROW_BYTES:.*]] = llvm.mul %[[ROW]], %[[STRIDE_BYTES]] : i32
// CHECK: %[[SRC_PTR:.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
// CHECK: %[[SRC_SPACE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[DST_PTR:.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
// CHECK: %[[DST_SPACE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[BYP_SRC:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[BYP_DST:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[STATUS_PTR:.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
// CHECK: llvm.call @hexagon_runtime_dma2d_start(%[[SRC_PTR]], %[[SRC_SPACE]], %[[DST_PTR]], %[[DST_SPACE]], %[[WIDTH]], %[[HEIGHT]], %[[STRIDE_BYTES]], %[[WIDTH]], %[[BYP_SRC]], %[[BYP_DST]], {{.*}}, {{.*}}, %[[STATUS_PTR]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NOT: llvm.call @hexagon_runtime_dma_start(
func.func @stage_dynamic_stride(
    %src: memref<64x1024xf16, strided<[?, ?], offset: ?>>,
    %slot: memref<32x1024xf16, 1>, %status: memref<1xi32>, %row: index) {
  %tok = hmx.stage ins(%src, %row
      : memref<64x1024xf16, strided<[?, ?], offset: ?>>)
      outs(%slot, %status : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
  hmx.await ins(%tok : i32) outs(%slot : memref<32x1024xf16, 1>)
      -> memref<32x1024xf16, 1>
  return
}
