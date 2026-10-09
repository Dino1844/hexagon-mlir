//===- stage-strided-src-2d.mlir - stage issues a 2D DMA on a strided source --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
// `hmx.stage` moves one 32-row tile out of a row-major source. When that source
// is one tile of a wider matrix, its row stride is wider than the tile's column
// count, and the transfer must be issued as a 2D DMA: one rectangle of 32 rows x
// `srcCols` columns, each row read at the source's real row stride.
//
// The bug this pins: the lowering emitted ONE 1D run of `slotBytes` bytes. The
// start address was computed with the true row stride, but the length was the
// whole slot, so a `strided<[2048,1]>` source with a 1024-column tile read 16
// whole rows (both K halves, 2048 columns) into a slot that holds 32 rows of
// one K half -- a sheared activation, which the engine contracted as a uniform
// ~2% relative error (root-caused bit for bit in
// docs/analysis/a-kloop-rel-investigation-2026-10-09.md, section 5).
//
// So the call must be `hexagon_runtime_dma2d_start` with
//   width      = srcCols * elemBytes          (1024 * 2 = 2048)
//   height     = 32                           (one crouton tile)
//   srcStride  = the source's row stride * 2   (2048 * 2 = 4096 -- the REAL
//                                               stride, never the width)
//   dstStride  = width                        (the slot is contiguous)
// and the source address must still start at row * srcStride * elemBytes.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @stage_strided_src
// The row offset is scaled by the SOURCE's real row stride (2048 elements), not
// by the tile's 1024-column width.
// CHECK: %[[ROW:.*]] = llvm.trunc %{{.*}} : i64 to i32
// CHECK: %[[COLS:.*]] = llvm.mlir.constant(1024 : i32)
// CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK: %[[W_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[WIDTH:.*]] = llvm.mul %[[COLS]], %[[W_ESZ]] : i32
// CHECK: %[[HEIGHT:.*]] = llvm.mlir.constant(32 : i32)
// CHECK: %[[S_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[STRIDE_BYTES:.*]] = llvm.mul %[[STRIDE]], %[[S_ESZ]] : i32
// CHECK: %[[ROW_BYTES:.*]] = llvm.mul %[[ROW]], %[[STRIDE_BYTES]] : i32
// CHECK: %[[SRC_ADDR:.*]] = llvm.add %{{.*}}, %[[ROW_BYTES]] : i32
// CHECK: %[[SRC_PTR:.*]] = llvm.inttoptr %[[SRC_ADDR]] : i32 to !llvm.ptr
// The source is DDR (space 0), the slot VTCM (space 1).
// CHECK: %[[SRC_SPACE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[DST_PTR:.*]] = llvm.inttoptr
// CHECK: %[[DST_SPACE:.*]] = llvm.mlir.constant(1 : i32)
// bypassCacheSrc = 0 (DDR), bypassCacheDst = 1 (the VTCM slot).
// CHECK: %[[BYP_SRC:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[BYP_DST:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[STATUS_PTR:.*]] = llvm.inttoptr
// The 13-argument 2D entry, with the real row stride as its 7th argument: a
// srcStride of 1024 (the width) would be the shear again.
// CHECK: %[[TOKEN:.*]] = llvm.call @hexagon_runtime_dma2d_start(%[[SRC_PTR]], %[[SRC_SPACE]], %[[DST_PTR]], %[[DST_SPACE]], %[[WIDTH]], %[[HEIGHT]], %[[STRIDE_BYTES]], %[[WIDTH]], %[[BYP_SRC]], %[[BYP_DST]], {{.*}}, {{.*}}, %[[STATUS_PTR]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// No 1D start survives on this path.
// CHECK-NOT: llvm.call @hexagon_runtime_dma_start(
// The wait still takes the token the start returned.
// CHECK: llvm.call @hexagon_runtime_dma_wait(%[[TOKEN]]) : (i32) -> ()
func.func @stage_strided_src(%src: memref<64x1024xf16, strided<[2048, 1]>>,
                             %slot: memref<32x1024xf16, 1>,
                             %status: memref<1xi32>, %row: index) {
  %tok = hmx.stage ins(%src, %row : memref<64x1024xf16, strided<[2048, 1]>>)
      outs(%slot, %status : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
  hmx.await ins(%tok : i32) outs(%slot : memref<32x1024xf16, 1>)
      -> memref<32x1024xf16, 1>
  return
}
