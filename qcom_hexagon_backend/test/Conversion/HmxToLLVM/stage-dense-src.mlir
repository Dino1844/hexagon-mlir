//===- stage-dense-src.mlir - a dense source degenerates to the same bytes ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
// The other direction of the stage pin: a DENSE source (stride(0) == width)
// must keep the movement it always had. The lowering is unconditionally 2D now,
// so on this shape the descriptor's srcStride equals its width and the
// rectangle is a contiguous 65536-byte run -- byte for byte what the old 1D
// `hexagon_runtime_dma_start(length)` did. That is the point of not adding a
// "strided only" branch: there is nothing to branch on, and nothing about a
// dense tile's transfer changes.
//
// This file also pins the two `bypassCache` flags, which are taken per endpoint
// from the memref's memory space: a VTCM endpoint is not cacheable memory, so
// the descriptor's snoop-and-invalidate is redundant there and may be skipped,
// while a DDR endpoint must keep it (bin/runtime/include/RuntimeDMA.cc; the
// same rule as llama.cpp's hand-written dma_queue_push_single_*). Flipping
// either one makes this file red.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @stage_dense_src
// stride(0) == 1024 == the width, so both the row step and the descriptor's
// srcStride are 1024 * 2 = 2048 bytes: the same contiguous run the 1D entry
// used to issue as one length.
// CHECK: %[[ROW:.*]] = llvm.trunc %{{.*}} : i64 to i32
// CHECK: %[[COLS:.*]] = llvm.mlir.constant(1024 : i32)
// CHECK: %[[W_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[WIDTH:.*]] = llvm.mul %[[COLS]], %[[W_ESZ]] : i32
// CHECK: %[[HEIGHT:.*]] = llvm.mlir.constant(32 : i32)
// CHECK: %[[S_ESZ:.*]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[STRIDE_BYTES:.*]] = llvm.mul %[[COLS]], %[[S_ESZ]] : i32
// CHECK: %[[ROW_BYTES:.*]] = llvm.mul %[[ROW]], %[[STRIDE_BYTES]] : i32
// CHECK: %[[SRC_PTR:.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
// CHECK: %[[SRC_SPACE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[DST_PTR:.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
// CHECK: %[[DST_SPACE:.*]] = llvm.mlir.constant(1 : i32)
// bypassCacheSrc = 0 (DDR source), bypassCacheDst = 1 (the VTCM slot).
// CHECK: %[[BYP_SRC:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[BYP_DST:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[STATUS_PTR:.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
// CHECK: %[[TOK:.*]] = llvm.call @hexagon_runtime_dma2d_start(%[[SRC_PTR]], %[[SRC_SPACE]], %[[DST_PTR]], %[[DST_SPACE]], %[[WIDTH]], %[[HEIGHT]], %[[STRIDE_BYTES]], %[[WIDTH]], %[[BYP_SRC]], %[[BYP_DST]], {{.*}}, {{.*}}, %[[STATUS_PTR]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// The 2D entry is the only DMA start this path emits.
// CHECK-NOT: llvm.call @hexagon_runtime_dma_start(
// CHECK: llvm.call @hexagon_runtime_dma_wait(%[[TOK]]) : (i32) -> ()
// A staging/awaiting function runs no HMX instruction, so it needs no engine
// ensure/unlock pair: these are plain DMA calls.
// CHECK-NOT: llvm.call @hexagon_runtime_hmx_ensure_dsp
// CHECK-NOT: llvm.call @hexagon_runtime_hmx_unlock_dsp
func.func @stage_dense_src(%src: memref<64x1024xf16>,
                           %slot: memref<32x1024xf16, 1>,
                           %status: memref<1xi32>, %row: index) {
  %tok = hmx.stage ins(%src, %row : memref<64x1024xf16>)
      outs(%slot, %status : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
  hmx.await ins(%tok : i32) outs(%slot : memref<32x1024xf16, 1>)
      -> memref<32x1024xf16, 1>
  return
}
