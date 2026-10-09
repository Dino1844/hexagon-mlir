//===- hmx-nlooped-matmul-pipeline.mlir - an N-loop-only matmul -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The contraction whose ONLY loop is the kernel's N-blocking loop: `MM = 1`
// folds the m-loop to a single trip and `BK = K` folds the k-loop away, so one
// `scf.for` (no iter_args) wraps the `linalg.matmul` and nothing else does.
//
// Why this test exists. `matmul-to-hmx` wraps its activation pack in a loop
// over the bridge's outer tiles, but canonicalization folds the single-tile
// case (`Mt == 1`, a 32-row block) into a bare `hmx.pack_act`. What encloses
// that pack is then the KERNEL's loop, and `findActivationBridge` used to claim
// it as the bridge's own pack loop. `hmx-partition`'s staged emitter builds its
// tile loop *inside* wherever the matmul sits and afterwards erases
// `bridge->loop` -- which erased the N loop, the freshly built tile loop, and
// the matmul's result write with it. The build still succeeded and the
// manifest still recorded `plan = "full-hmx"` / `selected = "staged"`, over an
// empty span: 0 mma, 0 pack, 0 unpack, no store to the output (measured
// host-side on the `MM=1, NN=8192, KK=8192, (BM,BN,BK)=(32,128,8192)` kernel,
// 2026-10-09).
//
// The mirror-image shapes are covered by their own tests: the K loop keeps its
// serial/shallow-K verdict (`hmx-klooped-matmul-pipeline.mlir`), and an m-tiled
// bridge still has its pack loop to claim. This one pins the third case, where
// there is no pack loop left to claim and the bridge must be retired pack by
// pack instead of by erasing the loop it happens to sit in.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The engine is selected, and the staged record is now TRUE: the ring it
// claims is in the IR as real sites.
// CHECK: hmx.kernel_manifest =
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: reason = "selected-aligned"
// CHECK-DAG: blocking = "whole"
// CHECK-DAG: selected = "staged"
// CHECK-DAG: pack_act_sites = 1 : i64
// CHECK-DAG: pack_weight_sites = 1 : i64
// CHECK-DAG: unpack_sites = 1 : i64

// The engine, in the order it runs: the weight bridge packs the N tile, the
// staged ring issues and awaits the activation transfer, the scratch is packed
// from it, then the accumulator is cleared, the mma issued and the accumulator
// read back.
// CHECK: llvm.call @hmx_pack_weight_f16_bulk
// CHECK: llvm.call @hexagon_runtime_dma_start
// CHECK: llvm.call @hexagon_runtime_dma_wait
// CHECK: llvm.call @hmx_pack_act_f16_bulk
// CHECK: llvm.call @hmx_acc_clear_f16
// CHECK: llvm.call @hmx_mma_f16
// CHECK: llvm.call @hmx_acc_store_f16

// The read-out, and then the N loop's result write: the loop this matmul sits
// in must survive the partition, and its store with it. Without it the kernel
// computes nothing into the output at all.
// CHECK: llvm.call @hmx_unpack_acc_f16_bulk
// CHECK: llvm.call @memrefCopy

module {
  func.func @nlooped(%a: memref<32x8192xf16>, %b: memref<8192x8192xf16>,
                     %c: memref<32x8192xf16>) {
    %cz = arith.constant 0.000000e+00 : f16
    %cf = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c8192 = arith.constant 8192 : index

    // The N-blocking loop and nothing else: 64 N tiles of 128 columns.
    scf.for %j = %c0 to %c8192 step %c128 {
      %w = memref.subview %b[0, %j] [8192, 128] [1, 1]
          : memref<8192x8192xf16> to memref<8192x128xf16, strided<[8192, 1], offset: ?>>

      // The activation source is materialised INSIDE the loop, as the front
      // end does (it zero-fills the rows past M=1), so the pack bridge is
      // emitted here and canonicalization leaves it bare -- one crouton row for
      // a 32-row block. That bare pack is the shape this test is about.
      %abuf = memref.alloc() : memref<32x8192xf16>
      linalg.fill ins(%cz : f16) outs(%abuf : memref<32x8192xf16>)
      %at = bufferization.to_tensor %abuf restrict writable
          : memref<32x8192xf16> to tensor<32x8192xf16>
      %wt = bufferization.to_tensor %w restrict writable
          : memref<8192x128xf16, strided<[8192, 1], offset: ?>>
        to tensor<8192x128xf16>

      // Fresh accumulator per iteration: each N tile writes a disjoint slice,
      // so nothing carries across iterations.
      %z = tensor.empty() : tensor<32x128xf32>
      %init = linalg.fill ins(%cf : f32) outs(%z : tensor<32x128xf32>)
          -> tensor<32x128xf32>
      %m = linalg.matmul ins(%at, %wt : tensor<32x8192xf16>, tensor<8192x128xf16>)
                        outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>

      %ot = tensor.empty() : tensor<32x128xf16>
      %r = linalg.generic {
             indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                              affine_map<(d0, d1) -> (d0, d1)>],
             iterator_types = ["parallel", "parallel"]}
           ins(%m : tensor<32x128xf32>) outs(%ot : tensor<32x128xf16>) {
      ^bb0(%in: f32, %out: f16):
        %t = arith.truncf %in : f32 to f16
        linalg.yield %t : f16
      } -> tensor<32x128xf16>

      %dst = memref.subview %c[0, %j] [32, 128] [1, 1]
          : memref<32x8192xf16> to memref<32x128xf16, strided<[8192, 1], offset: ?>>
      bufferization.materialize_in_destination %r in writable %dst
          : (tensor<32x128xf16>,
             memref<32x128xf16, strided<[8192, 1], offset: ?>>) -> ()
    }
    return
  }
}
