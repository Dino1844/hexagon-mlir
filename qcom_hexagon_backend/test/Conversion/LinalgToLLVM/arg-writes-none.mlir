//===- arg-writes-none.mlir ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The `[]` arm: a kernel that writes no tensor argument at all.  Its result
// stays in a buffer it owns (`memref.alloc`), so the write set is the empty
// list -- a *positive* claim, and a different fact from `null`.
//
// The distinction is load-bearing.  `[]` says "nothing came back, and that is
// the truth"; `null` says "the producer could not prove anything, so keep
// today's behaviour of returning every ranked input".  A launcher that treated
// the two the same would turn an unknown into a decision, exactly the failure
// this field exists to avoid -- so the boundary keeps them apart and this
// fixture is what pins it.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s
//
// CHECK-LABEL: llvm.func @no_argument_write(
// Both buffers are the kernel's own: one stages the input, the other holds the
// result, and both are freed.  Nothing written survives into an argument, so
// the write set is `[]` -- asserted on the envelope by the Python-side
// contract test.  What this arm can express at the IR level is the ownership:
// a result that had to reach the caller would be stored through an argument
// descriptor instead.
// CHECK: llvm.call @malloc
// CHECK: llvm.call @malloc
// CHECK: llvm.call @free
// CHECK: llvm.call @free
//
//===----------------------------------------------------------------------===//{{^}}
// the CHECK lines above end here

#map = affine_map<(d0) -> (d0)>

module {
  func.func @no_argument_write(%arg0: memref<*xf32> {tt.divisibility = 16 : i32},
                               %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
                               %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32,
                               %arg6: i32) {
    %two = arith.constant 2.000000e+00 : f32
    %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    %scale = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    %buf = memref.alloc() : memref<4xf32>
    memref.copy %src, %buf : memref<4xf32, strided<[1]>> to memref<4xf32>
    %t = bufferization.to_tensor %buf restrict writable : memref<4xf32> to tensor<4xf32>
    %s = bufferization.to_tensor %scale restrict writable : memref<4xf32, strided<[1]>> to tensor<4xf32>
    %scaled = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%t, %s : tensor<4xf32>, tensor<4xf32>) outs(%t : tensor<4xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %m = arith.mulf %in, %in_1 : f32
      linalg.yield %m : f32
    } -> tensor<4xf32>
    // The destination is a buffer the kernel owns, not an argument.  A write
    // into fresh memory resolves to "no argument", which is what makes the
    // answer `[]` rather than a decline.
    bufferization.materialize_in_destination %scaled in writable %buf : (tensor<4xf32>, memref<4xf32>) -> ()
    return
  }
}
