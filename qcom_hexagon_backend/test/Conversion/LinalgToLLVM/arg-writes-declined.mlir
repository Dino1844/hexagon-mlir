//===- arg-writes-declined.mlir --------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The negative arm: a write the extractor does not model, which must publish
// `null` rather than guess.
//
// The kernel calls an external function with an argument pointer.  Nothing in
// this module says whether that function writes through it, so an extractor
// that assumed "no" would drop a tensor the kernel may well have written --
// the silent-lost-result failure the whole design is built to avoid.  The
// extractor therefore declines the extraction, the envelope publishes `null`,
// and the launcher keeps today's behaviour of dumping, pulling and copying
// back every ranked input.
//
// The same decline covers every other structure the extractor does not model:
// an unrecognized op with a memory write, a view it cannot follow, a block
// argument that is not the function's own.  This fixture is the one that is
// cheap to write and cheap to keep working.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s
//
// CHECK-LABEL: llvm.func @declined_write(
// The unmodelled writer survives the pipeline and takes the output argument's
// descriptor with it.  Whether that function writes is not a question this
// module answers, which is exactly why the extractor declines and the
// envelope publishes `null` (asserted by the Python-side contract test) rather
// than claiming a set it cannot prove.
// CHECK: llvm.call @extern_scale
//
//===----------------------------------------------------------------------===//{{^}}
// the CHECK lines above end here

#map = affine_map<(d0) -> (d0)>

module {
  func.func private @extern_scale(memref<*xf32>)
  func.func @declined_write(%arg0: memref<*xf32> {tt.divisibility = 16 : i32},
                            %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
                            %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32,
                            %arg6: i32) {
    %two = arith.constant 2.000000e+00 : f32
    %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    %buf = memref.alloc() : memref<4xf32>
    memref.copy %src, %buf : memref<4xf32, strided<[1]>> to memref<4xf32>
    %t = bufferization.to_tensor %buf restrict writable : memref<4xf32> to tensor<4xf32>
    %scaled = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%t : tensor<4xf32>) outs(%t : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.mulf %in, %two : f32
      linalg.yield %m : f32
    } -> tensor<4xf32>
    %dst = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    bufferization.materialize_in_destination %scaled in writable %dst : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
    // The unmodelled writer: whether this writes %arg1 is not a question this
    // module answers, so the extraction declines.
    func.call @extern_scale(%arg1) : (memref<*xf32>) -> ()
    return
  }
}
