//===- arg-writes-partial.mlir ---------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The positive arm of the `arg_writes` envelope child: three tensor arguments,
// one of them written.  This is the shape Triton's own lowering produces (the
// `silu_kernel` ttsharedir is this kernel), and it is the shape every Triton
// launch in the tree has -- an input, an output, and argument-only tensors.
//
// The output pointer is written through `bufferization.materialize_in_destination`
// whose destination chains back through `memref.reinterpret_cast` to `%arg1`;
// the input is staged through a `memref.copy` into a fresh `memref.alloc`, and
// the scale is only read.  So the write set is exactly `[1]`, and the two
// unwritten tensors are what the launcher used to dump, pull and copy back on
// every launch for nothing.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s
//
// CHECK-LABEL: llvm.func @partial_write(
// The write is a single store, and its pointer is the data pointer loaded out
// of the SECOND tensor argument's descriptor.  In the wrapper calling
// convention each tensor argument contributes `(i64 rank, ptr descriptor)`, so
// `%arg1` is the first tensor's descriptor, `%arg3` the second's and `%arg5`
// the third's: which one feeds the store is the whole answer.  The ordinal
// itself (`[1]`) is asserted on the envelope by the Python-side contract test,
// because that is where the launcher reads it from.
// CHECK: llvm.getelementptr %arg3[1]
// CHECK: llvm.load
// CHECK: llvm.getelementptr
// CHECK: llvm.store
// CHECK-NOT: llvm.store
//
//===----------------------------------------------------------------------===//{{^}} 
// the CHECK lines above end here

#map = affine_map<(d0) -> (d0)>

module {
  func.func @partial_write(%arg0: memref<*xf32> {tt.divisibility = 16 : i32},
                           %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
                           %arg2: memref<*xf32> {tt.divisibility = 16 : i32},
                           %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32,
                           %arg7: i32) {
    %c4 = arith.constant 4 : index
    %two = arith.constant 2.000000e+00 : f32
    %src = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    %scale = memref.reinterpret_cast %arg2 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    %buf = memref.alloc() : memref<4xf32>
    memref.copy %src, %buf : memref<4xf32, strided<[1]>> to memref<4xf32>
    %t = bufferization.to_tensor %buf restrict writable : memref<4xf32> to tensor<4xf32>
    %s = bufferization.to_tensor %scale restrict writable : memref<4xf32, strided<[1]>> to tensor<4xf32>
    %scaled = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%t, %s : tensor<4xf32>, tensor<4xf32>) outs(%t : tensor<4xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %m = arith.mulf %in, %in_1 : f32
      linalg.yield %m : f32
    } -> tensor<4xf32>
    %doubled = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%scaled : tensor<4xf32>) outs(%scaled : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.mulf %in, %two : f32
      linalg.yield %m : f32
    } -> tensor<4xf32>
    %dst = memref.reinterpret_cast %arg1 to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
    bufferization.materialize_in_destination %doubled in writable %dst : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
    return
  }
}
