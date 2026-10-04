//===- l2-prefetch-pipeline.mlir - streaming prefetch end to end ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The flat elementwise kernel (vec_add, taken verbatim from the compiler's
// ttsharedir artifact) through the whole LinalgToLLVM pipeline with the L2
// prefetch on. This is the gate for "the fetches survive the rest of the
// pipeline": the pass inserts llvm-dialect calls at the memref level, and
// every later stage (async formation, vector lowering, strided-metadata
// expansion, index/arith conversion) has to carry them to translation.
//
// The OFF arm pins the escape hatch: with the option explicitly off there
// are no fetches anywhere (the default flipped ON 2026-10-05 after the
// device A/B passed; the off arm keeps the opt-out honest).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-l2-prefetch=true})' | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-l2-prefetch=false})' | FileCheck %s --check-prefix=OFF
// OFF-NOT: l2fetch

// CHECK: llvm.call @llvm.hexagon.Y5.l2fetch({{.+}}, {{.+}}) : (!llvm.ptr, i64) -> ()
// CHECK: llvm.call @llvm.hexagon.Y5.l2fetch({{.+}}, {{.+}}) : (!llvm.ptr, i64) -> ()
// CHECK-NOT: llvm.call @llvm.hexagon.Y5.l2fetch
// CHECK: llvm.func @llvm.hexagon.Y5.l2fetch(!llvm.ptr, i64)

#map = affine_map<(d0) -> (d0)>
module {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<131072xf32>
    memref.copy %reinterpret_cast, %alloc : memref<131072xf32, strided<[1]>> to memref<131072xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<131072xf32> to tensor<131072xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<131072xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<131072xf32, strided<[1]>> to memref<131072xf32>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<131072xf32> to tensor<131072xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1 : tensor<131072xf32>, tensor<131072xf32>) outs(%0 : tensor<131072xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %3 = arith.addf %in, %in_3 : f32
      linalg.yield %3 : f32
    } -> tensor<131072xf32>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_2 : (tensor<131072xf32>, memref<131072xf32, strided<[1]>>) -> ()
    return
  }
}
