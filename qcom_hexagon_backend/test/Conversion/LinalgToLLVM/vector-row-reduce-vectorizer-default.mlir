//===- vector-row-reduce-vectorizer-default.mlir ---------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -hexagon-vectorization | FileCheck %s
//===----------------------------------------------------------------------===//

// Default (skip-vector-row-reduce=false): the vectorizer consumes the [1,32]
// row reduce into a vector.multi_reduction, exactly as before the skip gate
// existed. This pins the default pipeline against the gate.
//
// CHECK-LABEL: func.func @rowmax_vectorized_by_default
func.func @rowmax_vectorized_by_default(%src: tensor<1x32xf32>, %out: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-NOT: linalg.reduce
  // CHECK: vector.multi_reduction <maxnumf>
  %0 = linalg.reduce ins(%src : tensor<1x32xf32>) outs(%out : tensor<1xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
    %1 = arith.maxnumf %in, %acc : f32
    linalg.yield %1 : f32
  }
  return %0 : tensor<1xf32>
}
