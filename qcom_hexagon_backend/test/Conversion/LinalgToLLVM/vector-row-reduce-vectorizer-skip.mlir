//===- vector-row-reduce-vectorizer-skip.mlir ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -hexagon-vectorization=skip-vector-row-reduce=true | FileCheck %s
//===----------------------------------------------------------------------===//

// With skip-vector-row-reduce, HexagonVectorization leaves the row reduces
// that match the vector-row-reduce pattern unvectorized, so they can reach
// that pass and its hvx.vror butterfly instead of becoming a
// vector.multi_reduction whose LLVM lowering is the ExpandReductions valign
// tree. The [1,32] shape is what the pipeline's HexagonTiling produces for
// the FA [512,128] row max (one HVX vector per tile).
//
// CHECK-LABEL: func.func @tiled_rowmax_survives
func.func @tiled_rowmax_survives(%src: tensor<1x32xf32>, %out: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: linalg.reduce ins({{.*}} : tensor<1x32xf32>) outs({{.*}} : tensor<1xf32>) dimensions = [1]
  // CHECK: arith.maxnumf
  // CHECK: linalg.yield
  // CHECK-NOT: vector.multi_reduction
  %0 = linalg.reduce ins(%src : tensor<1x32xf32>) outs(%out : tensor<1xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
    %1 = arith.maxnumf %in, %acc : f32
    linalg.yield %1 : f32
  }
  return %0 : tensor<1xf32>
}

// -----

// A matching reduce survives wherever it appears; the [256,32] full-row form
// matches too.
// CHECK-LABEL: func.func @rowmax_survives
func.func @rowmax_survives(%src: tensor<256x32xf32>, %out: tensor<256xf32>) -> tensor<256xf32> {
  // CHECK: linalg.reduce ins({{.*}} : tensor<256x32xf32>) outs({{.*}} : tensor<256xf32>) dimensions = [1]
  // CHECK: arith.maxnumf
  %0 = linalg.reduce ins(%src : tensor<256x32xf32>) outs(%out : tensor<256xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
    %1 = arith.maxnumf %in, %acc : f32
    linalg.yield %1 : f32
  }
  return %0 : tensor<256xf32>
}

// -----

// The fused producer body (rms_norm's x*x) matches the pattern too -- the
// butterfly re-creates the chain on the vector chunks -- so it is likewise
// left for the vector-row-reduce pass.
// CHECK-LABEL: func.func @fused_body_survives
func.func @fused_body_survives(%src: tensor<1x32xf32>, %out: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: linalg.reduce ins({{.*}} : tensor<1x32xf32>) outs({{.*}} : tensor<1xf32>) dimensions = [1]
  // CHECK: arith.mulf
  // CHECK: arith.maxnumf
  // CHECK-NOT: vector.multi_reduction
  %0 = linalg.reduce ins(%src : tensor<1x32xf32>) outs(%out : tensor<1xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
    %t = arith.mulf %in, %in : f32
    %1 = arith.maxnumf %t, %acc : f32
    linalg.yield %1 : f32
  }
  return %0 : tensor<1xf32>
}

// -----

// The widening form (an f16 row summed in f32) matches too and is likewise
// left for the vector-row-reduce pass.
// CHECK-LABEL: func.func @f16_row_f32_fold_survives
func.func @f16_row_f32_fold_survives(%src: tensor<1x32xf16>, %out: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: linalg.reduce ins({{.*}} : tensor<1x32xf16>) outs({{.*}} : tensor<1xf32>) dimensions = [1]
  // CHECK: arith.extf
  // CHECK: arith.addf
  // CHECK-NOT: vector.multi_reduction
  %0 = linalg.reduce ins(%src : tensor<1x32xf16>) outs(%out : tensor<1xf32>) dimensions = [1]
    (%in: f16, %acc: f32) {
    %e = arith.extf %in : f16 to f32
    %1 = arith.addf %e, %acc : f32
    linalg.yield %1 : f32
  }
  return %0 : tensor<1xf32>
}

// -----

// Selective: a chain that reads the init arg does not match (the running
// accumulator is not a per-element value). Upstream vectorization rejects
// that body too, so the reduce simply stays scalar.
// CHECK-LABEL: func.func @init_in_chain_stays_scalar
func.func @init_in_chain_stays_scalar(%src: tensor<1x32xf32>, %out: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: linalg.reduce
  // CHECK: arith.mulf
  // CHECK-NOT: vector.multi_reduction
  %0 = linalg.reduce ins(%src : tensor<1x32xf32>) outs(%out : tensor<1xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
    %t = arith.mulf %in, %acc : f32
    %1 = arith.maxnumf %t, %acc : f32
    linalg.yield %1 : f32
  }
  return %0 : tensor<1xf32>
}
