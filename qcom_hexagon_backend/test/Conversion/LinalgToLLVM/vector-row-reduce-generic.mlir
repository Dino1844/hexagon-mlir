//===- vector-row-reduce-generic.mlir - generalized reduce form ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// LinalgGeneralize runs earlier in the production pipeline and rewrites every
// linalg.reduce into a linalg.generic, so the form the row-reduce butterfly
// (and the vectorizer's skip gate) meets in production is the *generalized*
// one: a generic carrying a single reduction iterator over the innermost
// dimension. These tests pin that form: it must be diverted to the hvx.vror
// butterfly, and the vectorizer must leave it alone.
//
// RUN: linalg-hexagon-opt %s -split-input-file -vector-row-reduce | FileCheck %s --check-prefix=RR
// RUN: linalg-hexagon-opt %s -split-input-file -hexagon-vectorization=skip-vector-row-reduce=true | FileCheck %s --check-prefix=SKIP
//===----------------------------------------------------------------------===//

// The production shape: one row of 32 f32 reduced to a scalar (one HVX vector
// per row). The butterfly replaces ExpandReductions' valign tree + fmaxf
// libcall chain.
//
// RR-LABEL: func.func @generic_rowmax_f32
// RR: scf.for
// RR: %[[v0:.*]] = vector.transfer_read
// RR: %[[r64:.*]] = hvx.vror %[[v0]], 64 : vector<32xf32>
// RR: arith.maxnumf
// RR: hvx.vror %{{.*}}, 4 : vector<32xf32>
// RR: vector.extract
// RR-NOT: linalg.generic
// RR-NOT: vector.reduction
func.func @generic_rowmax_f32(%src: memref<32xf32>, %dst: memref<f32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
                  iterator_types = ["reduction"]}
    ins(%src : memref<32xf32>) outs(%dst : memref<f32>) {
  ^bb0(%in: f32, %acc: f32):
    %0 = arith.maxnumf %in, %acc : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// Same shape, add reduction (the row-sum half of softmax): also diverted.
//
// RR-LABEL: func.func @generic_rowsum_f32
// RR: hvx.vror
// RR: arith.addf
// RR-NOT: linalg.generic
func.func @generic_rowsum_f32(%src: memref<32xf32>, %dst: memref<f32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
                  iterator_types = ["reduction"]}
    ins(%src : memref<32xf32>) outs(%dst : memref<f32>) {
  ^bb0(%in: f32, %acc: f32):
    %0 = arith.addf %in, %acc : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// The mechanical gate: the reduction must be the innermost dimension so a row
// stays contiguous under unit stride. Here the reduction is dim0 and dim1 is
// parallel, so the butterfly does not apply and the op is left alone.
//
// RR-LABEL: func.func @generic_reduction_not_innermost
// RR-NOT: hvx.vror
// RR: linalg.generic
func.func @generic_reduction_not_innermost(%src: memref<32x32xf32>, %dst: memref<32xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>],
                  iterator_types = ["reduction", "parallel"]}
    ins(%src : memref<32x32xf32>) outs(%dst : memref<32xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %0 = arith.maxnumf %in, %acc : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// Body contract: a fused elementwise producer between the input arg and the
// fold matches -- the chain re-creates on the vector chunk, then the
// butterfly folds it.
//
// RR-LABEL: func.func @generic_fused_chain
// RR: arith.mulf {{.*}} : vector<32xf32>
// RR: hvx.vror
// RR-NOT: linalg.generic
func.func @generic_fused_chain(%src: memref<32xf32>, %dst: memref<f32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
                  iterator_types = ["reduction"]}
    ins(%src : memref<32xf32>) outs(%dst : memref<f32>) {
  ^bb0(%in: f32, %acc: f32):
    %t = arith.mulf %in, %in : f32
    %0 = arith.maxnumf %t, %acc : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// Body contract: a chain that reads the running acc is not a per-element
// value -> no match, the scalar path keeps the generic.
//
// RR-LABEL: func.func @generic_chain_reads_acc
// RR-NOT: hvx.vror
// RR: linalg.generic
func.func @generic_chain_reads_acc(%src: memref<32xf32>, %dst: memref<f32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
                  iterator_types = ["reduction"]}
    ins(%src : memref<32xf32>) outs(%dst : memref<f32>) {
  ^bb0(%in: f32, %acc: f32):
    %t = arith.mulf %in, %acc : f32
    %0 = arith.maxnumf %t, %acc : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// The skip gate: HexagonVectorization must leave the generalized form alone
// (tensor stage, before bufferization) so it reaches the butterfly instead of
// becoming a vector.reduction whose LLVM lowering is the ExpandReductions
// valign tree.
//
// SKIP-LABEL: func.func @skip_generic_rowmax
// SKIP: linalg.generic
// SKIP: arith.maxnumf
// SKIP: linalg.yield
// SKIP-NOT: vector.reduction
// SKIP-NOT: vector.multi_reduction
func.func @skip_generic_rowmax(%src: tensor<32xf32>, %out: tensor<f32>) -> tensor<f32> {
  %init = arith.constant 0.000000e+00 : f32
  %o = tensor.empty() : tensor<f32>
  %fill = linalg.fill ins(%init : f32) outs(%o : tensor<f32>) -> tensor<f32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
                       iterator_types = ["reduction"]}
      ins(%src : tensor<32xf32>) outs(%fill : tensor<f32>) {
  ^bb0(%in: f32, %acc: f32):
    %r = arith.maxnumf %in, %acc : f32
    linalg.yield %r : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}
