//===- maxnum-legalize.mlir - vector maxnumf -> vmax + NaN fixup -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -hvx-maxnum-legalize -canonicalize | FileCheck %s
//===----------------------------------------------------------------------===//

// A vector f32 maxnumf becomes maximumf (-> V6_vmax via FMAXIMUMNUM) plus the
// two oeq self-compares and two selects that restore strict maxnum semantics.
// The maximumf keeps the original op's fastmath flags; the fixup compares get
// none.
//
// CHECK-LABEL: func.func @maxnum_v32f32
func.func @maxnum_v32f32(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  // CHECK: %[[vmax:.*]] = arith.maximumf %[[a:.*]], %[[b:.*]] : vector<32xf32>
  // CHECK: %[[anum:.*]] = arith.cmpf oeq, %[[a]], %[[a]] : vector<32xf32>
  // CHECK: %[[bnum:.*]] = arith.cmpf oeq, %[[b]], %[[b]] : vector<32xf32>
  // CHECK: %[[fixb:.*]] = arith.select %[[bnum]], %[[vmax]], %[[a]] : vector<32xi1>, vector<32xf32>
  // CHECK: %[[fixa:.*]] = arith.select %[[anum]], %[[fixb]], %[[b]] : vector<32xi1>, vector<32xf32>
  // CHECK: return %[[fixa]]
  %r = arith.maxnumf %a, %b : vector<32xf32>
  return %r : vector<32xf32>
}

// -----

// Same for f16 (64 lanes), with the fastmath flags carried by the maximumf.
// CHECK-LABEL: func.func @maxnum_f16_fast
func.func @maxnum_f16_fast(%a: vector<64xf16>, %b: vector<64xf16>) -> vector<64xf16> {
  // CHECK: arith.maximumf {{%.*}}, {{%.*}} fastmath<fast> : vector<64xf16>
  // CHECK: arith.cmpf oeq, {{%.*}}, {{%.*}} : vector<64xf16>
  // CHECK-NOT: fastmath
  // CHECK: arith.select
  // CHECK: arith.select
  %r = arith.maxnumf %a, %b fastmath<fast> : vector<64xf16>
  return %r : vector<64xf16>
}

// -----

// Value check, non-NaN: maxnumf(inf, 1.0) = inf. (The maximumf folds; the
// selects keep the fixup shape on constants.)
// CHECK-LABEL: func.func @vec_inf_first
func.func @vec_inf_first() -> vector<4xf32> {
  // CHECK-DAG: %[[inf:.*]] = arith.constant dense<0x7F800000>
  // CHECK-DAG: %[[one:.*]] = arith.constant dense<1.000000e+00>
  // CHECK: arith.select {{.*}}, %[[inf]], %[[one]]
  %one = arith.constant dense<1.0> : vector<4xf32>
  %inf = arith.constant dense<0x7F800000> : vector<4xf32>
  %r = arith.maxnumf %inf, %one : vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// Value check, both operands NaN: strict maxnum returns (some) NaN -- the
// fixup folds all the way to the NaN constant, it does not leak a vmax of two
// NaNs through.
// CHECK-LABEL: func.func @vec_both_nan
func.func @vec_both_nan() -> vector<4xf32> {
  // CHECK-NOT: arith.maximumf
  // CHECK-NOT: arith.select
  // CHECK: %[[nan:.*]] = arith.constant dense<0x{{.*}}>
  // CHECK: return %[[nan]]
  %nan = arith.constant dense<0x7FC00000> : vector<4xf32>
  %r = arith.maxnumf %nan, %nan : vector<4xf32>
  return %r : vector<4xf32>
}

// -----

// Negative: scalars are left alone (their lowering is at worst one fmaxf
// libcall).
// CHECK-LABEL: func.func @scalar_kept
func.func @scalar_kept(%a: f32, %b: f32) -> f32 {
  // CHECK: arith.maxnumf %{{.*}}, %{{.*}} : f32
  // CHECK-NOT: arith.maximumf
  %r = arith.maxnumf %a, %b : f32
  return %r : f32
}

// -----

// Negative: integer vectors are not touched.
// CHECK-LABEL: func.func @int_vec_kept
func.func @int_vec_kept(%a: vector<32xi32>, %b: vector<32xi32>) -> vector<32xi32> {
  // CHECK: arith.maxsi
  // CHECK-NOT: arith.maximumf
  %r = arith.maxsi %a, %b : vector<32xi32>
  return %r : vector<32xi32>
}
