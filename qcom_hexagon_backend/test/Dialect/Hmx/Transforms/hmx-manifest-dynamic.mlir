//===- hmx-manifest-dynamic.mlir - tagged dynamic HVX manifest ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A dynamic contraction must not lose its M/N/K facts when it stays on HVX.
// The record carries three canonical dynamic axes and no inferred SSA names.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_manifest = {
// CHECK: dtypes = {lhs = "f16", out = "f16", rhs = "f16"}
// CHECK: function = "all_dynamic"
// CHECK: logical = {k = {kind = "dynamic", symbol = "k"}, m = {kind = "dynamic", symbol = "m"}, n = {kind = "dynamic", symbol = "n"}}
// CHECK: plan = "hvx"
// CHECK: reason = "dynamic-shape"
// CHECK: shape_state = "dynamic"
module {
  func.func @all_dynamic(%a: tensor<?x?xf16>, %b: tensor<?x?xf16>,
                          %m: index, %n: index) -> tensor<?x?xf16> {
    %c = tensor.empty(%m, %n) : tensor<?x?xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>)
                       outs(%c : tensor<?x?xf16>) -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
}
