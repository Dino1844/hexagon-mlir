//===- matmul-to-hmx-diagnostic-tail-budget-reject.mlir - fail closed ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=10000}))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-DAG: plan = "hvx"
// CHECK-DAG: reason = "vtcm-budget"
// CHECK-LABEL: func.func @tail_budget_reject
// CHECK: linalg.matmul
// CHECK: hmx.diagnostic_tail_partition
func.func @tail_budget_reject(%a: tensor<33x33xf16>,
                               %b: tensor<33x33xf16>) -> tensor<33x33xf16> {
  %c = tensor.empty() : tensor<33x33xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<33x33xf16>, tensor<33x33xf16>)
      outs(%c : tensor<33x33xf16>) {hmx.diagnostic_tail_partition}
      -> tensor<33x33xf16>
  return %0 : tensor<33x33xf16>
}
