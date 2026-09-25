//===- hmx-vtcm-liveness-external.mlir - declaration status --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A declaration has no executable lifetime to analyze. It must be reported as
// not-applicable and must not be counted as one successfully analyzed function;
// module completeness covers definitions only.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: function_coverage = "definitions-only"
// CHECK-DAG: definition_count = 1 : i64
// CHECK-DAG: external_declaration_count = 1 : i64
// CHECK-DAG: symbol = "external"
// CHECK-DAG: status = "not-applicable"
// CHECK-DAG: allocation_site_coverage = "not-applicable"
// CHECK-DAG: reason = "external declaration has no body to analyze"
// CHECK-DAG: symbol = "defined"
// CHECK-DAG: status = "complete"
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func private @external()
  func.func @defined() {
    return
  }
}
