//===- hmx-vtcm-accounting-crouton.mlir - legacy and HMX crouton sites --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: kind = "allocation-site-census"
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: transient_bytes = 8396800 : i64
// CHECK-DAG: raw_site_sum_bytes = 8396800 : i64
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @crouton_sites() {
    %legacy = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<2x2x16x32x2xf16, vtcm>
    %hmx = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    return
  }
}
