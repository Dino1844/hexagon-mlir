//===- hmx-vtcm-accounting-resident-declaration-mismatch.mlir - site sum --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A declared resident aggregate is a cross-check, not an allocation site.  If
// it disagrees with the descriptors discovered in the IR, the result remains
// incomplete and the pass fails, but resident_site_sum_bytes must contain only
// the discovered 4096 bytes.
//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' 2>&1 | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: status = "incomplete"
// CHECK-DAG: weight_resident_bytes = 4096 : i64
// CHECK-DAG: resident_site_sum_bytes = 4096 : i64
// CHECK-NOT: resident_site_sum_bytes = 8192 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.weight_resident_bytes = 8192 : i64} {
  memref.global "private" constant @weight : memref<64x32xf16> = dense<1.000000e+00> {alignment = 128 : i64}
  func.func @declared_mismatch() {
    %weight = hexagonmem.alloc() {hmx.weight_resident = {global = @weight, bytes = 4096 : i64}}
        : memref<64x32xf16, 1>
    return
  }
}
