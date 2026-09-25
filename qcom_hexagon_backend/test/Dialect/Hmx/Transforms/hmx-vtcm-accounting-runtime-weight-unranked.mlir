//===- hmx-vtcm-accounting-runtime-weight-unranked.mlir - unranked ABI --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The real Triton runtime ABI may expose an unranked f16 memref. It is still
// a valid entry-argument source for the aligned-pointer resident key.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: kind = "allocation-site-census"
// CHECK-DAG: status = "complete"
// CHECK-DAG: weight_resident_bytes = 8192 : i64
// CHECK-DAG: raw_site_sum_bytes = 8192 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.weight_resident_bytes = 8192 : i64} {
  func.func @runtime_weight_unranked(%weight: memref<*xf16>) {
    %address = memref.extract_aligned_pointer_as_index %weight
        : memref<*xf16> -> index
    %resident = hexagonmem.alloc(%address)
        {hmx.weight_resident = {address, bytes = 8192 : i64}}
        : memref<2x2x16x32x2xf16, 1>
    return
  }
}
