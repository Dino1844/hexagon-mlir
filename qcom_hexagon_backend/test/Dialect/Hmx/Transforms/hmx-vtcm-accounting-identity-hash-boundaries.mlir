//===- hmx-vtcm-accounting-identity-hash-boundaries.mlir - hash KAT ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// These known-answer values lock the standard FNV-1a-64 basis/prime and the
// fixed-width component lengths.  The two tuples would be ambiguous under raw
// string concatenation, but must remain distinct.  This is compiler-static
// identity evidence only; it is not a runtime observation or launcher join.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @ab
// CHECK-DAG: id_schema = "hmx.resident-key/fnv1a64/v1"
// CHECK-DAG: function_id = 4187358797774944965 : i64
// CHECK-DAG: site_id = 3463946795366118809 : i64
module @ab attributes {hmx.diagnostic_vtcm_accounting,
                        hmx.diagnostic_vtcm_identity} {
  func.func @c() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("identity-boundary":1:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// CHECK-LABEL: module @a
// CHECK-DAG: id_schema = "hmx.resident-key/fnv1a64/v1"
// CHECK-DAG: function_id = -2602436766687962171 : i64
// CHECK-DAG: site_id = -5015131223447551891 : i64
module @a attributes {hmx.diagnostic_vtcm_accounting,
                       hmx.diagnostic_vtcm_identity} {
  func.func @bc() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("identity-boundary":1:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}
