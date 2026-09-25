//===- hmx-vtcm-accounting-resident-reject.mlir - resident bytes proven ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' -verify-diagnostics
//===----------------------------------------------------------------------===//

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @bad_resident_bytes() {
    %a = memref.alloc() {hmx.workspace_resident = {bytes = 1 : i64}}
        : memref<64x64xf16, 1>
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }
}
