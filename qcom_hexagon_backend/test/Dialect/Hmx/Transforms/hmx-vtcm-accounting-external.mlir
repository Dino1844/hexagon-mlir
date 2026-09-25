//===- hmx-vtcm-accounting-external.mlir - external VTCM is incomplete ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(hmx-vtcm-accounting)'
//===----------------------------------------------------------------------===//

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  memref.global "private" constant @external_vtcm : memref<64xf16, 1> = dense<0.0>
  func.func @global_storage() {
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @argument_storage(%external: memref<64xf16, 1>) {
    return
  }
}
