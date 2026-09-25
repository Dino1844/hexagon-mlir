//===- hmx-vtcm-accounting-reject.mlir - unknown VTCM bytes fail closed ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A dynamic extent is admitted only with a proven compile-time constant upper
// bound (see hmx-vtcm-accounting-dynamic-extent.mlir for the admitted form).
// An extent that comes from a runtime value has no proven bound, so it stays
// unproven and the census fails closed instead of reporting a guess or zero.
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @dynamic_vtcm(%d: index) {
    %a = memref.alloc(%d) : memref<?x32xf16, 1>
    memref.dealloc %a : memref<?x32xf16, 1>
    return
  }
}

// -----

// A computed bound is not a proven bound: clamping or capping an argument at
// compile time says nothing about the value the runtime passes.
// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @computed_bound_vtcm(%d: index) {
    %c8 = arith.constant 8 : index
    %bounded = arith.minui %d, %c8 : index
    %a = memref.alloc(%bounded) : memref<?x32xf16, 1>
    memref.dealloc %a : memref<?x32xf16, 1>
    return
  }
}

// -----

// A negative constant is not an extent.  Reporting a zero or wrapping byte
// count here would be exactly the "unsupported turned into zero" failure.
// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @nonpositive_bound_vtcm() {
    %neg = arith.constant -1 : index
    %a = memref.alloc(%neg) : memref<?x32xf16, 1>
    memref.dealloc %a : memref<?x32xf16, 1>
    return
  }
}
