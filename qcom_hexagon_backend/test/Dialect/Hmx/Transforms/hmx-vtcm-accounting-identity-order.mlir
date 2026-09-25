//===- hmx-vtcm-accounting-identity-order.mlir - stable site IDs -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Reordering independent allocations must not change the identity attached to
// either source location.  The helper checks the source-to-ID mapping rather
// than relying on output order or an allocation ordinal.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | python3 "$(dirname %s)/hmx-vtcm-accounting-identity-order.py"
//===----------------------------------------------------------------------===//

module @identity_module attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity} {
  func.func @identity() {
    %a = memref.alloc() : memref<64x64xf16, 1> loc("order":10:1)
    %b = memref.alloc() : memref<32x32xf16, 1> loc("order":11:1)
    memref.dealloc %a : memref<64x64xf16, 1>
    memref.dealloc %b : memref<32x32xf16, 1>
    return
  }
}

// -----

module @identity_module attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity} {
  func.func @identity() {
    %b = memref.alloc() : memref<32x32xf16, 1> loc("order":11:1)
    %a = memref.alloc() : memref<64x64xf16, 1> loc("order":10:1)
    memref.dealloc %a : memref<64x64xf16, 1>
    memref.dealloc %b : memref<32x32xf16, 1>
    return
  }
}
