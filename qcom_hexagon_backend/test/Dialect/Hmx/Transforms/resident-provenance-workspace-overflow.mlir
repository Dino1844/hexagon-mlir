//===- resident-provenance-workspace-overflow.mlir - v2 byte width -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))'
//===----------------------------------------------------------------------===//

module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_identity} {
  func.func @workspace_overflow() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("workspace":1:1)
    %b = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("workspace":2:1)
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("workspace":3:1)
    // expected-error @+1 {{resident workspace byte size exceeds the v2 uint32 ABI}}
    %huge = memref.alloc() : memref<4294967296xi8, 1>
    hmx.matmul ins(%a, %b : memref<2x2x16x32x2xf16, 1>,
                         memref<2x2x16x32x2xf16, 1>)
      outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %b : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %huge : memref<4294967296xi8, 1>
    return
  }
}
