//===- resident-provenance-order.mlir - stable site keys ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Reordering two otherwise independent allocations must not change either
// site's resident key.  This is a source-level check of the identity contract;
// it does not claim that a real launch's grid scope has been proven.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' | python3 "$(dirname %s)/resident-provenance-order.py"
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' | linalg-hexagon-opt -split-input-file -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' | python3 "$(dirname %s)/resident-provenance-order.py"
//===----------------------------------------------------------------------===//

module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @ordered() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("order":1:1)
    %b = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("order":2:1)
    hmx.matmul ins(%a, %b : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%a : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %b : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @ordered() {
    %b = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("order":2:1)
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("order":1:1)
    hmx.matmul ins(%a, %b : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%a : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %b : memref<2x2x16x32x2xf16, 1>
    return
  }
}
