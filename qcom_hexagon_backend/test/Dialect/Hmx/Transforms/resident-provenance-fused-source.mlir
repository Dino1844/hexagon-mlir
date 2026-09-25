//===- resident-provenance-fused-source.mlir - fused workspace site -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Strict resident provenance and the accounting sidecar use the same source
// walker.  A debug label plus one file location is therefore a stable site;
// an unknown or multiply-sourced fused location remains not-proven.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fused_workspace
// CHECK: hmx.resident_provenance =
// CHECK-DAG: site = "file:fusedws:10:1"
// CHECK-DAG: principal_status = "module-symbol"
// CHECK-DAG: descriptor_status = "checked"
module @fused_workspace attributes {hmx.diagnostic_vtcm_accounting,
                                     hmx.diagnostic_vtcm_identity} {
  func.func @fused_workspace() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
        loc(fused["workspace", "fusedws":10:1])
    %w = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>,
                       memref<2x2x16x32x2xf16, 1>)
        outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %w : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}
