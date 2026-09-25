//===- hmx-vtcm-accounting-noop.mlir - unmarked accounting is inert -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module
// CHECK-NOT: hmx.kernel_vtcm_accounting
module {
  func.func @unmarked() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }
}
