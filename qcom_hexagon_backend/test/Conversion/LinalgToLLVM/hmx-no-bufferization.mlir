//===- hmx-no-bufferization.mlir - attribution without bufferization -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
// A manually managed memref linalg.matmul must still be represented in the
// module manifest.  With bufferization disabled the attribution pass is
// record-only: VTCM is unavailable and no hmx.matmul is emitted.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-bufferization=false})' 2>&1 | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-NOT: HMX disabled
// CHECK-NOT: HMX not applied
// CHECK: hmx.kernel_manifest = {
// CHECK-SAME: engine = "hvx"
// CHECK-SAME: reason = "vtcm-allocator-disabled"
// CHECK-NOT: hmx.matmul
// CHECK: llvm.func @manual_buffer_matmul
// CHECK-NOT: hmx.matmul
// CHECK-NOT: HMX disabled
module {
  func.func @manual_buffer_matmul(%a: memref<64x64xf16>,
                                  %b: memref<64x64xf16>,
                                  %c: memref<64x64xf16>) {
    linalg.matmul ins(%a, %b : memref<64x64xf16>, memref<64x64xf16>)
                 outs(%c : memref<64x64xf16>)
    return
  }
}
