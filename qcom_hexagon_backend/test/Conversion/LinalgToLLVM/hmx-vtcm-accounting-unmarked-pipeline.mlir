//===- hmx-vtcm-accounting-unmarked-pipeline.mlir - census stays inert ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The real LinalgToLLVM pipeline must not publish an accounting attribute
// without the explicit internal module marker.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record{production=enable-workspace-resident})' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module
// CHECK-NOT: hmx.kernel_vtcm_accounting
// CHECK-NOT: hmx.kernel_vtcm_live_range
module {
  func.func @unmarked_pipeline(%a: tensor<64x64xf16>,
                                %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}
