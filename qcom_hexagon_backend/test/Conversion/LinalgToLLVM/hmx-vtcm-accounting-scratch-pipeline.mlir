//===- hmx-vtcm-accounting-scratch-pipeline.mlir - final IR census -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Accounting must run after scratch argument/view rewriting. The final IR has
// an external VTCM argument and views rather than the original allocations, so
// the diagnostic must report that structure and fail closed instead of
// publishing the pre-scratch allocation census.
//
// The workspace-resident option is spelled out as false because this test's
// subject is the scratch-argument path, and the resident path rewrites the
// workspace structure it checks (the views become resident addresses). The
// resident path has its own pipeline coverage (hmx-workspace-resident-
// pipeline.mlir); the default flipped to on 2026-10-04, so the flag is what
// keeps this fixture on the path it was written for.
//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record{production=scratch=1048576,enable-workspace-resident=false})' 2>&1 | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven
// CHECK: hexagon.scratch
// CHECK: memref.view
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: status = "incomplete"
// CHECK-DAG: external_scratch = "unsupported"
// CHECK-DAG: external_vtcm = "function-argument"
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @accounting_scratch_pipeline(
      %a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}
