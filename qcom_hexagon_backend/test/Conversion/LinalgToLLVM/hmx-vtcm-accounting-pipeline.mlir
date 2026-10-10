//===- hmx-vtcm-accounting-pipeline.mlir - accounting in the real pipeline -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The module marker is diagnostic-only. The pass must run after HMX partition,
// workspace residency, convert-to-hexagonmem, and optional scratch/view
// rewriting; the resulting internal attribute must survive later lowering as
// metadata.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record{production=enable-workspace-resident})' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v2"
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: kind = "allocation-site-census"
// CHECK-DAG: status = "complete"
// CHECK-DAG: peak_status = "not-proven"
// CHECK-DAG: external_scratch = "none"
// CHECK-DAG: raw_site_sum_bytes = {{[0-9]+}} : i64
// CHECK: llvm.func @hexagon_runtime_workspace_resident_v2_dsp
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @accounting_pipeline(%a: tensor<64x64xf16>,
                                   %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}
