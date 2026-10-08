//===- hmx-vtcm-liveness-pipeline.mlir - production-shaped liveness -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This is the narrow host-side pipeline fixture for the R-B liveness slice. It
// keeps the real LinalgToLLVM/HMX allocation shape, while the explicit marker
// enables only the internal diagnostic sidecar. The resident floor is
// reported separately from transient requested bytes; no allocator or observed
// high-water claim is introduced.
//
// The workspace figure is the folded-serial one (S2.5): the whole 2x2
// activation array (8192 bytes) is retired by the serial pack fold and a
// one-row scratch (4096 bytes) takes its place, so the workspace sum is
// 256 (conversion state) + 8192 (accumulator) + 4096 (scratch) = 12544 --
// 4096 less than the whole-array form.
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-workspace-resident})' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v2"
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "accounting_pipeline"
// CHECK-DAG: allocation_sites = 4 : i64
// CHECK-DAG: deallocation_sites = 0 : i64
// CHECK-DAG: transient_requested_peak_bytes = 0 : i64
// CHECK-DAG: weight_resident_requested_bytes = 8192 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 12544 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 20736 : i64
// CHECK-DAG: allocator_peak_status = "not-proven"
// CHECK-DAG: resident_runtime_state = "not-proven"
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
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
