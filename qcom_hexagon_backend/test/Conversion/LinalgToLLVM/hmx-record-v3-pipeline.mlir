//===- hmx-record-v3-pipeline.mlir - record-only v3 through the full pipeline ====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The record-only v3 envelope end to end.  The internal
// `hmx.diagnostic_v3_record` marker survives the whole LinalgToLLVM pipeline, so
// the publication boundary emits a `hex.hmx.translation/v2` envelope: the same
// v2 execution manifest as always, plus a separate `hmx_record` v3 child.
//
// Nothing here changes the generated code.  The v3 record rides along as
// metadata; the kernel lowering is byte-for-byte the kernel the unmarked module
// would have produced, which is the whole point of a record-only schema.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The two schemas describe the same compile, so the plan they name for the same
// function-local id must agree.  `hmx.kernel_manifest` is printed first (module
// attributes are sorted), so pinning both here is the whole check: a v3 record
// that disagreed with the v2 execution child would be a record describing a
// kernel that was not built.
// CHECK: hmx.kernel_manifest = {
// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v2"
// CHECK-DAG: function = "recorded_weight"
// CHECK-DAG: id = 0 : i64
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: reason = "selected-aligned"
// CHECK: "hmx.kernel_record/v3" = {
// CHECK-DAG: admission = "not-authorized"
// CHECK-DAG: record_mode = "record-only"
// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v3"
// CHECK-DAG: function = "recorded_weight"
// CHECK-DAG: id = 0 : i64
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
// The allocator census and liveness sidecars are marker-gated, so a plain
// compilation proves no requested bytes either.
// CHECK-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}
module attributes {hmx.diagnostic_v3_record} {
  func.func @recorded_weight(%a: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %w = arith.constant dense<1.000000e+00> : tensor<64x64xf16>
    %c0 = tensor.empty() : tensor<64x64xf16>
    %m0 = linalg.matmul ins(%a, %w : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c0 : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m0 : tensor<64x64xf16>
  }
}
