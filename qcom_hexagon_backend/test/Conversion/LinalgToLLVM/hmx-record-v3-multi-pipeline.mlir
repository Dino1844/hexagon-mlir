//===- hmx-record-v3-multi-pipeline.mlir - two matmuls, full pipeline ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The two-matmul regression at the publication boundary.
//
// `hmx-record-v3-multi.mlir` covers the same case at the attribution boundary.
// This one proves the document survives the whole LinalgToLLVM pipeline and
// leaves the envelope with two digested records rather than two skeletons, and
// that the v2 execution child still owns both records.
//
// The whole v3 document is printed on one line, so each claim gets its own
// FileCheck prefix: a shared scan would let one directive's match position
// decide whether the next one still finds its own.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s --check-prefix=V2
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s --check-prefix=RECS
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s --check-prefix=DIGESTS
//===----------------------------------------------------------------------===//

// The v2 execution manifest is unaffected and stays the execution authority.
// V2: hmx.kernel_manifest = {
// V2-DAG: schema = "hex.hmx.kernel_manifest/v2"
// V2-DAG: function = "two"
// V2-DAG: id = 0 : i64
// V2-DAG: id = 1 : i64

// Both function-local ids and both shapes are published.
// RECS: "hmx.kernel_record/v3" = {
// RECS-DAG: schema = "hex.hmx.kernel_manifest/v3"
// RECS-DAG: record_mode = "record-only"
// RECS-DAG: admission = "not-authorized"
// RECS-DAG: function = "two"
// RECS-DAG: id = 0 : i64
// RECS-DAG: id = 1 : i64
// RECS-DAG: shape = {logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 64 : i64}, n = {kind = "static", value = 64 : i64}}, specialization = "upstream-static", state = "static"}
// RECS-DAG: shape = {logical = {k = {kind = "static", value = 32 : i64}, m = {kind = "static", value = 32 : i64}, n = {kind = "static", value = 32 : i64}}, specialization = "upstream-static", state = "static"}

// Both records carry a digest.  A skeleton -- a record without one -- is not a
// publishable record, so this is the assertion that would have caught the
// two-matmul case failing to finalize.
// DIGESTS: "hmx.kernel_record/v3" = {
// DIGESTS-COUNT-2: record_fingerprint = "sha256:
module attributes {hmx.diagnostic_v3_record} {
  func.func @two(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                 %c: tensor<32x32xf16>, %d: tensor<32x32xf16>)
      -> (tensor<64x64xf16>, tensor<32x32xf16>) {
    %e0 = tensor.empty() : tensor<64x64xf16>
    %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
    %e1 = tensor.empty() : tensor<32x32xf16>
    %m1 = linalg.matmul ins(%c, %d : tensor<32x32xf16>, tensor<32x32xf16>)
                       outs(%e1 : tensor<32x32xf16>) -> tensor<32x32xf16>
    return %m0, %m1 : tensor<64x64xf16>, tensor<32x32xf16>
  }
}
