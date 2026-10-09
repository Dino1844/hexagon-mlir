//===- hmx-vtcm-ledger-tensor.mlir - the ledger's tensor population --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// HmxVtcmLedger.h owns the answer to "how many bytes of VTCM does this
// function already hold?", which five passes used to compute with five
// hand-written walks (the argument for the extraction -- that those walks were
// copies rather than cross-checks -- is in the ledger's header).
//
// This file pins the *tensor* population against numbers derived from the walk
// that was deleted, so an implementation change that moves a single byte fails
// here even though every budget test still passes. Nothing rewrites an existing
// expectation: the literals below are computed by hand from the old arithmetic.
//
// TENSOR (caller: matmul-to-hmx). The function already holds one
// `hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>` = 2*2*16*32*2 = 4096 elements
// x 2 bytes = **8192**, so the VTCM-budget refusal must report `vtcmUsed=8192`.
//
// CENSUS (caller: hmx-vtcm-accounting). The accounting pass derives the same
// population from the allocation sites instead of from the ledger walk; on this
// IR it must publish `transient_bytes = 8192`. Two callers, one number.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=10000}))' 2>%t.err | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=TENSOR < %t.err
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s --check-prefix=CENSUS
//===----------------------------------------------------------------------===//

// A refusal leaves the IR alone: the contraction is still a `linalg.matmul`.
// IR: func.func @tensor_adapter
// IR: linalg.matmul
// TENSOR: vtcmUsed=8192
// CENSUS-DAG: status = "complete"
// CENSUS-DAG: transient_bytes = 8192 : i64
// CENSUS-DAG: raw_site_sum_bytes = 8192 : i64
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @tensor_adapter(%a: tensor<64x64xf16>,
                            %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    // 2*2*16*32*2 = 4096 elements, 2 bytes each.
    %p = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}
