//===- hmx-target-query-refusals.mlir - the refusal-side query reads --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The other half of the query-facade equality pin (ticket 15, Phase 1): the
// planning facts that reach a reader through a refusal instead of through
// the manifest. Companion to hmx-target-query-equality.mlir, which pins the
// same queries on the selected path; test/test_hmx_target_query_facade.py
// evaluates the query methods against the passes' spellings and these same
// frozen values.
//
//   * resolveVtcmBudget(4096) = 4096: the narrowing option reaches the
//     refusal remark, which is what proves the pass resolved the option it
//     was given rather than falling back to the device default.
//   * hasEnoughRows(2) = false: the refusal states minRows = 4 by value
//     ("M > 4"), the one place a reader sees the floor.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=4096}))' -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

// 64x128x64 needs croutonBytes(64, 64, 128) = 40960 bytes and even its
// smallest M block (one 32-row tile) is over 4096, so not even one block
// fits. roomBeside(0) = 4096 - 0 = 4096 is what the plan is weighed against,
// and the remark names the budget it resolved.
// expected-warning @+1 {{HMX: 1 matmul(s) skipped; first refusal: HMX not applied: matmul (M=64, N=64, K=128, lhsElem='f16', rhsElem='f16', outElem='f16', vtcmUsed=0) bridge footprint does not fit remaining VTCM (vtcmBudget=4096 bytes) [reason=vtcm-budget]; matmul M=64, N=64, K=128, lhsElem='f16', rhsElem='f16', outElem='f16'}}
module {
func.func @refused_budget(%a: tensor<64x128xf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  // expected-remark @+1 {{HMX not applied: matmul (M=64, N=64, K=128, lhsElem='f16', rhsElem='f16', outElem='f16', vtcmUsed=0) bridge footprint does not fit remaining VTCM (vtcmBudget=4096 bytes) [reason=vtcm-budget]}}
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
}

// -----

// Too few rows to clear the engine's floor: M = 2 is at or below minRows = 4,
// so hasEnoughRows(2) is false and the refusal states the bound by value.
// The manifest record carries the same refusal as its reason.
// expected-warning @+1 {{HMX: 1 matmul(s) skipped; first refusal: HMX not applied: needs 2D static shapes, M/N/K multiples of 32, M > 4 [reason=min-rows]; matmul M=2, N=64, K=64, lhsElem='f16', rhsElem='f16', outElem='f16'}}
module {
func.func @too_few_rows(%a: tensor<2x64xf16>, %b: tensor<64x64xf16>) -> tensor<2x64xf16> {
  %c = tensor.empty() : tensor<2x64xf16>
  // expected-remark @+1 {{HMX not applied: needs 2D static shapes, M/N/K multiples of 32, M > 4 [reason=min-rows]}}
  %0 = linalg.matmul ins(%a, %b : tensor<2x64xf16>, tensor<64x64xf16>)
                     outs(%c : tensor<2x64xf16>) -> tensor<2x64xf16>
  return %0 : tensor<2x64xf16>
}
}
