//===- hmx-target-query-equality.mlir - the query facade's reads, pinned ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Phase 1 of the HmxTarget query facade (operator-parity ticket 15).
//
// HmxTarget.h names the planning facts the passes each read on their own as
// query methods (resolveVtcmBudget / roomBeside / tilesIn / isTileAligned /
// hasEnoughRows / the croutonBytes terms). Phase 1 changes no behaviour and
// migrates no consumer: the passes still spell the same arithmetic in place,
// and this file plus test/test_hmx_target_query_facade.py hold the two
// spellings at the same value.
//
// The two sides of the pin:
//   * the checks below are the CONSUMER side -- the real pass, this build,
//     this IR: the budget it resolved, the tile counts it divided by, the
//     bytes it committed;
//   * the Python pin evaluates each QUERY method from HmxTarget.h against
//     the consumer's extracted spelling and against the same frozen values
//     written here.
//
// The refusal-side reads (the budget a remark names, the row-floor message)
// are pinned in hmx-target-query-refusals.mlir, which runs with diagnostics
// verification on.
//
// A query that drifts from its consumer, or a consumer that re-derives a
// fact the facade answers, turns one of the two red. Nothing below is a new
// expectation: every number is the one the pass produced before the facade
// existed.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s --check-prefix=UNSET
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=1234567}))' -split-input-file | FileCheck %s --check-prefix=SET
//===----------------------------------------------------------------------===//

// resolveVtcmBudget(0) = the device default, 8 MiB; resolveVtcmBudget(1234567)
// = the option's own value. 64x64x64 also pins the croutonBytes terms:
// (64*64 + 64*64 + 64*64) * 2 = 24576, the bridge-peak bytes the manifest
// publishes for it -- activationBytes + weightBytes + readoutBytes, the three
// named queries, summed by croutonBytes.
// UNSET: vtcm_bridge_peak_bytes = 24576 : i64
// UNSET: vtcm_budget_bytes = 8388608 : i64
// SET: vtcm_bridge_peak_bytes = 24576 : i64
// SET: vtcm_budget_bytes = 1234567 : i64
module {
func.func @budget_reads(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
}

// -----

// tilesIn, three reads in one bridge: the activation's outer walk is
// tilesIn(96) = 3 tiles (its [Mt, Kt] grid), its K run tilesIn(160) = 5 (the
// pack's `count`), and the weight's grid is tilesIn(64) = 2 by the same 5.
// The bridge peak is croutonBytes(96, 64, 160) = (96*160 + 160*64 + 96*64) *
// 2 = 63488, and the resolved budget is the option in the SET run and the
// device default in the UNSET run.
// UNSET: vtcm_bridge_peak_bytes = 63488 : i64
// UNSET: vtcm_budget_bytes = 8388608 : i64
// UNSET: hmx.pack_act ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<96x160xf16>) outs(%{{.*}} : tensor<3x5x16x32x2xf16>) {count = 5 : i64,
// UNSET: hmx.pack_weight ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<160x64xf16>) outs(%{{.*}} : tensor<2x5x16x32x2xf16>) {count = 5 : i64,
// UNSET: hmx.unpack_acc ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<3x2x16x32x2xf16>
// SET: vtcm_bridge_peak_bytes = 63488 : i64
// SET: vtcm_budget_bytes = 1234567 : i64
// SET: hmx.pack_act ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<96x160xf16>) outs(%{{.*}} : tensor<3x5x16x32x2xf16>) {count = 5 : i64,
// SET: hmx.pack_weight ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<160x64xf16>) outs(%{{.*}} : tensor<2x5x16x32x2xf16>) {count = 5 : i64,
// SET: hmx.unpack_acc ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<3x2x16x32x2xf16>
module {
func.func @tile_counts(%a: tensor<96x160xf16>, %b: tensor<160x64xf16>) -> tensor<96x64xf16> {
  %c = tensor.empty() : tensor<96x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<96x160xf16>, tensor<160x64xf16>)
                     outs(%c : tensor<96x64xf16>) -> tensor<96x64xf16>
  return %0 : tensor<96x64xf16>
}
}
