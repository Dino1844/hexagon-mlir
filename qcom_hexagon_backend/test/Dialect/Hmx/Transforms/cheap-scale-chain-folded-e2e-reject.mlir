//===- cheap-scale-chain-folded-e2e-reject.mlir - the elementwise fold does not survive e2e --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' 2>&1 \
// RUN:   | FileCheck %s
//
// The `not` above is INTENT, not mechanism: in `not A | B` the pipeline status
// comes from B, so FileCheck is what actually decides this test (an audit on
// 2026-10-01 pointed this out). It is kept because it tells the next reader
// that a non-zero opt exit is expected here -- but do not "fix" it by removing
// FileCheck, and do not trust `not` as the assertion. The assertion is the two
// CHECK lines: a tool that succeeded would print neither, and FileCheck fails
// on empty input.
//===----------------------------------------------------------------------===//
//
// WHY THIS FILE EXISTS, AND WHY IT IS CHECKED IN RATHER THAN COMPUTED.
//
// `hmx-layout-propagation.mlir` already pins this function
// (`@cheap_scale_chain`, func.func at :228) as FOLDED, and that test is green.
// Its RUN line (:24) runs ONE pass:
//
//   -pass-pipeline='builtin.module(func.func(matmul-to-hmx))'
//
// so the suite has never asked whether the folded IR lowers. Asking, on this
// build, gives:
//
//   remark: the activation is not a row-major staging bridge (nothing to
//          re-host onto a VTCM slot), so the tile loop reads the crouton array
//          where it is
//   error: 'hmx.mma' op act must be in VTCM (memory space 1)
//
// Mechanism: once the inter-matmul pack is folded away, findActivationBridge(act)
// returns null (HmxPartitionPass.cpp:1414-1417, NoRowMajorBridge), so no staging
// loop is emitted, and the surviving hmx.matmul trips the verifier at
// HmxOps.cpp:56/68.
//
// So: FoldElementwiseIntoLayout (MatmulToHmxPass.cpp:2206) has never produced a
// kernel that compiles. A green single-pass test says the fold FIRES; it says
// nothing about the result being REACHABLE, and here the result is not.
//
// The folded IR is checked in verbatim rather than produced by a second RUN
// line, for two reasons:
//   * the repo's manual lit runner (tools/hexmlir/run_lit_manual.sh, which is
//     what the gate actually runs) SKIPS files that need %t, so a
//     compute-then-check test would be skipped, and a skipped test is worth
//     nothing as evidence;
//   * the checked-in copy is the artifact the claim is about. If the fold ever
//     starts producing something that lowers, this file fails and whoever fixed
//     it deletes this file with a commit message saying so.
//
// Note the module attribute below: the manifest reports
// plan = "full-hmx" for BOTH contraction sites of a kernel that cannot
// compile. That is the "reports success" pattern again, one level up.
//
// WHAT THIS DOES NOT CLAIM: nothing about performance, nothing about numerical
// correctness. Zero device runs, zero launches. It claims exactly one thing --
// the folded form does not lower.

// CHECK: remark: {{.*}}activation is not a row-major staging bridge
// CHECK: error: 'hmx.mma' op act must be in VTCM

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// The manifest below claims plan = "full-hmx" for both sites. It is
// reproduced verbatim, attribute and all, because that claim is part of what
// this file exists to pin.
module attributes {hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 64 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 0 : i64}}, full = {k = 64 : i64, m = 64 : i64, n = 64 : i64}, function = "cheap_scale_chain", grid_policy = "legacy-runtime", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 64 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 64 : i64, m = 64 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 24576 : i64, vtcm_bridge_peak_bytes = 49152 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "cheap_scale_chain", slot = 1 : i64}}, workspace_class = "runtime-internal"}, {dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 64 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 0 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64}}, full = {k = 64 : i64, m = 64 : i64, n = 64 : i64}, function = "cheap_scale_chain", grid_policy = "legacy-runtime", id = 1 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 64 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 64 : i64, m = 64 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 24576 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "cheap_scale_chain", slot = 2 : i64}}, workspace_class = "runtime-internal"}], pack_act_sites = 1 : i64, pack_weight_sites = 2 : i64, schema = "hex.hmx.kernel_manifest/v2", unpack_sites = 1 : i64, weight_policies = []}} {
  func.func @cheap_scale_chain(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>, %arg2: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.500000e-01 : f16
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    %1 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %0) -> (tensor<2x2x16x32x2xf16>) {
      %15 = hmx.pack_act ins(%arg0, %arg3, %c0 : tensor<64x64xf16>) outs(%arg4 : tensor<2x2x16x32x2xf16>) {count = 2 : i64, hmx.decision_id = 0 : i64} -> tensor<2x2x16x32x2xf16>
      scf.yield %15 : tensor<2x2x16x32x2xf16>
    }
    %2 = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    %3 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %2) -> (tensor<2x2x16x32x2xf16>) {
      %15 = hmx.pack_weight ins(%arg1, %c0, %arg3 : tensor<64x64xf16>) outs(%arg4 : tensor<2x2x16x32x2xf16>) {count = 2 : i64, hmx.decision_id = 0 : i64} -> tensor<2x2x16x32x2xf16>
      scf.yield %15 : tensor<2x2x16x32x2xf16>
    }
    %4 = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    %5 = hmx.matmul ins(%1, %3 : tensor<2x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>) outs(%4 : tensor<2x2x16x32x2xf16>) {hmx.decision_id = 0 : i64} -> tensor<2x2x16x32x2xf16>
    %6 = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<2x2x16x32x2xf16>) outs(%6 : tensor<2x2x16x32x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %15 = arith.mulf %in, %cst_0 : f16
      %16 = arith.addf %15, %cst : f16
      linalg.yield %16 : f16
    } -> tensor<2x2x16x32x2xf16>
    %8 = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    %9 = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    %10 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %9) -> (tensor<2x2x16x32x2xf16>) {
      %15 = hmx.pack_weight ins(%arg2, %c0, %arg3 : tensor<64x64xf16>) outs(%arg4 : tensor<2x2x16x32x2xf16>) {count = 2 : i64, hmx.decision_id = 1 : i64} -> tensor<2x2x16x32x2xf16>
      scf.yield %15 : tensor<2x2x16x32x2xf16>
    }
    %11 = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
    %12 = hmx.matmul ins(%7, %10 : tensor<2x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>) outs(%11 : tensor<2x2x16x32x2xf16>) {hmx.decision_id = 1 : i64} -> tensor<2x2x16x32x2xf16>
    %13 = tensor.empty() : tensor<64x64xf16>
    %14 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %13) -> (tensor<64x64xf16>) {
      %15 = hmx.unpack_acc ins(%12, %arg3, %c0 : tensor<2x2x16x32x2xf16>) outs(%arg4 : tensor<64x64xf16>) {count = 16 : i64, hmx.decision_id = 1 : i64} -> tensor<64x64xf16>
      scf.yield %15 : tensor<64x64xf16>
    }
    return %14 : tensor<64x64xf16>
  }
}
