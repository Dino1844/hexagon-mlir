//===- workspace-resident-multi-function.mlir - module state is complete ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Two sibling functions exercise nested func.func pass updates to the shared
// module manifest. Each function's read-modify-write must preserve the other
// function's workspace/grid classification.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' | FileCheck %s
// CHECK: hmx.kernel_manifest =
// CHECK: function = "workspace_a", grid_policy = "single-instance"
// CHECK-SAME: workspace_class = "resident-single-instance"
// CHECK: function = "workspace_b", grid_policy = "single-instance"
// CHECK-SAME: workspace_class = "resident-single-instance"
// CHECK-LABEL: func.func @workspace_a
// CHECK: hmx.workspace_resident
// CHECK-LABEL: func.func @workspace_b
// CHECK: hmx.workspace_resident
module attributes {hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 32 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 0 : i64, pack_weight_sites = 0 : i64, unpack_sites = 0 : i64}}, full = {k = 32 : i64, m = 32 : i64, n = 32 : i64}, function = "workspace_a", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 32 : i64}, m = {kind = "static", value = 32 : i64}, n = {kind = "static", value = 32 : i64}}, padded = {k = 32 : i64, m = 32 : i64, n = 32 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 24576 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "workspace_a", slot = 1 : i64}}, workspace_class = "runtime-internal", grid_policy = "legacy-runtime"}, {dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 32 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 0 : i64, pack_weight_sites = 0 : i64, unpack_sites = 0 : i64}}, full = {k = 32 : i64, m = 32 : i64, n = 32 : i64}, function = "workspace_b", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 32 : i64}, m = {kind = "static", value = 32 : i64}, n = {kind = "static", value = 32 : i64}}, padded = {k = 32 : i64, m = 32 : i64, n = 32 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 24576 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "workspace_b", slot = 1 : i64}}, workspace_class = "runtime-internal", grid_policy = "legacy-runtime"}], pack_act_sites = 0 : i64, pack_weight_sites = 0 : i64, schema = "hex.hmx.kernel_manifest/v2", unpack_sites = 0 : i64, weight_policies = []}} {
  func.func @workspace_a() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %w = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %w : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }

  func.func @workspace_b() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %w = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %w : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}
