//===- hmx-partition-depth.mlir - the pipeline-depth knob -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The activation-staging ring depth is a knob so a shape can be A/B'd inside one
// build: 0 (default) is auto -- the deepest ring the VTCM budget and the tile
// count allow -- 1 forces the serial staged ring, and 2 asks for the double ring
// (narrowed to the deepest ring that fits, with a remark, when the budget cannot
// pay for it). The fixture is 4 tiles of Kt=32, so both depths have a steady
// state to overlap. Depth 1 is the serial source loop (issue -> await ->
// compute) left unpipelined; depth 2 is the same loop handed to the SCF
// pipeliner, which generates the prologue, the steady kernel and the peeled
// epilogue itself.
//
// 3 is the other kind of arm: it skips staging entirely and emits the plain tile
// loop, keeping the activation bridge and its array. It is the third A/B arm,
// not a deeper ring.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=1}))' | FileCheck %s --check-prefix=DEPTH1
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2}))' | FileCheck %s --check-prefix=DEPTH2
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2 vtcm-budget=300592}))' -verify-diagnostics
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2 vtcm-budget=300592}))' | FileCheck %s --check-prefix=DOWNGRADE
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=3}))' | FileCheck %s --check-prefix=DEPTH3 --implicit-check-not=hmx.stage --implicit-check-not=hmx.await
//===----------------------------------------------------------------------===//

// The explicit decision-id attribute is the hand-off used when tensor
// bufferization rebuilds the op. The module manifest is updated in place before
// hmx.matmul is erased.
//
// DEPTH1 is the forced-depth-1 arm, and budget_depth = 2 there is the whole
// point of the field: the knob asked for 1 while the budget would have paid for
// 2, so budget_depth > depth says "chose shallower on purpose". Contrast
// DOWNGRADE below, where budget_depth == depth == 1 and the reason is
// vtcm-budget, which says "the budget would not pay for more". With only depth,
// those two arms were the same number and only the reason told them apart.
// DEPTH1: hmx.kernel_manifest = {
// DEPTH1: pipeline = {budget_depth = 2 : i64, depth = 1 : i64, requested = 1 : i64, selected = "staged"}

// DEPTH2: hmx.kernel_manifest = {
// DEPTH2: pipeline = {budget_depth = 2 : i64, depth = 2 : i64, requested = 2 : i64, selected = "staged"}

// A requested depth-2 ring that the budget cannot pay for remains staged, but
// the selected depth and canonical reason describe the downgrade.
// budget_depth is 1 here, not 2: it records what the VTCM budget allowed,
// which is the one number that was previously unobservable and is what
// separates "clamped by the budget" from "chose depth 1 anyway".
// DOWNGRADE: hmx.kernel_manifest = {
// DOWNGRADE: pipeline = {budget_depth = 1 : i64, depth = 1 : i64, reason = "vtcm-budget", requested = 2 : i64, selected = "staged"}

// DEPTH3: hmx.kernel_manifest = {
// DEPTH3: pipeline = {budget_depth = 0 : i64, depth = 0 : i64, reason = "serial-requested", requested = 3 : i64, selected = "serial"}

// Forced depth 1: one scratch, one slot, one status word, and the serial
// source loop -- issue, await, compute -- left unpipelined (no iter_args, no
// prologue, no epilogue).
// DEPTH1-LABEL: func.func @depth
// DEPTH1: %[[SCRATCH:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
// DEPTH1: %[[SLOT:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// DEPTH1: %[[ST:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// DEPTH1-NOT: memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// DEPTH1: scf.for %[[M:.*]] = {{.*}} {
// DEPTH1: %[[ROW:.*]] = arith.muli %[[M]], {{.*}} : index
// DEPTH1: %[[T:.*]] = hmx.stage ins(%arg0, %[[ROW]] : memref<128x1024xf16>) outs(%[[SLOT]], %[[ST]] : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
// DEPTH1: %[[READY:.*]] = hmx.await ins(%[[T]] : i32) outs(%[[SLOT]] : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
// DEPTH1: hmx.pack_act ins(%[[READY]], {{.*}}, {{.*}} : memref<32x1024xf16, 1>) outs(%[[SCRATCH]] : memref<1x32x16x32x2xf16, 1>) {count = 32 : i64, hmx.decision_id = 0 : i64}
// DEPTH1: hmx.mma %[[SCRATCH]], {{.*}}, {{.*}}, {{.*}}, {{.*}} {n_croutons = 32 : i32}
// DEPTH1-NOT: hmx.matmul

// Forced depth 2 with the default budget: the double ring fits, so two slots
// and two status words are allocated; the pipeliner's prologue issues tile 0
// through the parity slot select and the kernel carries the (token, slot) pair
// the await consumes as iter_args.
// DEPTH2-LABEL: func.func @depth
// DEPTH2: %[[SCRATCH:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
// DEPTH2: %[[SLOT0:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// DEPTH2: %[[ST0:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// DEPTH2: %[[SLOT1:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x1024xf16, 1>
// DEPTH2: %[[ST1:.*]] = memref.alloc() {alignment = 4 : i64} : memref<1xi32>
// DEPTH2: %[[SSEL0:.*]] = arith.select {{.*}}, %[[SLOT1]], %[[SLOT0]] : memref<32x1024xf16, 1>
// DEPTH2: %[[T0:.*]] = hmx.stage ins(%arg0, {{.*}} : memref<128x1024xf16>) outs(%[[SSEL0]], {{.*}} : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
// DEPTH2: scf.for {{.*}} iter_args(%[[T:.*]] = %[[T0]], %[[S:.*]] = %[[SSEL0]]) -> (i32, memref<32x1024xf16, 1>) {
// DEPTH2: %[[READY:.*]] = hmx.await ins(%[[T]] : i32) outs(%[[S]] : memref<32x1024xf16, 1>) -> memref<32x1024xf16, 1>
// DEPTH2: hmx.pack_act ins(%[[READY]], {{.*}}, {{.*}} : memref<32x1024xf16, 1>) outs(%[[SCRATCH]] : memref<1x32x16x32x2xf16, 1>) {count = 32 : i64, hmx.decision_id = 0 : i64}
// DEPTH2: hmx.mma %[[SCRATCH]], {{.*}}, {{.*}}, {{.*}}, {{.*}} {n_croutons = 32 : i32}
// DEPTH2-NOT: hmx.matmul

// Forced serial (pipeline-depth=3): no staging rewrite at all. The activation
// bridge and its whole crouton array survive (the mmas read that array), so
// there is no crouton-row scratch and no ring -- and, by the implicit checks on
// the RUN line, no hmx.stage/hmx.await anywhere in the module. What is left is
// the plain (m, n) tile nest.
// DEPTH3-LABEL: func.func @depth
// DEPTH3: hmx.bias_init
// The activation array is kept: it is what the bridge fills and the mmas read.
// DEPTH3: %[[ACT:.*]] = memref.alloc() : memref<4x32x16x32x2xf16, 1>
// DEPTH3: scf.for {{.*}} {
// DEPTH3: hmx.pack_act ins(%{{.*}} : memref<128x1024xf16>) outs(%[[ACT]] : memref<4x32x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
// DEPTH3: %[[W:.*]] = memref.alloc() : memref<2x32x16x32x2xf16, 1>
// DEPTH3: %[[ACC:.*]] = memref.alloc() : memref<4x2x16x32x2xf16, 1>
// DEPTH3: scf.for %[[M:.*]] = {{.*}} to {{.*}} step
// DEPTH3: scf.for %[[N:.*]] = {{.*}} to {{.*}} step
// DEPTH3: hmx.acc_clear
// DEPTH3-NOT: scf.for
// DEPTH3: %[[K:.*]] = arith.constant 0 : index
// DEPTH3-NEXT: hmx.mma %[[ACT]], %[[W]], %[[M]], %[[N]], %[[K]] {n_croutons = 32 : i32}
// DEPTH3: hmx.acc_read %{{.*}}, %[[ACC]], %[[M]], %[[N]] {bias_set = 0 : i32}
// DEPTH3-NOT: hmx.matmul

module attributes {hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 128 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64}}, full = {k = 1024 : i64, m = 128 : i64, n = 64 : i64}, function = "depth", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 1024 : i64}, m = {kind = "static", value = 128 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 1024 : i64, m = 128 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 409600 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "depth", slot = 1 : i64}}, workspace_class = "runtime-internal"}], pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, schema = "hex.hmx.kernel_manifest/v2", unpack_sites = 1 : i64, weight_policies = []}} {
func.func @depth(%a: memref<128x1024xf16>, %w: memref<1024x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %ca = memref.alloc() : memref<4x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c128 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<128x1024xf16>) outs(%ca : memref<4x32x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %ar = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  // A forced depth-2 request that the budget cannot pay for narrows to the
  // serial ring and says so. The committed arrays are 409856 bytes (activation
  // 4x32x16x32x2 262144, weight 2x32x16x32x2 131072, accumulator 4x2x16x32x2
  // 16384, state 256) and the staged path frees the 262144-byte activation, so
  // the room is 300592 - 409856 + 262144 = 152880: the scratch plus the serial
  // ring is 131076, the scratch plus the double ring is 196616.
  // expected-remark @+1 {{HMX pipeline depth 2 requested, but only 152880 bytes of VTCM are free after the crouton scratch: a depth-1 ring needs 131076 bytes; using depth 1}}
  hmx.matmul ins(%ca, %cw : memref<4x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<4x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  memref.dealloc %ca : memref<4x32x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x32x16x32x2xf16, 1>
  memref.dealloc %ar : memref<4x2x16x32x2xf16, 1>
  return
}
}
