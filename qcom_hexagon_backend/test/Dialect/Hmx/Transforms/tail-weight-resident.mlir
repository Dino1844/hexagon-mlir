//===- tail-weight-resident.mlir - M<32 tail accepts a weight-resident weight --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and its subsidiaries.
// SPDX-License-Identifier: BSD-2.0-ONLY
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' \
// RUN:   | FileCheck %s --implicit-check-not=hmx.pack_weight
//===----------------------------------------------------------------------===//
//
// This is the flip of the former `tail-weight-resident-reject.mlir`, which pinned
// the OLD behaviour (refuse). 2026-10-01: that refusal made the whole
// diagnostic tail path unreachable for the DEFAULT configuration --
// `enableWeightResident` is True by default (hexagon_options.py), and a resident
// weight arrives already in crouton form, so no `hmx.pack_weight` writes it and
// `findPackBridge(rhs, isWeight=true)` necessarily returned null.
//
// The fix tells the two reasons for that null apart:
//   * no PackWeightOp writes the array => already resident => ACCEPT, and the
//     weight-side source checks are skipped because there IS no source;
//   * a PackWeightOp writes it but the bridge is malformed => REFUSE, unchanged.
// The malformed case is still covered by `tail-bridge-loop-reject.mlir`.
//
// 33x33 is the tail shape the sibling tail tests use (33 = 32 full + 1 tail
// lane). The activation has a DIRECT pack bridge; the weight is a block argument
// already in crouton form.

// CHECK-LABEL: func.func @tail_weight_resident
// The tail plan must now LOWER: a 2x2 tile grid => 4 mma, each preceded by an
// accumulator clear and followed by a read, plus a bounds-safe unpack for the
// 33x33 edge. The three op kinds INTERLEAVE per tile, so the counts are DAG
// matches; CHECK-COUNT would demand adjacency and fail.
// CHECK: hmx.bias_init
// The resident weight must NOT be packed: there is no row-major source, so a
// hmx.pack_weight here would read a value that does not exist. That is the
// substantive difference from the rejected form, which emitted nothing at all
// because the path bailed before lowering.
//
// POSITION MATTERS, and getting it wrong makes this assertion VACUOUS.
// 2026-10-01: an independent audit measured that with this CHECK-NOT placed
// AFTER the CHECK-DAG group, the test still passed on an output containing
// FOUR hmx.pack_weight. FileCheck's cursor sits past the DAG block by then, so
// the NOT's range starts after the packs and never covers them. Verified with a
// mutant that gives the weight a real 4-pack bridge.
//
// Hence two things: the NOT lives here, immediately after the first positive
// match, AND the RUN line carries --implicit-check-not so the ban holds over the
// WHOLE input regardless of where the cursor ends up. The second one is the
// load-bearing half; the CHECK-NOT is kept for the reader.
// CHECK-NOT: hmx.pack_weight
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.acc_clear
// CHECK-DAG: hmx.mma
// CHECK-DAG: hmx.mma
// CHECK-DAG: hmx.mma
// CHECK-DAG: hmx.mma
// CHECK-DAG: hmx.acc_read
// CHECK-DAG: hmx.acc_read
// CHECK-DAG: hmx.acc_read
// CHECK-DAG: hmx.acc_read
// The 33x33 edge is 32 full tiles plus a 1x1 corner, so the corner unpack must
// be a BOUNDS-SAFE store. The attribute dict is printed alphabetically
// (n_tile, valid_cols, valid_rows), so one regex pins all three. Matching a
// bare `hmx.unpack_acc` would hit the full-tile unpack first -- that one
// legitimately carries no bounds, so a CHECK-SAME after it would be testing
// nothing.
// CHECK: hmx.unpack_acc{{.*}}n_tile = 1{{.*}}valid_cols = 1{{.*}}valid_rows = 1
// CHECK-NOT: hmx.matmul
func.func @tail_weight_resident(%a: memref<33x33xf16>,
                               %cw: memref<2x2x16x32x2xf16, 1>,
                               %out: memref<33x33xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.pack_act ins(%a, %c0, %c0 : memref<33x33xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c0 : memref<33x33xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c0, %c1 : memref<33x33xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c1 : memref<33x33xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.matmul ins(%act, %cw : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}
  hmx.unpack_acc ins(%ar, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  return
}
