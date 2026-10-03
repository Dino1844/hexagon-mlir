//===- hmx-partition-unpack-hoist-late-dst.mlir - a late destination -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// WHERE THE READ-OUT HOIST DIES, AND WHY THIS FILE EXISTS
// ------------------------------------------------------
// hmx-partition moves the accumulator read-out into the m-tile loop so it can
// overlap the engine's next tiles (HmxPartitionPass.cpp:990-1083). The clone is
// emitted at the `hmx.matmul`, which in production sits BEFORE the separate
// read-out loop, so everything the read-out captures must be available there.
//
// A Triton block pointer with `offsets=(0, 0)` materialises its destination
// memref in the entry block, so it dominates and the hoist works. The same
// kernel with `offsets=(pid * BLOCK_M, 0)` does not: the destination offset
// depends on `tl.program_id`, so `materialize_in_destination` leaves the
// `memref.reinterpret_cast` where it built it -- AFTER the matmul -- and the
// hoist declined. Nothing was wrong with the read-out and nothing was wrong
// with the batching pass downstream: there was simply no in-loop read-out left
// for it to attach to, and `hmx-vector-readout` declined in turn, silently,
// which is why seven measurement sweeps read "no speedup" for a feature that was
// absent from every kernel they compiled. Measured 2026-10-03 on the real
// 1024x512x64 matmul: moving that one `reinterpret_cast` above the matmul was
// the whole difference between 5 read-out symbols in the object and 0.
//
// `memref.reinterpret_cast` is `Pure` (MemRefOps.td:1490), so it may always be
// evaluated earlier, and no canonicalizer runs between
// `convert-bufferization-to-memref` and this pass (LinalgToLLVMPass.cpp:502-537)
// to evaluate it for us. So this pass hoists it instead of declining: a
// relocation of descriptor arithmetic, not a repair of the read-out's shape.
//
// The two arms are the point. `@late_dst` is the production grid shape and must
// hoist. `@late_alloc_dst` differs ONLY in that its destination is a
// `memref.alloc`, which has an `Allocate` memory effect and is therefore not
// `Pure`; it must still decline, with the separate read-out loop intact. A
// hoister that accepted everything would pass the first arm and fail the second,
// so the second arm is what makes the first one mean something.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2}))' | FileCheck %s
//===----------------------------------------------------------------------===//

// The destination's `reinterpret_cast` and the offset arithmetic under it are
// hoisted above every loop in the function -- the NOT runs from the start of
// the output to that cast, so it says the cast is in the entry block rather
// than merely somewhere above the tile loop. The hoisted `unpack_acc` then rides
// the tile loop and the separate read-out loop is retired with the matmul.
module attributes {hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 128 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64}}, full = {k = 64 : i64, m = 128 : i64, n = 64 : i64}, function = "late_dst", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 128 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 64 : i64, m = 128 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 40960 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "late_dst", slot = 1 : i64}}, workspace_class = "runtime-internal"}, {dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 128 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64}}, full = {k = 64 : i64, m = 128 : i64, n = 64 : i64}, function = "late_alloc_dst", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 128 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 64 : i64, m = 128 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 40960 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "late_alloc_dst", slot = 1 : i64}}, workspace_class = "runtime-internal"}], pack_act_sites = 2 : i64, pack_weight_sites = 2 : i64, schema = "hex.hmx.kernel_manifest/v2", unpack_sites = 2 : i64, weight_policies = []}} {
func.func @late_dst(%a: memref<128x64xf16>, %w: memref<64x64xf16>, %raw: memref<*xf16>, %pid: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %ca = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<128x64xf16>) outs(%ca : memref<4x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %cw = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c2 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<64x64xf16>) outs(%cw : memref<2x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %ar = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<4x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
             outs(%ar : memref<4x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  // THE PRODUCTION GRID CASE. `%out` is the whole-matrix kernel's destination
  // with the `tl.program_id` offset in it: `arith.muli` under a
  // `memref.reinterpret_cast`, both defined AFTER the matmul. Both are pure, so
  // both move above it. The row is still the read-out loop's own induction
  // variable, which is the identity the hoist turns on.
  %off = arith.muli %pid, %c64 : index
  %out = memref.reinterpret_cast %raw to offset: [%off], sizes: [128, 64], strides: [64, 1]
      : memref<*xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
  scf.for %i = %c0 to %c4 step %c1 {
    %r = hmx.unpack_acc ins(%ar, %i, %c0 : memref<4x2x16x32x2xf16, 1>)
        outs(%out : memref<128x64xf16, strided<[64, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<128x64xf16, strided<[64, 1], offset: ?>>
  }
  memref.dealloc %ca : memref<4x2x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x2x16x32x2xf16, 1>
  memref.dealloc %ar : memref<4x2x16x32x2xf16, 1>
  return
}

// Nothing above this marker belongs to the hoisting arm; the checks that follow
// are the decline arm's.
// -----

// The negative arm. Identical to `@late_dst` in every respect except that the
// destination is a `memref.alloc` after the matmul instead of a pure cast over a
// function argument. An `alloc` has an `Allocate` memory effect, so it is not
// `Pure` and there is no version of "move it earlier" that means anything: the
// buffer must not exist before the matmul that produces its contents.
//
// So the hoist declines, and the consequence is the pre-existing shape: the
// read-out stays a separate loop walking AR rows after the tile loop has
// retired. `hmx-vector-readout` then has nothing to attach to and declines in
// turn -- a silent no-op, which is the whole failure mode this file is about and
// the reason the object-level gate exists rather than an IR-level one.
func.func @late_alloc_dst(%a: memref<128x64xf16>, %w: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %ca = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<128x64xf16>) outs(%ca : memref<4x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %cw = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c2 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<64x64xf16>) outs(%cw : memref<2x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %ar = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<4x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
             outs(%ar : memref<4x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  %out = memref.alloc() : memref<128x64xf16>
  scf.for %i = %c0 to %c4 step %c1 {
    %r = hmx.unpack_acc ins(%ar, %i, %c0 : memref<4x2x16x32x2xf16, 1>)
        outs(%out : memref<128x64xf16>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<128x64xf16>
  }
  memref.dealloc %out : memref<128x64xf16>
  memref.dealloc %ca : memref<4x2x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x2x16x32x2xf16, 1>
  memref.dealloc %ar : memref<4x2x16x32x2xf16, 1>
  return
}
}

// The positive arm. The destination cast is emitted BEFORE the staged tile loop
// -- that ordering is the whole claim, and it is what the matmul's original
// position produced: the cast was materialised after the matmul, so before this
// change it printed after the tile loops instead. The `iter_args` on the tile
// loop is what makes that loop the staged one rather than the activation pack
// loop, which also sits in this function and also runs earlier.
 // CHECK-LABEL: func.func @late_dst
// CHECK: %[[CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [{{.*}}]
// CHECK: scf.for %{{.*}} iter_args
// The hoisted clone reads that same cast, so the value the read-out captures is
// the one that moved -- not a coincidence of two casts.
// CHECK: hmx.unpack_acc ins(%{{.*}}) outs(%[[CAST]]
// CHECK-NOT: hmx.matmul

// The negative arm, which is what stops the check above from being a statement
// about nothing. The alloc destination is NOT hoisted: it is emitted after the
// tile loops, where it started, and the read-out is therefore still a separate
// loop walking AR rows after the tile loop has retired -- the shape
// `hmx-vector-readout` declines on. `unpack_sites` is 1 either way, so the
// count cannot tell the two apart and the ordering below is the check.
// CHECK-LABEL: func.func @late_alloc_dst
// CHECK: scf.for %{{.*}} iter_args
// CHECK: memref.alloc() : memref<128x64xf16>
// CHECK: scf.for %{{.*}} {
// CHECK: hmx.unpack_acc