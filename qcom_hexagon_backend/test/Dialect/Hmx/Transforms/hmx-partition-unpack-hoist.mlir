//===- hmx-partition-unpack-hoist.mlir - the read-out rides the tile loop ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The accumulator read-out used to be a strictly serial tail: the staged tile
// loop retired, and a second loop after it walked every AR row and unpacked it.
// LWP measured that tail at 39.79% of a 1024x512x64 kernel at depth 2 -- about
// 24 us of vector work overlapping nothing, because after the loop there is
// nothing left to overlap with.
//
// hmx-partition therefore moves one `hmx.unpack_acc` into the tail of each
// tile-loop iteration and retires the separate loop. The move is legal by
// construction, not by scheduling luck:
//
//   * `hmx.mma` writes no memory. Its accumulator is the implicit engine
//     register, carried on `Hmx_EngineResource`. The accumulator's memory image
//     is written only by `hmx.acc_read`, at AR (row m, col n); `unpack_acc` reads
//     AR (row m, col c). `croutonAddr` is `base + (row*stride + col) * 2048`, so
//     different rows are byte-disjoint croutons.
//   * `hmx.unpack_acc` carries only default-resource MemRead/MemWrite. It has no
//     `Hmx_EngineResource` effect, so it neither conflicts with nor must be
//     ordered against `acc_clear`/`mma`/`acc_read` on the implicit register.
//   * Inside one iteration the unpack is emitted after the inner N loop, which
//     is where the `acc_read`s filling AR row `m` live: the read follows the
//     write in the same block.
//
// Row coverage needs no separate argument. The hoisted op sits in the same
// iteration body as the `acc_read` it reads, so it inherits exactly the rows the
// pipeliner already guarantees `acc_read` covers -- the kernel over m in
// [0, Mt-1) and, at depth 2, the peeled epilogue's last row. DEPTH2 below pins
// that epilogue: the unpack is cloned there with the same `%[[LAST]]` row the
// epilogue's `acc_read` uses, so the peeled row is unpacked exactly once and no
// row is unpacked twice.
//
// Depth 1 is the unpipelined source loop, so one unpack per iteration covers all
// Mt rows directly. Depth 3 skips staging entirely and keeps the separate loop:
// that arm reproduces the pre-pipeline codegen and is the A/B baseline, so its
// shape is left exactly as it was (see DEPTH3 below).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=1}))' | FileCheck %s --check-prefix=DEPTH1
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2}))' | FileCheck %s --check-prefix=DEPTH2
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=0}))' | FileCheck %s --check-prefix=AUTO
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=3}))' | FileCheck %s --check-prefix=DEPTH3
//===----------------------------------------------------------------------===//

// At depth 1 the staged loop is the unpipelined source loop, so one unpack per
// iteration covers all Mt rows and there is no epilogue to account for. The
// read-out site count stays 1 and `hmx.matmul` is gone: the clone carries
// `hmx.decision_id = 0` verbatim, which is what keeps
// refreshHmxManifestBridgeCounts attributing it to the same manifest record.
//
// DEPTH1: bridge_counts = {{.*}}pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64
// DEPTH1: pipeline = {{.*}}depth = 1 : i64, requested = 1 : i64, selected = "staged"
// DEPTH1-LABEL: func.func @hoist
// DEPTH1: scf.for %[[M:.*]] = {{.*}} {
// DEPTH1: hmx.stage
// DEPTH1: hmx.acc_read {{.*}} %[[ACC:.*]], %[[M]], {{.*}} {bias_set = 0 : i32}
// DEPTH1: hmx.unpack_acc ins(%[[ACC]], %[[M]], {{.*}} : {{.*}}) outs({{.*}}) {count = 16 : i64, hmx.decision_id = 0 : i64}
// The read-out is inside the tile loop and nowhere else: the separate loop is
// retired, so no second unpack follows the tile loop's deallocations. The NOT
// runs to the next LABEL, which is what stops it from also excluding @deep_k's
// own (identical) hoisted read-out further down the module.
// DEPTH1-NOT: hmx.unpack_acc
// DEPTH1-NOT: hmx.matmul
// DEPTH1-LABEL: func.func @deep_k
// DEPTH1: hmx.unpack_acc
// DEPTH1-NOT: hmx.matmul

// At depth 2 the pipeliner clones the loop body into both the kernel and the
// peeled epilogue, so the read-out is cloned with it. That is what makes the
// coverage argument short: the hoisted unpack sits in the same body as the
// `acc_read` it reads, so it inherits exactly the rows `acc_read` already
// covers. The two checks below are that argument -- the kernel's unpack reads
// its own induction variable, and the epilogue's reads exactly the row the
// epilogue's `acc_read` just wrote, the peeled last row. Rows [0, Mt-1) and
// Mt-1: every row once, none twice.
//
// DEPTH2: bridge_counts = {{.*}}unpack_sites = 2 : i64
// DEPTH2: pipeline = {{.*}}depth = 2 : i64, requested = 2 : i64, selected = "staged"
// DEPTH2-LABEL: func.func @hoist
// DEPTH2: scf.for %[[M:.*]] = {{.*}} iter_args
// In the kernel both rows are the loop's own induction variable reached through
// the tile loop's `m * 1 + 0` row arithmetic, so the acc_read writes AR row %[[M]]
// and the read-out that follows reads it. The three-step capture is what pins
// each to %[[M]]: the `arith.constant 0` factor rules out the loop's other row
// expressions (the parity addi multiplies by 1, not 0), and the `addi` names the
// induction variable outright. Both rows are checked separately because they are
// distinct SSA values computing the same index.
// DEPTH2: %[[KZ:.*]] = arith.constant 0 : index
// DEPTH2-NEXT: %[[K1:.*]] = arith.muli %{{.*}}, %[[KZ]] : index
// DEPTH2-NEXT: %[[KROW:.*]] = arith.addi %[[M]], %[[K1]] : index
// DEPTH2: hmx.acc_read {{.*}} %[[ACC:.*]], %[[KROW]], {{.*}} {bias_set = 0 : i32}
// DEPTH2: %[[UZ:.*]] = arith.constant 0 : index
// DEPTH2-NEXT: %[[U1:.*]] = arith.muli %{{.*}}, %[[UZ]] : index
// DEPTH2-NEXT: %[[UROW:.*]] = arith.addi %[[M]], %[[U1]] : index
// DEPTH2-NEXT: hmx.unpack_acc ins(%[[ACC]], %[[UROW]], {{.*}} : {{.*}}) outs({{.*}}) {count = 16 : i64, hmx.decision_id = 0 : i64}
// The peeled epilogue: an await and a compute for the last tile, with the
// read-out on that tile's row. Here the row is one SSA value reused by name, so
// the check is literal equality rather than a pattern -- this is the row the
// kernel cannot cover, and pinning it to the epilogue's own acc_read is the
// peeled-row half of the coverage argument.
// DEPTH2: hmx.await
// DEPTH2: hmx.acc_read {{.*}} %[[ACC]], %[[LAST:.*]], {{.*}} {bias_set = 0 : i32}
// DEPTH2: hmx.unpack_acc ins(%[[ACC]], %[[LAST]], {{.*}} : {{.*}}) outs({{.*}}) {count = 16 : i64, hmx.decision_id = 0 : i64}
// DEPTH2-NOT: hmx.matmul
// @deep_k stages at depth 2 under this knob too, and is hoisted the same way:
// the two unpack sites in the module-wide count above are @hoist's kernel and
// epilogue; this is @deep_k's pair, so the total is 4 and both functions agree.
// DEPTH2-LABEL: func.func @deep_k
// DEPTH2: hmx.unpack_acc
// DEPTH2: hmx.unpack_acc
// DEPTH2-NOT: hmx.matmul

// `auto` on @hoist declines staging -- Kt = 1 is below the auto K floor -- so
// the plain tile loop runs and the separate read-out loop stays exactly where it
// was. `auto` on @deep_k (Kt = 32, at the floor) does stage, and the read-out is
// hoisted exactly as at an explicit depth 2. The hoist therefore follows the
// *staging decision*, not the requested depth: these two functions differ only
// in K, and neither arm disturbs the other. Both manifest fields are checked up
// front because the manifest is one module-level attribute printed before
// either function body, so a check placed after a LABEL could not see them.
// AUTO: pipeline = {{.*}}reason = "shallow-k", requested = 0 : i64, selected = "serial"
// AUTO: pipeline = {{.*}}depth = 2 : i64, requested = 0 : i64, selected = "staged"
// AUTO-LABEL: func.func @hoist
// The activation bridge survives, exactly as in the pre-pipeline codegen.
// AUTO: hmx.pack_act
// AUTO-NOT: hmx.stage
// The tile loop is the plain m/n nest reading the whole packed activation array
// (%alloc_0, the array `pack_act` filled above -- not a crouton-row scratch, which
// only the staged path allocates), and the read-out loop follows it unchanged.
// AUTO: hmx.mma %alloc_0,
// AUTO: scf.for {{.*}} {
// AUTO: hmx.unpack_acc
// @deep_k stages, so it carries one unpack in the pipelined kernel and one in
// the peeled epilogue, and its separate read-out loop is gone.
// AUTO-LABEL: func.func @deep_k
// AUTO: hmx.stage
// AUTO: hmx.unpack_acc
// AUTO: hmx.unpack_acc
// AUTO-NOT: hmx.matmul

// `pipeline-depth=3` skips staging entirely: this is the A/B baseline arm that
// reproduces the pre-pipeline codegen, and the separate read-out loop stays.
// AUTO is not reused here so the two arms cannot be confused for each other.
// DEPTH3-LABEL: func.func @hoist
// DEPTH3-NOT: hmx.stage
// DEPTH3: scf.for {{.*}} {
// DEPTH3: hmx.unpack_acc
// DEPTH3-NOT: hmx.matmul

module attributes {hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 128 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64}}, full = {k = 64 : i64, m = 128 : i64, n = 64 : i64}, function = "hoist", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 128 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 64 : i64, m = 128 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 40960 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "hoist", slot = 1 : i64}}, workspace_class = "runtime-internal"}, {dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 128 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64}}, full = {k = 1024 : i64, m = 128 : i64, n = 64 : i64}, function = "deep_k", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 1024 : i64}, m = {kind = "static", value = 128 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 1024 : i64, m = 128 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 532480 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "deep_k", slot = 1 : i64}}, workspace_class = "runtime-internal"}], pack_act_sites = 2 : i64, pack_weight_sites = 2 : i64, schema = "hex.hmx.kernel_manifest/v2", unpack_sites = 2 : i64, weight_policies = []}} {
func.func @hoist(%a: memref<128x64xf16>, %w: memref<64x64xf16>, %out: memref<128x64xf16>) {
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
  // The read-out this pass hoists: one unpack per AR row, over rows [0, Mt) =
  // [0, 4), reading the accumulator the matmul above writes.
  scf.for %i = %c0 to %c4 step %c1 {
    %r = hmx.unpack_acc ins(%ar, %i, %c0 : memref<4x2x16x32x2xf16, 1>) outs(%out : memref<128x64xf16>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<128x64xf16>
  }
  memref.dealloc %ca : memref<4x2x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x2x16x32x2xf16, 1>
  memref.dealloc %ar : memref<4x2x16x32x2xf16, 1>
  return
}

// The second fixture differs only in K. `@hoist` above is Kt = 1, below the
// auto K floor (kStageMinKTiles = 32), which is why it declines to stage under
// `pipeline-depth=0`. `@deep_k` is Kt = 32, so `auto` accepts it and the
// read-out is hoisted. Together the two functions show that the hoist follows
// the *staging decision*, not the requested depth: on the shape that stages it
// happens, on the shape that does not it does not, and neither arm disturbs the
// other.
func.func @deep_k(%a: memref<128x1024xf16>, %w: memref<1024x64xf16>, %out: memref<128x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %ca = memref.alloc() : memref<4x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c128 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<128x1024xf16>) outs(%ca : memref<4x32x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c1024 step %c2 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %ar = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<4x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<4x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  scf.for %i = %c0 to %c4 step %c1 {
    %r = hmx.unpack_acc ins(%ar, %i, %c0 : memref<4x2x16x32x2xf16, 1>) outs(%out : memref<128x64xf16>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<128x64xf16>
  }
  memref.dealloc %ca : memref<4x32x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x32x16x32x2xf16, 1>
  memref.dealloc %ar : memref<4x2x16x32x2xf16, 1>
  return
}
}