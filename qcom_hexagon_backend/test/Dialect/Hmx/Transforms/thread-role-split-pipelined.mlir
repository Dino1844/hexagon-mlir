//===- thread-role-split-pipelined.mlir - the merged R2+R3 form ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The depth-2 PIPELINED form of the thread-role split: the merged R2+R3
// unit (docs/architecture/
// hmx-coscheduling-architecture-review-2026-10-09.md section 9's plan
// correction -- R2's readout coexistence and R3's pipelined matcher in one
// emission, because R2 alone lands ~10 us above the default face).
//
// The fixture is a Kt=32 matmul (128x64x1024, Mt=4, Nt=2) WITH a hoistable
// read-out: hmx-partition at depth 2 peels an epilogue, hmx-vector-readout
// (which runs BEFORE thread-role-partition in the production pipeline -- the
// order is load-bearing, see LinalgToLLVMPass.cpp) turns the read-out into
// publishes, and thread-role-partition splits what is left. What this file
// pins is the composed mechanism, in order:
//
//   SPLIT     the two-role form with the read-out coexistence, LAGGED one
//             submit batch: the steady loop keeps the DMA ring and pack
//             (into the rotating rows), the engine nest is GONE, and the
//             in-loop order inside one iteration is submit(batch) ->
//             wait_retired(previous boundary + 1) -> publish(previous
//             batch's group). The publish of the group ending at tile m
//             fires at iteration m + batch -- the NEXT submit boundary --
//             so the wait targets a batch the ring has had for a full batch
//             of iterations (the zero-lag form waited on the batch the same
//             iteration had just submitted, serializing producer behind
//             engine: +10.0 us / +27% on the S1-class shape, see
//             docs/results/r2r3-ab-2026-10-09.md and the emission comment in
//             ThreadRolePartition.cpp). The peeled epilogue: await, pack
//             into rows[Mt-1], the tail submit (which carries the peeled
//             tile as its last descriptor -- the "peel tail into the
//             section" property), the exit drain, then the read-out split's
//             tail publish with NO wait of its own: behind the drain, every
//             group is provably retired, which is the proof the tail wait
//             used to stand in for. One work function covers every tile:
//             its per-tile loop derives the row from the descriptor's
//             rowStart, so the peeled tile is just the last descriptor.
//   SPLIT12   the same pipeline on a 12-tile shape (384x64x1024, Mt=12),
//             where the steady loop actually REACHES publish boundaries: the
//             lagged publish fires in the loop (at iv=7, publishing group
//             [0..4) behind wait_retired(4)), and the LAST live group
//             [4..8) CLAMPS -- its lagged position (iv=11) is past the
//             loop, so it publishes at the tail-submit boundary instead,
//             keeping its wait_retired(8): submit -> wait -> publish ->
//             drain -> tail publish (rows [8..12), no wait) -> exec drain.
//   WIRE      the same shape through convert-func-to-llvm and hmx-to-llvm:
//             the exported entry point and depth object, the thread
//             contract on the work function, and -- load-bearing for the
//             lock migration -- NO ensure/unlock pair anywhere.
//   NOREADOUT the pipelined form with the read-out split off. The read-outs
//             move past the exit drain (the first form's behavior -- the
//             only producer-side point where every tile's engine work is
//             provably done without a wait), and NO wait_retired CALL is
//             emitted (nothing in the loop reads `ar`, so there is nothing
//             to prove to). The declaration may exist; a call may not.
//   DECLINE   the coverage gate: a read-out batch (2) finer than the role
//             submit batch (4) puts publishes at iterations that are not
//             submit boundaries (and could clamp several groups past the
//             loop at once) -- a form the lagged emission's position
//             argument does not describe, so the split declines loudly and
//             the kernel keeps the single-thread pipeline with its read-out
//             intact.
//   OFF       the byte-laziness arm: the same shape with
//             thread-role-partition absent. No submit, no drain, no wait,
//             no work function, no handoff record.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2}, hmx-vector-readout, thread-role-partition))' | FileCheck %s --check-prefix=SPLIT
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2}, hmx-vector-readout, thread-role-partition), convert-scf-to-cf, convert-func-to-llvm, hmx-to-llvm)' | FileCheck %s --check-prefix=WIRE
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2}, thread-role-partition))' | FileCheck %s --check-prefix=NOREADOUT
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2}, hmx-vector-readout{hmx-readout-batch=2}, thread-role-partition))' 2>%t.err | FileCheck %s --check-prefix=DECLINE
// RUN: FileCheck %s --check-prefix=DECLINE-MSG < %t.err
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=2}, hmx-vector-readout))' | FileCheck %s --check-prefix=OFF
//===----------------------------------------------------------------------===//

// == The manifest verdict and both handoff records == -----------------
//
// The read-out handoff (hmx.readout.handoffs) sits beside the role handoff
// (hmx.role.handoffs): the two splits compose, and the depth recorded for
// the role ring is the tile count (4) -- one row per tile in the rotating
// scratch.
// SPLIT: module attributes {{.*}}topology = "role-split-ok"
// SPLIT-SAME: hmx.readout.handoffs = [{ar = memref<4x2x16x32x2xf16, 1>
// SPLIT-SAME: hmx.role.handoffs = [{ar = memref<4x2x16x32x2xf16, 1>, bias = memref<256xi8, 1>, depth = 4 : i64, engine = "deep_k", rows = memref<4x32x16x32x2xf16, 1>, work = "__hmx_role_section", wt = memref<2x32x16x32x2xf16, 1>}]

// The per-launch buffer table: four internal words.
// SPLIT: llvm.mlir.global internal @__hmx_role_ar
// SPLIT: llvm.mlir.global internal @__hmx_role_bias
// SPLIT: llvm.mlir.global internal @__hmx_role_wt
// SPLIT: llvm.mlir.global internal @__hmx_role_rows

// The producer kernel.
// SPLIT-LABEL: func.func @deep_k(

// The rotating rows replace the one-row scratch: one row per tile, so the
// producer never overwrites a row the consumer has not retired.
// SPLIT: memref.alloc() : memref<4x32x16x32x2xf16, 1>

// The steady loop KEEPS the pipeliner's shape: the (token, slot) iter_args
// ride through untouched (the DMA ring is engine-independent, producer-side
// by design), and the issue for the NEXT tile still leads the iteration.
// SPLIT: scf.for {{.*}} iter_args({{.*}}) -> (i32, memref<32x1024xf16, 1>) {
// SPLIT: hmx.stage
// SPLIT: hmx.await
// The pack lands in ROW m of the rotating rows, through a one-row view (the
// source-meaning rule: the row operand names the SOURCE block too, so the
// destination moves by view, never by operand).
// SPLIT: memref.subview {{.*}}[{{.*}}, 0, 0, 0, 0] [1, 32, 16, 32, 2]
// SPLIT: hmx.pack_act

// THE LAGGED PROTOCOL CHAIN, in emission order inside one steady iteration:
// the batched submit (with its short-return recovery), then the lagged
// publish-if. Its guard is the read-out split's boundary test evaluated one
// batch late -- mPrev = iv - batch, boundary = (mPrev+1) % G == 0 -- plus
// `mPrev+1 >= G`, which keeps the first boundary publishless (group 0
// publishes at the SECOND boundary; at mPrev+1 = 0 the remainder test holds
// but the group would start below row 0). Inside: the wait -- target
// mPrev+1, the previous boundary's group end, one full batch behind the
// iteration, so the engine has been chewing it since before the producer
// packed this batch -- then the boundary block's own stores and call, with
// the rowStart re-based to mPrev - (G-1). (This 4-tile fixture never
// reaches a boundary -- the loop stops at m=2 -- so the block is statically
// present and dynamically dead; SPLIT12 below exercises it firing.)
// SPLIT: call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: %[[MPREV:.*]] = arith.subi %{{.*}}, %{{.*}} : index
// SPLIT: %[[DONE:.*]] = arith.addi %[[MPREV]], %{{.*}} : index
// SPLIT: %{{.*}} = arith.remsi %[[DONE]], %{{.*}} : index
// SPLIT: %{{.*}} = arith.cmpi sge, %[[DONE]], %{{.*}} : index
// SPLIT: %[[LAG:.*]] = arith.andi %{{.*}}, %{{.*}} : i1
// SPLIT: scf.if %[[LAG]] {
// SPLIT: %[[TGT:.*]] = arith.index_cast %[[DONE]] : index to i32
// SPLIT: call @hexagon_runtime_hmx_role_wait_retired(%[[TGT]]) : (i32) -> ()
// SPLIT: %[[ROW:.*]] = arith.subi %[[MPREV]], %{{.*}} : index
// SPLIT: call @hexagon_runtime_hmx_exec_publish({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: scf.yield

// The engine nest is GONE from the producer -- in the steady loop AND after
// it. No acc_clear, mma or acc_read anywhere in the kernel.
// SPLIT-NOT: hmx.acc_read
// SPLIT-NOT: hmx.mma
// SPLIT-NOT: hmx.acc_clear

// The peeled epilogue: the await and the pack into rows[Mt-1] stay
// producer-side, and the TAIL SUBMIT carries the peeled tile as its last
// descriptor (rowStart 3 rides the same 24-byte descriptor as every other
// tile -- the section's (rows, wt, bias, ar, m0, count) signature covers it
// with no special case). Then the kernel's own exit barrier (the role
// drain: no section writes into caller memory after it returns), the
// read-out split's tail publish, and the vector drain (the read-out's own
// barrier, before the AR release). NO tail wait any more: the drain is the
// all-groups proof -- and on this 4-tile shape no clamped group exists
// (the loop never reaches a publish boundary, so the tail publish is the
// only one after the loop).
// SPLIT: hmx.await
// SPLIT: memref.subview {{.*}}[{{.*}}] [1, 32, 16, 32, 2]
// SPLIT: hmx.pack_act
// SPLIT: call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: call @hexagon_runtime_hmx_role_drain() : () -> ()
// SPLIT: call @hexagon_runtime_hmx_role_drain() : () -> ()
// SPLIT-NOT: call @hexagon_runtime_hmx_role_wait_retired
// SPLIT: call @hexagon_runtime_hmx_exec_publish({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: call @hexagon_runtime_hmx_exec_drain() : () -> ()

// The work function: private, region-marked, the four buffers as parameters
// plus (m0, count); the bias init moves in with it (engine work -- the
// producer must never issue an engine instruction); and the per-tile loop
// derives the row from m0 + i, so ONE section covers the steady tiles and
// the peeled tile alike -- the last descriptor is just rowStart = 3.
// SPLIT: func.func private @__hmx_role_section({{.*}}: memref<4x32x16x32x2xf16, 1>, {{.*}}) attributes {hex.thread_role = "hmx"}
// SPLIT: hmx.bias_init
// SPLIT: scf.for {{.*}} {
// SPLIT: %[[ROW:.*]] = arith.addi {{.*}}, {{.*}} : index
// SPLIT: hmx.mma {{.*}}, %[[ROW]],
// SPLIT: hmx.acc_read {{.*}}, %[[ROW]],

// == The wiring, at the LLVM level == -------------------------------------------------
//
// The producer's chain survives the conversion: submit, wait, publish, and
// both exit barriers, in that order.
// WIRE: llvm.call @hexagon_runtime_hmx_role_submit
// WIRE: llvm.call @hexagon_runtime_hmx_role_wait_retired
// WIRE: llvm.call @hexagon_runtime_hmx_exec_publish
// WIRE: llvm.call @hexagon_runtime_hmx_role_drain
// WIRE: llvm.call @hexagon_runtime_hmx_exec_drain

// The work function kept the thread contract (upstream PR #222340's TTI
// hooks -- the S0 backport), and the exported channel symbols are there for
// the launch-side probe: the entry point unpacking the frozen 24-byte
// descriptor onto the work function's parameters, and the companion depth
// object holding the tile count.
// WIRE: llvm.func @__hmx_role_section({{.*}}) attributes {{.*}}passthrough = ["hexagon_hmx"]
// WIRE: llvm.func @deep_k__hmx_section({{.*}}: !llvm.ptr, {{.*}}: i32)
// WIRE: llvm.mlir.global external constant @deep_k__hmx_role_depth(4 : i32)

// THE LOCK MIGRATION'S FIRST HALF: no ensure/unlock pair anywhere -- the
// bound thread's lifetime lock replaces the per-kernel pairing.
// WIRE-NOT: hexagon_runtime_hmx_ensure
// WIRE-NOT: hexagon_runtime_hmx_unlock
// WIRE-NOT: hmx.role.handoffs

// == The read-out split off == -------------------------------------
//
// Without publishes there is nothing to prove to inside the loop, so NO
// wait_retired CALL is emitted (the declaration is inert); the read-outs
// move past the exit drain -- the first form's behavior -- as one clone per
// row in a fresh loop, with the pipelined form's PEELED read-out dropped
// (the moved loop's last iteration covers that row; the matcher verified
// the pair differ only in the row).
// NOREADOUT-LABEL: func.func @deep_k(
// NOREADOUT: call @hexagon_runtime_hmx_role_submit
// NOREADOUT-NOT: call @hexagon_runtime_hmx_role_wait_retired
// NOREADOUT: call @hexagon_runtime_hmx_role_drain() : () -> ()
// The moved read-out: one unpack per row, in a loop after the drain.
// NOREADOUT: scf.for {{.*}} {
// NOREADOUT: hmx.unpack_acc
// NOREADOUT-NOT: hmx.mma
// NOREADOUT: func.func private @__hmx_role_section

// == The coverage gate decline == -------------------------------------------
//
// A read-out batch of 2 against a submit batch of 4: the lagged publish
// rides the submit boundary AFTER its group (submit(batch k+1) -> wait ->
// publish(batch k)), which only exists when every publish boundary is also
// a submit boundary -- and a G finer than the submit batch could also
// clamp several groups past the loop at once. So the split declines with
// the reason named, and the kernel keeps the single-thread pipelined form
// WITH its read-out (the publish machinery is still there; the role
// machinery is not).
// DECLINE: func.func @deep_k(
// DECLINE: call @hexagon_runtime_hmx_exec_publish
// DECLINE-NOT: call @hexagon_runtime_hmx_role_submit
// DECLINE-NOT: __hmx_role_section
// DECLINE-MSG: remark: thread-role split not applied: the read-out publishes every 2 tiles but the role submit batches every 4; the lagged publish rides the submit boundary after its group, which only exists when every publish boundary is a submit boundary. The kernel keeps the single-thread pipeline with its read-out intact

// == The byte-laziness arm == -------------------------------------------
//
// No split, no channel, no work function: the pipelined depth-2 form with
// its read-out thread exactly as hmx-partition and hmx-vector-readout
// emitted it.
// OFF-NOT: hexagon_runtime_hmx_role
// OFF-NOT: __hmx_role_section
// OFF-NOT: hmx.role.handoffs
// OFF-NOT: __hmx_role_rows
// OFF-LABEL: func.func @deep_k(
// OFF: hmx.mma
// OFF: call @hexagon_runtime_hmx_exec_publish

// == The 12-tile shape: the lagged publish FIRES, and the clamped group ==
// ========================================================================
//
// The 4-tile fixture above never reaches a publish boundary, so its lagged
// block is statically present but dynamically dead. This section is the
// same pipeline on a shape that does: 384x64x1024, Mt=12, steady loop
// m=0..10 (upper 11), live publish boundaries at m=3 and m=7, tail rows
// [8..12).
//
// The lagged publish fires ONCE in the loop, at iv=7: it publishes group
// [0..4) (rowStart = 7-4-3 = 0) behind wait_retired(4) -- one full batch
// after that group's own boundary, which is the whole point of the lag.
// Group [4..8) CLAMPS: its lagged position (iv=11) is past the loop, so it
// publishes at the tail-submit boundary instead -- after the tail submit
// (which hands the engine tiles [8..12)), before the exit drain, keeping
// its wait_retired(8): the vector thread chews rows [4..8) while the
// engine finishes the tail batch. The tail publish (rows [8..12)) follows
// the drain with NO wait: the drain is the all-groups proof. Rows
// published: [0..4) in-loop, [4..8) clamped, [8..12) tail -- every row
// exactly once.
// SPLIT-LABEL: func.func @deep_k12(
// The steady loop's own submit, then the lagged publish-if (the same
// structure as the 4-tile fixture's, firing here at iv=7).
// SPLIT: call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: %[[MPREV2:.*]] = arith.subi %{{.*}}, %{{.*}} : index
// SPLIT: %[[DONE2:.*]] = arith.addi %[[MPREV2]], %{{.*}} : index
// SPLIT: %{{.*}} = arith.remsi %[[DONE2]], %{{.*}} : index
// SPLIT: %[[LAG2:.*]] = arith.andi %{{.*}}, %{{.*}} : i1
// SPLIT: scf.if %[[LAG2]] {
// SPLIT: %[[TGT2:.*]] = arith.index_cast %[[DONE2]] : index to i32
// SPLIT: call @hexagon_runtime_hmx_role_wait_retired(%[[TGT2]]) : (i32) -> ()
// SPLIT: %[[ROW2:.*]] = arith.subi %[[MPREV2]], %{{.*}} : index
// SPLIT: call @hexagon_runtime_hmx_exec_publish({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: scf.yield
// The epilogue's tail submit (tiles [8..12), the peeled tile 11 included),
// then the CLAMPED group: wait_retired(8) -- K*G with K=2 live groups --
// and the publish of rows [4..8) (rowStart 4, rowCount 4).
// SPLIT: call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: %[[W8:.*]] = arith.constant 8 : i32
// SPLIT: call @hexagon_runtime_hmx_role_wait_retired(%[[W8]]) : (i32) -> ()
// SPLIT: %[[RS4:.*]] = arith.constant 4 : index
// SPLIT: %[[RS4I:.*]] = arith.index_cast %[[RS4]] : index to i32
// SPLIT: memref.store %[[RS4I]], %{{.*}}[%{{.*}}] : memref<6xi32>
// SPLIT: call @hexagon_runtime_hmx_exec_publish({{.*}}, {{.*}}) : (i32, i32) -> i32
// The exit drain, then the tail publish of rows [8..12) (rowStart 8) with
// NO wait between them, then the vector drain.
// SPLIT: call @hexagon_runtime_hmx_role_drain() : () -> ()
// SPLIT-NOT: call @hexagon_runtime_hmx_role_wait_retired
// SPLIT: %[[RS8:.*]] = arith.constant 8 : index
// SPLIT: %[[RS8I:.*]] = arith.index_cast %[[RS8]] : index to i32
// SPLIT: memref.store %[[RS8I]], %{{.*}}[%{{.*}}] : memref<6xi32>
// SPLIT: call @hexagon_runtime_hmx_exec_publish({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: call @hexagon_runtime_hmx_exec_drain() : () -> ()

module attributes {hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 128 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64}}, full = {k = 1024 : i64, m = 128 : i64, n = 64 : i64}, function = "deep_k", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 1024 : i64}, m = {kind = "static", value = 128 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 1024 : i64, m = 128 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 532480 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "deep_k", slot = 1 : i64}}, workspace_class = "runtime-internal"}], pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, schema = "hex.hmx.kernel_manifest/v2", unpack_sites = 1 : i64, weight_policies = []}} {
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

// -----

// The 12-tile shape (Mt=12): the lagged publish FIRES and the last live
// group CLAMPS; see the SPLIT12 block at the top of this file. The fixture
// is the 4-tile one with M=384 (12 m-tiles); everything else -- Kt=32,
// Nt=2, the hoistable read-out -- is identical.
module attributes {hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}, execution = {block_m = 384 : i64, blocking = "whole", bridge_counts = {count_semantics = "ir_sites", pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, unpack_sites = 1 : i64}}, full = {k = 1024 : i64, m = 384 : i64, n = 64 : i64}, function = "deep_k12", id = 0 : i64, layout = "row-major-inner-contiguous", logical = {k = {kind = "static", value = 1024 : i64}, m = {kind = "static", value = 384 : i64}, n = {kind = "static", value = 64 : i64}}, padded = {k = 1024 : i64, m = 384 : i64, n = 64 : i64}, plan = "full-hmx", reason = "selected-aligned", shape_state = "static", tail = {k = 0 : i64, m = 0 : i64, n = 0 : i64}, vtcm_accounting = "bridge-only", vtcm_before_bytes = 0 : i64, vtcm_bridge_peak_bytes = 532480 : i64, vtcm_budget_bytes = 8388608 : i64, weight_binding = {kind = "argument-slot", policy_ref = {function = "deep_k12", slot = 1 : i64}}, workspace_class = "runtime-internal"}], pack_act_sites = 1 : i64, pack_weight_sites = 1 : i64, schema = "hex.hmx.kernel_manifest/v2", unpack_sites = 1 : i64, weight_policies = []}} {
func.func @deep_k12(%a: memref<384x1024xf16>, %w: memref<1024x64xf16>, %out: memref<384x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %c384 = arith.constant 384 : index
  %c1024 = arith.constant 1024 : index
  %ca = memref.alloc() : memref<12x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c384 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<384x1024xf16>) outs(%ca : memref<12x32x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c1024 step %c2 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  }
  %ar = memref.alloc() : memref<12x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<12x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<12x2x16x32x2xf16, 1>) {hmx.decision_id = 0 : i64}
  scf.for %i = %c0 to %c12 step %c1 {
    %r = hmx.unpack_acc ins(%ar, %i, %c0 : memref<12x2x16x32x2xf16, 1>) outs(%out : memref<384x64xf16>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<384x64xf16>
  }
  memref.dealloc %ca : memref<12x32x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x32x16x32x2xf16, 1>
  memref.dealloc %ar : memref<12x2x16x32x2xf16, 1>
  return
}
}
