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
//   SPLIT     the two-role form with readout coexistence: the steady loop
//             keeps the DMA ring and pack (into the rotating rows), the
//             engine nest is GONE, and the in-loop order inside one
//             iteration is submit(batch) -> wait_retired(m+1) ->
//             publish(group) -- the protocol chain the R2 mechanism exists
//             to establish, with the wait as the per-group completion proof
//             that replaces the first form's exit-drain proof. The peeled
//             epilogue: await, pack into rows[Mt-1], the tail submit (which
//             carries the peeled tile as its last descriptor -- the
//             "peel tail into the section" property), the tail wait, then
//             the read-out split's tail publish. One work function covers
//             every tile: its per-tile loop derives the row from the
//             descriptor's rowStart, so the peeled tile is just the last
//             descriptor.
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
//             submit batch (4) would make the producer wait for tiles the
//             ring has not been handed -- a deadlock, so the split declines
//             loudly and the kernel keeps the single-thread pipeline with
//             its read-out intact.
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

// THE PROTOCOL CHAIN, in emission order inside one steady iteration: the
// batched submit (with its short-return recovery), then the granular
// barrier, then the publish. The wait is the R2 mechanism's whole point --
// "publish only after the engine section of every tile the publish names has
// RETIRED", the per-group upgrade of the first form's exit-drain proof.
// SPLIT: call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: call @hexagon_runtime_hmx_role_wait_retired({{.*}}) : (i32) -> ()
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
// drain: no section writes into caller memory after it returns), the tail
// wait (target = the tile count: every tile retired -- trivially satisfied
// behind the drain, emitted anyway because the publish's proof is the
// wait, not the drain), the read-out split's tail publish, and the vector
// drain (the read-out's own barrier, before the AR release).
// SPLIT: hmx.await
// SPLIT: memref.subview {{.*}}[{{.*}}] [1, 32, 16, 32, 2]
// SPLIT: hmx.pack_act
// SPLIT: call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32
// SPLIT: call @hexagon_runtime_hmx_role_drain() : () -> ()
// SPLIT: call @hexagon_runtime_hmx_role_wait_retired({{.*}}) : (i32) -> ()
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
// A read-out batch of 2 against a submit batch of 4: the publish at m=1
// would wait for tiles 0..1 whose descriptors are only submitted at m=3 --
// a deadlock, so the split declines with the reason named, and the kernel
// keeps the single-thread pipelined form WITH its read-out (the publish
// machinery is still there; the role machinery is not).
// DECLINE: func.func @deep_k(
// DECLINE: call @hexagon_runtime_hmx_exec_publish
// DECLINE-NOT: call @hexagon_runtime_hmx_role_submit
// DECLINE-NOT: __hmx_role_section
// DECLINE-MSG: remark: thread-role split not applied: the read-out publishes every 2 tiles but the role submit batches every 4; a publish boundary the submit does not cover would wait for tiles the ring never received (deadlock). The kernel keeps the single-thread pipeline with its read-out intact

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
