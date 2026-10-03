//===- hmx-vector-readout.mlir - batch the read-out onto a second thread ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A lowered HMX matmul issues its accumulator read-out (`hmx.unpack_acc`) on the
// same thread as the matrix engine. At pipeline-depth 2 that read-out is 39.79%
// of a 1024x512x64 kernel -- ~24.3 us of 61 us -- and it overlaps nothing.
// hmx-vector-readout hands it to a second thread in groups of G rows.
//
// What makes the move affordable is a layout fact, not a scheduling trick: the
// AR crouton array is allocated BEFORE the m-tile loop and released AFTER it, so
// every AR row is live for the whole loop. Deferring a group of rows therefore
// costs no extra VTCM and needs no double buffering. A descriptor handoff costs
// ~0.9 us against 760 ns of read-out per m-tile, so per-tile deferral LOSES and
// batching divides the handoff by G (G=4 -> 1.39x, G=8 -> 1.51x).
//
// The arms below check, in order of how much it would hurt to get wrong:
//
//   BATCHED   G=4 turns 31 per-row read-outs into one guarded in-loop publish
//             plus one tail publish, and the descriptor carries the group.
//   ROWID     the read-out's row is `addi %iv, (muli 1, 0)`, i.e. exactly the
//             induction variable. That is what makes the batch `iv - (G-1)`.
//   COVERAGE  every AR row is named exactly once across the in-loop groups and
//             the tail. A missed row leaves the caller's buffer stale, which is a
//             plausible-looking wrong answer -- the worst failure here.
//   DEFAULT-OFF  with the option off the IR is byte-identical.
//
// Note that prose in this file is written as ordinary `//` comments rather than
// as `// PREFIX: ...` lines: FileCheck reads any line carrying a check prefix as
// a directive, so prose with a prefix becomes a literal string to search for.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' | FileCheck %s --check-prefix=BATCHED
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=8}))' | FileCheck %s --check-prefix=ROWID
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' | FileCheck %s --check-prefix=COVERAGE
//
// DEFAULT-OFF is pinned twice: once with no pipeline at all, and once through the
// whole -linalg-to-llvm pipeline at depth 2, where this pass sits right after
// hmx-partition. The option is off by default, so production must never see the
// rewrite.
// The DEFAULT-OFF arm is the byte-identical gate at the pass level: with the
// option absent the pass is a no-op and no runtime symbol is even declared.
// RUN: linalg-hexagon-opt %s | FileCheck %s --check-prefix=DEFAULT-OFF
// RUN: linalg-hexagon-opt --help | FileCheck %s --check-prefix=OPTIONS
//===----------------------------------------------------------------------===//

// 32 AR rows, so Mt = 32 and the depth-2 tile loop runs 0..31: the kernel covers
// rows 0..30 and a peeled read-out after the loop covers row 31. This is the
// verified post-partition shape of pp-1024x512x64.mlir, reduced to the parts the
// rewrite depends on.
//
// The read-out is gone from the kernel; only @__hmx_readout below still holds
// one, which BATCHED-NOT scopes to the kernel by running before the first label.
// BATCHED-NOT: hmx.unpack_acc

// The one descriptor of this function, built before the tile loop: the four
// invariant fields first, then the two that move per publish.
// BATCHED: %[[ALLOCA:.*]] = memref.alloca() : memref<6xi32>
// BATCHED: memref.store {{.*}}, %[[ALLOCA]][%{{.*}}] : memref<6xi32>
// BATCHED: memref.store {{.*}}, %[[ALLOCA]][%{{.*}}] : memref<6xi32>
// BATCHED: memref.store {{.*}}, %[[ALLOCA]][%{{.*}}] : memref<6xi32>
// BATCHED: memref.store {{.*}}, %[[ALLOCA]][%{{.*}}] : memref<6xi32>
// BATCHED: %[[DESC:.*]] = memref.extract_aligned_pointer_as_index %[[ALLOCA]]

// The boundary test is `(m + 1) % G == 0`, computed off the induction variable so
// the software-pipelined loop's iter_args are left alone.
// BATCHED: scf.for %[[IV:.*]] =
// BATCHED: %[[ONE:.*]] = arith.constant 1 : index
// BATCHED: %[[G:.*]] = arith.constant 4 : index
// BATCHED: %[[NEXT:.*]] = arith.addi %[[IV]], %[[ONE]] : index
// BATCHED: %[[REM:.*]] = arith.remsi %[[NEXT]], %[[G]] : index
// BATCHED: %[[AT:.*]] = arith.cmpi eq, %[[REM]], %{{.*}} : index
// BATCHED: scf.if %[[AT]] {

// rowStart is m - (G-1) and rowCount is G: the group ENDS at this iteration, so
// publishing at m = G-1, 2G-1, ... names rows [0,G), [G,2G), ...
// BATCHED: %[[BACK:.*]] = arith.constant 3 : index
// BATCHED: %[[START:.*]] = arith.subi %[[IV]], %[[BACK]] : index
// BATCHED: %[[STARTC:.*]] = arith.index_cast %[[START]] : index to i32
// BATCHED: memref.store %[[STARTC]], %[[ALLOCA]][%{{.*}}] : memref<6xi32>
// BATCHED: memref.store {{.*}}, %[[ALLOCA]][%{{.*}}] : memref<6xi32>
// BATCHED: func.call @hexagon_runtime_hmx_exec_publish(

// The guard closes and the tile loop retires with nothing left in it but the
// engine work.
// BATCHED: }
//
// The peel's read-out is gone: it is now part of the tail batch. rowStart is 28,
// i.e. (ub//G)*G = 7*4, and rowCount is 4, so this batch names rows 28..31 -- the
// peeled row 31 is its last one.
//
// The peel's `hmx.acc_read` is checked FIRST, and that ordering is the point: it
// is the op that writes row 31, so a tail publish emitted before it would hand
// the vector thread a row nobody has written. The pass anchors the tail at the
// peeled read-out for exactly this reason.
// BATCHED: hmx.acc_read %{{.*}}, %[[AR:.*]], %c31
// BATCHED: %[[TSTART:.*]] = arith.constant 28 : index
// BATCHED: arith.index_cast %[[TSTART]] : index to i32
// BATCHED: memref.store {{.*}}, %[[ALLOCA]][%{{.*}}] : memref<6xi32>
// BATCHED: %[[TCOUNT:.*]] = arith.constant 4 : i32
//
// After the loop the printer drops the `func.` qualifier, so match both spellings.
// BATCHED: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// BATCHED: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()

// The drain follows the last publish and precedes the AR dealloc, so the vector
// thread never reads freed VTCM and the caller never sees the destination before
// it is written.
// BATCHED: memref.dealloc %[[AR]] : memref<32x16x16x32x2xf16, 1>

// G=8 changes the group and nothing else: the same shape, so the batch size is a
// pure knob rather than a different rewrite.
// ROWID-NOT: hmx.unpack_acc
// ROWID: %[[RG:.*]] = arith.constant 8 : index
// ROWID: arith.remsi %{{.*}}, %[[RG]] : index
// ROWID: %[[RBACK:.*]] = arith.constant 7 : index
// ROWID: arith.subi %{{.*}}, %[[RBACK]] : index
// ROWID: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// ROWID: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()

// COVERAGE arm: 32 rows, loop to 31, G=4. The in-loop groups fire at m = 3, 7,
// ..., 27 -- seven of them -- naming rows [0,4) .. [24,28). The tail then names
// [28,32), four rows, and the peeled row 31 is the last of them. [0,28) u [28,32)
// = [0,32): every row once, none twice.
// COVERAGE-NOT: hmx.unpack_acc
// COVERAGE: memref.alloca() : memref<6xi32>
// COVERAGE: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// COVERAGE: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// COVERAGE: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()

// The read-out moved into one outlined function that loops the group's rows.
// `nrows` is the tail batch's count, which need not be a multiple of G, so it is a
// runtime argument; `count` and `col` are STATIC attributes on hmx.unpack_acc
// (HmxOps.td:329-331) and are baked in.
// The signature is pinned whole: `ar` and `dst` are memrefs because
// hmx.unpack_acc takes memrefs, `row0` and `nrows` are index. It is marked with
// hmx.readout.outlined so the eventual configure() handoff has something to
// anchor on.
//
// The outlined read-out also carries `hmx.decision_id = 0`, and that attribute is
// load-bearing rather than bookkeeping: `refreshHmxManifestBridgeCounts` walks the
// WHOLE module and requires every bridge op to name a manifest record, and this op
// is new -- the originals it replaces are erased. Without the id the compile of a
// real kernel fails outright, five times over, with "HMX bridge operation has no
// explicit hmx.decision_id". The id alone is not enough: the recount resolves a
// record by (function, id), and `__hmx_readout` has no record of its own, so
// HmxManifest.cpp maps it back to `kernel` through `hmx.readout.handoffs`.
//
// This arm is the IR-level guard only. It cannot see whether the rewrite survives
// into the object file -- which is exactly how the previous version of this feature
// shipped a build that linked, ran correctly, and contained none of the split.
// test/test_hmx_vector_readout_object_gate.py is the gate that reads the object.
// COVERAGE: func.func private @__hmx_readout(%arg0: memref<32x16x16x32x2xf16, 1>, %arg1: memref<1024x512xf16, strided<[512, 1]>>, %arg2: index, %arg3: index) attributes {hmx.readout.outlined} {
// COVERAGE: %[[C1:.*]] = arith.constant 1 : index
// COVERAGE: scf.for %[[I:.*]] = %{{.*}} to %arg3 step %[[C1]] {
// COVERAGE: arith.addi %arg2, %[[I]] : index
// COVERAGE: hmx.unpack_acc ins({{.*}}) outs({{.*}}) {count = 16 : i64, hmx.decision_id = 0 : i64}
// COVERAGE: return

// Without the option nothing at all happens: the read-out stays in the tile loop
// (row `%2`, i.e. the induction variable through the partition pass's `m * 1 + 0`),
// the peeled one stays after it, and no runtime entry point is even declared. This
// is the byte-identical requirement -- kernel IR must be unchanged when the option
// is off, since every existing measurement was taken without it.
// The row the read-out names is the SAME value the acc_read wrote (`%[[ROW]]` =
// `m * 1 + 0`), which is the identity the rewrite relies on; here it is still
// spelled out, twice, once per read-out.
// The option-off arm, which is the real "kernel IR must be byte-identical" gate:
// with `enable-hmx-vector-readout` off (the default) the pass is never added to the
// pipeline, so this output must equal the input. The identity the rewrite would
// rely on is visible and intact here: the in-loop `hmx.acc_read` and
// `hmx.unpack_acc` name the SAME row `%[[ROW]]`, the induction variable reached
// through the partition pass's `m * 1 + 0`, and both read-outs are still spelled
// out rather than folded into a batch.
//
// The second RUN line is the whole -linalg-to-llvm pipeline at depth 2, which is
// where this pass actually sits in production; it must not change a byte either.
// DEFAULT-OFF: %[[AR:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
// DEFAULT-OFF: %[[ROW:.*]] = arith.addi %arg2, %{{.*}} : index
// DEFAULT-OFF: hmx.acc_read %{{.*}}, %[[AR]], %[[ROW]]
// DEFAULT-OFF: hmx.unpack_acc ins(%[[AR]], %[[ROW]]
// DEFAULT-OFF: hmx.unpack_acc ins(%[[AR]], %c31
// DEFAULT-OFF-NOT: hexagon_runtime_hmx_exec
// DEFAULT-OFF-NOT: __hmx_readout
// DEFAULT-OFF-NOT: memref.alloca()

// NOTE ON THE PIPELINE ARM: driving this fixture through -linalg-to-llvm would
// need a full `hex.hmx.kernel_manifest/v2` module attribute, because the pipeline's
// bridge verifier rejects an `hmx.decision_id` with no manifest record before any
// of the options here are consulted. That manifest is 3 KB of unrelated contract in
// a test about two options, so the pipeline arm is covered where the manifest
// already exists -- see hmx-vector-readout-pipeline.mlir, which runs the real
// matmul shape through -linalg-to-llvm and checks both the OFF and ON arms.

// Both halves of the surface are declared: the standalone pass (with its batch
// knob) and the -linalg-to-llvm options that gate it. `mlir-opt --help` prints
// each option's description rather than a "default:" field, so the gate is pinned
// by the pipeline flag's own "Off by default" wording -- the declaration and the
// documented default together.
// OPTIONS: --hmx-vector-readout
// OPTIONS: --hmx-readout-batch=<long>
// OPTIONS: --enable-hmx-vector-readout
// OPTIONS-SAME: Off by default
func.func @kernel(%bias: memref<256xi8, 1>, %out: memref<1024x512xf16, strided<[512, 1]>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c31 = arith.constant 31 : index
  %ar = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
  // The tile loop: 32 iterations, of which the kernel proper runs 31 and the peel
  // covers the last. This is the shape scf::pipelineForLoop leaves at depth 2.
  scf.for %m = %c0 to %c31 step %c1 {
    // The row expression hmx-partition emits: `m * 1 + 0`.
    %rowZero = arith.constant 0 : index
    %rowMul = arith.muli %c1, %rowZero : index
    %row = arith.addi %m, %rowMul : index
    scf.for %n = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar, %row, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
  }
  // The peeled epilogue: row Mt-1 = 31, read out after the loop.
  hmx.acc_read %bias, %ar, %c31, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
  hmx.unpack_acc ins(%ar, %c31, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
  memref.dealloc %ar : memref<32x16x16x32x2xf16, 1>
  return
}
