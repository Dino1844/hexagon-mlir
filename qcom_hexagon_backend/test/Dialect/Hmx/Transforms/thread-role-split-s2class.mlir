//===- thread-role-split-s2class.mlir - the S3 first shape, split ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The S3 first shape (ROADMAP1001 section 5, S3 row): an S2-class matmul,
// 256x64x2048 -- Mt=8, Nt=2, Kt=64 -- driven through the serial staged source
// ring (hmx-partition at pipeline-depth 1, which is what the production
// pipeline requests while the thread-role gate is on; see
// LinalgToLLVMPass.cpp) and then through thread-role-partition with the split
// enabled. This file pins the two-role form structurally: correctness cannot
// be checked without a device, so what is pinned is that every piece of the
// mechanism is present, in the right order, with the right names.
//
// The checks are ordered the way the module prints, because FileCheck matches
// in order: module attributes first (the manifest verdict and the handoff
// record both live there), then the buffer-table globals, then the producer
// kernel, then the outlined work function.
//
// The three arms:
//
//   SPLIT   the func-level emission: the outlined work function (with the
//           `hex.thread_role` region marker and the engine nest reading the
//           tile's own row), the producer loop (pack into the rotating rows,
//           descriptor fill, the batched submit with its short-return
//           recovery), the exit drain, the buffer-table stores, and the
//           handoff record the wiring consumes.
//   WIRE    the same shape through convert-func-to-llvm and hmx-to-llvm: the
//           exported entry point `s2_class__hmx_section` unpacking the
//           descriptor, the exported depth object `s2_class__hmx_role_depth`
//           holding the tile count, the `hexagon_hmx` passthrough on the work
//           function, and -- load-bearing for the lock migration -- NO
//           ensure/unlock pair anywhere (the bound thread's lifetime lock
//           replaces the per-kernel pairing).
//   OFF     the same shape with thread-role-partition absent: no submit, no
//           drain, no work function, no handoff record. This is the
//           byte-laziness arm -- the OFF pipeline never sees the split.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=1}, thread-role-partition))' | FileCheck %s --check-prefix=SPLIT
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=1}, thread-role-partition), convert-scf-to-cf, convert-func-to-llvm, hmx-to-llvm)' | FileCheck %s --check-prefix=WIRE
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=1}))' | FileCheck %s --check-prefix=OFF
//===----------------------------------------------------------------------===//

// The manifest verdict the split was gated on, unchanged by the emission:
// `role-split-ok` is the verdict; the outlined function's existence is what
// says the split happened (the v3 manifest is frozen -- no new fields). The
// handoff record sits beside it: the engine, the work function, the depth
// (the tile count -- the ring is as deep as the tile loop), and the four
// buffer types.
// SPLIT: module attributes {{.*}}topology = "role-split-ok"
// SPLIT-SAME: hmx.role.handoffs = [{ar = memref<8x2x16x32x2xf16, 1>, bias = memref<256xi8, 1>, depth = 8 : i64, engine = "s2_class", rows = memref<8x64x16x32x2xf16, 1>, work = "__hmx_role_section", wt = memref<2x64x16x32x2xf16, 1>}]

// The per-launch buffer table: four internal i32 words, one per buffer the
// engine section closes over. The producer's stores are LLVM ops in a func
// body on purpose -- the entry point addresses these globals after
// convert-func-to-llvm, and pre-lowered LLVM ops survive that conversion.
// SPLIT: llvm.mlir.global internal @__hmx_role_ar
// SPLIT: llvm.mlir.global internal @__hmx_role_bias
// SPLIT: llvm.mlir.global internal @__hmx_role_wt
// SPLIT: llvm.mlir.global internal @__hmx_role_rows

// The producer kernel.
// SPLIT-LABEL: func.func @s2_class(

// The rotating scratch replaces the one-row scratch: [Mt=8, Kt=64] croutons,
// one row per tile, so the producer never overwrites a row the consumer has
// not retired -- that single-row assumption is exactly what the split removes.
// The submit buffer is a 24-i32 stack object (batch 4 x the descriptor's six
// i32 words), alive for the whole launch because the exit drain runs before
// the function returns.
// SPLIT: %[[ROWS:.*]] = memref.alloc() : memref<8x64x16x32x2xf16, 1>
// SPLIT: %[[DESC:.*]] = memref.alloca() : memref<24xi32>

// One store per buffer word, addresses taken the way every address crosses
// into the runtime on this target (extract_aligned_pointer_as_index -> i32).
// SPLIT: llvm.mlir.addressof @__hmx_role_rows
// SPLIT: llvm.store {{.*}}, {{.*}} : i32, !llvm.ptr
// SPLIT: llvm.mlir.addressof @__hmx_role_wt
// SPLIT: llvm.mlir.addressof @__hmx_role_bias
// SPLIT: llvm.mlir.addressof @__hmx_role_ar

// The producer loop keeps the staged arm's DMA (stage/await stay
// producer-side: the transfer is engine-independent) and the pack, now into
// ROW m of the rotating rows -- the pack's row operand is the tile index.
// SPLIT: scf.for %[[M:.*]] = {{.*}} to {{.*}} step {{.*}} {
// SPLIT: hmx.stage
// SPLIT: hmx.await
// SPLIT: hmx.pack_act ins({{.*}}, %[[M]], {{.*}} : memref<32x2048xf16, 1>) outs(%[[ROWS]] : memref<8x64x16x32x2xf16, 1>)

// The engine nest is GONE from the producer -- no acc_clear, mma or acc_read
// between the pack and the submit. What replaces it is the descriptor fill
// (slot = rowStart = m, rowCount = 1, the pad and event words zeroed) and
// the batched submit every 4th iteration.
// SPLIT-NOT: hmx.acc_read
// SPLIT-NOT: hmx.mma
// SPLIT: func.call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32

// The short-return recovery: the ABI makes a dropped group the caller's
// problem (a dropped group is an unwritten output -- a wrong answer, not a
// slow one), so the return is compared, and a short return drains (the one
// barrier the frozen executor offers) and resubmits the tail by address.
// SPLIT: call @hexagon_runtime_hmx_role_drain() : () -> ()
// SPLIT: memref.subview %[[DESC]][{{.*}}] [{{.*}}] [{{.*}}] : memref<24xi32> to memref<?xi32
// SPLIT: func.call @hexagon_runtime_hmx_role_submit({{.*}}, {{.*}}) : (i32, i32) -> i32

// After the loop: no tail batch (Mt=8, batch=4 -- the boundary fires twice,
// the tail is empty), then the exit drain before the rows are released:
// `ar` is the caller's memory, and the consumer owns it until retired.
// SPLIT: call @hexagon_runtime_hmx_role_drain() : () -> ()
// SPLIT: memref.dealloc %[[ROWS]]

// The work function: private, region-marked, buffers as explicit parameters
// (the read-out precedent), then the bias init (engine work -- it moved with
// the section; the producer must never issue an engine instruction) and the
// per-tile loop over the descriptor's rows with the engine nest reading row
// m of the rotating rows.
// SPLIT: func.func private @__hmx_role_section({{.*}}: memref<8x64x16x32x2xf16, 1>, {{.*}}) attributes {hex.thread_role = "hmx"}
// SPLIT: hmx.bias_init
// SPLIT: %[[MI:.*]] = arith.addi {{.*}}, {{.*}} : index
// SPLIT: scf.for %[[N:.*]] = {{.*}} to {{.*}} step {{.*}} {
// SPLIT: hmx.acc_clear
// SPLIT: hmx.mma {{.*}}, %[[MI]], %[[N]]
// SPLIT: hmx.acc_read {{.*}}, %[[MI]], %[[N]] {bias_set = 0 : i32}

// ---- The wiring, at the LLVM level -----------------------------------------
//
// The module prints in this order: the producer kernel, the work function,
// the entry point, the depth object -- and the checks below follow it.

// The producer's submit and drain, through the conversion.
// WIRE: llvm.func @s2_class({{.*}}) {
// WIRE: llvm.call @hexagon_runtime_hmx_role_submit({{.*}}) : (i32, i32) -> i32
// WIRE: llvm.call @hexagon_runtime_hmx_role_drain

// The work function kept the thread contract: the passthrough attribute the
// LLVM IR translation turns into the "hexagon_hmx" fn attribute upstream
// PR #222340's TTI hooks read (the S0 backport), and which
// CollapseAddressSpace copies when it rebuilds the function. (The
// `hex.thread_role` marker rides along in the attribute dictionary; the
// passthrough is the load-bearing half.)
// WIRE: llvm.func @__hmx_role_section({{.*}}) attributes {{.*}}passthrough = ["hexagon_hmx"]

// The exported entry point, named from the kernel entry plus the shared
// suffix (HmxRoleChannel.h -- the same header the runtime probe includes):
// PUBLIC, unlike the read-out entry, because the launch-side probe dlsym's
// it in the loaded module. Its signature IS the ABI's. It reads the
// per-launch buffer table (the frozen 24-byte descriptor has no address
// words, so the buffers travel beside the ring) and calls the work function.
// WIRE: llvm.func @s2_class__hmx_section({{.*}}: !llvm.ptr, {{.*}}: i32)
// WIRE: llvm.call @__hmx_role_section(

// The exported depth object: external linkage WITH an initializer, so it is
// a defined, exported i32 the probe reads to bind with -- the ring depth the
// producer derived from the tile-ring geometry (the tile count, 8).
// WIRE: llvm.mlir.global external constant @s2_class__hmx_role_depth(8 : i32)

// THE LOCK MIGRATION'S FIRST HALF, MADE VISIBLE: no function in the module
// gets an ensure/unlock pair. The work function issues the engine leaves and
// runs on the bound thread, whose lifetime lock (taken once at thread start)
// replaces the pairing; the kernel no longer issues any engine leaf at all.
// An unlock at the work function's exit would release the lifetime lock
// under the executor's feet -- the exact bug the exemption exists to
// prevent, and the read-out era's per-kernel pairing is what it retires.
// WIRE-NOT: hexagon_runtime_hmx_ensure
// WIRE-NOT: hexagon_runtime_hmx_unlock

// The consumed record: the wiring erased it, so a second consumer cannot
// shadow the entry point.
// WIRE-NOT: hmx.role.handoffs

// ---- The OFF arm -----------------------------------------------------------

// No split, no channel, no work function: the serial staged ring exactly as
// hmx-partition emitted it.
// OFF-NOT: hexagon_runtime_hmx_role
// OFF-NOT: __hmx_role_section
// OFF-NOT: hmx.role.handoffs
// OFF-NOT: __hmx_role_rows
// OFF-LABEL: func.func @s2_class(
func.func @s2_class(%a: memref<256x2048xf16>, %w: memref<2048x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %ca = memref.alloc() : memref<8x64x16x32x2xf16, 1>
  scf.for %i = %c0 to %c512 step %c1 {
    %r = arith.divui %i, %c64 : index
    %cc = arith.remui %i, %c64 : index
    hmx.pack_act ins(%a, %r, %cc : memref<256x2048xf16>) outs(%ca : memref<8x64x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x64x16x32x2xf16, 1>
  scf.for %i = %c0 to %c128 step %c1 {
    %r = arith.divui %i, %c64 : index
    %cc = arith.remui %i, %c64 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<2048x64xf16>) outs(%cw : memref<2x64x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<8x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<8x64x16x32x2xf16, 1>, memref<2x64x16x32x2xf16, 1>)
             outs(%ar : memref<8x2x16x32x2xf16, 1>)
  return
}
