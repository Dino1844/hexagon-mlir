//===- hmx-vector-readout-no-peel.mlir - the drain is after the last publish -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// WHY THIS FILE EXISTS
// WHY THIS FILE EXISTS ==========================
// hmx-vector-readout.mlir pins the rewrite on the shape every earlier measurement
// used: a 32-row AR array read out by a 0..31 loop plus a peeled epilogue. That
// shape always has an epilogue, so the tail handoff and the `drain` were always
// anchored at the epilogue's read-out and no ordering question arose.
//
// A kernel with NO peeled epilogue has a different anchor, and the anchor was
// wrong: `setInsertionPoint(loop)` inserts BEFORE the loop, so the tail handoff
// and the `drain` were both emitted ahead of the tile loop. The in-loop groups'
// `publish` calls then landed after the last `drain`, and the kernel returned
// with a batch in flight -- the vector thread writing the caller's destination
// after the caller had stopped waiting for it. Measured 2026-10-03 on the real
// 1024x512x64 matmul at `pipeline-depth=1` (no pipelining, so no peel, so this
// anchor) and on a grid-tiled kernel small enough that `Mt <= pipeline-depth`
// caps the ring at depth 1: 7 of 13 configurations drained before their last
// publish.
//
// Nothing about that failure looks like a failure at the IR level. The publish
// exists, the drain exists, the row arithmetic is right, and the option is
// spelled on. So the ordering is pinned here directly.
//
// THE COVERAGE ARGUMENT, ONCE, FOR BOTH SHAPES
// THE COVERAGE ARGUMENT, ONCE, FOR BOTH SHAPES =============
// The in-loop groups fire at `iv` with `(iv+1) % G == 0` and publish
// `[iv-(G-1), iv]`, so their union is `[0, G*floor(upper/G) - 1]`. The tail
// publishes `[G*floor(upper/G), Mt)`. Those two ranges are ADJACENT by
// construction and their union is every row exactly once, for any `Mt` and any
// `G`. `@tail_batch` (Mt=30, G=4: groups cover 0..27, tail names 28..29) and
// `@exact_multiple` (Mt=32, G=4: groups cover 0..31, tail is EMPTY and is not
// emitted) are the two ends of that, and the numbers in the checks below are
// those two answers written out.
//
// ONE FUNCTION PER MODULE, and why. `hmx-vector-readout` is registered per
// function (`addNestedPass<func::FuncOp>`, LinalgToLLVMPass.cpp:572) and the
// pass manager runs it on every `func.func` IN PARALLEL, while `rewrite`
// appends the outlined read-out and the runtime declarations to the MODULE
// body. Two matching matmuls in one module therefore race on the module's
// symbol table: measured 2026-10-03, two such functions failed ~40% of runs
// and three failed 100%, always with "redefinition of symbol named
// 'hexagon_runtime_hmx_exec_configure'". A single matmul per module -- which is
// every Triton kernel, and every earlier measurement here -- cannot hit it, so
// it is a separate defect from the one this file pins, and it is reported
// rather than fixed. `-split-input-file` keeps this file out of it so that the
// ordering assertion below cannot be flaky for a reason that has nothing to do
// with drain ordering.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' | FileCheck %s --check-prefix=TAIL
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' | FileCheck %s --check-prefix=EXACT
//===----------------------------------------------------------------------===//

// Mt = 30, so the loop runs 0..29 and there is no peel: the whole shape in one
// tile loop, which is what `pipeline-depth=1` produces for the shipped matmul.
// `upper == mTiles` is the case `matchReadout` admits without an epilogue.
//
// The NOTs below sit AFTER the LABEL and BEFORE the tile loop, so they say "the
// head of this function has no handoff and no drain in it". That is the anchor
// bug: a tail handoff or a drain emitted ahead of the tile loop lands here.
//
// The tail's own numbers are the load-bearing part. rowStart 28 and rowCount 2
// are read out of the descriptor stores, because a wrong rowStart is the failure
// this whole file is about: it reads the wrong accumulator rows and the kernel
// returns plausible garbage.
// TAIL-LABEL: func.func @tail_batch
// TAIL-NOT: hexagon_runtime_hmx_exec_publish
// TAIL-NOT: hexagon_runtime_hmx_exec_drain
// TAIL: scf.for %[[M:.*]] = {{.*}} to {{.*}} step
// The boundary test: `(m+1) % 4 == 0`, then rowStart `m-3`, count 4.
// TAIL: arith.addi %[[M]], {{.*}} : index
// TAIL: arith.remsi %{{.*}}, {{.*}} : index
// TAIL: scf.if
// TAIL: arith.subi %[[M]], {{.*}} : index
// TAIL: memref.store {{.*}}, %{{.*}}[%{{.*}}] : memref<6xi32>
// TAIL: call @hexagon_runtime_hmx_exec_publish
// Past the tile loop: the tail handoff. 28 = G*floor(30/4), 2 = 30-28, so the
// two rows the in-loop groups could not reach are named exactly.
// TAIL: %[[C28:.*]] = arith.constant 28 : index
// TAIL: arith.index_cast %[[C28]] : index to i32
// TAIL: memref.store {{.*}}, %{{.*}}[%{{.*}}] : memref<6xi32>
// TAIL: %[[C2:.*]] = arith.constant 2 : i32
// TAIL: memref.store %[[C2]], %{{.*}}[%{{.*}}] : memref<6xi32>
// TAIL: call @hexagon_runtime_hmx_exec_publish
// And the drain AFTER that handoff, not before it. A drain ahead of the last
// publish is the bug this file was added for.
// TAIL: call @hexagon_runtime_hmx_exec_drain()
// TAIL-NOT: call @hexagon_runtime_hmx_exec_publish
// TAIL: memref.dealloc
func.func @tail_batch(%bias: memref<256xi8, 1>, %out: memref<1024x512xf16, strided<[512, 1]>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c30 = arith.constant 30 : index
  %ar = memref.alloc() {alignment = 128 : i64} : memref<30x16x16x32x2xf16, 1>
  scf.for %m = %c0 to %c30 step %c1 {
    %rowZero = arith.constant 0 : index
    %rowMul = arith.muli %c1, %rowZero : index
    %row = arith.addi %m, %rowMul : index
    scf.for %n = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<30x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar, %row, %c0 : memref<30x16x16x32x2xf16, 1>) outs(%out : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
  }
  memref.dealloc %ar : memref<30x16x16x32x2xf16, 1>
  return
}

// -----

// The other end: Mt = 32 with G = 4, so the in-loop groups cover 0..31 exactly
// and the tail is EMPTY. The empty tail must not be emitted at all -- a
// zero-count publish is a handoff the executor has to reject at run time --
// while the drain is still required, because the in-loop groups need it. So
// after the loop, `drain` is the ONLY handoff-adjacent call and it is after it.
func.func @exact_multiple(%bias: memref<256xi8, 1>, %out: memref<1024x512xf16, strided<[512, 1]>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %ar = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
  scf.for %m = %c0 to %c32 step %c1 {
    %rowZero = arith.constant 0 : index
    %rowMul = arith.muli %c1, %rowZero : index
    %row = arith.addi %m, %rowMul : index
    scf.for %n = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar, %row, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
  }
  memref.dealloc %ar : memref<32x16x16x32x2xf16, 1>
  return
}

// EXACT-LABEL: func.func @exact_multiple
// Nothing in the head of this function either.
// EXACT-NOT: hexagon_runtime_hmx_exec_publish
// EXACT-NOT: hexagon_runtime_hmx_exec_drain
// EXACT: scf.for %[[M2:.*]] = {{.*}} to {{.*}} step
// EXACT: scf.if
// EXACT: call @hexagon_runtime_hmx_exec_publish
// Past the loop: no tail publish (the groups already named every row), and the
// drain. The NOT between them is what says "empty tail"; the drain after it is
// what says "drain still emitted when the tail was empty".
// EXACT-NOT: call @hexagon_runtime_hmx_exec_publish
// EXACT: call @hexagon_runtime_hmx_exec_drain()
// EXACT-NOT: call @hexagon_runtime_hmx_exec_drain
// EXACT: memref.dealloc