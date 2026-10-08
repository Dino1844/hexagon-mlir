//===- hmx-partition-serial-fold.mlir - the serial bridge folds -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause. For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The unstaged serial path folds the whole-array activation bridge into the
// m-tile loop (S2.5, 2026-10-08): one pack per m-tile, from a per-tile view of
// the bridge's row-major source into a one-row crouton scratch, the mmas
// reading the scratch, and the whole array retired with the bridge. No ring,
// no pipeliner, no hand-written peel -- the serial source loop is the staged
// form's compute half with the source view in place of the awaited slot.
//
// What the fold buys is a *thread-role* fact, so the file checks both halves:
// the IR form hmx-partition emits, and the verdict thread-role-partition
// derives from it. A pack held directly by the m-tile loop whose body carries
// the engine nest is a per-tile producer stream -- while tile i is in the
// matrix engine, another thread can build tile i+1 -- which is
// `role-split-ok`, the verdict the whole-array form could never reach (its
// pack loop runs to completion before the first mma, `role-split-nopack`).
// The decline cases pin the two boundaries that keep the fold honest: a
// matmul an outer loop re-executes keeps its bridge (the pack would be
// re-paid per iteration), and the no-bridge serial shape is not declined at
// all -- there is nothing to fold and nothing to say.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition))' | FileCheck %s --check-prefix=FOLD
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-partition, thread-role-partition))' | FileCheck %s --check-prefix=ROLE
//===----------------------------------------------------------------------===//

// The folded serial source loop. Mt=2, Kt=4 (K=128): a `memref<64x128xf16>`
// source, a `2x4x16x32x2` activation array the fold retires, a one-row
// `1x4x16x32x2` scratch in its place.
// FOLD-LABEL: func.func @shallow_kt_folded
// FOLD: hmx.bias_init
// The whole activation array is gone; the tile loop fills a one-row scratch.
// FOLD: %[[SCRATCH:.*]] = memref.alloc() : memref<1x4x16x32x2xf16, 1>
// FOLD: scf.for %[[M:.*]] = {{.*}} to
// pack(m) at the top of the iteration: the source view at element row m*32,
// then the ranged pack covering the whole K run.
// FOLD: %[[ROW:.*]] = arith.muli %[[M]], {{.*}} : index
// FOLD: %[[TILE:.*]] = memref.subview %arg0[%[[ROW]], 0] [32, 128] [1, 1] : memref<64x128xf16> to memref<32x128xf16, strided<[128, 1], offset: ?>>
// FOLD: hmx.pack_act ins(%[[TILE]], {{.*}}, {{.*}} : memref<32x128xf16, strided<[128, 1], offset: ?>>) outs(%[[SCRATCH]] : memref<1x4x16x32x2xf16, 1>) {count = 4 : i64}
// The engine nest one loop deeper: the mma reads the scratch at crouton row 0
// and `acc_read` still writes output tile row m.
// FOLD: scf.for %[[N:.*]] = {{.*}} to
// FOLD: hmx.acc_clear
// FOLD: hmx.mma %[[SCRATCH]], {{.*}}, {{.*}}, %[[N]], {{.*}} {n_croutons = 4 : i32}
// FOLD: hmx.acc_read {{.*}}, {{.*}}, %[[M]], %[[N]] {bias_set = 0 : i32}
// FOLD: memref.dealloc %[[SCRATCH]]
// FOLD-NOT: hmx.matmul
// FOLD-NOT: hmx.stage
// The same shape through the thread-role classifier: the pack rides the
// m-tile loop's iterations alongside the engine nest, so the kernel is a
// per-tile producer stream.
// ROLE: module attributes {{.*}}topology = "role-split-ok"
// ROLE-LABEL: func.func @shallow_kt_folded
func.func @shallow_kt_folded(%a: memref<64x128xf16>, %w: memref<128x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %ca = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  scf.for %i = %c0 to %c8 step %c1 {
    %r = arith.divui %i, %c4 : index
    %cc = arith.remui %i, %c4 : index
    hmx.pack_act ins(%a, %r, %cc : memref<64x128xf16>) outs(%ca : memref<2x4x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  scf.for %i = %c0 to %c8 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<128x64xf16>) outs(%cw : memref<2x4x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>)
             outs(%ar : memref<2x2x16x32x2xf16, 1>)
  return
}

// -----

// A matmul an outer loop re-executes keeps its whole-array bridge: the fold
// would re-pay the pack per outer iteration, so the bridge -- which the outer
// loop amortizes -- stays in front of the tile loop. This is flash artist's
// per-chunk shape, and it is the boundary between a producer stream and a
// producer that has already run to completion inside the iteration. (`auto`
// declines staging on Kt=4 with its own floor remark before the serial path
// runs; the fold's decline is silent here because the plain serial form is
// today's shape, not a fallback from something better.)
// FOLD-LABEL: func.func @outer_loop_keeps_bridge
// FOLD: hmx.bias_init
// The weight bridge loop runs first, as in the fixture ...
// FOLD: hmx.pack_weight
// ... then the outer loop, and inside it the whole activation array survives:
// FOLD: scf.for {{.*}} {
// FOLD: %[[ACT:.*]] = memref.alloc() : memref<2x4x16x32x2xf16, 1>
// FOLD: scf.for {{.*}} {
// FOLD: hmx.pack_act ins(%arg0, {{.*}} : memref<64x128xf16>) outs(%[[ACT]] :
// ... and the mma reads it, not a scratch.
// FOLD: hmx.mma %[[ACT]], {{.*}} {n_croutons = 4 : i32}
// FOLD-NOT: memref.subview
// FOLD-NOT: hmx.matmul
// ROLE: module attributes {{.*}}topology = "role-split-nopack"
// ROLE-LABEL: func.func @outer_loop_keeps_bridge
func.func @outer_loop_keeps_bridge(%a: memref<64x128xf16>, %w: memref<128x64xf16>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %cw = memref.alloc() : memref<2x4x16x32x2xf16, 1>
  scf.for %i = %c0 to %c8 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<128x64xf16>) outs(%cw : memref<2x4x16x32x2xf16, 1>)
  }
  scf.for %chunk = %c0 to %c16 step %c1 {
    %ca = memref.alloc() : memref<2x4x16x32x2xf16, 1>
    scf.for %i = %c0 to %c8 step %c1 {
      %r = arith.divui %i, %c4 : index
      %cc = arith.remui %i, %c4 : index
      hmx.pack_act ins(%a, %r, %cc : memref<64x128xf16>) outs(%ca : memref<2x4x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%ca, %cw : memref<2x4x16x32x2xf16, 1>, memref<2x4x16x32x2xf16, 1>)
               outs(%ar : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %ca : memref<2x4x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
  }
  return
}
