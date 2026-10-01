//===- hmx-partition-crontons-per-mma.mlir - the K batch is an option ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// `croutons-per-mma` is how many K croutons one `hmx.mma` walks -- the engine's
// repeat count. Its default is the hardware maximum (32), so this file's
// default arm is the same code `hmx-partition-deep-crontons.mlir` already pins,
// and the option exists for one reason: to make "batch fewer croutons per
// instruction" expressible, so the batching can be A/B'd inside one build
// instead of by editing a constant and rebuilding (which invalidates the device
// anchor, docs/hmx/ncroutons-k-fusion-2026-10-01.md §3.5.3).
//
// Three things have to be true for that to be a measurement rather than a knob,
// and each is a run line below:
//
//   1. The default is unchanged. Proved by *diffing whole outputs* rather than
//      by re-writing the same CHECK pattern three times: "not passed", "passed 0"
//      and "passed 32" must be one request, so their outputs are compared as
//      files. Three identical patterns would only prove the patterns were
//      written alike.
//   2. `= 1` reaches the emitter, and it is *distinguishable* from the default on
//      the same IR: 32 mmas in a software loop against 1 mma and no loop. If the
//      option were plumbed but ignored, `@kt32` under ONE would still show the
//      MAX body and this file would go red -- that is the resolving power the
//      whole change is for.
//   3. The tail honours the option too. `@kt33` at `= 2` walks 16 mmas of 2
//      croutons and then a tail of 1, so a tail left hardcoded at 32 (the shape
//      the default produces) is a failure here rather than a surprise on device.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' > %t.default
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{croutons-per-mma=0}))' > %t.zero
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{croutons-per-mma=32}))' > %t.max
// RUN: diff %t.default %t.zero
// RUN: diff %t.default %t.max
// RUN: FileCheck %s --check-prefix=MAX < %t.default
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{croutons-per-mma=1}))' | FileCheck %s --check-prefix=ONE
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{croutons-per-mma=2}))' | FileCheck %s --check-prefix=TWO
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=1}))' | FileCheck %s --check-prefix=STAGEDMAX
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=1 croutons-per-mma=1}))' | FileCheck %s --check-prefix=STAGEDONE
//===----------------------------------------------------------------------===//

// ---- Kt = 32 under the default: one instruction, no K loop. The `CHECK-NOT:
// scf.for` between the clear and the read is the whole difference from the ONE
// arm below, so it is what makes that arm able to fail.
// MAX-LABEL: func.func @kt32
// MAX: scf.for %[[M:.*]] =
// MAX: scf.for %[[N:.*]] =
// MAX: hmx.acc_clear
// MAX-NOT: scf.for
// MAX: %[[K:.*]] = arith.constant 0 : index
// MAX-NEXT: hmx.mma %{{.*}}, %{{.*}}, %[[M]], %[[N]], %[[K]] {n_croutons = 32 : i32}
// MAX: hmx.acc_read

// ---- The same IR at `croutons-per-mma=1`: the pre-batching form. 32 mmas in a
// 32-trip loop that steps k by 1, and no tail because 32 % 1 == 0. The bound is
// `batches * batch` = 32, which happens to equal Kt here; @kt33 below is where
// the bound and Kt part company.
// ONE-LABEL: func.func @kt32
// ONE: scf.for %[[M1:.*]] =
// ONE: scf.for %[[N1:.*]] =
// ONE: hmx.acc_clear
// ONE: %[[Z1:.*]] = arith.constant 0 : index
// ONE: %[[S1:.*]] = arith.constant 1 : index
// ONE: %[[E1:.*]] = arith.constant 32 : index
// ONE: scf.for %[[K1:.*]] = %[[Z1]] to %[[E1]] step %[[S1]] {
// ONE: hmx.mma %{{.*}}, %{{.*}}, %[[M1]], %[[N1]], %[[K1]] {n_croutons = 1 : i32}
// ONE: }
// ONE: hmx.acc_read
func.func @kt32() {
  %a = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}

// ---- Kt = 33 at `croutons-per-mma=2`: 16 mmas of 2 croutons walking k by 2 to
// a bound of 32 -- NOT 33, because a bound of 34 would run the loop a second time
// and read past the end of the crouton array -- then a tail of 1 at k = 32. The
// tail's count and its k both come from the batch, so a tail still written
// against 32 shows up here as `n_croutons = 1` in the wrong place.
// TWO-LABEL: func.func @kt33
// TWO: scf.for %[[M2:.*]] =
// TWO: scf.for %[[N2:.*]] =
// TWO: hmx.acc_clear
// TWO: %[[Z2:.*]] = arith.constant 0 : index
// TWO: %[[S2:.*]] = arith.constant 2 : index
// TWO: %[[E2:.*]] = arith.constant 32 : index
// TWO: scf.for %[[K2:.*]] = %[[Z2]] to %[[E2]] step %[[S2]] {
// TWO: hmx.mma %{{.*}}, %{{.*}}, %[[M2]], %[[N2]], %[[K2]] {n_croutons = 2 : i32}
// TWO: }
// TWO: %[[TAIL2:.*]] = arith.constant 32 : index
// TWO-NEXT: hmx.mma %{{.*}}, %{{.*}}, %[[M2]], %[[N2]], %[[TAIL2]] {n_croutons = 1 : i32}
// TWO: hmx.acc_read
func.func @kt33() {
  %a = memref.alloc() : memref<2x33x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x33x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x33x16x32x2xf16, 1>, memref<2x33x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}

// ---- The staged arm reads the same option, and it is a separate emitter:
// `emitTileCompute` and `emitSerialTileLoop` each call `emitMmaKLoop`, so an
// option wired into only one of them would pass every check above and leave the
// other batched at 32. The activation is 4x32 croutons, so Kt = 32 and the mma
// walks a one-row scratch. STAGEDMAX is the unchanged default (pinned so the
// fixture itself is known good and the delta below is the option and nothing
// else); STAGEDONE is the same nest with the 32-trip loop inside it.
// STAGEDMAX-LABEL: func.func @staged
// STAGEDMAX: %[[SCRATCH0:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
// STAGEDMAX: hmx.stage
// STAGEDMAX: hmx.await
// STAGEDMAX: hmx.pack_act
// STAGEDMAX: scf.for %[[N0:.*]] = {{.*}} {
// STAGEDMAX: hmx.acc_clear
// STAGEDMAX-NOT: scf.for
// STAGEDMAX: %[[K0:.*]] = arith.constant 0 : index
// STAGEDMAX-NEXT: hmx.mma %[[SCRATCH0]], {{.*}}, {{.*}}, %[[N0]], %[[K0]] {n_croutons = 32 : i32}
// STAGEDMAX: hmx.acc_read
// STAGEDMAX-NOT: hmx.matmul

// STAGEDONE-LABEL: func.func @staged
// STAGEDONE: %[[SCRATCH:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
// STAGEDONE: hmx.stage
// STAGEDONE: hmx.await
// STAGEDONE: hmx.pack_act
// STAGEDONE: scf.for %[[N:.*]] = {{.*}} {
// STAGEDONE: hmx.acc_clear
// STAGEDONE: %[[Z:.*]] = arith.constant 0 : index
// STAGEDONE: %[[S:.*]] = arith.constant 1 : index
// STAGEDONE: %[[E:.*]] = arith.constant 32 : index
// STAGEDONE: scf.for %[[K:.*]] = %[[Z]] to %[[E]] step %[[S]] {
// STAGEDONE: hmx.mma %[[SCRATCH]], {{.*}}, {{.*}}, %[[N]], %[[K]] {n_croutons = 1 : i32}
// STAGEDONE: }
// STAGEDONE: hmx.acc_read
// STAGEDONE-NOT: hmx.matmul
func.func @staged(%a: memref<128x1024xf16>, %w: memref<1024x64xf16>) {
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
    hmx.pack_act ins(%a, %r, %cc : memref<128x1024xf16>) outs(%ca : memref<4x32x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  hmx.matmul ins(%ca, %cw : memref<4x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<4x2x16x32x2xf16, 1>)
  memref.dealloc %ca : memref<4x32x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x32x16x32x2xf16, 1>
  memref.dealloc %ar : memref<4x2x16x32x2xf16, 1>
  return
}
