//===- hmx-partition-nloop.mlir - the kernel's N loop is not the bridge ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The matmul whose ONLY enclosing loop is the kernel's own N-blocking loop, at
// the pass level: one `scf.for` holding a bare activation pack, the weight
// bridge, the matmul, the read-out and the N tile's output write.
//
// Why this test exists. `matmul-to-hmx` wraps its pack in a loop over the
// bridge's outer tiles, but canonicalization folds the single-tile case away
// (`Mt == 1` -- a 32-row block), leaving a bare `hmx.pack_act`. What encloses
// that pack is then the KERNEL's loop, and `findActivationBridge` used to claim
// it as the bridge's own pack loop. `emitStageLoop` builds its tile loop
// *inside* wherever the matmul sits and afterwards erases `bridge->loop` to
// retire the bridge -- which erased the N loop, the tile loop it had just
// built, and the output write with it, while still recording
// `selected = "staged"`. Measured host-side on the
// `MM=1, NN=8192, KK=8192, (BM,BN,BK)=(32,128,8192)` kernel, 2026-10-09. The
// same claim is not only silent: on this fixture, where the activation array's
// deallocation sits INSIDE the loop the claim erases, the original code
// double-freed it (`free(): double free detected in tcache 2`). Production IR
// gets the silent form because buffer-loop-hoisting lifts that deallocation out
// of the loop first, which is why the defect showed up as an empty span there.
//
// Both arms must keep the kernel's loop: the staged arm retires the bridge pack
// by pack instead of by erasing the loop it sits in, and the unstaged arm
// (`pipeline-depth=3`) keeps the bridge whole in front of the plain tile loop.
// The serial arm already refused this ownership claim -- `findPackBridge`
// rejects a loop whose body is not packs and index arithmetic -- which is the
// rule `isBridgePackLoop` now brings to the staged arm's finder.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{pipeline-depth=3}))' | FileCheck %s --check-prefix=SERIAL
//===----------------------------------------------------------------------===//

// The staged arm: the kernel's loop survives, and everything in it that is not
// the bridge survives with it.
// CHECK-LABEL: func.func @nloop_only
// The conversion state is kernel-level setup, created once before the loop.
// CHECK: %[[BIAS:.*]] = memref.alloc() {alignment = 256 : i64} : memref<256xi8, 1>
// CHECK: hmx.bias_init %[[BIAS]]
// CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// The whole-array bridge allocation is GONE -- retired pack by pack -- while
// the loop it sat in is not. (The staged path's crouton-row scratch has the
// same element type but no bufferization alignment, so this is the bridge's
// own allocation.)
// CHECK-NOT: {alignment = 128 : i64} : memref<1x32x16x32x2xf16, 1>
// The weight bridge, which fills a different array this pass does not retire.
// CHECK: hmx.pack_weight
// The staged ring the manifest records: a transfer issued and awaited, then the
// scratch packed from the slot it filled.
// CHECK: hmx.stage
// CHECK: hmx.await
// CHECK: hmx.pack_act
// The engine, inside the ring. Kt = 32 is exactly one hardware batch, so the
// K walk is one mma at k = 0 with the whole K as its repeat count -- no K loop.
// CHECK: hmx.acc_clear
// CHECK-NEXT: %[[KZ:.*]] = arith.constant 0 : index
// CHECK-NEXT: hmx.mma %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[KZ]] {n_croutons = 32 : i32}
// CHECK: hmx.acc_read
// The read-out is left where the matmul left it: a bare unpack, which the
// hoist does not take (it only moves an unpack that is itself the body of a
// row loop).
// CHECK: hmx.unpack_acc
// The N tile's write into its own slice of the output -- this is the store the
// old ownership claim deleted along with the loop.
// CHECK: memref.copy
// CHECK: return
// CHECK-NOT: hmx.matmul

// The unstaged arm on the same structure: the plain tile loop, the bridge kept
// whole in front of it, and the kernel's loop still standing.
// SERIAL-LABEL: func.func @nloop_only
// SERIAL: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// SERIAL: hmx.pack_act
// SERIAL: hmx.pack_weight
// SERIAL: hmx.acc_clear
// SERIAL: hmx.mma
// SERIAL: hmx.acc_read
// SERIAL: hmx.unpack_acc
// SERIAL: memref.copy
// SERIAL: return
// SERIAL-NOT: hmx.matmul
func.func @nloop_only(%a: memref<32x1024xf16>, %w: memref<1024x128xf16>,
                      %dst: memref<32x1024xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index

  // The kernel's N-blocking loop, and the matmul's only enclosing loop: the
  // m-loop and the k-loop are folded to a single iteration by construction.
  scf.for %j = %c0 to %c1024 step %c128 {
    // The bridge with no pack loop of its own: Mt = 1, so the outer-tile walk
    // canonicalized to this one pack. It sits in the kernel's loop, not in a
    // loop of its own -- the shape this test is about. The 128 B alignment is
    // what one-shot bufferization gives a production activation array.
    %ca = memref.alloc() {alignment = 128 : i64} : memref<1x32x16x32x2xf16, 1>
    hmx.pack_act ins(%a, %c0, %c0 : memref<32x1024xf16>)
                outs(%ca : memref<1x32x16x32x2xf16, 1>)
                {count = 32 : i64}

    // The weight bridge, one N tile per iteration of its own loop.
    %cw = memref.alloc() : memref<4x32x16x32x2xf16, 1>
    scf.for %n = %c0 to %c4 step %c1 {
      hmx.pack_weight ins(%w, %c0, %n : memref<1024x128xf16>)
                     outs(%cw : memref<4x32x16x32x2xf16, 1>)
                     {count = 32 : i64}
    }

    %ar = memref.alloc() : memref<1x4x16x32x2xf16, 1>
    hmx.matmul ins(%ca, %cw : memref<1x32x16x32x2xf16, 1>,
                              memref<4x32x16x32x2xf16, 1>)
               outs(%ar : memref<1x4x16x32x2xf16, 1>)

    // The read-out, then this N tile's write into its own slice of the output.
    %tile = memref.alloc() : memref<32x128xf16>
    %r = hmx.unpack_acc ins(%ar, %c0, %c0 : memref<1x4x16x32x2xf16, 1>)
                       outs(%tile : memref<32x128xf16>)
                       {count = 16 : i64}
                       -> memref<32x128xf16>
    %sub = memref.subview %dst[0, %j] [32, 128] [1, 1]
        : memref<32x1024xf16> to memref<32x128xf16, strided<[1024, 1], offset: ?>>
    memref.copy %r, %sub
        : memref<32x128xf16> to memref<32x128xf16, strided<[1024, 1], offset: ?>>

    memref.dealloc %ca : memref<1x32x16x32x2xf16, 1>
    memref.dealloc %cw : memref<4x32x16x32x2xf16, 1>
    memref.dealloc %ar : memref<1x4x16x32x2xf16, 1>
    memref.dealloc %tile : memref<32x128xf16>
  }
  return
}
