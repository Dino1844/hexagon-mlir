//===- weight-resident-vtcm-budget.mlir - the resident capacity gate ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A resident weight holds VTCM for the whole kernel, not for one launch, so the
// pass has to know whether the pool can still hold it before it commits. Before
// this gate it did not, and the device's fail-closed `requireAllocationResult`
// turned a compile-time-visible overrun into a PD abort at load time.
//
// The budget is `HmxTarget::defaultVtcmBudget` = 8 MiB = 8388608 bytes, and one
// crouton is 16*32*2 f16 = 2048 bytes, so the pool is exactly 4096 croutons.
// Every case below is a runtime weight of grid [32, 32] = 1024 croutons =
// 2097152 bytes, with an activation and an accumulator of [1, 32] = 32 croutons
// each and the pack bridge's own array = 1024 croutons. The `%ballast` crouton
// array is that function's remaining transient VTCM, the knob this file turns to
// walk the total to the boundary.
//
// One crouton of ballast moves the total by one crouton, which is why the
// boundary below is exact rather than approximate. Note what the total is made
// of at the "just below" case: the pack array is counted *and* the resident that
// will replace it, so a runtime weight is effectively charged for two copies of
// itself. That is the documented conservatism -- the pack array dies the moment
// the resident is what the engine reads, so the check can refuse a VTCM placement
// that would have fitted. The safe direction to be wrong in: it costs the pool a
// buffer it might have held, never HMX.
//
// What the gate decides for a runtime weight is **where the image lives**, not
// whether the weight is resident: a refusal puts the image in the permanent DDR
// mirror (the `location = "ddr"` descriptor below), replaces the pack with one
// contiguous fetch of the block the engine reads, and publishes the same
// pre-pack contract with `location` naming the placement. The DDR bytes are
// deliberately *not* added to `hmx.weight_resident_bytes`: that aggregate is the
// pool footprint the other budget readers compare against, and a mirror does not
// come out of the pool -- which is why the refused cases below declare no
// footprint while still becoming resident.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' -verify-diagnostics -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// Fits with room to spare (total 4325376 = 2112 croutons), and unchanged by the
// gate: the resident buffer is still created in VTCM, the bridge still goes
// away, and neither the descriptor nor the IR carries a placement fact -- VTCM
// is what a resident was before there was anywhere else to put one.
// CHECK: hmx.weight_resident_bytes = 2097152 : i64
// CHECK-LABEL: func.func @resident_fits
// CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 2097152 : i64}} : memref<32x32x16x32x2xf16, 1>
// CHECK: hmx.matmul ins(%{{.*}}, %[[W]] :
// CHECK-NOT: hmx.pack_weight
module {
  func.func @resident_fits(%a: memref<1024x1024xf16>, %w: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %aa = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<32x32x16x32x2xf16, 1>
    scf.for %i = %c0 to %c1024 step %c1 {
      %r = arith.divui %i, %c32 : index
      %c = arith.remui %i, %c32 : index
      hmx.pack_weight ins(%w, %r, %c : memref<1024x1024xf16>) outs(%wa : memref<32x32x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    hmx.matmul ins(%aa, %wa : memref<1x32x16x32x2xf16, 1>, memref<32x32x16x32x2xf16, 1>) outs(%ar : memref<1x32x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<32x32x16x32x2xf16, 1>
    memref.dealloc %aa : memref<1x32x16x32x2xf16, 1>
    memref.dealloc %ar : memref<1x32x16x32x2xf16, 1>
    return
  }
}

// -----

// Does not fit the pool. The weight becomes resident *somewhere else*: the DDR
// mirror, reached through the same argument-address key the VTCM entry uses.
// The engine keeps reading the bridge's own VTCM array, and the pack that used
// to fill it becomes one contiguous copy of the mirror's block -- so the pack
// loop is gone, the array and its deallocation stay (the copy writes it every
// iteration), and the host contract says where the image lives.
//
// The remark is the only thing that says *why* this weight is not in VTCM, so
// it carries the buffer and all four numbers.
// CHECK-NOT: hmx.weight_resident_bytes
// CHECK-LABEL: func.func @over_budget_lands_in_ddr
// CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 2097152 : i64, location = "ddr"}} : memref<32x32x16x32x2xf16>
// CHECK: memref.copy %[[W]], %{{.*}} : memref<32x32x16x32x2xf16> to memref<32x32x16x32x2xf16, 1>
// CHECK: hmx.matmul
// CHECK-NOT: hmx.pack_weight
module {
  func.func @over_budget_lands_in_ddr(%a: memref<1024x1024xf16>, %w: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    // 2048 croutons of transient VTCM the other buffers of the kernel hold.
    %ballast = memref.alloc() : memref<2048x1x16x32x2xf16, 1>
    memref.dealloc %ballast : memref<2048x1x16x32x2xf16, 1>
    %aa = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<32x32x16x32x2xf16, 1>
    scf.for %i = %c0 to %c1024 step %c1 {
      %r = arith.divui %i, %c32 : index
      %c = arith.remui %i, %c32 : index
      hmx.pack_weight ins(%w, %r, %c : memref<1024x1024xf16>) outs(%wa : memref<32x32x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    // expected-remark @+1 {{resident weight for argument slot 1 does not fit the persistent VTCM pool: 2097152 bytes would make the persistent VTCM total 8519680 bytes, over the 8388608 byte budget (transient 6422528 + resident 0 + this buffer); placed in the DDR permanent region instead and its pack bridge becomes a contiguous fetch}}
    hmx.matmul ins(%aa, %wa : memref<1x32x16x32x2xf16, 1>, memref<32x32x16x32x2xf16, 1>) outs(%ar : memref<1x32x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<32x32x16x32x2xf16, 1>
    memref.dealloc %aa : memref<1x32x16x32x2xf16, 1>
    memref.dealloc %ar : memref<1x32x16x32x2xf16, 1>
    return
  }
}

// -----

// The boundary, one crouton below it: total 8386560 = 4095 croutons, which is
// the budget less a single crouton. Admitted into VTCM, so the resident is
// created there and the bridge is dropped. 1983 croutons of ballast, against
// 1984 for the placement below: that one-cronton difference is the whole
// boundary.
// CHECK: hmx.weight_resident_bytes = 2097152 : i64
// CHECK-LABEL: func.func @just_below_budget
// CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 2097152 : i64}} : memref<32x32x16x32x2xf16, 1>
// CHECK: hmx.matmul ins(%{{.*}}, %[[W]] :
// CHECK-NOT: hmx.pack_weight
module {
  func.func @just_below_budget(%a: memref<1024x1024xf16>, %w: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %ballast = memref.alloc() : memref<1983x1x16x32x2xf16, 1>
    memref.dealloc %ballast : memref<1983x1x16x32x2xf16, 1>
    %aa = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<32x32x16x32x2xf16, 1>
    scf.for %i = %c0 to %c1024 step %c1 {
      %r = arith.divui %i, %c32 : index
      %c = arith.remui %i, %c32 : index
      hmx.pack_weight ins(%w, %r, %c : memref<1024x1024xf16>) outs(%wa : memref<32x32x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    hmx.matmul ins(%aa, %wa : memref<1x32x16x32x2xf16, 1>, memref<32x32x16x32x2xf16, 1>) outs(%ar : memref<1x32x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<32x32x16x32x2xf16, 1>
    memref.dealloc %aa : memref<1x32x16x32x2xf16, 1>
    memref.dealloc %ar : memref<1x32x16x32x2xf16, 1>
    return
  }
}

// -----

// Exactly at the budget: total 8388608 = 4096 croutons, refused for VTCM. The
// boundary is the strict `<` that `HmxTarget::planBridge` also refuses at --
// the budget is a pool size, so a total equal to it leaves no room for the
// allocation's own 128 B alignment. Refusal means placement, so this case takes
// the same DDR route as the one above, one crouton's ballast lighter.
// CHECK-NOT: hmx.weight_resident_bytes
// CHECK-LABEL: func.func @exactly_at_budget
// CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 2097152 : i64, location = "ddr"}} : memref<32x32x16x32x2xf16>
// CHECK: memref.copy %[[W]], %{{.*}} : memref<32x32x16x32x2xf16> to memref<32x32x16x32x2xf16, 1>
// CHECK: hmx.matmul
// CHECK-NOT: hmx.pack_weight
module {
  func.func @exactly_at_budget(%a: memref<1024x1024xf16>, %w: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %ballast = memref.alloc() : memref<1984x1x16x32x2xf16, 1>
    memref.dealloc %ballast : memref<1984x1x16x32x2xf16, 1>
    %aa = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<32x32x16x32x2xf16, 1>
    scf.for %i = %c0 to %c1024 step %c1 {
      %r = arith.divui %i, %c32 : index
      %c = arith.remui %i, %c32 : index
      hmx.pack_weight ins(%w, %r, %c : memref<1024x1024xf16>) outs(%wa : memref<32x32x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    // expected-remark @+1 {{resident weight for argument slot 1 does not fit the persistent VTCM pool: 2097152 bytes would make the persistent VTCM total 8388608 bytes, over the 8388608 byte budget (transient 6291456 + resident 0 + this buffer); placed in the DDR permanent region instead and its pack bridge becomes a contiguous fetch}}
    hmx.matmul ins(%aa, %wa : memref<1x32x16x32x2xf16, 1>, memref<32x32x16x32x2xf16, 1>) outs(%ar : memref<1x32x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<32x32x16x32x2xf16, 1>
    memref.dealloc %aa : memref<1x32x16x32x2xf16, 1>
    memref.dealloc %ar : memref<1x32x16x32x2xf16, 1>
    return
  }
}

// -----

// The third term of the total: what *earlier* functions already committed. The
// pass is a per-function pass that writes `hmx.weight_resident_bytes` only when
// it ends, so a later function reads the earlier one's footprint off the module
// and adds its own on top. `@resident_first` is admitted (2112 croutons) and
// leaves 2097152 bytes resident; `@resident_second` is then charged for it, and
// 960 croutons of ballast put it at exactly 4096 -- refused for VTCM, placed in
// DDR. Drop the "resident so far" term and this one would wrongly go into the
// pool at 3072 croutons, so this case is what keeps that term honest; and
// because the DDR placement adds nothing to the aggregate, the footprint below
// stays at `@resident_first`'s 2097152 rather than growing by the mirror.
//
// The remark says `resident 2097152`, which is the term that is easy to get
// wrong: a pass that recomputed the total from its own function alone would
// report 0 there. The first CHECK also bounds the previous section's trailing
// CHECK-NOTs, which would otherwise scan forward into this module's attributes.
// CHECK: hmx.weight_resident_bytes = 2097152 : i64
// CHECK-LABEL: func.func @resident_first
// CHECK: hexagonmem.alloc
// CHECK-NOT: hmx.pack_weight
// CHECK-LABEL: func.func @resident_second
// CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 2097152 : i64, location = "ddr"}} : memref<32x32x16x32x2xf16>
// CHECK: memref.copy %[[W]], %{{.*}} : memref<32x32x16x32x2xf16> to memref<32x32x16x32x2xf16, 1>
// CHECK: hmx.matmul
// CHECK-NOT: hmx.pack_weight
module {
  func.func @resident_first(%a: memref<1024x1024xf16>, %w: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %aa = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<32x32x16x32x2xf16, 1>
    scf.for %i = %c0 to %c1024 step %c1 {
      %r = arith.divui %i, %c32 : index
      %c = arith.remui %i, %c32 : index
      hmx.pack_weight ins(%w, %r, %c : memref<1024x1024xf16>) outs(%wa : memref<32x32x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    hmx.matmul ins(%aa, %wa : memref<1x32x16x32x2xf16, 1>, memref<32x32x16x32x2xf16, 1>) outs(%ar : memref<1x32x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<32x32x16x32x2xf16, 1>
    memref.dealloc %aa : memref<1x32x16x32x2xf16, 1>
    memref.dealloc %ar : memref<1x32x16x32x2xf16, 1>
    return
  }

  func.func @resident_second(%a: memref<1024x1024xf16>, %w: memref<1024x1024xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %ballast = memref.alloc() : memref<960x1x16x32x2xf16, 1>
    memref.dealloc %ballast : memref<960x1x16x32x2xf16, 1>
    %aa = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<32x32x16x32x2xf16, 1>
    scf.for %i = %c0 to %c1024 step %c1 {
      %r = arith.divui %i, %c32 : index
      %c = arith.remui %i, %c32 : index
      hmx.pack_weight ins(%w, %r, %c : memref<1024x1024xf16>) outs(%wa : memref<32x32x16x32x2xf16, 1>)
    }
    %ar = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    // expected-remark @+1 {{resident weight for argument slot 1 does not fit the persistent VTCM pool: 2097152 bytes would make the persistent VTCM total 8388608 bytes, over the 8388608 byte budget (transient 4194304 + resident 2097152 + this buffer); placed in the DDR permanent region instead and its pack bridge becomes a contiguous fetch}}
    hmx.matmul ins(%aa, %wa : memref<1x32x16x32x2xf16, 1>, memref<32x32x16x32x2xf16, 1>) outs(%ar : memref<1x32x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<32x32x16x32x2xf16, 1>
    memref.dealloc %aa : memref<1x32x16x32x2xf16, 1>
    memref.dealloc %ar : memref<1x32x16x32x2xf16, 1>
    return
  }
}
