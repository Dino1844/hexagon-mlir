//===- hmx-partition-readout-pricing.mlir - the channel's Phase-A price ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The read-out channel of `auto` staging is a binary verdict today: a matmul
// an outer loop re-executes keeps the channel closed (the staged ring's fixed
// cost is per tile-loop EXECUTION). Phase A of
// docs/hmx/readout-channel-pricing-design-2026-10-10.md section 4 prices the
// same channel -- `open <=> g-hat(geometry) > R`, per execution, T cancels --
// WITHOUT moving the verdict, and remarks only where the price and the
// verdict DISAGREE. Agreement is the silent case: that is what keeps the
// default configuration (where the stock operators price as the gate already
// decides) at zero new remarks.
//
//   R     = 7.4 us per ring (same-build FA measurement; 7-18 cross-build)
//   price = 12.95 us per MiB of read-out (S1's unpack share; the leaf's
//           small-call price is deliberately not used -- it would price FA's
//           128 KiB read-out as open, against its measured -236 us)
//
// A read-out is Mt * Nt * 32 * 32 * 2 bytes, so the price opens at
// Mt * Nt > 292 tiles (7.4 / 12.95 MiB). The two disagreement forms:
//
//   * priced OPEN, gate CLOSED: S1's geometry inside an outer loop. The
//     remark names the miss and its per-execution and per-launch size.
//   * priced CLOSED, gate OPEN: a shallow-K loop-free shape whose read-out is
//     too small to pay for a ring. The m-tile floor note is part of the text
//     so the remark cannot be read as "the channel would stage this".
//
// The third case is the one that must say nothing: flash attention's
// per-chunk geometry, where price and gate already agree. Zero change to the
// emitted IR is proven by the rest of the hmx-partition family running
// unmodified under this pass (the option defaults to 0, which is outside the
// channel's regime, so those files see no remark at all).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{staged-readout-m-tiles=8}))' -verify-diagnostics
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{staged-readout-m-tiles=8}))' | FileCheck %s
// A Triton compile never shows a remark -- this MLIR's DiagnosticEngine drops
// it below the Error print threshold before any handler sees it (Common.h) --
// so the observation record is taken through the stderr mirror instead. It is
// the same text, gated by HEXMLIR_DIAG_REMARKS, and this is its only coverage.
// RUN: env HEXMLIR_DIAG_REMARKS=1 linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{staged-readout-m-tiles=8}))' 2>&1 | FileCheck %s --check-prefix=MIRROR
//===----------------------------------------------------------------------===//

// Priced OPEN where the binary gate says CLOSED: S1's geometry (Mt=32,
// Nt=16, Kt=2 => a 1 MiB read-out, an estimated 12.95 us hideable against
// R = 7.4 us) sitting inside a 4-iteration outer loop. The verdict is
// unchanged -- no ring, no stage/await -- and the remark records the
// disagreement with its per-execution and per-launch size.
// CHECK-LABEL: func.func @priced_open_gate_closed
// CHECK: hmx.bias_init
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<32x16x16x32x2xf16, 1>
// The plain tile loop: the outer loop that closed the channel is still the
// reason nothing stages.
// CHECK: scf.for
// CHECK: scf.for
// CHECK: hmx.acc_clear
// CHECK: hmx.mma
// CHECK: hmx.acc_read
// CHECK-NOT: hmx.stage
// CHECK-NOT: hmx.await
// CHECK-NOT: hmx.matmul
func.func @priced_open_gate_closed(%a: memref<1024x64xf16>, %w: memref<64x512xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %ca = memref.alloc() : memref<32x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_act ins(%a, %r, %cc : memref<1024x64xf16>) outs(%ca : memref<32x2x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<16x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c32 step %c1 {
    %r = arith.divui %i, %c16 : index
    %cc = arith.remui %i, %c16 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<64x512xf16>) outs(%cw : memref<16x2x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<32x16x16x32x2xf16, 1>
  scf.for %j = %c0 to %c4 step %c1 {
    // The price disagrees with the gate, so the remark fires. The two
    // sibling remarks are the unchanged decisions this matmul takes: the
    // ShallowK decline and the serial-fold refusal (an outer loop
    // re-executes the matmul). All three are declared, because the
    // verifier treats an undeclared remark as a failure.
    // expected-remark-re @+3 {{HMX readout channel pricing {{.*Mt=32 Nt=16 Kt=2, the tile loop executes 4 times per launch.*1\.000 MiB per execution, an estimated 12\.95 us hideable.*pricing suggests OPEN, the binary gate says CLOSED, difference \+5\.55 us per execution \(\+22\.20 us per launch\)}}}}
    // expected-remark @+2 {{HMX pipeline not applied: Kt 2 is below the staging floor of 32 -- one transfer is too small to hide the DMA engine's fixed cost behind the tile's compute}}
    // expected-remark @+1 {{HMX serial pack fold not applied: an outer loop re-executes this matmul}}
    hmx.matmul ins(%ca, %cw : memref<32x2x16x32x2xf16, 1>, memref<16x2x16x32x2xf16, 1>)
               outs(%ar : memref<32x16x16x32x2xf16, 1>)
  }
  memref.dealloc %ar : memref<32x16x16x32x2xf16, 1>
  memref.dealloc %cw : memref<16x2x16x32x2xf16, 1>
  memref.dealloc %ca : memref<32x2x16x32x2xf16, 1>
  return
}

// Priced CLOSED where the binary gate says OPEN: no outer loop, so the
// channel opens, but the read-out is only 128 KiB (Mt=4, Nt=16, Kt=2) -- an
// estimated 1.62 us against R = 7.4 us, a -5.78 us per execution miss. The
// staging decline itself is unchanged (ShallowK, on the m-tile floor this
// shape also trips, so the folded serial source loop is what runs) and the
// remark names that floor so it cannot be read as a lost win.
// CHECK-LABEL: func.func @priced_closed_gate_open
// CHECK: hmx.bias_init
// CHECK: %[[W:.*]] = memref.alloc() : memref<16x2x16x32x2xf16, 1>
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<4x16x16x32x2xf16, 1>
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<1x2x16x32x2xf16, 1>
// CHECK: scf.for %[[M:.*]] = {{.*}} to
// CHECK: hmx.pack_act ins(%{{.*}} : memref<32x64xf16, strided<[64, 1], offset: ?>>) outs(%[[SCRATCH]] : memref<1x2x16x32x2xf16, 1>) {count = 2 : i64}
// CHECK: scf.for %[[N:.*]] = {{.*}} to
// CHECK: hmx.acc_clear
// CHECK-NOT: scf.for
// CHECK: hmx.mma %[[SCRATCH]], %[[W]], {{.*}}, %[[N]], {{.*}} {n_croutons = 2 : i32}
// CHECK: hmx.acc_read {{.*}}, %[[ACC]], %[[M]], %[[N]] {bias_set = 0 : i32}
// CHECK: memref.dealloc %[[SCRATCH]]
// CHECK-NOT: hmx.stage
// CHECK-NOT: hmx.await
// CHECK-NOT: hmx.matmul
func.func @priced_closed_gate_open(%a: memref<128x64xf16>, %w: memref<64x512xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %ca = memref.alloc() : memref<4x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c8 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_act ins(%a, %r, %cc : memref<128x64xf16>) outs(%ca : memref<4x2x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<16x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c32 step %c1 {
    %r = arith.divui %i, %c16 : index
    %cc = arith.remui %i, %c16 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<64x512xf16>) outs(%cw : memref<16x2x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<4x16x16x32x2xf16, 1>
  // The price disagrees with the gate here too (the read-out is too small
  // to pay for a ring, yet nothing re-executes the matmul), and the note in
  // the text names the m-tile floor that declines the staging on its own.
  // The ShallowK remark is that decline, unchanged.
  // expected-remark-re @+2 {{HMX readout channel pricing {{.*Mt=4 Nt=16 Kt=2, the tile loop executes 1 time per launch.*0\.125 MiB per execution, an estimated 1\.62 us hideable.*pricing suggests CLOSED, the binary gate says OPEN, difference -5\.78 us per execution.*below the read-out batch floor of 8}}}}
  // expected-remark @+1 {{HMX pipeline not applied: Kt 2 is below the staging floor of 32 -- one transfer is too small to hide the DMA engine's fixed cost behind the tile's compute}}
  hmx.matmul ins(%ca, %cw : memref<4x2x16x32x2xf16, 1>, memref<16x2x16x32x2xf16, 1>)
             outs(%ar : memref<4x16x16x32x2xf16, 1>)
  memref.dealloc %ar : memref<4x16x16x32x2xf16, 1>
  memref.dealloc %cw : memref<16x2x16x32x2xf16, 1>
  memref.dealloc %ca : memref<4x2x16x32x2xf16, 1>
  return
}

// The disagreement-free shape says nothing: flash attention's per-chunk
// geometry (Kt=2, Mt=32, Nt=2, inside a 16-iteration loop) prices CLOSED --
// 128 KiB, an estimated 1.62 us against R = 7.4 us -- which is exactly what
// the binary gate already says, so the remark discipline keeps it silent.
// This is the case the default configuration lives in: a remark here would
// have been new noise on every stock flash-attention compile.
// CHECK-LABEL: func.func @priced_silent_when_agreeing
// CHECK: hmx.bias_init
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<32x2x16x32x2xf16, 1>
// CHECK: scf.for
// CHECK: hmx.mma
// CHECK-NOT: hmx.stage
// CHECK-NOT: hmx.await
// CHECK-NOT: hmx.matmul
// CHECK-NOT: HMX readout channel pricing
func.func @priced_silent_when_agreeing(%a: memref<1024x64xf16>, %w: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %ca = memref.alloc() : memref<32x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_act ins(%a, %r, %cc : memref<1024x64xf16>) outs(%ca : memref<32x2x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c4 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<64x64xf16>) outs(%cw : memref<2x2x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<32x2x16x32x2xf16, 1>
  scf.for %j = %c0 to %c16 step %c1 {
    // No pricing remark here -- the price and the gate agree. The two
    // unchanged decision remarks are still declared.
    // expected-remark @+2 {{HMX pipeline not applied: Kt 2 is below the staging floor of 32 -- one transfer is too small to hide the DMA engine's fixed cost behind the tile's compute}}
    // expected-remark @+1 {{HMX serial pack fold not applied: an outer loop re-executes this matmul}}
    hmx.matmul ins(%ca, %cw : memref<32x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
               outs(%ar : memref<32x2x16x32x2xf16, 1>)
  }
  memref.dealloc %ar : memref<32x2x16x32x2xf16, 1>
  memref.dealloc %cw : memref<2x2x16x32x2xf16, 1>
  memref.dealloc %ca : memref<32x2x16x32x2xf16, 1>
  return
}

// The stderr mirror (third RUN line): the two disagreements print the same
// text under the `[census]` prefix, in module order, and the agreeing shape
// prints nothing.
// MIRROR: [census] HMX readout channel pricing {{.*}}Mt=32 Nt=16 Kt=2
// MIRROR: [census] HMX readout channel pricing {{.*}}Mt=4 Nt=16 Kt=2
// MIRROR-NOT: [census]
