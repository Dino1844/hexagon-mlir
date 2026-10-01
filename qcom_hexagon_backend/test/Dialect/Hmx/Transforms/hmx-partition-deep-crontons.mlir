//===- hmx-partition-deep-crontons.mlir - K is walked in engine batches ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// `hmx.mma`'s `n_croutons` is the engine's K repeat count: the op starts at
// crouton (m, k) and walks forward `n_croutons` croutons (HmxOps.td:474-476),
// and the count travels only in that attribute -- the address is a function of
// `k` alone (`test/Conversion/HmxToLLVM/mma-deep-croutons.mlir`). So K is walked
// as
//
//     Kt <= 32            : ONE mma at k = 0 with n_croutons = Kt, no loop
//     Kt  = 32q + r, r > 0 : q mmas of 32 at k = 0, 32, .. 32(q-1), then one
//                           tail of r at k = 32q
//     Kt  = 32q           : q mmas of 32 and NO tail
//
// The three Kt values that separate those cases are 31 / 32 / 33 (and 64 / 65
// for the two-batch versions), and they are all here. The tail is the case that
// is easy to get wrong in both directions: at `Kt % 32 == 0` emitting one would
// be a zero-count mma the verifier rejects, and dropping it when `Kt % 32 != 0`
// would silently not multiply the last r croutons -- a wrong answer, not an
// error.
//
// The batch count is a hardware bound, not a policy: `Rt[dC]` is five bits, so
// one mma covers at most 32 croutons = 1024 input channels (V81 PRM 4.2.1,
// "up to 32 croutons (1024 input channels)"), which is the verifier's
// `n_croutons <= 32` (HmxOps.cpp:410-411).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' | FileCheck %s
//===----------------------------------------------------------------------===//

// ---- Kt = 2 (the S1 anchor's depth): the whole K is one instruction, so the
// `Kt`-trip software loop the one-per-cronton form emitted is gone entirely.
// CHECK-LABEL: func.func @kt2
// CHECK: scf.for %[[M:.*]] =
// CHECK: scf.for %[[N:.*]] =
// CHECK: hmx.acc_clear
// CHECK-NOT: scf.for
// CHECK: %[[K0:.*]] = arith.constant 0 : index
// CHECK-NEXT: hmx.mma %{{.*}}, %{{.*}}, %[[M]], %[[N]], %[[K0]] {n_croutons = 2 : i32}
// CHECK: hmx.acc_read
func.func @kt2() {
  %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}

// ---- Kt = 31: one short of a batch, and still ONE instruction. If the emitter
// batched by 32 unconditionally this would have become a one-trip loop.
// CHECK-LABEL: func.func @kt31
// CHECK: scf.for %[[M31:.*]] =
// CHECK: scf.for %[[N31:.*]] =
// CHECK: hmx.acc_clear
// CHECK-NOT: scf.for
// CHECK: %[[K31:.*]] = arith.constant 0 : index
// CHECK-NEXT: hmx.mma %{{.*}}, %{{.*}}, %[[M31]], %[[N31]], %[[K31]] {n_croutons = 31 : i32}
// CHECK: hmx.acc_read
func.func @kt31() {
  %a = memref.alloc() : memref<2x31x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x31x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x31x16x32x2xf16, 1>, memref<2x31x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}

// ---- Kt = 32: exactly one batch. Still no loop -- the batch loop would have
// one iteration, and paying loop overhead for one engine packet is the cost
// this change exists to remove.
// CHECK-LABEL: func.func @kt32
// CHECK: scf.for %[[M32:.*]] =
// CHECK: scf.for %[[N32:.*]] =
// CHECK: hmx.acc_clear
// CHECK-NOT: scf.for
// CHECK: %[[K32:.*]] = arith.constant 0 : index
// CHECK-NEXT: hmx.mma %{{.*}}, %{{.*}}, %[[M32]], %[[N32]], %[[K32]] {n_croutons = 32 : i32}
// CHECK: hmx.acc_read
func.func @kt32() {
  %a = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}

// ---- Kt = 33: one full batch in a loop that steps k by 32, then a tail of 1 at
// k = 32. The induction variable IS the crouton index -- no multiply -- and the
// loop's upper bound is `floor(Kt/32) * 32`, NOT Kt: an upper bound of 33 with
// step 32 would still run once, but a bound of 34 would run twice and the second
// iteration would read croutons past the end of the array.
// CHECK-LABEL: func.func @kt33
// CHECK: scf.for %[[M33:.*]] =
// CHECK: scf.for %[[N33:.*]] =
// CHECK: hmx.acc_clear
// CHECK: %[[C0_33:.*]] = arith.constant 0 : index
// CHECK: %[[C32_33:.*]] = arith.constant 32 : index
// CHECK: %[[CEND_33:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[K33:.*]] = %[[C0_33]] to %[[CEND_33]] step %[[C32_33]] {
// CHECK: hmx.mma %{{.*}}, %{{.*}}, %[[M33]], %[[N33]], %[[K33]] {n_croutons = 32 : i32}
// CHECK: }
// CHECK: %[[CTA_33:.*]] = arith.constant 32 : index
// CHECK-NEXT: hmx.mma %{{.*}}, %{{.*}}, %[[M33]], %[[N33]], %[[CTA_33]] {n_croutons = 1 : i32}
// CHECK: hmx.acc_read
func.func @kt33() {
  %a = memref.alloc() : memref<2x33x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x33x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x33x16x32x2xf16, 1>, memref<2x33x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}

// ---- Kt = 64: two batches and NO tail. The bound is Kt itself (32*2), so this
// is the case where a `if (rem) emit tail` written without the guard would add a
// zero-count mma and the verifier would reject the module.
// CHECK-LABEL: func.func @kt64
// CHECK: scf.for %[[M64:.*]] =
// CHECK: scf.for %[[N64:.*]] =
// CHECK: hmx.acc_clear
// CHECK: %[[C0_64:.*]] = arith.constant 0 : index
// CHECK: %[[C32_64:.*]] = arith.constant 32 : index
// CHECK: %[[CEND_64:.*]] = arith.constant 64 : index
// CHECK: scf.for %[[K64:.*]] = %[[C0_64]] to %[[CEND_64]] step %[[C32_64]] {
// CHECK: hmx.mma %{{.*}}, %{{.*}}, %[[M64]], %[[N64]], %[[K64]] {n_croutons = 32 : i32}
// CHECK: }
// Nothing but the read-out between the loop's end and the accumulator drain:
// no tail mma, whose count would be 0.
// CHECK-NOT: hmx.mma
// CHECK: hmx.acc_read
func.func @kt64() {
  %a = memref.alloc() : memref<2x64x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x64x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x64x16x32x2xf16, 1>, memref<2x64x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}

// ---- Kt = 65: two batches, then a tail of 1 at k = 64.
// CHECK-LABEL: func.func @kt65
// CHECK: scf.for %[[M65:.*]] =
// CHECK: scf.for %[[N65:.*]] =
// CHECK: hmx.acc_clear
// CHECK: %[[C0_65:.*]] = arith.constant 0 : index
// CHECK: %[[C32_65:.*]] = arith.constant 32 : index
// CHECK: %[[CEND_65:.*]] = arith.constant 64 : index
// CHECK: scf.for %[[K65:.*]] = %[[C0_65]] to %[[CEND_65]] step %[[C32_65]] {
// CHECK: hmx.mma %{{.*}}, %{{.*}}, %[[M65]], %[[N65]], %[[K65]] {n_croutons = 32 : i32}
// CHECK: }
// CHECK: %[[CTA_65:.*]] = arith.constant 64 : index
// CHECK-NEXT: hmx.mma %{{.*}}, %{{.*}}, %[[M65]], %[[N65]], %[[CTA_65]] {n_croutons = 1 : i32}
// CHECK: hmx.acc_read
func.func @kt65() {
  %a = memref.alloc() : memref<2x65x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x65x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x65x16x32x2xf16, 1>, memref<2x65x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}