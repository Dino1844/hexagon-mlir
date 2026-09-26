//===- weight-resident-vtcm-budget-constant.mlir - no fallback for a constant -=//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The companion of weight-resident-vtcm-budget.mlir, which declines a runtime
// weight and keeps its bridge. A *constant* weight has no such route: the
// resident VTCM buffer is the only form its operand can legally take, because
// `hmx.mma` requires its weight in VTCM and the operand the resident replaces is
// a DDR `memref.get_global`. So the same capacity gate is reported as an error
// here rather than as a decline.
//
// The alternative was checked, not assumed. Leaving the weight on the DDR global
// does not produce a slower kernel, it produces a module that fails verification
// three passes later in `hmx-partition`:
//
//   error: 'hmx.mma' op wt must be in VTCM (memory space 1)
//
// which names neither the budget nor the residency, so the real cause is
// invisible from the diagnostic. Inventing a per-launch copy for the constant
// case would be a new mechanism with no lowering behind it, so the overrun is
// reported where the numbers are known and compilation stops.
//
// Same arithmetic as the companion file: budget 8388608 = 4096 croutons, weight
// [32, 32] = 2097152 bytes, activation and accumulator [1, 32] = 65536 bytes each.
// The constant global is in DDR, so unlike the runtime case the transient walk
// does not count the weight twice -- which is why the boundary here moves in
// croutons of ballast one at a time, and why these two files reach it with
// different ballast sizes.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident))' -verify-diagnostics -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// One crouton below the budget: total 8386560 < 8388608, so the resident buffer
// is created exactly as before the gate existed.
// CHECK: hmx.weight_resident_bytes = 2097152 : i64
// CHECK: memref.global "public" constant @__w : memref<32x32x16x32x2xf16>
// CHECK-LABEL: func.func @constant_just_below_budget
// CHECK: %[[W:.*]] = hexagonmem.alloc() {hmx.weight_resident = {bytes = 2097152 : i64, global = @__w}} : memref<32x32x16x32x2xf16, 1>
// CHECK: hmx.matmul ins(%{{.*}}, %[[W]] :
module {
  memref.global "private" constant @__w : memref<32x32x16x32x2xf16> = uninitialized
  func.func @constant_just_below_budget() {
    %ballast = memref.alloc() : memref<3007x1x16x32x2xf16, 1>
    memref.dealloc %ballast : memref<3007x1x16x32x2xf16, 1>
    %a = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    %w = memref.get_global @__w : memref<32x32x16x32x2xf16>
    %r = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    hmx.matmul ins(%a, %w : memref<1x32x16x32x2xf16, 1>, memref<32x32x16x32x2xf16>) outs(%r : memref<1x32x16x32x2xf16, 1>)
    return
  }
}

// -----

// Exactly at the budget: total 8388608, refused with the numbers. The pass
// stops, so this split contributes no output for FileCheck to match -- which is
// the point: there is no IR this pass can emit here.
module {
  memref.global "private" constant @__w : memref<32x32x16x32x2xf16> = uninitialized
  func.func @constant_exactly_at_budget() {
    %ballast = memref.alloc() : memref<3008x1x16x32x2xf16, 1>
    memref.dealloc %ballast : memref<3008x1x16x32x2xf16, 1>
    %a = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    %w = memref.get_global @__w : memref<32x32x16x32x2xf16>
    %r = memref.alloc() : memref<1x32x16x32x2xf16, 1>
    // expected-error @+1 {{resident weight @__w needs 2097152 bytes but the persistent VTCM total would be 8388608 bytes, over the 8388608 byte budget (transient 6291456 + resident 0 + this buffer); a constant weight has no per-launch fallback, so the module cannot be lowered}}
    hmx.matmul ins(%a, %w : memref<1x32x16x32x2xf16, 1>, memref<32x32x16x32x2xf16>) outs(%r : memref<1x32x16x32x2xf16, 1>)
    return
  }
}
