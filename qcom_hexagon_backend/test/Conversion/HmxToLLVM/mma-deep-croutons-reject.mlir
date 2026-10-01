//===- mma-deep-croutons-reject.mlir - n_croutons bounds ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The two ends of the repeat count's range. Both matter to a batched K emitter:
// 33 is the overflow into the engine's spatial mask, and 0 is what a
// `Kt % 32 == 0` tail would emit if the tail were not guarded (the batch count
// is floor(Kt/32), so a Kt that IS a multiple of 32 has no tail at all).
//
// RUN: linalg-hexagon-opt %s -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

// The engine's repeat field Rt[dC] is five bits, so 33 would silently overflow
// into the spatial mask in the runtime leaf (HmxOps.cpp:407-411).
func.func @reject_33(%act: memref<4x33x16x32x2xf16, 1>, %wt: memref<3x33x16x32x2xf16, 1>, %m: index, %n: index, %k: index) {
  // expected-error @+1 {{n_croutons must be at most 32}}
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 33 : i32} : memref<4x33x16x32x2xf16, 1>, memref<3x33x16x32x2xf16, 1>
  return
}

// -----

// Zero is not a repeat count, and a batched emitter that computed
// `min(Kt, 32) - something` could produce it.
func.func @reject_0(%act: memref<4x32x16x32x2xf16, 1>, %wt: memref<3x32x16x32x2xf16, 1>, %m: index, %n: index, %k: index) {
  // expected-error @+1 {{n_croutons must be at least 1}}
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 0 : i32} : memref<4x32x16x32x2xf16, 1>, memref<3x32x16x32x2xf16, 1>
  return
}