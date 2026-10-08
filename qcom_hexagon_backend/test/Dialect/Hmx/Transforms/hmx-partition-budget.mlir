//===- hmx-partition-budget.mlir - the pipeline yields to the budget -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The software pipeline needs one crouton-row scratch (one 32 x K f16 tile, the
// pack destination) plus `depth` staging slots (one 32 x K f16 tile each) and a
// status word per slot. The staged path retires the whole activation array, so
// its bytes come back to the room. When even the serial ring (depth 1) plus the
// scratch does not fit the budget the pass is given, the staged loop is
// declined with a remark: the pipeline is an optimisation, never a precondition
// for correctness. The serial tile loop then runs -- and the serial path still
// folds its whole-array bridge into the m-tile loop (S2.5): the fold needs only
// the 65536-byte scratch, and the 131072-byte activation array it retires
// leaves plenty of room for that, so a budget that kills the ring does not kill
// the fold. The depth-2 ring that does not fit but leaves the depth-1 ring
// fitting is hmx-partition-serial-ring.mlir.
//
// The shape uses Kt=32 so the scratch and each ring slot are a full 32 x 1024
// tile. The committed arrays (activation 2x32x16x32x2 131072, weight
// 2x32x16x32x2 131072, accumulator 2x2x16x32x2 8192, plus the 256-byte
// conversion state) come to 270592 bytes; the staged path frees the 131072-byte
// activation, so the room is `budget - 270592 + 131072 = budget - 139520`. A
// 270592-byte budget leaves 131072 free: the 65536-byte scratch plus the
// 65540-byte serial ring needs 131076 -- four bytes (the status word) more than
// that, so not even the serial ring fits. The fold's scratch alone (65536)
// fits twice over in that room, so the folded serial loop is what runs.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{vtcm-budget=270592}))' -verify-diagnostics
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{vtcm-budget=270592}))' | FileCheck %s
//===----------------------------------------------------------------------===//

// Under this budget not even one staging slot fits, the pass says why, and the
// serial path folds the bridge: the whole 2x32 activation array is retired and
// a one-row scratch takes its place. The activation and the weight share the
// 2x32 shape here, so the array's absence is pinned structurally instead: the
// only `hmx.pack_act` reads a strided 32x1024 source view (not the source, and
// not into any 2x32 destination) and the mma reads the one-row scratch.
// CHECK-LABEL: func.func @over_budget
// CHECK: hmx.bias_init
// CHECK: %[[W:.*]] = memref.alloc() : memref<2x32x16x32x2xf16, 1>
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<2x2x16x32x2xf16, 1>
// CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<1x32x16x32x2xf16, 1>
//
// The folded serial source loop: pack(m) from the source view at row m*32,
// then the engine nest. The whole K extent (32 croutons) is walked by ONE
// `hmx.mma` carrying `n_croutons = 32` rather than by a software K loop
// (`hfm` packs the hardware repeat-count field), so no third `scf.for` sits
// between the clear and the read.
// CHECK: scf.for %[[M:.*]] = {{.*}} to
// CHECK: %[[ROW:.*]] = arith.muli %[[M]], {{.*}} : index
// CHECK: hmx.pack_act ins(%{{.*}} : memref<32x1024xf16, strided<[1024, 1], offset: ?>>) outs(%[[SCRATCH]] : memref<1x32x16x32x2xf16, 1>) {count = 32 : i64}
// CHECK: scf.for %[[N:.*]] = {{.*}} to
// CHECK: hmx.acc_clear
// CHECK-NOT: scf.for
// CHECK: hmx.mma %[[SCRATCH]], %[[W]], {{.*}}, %[[N]], {{.*}} {n_croutons = 32 : i32}
// CHECK: hmx.acc_read {{.*}}, %[[ACC]], %[[M]], %[[N]] {bias_set = 0 : i32}
// The scratch is released after the loop.
// CHECK: memref.dealloc %[[SCRATCH]]
// CHECK-NOT: hmx.stage
// CHECK-NOT: hmx.await
// CHECK-NOT: memref.dma_start
// CHECK-NOT: hmx.matmul
func.func @over_budget(%a: memref<64x1024xf16>, %w: memref<1024x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %ca = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c32 : index
    %cc = arith.remui %i, %c32 : index
    hmx.pack_act ins(%a, %r, %cc : memref<64x1024xf16>) outs(%ca : memref<2x32x16x32x2xf16, 1>)
  }
  %cw = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  scf.for %i = %c0 to %c64 step %c1 {
    %r = arith.divui %i, %c2 : index
    %cc = arith.remui %i, %c2 : index
    hmx.pack_weight ins(%w, %r, %cc : memref<1024x64xf16>) outs(%cw : memref<2x32x16x32x2xf16, 1>)
  }
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  // expected-remark @+1 {{HMX pipeline not applied: activation staging needs 131076 bytes of VTCM (one crouton scratch plus the serial ring), only 131072 are free}}
  hmx.matmul ins(%ca, %cw : memref<2x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%ar : memref<2x2x16x32x2xf16, 1>)
  return
}
