//===- resident-provenance-slot-conflict.mlir - one slot, two views --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))'
//===----------------------------------------------------------------------===//

module @slot_conflict attributes {hmx.diagnostic_vtcm_accounting,
                                  hmx.diagnostic_vtcm_identity} {
  func.func @same_slot_two_views(
      %a: memref<2x2x16x32x2xf16, 1>,
      %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %view0 = memref.reinterpret_cast %w to offset: [0], sizes: [64, 32], strides: [64, 1]
        : memref<64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: 0>>
    %view1 = memref.reinterpret_cast %w to offset: [32], sizes: [64, 32], strides: [64, 1]
        : memref<64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: 32>>
    %wa0 = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    %wa1 = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    %r0 = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    %r1 = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    scf.for %i0 = %c0 to %c2 step %c1 {
      %rr0 = arith.divui %i0, %c1 : index
      %cc0 = arith.remui %i0, %c1 : index
      hmx.pack_weight ins(%view0, %rr0, %cc0 : memref<64x32xf16, strided<[64, 1], offset: 0>>)
        outs(%wa0 : memref<1x2x16x32x2xf16, 1>)
    }
    scf.for %i1 = %c0 to %c2 step %c1 {
      %rr1 = arith.divui %i1, %c1 : index
      %cc1 = arith.remui %i1, %c1 : index
      hmx.pack_weight ins(%view1, %rr1, %cc1 : memref<64x32xf16, strided<[64, 1], offset: 32>>)
        outs(%wa1 : memref<1x2x16x32x2xf16, 1>)
    }
    hmx.matmul ins(%a, %wa0 : memref<2x2x16x32x2xf16, 1>,
                         memref<1x2x16x32x2xf16, 1>)
      outs(%r0 : memref<2x1x16x32x2xf16, 1>)
    // expected-error @+1 {{strict resident runtime weight slot has conflicting source views}}
    hmx.matmul ins(%a, %wa1 : memref<2x2x16x32x2xf16, 1>,
                         memref<1x2x16x32x2xf16, 1>)
      outs(%r1 : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa0 : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %wa1 : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %r0 : memref<2x1x16x32x2xf16, 1>
    memref.dealloc %r1 : memref<2x1x16x32x2xf16, 1>
    return
  }
}
