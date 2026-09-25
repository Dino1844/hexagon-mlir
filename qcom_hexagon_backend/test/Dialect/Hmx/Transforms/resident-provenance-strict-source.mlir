//===- resident-provenance-strict-source.mlir - strict source descriptor ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A strict resident source must be a ranked, row-major f16 entry memref. A
// statically shaped but non-row-major view is not silently treated as a dense
// whole-argument source.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))'
//===----------------------------------------------------------------------===//

module @strict_source attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity} {
  func.func @bad_stride(
      %a: memref<2x2x16x32x2xf16, 1>,
      %w: memref<64x64xf16, strided<[65, 1]>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16, strided<[65, 1]>>)
        outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    // expected-error @+1 {{strict resident runtime weight argument has no exact ranked row-major shape/stride}}
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>,
                         memref<2x2x16x32x2xf16, 1>)
      outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}
