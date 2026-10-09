//===- hmx-vtcm-ledger-memref.mlir - the ledger's memref population --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The memref half of HmxVtcmLedger.h: the walk over static space-1
// `memref.alloc`s plus the `hmx.weight_resident_bytes` declaration, which is
// where a resident `hexagonmem.alloc`'s footprint enters (the walk cannot see
// it -- it is not a `memref.alloc`). The numbers are the old walks' arithmetic,
// computed by hand; the existing budget fixtures keep their own literals.
//
// The function's own static space-1 allocations
//
//   activation  2x32x16x32x2xf16   131072
//   weight      2x32x16x32x2xf16   131072
//   accumulator 2x2x16x32x2xf16      8192
//   workspace   64x16xf16            2048   (tagged workspace-resident: an
//                                           ordinary memref.alloc to the walk)
//   resident declaration (module)    4096   (the hexagonmem.alloc above)
//
// = **276736**. hmx-partition creates its own 256 B conversion state before it
// asks, so at the budget check the ledger returns 272384 + 256 + 4096 = 276736.
//
// With `vtcm-budget=200000` and the 131072 B activation array retired by the
// staging path, `room = 200000 - 276736 + 131072` = **54336**: smaller than
// both the 131076 B serial ring and the fold's 65536 B scratch, so both paths
// decline and both quote that same room -- two call sites, one number.
//
// The census runs behind the partition on the IR it produced and re-derives
// the very same total from allocation sites: 270592 transient (its own 256 B
// state included) + 2048 workspace-resident + 4096 weight-resident = **276736**.
// The census is the ledger's strict superset; on this IR the superset is tight.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{vtcm-budget=200000}), hmx-vtcm-accounting)' 2>%t.err | FileCheck %s --check-prefix=CENSUS
// RUN: FileCheck %s --check-prefix=ROOM < %t.err
//===----------------------------------------------------------------------===//

// Both declines quote the room: 200000 - 276736 + 131072.
// ROOM: activation staging needs 131076 bytes of VTCM (one crouton scratch plus the serial ring), only 54336 are free
// ROOM: HMX serial pack fold not applied: the crouton-row scratch needs 65536 bytes of VTCM, only 54336 are free
// CENSUS-DAG: status = "complete"
// CENSUS-DAG: transient_bytes = 270592 : i64
// CENSUS-DAG: workspace_resident_bytes = 2048 : i64
// CENSUS-DAG: weight_resident_bytes = 4096 : i64
// CENSUS-DAG: raw_site_sum_bytes = 276736 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.weight_resident_bytes = 4096 : i64} {
  memref.global "private" constant @weight : memref<64x32xf16> = dense<1.000000e+00> {alignment = 128 : i64}

  func.func @memref_adapter(%a: memref<64x1024xf16>, %w: memref<1024x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    // The resident weight: a `hexagonmem.alloc`, invisible to the memref walk,
    // so its 4096 bytes reach the ledger through the declaration above.
    // Unused on purpose -- the budget is what is under test.
    %wr = hexagonmem.alloc() {hmx.weight_resident = {global = @weight, bytes = 4096 : i64}}
        : memref<64x32xf16, 1>
    // A workspace-resident buffer is an ordinary `memref.alloc` to the walk:
    // the census files it under `workspace_resident_bytes`, the ledger counts
    // it in the walk. Same bytes either way.
    %ws = memref.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 2048 : i64}}
        : memref<64x16xf16, 1>
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
    hmx.matmul ins(%ca, %cw : memref<2x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
               outs(%ar : memref<2x2x16x32x2xf16, 1>)
    return
  }
}
