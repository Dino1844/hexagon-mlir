//===- tail-partition-rejects.mlir - hmx-partition fail-closed gates ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
// Cases that used to live in separate files, one per rejection reason:
//   tail-bridge-loop-reject.mlir        - looped tail bridges fail closed
//   tail-external-writer-reject.mlir    - carried bridge ownership
//   tail-marker-invalid-partition.mlir  - marker type gate
//   tail-plan-invalid-reject.mlir       - stale plan arithmetic fails closed
//   tail-plan-partition-reject.mlir     - no silent tail-plan drop
//   tail-source-dominance-reject.mlir   - no non-dominating bridge use
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition))' -split-input-file -verify-diagnostics
//===----------------------------------------------------------------------===//

//===- tail-bridge-loop-reject.mlir - looped tail bridges fail closed ----===//

//===----------------------------------------------------------------------===//

func.func private @opaque()

func.func @reject_looped_bridge(%a: memref<33x33xf16>,
                                %w: memref<33x33xf16>,
                                %out: memref<33x33xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  scf.for %i = %c0 to %c2 step %c1 {
    func.call @opaque() : () -> ()
    hmx.pack_act ins(%a, %c0, %c0 : memref<33x33xf16>)
        outs(%act : memref<2x2x16x32x2xf16, 1>)
  }
  scf.for %j = %c0 to %c2 step %c1 {
    hmx.pack_weight ins(%w, %j, %j : memref<33x33xf16>)
        outs(%wt : memref<2x2x16x32x2xf16, 1>)
  }
  // expected-error @+1 {{diagnostic tail partition requires direct activation and weight pack bridges}}
  hmx.matmul ins(%act, %wt : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}
  hmx.unpack_acc ins(%ar, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  return
}

// -----

//===- tail-external-writer-reject.mlir - carried bridge ownership -------===//

//===----------------------------------------------------------------------===//

func.func @external_carried_writer(%a: memref<33x33xf16>,
                                   %w: memref<33x33xf16>) -> memref<33x33xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %act0 = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt0 = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %out0 = memref.alloc() : memref<33x33xf16>
  %act = scf.for %i = %c0 to %c4 step %c1 iter_args(%carry = %act0) -> (memref<2x2x16x32x2xf16, 1>) {
    %m = arith.divui %i, %c2 : index
    %k = arith.remui %i, %c2 : index
    %p = hmx.pack_act ins(%a, %m, %k : memref<33x33xf16>)
        outs(%carry : memref<2x2x16x32x2xf16, 1>) -> memref<2x2x16x32x2xf16, 1>
    scf.yield %p : memref<2x2x16x32x2xf16, 1>
  }
  %wt = scf.for %i = %c0 to %c4 step %c1 iter_args(%carry = %wt0) -> (memref<2x2x16x32x2xf16, 1>) {
    %k = arith.divui %i, %c2 : index
    %n = arith.remui %i, %c2 : index
    %p = hmx.pack_weight ins(%w, %k, %n : memref<33x33xf16>)
        outs(%carry : memref<2x2x16x32x2xf16, 1>) -> memref<2x2x16x32x2xf16, 1>
    scf.yield %p : memref<2x2x16x32x2xf16, 1>
  }
  // expected-error @+1 {{diagnostic tail partition found an unowned activation/weight bridge user}}
  hmx.matmul ins(%act, %wt : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}
  // This direct pack is outside the canonical carried loop and would be a
  // second writer of the array after the engine has consumed it.
  hmx.pack_act ins(%a, %c0, %c0 : memref<33x33xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%carry = %out0) -> (memref<33x33xf16>) {
    %m = arith.divui %i, %c1 : index
    %n = arith.remui %i, %c1 : index
    %u = hmx.unpack_acc ins(%ar, %m, %n : memref<2x2x16x32x2xf16, 1>)
        outs(%carry : memref<33x33xf16>) -> memref<33x33xf16>
    scf.yield %u : memref<33x33xf16>
  }
  return %out : memref<33x33xf16>
}

// -----

//===- tail-marker-invalid-partition.mlir - marker type gate ---------------===//

//===----------------------------------------------------------------------===//

func.func @invalid_partition_marker() {
  %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  // expected-error @+1 {{hmx.diagnostic_tail_partition must be a unit attribute}}
  hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%r : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition = "bad"}
  return
}

// -----

//===- tail-plan-invalid-reject.mlir - stale plan arithmetic fails closed --===//

//===----------------------------------------------------------------------===//

func.func @reject_invalid_tail_plan(%a: memref<2x2x16x32x2xf16, 1>,
                                     %w: memref<2x2x16x32x2xf16, 1>,
                                     %r: memref<2x2x16x32x2xf16, 1>) {
  // expected-error @+3 {{logical must equal full + tail at axis 0}}
  hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%r : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [0, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}
  return
}

// -----

//===- tail-plan-partition-reject.mlir - no silent tail-plan drop -------===//

//===----------------------------------------------------------------------===//

func.func @reject_tail_plan() {
  %a = memref.alloc() : memref<3x3x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x3x16x32x2xf16, 1>
  %r = memref.alloc() : memref<3x2x16x32x2xf16, 1>
  // expected-error @+1 {{tail_plan is not yet supported by hmx-partition}}
  hmx.matmul ins(%a, %w : memref<3x3x16x32x2xf16, 1>, memref<2x3x16x32x2xf16, 1>)
      outs(%r : memref<3x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [65, 47, 70], padded = [96, 64, 96], full = [64, 32, 64], tail = [1, 15, 6], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">}
  return
}

// -----

//===- tail-source-dominance-reject.mlir - no non-dominating bridge use ---===//

//===----------------------------------------------------------------------===//

func.func @tail_source_after_matmul() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %out = memref.alloc() : memref<33x33xf16>

  // expected-error @+1 {{diagnostic tail partition requires pack sources that dominate the matmul}}
  hmx.matmul ins(%act, %wt : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>)
      outs(%ar : memref<2x2x16x32x2xf16, 1>)
      {tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1], k_policy = "zero-pad-both-operands", mn_policy = "padded-edge-tile-bounded-store">, hmx.diagnostic_tail_partition}

  // The row-major source is defined only after the matmul. The diagnostic
  // rewrite must reject it rather than capture it in a newly inserted pack.
  %a = memref.alloc() : memref<64x64xf16>
  %w = memref.alloc() : memref<64x64xf16>
  hmx.pack_act ins(%a, %c0, %c0 : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c0, %c1 : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c0 : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_act ins(%a, %c1, %c1 : memref<64x64xf16>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c0, %c0 : memref<64x64xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c0, %c1 : memref<64x64xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c1, %c0 : memref<64x64xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%w, %c1, %c1 : memref<64x64xf16>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  hmx.unpack_acc ins(%ar, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%out : memref<33x33xf16>)
  return
}
