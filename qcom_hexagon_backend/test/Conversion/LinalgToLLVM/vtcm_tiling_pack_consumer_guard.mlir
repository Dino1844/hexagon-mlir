//===- vtcm_tiling_pack_consumer_guard.mlir - the pack-consumer guard ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The copy-back in `copyResultsToDDR` exists to serve DDR consumers. A
// read-modify-write out that is staged into VTCM therefore keeps its VTCM
// buffer when every consumer of the result can read that buffer directly:
// `hmx.matmul` must (the verifier rejects a space-0 crouton operand) and
// `hmx.pack_act` / `hmx.pack_weight` may (`verifyPackSource` constrains the
// source's shape and row stride but not its address space; the matmul ring has
// pack_act reading VTCM slots on device).
//
// The matmul clause was unreachable: LowerPack runs before VTCMTiling, so the
// consumer of a crouton-shaped generic result is a `hmx.pack_*`, never the
// matmul itself. The pack clause closes that gap: a staged result with only
// pack consumers no longer makes a DDR round trip to hand the pack a buffer it
// could have read in place.
//
// Two arms pin the rule:
//   * pack is the only consumer  -> the result stays in VTCM, no copy-back,
//     and the decision says so with a remark (pinned by -verify-diagnostics);
//   * one DDR consumer remains   -> the copy-back stays (it serves that
//     consumer) and the guard is silent.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(vtcm-tiling))' -verify-diagnostics
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(vtcm-tiling), \
// RUN:   canonicalize, one-shot-bufferize, func.func(buffer-loop-hoisting), \
// RUN:   canonicalize, buffer-deallocation-pipeline)' | FileCheck %s
//===----------------------------------------------------------------------===//

#mapId2 = affine_map<(d0, d1) -> (d0, d1)>
#mapRow = affine_map<(d0, d1) -> (d0)>
module {
// Same generic in both arms: a broadcast input (so the op is not the streaming
// skip) and a read-modify-write out (the out reads its init block arg, so the
// out is staged and reaches the copy-back decision).
func.func @pack_only_consumer_keeps_vtcm(
               %A : memref<64x64xf16>,
               %B : memref<64xf16>,
               %C : memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %tA = bufferization.to_tensor %A restrict : memref<64x64xf16> to tensor<64x64xf16>
  %tB = bufferization.to_tensor %B restrict : memref<64xf16> to tensor<64xf16>
  %tC = bufferization.to_tensor %C restrict writable : memref<64x64xf16> to tensor<64x64xf16>

  // expected-remark @+1 {{vtcm-tiling: staged result stays in VTCM; every consumer is an hmx.pack_* that reads the buffer directly, so the DDR copy-back is dropped}}
  %t3 = linalg.generic {
          indexing_maps = [#mapId2, #mapRow, #mapId2],
          iterator_types = ["parallel", "parallel"]}
          ins(%tA, %tB : tensor<64x64xf16>, tensor<64xf16>)
          outs(%tC : tensor<64x64xf16>) {
  ^bb0(%a: f16, %b : f16, %c : f16):
    %s = arith.mulf %a, %b : f16
    %r = arith.addf %s, %c : f16
    linalg.yield %r : f16
  } -> tensor<64x64xf16>

  %dst = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
  %packed = hmx.pack_act ins(%t3, %c0, %c0 : tensor<64x64xf16>)
              outs(%dst : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
  return
}
}

// -----

#mapId2 = affine_map<(d0, d1) -> (d0, d1)>
#mapRow = affine_map<(d0, d1) -> (d0)>
module {
func.func @mixed_consumers_keep_copy_back(
               %A : memref<64x64xf16>,
               %B : memref<64xf16>,
               %C : memref<64x64xf16>,
               %D : memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  %tA = bufferization.to_tensor %A restrict : memref<64x64xf16> to tensor<64x64xf16>
  %tB = bufferization.to_tensor %B restrict : memref<64xf16> to tensor<64xf16>
  %tC = bufferization.to_tensor %C restrict writable : memref<64x64xf16> to tensor<64x64xf16>
  %tD = bufferization.to_tensor %D restrict writable : memref<64x64xf16> to tensor<64x64xf16>

  %t3 = linalg.generic {
          indexing_maps = [#mapId2, #mapRow, #mapId2],
          iterator_types = ["parallel", "parallel"]}
          ins(%tA, %tB : tensor<64x64xf16>, tensor<64xf16>)
          outs(%tC : tensor<64x64xf16>) {
  ^bb0(%a: f16, %b : f16, %c : f16):
    %s = arith.mulf %a, %b : f16
    %r = arith.addf %s, %c : f16
    linalg.yield %r : f16
  } -> tensor<64x64xf16>

  %t4 = linalg.generic {
          indexing_maps = [#mapId2, #mapId2],
          iterator_types = ["parallel", "parallel"]}
          ins(%t3 : tensor<64x64xf16>)
          outs(%tD : tensor<64x64xf16>) {
  ^bb0(%a: f16, %d : f16):
    %r = arith.addf %a, %d : f16
    linalg.yield %r : f16
  } -> tensor<64x64xf16>
  bufferization.materialize_in_destination %t3 in writable %C
     : (tensor<64x64xf16>, memref<64x64xf16>) -> ()
  bufferization.materialize_in_destination %t4 in writable %D
     : (tensor<64x64xf16>, memref<64x64xf16>) -> ()

  %dst = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16>
  %packed = hmx.pack_act ins(%t3, %c0, %c0 : tensor<64x64xf16>)
              outs(%dst : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
  return
}
}

// The out is staged into a VTCM slot and its init value is copied in.
// CHECK-LABEL: func.func @pack_only_consumer_keeps_vtcm(
// CHECK: %[[OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16, 1>
// CHECK: memref.copy %{{.*}}, %[[OUT]] : memref<64x64xf16> to memref<64x64xf16, 1>
// CHECK: linalg.generic {{.*}} outs(%[[OUT]] : memref<64x64xf16, 1>) {
// No copy-back: the generic writes the VTCM slot and the pack reads it there.
// CHECK-NOT: memref.copy
// CHECK: hmx.pack_act ins(%[[OUT]],

// The DDR consumer keeps the round trip: the staged result is copied to a DDR
// buffer, the elementwise consumer and the materialization read that buffer,
// and so does the pack. The guard is silent here (the first RUN pins that no
// remark fires, because an unexpected one fails -verify-diagnostics).
// CHECK-LABEL: func.func @mixed_consumers_keep_copy_back(
// CHECK: %[[OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16, 1>
// CHECK: memref.copy %{{.*}}, %[[OUT]] : memref<64x64xf16> to memref<64x64xf16, 1>
// CHECK: linalg.generic {{.*}} outs(%[[OUT]] : memref<64x64xf16, 1>) {
// CHECK: %[[DDR:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16>
// CHECK: memref.copy %[[OUT]], %[[DDR]] : memref<64x64xf16, 1> to memref<64x64xf16>
// CHECK: hmx.pack_act ins(%[[DDR]],
