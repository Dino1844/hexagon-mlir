// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(vtcm-tiling{tile-size-override=32,128}), \
// RUN:         one-shot-bufferize,func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' \
// RUN: | FileCheck %s -check-prefixes=CHECK

// Operand-level staging rules (see computeOperandStaging in VTCMTiling.cpp):
// a generic is staged through VTCM per operand, not all-or-nothing --
//   * an input whose indexing map is injective over the iteration space is
//     read at most once and stays on its DDR buffer;
//   * an out whose region never reads the init block arg is write-only: its
//     old value cannot survive into the result, so it is written directly;
//   * everything else (broadcast or otherwise non-injective input maps,
//     read-modify-write outs) keeps the staged behavior.
// The whole-generic streaming skip (all-parallel + all-identity) is a separate,
// earlier decision and is not exercised here: every case below has either a
// non-identity map or a reduction loop.

#mapId2 = affine_map<(d0, d1) -> (d0, d1)>
#mapRow = affine_map<(d0, d1) -> (d0)>
module {
func.func @single_read_dropped_broadcast_kept_write_only_dropped(
               %A : memref<64x128xf32>,
               %B : memref<64xf32>,
               %C : memref<64x128xf32>) {
  %tA = bufferization.to_tensor %A restrict : memref<64x128xf32> to tensor<64x128xf32>
  %tB = bufferization.to_tensor %B restrict : memref<64xf32> to tensor<64xf32>
  %tC = bufferization.to_tensor %C restrict writable : memref<64x128xf32> to tensor<64x128xf32>

  %t3 = linalg.generic {
          indexing_maps = [#mapId2, #mapRow, #mapId2],
          iterator_types = ["parallel", "parallel"]}
          ins(%tA, %tB : tensor<64x128xf32>, tensor<64xf32>)
          outs(%tC : tensor<64x128xf32>) {
  ^bb0(%a: f32, %b : f32, %c : f32):
    %ab = arith.mulf %a, %b : f32
    linalg.yield %ab : f32
  } -> tensor<64x128xf32>

  bufferization.materialize_in_destination %t3 in writable %C
     : (tensor<64x128xf32>, memref<64x128xf32>) -> ()
  return
}
}

// A's map is the identity (single read) and the out never reads %c
// (write-only): both stay on DDR. B broadcasts over d1 (its map drops d1), so
// B alone is staged into VTCM.
// CHECK-LABEL: func.func @single_read_dropped_broadcast_kept_write_only_dropped(
// CHECK-SAME: %[[A:.+]]: memref<64x128xf32>, %[[B:.+]]: memref<64xf32>, %[[C:.+]]: memref<64x128xf32>
// The broadcast input B alone gets a VTCM buffer.
// CHECK: %[[ALLOC_B:.*]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32, 1>
// CHECK: scf.for %[[I:.+]] = %c0 to %c64 step %c32 {
// The identity-mapped input A is read straight from its DDR subview.
// CHECK: %[[VA:.*]] = memref.subview %[[A]][%[[I]], 0] [32, 128] [1, 1] : memref<64x128xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>
// CHECK: %[[VB:.*]] = memref.subview %[[B]][%[[I]]] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>
// CHECK: memref.copy %[[VB]], %[[ALLOC_B]] : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32, 1>
// The write-only out is written straight to its DDR subview: no init copy,
// no VTCM slot, no copy-back.
// CHECK: %[[VC:.*]] = memref.subview %[[C]][%[[I]], 0] [32, 128] [1, 1] : memref<64x128xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>
// CHECK: linalg.generic {{.*}} ins(%[[VA]], %[[ALLOC_B]] : memref<32x128xf32, strided<[128, 1], offset: ?>>, memref<32xf32, 1>) outs(%[[VC]] : memref<32x128xf32, strided<[128, 1], offset: ?>>) {
// CHECK-NOT: memref.copy
// CHECK: memref.dealloc %[[ALLOC_B]] : memref<32xf32, 1>

// -----

#mapId2 = affine_map<(d0, d1) -> (d0, d1)>
#mapRow = affine_map<(d0, d1) -> (d0)>
module {
func.func @rmw_out_kept(
               %A : memref<64x128xf32>,
               %B : memref<64xf32>) {
  %tA = bufferization.to_tensor %A restrict : memref<64x128xf32> to tensor<64x128xf32>
  %tB = bufferization.to_tensor %B restrict writable : memref<64xf32> to tensor<64xf32>

  %t2 = linalg.generic {
          indexing_maps = [#mapId2, #mapRow],
          iterator_types = ["parallel", "reduction"]}
          ins(%tA : tensor<64x128xf32>)
          outs(%tB : tensor<64xf32>) {
  ^bb0(%a: f32, %b : f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>

  bufferization.materialize_in_destination %t2 in writable %B
     : (tensor<64xf32>, memref<64xf32>) -> ()
  return
}
}

// The d1 loop is a reduction whose body accumulates into the out: the out is
// read-modify-write and keeps its VTCM round trip (copied in before the
// generic, copied back out after), while the identity-mapped input is consumed
// straight from DDR.
// CHECK-LABEL: func.func @rmw_out_kept(
// CHECK-SAME: %[[A:.+]]: memref<64x128xf32>, %[[B:.+]]: memref<64xf32>
// The read-modify-write out keeps its VTCM buffer.
// CHECK: %[[ALLOC_B:.*]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32, 1>
// CHECK: scf.for %[[I:.+]] = %c0 to %c64 step %c32 {
// CHECK: %[[VA:.*]] = memref.subview %[[A]][%[[I]], 0] [32, 128] [1, 1] : memref<64x128xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>
// CHECK: %[[VB:.*]] = memref.subview %[[B]][%[[I]]] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1], offset: ?>
// CHECK: memref.copy %[[VB]], %[[ALLOC_B]] : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32, 1>
// CHECK: linalg.generic {{.*}} ins(%[[VA]] : memref<32x128xf32, strided<[128, 1], offset: ?>>) outs(%[[ALLOC_B]] : memref<32xf32, 1>) {
// CHECK: memref.copy %[[ALLOC_B]], %[[VB]] : memref<32xf32, 1> to memref<32xf32, strided<[1], offset: ?>>
// CHECK: memref.dealloc %[[ALLOC_B]] : memref<32xf32, 1>
