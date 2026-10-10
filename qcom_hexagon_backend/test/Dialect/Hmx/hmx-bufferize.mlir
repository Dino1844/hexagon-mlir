//===- hmx-bufferize.mlir - hmx.matmul goes through bufferization --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// hmx.matmul is a destination-style op, so one-shot bufferization turns it into
// the memref form that the tile-level partitioning pass needs. Without the
// BufferizableOpInterface it would simply not bufferize.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries})' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @matmul_buffers
// CHECK: hmx.matmul ins(
// CHECK-SAME: memref<2x4x16x32x2xf16,
// CHECK-SAME: memref<2x4x16x32x2xf16,
// CHECK-SAME: outs(
// CHECK-SAME: memref<2x2x16x32x2xf16,
// CHECK-SAME: hmx.decision_id = 7 : i64
// CHECK-NOT: tensor<
func.func @matmul_buffers(%a: tensor<2x4x16x32x2xf16>,
                          %b: tensor<2x4x16x32x2xf16>,
                          %c: tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16> {
  %0 = hmx.matmul ins(%a, %b : tensor<2x4x16x32x2xf16>, tensor<2x4x16x32x2xf16>)
                  outs(%c : tensor<2x2x16x32x2xf16>) {hmx.decision_id = 7 : i64} -> tensor<2x2x16x32x2xf16>
  return %0 : tensor<2x2x16x32x2xf16>
}

// -----

// The fused tail bufferizes the way the bridge emits it: the read-out array
// lives in VTCM (memory space 1, like every crouton the bridge stages) and the
// loop-carried f32 destination becomes a memref the leaf writes in place, with
// no tensor left.
// CHECK-LABEL: func.func @fused_tail_buffers
// CHECK: hmx.unpack_acc_f32 ins(
// CHECK-SAME: memref<2x2x16x32x2xf16, 1>) outs(
// CHECK-SAME: memref<64x64xf32>)
// CHECK-SAME: hmx.decision_id = 7 : i64
// CHECK-NOT: tensor<
func.func @fused_tail_buffers(%a: tensor<2x2x16x32x2xf16>,
                              %b: tensor<2x2x16x32x2xf16>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %crout = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<2x2x16x32x2xf16>
  %mm = hmx.matmul ins(%a, %b : tensor<2x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>)
                   outs(%crout : tensor<2x2x16x32x2xf16>) {hmx.decision_id = 7 : i64} -> tensor<2x2x16x32x2xf16>
  %e = tensor.empty() : tensor<64x64xf32>
  %0 = scf.for %i = %c0 to %c32 step %c1 iter_args(%d = %e) -> (tensor<64x64xf32>) {
    %row = arith.divui %i, %c16 : index
    %col = arith.remui %i, %c16 : index
    %u = hmx.unpack_acc_f32 ins(%mm, %row, %col : tensor<2x2x16x32x2xf16>)
                            outs(%d : tensor<64x64xf32>) {hmx.decision_id = 7 : i64} -> tensor<64x64xf32>
    scf.yield %u : tensor<64x64xf32>
  }
  return %0 : tensor<64x64xf32>
}

// -----

// Same, with the residual threaded: the C term bufferizes alongside and the
// memref op carries it, so the lowering can pass its address to the leaf. The
// residual itself is an ordinary f32 buffer and carries no space requirement,
// but it must bufferize to exactly the destination type (a strided view would
// neither verify nor carry the static stride the leaf assumes) -- which is why
// the wiring only threads dense-internal residuals and leaves the rest on the
// old epilogue.
// CHECK-LABEL: func.func @fused_tail_residual_buffers
// CHECK: hmx.unpack_acc_f32 ins(
// CHECK-SAME: memref<2x2x16x32x2xf16, 1>, memref<64x64xf32>) outs(
// CHECK-SAME: memref<64x64xf32>)
// CHECK-SAME: hmx.decision_id = 7 : i64
// CHECK-NOT: tensor<
func.func @fused_tail_residual_buffers(%a: tensor<2x2x16x32x2xf16>,
                                       %b: tensor<2x2x16x32x2xf16>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %crout = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<2x2x16x32x2xf16>
  %mm = hmx.matmul ins(%a, %b : tensor<2x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>)
                   outs(%crout : tensor<2x2x16x32x2xf16>) {hmx.decision_id = 7 : i64} -> tensor<2x2x16x32x2xf16>
  %res = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
  %e = tensor.empty() : tensor<64x64xf32>
  %0 = scf.for %i = %c0 to %c32 step %c1 iter_args(%d = %e) -> (tensor<64x64xf32>) {
    %row = arith.divui %i, %c16 : index
    %col = arith.remui %i, %c16 : index
    %u = hmx.unpack_acc_f32 ins(%mm, %row, %col, %res : tensor<2x2x16x32x2xf16>, tensor<64x64xf32>)
                            outs(%d : tensor<64x64xf32>) {hmx.decision_id = 7 : i64} -> tensor<64x64xf32>
    scf.yield %u : tensor<64x64xf32>
  }
  return %0 : tensor<64x64xf32>
}

// -----

// The residual threading the K-loop accumulator uses: the destination IS the
// residual -- one VTCM buffer carried across a K loop's segments, read and
// overwritten in place by every leaf call. The op declares the pair
// non-conflicting (`hmx__unpack_f32_tile` reads element (i, j) and writes
// element (i, j) of the same matrix in one call), so bufferization keeps the
// destination in place: no `memref.copy` of the accumulator appears, which is
// the whole point of the form (docs/hmx/kchunk-accumulator-residency-2026-10-10.md
// section 4.1). The post-loop consumer is a store, as it is in every real
// kernel (a VTCM matrix is never a function result).
// CHECK-LABEL: func.func @fused_tail_inplace_buffers
// CHECK-DAG: memref.alloc() {{.*}} : memref<64x64xf32, 1>
// The K loop carries that one allocation.
// CHECK-DAG: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{[a-z0-9_]+}} = %{{[a-z0-9_]+}}) -> (memref<64x64xf32, 1>)
// The read-out loop carries the same buffer and the leaf's residual IS its
// destination: one allocation, read and overwritten in place.
// CHECK-DAG: hmx.unpack_acc_f32 ins(%{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[arg:[a-z0-9_]+]] : memref<2x2x16x32x2xf16, 1>, memref<64x64xf32, 1>) outs(%[[arg]] : memref<64x64xf32, 1>)
// No copy of the f32 accumulator anywhere: not at the entry, not per leaf
// call. The one copy this kernel has is the narrowing store to the output,
// and it is the f16 image's, not the accumulator's.
// CHECK-NOT: memref.copy %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : memref<64x64xf32
// CHECK-DAG: memref.copy %{{[a-z0-9_]+}}, %arg2 : memref<64x64xf16> to memref<64x64xf16>
func.func @fused_tail_inplace_buffers(%a: tensor<2x2x16x32x2xf16>,
                                      %b: tensor<2x2x16x32x2xf16>,
                                      %dstbuf: memref<64x64xf16>) {
  %cst = arith.constant 0.000000e+00 : f32
  %acc = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<64x64xf32>
  %acc0 = linalg.fill ins(%cst : f32) outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = scf.for %k = %c0 to %c2 step %c1 iter_args(%carried = %acc0) -> (tensor<64x64xf32>) {
    %crout = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<2x2x16x32x2xf16>
    %mm = hmx.matmul ins(%a, %b : tensor<2x2x16x32x2xf16>, tensor<2x2x16x32x2xf16>)
                    outs(%crout : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
    %u = scf.for %i = %c0 to %c2 step %c1 iter_args(%d = %carried) -> (tensor<64x64xf32>) {
      %x = hmx.unpack_acc_f32 ins(%mm, %i, %c0, %d : tensor<2x2x16x32x2xf16>, tensor<64x64xf32>)
                              outs(%d : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %x : tensor<64x64xf32>
    }
    scf.yield %u : tensor<64x64xf32>
  }
  %oe = tensor.empty() : tensor<64x64xf16>
  %g = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<64x64xf32>) outs(%oe : tensor<64x64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %n = arith.truncf %in : f32 to f16
    linalg.yield %n : f16
  } -> tensor<64x64xf16>
  bufferization.materialize_in_destination %g in writable %dstbuf : (tensor<64x64xf16>, memref<64x64xf16>) -> ()
  return
}

// -----

// The bulk `count` is an op attribute, so the interface rebuilds the memref op
// with it: the ranged range survives bufferization and the lowering can still
// select the ranged leaf.
// CHECK-LABEL: func.func @ranged_buffers
// CHECK: hmx.pack_act {{.*}} {count = 2 : i64, hmx.decision_id = 8 : i64}
// CHECK: hmx.unpack_acc {{.*}} {count = 16 : i64, hmx.decision_id = 8 : i64}
func.func @ranged_buffers(%src: tensor<64x64xf16>,
                          %act: tensor<2x2x16x32x2xf16>,
                          %dst: tensor<64x32xf16>) -> tensor<64x32xf16> {
  %c0 = arith.constant 0 : index
  %p = hmx.pack_act ins(%src, %c0, %c0 : tensor<64x64xf16>)
      outs(%act : tensor<2x2x16x32x2xf16>) {count = 2 : i64, hmx.decision_id = 8 : i64} -> tensor<2x2x16x32x2xf16>
  %u = hmx.unpack_acc ins(%p, %c0, %c0 : tensor<2x2x16x32x2xf16>)
      outs(%dst : tensor<64x32xf16>) {count = 16 : i64, hmx.decision_id = 8 : i64} -> tensor<64x32xf16>
  return %u : tensor<64x32xf16>
}

// -----

// The weight bridge uses the same explicit hand-off as the activation and
// read-out bridges. This pins the PackWeightOp external model as well.
// CHECK-LABEL: func.func @weight_bridge_buffers
// CHECK: hmx.pack_weight {{.*}} {hmx.decision_id = 9 : i64}
func.func @weight_bridge_buffers(%src: tensor<64x64xf16>,
                                  %dst: tensor<2x2x16x32x2xf16>)
    -> tensor<2x2x16x32x2xf16> {
  %c0 = arith.constant 0 : index
  %p = hmx.pack_weight ins(%src, %c0, %c0 : tensor<64x64xf16>)
      outs(%dst : tensor<2x2x16x32x2xf16>) {hmx.decision_id = 9 : i64}
      -> tensor<2x2x16x32x2xf16>
  return %p : tensor<2x2x16x32x2xf16>
}

// -----

// The memref-side carrier of the crouton layout: a rank-5 memref tagged with
// the identity-map `#hmx.crouton_memref_layout`. Once the pipeline can inject
// its crouton-aware type converter into bufferization (the seam
// hmx-interface-gaps.md section 2.2 leaves open -- the pinned one-shot
// bufferize pass cannot take a custom converter, and `alloc_tensor` infers its
// own buffer type), this is the layout an encoded tensor bufferizes to. The
// CLI one-shot-bufferize used here has no hook for it, so the input carries
// the layout directly; the case pins the carrier itself -- it parses, it
// verifies (`verifyLayout` agrees with the grid, the memref-form ops accept
// it) and it passes through bufferization untouched.
// CHECK-LABEL: func.func @memref_layout_carried
// CHECK: hmx.matmul ins(%{{.*}}, %{{.*}} : memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>, memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>) outs(%{{.*}} : memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>)
// CHECK: hmx.mma %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {n_croutons = 1 : i32} : memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>, memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>
// CHECK-NOT: tensor<
func.func @memref_layout_carried(
    %a: memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>,
    %b: memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>,
    %c: memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>) {
  %c0 = arith.constant 0 : index
  hmx.matmul ins(%a, %b : memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>,
                              memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>)
              outs(%c : memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>)
  hmx.mma %a, %b, %c0, %c0, %c0 {n_croutons = 1 : i32}
      : memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>,
        memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>
  return
}

// -----

// The seam closed: `hmx.alloc_crouton` is the dialect's own allocation, and its
// BufferizableOpInterface maps the `#hmx.crouton` encoding on its result to the
// identity-map `#hmx.crouton_memref_layout` -- with the VTCM space the op pins
// -- so the allocated crouton array is typed as a laid-out space-1 memref. The
// stock `bufferization.alloc_tensor` could never produce this (it drops the
// encoding and emits fully dynamic strides; see
// hmx-bufferize-encoding-rejects.mlir). The `hmx.matmul` reading and writing
// the buffer through its destination keeps the same laid-out type, so the
// layout reaches every op on the buffer's SSA chain.
// CHECK-LABEL: func.func @crouton_alloc_keeps_layout
// CHECK: memref.alloc
// CHECK-SAME: memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>
// CHECK: hmx.matmul
// CHECK-NOT: tensor<
func.func @crouton_alloc_keeps_layout(
    %a: tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>,
    %b: tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>) {
  %0 = hmx.alloc_crouton -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
  hmx.matmul ins(%a, %b : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>,
                          tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>)
             outs(%0 : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>)
      -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
  return
}
