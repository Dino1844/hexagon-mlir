// RUN: linalg-hexagon-opt %s -dma-to-llvm | FileCheck %s
//
// The two `bypassCache` descriptor words are taken PER ENDPOINT, never shared.
// A descriptor's "snoop and invalidate" is redundant on an endpoint that is not
// cacheable memory at all, and must be kept on a DDR endpoint, where skipping
// it breaks coherence (see the header comment in
// bin/runtime/UserDMA/RuntimeDMA.cc). The reference implementation for this
// project -- llama.cpp's ggml-hexagon htp/dma-queue.h -- uses exactly this
// rule: `desc->src_bypass = dma_is_vtcm(ptr) ? 1 : q->nocache`.
//
// So a VTCM endpoint gets 1 and a DDR endpoint 0, in both directions. Both
// directions are pinned because a predicate that only ever produced 0 (the old
// behaviour, which is also what a broken `isVTCM` produces) satisfies either
// one alone.

// ---- DDR -> VTCM: the direction the HMX staging channel uses ----

// CHECK-LABEL: func.func @ddr_to_vtcm_2d(
// The source is DDR (space 0), the slot is VTCM (space 1).
// CHECK: %[[SRC_SPACE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[DST_SPACE:.*]] = llvm.mlir.constant(1 : i32)
// bypassCacheSrc = 0 (DDR), bypassCacheDst = 1 (VTCM): the snoop-and-invalidate
// is redundant only on the endpoint that is not cacheable memory at all.
// CHECK: %[[BYP_SRC:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[BYP_DST:.*]] = llvm.mlir.constant(1 : i32)
// isOrdered and the cache allocation policy are unchanged 0s.
// CHECK: %[[ORDERED:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[CAP:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @hexagon_runtime_dma2d_start({{.*}}, %[[SRC_SPACE]], {{.*}}, %[[DST_SPACE]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[BYP_SRC]], %[[BYP_DST]], %[[ORDERED]], %[[CAP]], %{{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

func.func @ddr_to_vtcm_2d(%arg0: memref<1024x1024xi8, strided<[2048, 1]>>,
                          %arg1: memref<1024x1024xi8, strided<[1024, 1]>, 1>) -> () {
  %tag = memref.alloc() : memref<1xi32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %n = arith.constant 1048576 : index
  %stride = arith.constant 2048 : index
  %width = arith.constant 1024 : index
  memref.dma_start %arg0[%c0, %c0], %arg1[%c0, %c0], %n, %tag[%c0], %stride, %width
      : memref<1024x1024xi8, strided<[2048, 1]>>, memref<1024x1024xi8, strided<[1024, 1]>, 1>, memref<1xi32, strided<[1]>>
  memref.dma_wait %tag[%c0], %n : memref<1xi32, strided<[1]>>
  return
}

// -----

// ---- VTCM -> DDR: the read-out direction ----

// CHECK-LABEL: func.func @vtcm_to_ddr_2d(
// The 1-byte element size is the other `1` in this function, so bind it first
// and take the spaces from what follows it.
// CHECK: %[[ESZ:.*]] = llvm.mlir.constant(1 : i32)
// The source is VTCM, the destination is DDR.
// CHECK: %[[SRC_SPACE:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[DST_SPACE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[BYP_SRC:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[BYP_DST:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[ORDERED:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[CAP:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @hexagon_runtime_dma2d_start({{.*}}, %[[SRC_SPACE]], {{.*}}, %[[DST_SPACE]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[BYP_SRC]], %[[BYP_DST]], %[[ORDERED]], %[[CAP]], %{{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

func.func @vtcm_to_ddr_2d(%arg0: memref<1024x1024xi8, strided<[2048, 1]>, 1>,
                          %arg1: memref<1024x1024xi8, strided<[1024, 1]>>) -> () {
  %tag = memref.alloc() : memref<1xi32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %n = arith.constant 1048576 : index
  %stride = arith.constant 2048 : index
  %width = arith.constant 1024 : index
  memref.dma_start %arg0[%c0, %c0], %arg1[%c0, %c0], %n, %tag[%c0], %stride, %width
      : memref<1024x1024xi8, strided<[2048, 1]>, 1>, memref<1024x1024xi8, strided<[1024, 1]>>, memref<1xi32, strided<[1]>>
  memref.dma_wait %tag[%c0], %n : memref<1xi32, strided<[1]>>
  return
}

// -----

// ---- DDR -> DDR, the 1D (non-strided) path: both endpoints stay 0 ----

// CHECK-LABEL: func.func @ddr_to_ddr_1d(
// CHECK: %[[SRC_SPACE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[DST_SPACE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[BYP_SRC:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[BYP_DST:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @hexagon_runtime_dma_start({{.*}}, %[[SRC_SPACE]], {{.*}}, %[[DST_SPACE]], {{.*}}, %[[BYP_SRC]], %[[BYP_DST]], %{{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32

func.func @ddr_to_ddr_1d(%arg0: memref<1024x1024xi8, strided<[1024, 1]>>,
                         %arg1: memref<1024x1024xi8, strided<[1024, 1]>>) -> () {
  %tag = memref.alloc() : memref<1xi32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %n = arith.constant 1048576 : index
  memref.dma_start %arg0[%c0, %c0], %arg1[%c0, %c0], %n, %tag[%c0]
      : memref<1024x1024xi8, strided<[1024, 1]>>, memref<1024x1024xi8, strided<[1024, 1]>>, memref<1xi32, strided<[1]>>
  memref.dma_wait %tag[%c0], %n : memref<1xi32, strided<[1]>>
  return
}
