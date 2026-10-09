// RUN: linalg-hexagon-opt %s -dma-to-llvm | FileCheck %s
//
// Status must live in its own word, never in the tag word that holds the
// token (roadmap/ARCH-REVIEW.md bug #2). The runtime reports a refused
// start by writing `*status = DMAFailure` through the call's last pointer
// argument; this pass stores the returned token into the *tag* for
// `dma_wait` to poll. When both pointed at the tag, that store erased the
// failure report one instruction after it was written, so a refused
// transfer was reported nowhere (the runtime papered over it with a forged
// completed descriptor -- now replaced by `DMA_TOKEN_NONE`, see
// bin/runtime/include/RuntimeDMA.h).
//
// Pinned here, both ABI shapes: the call's status argument is a dedicated
// entry-block `llvm.alloca`, the token is stored to the tag GEP, and no
// store ever writes the token into the status slot.

// ---- 1D start: 8-argument hexagon_runtime_dma_start ----

// CHECK-LABEL: func.func @dma_start_1d(
// The slot is at the top of the function, before any body op: an alloca
// emitted where the dma_start sits would be inside the double-buffering
// loop and lower to a dynamic stack allocation (REGRESSION-HOIST-NOT).
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[STATUS:.*]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// The tag is the i32 GEP (the data GEPs are i8 here), bound before the
// call so the two words are tracked by name from here on.
// CHECK: %[[TAG:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// The call's LAST argument -- the runtime's DMAStatus* out-param -- is the
// status slot, not the tag.
// CHECK: %[[TOK:.*]] = llvm.call @hexagon_runtime_dma_start({{.*}}%[[STATUS]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// The token is stored to the tag word, and to the status slot nowhere --
// neither between the call and that store...
// CHECK-NOT: llvm.store %[[TOK]], %[[STATUS]]
// CHECK: llvm.store %[[TOK]], %[[TAG]] : i32, !llvm.ptr
// ...nor after it.
// CHECK-NOT: llvm.store %[[TOK]], %[[STATUS]]
// CHECK: llvm.call @hexagon_runtime_dma_wait
// CHECK-NOT: llvm.store %[[TOK]], %[[STATUS]]
func.func @dma_start_1d(%arg0: memref<1024x1024xi8, strided<[1024, 1]>>, %arg1: memref<1024x1024xi8, strided<[1024, 1]>>) -> () {
  %tag = memref.alloc() : memref<1xi32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %c1M = arith.constant 1048576 : index
  memref.dma_start %arg0[%c0, %c0], %arg1[%c0, %c0], %c1M, %tag[%c0] : memref<1024x1024xi8, strided<[1024, 1]>>, memref<1024x1024xi8, strided<[1024, 1]>>, memref<1xi32, strided<[1]>>
  memref.dma_wait %tag[%c0], %c1M : memref<1xi32, strided<[1]>>
  return
}

// ---- 2D start: 13-argument hexagon_runtime_dma2d_start ----

// CHECK-LABEL: func.func @dma_start_2d(
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[STATUS:.*]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// CHECK: %[[TAG:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: %[[TOK:.*]] = llvm.call @hexagon_runtime_dma2d_start({{.*}}%[[STATUS]]) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NOT: llvm.store %[[TOK]], %[[STATUS]]
// CHECK: llvm.store %[[TOK]], %[[TAG]] : i32, !llvm.ptr
// CHECK-NOT: llvm.store %[[TOK]], %[[STATUS]]
// CHECK: llvm.call @hexagon_runtime_dma_wait
// CHECK-NOT: llvm.store %[[TOK]], %[[STATUS]]
func.func @dma_start_2d(%arg0: memref<1024x1024xi8, strided<[2048, 1]>>, %arg1: memref<1024x1024xi8, strided<[1024, 1]>, 1>) -> () {
  %tag = memref.alloc() : memref<1xi32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %number_elements = arith.constant 1048576 : index
  %cStride = arith.constant 2048 : index
  %cWidth = arith.constant 1024 : index
  memref.dma_start %arg0[%c0, %c0], %arg1[%c0, %c0], %number_elements, %tag[%c0], %cStride, %cWidth : memref<1024x1024xi8, strided<[2048, 1]>>, memref<1024x1024xi8, strided<[1024, 1]>, 1>, memref<1xi32, strided<[1]>>
  memref.dma_wait %tag[%c0], %number_elements : memref<1xi32, strided<[1]>>
  return
}
