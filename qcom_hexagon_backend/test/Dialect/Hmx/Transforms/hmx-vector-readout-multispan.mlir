//===- hmx-vector-readout-multispan.mlir - one descriptor per span ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The vector read-out moves each span of a grid-tiled matmul's read-out onto a
// second thread. Every span writes a different window of the output, so each
// span's batch descriptor must carry ITS OWN destination address -- the
// aligned pointer of its span view plus the view's offset, element-scaled, the
// same rule `asAddress` applies on the LLVM side (HmxToLLVMPass.cpp).
//
// WHY THIS FILE EXISTS. On the device (2048^3-B: 2048^3, BM=BN=512, a 4x4 grid
// of spans) the descriptor's `dst` word was published as the bare aligned
// pointer, which is the SAME value for every span: all 16 spans wrote the
// top-left 512x512 block, the other 15/16 of C kept the caller's zeros, and
// rel = 0.96817 where sqrt(15/16) = 0.96825 -- the exact signature of a missing
// offset. The single-span fixtures (hmx-vector-readout.mlir, c0's 1x8192x8192)
// have m0 = n0 = 0, so their offset term is zero and the bug is invisible there,
// which is how the gates stayed green.
//
// The fixture is the same matmul over a 2x2 grid of spans of one 2048x2048
// output: four identical tiles, four views at four different offsets
// (m0*2048 + n0 for (0,0), (0,1), (1,0), (1,1)). What the arm checks, in order
// of how much it would hurt to get wrong:
//
//   PER-SPAN WORD  each span's descriptor word is built from ITS OWN view and
//                  ITS OWN offset: the extract is anchored on that span's
//                  memref value (captured from that span's own
//                  `reinterpret_cast`, so a shared chain cannot satisfy two
//                  arms), and the word is the aligned pointer PLUS the offset
//                  term. The `arith.muli <offset>, 2` is the regression: the
//                  bug had no such arithmetic at all, every word was the bare
//                  pointer, and all four spans wrote the top-left block.
//   DIFFERENT      span (0,0)'s word and span (1,1)'s word start from two
//     WORDS        different views at offsets 0 and 2560 elements, so the words
//                  differ; the production evidence is a 2048^3 llir in which
//                  the same word is computed from the m0/n0 span induction
//                  variables.
//   GROUPINGS      one `memref<6xi32>` and one configure per span, eight
//                  publish sites, four outlined read-outs: one group per span,
//                  and the runtime's drain-before-swap is what makes the
//                  sequence sound.
//
// A single-span compile is not what this file pins; hmx-vector-readout.mlir
// keeps the single-span arms, and this file is the multi-span one that was
// missing entirely.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' | FileCheck %s --check-prefix=BATCHED
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' | FileCheck %s --check-prefix=GROUPS
//===----------------------------------------------------------------------===//

// 32 AR rows per span, so each tile loop runs 0..31 at depth 2 (rows 0..30 in
// the loop, row 31 peeled), and each span gets its own AR array, its own
// descriptor and its own outlined read-out -- one matmul group per span.
func.func @kernel(%bias: memref<256xi8, 1>, %out: memref<2048x2048xf16, strided<[2048, 1]>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c31 = arith.constant 31 : index
  %c512 = arith.constant 512 : index
  %c2048 = arith.constant 2048 : index
  %c2560 = arith.constant 2560 : index
  // The 2x2 grid: m0*N + n0 with N = 2048, so the four views differ.
  %span_00 = memref.reinterpret_cast %out to offset: [%c0], sizes: [512, 512], strides: [2048, 1] : memref<2048x2048xf16, strided<[2048, 1]>> to memref<512x512xf16, strided<[2048, 1], offset: ?>>
  %span_01 = memref.reinterpret_cast %out to offset: [%c512], sizes: [512, 512], strides: [2048, 1] : memref<2048x2048xf16, strided<[2048, 1]>> to memref<512x512xf16, strided<[2048, 1], offset: ?>>
  %span_10 = memref.reinterpret_cast %out to offset: [%c2048], sizes: [512, 512], strides: [2048, 1] : memref<2048x2048xf16, strided<[2048, 1]>> to memref<512x512xf16, strided<[2048, 1], offset: ?>>
  %span_11 = memref.reinterpret_cast %out to offset: [%c2560], sizes: [512, 512], strides: [2048, 1] : memref<2048x2048xf16, strided<[2048, 1]>> to memref<512x512xf16, strided<[2048, 1], offset: ?>>
  %ar_00 = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
  %ar_01 = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
  %ar_10 = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
  %ar_11 = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
  // Span (0,0): offset zero, the case the single-span fixtures exercise. The
  // same chain as every other span -- shape-generic, not specialised.
  scf.for %m = %c0 to %c31 step %c1 {
    %row = arith.addi %m, %c0 : index
    scf.for %n = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar_00, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar_00, %row, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%span_00 : memref<512x512xf16, strided<[2048, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<512x512xf16, strided<[2048, 1], offset: ?>>
  }
  hmx.acc_read %bias, %ar_00, %c31, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
  hmx.unpack_acc ins(%ar_00, %c31, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%span_00 : memref<512x512xf16, strided<[2048, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<512x512xf16, strided<[2048, 1], offset: ?>>
  // Span (0,1): offset 512.
  scf.for %m = %c0 to %c31 step %c1 {
    %row = arith.addi %m, %c0 : index
    scf.for %n = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar_01, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar_01, %row, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%span_01 : memref<512x512xf16, strided<[2048, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<512x512xf16, strided<[2048, 1], offset: ?>>
  }
  hmx.acc_read %bias, %ar_01, %c31, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
  hmx.unpack_acc ins(%ar_01, %c31, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%span_01 : memref<512x512xf16, strided<[2048, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<512x512xf16, strided<[2048, 1], offset: ?>>
  // Span (1,0): offset 2048.
  scf.for %m = %c0 to %c31 step %c1 {
    %row = arith.addi %m, %c0 : index
    scf.for %n = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar_10, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar_10, %row, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%span_10 : memref<512x512xf16, strided<[2048, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<512x512xf16, strided<[2048, 1], offset: ?>>
  }
  hmx.acc_read %bias, %ar_10, %c31, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
  hmx.unpack_acc ins(%ar_10, %c31, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%span_10 : memref<512x512xf16, strided<[2048, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<512x512xf16, strided<[2048, 1], offset: ?>>
  // Span (1,1): offset 2560, the corner the device never wrote.
  scf.for %m = %c0 to %c31 step %c1 {
    %row = arith.addi %m, %c0 : index
    scf.for %n = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar_11, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar_11, %row, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%span_11 : memref<512x512xf16, strided<[2048, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<512x512xf16, strided<[2048, 1], offset: ?>>
  }
  hmx.acc_read %bias, %ar_11, %c31, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
  hmx.unpack_acc ins(%ar_11, %c31, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%span_11 : memref<512x512xf16, strided<[2048, 1], offset: ?>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<512x512xf16, strided<[2048, 1], offset: ?>>
  memref.dealloc %ar_00 : memref<32x16x16x32x2xf16, 1>
  memref.dealloc %ar_01 : memref<32x16x16x32x2xf16, 1>
  memref.dealloc %ar_10 : memref<32x16x16x32x2xf16, 1>
  memref.dealloc %ar_11 : memref<32x16x16x32x2xf16, 1>
  return
}

// span (0,0) and span (1,1) are DIFFERENT memref values with different
// offsets. Captured here so each span's word below is anchored on its own
// view: a chain shared by two spans cannot satisfy both arms.
// BATCHED: %[[VIEW_00:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%c0], sizes: [512, 512], strides: [2048, 1] : memref<2048x2048xf16, strided<[2048, 1]>> to memref<512x512xf16, strided<[2048, 1], offset: ?>>
// BATCHED: %[[VIEW_11:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%c2560], sizes: [512, 512], strides: [2048, 1] : memref<2048x2048xf16, strided<[2048, 1]>> to memref<512x512xf16, strided<[2048, 1], offset: ?>>

// Span (0,0)'s descriptor word: its own view's offset, element-scaled and added
// to the aligned pointer of that view's base buffer. Every link in the chain
// references the value captured for span (0,0) and for no other span. The
// `arith.muli <offset>, 2` is the regression: the bug had no such arithmetic at
// all, every word was the bare aligned pointer.
// BATCHED: %[[PTR_00:.*]] = memref.extract_aligned_pointer_as_index %[[VIEW_00]] : memref<512x512xf16, strided<[2048, 1], offset: ?>> -> index
// BATCHED: %[[BB_00:[^,]*]], %[[OFF_00:[^,]*]], %{{.*}} = memref.extract_strided_metadata %[[VIEW_00]] : memref<512x512xf16, strided<[2048, 1], offset: ?>>
// BATCHED: %[[SCALED_00:.*]] = arith.muli %[[OFF_00]], %{{.*}} : index
// BATCHED: %[[SUM_00:.*]] = arith.addi %[[PTR_00]], %[[SCALED_00]] : index
// BATCHED: %[[WORD_00:.*]] = arith.index_cast %[[SUM_00]] : index to i32
// BATCHED: memref.store %[[WORD_00]], %[[DESC_00:.*]][%{{.*}}] : memref<6xi32>

// Span (1,1): its own view at a different offset, and its own descriptor
// receiving the word -- two spans writing the same window of C is exactly what
// the device saw wrong.
// BATCHED: %[[PTR_11:.*]] = memref.extract_aligned_pointer_as_index %[[VIEW_11]] : memref<512x512xf16, strided<[2048, 1], offset: ?>> -> index
// BATCHED: %[[BB_11:[^,]*]], %[[OFF_11:[^,]*]], %{{.*}} = memref.extract_strided_metadata %[[VIEW_11]] : memref<512x512xf16, strided<[2048, 1], offset: ?>>
// BATCHED: %[[SCALED_11:.*]] = arith.muli %[[OFF_11]], %{{.*}} : index
// BATCHED: %[[SUM_11:.*]] = arith.addi %[[PTR_11]], %[[SCALED_11]] : index
// BATCHED: %[[WORD_11:.*]] = arith.index_cast %[[SUM_11]] : index to i32
// BATCHED: memref.store %[[WORD_11]], %[[DESC_11:.*]][%{{.*}}] : memref<6xi32>

// One group per span: four descriptors, four configure placeholders, eight
// publish sites (each span's in-loop group and tail batch) and four drains.
// Matched as one order-independent group because a span's descriptor, publishes
// and drain are spread across the kernel while its read-out is outlined after
// all of them. The `func.` qualifier is spelled by the printer for some calls
// and dropped for others, so both spellings are matched -- the same reason
// hmx-vector-readout.mlir does.
// GROUPS-DAG: memref.alloca() : memref<6xi32>
// GROUPS-DAG: memref.alloca() : memref<6xi32>
// GROUPS-DAG: memref.alloca() : memref<6xi32>
// GROUPS-DAG: memref.alloca() : memref<6xi32>
// GROUPS-DAG: memref.extract_strided_metadata %{{.*}} : memref<32x16x16x32x2xf16, 1> -> memref<f16, 1>, index, index, index, index, index, index, index, index, index, index, index
// GROUPS-DAG: memref.extract_strided_metadata %{{.*}} : memref<32x16x16x32x2xf16, 1> -> memref<f16, 1>, index, index, index, index, index, index, index, index, index, index, index
// GROUPS-DAG: memref.extract_strided_metadata %{{.*}} : memref<32x16x16x32x2xf16, 1> -> memref<f16, 1>, index, index, index, index, index, index, index, index, index, index, index
// GROUPS-DAG: memref.extract_strided_metadata %{{.*}} : memref<32x16x16x32x2xf16, 1> -> memref<f16, 1>, index, index, index, index, index, index, index, index, index, index, index
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_configure(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_configure(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_configure(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_configure(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()
// GROUPS-DAG: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()

// Four specialisations of the read-out, one per span: same shape, different
// arrays.
// GROUPS-DAG: func.func private @__hmx_readout(%{{.*}}: memref<32x16x16x32x2xf16, 1>, %{{.*}}: memref<512x512xf16, strided<[2048, 1], offset: ?>>
// GROUPS-DAG: func.func private @__hmx_readout_1(%{{.*}}: memref<32x16x16x32x2xf16, 1>, %{{.*}}: memref<512x512xf16, strided<[2048, 1], offset: ?>>
// GROUPS-DAG: func.func private @__hmx_readout_2(%{{.*}}: memref<32x16x16x32x2xf16, 1>, %{{.*}}: memref<512x512xf16, strided<[2048, 1], offset: ?>>
// GROUPS-DAG: func.func private @__hmx_readout_3(%{{.*}}: memref<32x16x16x32x2xf16, 1>, %{{.*}}: memref<512x512xf16, strided<[2048, 1], offset: ?>>