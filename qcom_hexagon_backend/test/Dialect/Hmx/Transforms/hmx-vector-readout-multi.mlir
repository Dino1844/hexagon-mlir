//===- hmx-vector-readout-multi.mlir - one group per matmul, in sequence ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A multi-matmul function -- flash attention's Q@K and P@V -- is no longer
// declined whole. Each matmul's read-out names its own AR array, destination
// and decision id; the pass takes them as one GROUP per matmul, in program
// order, and each group gets its own descriptor, outlined read-out, handoff
// record and `configure` call. What makes the sequence sound is a runtime
// fact, not a compiler one: `configure` drains the ring before swapping the
// function pointer (HmxVectorExecutor.cpp configure), so the first group's
// in-flight batches finish under their own read-out before the second group's
// batches can run at all.
//
// The arms:
//
//   MULTI    two matmuls, different shapes, different decision ids: two
//            descriptors, two configure placeholders, two outlined read-outs
//            (each carrying ITS matmul's decision id), two handoff records.
//            The different ids are the point: the old pass declined the whole
//            function on exactly that disagreement.
//   NESTED   one group's tile loop inside another's: declined with a remark,
//            because the inner group's configure would fire between the outer
//            group's publishes. The read-outs stay where they were.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' | FileCheck %s --check-prefix=MULTI
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' 2>&1 | FileCheck %s --check-prefix=NESTED
//===----------------------------------------------------------------------===//

// Both handoff records, in group order -- the order the lowering pairs with
// the engine's configure calls. The records carry the two DIFFERENT decision
// ids, which is the disagreement the old pass declined the function on.
// MULTI: hmx.readout.handoffs = [
// MULTI: engine = "kernel", work = "__hmx_readout"{{.*}}engine = "kernel", work = "__hmx_readout_1"

// Scoped to @kernel. One descriptor and one configure placeholder per GROUP,
// each before its own group's tile loop; the read-outs are gone from the
// kernel (the NOTs span the loop bodies and the tail). The closing label ends
// the region before @nested, which is declined and keeps its read-outs.
// MULTI-LABEL: func.func @kernel(
// MULTI-NOT: hmx.unpack_acc
// MULTI: %[[AR1:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
// MULTI: %[[D1:.*]] = memref.alloca() : memref<6xi32>
// MULTI: memref.store {{.*}}, %[[D1]][%{{.*}}] : memref<6xi32>
// MULTI: memref.store {{.*}}, %[[D1]][%{{.*}}] : memref<6xi32>
// MULTI: memref.store {{.*}}, %[[D1]][%{{.*}}] : memref<6xi32>
// MULTI: memref.store {{.*}}, %[[D1]][%{{.*}}] : memref<6xi32>
// The first group's configure. The 0 is the placeholder the lowering fills
// with the entry point's address (HmxToLLVMPass wireConfigureCalls).
// MULTI: {{(func\.)?}}call @hexagon_runtime_hmx_exec_configure({{.*}}) : (i32, i32) -> i32
// MULTI: scf.for %{{.*}} to %c31 step
// MULTI-NOT: hmx.unpack_acc
// MULTI: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// MULTI: hmx.acc_read %{{.*}}, %[[AR1]], %c31
// MULTI: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// MULTI: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()
// MULTI: memref.dealloc %[[AR1]]
// The second group: a SECOND descriptor and a SECOND configure, before the
// second matmul's tile loop -- and that position is the correctness barrier,
// not a style choice: the runtime drains the first group's batches inside
// this configure before the second group's work function takes over.
// MULTI: %[[AR2:.*]] = memref.alloc() {alignment = 128 : i64} : memref<16x16x16x32x2xf16, 1>
// MULTI: %[[D2:.*]] = memref.alloca() : memref<6xi32>
// MULTI: memref.store {{.*}}, %[[D2]][%{{.*}}] : memref<6xi32>
// MULTI: memref.store {{.*}}, %[[D2]][%{{.*}}] : memref<6xi32>
// MULTI: memref.store {{.*}}, %[[D2]][%{{.*}}] : memref<6xi32>
// MULTI: memref.store {{.*}}, %[[D2]][%{{.*}}] : memref<6xi32>
// MULTI: {{(func\.)?}}call @hexagon_runtime_hmx_exec_configure({{.*}}) : (i32, i32) -> i32
// MULTI: scf.for %{{.*}} to %c15 step
// MULTI-NOT: hmx.unpack_acc
// MULTI: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// MULTI: hmx.acc_read %{{.*}}, %[[AR2]], %c15
// MULTI: {{(func\.)?}}call @hexagon_runtime_hmx_exec_publish(
// MULTI: {{(func\.)?}}call @hexagon_runtime_hmx_exec_drain()
// MULTI: memref.dealloc %[[AR2]]
// MULTI-NOT: hmx.unpack_acc
// MULTI-LABEL: func.func @nested(

// The two outlined read-outs, each carrying ITS matmul's decision id.
// MULTI-LABEL: func.func private @__hmx_readout(
// MULTI: hmx.unpack_acc ins({{.*}}) outs({{.*}}) {count = 16 : i64, hmx.decision_id = 0 : i64}
// MULTI-LABEL: func.func private @__hmx_readout_1(
// MULTI: hmx.unpack_acc ins({{.*}}) outs({{.*}}) {count = 16 : i64, hmx.decision_id = 1 : i64}

// The nested shape is declined, and loudly: the inner group's configure would
// fire per outer iteration, between the outer group's publishes. Both
// read-outs stay exactly where the input put them, and no executor call
// appears inside the function.
//
// This arm runs with 2>&1 so the remark itself is checked. The remark's
// diagnostic note REPRINTS @nested with SSA names, and a label anchored on
// @nested would lock onto that reprint (a CHECK-LABEL binds to its FIRST
// occurrence, and everything before it is closed off), so the arm steps past
// the whole diagnostic first: `module attributes` exists only in the module
// print, and the executor call after it is @kernel's, so the @nested label
// can only match the module print's @nested.
// NESTED: remark: hmx-vector-readout declined: one matmul group's tile loop contains another's
// NESTED-LABEL: module attributes
// NESTED: {{(func\.)?}}call @hexagon_runtime_hmx_exec_configure(
// NESTED-LABEL: func.func @nested(
// NESTED-NOT: hexagon_runtime_hmx_exec
// NESTED: %[[NA1:.*]] = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
// NESTED: %[[NA2:.*]] = memref.alloc() {alignment = 128 : i64} : memref<16x16x16x32x2xf16, 1>
// NESTED: scf.for %{{.*}} to %c31 step
// NESTED: hmx.unpack_acc ins(%[[NA1]]
// NESTED: scf.for %{{.*}} to %c16 step
// NESTED: hmx.unpack_acc ins(%[[NA2]]
// NESTED: hmx.unpack_acc ins(%[[NA1]], %c31
// NESTED: memref.dealloc %[[NA1]]
// NESTED-NOT: hexagon_runtime_hmx_exec
// NESTED: memref.dealloc %[[NA2]]
// NESTED-NOT: hexagon_runtime_hmx_exec
// NESTED: return

func.func @kernel(%bias: memref<256xi8, 1>, %out1: memref<1024x512xf16, strided<[512, 1]>>, %out2: memref<512x512xf16, strided<[512, 1]>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c31 = arith.constant 31 : index
  %c15 = arith.constant 15 : index
  %c16 = arith.constant 16 : index

  // The first matmul: 32 AR rows, loop 0..31, peeled row 31.
  %ar1 = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
  scf.for %m = %c0 to %c31 step %c1 {
    %rowZero1 = arith.constant 0 : index
    %rowMul1 = arith.muli %c1, %rowZero1 : index
    %row1 = arith.addi %m, %rowMul1 : index
    scf.for %n1 = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar1, %row1, %n1 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar1, %row1, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out1 : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
  }
  hmx.acc_read %bias, %ar1, %c31, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
  hmx.unpack_acc ins(%ar1, %c31, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out1 : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
  memref.dealloc %ar1 : memref<32x16x16x32x2xf16, 1>

  // The second matmul: 16 AR rows, loop 0..15, peeled row 15. A different
  // shape, a different destination and a different decision id -- the three
  // facts the old pass declined the whole function on.
  %ar2 = memref.alloc() {alignment = 128 : i64} : memref<16x16x16x32x2xf16, 1>
  scf.for %m2 = %c0 to %c15 step %c1 {
    %rowZero2 = arith.constant 0 : index
    %rowMul2 = arith.muli %c1, %rowZero2 : index
    %row2 = arith.addi %m2, %rowMul2 : index
    scf.for %n2 = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar2, %row2, %n2 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<16x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar2, %row2, %c0 : memref<16x16x16x32x2xf16, 1>) outs(%out2 : memref<512x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 1 : i64} -> memref<512x512xf16, strided<[512, 1]>>
  }
  hmx.acc_read %bias, %ar2, %c15, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<16x16x16x32x2xf16, 1>
  hmx.unpack_acc ins(%ar2, %c15, %c0 : memref<16x16x16x32x2xf16, 1>) outs(%out2 : memref<512x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 1 : i64} -> memref<512x512xf16, strided<[512, 1]>>
  memref.dealloc %ar2 : memref<16x16x16x32x2xf16, 1>
  return
}

// The nested shape: the second matmul's tile loop sits inside the first's.
// The inner loop covers all of its AR's rows (no peel), so both loops match
// individually -- and the nesting is the one relation the group sequence rule
// must refuse.
func.func @nested(%bias: memref<256xi8, 1>, %out1: memref<1024x512xf16, strided<[512, 1]>>, %out2: memref<512x512xf16, strided<[512, 1]>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c31 = arith.constant 31 : index
  %c16 = arith.constant 16 : index
  %ar1 = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
  %ar2 = memref.alloc() {alignment = 128 : i64} : memref<16x16x16x32x2xf16, 1>
  scf.for %m = %c0 to %c31 step %c1 {
    %rowZero1 = arith.constant 0 : index
    %rowMul1 = arith.muli %c1, %rowZero1 : index
    %row1 = arith.addi %m, %rowMul1 : index
    scf.for %n1 = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar1, %row1, %n1 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar1, %row1, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out1 : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
    scf.for %m2 = %c0 to %c16 step %c1 {
      %row2 = arith.addi %m2, %c0 : index
      scf.for %n2 = %c0 to %c0 step %c1 {
        hmx.acc_clear
        hmx.acc_read %bias, %ar2, %row2, %n2 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<16x16x16x32x2xf16, 1>
      }
      hmx.unpack_acc ins(%ar2, %row2, %c0 : memref<16x16x16x32x2xf16, 1>) outs(%out2 : memref<512x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 1 : i64} -> memref<512x512xf16, strided<[512, 1]>>
    }
  }
  hmx.acc_read %bias, %ar1, %c31, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
  hmx.unpack_acc ins(%ar1, %c31, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out1 : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
  memref.dealloc %ar1 : memref<32x16x16x32x2xf16, 1>
  memref.dealloc %ar2 : memref<16x16x16x32x2xf16, 1>
  return
}
