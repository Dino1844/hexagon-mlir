//===- hmx-vector-readout-deferred-drain.mlir - move the exit drain off the hot path ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// defer-drain drops the kernel-exit `drain` so the tail batch's consumer work
// overlaps the function epilogue (AR release, return) instead of serialising
// behind it. The barrier is not removed, it moves: the next call's configure()
// drains before it does anything else, and the wrapper drains the last call of
// a launch outside the timed region (weak symbol, skipped by non-readout
// kernels -- see backend/hexagon_launcher_base.py).
//
// This file pins the one thing that could silently go wrong at the IR level:
// the drain call must be GONE from the rewritten kernel, everywhere, while the
// tail publish and the row coverage stay exactly where the non-deferred
// rewrite puts them. A drain that survives anywhere in the function is the
// option not firing; a tail publish that moves or disappears is the
// stale-bytes bug the main file exists to prevent.
//
// The DEFAULT arm is the other half of the contract: with defer-drain absent
// the drain is exactly where hmx-vector-readout-no-peel.mlir pins it, after
// the last publish and before the AR release. The option is off by default,
// so that is production.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4 defer-drain=true}))' | FileCheck %s --check-prefix=DEFER
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}))' | FileCheck %s --check-prefix=KEEP
//===----------------------------------------------------------------------===//

// Mt = 30, no peel: the whole matmul in one tile loop (`pipeline-depth=1`
// shape). In-loop groups cover rows 0..27, the tail names 28..29.
// DEFER-LABEL: func.func @single_loop
// DEFER-NOT: hexagon_runtime_hmx_exec_publish
// DEFER-NOT: hexagon_runtime_hmx_exec_drain
// DEFER: scf.for %[[M:.*]] = {{.*}} to {{.*}} step
// DEFER: scf.if
// DEFER: call @hexagon_runtime_hmx_exec_publish
// Past the loop: the tail handoff, 28 = G*floor(30/4), 2 = 30-28 -- the same
// numbers the non-deferred rewrite emits, because deferral moves the barrier,
// not the coverage.
// DEFER: %[[C28:.*]] = arith.constant 28 : index
// DEFER: arith.index_cast %[[C28]] : index to i32
// DEFER: memref.store {{.*}}, %{{.*}}[%{{.*}}] : memref<6xi32>
// DEFER: %[[C2:.*]] = arith.constant 2 : i32
// DEFER: memref.store %[[C2]], %{{.*}}[%{{.*}}] : memref<6xi32>
// DEFER: call @hexagon_runtime_hmx_exec_publish
// THE WHOLE POINT: no drain anywhere in the function -- not after the tail
// publish, not before the AR release, not anywhere. The next call's
// configure() and the wrapper's post-loop drain own the barrier now. (The
// trailing NOTs say `call @`: with deferral on, not even the module-level
// declaration exists, since emitDrain was the only thing that declared it.)
// DEFER-NOT: call @hexagon_runtime_hmx_exec_drain
// DEFER: memref.dealloc
// DEFER-NOT: hexagon_runtime_hmx_exec_drain
func.func @single_loop(%bias: memref<256xi8, 1>, %out: memref<1024x512xf16, strided<[512, 1]>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c30 = arith.constant 30 : index
  %ar = memref.alloc() {alignment = 128 : i64} : memref<30x16x16x32x2xf16, 1>
  scf.for %m = %c0 to %c30 step %c1 {
    %rowZero = arith.constant 0 : index
    %rowMul = arith.muli %c1, %rowZero : index
    %row = arith.addi %m, %rowMul : index
    scf.for %n = %c0 to %c0 step %c1 {
      hmx.acc_clear
      hmx.acc_read %bias, %ar, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<30x16x16x32x2xf16, 1>
    }
    hmx.unpack_acc ins(%ar, %row, %c0 : memref<30x16x16x32x2xf16, 1>) outs(%out : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
  }
  memref.dealloc %ar : memref<30x16x16x32x2xf16, 1>
  return
}

// KEEP runs the SAME input without defer-drain and pins that nothing about
// the deferral arm leaked into the default: the drain is emitted after the
// last publish and before the AR release, exactly as
// hmx-vector-readout-no-peel.mlir @tail_batch requires.
// KEEP-LABEL: func.func @single_loop
// KEEP-NOT: hexagon_runtime_hmx_exec_drain
// KEEP: scf.for %[[M2:.*]] = {{.*}} to {{.*}} step
// KEEP: scf.if
// KEEP: call @hexagon_runtime_hmx_exec_publish
// KEEP: call @hexagon_runtime_hmx_exec_publish
// KEEP: call @hexagon_runtime_hmx_exec_drain()
// KEEP: memref.dealloc
// KEEP-NOT: call @hexagon_runtime_hmx_exec_drain
