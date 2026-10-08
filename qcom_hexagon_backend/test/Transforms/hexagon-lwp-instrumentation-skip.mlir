// RUN: linalg-hexagon-opt %s -split-input-file -hexagon-lwp-instrumentation | FileCheck %s

//===----------------------------------------------------------------------===//
// 1. The crash configuration's minimal form: a module holding a function
//    DECLARATION. The HMX read-out split publishes its runtime entry points
//    as exactly this shape (HmxVectorReadoutPass::declareRuntime), and this
//    pass is scheduled addNestedPass<func::FuncOp>, so it visits every
//    function the module holds, declarations included. A declaration's body
//    region is empty, and the entry instrumentation used to dereference the
//    body's entry block -- `front()` on an empty region is the block-list
//    sentinel, the `!isKnownSentinel()` assert at compile time. Before the
//    skip (2026-10-08) this aborted: enableLWP x enableHmxVectorReadout on
//    every shape whose read-out the pass rewrites.
//
//    A declaration has nothing to instrument: it comes out unchanged.
//===----------------------------------------------------------------------===//

// CHECK: module {
// CHECK-NEXT:   func.func private @hexagon_runtime_hmx_exec_configure(i32, i32) -> i32
// CHECK-NOT: handler_name
// CHECK-NOT: instrprof
// CHECK: }

module {
  func.func private @hexagon_runtime_hmx_exec_configure(i32, i32) -> i32
}

// -----

//===----------------------------------------------------------------------===//
// 2. The read-out GROUP form: a kernel whose read-out was outlined, plus the
//    runtime declarations, in one module -- the module enableLWP now has to
//    survive with enableHmxVectorReadout on.
//
//    The kernel is instrumented (entry, loop, exit), and the executor's work
//    function and the declarations are left alone. The outlined read-out is
//    not a kernel (same skip in ThreadRolePartition); instrumenting it would
//    also put the partition tooling's "ID 1 = kernel total" row
//    (exp/hmx/t1_partition_2026_10_08/parse_partition.py) on whichever
//    function the concurrent pass manager happened to reach first.
//===----------------------------------------------------------------------===//

// CHECK: module {
// CHECK: llvm.mlir.global internal constant @handler_name("lwp_handler\00")
// CHECK: llvm.func @llvm.hexagon.instrprof.custom(!llvm.ptr, i32)
// CHECK: func.func @kernel
// CHECK: llvm.call @llvm.hexagon.instrprof.custom
// CHECK: scf.for
// CHECK: llvm.call @llvm.hexagon.instrprof.custom
// CHECK: return
// The outlined read-out: not a kernel, untouched.
// CHECK: func.func private @__hmx_readout
// CHECK-NOT: llvm.call @llvm.hexagon.instrprof.custom
// CHECK: return
// The runtime declaration: still a declaration, still untouched.
// CHECK: func.func private @hexagon_runtime_hmx_exec_drain()
// CHECK-NOT: llvm.call @llvm.hexagon.instrprof.custom
// CHECK: }

module {
  func.func @kernel(%in: memref<64x64xf16>, %out: memref<64x64xf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %in[%i, %i] : memref<64x64xf16>
      memref.store %v, %out[%i, %i] : memref<64x64xf16>
    }
    return
  }
  func.func private @__hmx_readout(%arg0: memref<32x16x16x32x2xf16, 1>, %arg1: memref<64x64xf16>, %arg2: index, %arg3: index) attributes {hmx.readout.outlined} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %arg3 step %c1 {
      %0 = arith.addi %arg2, %arg4 : index
      %1 = hmx.unpack_acc ins(%arg0, %0, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%arg1 : memref<64x64xf16>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<64x64xf16>
    }
    return
  }
  func.func private @hexagon_runtime_hmx_exec_drain()
}

// -----

//===----------------------------------------------------------------------===//
// 3. Two instrumentable functions in one module. Both are instrumented, and
//    the module-level handler global and intrinsic declaration are created
//    ONCE: the get-or-create is a lookup, so the second function is not a
//    redefinition of `handler_name` (which is exactly what the read-out era
//    exposed once declarations stopped aborting the pass -- a module with
//    two body-carrying functions was new then).
//===----------------------------------------------------------------------===//

// CHECK: module {
// CHECK: llvm.mlir.global internal constant @handler_name("lwp_handler\00")
// CHECK-NOT: llvm.mlir.global internal constant @handler_name
// CHECK: llvm.func @llvm.hexagon.instrprof.custom(!llvm.ptr, i32)
// CHECK-NOT: llvm.func @llvm.hexagon.instrprof.custom
// CHECK: func.func @first
// CHECK: llvm.call @llvm.hexagon.instrprof.custom
// CHECK: func.func @second
// CHECK: llvm.call @llvm.hexagon.instrprof.custom
// CHECK: }

module {
  func.func @first(%in: memref<64x64xf16>, %out: memref<64x64xf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %in[%i, %i] : memref<64x64xf16>
      memref.store %v, %out[%i, %i] : memref<64x64xf16>
    }
    return
  }
  func.func @second(%in: memref<64x64xf16>, %out: memref<64x64xf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %in[%i, %i] : memref<64x64xf16>
      memref.store %v, %out[%i, %i] : memref<64x64xf16>
    }
    return
  }
}
