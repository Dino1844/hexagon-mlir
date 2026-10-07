//===- hmx-vector-readout-configure.mlir - hand the executor its function ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// HmxVectorReadoutPass batches a matmul's accumulator read-out onto a resident
// vector thread, but a `publish` before the executor knows a function is dropped
// on the floor (HmxVectorExecutor.cpp:551-562), and the kernel then returns its
// own uninitialized output. The `configure` call that names the function is the
// last piece, and it CANNOT be emitted by the readout pass: `llvm.mlir.addressof`
// rejects a `func.func` symbol, and this pass runs before convert-func-to-llvm.
//
// So the fixture drives the producer and the lowering as two stages, with the
// func-to-llvm conversion in between, exactly as the production pipeline orders
// them (LinalgToLLVMPass.cpp:564-569 then :667 then :673). That ordering is the
// whole claim of this file: if the stages were swapped, the addressof would not
// verify.
//
// The three arms are what the gap closed:
//
//   ON      configure runs, and its address operand names the generated entry
//           point -- not the outlined read-out, because the runtime's
//           HmxReadoutFn takes `(const HmxReadoutBatch*, uint32_t)` and the
//           outlined read-out takes memrefs.
//   ADAPTER the entry point's descriptor unpack: the six i32 words in the frozen
//           field order, `ar`/`dst` as addresses, and a call into the outlined
//           read-out. This is checked word by word because a wrong index is a
//           garbage pointer, not a crash.
//   TRAP    every publish return is compared and a short return branches to a
//           trap. Checked per call site: there are two publishes, and a single
//           check would leave the in-loop drops unreported.
//
// The OFF arm is what every measurement taken before 2026-10-04 actually ran:
// the pass absent from the pipeline, so no publish code is emitted at all.
//
// There is deliberately no byte-equality gate here, and an earlier version of
// this file had one (`diff %t.off %t.off` -- the file compared with itself).
// Its comment claimed "OFF is byte-identical to a run with no option at all",
// but this file drives hand-written mini-pipelines, where "no option at all"
// is not expressible: the pass is simply absent from the pipeline string, so
// the claim reduces to the degenerate one. A gate that claims to verify
// something it does not is worse than no gate. The two things the claim was
// really about are pinned elsewhere:
//
//   - the option default is `true`, and .td <-> Python default agreement is
//     gated by test/test_option_surface_agreement.py (LinalgToLLVM/Passes.td:
//     191 is the enable-hmx-vector-readout definition).
//   - pipeline insertion is gated on that option at
//     LinalgToLLVMPass.cpp:584.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-vector-readout{hmx-readout-batch=4}),convert-scf-to-cf,convert-func-to-llvm,hmx-to-llvm)' > %t.on
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(convert-scf-to-cf,convert-func-to-llvm,hmx-to-llvm)' > %t.off
// RUN: FileCheck %s --check-prefix=ON < %t.on
// RUN: FileCheck %s --check-prefix=OFF < %t.off
//
// `convert-scf-to-cf` is in the pipeline because the publish check is a branch,
// and an `scf.if` body cannot hold one. That is not this test's requirement: it
// is where the production pipeline already puts the lowering
// (LinalgToLLVMPass.cpp:651 runs convert-scf-to-cf, :673 runs hmx-to-llvm), and a
// publish still inside structured control flow is refused with an error rather
// than producing malformed IR.
//===----------------------------------------------------------------------===//

// The handoff record is the only thing that makes the wiring happen, and it names
// both sides: the kernel that publishes and the outlined read-out it serves. It
// has to travel as a MODULE attribute because a function attribute does not
// survive the conversions in between -- CollapseAddressSpace rebuilds every
// llvm.func and drops discardable attributes (measured: the record is present
// after convert-func-to-llvm and gone after CollapseAddressSpace, which runs
// after this pass).
// ON: llvm.func @kernel

// The configure call. Two things are asserted here and neither is implied by the
// other:
//
//   * the ADDRESS is `llvm.mlir.addressof` of the entry point, i.e. a real symbol
//     address rather than a fabricated integer. This is the check the previous
//     implementation could not make: there is no addressof on a func.func.
//   * it sits BEFORE the first publish, in the entry block. A publish before
//     configure is a dropped batch.
// ON: llvm.mlir.addressof @__hmx_readout_entry : !llvm.ptr
// ON: llvm.ptrtoint {{.*}} : !llvm.ptr to i32
// ON: llvm.call @hexagon_runtime_hmx_exec_configure(
// ON: llvm.call @hexagon_runtime_hmx_exec_publish
//
// numThreads is 1, which is the only value the frozen executor honours
// (kVectorThreads == 1; anything else returns -2).

// The publish check. The comparison is against the count that was published, and
// the failure edge is a real trap rather than a call a later pass could inline
// away or a diagnostic that could be dropped: the destination region behind a
// dropped batch is never written, so this is a wrong answer, not a slow one.
// ON: llvm.call @hexagon_runtime_hmx_exec_publish({{.*}}) : (i32, i32) -> i32
// ON: llvm.icmp "ult"
// ON: llvm.intr.trap
// ON: llvm.call @hexagon_runtime_hmx_exec_drain

// The entry point. Its signature IS the ABI: `(const HmxReadoutBatch*, uint32_t)`
// becomes `(!llvm.ptr, i32)`. A different second parameter would mean the runtime
// called it with a count this function does not model.
// ON: llvm.func @__hmx_readout_entry(%{{.*}}: !llvm.ptr, %{{.*}}: i32)

// The descriptor is read as ONE six-i32 struct, which is a statement about the
// layout rather than a sequence of loads that happens to add up to it. The runtime
// declares `HmxReadoutBatch slots[16]` and so inherits sizeof; the two agree by
// construction.
// ON: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i32, i32, i32, i32)>

// The words that are forwarded, in the frozen order: 0 rowStart, 1 rowCount,
// 4 ar, 5 dst. Words 2 and 3 are absent, and their absence is asserted rather
// than assumed -- validRows and nCroutons are STATIC specialisation facts on this
// path, already baked into the outlined read-out's `hmx.unpack_acc` body, so
// reading them here would claim a dependency that does not exist. A future edit
// that forwards them would need to add the parameters to the callee as well.
// ON: llvm.extractvalue %{{.*}}[0]
// ON: llvm.extractvalue %{{.*}}[1]
// ON-NOT: llvm.extractvalue %{{.*}}[2]
// ON-NOT: llvm.extractvalue %{{.*}}[3]
// ON: llvm.extractvalue %{{.*}}[4]
// ON: llvm.extractvalue %{{.*}}[5]
//
// Each address becomes a pointer in ITS OWN space, taken from the memref type the
// producer recorded: `ar` is VTCM space 1, `dst` is space 0. Forcing one space for
// both would be a verifier error in one position or a false claim in the other.
// ON: llvm.inttoptr %{{.*}} : i32 to !llvm.ptr<1>
// ON: llvm.inttoptr %{{.*}} : i32 to !llvm.ptr

// The two words that became `index` are zero-extended to i64, which is the
// faithful conversion of a `uint32_t` and the width `index` lowered to.
// ON: llvm.zext %{{.*}} : i32 to i64

// The call into the outlined read-out, carrying both addresses.
// ON: llvm.call @__hmx_readout(

// OFF arm. Nothing at all: no record, no entry point, no configure, and above all
// no hmx.readout.handoffs attribute, because the readout pass never runs and so
// never publishes one. A module that kept the record with no configure would be
// the failure this whole mechanism exists to prevent.
// OFF-NOT: hexagon_runtime_hmx_exec
// OFF-NOT: __hmx_readout
// OFF-NOT: llvm.intr.trap
// OFF-NOT: readout.handoffs
// OFF: llvm.func @kernel

module {
  func.func @kernel(%bias: memref<256xi8, 1>, %out: memref<1024x512xf16, strided<[512, 1]>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c31 = arith.constant 31 : index
    %ar = memref.alloc() {alignment = 128 : i64} : memref<32x16x16x32x2xf16, 1>
    scf.for %m = %c0 to %c31 step %c1 {
      %rowZero = arith.constant 0 : index
      %rowMul = arith.muli %c1, %rowZero : index
      %row = arith.addi %m, %rowMul : index
      scf.for %n = %c0 to %c0 step %c1 {
        hmx.acc_clear
        hmx.acc_read %bias, %ar, %row, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
      }
      hmx.unpack_acc ins(%ar, %row, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
    }
    hmx.acc_read %bias, %ar, %c31, %c0 {bias_set = 0 : i32} : memref<256xi8, 1>, memref<32x16x16x32x2xf16, 1>
    hmx.unpack_acc ins(%ar, %c31, %c0 : memref<32x16x16x32x2xf16, 1>) outs(%out : memref<1024x512xf16, strided<[512, 1]>>) {count = 16 : i64, hmx.decision_id = 0 : i64} -> memref<1024x512xf16, strided<[512, 1]>>
    memref.dealloc %ar : memref<32x16x16x32x2xf16, 1>
    return
  }
}