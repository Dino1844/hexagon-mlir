//===- hmx-vector-readout-pipeline.mlir - the switch, end to end -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// hmx-vector-readout.mlir pins the rewrite at the pass level. This file pins the
// SWITCH at the pipeline level, which is a different claim and needs a different
// fixture. It is the Triton-generated 1024x512x64 matmul verbatim -- the shape the
// 39.79% / 24.3 us measurement is about -- driven through -linalg-to-llvm at
// pipeline-depth=2 with the option off and then on.
//
// The fixture has to be this one rather than a hand-written `linalg.matmul`, for a
// reason worth recording: the pipeline's bridge verifier rejects an
// `hmx.decision_id` that has no `hex.hmx.kernel_manifest/v2` record, so a fixture
// that does not go through `matmul-to-hmx` cannot reach the options under test at
// all. This file's input lets `matmul-to-hmx` write the manifest itself, which is
// also why it is the real Triton output rather than something simpler: the
// read-out this pass rewrites is only hoisted into the tile loop when hmx-partition
// recognises the production shape (HmxPartitionPass.cpp:990-1083), and a synthetic
// matmul does not always produce it. A test that passed on a synthetic fixture
// could therefore be testing a shape the pass never sees in production.
//
// The two arms are the point. OFF (explicitly false) must be byte-identical
// to ... itself, so the arm the OFF checks describe is pinned by spelling the
// flag out; the DEFAULT arm is what the bare run now is -- the option flipped
// to default-on (2026-10-04), so the byte-identity gate moved with it: the
// bare run must be identical to the explicitly-ON run, or the .td default and
// the Python default have drifted apart again (the exact drift the option
// surface contract once caught). ON must show the handoff to the vector
// executor and the drain. Both directions are checked, so neither arm can
// pass vacuously.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-hmx-pipeline-depth=2})' > %t.default
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-hmx-pipeline-depth=2 enable-hmx-vector-readout=false})' > %t.off
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-hmx-pipeline-depth=2 enable-hmx-vector-readout=true})' > %t.on
// RUN: FileCheck %s --check-prefix=OFF < %t.off
// RUN: FileCheck %s --check-prefix=ON < %t.on
// RUN: diff %t.on %t.default
// RUN: not diff %t.off %t.on
//
// That last one is the switch itself: the option ON must produce different
// code from OFF, or the OFF arm would be proving nothing. `diff` is used
// rather than `cmp` so the failure names the lines that moved.
//===----------------------------------------------------------------------===//

// OFF arm. FileCheck directives must appear in OUTPUT order, so the NOTs come
// first: from the start of the output to the first engine leaf there is no executor
// traffic of any kind, which is what "off" has to mean.
// OFF-NOT: hexagon_runtime_hmx_exec_publish
// OFF-NOT: hexagon_runtime_hmx_exec_drain
// OFF-NOT: __hmx_readout
//
// The engine is reached, and the read-out runs inline on the engine's own thread --
// the pre-existing behaviour every earlier measurement describes. The bulk read-out
// leaf is the one hmx-partition's hoist produces for this shape (count = 16 croutons
// per row-pair).
// OFF: llvm.call @hmx_mma_f16
// OFF: llvm.call @hmx_unpack_acc_f16_bulk
//
// ON arm, also in output order. The engine still runs the matmul, and the read-out
// leaf still exists -- it has simply moved to the outlined function the vector thread
// runs, one call per row of the group.
//
// What the engine thread does instead is a handoff. It appears twice: once guarded
// inside the tile loop (the in-loop groups) and once after it (the tail batch, which
// carries the peeled row). Two handoffs for 32 rows is what "batched" means --
// without the option those 32 rows were 32 inline read-out sites.
// ON: llvm.call @hmx_mma_f16
// ON: llvm.call @hexagon_runtime_hmx_exec_publish
// ON: llvm.call @hexagon_runtime_hmx_exec_publish
//
// The engine thread's own read-out is gone. From its last handoff to the kernel's
// return -- past the frees and the HMX unlock -- there is no hmx_unpack_acc_f16_bulk
// at all. That absence is the actual claim of this option, so it is checked
// directly rather than inferred from the presence of a handoff.
// ON: llvm.call @hexagon_runtime_hmx_exec_drain()
// ON-NOT: llvm.call @hmx_unpack_acc_f16_bulk
// ON: llvm.return
//
// Then the outlined function, which is the only remaining read-out in the module.
// Its signature is the batch ABI -- (ar, dst, row0, nrows) -- which is what the
// runtime's HmxReadoutFn adapter will receive.
// ON: llvm.func @__hmx_readout(
// ON: llvm.call @hmx_unpack_acc_f16_bulk

module {
  func.func @matmul_kernel(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 0.000000e+00 : f16
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024, 64], strides: [64, 1] : memref<*xf16> to memref<1024x64xf16, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<1024x64xf16>
    memref.copy %reinterpret_cast, %alloc : memref<1024x64xf16, strided<[64, 1]>> to memref<1024x64xf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<1024x64xf16> to tensor<1024x64xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 512], strides: [512, 1] : memref<*xf16> to memref<64x512xf16, strided<[512, 1]>>
    %alloc_1 = memref.alloc() : memref<64x512xf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x512xf16, strided<[512, 1]>> to memref<64x512xf16>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x512xf16> to tensor<64x512xf16>
    %2 = tensor.empty() : tensor<1024x512xf16>
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<1024x512xf16>) -> tensor<1024x512xf16>
    %4 = linalg.matmul ins(%0, %1 : tensor<1024x64xf16>, tensor<64x512xf16>) outs(%3 : tensor<1024x512xf16>) -> tensor<1024x512xf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024, 512], strides: [512, 1] : memref<*xf16> to memref<1024x512xf16, strided<[512, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<1024x512xf16>, memref<1024x512xf16, strided<[512, 1]>>) -> ()
    return
  }
}
