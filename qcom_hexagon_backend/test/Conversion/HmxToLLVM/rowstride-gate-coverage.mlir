//===- rowstride-gate-coverage.mlir - the row_stride gate's other two surfaces --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-to-llvm)' -verify-diagnostics
//===----------------------------------------------------------------------===//
//
// `rowStride` (HmxToLLVMPass.cpp) rejects a STATIC non-inner-contiguous layout
// with "HMX leaf requires an inner-contiguous buffer". 2026-10-01: an
// independent audit established that the gate fires on THREE surfaces, and that
// only ONE of them had a test. The other two are pinned here.
//
//   surface                          covered by
//   ------------------------------   ------------------------------------------
//   unpack DESTINATION               unpack-dst-non-contiguous-reject.mlir
//   unpack RESIDUAL (f32 tail)       this file, case 1
//   hmx.stage SOURCE                 this file, case 2
//   pack SOURCE (act, weight)        already refused upstream by the op
//                                    verifier -- "src row stride 1 is smaller
//                                    than its N columns", so the gate there is
//                                    unreachable and needs no test
//
// -split-input-file is load-bearing for the same reason it is in
// unpack-dst-non-contiguous-reject.mlir: dialect conversion ABORTS at the first
// legalization failure, so with several cases in one module the later ones are
// never converted and their expectations go unchecked. Do not merge these cases
// into one module, and do not spell the split marker inside a comment.

// -----

// The f32 residual of a tail unpack is a rank-2 leaf buffer, so a transposed
// residual is the same hazard as a transposed destination, reached by a
// different call site (the f32 tail lowering). `residual` must have exactly
// `dst`'s type (HmxOps.td), so both are strided the same way here.
func.func @transposed_f32_residual(
    %src: memref<1x1x16x32x2xf16, 1>,
    %dst: memref<32x32xf32, strided<[1, 32]>>,
    %res: memref<32x32xf32, strided<[1, 32]>>,
    %row: index, %col: index) {
  // expected-error @+2 {{HMX leaf requires an inner-contiguous buffer; got inner stride 32}}
  // expected-error @+1 {{failed to legalize operation 'hmx.unpack_acc_f32'}}
  hmx.unpack_acc_f32 ins(%src, %row, %col, %res
                    : memref<1x1x16x32x2xf16, 1>, memref<32x32xf32, strided<[1, 32]>>)
      outs(%dst : memref<32x32xf32, strided<[1, 32]>>)
  return
}

// -----

// hmx.stage's SOURCE is addressed by the leaf, so the same rule applies there.
// A row-major staged source with stride(0) > width is the case the gate must
// keep ALLOWING, and a transposed one is the case it must refuse; both are here
// so the third surface is pinned in both directions.
func.func @row_major_stage_source_is_still_accepted(
    %src: memref<32x64xf16, strided<[64, 1]>>, %dst: memref<1x2x16x32x2xf16, 1>,
    %row: index, %col: index) {
  hmx.pack_act ins(%src, %row, %col : memref<32x64xf16, strided<[64, 1]>>)
      outs(%dst : memref<1x2x16x32x2xf16, 1>)
  return
}

// -----

func.func @transposed_stage_source(
    %src: memref<32x32xf16, strided<[1, 32]>>, %dst: memref<1x1x16x32x2xf16, 1>,
    %row: index, %col: index) {
  // Only ONE diagnostic here, and that is the point of the case: the op
  // verifier rejects this at verify time, so conversion never starts and there
  // is no "failed to legalize" second line. A transposed pack source is
  // therefore unreachable by the rowStride gate -- the audit measured this, and
  // it is why the gate's two pack-source call sites need no test.
  // expected-error @+1 {{src row stride 1 is smaller than its 32 columns}}
  hmx.pack_act ins(%src, %row, %col : memref<32x32xf16, strided<[1, 32]>>)
      outs(%dst : memref<1x1x16x32x2xf16, 1>)
  return
}
