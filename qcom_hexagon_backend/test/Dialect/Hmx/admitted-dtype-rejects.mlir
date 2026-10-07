//===- admitted-dtype-rejects.mlir - non-admitted pack source dtypes ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The ODS side of the engine's admitted source-dtype set: a pack (or stage)
// source must be f16 or f32 -- anything else is rejected by the op verifier
// before any lowering runs. The set's other spellings are
// dtype::isAdmittedFloat (HmxDType.h) and the packActLeafFamily /
// packWeightLeafFamily tables (HmxToLLVMPass.cpp), which fail the conversion
// loudly on an unadmitted type rather than reading the source as f16;
// test/test_hmx_dtype_admitted_set_contract.py binds the three together.
//
// f64 is here on purpose: it is a float the engine still does not admit, so
// this pins the set as exactly {f16, f32}, not "any float".
//
// RUN: linalg-hexagon-opt %s -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

//--- !!! hmx.pack_act packs an f16/f32 source; an i32 source has no leaf
func.func @bad_pack_act_i32(%src: memref<64x64xi32>, %act: memref<2x2x16x32x2xf16, 1>, %row: index, %col: index) {
  // expected-error @+1 {{operand #1 must be non-0-ranked.tensor of 16-bit float or 32-bit float values or non-0-ranked.memref of 16-bit float or 32-bit float values}}
  hmx.pack_act ins(%src, %row, %col : memref<64x64xi32>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  return
}

//--- !!! ... and a wider float is still outside the set: {f16, f32}, not any
//--- float
func.func @bad_pack_act_f64(%src: memref<64x64xf64>, %act: memref<2x2x16x32x2xf16, 1>, %row: index, %col: index) {
  // expected-error @+1 {{operand #1 must be non-0-ranked.tensor of 16-bit float or 32-bit float values or non-0-ranked.memref of 16-bit float or 32-bit float values}}
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf64>) outs(%act : memref<2x2x16x32x2xf16, 1>)
  return
}

//--- !!! hmx.pack_weight shares the admitted set
func.func @bad_pack_weight_i32(%src: memref<64x64xi32>, %wt: memref<2x2x16x32x2xf16, 1>, %k: index, %n: index) {
  // expected-error @+1 {{operand #1 must be non-0-ranked.tensor of 16-bit float or 32-bit float values or non-0-ranked.memref of 16-bit float or 32-bit float values}}
  hmx.pack_weight ins(%src, %k, %n : memref<64x64xi32>) outs(%wt : memref<2x2x16x32x2xf16, 1>)
  return
}

//--- !!! hmx.stage carries the same source rows across the DMA, so the same
//--- set applies
func.func @bad_stage_i32(%src: memref<64x1024xi32>, %slot: memref<32x1024xf16, 1>, %status: memref<1xi32>, %row: index) {
  // expected-error @+1 {{operand #0 must be non-0-ranked.memref of 16-bit float or 32-bit float values}}
  %tok = hmx.stage ins(%src, %row : memref<64x1024xi32>) outs(%slot, %status : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
  return
}
