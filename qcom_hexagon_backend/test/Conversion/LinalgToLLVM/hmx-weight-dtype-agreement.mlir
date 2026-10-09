//===- hmx-weight-dtype-agreement.mlir - one weight dtype, two fields ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The weight's element type is one fact with two serializations, and the two
// sides of the manifest/host rule about it read different serializations:
//
//   * the C++ finalize check (`HmxManifest.cpp`, the `dtypeMatchesReason`
//     conjunction) reads the record's `dtypes.rhs` when it admits
//     `eligible-aligned-f16` / `eligible-quantized-f32`;
//   * the host check (`backend/utils.py`, `_validate_manifest_weight_prepack`)
//     reads the pre-pack contract's `weights[].dtype` for the same rule.
//
// Nothing compared the two fields.  If they drifted, the cross-language gate in
// `test_hmx_plan_reason_rules_agreement.py` would be asking its two sides about
// different facts and would still agree -- which is the shape of a gate that
// always passes.
//
// They are not allowed to differ: both spell the weight argument's element type
// from the same resident source, in the same closed vocabulary {f16, f32} --
// `typeName` of the matmul's rhs operand in `MatmulToHmxPass.cpp` for the
// record, the `isF32(elem) ? "f32" : "f16"` ternary over the resident source
// in `WeightResidentPass.cpp` for the contract, with every other value refused
// on both sides (`HmxManifest.cpp` for `dtypes`, `validate_weight_prepack` for
// the entry).  The f16 and f32 fixtures below are therefore the whole
// vocabulary, not a sample of it.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-weight-resident=true})' -split-input-file | FileCheck %s --check-prefixes=F16,F32
//===----------------------------------------------------------------------===//

// Per dtype, the two serializations are checked together with the reason the
// rule derives from them, so a drift between the record's `dtypes.rhs` and the
// contract's `dtype` cannot hide behind a passing pair.

// F16-DAG: hmx.kernel_manifest
// F16-DAG: function = "runtime_weight_f16"
// F16-DAG: dtypes = {crouton = "f16", lhs = "f16", out = "f16", rhs = "f16"}
// F16-DAG: reason = "eligible-aligned-f16"
// F16-DAG: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_f16\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2],\22dtype\22:\22f16\22,\22location\22:\22vtcm\22}]"

// F32-DAG: hmx.kernel_manifest
// F32-DAG: function = "runtime_weight_f32"
// F32-DAG: dtypes = {crouton = "f16", lhs = "f32", out = "f32", rhs = "f32"}
// F32-DAG: reason = "eligible-quantized-f32"
// F32-DAG: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_f32\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2],\22dtype\22:\22f32\22,\22location\22:\22vtcm\22}]"

module {
  func.func @runtime_weight_f16(%a: tensor<64x64xf16>, %w: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %w : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}

// -----

module {
  func.func @runtime_weight_f32(%a: tensor<64x64xf32>, %w: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %empty = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %c = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %m = linalg.matmul ins(%a, %w : tensor<64x64xf32>, tensor<64x64xf32>)
                       outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %m : tensor<64x64xf32>
  }
}
