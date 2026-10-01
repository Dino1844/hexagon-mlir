//===- matmul-to-hmx-contraction-interface.mlir - interface-driven matching ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// `matmul-to-hmx` matches `linalg::ContractionOpInterface`, not the concrete
// `linalg::MatmulOp`. That is the difference between "coverage grows by
// implementing an interface" and "coverage grows by writing another pass".
//
// It is not hypothetical: the Triton frontend picks the linalg op by result
// rank (`MatmulConverter` in triton-shared ConversionPatterns.hpp) -- rank 2
// becomes `linalg.matmul`, rank 3 becomes `linalg::BatchMatmulOp`. A
// matmul-only walk therefore never even offered a batched dot to this pass: no
// HMX site, no manifest record, no diagnostic, nothing to grep for.
//
// HMX's crouton contract is defined on rank-2 tiles, so the honest answer for a
// batched contraction today is a refusal carrying a published reason -- visible,
// not silently absent. Lifting the rank restriction is separate work and must
// not be smuggled in here.
//
// The manifest lives on the module attribute line, which prints *before* the
// function, so the record assertions come first and the function label after.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// A batched contraction is attributed instead of ignored. Before the interface
// the manifest for this function was empty: the op matched nothing.
// CHECK: function = "batched_contraction_is_attributed", id = 0 : i64, logical, plan = "hvx", reason = "non-rank-2", shape_state = "unavailable"
// CHECK-LABEL: func.func @batched_contraction_is_attributed
// No HMX site is invented for a shape the engine cannot take.
// CHECK-NOT: hmx.matmul
// CHECK-NOT: hmx.pack_act
func.func @batched_contraction_is_attributed(
    %a: tensor<2x64x128xf16>, %b: tensor<2x128x64xf16>
) -> tensor<2x64x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %init = tensor.empty() : tensor<2x64x64xf16>
  %c = linalg.fill ins(%zero : f16) outs(%init : tensor<2x64x64xf16>)
      -> tensor<2x64x64xf16>
  %0 = linalg.batch_matmul
      ins(%a, %b : tensor<2x64x128xf16>, tensor<2x128x64xf16>)
      outs(%c : tensor<2x64x64xf16>) -> tensor<2x64x64xf16>
  return %0 : tensor<2x64x64xf16>
}

// -----

// Generalising the match must not perturb what already worked: the rank-2
// matmul still takes the full-HMX path with the same bridge and site.
// CHECK: function = "rank2_contraction_unchanged"
// CHECK: plan = "full-hmx", reason = "selected-aligned"
// CHECK-LABEL: func.func @rank2_contraction_unchanged
// CHECK: hmx.matmul
func.func @rank2_contraction_unchanged(
    %a: tensor<64x128xf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// A `linalg.contract` with a transposed operand map is a legal op that the
// bridge cannot express. Matching the interface opened this input: `linalg.matmul`
// could never carry an arbitrary map (its ODS only permits broadcast maps), so
// before the interface it was unreachable. It must be refused with a reason, not
// packed as if row-major and left to fail inside the verifier.
// CHECK: function = "transposed_contract_is_refused"
// CHECK: plan = "hvx", reason = "unsupported-layout"
// CHECK-LABEL: func.func @transposed_contract_is_refused
// CHECK-NOT: hmx.matmul
func.func @transposed_contract_is_refused(
    %a: tensor<128x64xf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %init = tensor.empty() : tensor<64x64xf16>
  %c = linalg.fill ins(%zero : f16) outs(%init : tensor<64x64xf16>)
      -> tensor<64x64xf16>
  %0 = linalg.contract
      indexing_maps = [affine_map<(m, n, k) -> (k, m)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%a, %b : tensor<128x64xf16>, tensor<128x64xf16>)
      outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
