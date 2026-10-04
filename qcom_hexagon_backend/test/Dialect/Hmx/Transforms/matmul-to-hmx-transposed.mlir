//===- matmul-to-hmx-transposed.mlir - packing a K^T view directly --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The weight crouton stores Wᵀ, so a matmul rhs produced by
// `linalg.transpose(x){perm = [1, 0]}` is exactly the layout change the bridge
// wants: the pack reads `x` directly -- marked `src_transposed` on
// `hmx.pack_weight` -- and the transpose's result tensor never materialises
// (under one-shot bufferization it is a full-matrix copy; 2048 of them in
// flash attention, docs/hmx/fa-transpose-copy-design-2026-09-30.md).
//
// Admission is mechanism-only: the exact 2D swap, a static rank-2 input, and
// the matmul as the transpose result's only user. Every refusal below keeps
// the materialised form.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// 64x128 * 128x64: `x` is the [N, K] = [64, 128] transpose input, the pack
// loop walks its 2 N tiles with the whole 4-tile K run in `count`, and the
// weight crouton grid is the same [Nt, Kt] = [2, 4] the materialised form
// builds. The transpose itself is gone.
// CHECK-LABEL: func.func @transposed_weight
// CHECK: hmx.pack_weight ins(%arg1, %c0, %{{.*}} : tensor<64x128xf16>)
// CHECK-SAME: src_transposed
// CHECK: hmx.matmul ins(%{{.*}}, %{{.*}} : tensor<2x4x16x32x2xf16>, tensor<2x4x16x32x2xf16>)
// CHECK: hmx.unpack_acc
// CHECK-NOT: linalg.transpose
func.func @transposed_weight(%a: tensor<64x128xf16>, %w: tensor<64x128xf16>) -> tensor<64x64xf16> {
  %init = tensor.empty() : tensor<128x64xf16>
  %wt = linalg.transpose ins(%w : tensor<64x128xf16>) outs(%init : tensor<128x64xf16>) permutation = [1, 0]
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %wt : tensor<64x128xf16>, tensor<128x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// The transpose result has a second user, so the materialised form stays (the
// second user still needs it) and neither pack carries the marker.
// CHECK-LABEL: func.func @transposed_weight_shared
// CHECK: linalg.transpose
// CHECK: hmx.pack_weight
// CHECK: hmx.pack_weight
// CHECK-NOT: src_transposed
func.func @transposed_weight_shared(%a: tensor<64x128xf16>, %w: tensor<64x128xf16>) -> tensor<64x64xf16> {
  %init = tensor.empty() : tensor<128x64xf16>
  %wt = linalg.transpose ins(%w : tensor<64x128xf16>) outs(%init : tensor<128x64xf16>) permutation = [1, 0]
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %wt : tensor<64x128xf16>, tensor<128x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  %1 = linalg.matmul ins(%a, %wt : tensor<64x128xf16>, tensor<128x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  %2 = arith.addf %0, %1 : tensor<64x64xf16>
  return %2 : tensor<64x64xf16>
}

// -----

// A constant transpose input folds with the orientation folded in too: the
// prepacked constant is built by reading W[k, n] at n * K + k, so no runtime
// packer runs for it at all.
// CHECK-LABEL: func.func @transposed_constant
// CHECK-NOT: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK-NOT: linalg.transpose
func.func @transposed_constant(%a: tensor<64x128xf16>) -> tensor<64x64xf16> {
  %w = arith.constant dense<1.000000e+00> : tensor<64x128xf16>
  %init = tensor.empty() : tensor<128x64xf16>
  %wt = linalg.transpose ins(%w : tensor<64x128xf16>) outs(%init : tensor<128x64xf16>) permutation = [1, 0]
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %wt : tensor<64x128xf16>, tensor<128x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// The M-blocked form resolves the same view: 32768x128 exceeds one bridge's
// VTCM budget, so M walks in 16384-row blocks while the weight bridge packs
// from the transpose input, hoisted out of the block loop exactly as the
// materialised form's would be.
// CHECK-LABEL: func.func @transposed_weight_blocked
// CHECK: hmx.pack_weight ins(%arg1, %c0, %{{.*}} : tensor<64x128xf16>)
// CHECK-SAME: src_transposed
// CHECK: scf.for %{{.*}} to %c32768
// CHECK: hmx.matmul
// CHECK-NOT: linalg.transpose
func.func @transposed_weight_blocked(%a: tensor<32768x128xf16>, %w: tensor<64x128xf16>) -> tensor<32768x64xf16> {
  %init = tensor.empty() : tensor<128x64xf16>
  %wt = linalg.transpose ins(%w : tensor<64x128xf16>) outs(%init : tensor<128x64xf16>) permutation = [1, 0]
  %c = tensor.empty() : tensor<32768x64xf16>
  %0 = linalg.matmul ins(%a, %wt : tensor<32768x128xf16>, tensor<128x64xf16>) outs(%c : tensor<32768x64xf16>) -> tensor<32768x64xf16>
  return %0 : tensor<32768x64xf16>
}
