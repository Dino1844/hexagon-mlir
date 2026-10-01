//===- hmx-weight-resident-f32-pipeline.mlir - f32 runtime weight end to end ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// An f32 runtime weight through the whole LinalgToLLVM pipeline with the P2
// gate on. The host pre-packs the crouton image by quantising the weight to the
// engine's fp16 (the same round-to-nearest conversion the device pack leaf
// runs) and permuting it, so the per-launch `hmx_pack_weight_f32` leaf
// disappears exactly as the f16 one does. The manifest names the quantising
// policy reason, and the resident call receives the argument's address.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-weight-resident=true})' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_manifest =
// CHECK-DAG: function = "runtime_weight_f32"
// CHECK-DAG: pack_weight_sites = 0 : i64
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: reason = "selected-aligned"
// CHECK-DAG: weight_policies = [{consumers = [0], function = "runtime_weight_f32", policy = "resident-prepack", reason = "eligible-quantized-f32", slot = 1 : i64}]
// The runtime gets the argument's address; no per-launch pack leaf is emitted.
// CHECK: llvm.func @hexagon_runtime_weight_resident_v2_dsp(i64, i32, i32) -> !llvm.ptr
// CHECK-LABEL: llvm.func @runtime_weight_f32
// CHECK: llvm.call @hexagon_runtime_weight_resident_v2_dsp
// CHECK-NOT: llvm.call @hmx_pack_weight_f32
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
