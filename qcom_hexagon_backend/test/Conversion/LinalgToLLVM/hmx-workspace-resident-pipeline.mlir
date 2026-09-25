//===- hmx-workspace-resident-pipeline.mlir - workspace end to end --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A runtime-weight HMX matmul through the whole LinalgToLLVM pipeline, with the
// per-launch VTCM workspace made resident. The crouton arrays, the conversion
// state and every other static VTCM allocation are allocated once per process
// by the runtime; the kernel emits a resident call for each and no per-launch
// allocator call at all. The option is off by default (its boundary is
// single-instance execution).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-workspace-resident})' | FileCheck %s
//===----------------------------------------------------------------------===//

// The pre-pack contract is published on the module: with enable-weight-resident
// on by default (the production value), the runtime weight is host-pre-packed
// once instead of being a per-launch workspace.
// CHECK: hmx.weight_prepack =
// The weight-residency entry is declared first, then the runtime entry that
// pins a workspace buffer: key + bytes + alignment (there is nothing to copy
// for a workspace).
// CHECK: llvm.func @hexagon_runtime_weight_resident_v2_dsp(i64, i32, i32) -> !llvm.ptr
// CHECK: llvm.func @hexagon_runtime_workspace_resident_v2_dsp(i64, i32, i32) -> !llvm.ptr

// CHECK-LABEL: llvm.func @runtime_weight
// The conversion-state allocation keeps its 256-byte alignment through the
// resident ABI; ordinary HMX workspaces remain at their existing 128-byte
// alignment.
// CHECK-DAG: %[[STATE_ALIGN:.*]] = llvm.mlir.constant(256 : i32) : i32
// One resident call per remaining per-launch VTCM workspace: the conversion
// state, the activation and the output crouton array. The weight is not one of
// them anymore.
// CHECK: llvm.call @hexagon_runtime_workspace_resident_v2_dsp({{.*}}, {{.*}}, %[[STATE_ALIGN]])
// CHECK: llvm.call @hexagon_runtime_workspace_resident_v2_dsp
// CHECK: llvm.call @hexagon_runtime_workspace_resident_v2_dsp
// The weight comes from the host pre-pack instead.
// CHECK: llvm.call @hexagon_runtime_weight_resident_v2_dsp
// No per-launch allocation or deallocation survives for any of them.
// CHECK-NOT: hexagon_runtime_alloc_1d_dsp
// CHECK-NOT: hexagon_runtime_free_1d_dsp
module {
  func.func @runtime_weight(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}
