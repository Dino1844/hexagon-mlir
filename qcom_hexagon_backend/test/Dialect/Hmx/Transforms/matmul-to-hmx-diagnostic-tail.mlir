//===- matmul-to-hmx-diagnostic-tail.mlir - producer tail contract -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The unit marker is an internal test-only producer trigger. It is intentionally
// placed on the original linalg.matmul; the normal backend never sets it.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx),one-shot-bufferize{bufferize-function-boundaries},func.func(hmx-partition),hmx-to-llvm)' | FileCheck %s --check-prefix=E2E
//===----------------------------------------------------------------------===//

// CHECK: module attributes
// CHECK-SAME: hmx.diagnostic_tail_partition
// CHECK-DAG: plan = "hmx-tail"
// CHECK-DAG: grid_policy = "single-instance"
// CHECK-DAG: tail_policy
// E2E: module attributes
// E2E-SAME: hmx.diagnostic_tail_partition
// E2E-DAG: llvm.call @hmx_pack_act_tail_f16
// E2E-DAG: llvm.call @hmx_pack_weight_tail_f16
// E2E-DAG: llvm.call @hmx_mma_f16
// E2E-DAG: llvm.call @hmx_unpack_acc_tail_f16
// E2E-DAG: reason = "tail-peeled-edge"
// E2E-DAG: pack_act_sites = 4 : i64
// E2E-NOT: hmx.matmul
// CHECK-LABEL: func.func @diagnostic_tail
// CHECK: arith.constant 0.000000e+00 : f16
// CHECK-COUNT-2: tensor.pad
// CHECK: hmx.pack_act
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK-SAME: hmx.diagnostic_tail_partition
// CHECK-SAME: tail_plan = #hmx.tail_plan<logical = [33, 33, 33], padded = [64, 64, 64], full = [32, 32, 32], tail = [1, 1, 1]
// CHECK: hmx.unpack_acc
func.func @diagnostic_tail(%a: tensor<33x33xf16>,
                           %b: tensor<33x33xf16>) -> tensor<33x33xf16> {
  %c = tensor.empty() : tensor<33x33xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<33x33xf16>, tensor<33x33xf16>)
      outs(%c : tensor<33x33xf16>) {hmx.diagnostic_tail_partition}
      -> tensor<33x33xf16>
  return %0 : tensor<33x33xf16>
}
