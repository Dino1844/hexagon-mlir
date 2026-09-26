//===- matmul-to-hmx-dead-matmul.mlir - a matmul the driver removed -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The attribution table is built before the greedy driver runs, but the driver
// dead-code-eliminates a matmul whose result is unused *before* it offers the
// op to any pattern. Such a decision then outlives its operation: there is no
// `hmx.matmul` and no bridge behind it, and its operation handle dangles.
//
// A decision like that is therefore described by nothing and read by nothing:
// no manifest record (a selected record with no `hmx.matmul` to bind is what
// `restoreHmxManifestDecisionIds` rejects at the bufferization hand-off), no v3
// skeleton, and no remark to anchor. The matmuls that survive in the same
// function are unaffected, which is what these cases pin.
//
// Record-only rewrites nothing, so nothing is removed and every matmul it was
// given is still described -- the asymmetry the last case pins.
//
// The CHECKs are one flat scan in output order (the manifest is printed ahead
// of the function it describes, so a CHECK-LABEL would cut it off).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file -verify-diagnostics | FileCheck %s
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=1}))' -split-input-file | FileCheck %s --check-prefix=RECORD
//===----------------------------------------------------------------------===//

// Two eligible dots, the first one's result never read. The removed dot leaves
// no record and no bridge behind; the second is attributed under its own id.
module {
func.func @dead_first_dot(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                          %c: tensor<64x64xf16>,
                          %d: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                     outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %1 = linalg.matmul ins(%c, %d : tensor<64x64xf16>, tensor<64x64xf16>)
                     outs(%e1 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %1 : tensor<64x64xf16>
}
}
// CHECK: matmuls = [{
// CHECK-NOT: id = 0 : i64
// CHECK: id = 1 : i64
// CHECK: plan = "full-hmx"
// CHECK: reason = "selected-aligned"
// CHECK: pack_act_sites = 1 : i64
// CHECK: pack_weight_sites = 1 : i64
// CHECK: schema = "hex.hmx.kernel_manifest/v2"
// CHECK: unpack_sites = 1 : i64
// CHECK: func.func @dead_first_dot
// CHECK: hmx.matmul ins({{.*}}) outs({{.*}}) {hmx.decision_id = 1 : i64}
// CHECK-NOT: linalg.matmul

// -----

// The same removal on the diagnostic side: the dead dot is a refusal the
// engine would have explained, but its operation is gone, so it is neither
// remarked nor counted. The one live refusal is reported exactly as if the
// dead dot were not in the function, and its record is the only one published.
// expected-warning @+1 {{HMX: 1 matmul(s) skipped; first refusal: HMX not applied: needs f16/f32 inputs and an f16/f32 result, 2D static shapes, M/N/K multiples of 32, M > 4 [reason=min-rows]; matmul M=2, N=64, K=64}}
module {
func.func @dead_refusal(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>,
                        %c: tensor<2x64xf16>,
                        %d: tensor<64x64xf16>) -> tensor<2x64xf16> {
  %e0 = tensor.empty() : tensor<16x16xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<16x16xf16>, tensor<16x16xf16>)
                     outs(%e0 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %e1 = tensor.empty() : tensor<2x64xf16>
  // expected-remark @+1 {{HMX not applied: needs f16/f32 inputs and an f16/f32 result, 2D static shapes, M/N/K multiples of 32, M > 4 [reason=min-rows]}}
  %1 = linalg.matmul ins(%c, %d : tensor<2x64xf16>, tensor<64x64xf16>)
                     outs(%e1 : tensor<2x64xf16>) -> tensor<2x64xf16>
  return %1 : tensor<2x64xf16>
}
}
// CHECK: matmuls = [{
// CHECK-NOT: id = 0 : i64
// CHECK: id = 1 : i64
// CHECK: plan = "hvx"
// CHECK: reason = "min-rows"
// CHECK: pack_act_sites = 0 : i64
// CHECK: schema = "hex.hmx.kernel_manifest/v2"
// CHECK: func.func @dead_refusal
// CHECK: linalg.matmul

// -----

// Record-only promises not to rewrite its input, so it describes every matmul
// it was handed -- including one whose result is unused, which the rewriting
// pass above finds gone.
module {
func.func @record_only_keeps_dead(%a: tensor<64x64xf16>,
                                  %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                     outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %e0 : tensor<64x64xf16>
}
}
// RECORD: matmuls = [{
// RECORD: id = 0 : i64
// RECORD: logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 64 : i64}, n = {kind = "static", value = 64 : i64}}
// RECORD: plan = "hvx"
// RECORD: reason = "vtcm-allocator-disabled"
// RECORD: func.func @record_only_keeps_dead
// RECORD: linalg.matmul
// RECORD-NOT: hmx.matmul
