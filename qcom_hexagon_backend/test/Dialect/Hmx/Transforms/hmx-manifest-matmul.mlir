//===- hmx-manifest-matmul.mlir - v1 attribution manifest -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Every original linalg.matmul owns one function-local record. Selected records
// carry the target/budget decision; HVX records carry the first canonical reason
// that kept the op on the existing path. Bridge counts are static IR sites, not
// launch-time call counts.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s --check-prefix=EMPTY
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s --check-prefix=SELECTED
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s --check-prefix=BLOCKED
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s --check-prefix=REFUSALS
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=10000}))' -split-input-file | FileCheck %s --check-prefix=BUDGET
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-allocator=false}))' -split-input-file | FileCheck %s --check-prefix=ALLOC
//===----------------------------------------------------------------------===//

// A module with no matmul still publishes the schema and an empty array.
// EMPTY: hmx.kernel_manifest = {
// EMPTY: count_semantics = "ir_sites"
// EMPTY: matmuls = []
// EMPTY: pack_act_sites = 0 : i64
// EMPTY: pack_weight_sites = 0 : i64
// EMPTY: unpack_sites = 0 : i64
// EMPTY: schema = "hex.hmx.kernel_manifest/v1"
module {
  func.func @empty() {
    return
  }
}

// -----

// A legal contraction is selected, its generated hmx.matmul carries the stable
// decision id, and the manifest reports the actual bridge footprint.
// SELECTED: hmx.kernel_manifest = {
// SELECTED: count_semantics = "ir_sites"
// SELECTED: block_m = 64 : i64
// SELECTED: blocking = "whole"
// SELECTED: count_semantics = "ir_sites"
// SELECTED: engine = "hmx"
// SELECTED: function = "selected"
// SELECTED: id = 0 : i64
// SELECTED: k = 64 : i64
// SELECTED: lhs_elem = "f16"
// SELECTED: m = 64 : i64
// SELECTED: n = 64 : i64
// SELECTED: out_elem = "f16"
// SELECTED: pack_act_sites = 1 : i64
// SELECTED: pack_weight_sites = 1 : i64
// SELECTED: reason = "selected"
// SELECTED: rhs_elem = "f16"
// SELECTED: schema = "hex.hmx.kernel_manifest/v1"
// SELECTED: unpack_sites = 1 : i64
// SELECTED: vtcm_before = 0 : i64
// SELECTED: vtcm_budget = 8388608 : i64
// SELECTED: vtcm_peak = 24576 : i64
// SELECTED: hmx.matmul
// SELECTED: hmx.decision_id = 0 : i64
// With the allocator off, even the otherwise legal contraction stays HVX and
// publishes the environment code rather than a capability/budget code.
// ALLOC: hmx.kernel_manifest = {
// ALLOC: engine = "hvx"
// ALLOC: function = "selected"
// ALLOC: reason = "vtcm-allocator-disabled"
module {
  func.func @selected(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// A too-large whole contraction is still selected; the M-block decision is
// explicit rather than inferred by the host from the logical shape.
// BLOCKED: hmx.kernel_manifest = {
// BLOCKED: block_m = 512 : i64
// BLOCKED: blocking = "m_blocked"
// BLOCKED: count_semantics = "ir_sites"
// BLOCKED: reason = "selected"
// BLOCKED: vtcm_before = 0 : i64
// BLOCKED: vtcm_peak = 4784128 : i64
// BLOCKED: hmx.matmul
// BLOCKED: hmx.decision_id = 0 : i64
module {
  func.func @blocked(%a: tensor<4096x64xf16>, %b: tensor<64x4096xf16>) -> tensor<4096x4096xf16> {
    %c = tensor.empty() : tensor<4096x4096xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<4096x64xf16>, tensor<64x4096xf16>)
                       outs(%c : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
    return %0 : tensor<4096x4096xf16>
  }
}

// -----

// The four HVX decisions remain four records in deterministic function-local
// walk order. This is the duplicate-counting guard: the greedy driver may ask
// the pattern repeatedly, but these ids are created before it starts.
// REFUSALS: hmx.kernel_manifest = {
// REFUSALS: matmuls = [{engine = "hvx", function = "refusals", id = 0 : i64
// REFUSALS: reason = "tile-alignment"
// REFUSALS: }, {engine = "hvx", function = "refusals", id = 1 : i64
// REFUSALS: reason = "unsupported-dtype"
// REFUSALS: }, {engine = "hvx", function = "refusals", id = 2 : i64
// REFUSALS: reason = "library-call"
// REFUSALS: }, {engine = "hvx", function = "refusals", id = 3 : i64
// REFUSALS: reason = "dynamic-shape"
module {
  func.func @refusals(%a: tensor<64x31xf16>, %b: tensor<31x64xf16>,
                      %d: tensor<64x64xf64>, %e: tensor<64x64xf64>,
                      %x: tensor<?x31xf16>, %n: index) -> (tensor<64x64xf16>, tensor<64x64xf64>, tensor<64x64xf16>, tensor<?x64xf16>) {
    %c0 = tensor.empty() : tensor<64x64xf16>
    %m0 = linalg.matmul ins(%a, %b : tensor<64x31xf16>, tensor<31x64xf16>)
                       outs(%c0 : tensor<64x64xf16>) -> tensor<64x64xf16>

    %c1 = tensor.empty() : tensor<64x64xf64>
    %m1 = linalg.matmul ins(%d, %e : tensor<64x64xf64>, tensor<64x64xf64>)
                       outs(%c1 : tensor<64x64xf64>) -> tensor<64x64xf64>

    %c2 = tensor.empty() : tensor<64x64xf16>
    %m2 = linalg.matmul {library_call = "custom_mm"}
                       ins(%a, %b : tensor<64x31xf16>, tensor<31x64xf16>)
                       outs(%c2 : tensor<64x64xf16>) -> tensor<64x64xf16>

    %c3 = tensor.empty(%n) : tensor<?x64xf16>
    %m3 = linalg.matmul ins(%x, %b : tensor<?x31xf16>, tensor<31x64xf16>)
                       outs(%c3 : tensor<?x64xf16>) -> tensor<?x64xf16>
    return %m0, %m1, %m2, %m3 : tensor<64x64xf16>, tensor<64x64xf64>, tensor<64x64xf16>, tensor<?x64xf16>
  }
}

// -----

// Capability passed, but the bridge did not fit. The record remains HVX and
// carries the budget reason; it must not publish selected-only budget fields.
// BUDGET: hmx.kernel_manifest = {
// BUDGET: engine = "hvx"
// BUDGET: function = "over_budget"
// BUDGET: id = 0 : i64
// BUDGET: reason = "vtcm-budget"
// BUDGET: schema = "hex.hmx.kernel_manifest/v1"
module {
  func.func @over_budget(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}
