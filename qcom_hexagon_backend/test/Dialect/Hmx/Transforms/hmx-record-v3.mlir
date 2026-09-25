//===- hmx-record-v3.mlir - record-only v3 document shape ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The host-side fixture for the record-only `hex.hmx.kernel_manifest/v3`
// contract.  Each split is one case of
// docs/hmx/hmx-v3-manifest-decision.md section 2: the closed field set, the
// three separate capacity facts, and the four independent proof statuses.
//
// The producer is marker-gated, so the last two cases are the ones that matter
// most: with no marker nothing is published at all, and the v2 manifest is the
// same record it always was.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=STATIC
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=DYNAMIC
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=TAIL
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=MULTI
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=INERT
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=V2
//===----------------------------------------------------------------------===//

// A statically specialized HVX record.  `record-only` refuses HMX for lack of a
// VTCM allocator, which is a compile-time outcome: the v3 record publishes the
// shape it was given and nothing else.
//
// Every capacity fact starts unknown -- `not-proven` with a `null` quantity,
// never a zero.  The only numbers in a fresh record are the record id and the
// fixed analysis scope.
//
// STATIC: hmx.kernel_record/v3" = {
// STATIC-DAG: admission = "not-authorized"
// STATIC-DAG: record_mode = "record-only"
// STATIC-DAG: schema = "hex.hmx.kernel_manifest/v3"
// STATIC-DAG: fallback = {on_malformed_record = "reject-v3-record"}
// STATIC-DAG: function = "matmul_kernel"
// STATIC-DAG: id = 0 : i64
// STATIC-DAG: plan = "hvx"
// STATIC-DAG: shape = {logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 64 : i64}, n = {kind = "static", value = 64 : i64}}, specialization = "upstream-static", state = "static"}
// STATIC-DAG: scope = {function = "matmul_kernel", grid = {policy = "single-instance", required_product = 1 : i64}, invocations = 1 : i64, resident = "process-floor"}
// STATIC-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}
// STATIC-DAG: allocator_aligned = {basis = "allocator-model", modeled_aligned_peak_bytes, resident_aligned_bytes, status = "not-proven", transient_aligned_peak_bytes, unit = "bytes"}
// STATIC-DAG: observed_high_water = {basis = "runtime-observation", scope = "process-high-water", source, status = "not-proven", unit = "bytes", value_bytes}
// STATIC-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
// STATIC-DAG: record_fingerprint = "sha256:
module attributes {hmx.diagnostic_v3_record} {
  func.func @matmul_kernel(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// A dynamic axis names itself and carries no runtime guess.  `shape.state` is
// re-derived from the three tagged axes, and the specialization policy follows
// from the same three axes: v3 does not introduce a dispatcher that would turn a
// dynamic extent into an implicit HMX specialization.
//
// DYNAMIC: hmx.kernel_record/v3" = {
// DYNAMIC-DAG: shape = {logical = {k = {kind = "dynamic", symbol = "k"}, m = {kind = "static", value = 64 : i64}, n = {kind = "static", value = 64 : i64}}, specialization = "upstream-only", state = "partially-dynamic"}
// DYNAMIC-DAG: plan = "hvx"
module attributes {hmx.diagnostic_v3_record} {
  func.func @dynamic_kernel(%a: tensor<64x?xf16>, %b: tensor<?x64xf16>, %n: index) -> tensor<64x?xf16> {
    %e = tensor.empty(%n) : tensor<64x?xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x?xf16>, tensor<?x64xf16>) outs(%e : tensor<64x?xf16>) -> tensor<64x?xf16>
    return %0 : tensor<64x?xf16>
  }
}

// -----

// `hmx-tail` is a diagnostic/test slice.  v3 records which plan the compile
// produced; it does not authorize the plan, and the record's own fallback says
// so: an unproven proof retains the v2 plan, and a descriptor mismatch rejects.
//
// TAIL: hmx.kernel_record/v3" = {
// TAIL-DAG: plan = "hmx-tail"
// TAIL-DAG: shape = {logical = {k = {kind = "static", value = 33 : i64}, m = {kind = "static", value = 33 : i64}, n = {kind = "static", value = 33 : i64}}, specialization = "upstream-static", state = "static"}
// TAIL-DAG: scope = {function = "diagnostic_tail", grid = {policy = "single-instance", required_product = 1 : i64}, invocations = 1 : i64, resident = "process-floor"}
// TAIL-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
module attributes {hmx.diagnostic_v3_record} {
  func.func @diagnostic_tail(%a: tensor<33x33xf16>, %b: tensor<33x33xf16>) -> tensor<33x33xf16> {
    %c = tensor.empty() : tensor<33x33xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<33x33xf16>, tensor<33x33xf16>)
        outs(%c : tensor<33x33xf16>) {hmx.diagnostic_tail_partition}
        -> tensor<33x33xf16>
    return %0 : tensor<33x33xf16>
  }
}

// -----

// Two recordable functions plus a function with nothing to record.  The
// published document is a list of per-function facts in canonical order; v3
// never sums them into a module total, because a per-function peak is not a
// module peak.
//
// `@library_kernel` still gets a record: v3 publishes the compile-time facts
// (shape and plan), not a reason and not an admission, so a library dispatch is
// recorded as the HVX plan the compile produced.  `@no_matmul` gets nothing at
// all -- a record is a per-function fact about a matmul, and inventing one for
// a function with no matmul would be a fabricated claim.
//
// MULTI: hmx.kernel_record/v3" = {
// MULTI-DAG: function = "alpha"
// MULTI-DAG: function = "beta"
// MULTI-DAG: function = "library_kernel"
// MULTI-DAG: id = 0 : i64
// MULTI: func.func @beta
// MULTI: func.func @alpha
// MULTI: func.func @library_kernel
// MULTI: func.func @no_matmul
module attributes {hmx.diagnostic_v3_record} {
  func.func @beta(%a: tensor<32x64xf16>, %b: tensor<64x32xf16>) -> tensor<32x32xf16> {
    %c = tensor.empty() : tensor<32x32xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<32x64xf16>, tensor<64x32xf16>) outs(%c : tensor<32x32xf16>) -> tensor<32x32xf16>
    return %0 : tensor<32x32xf16>
  }
  func.func @alpha(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
  func.func @library_kernel(%a: tensor<8x8xf16>, %b: tensor<8x8xf16>) -> tensor<8x8xf16> {
    %c = tensor.empty() : tensor<8x8xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<8x8xf16>, tensor<8x8xf16>) outs(%c : tensor<8x8xf16>) {library_call = "external"} -> tensor<8x8xf16>
    return %0 : tensor<8x8xf16>
  }
  func.func @no_matmul(%a: memref<64xf16>) {
    return
  }
}

// -----

// Without the marker the producer is inert.  No document is published, rather
// than an empty one a consumer could misread as "no proofs exist".
//
// INERT-NOT: hmx.kernel_record/v3
// INERT: hmx.kernel_manifest = {
// INERT-DAG: schema = "hex.hmx.kernel_manifest/v2"
module {
  func.func @plain(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// With the record-mode marker present, the v2 manifest is still published and is
// still the execution authority; the v3 document sits beside it as a sibling
// record, never a replacement.  This split only shows the *marked* module -- it
// compares nothing.  The marker-on/off A/B and the generated-code invariance
// claim live in the host test, which can compile both variants and compare them:
// `test_hmx_record_v3.py::CppRoundTripTest::
// test_the_v2_execution_child_is_identical_in_both_envelopes` (the v2 children)
// and `::GeneratedCodeInvarianceTest::
// test_marked_and_unmarked_modules_lower_to_identical_code` (the lowered IR).
// The marker-off case is the `INERT` split above.
//
// V2: hmx.kernel_manifest = {
// V2-DAG: schema = "hex.hmx.kernel_manifest/v2"
// V2-DAG: function = "matmul_kernel"
// V2-DAG: plan = "hvx"
// V2-DAG: reason = "vtcm-allocator-disabled"
// V2: hmx.kernel_record/v3" = {
module attributes {hmx.diagnostic_v3_record} {
  func.func @matmul_kernel(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}
