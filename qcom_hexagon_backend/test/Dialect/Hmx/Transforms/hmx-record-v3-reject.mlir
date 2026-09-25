//===- hmx-record-v3-reject.mlir - record-only v3 fails closed ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// `hex.hmx.kernel_manifest/v3` is a closed contract.  Every case here is a
// refusal: an unknown or malformed field never becomes a default, a not-proven
// axis never becomes a zero, and a requested record document that cannot be
// proven fails the module instead of being published in reduced form.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

// The mode marker is presence- and type-checked.  A marker carrying a payload is
// malformed diagnostic IR, not an implicit opt-in into a record-only contract.
// expected-error @+1 {{hmx.diagnostic_v3_record must be a unit attribute}}
module attributes {hmx.diagnostic_v3_record = "yes"} {
  func.func @bad_marker(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// A record document without the mode marker is a forged or stale artifact.  The
// producer leaves it alone (there was nothing to produce) and the finalizer
// rejects it, so nobody can later read it as a proven record.
// expected-error @+1 {{hmx.kernel_record/v3 is present without hmx.diagnostic_v3_record}}
module attributes {"hmx.kernel_record/v3" = {admission = "not-authorized", record_mode = "record-only", records = [], schema = "hex.hmx.kernel_manifest/v3"}} {
  func.func @stale_document(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// A staging dictionary carrying anything but `records` is not a staging
// dictionary.  Accepting an unknown top-level key would be how a v4 field would
// leak into a v3 document unnoticed.  (A *finalized* document is refused for a
// different reason: restaging it would invalidate stored fingerprints.)
// expected-error @+1 {{hmx.kernel_record/v3 is not a well-formed staging dictionary}}
module attributes {hmx.diagnostic_v3_record, "hmx.kernel_record/v3" = {records = [], schema = "hex.hmx.kernel_manifest/v3"}} {
  func.func @stale_staging(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// A pre-staged record is a forged proof: it claims requested bytes and a
// `complete` liveness proof before the sidecars that could prove them exist.
// Finalization validates every record against the closed contract, so a value in
// a `not-proven` block is rejected rather than republished.
// expected-error @+1 {{the HMX v3 record document failed its own validation}}
module attributes {
  hmx.diagnostic_v3_record,
  "hmx.kernel_record/v3" = {records = [{
    fallback = {on_malformed_record = "reject-v3-record"},
    function = "forged",
    id = 0 : i64,
    plan = "full-hmx",
    proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}},
    resources = {
      allocator_aligned = {basis = "allocator-model", modeled_aligned_peak_bytes, resident_aligned_bytes, status = "not-proven", transient_aligned_peak_bytes, unit = "bytes"},
      observed_high_water = {basis = "runtime-observation", scope = "process-high-water", source, status = "not-proven", unit = "bytes", value_bytes},
      requested = {basis = "compile-time-requested", modeled_requested_peak_bytes = 4096 : i64, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}},
    scope = {function = "forged", grid = {policy = "single-instance", required_product = 1 : i64}, invocations = 1 : i64, resident = "process-floor"},
    shape = {logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 64 : i64}, n = {kind = "static", value = 64 : i64}}, specialization = "upstream-static", state = "static"}}]}} {
  func.func @forged() {
    return
  }
}
