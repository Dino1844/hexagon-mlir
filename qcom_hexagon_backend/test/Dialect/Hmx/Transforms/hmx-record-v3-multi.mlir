//===- hmx-record-v3-multi.mlir - two matmuls in one function ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// One function, two `linalg.matmul`s, two v3 records.
//
// This is the case that separates "has no evidence" from "is not published".
// The per-function requested-byte bound the P1.5 sidecars can prove belongs to
// the *function*, so it is attributable to a record only when the function has
// exactly one.  Here it has two, so neither record may claim a capacity -- but
// both must still be published, each with its own digest.  Skipping the digest
// along with the evidence would make the document unpublishable, and dropping a
// record to keep the document valid would be a silent zip loss.
//
// Each claim gets its own FileCheck prefix so its scan starts from the top of
// the output: both records live on one line, and a shared scan would let one
// directive's position decide whether the next one still sees its match.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=SHAPES
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=DIGESTS
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=V2
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=NOEVIDENCE
//===----------------------------------------------------------------------===//

// Both records are published, with distinct shapes and ids.
// SHAPES: "hmx.kernel_record/v3" = {
// SHAPES-DAG: admission = "not-authorized"
// SHAPES-DAG: record_mode = "record-only"
// SHAPES-DAG: schema = "hex.hmx.kernel_manifest/v3"
// SHAPES-DAG: function = "two"
// SHAPES-DAG: id = 0 : i64
// SHAPES-DAG: id = 1 : i64
// SHAPES-DAG: shape = {logical = {k = {kind = "static", value = 64 : i64}, m = {kind = "static", value = 64 : i64}, n = {kind = "static", value = 64 : i64}}, specialization = "upstream-static", state = "static"}
// SHAPES-DAG: shape = {logical = {k = {kind = "static", value = 32 : i64}, m = {kind = "static", value = 32 : i64}, n = {kind = "static", value = 32 : i64}}, specialization = "upstream-static", state = "static"}

// Both carry a digest, and neither claims a capacity, because the
// function-level bound cannot be split between them without inventing an
// attribution.  The three COUNT directives are in the order the fields appear
// inside a record, because a COUNT advances the scan past its own match.
// DIGESTS: "hmx.kernel_record/v3" = {
// DIGESTS-COUNT-2: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
// DIGESTS-COUNT-2: record_fingerprint = "sha256:
// DIGESTS-COUNT-2: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}

// The v2 manifest is unaffected: it still owns both records, and it remains the
// execution authority.
// V2: hmx.kernel_manifest = {
// V2-DAG: schema = "hex.hmx.kernel_manifest/v2"
// V2-DAG: function = "two"
// V2-DAG: id = 0 : i64
// V2-DAG: id = 1 : i64
// V2-DAG: shape_state = "static"
// V2: func.func @two
module attributes {hmx.diagnostic_v3_record} {
  func.func @two(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                 %c: tensor<32x32xf16>, %d: tensor<32x32xf16>)
      -> (tensor<64x64xf16>, tensor<32x32xf16>) {
    %e0 = tensor.empty() : tensor<64x64xf16>
    %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
    %e1 = tensor.empty() : tensor<32x32xf16>
    %m1 = linalg.matmul ins(%c, %d : tensor<32x32xf16>, tensor<32x32xf16>)
                       outs(%e1 : tensor<32x32xf16>) -> tensor<32x32xf16>
    return %m0, %m1 : tensor<64x64xf16>, tensor<32x32xf16>
  }
}

// -----

// The same two matmuls with the census + liveness sidecars present.  The
// records are still published and still unproven, because the *shape* of the
// join -- one record per function -- is what gates the evidence, not the
// presence of a sidecar.  A sidecar that proves the function's bound proves it
// for the function, and v3 has no way to say which of two records it belongs to.
//
// NOEVIDENCE: "hmx.kernel_record/v3" = {
// NOEVIDENCE-COUNT-2: record_fingerprint = "sha256:
// NOEVIDENCE-COUNT-2: status = "not-proven", transient_requested_peak_bytes
module attributes {
  hmx.diagnostic_v3_record,
  hmx.kernel_vtcm_accounting = {
    allocation_sites = 2 : i64, external_scratch = "none", external_vtcm = "none",
    kind = "allocation-site-census", peak_status = "not-proven",
    raw_site_sum_bytes = 10240 : i64, resident_site_sum_bytes = 0 : i64,
    status = "complete", transient_bytes = 10240 : i64,
    unknown_allocations = 0 : i64, weight_resident_bytes = 0 : i64,
    workspace_resident_bytes = 0 : i64},
  hmx.kernel_vtcm_live_range = {
    allocation_site_coverage = "complete", allocator_peak_status = "not-proven",
    async_policy = "no-summary-incomplete", call_policy = "no-summary-incomplete",
    control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region",
    definition_count = 1 : i64, extent_policy = "no-summary-incomplete",
    external_declaration_count = 0 : i64,
    fragmentation_status = "not-proven", free_cache_retention_status = "not-proven",
    functions = [{allocation_site_coverage = "complete", allocation_sites = 2 : i64,
      deallocation_sites = 2 : i64, fixpoint_rounds = 1 : i64,
      modeled_requested_peak_bytes = 10240 : i64, peak_site_count = 2 : i64,
      peak_site_id_status = "not-proven",
      peak_status = "structured-upper-bound", revised_blocks = 1 : i64,
      status = "complete", symbol = "two",
      transient_requested_peak_bytes = 10240 : i64,
      weight_resident_requested_bytes = 0 : i64,
      workspace_resident_requested_bytes = 0 : i64}],
    grid_status = "not-proven", join_policy = "union-join-upper-bound",
    kind = "structured-allocator-events-v1", resident_runtime_state = "not-proven",
    scope = "per-function-single-invocation", status = "complete",
    unit = "requested-bytes"}} {
  func.func @two(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                 %c: tensor<32x32xf16>, %d: tensor<32x32xf16>)
      -> (tensor<64x64xf16>, tensor<32x32xf16>) {
    %e0 = tensor.empty() : tensor<64x64xf16>
    %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
    %e1 = tensor.empty() : tensor<32x32xf16>
    %m1 = linalg.matmul ins(%c, %d : tensor<32x32xf16>, tensor<32x32xf16>)
                       outs(%e1 : tensor<32x32xf16>) -> tensor<32x32xf16>
    return %m0, %m1 : tensor<64x64xf16>, tensor<32x32xf16>
  }
}
