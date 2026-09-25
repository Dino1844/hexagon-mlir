//===- hmx-record-v3-evidence.mlir - requested bytes from P1.5 sidecars --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The versioned mapping from the P1.5 diagnostic sidecars onto the v3 record,
// for the two cases that matter: the facts the structured liveness analysis can
// prove, and the three axes that stay unproven no matter what the process-level
// capture shows.
//
// The sidecars are written by hand here on purpose.  This file is about the v3
// contract -- what the record is allowed to claim and what it must keep
// refusing to claim -- not about the P1.5 producer, which has its own fixtures.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=PROVEN
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=PARTIAL
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{record-only=true}),hmx-v3-record)' -split-input-file | FileCheck %s --check-prefix=NOSCOPE
//===----------------------------------------------------------------------===//

// A complete structured liveness analysis for the one function in the module
// moves exactly two things: the `requested` capacity block and the `liveness`
// proof.  Everything else is untouched -- `allocator_aligned` and
// `observed_high_water` have no accepted evidence source, so they stay
// `not-proven` with `null` quantities, and `allocator`, `grid` and `resident`
// keep the `not-proven` the sidecar itself publishes for them.
//
// `resident_requested_bytes` is a process floor, kept in its own field: it is not
// folded into the transient peak, and the two still add up to the model bound.
//
// PROVEN: hmx.kernel_record/v3" = {
// PROVEN-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes = 10240 : i64, resident_requested_bytes = 2048 : i64, status = "complete", transient_requested_peak_bytes = 8192 : i64, unit = "bytes"}
// PROVEN-DAG: allocator_aligned = {basis = "allocator-model", modeled_aligned_peak_bytes, resident_aligned_bytes, status = "not-proven", transient_aligned_peak_bytes, unit = "bytes"}
// PROVEN-DAG: observed_high_water = {basis = "runtime-observation", scope = "process-high-water", source, status = "not-proven", unit = "bytes", value_bytes}
// PROVEN-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis = "structured-requested-upper-bound", status = "complete"}, resident = {basis, status = "not-proven"}}
// PROVEN-DAG: record_fingerprint = "sha256:
module attributes {
  hmx.diagnostic_v3_record,
  hmx.kernel_vtcm_accounting = {
    allocation_sites = 3 : i64, external_scratch = "none", external_vtcm = "none",
    kind = "allocation-site-census", peak_status = "not-proven",
    raw_site_sum_bytes = 10240 : i64, resident_site_sum_bytes = 2048 : i64,
    status = "complete", transient_bytes = 8192 : i64,
    unknown_allocations = 0 : i64, weight_resident_bytes = 2048 : i64,
    workspace_resident_bytes = 0 : i64},
  hmx.kernel_vtcm_live_range = {
    allocation_site_coverage = "complete", allocator_peak_status = "not-proven",
    async_policy = "no-summary-incomplete", call_policy = "no-summary-incomplete",
    control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region",
    definition_count = 1 : i64, extent_policy = "no-summary-incomplete",
    external_declaration_count = 0 : i64,
    fragmentation_status = "not-proven", free_cache_retention_status = "not-proven",
    functions = [{allocation_site_coverage = "complete", allocation_sites = 3 : i64,
      deallocation_sites = 2 : i64, fixpoint_rounds = 1 : i64,
      modeled_requested_peak_bytes = 10240 : i64, peak_site_count = 3 : i64,
      peak_site_id_status = "not-proven",
      peak_status = "structured-upper-bound", revised_blocks = 1 : i64,
      status = "complete", symbol = "matmul_kernel",
      transient_requested_peak_bytes = 8192 : i64,
      weight_resident_requested_bytes = 2048 : i64,
      workspace_resident_requested_bytes = 0 : i64}],
    grid_status = "not-proven", join_policy = "union-join-upper-bound",
    kind = "structured-allocator-events-v1", resident_runtime_state = "not-proven",
    scope = "per-function-single-invocation", status = "complete",
    unit = "requested-bytes"}} {
  func.func @matmul_kernel(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// An incomplete liveness analysis publishes no number at all.  The record keeps
// the `not-proven` status and the `null` quantities rather than publishing the
// partial figure it does have: a half-measured capacity is indistinguishable
// from a small one at the consumer, and the four axes exist so that gap stays
// visible instead of being rounded away.
//
// PARTIAL: hmx.kernel_record/v3" = {
// PARTIAL-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}
// PARTIAL-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
module attributes {
  hmx.diagnostic_v3_record,
  hmx.kernel_vtcm_accounting = {
    allocation_sites = 3 : i64, external_scratch = "none", external_vtcm = "none",
    kind = "allocation-site-census", peak_status = "not-proven",
    raw_site_sum_bytes = 10240 : i64, resident_site_sum_bytes = 2048 : i64,
    status = "complete", transient_bytes = 8192 : i64,
    unknown_allocations = 0 : i64, weight_resident_bytes = 2048 : i64,
    workspace_resident_bytes = 0 : i64},
  hmx.kernel_vtcm_live_range = {
    allocation_site_coverage = "incomplete", allocator_peak_status = "not-proven",
    async_policy = "no-summary-incomplete", call_policy = "no-summary-incomplete",
    control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region",
    definition_count = 1 : i64, extent_policy = "no-summary-incomplete",
    external_declaration_count = 0 : i64,
    fragmentation_status = "not-proven", free_cache_retention_status = "not-proven",
    functions = [{allocation_site_coverage = "incomplete", allocation_sites = 3 : i64,
      reason = "unmodelled-region", status = "incomplete", symbol = "matmul_kernel"}],
    grid_status = "not-proven", join_policy = "union-join-upper-bound",
    kind = "structured-allocator-events-v1", resident_runtime_state = "not-proven",
    scope = "per-function-single-invocation", status = "incomplete",
    unit = "requested-bytes"}} {
  func.func @matmul_kernel(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// The join must be exact in unit, scope and symbol.  A sidecar that describes a
// different scope is not evidence about this record's scope, so nothing is
// published -- the record does not narrow the claim to "it matched, sort of".
//
// NOSCOPE: hmx.kernel_record/v3" = {
// NOSCOPE-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}
// NOSCOPE-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
module attributes {
  hmx.diagnostic_v3_record,
  hmx.kernel_vtcm_accounting = {
    allocation_sites = 3 : i64, external_scratch = "none", external_vtcm = "none",
    kind = "allocation-site-census", peak_status = "not-proven",
    raw_site_sum_bytes = 10240 : i64, resident_site_sum_bytes = 2048 : i64,
    status = "complete", transient_bytes = 8192 : i64,
    unknown_allocations = 0 : i64, weight_resident_bytes = 2048 : i64,
    workspace_resident_bytes = 0 : i64},
  hmx.kernel_vtcm_live_range = {
    allocation_site_coverage = "complete", allocator_peak_status = "not-proven",
    async_policy = "no-summary-incomplete", call_policy = "no-summary-incomplete",
    control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region",
    definition_count = 1 : i64, extent_policy = "no-summary-incomplete",
    external_declaration_count = 0 : i64,
    fragmentation_status = "not-proven", free_cache_retention_status = "not-proven",
    functions = [{allocation_site_coverage = "complete", allocation_sites = 3 : i64,
      deallocation_sites = 2 : i64, fixpoint_rounds = 1 : i64,
      modeled_requested_peak_bytes = 10240 : i64, peak_site_count = 3 : i64,
      peak_site_id_status = "not-proven",
      peak_status = "structured-upper-bound", revised_blocks = 1 : i64,
      status = "complete", symbol = "matmul_kernel",
      transient_requested_peak_bytes = 8192 : i64,
      weight_resident_requested_bytes = 2048 : i64,
      workspace_resident_requested_bytes = 0 : i64}],
    grid_status = "not-proven", join_policy = "union-join-upper-bound",
    kind = "structured-allocator-events-v1", resident_runtime_state = "not-proven",
    scope = "per-function-whole-process", status = "complete",
    unit = "requested-bytes"}} {
  func.func @matmul_kernel(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}
