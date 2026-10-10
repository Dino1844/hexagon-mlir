//===- hmx-record-v3-evidence-pipeline.mlir - enriched record end to end -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The record-only v3 envelope end to end *with* proven requested bytes.
//
// This is the fixture the C++-to-Python fingerprint parity test runs against, so
// it must contain real integer quantities and a real liveness basis: an
// all-`null` record would agree on its digest for reasons that have nothing to
// do with the digest being computed the same way on both sides.
//
// The census and liveness sidecars are written by hand here, exactly as in
// hmx-record-v3-evidence.mlir.  The reason is stated once, in that file: the
// sidecar *producer* is a separate marker-gated diagnostic with its own
// fixtures, and this file is about what v3 does with a proven fact.  What the
// parity test additionally asserts is that these exact attribute values are the
// ones that reach the record, so a change to the join cannot pass unnoticed.
//
// Each claim gets its own FileCheck prefix.  The whole record is printed on one
// line, and a shared DAG group would let one directive's match position decide
// whether the next one still finds its own.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s --check-prefix=V2
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s --check-prefix=PROVEN
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s --check-prefix=UNPROVEN
//===----------------------------------------------------------------------===//

// The v2 execution manifest is unchanged and remains the execution authority.
// V2: hmx.kernel_manifest = {
// V2-DAG: schema = "hex.hmx.kernel_manifest/v2"
// V2-DAG: function = "recorded_weight"
// V2-DAG: id = 0 : i64
// V2-DAG: plan = "full-hmx"
// V2-DAG: reason = "selected-aligned"
// V2-DAG: vtcm_accounting = "bridge-only"

// The record proves the requested-byte bound, and names the sidecar fact it
// consumed.  The process floor stays in its own field: it is not folded into the
// transient peak, and the two still add up to the model bound the sidecar
// published.
// PROVEN: "hmx.kernel_record/v3" = {
// PROVEN-DAG: schema = "hex.hmx.kernel_manifest/v3"
// PROVEN-DAG: record_mode = "record-only"
// PROVEN-DAG: admission = "not-authorized"
// PROVEN-DAG: function = "recorded_weight"
// PROVEN-DAG: id = 0 : i64
// PROVEN-DAG: plan = "full-hmx"
// PROVEN-DAG: liveness = {basis = "structured-requested-upper-bound", status = "complete"}
// PROVEN-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes = 10240 : i64, resident_requested_bytes = 2048 : i64, status = "complete", transient_requested_peak_bytes = 8192 : i64, unit = "bytes"}

// The three axes with no accepted evidence source stay unproven, and the
// allocator model is still `null` even though the liveness axis is complete:
// one proven axis never promotes another.
// UNPROVEN: "hmx.kernel_record/v3" = {
// UNPROVEN-DAG: allocator = {basis, status = "not-proven"}
// UNPROVEN-DAG: grid = {basis, status = "not-proven"}
// UNPROVEN-DAG: resident = {basis, status = "not-proven"}
// UNPROVEN-DAG: allocator_aligned = {basis = "allocator-model", modeled_aligned_peak_bytes, resident_aligned_bytes, status = "not-proven", transient_aligned_peak_bytes, unit = "bytes"}
// UNPROVEN-DAG: observed_high_water = {basis = "runtime-observation", scope = "process-high-water", source, status = "not-proven", unit = "bytes", value_bytes}
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
      status = "complete", symbol = "recorded_weight",
      transient_requested_peak_bytes = 8192 : i64,
      weight_resident_requested_bytes = 2048 : i64,
      workspace_resident_requested_bytes = 0 : i64}],
    grid_status = "not-proven", join_policy = "union-join-upper-bound",
    kind = "structured-allocator-events-v1", resident_runtime_state = "not-proven",
    scope = "per-function-single-invocation", status = "complete",
    unit = "requested-bytes"}} {
  func.func @recorded_weight(%a: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %w = arith.constant dense<1.000000e+00> : tensor<64x64xf16>
    %c0 = tensor.empty() : tensor<64x64xf16>
    %m0 = linalg.matmul ins(%a, %w : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c0 : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m0 : tensor<64x64xf16>
  }
}
