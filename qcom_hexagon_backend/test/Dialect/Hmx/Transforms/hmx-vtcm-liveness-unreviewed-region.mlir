//===- hmx-vtcm-liveness-unreviewed-region.mlir - unreviewed region owner --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A region entry block that carries arguments is only analysed when its owner
// has a reviewed entry rule.  `scf.for` and `scf.while` do; an operation the
// alias lattice has not been reviewed for does not, and `linalg.fill` is the one
// that shows up in every real matmul module -- its block arguments (the
// index-semantics scalars) are not tracked storage, but its region is not a
// reviewed structured region either, so the class stays closed.
//
// The fixture is the real diagnostic chain over a module whose matmul is *not*
// attributed, so a raw `linalg.matmul` and its `linalg.fill` prologue are still
// in the IR when the census runs.  Before the entry rule stopped assuming
// `scf.while`, this module reached the assertion inside `ProvenanceAnalysis`
// instead of reporting the class: a diagnostic that dies on a legal module
// teaches nothing, and a compiler that aborts here publishes no census at all
// for a kernel the engine will never touch.
//
// What is pinned is the fail-closed behaviour, not a new admission:
//  * no crash -- the assertion is checked explicitly, because "the tool did not
//    abort" would otherwise only be implied by the error text matching;
//  * the unreviewed owner is reported by name, in the per-function record the
//    liveness sidecar publishes;
//  * the *census* is still published, because it is a separate surface and this
//    module genuinely has no VTCM allocation site;
//  * the v2 manifest is untouched and stays the execution authority;
//  * the staged v3 record attribution published stays honest `not-proven`, and
//    the finalizer -- which runs after the failed census -- never stamps the
//    document, so nothing here is mistaken for a finished one.
//
// (A failed pass prints the module it was working on as its diagnostic note;
// that is where the sidecars below are read from, as in
// hmx-vtcm-accounting-resident-declaration-mismatch.mlir.  Module attributes
// print in name order, which is the order the checks follow.)
//
// The positive control -- the same chain publishing real requested/liveness
// bytes on modules the analysis does admit -- is
// ../../Conversion/LinalgToLLVM/hmx-vtcm-accounting-v3-record-pipeline.mlir.
//
//===----------------------------------------------------------------------===//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx),hmx-vtcm-accounting,hmx-v3-record)' 2>&1 | FileCheck %s
//===----------------------------------------------------------------------===//

// The attribution pass leaves this matmul alone (M=48 is not a tile multiple),
// so the raw `linalg.matmul` and the `linalg.fill` prologue are still here.
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness,
                    hmx.diagnostic_v3_record} {
  func.func @raw_matmul(%a: tensor<48x64xf16>, %b: tensor<64x64xf16>) -> tensor<48x64xf16> {
    %empty = tensor.empty() : tensor<48x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<48x64xf16>) -> tensor<48x64xf16>
    %m = linalg.matmul ins(%a, %b : tensor<48x64xf16>, tensor<64x64xf16>)
                      outs(%c : tensor<48x64xf16>) -> tensor<48x64xf16>
    return %m : tensor<48x64xf16>
  }
}

// A crash is not a diagnostic, so the assertion this file exists for is a hard
// check rather than an implicit consequence of the error line matching.
// CHECK-NOT: PLEASE submit a bug report
// CHECK-NOT: Assertion

// CHECK: error: HMX VTCM structured liveness is not proven for this IR: region block argument has no reviewed incoming value

// The v2 manifest is the execution authority and is not rewritten by a
// diagnostic; the closed v3 schema stamp belongs to the finalizer, which never
// ran.
// CHECK: schema = "hex.hmx.kernel_manifest/v2"
// CHECK-NOT: schema = "hex.hmx.kernel_manifest/v3"

// The record attribution staged for the unattributed matmul carries no byte
// figure at all: the requested-byte names are published empty next to a
// `not-proven` status, which is the difference between "not measured" and
// "measured zero".
// CHECK: "hmx.kernel_record/v3" = {records = [{fallback = {on_malformed_record = "reject-v3-record"}, function = "raw_matmul", id = 0 : i64, plan = "hvx"
// CHECK-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
// CHECK-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}

// The census is the other surface, and this module really has no tracked VTCM
// allocation site: it stays complete, with a zero site count rather than a
// missing attribute.
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: allocation_sites = 0 : i64
// CHECK-DAG: external_scratch = "none"
// CHECK-DAG: external_vtcm = "none"
// CHECK-DAG: kind = "allocation-site-census"
// CHECK-DAG: status = "complete"
// CHECK-DAG: transient_bytes = 0 : i64
// CHECK-DAG: unknown_allocations = 0 : i64
// CHECK-DAG: weight_resident_bytes = 0 : i64
// CHECK-DAG: workspace_resident_bytes = 0 : i64

// The liveness sidecar keeps the module visible while the function itself is
// reported incomplete, with the reason named rather than a zero peak.
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: functions = [{allocation_site_coverage = "complete", allocation_sites = 0 : i64, reason = "region block argument has no reviewed incoming value", status = "incomplete", symbol = "raw_matmul"}]
// CHECK-DAG: grid_status = "not-proven"
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: resident_runtime_state = "not-proven"
// CHECK-DAG: status = "incomplete"
// CHECK-DAG: unit = "requested-bytes"
