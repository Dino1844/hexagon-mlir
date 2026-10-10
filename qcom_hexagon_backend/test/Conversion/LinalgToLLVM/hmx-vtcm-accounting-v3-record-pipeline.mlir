//===- hmx-vtcm-accounting-v3-record-pipeline.mlir - real sidecars to record -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The sidecar-to-record mapping fed by the *producers* instead of by attributes
// typed into the test.  hmx-record-v3-evidence.mlir pins the same join against
// hand-written dictionaries; this file runs the real `linalg-to-llvm` pipeline,
// which is where `matmul-to-hmx` attributes the matmuls and where
// `hmx-vtcm-accounting` writes the census and liveness sidecars immediately
// before `hmx-v3-record` finalizes the document.  The point is that the bytes
// the record carries are the ones the analysis derived from this IR: a fixture
// that typed the sidecars in could not tell a real join from a copy.
//
// Four modules, four properties:
//
//  1. One attributed matmul with both census markers.  The census finds the four
//     VTCM allocation sites the tile level emits (the bias state, the
//     activation bridge, the resident weight, the accumulator), and the
//     structured liveness analysis closes every transient one of them inside the
//     function, so the record publishes a `complete` peak.  The figure is real
//     -- it moves with the shape -- and a silent zero would satisfy every status
//     check, so the exact bytes are pinned.
//
//  2. Two attributed matmuls in one function, the second reading the first's
//     result so neither is dead code.  The per-function sidecars carry the real
//     totals for both (six sites, two resident weights), while each *record*
//     stays `not-proven`: the sidecar's number belongs to the function, and
//     splitting it across two records would be a guess.  That is the
//     multi-record rule, seen with real evidence instead of a synthetic one.
//
//  3. The same kernel with only the record marker.  Without the census and
//     liveness markers their producers stay inert, so there is nothing to join
//     and the record must publish no requested byte at all -- the three names
//     are present but empty next to `not-proven`.  A record that derived a
//     number from the manifest alone would look like case 1 from outside.
//
//  4. Case 1 with an f32 weight: the resident contract admits it (the host
//     quantises the crouton image), so the accounting pass's own copy of that
//     admission must admit it too.  Before that copy was widened, this module
//     failed with "HMX VTCM accounting is incomplete ..." even though the IR
//     came out of this very pipeline.  The f16 byte figures are unchanged --
//     the crouton is fp16 either way -- so only the policy reason and the
//     source dtype differ.
//
// Nothing here changes the execution contract: the v2 manifest is still the
// authority and the v3 document is still `admission = "not-authorized"`.
//
// The fail-closed counterpart -- a module the liveness analysis does not admit,
// which must report rather than abort -- is
// ../../Dialect/Hmx/Transforms/hmx-vtcm-liveness-unreviewed-region.mlir.
// The sidecar-only view of the same pipeline (no v3 document, workspace
// residency on) is hmx-vtcm-liveness-pipeline.mlir.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record{production=enable-workspace-resident=false})' -split-input-file | FileCheck %s
//
// LAYERED arm: the same 102 directives against the same entry stopped right
// after the placement-stage census. Everything these four modules pin is
// produced inside that layer -- `matmul-to-hmx` writes the manifest,
// `weight-resident` decides the policy, the census and the liveness analysis
// write their sidecars, and `hmx-v3-record` finalises the document from them
// on the other side of the stage hook. So the identical CHECK lines hold on
// the stage output alone, and the arm doubles as the packet-census -> record
// join's own layer test: if the join ever started depending on something only
// the lowering produces (or only the pre-placement IR has), this run goes red
// while the one above stays green.
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-diagnostic-record{production=enable-workspace-resident=false stop-after-diagnostic-stage})' | FileCheck %s
//===----------------------------------------------------------------------===//

// -----

// One attributed 64x64x64 matmul.  The weight is a runtime argument, so the
// weight-resident pass turns it into a resident VTCM buffer and the kernel drops
// its per-launch pack; the census sees that buffer as a resident site and the
// bias state, the folded serial path's one-row activation scratch and the
// accumulator as transient ones (S2.5: the whole 2x2 activation array, 8192
// bytes, is retired by the serial pack fold and the 4096-byte scratch takes
// its place, so every transient figure below is 4096 less than the whole-array
// form).
//
// The workspace-resident option is spelled out as false because this fixture's
// arithmetic is the MIXED one -- a resident weight against three transient
// workspace sites -- and the all-resident world (the default since 2026-10-04)
// zeroes the transient side of every sum below, which would stop checking the
// census math rather than check it harder. The resident world has its own
// coverage (hmx-workspace-resident-pipeline.mlir).
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness,
                    hmx.diagnostic_v3_record} {
  func.func @one_matmul(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}

// The v2 manifest is what the launcher reads, and it still shows the runtime
// weight as resident: the kernel's own pack_weight site is gone.
// CHECK: hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes
// CHECK-DAG: pack_weight_sites = 0 : i64
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: reason = "selected-aligned"
// CHECK-DAG: weight_policies = [{consumers = [0], function = "one_matmul", policy = "resident-prepack", reason = "eligible-aligned-f16", slot = 1 : i64}]
// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v2"

// The finalized document: record-only, authorizes nothing, and the single
// record the function owns takes exactly the sidecar's numbers.
// CHECK: "hmx.kernel_record/v3" = {admission = "not-authorized", record_mode = "record-only", records = [{fallback = {on_malformed_record = "reject-v3-record"}, function = "one_matmul", id = 0 : i64, plan = "full-hmx"
// CHECK-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis = "structured-requested-upper-bound", status = "complete"}, resident = {basis, status = "not-proven"}}
// `allocator_aligned` and `observed_high_water` have no accepted evidence
// source, so they stay empty rather than borrowing the requested figure.
// CHECK-DAG: allocator_aligned = {basis = "allocator-model", modeled_aligned_peak_bytes, resident_aligned_bytes, status = "not-proven", transient_aligned_peak_bytes, unit = "bytes"}
// CHECK-DAG: observed_high_water = {basis = "runtime-observation", scope = "process-high-water", source, status = "not-proven", unit = "bytes", value_bytes}
// CHECK-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes = 20736 : i64, resident_requested_bytes = 8192 : i64, status = "complete", transient_requested_peak_bytes = 12544 : i64, unit = "bytes"}
// CHECK-DAG: scope = {function = "one_matmul", grid = {policy = "single-instance", required_product = 1 : i64}, invocations = 1 : i64, resident = "process-floor"}
// CHECK-DAG: shape

// The census, straight from the real analysis: four sites, the resident weight
// split out from the transient sum, and no unmodelled allocation.  Its
// `peak_status` is the raw surface's own and stays `not-proven`; the peak below
// comes from the liveness sidecar, not from here.
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: allocation_sites = 4 : i64
// CHECK-DAG: external_scratch = "none"
// CHECK-DAG: external_vtcm = "none"
// CHECK-DAG: kind = "allocation-site-census"
// CHECK-DAG: peak_status = "not-proven"
// CHECK-DAG: raw_site_sum_bytes = 20736 : i64
// CHECK-DAG: resident_site_sum_bytes = 8192 : i64
// CHECK-DAG: status = "complete"
// CHECK-DAG: transient_bytes = 12544 : i64
// CHECK-DAG: unknown_allocations = 0 : i64
// CHECK-DAG: weight_resident_bytes = 8192 : i64
// CHECK-DAG: workspace_resident_bytes = 0 : i64

// The liveness sidecar.  The three transient sites are the bias state, the
// folded serial path's one-row activation scratch and the accumulator
// (256 + 4096 + 8192), and all three are live at the peak, so
// `transient_requested_peak_bytes` reaches the census sum.
// `modeled_requested_peak_bytes` adds the resident weight, which is never
// released inside the function: four sites, three release events.
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: allocator_peak_status = "not-proven"
// CHECK-DAG: fragmentation_status = "not-proven"
// CHECK-DAG: functions = [{aligned_charge_basis = "runtime-size-quantum-no-header-no-address-padding", allocation_site_coverage = "complete", allocation_sites = 4 : i64, constant_bounded_extent_sites = 0 : i64, deallocation_sites = 3 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 20736 : i64
// CHECK-DAG: peak_site_count = 3 : i64
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: status = "complete", symbol = "one_matmul"
// CHECK-DAG: transient_requested_peak_bytes = 12544 : i64
// CHECK-DAG: weight_resident_requested_bytes = 8192 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 0 : i64}]
// CHECK-DAG: grid_status = "not-proven"
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: resident_runtime_state = "not-proven"
// CHECK-DAG: scope = "per-function-single-invocation"
// CHECK-DAG: status = "complete"
// CHECK-DAG: unit = "requested-bytes"

// -----

// Two attributed matmuls in one function.
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness,
                    hmx.diagnostic_v3_record} {
  func.func @two_matmuls(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c0 = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c0 : tensor<64x64xf16>) -> tensor<64x64xf16>
    %c1 = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m1 = linalg.matmul ins(%m0, %a : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c1 : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m1 : tensor<64x64xf16>
  }
}

// Both matmuls are attributed: the manifest carries two decision ids and the
// document two records, for one function.
// CHECK: hmx.kernel_manifest = {
// CHECK-DAG: id = 0 : i64
// CHECK-DAG: id = 1 : i64
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v2"
// CHECK: "hmx.kernel_record/v3" = {admission = "not-authorized", record_mode = "record-only", records = [{fallback = {on_malformed_record = "reject-v3-record"}, function = "two_matmuls", id = 0 : i64
// CHECK-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
// CHECK-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}
// CHECK-DAG: function = "two_matmuls", id = 1 : i64, plan = "full-hmx"
// CHECK-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
// CHECK-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}

// The sidecars, meanwhile, do carry the real figures for the whole function:
// six sites -- the shared bias state, three crouton arrays (the first
// matmul's folded one-row scratch, the first matmul's accumulator -- which is
// the second matmul's chained activation, so it never folds -- and the second
// matmul's accumulator), two resident weights. The first matmul's bridge
// folds (S2.5), so its activation is the 4096-byte scratch, retired with the
// first tile loop; the second matmul's activation is the first's read-out
// array and has no bridge to fold. Four sites are transient: the census sums
// them to 20736 bytes, while the liveness peak sees the scratch die before
// the second accumulator appears and peaks at 16640 -- the two derivations
// agreed on the whole-array form (both 24832) and now honestly disagree,
// because the fold made the first activation short-lived. The resident floor
// brings the model bound to 33024. The *record* is still the only surface
// that must not spread one function's number over its two records.
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: allocation_sites = 6 : i64
// CHECK-DAG: status = "complete"
// CHECK-DAG: transient_bytes = 20736 : i64
// CHECK-DAG: weight_resident_bytes = 16384 : i64
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: allocation_site_coverage = "complete"
// `aligned_charge_basis` sorts first in the function record, so it leads the
// dictionary. The size-aligned figures are sidecar-only: the v3 record below
// still refuses an allocator claim.
// CHECK-DAG: functions = [{aligned_charge_basis = "runtime-size-quantum-no-header-no-address-padding", allocation_site_coverage = "complete", allocation_sites = 6 : i64
// CHECK-DAG: deallocation_sites = 4 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 33024 : i64
// CHECK-DAG: peak_site_count = 3 : i64
// CHECK-DAG: status = "complete", symbol = "two_matmuls"
// CHECK-DAG: transient_requested_peak_bytes = 16640 : i64
// CHECK-DAG: weight_resident_requested_bytes = 16384 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 0 : i64}]
// CHECK-DAG: status = "complete"
// CHECK-DAG: unit = "requested-bytes"

// -----

// The same kernel with only the record marker.
module attributes {hmx.diagnostic_v3_record} {
  func.func @no_sidecars(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}

// The attribution manifest is still there -- the record's shape and plan facts
// come from it -- and the census/liveness producers never ran.
// CHECK: hmx.kernel_manifest = {
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v2"
// CHECK-NOT: hmx.kernel_vtcm_accounting
// CHECK-NOT: hmx.kernel_vtcm_live_range
// So the record is built, and says nothing about bytes.
// CHECK: "hmx.kernel_record/v3" = {admission = "not-authorized", record_mode = "record-only", records = [{fallback = {on_malformed_record = "reject-v3-record"}, function = "no_sidecars", id = 0 : i64, plan = "full-hmx"
// CHECK-DAG: proofs = {allocator = {basis, status = "not-proven"}, grid = {basis, status = "not-proven"}, liveness = {basis, status = "not-proven"}, resident = {basis, status = "not-proven"}}
// CHECK-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes, resident_requested_bytes, status = "not-proven", transient_requested_peak_bytes, unit = "bytes"}
// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v3"

// -----

// Case 1 with an f32 weight.  The manifest names the quantising policy, the
// weight pack site is still gone, and the census/liveness/record figures are
// the f16 ones: the resident crouton is fp16 whatever the source dtype is.
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness,
                    hmx.diagnostic_v3_record} {
  func.func @one_matmul_f32(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %empty = tensor.empty() : tensor<64x64xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %c = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
                      outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %m : tensor<64x64xf32>
  }
}

// Attributes print alphabetically, so the checks follow the output order:
// manifest, record, census, liveness.
// CHECK: hmx.kernel_manifest = {count_semantics = "ir_sites", matmuls = [{dtypes = {crouton = "f16", lhs = "f32", out = "f32", rhs = "f32"}
// CHECK-DAG: pack_weight_sites = 0 : i64
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: weight_policies = [{consumers = [0], function = "one_matmul_f32", policy = "resident-prepack", reason = "eligible-quantized-f32", slot = 1 : i64}]
// CHECK-DAG: schema = "hex.hmx.kernel_manifest/v2"

// The record takes the sidecar's numbers, exactly as the f16 case does (the
// folded-serial figures: the whole activation array is retired by the serial
// pack fold, so the transient side is 4096 less than the whole-array form).
// CHECK: "hmx.kernel_record/v3" = {admission = "not-authorized", record_mode = "record-only", records = [{fallback = {on_malformed_record = "reject-v3-record"}, function = "one_matmul_f32", id = 0 : i64, plan = "full-hmx"
// CHECK-DAG: requested = {basis = "compile-time-requested", modeled_requested_peak_bytes = 20736 : i64, resident_requested_bytes = 8192 : i64, status = "complete", transient_requested_peak_bytes = 12544 : i64, unit = "bytes"}

// The census admits the f32 resident source and closes every transient site.
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: allocation_sites = 4 : i64
// CHECK-DAG: raw_site_sum_bytes = 20736 : i64
// CHECK-DAG: resident_site_sum_bytes = 8192 : i64
// CHECK-DAG: status = "complete"
// CHECK-DAG: transient_bytes = 12544 : i64
// CHECK-DAG: unknown_allocations = 0 : i64
// CHECK-DAG: weight_resident_bytes = 8192 : i64

// And so does the liveness sidecar.
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: modeled_requested_peak_bytes = 20736 : i64
// CHECK-DAG: status = "complete", symbol = "one_matmul_f32"
// CHECK-DAG: transient_requested_peak_bytes = 12544 : i64
// CHECK-DAG: weight_resident_requested_bytes = 8192 : i64
