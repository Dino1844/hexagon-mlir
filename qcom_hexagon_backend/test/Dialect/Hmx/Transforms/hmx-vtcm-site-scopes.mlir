//===- hmx-vtcm-site-scopes.mlir - per-canonical-site scope table ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The frame token names one observation. This table names every canonical
// allocation site inside it, each with its own token, derived with the role and
// the checked resident slot in the hash. Two allocations that share a
// file:line:col but differ in role or slot are two sites, and the token is what
// makes that visible on the device.
//
// The multi-site module is the point: the frame context refuses it (it needs
// exactly one site), while the site table is precisely the shape that needs
// per-site names.
//
// The pipeline runs `convert-to-hexagonmem` first, and that is not incidental.
// A standalone space-1 `memref.alloc` is *not yet* a pool-backed allocation:
// it becomes one only when the space-1 allocator conversion has turned it into
// `hexagonmem.alloc`, which is the op the lowering pass actually brackets. The
// producer's pool-backed predicate is what makes the site table meaningful --
// a site the lowering could not bracket must not be named -- so the fixture has
// to reach the same IR shape the real pipeline produces rather than asserting
// against a pre-conversion allocation. Weakening the predicate to accept
// pre-conversion allocations would publish sites that no bracket exists for.
//
// Both sites are transient VTCM allocations at distinct source locations. A
// resident site additionally needs a fully checked provenance descriptor --
// kind, role, function, module, site, byte count, alignment, slot, and the two
// identity hashes -- emitted by the workspace/weight residency passes.
// Hand-typing that into a fixture is how a test ends up asserting against a
// vocabulary the producer never accepted; the role/slot components of the token
// are exercised by the producer's own closed `role` field here, and the
// same-source-distinguished-by-role case belongs to the residency pass that
// emits the provenance.
//
// Marker-gated and diagnostic only: without the marker the pass publishes
// neither the table nor the stamps, so the object is unchanged.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(convert-to-hexagonmem),hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// -----

// Two canonical transient sites in one function. The frame context is
// ineligible here, and that refusal is independent of the site table's own
// verdict: the table does not relax the one-site frame rule, it names sites the
// frame never had to.
module @n2_multi_site attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2":7:1)
    %c = memref.alloc() : memref<64xf16, 1> loc("n2":8:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                                     affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b, %c : memref<64xf16, 1>, memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      linalg.yield %x, %x : f16, f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    memref.copy %c, %a : memref<64xf16, 1> to memref<64xf16>
    return
  }
}

// The census is complete: two sites, both proven, no external VTCM. The site
// table is only meaningful on top of that.
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: external_scratch = "none"
// CHECK-DAG: external_vtcm = "none"
// CHECK-DAG: status = "complete"
// CHECK-DAG: transient_bytes = 256 : i64

// The frame context still refuses two sites.
// CHECK: hmx.kernel_vtcm_event_context = {
// CHECK-DAG: eligible = false
// CHECK-DAG: reason = "event context requires one function and one canonical site"

// The table is the closed eligible shape. `site_count` counts canonical sites,
// not allocations and not traversal positions.
// CHECK: hmx.kernel_vtcm_site_scopes = {
// CHECK-DAG: kind = "vtcm-site-scopes"
// CHECK-DAG: schema = "hmx.vtcm-site-scopes/v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: eligible = true
// CHECK-DAG: mode = "diagnostic-only"
// CHECK-DAG: principal = "n2_multi_site"
// CHECK-DAG: module = "n2_multi_site"
// CHECK-DAG: function = "kernel"
// CHECK-DAG: function_id = {{-?[0-9]+}} : i64
// CHECK-DAG: accounting_scope_id = {{-?[0-9]+}} : i64
// CHECK-DAG: build_id_low = 5 : i64
// CHECK-DAG: build_id_high = 9 : i64
// CHECK-DAG: grid_product = 1 : i64
// CHECK-DAG: invocation_id = 1 : i64
// CHECK-DAG: immutable = 1 : i64
// CHECK-DAG: token_basis = "hmx.vtcm-site-token/fnv1a128/v1"
// CHECK-DAG: site_count = 2 : i64
// CHECK-DAG: performance_claimed = 0 : i64
// An eligible table carries a `sites` array and no `reason`.
// CHECK-DAG: sites = [
// CHECK-DAG: constant_bounded_extent = 0 : i64
// CHECK-DAG: function = "kernel"
// CHECK-DAG: requested_bytes = 128 : i64
// CHECK-DAG: role = "transient-hexagonmem"
// CHECK-DAG: slot = 0 : i64
// CHECK-DAG: source = "file:n2:7:1"
// CHECK-DAG: token_basis = "hmx.vtcm-site-token/fnv1a128/v1"
// CHECK-DAG: token_bits = 128 : i64
// CHECK-DAG: token_high = {{-?[0-9]+}} : i64
// CHECK-DAG: token_low = {{-?[0-9]+}} : i64
// CHECK-DAG: site_id = {{-?[0-9]+}} : i64
// The second entry is the other site, with its own id and its own token.
// CHECK-DAG: source = "file:n2:8:1"
// CHECK-NOT: reason =

// Each pool-backed allocation carries the closed per-op record with its own
// words, so the lowering pass can bracket one allocation with the token that
// belongs to it and not to the other.
// CHECK: hmx.vtcm_site_scope = {
// CHECK-DAG: kind = "vtcm-site-scope"
// CHECK-DAG: schema = "hmx.vtcm-site-scopes/v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: role = "transient-hexagonmem"
// CHECK-DAG: slot = 0 : i64
// CHECK-DAG: source = "file:n2:7:1"
// CHECK-DAG: function = "kernel"
// CHECK-DAG: function_id = {{-?[0-9]+}} : i64
// CHECK-DAG: site_id = {{-?[0-9]+}} : i64
// CHECK-DAG: accounting_scope_id = {{-?[0-9]+}} : i64
// CHECK-DAG: token_basis = "hmx.vtcm-site-token/fnv1a128/v1"
// CHECK-DAG: token_bits = 128 : i64
// CHECK-DAG: token_low = {{-?[0-9]+}} : i64
// CHECK-DAG: token_high = {{-?[0-9]+}} : i64
// CHECK-DAG: build_id_low = 5 : i64
// CHECK-DAG: build_id_high = 9 : i64
// CHECK-DAG: invocation_id = 1 : i64
// CHECK-DAG: grid_product = 1 : i64
// CHECK: hmx.vtcm_site_scope = {
// CHECK-DAG: source = "file:n2:8:1"
// CHECK-NOT: 0x
// CHECK-NOT: ptr=

// -----

// A module with no canonical site is refused, and no op is stamped. A partial
// stamp set would let a consumer believe the unmarked sites were outside the
// scope.
// CHECK: hmx.kernel_vtcm_site_scopes = {
// CHECK-DAG: eligible = false
// CHECK-DAG: status = "not-proven"
// CHECK-DAG: site_count = 0 : i64
// CHECK-DAG: reason = "site scopes require at least one canonical site"
// CHECK-NOT: hmx.vtcm_site_scope
// CHECK-NOT: sites =
module @n2_ddr_site attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16> loc("n2":7:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b : memref<64xf16>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
    }
    return
  }
}

// -----

// Without the site marker nothing is published and nothing is stamped, so the
// default object is unchanged.
// CHECK-LABEL: module @n2_site_unmarked
// CHECK-NOT: hmx.kernel_vtcm_site_scopes
// CHECK-NOT: hmx.vtcm_site_scope
module @n2_site_unmarked attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2":7:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b : memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    return
  }
}
