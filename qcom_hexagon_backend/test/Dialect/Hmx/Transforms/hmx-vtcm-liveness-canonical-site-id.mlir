//===- hmx-vtcm-liveness-canonical-site-id.mlir - canonical peak site join --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Two properties are pinned here.
//
// First, peak attribution is by *canonical static site identity*, never by a
// census walk ordinal.  A walk index changes when an unrelated module is
// reordered, so publishing it as a site ID would make the record unstable; the
// IDs below are the versioned `hmx.vtcm-static-identity/v1` values, captured out
// of the identity sidecar and matched against the liveness record.  Note that
// the census site count and the identity site count both appear, so a reader can
// see the two surfaces are independent rather than aliased.
//
// A canonical site identity requires a provable *module principal*: the site ID
// is keyed on the principal, and an anonymous module is not evidence that two
// objects can safely share one identity (two distinct anonymous modules would
// otherwise hash to the same site).  This module therefore carries an explicit
// symbol; the anonymous-module case is covered by
// hmx-vtcm-accounting-identity.mlir, which stays `not-proven` by design.
//
// Second, a per-invocation upper bound is never escalated into a runtime,
// grid, or allocator claim.  The module deliberately carries an externally
// supplied grid-shaped attribute; the analysis must not consume it, and every
// runtime fact must stay `not-proven` even though the byte peak is complete.
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The identity sidecar prints before the liveness sidecar, so the canonical ID
// is captured first and then matched inside the liveness record.
// CHECK-DAG: schema = "hmx.vtcm-static-identity/v1"
// CHECK-DAG: id_schema = "hmx.resident-key/fnv1a64/v1"
// CHECK-DAG: site_id_basis = "module-principal+function-symbol+source+role+slot"
// CHECK-DAG: site_count = 1 : i64
// CHECK-DAG: site_id = [[PEAK_SITE_ID:-?[0-9]+]] : i64

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: symbol = "canonical_peak"
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: deallocation_sites = 1 : i64
// CHECK-DAG: transient_requested_peak_bytes = 512 : i64
// CHECK-DAG: peak_site_count = 1 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// The peak list is the canonical identity, not an ordinal.
// CHECK-DAG: peak_site_ids = array<i64: [[PEAK_SITE_ID]]>

// A proven byte peak must not imply any runtime, grid, allocator, or
// fragmentation fact, and the scope must stay per-invocation.
// CHECK-DAG: scope = "per-function-single-invocation"
// CHECK-DAG: allocator_peak_status = "not-proven"
// CHECK-DAG: grid_status = "not-proven"
// CHECK-DAG: resident_runtime_state = "not-proven"
// CHECK-DAG: fragmentation_status = "not-proven"
// CHECK-DAG: free_cache_retention_status = "not-proven"
// CHECK-DAG: status = "complete"
module @canonical_peak_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness,
                    // An externally supplied, grid-shaped claim. The liveness
                    // analysis must not read it, and must not turn a
                    // per-invocation figure into a grid-level one.
                    hmx.grid_policy = "legacy-runtime",
                    hmx.kernel_vtcm_grid = 4 : i64} {
  func.func @canonical_peak() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %zero = arith.constant 0 : index
    %value = memref.load %a[%zero, %zero] : memref<16x16xf16, 1>
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}
