//===- hmx-vtcm-liveness-aligned.mlir - size-aligned bound, sidecar only --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//
//
// The size-aligned view of the same liveness result. The sizes here are chosen
// so the quantum is actually observable and both regimes are exercised:
//
//   memref<129xf16,1>  =  258 B  ->  384 B   (128 B quantum, below threshold)
//   memref<1025xf16,1> = 2050 B  -> 4096 B   (2048 B quantum, at threshold)
//
// so the raw peak is 2308 while the aligned peak is 4480. A fixture whose sizes
// were already quantum multiples would show the two figures equal and prove
// nothing about the rounding.
//
// What this file deliberately does NOT claim: 4480 is an upper bound over a
// union-joined live set, not a peak, and not an occupancy. The status fields say
// so, and the v3 `allocator_aligned` block stays not-proven because a bound is
// not the split/coalesce model that block would have to mean.
//
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: functions = [{
// The raw figures, unchanged by this work.
// CHECK-DAG: transient_requested_peak_bytes = 2308 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 2308 : i64
// The aligned figures, with the bound/exact distinction carried in the data.
// CHECK-DAG: transient_aligned_peak_bytes = 4480 : i64
// CHECK-DAG: transient_aligned_peak_status = "upper-bound-not-exact-peak"
// CHECK-DAG: resident_aligned_bytes = 0 : i64
// CHECK-DAG: resident_aligned_status = "exact-process-floor"
// CHECK-DAG: modeled_aligned_peak_bytes = 4480 : i64
// CHECK-DAG: modeled_aligned_peak_status = "upper-bound-not-occupancy"
// The basis is stated, so a reader cannot read the number as a header-inclusive
// or address-padded charge.
// CHECK-DAG: aligned_charge_basis = "runtime-size-quantum-no-header-no-address-padding"
// The aligned peak carries its own attribution. It is tracked separately from
// the raw peak because the two maxima can be attained at different program
// points, so reusing the raw peak's sites here would name the wrong one.
// CHECK-DAG: peak_aligned_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_aligned_site_ids = array<i64:
// CHECK-DAG: status = "complete"
module @aligned_liveness attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @aligned_liveness() {
    %a = memref.alloc() : memref<129xf16, 1>
    %b = memref.alloc() : memref<1025xf16, 1>
    memref.dealloc %a : memref<129xf16, 1>
    memref.dealloc %b : memref<1025xf16, 1>
    return
  }
}

// The incomplete case is not repeated here on purpose. The aligned fields sit
// inside the same `facts.applicable && facts.complete` guard as the raw figures,
// so an incomplete function structurally cannot publish an aligned peak at all
// -- there is no "partially aligned" record to get wrong. The refusals
// themselves are covered by hmx-vtcm-liveness-rejects.mlir.
