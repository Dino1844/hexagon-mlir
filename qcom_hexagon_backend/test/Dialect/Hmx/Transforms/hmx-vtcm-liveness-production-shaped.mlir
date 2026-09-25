//===- hmx-vtcm-liveness-production-shaped.mlir - DPS alias liveness ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The bufferized HMX layout ops return their destination memref as a DPS
// result.  That result is an alias of the destination allocation, not of the
// row-major source.  The liveness probe must follow that edge so a real
// pack/unpack sequence can be covered without treating the result as an
// untracked VTCM allocation.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "pack_alias"
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: deallocation_sites = 1 : i64
// CHECK-DAG: transient_requested_peak_bytes = 2048 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 2048 : i64
// CHECK-DAG: peak_site_count = 1 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// CHECK-DAG: allocator_peak_status = "not-proven"
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @pack_alias_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @pack_alias(%src: memref<32x32xf16>, %out: memref<32x32xf16>,
                        %row: index, %col: index) {
    %dst = memref.alloc() : memref<1x1x16x32x2xf16, 1>
    %packed = hmx.pack_act ins(%src, %row, %col : memref<32x32xf16>)
        outs(%dst : memref<1x1x16x32x2xf16, 1>)
        -> memref<1x1x16x32x2xf16, 1>
    %unpacked = hmx.unpack_acc ins(%packed, %row, %col
        : memref<1x1x16x32x2xf16, 1>)
        outs(%out : memref<32x32xf16>) -> memref<32x32xf16>
    memref.dealloc %dst : memref<1x1x16x32x2xf16, 1>
    return
  }
}
