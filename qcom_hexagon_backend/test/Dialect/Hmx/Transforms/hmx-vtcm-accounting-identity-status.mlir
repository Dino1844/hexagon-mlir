//===- hmx-vtcm-accounting-identity-status.mlir - typed identity status --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Status is carried by typed identity facts, not inferred by searching reason
// strings.  A repeated semantic tuple is incomplete because analysis found a
// recoverable ambiguity; an unknown source remains not-proven.  The raw census
// itself is complete in both cases.
//
// A repeated tuple is ambiguous on more than one identity axis, so the reason
// carries one clause per axis.  Two sites that share a canonical tuple also
// share the derived 128-bit event token, so the token axis collides as well;
// both facts are real and neither publishes a site ID.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @duplicate_tuple
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: status = "complete"
// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: status = "incomplete"
// CHECK-DAG: identity_status = "incomplete"
// CHECK-DAG: reason = "duplicate or ambiguous allocation-site identity; allocation-site identity hash collision"
// A repeated tuple must not publish a site ID on any axis.
// CHECK-NOT: site_id =
module @duplicate_tuple attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @duplicate() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("duplicate":1:1)
    %b = memref.alloc() : memref<8x16xf16, 1> loc("duplicate":1:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    memref.dealloc %b : memref<8x16xf16, 1>
    return
  }
}

// -----

// CHECK-LABEL: module @unknown_source
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: status = "complete"
// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: status = "not-proven"
// CHECK-DAG: identity_status = "not-proven"
// CHECK-DAG: reason = "unknown source location"
module @unknown_source attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @unknown() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc(unknown)
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}
