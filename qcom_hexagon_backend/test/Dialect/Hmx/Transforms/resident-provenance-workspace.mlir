//===- resident-provenance-workspace.mlir - resident identity contract --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The accounting + static-identity markers enable the strict, process-principal
// workspace contract.  They change only diagnostic evidence: the runtime key
// stays the production compatibility key for marked and unmarked modules
// alike.  The strict site identity (provenance site_id) is derived from
// principal/function/source-location, never from allocation order.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' | linalg-hexagon-opt -split-input-file -pass-pipeline='builtin.module(func.func(hmx-workspace-resident))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @workspace
// CHECK: hmx.resident_provenance =
// CHECK-DAG: key_namespace = "hmx.resident/v1"
// CHECK-DAG: content_status = "per-launch-refill"
// CHECK-DAG: launch_status = "not-proven"
// The pass records scope only; it does not claim a launcher/grid proof.
// CHECK-NOT: grid_status
// CHECK-NOT: launch_complete
// CHECK-DAG: kind = "workspace"
// CHECK-DAG: module = "resident_principal"
// CHECK-DAG: principal_status = "module-symbol"
// CHECK-DAG: scope = "one-immutable-principal/process"
// CHECK-DAG: site = "file:resident:10:1"
// CHECK-DAG: key = -{{[0-9]+}} : i64
// CHECK: hmx.workspace_resident =
// CHECK-DAG: bytes = 8192 : i64
// CHECK-DAG: alignment = 128 : i64
// CHECK-NOT: memref.dealloc
module @resident_principal attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @workspace() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("resident":10:1)
    %w = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("resident":11:1)
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("resident":12:1)
    hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %w : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// A repeated source site is a key collision, even if the allocations have
// identical types.  The pass rejects it before erasing either deallocation.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  // expected-error @+1 {{resident workspace key collision for stable site}}
  func.func @workspace_collision() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("collision":30:1)
    %b = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("collision":30:1)
    hmx.matmul ins(%a, %b : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%a : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %b : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// A view/cast deallocation is not the principal allocation's deallocation.  It
// is rejected rather than converted into a resident free by accident.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  // expected-error @+1 {{resident workspace has a deallocation through a memref alias/view}}
  func.func @workspace_alias_dealloc() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("alias":40:1)
    %view = memref.subview %a[0, 0, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] : memref<2x2x16x32x2xf16, 1> to memref<1x1x16x32x2xf16, strided<[2048, 1024, 64, 2, 1]>, 1>
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("alias":42:1)
    hmx.matmul ins(%a, %a : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %view : memref<1x1x16x32x2xf16, strided<[2048, 1024, 64, 2, 1]>, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// The raw census marker remains observational: it keeps the compatibility key
// and does not invent strict provenance.
// CHECK-LABEL: func.func @accounting_only
// CHECK-NOT: hmx.resident_provenance
// CHECK: hmx.workspace_resident
// CHECK-DAG: key =
// CHECK-DAG: bytes = 8192 : i64
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @accounting_only() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("raw":50:1)
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1> loc("raw":51:1)
    hmx.matmul ins(%a, %a : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// Malformed opt-in markers are not silently downgraded to the compatibility
// path. A caller that asks for strict identity must use unit markers.
// expected-error @+1 {{hmx.diagnostic_vtcm_identity must be a unit attribute}}
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity = "strict"} {
  func.func @malformed_identity_marker() {
    return
  }
}
