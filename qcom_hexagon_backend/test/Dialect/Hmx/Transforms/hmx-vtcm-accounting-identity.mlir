//===- hmx-vtcm-accounting-identity.mlir - static site identity ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This is an internal static sidecar, not a manifest v2 field and not a
// runtime observation.  In particular, an absent build identity remains
// explicitly not-proven; the pass does not manufacture one from IR contents.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: kind = "static-vtcm-site-identity-v1"
// CHECK-DAG: schema = "hmx.vtcm-static-identity/v1"
// CHECK-DAG: id_schema = "hmx.resident-key/fnv1a64/v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: identity_status = "complete"
// CHECK-DAG: build_id_status = "not-proven"
// CHECK-DAG: build_id_source = "not-present"
// CHECK-DAG: runtime_join_status = "not-integrated"
// CHECK-DAG: join_status = "not-proven"
// CHECK-DAG: principal = "identity_module"
// CHECK-DAG: principal_status = "module-symbol"
// CHECK-DAG: multi_function_status = "not-applicable"
// CHECK-DAG: function = "identity"
// CHECK-DAG: function_id = 7377597695996945320 : i64
// CHECK-DAG: source = "file:identity:10:1"
// CHECK-DAG: role = "transient-memref"
// CHECK-DAG: slot = 0 : i64
// CHECK-DAG: site_id = 3996215541609978327 : i64
// CHECK-DAG: source = "file:identity:11:1"
// CHECK-DAG: role = "workspace-resident"
// CHECK-DAG: site_id = 6453376259803496673 : i64
// CHECK-DAG: resident_provenance_status = "checked"
// CHECK-DAG: runtime_observation_status = "not-integrated"
module @identity_module attributes {
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity} {
  func.func @identity() {
    %transient = memref.alloc() : memref<64x64xf16, 1> loc("identity":10:1)
    %workspace = memref.alloc()
        {hmx.workspace_resident = {key = -1 : i64, bytes = 1024 : i64},
         hmx.resident_provenance = {alignment = 128 : i64, bytes = 1024 : i64,
           descriptor_status = "checked", function = "identity",
           function_id = 7377597695996945320 : i64, key = -1 : i64,
           kind = "workspace", module = "identity_module",
           principal_status = "module-symbol", role = "workspace-resident",
           schema = "hmx.resident-key/fnv1a64/v1",
           scope = "one-immutable-principal/process",
           site = "file:identity:11:1", site_id = 6453376259803496673 : i64,
           slot = 0 : i64, launch_status = "not-proven",
           content_status = "per-launch-refill", reuse_status = "process-resident"}}
        : memref<32x16xf16, 1> loc("identity":11:1)
    memref.dealloc %transient : memref<64x64xf16, 1>
    return
  }
}
