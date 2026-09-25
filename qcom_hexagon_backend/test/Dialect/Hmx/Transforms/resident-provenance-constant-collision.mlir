//===- resident-provenance-constant-collision.mlir - global owner -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Constant site identity is owned by the global symbol as well as the site
// hash. A pre-existing @old_weight record therefore collides with a new
// @new_weight allocation at the same source location instead of overwriting
// the old owner.
//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' 2>&1 | FileCheck %s
// CHECK: strict resident weight identity key collision
//===----------------------------------------------------------------------===//

module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_identity,
                    hmx.weight_resident_bytes = 8192 : i64} {
  memref.global "private" constant @old_weight : memref<2x2x16x32x2xf16> = dense<1.0> {alignment = 128 : i64}
  memref.global "private" constant @new_weight : memref<2x2x16x32x2xf16> = dense<2.0> {alignment = 128 : i64}
  func.func @constant_collision() {
    %old_resident = hexagonmem.alloc() {hmx.weight_resident = {bytes = 8192 : i64, global = @old_weight}, hmx.resident_provenance = {address_reuse_status = "immutable-source", alignment = 128 : i64, bytes = 8192 : i64, content_identity = "compile-time-symbol", content_status = "immutable-compile-time-global", descriptor_status = "checked", function = "constant_collision", function_id = -1678311832245279874 : i64, identity_key = 7177175867011946022 : i64, key_namespace = "hmx.resident/v1", kind = "weight", launch_status = "not-proven", module = "<anonymous-principal>", principal_status = "not-proven", reuse_status = "process-resident", role = "weight-resident", runtime_key_kind = "global-address", schema = "hmx.resident-key/fnv1a64/v1", scope = "one-immutable-principal/process", site = "file:existing:1:1", site_id = 7177175867011946022 : i64, slot = 0 : i64, source = "global:old_weight"}} : memref<2x2x16x32x2xf16, 1> loc("existing":1:1)
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %r0 = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %old_resident : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r0 : memref<2x2x16x32x2xf16, 1>)

    %new_weight = memref.get_global @new_weight : memref<2x2x16x32x2xf16>
    %r1 = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %new_weight : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16>) outs(%r1 : memref<2x2x16x32x2xf16, 1>) loc("existing":1:1)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r0 : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r1 : memref<2x2x16x32x2xf16, 1>
    return
  }
}
