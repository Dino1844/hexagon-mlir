//===- resident-provenance-repeat.mlir - repeated source contracts -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// An address-keyed resident must retain the source view that justified its
// first lowering. These negative cases model IR after the pack bridge has been
// erased, where the argument type alone no longer describes an unranked view or
// distinguishes a whole object from a static N slice.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))'
//===----------------------------------------------------------------------===//

// A repeated lowering cannot accept a changed unranked view descriptor: the
// resident still covers [64,64], so the persisted K=32 view is contradictory.
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_identity,
                    hmx.weight_resident_bytes = 8192 : i64} {
  // expected-error @+1 {{strict resident weight source-view descriptor does not match its source}}
  func.func @changed_unranked(%a: memref<2x2x16x32x2xf16, 1>, %w: memref<*xf16>) {
    %addr = memref.extract_aligned_pointer_as_index %w : memref<*xf16> -> index
    %resident = hexagonmem.alloc(%addr) {hmx.weight_resident = {address, bytes = 8192 : i64}, hmx.resident_provenance = {address_reuse_status = "not-proven", alignment = 128 : i64, bytes = 8192 : i64, content_identity = "not-carried-to-device", content_status = "not-proven", descriptor_status = "checked", function = "changed_unranked", function_id = 4811206937765499214 : i64, identity_key = -1726279262899540951 : i64, key_namespace = "hmx.resident/v1", kind = "weight", launch_status = "not-proven", module = "<anonymous-principal>", principal_status = "not-proven", reuse_status = "process-resident", role = "weight-resident", runtime_key_kind = "argument-address", schema = "hmx.resident-key/fnv1a64/v1", scope = "one-immutable-principal/process", site = "file:existing:1:1", site_id = -1726279262899540951 : i64, slot = 1 : i64, source = "entry-argument-slot:1", source_view = {argument_kind = "unranked-memref", offset = 0 : i64, shape = [32, 64], strides = [64, 1], view_kind = "whole-argument", whole_n = 64 : i64}}} : memref<2x2x16x32x2xf16, 1> loc("existing":1:1)
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %resident : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// The preflight sees the existing whole-argument record before mutation. A new
// static N slice for the same address slot is a descriptor conflict, even
// though both views derive from argument slot 1.
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_identity,
                    hmx.weight_resident_bytes = 8192 : i64} {
  func.func @preexisting_slot(%a: memref<2x2x16x32x2xf16, 1>, %w: memref<64x64xf16>) {
    %addr = memref.extract_aligned_pointer_as_index %w : memref<64x64xf16> -> index
    %resident = hexagonmem.alloc(%addr) {hmx.weight_resident = {address, bytes = 8192 : i64}, hmx.resident_provenance = {address_reuse_status = "not-proven", alignment = 128 : i64, bytes = 8192 : i64, content_identity = "not-carried-to-device", content_status = "not-proven", descriptor_status = "checked", function = "preexisting_slot", function_id = -7452643547199112454 : i64, identity_key = -5321718700737576979 : i64, key_namespace = "hmx.resident/v1", kind = "weight", launch_status = "not-proven", module = "<anonymous-principal>", principal_status = "not-proven", reuse_status = "process-resident", role = "weight-resident", runtime_key_kind = "argument-address", schema = "hmx.resident-key/fnv1a64/v1", scope = "one-immutable-principal/process", site = "file:existing:1:1", site_id = -5321718700737576979 : i64, slot = 1 : i64, source = "entry-argument-slot:1", source_view = {argument_kind = "ranked-memref", offset = 0 : i64, shape = [64, 64], strides = [64, 1], view_kind = "whole-argument", whole_n = 64 : i64}}} : memref<2x2x16x32x2xf16, 1> loc("existing":1:1)
    %r0 = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %resident : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r0 : memref<2x2x16x32x2xf16, 1>)

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %view = memref.reinterpret_cast %w to offset: [32], sizes: [64, 32], strides: [64, 1] : memref<64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: 32>>
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c1 : index
      %c = arith.remui %i, %c1 : index
      hmx.pack_weight ins(%view, %r, %c : memref<64x32xf16, strided<[64, 1], offset: 32>>) outs(%wa : memref<1x2x16x32x2xf16, 1>)
    }
    %r1 = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    // expected-error @+1 {{strict resident runtime weight slot has conflicting source views}}
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>, memref<1x2x16x32x2xf16, 1>) outs(%r1 : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %r0 : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r1 : memref<2x1x16x32x2xf16, 1>
    return
  }
}
