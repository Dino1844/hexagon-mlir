//===- resident-provenance-weight.mlir - weight identity contract --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The accounting + static-identity markers make unsupported runtime sources
// and ambiguous process scope fail closed. Runtime weights retain an address
// runtime key, but their content status is explicitly not-proven: no host
// prepack hash is claimed as
// a device content identity.  The scope field records the required external
// one-principal/process contract; it is not a grid or launch proof.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' | FileCheck %s
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' | linalg-hexagon-opt -split-input-file -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @constant_weight
// CHECK: hmx.resident_provenance =
// CHECK-DAG: key_namespace = "hmx.resident/v1"
// CHECK-DAG: content_identity = "compile-time-symbol"
// CHECK-DAG: content_status = "immutable-compile-time-global"
// CHECK-DAG: address_reuse_status = "immutable-source"
// CHECK-DAG: runtime_key_kind = "global-address"
// CHECK-DAG: module = "<anonymous-principal>"
// CHECK-DAG: principal_status = "not-proven"
// CHECK-DAG: scope = "one-immutable-principal/process"
// CHECK-DAG: launch_status = "not-proven"
// CHECK-NOT: grid_status
// CHECK-NOT: launch_complete
// CHECK: hmx.weight_resident =
// CHECK-DAG: global = @weight
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  memref.global "private" constant @weight : memref<2x2x16x32x2xf16> = dense<1.0> {alignment = 128 : i64}
  func.func @constant_weight() {
    %a = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %w = memref.get_global @weight : memref<2x2x16x32x2xf16>
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %w : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %a : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// A ranked f16 entry argument is a supported source descriptor, but its
// content is not proven by the compiler.  In particular, this output must not
// contain a host hash field.
// CHECK-LABEL: func.func @runtime_weight
// CHECK: hmx.resident_provenance =
// CHECK-DAG: key_namespace = "hmx.resident/v1"
// CHECK-DAG: content_identity = "not-carried-to-device"
// CHECK-DAG: content_status = "not-proven"
// CHECK-DAG: address_reuse_status = "not-proven"
// CHECK-DAG: launch_status = "not-proven"
// CHECK-DAG: runtime_key_kind = "argument-address"
// CHECK-DAG: source_view = {argument_kind = "ranked-memref", offset = 0 : i64, shape = [64, 64], strides = [64, 1], view_kind = "whole-argument", whole_n = 64 : i64}
// CHECK-DAG: module = "<anonymous-principal>"
// CHECK-DAG: principal_status = "not-proven"
// CHECK-DAG: scope = "one-immutable-principal/process"
// CHECK-NOT: content_hash
// CHECK-NOT: prepack_hash
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @runtime_weight(%a: memref<2x2x16x32x2xf16, 1>, %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// The unranked entry ABI is supported when the view proves the whole f16
// object.  It still gets an address key and a not-proven content status.
// CHECK-LABEL: func.func @runtime_weight_unranked
// CHECK: hmx.resident_provenance =
// CHECK-DAG: content_status = "not-proven"
// CHECK-DAG: address_reuse_status = "not-proven"
// CHECK-DAG: launch_status = "not-proven"
// CHECK-DAG: runtime_key_kind = "argument-address"
// CHECK-DAG: source_view = {argument_kind = "unranked-memref", offset = 0 : i64, shape = [64, 64], strides = [64, 1], view_kind = "whole-argument", whole_n = 64 : i64}
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @runtime_weight_unranked(%a: memref<2x2x16x32x2xf16, 1>, %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %view = memref.reinterpret_cast %w to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<*xf16> to memref<64x64xf16, strided<[64, 1]>>
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%view, %r, %c : memref<64x64xf16, strided<[64, 1]>>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// A dynamic N offset can select a different slice on every invocation.  It is
// not a resident content identity, even when the arithmetic is tile-aligned.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @dynamic_weight(%a: memref<2x2x16x32x2xf16, 1>, %w: memref<*xf16>, %off: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %n0 = arith.muli %off, %c32 : index
    %view = memref.reinterpret_cast %w to offset: [%n0], sizes: [64, 32], strides: [64, 1] : memref<*xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c1 : index
      %c = arith.remui %i, %c1 : index
      hmx.pack_weight ins(%view, %r, %c : memref<64x32xf16, strided<[64, 1], offset: ?>>) outs(%wa : memref<1x2x16x32x2xf16, 1>)
    }
    %r = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    // expected-error @+1 {{strict resident runtime weight source has a dynamic offset}}
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>, memref<1x2x16x32x2xf16, 1>) outs(%r : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x1x16x32x2xf16, 1>
    return
  }
}

// -----

// A source allocated inside the function is not an entry-object contract.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @internal_weight(%a: memref<2x2x16x32x2xf16, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %src = memref.alloc() : memref<64x64xf16>
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%src, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    // expected-error @+1 {{strict resident runtime weight source is not a whole entry-argument view or a static N slice}}
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %src : memref<64x64xf16>
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// Two functions with address-keyed runtime weights are ambiguous in one
// process: the compiler cannot prove that their pointers do not alias or that
// the host object is the same immutable principal.
// expected-error @+1 {{strict resident weight provenance is ambiguous across functions}}
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @runtime_a(%a: memref<2x2x16x32x2xf16, 1>, %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
  func.func @runtime_b(%a: memref<2x2x16x32x2xf16, 1>, %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// A host-side hash is not accepted as a substitute for device content
// provenance. This fixture is otherwise exact and is rejected only because it
// claims that the compiler can identify device content by a prepack hash.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  memref.global "private" constant @weight : memref<2x2x16x32x2xf16> = dense<1.0> {alignment = 128 : i64}
  // expected-error @+1 {{strict resident weight provenance does not match its source}}
  func.func @bad_content() {
    %r = hexagonmem.alloc() {hmx.weight_resident = {bytes = 8192 : i64, global = @weight}, hmx.resident_provenance = {address_reuse_status = "immutable-source", alignment = 128 : i64, bytes = 8192 : i64, content_identity = "host-prepack-hash", content_status = "immutable-compile-time-global", descriptor_status = "checked", function = "bad_content", function_id = 6189966341557759861 : i64, identity_key = -2043504764193463941 : i64, key_namespace = "hmx.resident/v1", kind = "weight", launch_status = "not-proven", module = "<anonymous-principal>", principal_status = "not-proven", role = "weight-resident", reuse_status = "process-resident", runtime_key_kind = "global-address", schema = "hmx.resident-key/fnv1a64/v1", scope = "one-immutable-principal/process", site = "file:bad:200:1", site_id = -2043504764193463941 : i64, slot = 0 : i64, source = "global:weight"}} : memref<2x2x16x32x2xf16, 1> loc("bad":200:1)
    return
  }
}

// -----

// A descriptor whose byte count disagrees with its type is rejected before it
// can be used as a resident lookup key. All other provenance is exact.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  memref.global "private" constant @weight : memref<2x2x16x32x2xf16> = dense<1.0> {alignment = 128 : i64}
  // expected-error @+1 {{strict resident weight has an inexact byte/alignment descriptor}}
  func.func @bad_size() {
    %r = hexagonmem.alloc() {hmx.weight_resident = {bytes = 4096 : i64, global = @weight}, hmx.resident_provenance = {address_reuse_status = "immutable-source", alignment = 128 : i64, bytes = 4096 : i64, content_identity = "compile-time-symbol", content_status = "immutable-compile-time-global", descriptor_status = "checked", function = "bad_size", function_id = 8946912820164523406 : i64, identity_key = 407864776903668571 : i64, key_namespace = "hmx.resident/v1", kind = "weight", launch_status = "not-proven", module = "<anonymous-principal>", principal_status = "not-proven", role = "weight-resident", reuse_status = "process-resident", runtime_key_kind = "global-address", schema = "hmx.resident-key/fnv1a64/v1", scope = "one-immutable-principal/process", site = "file:bad:210:1", site_id = 407864776903668571 : i64, slot = 0 : i64, source = "global:weight"}} : memref<2x2x16x32x2xf16, 1> loc("bad":210:1)
    return
  }
}

// -----

// Launch scope is externally supplied and is not proved by a compiler-side
// allocation site. A hand-written "complete" claim therefore fails closed.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  memref.global "private" constant @weight : memref<2x2x16x32x2xf16> = dense<1.0> {alignment = 128 : i64}
  // expected-error @+1 {{strict resident weight provenance does not match its source}}
  func.func @bad_launch() {
    %r = hexagonmem.alloc() {hmx.weight_resident = {bytes = 8192 : i64, global = @weight}, hmx.resident_provenance = {address_reuse_status = "immutable-source", alignment = 128 : i64, bytes = 8192 : i64, content_identity = "compile-time-symbol", content_status = "immutable-compile-time-global", descriptor_status = "checked", function = "bad_launch", function_id = 5975553249072643054 : i64, identity_key = -518874878206567560 : i64, key_namespace = "hmx.resident/v1", kind = "weight", launch_status = "complete", module = "<anonymous-principal>", principal_status = "not-proven", role = "weight-resident", reuse_status = "process-resident", runtime_key_kind = "global-address", schema = "hmx.resident-key/fnv1a64/v1", scope = "one-immutable-principal/process", site = "file:bad:220:1", site_id = -518874878206567560 : i64, slot = 0 : i64, source = "global:weight"}} : memref<2x2x16x32x2xf16, 1> loc("bad":220:1)
    return
  }
}

// -----

// An immutable compile-time global has a stronger address-reuse claim than an
// entry argument. The compiler still rejects a forged weaker status.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  memref.global "private" constant @weight : memref<2x2x16x32x2xf16> = dense<1.0> {alignment = 128 : i64}
  // expected-error @+1 {{strict resident weight provenance does not match its source}}
  func.func @bad_address_reuse() {
    %r = hexagonmem.alloc() {hmx.weight_resident = {bytes = 8192 : i64, global = @weight}, hmx.resident_provenance = {address_reuse_status = "not-proven", alignment = 128 : i64, bytes = 8192 : i64, content_identity = "compile-time-symbol", content_status = "immutable-compile-time-global", descriptor_status = "checked", function = "bad_address_reuse", function_id = 1744784338981317249 : i64, identity_key = -3249338822386932184 : i64, key_namespace = "hmx.resident/v1", kind = "weight", launch_status = "not-proven", module = "<anonymous-principal>", principal_status = "not-proven", role = "weight-resident", reuse_status = "process-resident", runtime_key_kind = "global-address", schema = "hmx.resident-key/fnv1a64/v1", scope = "one-immutable-principal/process", site = "file:bad:230:1", site_id = -3249338822386932184 : i64, slot = 0 : i64, source = "global:weight"}} : memref<2x2x16x32x2xf16, 1> loc("bad":230:1)
    return
  }
}

// -----

// Function/site IDs are part of the exact diagnostic join contract, not
// decorative metadata. A one-bit site-ID change is rejected.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  memref.global "private" constant @weight : memref<2x2x16x32x2xf16> = dense<1.0> {alignment = 128 : i64}
  // expected-error @+1 {{strict resident weight provenance does not match its source}}
  func.func @bad_identity() {
    %r = hexagonmem.alloc() {hmx.weight_resident = {bytes = 8192 : i64, global = @weight}, hmx.resident_provenance = {address_reuse_status = "immutable-source", alignment = 128 : i64, bytes = 8192 : i64, content_identity = "compile-time-symbol", content_status = "immutable-compile-time-global", descriptor_status = "checked", function = "bad_identity", function_id = -8367400227387443155 : i64, identity_key = -1512228204336236331 : i64, key_namespace = "hmx.resident/v1", kind = "weight", launch_status = "not-proven", module = "<anonymous-principal>", principal_status = "not-proven", role = "weight-resident", reuse_status = "process-resident", runtime_key_kind = "global-address", schema = "hmx.resident-key/fnv1a64/v1", scope = "one-immutable-principal/process", site = "file:bad:240:1", site_id = -1512228204336236330 : i64, slot = 0 : i64, source = "global:weight"}} : memref<2x2x16x32x2xf16, 1> loc("bad":240:1)
    return
  }
}

// -----

// Alias deallocation is rejected for the runtime bridge too; the principal
// pack array and its view are not interchangeable resident objects.
module attributes {hmx.diagnostic_vtcm_accounting, hmx.diagnostic_vtcm_identity} {
  func.func @runtime_alias_dealloc(%a: memref<2x2x16x32x2xf16, 1>, %w: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf16>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %view = memref.subview %wa[0, 0, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] : memref<2x2x16x32x2xf16, 1> to memref<1x1x16x32x2xf16, strided<[2048, 1024, 64, 2, 1]>, 1>
    %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    // expected-error @+1 {{resident weight bridge has a deallocation through a memref alias/view}}
    hmx.matmul ins(%a, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%r : memref<2x2x16x32x2xf16, 1>)
    memref.dealloc %view : memref<1x1x16x32x2xf16, strided<[2048, 1024, 64, 2, 1]>, 1>
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %r : memref<2x2x16x32x2xf16, 1>
    return
  }
}
