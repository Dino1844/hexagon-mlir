//===- hmx-vtcm-accounting-runtime-weight-slots.mlir - provenance slots --===//
//
// A runtime weight's allocator-site identity must use the argument slot carried
// by hmx.resident_provenance. Two allocations from the same source location
// must not silently collapse to slot zero.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_vtcm_identity =
// CHECK-DAG: function_id = -4084883324461118049 : i64
// CHECK-DAG: role = "weight-resident"
// CHECK-DAG: slot = 0 : i64
// CHECK-DAG: site_id = -3743885154271517642 : i64
// CHECK-DAG: resident_provenance_status = "checked"
// CHECK-DAG: slot = 1 : i64
// CHECK-DAG: site_id = 5959734804126007510 : i64
// CHECK-DAG: resident_provenance_status = "checked"
module @two_slots attributes {
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.weight_resident_bytes = 16384 : i64} {
  func.func @two_runtime_weights(%a: memref<64x64xf16>, %b: memref<64x64xf16>) {
    %aa = memref.extract_aligned_pointer_as_index %a : memref<64x64xf16> -> index
    %ra = hexagonmem.alloc(%aa) {
      hmx.weight_resident = {address, bytes = 8192 : i64},
      hmx.resident_provenance = {
        alignment = 128 : i64, bytes = 8192 : i64,
        descriptor_status = "checked", function = "two_runtime_weights",
        function_id = -4084883324461118049 : i64, kind = "weight",
        module = "two_slots", principal_status = "module-symbol",
        role = "weight-resident", runtime_key_kind = "argument-address",
        schema = "hmx.resident-key/fnv1a64/v1",
        scope = "one-immutable-principal/process", site = "file:two:10:1",
        site_id = -3743885154271517642 : i64, slot = 0 : i64,
        source = "entry-argument-slot:0", launch_status = "not-proven",
        content_status = "not-proven", content_identity = "not-carried-to-device",
        address_reuse_status = "not-proven", reuse_status = "process-resident",
        source_view = {argument_kind = "ranked-memref", offset = 0 : i64,
                       shape = [64, 64], strides = [64, 1],
                       view_kind = "whole-argument", whole_n = 64 : i64}
      }} : memref<2x2x16x32x2xf16, 1> loc("two":10:1)
    %ab = memref.extract_aligned_pointer_as_index %b : memref<64x64xf16> -> index
    %rb = hexagonmem.alloc(%ab) {
      hmx.weight_resident = {address, bytes = 8192 : i64},
      hmx.resident_provenance = {
        alignment = 128 : i64, bytes = 8192 : i64,
        descriptor_status = "checked", function = "two_runtime_weights",
        function_id = -4084883324461118049 : i64, kind = "weight",
        module = "two_slots", principal_status = "module-symbol",
        role = "weight-resident", runtime_key_kind = "argument-address",
        schema = "hmx.resident-key/fnv1a64/v1",
        scope = "one-immutable-principal/process", site = "file:two:11:1",
        site_id = 5959734804126007510 : i64, slot = 1 : i64,
        source = "entry-argument-slot:1", launch_status = "not-proven",
        content_status = "not-proven", content_identity = "not-carried-to-device",
        address_reuse_status = "not-proven", reuse_status = "process-resident",
        source_view = {argument_kind = "ranked-memref", offset = 0 : i64,
                       shape = [64, 64], strides = [64, 1],
                       view_kind = "whole-argument", whole_n = 64 : i64}
      }} : memref<2x2x16x32x2xf16, 1> loc("two":11:1)
    return
  }
}
