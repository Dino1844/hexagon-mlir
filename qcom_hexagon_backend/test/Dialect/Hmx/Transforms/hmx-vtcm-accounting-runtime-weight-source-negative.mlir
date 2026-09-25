//===- hmx-vtcm-accounting-runtime-weight-source-negative.mlir - source -===//
//
// A resident descriptor is not allowed to turn a wrong, dynamic, or strided
// source view into a static site identity. These are negative identity cases;
// the raw census remains separate from the strict resident contract.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(hmx-vtcm-accounting)'
//===----------------------------------------------------------------------===//

// expected-error @+1 {{HMX VTCM static identity is not proven; resident source descriptor is not statically proven}}
module @source_negative attributes {hmx.diagnostic_vtcm_accounting,
                                  hmx.diagnostic_vtcm_identity,
                                  hmx.weight_resident_bytes = 8192 : i64} {
  func.func @bad_source(%a: memref<64x64xf16>, %w: memref<64x64xf16>) {
    %addr = memref.extract_aligned_pointer_as_index %w : memref<64x64xf16> -> index
    %r = hexagonmem.alloc(%addr) {
      hmx.weight_resident = {address, bytes = 8192 : i64},
      hmx.resident_provenance = {alignment = 128 : i64, bytes = 8192 : i64,
        descriptor_status = "checked", function = "bad_source",
        function_id = 4846240067381642463 : i64, kind = "weight",
        module = "source_negative", principal_status = "module-symbol",
        role = "weight-resident", runtime_key_kind = "argument-address",
        schema = "hmx.resident-key/fnv1a64/v1", site = "file:negative:10:1",
        site_id = -7330513106228430871 : i64, slot = 1 : i64,
        source = "entry-argument-slot:1", source_view = {
          argument_kind = "ranked-memref", offset = 0 : i64, shape = [32, 64],
          strides = [64, 1], view_kind = "whole-argument", whole_n = 64 : i64}}
    } : memref<2x2x16x32x2xf16, 1> loc("negative":10:1)
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM static identity is not proven; resident source descriptor is not statically proven}}
module @source_negative attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity,
                                    hmx.weight_resident_bytes = 8192 : i64} {
  func.func @bad_source(%a: memref<64x64xf16>, %w: memref<64x64xf16>) {
    %addr = memref.extract_aligned_pointer_as_index %w : memref<64x64xf16> -> index
    %r = hexagonmem.alloc(%addr) {
      hmx.weight_resident = {address, bytes = 8192 : i64},
      hmx.resident_provenance = {alignment = 128 : i64, bytes = 8192 : i64,
        descriptor_status = "checked", function = "bad_source",
        function_id = 4846240067381642463 : i64, kind = "weight",
        module = "source_negative", principal_status = "module-symbol",
        role = "weight-resident", runtime_key_kind = "argument-address",
        schema = "hmx.resident-key/fnv1a64/v1", site = "file:negative:20:1",
        site_id = 8436001843365609010 : i64, slot = 1 : i64,
        source = "entry-argument-slot:1", source_view = {
          argument_kind = "ranked-memref", offset = -1 : i64, shape = [64, 64],
          strides = [64, 1], view_kind = "whole-argument", whole_n = 64 : i64}}
    } : memref<2x2x16x32x2xf16, 1> loc("negative":20:1)
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM static identity is not proven; resident source descriptor is not statically proven}}
module @source_negative attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity,
                                    hmx.weight_resident_bytes = 8192 : i64} {
  func.func @bad_source(%a: memref<64x64xf16>, %w: memref<64x64xf16>) {
    %addr = memref.extract_aligned_pointer_as_index %w : memref<64x64xf16> -> index
    %r = hexagonmem.alloc(%addr) {
      hmx.weight_resident = {address, bytes = 8192 : i64},
      hmx.resident_provenance = {alignment = 128 : i64, bytes = 8192 : i64,
        descriptor_status = "checked", function = "bad_source",
        function_id = 4846240067381642463 : i64, kind = "weight",
        module = "source_negative", principal_status = "module-symbol",
        role = "weight-resident", runtime_key_kind = "argument-address",
        schema = "hmx.resident-key/fnv1a64/v1", site = "file:negative:30:1",
        site_id = 8228668596080719811 : i64, slot = 1 : i64,
        source = "entry-argument-slot:1", source_view = {
          argument_kind = "ranked-memref", offset = 0 : i64, shape = [64, 64],
          strides = [65, 1], view_kind = "whole-argument", whole_n = 64 : i64}}
    } : memref<2x2x16x32x2xf16, 1> loc("negative":30:1)
    return
  }
}
