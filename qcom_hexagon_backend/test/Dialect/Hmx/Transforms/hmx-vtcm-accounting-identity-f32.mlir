//===- hmx-vtcm-accounting-identity-f32.mlir - f32 resident source id --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The static-identity sidecar validates the resident weight's source view
// (`validRuntimeWeightSourceView`) -- the accounting pass's second copy of the
// same admission the resident pass applies. With that copy left f16-only, an
// f32 runtime weight made this fixture fail hard:
//
//   error: HMX VTCM static identity is not proven; resident source descriptor
//          is not statically proven
//
// The source view is an argument contract, so it widens with the resident
// contract (f16, or f32 which the host quantises); the resident *destination*
// stays a crouton and is still checked as one. The site below is the proof:
// `identity_status = "complete"` and `resident_provenance_status = "checked"`
// can only be reached after the source view validates, and the f32 contract in
// the manifest is the same one the launcher consumes.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}),hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// Attributes print alphabetically: markers, census, identity, prepack.
// CHECK: module @acc_identity_f32
// The census admits the f32 resident and closes the two transient croutons.
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: allocation_sites = 3 : i64
// CHECK-DAG: raw_site_sum_bytes = 24576 : i64
// CHECK-DAG: resident_site_sum_bytes = 8192 : i64
// CHECK-DAG: status = "complete"
// CHECK-DAG: transient_bytes = 16384 : i64
// CHECK-DAG: unknown_allocations = 0 : i64
// CHECK-DAG: weight_resident_bytes = 8192 : i64
// The identity sidecar: the resident site is fully identified, which requires
// the f32 source view to have passed. One contiguous run of the site's fields
// pins that this is the resident site and not a transient one.
// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: principal = "acc_identity_f32"
// CHECK-DAG: principal_status = "module-symbol"
// CHECK-DAG: site_count = 3 : i64
// CHECK-DAG: identity_status = "complete", principal = "acc_identity_f32", principal_status = "module-symbol", requested_bytes = 8192 : i64, resident_provenance_status = "checked", role = "weight-resident", role_status = "reviewed", runtime_observation_status = "not-integrated", site_id = {{-?[0-9]+}} : i64, size_status = "complete", slot = 1 : i64, slot_status = "provenance"
// The host contract names the f32 source; the image is the fp16 crouton.
// CHECK: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_f32\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2],\22dtype\22:\22f32\22}]"
module @acc_identity_f32 attributes {
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity} {
  func.func @runtime_weight_f32(%a: memref<64x64xf16>, %w: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    // The bridge is erased; the resident reads the argument's address.
    %wa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    scf.for %i = %c0 to %c4 step %c1 {
      %r = arith.divui %i, %c2 : index
      %c = arith.remui %i, %c2 : index
      hmx.pack_weight ins(%w, %r, %c : memref<64x64xf32>) outs(%wa : memref<2x2x16x32x2xf16, 1>)
    }
    %aa = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %ar = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    hmx.matmul ins(%aa, %wa : memref<2x2x16x32x2xf16, 1>, memref<2x2x16x32x2xf16, 1>) outs(%ar : memref<2x2x16x32x2xf16, 1>) loc("acc_identity_f32":4:5)
    memref.dealloc %wa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %aa : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x2x16x32x2xf16, 1>
    return
  }
}
