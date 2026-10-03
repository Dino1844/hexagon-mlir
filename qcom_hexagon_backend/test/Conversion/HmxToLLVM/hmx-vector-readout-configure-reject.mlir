//===- hmx-vector-readout-configure-reject.mlir - refuse, never repair ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Wiring the vector executor into a kernel that cannot be wired produces a kernel
// that hands the caller unwritten output while reporting success -- the same class
// of bug as a mis-read accumulator row, and worse because it is silent. So every
// unresolvable handoff is a hard error here.
//
// Each arm is a REFUSAL, and each checks the message rather than a flag, because
// what has to be preserved is that the compiler says which fact it could not
// establish. A silently degraded path would produce a plausible-looking wrong
// answer, which is exactly what this split must never ship.
//
//   MISSING-WORK  a record naming an outlined read-out that is not in the module.
//                 Wiring it would hand the executor a symbol that does not exist.
//   WRONG-SHAPE   an outlined read-out whose signature the entry point does not
//                 model. Under the default convert-func-to-llvm convention the
//                 callee has to expand to exactly (memref, memref, index, index);
//                 anything else would pass the wrong words in the wrong order,
//                 silently.
//   TENSOR        ar recorded as a tensor rather than a memref. The tensor form
//                 of hmx.unpack_acc has no address for a descriptor to carry.
//   NOT-ARRAY     a record that is not an array of records at all.
//
// Note the first three name an `llvm.func` for `work` where the arm needs one, so
// the refusal is about the SPECIFIC missing fact rather than about the shape of
// the module; the fourth needs no function at all.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(convert-scf-to-cf,convert-func-to-llvm,hmx-to-llvm)' -verify-diagnostics
//===----------------------------------------------------------------------===//

// The diagnostic is anchored on the MODULE, not on the function the record names,
// because the module is what carries the record and what failed to resolve it.
// expected-error @below {{handoff names outlined read-out '__hmx_readout_absent', which is not an llvm.func}}
module attributes {hmx.readout.handoffs = [{ar = memref<32x16x16x32x2xf16, 1>, dst = memref<1024x512xf16>, engine = "kernel", work = "__hmx_readout_absent"}]} {
  func.func @kernel() {
    return
  }
}

// -----

module attributes {hmx.readout.handoffs = [{ar = memref<32x16x16x32x2xf16, 1>, dst = memref<1024x512xf16>, engine = "kernel", work = "__hmx_readout_wrong_shape"}]} {
  // Two parameters instead of four. After conversion the memref has expanded, so
  // the counts are the EXPANDED ones: 14 against the 22 the adapter would build.
  // The refusal names both, because "the arity is wrong" is not actionable on its
  // own -- someone reading the message needs to see which one it expected.
  // expected-error @below {{has 14 parameters; the vector-readout entry point models exactly}}
  func.func @__hmx_readout_wrong_shape(%arg0: memref<32x16x16x32x2xf16, 1>, %arg1: index) {
    return
  }
  func.func @kernel() {
    return
  }
}

// -----

// expected-error @below {{hmx.readout.handoffs ar/dst must be memref types}}
module attributes {hmx.readout.handoffs = [{ar = tensor<32x16x16x32x2xf16>, dst = memref<1024x512xf16>, engine = "kernel", work = "__hmx_readout_tensor"}]} {
  func.func @kernel() {
    return
  }
}

// -----

// expected-error @below {{hmx.readout.handoffs must be an array of handoff records}}
module attributes {hmx.readout.handoffs = "not-a-list"} {
  func.func @kernel() {
    return
  }
}