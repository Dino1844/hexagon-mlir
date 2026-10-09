//===- hexagonmem_weight_resident_ddr.mlir - DDR mirror lowering -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// A resident weight whose image does not fit the VTCM pool is placed in the
// permanent DDR mirror: same argument-address key, same byte count and
// alignment, but its own versioned entry -- because what the runtime does with
// the request differs (an ordinary DDR allocation filled once, not a pinned
// pool buffer). The placement is the descriptor's `location` fact, and it has
// to agree with the buffer's memory space: an entry that claims one home and
// allocates in the other is a producer/consumer disagreement, not a default to
// fall back on, so it is refused before anything is emitted.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -hexagonmem-to-llvm | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK: llvm.func @hexagon_runtime_weight_resident_ddr_v2_dsp(i64, i32, i32) -> !llvm.ptr

module {
  // CHECK-LABEL: func.func @resident_ddr
  func.func @resident_ddr() {
    %addr = arith.constant 4096 : index
    // The residency key operand (the weight's aligned pointer) becomes the
    // call's first argument; the descriptor's byte count and requested
    // alignment follow, exactly as on the VTCM entry.
    // CHECK-DAG: %[[BYTES:.*]] = llvm.mlir.constant(8192 : i32) : i32
    // CHECK-DAG: %[[ALIGN:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK: llvm.call @hexagon_runtime_weight_resident_ddr_v2_dsp({{.*}}, %[[BYTES]], %[[ALIGN]]) : (i64, i32, i32) -> !llvm.ptr
    %w = hexagonmem.alloc(%addr) {alignment = 128 : i64, hmx.weight_resident = {address, bytes = 8192 : i64, location = "ddr"}} : memref<2x2x16x32x2xf16>
    return
  }
}

// -----

// A `ddr` label on a buffer the pool would allocate: refused, with the two
// homes named, rather than silently resolved to one of them. The second
// expectation is the conversion's own reaction to a pattern that declined the
// op (`failed to legalize`), which is how every rejected op in this pass ends.
module {
  func.func @resident_ddr_in_vtcm() {
    %addr = arith.constant 4096 : index
    // expected-error @+2 {{resident weight location disagrees with its memory space}}
    // expected-error @+1 {{failed to legalize operation}}
    %w = hexagonmem.alloc(%addr) {alignment = 128 : i64, hmx.weight_resident = {address, bytes = 8192 : i64, location = "ddr"}} : memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

// A location spelling this conversion does not know is not "VTCM by default":
// the key exists, so it has to say something the conversion understands.
module {
  func.func @resident_unknown_location() {
    %addr = arith.constant 4096 : index
    // expected-error @+2 {{resident weight location must be "ddr"}}
    // expected-error @+1 {{failed to legalize operation}}
    %w = hexagonmem.alloc(%addr) {alignment = 128 : i64, hmx.weight_resident = {address, bytes = 8192 : i64, location = "pool"}} : memref<2x2x16x32x2xf16, 1>
    return
  }
}
