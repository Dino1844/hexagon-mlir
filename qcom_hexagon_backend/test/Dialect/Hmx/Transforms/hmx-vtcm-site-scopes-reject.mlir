//===- hmx-vtcm-site-scopes-reject.mlir - site table refusals ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Every reason the per-site table refuses, as a hard diagnostic rather than a
// silently smaller scope. A missing build identity, a non-unit marker, a grid
// that is not explicitly one, a call in the module, and a site outside the
// single function. The last two matter most: a call means the enter/leave span
// could not own a site across the call boundary, and a second function means
// the per-site owner would have to be a process fact rather than a per-thread
// one.
//
// RUN: not linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(convert-to-hexagonmem),hmx-vtcm-accounting)' 2>&1 | FileCheck %s
//===----------------------------------------------------------------------===//

// -----

// A token needs an explicit build identity. It is never derived from IR
// contents, a pointer, or a build path.
// CHECK: hmx.kernel_vtcm_site_scopes =
// CHECK-DAG: eligible = false
// CHECK-DAG: reason = "explicit build identity is required for a site token"
module @n2_site_no_build_id attributes {
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2":7:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b : memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    return
  }
}

// -----

// The marker must be a unit attribute. A non-unit spelling is a present record
// of an unknown shape, which is not the same as no request at all.
// CHECK: hmx.diagnostic_vtcm_site_scope must be a unit attribute
module @n2_site_bad_marker attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope = "yes"} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2":7:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b : memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    return
  }
}

// -----

// The grid must be declared one. A site scope is per invocation; with no
// explicit grid=1 there is no invocation to spend the single invocation
// ordinal on.
// CHECK: hmx.kernel_vtcm_site_scopes =
// CHECK-DAG: eligible = false
// CHECK-DAG: reason = "explicit hmx.kernel_vtcm_grid=1 is required"
module @n2_site_no_grid attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2":7:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b : memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    return
  }
}

// -----

// A call in the module is refused for the site table. The per-site enter/leave
// span has to be the tightest possible bracket around one allocation, and a
// call is a point where another thread's scope could be current and where this
// thread's owner could be cleared without this pass knowing. The call is
// indirect so the module still has exactly one function: the refusal has to be
// about the call, not about the function count.
// CHECK: hmx.kernel_vtcm_site_scopes =
// CHECK-DAG: eligible = false
// CHECK-DAG: reason = "call, async, dynamic-grid, or pointer-escape scope is unsupported"
module @n2_site_with_call attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  llvm.func @helper(!llvm.ptr) -> !llvm.ptr
  func.func @kernel(%a: memref<64xf16>, %f: !llvm.ptr) {
    %b = hexagonmem.alloc() : memref<64xf16, 1> loc("n2":7:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b : memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    %r = llvm.call @helper(%f) : (!llvm.ptr) -> !llvm.ptr
    return
  }
}

// -----

// Two functions are refused even when only one of them allocates: a per-site
// owner is a per-thread fact, and with two functions the table would have to
// claim a process-level owner for a site.
// CHECK: hmx.kernel_vtcm_site_scopes =
// CHECK-DAG: eligible = false
// CHECK-DAG: reason = "site scopes require exactly one function"
module @n2_site_two_functions attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2":7:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b : memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    return
  }
  func.func @other(%a: memref<64xf16>) {
    return
  }
}
