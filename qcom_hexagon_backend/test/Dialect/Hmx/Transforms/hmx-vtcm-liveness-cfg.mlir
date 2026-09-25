//===- hmx-vtcm-liveness-cfg.mlir - general control-flow liveness ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The liveness engine is a finite worklist fixpoint over the whole function
// CFG, so shapes that used to be rejected outright are now evaluated exactly
// as long as the allocation state is balanced at every merge:
//
//   * a multi-block function body;
//   * a plain-CFG cycle with a balanced body;
//   * a multi-block structured region;
//   * `scf.while`, whose before region exits through `scf.condition`;
//   * a loop-carried *alias* of a site allocated outside the loop, which does
//     not transfer ownership and therefore does not change any lifetime.
//
// `fixpoint_rounds` and `revised_blocks` are published so a reader can see
// that these really were solved by iteration rather than by a single linear
// walk, and `join_policy` states that a path merge yields an upper bound.
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' | FileCheck %s
//===----------------------------------------------------------------------===//

// A multi-block body: the allocation is released in a later block, so a single
// linear walk of one block would not see the pair at all.
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region"
// CHECK-DAG: join_policy = "union-join-upper-bound"
// CHECK-DAG: symbol = "multi_block_body"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: transient_requested_peak_bytes = 640 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 640 : i64
// CHECK-DAG: peak_site_count = 2 : i64

// A plain-CFG cycle.  The header is reached from the entry and from the body, so
// its entry state is a genuine join that the fixpoint has to revisit.
// CHECK-DAG: symbol = "cfg_cycle"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: transient_requested_peak_bytes = 640 : i64

// `scf.while` with a balanced body in both regions.
// CHECK-DAG: symbol = "structured_while"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: transient_requested_peak_bytes = 8704 : i64

// A multi-block structured region: the region owns a temporary and releases it
// before the enclosing block continues.
// CHECK-DAG: symbol = "multi_block_region"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: transient_requested_peak_bytes = 640 : i64

// A loop-carried alias of a site allocated outside the loop.  Carrying a handle
// is not transferring ownership, so the single site is still charged once.
// CHECK-DAG: symbol = "carried_alias"
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: deallocation_sites = 1 : i64
// CHECK-DAG: transient_requested_peak_bytes = 512 : i64

// A branch merge where each arm releases the same site: the join is empty, so
// the pair is proven rather than approximated.
// CHECK-DAG: symbol = "balanced_diamond"
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: transient_requested_peak_bytes = 512 : i64
// CHECK-DAG: status = "complete"
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @multi_block_body() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    cf.br ^release
  ^release:
    %t = memref.alloc() : memref<8x8xf16, 1>
    memref.dealloc %t : memref<8x8xf16, 1>
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }

  func.func @cfg_cycle() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    cf.br ^header
  ^header:
    %cond = arith.constant true
    cf.cond_br %cond, ^body, ^done
  ^body:
    %t = memref.alloc() : memref<8x8xf16, 1>
    memref.dealloc %t : memref<8x8xf16, 1>
    cf.br ^header
  ^done:
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }

  func.func @structured_while() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %result = scf.while (%arg = %c0) : (index) -> (index) {
      %b = memref.alloc() : memref<16x16xf16, 1>
      memref.dealloc %b : memref<16x16xf16, 1>
      scf.condition(%true) %arg : index
    } do {
    ^bb0(%arg: index):
      scf.yield %arg : index
    }
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }

  func.func @multi_block_region() {
    %outer = memref.alloc() : memref<16x16xf16, 1>
    scf.execute_region {
      cf.br ^body
    ^body:
      %inner = memref.alloc() : memref<8x8xf16, 1>
      memref.dealloc %inner : memref<8x8xf16, 1>
      scf.yield
    }
    memref.dealloc %outer : memref<16x16xf16, 1>
    return
  }

  func.func @carried_alias(%c0: index, %c1: index, %c2: index) {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %zero = arith.constant 0 : index
    %carried = scf.for %i = %c0 to %c2 step %c1
        iter_args(%handle = %a) -> (memref<16x16xf16, 1>) {
      %view = memref.cast %handle
          : memref<16x16xf16, 1> to memref<16x16xf16, 1>
      %loaded = memref.load %view[%zero, %zero] : memref<16x16xf16, 1>
      scf.yield %handle : memref<16x16xf16, 1>
    }
    %used = memref.load %carried[%zero, %zero] : memref<16x16xf16, 1>
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }

  func.func @balanced_diamond() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %cond = arith.constant true
    cf.cond_br %cond, ^left, ^right
  ^left:
    memref.dealloc %a : memref<16x16xf16, 1>
    cf.br ^join
  ^right:
    memref.dealloc %a : memref<16x16xf16, 1>
    cf.br ^join
  ^join:
    return
  }
}
