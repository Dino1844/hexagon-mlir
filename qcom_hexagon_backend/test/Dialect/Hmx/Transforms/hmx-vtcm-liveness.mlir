//===- hmx-vtcm-liveness.mlir - structured VTCM liveness suite ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
// The proven-liveness suite.  These cases used to live in one file per
// boundary shape; they share one RUN line, so they are cases of a single
// -split-input-file fixture.  Each case keeps its own heading and its own
// note verbatim.  The exact site-identity join stays a separate fixture,
// hmx-vtcm-liveness-canonical-site-id.mlir, because five notes below
// point at it by name.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

//===- hmx-vtcm-liveness.mlir - structured linear live upper bound -------===//

// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: kind = "allocation-site-census"
// CHECK-DAG: peak_status = "not-proven"
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region"
// CHECK-DAG: unit = "requested-bytes"
// CHECK-DAG: functions = [{
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 10240 : i64
// CHECK-DAG: peak_site_count = 2 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "linear_liveness"
// CHECK-DAG: transient_requested_peak_bytes = 10240 : i64
// CHECK-DAG: weight_resident_requested_bytes = 0 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 0 : i64
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @linear_liveness_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @linear_liveness() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    %b = memref.alloc() : memref<32x32xf16, 1>
    memref.dealloc %a : memref<64x64xf16, 1>
    memref.dealloc %b : memref<32x32xf16, 1>
    return
  }
}

// -----

//===- hmx-vtcm-liveness-structured.mlir - SCF event analysis -----------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: functions = [{
// CHECK-DAG: symbol = "structured_events"
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: transient_requested_peak_bytes = 10240 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 10240 : i64
// CHECK-DAG: deallocation_sites = 3 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @structured_events() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    %cond = arith.constant true
    scf.if %cond {
      %b = memref.alloc() : memref<32x32xf16, 1>
      memref.dealloc %b : memref<32x32xf16, 1>
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %c = memref.alloc() : memref<16x16xf16, 1>
      memref.dealloc %c : memref<16x16xf16, 1>
      scf.yield
    }
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }
}

// -----

//===- hmx-vtcm-liveness-cfg.mlir - general control-flow liveness ------------===//
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

// -----

//===- hmx-vtcm-liveness-nested-region.mlir - execute-region nesting -----===//
// A single-block scf.execute_region has the same sequential lifetime semantics
// as a nested scf.if/scf.for body when it carries no VTCM values.  This
// fixture keeps that boundary explicit: the region owns a temporary alias and
// releases it before the enclosing loop continues.
//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "nested_execute_region"
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: deallocation_sites = 2 : i64
// CHECK-DAG: transient_requested_peak_bytes = 8704 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 8704 : i64
// CHECK-DAG: peak_site_count = 2 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// CHECK-DAG: control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region"
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @nested_region_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @nested_execute_region() {
    %outer = memref.alloc() : memref<64x64xf16, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.execute_region {
        %inner = memref.alloc() : memref<16x16xf16, 1>
        %alias = memref.cast %inner : memref<16x16xf16, 1>
            to memref<16x16xf16, 1>
        %zero = arith.constant 0 : index
        %value = memref.load %alias[%zero, %zero] : memref<16x16xf16, 1>
        memref.dealloc %inner : memref<16x16xf16, 1>
        scf.yield
      }
      scf.yield
    }
    memref.dealloc %outer : memref<64x64xf16, 1>
    return
  }
}

// -----

//===- hmx-vtcm-liveness-scalar-loop.mlir - non-VTCM loop carry ----------===//
// Region-carried scalar/index values are unrelated to VTCM allocation
// lifetime.  The structured analyzer may admit them; a memref/VTCM value in
// the same position remains fail-closed in the reject fixture.
//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "scalar_loop_carry"
// CHECK-DAG: allocation_sites = 0 : i64
// CHECK-DAG: deallocation_sites = 0 : i64
// CHECK-DAG: transient_requested_peak_bytes = 0 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @scalar_loop_carry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = scf.for %i = %c0 to %c2 step %c1
        iter_args(%carry = %c0) -> (index) {
      scf.yield %carry : index
    }
    return
  }
}

// -----

//===- hmx-vtcm-liveness-aliases.mlir - view alias liveness ----------------===//
// A view is an alias of the principal allocation.  Reading through the view
// before releasing the principal is lifetime-safe; releasing the view itself
// remains an explicit negative case in hmx-vtcm-liveness-rejects.mlir.
//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "view_alias"
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: deallocation_sites = 1 : i64
// CHECK-DAG: transient_requested_peak_bytes = 512 : i64
// CHECK-DAG: peak_site_count = 1 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @view_alias_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @view_alias() {
    %a = memref.alloc() : memref<512xi8, 1>
    %c0 = arith.constant 0 : index
    %view = memref.view %a[%c0][]
        : memref<512xi8, 1> to memref<256xi8, 1>
    %value = memref.load %view[%c0] : memref<256xi8, 1>
    memref.dealloc %a : memref<512xi8, 1>
    return
  }
}

// -----

//===- hmx-vtcm-liveness-production-shaped.mlir - DPS alias liveness ------===//
// The bufferized HMX layout ops return their destination memref as a DPS
// result.  That result is an alias of the destination allocation, not of the
// row-major source.  The liveness probe must follow that edge so a real
// pack/unpack sequence can be covered without treating the result as an
// untracked VTCM allocation.
//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "pack_alias"
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: deallocation_sites = 1 : i64
// CHECK-DAG: transient_requested_peak_bytes = 2048 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 2048 : i64
// CHECK-DAG: peak_site_count = 1 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// CHECK-DAG: allocator_peak_status = "not-proven"
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @pack_alias_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @pack_alias(%src: memref<32x32xf16>, %out: memref<32x32xf16>,
                        %row: index, %col: index) {
    %dst = memref.alloc() : memref<1x1x16x32x2xf16, 1>
    %packed = hmx.pack_act ins(%src, %row, %col : memref<32x32xf16>)
        outs(%dst : memref<1x1x16x32x2xf16, 1>)
        -> memref<1x1x16x32x2xf16, 1>
    %unpacked = hmx.unpack_acc ins(%packed, %row, %col
        : memref<1x1x16x32x2xf16, 1>)
        outs(%out : memref<32x32xf16>) -> memref<32x32xf16>
    memref.dealloc %dst : memref<1x1x16x32x2xf16, 1>
    return
  }
}

// -----

//===- hmx-vtcm-liveness-resident.mlir - resident live floor ------------===//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: functions = [{
// CHECK-DAG: symbol = "resident_floor"
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: deallocation_sites = 0 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 1024 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 1024 : i64
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @resident_floor() {
    %workspace = memref.alloc() {hmx.workspace_resident = {key = -2 : i64, bytes = 1024 : i64}}
        : memref<32x16xf16, 1>
    return
  }
}

// -----

//===- hmx-vtcm-liveness-mixed.mlir - transient plus resident floor --------===//
// The modeled requested bound keeps the transient live range separate from
// the process-resident floor.  This is not an allocator/charged or observed
// pool value.
//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: symbol = "mixed_resident_floor"
// CHECK-DAG: transient_requested_peak_bytes = 8192 : i64
// CHECK-DAG: workspace_resident_requested_bytes = 1024 : i64
// CHECK-DAG: weight_resident_requested_bytes = 0 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 9216 : i64
// CHECK-DAG: allocator_peak_status = "not-proven"
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @mixed_resident_floor() {
    %transient = memref.alloc() : memref<64x64xf16, 1>
    %workspace = memref.alloc()
        {hmx.workspace_resident = {key = -9 : i64, bytes = 1024 : i64}}
        : memref<32x16xf16, 1>
    memref.dealloc %transient : memref<64x64xf16, 1>
    return
  }
}

// -----

//===- hmx-vtcm-liveness-external.mlir - declaration status --------------===//
// A declaration has no executable lifetime to analyze. It must be reported as
// not-applicable and must not be counted as one successfully analyzed function;
// module completeness covers definitions only.
//

// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: function_coverage = "definitions-only"
// CHECK-DAG: definition_count = 1 : i64
// CHECK-DAG: external_declaration_count = 1 : i64
// CHECK-DAG: symbol = "external"
// CHECK-DAG: status = "not-applicable"
// CHECK-DAG: allocation_site_coverage = "not-applicable"
// CHECK-DAG: reason = "external declaration has no body to analyze"
// CHECK-DAG: symbol = "defined"
// CHECK-DAG: status = "complete"
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func private @external()
  func.func @defined() {
    return
  }
}

// -----

//===- hmx-vtcm-liveness-aligned.mlir - size-aligned bound, sidecar only --===//

//
// The size-aligned view of the same liveness result. The sizes here are chosen
// so the quantum is actually observable and both regimes are exercised:
//
//   memref<129xf16,1>  =  258 B  ->  384 B   (128 B quantum, below threshold)
//   memref<1025xf16,1> = 2050 B  -> 4096 B   (2048 B quantum, at threshold)
//
// so the raw peak is 2308 while the aligned peak is 4480. A fixture whose sizes
// were already quantum multiples would show the two figures equal and prove
// nothing about the rounding.
//
// What this file deliberately does NOT claim: 4480 is an upper bound over a
// union-joined live set, not a peak, and not an occupancy. The status fields say
// so, and the v3 `allocator_aligned` block stays not-proven because a bound is
// not the split/coalesce model that block would have to mean.
//
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: functions = [{
// The raw figures, unchanged by this work.
// CHECK-DAG: transient_requested_peak_bytes = 2308 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 2308 : i64
// The aligned figures, with the bound/exact distinction carried in the data.
// CHECK-DAG: transient_aligned_peak_bytes = 4480 : i64
// CHECK-DAG: transient_aligned_peak_status = "upper-bound-not-exact-peak"
// CHECK-DAG: resident_aligned_bytes = 0 : i64
// CHECK-DAG: resident_aligned_status = "exact-process-floor"
// CHECK-DAG: modeled_aligned_peak_bytes = 4480 : i64
// CHECK-DAG: modeled_aligned_peak_status = "upper-bound-not-occupancy"
// The basis is stated, so a reader cannot read the number as a header-inclusive
// or address-padded charge.
// CHECK-DAG: aligned_charge_basis = "runtime-size-quantum-no-header-no-address-padding"
// The aligned peak carries its own attribution. It is tracked separately from
// the raw peak because the two maxima can be attained at different program
// points, so reusing the raw peak's sites here would name the wrong one.
// CHECK-DAG: peak_aligned_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_aligned_site_ids = array<i64:
// CHECK-DAG: status = "complete"
module @aligned_liveness attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @aligned_liveness() {
    %a = memref.alloc() : memref<129xf16, 1>
    %b = memref.alloc() : memref<1025xf16, 1>
    memref.dealloc %a : memref<129xf16, 1>
    memref.dealloc %b : memref<1025xf16, 1>
    return
  }
}

// The incomplete case is not repeated here on purpose. The aligned fields sit
// inside the same `facts.applicable && facts.complete` guard as the raw figures,
// so an incomplete function structurally cannot publish an aligned peak at all
// -- there is no "partially aligned" record to get wrong. The refusals
// themselves are covered by hmx-vtcm-liveness-rejects.mlir.

// -----

//===- hmx-vtcm-liveness-supported.mlir - proven liveness subset ----------===//

// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_sites = 4 : i64
// CHECK-DAG: unknown_allocations = 0 : i64
// CHECK: hmx.kernel_vtcm_live_range = {
// CHECK-DAG: kind = "structured-allocator-events-v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: allocation_site_coverage = "complete"
// CHECK-DAG: control_flow = "finite-cfg-fixpoint-nested-scf-if-for-while-execute-region"
// CHECK-DAG: symbol = "nested_scf"
// CHECK-DAG: peak_status = "structured-upper-bound"
// CHECK-DAG: allocation_sites = 3 : i64
// CHECK-DAG: deallocation_sites = 3 : i64
// CHECK-DAG: transient_requested_peak_bytes = 10240 : i64
// CHECK-DAG: modeled_requested_peak_bytes = 10240 : i64
// CHECK-DAG: symbol = "synchronous_alias_use"
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: deallocation_sites = 1 : i64
// CHECK-DAG: transient_requested_peak_bytes = 512 : i64
// CHECK-DAG: peak_site_count = 1 : i64
// CHECK-DAG: peak_site_id_status = "canonical-static-site-identity"
// CHECK-DAG: peak_site_ids = array<i64:
// Peak site attribution uses the canonical static site identity, never a
// census walk ordinal.  The literal hash depends on this file's path, so
// these fixtures check the status and the presence of the list; the exact
// join against the identity sidecar is covered by
// hmx-vtcm-liveness-canonical-site-id.mlir.
module @supported_subset_module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @nested_scf() {
    %outer = memref.alloc() : memref<64x64xf16, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cond = arith.constant true
    scf.for %i = %c0 to %c2 step %c1 {
      scf.if %cond {
        %then = memref.alloc() : memref<32x32xf16, 1>
        memref.dealloc %then : memref<32x32xf16, 1>
      } else {
        %else = memref.alloc() : memref<16x16xf16, 1>
        memref.dealloc %else : memref<16x16xf16, 1>
      }
      scf.yield
    }
    memref.dealloc %outer : memref<64x64xf16, 1>
    return
  }

  func.func @synchronous_alias_use() {
    %value = memref.alloc() : memref<16x16xf16, 1>
    %scratch = memref.alloc() : memref<4xf16>
    %zero = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    memref.store %one, %value[%zero, %zero] : memref<16x16xf16, 1>
    %loaded = memref.load %value[%zero, %zero] : memref<16x16xf16, 1>
    %alias = memref.cast %value : memref<16x16xf16, 1> to memref<16x16xf16, 1>
    memref.store %loaded, %alias[%zero, %zero] : memref<16x16xf16, 1>
    memref.store %loaded, %scratch[%zero] : memref<4xf16>
    memref.dealloc %value : memref<16x16xf16, 1>
    memref.dealloc %scratch : memref<4xf16>
    return
  }
}
