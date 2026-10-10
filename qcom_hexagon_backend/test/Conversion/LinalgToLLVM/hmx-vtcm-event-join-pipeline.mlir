//===- hmx-vtcm-event-join-pipeline.mlir - static to ABI per-site join ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The smallest honest static-to-runtime per-site join needs three links, and
// this file proves the first two on the *real* pipeline.  The third (an
// observed runtime event carrying the same words) is
// exp/hmx/vtcm_accounting_probe/event_join.py, and it stays not-proven until a
// gate-ON capture exists.
//
//   1. static: `hmx-vtcm-accounting` publishes one canonical site, an opaque
//      event token derived from it, and the scope/function/site identities in
//      two separate sidecars;
//   2. ABI: `hmx-to-llvm` carries exactly those words into the runtime entry
//      point, spans the allocation, and leaves before every return.
//
// Both come from the same run here, so the two cannot drift apart the way a
// hand-typed sidecar and a separately checked call would.  The RUN line pipes
// the same output into the sibling checker, which binds the nine call operands
// to the six sidecar words -- a FileCheck pattern cannot, because the constants
// are hoisted into `llvm.mlir.constant` results whose names carry no contract
// and two operands share one value.
//
// The shape is deliberately one function with one VTCM allocation site, because
// that is the only scope the event token is defined for.  A Triton matmul is
// not usable here and neither is a two-resident-weight fixture: `matmul-to-hmx`
// emits four VTCM sites for one matmul (bias state, activation bridge, resident
// weight, accumulator), so the identity is `not-proven` and the token would
// have no single site to name.  The second module therefore stays on transient
// sites and shows the refusal: two canonical sites in one function yield an
// ineligible record and *neither* half of the ABI.
//
// Nothing here is a backend option, an admission decision, a grid authority, or
// a v2/v3 manifest field, and no device is involved.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s
// The checker is told which module symbols to check, and that list is required:
// it cannot use position.  With `-split-input-file` the async-runtime pass
// prepends a bare module holding its private runtime declarations, so the first
// printed module is not the first input module. Naming the two fixture modules
// keeps the check exact -- a missing, renamed, or duplicated module fails, and a
// sidecar appearing in an unnamed module fails -- while the declaration-only
// preamble is simply not a target.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-diagnostic-record)' | python3 "$(dirname %s)/hmx-vtcm-event-join-pipeline.py" --expect n2_event_join_pipeline --expect n2_two_site_pipeline
// The checker's own attribute reader is the one piece that is not a direct
// equality check, so it has its own tests: the whitespace-after-`=` shape, the
// multi-group token that broke integer parsing, and the rejections that must
// survive both fixes.
// RUN: python3 "$(dirname %s)/hmx-vtcm-event-join-parser-test.py"
//===----------------------------------------------------------------------===//

// -----

// One DDR input and one VTCM allocation: the input stays in address space 0
// because a space-1 function argument is external VTCM and the census refuses
// to publish bytes for memory it does not own.  The allocation carries a file
// location because the canonical site identity is (module, function, source,
// role, slot) -- without a source fact the site has no identity at all.
//
// `hmx.build_id` and `hmx.kernel_vtcm_grid` are explicitly supplied contract
// inputs. They are never synthesized from IR contents, a pointer, or a build
// path: no build identity means no token, and a grid that is not declared 1
// means no eligible record.
//
// The module prints its attributes inline and in sorted order, so the checks
// follow that order: markers, census, event context, grid, identity.
// CHECK-LABEL: module @n2_event_join_pipeline
// CHECK-DAG: hmx.build_id = {high = 11 : i64, low = 7 : i64}
// CHECK-DAG: hmx.diagnostic_vtcm_accounting
// CHECK-DAG: hmx.diagnostic_vtcm_evidence_context
// CHECK-DAG: hmx.diagnostic_vtcm_identity

// The census is complete, and the identity names exactly one function and one
// canonical site, which is what makes the event context eligible.
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: allocation_sites = 1 : i64
// CHECK-DAG: external_scratch = "none"
// CHECK-DAG: external_vtcm = "none"
// CHECK-DAG: status = "complete"
// CHECK-DAG: transient_bytes = 128 : i64
// CHECK: hmx.kernel_vtcm_event_context = {
// CHECK-DAG: eligible = true
// CHECK-DAG: function = "kernel"
// CHECK-DAG: grid_product = 1 : i64
// CHECK-DAG: invocation_id = 1 : i64
// CHECK-DAG: immutable = 1 : i64
// CHECK-DAG: kind = "vtcm-event-context"
// CHECK-DAG: mode = "diagnostic-only"
// CHECK-DAG: schema = "hmx.vtcm-event-context/v1"
// CHECK-DAG: status = "complete"
// CHECK-DAG: token_basis = "hmx.vtcm-event-token/fnv1a128/v1"
// CHECK-DAG: token_bits = 128 : i64
// CHECK-DAG: accounting_scope_id = {{-?[0-9]+}} : i64
// CHECK-DAG: function_id = {{-?[0-9]+}} : i64
// CHECK-DAG: allocation_site_id = {{-?[0-9]+}} : i64
// CHECK-DAG: token_low = {{-?[0-9]+}} : i64
// CHECK-DAG: token_high = {{-?[0-9]+}} : i64

// The identity sidecar is the independent record of what the token was derived
// from: same scope, same function, one site, same token.  Whether the six
// numbers above are equal to the six numbers in the emitted call is the sibling
// checker's job, because FileCheck cannot bind a hoisted constant to a field.
// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: function_count = 1 : i64
// CHECK-DAG: identity_status = "complete"
// CHECK-DAG: principal = "n2_event_join_pipeline"
// CHECK-DAG: principal_status = "module-symbol"
// CHECK-DAG: site_count = 1 : i64
// CHECK-DAG: site_id_basis = "module-principal+function-symbol+source+role+slot"
// CHECK-DAG: functions = [{function = "kernel"
// CHECK-DAG: sites = [{event_token_basis = "hmx.vtcm-event-token/fnv1a128/v1"
// CHECK-DAG: identity_status = "complete"
// CHECK-DAG: role = "transient-hexagonmem"
// CHECK-DAG: source = "file:n2_event_join:2:1"

// The lowered kernel: the enter call opens the span, the VTCM allocation is
// inside it, and the leave precedes the single return.  Nothing address-shaped
// is emitted, and the declared grid/invocation scope is still `not-proven` --
// this pass does not observe a launch.
// CHECK: llvm.func @kernel
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_event_context_enter_v1_dsp
// CHECK: llvm.call @hexagon_runtime_alloc_1d_dsp
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_event_context_leave_v1_dsp
// CHECK: llvm.return
// CHECK-NOT: 0x
// CHECK-NOT: ptr=
// CHECK-NOT: %p

module @n2_event_join_pipeline attributes {
    hmx.build_id = {low = 7 : i64, high = 11 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2_event_join":2:1)
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

// The same kernel with a second VTCM allocation.  Both sites are transient, so
// the strict resident contract is never entered and the refusal is about the
// site count alone: an event token names one site, and choosing one of two
// would be a guess the compiler must not make.  The allocations still happen;
// the record simply refuses to attribute them.
//
// CHECK-LABEL: module @n2_two_site_pipeline
// CHECK: hmx.kernel_vtcm_accounting = {
// CHECK-DAG: allocation_sites = 2 : i64
// CHECK-DAG: status = "complete"
// CHECK: hmx.kernel_vtcm_event_context = {
// CHECK-DAG: eligible = false
// CHECK-DAG: immutable = 0 : i64
// CHECK-DAG: status = "not-proven"
// CHECK-DAG: reason = "event context requires one function and one canonical site"
// CHECK: hmx.kernel_vtcm_identity = {
// CHECK-DAG: site_count = 2 : i64
// CHECK: llvm.func @kernel
// CHECK: llvm.call @hexagon_runtime_alloc_1d_dsp
// CHECK-NOT: event_context_enter_v1_dsp
// CHECK-NOT: event_context_leave_v1_dsp

module @n2_two_site_pipeline attributes {
    hmx.build_id = {low = 7 : i64, high = 11 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context} {
  func.func @kernel(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2_two_site":2:1)
    %c = memref.alloc() : memref<64xf16, 1> loc("n2_two_site":3:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                                     affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b, %c : memref<64xf16, 1>, memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      linalg.yield %x, %x : f16, f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    return
  }
}
