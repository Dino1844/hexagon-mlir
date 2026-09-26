//===- hmx-vtcm-site-scope-lowering-reject.mlir - bracket refusals -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Three ways the bracket contract refuses, all of which are silent-wrong-answer
// failures if they were tolerated:
//
//  * a stamp that is not the closed per-site shape. Every word becomes a runtime
//    constant, so a missing or extra field would have to be defaulted or
//    ignored -- and a zeroed identity names a site nobody published.
//  * a stamp on a DDR allocation, which never reaches the VTCM pool. Accepting
//    it would attribute a host allocation to a VTCM site. The case is written as
//    a `hexagonmem.alloc` with a non-VTCM type, because a plain `memref.alloc` is
//    not an operand of this lowering at all and so could never be checked here.
//  * a stamp whose constants are outside the site ABI -- here a grid of two. The
//    ABI has exactly one invocation ordinal to spend, so a grid of two names no
//    site.
//  * an eligible site table that reaches hmx-to-llvm with no bracket anywhere.
//    The table and the kernel would then disagree about whether the attribution
//    was ever requested, and a host would read a per-site claim the device was
//    never asked to make.
//
// The stamps here are hand-written and the passes are run standalone on purpose.
// Through the full pipeline the accounting pass is the *producer* of both the
// stamps and the table, so it would overwrite every malformed value below before
// the lowering ever saw it; a full-pipeline test of these refusals would prove
// nothing about the lowering. The producer's own refusals are covered in
// ../../Dialect/Hmx/Transforms/hmx-vtcm-site-scopes{,-reject}.mlir.
//
// RUN: not linalg-hexagon-opt %s -split-input-file -hexagonmem-to-llvm 2>&1 | FileCheck %s --check-prefixes=LOWER
// The second run is only about the module table: hmx-to-llvm has no opinion on a
// per-op stamp, so the first four modules are expected to pass through it
// untouched. What it must never do is accept an eligible table with no bracket.
// RUN: not linalg-hexagon-opt %s -split-input-file -hmx-to-llvm 2>&1 | FileCheck %s --check-prefixes=FRAME
//===----------------------------------------------------------------------===//

// -----

// A stamp missing its site token is refused rather than defaulted: the token is
// the only word that names which of two sites this is.
// LOWER: hmx.vtcm_site_scope is not the closed per-site stamp shape
module @n2_stamp_missing_token {
  func.func @kernel() {
    %b = hexagonmem.alloc() {hmx.vtcm_site_scope = {
        accounting_scope_id = 7 : i64, build_id_high = 9 : i64, build_id_low = 5 : i64,
        function = "kernel", function_id = 11 : i64, grid_product = 1 : i64,
        invocation_id = 1 : i64, kind = "vtcm-site-scope", role = "transient-hexagonmem",
        schema = "hmx.vtcm-site-scopes/v1", site_id = 13 : i64, slot = 0 : i64,
        source = "file:n2:7:1", status = "complete",
        token_basis = "hmx.vtcm-site-token/fnv1a128/v1", token_bits = 128 : i64,
        token_high = 3 : i64}} : memref<64xf16, 1> loc("n2":7:1)
    memref.dealloc %b : memref<64xf16, 1>
    return
  }
}

// -----

// A stamp whose token is all zero is refused: the runtime ABI treats a zero
// token as "no site named", so a bracket carrying it would silently widen the
// claim from one site to the whole invocation.
// LOWER: hmx.vtcm_site_scope carries a zero identity the site ABI refuses to bind
module @n2_stamp_zero_token {
  func.func @kernel() {
    %b = hexagonmem.alloc() {hmx.vtcm_site_scope = {
        accounting_scope_id = 7 : i64, build_id_high = 9 : i64, build_id_low = 5 : i64,
        function = "kernel", function_id = 11 : i64, grid_product = 1 : i64,
        invocation_id = 1 : i64, kind = "vtcm-site-scope", role = "transient-hexagonmem",
        schema = "hmx.vtcm-site-scopes/v1", site_id = 13 : i64, slot = 0 : i64,
        source = "file:n2:7:1", status = "complete",
        token_basis = "hmx.vtcm-site-token/fnv1a128/v1", token_bits = 128 : i64,
        token_high = 0 : i64, token_low = 0 : i64}} : memref<64xf16, 1> loc("n2":7:1)
    memref.dealloc %b : memref<64xf16, 1>
    return
  }
}

// -----

// A stamp whose basis is the *frame* token schema is refused -- a closed stamp
// with the wrong identity basis. The two tokens are
// different claims over different tuples, and accepting the wrong basis would
// hand the runtime a frame token where a site token belongs.
// LOWER: hmx.vtcm_site_scope is missing a required word or carries a value outside the site ABI
module @n2_stamp_frame_token {
  func.func @kernel() {
    %b = hexagonmem.alloc() {hmx.vtcm_site_scope = {
        accounting_scope_id = 7 : i64, build_id_high = 9 : i64, build_id_low = 5 : i64,
        function = "kernel", function_id = 11 : i64, grid_product = 1 : i64,
        invocation_id = 1 : i64, kind = "vtcm-site-scope", role = "transient-hexagonmem",
        schema = "hmx.vtcm-site-scopes/v1", site_id = 13 : i64, slot = 0 : i64,
        source = "file:n2:7:1", status = "complete",
        token_basis = "hmx.vtcm-event-token/fnv1a128/v1", token_bits = 128 : i64,
        token_high = 3 : i64, token_low = 2 : i64}} : memref<64xf16, 1> loc("n2":7:1)
    memref.dealloc %b : memref<64xf16, 1>
    return
  }
}

// -----

// A stamp on a DDR allocation is a producer disagreement: the site table only
// names pool-backed sites, so a stamp here means the two halves disagree about
// what this operation allocates.
// LOWER: hmx.vtcm_site_scope is attached to a DDR allocation
module @n2_stamp_on_ddr {
  func.func @kernel() {
    %b = hexagonmem.alloc() {hmx.vtcm_site_scope = {
        accounting_scope_id = 7 : i64, build_id_high = 9 : i64, build_id_low = 5 : i64,
        function = "kernel", function_id = 11 : i64, grid_product = 1 : i64,
        invocation_id = 1 : i64, kind = "vtcm-site-scope", role = "transient-hexagonmem",
        schema = "hmx.vtcm-site-scopes/v1", site_id = 13 : i64, slot = 0 : i64,
        source = "file:n2:7:1", status = "complete",
        token_basis = "hmx.vtcm-site-token/fnv1a128/v1", token_bits = 128 : i64,
        token_high = 3 : i64, token_low = 2 : i64}} : memref<64xf16> loc("n2":7:1)
    memref.dealloc %b : memref<64xf16>
    return
  }
}

// -----

// A stamp that is closed but carries a grid of two names no site: the ABI spends
// exactly one invocation ordinal, so there is nothing for the second instance to
// be distinguished from.
// LOWER: hmx.vtcm_site_scope is missing a required word or carries a value outside the site ABI
module @n2_stamp_wrong_constants {
  func.func @kernel() {
    %b = hexagonmem.alloc() {hmx.vtcm_site_scope = {
        accounting_scope_id = 7 : i64, build_id_high = 9 : i64, build_id_low = 5 : i64,
        function = "kernel", function_id = 11 : i64, grid_product = 2 : i64,
        invocation_id = 1 : i64, kind = "vtcm-site-scope", role = "transient-hexagonmem",
        schema = "hmx.vtcm-site-scopes/v1", site_id = 13 : i64, slot = 0 : i64,
        source = "file:n2:7:1", status = "complete",
        token_basis = "hmx.vtcm-site-token/fnv1a128/v1", token_bits = 128 : i64,
        token_high = 3 : i64, token_low = 2 : i64}} : memref<64xf16, 1> loc("n2":7:1)
    memref.dealloc %b : memref<64xf16, 1>
    return
  }
}

// -----

// A hand-written eligible table with no bracket anywhere is refused by
// hmx-to-llvm. Nothing brackets a pool-backed allocation here, so the table
// would otherwise be a per-site attribution the device was never asked to make.
// FRAME: eligible hmx.kernel_vtcm_site_scopes table has no
// FRAME-SAME: hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp call
module @n2_unbracketed_table attributes {hmx.kernel_vtcm_site_scopes = {
      accounting_scope_id = 7 : i64, build_id_high = 9 : i64, build_id_low = 5 : i64,
      eligible = true, function = "kernel", function_id = 11 : i64,
      grid_product = 1 : i64, immutable = 1 : i64, invocation_id = 1 : i64,
      kind = "vtcm-site-scopes", mode = "diagnostic-only", module = "n2_unbracketed_table",
      performance_claimed = 0 : i64, principal = "n2_unbracketed_table",
      schema = "hmx.vtcm-site-scopes/v1", site_count = 1 : i64, status = "complete",
      token_basis = "hmx.vtcm-site-token/fnv1a128/v1",
      sites = [{constant_bounded_extent = 0 : i64, function = "kernel",
                function_id = 11 : i64, requested_bytes = 128 : i64,
                role = "transient-hexagonmem", site_id = 13 : i64, slot = 0 : i64,
                source = "file:n2:7:1",
                token_basis = "hmx.vtcm-site-token/fnv1a128/v1",
                token_bits = 128 : i64, token_high = 3 : i64, token_low = 2 : i64}]}} {
  func.func @kernel(%a: memref<64xf16>) {
    return
  }
}
