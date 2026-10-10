//===- hmx-vtcm-site-scope-lowering.mlir - per-site brackets -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The site table is produced by the accounting pass and consumed by the
// hexagonmem lowering. This is the consumption half: every pool-backed
// allocation is bracketed by its own site enter/leave pair, so the runtime knows
// which canonical site the allocation belongs to while the call is made.
//
// The bracket is deliberately the tightest one available -- enter, the
// allocation, leave -- rather than one span around the whole function. A wider
// span would be indistinguishable from the frame context and would make two
// sites in one kernel share one owner. The free is *not* inside the bracket:
// the runtime keeps the owner with the block, so a release long after the span
// closed is still attributed to the site that created it.
//
// The three refusals matter as much as the two brackets:
//
//  * a DDR allocation is never bracketed, because it never reaches the VTCM
//    pool and a stamp there is a producer disagreement;
//  * a stamp that is not the closed shape fails the pass rather than being
//    defaulted, because a zeroed identity would name a site nobody published;
//  * a module that still carries an eligible table after lowering, with no
//    bracket anywhere, fails in hmx-to-llvm: the table and the kernel would
//    otherwise disagree about whether the attribution was ever requested.
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s
//===----------------------------------------------------------------------===//

// -----

// Two canonical transient sites, each bracketed separately. The exact operand
// order is the ABI: version, flags, token low, token high, accounting scope,
// invocation, function, site, build low, build high, grid.
//
// The two ABI declarations, with the exact eleven-word enter signature. They are
// printed before the kernel that calls them, so they are checked before the
// CHECK-LABEL block that starts at the kernel.
// CHECK-DAG: llvm.func @hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp(i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)
// CHECK-DAG: llvm.func @hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp()
// CHECK-LABEL: llvm.func @n2_two_sites
// The eleven operands are the ABI in order: version, flags, token low, token
// high, accounting scope, invocation, function, site, build low, build high,
// grid. Capturing them pins the word order instead of only the presence of a
// call, and reusing the shared captures across both brackets shows the scope,
// invocation, function and build words really are the same while the token and
// site words differ -- two sites, two identities, one kernel.
// CHECK-DAG: [[VER:%[0-9]+]] = llvm.mlir.constant(1 : i32)
// CHECK-DAG: [[FLAGS:%[0-9]+]] = llvm.mlir.constant(3 : i32)
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp([[VER]], [[FLAGS]], [[TOK1L:%[0-9]+]], [[TOK1H:%[0-9]+]], [[SCOPE:%[0-9]+]], [[INVO:%[0-9]+]], [[FUNC:%[0-9]+]], [[SITE1:%[0-9]+]], [[BLD_L:%[0-9]+]], [[BLD_H:%[0-9]+]], [[VER]])
// CHECK: llvm.call @hexagon_runtime_alloc_1d_dsp
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp([[VER]], [[FLAGS]], [[TOK2L:%[0-9]+]], [[TOK2H:%[0-9]+]], [[SCOPE]], [[INVO]], [[FUNC]], [[SITE2:%[0-9]+]], [[BLD_L]], [[BLD_H]], [[VER]])
// CHECK: llvm.call @hexagon_runtime_alloc_1d_dsp
// CHECK: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp
// The frees are outside every bracket: the owner is retained with the block, so
// a release long after the span closed is still attributed to the site that
// created it.
// CHECK: llvm.call @hexagon_runtime_free_1d_dsp
// CHECK-NOT: site_scope_enter
// CHECK-NOT: site_scope_leave
// CHECK: llvm.return
// The stamp is consumed by the lowering, so no per-op attribute survives.
// CHECK-NOT: hmx.vtcm_site_scope
module @n2_two_sites attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  func.func @n2_two_sites(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16, 1> loc("n2":7:1)
    %c = memref.alloc() : memref<64xf16, 1> loc("n2":8:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                                     affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b, %c : memref<64xf16, 1>, memref<64xf16, 1>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      linalg.yield %x, %x : f16, f16
    }
    memref.copy %b, %a : memref<64xf16, 1> to memref<64xf16>
    memref.copy %c, %a : memref<64xf16, 1> to memref<64xf16>
    return
  }
}

// -----

// A DDR allocation is never bracketed. The site table itself refuses to name it,
// so this module reaches the lowering with no stamp and gets no bracket: the DDR
// path is a plain memref.alloc with no runtime allocation call at all.
//
// The module keeps the producer's *ineligible* record -- a typed refusal with a
// reason -- and no op is stamped, so the lowering has nothing to bracket. The
// DDR allocation is a plain host allocation.
// CHECK-LABEL: module @n2_ddr_only
// CHECK: hmx.kernel_vtcm_site_scopes = {
// CHECK-DAG: eligible = false
// CHECK-DAG: status = "not-proven"
// CHECK-DAG: site_count = 0 : i64
// CHECK-DAG: reason = "site scopes require at least one canonical site"
// CHECK: llvm.func @n2_ddr_only
// CHECK: llvm.call @malloc
// CHECK-NOT: site_scope_enter
// CHECK-NOT: site_scope_leave
// CHECK-NOT: hmx.vtcm_site_scope
module @n2_ddr_only attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  func.func @n2_ddr_only(%a: memref<64xf16>) {
    %b = memref.alloc() : memref<64xf16> loc("n2":7:1)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                    iterator_types = ["parallel"]}
        ins(%a : memref<64xf16>) outs(%b : memref<64xf16>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
    }
    return
  }
}
