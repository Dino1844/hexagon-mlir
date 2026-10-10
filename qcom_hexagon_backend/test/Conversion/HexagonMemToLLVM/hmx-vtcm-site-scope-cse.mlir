//===- hmx-vtcm-site-scope-cse.mlir - brackets survive CSE -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The site brackets are evidence, not bookkeeping. Two sites in one kernel
// produce two identical-looking enter calls that differ only in their constant
// operands, and a later CSE or canonicalization pass is exactly the kind of
// thing that would be tempted to merge them -- after which both allocations
// would be attributed to one site, which is the failure the whole ABI exists to
// prevent.
//
// The calls are therefore side-effecting and `noinline`, and this test pins the
// consequence: after a full CSE + canonicalize pass over the lowered kernel
// there are still two enters, two leaves, each still wrapped around its own
// allocation, and the two token/site operand pairs are still different.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s --check-prefixes=RAW
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | linalg-hexagon-opt -pass-pipeline='builtin.module(cse,canonicalize)' | FileCheck %s --check-prefixes=CSE
//===----------------------------------------------------------------------===//

module @n2_cse_stable attributes {
    hmx.build_id = {low = 5 : i64, high = 9 : i64},
    hmx.kernel_vtcm_grid = 1 : i64,
    hmx.diagnostic_vtcm_accounting,
    hmx.diagnostic_vtcm_identity,
    hmx.diagnostic_vtcm_evidence_context,
    hmx.diagnostic_vtcm_site_scope} {
  func.func @n2_cse_stable(%a: memref<64xf16>) {
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

// RAW-LABEL: llvm.func @n2_cse_stable
// RAW: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp
// RAW: llvm.call @hexagon_runtime_alloc_1d_dsp
// RAW: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp
// RAW: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp
// RAW: llvm.call @hexagon_runtime_alloc_1d_dsp
// RAW: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp

// After CSE the order and count are unchanged, and the two brackets still carry
// different token and site operands.
// CSE-LABEL: llvm.func @n2_cse_stable
// CSE-DAG: [[VER:%[0-9]+]] = llvm.mlir.constant(1 : i32)
// CSE-DAG: [[FLAGS:%[0-9]+]] = llvm.mlir.constant(3 : i32)
// CSE: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp([[VER]], [[FLAGS]], [[TOK1L:%[0-9]+]], [[TOK1H:%[0-9]+]], [[SCOPE:%[0-9]+]], [[INVO:%[0-9]+]], [[FUNC:%[0-9]+]], [[SITE1:%[0-9]+]], [[BLD_L:%[0-9]+]], [[BLD_H:%[0-9]+]], [[VER]])
// CSE: llvm.call @hexagon_runtime_alloc_1d_dsp
// CSE: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp
// CSE: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp([[VER]], [[FLAGS]], [[TOK2L:%[0-9]+]], [[TOK2H:%[0-9]+]], [[SCOPE]], [[INVO]], [[FUNC]], [[SITE2:%[0-9]+]], [[BLD_L]], [[BLD_H]], [[VER]])
// CSE: llvm.call @hexagon_runtime_alloc_1d_dsp
// CSE: llvm.call @hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp
// The function ends here, and nothing follows it: a merged or duplicated
// bracket would have to appear before the return, which the two ordered blocks
// above already pin, or after it, which these two checks exclude.
// CSE: llvm.return
// CSE-NOT: site_scope_enter_v1_dsp
// CSE-NOT: site_scope_leave_v1_dsp
