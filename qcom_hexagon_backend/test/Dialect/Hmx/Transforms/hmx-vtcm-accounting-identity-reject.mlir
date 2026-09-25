//===- hmx-vtcm-accounting-identity-reject.mlir - identity fail closed -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The identity marker is strict: an unknown source location, a repeated
// source/role/slot tuple, and an allocation without a unique function scope
// cannot produce a join key.  The raw census remains a separate diagnostic.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(hmx-vtcm-accounting)'
//===----------------------------------------------------------------------===//

// expected-error @+1 {{HMX VTCM static identity is not proven; unknown source location}}
module @identity_module attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity} {
  func.func @unknown_source() {
    %a = memref.alloc() : memref<64x64xf16, 1> loc(unknown)
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM static identity is not proven; duplicate or ambiguous allocation-site identity}}
module @identity_module attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity} {
  func.func @same_source() {
    %a = memref.alloc() : memref<64x64xf16, 1> loc("same":20:1)
    %b = memref.alloc() : memref<32x32xf16, 1> loc("same":20:1)
    memref.dealloc %a : memref<64x64xf16, 1>
    memref.dealloc %b : memref<32x32xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM static identity is not proven; missing or ambiguous function scope}}
module @identity_module attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity} {
  %orphan = memref.alloc() : memref<64x64xf16, 1> loc("orphan":30:1)
}

// -----

// Two file locations in one fused location are ambiguous even when their
// textual file/line/column values happen to match.
// expected-error @+1 {{HMX VTCM static identity is not proven; unknown source location}}
module @identity_module attributes {hmx.diagnostic_vtcm_accounting,
                                    hmx.diagnostic_vtcm_identity} {
  func.func @ambiguous_fused_source() {
    %a = memref.alloc() : memref<16x16xf16, 1>
        loc(fused["first":50:1, "second":50:1])
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// Even an empty allocation census cannot turn an anonymous module into a
// proven process principal.
// expected-error @+1 {{HMX VTCM static identity is not proven; missing explicit module symbol}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_identity} {
}

// -----

// A module without an explicit symbol has no stable principal.  Even when
// the allocation site itself is otherwise reviewable, strict identity must
// report the missing principal rather than treating the placeholder as a
// process identity.
// expected-error @+1 {{HMX VTCM static identity is not proven; missing explicit module symbol}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_identity} {
  func.func @anonymous_principal() {
    %a = memref.alloc() : memref<16x16xf16, 1> loc("anonymous":40:1)
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}
