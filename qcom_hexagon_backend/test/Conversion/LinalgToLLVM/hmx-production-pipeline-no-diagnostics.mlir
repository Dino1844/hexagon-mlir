//===- hmx-production-pipeline-no-diagnostics.mlir ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The production pipeline schedules no diagnostic pass.
//
// The two HMX diagnostic passes (the VTCM allocation census and the record-only
// v3 document) used to be mounted unconditionally inside `linalg-to-llvm` and
// were made inert with an internal marker. They are now reachable only from the
// `hmx-diagnostic-record` entry point, which lives in the HmxDiagnostics
// library -- linked into `linalg-hexagon-opt` only, so no diagnostic pass code is
// linked into the backend library that ships to the device either.
//
// What this pins is the part that is a property of the IR rather than of the
// linker: the production pipeline produces no census sidecar, even when the
// module carries the internal markers that a diagnostic run uses to ask for
// them. A module that asks for the census and does not get it is exactly the
// regression this guards.
//
// Note the v3 *skeleton* is deliberately not in the exclusion list: that one is
// published by `matmul-to-hmx` from compile-time facts and is part of the
// production path. Only the sidecars the census pass itself emits are pinned out.
//===----------------------------------------------------------------------===//

// BARE arm -- the production pipeline, on a module that carries all three markers.
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s --check-prefix=BARE
// BARE-NOT: hmx.kernel_vtcm_accounting
// BARE-NOT: hmx.kernel_vtcm_identity
// BARE-NOT: hmx.kernel_vtcm_live_range

// DIAG arm -- the same module, through the diagnostic entry, does get them. This
// what makes the BARE arm a statement about scheduling rather than about the
// census being broken.
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record)' | FileCheck %s --check-prefix=DIAG
// DIAG: hmx.kernel_vtcm_accounting = {
// DIAG: kind = "allocation-site-census"
// DIAG: status = "complete"

//===----------------------------------------------------------------------===//

module attributes {hmx.diagnostic_vtcm_accounting,
                   hmx.diagnostic_vtcm_liveness,
                   hmx.diagnostic_v3_record} {
  func.func @production_no_diagnostics(%a: tensor<64x64xf16>,
                                       %b: tensor<64x64xf16>,
                                       %c: tensor<64x64xf32>) {
    %zero = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%zero : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return
  }
}
