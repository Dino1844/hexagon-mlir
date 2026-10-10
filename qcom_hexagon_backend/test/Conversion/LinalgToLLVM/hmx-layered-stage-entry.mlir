//===- hmx-layered-stage-entry.mlir - the stop entry, pinned ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The layered-lit entry itself: `hmx-diagnostic-record`'s
// `stop-after-diagnostic-stage`, which ends the production sequence right
// after its one stage hook (the census plus the marker-gated record document
// that follows it) instead of lowering through translation.
//
// Three arms, because one arm could pass vacuously:
//
//   LAYERED -- the stage output. The census sidecar is there (the module asks
//     for it), and so are the placement-layer facts every mid-pipeline
//     mechanism leaves behind: the resident VTCM allocations, the mma, the
//     read-out. This is what a test about a mid-pipeline mechanism wants to
//     assert, without the rest of the pipeline having to agree first.
//
//   LAYERED-NOT -- the same output must NOT contain the lowered forms. Without
//     this the first arm would still pass if the stop silently did nothing
//     (the stage facts survive lowering too), which is the failure mode that
//     would make every layered assertion worthless.
//
//   FULL -- the same module through the same entry without the stop, which
//     still lowers all the way. This is the arm that keeps the equivalence
//     honest: a stop option that also changed the full path would be a
//     production change wearing a test-only hat.
//
// The two runs use the same `production=...` text, so the only difference is
// the stop flag itself.
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record{production=enable-workspace-resident=true stop-after-diagnostic-stage})' | FileCheck %s --check-prefix=LAYERED
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-diagnostic-record{production=enable-workspace-resident=true})' | FileCheck %s --check-prefix=FULL

// The census ran at the stage and its sidecar is on the module, next to the
// manifest the placement layer finalized.
// LAYERED: hmx.kernel_manifest = {
// LAYERED: hmx.kernel_vtcm_accounting = {
// LAYERED-DAG: status = "complete"
// The placement layer is intact: resident allocations, the engine leaves, the
// staged ring's ops -- mid-pipeline forms that no lowered run would show.
// LAYERED-DAG: hexagonmem.alloc() {alignment = 256 : i64, hmx.workspace_resident
// LAYERED-DAG: hexagonmem.alloc(%intptr) {hmx.weight_resident
// LAYERED-DAG: hmx.mma
// LAYERED-DAG: hmx.unpack_acc

// ... and nothing was lowered: no llvm.func at all, in particular not the
// runtime entry the full run declares.
// LAYERED-NOT: llvm.func
// LAYERED-NOT: llvm.call

// The full arm reaches translation, which is what makes the stop a stop. The
// module attributes come first in the output, so they are checked before the
// lowered declarations (a plain CHECK fixes the position the DAGs follow).
// FULL: hmx.kernel_vtcm_accounting = {
// FULL-DAG: status = "complete"
// FULL: llvm.func @hexagon_runtime_workspace_resident_v2_dsp
// FULL: llvm.call @hmx_mma_f16

module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @layered_entry(%a: tensor<64x64xf16>,
                           %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>)
                       -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}
