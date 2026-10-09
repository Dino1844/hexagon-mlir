//===- hmx-weight-resident-ddr-pipeline.mlir - a weight too big for VTCM ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The end-to-end shape of a runtime weight that does not fit the persistent
// VTCM pool. Its crouton image is 5242880 B and the bridge's own array is
// counted beside it (the documented conservatism of `admitResidentVtcm`), so
// the placement gate sends the image to the permanent DDR mirror instead of
// declining the weight -- and the pack that used to build that image on the
// device is gone, because the host already built it and the kernel now takes
// the block it needs with one contiguous copy.
//
// Everything the contract and the manifest say about the weight is pinned here
// because the three have to agree: `location` in `hmx.weight_prepack`,
// `resident-prepack-ddr` in `weight_policies`, and the DDR entry in the
// lowered call. The VTCM footprint aggregate is deliberately absent: a mirror
// is not out of the pool, and charging it to the pool would make every later
// admission refuse.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-weight-resident=true})' | FileCheck %s --implicit-check-not=hmx_pack_weight --implicit-check-not=hmx.weight_resident_bytes
//===----------------------------------------------------------------------===//

// CHECK: hmx.kernel_manifest =
// CHECK-DAG: function = "runtime_weight_ddr"
// CHECK-DAG: pack_weight_sites = 0 : i64
// CHECK-DAG: weight_policies = [{consumers = [0], function = "runtime_weight_ddr", policy = "resident-prepack-ddr", reason = "eligible-aligned-f16", slot = 1 : i64}]
// CHECK-DAG: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_ddr\22,\22slot\22:1,\22shape\22:[1280,2048],\22crouton\22:[64,40,16,32,2],\22dtype\22:\22f16\22,\22location\22:\22ddr\22}]"
// No pool footprint for a mirror (the RUN line's implicit check-not covers the
// whole module), and no per-launch pack leaf anywhere in the object either.
// The runtime gets the argument's address through the mirror's own entry.
// CHECK: llvm.func @hexagon_runtime_weight_resident_ddr_v2_dsp(i64, i32, i32) -> !llvm.ptr
// CHECK-LABEL: llvm.func @runtime_weight_ddr
// CHECK: llvm.call @hexagon_runtime_weight_resident_ddr_v2_dsp
// The fetch: mirror -> the VTCM array the engine reads, one contiguous copy.
// CHECK: llvm.call @hexagon_runtime_copy_dsp
module {
  func.func @runtime_weight_ddr(%a: tensor<64x1280xf16>,
                                %w: tensor<1280x2048xf16>) -> tensor<64x2048xf16> {
    %empty = tensor.empty() : tensor<64x2048xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x2048xf16>)
                       -> tensor<64x2048xf16>
    %m = linalg.matmul ins(%a, %w : tensor<64x1280xf16>, tensor<1280x2048xf16>)
                       outs(%c : tensor<64x2048xf16>) -> tensor<64x2048xf16>
    return %m : tensor<64x2048xf16>
  }
}
