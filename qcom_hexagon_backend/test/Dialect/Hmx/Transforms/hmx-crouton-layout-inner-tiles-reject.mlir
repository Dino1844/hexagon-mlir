//===- hmx-crouton-layout-inner-tiles-reject.mlir - inner_tiles arity gate -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The reject arm of hmx-crouton-layout.mlir, which carries the positive half:
// the crouton layout is linalg.pack with inner_tiles = [2], i.e. exactly one
// tiling factor for the one entry in inner_dims_pos.
//
// inner_tiles = [2, 1] looks equivalent -- it splits the row dimension the same
// way -- but the unit tile is a second tiling factor, and the rank-2 source has
// only one dim in inner_dims_pos. linalg.pack rejects it before any folding
// happens, with:
//
//   tiling factors must equal the number of dimensions to tile
//
// That message was previously mis-stated in hmx-crouton-layout.mlir as
// "packed rank != (unpacked rank + num tiling factors)", and no test pinned
// either string. This file pins the real one.
//
// RUN: linalg-hexagon-opt %s -fold-pack-unpack-constants -verify-diagnostics

// expected-error @+3 {{tiling factors must equal the number of dimensions to tile}}
func.func @bad_inner_tiles(%w: tensor<32x32xf16>) -> tensor<16x32x2x1xf16> {
  %e = tensor.empty() : tensor<16x32x2x1xf16>
  %p = linalg.pack %w inner_dims_pos = [0] inner_tiles = [2, 1] into %e
       : tensor<32x32xf16> -> tensor<16x32x2x1xf16>
  return %p : tensor<16x32x2x1xf16>
}
