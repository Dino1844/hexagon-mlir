//===- hmx-tile-span-notice.mlir - the tile notice's compile-side inputs -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The launch-time tile recommendation (hmx_tile_notice in backend/utils.py) is
// computed from three published facts and nothing else: the record's static
// logical shape, execution.block_m, and the committed/budget VTCM bytes. This
// test runs the real pipeline and feeds the real manifest to the real reporter,
// so the fields it reads are pinned to what the compiler emits rather than to
// what a host fixture happens to carry.
//
// Three cases, because the answer has three shapes:
//
//   * a contraction the bridge must walk in M blocks (block_m below M): the
//     notice fires and its numbers come from the manifest;
//   * a whole-block contraction: no notice at all, which is what keeps this
//     off the default path -- every kernel this tree compiles today is in this
//     case, and the 2026-10-09 bench of the user's 12 shapes used tiles that
//     all land here too (exp/hmx/op_bench/mm_user_shapes.py TILES);
//   * a contraction refused on the budget: no notice from this reporter, since
//     a refused record publishes no bridge plan to recommend a block for (the
//     refusal reporter answers that one).
//
// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' | %PYTHON "$(dirname %s)/hmx-tile-span-notice.py"
//===----------------------------------------------------------------------===//

// 4096x4096x64: the whole bridge needs 34603008 bytes against the 8 MiB pool.
// The weight alone fits, so M is walked in blocks -- the largest tile divisor
// that fits is 16 tiles = 512 rows, hence 8 spans.
module {
  func.func @one_dot_for_a_4096x4096_output(%a: tensor<4096x64xf16>, %b: tensor<64x4096xf16>) -> tensor<4096x4096xf16> {
    %c = tensor.empty() : tensor<4096x4096xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<4096x64xf16>, tensor<64x4096xf16>)
                       outs(%c : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
    return %0 : tensor<4096x4096xf16>
  }
}

// -----

// 64x64x64: 24576 bytes, one span, one read-out. The default case.
module {
  func.func @whole_block(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}

// -----

// 64x4096x1024: the weight is 1024*4096*2 = 8388608 bytes, the whole pool, so
// not even one 32-row tile fits beside it. The contraction is refused and the
// record carries no execution block at all.
module {
  func.func @weight_alone_fills_the_pool(%a: tensor<64x1024xf16>, %b: tensor<1024x4096xf16>) -> tensor<64x4096xf16> {
    %c = tensor.empty() : tensor<64x4096xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x1024xf16>, tensor<1024x4096xf16>)
                       outs(%c : tensor<64x4096xf16>) -> tensor<64x4096xf16>
    return %0 : tensor<64x4096xf16>
  }
}
