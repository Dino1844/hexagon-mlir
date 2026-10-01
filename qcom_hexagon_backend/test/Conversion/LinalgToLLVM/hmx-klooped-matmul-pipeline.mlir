//===- hmx-klooped-matmul-pipeline.mlir - K-looped matmul end to end -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The classic tiled GEMM shape: the contraction's K is walked in a loop and the
// running partial product is carried as the `scf.for` iter_arg -- so the
// `linalg.matmul`'s `outs` initialiser is a NON-ZERO, NON-CONSTANT value.
//
// Why this test exists. Every matmul kernel in this tree is whole-block (one
// `tl.dot` over the entire K), so `MatmulToHmxPass`'s accumulator handling had
// no coverage for the carried case:
//
//   * `isEmptyInit()` (`MatmulToHmxPass.cpp:96-99`) is the check for letting an
//     engine that CLEARS its hardware accumulator take the op -- `linalg.matmul`
//     adds into C, `hmx.matmul` overwrites. A carried C fails that check.
//   * The `residual` path (:1028-1032, :1101-1103) is what covers it: the
//     original C term is added back after the read-out.
//
// It matters beyond coverage. Whole-block is exactly the form that cannot
// express a real contraction: Triton caps one block at 2**20 elements and
// requires every block dim to be a power of two
// (`triton/python/triton/_utils.py:68,71`), so K = 14336 -- LLaMA-3 8B's FFN
// width -- is rejected in the front end, and even a power-of-two K of that size
// would make the weight that must fit VTCM whole be `K x N`, not `K x N_block`.
// In the tiled form each `tl.dot` sees only BLOCK_K, so the weight per
// contraction is `BLOCK_K x BLOCK_N` and stays small by construction.
//
// Measured host-side before writing this: the same shape reaches HMX with
// `plan = "full-hmx"` and the weight at 64 KB per contraction, where the
// whole-block form is refused by the front end outright.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// The contraction is accepted, and it is the per-block K (128) that reaches the
// engine -- not the loop's full extent. That is the whole point of the form:
// each iteration is its own small, VTCM-fitting contraction.
// CHECK: hmx.kernel_manifest =
// CHECK-DAG: plan = "full-hmx"
// CHECK-DAG: reason = "selected-aligned"
// CHECK-DAG: full = {k = 128 : i64, m = 64 : i64, n = 64 : i64}
// CHECK-DAG: blocking = "whole"

// The weight is a function argument here, so it rides the resident-prepack
// policy (host-side prepack) rather than a per-launch pack leaf.
// CHECK-DAG: policy = "resident-prepack"

// The engine is reached, in the four steps of one contraction.
// CHECK: llvm.call @hmx_pack_act_f16_bulk
// CHECK: llvm.call @hmx_mma_f16
// CHECK: llvm.call @hmx_unpack_acc_f16_bulk

// The carried accumulator is added back AFTER the read-out -- the residual
// path. Without it the loop would silently keep only the last K block's
// product, which is the failure this test is here to catch.
// CHECK: llvm.fadd

module {
  // K = 256, walked as 2 x 128. `%init` is an opaque function argument: neither
  // a `tensor.empty` nor a zero splat, so the accumulator is provably carried
  // and the residual path is the only correct lowering.
  func.func @klooped(%a: tensor<64x128xf16>, %w: tensor<128x64xf16>,
                     %init: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %r = scf.for %i = %c0 to %c2 step %c1
        iter_args(%acc = %init) -> (tensor<64x64xf16>) {
      %m = linalg.matmul ins(%a, %w : tensor<64x128xf16>, tensor<128x64xf16>)
                         outs(%acc : tensor<64x64xf16>) -> tensor<64x64xf16>
      scf.yield %m : tensor<64x64xf16>
    }
    return %r : tensor<64x64xf16>
  }
}
