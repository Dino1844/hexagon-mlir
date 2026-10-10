//===- arg-writes-return-kernel.mlir ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The return arm: a kernel whose result comes back through the return value
// rather than an output pointer (the torch-mlir shape -- `return_types` is
// non-empty, so the launcher's slot list is the returns and the argument write
// set is not consulted at all).
//
// It still carries a write set, because the extractor reads the argument
// signatures and does not know or care which launcher will consume the result:
// here no argument is written (the linalg output chains to a `tensor.empty`),
// so the set is `[]`.  Pinning both arms is what keeps "the field exists for
// every kernel" from silently becoming "the field exists for the kernels that
// need it".
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm)' | FileCheck %s
//
// CHECK-LABEL: llvm.func @return_kernel(
// The lit arm is a lowering check only: this fixture reaches HMX (the matmul
// is admitted), so the interesting part of the pipeline runs.  The claim the
// fixture exists for -- that a return-valued kernel still carries a write set
// (`[]` here) -- lives in the Python-side contract test, because the launcher
// reads it from the envelope and not from the IR.
//
//===----------------------------------------------------------------------===//{{^}}
// the CHECK lines above end here

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @return_kernel(%a: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %w = arith.constant dense<1.000000e+00> : tensor<64x64xf16>
    %empty = tensor.empty() : tensor<64x64xf16>
    %zero = arith.constant 0.000000e+00 : f16
    %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %m = linalg.matmul ins(%a, %w : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m : tensor<64x64xf16>
  }
}
