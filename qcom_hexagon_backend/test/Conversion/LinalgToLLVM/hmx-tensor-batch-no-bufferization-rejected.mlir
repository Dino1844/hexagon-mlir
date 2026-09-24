//===- hmx-tensor-batch-no-bufferization-rejected.mlir - batch tensor path --===//
//
// A tensor-valued batch contraction must not evade the no-bufferization guard
// by being reduced to a matmul/generic after the initial Linalg scan.
//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-bufferization=false})' 2>&1 | FileCheck %s
// CHECK: tensor-valued Linalg operations require enableBufferization=true
//===----------------------------------------------------------------------===//

module {
  func.func @tensor_batch_matmul(%a: tensor<2x64x64xf16>,
                                 %b: tensor<2x64x64xf16>,
                                 %c: tensor<2x64x64xf16>) -> tensor<2x64x64xf16> {
    %0 = linalg.batch_matmul ins(%a, %b : tensor<2x64x64xf16>, tensor<2x64x64xf16>)
                          outs(%c : tensor<2x64x64xf16>) -> tensor<2x64x64xf16>
    return %0 : tensor<2x64x64xf16>
  }
}
