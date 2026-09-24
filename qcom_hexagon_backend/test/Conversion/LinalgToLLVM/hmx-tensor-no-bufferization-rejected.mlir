//===- hmx-tensor-no-bufferization-rejected.mlir - tensor path needs bufferization -===//
//
// A tensor-valued linalg entry (the ABI shape produced by Triton) cannot be
// translated with bufferization disabled. The LinalgToLLVM boundary rejects it
// explicitly instead of allowing an unrelated downstream legalization failure;
// this is distinct from the manually managed memref record-only fixture.
//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-bufferization=false})' 2>&1 | FileCheck %s
// CHECK: tensor-valued Linalg operations require enableBufferization=true
//===----------------------------------------------------------------------===//

module {
  func.func @tensor_matmul(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}
