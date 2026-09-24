//===- hmx-hexkl-rejected.mlir - HexKL cannot bypass HMX attribution -------===//
//
// HexKL consumes linalg.matmul before the HMX manifest pass can attribute it.
// The semantic HMX manifest contract therefore rejects the combination explicitly.
//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-hexkl=true})' 2>&1 | FileCheck %s
// CHECK: error: enableHexKL is incompatible with the HMX manifest contract
//===----------------------------------------------------------------------===//

module {
  func.func @hexkl_rejected(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    %c = tensor.empty() : tensor<64x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %0 : tensor<64x64xf16>
  }
}
