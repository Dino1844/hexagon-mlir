//===- hmx-support-priority.mlir - canonical refusal ordering --------------===//
//
// Several inputs miss more than one HMX condition. The manifest must expose
// one stable first reason, in the order implemented by AttributionTally.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s --check-prefix=PRIORITY
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-allocator=false}))' -split-input-file | FileCheck %s --check-prefix=ALLOC
//===----------------------------------------------------------------------===//

// PRIORITY: hmx.kernel_manifest = {
// PRIORITY: function = "priority"
// PRIORITY-SAME: id = 0 : i64
// PRIORITY-SAME: plan = "hvx"
// PRIORITY-SAME: reason = "library-call"
// PRIORITY-SAME: function = "priority"
// PRIORITY-SAME: id = 1 : i64
// PRIORITY-SAME: plan = "hvx"
// PRIORITY-SAME: reason = "unsupported-dtype"
// PRIORITY-SAME: function = "priority"
// PRIORITY-SAME: id = 2 : i64
// PRIORITY-SAME: plan = "hvx"
// PRIORITY-SAME: reason = "min-rows"
// PRIORITY-SAME: function = "priority"
// PRIORITY-SAME: id = 3 : i64
// PRIORITY-SAME: plan = "hvx"
// PRIORITY-SAME: reason = "tile-alignment"
module {
  func.func @priority(
      %a: tensor<64x31xf16>, %b: tensor<31x64xf16>,
      %x: tensor<?x31xf16>, %n: index,
      %d0: tensor<2x31xf64>, %d1: tensor<31x2xf64>,
      %small_a: tensor<2x31xf16>, %small_b: tensor<31x31xf16>)
      -> (tensor<?x64xf16>, tensor<2x2xf64>, tensor<2x31xf16>, tensor<64x64xf16>) {
    %c0 = tensor.empty(%n) : tensor<?x64xf16>
    %m0 = linalg.matmul {library_call = "custom_mm"}
                       ins(%x, %b : tensor<?x31xf16>, tensor<31x64xf16>)
                       outs(%c0 : tensor<?x64xf16>) -> tensor<?x64xf16>

    %c1 = tensor.empty() : tensor<2x2xf64>
    %m1 = linalg.matmul ins(%d0, %d1 : tensor<2x31xf64>, tensor<31x2xf64>)
                       outs(%c1 : tensor<2x2xf64>) -> tensor<2x2xf64>

    %c2 = tensor.empty() : tensor<2x31xf16>
    %m2 = linalg.matmul ins(%small_a, %small_b : tensor<2x31xf16>, tensor<31x31xf16>)
                       outs(%c2 : tensor<2x31xf16>) -> tensor<2x31xf16>

    %c3 = tensor.empty() : tensor<64x64xf16>
    %m3 = linalg.matmul ins(%a, %b : tensor<64x31xf16>, tensor<31x64xf16>)
                       outs(%c3 : tensor<64x64xf16>) -> tensor<64x64xf16>
    return %m0, %m1, %m2, %m3 : tensor<?x64xf16>, tensor<2x2xf64>, tensor<2x31xf16>, tensor<64x64xf16>
  }
}

// -----

// Shape attribution is checked before allocator capability. With the
// allocator disabled, this dynamic matmul therefore reports the shape reason
// rather than the environment reason.
// ALLOC: hmx.kernel_manifest = {
// ALLOC: function = "allocator_priority"
// ALLOC: plan = "hvx"
// ALLOC: reason = "dynamic-shape"
module {
  func.func @allocator_priority(%a: tensor<?x64xf16>, %b: tensor<64x64xf16>,
                                %m: index) -> tensor<?x64xf16> {
    %c = tensor.empty(%m) : tensor<?x64xf16>
    %0 = linalg.matmul ins(%a, %b : tensor<?x64xf16>, tensor<64x64xf16>)
                       outs(%c : tensor<?x64xf16>) -> tensor<?x64xf16>
    return %0 : tensor<?x64xf16>
  }
}
