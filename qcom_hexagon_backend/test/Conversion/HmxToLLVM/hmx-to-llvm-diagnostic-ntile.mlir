//===- hmx-to-llvm-diagnostic-ntile.mlir - explicit N tile read-out ------===//
//
// The diagnostic tail bridge keeps the original ranked descriptors and gives
// HmxToLLVM a static physical N-tile coordinate.  The lowering must narrow the
// runtime width for the last tile and add the byte displacement to the
// descriptor-derived address; it must not infer an N tile from `col`.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @diagnostic_ntile
// The first N tile is 32 columns; the second N tile is the 1-column remainder.
// CHECK-DAG: llvm.call @hmx_unpack_acc_f16_bulk
// CHECK-DAG: llvm.call @hmx_unpack_acc_f16_bulk
// CHECK-DAG: llvm.mlir.constant(32 : i32)
// CHECK-DAG: llvm.mlir.constant(1 : i32)
module attributes {hmx.diagnostic_tail_partition} {
func.func @diagnostic_ntile(
    %src: memref<2x2x16x32x2xf16, 1>, %dst: memref<64x33xf16>) {
  %c0 = arith.constant 0 : index
  hmx.unpack_acc ins(%src, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%dst : memref<64x33xf16>) {count = 16 : i64, n_tile = 0 : i64}
  hmx.unpack_acc ins(%src, %c0, %c0 : memref<2x2x16x32x2xf16, 1>)
      outs(%dst : memref<64x33xf16>) {count = 16 : i64, n_tile = 1 : i64}
  return
}
}
