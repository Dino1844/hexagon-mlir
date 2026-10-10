//===- scf-threading-mutex-rejected.mlir - scf threading has no peer -------===//
//
// SCF threading rewrites the loop nest into scf.parallel; multi-threading, VTCM
// tiling and the external scratch buffer each expect to be the sole owner of
// that nest / of the per-instance VTCM budget. The combination used to be
// guarded by an assert, which is compiled out of Release/NDEBUG builds and
// silently emitted the undefined pipeline. It is now an explicit rejection at
// the same translation boundary as the other contract checks, so it fires in
// every build configuration. One RUN per conflicting arm.
//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-scf-threading=true enable-multi-threading=true})' 2>&1 | FileCheck %s
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-scf-threading=true enable-vtcm-tiling=true})' 2>&1 | FileCheck %s
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-to-llvm{enable-scf-threading=true scratch=1024})' 2>&1 | FileCheck %s
// CHECK: error: enableSCFThreading is incompatible with enableMultiThreading
//===----------------------------------------------------------------------===//

module {
  func.func @scf_threading_mutex(%a: memref<64x64xf32>, %b: memref<64x64xf32>,
                                 %c: memref<64x64xf32>) {
    linalg.matmul ins(%a, %b : memref<64x64xf32>, memref<64x64xf32>)
                 outs(%c : memref<64x64xf32>)
    return
  }
}
