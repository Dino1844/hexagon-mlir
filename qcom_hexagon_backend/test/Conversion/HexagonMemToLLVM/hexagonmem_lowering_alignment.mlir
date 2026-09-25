// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s

// The alignment on a VTCM allocation is part of the runtime allocation ABI.
// This is the non-resident counterpart of the 256-byte conversion-state
// allocation emitted by hmx-partition.

// CHECK: llvm.func @hexagon_runtime_alloc_1d_dsp(i32, i64, i1) -> !llvm.ptr
// CHECK-DAG: %[[ALIGN:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK: llvm.call @hexagon_runtime_alloc_1d_dsp({{.*}}, %[[ALIGN]], %{{.*}}) : (i32, i64, i1) -> !llvm.ptr
// CHECK: llvm.call @hexagon_runtime_free_1d_dsp

func.func @alignment() {
  %alloc = hexagonmem.alloc() {alignment = 256 : i64} : memref<256xi8, 1>
  hexagonmem.dealloc %alloc : memref<256xi8, 1>
  return
}
