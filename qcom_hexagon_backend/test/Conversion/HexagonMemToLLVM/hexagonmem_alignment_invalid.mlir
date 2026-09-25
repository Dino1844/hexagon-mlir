// RUN: not linalg-hexagon-opt %s -hexagonmem-to-llvm 2>&1 | FileCheck %s

// The allocation contract requires a non-negative alignment.  A malformed
// negative value must not be normalized into the 128-byte default.

// CHECK: alignment' failed to satisfy constraint

func.func @invalid_alignment() {
  %alloc = hexagonmem.alloc() {alignment = -1 : i64} : memref<256xi8, 1>
  hexagonmem.dealloc %alloc : memref<256xi8, 1>
  return
}
