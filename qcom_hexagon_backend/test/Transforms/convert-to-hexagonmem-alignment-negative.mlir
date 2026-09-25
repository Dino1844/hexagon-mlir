// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(convert-to-hexagonmem))'

// The ConvertToHexagonmem -> HexagonMem boundary must reject an explicit
// unsupported alignment rather than letting the runtime default it.

func.func @zero() {
  // expected-error @+1 {{failed to satisfy constraint: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  %a = memref.alloc() {alignment = 0 : i64} : memref<256xi8, 1>
  memref.dealloc %a : memref<256xi8, 1>
  return
}

// -----

func.func @three() {
  // expected-error @+1 {{failed to satisfy constraint: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0}}
  %a = memref.alloc() {alignment = 3 : i64} : memref<256xi8, 1>
  memref.dealloc %a : memref<256xi8, 1>
  return
}

// -----

func.func @four_k() {
  // expected-error @+1 {{alignment must be a power of two in [1, 2048]}}
  %a = memref.alloc() {alignment = 4096 : i64} : memref<256xi8, 1>
  memref.dealloc %a : memref<256xi8, 1>
  return
}
