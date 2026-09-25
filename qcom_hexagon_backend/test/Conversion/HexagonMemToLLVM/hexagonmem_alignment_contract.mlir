// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics

// The runtime allocator accepts positive powers of two through 2048 bytes.
// These values are representable in MLIR but cannot produce a valid aligned
// allocation, so the HexagonMem verifier must reject them before lowering.

func.func @zero_alignment() {
  // expected-error @+1 {{alignment must be a power of two in [1, 2048]}}
  %a = hexagonmem.alloc() {alignment = 0 : i64} : memref<256xi8, 1>
  return
}

// -----

func.func @non_power_of_two_alignment() {
  // expected-error @+1 {{alignment must be a power of two in [1, 2048]}}
  %a = hexagonmem.alloc() {alignment = 3 : i64} : memref<256xi8, 1>
  return
}

// -----

func.func @above_hardware_quantum_alignment() {
  // expected-error @+1 {{alignment must be a power of two in [1, 2048]}}
  %a = hexagonmem.alloc() {alignment = 4096 : i64} : memref<256xi8, 1>
  return
}
