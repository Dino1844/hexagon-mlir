// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(convert-to-hexagonmem))' | FileCheck %s

// hmx-partition creates a 256-byte conversion-state memref.  The conversion
// must retain that explicit alignment instead of rebuilding the allocation with
// hexagonmem.alloc's 128-byte default.

// CHECK-LABEL: func.func @forward_alignment
// CHECK: %[[STATE:.*]] = hexagonmem.alloc() {alignment = 256 : i64} : memref<256xi8, 1>
// CHECK: hexagonmem.dealloc %[[STATE]] : memref<256xi8, 1>

func.func @forward_alignment() {
  %state = memref.alloc() {alignment = 256 : i64} : memref<256xi8, 1>
  memref.dealloc %state : memref<256xi8, 1>
  return
}
