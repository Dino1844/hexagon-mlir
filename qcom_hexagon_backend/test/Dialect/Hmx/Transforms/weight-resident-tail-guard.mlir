//===- weight-resident-tail-guard.mlir - the two view-matcher gaps --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Two shapes a real kernel's weight bridge reads through, which the matcher
// used to refuse silently (docs/hmx/pack-redundancy-fix-plan-2026-10-09.md §2.2).
// Both are refusals of the *same* proof the unguarded form already gives, so
// both are matched now; every refusal this pass can still reach is reported
// with a reason, which is what the `-verify-diagnostics` cases below pin --
// deleting any of those remarks turns the file red.
//
//   [1] tail guard: the masked load of the last N block is an `scf.if` whose
//       arms `memref.cast` into one result type, so the pack source is the
//       `scf.if`'s result and has no `reinterpret_cast` defining op. Both arms
//       must read the same entry-argument view; the gather arm reads a prefix
//       of it. The differing columns are the ones past N, which a correct
//       kernel never stores (see the note on `underlyingSliceArgument`).
//   [2] static-step loop induction variable: the N block offset is the IV of
//       `scf.for %n0 = 0 to N step BN`, not a `pid * BN` multiply. Both the
//       tile-edge proof and the "provably a multiple of N means rows" guard
//       read it through `isMultipleOf`, which understands the IV.
//   [3]-[5] the refusals that must stay refusals, each with its reason.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(weight-resident{prepack-runtime-weights=true}))' -verify-diagnostics -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// [1] tail-guarded N block with a loop IV offset: whole-weight residency, the
// bridge and the guard both go away.
// CHECK: hmx.weight_prepack = "[{\22func\22:\22runtime_weight_guarded\22,\22slot\22:1,\22shape\22:[64,64],\22crouton\22:[2,2,16,32,2]
// CHECK: hmx.weight_resident_bytes = 8192 : i64
module {
  // CHECK-LABEL: func.func @runtime_weight_guarded
  // CHECK-NOT: scf.if
  // CHECK-NOT: hmx.pack_weight
  // CHECK: %[[ADDR:.*]] = memref.extract_aligned_pointer_as_index
  // CHECK: %[[W:.*]] = hexagonmem.alloc(%[[ADDR]]) {hmx.weight_resident = {address, bytes = 8192 : i64}} : memref<2x2x16x32x2xf16, 1>
  // CHECK: %[[N0C:.*]] = arith.divui
  // CHECK: memref.subview %[[W]][%[[N0C]], 0, 0, 0, 0]
  func.func @runtime_weight_guarded(%a: memref<64x64xf16>,
                                    %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0.0 : f16
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    scf.for %n0 = %c0 to %c64 step %c32 {
      %view = memref.reinterpret_cast %w to offset: [%n0], sizes: [64, 32],
          strides: [64, 1]
          : memref<*xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
      %left = arith.subi %c64, %n0 : index
      %tail = arith.cmpi slt, %left, %c32 : index
      %src = scf.if %tail -> (memref<64x32xf16, strided<[?, ?], offset: ?>>) {
        %valid = arith.maxsi %left, %c0 : index
        %buf = memref.alloc() : memref<64x32xf16>
        %short = arith.cmpi slt, %valid, %c32 : index
        scf.if %short {
          linalg.fill ins(%zero : f16) outs(%buf : memref<64x32xf16>)
        }
        %sv = memref.subview %view[0, 0] [64, %valid] [1, 1]
            : memref<64x32xf16, strided<[64, 1], offset: ?>>
            to memref<64x?xf16, strided<[64, 1], offset: ?>>
        %sb = memref.subview %buf[0, 0] [64, %valid] [1, 1]
            : memref<64x32xf16> to memref<64x?xf16, strided<[32, 1]>>
        memref.copy %sv, %sb
            : memref<64x?xf16, strided<[64, 1], offset: ?>>
            to memref<64x?xf16, strided<[32, 1]>>
        %cast = memref.cast %buf
            : memref<64x32xf16> to memref<64x32xf16, strided<[?, ?], offset: ?>>
        scf.yield %cast : memref<64x32xf16, strided<[?, ?], offset: ?>>
      } else {
        %cast = memref.cast %view
            : memref<64x32xf16, strided<[64, 1], offset: ?>>
            to memref<64x32xf16, strided<[?, ?], offset: ?>>
        scf.yield %cast : memref<64x32xf16, strided<[?, ?], offset: ?>>
      }
      scf.for %i = %c0 to %c2 step %c1 {
        %r = arith.divui %i, %c1 : index
        %cc = arith.remui %i, %c1 : index
        hmx.pack_weight ins(%src, %r, %cc
                : memref<64x32xf16, strided<[?, ?], offset: ?>>)
            outs(%wa : memref<1x2x16x32x2xf16, 1>)
      }
      hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>,
                            memref<1x2x16x32x2xf16, 1>)
          outs(%ar : memref<2x1x16x32x2xf16, 1>)
    }
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}

// -----

// [2] the guard's arms read different views: one arm gathers from another
// argument's view, so there is no single N block to make resident. Declined,
// and the decline says which proof failed.
module {
  // CHECK-LABEL: func.func @runtime_weight_guarded_mixed_arms
  // CHECK: hmx.pack_weight
  func.func @runtime_weight_guarded_mixed_arms(%a: memref<64x64xf16>,
                                               %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0.0 : f16
    %true = arith.constant true
    %n0 = arith.muli %c0, %c32 : index
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    %view = memref.reinterpret_cast %w to offset: [%n0], sizes: [64, 32],
        strides: [64, 1]
        : memref<*xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
    // The other arm's view: same geometry, a different argument, so a
    // different `reinterpret_cast` operation.
    %other = memref.reinterpret_cast %a to offset: [%n0], sizes: [64, 32],
        strides: [64, 1]
        : memref<64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
    %src = scf.if %true -> (memref<64x32xf16, strided<[?, ?], offset: ?>>) {
      %buf = memref.alloc() : memref<64x32xf16>
      %valid = arith.maxsi %c32, %c0 : index
      linalg.fill ins(%zero : f16) outs(%buf : memref<64x32xf16>)
      %sv = memref.subview %other[0, 0] [64, %valid] [1, 1]
          : memref<64x32xf16, strided<[64, 1], offset: ?>>
          to memref<64x?xf16, strided<[64, 1], offset: ?>>
      %sb = memref.subview %buf[0, 0] [64, %valid] [1, 1]
          : memref<64x32xf16> to memref<64x?xf16, strided<[32, 1]>>
      memref.copy %sv, %sb
          : memref<64x?xf16, strided<[64, 1], offset: ?>>
          to memref<64x?xf16, strided<[32, 1]>>
      %cast = memref.cast %buf
          : memref<64x32xf16> to memref<64x32xf16, strided<[?, ?], offset: ?>>
      scf.yield %cast : memref<64x32xf16, strided<[?, ?], offset: ?>>
    } else {
      %cast = memref.cast %view
          : memref<64x32xf16, strided<[64, 1], offset: ?>>
          to memref<64x32xf16, strided<[?, ?], offset: ?>>
      scf.yield %cast : memref<64x32xf16, strided<[?, ?], offset: ?>>
    }
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c1 : index
      %cc = arith.remui %i, %c1 : index
      hmx.pack_weight ins(%src, %r, %cc
              : memref<64x32xf16, strided<[?, ?], offset: ?>>)
          outs(%wa : memref<1x2x16x32x2xf16, 1>)
    }
// expected-remark @+1 {{resident weight declined: the tail guard's arms do not all read the same entry-argument N view; the weight keeps its per-launch pack bridge}}
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>,
                          memref<1x2x16x32x2xf16, 1>)
        outs(%ar : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}

// -----

// [3] loop IV whose step is not a multiple of the tile edge: the crouton
// subview's index could not be divided exactly, so residency stays refused --
// and says so instead of dropping the reason on the floor.
module {
  // CHECK-LABEL: func.func @runtime_weight_iv_unaligned
  // CHECK: hmx.pack_weight
  func.func @runtime_weight_iv_unaligned(%a: memref<64x64xf16>,
                                         %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    scf.for %n0 = %c0 to %c64 step %c16 {
      %view = memref.reinterpret_cast %w to offset: [%n0], sizes: [64, 32],
          strides: [64, 1]
          : memref<*xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
      scf.for %i = %c0 to %c2 step %c1 {
        %r = arith.divui %i, %c1 : index
        %cc = arith.remui %i, %c1 : index
        hmx.pack_weight ins(%view, %r, %cc
                : memref<64x32xf16, strided<[64, 1], offset: ?>>)
            outs(%wa : memref<1x2x16x32x2xf16, 1>)
      }
// expected-remark @+1 {{resident weight declined: the N offset is not provably a multiple of 32, so the resident's crouton subview cannot be indexed exactly; the weight keeps its per-launch pack bridge}}
      hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>,
                            memref<1x2x16x32x2xf16, 1>)
          outs(%ar : memref<2x1x16x32x2xf16, 1>)
    }
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}

// -----

// [4] loop IV whose step IS a whole number of rows: the offset is provably a
// multiple of N, so it selects rows (a K block) and not an N block. The guard
// is the same one the `pid * BN` form answers; the loop form has to answer it
// too, which is what keeps a loop-varying row block from becoming a resident
// that holds one block while the offset walks past its end.
module {
  // CHECK-LABEL: func.func @runtime_weight_iv_row_offset
  // CHECK: hmx.pack_weight
  func.func @runtime_weight_iv_row_offset(%a: memref<64x64xf16>,
                                          %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    scf.for %n0 = %c0 to %c64 step %c64 {
      %view = memref.reinterpret_cast %w to offset: [%n0], sizes: [64, 32],
          strides: [64, 1]
          : memref<*xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
      scf.for %i = %c0 to %c2 step %c1 {
        %r = arith.divui %i, %c1 : index
        %cc = arith.remui %i, %c1 : index
        hmx.pack_weight ins(%view, %r, %cc
                : memref<64x32xf16, strided<[64, 1], offset: ?>>)
            outs(%wa : memref<1x2x16x32x2xf16, 1>)
      }
// expected-remark @+1 {{resident weight declined: the offset is provably a multiple of N, so it selects rows (a K block), not an N block; the weight keeps its per-launch pack bridge}}
      hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>,
                            memref<1x2x16x32x2xf16, 1>)
          outs(%ar : memref<2x1x16x32x2xf16, 1>)
    }
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}

// -----

// [5] a source that is neither a view nor a guard over one: the matcher's own
// decline, reported with the reason it reached.
module {
  // CHECK-LABEL: func.func @runtime_weight_internal_buffer
  // CHECK: hmx.pack_weight
  func.func @runtime_weight_internal_buffer(%a: memref<64x64xf16>,
                                            %w: memref<*xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.0 : f16
    %ca = memref.alloc() : memref<2x2x16x32x2xf16, 1>
    %wa = memref.alloc() : memref<1x2x16x32x2xf16, 1>
    %ar = memref.alloc() : memref<2x1x16x32x2xf16, 1>
    // An internal buffer, not a view of either argument: nothing this pass
    // publishes can describe it.
    %internal = memref.alloc() : memref<64x32xf16>
    linalg.fill ins(%zero : f16) outs(%internal : memref<64x32xf16>)
    scf.for %i = %c0 to %c2 step %c1 {
      %r = arith.divui %i, %c1 : index
      %cc = arith.remui %i, %c1 : index
      hmx.pack_weight ins(%internal, %r, %cc : memref<64x32xf16>)
          outs(%wa : memref<1x2x16x32x2xf16, 1>)
    }
// expected-remark @+1 {{resident weight declined: the pack source is neither an entry-argument view nor a tail guard over one; the weight keeps its per-launch pack bridge}}
    hmx.matmul ins(%ca, %wa : memref<2x2x16x32x2xf16, 1>,
                          memref<1x2x16x32x2xf16, 1>)
        outs(%ar : memref<2x1x16x32x2xf16, 1>)
    memref.dealloc %wa : memref<1x2x16x32x2xf16, 1>
    memref.dealloc %ca : memref<2x2x16x32x2xf16, 1>
    memref.dealloc %ar : memref<2x1x16x32x2xf16, 1>
    return
  }
}
