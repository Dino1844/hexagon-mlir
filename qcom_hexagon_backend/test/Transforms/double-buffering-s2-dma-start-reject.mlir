//===- double-buffering-s2-dma-start-reject.mlir --------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Reject paths of stage-2 double buffering: when `createDMAStartOp` refuses a
// `memref.copy` (here: source and target in the same memory space), the pass
// must fail loudly and leave the copy alone. Before the fix each site erased
// the copy (and the store site emitted the dma_wait too) regardless of the
// outcome, silently dropping the data movement.
//
// RUN: linalg-hexagon-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(func.func(hexagon-double-buffer-generic-s2))'
//===----------------------------------------------------------------------===//

// The rejected copy is the prologue preload: source (default space) and
// target (also default space) share a memory space, so no dma_start is
// created. The copy must survive and the pass must fail.
func.func @preload_same_space_rejected(%arg0: memref<128x64xf32>, %arg1: memref<128x64xf32>, %arg2: memref<128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %false = arith.constant false
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
  %toggle = memref.alloc() : memref<i1>
  memref.store %true, %toggle[] : memref<i1>
  %ping0 = memref.alloc() {alignment = 2048 : i64} : memref<64x64xf32, 1>
  %pong0 = memref.alloc() {alignment = 2048 : i64} : memref<64x64xf32, 1>
  %preload_ddr = memref.alloc() : memref<64x64xf32>
  %0 = arith.cmpi slt, %c0, %c128 : index
  scf.if %0 {
    %sv = memref.subview %arg0[%c0, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
    // expected-error @+1 {{failed to create memref.dma_start}}
    memref.copy %sv, %preload_ddr : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32>
  } {db_generic = 0 : i64, db_prologue}
  scf.for %iv = %c0 to %c128 step %c64 {
    %is_ping = memref.load %toggle[] : memref<i1>
    %next_iv = arith.addi %iv, %c64 : index
    %next_next_iv = arith.addi %next_iv, %c64 : index
    %not_last = arith.cmpi sle, %next_next_iv, %c128 : index
    scf.if %is_ping {
      scf.if %not_last {
        %sv = memref.subview %arg0[%next_iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %sv, %pong0 : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, 1>
      } {db_prefetch}
      scf.for %i = %c0 to %c64 step %c1 {
        scf.yield
      }
      %wsv = memref.subview %alloc[%iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      memref.copy %ping0, %wsv : memref<64x64xf32, 1> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      memref.store %false, %toggle[] : memref<i1>
    } {db_ping_kernel}
    %not_ping = arith.xori %is_ping, %true : i1
    scf.if %not_ping {
      scf.if %not_last {
        %sv = memref.subview %arg0[%next_iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %sv, %ping0 : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, 1>
      } {db_prefetch}
      scf.for %i = %c0 to %c64 step %c1 {
        scf.yield
      }
      %wsv = memref.subview %alloc[%iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      memref.copy %pong0, %wsv : memref<64x64xf32, 1> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      memref.store %true, %toggle[] : memref<i1>
    } {db_pong_kernel}
  } {db_generic = 0 : i64}
  memref.copy %alloc, %arg2 : memref<128x64xf32> to memref<128x64xf32>
  return
}

// -----

// The rejected copy is a db_prefetch copy: ping's prefetch into a default
// space buffer matches the source's default space. No dma_start, no erase,
// pass failure.
func.func @prefetch_same_space_rejected(%arg0: memref<128x64xf32>, %arg1: memref<128x64xf32>, %arg2: memref<128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %false = arith.constant false
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
  %toggle = memref.alloc() : memref<i1>
  memref.store %true, %toggle[] : memref<i1>
  %ping0 = memref.alloc() {alignment = 2048 : i64} : memref<64x64xf32, 1>
  %pong0 = memref.alloc() {alignment = 2048 : i64} : memref<64x64xf32, 1>
  %pong0_ddr = memref.alloc() : memref<64x64xf32>
  %0 = arith.cmpi slt, %c0, %c128 : index
  scf.if %0 {
    %sv = memref.subview %arg0[%c0, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
    memref.copy %sv, %ping0 : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, 1>
  } {db_generic = 0 : i64, db_prologue}
  scf.for %iv = %c0 to %c128 step %c64 {
    %is_ping = memref.load %toggle[] : memref<i1>
    %next_iv = arith.addi %iv, %c64 : index
    %next_next_iv = arith.addi %next_iv, %c64 : index
    %not_last = arith.cmpi sle, %next_next_iv, %c128 : index
    scf.if %is_ping {
      scf.if %not_last {
        %sv = memref.subview %arg0[%next_iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        // expected-error @+1 {{failed to create memref.dma_start}}
        memref.copy %sv, %pong0_ddr : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32>
      } {db_prefetch}
      scf.for %i = %c0 to %c64 step %c1 {
        scf.yield
      }
      %wsv = memref.subview %alloc[%iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      memref.copy %ping0, %wsv : memref<64x64xf32, 1> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      memref.store %false, %toggle[] : memref<i1>
    } {db_ping_kernel}
    %not_ping = arith.xori %is_ping, %true : i1
    scf.if %not_ping {
      scf.if %not_last {
        %sv = memref.subview %arg0[%next_iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %sv, %ping0 : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, 1>
      } {db_prefetch}
      scf.for %i = %c0 to %c64 step %c1 {
        scf.yield
      }
      %wsv = memref.subview %alloc[%iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      memref.copy %pong0, %wsv : memref<64x64xf32, 1> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      memref.store %true, %toggle[] : memref<i1>
    } {db_pong_kernel}
  } {db_generic = 0 : i64}
  memref.copy %alloc, %arg2 : memref<128x64xf32> to memref<128x64xf32>
  return
}

// -----

// The rejected copy is ping's store-back: both operands are default space.
// This is the site that used to emit the dma_wait unconditionally before
// erasing the copy. Ping is processed first, so exactly one error is
// expected even though pong's store is rejected too.
func.func @store_same_space_rejected(%arg0: memref<128x64xf32>, %arg1: memref<128x64xf32>, %arg2: memref<128x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %false = arith.constant false
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
  %toggle = memref.alloc() : memref<i1>
  memref.store %true, %toggle[] : memref<i1>
  %ping0 = memref.alloc() {alignment = 2048 : i64} : memref<64x64xf32, 1>
  %pong0 = memref.alloc() {alignment = 2048 : i64} : memref<64x64xf32, 1>
  %store_ddr = memref.alloc() : memref<64x64xf32>
  %store_dst = memref.alloc() : memref<64x64xf32>
  %0 = arith.cmpi slt, %c0, %c128 : index
  scf.if %0 {
    %sv = memref.subview %arg0[%c0, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
    memref.copy %sv, %ping0 : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, 1>
  } {db_generic = 0 : i64, db_prologue}
  scf.for %iv = %c0 to %c128 step %c64 {
    %is_ping = memref.load %toggle[] : memref<i1>
    %next_iv = arith.addi %iv, %c64 : index
    %next_next_iv = arith.addi %next_iv, %c64 : index
    %not_last = arith.cmpi sle, %next_next_iv, %c128 : index
    scf.if %is_ping {
      scf.if %not_last {
        %sv = memref.subview %arg0[%next_iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %sv, %pong0 : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, 1>
      } {db_prefetch}
      scf.for %i = %c0 to %c64 step %c1 {
        scf.yield
      }
      // expected-error @+1 {{failed to create memref.dma_start}}
      memref.copy %store_ddr, %store_dst : memref<64x64xf32> to memref<64x64xf32>
      memref.store %false, %toggle[] : memref<i1>
    } {db_ping_kernel}
    %not_ping = arith.xori %is_ping, %true : i1
    scf.if %not_ping {
      scf.if %not_last {
        %sv = memref.subview %arg0[%next_iv, 0] [64, 64] [1, 1] : memref<128x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        memref.copy %sv, %ping0 : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, 1>
      } {db_prefetch}
      scf.for %i = %c0 to %c64 step %c1 {
        scf.yield
      }
      memref.copy %store_ddr, %store_dst : memref<64x64xf32> to memref<64x64xf32>
      memref.store %true, %toggle[] : memref<i1>
    } {db_pong_kernel}
  } {db_generic = 0 : i64}
  memref.copy %alloc, %arg2 : memref<128x64xf32> to memref<128x64xf32>
  return
}
