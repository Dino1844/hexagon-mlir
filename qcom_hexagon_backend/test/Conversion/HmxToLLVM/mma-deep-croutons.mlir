//===- mma-deep-croutons.mlir - a deep hmx.mma lowers at k, not at k+n ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// `hmx.mma`'s `n_croutons` is a REPEAT count the engine applies from the crouton
// it is handed: the op is documented as "starting at position (m, k) ... and
// walking forward from there" (HmxOps.td:474-476). So `n_croutons = 32` with
// Kt = 64 must address crouton (m, k) exactly as `n_croutons = 1` does -- the
// count travels in the THIRD argument only and must not reach the address
// arithmetic `base + (row*rowStride + col) * 2048`.
//
// That separation is what makes the batched K traversal addressable at all:
// emitting floor(Kt/32) mmas of 32 plus one tail of Kt%32, advancing k by 32
// each time, is only correct if the count does not perturb the address. If it
// did, the tail batch would land Kt%32 croutons further along as well.
//
// The K extents below are the ones that tail logic has to survive: 31 / 32 /
// 33 / 64 / 65 -- one short of a batch, exactly one batch, one over, two
// batches, two batches plus a tail. `croutonTileStride` reads the row stride out
// of the memref, so a Kt that is not a multiple of 32 must still yield
// `rowStride == Kt` rather than a rounded-up batch count. @mma_deep_32_kt33 and
// @mma_deep_32_kt65 are the two that would fail if the emitter assumed batch
// multiples.
//
// Each function is checked with one unbroken CHECK-NEXT chain from
// `hmx_acc_clear_f16` to the `hmx_mma_f16` call, over the FULL address
// expression on both operands. That is stronger than a CHECK-NOT for "the count
// does not reach the address": any extra operation inserted between the row
// stride and its multiply is a failure. (A CHECK-NOT written after a CHECK-DAG
// is an empty assertion that passes regardless -- do not use one here.)
//
// @mma_shallow_1_kt64 is the control: it differs from @mma_deep_32_kt64 only in
// the attribute value, so if a change ever folded the count into the address
// the two chains would diverge.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mma_deep_32_kt64
// CHECK: llvm.call @hmx_acc_clear_f16() : () -> ()
// CHECK-NEXT: %[[AA_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[AA_BASE:.*]] = llvm.ptrtoint %[[AA_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[AA_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[AA_OFFT:.*]] = llvm.trunc %[[AA_OFF]] : i64 to i32
// CHECK-NEXT: %[[AA_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[AA_OM:.*]] = llvm.mul %[[AA_OFFT]], %[[AA_EB]]
// CHECK-NEXT: %[[AA_O:.*]] = llvm.add %[[AA_BASE]], %[[AA_OM]]
// CHECK-NEXT: %[[AA_RS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK-NEXT: %[[AA_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[AA_RM:.*]] = llvm.mul %[[AA_R]], %[[AA_RS]]
// CHECK-NEXT: %[[AA_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[AA_T:.*]] = llvm.add %[[AA_RM]], %[[AA_C]]
// CHECK-NEXT: %[[AA_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[AA_BY:.*]] = llvm.mul %[[AA_T]], %[[AA_B]]
// CHECK-NEXT: %[[AA_ADDR:.*]] = llvm.add %[[AA_O]], %[[AA_BY]]
// CHECK-NEXT: %[[AW_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[AW_BASE:.*]] = llvm.ptrtoint %[[AW_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[AW_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[AW_OFFT:.*]] = llvm.trunc %[[AW_OFF]] : i64 to i32
// CHECK-NEXT: %[[AW_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[AW_OM:.*]] = llvm.mul %[[AW_OFFT]], %[[AW_EB]]
// CHECK-NEXT: %[[AW_O:.*]] = llvm.add %[[AW_BASE]], %[[AW_OM]]
// CHECK-NEXT: %[[AW_RS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK-NEXT: %[[AW_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[AW_RM:.*]] = llvm.mul %[[AW_R]], %[[AW_RS]]
// CHECK-NEXT: %[[AW_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[AW_T:.*]] = llvm.add %[[AW_RM]], %[[AW_C]]
// CHECK-NEXT: %[[AW_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[AW_BY:.*]] = llvm.mul %[[AW_T]], %[[AW_B]]
// CHECK-NEXT: %[[AW_ADDR:.*]] = llvm.add %[[AW_O]], %[[AW_BY]]
// CHECK-NEXT: %[[AN:.*]] = llvm.mlir.constant(32 : i32)
// CHECK-NEXT: llvm.call @hmx_mma_f16(%[[AA_ADDR]], %[[AW_ADDR]], %[[AN]]) : (i32, i32, i32) -> ()
func.func @mma_deep_32_kt64(%bias: memref<256xi8, 1>,
                            %act: memref<4x64x16x32x2xf16, 1>,
                            %wt: memref<3x64x16x32x2xf16, 1>,
                            %ar: memref<4x2x16x32x2xf16, 1>,
                            %m: index, %n: index, %k: index) {
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 32 : i32}
      : memref<4x64x16x32x2xf16, 1>, memref<3x64x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}

// CHECK-LABEL: func.func @mma_shallow_1_kt64
// CHECK: llvm.call @hmx_acc_clear_f16() : () -> ()
// CHECK-NEXT: %[[BA_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[BA_BASE:.*]] = llvm.ptrtoint %[[BA_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[BA_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[BA_OFFT:.*]] = llvm.trunc %[[BA_OFF]] : i64 to i32
// CHECK-NEXT: %[[BA_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[BA_OM:.*]] = llvm.mul %[[BA_OFFT]], %[[BA_EB]]
// CHECK-NEXT: %[[BA_O:.*]] = llvm.add %[[BA_BASE]], %[[BA_OM]]
// CHECK-NEXT: %[[BA_RS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK-NEXT: %[[BA_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[BA_RM:.*]] = llvm.mul %[[BA_R]], %[[BA_RS]]
// CHECK-NEXT: %[[BA_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[BA_T:.*]] = llvm.add %[[BA_RM]], %[[BA_C]]
// CHECK-NEXT: %[[BA_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[BA_BY:.*]] = llvm.mul %[[BA_T]], %[[BA_B]]
// CHECK-NEXT: %[[BA_ADDR:.*]] = llvm.add %[[BA_O]], %[[BA_BY]]
// CHECK-NEXT: %[[BW_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[BW_BASE:.*]] = llvm.ptrtoint %[[BW_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[BW_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[BW_OFFT:.*]] = llvm.trunc %[[BW_OFF]] : i64 to i32
// CHECK-NEXT: %[[BW_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[BW_OM:.*]] = llvm.mul %[[BW_OFFT]], %[[BW_EB]]
// CHECK-NEXT: %[[BW_O:.*]] = llvm.add %[[BW_BASE]], %[[BW_OM]]
// CHECK-NEXT: %[[BW_RS:.*]] = llvm.mlir.constant(64 : i32)
// CHECK-NEXT: %[[BW_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[BW_RM:.*]] = llvm.mul %[[BW_R]], %[[BW_RS]]
// CHECK-NEXT: %[[BW_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[BW_T:.*]] = llvm.add %[[BW_RM]], %[[BW_C]]
// CHECK-NEXT: %[[BW_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[BW_BY:.*]] = llvm.mul %[[BW_T]], %[[BW_B]]
// CHECK-NEXT: %[[BW_ADDR:.*]] = llvm.add %[[BW_O]], %[[BW_BY]]
// CHECK-NEXT: %[[BN:.*]] = llvm.mlir.constant(1 : i32)
// CHECK-NEXT: llvm.call @hmx_mma_f16(%[[BA_ADDR]], %[[BW_ADDR]], %[[BN]]) : (i32, i32, i32) -> ()
func.func @mma_shallow_1_kt64(%bias: memref<256xi8, 1>,
                              %act: memref<4x64x16x32x2xf16, 1>,
                              %wt: memref<3x64x16x32x2xf16, 1>,
                              %ar: memref<4x2x16x32x2xf16, 1>,
                              %m: index, %n: index, %k: index) {
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 1 : i32}
      : memref<4x64x16x32x2xf16, 1>, memref<3x64x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}

// CHECK-LABEL: func.func @mma_deep_2_kt2
// CHECK: llvm.call @hmx_acc_clear_f16() : () -> ()
// CHECK-NEXT: %[[CA_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[CA_BASE:.*]] = llvm.ptrtoint %[[CA_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[CA_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[CA_OFFT:.*]] = llvm.trunc %[[CA_OFF]] : i64 to i32
// CHECK-NEXT: %[[CA_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[CA_OM:.*]] = llvm.mul %[[CA_OFFT]], %[[CA_EB]]
// CHECK-NEXT: %[[CA_O:.*]] = llvm.add %[[CA_BASE]], %[[CA_OM]]
// CHECK-NEXT: %[[CA_RS:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[CA_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[CA_RM:.*]] = llvm.mul %[[CA_R]], %[[CA_RS]]
// CHECK-NEXT: %[[CA_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[CA_T:.*]] = llvm.add %[[CA_RM]], %[[CA_C]]
// CHECK-NEXT: %[[CA_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[CA_BY:.*]] = llvm.mul %[[CA_T]], %[[CA_B]]
// CHECK-NEXT: %[[CA_ADDR:.*]] = llvm.add %[[CA_O]], %[[CA_BY]]
// CHECK-NEXT: %[[CW_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[CW_BASE:.*]] = llvm.ptrtoint %[[CW_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[CW_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[CW_OFFT:.*]] = llvm.trunc %[[CW_OFF]] : i64 to i32
// CHECK-NEXT: %[[CW_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[CW_OM:.*]] = llvm.mul %[[CW_OFFT]], %[[CW_EB]]
// CHECK-NEXT: %[[CW_O:.*]] = llvm.add %[[CW_BASE]], %[[CW_OM]]
// CHECK-NEXT: %[[CW_RS:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[CW_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[CW_RM:.*]] = llvm.mul %[[CW_R]], %[[CW_RS]]
// CHECK-NEXT: %[[CW_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[CW_T:.*]] = llvm.add %[[CW_RM]], %[[CW_C]]
// CHECK-NEXT: %[[CW_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[CW_BY:.*]] = llvm.mul %[[CW_T]], %[[CW_B]]
// CHECK-NEXT: %[[CW_ADDR:.*]] = llvm.add %[[CW_O]], %[[CW_BY]]
// CHECK-NEXT: %[[CN:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: llvm.call @hmx_mma_f16(%[[CA_ADDR]], %[[CW_ADDR]], %[[CN]]) : (i32, i32, i32) -> ()
func.func @mma_deep_2_kt2(%bias: memref<256xi8, 1>,
                          %act: memref<4x2x16x32x2xf16, 1>,
                          %wt: memref<3x2x16x32x2xf16, 1>,
                          %ar: memref<4x2x16x32x2xf16, 1>,
                          %m: index, %n: index, %k: index) {
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 2 : i32}
      : memref<4x2x16x32x2xf16, 1>, memref<3x2x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}

// CHECK-LABEL: func.func @mma_deep_31_kt31
// CHECK: llvm.call @hmx_acc_clear_f16() : () -> ()
// CHECK-NEXT: %[[DA_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[DA_BASE:.*]] = llvm.ptrtoint %[[DA_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[DA_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[DA_OFFT:.*]] = llvm.trunc %[[DA_OFF]] : i64 to i32
// CHECK-NEXT: %[[DA_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[DA_OM:.*]] = llvm.mul %[[DA_OFFT]], %[[DA_EB]]
// CHECK-NEXT: %[[DA_O:.*]] = llvm.add %[[DA_BASE]], %[[DA_OM]]
// CHECK-NEXT: %[[DA_RS:.*]] = llvm.mlir.constant(31 : i32)
// CHECK-NEXT: %[[DA_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[DA_RM:.*]] = llvm.mul %[[DA_R]], %[[DA_RS]]
// CHECK-NEXT: %[[DA_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[DA_T:.*]] = llvm.add %[[DA_RM]], %[[DA_C]]
// CHECK-NEXT: %[[DA_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[DA_BY:.*]] = llvm.mul %[[DA_T]], %[[DA_B]]
// CHECK-NEXT: %[[DA_ADDR:.*]] = llvm.add %[[DA_O]], %[[DA_BY]]
// CHECK-NEXT: %[[DW_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[DW_BASE:.*]] = llvm.ptrtoint %[[DW_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[DW_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[DW_OFFT:.*]] = llvm.trunc %[[DW_OFF]] : i64 to i32
// CHECK-NEXT: %[[DW_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[DW_OM:.*]] = llvm.mul %[[DW_OFFT]], %[[DW_EB]]
// CHECK-NEXT: %[[DW_O:.*]] = llvm.add %[[DW_BASE]], %[[DW_OM]]
// CHECK-NEXT: %[[DW_RS:.*]] = llvm.mlir.constant(31 : i32)
// CHECK-NEXT: %[[DW_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[DW_RM:.*]] = llvm.mul %[[DW_R]], %[[DW_RS]]
// CHECK-NEXT: %[[DW_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[DW_T:.*]] = llvm.add %[[DW_RM]], %[[DW_C]]
// CHECK-NEXT: %[[DW_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[DW_BY:.*]] = llvm.mul %[[DW_T]], %[[DW_B]]
// CHECK-NEXT: %[[DW_ADDR:.*]] = llvm.add %[[DW_O]], %[[DW_BY]]
// CHECK-NEXT: %[[DN:.*]] = llvm.mlir.constant(31 : i32)
// CHECK-NEXT: llvm.call @hmx_mma_f16(%[[DA_ADDR]], %[[DW_ADDR]], %[[DN]]) : (i32, i32, i32) -> ()
func.func @mma_deep_31_kt31(%bias: memref<256xi8, 1>,
                            %act: memref<4x31x16x32x2xf16, 1>,
                            %wt: memref<3x31x16x32x2xf16, 1>,
                            %ar: memref<4x2x16x32x2xf16, 1>,
                            %m: index, %n: index, %k: index) {
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 31 : i32}
      : memref<4x31x16x32x2xf16, 1>, memref<3x31x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}

// CHECK-LABEL: func.func @mma_deep_32_kt33
// CHECK: llvm.call @hmx_acc_clear_f16() : () -> ()
// CHECK-NEXT: %[[EA_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[EA_BASE:.*]] = llvm.ptrtoint %[[EA_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[EA_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[EA_OFFT:.*]] = llvm.trunc %[[EA_OFF]] : i64 to i32
// CHECK-NEXT: %[[EA_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[EA_OM:.*]] = llvm.mul %[[EA_OFFT]], %[[EA_EB]]
// CHECK-NEXT: %[[EA_O:.*]] = llvm.add %[[EA_BASE]], %[[EA_OM]]
// CHECK-NEXT: %[[EA_RS:.*]] = llvm.mlir.constant(33 : i32)
// CHECK-NEXT: %[[EA_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[EA_RM:.*]] = llvm.mul %[[EA_R]], %[[EA_RS]]
// CHECK-NEXT: %[[EA_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[EA_T:.*]] = llvm.add %[[EA_RM]], %[[EA_C]]
// CHECK-NEXT: %[[EA_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[EA_BY:.*]] = llvm.mul %[[EA_T]], %[[EA_B]]
// CHECK-NEXT: %[[EA_ADDR:.*]] = llvm.add %[[EA_O]], %[[EA_BY]]
// CHECK-NEXT: %[[EW_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[EW_BASE:.*]] = llvm.ptrtoint %[[EW_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[EW_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[EW_OFFT:.*]] = llvm.trunc %[[EW_OFF]] : i64 to i32
// CHECK-NEXT: %[[EW_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[EW_OM:.*]] = llvm.mul %[[EW_OFFT]], %[[EW_EB]]
// CHECK-NEXT: %[[EW_O:.*]] = llvm.add %[[EW_BASE]], %[[EW_OM]]
// CHECK-NEXT: %[[EW_RS:.*]] = llvm.mlir.constant(33 : i32)
// CHECK-NEXT: %[[EW_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[EW_RM:.*]] = llvm.mul %[[EW_R]], %[[EW_RS]]
// CHECK-NEXT: %[[EW_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[EW_T:.*]] = llvm.add %[[EW_RM]], %[[EW_C]]
// CHECK-NEXT: %[[EW_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[EW_BY:.*]] = llvm.mul %[[EW_T]], %[[EW_B]]
// CHECK-NEXT: %[[EW_ADDR:.*]] = llvm.add %[[EW_O]], %[[EW_BY]]
// CHECK-NEXT: %[[EN:.*]] = llvm.mlir.constant(32 : i32)
// CHECK-NEXT: llvm.call @hmx_mma_f16(%[[EA_ADDR]], %[[EW_ADDR]], %[[EN]]) : (i32, i32, i32) -> ()
func.func @mma_deep_32_kt33(%bias: memref<256xi8, 1>,
                            %act: memref<4x33x16x32x2xf16, 1>,
                            %wt: memref<3x33x16x32x2xf16, 1>,
                            %ar: memref<4x2x16x32x2xf16, 1>,
                            %m: index, %n: index, %k: index) {
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 32 : i32}
      : memref<4x33x16x32x2xf16, 1>, memref<3x33x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}

// CHECK-LABEL: func.func @mma_deep_32_kt65
// CHECK: llvm.call @hmx_acc_clear_f16() : () -> ()
// CHECK-NEXT: %[[FA_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[FA_BASE:.*]] = llvm.ptrtoint %[[FA_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[FA_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[FA_OFFT:.*]] = llvm.trunc %[[FA_OFF]] : i64 to i32
// CHECK-NEXT: %[[FA_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[FA_OM:.*]] = llvm.mul %[[FA_OFFT]], %[[FA_EB]]
// CHECK-NEXT: %[[FA_O:.*]] = llvm.add %[[FA_BASE]], %[[FA_OM]]
// CHECK-NEXT: %[[FA_RS:.*]] = llvm.mlir.constant(65 : i32)
// CHECK-NEXT: %[[FA_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[FA_RM:.*]] = llvm.mul %[[FA_R]], %[[FA_RS]]
// CHECK-NEXT: %[[FA_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[FA_T:.*]] = llvm.add %[[FA_RM]], %[[FA_C]]
// CHECK-NEXT: %[[FA_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[FA_BY:.*]] = llvm.mul %[[FA_T]], %[[FA_B]]
// CHECK-NEXT: %[[FA_ADDR:.*]] = llvm.add %[[FA_O]], %[[FA_BY]]
// CHECK-NEXT: %[[FW_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[FW_BASE:.*]] = llvm.ptrtoint %[[FW_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[FW_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[FW_OFFT:.*]] = llvm.trunc %[[FW_OFF]] : i64 to i32
// CHECK-NEXT: %[[FW_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[FW_OM:.*]] = llvm.mul %[[FW_OFFT]], %[[FW_EB]]
// CHECK-NEXT: %[[FW_O:.*]] = llvm.add %[[FW_BASE]], %[[FW_OM]]
// CHECK-NEXT: %[[FW_RS:.*]] = llvm.mlir.constant(65 : i32)
// CHECK-NEXT: %[[FW_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[FW_RM:.*]] = llvm.mul %[[FW_R]], %[[FW_RS]]
// CHECK-NEXT: %[[FW_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[FW_T:.*]] = llvm.add %[[FW_RM]], %[[FW_C]]
// CHECK-NEXT: %[[FW_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[FW_BY:.*]] = llvm.mul %[[FW_T]], %[[FW_B]]
// CHECK-NEXT: %[[FW_ADDR:.*]] = llvm.add %[[FW_O]], %[[FW_BY]]
// CHECK-NEXT: %[[FN:.*]] = llvm.mlir.constant(32 : i32)
// CHECK-NEXT: llvm.call @hmx_mma_f16(%[[FA_ADDR]], %[[FW_ADDR]], %[[FN]]) : (i32, i32, i32) -> ()
func.func @mma_deep_32_kt65(%bias: memref<256xi8, 1>,
                            %act: memref<4x65x16x32x2xf16, 1>,
                            %wt: memref<3x65x16x32x2xf16, 1>,
                            %ar: memref<4x2x16x32x2xf16, 1>,
                            %m: index, %n: index, %k: index) {
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wt, %m, %n, %k {n_croutons = 32 : i32}
      : memref<4x65x16x32x2xf16, 1>, memref<3x65x16x32x2xf16, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}

// CHECK-LABEL: func.func @mma_deep_strided_wt
// CHECK: llvm.call @hmx_acc_clear_f16() : () -> ()
// CHECK-NEXT: %[[GA_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[GA_BASE:.*]] = llvm.ptrtoint %[[GA_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[GA_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[GA_OFFT:.*]] = llvm.trunc %[[GA_OFF]] : i64 to i32
// CHECK-NEXT: %[[GA_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[GA_OM:.*]] = llvm.mul %[[GA_OFFT]], %[[GA_EB]]
// CHECK-NEXT: %[[GA_O:.*]] = llvm.add %[[GA_BASE]], %[[GA_OM]]
// CHECK-NEXT: %[[GA_RS:.*]] = llvm.mlir.constant(32 : i32)
// CHECK-NEXT: %[[GA_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[GA_RM:.*]] = llvm.mul %[[GA_R]], %[[GA_RS]]
// CHECK-NEXT: %[[GA_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[GA_T:.*]] = llvm.add %[[GA_RM]], %[[GA_C]]
// CHECK-NEXT: %[[GA_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[GA_BY:.*]] = llvm.mul %[[GA_T]], %[[GA_B]]
// CHECK-NEXT: %[[GA_ADDR:.*]] = llvm.add %[[GA_O]], %[[GA_BY]]
// CHECK-NEXT: %[[GW_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK-NEXT: %[[GW_BASE:.*]] = llvm.ptrtoint %[[GW_PTR]] : !llvm.ptr<1> to i32
// CHECK-NEXT: %[[GW_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK-NEXT: %[[GW_OFFT:.*]] = llvm.trunc %[[GW_OFF]] : i64 to i32
// CHECK-NEXT: %[[GW_EB:.*]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[GW_OM:.*]] = llvm.mul %[[GW_OFFT]], %[[GW_EB]]
// CHECK-NEXT: %[[GW_O:.*]] = llvm.add %[[GW_BASE]], %[[GW_OM]]
// CHECK-NEXT: %[[GW_RS:.*]] = llvm.mlir.constant(32 : i32)
// CHECK-NEXT: %[[GW_R:.*]] = llvm.trunc
// CHECK-NEXT: %[[GW_RM:.*]] = llvm.mul %[[GW_R]], %[[GW_RS]]
// CHECK-NEXT: %[[GW_C:.*]] = llvm.trunc
// CHECK-NEXT: %[[GW_T:.*]] = llvm.add %[[GW_RM]], %[[GW_C]]
// CHECK-NEXT: %[[GW_B:.*]] = llvm.mlir.constant(2048 : i32)
// CHECK-NEXT: %[[GW_BY:.*]] = llvm.mul %[[GW_T]], %[[GW_B]]
// CHECK-NEXT: %[[GW_ADDR:.*]] = llvm.add %[[GW_O]], %[[GW_BY]]
// CHECK-NEXT: %[[GN:.*]] = llvm.mlir.constant(32 : i32)
// CHECK-NEXT: llvm.call @hmx_mma_f16(%[[GA_ADDR]], %[[GW_ADDR]], %[[GN]]) : (i32, i32, i32) -> ()
func.func @mma_deep_strided_wt(%bias: memref<256xi8, 1>,
                               %act: memref<4x32x16x32x2xf16, 1>,
                               %whole: memref<8x32x16x32x2xf16, 1>,
                               %ar: memref<4x2x16x32x2xf16, 1>,
                               %m: index, %n: index, %k: index) {
  %wsub = memref.subview %whole[2, 0, 0, 0, 0] [2, 32, 16, 32, 2] [1, 1, 1, 1, 1]
      : memref<8x32x16x32x2xf16, 1> to memref<2x32x16x32x2xf16, strided<[32768, 1024, 64, 2, 1], offset: 65536>, 1>
  hmx.bias_init %bias : memref<256xi8, 1>
  hmx.acc_clear
  hmx.mma %act, %wsub, %m, %n, %k {n_croutons = 32 : i32}
      : memref<4x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, strided<[32768, 1024, 64, 2, 1], offset: 65536>, 1>
  hmx.acc_read %bias, %ar, %m, %n {bias_set = 2 : i32}
      : memref<256xi8, 1>, memref<4x2x16x32x2xf16, 1>
  return
}
