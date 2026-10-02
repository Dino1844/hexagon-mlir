// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-to-llvm)' | FileCheck %s

// The engine-lock contract for HMX layout ops, and the check that protects it.
//
// `pack_act` / `pack_weight` / `unpack_acc` / `unpack_acc_f32` move data in and
// out of HMX layout but issue no engine instruction: HMXAPI.c has zero engine
// intrinsics and every `Q6_*` symbol in HMXLayout.c is a HVX vector intrinsic.
// Before `HmxLayoutHvx` they were classified as engine ops, so every kernel whose
// only HMX work was packing took a lock round-trip per launch for nothing -- and
// a NON_SHARED unlock that also cleared the accumulator
// (bin/runtime/src/HexagonCAPI.cpp:212-213).
//
// Two things are pinned here. A third was drafted and dropped: verifyHmxLeafCallers
// only walks LLVM::CallOp, but a call written as `llvm.call` inside a `func.func`
// is a func.call by the time the check runs, so the check cannot see it. That is a
// pre-existing hole, independent of this change -- see ROADMAP1001.md section 9.5.

//===----------------------------------------------------------------------===//
// 1. A layout-only function takes no lock.
//===----------------------------------------------------------------------===//

// CHECK-DAG: llvm.func @hmx_pack_act_f16
// CHECK-DAG: llvm.func @hmx_pack_weight_f16
// CHECK-DAG: llvm.func @hmx_unpack_acc_f16
// CHECK-DAG: llvm.func @hmx_unpack_acc_f32

// CHECK-LABEL: func.func @pack_only
// CHECK-NOT: llvm.call @hexagon_runtime_hmx_ensure_dsp
// CHECK-NOT: llvm.call @hexagon_runtime_hmx_unlock_dsp
// CHECK: llvm.call @hmx_pack_act_f16
func.func @pack_only(%src: memref<64x64xf16>, %row: index, %col: index) {
  %dst = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf16>)
      outs(%dst : memref<2x2x16x32x2xf16, 1>)
  return
}

//===----------------------------------------------------------------------===//
// 2. All four layout ops together, still no lock, and every leaf is declared.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @all_four_layout_ops
// CHECK-NOT: llvm.call @hexagon_runtime_hmx_ensure_dsp
// CHECK-NOT: llvm.call @hexagon_runtime_hmx_unlock_dsp
func.func @all_four_layout_ops(%src: memref<64x64xf16>, %wsrc: memref<64x32xf16>,
                               %row: index, %col: index) {
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %wt = memref.alloc() : memref<1x2x16x32x2xf16, 1>
  %dst16 = memref.alloc() : memref<64x32xf16>
  %dst32 = memref.alloc() : memref<64x32xf32>
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.pack_weight ins(%wsrc, %row, %col : memref<64x32xf16>)
      outs(%wt : memref<1x2x16x32x2xf16, 1>)
  hmx.unpack_acc ins(%act, %row, %col : memref<2x2x16x32x2xf16, 1>)
      outs(%dst16 : memref<64x32xf16>)
  hmx.unpack_acc_f32 ins(%act, %row, %col : memref<2x2x16x32x2xf16, 1>)
      outs(%dst32 : memref<64x32xf32>)
  return
}

//===----------------------------------------------------------------------===//
// 3. One engine op is enough, and it flips the pair on. @bias_init is not
//    layout, so this function keeps the lock -- the trait narrows the exemption,
//    it does not disable the mechanism.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @layout_plus_one_engine_op
// CHECK: llvm.call @hexagon_runtime_hmx_ensure_dsp
// CHECK: llvm.call @hmx_bias_init_unit_f16
// CHECK: llvm.call @hexagon_runtime_hmx_unlock_dsp
func.func @layout_plus_one_engine_op(%src: memref<64x64xf16>, %row: index, %col: index) {
  %act = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  %bias = memref.alloc() : memref<256xi8, 1>
  hmx.pack_act ins(%src, %row, %col : memref<64x64xf16>)
      outs(%act : memref<2x2x16x32x2xf16, 1>)
  hmx.bias_init %bias : memref<256xi8, 1>
  return
}
