//===- HmxExternalFnNames.cpp - Runtime leaf names for hmx ops ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/HmxToLLVM/HmxExternalFnNames.h"

#include "hexagon/Conversion/DMAToLLVM/DMAExternalFnNames.h"

using namespace mlir;

// Every name here must match a declaration in
// qcom_hexagon_backend/bin/runtime/hmx/include/HMXAPI.h, which is what
// libhmxapi.a implements.
std::string hmx::getBiasInitUnitF16FnName() { return "hmx_bias_init_unit_f16"; }
std::string hmx::getBiasLoadF16FnName() { return "hmx_bias_load_f16"; }
std::string hmx::getAccClearF16FnName() { return "hmx_acc_clear_f16"; }
std::string hmx::getAccStoreF16FnName() { return "hmx_acc_store_f16"; }
std::string hmx::getMmaF16FnName() { return "hmx_mma_f16"; }
std::string hmx::getPackActF16FnName() { return "hmx_pack_act_f16"; }
std::string hmx::getPackWeightF16FnName() { return "hmx_pack_weight_f16"; }
std::string hmx::getPackActF32FnName() { return "hmx_pack_act_f32"; }
std::string hmx::getPackWeightF32FnName() { return "hmx_pack_weight_f32"; }
std::string hmx::getUnpackAccF16FnName() { return "hmx_unpack_acc_f16"; }
std::string hmx::getUnpackAccF32FnName() { return "hmx_unpack_acc_f32"; }
std::string hmx::getPackActF16BulkFnName() { return "hmx_pack_act_f16_bulk"; }
std::string hmx::getPackWeightF16BulkFnName() {
  return "hmx_pack_weight_f16_bulk";
}
std::string hmx::getPackActF32BulkFnName() { return "hmx_pack_act_f32_bulk"; }
std::string hmx::getPackWeightF32BulkFnName() {
  return "hmx_pack_weight_f32_bulk";
}
std::string hmx::getUnpackAccF16BulkFnName() {
  return "hmx_unpack_acc_f16_bulk";
}
std::string hmx::getUnpackAccF32BulkFnName() {
  return "hmx_unpack_acc_f32_bulk";
}
std::string hmx::getPackActTailF16FnName() { return "hmx_pack_act_tail_f16"; }
std::string hmx::getPackWeightTailF16FnName() {
  return "hmx_pack_weight_tail_f16";
}
std::string hmx::getPackActTailF32FnName() { return "hmx_pack_act_tail_f32"; }
std::string hmx::getPackWeightTailF32FnName() {
  return "hmx_pack_weight_tail_f32";
}
std::string hmx::getUnpackAccTailF16FnName() {
  return "hmx_unpack_acc_tail_f16";
}
std::string hmx::getUnpackAccTailF32FnName() {
  return "hmx_unpack_acc_tail_f32";
}
std::string hmx::getPackWeightF16TFnName() { return "hmx_pack_weight_f16_T"; }
std::string hmx::getPackWeightF32TFnName() { return "hmx_pack_weight_f32_T"; }
std::string hmx::getPackWeightF16TBulkFnName() {
  return "hmx_pack_weight_f16_T_bulk";
}
std::string hmx::getPackWeightF32TBulkFnName() {
  return "hmx_pack_weight_f32_T_bulk";
}
std::string hmx::getPackWeightTailF16TFnName() {
  return "hmx_pack_weight_tail_f16_T";
}
std::string hmx::getPackWeightTailF32TFnName() {
  return "hmx_pack_weight_tail_f32_T";
}

// The staging ops are DMA runtime calls, so the names are the DMA ones; see the
// header for why they are re-exported here.
std::string hmx::getStageDmaStartFnName() {
  return mlir::hexagon::getDMAStartFnName();
}
std::string hmx::getAwaitDmaWaitFnName() {
  return mlir::hexagon::getDMAWaitFnName();
}
