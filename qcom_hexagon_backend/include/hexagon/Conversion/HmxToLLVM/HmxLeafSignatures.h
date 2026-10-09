//===- HmxLeafSignatures.h - ABI signature of every HMX runtime leaf -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// One row per leaf libhmxapi.a exports: the symbol the lowering emits and the
// type of every parameter, in declaration order.
//
// Why this exists. Three files have to agree about a leaf's ABI, and the
// compiler checks none of them against each other:
//
//   bin/runtime/hmx/include/HMXAPI.h  what the runtime declares and
//                                     libhmxapi.a implements (the authority)
//   HmxExternalFnNames.cpp            the symbol the lowering emits
//   this table                        the argument types the lowering passes
//
// `test_hmx_leaf_names_contract.py` binds the first two *by name*. Nothing
// bound the third: every lowering site in HmxToLLVMPass.cpp spelled its own
// `SmallVector<Type>(9, i32Ty)`, so an arity or type change in HMXAPI.h was a
// silent ABI mismatch -- it compiles, it links, it passes a name-only
// contract, and on the device it reads arguments out of the wrong registers.
// `test_hmx_leaf_signature_agreement.py` reads all three and fails, by leaf
// name, when any two disagree.
//
// Rules for editing kHmxLeafSignatures:
//
//   * It is a transcription of HMXAPI.h, in that header's declaration order,
//     so the two files can be diffed by eye -- which is how a human reviews a
//     signature change before the gate ever runs, and the gate enforces it.
//   * The C spellings are the header's, not MLIR types: `unsigned` is what
//     libhmxapi.a takes, and `argTys` is the single translation to i32. A
//     spelling it does not know fails the lookup instead of defaulting -- a
//     silent i32 would put the mismatch back on the device, which is the
//     failure this table exists to remove.
//   * The list is deliberately *not* merged with HmxExternalFnNames.cpp: that
//     file maps hmx ops to names and reaches into the DMA table for the two
//     staging leaves, so folding the two together would make the leaf ABI a
//     third opinion about the DMA ABI. The gate compares them instead.
//
// Scope: the leaves HMXAPI.h declares. `hmx.stage` / `hmx.await` lower to the
// DMA runtime entries (`hexagon_runtime_dma_*`), whose signature DMAToLLVMPass
// declares for the same symbols; those are deliberately NOT rows here, so this
// table never becomes a second opinion about the DMA ABI.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HMXTOLLVM_HMXLEAFSIGNATURES_H
#define HEXAGON_CONVERSION_HMXTOLLVM_HMXLEAFSIGNATURES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hmx {

/// One runtime leaf, as HMXAPI.h declares it.
struct HmxLeafSignature {
  /// The emitted symbol, byte for byte the declaration's name.
  const char *name;
  /// The parameter types in declaration order, spelled exactly as HMXAPI.h
  /// spells them, comma separated with no spaces. Empty means a parameterless
  /// leaf (`void hmx_acc_clear_f16(void)`).
  const char *paramTypes;

  /// This row's parameters as MLIR types, in declaration order. Fails -- with
  /// no diagnostic, so the caller can name the leaf -- when a spelling has no
  /// MLIR image; guessing i32 there would put the mismatch back on the device.
  FailureOr<SmallVector<Type>> argTys(MLIRContext *ctx) const;
};

/// Every leaf HMXAPI.h declares, in that header's declaration order.
/// Header-only because the one consumer is HmxToLLVMPass.cpp and the other
/// consumer is a Python gate that reads the text; no target owns a .cpp here.
inline constexpr HmxLeafSignature kHmxLeafSignatures[] = {
    {"hmx_bias_init_unit_f16", "unsigned"},
    {"hmx_bias_load_f16", "unsigned,unsigned"},
    {"hmx_acc_clear_f16", ""},
    {"hmx_acc_store_f16", "unsigned,unsigned"},
    {"hmx_mma_f16", "unsigned,unsigned,unsigned"},
    {"hmx_pack_act_f16", "unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_f16", "unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_act_f16_bulk", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_f16_bulk", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_act_f32", "unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_f32", "unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_act_f32_bulk", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_f32_bulk", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_f16_T", "unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_f16_T_bulk", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_f32_T", "unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_f32_T_bulk", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_act_tail_f16", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_tail_f16", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_act_tail_f32", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_tail_f32", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_tail_f16_T", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_pack_weight_tail_f32_T", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_unpack_acc_f16", "unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_unpack_acc_f16_bulk", "unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned"},
    {"hmx_unpack_acc_f32", "unsigned,unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_unpack_acc_f32_bulk", "unsigned,unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_unpack_acc_tail_f16", "unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned"},
    {"hmx_unpack_acc_tail_f32",
     "unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,"
     "unsigned,unsigned,unsigned,unsigned,unsigned,unsigned"},
};

/// The row for `name`, or null when it is not an HMX leaf. For a lowering that
/// null is a bug, not a lookup miss: `getVoidLeaf` in HmxToLLVMPass.cpp reports
/// it instead of falling back to a hand-written type vector.
inline const HmxLeafSignature *lookupHmxLeafSignature(llvm::StringRef name) {
  for (const HmxLeafSignature &leaf : kHmxLeafSignatures)
    if (name == leaf.name)
      return &leaf;
  return nullptr;
}

inline FailureOr<SmallVector<Type>>
HmxLeafSignature::argTys(MLIRContext *ctx) const {
  SmallVector<Type> argTys;
  for (llvm::StringRef spelling :
       llvm::split(llvm::StringRef(paramTypes), ',')) {
    if (spelling.empty())
      continue; // the parameterless leaf: `""`, not a missing entry
    if (spelling == "unsigned") {
      // HMXAPI.h's contract is "no pointer-typed or _Float16 * arguments
      // anywhere": every parameter is an address or a word, both 32-bit here.
      argTys.push_back(IntegerType::get(ctx, 32));
      continue;
    }
    return failure();
  }
  return argTys;
}

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_CONVERSION_HMXTOLLVM_HMXLEAFSIGNATURES_H
