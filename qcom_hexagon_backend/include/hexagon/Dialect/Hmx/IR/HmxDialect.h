//===- HmxDialect.h - HMX Dialect  ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_H
#define HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "hexagon/Dialect/Hmx/IR/HmxCroutonLayout.h"
#include "hexagon/Dialect/Hmx/IR/HmxDType.h"
#include "llvm/ADT/StringRef.h"

//===----------------------------------------------------------------------===//
// HMX Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h.inc"

//===----------------------------------------------------------------------===//
// HMX Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/Hmx/IR/HmxAttrs.h.inc"

//===----------------------------------------------------------------------===//
// Layout contract
//===----------------------------------------------------------------------===//
//
// The HMX engine's data layout as one contract, not three conventions.
//
// A crouton array is a 2D grid of 32x32 fp16 croutons whose trailing dims are
// exactly {16, 32, 2}, living in VTCM. The three roles the engine uses it in --
// activation (AH), weight (WH) and read-out (AR) -- are the *same* permutation,
// verified bit-exactly on device: a matmul chained off another matmul reads the
// producer's AR array as its activation with no conversion, and the two agree on
// every element (see docs/history/hmx/hmx-generality.md and STATE-OF-PLAY
// section 4).
// So compatibility is "same logical shape and element type", and the role is a
// property of the *use*, not of the type.
//
// The layout is now carried **in the type**, as the `#hmx.crouton` tensor
// encoding (a `VerifiableTensorEncoding`, checked automatically by
// `RankedTensorType::verify`). That is what makes the layout propagatable: any
// producer and any consumer of a crouton array are compatible by construction,
// and the only question left is where a conversion has to be materialised.
//
// See docs/hmx/layout-representation-survey.md for the mechanism and the
// reasoning (tensor encoding rather than memref layout; no role in the type).
//
// The layout is named by the `#hmx.crouton` encoding, not inferred from a rank-5
// shape, so a bare shape is not a crouton.
namespace mlir {
namespace hmx {

/// The closed policy vocabulary carried by #hmx.tail_plan. Keep these names
/// in one place so the parser, target planner, and leaf lowering cannot drift.
inline constexpr llvm::StringLiteral kHmxTailKPolicy =
    "zero-pad-both-operands";
inline constexpr llvm::StringLiteral kHmxTailMNPolicy =
    "padded-edge-tile-bounded-store";

/// The implicit single hardware resource of the HMX path: the engine's
/// accumulator / bias-register state and its staging pipeline. There is one
/// such unit, so every op that mutates it conflicts with every other one
/// (`hmx.mma` cannot be reordered around `hmx.acc_clear` / `hmx.acc_read`).
///
/// Non-addressable: this is register state, not pointer-based memory, so a
/// value-based memory access is never told it aliases the engine. The
/// resource's parent stays the default one (the `SideEffects::Resource`
/// contract), so it is *not* disjoint from ordinary memory effects either --
/// effects that also touch real memory (e.g. `hmx.acc_read`, `hmx.stage`) keep
/// their plain `MemRead`/`MemWrite` on the default resource next to their
/// engine instance, while an engine-only op (`hmx.acc_clear`, whose write was
/// never a memory write) no longer claims to mutate all memory.
///
/// Named from the `.td` side by `Hmx_EngineResource` (HmxOps.td); no
/// registration is needed -- resources are CRTP singletons.
struct HmxEngineResource
    : public SideEffects::Resource::Base<HmxEngineResource> {
  llvm::StringRef getName() const final { return "HmxEngine"; }
  bool isAddressable() const final { return false; }
};

/// The crouton encoding on `type`, or a null attribute when `type` is not a
/// crouton-encoded ranked tensor.
CroutonLayoutAttr getCroutonEncoding(Type type);

/// The memref-side layout (`#hmx.crouton_memref_layout`) for the crouton
/// encoding on `type`, or a null attribute when `type` is not a
/// crouton-encoded ranked tensor. This is the encoding → layout mapping the
/// bufferization seam has to apply; see CroutonMemRefLayoutAttr.
CroutonMemRefLayoutAttr croutonMemRefLayoutOf(Type type);

/// True when `type` is a ranked tensor carrying the `#hmx.crouton` encoding.
bool hasCroutonEncoding(Type type);

/// True when `a` and `b` are interchangeable crouton layouts: same logical
/// shape and element type. The role (AH/WH/AR) is not part of the comparison --
/// the three are one permutation.
bool sameCroutonEncoding(Type a, Type b);

/// The physical type of the crouton array holding a row-major `[M, N]` matrix:
/// `tensor<M/32, N/32, 16, 32, 2 x f16, #hmx.crouton<logical=[M,N]>>`. `logical`
/// must be 32-aligned in both dims; its element type is only the *source's* --
/// the crouton is always the engine's fp16 image, a wider source being quantised
/// by the pack that materialises the array.
RankedTensorType croutonLayoutType(RankedTensorType logical);

/// The physical type of the *weight* crouton array for a row-major `[K, N]`
/// weight. The engine always walks the grid's second dim, so the weight grid is
/// `[Nt, Kt, 16, 32, 2]` with `logical = [N, K]`: the weight is stored as Wᵀ, and
/// this is the same `#hmx.crouton` encoding as any other crouton array, just
/// attached to the transposed logical shape. `weightRowMajorKN` must be
/// 32-aligned in both dims (enforced inside `croutonLayoutType`).
RankedTensorType weightCroutonType(RankedTensorType weightRowMajorKN);

/// The K extent (in 32-wide croutons) of a weight crouton array: the grid's
/// *second* dim. This is the single named source for "which weight grid dim is
/// K" -- do not spell out `getDimSize(0/1)` for a weight elsewhere.
int64_t weightKTiles(ShapedType weightCrouton);

/// The N extent (in 32-wide croutons) of a weight crouton array: the grid's
/// *first* dim.
int64_t weightNTiles(ShapedType weightCrouton);

} // namespace hmx
} // namespace mlir

//===----------------------------------------------------------------------===//
// Marker trait: dialect member, but NOT an engine op
//===----------------------------------------------------------------------===//
namespace mlir {
namespace OpTrait {

/// Marks an `hmx` op that lowers to the plain DMA runtime entries and issues no
/// HMX instruction -- today `hmx.stage` and `hmx.await`, and nothing else.
///
/// `HmxToLLVM`'s engine ensure/unlock decision reads this trait instead of a
/// list of op classes. The polarity is deliberate and was reviewed: every
/// *unmarked* dialect op counts as an engine op until proven otherwise, so a
/// future engine op that is marked nowhere still gets its (harmless,
/// self-released) pair, while a future DMA-only op must set this marker --
/// omitting it costs one lock round-trip, whereas excluding an engine op would
/// park the next thread in `HAP_compute_res_hmx_lock` forever. The contract is
/// pinned by test/Conversion/HmxToLLVM/hmx-to-llvm.mlir @stage_await; do not
/// replace it with callee-name matching (silent wrong decision on a rename or
/// a wrapper) or with an engine whitelist (omission hangs the device).
template <typename ConcreteType>
struct HmxDmaOnly : public TraitBase<ConcreteType, HmxDmaOnly> {};
  //===----------------------------------------------------------------------===//
  // Marker trait: HMX data layout, implemented by the vector unit
  //===----------------------------------------------------------------------===//

  /// Marks an `hmx` op that moves data in and out of HMX layout but issues no
  /// HMX instruction. Today the four pack/unpack ops, and nothing else.
  ///
  /// `pack_act`/`pack_weight`/`unpack_acc`/`unpack_acc_f32` are HVX code: the HMX
  /// API translation unit has zero engine intrinsics, while every `Q6_*` symbol
  /// in the layout unit is a HVX vector intrinsic. So the engine ensure/unlock
  /// pair they used to carry bought nothing -- one lock round-trip per kernel,
  /// and a NON_SHARED unlock that also cleared the accumulator
  /// (bin/runtime/src/HexagonCAPI.cpp:212-213).
  ///
  /// Polarity, re-audited 2026-10-02 in both directions:
  ///
  ///   mis-marked as layout  -> no lock, engine instruction issued anyway ->
  ///                             device abort. Loud, and it names the function.
  ///   mis-marked as engine  -> one extra ensure/unlock, self-released, so the
  ///                             function still behaves exactly as it does today.
  ///
  /// So "unmarked means engine" still holds, and the new error direction is the
  /// easier one to diagnose.
  ///
  /// SCOPE: this only answers "does this function touch the engine". It is NOT
  /// the trait a thread-role predicate should read -- that one needs the
  /// opposite default (unmarked means HVX), because a layout op left unmarked
  /// there would be placed on a thread that holds no VTCM and hang. One bit
  /// cannot serve both, which is why this one is a negative exemption and that
  /// other one has to be positive. See roadmap/ROADMAP1001.md section 5.4.
  ///
  /// Pinned by test/Conversion/HmxToLLVM/hmx-tail-leaves.mlir (@tail_pack_f16 and
  /// its five siblings must carry no ensure pair) and by hmx-to-llvm.mlir
  /// @bridge, which mixes all four with no engine op.

  template <typename ConcreteType>
struct HmxLayoutHvx : public TraitBase<ConcreteType, HmxLayoutHvx> {};

  //===----------------------------------------------------------------------===//
  // Marker trait: must run on the HMX thread
  //===----------------------------------------------------------------------===//

  /// Marks an `hmx` op that issues an engine instruction and therefore may only
  /// run on a thread that holds the HMX resource. Today `matmul`,
  /// `alloc_crouton`, `bias_init`, `acc_clear`, `mma` and `acc_read`.
  ///
  /// This exists because `HmxLayoutHvx` cannot serve both predicates at once
  /// (roadmap section 5.4). Its polarity is "unmarked means engine", which is
  /// what the ensure/unlock decision wants: forgetting a marker there costs one
  /// harmless lock round-trip. A thread-role predicate reading the same trait
  /// would send an unmarked LAYOUT op to the HMX thread, which holds no VTCM and
  /// hangs with no diagnostic -- the deadlock
  /// HexagonTargetTransformInfo.cpp:453-458 describes. So this predicate is
  /// positive instead, and the failure direction is the survivable one:
  ///
  ///   engine op left unmarked   -> runs on the HVX thread -> it acquires the
  ///                                HMX lock, so there is contention and a
  ///                                possible loss of parallelism, but no deadlock
  ///   layout op wrongly marked  -> also the survivable direction above, since
  ///                                marking is per-op and explicit
  ///
  /// `matmul` and `alloc_crouton` are included deliberately: they exist at
  /// hmx-partition's decision point and are gone by HmxToLLVM's, because
  /// HmxToLLVMPass.cpp:1999 adds the dialect as illegal. A thread-role pass that
  /// runs after hmx-partition therefore reads `mma` (and `bias_init`,
  /// `acc_clear`, `acc_read`) and never sees those two; that is fine, because
  /// `mma` is the op that actually issues the matrix instruction.
  ///
  /// Read by ThreadRolePartition (thread-role-partition) as its engine-thread
  /// predicate. The predicate there wants the opposite default from
  /// HmxToLLVMPass's engine-leaf scan -- unmarked means vector, not engine --
  /// which is why this positive trait exists next to `HmxDmaOnly` and
  /// `HmxLayoutHvx` instead of being derived from them.
  template <typename ConcreteType>
struct HmxEngineIns : public TraitBase<ConcreteType, HmxEngineIns> {};

} // namespace OpTrait
} // namespace mlir

//===----------------------------------------------------------------------===//
// HMX Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/Hmx/IR/HmxOps.h.inc"

#endif // HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_H
