//===- Common.h   - some useful common types and functions ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_COMMON_H
#define HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_COMMON_H
#include "hexagon/Common/Common.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace mlir {
class TensorType;
class ModuleOp;
namespace linalg {
class LinalgOp;
} // namespace linalg

namespace hexagon {

/// Mirror a census remark to stderr when HEXMLIR_DIAG_REMARKS is set.
///
/// WHY THIS IS A PRINT AND NOT A DIAGNOSTIC HANDLER. Passes in this backend
/// report "how many candidates did I see, how many did I rewrite" with
/// `emitRemark`, because a census is what makes "the gate was too narrow" and
/// "the kernel got faster" distinguishable. That works under
/// `linalg-hexagon-opt`. It does NOT work in a Triton compile: this MLIR's
/// `DiagnosticEngine` drops a diagnostic whose severity is below its print
/// threshold *before* consulting any handler, and the threshold defaults to
/// Error -- so a remark never reaches a registered handler, and the pinned
/// `DiagnosticEngine` exposes no setter to lower it. A Triton compile therefore
/// produced no remark at all, and "my pass never fired" was indistinguishable
/// from "my pass fired and the rewrite was erased" (see
/// docs/hmx/rgs-step2-machine-code-2026-09-30.md).
///
/// So the text is printed directly. The message is passed in already formatted,
/// which keeps it byte-identical to the remark the same pass emits -- two docs
/// quote that text, and re-wording it silently would invalidate both.
///
/// GATED because this is diagnostic visibility, not semantics: unconditional
/// printing would make every Triton compile in CI unusable. Off unless
/// HEXMLIR_DIAG_REMARKS is set to something other than "" / "0".
inline void printCensusRemarkToStderr(const std::string &message) {
  static const bool enabled = [] {
    const char *flag = std::getenv("HEXMLIR_DIAG_REMARKS");
    return flag && flag[0] != '\0' && std::string(flag) != "0";
  }();
  if (enabled)
    llvm::errs() << "[census] " << message << "\n";
}

inline constexpr unsigned nativeVectorWidthInBytes = 128;
inline constexpr unsigned maxElemSizeInByte = 8;

/// The size of the vtcm memory size in bytes.
inline constexpr unsigned vtcmSizeInBytes = 2 * 1024 * 1024; // 2MB

/// Based on op-operands determines the element type size.
/// If it can't determine returns nullopt.
std::optional<unsigned> computeSmallestOperandTypeSize(Operation *);

/// Returns element type of higher dimensional types.
Type getElementType(Type);

/// Returns element type size in bytes. nullopt if
/// type is not handled currently.
std::optional<unsigned> getElementSizeInBytes(Type);

/// Based on target hardware vector size and operand elemen-type
/// for this op, returns a suitable number of elements in vec.
std::optional<unsigned> computeDataTileSize(Operation *);

// Check whether the loop range is a multiple of tile size
bool isPerfectlyTileable(unsigned LoopRange, unsigned dataTileSize);

/// Checks if inner loop dimension and vector size match
/// for neat vectorization. This is quite a complex decision
/// and currently it decides it with equality.
bool perfectlyVectorizable(unsigned dataTileSize, unsigned innerLoopRange);

/// The shape contract of a row reduction the vector-row-reduce pass handles:
/// a static-shape 2-D f16/f32 view, contiguous (unit stride) along the last
/// dim, whose row byte width is a whole number of 128-byte HVX vectors.
/// Accepts both the memref form (post-bufferize, the pass's view) and the
/// tensor form (the vectorizer's view, pre-bufferize).
struct RowReduceShape {
  int64_t rows;      // dim 0
  int64_t cols;      // dim 1
  int64_t elemBytes; // f16 -> 2, f32 -> 4
};
std::optional<RowReduceShape> vectorRowReduceShapeOf(Type type);

/// One HVX vector register, in bytes (v79: 128). The single source for the
/// row-chunk width shared by the row-reduce pass and the vectorizer's skip
/// gate -- both previously kept their own copy.
inline constexpr int64_t kHvxVectorBytes = 128;

/// The fold a matching row-reduce body must be: one binary maxnumf/addf whose
/// one operand is the init block arg and whose other operand is the input
/// block arg, possibly through a chain of pure elementwise ops fused into the
/// body (rms_norm's x*x before the row sum), yielded directly.
struct RowReduceFold {
  /// The fused elementwise producer between the input block arg and the fold,
  /// in block order. Empty for the classic chain-less body. Members are
  /// validated as re-creatable on vectors by the matcher (see
  /// matchVectorRowReduce); the pass re-creates them per HVX chunk.
  SmallVector<Operation *, 4> chain;

  RowReduceShape shape;
  bool isMaxNum;
  arith::FastMathFlags fastmath;
  /// The element type of the fold (and of the accumulator/init). Equal to the
  /// input element type for the classic same-type body; wider when the chain
  /// upcasts in the body (an f16 row summed in f32 -- Triton's tl.sum on f16,
  /// e.g. rms_norm). The butterfly runs at this width: the chunk reads are
  /// lanes = kHvxVectorBytes / foldElemBytes input elements each.
  Type foldElemTy;
};

/// Full match of the "row reduce rewritten into a vector fold + hvx.vror
/// butterfly" pattern, form-agnostic (tensor or memref). One source of truth
/// for the vector-row-reduce pass and for the vectorizer's skip gate, so the
/// two can never disagree about which reduces are diverted to the butterfly.
std::optional<RowReduceFold> matchVectorRowReduce(linalg::LinalgOp op);

/// Hexagon target triple and datalayout
void setTargetTriple(ModuleOp);
void setDataLayout(ModuleOp);

// To check if there functions which return values.
bool doesFuncReturnValue(ModuleOp);

} // namespace hexagon
} // namespace mlir
#endif // HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_COMMON_H
