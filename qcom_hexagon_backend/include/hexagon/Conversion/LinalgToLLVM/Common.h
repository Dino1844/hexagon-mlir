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
#include <optional>

namespace mlir {
class TensorType;
class ModuleOp;
namespace linalg {
class LinalgOp;
} // namespace linalg

namespace hexagon {

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

/// The fold a matching row-reduce body must be: one binary maxnumf/addf over
/// the two block args, yielded directly.
struct RowReduceFold {
  RowReduceShape shape;
  bool isMaxNum;
  arith::FastMathFlags fastmath;
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
