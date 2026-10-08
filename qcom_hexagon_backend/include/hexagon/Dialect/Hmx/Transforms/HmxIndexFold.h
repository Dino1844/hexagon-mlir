//===-- HmxIndexFold.h - fold the index arithmetic the emitters produce ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// WHY A SHARED HEADER
// -------------------
// The post-emission index arithmetic of the HMX passes is NOT bare constants:
// the SCF pipeliner rewrites every row expression of the source loop into a
// shifted chain (`addi iv, (muli 1, 0)` for "this stage's row") and its peeled
// epilogue into a folded constant chain (`addi 0, (muli 1, (maxsi 0, (subi
// (divsi ...), 1)))` for "the last row"). Every pass that has to recognise a
// tile row in that IR therefore has to fold the same chains, and two
// independent folders would be two opportunities to disagree about what a
// chain folds to -- a disagreement that surfaces as one pass matching a shape
// the other declined, i.e. exactly the silent-coverage bug class both passes
// exist to prevent.
//
// The two current consumers (and the shapes they fold):
//   * HmxVectorReadoutPass: the loop bound (`subi c32, (muli 1, 1)`) and the
//     peeled row (the maxsi/divsi chain above).
//   * ThreadRolePartition: the pipelined steady loop's upper bound and the
//     peeled epilogue's row, which name the same numbers for the same
//     reasons.
//
// Everything here is pure integer arithmetic on values the caller has already
// proven constant, so folding cannot change meaning -- it only recovers a
// number that was there all along. Overflow and division by zero decline
// rather than wrap.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXINDEXFOLD_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXINDEXFOLD_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir {
namespace hmx {

/// NOT-A-DECISION this is a safety bound, not a tuned limit: it exists so
/// folding cannot hang, not because 16 is the right depth for any shape. The
/// chains this walks are two or three ops deep in production -- the peeled
/// row alone is `addi c0, (muli c1, (maxsi c0, (subi (divsi (...), c1),
/// c1)))`, six ops -- so 16 is an order of magnitude of headroom over the
/// longest real chain. A chain deeper than the cap simply declines rather
/// than folding something wrong.
inline constexpr unsigned kHmxFoldDepth = 16;

/// Fold an `i1`. Needed because the peeled row in the verified post-partition
/// IR is gated on a comparison (see foldIndex below for the chain). The two
/// are mutually recursive -- foldBool reads the i1 that gates the peeled row,
/// and foldIndex reads the i1's operands -- so foldIndex is forward-declared
/// here.
inline std::optional<int64_t> foldIndex(Value value, unsigned depth = 0);

inline std::optional<bool> foldBool(Value value, unsigned depth = 0) {
  if (depth > kHmxFoldDepth)
    return std::nullopt;
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (def->hasTrait<OpTrait::ConstantLike>()) {
    if (auto attr = dyn_cast_or_null<IntegerAttr>(def->getAttr("value")))
      if (value.getType().isInteger(1))
        return attr.getValue().getZExtValue() != 0;
    return std::nullopt;
  }
  auto cmp = dyn_cast<arith::CmpIOp>(def);
  if (!cmp)
    return std::nullopt;
  // Both sides must be index values; the passes here never compare two wide
  // integers, and assuming they did would fold a comparison it has not
  // actually evaluated.
  std::optional<int64_t> lhs = foldIndex(cmp.getLhs(), depth + 1);
  std::optional<int64_t> rhs = foldIndex(cmp.getRhs(), depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;

  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return *lhs == *rhs;
  case arith::CmpIPredicate::ne:
    return *lhs != *rhs;
  case arith::CmpIPredicate::slt:
    return *lhs < *rhs;
  case arith::CmpIPredicate::sle:
    return *lhs <= *rhs;
  case arith::CmpIPredicate::sgt:
    return *lhs > *rhs;
  case arith::CmpIPredicate::sge:
    return *lhs >= *rhs;
  case arith::CmpIPredicate::ult:
    return static_cast<uint64_t>(*lhs) < static_cast<uint64_t>(*rhs);
  case arith::CmpIPredicate::ule:
    return static_cast<uint64_t>(*lhs) <= static_cast<uint64_t>(*rhs);
  case arith::CmpIPredicate::ugt:
    return static_cast<uint64_t>(*lhs) > static_cast<uint64_t>(*rhs);
  case arith::CmpIPredicate::uge:
    return static_cast<uint64_t>(*lhs) >= static_cast<uint64_t>(*rhs);
  }
  return std::nullopt;
}

/// Constant-fold an index through the `arith` integer ops the emitters and the
/// SCF pipeliner actually produce: constants, add/sub/mul, min/max, signed
/// truncating division and remainder, and a select whose condition folds.
///
/// `constantIndexValue` in HmxPartitionPass.cpp accepts only a bare
/// `arith.constant`, which is not enough for the pipeliner's output. Two
/// chains matter:
///
///   * the loop bound, which is `subi %c32, (muli %c1, %c1)`; and
///   * the peeled row, which in the verified post-partition IR is
///         `%19 = maxsi 0, (subi (divsi (...), 1), 1)`
///         `%20 = muli %c1, %19`
///         `%21 = addi %c0, %20`
///     so without max/div the folder declines every depth-2 kernel, which is
///     the only shape the pipelined passes exist for.
/// (The forward declaration above is the mutual-recursion half; this is the
/// definition both it and the consumers call. No default argument here: the
/// forward declaration already gave it one, and C++ forbids the repeat.)
inline std::optional<int64_t> foldIndex(Value value, unsigned depth) {
  if (depth > kHmxFoldDepth)
    return std::nullopt;
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (def->hasTrait<OpTrait::ConstantLike>()) {
    if (auto attr = dyn_cast_or_null<IntegerAttr>(def->getAttr("value")))
      if (value.getType().isIndex())
        return attr.getValue().getSExtValue();
    return std::nullopt;
  }

  auto operands = [&](Operation *op, std::optional<int64_t> &lhs,
                      std::optional<int64_t> &rhs) {
    lhs = foldIndex(op->getOperand(0), depth + 1);
    rhs = foldIndex(op->getOperand(1), depth + 1);
    return lhs && rhs;
  };
  int64_t result = 0;

  if (isa<arith::AddIOp>(def)) {
    std::optional<int64_t> lhs, rhs;
    if (!operands(def, lhs, rhs) || __builtin_add_overflow(*lhs, *rhs, &result))
      return std::nullopt;
    return result;
  }
  if (isa<arith::SubIOp>(def)) {
    std::optional<int64_t> lhs, rhs;
    if (!operands(def, lhs, rhs) || __builtin_sub_overflow(*lhs, *rhs, &result))
      return std::nullopt;
    return result;
  }
  if (isa<arith::MulIOp>(def)) {
    std::optional<int64_t> lhs, rhs;
    if (!operands(def, lhs, rhs) || __builtin_mul_overflow(*lhs, *rhs, &result))
      return std::nullopt;
    return result;
  }
  if (isa<arith::MaxSIOp, arith::MaxUIOp>(def)) {
    std::optional<int64_t> lhs, rhs;
    if (!operands(def, lhs, rhs))
      return std::nullopt;
    return std::max(*lhs, *rhs);
  }
  if (isa<arith::MinSIOp, arith::MinUIOp>(def)) {
    std::optional<int64_t> lhs, rhs;
    if (!operands(def, lhs, rhs))
      return std::nullopt;
    return std::min(*lhs, *rhs);
  }
  // Truncating division: these tile bounds are non-negative, so trunc and
  // floor agree and there is no remainder case to get wrong. A zero divisor
  // is a decline, not a wrap -- folding it to 0 would be a wrong row index.
  if (isa<arith::DivSIOp, arith::DivUIOp>(def)) {
    std::optional<int64_t> lhs, rhs;
    if (!operands(def, lhs, rhs) || *rhs == 0)
      return std::nullopt;
    return *lhs / *rhs;
  }
  if (isa<arith::RemSIOp, arith::RemUIOp>(def)) {
    std::optional<int64_t> lhs, rhs;
    if (!operands(def, lhs, rhs) || *rhs == 0)
      return std::nullopt;
    return *lhs % *rhs;
  }
  // `arith.select` on a foldable condition is the arm it picks. The condition
  // is an `i1`, so it goes through foldBool rather than foldIndex.
  if (auto select = dyn_cast<arith::SelectOp>(def)) {
    std::optional<bool> condition = foldBool(select.getCondition(), depth + 1);
    if (!condition)
      return std::nullopt;
    return *condition ? foldIndex(select.getTrueValue(), depth + 1)
                      : foldIndex(select.getFalseValue(), depth + 1);
  }
  return std::nullopt;
}

/// Prove `value == iv + offset` for a compile-time `offset`, walking `addi`
/// and `muli` over `arith` ops.
///
/// This is what turns the post-pipeliner `addi %iv, (muli 1, 0)` into "the
/// row IS the induction variable" instead of "the row is some expression the
/// caller would have to reason about". Only an offset of 0 is a row identity;
/// the general form is kept so that a near miss declines with a number
/// attached rather than as an unrecognised shape.
inline std::optional<int64_t> inductionOffset(Value value, Value iv,
                                              unsigned depth = 0) {
  if (value == iv)
    return 0;
  if (depth > kHmxFoldDepth)
    return std::nullopt;
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    if (std::optional<int64_t> base =
            inductionOffset(add.getLhs(), iv, depth + 1))
      if (std::optional<int64_t> delta = foldIndex(add.getRhs(), depth + 1))
        return *base + *delta;
    if (std::optional<int64_t> base =
            inductionOffset(add.getRhs(), iv, depth + 1))
      if (std::optional<int64_t> delta = foldIndex(add.getLhs(), depth + 1))
        return *base + *delta;
    return std::nullopt;
  }
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    if (std::optional<int64_t> base =
            inductionOffset(mul.getLhs(), iv, depth + 1))
      if (std::optional<int64_t> factor = foldIndex(mul.getRhs(), depth + 1))
        return *base * *factor;
    if (std::optional<int64_t> base =
            inductionOffset(mul.getRhs(), iv, depth + 1))
      if (std::optional<int64_t> factor = foldIndex(mul.getLhs(), depth + 1))
        return *base * *factor;
    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXINDEXFOLD_H
