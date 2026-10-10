//===-- HmxDType.h - the engine's element-type vocabulary --------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The one home for the HMX engine's element-type contract, next to
// `HmxCroutonLayout.h`'s physical-shape constants. A pass or a verifier asks
// this vocabulary "is this the engine's element type", not "is this f16": the
// source operand, the crouton the engine computes on, and the read-out are
// three different things, and each has one name here.
//
// The ODS constraints the ops already carry (`F16`, `F16, F32`) express the same
// contract, but they are no longer two copies kept in step by hand:
// test_hmx_dtype_admitted_set_contract.py binds the ODS element lists to
// `dtype::isAdmittedFloat` below (and to the lowering's leaf-family tables and
// the host prepack's dtype map), so a one-sided rename is a red test rather
// than a silent drift.
//
// Why not MLIR's type system alone: MLIR represents a float as `mlir::FloatType`
// (the base of every f16/f32/bf16/f8) and offers `Type::isFloat()`,
// `Type::isFloat(unsigned width)` and `getIntOrFloatBitWidth()`; there is no
// `DType`/`DataType` class. A width predicate is deliberately NOT used here:
// `isFloat(16)` matches both f16 and bf16, and the engine's crouton is
// specifically f16 (bf16 is not an operand type -- see
// docs/hmx/dtype-capability-research-2026-09-28.md). The exact kind is kept, and
// this vocabulary is what a new dtype extends.
//
//   * the engine computes in fp16 -- every crouton array holds fp16, whatever
//     the source's element type is (a wider source is quantised by the pack
//     that materialises the crouton: `hmx.pack_act` / `hmx.pack_weight`);
//   * a source operand or a read-out may be fp16 or fp32: fp32 is packed to the
//     crouton's fp16 on the way in and widened back on the way out.
//
// A quantized-operand contract (int8/bf16/fp8) extends this vocabulary, and only
// this vocabulary, rather than another `isF16()` at a call site.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_IR_HMXDTYPE_H
#define HEXAGON_DIALECT_HMX_IR_HMXDTYPE_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hmx {
namespace dtype {

/// The engine's fp16 element. Every crouton array holds it -- including the
/// image a prepack/resident weight materialises -- whatever the source's
/// element type is: a wider source is quantised to it by the pack, so "what
/// may a source be" is an `isAdmittedFloat` question, not this one.
///
/// A quantized crouton (int8) would change this predicate. Sites that mean "an
/// f16 payload the layout whitelist can re-host" rather than "the engine's
/// element" must then name their own predicate instead of following this one.
inline bool isCroutonElement(Type type) { return type.isF16(); }

/// The engine's fp16 element as a type, for building a crouton array's type.
inline Type croutonElementType(MLIRContext *context) {
  return Float16Type::get(context);
}

/// A floating element the HMX path admits on a source operand or a read-out:
/// fp16 directly, fp32 through the pack (in) / widen (out). Anything else --
/// other float widths, integers, bytes -- is outside the engine's contract.
inline bool isAdmittedFloat(Type type) { return type.isF16() || type.isF32(); }

/// The wider member of the admitted set: fp32. A source of this type is
/// quantised to the crouton's fp16 by the pack; a read-out of this type is what
/// the unpack widens to. The exact kind is checked (not `isFloat(32)`) so the
/// set stays exactly the one `isAdmittedFloat` names.
inline bool isF32(Type type) { return type.isF32(); }

} // namespace dtype
} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_IR_HMXDTYPE_H
