//===- HmxAttrs.cpp - HMX Dialect attributes ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the HMX dialect attributes: the `#hmx.crouton` tensor
// encoding that turns the crouton layout from a shape convention into a checked
// property of the type.
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDType.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::hmx;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom attributes for the dialect.
void HmxDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hexagon/Dialect/Hmx/IR/HmxAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/Hmx/IR/HmxAttrs.cpp.inc"

namespace {

ParseResult parseTailPlanList(AsmParser &parser, StringRef name,
                              SmallVectorImpl<int64_t> &values) {
  if (parser.parseKeyword(name) || parser.parseEqual())
    return failure();

  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square, [&]() -> ParseResult {
            int64_t dim = 0;
            if (parser.parseInteger(dim))
              return failure();
            values.push_back(dim);
            return success();
          }))
    return failure();
  return success();
}

ParseResult parseTailPlanString(AsmParser &parser, StringRef name,
                                std::string &value) {
  if (parser.parseKeyword(name) || parser.parseEqual())
    return failure();
  return parser.parseString(&value);
}

bool alignUpTile(int64_t value, int64_t &result) {
  // NOT-A-DECISION: a local alias of layout::kTileEdge so the arithmetic below
  // reads in one unit. It carries no value of its own -- changing it changes
  // nothing, which is exactly why it is not a decision.
  constexpr int64_t kTile = layout::kTileEdge;
  if (value <= 0 || value > std::numeric_limits<int64_t>::max() - (kTile - 1))
    return false;
  result = ((value + kTile - 1) / kTile) * kTile;
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// Parsing and printing: #hmx.tail_plan<...>
//===----------------------------------------------------------------------===//

Attribute TailPlanAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};

  SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> logical;
  SmallVector<int64_t> padded;
  SmallVector<int64_t> full;
  SmallVector<int64_t> tail;
  std::string kPolicy;
  std::string mnPolicy;
  if (parseTailPlanList(parser, "logical", logical) || parser.parseComma() ||
      parseTailPlanList(parser, "padded", padded) || parser.parseComma() ||
      parseTailPlanList(parser, "full", full) || parser.parseComma() ||
      parseTailPlanList(parser, "tail", tail) || parser.parseComma() ||
      parseTailPlanString(parser, "k_policy", kPolicy) || parser.parseComma() ||
      parseTailPlanString(parser, "mn_policy", mnPolicy) ||
      parser.parseGreater())
    return {};

  return TailPlanAttr::getChecked(
      [&]() { return parser.emitError(loc); }, parser.getContext(), logical,
      padded, full, tail, StringRef(kPolicy), StringRef(mnPolicy));
}

void TailPlanAttr::print(AsmPrinter &printer) const {
  printer << "<logical = [";
  llvm::interleaveComma(getLogical(), printer);
  printer << "], padded = [";
  llvm::interleaveComma(getPadded(), printer);
  printer << "], full = [";
  llvm::interleaveComma(getFull(), printer);
  printer << "], tail = [";
  llvm::interleaveComma(getTail(), printer);
  printer << "], k_policy = ";
  printer.printString(getKPolicy());
  printer << ", mn_policy = ";
  printer.printString(getMnPolicy());
  printer << ">";
}

LogicalResult TailPlanAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> logical,
    ArrayRef<int64_t> padded, ArrayRef<int64_t> full, ArrayRef<int64_t> tail,
    StringRef kPolicy, StringRef mnPolicy) {
  if (logical.size() != 3 || padded.size() != 3 || full.size() != 3 ||
      tail.size() != 3)
    return emitError() << "expects exactly three extents for each of logical, "
                          "padded, full, and tail";

  if (kPolicy != kHmxTailKPolicy)
    return emitError() << "k_policy must be '" << kHmxTailKPolicy
                       << "', got '" << kPolicy << "'";
  if (mnPolicy != kHmxTailMNPolicy)
    return emitError() << "mn_policy must be '" << kHmxTailMNPolicy
                       << "', got '" << mnPolicy << "'";

  for (int64_t i = 0; i < 3; ++i) {
    if (logical[i] <= 0)
      return emitError() << "logical extents must be positive, got ["
                         << logical[0] << ", " << logical[1] << ", "
                         << logical[2] << "]";
    if (padded[i] <= 0 || full[i] < 0 || tail[i] < 0)
      return emitError() << "padded must be positive and full/tail non-negative, "
                            "got padded ["
                         << padded[0] << ", " << padded[1] << ", " << padded[2]
                         << "], full [" << full[0] << ", " << full[1] << ", "
                         << full[2] << "], tail [" << tail[0] << ", "
                         << tail[1] << ", " << tail[2] << "]";
    if (full[i] > logical[i])
      return emitError() << "full extent exceeds logical extent at axis " << i;
    if (full[i] % layout::kTileEdge != 0)
      return emitError() << "full extent must be a multiple of "
                         << layout::kTileEdge << " at axis " << i << ", got "
                         << full[i];
    if (tail[i] != logical[i] - full[i])
      return emitError() << "logical must equal full + tail at axis " << i;
    if (tail[i] >= layout::kTileEdge)
      return emitError() << "tail extent must be less than "
                         << layout::kTileEdge << " at axis " << i << ", got "
                         << tail[i];

    int64_t expectedPadded = 0;
    if (!alignUpTile(logical[i], expectedPadded))
      return emitError() << "logical extent cannot be aligned safely at axis "
                         << i << ", got " << logical[i];
    if (padded[i] != expectedPadded)
      return emitError() << "padded extent must be align_up(logical, 32) at "
                            "axis "
                         << i << ", expected " << expectedPadded << ", got "
                         << padded[i];
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parsing and printing: #hmx.crouton<logical = [M, N]>
//===----------------------------------------------------------------------===//

Attribute CroutonLayoutAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess() || parser.parseKeyword("logical") ||
      parser.parseEqual())
    return {};

  SmallVector<int64_t> logical;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square, [&]() -> ParseResult {
            int64_t dim = 0;
            if (parser.parseInteger(dim))
              return failure();
            logical.push_back(dim);
            return success();
          }))
    return {};

  if (parser.parseGreater())
    return {};

  // `getChecked` (not `get`) so an illegal `logical` is reported as a parse
  // error instead of tripping the attribute uniquer's verification assert.
  SMLoc loc = parser.getCurrentLocation();
  return CroutonLayoutAttr::getChecked(
      [&]() { return parser.emitError(loc); }, parser.getContext(), logical);
}

void CroutonLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<logical = [";
  llvm::interleaveComma(getLogical(), printer);
  printer << "]>";
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//
//
// Two levels of checking, both at type-construction time:
//
//   * `verify` checks the parameters on their own (a logical shape must be a
//     positive pair of 32-multiples);
//   * `verifyEncoding` is the `VerifiableTensorEncoding` hook that
//     `RankedTensorType::verify` calls, so it cross-checks the encoding against
//     the shape and element type it is attached to. This is what makes an
//     illegal crouton type impossible to form: an op verifier never has to
//     repeat it.
//
// Rejected (see docs/hmx/layout-representation-survey.md section 4.2):
//   1. rank != 5                 -- a crouton array is a 2D grid of croutons;
//   2. element type != f16       -- the engine only reads f16;
//   3. tail != {16, 32, 2}       -- not a crouton;
//   4. non-empty grid            -- shape[0] > 0 and shape[1] > 0;
//   5. logical != grid * 32      -- the logical shape and the physical grid
//                                   must describe the same matrix.

LogicalResult
CroutonLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<int64_t> logical) {
  if (logical.size() != 2)
    return emitError() << "expects exactly two logical dimensions, got "
                       << logical.size();
  if (logical[0] <= 0 || logical[1] <= 0)
    return emitError() << "logical dimensions must be positive, got ["
                       << logical[0] << ", " << logical[1] << "]";
  if (logical[0] % kTile != 0 || logical[1] % kTile != 0)
    return emitError() << "logical dimensions must be multiples of the crouton "
                          "tile "
                       << kTile << ", got [" << logical[0] << ", " << logical[1]
                       << "]";
  return success();
}

LogicalResult CroutonLayoutAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (shape.size() != 5)
    return emitError() << "hmx.crouton expects a rank-5 crouton array, got "
                          "rank "
                       << shape.size();
  if (!dtype::isCroutonElement(elementType))
    return emitError() << "hmx.crouton expects f16 elements, got "
                       << elementType;
  if (shape[2] != kCroutonPair || shape[3] != kCroutonCol ||
      shape[4] != kCroutonHalf)
    return emitError() << "must end in a crouton: ..., " << kCroutonPair << ", "
                       << kCroutonCol << ", " << kCroutonHalf;
  if (shape[0] <= 0 || shape[1] <= 0)
    return emitError() << "crouton grid must be non-empty, got ["
                       << shape[0] << ", " << shape[1] << "]";
  if (getLogical()[0] != shape[0] * kTile || getLogical()[1] != shape[1] * kTile)
    return emitError() << "logical shape [" << getLogical()[0] << ", "
                       << getLogical()[1] << "] must equal grid * " << kTile
                       << " = [" << shape[0] * kTile << ", "
                       << shape[1] * kTile << "]";
  return success();
}

//===----------------------------------------------------------------------===//
// Layout helpers (see include/hexagon/Dialect/Hmx/IR/HmxDialect.h)
//===----------------------------------------------------------------------===//

CroutonLayoutAttr mlir::hmx::getCroutonEncoding(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return {};
  return dyn_cast_or_null<CroutonLayoutAttr>(tensor.getEncoding());
}

bool mlir::hmx::hasCroutonEncoding(Type type) {
  return static_cast<bool>(getCroutonEncoding(type));
}

bool mlir::hmx::sameCroutonEncoding(Type a, Type b) {
  auto ea = getCroutonEncoding(a);
  auto eb = getCroutonEncoding(b);
  if (!ea || !eb)
    return false;
  // AR == AH == WH: the role is a property of the use, not of the type, so it is
  // deliberately not part of the encoding. Two crouton arrays are
  // interchangeable exactly when they hold the same logical f16 matrix.
  return ea.getLogical() == eb.getLogical() &&
         cast<RankedTensorType>(a).getElementType() ==
             cast<RankedTensorType>(b).getElementType();
}

RankedTensorType mlir::hmx::croutonLayoutType(RankedTensorType logical) {
  assert(logical.hasStaticShape() && logical.getRank() == 2 &&
         "crouton layout needs a static rank-2 matrix");
  // The crouton's element type is the engine's fp16, not the source's: a wider
  // source is quantised by the pack that materialises the crouton
  // (`hmx.pack_act` / `hmx.pack_weight`), so every crouton array holds the fp16
  // image of its logical matrix whatever the source's element type is.
  int64_t m = logical.getDimSize(0);
  int64_t n = logical.getDimSize(1);
  assert(m % CroutonLayoutAttr::kTile == 0 &&
         n % CroutonLayoutAttr::kTile == 0 &&
         "crouton layout needs 32-aligned extents");

  auto encoding = CroutonLayoutAttr::get(logical.getContext(), {m, n});
  return RankedTensorType::get(
      {m / CroutonLayoutAttr::kTile, n / CroutonLayoutAttr::kTile,
       CroutonLayoutAttr::kCroutonPair, CroutonLayoutAttr::kCroutonCol,
       CroutonLayoutAttr::kCroutonHalf},
      dtype::croutonElementType(logical.getContext()), encoding);
}

//===----------------------------------------------------------------------===//
// The memref-side carrier: #hmx.crouton_memref_layout
//
// Once the crouton array is materialised the rank-raising permutation is done,
// so on a memref the layout is contiguous storage over the rank-5 shape. This
// attribute is the identity map over that shape (the one thing
// `MemRefLayoutAttrInterface` can express for a crouton), carrying `logical`
// so the memref side keeps the cross-check the tensor encoding had. It is
// created by the pipeline's bufferization `unknownTypeConverterFn` from an
// encoded tensor; see LinalgToLLVMPass.
//===----------------------------------------------------------------------===//

Attribute CroutonMemRefLayoutAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess() || parser.parseKeyword("logical") ||
      parser.parseEqual())
    return {};

  SmallVector<int64_t> logical;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square, [&]() -> ParseResult {
            int64_t dim = 0;
            if (parser.parseInteger(dim))
              return failure();
            logical.push_back(dim);
            return success();
          }))
    return {};

  if (parser.parseGreater())
    return {};

  // `getChecked` (not `get`) so an illegal `logical` is reported as a parse
  // error instead of tripping the attribute uniquer's verification assert.
  SMLoc loc = parser.getCurrentLocation();
  return CroutonMemRefLayoutAttr::getChecked(
      [&]() { return parser.emitError(loc); }, parser.getContext(), logical);
}

void CroutonMemRefLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<logical = [";
  llvm::interleaveComma(getLogical(), printer);
  printer << "]>";
}

LogicalResult
CroutonMemRefLayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<int64_t> logical) {
  // Same parameter contract as the tensor encoding.
  return CroutonLayoutAttr::verify(emitError, logical);
}

// The memref's layout map: the identity over the rank-5 crouton array. Because
// it is an identity, the interface's default `isIdentity()` holds and
// `getStridesAndOffset` yields exactly the static canonical strides of a
// layout-less memref of the same shape -- the dense contract the lowering
// already assumes.
AffineMap CroutonMemRefLayoutAttr::getAffineMap() const {
  // The rank is pinned by `verifyLayout`: this layout is only legal on the
  // rank-5 crouton array (`grid x 16 x 32 x 2`).
  return AffineMap::getMultiDimIdentityMap(
      /*numDims=*/5, getContext());
}

// `MemRefType::verify` calls this whenever the type is formed, so an illegal
// pairing of this layout with a shape cannot exist in the IR. The checks are
// `CroutonLayoutAttr::verifyEncoding`'s shape conditions minus the element
// type (a layout never sees it).
LogicalResult CroutonMemRefLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (shape.size() != 5)
    return emitError() << "hmx.crouton_memref_layout expects a rank-5 crouton "
                          "array, got rank "
                       << shape.size();
  if (shape[2] != kCroutonPair || shape[3] != kCroutonCol ||
      shape[4] != kCroutonHalf)
    return emitError() << "must end in a crouton: ..., " << kCroutonPair << ", "
                       << kCroutonCol << ", " << kCroutonHalf;
  if (shape[0] <= 0 || shape[1] <= 0)
    return emitError() << "crouton grid must be non-empty, got ["
                       << shape[0] << ", " << shape[1] << "]";
  if (getLogical()[0] != shape[0] * kTile ||
      getLogical()[1] != shape[1] * kTile)
    return emitError() << "logical shape [" << getLogical()[0] << ", "
                       << getLogical()[1] << "] must equal grid * " << kTile
                       << " = [" << shape[0] * kTile << ", " << shape[1] * kTile
                       << "]";
  return success();
}

//===----------------------------------------------------------------------===//
// Encoding → memref-layout handoff (see include/hexagon/Dialect/Hmx/IR/HmxDialect.h)
//===----------------------------------------------------------------------===//

// The memref-side layout for an encoded tensor, or a null attribute when
// `type` is not a crouton-encoded ranked tensor. This is the mapping the
// bufferization seam applies: the pinned one-shot bufferize pass cannot be
// given a custom type converter, so the dialect has its own allocation op
// (`hmx.alloc_crouton`, emitted by `matmul-to-hmx`) whose BufferizableOpInterface
// maps its encoded result to a memref carrying this layout.
CroutonMemRefLayoutAttr mlir::hmx::croutonMemRefLayoutOf(Type type) {
  CroutonLayoutAttr encoding = getCroutonEncoding(type);
  if (!encoding)
    return {};
  return CroutonMemRefLayoutAttr::get(type.getContext(),
                                      encoding.getLogical());
}
