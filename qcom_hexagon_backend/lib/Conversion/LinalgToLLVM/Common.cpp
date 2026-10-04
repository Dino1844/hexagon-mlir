//===- Common.cpp - some useful common types and functions ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements some common types and functions used across
// the Hexagon backend conversion passes.
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace hexagon {

// === Row-reduce matching shared by the vector-row-reduce pass and the ===
// === vectorizer's skip gate                                          ===

// kHvxVectorBytes (the 128 B row-chunk width) lives in Common.h: it is one
// contract shared with VectorRowReducePass, not a file-local constant.

std::optional<RowReduceShape> vectorRowReduceShapeOf(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  // Rank 1 = a single row reduced to a scalar (the form LinalgGeneralize
  // produces for a one-dimensional row); rank 2 = a batch of rows.
  if (!shaped || !shaped.hasStaticShape() ||
      (shaped.getRank() != 1 && shaped.getRank() != 2))
    return std::nullopt;
  Type elemTy = shaped.getElementType();
  int64_t elemBytes;
  if (elemTy.isF32())
    elemBytes = 4;
  else if (elemTy.isF16())
    elemBytes = 2;
  else
    return std::nullopt;
  // A row must be contiguous (the vector reads walk it with unit stride);
  // the outer stride may be anything -- each row is addressed on its own.
  int64_t innerStride;
  if (auto memref = dyn_cast<MemRefType>(type)) {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(memref.getStridesAndOffset(strides, offset)) || strides.empty())
      return std::nullopt;
    innerStride = strides.back();
  } else if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    // No encoding = dense row-major. Any encoding (a crouton layout, say)
    // means the logical row order is not the memory order -- leave it alone.
    if (tensor.getEncoding())
      return std::nullopt;
    innerStride = 1;
  } else {
    return std::nullopt;
  }
  if (innerStride != 1)
    return std::nullopt;
  int64_t cols = shaped.getDimSize(shaped.getRank() - 1);
  int64_t rows = shaped.getRank() == 2 ? shaped.getDimSize(0) : 1;
  if (rows == 0 || cols == 0)
    return std::nullopt;
  // The row-width divisibility gate lives in matchVectorRowReduce, not here:
  // the chunk width that a row must be a whole multiple of depends on the
  // fold's element type (an f16 row summed in f32 chunks at the f32 width),
  // which only the body contract knows.
  return RowReduceShape{rows, cols, elemBytes};
}

std::optional<RowReduceFold> matchVectorRowReduce(linalg::LinalgOp op) {
  // Either the bufferized form (no results; the outs are memrefs) or the
  // tensor form (one result; the init is a tensor).
  if (op->getNumResults() > 1)
    return std::nullopt;

  // The op reaches us either still as a `linalg.reduce` or, since
  // LinalgGeneralize rewrites every reduce to a generic earlier in the
  // pipeline, as a `linalg.generic` carrying a reduction iterator. Both are
  // accepted -- same predicate for the vectorizer's skip gate and for this
  // pass, so the two can never disagree.
  Value in, init;
  SmallVector<int64_t, 4> reducePos, parallelPos;
  if (auto ro = dyn_cast<linalg::ReduceOp>(op.getOperation())) {
    if (ro.getInputs().size() != 1 || ro.getInits().size() != 1)
      return std::nullopt;
    in = ro.getInputs()[0];
    init = ro.getInits()[0];
    auto inTy = dyn_cast<ShapedType>(in.getType());
    if (!inTy || !inTy.hasStaticShape())
      return std::nullopt;
    reducePos.assign(ro.getDimensions().begin(), ro.getDimensions().end());
    for (int64_t i = 0, r = inTy.getRank(); i < r; ++i)
      if (!llvm::is_contained(reducePos, i))
        parallelPos.push_back(i);
  } else if (auto g = dyn_cast<linalg::GenericOp>(op.getOperation())) {
    if (g.getInputs().size() != 1 || g.getOutputs().size() != 1)
      return std::nullopt;
    if (g.getBody()->getArguments().size() != 2)
      return std::nullopt;
    in = g.getInputs()[0];
    init = g.getOutputs()[0];
    auto iters = g.getIteratorTypesArray();
    for (auto [i, t] : llvm::enumerate(iters)) {
      if (t == utils::IteratorType::reduction)
        reducePos.push_back((int64_t)i);
      else
        parallelPos.push_back((int64_t)i);
    }
    if (reducePos.size() != 1)
      return std::nullopt;
    // The input must read the iteration space in order (minor identity), and
    // the result may only mention parallel dimensions -- otherwise the value
    // written per row is not that row's reduction.
    auto maps = g.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isMinorIdentity())
      return std::nullopt;
    for (auto d : maps[1].getResults()) {
      auto de = dyn_cast<AffineDimExpr>(d);
      if (!de || !llvm::is_contained(parallelPos, (int64_t)de.getPosition()))
        return std::nullopt;
    }
  } else {
    return std::nullopt;
  }

  auto shape = vectorRowReduceShapeOf(in.getType());
  if (!shape) // rows == 0 already rejected inside vectorRowReduceShapeOf
    return std::nullopt;
  // Exactly one reduction and it must be the innermost dimension: that is what
  // keeps a row contiguous under unit stride (the mechanical gate).
  if (reducePos.size() != 1)
    return std::nullopt;
  auto inShaped = cast<ShapedType>(in.getType());
  if (reducePos[0] != inShaped.getRank() - 1)
    return std::nullopt;

  // The body contract: one binary fold (maxnumf/addf) whose one operand is
  // the init block arg and whose other operand is the input block arg,
  // possibly through a chain of pure elementwise ops fused into the body
  // (rms_norm's x*x before the row sum -- by the time anyone looks, the
  // elementwise producer has been fused in and the row chunked). The chain
  // may upcast (an f16 row summed in f32, Triton's tl.sum on f16); the fold
  // and accumulator then live at the wider element type. Everything else
  // stays on the scalar path.
  Block &block = op->getRegion(0).front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  Operation *bin = yield->getOperand(0).getDefiningOp();
  if (!bin)
    return std::nullopt;
  Value blockIn = block.getArgument(0), blockInit = block.getArgument(1);

  bool isMaxNum;
  arith::FastMathFlags fastmath;
  if (auto maxnumf = dyn_cast<arith::MaxNumFOp>(bin)) {
    isMaxNum = true;
    fastmath = maxnumf.getFastmath();
  } else if (auto addf = dyn_cast<arith::AddFOp>(bin)) {
    isMaxNum = false;
    fastmath = addf.getFastmath();
  } else {
    return std::nullopt;
  }
  // Exactly one side of the fold is the running init; the other side is the
  // row value the chain computes.
  Value folded;
  if (bin->getOperand(0) == blockInit && bin->getOperand(1) != blockInit)
    folded = bin->getOperand(1);
  else if (bin->getOperand(1) == blockInit && bin->getOperand(0) != blockInit)
    folded = bin->getOperand(0);
  else
    return std::nullopt;
  // The fold's element type: the accumulator's. Must be a scalar f16/f32 the
  // butterfly can run at, and at least as wide as the row's own element type
  // (a narrowing fold would need a chunk wider than one HVX vector).
  Type foldElemTy = blockInit.getType();
  int64_t foldElemBytes;
  if (foldElemTy.isF32())
    foldElemBytes = 4;
  else if (foldElemTy.isF16())
    foldElemBytes = 2;
  else
    return std::nullopt;
  if (folded.getType() != foldElemTy)
    return std::nullopt;
  if (foldElemBytes < shape->elemBytes)
    return std::nullopt;
  // Each chunk the rewrite reads is one HVX vector at the row's own width,
  // so the row must be a whole number of those (independent of the fold's
  // width: a widening fold accumulates wide and halves down afterwards).
  if ((shape->cols * shape->elemBytes) % kHvxVectorBytes != 0)
    return std::nullopt;

  // The chain: every body op except the fold and the yield, in block order.
  // A member is either a scalar float constant (f16 or f32 -- the rewrite
  // re-anchors it as a broadcast) or a pure, single-result,
  // elementwise-mappable op of a scalar f16/f32 type whose operands are
  // closed over the input arg and earlier chain results. `Vectorizable` is
  // the trait contract "all scalar operands and results are replaced by
  // vectors of the respective element type" -- exactly the re-creation the
  // rewrite performs -- so an op without it (or an i1-producing compare,
  // which the element-type gate rejects) keeps the whole reduce scalar. The
  // init arg may not enter the chain: it is the running accumulator, not a
  // per-element value.
  SmallVector<Operation *, 4> chain;
  auto isChainValue = [&](Value v) {
    return v == blockIn ||
           (v.getDefiningOp() && llvm::is_contained(chain, v.getDefiningOp()));
  };
  for (auto &bodyOp : block.getOperations()) {
    if (&bodyOp == bin || isa<linalg::YieldOp>(bodyOp))
      continue;
    if (auto cst = dyn_cast<arith::ConstantOp>(&bodyOp)) {
      if (!isa<FloatAttr>(cst.getValue()) ||
          (!cst.getType().isF32() && !cst.getType().isF16()))
        return std::nullopt;
    } else {
      Type resTy = bodyOp.getNumResults() == 1
                       ? bodyOp.getResult(0).getType()
                       : Type();
      if (!isMemoryEffectFree(&bodyOp) ||
          !bodyOp.hasTrait<OpTrait::Vectorizable>() ||
          bodyOp.getNumResults() != 1 ||
          (!resTy.isF32() && !resTy.isF16()))
        return std::nullopt;
      for (Value operand : bodyOp.getOperands())
        if (!isChainValue(operand))
          return std::nullopt;
    }
    chain.push_back(&bodyOp);
  }
  // The folded row value must be what the chain computes: the input arg
  // itself (the classic chain-less body) or a chain result -- and it must
  // actually depend on the input arg, otherwise the body ignores the row it
  // is reducing (a constant folded straight into the fold).
  SmallVector<Value, 8> dependsOnIn;
  auto depends = [&](Value v) {
    return llvm::is_contained(dependsOnIn, v);
  };
  dependsOnIn.push_back(blockIn);
  for (Operation *op : chain) {
    bool any = false;
    for (Value operand : op->getOperands())
      any = any || depends(operand);
    if (any)
      dependsOnIn.push_back(op->getResult(0));
  }
  if (!depends(folded))
    return std::nullopt;

  // The init (outs) and, in tensor form, the result carry one value per row
  // of the fold's element type. A single-row reduce writes a scalar.
  auto perRow = [&](Type t) {
    auto s = dyn_cast<ShapedType>(t);
    if (!s || !s.hasStaticShape() || s.getElementType() != foldElemTy)
      return false;
    if (shape->rows == 1)
      return s.getRank() == 0 ||
             (s.getRank() == 1 && s.getDimSize(0) == 1);
    return s.getRank() == 1 && s.getDimSize(0) == shape->rows;
  };
  if (!perRow(init.getType()))
    return std::nullopt;
  if (op->getNumResults() == 1 && !perRow(op->getResult(0).getType()))
    return std::nullopt;

  return RowReduceFold{chain, *shape, isMaxNum, fastmath, foldElemTy};
}

// Does the generic body carry an exp-family transcendental?
//
// This gates the i1 handling below, and it is deliberately narrow. A tensor of
// i1 is a predicate (mask), not data, so its 1-byte storage must not drive the
// vector width: counting it makes computeDataTileSize return 128 for an f32
// loop of 64, and perfectlyVectorizable then demands `tile == innerLoop`
// (128 != 64) and blocks vectorization of the entire chain. When that chain is
// a masked feature map -- compare + select around a `math.exp` -- the exp never
// reaches a native-width vector and hexagon-clang later scalarizes it into
// per-element `call expf` plus soft f16<->f32 conversions (`llvm.exp` on
// <32 x float> has no Hexagon lowering; only `llvm.exp2` does).
//
// Other i1-consuming chains (e.g. the causal mask / sum chains of chunked
// linear attention) are *not* stalled this way -- they vectorize fine once the
// surrounding tiling is left alone -- so opening the door for them changes
// their shape for no gain. Gate on the exp family to fix exactly the chain the
// mechanism actually blocks. See hexagon-mlir-local.patch.
static bool bodyHasExpFamilyOp(Operation *op) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp)
    return false;
  return llvm::any_of(genericOp.getBody()->getOperations(), [](Operation &op) {
    return isa<math::ExpOp, math::Exp2Op, math::ExpM1Op>(&op);
  });
}

std::optional<unsigned> computeSmallestOperandTypeSize(Operation *op) {
  // Returns the size of the smallest input operand's type.
  // Choosing the smallest-sized elements results in the largest tile size,
  // which aligns better with Hexagon. We prefer breaking large vectors
  // into multiple vectors over dealing with partial vectors.
  //
  // i1 (mask) operands/results are skipped, but only for the exp-family
  // generics described above; every other op keeps the original behaviour.
  const bool skipMaskElements = bodyHasExpFamilyOp(op);

  unsigned int minElemSize = maxElemSizeInByte + 1;
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type operandType = opOperand.get().getType();
    if (auto type = dyn_cast<RankedTensorType>(operandType)) {
      if (skipMaskElements && type.getElementType().isInteger(1))
        continue;
      auto operandSize = getElementSizeInBytes(type);
      if (operandSize) {
        minElemSize = std::min(minElemSize, *operandSize);
      }
    }
  }

  // For genericOp determine by inspecting body ops.
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (Operation &op : genericOp.getBody()->getOperations()) {
      if (op.hasTrait<OpTrait::Vectorizable>()) {
        for (auto res : op.getResults()) {
          if (skipMaskElements &&
              getElementType(res.getType()).isInteger(1))
            continue;
          auto resSize = getElementSizeInBytes(res.getType());
          if (resSize)
            minElemSize = std::min(minElemSize, *resSize);
        }
      }
    }
  }

  if (minElemSize < maxElemSizeInByte + 1) {
    return minElemSize;
  }
  return std::nullopt;
}

Type getElementType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<VectorType, RankedTensorType>(
          [](auto ty) { return ty.getElementType(); })
      .Default([](Type ty) -> Type { return ty; });
}

std::optional<unsigned> getElementSizeInBytes(Type type) {
  Type elemType = getElementType(type);
  if (elemType.isInteger(1) || elemType.isInteger(8))
    return 1;
  if (elemType.isF16() || elemType.isBF16() || elemType.isInteger(16))
    return 2;
  if (elemType.isF32() || elemType.isInteger(32))
    return 4;
  if (elemType.isF64() || elemType.isInteger(64))
    return 8;
  return std::nullopt;
}

std::optional<unsigned> computeDataTileSize(Operation *op) {
  auto elSize = computeSmallestOperandTypeSize(op);
  if (!elSize)
    return std::nullopt;
  return nativeVectorWidthInBytes / *elSize;
}

bool isPerfectlyTileable(unsigned LoopRange, unsigned dataTileSize) {
  if (LoopRange % dataTileSize != 0) {
    return false;
  }
  return true;
}

bool perfectlyVectorizable(unsigned dataTileSize, unsigned innerLoopRange) {
  return dataTileSize == innerLoopRange;
}

void setTargetTriple(ModuleOp moduleOp) {
  std::string targetTripleStr = "hexagon";
  moduleOp->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                    StringAttr::get(moduleOp->getContext(), targetTripleStr));
}

void setDataLayout(ModuleOp moduleOp) {
  std::string dataLayoutStr =
      "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32"
      "-i16:16:16-i1"
      ":8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512"
      ":512:512-v1024:1024:1024-v2048:2048:2048";

  // Commented out till index i64/i32 is sorted.
  // moduleOp->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
  //                   StringAttr::get(moduleOp->getContext(), dataLayoutStr));
}

bool doesFuncReturnValue(ModuleOp moduleOp) {

  return moduleOp
      .walk([&](func::FuncOp op) {
        if (!op.getResultTypes().empty())
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

} // namespace hexagon

} // namespace mlir
