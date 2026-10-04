//===- VectorRowReducePass.cpp - 2-D row reduce to vector fold + butterfly -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// A 2-D last-dim `linalg.reduce` (f16/f32 max/add) that reaches the Hexagon
// backend is scalarized: ConvertLinalgToLoops turns it into a per-lane
// scf.for chain, and HexagonISelLowering only gives VECREDUCE_ADD an HVX DAG
// combine -- a `llvm.vector.reduce.fmax` becomes a serial
// memw/sfcmp/mux/vinsert chain per lane, which was 55% of the FA softmax
// kernel body.
//
// This pass replaces the reduce before that point with explicit vector IR:
// each row's chunks -- one HVX vector at the fold's width -- are folded
// elementwise (one vector max/add per chunk), then a vror butterfly --
// rotate the accumulator by half the register and fold, halving the lane
// span each step, llama.cpp hvx-reduce.h style -- leaves every lane of one
// HVX vector holding the row's reduction. An elementwise producer fused
// into the body (rms_norm's x*x, an f16 row's upcast to an f32 fold) is
// re-created on each chunk before it folds in. The original outs init is
// folded in and lane 0 is stored back, so the scalar contract of the
// reduce (init folded in, one value per row) is preserved; the
// butterfly's all-lanes-redundant result is what a later consumer
// optimization (splats instead of extract) can exploit.
//
// Everything it cannot prove static, contiguous and whole-vector is left for
// the scalar path. Off by default (`enable-vector-row-reduce`): the knob is
// the device A/B switch, not a correctness guard.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Dialect/Hvx/IR/HvxDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::hexagon;

// At global scope, like the rest of this family: the generated pass base lands
// in ::impl.
#define GEN_PASS_DEF_VECTORROWREDUCE
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

// kHvxVectorBytes (the 128 B row-chunk width) comes from Common.h -- one
// contract shared with the vectorizer's skip gate.

struct VectorRowReducePass
    : public ::impl::VectorRowReduceBase<VectorRowReducePass> {
public:
  explicit VectorRowReducePass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    // The rewrite emits vector/scf/memref/arith ops and hvx.vror; without
    // this a kernel whose input never mentions the vector dialect would die
    // on "created with unregistered dialect".
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect, hvx::HvxDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    // Collect first: rewriting invalidates the walk.
    SmallVector<std::pair<linalg::LinalgOp, RowReduceFold>> candidates;
    fn.walk([&](linalg::LinalgOp op) {
      // Bufferized form only: the rewrite walks memrefs.
      if (op->getNumResults() != 0)
        return;
      if (auto fold = matchVectorRowReduce(op))
        candidates.emplace_back(op, *fold);
    });
    for (auto &[op, fold] : candidates) {
      if (failed(rewrite(op, fold)))
        return signalPassFailure();
    }
  }

private:
  static Value emitFold(OpBuilder &b, Location loc, const RowReduceFold &fold,
                        Value l, Value r) {
    if (fold.isMaxNum)
      return arith::MaxNumFOp::create(b, loc, l, r, fold.fastmath).getResult();
    return arith::AddFOp::create(b, loc, l, r, fold.fastmath).getResult();
  }

  // Re-create the fused elementwise chain (rms_norm's x*x; an f16 row's
  // upcast to an f32 fold) over one chunk of `lanes` input elements. The
  // matcher validated every member as pure and elementwise mappable, so each
  // op re-creates at vector<lanes x its scalar type>; a scalar constant
  // re-anchors as a broadcast. The classic chain-less body returns the chunk
  // unchanged.
  static FailureOr<Value> emitChain(OpBuilder &b, Location loc,
                                    const RowReduceFold &fold, int64_t lanes,
                                    Value chunk, Value blockIn) {
    if (fold.chain.empty())
      return chunk;
    auto vecOf = [&](Type scalarTy) {
      return VectorType::get({lanes}, scalarTy);
    };
    llvm::SmallDenseMap<Value, Value> vmap;
    vmap[blockIn] = chunk;
    Value last = chunk;
    for (Operation *op : fold.chain) {
      if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
        Value scalar = arith::ConstantOp::create(b, loc, cst.getType(),
                                                 cst.getValue());
        last = vector::BroadcastOp::create(b, loc, vecOf(cst.getType()),
                                           scalar);
      } else {
        SmallVector<Value, 4> operands;
        for (Value operand : op->getOperands()) {
          auto it = vmap.find(operand);
          if (it == vmap.end()) {
            // The matcher closes the chain over the input arg and earlier
            // results; reaching here means that contract was violated.
            op->emitOpError("row-reduce rewrite: chain operand not mapped");
            return failure();
          }
          operands.push_back(it->second);
        }
        OperationState state(loc, op->getName());
        state.addOperands(operands);
        state.addTypes(vecOf(op->getResult(0).getType()));
        state.addAttributes(op->getAttrs());
        last = b.create(state)->getResult(0);
      }
      vmap[op->getResult(0)] = last;
    }
    return last;
  }

  LogicalResult rewrite(linalg::LinalgOp op, const RowReduceFold &fold) const {
    // A generalized reduce arrives as linalg.generic (LinalgGeneralize runs
    // earlier), a reduce still as linalg.reduce; the outs carry one value per
    // row (or a scalar for a single-row reduce).
    Value src, dst;
    if (auto ro = dyn_cast<linalg::ReduceOp>(op.getOperation())) {
      src = ro.getInputs()[0];
      dst = ro.getInits()[0];
    } else if (auto g = dyn_cast<linalg::GenericOp>(op.getOperation())) {
      src = g.getInputs()[0];
      dst = g.getOutputs()[0];
    } else {
      op->emitOpError("row-reduce rewrite: unsupported linalg op kind");
      return failure();
    }
    if (!src || !dst) {
      op->emitOpError("row-reduce rewrite: missing src/dst");
      return failure();
    }
    if (!isa<MemRefType>(src.getType()) || !isa<MemRefType>(dst.getType())) {
      op->emitOpError("row-reduce rewrite: not bufferized (src=")
          .attachNote()
          << src.getType() << " dst=" << dst.getType();
      return failure();
    }
    auto srcTy = cast<MemRefType>(src.getType());
    auto dstTy = cast<MemRefType>(dst.getType());
    Type elemTy = srcTy.getElementType();
    // The fold (and the accumulator/init) may be wider than the row's own
    // element type when the fused chain upcasts (an f16 row summed in f32).
    // Chunks are read at the row's own width (a full HVX vector of input
    // elements); a widening fold accumulates at the wider lane count and is
    // halved down to one HVX vector of fold lanes before the butterfly.
    Type foldElemTy = fold.foldElemTy;
    int64_t foldElemBytes = foldElemTy.isF32() ? 4 : 2;
    RowReduceShape shape = fold.shape;

    int64_t lanes = kHvxVectorBytes / foldElemBytes; // f32: 32, f16: 64
    int64_t chunkLanes = kHvxVectorBytes / shape.elemBytes;
    int64_t chunks = shape.cols / chunkLanes;
    auto readTy = VectorType::get({chunkLanes}, elemTy);
    auto accTy = VectorType::get({chunkLanes}, foldElemTy);
    auto vecTy = VectorType::get({lanes}, foldElemTy);

    Location loc = op.getLoc();
    OpBuilder b(op);

    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value cRows = arith::ConstantIndexOp::create(b, loc, shape.rows);
    Value pad = arith::ConstantOp::create(
        b, loc, elemTy, b.getZeroAttr(elemTy));

    // One row per iteration. `in_bounds` is true: the row is a full,
    // statically known extent.
    auto loop = scf::ForOp::create(
        b, loc, c0, cRows, c1, ValueRange{},
        [](OpBuilder &, Location, Value, ValueRange) {});
    loop->moveBefore(op);
    b.setInsertionPointToStart(loop.getBody());
    Value r = loop.getInductionVar();

    // The row as a contiguous 1-D view (rank reduced by the size-1 row dim);
    // the outer stride stays in the subview. Static sizes/strides: the
    // rank-reduced verifier check needs the dropped dim to be statically 1.
    // The memory space carries over: the reduce source may live in VTCM
    // (hexagonmem, space 1), and the reads must stay in it.
    Value row;
    if (srcTy.getRank() == 2) {
      // Rank-reduced view of the size-1 row dim of the current row; the outer
      // stride stays in the subview, which is why the offset is dynamic.
      // Static sizes/strides: the rank-reduced verifier check needs the
      // dropped dim to be statically 1. The memory space carries over: the
      // source may live in VTCM (hexagonmem, space 1), and the reads must
      // stay in it.
      auto rowTy = MemRefType::get(
          {shape.cols}, elemTy,
          StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, {1}),
          srcTy.getMemorySpace());
      SmallVector<OpFoldResult> rowOffsets{r, b.getIndexAttr(0)};
      SmallVector<OpFoldResult> rowSizes{b.getIndexAttr(1),
                                         b.getIndexAttr(shape.cols)};
      SmallVector<OpFoldResult> rowStrides{b.getIndexAttr(1),
                                           b.getIndexAttr(1)};
      row = memref::SubViewOp::create(b, loc, rowTy, src, rowOffsets, rowSizes,
                                      rowStrides);
    } else {
      // Single row (rank-1 source): the source *is* the row. No subview, so
      // no layout to guess -- a subview here would have to declare an offset
      // segment that the natural result (static offset 0) does not have.
      row = src;
    }

    // Cross-chunk elementwise fold: lane i holds the fold over the same lane
    // of every chunk of the row (one HVX vector of input elements). A fused
    // elementwise producer (the chain) is applied to each chunk before it
    // folds in.
    static constexpr bool kInBounds[] = {true};
    Value blockIn = op->getRegion(0).front().getArgument(0);
    Value acc;
    for (int64_t k = 0; k < chunks; ++k) {
      Value col = k == 0 ? c0 : arith::ConstantIndexOp::create(b, loc, k * chunkLanes);
      Value chunk = vector::TransferReadOp::create(
          b, loc, readTy, row, ValueRange{col}, pad,
          llvm::ArrayRef<bool>(kInBounds));
      FailureOr<Value> mapped = emitChain(b, loc, fold, chunkLanes, chunk,
                                          blockIn);
      if (failed(mapped))
        return failure();
      acc = k == 0 ? *mapped : emitFold(b, loc, fold, acc, *mapped);
    }

    // A widening fold accumulated more lanes than one HVX vector of fold
    // elements: shuffle the wide accumulator into halves and fold them
    // together, halving until it is one fold-width vector. (The fold is
    // commutative, so folding the halves in either order is the same row
    // value.)
    for (int64_t wide = chunkLanes; wide > lanes; wide /= 2) {
      auto halfTy = VectorType::get({wide / 2}, foldElemTy);
      SmallVector<int64_t, 64> lo, hi;
      for (int64_t i = 0; i < wide / 2; ++i)
        lo.push_back(i);
      for (int64_t i = wide / 2; i < wide; ++i)
        hi.push_back(i);
      Value loV = vector::ShuffleOp::create(b, loc, halfTy, acc, acc, lo);
      Value hiV = vector::ShuffleOp::create(b, loc, halfTy, acc, acc, hi);
      acc = emitFold(b, loc, fold, loV, hiV);
    }

    // Butterfly: rotate the register right by half the remaining span and
    // fold; after log2(128/foldElemBytes) steps every lane holds the row
    // value. The span bottoms out at the fold's element size -- the register
    // lanes are fold-typed, so a byte-granularity rotation below that would
    // misalign them (an f16 row folded in f32 stops at 4 B, not 2 B).
    for (int64_t bytes = kHvxVectorBytes / 2; bytes >= foldElemBytes;
         bytes /= 2) {
      Value rotated = hvx::VrorOp::create(b, loc, vecTy, acc,
                                          b.getI32IntegerAttr(bytes));
      acc = emitFold(b, loc, fold, rotated, acc);
    }

    // Fold in the reduce's own init (the outs element) and keep the scalar
    // contract: one reduced value per row, stored where the reduce wrote it.
    // A scalar outs (rank 0, single-row reduce) takes no index; a rank-1
    // outs of one element is addressed by r (= 0).
    SmallVector<Value> dstIdx;
    if (dstTy.getRank() > 0)
      dstIdx.push_back(r);
    Value init = memref::LoadOp::create(b, loc, dst, dstIdx);
    Value initVec = vector::BroadcastOp::create(b, loc, vecTy, init);
    acc = emitFold(b, loc, fold, acc, initVec);
    Value scalar = vector::ExtractOp::create(b, loc, acc, 0);
    memref::StoreOp::create(b, loc, scalar, dst, dstIdx);
    scf::YieldOp::create(b, loc);

    op.erase();
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createVectorRowReducePass() {
  return std::make_unique<VectorRowReducePass>();
}
