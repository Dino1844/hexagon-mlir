//===--- VTCMTilingPass.cpp - implement a basic tiling for VTCM pass  ----====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This pass tiles a (potentially fused) linalg.generic-on-tensors.
// It assumes that the tensor ins/outs are in DDR address space,
// the tiles need to be brought into VTCM, and the result stored back into DDR.
// Small tensors are completely prefetched (copied) to VTCM before the tiling.
//
// The tiling-option is external and is queried by this pass.
// This pass tiles and rewrites the IR so that after bufferization:
//    - space is allocated on VTCM.
//    - slices (or entire small tensors) are copied from DDR to VTCM.
//    - slices in VTCM are passed to the tiled generic.
//    - results copied back to DDR (by bufferizer or explicitly for small
//      tensors).
// Example:
// ```
//    %.. = linalg.generic {
//            indexing_maps = [#map, ...], iterator_types = [..]}
//            ins(%... : tensor<1024x256xf32>, ...)
//            outs(%... : tensor<1024x256xf32>, ...) { ... }
// ```
// will be re-written as:
// ```
//   % = scf.for %.. = %c0 to %c1024 step %c32_0 iter_args(%arg4 = %..)
//          -> (tensor<1024x256xf32>) {
//     % = scf.for %.. = %c0_1 to %c256 step %c64_2 iter_args(%arg6 = %..)
//            -> (tensor<1024x256xf32>) {
//          ...
//          %vtctm_tensor = bufferization.alloc_tensor()
//              copy(%ddr_tensor_slice) {memory_space = 1 : i64}
//              : tensor<32x64xf32>
//          ...
//          %.. = linalg.generic
//                 {indexing_maps = [#map, ...], iterator_types = [..]}
//                 ins(%vtcm_tensor, ... : tensor<32x64xf32>, ...)
//                 outs(%vtcm_tensor2 : tensor<32x64xf32>) { ... }
// ```
// which after bufferization, deallocation and hoisting will be:
// ```
//    %vtcm_alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32, 1>
//    scf.for %.. = %c0 to %c1024 step %c32 {
//      scf.for %.. = %c0 to %c256 step %c64 {
//        ...
//         memref.copy %ddr_subview, %vtcm_alloc
//               : memref<32x64xf32, strided<[256, 1], offset: ?>>
//                 to memref<32x64xf32, 1>
//         ...
//         %tile_result_on_vtcm  = linalg.generic {indexing_maps = }
//                  ins(%vtcm_alloc, ... : memref<32x64xf32, 1>, ...)
//                  outs(%vtcm_alloc2 : memref<32x64xf32, 1>) {  ... }
//         ...
//         memref.copy %tile_result_on_vtcm, %full_result_ddr
//                  : memref<32x64xf32, 1>
//                    to memref<32x64xf32, strided<[256, 1], offset: ?>>
//      }
//    }
//    memref.dealloc %vtcm_alloc : memref<32x64xf32, 1>
//```
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Transforms/OptionsParsing.h"

#define DEBUG_TYPE "vtcm-tiling"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_VTCMTILING
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

struct VTCMTilingPass : public ::impl::VTCMTilingBase<VTCMTilingPass> {
  explicit VTCMTilingPass(const VTCMTilingOptions &options) : Base(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  void runOnOperation() override;
};

// Annotate tiled IR for later passes (e.g. double buffering)
void annotateOp(Operation *op) {
  op->setAttr("all_parallel", mlir::UnitAttr::get(op->getContext()));
  op->setAttr("tiled_generic", mlir::UnitAttr::get(op->getContext()));
}

void annotateTiledLoop(linalg::GenericOp op, linalg::TiledLinalgOp tiledOp) {
  MLIRContext *ctx = op.getContext();
  bool allParallel = true;
  for (auto iterType : op.getIteratorTypesArray())
    if (iterType != utils::IteratorType::parallel)
      allParallel = false;

  if (allParallel && tiledOp.loops.size() > 0 &&
      isa<scf::ForOp>(tiledOp.loops[0]))
    annotateOp(tiledOp.loops[0]);
}

/// Takes a 'rankedTensor' value and returns a tensor on the specified memory
/// space.
Value copyToMemorySpace(IRRewriter &rewriter, Value rankedTensor, Location loc,
                        unsigned memorySpace) {
  // As 'copy' is specified, dynamicSizes are inferred and not supplied.
  SmallVector<Value> dynamicSizes;
  auto sourceType = ::llvm::cast<RankedTensorType>(rankedTensor.getType());

  rewriter.setInsertionPointAfterValue(rankedTensor);
  auto copyOp = bufferization::AllocTensorOp::create(
      rewriter, loc, sourceType, dynamicSizes, rankedTensor);
  copyOp.setMemorySpaceAttr(
      rewriter.getIntegerAttr(rewriter.getI64Type(), memorySpace));
  return copyOp.getResult();
}

/// Takes a 'rankedTensor' value and returns a tensor on DDR.
inline Value copyToDDR(IRRewriter &rewriter, Value rankedTensor, Location loc) {
  return copyToMemorySpace(rewriter, rankedTensor, loc,
                           DEFAULT_DDR_ADDRESS_SPACE);
}

/// Takes a 'rankedTensor' value and returns a tensor on VTCM.
inline Value copyToVTCM(IRRewriter &rewriter, Value rankedTensor,
                        Location loc) {
  return copyToMemorySpace(rewriter, rankedTensor, loc, VTCM_ADDRESS_SPACE);
}

/// Return a cloned generic with `newIns` and `newOuts`.
GenericOp getGenericWithNewOperands(IRRewriter &rewriter, GenericOp op,
                                    SmallVector<Value> &newIns,
                                    SmallVector<Value> &newOuts) {
  rewriter.setInsertionPointAfter(op);
  GenericOp newOp = GenericOp::create(
      rewriter, op.getLoc(), op.getResultTypes(), newIns, newOuts,
      op.getIndexingMapsArray(), op.getIteratorTypesArray(),
      /*bodyBuild=*/nullptr);
  rewriter.inlineRegionBefore(op->getRegion(0), newOp.getRegion(),
                              newOp.getRegion().begin());
  return newOp;
}

/// Accumulate the linear coefficients of `e` w.r.t. the iteration dims into
/// `coeffs` (indexed by dim position), scaled by `scale`. Returns false when
/// the expression is not a linear function of the dims (a product of two
/// dims, div/mod, a symbol): injectivity is then not provable here.
static bool collectLinearCoeffs(AffineExpr e, int64_t scale,
                                SmallVectorImpl<int64_t> &coeffs) {
  if (auto dim = dyn_cast<AffineDimExpr>(e)) {
    coeffs[dim.getPosition()] += scale;
    return true;
  }
  if (isa<AffineConstantExpr>(e))
    return true;
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
    switch (bin.getKind()) {
    case AffineExprKind::Add:
      return collectLinearCoeffs(bin.getLHS(), scale, coeffs) &&
             collectLinearCoeffs(bin.getRHS(), scale, coeffs);
    case AffineExprKind::Mul:
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        return collectLinearCoeffs(bin.getLHS(), scale * c.getValue(), coeffs);
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS()))
        return collectLinearCoeffs(bin.getRHS(), scale * c.getValue(), coeffs);
      return false;
    default:
      return false; // Mod/FloorDiv/CeilDiv: not linear.
    }
  }
  return false; // SymbolId or anything else.
}

/// Rank of a small integer matrix via fraction-free Gaussian elimination.
/// The matrices here are (num loops x operand rank); both are tiny and their
/// entries are unit-scale, so int64 cannot overflow in practice.
static unsigned rankOf(SmallVector<SmallVector<int64_t>> m) {
  const unsigned rows = m.size();
  const unsigned cols = rows ? m[0].size() : 0;
  unsigned rank = 0;
  for (unsigned col = 0; col != cols && rank != rows; ++col) {
    unsigned pivot = rank;
    while (pivot != rows && m[pivot][col] == 0)
      ++pivot;
    if (pivot == rows)
      continue;
    std::swap(m[rank], m[pivot]);
    for (unsigned r = rank + 1; r != rows; ++r) {
      if (m[r][col] == 0)
        continue;
      int64_t a = m[rank][col], b = m[r][col];
      for (unsigned c = 0; c != cols; ++c)
        m[r][c] = m[r][c] * a - m[rank][c] * b;
    }
    ++rank;
  }
  return rank;
}

/// Is `map` injective over its whole domain, hence over any iteration box?
/// True when the map's linear part has full column rank: distinct iteration
/// points then address distinct tensor elements, i.e. every element of the
/// operand is read at most once. This is a sufficient, shape-free condition;
/// a map that is injective only on a restricted box (a unit-trip-count
/// dimension collapsed by the map) conservatively returns false.
static bool isSingleReadIndexingMap(AffineMap map) {
  const unsigned numDims = map.getNumDims();
  if (numDims == 0)
    return true;
  if (map.getNumResults() < numDims)
    return false; // iteration dims collapse: not injective.
  SmallVector<SmallVector<int64_t>> lin(
      numDims, SmallVector<int64_t>(map.getNumResults(), 0));
  for (auto [j, res] : llvm::enumerate(map.getResults())) {
    SmallVector<int64_t> col(numDims, 0);
    if (!collectLinearCoeffs(res, 1, col))
      return false;
    for (unsigned i = 0; i != numDims; ++i)
      lin[i][j] = col[i];
  }
  return rankOf(std::move(lin)) == numDims;
}

/// Per-operand staging decision for a generic. Staging a tensor through VTCM
/// only pays off when the data is *reused* (or bulk-prefetched), so:
///  - an input whose indexing map is injective over the iteration space is
///    read at most once and stays in DDR;
///  - an out whose region never reads the init block arg is write-only: its
///    old value cannot survive into the result, so the destination is written
///    directly -- no init copy, no VTCM slot, no copy-back;
///  - everything else (broadcast or complex maps, read-modify-write outs)
///    keeps the staged behavior.
/// The decision is orthogonal to the tiling: whether a tile covers a whole
/// operand (the `prefetch` set) only picks whole-tensor vs per-tile staging
/// for the operands that are staged at all.
static void computeOperandStaging(linalg::GenericOp op,
                                  SmallVector<bool> &stage) {
  stage.assign(op.getNumOperands(), true);
  for (OpOperand &opOperand : op->getOpOperands()) {
    unsigned idx = opOperand.getOperandNumber();
    if (op.isDpsInit(&opOperand)) {
      if (op.getMatchingBlockArgument(&opOperand).use_empty())
        stage[idx] = false;
    } else {
      if (isSingleReadIndexingMap(op.getMatchingIndexingMap(&opOperand)))
        stage[idx] = false;
    }
  }
}

/// Return a linalg generic where `prefetch` tensors are VTCM copies
/// of data on DDR. Other operands are as in original. Operands that are not
/// staged (`stage` false) keep their original (DDR) buffer.
GenericOp replaceGenericWithPrefetchedOperands(IRRewriter &rewriter,
                                               GenericOp op,
                                               SmallVector<bool> prefetch,
                                               SmallVector<bool> stage) {
  SmallVector<Value> newIns;
  SmallVector<Value> newOuts;

  for (OpOperand &opOperand : op->getOpOperands()) {
    auto idx = opOperand.getOperandNumber();
    Value globalTensor = op->getOperand(idx);
    Value newTensor = globalTensor;

    if (prefetch[idx] && stage[idx])
      newTensor = copyToVTCM(rewriter, globalTensor, op.getLoc());
    op.isDpsInit(&opOperand) ? newOuts.push_back(newTensor)
                             : newIns.push_back(newTensor);
  }
  auto newOp = getGenericWithNewOperands(rewriter, op, newIns, newOuts);
  rewriter.replaceOp(op, newOp->getResults());
  return newOp;
}

/// Replace tiled generic that operates on slices from DDR,
/// to a new generic that operates on copies on VTCM. Slices of operands that
/// are not staged keep operating on the DDR slices directly.
LogicalResult replaceTiledGenericWithVTCMSlices(IRRewriter &rewriter,
                                                GenericOp top,
                                                SmallVector<bool> prefetch,
                                                SmallVector<bool> stage) {
  SmallVector<Value> newIns, newOuts;
  for (OpOperand &opOperand : top->getOpOperands()) {
    auto idx = opOperand.getOperandNumber();
    Value globalTensor = top->getOperand(idx);
    Value newTensor = globalTensor;

    if (!prefetch[idx] && stage[idx]) {
      if (!globalTensor.template getDefiningOp<tensor::ExtractSliceOp>())
        return failure();
      newTensor = copyToVTCM(rewriter, globalTensor, top.getLoc());
    }
    top.isDpsInit(&opOperand) ? newOuts.push_back(newTensor)
                              : newIns.push_back(newTensor);
  }
  GenericOp newOp = getGenericWithNewOperands(rewriter, top, newIns, newOuts);
  rewriter.replaceOp(top, newOp->getResults());
  return success();
}

/// Copying the results corresponding to the operands to be "prefetched" for a
/// linalg op to DDR and replacing the uses to the copied tensor on DDR.
/// Results that were never staged were computed directly on their DDR
/// destination and need no copy-back.
void copyResultsToDDR(IRRewriter &rewriter, GenericOp op,
                      SmallVector<bool> prefetch,
                      SmallVector<bool> stage) {
  for (int idx = 0; idx < op.getNumDpsInits(); ++idx) {
    auto operandIdx = op.getNumDpsInputs() + idx;
    if (prefetch[operandIdx] && stage[operandIdx]) {
      Value resultTensor = op.getResult(idx);
      // A crouton result read by `hmx.matmul` must keep its VTCM buffer: the
      // engine reads a crouton, and the DDR copy would hand `hmx.mma` a space-0
      // buffer, which the op verifier rejects.
      if (llvm::any_of(resultTensor.getUsers(), [](Operation *user) {
            return isa<hmx::MatmulOp>(user);
          }))
        continue;
      auto newTensor = copyToDDR(rewriter, resultTensor, op.getLoc());
      rewriter.replaceAllUsesExcept(resultTensor, newTensor,
                                    newTensor.getDefiningOp());
    }
  }
}

/// Does `v` (transitively) reach an `scf.yield`? Staging such a result makes the
/// enclosing loop's init_arg and yielded value disagree on the memory space, and
/// one-shot-bufferize rejects that outright:
///   'scf.for' op init_arg and yielded value bufferize to inconsistent memory spaces
/// A loop-carried tensor read elementwise in the body (a state vector such as the
/// key sum of a chunked linear attention, or a running mean) hits exactly this.
static bool reachesYield(Value v, unsigned depth = 0) {
  if (depth > 8)
    return false;
  for (Operation *user : v.getUsers()) {
    if (isa<scf::YieldOp>(user))
      return true;
    if (user->getBlock() != v.getParentBlock())
      continue; // only follow within the same region
    for (Value r : user->getResults())
      if (reachesYield(r, depth + 1))
        return true;
  }
  return false;
}

/// A generic that accumulates in place into a loop-carried block argument.
static bool accumulatesIntoLoopCarry(linalg::GenericOp op) {
  for (Value out : op.getDpsInits())
    if (auto arg = dyn_cast<BlockArgument>(out))
      if (isa_and_nonnull<scf::ForOp>(arg.getOwner()->getParentOp()))
        return true;
  return false;
}

void VTCMTilingPass::runOnOperation() {
  auto userProvidedTileSizes = parseTileSizes(tileSizes);
  auto funcOp = getOperation();

  funcOp.walk([&](linalg::GenericOp op) {
    // Staging a tile through VTCM only pays off when the tile is *reused*. A
    // generic whose iteration space is all-parallel and whose operands are each
    // read exactly once (identity maps) streams straight through DDR, and putting
    // it through VTCM costs an extra round trip plus allocation churn in the
    // runtime pool. Measured on device: 34x on a plain 131072-element add
    // (4320 -> 128 us) and 6.9x on silu (773 -> 112 us).
    //
    // The same reuse principle also applies per operand (see
    // computeOperandStaging): single-read inputs (injective indexing maps)
    // and write-only outs (regions that never read the init block arg) stay
    // on their DDR buffers; only reused inputs and read-modify-write outs pay
    // for a VTCM round trip. The streaming skip below stays whole: it is the
    // all-operands-at-once special case and also covers the outs.
    const bool streaming =
        llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
          return t == utils::IteratorType::parallel;
        }) &&
        llvm::all_of(op.getIndexingMapsArray(),
                     [](AffineMap m) { return m.isIdentity(); });
    // Staging a generic whose result feeds a loop's yielded value (or which
    // accumulates into a loop-carried argument) leaves the loop's init_arg in DDR
    // and the yielded value in VTCM, which bufferization rejects. Skipping it only
    // costs the staging, never correctness, and it unblocks every state-vector
    // kernel (chunked linear attention, running statistics).
    const bool loopCarry =
        llvm::any_of(op->getResults(),
                     [](Value r) { return reachesYield(r); }) ||
        accumulatesIntoLoopCarry(op);
    if (streaming || loopCarry)
      return WalkResult::advance();

    IRRewriter rewriter(op.getContext());
    SmallVector<bool> stage;
    computeOperandStaging(op, stage);
    SmallVector<bool> prefetch(op.getNumOperands(), false);
    FailureOr<linalg::LinalgTilingOptions> vtcmTilingOptions =
        getVTCMTilingOptions(op, userProvidedTileSizes, prefetch, vtcmBudget);
    if (failed(vtcmTilingOptions))
      return WalkResult::advance();

    // Replace with a new generic where operands which fit completely
    // into VTCM (prefetch 'set') are copied to VTCM before tiling, then copy
    // the corresponding results back to DDR.
    linalg::GenericOp prefetchOp =
        replaceGenericWithPrefetchedOperands(rewriter, op, prefetch, stage);
    copyResultsToDDR(rewriter, prefetchOp, prefetch, stage);

    rewriter.setInsertionPointAfter(prefetchOp);
    FailureOr<linalg::TiledLinalgOp> tiledOp =
        linalg::tileLinalgOp(rewriter, prefetchOp, *vtcmTilingOptions);
    if (failed(tiledOp))
      return WalkResult::advance();

    // annotate generated loop for reference.
    annotateTiledLoop(prefetchOp, *tiledOp);

    auto top = llvm::dyn_cast<GenericOp>(tiledOp->op.getOperation());
    if (failed(replaceTiledGenericWithVTCMSlices(rewriter, top, prefetch,
                                                 stage)))
      return WalkResult::advance();

    rewriter.replaceOp(prefetchOp, tiledOp->tensorResults);
    return WalkResult::advance();
  });
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createVTCMTilingPass(const VTCMTilingOptions &options) {
  return std::make_unique<VTCMTilingPass>(options);
}
