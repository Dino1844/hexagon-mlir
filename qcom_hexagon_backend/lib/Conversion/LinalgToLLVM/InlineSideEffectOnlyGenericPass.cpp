//===- InlineSideEffectOnlyGenericPass.cpp --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass removes `linalg.generic` ops that no lowering can accept: no
// operands, no results, no loop dimensions. Such an op defines no values, so
// every consumer of linalg skips it -- upstream `ConvertLinalgToLoops`'
// rewrite pattern requires `hasPureBufferSemantics()` (at least one memref
// operand) and simply does not match, with no diagnostic -- and the op
// survives into LLVM translation, which has no interface for it and fails
// with "missing LLVMTranslationDialectInterface".
//
// One thing in this tree produces that form, and not on purpose. The
// unstructured-to-memref lowering of a Triton store whose pointer has no
// make_range-derived axis -- the shape of a keepdim reduce whose [1] result
// is stored at a scalar offset -- emits a store generic with two inputs and
// no outputs. Elementwise fusion then folds the two `linalg.fill`s feeding
// it into the region and erases the operands the fold made unused (the
// fusion group's `FoldFillWithGenericOp` plus
// `EraseUnusedOperandsAndResultsPattern`), which leaves the zero-operand
// shell below. Its region body is the store itself -- already lowered by
// the time this pass runs -- so the only thing wrong with the op is that it
// still exists: without the wrapper the pipeline compiles clean (the same
// IR with fusion off has zero residual linalg).
//
// The repair is structural, not a gate. The match condition is a fact about
// the op (nothing crosses the region boundary, there are no loops to run),
// not about shapes, operators or kernel names. Because the form has no
// operands, no results and no iterators, the region executes exactly once
// at the generic's position, and every value its body uses is defined above
// the generic (MLIR dominance), so moving each body op into the parent block
// just above the generic preserves dominance, order and semantics for any
// body whatsoever. If fusion stops producing the shell, or upstream starts
// accepting zero-operand generics, this pass becomes a no-op.
//
// Example transformation:
//   Before:
//     linalg.generic {indexing_maps = [], iterator_types = []} {
//       %i = arith.index_cast %pid : i32 to index
//       memref.store %v, %p[%i] : memref<?xf32>
//       linalg.yield
//     }
//   After:
//     %i = arith.index_cast %pid : i32 to index
//     memref.store %v, %p[%i] : memref<?xf32>
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "inline-side-effect-only-generic"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

#define GEN_PASS_DEF_INLINESIDEEFFECTONLYGENERIC
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

/// A generic with no operands, no results and no loop dimensions: its region
/// runs once at the op's position and cannot exchange a value with the
/// enclosing scope. This is what a store generic becomes after elementwise
/// fusion folds its fills into the region and erases its operands.
static bool isSideEffectOnlyShell(linalg::GenericOp gop) {
  return gop.getNumDpsInputs() == 0 && gop.getNumDpsInits() == 0 &&
         gop.getNumResults() == 0 && gop.getIteratorTypesArray().empty() &&
         !gop.getRegion().empty();
}

struct InlineSideEffectOnlyGenericPass
    : public ::impl::InlineSideEffectOnlyGenericBase<
          InlineSideEffectOnlyGenericPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    DBG("Running InlineSideEffectOnlyGeneric pass on function: "
        << funcOp.getName());

    SmallVector<linalg::GenericOp> shells;
    funcOp.walk([&](linalg::GenericOp gop) {
      if (isSideEffectOnlyShell(gop))
        shells.push_back(gop);
    });

    for (linalg::GenericOp gop : shells) {
      // No operands means the region block takes no arguments, and no
      // results means the terminator yields nothing, so every op in the
      // region except the terminator is body to inline.
      Block *body = &gop.getRegion().front();
      SmallVector<Operation *> toMove;
      for (Operation &op : body->getOperations())
        if (!op.hasTrait<OpTrait::IsTerminator>())
          toMove.push_back(&op);

      // Each op lands just above the generic, so moving them in body order
      // preserves that order in the parent block.
      for (Operation *op : toMove)
        op->moveBefore(gop);

      DBG("Inlined " << toMove.size() << " op(s), erased the shell");
      gop->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createInlineSideEffectOnlyGenericPass() {
  return std::make_unique<InlineSideEffectOnlyGenericPass>();
}
