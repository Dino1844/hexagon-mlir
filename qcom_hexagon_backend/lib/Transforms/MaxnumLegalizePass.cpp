//===- MaxnumLegalizePass.cpp - vector fp maxnum -> vmax + NaN fixup -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The Hexagon HVX backend marks only ISD::FMAXIMUMNUM Legal (V6_vmax_sf/hf,
// HexagonISelLoweringHVX.cpp:151); FMAXNUM -- what arith.maxnumf lowers to --
// has no HVX action, so a vector maxnumf is expanded per lane: stack
// round-trip + scalar load + sfcmp/mux + vinsert/valign rebuild, ~130
// instructions per <32xf32> (or fmaxf libcalls). This pass rewrites every
// vector f16/f32 maxnumf into the vmax + explicit NaN-fixup sequence
// described in the pass tablegen record, restoring strict maxnum semantics
// on top of the NaN-agnostic V6_vmax.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include "hexagon/Transforms/Transforms.h"

using namespace mlir;

#define GEN_PASS_DEF_HVXMAXNUMLEGALIZE
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Is this maxnumf one the HVX backend cannot lower natively: a vector of a
/// float type with native HVX fp-max support (f16/f32). Scalars are left
/// alone (their lowering is at worst one fmaxf libcall), and so is every
/// other element type.
static bool isHvxVectorMaxnum(arith::MaxNumFOp op) {
  auto vecTy = dyn_cast<VectorType>(op.getType());
  if (!vecTy)
    return false;
  Type elemTy = vecTy.getElementType();
  return elemTy.isF32() || elemTy.isF16();
}

struct HvxMaxnumLegalizePass
    : public ::impl::HvxMaxnumLegalizeBase<HvxMaxnumLegalizePass> {
public:
  explicit HvxMaxnumLegalizePass() = default;

  // Bisection knob (set by the factory, not a pass option): false = bare
  // maximumf without the NaN fixup. NOT semantically safe in general (drops
  // strict maxnum semantics); only for isolating device-side failures.
  bool emitFixup = true;
  // Site selector for bisection: <0 = rewrite all candidates (normal);
  // >=0 = rewrite only the first N walk-ordered sites (device crash triage,
  // rowmax plan §10).
  int rewriteLimit = -1;
  // Additionally skip this many walk-ordered sites from the start.
  int skipFirst = 0;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    // Collect first: rewriting invalidates the walk.
    SmallVector<arith::MaxNumFOp> candidates;
    fn.walk([&](arith::MaxNumFOp op) {
      if (isHvxVectorMaxnum(op))
        candidates.push_back(op);
    });
    // Always-on observability for the success path (this pass used to dump to
    // raw stderr): a remark both reports the bisection state and doubles as
    // the wiring assertion of
    // test/Conversion/LinalgToLLVM/maxnum-legalize-pipeline.mlir.
    fn->emitRemark() << "hvx-maxnum-legalize: candidates=" << candidates.size()
                     << " skip=" << skipFirst << " limit=" << rewriteLimit;
    size_t idx = 0;
    for (arith::MaxNumFOp op : candidates) {
      if (idx++ < static_cast<size_t>(skipFirst > 0 ? skipFirst : 0))
        continue;
      if (rewriteLimit >= 0 &&
          static_cast<int>(idx - (skipFirst > 0 ? skipFirst : 0)) >
              rewriteLimit)
        break;
      rewrite(op);
    }
  }

private:
  void rewrite(arith::MaxNumFOp op) const {
    Location loc = op.getLoc();
    OpBuilder b(op);
    Value a = op.getLhs(), c = op.getRhs();
    Type ty = op.getType();

    // vmax: maximumf lowers to llvm.maximum -> FMAXIMUMNUM, which the HVX
    // ISel selects into V6_vmax. The original op's fastmath flags carry
    // over: they are assertions about the inputs, not a semantic change.
    Value vmax = arith::MaximumFOp::create(b, loc, a, c, op.getFastmath());
    if (!emitFixup) {
      op.replaceAllUsesWith(vmax);
      op.erase();
      return;
    }

    // Strict maxnum NaN semantics (a NaN operand yields the other operand):
    // oeq(x, x) is false exactly when x is NaN, so each select substitutes
    // the other operand whenever vmax's NaN lane behavior could leak.
    // These compares get no fastmath flags -- under nnan the fixup would be
    // provably dead and deletable, but that decision belongs to a pass that
    // can see the whole dataflow, not to this rewrite.
    Value aIsNum = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ,
                                         a, a);
    Value cIsNum = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ,
                                         c, c);
    Value fixup = arith::SelectOp::create(b, loc, cIsNum, vmax, a);
    Value result = arith::SelectOp::create(b, loc, aIsNum, fixup, c);

    op.replaceAllUsesWith(result);
    op.erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createHvxMaxnumLegalizePass(bool emitFixup, int rewriteLimit,
                                            int skipFirst) {
  auto pass = std::make_unique<HvxMaxnumLegalizePass>();
  pass->emitFixup = emitFixup;
  pass->rewriteLimit = rewriteLimit;
  pass->skipFirst = skipFirst;
  return pass;
}
