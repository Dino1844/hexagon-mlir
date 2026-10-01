//===- RowReduceGroupStorePass.cpp - row reduction stays in the vector domain -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// `HexagonVectorLowering` turns a vectorized 2-D last-dim row reduction into a
// per-row loop nest that ends in
//
//   %v   = vector.transfer_read <one 128 B chunk of the row>
//   %acc = tensor.extract <the row's current value from a tensor<rows x T>>
//   %r   = vector.reduction <maxnumf>, %v, %acc : vector<32xf32> into f32
//   %out = tensor.insert %r into <rank-0 slice of that same tensor<rows x T>>
//
// which is already the vector domain right up to its last two ops. Those two
// are what cost: Hexagon HVX has no "vector lane 0 to GPR" instruction, so an
// `f32` reduction result can only reach a scalar through one 128-byte vector
// stack write plus a 4-byte stack read back. Per row, per 128-byte chunk, in a
// loop-carried dependency chain.
//
// This pass moves the boundary. The row loop steps by one whole HVX vector of
// rows instead of one row; the group's running values stay in a
// `vector<lanes x T>` loop-carried value; each row's horizontal butterfly
// (`hvx.vror` + `maxnumf`, llama.cpp hvx-reduce.h style) stays a vector that is
// never extracted; and an `arith.cmpi eq` against a constant lane-index vector
// plus `arith.select` puts the all-lanes-equal result into the group's lane. One
// `vector.transfer_write` then retires `lanes` rows.
//
// The group carrier is a vector, not a `tensor<lanes x T>`, on purpose: the
// one-shot bufferizer leaves a vector-typed `scf.for` iter_arg completely alone
// (measured on this pin), whereas a tensor-typed one is materialized into a
// temporary memref, which would reintroduce exactly the per-row 128-byte stack
// round trip this pass exists to delete. The loop that replaces the row loop
// still carries the original `tensor<rows x T>`, so everything downstream --
// the enclosing `scf.forall`'s `parallel_insert_slice`, `FormAsyncThreads` --
// sees the same shape it saw before.
//
// The group's previous value is folded in *before* the select, elementwise:
//
//   %merged = maxnumf %butterfly, %group     ; lane k is garbage for k != j
//   %sel    = select (laneIdx == %j), %merged, %group
//
// so lane j receives `max(rowmax_j, group[j])` and every other lane keeps its
// previous value untouched. That is the scalar nest's running update
// (`group[j] = max(group[j], max over the row's chunks)`) with no scalar
// anywhere in it. The operand order is load-bearing: `merged` true, `group`
// false. Swapped, lane j keeps its old value and the kernel computes a stale row
// max -- silently, and with no other symptom.
//
// Only `maxnumf` is rewritten. `addf` is the same shape, but folding the chunks
// before the butterfly reassociates the sum -- `max(max(reduce(c0), m),
// reduce(c1))` becomes `max(m, max(c0, c1))`. That is inside the pipeline's
// `reassoc` contract, and it is still a numerical change nobody asked for, so
// the row *sum* stays on the scalar path until a measurement asks for it.
//
// The folds carry the reduction's own fastmath merged with `nnan`. Two separate
// facts, and both are worth keeping straight:
//
//  * Why `nnan` at all. A bare `llvm.maxnum` with no `nnan` is not selectable
//    into a single `vmax`: on the probe in
//    docs/hmx/row-reduce-vector-domain-design-2026-09-30.md it came out as
//    187 `valign` and 0 `vmax`. With `nnan` the pinned `llc` emits 1 `vmax` and
//    0 `valign`, and so does the SDK compiler given the whole `fast` flag.
//
//  * Why only `nnan`, never `fast`. `ninf` would be a lie -- the running values
//    are initialised to `-inf` -- and `reassoc` / `contract` / `afn` are
//    profitability claims this pass has no business making.
//
// What this buys and what it does not, stated plainly because an earlier version
// of this comment got it backwards: `HexagonAddFastMath` runs two passes later on
// the default path and *overwrites* the attribute (AddFastMathPass.cpp:71 is
// `op->setAttr`, an overwrite, not an OR), so on that path this pass's careful
// "never `ninf`" is undone a moment later. With `add-fastmath=false` this pass is
// the only thing that stamps these ops, and it introduces an `nnan` promise the
// original `vector.reduction` did not make. That is a real, deliberate trade --
// the butterfly is 5 instructions with `nnan` and a valign chain without it --
// and it is one more reason the knob is off by default.
//
// Scope of the gates, stated so nobody has to derive it:
//
//  * The rewrite re-derives the whole data flow from two values (the row carrier
//    and the source tile) rather than reusing any operand of the original chain.
//    So the matcher matches that chain as an *identity* -- this exact
//    `extract_slice`, this exact `reduction`, this exact `insert_slice`, each
//    addressed by the two induction variables -- and refuses anything it cannot
//    name. test/Conversion/LinalgToLLVM/row-reduce-group-store-gates.mlir has
//    one case per refusal. **None of them is reachable from today's
//    `HexagonVectorLowering` output**: the vectorizer emits one chunk read, one
//    reduction and one write-back per column step, all addressed by the two
//    induction variables with unit strides. The defect those cases cover was
//    under-specified gates, not a live miscompile. They are here so that stays
//    true if the vectorizer's output ever changes.
//
//  * G3 inspects the *immediately* enclosing `scf.forall`: its rank, its
//    lb/step/extent, and -- this is the part a bound-only check misses -- that
//    the tile's dim-0 offset is that forall's induction variable or a constant.
//    A more distant loop that shifts the base shows up in that last check:
//    `arith.addi %outerBase, %iv` is not the induction variable, so the pass
//    refuses rather than emitting a group store at an unaligned global row.
//
//  * In tensor form the `scf.forall` legitimately carries a result; what this
//    pass requires is that the row loop's result has exactly one use and that
//    use is a `tensor.parallel_insert_slice` into that forall's out-argument,
//    covering the tile's rows. The rank-1 / no-results / no-outputs shape that
//    `FormAsyncThreads` needs is a *post-bufferize* property and is left to it.
//
// Off by default (`enable-row-reduce-group-store`): the knob is the device A/B
// switch, not a correctness guard. Every refusal emits a remark naming the gate,
// the actual value and the expected one, because "did nothing" and "won" look
// identical in a performance number.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Dialect/Hvx/IR/HvxDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::hexagon;

// At global scope, like the rest of this family: the generated pass base lands
// in ::impl.
#define GEN_PASS_DEF_ROWREDUCEGROUPSTORE
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

// kHvxVectorBytes (the 128 B row-chunk width) comes from Common.h -- one
// contract shared with the vectorizer's skip gate and the vector-row-reduce
// butterfly. Not redefined here.

/// The name every remark is prefixed with. One string, so the lit tests and the
/// diagnostics cannot drift apart.
static constexpr llvm::StringLiteral kPassName = "row-reduce-group-store";

/// Render a Type for a diagnostic. `Twine` cannot hold a Type, and a gate message
/// that does not name the actual type is not actionable.
static std::string typeToString(Type t) {
  std::string out;
  llvm::raw_string_ostream os(out);
  t.print(os);
  return os.str();
}

/// Extract a constant integer from a Value. The same shape as the helper in
/// SCFLoopUnrollPass.cpp; kept local rather than shared because it is four lines.
static std::optional<int64_t> constantIntOf(Value value) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;
  return intAttr.getInt();
}

/// The same for an `OpFoldResult`, which is how an `scf.forall` spells its bounds
/// and steps. `OpFoldResult` is `llvm::PointerUnion<Attribute, Value>`, and
/// asking that union for a *derived* attribute type
/// (`dyn_cast_if_present<IntegerAttr>`) instantiates `llvm::FirstIndexOfType`
/// over the union's exact pack -- a hard compile error on this LLVM, because
/// `IntegerAttr` is not one of the two members. So ask for `Attribute` and
/// `Value`, the members, and cast from there. This PointerUnion also has no
/// `dyn_cast<T>()`, only `[[deprecated]] is<T>()` / `get<T>()`.
static std::optional<int64_t> constantIntOf(OpFoldResult ofr) {
  if (llvm::isa<Attribute>(ofr)) {
    auto intAttr = dyn_cast<IntegerAttr>(llvm::cast<Attribute>(ofr));
    if (!intAttr)
      return std::nullopt;
    return intAttr.getInt();
  }
  if (llvm::isa<Value>(ofr))
    return constantIntOf(llvm::cast<Value>(ofr));
  return std::nullopt;
}

/// One entry of an extract_slice/insert_slice offsets|sizes|strides list: a
/// static constant, or an SSA value.
struct SliceIndex {
  std::optional<int64_t> constant;
  Value value;
};

/// Join an op's static offsets|sizes|strides array with its dynamic operands.
///
/// This pin's `tensor::ExtractSliceOp` / `tensor::InsertSliceOp` /
/// `tensor::ParallelInsertSliceOp` expose the two halves separately --
/// `getStaticOffsets()` (ArrayRef<int64_t>, `kDynamic` in the SSA slots) and
/// `getOffsets()` (operands, *only* the dynamic ones) -- and have no
/// `getMixedOffsets()`. Reading `getOffsets()` alone therefore sees an empty
/// range on a fully static slice, which is exactly the kind of hole these gates
/// exist to close, so the halves are joined here once instead of at each site.
static SmallVector<SliceIndex>
sliceIndices(ArrayRef<int64_t> staticVals, OperandRange dynamicVals) {
  SmallVector<SliceIndex> out;
  out.reserve(staticVals.size());
  unsigned next = 0;
  for (int64_t v : staticVals) {
    if (v == ShapedType::kDynamic && next < dynamicVals.size())
      out.push_back({std::nullopt, Value(dynamicVals[next++])});
    else
      out.push_back({v, Value()});
  }
  return out;
}

static bool isAt(SliceIndex e, Value v) { return e.value && e.value == v; }
static bool isConst(SliceIndex e, int64_t k) {
  return !e.value && e.constant && *e.constant == k;
}

/// Bytes per lane, or nothing: only f32 and f16 have a lowering here, and a
/// reduction over anything else cannot be re-derived from a lane count.
static std::optional<int64_t> elemBytesOf(Type elemTy) {
  if (elemTy.isF32())
    return 4;
  if (elemTy.isF16())
    return 2;
  return std::nullopt;
}

/// Everything the rewrite needs once every gate has passed.
struct GroupStoreShape {
  Value rowCarrier;      // tensor<rows x T>, the row loop's init
  Value sourceTile;      // tensor<rows x cols x T>, the reduction's input tile
  int64_t rows = 0;      // dim 0 of rowCarrier
  int64_t cols = 0;      // dim 1 of sourceTile
  int64_t lanes = 0;     // kHvxVectorBytes / elemBytes
  int64_t elemBytes = 0; // 4 (f32) or 2 (f16)
  int64_t rowLower = 0;  // row loop lower bound, a multiple of `lanes`
  int64_t rowUpper = 0;  // row loop upper bound
};

/// The row loop's body must be exactly one column loop and the yield. Anything
/// else means the rewrite is not local, so the loop is not a candidate. This is
/// the *candidate filter*, run once in `runOnOperation`; `rewriteRowLoop` relies
/// on it rather than repeating it.
static bool isCandidateRowLoop(scf::ForOp rowLoop) {
  unsigned nested = 0;
  for (Operation &op : *rowLoop.getBody()) {
    if (isa<scf::ForOp>(op)) {
      ++nested;
      continue;
    }
    if (isa<scf::YieldOp>(op))
      continue;
    return false;
  }
  return nested == 1;
}

/// Every op the rewrite re-derives, held as an *identity* rather than a shape.
/// The rewrite does not reuse a single operand of the chain -- it rebuilds the
/// data flow from the row carrier and the source tile -- so anything it cannot
/// name is something it would silently drop. That is the whole reason this is a
/// chain match and not a type match.
struct ColumnChain {
  tensor::ExtractSliceOp chunkSlice;
  vector::TransferReadOp chunkRead;
  tensor::ExtractSliceOp accSlice;
  tensor::ExtractOp accExtract;
  vector::ReductionOp reduction;
  tensor::InsertOp insert;
  tensor::InsertSliceOp writeBack;
  arith::FastMathFlags fastmath = arith::FastMathFlags::none;
};

struct RowReduceGroupStorePass
    : public ::impl::RowReduceGroupStoreBase<RowReduceGroupStorePass> {
public:
  explicit RowReduceGroupStorePass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    // The rewrite emits arith / tensor / scf / vector ops and hvx.vror; without
    // this a kernel whose input never mentions the vector dialect would die on
    // "created with unregistered dialect".
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    tensor::TensorDialect, vector::VectorDialect,
                    hvx::HvxDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    // Collect first: rewriting invalidates the walk. A "candidate" is a row loop
    // whose shape could plausibly be one of these -- a row loop holding a column
    // loop holding a vector.reduction -- not one that passed every gate. So the
    // census says how many nests this pass looked at, and the remarks say which
    // of them it left alone.
    SmallVector<scf::ForOp> candidates;
    fn.walk([&](scf::ForOp forOp) {
      if (forOp.getNumRegionIterArgs() != 1)
        return;
      if (isCandidateRowLoop(forOp))
        candidates.push_back(forOp);
    });

    size_t rewritten = 0;
    for (scf::ForOp rowLoop : candidates)
      rewritten += rewriteRowLoop(rowLoop);

    // Always-on observability, mirroring hvx-maxnum-legalize: without it,
    // "the gate was too narrow" and "the kernel got faster" are the same
    // observation.
    //
    // The text is formatted once and used twice on purpose: the remark is what
    // `linalg-hexagon-opt` shows, and printCensusRemarkToStderr is what a Triton
    // compile shows (MLIR's engine drops remarks before any handler, and this
    // DiagnosticEngine has no threshold setter -- see Common.h). Two docs quote
    // this exact string, so it must not drift between the two sinks.
    std::string censusText =
        (kPassName + ": candidates=" + Twine(candidates.size()) +
         ", rewritten=" + Twine(rewritten))
            .str();
    fn.emitRemark() << censusText;
    printCensusRemarkToStderr(censusText);
  }

private:
  static Value emitFold(OpBuilder &b, Location loc, Value l, Value r,
                        arith::FastMathFlags fastmath) {
    return arith::MaxNumFOp::create(b, loc, l, r, fastmath).getResult();
  }

  /// Refuse and say why, in one place, so every gate's message has the same
  /// shape: pass name, what is wrong, what was expected, "leaving IR unchanged".
  /// Always returns true, so every call site is `if (reject(...)) return 0;`
  /// and there is no way to emit a refusal and then fall through it.
  template <typename OpT>
  static bool reject(OpT op, const Twine &what) {
    std::string text =
        (kPassName + ": " + what + "; leaving IR unchanged").str();
    op.emitRemark() << text;
    printCensusRemarkToStderr(text);
    return true;
  }

  // ======================= the column-loop op chain =============================
  //
  // The exact sequence the rewrite stands in for. Each check names an op, not a
  // shape; see the header for why. `chain` and `shape.sourceTile` / `shape.cols`
  // are filled in on success; on failure `reason` says which gate.

  static LogicalResult matchColumnChain(scf::ForOp colLoop, Value rowIV,
                                        Value colIV, Type elemTy, int64_t lanes,
                                        GroupStoreShape &shape,
                                        ColumnChain &chain, std::string &reason) {
    Block *body = colLoop.getBody();
    Value colCarrier = colLoop.getRegionIterArg(0);

    // ---- every op accounted for, and exactly one per role ---------------------
    // "Last one wins" is not good enough: a body with two reductions folds two
    // chunks and the rewrite would keep one and drop the other.
    SmallVector<vector::TransferReadOp> reads;
    SmallVector<vector::ReductionOp> reductions;
    SmallVector<tensor::ExtractOp> extracts;
    SmallVector<tensor::InsertOp> inserts;
    SmallVector<tensor::InsertSliceOp> insertSlices;
    SmallVector<tensor::ExtractSliceOp> slices;
    for (Operation &op : *body) {
      if (auto v = dyn_cast<vector::TransferReadOp>(op))
        reads.push_back(v);
      else if (auto v = dyn_cast<vector::ReductionOp>(op))
        reductions.push_back(v);
      else if (auto v = dyn_cast<tensor::ExtractOp>(op))
        extracts.push_back(v);
      else if (auto v = dyn_cast<tensor::InsertOp>(op))
        inserts.push_back(v);
      else if (auto v = dyn_cast<tensor::InsertSliceOp>(op))
        insertSlices.push_back(v);
      else if (auto v = dyn_cast<tensor::ExtractSliceOp>(op))
        slices.push_back(v);
      else if (&op != body->getTerminator()) {
        reason = (Twine("the column loop body contains a ") +
                  op.getName().getStringRef() +
                  "; every op in it must be part of the chunk-read / fold / "
                  "write-back chain this pass re-derives")
                     .str();
        return failure();
      }
    }
    auto exactlyOne = [&](size_t n, llvm::StringRef what) {
      if (n == 1)
        return true;
      reason = (Twine("the column loop body has ") + Twine(n) + " " + what +
                "; expected exactly 1, because the rewrite re-derives the chunk "
                "count from the tile width and cannot reproduce a body that folds "
                "more than one chunk per iteration")
                   .str();
      return false;
    };
    if (!exactlyOne(reads.size(), "vector.transfer_read") ||
        !exactlyOne(reductions.size(), "vector.reduction") ||
        !exactlyOne(extracts.size(), "tensor.extract") ||
        !exactlyOne(inserts.size(), "tensor.insert") ||
        !exactlyOne(insertSlices.size(), "tensor.insert_slice"))
      return failure();

    // ---- the chunk slice: indexed by (rowIV, colIV), size [1, lanes], stride 1 --
    tensor::ExtractSliceOp chunkSlice;
    for (tensor::ExtractSliceOp s : slices) {
      auto offs = sliceIndices(s.getStaticOffsets(), s.getOffsets());
      if (offs.size() != 2 || !isAt(offs[0], rowIV) || !isAt(offs[1], colIV))
        continue;
      if (chunkSlice) {
        reason = "the column loop body has two tensor.extract_slice ops indexed "
                 "[rowIV, colIV]; expected exactly 1";
        return failure();
      }
      chunkSlice = s;
    }
    if (!chunkSlice) {
      reason = "no tensor.extract_slice in the column loop body is indexed "
               "[rowIV, colIV]; expected the chunk read to be addressed by the "
               "two induction variables, because the rewrite re-derives the same "
               "element from them";
      return failure();
    }
    {
      auto sizes =
          sliceIndices(chunkSlice.getStaticSizes(), chunkSlice.getSizes());
      auto strides =
          sliceIndices(chunkSlice.getStaticStrides(), chunkSlice.getStrides());
      if (sizes.size() != 2 || !isConst(sizes[0], 1) || !isConst(sizes[1], lanes)) {
        reason = "the chunk slice is not sized [1, lanes] -- one row, one whole "
                 "HVX vector; the butterfly is only defined on a full register";
        return failure();
      }
      if (strides.size() != 2 || !isConst(strides[0], 1) ||
          !isConst(strides[1], 1)) {
        reason = "the chunk slice is not unit-strided [1, 1]; the rewrite reads it "
                 "with unit stride, so a strided chunk would silently change which "
                 "columns are folded";
        return failure();
      }
      auto chunkTy = dyn_cast<RankedTensorType>(chunkSlice.getType());
      if (!chunkTy || chunkTy.getRank() != 1 ||
          chunkTy.getElementType() != elemTy || chunkTy.getDimSize(0) != lanes) {
        reason = (Twine("the chunk slice is ") +
                  typeToString(chunkSlice.getType()) + "; expected tensor<" +
                  Twine(lanes) + " x " + typeToString(elemTy) + ">")
                     .str();
        return failure();
      }
      auto srcTy = dyn_cast<RankedTensorType>(chunkSlice.getSource().getType());
      if (!srcTy || srcTy.getRank() != 2 || !srcTy.hasStaticShape() ||
          srcTy.getElementType() != elemTy) {
        reason = "the chunk is not read from a static rank-2 tensor of the row "
                 "carrier's element type";
        return failure();
      }
      shape.sourceTile = chunkSlice.getSource();
      shape.cols = srcTy.getDimSize(1);
    }
    if (!chunkSlice.getResult().hasOneUse() ||
        chunkSlice.getResult().use_begin()->getOwner() !=
            reads[0].getOperation()) {
      reason = "the chunk slice is not read by exactly the vector.transfer_read "
               "this pass re-derives; a second reader would be dropped";
      return failure();
    }
    chain.chunkSlice = chunkSlice;
    chain.chunkRead = reads[0];

    // ---- the accumulator slice: the column loop's own carrier, at rowIV ----------
    auto extract = extracts[0];
    if (!extract.getIndices().empty()) {
      reason = "the reduction's accumulator is a tensor.extract with indices; "
               "expected a bare extract of a rank-0 slice";
      return failure();
    }
    auto accSlice = extract.getTensor().getDefiningOp<tensor::ExtractSliceOp>();
    if (!accSlice) {
      reason = "the reduction's accumulator is not a tensor.extract of a slice; "
               "expected the running value this pass folds in";
      return failure();
    }
    {
      auto offs = sliceIndices(accSlice.getStaticOffsets(), accSlice.getOffsets());
      if (offs.size() != 1 || !isAt(offs[0], rowIV)) {
        reason = "the accumulator slice is not indexed by rowIV alone; expected "
                 "the running value of the row this iteration owns";
        return failure();
      }
      if (accSlice.getSource() != colCarrier) {
        reason = "the accumulator slice does not come from the column loop's own "
                 "loop-carried carrier; the rewrite folds in the carrier's value, "
                 "which is a different number";
        return failure();
      }
      if (accSlice.getType().getRank() != 0 ||
          accSlice.getType().getElementType() != elemTy) {
        reason = (Twine("the accumulator slice is ") +
                  typeToString(accSlice.getType()) +
                  "; expected a rank-0 slice of " + typeToString(elemTy))
                     .str();
        return failure();
      }
    }
    chain.accSlice = accSlice;
    chain.accExtract = extract;

    // ---- the reduction ------------------------------------------------------------
    auto red = reductions[0];
    if (red.getKind() != vector::CombiningKind::MAXNUMF) {
      reason = (Twine("reduction kind ") +
                Twine(vector::stringifyCombiningKind(red.getKind())) +
                " is not maxnumf; only the max fold is rewritten here (addf would "
                "reassociate the row sum)")
                   .str();
      return failure();
    }
    if (red.getVector() != chain.chunkRead.getResult()) {
      reason = "the reduction does not fold the chunk vector this pass re-reads";
      return failure();
    }
    if (red.getAcc() != extract.getResult()) {
      reason = "the reduction's accumulator is not the running value this pass "
               "folds in";
      return failure();
    }
    chain.reduction = red;
    chain.fastmath = red.getFastmath();

    // ---- the write-back: insert into *that* slice, splice into the carrier -------
    auto ins = inserts[0];
    if (ins.getDest() != accSlice.getResult()) {
      reason = "the tensor.insert does not target the accumulator slice the "
               "reduction reads; the rewrite writes the carrier instead";
      return failure();
    }
    if (ins.getScalar() != red.getResult()) {
      reason = "the tensor.insert does not insert the reduction's result";
      return failure();
    }
    chain.insert = ins;

    auto wb = insertSlices[0];
    {
      auto offs = sliceIndices(wb.getStaticOffsets(), wb.getOffsets());
      if (offs.size() != 1 || !isAt(offs[0], rowIV)) {
        reason = "the tensor.insert_slice is not indexed by rowIV alone; expected "
                 "the running value of the row this iteration owns";
        return failure();
      }
      if (wb.getDest() != colCarrier) {
        reason = "the tensor.insert_slice does not splice into the column loop's "
                 "own loop-carried carrier";
        return failure();
      }
      if (wb.getSource() != ins.getResult()) {
        reason = "the tensor.insert_slice does not splice the tensor.insert's "
                 "result";
        return failure();
      }
      if (wb.getType() != colCarrier.getType()) {
        reason = "the tensor.insert_slice does not reproduce the carrier's type";
        return failure();
      }
    }
    // The column loop must actually *yield* the write-back, or the reduction is
    // dead in the original IR and the rewrite would turn a dead store live.
    auto colTerm = dyn_cast<scf::YieldOp>(body->getTerminator());
    if (!colTerm || colTerm.getResults().size() != 1 ||
        colTerm.getResults()[0] != wb.getResult()) {
      reason = "the column loop does not yield the value its tensor.insert_slice "
               "produced, so the reduction is dead in the original IR; the rewrite "
               "would turn a dead store into a live one";
      return failure();
    }
    chain.writeBack = wb;
    return success();
  }

  // =========================== enclosing-forall gates ==========================

  /// The `scf.forall` a row loop lives in, if any. Its tile base is what makes
  /// `globalRow % lanes == localRow % lanes` hold; see the G3 gate.
  static scf::ForallOp enclosingForall(Operation *op) {
    for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
      if (auto forall = dyn_cast<scf::ForallOp>(p))
        return forall;
      if (isa<func::FuncOp>(p))
        break;
    }
    return {};
  }

  /// G3 for the enclosing forall. Every tile base is `lb + k * step`, and
  /// `tileBase` itself is not a constant -- so what has to be a multiple of the
  /// lane count is that pair, plus the tile extent. A forall of any other rank
  /// is refused, not skipped: the gate's argument is about the *global* row
  /// base, and a rank-2 forall's dim-0 lb moves it just as much.
  static bool gateForallTile(scf::ForallOp forall, int64_t lanes) {
    if (!forall)
      return false;
    if (forall.getRank() != 1)
      return reject(forall,
                    Twine("the enclosing scf.forall has rank ") +
                        Twine(forall.getRank()) +
                        "; expected rank 1, because the group alignment argument "
                        "is about one global row base and a higher-rank forall "
                        "has more than one");
    auto lb = constantIntOf(forall.getMixedLowerBound().front());
    auto ub = constantIntOf(forall.getMixedUpperBound().front());
    auto step = constantIntOf(forall.getMixedStep().front());
    if (!lb || !ub || !step)
      return reject(forall,
                    Twine("the enclosing scf.forall bounds are not static "
                          "constants; expected constants so that tile lb + k*step "
                          "is a multiple of the ") +
                        Twine(lanes) + "-lane group");
    int64_t span = *ub - *lb;
    if (*lb % lanes != 0 || *step % lanes != 0 || span % lanes != 0)
      return reject(forall,
                    Twine("the enclosing scf.forall tile [lb=") + Twine(*lb) +
                        ", ub=" + Twine(*ub) + ") step " + Twine(*step) +
                        " is not groupable by the " + Twine(lanes) +
                        "-lane group (needs lb % " + Twine(lanes) +
                        " == 0, step % " + Twine(lanes) + " == 0, (ub-lb) % " +
                        Twine(lanes) +
                        " == 0; the tile base is lb + k*step, not a constant)");
    return false;
  }

  /// The tile's dim-0 offset must be the enclosing forall's induction variable,
  /// or the constant 0. This is what catches a *more distant* loop that shifts
  /// the base: `tensor.extract_slice %S[%row0]` with
  /// `%row0 = arith.addi %outerBase, %iv` is not the induction variable, so the
  /// group's global row base is not provably a multiple of the lane count.
  static bool gateTileOrigin(scf::ForallOp forall, Value tile, int64_t lanes) {
    auto slice = tile.getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice)
      return false; // not an extract_slice: there is no offset to reason about
    auto offs = sliceIndices(slice.getStaticOffsets(), slice.getOffsets());
    if (offs.empty() || isConst(offs[0], 0))
      return false;
    if (forall && forall.getInductionVars().size() == 1 &&
        isAt(offs[0], forall.getInductionVar(0)))
      return false;
    return reject(slice,
                  Twine("the tile's dim-0 offset is not the enclosing "
                        "scf.forall's induction variable nor the constant 0; "
                        "something outside re-bases the global rows, so the group "
                        "store would land at a row base that is not a multiple of "
                        "the ") +
                      Twine(lanes) + "-lane group");
  }

  /// In tensor form the `scf.forall` legitimately carries a result; what this
  /// pass needs is that the row loop's result reaches the tile through exactly
  /// one `tensor.parallel_insert_slice`, into that forall's out-argument,
  /// covering the tile's rows. (The rank-1 / no-results / no-outputs shape
  /// `FormAsyncThreads` needs is a post-bufferize property, left to that pass.)
  static bool gateForallWriteBack(scf::ForallOp forall, scf::ForOp rowLoop,
                                  int64_t rows) {
    if (!forall)
      return false;
    if (rowLoop.getNumResults() != 1 || !rowLoop.getResult(0).hasOneUse())
      return reject(forall,
                    Twine("the row loop's result has ") +
                        Twine(rowLoop.getNumResults() != 1
                                  ? "a different number of results"
                                  : "more than one use") +
                        " uses; expected exactly 1, feeding a single "
                        "tensor.parallel_insert_slice into the scf.forall's "
                        "out-argument");
    Operation *user = (*rowLoop.getResult(0).use_begin()).getOwner();
    auto pis = dyn_cast<tensor::ParallelInsertSliceOp>(user);
    if (!pis || pis.getSource() != rowLoop.getResult(0))
      return reject(rowLoop,
                    Twine("the row loop's only user is a ") +
                        (pis ? Twine("tensor.parallel_insert_slice that reads "
                                     "something else")
                             : Twine(user->getName().getStringRef())) +
                        "; expected a tensor.parallel_insert_slice whose source is "
                        "the row loop's result");
    auto outArgs = forall.getRegionOutArgs();
    if (outArgs.empty() || pis.getDest() != outArgs.front())
      return reject(forall,
                    "the tensor.parallel_insert_slice does not target the "
                    "scf.forall's out-argument");
    auto sizes = sliceIndices(pis.getStaticSizes(), pis.getSizes());
    if (sizes.size() != 1 || !isConst(sizes[0], rows))
      return reject(pis,
                    Twine("the tensor.parallel_insert_slice does not cover the "
                          "tile's ") +
                        Twine(rows) +
                        " rows; expected sizes [rows] so the group store and the "
                        "scalar store retire the same set of rows");
    return false;
  }

  // ================================ the driver ================================

  /// Try to rewrite `rowLoop`. Returns 1 when it was rewritten and 0 when a gate
  /// refused it, after emitting the remark that says which gate. Never fails the
  /// pass: this is an optimization, and a kernel it cannot handle is a kernel
  /// that must keep compiling.
  int64_t rewriteRowLoop(scf::ForOp rowLoop) {
    // No `isCandidateRowLoop` here: the candidate set was already filtered by it
    // in `runOnOperation`, so re-running it would be dead work. The null guard
    // below is kept because `colLoop` is dereferenced unconditionally after it,
    // and it is the only thing between a future change to the filter and a null
    // dereference.
    scf::ForOp colLoop;
    for (Operation &op : *rowLoop.getBody()) {
      if (auto inner = dyn_cast<scf::ForOp>(op))
        colLoop = inner;
    }
    if (!colLoop) {
      if (reject(rowLoop, "no column scf.for to rewrite; expected one nested "
                          "column loop (guaranteed by the candidate filter)"))
        return 0;
    }

    GroupStoreShape shape;
    shape.rowCarrier = rowLoop.getInitArgs()[0];
    auto carrierTy = dyn_cast<RankedTensorType>(shape.rowCarrier.getType());
    if (!carrierTy || carrierTy.getRank() != 1 || !carrierTy.hasStaticShape()) {
      if (reject(rowLoop, Twine("the row carrier ") +
                               typeToString(shape.rowCarrier.getType()) +
                               " is not a static rank-1 tensor<rows x T>; "
                               "expected one running value per row"))
        return 0;
    }
    shape.rows = carrierTy.getDimSize(0);
    auto elemBytes = elemBytesOf(carrierTy.getElementType());
    if (!elemBytes) {
      if (reject(rowLoop, Twine("the row carrier element type ") +
                               typeToString(carrierTy.getElementType()) +
                               " has no HVX lane count; expected f32 or f16"))
        return 0;
    }
    shape.elemBytes = *elemBytes;
    shape.lanes = kHvxVectorBytes / shape.elemBytes;
    Type elemTy = carrierTy.getElementType();

    // The enclosing forall is checked first: its rank and tile bounds need only
    // `lanes`, and a rank-2 forall should be named as such rather than as
    // whichever chain gate happens to trip next.
    scf::ForallOp forall = enclosingForall(rowLoop.getOperation());
    if (gateForallTile(forall, shape.lanes))
      return 0;

    // ---- the column loop's static geometry (G0) ----------------------------------
    auto colLb = constantIntOf(colLoop.getLowerBound());
    auto colUb = constantIntOf(colLoop.getUpperBound());
    auto colStep = constantIntOf(colLoop.getStep());
    if (!colLb || !colUb || !colStep) {
      if (reject(colLoop, "the column loop bounds are not static constants; "
                          "expected constants so that the chunk count is a "
                          "compile-time number"))
        return 0;
    }
    if (*colLb != 0) {
      if (reject(colLoop, Twine("the column loop lower bound ") + Twine(*colLb) +
                               " is not 0; expected 0 so chunk k starts at k*lanes"))
        return 0;
    }
    if (*colStep != shape.lanes) {
      if (reject(colLoop,
                 Twine("the column step ") + Twine(*colStep) + " is not the " +
                     Twine(shape.lanes) +
                     "-lane group width (kHvxVectorBytes/elemBytes = " +
                     Twine(kHvxVectorBytes) + "/" + Twine(shape.elemBytes) +
                     "); the horizontal butterfly is only defined on a whole "
                     "128 B register"))
        return 0;
    }
    int64_t colSpan = *colUb - *colLb;
    if (colSpan <= 0 || colSpan % shape.lanes != 0) {
      if (reject(colLoop,
                 Twine("the column width ") + Twine(colSpan * shape.elemBytes) +
                     " B is not a positive multiple of the " +
                     Twine(kHvxVectorBytes) + " B vector (" + Twine(colSpan) +
                     " elements is not a multiple of " + Twine(shape.lanes) +
                     " lanes)"))
        return 0;
    }

    // ---- the column loop's op chain ---------------------------------------------
    ColumnChain chain;
    std::string reason;
    if (failed(matchColumnChain(colLoop, rowLoop.getInductionVar(),
                                colLoop.getInductionVar(), elemTy, shape.lanes,
                                shape, chain, reason))) {
      if (reject(colLoop, reason))
        return 0;
    }

    // The loop must cover the tile exactly. The rewrite emits one
    // `transfer_read` per chunk out of the tile's own dim 1, so a loop that
    // stops short of it (or runs past it) would read the wrong columns, and a
    // zero-width tile would leave the butterfly with no accumulator at all.
    if (colSpan != shape.cols) {
      if (reject(colLoop, Twine("the column loop covers ") + Twine(colSpan) +
                               " elements but the source tile is " +
                               Twine(shape.cols) +
                               " wide; expected them to be equal so that one "
                               "transfer_read per chunk addresses the right "
                               "columns"))
        return 0;
    }

    // ---- the row loop's own geometry (G3, row half) -------------------------------
    auto rowLb = constantIntOf(rowLoop.getLowerBound());
    auto rowUb = constantIntOf(rowLoop.getUpperBound());
    auto rowStep = constantIntOf(rowLoop.getStep());
    if (!rowLb || !rowUb || !rowStep) {
      if (reject(rowLoop, "the row loop bounds are not static constants"))
        return 0;
    }
    shape.rowLower = *rowLb;
    shape.rowUpper = *rowUb;
    int64_t rowSpan = *rowUb - *rowLb;
    if (*rowLb % shape.lanes != 0 || *rowStep != 1 ||
        rowSpan % shape.lanes != 0) {
      if (reject(rowLoop,
                 Twine("the row range [lb=") + Twine(*rowLb) + ", ub=" +
                     Twine(*rowUb) + ") step " + Twine(*rowStep) +
                     " is not groupable by the " + Twine(shape.lanes) +
                     "-lane group (needs lb % " + Twine(shape.lanes) +
                     " == 0, step == 1, (ub-lb) % " + Twine(shape.lanes) +
                     " == 0)"))
        return 0;
    }
    if (rowSpan != shape.rows) {
      if (reject(rowLoop, Twine("the row loop covers ") + Twine(rowSpan) +
                               " rows but the row carrier has " +
                               Twine(shape.rows)))
        return 0;
    }

    // ---- the enclosing forall: tile origin and write-back ----------------------
    if (gateTileOrigin(forall, shape.sourceTile, shape.lanes))
      return 0;
    if (gateTileOrigin(forall, shape.rowCarrier, shape.lanes))
      return 0;
    if (gateForallWriteBack(forall, rowLoop, shape.rows))
      return 0;

    // ---- locality: the value flows through this nest and nowhere else ------------
    // The same contract from both ends: a second reader would see a stale value
    // in the scalar form and a differently-timed one here.
    if (!shape.rowCarrier.hasOneUse()) {
      if (reject(rowLoop, Twine("the row carrier has ") +
                               Twine(shape.rowCarrier.getNumUses()) +
                               " uses; expected 1, because a second reader would "
                               "see the group store at a different time than the "
                               "scalar store"))
        return 0;
    }
    if (!forall && (rowLoop.getNumResults() != 1 ||
                    !rowLoop.getResult(0).hasOneUse())) {
      if (reject(rowLoop,
                 "the row loop's result has more than one use; expected 1"))
        return 0;
    }
    if (colLoop.getInitArgs().size() != 1 ||
        colLoop.getInitArgs().front() != rowLoop.getRegionIterArg(0)) {
      if (reject(colLoop, "the column loop does not thread the row carrier"))
        return 0;
    }

    // ---- the row loop must consume the column loop's result ---------------------
    // Otherwise the column loop's write-back is dead and the rewrite would
    // resurrect it as a live running update.
    auto rowTerm = dyn_cast<scf::YieldOp>(rowLoop.getBody()->getTerminator());
    if (!rowTerm || rowTerm.getResults().size() != 1 ||
        rowTerm.getResults()[0] != colLoop.getResult(0)) {
      if (reject(rowLoop,
                 "the row loop does not yield the column loop's result, so the "
                 "reduction is dead in the original IR; the rewrite would turn a "
                 "dead store live"))
        return 0;
    }

    rewrite(rowLoop, shape, chain.fastmath);
    return 1;
  }

  /// Emit the group form in place of the row loop, and erase what it replaces.
  /// The column loop is neither passed in nor touched: it lives in the row
  /// loop's own region, so erasing the row loop takes it with it.
  void rewrite(scf::ForOp rowLoop, const GroupStoreShape &shape,
               arith::FastMathFlags redFastmath) {
    Location loc = rowLoop.getLoc();
    OpBuilder b(rowLoop);

    const int64_t lanes = shape.lanes;
    const int64_t chunks = shape.cols / lanes;
    // `mlir::cast<>` as a free function: `Type` has no member `.cast<>` on this pin.
    Type elemTy =
        mlir::cast<RankedTensorType>(shape.rowCarrier.getType()).getElementType();
    auto vecTy = VectorType::get({lanes}, elemTy);
    auto vecI32Ty = VectorType::get({lanes}, b.getI32Type());
    // tensor<lanes x T>: both the group slice and each row chunk land on it.
    auto vecSliceTy = RankedTensorType::get({lanes}, elemTy);

    // `nnan` only, never the whole `fast`; see the header for why, and for what
    // `HexagonAddFastMath` does to this attribute two passes later.
    arith::FastMathFlags foldFlags = redFastmath | arith::FastMathFlags::nnan;

    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value cLanes = arith::ConstantIndexOp::create(b, loc, lanes);
    Value cRowLower = arith::ConstantIndexOp::create(b, loc, shape.rowLower);
    Value cRowUpper = arith::ConstantIndexOp::create(b, loc, shape.rowUpper);
    Value pad =
        arith::ConstantOp::create(b, loc, elemTy, b.getZeroAttr(elemTy));

    // The constant lane-index vector the dynamic predicate compares against:
    // lane k of the group is row `g + k`, so `laneIdx == %j` selects the lane
    // this iteration of the inner loop owns.
    SmallVector<int32_t> laneIdx(static_cast<size_t>(lanes));
    for (int64_t i = 0; i < lanes; ++i)
      laneIdx[i] = static_cast<int32_t>(i);
    Value laneIdxVec = arith::ConstantOp::create(
        b, loc, vecI32Ty,
        // The ArrayRef is spelled out: DenseElementsAttr's integer overload is
        // a template on ArrayRef<T>, and template deduction cannot see through
        // the implicit SmallVector -> ArrayRef conversion.
        DenseElementsAttr::get(vecI32Ty, llvm::ArrayRef<int32_t>(laneIdx)));

    // Chunk column offsets, hoisted out of both loops.
    SmallVector<Value> chunkCol;
    chunkCol.reserve(chunks);
    for (int64_t k = 0; k < chunks; ++k)
      chunkCol.push_back(k == 0 ? c0
                                : arith::ConstantIndexOp::create(b, loc, k * lanes)
                                      .getResult());

    static constexpr bool kInBounds[] = {true};
    const llvm::ArrayRef<bool> inBounds(kInBounds);
    // Sizes and strides, per rank. A rank-1 slice (the group) is [lanes] / [1];
    // a rank-2 chunk is [1, lanes] / [1, 1].
    SmallVector<OpFoldResult> oneSize{b.getIndexAttr(1)};
    SmallVector<OpFoldResult> laneSize{b.getIndexAttr(lanes)};
    SmallVector<OpFoldResult> chunkSizes{oneSize.front(), laneSize.front()};
    SmallVector<OpFoldResult> unitStrides{b.getIndexAttr(1), b.getIndexAttr(1)};

    // ---- the group loop: replaces the row loop, one iteration per `lanes` rows --
    auto groupLoop = scf::ForOp::create(
        b, loc, cRowLower, cRowUpper, cLanes, ValueRange{shape.rowCarrier},
        [](OpBuilder &, Location, Value, ValueRange) {});
    b.setInsertionPointToStart(groupLoop.getBody());
    Value g = groupLoop.getInductionVar();
    Value groupCarrier = groupLoop.getRegionIterArg(0);

    // The group's running values as one vector. The slice is of the same buffer
    // the scalar form wrote one element of, so this is a change of emission, not
    // of layout.
    Value groupSlice = tensor::ExtractSliceOp::create(
        b, loc, vecSliceTy, groupCarrier, SmallVector<OpFoldResult>{g}, laneSize,
        oneSize);
    Value groupVec = vector::TransferReadOp::create(
        b, loc, vecTy, groupSlice, ValueRange{c0}, pad, inBounds);

    // ---- one iteration per row of the group --------------------------------------
    auto laneLoop = scf::ForOp::create(
        b, loc, c0, cLanes, c1, ValueRange{groupVec},
        [](OpBuilder &, Location, Value, ValueRange) {});
    b.setInsertionPointToStart(laneLoop.getBody());
    Value j = laneLoop.getInductionVar();
    Value group = laneLoop.getRegionIterArg(0);

    // Row `g + j` of the tile. The old row loop's induction variable indexed the
    // tile directly, so `g` has to appear here too: the group's rows are
    // `g .. g+lanes-1` and iteration `j` owns lane `j` of them.
    Value row = arith::AddIOp::create(b, loc, g, j).getResult();

    // Elementwise fold of this row's 128 B chunks: lane i ends up holding the
    // fold over column i of every chunk, which is the butterfly's input.
    Value acc;
    for (int64_t k = 0; k < chunks; ++k) {
      Value chunk = tensor::ExtractSliceOp::create(
          b, loc, vecSliceTy, shape.sourceTile,
          SmallVector<OpFoldResult>{row, chunkCol[k]}, chunkSizes, unitStrides);
      Value read = vector::TransferReadOp::create(
          b, loc, vecTy, chunk, ValueRange{c0}, pad, inBounds);
      acc = k == 0 ? read : emitFold(b, loc, acc, read, foldFlags);
    }

    // Butterfly: rotate the register right by half the remaining span and fold;
    // after log2(128/elemBytes) steps every lane holds the row's reduction.
    // Copied from VectorRowReducePass.cpp, with one difference: it stops before
    // that pass's `vector.extract` + `memref.store`, which is the whole point.
    for (int64_t bytes = kHvxVectorBytes / 2; bytes >= shape.elemBytes;
         bytes /= 2) {
      Value rotated =
          hvx::VrorOp::create(b, loc, vecTy, acc, b.getI32IntegerAttr(bytes));
      acc = emitFold(b, loc, rotated, acc, foldFlags);
    }

    // Fold the group's previous value in elementwise, then place the result in
    // lane j only. Lane j of the merge is `max(rowmax_j, group[j])`, which is
    // exactly the scalar nest's running update; the other lanes are thrown away
    // by the select and take `group` unchanged.
    Value merged = emitFold(b, loc, acc, group, foldFlags);

    // Dynamic lane predicate. `vector.broadcast` needs an i32 source, so the
    // index induction variable is cast first. The select's operand order is
    // load-bearing: merged (true), group (false).
    Value j32 = arith::IndexCastOp::create(b, loc, b.getI32Type(), j);
    Value jSplat = vector::BroadcastOp::create(b, loc, vecI32Ty, j32);
    Value mask = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                       laneIdxVec, jSplat);
    Value selected = arith::SelectOp::create(b, loc, mask, merged, group);
    scf::YieldOp::create(b, loc, selected);

    // ---- retire the whole group with one 128 B write -------------------------------
    b.setInsertionPointAfter(laneLoop);
    // `getResult()`, not an implicit op->Value: this op's result is
    // `Optional<AnyRankedTensor>`, so it carries ZeroOrOneResult and offers no
    // converting constructor.
    Value written = vector::TransferWriteOp::create(
                        b, loc, laneLoop.getResult(0), groupSlice, ValueRange{c0},
                        std::optional<llvm::ArrayRef<bool>>(inBounds))
                        .getResult();
    Value newCarrier = tensor::InsertSliceOp::create(
        b, loc, written, groupCarrier, SmallVector<OpFoldResult>{g}, laneSize,
        oneSize);
    scf::YieldOp::create(b, loc, newCarrier);

    // Only the row loop is erased, and only after its own results have been
    // rewired: `~Operation` refuses to destroy an op whose results still have
    // uses, and the *column* loop's result is still used by the row loop's own
    // terminator. Erasing the column loop first therefore aborts the compiler.
    // The row loop owns the column loop's region, so erasing the outer op takes
    // the inner one with it -- and the outer loop's results are the only ones
    // `use_empty()` is checked on.
    rowLoop.getResult(0).replaceAllUsesWith(groupLoop.getResult(0));
    rowLoop.erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createRowReduceGroupStorePass() {
  return std::make_unique<RowReduceGroupStorePass>();
}