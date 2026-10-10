//===- HmxVectorReadoutPass.cpp - batch the read-out onto a second thread ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Moves a lowered HMX matmul's accumulator read-out off the matrix engine's
// thread. The design, the measurements it rests on and the runtime ABI live in
// bin/runtime/include/HmxVectorExecutor.h; the option semantics live in
// Passes.td (HmxVectorReadout).
//
// What this file is responsible for is the half that can go wrong SILENTLY:
// deciding which loops it is entitled to rewrite, and proving that the rewrite
// still reads every AR row exactly once. A row that is never read leaves the
// caller's output buffer holding stale bytes, which is a plausible-looking wrong
// answer -- strictly worse than leaving the read-out where it was. So every
// condition in matchReadout is a decline and never a repair, and a shape the
// pass does not fully understand comes out byte-identical.
//
// The load-bearing layout fact, re-checked here rather than assumed: the AR
// crouton array is allocated before the m-tile loop and released after it, so
// deferring a group of rows costs no extra VTCM and needs no double buffering.
// That is why a rolling accumulator would make this rewrite unsound, and why
// "allocated before the loop, released after it, exactly once" is checked.
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDType.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxIndexFold.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxReadoutHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_HMXVECTORREADOUT
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

using namespace mlir;
using namespace mlir::hmx;

namespace {

/// Runtime entry points and the descriptor layout are NOT spelled here: they are
/// the frozen ABI in bin/runtime/include/HmxVectorExecutor.h, and the compiler
/// half of that ABI lives in one header shared with the pass that emits the
/// `configure` call (HmxReadoutHandoff.h). Declaring `configure` here and
/// calling it from there would make a one-sided rename an unresolved link and a
/// one-sided reordering a garbage pointer, with nothing to catch either.

/// Base name of the outlined read-out. One function per specialisation; a
/// second, differently-shaped specialisation becomes `__hmx_readout_1`, so a
/// module holding two shapes keeps two correct functions rather than one
/// function whose body silently describes the wrong layout.
constexpr StringLiteral kReadoutFnPrefix = "__hmx_readout";

/// The index folding this pass needs -- loop bounds, the pipeliner's shifted
/// row chains, the peeled epilogue row -- lives in the shared HmxIndexFold.h,
/// together with the reasons it must not be duplicated: the same chains are
/// folded by ThreadRolePartition, and two folders that disagree about a chain
/// would be a silent coverage bug, not a style difference.

/// Erase `value`'s defining op once nothing uses it, walking back through the
/// chain so the tile loop does not keep the `addi`/`muli` only the read-out
/// needed. The loop's iter_args are deliberately untouched: this pass publishes
/// off the induction variable rather than off a carried pending-count, so the
/// software-pipelined loop keeps exactly the cross-stage (token, slot) pair the
/// pipeliner gave it, and no loop-carried state is invented.
static void eraseDeadDefiners(Value value, unsigned depth = 0) {
  if (depth > kHmxFoldDepth)
    return;
  Operation *def = value.getDefiningOp();
  // `isOpTriviallyDead`, NOT `wouldOpBeTriviallyDead`: the latter asks "would
  // this op be dead if unused", so it says yes for a value some other op still
  // uses -- erasing that would destroy a live op. The row arithmetic here is
  // shared with the `hmx.acc_read` that writes the row, which is exactly the
  // case that distinction exists for.
  if (!def || !mlir::isOpTriviallyDead(def))
    return;
  SmallVector<Value> operands(def->getOperands());
  def->erase();
  for (Value operand : operands)
    eraseDeadDefiners(operand, depth + 1);
}

/// One rewrite candidate: a tile loop this pass has decided it fully understands.
struct ReadoutMatch {
  scf::ForOp loop;
  UnpackAccOp unpack;   ///< the read-out inside the loop body
  UnpackAccOp epilogue; ///< the peeled row's read-out, or null
  Value ar;
  Value dst;
  MemRefType arType;
  MemRefType dstType;
  int64_t mTiles; ///< ar dim 0
  int64_t upper;  ///< loop upper bound, folded
  int64_t col;    ///< the read-out's crouton block, folded
  int64_t count;  ///< the read-out's `count` attribute
};

/// The AR array must be a VTCM (space 1) buffer: that is the only address space
/// the vector thread reads the croutons out of, and it is what `hmx.acc_read`
/// writes.
static bool isVtcmMemref(MemRefType type) {
  Attribute space = type.getMemorySpace();
  auto integer = dyn_cast_or_null<IntegerAttr>(space);
  return integer && integer.getInt() == 1;
}

/// Decide whether `loop` is the shape this pass rewrites. Nothing here repairs
/// anything: every failure is a decline, because a mis-read row is a wrong answer
/// that still looks right.
static std::optional<ReadoutMatch> matchReadout(scf::ForOp loop,
                                                DominanceInfo &dominance) {
  Block *body = loop.getBody();
  Value iv = loop.getInductionVar();

  // A plain 0..ub step-1 loop. A loop that cannot run leaves nothing to publish,
  // and would make the one-per-function `configure` fire on a path that never
  // hands anything over.
  std::optional<int64_t> lower = foldIndex(loop.getLowerBound());
  std::optional<int64_t> upper = foldIndex(loop.getUpperBound());
  std::optional<int64_t> step = foldIndex(loop.getStep());
  if (!lower || !upper || !step || *lower != 0 || *step != 1 || *upper < 1)
    return std::nullopt;

  // Exactly one read-out, sitting directly in this body. A nested one would need
  // the inner loop rewritten too, and this pass does not claim to understand a
  // nest it has not read.
  UnpackAccOp unpack;
  for (Operation &op : *body) {
    if (auto candidate = dyn_cast<UnpackAccOp>(&op)) {
      if (unpack)
        return std::nullopt;
      unpack = candidate;
    }
  }
  if (!unpack)
    return std::nullopt;

  // The bounds-safe tail form lowers to `hmx_unpack_acc_tail_f16` and carries
  // STATIC valid extents. Per-row validity is a property of the LAST row of the
  // group, which a static attribute cannot express once rows are grouped, so it
  // is declined rather than approximated. `n_tile` is the diagnostic output
  // bridge and is declined for the same reason.
  if (unpack.getValidRows() || unpack.getValidCols() || unpack.getNTile())
    return std::nullopt;

  // The row must be the induction variable itself, and the crouton block must
  // not be: a `col` that moved with `iv` would make the group's crouton walk
  // meaningless.
  if (inductionOffset(unpack.getRow(), iv) != 0)
    return std::nullopt;
  if (unpack.getCol() == iv)
    return std::nullopt;
  std::optional<int64_t> col = foldIndex(unpack.getCol());
  if (!col)
    return std::nullopt;

  // The AR array: a plain VTCM allocation outside the loop. `memref.alloc` and
  // not a view, because the descriptor carries the array's own base address and
  // a view's address is not the array's.
  if (!unpack.getSrc().getDefiningOp<memref::AllocOp>())
    return std::nullopt;
  auto arType = dyn_cast<MemRefType>(unpack.getSrc().getType());
  if (!arType || !arType.hasStaticShape() || !isVtcmMemref(arType))
    return std::nullopt;
  if (!dominance.dominates(unpack.getSrc(), loop))
    return std::nullopt;

  // The destination must be defined outside the loop too. That it is the SAME
  // value every iteration is what lets one descriptor name a whole group: the
  // `row` operand alone selects AR row m and destination rows m*32 .. m*32+31
  // (`hmx__unpack_acc_f16_body`, HMXLayout.c:519: r0 = tile_row*32 + 2*block_j).
  Value dst = unpack.getDst();
  auto dstType = dyn_cast<MemRefType>(dst.getType());
  if (!dstType || !dstType.hasStaticShape())
    return std::nullopt;
  if (!dominance.dominates(dst, loop))
    return std::nullopt;

  // Something in this body must have written the row the read-out names;
  // otherwise "publish at the end of iteration m" hands the vector thread a row
  // no engine work produced.
  bool writtenHere = false;
  loop.walk([&](AccReadOp read) {
    if (read.getDst() == unpack.getSrc() &&
        inductionOffset(read.getM(), iv) == 0)
      writtenHere = true;
  });
  if (!writtenHere)
    return std::nullopt;

  // Every AR row is live for the whole loop: the array is released after the
  // loop, and exactly once. This is what the deferral rests on -- a per-
  // iteration or per-group AR buffer would make a batch read freed memory.
  func::FuncOp function = loop->getParentOfType<func::FuncOp>();
  bool releasedAfter = false;
  bool releasedInside = false;
  int64_t releases = 0;
  function.walk([&](memref::DeallocOp dealloc) {
    if (dealloc.getMemref() != unpack.getSrc())
      return;
    ++releases;
    if (loop->isAncestor(dealloc))
      releasedInside = true;
    else if (!releasedAfter)
      releasedAfter = true;
  });
  if (!releasedAfter || releasedInside || releases != 1)
    return std::nullopt;

  // The peeled epilogue's read-out, if this kernel has one. Depth 2 peels
  // exactly one row, so there must be exactly one post-loop read-out of this AR
  // array and it must name the row the loop did not run. If the original IR
  // already leaves a gap between `upper` and that row, publishing the gap would
  // read a row no `acc_read` ever wrote -- i.e. whatever the previous launch
  // left in the array -- so this is declined, never repaired.
  int64_t outside = 0;
  UnpackAccOp epilogue;
  function.walk([&](UnpackAccOp other) {
    if (other == unpack || other.getSrc() != unpack.getSrc())
      return;
    if (loop->isAncestor(other))
      return;
    ++outside;
    epilogue = other;
  });
  int64_t mTiles = arType.getDimSize(0);
  if (outside > 1)
    return std::nullopt;
  if (epilogue) {
    if (epilogue.getDst() != dst || epilogue.getValidRows() ||
        epilogue.getValidCols() || epilogue.getNTile() ||
        epilogue.getCount().value_or(1) != unpack.getCount().value_or(1))
      return std::nullopt;
    std::optional<int64_t> peeled = foldIndex(epilogue.getRow());
    if (!peeled || *peeled != mTiles - 1 || *upper != mTiles - 1)
      return std::nullopt;
  } else if (*upper != mTiles) {
    return std::nullopt;
  }

  return ReadoutMatch{
      loop, unpack, epilogue, unpack.getSrc(), dst, arType, dstType, mTiles,
      *upper, *col, static_cast<int64_t>(unpack.getCount().value_or(1))};
}

/// Declare a runtime entry point once, as a private `func.func` declaration.
///
/// Same shape as the engine leaves at `HmxToLLVMPass.cpp:58-63` (`lookupOrCreateFn`
/// on the `llvm.func` side): a plain external symbol with no body, resolved by
/// the linker against the runtime library. Declaring it in the `func` dialect is
/// what lets a pass running BEFORE `convert-func-to-llvm` emit a call to it; the
/// conversion carries the declaration through unchanged, and a bare `!llvm.ptr`
/// parameter survives it (verified: `func.func private @f(!llvm.ptr, i32)`
/// converts to `llvm.func @f(!llvm.ptr, i32)`).
///
/// `setPrivate()` is REQUIRED here and is NOT a statement of intent. On this
/// MLIR version a body-less `func.func` declaration with public visibility is
/// rejected outright -- "'func.func' op symbol declaration cannot have public
/// visibility" -- so this is the only legal spelling of an external declaration
/// in the func dialect. Measured (2026-10-03): removing it makes the real Triton
/// compile fail with that error three times.
///
/// IT IS ALSO INERT, which is the second half of the same fact and the reason it
/// is safe. `ConvertFuncToLLVMPass` propagates the attribute to
/// `llvm.func ... sym_visibility = "private"` (FuncToLLVM.cpp:411), but this
/// MLIR's LLVM-dialect -> LLVM-IR translation never reads it
/// (LLVMToLLVMIRTranslation.cpp has no `sym_visibility` / `setLinkage` hit), so
/// the attribute evaporates and the declaration becomes an ordinary external
/// symbol. That is not an accident to be tolerated but a necessity: LLVM IR
/// cannot express an internal-linkage declaration at all, and `llc` rejects one
/// with "error: invalid linkage for function declaration". Verified end to end
/// by test_hmx_vector_readout_object_gate.py, which reads the produced OBJECT:
/// a real 1024x512x64 matmul compiled with enableHmxVectorReadout=True defines
/// `__hmx_readout` and `__hmx_readout_entry` and references
/// `hexagon_runtime_hmx_exec_{configure,publish,drain}`.
///
/// `results` is the frozen header's result list, not a choice: `configure` and
/// `publish` both return a count in the ABI
/// (bin/runtime/include/HmxVectorExecutor.h:122,135). Declaring them `void`
/// would leave the lowering unable to declare the real types -- `lookupOrCreateFn`
/// rejects a redefinition of a different type -- so the return would have to be
/// checked from a declaration that does not have one.
static func::FuncOp declareRuntime(ModuleOp module, StringRef name,
                                   ArrayRef<Type> params,
                                   ArrayRef<Type> results = {}) {
  // Sibling function passes run CONCURRENTLY (the pass manager is
  // multithreaded), and two invocations racing check-then-insert here produce
  // two declarations of the same symbol -- a verifier error, not a warning.
  // The shared module-state mutex is the established serializer for exactly
  // this class of module mutation (WeightResidentPass's manifest setter,
  // MatmulToHmx's records); nothing that holds it calls back into this pass,
  // so the lock cannot nest.
  std::lock_guard<std::mutex> guard(hmxModuleStateMutex());
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  // The builder-taking overload is the one that INSERTS at the builder's
  // insertion point; the location-only overload builds a detached op that never
  // lands in the module.
  auto fn = func::FuncOp::create(builder, builder.getUnknownLoc(), name,
                                 builder.getFunctionType(params, results));
  fn.setPrivate();
  return fn;
}

/// The outlined read-out: one `hmx.unpack_acc` per row of the group, with the
/// destination and the crouton walk loop-invariant.
///
/// `ar` and `dst` are memrefs because `hmx.unpack_acc` takes memrefs, and there
/// is no way to have the outlined function take the RUNTIME's shape
/// `(const HmxReadoutBatch*, uint32_t)` instead: it would have to read the
/// descriptor's `ar`/`dst` words and turn them back into memrefs, and the `func`
/// dialect has no operation that builds a memref from a pointer. `memref.cast`,
/// `reinterpret_cast` and `get_global` all require a memref (or a symbol) to
/// start from, and the ops that do produce memrefs (`alloc`, `alloca`, `get_global`)
/// produce pointers the descriptor does not contain. So the outlined read-out
/// keeps the memref shape, and HmxToLLVMPass generates the small entry point that
/// unpacks the descriptor and calls this function (see HmxReadoutHandoff.h).
///
/// That is the whole reason the specialisation is keyed on (arType, dstType):
/// the entry point has to reconstruct each memref's descriptor, and it can only
/// do that from the types, which is why they are published in the handoff
/// record.
///
/// `nrows` is a runtime argument rather than a constant because it is the tail
/// batch's row count, which need not be a multiple of the group size. `col` and
/// `count` are baked in because `hmx.unpack_acc` takes both as STATIC attributes
/// (HmxOps.td:329-331) -- there is no way to make them runtime values -- and both
/// are loop-invariant, so one specialisation per (arType, dstType, count, col) is
/// exact.
///
/// `decisionId` is the identity the op it REPLACES carried, and it is load-bearing
/// rather than bookkeeping. `refreshHmxManifestBridgeCounts` walks the whole
/// module and requires every bridge op to name a manifest record, and a record is
/// keyed by (function name, decision id); `rewriteLoop` erases the original
/// `hmx.unpack_acc`, so an outlined op without the attribute is a bridge site that
/// belongs to no record at all. Measured: a real 1024x512x64 Triton compile with
/// enableHmxVectorReadout=True failed with "HMX bridge operation has no explicit
/// hmx.decision_id" five times -- once per function the pass had added to the
/// module, because a nested `pm.addNestedPass<func::FuncOp>` pipeline re-runs
/// hmx-partition on each function the pass appends.
///
/// The value may be absent, and then nothing is set: a standalone invocation on
/// IR that never went through `matmul-to-hmx` has no manifest to attribute to,
/// and inventing an id there would be worse than having none.
static func::FuncOp getOrCreateReadout(ModuleOp module, MemRefType arType,
                                       MemRefType dstType, int64_t count,
                                       int64_t col, Attribute decisionId) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());

  // The name is picked by probe-then-create, so two concurrent sibling
  // invocations could pick the same free name and both create it: the probe
  // and the create are one critical section on the shared module-state mutex
  // (see declareRuntime). The body is built on the new function afterwards,
  // outside the lock, because nothing else can name it yet.
  std::string name = kReadoutFnPrefix.str();
  func::FuncOp fn;
  {
    std::lock_guard<std::mutex> guard(hmxModuleStateMutex());
    for (unsigned n = 1; module.lookupSymbol(name); ++n)
      name = (Twine(kReadoutFnPrefix) + "_" + Twine(n)).str();

    fn = func::FuncOp::create(
        builder, builder.getUnknownLoc(), name,
        builder.getFunctionType({arType, dstType, builder.getIndexType(),
                                 builder.getIndexType()},
                                {}));
  }
  fn.setPrivate();
  // Marks this as the executor's work function. Nothing calls it HERE -- the
  // `configure` handoff that would is emitted by HmxToLLVMPass, which runs after
  // convert-func-to-llvm and is the only pass that can take a symbol's address
  // (see rewriteFunction) -- so without a marker it would be unreferenced IR that
  // nothing tests. The attribute names the function; the pairing with its kernel
  // and the two memref types travel in the handoff record, because those are what
  // survive the conversions in between.
  //
  // It is ALSO what tells the manifest recount that this function's bridge ops
  // belong to another function's record; see refreshHmxManifestBridgeCounts.
  fn->setAttr(kHmxReadoutOutlinedAttr, builder.getUnitAttr());
  Block *entry = fn.addEntryBlock();

  OpBuilder body(builder.getContext());
  body.setInsertionPointToEnd(entry);
  Location loc = fn.getLoc();
  Value ar = entry->getArgument(0);
  Value dst = entry->getArgument(1);
  Value row0 = entry->getArgument(2);
  Value nrows = entry->getArgument(3);

  Value c0 = arith::ConstantIndexOp::create(body, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(body, loc, 1);
  Value colValue = arith::ConstantIndexOp::create(body, loc, col);
  IntegerAttr countAttr = body.getI64IntegerAttr(count);

  scf::ForOp loop = scf::ForOp::create(body, loc, c0, nrows, c1, ValueRange{});
  OpBuilder inner(builder.getContext());
  // Before the terminator `ForOp::create` already emitted: appending after it
  // would put the read-out outside the loop it is supposed to repeat.
  inner.setInsertionPoint(loop.getBody()->getTerminator());
  Value row = arith::AddIOp::create(inner, loc, row0, loop.getInductionVar());
  UnpackAccOp outlined = UnpackAccOp::create(
      inner, loc, TypeRange{dstType}, ar, dst, row, colValue, countAttr,
      IntegerAttr(), IntegerAttr(), IntegerAttr());
  if (decisionId)
    outlined->setAttr(kHmxDecisionIdAttr, decisionId);
  // The return belongs to the FUNCTION, so it is built with the outer builder --
  // building it with `inner` would close the loop body instead.
  func::ReturnOp::create(body, loc);
  return fn;
}

/// The one `HmxReadoutBatch` of this function: a 24-byte stack object, written
/// once for the invariant fields and per publish for the two that move.
///
/// Stack rather than VTCM: the object is live for the whole kernel and `drain`
/// runs before the function returns, so the engine thread's frame is guaranteed
/// to be there for the vector thread to read -- and it keeps the object out of
/// the VTCM accounting, the allocator and the residency rewrites entirely.
struct Descriptor {
  Value buffer;  ///< memref<6xi32>
  Value address; ///< i32 base address of `buffer`
};

static Descriptor buildDescriptor(OpBuilder &builder, Location loc,
                                  ModuleOp module, Value ar, Value dst,
                                  int64_t validRows, int64_t nCroutons) {
  Descriptor desc;
  Type i32 = builder.getI32Type();
  memref::AllocaOp alloca = memref::AllocaOp::create(
      builder, loc, MemRefType::get({kHmxReadoutBatchWords}, i32));
  desc.buffer = alloca.getResult();

  // A descriptor word is an i32 ADDRESS (the runtime ABI: `void *ar, *dst`,
  // HmxVectorExecutor.h), and an address is the memref's aligned pointer PLUS
  // its descriptor offset, element-scaled. That is the rule `asAddress`
  // applies on the LLVM side (HmxToLLVMPass.cpp:1476), spelled in the memref
  // dialect; the offset term is why a pack source that is a `reinterpret_cast`
  // with a per-iteration offset packs the right block there.
  //
  // WHY THIS IS NOT `extract_aligned_pointer_as_index` ALONE. That op returns
  // the ALIGNED pointer field, and a view into the middle of a buffer keeps
  // its offset in a SEPARATE field. Every span of a grid-tiled matmul stores
  // through such a view -- a `reinterpret_cast` of the output whose offset is
  // `m0*N + n0` -- so the aligned pointer is the SAME value for all of them.
  // Measured on the device (2048^3-B, 16 spans): all 16 spans wrote the
  // top-left 512x512 block, the other 15/16 of C kept the caller's zeros, and
  // rel = 0.96817 ~ sqrt(15/16) -- the exact signature of a span offset that
  // is missing rather than wrong. `expand-strided-metadata` folds the extract
  // through the `reinterpret_cast` later on the LLVM path and drops the offset
  // for good, so there is nothing downstream this pass could lean on.
  //
  // The offset is read through `extract_strided_metadata`, the one op that
  // exposes it as a value (the same idiom HexagonL2PrefetchPass uses to build
  // a fetch address), rather than by recognising how the view was spelled:
  // the producer does not know, and a subview, a `reinterpret_cast` and a
  // cast all end up here with the offset in the same field.
  auto addressOf = [&](Value buffer) {
    Value pointer = memref::ExtractAlignedPointerAsIndexOp::create(builder,
                                                                   loc, buffer);
    Value offset = memref::ExtractStridedMetadataOp::create(builder, loc,
                                                           buffer)
                       .getOffset();
    int64_t elemBytes =
        cast<MemRefType>(buffer.getType()).getElementTypeBitWidth() / 8;
    Value byteOffset = arith::MulIOp::create(
        builder, loc, offset,
        arith::ConstantIndexOp::create(builder, loc, elemBytes));
    Value absolute = arith::AddIOp::create(builder, loc, pointer, byteOffset);
    return arith::IndexCastOp::create(builder, loc, i32, absolute);
  };
  auto store = [&](int64_t field, Value value) {
    Value index = arith::ConstantIndexOp::create(builder, loc, field);
    memref::StoreOp::create(builder, loc, value, desc.buffer, ValueRange{index});
  };

  // `validRows` is the destination's own row count: the outlined read-out's leaf
  // derives its bound from the memref type it is handed, so for the shapes this
  // pass accepts the field is informational for the executor. `nCroutons` is
  // `acc_read`'s per-row count, which is the read-out's `count` attribute.
  store(kHmxReadoutValidRows,
        arith::ConstantIntOp::create(builder, loc, validRows, 32));
  store(kHmxReadoutNCroutons,
        arith::ConstantIntOp::create(builder, loc, nCroutons, 32));
  store(kHmxReadoutAr, addressOf(ar));
  store(kHmxReadoutDst, addressOf(dst));
  desc.address = addressOf(desc.buffer);

  // `publish` returns the number of batches it accepted. The call discards it
  // because the check needs the LLVM dialect's block structure, which does not
  // exist yet; HmxToLLVMPass turns every one of these calls into a
  // compare-and-trap, and the declaration must already have the ABI's i32 for
  // that to be possible.
  declareRuntime(module, kHmxExecPublishFn, {i32, i32}, {i32});
  return desc;
}

static void emitPublish(OpBuilder &builder, Location loc, ModuleOp module,
                        const Descriptor &desc, Value rowStart,
                        int64_t rowCount) {
  Type i32 = builder.getI32Type();
  auto store = [&](int64_t field, Value value) {
    Value index = arith::ConstantIndexOp::create(builder, loc, field);
    memref::StoreOp::create(builder, loc, value, desc.buffer, ValueRange{index});
  };
  store(kHmxReadoutRowStart,
        arith::IndexCastOp::create(builder, loc, i32, rowStart));
  store(kHmxReadoutRowCount,
        arith::ConstantIntOp::create(builder, loc, rowCount, 32));
  func::CallOp::create(
      builder, loc, module.lookupSymbol<func::FuncOp>(kHmxExecPublishFn),
      ValueRange{desc.address,
                 arith::ConstantIntOp::create(builder, loc, 1, 32)});
}

static void emitDrain(OpBuilder &builder, Location loc, ModuleOp module) {
  declareRuntime(module, kHmxExecDrainFn, {});
  func::CallOp::create(builder, loc,
                       module.lookupSymbol<func::FuncOp>(kHmxExecDrainFn),
                       ValueRange{});
}

/// Emit one matmul group's `configure` call with a placeholder function
/// pointer.
///
/// The real address cannot be spelled here: `llvm.mlir.addressof` rejects a
/// `func.func` symbol, and this pass runs before convert-func-to-llvm. So the
/// placeholder travels to the lowering (HmxToLLVMPass, wireConfigureCalls),
/// which pairs the engine's configure calls with the handoff records emitted
/// beside them and writes each entry point's address in. A placeholder of 0 is
/// safe even if the lowering never runs: the runtime's null-function check
/// returns -1 and the publish check traps -- a loud refusal, not a silent
/// wrong answer.
///
/// Placement is before the GROUP's first tile loop, not the function's entry
/// block: for every group but the first, this call is also the correctness
/// barrier. The runtime drains the ring before swapping the function pointer
/// (HmxVectorExecutor.cpp configure), so the previous group's in-flight
/// batches finish under their own read-out before this group's batches can
/// run at all. `configure` is idempotent, and one call per group is one call
/// per work function -- never per iteration.
static void emitConfigurePlaceholder(OpBuilder &builder, Location loc,
                                     ModuleOp module) {
  Type i32 = builder.getI32Type();
  func::CallOp::create(
      builder, loc,
      module.lookupSymbol<func::FuncOp>(kHmxExecConfigureFn),
      ValueRange{arith::ConstantIntOp::create(builder, loc, 0, 32),
                 arith::ConstantIntOp::create(builder, loc, 1, 32)});
}

/// Record which kernel's publishes belong to which outlined read-out, and the
/// two memref types the lowering needs to call it.
///
/// The producer is the only place that still holds those types: by the time a
/// pass can take a function's address, `convert-func-to-llvm` has already
/// replaced each memref argument with its components, so the outlined signature
/// no longer says which words belong to which buffer. See HmxReadoutHandoff.h.
static void recordHandoff(ModuleOp module, func::FuncOp engine,
                          func::FuncOp work, MemRefType arType,
                          MemRefType dstType) {
  // The handoff array is a read-modify-write of a module attribute, and
  // sibling function passes run concurrently: an unlocked append loses one
  // side's record. Same mutex, same reason, as every other module-state
  // mutation in this pass.
  std::lock_guard<std::mutex> guard(hmxModuleStateMutex());
  OpBuilder builder(module.getContext());
  auto existing = module->getAttrOfType<ArrayAttr>(kHmxReadoutHandoffsAttr);
  SmallVector<Attribute> records;
  if (existing)
    records.assign(existing.getValue().begin(), existing.getValue().end());
  records.push_back(builder.getDictionaryAttr(
      SmallVector<NamedAttribute>{
          builder.getNamedAttr(kHmxReadoutEngineField,
                               builder.getStringAttr(engine.getName())),
          builder.getNamedAttr(kHmxReadoutWorkField,
                               builder.getStringAttr(work.getName())),
          builder.getNamedAttr(kHmxReadoutArField, TypeAttr::get(arType)),
          builder.getNamedAttr(kHmxReadoutDstField, TypeAttr::get(dstType))}));
  module->setAttr(kHmxReadoutHandoffsAttr, builder.getArrayAttr(records));
}

struct HmxVectorReadoutPass
    : public mlir::hmx::impl::HmxVectorReadoutBase<HmxVectorReadoutPass> {
  using mlir::hmx::impl::HmxVectorReadoutBase<
      HmxVectorReadoutPass>::HmxVectorReadoutBase;

  void runOnOperation() override {
    // A batch size below one is an error, not a clamp: a clamped request would
    // look honoured when it is not, and G=1 measurably loses.
    if (batch < 1) {
      getOperation().emitError()
          // The explicit long long cast is load-bearing, not decoration: `batch` is
          // an int64_t, and streaming a plain int64_t into an mlir::Diagnostic
          // picks the `unsigned long` overload on this platform, which prints the
          // low byte as a raw character -- so 0 came out as a blank and -1 as a
          // replacement glyph. The cast pins the signed decimal spelling.
          << "hmx-readout-batch must be >= 1, got "
          << static_cast<long long>(batch)
          << " (G=1 measurably loses to the handoff cost; see "
             "bin/runtime/include/HmxVectorExecutor.h)";

      return signalPassFailure();
    }

    getOperation().walk([&](func::FuncOp function) {
      rewrite(function, batch);
    });
  }

private:
  void rewrite(func::FuncOp function, int64_t group) {
    if (function.isDeclaration())
      return;
    // This pass's own outlined read-outs are functions in the same module, and
    // a nested `func.func` pipeline re-runs the pass on everything the module
    // holds -- including them. Their AR is an argument rather than an
    // allocation, so matchReadout declines them anyway; the guard says so
    // without paying the walk, and keeps a future matchReadout change from
    // ever re-processing the pass's own output.
    if (function->hasAttr(kHmxReadoutOutlinedAttr))
      return;
    ModuleOp module = function->getParentOfType<ModuleOp>();
    DominanceInfo dominance(function);

    // Collect every candidate before touching anything. A half-rewritten
    // function is the worst possible outcome here, so the decision to touch the
    // IR is made first, in full.
    SmallVector<ReadoutMatch> matches;
    function.walk([&](scf::ForOp loop) {
      if (std::optional<ReadoutMatch> match = matchReadout(loop, dominance))
        matches.push_back(*match);
    });
    if (matches.empty())
      return;

    // One matmul's blocks share the AR array and the destination, and one
    // descriptor per matmul is the design ("no descriptor array is
    // needed"). A multi-matmul function -- flash attention's Q@K and P@V --
    // therefore holds one GROUP per AR array, and the groups are taken in
    // sequence: each gets its own descriptor, outlined read-out, handoff
    // record and `configure` call, and the runtime's drain-before-swap
    // (HmxVectorExecutor.cpp configure) is what makes the sequence sound.
    //
    // `function.walk` visits in program order, so the matches are numbered in
    // the order their loops execute and the groups form in first-occurrence
    // order of their AR array.
    SmallVector<SmallVector<unsigned>> groups;
    DenseMap<Value, unsigned> groupOfAr;
    for (unsigned i = 0; i < matches.size(); ++i) {
      auto [it, inserted] = groupOfAr.try_emplace(matches[i].ar, groups.size());
      if (inserted)
        groups.emplace_back();
      groups[it->second].push_back(i);
    }

    // The groups must run SEQUENTIALLY: every loop of an earlier group before
    // every loop of a later one. A group whose `configure` fires while an
    // earlier group still has publishes behind it would hand those publishes
    // to the wrong read-out -- the plausible-looking wrong answer this pass
    // exists to never ship. Matches are numbered in walk order, so the test is
    // that each group's indices form one contiguous run: a gap inside a group
    // is an interleave.
    {
      unsigned expected = 0;
      for (const auto &groupMatches : groups) {
        for (unsigned k = 0; k < groupMatches.size(); ++k, ++expected)
          if (groupMatches[k] != expected) {
            function.emitRemark()
                << "hmx-vector-readout declined: matmul groups interleave (AR"
                   " arrays alternate inside the function); a configure"
                   " between another group's publishes would swap the"
                   " executor's read-out mid-group";
            return;
          }
      }
    }

    // No group's loop may contain another group's loop either: a nested
    // group's `configure` fires per outer iteration, between the outer
    // group's publishes. Contiguity does not catch this -- `function.walk`
    // visits in POST-order, so the inner loop's match is numbered BEFORE the
    // outer one's and the runs still look contiguous -- so the ancestor
    // relation is checked on its own, in both directions.
    for (unsigned g = 0; g < groups.size(); ++g)
      for (unsigned h = g + 1; h < groups.size(); ++h)
        for (unsigned i : groups[g])
          for (unsigned j : groups[h])
            if (matches[i].loop->isAncestor(matches[j].loop) ||
                matches[j].loop->isAncestor(matches[i].loop)) {
              function.emitRemark()
                  << "hmx-vector-readout declined: one matmul group's tile"
                     " loop contains another's; the inner configure would"
                     " fire between the outer group's publishes";
              return;
            }

    // Within ONE matmul, every block still has to agree on the destination
    // and the decision id -- the same "decline, never repair" rule as before,
    // now per group instead of per function. A block that disagrees is a
    // malformed single matmul, and the function is declined whole rather than
    // given a second descriptor for one matmul.
    //
    // Absent is not disagreement. Standalone invocations on IR that never went
    // through `matmul-to-hmx` carry no id at all, and that has always worked, so
    // "none of them has one" leaves every one unset.
    for (const auto &groupMatches : groups) {
      Value dst = matches[groupMatches.front()].dst;
      Attribute decisionId =
          matches[groupMatches.front()].unpack->getAttr(kHmxDecisionIdAttr);
      for (unsigned i : groupMatches) {
        if (matches[i].dst != dst) {
          function.emitRemark()
              << "hmx-vector-readout declined: one matmul's blocks name"
                 " different destinations; the pass does not model a second"
                 " descriptor for one matmul";
          return;
        }
        if (matches[i].unpack->getAttr(kHmxDecisionIdAttr) != decisionId) {
          function.emitRemark()
              << "hmx-vector-readout declined: one matmul's read-outs name"
                 " different manifest decision ids, so the outlined read-out"
                 " could not be attributed";
          return;
        }
      }
    }

    OpBuilder builder(function.getContext());

    // `configure` names the work the vector thread runs, once per GROUP,
    // before its first tile loop -- see emitConfigurePlaceholder for why the
    // address is a placeholder here and for why the position, not just the
    // call, is load-bearing.
    //
    // The declarations are made here, so the symbols the linker must resolve
    // are recorded and the lit tests pin the exact ABI the executor sees.
    // `configure` is declared with the ABI's i32 result, which is what lets the
    // lowering re-declare it through `lookupOrCreateFn` (that helper rejects a
    // redefinition of a different type).
    declareRuntime(module, kHmxExecConfigureFn,
                   {builder.getI32Type(), builder.getI32Type()},
                   {builder.getI32Type()});
    declareRuntime(module, kHmxExecShutdownFn, {});

    // One descriptor per group, placed before its first tile loop so it
    // dominates every publish of that group, with the group's configure
    // placeholder, outlined read-out and handoff record beside it. Records are
    // appended in group order, which is the order the lowering pairs with the
    // engine's configure calls.
    SmallVector<Descriptor> descriptors;
    for (const auto &groupMatches : groups) {
      ReadoutMatch &first = matches[groupMatches.front()];

      builder.setInsertionPoint(first.loop);
      Descriptor desc = buildDescriptor(
          builder, first.loop.getLoc(), module, first.ar, first.dst,
          first.dstType.getDimSize(0), first.count);
      emitConfigurePlaceholder(builder, first.loop.getLoc(), module);

      Attribute decisionId = first.unpack->getAttr(kHmxDecisionIdAttr);
      func::FuncOp work = getOrCreateReadout(module, first.arType,
                                             first.dstType, first.count,
                                             first.col, decisionId);
      recordHandoff(module, function, work, first.arType, first.dstType);
      descriptors.push_back(desc);
    }

    // Deferred drain: drop the exit drain so the tail batch is waited for by
    // the NEXT call's configure() barrier instead (the executor drains before
    // swapping the function pointer, HmxVectorExecutor.cpp configure), and by
    // the wrapper after the last call. A decline, never a repair: it applies
    // only when the whole function holds exactly ONE readout loop, because the
    // per-loop drain is also the only thing standing between one loop's
    // batches and the NEXT loop's publishes -- the ring is 15 deep and a full
    // ring drops batches, which checkPublishReturns turns into a trap. A
    // multi-loop or multi-group function therefore keeps every drain and says
    // so, rather than silently shipping a deferral it cannot honour.
    const bool deferDrain =
        this->deferDrain && groups.size() == 1 && groups.front().size() == 1;
    if (this->deferDrain && !deferDrain)
      function.emitRemark()
          << "hmx-vector-readout defer-drain declined: function holds "
          << matches.size() << " readout loops in " << groups.size()
          << " matmul group(s) sharing one ring; keeping the per-loop drain"
             " (a removed barrier would let a later loop's publishes overflow"
             " the ring and drop batches)";

    for (unsigned g = 0; g < groups.size(); ++g)
      for (unsigned i : groups[g])
        rewriteLoop(builder, matches[i], descriptors[g], group, module,
                    deferDrain);
  }

  /// Replace one tile loop's read-out with a batched publish, and its tail with
  /// the tail batch plus the drain.
  ///
  /// Coverage, stated once so the code can be read against it. The loop runs
  /// 0..upper step 1 and the publish fires at the iterations where
  /// `(m + 1) % G == 0`, i.e. m = G-1, 2G-1, ..., `floor(upper/G)*G - 1`: exactly
  /// `upper/G` handoffs naming the rows 0 .. `(upper/G)*G - 1`. The tail batch
  /// names the rows `(upper/G)*G` .. `Mt-1`, count `Mt - (upper/G)*G`. The two
  /// ranges are adjacent and their union is every row, so each AR row is read out
  /// once and once only -- which is the whole reason this rewrite is allowed to
  /// exist.
  void rewriteLoop(OpBuilder &builder, ReadoutMatch &match,
                   const Descriptor &desc, int64_t group, ModuleOp module,
                   bool deferDrain) {
    Location loc = match.unpack.getLoc();
    Value iv = match.loop.getInductionVar();

    // --- in-loop: publish a full group every G-th iteration -------------------
    builder.setInsertionPoint(match.unpack);
    Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(builder, loc, 1);
    Value cGroup = arith::ConstantIndexOp::create(builder, loc, group);
    // `(m + 1) % G == 0`. Computed off the induction variable rather than off a
    // carried pending-count: the two are the same test, and this one leaves the
    // software-pipelined loop's iter_args alone.
    Value completed = arith::AddIOp::create(builder, loc, iv, c1);
    Value remainder = arith::RemSIOp::create(builder, loc, completed, cGroup);
    Value atBoundary = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, remainder, c0);

    scf::IfOp boundary =
        scf::IfOp::create(builder, loc, atBoundary, /*withElseRegion=*/false);
    OpBuilder inner(builder.getContext());
    // Before the terminator `IfOp::create` already emitted, not after it: a
    // second yield would be a second terminator and the block would not verify.
    inner.setInsertionPoint(boundary.getThenRegion().front().getTerminator());
    // At m = kG-1 the group is rows (kG-G) .. (kG-1), so the start is m - (G-1).
    Value start = iv;
    if (group > 1)
      start = arith::SubIOp::create(
          inner, loc, iv, arith::ConstantIndexOp::create(inner, loc, group - 1));
    emitPublish(inner, loc, module, desc, start, group);

    // --- drop the read-out this replaces, and the arithmetic only it needed ---
    Value rowOperand = match.unpack.getRow();
    Value colOperand = match.unpack.getCol();
    match.unpack->erase();
    eraseDeadDefiners(rowOperand);
    eraseDeadDefiners(colOperand);

    // --- tail: the rows the loop's groups did not reach -----------------------
    //
    // With no peeled epilogue and `Mt % G == 0` the tail is empty and there is
    // nothing to publish -- the in-loop groups already named every row. That is
    // the one case where the naive "`Mt % G != 0`" test is right. At depth 2 the
    // kernel peels a row, so the loop stops at `Mt-1` and the tail is never empty
    // even when `Mt` is a multiple of G (1024 rows at G=4: the loop covers rows
    // 0..27 and the tail names 28..31, the peeled row 31 included). Gating on
    // `Mt % G` alone would drop that row, which is the stale-bytes bug this pass
    // exists to avoid; so the test is on the computed tail range instead.
    //
    // `tailStart` is the FIRST row no in-loop group named, and that is exactly
    // `floor(upper / G) * G`: batch k covers `[(k-1)G, kG-1]` and batch k only
    // runs when `iv = kG-1 < upper`, so k reaches `floor(upper/G)` and the union
    // is `[0, floor(upper/G)*G - 1]`. The `+ 1` that used to be here skipped
    // row `floor(upper/G)` on EVERY shape, which is the exact failure this pass
    // exists to prevent and the one no symbol table or error message can show.
    //
    // Measured 2026-10-03 on the emitted IR, before the fix, with the shipped
    // `batch = 4`:
    //
    //   shape      Mt   loop  in-loop rows  tail EMITTED  must be    row lost
    //   1024x512x64 32  [0,31) 0..27       (29, 3)      (28, 4)    28
    //   grid BM=256  8  [0, 7) 0..3        ( 5, 3)      ( 4, 4)     4
    //   grid BM=64   2  [0, 2) (none)      ( 1, 1)      ( 0, 2)     0
    //
    // So the caller's output kept stale bytes for one 32-row band on every
    // kernel -- numerically wrong, with a clean compile and a healthy object.
    // The gate could not see it either: its row-coverage check recomputed the
    // formula above instead of reading the emitted `rowStart`, so it agreed with
    // the intent by construction and was green against the bug. That check now
    // reads the emitted constant (see `_published_rows` in
    // test_hmx_vector_readout_object_gate.py).
    int64_t tailStart = (match.upper / group) * group;
    int64_t tailCount = match.mTiles - tailStart;

    // The tail publish goes AFTER the loop, and the two anchors are NOT
    // interchangeable -- `setInsertionPoint(loop)` would put it BEFORE the loop,
    // which is the bug this comment replaces.
    //
    // With an epilogue the anchor is the epilogue's read-out: the peeled row is
    // written by that `hmx.acc_read`, which runs after the loop, so a publish
    // placed immediately after the loop would hand the vector thread a row nobody
    // has written yet.
    //
    // WITHOUT an epilogue there is nothing after the loop that writes a row, and
    // the anchor is therefore the loop itself -- set AFTER. Measured 2026-10-03:
    // anchoring at `setInsertionPoint(loop)` emitted the tail publish AND the
    // drain before the tile loop, so the in-loop groups' `publish` calls landed
    // after the last drain and the kernel returned with a batch in flight. Two
    // ways to reach it, both reachable with the shipped defaults: any
    // `pipeline-depth=1` shape (no pipelining, so no peel -- the whole-matrix
    // 1024x512x64 matmul at depth 1), and a grid-tiled kernel whose BLOCK_M is
    // small enough that `Mt <= pipeline-depth` caps the ring at depth 1.
    if (match.epilogue)
      builder.setInsertionPoint(match.epilogue.getOperation());
    else
      builder.setInsertionPointAfter(match.loop);
    if (tailCount > 0)
      emitPublish(builder, loc, module, desc,
                  arith::ConstantIndexOp::create(builder, loc, tailStart),
                  tailCount);

    // `drain` after the last publish -- whichever kind -- and before the AR array
    // is released, so the vector thread never reads freed VTCM and the caller
    // never sees the destination before it is written. It is emitted even when
    // the tail was empty, because the in-loop groups still need it, and it shares
    // the anchor above precisely so that "after the last publish" is a property
    // of the emission order rather than something to be re-derived.
    //
    // UNLESS deferDrain: then this call is the whole point of the option. The
    // barrier is not removed, it is MOVED -- the next call's configure() drains
    // before doing anything else (the engine function's entry block, so before
    // any allocation the next call could reuse the AR's pages with), and the
    // wrapper drains the last call outside the timed region. What that buys is
    // the overlap the exit drain forbids: the tail batch's consumer work runs
    // during the function epilogue (AR release, return) instead of after it.
    //
    // The AR-release ordering this comment used to guarantee -- "the vector
    // thread never reads freed VTCM" -- becomes a window rather than a
    // non-overlap: the AR is released while the consumer may still be reading
    // it. That is safe for exactly the sequential single-instance launch the
    // option documents, for two reasons that are properties of the runtime
    // rather than hopes: the release path (VtcmPool::Free / the BufferManager
    // free cache) keeps its bookkeeping OUT of the released block, so the
    // bytes survive the free; and nothing else allocates between the release
    // and the next call's configure-drain, because the device-side benchmark
    // loop is single-threaded and configure is the first thing the next call
    // does. A grid>1 launch (or any second publisher into the shared ring)
    // breaks both reasons and is outside the option's contract.
    if (!deferDrain)
      emitDrain(builder, loc, module);

    // The peel's read-out is now part of the tail batch: erasing it is what keeps
    // the peeled row from being read out twice. Erased LAST, because both the
    // tail publish and the drain are anchored at it.
    if (match.epilogue) {
      Value peeledRow = match.epilogue.getRow();
      match.epilogue->erase();
      eraseDeadDefiners(peeledRow);
    }
  }
};

} // namespace

namespace mlir {
namespace hmx {
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHmxVectorReadoutPass(const HmxVectorReadoutOptions &options) {
  return std::make_unique<HmxVectorReadoutPass>(options);
}
} // namespace hmx
} // namespace mlir
