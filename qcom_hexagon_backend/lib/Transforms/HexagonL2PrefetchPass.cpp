//===- HexagonL2PrefetchPass.cpp - L2 prefetch for streaming vector loops -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Insert `l2fetch` (the L2 prefetch engine, 2D block form) into the streaming
// vector loops the vectorizer emits, so the DDR stream is pulled into L2 ahead
// of the demand loads instead of through the per-core L2-miss path.
//
// Why (measured on device, exp/hmx/streaming_bw_probe, 2026-10-04, 3
// processes, 32 MiB/array f16 add -- the gap-table ceiling working set):
//
//   serial 1 vector/iteration ........ 10.9-13.4 GB/s (rd+wr)
//   2/4/8 vectors in flight .......... +-15%          (miss path, not MLP)
//   vmem vs vmemu (aligned vs not) ... identical      (alignment is not it)
//   dcfetch (L1 sector prefetch) ..... +22%
//   l2fetch distance 2 KiB ........... 29-32 GB/s
//   l2fetch distance 4 KiB ........... 36-41 GB/s
//   l2fetch distance 8 KiB ........... 42-47 GB/s   <- the knee
//   l2fetch distance 32 KiB .......... 42-47 GB/s   <- saturated
//
// The same traffic through the DMA engine is ~96 GB/s: the 3.5x gap of the
// streaming codegen (docs/results/gap-table-2026-10-04.md section 2) is the
// per-core miss path (~128 B per ~25 cycles single-threaded), and the L2
// prefetch engine is the mechanism that goes around it. This is the same
// pattern the handwritten reference uses in every streaming kernel
// (llama.cpp ggml-hexagon htp/hex-utils.h hex_l2fetch).
//
// What it matches: an innermost scf.for whose body reads slices of a dense,
// DDR-resident memref through vector.transfer_read at [0]. The loop may walk
// the source's linear element space in any of the three forms the pipeline
// actually produces:
//
//   * flat:      subview %src[%iv] [S] [1], S == step -- the elementwise
//                kernels (vec_add/silu/gelu) and the grid-strided program
//                slice, whose base offset is a runtime value read through
//                extract_strided_metadata;
//   * row slice: subview %src[%row, %iv] [1, S] of a rank-2 dense source --
//                the loop walks the contiguous (last) dimension while the row
//                offset is a constant or an enclosing loop's IV (linattn's
//                masked and rowsum loops);
//   * row walk:  subview %src[%iv, %c] [1, W] with W == the row stride -- the
//                loop walks whole contiguous rows (linattn's output norm).
//
// Unrolled bodies match when the copies at offsets %iv + m*(slice size) tile
// the step (the size divides the step and every m in [0, step/size) is
// present): the per-iteration reads then cover the loop's whole linear
// advance with no gaps. Reduction loops are in scope: a reduction accumulator
// does not keep the input stream from being sequential (linattn's rowsum
// reads its largest DDR stream through one).
//
// All three forms are one linear stream: the iteration's first read sits at
// source-element offset base + %iv*sigma (sigma = the source stride along the
// dimension the IV indexes), each iteration advances step*sigma elements, and
// a dense row-major source makes that advance contiguous across the inner
// loop's row wraps.
//
// What it emits, per distinct source memref:
//   - hoisted before the loop: extract_strided_metadata + the aligned
//     pointer, both pure metadata (the row forms also hoist the row-offset
//     terms of the base position);
//   - inside the loop, guarded by one fire per fetch block of stream
//     advance plus a room term, one `llvm.call @llvm.hexagon.Y5.l2fetch(ptr,
//     ctrl)` targeting the stream position `distance` bytes ahead, with a 2D
//     control word that fetches one contiguous block (width = stride =
//     block, height 1). The flat form keeps the original guard
//     `(iv - lb) % period == 0` (entry-relative, measured in IV units); the
//     row forms fire at the stream's first position inside each block-sized
//     window (`streamPos % block < advance`), because an entry-relative
//     period is vacuous in loops an outer loop re-enters with a fresh base
//     every row -- it would fire at every entry, block/advance times the
//     calibrated rate. The flat form measures the room against the loop's
//     own upper bound; the row forms measure it against the source's static
//     linear extent, because their stream crosses the inner loop's rows --
//     and a stream shorter than distance + block is declined outright,
//     since the guard could never fire in it. The room term keeps every
//     fetch inside a range the memref provably backs, and unlike a demand
//     load a prefetch has no fault handler contract to lean on.
//
// Every decline of a near-miss emits a remark. Slices no vector read
// consumes are out of scope and silent: the store side, scalar memref.load
// accumulators, and memref.copy sources (the DMA path owns those). So are
// loops without a read slice (HMX engine loops) and sources outside the
// default address space (VTCM is not behind the L2-miss path this pass
// works around).
//
// The two constants below are calibration results, not tunables: the
// distance was swept at 2/4/8/32 KiB and saturates at 8 KiB (the knee), and
// the block size is the 2 KiB the sweep ran with. If they ever need
// recalibrating, the probe is the instrument; a knob here would only invite
// tuning against shapes the probe never measured.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <mutex>

#define DEBUG_TYPE "hexagon-l2-prefetch"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_HEXAGONL2PREFETCH
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// One fetch per distinct source memref; the fetch target runs `distance`
// bytes ahead of the current iteration's read.
static constexpr int64_t kPrefetchDistanceBytes = 8192;
// The contiguous block one l2fetch pulls: 2 KiB at width = stride, height 1.
static constexpr int64_t kFetchBlockBytes = 2048;

// The Y5 (2D) l2fetch control word, 64-bit Rtt form (Hexagon V79 PRM, "L2
// cache prefetch instructions"): stride[47:32] | width[31:16] |
// height[15:0]; bit 48 is direction (0 = row-major). This is the layout the
// handwritten reference encodes (llama.cpp ggml-hexagon htp/hex-utils.h
// hex_l2fetch: Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height)))
// and the form the probe measured the 3.9x with -- verified in the probe's
// own disassembly (r23:22 = 0x800:0x08000001 for width=stride=2048,
// height=1).
//
// A subfield programmed as zero does NOT fetch: per the PRM it "cancels all
// pending prefetches by the calling thread". The first version of this pass
// packed the fields as stride | width<<32 | height<<48, which put ZERO in
// the width field: every fetch the kernel issued was a cancel, and the
// device A/B measured zero benefit from otherwise byte-verified fetch code
// (docs/results/l2-prefetch-2026-10-04.md). The layout is load-bearing;
// do not "simplify" it.
static int64_t l2fetchControlWord(int64_t width, int64_t stride,
                                  int64_t height) {
  return (stride << 32) | (width << 16) | height;
}

/// Constant integer value of an OpFoldResult, or nullopt.
static std::optional<int64_t> getConstIdx(OpFoldResult ofr) {
  return mlir::getConstantIntValue(ofr);
}

/// One matched read slice: the m-th copy of a source's per-iteration read
/// band. The copy's linear position in the source's element space is
/// base + (iv + m*ivDimSize) * sigma, where the base is the constant and
/// invariant-offset terms of the non-IV dimensions.
struct SliceSite {
  memref::SubViewOp subview;
  vector::TransferReadOp read;
  Value src;
  unsigned ivDim;      // the source dim whose offset carries this loop's IV
  int64_t ivDimSize;   // the subview's size in ivDim: the copy spacing
  int64_t m;           // copy index: the offset in ivDim is iv + m*ivDimSize
  int64_t sigma;       // source stride along ivDim, elements per IV unit
  int64_t width;       // the slice's contiguous width, in source elements
  int64_t elemBytes;
  int64_t extentElems; // the source's linear extent in elements, or -1 when
                       // only known dynamically (rank-1 dynamic size)
  int64_t baseConst;   // constant part of the linear base position
  // Invariant-offset terms: (source stride, offset value) pairs.
  SmallVector<std::pair<int64_t, Value>> baseDyn;
  // The flat form (rank-1 source, IV on its only dim, single copy): emitted
  // by the original path, guard and all, so its IR stays byte-identical.
  bool flatForm;
};

/// Returns true if forOp is innermost (no scf loop nested inside).
static bool isInnermostLoop(scf::ForOp forOp) {
  bool hasNested = false;
  forOp.getBody()->walk([&](Operation *op) {
    if (op == forOp.getOperation())
      return WalkResult::advance();
    if (isa<scf::ForOp, scf::ForallOp>(op)) {
      hasNested = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !hasNested;
}

/// True when the value cannot change across the loop's iterations: defined
/// outside the loop entirely. The loop's own block arguments (the IV and the
/// iter_args) are loop-carried and therefore not invariant.
static bool isLoopInvariant(Value v, scf::ForOp forOp) {
  if (auto blockArg = dyn_cast<BlockArgument>(v))
    return !forOp->isAncestor(blockArg.getOwner()->getParentOp());
  Operation *def = v.getDefiningOp();
  return def && !forOp->isAncestor(def);
}

/// A loop is a candidate when its body slices a memref that a vector read
/// consumes: that is the vectorizer's streaming form, in any of its shapes.
/// Candidates get the full analysis and remarks for every failed condition;
/// everything else is out of scope and stays silent.
static bool isCandidateLoop(scf::ForOp forOp) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto subview = dyn_cast<memref::SubViewOp>(op);
    if (!subview)
      continue;
    bool anyRead = llvm::any_of(subview.getResult().getUsers(),
                                [](Operation *u) {
                                  return isa<vector::TransferReadOp>(u);
                                });
    if (anyRead)
      return true;
  }
  return false;
}

/// Checks one subview against the slice form. Emits the remark and returns
/// nullopt on every failure: no silent near-misses -- except a slice no read
/// consumes at all (the store side of an elementwise kernel, a scalar
/// accumulator, a memref.copy source): those are out of scope, not near
/// misses, and remarking on every healthy kernel's output stream would bury
/// the reports that matter.
static std::optional<SliceSite> matchSite(scf::ForOp forOp,
                                          memref::SubViewOp subview,
                                          DominanceInfo &dom) {
  auto decline = [&](const Twine &reason) {
    subview.emitRemark("hexagon-l2-prefetch declined: ") << reason;
    return std::optional<SliceSite>();
  };

  // A slice nothing reads is out of scope: prefetch only moves lines for
  // demand loads.
  bool anyRead = llvm::any_of(subview.getResult().getUsers(), [](Operation *u) {
    return isa<vector::TransferReadOp>(u);
  });
  if (!anyRead)
    return std::nullopt;

  // The slice result is the vectorizer's streaming shape: a rank-1
  // unit-stride view. Rank-1 sources slice down to it directly; rank-2
  // sources rank-reduce their size-1 row dim, so a row slice and a flat
  // slice look the same to the consumer.
  auto resTy = dyn_cast<MemRefType>(subview.getResult().getType());
  if (!resTy || resTy.getRank() != 1)
    return decline("slice result is not rank-1");
  SmallVector<int64_t, 1> resStrides;
  int64_t resOffset;
  if (failed(resTy.getStridesAndOffset(resStrides, resOffset)) ||
      resStrides.size() != 1 || resStrides[0] != 1)
    return decline("slice result is not unit-stride");
  if (resTy.isDynamicDim(0))
    return decline("slice size is not static");
  int64_t width = resTy.getDimSize(0);

  // At least one consumer reads the slice as the iteration's vector, whole
  // and unmasked, at [0], in bounds. Other consumers of the same slice are
  // harmless: the fetch only moves lines, and the address stream is the
  // same.
  vector::TransferReadOp read;
  for (Operation *user : subview.getResult().getUsers()) {
    auto candidate = dyn_cast<vector::TransferReadOp>(user);
    if (!candidate || candidate.getBase() != subview.getResult())
      continue;
    if (candidate.getMask())
      return decline("transfer_read is masked");
    auto indices = candidate.getIndices();
    auto pos = indices.size() == 1
                   ? mlir::getConstantIntValue(indices[0])
                   : std::nullopt;
    if (!pos || *pos != 0)
      return decline("transfer_read position is not [0]");
    for (Attribute b : candidate.getInBounds())
      if (!cast<BoolAttr>(b).getValue())
        return decline("transfer_read is not in-bounds");
    auto vecTy = dyn_cast<VectorType>(candidate.getResult().getType());
    if (!vecTy || vecTy.getRank() != 1 || vecTy.getDimSize(0) != width)
      return decline("transfer vector is not the slice width");
    if (vecTy.getElementType() != resTy.getElementType())
      return decline("transfer element type differs from the source");
    read = candidate;
    break;
  }
  if (!read)
    return decline("no vector.transfer_read of the slice at [0]");

  // The source: a ranked memref in the default (DDR) address space, defined
  // outside the loop. VTCM and every other space is not behind the L2-miss
  // path this pass works around: fetching it is wasted fetch slots at best.
  Value src = subview.getSource();
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy)
    return decline("source is not a ranked memref");
  int memSpace = 0;
  if (!hexagon::isMemorySpaceIntTypeOrDefault(srcTy, memSpace) ||
      memSpace != hexagon::DEFAULT_DDR_ADDRESS_SPACE)
    return decline("source is not in the default (DDR) address space");
  if (!dom.dominates(src, forOp.getOperation()))
    return decline("source does not dominate the loop");

  unsigned rank = srcTy.getRank();
  SmallVector<int64_t, 4> strides;
  int64_t srcOffset;
  if (failed(srcTy.getStridesAndOffset(strides, srcOffset)) ||
      strides.size() != rank)
    return decline("source layout is not strided");
  for (int64_t s : strides)
    if (ShapedType::isDynamic(s) || s < 1)
      return decline("source is not a dense row-major memref");

  // A rank-1 source is dense iff its stride is 1, which the unit-stride
  // result already proved. Rank-2 and up must be statically dense row-major
  // (stride[d] == stride[d+1] * size[d+1], last stride 1): then the whole
  // source is one contiguous block, every byte of its linear extent is
  // backed, and a fetch anywhere inside that extent is as safe as a demand
  // load -- including across the row gaps a padded (non-dense) layout would
  // leave unprovable.
  int64_t extentElems = -1;
  if (rank == 1) {
    if (!ShapedType::isDynamic(srcTy.getDimSize(0)))
      extentElems = srcTy.getDimSize(0);
  } else {
    if (strides[rank - 1] != 1)
      return decline("source is not a dense row-major memref");
    int64_t extent = 1;
    for (unsigned d = 0; d < rank; ++d) {
      if (ShapedType::isDynamic(srcTy.getDimSize(d)))
        return decline("source is not a dense row-major memref");
      extent *= srcTy.getDimSize(d);
    }
    for (unsigned d = 0; d + 1 < rank; ++d)
      if (strides[d] != strides[d + 1] * srcTy.getDimSize(d + 1))
        return decline("source is not a dense row-major memref");
    extentElems = extent;
  }

  // Decompose the offsets. Every dim contributes off * stride to the linear
  // position; exactly one dim (the IV dim) carries the loop's induction
  // variable, as iv itself or as the m-th unrolled copy iv + m*(dim size).
  Value iv = forOp.getInductionVar();
  auto offsets = subview.getMixedOffsets();
  auto sizes = subview.getMixedSizes();
  if (offsets.size() != rank || sizes.size() != rank)
    return decline("subview does not match its source rank");

  std::optional<unsigned> ivDim;
  int64_t m = 0;
  int64_t baseConst = 0;
  SmallVector<std::pair<int64_t, Value>> baseDyn;
  auto setIvDim = [&](unsigned d, int64_t copy) -> bool {
    if (ivDim)
      return false;
    ivDim = d;
    m = copy;
    return true;
  };
  for (unsigned d = 0; d < rank; ++d) {
    int64_t stride = strides[d];
    if (auto c = getConstIdx(offsets[d])) {
      baseConst += *c * stride;
      continue;
    }
    Value v = cast<Value>(offsets[d]);
    if (v == iv) {
      if (!setIvDim(d, 0))
        return decline("offset carries the induction variable in more than "
                       "one dimension");
      continue;
    }
    // The m-th unrolled copy: iv plus a constant. The vectorizer's unroller
    // advances each copy's offset by the slice size, so the constant must be
    // a whole number of slice sizes for the copies to tile the step.
    if (auto add = v.getDefiningOp<arith::AddIOp>()) {
      Value other = nullptr;
      if (add.getLhs() == iv)
        other = add.getRhs();
      else if (add.getRhs() == iv)
        other = add.getLhs();
      auto c = other ? getConstIdx(OpFoldResult(other)) : std::nullopt;
      if (c) {
        auto sizeOpt = getConstIdx(sizes[d]);
        if (!sizeOpt || *sizeOpt < 1 || *c % *sizeOpt != 0)
          return decline("unrolled copy offset does not tile the slice size");
        if (!setIvDim(d, *c / *sizeOpt))
          return decline("offset carries the induction variable in more than "
                         "one dimension");
        continue;
      }
    }
    // Anything else must be loop-invariant (a constant row base or an
    // enclosing loop's IV): it fixes the stream's base position. A value
    // defined inside the loop that is not the recognized copy form varies
    // per iteration in a way the linear stream cannot express.
    if (!isLoopInvariant(v, forOp))
      return decline("slice offset is not the induction variable plus a "
                     "constant");
    baseDyn.emplace_back(stride, v);
  }
  if (!ivDim)
    return decline("slice offset does not depend on the induction variable");

  // The copy spacing along the IV dim, and the source's stride there: the
  // two facts the stream's advance is made of.
  auto ivDimSizeOpt = getConstIdx(sizes[*ivDim]);
  if (!ivDimSizeOpt || *ivDimSizeOpt < 1)
    return decline("slice size is not static");

  Type elemTy = srcTy.getElementType();
  if (elemTy.getIntOrFloatBitWidth() % 8 != 0)
    return decline("sub-byte element type");
  int64_t elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

  return SliceSite{subview,      read,
                   src,          *ivDim,
                   *ivDimSizeOpt, m,
                   strides[*ivDim], width,
                   elemBytes,    extentElems,
                   baseConst,    std::move(baseDyn),
                   /*flatForm=*/rank == 1 && *ivDim == 0 &&
                       strides[*ivDim] == 1};
}

/// The module-level declaration of the l2fetch intrinsic. Like every
/// module-state mutation in this tree, the check-then-insert runs under the
/// shared module-state mutex: the pass manager runs sibling functions
/// concurrently, and a duplicate symbol is a verifier error no single
/// function fixture can hit (the read-out pass hit it).
static LLVM::LLVMFuncOp getL2FetchFn(ModuleOp module) {
  const char *kName = "llvm.hexagon.Y5.l2fetch";
  std::lock_guard<std::mutex> guard(mlir::hmx::hmxModuleStateMutex());
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(kName))
    return existing;
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto fnTy = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(builder.getContext()),
      {ptrTy, builder.getI64Type()},
      /*isVarArg=*/false);
  auto fn = LLVM::LLVMFuncOp::create(builder, module.getLoc(), kName, fnTy);
  // Private for the same reason the outlined read-out is: the symbol is not
  // part of the kernel's ABI and nothing resolves it from outside.
  fn.setVisibility(SymbolTable::Visibility::Private);
  return fn;
}

/// Emits the fetches for the flat form: the hoisted metadata, one shared
/// guard measured against the loop's own upper bound, and one l2fetch per
/// distinct source memref. This is the original emitter, kept verbatim so
/// the flat form's IR stays byte-identical.
static void emitFetches(scf::ForOp forOp, IRRewriter &rewriter,
                        ArrayRef<Value> srcs, int64_t step, int64_t elemBytes,
                        LLVM::LLVMFuncOp l2fetch) {
  Location loc = forOp.getLoc();
  Value iv = forOp.getInductionVar();
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();

  int64_t vecBytes = step * elemBytes;
  int64_t blockIters = kFetchBlockBytes / vecBytes;
  int64_t distIters = kPrefetchDistanceBytes / vecBytes;
  int64_t periodElems = blockIters * step;
  int64_t distElems = distIters * step;
  int64_t needElems = (distIters + blockIters) * step;

  // Hoisted: the buffer pointer and the source offset, in i64 for the
  // address arithmetic below. Both are pure metadata ops.
  struct Base {
    Value ptr, off;
  };
  SmallVector<Base> bases;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp);
    auto i64Ty = rewriter.getI64Type();
    for (Value src : srcs) {
      auto meta =
          memref::ExtractStridedMetadataOp::create(rewriter, loc, src);
      Value ptr = memref::ExtractAlignedPointerAsIndexOp::create(
          rewriter, loc, meta.getBaseBuffer());
      bases.push_back(
          {arith::IndexCastOp::create(rewriter, loc, i64Ty, ptr),
           arith::IndexCastOp::create(rewriter, loc, i64Ty, meta.getOffset())});
    }
  }

  // The guard, at the top of the body: one fetch point per period, and only
  // while the whole fetch block (distance ahead) is inside the stream.
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value j = arith::SubIOp::create(rewriter, loc, iv, lb);
  Value period = arith::ConstantIndexOp::create(rewriter, loc, periodElems);
  Value rem = arith::RemSIOp::create(rewriter, loc, j, period);
  Value atBoundary = arith::CmpIOp::create(rewriter, loc,
                                           arith::CmpIPredicate::eq, rem, c0);
  Value room = arith::SubIOp::create(rewriter, loc, ub, iv);
  Value need = arith::ConstantIndexOp::create(rewriter, loc, needElems);
  Value roomy = arith::CmpIOp::create(rewriter, loc,
                                      arith::CmpIPredicate::sge, room, need);
  Value cond = arith::AndIOp::create(rewriter, loc, atBoundary, roomy);

  // The fetches: one call per source, inside the guard.
  scf::IfOp ifOp = scf::IfOp::create(rewriter, loc, cond,
                                     /*withElseRegion=*/false);
  OpBuilder inner(rewriter.getContext());
  // Before the terminator IfOp::create already emitted, not after it: a
  // second yield would be a second terminator and the block would not verify.
  inner.setInsertionPoint(ifOp.getThenRegion().front().getTerminator());
  auto i64Ty = inner.getI64Type();
  Value ivI64 = arith::IndexCastOp::create(inner, loc, i64Ty, iv);
  Value cElemBytes =
      arith::ConstantIntOp::create(inner, loc, i64Ty, elemBytes);
  Value cDist =
      arith::ConstantIntOp::create(inner, loc, i64Ty, distElems * elemBytes);
  Value ctrl = arith::ConstantIntOp::create(
      inner, loc, i64Ty,
      l2fetchControlWord(kFetchBlockBytes, kFetchBlockBytes, 1));
  auto ptrTy = LLVM::LLVMPointerType::get(inner.getContext());
  for (const Base &base : bases) {
    // (srcOffset + iv) elements -> bytes, plus the distance, plus the buffer.
    Value elems = arith::AddIOp::create(inner, loc, base.off, ivI64);
    Value byteOff = arith::MulIOp::create(inner, loc, elems, cElemBytes);
    byteOff = arith::AddIOp::create(inner, loc, byteOff, cDist);
    Value addr = arith::AddIOp::create(inner, loc, base.ptr, byteOff);
    Value ptr = LLVM::IntToPtrOp::create(inner, loc, ptrTy, addr);
    LLVM::CallOp::create(inner, loc, l2fetch, ValueRange{ptr, ctrl});
  }
}

/// Emits the fetch for one linear (row-slice / row-walk / unrolled) stream.
/// The guard is per source: both the fire discipline and the room depend on
/// the source's own stride and extent. The room is measured against the
/// source's static linear extent -- the stream crosses the inner loop's rows,
/// so the loop's own upper bound would cut it short by all but the current
/// row.
static void emitLinearFetches(scf::ForOp forOp, IRRewriter &rewriter,
                              const SliceSite &site, int64_t step,
                              LLVM::LLVMFuncOp l2fetch) {
  Location loc = forOp.getLoc();
  Value iv = forOp.getInductionVar();

  int64_t elemBytes = site.elemBytes;
  int64_t blockElems = kFetchBlockBytes / elemBytes;
  int64_t distElems = kPrefetchDistanceBytes / elemBytes;
  int64_t needElems = distElems + blockElems;
  // The stream's advance per loop iteration, in source elements.
  int64_t advance = step * site.sigma;

  // Hoisted before the loop: the source metadata (base pointer, offset) and
  // the invariant terms of the base position. All pure metadata and index
  // arithmetic; the offset values dominate the loop or the IR would not be
  // valid.
  Value ptrI64, offI64, base;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp);
    auto i64Ty = rewriter.getI64Type();
    auto meta =
        memref::ExtractStridedMetadataOp::create(rewriter, loc, site.src);
    Value ptr = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, meta.getBaseBuffer());
    ptrI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, ptr);
    offI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, meta.getOffset());
    if (site.baseConst != 0)
      base = arith::ConstantIndexOp::create(rewriter, loc, site.baseConst);
    for (auto [stride, v] : site.baseDyn) {
      Value term = arith::MulIOp::create(
          rewriter, loc, v, arith::ConstantIndexOp::create(rewriter, loc,
                                                           stride));
      base = base ? Value(arith::AddIOp::create(rewriter, loc, base, term))
                  : term;
    }
  }

  // The guard, at the top of the body: fire at the stream's first position
  // inside each block-sized window, and only while the whole fetch block
  // (distance ahead) stays inside the source's linear extent.
  //
  // The fire discipline is on the STREAM POSITION (lin), not on the IV: an
  // IV-unit period is entry-relative (iv restarts at lb every time an outer
  // loop re-enters this one), and the base terms carry an advance the IV
  // cannot see. When this loop's own span is shorter than the period -- the
  // masked row-slice loops re-entered per row, or a fully unrolled inner
  // loop that runs one iteration per row -- an entry-relative period fires
  // at every entry, up to block/advance times the calibrated rate. The
  // window-entry form fires exactly once per block of stream advance no
  // matter which loop carries it: with advance | blockElems (the caller
  // checked), the visited positions land in the entry zone [k*blockElems,
  // k*blockElems + advance) of every window exactly once, so the fetch
  // blocks abut and tile [distance, extent). The one asymmetry: a stream
  // that starts mid-window (a runtime base off the block lattice) delays
  // the first fire by less than one block -- a cold prefix bounded by 2 KiB
  // against the 8 KiB distance, where the entry-relative form had none.
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value lin = site.sigma == 1
                  ? Value(iv)
                  : arith::MulIOp::create(
                        rewriter, loc, iv,
                        arith::ConstantIndexOp::create(rewriter, loc,
                                                        site.sigma));
  if (base)
    lin = arith::AddIOp::create(rewriter, loc, base, lin);
  Value rem = arith::RemSIOp::create(
      rewriter, loc, lin,
      arith::ConstantIndexOp::create(rewriter, loc, blockElems));
  Value atWindow = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, rem,
      arith::ConstantIndexOp::create(rewriter, loc, advance));
  Value room = arith::SubIOp::create(
      rewriter, loc,
      arith::ConstantIndexOp::create(rewriter, loc, site.extentElems), lin);
  Value need = arith::ConstantIndexOp::create(rewriter, loc, needElems);
  Value roomy = arith::CmpIOp::create(rewriter, loc,
                                      arith::CmpIPredicate::sge, room, need);
  Value cond = arith::AndIOp::create(rewriter, loc, atWindow, roomy);

  // The fetch: the stream position `distance` bytes ahead, one contiguous
  // block. Same control word as the flat form: the stream is contiguous by
  // the tiling predicate, so a contiguous block is the right shape.
  scf::IfOp ifOp = scf::IfOp::create(rewriter, loc, cond,
                                     /*withElseRegion=*/false);
  OpBuilder inner(rewriter.getContext());
  inner.setInsertionPoint(ifOp.getThenRegion().front().getTerminator());
  auto i64Ty = inner.getI64Type();
  Value linI64 = arith::IndexCastOp::create(inner, loc, i64Ty, lin);
  Value elems = arith::AddIOp::create(inner, loc, offI64, linI64);
  Value byteOff = arith::MulIOp::create(
      inner, loc, elems, arith::ConstantIntOp::create(inner, loc, i64Ty,
                                                      elemBytes));
  byteOff = arith::AddIOp::create(
      inner, loc, byteOff,
      arith::ConstantIntOp::create(inner, loc, i64Ty,
                                   kPrefetchDistanceBytes));
  Value addr = arith::AddIOp::create(inner, loc, ptrI64, byteOff);
  Value ptr = LLVM::IntToPtrOp::create(
      inner, loc, LLVM::LLVMPointerType::get(inner.getContext()), addr);
  Value ctrl = arith::ConstantIntOp::create(
      inner, loc, i64Ty,
      l2fetchControlWord(kFetchBlockBytes, kFetchBlockBytes, 1));
  LLVM::CallOp::create(inner, loc, l2fetch, ValueRange{ptr, ctrl});
}

struct HexagonL2PrefetchPass
    : public ::impl::HexagonL2PrefetchBase<HexagonL2PrefetchPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    mlir::LLVM::LLVMDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Collect first, mutate after: the emitter inserts ops around and inside
    // the loops, and a walk that mutates skips what it moves.
    SmallVector<scf::ForOp> loops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (isInnermostLoop(forOp) && isCandidateLoop(forOp))
        loops.push_back(forOp);
    });
    if (loops.empty())
      return;

    ModuleOp module = funcOp->getParentOfType<ModuleOp>();
    LLVM::LLVMFuncOp l2fetch = getL2FetchFn(module);
    DominanceInfo dom(funcOp);
    IRRewriter rewriter(context);

    for (scf::ForOp forOp : loops) {
      // The step must be a positive constant: the stream's per-iteration
      // advance (and the flat form's fire period) are element counts
      // derived from it.
      auto stepOpt = mlir::getConstantIntValue(forOp.getStep());
      if (!stepOpt || *stepOpt <= 0) {
        forOp.emitRemark("hexagon-l2-prefetch declined: step is not a "
                         "positive constant");
        continue;
      }
      int64_t step = *stepOpt;

      // One site per subview, declined by remark per site; the rest still
      // fetch. Mixed element widths in one loop are declined: the fire
      // window is a loop-level fact. The vectorizer emits uniform widths.
      SmallVector<SliceSite> sites;
      std::optional<int64_t> elemBytes;
      for (Operation &op : forOp.getBody()->without_terminator()) {
        auto subview = dyn_cast<memref::SubViewOp>(op);
        if (!subview)
          continue;
        std::optional<SliceSite> site = matchSite(forOp, subview, dom);
        if (!site)
          continue;
        if (elemBytes && *elemBytes != site->elemBytes) {
          subview.emitRemark("hexagon-l2-prefetch declined: mixed element "
                             "widths in one loop");
          continue;
        }
        elemBytes = site->elemBytes;
        sites.push_back(std::move(*site));
      }
      if (sites.empty() || !elemBytes)
        continue;

      // One stream per distinct source: the copies of one source share the
      // stream, so one fetch covers them all.
      SmallVector<std::pair<Value, SmallVector<SliceSite>>> streams;
      for (SliceSite &site : sites) {
        auto *stream =
            llvm::find_if(streams, [&](auto &s) { return s.first == site.src; });
        if (stream == streams.end()) {
          streams.emplace_back();
          streams.back().first = site.src;
          stream = &streams.back();
        }
        stream->second.push_back(std::move(site));
      }

      SetVector<Value> flatSrcs;
      int64_t fetched = 0;
      for (auto &[src, group] : streams) {
        SliceSite &s0 = group.front();

        // The copies must agree on everything but their index: one stream,
        // one base, one advance.
        bool consistent = true;
        for (const SliceSite &s : llvm::drop_begin(group)) {
          if (s.ivDim != s0.ivDim || s.ivDimSize != s0.ivDimSize ||
              s.sigma != s0.sigma || s.baseConst != s0.baseConst ||
              s.baseDyn != s0.baseDyn) {
            s.subview->emitRemark("hexagon-l2-prefetch declined: another "
                                 "slice of this source does not share the "
                                 "stream's base");
            consistent = false;
            break;
          }
        }
        if (!consistent)
          continue;

        // Unroll tiling: the slice size divides the step and the copies at
        // iv + m*size for every m in [0, step/size) are all present, so the
        // per-iteration reads cover the loop's whole linear advance.
        if (step % s0.ivDimSize != 0) {
          forOp.emitRemark("hexagon-l2-prefetch declined: slice size does "
                           "not divide the loop step");
          continue;
        }
        int64_t copies = step / s0.ivDimSize;
        SmallVector<bool> seen(copies, false);
        bool inRange = true;
        for (const SliceSite &s : group) {
          if (s.m < 0 || s.m >= copies) {
            inRange = false;
            break;
          }
          seen[s.m] = true;
        }
        if (!inRange || !llvm::all_of(seen, [](bool b) { return b; })) {
          forOp.emitRemark("hexagon-l2-prefetch declined: copies of this "
                           "source do not tile the loop step");
          continue;
        }

        // The slice's contiguous width must equal the copy spacing scaled by
        // the source stride: adjacent copies then abut, and the per-iteration
        // band is gapless. A slice narrower than the row it walks leaves the
        // stream strided, which the calibration never measured.
        if (s0.width != s0.ivDimSize * s0.sigma) {
          forOp.emitRemark("hexagon-l2-prefetch declined: slice width does "
                           "not cover the loop's per-iteration advance");
          continue;
        }

        // The block and the distance must be whole numbers of the loop's
        // linear advance; anything else has no calibrated meaning.
        int64_t advanceBytes = step * s0.sigma * s0.elemBytes;
        if (kFetchBlockBytes % advanceBytes != 0 ||
            kPrefetchDistanceBytes % advanceBytes != 0) {
          forOp.emitRemark("hexagon-l2-prefetch declined: iteration advance "
                           "does not divide the fetch block or distance");
          continue;
        }

        // The flat form keeps the original path and its loop-bound guard,
        // byte-identical.
        if (s0.flatForm && copies == 1) {
          flatSrcs.insert(src);
          fetched++;
          continue;
        }

        // The linear guard needs the source's static extent; a stream shorter
        // than distance + block can never fire it, so it is declined outright
        // instead of carrying dead guard arithmetic.
        if (s0.extentElems < 0) {
          forOp.emitRemark("hexagon-l2-prefetch declined: source size is not "
                           "static (the stream-extent guard needs it)");
          continue;
        }
        int64_t needElems =
            (kPrefetchDistanceBytes + kFetchBlockBytes) / s0.elemBytes;
        if (s0.extentElems < needElems) {
          forOp.emitRemark("hexagon-l2-prefetch declined: stream is shorter "
                           "than the fetch distance plus block");
          continue;
        }

        emitLinearFetches(forOp, rewriter, s0, step, l2fetch);
        fetched++;
      }

      if (!flatSrcs.empty())
        emitFetches(forOp, rewriter, flatSrcs.getArrayRef(), step, *elemBytes,
                    l2fetch);
      if (fetched > 0) {
        forOp.emitRemark("hexagon-l2-prefetch: fetching ")
            << fetched << " stream(s), distance " << kPrefetchDistanceBytes
            << " B, block " << kFetchBlockBytes << " B";
        DBG("prefetched " << fetched << " stream(s) in loop at "
                          << forOp.getLoc());
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexagon::createHexagonL2PrefetchPass() {
  return std::make_unique<HexagonL2PrefetchPass>();
}
