//===- HexagonL2PrefetchPass.cpp - L2 prefetch for streaming vector loops -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Insert `l2fetch` (the L2 prefetch engine, 2D block form) into the flat
// streaming vector loops the vectorizer emits for elementwise kernels, so the
// DDR stream is pulled into L2 ahead of the demand loads instead of through
// the per-core L2-miss path.
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
// What it matches: the vectorizer's slice loop -- an innermost scf.for whose
// body reads 1-D subviews `subview %src[%iv] [S] [1]` through
// vector.transfer_read at [0], one vector per iteration (S == step). The
// source may be any rank-1 stride-1 memref defined outside the loop,
// including one with a dynamic offset (a grid-strided program slice): its
// buffer and offset are read through extract_strided_metadata at runtime.
//
// What it emits, per distinct source memref:
//   - hoisted before the loop: extract_strided_metadata + the aligned
//     pointer, both pure metadata;
//   - inside the loop, guarded by `(iv - lb) % period == 0 && ub - iv >=
//     need`, one `llvm.call @llvm.hexagon.Y5.l2fetch(ptr, ctrl)` targeting
//     the stream position `distance` bytes ahead, with a 2D control word
//     that fetches one contiguous block (width = stride = block, height 1).
//     The room term keeps every fetch inside the loop's own stream: a fetch
//     past the end is not provably mapped, and unlike a demand load a
//     prefetch has no fault handler contract to lean on. The guard fires
//     once per `block / vectorBytes` iterations, so the issue cost is
//     ~0.125 slots per vector at the calibrated sizes.
//
// Every decline of a near-miss emits a remark; loops without the streaming
// shape (HMX engine loops, reductions, ...) are out of scope and silent.
//
// The two constants below are calibration results, not tunables: the
// distance was swept at 2/4/8/32 KiB and saturates at 8 KiB (the knee), and
// the block size is the 2 KiB the sweep ran with. If they ever need
// recalibrating, the probe is the instrument; a knob here would only invite
// tuning against shapes the probe never measured.
//
//===----------------------------------------------------------------------===//

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

/// A qualifying streaming read inside one loop iteration.
struct StreamSite {
  memref::SubViewOp subview;
  vector::TransferReadOp read;
  int64_t elemBytes;
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

/// A loop is a candidate when its body slices a 1-D view at the induction
/// variable: that is the vectorizer's streaming form. Candidates get the full
/// analysis and remarks for every failed condition; everything else is out of
/// scope and stays silent.
static bool isCandidateLoop(scf::ForOp forOp) {
  Value iv = forOp.getInductionVar();
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto subview = dyn_cast<memref::SubViewOp>(op);
    if (!subview)
      continue;
    auto offsets = subview.getMixedOffsets();
    if (offsets.size() == 1 && isa<Value>(offsets[0]) &&
        cast<Value>(offsets[0]) == iv)
      return true;
  }
  return false;
}

/// Checks one subview against the slice form. Emits the remark and returns
/// nullopt on every failure: no silent near-misses -- except a slice no read
/// consumes at all (the store side of an elementwise kernel): that is out of
/// scope, not a near miss, and remarking on every healthy kernel's output
/// stream would bury the reports that matter.
static std::optional<StreamSite> matchSite(scf::ForOp forOp,
                                           memref::SubViewOp subview,
                                           int64_t step,
                                           DominanceInfo &dom) {
  auto decline = [&](const Twine &reason) {
    subview.emitRemark("hexagon-l2-prefetch declined: ") << reason;
    return std::optional<StreamSite>();
  };

  // A slice nothing reads (the store side) is out of scope: prefetch only
  // moves lines for demand loads.
  bool anyRead = llvm::any_of(subview.getResult().getUsers(), [](Operation *u) {
    return isa<vector::TransferReadOp>(u);
  });
  if (!anyRead)
    return std::nullopt;

  // The slice is exactly the iteration: offset = iv, size = step, stride 1.
  Value iv = forOp.getInductionVar();
  auto offsets = subview.getMixedOffsets();
  auto sizes = subview.getMixedSizes();
  auto strides = subview.getMixedStrides();
  if (offsets.size() != 1 || !isa<Value>(offsets[0]) ||
      cast<Value>(offsets[0]) != iv)
    return decline("subview offset is not the induction variable");
  auto size = getConstIdx(sizes[0]);
  if (sizes.size() != 1 || !size || *size != step)
    return decline("subview size is not the loop step (one vector per "
                   "iteration)");
  auto stride = getConstIdx(strides[0]);
  if (strides.size() != 1 || !stride || *stride != 1)
    return decline("subview stride is not 1");

  // The source is a rank-1, unit-stride memref defined outside the loop: the
  // hoisted metadata reads need it to dominate the loop.
  Value src = subview.getSource();
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy || srcTy.getRank() != 1)
    return decline("source is not a rank-1 memref");
  SmallVector<int64_t, 1> srcStrides;
  int64_t srcOffset;
  if (failed(srcTy.getStridesAndOffset(srcStrides, srcOffset)) ||
      srcStrides[0] != 1)
    return decline("source stride is not statically 1");
  if (!dom.dominates(src, forOp.getOperation()))
    return decline("source does not dominate the loop");

  Type elemTy = srcTy.getElementType();
  if (elemTy.getIntOrFloatBitWidth() % 8 != 0)
    return decline("sub-byte element type");
  int64_t elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

  // At least one consumer reads the slice as the iteration's vector, at [0],
  // unmasked, in bounds. Other consumers of the same slice are harmless: the
  // fetch only moves lines, and the address stream is the same.
  for (Operation *user : subview.getResult().getUsers()) {
    auto read = dyn_cast<vector::TransferReadOp>(user);
    if (!read || read.getBase() != subview.getResult())
      continue;
    if (read.getMask())
      return decline("transfer_read is masked");
    auto indices = read.getIndices();
    auto pos = indices.size() == 1
                   ? mlir::getConstantIntValue(indices[0])
                   : std::nullopt;
    if (!pos || *pos != 0)
      return decline("transfer_read position is not [0]");
    for (Attribute b : read.getInBounds())
      if (!cast<BoolAttr>(b).getValue())
        return decline("transfer_read is not in-bounds");
    auto vecTy = dyn_cast<VectorType>(read.getResult().getType());
    if (!vecTy || vecTy.getRank() != 1 || vecTy.getDimSize(0) != step)
      return decline("transfer vector is not one step wide");
    if (vecTy.getElementType() != elemTy)
      return decline("transfer element type differs from the source");
    return StreamSite{subview, read, elemBytes};
  }
  return decline("no vector.transfer_read of the slice at [0]");
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

/// Emits the fetches for one matched loop: the hoisted metadata, the guard,
/// and one l2fetch per distinct source memref.
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
      // The step must be a positive constant: both the fetch period and the
      // distance in iterations are element counts derived from it.
      auto stepOpt = mlir::getConstantIntValue(forOp.getStep());
      if (!stepOpt || *stepOpt <= 0) {
        forOp.emitRemark("hexagon-l2-prefetch declined: step is not a "
                         "positive constant");
        continue;
      }
      int64_t step = *stepOpt;

      // One site per subview, one fetch per distinct source memref. Sites
      // that fail a condition are declined by remark; the rest still fetch.
      SmallVector<StreamSite> sites;
      SetVector<Value> srcs;
      std::optional<int64_t> elemBytes;
      for (Operation &op : forOp.getBody()->without_terminator()) {
        auto subview = dyn_cast<memref::SubViewOp>(op);
        if (!subview)
          continue;
        std::optional<StreamSite> site =
            matchSite(forOp, subview, step, dom);
        if (!site)
          continue;
        if (elemBytes && *elemBytes != site->elemBytes) {
          // The fetch period is one loop-level fact; mixed vector widths
          // would need one guard per width. The vectorizer emits uniform
          // widths, so this is a shape it does not produce today.
          subview.emitRemark("hexagon-l2-prefetch declined: mixed element "
                             "widths in one loop");
          continue;
        }
        elemBytes = site->elemBytes;
        if (srcs.insert(site->subview.getSource()))
          sites.push_back(*site);
      }
      if (sites.empty() || !elemBytes)
        continue;

      // The block and the distance are whole numbers of this loop's vectors;
      // anything else has no calibrated meaning.
      int64_t vecBytes = step * *elemBytes;
      if (kFetchBlockBytes % vecBytes != 0 ||
          kPrefetchDistanceBytes % vecBytes != 0) {
        forOp.emitRemark("hexagon-l2-prefetch declined: vector size "
                         "does not divide the fetch block or distance");
        continue;
      }

      emitFetches(forOp, rewriter, srcs.getArrayRef(), step, *elemBytes,
                  l2fetch);
      forOp.emitRemark("hexagon-l2-prefetch: fetching ")
          << srcs.size() << " stream(s), distance " << kPrefetchDistanceBytes
          << " B, block " << kFetchBlockBytes << " B";
      DBG("prefetched " << srcs.size() << " stream(s) in loop at "
                        << forOp.getLoc());
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexagon::createHexagonL2PrefetchPass() {
  return std::make_unique<HexagonL2PrefetchPass>();
}
