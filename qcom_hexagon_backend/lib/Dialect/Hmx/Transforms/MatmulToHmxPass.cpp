//===-- MatmulToHmxPass.cpp - linalg.matmul to hmx.matmul --------*- C++
//-*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Engine attribution for the HMX matrix engine.
//
// The whole decision is one capability query (`HmxTarget::queryContraction`)
// followed by a worth-it question (`HmxTarget::planBridge`) asking how the
// crouton bridge fits the VTCM pool -- whole when it fits, in M blocks when it
// does not -- the same shape Triton's AccelerateMatmul uses.
// There is no user-facing switch: a matmul either fits the engine's contract
// and repays its bridge, or it is left alone.
//
// An op that does not fit is left completely untouched. "This op is not for
// HMX" is an engine choice that the existing HVX and linalg paths already
// handle, not a failure -- so this pass never errors. When the element types
// *are* the engine's (f16) a remark says why the op was skipped, because a f16
// matmul that misses the tile grid is the one case a user would want to know
// about.
//
// A remark alone has no visible output on the production path, though, so the
// end of the run also emits ONE warning on the module whenever a matmul was
// refused: how many were skipped (and how many attributed), quoting the first
// refusal -- and, when that refusal was the pipeline's missing VTCM allocator,
// naming the switch that closed the whole HMX path, not just one op.
//
// The engine reads croutons, so `hmx.matmul` takes crouton operands and this
// pass is what bridges row-major producers to it with explicit `linalg.pack`s
// (and bridges the result back with `linalg.unpack`s). Two consequences fall
// out of keeping the layout in the type:
//
//   * a matmul chained off another matmul needs no bridge at all -- the
//   read-out
//     layout AR and the activation layout AH are the same permutation, so the
//     types already match (see docs/hmx/hmx-system-design.md §4.1);
//   * a constant weight folds: `fold-pack-unpack-constants` turns the pack of a
//     constant into a prepacked constant at compile time, so W never costs a
//     runtime pack.
//
// The engine's read-out is an fp16 crouton. f32 source and result values are
// admitted as a quantization/widening ABI: the pack leaf materializes f16
// croutons and the fused tail reads them back as f32. This is not bit-exact
// fp32 accumulation; callers requiring that semantic belong on a non-HMX path.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxRecordV3.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxTarget.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>

#define DEBUG_TYPE "matmul-to-hmx"

using namespace mlir;
using namespace hexagon;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_MATMULTOHMX
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

/// True when the accumulator initialiser is provably empty or all zeros. Only
/// then is it safe to hand the matmul to an engine that clears its hardware
/// accumulator: `linalg.matmul` adds into C, while `hmx.matmul` overwrites.
bool isEmptyInit(Value init) {
  if (init.getDefiningOp<tensor::EmptyOp>())
    return true;
  // A zero fill is how the tests spell an accumulator initialiser.
  if (auto fill = init.getDefiningOp<linalg::FillOp>())
    if (auto cst = fill.getDpsInputOperand(0)
                       ->get()
                       .getDefiningOp<arith::ConstantOp>())
      if (auto scalar = dyn_cast<FloatAttr>(cst.getValue()))
        return scalar.getValue().isZero();
  if (auto cst = init.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
      return dense.isSplat() && dense.getSplatValue<APFloat>().isZero();
  }
  return false;
}

/// The op-specific adapter: what this matmul's extents and element types are.
/// Everything op-independent -- the tile grid, the accumulation width, the row
/// minimum -- is asked of `HmxTarget`, so a second op class only writes its own
/// adapter rather than its own pass.
struct MatmulContract {
  int64_t m, n, k;
  Type lhsElem, rhsElem, outElem;
};

std::string typeName(Type type);

enum class ManifestShapeState {
  Static,
  PartiallyDynamic,
  Dynamic,
  Unavailable
};
enum class ManifestWeightSource {
  ArgumentSlot,
  CompileTimeConstant,
  InternalValue
};

struct ManifestFacts {
  ManifestShapeState shapeState = ManifestShapeState::Unavailable;
  int64_t dims[3] = {-1, -1, -1};
  std::string lhsElem;
  std::string rhsElem;
  std::string outElem;
  ManifestWeightSource weightSource = ManifestWeightSource::InternalValue;
  int64_t argumentSlot = -1;
};

/// One canonical decision code per original linalg.matmul. The first
/// successful rewrite of that op is the only point at which the code becomes
/// `selected-aligned`; every later failure updates the same record in place.
enum class MatmulReason {
  SelectedAligned,
  SelectedTail,
  LibraryCall,
  VtcmAllocatorDisabled,
  NonRank2,
  DynamicShape,
  UnsupportedDType,
  MinRows,
  TileAlignment,
  UnsupportedLayout,
  VtcmBudget,
};

StringRef reasonCode(MatmulReason reason) {
  switch (reason) {
  case MatmulReason::SelectedAligned:
    return kHmxReasonSelectedAligned;
  case MatmulReason::SelectedTail:
    return kHmxReasonSelectedTail;
  case MatmulReason::LibraryCall:
    return kHmxReasonLibraryCall;
  case MatmulReason::VtcmAllocatorDisabled:
    return kHmxReasonVtcmAllocatorDisabled;
  case MatmulReason::NonRank2:
    return kHmxReasonNonRank2;
  case MatmulReason::DynamicShape:
    return kHmxReasonDynamicShape;
  case MatmulReason::UnsupportedDType:
    return kHmxReasonUnsupportedDType;
  case MatmulReason::MinRows:
    return kHmxReasonMinRows;
  case MatmulReason::TileAlignment:
    return kHmxReasonTileAlignment;
  case MatmulReason::UnsupportedLayout:
    return kHmxReasonUnsupportedLayout;
  case MatmulReason::VtcmBudget:
    return kHmxReasonVtcmBudget;
  }
  llvm_unreachable("unknown HMX manifest reason");
}

struct MatmulDecision {
  linalg::MatmulOp op;
  std::string functionName;
  int64_t id = 0;
  std::optional<MatmulContract> contract;
  MatmulReason reason = MatmulReason::SelectedAligned;
  int64_t vtcmBefore = 0;
  int64_t vtcmPeak = 0;
  HmxTarget::ContractionPlan contractionPlan = HmxTarget::ContractionPlan::HVX;
  std::shared_ptr<ManifestFacts> facts;
  std::optional<HmxTarget::BridgePlan> plan;
  /// The rewrite ran on `op`. The greedy driver dead-code-eliminates a matmul
  /// whose result is unused *before* it offers the op to any pattern, so a
  /// decision can outlive its operation: `op` dangles, and nothing may read it
  /// once the driver returns. Record-only never rewrites, and needs no such
  /// marker.
  bool visited = false;

  /// Whether the decision claims an `hmx.matmul` site, i.e. one of the two
  /// reasons that describe an attribution rather than a refusal.
  bool claimsHmxSite() const {
    return reason == MatmulReason::SelectedAligned ||
           reason == MatmulReason::SelectedTail;
  }
};

static int64_t vtcmBytesCommitted(Operation *within);

/// Stamp an HMX bridge operation with the function-local decision id. The
/// bufferization model copies this explicit attribute when it rebuilds the op;
/// locations are intentionally not used as a metadata channel.
void setDecisionId(Operation *op, int64_t id) {
  op->setAttr(kHmxDecisionIdAttr,
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), id));
}

static std::shared_ptr<ManifestFacts> makeManifestFacts(linalg::MatmulOp op) {
  auto facts = std::make_shared<ManifestFacts>();
  auto shaped = [](Value value) -> ShapedType {
    return dyn_cast<ShapedType>(value.getType());
  };
  ShapedType lhs = shaped(op.getDpsInputOperand(0)->get());
  ShapedType rhs = shaped(op.getDpsInputOperand(1)->get());
  ShapedType out = shaped(op.getDpsInitOperand(0)->get());
  auto recordType = [&](ShapedType type, std::string &destination) {
    if (type)
      destination = typeName(type.getElementType());
    else
      destination = "unavailable";
  };
  recordType(lhs, facts->lhsElem);
  recordType(rhs, facts->rhsElem);
  recordType(out, facts->outElem);

  if (lhs && rhs && out && lhs.getRank() == 2 && rhs.getRank() == 2 &&
      out.getRank() == 2) {
    auto dim = [](ShapedType type, int64_t index) {
      return type.isDynamicDim(index) ? int64_t(-1) : type.getDimSize(index);
    };
    facts->dims[0] = dim(lhs, 0);
    facts->dims[1] = dim(rhs, 1);
    facts->dims[2] = dim(lhs, 1);
    int dynamic = 0;
    for (int64_t extent : facts->dims)
      dynamic += extent < 0;
    facts->shapeState = dynamic == 0   ? ManifestShapeState::Static
                        : dynamic == 3 ? ManifestShapeState::Dynamic
                                       : ManifestShapeState::PartiallyDynamic;
  }

  Value source = op.getDpsInputOperand(1)->get();
  if (auto argument = dyn_cast<BlockArgument>(source)) {
    if (auto function = op->getParentOfType<func::FuncOp>()) {
      int64_t slot = 0;
      for (BlockArgument candidate : function.getArguments()) {
        if (!isa<RankedTensorType, MemRefType, UnrankedMemRefType>(
                candidate.getType()))
          continue;
        if (candidate == argument) {
          facts->weightSource = ManifestWeightSource::ArgumentSlot;
          facts->argumentSlot = slot;
          break;
        }
        ++slot;
      }
    }
  } else if (source.getDefiningOp<arith::ConstantOp>()) {
    facts->weightSource = ManifestWeightSource::CompileTimeConstant;
  }
  return facts;
}

/// The pass-scoped, per-op decision table. It is populated before the greedy
/// driver starts, so a pattern that tries an op repeatedly still updates one
/// record and the final manifest has one entry per original op -- except for a
/// matmul the driver removed before any pattern saw it, which has no entry
/// because it has no site left to describe (`describes` below).
class AttributionTally {
public:
  AttributionTally(func::FuncOp func, const HmxTarget &target, bool recordOnly)
      : recordOnly(recordOnly) {
    int64_t nextId = 0;
    func.walk([&](linalg::MatmulOp op) {
      auto shapedType = [&](Value value) -> ShapedType {
        if (auto tensor = dyn_cast<RankedTensorType>(value.getType()))
          return tensor;
        // Record-only also observes manually managed memref form. The normal
        // rewrite remains tensor-only because its bridge builders consume
        // tensor values, but metadata must not lose those matmuls.
        if (recordOnly)
          return dyn_cast<ShapedType>(value.getType());
        return {};
      };
      ShapedType lhsType = shapedType(op.getDpsInputOperand(0)->get());
      ShapedType rhsType = shapedType(op.getDpsInputOperand(1)->get());
      ShapedType outType = shapedType(op.getDpsInitOperand(0)->get());
      bool ranked = lhsType && rhsType && outType;
      bool rank2 = ranked && lhsType.getRank() == 2 && rhsType.getRank() == 2 &&
                   outType.getRank() == 2;
      bool dynamic =
          rank2 && (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
                    !outType.hasStaticShape());

      MatmulDecision decision{op, func.getName().str(), nextId++};
      decision.facts = makeManifestFacts(op);
      if (rank2 && !dynamic) {
        decision.contract =
            MatmulContract{lhsType.getDimSize(0),    rhsType.getDimSize(1),
                           lhsType.getDimSize(1),    lhsType.getElementType(),
                           rhsType.getElementType(), outType.getElementType()};
      }

      if (op->hasAttr("library_call")) {
        decision.reason = MatmulReason::LibraryCall;
      } else if (!rank2) {
        decision.reason = MatmulReason::NonRank2;
      } else if (dynamic) {
        decision.reason = MatmulReason::DynamicShape;
      } else if (!target.vtcmAllocator) {
        decision.reason = MatmulReason::VtcmAllocatorDisabled;
      } else {
        HmxTarget::ContractionDecision capability = target.queryContraction(
            decision.contract->m, decision.contract->n, decision.contract->k,
            decision.contract->lhsElem, decision.contract->rhsElem,
            decision.contract->outElem);
        decision.contractionPlan = capability.plan;
        if (!capability.supported()) {
          bool diagnosticTail =
              isHmxDiagnosticTailMarker(op.getOperation()) &&
              capability.refusal ==
                  HmxTarget::ContractionRefusal::TileAlignment &&
              capability.tailCandidate();
          if (diagnosticTail) {
            const HmxTarget::ContractionShape &shape = capability.shape;
            auto paddedPlan = target.planBridge(shape.mp, shape.np, shape.kp,
                                                vtcmBytesCommitted(op));
            if (paddedPlan && !paddedPlan.blocked(shape.mp)) {
              decision.reason = MatmulReason::SelectedTail;
              decision.plan = paddedPlan;
            } else {
              decision.reason = MatmulReason::VtcmBudget;
            }
          } else {
            switch (capability.refusal) {
            case HmxTarget::ContractionRefusal::UnsupportedDType:
              decision.reason = MatmulReason::UnsupportedDType;
              break;
            case HmxTarget::ContractionRefusal::MinRows:
              decision.reason = MatmulReason::MinRows;
              break;
            case HmxTarget::ContractionRefusal::TileAlignment:
              decision.reason = MatmulReason::TileAlignment;
              break;
            case HmxTarget::ContractionRefusal::None:
              llvm_unreachable("supported contraction has a refusal");
            }
          }
        } else {
          decision.reason = MatmulReason::SelectedAligned;
        }
      }
      byOp[op.getOperation()] = decisions.size();
      decisions.push_back(std::move(decision));
    });
  }

  MatmulDecision &operator[](linalg::MatmulOp op) {
    auto it = byOp.find(op.getOperation());
    assert(it != byOp.end() && "matmul has no attribution record");
    return decisions[it->second];
  }

  ArrayRef<MatmulDecision> records() const { return decisions; }

  /// Whether this decision still describes a matmul in the module, i.e. whether
  /// the manifest, the v3 document and the diagnostics may speak about it. The
  /// greedy driver removes a matmul whose result is unused before any pattern
  /// sees it: such a decision has no site left (no `hmx.matmul`, no bridge, and
  /// no record `restoreHmxManifestDecisionIds` could bind) and its operation
  /// handle dangles, so it is described by nothing and read by nothing.
  /// Record-only rewrites nothing, so every matmul it was given is still there
  /// and every one of them is described.
  bool describes(const MatmulDecision &decision) const {
    return recordOnly || decision.visited;
  }

private:
  llvm::DenseMap<Operation *, size_t> byOp;
  SmallVector<MatmulDecision> decisions;
  bool recordOnly = false;
};

/// The diagnostic view is intentionally a projection of the manifest record,
/// rather than a second set of predicates. The few cases that were silent
/// before attribution (library dispatch, missing rank/static shape and an
/// unsupported element type) stay silent; their reason is still published.
bool reportsRefusal(MatmulReason reason) {
  return reason == MatmulReason::VtcmAllocatorDisabled ||
         reason == MatmulReason::MinRows ||
         reason == MatmulReason::TileAlignment ||
         reason == MatmulReason::UnsupportedLayout ||
         reason == MatmulReason::VtcmBudget;
}

InFlightDiagnostic &renderRefusal(InFlightDiagnostic &diag,
                                  const MatmulDecision &decision,
                                  const HmxTarget &target) {
  diag << "HMX not applied: ";
  switch (decision.reason) {
  case MatmulReason::VtcmAllocatorDisabled:
    diag << "the crouton arrays live in VTCM and this pipeline has no VTCM "
            "allocator (the hexagonmem path)";
    break;
  case MatmulReason::UnsupportedDType:
  case MatmulReason::MinRows:
  case MatmulReason::TileAlignment:
  case MatmulReason::UnsupportedLayout:
    diag << "needs f16/f32 inputs and an f16/f32 result, 2D static shapes, "
            "M/N/K multiples of "
         << HmxTarget::tileEdge << ", M > " << HmxTarget::minRows;
    break;
  case MatmulReason::VtcmBudget:
    diag << "matmul (M=" << decision.contract->m
         << ", N=" << decision.contract->n << ", K=" << decision.contract->k
         << ", lhsElem=" << decision.contract->lhsElem
         << ", rhsElem=" << decision.contract->rhsElem
         << ", outElem=" << decision.contract->outElem
         << ", vtcmUsed=" << decision.vtcmBefore
         << ") bridge footprint does not fit remaining VTCM (vtcmBudget="
         << target.vtcmBudget << " bytes)";
    break;
  case MatmulReason::LibraryCall:
  case MatmulReason::NonRank2:
  case MatmulReason::DynamicShape:
  case MatmulReason::SelectedAligned:
  case MatmulReason::SelectedTail:
    llvm_unreachable("silent or successful decision rendered as refusal");
  }
  if (decision.reason == MatmulReason::TileAlignment && decision.contract) {
    HmxTarget::ContractionDecision capability = target.queryContraction(
        decision.contract->m, decision.contract->n, decision.contract->k,
        decision.contract->lhsElem, decision.contract->rhsElem,
        decision.contract->outElem);
    if (capability.tailCandidate()) {
      const HmxTarget::ContractionShape &shape = capability.shape;
      diag << " [candidate=hmx-tail, padded=(" << shape.mp << ", " << shape.np
           << ", " << shape.kp << "), full=(" << shape.mf << ", " << shape.nf
           << ", " << shape.kf << "), tail=(" << shape.mt << ", " << shape.nt
           << ", " << shape.kt << ")]";
    }
  }
  return diag << " [reason=" << reasonCode(decision.reason) << "]";
}

void emitDecisionDiagnostics(ModuleOp module, const HmxTarget &target,
                             const AttributionTally &tally) {
  int64_t attributed = 0;
  SmallVector<const MatmulDecision *> reportable;
  for (const MatmulDecision &decision : tally.records()) {
    // A matmul the driver removed never reached a pattern, so there is no
    // operation to anchor a remark to and nothing in the IR to report about.
    if (!tally.describes(decision))
      continue;
    if (decision.claimsHmxSite()) {
      ++attributed;
      continue;
    }
    if (reportsRefusal(decision.reason)) {
      InFlightDiagnostic diag = decision.op->emitRemark();
      renderRefusal(diag, decision, target);
      reportable.push_back(&decision);
    }
  }

  if (reportable.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE << "] " << attributed
               << " matmul(s) attributed, no reportable refusal\n");
    return;
  }

  const MatmulDecision &first = *reportable.front();
  InFlightDiagnostic diag = module.emitWarning();
  if (first.reason == MatmulReason::VtcmAllocatorDisabled)
    diag << "HMX disabled: this pipeline has no VTCM allocator "
            "(enableConvertToHexagonmem is off); ";
  diag << "HMX: " << reportable.size() << " matmul(s) skipped";
  if (attributed)
    diag << ", " << attributed << " attributed";
  diag << "; first refusal: ";
  renderRefusal(diag, first, target);
  if (first.contract)
    diag << "; matmul M=" << first.contract->m << ", N=" << first.contract->n
         << ", K=" << first.contract->k
         << ", lhsElem=" << first.contract->lhsElem
         << ", rhsElem=" << first.contract->rhsElem
         << ", outElem=" << first.contract->outElem;
}

std::string typeName(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

static StringRef shapeStateName(ManifestShapeState state) {
  switch (state) {
  case ManifestShapeState::Static:
    return "static";
  case ManifestShapeState::PartiallyDynamic:
    return "partially-dynamic";
  case ManifestShapeState::Dynamic:
    return "dynamic";
  case ManifestShapeState::Unavailable:
    return "unavailable";
  }
  llvm_unreachable("unknown manifest shape state");
}

static DictionaryAttr makeDimensionAttr(MLIRContext *ctx, int64_t extent,
                                        StringRef symbol) {
  NamedAttrList fields;
  fields.append("kind",
                StringAttr::get(ctx, extent >= 0 ? "static" : "dynamic"));
  if (extent >= 0)
    fields.append("value", IntegerAttr::get(IntegerType::get(ctx, 64), extent));
  else
    fields.append("symbol", StringAttr::get(ctx, symbol));
  return fields.getDictionary(ctx);
}

static DictionaryAttr makeLogicalAttr(MLIRContext *ctx,
                                      const ManifestFacts &facts) {
  if (facts.shapeState == ManifestShapeState::Unavailable)
    return {};
  NamedAttrList fields;
  fields.append("m", makeDimensionAttr(ctx, facts.dims[0], "m"));
  fields.append("n", makeDimensionAttr(ctx, facts.dims[1], "n"));
  fields.append("k", makeDimensionAttr(ctx, facts.dims[2], "k"));
  return fields.getDictionary(ctx);
}

static DictionaryAttr makeDTypesAttr(MLIRContext *ctx,
                                     const ManifestFacts &facts, bool hmx) {
  NamedAttrList fields;
  fields.append("lhs", StringAttr::get(ctx, facts.lhsElem));
  fields.append("rhs", StringAttr::get(ctx, facts.rhsElem));
  fields.append("out", StringAttr::get(ctx, facts.outElem));
  if (hmx)
    fields.append("crouton", StringAttr::get(ctx, "f16"));
  return fields.getDictionary(ctx);
}

static DictionaryAttr makeExtentAttr(MLIRContext *ctx, int64_t m, int64_t n,
                                     int64_t k) {
  auto i64 = IntegerType::get(ctx, 64);
  NamedAttrList fields;
  fields.append("m", IntegerAttr::get(i64, m));
  fields.append("n", IntegerAttr::get(i64, n));
  fields.append("k", IntegerAttr::get(i64, k));
  return fields.getDictionary(ctx);
}

static DictionaryAttr makeExecutionAttr(MLIRContext *ctx,
                                        const MatmulDecision &decision) {
  auto i64 = IntegerType::get(ctx, 64);
  assert(decision.plan && decision.contract &&
         "selected matmul has no bridge plan");
  bool tail = decision.reason == MatmulReason::SelectedTail;
  NamedAttrList fields;
  fields.append(
      "blocking",
      StringAttr::get(ctx, !tail && decision.plan->blocked(decision.contract->m)
                               ? "m_blocked"
                               : "whole"));
  fields.append("block_m", IntegerAttr::get(i64, tail ? decision.contract->m
                                                      : decision.plan->blockM));
  NamedAttrList counts;
  counts.append("pack_act_sites", IntegerAttr::get(i64, 0));
  counts.append("pack_weight_sites", IntegerAttr::get(i64, 0));
  counts.append("unpack_sites", IntegerAttr::get(i64, 0));
  counts.append("count_semantics", StringAttr::get(ctx, "ir_sites"));
  fields.append("bridge_counts", counts.getDictionary(ctx));
  return fields.getDictionary(ctx);
}

static DictionaryAttr makeWeightBindingAttr(MLIRContext *ctx,
                                            const MatmulDecision &decision) {
  const ManifestFacts &facts = *decision.facts;
  NamedAttrList fields;
  switch (facts.weightSource) {
  case ManifestWeightSource::ArgumentSlot: {
    fields.append("kind", StringAttr::get(ctx, "argument-slot"));
    NamedAttrList reference;
    reference.append("function", StringAttr::get(ctx, decision.functionName));
    reference.append("slot", IntegerAttr::get(IntegerType::get(ctx, 64),
                                              facts.argumentSlot));
    fields.append("policy_ref", reference.getDictionary(ctx));
    break;
  }
  case ManifestWeightSource::CompileTimeConstant:
    fields.append("kind", StringAttr::get(ctx, "compile-time-constant"));
    break;
  case ManifestWeightSource::InternalValue:
    fields.append("kind", StringAttr::get(ctx, "internal-value"));
    break;
  }
  return fields.getDictionary(ctx);
}

DictionaryAttr manifestRecord(MLIRContext *ctx, const MatmulDecision &decision,
                              const HmxTarget &target) {
  auto i64 = IntegerType::get(ctx, 64);
  assert(decision.facts && "matmul decision has no semantic facts");
  const ManifestFacts &facts = *decision.facts;
  // Only a decision that still describes a matmul in the module reaches here
  // (see `AttributionTally::describes`), and attribution becomes true only when
  // the rewrite runs, so an `hmx` record always carries the contract and the
  // bridge plan the rewrite chose.
  bool hmx = decision.claimsHmxSite();
  bool tail = decision.reason == MatmulReason::SelectedTail;
  StringRef plan =
      tail ? kHmxPlanHMXTail : (hmx ? kHmxPlanFullHMX : kHmxPlanHVX);

  NamedAttrList fields;
  fields.append("function", StringAttr::get(ctx, decision.functionName));
  fields.append("id", IntegerAttr::get(i64, decision.id));
  fields.append("plan", StringAttr::get(ctx, plan));
  fields.append("reason", StringAttr::get(ctx, reasonCode(decision.reason)));
  fields.append("shape_state",
                StringAttr::get(ctx, shapeStateName(facts.shapeState)));
  if (facts.shapeState == ManifestShapeState::Unavailable)
    fields.append("logical", UnitAttr::get(ctx));
  else
    fields.append("logical", makeLogicalAttr(ctx, facts));
  fields.append("dtypes", makeDTypesAttr(ctx, facts, hmx));

  if (hmx) {
    assert(decision.contract && decision.plan &&
           "selected HMX record has no contract or bridge plan");
    HmxTarget::ContractionDecision capability = target.queryContraction(
        decision.contract->m, decision.contract->n, decision.contract->k,
        decision.contract->lhsElem, decision.contract->rhsElem,
        decision.contract->outElem);
    assert(
        ((tail && capability.plan == HmxTarget::ContractionPlan::HMXTail) ||
         (!tail && capability.plan == HmxTarget::ContractionPlan::FullHMX)) &&
        "selected HMX record has the wrong contraction plan");
    const HmxTarget::ContractionShape &shape = capability.shape;
    fields.append("padded", makeExtentAttr(ctx, shape.mp, shape.np, shape.kp));
    fields.append("full", makeExtentAttr(ctx, shape.mf, shape.nf, shape.kf));
    fields.append("tail", makeExtentAttr(ctx, shape.mt, shape.nt, shape.kt));
    fields.append("layout", StringAttr::get(ctx, "row-major-inner-contiguous"));
    fields.append("workspace_class", StringAttr::get(ctx, "runtime-internal"));
    fields.append("grid_policy", StringAttr::get(ctx, tail ? "single-instance"
                                                           : "legacy-runtime"));
    fields.append("vtcm_accounting", StringAttr::get(ctx, "bridge-only"));
    fields.append("vtcm_budget_bytes",
                  IntegerAttr::get(i64, target.vtcmBudget));
    fields.append("vtcm_before_bytes",
                  IntegerAttr::get(i64, decision.vtcmBefore));
    fields.append("vtcm_bridge_peak_bytes",
                  IntegerAttr::get(i64, decision.vtcmPeak));
    if (tail) {
      NamedAttrList tailPolicy;
      tailPolicy.append("k", StringAttr::get(ctx, "zero-pad-both-operands"));
      tailPolicy.append("mn",
                        StringAttr::get(ctx, "padded-edge-tile-bounded-store"));
      fields.append("tail_policy", tailPolicy.getDictionary(ctx));
    }
    fields.append("execution", makeExecutionAttr(ctx, decision));
    fields.append("weight_binding", makeWeightBindingAttr(ctx, decision));
  }
  return fields.getDictionary(ctx);
}

/// The record-only v3 skeleton for the same decision.
///
/// This is a *second producer over the same compile-time facts*, not a
/// projection of `manifestRecord()`: it re-reads the decision and re-derives
/// everything it publishes, and it shares no field name with the serialized v2
/// contract.  A library call still gets a record when its operands are shaped
/// -- the record publishes the compile-time outcome, not a reason -- while a
/// fact the compile inputs do not carry produces no record instead of a guessed
/// one.
DictionaryAttr hmxRecordV3Skeleton(MLIRContext *ctx,
                                   const MatmulDecision &decision) {
  assert(decision.facts && "matmul decision has no semantic facts");
  const ManifestFacts &facts = *decision.facts;
  bool hmx = decision.claimsHmxSite();
  bool tail = decision.reason == MatmulReason::SelectedTail;
  StringRef plan =
      tail ? kHmxPlanHMXTail : (hmx ? kHmxPlanFullHMX : kHmxPlanHVX);
  if (facts.shapeState == ManifestShapeState::Unavailable)
    return {};
  Attribute skeleton = mlir::hmx::buildHmxRecordV3Skeleton(
      ctx, decision.functionName, decision.id, plan,
      ArrayRef<int64_t>(facts.dims, 3));
  return dyn_cast_or_null<DictionaryAttr>(skeleton);
}

/// The publishable v3 skeletons for every decided matmul in one function. The
/// document describes the same sites as the v2 manifest, so a decision that is
/// not described there is not described here either.
SmallVector<DictionaryAttr> hmxRecordV3Skeletons(MLIRContext *ctx,
                                                 const AttributionTally &tally) {
  SmallVector<DictionaryAttr> skeletons;
  for (const MatmulDecision &decision : tally.records())
    if (tally.describes(decision))
      if (DictionaryAttr skeleton = hmxRecordV3Skeleton(ctx, decision))
        skeletons.push_back(skeleton);
  return skeletons;
}

/// Materialize the zero-padded row-major view consumed by a diagnostic tail
/// bridge. The tail ABI requires both K operands to own their padded K lanes;
/// relying on an over-read or on an allocator's incidental bytes is forbidden.
/// Even an already-aligned operand is routed through an owned tensor
/// materialization (a generic copy) so the bufferizer gives the diagnostic
/// bridge a statically row-major buffer instead of a dynamic-stride function
/// argument.
static Value padMatrix(RewriterBase &b, Location loc, Value value,
                       int64_t paddedRows, int64_t paddedCols) {
  auto type = cast<RankedTensorType>(value.getType());
  if (!type.hasStaticShape() || type.getDimSize(0) > paddedRows ||
      type.getDimSize(1) > paddedCols)
    return {};
  if (type.getDimSize(0) == paddedRows && type.getDimSize(1) == paddedCols) {
    SmallVector<int64_t> paddedShape = {paddedRows, paddedCols};
    Value empty =
        tensor::EmptyOp::create(b, loc, paddedShape, type.getElementType());
    SmallVector<AffineMap> maps(2, b.getMultiDimIdentityMap(2));
    SmallVector<utils::IteratorType> iterators(2,
                                               utils::IteratorType::parallel);
    auto copy = linalg::GenericOp::create(
        b, loc, TypeRange{type}, ValueRange{value}, ValueRange{empty}, maps,
        iterators,
        [](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange args) {
          linalg::YieldOp::create(bodyBuilder, bodyLoc, args[0]);
        });
    return copy.getResult(0);
  }
  SmallVector<int64_t> low = {0, 0};
  SmallVector<int64_t> high = {paddedRows - type.getDimSize(0),
                               paddedCols - type.getDimSize(1)};
  auto paddedType =
      RankedTensorType::get({paddedRows, paddedCols}, type.getElementType());
  Value zero =
      arith::ConstantOp::create(b, loc, b.getZeroAttr(type.getElementType()));
  return tensor::PadOp::create(b, loc, paddedType, value,
                               getAsIndexOpFoldResult(b.getContext(), low),
                               getAsIndexOpFoldResult(b.getContext(), high),
                               zero, false)
      .getResult();
}

/// A destination the engine will read: a crouton array allocated in VTCM.
///
/// `hmx.alloc_crouton` is the dialect's own allocation op, emitted instead of
/// `bufferization.alloc_tensor`: the stock op infers its own static-identity
/// buffer type and the pinned one-shot pass cannot be given a converter, so it
/// could never carry the `#hmx.crouton` encoding across bufferization
/// (hmx-interface-gaps.md section 2.2). The dialect op's
/// BufferizableOpInterface maps the encoding to
/// `#hmx.crouton_memref_layout`; with `drop-encodings` (the default) the
/// encoding is gone and the bufferized type is byte-identical to what
/// `alloc_tensor` produced. The VTCM placement is the op's meaning, not an
/// attribute: the engine reads and writes VTCM only.
///
/// Declaring the placement here, instead of rewriting memory spaces after
/// bufferization, is what keeps the later binding of these buffers to the
/// region the runtime acquires a *type-compatible* edit. The tensor type itself
/// carries no memory space, so nothing about the op signatures changes.
Value vtcmEmpty(RewriterBase &b, Location loc, RankedTensorType type) {
  return AllocCroutonOp::create(b, loc, type);
}

/// Bytes already committed to VTCM by earlier attributions in this function:
/// the static `hmx.alloc_crouton` crouton arrays previous rewrites created.
/// This pass runs before
/// bufferization, so at this stage those are exactly the crouton arrays
/// previous rewrites created -- reading them back is what lets the second dot
/// of an attention pair see the first dot's residency. Without it Gate 3 would
/// check every dot against an empty budget, which is fiction once two dots
/// share a kernel. Deliberately a query, not an allocator: placement stays with
/// the existing space-1 machinery.
static int64_t vtcmBytesCommitted(Operation *within) {
  auto func = within->getParentOfType<func::FuncOp>();
  if (!func)
    return 0;
  int64_t bytes = 0;
  func.walk([&](AllocCroutonOp alloc) {
    auto type = dyn_cast<RankedTensorType>(alloc.getResult().getType());
    if (!type || !type.hasStaticShape())
      return;
    bytes += type.getNumElements() * (type.getElementTypeBitWidth() / 8);
  });
  // Resident constant weights are not `hmx.alloc_crouton`s here -- the
  // residency declaration on the module is their single source of truth, so the
  // budget this pass checks cannot drift from the one the runtime reserves.
  if (auto module = within->getParentOfType<ModuleOp>())
    if (auto resident =
            module->getAttrOfType<IntegerAttr>("hmx.weight_resident_bytes"))
      bytes += resident.getInt();
  return bytes;
}

/// The AH/WH permutation applied at compile time. For an activation (a
/// row-major [M, K] source, a [Mt, Kt, ...] crouton) element (t0, t1, j, col,
/// h) of the crouton array is src[t0*32 + 2j + h][t1*32 + col]. For a weight
/// the crouton is [Nt, Kt, ...] over a row-major [K, N] source (the engine's K
/// runs along dim1), so the two tile axes are read transposed: element
/// (t0=n_tile, t1=k_tile, j, col, h) is src[t1*32 + 2j + h][t0*32 + col]. See
/// docs/hmx/hmx-weight-layout-plan.md §0. Either way a constant operand (a
/// weight in inference, where W is baked in) never needs the runtime packer.
static DenseElementsAttr
prepackCrouton(DenseElementsAttr src, RankedTensorType crouton, bool isWeight) {
  auto srcType = cast<RankedTensorType>(src.getType());
  const int64_t cols = srcType.getDimSize(1);
  auto at = [&](int64_t k, int64_t n) {
    return src.getValues<APFloat>()[k * cols + n];
  };
  SmallVector<APFloat> packed;
  packed.reserve(crouton.getNumElements());
  for (int64_t t0 = 0; t0 < crouton.getDimSize(0); ++t0)
    for (int64_t t1 = 0; t1 < crouton.getDimSize(1); ++t1)
      for (int64_t j = 0; j < hmx::layout::kCroutonPair; ++j)
        for (int64_t c = 0; c < hmx::layout::kCroutonCol; ++c)
          for (int64_t h = 0; h < hmx::layout::kCroutonHalf; ++h)
            packed.push_back(isWeight
                                 ? at(t1 * HmxTarget::tileEdge +
                                          hmx::layout::kCroutonHalf * j + h,
                                      t0 * HmxTarget::tileEdge + c)
                                 : at(t0 * HmxTarget::tileEdge +
                                          hmx::layout::kCroutonHalf * j + h,
                                      t1 * HmxTarget::tileEdge + c));
  return DenseElementsAttr::get(crouton, packed);
}

/// The prepacked constant, copied into its VTCM crouton array: the permutation
/// already ran at compile time, so only a straight elementwise copy is left.
Value prepackedCrouton(RewriterBase &b, Location loc, Value src,
                       RankedTensorType crouton, bool isWeight) {
  auto cst = src.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return {};
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.getType().getElementType().isF16())
    return {};

  // The prepacked constant carries the physical rank-5 *shape* but not the
  // encoding: the layout is a property of the engine's operands, and the
  // constant is only ever read by the copy below. Keeping it unannotated also
  // means no `arith.constant` has to be rewritten when the encoding is dropped.
  auto plain =
      RankedTensorType::get(crouton.getShape(), crouton.getElementType());
  Value packed =
      arith::ConstantOp::create(b, loc, prepackCrouton(dense, plain, isWeight));
  Value out = vtcmEmpty(b, loc, crouton);
  SmallVector<AffineMap> maps(2, b.getMultiDimIdentityMap(crouton.getRank()));
  SmallVector<utils::IteratorType> iterators(crouton.getRank(),
                                             utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{crouton}, ValueRange{packed}, ValueRange{out}, maps,
      iterators, [&](OpBuilder &bodyBuilder, Location l, ValueRange args) {
        linalg::YieldOp::create(bodyBuilder, l, args[0]);
      });
  return generic.getResult(0);
}

/// The crouton bridge built out of the runtime's vectorised pack leaves,
/// carried through a loop so the destination is a single buffer (a chain of DPS
/// ops on one buffer would make one-shot bufferization insert a copy per tile).
///
/// Both crouton arrays have K in their second, contiguous dimension -- the
/// activation is [Mt, Kt] and the weight [Nt, Kt] (see
/// docs/hmx/hmx-weight-layout-plan.md §0) -- so the bridge walks the *outer*
/// tile only and hands each op a `count` range that covers the whole K run.
/// One ranged leaf call then packs every K tile of that outer tile with one
/// offset-table load, instead of one call (and one table load) per crouton;
/// when the K extent is a single tile the range degenerates to the old single
/// call. The `(row, col)` handed to the leaf stay source-block coordinates --
/// (axis 0 tile, axis 1 tile) -- never crouton-grid coordinates: an activation
/// packs K along the source's column axis (its source block row is the outer
/// tile), a weight packs K along the source's row axis (its source block column
/// is the outer tile).
///
/// This is the measurement-gated half of §10.13: the generic `linalg.pack`
/// lowering synthesises element-wise moves, measured at 99.3% of the kernel's
/// runtime, while the leaves are the vectorised HVX packers from Phase 0.
Value packCroutonsWithLeaves(RewriterBase &b, Location loc, Value src,
                             RankedTensorType crouton, bool isWeight,
                             int64_t decisionId) {
  if (Value folded = prepackedCrouton(b, loc, src, crouton, isWeight))
    return folded;
  auto srcType = cast<RankedTensorType>(src.getType());
  int64_t rows = srcType.getDimSize(0);
  int64_t cols = srcType.getDimSize(1);
  int64_t tileCols = cols / HmxTarget::tileEdge;
  int64_t rowTiles = rows / HmxTarget::tileEdge;

  // Outer tile count and K-run length in source-block coordinates. A weight's K
  // runs down the source rows, an activation's along its columns.
  int64_t outerTiles = isWeight ? tileCols : rowTiles;
  int64_t kTiles = isWeight ? rowTiles : tileCols;

  Value dst = vtcmEmpty(b, loc, crouton);
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value outer = arith::ConstantIndexOp::create(b, loc, outerTiles);
  IntegerAttr count = b.getI64IntegerAttr(kTiles);

  auto loop = scf::ForOp::create(b, loc, zero, outer, one, ValueRange{dst});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value i = loop.getInductionVar();
    Value carried = loop.getRegionIterArg(0);
    // Activation: (row = m tile = i, col = k run [0, Kt)); weight: (k run
    // [0, Kt), n tile = i).
    Value out;
    if (isWeight) {
      auto pack =
          hmx::PackWeightOp::create(b, loc, crouton, carried, src, zero, i,
                                    count, IntegerAttr(), IntegerAttr());
      setDecisionId(pack.getOperation(), decisionId);
      out = pack->getResult(0);
    } else {
      auto pack = hmx::PackActOp::create(b, loc, crouton, carried, src, i, zero,
                                         count, IntegerAttr(), IntegerAttr());
      setDecisionId(pack.getOperation(), decisionId);
      out = pack->getResult(0);
    }
    scf::YieldOp::create(b, loc, out);
  }
  return loop.getResult(0);
}

/// Whether the fused tail (`hmx.unpack_acc_f32`) may replace the
/// unpack+widen[+add] epilogue: a static row-major f32 grid whose extents are
/// whole crouton tiles. The leaf stores every row as aligned 128 B units, so
/// its stride precondition (`dst_stride % 32 == 0`) is exactly `N % 32 == 0`
/// for the dense destination the bridge materialises; the base-address half
/// (128 B aligned) is guaranteed allocator-side by the pipeline's one-shot
/// bufferization (`buffer-alignment = 128` in LinalgToLLVMPass), which is what
/// allocates that destination. Both halves are compile-time facts, so the
/// choice needs no runtime switch. The check is explicit rather than inherited
/// from legality so a future widening of the engine contract falls back to the
/// old epilogue instead of silently violating the leaf.
bool fusedTailLegal(RankedTensorType outType) {
  if (outType.getRank() != 2 || !outType.hasStaticShape())
    return false;
  return outType.getDimSize(0) % HmxTarget::tileEdge == 0 &&
         outType.getDimSize(1) % HmxTarget::tileEdge == 0;
}

/// True when the matmul result escapes the kernel unconsumed (every user is a
/// function return): the only shape the fused tail is measured good on.
/// A result feeding elementwise chains keeps the old epilogue even when it is
/// f32: the f32-direct form re-tiles downstream lowering (observed: the same
/// row-sum lowering flips 128-wide to 32-wide and quadruples its static
/// reduces), so firing there trades a removed widen loop for unpredictable
/// downstream motion. Narrow by construction; widen with measurements.
bool resultEscapesUnconsumed(linalg::MatmulOp op) {
  return llvm::all_of(op->getUsers(), [](Operation *user) {
    return isa<func::ReturnOp>(user);
  });
}

/// True when `v` materialises as a dense internal buffer: the fused tail reads
/// its residual through the static column count (the way the old unpack leaf
/// addresses its destination), so a residual that could bufferize to a strided
/// view -- a function argument, a slice, anything not produced here -- must
/// keep the old descriptor-based `addInto`. Only values owned densely here
/// qualify; anything else falls back to the old epilogue.
bool isDenseInternal(Value v, int depth = 0) {
  if (depth > 8)
    return false;
  if (v.getDefiningOp<tensor::EmptyOp>())
    return true;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    return isa<DenseElementsAttr>(cst.getValue());
  if (auto fill = v.getDefiningOp<linalg::FillOp>())
    return isDenseInternal(fill.getDpsInitOperand(0)->get(), depth + 1);
  return false;
}

/// The mirror of the pack: one leaf call per AR crouton row-pair. `fused`
/// selects the leaf: `hmx.unpack_acc_f32` (the fused tail) widens and adds the
/// optional `residual` -- the original accumulator's C term -- in the same call
/// instead of running separate DDR passes, and its `outType` is the f32 result.
/// Without `fused` the leaf is `hmx.unpack_acc`, which writes row-major fp16
/// (the leaf is memoised in the runtime); a wider result is that fp16 image
/// widened afterwards, exactly like the accumulator read-out itself.
///
/// The leaf walks the crouton row itself, so one call can cover a whole tile
/// row
/// (`count` = 16 row-pairs) and the loop is one iteration per 32 output rows,
/// not per row-pair. Every destination row is still written as one sequential
/// pass. Both forms share the loop so bufferization treats the destination the
/// same way (a fresh internal buffer, copied out to the kernel output
/// afterwards) -- which is what the allocator-side 128 B guarantee applies to.
Value unpackWithLeaves(RewriterBase &b, Location loc, Value ar,
                       RankedTensorType outType, bool fused, int64_t decisionId,
                       Value residual = {}) {
  auto arType = cast<RankedTensorType>(ar.getType());
  Value dst = tensor::EmptyOp::create(b, loc, outType.getShape(),
                                      outType.getElementType());
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  // One tile row per call: the crouton grid's first dim is the 32-row tile
  // count, and `count` covers its 16 row-pairs.
  Value tiles = arith::ConstantIndexOp::create(b, loc, arType.getDimSize(0));
  IntegerAttr count = b.getI64IntegerAttr(hmx::layout::kCroutonPair);

  auto loop = scf::ForOp::create(b, loc, zero, tiles, one, ValueRange{dst});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value i = loop.getInductionVar();
    Value arg = loop.getRegionIterArg(0);
    Value out;
    if (fused) {
      auto unpack = hmx::UnpackAccF32Op::create(
          b, loc, outType, ar, arg, i, zero, residual, count, IntegerAttr(),
          IntegerAttr(), IntegerAttr());
      setDecisionId(unpack.getOperation(), decisionId);
      out = unpack->getResult(0);
    } else {
      auto unpack =
          hmx::UnpackAccOp::create(b, loc, outType, ar, arg, i, zero, count,
                                   IntegerAttr(), IntegerAttr(), IntegerAttr());
      setDecisionId(unpack.getOperation(), decisionId);
      out = unpack->getResult(0);
    }
    scf::YieldOp::create(b, loc, out);
  }
  return loop.getResult(0);
}

/// Widens an fp16 tensor to fp32 the way the rest of this pipeline does
/// elementwise work -- inside a `linalg.generic` -- rather than as a bare
/// tensor-level `arith.extf`, which bufferization does not accept. Upstream's
/// ConversionToFp16Pass does the same thing for the same reason.
Value widenToF32(OpBuilder &b, Location loc, Value src) {
  auto srcType = cast<RankedTensorType>(src.getType());
  auto dstType = RankedTensorType::get(srcType.getShape(), b.getF32Type());
  Value empty = tensor::EmptyOp::create(b, loc, dstType.getShape(),
                                        dstType.getElementType());
  SmallVector<AffineMap> maps(2, b.getMultiDimIdentityMap(srcType.getRank()));
  SmallVector<utils::IteratorType> iterators(srcType.getRank(),
                                             utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{dstType}, ValueRange{src}, ValueRange{empty}, maps,
      iterators, [&](OpBuilder &bodyBuilder, Location l, ValueRange args) {
        Value extended = arith::ExtFOp::create(
            bodyBuilder, l, bodyBuilder.getF32Type(), args[0]);
        linalg::YieldOp::create(bodyBuilder, l, extended);
      });
  return generic.getResult(0);
}

/// The C term of the original matmul. The engine overwrites its accumulator, so
/// `linalg.matmul`'s "add into C" is materialised after the read-out, in the
/// initialiser's own element type (fp32 for the common f16-in/f32-out case) and
/// inside a `linalg.generic` so the rest of the pipeline can vectorize it.
Value addInto(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  auto type = cast<RankedTensorType>(lhs.getType());
  Value empty =
      tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
  SmallVector<AffineMap> maps(3, b.getMultiDimIdentityMap(type.getRank()));
  SmallVector<utils::IteratorType> iterators(type.getRank(),
                                             utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{type}, ValueRange{lhs, rhs}, ValueRange{empty}, maps,
      iterators, [&](OpBuilder &bodyBuilder, Location l, ValueRange args) {
        Value sum = arith::AddFOp::create(bodyBuilder, l, args[0], args[1]);
        linalg::YieldOp::create(bodyBuilder, l, sum);
      });
  return generic.getResult(0);
}

/// The epilogue shared by the whole and the M-blocked forms: turn the engine's
/// fp16 read-out `ar` (a crouton array over `outType`'s logical shape) back
/// into a row-major tensor, widening and adding `residual` (the original
/// `linalg.matmul` C term) as needed. `canFuseTail` is the caller's
/// `resultEscapesUnconsumed` answer: the fused tail is only measured good on a
/// result that escapes unconsumed, so this keeps that narrow by construction.
/// A `residual` that is not a dense internal buffer keeps the descriptor-based
/// `addInto` (see `isDenseInternal`).
static Value emitEpilogue(RewriterBase &b, Location loc, Value ar,
                          RankedTensorType outType, Value residual,
                          bool canFuseTail, int64_t decisionId) {
  if (outType.getElementType().isF32() && fusedTailLegal(outType) &&
      canFuseTail && (!residual || isDenseInternal(residual)))
    return unpackWithLeaves(b, loc, ar, outType, /*fused=*/true, decisionId,
                            residual);

  auto f16Out = RankedTensorType::get(outType.getShape(), b.getF16Type());
  Value result =
      unpackWithLeaves(b, loc, ar, f16Out, /*fused=*/false, decisionId);
  if (outType.getElementType().isF32())
    result = widenToF32(b, loc, result);
  if (residual)
    result = addInto(b, loc, result, residual);
  return result;
}

/// Emits the crouton bridge for `src` at a point that dominates `consumer`:
/// hoisted out of every enclosing loop `src` does not depend on when `hoist`
/// holds, otherwise right above the consumer (re-packed once per block
/// iteration). In an attention inner loop the query is loop-invariant, and
/// packing it once instead of once per KV block is the whole difference between
/// the bridge showing up in the profile and not.
///
/// `hoist` is invariant-plus-fits: the caller passes loop-invariance ANDed with
/// the VTCM fit of the hoisted buffer, so a hoisted (resident) bridge is never
/// emitted on a budget the attribution did not account for. Either placement is
/// correct; only the residency differs.
Value emitBridgeAbove(RewriterBase &b, Location loc, Value src,
                      RankedTensorType crouton, bool isWeight,
                      Operation *consumer, bool hoist, int64_t decisionId) {
  Operation *insertBefore = consumer;
  if (hoist) {
    for (Operation *parent = consumer->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (!isa<scf::ForOp>(parent))
        continue;
      Operation *definedBy = src.getDefiningOp();
      if (definedBy && parent->isAncestor(definedBy))
        break; /* the value is produced inside this loop: cannot hoist above it
                */
      insertBefore = parent;
    }
  }
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(insertBefore);
  return packCroutonsWithLeaves(b, loc, src, crouton, isWeight, decisionId);
}

/// True when `v` is defined outside `loop` (a block argument counts as defined
/// by the op owning its block, so a value carried by the loop is variant).
static bool definedOutsideLoop(Value v, Operation *loop) {
  if (Operation *def = v.getDefiningOp())
    return !loop->isAncestor(def);
  Operation *owner = v.getParentBlock()->getParentOp();
  return owner != loop && !loop->isAncestor(owner);
}

/// True when `v` is invariant with respect to every scf.for enclosing `op`
/// *and* there is at least one: exactly the condition under which
/// `emitBridgeAbove` hoists the pack out of the loop, which is what lets
/// `HmxTarget::planBridge` amortise that operand's pack cost away.
static bool isLoopInvariant(Value v, Operation *op) {
  bool inLoop = false;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (!isa<scf::ForOp>(parent))
      continue;
    inLoop = true;
    if (!definedOutsideLoop(v, parent))
      return false;
  }
  return inLoop;
}

static LogicalResult emitDiagnosticTailMatmul(linalg::MatmulOp op,
                                              MatmulDecision &decision,
                                              const HmxTarget &target,
                                              PatternRewriter &rewriter) {
  assert(decision.reason == MatmulReason::SelectedTail && decision.contract &&
         decision.plan && "diagnostic tail decision is incomplete");
  const MatmulContract &contract = *decision.contract;
  HmxTarget::ContractionDecision capability = target.queryContraction(
      contract.m, contract.n, contract.k, contract.lhsElem, contract.rhsElem,
      contract.outElem);
  assert(capability.plan == HmxTarget::ContractionPlan::HMXTail &&
         "diagnostic tail decision lost its shape plan");
  const HmxTarget::ContractionShape &shape = capability.shape;
  Location loc = op.getLoc();
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();
  auto outType = cast<RankedTensorType>(init.getType());
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());

  Value paddedLhs = padMatrix(rewriter, loc, lhs, shape.mp, shape.kp);
  Value paddedRhs = padMatrix(rewriter, loc, rhs, shape.kp, shape.np);
  if (!paddedLhs || !paddedRhs)
    return rewriter.notifyMatchFailure(
        op, "diagnostic tail requires static row-major operands");

  auto paddedLhsType = hmx::croutonLayoutType(
      RankedTensorType::get({shape.mp, shape.kp}, lhsType.getElementType()));
  auto paddedRhsType = hmx::weightCroutonType(
      RankedTensorType::get({shape.np, shape.kp}, rhsType.getElementType()));
  auto paddedF16Out =
      RankedTensorType::get({shape.mp, shape.np}, rewriter.getF16Type());
  auto tailPlan = TailPlanAttr::get(
      rewriter.getContext(), {contract.m, contract.n, contract.k},
      {shape.mp, shape.np, shape.kp}, {shape.mf, shape.nf, shape.kf},
      {shape.mt, shape.nt, shape.kt}, "zero-pad-both-operands",
      "padded-edge-tile-bounded-store");

  Value packedLhs =
      emitBridgeAbove(rewriter, loc, paddedLhs, paddedLhsType,
                      /*isWeight=*/false, op, /*hoist=*/false, decision.id);
  Value packedRhs =
      emitBridgeAbove(rewriter, loc, paddedRhs, paddedRhsType,
                      /*isWeight=*/true, op, /*hoist=*/false, decision.id);
  Value outEmpty =
      vtcmEmpty(rewriter, loc, hmx::croutonLayoutType(paddedF16Out));
  auto matmul =
      hmx::MatmulOp::create(rewriter, loc, hmx::croutonLayoutType(paddedF16Out),
                            packedLhs, packedRhs, outEmpty, tailPlan);
  matmul->setAttr(kHmxDiagnosticTailAttr, UnitAttr::get(rewriter.getContext()));
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module || failed(markHmxDiagnosticTailModule(module)))
    return failure();
  setDecisionId(matmul.getOperation(), decision.id);

  bool empty = isEmptyInit(init);
  bool escapes = resultEscapesUnconsumed(op);
  // The normal epilogue bridge is deliberately left in the canonical loop form;
  // hmx-partition's marked diagnostic path owns the full/edge rewrite and is
  // the first consumer that gives its valid M/N extents to the unpack leaf.
  Value result = emitEpilogue(rewriter, loc, matmul->getResult(0), outType,
                              empty ? Value{} : init, escapes, decision.id);
  rewriter.replaceOp(op, result);
  return success();
}

/// `linalg.matmul` -> `hmx.matmul` on croutons, bridged in and out of
/// row-major.
struct MatmulToHmx : public OpRewritePattern<linalg::MatmulOp> {
  MatmulToHmx(MLIRContext *ctx, HmxTarget target, AttributionTally *tally)
      : OpRewritePattern<linalg::MatmulOp>(ctx), target(target), tally(tally) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(kHmxDiagnosticTailAttr) &&
        !isHmxDiagnosticTailMarker(op.getOperation())) {
      op.emitError("hmx.diagnostic_tail_partition must be a unit attribute");
      return failure();
    }
    MatmulDecision &decision = (*tally)[op];
    // The driver offers the op to a pattern only while it is still in the IR,
    // so from here on the decision may be read (and its op remarked on) after
    // the rewrite. A matmul it removed instead never gets here, and is never
    // read.
    decision.visited = true;
    // A marked tail candidate that does not fit the padded bridge budget is a
    // diagnostic refusal, never permission to retry the logical shape through
    // the full-HMX path.
    if (isHmxDiagnosticTailMarker(op.getOperation()) &&
        decision.contractionPlan == HmxTarget::ContractionPlan::HMXTail &&
        decision.reason != MatmulReason::SelectedTail)
      return rewriter.notifyMatchFailure(
          op, "diagnostic tail bridge exceeds VTCM budget");
    if (decision.reason != MatmulReason::SelectedAligned &&
        decision.reason != MatmulReason::SelectedTail &&
        decision.reason != MatmulReason::VtcmBudget)
      return rewriter.notifyMatchFailure(op, reasonCode(decision.reason));
    assert(decision.contract && "attributable matmul has no contract");
    const MatmulContract &contract = *decision.contract;

    Location loc = op.getLoc();
    Value lhs = op.getDpsInputOperand(0)->get();
    Value rhs = op.getDpsInputOperand(1)->get();
    Value init = op.getDpsInitOperand(0)->get();
    auto outType = cast<RankedTensorType>(init.getType());
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());

    // Region residency, not a single-op fiction: bytes earlier attributions in
    // this function already committed to VTCM, so the second dot of an
    // attention pair is weighed with the first dot's arrays resident.
    int64_t vtcmUsed = vtcmBytesCommitted(op);

    if (decision.reason == MatmulReason::SelectedTail) {
      decision.vtcmBefore = vtcmUsed;
      if (!decision.plan) {
        decision.reason = MatmulReason::VtcmBudget;
        return rewriter.notifyMatchFailure(op, reasonCode(decision.reason));
      }
      decision.vtcmPeak = vtcmUsed + decision.plan->bytes;
      return emitDiagnosticTailMatmul(op, decision, target, rewriter);
    }

    // The second question after legality: the crouton bridge must pay for
    // itself. The plan names the M block the bridge allocates for -- the whole
    // M when the contraction fits, a smaller block when only a block does, and
    // an empty plan when not even one block does. A "no" leaves the IR exactly
    // as it was -- an untouched linalg.matmul is an engine choice the
    // HVX/linalg paths handle, never a half-built hmx region the verifier could
    // see.
    HmxTarget::BridgePlan plan =
        target.planBridge(contract.m, contract.n, contract.k, vtcmUsed);
    decision.vtcmBefore = vtcmUsed;
    if (!plan) {
      decision.reason = MatmulReason::VtcmBudget;
      decision.plan.reset();
      return rewriter.notifyMatchFailure(op, reasonCode(decision.reason));
    }
    decision.reason = MatmulReason::SelectedAligned;
    assert(decision.contractionPlan == HmxTarget::ContractionPlan::FullHMX &&
           "selected HMX matmul has no full contraction shape plan");
    decision.vtcmPeak = vtcmUsed + plan.bytes;
    decision.plan = plan;

    // The engine's read-out is an fp16 crouton regardless of the result's
    // element type; a wider result is the fp16 image widened after the unpack.
    auto f16Out =
        RankedTensorType::get(outType.getShape(), rewriter.getF16Type());
    bool empty = isEmptyInit(init);
    bool escapes = resultEscapesUnconsumed(op);

    // The bridge stages crouton arrays, which are the engine's fp16 whatever
    // the sources' element type is, so every byte figure here is a crouton
    // byte.
    constexpr int64_t inBytes = HmxTarget::croutonElemBytes;
    int64_t room = target.vtcmBudget - vtcmUsed;
    int64_t rhsBytes = contract.k * contract.n * inBytes;
    bool aInvariant = isLoopInvariant(lhs, op);
    bool bInvariant = isLoopInvariant(rhs, op);

    // The crouton bridge is the runtime's vectorised pack leaves. Measured on
    // device: 113 us for the packs against 2064 us for the generic linalg
    // lowering, whose element-wise moves cost more than the mma itself. Chains
    // never get here: FoldChainedPack replaces this pack with the producer's
    // read-out array.
    if (!plan.blocked(contract.m)) {
      // Whole contraction: one bridge, one hmx.matmul -- the pre-blocking form,
      // byte-identical to before for every shape whose arrays fit whole.
      //
      // Whole-package promotion vs per-block packing: an invariant bridge is
      // hoisted (packed once, kept resident) only if its buffer fits what is
      // left after prior residency and the rest of this attribution; otherwise
      // it is emitted at the consumer and re-packed once per block iteration.
      int64_t lhsBytes = contract.m * contract.k * inBytes;
      int64_t outBytes = contract.m * contract.n * 2;
      bool hoistLhs = aInvariant && lhsBytes <= room - rhsBytes - outBytes;
      bool hoistRhs =
          bInvariant && rhsBytes <= room - (hoistLhs ? lhsBytes : 0) - outBytes;

      Value packedLhs =
          emitBridgeAbove(rewriter, loc, lhs, hmx::croutonLayoutType(lhsType),
                          /*isWeight=*/false, op, hoistLhs, decision.id);
      Value packedRhs =
          emitBridgeAbove(rewriter, loc, rhs, hmx::weightCroutonType(rhsType),
                          /*isWeight=*/true, op, hoistRhs, decision.id);

      // The accumulator is read out into VTCM: the engine has nowhere else to
      // write.
      Value outEmpty = vtcmEmpty(rewriter, loc, hmx::croutonLayoutType(f16Out));
      auto matmul = hmx::MatmulOp::create(
          rewriter, loc, hmx::croutonLayoutType(f16Out), packedLhs, packedRhs,
          outEmpty, /*tail_plan=*/{});
      setDecisionId(matmul.getOperation(), decision.id);
      // The fused tail replaces the unpack+widen[+add] epilogue with one leaf
      // when the result is f32 and the leaf contract holds by construction (see
      // `fusedTailLegal`); the residual is threaded only when it is dense by
      // construction, anything else keeps the old epilogue.
      Value result = emitEpilogue(rewriter, loc, matmul->getResult(0), outType,
                                  empty ? Value{} : init, escapes, decision.id);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Blocked contraction: walk M in `blockM`-row blocks, each an hmx.matmul on
    // block-sized croutons, and write each block straight back into the carried
    // row-major output. Only one block's activation and read-out are live; the
    // weight stays whole. `blockM` divides M, so the blocks tile M exactly --
    // no partial block and no bounds guard. The weight pack is hoisted out of
    // the block loop exactly as in the whole form, so it is paid once.
    int64_t blockM = plan.blockM;
    auto blockLhsType =
        RankedTensorType::get({blockM, contract.k}, lhsType.getElementType());
    auto blockF16Out =
        RankedTensorType::get({blockM, contract.n}, rewriter.getF16Type());
    auto blockOutType =
        RankedTensorType::get({blockM, contract.n}, outType.getElementType());
    auto blockCroutonOut = hmx::croutonLayoutType(blockF16Out);

    // The block's arrays are the activation block and the read-out block; the
    // weight is whole. This is exactly `plan.bytes`, and the same budget the
    // plan was chosen against.
    int64_t blockActBytes = blockM * contract.k * inBytes;
    int64_t blockArBytes = blockM * contract.n * 2;
    bool hoistRhs =
        bInvariant && rhsBytes <= room - blockActBytes - blockArBytes;

    Value packedRhs =
        emitBridgeAbove(rewriter, loc, rhs, hmx::weightCroutonType(rhsType),
                        /*isWeight=*/true, op, hoistRhs, decision.id);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value cM = arith::ConstantIndexOp::create(rewriter, loc, contract.m);
    Value cBlock = arith::ConstantIndexOp::create(rewriter, loc, blockM);

    // The full output is carried and each block inserted into it, so a block
    // that does not escape still composes with its siblings. `init` is the C
    // term (or an undefined `tensor.empty`, which the blocks fully overwrite).
    auto blockLoop =
        scf::ForOp::create(rewriter, loc, c0, cM, cBlock, ValueRange{init});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      Value m0 = blockLoop.getInductionVar();
      Value carried = blockLoop.getRegionIterArg(0);

      // The block's row offset is the loop's induction variable: block `m0`
      // covers rows `[m0, m0 + blockM)`.
      SmallVector<OpFoldResult> blockOffsets{m0, c0};
      SmallVector<OpFoldResult> blockSizes{rewriter.getIndexAttr(blockM),
                                           rewriter.getIndexAttr(contract.k)};
      SmallVector<OpFoldResult> outSizes{rewriter.getIndexAttr(blockM),
                                         rewriter.getIndexAttr(contract.n)};
      SmallVector<OpFoldResult> oneStrides{rewriter.getIndexAttr(1),
                                           rewriter.getIndexAttr(1)};

      Value lhsBlock =
          tensor::ExtractSliceOp::create(rewriter, loc, blockLhsType, lhs,
                                         blockOffsets, blockSizes, oneStrides);
      Value packedLhs = packCroutonsWithLeaves(
          rewriter, loc, lhsBlock, hmx::croutonLayoutType(blockLhsType),
          /*isWeight=*/false, decision.id);

      // The C term is materialised per block. An empty initialiser has no C
      // term at all, so the block starts from the engine's cleared accumulator.
      Value residual;
      if (!empty)
        residual =
            tensor::ExtractSliceOp::create(rewriter, loc, blockOutType, init,
                                           blockOffsets, outSizes, oneStrides);

      Value outEmpty = vtcmEmpty(rewriter, loc, blockCroutonOut);
      auto matmul = hmx::MatmulOp::create(rewriter, loc, blockCroutonOut,
                                          packedLhs, packedRhs, outEmpty,
                                          /*tail_plan=*/{});
      setDecisionId(matmul.getOperation(), decision.id);
      Value blockResult =
          emitEpilogue(rewriter, loc, matmul->getResult(0), blockOutType,
                       residual, escapes, decision.id);

      Value inserted =
          tensor::InsertSliceOp::create(rewriter, loc, blockResult, carried,
                                        blockOffsets, outSizes, oneStrides);
      scf::YieldOp::create(rewriter, loc, ValueRange{inserted});
    }
    rewriter.replaceOp(op, blockLoop.getResult(0));
    return success();
  }

private:
  const HmxTarget target;
  AttributionTally *const tally;
};

/// The read-out array behind a value, looking through the loop an
/// `hmx.unpack_acc` result travels in: the bridge emits the unpack inside an
/// `scf.for` (one call per row-pair), so a consumer sees the loop's result.
static Value readoutBehind(Value v, scf::ForOp &bridgeLoop,
                           hmx::UnpackAccOp &unpack) {
  if (auto loop = v.getDefiningOp<scf::ForOp>())
    for (auto [i, res] : llvm::enumerate(loop.getResults()))
      if (res == v)
        if (auto op = loop.getBody()
                          ->getTerminator()
                          ->getOperand(i)
                          .getDefiningOp<hmx::UnpackAccOp>()) {
          bridgeLoop = loop;
          unpack = op;
          return op.getSrc();
        }
  return {};
}

//===----------------------------------------------------------------------===//
// Layout propagation
//===----------------------------------------------------------------------===//
//
// A crouton array is a permutation of a logical 2D matrix, so any *elementwise*
// map commutes with it: a map that touches neither indices nor neighbours can
// run on the array directly, and the read-out of one matmul can feed the next
// with no unpack/pack pair in between. This is the "propagation" half: it walks
// the all-parallel producer chain of a bridge (a `hmx.pack_*`) and re-hosts it
// on the crouton iteration space, so the bridge itself disappears.
//
// The rules are the narrow version of Triton's RemoveLayoutConversions
// (docs/hmx/layout-representation-survey.md section 4.3), narrowed further
// by measurement (docs/history/hmx/s1-layout-native-plan.md: the 2026-09-18 LWP
// showed the mask/phi work costs ~10x in crouton order, regressing naive
// linear attention 2.56x, while the removed round trip is <1%):
//
//   * propagate only through *cheap pure-f16* maps (`isCheapF16Elementwise`):
//     all-parallel, one result, all operands/results f16, body limited to
//     `addf`/`subf`/`mulf`/`divf` plus constants. Anything else -- an i1 mask,
//     an i32 index vector, `cmpi`/`select`, `index`, transcendentals, `extf`/
//     `trunc`, max/min-style ops -- stops the fold and stays row-major;
//   * a single-use producer is re-hosted; a shared one is left alone.

/// f16 is the only element type the crouton encoding admits, so only an f16
/// tensor can be a layout operand. `Type::isF16()` is the element type query;
/// a `RankedTensorType` is never itself f16, hence the explicit unwrap.
static bool isF16Tensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getElementType().isF16();
}

/// An elementwise map that commutes with the layout: the whole propagation
/// whitelist. Reductions (any non-parallel iterator) and index readers are the
/// boundary; multi-result maps are conservative stops.
static bool isTransparentMap(linalg::GenericOp map) {
  if (!llvm::all_of(map.getIteratorTypesArray(), [](utils::IteratorType t) {
        return t == utils::IteratorType::parallel;
      }))
    return false;
  if (map.getNumDpsInits() != 1 || map->getNumResults() != 1)
    return false;
  bool usesIndex = false;
  map.walk([&](linalg::IndexOp) { usesIndex = true; });
  return !usesIndex;
}

/// A map cheap enough to run in crouton order: pure f16 elementwise arithmetic
/// and nothing else.
///
/// This is deliberately a whitelist, not a blacklist. Measured on device (LWP,
/// 2026-09-18): moving a mask/phi chain (`cmpi`/`select`/`extf`/`exp`) into
/// crouton order costs ~10x the same work in row-major, regressing naive
/// linear attention 2.56x -- while the pack/unpack round trip it removes is
/// <1% of runtime. So only `addf`/`subf`/`mulf`/`divf` on f16 tensors (plus an
/// in-body splat constant) may be re-hosted. Everything else stays row-major:
///
///   * any i1/i32 operand or result (masks, index vectors) -- the crouton
///     encoding is f16-only, and a full-size i1/i32 tensor read through a
///     strided projection scalarises;
///   * `cmpi`/`select`/`index`/transcendentals (`exp`/`log`/`sqrt`/...) in the
///     body -- the mask/phi work that proved expensive in crouton order;
///   * `extf`/`trunc` -- a precision conversion is not free elementwise work,
///     and the f32 side it implies cannot carry the encoding anyway;
///   * max/min-style ops (`maximumf`/`minimumf`/`cmpf`-then-`select`, ...) --
///     rejected by the whitelist along with everything not listed.
static bool isCheapF16Elementwise(linalg::GenericOp map) {
  if (!isTransparentMap(map))
    return false;
  for (Value operand : map.getDpsInputs()) {
    auto tensor = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensor || !tensor.getElementType().isF16())
      return false;
  }
  for (Value init : map.getDpsInits()) {
    auto tensor = dyn_cast<RankedTensorType>(init.getType());
    if (!tensor || !tensor.getElementType().isF16())
      return false;
  }
  for (Value result : map->getResults()) {
    auto tensor = dyn_cast<RankedTensorType>(result.getType());
    if (!tensor || !tensor.getElementType().isF16())
      return false;
  }
  for (Operation &inner : map.getRegion().front().without_terminator()) {
    if (!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::ConstantOp>(inner))
      return false;
  }
  return true;
}

/// The crouton element (t0, t1, j, col, h) holds logical
/// (t0*32 + 2j + h, t1*32 + col): axis 0 is the pair, axis 1 the stride. This
/// is the map an operand indexing map is composed with when the operand stays
/// row-major.
static AffineMap croutonToLogicalMap(MLIRContext *ctx) {
  AffineExpr t0 = getAffineDimExpr(0, ctx), t1 = getAffineDimExpr(1, ctx);
  AffineExpr j = getAffineDimExpr(2, ctx), col = getAffineDimExpr(3, ctx),
             h = getAffineDimExpr(4, ctx);
  return AffineMap::get(
      5, 0,
      {t0 * HmxTarget::tileEdge + j * hmx::layout::kCroutonHalf + h,
       t1 * HmxTarget::tileEdge + col},
      ctx);
}

/// The logical 2D shape of a crouton array, or nullopt when the tensor does not
/// carry the encoding.
static bool logicalMatchesCrouton(Type type, RankedTensorType crouton) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  auto encoding = hmx::getCroutonEncoding(crouton);
  if (!tensor || !encoding || tensor.getRank() != 2)
    return false;
  return tensor.getDimSize(0) == encoding.getLogical()[0] &&
         tensor.getDimSize(1) == encoding.getLogical()[1];
}

/// True when re-hosting `v` on `crouton` would reach an engine read-out, i.e.
/// the fold would actually remove a bridge. Without this gate a pack of an
/// argument-derived chain would also "fold" (into projected reads of its
/// inputs), which is not a layout win and must not happen.
static bool rehostReachesReadout(Value v, RankedTensorType crouton) {
  scf::ForOp loop;
  hmx::UnpackAccOp unpack;
  if (Value readout = readoutBehind(v, loop, unpack))
    return hmx::sameCroutonEncoding(readout.getType(), crouton);
  auto map = v.getDefiningOp<linalg::GenericOp>();
  if (!map || !isTransparentMap(map) ||
      !logicalMatchesCrouton(v.getType(), crouton) || !v.hasOneUse())
    return false;
  // Only an f16 operand can be a crouton (the encoding is f16-only); the rest
  // are read through projections and cannot carry the read-out.
  return llvm::any_of(map.getDpsInputs(), [&](Value operand) {
    return isF16Tensor(operand.getType()) &&
           rehostReachesReadout(operand, crouton);
  });
}

/// A generic whose body is exactly `yield(input)`: a pure relabel/reshape/
/// broadcast. Materialising a rank-5 copy of one adds only address arithmetic,
/// so the propagation strips it and projects its (usually tiny) base instead.
static bool isRelabelMap(linalg::GenericOp g) {
  if (!isTransparentMap(g) || g.getNumDpsInputs() != 1 ||
      g.getNumDpsInits() != 1)
    return false;
  Block &body = g.getRegion().front();
  if (!llvm::hasSingleElement(body.getOperations()))
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.front());
  return yield && yield.getOperand(0) == body.getArgument(0);
}

/// The value of a *splat* float tensor (an `arith.constant` dense splat or a
/// `linalg.fill` of a scalar constant). Carrying such an operand through the
/// crouton iteration space would read M*N values through a projection just to
/// broadcast one scalar; it is materialised as a scalar in the map body
/// instead.
static std::optional<APFloat> splatScalar(Value v) {
  if (auto fill = v.getDefiningOp<linalg::FillOp>()) {
    Value scalar = fill.getDpsInputOperand(0)->get();
    if (auto cst = scalar.getDefiningOp<arith::ConstantOp>())
      if (auto f = dyn_cast<FloatAttr>(cst.getValue()))
        return f.getValue();
  }
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
      if (dense.isSplat() && dense.getType().getElementType().isF32())
        return dense.getSplatValue<APFloat>();
  return std::nullopt;
}

/// True when `type` is a rank-2 tensor whose logical shape is `target`'s grid
/// times the tile edge -- i.e. the same matrix in row-major.
static bool logicalMatchesTarget(Type type, RankedTensorType target) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || tensor.getRank() != target.getRank() - 3)
    return false;
  for (int64_t i = 0; i < tensor.getRank(); ++i)
    if (tensor.getDimSize(i) != target.getDimSize(i) * HmxTarget::tileEdge)
      return false;
  return true;
}

/// True when the whole producer tree behind `v` is cheap pure-f16 elementwise
/// work reaching an engine read-out -- i.e. the fold moves nothing but f16
/// arithmetic into crouton order. This is the gate that keeps masks and feature
/// maps row-major: unlike `rehostReachesReadout` (which is satisfied by *one*
/// f16 path to a read-out) every node must pass `isCheapF16Elementwise`, so a
/// single `select`/`cmpi`/`exp`/`extf` anywhere in the chain vetoes the fold.
///
/// Leaves: an engine read-out or an already-crouton value is free; a splat
/// scalar is materialised in the fused body, not read through a projection, so
/// it is free too -- but only in f16. Any other leaf (a full-size i1/i32/f32
/// tensor, a non-splat constant, a block argument, ...) would become a
/// full-size strided projection read, so it vetoes. Non-generic producers that
/// are none of the above veto as well.
static bool chainIsCheapF16Only(Value v, RankedTensorType crouton,
                                int depth = 0) {
  if (depth > 16)
    return false;
  if (v.getType() == crouton)
    return true;
  {
    scf::ForOp loop;
    hmx::UnpackAccOp unpack;
    if (Value readout = readoutBehind(v, loop, unpack))
      return hmx::sameCroutonEncoding(readout.getType(), crouton);
  }
  if (splatScalar(v)) {
    auto tensor = dyn_cast<RankedTensorType>(v.getType());
    return tensor && tensor.getElementType().isF16();
  }
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
      if (dense.isSplat()) {
        auto tensor = dyn_cast<RankedTensorType>(v.getType());
        return tensor && tensor.getElementType().isF16();
      }
    return false;
  }
  auto producer = v.getDefiningOp<linalg::GenericOp>();
  if (!producer || !logicalMatchesCrouton(v.getType(), crouton) ||
      !v.hasOneUse())
    return false;
  if (!isCheapF16Elementwise(producer))
    return false;
  return llvm::all_of(producer.getDpsInputs(), [&](Value operand) {
    return chainIsCheapF16Only(operand, crouton, depth + 1);
  });
}

/// Re-host `v` so its result is the rank-5 layout `target`, fusing the whole
/// elementwise producer tree into one `linalg.generic`. Returns null when `v`
/// is not an elementwise producer of that layout, in which case the caller
/// falls back to a projection on this edge. `dead` collects the originals whose
/// body was cloned (and the read-out loops that lost their last use).
///
/// The caller (`FoldElementwiseIntoLayout`) only reaches here after
/// `chainIsCheapF16Only` has vetted the whole tree, so in practice every
/// inlined node is cheap f16 arithmetic. The inline gate below re-checks
/// `isCheapF16Elementwise` anyway (belt and braces): an expensive producer that
/// somehow gets here becomes a row-major projection leaf instead of crouton-
/// order work -- never incorrect, only less fused.
static Value rehostAsCrouton(RewriterBase &b, Value v, RankedTensorType target,
                             SmallVectorImpl<Operation *> &dead,
                             int depth = 0) {
  if (v.getType() == target)
    return v;
  // A guard against a diamond-shaped chain duplicating without bound; a deeper
  // chain simply falls back to a projection (never incorrect, only less fused).
  if (depth > 16)
    return {};

  // An engine read-out (f16) is already the crouton it was unpacked from.
  {
    scf::ForOp loop;
    hmx::UnpackAccOp unpack;
    if (Value readout = readoutBehind(v, loop, unpack)) {
      if (readout.getType() != target)
        return {};
      dead.push_back(loop);
      return readout;
    }
  }

  AffineMap toLogical = croutonToLogicalMap(b.getContext());
  unsigned rootRank = target.getRank();
  unsigned logicalRank = rootRank - 3;
  AffineMap identity2 = b.getMultiDimIdentityMap(logicalRank);

  // The fused expression tree rooted at `v`: every cheap f16 producer becomes
  // an `Inline` node and the whole tree lowers to a *single* `linalg.generic`.
  // Composing the tree (rather than re-hosting each producer as its own rank-5
  // op) is what keeps the fused segment from materialising a rank-5
  // intermediate: only its leaves are operands. Anything not cheap (a mask, an
  // index chain, a conversion, a transcendental) never becomes `Inline` -- it
  // stays a row-major value read through a projection, or vetoes the fold
  // outright via `chainIsCheapF16Only`.
  enum Kind { Operand, Splat, Inline };
  struct Node {
    Kind kind = Operand;
    bool crouton = false; // Operand: already in the engine layout
    unsigned slot = 0;    // Operand: index into `operands`
    std::optional<APFloat> value;
    Type type;                  // Splat element type
    linalg::GenericOp producer; // Inline
    SmallVector<unsigned> children;
  };
  SmallVector<Node> nodes;
  SmallVector<Value> operands;
  SmallVector<AffineMap> maps;
  constexpr unsigned kNodeBudget = 128;

  std::function<std::optional<unsigned>(Value, AffineMap, int)> build =
      [&](Value value, AffineMap mapToValue, int d) -> std::optional<unsigned> {
    if (d > 16 || nodes.size() > kNodeBudget)
      return std::nullopt;

    Node node;
    auto addLeaf = [&](Value leaf, bool crouton,
                       AffineMap projection) -> unsigned {
      node.kind = Operand;
      node.crouton = crouton;
      node.slot = operands.size();
      operands.push_back(leaf);
      maps.push_back(crouton ? b.getMultiDimIdentityMap(rootRank)
                             : projection.compose(toLogical));
      nodes.push_back(node);
      return nodes.size() - 1;
    };

    if (value.getType() == target)
      return addLeaf(value, /*crouton=*/true, AffineMap());
    scf::ForOp loop;
    hmx::UnpackAccOp unpack;
    if (Value readout = readoutBehind(value, loop, unpack)) {
      if (readout.getType() != target)
        return std::nullopt;
      dead.push_back(loop);
      return addLeaf(readout, /*crouton=*/true, AffineMap());
    }
    // Strip broadcast/reshape relabels: a pure broadcast carries no
    // computation, so its rank-5 copy is pure address arithmetic.
    while (true) {
      auto relabel = value.getDefiningOp<linalg::GenericOp>();
      if (!relabel || !isRelabelMap(relabel))
        break;
      Value input = relabel.getDpsInputOperand(0)->get();
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      auto outputType = dyn_cast<RankedTensorType>(value.getType());
      if (!inputType || !outputType ||
          inputType.getShape() == outputType.getShape())
        break;
      mapToValue = relabel.getIndexingMapsArray()[0].compose(mapToValue);
      value = input;
    }
    if (value.getType() == target)
      return addLeaf(value, /*crouton=*/true, AffineMap());
    if (auto splat = splatScalar(value)) {
      node.kind = Splat;
      node.value = splat;
      node.type = cast<RankedTensorType>(value.getType()).getElementType();
      nodes.push_back(node);
      return nodes.size() - 1;
    }
    auto producer = value.getDefiningOp<linalg::GenericOp>();
    if (producer && isCheapF16Elementwise(producer) &&
        logicalMatchesTarget(value.getType(), target)) {
      node.kind = Inline;
      node.producer = producer;
      unsigned self = nodes.size();
      nodes.push_back(node);
      for (auto [index, operand] : llvm::enumerate(producer.getDpsInputs())) {
        AffineMap childMap =
            producer.getIndexingMapsArray()[index].compose(mapToValue);
        auto child = build(operand, childMap, d + 1);
        if (!child)
          return std::nullopt;
        nodes[self].children.push_back(*child);
      }
      // Duplicated rather than stolen (the body is cloned), so a value shared
      // with a row-major consumer keeps its own copy.
      dead.push_back(producer);
      return self;
    }
    return addLeaf(value, /*crouton=*/false, mapToValue);
  };

  std::optional<unsigned> root = build(v, identity2, depth);
  if (!root)
    return {};

  maps.push_back(b.getMultiDimIdentityMap(target.getRank()));
  Location loc =
      v.getDefiningOp() ? v.getDefiningOp()->getLoc() : b.getUnknownLoc();
  // Only the engine's own layout lives in VTCM; the i1/i32/f32 intermediates
  // are ordinary tensors and must not claim a VTCM buffer.
  Value init = target.getElementType().isF16()
                   ? vtcmEmpty(b, loc, target)
                   : tensor::EmptyOp::create(b, loc, target.getShape(),
                                             target.getElementType());
  SmallVector<utils::IteratorType> iterators(target.getRank(),
                                             utils::IteratorType::parallel);
  auto laidOut = linalg::GenericOp::create(
      b, loc, TypeRange{target}, operands, ValueRange{init}, maps, iterators,
      [&](OpBuilder &builder, Location l, ValueRange args) {
        SmallVector<Value> valueOf(nodes.size());
        std::function<Value(unsigned)> emit = [&](unsigned id) -> Value {
          if (valueOf[id])
            return valueOf[id];
          Node &n = nodes[id];
          if (n.kind == Operand) {
            valueOf[id] = args[n.slot];
          } else if (n.kind == Splat) {
            valueOf[id] = arith::ConstantOp::create(
                builder, l, builder.getFloatAttr(n.type, *n.value));
          } else {
            SmallVector<Value> in;
            for (unsigned child : n.children)
              in.push_back(emit(child));
            IRMapping mapping;
            Block &body = n.producer.getRegion().front();
            for (unsigned i = 0; i < in.size(); ++i)
              mapping.map(body.getArgument(i), in[i]);
            for (Operation &inner : body.without_terminator())
              builder.clone(inner, mapping);
            valueOf[id] =
                mapping.lookupOrDefault(body.getTerminator()->getOperand(0));
          }
          return valueOf[id];
        };
        linalg::YieldOp::create(builder, l, emit(*root));
      });
  return laidOut.getResult(0);
}

/// A matmul chained off another matmul needs no bridge: the read-out layout AR
///
/// Note on the shape check below: for a legal chain the grids always match --
/// K of the consumer *is* N of the producer, and M is the same -- so the check
/// is a structural assertion rather than a filter that can reject a valid
/// chain. and the activation layout AH are the same permutation, so when the
/// crouton grids line up the read-out array *is* a valid activation array, and
/// the pack (plus the unpack it would consume) disappear.
///
/// This is a pattern on the pack rather than logic inside the matmul rewrite
/// because the greedy driver may visit the consumer before the producer: the
/// bridge only exists after both rewrites have run.
///
/// Verified on device with a same-input A/B (fixed seed, 128x128, folded vs
/// unfolded): every element agrees (0 of 16384 differ, max delta 0.0).
/// Comparing either side against a torch fp16 reference instead is misleading
/// -- the engine accumulates in 37 bits, torch in fp16 -- which made this look
/// broken twice.
struct FoldChainedPack : public OpRewritePattern<hmx::PackActOp> {
  using OpRewritePattern<hmx::PackActOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hmx::PackActOp op,
                                PatternRewriter &rewriter) const override {
    scf::ForOp loop;
    hmx::UnpackAccOp unpack;
    Value readout = readoutBehind(op.getSrc(), loop, unpack);
    if (!readout)
      return rewriter.notifyMatchFailure(op, "not a read-out");
    // The layout contract lives in the dialect: AR, AH and WH are one layout,
    // so "same crouton shape" is what makes a producer's read-out usable as
    // this matmul's activation with no conversion.
    if (!hmx::sameCroutonEncoding(readout.getType(), op.getDst().getType()))
      return rewriter.notifyMatchFailure(op, "crouton grids differ");

    rewriter.replaceOp(op, readout);
    if (loop && loop->use_empty())
      rewriter.eraseOp(loop); /* takes the unpack in its body with it */
    return success();
  }
};

/// A cheap pure-f16 elementwise map between two matmuls, applied to the first
/// one's read-out, needs no conversion either: a map that touches neither
/// indices nor neighbours commutes with the layout permutation, so it can run
/// directly on the crouton array. Both bridges (the unpack that would read the
/// read-out and the pack that would feed the next matmul) disappear.
///
/// This is the second half of the layout propagation: the first
/// (FoldChainedPack) handles a bare matmul->matmul chain, this one the cheap
/// elementwise segment in between -- e.g. an f16 scale. Masks, feature maps
/// and normalisation deliberately stay row-major: moving them into crouton
/// order costs ~10x (LWP, 2026-09-18), so `chainIsCheapF16Only` vetoes any
/// chain containing an i1/i32 value, `cmpi`/`select`, `index`,
/// transcendentals, `extf`/`trunc` or max/min-style ops. Reductions are
/// likewise *not* fused: a row-sum spans croutons, so it needs the conversion.
/// That boundary is the co-scheduling point, not a compromise.
///
/// Unlike the old one-level fold this walks the whole producer chain
/// (`rehostAsCrouton`), so a chain `read-out -> map1 -> map2 -> pack` of cheap
/// f16 maps folds as one.
struct FoldElementwiseIntoLayout : public OpRewritePattern<hmx::PackActOp> {
  using OpRewritePattern<hmx::PackActOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hmx::PackActOp op,
                                PatternRewriter &rewriter) const override {
    auto crouton = dyn_cast<RankedTensorType>(op.getDst().getType());
    if (!crouton || !hmx::hasCroutonEncoding(crouton))
      return rewriter.notifyMatchFailure(op, "destination is not a crouton");
    if (!rehostReachesReadout(op.getSrc(), crouton))
      return rewriter.notifyMatchFailure(op, "no read-out behind the pack");
    if (!chainIsCheapF16Only(op.getSrc(), crouton))
      return rewriter.notifyMatchFailure(op, "chain is not cheap f16-only");

    // Emit the re-hosted chain where the row-major chain was, not at the pack:
    // the pack sits inside its own tile loop (a single carried buffer), so
    // building the replacement there would leave a second alloc in the loop and
    // break the loop-carried buffer's memory space.
    Operation *anchor = op.getSrc().getDefiningOp();
    if (!anchor)
      return rewriter.notifyMatchFailure(op, "source has no defining op");
    rewriter.setInsertionPoint(anchor);
    SmallVector<Operation *> dead;
    Value laidOut = rehostAsCrouton(rewriter, op.getSrc(), crouton, dead);
    if (!laidOut)
      return rewriter.notifyMatchFailure(
          op, "cannot re-host the elementwise chain");

    // The pack sat inside the tile loop that `packCroutonsWithLeaves` emits.
    // With the pack gone that loop only carries the re-hosted value, and a
    // loop-carried buffer of a different space is exactly what makes
    // bufferization keep a space-0 copy: hand its result to the consumer and
    // drop the loop.
    scf::ForOp packLoop = op->getParentOfType<scf::ForOp>();
    rewriter.replaceOp(op, laidOut);
    if (packLoop && packLoop.getNumResults() == 1 &&
        llvm::all_of(
            packLoop.getBody()->without_terminator(),
            [](Operation &inner) { return isMemoryEffectFree(&inner); }))
      rewriter.replaceOp(packLoop, laidOut);

    // `dead` is built inner-first (a re-hosted producer before the map that
    // consumed it), so erase in reverse: the map goes first, which is what
    // frees the read-out loop it used to consume.
    for (Operation *op : llvm::reverse(dead))
      if (op->use_empty())
        rewriter.eraseOp(op); /* a loop takes its unpack in its body with it */
    return success();
  }
};

/// Erase every `#hmx.crouton` encoding once the propagation is done. The
/// encoding is a tensor-stage annotation: default bufferization drops it and
/// produces fully dynamic strides (docs/hmx/layout-representation-survey.md
/// section 1.3), so by default it must not survive this pass. Physical shape
/// and element type are unchanged, so this is a pure metadata erasure -- it
/// restores exactly the types the pipeline produced before the encoding
/// existed. (`drop-encodings=0` keeps them for a bufferizer that maps them to
/// the identity-map `#hmx.crouton_memref_layout` instead.)
static void dropCroutonEncodings(Operation *root) {
  auto drop = [](Type type) -> Type {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.getEncoding())
      return type;
    return RankedTensorType::get(tensor.getShape(), tensor.getElementType());
  };
  root->walk([&](Operation *op) {
    // `arith.constant` carries the type in its value attribute too, so it has
    // to be rebuilt or the op verifier rejects the mutation.
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue())) {
        auto tensor = dyn_cast<RankedTensorType>(dense.getType());
        if (tensor && tensor.getEncoding())
          constant->setAttr("value",
                            DenseElementsAttr::getFromRawBuffer(
                                RankedTensorType::get(tensor.getShape(),
                                                      tensor.getElementType()),
                                dense.getRawData()));
      }
    }
    for (Value result : op->getResults())
      result.setType(drop(result.getType()));
    // `scf.for` loop-carried values (and the function entry) are block
    // arguments; both move with the value's type.
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          arg.setType(drop(arg.getType()));
  });
}

struct MatmulToHmxPass
    : public mlir::hmx::impl::MatmulToHmxBase<MatmulToHmxPass> {
  using mlir::hmx::impl::MatmulToHmxBase<MatmulToHmxPass>::MatmulToHmxBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // The pass emits only hmx/arith/linalg/scf/tensor ops: the crouton arrays
    // go through the dialect's own `hmx.alloc_crouton`, not through the
    // bufferization dialect.
    registry.insert<hmx::HmxDialect, arith::ArithDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    // The manifest is module state while this interface pass is nested under
    // func.func. MLIR may invoke those function instances concurrently; keep
    // the rewrite/read/refresh transaction serial so records cannot race.
    std::lock_guard<std::mutex> manifestGuard(hmxModuleStateMutex());

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE << "] running\n");

    // The engine's contract, with the fields a caller may narrow: the VTCM
    // budget (0 meaning the device default, see HmxTarget) and whether this
    // pipeline provides the VTCM allocator the crouton arrays need.
    HmxTarget target;
    if (vtcmBudgetBytes > 0)
      target.vtcmBudget = vtcmBudgetBytes;
    // Record-only is a metadata pass, not a partial HMX lowering. It must not
    // claim that VTCM is usable when no HMX allocation will be emitted.
    target.vtcmAllocator = recordOnly ? false : vtcmAllocator;

    auto func = cast<func::FuncOp>(getOperation());
    auto module = func->getParentOfType<ModuleOp>();
    if (!module) {
      getOperation()->emitError(
          "matmul-to-hmx requires a func.func inside a builtin.module");
      return signalPassFailure();
    }

    bool invalidDiagnosticMarker = false;
    func.walk([&](linalg::MatmulOp matmul) {
      if (matmul->hasAttr(kHmxDiagnosticTailAttr) &&
          !isHmxDiagnosticTailMarker(matmul.getOperation())) {
        matmul.emitError(
            "hmx.diagnostic_tail_partition must be a unit attribute");
        invalidDiagnosticMarker = true;
      }
    });
    if (invalidDiagnosticMarker)
      return signalPassFailure();

    // Pass-scoped, exactly as long as the rewrite. It is populated before the
    // greedy driver and keyed by the original op, so repeated pattern attempts
    // cannot duplicate a record.
    AttributionTally tally(func, target, recordOnly);

    if (!recordOnly) {
      RewritePatternSet patterns(&getContext());
      patterns.add<MatmulToHmx>(&getContext(), target, &tally);
      patterns.add<FoldChainedPack, FoldElementwiseIntoLayout>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
    }

    SmallVector<DictionaryAttr> records;
    records.reserve(tally.records().size());
    for (const MatmulDecision &decision : tally.records())
      if (tally.describes(decision))
        records.push_back(manifestRecord(&getContext(), decision, target));
    if (failed(addOrReplaceHmxManifestRecords(module, records)) ||
        failed(refreshHmxManifestBridgeCounts(module)))
      return signalPassFailure();

    // The record-only v3 document is produced from the same decisions through a
    // separate builder, and is published only when the internal record-mode
    // marker is present.  With the marker absent this is a no-op, so the v2
    // manifest bytes and every launcher decision are untouched.
    if (failed(publishHmxRecordV3(module, hmxRecordV3Skeletons(
                                         &getContext(), tally))))
      return signalPassFailure();

    // The manifest and the diagnostics are two views of the same final records.
    // In particular, no remark is emitted while the greedy driver is still
    // changing a decision. Record-only is intentionally silent: its expected
    // allocator-disabled records describe metadata, not a user-visible refusal.
    if (!recordOnly)
      emitDecisionDiagnostics(module, target, tally);

    // The encoding is a tensor-stage annotation; bufferization would drop it
    // and fall back to fully dynamic strides (survey section 1.3), so erase it
    // here by default, leaving the rank-5 physical types the rest of the
    // pipeline knows. `drop-encodings=0` keeps them for the bufferizer that
    // carries the layout over as #hmx.crouton_memref_layout. Record-only never
    // drops encodings because it promises not to rewrite the input IR.
    if (!recordOnly && dropEncodings)
      dropCroutonEncodings(getOperation());
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hmx::createMatmulToHmxPass(const MatmulToHmxOptions &options) {
  return std::make_unique<MatmulToHmxPass>(options);
}
