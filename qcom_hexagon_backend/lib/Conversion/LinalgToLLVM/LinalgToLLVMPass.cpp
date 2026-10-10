//===- LinalgToLLVMPass.cpp - Linalg to LLVM  conversion       ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements optimization and lowering of MLIR IR to LLVM.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h"
#include "hexagon/Conversion/HmxToLLVM/HmxToLLVM.h"
#include "hexagon/Conversion/HvxToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/Passes.h"
#include "hexagon/Dialect/Hvx/IR/HvxDialect.h"
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "hexagon/Transforms/Passes.h"

#include <algorithm>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "linalg-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LINALGTOLLVM
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

bool hasTensorValue(Operation *op) {
  for (Value value : op->getOperands())
    if (isa<RankedTensorType, UnrankedTensorType>(value.getType()))
      return true;
  for (Value value : op->getResults())
    if (isa<RankedTensorType, UnrankedTensorType>(value.getType()))
      return true;
  return false;
}

bool containsTensorLinalgOp(ModuleOp module) {
  bool found = false;
  module.walk([&](linalg::LinalgOp op) {
    if (hasTensorValue(op.getOperation()))
      found = true;
  });
  return found;
}

struct LinalgToLLVMPass : public ::impl::LinalgToLLVMBase<LinalgToLLVMPass> {
public:
  explicit LinalgToLLVMPass(const LinalgToLLVMOptions &options)
      : Base(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, async::AsyncDialect, tensor::TensorDialect,
                    cf::ControlFlowDialect, bufferization::BufferizationDialect,
                    vector::VectorDialect, memref::MemRefDialect,
                    LLVM::LLVMDialect, ttx::TTXDialect,
                    hexagonmem::HexagonMemDialect, hmx::HmxDialect,
                    hvx::HvxDialect, quant::QuantDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // The record-only path is intentionally available to hand-written memref
    // IR, but it cannot translate the tensor ABI produced by Triton: the HVX
    // pipeline needs a bufferized memref entry before it can lower the op. Do
    // this check at the translation boundary, over every Linalg operation, so
    // batch/generic tensor paths cannot evade the guard via rank reduction.
    if (!enableBufferization && containsTensorLinalgOp(moduleOp)) {
      moduleOp.emitError(
          "tensor-valued Linalg operations require enableBufferization=true; "
          "the no-bufferization path accepts manually managed memrefs only");
      signalPassFailure();
      return;
    }

    // SCF threading rewrites the loop nest into scf.parallel to be picked up by
    // virtual threads; multi-threading, VTCM tiling and the external scratch
    // buffer each expect to be the sole owner of that nest / of the per-instance
    // VTCM budget. Combining them is an undefined pipeline, so reject it loudly
    // here instead of asserting: an assert is compiled out of Release/NDEBUG
    // builds and would silently emit the broken pipeline.
    if (enableSCFThreading &&
        (enableMultiThreading || enableVTCMTiling || scratch > 0)) {
      moduleOp.emitError(
          "enableSCFThreading is incompatible with enableMultiThreading, "
          "enableVTCMTiling and scratch>0; scf-threading can be enabled only "
          "if linalg multi-threading, vtcm tiling and the external scratch "
          "buffer are all off");
      signalPassFailure();
      return;
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    addProductionPasses(pm, moduleOp);

    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }

  /// Set by the diagnostic entry point through
  /// `runLinalgToLLVMPipeline`. Empty on the production pass, which has no such
  /// stage. A `std::function` rather than a `function_ref` because the pass may
  /// be cloned (and run) after the caller's lambda has gone out of scope.
  std::function<void(PassManager &)> diagnosticStage;

  /// Append the whole production sequence to `pm`.
  ///
  /// The pass owns the guards and the run; the diagnostic entry point
  /// (`hmx-diagnostic-record`, in the HmxDiagnostics library) calls this and
  /// then appends the two diagnostic passes at the stage they belong. Keeping
  /// the sequence here is what lets the diagnostics live outside this file and
  /// outside `libtriton.so`: there is exactly one copy of the production order,
  /// and it does not reference the diagnostic passes at all.
  ///
  /// `moduleOp` is the module the passes will run on; it is used for the target
  /// triple and data layout, which the caller's IR already depends on.
  void addProductionPasses(PassManager &pm, ModuleOp moduleOp,
                            llvm::function_ref<void(PassManager &)>
                                atDiagnosticStage = {});
};

} // namespace

void LinalgToLLVMPass::addProductionPasses(
    PassManager &pm, ModuleOp moduleOp,
    llvm::function_ref<void(PassManager &)> atDiagnosticStage) {
    MLIRContext *context = moduleOp.getContext();

    setTargetTriple(moduleOp);
    setDataLayout(moduleOp);

    auto setIndexBitwidth = [&](auto passOption) {
      passOption.indexBitwidth = 32;
      return passOption;
    };

    auto setFusion = [&](auto passOption) {
      passOption.fusionAllowRecompute = fusionAllowRecompute;
      passOption.fusionDoMultiUse = fusionDoMultiUse;
      return passOption;
    };

    auto setExtendPack = [&](auto passOption) {
      passOption.upperFrontier = extendPackUpperFrontier;
      passOption.lowerFrontier = extendPackLowerFrontier;
      passOption.parallelsOnly = extendPackParallelsOnly;
      return passOption;
    };

    auto setVTCMTiling = [&](auto passOption) {
      passOption.tileSizes = tileSizes;
      passOption.vtcmBudget = scratch > 0 ? scratch : 0;
      return passOption;
    };

    auto setuseInterchangeVector = [&](auto passOption) {
      passOption.useInterchangeVector = useInterchangeVector;
      return passOption;
    };

    auto setOpSlicingFactor = [&](auto passOption) {
      passOption.slicingFactor = slicingFactor;
      return passOption;
    };

    auto setsplitTilingRange = [&](auto passOption) {
      passOption.splitTilingRange = splitTilingRange;
      return passOption;
    };

    auto setenableSplitReduction = [&](auto passOption) {
      passOption.enableSplitReduction = enableSplitReduction;
      return passOption;
    };

    auto setLWP = [&](auto passOption) {
      passOption.disableLWPLoop = disableLWPLoop;
      passOption.LWPloopDepth = LWPloopDepth;
      return passOption;
    };

    auto setDeviceType = [&](auto passOption) {
      passOption.device_type = device_type;
      return passOption;
    };

    // Set ConvTiling flags
    auto setConvTiling = [&](auto passOption) {
      passOption.convTileSizes = convTileSizes;
      return passOption;
    };


    // RequestCWrappersPass adds an attribute to a function if it has a return
    // value which would generate a c-wrapper function during the
    // FuncToLLVMPass. See here for more information:
    // https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
    //
    // For example given the following function definition:
    // func.func @foobar(%arg0: memref<128x128xf32>)
    // -> memref<128x128xf32>
    //
    // After the RequestCWrappersPass it is converted to,
    // func.func @foobar(%arg0: memref<128x128xf32>)
    // -> memref<128x128xf32> attributes {llvm.emit_c_interface}
    //
    // After FuncToLLVMPass we get an additional function,
    // llvm.func @_mlir_ciface_foobar(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
    // attributes {llvm.emit_c_interface} {
    //   %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64,
    //   array<2 x i64>, array<2 x i64>)>
    //   %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x
    //   i64>, array<2 x i64>)>
    //   ...
    //   %8 = llvm.call @foobar(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr,
    //   !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64,
    //   array<2 x i64>, array<2 x i64>)>
    //
    //   llvm.store %8, %arg0 :
    //   !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>,
    //   !llvm.ptr
    //
    //   llvm.return
    // }
    //
    if (doesFuncReturnValue(moduleOp))
      pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());

    pm.addNestedPass<func::FuncOp>(createLowerTTXPass());
    pm.addPass(createLowerLibdevicePass());
    pm.addNestedPass<func::FuncOp>(createHexagonLowerTmTensorPass());
    pm.addNestedPass<func::FuncOp>(createReduceContractionRankPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (puntBuffer)
      pm.addNestedPass<func::FuncOp>(createHexagonPuntBufferPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung allocs

    if (enableConversionToFp16)
      pm.addNestedPass<func::FuncOp>(createConversionToFp16Pass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addNestedPass<func::FuncOp>(createOptimizeExtfTruncfOpPass());

    // Optimize division to multiplication in linalg.generic
    pm.addNestedPass<func::FuncOp>(createDivToMulOptimizationPass());

    // Quantization related passes in this block
    // Lower quant.qcast and quant.dcast ops to arith dialect
    pm.addNestedPass<func::FuncOp>(quant::createLowerQuantOps());
    // Convert arith ops to linalg elementwise ops
    pm.addPass(createConvertElementwiseToLinalgPass());
    // Remove quant.scast ops
    pm.addPass(createCSEPass());
    // HMX engine attribution is always scheduled so the module manifest
    // describes every linalg.matmul, including runs that intentionally keep
    // manual buffer management. The normal path rewrites eligible matmuls;
    // without bufferization the same tally runs in record-only mode, with VTCM
    // capability disabled, so it cannot leave an hmx.matmul that HmxToLLVM
    // could not lower. Attribution stays before lower-pack, which turns the
    // crouton layouts it seeds into data movement.
    mlir::hmx::MatmulToHmxOptions matmulToHmxOpts;
    matmulToHmxOpts.recordOnly = !enableBufferization;
    matmulToHmxOpts.vtcmAllocator =
        enableBufferization && enableConvertToHexagonmem;
    pm.addNestedPass<func::FuncOp>(
        mlir::hmx::createMatmulToHmxPass(matmulToHmxOpts));
    pm.addPass(createCanonicalizerPass());

    // enableMatmulToConv and enableSeedLayoutConversions are supposed to be set
    // for unit test only. They are not supposed to run on Full models
    if (enableMatmulToConv && enableSeedLayoutConversions) {
      pm.addNestedPass<func::FuncOp>(createMatmulToConvPass());
      pm.addNestedPass<func::FuncOp>(createSeedLayoutConversionsPass());
      pm.addNestedPass<func::FuncOp>(createHexagonExtendPackPass(
          setExtendPack(HexagonExtendPackOptions{})));
      pm.addPass(createCSEPass());
    }

    if (enableConvTiling) {
      pm.addNestedPass<func::FuncOp>(
          createConvTilingPass(setConvTiling(ConvTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    if (enableSeedLayoutConversions) {
      pm.addNestedPass<func::FuncOp>(createPreprocessTiledConv2DPass());
    }

    pm.addNestedPass<func::FuncOp>(createScheduleMatmulForHVXPass());
    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizePass());

    if (returnValueOptimization)
      pm.addNestedPass<func::FuncOp>(createHexagonRVOPass());
    pm.addPass(createCanonicalizerPass()); // erase unstrung re-interprets
    pm.addPass(createCSEPass());

    if (enableSCFThreading) {
      pm.addNestedPass<func::FuncOp>(createFormSCFThreadsPass());
    }

    if (fusion)
      pm.addNestedPass<func::FuncOp>(
          createHexagonFusionPass(setFusion(HexagonFusionOptions{})));
    pm.addPass(createEraseUnusedLinalgOperands());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (enableSlicing)
      pm.addPass(createHexagonSlicingPass(
          setOpSlicingFactor(HexagonSlicingOptions{})));

    pm.addNestedPass<func::FuncOp>(createDecomposeTensorConcatPass());
    if (forceHVXCroutonization) {
      pm.addNestedPass<func::FuncOp>(createForceHVXCroutonPass());
      // setExtendPack unconditionally assigns upperFrontier from the
      // extendPackUpperFrontier option (default true), so it would clobber a
      // designated initializer: run the wrapper first and force upper=false
      // afterwards, otherwise this path silently runs with upper=true.
      auto forceExtendPackOpts = setExtendPack(HexagonExtendPackOptions{});
      forceExtendPackOpts.upperFrontier = false;
      pm.addNestedPass<func::FuncOp>(
          createHexagonExtendPackPass(forceExtendPackOpts));
    }

    pm.addNestedPass<func::FuncOp>(createLowerPackPass());
    pm.addPass(createCSEPass());

    // VTCMTilingPass must run when scratch > 0 to create the VTCM allocs
    // that MemoryOffsetsPass will replace with views into the scratch buffer.
    // When scratch > 0, pass it as vtcmBudget so tile sizes respect the
    // per-instance budget rather than the hardcoded 2 MB default.
    if (enableVTCMTiling || scratch > 0) {
      pm.addNestedPass<func::FuncOp>(
          createVTCMTilingPass(setVTCMTiling(VTCMTilingOptions{})));
      pm.addPass(createCanonicalizerPass());
    }

    // split linalg.reduce into [parallel,reduce] followed by smaller [reduce].
    if (enableSplitReduceGeneric) {
      pm.addNestedPass<func::FuncOp>(createSplitReduceGenericPass());
    }

    if (enableMultiThreading) {
      pm.addNestedPass<func::FuncOp>(
          createFormVirtualThreadsPass(FormVirtualThreadsOptions{}));
    }

    pm.addPass(removeMLProgramPass());
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    if (enableVectorization) {
      pm.addPass(
          createHexagonTilingPass(setsplitTilingRange(setuseInterchangeVector(
              setenableSplitReduction(HexagonTilingOptions{})))));
    }

    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(
        createSmallExponentToMultiplyPass(SmallExponentToMultiplyOptions{}));
    // ===== STEP 1: HOIST SCALAR OPS =====
    // Run before vectorization to expose scalar invariants
    pm.addNestedPass<func::FuncOp>(createHoistScalarOpsPass());
    pm.addPass(createEraseUnusedLinalgOperands());
    pm.addPass(createCSEPass());

    // ===== STEP 1.5: LOOP INVARIANT CODE MOTION =====
    // Move hoisted scalars further up the loop nest
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Run LICM again after canonicalization to catch newly exposed
    // opportunities
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());

    // ===== STEP 2: VECTORIZATION =====
    // Vectorizer now sees cleaner IR with hoisted scalars
    // Vectorization pass for HVX. When the row-reduce butterfly is on, its
    // reduces are left unvectorized here so they reach the butterfly instead
    // of becoming a vector.multi_reduction (whose LLVM lowering is the
    // ExpandReductions valign tree).
    if (enableVectorization) {
      HexagonVectorizationOptions vectorizeOpts;
      vectorizeOpts.skipVectorRowReduce = enableVectorRowReduce;
      pm.addPass(createHexagonVectorizationPass(vectorizeOpts));
    }
    pm.addPass(createRewriteUBPoisonToZeroPass());
    pm.addPass(createHexagonVectorLoweringPass());
    // A row reduction whose per-row result is stored into a rank-0 slice of a
    // tensor<rows x T> is rewritten to keep that result in the vector domain:
    // the row loop steps by a whole HVX vector of rows, the group's running
    // values stay a vector<lanes x T> loop-carried value (which one-shot
    // bufferization leaves alone), the hvx.vror butterfly is not extracted, and
    // an arith.cmpi + arith.select places it in the group's lane. Hexagon HVX has
    // no vector-lane-to-GPR instruction, so a reduction result that becomes a
    // scalar costs one 128 B stack write plus a 4 B stack read back, per row per
    // 128 B chunk. Runs here because createHexagonVectorLoweringPass is what
    // expands vector.multi_reduction into the per-chunk vector.reduction this
    // pass rewrites -- before it, the reduction is still one op per row. Stays
    // before HexagonAddFastMath so the butterfly's own folds carry the nnan it
    // needs rather than depending on that pass to stamp them. Off by default;
    // the knob is the device A/B switch.
    if (enableRowReduceGroupStore)
      pm.addNestedPass<func::FuncOp>(createRowReduceGroupStorePass());
    pm.addPass(createCanonicalizerPass());

    if (addFastMath) {
      pm.addPass(createHexagonAddFastMathPass());
      pm.addNestedPass<func::FuncOp>(createFoldMulFByZeroPass());
      pm.addPass(createCanonicalizerPass());
    }
    pm.addPass(memref::createResolveShapedTypeResultDimsPass());

    if (enableBufferization) {
      pm.addPass(bufferization::createEmptyTensorEliminationPass());

      // Erase unnecessary vector-to-tensor writeback in loops before
      // bufferization.
      pm.addNestedPass<func::FuncOp>(createEraseVectorToTensorWritebackPass());

      mlir::bufferization::OneShotBufferizePassOptions passOpts;
      passOpts.bufferizeFunctionBoundaries = true;
      passOpts.allowReturnAllocsFromLoops = true;
      // The fused HMX tail (`hmx.unpack_acc_f32`) issues aligned 128 B stores
      // into its destination, which bufferization allocates as a fresh
      // internal buffer (same loop structure as the old unpack destination).
      // Raising the pipeline-wide allocation alignment from the upstream
      // default 64 to 128 guarantees that precondition allocator-side, at
      // compile time, with no runtime check. Over-alignment is always safe;
      // VTCM allocations retain their explicit alignment when
      // ConvertToHexagonmem lowers them to hexagonmem.alloc.
      passOpts.bufferAlignment = 128;
      pm.addPass(bufferization::createOneShotBufferizePass(passOpts));
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());

      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            createHexagonDoubleBufferGenericS1Pass());
      }

      pm.addNestedPass<func::FuncOp>(
          bufferization::createBufferLoopHoistingPass());

      pm.addNestedPass<func::FuncOp>(createCopyCanonicalizationPass());
      pm.addPass(createCanonicalizerPass());

      bufferization::buildBufferDeallocationPipeline(
          pm, bufferization::BufferDeallocationPipelineOptions{});

      pm.addPass(createCSEPass());
      if (enableDoubleBuffering) {
        pm.addNestedPass<func::FuncOp>(
            createHexagonDoubleBufferGenericS2Pass());
      }

      // SCF Loop Unrolling of innermost loop after vectorization.
      if (enableSCFLoopUnroll) {
        pm.addNestedPass<func::FuncOp>(createSCFLoopUnrollPass());
      }

      pm.addNestedPass<func::FuncOp>(createConvertZeroSizeMemrefPass());
      pm.addPass(createConvertBufferizationToMemRefPass());
      // The streaming slice loops are in their final form here (subview of
      // the induction variable + vector.transfer_read), and the async
      // formation below clones whole loop bodies, so the fetch calls ride
      // along into every chunk with the chunk's own bounds.
      if (enableL2Prefetch) {
        pm.addNestedPass<func::FuncOp>(createHexagonL2PrefetchPass());
      }
      // A compile-time constant weight arrives here as a `memref.get_global`,
      // which the engine cannot read; give it a resident VTCM buffer before the
      // tile level asks for one (see WeightResidentPass). With
      // `enableWeightResident`, the same pass also makes a runtime weight
      // resident: its per-launch pack bridge is dropped and the host pre-packs
      // it (hmx.weight_prepack is the contract).
      mlir::hmx::WeightResidentOptions weightResidentOpts;
      weightResidentOpts.prepackRuntimeWeights = enableWeightResident;
      pm.addNestedPass<func::FuncOp>(
          mlir::hmx::createWeightResidentPass(weightResidentOpts));
      // The HMX tile level runs while the crouton buffers are still the ones
      // bufferization allocated: the VTCM machinery below rewrites space-1
      // buffers for the HVX scratch path, and these belong to the region the
      // runtime acquires instead (docs/hmx/hmx-system-design.md 2.B / 10.12).
      //
      // No interaction with `FormSCFThreadsPass` (above, before bufferization):
      // it selects the linalg ops that exist at that point, and both the HMX
      // tile loop (created here) and its bridge pack loop (created by
      // `lower-pack`, above this pass but after FormSCFThreads) come into being
      // only after FormSCFThreads has already run. The HMX loops are therefore
      // never candidates for scf-threading, and the partition pass needs no
      // threading special case.
      // The HMX tile loop's activation-staging ring depth: 0 = auto, 1 = force
      // the serial ring, 2 = request the double ring (see hmx-partition).
      // The K batch per mma travels next to it: 0 = the hardware maximum (32),
      // so the default is the same code it was before the option existed, and
      // the knob exists to make "batch fewer croutons per instruction"
      // measurable inside one build instead of by editing a constant and
      // rebuilding (docs/hmx/ncroutons-k-fusion-2026-10-01.md §3.5.3).

      mlir::hmx::HmxPartitionOptions hmxPartitionOpts;
      hmxPartitionOpts.pipelineDepth = enableHmxPipelineDepth;
      hmxPartitionOpts.croutonsPerMma = hmxCroutonsPerMma;
      // The read-out channel of `auto` staging: the staged loop is what the
      // read-out split's m-tile loop attaches to, and with the split enabled
      // a shape with at least two read-out batches (m-tiles >= 2 x batch) has
      // overlap to hide the handoff behind even when K is shallow (the
      // transfer channel alone would decline it). Wired only when the split
      // runs, so disabling the split restores the transfer-only floor.
      hmxPartitionOpts.stagedReadoutMTiles =
          enableHmxVectorReadout ? 2 * hmxReadoutBatch : 0;
      // NOTE: the thread-role split no longer forces `auto` staging to the
      // serial ring. It did while the split's matcher only understood the
      // serial source loop (the pipelined depth-2 form peeled an epilogue
      // the matcher declined); the matcher now outlines the peeled epilogue
      // into the same engine section as the steady loop, so `auto` keeps its
      // own decision and the dual-role A/B runs on the same depth the
      // single-thread pipeline would choose.
      pm.addNestedPass<func::FuncOp>(
          mlir::hmx::createHmxPartitionPass(hmxPartitionOpts));
        // Hand the accumulator read-out to a second thread. It runs BEFORE
        // thread-role-partition, and that order is load-bearing in BOTH
        // directions:
        //
        // * It runs after hmx-partition, because hmx-partition is what
        //   creates the m-tile loop and hoists the read-out into it (see
        //   HmxPartitionPass.cpp:990-1083) -- before that, the read-out is a
        //   separate loop after the matmul and the batching has nothing to
        //   attach to. It must run before the residency and
        //   convert-to-hexagonmem rewrites below, which are about VTCM
        //   placement and have nothing to say about which thread runs the
        //   vector work.
        //
        // * It runs before THREAD-ROLE-PARTITION because the two splits
        //   compose in that order and not the other way: the read-out split
        //   matches while its completion proof (an in-body `acc_read`) is
        //   still in the tile loop, replaces the read-out with publishes at
        //   that position, and THEN the role split moves the `acc_read` onto
        //   the bound thread and re-establishes the proof per group with a
        //   `wait_retired` barrier ahead of each publish (the safety
        //   argument is ThreadRolePartition.cpp's emission comment). With
        //   the order reversed, the role split would first orphan the
        //   read-outs past its exit drain and the read-out split would find
        //   nothing to attach to -- the exact "readout gives way" regression
        //   the R2 mechanism exists to remove.
        //
        // On by default since the gap-table measurement (2026-10-04): OFF->G4
        // is -24% on S1 (iters=1000) and every non-matching structure declines
        // with a remark, so nothing else is silently rewritten.
        //
        // The pass itself cannot emit the `configure()` call that hands the
        // executor its function pointer, because a function's address is not
        // expressible before convert-func-to-llvm (`llvm.mlir.addressof` rejects
        // a `func.func` symbol). HmxToLLVMPass emits it instead, after the
        // conversion -- see wireVectorReadout there, and the handoff record the
        // readout pass publishes for it in HmxReadoutHandoff.h.
        if (enableHmxVectorReadout) {
          mlir::hmx::HmxVectorReadoutOptions readoutOpts;
          readoutOpts.batch = hmxReadoutBatch;
          readoutOpts.deferDrain = hmxReadoutDeferredDrain;
          pm.addNestedPass<func::FuncOp>(
              mlir::hmx::createHmxVectorReadoutPass(readoutOpts));
        }

        // Thread-role classification runs AFTER hmx-partition and the
        // read-out split, and both orders are load-bearing. Before
        // hmx-partition every HMX kernel looks the same: one `hmx.matmul`
        // sitting between two independent pack loops, so "is there a pack to
        // stream" has the same answer for all of them and the verdict carries
        // no information. hmx-partition is what creates the tile loop and, at
        // pipeline-depth 2, moves the pack inside it -- which is exactly the
        // difference the measured depth-1-vs-depth-2 A/B turns on (14-35%).
        // Running here is what lets the pass see that difference instead of
        // predicting it. After the read-out split, for the reason above.
        if (enableThreadRolePartition)
          pm.addNestedPass<func::FuncOp>(
              mlir::hmx::createThreadRolePartition());

      // Per-launch VTCM workspace becomes a resident buffer. On by default:
      // the runtime keys workspace residency by the calling thread's ordinal
      // (VtcmPool::Resident's slot), so concurrent instances of a grid>1
      // launch get separate buffers, and a serial launch shares one buffer
      // soundly (each instance overwrites the whole workspace before reading
      // it). Runs here so the partition pass's conversion state / ring /
      // scratch exist, and before convert-to-hexagonmem carries the tag to
      // the lowering.
      if (enableWorkspaceResident)
        pm.addNestedPass<func::FuncOp>(
            mlir::hmx::createHmxWorkspaceResidentPass());
    }

    if (enableConvertToHexagonmem)
      pm.addNestedPass<func::FuncOp>(createConvertToHexagonmemPass());

    // External VTCM scratch mode: inject scratch arg and replace VTCM allocs
    // with views into the per-instance scratch buffer (hexagon.scratch).
    if (scratch > 0) {
      InsertScratchArgOptions scratchOpts;
      scratchOpts.scratch = scratch;
      pm.addNestedPass<func::FuncOp>(createInsertScratchArgPass(scratchOpts));
      pm.addNestedPass<func::FuncOp>(createMemoryOffsetsPass());
    }

    // The one place a stage exists for the HMX diagnostics. The production
    // pipeline passes nothing here, so nothing diagnostic is scheduled, no
    // diagnostic pass is linked into the backend library, and the diagnostics
    // still run at the stage they need: after every placement rewrite, before
    // the memref lowerings that erase the VTCM allocations the census inspects
    // and the facts the record folds in.
    // Two sources, not one: the caller may hand a stage in (the diagnostic
    // entry point does) or the pass may have been given one before it runs
    // (same entry point, installed through runLinalgToLLVMPipeline). A
    // function_ref built from an *empty* std::function is non-null, so the
    // emptiness has to be tested on the std::function itself.
    if (atDiagnosticStage)
      atDiagnosticStage(pm);
    else if (diagnosticStage)
      diagnosticStage(pm);

    // Finalize the record-only v3 document from the compile-time facts
    // attribution published and the census sidecars just produced.  Inert
    // without the internal `hmx.diagnostic_v3_record` marker, so this changes
    // nothing on the v2 path: the pass returns immediately when the marker is
    // absent.  It sits after the stage above on purpose -- the census runs there
    // in a diagnostic run, and the record folds in the facts it proves.
    pm.addPass(mlir::hmx::createHmxRecordV3Pass());

    // Lower linalg ops with library_call attribute set to custom fns.
    pm.addPass(createHexagonReplaceWithLibraryCallsPass());
    if (enableHexagonmemCopyToDMA)
      pm.addNestedPass<func::FuncOp>(createHexmemCpyToDMAPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    // Row reductions that would be scalarized below are rewritten into a
    // vector fold plus an hvx.vror butterfly while they are still linalg ops
    // on memrefs (the last point where the 2-D row structure is visible).
    // Off by default; the knob is the device A/B switch.
    if (enableVectorRowReduce)
      pm.addNestedPass<func::FuncOp>(createVectorRowReducePass());
    // Vector float maxnumf has no HVX lowering (only FMAXIMUMNUM is Legal;
    // FMAXNUM expands per lane into a ~130-instruction scalar chain or
    // fmaxf libcalls). Rewrite it into vmax + a NaN fixup that restores
    // strict maxnum semantics. Last of the maxnumf producers: the
    // vectorizer's elementwise updates and, with the knob above, the
    // row-reduce butterfly's fold steps. Runs after AddFastMath, so the
    // fixup ops it emits are never stamped with the nnan assertion.
    if (enableMaxnumLegalize)
      pm.addNestedPass<func::FuncOp>(createHvxMaxnumLegalizePass());
    pm.addPass(createConvertLinalgToLoopsPass());

    pm.addNestedPass<func::FuncOp>(createFormAsyncThreadsPass());
    pm.addPass(createAsyncFuncToAsyncRuntimePass());
    pm.addPass(createAsyncToAsyncRuntimePass());

    pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());

    if (enableLWP)
      pm.addNestedPass<func::FuncOp>(
          createHexagonLWPPass(setLWP(mlir::hexagon::HexagonLWPPassOptions{})));

    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addNestedPass<func::FuncOp>(createExpandMathOpsPass());

    if (expandBoolVec)
      pm.addNestedPass<func::FuncOp>(createExpandBoolVecPass());

    pm.addPass(createFastInversePass());
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createConvertIndexToLLVMPass(
        setIndexBitwidth(ConvertIndexToLLVMPassOptions{})));

    pm.addPass(createConvertAsyncToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass(ConvertFuncToLLVMPassOptions{}));

    pm.addPass(hexagon::createDMAToLLVMPass());
    pm.addPass(hexagonmem::createHexagonMemToLLVMPass(
        setDeviceType(hexagonmem::HexagonMemToLLVMOptions{})));
    pm.addPass(mlir::hmx::createHmxToLLVMPass());
    pm.addPass(mlir::hvx::createHvxToLLVMPass());

    if (enableCollapseAddressSpace) {
      pm.addPass(createCollapseAddressSpacePass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    }

    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (enableHexagonRoutines)
      pm.addPass(createHexagonLLVMEnableHexagonRoutinesPass());
}

void hexagon::addLinalgToLLVMDependentDialects(DialectRegistry &registry) {
  // The production pass owns the list; construct one just to read it.
  LinalgToLLVMPass producer(LinalgToLLVMOptions{});
  producer.getDependentDialects(registry);
}

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLinalgToLLVMPass(const LinalgToLLVMOptions &options) {
  return std::make_unique<LinalgToLLVMPass>(options);
}

LogicalResult hexagon::runLinalgToLLVMPipeline(
    MLIRContext &context, ModuleOp module, StringRef options,
    llvm::function_ref<void(PassManager &)> atDiagnosticStage) {
  // The pipeline parser resolves pass names through the global registry, so the
  // registration must have happened. `hexagon/Conversion/LinalgToLLVM/Passes.h`
  // (already included above) declares it; re-registering is idempotent.
  registerLinalgToLLVMPass();

  // `linalg-to-llvm` parses its own options, so build the pipeline from the
  // forwarded text and take the pass the parser produced. Parsing them here
  // would create a second option surface that can drift from the .td.
  //
  // The pipeline parser breaks a pass's options on whitespace, so a forwarded
  // value cannot contain a space; the comma is the separator and is translated
  // back into what that parser expects. No current option value contains a
  // comma, and if one ever does this is where to fix it -- not by handing the
  // diagnostic entry a second option surface.
  std::string forwarded(options.str());
  std::replace(forwarded.begin(), forwarded.end(), ',', ' ');
  std::string pipeline = "linalg-to-llvm";
  if (!forwarded.empty())
    pipeline += "{" + forwarded + "}";

  PassManager pm(&context, module.getOperationName());
  if (failed(parsePassPipeline(pipeline, pm)))
    return failure();

  // Instal the stage on the pass the parser built, not on a copy of it: the
  // generated copy constructor does not carry the option values, so a copy
  // would silently fall back to the production defaults.
  for (Pass &pass : pm.getPasses())
    if (auto *linalgToLLVM = dyn_cast_if_present<LinalgToLLVMPass>(&pass))
      linalgToLLVM->diagnosticStage = atDiagnosticStage;

  // `PassManager::run` loads the dependent dialects of the passes it runs, so
  // the production sequence and the diagnostics alike get the dialects they
  // declare.
  if (failed(pm.run(module)))
    return failure();
  return success();
}
