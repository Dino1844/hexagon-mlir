//===- Passes.td - Linalg to LLVM conversion passes  -------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_LINALG_TO_LLVM_CONVERSION_PASSES
#define HEXAGON_LINALG_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// -- The main linalg-to-llvm pass-pipeline --
def LinalgToLLVM : Pass<"linalg-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Linalg to LLVM";
  let constructor = "hexagon::createLinalgToLLVMPass()";
  let options = [
    Option<"puntBuffer", "punt-buffer", "bool",
           /*default=*/"true",
           "Punt input kernel buffer pointer directly to user instead"
           " of making local copy first.">,
    Option<"returnValueOptimization", "rvo", "bool",
           /*default=*/"true",
           "Rewrites the IR so that it is easier for one-shot bufferizer and"
           " eliminate empty-tensor to fold away unnecessary copying from"
           " local buffer to output buffer">,
    Option<"enableMatmulToConv", "enableMatmulToConv", "bool",
            /*default=*/"false",
            "Lower matmul ops to conv ops">,
    Option<"fusion", "fusion", "bool",
           /*default=*/"true",
           "Attempt to fuse linalg generic ops">,
    Option<"fusionAllowRecompute", "far", "bool",
           /*default=*/"false",
           "Allow recomputing values when fusing linalg generic ops">,
    Option<"fusionDoMultiUse", "fdmu", "bool",
           /*default=*/"true",
           "Do multi-use fusion">,
    Option<"useInterchangeVector", "use-interchange-vector", "bool",
           /*default=*/"false",
           "Use interchange vector in the HexagonTiling pass"
           " if default inner loop is not conducive for vectorization.">,
    Option<"enableDoubleBuffering", "enable-double-buffering", "bool",
          /*default=*/"false",
          "Enable double buffering optimization">,
    Option<"enableSCFThreading", "enable-scf-threading", "bool",
          /*default=*/"false",
          "Enable scf-level multi-threading">,
    Option<"enableMultiThreading", "enable-multi-threading", "bool",
          /*default=*/"false",
          "Enable asnyc multi-threading">,
    Option<"addFastMath", "add-fastmath", "bool",
           /*default=*/"true",
           "Add Fast Math Flag to HexagonAddFastMath Pass.">,
    Option<"splitTilingRange", "split-tiling-range", "bool",
          /*default=*/"true",
          "Configure HexagonTiling pass to do op splitting, "
          "instead of padding, when dealing with loops "
          "which are not  perfectly tileable.">,
    Option<"enableSplitReduction", "enable-split-reduction", "bool",
           /*default=*/"false",
           "Configure HexagonTiling pass to enable split reduction, "
           " for reduction loops. ">,
    Option<"enableSplitReduceGeneric", "enable-split-reduce-generic", "bool",
           /*default=*/"false",
           "Enable split-reduce-generic to convert linalg.generic [reduce] {..}"
           "into a linalg.generic [reduce, parallel] {..} for vectorization, "
           "followed by a final single reduction.">,
    Option<"enableSlicing", "enable-slicing", "bool",
           /*default=*/"false",
           "Flag to enable/disable operator slicing pass">,
    Option<"slicingFactor", "slicing-factor", "int",
           /*default=*/"1",
           "Operator slicing factor.">,
    Option<"expandBoolVec", "expand-bool-vec", "bool",
           /*default=*/"true",
           "replace ui1_to_fp with select %cond %vec_1s %vec_0s">,
    Option<"enableHexagonRoutines", "enable-hexagon-routines", "bool",
          /*default=*/"true",
          "Replaces LLVM intrisics with vectorized Hexagon routines.">,
    Option<"enableVTCMTiling", "enable-vtcm-tiling", "bool",
          /*default=*/"true",
          "enable lowering linalg-on-tensors (DDR) to tiled-compute on VTCM">,
    Option<"scratch", "scratch", "int64_t",
          /*default=*/"0",
          "per-instance VTCM scratch size in bytes (0 = disabled). When >0, "
          "enables external VTCM flow: InsertScratchArg injects a "
          "memref<Nxi8,1> {hexagon.scratch} arg, VTCMTilingPass tiles within "
          "this budget, and MemoryOffsetsPass replaces allocs with views.">,
    Option<"tileSizes", "tile-sizes-override", "std::string",
           /*default=*/"",
           "override tile size.">,
    Option<"enableConvertToHexagonmem", "enable-convert-to-hexagonmem", "bool",
          /*default=*/"true",
          "enable conversion from memref to hexagonmem ops">,
    Option<"enableHexagonmemCopyToDMA", "enable-hexagonmem-copy-to-dma", "bool",
          /*default=*/"false",
          "enable conversion from hexagonmem copy to dma ops">,
    Option<"enableCollapseAddressSpace", "enable-collapse-address-space", "bool",
          /*default=*/"true",
          "replace llvm.ptr<1> with llvm.ptr at last stage of convert-to-llvm">,
    Option<"lowerConstantsInSeparateSharedObjects", "lower-constants-in-separate-shared-objects", "bool",
          /*default=*/"false",
          "Lowers the constant tensors separately in their own MLIR modules, which then get compiled independently into their own .o, and then independently into various shared objects (.so)">,
    Option<"enableBufferization", "enable-bufferization", "bool",
          /*default=*/"true",
          "enable bufferization passes; set to false for manual buffer management">,
    Option<"enableSeedLayoutConversions", "enable-seed-layout-conversions", "bool",
          /*default=*/"false",
          "wraps HMX applicable convolutions with layout conversions">,
    Option<"enableConvTiling", "enable-conv-tiling", "bool",
          /*default=*/"false",
          "Tile the Convolution Op">,
    Option<"convTileSizes", "conv-tile-sizes", "std::string",
          /*default=*/"",
          "Provide the tile sizes in the following format: '<dim1>:<tilesize1>, <dim2>:<tilesize2>,...' ">,
    Option<"enableLWP", "enable-lwp", "bool",
           /*default=*/"false",
           "Flag to enable/disable lwp instrumentation">,
    Option<"disableLWPLoop", "loop", "bool",
           /*default=*/"false",
           "Turn on or off loop level instrumentation">,
    Option<"LWPloopDepth", "depth", "int",
           /*default=*/"1",
           "Depth level instrumentation">,
    Option<"extendPackUpperFrontier", "upper", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.pack frontiers.">,
    Option<"extendPackLowerFrontier", "lower", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.unpack frontiers.">,
    Option<"extendPackParallelsOnly", "parallels-only", "bool",
           /*default=*/"true",
           "Allow un-staging linalg ops that resolve to reduction "
           "generics even when it is possibly to pack those. This impacts "
           "both upper and lower frontier">,
    Option<"device_type", "device-type", "std::string",
	   /*default=*/"\"hexagon\"",
           "Target device type for backend-specific lowering">,
    Option<"enableVectorization", "enable-vectorization", "bool",
          /*default=*/"true",
          "Enable vectorization for HVX">,
    Option<"enableConversionToFp16", "enable-conversion-to-fp16", "bool",
          /*default=*/"false",
          "Enables the pass which converts fp64/fp32 operations to fp16">,
    Option<"enableHVXInlining", "enable-hvx-inlining", "bool",
          /*default=*/"false",
          "Enable aggressive inlining after linking runtime modules">,
    Option<"enableSCFLoopUnroll", "enable-scf-loop-unroll", "bool",
          /*default=*/"false",
          "Enable SCF loop unrolling optimization">,
    Option<"enableL2Prefetch", "enable-l2-prefetch", "bool",
          /*default=*/"true",
          "Insert l2fetch (L2 prefetch engine) into flat streaming vector "
          "loops: the single-core L2-miss path moves ~11 GB/s while the "
          "prefetch engine side-steps it (3.9x measured on the gap-table "
          "ceiling working set). Default ON since 2026-10-05: the device A/B "
          "passed and this .td was flipped together with "
          "backend/hexagon_options.py (see "
          "docs/results/l2-prefetch-2026-10-04.md). The OFF arm must now be "
          "requested explicitly with enable-l2-prefetch=false">,
    Option<"forceHVXCroutonization", "force-hvx-croutonization", "bool",
          /*default=*/"false",
          "Force HVX croutonization pass">,
    Option<"enableWeightResident", "enable-weight-resident", "bool",
          /*default=*/"true",
          "Host pre-packs runtime weights; the kernel drops their per-launch "
          "pack bridge and reads them from a resident VTCM buffer. On by "
          "default: the generated launcher always pre-packs, and only a "
          "caller that honours hmx.weight_prepack can run this kernel.">,
  Option<"enableThreadRolePartition", "enable-thread-role-partition", "bool",
           /*default=*/"false",
           "Classify each region's thread role and record the decision "
           "(thread-role-partition). Off by default; adds no runtime behaviour.">,
    Option<"enableHmxPipelineDepth", "enable-hmx-pipeline-depth", "int64_t",
          /*default=*/"0",
          "HMX tile-level staging ring depth (hmx-partition): 0 = auto, "
          "1 = force the serial ring, 2 = request double buffering, 3 = skip "
          "staging (emit the unstaged serial tile loop).">,
    Option<"hmxCroutonsPerMma", "hmx-croutons-per-mma", "int64_t",
          /*default=*/"0",
          "K croutons walked by one hmx.mma (hmx-partition's croutons-per-mma): "
          "0 = the hardware maximum (32), 1 = one mma per crouton, 32 = the "
          "maximum. 0 and 32 are the same request. Anything outside {0} u "
          "[1, 32] is rejected by hmx-partition rather than clamped.">,
    Option<"enableHmxVectorReadout", "enable-hmx-vector-readout", "bool",
           /*default=*/"true",
           "Hand a lowered HMX matmul's accumulator read-out to a second "
           "thread, in batches of hmx-readout-batch rows (hmx-vector-readout). "
           "Default on since 2026-10-04 (the Python option flipped first and "
           "this .td default follows it, so a bare manual pipeline run and a "
           "production compile agree; they briefly did not, and a manual run "
           "then showed no read-out with no diagnostic). What the pass cannot "
           "take it declines with a remark.">,
    Option<"hmxReadoutBatch", "hmx-readout-batch", "int64_t",
          /*default=*/"4",
          "How many AR rows one hmx vector-executor handoff names (G). Must be "
          ">= 1; 0 or negative is rejected by hmx-vector-readout rather than "
          "clamped. Measured on this device: G=1 loses to the handoff cost, "
          "G=4 is 1.39x, G=8 is 1.51x.">,
    Option<"hmxReadoutDeferredDrain", "hmx-readout-deferred-drain", "bool",
          /*default=*/"false",
          "Drop the kernel-exit drain of hmx-vector-readout and let the next "
          "call's configure() barrier (plus one wrapper-side drain after the "
          "last call, outside the timed region) wait for the tail batch, so "
          "the tail read-out overlaps the function epilogue instead of "
          "serialising behind it. Declined with a remark for a function with "
          "more than one readout loop. Off by default: a batch may still be "
          "in flight after the kernel returns, which is only sound for "
          "sequential single-instance (grid=1) launches.">,
    Option<"enableWorkspaceResident", "enable-workspace-resident", "bool",
          /*default=*/"true",
          "Make every per-launch VTCM workspace of an HMX kernel a "
          "process-resident buffer (hmx-workspace-resident). Default on "
          "since 2026-10-04: residency slots key on the caller's flat "
          "program id (the 4th argument of "
          "hexagon_runtime_workspace_resident_v2_dsp), so a grid>1 launch "
          "no longer shares one resident buffer across instances -- the "
          "hazard that once kept this off.">,
    Option<"enableVectorRowReduce", "enable-vector-row-reduce", "bool",
          /*default=*/"false",
          "Lower 2-D last-dim linalg.reduce (f16/f32 max/add) to per-vector "
          "elementwise folding plus an hvx.vror butterfly, instead of letting "
          "ConvertLinalgToLoops scalarize it into a per-lane serial chain that "
          "the Hexagon backend cannot revectorize (only VECREDUCE_ADD gets an "
          "HVX DAG combine). Off by default.">,
    Option<"enableMaxnumLegalize", "enable-maxnum-legalize", "bool",
          /*default=*/"false",
          "Rewrite vector f16/f32 arith.maxnumf into arith.maximumf plus an "
          "explicit NaN correction before the LLVM conversion: the Hexagon "
          "HVX backend has no lowering for FMAXNUM (only FMAXIMUMNUM is "
          "Legal), so a vector maxnumf is expanded per lane into a ~130-"
          "instruction stack/scalar/build-vector chain (or fmaxf libcalls). "
          "The rewrite preserves strict maxnum semantics. Off by default: the "
          "R1-ON kernel aborted at launch on the device hexmem path with a "
          ">= [512,128] maxnum tile; the root cause is three llvm_triton "
          "Hexagon backend RA bugs (docs/history/hmx/llvm-hexagon-ra-bugs.md; "
          "the PS_aligna one is cherry-picked under tools/hexmlir, the rest "
          "await upstream) -- keep OFF until then.">,
    Option<"enableRowReduceGroupStore", "enable-row-reduce-group-store", "bool",
           /*default=*/"false",
           "Rewrite a vectorized row reduction that lands its per-row result in a "
           "rank-0 tensor slice into a per-group form: the row loop steps by one "
           "HVX vector instead of one row, a group's running values stay in a "
           "vector<lanes x T> loop-carried value, each row's horizontal butterfly "
           "stays in the vector domain (no vector.extract, no scalar store), and an "
           "arith.cmpi + arith.select pair places it in the group's lane. The "
           "Hexagon backend has no vector-lane-to-GPR instruction, so any reduction "
           "result that becomes a scalar costs one 128-byte stack write plus a 4-byte "
           "stack read back, per row per 128-byte chunk. Runs after "
           "HexagonVectorLowering (which is what expands vector.multi_reduction into "
           "the per-chunk vector.reduction this pass rewrites) and before bufferize. "
           "Off by default: the knob is the device A/B switch, not a correctness "
           "guard.">,
   ];
  // The pass's `emitMaxnumFixup` / `maxnumSiteLimit` / `maxnumSiteSkip` options
  // were removed 2026-09-30. Fixup was a semantics switch (false emits bare
  // `maximumf`, dropping strict maxnum NaN handling), and Sel/Skip bisected a
  // crash since attributed to the LLVM Hexagon AP under-alignment. No lit test
  // ever passed any of them; leaving them would be a legacy surface with no
  // user. See backend/hexagon_options.py for the same note.
}

// --- Supporting passes ---

def VectorRowReduce : Pass<"vector-row-reduce", "func::FuncOp"> {
  let summary = "Lower 2-D last-dim reductions to vector fold + vror butterfly";
  let description = [{
    Matches `linalg.reduce` along dim 1 of a static-shape 2-D memref whose row
    byte width is a multiple of 128 (one or more whole HVX vectors per row),
    whose body is a single maxnumf/addf fold of the two block args, and whose
    element type is f16 or f32. Replaces it with explicit vector IR: each row's
    128-byte chunks are folded elementwise, then a vror butterfly
    (log2(128/elemBytes) rotate+fold steps, llama.cpp hvx-reduce.h style) leaves
    every lane holding the row's reduction; the original outs init is folded in
    and lane 0 is stored back, so the scalar semantics of the reduce are
    preserved exactly (barring float reassociation, which fastmath already
    grants).

    Everything else -- dynamic shapes, non-contiguous rows, other element
    types, other reduction kinds -- is left untouched for the scalar path.
  }];

  let constructor = "hexagon::createVectorRowReducePass()";
}

def RowReduceGroupStore : Pass<"row-reduce-group-store", "func::FuncOp"> {
  let summary = "Keep a vectorized row reduction's per-row result in the vector domain";
  let description = [{
    Matches the shape `HexagonVectorLowering` leaves behind for a 2-D last-dim
    row reduction of a static f16/f32 tensor: an `scf.for` over rows carrying a
    `tensor<rows x T>`, whose body is a single `scf.for` over columns stepping by
    one whole 128-byte HVX vector, whose body reads each chunk with a
    `vector.transfer_read`, folds it with `vector.reduction <maxnumf>` against a
    `tensor.extract` of the row's current value, and writes the scalar result
    into a rank-0 slice of that same carrier.

    Rewrites it so the per-row result never becomes a scalar. The row loop steps
    by `lanes` instead of 1; the group's running values live in a
    `vector<lanes x T>` loop-carried value (which one-shot bufferization leaves
    alone, so it stays in registers); each row's horizontal butterfly
    (`hvx.vror` + `maxnumf`, `log2(128/elemBytes)` steps, llama.cpp
    hvx-reduce.h style) produces an all-lanes-equal vector that is *not*
    extracted; an `arith.cmpi eq` against a constant lane-index vector plus
    `arith.select` places it in the group's lane, folding the group's previous
    value in on the way; and the group is written back with one
    `vector.transfer_write` per `lanes` rows.

    This is the only reason the scalar form is worth changing: the Hexagon HVX
    backend has no "vector lane 0 to GPR" instruction, so one `vector.reduction`
    result that becomes an `f32` costs a 128-byte vector stack write followed by
    a 4-byte stack read back -- per row, per 128-byte chunk, in a loop-carried
    dependency chain.

    Everything else -- dynamic shapes, a column loop that does not step by a
    whole vector, a reduction kind other than `maxnumf`, a result that does not
    land in a rank-0 slice, a carrier with another reader, a row tile or tile
    base that is not a multiple of the lane count -- is left untouched, and each
    refusal says which gate stopped it. The gate is structural: no shape, kernel
    name or operator name appears in it.
  }];

  let constructor = "hexagon::createRowReduceGroupStorePass()";
}

def FormAsyncThreads : Pass<"form-async-threads", "func::FuncOp"> {
  let summary = "Rewrite scf::forall as async threads";
  let constructor = "hexagon::createFormAsyncThreadsPass()";
}
def FormSCFThreads : Pass<"form-scf-threads", "func::FuncOp"> {
  let summary = "Multi-thread SCF loop";
  let description = [{
    This pass turns scf::for into parallel loops for multi-threading.
    It runs early on in the pipeline and looks for scf:for that is
    over-arching over most of the kernel. The pass will do some
    checks but the onus is currently mostly on the user to guarantee
    parallel conversion correctness.
  }];
  let constructor = "hexagon::createFormSCFThreadsPass()";
}

def FormVirtualThreads : Pass<"form-virtual-threads", "func::FuncOp"> {
  let summary = "Rewrite generics as scf::forall for thread formation"
                 "after further lowering.";
  let constructor = "hexagon::createFormVirtualThreadsPass()";
  let options = [
       Option<"preferredNumThreads", "preferred-num-threads", "int64_t",
              /*default=*/"4",
              "Preferred number of virtual threads to create.">,
       Option<"blockSize", "block-size", "int64_t",
              /*default=*/"0",
              "Block size for testing (user-override)">,
  ];
}

def HexmemCpyToDMA :
    InterfacePass<"hexmem-cpy-to-dma", "mlir::FunctionOpInterface"> {
  let summary = "Pass to rewrite hexagonmem.copy and memref.copy ops to memref.dma ops";

  let constructor =
       "hexagon::createHexmemCpyToDMAPass()";
}

def HexagonExtendPack:
    InterfacePass<"extend-pack-frontier", "mlir::FunctionOpInterface"> {
  let summary = "Extend pack and unpack frontier around linalg ops.";

  let options = [
    Option<"upperFrontier", "upper", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.pack frontiers.">,
    Option<"lowerFrontier", "lower", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.unpack frontiers.">,
    Option<"parallelsOnly", "parallels-only", "bool",
           /*default=*/"true",
           "Allow un-staging linalg ops that resolve to reduction "
           "generics even when it is possibly to pack those. This impacts "
           "both upper and lower frontier">
  ];

  let constructor =
       "hexagon::createHexagonExtendPackPass()";
}

def HexagonFusion :
    InterfacePass<"hexagon-fusion", "mlir::FunctionOpInterface"> {
  let summary = "Pass to fuse linalg generic ops";
  let options = [
    Option<"fusionAllowRecompute", "far", "bool",
           /*default=*/"false",
           "Allow recomputing values when fusing linalg generic ops">,
    Option<"fusionDoMultiUse", "fdmu", "bool",
           /*default=*/"true",
           "Do multi-use fusion">
   ];
  let constructor =
       "hexagon::createHexagonFusionPass()";
}

def SplitReduceGeneric : Pass<"split-reduce-generic", "func::FuncOp"> {
  let summary = "Split generic reduction operations for optimization";
  let description = [{
    This pass splits a linalg.generic [reduce] {..}into a
    linalg.generic [reduce, parallel] {..} that can be more effectively
    vectorized. That is followed by a final horizontal reduction.
  }];
  let constructor = "hexagon::createSplitReduceGenericPass()";
}

def EraseVectorToTensorWriteback : Pass<"erase-vector-to-tensor-writeback", "func::FuncOp"> {
  let summary = "Erase unnecessary vector-to-tensor writeback in loops";
  let description = [{
    This pass optimizes IR where vectors are written back to tensors
    and yielded in scf.for loops, only to be immediately read back
    as vectors after the loop. When the tensor has no other uses,
    the loop can directly yield the vector instead.
  }];
  let constructor = "hexagon::createEraseVectorToTensorWritebackPass()";
}

def HexagonSlicing :
    Pass<"hexagon-slicing", "mlir::ModuleOp"> {
  let summary = "Pass to slice ops into K parts, "
                " where K is identified by slicingFactor";
  let options = [
    Option<"slicingFactor", "slicing-factor", "int",
           /*default=*/"1",
           "Operator slicing factor.">
  ];
  let constructor = "hexagon::createHexagonSlicingPass()";
}

def HexagonTiling :
    Pass<"hexagon-tiling", "mlir::ModuleOp"> {
  let summary = "Pass to tile tensor linalg ops. Presently, "
                " just for vectorization.";

  let options = [
    Option<"useInterchangeVector", "use-interchange-vector", "bool",
           /*default=*/"false",
           "Use interchange vector if default inner loop "
           "is not conducive for vectorization.">,
    Option<"splitTilingRange", "split-tiling-range", "bool",
           /*default=*/"true",
           "Split the not perfectly tileable loop into perfectly tileable "
           " and the remainder parts. If set to false, padding is added "
           "to make it a perfectly tileable shape">,
    Option<"enableSplitReduction", "enable-split-reduction", "bool",
           /*default=*/"false",
           "Splits reduction loop to expose more opportunities for "
           "vectorization, ">
   ];

  let constructor = "hexagon::createHexagonTilingPass()";
}

def HexagonVectorization :
    Pass<"hexagon-vectorization", "mlir::ModuleOp"> {
  let summary = "Pass to vectorize tensor linalg ops.";
  let options = [
    Option<"skipVectorRowReduce", "skip-vector-row-reduce", "bool",
           /*default=*/"false",
           "Leave row reduces matching the vector-row-reduce pattern "
           "unvectorized so they reach that pass instead of becoming a "
           "vector.multi_reduction whose LLVM lowering is a valign tree. "
           "Driven by the pipeline's enable-vector-row-reduce knob.">
  ];
  let constructor = "hexagon::createHexagonVectorizationPass()";
}

def LowerPack : Pass<"lower-pack", "func::FuncOp"> {
  let summary = "lower linalg.pack to simpler ops";
  let constructor = "hexagon::createLowerPackPass()";
}

def LowerConstantsSeparately : Pass<"lower-constants-separately", "mlir::ModuleOp"> {
  let summary = "Lowers constants separately from the code, possibly in multiple modules if above a certain threshold";
  let constructor = "hexagon::createLowerConstantsSeparatelyPass()";
}

def RewriteUBPoisonToZero :
    Pass<"rewrite-ub-poison-to-zero", "mlir::ModuleOp"> {
  let summary = "Pass to rewrite ub poison values to zero.";
  let constructor = "hexagon::createRewriteUBPoisonToZeroPass()";
}

def VTCMTiling :
     InterfacePass<"vtcm-tiling", "mlir::FunctionOpInterface"> {
  let summary = "Lower linalg-on-tensors (DDR) to tiled-compute on VTCM";
  let options = [
    Option<"tileSizes", "tile-size-override", "std::string",
           /*default=*/"",
           "override tile size. Only intended for debugging.">,
    Option<"vtcmBudget", "vtcm-budget", "int64_t",
           /*default=*/"0",
           "per-instance VTCM budget in bytes for tile size computation. "
           "0 = use default (vtcmSizeInBytes = 2 MB).">
  ];
  let constructor = "hexagon::createVTCMTilingPass()";
}

def ConversionToFp16 :
    InterfacePass<"conversion-to-fp16", "mlir::FunctionOpInterface"> {
  let summary =
        "Pass to convert intermediate fp32 and fp64 operations to fp16 operations";
  let constructor = "hexagon::createConversionToFp16Pass()";
}

def OptimizeExtfTruncfOp :
    InterfacePass<"optimize-extf-truncf-op", "mlir::FunctionOpInterface"> {
  let summary =
        "Pass to optimize IR and remove redundant extf and truncf op";
  let constructor = "hexagon::createOptimizeExtfTruncfOpPass()";
}

#endif
