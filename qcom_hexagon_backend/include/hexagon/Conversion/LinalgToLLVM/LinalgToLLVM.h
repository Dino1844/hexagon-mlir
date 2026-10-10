//===- LinalgToLLVM.h - Some of the passes involved in lowering -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H
#define HEXAGON_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexagon {
#define GEN_PASS_DECL
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<func::FuncOp>> createFormSCFThreadsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createFormAsyncThreadsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createFormVirtualThreadsPass(
    const FormVirtualThreadsOptions &options = FormVirtualThreadsOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>> createHexagonExtendPackPass(
    const HexagonExtendPackOptions &options = HexagonExtendPackOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>> createHexagonFusionPass(
    const HexagonFusionOptions &options = HexagonFusionOptions());

std::unique_ptr<OperationPass<ModuleOp>> createHexagonSlicingPass(
    const HexagonSlicingOptions &options = HexagonSlicingOptions());

std::unique_ptr<OperationPass<ModuleOp>> createHexagonTilingPass(
    const HexagonTilingOptions &options = HexagonTilingOptions());

std::unique_ptr<OperationPass<ModuleOp>> createHexagonVectorizationPass(
    const HexagonVectorizationOptions &options =
        HexagonVectorizationOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>> createHexmemCpyToDMAPass();

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToLLVMPass(
    const LinalgToLLVMOptions &options = LinalgToLLVMOptions());

/// The whole diagnostic entry point in one call: build the production pipeline
/// from `options` (written in `linalg-to-llvm`'s own key=value spelling, comma
/// separated, empty text = the production defaults) and run it, invoking
/// `atDiagnosticStage` at the one place where the production sequence used to
/// mount the two HMX diagnostic passes.
///
/// The stage is the load-bearing part. The census inspects VTCM allocations and
/// the record folds in facts the census sidecars prove, so both must run while
/// those still exist -- after every placement rewrite, before the memref
/// lowerings that erase them. Injecting them from inside the production pass is
/// what puts them there; appending them to a pipeline string would put them
/// after lowering, where they see nothing.
///
/// The options are parsed by `linalg-to-llvm`'s own parser, because the pass is
/// built by that parser rather than copied: the generated pass copy constructor
/// does not carry the option values, so a copy would silently fall back to the
/// defaults. Building the pass in place keeps the forwarding honest.
///
/// `stopAfterDiagnosticStage` ends the production sequence right after the
/// stage above (the census plus the marker-gated record document that follows
/// it), so the run stops at the placement layer instead of lowering through
/// translation. This is the layered-lit entry: one copy of the production
/// order, one extra pass to run. The production pass never sets it.
/// The dialects the production sequence needs.
///
/// A pass that runs the sequence in a nested pass manager has to declare them
/// itself: the dialects are loaded from a `PassManager`, and a context refuses
/// to load one once it is marked multi-threaded, which it is by the time a
/// nested pipeline runs. Declaring them on the host pass makes the *outer*
/// manager load them before anything runs. Taking them from here keeps the list
/// in one place.
void addLinalgToLLVMDependentDialects(DialectRegistry &registry);

LogicalResult runLinalgToLLVMPipeline(
    MLIRContext &context, ModuleOp module, StringRef options,
    llvm::function_ref<void(PassManager &)> atDiagnosticStage,
    bool stopAfterDiagnosticStage = false);

std::unique_ptr<OperationPass<ModuleOp>> createLowerConstantsSeparatelyPass();

std::unique_ptr<OperationPass<func::FuncOp>> createLowerPackPass();

std::unique_ptr<OperationPass<func::FuncOp>> createSplitReduceGenericPass();

std::unique_ptr<OperationPass<func::FuncOp>> createVectorRowReducePass();


// Row reduction whose horizontal fold stays in the vector domain: the butterfly
// result is placed into lane j of a group register and written once per group.
// Declared here (like createVectorRowReducePass) because the generated
// Passes.h.inc calls it too; without the declaration both call sites see an
// implicit `void` and fail to compile.
std::unique_ptr<OperationPass<func::FuncOp>>
createRowReduceGroupStorePass();

std::unique_ptr<OperationPass<func::FuncOp>>
createEraseVectorToTensorWritebackPass();

// Zero-operand linalg.generic shells (what elementwise fusion leaves of a
// store generic) have no lowering; inline them before ConvertLinalgToLoops.
std::unique_ptr<OperationPass<func::FuncOp>>
createInlineSideEffectOnlyGenericPass();

std::unique_ptr<OperationPass<ModuleOp>> createRewriteUBPoisonToZeroPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createVTCMTilingPass(const VTCMTilingOptions &options = VTCMTilingOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConversionToFp16Pass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createOptimizeExtfTruncfOpPass();
} // namespace hexagon
} // namespace mlir

#endif //  HEXAGON_CONVERSION_LINALGTOLLVM_LINALGTOLLVM_H
