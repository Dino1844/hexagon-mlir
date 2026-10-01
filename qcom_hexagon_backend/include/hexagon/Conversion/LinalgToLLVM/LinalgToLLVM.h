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
