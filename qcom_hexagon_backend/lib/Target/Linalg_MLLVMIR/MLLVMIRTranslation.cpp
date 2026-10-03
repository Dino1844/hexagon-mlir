//===-- MLLVMIRTranslation.cpp - Linalg to LLVM IR Translation ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg to LLVM IR translation registration for
// Hexagon target.
//===----------------------------------------------------------------------===//

#include "hexagon/Target/Linalg_MLLVMIR/MLLVMIRTranslation.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/Passes.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "mlir/InitAllPasses.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include "llvm/IR/Constants.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include <dlfcn.h>
#include <filesystem>
#include <iterator>

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hexagon/Conversion/LinalgToLLVM/LowerConstantsSeparately.h"

void setLinalgToLLVMOptions(
    mlir::hexagon::LinalgToLLVMOptions &options,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {

  // Note: seems very counter-intuitive due to the fact that compare() returns 0
  // when the strings are actually equal, which is why we negate it to convert
  // it to a boolean.
  const std::string TRUE("True");
  options.fusion = !arch_kwargs.at("fusion").compare(TRUE);
  options.fusionAllowRecompute =
      !arch_kwargs.at("fusionAllowRecompute").compare(TRUE);
  options.fusionDoMultiUse = !arch_kwargs.at("fusionDoMultiUse").compare(TRUE);
  options.enableDoubleBuffering =
      !arch_kwargs.at("enableDoubleBuffering").compare(TRUE);
  options.enableSCFThreading =
      !arch_kwargs.at("enableSCFThreading").compare(TRUE);
  options.enableMultiThreading =
      !arch_kwargs.at("enableMultiThreading").compare(TRUE);
  options.enableVTCMTiling = !arch_kwargs.at("enableVTCMTiling").compare(TRUE);
  options.scratch = std::stoll(arch_kwargs.at("scratch"));
  options.enableConvertToHexagonmem =
      !arch_kwargs.at("enableConvertToHexagonmem").compare(TRUE);
  options.enableHexagonmemCopyToDMA =
      !arch_kwargs.at("enableHexagonmemCopyToDMA").compare(TRUE);
  options.enableHexKL = !arch_kwargs.at("enableHexKL").compare(TRUE);
  // hexKLMode: the Python field is GONE as of 2026-09-30, so there is nothing
  // to read here any more. Left `std::string` unset so the declared default
  // ("micro", Passes.td) applies -- deliberately NOT an `at()` read, because
  // `at()` throws on a missing key and 16 tests build partial option maps.
  //
  // Why deleting the field is behaviour-preserving, verified rather than argued:
  //  * every `hexKLMode == "macro"` branch sits behind `enableHexKL`, and
  //    LinalgToLLVMPass rejects that combination outright --
  //    "enableHexKL is incompatible with the HMX manifest contract" -- so those
  //    branches were unreachable before this change and are still unreachable;
  //  * the only caller that ever set the field is
  //    test/python/torch-mlir/test_hexkl_macro_matmul.py, and it already fails
  //    on exactly that error, and is not part of the host gate;
  //  * the direct-drive lit path (test/Conversion/LinalgToLLVM/
  //    matmul_to_hexkl.mlir) passes `(matmul-to-hexkl)` its own options and
  //    never goes through this function, so the pass option stays in Passes.td.
  // See docs/codegen/knob-fork-classification-2026-09-30.md §1.3.1-5.
  options.enableCollapseAddressSpace =
      !arch_kwargs.at("enableCollapseAddressSpace").compare(TRUE);
  options.tileSizes = arch_kwargs.at("tileSizes");
  options.lowerConstantsInSeparateSharedObjects =
      !arch_kwargs.at("lowerConstantsInSeparateSharedObjects").compare(TRUE);
  options.enableBufferization =
      !arch_kwargs.at("enableBufferization").compare(TRUE);
  // `enableSeedLayoutConversions` is no longer read from arch_kwargs (its
  // Python field was removed 2026-09-30), so the pipeline now takes the pass
  // default -- the same way the other pass options with no Python field do.
  // The upstream pass option itself is untouched and still reachable directly
  // via -linalg-to-llvm="enable-seed-layout-conversions=true".
  //
  // Note the coupling this line used to hide: `arch_kwargs.at(...)` THROWS on a
  // missing key, and the option dict these keys come from is built from
  // `HexagonOptions().__dict__` (see test/test_hmx_record_v3.py). So deleting a
  // Python field without deleting its `at()` here is a hard failure, not a
  // silent default. test/test_arch_kwargs_contract.py is what keeps the two
  // sides in step.
  options.extendPackUpperFrontier =
      !arch_kwargs.at("extendPackUpperFrontier").compare(TRUE);
  options.extendPackLowerFrontier =
      !arch_kwargs.at("extendPackLowerFrontier").compare(TRUE);
  options.forceHVXCroutonization =
      !arch_kwargs.at("forceHVXCroutonization").compare(TRUE);
  options.enableSplitReduction =
      !arch_kwargs.at("enableSplitReduction").compare(TRUE);
  options.enableConvTiling = !arch_kwargs.at("enableConvTiling").compare(TRUE);
  options.convTileSizes = arch_kwargs.at("convTileSizes");
  options.enableLWP = !arch_kwargs.at("enableLWP").compare(TRUE);
  options.disableLWPLoop = !arch_kwargs.at("disableLWPLoop").compare(TRUE);
  options.enableVectorization =
      !arch_kwargs.at("enableVectorization").compare(TRUE);
  options.enableSplitReduceGeneric =
      !arch_kwargs.at("enableSplitReduceGeneric").compare(TRUE);
  auto it = arch_kwargs.find("device_type");
  if (it != arch_kwargs.end()) {
    options.device_type = it->second;
  } else {
    options.device_type = "hexagon"; // default value
  }
  options.enableHVXInlining =
      !arch_kwargs.at("enableHVXInlining").compare(TRUE);
  options.enableSCFLoopUnroll =
      !arch_kwargs.at("enableSCFLoopUnroll").compare(TRUE);
  options.enableConversionToFp16 =
      !arch_kwargs.at("enableConversionToFp16").compare(TRUE);
  // Tolerant read: several probe scripts build a partial options map, and a new
  // gate must not turn their missing key into a throw. Absent = the declared
  // default (on), so a partial map behaves like production instead of silently
  // dropping the optimization.
  auto weightResident = arch_kwargs.find("enableWeightResident");
  options.enableWeightResident =
      weightResident == arch_kwargs.end() ||
      !weightResident->second.compare(TRUE);
  // Tolerant read: absent = the declared default (off), so a probe that builds a
  // partial options map keeps today's single-thread behaviour.
  auto threadRole = arch_kwargs.find("enableThreadRolePartition");
  options.enableThreadRolePartition =
      threadRole != arch_kwargs.end() && !threadRole->second.compare(TRUE);
  // Tolerant read for the same reason: absent = 0 (auto).
  auto hmxPipelineDepth = arch_kwargs.find("enableHmxPipelineDepth");
  options.enableHmxPipelineDepth =
      hmxPipelineDepth != arch_kwargs.end()
          ? std::stoll(hmxPipelineDepth->second)
          : 0;
  // K croutons per hmx.mma. Tolerant read for the same reason: absent = 0,
  // which hmx-partition resolves to the hardware maximum, i.e. the behaviour
  // that existed before the option was split out. The range is not checked
  // here; hmx-partition owns the {0} u [1, 32] domain so that one place decides
  // what an out-of-domain value means, whoever supplied it.
  auto croutonsPerMma = arch_kwargs.find("hmxCroutonsPerMma");
  options.hmxCroutonsPerMma = croutonsPerMma != arch_kwargs.end()
                                 ? std::stoll(croutonsPerMma->second)
                                 : 0;
  // Per-launch VTCM workspace residency. Tolerant read: absent = off, so a
  // partial options map never throws.
  auto workspaceResident = arch_kwargs.find("enableWorkspaceResident");
  options.enableWorkspaceResident =
      workspaceResident != arch_kwargs.end() &&
      !workspaceResident->second.compare(TRUE);
  // Move the HMX accumulator read-out onto a second thread. Tolerant read for
  // the same reason: absent = off, so a probe that builds a partial options map
  // keeps the single-thread behaviour.
  auto hmxVectorReadout = arch_kwargs.find("enableHmxVectorReadout");
  options.enableHmxVectorReadout =
      hmxVectorReadout != arch_kwargs.end() &&
      !hmxVectorReadout->second.compare(TRUE);
  // Rows per handoff. Tolerant read: absent = 4, the batch size this device
  // measured best (1.39x). The range is not checked here; hmx-vector-readout
  // owns the >= 1 domain so that one place decides what an out-of-domain value
  // means, whoever supplied it.
  auto readoutBatch = arch_kwargs.find("hmxReadoutBatch");
  options.hmxReadoutBatch =
      readoutBatch == arch_kwargs.end() ? 4 : std::stoll(readoutBatch->second);
  // Drop the kernel-exit drain of the read-out split. Tolerant read for the
  // same reason: absent = off, the drain stays where the rewrite put it.
  auto readoutDeferredDrain = arch_kwargs.find("hmxReadoutDeferredDrain");
  options.hmxReadoutDeferredDrain =
      readoutDeferredDrain != arch_kwargs.end() &&
      !readoutDeferredDrain->second.compare(TRUE);
  // Row reductions as vector fold + hvx.vror butterfly. Tolerant read for the
  // same reason: absent = off.
  auto vectorRowReduce = arch_kwargs.find("enableVectorRowReduce");
  options.enableVectorRowReduce =
      vectorRowReduce != arch_kwargs.end() &&
      !vectorRowReduce->second.compare(TRUE);
  // Vector maxnumf -> vmax + NaN fixup legalization. Tolerant read: absent
  // keeps the pass default (off); the key overrides it.
  auto maxnumLegalize = arch_kwargs.find("enableMaxnumLegalize");
  if (maxnumLegalize != arch_kwargs.end())
    options.enableMaxnumLegalize = !maxnumLegalize->second.compare(TRUE);
  // Row reduction kept in the vector domain. Tolerant read for the same reason:
  // absent = off.
  auto rowReduceGroupStore = arch_kwargs.find("enableRowReduceGroupStore");
  options.enableRowReduceGroupStore =
      rowReduceGroupStore != arch_kwargs.end() &&
      !rowReduceGroupStore->second.compare(TRUE);
  // The pass's three bisection knobs are gone (2026-09-30) and are no longer
  // read here, so a stale key in a saved arch_kwargs map is ignored rather than
  // silently driving the pass.
}

namespace mlir {
namespace hexagon {

mlir::ModuleOp translateLinalgToLLVMMLIR(
    mlir::ModuleOp mod,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {
  mlir::PassManager pm(mod->getContext());
  mlir::registerPassManagerCLOptions();
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "failed to apply pass manager CL options\n";
    return nullptr;
  }

  auto printingFlags = mlir::OpPrintingFlags();
  if (mlir::hexagon::isEnvTrue("MLIR_ELIDE_LARGE_CONST_PRINT")) {
    printingFlags.elideLargeElementsAttrs(1);
    printingFlags.elideLargeResourceString(1);
  } else {
    printingFlags.elideLargeElementsAttrs(16);
  }
  // Print the IR after HexagonLWPPass if enabled for debug purpose
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *pass, mlir::Operation *) {
        return mlir::hexagon::isEnvTrue("MLIR_ENABLE_DUMP") ||
               llvm::StringRef(pass->getName()).contains("HexagonLWP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  // set your enable/disable individual pass options here
  // or funnel to here.
  LinalgToLLVMOptions options;
  setLinalgToLLVMOptions(options, arch_kwargs);
  pm.addPass(createLinalgToLLVMPass(options));

  if (failed(pm.run(mod))) {
    llvm::errs() << "Linalg to Hexagon Pass execution failed";
    return nullptr;
  }
  return mod;
}

// -------------------------------------------------
// --- Translating to multiple LLVM/MLIR modules ---
// -------------------------------------------------

class CustomPassManager : public mlir::PassManager {
public:
  CustomPassManager(mlir::MLIRContext *context, LinalgToLLVMOptions options)
      : mlir::PassManager(context) {
    addPass(createLinalgToLLVMPass(options));

    // Careful: here we are giving ownership on this pass, so we can't access it
    // ourselves anymore directly
    addPass(std::make_unique<LowerConstantsSeparatelyPass>());
  }

  std::vector<ModuleOp> getProducedModules() const {
    LowerConstantsSeparatelyPass *ptr_lowerConstantsPass;
    auto passes = getPasses();

    // Trick to get the LowerConstantsSeparatelyPass since we had given
    // ownership to it. It's just some plain bureaucracy: finding the pass
    // LowerConstantsSeparatelyPass amongst all the passes that were added to
    // the pass manager
    for (auto &pass : passes) {
      if ((ptr_lowerConstantsPass =
               dynamic_cast<LowerConstantsSeparatelyPass *>(&pass))) {
        // We found the LowerConstantsSeparatelyPass pass in the pass manager
        break;
      }
    }
    if (!ptr_lowerConstantsPass) {
      std::cerr << "Error: pass_lower_constants_separately is null!"
                << std::endl;
      return {};
    }

    // Returning the list of modules the pass LowerConstantsSeparatelyPass has
    // produced
    return ptr_lowerConstantsPass->getProducedModules();
  }
};

std::vector<ModuleOp> translateLinalgToMultipleLLVMMLIRModules(
    ModuleOp mod,
    const std::unordered_map<std::string, std::string> &arch_kwargs) {

  LinalgToLLVMOptions options;
  setLinalgToLLVMOptions(options, arch_kwargs);

  CustomPassManager pm(mod->getContext(), options);
  if (failed(pm.run(mod))) {
    llvm::errs() << "Custom pass manager for producing multiple modules failed";
    return std::vector<ModuleOp>();
  }

  std::vector<ModuleOp> all_modules = pm.getProducedModules();
  // Insert the main module (mutated) at the front of the vector that will be
  // returned
  all_modules.insert(all_modules.begin(), mod);
  return all_modules;
}

} // namespace hexagon
} // namespace mlir
