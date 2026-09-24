//===-- triton_qcom_hexagon_backend_api.cc --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "triton_qcom_hexagon_backend_api.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Dialect/Crouton/IR/CroutonDialect.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hvx/IR/HvxDialect.h"
#include "hexagon/Dialect/HexKL/Transforms/BufferizableOpInterfaceImpl.h"
#include "hexagon/Dialect/Hmx/Transforms/BufferizableOpInterfaceImpl.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "hexagon/Target/HEX_LLVMIR/LLVMIRTranslation.h"
#include "hexagon/Target/Linalg_MLLVMIR/MLLVMIRTranslation.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // added this for the loadDialect

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h" // for llvm::isa / mlir::isa

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <regex>

#include "LinkRuntimeModules.h"

#define DEBUG_TYPE "triton-qcom-hexagon-backend-api"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Wrapper over std::runtime_error to signal LLVM IR parsing errors
void fail(const std::string &message) { throw std::runtime_error(message); }

std::unique_ptr<llvm::Module>
fixAlignedAllocTypes(llvm::LLVMContext &context, llvm::Module *originalModule,
                     const std::string &outputFilePath) {

  // Dump LLVM IR to string
  std::string llvm_ir_str;
  llvm::raw_string_ostream rso(llvm_ir_str);
  originalModule->print(rso, nullptr);
  rso.flush();

  // Regular expressions to replace aligned_alloc declarations and calls
  // of type (i64, i64) with (i32, i32), which was needed to support
  // the async runtime.
  std::regex decl_pattern(R"(declare\s+ptr\s+@aligned_alloc\(i64,\s*i64\))");
  std::string decl_replacement = "declare ptr @aligned_alloc(i32, i32)";

  std::regex call_pattern(R"(@aligned_alloc\(i64\s+(\d+),\s+i64\s+(\d+)\))");
  std::string call_replacement = "@aligned_alloc(i32 $1, i32 $2)";

  std::string modified_ir_str =
      std::regex_replace(llvm_ir_str, decl_pattern, decl_replacement);
  modified_ir_str =
      std::regex_replace(modified_ir_str, call_pattern, call_replacement);

  // Optionally write to file
  if (!outputFilePath.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream outFile(outputFilePath, EC, llvm::sys::fs::OF_Text);
    if (EC)
      llvm::errs() << "Error writing to file: " << EC.message() << "\n";
    else
      // Parse modified IR back into llvm::Module
      outFile << modified_ir_str;
  }

  // Parse modified IR back into llvm::Module
  llvm::SMDiagnostic err;
  auto modifiedModule = llvm::parseIR(
      llvm::MemoryBufferRef(modified_ir_str, "modified_module"), err, context);

  if (!modifiedModule)
    fail("Failed to parse modified LLVM IR after aligned_alloc fix.");

  return modifiedModule;
}

namespace hexagon_backend {

/// Classify argument types. We treat tensors, memrefs, and LLVM pointers as
/// "buffer-like". Everything else (integers, floats, index, vectors, etc.)
/// is "scalar-like".
// bool isBufferLike(mlir::Type t) {
//   if (mlir::isa<mlir::TensorType>(t))
//     return true;
//   if (mlir::isa<mlir::MemRefType>(t) ||
//   mlir::isa<mlir::UnrankedMemRefType>(t))
//     return true;
//   if (mlir::isa<mlir::LLVM::LLVMPointerType>(t))
//     return true;
//   return false;
// }

bool isBufferLike(mlir::Type t) {
  return llvm::isa<mlir::TensorType, mlir::BaseMemRefType,
                   mlir::LLVM::LLVMPointerType>(t);
}

/// Build a permutation that moves all buffer-like arguments to the front,
/// preserving the original relative order within buffers and within scalars.
///
/// Example: types [i32, memref<4xf32>, f32, tensor<2xf32>] =>
/// permutation [1, 3, 0, 2] (new order: memref, tensor, i32, f32).
llvm::SmallVector<unsigned>
buildBufferFirstPermutation(mlir::func::FuncOp func) {
  llvm::SmallVector<unsigned> bufferIdxs;
  llvm::SmallVector<unsigned> scalarIdxs;

  unsigned n = func.getNumArguments();
  bufferIdxs.reserve(n);
  scalarIdxs.reserve(n);

  for (unsigned i = 0; i < n; ++i) {
    mlir::Type t = func.getArgument(i).getType();
    (isBufferLike(t) ? bufferIdxs : scalarIdxs).push_back(i);
  }

  llvm::SmallVector<unsigned> perm;
  perm.reserve(n);
  perm.append(bufferIdxs.begin(), bufferIdxs.end());
  perm.append(scalarIdxs.begin(), scalarIdxs.end());
  return perm;
}

/// Reorder arguments and update call sites.
/// In-place reorder of one function's arguments:
/// - validates the permutation
/// - updates func::CallOp call sites
/// - rewrites entry block arguments
/// - updates function type and arg attributes
mlir::LogicalResult reorderFuncArgsAndCalls(mlir::ModuleOp &module,
                                            mlir::func::FuncOp func,
                                            llvm::ArrayRef<unsigned> perm) {
  unsigned n = func.getNumArguments();
  if (perm.size() != n)
    return func.emitError("Permutation size mismatch: got ")
           << perm.size() << ", expected " << n;

  // Validate permutation: values must be unique and < n
  llvm::SmallVector<unsigned> seen(n, 0);
  llvm::SmallVector<unsigned> invPerm(n); // inverse permutation
  for (unsigned newIdx = 0; newIdx < n; ++newIdx) {
    unsigned oldIdx = perm[newIdx];
    if (oldIdx >= n)
      return func.emitError("Invalid old index in permutation: ") << oldIdx;
    if (++seen[oldIdx] != 1)
      return func.emitError("Permutation must be a bijection. Index ")
             << oldIdx << " appears multiple times.";
    invPerm[oldIdx] = newIdx;
  }

  // Prepare old types
  llvm::SmallVector<mlir::Type> oldTypes(func.getArgumentTypes().begin(),
                                         func.getArgumentTypes().end());

  // New function type
  llvm::SmallVector<mlir::Type> newTypes(n);
  for (unsigned newIdx = 0; newIdx < n; ++newIdx)
    newTypes[newIdx] = oldTypes[perm[newIdx]];
  auto newFuncType = mlir::FunctionType::get(func.getContext(), newTypes,
                                             func.getResultTypes());

  // Extract old arg attrs
  llvm::SmallVector<mlir::DictionaryAttr> oldArgAttrs(n);
  if (auto allAttrsOpt = func.getArgAttrs()) {
    auto allAttrs = allAttrsOpt.value();
    for (unsigned i = 0; i < n && i < allAttrs.size(); ++i)
      oldArgAttrs[i] =
          mlir::dyn_cast_or_null<mlir::DictionaryAttr>(allAttrs[i]);
  }

  // Compute new arg attrs
  llvm::SmallVector<mlir::DictionaryAttr> newArgAttrs(n);
  for (unsigned newIdx = 0; newIdx < n; ++newIdx)
    newArgAttrs[newIdx] = oldArgAttrs[perm[newIdx]];

  // Update call sites
  if (auto uses = mlir::SymbolTable::getSymbolUses(func, module)) {
    for (auto &use : *uses) {
      mlir::Operation *user = use.getUser();
      if (auto call = llvm::dyn_cast<mlir::func::CallOp>(user)) {
        auto oldOperands = call.getArgOperands();
        if (oldOperands.size() != n)
          return call.emitError("Call operand count mismatch: got ")
                 << oldOperands.size() << ", expected " << n;

        llvm::SmallVector<mlir::Value> newOperands(n);
        for (unsigned newIdx = 0; newIdx < n; ++newIdx)
          newOperands[newIdx] = oldOperands[perm[newIdx]];
        call->setOperands(newOperands);
      }
    }
  }

  // External function: just update type and attrs
  if (func.isExternal()) {
    func.setType(newFuncType);
    for (unsigned i = 0; i < n; ++i)
      func.setArgAttrs(i, newArgAttrs[i]);
    return mlir::success();
  }

  // Rewrite entry block
  mlir::Block &entry = func.getBody().front();
  llvm::SmallVector<mlir::BlockArgument> oldArgs(entry.getArguments().begin(),
                                                 entry.getArguments().end());
  if (oldArgs.size() != n)
    return func.emitError("Entry block arg count mismatch: got ")
           << oldArgs.size() << ", expected " << n;

  llvm::SmallVector<mlir::Location> oldLocs;
  oldLocs.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    oldLocs.push_back(oldArgs[i].getLoc());

  llvm::SmallVector<mlir::BlockArgument> newArgs(n);
  for (unsigned newIdx = 0; newIdx < n; ++newIdx) {
    unsigned oldIdx = perm[newIdx];
    newArgs[newIdx] =
        entry.addArgument(oldArgs[oldIdx].getType(), oldLocs[oldIdx]);
  }

  // Replace uses (O(n) using invPerm)
  for (unsigned oldIdx = 0; oldIdx < n; ++oldIdx) {
    unsigned newIdx = invPerm[oldIdx];
    oldArgs[oldIdx].replaceAllUsesWith(newArgs[newIdx]);
  }

  // Erase old arguments
  for (int i = static_cast<int>(n) - 1; i >= 0; --i)
    entry.eraseArgument(i);

  // Update function type and attrs
  func.setType(newFuncType);
  for (unsigned i = 0; i < n; ++i)
    func.setArgAttrs(i, newArgAttrs[i]);

  return mlir::success();
}

bool reorderFuncArgsAndCallsTensorFirst(mlir::ModuleOp &module_op,
                                        const std::string &fname) {
  bool changed = false;
  for (auto func : module_op.getOps<mlir::func::FuncOp>()) {
    if (func.getName() != fname)
      continue;
    unsigned n = func.getNumArguments();
    if (n == 0)
      continue;

    // Build buffer-first permutation.
    llvm::SmallVector<unsigned> perm = buildBufferFirstPermutation(func);

    // If already buffer-first, skip.
    // Requires: perm is a permutation of 0..n-1 (distinct, full coverage)
    bool isIdentity = llvm::is_sorted(perm) && !perm.empty() &&
                      perm.front() == 0 && perm.back() == perm.size() - 1;

    if (isIdentity)
      continue;

    if (!failed(reorderFuncArgsAndCalls(module_op, func, perm))) {
      changed = true;
    }
  }
  return changed;
}

// Extract the return types (rank and dtype) of a function given its name.
std::vector<std::pair<int, mlir::Type>>
getReturnList(mlir::ModuleOp module_op, const std::string &fName) {
  std::vector<std::pair<int, mlir::Type>> resultList;

  for (auto funcOp : module_op.getOps<mlir::func::FuncOp>()) {
    if (funcOp.getName() != fName)
      continue;
    for (auto resultType : funcOp.getResultTypes()) {
      int rank;
      mlir::Type dtype;

      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
        rank = tensorType.getRank();
        dtype = tensorType.getElementType();
      } else {
        // Assuming if it's not RankedTensor, it's a scalar value.
        rank = 0;
        dtype = resultType;
      }
      resultList.push_back(std::make_pair(rank, dtype));
    }
  }
  // Return the list of return types.
  return resultList;
}

std::string extractSingleFuncName(mlir::ModuleOp &module_op) {
  std::string module_func_name;
  std::vector<std::string> vector_func_names;
  for (auto func : module_op.getOps<mlir::func::FuncOp>())
    vector_func_names.push_back(func.getName().str());
  if (vector_func_names.size() != 1)
    fail("Expected exactly one function in the module.");

  module_func_name = vector_func_names[0];
  return module_func_name;
}

void loadDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);   // TODO: restrict
  mlir::registerAllExtensions(registry); // TODO: restrict

  registry.insert<mlir::crouton::CroutonDialect>();
  registry.insert<mlir::hexagonmem::HexagonMemDialect>();
  registry.insert<mlir::hexkl::HexKLDialect>();
  registry.insert<mlir::hmx::HmxDialect>();
  registry.insert<mlir::hvx::HvxDialect>();
  registry.insert<mlir::tm_tensor::TmTensorDialect>();
  registry.insert<mlir::ttx::TTXDialect>();
  registry.insert<mlir::tptr::HexagonTPtrDialect>();

  // Register all external models.
  mlir::hexkl::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::hmx::registerBufferizableOpInterfaceExternalModels(registry);

  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::triton::TritonDialect>();
  context.loadAllAvailableDialects();
}

mlir::ModuleOp parseMlirFromFile(const std::string &path,
                                 mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
  if (!module)
    fail("Parse MLIR file failed.");
  return module->clone();
}

mlir::ModuleOp parseMlirFromString(const std::string &src,
                                   mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(src, &context);
  if (!module)
    fail("Parse MLIR string failed.");
  return module->clone();
}

static void validatePrepackAttributes(mlir::ModuleOp module) {
  auto validate = [&](llvm::StringRef name, bool array) {
    mlir::Attribute raw = module->getAttr(name);
    if (!raw)
      return;
    auto string = mlir::dyn_cast<mlir::StringAttr>(raw);
    if (!string || string.getValue().empty())
      fail("malformed HMX prepack attribute: " + name.str());
    auto parsed = llvm::json::parse(string.getValue());
    if (!parsed || (array ? !parsed->getAsArray() : !parsed->getAsObject()))
      fail("malformed HMX prepack JSON: " + name.str());
  };
  validate("hmx.weight_prepack", /*array=*/true);
  validate("hmx.weight_prepack_layout", /*array=*/false);
}

static std::string buildTranslationMetadata(mlir::ModuleOp module) {
  validatePrepackAttributes(module);
  auto weightAttr = module->getAttrOfType<mlir::StringAttr>(
      "hmx.weight_prepack");
  auto layoutAttr = module->getAttrOfType<mlir::StringAttr>(
      "hmx.weight_prepack_layout");

  std::string json = "{\"schema\":\"hex.hmx.translation/v1\",";
  json += "\"weight_prepack\":{\"layout\":";
  json += layoutAttr ? layoutAttr.getValue().str() : "null";
  json += ",\"weights\":";
  json += weightAttr ? weightAttr.getValue().str() : "[]";
  std::string manifest = mlir::hmx::serializeHmxManifestJson(module);
  if (manifest == "{}")
    fail("HMX manifest is missing or malformed for the translated module");
  json += "},\"hmx_manifest\":";
  json += manifest;
  json += "}";
  return json;
}

std::vector<std::vector<char>> translateLinalgToObj(
    mlir::ModuleOp &linalg_module,
    const std::unordered_map<std::string, std::string> &options_map,
    std::string *outMetadata) {
  // The collection of object codes (each seen as a sequence of
  // bytes/char) that will be produced
  std::vector<std::vector<char>> mods_object_codes_as_bytes;
  validatePrepackAttributes(linalg_module);

  // Needed to know if we should lower the constants separately or not
  mlir::hexagon::LinalgToLLVMOptions optionsLinalgToLLVM;
  // sets optionsLinalgToLLVM using options_map
  setLinalgToLLVMOptions(optionsLinalgToLLVM, options_map);

  std::vector<mlir::ModuleOp> mods;
  // Normal mode, where constants stay in the main module
  // ----------------------------------------------------
  if (!optionsLinalgToLLVM.lowerConstantsInSeparateSharedObjects) {
    // Old (default) mode where the Linalg/MLIR module gets lowered
    // to a single LLVM/MLIR module
    auto mod =
        ::mlir::hexagon::translateLinalgToLLVMMLIR(linalg_module, options_map);
    if (!mod)
      fail("Failed to convert Triton Linalg to LLVM MLIR.");
    mods.push_back(mod);
  } else {
    // New mode where constants are lowered separately into their own .o
    // and .so
    DBG("STEP 0 - Calling translateLinalgToMultipleLLVMMLIRModules"
        << "\n");
    mods = ::mlir::hexagon::translateLinalgToMultipleLLVMMLIRModules(
        linalg_module, options_map);
    if (mods.empty())
      fail("Error in the production of multiple modules: no modules produced");
  }

  DBG("There are a total of " << mods.size()
                              << " modules, including the main (code) one"
                              << "\n");

  if (!outMetadata) {
    if (mods.empty() ||
        mlir::hmx::serializeHmxManifestJson(mods[0]) == "{}")
      fail("translated module is missing a valid HMX manifest; request "
           "with_meta=true for the versioned envelope");
    validatePrepackAttributes(mods[0]);
    if (auto prepack =
            mods[0]->getAttrOfType<mlir::StringAttr>("hmx.weight_prepack")) {
      llvm::StringRef value = prepack.getValue();
      if (!value.empty() && value != "[]")
        fail("translated module contains a runtime weight-prepack contract; "
             "request with_meta=true and consume it");
    }
  }

  // Return one versioned translation envelope alongside the object bytes.  The
  // envelope is an internal C++/Python transport contract: the weight object
  // is consumed by the launcher, while the HMX manifest is host diagnostics
  // only.  The core owns manifest serialization; the weight fragments below
  // preserve the existing launcher contract.
  if (outMetadata) {
    if (mods.empty())
      fail("Cannot construct translation metadata: no translated module was "
           "produced.");
    *outMetadata = buildTranslationMetadata(mods[0]);
  }

  // Iterating through each LLVM/MLIR module that has been produced
  int module_id = -1; // current module being treated
  for (mlir::ModuleOp current_mlir_mod : mods) {
    module_id++;
    if (!current_mlir_mod)
      fail("Failed to convert Triton Linalg to LLVM/MLIR for the current "
           "module" +
           std::to_string(module_id));

    llvm::LLVMContext llvmContext;
    DBG("[Module " << module_id
                   << "] - STEP 1 - Calling translateHexagonMlirLlvmToLLVMIR"
                   << "\n");
    // If it's the main module (for the code), use the optLevel 3,
    // otherwise no opt to avoid the modules containing only constants of
    // being completely elided
    mlir::hexagon::LLVMOptimizationLevel opt_level_translation =
        (module_id == 0) ? mlir::hexagon::LLVMOptimizationLevel::O3
                         : mlir::hexagon::LLVMOptimizationLevel::NoOptimization;
    // LLVM/MLIR module to LLVM-IR
    std::unique_ptr<llvm::Module> current_llvm_mod =
        ::mlir::hexagon::translateHexagonMlirLlvmToLLVMIR(
            &llvmContext, current_mlir_mod, options_map, opt_level_translation);

    if (!current_llvm_mod)
      fail("Failed to translate LLVM/MLIR to LLVM-IR for the current module" +
           std::to_string(module_id));

    // Only if it's the principal module (=code), link the runtime modules
    // with it
    if (module_id == 0) {
      DBG("[Module " << module_id
                     << "] - Extra step - Calling linkRuntimeModules for the "
                        "principal module"
                     << "\n");
      mlir::Hexagon::Translate::linkRuntimeModules(
          llvmContext, current_llvm_mod, options_map);

      // HEXAGON_PTR_LOG (fa-rowmax plan §15 Step-1, temporary): log the
      // runtime pointer values flowing into memset (arg0) and returned by the
      // allocators, so the bad pointer behind the FA launch crash can be
      // named from the device-side trace (rt_trc.txt).
      if (mlir::hexagon::isEnvTrue("HEXAGON_PTR_LOG")) {
        llvm::IRBuilder<> b(llvmContext);
        llvm::LLVMContext &ctx = llvmContext;
        auto *i32 = llvm::Type::getInt32Ty(ctx);
        auto *ptrt = llvm::PointerType::getUnqual(ctx);
        auto *fty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                            {i32, ptrt}, false);
        llvm::FunctionCallee logger = current_llvm_mod->getOrInsertFunction(
            "hexagon_runtime_dbg_log_ptr", fty);
        struct Site {
          llvm::Instruction *ip;
          llvm::Value *logged;
          int id;
          bool after;
        };
        std::vector<Site> sites;
        int id = 0;
        for (llvm::Function &F : *current_llvm_mod)
          for (llvm::BasicBlock &BB : F)
            for (llvm::Instruction &I : BB) {
              if (auto *MI = llvm::dyn_cast<llvm::MemIntrinsic>(&I)) {
                // llvm.memset.* intrinsic (becomes `call memset` at ISel)
                sites.push_back({&I, MI->getRawDest(), 1000 + id++, false});
                ++id; // Deliberate hole: these ids are labels in the archived
                      // triage traces (rt_trc.txt), so the sequence stays as
                      // recorded -- do not "fix" the numbering, it would break
                      // comparison with those logs.
                continue;
              }
              auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
              if (!CI)
                continue;
              auto *CF = CI->getCalledFunction();
              if (!CF)
                continue;
              llvm::StringRef name = CF->getName();
              // id kind-hundreds: 1000=memset, 2000=malloc,
              // 3000=aligned_alloc, 4000=hexagon_runtime_alloc_*
              if (name == "memset") {
                sites.push_back({CI, CI->getArgOperand(0), 1000 + id++,
                                 /*after=*/false});
              } else if (name == "mlirAsyncRuntimeIsGroupError") {
                // 9003 = group value ENTERING the call (before): names a
                // corrupt group pointer before IsGroupError can fault on it.
                sites.push_back({CI, CI->getArgOperand(0), 9003,
                                 /*after=*/false});
                sites.push_back({CI, CI->getArgOperand(0), 9001, /*after=*/true});
                // 9004 = the i1 result (p0) as 0/1: proves abort-taken vs
                // fall-through. Build zext+inttoptr right after the call.
                llvm::IRBuilder<> nb(CI->getNextNode());
                llvm::Value *ext = nb.CreateZExt(CI, i32, "p0.zext");
                llvm::Value *pt =
                    nb.CreateIntToPtr(ext, ptrt, "p0.ptr");
                sites.push_back({llvm::cast<llvm::Instruction>(pt), pt, 9004,
                                 /*after=*/true});
              } else if (name == "mlirAsyncRuntimeAwaitAllInGroup") {
                sites.push_back({CI, CI->getArgOperand(0), 9002, /*after=*/true});
              } else if (name == "malloc") {
                sites.push_back({CI, CI, 2000 + id++, /*after=*/true});
              } else if (name == "aligned_alloc") {
                sites.push_back({CI, CI, 3000 + id++, /*after=*/true});
              } else if (name.starts_with("hexagon_runtime_alloc")) {
                sites.push_back({CI, CI, 4000 + id++, /*after=*/true});
              } else {
                ++id;
              }
            }
        for (auto &s : sites) {
          b.SetInsertPoint(s.after ? s.ip->getNextNode() : s.ip);
          b.CreateCall(logger,
                       {llvm::ConstantInt::get(i32, s.id), s.logged});
        }
        llvm::errs() << "[ptr-log] instrumented " << sites.size()
                     << " pointer sites\n";
      }

      // HEXAGON_GUARD_BOOL_PIN (fa-crash-investigation-log R16/T1): pin the
      // await-guard condition to memory. PROVEN on device (naked binary):
      // after `call IsGroupError`, codegen emits `r0 = r21` (a move for a
      // LATER group use) BEFORE `p0 = r0`, so p0 reads the group pointer
      // (nonzero) instead of the call result -> abort always taken -> silent
      // launch death at the first window; nopping the abort jump lets 3
      // windows through. Same interference family as %1579 (overlapping
      // ranges share a register). The store captures the true i1 right
      // after the call; the reload sits adjacent before its br (a
      // terminator: nothing can slip between), so no barrier is needed.
      // Gate: calls to mlirAsyncRuntimeIs{Group,Token,Value}Error.
      // Env "0" disables (A/B).
      {
        bool guardPinOn =
            mlir::hexagon::isEnvTrue("HEXAGON_GUARD_BOOL_PIN");
        if (guardPinOn) {
          unsigned nGuard = 0;
          for (llvm::Function &F : *current_llvm_mod) {
            if (F.isDeclaration())
              continue;
            for (llvm::BasicBlock &BB : F) {
              // collect first (inserting while iterating is unsafe)
              std::vector<llvm::CallInst *> guards;
              for (llvm::Instruction &I : BB) {
                auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
                if (!CI || !CI->getCalledFunction())
                  continue;
                llvm::StringRef n = CI->getCalledFunction()->getName();
                if (n == "mlirAsyncRuntimeIsGroupError" ||
                    n == "mlirAsyncRuntimeIsTokenError" ||
                    n == "mlirAsyncRuntimeIsValueError")
                  guards.push_back(CI);
              }
              for (auto *CI : guards) {
                auto *nx = CI->getNextNode();
                if (!nx)
                  continue;
                auto &entry = CI->getParent()->getParent()->getEntryBlock();
                auto insPt = entry.getFirstInsertionPt();
                llvm::Instruction *anchor = (insPt == entry.end())
                                                ? entry.getTerminator()
                                                : &*insPt;
                llvm::IRBuilder<> eb(anchor);
                auto *slot = eb.CreateAlloca(CI->getType(), nullptr,
                                             "guard.cond");
                llvm::IRBuilder<> sb(nx);
                sb.CreateStore(CI, slot);
                // rewire every non-PHI instruction user to a reload
                std::vector<llvm::User *> users(CI->users().begin(),
                                                CI->users().end());
                for (llvm::User *U : users) {
                  if (U == slot)
                    continue;
                  auto *st = llvm::dyn_cast<llvm::StoreInst>(U);
                  if (st && st->getPointerOperand() == slot)
                    continue; // our own store
                  auto *UI = llvm::dyn_cast<llvm::Instruction>(U);
                  if (!UI || llvm::isa<llvm::PHINode>(UI))
                    continue;
                  llvm::IRBuilder<> ub(UI);
                  llvm::LoadInst *ld = ub.CreateLoad(
                      CI->getType(), slot, "guard.reload");
                  for (unsigned oi = 0; oi < U->getNumOperands(); ++oi)
                    if (U->getOperand(oi) == CI)
                      U->setOperand(oi, ld);
                }
                ++nGuard;
              }
            }
          }
          llvm::errs() << "[guard-bool-pin] pinned " << nGuard
                       << " await-guard conditions\n";
        }
      }

      // HEXAGON_EPI_LOG (TEMP triage, fa-crash R21): log every HMX leaf
      // call's integer address operands (they carry addresses-as-ints).
      // Straight-line code only in practice; names a wild HMX address with
      // ~15 cold loggers. Env-gated, default off. DELETE after triage.
      if (mlir::hexagon::isEnvTrue("HEXAGON_EPI_LOG")) {
        llvm::IRBuilder<> b(llvmContext);
        llvm::LLVMContext &ctx = llvmContext;
        auto *i32 = llvm::Type::getInt32Ty(ctx);
        auto *ptrt = llvm::PointerType::getUnqual(ctx);
        auto *fty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                            {i32, ptrt}, false);
        llvm::FunctionCallee logger = current_llvm_mod->getOrInsertFunction(
            "hexagon_runtime_dbg_log_ptr", fty);
        int eid = 7000;
        std::vector<std::tuple<llvm::CallInst *, llvm::Value *, int>> esites;
        for (llvm::Function &F : *current_llvm_mod) {
          if (F.isDeclaration())
            continue;
          for (llvm::BasicBlock &BB : F)
            for (llvm::Instruction &I : BB) {
              auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
              if (!CI || !CI->getCalledFunction())
                continue;
              llvm::StringRef n = CI->getCalledFunction()->getName();
              if (!n.starts_with("hmx_"))
                continue;
              for (unsigned ai = 0; ai < CI->arg_size(); ++ai) {
                llvm::Value *a = CI->getArgOperand(ai);
                if (!a->getType()->isIntegerTy(32) &&
                    !a->getType()->isIntegerTy(64) &&
                    !a->getType()->isPointerTy())
                  continue;
                esites.push_back({CI, a, eid++});
              }
            }
        }
        for (auto &t : esites) {
          llvm::CallInst *CI;
          llvm::Value *a;
          int id;
          std::tie(CI, a, id) = t;
          llvm::IRBuilder<> eb(CI);
          llvm::Value *p = a;
          if (!a->getType()->isPointerTy())
            p = eb.CreateIntToPtr(a, ptrt, "epi.ptr");
          eb.CreateCall(logger,
                        {llvm::ConstantInt::get(i32, id), p});
        }
        llvm::errs() << "[epi-log] instrumented " << esites.size()
                     << " hmx arg sites\n";
      }

      // HEXAGON_MEMSET_DST_FIX (fa-rowmax plan §16, the R1 unblocker):
      // The FA crash root cause: a memset destination whose def-to-use span
      // crosses the whole kernel (async groups etc.) came out of the RA holding
      // a foreign value (observed: float bits where `and -128` guarantees a
      // 128-aligned pointer, validated against an R1-OFF control) -- a
      // long-range register/spill-slot interference that static analysis
      // cannot see. Fix: pin such destinations to memory -- store the value to
      // a DEDICATED alloca right after its def and re-load it immediately
      // before every use, so no register ever carries it across the region.
      // Gate (triage form R19, NOT a general mechanism): only kernels whose
      // name contains "attention" or starts with "async_execute" (the R8
      // scope), only IntToPtrInst results, and only when some use sits more
      // than 150 IR-walk instructions from the def. The kernel-name scoping
      // is a deliberate triage concession under AGENTS.md's no-shape/no-name
      // rule -- the interference was only ever observed there, and widening
      // the scope is a mechanism question, not a knob. Env "0" disables (A/B).
      {
        bool dstFixOn =
            mlir::hexagon::isEnvTrue("HEXAGON_MEMSET_DST_FIX");
        if (dstFixOn) {
          // ---- RA-corruption guard (fa-rowmax plan §16) -------------------
          // Evidence (device, runtime value dump + R1-OFF control): a
          // long-lived pointer SSA reconstituted through integer arithmetic
          // (`ptrtoint -> add -> and -> inttoptr`, the allocator-alignment
          // idiom) came out of codegen holding a foreign value (float bits
          // where `and -128` guarantees alignment) and was passed to memset:
          // memset(0x3f38aa3b, 0, 2048) -> hard fault. The def-to-use span
          // crosses the whole kernel; static analysis cannot see the
          // interference, but pinning the value to memory makes it invisible
          // to the register allocator.
          // Store the gated value to a DEDICATED entry-block alloca right
          // after its def and re-load it immediately before each use (the
          // span/name/int-type gate itself is stated at the block header;
          // R19 form: args are NOT candidates, NO barriers anywhere).
          llvm::DenseMap<llvm::Value *, llvm::AllocaInst *> pinned;
          unsigned nPinned = 0;

          auto rewireAllUses = [&](llvm::Value *d, llvm::AllocaInst *slot) {
            auto *defI = llvm::dyn_cast<llvm::Instruction>(d);
            llvm::BasicBlock *defBB = defI ? defI->getParent() : nullptr;
            std::vector<llvm::User *> users(d->users().begin(),
                                            d->users().end());
            for (llvm::User *U : users) {
              if (U == slot)
                continue;
              if (auto *st = llvm::dyn_cast<llvm::StoreInst>(U)) {
                // any of our own pin stores (pointer operand is a pin
                // slot): leave alone, else we would rewire our own stores.
                if (pinned.count(st->getPointerOperand()))
                  continue;
              }
              auto *I = llvm::dyn_cast<llvm::Instruction>(U);
              if (!I || llvm::isa<llvm::PHINode>(I))
                continue;
              // R20: rewire ONLY uses across a call. A same-block use with
              // no intervening call already receives the value correctly
              // (adjacent, no clobber opportunity); rewiring it only
              // perturbs scheduling (this killed the prologue in R11-R19:
              // reloads before prologue-adjacent uses reshuffled the hmx
              // handoff). Skipping them leaves the prologue byte-identical
              // in effect while far uses get memory-roundtripped values.
              // Bonus: the original range shrinks to its last kept use.
              if (defBB && I->getParent() == defBB) {
                bool callBetween = false;
                for (llvm::Instruction *K = defI->getNextNode();
                     K && K != I; K = K->getNextNode()) {
                  if (llvm::isa<llvm::CallInst>(K)) {
                    callBetween = true;
                    break;
                  }
                }
                if (!callBetween)
                  continue; // keep original: already correct, zero perturbation
              }
              llvm::IRBuilder<> ub(I);
              llvm::LoadInst *ld =
                  ub.CreateLoad(d->getType(), slot, "mst.reload");
              // R21: volatile. Machine-CSE merges identical plain reloads
              // of the same slot back into ONE reload feeding all far uses
              // -- recreating exactly the behemoth this pin exists to split
              // (evidence: R20 IR has 112 single-use reloads; the bug needs
              // multi-use ranges). Volatile loads are never merged,
              // eliminated, or reordered across the slot store.
              ld->setVolatile(true);
              // R16/T2a: NO barriers anywhere. Barriers are calls; every
              // added call perturbs scheduling (R7/R12/R14 all moved deaths
              // without converging). Memops-only fix here.
              for (unsigned oi = 0; oi < U->getNumOperands(); ++oi)
                if (U->getOperand(oi) == d)
                  U->setOperand(oi, ld);
            }
          };

          // Pass 1 (R19-distance): pin ONLY IntToPtrInst results with a use
          // >150 IR-walk instructions away (kernel+task fns). Rationale:
          // cross-block alone caught 14 (incl. prologue chains whose near
          // uses sit across a branch) and the prologue died again. What
          // matters is SPAN: behemoths (%135/%1579-class) span 500-1500
          // lines across dozens of calls; prologue-local chains span <50.
          // 150 splits them with margin. (A loop back-edge can only
          // UNDER-count dynamic span; acceptable: it errs toward pinning.)
          // NO args/calls/GEPs/loads, NO barriers. T1-guard stays.
          std::vector<std::pair<llvm::Value *, llvm::Instruction *>>
              candidates;
          for (llvm::Function &F : *current_llvm_mod) {
            if (F.isDeclaration())
              continue; // no body: no entry block, nothing to pin
            llvm::StringRef fn = F.getName();
            if (!fn.contains("attention") && !fn.starts_with("async_execute"))
              continue; // R8: kernel+task fns only (R1-affected code)
            // sequence numbers for span measurement
            llvm::DenseMap<llvm::Instruction *, unsigned> seq;
            unsigned n = 0;
            for (llvm::BasicBlock &BB : F)
              for (llvm::Instruction &I : BB)
                seq[&I] = n++;
            for (llvm::BasicBlock &BB : F)
              for (llvm::Instruction &I : BB) {
                auto *ITP = llvm::dyn_cast<llvm::IntToPtrInst>(&I);
                if (!ITP)
                  continue;
                bool farUse = false;
                for (llvm::User *U : ITP->users()) {
                  auto *UI = llvm::dyn_cast<llvm::Instruction>(U);
                  if (!UI) {
                    farUse = true;
                    break;
                  }
                  auto it = seq.find(UI);
                  if (it == seq.end()) {
                    farUse = true;
                    break;
                  }
                  unsigned d = it->second > seq[&I] ? it->second - seq[&I]
                                                    : seq[&I] - it->second;
                  if (d > 150) {
                    farUse = true;
                    break;
                  }
                }
                if (!farUse)
                  continue;
                candidates.push_back({&I, &I});
              }
          }
          // Pass 2: ensure a slot after each def (store), then rewire uses.
          for (auto &c : candidates) {
            if (pinned.count(c.first))
              continue;
            // Pass 1 only ever records {&I, &I} (IntToPtrInst results), so the
            // old llvm::Argument fallback here was unreachable dead code
            // (removed 2026-09-23); the null guard stays as the cheap proof.
            if (!c.second)
              continue;
            llvm::Function *FN = c.second->getParent()->getParent();
            llvm::Instruction *nx = c.second->getNextNode();
            if (!nx)
              continue; // terminator-positioned def (e.g. invoke): skip
            // CRITICAL: the slot alloca must be created in the ENTRY block --
            // an alloca in a mid-function block lowers to a *dynamic* stack
            // adjustment with no matching stackrestore, so r29 skews across
            // every subsequent call => wild accesses and garbage values
            // (fa-crash-investigation-log R6). The store itself stays right
            // after the def (args removed; the store stays right after the def) so the value enters memory
            // unconditionally on every path that can reach a use.
            llvm::BasicBlock &entry = FN->getEntryBlock();
            auto insPt = entry.getFirstInsertionPt();
            // A degenerate entry holding only a terminator has no valid
            // first-insertion-point: fall back to the terminator itself.
            llvm::Instruction *anchor =
                (insPt == entry.end()) ? entry.getTerminator() : &*insPt;
            llvm::IRBuilder<> eb(anchor);
            auto *slot = eb.CreateAlloca(c.first->getType(), nullptr,
                                         "mst.dst");
            llvm::IRBuilder<> sb(nx); // nx non-null: candidates are defs
            sb.CreateStore(c.first, slot);
            pinned.insert({c.first, slot});
            ++nPinned;
          }
          // Pass 3: rewire every use of every pinned value to a reload.
          for (auto &kv : pinned)
            rewireAllUses(kv.first, kv.second);
          // (R16/T2a: Pass 4 barriers-before-calls REMOVED -- barriers are
          // calls and every added call perturbs scheduling. Memops-only.)
          llvm::errs() << "[memset-dst-fix] pinned " << nPinned
                       << " long-lived pointer values\n";
        } // dstFixOn
      }     // HEXAGON_MEMSET_DST_FIX scope

      // Aggressive inlining to improve performance after linking runtime
      // modules Check if inlining is enabled via options_map
      mlir::hexagon::cond_run_inliner(current_llvm_mod,
                                      optionsLinalgToLLVM.enableHVXInlining);

      // Dumping of the principal module if requested
      const char *dumpFile = std::getenv("LLVM_IR_DUMP_TO_FILE");
      if (dumpFile && dumpFile[0] != '\0') {
        std::error_code EC;
        llvm::raw_fd_ostream outFile(dumpFile, EC, llvm::sys::fs::OF_Text);
        if (EC) {
          llvm::errs() << "Error opening LLVM IR file: " << EC.message()
                       << "\n";
        } else {
          current_llvm_mod->print(outFile, nullptr);
          outFile.flush();
        }
      }
    }

    DBG("[Module " << module_id
                   << "] - STEP 2 - Calling llvm_module_to_obj_string"
                   << "\n");
    auto replacedAlignedModule =
        fixAlignedAllocTypes(llvmContext, current_llvm_mod.get(), "");

    if (!replacedAlignedModule)
      fail("Failed to fix aligned_alloc types in LLVM module for the current "
           "module" +
           std::to_string(module_id));

    if (mlir::hexagon::isEnvTrue("HEXAGON_ASM_DUMP") ||
        mlir::hexagon::isEnvTrue("HEXAGON_ASM_TO_OBJ")) {
      const char *archEnv = std::getenv("HEXAGON_ARCH_VERSION");
      if (!archEnv) {
        llvm::errs() << "Warning: HEXAGON_ASM_DUMP is set but "
                        "HEXAGON_ARCH_VERSION is not set. Skipping "
                        "assembly dump.\n";
      } else {
        std::string archVersion = std::string(archEnv);
        mlir::hexagon::dumpHexagonAssembly(*replacedAlignedModule, module_id,
                                           archVersion);
      }
    }

    std::vector<char> object_code_as_bytes =
        mlir::hexagon::llvm_module_to_obj_string(replacedAlignedModule);
    // For now, the collection returned has only one object code
    mods_object_codes_as_bytes.push_back(object_code_as_bytes);
  }

  return mods_object_codes_as_bytes;
}

std::string translateLinalgToLLVMIR(
    mlir::ModuleOp &linalg_module,
    const std::unordered_map<std::string, std::string> &options_map,
    std::string *outMetadata) {
  validatePrepackAttributes(linalg_module);
  auto mod =
      ::mlir::hexagon::translateLinalgToLLVMMLIR(linalg_module, options_map);
  if (!mod)
    fail("Failed to convert Triton Linalg to LLVM MLIR.");

  if (outMetadata) {
    *outMetadata = buildTranslationMetadata(mod);
  } else {
    validatePrepackAttributes(mod);
    if (mlir::hmx::serializeHmxManifestJson(mod) == "{}")
      fail("translated module is missing a valid HMX manifest; request "
           "metadata for the versioned envelope");
    if (auto prepack =
            mod->getAttrOfType<mlir::StringAttr>("hmx.weight_prepack")) {
      llvm::StringRef value = prepack.getValue();
      if (!value.empty() && value != "[]")
        fail("translated module contains a runtime weight-prepack contract; "
             "request metadata and consume it");
    }
  }

  // llvm mlir module to llvm ir
  llvm::LLVMContext llvmContext;

  auto llvmModule = ::mlir::hexagon::translateHexagonMlirLlvmToLLVMIR(
      &llvmContext, mod, options_map, mlir::hexagon::LLVMOptimizationLevel::O3);

  if (!llvmModule)
    fail("Failed to translate TritonHexagon to LLVM IR.");

  mlir::Hexagon::Translate::linkRuntimeModules(llvmContext, llvmModule,
                                               options_map);

  // Aggressive inlining to improve performance after linking runtime modules
  // Check if inlining is enabled via options_map
  mlir::hexagon::LinalgToLLVMOptions optionsLinalgToLLVM;
  setLinalgToLLVMOptions(optionsLinalgToLLVM, options_map);

  mlir::hexagon::cond_run_inliner(llvmModule,
                                  optionsLinalgToLLVM.enableHVXInlining);

  std::string str;
  llvm::raw_string_ostream os(str);
  llvmModule->print(os, nullptr);
  os.flush();
  return str;
}

} // namespace hexagon_backend