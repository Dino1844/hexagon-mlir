//===- HexagonLWPInstrumentation.cpp - light weight profiler --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// A light weight profiler.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxReadoutHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxRoleHandoff.h"
#include "hexagon/Transforms/OptionsParsing.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <string>

#define DEBUG_TYPE "hexagon-lwp-instrumentation"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace std::literals;
using namespace mlir;
using namespace mlir::scf;
using namespace mlir::LLVM;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONLWPPASS
#include "hexagon/Transforms/Passes.h.inc"

namespace {
struct HexagonLWPPass : public ::impl::HexagonLWPPassBase<HexagonLWPPass> {

  explicit HexagonLWPPass(const HexagonLWPPassOptions &options)
      : HexagonLWPPassBase(options) {}

  // The instrumentation below creates LLVM-dialect ops -- the handler global,
  // the intrinsic declaration, the GEP and the calls. Declaring the
  // dependency makes the pass runnable on IR that holds none yet (a module
  // with only the kernel to instrument); in the full pipeline other passes
  // load the dialect first, which is why this never fired there.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  // Compute loop depth (distance to outermost loop)
  static int getLoopDepth(Operation *op) {
    int depth = 0;
    while (auto parent = dyn_cast_or_null<scf::ForOp>(op->getParentOp())) {
      ++depth;
      op = parent;
    }
    return depth;
  }

  // Check if a loop has at least one sibling scf.for loop. Siblings are ops
  // in the loop's OWN block, not in the parent op's first region: a loop can
  // sit in any region of a multi-region parent (an scf.if branch, say), and
  // region 0 of such a parent can be empty -- `front()` on an empty region
  // dereferences the block-list sentinel (`!isKnownSentinel()`), and even
  // when region 0 is populated it is the wrong block for a loop that lives
  // in a later region.
  static bool hasSiblingLoop(scf::ForOp forOp) {
    for (Operation &op : *forOp->getBlock()) {
      if (&op != forOp.getOperation() && isa<scf::ForOp>(op))
        return true;
    }
    return false;
  }

  // Dump the location and ops info into a file so they can be postprocessed.
  void printLines(const std::vector<int> &lines, int loopId,
                  const std::vector<std::string> &opNames) const {
    static bool fileInitialized = false;
    const std::string filePath = "/tmp/lwp_infodump.txt";

    // Remove the stale file for fresh start.
    if (!fileInitialized) {
      if (llvm::sys::fs::exists(filePath)) {
        llvm::sys::fs::remove(filePath);
      }
      fileInitialized = true;
    }

    std::error_code EC;
    llvm::raw_fd_ostream fileStream(filePath, EC, llvm::sys::fs::OF_Append);

    if (EC) {
      llvm::errs() << "Error opening file: " << EC.message() << "\n";
      return;
    }

    fileStream << "Location ";
    for (size_t i = 0; i < lines.size(); ++i) {
      fileStream << lines[i];
      if (i != lines.size() - 1)
        fileStream << ", ";
    }

    fileStream << " corresponds to ID " << loopId << " | Collected ops: ";

    if (opNames.empty()) {
      fileStream << "\n";
    } else {
      for (size_t i = 0; i < opNames.size(); ++i) {
        fileStream << opNames[i];
        if (i != opNames.size() - 1)
          fileStream << ", ";
      }
      fileStream << "\n";
    }

    fileStream.flush();
  }

  // Get location info in a list.
  static std::vector<int> collectLoc(Location loc) {
    std::vector<int> lines;
    // FusedLoc can have nested FusedLoc.
    // Recursively get all the relevant locations in lines list.
    std::function<void(Location)> recurse = [&](Location loc) {
      if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
        for (Location subLoc : fusedLoc.getLocations())
          recurse(subLoc);
      } else if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
        lines.push_back(fileLoc.getLine());
      }
    };

    recurse(loc);
    return lines;
  }

  // Collect ops of interest.
  std::vector<std::string> collectOps(scf::ForOp forOp) {
    std::vector<std::string> opNames;
    forOp.getBody()->walk([&](Operation *op) {
      // Filter out non-trivial operations
      auto opName = op->getName().getStringRef();

      if (opName == "memref.subview" || opName == "memref.load" ||
          opName == "memref.store" || opName == "vector.insertelement" ||
          opName == "vector.transfer_read" ||
          opName == "vector.transfer_write" || opName == "vector.broadcast" ||
          opName == "vector.extractelement" || opName == "affine.apply" ||
          opName == "cf.assert" || opName == "scf.yield" ||
          opName == "scf.for") {
        return;
      }

      if (auto reductionOp = dyn_cast<mlir::vector::ReductionOp>(op)) {
        auto kind =
            reductionOp.getKind(); // This returns a vector::CombiningKind enum
        std::string formattedName = "vector.reduction<" +
                                    vector::stringifyCombiningKind(kind).str() +
                                    ">";
        opNames.push_back(formattedName);
      } else {
        opNames.push_back(opName.str());
      }
    });

    return opNames;
  }

  void runOnOperation() override {
    auto func = getOperation();

    // A declaration has no body: nothing to instrument, and the entry
    // instrumentation below dereferences the body's entry block
    // (`func.getBody().front()`), which on an empty region is the block-list
    // sentinel -- the `!isKnownSentinel()` assert, at compile time.
    // Declarations in this pipeline are not hypothetical: the HMX read-out
    // split publishes its runtime entry points as private declarations
    // (HmxVectorReadoutPass::declareRuntime), and this pass is scheduled
    // addNestedPass<func::FuncOp>, so it visits every function the module
    // holds, declarations included. Before the guard, enableLWP x
    // enableHmxVectorReadout aborted on the first such declaration on every
    // shape whose read-out the pass rewrites.
    if (func.isDeclaration())
      return;

    // The read-out split's outlined function is the executor's work function,
    // not a kernel (same call in ThreadRolePartition, same reason). Skipping
    // it is what keeps the ID space deterministic: the IDs below come from a
    // process-wide counter, the partition tooling reads "ID 1" as the kernel's
    // total row (exp/hmx/t1_partition_2026_10_08/parse_partition.py), and a
    // second instrumented function would take ID 1 whenever the pass manager
    // -- which runs sibling functions concurrently -- happened to reach it
    // first. It also keeps the read-out's own loop out of the timed path the
    // measurement is taking.
    if (func->hasAttr(mlir::hmx::kHmxReadoutOutlinedAttr))
      return;

    // The role split's outlined engine section is the same kind of
    // bystander, for the same two reasons (same call in
    // ThreadRolePartition): it is not a kernel -- it runs on the bound
    // thread, reached only through the channel's bind -- and instrumenting
    // it would corrupt the ID space the partition tooling reads. The marker
    // is the `hex.thread_role` region attribute (HmxRoleHandoff.h).
    if (func->hasAttr(mlir::hmx::kHmxThreadRoleAttr))
      return;

    // Instrument one function at a time. The module-level get-or-creates below
    // are the check-then-insert class hmxModuleStateMutex exists for (sibling
    // functions run concurrently; two racing inserts are a redefinition), and
    // the ID counter and the /tmp dump file it feeds are shared state of the
    // same kind. The lambdas run only inside this scope, so they lock nothing
    // themselves.
    std::lock_guard<std::mutex> moduleStateGuard(
        mlir::hmx::hmxModuleStateMutex());

    ModuleOp module = func->getParentOfType<ModuleOp>();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(module.getContext());

    auto i8Ty = IntegerType::get(ctx, 8);
    auto strType = LLVM::LLVMArrayType::get(i8Ty, 12);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i8PtrTy = LLVMPointerType::get(ctx);

    static int increment_loopId = 1;

    // Retrieve an existing LLVM global variable named "handler_name" or
    // create it to store the string "lwp_handler". The lookup is
    // load-bearing, not bookkeeping: this pass runs once per function, and a
    // module can hold more than one instrumentable function -- the read-out
    // era's kernel plus, in general, any helper -- so an unconditional create
    // is a redefinition of `handler_name` on the second function.
    auto getOrCreateHandlerGlobal = [&]() -> LLVM::GlobalOp {
      if (auto existing =
              module.lookupSymbol<LLVM::GlobalOp>("handler_name"))
        return existing;

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto str = builder.getStringAttr("lwp_handler\0"s);

      auto global = LLVM::GlobalOp::create(builder, module.getLoc(),
                                           /*type=*/strType,
                                           /*isConstant=*/true,
                                           /*linkage=*/LLVM::Linkage::Internal,
                                           /*name=*/"handler_name",
                                           /*value=*/str,
                                           /*alignment=*/0,
                                           /*addrSpace=*/0,
                                           /*dsoLocal=*/false,
                                           /*thread_local=*/false);

      return global;
    };

    // Retrieve or create the LLVM function 'llvm.hexagon.instrprof.custom'.
    auto getOrCreateInstrFunc = [&]() {
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
              "llvm.hexagon.instrprof.custom")) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto voidTy = LLVM::LLVMVoidType::get(ctx);
        auto funcTy = LLVM::LLVMFunctionType::get(
            voidTy, ArrayRef<Type>{i8PtrTy, i32Ty}, false);
        auto func = LLVM::LLVMFuncOp::create(
            builder, module.getLoc(), "llvm.hexagon.instrprof.custom", funcTy);
      }
    };

    // Create the call to the LLVM intrinsic
    auto emitInstrumentationCall = [&](OpBuilder builder, Location loc,
                                       auto gep, Value id) {
      auto symbolRef =
          FlatSymbolRefAttr::get(ctx, "llvm.hexagon.instrprof.custom");
      auto instr = LLVM::CallOp::create(builder, loc, TypeRange{}, symbolRef,
                                        ValueRange{gep, id});
    };

    // Get or create pointer to the string data "lwp_handler".
    LLVM::GEPOp globalGEP;
    auto getOrCreateGlobalGEP = [&](OpBuilder builder,
                                    Location loc) -> LLVM::GEPOp {
      if (globalGEP)
        return globalGEP;

      // Get the global variable "handler_name".
      auto global = getOrCreateHandlerGlobal();
      auto addr = LLVM::AddressOfOp::create(builder, loc, global);
      auto zero = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                           builder.getI32IntegerAttr(0));
      globalGEP = LLVM::GEPOp::create(builder, loc, i8PtrTy, strType, addr,
                                      ArrayRef<Value>{zero, zero});
      return globalGEP;
    };

    getOrCreateInstrFunc();

    // === Instrument Functions ===
    {
      Location loc = func.getLoc();
      std::vector<int> lines = collectLoc(loc);
      printLines(lines, increment_loopId, {});

      OpBuilder builder(func.getBody());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&func.getBody().front());

      auto gep = getOrCreateGlobalGEP(builder, loc);
      auto id = LLVM::ConstantOp::create(
          builder, loc, i32Ty, builder.getI32IntegerAttr(increment_loopId++));

      emitInstrumentationCall(builder, loc, gep, id);

      func.walk([&](func::ReturnOp retOp) {
        builder.setInsertionPoint(retOp);
        emitInstrumentationCall(builder, loc, gep, id);
      });
    }

    if (!disableLWPLoop) {
      // === Instrument Loops ===
      func.walk([&](scf::ForOp forOp) {
        int depth = getLoopDepth(forOp);

        bool shouldInstrument =
            (depth == 0) || (depth <= LWPloopDepth && hasSiblingLoop(forOp));

        if (!shouldInstrument)
          return;

        Location loc = forOp.getLoc();
        OpBuilder builder(forOp);

        std::vector<std::string> opNames = collectOps(forOp);
        std::vector<int> lines = collectLoc(loc);

        printLines(lines, increment_loopId, opNames);

        auto gep = getOrCreateGlobalGEP(builder, loc);
        auto id = LLVM::ConstantOp::create(
            builder, loc, i32Ty, builder.getI32IntegerAttr(increment_loopId++));

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(forOp);
          emitInstrumentationCall(builder, loc, gep, id);
        }

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(forOp);
          emitInstrumentationCall(builder, loc, gep, id);
        }
      });
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createHexagonLWPPass(const HexagonLWPPassOptions &options) {
  return std::make_unique<HexagonLWPPass>(options);
}
