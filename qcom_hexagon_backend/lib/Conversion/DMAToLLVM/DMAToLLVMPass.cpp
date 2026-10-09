//===- DMAToLLVMPass.cpp - Lowering DMA ops to LLVM -----------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert DMA dialect to LLVM dialect.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/DMAToLLVM/DMAExternalFnNames.h"
#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "dma-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;
using namespace mlir::hexagonmem;

#define GEN_PASS_DEF_DMATOLLVM
#include "hexagon/Conversion/DMAToLLVM/Passes.h.inc"
#undef GEN_PASS_DEF_DMATOLLVM

namespace {

// Function to get an LLVM pointer type
static LLVM::LLVMPointerType getPtrTy(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context);
}

// Function to get an LLVM void type
static LLVM::LLVMVoidType getVoidTy(MLIRContext *context) {
  return LLVM::LLVMVoidType::get(context);
}

// Function to create an LLVM constant of i32 type
static LLVM::ConstantOp getI32Constant(ConversionPatternRewriter &rewriter,
                                       Location loc, int64_t val) {
  // Get the context from the rewriter
  MLIRContext *context = rewriter.getContext();
  Type paramTy = rewriter.getI32Type();
  // Create and return an LLVM constant operation with the given value
  return LLVM::ConstantOp::create(rewriter, loc, paramTy, val);
}

// Function to truncate a value to i32 type
static Value truncToI32(ConversionPatternRewriter &rewriter, Location loc,
                        Value val) {
  // Ensure the value is of an integer type
  assert(llvm::isa<IntegerType>(val.getType()) && "Expected an integer type");
  MLIRContext *context = rewriter.getContext();
  // Create and return a truncation operation to i32 type
  return LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), val);
}

//===----------------------------------------------------------------------===//
// Lower DMAStart
//===----------------------------------------------------------------------===//

// Structure to lower DMA start operation to LLVM pattern
struct LowerDMAStart : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
  using ConvertOpToLLVMPattern<memref::DmaStartOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Function to get or create the DMA start function in the module.
// ABI (must match bin/runtime/include/RuntimeDMA.h): the final !llvm.ptr is
// `DMAStatus *status` -- an out-parameter pointing at a dedicated word, never
// the tag the returned token is stored into; a refused start returns
// `DMA_TOKEN_NONE` from this call with `*status = DMAFailure`.
static FailureOr<LLVM::LLVMFuncOp>
getDMAStartFn(ModuleOp module, StringRef fnName,
              ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {getPtrTy(context), rewriter.getI32Type(),
                                 getPtrTy(context), rewriter.getI32Type(),
                                 rewriter.getI32Type(), rewriter.getI32Type(),
                                 rewriter.getI32Type(), getPtrTy(context)},
                                rewriter.getI32Type());
}

// Function to get or create the 2D DMA start function in the module.
// Same status/token ABI as getDMAStartFn above: final !llvm.ptr = status,
// never the tag; refusal returns DMA_TOKEN_NONE.
static FailureOr<LLVM::LLVMFuncOp>
getDMA2DStartFn(ModuleOp module, StringRef fnName,
                ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(
      rewriter, module, fnName,
      {getPtrTy(context), rewriter.getI32Type(), getPtrTy(context),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
       getPtrTy(context)},
      rewriter.getI32Type());
}

// Function to convert a value to i32 type
Value convertToI32Type(mlir::Location loc, Value value,
                       ConversionPatternRewriter &rewriter,
                       mlir::MLIRContext *context) {
  // Get the type of the value
  mlir::Type type = value.getType();
  assert(mlir::LLVM::isCompatibleType(type));
  return LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), value);
}

// Function to cast a pointer to a different address space
static Value AddrSpaceCast(mlir::Location loc, Value inputPtr,
                           ConversionPatternRewriter &rewriter,
                           mlir::MLIRContext *context) {
  auto i32Type = rewriter.getI32Type();
  auto asI32 = LLVM::PtrToIntOp::create(rewriter, loc, i32Type, inputPtr);
  return LLVM::IntToPtrOp::create(rewriter, loc, getPtrTy(context),
                                  asI32->getResult(0));
}

// A dedicated i32 slot for the runtime's `DMAStatus` out-parameter of
// `hexagon_runtime_dma{,2d}_start`.
//
// Status must never share storage with the token: the token returned by the
// call is stored into the *tag* word for `dma_wait` to poll, and when this
// pass passed the tag pointer as the status pointer too, that store
// overwrote the runtime's `*status = DMAFailure` one instruction later --
// a refused transfer was reported nowhere (roadmap/ARCH-REVIEW.md bug #2;
// the runtime papered over it with a forged completed descriptor, since
// removed in favor of `DMA_TOKEN_NONE`, see bin/runtime/include/RuntimeDMA.h).
//
// One slot per start, hoisted to the function entry block: an alloca created
// where the `dma_start` sits would sit inside the double-buffering loop and
// lower to a dynamic stack allocation that grows every iteration.
static Value createDMAStatusSlot(ConversionPatternRewriter &rewriter,
                                 mlir::Location loc, Operation *op) {
  RewriterBase::InsertionGuard guard(rewriter);
  if (auto func = op->getParentOfType<FunctionOpInterface>())
    rewriter.setInsertionPointToStart(&func.getBlocks().front());
  Value count = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                         rewriter.getI64IntegerAttr(1));
  return LLVM::AllocaOp::create(rewriter, loc, getPtrTy(rewriter.getContext()),
                                rewriter.getI32Type(), count);
}

// Function to create DMAStart operation
LogicalResult
LowerDMAStart::matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  // Get the context and location of the operation
  auto context = op->getContext();
  auto loc = op->getLoc();

  // Get the parent module of the operation
  auto module = op->getParentOfType<ModuleOp>();

  // Get the name of the DMA start function
  auto dmaStartFnName = mlir::hexagon::getDMA2DStartFnName();
  FailureOr<LLVM::LLVMFuncOp> funcOp =
      getDMA2DStartFn(module, dmaStartFnName, rewriter);
  if (failed(funcOp))
    return failure();

  auto isVTCM = [](Value memref) {
    assert(llvm::isa<MemRefType>(memref.getType()) && "Unexpected type passed");
    auto memrefType = cast<MemRefType>(memref.getType());
    return isInVTCMAddressSpace(memrefType);
  };

  // Get the LLVM pointer to the memory reference
  auto getMemRefPtr = [this, loc,
                       context](Value memref, int rank, signed offset,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) {
    auto memrefType = cast<MemRefType>(memref.getType());
    auto llvmMemref = *(adaptor.getOperands().begin() + offset);
    ValueRange indices = {
        std::next(adaptor.getOperands().begin(), offset + 1),
        std::next(adaptor.getOperands().begin(), offset + rank + 1)};
    auto llvmPtr =
        getStridedElementPtr(rewriter, loc, memrefType, llvmMemref, indices);
    return llvmPtr;
  };

  // Get the source memory reference and its rank
  auto srcMemref = op.getSrcMemRef();
  auto srcRank = op.getSrcMemRefRank();
  auto tagRank = op.getTagMemRefRank();
  // Get the LLVM pointer to the source memory reference
  auto llvmSrcPtr = getMemRefPtr(srcMemref, srcRank, 0, adaptor, rewriter);

  // Get the destination memory reference and its rank
  auto dstMemref = op.getDstMemRef();
  auto dstRank = op.getDstMemRefRank();
  // Get the LLVM pointer to the destination memory reference
  auto llvmDstPtr =
      getMemRefPtr(dstMemref, dstRank, 1 + srcRank, adaptor, rewriter);

  // Get the tag memory reference and its LLVM pointer
  // This is used to synchronize between dma_start and dma_wait operation
  auto tagMemref = op.getTagMemRef();
  auto llvmTagPtr = getMemRefPtr(tagMemref, op.getTagMemRefRank(),
                                 3 + srcRank + dstRank, adaptor, rewriter);

  // Convert the number of elements to an LLVM integer type
  auto numElements = convertToI32Type(
      loc, *(adaptor.getOperands().begin() + 2 + srcRank + dstRank), rewriter,
      context);
  // Get the size of the element type in bytes
  auto typeSizeInBytes = getI32Constant(rewriter, loc,
                                        cast<MemRefType>(srcMemref.getType())
                                                .getElementType()
                                                .getIntOrFloatBitWidth() /
                                            8);
  // Calculate the transfer size
  auto transferSize =
      LLVM::MulOp::create(rewriter, loc, numElements, typeSizeInBytes);

  // Get the source and destination memory spaces
  auto srcMemSpace = getI32Constant(rewriter, loc, isVTCM(srcMemref) ? 1 : 0);
  auto dstMemSpace = getI32Constant(rewriter, loc, isVTCM(dstMemref) ? 1 : 0);
  // Set the bypass cache flag to 0
  auto bypassCache = getI32Constant(rewriter, loc, 0);
  // Status out-word: a slot of its own (createDMAStatusSlot). The tag below
  // carries the token only -- status and token must never be the same word.
  Value statusPtr = createDMAStatusSlot(rewriter, loc, op);
  SmallVector<Value, 16> operands;

  if (op.isStrided()) {
    Value width = convertToI32Type(
        loc, *(adaptor.getOperands().begin() + srcRank + dstRank + tagRank + 5),
        rewriter, context);
    Value widthInBytes =
        LLVM::MulOp::create(rewriter, loc, width, typeSizeInBytes);
    Value height =
        (LLVM::SDivOp::create(rewriter, loc, transferSize, widthInBytes))
            ->getResult(0);
    // Stride arg is optional, and if present, it is only slow address space.
    // source stride iff stride is present, and dest is faster.
    // If no stride, use stride = width
    auto srcStride =
        (isVTCM(dstMemref) && (!(isVTCM(srcMemref))))
            ? convertToI32Type(loc,
                               (*(adaptor.getOperands().begin() + srcRank +
                                  dstRank + tagRank + 4)),
                               rewriter, context)
            : width;
    auto srcStrideInBytes =
        LLVM::MulOp::create(rewriter, loc, srcStride, typeSizeInBytes);
    // dest stride iff stride is present, and src is faster.
    auto dstStride =
        (isVTCM(srcMemref) && (!(isVTCM(dstMemref))))
            ? convertToI32Type(loc,
                               (*(adaptor.getOperands().begin() + srcRank +
                                  dstRank + tagRank + 4)),
                               rewriter, context)
            : width;
    auto dstStrideInBytes =
        LLVM::MulOp::create(rewriter, loc, dstStride, typeSizeInBytes);

    // Create a vector of operands for the DMA copy operation
    operands = {llvmSrcPtr,
                srcMemSpace,
                llvmDstPtr,
                dstMemSpace,
                widthInBytes,
                height,
                srcStrideInBytes,
                dstStrideInBytes,
                bypassCache,
                bypassCache,
                getI32Constant(rewriter, loc, 0), // isOrdered
                getI32Constant(rewriter, loc, 0), // Cache Allocation Policy
                statusPtr};
  } else {
    dmaStartFnName = mlir::hexagon::getDMAStartFnName();
    funcOp = getDMAStartFn(module, dmaStartFnName, rewriter);
    operands = {llvmSrcPtr,   srcMemSpace, llvmDstPtr,  dstMemSpace,
                transferSize, bypassCache, bypassCache, statusPtr};
  }
  // Erase the original operation
  rewriter.eraseOp(op);

  // Create the DMA copy operation
  auto dmaCopyOp =
      LLVM::CallOp::create(rewriter, loc, funcOp.value(), operands);

  // Get the token from the DMA copy operation
  auto token = *dmaCopyOp.getODSResults(0).begin();

  // Store the token in the tag memory reference. The tag word holds the
  // token and nothing else; the runtime's status report lives in statusPtr,
  // so this store can no longer erase `*status = DMAFailure`.
  LLVM::StoreOp::create(rewriter, loc, token, llvmTagPtr);

  return success();
}

//===----------------------------------------------------------------------===//
// Lower DMAWait
//===----------------------------------------------------------------------===//

// Structure to lower DMA wait operation to LLVM pattern
struct LowerDMAWait : public ConvertOpToLLVMPattern<memref::DmaWaitOp> {
  // Inherit the constructor from the base class
  using ConvertOpToLLVMPattern<memref::DmaWaitOp>::ConvertOpToLLVMPattern;

  // Override the matchAndRewrite function to perform the conversion
  LogicalResult
  matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Function to get or create the DMA wait function in the module
static FailureOr<LLVM::LLVMFuncOp>
getDMAWaitFn(ModuleOp module, StringRef fnName,
             ConversionPatternRewriter &rewriter) {
  // Get the context from the module
  MLIRContext *context = module->getContext();
  // Lookup or create the DMA wait function with the specified name and types
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {rewriter.getI32Type()}, getVoidTy(context));
}

// Override the matchAndRewrite function to perform the conversion
LogicalResult
LowerDMAWait::matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  // Get the context and location of the operation
  auto context = op->getContext();
  auto loc = op->getLoc();
  // Get the parent module of the operation
  auto module = op->getParentOfType<ModuleOp>();

  // Get the name of the DMA wait function
  auto dmaWaitFnName = mlir::hexagon::getDMAWaitFnName();
  // Get the LLVM function for DMA wait
  FailureOr<LLVM::LLVMFuncOp> funcOp =
      getDMAWaitFn(module, dmaWaitFnName, rewriter);
  if (failed(funcOp))
    return failure();

  // Get the tag memory reference and its type
  auto tagMemref = op.getTagMemRef();
  auto tagMemrefType = cast<MemRefType>(tagMemref.getType());
  // Get the LLVM tag memory reference
  auto llvmTagMemref = adaptor.getTagMemRef();
  // Get the pointer to the tag memory reference
  auto llvmTagPtr = getStridedElementPtr(
      rewriter, loc, tagMemrefType, llvmTagMemref, adaptor.getTagIndices());
  // Erase the original operation
  rewriter.eraseOp(op);
  // Create a load operation to get the token
  auto tokenLoadOp =
      LLVM::LoadOp::create(rewriter, loc, rewriter.getI32Type(), llvmTagPtr);
  // Create a vector of operands for the DMA wait operation
  SmallVector<Value, 10> operands = {tokenLoadOp.getResult()};
  // Create the DMA wait operation
  auto DMAWaitOp =
      LLVM::CallOp::create(rewriter, loc, funcOp.value(), operands);
  // Return success
  return success();
}

//===----------------------------------------------------------------------===//
// Setup the Lowering Pass and patterns
//===----------------------------------------------------------------------===//

void populateDMAToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<LowerDMAStart, LowerDMAWait>(converter);
}

struct DMAToLLVMPass : public ::impl::DMAToLLVMBase<DMAToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();

    LLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);
    LowerToLLVMOptions options(context);
    LLVMTypeConverter typeConverter(context, options);

    hexagon::addTypeConversions(context, typeConverter);
    populateDMAToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexagon::createDMAToLLVMPass() {
  return std::make_unique<DMAToLLVMPass>();
}
