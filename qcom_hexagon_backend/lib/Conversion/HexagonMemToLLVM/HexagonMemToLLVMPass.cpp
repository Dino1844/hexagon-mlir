//===- HexagonMemToLLVMPass.cpp - HexagonMem to LLVM Pass -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert HexagonMem dialect to LLVM dialect.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemExternalFnNames.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemToLLVM.h"
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hexagonmem-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hexagonmem;

#define GEN_PASS_DEF_HEXAGONMEMTOLLVM
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h.inc"
#undef GEN_PASS_DEF_HEXAGONMEMTOLLVM

namespace {

/// This defines the default crouton size used for the crouton type. This needs
/// to be updated if the crouton type is modified to specific the size as part
/// of its parameters
constexpr size_t DEFAULT_CROUTON_SIZE = 2048;

static LLVM::LLVMPointerType getPtrTy(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context);
}

static LLVM::LLVMVoidType getVoidTy(MLIRContext *context) {
  return LLVM::LLVMVoidType::get(context);
}

static LLVM::ConstantOp getI32Constant(ConversionPatternRewriter &rewriter,
                                       Location loc, int64_t val) {
  Type paramTy = rewriter.getI32Type();
  return LLVM::ConstantOp::create(rewriter, loc, paramTy, val);
}

static int64_t getAllocationSize(crouton::CroutonType cTy) {
  return cTy.getNumElements();
}

static int64_t getAllocationSize(MemRefType cTy) {
  return cTy.getNumElements() * cTy.getElementTypeBitWidth() / 8;
}

static Value computeAllocationSize(MemRefType type,
                                   ConversionPatternRewriter &rewriter,
                                   MemRefDescriptor desc, Location loc,
                                   Type indexType, Value sizeInBytes) {

  if (type.hasStaticShape()) {
    auto size = getAllocationSize(type);
    return getI32Constant(rewriter, loc, size);
  }

  // Compute number of elements.
  Value numElements = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                               rewriter.getIndexAttr(1));
  for (int pos = 0; pos < type.getRank(); ++pos) {
    auto size = desc.size(rewriter, loc, pos);
    numElements = LLVM::MulOp::create(rewriter, loc, numElements, size);
  }
  Value totalSize =
      LLVM::MulOp::create(rewriter, loc, numElements, sizeInBytes);
  Value sizeI32 =
      LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), totalSize);
  return sizeI32;
}

/// the ordinary runtime allocation prototypes are:
/// void* (size_t bytes, uint64_t alignment, bool isVtcm) for memref type
/// void* (size_t numBlocks, size_t blockSize, uint64_t alignment, bool isVtcm)
/// for crouton type. Resident allocations use the versioned ABI with
/// key/source, bytes, and an i32 alignment in that order.
static FailureOr<LLVM::LLVMFuncOp>
getAllocFn(Operation *module, StringRef fnName,
           ConversionPatternRewriter &rewriter, bool isCroutonType) {
  MLIRContext *context = module->getContext();
  FailureOr<LLVM::LLVMFuncOp> funcOp;

  if (isCroutonType)
    funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, fnName,
                               {rewriter.getI32Type(), rewriter.getI32Type(),
                                rewriter.getI64Type(), rewriter.getI1Type()},
                               getPtrTy(context));
  else
    funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, fnName,
        {rewriter.getI32Type(), rewriter.getI64Type(), rewriter.getI1Type()},
        getPtrTy(context));

  if (succeeded(funcOp)) {
    // Mark function as having side effects to prevent LLVM optimizer
    // from removing or inlining these calls inappropriately.
    (*funcOp)->setAttr(
        "passthrough",
        rewriter.getArrayAttr({rewriter.getStringAttr("noinline"),
                               rewriter.getStringAttr("willreturn")}));
  }

  return funcOp;
}

/// Both memref and crouton type dealloc just need the pointer
/// void(void *ptr) for memref type
static FailureOr<LLVM::LLVMFuncOp>
getDeallocFn(ModuleOp module, StringRef fnName,
             ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  auto funcOp = LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                       {getPtrTy(context)}, getVoidTy(context));

  if (succeeded(funcOp)) {
    // Mark function as having side effects to prevent LLVM O3 optimizer
    // from removing dealloc calls as dead code.
    // Without these attributes, LLVM sees the function as having no side
    // effects and removes all calls during optimization.
    (*funcOp)->setAttr(
        "passthrough",
        rewriter.getArrayAttr({rewriter.getStringAttr("noinline"),
                               rewriter.getStringAttr("willreturn")}));
  }

  return funcOp;
}

/// Common code to determine the type used and it's memory space
template <typename AllocDeallocOp>
std::tuple<LogicalResult, bool, bool>
computeTypeInfo(AllocDeallocOp op, ConversionPatternRewriter &rewriter,
                bool isAlloc) {
  auto type = op.getBuffer().getType();
  bool isInVtcm = false;
  bool isCroutonType = true;
  if (auto croutonType = mlir::dyn_cast<crouton::CroutonType>(type)) {
    isInVtcm = croutonType.getVtcm().getValue();
  } else if (auto memrefType = mlir::dyn_cast<MemRefType>(type)) {
    isCroutonType = false;
    isInVtcm = hexagon::isInVTCMAddressSpace(memrefType);
  } else {
    llvm::errs() << "Invalid type passed to hexagonmem.alloc\n";
    return {failure(), isInVtcm, isCroutonType};
  }

  return {success(), isInVtcm, isCroutonType};
}

/// The per-site scope carried by one pool-backed allocation, read from the
/// `hmx.vtcm_site_scope` stamp the accounting pass attached.
///
/// Every field is validated before a single op is created. A stamp that is
/// present but malformed is a refusal, not a default: emitting a bracket with a
/// zeroed word would attribute the allocation to a site nobody named, and
/// skipping the bracket would silently shrink the scope the table published.
struct SiteScopeBracket {
  bool active{false};
  uint64_t tokenLow{0};
  uint64_t tokenHigh{0};
  uint64_t accountingScopeId{0};
  uint64_t invocationId{0};
  uint64_t functionId{0};
  uint64_t allocationSiteId{0};
  uint64_t buildIdLow{0};
  uint64_t buildIdHigh{0};
};

/// Read a stamped 64-bit word. MLIR prints `i64` in signed decimal, so the
/// stored pattern is normalized to its unsigned value before it becomes a
/// runtime constant; the two must be the same number on the device.
static bool readSiteScopeWord(DictionaryAttr dict, StringRef name,
                              uint64_t &value) {
  Attribute attr = dict.get(name);
  if (auto pair = dyn_cast_or_null<DictionaryAttr>(attr))
    attr = pair.get("value");
  auto integer = dyn_cast_or_null<IntegerAttr>(attr);
  if (!integer || !integer.getType().isSignlessInteger(64))
    return false;
  value = integer.getValue().getZExtValue();
  return true;
}

static FailureOr<SiteScopeBracket>
readSiteScope(Operation *op, bool isPoolBacked) {
  SiteScopeBracket bracket;
  Attribute raw = op->getAttr(mlir::hmx::kHmxVtcmSiteScopeAttr);
  if (!raw)
    return bracket;
  auto stamp = dyn_cast<DictionaryAttr>(raw);
  if (!stamp)
    return failure();
  // The per-op stamp has exactly one closed shape: the producer only stamps
  // when the module-level table is eligible, so there is no "ineligible stamp"
  // to accept here. A stamp of any other shape -- missing a word, carrying an
  // extra one, or holding the table's `reason` in place of an identity -- is a
  // refusal, because each of those would have to be defaulted or ignored before
  // a bracket could be built from it.
  if (!mlir::hmx::hasExactHmxDiagnosticSiteScopeKeys(
          stamp, mlir::hmx::hmxDiagnosticSiteScopeKeys())) {
    op->emitError("hmx.vtcm_site_scope is not the closed per-site stamp shape");
    return failure();
  }
  // The stamp is only meaningful for a pool-backed allocation. A DDR allocation
  // never reaches the VTCM pool, so bracketing one would name a site the
  // runtime never saw.
  if (!isPoolBacked) {
    op->emitError("hmx.vtcm_site_scope is attached to a non-pool-backed "
                  "allocation");
    return failure();
  }
  auto kind = stamp.getAs<StringAttr>("kind");
  auto schema = stamp.getAs<StringAttr>("schema");
  auto status = stamp.getAs<StringAttr>("status");
  auto tokenBasis = stamp.getAs<StringAttr>("token_basis");
  auto tokenBits = stamp.getAs<IntegerAttr>("token_bits");
  auto role = stamp.getAs<StringAttr>("role");
  auto source = stamp.getAs<StringAttr>("source");
  auto function = stamp.getAs<StringAttr>("function");
  auto gridProduct = stamp.getAs<IntegerAttr>("grid_product");
  auto invocationId = stamp.getAs<IntegerAttr>("invocation_id");
  if (!kind || kind.getValue() != "vtcm-site-scope" || !schema ||
      schema.getValue() != mlir::hmx::kHmxVtcmSiteScopesSchema || !status ||
      status.getValue() != "complete" || !tokenBasis ||
      tokenBasis.getValue() != mlir::hmx::kHmxDiagnosticSiteTokenSchema ||
      !tokenBits || !tokenBits.getType().isSignlessInteger(64) ||
      tokenBits.getInt() != 128 || !role || role.getValue().empty() || !source ||
      source.getValue().empty() || !function || function.getValue().empty() ||
      !gridProduct || !gridProduct.getType().isSignlessInteger(64) ||
      gridProduct.getInt() != 1 || !invocationId ||
      !invocationId.getType().isSignlessInteger(64) ||
      invocationId.getInt() != 1 ||
      !readSiteScopeWord(stamp, "function_id", bracket.functionId) ||
      !readSiteScopeWord(stamp, "site_id", bracket.allocationSiteId) ||
      !readSiteScopeWord(stamp, "accounting_scope_id",
                         bracket.accountingScopeId) ||
      !readSiteScopeWord(stamp, "token_low", bracket.tokenLow) ||
      !readSiteScopeWord(stamp, "token_high", bracket.tokenHigh) ||
      !readSiteScopeWord(stamp, "build_id_low", bracket.buildIdLow) ||
      !readSiteScopeWord(stamp, "build_id_high", bracket.buildIdHigh)) {
    op->emitError("hmx.vtcm_site_scope is missing a required word or carries a "
                  "value outside the site ABI");
    return failure();
  }
  // The invocation ordinal is a scope fact rather than an identity, so it is
  // validated from the typed attribute instead of through readSiteScopeWord. It
  // still has to be stored: the zero check below is what refuses a scope with
  // no invocation to spend, and it reads this field.
  bracket.invocationId = static_cast<uint64_t>(invocationId.getInt());
  if ((bracket.tokenLow == 0 && bracket.tokenHigh == 0) ||
      bracket.accountingScopeId == 0 || bracket.functionId == 0 ||
      bracket.allocationSiteId == 0 || bracket.invocationId == 0 ||
      (bracket.buildIdLow == 0 && bracket.buildIdHigh == 0)) {
    op->emitError("hmx.vtcm_site_scope carries a zero identity the site ABI "
                  "refuses to bind");
    return failure();
  }
  bracket.active = true;
  return bracket;
}

/// Create the enter call that opens the bracket. It is emitted immediately
/// before the pool allocation, at the allocation's own insertion point, so the
/// runtime owner is current for exactly that call.
static FailureOr<LLVM::LLVMFuncOp>
getSiteScopeEnterFn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module.getContext();
  SmallVector<Type> params{rewriter.getI32Type(), rewriter.getI32Type()};
  // token low, token high, accounting scope, invocation, function, site, build
  // low, build high: eight 64-bit words between the two 32-bit fields.
  for (unsigned index = 0; index != 8; ++index)
    params.push_back(rewriter.getI64Type());
  params.push_back(rewriter.getI32Type());
  FailureOr<LLVM::LLVMFuncOp> fn = LLVM::lookupOrCreateFn(
      rewriter, module, mlir::hmx::kHmxDiagnosticSiteScopeEnterFn, params,
      getVoidTy(context));
  if (succeeded(fn))
    // Side-effecting and opaque, so neither the CSE pass nor a later
    // optimization may merge two brackets or drop one. The bracket is evidence:
    // losing it would leave an allocation attributed to nothing.
    (*fn)->setAttr("passthrough",
                   rewriter.getArrayAttr({rewriter.getStringAttr("noinline"),
                                          rewriter.getStringAttr("willreturn")}));
  return fn;
}

static FailureOr<LLVM::LLVMFuncOp>
getSiteScopeLeaveFn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  FailureOr<LLVM::LLVMFuncOp> fn = LLVM::lookupOrCreateFn(
      rewriter, module, mlir::hmx::kHmxDiagnosticSiteScopeLeaveFn,
      ArrayRef<Type>{}, getVoidTy(module.getContext()));
  if (succeeded(fn))
    (*fn)->setAttr("passthrough",
                   rewriter.getArrayAttr({rewriter.getStringAttr("noinline"),
                                          rewriter.getStringAttr("willreturn")}));
  return fn;
}

static Value siteScopeI32(ConversionPatternRewriter &rewriter, Location loc,
                          uint32_t value) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  static_cast<int32_t>(value));
}

static Value siteScopeI64(ConversionPatternRewriter &rewriter, Location loc,
                          uint64_t value) {
  return LLVM::ConstantOp::create(
      rewriter, loc, rewriter.getI64Type(),
      IntegerAttr::get(rewriter.getI64Type(), APInt(64, value)));
}

/// Emit the enter half. No-op when the allocation is not stamped, which is the
/// default path: production IR carries no stamp and gets no bracket.
static LogicalResult
emitSiteScopeEnter(ConversionPatternRewriter &rewriter, hexagonmem::AllocOp op,
                   const SiteScopeBracket &bracket) {
  if (!bracket.active)
    return success();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  FailureOr<LLVM::LLVMFuncOp> fn = getSiteScopeEnterFn(module, rewriter);
  if (failed(fn))
    return failure();
  Location loc = op->getLoc();
  SmallVector<Value> operands{
      siteScopeI32(rewriter, loc, mlir::hmx::kHmxDiagnosticSiteScopeAbiVersion),
      siteScopeI32(rewriter, loc,
                   mlir::hmx::kHmxDiagnosticSiteScopeFlagSingleInvocation |
                       mlir::hmx::kHmxDiagnosticSiteScopeFlagGridOne),
      siteScopeI64(rewriter, loc, bracket.tokenLow),
      siteScopeI64(rewriter, loc, bracket.tokenHigh),
      siteScopeI64(rewriter, loc, bracket.accountingScopeId),
      siteScopeI64(rewriter, loc, bracket.invocationId),
      siteScopeI64(rewriter, loc, bracket.functionId),
      siteScopeI64(rewriter, loc, bracket.allocationSiteId),
      siteScopeI64(rewriter, loc, bracket.buildIdLow),
      siteScopeI64(rewriter, loc, bracket.buildIdHigh),
      siteScopeI32(rewriter, loc, 1)};
  LLVM::CallOp::create(rewriter, loc, TypeRange{},
                       FlatSymbolRefAttr::get(fn->getOperation()), operands);
  return success();
}

/// Emit the leave half immediately after the allocation call, so the bracket
/// closes on the allocation it names and the free -- which happens later and is
/// attributed through the owner the runtime retained -- still sees the site.
static LogicalResult
emitSiteScopeLeave(ConversionPatternRewriter &rewriter, hexagonmem::AllocOp op,
                   Operation *after) {
  auto bracket = readSiteScope(op, /*isPoolBacked=*/true);
  if (failed(bracket) || !bracket->active)
    return success();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  FailureOr<LLVM::LLVMFuncOp> fn = getSiteScopeLeaveFn(module, rewriter);
  if (failed(fn))
    return failure();
  rewriter.setInsertionPointAfter(after);
  LLVM::CallOp::create(rewriter, op->getLoc(), TypeRange{},
                       FlatSymbolRefAttr::get(fn->getOperation()),
                       ValueRange{});
  return success();
}

//===----------------------------------------------------------------------===//
// Lower hexagonmem::AllocOp
//===----------------------------------------------------------------------===//

struct LowerAlloc : public ConvertOpToLLVMPattern<hexagonmem::AllocOp> {
  using ConvertOpToLLVMPattern<hexagonmem::AllocOp>::ConvertOpToLLVMPattern;

private:
  std::string deviceType;

public:
  explicit LowerAlloc(LLVMTypeConverter &converter, const std::string &devType)
      : ConvertOpToLLVMPattern<hexagonmem::AllocOp>(converter),
        deviceType(devType) {}

  /// Lower `hexagonmem.alloc` to `llvm.call @hexagon_runtime_alloc_1d/2d` call
  LogicalResult matchAndRewrite(hexagonmem::AllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto [result, isInVtcm, isCroutonType] =
        computeTypeInfo<hexagonmem::AllocOp>(op, rewriter,
                                             /* isAlloc */ true);

    if (failed(result))
      return result;

    auto loc = op->getLoc();
    auto type = op.getBuffer().getType();

    // Replace "hexagonmem.alloc" with "memref.alloc" for flat DDR allocations
    if (!isInVtcm && !isCroutonType) {
      // A stamp on a DDR allocation is a producer disagreement, not a no-op: the
      // site table only names pool-backed sites, so a stamp here means the two
      // disagree about what this operation allocates.
      if (op->hasAttr(mlir::hmx::kHmxVtcmSiteScopeAttr)) {
        op.emitError("hmx.vtcm_site_scope is attached to a DDR allocation, "
                     "which never reaches the VTCM pool");
        return failure();
      }
      rewriter.replaceOpWithNewOp<memref::AllocOp>(
          op, mlir::cast<MemRefType>(type));
      return success();
    }

    // Read and fully validate the per-site stamp before creating anything, so a
    // malformed stamp cannot leave a half-emitted bracket behind.
    FailureOr<SiteScopeBracket> bracket = readSiteScope(op, isInVtcm);
    if (failed(bracket))
      return failure();

    // Keep the requested alignment on every VTCM allocation ABI.  The ordinary
    // allocator already accepts it, while resident entries use a distinct
    // versioned ABI so a 256-byte workspace cannot silently become a 128-byte
    // buffer on its way through the resident lookup.
    const uint64_t alignment = op.getAlignment();

    // A resident workspace skips the per-launch allocator entirely: the runtime
    // allocates the buffer on the first launch, pins it (the per-launch
    // deallocation was dropped by the marking pass), and returns the same
    // address forever. Unlike the weight path there is no source: the kernel
    // owns the contents and refills the buffer every launch, so the key alone
    // identifies the resident. Only the call and the lookup remain in the
    // prologue.
    if (auto workspace =
            op->getAttrOfType<DictionaryAttr>(mlir::hmx::kHmxWorkspaceResidentAttr)) {
      auto keyAttr = workspace.getAs<IntegerAttr>("key");
      auto bytesAttr = workspace.getAs<IntegerAttr>("bytes");
      if (!keyAttr || !bytesAttr) {
        op.emitError("resident workspace is missing its key or byte count");
        return failure();
      }
      // The resident's instance discriminator: the flat program id of this
      // launch, computed exactly the way the wrapper does (pid_X * np_Y * np_Z
      // + pid_Y * np_Z + pid_Z, triton_hexagon_launcher.py's grid_strides).
      // Concurrent instances of a grid>1 launch carry distinct pids, so each
      // gets its own resident buffer; the same pid across launches reuses the
      // same buffer. A thread id would be wrong here: the wrapper's
      // ThreadManager spawns fresh qurt threads per launch ("keep the thread
      // pool alive" is still a TODO in multithreading.h), so a thread-keyed
      // residency would allocate a never-reused buffer set every launch and
      // grow the resident map without bound (measured: mha_fa grid=4, +73%).
      // IR without the trailing program-info pack (direct pass invocations,
      // lit tests) is single-instance by construction: instance 0.
      auto func = op->getParentOfType<func::FuncOp>();
      Value instanceValue = getI32Constant(rewriter, loc, 0);
      if (func && func.getNumArguments() >= 6) {
        unsigned n = func.getNumArguments();
        bool pack = true;
        for (unsigned i = n - 6; i < n; ++i) {
          auto ty = dyn_cast<IntegerType>(func.getArgument(i).getType());
          if (!ty || ty.getWidth() != 32) {
            pack = false;
            break;
          }
        }
        if (pack) {
          Value npY = func.getArgument(n - 5), npZ = func.getArgument(n - 4);
          Value pidX = func.getArgument(n - 3), pidY = func.getArgument(n - 2),
                 pidZ = func.getArgument(n - 1);
          Value yz = arith::MulIOp::create(rewriter, loc, npY, npZ);
          Value xTerm = arith::MulIOp::create(rewriter, loc, pidX, yz);
          Value yTerm = arith::MulIOp::create(rewriter, loc, pidY, npZ);
          Value xy = arith::AddIOp::create(rewriter, loc, xTerm, yTerm);
          instanceValue = arith::AddIOp::create(rewriter, loc, xy, pidZ);
        }
      }
      FailureOr<LLVM::LLVMFuncOp> residentFn = LLVM::lookupOrCreateFn(
          rewriter, op->getParentOfType<ModuleOp>(),
          "hexagon_runtime_workspace_resident_v2_dsp",
          {rewriter.getI64Type(), rewriter.getI32Type(), rewriter.getI32Type(),
           rewriter.getI32Type()},
          getPtrTy(rewriter.getContext()));
      if (failed(residentFn))
        return failure();
      (*residentFn)->setAttr(
          "passthrough",
          rewriter.getArrayAttr({rewriter.getStringAttr("noinline"),
                                 rewriter.getStringAttr("willreturn")}));
      Value keyValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(keyAttr.getInt()));
      Value bytesValue = getI32Constant(rewriter, loc, bytesAttr.getInt());
      Value residentAlignmentValue =
          getI32Constant(rewriter, loc, static_cast<int64_t>(alignment));
      if (failed(emitSiteScopeEnter(rewriter, op, *bracket)))
        return failure();
      auto callOp = LLVM::CallOp::create(
          rewriter, loc, residentFn.value(),
          ValueRange({keyValue, bytesValue, residentAlignmentValue,
                      instanceValue}));
      if (failed(emitSiteScopeLeave(rewriter, op, callOp)))
        return failure();

      auto origMemRefType = mlir::cast<MemRefType>(type);
      auto memRefType = mlir::affine::normalizeMemRefType(origMemRefType);
      SmallVector<Value, 4> sizes;
      SmallVector<Value, 4> strides;
      Value size;
      // The resident buffer is static; there is no shape operand.
      this->getMemRefDescriptorSizes(loc, memRefType, ValueRange{}, rewriter,
                                     sizes, strides, size,
                                     /* sizeInBytes */ true);
      auto memRefDescriptor = this->createMemRefDescriptor(
          loc, memRefType, callOp.getResult(), callOp.getResult(), sizes,
          strides, rewriter);
      rewriter.replaceOp(op, {memRefDescriptor});
      return success();
    }

    // A resident weight is not allocated per launch: the runtime hands back the
    // same pinned VTCM buffer every time and fills it on the first call only.
    // The kernel-side alloc/free disappears entirely (the free is still emitted
    // and swallowed by the pool), so the only per-launch cost left is the call.
    // Two sources: the address of a prepacked compile-time global, or the data
    // pointer of the runtime function argument the host has pre-packed.
    if (auto resident = op->getAttrOfType<DictionaryAttr>(mlir::hmx::kHmxWeightResidentAttr)) {
      auto bytesAttr = resident.getAs<IntegerAttr>("bytes");
      if (!bytesAttr) {
        op.emitError("resident weight is missing its byte count");
        return failure();
      }
      Type srcType = rewriter.getI64Type();
      Value src;
      if (auto globalRef = resident.getAs<FlatSymbolRefAttr>("global")) {
        Operation *symbolTable = op->getParentWithTrait<OpTrait::SymbolTable>();
        Operation *global = SymbolTable::lookupSymbolIn(symbolTable, globalRef);
        if (auto memrefGlobal = dyn_cast_or_null<memref::GlobalOp>(global)) {
          // Still a memref.global here: finalize-memref-to-llvm lowers the pair
          // into llvm.mlir.addressof + llvm.ptrtoint after this pass.
          Value loaded = memref::GetGlobalOp::create(rewriter, loc,
                                                      memrefGlobal.getType(),
                                                      globalRef.getValue());
          Value asIndex =
              memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, loaded);
          src = arith::IndexCastUIOp::create(rewriter, loc, srcType, asIndex);
        } else if (auto llvmGlobal = dyn_cast_or_null<LLVM::GlobalOp>(global)) {
          Value address = LLVM::AddressOfOp::create(
              rewriter, loc, getPtrTy(rewriter.getContext()),
              llvmGlobal.getSymName());
          src = LLVM::PtrToIntOp::create(rewriter, loc, srcType, address);
        } else {
          op.emitError("resident weight source is not a global: ") << globalRef;
          return failure();
        }
      } else if (resident.get("address")) {
        // The runtime weight's residency key: the source argument's aligned
        // pointer, carried as the alloc's extra operand. It has already crossed
        // index -> i64 by the time this conversion runs.
        if (adaptor.getOperands().empty()) {
          op.emitError("resident weight is missing its source address operand");
          return failure();
        }
        Value address = adaptor.getOperands().front();
        if (address.getType().isIndex())
          address =
              arith::IndexCastUIOp::create(rewriter, loc, srcType, address);
        src = address;
      } else {
        op.emitError("resident weight is missing its source");
        return failure();
      }

      FailureOr<LLVM::LLVMFuncOp> residentFn = LLVM::lookupOrCreateFn(
          rewriter, op->getParentOfType<ModuleOp>(),
          "hexagon_runtime_weight_resident_v2_dsp",
          {srcType, rewriter.getI32Type(), rewriter.getI32Type()},
          getPtrTy(rewriter.getContext()));
      if (failed(residentFn))
        return failure();
      (*residentFn)->setAttr(
          "passthrough",
          rewriter.getArrayAttr({rewriter.getStringAttr("noinline"),
                                 rewriter.getStringAttr("willreturn")}));
      Value bytesValue = getI32Constant(rewriter, loc, bytesAttr.getInt());
      Value residentAlignmentValue =
          getI32Constant(rewriter, loc, static_cast<int64_t>(alignment));
      if (failed(emitSiteScopeEnter(rewriter, op, *bracket)))
        return failure();
      auto callOp = LLVM::CallOp::create(
          rewriter, loc, residentFn.value(),
          ValueRange({src, bytesValue, residentAlignmentValue}));
      if (failed(emitSiteScopeLeave(rewriter, op, callOp)))
        return failure();

      auto origMemRefType = mlir::cast<MemRefType>(type);
      auto memRefType = mlir::affine::normalizeMemRefType(origMemRefType);
      SmallVector<Value, 4> sizes;
      SmallVector<Value, 4> strides;
      Value size;
      // The resident buffer is static; its operands (the source address) are not
      // shape. Build the descriptor from the type alone.
      this->getMemRefDescriptorSizes(loc, memRefType, ValueRange{}, rewriter,
                                     sizes, strides, size,
                                     /* sizeInBytes */ true);
      auto memRefDescriptor = this->createMemRefDescriptor(
          loc, memRefType, callOp.getResult(), callOp.getResult(), sizes,
          strides, rewriter);
      rewriter.replaceOp(op, {memRefDescriptor});
      return success();
    }

    Value alignmentValue = createIndexAttrConstant(
        rewriter, loc, rewriter.getI64Type(), static_cast<int64_t>(alignment));
    auto allocFnName = getAllocFnName(isCroutonType, deviceType);
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        getAllocFn(op->getParentWithTrait<OpTrait::SymbolTable>(), allocFnName,
                   rewriter, isCroutonType);
    if (failed(funcOp))
      return failure();

    Value isInVtcmValue =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), isInVtcm);

    if (isCroutonType) {
      crouton::CroutonType croutonType = mlir::cast<crouton::CroutonType>(type);
      Value size =
          getI32Constant(rewriter, loc, getAllocationSize(croutonType));
      Value blockSizeValue =
          getI32Constant(rewriter, loc, DEFAULT_CROUTON_SIZE);
      if (failed(emitSiteScopeEnter(rewriter, op, *bracket)))
        return failure();
      auto croutonCall = LLVM::CallOp::create(
          rewriter, loc, funcOp.value(),
          ValueRange({size, blockSizeValue, alignmentValue, isInVtcmValue}));
      if (failed(emitSiteScopeLeave(rewriter, op, croutonCall)))
        return failure();
      rewriter.replaceOp(op, croutonCall->getResults());
    } else {
      auto origMemRefType = mlir::cast<MemRefType>(type);
      auto memRefType = mlir::affine::normalizeMemRefType(origMemRefType);
      Value size;

      // Get actual sizes of the memref as values: static sizes are constant
      // values and dynamic sizes are passed to 'alloc' as operands.  In case of
      // zero-dimensional memref, assume a scalar (size 1).
      SmallVector<Value, 4> sizes;
      SmallVector<Value, 4> strides;
      Value sizeAsI32;
      this->getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(),
                                     rewriter, sizes, strides, size,
                                     /* sizeInBytes */ true);
      sizeAsI32 = LLVM::TruncOp::create(rewriter, loc,
                                        rewriter.getIntegerType(32), size);
      if (failed(emitSiteScopeEnter(rewriter, op, *bracket)))
        return failure();
      mlir::LLVM::CallOp callOp = LLVM::CallOp::create(
          rewriter, loc, funcOp.value(),
          ValueRange({sizeAsI32, alignmentValue, isInVtcmValue}));
      if (failed(emitSiteScopeLeave(rewriter, op, callOp)))
        return failure();
      // The runtime pointer-returning ABI enforces a non-null result contract
      // (allocation failure aborts before this point). Do not build a descriptor
      // from an unchecked null result.
      auto memRefDescriptor = this->createMemRefDescriptor(
          loc, memRefType, callOp.getResult(), callOp.getResult(), sizes,
          strides, rewriter);
      rewriter.replaceOp(op, {memRefDescriptor});
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexagonmem::DeallocOp
//===----------------------------------------------------------------------===//

struct LowerDealloc : public ConvertOpToLLVMPattern<hexagonmem::DeallocOp> {
  using ConvertOpToLLVMPattern<hexagonmem::DeallocOp>::ConvertOpToLLVMPattern;

private:
  std::string deviceType;

public:
  explicit LowerDealloc(LLVMTypeConverter &converter,
                        const std::string &devType)
      : ConvertOpToLLVMPattern<hexagonmem::DeallocOp>(converter),
        deviceType(devType) {}
  /// Lower `hexagonmem.dealloc` to `llvm.call @hexagon_runtime_free_1d/2d` call
  LogicalResult matchAndRewrite(hexagonmem::DeallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto [result, isInVtcm, isCroutonType] =
        computeTypeInfo<hexagonmem::DeallocOp>(op, rewriter,
                                               /* isAlloc */ false);

    if (failed(result))
      return result;

    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Replace "hexagonmem.dealloc" with "memref.dealloc" for flat DDR
    // allocations
    if (!isInVtcm && !isCroutonType) {
      rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, op.getBuffer());
      return success();
    }

    auto deallocFnName = getDeallocFnName(isCroutonType, deviceType);
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        getDeallocFn(module, deallocFnName, rewriter);
    if (failed(funcOp))
      return failure();

    auto bufferPtr = adaptor.getBuffer();
    if (!isCroutonType) {
      MemRefDescriptor bufferDesc(adaptor.getBuffer());
      bufferPtr = bufferDesc.alignedPtr(rewriter, loc);
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp.value(),
                                              ValueRange({bufferPtr}));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexagonmem::CopyOp
//===----------------------------------------------------------------------===//

static FailureOr<LLVM::LLVMFuncOp>
getCopyFn(ModuleOp module, StringRef fnName,
          ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {getPtrTy(context), getPtrTy(context),
                                 rewriter.getI32Type(), rewriter.getI1Type(),
                                 rewriter.getI1Type()},
                                getVoidTy(context));
}

struct LowerCopy : public ConvertOpToLLVMPattern<hexagonmem::CopyOp> {
  using ConvertOpToLLVMPattern<hexagonmem::CopyOp>::ConvertOpToLLVMPattern;

private:
  std::string deviceType;

public:
  explicit LowerCopy(LLVMTypeConverter &converter, const std::string &devType)
      : ConvertOpToLLVMPattern<hexagonmem::CopyOp>(converter),
        deviceType(devType) {}

  // This method is an exact replica of the one in MemRefToLLVM lowering.
  // TODO: Expose the upstream method and use that instead of a local copy
  LogicalResult
  lowerToMemCopyFunctionCall(hexagonmem::CopyOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcType = cast<BaseMemRefType>(op.getSource().getType());
    auto targetType = cast<BaseMemRefType>(op.getTarget().getType());

    // First make sure we have an unranked memref descriptor representation.
    auto makeUnranked = [&, this](Value ranked, MemRefType type) {
      auto rank = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                           type.getRank());
      auto *typeConverter = getTypeConverter();
      auto ptr =
          typeConverter->promoteOneMemRefDescriptor(loc, ranked, rewriter);

      auto unrankedType =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      return UnrankedMemRefDescriptor::pack(
          rewriter, loc, *typeConverter, unrankedType, ValueRange{rank, ptr});
    };

    // Save stack position before promoting descriptors
    auto stackSaveOp = LLVM::StackSaveOp::create(rewriter, loc, getPtrType());

    auto srcMemRefType = dyn_cast<MemRefType>(srcType);
    Value unrankedSource =
        srcMemRefType ? makeUnranked(adaptor.getSource(), srcMemRefType)
                      : adaptor.getSource();
    auto targetMemRefType = dyn_cast<MemRefType>(targetType);
    Value unrankedTarget =
        targetMemRefType ? makeUnranked(adaptor.getTarget(), targetMemRefType)
                         : adaptor.getTarget();

    // Now promote the unranked descriptors to the stack.
    auto one = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                        rewriter.getIndexAttr(1));
    auto promote = [&](Value desc) {
      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      auto allocated =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, desc.getType(), one);
      LLVM::StoreOp::create(rewriter, loc, desc, allocated);
      return allocated;
    };

    auto sourcePtr = promote(unrankedSource);
    auto targetPtr = promote(unrankedTarget);

    // Derive size from llvm.getelementptr which will account for any
    // potential alignment
    auto elemSize = getSizeInBytes(loc, srcType.getElementType(), rewriter);
    auto copyFn = LLVM::lookupOrCreateMemRefCopyFn(
        rewriter, op->getParentOfType<ModuleOp>(), getIndexType(),
        sourcePtr.getType());
    if (failed(copyFn))
      return failure();
    LLVM::CallOp::create(rewriter, loc, copyFn.value(),
                         ValueRange{elemSize, sourcePtr, targetPtr});

    // Restore stack used for descriptors
    LLVM::StackRestoreOp::create(rewriter, loc, stackSaveOp);

    rewriter.eraseOp(op);

    return success();
  }

  /// Lower `hexagonmem.` to `llvm.call @hexagon_runtime_copy` call
  LogicalResult matchAndRewrite(hexagonmem::CopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    auto sourceType = op.getSource().getType();
    auto targetType = op.getTarget().getType();

    bool isCroutonType;

    // Verify that the source and targe types match
    if (mlir::isa<crouton::CroutonType>(sourceType) &&
        mlir::isa<crouton::CroutonType>(targetType)) {
      isCroutonType = true;
    } else if (mlir::isa<MemRefType>(sourceType) &&
               mlir::isa<MemRefType>(targetType)) {
      isCroutonType = false;
    } else {
      llvm::errs() << "hexagonmem.copy op expects the source and target types "
                      "to match\n";
      return failure();
    }

    auto copyFnName = getCopyFnName(deviceType);
    // TODO: Cleanup the common code for crouton/memref types and make the
    // function smaller
    if (isCroutonType) {
      auto srcCroutonType =
          mlir::cast<crouton::CroutonType>(op.getSource().getType());
      auto tgtCroutonType =
          mlir::cast<crouton::CroutonType>(op.getTarget().getType());
      bool sourceIsVTCM = srcCroutonType.getVtcm().getValue();
      bool targetIsVTCM = tgtCroutonType.getVtcm().getValue();
      int64_t copySize = getAllocationSize(srcCroutonType);
      assert(getAllocationSize(tgtCroutonType) == copySize &&
             "The crouton sizes don't match");

      FailureOr<LLVM::LLVMFuncOp> funcOp =
          getCopyFn(module, copyFnName, rewriter);
      if (failed(funcOp))
        return failure();

      Value copySizeValue =
          getI32Constant(rewriter, loc, copySize * DEFAULT_CROUTON_SIZE);
      Value sourceIsVTCMValue = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI1Type(), sourceIsVTCM);
      Value targetIsVTCMValue = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI1Type(), targetIsVTCM);
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, funcOp.value(),
          ValueRange({adaptor.getTarget(), adaptor.getSource(), copySizeValue,
                      targetIsVTCMValue, sourceIsVTCMValue}));

    } else {
      auto srcMemrefType = mlir::cast<MemRefType>(op.getSource().getType());
      auto tgtMemrefType = mlir::cast<MemRefType>(op.getTarget().getType());

      bool sourceIsVTCM = hexagon::isInVTCMAddressSpace(srcMemrefType);
      bool targetIsVTCM = hexagon::isInVTCMAddressSpace(tgtMemrefType);

      if (!sourceIsVTCM && !targetIsVTCM) {
        rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSource(),
                                                    op.getTarget());
        return success();
      }

      if (!hexagon::isContiguousMemrefType(srcMemrefType) ||
          !hexagon::isContiguousMemrefType(tgtMemrefType)) {
        return lowerToMemCopyFunctionCall(op, adaptor, rewriter);
      }

      FailureOr<LLVM::LLVMFuncOp> funcOp =
          getCopyFn(module, copyFnName, rewriter);
      if (failed(funcOp))
        return failure();

      auto getDescAndPtr =
          [&](MemRefType memRefType,
              Value value) -> std::tuple<MemRefDescriptor, Value> {
        Type elementType =
            typeConverter->convertType(memRefType.getElementType());
        MemRefDescriptor desc(value);
        Value basePtr = desc.alignedPtr(rewriter, loc);
        Value offset = desc.offset(rewriter, loc);
        return {desc, LLVM::GEPOp::create(rewriter, loc, basePtr.getType(),
                                          elementType, basePtr, offset)};
      };

      auto [sourceDesc, sourcePtr] =
          getDescAndPtr(srcMemrefType, adaptor.getSource());
      auto [targetDesc, targetPtr] =
          getDescAndPtr(tgtMemrefType, adaptor.getTarget());

      // TODO: Check if it's necessary to compare allocation sizes and is it
      // okay to do so for dynamic shapes as well
      Value elementSizeInBytes =
          getSizeInBytes(loc, srcMemrefType.getElementType(), rewriter);
      Value copySize =
          computeAllocationSize(srcMemrefType, rewriter, sourceDesc, loc,
                                getIndexType(), elementSizeInBytes);

      Value sourceIsVTCMValue = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI1Type(), sourceIsVTCM);
      Value targetIsVTCMValue = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI1Type(), targetIsVTCM);
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, funcOp.value(),
          ValueRange({targetPtr, sourcePtr, copySize, targetIsVTCMValue,
                      sourceIsVTCMValue}));
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexagonmem::MemrefToCroutonOp
//===----------------------------------------------------------------------===//

static FailureOr<LLVM::LLVMFuncOp>
getMemrefToCroutonFn(ModuleOp module, StringRef fnName,
                     ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {getPtrTy(context), rewriter.getI32Type()},
                                getPtrTy(context));
}

struct LowerMemrefToCrouton
    : public ConvertOpToLLVMPattern<hexagonmem::MemrefToCroutonOp> {
  using ConvertOpToLLVMPattern<
      hexagonmem::MemrefToCroutonOp>::ConvertOpToLLVMPattern;

private:
  std::string deviceType;

public:
  explicit LowerMemrefToCrouton(LLVMTypeConverter &converter,
                                const std::string &devType)
      : ConvertOpToLLVMPattern<hexagonmem::MemrefToCroutonOp>(converter),
        deviceType(devType) {}

  LogicalResult matchAndRewrite(hexagonmem::MemrefToCroutonOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto memrefToCroutonFnName = getMemrefToCroutonFnName(deviceType);
    MemRefType sourceType = llvm::cast<MemRefType>(op.getSource().getType());

    auto module = op->getParentOfType<ModuleOp>();
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        getMemrefToCroutonFn(module, memrefToCroutonFnName, rewriter);
    if (failed(funcOp))
      return failure();

    MemRefDescriptor bufferDesc(adaptor.getSource());
    auto bufferPtr = bufferDesc.alignedPtr(rewriter, loc);

    Value elementSizeInBytes =
        getSizeInBytes(loc, sourceType.getElementType(), rewriter);
    Value size = computeAllocationSize(sourceType, rewriter, bufferDesc, loc,
                                       getIndexType(), elementSizeInBytes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp.value(),
                                              ValueRange({bufferPtr, size}));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexagonmem::CroutonToMemrefOp
//===----------------------------------------------------------------------===//

static FailureOr<LLVM::LLVMFuncOp>
getCroutonToMemrefFn(ModuleOp module, StringRef fnName,
                     ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(rewriter, module, fnName, {getPtrTy(context)},
                                getPtrTy(context));
}

struct LowerCroutonToMemref
    : public ConvertOpToLLVMPattern<hexagonmem::CroutonToMemrefOp> {
  using ConvertOpToLLVMPattern<
      hexagonmem::CroutonToMemrefOp>::ConvertOpToLLVMPattern;

private:
  std::string deviceType;

public:
  explicit LowerCroutonToMemref(LLVMTypeConverter &converter,
                                const std::string &devType)
      : ConvertOpToLLVMPattern<hexagonmem::CroutonToMemrefOp>(converter),
        deviceType(devType) {}

  LogicalResult matchAndRewrite(hexagonmem::CroutonToMemrefOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto croutonToMemrefFnName = getCroutonToMemrefFnName(deviceType);

    auto module = op->getParentOfType<ModuleOp>();
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        getCroutonToMemrefFn(module, croutonToMemrefFnName, rewriter);
    if (failed(funcOp))
      return failure();

    auto sourcePtr = adaptor.getSource();

    mlir::LLVM::CallOp callOp = LLVM::CallOp::create(
        rewriter, loc, funcOp.value(), ValueRange({sourcePtr}));

    auto memRefType = mlir::cast<MemRefType>(op.getResult().getType());
    Value size;

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    this->getMemRefDescriptorSizes(loc, memRefType, {}, rewriter, sizes,
                                   strides, size,
                                   /* sizeInBytes */ true);

    auto memRefDescriptor = this->createMemRefDescriptor(
        loc, memRefType, callOp.getResult(), callOp.getResult(), sizes, strides,
        rewriter);
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Setup the Lowering Pass and patterns
//===----------------------------------------------------------------------===//

void populateHexagonMemToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns,
                                                const std::string &deviceType) {
  patterns.add<LowerAlloc>(converter, deviceType);
  patterns.add<LowerDealloc>(converter, deviceType);
  patterns.add<LowerCopy>(converter, deviceType);
  patterns.add<LowerMemrefToCrouton>(converter, deviceType);
  patterns.add<LowerCroutonToMemref>(converter, deviceType);
}

struct HexagonMemToLLVMPass
    : public ::impl::HexagonMemToLLVMBase<HexagonMemToLLVMPass> {
  explicit HexagonMemToLLVMPass(const HexagonMemToLLVMOptions &options)
      : Base(options) {}

  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, LLVM::LLVMDialect, HexagonMemDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);
    LowerToLLVMOptions options(context,
                               dataLayoutAnalysis.getAtOrAbove(moduleOp));
    LLVMTypeConverter typeConverter(context, options);

    target.addLegalDialect<memref::MemRefDialect>();
    // The resident-weight path derives the source address through the memref
    // dialect (`extract_aligned_pointer_as_index`), whose result is an `index`
    // that has to cross into i64. The cast is ordinary arith, lowered by the
    // ArithToLLVM pass that runs after this one; leaving arith legal here is
    // what lets the newly created cast survive this conversion.
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<HexagonMemDialect>();

    hexagon::addTypeConversions(context, typeConverter);
    populateHexagonMemToLLVMConversionPatterns(typeConverter, patterns,
                                               device_type);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

/// Implement the interface to convert HexagonMem to LLVM.
struct HexagonMemToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  HexagonMemToLLVMDialectInterface(mlir::Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<mlir::crouton::CroutonDialect>();
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    hexagon::addTypeConversions(getContext(), typeConverter);
    populateHexagonMemToLLVMConversionPatterns(typeConverter, patterns,
                                               "hexagon");
  }
};

} // namespace

void mlir::hexagonmem::registerConvertHexagonMemToLLVMInterface(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, hexagonmem::HexagonMemDialect *dialect) {
        dialect->addInterfaces<HexagonMemToLLVMDialectInterface>();
      });
}

std::unique_ptr<OperationPass<ModuleOp>>
hexagonmem::createHexagonMemToLLVMPass(const HexagonMemToLLVMOptions &options) {
  return std::make_unique<HexagonMemToLLVMPass>(options);
}
