//===-- HmxToLLVMPass.cpp - Lower hmx ops to LLVM -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// hmx.* -> llvm.call on the runtime leaves.
//
// The runtime is compiled by the SDK's clang and speaks in plain integers, so
// every buffer argument becomes the integer image of the memref's aligned
// pointer, and the calls are plain `llvm.call`s on `llvm.func` declarations (a
// `func.func` would grow an `_mlir_ciface_*` wrapper around the symbol).
//
// The assembly of the pass -- LLVMConversionTarget, DataLayoutAnalysis, memref
// descriptors -- follows HexKLToLLVM, which is the local precedent for lowering
// to this runtime.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/HmxToLLVM/HmxExternalFnNames.h"
#include "hexagon/Conversion/HmxToLLVM/HmxLeafSignatures.h"
#include "hexagon/Conversion/HmxToLLVM/HmxToLLVM.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDType.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxReadoutHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxRoleHandoff.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxVtcmAccounting.h"

// The runtime half of the channel's shared names: the suffixes that build
// the exported entry-point and depth-object symbols. Same header the
// launch-side probe includes (see HmxToLLVM's CMakeLists for why this
// include dir is on this target's path).
#include "HmxRoleChannel.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::hmx;

// At global scope, like HexKLToLLVM: the generated pass base lands in ::impl.
#define GEN_PASS_DEF_HMXTOLLVM
#include "hexagon/Conversion/HmxToLLVM/Passes.h.inc"

namespace {

/// Declare (once) an HMX runtime leaf with no result.
///
/// The argument types come from HmxLeafSignatures.h, never from the call site:
/// a lowering that spelled its own `SmallVector<Type>(9, i32Ty)` could not
/// disagree with HMXAPI.h at compile time, and did disagree on the device. A
/// name with no row, or a row whose C spelling `argTys` cannot place, is a
/// bug in the compiler, so both fail here with the leaf named rather than
/// falling back to a guess.
FailureOr<LLVM::LLVMFuncOp> getVoidLeaf(ModuleOp module, StringRef name,
                                        ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = module->getContext();
  const HmxLeafSignature *signature = lookupHmxLeafSignature(name);
  if (!signature) {
    module.emitError() << "hmx lowering calls '" << name
                       << "', which is not a leaf HMXAPI.h declares "
                          "(no row in HmxLeafSignatures.h)";
    return failure();
  }
  FailureOr<SmallVector<Type>> argTys = signature->argTys(ctx);
  if (failed(argTys)) {
    module.emitError() << "HMX leaf '" << name
                       << "' has a parameter type with no MLIR image "
                          "(HmxLeafSignatures.h::argTys)";
    return failure();
  }
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module, name, *argTys, voidTy);
}

/// Runtime entry that constructs the runtime's global singleton, i.e. powers
/// HMX up and acquires it. It allocates nothing, so it is the only way for a
/// kernel that never calls a runtime allocation to satisfy the engine
/// precondition.
static constexpr const char *kHmxEnsureFn = "hexagon_runtime_hmx_ensure_dsp";
/// Runtime entry that releases this thread's HMX lock at the end of the span
/// that kHmxEnsureFn opened; see the invariant on ensureHmxEngine.
static constexpr const char *kHmxUnlockFn = "hexagon_runtime_hmx_unlock_dsp";

/// Internal, marker-gated accounting ABI. The compiler emits one entry call
/// only for the static one-function/one-site/grid=1 context; all ordinary
/// kernels leave this path untouched. The token is passed as two opaque i64
/// words and is never reconstructed from a pointer. The symbols, version and
/// flag bits live in HmxResidentContract.h so the compiler half of the ABI is
/// spelled in exactly one place; this pass must not restate them.

/// Whether `fn` will execute on the HMX engine: its body contains an op of the
/// HMX dialect that lowers to an HMX engine leaf (mma / acc / bias / pack /
/// unpack) -- exactly the functions the ensure/unlock pair guards.
///
/// The test is on the dialect ops themselves, and it has to run *before* the
/// conversion below erases them; it never looks at callee names. A callee-name
/// prefix decides wrongly as soon as a leaf is renamed or wrapped, or when any
/// unrelated symbol happens to carry the prefix, and the failure is silent: a
/// missing unlock parks the next thread in HAP_compute_res_hmx_lock forever
/// (bin/runtime/src/HexagonAPI.cpp, EnsureHmxLockForThisThread), while a
/// wrongly-added pair only costs a lock round-trip the function releases
/// itself.
///
/// The engine/non-engine split is carried structurally by the
/// `OpTrait::HmxDmaOnly` marker (defined with the dialect in HmxDialect.h):
/// `hmx.stage` / `hmx.await` carry it because they lower to the plain DMA
/// runtime entries (`hexagon_runtime_dma_*`), execute no HMX instruction and
/// need no engine (see HmxExternalFnNames.cpp). The polarity is deliberate and
/// was reviewed: every *unmarked* dialect op counts as engine until proven
/// otherwise -- a wrongly-included op adds a pair the function releases itself,
/// a wrongly-excluded one hangs the device -- so a future engine op needs no
/// marker, while a future DMA-only op must set one.
///
/// The contract is pinned by test/Conversion/HmxToLLVM/hmx-to-llvm.mlir
/// @stage_await ("a function that only stages and awaits issues no HMX
/// instruction, so it needs no engine ensure/unlock"). Do not "simplify" this
/// away: deciding by callee-name prefix would go wrong silently on a renamed or
/// a wrapped leaf, and an engine whitelist would silently drop the pair of any
/// future engine op and hang the device.
/// Read one unsigned 64-bit identity word of the event wire. A narrowed or
/// signed attribute is a different type, not a value to reinterpret: the ABI
/// words are opaque and unsigned, so a mismatch is a schema error.
static bool readEventWord(Attribute attr, uint64_t &value) {
  auto integer = dyn_cast_or_null<IntegerAttr>(attr);
  if (!integer || !integer.getType().isSignlessInteger(64))
    return false;
  value = integer.getValue().getZExtValue();
  return true;
}

/// Reconcile the eligible event context with the static identity sidecar when
/// the latter is present.
///
/// The token is derived from the identity sidecar's canonical function, source
/// site and scope, so a record that names a scope but cannot be matched back to
/// the identity that produced it is unattributable: the device would compare two
/// opaque words that no compiler fact backs. Two sidecars that disagree, or an
/// identity record missing a fact the event context claims, are therefore a
/// refusal to lower rather than a choice between them. Nothing here merges the
/// schemas; the two remain separate records joined by exact equality.
static LogicalResult reconcileEventContextWithIdentity(
    ModuleOp moduleOp, StringRef functionName, uint64_t accountingScopeId,
    uint64_t functionId, uint64_t siteId, uint64_t tokenLow, uint64_t tokenHigh) {
  const char *mismatch =
      "eligible diagnostic VTCM event context does not reconcile with "
      "hmx.kernel_vtcm_identity";
  auto identity =
      dyn_cast_or_null<DictionaryAttr>(moduleOp->getAttr(kHmxVtcmIdentityAttr));
  if (!identity) {
    moduleOp.emitError() << mismatch << ": identity sidecar is not a dictionary";
    return failure();
  }

  auto identityStatus = identity.getAs<StringAttr>("status");
  uint64_t identityScopeId = 0;
  if (!identityStatus || identityStatus.getValue() != "complete" ||
      !readEventWord(identity.get("scope_id"), identityScopeId) ||
      identityScopeId != accountingScopeId) {
    moduleOp.emitError() << mismatch << ": scope_id";
    return failure();
  }

  auto functions = identity.getAs<ArrayAttr>("functions");
  if (!functions) {
    moduleOp.emitError() << mismatch << ": functions";
    return failure();
  }
  // The eligible scope is one function, so exactly one entry may carry an
  // identity claim. An unresolved entry has no function_id and is not a claim.
  DictionaryAttr claimedFunction;
  for (Attribute entry : functions) {
    auto function = dyn_cast<DictionaryAttr>(entry);
    if (!function) {
      moduleOp.emitError() << mismatch << ": functions entry is not a dictionary";
      return failure();
    }
    if (!function.get("function_id"))
      continue;
    if (claimedFunction) {
      moduleOp.emitError() << mismatch << ": two functions claim an identity";
      return failure();
    }
    claimedFunction = function;
  }
  if (!claimedFunction) {
    moduleOp.emitError() << mismatch << ": no function claims an identity";
    return failure();
  }
  auto claimedName = claimedFunction.getAs<StringAttr>("function");
  uint64_t claimedFunctionId = 0;
  if (!claimedName || claimedName.getValue() != functionName ||
      !readEventWord(claimedFunction.get("function_id"), claimedFunctionId) ||
      claimedFunctionId != functionId) {
    moduleOp.emitError() << mismatch << ": function_id";
    return failure();
  }

  auto sites = claimedFunction.getAs<ArrayAttr>("sites");
  if (!sites) {
    moduleOp.emitError() << mismatch << ": sites";
    return failure();
  }
  DictionaryAttr claimedSite;
  for (Attribute entry : sites) {
    auto site = dyn_cast<DictionaryAttr>(entry);
    if (!site) {
      moduleOp.emitError() << mismatch << ": sites entry is not a dictionary";
      return failure();
    }
    if (!site.get("site_id"))
      continue;
    if (claimedSite) {
      moduleOp.emitError() << mismatch << ": two sites claim an identity";
      return failure();
    }
    claimedSite = site;
  }
  if (!claimedSite) {
    moduleOp.emitError() << mismatch << ": no site claims an identity";
    return failure();
  }
  auto siteStatus = claimedSite.getAs<StringAttr>("identity_status");
  auto siteFunction = claimedSite.getAs<StringAttr>("function");
  auto siteBasis = claimedSite.getAs<StringAttr>("event_token_basis");
  uint64_t claimedSiteId = 0;
  uint64_t claimedSiteFunctionId = 0;
  uint64_t claimedTokenBits = 0;
  uint64_t claimedTokenLow = 0;
  uint64_t claimedTokenHigh = 0;
  if (!siteStatus || siteStatus.getValue() != "complete" || !siteFunction ||
      siteFunction.getValue() != functionName || !siteBasis ||
      siteBasis.getValue() != kHmxDiagnosticEventTokenSchema ||
      !readEventWord(claimedSite.get("site_id"), claimedSiteId) ||
      claimedSiteId != siteId ||
      !readEventWord(claimedSite.get("function_id"), claimedSiteFunctionId) ||
      claimedSiteFunctionId != functionId ||
      !readEventWord(claimedSite.get("event_token_bits"), claimedTokenBits) ||
      claimedTokenBits != 128 ||
      !readEventWord(claimedSite.get("event_token_low"), claimedTokenLow) ||
      claimedTokenLow != tokenLow ||
      !readEventWord(claimedSite.get("event_token_high"), claimedTokenHigh) ||
      claimedTokenHigh != tokenHigh) {
    moduleOp.emitError() << mismatch << ": canonical site or event token";
    return failure();
  }
  return success();
}

static LogicalResult insertDiagnosticEventContext(ModuleOp moduleOp) {
  MLIRContext *context = moduleOp.getContext();
  // A missing sidecar is the only silent "no context". Any other spelling is a
  // present record: reading it as a dictionary (or as an eligible flag) that
  // is not there would lower an un-instrumented kernel and report success,
  // which is the same observable failure as not running the producer at all.
  Attribute rawSidecar = moduleOp->getAttr(kHmxVtcmEventContextAttr);
  if (!rawSidecar)
    return success();
  auto contextAttr = dyn_cast<DictionaryAttr>(rawSidecar);
  if (!contextAttr) {
    moduleOp.emitError(
        "diagnostic VTCM event-context sidecar must be a dictionary");
    return failure();
  }
  auto eligible = dyn_cast_or_null<BoolAttr>(contextAttr.get("eligible"));
  if (!eligible) {
    moduleOp.emitError(
        "diagnostic VTCM event-context sidecar requires a BoolAttr 'eligible'");
    return failure();
  }
  if (!moduleOp->getAttr(kHmxDiagnosticVtcmEvidenceContextAttr) ||
      !isa<UnitAttr>(moduleOp->getAttr(kHmxDiagnosticVtcmEvidenceContextAttr))) {
    moduleOp.emitError(
        "eligible diagnostic VTCM event context requires hmx.diagnostic_vtcm_evidence_context");
    return failure();
  }
  auto kind = contextAttr.getAs<StringAttr>("kind");
  if (!kind || kind.getValue() != kHmxDiagnosticEventContextKind) {
    moduleOp.emitError("diagnostic VTCM event-context sidecar requires kind = "
                       "\"vtcm-event-context\"");
    return failure();
  }
  if (!eligible.getValue()) {
    // The producer deliberately retains a negative record for unsupported
    // scopes. It is evidence, not an instruction to emit a partial context --
    // but it is still a typed refusal, so it must match the ineligible schema
    // exactly rather than being accepted as a dictionary of unknown shape.
    if (!hasExactHmxDiagnosticEventContextKeys(
            contextAttr, hmxDiagnosticEventContextIneligibleKeys())) {
      moduleOp.emitError(
          "malformed ineligible hmx diagnostic VTCM event-context sidecar");
      return failure();
    }
    return success();
  }
  if (!hasExactHmxDiagnosticEventContextKeys(
          contextAttr, hmxDiagnosticEventContextEligibleKeys())) {
    moduleOp.emitError(
        "malformed eligible hmx diagnostic VTCM event-context sidecar");
    return failure();
  }

  auto schema = contextAttr.getAs<StringAttr>("schema");
  auto status = contextAttr.getAs<StringAttr>("status");
  auto tokenBasis = contextAttr.getAs<StringAttr>("token_basis");
  auto mode = contextAttr.getAs<StringAttr>("mode");
  auto delayedCacheOwner = contextAttr.getAs<StringAttr>(
      "delayed_cache_owner_status");
  auto gridScopeStatus = contextAttr.getAs<StringAttr>("grid_scope_status");
  auto runtimeJoinStatus = contextAttr.getAs<StringAttr>(
      "runtime_event_join_status");
  auto performanceClaimed = contextAttr.getAs<IntegerAttr>(
      "performance_claimed");
  auto functionName = contextAttr.getAs<StringAttr>("function");
  auto gridProduct = contextAttr.getAs<IntegerAttr>("grid_product");
  auto invocationId = contextAttr.getAs<IntegerAttr>("invocation_id");
  auto immutable = contextAttr.getAs<IntegerAttr>("immutable");
  auto accountingScopeId = contextAttr.getAs<IntegerAttr>("accounting_scope_id");
  auto functionId = contextAttr.getAs<IntegerAttr>("function_id");
  auto allocationSiteId = contextAttr.getAs<IntegerAttr>("allocation_site_id");
  auto tokenBits = contextAttr.getAs<IntegerAttr>("token_bits");
  auto tokenLow = contextAttr.getAs<IntegerAttr>("token_low");
  auto tokenHigh = contextAttr.getAs<IntegerAttr>("token_high");
  uint64_t accountingScopeValue = 0;
  uint64_t functionIdValue = 0;
  uint64_t allocationSiteIdValue = 0;
  uint64_t tokenBitsValue = 0;
  uint64_t tokenLowValue = 0;
  uint64_t tokenHighValue = 0;
  if (!schema || schema.getValue() != kHmxVtcmEventContextSchema ||
      !status || status.getValue() != "complete" || !tokenBasis ||
      tokenBasis.getValue() != kHmxDiagnosticEventTokenSchema || !mode ||
      mode.getValue() != "diagnostic-only" || !delayedCacheOwner ||
      delayedCacheOwner.getValue() != "aggregate" || !gridScopeStatus ||
      gridScopeStatus.getValue() != "not-proven" || !runtimeJoinStatus ||
      runtimeJoinStatus.getValue() != "not-proven" || !performanceClaimed ||
      !performanceClaimed.getType().isSignlessInteger(64) ||
      performanceClaimed.getInt() != 0 || !functionName ||
      functionName.getValue().empty() || !gridProduct ||
      !gridProduct.getType().isSignlessInteger(64) || gridProduct.getInt() != 1 ||
      !invocationId || !invocationId.getType().isSignlessInteger(64) ||
      invocationId.getInt() != 1 || !immutable ||
      !immutable.getType().isSignlessInteger(64) || immutable.getInt() != 1 ||
      !readEventWord(accountingScopeId, accountingScopeValue) ||
      accountingScopeValue == 0 || !readEventWord(functionId, functionIdValue) ||
      functionIdValue == 0 ||
      !readEventWord(allocationSiteId, allocationSiteIdValue) ||
      allocationSiteIdValue == 0 || !readEventWord(tokenBits, tokenBitsValue) ||
      tokenBitsValue != 128 || !readEventWord(tokenLow, tokenLowValue) ||
      !readEventWord(tokenHigh, tokenHighValue) ||
      (tokenLowValue == 0 && tokenHighValue == 0)) {
    moduleOp.emitError(
        "malformed eligible hmx diagnostic VTCM event-context sidecar");
    return failure();
  }

  // The identity sidecar is optional on its own (a hand-written module may carry
  // only the event record), but once it is present it is the authority for the
  // canonical facts the token was derived from.
  if (moduleOp->hasAttr(kHmxVtcmIdentityAttr) &&
      failed(reconcileEventContextWithIdentity(
          moduleOp, functionName.getValue(), accountingScopeValue,
          functionIdValue, allocationSiteIdValue, tokenLowValue,
          tokenHighValue)))
    return failure();

  Operation *target = nullptr;
  bool multipleTargets = false;
  moduleOp.walk([&](LLVM::LLVMFuncOp function) {
    if (function.getName() != functionName.getValue())
      return;
    if (target != nullptr)
      multipleTargets = true;
    target = function.getOperation();
  });
  moduleOp.walk([&](func::FuncOp function) {
    if (function.getName() != functionName.getValue())
      return;
    if (target != nullptr)
      multipleTargets = true;
    target = function.getOperation();
  });
  if (multipleTargets || target == nullptr ||
      (isa<LLVM::LLVMFuncOp>(target) && cast<LLVM::LLVMFuncOp>(target).isDeclaration()) ||
      (isa<func::FuncOp>(target) && cast<func::FuncOp>(target).isDeclaration())) {
    moduleOp.emitError("diagnostic VTCM event-context target function is missing or ambiguous");
    return failure();
  }

  bool alreadyInserted = false;
  target->walk([&](LLVM::CallOp call) {
    if (call.getCallee() == kHmxDiagnosticEventContextEnterFn)
      alreadyInserted = true;
  });
  if (alreadyInserted)
    return success();

  OpBuilder builder(context);
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  FailureOr<LLVM::LLVMFuncOp> entry = LLVM::lookupOrCreateFn(
      builder, moduleOp, kHmxDiagnosticEventContextEnterFn,
      ArrayRef<Type>{i32, i32, i64, i64, i64, i64, i64, i64, i32},
      LLVM::LLVMVoidType::get(context));
  if (failed(entry)) {
    moduleOp.emitError("could not create diagnostic VTCM event-context ABI declaration");
    return failure();
  }
  entry.value()->setAttr(
      "passthrough",
      builder.getArrayAttr({builder.getStringAttr("noinline"),
                             builder.getStringAttr("willreturn")}));
  FailureOr<LLVM::LLVMFuncOp> leave = LLVM::lookupOrCreateFn(
      builder, moduleOp, kHmxDiagnosticEventContextLeaveFn, ArrayRef<Type>{},
      LLVM::LLVMVoidType::get(context));
  if (failed(leave)) {
    moduleOp.emitError("could not create diagnostic VTCM event-context leave ABI declaration");
    return failure();
  }
  leave.value()->setAttr(
      "passthrough",
      builder.getArrayAttr({builder.getStringAttr("noinline"),
                             builder.getStringAttr("willreturn")}));

  builder.setInsertionPointToStart(&target->getRegion(0).front());
  auto constant = [&](Type type, uint64_t value) -> Value {
    if (type == i32) {
      // All i32 event ABI fields are checked above (version, flags, and
      // grid=1), so this conversion is bounded by the validation gate rather
      // than by a nullable helper result.
      return LLVM::ConstantOp::create(builder, target->getLoc(), type,
                                      builder.getI32IntegerAttr(
                                          static_cast<int32_t>(value)));
    }
    return LLVM::ConstantOp::create(
        builder, target->getLoc(), type,
        IntegerAttr::get(IntegerType::get(context, 64), APInt(64, value)));
  };
  SmallVector<Value> operands;
  operands.push_back(constant(i32, kHmxDiagnosticEventContextAbiVersion));
  operands.push_back(constant(
      i32, kHmxDiagnosticEventContextFlagSingleInvocation |
               kHmxDiagnosticEventContextFlagGridOne));
  operands.push_back(constant(i64, tokenLowValue));
  operands.push_back(constant(i64, tokenHighValue));
  operands.push_back(constant(i64, accountingScopeValue));
  operands.push_back(constant(i64, invocationId.getInt()));
  operands.push_back(constant(i64, functionIdValue));
  operands.push_back(constant(i64, allocationSiteIdValue));
  operands.push_back(constant(i32, gridProduct.getInt()));

  LLVM::CallOp::create(
      builder, target->getLoc(), TypeRange{},
      FlatSymbolRefAttr::get(entry->getOperation()), operands);
  SmallVector<Operation *> returns;
  target->walk([&](Operation *op) {
    if (isa<LLVM::ReturnOp, func::ReturnOp>(op))
      returns.push_back(op);
  });
  for (Operation *returnOp : returns) {
    builder.setInsertionPoint(returnOp);
    LLVM::CallOp::create(
        builder, target->getLoc(), TypeRange{},
        FlatSymbolRefAttr::get(leave->getOperation()), ValueRange{});
  }
  return success();
}

/// Refuse a module that claims an eligible per-site scope table but never
/// bracketed one allocation with it.
///
/// The site table is produced by the accounting pass and consumed by the
/// hexagonmem lowering, which runs earlier in the pipeline. If the table says
/// `eligible` and no bracket exists anywhere in the module, the two halves
/// disagree: either the lowering never ran, or it ran on IR that no longer
/// matched. Publishing the table anyway would let a host read a per-site
/// attribution that the kernel never asked the device to make, so this is a hard
/// error rather than a warning.
///
/// The frame context is deliberately untouched here: it is emitted by this pass
/// itself, so it cannot disagree with itself.
static LogicalResult verifySiteScopeBrackets(ModuleOp moduleOp) {
  Attribute raw = moduleOp->getAttr(kHmxVtcmSiteScopesAttr);
  if (!raw)
    return success();
  auto table = dyn_cast<DictionaryAttr>(raw);
  if (!table) {
    moduleOp.emitError("hmx.kernel_vtcm_site_scopes must be a dictionary");
    return failure();
  }
  // `BoolAttr` is an optional-like wrapper: a *present* `eligible = false` is
  // still truthy as an attribute. The flag therefore has to be read through
  // getValue(), or every ineligible table would be validated as an eligible one.
  auto eligible = table.getAs<BoolAttr>("eligible");
  if (!eligible) {
    moduleOp.emitError(
        "hmx.kernel_vtcm_site_scopes requires a BoolAttr 'eligible'");
    return failure();
  }
  if (!eligible.getValue()) {
    // An ineligible table is a typed refusal. It must still be the closed
    // ineligible shape, so a stale or foreign spelling cannot pass as "no
    // claim" while carrying identity fields.
    if (!hasExactHmxDiagnosticSiteScopeKeys(
            table, hmxDiagnosticSiteScopesIneligibleKeys())) {
      moduleOp.emitError("malformed ineligible hmx.kernel_vtcm_site_scopes");
      return failure();
    }
    return success();
  }
  if (!hasExactHmxDiagnosticSiteScopeKeys(
          table, hmxDiagnosticSiteScopesEligibleKeys())) {
    moduleOp.emitError("malformed eligible hmx.kernel_vtcm_site_scopes");
    return failure();
  }
  unsigned brackets = 0;
  moduleOp.walk([&](LLVM::CallOp call) {
    if (call.getCallee() == kHmxDiagnosticSiteScopeEnterFn)
      ++brackets;
  });
  if (brackets == 0) {
    moduleOp.emitError(
        "eligible hmx.kernel_vtcm_site_scopes table has no "
        "hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp call: the "
        "pool-backed allocations were never bracketed with their site scope");
    return failure();
  }
  return success();
}

static bool issuesHmxEngineLeaves(Operation *fn) {
  bool found = false;
  fn->walk([&](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (dialect &&
        dialect->getNamespace() == HmxDialect::getDialectNamespace() &&
        !op->hasTrait<OpTrait::HmxDmaOnly>() &&
        !op->hasTrait<OpTrait::HmxLayoutHvx>())
      found = true;
  });
  return found;
}

/// The five runtime leaves that issue an HMX instruction, by symbol name.
///
/// Narrowed from a `hmx_` prefix test on 2026-10-02. The prefix covered the 37
/// layout symbols in HMXLayout.c as well as these 5, so once pack/unpack stop
/// counting as engine ops (HmxLayoutHvx) every pack-only function became an
/// unverifiable "engine" caller. Verified disjoint: HMXAPI.c and HMXLayout.c
/// share no symbol name, so this predicate separates them without ambiguity.
/// Naming all five also makes the check strictly stronger than the prefix
/// test -- it fires on exactly the leaves that need the pair, not on a
/// superset -- while still never selecting a function for insertion.
///
/// Do not turn this into a whitelist for `issuesHmxEngineLeaves`. That
/// decision is made on dialect ops, before conversion, and its polarity is
/// "unmarked means engine": omission there costs one harmless lock
/// round-trip, and every one of these five is an engine instruction.
static bool isHmxEngineLeaf(StringRef callee) {
  return callee == hmx::getBiasInitUnitF16FnName() ||
         callee == hmx::getBiasLoadF16FnName() ||
         callee == hmx::getAccClearF16FnName() ||
         callee == hmx::getAccStoreF16FnName() ||
         callee == hmx::getMmaF16FnName();
}

/// Post-conversion invariant check, deliberately run *before* ensureHmxEngine
/// so a diagnostic sees IR the insertion has not touched yet. A function that
/// calls an HMX leaf but is absent from `engineKernels` would run HMX
/// instructions with no ensure/unlock pair, and the next thread would block
/// forever in HAP_compute_res_hmx_lock (bin/runtime/src/HexagonAPI.cpp,
/// EnsureHmxLockForThisThread) -- so that state must fail the build loudly
/// instead of shipping a kernel that hangs the device.
///
  /// The engine-leaf names appear here **for verification only, never for the
  /// decision**: which functions get the pair is decided solely by
  /// issuesHmxEngineLeaves, on the hmx dialect ops, before the conversion. The
  /// dialect test cannot see a leaf call that exists without a dialect op
  /// behind it -- and this check turns exactly that one direction, which used
  /// to be a *silent* missing pair, into a compile-time error. It never
  /// selects a function for insertion.
static void verifyHmxLeafCallers(ModuleOp moduleOp,
                                 const llvm::StringSet<> &engineKernels,
                                 const llvm::StringSet<> &roleWorkFns) {
  auto verify = [&](auto fn) {
    if (fn.isDeclaration() || engineKernels.count(fn.getName()))
      return;
    // The role split's work function is the one deliberate exception: it
    // runs on the bound thread, whose lifetime lock (taken once at thread
    // start) replaces the per-kernel pair. An unlock inserted here would
    // release that lifetime lock; a missing ensure would be equally wrong
    // in the other direction. The exemption is the channel's contract, not
    // a check someone forgot.
    if (roleWorkFns.count(fn.getName()))
      return;
    bool found = false;
    fn->walk([&](LLVM::CallOp call) {
      if (std::optional<StringRef> callee = call.getCallee())
        found |= isHmxEngineLeaf(*callee);
    });
    if (found)
      fn.emitError()
          << "function '" << fn.getName()
          << "' calls an HMX engine leaf (" << hmx::getMmaF16FnName()
          << " and its four siblings) but issuesHmxEngineLeaves did not "
             "recognise it from its "
             "hmx dialect ops, so it gets no "
             "hexagon_runtime_hmx_ensure_dsp/hexagon_runtime_hmx_unlock_dsp "
             "pair: it would execute HMX instructions without the engine "
             "brought up, and the next thread would block forever in "
             "HAP_compute_res_hmx_lock";
  };
  moduleOp.walk([&](LLVM::LLVMFuncOp fn) { verify(fn); });
  moduleOp.walk([&](func::FuncOp fn) { verify(fn); });
}

//===----------------------------------------------------------------------===//
// VECTOR-READOUT EXECUTOR HANDOFF
//===----------------------------------------------------------------------===//
//
// HmxVectorReadoutPass moves a matmul's `hmx.unpack_acc` read-out off the matrix
// engine's thread by handing groups of AR rows to a resident vector thread
// through `hexagon_runtime_hmx_exec_publish`. It runs while kernels are still
// `func.func`, which is too early for the one call that makes the split work: the
// executor has no function to run until something passes it a pointer to the
// outlined read-out.
//
// This is that something. It runs after `convert-func-to-llvm`, so
// `llvm.mlir.addressof` accepts the outlined symbol, and it owns three jobs:
//
//   * build the entry point that adapts the runtime's `HmxReadoutFn`
//     (`void(const HmxReadoutBatch*, uint32_t)`) onto the outlined read-out;
//   * call `hexagon_runtime_hmx_exec_configure` with that entry point's address;
//   * check every `publish` return, because a dropped batch is unwritten output.
//
// The design and its measurements are in bin/runtime/include/HmxVectorExecutor.h;
// the compiler half of the ABI is in HmxReadoutHandoff.h, shared with the
// producer pass so the symbols and the descriptor layout are spelled once.

/// The count each `hexagon_runtime_hmx_exec_publish` call was given.
// TRACEABILITY: kHmxExecPublishedBatches
//   mechanism: the count each hexagon_runtime_hmx_exec_publish call was given,
//     which the drop check compares the return against. It does not steer the
//     generated schedule: it is a property of what the producer pass emitted.
//     (`configure`'s numThreads is now also a producer fact: the placeholder
//     call passes 1, the only value the frozen runtime honours --
//     kVectorThreads == 1, HmxVectorExecutor.cpp; anything else returns -2.)
//   measurement: kHmxExecPublishedBatches is 1 because the producer emits
//     `publish(addr, 1)` -- one descriptor, one batch -- and the runtime calls the
//     read-out once per batch. Its correctness does not rest on the value: the
//     check is `<` against what was asked, so it fires on any short return, which
//     is the only failure that matters.
//   shape set: n/a. Not a shape or size threshold; fixed by an interface
//     contract.
//   workload representativeness: n/a, for the same reason.
static constexpr int32_t kHmxExecPublishedBatches = 1;

/// Append one memref's expanded `convert-func-to-llvm` argument list.
///
/// Under the default (non-bare-pointer) convention every memref parameter becomes
/// `(allocated, aligned, offset, sizes..., strides...)`, each as its own LLVM
/// value; see `llvm::detail::passFunctionLike` / `convertFuncToLLVM`. That is the
/// shape the outlined read-out's signature has by the time this runs, so the
/// entry point has to hand it exactly that.
///
/// `address` is the aligned pointer word from the descriptor, which is the base
/// the leaf addressing arithmetic uses (`asAddress`, below). The producer
/// publishes `memref.extract_aligned_pointer_as_index` of the two buffers, so the
/// word is the ALIGNED pointer and the static offset is part of the type --
/// re-deriving `allocated` from it keeps both words consistent instead of leaving
/// one of them describing a buffer nobody allocated. The read-out reads only
/// `aligned` and `offset`; `allocated`, sizes and strides are what the calling
/// convention requires to be present, and passing the real values (rather than
/// zeros) means a future reader of them is not reading poison.
static int64_t memrefAddressSpace(MemRefType type, int64_t fallback);

static void appendMemrefArgs(OpBuilder &rewriter, Location loc,
                             MemRefType type, Value addressWord,
                             SmallVectorImpl<Value> &out) {
  auto i64Ty = rewriter.getI64Type();
  // The callee's own signature is what decides this, and it carries the
  // accumulator's VTCM space: `convert-func-to-llvm` turns
  // `memref<..., 1>` into `!llvm.ptr<1>` here, and the two arguments differ
  // (`ar` is space 1, `dst` is space 0). Taking the space from the memref type
  // the producer recorded is what keeps the call type-correct in both positions;
  // forcing space 0 would be a mismatch the verifier catches, and forcing space
  // 1 for `dst` would be a claim the surrounding IR does not make.
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                         memrefAddressSpace(type, 0));
  auto constant = [&](int64_t value) -> Value {
    return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                    rewriter.getI64IntegerAttr(value));
  };

  // The descriptor word is an i32 ADDRESS, not a pointer: `asAddress` in this file
  // and `memref.extract_aligned_pointer_as_index` in the producer both speak i32,
  // because that is how every address crosses into the runtime on this target. So
  // the word is turned back into a pointer here, in the callee's own address space.
  Value address = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, addressWord);

  SmallVector<int64_t, 8> strides;
  int64_t offset = 0;
  int64_t rank = type.getRank();
  bool stridesKnown =
      succeeded(type.getStridesAndOffset(strides, offset)) && offset >= 0 &&
      llvm::all_of(strides,
                   [](int64_t stride) { return !ShapedType::isDynamic(stride); });

  // The allocated pointer is the aligned pointer minus the type's static offset,
  // which is what a `memref.reinterpret_cast` of a larger buffer would have
  // produced. A buffer the producer never offset needs no arithmetic at all.
  Value allocated = address;
  if (stridesKnown && offset != 0) {
    // Computed in the width the word arrived in, which is the runtime's own
    // address ABI (i32 on this target). Widening to i64 to hold the byte count
    // would need a trunc back, and a byte offset that does not fit the word is
    // a buffer this ABI cannot address at all -- so it is left alone rather than
    // wrapped, which would be a pointer to the wrong row.
    Type addrTy = addressWord.getType();
    int64_t elementBytes = type.getElementTypeBitWidth() / 8;
    int64_t byteOffset = offset * elementBytes;
    unsigned width = addrTy.getIntOrFloatBitWidth();
    int64_t addrMax = width >= 64
                          ? std::numeric_limits<int64_t>::max()
                          : (int64_t{1} << (width - 1)) - 1;
    if (ShapedType::isDynamic(byteOffset) || byteOffset > addrMax)
      allocated = address;
    else {
      Value shifted = LLVM::SubOp::create(
          rewriter, loc, addrTy, addressWord,
          LLVM::ConstantOp::create(rewriter, loc, addrTy,
                                   rewriter.getIntegerAttr(addrTy, byteOffset)));
      allocated = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, shifted);
    }
  }

  out.push_back(allocated);
  out.push_back(address);
  out.push_back(constant(offset));
  for (int64_t dim = 0; dim < rank; ++dim)
    out.push_back(stridesKnown ? constant(type.getDimSize(dim))
                               : LLVM::UndefOp::create(rewriter, loc, i64Ty)
                                     .getResult());
  for (int64_t dim = 0; dim < rank; ++dim)
    out.push_back(stridesKnown ? constant(strides[dim])
                               : LLVM::UndefOp::create(rewriter, loc, i64Ty)
                                     .getResult());
}

/// Generate the function the executor runs: `void(const HmxReadoutBatch*, u32)`.
///
/// WHY AN ADAPTER RATHER THAN THE OUTLINED READ-OUT DIRECTLY. The runtime calls
/// `HmxReadoutFn = void(*)(const HmxReadoutBatch*, uint32_t)` and the descriptor
/// is six i32 words, so in principle the outlined read-out could take
/// `(i8* batch, i32 count)` and load what it needs. It cannot: the words are
/// runtime values, and turning an `ar`/`dst` address back into a memref requires
/// the descriptor form (`allocated/aligned/offset/sizes/strides`), and the `func`
/// dialect has no op that builds a memref from a pointer -- `memref.cast`,
/// `memref.reinterpret_cast` and `memref.get_global` all need a memref or a
/// symbol to start from. So the outlined read-out keeps its memref parameters and
/// this adapter does the translation, at the one pipeline point where an address
/// is expressible.
///
/// What it forwards, and what it does not, is the substantive part:
///
///   * `rowStart`/`rowCount` come out of words 0 and 1. These are the group's
///     bounds, so the adapter can reconstruct them exactly.
///   * `ar`/`dst` come out of words 4 and 5 as i32 addresses.
///   * `validRows`/`nCroutons` (words 2 and 3) are NOT forwarded. They are STATIC
///     specialisation facts on this path: `hmx.unpack_acc` takes `count`,
///     `valid_rows` and `valid_cols` as attributes (HmxOps.td:329-331), so they
///     are already baked into the outlined read-out's body, and forwarding them
///     would require an outlined function that takes parameters the op cannot be
///     given. Reading them and doing nothing with them would be a lie about the
///     dependency.
///   * `count` is not used. The runtime calls the function once per batch with
///     `count == 1` (HmxVectorExecutor.cpp:576 loads the slot and calls it), and
///     a descriptor names one group, so looping here would be a second place to
///     get the batching wrong. The parameter stays because the ABI carries it.
///
/// If the outlined signature ever stops being `(memref, memref, index, index)`
/// this stops being correct, so the shape is verified rather than assumed:
/// `emitReadoutEntry` reads the actual parameter count and requires the two
/// trailing `index` parameters to have become two i64s (what `index` lowers to
/// at the default bitwidth).
static LLVM::LLVMFuncOp emitReadoutEntry(OpBuilder &builder, ModuleOp moduleOp,
                                         LLVM::LLVMFuncOp work,
                                         MemRefType arType,
                                         MemRefType dstType, StringRef name) {
  MLIRContext *context = moduleOp.getContext();
  auto i32Ty = builder.getI32Type();
  auto i64Ty = builder.getI64Type();
  auto ptrTy = LLVM::LLVMPointerType::get(context, 0);
  // The descriptor as a value: six i32 words in the frozen field order. Reading
  // it as one struct rather than six GEP+load pairs is both shorter and a
  // statement about the layout -- the same `HmxReadoutBatch` the runtime's own
  // `slots[]` array is, so the two agree by construction rather than by
  // convention.
  auto batchTy = LLVM::LLVMStructType::getLiteral(
      context, SmallVector<Type>(kHmxReadoutBatchWords, i32Ty));

  LLVM::LLVMFunctionType fnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(context), {ptrTy, i32Ty}, /*isVarArg=*/false);
  OpBuilder funcBuilder = OpBuilder::atBlockEnd(moduleOp.getBody());
  LLVM::LLVMFuncOp entry =
      LLVM::LLVMFuncOp::create(funcBuilder, moduleOp.getLoc(), name, fnType);
  // Private for the same reason the outlined read-out is: neither symbol is part
  // of the kernel's ABI, and `configure` takes the address, so nothing resolves
  // them from outside the module.
  entry.setVisibility(SymbolTable::Visibility::Private);
  Block *block = entry.addEntryBlock(funcBuilder);
  OpBuilder body = OpBuilder::atBlockEnd(block);
  Location loc = moduleOp.getLoc();

  Value batch = LLVM::LoadOp::create(body, loc, batchTy, block->getArgument(0));
  auto word = [&](int64_t index) -> Value {
    return LLVM::ExtractValueOp::create(body, loc, i32Ty, batch,
                                        ArrayRef<int64_t>{index});
  };
  Value rowStart = word(kHmxReadoutRowStart);
  Value rowCount = word(kHmxReadoutRowCount);
  Value ar = word(kHmxReadoutAr);
  Value dst = word(kHmxReadoutDst);

  SmallVector<Value> args;
  appendMemrefArgs(body, loc, arType, ar, args);
  appendMemrefArgs(body, loc, dstType, dst, args);
  // `rowStart`/`rowCount` are `uint32_t` in the ABI and `index` in the outlined
  // read-out, and `index` lowers to i64 at this pipeline's default bitwidth. Zero
  // extension is the faithful conversion of an unsigned 32-bit word to i64.
  args.push_back(LLVM::ZExtOp::create(body, loc, i64Ty, rowStart));
  args.push_back(LLVM::ZExtOp::create(body, loc, i64Ty, rowCount));

  LLVM::CallOp::create(body, loc, TypeRange{},
                       FlatSymbolRefAttr::get(work), args);
  LLVM::ReturnOp::create(body, loc, ValueRange{});
  return entry;
}

/// Write each handoff's entry-point address into the engine's placeholder
/// `configure` calls, in program order.
///
/// The producer pass emits one placeholder per matmul group -- before the
/// group's first tile loop -- because a function's address is not expressible
/// before `convert-func-to-llvm` (`llvm.mlir.addressof` rejects a `func.func`
/// symbol). This pass runs after that conversion and is the only one that can
/// take the address, so the pairing happens here: the engine's k-th configure
/// call names the k-th handoff record's entry point. The producer emits calls
/// and records in the same group order, so program order pairs them without
/// any other identity travelling between the two passes.
///
/// For every group but the first, the call's POSITION is load-bearing, not
/// just its address: `configure` drains the ring before swapping the function
/// pointer (HmxVectorExecutor.cpp configure), so the previous group's
/// in-flight batches finish under their own read-out before this group's
/// batches can run at all. That drain is why multiple handoffs per engine are
/// sound at all, and why this wiring must not move the calls.
///
/// The counts must agree exactly, in both directions: a record without its
/// call leaves that group's publishes unconfigured (they trap on the first
/// publish), and a call without its record names an address nobody outlined.
/// Both are refusals, not repairs.
static LogicalResult wireConfigureCalls(LLVM::LLVMFuncOp engine,
                                        ArrayRef<LLVM::LLVMFuncOp> entries) {
  // Collect before editing: the walk must not observe anything this edit adds.
  SmallVector<LLVM::CallOp> calls;
  engine.walk([&](LLVM::CallOp call) {
    if (std::optional<StringRef> callee = call.getCallee())
      if (*callee == kHmxExecConfigureFn)
        calls.push_back(call);
  });
  if (calls.size() != entries.size())
    return engine.emitError()
           << "engine '" << engine.getName() << "' holds " << calls.size()
           << " hexagon_runtime_hmx_exec_configure call(s) for "
           << entries.size()
           << " readout handoff record(s); the producer emits exactly one"
              " call per record, in the same order";

  OpBuilder builder(engine.getContext());
  auto i32Ty = builder.getI32Type();
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext(), 0);
  Location loc = engine.getLoc();
  for (unsigned k = 0; k < calls.size(); ++k) {
    LLVM::CallOp call = calls[k];
    LLVM::LLVMFuncOp entry = entries[k];
    if (call.getNumOperands() != 2)
      return call.emitError()
             << "hexagon_runtime_hmx_exec_configure takes"
                " (HmxReadoutFn, int32_t); a different arity here means the"
                " placeholder was not emitted by hmx-vector-readout";
    // The address crosses as the same i32 every other address crosses as on
    // this target (`asAddress`): the ABI takes a C function pointer, which is
    // 32-bit here.
    builder.setInsertionPoint(call);
    Value address = LLVM::AddressOfOp::create(
        builder, loc, ptrTy, FlatSymbolRefAttr::get(entry.getOperation()));
    Value asInt = LLVM::PtrToIntOp::create(builder, loc, i32Ty, address);
    call.setOperand(0, asInt);
  }
  return success();
}

/// Turn a short `publish` return into a trap.
///
/// This is not an optimisation guard; it is a correctness requirement, and the
/// runtime's own header says so in as many words: a dropped batch means that
/// destination region is never written, "so the kernel returns its own
/// uninitialized output -- a wrong answer, not a slower one", and there is
/// deliberately no silent inline fallback in the runtime for a caller to lean on.
/// The return value is the only signal there is, so ignoring it is a
/// silent-corruption bug waiting for a full ring.
///
/// WHAT IS EMITTED, AND WHY IT CANNOT BE REMOVED. The failure edge ends in
/// `llvm.intr.trap`. That is the whole mechanism, and it is the right one for a
/// reason worth stating because the obvious alternatives are worse:
///
///   * Silently continuing is the failure the ABI forbids. The kernel would
///     return its own uninitialised output -- a plausible-looking wrong answer.
///   * Calling back into the runtime to read the tail out inline would be the
///     documented *fallback*, but the ABI says it must not be silent, and there
///     is no entry point for it: `HmxVectorExecutor.h` is frozen and declares no
///     such function, so the compiler cannot invent one.
///   * Reporting through a diagnostic entry would need one too, for the same
///     reason.
///
/// A trap is unremovable by construction: it is an instruction with a side
/// effect on every path that reaches it, so no later pass can prove the block
/// dead. The branch that reaches it is the `publish` return compared against the
/// count that was asked for, so the edge is live exactly when the runtime would
/// have dropped a batch. `trap` is also the right severity here: the kernel's
/// output buffer is the caller's memory, the alternative is returning a wrong
/// result silently, and the runtime has already logged why at ERROR level
/// (HmxVectorExecutor.cpp:580,599).
///
/// The check is per call site rather than once per function because the two
/// publishes are different things: the in-loop group and the tail batch. A single
/// check on the last one would leave the in-loop drops unreported, and those are
/// the common case.
///
/// Ordering is load-bearing and matches the runtime: publish first, then check.
static LogicalResult checkPublishReturns(LLVM::LLVMFuncOp engine) {
  MLIRContext *context = engine.getContext();
  OpBuilder builder(context);
  Location loc = engine.getLoc();
  auto i32Ty = builder.getI32Type();

  // Collect before editing: inserting blocks invalidates a walk.
  SmallVector<LLVM::CallOp> publishes;
  engine.walk([&](LLVM::CallOp call) {
    if (std::optional<StringRef> callee = call.getCallee())
      if (*callee == kHmxExecPublishFn)
        publishes.push_back(call);
  });

  for (LLVM::CallOp call : publishes) {
    if (call->getNumResults() != 1 ||
        !call->getResult(0).getType().isInteger(32))
      return call.emitError()
             << "hexagon_runtime_hmx_exec_publish must return the number of "
                "batches it accepted (uint32_t); a void declaration here means "
                "a dropped batch cannot be detected, which is a wrong answer "
                "rather than a slower one";

    // The check is a BRANCH, and only an LLVM-dialect region can hold one: an
    // `scf.if`/`scf.for` body must stay structured, and inserting `llvm.cond_br`
    // plus two bare blocks into one makes the op unverifiable rather than slow.
    // So a publish still sitting inside structured control flow is refused here
    // instead of producing malformed IR.
    //
    // This is not a restriction the production pipeline can hit: HmxToLLVM runs
    // after `convert-scf-to-cf` (LinalgToLLVMPass.cpp:651 then :673), so every
    // publish is already in a plain LLVM block by the time this runs. It is a
    // guard for anyone driving the pass standalone.
    // `Block::getParent()` is the op that owns the region the block sits in, so
    // comparing it to the enclosing LLVMFuncOp answers "is this block directly in
    // a function body" -- i.e. not nested in an scf region.
    Region *owner = call->getBlock()->getParent();
    if (!isa_and_nonnull<LLVM::LLVMFuncOp>(owner->getParentOp()))
      return call.emitError()
             << "hexagon_runtime_hmx_exec_publish is inside structured control "
                "flow, so its return value cannot be branched on here. "
                "HmxToLLVM must run after convert-scf-to-cf, which is where the "
                "production pipeline places it";

    Block *block = call->getBlock();
    Block *continuation = block->splitBlock(call->getNextNode());
    Block *failure = new Block();
    engine.getBody().getBlocks().insert(continuation->getIterator(), failure);

    builder.setInsertionPointToEnd(block);
    Value accepted = call->getResult(0);
    Value wanted = LLVM::ConstantOp::create(
        builder, loc, i32Ty, builder.getI32IntegerAttr(kHmxExecPublishedBatches));
    Value short_ = LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ult,
                                        accepted, wanted);
    LLVM::CondBrOp::create(builder, loc, short_, failure, ValueRange{},
                           continuation, ValueRange{});

    builder.setInsertionPointToEnd(failure);
    // The trap is the guarantee. `llvm.intr.trap` is a real instruction on every
    // path that reaches it and has no side conditions an optimiser can discharge.
    LLVM::Trap::create(builder, loc);
    LLVM::BrOp::create(builder, loc, ValueRange{}, continuation);
  }
  return success();
}

/// Wire every handoff the producer pass recorded: entry point, `configure`, and
/// the `publish` check. A module with no record is untouched, which is what keeps
/// the default-off path byte-identical.
///
/// Every failure here is a refusal rather than a repair. A handoff that cannot be
/// wired exactly -- a missing kernel, an outlined function with a signature this
/// adapter does not model, a `publish` without a return value -- would otherwise
/// produce a kernel that hands the caller unwritten output while reporting
/// success, which is the same class of bug as a mis-read accumulator row.
static LogicalResult wireVectorReadout(ModuleOp moduleOp) {
  Attribute raw = moduleOp->getAttr(kHmxReadoutHandoffsAttr);
  if (!raw)
    return success();
  auto records = dyn_cast<ArrayAttr>(raw);
  if (!records)
    return moduleOp.emitError(
        "hmx.readout.handoffs must be an array of handoff records");

  OpBuilder builder(moduleOp.getContext());

  // One engine can now hold SEVERAL handoffs -- one per matmul group -- and
  // both the configure pairing and the publish check are per engine: the calls
  // are paired with this engine's records in program order, and the trap
  // rewrite must visit each publish exactly once. Records for one engine are
  // collected here in record order, which is the producer's group order.
  SmallVector<LLVM::LLVMFuncOp> engineOrder;
  DenseMap<LLVM::LLVMFuncOp, SmallVector<LLVM::LLVMFuncOp>> engineEntries;
  for (Attribute raw : records) {
    auto record = dyn_cast<DictionaryAttr>(raw);
    if (!record)
      return moduleOp.emitError("hmx.readout.handoffs entry must be a dictionary");
    auto engineName = record.getAs<StringAttr>(kHmxReadoutEngineField);
    auto workName = record.getAs<StringAttr>(kHmxReadoutWorkField);
    auto arAttr = record.getAs<TypeAttr>(kHmxReadoutArField);
    auto dstAttr = record.getAs<TypeAttr>(kHmxReadoutDstField);
    if (!engineName || !workName || !arAttr || !dstAttr)
      return moduleOp.emitError(
          "hmx.readout.handoffs entry needs engine, work, ar and dst");
    auto arType = dyn_cast<MemRefType>(arAttr.getValue());
    auto dstType = dyn_cast<MemRefType>(dstAttr.getValue());
    if (!arType || !dstType)
      return moduleOp.emitError(
          "hmx.readout.handoffs ar/dst must be memref types; the outlined "
          "read-out cannot be reconstructed from a tensor");

    // Both must be `llvm.func` and neither a declaration. This runs after
    // convert-func-to-llvm in the production pipeline, so the outlined read-out
    // is already one; a `func.func` here would mean this pass was asked to run
    // before that conversion, when `llvm.mlir.addressof` cannot name it at all.
    LLVM::LLVMFuncOp work =
        moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(workName.getValue());
    if (!work)
      return moduleOp.emitError()
             << "handoff names outlined read-out '" << workName.getValue()
             << "', which is not an llvm.func. HmxToLLVM must run after "
                "convert-func-to-llvm, and the read-out must have been "
                "converted with the rest of the module";
    if (work.isDeclaration())
      return work.emitError() << "outlined read-out '" << workName
                              << "' is a declaration; it has no body to run";

    // The adapter models exactly `(ar: memref, dst: memref, row0: index,
    // nrows: index)`. Anything else -- a different arity, a non-memref buffer,
    // a `!llvm.ptr` where a memref was -- would make the emitted call pass the
    // wrong words in the wrong order, which is a silent mis-read rather than a
    // crash, so the shape is checked instead of trusted.
    unsigned expanded = 3 + 2 * static_cast<unsigned>(arType.getRank()) +
                        3 + 2 * static_cast<unsigned>(dstType.getRank()) + 2;
    unsigned actual = work.getNumArguments();
    if (actual != expanded)
      return work.emitError()
             << "outlined read-out '" << workName.getValue() << "' has " << actual
             << " parameters; the vector-readout entry point models exactly "
                "(memref, memref, index, index) under the default "
                "convert-func-to-llvm convention, which is "
             << expanded
             << ". Refusing to guess which words are which buffer";

    LLVM::LLVMFuncOp engine = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(engineName);
    if (!engine || engine.isDeclaration())
      return moduleOp.emitError()
             << "handoff names engine '" << engineName
             << "', which is not an llvm.func definition";

    std::string entryName =
        (workName.getValue() + kHmxReadoutEntrySuffix).str();
    // The handoff is consumed here, so the record cannot be acted on twice if a
    // later pass ever re-runs this one: a second `configure` would re-store the
    // function and a second entry point would shadow the first.
    if (moduleOp.lookupSymbol(entryName))
      return moduleOp.emitError()
             << "vector-readout entry point '" << entryName
             << "' already exists; the handoff was already wired";

    LLVM::LLVMFuncOp entry = emitReadoutEntry(builder, moduleOp, work, arType,
                                              dstType, entryName);
    if (engineEntries.count(engine) == 0)
      engineOrder.push_back(engine);
    engineEntries[engine].push_back(entry);
  }

  for (LLVM::LLVMFuncOp engine : engineOrder) {
    if (failed(wireConfigureCalls(engine, engineEntries[engine])))
      return failure();
    if (failed(checkPublishReturns(engine)))
      return failure();
  }

  // The record is consumed, not left for a later reader. Nothing else may act on
  // it: the entry point now exists and the `configure` call has been emitted, so a
  // second consumer would re-configure the executor and shadow the entry point.
  // Erasing it also means the record cannot outlive what it describes -- if a
  // later pass rebuilt `__hmx_readout` without republishing, this pass would have
  // nothing to re-wire and would silently leave the kernel publishing to a
  // function nobody registered, rather than failing on a stale record.
  moduleOp->removeAttr(kHmxReadoutHandoffsAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// THREAD-ROLE CHANNEL (S3; ROADMAP1001 sections 3.2, 4.6)
//===----------------------------------------------------------------------===//
//
// ThreadRolePartition outlined the engine section into a private work function
// and rewrote the kernel into the producer side (submit/drain). Two things are
// still missing, and both need this pass's position after
// convert-func-to-llvm:
//
//   * the ENTRY POINT, `<engine>__hmx_section`, adapting the runtime's
//     `HmxSectionFn` (void(const HmxTileGroupDesc*, uint32_t)) onto the work
//     function -- the same adapter-with-addresses arrangement the read-out
//     entry uses, except the buffers' addresses come from the per-launch
//     table (HmxRoleHandoff.h), because the frozen 24-byte descriptor has no
//     address words;
//   * the DEPTH OBJECT, `<engine>__hmx_role_depth`, the exported uint32_t the
//     launch-side probe reads to bind with the ring depth the producer
//     derived from the tile-ring geometry.
//
// Plus one change to existing machinery: the work function must NOT get the
// per-kernel ensure/unlock pair. Its lock is the bound thread's lifetime lock
// (HmxRoleExecutor.cpp's roleThreadEntry calls ensure once at thread start and
// never unlocks), and an inserted unlock at the work function's exit would
// RELEASE that lifetime lock under the executor's feet -- the exact
// "per-kernel pairing" the lock migration's first half retires. The exemption
// is keyed on the handoff record's names, and verifyHmxLeafCallers gets the
// same set so the exemption is visible as an exemption, not as a missed pair.
//
// The work function also gets the LLVM fn attribute that carries the thread
// contract to the backend: `passthrough = ["hexagon_hmx"]`, which the LLVM IR
// translation turns into the string fn attribute upstream PR #222340's TTI
// hooks read (the S0 backport). It is set here, on the llvm.func, because the
// `hex.thread_role` marker on the func.func does not survive every conversion
// in between (the readout marker's lesson); CollapseAddressSpace, which
// rebuilds llvm.func, copies the passthrough attribute explicitly.

/// Build the section entry point: `void(<engine>__hmx_section)(ptr, i32)`.
///
/// What it forwards, and what it does not:
///
///   * `rowStart`/`rowCount` come out of descriptor words 1 and 2, zero
///     extended to the i64 the work function's `index` parameters became.
///   * the four buffer addresses come out of the per-launch table's words,
///     each rebuilt into the work function's exploded memref arguments
///     (`appendMemrefArgs` -- the same reconstruction the read-out entry
///     does for its two buffers).
///   * `slot` (word 0) is NOT forwarded: the work function derives the
///     crouton row from `rowStart` directly (one row per tile, so the row is
///     the tile index). Reading the word and doing nothing with it would be
///     a lie about the dependency -- the same rule the read-out entry
///     applies to its informational words.
///   * `count` is not used: the runtime calls the section once per
///     descriptor with count == 1 (HmxRoleExecutor.h's per-item discipline),
///     and the descriptor's own `rowCount` is the loop bound. The parameter
///     stays because the ABI carries it.
///
/// PUBLIC, unlike the read-out entry: the launch-side probe dlsym's this
/// symbol in the loaded module, so it must survive as an exported symbol.
static LLVM::LLVMFuncOp emitRoleEntry(OpBuilder &builder, ModuleOp moduleOp,
                                      LLVM::LLVMFuncOp work,
                                      MemRefType rowsType, MemRefType wtType,
                                      MemRefType biasType, MemRefType arType,
                                      StringRef name) {
  MLIRContext *context = moduleOp.getContext();
  auto i32Ty = builder.getI32Type();
  auto i64Ty = builder.getI64Type();
  auto ptrTy = LLVM::LLVMPointerType::get(context, 0);
  // The descriptor as a value: six i32 words in the frozen field order
  // (HmxRoleHandoff.h), read as one struct the same way the read-out entry
  // reads its batch -- the layout agreement is with the header both sides
  // include, and reading it as a struct states that.
  auto descTy = LLVM::LLVMStructType::getLiteral(
      context, SmallVector<Type>(kHmxRoleDescWords, i32Ty));

  LLVM::LLVMFunctionType fnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(context), {ptrTy, i32Ty}, /*isVarArg=*/false);
  OpBuilder funcBuilder = OpBuilder::atBlockEnd(moduleOp.getBody());
  LLVM::LLVMFuncOp entry =
      LLVM::LLVMFuncOp::create(funcBuilder, moduleOp.getLoc(), name, fnType);
  // Public on purpose -- see the function comment.
  Block *block = entry.addEntryBlock(funcBuilder);
  OpBuilder body = OpBuilder::atBlockEnd(block);
  Location loc = moduleOp.getLoc();

  Value desc = LLVM::LoadOp::create(body, loc, descTy, block->getArgument(0));
  auto word = [&](int64_t index) -> Value {
    return LLVM::ExtractValueOp::create(body, loc, i32Ty, desc,
                                        ArrayRef<int64_t>{index});
  };
  Value rowStart = word(kHmxRoleDescRowStart);
  Value rowCount = word(kHmxRoleDescRowCount);

  // The per-launch buffer table. Four loads per section call: four DDR
  // loads against an engine section's worth of work is noise, and the table
  // is the only per-launch channel the frozen descriptor leaves.
  auto tableAddr = [&](StringRef global) -> Value {
    LLVM::GlobalOp word = moduleOp.lookupSymbol<LLVM::GlobalOp>(global);
    assert(word && "ThreadRolePartition created the table words");
    Value slot = LLVM::AddressOfOp::create(body, loc, word);
    return LLVM::LoadOp::create(body, loc, i32Ty, slot);
  };

  SmallVector<Value> args;
  appendMemrefArgs(body, loc, rowsType, tableAddr(kHmxRoleRowsGlobal), args);
  appendMemrefArgs(body, loc, wtType, tableAddr(kHmxRoleWtGlobal), args);
  appendMemrefArgs(body, loc, biasType, tableAddr(kHmxRoleBiasGlobal), args);
  appendMemrefArgs(body, loc, arType, tableAddr(kHmxRoleArGlobal), args);
  args.push_back(LLVM::ZExtOp::create(body, loc, i64Ty, rowStart));
  args.push_back(LLVM::ZExtOp::create(body, loc, i64Ty, rowCount));

  LLVM::CallOp::create(body, loc, TypeRange{},
                       FlatSymbolRefAttr::get(work.getOperation()), args);
  LLVM::ReturnOp::create(body, loc, ValueRange{});
  return entry;
}

/// The thread-role channel's compiler half: entry point, depth object, the
/// LLVM thread-contract attribute. The ensure/unlock exemption is consumed
/// earlier -- the work-function names have to be known before the
/// engine-kernel collection runs (see runOnOperation), which is why the
/// sidecar is read twice: once for the names, once for the types.
static LogicalResult wireRoleChannel(ModuleOp moduleOp) {
  auto handoffs = moduleOp->getAttrOfType<ArrayAttr>(kHmxRoleHandoffsAttr);
  if (!handoffs || handoffs.empty())
    return success();

  OpBuilder builder(moduleOp.getContext());
  for (Attribute record : handoffs.getValue()) {
    auto dict = cast<DictionaryAttr>(record);
    auto engineName = dict.getAs<StringAttr>(kHmxRoleEngineField);
    auto workName = dict.getAs<StringAttr>(kHmxRoleWorkField);
    auto depthAttr = dict.getAs<IntegerAttr>(kHmxRoleDepthField);
    auto rowsTypeAttr = dict.getAs<TypeAttr>(kHmxRoleRowsField);
    auto wtTypeAttr = dict.getAs<TypeAttr>(kHmxRoleWtField);
    auto biasTypeAttr = dict.getAs<TypeAttr>(kHmxRoleBiasField);
    auto arTypeAttr = dict.getAs<TypeAttr>(kHmxRoleArField);
    if (!engineName || !workName || !depthAttr || !rowsTypeAttr ||
        !wtTypeAttr || !biasTypeAttr || !arTypeAttr)
      return moduleOp.emitError()
             << "hmx.role.handoffs record is missing a field; the producer "
                "pass and this pass disagree on the schema";

    auto rowsType = dyn_cast<MemRefType>(rowsTypeAttr.getValue());
    auto wtType = dyn_cast<MemRefType>(wtTypeAttr.getValue());
    auto biasType = dyn_cast<MemRefType>(biasTypeAttr.getValue());
    auto arType = dyn_cast<MemRefType>(arTypeAttr.getValue());
    if (!rowsType || !wtType || !biasType || !arType)
      return moduleOp.emitError()
             << "hmx.role.handoffs buffer fields are not memref types";

    LLVM::LLVMFuncOp work =
        moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(workName.getValue());
    if (!work)
      return moduleOp.emitError()
             << "handoff names work function '" << workName.getValue()
             << "', which is not an llvm.func; HmxToLLVM must run after "
                "convert-func-to-llvm";
    if (work.isDeclaration())
      return work.emitError() << "work function '" << workName
                              << "' is a declaration; it has no body to run";

    // The work function's signature, verified rather than trusted: four
    // memrefs then two i64s under the default convert-func-to-llvm
    // convention. A mismatch would pass the wrong words in the wrong order
    // -- a silent mis-read, not a crash.
    unsigned expanded = 0;
    for (MemRefType type : {rowsType, wtType, biasType, arType})
      expanded += 3 + 2 * static_cast<unsigned>(type.getRank());
    expanded += 2;
    if (work.getNumArguments() != expanded)
      return work.emitError()
             << "work function '" << workName.getValue() << "' has "
             << work.getNumArguments()
             << " parameters; the section entry models exactly (memref x4, "
                "index, index) under the default convert-func-to-llvm "
                "convention, which is "
             << expanded << ". Refusing to guess which words are which buffer";

    // The LLVM thread contract (see the section banner): passthrough is the
    // standard route to a string fn attribute, and CollapseAddressSpace
    // copies it when it rebuilds the function.
    SmallVector<Attribute> passthrough;
    if (auto existing = work->getAttrOfType<ArrayAttr>(
            work.getPassthroughAttrName()))
      passthrough.assign(existing.getValue().begin(),
                         existing.getValue().end());
    passthrough.push_back(builder.getStringAttr("hexagon_hmx"));
    work.setPassthroughAttr(builder.getArrayAttr(passthrough));

    // The exported channel symbols. Both names are built from the suffixes
    // in HmxRoleChannel.h -- the same header the runtime probe includes --
    // so the spellings cannot drift.
    std::string entryName =
        (engineName.getValue() + HMX_ROLE_SECTION_SUFFIX).str();
    if (moduleOp.lookupSymbol(entryName))
      return moduleOp.emitError()
             << "thread-role entry point '" << entryName
             << "' already exists; the handoff was already wired";
    LLVM::LLVMFuncOp entry = emitRoleEntry(builder, moduleOp, work, rowsType,
                                           wtType, biasType, arType,
                                           entryName);
    (void)entry;

    std::string depthName =
        (engineName.getValue() + HMX_ROLE_DEPTH_SUFFIX).str();
    if (moduleOp.lookupSymbol(depthName))
      return moduleOp.emitError()
             << "thread-role depth object '" << depthName
             << "' already exists; the handoff was already wired";
    // External linkage AND an initializer: a defined, exported object --
    // dlsym finds it in the loaded module, and the probe reads the depth
    // the producer derived from the tile-ring geometry (the ring is as deep
    // as the tile count; see ThreadRolePartition's emission).
    OpBuilder globalBuilder = OpBuilder::atBlockEnd(moduleOp.getBody());
    LLVM::GlobalOp::create(
        globalBuilder, moduleOp.getLoc(), builder.getI32Type(),
        /*isConstant=*/true, LLVM::Linkage::External, depthName,
        builder.getIntegerAttr(builder.getI32Type(),
                               static_cast<int32_t>(depthAttr.getInt())),
        /*alignment=*/0, /*addrSpace=*/0, /*dsoLocal=*/false,
        /*thread_local=*/false);

    // The ensure/unlock exemption (the section banner): the work function's
    // lock is the bound thread's lifetime lock. Recorded for
    // runOnOperation's collection, which already skipped these names.
  }

  // Consumed, like the read-out record: a second consumer would emit a
  // second entry point and shadow the first.
  moduleOp->removeAttr(kHmxRoleHandoffsAttr);
  return success();
}

/// Bring the engine up before every HMX accumulator sequence and release it
/// after. Without the ensure calls the DSP aborts as soon as an HMX instruction
/// executes: a kernel whose HMX ops never go through a runtime allocation has
/// nothing else that would power the engine up. Without the unlock calls a
/// second thread's NON_SHARED lock blocks forever (qurt_hmx.h: the waiter
/// suspends until the unit is available), hanging the device as soon as two
/// threads run HMX kernels. The lock is one pairing per kernel: one ensure at
/// entry, one unlock before each return.
///
/// `engineKernels` is collected before the conversion (see runOnOperation),
/// keyed by symbol name: names are unique in a module's symbol table and this
/// pass never renames a function, so the key identifies the same function after
/// the conversion, whereas an `Operation *` held across it could dangle if a
/// future lowering rebuilds a function. Every name still present matches (a
/// miss would need a rename, which does not happen here); a stale name whose
/// function is gone inserts nowhere, and that function executes nothing.
static void ensureHmxEngine(ModuleOp moduleOp,
                            const llvm::StringSet<> &engineKernels) {
  // Both function forms: in the full LinalgToLLVM pipeline func-to-llvm runs
  // before this pass, so kernels are llvm.func; in the standalone lit tests
  // they are still func.func. The engine precondition applies either way.
  SmallVector<Operation *> kernels;
  moduleOp.walk([&](LLVM::LLVMFuncOp fn) {
    if (!fn.isDeclaration() && engineKernels.count(fn.getName()))
      kernels.push_back(fn);
  });
  moduleOp.walk([&](func::FuncOp fn) {
    if (!fn.isDeclaration() && engineKernels.count(fn.getName()))
      kernels.push_back(fn);
  });
  if (kernels.empty())
    return;
  OpBuilder builder(moduleOp.getContext());
  FailureOr<LLVM::LLVMFuncOp> ensureFn = LLVM::lookupOrCreateFn(
      builder, moduleOp, kHmxEnsureFn, /*paramTypes=*/ArrayRef<Type>{},
      LLVM::LLVMVoidType::get(moduleOp.getContext()));
  if (failed(ensureFn))
    return;
  FailureOr<LLVM::LLVMFuncOp> unlockFn = LLVM::lookupOrCreateFn(
      builder, moduleOp, kHmxUnlockFn, /*paramTypes=*/ArrayRef<Type>{},
      LLVM::LLVMVoidType::get(moduleOp.getContext()));
  if (failed(unlockFn))
    return;
  FlatSymbolRefAttr ensureCallee =
      FlatSymbolRefAttr::get(ensureFn->getOperation());
  FlatSymbolRefAttr unlockCallee =
      FlatSymbolRefAttr::get(unlockFn->getOperation());
  for (Operation *fn : kernels) {
    builder.setInsertionPointToStart(&fn->getRegion(0).front());
    LLVM::CallOp::create(builder, fn->getLoc(), TypeRange{}, ensureCallee,
                         ValueRange{});
    // Collect first: inserting while walking invalidates the walk. Both
    // return forms: llvm.return in llvm.func, func.return in func.func.
    SmallVector<Operation *> returns;
    fn->walk([&](Operation *op) {
      if (isa<LLVM::ReturnOp, func::ReturnOp>(op))
        returns.push_back(op);
    });
    for (Operation *ret : returns) {
      builder.setInsertionPoint(ret);
      LLVM::CallOp::create(builder, fn->getLoc(), TypeRange{}, unlockCallee,
                           ValueRange{});
    }
  }
}

/// The runtime ABI is i32, so an index operand has to come down to that.
Value toI32(ConversionPatternRewriter &rewriter, Location loc, Value v) {
  if (v.getType().isInteger(32))
    return v;
  return LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), v);
}

/// An i32 constant materialised at `loc`. The eight pack/unpack converters
/// each declared this as a local lambda capturing their local `i32Ty`; one
/// named helper replaces the eight copies. `rewriter.getI32Type()` is the
/// same interned type those lambdas captured, so every call site emits the
/// IR it did before.
Value dimCst(ConversionPatternRewriter &rewriter, Location loc, int64_t v) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(v))
      .getResult();
}

/// A buffer argument is an address: the integer image of the memref's aligned
/// pointer *plus the descriptor's offset*. The offset is not optional: a pack
/// source is often a view into the middle of a buffer (the K block of an
/// attention matmul is a `reinterpret_cast` with a per-iteration offset), and
/// dropping it makes every iteration pack the first block -- measured as a
/// wrong f16 attention result that grows with the number of K blocks.
/// `elemBytes` is the size of one element of the *viewed* array, so callers
/// pass the crouton size for crouton arrays and the element size otherwise.
/// A memref descriptor carries no proof that an arbitrary runtime pointer is
/// non-null or aligned for an HMX leaf. This boundary therefore remains an
/// explicit runtime/allocator precondition; the compiler does not invent a
/// null check or silently normalize an unknown alignment here.
Value asAddress(ConversionPatternRewriter &rewriter, Location loc,
                Value memrefDesc, int64_t elemBytes) {
  auto i32Ty = rewriter.getI32Type();
  MemRefDescriptor desc(memrefDesc);
  Value addr = LLVM::PtrToIntOp::create(rewriter, loc, i32Ty,
                                        desc.alignedPtr(rewriter, loc));
  Value bytes = LLVM::MulOp::create(
      rewriter, loc, i32Ty, toI32(rewriter, loc, desc.offset(rewriter, loc)),
      LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                               rewriter.getI32IntegerAttr(elemBytes)));
  return LLVM::AddOp::create(rewriter, loc, i32Ty, addr, bytes);
}

/// The stride, in croutons, of tile-grid dimension `dim` (0 = K/M tiles, 1 = N
/// tiles) of a crouton array, read from the operand's own layout. A *slice* of
/// a larger crouton array -- a `memref.subview` over the N tiles, which is how
/// a whole resident weight is read one N block at a time -- keeps dim 0's
/// stride at the whole array's `N/32` croutons, not at its own `dimSize(1)`;
/// addressing it with `dimSize(1)` walks into the wrong K tile for every row
/// past the first. A dense array returns `dimSize(1)` (dim 0) and 1 (dim 1) --
/// the exact constants the old arithmetic hardcoded -- and a dynamic,
/// non-positive or non-crouton-aligned stride falls back to that dense contract
/// rather than guessing (same policy as `rowStride`).
static int64_t croutonTileStride(MemRefType type, int64_t dim) {
  int64_t dense = dim == 0 ? type.getDimSize(1) : 1;
  SmallVector<int64_t, 5> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      dim >= static_cast<int64_t>(strides.size()))
    return dense;
  int64_t stride = strides[dim];
  if (ShapedType::isDynamic(stride) || stride <= 0 ||
      stride % layout::kCroutonElements != 0)
    return dense;
  return stride / layout::kCroutonElements;
}

/// The address of crouton `(row, col)` of a crouton array: the array's base
/// plus
/// `(row * stride(0) + col * stride(1))` croutons. The tile-grid strides come
/// from the operand's layout, so a crouton array sliced out of a larger one is
/// addressed at its real position; a dense array emits exactly the old
/// `(row * dimSize(1) + col) * 2048` byte arithmetic (the column step stays 1,
/// so the old `+ col` form is kept verbatim).
Value croutonAddr(ConversionPatternRewriter &rewriter, Location loc,
                  Value memrefDesc, MemRefType type, Value row, Value col,
                  int64_t elemBytes) {
  auto i32Ty = rewriter.getI32Type();
  // The descriptor's offset is counted in *elements*, not in croutons, so the
  // base uses the element size while the tile term below is in croutons.
  Value base = asAddress(rewriter, loc, memrefDesc, elemBytes);
  auto cst = [&](int64_t v) -> Value {
    return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                    rewriter.getI32IntegerAttr(v));
  };
  // Materialise the row-stride constant before the row index's cast so the
  // dense emission order (constant, trunc, mul) is exactly the old one.
  Value rowStride = cst(croutonTileStride(type, 0));
  Value rowOff = LLVM::MulOp::create(rewriter, loc, i32Ty,
                                     toI32(rewriter, loc, row), rowStride);
  // The column step is 1 for every crouton array the pipeline builds, so keep
  // the old `row*cols + col` form verbatim; a wider step (a layout that packs
  // several N tiles per crouton stride) scales the column index instead.
  Value colIdx = toI32(rewriter, loc, col);
  int64_t colStep = croutonTileStride(type, 1);
  if (colStep != 1)
    colIdx = LLVM::MulOp::create(rewriter, loc, i32Ty, colIdx, cst(colStep));
  Value tile = LLVM::AddOp::create(rewriter, loc, i32Ty, rowOff, colIdx);
  Value bytes =
      LLVM::MulOp::create(rewriter, loc, i32Ty, tile,
                          LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                                   rewriter.getI32IntegerAttr(
                                                       layout::kCroutonBytes)));
  return LLVM::AddOp::create(rewriter, loc, i32Ty, base, bytes);
}

/// The row stride of a rank-2 row-major buffer, in elements. Both a pack source
/// and an unpack destination are addressed row-by-row by the leaf using this
/// stride: the width for a dense buffer, but larger when the buffer is one
/// N-tile of a wider matrix (`memref<M x BN, strided<[N, 1]>>` with `N > BN`)
/// -- the K blocks of an attention matmul (source side) and the destination of
/// an N-split store (destination side). Threading the width there instead of
/// the stride reads/writes the wrong columns of every row past the first.
///
/// `width` is the already-materialised column-count value, returned unchanged
/// when the buffer is dense (stride(rank-2) == width) or has an identity
/// layout, so the dense path emits exactly the IR it did before. A dynamic
/// stride cannot be read from the TYPE here, so it is read from the RUNTIME
/// descriptor instead (MemRefDescriptor::stride -- the same struct read
/// `asAddress` already performs): the width substitution that used to stand in
/// there is only valid for dense sources and silently shears a raw strided
/// view (see the TRACEABILITY note below).
///
/// A STATIC non-inner-contiguous layout is the case that used to be silent, and
/// it is not a stride question at all. The leaf walks ROWS, so no `row_stride`
/// value makes a column-major buffer come out right: the address arithmetic is
/// transposed relative to the writer. `strided<[1, N]>` puts element (i,j) at
/// `i + N*j`, so handing the leaf the declared stride(rank-2)=1 makes it store
/// at `i + j` -- element `(i+j, 0)` where it meant `(i, j)`. That is a silent
/// wrong-value store, not a crash, and nothing downstream can notice it.
///
/// So this fails closed, which is what `verifyDiagnosticRowMajor` already does
/// on the diagnostic tail path (`HmxPartitionPass.cpp:1175`, requires
/// `strides[1] == 1`) and what the pack-source tail leaf already does
/// (`test/Conversion/HmxToLLVM/hmx-tail-layout-reject.mlir`). The ordinary
/// full-tile path had no such gate: the destination went straight from the op
/// into `rowStride` with nothing in between.
///
/// A DYNAMIC outer stride is NOT a guess and must not be gated. 2026-10-01: a
/// first attempt here rejected `ShapedType::isDynamic(stride)` as "an unreadable
/// stride cannot be guessed", and the gate immediately failed **12** lit tests
/// -- which is how the mistake became visible. The memrefs that broke are
/// `memref<64x64xf16, strided<[?, ?], offset: ?>>` weight-RESIDENT source views
/// (`hmx-weight-resident-pipeline.mlir` and siblings): a resident weight is a
/// view into a packed buffer whose stride only resolves at run time, so the
/// answer has to be a `Value`, not a constant. 2026-10-09: substituting
/// `width` for that Value was the wrong Value -- correct for the dense
/// resident contract, a silent shear for any raw view whose real stride is
/// wider than its width (measured on device object code, below). Extracting
/// the runtime stride keeps those 12 tests' semantics (their stride resolves
/// to the same number at run time) while making the raw-view branch correct.
///
/// TRACEABILITY: rowStride dynamic-stride extraction
///   mechanism: bufferization's branch merge (dense edge-temp gather vs raw
///     strided view) casts both sources to `strided<[?, ?]>`; memref.cast
///     erases the stride from the TYPE only -- the runtime descriptor still
///     carries it, so MemRefDescriptor::stride recovers the truth per branch.
///   measurement: 2026-10-09 device object code (mm_user.o, the binary the
///     measured launch ran): pack_weight called with src = raw B + n0 and
///     src_stride = the slice width (r4 = #0x200) while B's rows are the
///     whole N apart; the leaf addresses rows at `mpyi(row, src_stride)`
///     (libhmxapi.a) -> sheared weight; the packed crouton feeds hmx.mma
///     (same SSA value). Pinned host-side by
///     test/Conversion/HmxToLLVM/pack-dynamic-stride.mlir.
///   shape set: n/a (layout, not shape).
///   failure mode: before the fix, silent wrong values; after, each branch
///     passes its own runtime stride, and the static paths are bit-identical.
///
/// The genuinely unknown case is narrower and is still open, NOT fixed here: a
/// dynamic INNER stride means the leaf's contiguous inner run is not
/// contiguous, which the leaf cannot express at all. No such memref was found
/// reachable (every dynamic-stride memref in the suite has an identity or
/// unit-inner layout), so there is no evidence to gate on and nothing was
/// changed. Recorded rather than guessed at -- see
/// docs/hmx/rowstride-inner-contiguity-2026-10-01.md.
// TRACEABILITY: rowStride inner-contiguity gate
//   mechanism: the leaf addresses row-by-row, so a transposed layout cannot be
//     expressed by any row_stride value; only the inner stride can rule it out,
//     and only when it is statically known.
//   measurement: found by lowering a probe and reading the emitted leaf call --
//     test/Conversion/HmxToLLVM/unpack-dst-non-contiguous-reject.mlir.
//     `strided<[1, 32]>` before the gate: rc=0, no diagnostic,
//     row_stride=1, i.e. a silent store to element (i+j, 0) where (i, j) was
//     meant. After the gate: an error.
//   shape set: n/a (layout, not shape).
//   workload representativeness: n/a.
//   failure mode: reported as an error before lowering, never silently
//     mis-addressed. Same "degrade to unknown, not to wrong" shape as
//     kMaxDiagnosticNTile and kMaxProvenanceRounds.
//
// WHY THIS BUG SURVIVED SO LONG, and the generalisable lesson (2026-10-01).
// `verifyStaticTailLayout(..., rowMajor = true)` in this same file ALREADY
// refuses a non-unit inner stride: `if (strides.back() != 1) return
// emitError() << ... << " tail leaf requires unit inner stride"`. So on the TAIL
// path this gate is redundant -- and reading that function gives the false
// impression that the invariant is covered. It is covered on the tail path and
// nowhere else. The ordinary full-tile path had no inner-stride check at all,
// which is exactly where the silent store lived.
//
// The lesson worth keeping: "there is a check for this somewhere in the file" is
// not the same claim as "this call site is checked", and the two diverge
// silently whenever coverage is per-path rather than per-function. When adding a
// guard, enumerate the PATHS that reach it, not the functions that contain
// similar-looking code.
//
// SCOPE, measured 2026-10-01 by an independent audit -- three surfaces reach
// this function, and only ONE of them can actually be a non-contiguous buffer:
//   * unpack destination  -- REACHABLE, and the case that was silent. Pinned by
//     test/Conversion/HmxToLLVM/unpack-dst-non-contiguous-reject.mlir.
//   * unpack f32 residual -- REACHABLE (the f32 tail lowering). Pinned by
//     test/Conversion/HmxToLLVM/rowstride-gate-coverage.mlir.
//   * hmx.stage source    -- REACHABLE. Pinned there too, in both directions.
//   * pack source (act and weight, four call sites) -- NOT REACHABLE. The
//     hmx.pack_act / hmx.pack_weight op verifier rejects an overlapping-row
//     source first ("src row stride 1 is smaller than its N columns"), so
//     conversion never starts. The checks at those call sites are defence in
//     depth. Do not write a test for them expecting the gate to fire -- the
//     audit measured that it cannot; a test would have to assert the op
//     verifier's message instead.
LogicalResult rowStride(ConversionPatternRewriter &rewriter, Location loc,
                        MemRefType type, Value src, Value width,
                        Operation *anchor, Value &out) {
  SmallVector<int64_t, 2> strides;
  int64_t offset;
  int64_t widthC = type.getDimSize(type.getRank() - 1);
  if (succeeded(type.getStridesAndOffset(strides, offset)) &&
      strides.size() >= 2) {
    int64_t inner = strides[strides.size() - 1];
    int64_t stride = strides[strides.size() - 2];
    if (!ShapedType::isDynamic(inner) && inner != 1)
      return anchor->emitError()
             << "HMX leaf requires an inner-contiguous buffer; got inner "
                "stride "
             << inner << " on a rank-" << type.getRank()
             << " memref. The leaf walks rows, so a transposed layout cannot be "
                "addressed by any row stride -- passing one would store to the "
                "wrong elements silently.";
    if (ShapedType::isDynamic(stride)) {
      // A dynamic outer stride is the type-erased form of a REAL runtime
      // stride: reconciling the gather branch (dense edge temp) with the
      // raw-view branch casts both into `strided<[?, ?]>`, and the width
      // fallback below reads the wrong row distance for the raw branch --
      // the leaf then walks `B[n0 + width*r + c]` where the source row is
      // wider than `width`, i.e. a sheared matrix (measured on the device
      // object code 2026-10-09: pack_weight src = raw B + n0, stride = the
      // slice width, true stride = the whole N). The static type lost the
      // stride, but the runtime descriptor still carries it, so take it
      // from there instead of substituting an unrelated number -- the same
      // read `asAddress` already does on these operands (MemRefDescriptor),
      // one step less strict than the inner gate because the value IS
      // recoverable.
      MemRefDescriptor desc(src);
      out = toI32(rewriter, loc,
                  desc.stride(rewriter, loc, type.getRank() - 2));
      return success();
    }
    if (stride != widthC) {
      out = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(stride));
      return success();
    }
  }
  out = width;
  return success();
}

/// The runtime reads buffer addresses as i32. Reserve the widest supported
/// unpack element (f32) when bounding a diagnostic N-tile coordinate, so both
/// its logical-column and byte-offset products remain representable by the
/// signed i32 ABI before lowering starts.
// TRACEABILITY: kMaxDiagnosticNTile
//   mechanism: the runtime reads buffer addresses as i32, so a diagnostic
//     N-tile coordinate must keep both its logical-column and byte-offset
//     products representable in signed i32. The bound is therefore derived
//     from that ABI and the widest supported unpack element (f32), not chosen.
//   measurement: none needed -- it is arithmetic on the ABI.
//   shape set: n/a.
//   workload representativeness: n/a.
//   failure mode: exceeding it is reported as an error before lowering, never
//     silently truncated. Same "degrade to unknown, not to wrong" shape as
//     kMaxProvenanceRounds.
static constexpr int64_t kMaxDiagnosticNTile =
    std::numeric_limits<int32_t>::max() /
    (layout::kTileEdge * /*max element bytes=*/4);

static LogicalResult reportDiagnosticNTileOverflow(Operation *op) {
  return op->emitError(
      "diagnostic n_tile exceeds the signed i32 read-out ABI limit");
}

static LogicalResult verifyDiagnosticNTileUpperBound(Operation *op,
                                                     int64_t nTile) {
  if (nTile > kMaxDiagnosticNTile)
    return reportDiagnosticNTileOverflow(op);
  return success();
}

/// `n_tile` changes the read-out ABI selected by HmxToLLVM. It is therefore
/// diagnostic input, not an ordinary free-form discardable attribute. The
/// producer copies the source matmul's UnitAttr marker to the module;
/// standalone hmx-partition does the same before erasing that matmul. Validate
/// the complete module contract before partial conversion so an unauthorized
/// attribute can never reach a lowering pattern.
static LogicalResult verifyDiagnosticNTileContract(ModuleOp module) {
  if (module->hasAttr(kHmxDiagnosticTailAttr) &&
      !isHmxDiagnosticTailMarker(module.getOperation()))
    return module.emitError(
        "hmx.diagnostic_tail_partition must be a unit attribute");

  const bool marked = isHmxDiagnosticTailMarker(module.getOperation());
  LogicalResult result = success();
  module.walk([&](Operation *op) {
    Attribute raw = op->getAttr(kHmxDiagnosticNTileAttr);
    if (!raw)
      return;
    if (!isa<UnpackAccOp, UnpackAccF32Op>(op)) {
      result =
          op->emitError("diagnostic n_tile is only valid on hmx.unpack_acc and "
                        "hmx.unpack_acc_f32");
      return;
    }
    if (!marked) {
      result = op->emitError("diagnostic n_tile requires module marker "
                             "hmx.diagnostic_tail_partition");
      return;
    }
    auto integer = dyn_cast<IntegerAttr>(raw);
    if (!integer || !integer.getType().isSignlessInteger(64) ||
        integer.getInt() < 0) {
      result = op->emitError(
          "diagnostic n_tile must be a non-negative signless i64");
      return;
    }
    if (failed(verifyDiagnosticNTileUpperBound(op, integer.getInt()))) {
      result = failure();
      return;
    }
  });
  return result;
}

static std::optional<int64_t> readDiagnosticNTile(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(kHmxDiagnosticNTileAttr))
    return attr.getInt();
  return std::nullopt;
}

/// Return the local width of the physical output tile selected by the
/// diagnostic N-tile coordinate. The original ranked descriptor remains the
/// operand; only the runtime leaf's width is narrowed.
static FailureOr<int64_t> diagnosticTileColumns(Operation *op, MemRefType type,
                                                std::optional<int64_t> nTile,
                                                StringRef name) {
  int64_t total = type.getDimSize(1);
  if (!nTile)
    return total;
  if (*nTile < 0)
    return op->emitError() << name << " n_tile " << *nTile
                           << " is outside the logical N extent " << total;
  if (failed(verifyDiagnosticNTileUpperBound(op, *nTile)))
    return failure();
  int64_t begin;
  if (llvm::MulOverflow(*nTile, layout::kTileEdge, begin))
    return reportDiagnosticNTileOverflow(op);
  if (begin >= total)
    return op->emitError() << name << " n_tile " << *nTile
                           << " is outside the logical N extent " << total;
  return std::min<int64_t>(layout::kTileEdge, total - begin);
}

/// Add the explicit physical N-tile byte displacement to a descriptor-derived
/// address. This deliberately uses the original ranked descriptor rather than
/// a memref view: the backend's final memref-to-LLVM conversion must not be
/// relied on to compose a parent descriptor offset with a later subview.
static FailureOr<Value>
addDiagnosticNTileOffset(Operation *op, ConversionPatternRewriter &rewriter,
                         Location loc, Value base, std::optional<int64_t> nTile,
                         int64_t elemBytes) {
  if (!nTile)
    return base;
  if (failed(verifyDiagnosticNTileUpperBound(op, *nTile)))
    return failure();

  int64_t elementOffset;
  if (llvm::MulOverflow(*nTile, layout::kTileEdge, elementOffset))
    return reportDiagnosticNTileOverflow(op);
  int64_t byteOffset;
  if (llvm::MulOverflow(elementOffset, elemBytes, byteOffset) ||
      byteOffset > std::numeric_limits<int32_t>::max())
    return reportDiagnosticNTileOverflow(op);

  auto i32Ty = rewriter.getI32Type();
  return LLVM::AddOp::create(
             rewriter, loc, i32Ty, base,
             LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(byteOffset)))
      .getResult();
}

/// The tail ABI has no descriptor-to-leaf reconstruction. A tail call is
/// therefore admitted only when every address/stride it needs is statically
/// known. The full-tile lowering keeps its historical width fallback; silently
/// applying that fallback to a partial tile would turn an unknown layout into
/// a wrong address rather than a refusal.
static LogicalResult verifyStaticTailLayout(Operation *op, MemRefType type,
                                            StringRef name, bool rowMajor) {
  if (!type.hasStaticShape() || type.getRank() < 2 ||
      (rowMajor && type.getRank() != 2))
    return op->emitError() << name
                           << " tail leaf requires a static rank>=2 memref";

  SmallVector<int64_t, 8> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return op->emitError() << name << " tail leaf requires static strides";
  for (int64_t stride : strides)
    if (ShapedType::isDynamic(stride) || stride <= 0)
      return op->emitError() << name << " tail leaf requires static strides";

  if (!rowMajor)
    return success();
  if (strides.back() != 1)
    return op->emitError() << name << " tail leaf requires unit inner stride";
  if (strides[strides.size() - 2] < type.getDimSize(type.getRank() - 1))
    return op->emitError() << name
                           << " tail leaf row stride overlaps logical rows";
  return success();
}

/// The runtime's `AddrSpace` for a memref: the memref's own memory space (0 =
/// DDR, 1 = VTCM), which is exactly the enum. `fallback` is used when the
/// memref has no space attribute, so the lowering never silently invents a
/// space.
static int64_t memrefAddressSpace(MemRefType type, int64_t fallback) {
  if (auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace()))
    return space.getInt();
  return fallback;
}

/// The DMA 2D start entry the `hmx.stage` op lowers to. The signature is the
/// one DMAToLLVMPass declares for the same symbol: the pointer arguments are
/// real `!llvm.ptr`s (the host data layout widens them), while the address
/// spaces, the two bypass flags, `isOrdered` and the cache allocation policy
/// travel as i32.
///
/// Spelled out here rather than looked up: the DMA ABI is DMAToLLVMPass's
/// contract, and HmxLeafSignatures.h deliberately has no row for it so the
/// leaf table never becomes a second opinion about the DMA signature.
static FailureOr<LLVM::LLVMFuncOp>
getDma2DStartLeaf(ModuleOp module, ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(context);
  auto i32Ty = rewriter.getI32Type();
  return LLVM::lookupOrCreateFn(rewriter, module, getStageDma2DStartFnName(),
                                {ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty,
                                 i32Ty, i32Ty, i32Ty, i32Ty, i32Ty, ptrTy},
                                i32Ty);
}

/// The DMA wait entry `hmx.await` lowers to: `void dma_wait(i32 token)`. Same
/// reason as getDma2DStartLeaf for not consulting the leaf table.
static FailureOr<LLVM::LLVMFuncOp>
getDmaWaitLeaf(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto voidTy = LLVM::LLVMVoidType::get(module->getContext());
  return LLVM::lookupOrCreateFn(rewriter, module, getAwaitDmaWaitFnName(),
                                {rewriter.getI32Type()}, voidTy);
}

/// Emit a runtime leaf call and bind the op's result. The memref-form
/// pack/unpack ops are DPS and return the buffer they wrote (the interface
/// plan's C3), so the result IS the `dst` operand's address: the replacement is
/// the converted `dst` descriptor itself -- no copy, no descriptor rebuild,
/// zero cost. An op with no result (the pre-DPS form) just erases.
static void replaceWithLeafCall(ConversionPatternRewriter &rewriter,
                                Location loc, Operation *op, Value result,
                                LLVM::LLVMFuncOp fn, ValueRange args) {
  LLVM::CallOp::create(rewriter, loc, TypeRange{},
                       FlatSymbolRefAttr::get(fn.getOperation()), args);
  if (op->getNumResults() == 0)
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, result);
}

/// `hmx.mma` -> `hmx_mma_f16(act_at(m,k), wt_at(n,k), n_croutons)`. The weight
/// crouton grid is `[Nt, Kt, ...]` (logical `[N,K]`, K contiguous in dim1, see
/// `docs/hmx/hmx-weight-layout-plan.md` §0), so its tile index pair is
/// `(n_tile, k_tile)`
/// -- dim0 is N, dim1 is K -- not the `(k,n)` of the activation's `[Mt,Kt]`.
struct LowerMma : public ConvertOpToLLVMPattern<MmaOp> {
  using ConvertOpToLLVMPattern<MmaOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    auto fn = getVoidLeaf(module, getMmaF16FnName(), rewriter);
    if (failed(fn))
      return failure();

    auto actType = cast<MemRefType>(op.getAct().getType());
    auto wtType = cast<MemRefType>(op.getWt().getType());

    Value actAddr =
        croutonAddr(rewriter, loc, adaptor.getAct(), actType, adaptor.getM(),
                    adaptor.getK(), actType.getElementTypeBitWidth() / 8);
    // weight grid is [Nt, Kt]: dim0 = N, dim1 = K (layout A), so row = n_tile,
    // col = k_tile.
    Value wtAddr =
        croutonAddr(rewriter, loc, adaptor.getWt(), wtType, adaptor.getN(),
                    adaptor.getK(), wtType.getElementTypeBitWidth() / 8);
    Value nCroutons = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(op.getNCroutons()));

    SmallVector<Value> args{actAddr, wtAddr, nCroutons};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*fn).getOperation()), args);
    return success();
  }
};

/// `hmx.acc_read` -> `hmx_bias_load_f16(bias, set)` then
/// `hmx_acc_store_f16(dst, set)`. The bias registers have to be loaded for the
/// selected set before the read-out, and that pairing is this op's contract.
struct LowerAccRead : public ConvertOpToLLVMPattern<AccReadOp> {
  using ConvertOpToLLVMPattern<AccReadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AccReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    auto loadFn = getVoidLeaf(module, getBiasLoadF16FnName(), rewriter);
    auto storeFn = getVoidLeaf(module, getAccStoreF16FnName(), rewriter);
    if (failed(loadFn) || failed(storeFn))
      return failure();

    Value set = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(op.getBiasSet()));
    // The bias block is an I8 memref, so the offset is already in bytes.
    Value bias = asAddress(rewriter, loc, adaptor.getBias(), 1);
    auto dstType = cast<MemRefType>(op.getDst().getType());
    Value dst =
        croutonAddr(rewriter, loc, adaptor.getDst(), dstType, adaptor.getM(),
                    adaptor.getN(), dstType.getElementTypeBitWidth() / 8);

    LLVM::CallOp::create(rewriter, loc, TypeRange{},
                         FlatSymbolRefAttr::get((*loadFn).getOperation()),
                         SmallVector<Value>{bias, set});
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*storeFn).getOperation()),
        SmallVector<Value>{dst, set});
    return success();
  }
};

/// `hmx.bias_init` -> `hmx_bias_init_unit_f16(bias)`.
struct LowerBiasInit : public ConvertOpToLLVMPattern<BiasInitOp> {
  using ConvertOpToLLVMPattern<BiasInitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BiasInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fn = getVoidLeaf(module, getBiasInitUnitF16FnName(), rewriter);
    if (failed(fn))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*fn).getOperation()),
        SmallVector<Value>{asAddress(rewriter, loc, adaptor.getBias(), 1)});
    return success();
  }
};

/// `hmx.acc_clear` -> `hmx_acc_clear_f16()`.
struct LowerAccClear : public ConvertOpToLLVMPattern<AccClearOp> {
  using ConvertOpToLLVMPattern<AccClearOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AccClearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fn = getVoidLeaf(module, getAccClearF16FnName(), rewriter);
    if (failed(fn))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*fn).getOperation()),
        ValueRange{});
    return success();
  }
};

/// The runtime leaf family a pack source's element type selects: every form
/// of the call -- the single block, the ranged `_bulk` form, the bounds-safe
/// `tail` form, and (weight only) the `_T` transposed-source twins -- exists
/// once per admitted source dtype. One struct is one family; the functions
/// below are the dtype -> family tables.
struct PackActLeafFamily {
  std::string single, bulk, tail;
};

/// `hmx.pack_act`'s dtype -> leaf-family table, and the only place its
/// lowering answers "which leaf family does this source dtype read through".
/// f16 packs directly; an f32 source is quantised to the engine's fp16 inside
/// the pack (HMXAPI.h), which is why f32 is a twin leaf family rather than a
/// narrowing op ahead of the pack.
///
/// A source element type outside the admitted set -- {f16, f32} today;
/// `dtype::isAdmittedFloat` (HmxDType.h) is the predicate and the ODS source
/// constraint (HmxOps.td) is its other spelling -- is a *compile error*,
/// never a silent f16 fallback. Why loud and not a fallback: the f16 leaf
/// takes any buffer address and walks it as f16, so an f32- or bf16-typed
/// source handed to it is read as f16 -- wrong values, right exit code, no
/// diagnostic anywhere. That is the same structural trap as the stride-family
/// bugs (pack `src_stride` / unpack `dst_stride`: the leaf takes the number
/// and walks), and it fails closed the same way: degrade to unknown, not to
/// wrong.
///
/// Extending the admitted set (bf16, fp8, ...) means extending the three
/// spellings together -- HmxDType.h, HmxOps.td and this table. The Python
/// contract test_hmx_dtype_admitted_set_contract.py compares the three
/// mechanically, so none of the three can drift alone; the 2026-10-07 review
/// found the pre-refactor ternaries scattered over four sites with nothing
/// binding them.
static FailureOr<PackActLeafFamily> packActLeafFamily(Operation *op,
                                                      Type srcElem) {
  if (srcElem.isF16())
    return PackActLeafFamily{getPackActF16FnName(), getPackActF16BulkFnName(),
                             getPackActTailF16FnName()};
  if (dtype::isF32(srcElem))
    return PackActLeafFamily{getPackActF32FnName(), getPackActF32BulkFnName(),
                             getPackActTailF32FnName()};
  return op->emitError()
         << "hmx.pack_act source element type " << srcElem
         << " is outside the engine's admitted source set {f16, f32} "
            "(dtype::isAdmittedFloat): the pack leaves exist only for those, "
            "and reading the source as f16 would silently pack wrong values. "
            "Extend dtype::isAdmittedFloat (HmxDType.h), the HmxOps.td source "
            "constraint and packActLeafFamily together -- "
            "test_hmx_dtype_admitted_set_contract.py pins the three together";
}

/// The `hmx.pack_weight` family: the same forms as `PackActLeafFamily` plus
/// the `_T` twins for a `src_transposed` source (the [N, K] transpose input;
/// see the op and HMXAPI.h).
struct PackWeightLeafFamily {
  std::string single, singleT, bulk, bulkT, tail, tailT;
};

/// `hmx.pack_weight`'s dtype -> leaf-family table: the same admitted set and
/// the same loud-failure contract as `packActLeafFamily`, its own leaf names
/// (the engine indexes the weight array along different dims, so the two ops
/// cannot share a family).
static FailureOr<PackWeightLeafFamily> packWeightLeafFamily(Operation *op,
                                                            Type srcElem) {
  if (srcElem.isF16())
    return PackWeightLeafFamily{
        getPackWeightF16FnName(), getPackWeightF16TFnName(),
        getPackWeightF16BulkFnName(), getPackWeightF16TBulkFnName(),
        getPackWeightTailF16FnName(), getPackWeightTailF16TFnName()};
  if (dtype::isF32(srcElem))
    return PackWeightLeafFamily{
        getPackWeightF32FnName(), getPackWeightF32TFnName(),
        getPackWeightF32BulkFnName(), getPackWeightF32TBulkFnName(),
        getPackWeightTailF32FnName(), getPackWeightTailF32TFnName()};
  return op->emitError()
         << "hmx.pack_weight source element type " << srcElem
         << " is outside the engine's admitted source set {f16, f32} "
            "(dtype::isAdmittedFloat): the pack leaves exist only for those, "
            "and reading the source as f16 would silently pack wrong values. "
            "Extend dtype::isAdmittedFloat (HmxDType.h), the HmxOps.td source "
            "constraint and packWeightLeafFamily together -- "
            "test_hmx_dtype_admitted_set_contract.py pins the three together";
}

/// `hmx.pack_act` -> `hmx_pack_act_f16(...)` for an f16 source and
/// `hmx_pack_act_f32(...)` for an f32 one (which quantises to the engine's fp16
/// inside the pack), each in the single-block form or the ranged `_bulk` form
/// covering `count` consecutive K tiles in one call (chosen by the op's
/// `count`; its count of 1 is the same single block). `src_stride` is the
/// source's own row stride, so a source that is one N-tile of a wider matrix (a
/// strided view) is read from the right columns instead of from the first
/// `cols` elements of every row.
struct LowerPackAct : public ConvertOpToLLVMPattern<PackActOp> {
  using ConvertOpToLLVMPattern<PackActOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PackActOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto dstType = cast<MemRefType>(op.getDst().getType());
    auto srcType = cast<MemRefType>(op.getSrc().getType());

    int64_t count = op.getCount().value_or(1);
    IntegerAttr validRows = op.getValidRowsAttr();
    IntegerAttr validCols = op.getValidColsAttr();
    if (validRows || validCols) {
      if (count != 1 || !validRows || !validCols) {
        op.emitError("bounds-safe hmx.pack_act requires paired valid "
                     "extents and count=1");
        return failure();
      }
      if (failed(verifyStaticTailLayout(op.getOperation(), srcType,
                                        "pack source", /*rowMajor=*/true)) ||
          failed(verifyStaticTailLayout(op.getOperation(), dstType,
                                        "crouton destination",
                                        /*rowMajor=*/false)))
        return failure();
      FailureOr<PackActLeafFamily> family =
          packActLeafFamily(op.getOperation(), srcType.getElementType());
      if (failed(family))
        return failure();
      auto fn = getVoidLeaf(module, (*family).tail, rewriter);
      if (failed(fn))
        return failure();
      Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                              adaptor.getRow(), adaptor.getCol(),
                              dstType.getElementTypeBitWidth() / 8);
      Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                            srcType.getElementTypeBitWidth() / 8);
      Value rows = dimCst(rewriter, loc, srcType.getDimSize(0));
      Value cols = dimCst(rewriter, loc, srcType.getDimSize(1));
      Value srcStride;
      if (failed(rowStride(rewriter, loc, srcType, adaptor.getSrc(), cols, op,
                                    srcStride)))
        return failure();
      SmallVector<Value> args{dst,
                              src,
                              rows,
                              cols,
                              srcStride,
                              toI32(rewriter, loc, adaptor.getRow()),
                              toI32(rewriter, loc, adaptor.getCol()),
                              dimCst(rewriter, loc, validRows.getInt()),
                              dimCst(rewriter, loc, validCols.getInt())};
      replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
      return success();
    }
    bool bulk = count > 1;
    // The source's element type picks the leaf family (packActLeafFamily): an
    // f32 source is quantised to the engine's fp16 inside the pack, so it has
    // its own entry rather than a narrowing op of its own ahead of the pack.
    // The single and `_bulk` forms have different arities, and which one is
    // called is decided by the name, so the table carries both -- there is no
    // `argTys(7); if (bulk) push_back(i32Ty)` left to keep in step with it.
    FailureOr<PackActLeafFamily> family =
        packActLeafFamily(op.getOperation(), srcType.getElementType());
    if (failed(family))
      return failure();
    auto fn = getVoidLeaf(module, bulk ? (*family).bulk : (*family).single,
                          rewriter);
    if (failed(fn))
      return failure();

    Value dst =
        croutonAddr(rewriter, loc, adaptor.getDst(), dstType, adaptor.getRow(),
                    adaptor.getCol(), dstType.getElementTypeBitWidth() / 8);
    Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                          srcType.getElementTypeBitWidth() / 8);
    Value rows = dimCst(rewriter, loc, srcType.getDimSize(0));
    Value cols = dimCst(rewriter, loc, srcType.getDimSize(1));
    Value srcStride;
    if (failed(rowStride(rewriter, loc, srcType, adaptor.getSrc(), cols, op, srcStride)))
      return failure();

    SmallVector<Value> args{dst,
                            src,
                            rows,
                            cols,
                            srcStride,
                            toI32(rewriter, loc, adaptor.getRow()),
                            toI32(rewriter, loc, adaptor.getCol())};
    if (bulk)
      args.push_back(dimCst(rewriter, loc, count));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};

/// `hmx.pack_weight` -> `hmx_pack_weight_f16(...)` for an f16 source and
/// `hmx_pack_weight_f32(...)` for an f32 one, each in the single-block form or
/// the ranged `_bulk` form covering `count` consecutive K tiles in one call,
/// and each with a `_T` twin for a `src_transposed` source (the [N, K]
/// transpose input; HMXAPI.h explains why the transpose is the input plus the
/// marker, never a strided view).
/// The dst crouton grid is `[Nt, Kt, ...]`, so the address takes `(n_tile,
/// k_tile)` even though the source block coordinates the leaf receives stay
/// `(k, n)`. `src_stride` is the source's own row stride, the same contract as
/// `hmx.pack_act`: a K block that is one tile of a wider matrix is read with
/// its real row stride, not the width.
struct LowerPackWeight : public ConvertOpToLLVMPattern<PackWeightOp> {
  using ConvertOpToLLVMPattern<PackWeightOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PackWeightOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto dstType = cast<MemRefType>(op.getDst().getType());
    auto srcType = cast<MemRefType>(op.getSrc().getType());

    int64_t count = op.getCount().value_or(1);
    IntegerAttr validRows = op.getValidRowsAttr();
    IntegerAttr validCols = op.getValidColsAttr();
    if (validRows || validCols) {
      if (count != 1 || !validRows || !validCols) {
        op.emitError("bounds-safe hmx.pack_weight requires paired valid "
                     "extents and count=1");
        return failure();
      }
      if (failed(verifyStaticTailLayout(op.getOperation(), srcType,
                                        "pack source", /*rowMajor=*/true)) ||
          failed(verifyStaticTailLayout(op.getOperation(), dstType,
                                        "crouton destination",
                                        /*rowMajor=*/false)))
        return failure();
      bool srcTransposed = op.getSrcTransposed().value_or(false);
      FailureOr<PackWeightLeafFamily> family =
          packWeightLeafFamily(op.getOperation(), srcType.getElementType());
      if (failed(family))
        return failure();
      auto fn = getVoidLeaf(
          module, srcTransposed ? (*family).tailT : (*family).tail, rewriter);
      if (failed(fn))
        return failure();
      Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                              adaptor.getNTile(), adaptor.getKTile(),
                              dstType.getElementTypeBitWidth() / 8);
      Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                            srcType.getElementTypeBitWidth() / 8);
      // A transposed source is the [N, K] transpose input itself, so the
      // extents swap: K is dim 1 -- the contiguous, crouton-pair axis -- and N
      // dim 0. The stride fallback width stays dim 1 either way (N row-major,
      // K transposed).
      Value k = dimCst(rewriter, loc, srcType.getDimSize(srcTransposed ? 1 : 0));
      Value n = dimCst(rewriter, loc, srcType.getDimSize(srcTransposed ? 0 : 1));
      Value srcStride;
      if (failed(rowStride(rewriter, loc, srcType, adaptor.getSrc(),
                           srcTransposed ? k : n, op,
                           srcStride)))
        return failure();
      SmallVector<Value> args{dst,
                              src,
                              k,
                              n,
                              srcStride,
                              toI32(rewriter, loc, adaptor.getKTile()),
                              toI32(rewriter, loc, adaptor.getNTile()),
                              dimCst(rewriter, loc, validRows.getInt()),
                              dimCst(rewriter, loc, validCols.getInt())};
      replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
      return success();
    }
    bool bulk = count > 1;
    bool srcTransposed = op.getSrcTransposed().value_or(false);
    FailureOr<PackWeightLeafFamily> family =
        packWeightLeafFamily(op.getOperation(), srcType.getElementType());
    if (failed(family))
      return failure();
    // All four forms (single / bulk, straight / transposed) are separate rows,
    // so the name that is chosen decides the arity -- see LowerPackAct.
    auto fn = getVoidLeaf(
        module,
        bulk ? (srcTransposed ? (*family).bulkT : (*family).bulk)
             : (srcTransposed ? (*family).singleT : (*family).single),
        rewriter);
    if (failed(fn))
      return failure();

    // weight grid is [Nt, Kt]: dim0 = N, dim1 = K (layout A), so row = n_tile,
    // col = k_tile.
    Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                            adaptor.getNTile(), adaptor.getKTile(),
                            dstType.getElementTypeBitWidth() / 8);
    Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                          srcType.getElementTypeBitWidth() / 8);
    // Same transposed-extent swap as the tail branch above.
    Value k = dimCst(rewriter, loc, srcType.getDimSize(srcTransposed ? 1 : 0));
    Value n = dimCst(rewriter, loc, srcType.getDimSize(srcTransposed ? 0 : 1));
    Value srcStride;
    if (failed(rowStride(rewriter, loc, srcType, adaptor.getSrc(),
                         srcTransposed ? k : n, op,
                         srcStride)))
      return failure();

    SmallVector<Value> args{dst,
                            src,
                            k,
                            n,
                            srcStride,
                            toI32(rewriter, loc, adaptor.getKTile()),
                            toI32(rewriter, loc, adaptor.getNTile())};
    if (bulk)
      args.push_back(dimCst(rewriter, loc, count));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};

/// `hmx.unpack_acc` -> `hmx_unpack_acc_f16(dst, crouton(src,row,col), rows,
/// cols, dst_stride, row, col)` for one row-pair, or the ranged
/// `hmx_unpack_acc_f16_bulk(..., row, n_pairs)` covering `count` consecutive
/// row-pairs in one call. The leaf writes fp16 row-major; a wider result is
/// widened by the pipeline afterwards, which keeps one unpack implementation.
/// `dst_stride` is the destination's own row stride, so an N-split block (a
/// strided `memref<M x BN, strided<[N, 1]>>`) lands in the right columns
/// instead of being written contiguously.
struct LowerUnpackAcc : public ConvertOpToLLVMPattern<UnpackAccOp> {
  using ConvertOpToLLVMPattern<UnpackAccOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnpackAccOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    int64_t count = op.getCount().value_or(1);
    IntegerAttr validRows = op.getValidRowsAttr();
    IntegerAttr validCols = op.getValidColsAttr();
    std::optional<int64_t> nTile = readDiagnosticNTile(op.getOperation());
    auto localColumns = [&](MemRefType type,
                            StringRef name) -> FailureOr<int64_t> {
      return diagnosticTileColumns(op.getOperation(), type, nTile, name);
    };
    if (validRows || validCols) {
      if (count != 1 || !validRows || !validCols) {
        op.emitError("bounds-safe hmx.unpack_acc requires paired valid "
                     "extents and count=1");
        return failure();
      }
      auto srcType = cast<MemRefType>(op.getSrc().getType());
      auto dstType = cast<MemRefType>(op.getDst().getType());
      if (nTile && (*nTile < 0 || *nTile >= srcType.getDimSize(1)))
        return op.emitError()
               << "unpack source n_tile " << *nTile
               << " is outside the crouton N extent " << srcType.getDimSize(1);
      FailureOr<int64_t> maybeColumns =
          localColumns(dstType, "unpack destination");
      if (failed(maybeColumns))
        return failure();
      int64_t localColCount = *maybeColumns;
      if (validCols.getInt() > localColCount)
        return op.emitError()
               << "valid_cols " << validCols.getInt()
               << " exceeds selected N tile width " << localColCount;
      if (failed(verifyStaticTailLayout(op.getOperation(), srcType,
                                        "crouton source",
                                        /*rowMajor=*/false)) ||
          failed(verifyStaticTailLayout(op.getOperation(), dstType,
                                        "unpack destination",
                                        /*rowMajor=*/true)))
        return failure();
      auto fn = getVoidLeaf(module, getUnpackAccTailF16FnName(), rewriter);
      if (failed(fn))
        return failure();
      Value dst = asAddress(rewriter, loc, adaptor.getDst(),
                            dstType.getElementTypeBitWidth() / 8);
      if (nTile) {
        FailureOr<Value> shifted = addDiagnosticNTileOffset(
            op.getOperation(), rewriter, loc, dst, nTile,
            dstType.getElementTypeBitWidth() / 8);
        if (failed(shifted))
          return failure();
        dst = *shifted;
      }
      Value sourceCol = nTile ? dimCst(rewriter, loc, *nTile) : dimCst(rewriter, loc, 0);
      Value src = croutonAddr(rewriter, loc, adaptor.getSrc(), srcType,
                              adaptor.getRow(), sourceCol,
                              srcType.getElementTypeBitWidth() / 8);
      Value rows = dimCst(rewriter, loc, dstType.getDimSize(0));
      Value cols = dimCst(rewriter, loc, nTile ? localColCount : dstType.getDimSize(1));
      Value dstStride;
      if (failed(rowStride(rewriter, loc, dstType, adaptor.getDst(),
                           nTile ? dimCst(rewriter, loc, dstType.getDimSize(1)) : cols, op,
                           dstStride)))
        return failure();
      SmallVector<Value> args{dst,
                              src,
                              rows,
                              cols,
                              dstStride,
                              toI32(rewriter, loc, adaptor.getRow()),
                              toI32(rewriter, loc, adaptor.getCol()),
                              dimCst(rewriter, loc, validRows.getInt()),
                              dimCst(rewriter, loc, validCols.getInt())};
      replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
      return success();
    }
    bool bulk = count > 1;
    // The bulk leaf has the same arity as the single one: it replaces the
    // single form's `col` (always 0 here, the leaf walks the crouton row
    // itself) with the row-pair count. The two are separate rows with equal
    // arity, so the table states that fact twice instead of this call site
    // assuming it.
    auto fn = getVoidLeaf(
        module, bulk ? getUnpackAccF16BulkFnName() : getUnpackAccF16FnName(),
        rewriter);
    if (failed(fn))
      return failure();

    auto srcType = cast<MemRefType>(op.getSrc().getType());
    auto dstType = cast<MemRefType>(op.getDst().getType());
    if (nTile && (*nTile < 0 || *nTile >= srcType.getDimSize(1)))
      return op.emitError()
             << "unpack source n_tile " << *nTile
             << " is outside the crouton N extent " << srcType.getDimSize(1);
    FailureOr<int64_t> maybeColumns =
        localColumns(dstType, "unpack destination");
    if (failed(maybeColumns))
      return failure();
    int64_t localColCount = *maybeColumns;
    Value dst = asAddress(rewriter, loc, adaptor.getDst(),
                          dstType.getElementTypeBitWidth() / 8);
    if (nTile) {
      FailureOr<Value> shifted =
          addDiagnosticNTileOffset(op.getOperation(), rewriter, loc, dst, nTile,
                                   dstType.getElementTypeBitWidth() / 8);
      if (failed(shifted))
        return failure();
      dst = *shifted;
    }
    // The leaf unpacks a whole row-pair and walks the crouton row itself, so
    // the address it takes is the selected N tile's first crouton.
    Value sourceCol = nTile ? dimCst(rewriter, loc, *nTile) : dimCst(rewriter, loc, 0);
    Value src =
        croutonAddr(rewriter, loc, adaptor.getSrc(), srcType, adaptor.getRow(),
                    sourceCol, srcType.getElementTypeBitWidth() / 8);
    Value rows = dimCst(rewriter, loc, dstType.getDimSize(0));
    Value cols = dimCst(rewriter, loc, nTile ? localColCount : dstType.getDimSize(1));
    Value dstStride;
    if (failed(rowStride(rewriter, loc, dstType, adaptor.getDst(),
                         nTile ? dimCst(rewriter, loc, dstType.getDimSize(1)) : cols, op,
                         dstStride)))
      return failure();

    SmallVector<Value> args{dst,       src,
                            rows,      cols,
                            dstStride, toI32(rewriter, loc, adaptor.getRow())};
    if (bulk)
      args.push_back(dimCst(rewriter, loc, count));
    else
      args.push_back(toI32(rewriter, loc, adaptor.getCol()));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};

/// `hmx.unpack_acc_f32` -> `hmx_unpack_acc_f32(dst, res, has_res,
/// crouton(src,row,0), rows, cols, dst_stride, res_stride, row, col)`, or its
/// ranged form `..._bulk(..., row, n_pairs)` covering `count` row-pairs. The
/// fused tail: the leaf unpacks the row-pairs to fp32, adds the residual when
/// present, and stores, in one call. The residual travels as (address, flag):
/// absent is (0, 0), so the same symbol serves both forms and the wiring
/// phase can thread C through without a second op. `dst_stride` is the
/// destination's own row stride (the width for a dense destination, larger for
/// a strided view); the residual is dense by construction, so `res_stride` is
/// its column count.
struct LowerUnpackAccF32 : public ConvertOpToLLVMPattern<UnpackAccF32Op> {
  using ConvertOpToLLVMPattern<UnpackAccF32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnpackAccF32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    int64_t count = op.getCount().value_or(1);
    IntegerAttr validRows = op.getValidRowsAttr();
    IntegerAttr validCols = op.getValidColsAttr();
    std::optional<int64_t> nTile = readDiagnosticNTile(op.getOperation());
    auto localColumns = [&](MemRefType type,
                            StringRef name) -> FailureOr<int64_t> {
      return diagnosticTileColumns(op.getOperation(), type, nTile, name);
    };
    if (nTile) {
      FailureOr<int64_t> selectedColumns = localColumns(
          cast<MemRefType>(op.getDst().getType()), "unpack destination");
      if (failed(selectedColumns))
        return failure();
      if (*selectedColumns < layout::kTileEdge && !validRows && !validCols)
        return op.emitError(
            "diagnostic f32 n_tile selecting a partial tile requires "
            "bounds-safe valid_rows and valid_cols");
    }
    if (validRows || validCols) {
      if (count != 1 || !validRows || !validCols) {
        op.emitError("bounds-safe hmx.unpack_acc_f32 requires paired valid "
                     "extents and count=1");
        return failure();
      }
      auto srcType = cast<MemRefType>(op.getSrc().getType());
      auto dstType = cast<MemRefType>(op.getDst().getType());
      if (nTile && (*nTile < 0 || *nTile >= srcType.getDimSize(1)))
        return op.emitError()
               << "unpack source n_tile " << *nTile
               << " is outside the crouton N extent " << srcType.getDimSize(1);
      FailureOr<int64_t> maybeColumns =
          localColumns(dstType, "unpack destination");
      if (failed(maybeColumns))
        return failure();
      int64_t localColCount = *maybeColumns;
      if (validCols.getInt() > localColCount)
        return op.emitError()
               << "valid_cols " << validCols.getInt()
               << " exceeds selected N tile width " << localColCount;
      if (failed(verifyStaticTailLayout(op.getOperation(), srcType,
                                        "crouton source",
                                        /*rowMajor=*/false)) ||
          failed(verifyStaticTailLayout(op.getOperation(), dstType,
                                        "unpack destination",
                                        /*rowMajor=*/true)))
        return failure();
      auto fn = getVoidLeaf(module, getUnpackAccTailF32FnName(), rewriter);
      if (failed(fn))
        return failure();
      Value dst = asAddress(rewriter, loc, adaptor.getDst(),
                            dstType.getElementTypeBitWidth() / 8);
      if (nTile) {
        FailureOr<Value> shifted = addDiagnosticNTileOffset(
            op.getOperation(), rewriter, loc, dst, nTile,
            dstType.getElementTypeBitWidth() / 8);
        if (failed(shifted))
          return failure();
        dst = *shifted;
      }
      Value res;
      Value hasRes;
      Value resStride;
      if (Value residual = adaptor.getResidual()) {
        auto resType = cast<MemRefType>(op.getResidual().getType());
        res = asAddress(rewriter, loc, residual,
                        resType.getElementTypeBitWidth() / 8);
        if (nTile) {
          FailureOr<Value> shiftedResidual = addDiagnosticNTileOffset(
              op.getOperation(), rewriter, loc, res, nTile,
              resType.getElementTypeBitWidth() / 8);
          if (failed(shiftedResidual))
            return failure();
          res = *shiftedResidual;
        }
        if (failed(verifyStaticTailLayout(op.getOperation(), resType,
                                          "residual", /*rowMajor=*/true)))
          return failure();
        hasRes = dimCst(rewriter, loc, 1);
        Value resWidth = dimCst(rewriter, loc, resType.getDimSize(1));
        if (failed(rowStride(rewriter, loc, resType, residual, resWidth, op,
                              resStride)))
          return failure();
      } else {
        res = dimCst(rewriter, loc, 0);
        hasRes = dimCst(rewriter, loc, 0);
        resStride = dimCst(rewriter, loc, 0);
      }
      Value sourceCol = nTile ? dimCst(rewriter, loc, *nTile) : dimCst(rewriter, loc, 0);
      Value src = croutonAddr(rewriter, loc, adaptor.getSrc(), srcType,
                              adaptor.getRow(), sourceCol,
                              srcType.getElementTypeBitWidth() / 8);
      Value rows = dimCst(rewriter, loc, dstType.getDimSize(0));
      Value cols = dimCst(rewriter, loc, nTile ? localColCount : dstType.getDimSize(1));
      Value dstStride;
      if (failed(rowStride(rewriter, loc, dstType, adaptor.getDst(),
                           nTile ? dimCst(rewriter, loc, dstType.getDimSize(1)) : cols, op,
                           dstStride)))
        return failure();
      SmallVector<Value> args{dst,
                              res,
                              hasRes,
                              src,
                              rows,
                              cols,
                              dstStride,
                              resStride,
                              toI32(rewriter, loc, adaptor.getRow()),
                              toI32(rewriter, loc, adaptor.getCol()),
                              dimCst(rewriter, loc, validRows.getInt()),
                              dimCst(rewriter, loc, validCols.getInt())};
      replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
      return success();
    }
    bool bulk = count > 1;
    // Same arity rule as the fp16 unpack: `n_pairs` replaces `col`, it is not
    // an extra argument.
    auto fn = getVoidLeaf(
        module, bulk ? getUnpackAccF32BulkFnName() : getUnpackAccF32FnName(),
        rewriter);
    if (failed(fn))
      return failure();

    auto srcType = cast<MemRefType>(op.getSrc().getType());
    auto dstType = cast<MemRefType>(op.getDst().getType());
    if (nTile && (*nTile < 0 || *nTile >= srcType.getDimSize(1)))
      return op.emitError()
             << "unpack source n_tile " << *nTile
             << " is outside the crouton N extent " << srcType.getDimSize(1);
    FailureOr<int64_t> maybeColumns =
        localColumns(dstType, "unpack destination");
    if (failed(maybeColumns))
      return failure();
    int64_t localColCount = *maybeColumns;
    Value dst = asAddress(rewriter, loc, adaptor.getDst(),
                          dstType.getElementTypeBitWidth() / 8);
    if (nTile) {
      FailureOr<Value> shifted =
          addDiagnosticNTileOffset(op.getOperation(), rewriter, loc, dst, nTile,
                                   dstType.getElementTypeBitWidth() / 8);
      if (failed(shifted))
        return failure();
      dst = *shifted;
    }
    Value res;
    Value hasRes;
    Value resStride;
    if (Value residual = adaptor.getResidual()) {
      auto resType = cast<MemRefType>(op.getResidual().getType());
      res = asAddress(rewriter, loc, residual,
                      resType.getElementTypeBitWidth() / 8);
      if (nTile) {
        FailureOr<Value> shiftedResidual = addDiagnosticNTileOffset(
            op.getOperation(), rewriter, loc, res, nTile,
            resType.getElementTypeBitWidth() / 8);
        if (failed(shiftedResidual))
          return failure();
        res = *shiftedResidual;
      }
      hasRes = dimCst(rewriter, loc, 1);
      // The residual is a row-major block like the destination; its row stride
      // is stride(rank-2), not the width. A strided residual (an N-tile view)
      // read with the width would walk the wrong rows, exactly like the pack
      // `src_stride` / unpack `dst_stride` cases. Dense falls back to the
      // width.
      Value resWidth = dimCst(rewriter, loc, resType.getDimSize(1));
      if (failed(rowStride(rewriter, loc, resType, residual, resWidth, op, resStride)))
        return failure();
    } else {
      res = dimCst(rewriter, loc, 0);
      hasRes = dimCst(rewriter, loc, 0);
      resStride = dimCst(rewriter, loc, 0);
    }
    // Like the fp16 unpack, the leaf walks the crouton row itself, so the
    // address it takes is the selected N tile's first crouton.
    Value sourceCol = nTile ? dimCst(rewriter, loc, *nTile) : dimCst(rewriter, loc, 0);
    Value src =
        croutonAddr(rewriter, loc, adaptor.getSrc(), srcType, adaptor.getRow(),
                    sourceCol, srcType.getElementTypeBitWidth() / 8);
    Value rows = dimCst(rewriter, loc, dstType.getDimSize(0));
    Value cols = dimCst(rewriter, loc, nTile ? localColCount : dstType.getDimSize(1));
    Value dstStride;
    if (failed(rowStride(rewriter, loc, dstType, adaptor.getDst(),
                         nTile ? dimCst(rewriter, loc, dstType.getDimSize(1)) : cols, op,
                         dstStride)))
      return failure();

    SmallVector<Value> args{
        dst,       res,       hasRes,
        src,       rows,      cols,
        dstStride, resStride, toI32(rewriter, loc, adaptor.getRow())};
    if (bulk)
      args.push_back(dimCst(rewriter, loc, count));
    else
      args.push_back(toI32(rewriter, loc, adaptor.getCol()));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};

/// `hmx.stage` -> `hexagon_runtime_dma2d_start(src_row, DDR, slot, VTCM, width,
/// height, srcStride, dstStride, bypassSrc, bypassDst, isOrdered, cap, status)`,
/// returning the DMA token.
///
/// The transfer is a 2D one: a `kTileEdge` x `srcCols` rectangle, each row read
/// at the source's own row stride and written into the slot at the column width.
/// The op's contract *is* that rectangle (HmxOps.td: "lowering 按源自身的 row
/// stride 缩放"), so the strides travel per axis and nothing is assumed about
/// the source being contiguous. A 1D run of `slotBytes` bytes would satisfy the
/// same total length only when the rows happen to be packed back to back; on a
/// source that is one tile of a wider matrix (row stride > width) it walks off
/// the tile into the neighbouring columns and the engine contracts a sheared
/// activation -- measured on the device as a uniform ~2% relative error on the
/// 2048^3 A shape, root-caused bit for bit in
/// docs/analysis/a-kloop-rel-investigation-2026-10-09.md. The 2D form degenerates
/// to exactly that 1D movement when the stride equals the width (dense source),
/// so there is no layout branch here and no shape special case.
///
/// `row` is an element row offset into `src` (not a tile index); the verifier
/// pins `src` to a static rank-2 f16/f32 memref. The op issues its transfer
/// unconditionally: the partition pass keeps every row in range (its pipelined
/// kernel stops one iteration short per pipeline stage and the peeled tail only
/// awaits and computes, staging nothing), so there is no out-of-range case to
/// guard here. A token sentinel is not an option either -- the runtime's tokens
/// start at 0, so 0 cannot mean "no transfer".
struct LowerStage : public ConvertOpToLLVMPattern<StageOp> {
  using ConvertOpToLLVMPattern<StageOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StageOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = module->getContext();
    auto i32Ty = rewriter.getI32Type();
    auto ptrTy = LLVM::LLVMPointerType::get(context);

    auto fn = getDma2DStartLeaf(module, rewriter);
    if (failed(fn))
      return failure();

    auto srcType = cast<MemRefType>(op.getSrc().getType());
    auto dstType = cast<MemRefType>(op.getDst().getType());
    int64_t srcElemBytes = srcType.getElementTypeBitWidth() / 8;
    int64_t dstElemBytes = dstType.getElementTypeBitWidth() / 8;

    auto cst = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v));
    };

    // Source geometry: the row's byte offset, and the tile's own width/stride in
    // bytes. The row stride comes from the operand's layout (dense falls back to
    // the width); the verifier pins a static source, so the column count is a
    // constant and the dense `memref<MxK>` emits stride(0) == K == width.
    Value row = toI32(rewriter, loc, adaptor.getRow());
    Value srcCols = cst(srcType.getDimSize(1));
    Value srcStride;
    if (failed(rowStride(rewriter, loc, srcType, adaptor.getSrc(), srcCols, op, srcStride)))
      return failure();
    Value widthBytes =
        LLVM::MulOp::create(rewriter, loc, i32Ty, srcCols, cst(srcElemBytes));
    Value height = cst(layout::kTileEdge);
    Value srcStrideBytes = LLVM::MulOp::create(rewriter, loc, i32Ty, srcStride,
                                               cst(srcElemBytes));
    Value rowBytes = LLVM::MulOp::create(rewriter, loc, i32Ty, row,
                                         srcStrideBytes);
    Value srcAddr = LLVM::AddOp::create(
        rewriter, loc, i32Ty,
        asAddress(rewriter, loc, adaptor.getSrc(), srcElemBytes), rowBytes);

    // Destination and status are write-through pointers. The DMA address spaces
    // are the memrefs' own memory spaces (0 = DDR, 1 = VTCM), which is exactly
    // the runtime's AddrSpace enum. The slot is one contiguous crouton tile, so
    // its row distance is the width.
    Value dstAddr = asAddress(rewriter, loc, adaptor.getDst(), dstElemBytes);
    int64_t statusElemBytes =
        cast<MemRefType>(op.getStatus().getType()).getElementTypeBitWidth() / 8;
    Value statusAddr =
        asAddress(rewriter, loc, adaptor.getStatus(), statusElemBytes);

    // `bypassCache` per endpoint: the descriptor's snoop-and-invalidate is
    // redundant on an endpoint that is not cacheable memory at all, and must be
    // kept on a DDR one (RuntimeDMA.cc; the reference implementation in
    // llama.cpp/ggml-hexagon uses the same "VTCM endpoint -> 1, else 0" rule).
    auto bypassCache = [&](MemRefType type, int64_t fallback) -> Value {
      return cst(memrefAddressSpace(type, fallback) == hexagon::VTCM_ADDRESS_SPACE ? 1 : 0);
    };

    SmallVector<Value> args{
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, srcAddr),
        cst(memrefAddressSpace(srcType, 0)),
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, dstAddr),
        cst(memrefAddressSpace(dstType, 1)),
        widthBytes,
        height,
        srcStrideBytes,
        widthBytes, // dstStride
        bypassCache(srcType, 0),
        bypassCache(dstType, 1),
        cst(0), // isOrdered
        cst(0), // cacheAllocationPolicy
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, statusAddr)};

    // The call's i32 token is the loop-carried handle: the value an external
    // scheduler versions and rotates.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{i32Ty}, FlatSymbolRefAttr::get((*fn).getOperation()),
        args);
    return success();
  }
};

/// `hmx.await` -> `dma_wait(token)`, then the result is the staged slot itself.
/// The wait is unconditional (the matching `hmx.stage` always issued its
/// transfer), and the op is a pure alias: the result IS the `dst` descriptor,
/// so the consumer reads exactly the buffer the DMA wrote -- no copy, no
/// descriptor rebuild.
struct LowerAwait : public ConvertOpToLLVMPattern<AwaitOp> {
  using ConvertOpToLLVMPattern<AwaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AwaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fn = getDmaWaitLeaf(module, rewriter);
    if (failed(fn))
      return failure();

    Value token = toI32(rewriter, loc, adaptor.getToken());
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn,
                        ValueRange{token});
    return success();
  }
};

struct HmxToLLVMPass : public ::impl::HmxToLLVMBase<HmxToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, LLVM::LLVMDialect, HmxDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();
    if (failed(verifyDiagnosticNTileContract(moduleOp))) {
      signalPassFailure();
      return;
    }

    // Decide *before* converting which functions issue HMX leaves: the decision
    // is made on the hmx dialect ops, and the conversion below erases every one
    // of them, so it cannot be made afterwards -- and must not be made on
    // callee names (see issuesHmxEngineLeaves). The conversion registers
    // patterns for hmx ops only, so it neither renames nor rebuilds a function;
    // recording the symbol names here identifies the same functions after it,
    // even if a future lowering were to rebuild a function while keeping its
    // name.
    //
    // The role split's work functions are EXEMPT from the pair, and the
    // exemption has to be known here, before the collection: their lock is
    // the bound thread's lifetime lock (HmxRoleExecutor.cpp's
    // roleThreadEntry takes it once and never releases), and an unlock
    // inserted at a work function's exit would release that lifetime lock
    // under the executor's feet. verifyHmxLeafCallers sees the same set so
    // the exemption shows up as an exemption, not as a missed pair.
    llvm::StringSet<> roleWorkFns;
    if (auto roleHandoffs =
            moduleOp->getAttrOfType<ArrayAttr>(kHmxRoleHandoffsAttr))
      for (Attribute record : roleHandoffs.getValue())
        if (auto workName =
                cast<DictionaryAttr>(record).getAs<StringAttr>(kHmxRoleWorkField))
          roleWorkFns.insert(workName.getValue());

    llvm::StringSet<> engineKernels;
    auto collectEngineKernel = [&](auto fn) {
      if (!fn.isDeclaration() && issuesHmxEngineLeaves(fn) &&
          roleWorkFns.count(fn.getName()) == 0)
        engineKernels.insert(fn.getName());
    };
    moduleOp.walk([&](LLVM::LLVMFuncOp fn) { collectEngineKernel(fn); });
    moduleOp.walk([&](func::FuncOp fn) { collectEngineKernel(fn); });

    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);
    LowerToLLVMOptions options(context,
                               dataLayoutAnalysis.getAtOrAbove(moduleOp));
    LLVMTypeConverter typeConverter(context, options);

    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<HmxDialect>();

    hmx::populateHmxToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();

    // Invariant check before anything is inserted: a leaf caller the dialect
    // test declined must fail the build loudly, not ship without a pair (see
    // verifyHmxLeafCallers). Existing control flow is untouched: after
    // signalPassFailure() this still runs, exactly as it always has. The
    // role work functions are exempt with their reason: their lock is the
    // bound thread's lifetime lock, not a per-kernel pair.
    verifyHmxLeafCallers(moduleOp, engineKernels, roleWorkFns);

    // Every function collected above as issuing HMX leaves has to power the
    // engine on and release it (one ensure/unlock pair per kernel); no other
    // pass knows whether the engine is needed. The decision was made while the
    // hmx ops it is based on still existed -- by now they are leaf calls.
    ensureHmxEngine(moduleOp, engineKernels);

    // Hand the vector executor the function it runs. This is here rather than in
    // HmxVectorReadoutPass because only an `llvm.func` has an expressible
    // ADDRESS, and only this pass runs after convert-func-to-llvm (see the
    // section comment above wireVectorReadout). A module with no handoff record
    // is untouched, which is what keeps the default-off path byte-identical.
    if (failed(wireVectorReadout(moduleOp))) {
      signalPassFailure();
      return;
    }

    // The thread-role channel's compiler half: the exported entry point and
    // depth object the launch-side probe looks for, and the LLVM
    // thread-contract attribute on the work function. Same position rule as
    // the read-out wiring above; a module with no role handoff record is
    // untouched.
    if (failed(wireRoleChannel(moduleOp))) {
      signalPassFailure();
      return;
    }

    // This call is a separate diagnostic contract. It is emitted only when
    // the marker-gated compiler sidecar selected one immutable function/site
    // and an explicit grid=1 invocation; ordinary kernels never acquire an
    // event context merely because they allocate VTCM.
    if (failed(verifySiteScopeBrackets(moduleOp))) {
      signalPassFailure();
      return;
    }
    if (failed(insertDiagnosticEventContext(moduleOp))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void hmx::populateHmxToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  patterns.add<LowerMma, LowerAccRead, LowerBiasInit, LowerAccClear,
               LowerPackAct, LowerPackWeight, LowerUnpackAcc, LowerUnpackAccF32,
               LowerStage, LowerAwait>(typeConverter);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::hmx::createHmxToLLVMPass() {
  return std::make_unique<HmxToLLVMPass>();
}
