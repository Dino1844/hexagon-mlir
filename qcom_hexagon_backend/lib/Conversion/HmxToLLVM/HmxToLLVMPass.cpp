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

#include "hexagon/Conversion/HmxToLLVM/HmxExternalFnNames.h"
#include "hexagon/Conversion/HmxToLLVM/HmxToLLVM.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDType.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxVtcmAccounting.h"

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

/// Declare (once) a runtime leaf with the given argument types and no result.
FailureOr<LLVM::LLVMFuncOp> getVoidLeaf(ModuleOp module, StringRef name,
                                        ArrayRef<Type> argTys,
                                        ConversionPatternRewriter &rewriter) {
  auto voidTy = LLVM::LLVMVoidType::get(module->getContext());
  return LLVM::lookupOrCreateFn(rewriter, module, name, argTys, voidTy);
}

/// Declare (once) a runtime leaf that returns a value.
FailureOr<LLVM::LLVMFuncOp> getLeaf(ModuleOp module, StringRef name,
                                    ArrayRef<Type> argTys, Type resultTy,
                                    ConversionPatternRewriter &rewriter) {
  return LLVM::lookupOrCreateFn(rewriter, module, name, argTys, resultTy);
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
        !op->hasTrait<OpTrait::HmxDmaOnly>())
      found = true;
  });
  return found;
}

/// The symbol prefix of the runtime HMX leaves (all of HmxExternalFnNames.cpp).
/// Used only by verifyHmxLeafCallers below -- never to decide where a pair
/// goes.
static constexpr const char *kHmxLeafPrefix = "hmx_";

/// Post-conversion invariant check, deliberately run *before* ensureHmxEngine
/// so a diagnostic sees IR the insertion has not touched yet. A function that
/// calls an HMX leaf but is absent from `engineKernels` would run HMX
/// instructions with no ensure/unlock pair, and the next thread would block
/// forever in HAP_compute_res_hmx_lock (bin/runtime/src/HexagonAPI.cpp,
/// EnsureHmxLockForThisThread) -- so that state must fail the build loudly
/// instead of shipping a kernel that hangs the device.
///
/// The `hmx_` callee prefix appears here **for verification only, never for the
/// decision**: which functions get the pair is decided solely by
/// issuesHmxEngineLeaves, on the hmx dialect ops, before the conversion. The
/// dialect test is strictly narrower than the old prefix test -- it cannot see
/// a leaf call that exists without a dialect op behind it -- and this check
/// turns exactly that one direction, which used to be a *silent* missing pair,
/// into a compile-time error. It never selects a function for insertion.
static void verifyHmxLeafCallers(ModuleOp moduleOp,
                                 const llvm::StringSet<> &engineKernels) {
  auto verify = [&](auto fn) {
    if (fn.isDeclaration() || engineKernels.count(fn.getName()))
      return;
    bool found = false;
    fn->walk([&](LLVM::CallOp call) {
      if (std::optional<StringRef> callee = call.getCallee())
        found |= callee->starts_with(kHmxLeafPrefix);
    });
    if (found)
      fn.emitError()
          << "function '" << fn.getName() << "' issues HMX leaf calls ('"
          << kHmxLeafPrefix
          << "...') but issuesHmxEngineLeaves did not recognise it from its "
             "hmx dialect ops, so it gets no "
             "hexagon_runtime_hmx_ensure_dsp/hexagon_runtime_hmx_unlock_dsp "
             "pair: it would execute HMX instructions without the engine "
             "brought up, and the next thread would block forever in "
             "HAP_compute_res_hmx_lock";
  };
  moduleOp.walk([&](LLVM::LLVMFuncOp fn) { verify(fn); });
  moduleOp.walk([&](func::FuncOp fn) { verify(fn); });
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
/// stride cannot be read here, so it falls back to `width` (the dense contract)
/// rather than mis-addressing.
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
/// view into a packed buffer whose stride only resolves at run time, and that
/// `rowStride` returns a `Value` rather than a constant is the mechanism, not
/// an oversight. Gating it would have broken the feature on the strength of a
/// guess.
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
                        MemRefType type, Value width, Operation *anchor,
                        Value &out) {
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
    if (!ShapedType::isDynamic(stride) && stride != widthC) {
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

/// The element count of a memref as an i32. A static shape is a constant; a
/// dynamic one is the product of the descriptor's sizes.
static Value memrefNumElements(ConversionPatternRewriter &rewriter,
                               Location loc, Value memrefDesc,
                               MemRefType type) {
  auto i32Ty = rewriter.getI32Type();
  if (type.hasStaticShape())
    return LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(type.getNumElements()));
  MemRefDescriptor desc(memrefDesc);
  Value n = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                     rewriter.getI32IntegerAttr(1));
  for (unsigned i = 0; i < type.getRank(); ++i)
    n = LLVM::MulOp::create(rewriter, loc, i32Ty, n,
                            toI32(rewriter, loc, desc.size(rewriter, loc, i)));
  return n;
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

/// Declare (once) the DMA start entry the `hmx.stage` op lowers to. The
/// signature is the one DMAToLLVMPass declares for the same symbol: the pointer
/// arguments are real `!llvm.ptr`s (the host data layout widens them), while
/// the address spaces and the two bypass flags travel as i32.
static FailureOr<LLVM::LLVMFuncOp>
getDmaStartLeaf(ModuleOp module, ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(context);
  auto i32Ty = rewriter.getI32Type();
  return getLeaf(module, getStageDmaStartFnName(),
                 {ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, ptrTy},
                 i32Ty, rewriter);
}

/// The DMA wait entry `hmx.await` lowers to: `void dma_wait(i32 token)`.
static FailureOr<LLVM::LLVMFuncOp>
getDmaWaitLeaf(ModuleOp module, ConversionPatternRewriter &rewriter) {
  return getVoidLeaf(module, getAwaitDmaWaitFnName(), {rewriter.getI32Type()},
                     rewriter);
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

    auto fn =
        getVoidLeaf(module, getMmaF16FnName(), {i32Ty, i32Ty, i32Ty}, rewriter);
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

    auto loadFn =
        getVoidLeaf(module, getBiasLoadF16FnName(), {i32Ty, i32Ty}, rewriter);
    auto storeFn =
        getVoidLeaf(module, getAccStoreF16FnName(), {i32Ty, i32Ty}, rewriter);
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
    auto i32Ty = rewriter.getI32Type();

    auto fn =
        getVoidLeaf(module, getBiasInitUnitF16FnName(), {i32Ty}, rewriter);
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

    auto fn = getVoidLeaf(module, getAccClearF16FnName(), {}, rewriter);
    if (failed(fn))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*fn).getOperation()),
        ValueRange{});
    return success();
  }
};

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
    auto i32Ty = rewriter.getI32Type();

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
      bool srcIsF32 = dtype::isF32(srcType.getElementType());
      auto fn = getVoidLeaf(module,
                            srcIsF32 ? getPackActTailF32FnName()
                                     : getPackActTailF16FnName(),
                            SmallVector<Type>(9, i32Ty), rewriter);
      if (failed(fn))
        return failure();
      Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                              adaptor.getRow(), adaptor.getCol(),
                              dstType.getElementTypeBitWidth() / 8);
      Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                            srcType.getElementTypeBitWidth() / 8);
      auto dimCst = [&](int64_t v) {
        return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(v))
            .getResult();
      };
      Value rows = dimCst(srcType.getDimSize(0));
      Value cols = dimCst(srcType.getDimSize(1));
      Value srcStride;
      if (failed(rowStride(rewriter, loc, srcType, cols, op,
                                    srcStride)))
        return failure();
      SmallVector<Value> args{dst,
                              src,
                              rows,
                              cols,
                              srcStride,
                              toI32(rewriter, loc, adaptor.getRow()),
                              toI32(rewriter, loc, adaptor.getCol()),
                              dimCst(validRows.getInt()),
                              dimCst(validCols.getInt())};
      replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
      return success();
    }
    bool bulk = count > 1;
    SmallVector<Type> argTys(7, i32Ty);
    if (bulk)
      argTys.push_back(i32Ty);
    // The source's element type picks the leaf: an f32 source is quantised to
    // the engine's fp16 inside the pack, so it has its own entry rather than a
    // narrowing op of its own ahead of the pack.
    bool srcIsF32 = dtype::isF32(srcType.getElementType());
    auto fn = getVoidLeaf(
        module,
        srcIsF32 ? (bulk ? getPackActF32BulkFnName() : getPackActF32FnName())
                 : (bulk ? getPackActF16BulkFnName() : getPackActF16FnName()),
        argTys, rewriter);
    if (failed(fn))
      return failure();

    Value dst =
        croutonAddr(rewriter, loc, adaptor.getDst(), dstType, adaptor.getRow(),
                    adaptor.getCol(), dstType.getElementTypeBitWidth() / 8);
    Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                          srcType.getElementTypeBitWidth() / 8);
    auto dimCst = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v))
          .getResult();
    };
    Value rows = dimCst(srcType.getDimSize(0));
    Value cols = dimCst(srcType.getDimSize(1));
    Value srcStride;
    if (failed(rowStride(rewriter, loc, srcType, cols, op, srcStride)))
      return failure();

    SmallVector<Value> args{dst,
                            src,
                            rows,
                            cols,
                            srcStride,
                            toI32(rewriter, loc, adaptor.getRow()),
                            toI32(rewriter, loc, adaptor.getCol())};
    if (bulk)
      args.push_back(dimCst(count));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};

/// `hmx.pack_weight` -> `hmx_pack_weight_f16(...)` for an f16 source and
/// `hmx_pack_weight_f32(...)` for an f32 one, each in the single-block form or
/// the ranged `_bulk` form covering `count` consecutive K tiles in one call.
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
    auto i32Ty = rewriter.getI32Type();

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
      bool srcIsF32 = dtype::isF32(srcType.getElementType());
      auto fn = getVoidLeaf(module,
                            srcIsF32 ? getPackWeightTailF32FnName()
                                     : getPackWeightTailF16FnName(),
                            SmallVector<Type>(9, i32Ty), rewriter);
      if (failed(fn))
        return failure();
      Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                              adaptor.getNTile(), adaptor.getKTile(),
                              dstType.getElementTypeBitWidth() / 8);
      Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                            srcType.getElementTypeBitWidth() / 8);
      auto dimCst = [&](int64_t v) {
        return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(v))
            .getResult();
      };
      Value k = dimCst(srcType.getDimSize(0));
      Value n = dimCst(srcType.getDimSize(1));
      Value srcStride;
      if (failed(rowStride(rewriter, loc, srcType, n, op, srcStride)))
        return failure();
      SmallVector<Value> args{dst,
                              src,
                              k,
                              n,
                              srcStride,
                              toI32(rewriter, loc, adaptor.getKTile()),
                              toI32(rewriter, loc, adaptor.getNTile()),
                              dimCst(validRows.getInt()),
                              dimCst(validCols.getInt())};
      replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
      return success();
    }
    bool bulk = count > 1;
    SmallVector<Type> argTys(7, i32Ty);
    if (bulk)
      argTys.push_back(i32Ty);
    bool srcIsF32 = dtype::isF32(srcType.getElementType());
    auto fn = getVoidLeaf(
        module,
        srcIsF32
            ? (bulk ? getPackWeightF32BulkFnName() : getPackWeightF32FnName())
            : (bulk ? getPackWeightF16BulkFnName() : getPackWeightF16FnName()),
        argTys, rewriter);
    if (failed(fn))
      return failure();

    // weight grid is [Nt, Kt]: dim0 = N, dim1 = K (layout A), so row = n_tile,
    // col = k_tile.
    Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                            adaptor.getNTile(), adaptor.getKTile(),
                            dstType.getElementTypeBitWidth() / 8);
    Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                          srcType.getElementTypeBitWidth() / 8);
    auto dimCst = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v))
          .getResult();
    };
    Value k = dimCst(srcType.getDimSize(0));
    Value n = dimCst(srcType.getDimSize(1));
    Value srcStride;
    if (failed(rowStride(rewriter, loc, srcType, n, op, srcStride)))
      return failure();

    SmallVector<Value> args{dst,
                            src,
                            k,
                            n,
                            srcStride,
                            toI32(rewriter, loc, adaptor.getKTile()),
                            toI32(rewriter, loc, adaptor.getNTile())};
    if (bulk)
      args.push_back(dimCst(count));
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
    auto i32Ty = rewriter.getI32Type();

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
      auto fn = getVoidLeaf(module, getUnpackAccTailF16FnName(),
                            SmallVector<Type>(9, i32Ty), rewriter);
      if (failed(fn))
        return failure();
      auto dimCst = [&](int64_t v) {
        return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(v))
            .getResult();
      };
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
      Value sourceCol = nTile ? dimCst(*nTile) : dimCst(0);
      Value src = croutonAddr(rewriter, loc, adaptor.getSrc(), srcType,
                              adaptor.getRow(), sourceCol,
                              srcType.getElementTypeBitWidth() / 8);
      Value rows = dimCst(dstType.getDimSize(0));
      Value cols = dimCst(nTile ? localColCount : dstType.getDimSize(1));
      Value dstStride;
      if (failed(rowStride(rewriter, loc, dstType,
                           nTile ? dimCst(dstType.getDimSize(1)) : cols, op,
                           dstStride)))
        return failure();
      SmallVector<Value> args{dst,
                              src,
                              rows,
                              cols,
                              dstStride,
                              toI32(rewriter, loc, adaptor.getRow()),
                              toI32(rewriter, loc, adaptor.getCol()),
                              dimCst(validRows.getInt()),
                              dimCst(validCols.getInt())};
      replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
      return success();
    }
    bool bulk = count > 1;
    // The bulk leaf has the same arity as the single one: it replaces the
    // single form's `col` (always 0 here, the leaf walks the crouton row
    // itself) with the row-pair count. Appending it instead would shift the ABI
    // and the leaf would read n_pairs = 0, writing nothing.
    SmallVector<Type> argTys(7, i32Ty);
    auto fn = getVoidLeaf(
        module, bulk ? getUnpackAccF16BulkFnName() : getUnpackAccF16FnName(),
        argTys, rewriter);
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
    auto dimCst = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v))
          .getResult();
    };
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
    Value sourceCol = nTile ? dimCst(*nTile) : dimCst(0);
    Value src =
        croutonAddr(rewriter, loc, adaptor.getSrc(), srcType, adaptor.getRow(),
                    sourceCol, srcType.getElementTypeBitWidth() / 8);
    Value rows = dimCst(dstType.getDimSize(0));
    Value cols = dimCst(nTile ? localColCount : dstType.getDimSize(1));
    Value dstStride;
    if (failed(rowStride(rewriter, loc, dstType,
                         nTile ? dimCst(dstType.getDimSize(1)) : cols, op,
                         dstStride)))
      return failure();

    SmallVector<Value> args{dst,       src,
                            rows,      cols,
                            dstStride, toI32(rewriter, loc, adaptor.getRow())};
    if (bulk)
      args.push_back(dimCst(count));
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
    auto i32Ty = rewriter.getI32Type();

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
      auto fn = getVoidLeaf(module, getUnpackAccTailF32FnName(),
                            SmallVector<Type>(12, i32Ty), rewriter);
      if (failed(fn))
        return failure();
      auto dimCst = [&](int64_t v) {
        return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(v))
            .getResult();
      };
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
        hasRes = dimCst(1);
        Value resWidth = dimCst(resType.getDimSize(1));
        if (failed(rowStride(rewriter, loc, resType, resWidth, op,
                              resStride)))
          return failure();
      } else {
        res = dimCst(0);
        hasRes = dimCst(0);
        resStride = dimCst(0);
      }
      Value sourceCol = nTile ? dimCst(*nTile) : dimCst(0);
      Value src = croutonAddr(rewriter, loc, adaptor.getSrc(), srcType,
                              adaptor.getRow(), sourceCol,
                              srcType.getElementTypeBitWidth() / 8);
      Value rows = dimCst(dstType.getDimSize(0));
      Value cols = dimCst(nTile ? localColCount : dstType.getDimSize(1));
      Value dstStride;
      if (failed(rowStride(rewriter, loc, dstType,
                           nTile ? dimCst(dstType.getDimSize(1)) : cols, op,
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
                              dimCst(validRows.getInt()),
                              dimCst(validCols.getInt())};
      replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
      return success();
    }
    bool bulk = count > 1;
    // Same arity rule as the fp16 unpack: `n_pairs` replaces `col`, it is not
    // an extra argument.
    SmallVector<Type> argTys(10, i32Ty);
    auto fn = getVoidLeaf(
        module, bulk ? getUnpackAccF32BulkFnName() : getUnpackAccF32FnName(),
        argTys, rewriter);
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
    auto dimCst = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v))
          .getResult();
    };
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
      hasRes = dimCst(1);
      // The residual is a row-major block like the destination; its row stride
      // is stride(rank-2), not the width. A strided residual (an N-tile view)
      // read with the width would walk the wrong rows, exactly like the pack
      // `src_stride` / unpack `dst_stride` cases. Dense falls back to the
      // width.
      Value resWidth = dimCst(resType.getDimSize(1));
      if (failed(rowStride(rewriter, loc, resType, resWidth, op, resStride)))
        return failure();
    } else {
      res = dimCst(0);
      hasRes = dimCst(0);
      resStride = dimCst(0);
    }
    // Like the fp16 unpack, the leaf walks the crouton row itself, so the
    // address it takes is the selected N tile's first crouton.
    Value sourceCol = nTile ? dimCst(*nTile) : dimCst(0);
    Value src =
        croutonAddr(rewriter, loc, adaptor.getSrc(), srcType, adaptor.getRow(),
                    sourceCol, srcType.getElementTypeBitWidth() / 8);
    Value rows = dimCst(dstType.getDimSize(0));
    Value cols = dimCst(nTile ? localColCount : dstType.getDimSize(1));
    Value dstStride;
    if (failed(rowStride(rewriter, loc, dstType,
                         nTile ? dimCst(dstType.getDimSize(1)) : cols, op,
                         dstStride)))
      return failure();

    SmallVector<Value> args{
        dst,       res,       hasRes,
        src,       rows,      cols,
        dstStride, resStride, toI32(rewriter, loc, adaptor.getRow())};
    if (bulk)
      args.push_back(dimCst(count));
    else
      args.push_back(toI32(rewriter, loc, adaptor.getCol()));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};

/// `hmx.stage` -> `hexagon_runtime_dma_start(src_row, DDR, slot, VTCM,
/// slot_bytes, 0, 0, status)`, returning the DMA token. The slot is filled by
/// one 1D DMA, so the source is addressed as `base + row * stride(0) *
/// elemBytes`: the memref's *real* row stride, not its width, otherwise a
/// source that is one tile of a wider matrix starts at the wrong column. The
/// length is the destination's element count in bytes (the whole slot is
/// filled).
///
/// `row` is an element row offset into `src` (not a tile index); the verifier
/// pins `src` to a static rank-2 f16 memref. The op issues its transfer
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

    auto fn = getDmaStartLeaf(module, rewriter);
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

    // Source address: descriptor base+offset, then the row's byte offset. The
    // row stride comes from the operand's own layout (dense falls back to the
    // width); the verifier pins a static source, so the column count is a
    // constant and the dense `memref<MxK>` emits stride(0) == K == width.
    Value row = toI32(rewriter, loc, adaptor.getRow());
    Value srcCols = cst(srcType.getDimSize(1));
    Value srcStride;
    if (failed(rowStride(rewriter, loc, srcType, srcCols, op, srcStride)))
      return failure();
    Value rowBytes = LLVM::MulOp::create(
        rewriter, loc, i32Ty,
        LLVM::MulOp::create(rewriter, loc, i32Ty, row, srcStride),
        cst(srcElemBytes));
    Value srcAddr = LLVM::AddOp::create(
        rewriter, loc, i32Ty,
        asAddress(rewriter, loc, adaptor.getSrc(), srcElemBytes), rowBytes);

    // Destination and status are write-through pointers. The DMA address spaces
    // are the memrefs' own memory spaces (0 = DDR, 1 = VTCM), which is exactly
    // the runtime's AddrSpace enum.
    Value dstAddr = asAddress(rewriter, loc, adaptor.getDst(), dstElemBytes);
    int64_t statusElemBytes =
        cast<MemRefType>(op.getStatus().getType()).getElementTypeBitWidth() / 8;
    Value statusAddr =
        asAddress(rewriter, loc, adaptor.getStatus(), statusElemBytes);
    Value lengthBytes = LLVM::MulOp::create(
        rewriter, loc, i32Ty,
        memrefNumElements(rewriter, loc, adaptor.getDst(), dstType),
        cst(dstElemBytes));

    SmallVector<Value> args{
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, srcAddr),
        cst(memrefAddressSpace(srcType, 0)),
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, dstAddr),
        cst(memrefAddressSpace(dstType, 1)),
        lengthBytes,
        cst(0), // bypassCacheSrc: the source may be cached DDR
        cst(0), // bypassCacheDst
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
    llvm::StringSet<> engineKernels;
    moduleOp.walk([&](LLVM::LLVMFuncOp fn) {
      if (!fn.isDeclaration() && issuesHmxEngineLeaves(fn))
        engineKernels.insert(fn.getName());
    });
    moduleOp.walk([&](func::FuncOp fn) {
      if (!fn.isDeclaration() && issuesHmxEngineLeaves(fn))
        engineKernels.insert(fn.getName());
    });

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
    // signalPassFailure() this still runs, exactly as it always has.
    verifyHmxLeafCallers(moduleOp, engineKernels);

    // Every function collected above as issuing HMX leaves has to power the
    // engine on and release it (one ensure/unlock pair per kernel); no other
    // pass knows whether the engine is needed. The decision was made while the
    // hmx ops it is based on still existed -- by now they are leaf calls.
    ensureHmxEngine(moduleOp, engineKernels);

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
