//===-- HmxWorkspaceResidentPass.cpp - per-launch VTCM workspace, resident -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Every HMX kernel pays the same per-launch VTCM prologue: one allocation and
// one deallocation per workspace buffer (the crouton arrays of `matmul-to-hmx`,
// the partition pass's conversion state, activation-staging ring slots,
// statuses and the crouton-row scratch). The runtime's allocator does a
// best-fit scan, a split/coalesce and the buffer-manager bookkeeping for each,
// at 1.12 / 1.51 / 1.69 us per alloc/free pair (S1/S2/S3: A/B delta pcycles
// divided by the static pair counts, docs/hmx/m3.2-device-result-2026-09-29.md
// §4). On S3 that is a few us against a launch of the same order, so the tax is
// what dominates a small shape.
//
// Two corrections (2026-10-07), both from that same §4:
//   - "~6.3 us per pair" and "size-independent" were WRONG. The real value is
//     1.1-1.7 us, i.e. 4-6x lower, and the three points scatter by 1.5x, so
//     "size-independent" has no evidence behind it -- three points cannot
//     decide it (the pair counts came from a 09-28 build). Treat the
//     size-independence claim as untested, not as fact.
//   - the number was also high enough to matter: at 6.3 us/pair this pass
//     would look like a large win on any small shape. At the measured value
//     its effect is smaller, and it must be re-measured on the current build
//     before being quoted as a size.
//
// A workspace has no compile-time image: the kernel refills it on every launch.
// So all it needs is a stable buffer, which is exactly the residency mechanism
// the weight path already uses, minus the copy. This pass gives each per-launch
// VTCM workspace allocation a compile-time key and tags it
// `hmx.workspace_resident`; its lowering emits one call to
// `hexagon_runtime_workspace_resident_v2_dsp(key, bytes, alignment)`, which
// allocates the buffer
// on the first launch, pins it against the per-launch deallocation and returns
// the same address forever after. The allocation and deallocation calls
// disappear, leaving only the call and the lookup.
//
// The key is `(hash(function symbol) << 32) | index` with the top bit set: it
// is stable across launches of one kernel, unique per (function, buffer), and
// cannot collide with the address keys the weight residency uses. It is not
// stable across recompiles, which is fine -- the runtime's residency map lives
// in one process running one compiled kernel.
//
// Correctness boundary: a resident buffer is shared by every launch whose
// program instances run under the same flat program id. The lowering passes
// the caller's flat pid into the resident entry (VtcmPool::Resident's slot,
// hexagon_runtime_workspace_resident_v2_dsp's instance argument): concurrent
// instances of a grid>1 launch carry distinct pids and get separate buffers
// instead of a shared clobbered one, and the same pid across launches reuses
// the same buffer. A thread id would be the wrong discriminator: the wrapper
// spawns fresh qurt threads per launch (multithreading.h's "keep the thread
// pool alive" TODO), so thread-keyed residency would allocate a never-reused
// buffer set every launch and grow the resident map without bound (measured:
// mha_fa grid=4, +73%, 2026-10-04). IR without the trailing program-info
// pack (direct pass invocations) is single-instance by construction: slot 0.
// Weights keep the process-global slot 0: their content is immutable once
// copied. Cross-process reuse remains outside the contract, the same
// boundary the weight residency documents.
//
// Runs after `hmx-partition` (so the ring slots/statuses and the scratch exist)
// and before `convert-to-hexagonmem` (which carries the tag onto the
// `hexagonmem.alloc` the lowering reads).
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <limits>
#include <mutex>
#include <set>
#include <string>

#define DEBUG_TYPE "hmx-workspace-resident"

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_HMXWORKSPACERESIDENT
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

/// Per-buffer residency record on the allocation. `key` is the runtime map key,
/// `bytes` the buffer size; the lowering reads both to build the resident call.
constexpr StringLiteral kResidentAttr = kHmxWorkspaceResidentAttr;
constexpr const char *kResidentKeyAttr = "key";
constexpr const char *kResidentBytesAttr = "bytes";

/// A strict diagnostic site identity is deliberately separate from the
/// production key: it never becomes the runtime key (that stays the
/// compatibility ABI for marked and unmarked modules alike).  It drives
/// collision detection (at collection time) and is published via the
/// provenance record's site_id/function_id, proving the site is a stable
/// principal/function/source-location tuple rather than an allocation order.
struct WorkspaceResidentSite {
  memref::AllocOp alloc;
  int64_t bytes = 0;
  int64_t alignment = 0;
  std::string principal;
  std::string function;
  std::string site;
};

/// The workspace record's own field: the runtime key (the production
/// compatibility key; see the key-selection comment in `runOnOperation`).
/// The shared field skeleton (and the buffer-safety checks this pass audits
/// with) lives in `HmxResidentContract.h`; only what is unique to a workspace
/// stays here.
static DictionaryAttr
makeWorkspaceProvenance(MLIRContext *context, StringRef principal,
                        StringRef function, StringRef site, int64_t slot,
                        int64_t bytes, int64_t alignment, uint64_t key) {
  NamedAttrList fields;
  // Workspace bytes are deliberately not a content identity: every
  // launch refills the buffer.  This is an explicit statement, not an
  // implicit hash.
  appendResidentProvenanceCore(fields, context, "workspace",
                               "workspace-resident", principal, function, site,
                               slot, bytes, alignment, "per-launch-refill");
  fields.append("key", IntegerAttr::get(IntegerType::get(context, 64), key));
  return DictionaryAttr::get(context, fields);
}

/// Idempotence check shared with the weight pass (see
/// `HmxResidentContract.h`); this local entry point keeps the pass's own
/// subject wording in its diagnostic.
static LogicalResult setOrValidateWorkspaceProvenance(Operation *operation,
                                                      DictionaryAttr expected) {
  return setOrValidateResidentProvenance(operation, expected,
                                         "resident workspace");
}

struct HmxWorkspaceResidentPass
    : public mlir::hmx::impl::HmxWorkspaceResidentBase<
          HmxWorkspaceResidentPass> {
  using HmxWorkspaceResidentBase::HmxWorkspaceResidentBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = cast<func::FuncOp>(getOperation());
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return;
    if (!isValidResidentDiagnosticMarker(module)) {
      module.emitError(
          "hmx.diagnostic_vtcm_accounting must be a unit attribute");
      return signalPassFailure();
    }
    if (!isValidResidentIdentityMarker(module)) {
      module.emitError("hmx.diagnostic_vtcm_identity must be a unit attribute");
      return signalPassFailure();
    }
    // The raw census marker remains observational.  Only the paired static
    // identity marker opts into source-site/descriptor rejection; this keeps
    // existing diagnostic pipeline output and production key selection
    // unchanged.
    const bool strictResidentContract = hasStrictResidentContract(module);

    // Only HMX kernels have the per-launch workspace this pass is about. A
    // function the HMX path did not touch keeps its allocations exactly as
    // they were.
    bool hasHmx = false;
    func.walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (dialect && dialect->getNamespace() == HmxDialect::getDialectNamespace())
        hasHmx = true;
    });
    if (!hasHmx)
      return;

    SmallVector<memref::AllocOp> workspaces;
    SmallVector<WorkspaceResidentSite> strictSites;
    // In diagnostic mode this is a validation pass as well as a producer: do
    // all checks before mutating IR, so a collision cannot leave half a
    // resident contract behind.
    // Strict keys only need to be distinct, so the collision check keeps
    // a key set and no owner identity.
    std::set<uint64_t> keyOwners;
    LogicalResult collectionResult = success();
    const std::string principal = residentPrincipalName(module);
    const std::string function = func.getSymName().str();
    func.walk([&](memref::AllocOp alloc) {
      // The weight path already pinned its buffers; leave them to it.
      if (alloc->hasAttr(kHmxWeightResidentAttr))
        return;
      auto type = dyn_cast<MemRefType>(alloc.getType());
      if (!type || type.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
        return;
      if (strictResidentContract && !type.hasStaticShape()) {
        func.emitError(
            "strict resident workspace requires a statically sized allocation");
        collectionResult = failure();
        return;
      }
      if (!type.hasStaticShape())
        return;
      Type element = type.getElementType();
      // Sub-byte elements (i1 masks) have no byte size this pass could declare.
      if (!element.isIntOrFloat() || element.getIntOrFloatBitWidth() <= 0 ||
          element.getIntOrFloatBitWidth() % 8 != 0 ||
          type.getNumElements() == 0) {
        if (strictResidentContract) {
          func.emitError(
              "strict resident workspace has an unsupported element type or "
              "zero-sized allocation");
          collectionResult = failure();
        }
        return;
      }

      if (strictResidentContract) {
        int64_t elements = type.getNumElements();
        int64_t elementBytes = element.getIntOrFloatBitWidth() / 8;
        if (elementBytes <= 0 ||
            elements > std::numeric_limits<int64_t>::max() / elementBytes) {
          alloc.emitError("resident workspace byte size overflows int64");
          collectionResult = failure();
          return;
        }
        int64_t bytes = elements * elementBytes;
        if (bytes > std::numeric_limits<uint32_t>::max()) {
          alloc.emitError(
              "resident workspace byte size exceeds the v2 uint32 ABI");
          collectionResult = failure();
          return;
        }
        int64_t alignment = residentAlignment(alloc);
        if (!isSupportedResidentAlignment(alignment)) {
          alloc.emitError("resident workspace has an unsupported alignment");
          collectionResult = failure();
          return;
        }
        if (!type.getLayout().isIdentity()) {
          alloc.emitError(
              "resident workspace requires an identity memref layout");
          collectionResult = failure();
          return;
        }
        std::optional<std::string> site;
        if (auto provenance = alloc->getAttrOfType<DictionaryAttr>(
                kHmxResidentProvenanceAttr)) {
          auto persistedSite = provenance.getAs<StringAttr>("site");
          if (!persistedSite ||
              !isCanonicalResidentSite(persistedSite.getValue())) {
            alloc.emitError(
                "resident workspace has no canonical persisted source site");
            collectionResult = failure();
            return;
          }
          site = persistedSite.getValue().str();
        } else {
          site = residentSite(alloc);
        }
        if (!site) {
          alloc.emitError(
              "resident workspace has no stable source-location site; "
              "ordinal allocation keys are not an identity proof");
          collectionResult = failure();
          return;
        }
        if (residentFunctionIdentity(principal, function) == 0 ||
            residentSiteIdentity(principal, function, *site,
                                 "workspace-resident", /*slot=*/0) == 0) {
          alloc.emitError("strict resident workspace identity is zero");
          collectionResult = failure();
          return;
        }
        // Collision detection on the site identity, not on the runtime key:
        // two allocations claiming the same source site would make the
        // provenance evidence ambiguous.
        uint64_t siteKey = stableWorkspaceResidentKey(principal, function,
                                                      *site, /*slot=*/0);
        if (!keyOwners.insert(siteKey).second) {
          func.emitError("resident workspace key collision for stable site");
          collectionResult = failure();
          return;
        }
        strictSites.push_back({alloc, bytes, alignment, principal, function,
                               *site});
      }
      workspaces.push_back(alloc);
    });
    if (failed(collectionResult))
      return signalPassFailure();
    if (strictResidentContract) {
      for (WorkspaceResidentSite &site : strictSites) {
        unsigned directDeallocs = 0;
        bool aliasDealloc = false;
        // This pass runs before `convert-to-hexagonmem`, so its releases are
        // still memref-typed; see `ResidentDeallocForm`.
        if (failed(validateResidentDeallocs(
                site.alloc.getResult(), func.getOperation(), directDeallocs,
                aliasDealloc, ResidentDeallocForm::MemrefOnly,
                "resident workspace")))
          return signalPassFailure();
        if (directDeallocs > 1) {
          func.emitError(
              "resident workspace has more than one direct deallocation");
          return signalPassFailure();
        }
        // A fresh allocation must have one owner-visible release.  A tagged
        // allocation may already have had that deallocation removed by an
        // earlier invocation of this idempotent pass.
        if (directDeallocs == 0 && !site.alloc->hasAttr(kResidentAttr)) {
          func.emitError(
              "resident workspace has no direct deallocation to transfer");
          return signalPassFailure();
        }
      }
    }
    if (workspaces.empty())
      return;

    // Key by the function symbol and the buffer's order within it, never by an
    // address: the key has to be the same on every launch of the same kernel.
    // This is the compatibility key, for marked and unmarked modules alike:
    // strict diagnostics never swap it for the site identity, because a
    // diagnostic that changed the key distribution would measure a different
    // kernel than production.  The stronger principal/function/site identity
    // is published in the provenance record (site_id/function_id) instead.
    uint64_t hash = llvm::hash_value(func.getSymName());

    MLIRContext *context = func.getContext();
    auto i64 = IntegerType::get(context, 64);
    for (unsigned index = 0; index < workspaces.size(); ++index) {
      memref::AllocOp candidate = workspaces[index];
      auto type = cast<MemRefType>(candidate.getType());
      int64_t bytes = type.getNumElements() * (type.getElementTypeBitWidth() / 8);
      uint64_t key = kHmxWorkspaceResidentKeyTag | (hash << 32) |
                     (static_cast<uint64_t>(index) & 0xFFFFFFFFull);
      DictionaryAttr strictExpected;
      if (strictResidentContract) {
        auto strictIt =
            llvm::find_if(strictSites, [&](const WorkspaceResidentSite &site) {
              return site.alloc == candidate;
            });
        if (strictIt == strictSites.end()) {
          candidate.emitError("resident workspace lost its validated site");
          return signalPassFailure();
        }
        strictExpected = makeWorkspaceProvenance(
            context, strictIt->principal, strictIt->function, strictIt->site,
            /*slot=*/0, strictIt->bytes, strictIt->alignment, key);
        if (failed(setOrValidateWorkspaceProvenance(candidate.getOperation(),
                                                    strictExpected)))
          return signalPassFailure();
      }
      auto resident = candidate->getAttrOfType<DictionaryAttr>(kResidentAttr);
      if (resident) {
        auto oldKey = resident.getAs<IntegerAttr>(kResidentKeyAttr);
        auto oldBytes = resident.getAs<IntegerAttr>(kResidentBytesAttr);
        if (!oldKey || !oldBytes ||
            static_cast<uint64_t>(oldKey.getInt()) != key ||
            oldBytes.getInt() != bytes ||
            (strictResidentContract && resident != strictExpected)) {
          candidate.emitError(
              "resident workspace key or byte descriptor changed on rerun");
          return signalPassFailure();
        }
      } else if (strictResidentContract) {
        // Keep the evidence in the descriptor that ConvertToHexagonmem already
        // forwards.  A separate operation attribute alone would be lost at
        // that boundary, making the post-conversion claim unverifiable.
        candidate->setAttr(kResidentAttr, strictExpected);
      } else {
        candidate->setAttr(
            kResidentAttr,
            DictionaryAttr::get(
                context,
                {NamedAttribute(StringAttr::get(context, kResidentKeyAttr),
                                IntegerAttr::get(i64, key)),
                 NamedAttribute(StringAttr::get(context, kResidentBytesAttr),
                                IntegerAttr::get(i64, bytes))}));
      }

      // The pinned buffer is released by the process, not by the launch, so its
      // per-launch deallocation is dropped. Any other user of the buffer stays.
      SmallVector<memref::DeallocOp> deallocs;
      for (Operation *user : candidate->getUsers())
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          deallocs.push_back(dealloc);
      for (memref::DeallocOp dealloc : deallocs)
        dealloc.erase();

      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] resident workspace #" << index
                              << " (" << bytes << " bytes, key 0x"
                              << llvm::Twine::utohexstr(key) << ")\n");
    }

    // The setter is one unlocked read-modify-write primitive. Serialize the
    // complete manifest transaction because sibling function passes may run
    // concurrently; do not nest a lock inside the setter.
    std::lock_guard<std::mutex> manifestGuard(hmxModuleStateMutex());
    if (module->hasAttr("hmx.kernel_manifest") &&
        failed(setHmxManifestWorkspaceClass(module, func.getSymName(),
                                            kHmxWorkspaceResident)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hmx::createHmxWorkspaceResidentPass() {
  return std::make_unique<HmxWorkspaceResidentPass>();
}
