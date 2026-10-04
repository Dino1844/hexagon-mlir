//===- HexagonCAPI.cpp -  hexagon alloc free runtime calls (DSP) ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// / The source pointer is a pointer to the base of memref
//
//===----------------------------------------------------------------------===//

#include <cstdio> // fopen/fputs for the compiled-out _trc marker below

// Crash-triage progress marker (fa-rowmax plan §10/§15). This opens, writes
// and closes rt_trc.txt on every runtime call, which otherwise dominates small
// kernels (fa-crash log R24); it is compiled out unless HEXMLIR_RUNTIME_TRACE
// is defined. Never enable it for measurements.
#ifdef HEXMLIR_RUNTIME_TRACE
static void _trc(const char *m) {
  FILE *_f = fopen("/data/data/com.termux/files/home/csm/op/rt_trc.txt", "a");
  if (_f) {
    fputs(m, _f);
    fputc('\n', _f);
    fclose(_f);
  }
}
#else
#define _trc(m) ((void)0)
#endif
#include "HexagonCAPI.h"
#include "HexagonCommon.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <mutex>

namespace {
// The scope guard is intentionally independent of HexagonAPI: registering an
// ID must not construct VTCM, power up HMX, or otherwise create resources.
// There is one immutable resident scope per process. The lock only protects
// this tiny diagnostic state and is not on ordinary allocation paths.
struct ResidentScopeState {
  std::mutex mutex;
  bool registered = false;
  uint64_t low = 0;
  uint64_t high = 0;
};

ResidentScopeState gResidentScopeState;

bool residentScopeRegistered() {
  std::lock_guard<std::mutex> lock(gResidentScopeState.mutex);
  return gResidentScopeState.registered;
}

/// Pointer-returning allocation/alias entries have a fail-closed result
/// contract: a null result is never allowed to become a memref descriptor.
/// The existing runtime CHECK primitive is the smallest available failure
/// channel on this path; valid non-null results pass through unchanged.
void *requireAllocationResult(void *ptr) {
  CHECK(ptr != nullptr, "runtime allocation returned null");
  return ptr;
}
} // namespace

extern "C" {
// Crash-triage value dump (fa-rowmax plan §15 Step-1, temporary): logs the
// runtime pointer values the kernel passes to memset / receives from the
// allocators, so the bad pointer (slot-188 chain) can be named directly.
// NB: the former `hexagon_runtime_pk_barrier` stub is gone (2026-09-23): its
// emitter was already removed (R16/T2a -- barriers are calls and perturbed
// scheduling), leaving a dead export with no declaration and no caller.
void hexagon_runtime_dbg_log_ptr(int id, void *p) {
  FILE *_f = fopen("/data/data/com.termux/files/home/csm/op/rt_trc.txt", "a");
  if (_f) {
    fprintf(_f, "PTR id=%d p=%p\n", id, p);
    fclose(_f);
  }
}
void *hexagon_runtime_alloc_1d_dsp(size_t bytes, uint64_t alignment,
                                   bool isVtcm) {
  _trc("A1");
  return requireAllocationResult(
      HexagonAPI::Global()->Alloc(bytes, alignment, isVtcm));
}

void *hexagon_runtime_alloc_2d_dsp(size_t numBlocks, size_t blockSize,
                                   uint64_t alignment, bool isVtcm) {
  _trc("A2");
  return requireAllocationResult(
      HexagonAPI::Global()->Alloc(numBlocks, blockSize, alignment, isVtcm));
}

void hexagon_runtime_free_1d_dsp(void *ptr) {
  _trc("F1");
  HexagonAPI::Global()->Free(ptr);
}

void hexagon_runtime_free_2d_dsp(void *ptr) {
  _trc("F2");
  HexagonAPI::Global()->Free(ptr);
}

void hexagon_runtime_copy_dsp(void *dst, void *src, size_t nbytes,
                              bool isDstVtcm, bool isSrcVtcm) {
  _trc("C0");
  HexagonAPI::Global()->Copy(dst, src, nbytes);
}

/// The source pointer is a pointer to the base of memref
void *hexagon_runtime_build_crouton_dsp(void *source, size_t nbytes) {
  _trc("CR");
  assert(nbytes % CROUTON_SIZE == 0 &&
         "The size is expected to be a multiple of crouton size");
  return requireAllocationResult(
      HexagonAPI::Global()->CreateBufferAlias(source, nbytes));
}

/// The source pointer is a pointer to crouton table
void *hexagon_runtime_get_contiguous_memref_dsp(void *source) {
  _trc("GC");
  return requireAllocationResult(
      HexagonAPI::Global()->GetOrigBufferFromAlias(source));
}

/// Register the one immutable resident scope for this process. This entry only
/// records the 128-bit ID; it does not call HexagonAPI::Global() or construct
/// any runtime resource. A repeated identical ID succeeds, while any different
/// ID fails without replacing the first registration.
int32_t hexagon_runtime_resident_scope_enter_v2_dsp(uint64_t scopeLow64,
                                                    uint64_t scopeHigh64) {
  std::lock_guard<std::mutex> lock(gResidentScopeState.mutex);
  if (gResidentScopeState.registered &&
      (gResidentScopeState.low != scopeLow64 ||
       gResidentScopeState.high != scopeHigh64)) {
    FARF(ERROR, "VTCM resident scope mismatch");
    return -1;
  }
  if (!gResidentScopeState.registered) {
    gResidentScopeState.low = scopeLow64;
    gResidentScopeState.high = scopeHigh64;
    gResidentScopeState.registered = true;
  }
  return 0;
}

/// Versioned workspace resident ABI: (key, bytes, alignment, instance) with
/// i64/i32/i32/i32 arguments. `instance` is the caller's flat program id --
/// the discriminator that keeps concurrent instances of a grid>1 launch on
/// separate buffers (see HexagonAPI::WorkspaceResidentV2). Registration is
/// mandatory and checked before constructing the lazy runtime singleton;
/// there is no fallback to the unversioned entry.
void *hexagon_runtime_workspace_resident_v2_dsp(uint64_t key, uint32_t bytes,
                                                uint32_t alignment,
                                                uint32_t instance) {
  if (!residentScopeRegistered()) {
    FARF(ERROR, "VTCM workspace resident requested before scope entry");
    return requireAllocationResult(nullptr);
  }
  _trc("WS2");
  return requireAllocationResult(HexagonAPI::Global()->WorkspaceResidentV2(
      key, bytes, alignment, instance));
}

/// Versioned weight resident ABI: (source, bytes, alignment) with i64/i32/i32
/// arguments. Source-address reuse is valid only for the documented immutable
/// one-weight-object-per-process contract; runtime performs no content hash.
void *hexagon_runtime_weight_resident_v2_dsp(uint64_t src, uint32_t bytes,
                                             uint32_t alignment) {
  if (!residentScopeRegistered()) {
    FARF(ERROR, "VTCM weight resident requested before scope entry");
    return requireAllocationResult(nullptr);
  }
  _trc("WR2");
  return requireAllocationResult(
      HexagonAPI::Global()->WeightResidentV2(src, bytes, alignment));
}

int32_t hexagon_runtime_resident_free_v2_dsp(void *ptr, uint32_t bytes) {
  if (!residentScopeRegistered()) {
    FARF(ERROR, "VTCM resident free requested before scope entry");
    return -1;
  }
  return HexagonAPI::Global()->FreeResident(ptr, bytes) ? 0 : -1;
}


/// Bring the HMX engine up without allocating anything. Constructing the
/// HexagonAPI singleton runs AcquireResources(), which powers HMX up and
/// acquires it; that is the precondition for every HMX instruction. A kernel
/// that issues HMX leaves but never calls a runtime allocation (attention with
/// VTCM/hexagonmem off) has to call this explicitly, otherwise the first HMX
/// instruction kills the DSP.
/// Threading: the singleton is built exactly once under a process-wide lock,
/// so N qurt threads entering here concurrently (one per grid program under
/// tm.exec) issue a single HAP_compute_res_acquire; latecomers just observe
/// the published instance. The HMX unit lock is per-thread and is NOT held by
/// the constructing thread (initialize_and_acquire_hmx releases it), so the
/// thread that runs HMX locks the shared context for itself here (the
/// thread_local flag in HexagonAPI.cpp makes a repeated lock without an
/// intervening unlock a no-op). A lock on a held unit blocks in the resource
/// manager until the holder releases it, which serializes the single engine.
///
/// The compiler pairs this with hexagon_runtime_hmx_unlock_dsp: one ensure at
/// each function entry that issues HMX leaves, one unlock before each return
/// (HmxToLLVM::ensureHmxEngine).
void hexagon_runtime_hmx_ensure_dsp(void) {
  _trc("HE");
  HexagonAPI *api = HexagonAPI::Global();
  api->EnsureHmxLockForThisThread();
}

/// Release this thread's HMX lock. Paired 1:1 with the
/// hexagon_runtime_hmx_ensure_dsp at the function entry. A NON_SHARED unlock
/// clears the accumulators per qurt_hmx.h, so it must come after the last read,
/// which the compiler's position guarantees. Only unlocks when this thread
/// holds the lock; failures are logged, never silent.
void hexagon_runtime_hmx_unlock_dsp(void) {
  _trc("HU");
  HexagonAPI::Global()->ReleaseHmxLockForThisThread();
}

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
static_assert(VtcmPool::kAccountingContextVersion ==
                  HEXAGON_RUNTIME_VTCM_ACCOUNTING_CONTEXT_VERSION,
              "accounting context version must match the C ABI");
static_assert(
    VtcmPool::kAccountingContextHasResidentKeyDigest ==
        HEXAGON_RUNTIME_VTCM_ACCOUNTING_CONTEXT_HAS_RESIDENT_KEY_DIGEST,
    "accounting context flags must match the C ABI");
static_assert(VtcmPool::kAccountingEventContextVersion ==
                  HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_VERSION,
              "event context version must match the C ABI");
static_assert(
    VtcmPool::kAccountingEventContextSingleInvocation ==
        HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_SINGLE_INVOCATION &&
        VtcmPool::kAccountingEventContextGridOne ==
            HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_GRID_ONE,
    "event context flags must match the C ABI");

int32_t hexagon_runtime_vtcm_accounting_context_enter_v1_dsp(
    uint32_t version, uint32_t flags, uint64_t buildIdLow, uint64_t buildIdHigh,
    uint64_t functionId, uint64_t allocationSiteId, uint64_t scopeId,
    uint64_t residentKeyDigestLow, uint64_t residentKeyDigestHigh) {
  VtcmPool::AccountingContext context{};
  context.version = version;
  context.flags = flags;
  context.buildId = {buildIdLow, buildIdHigh};
  context.functionId = functionId;
  context.allocationSiteId = allocationSiteId;
  context.scopeId = scopeId;
  context.residentKeyDigest = {residentKeyDigestLow, residentKeyDigestHigh};
  return VtcmPool::registerAccountingContext(context) ? 0 : -1;
}

void hexagon_runtime_vtcm_accounting_event_context_enter_v1_dsp(
    uint32_t version, uint32_t flags, uint64_t tokenLow, uint64_t tokenHigh,
    uint64_t accountingScopeId, uint64_t invocationId, uint64_t functionId,
    uint64_t allocationSiteId, uint32_t gridProduct) {
  if (gridProduct != 1) {
    VtcmPool::clearAccountingEventContext();
    return;
  }
  VtcmPool::AccountingEventContext context{};
  context.version = version;
  context.flags = flags;
  context.token = {tokenLow, tokenHigh};
  context.accountingScopeId = accountingScopeId;
  context.invocationId = invocationId;
  context.functionId = functionId;
  context.allocationSiteId = allocationSiteId;
  (void)VtcmPool::registerAccountingEventContext(context);
}

void hexagon_runtime_vtcm_accounting_event_context_leave_v1_dsp(void) {
  VtcmPool::clearAccountingEventContext();
}

void hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp(
    uint32_t version, uint32_t flags, uint64_t tokenLow, uint64_t tokenHigh,
    uint64_t accountingScopeId, uint64_t invocationId, uint64_t functionId,
    uint64_t allocationSiteId, uint64_t buildIdLow, uint64_t buildIdHigh,
    uint32_t gridProduct) {
  // A grid that is not one has no per-site invocation to name, so the scope is
  // released rather than registered. This mirrors the frame entry: the refusal
  // is total, not a partial attribution.
  if (gridProduct != 1) {
    VtcmPool::clearAccountingSiteScope();
    return;
  }
  VtcmPool::AccountingSiteScope scope{};
  scope.version = version;
  scope.flags = flags;
  scope.token = {tokenLow, tokenHigh};
  scope.accountingScopeId = accountingScopeId;
  scope.invocationId = invocationId;
  scope.functionId = functionId;
  scope.allocationSiteId = allocationSiteId;
  scope.buildId = {buildIdLow, buildIdHigh};
  (void)VtcmPool::registerAccountingSiteScope(scope);
}

void hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp(void) {
  VtcmPool::clearAccountingSiteScope();
}

/// Return the diagnostic report without exposing a pointer or changing the
/// allocator. Calling this entry point constructs the lazy runtime singleton,
/// just like the other runtime entry points; probe callers normally invoke
/// the report after an allocation/ensure so the snapshot is the interesting
/// steady state.
int hexagon_runtime_vtcm_accounting_report(char *buf, int cap) {
  if (buf == nullptr || cap <= 0)
    return 0;

  HexagonAPI *api = HexagonAPI::Global();
  int offset = api->getVtcmPool()->writeAccountingReport(buf, cap);
  if (offset == VtcmPool::kAccountingReportTruncated)
    return offset;
  if (offset < 0)
    return 0;
  if (offset >= cap)
    return VtcmPool::kAccountingReportTruncated;

  const auto cache = api->getBufferManager()->getAccountingSnapshot();
  const auto pool = api->getVtcmPool()->getAccountingSnapshot();
  const uint64_t combinedUpperBound =
      pool.highWaterAllocatedBytes >
              std::numeric_limits<uint64_t>::max() -
                  cache.highWaterCachedVtcmBytes
          ? std::numeric_limits<uint64_t>::max()
          : pool.highWaterAllocatedBytes + cache.highWaterCachedVtcmBytes;
  // The pool ledger already charges blocks retained by BufferManager.  The
  // lower bound is therefore the pool charged high-water; the sum is only a
  // conservative upper bound until a same-time inclusion proof exists.  Do
  // not label either number as full-kernel occupancy.
  const uint64_t combinedLowerBound = pool.highWaterAllocatedBytes;
  const int written = snprintf(
      buf + offset, static_cast<size_t>(cap - offset),
      "VTCM_BUFFER_CACHE scope=process counter_scope=cumulative-process "
      "event_counter_scope=retained-ring "
      "requested_bytes_basis=raw-buffer-request "
      "allocator_input_bytes_basis=vtcm-pool-entry "
      "allocator_aligned_bytes_basis=size-rounding "
      "charged_bytes_basis=runtime-pool-block-length "
      "free_cache_hits=%llu free_cache_hit_bytes=%llu "
      "free_cache_hit_requested_bytes=%llu "
      "free_cache_retains=%llu free_cache_retained_bytes=%llu "
      "free_cache_retain_requested_bytes=%llu "
      "free_cache_drops=%llu free_cache_drop_bytes=%llu "
      "free_cache_evictions=%llu free_cache_eviction_bytes=%llu "
      "free_cache_eviction_events=%llu "
      "cached_vtcm_high_water_bytes=%llu cached_buffer_high_water=%llu "
      "pool_high_water_charged_bytes=%llu "
      "combined_occupancy_lower_bound_bytes=%llu "
      "combined_occupancy_upper_bound_bytes=%llu "
      "combined_occupancy_status=not-proven "
      "combined_occupancy_basis=pool-charge-plus-buffer-manager-cache-no-double-count "
      "full_kernel_occupancy_status=not-proven "
      "cached_buffers=%llu cached_buffers_scope=vtcm-ledger "
      "cached_bytes=%llu cached_bytes_scope=all-free-cache-occupancy "
      "cached_vtcm_bytes=%llu cached_vtcm_bytes_scope=vtcm-ledger "
      "cache_capacity_bytes=%llu cache_scope=vtcm-only "
      "ddr_cache_accounting=excluded "
      "cache_key_identity=footprint-only content_identity_status=not-proven "
      "eviction_retry_status=not-proven "
      "free_cache_retain_events=%llu free_cache_hit_events=%llu "
      "free_cache_drop_events=%llu free_cache_evict_events=%llu\n",
      static_cast<unsigned long long>(cache.freeCacheHits),
      static_cast<unsigned long long>(cache.freeCacheHitBytes),
      static_cast<unsigned long long>(cache.freeCacheHitRequestedBytes),
      static_cast<unsigned long long>(cache.freeCacheRetains),
      static_cast<unsigned long long>(cache.freeCacheRetainedBytes),
      static_cast<unsigned long long>(cache.freeCacheRetainRequestedBytes),
      static_cast<unsigned long long>(cache.freeCacheDrops),
      static_cast<unsigned long long>(cache.freeCacheDropBytes),
      static_cast<unsigned long long>(cache.freeCacheEvictions),
      static_cast<unsigned long long>(cache.freeCacheEvictionBytes),
      static_cast<unsigned long long>(cache.freeCacheEvictionEvents),
      static_cast<unsigned long long>(cache.highWaterCachedVtcmBytes),
      static_cast<unsigned long long>(cache.highWaterCachedBuffers),
      static_cast<unsigned long long>(pool.highWaterAllocatedBytes),
      static_cast<unsigned long long>(combinedLowerBound),
      static_cast<unsigned long long>(combinedUpperBound),
      static_cast<unsigned long long>(cache.cachedBuffers),
      static_cast<unsigned long long>(cache.cachedBytes),
      static_cast<unsigned long long>(cache.cachedVtcmBytes),
      static_cast<unsigned long long>(cache.cacheCapacityBytes),
      static_cast<unsigned long long>(pool.freeCacheRetainEvents),
      static_cast<unsigned long long>(pool.freeCacheHitEvents),
      static_cast<unsigned long long>(pool.freeCacheDropEvents),
      static_cast<unsigned long long>(pool.freeCacheEvictEvents));
  if (written < 0)
    return offset;
  if (written >= cap - offset)
    return VtcmPool::kAccountingReportTruncated;
  return offset + written;
}
#endif
}
