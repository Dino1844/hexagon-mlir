//===- HexagonCAPI.h - hexagon alloc-free runtime calls -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// / The source pointer is a pointer to the base of memref.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_
#define HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_

#include "HexagonAPI.h"
#include <cstdint>

extern "C" {
/// Pointer-returning allocation and alias entries enforce a fail-closed
/// non-null result contract: an allocation failure (including an unsupported
/// zero-size/descriptor request) aborts through the runtime guard before a
/// caller can build a memref descriptor. Non-null results pass through
/// unchanged.
void *hexagon_runtime_alloc_1d(size_t bytes, uint64_t alignment, bool isVtcm);
void hexagon_runtime_free_1d(void *ptr);
void *hexagon_runtime_alloc_2d(size_t numBlocks, size_t blockSize,
                               uint64_t alignment, bool isVtcm);
void hexagon_runtime_free_2d(void *ptr);
void hexagon_runtime_copy(void *dst, void *src, size_t nbytes, bool isDVtcm,
                          bool isSrcVtcm);

/// The source pointer is a pointer to the base of memref
void *hexagon_runtime_build_crouton(void *source, size_t nbytes);
/// The source pointer is a pointer to crouton table
void *hexagon_runtime_get_contiguous_memref(void *source);

/// The sole resident ABI is versioned and scope-gated. Device symbols append
/// `_dsp`: `(i64 key/source, i32 bytes, i32 alignment) -> ptr`. There is no
/// unversioned resident entry point; this keeps one resident-allocation choke
/// point and prevents legacy callers from bypassing scope registration. The
/// alignment argument must be one of the powers of two through 2048; the
/// runtime's VtcmPool::IsSupportedAlignment check is the single behavioral gate.
int32_t hexagon_runtime_resident_scope_enter_v2(uint64_t scopeLow64,
                                                uint64_t scopeHigh64);
void *hexagon_runtime_workspace_resident_v2(uint64_t key, uint32_t bytes,
                                            uint32_t alignment);
void *hexagon_runtime_weight_resident_v2(uint64_t src, uint32_t bytes,
                                         uint32_t alignment);
/// Record a resident deallocation without releasing its pinned block. Returns
/// zero only when the pointer and byte descriptor match a registered resident.
int32_t hexagon_runtime_resident_free_v2(void *ptr, uint32_t bytes);

/// Power the HMX engine up and acquire it, without allocating anything else.
/// Emitted once at the entry of every function that issues HMX leaves; see the
/// device-side definition for why it cannot be skipped.
void hexagon_runtime_hmx_ensure(void);

/// Unlock the HMX unit from the current thread. Emitted before every return of
/// a function that issues HMX leaves, paired with the ensure at its entry; see
/// the device-side definition. Only unlocks when this thread holds the lock
/// (thread_local flag), so a call without a matching ensure is a no-op.
void hexagon_runtime_hmx_unlock(void);

#ifdef HEXMLIR_RUNTIME_BRINGUP_PROBE
/// Format the recorded per-step bring-up timings into `buf` as "BRINGUP ..."
/// lines and return the number of bytes written (0 when the bring-up never ran
/// in this process). Defined in HexagonAPI.cpp; probe build only. See
/// exp/hmx/bringup_probe.
int hexagon_runtime_bringup_report(char *buf, int cap);
#endif

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
enum {
  HEXAGON_RUNTIME_VTCM_ACCOUNTING_CONTEXT_VERSION = 1,
  HEXAGON_RUNTIME_VTCM_ACCOUNTING_CONTEXT_HAS_RESIDENT_KEY_DIGEST = 1u << 0,
  HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_VERSION = 1,
  HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_SINGLE_INVOCATION = 1u << 0,
  HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_GRID_ONE = 1u << 1,
};

/// Register the immutable identity context for the opt-in accounting stream.
/// The version and flags are explicit; all IDs are supplied by the producer.
/// The runtime never derives them from a pointer, a function name, or an
/// address. `scopeId` is a distinct 64-bit process observation ID; it is not
/// the 128-bit resident-v2 scope pair (`scopeLow64`, `scopeHigh64`) and does
/// not establish a resident-scope join. resident_scope_binding=not-proven.
/// The first registration and an identical repeat return zero; a changed
/// context, an unknown version/flag, or registration after an event
/// fails closed. Version 1 is one immutable observation context per process;
/// it does not silently switch function/site IDs mid-stream. Version 1 has no
/// per-event site binding: all runtime events are process-scoped and suppress
/// the declared site IDs. Its optional resident digest is a process resident
/// identity, not a function/allocation-site/grid join; a future context-switch
/// ABI is required for multiple site/resident/grid joins. v1 does not provide
/// a full-kernel or grid-instance join (`grid_scope_status=not-bound-in-v1`).
/// Device symbols append `_dsp`.
int32_t hexagon_runtime_vtcm_accounting_context_enter_v1(
    uint32_t version, uint32_t flags, uint64_t buildIdLow, uint64_t buildIdHigh,
    uint64_t functionId, uint64_t allocationSiteId, uint64_t scopeId,
    uint64_t residentKeyDigestLow, uint64_t residentKeyDigestHigh);

/// Diagnostic-only per-thread event context. The token is an opaque,
/// non-address 128-bit pair supplied by the compiler sidecar; the
/// accounting-scope, invocation, function, site, and grid words are
/// declarations used only for exact equality checks. Version 1 requires an
/// explicit single-invocation, grid=1 contract and fails closed on any
/// mismatch, second thread, or late registration. The entry is void because a
/// rejected registration simply leaves the event stream aggregate; it never
/// changes kernel control flow.
void hexagon_runtime_vtcm_accounting_event_context_enter_v1(
    uint32_t version, uint32_t flags, uint64_t tokenLow, uint64_t tokenHigh,
    uint64_t accountingScopeId, uint64_t invocationId, uint64_t functionId,
    uint64_t allocationSiteId, uint32_t gridProduct);
/// Clear the current thread's event owner at the kernel return boundary. A
/// later invocation without its own enter call therefore remains aggregate.
void hexagon_runtime_vtcm_accounting_event_context_leave_v1(void);

/// Format the opt-in VTCM allocation/free accounting snapshot and bounded
/// event log into `buf`. The report contains allocator sizes and counters,
/// never raw addresses. With no registered context it remains the historical
/// process-aggregate report; with a context, v1 still emits process-scoped
/// events with an explicit unbound status and no site join key. The separate
/// accounting-event-v1 ABI, when registered, adds exact token-bound direct
/// events without changing this process declaration.
/// Returns the number of bytes written, 0 for an invalid buffer, or -2 when
/// the report was truncated and may be retried.
int hexagon_runtime_vtcm_accounting_report(char *buf, int cap);
#endif
}
#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_
