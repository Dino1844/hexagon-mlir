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
  if (_f) { fputs(m, _f); fputc('\n', _f); fclose(_f); }
}
#else
#define _trc(m) ((void)0)
#endif
#include "HexagonCAPI.h"
#include "HexagonCommon.h"
#include <cassert>
#include <cstdint>

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
                                   bool isVtcm) { _trc("A1");
  return HexagonAPI::Global()->Alloc(bytes, alignment, isVtcm);
}

void *hexagon_runtime_alloc_2d_dsp(size_t numBlocks, size_t blockSize,
                                   uint64_t alignment, bool isVtcm) { _trc("A2");
  return HexagonAPI::Global()->Alloc(numBlocks, blockSize, alignment, isVtcm);
}

void hexagon_runtime_free_1d_dsp(void *ptr) { _trc("F1"); HexagonAPI::Global()->Free(ptr); }

void hexagon_runtime_free_2d_dsp(void *ptr) { _trc("F2"); HexagonAPI::Global()->Free(ptr); }

void hexagon_runtime_copy_dsp(void *dst, void *src, size_t nbytes,
                              bool isDstVtcm, bool isSrcVtcm) { _trc("C0");
  HexagonAPI::Global()->Copy(dst, src, nbytes);
}

/// The source pointer is a pointer to the base of memref
void *hexagon_runtime_build_crouton_dsp(void *source, size_t nbytes) { _trc("CR");
  assert(nbytes % CROUTON_SIZE == 0 &&
         "The size is expected to be a multiple of crouton size");
  return HexagonAPI::Global()->CreateBufferAlias(source, nbytes);
}

/// The source pointer is a pointer to crouton table
void *hexagon_runtime_get_contiguous_memref_dsp(void *source) { _trc("GC");
  return HexagonAPI::Global()->GetOrigBufferFromAlias(source);
}

/// Allocate (once) a resident VTCM buffer for the weight whose compile-time
/// image starts at `src`, and copy it in on the first call. `src` doubles as
/// the key: it is the address of the weight's `memref.global` in this process,
/// which the compiler computes at the call site, so residency is explicit --
/// no runtime cache keyed off a tensor's data pointer. Later calls for the same
/// address return the resident buffer with no copy; the per-launch deallocation
/// is swallowed by VtcmPool so the weight stays pinned.
void *hexagon_runtime_weight_resident_dsp(uint64_t src, uint32_t bytes) { _trc("WR");
  return HexagonAPI::Global()->WeightResident(
      src, bytes, reinterpret_cast<const void *>(static_cast<uintptr_t>(src)));
}

/// Allocate (once) a resident, uninitialised workspace buffer for `key`. `key`
/// is a compile-time constant the kernel carries (function symbol hash + the
/// buffer's index within the function), so it is stable across launches of the
/// same kernel and distinct from the address keys the weight path uses. No copy
/// happens: the workspace has no compile-time image and the kernel refills it
/// every launch. The per-launch deallocation is swallowed by VtcmPool, so the
/// buffer is allocated exactly once per process.
void *hexagon_runtime_workspace_resident_dsp(uint64_t key, uint32_t bytes) { _trc("WS");
  return HexagonAPI::Global()->WorkspaceResident(key, bytes);
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
void hexagon_runtime_hmx_ensure_dsp(void) { _trc("HE");
  HexagonAPI *api = HexagonAPI::Global();
  api->EnsureHmxLockForThisThread();
}

/// Release this thread's HMX lock. Paired 1:1 with the
/// hexagon_runtime_hmx_ensure_dsp at the function entry. A NON_SHARED unlock
/// clears the accumulators per qurt_hmx.h, so it must come after the last read,
/// which the compiler's position guarantees. Only unlocks when this thread
/// holds the lock; failures are logged, never silent.
void hexagon_runtime_hmx_unlock_dsp(void) { _trc("HU");
  HexagonAPI::Global()->ReleaseHmxLockForThisThread();
}
}
