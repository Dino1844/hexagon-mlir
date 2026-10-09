//===- HexagonAPI.h - device API   to compile run on Hexagon --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Hexagon Device API that is compiled and run on Hexagon.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INC_HEXAGONAPI_H_
#define HEXAGON_BIN_RUNTIME_INC_HEXAGONAPI_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "HAP_compute_res.h"
#include "HAP_power.h"

#include "BufferManager.h"
#include "HexagonCommon.h"
#include "VTCMPool.h"

/// BRINGUP probe hooks (exp/hmx/bringup_probe). They time each step of the
/// one-shot runtime bring-up and record it; the probe reads the recording with
/// hexagon_runtime_bringup_report(). Compiled to nothing unless the probe build
/// defines HEXMLIR_RUNTIME_BRINGUP_PROBE (-DHEXMLIR_BRINGUP_PROBE=ON), so the
/// shipping runtime is unchanged.
#ifdef HEXMLIR_RUNTIME_BRINGUP_PROBE
void BringupProbeBegin();
void BringupProbeStep(const char *name);
void BringupProbeEnd();
#else
inline void BringupProbeBegin() {}
inline void BringupProbeStep(const char *) {}
inline void BringupProbeEnd() {}
#endif

/// Hexagon Device API that is compiled and run on Hexagon.
class HexagonAPI {
public:
  /// Retrieve the global singleton instance of the HexagonAPI.
  static HexagonAPI *Global();

  /// Constructor
  HexagonAPI() { this->AcquireResources(); }

  /// Destructor
  ~HexagonAPI() { this->ReleaseResources(); }

  /// Ensures resource managers are in a good state for the runtime
  void AcquireResources() {
    BringupProbeBegin();
    CHECK_EQ(runtimeVtcm, nullptr);
    runtimeVtcm = std::make_unique<VtcmPool>();
    BringupProbeStep("vtcm_pool_ctor");

    CHECK_EQ(bufferManager, nullptr);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    bufferManager = std::make_unique<BufferManager>(runtimeVtcm.get());
#else
    bufferManager = std::make_unique<BufferManager>();
#endif
    BringupProbeStep("buffer_manager_ctor");

    hmx_context_id = initialize_and_acquire_hmx();
    BringupProbeEnd();
  }

  /// Ensures all runtime resources are freed
  void ReleaseResources() {
    // The DDR mirrors are live BufferManager allocations, and BufferManager's
    // own destructor refuses to be destroyed while one is outstanding, so they
    // go back first. See WeightResidentDdrV2 for why they are not freed per
    // launch.
    freeDdrWeightResidents();
    CHECK((bufferManager), "bufferManager was not created in AcquireResources");
    bufferManager.reset();

    CHECK((runtimeVtcm), "runtimeVtcm was not created in AcquireResources");
    runtimeVtcm.reset();

    release_hmx();
  }

  /// Returns VtcmPool instance
  VtcmPool *getVtcmPool() {
    CHECK((runtimeVtcm), "runtimeVtcm has not been created");
    return runtimeVtcm.get();
  }

  /// Returns BufferManager instance
  BufferManager *getBufferManager() {
    CHECK((bufferManager), "bufferManager has not been created");
    return bufferManager.get();
  }

  /// Returns True, if the resources are initialized
  bool hasResources() {
    if (runtimeVtcm == nullptr || bufferManager == nullptr)
      return false;
    return true;
  }

  /// Allocate a single, contiguous memory region.
  void *Alloc(size_t nbytes, uint64_t alignment, bool isVtcm);

  /// Allocate the region(s) needed for Hexagon's indirect-tensor format.
  void *Alloc(size_t nallocs, size_t nbytes, uint64_t alignment, bool isVtcm);

  /// Allocate (once) a resident VTCM buffer through the sole versioned path.
  /// The key/source, requested byte count, and address alignment form an exact
  /// descriptor: only an identical request may reuse a process-local resident
  /// block. Scope registration is enforced by the C API before this method is
  /// reached.
  void *WeightResidentV2(uint64_t source, size_t nbytes, size_t alignment);

  /// Allocate (once) a **permanent DDR mirror** of the same image, for a weight
  /// whose crouton image does not fit the VTCM pool. Same versioned contract as
  /// `WeightResidentV2` -- key = source address, exact descriptor on reuse, one
  /// image per process -- but the buffer comes out of ordinary DDR instead of
  /// the pool, so it costs no VTCM and the kernel takes the block it needs from
  /// it with a contiguous copy instead of packing a strided view in place.
  ///
  /// Lifetime (docs/hmx/pack-redundancy-fix-plan-2026-10-09.md §7-2): the
  /// mirror lives as long as the runtime does and is returned in
  /// `ReleaseResources`, before the allocator it came from is destroyed. There
  /// is deliberately no per-launch free: the kernel never holds it, and a
  /// mirror that vanished between launches would turn the fetch into a
  /// use-after-free rather than a copy.
  void *WeightResidentDdrV2(uint64_t source, size_t nbytes, size_t alignment);
  /// `instance` is the caller's flat program id: concurrent instances of a
  /// grid>1 launch carry distinct pids, so each gets its own resident buffer,
  /// and the same pid across launches reuses the same buffer. A thread id
  /// would be the wrong discriminator: the wrapper's ThreadManager spawns
  /// fresh qurt threads per launch (multithreading.h's "keep the thread pool
  /// alive" TODO), so thread-keyed residency would allocate a never-reused
  /// buffer set every launch and grow the resident map without bound
  /// (measured: mha_fa grid=4, +73%, 2026-10-04).
  void *WorkspaceResidentV2(uint64_t key, size_t nbytes, size_t alignment,
                            uint32_t instance);

  /// Takes a `ptr` to the base of the memref and returns a pointer to the
  /// crouton table
  void *CreateBufferAlias(void *ptr, size_t nbytes);

  /// Takes the pointer to crouton table that was created as an alias and
  /// returns the base pointer to the memref
  void *GetOrigBufferFromAlias(void *aliasPtr);

  /// Frees the allocated memory region.
  void Free(void *ptr);

  /// Record a resident deallocation while keeping the pinned block alive.
  bool FreeResident(void *ptr, size_t nbytes);

  /// Copies the data from source src into destination dst.
  void Copy(void *dst, void *src, size_t nbytes);

  /// Lock the HMX unit to the calling thread; idempotent per thread.
  ///
  /// Invariant (lock pairing): ensure locks only when this thread does not
  /// already hold the lock (thread_local flag), and unlock
  /// (ReleaseHmxLockForThisThread) unlocks only when it does; the compiler
  /// emits one ensure at each function entry that issues HMX leaves and one
  /// unlock before each return, so every span pairs one lock with one unlock,
  /// no more, no less. A NOT_SHARED unlock clears the accumulators
  /// (qurt_hmx.h), so the unlock must come after the last read, which the
  /// read-out's position guarantees.
  ///
  /// The HMX lock is owned by a single thread (HAP_compute_res.h: the thread
  /// that executes HMX must be the thread that holds a valid lock on the one
  /// shared context). The constructor acquires the context but does NOT keep
  /// the unit lock, so the thread that runs HMX locks it for itself first. A
  /// plain hmx_lock on an already-held unit blocks until the holder releases
  /// it; that blocking wait is the serialization for the single HMX engine and
  /// matches hmx_lock2 NON_SHARED semantics without taking on caller-side
  /// timesharing.
  ///
  /// The per-thread "already locked" state is a thread_local flag (see the
  /// definition). The singleton itself is intentionally leaked (release_hmx
  /// never runs); the lock is released before each return, so sequential
  /// kernels on one thread re-lock every time with no leak.
  void EnsureHmxLockForThisThread();

  /// Unlock the HMX unit from the calling thread; no-op unless this thread
  /// holds the lock (thread_local flag). Second half of the pairing above:
  /// called before each function return, clearing the flag so the thread's
  /// next HMX kernel locks again.
  void ReleaseHmxLockForThisThread();

private:
  /// Manages runtime HexagonBuffer allocations
  std::unique_ptr<BufferManager> bufferManager;

  /// VTCM memory manager
  std::unique_ptr<VtcmPool> runtimeVtcm;

  /// The permanent DDR weight mirrors, keyed by the source address the same way
  /// `VtcmPool::Resident` keys its blocks: one process, one image per key.
  /// Values are (buffer, bytes) so a repeat request with a different byte count
  /// is a descriptor mismatch rather than a silently shorter copy.
  std::mutex ddrMirrorMutex;
  std::unordered_map<uint64_t, std::pair<void *, size_t>> ddrMirrors;

  /// Return every mirror to `bufferManager`. Called from `ReleaseResources`
  /// before the manager is destroyed; safe to call twice (the map is cleared).
  void freeDdrWeightResidents();

  /// HMX Context
  uint32_t hmx_context_id = 0;

  /// Power on and lock HMX context
  uint32_t initialize_and_acquire_hmx();

  void release_hmx() {
    if (hmx_context_id) {
      HAP_compute_res_hmx_unlock(hmx_context_id);
      HAP_compute_res_release(hmx_context_id);
      hmx_context_id = 0;
    }
  }
};
#endif // HEXAGON_BIN_RUNTIME_INC_HEXAGONAPI_H_
