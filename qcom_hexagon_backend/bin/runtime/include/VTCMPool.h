//===- VTCMPool.h - VTCM pool manager  ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_VTCMPOOL_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_VTCMPOOL_H

#include "HAP_compute_res.h"
#include "HexagonCommon.h"
#include <cstdint>
#include <mutex>
#include <vector>
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
#include <array>
#include <unordered_map>
#endif

class VtcmPool {
public:
  /// Allocates all of VTCM memory, and manages from the runtime
  VtcmPool();

  /// Destruction deallocates the underlying VTCM allocation.
  ~VtcmPool();

  /// Prevent copy construction of VtcmPool.
  VtcmPool(const VtcmPool &) = delete;

  /// Prevent copy assignment with VtcmPool.
  VtcmPool &operator=(const VtcmPool &) = delete;

  /// Prevent move construction.
  VtcmPool(VtcmPool &&) = delete;

  /// Prevent move assignment.
  VtcmPool &operator=(VtcmPool &&) = delete;

  /// Allocate memory from the VTCM manager with a power-of-two address
  /// alignment. Supported alignments are powers of two through 2048 bytes. The
  /// default preserves the historical request shape for callers that do not
  /// carry an alignment. If supplied, `chargedBytes` receives the legacy-rounded
  /// block size on success.
  void *Allocate(size_t nbytes);
  void *Allocate(size_t nbytes, size_t alignment);
  void *Allocate(size_t nbytes, size_t alignment, size_t *chargedBytes);
  static bool IsSupportedAlignment(size_t alignment);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  /// The runtime's size-rounding rule, exposed only to the opt-in diagnostic
  /// cache adapter. It is not a general allocator-model result: it excludes
  /// headers, address padding, and split/coalesce effects.
  static size_t SizeAlignedCharge(size_t nbytes);
#endif

  enum class ResidentKind : uint8_t {
    kWorkspace = 1,
    kWeight = 2,
  };

  enum class AccountingCoalesceCase : uint8_t {
    kNone = 0,
    kPrevious = 1,
    kNext = 2,
    kBoth = 3,
  };

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  /// The sole resident ABI label retained for address-free evidence. It is not
  /// part of the resident key and does not alter allocation or reuse decisions.
  enum class ResidentAbi : uint8_t {
    kV2 = 2,
  };

  /// The identity context is an internal, opt-in contract. Every field is
  /// supplied by the caller: the runtime never derives an ID from a pointer,
  /// a function name, or an address. The version is part of the C ABI even
  /// though the symbol is also versioned, so a future extension can reject an
  /// old producer instead of silently reinterpreting fields.
  static constexpr uint32_t kAccountingContextVersion = 1;
  static constexpr uint32_t kAccountingContextHasResidentKeyDigest = 1u << 0;

  /// Event-context ABI is independent from the process-declaration ABI above.
  /// The token is opaque to runtime; the other words are retained only to
  /// make a device report joinable to a compiler sidecar without interpreting
  /// an address. Version 1 accepts one explicitly declared grid=1 invocation.
  static constexpr uint32_t kAccountingEventContextVersion = 1;
  static constexpr uint32_t kAccountingEventContextSingleInvocation = 1u << 0;
  static constexpr uint32_t kAccountingEventContextGridOne = 1u << 1;

  struct AccountingDigest {
    // Opaque low/high words in the producer's declared order; the runtime does
    // not byte-swap, hash, or otherwise reinterpret them.
    uint64_t low{0};
    uint64_t high{0};

    bool operator==(const AccountingDigest &other) const {
      return low == other.low && high == other.high;
    }
  };

  struct AccountingContext {
    uint32_t version{0};
    uint32_t flags{0};
    AccountingDigest buildId{};
    uint64_t functionId{0};
    uint64_t allocationSiteId{0};
    /// A distinct 64-bit process observation ID, not an allocation-site
    /// identity and not the 128-bit resident-v2 scope pair. The runtime does
    /// not compare, truncate, extend, or join these values:
    /// resident_scope_binding=not-proven.
    uint64_t scopeId{0};
    /// Meaningful only when kAccountingContextHasResidentKeyDigest is set.
    /// The resident kind is attached by the operation that emits the event;
    /// this digest must describe the stable logical resident key/content
    /// contract, never a raw process address.
    AccountingDigest residentKeyDigest{};
  };

  struct AccountingEventContext {
    uint32_t version{0};
    uint32_t flags{0};
    AccountingDigest token{};
    uint64_t accountingScopeId{0};
    uint64_t invocationId{0};
    uint64_t functionId{0};
    uint64_t allocationSiteId{0};

    bool operator==(const AccountingEventContext &other) const {
      return version == other.version && flags == other.flags &&
             token == other.token && accountingScopeId == other.accountingScopeId &&
             invocationId == other.invocationId &&
             functionId == other.functionId &&
             allocationSiteId == other.allocationSiteId;
    }
  };

  /// Register one immutable process context for the accounting stream. An
  /// identical repeated registration succeeds; a different context fails
  /// closed. Registration is rejected after the first accounting event so a
  /// report cannot silently mix aggregate and joinable records. Version 1
  /// intentionally has no mid-process context switch and no per-event site
  /// binding; callers needing several functions/sites must use separate
  /// observation processes or a future ABI. The separate
  /// accounting-event-v1 ABI is the narrow, versioned exception for the
  /// one-function/one-site/grid=1 diagnostic scope. The registered
  /// function/site IDs are declared context metadata only in v1.
  /// All v1 runtime events are process-scoped and never acquire those site IDs.
  /// The optional resident
  /// digest therefore describes a process resident identity, not a function,
  /// allocation-site, or grid-instance join; a stream with several resident keys
  /// is not claimed to be fully joinable until that context-switch contract
  /// exists.
  static bool registerAccountingContext(const AccountingContext &context);
  static bool hasAccountingContext();

  /// Register one immutable per-thread event context. A repeated identical
  /// registration is idempotent; a different token or a registration after
  /// the first accounting event fails closed. Runtime never derives the token
  /// from a pointer and never persists it as an address.
  static bool registerAccountingEventContext(
      const AccountingEventContext &context);
  /// End the current kernel's per-thread event scope. This clears the TLS
  /// owner and marks the process registration historical; it cannot authorize a
  /// later invocation or another thread. The registered identity and the
  /// bound/aggregate counters are kept: a cleared scope is evidence that the
  /// owner existed, not a reason to rewind a monotonic count.
  static void clearAccountingEventContext();
  /// Whether a thread currently owns the registered event context. False after
  /// leave, after a rejected registration, and before the first enter, even
  /// when a historical registration is still recorded.
  static bool hasAccountingEventContext();

  enum class AccountingCacheEventKind : uint8_t {
    kRetain = 1,
    kHit = 2,
    kDrop = 3,
    kEvict = 4,
  };

  /// Add a bounded, address-free free-cache event to the same sequence as pool
  /// allocation events. This is deliberately a separate operation rather than
  /// a cache-key hash: the cache key is footprint data, not a stable identity.
  /// `requestedBytes` and `sizeAlignedBytes` describe the request which formed
  /// the cache entry; `chargedBytes` is the actual backing-store charge. For a
  /// mixed-key eviction the latter two are aggregate values and the event is
  /// marked non-modelable in the serialized contract.
  void recordAccountingCacheEvent(AccountingCacheEventKind kind,
                                  size_t requestedBytes,
                                  size_t allocatorInputBytes,
                                  size_t sizeAlignedBytes, size_t chargedBytes,
                                  size_t requestedAlignment, size_t cachedBytes,
                                  size_t cachedBuffers);

  /// Materialize the one scope-enter event when a context is registered after
  /// lazy pool construction. It is idempotent and emits no resource data.
  void recordAccountingScopeEvent();
#endif

  /// Allocate (once) a resident buffer for `key`. On the first call the buffer
  /// is allocated and, for a weight, filled from `src`; later calls with the
  /// exact same descriptor return the same address and do not copy. A
  /// descriptor mismatch for an existing key fails closed and never allocates
  /// another block under that key. Resident buffers are never returned to the
  /// free list by Free, so they survive every per-launch deallocation.
  void *Resident(ResidentKind kind, uint64_t key, size_t nbytes,
                 size_t alignment, const void *src);

  /// True when `ptr` points at a resident allocation.
  bool IsResident(void *ptr) const;

  /// Bytes held by resident allocations (aligned sizes, so they match the
  /// allocator's accounting).
  size_t getResidentBytes() const;

  /// Free nbytes from the allocated ptr
  void Free(void *ptr, size_t nbytes);

  /// Record a per-launch resident deallocation without releasing the pinned
  /// block. Returns false for a non-resident pointer or a raw-byte/charged-byte
  /// descriptor mismatch; allocation/reuse checks also validate alignment.
  bool FreeResident(void *ptr, size_t nbytes);

  /// Returns the total number of bytes in this pool
  size_t VtcmDeviceBytes() { return reinterpret_cast<size_t>(vtcmDeviceSize_); }

  /// Returns the total number of allocated bytes in this pool
  size_t VtcmAllocatedBytes() {
    return reinterpret_cast<size_t>(vtcmAllocatedSize_);
  }

  bool IsVtcm(void *ptr, unsigned size) {
    auto char_ptr = static_cast<char *>(ptr);
    CHECK(char_ptr != nullptr);
    auto char_vtcm = static_cast<char *>(vtcmData_);
    CHECK(vtcmData_ != nullptr);

    if (char_ptr >= char_vtcm &&
        (char_ptr + size) <= (char_vtcm + vtcmAllocatedSize_)) {
      return true;
    }
    return false;
  }

  // Query methods for diagnostics
  size_t getTotalSize() const { return vtcmAllocatedSize_; }
  size_t getTotalAllocated() const;
  size_t getTotalFree() const;
  size_t getLargestFreeBlock() const;
  size_t getNumAllocations() const { return allocations_.size(); }
  size_t getNumFreeBlocks() const { return free_.size(); }
  float getFragmentationScore() const;
  void printState() const;

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  /// A point-in-time, address-free view of the VTCM pool. All byte values are
  /// the allocator's aligned values, so this is comparable with the
  /// allocation/free event log below.
  struct AccountingSnapshot {
    size_t poolBytes;
    size_t allocatedBytes;
    size_t freeBytes;
    size_t largestFreeBytes;
    size_t allocationCount;
    size_t freeBlockCount;
    size_t residentBytes;
    size_t highWaterAllocatedBytes;
    uint32_t fragmentationPpm;
    uint64_t allocationEvents;
    uint64_t allocationFailures;
    uint64_t freeEvents;
    uint64_t residentAllocationEvents;
    uint64_t residentReuseEvents;
    uint64_t residentFreeEvents;
    uint64_t residentV2AllocationEvents;
    uint64_t residentV2ReuseEvents;
    uint64_t residentV2FreeEvents;
    uint64_t splitEvents;
    uint64_t splitPrefixBytes;
    uint64_t splitSuffixBytes;
    uint64_t coalesceEvents;
    uint64_t coalescedBlocks;
    uint64_t coalesceNoneEvents;
    uint64_t coalescePreviousEvents;
    uint64_t coalesceNextEvents;
    uint64_t coalesceBothEvents;
    size_t maxAllocationCount;
    size_t maxFreeBlockCount;
    size_t eventCount;
    size_t eventCapacity;
    uint64_t eventsDropped;
    uint64_t eventSequence;
    uint64_t eventWindowStartSequence;
    uint64_t eventWindowEndSequence;
    bool blockAccountingComplete;
    bool contextRegistered;
    uint32_t contextVersion;
    AccountingContext context;
    bool eventContextRegistered;
    /// True only while a thread is inside the enter/leave span. Registration is
    /// a historical fact that outlives the kernel, so a report must not read
    /// "registered" as "this thread still owns the token".
    bool eventContextOwnerActive;
    AccountingEventContext eventContext;
    uint64_t eventContextBoundEvents;
    uint64_t eventContextAggregateEvents;
    uint64_t scopeEvents;
    uint64_t freeCacheRetainEvents;
    uint64_t freeCacheHitEvents;
    uint64_t freeCacheDropEvents;
    uint64_t freeCacheEvictEvents;
  };

  AccountingSnapshot getAccountingSnapshot() const;
  static constexpr int kAccountingReportTruncated = -2;

  /// Write a deterministic, address-free accounting report. The return value
  /// is the number of bytes written, zero for an invalid buffer, or -2 when the
  /// report was truncated and can be retried with a larger buffer.
  int writeAccountingReport(char *buf, int cap) const;
#endif

private:
  /// Serialises the free list. A kernel launch can run on several quRT threads
  /// (`tm.exec` dispatches one program per thread) and the pool is a plain
  /// `std::vector` of segments, so unsynchronised `Allocate`/`Free` corrupt it
  /// -- measured as a dead DSP for f32 + VTCM + multi-threading. The runtime's
  /// thread pool already uses `std::mutex`, so the primitive is available here.
  /// `mutable` so the const residency query can take it.
  mutable std::mutex mutex_;

  /// Total size of VTCM memory on device
  unsigned int vtcmDeviceSize_;

  /// Total size of VTCM pool allocated on device
  unsigned int vtcmAllocatedSize_;

  /// Pointer to the beginning of the pool
  void *vtcmData_;

  /// Context for HAP_compute_res_*
  unsigned int contextId_{0};

  /// List of allocations
  std::vector<std::pair<char *, size_t>> allocations_;

  /// Resident identity is a process-local descriptor, not just a key. Keeping
  /// every charged field next to the pointer makes duplicate validation exact:
  /// a same-key request with a different kind, size, alignment, or allocator
  /// charge cannot accidentally reuse the block or allocate a second one.
  struct ResidentDescriptor {
    ResidentKind kind;
    uint64_t key;
    size_t bytes;
    size_t alignment;
    size_t chargedBytes;
  };
  struct ResidentBlock {
    ResidentDescriptor descriptor;
    char *ptr;
  };
  std::vector<ResidentBlock> resident_;

  /// Locked allocator body, shared by Allocate and Resident. The extra
  /// resident bit exists only in the opt-in accounting build; the default
  /// build keeps the original allocator ABI.
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  char *allocateLocked(size_t nbytes, size_t alignment, bool residentAllocation,
                       uint8_t residentKind);
#else
  char *allocateLocked(size_t nbytes, size_t alignment);
#endif

  static bool residentDescriptorsMatch(const ResidentDescriptor &lhs,
                                       const ResidentDescriptor &rhs);

  /// True when `ptr` is resident; caller must hold mutex_.
  bool isResidentLocked(char *ptr) const;

  /// Validate and account a resident deallocation; caller must hold mutex_.
  bool freeResidentLocked(char *ptr, size_t nbytes);

  /// Resident accounting query for callers that already hold mutex_.
  size_t getResidentBytesLocked() const;

  /// List of free segments
  std::vector<std::pair<char *, size_t>> free_;

  /// Pointer to allocated VTCM (for offset calculations)
  void *vtcmAllocatedPtr_{nullptr};

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  /// Keep the probe bounded: the report records the latest events and exposes
  /// the sequence/drop count, rather than allocating an unbounded log on the
  /// DSP. Events contain no addresses.
  static constexpr size_t kAccountingEventCapacity = 256;

  struct AccountingEvent {
    uint64_t sequence;
    uint8_t kind;
    uint8_t residentKind;
    uint8_t residentAbi;
    size_t requestedBytes;
    size_t allocatorInputBytes;
    size_t sizeAlignedBytes;
    size_t chargedBytes;
    size_t requestedAlignment;
    size_t effectiveAlignment;
    uint8_t coalesceCase;
    uint32_t coalescedBlocks;
    size_t allocatedBytes;
    size_t freeBytes;
    size_t largestFreeBytes;
    size_t allocationCount;
    size_t freeBlockCount;
    size_t residentBytes;
    size_t cachedBytes;
    size_t cachedBuffers;
    uint32_t fragmentationPpm;
    // This is a snapshot of the explicitly registered context, not a value
    // reconstructed from an allocation pointer. Version 1 process events clear
    // function/site fields before storage; a zero/absent context is aggregate.
    AccountingContext context;
    bool contextRegistered;
    AccountingEventContext eventContext;
    bool eventContextBound;
  };

  struct AccountingOwner {
    AccountingEventContext context;
    uint64_t threadOrdinal{0};
    bool bound{false};
  };

  std::array<AccountingEvent, kAccountingEventCapacity> accountingEvents_{};
  size_t accountingEventCount_{0};
  size_t accountingEventStart_{0};
  uint64_t accountingEventsDropped_{0};
  uint64_t accountingEventSequence_{0};
  size_t accountingHighWaterAllocated_{0};
  uint64_t accountingAllocationEvents_{0};
  uint64_t accountingAllocationFailures_{0};
  uint64_t accountingFreeEvents_{0};
  uint64_t accountingResidentAllocationEvents_{0};
  uint64_t accountingResidentReuseEvents_{0};
  uint64_t accountingResidentFreeEvents_{0};
  uint64_t accountingResidentV2AllocationEvents_{0};
  uint64_t accountingResidentV2ReuseEvents_{0};
  uint64_t accountingResidentV2FreeEvents_{0};
  uint64_t accountingSplitEvents_{0};
  uint64_t accountingSplitPrefixBytes_{0};
  uint64_t accountingSplitSuffixBytes_{0};
  uint64_t accountingCoalesceEvents_{0};
  uint64_t accountingCoalescedBlocks_{0};
  uint64_t accountingCoalesceNoneEvents_{0};
  uint64_t accountingCoalescePreviousEvents_{0};
  uint64_t accountingCoalesceNextEvents_{0};
  uint64_t accountingCoalesceBothEvents_{0};
  size_t accountingMaxAllocationCount_{0};
  size_t accountingMaxFreeBlockCount_{0};
  uint64_t accountingEventContextBoundEvents_{0};
  uint64_t accountingEventContextAggregateEvents_{0};
  bool accountingScopeEventRecorded_{false};
  std::unordered_map<void *, AccountingOwner> accountingOwners_;

  AccountingSnapshot getAccountingSnapshotLocked() const;
  AccountingOwner currentAccountingOwner() const;
  void rememberAccountingOwnerLocked(void *ptr,
                                     const AccountingOwner &owner);
  AccountingOwner lookupAccountingOwnerLocked(void *ptr) const;
  void recordAccountingSplitLocked(size_t prefixBytes, size_t suffixBytes);
  void recordAccountingAllocationLocked(
      uint8_t kind, size_t requestedBytes, size_t sizeAlignedBytes,
      size_t chargedBytes, bool success, uint8_t residentKind,
      size_t requestedAlignment = 0, size_t effectiveAlignment = 0,
      const AccountingOwner *owner = nullptr);
  void recordAccountingReuseLocked(size_t requestedBytes,
                                   size_t sizeAlignedBytes,
                                   size_t chargedBytes,
                                   uint8_t residentKind,
                                   const AccountingOwner *owner = nullptr);
  void recordAccountingFreeLocked(uint8_t kind, size_t requestedBytes,
                                  size_t sizeAlignedBytes,
                                  size_t chargedBytes, uint8_t residentKind,
                                  uint8_t coalesceCase = 0,
                                  const AccountingOwner *owner = nullptr);
  void recordAccountingEventLocked(
      uint8_t kind, size_t requestedBytes, size_t allocatorInputBytes,
      size_t sizeAlignedBytes, size_t chargedBytes, uint8_t residentKind,
      ResidentAbi residentAbi,
      size_t requestedAlignment, size_t effectiveAlignment,
      uint8_t coalesceCase, uint32_t coalescedBlocks, size_t cachedBytes,
      size_t cachedBuffers,
      bool emitScopeEvent = true,
      const AccountingOwner *owner = nullptr);
  int writeAccountingReportLocked(char *buf, int cap) const;
  static uint32_t fragmentationPpmLocked(size_t totalFree, size_t largestFree);
#endif

  // Allocation helpers. `nbytes` is the charged (legacy-rounded) block size.
  char *tryAllocateFromEnd(size_t nbytes, size_t alignment);
  char *allocateBestFit(size_t nbytes, size_t alignment);
  void handleAllocationFailure(size_t nbytes);
  void logAllocationSuccess(char *ptr, size_t nbytes);

  // Free helpers
  size_t coalesceAndAddToFreeList(char *ptr, size_t nbytes,
                                 uint8_t *coalesceCase = nullptr);
  void logFreeSuccess(char *ptr, size_t nbytes, size_t numCoalesced);

  // Validation (debug builds only)
  void validateInvariants() const;
  bool checkMemoryAccountedFor() const;

  /// Debug only dump of the state of the lists
  void DebugDump();
};

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_VTCMPOOL_H
