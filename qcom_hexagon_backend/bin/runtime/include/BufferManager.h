//===- BufferManager.h - hexagon buffer manager        --------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef BUFFERMANAGER_H_
#define BUFFERMANAGER_H_

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
#include <cstdint>
#endif

#include "HexagonBuffer.h"
#include "HexagonBufferAlias.h"
#include "HexagonCommon.h"
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
#include "VTCMPool.h"
#endif

class HexagonBufferAlias;

/// Hashes a HexagonBuffer::CacheKey. Defined by hand (not std::hash) so the key
/// stays a plain aggregate owned by HexagonBuffer.
struct HexagonBufferCacheKeyHash {
  size_t operator()(const HexagonBuffer::CacheKey &key) const {
    size_t h = std::hash<size_t>{}(key.ndim);
    auto mix = [&h](size_t value) {
      h ^= value + 0x9e3779b9u + (h << 6) + (h >> 2);
    };
    mix(std::hash<size_t>{}(key.numAllocations));
    mix(std::hash<size_t>{}(key.bytesPerAllocation));
    mix(std::hash<size_t>{}(key.alignment));
    mix(std::hash<bool>{}(key.isVtcm));
    return h;
  }
};

class BufferManager {
public:
  BufferManager() = default;
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  /// The optional pool link is used only by the opt-in diagnostic stream. It
  /// keeps free-cache events in the same bounded sequence as pool events
  /// without hashing the cache key or retaining any address in the report.
  explicit BufferManager(VtcmPool *accountingPool)
      : accountingPool_(accountingPool) {}
  /// Address-free counters for the optional VTCM accounting report. The
  /// cumulative counters are kept separately from the current cache occupancy
  /// so a report can distinguish reuse from retention.
  struct AccountingSnapshot {
    uint64_t freeCacheHits;
    uint64_t freeCacheHitBytes;
    uint64_t freeCacheHitRequestedBytes;
    uint64_t freeCacheRetains;
    uint64_t freeCacheRetainedBytes;
    uint64_t freeCacheRetainRequestedBytes;
    uint64_t freeCacheDrops;
    uint64_t freeCacheDropBytes;
    uint64_t freeCacheEvictions;
    uint64_t freeCacheEvictionBytes;
    // One allocation-failure eviction drops the whole VTCM-accounted cache, so
    // a single event can evict several buffers at once. This cumulative *event*
    // count is the only counterpart of the retained-ring
    // `free_cache_evict_events`; the buffer/byte counters above are a different
    // unit and can only bound it (one event evicts at least one buffer).
    uint64_t freeCacheEvictionEvents;
    // Historical retained-cache high-water is separate from the pool's
    // charged-allocation high-water. Keeping both ledgers prevents a cache
    // retention number from being mistaken for full VTCM occupancy.
    uint64_t highWaterCachedVtcmBytes;
    uint64_t highWaterCachedBuffers;
    size_t cachedBuffers;
    size_t cachedBytes;
    size_t cachedVtcmBytes;
    size_t cacheCapacityBytes;
  };

  AccountingSnapshot getAccountingSnapshot();
#endif

  ~BufferManager() {
    if (!bufferMap_.empty()) {
      CHECK((true), "BufferManager is not empty upon destruction");
    }
  }

  /// Free a HexagonBuffer.
  ///
  /// The wrapper (HexagonBuffer + its Allocation and vectors) and its reserved
  /// storage are kept in a free cache keyed by footprint, so the next launch's
  /// same-shaped allocation reuses them instead of rebuilding the object and
  /// re-running the pool's best-fit/coalesce bookkeeping. The pointer is
  /// *removed* from bufferMap_ either way, so a use-after-free still fails the
  /// same CHECK/Copy lookups as before; the cached buffer is simply not "live".
  void FreeHexagonBuffer(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = bufferMap_.find(ptr);
    CHECK((it != bufferMap_.end()),
          "Attempt made to free unknown or already freed allocation");
    CHECK(it->second != nullptr);
    std::unique_ptr<HexagonBuffer> buf = std::move(it->second);
    const HexagonBuffer::CacheKey cacheKey = buf->GetCacheKey();
    const size_t bytes = buf->GetAllocatedBytes();
    bufferMap_.erase(it);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    size_t requestedBytes = 0;
    size_t allocatorInputBytes = 0;
    size_t sizeAlignedBytes = 0;
    const bool vtcmAccounting =
        cacheAccountingSizes(cacheKey, requestedBytes, allocatorInputBytes,
                             sizeAlignedBytes);
#endif
    if (bytes != 0 && bytes <= kMaxCachedBytes &&
        cachedBytes_ + bytes <= kMaxCachedBytes) {
      cachedBytes_ += bytes;
      freeCache_[cacheKey].push_back(std::move(buf));
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
      if (vtcmAccounting) {
        ++freeCacheRetains_;
        freeCacheRetainedBytes_ += bytes;
        freeCacheRetainRequestedBytes_ += requestedBytes;
        cachedVtcmBytes_ += bytes;
        ++cachedBufferCount_;
        if (cachedVtcmBytes_ > highWaterCachedVtcmBytes_)
          highWaterCachedVtcmBytes_ = cachedVtcmBytes_;
        if (cachedBufferCount_ > highWaterCachedBuffers_)
          highWaterCachedBuffers_ = cachedBufferCount_;
        recordAccountingCacheEventLocked(
            VtcmPool::AccountingCacheEventKind::kRetain, requestedBytes,
            allocatorInputBytes, sizeAlignedBytes, bytes, cacheKey.alignment,
            ptr);
        validateAccountingStateLocked();
      }
#endif
    }
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    else if (vtcmAccounting) {
      ++freeCacheDrops_;
      freeCacheDropBytes_ += bytes;
      recordAccountingCacheEventLocked(
          VtcmPool::AccountingCacheEventKind::kDrop, requestedBytes,
          allocatorInputBytes, sizeAlignedBytes, bytes, cacheKey.alignment,
          ptr);
    }
#endif
    // Otherwise `buf` is destroyed at the end of this scope and the block
    // returns to the pool (the bounded-cache fallback).
  }

  /// Allocate a HexagonBuffer.
  template <typename... Args> void *AllocateHexagonBuffer(Args &&...args) {
    std::lock_guard<std::mutex> lock(mutex_);

    HexagonBuffer::CacheKey key = MakeCacheKey(args...);
    auto cached = freeCache_.find(key);
    if (cached != freeCache_.end() && !cached->second.empty()) {
      std::unique_ptr<HexagonBuffer> buf = std::move(cached->second.back());
      cached->second.pop_back();
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
      const size_t bytes = buf->GetAllocatedBytes();
      size_t requestedBytes = 0;
      size_t allocatorInputBytes = 0;
      size_t sizeAlignedBytes = 0;
      const bool vtcmAccounting = cacheAccountingSizes(
          key, requestedBytes, allocatorInputBytes, sizeAlignedBytes);
      cachedBytes_ -= bytes;
      if (vtcmAccounting) {
        ++freeCacheHits_;
        freeCacheHitBytes_ += bytes;
        freeCacheHitRequestedBytes_ += requestedBytes;
        assert(cachedVtcmBytes_ >= bytes);
        assert(cachedBufferCount_ != 0);
        cachedVtcmBytes_ -= bytes;
        if (cachedBufferCount_ != 0)
          --cachedBufferCount_;
        recordAccountingCacheEventLocked(
            VtcmPool::AccountingCacheEventKind::kHit, requestedBytes,
            allocatorInputBytes, sizeAlignedBytes, bytes, key.alignment,
            buf->GetPointer());
        validateAccountingStateLocked();
      }
#else
      cachedBytes_ -= buf->GetAllocatedBytes();
#endif
      void *ptr = buf->GetPointer();
      bufferMap_.insert({ptr, std::move(buf)});
      return ptr;
    }

    auto buf = std::make_unique<HexagonBuffer>(std::forward<Args>(args)...);
    if (!buf->HasValidAllocation()) {
      // The pool could not satisfy the request while the cache was holding
      // storage: drop the cache (returning those blocks to the pool) and run
      // the real allocator once more. The cache is a perf aid, never a reason
      // to fail an allocation that would otherwise succeed.
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
      size_t evictedBytes = 0;
      size_t evictedRequestedBytes = 0;
      size_t evictedAllocatorInputBytes = 0;
      size_t evictedSizeAlignedBytes = 0;
      size_t evictedBuffers = 0;
      for (const auto &entry : freeCache_) {
        size_t requestedBytes = 0;
        size_t allocatorInputBytes = 0;
        size_t sizeAlignedBytes = 0;
        if (!cacheAccountingSizes(entry.first, requestedBytes,
                                  allocatorInputBytes, sizeAlignedBytes))
          continue;
        evictedBuffers += entry.second.size();
        freeCacheEvictions_ += entry.second.size();
        for (const auto &cachedBuffer : entry.second) {
          const size_t charged = cachedBuffer->GetAllocatedBytes();
          evictedBytes += charged;
          freeCacheEvictionBytes_ += charged;
        }
        evictedRequestedBytes +=
            checkedMultiply(requestedBytes, entry.second.size());
        evictedAllocatorInputBytes +=
            checkedMultiply(allocatorInputBytes, entry.second.size());
        evictedSizeAlignedBytes +=
            checkedMultiply(sizeAlignedBytes, entry.second.size());
      }
      assert(evictedBuffers == cachedBufferCount_);
      assert(evictedBytes == cachedVtcmBytes_);
      cachedVtcmBytes_ = 0;
      cachedBufferCount_ = 0;
#endif
      freeCache_.clear();
      cachedBytes_ = 0;
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
      validateAccountingStateLocked();
      if (evictedBuffers != 0) {
        ++freeCacheEvictionEvents_;
        // No owner block: `freeCache_` has already been cleared, so there is no
        // single block this batch event is about. Nothing is lost by that --
        // every evicted block is released through the pool's own free path,
        // which records that block's site individually. This event stays a
        // cache-level fact rather than borrowing one block's identity for the
        // whole batch.
        recordAccountingCacheEventLocked(
            VtcmPool::AccountingCacheEventKind::kEvict, evictedRequestedBytes,
            evictedAllocatorInputBytes, evictedSizeAlignedBytes, evictedBytes, 0,
            nullptr);
      }
#endif
      buf = std::make_unique<HexagonBuffer>(std::forward<Args>(args)...);
    }
    if (!buf->HasValidAllocation())
      return nullptr;

    void *ptr = buf->GetPointer();
    bufferMap_.insert({ptr, std::move(buf)});
    return ptr;
  }

  /// Returns a pointer to crouton-table that is constructed using `nBytes`
  /// sized `buffer`. The table size is expected to be `nBytes/CROUTON_SIZE`
  void *CreateBufferAlias(void *ptr, size_t nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    HexagonBuffer *buffer = FindBuffer<HexagonBuffer>(ptr, bufferMap_);
    auto bufferAlias =
        std::make_unique<hexagon::HexagonBufferAlias>(*buffer, nbytes);

    auto *returnPtr = bufferAlias->GetCroutonTableBase();
    // bufferAliasMap_ would be needed to find aliases given a base pointer when
    // using HexagonBuffer::Copy calls
    // TODO: Modify HexagonBuffer::Copy to support aliases
    bufferAliasMap_.insert({returnPtr, std::move(bufferAlias)});
    return returnPtr;
  }

  /// Takes the crouton pointer table and returns the base pointer to the
  /// contiguous memref underneath
  void *GetOrigBufferFromAlias(void *croutonTablePtr) {
    std::lock_guard<std::mutex> lock(mutex_);
    hexagon::HexagonBufferAlias *aliasPtr =
        FindBuffer<hexagon::HexagonBufferAlias>(croutonTablePtr,
                                                bufferAliasMap_);
    assert(aliasPtr != nullptr &&
           "Expected the ptr to be a valid buffer alias created by "
           "memref_to_crouton op");
    auto *buffer = aliasPtr->origBuffer;
    return buffer->GetPointer();
  }

  /// Finds and returns the pointer from the given map if it exists and a
  /// nullptr otherwise
  template <typename BufferType>
  BufferType *
  FindBuffer(void *ptr,
             std::unordered_map<void *, std::unique_ptr<BufferType>> &map_) {
    auto it = map_.find(ptr);
    if (it != map_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  /// HexagonBuffer copy operations
  void Copy(void *dst, void *src, size_t nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    HexagonBuffer *hb_src = FindBuffer<HexagonBuffer>(src, bufferMap_);
    HexagonBuffer *hb_dst = FindBuffer<HexagonBuffer>(dst, bufferMap_);

    bool isSrcHb = (hb_src != nullptr);
    bool isDstHb = (hb_dst != nullptr);

    if (isSrcHb && isDstHb) {
      hb_dst->CopyFrom(*hb_src, nbytes);
    } else if (isSrcHb) {
      hb_src->CopyTo(dst, nbytes);
    } else if (isDstHb) {
      hb_dst->CopyFrom(src, nbytes);
    } else {
      CHECK((false), "One of the src/dst should be a hexagon buffer");
    }
  }

private:
  /// Contains the HexagonBuffer objects managed by this class.
  /// Serialises the buffer bookkeeping. A launch can run on several quRT
  /// threads (one program per thread), and the maps below are plain containers,
  /// so concurrent allocate/free corrupts them (measured: dead DSP when VTCM
  /// and multi-threading are both on).
  std::mutex mutex_;

  std::unordered_map<void *, std::unique_ptr<HexagonBuffer>> bufferMap_;

  /// Contains the HexagonBufferAlias objects managed by this class.
  std::unordered_map<void *, std::unique_ptr<hexagon::HexagonBufferAlias>>
      bufferAliasMap_;

  /// Upper bound on the bytes the free cache may hold. Sized to a per-launch
  /// activation set for the HMX paths (S1 is ~1.2 MiB: a 1 MiB accumulator plus
  /// 128 KiB/64 KiB operands) with headroom, while staying a small fraction of
  /// a multi-MiB VTCM pool so live allocations are never starved. This is a
  /// perf heuristic only: allocation failure drops the whole cache and retries,
  /// so the bound cannot turn a would-succeed allocation into a failure.
  static constexpr size_t kMaxCachedBytes = 2u * 1024 * 1024;

  /// Build the cache key from an alloc request. Overloaded on arity: the 3-arg
  /// form is the 1-D request (bytes, alignment, isVtcm) and sets ndim=1; the
  /// 4-arg form is the 2-D request (nallocs, bytes, alignment, isVtcm) and sets
  /// ndim=2. These are exactly the two HexagonBuffer constructor shapes.
  static HexagonBuffer::CacheKey MakeCacheKey(size_t nbytes, size_t alignment,
                                              bool isVtcm) {
    return HexagonBuffer::CacheKey{1, 1, nbytes, alignment, isVtcm};
  }
  static HexagonBuffer::CacheKey MakeCacheKey(size_t nallocs, size_t nbytes,
                                              size_t alignment, bool isVtcm) {
    return HexagonBuffer::CacheKey{2, nallocs, nbytes, alignment, isVtcm};
  }

  /// Freed-but-still-reserved buffers, keyed by footprint. LIFO per key so the
  /// most recently freed block (still the hottest in cache) is handed back
  /// first.
  std::unordered_map<HexagonBuffer::CacheKey,
                     std::vector<std::unique_ptr<HexagonBuffer>>,
                     HexagonBufferCacheKeyHash>
      freeCache_;

  /// Bytes currently held by freeCache_, charged on the aligned reservation so
  /// it matches the VTCM the cache is pinning.
  size_t cachedBytes_ = 0;

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  VtcmPool *accountingPool_{nullptr};

  /// Cache state is process-scoped; the pool event deliberately does not
  /// inherit the *current* function/allocation-site context. It does carry the
  /// owner retained with the block being retained or hit, which is the site that
  /// allocated those bytes -- a different fact from "who is running now", and
  /// the only one that makes the cached bytes attributable.
  void recordAccountingCacheEventLocked(
      VtcmPool::AccountingCacheEventKind kind, size_t requestedBytes,
      size_t allocatorInputBytes, size_t sizeAlignedBytes, size_t chargedBytes,
      size_t requestedAlignment, void *ownerBlock) const {
    if (accountingPool_ != nullptr) {
      // The pool resolves its own retained owner from the block pointer, so an
      // unattributed block simply yields no owner and the event stays
      // aggregate. That is deliberate: a block must not pick up an identity just
      // because the free cache happens to be in its path.
      accountingPool_->recordAccountingCacheEvent(
          kind, requestedBytes, allocatorInputBytes, sizeAlignedBytes,
          chargedBytes, requestedAlignment, cachedVtcmBytes_,
          cachedBufferCount_, ownerBlock);
    }
  }

  static size_t checkedMultiply(size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
      return std::numeric_limits<size_t>::max();
    return lhs * rhs;
  }

  static size_t alignUpForCache(size_t bytes, size_t alignment) {
    if (alignment == 0)
      return std::numeric_limits<size_t>::max();
    const size_t remainder = bytes % alignment;
    if (remainder == 0)
      return bytes;
    const size_t padding = alignment - remainder;
    if (bytes > std::numeric_limits<size_t>::max() - padding)
      return std::numeric_limits<size_t>::max();
    return bytes + padding;
  }

  void validateAccountingStateLocked() const {
    size_t allBytes = 0;
    size_t vtcmBytes = 0;
    size_t vtcmBuffers = 0;
    for (const auto &entry : freeCache_) {
      for (const auto &buffer : entry.second) {
        const size_t charged = buffer->GetAllocatedBytes();
        allBytes += charged;
        size_t requestedBytes = 0;
        size_t allocatorInputBytes = 0;
        size_t sizeAlignedBytes = 0;
        if (cacheAccountingSizes(entry.first, requestedBytes,
                                  allocatorInputBytes, sizeAlignedBytes)) {
          vtcmBytes += charged;
          ++vtcmBuffers;
        }
      }
    }
    assert(cachedBytes_ == allBytes);
    assert(cachedVtcmBytes_ == vtcmBytes);
    assert(cachedBufferCount_ == vtcmBuffers);
    assert(cachedVtcmBytes_ <= cachedBytes_);
  }

  static bool cacheAccountingSizes(const HexagonBuffer::CacheKey &key,
                                   size_t &requestedBytes,
                                   size_t &allocatorInputBytes,
                                   size_t &sizeAlignedBytes) {
    // `requestedBytes` is the raw BufferManager request. For a 2-D VTCM
    // buffer, HexagonBuffer first pads each block to the requested address
    // alignment before handing the monolithic region to VtcmPool; that
    // intermediate value is `allocatorInputBytes`. The allocator's independent
    // 128/2048 size quantum is applied only afterwards. DDR keys are outside
    // this VTCM accounting stream altogether.
    requestedBytes = checkedMultiply(key.numAllocations,
                                     key.bytesPerAllocation);
    allocatorInputBytes = 0;
    sizeAlignedBytes = 0;
    if (!key.isVtcm)
      return false;

    size_t perAllocation = key.bytesPerAllocation;
    // A 2-D allocation pads every block, including a one-block allocation;
    // use dimensionality rather than block count to decide this intermediate
    // allocator-input unit.
    if (key.ndim == 2)
      perAllocation = alignUpForCache(perAllocation, key.alignment);
    allocatorInputBytes = checkedMultiply(key.numAllocations, perAllocation);
    sizeAlignedBytes = VtcmPool::SizeAlignedCharge(allocatorInputBytes);
    return true;
  }

  uint64_t freeCacheHits_ = 0;
  uint64_t freeCacheHitBytes_ = 0;
  uint64_t freeCacheHitRequestedBytes_ = 0;
  uint64_t freeCacheRetains_ = 0;
  uint64_t freeCacheRetainedBytes_ = 0;
  uint64_t freeCacheRetainRequestedBytes_ = 0;
  uint64_t freeCacheDrops_ = 0;
  uint64_t freeCacheDropBytes_ = 0;
  uint64_t freeCacheEvictions_ = 0;
  uint64_t freeCacheEvictionBytes_ = 0;
  uint64_t freeCacheEvictionEvents_ = 0;
  uint64_t highWaterCachedVtcmBytes_ = 0;
  uint64_t highWaterCachedBuffers_ = 0;
  size_t cachedBufferCount_ = 0;
  size_t cachedVtcmBytes_ = 0;
#endif
};

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
inline BufferManager::AccountingSnapshot
BufferManager::getAccountingSnapshot() {
  std::lock_guard<std::mutex> lock(mutex_);
  return AccountingSnapshot{
      freeCacheHits_,          freeCacheHitBytes_,
      freeCacheHitRequestedBytes_, freeCacheRetains_,
      freeCacheRetainedBytes_, freeCacheRetainRequestedBytes_,
      freeCacheDrops_,         freeCacheDropBytes_, freeCacheEvictions_,
      freeCacheEvictionBytes_, freeCacheEvictionEvents_,
      highWaterCachedVtcmBytes_,
      highWaterCachedBuffers_, cachedBufferCount_, cachedBytes_,
      cachedVtcmBytes_, kMaxCachedBytes};
}
#endif

#endif // BUFFERMANAGER_H_
