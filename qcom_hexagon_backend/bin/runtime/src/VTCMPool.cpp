//===- VTCMPool.cpp - VTCM pool                                    --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "VTCMPool.h"
#include "HAP_compute_res.h"
#include "HexagonCommon.h"

#if defined(__hexagon__) || defined(HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE)
#include <cstdio>
#endif

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
#include <cstdarg>
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

//===----------------------------------------------------------------------===//
// Runtime diagnostics gate
//
// The device runtime bitcode is compiled at -O2 *without* -DNDEBUG (see
// HEXAGON_FLAGS in bin/runtime/CMakeLists.txt), so every `#ifndef NDEBUG` block
// and `assert` in this file is live in production. Two of them sit on the
// per-launch path:
//   * validateInvariants() is an O(allocations^2) overlap scan after every
//     Allocate and Free;
//   * the fmtKB()/fmtPct() arguments of the alloc/free log lines are evaluated
//     eagerly (they run snprintf) even though VTCM_DEBUG defaults to 0.
//
// Gate those diagnostics behind an explicit opt-in: a debug build defines
// HEXMLIR_RUNTIME_DEBUG (e.g. -DHEXMLIR_RUNTIME_DEBUG) and keeps every check
// and log; the release device build does not, so the work is compiled out. This
// changes no allocator behavior -- the real guards (the CHECK()s over an
// unallocated / wrong-size / already-free pointer, and the free-list overlap
// tests in coalesceAndAddToFreeList) run unconditionally.
//===----------------------------------------------------------------------===//
#ifdef HEXMLIR_RUNTIME_DEBUG
#define HEXMLIR_RT_DIAG 1
#else
#define HEXMLIR_RT_DIAG 0
#endif

namespace {
// Constants
constexpr size_t kSmallAlignment = 128;
constexpr size_t kLargeAlignment = 2048;
constexpr size_t kLargeThreshold = 2048;
constexpr size_t kMaxAlignment = 2048;

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
// The context is process-local because the current evidence scope is one
// immutable build/function/site observation. It is deliberately separate from
// the resident-scope ABI: callers must provide both contracts explicitly. The
// event gate records whether a context was present before the first event, so
// late registration cannot create a deceptively joinable mixed report.
struct AccountingContextState {
  std::mutex mutex;
  bool registered = false;
  bool sawEvent = false;
  bool sawEventForEventContext = false;
  VtcmPool::AccountingContext context{};
  bool eventRegistered = false;
  // Set while a thread is inside the enter/leave span and cleared by leave, a
  // rejected registration, or a poisoned owner. It is separate from
  // `eventRegistered` because registration is a historical fact: a report that
  // said "registered" after the owner left would claim a token binding that can
  // no longer attribute anything.
  bool eventOwnerActive = false;
  uint64_t eventThreadOrdinal = 0;
  VtcmPool::AccountingEventContext eventContext{};
};

AccountingContextState gAccountingContextState;
std::mutex gAccountingPoolMutex;
VtcmPool *gAccountingPool = nullptr;
std::atomic<uint64_t> gNextAccountingThreadOrdinal{1};
thread_local uint64_t tAccountingThreadOrdinal = 0;
thread_local bool tAccountingEventContextActive = false;
thread_local bool tAccountingEventContextPoisoned = false;
thread_local VtcmPool::AccountingEventContext tAccountingEventContext{};
constexpr uint8_t kAccountingScopeEnterKind = 8;

struct AccountingContextSnapshot {
  bool registered = false;
  VtcmPool::AccountingContext context{};
  bool eventRegistered = false;
  bool eventOwnerActive = false;
  VtcmPool::AccountingEventContext eventContext{};
};

uint64_t currentAccountingThreadOrdinal() {
  if (tAccountingThreadOrdinal == 0)
    tAccountingThreadOrdinal =
        gNextAccountingThreadOrdinal.fetch_add(1, std::memory_order_relaxed);
  return tAccountingThreadOrdinal;
}

void poisonAccountingEventContextLocked() {
  tAccountingEventContext = {};
  tAccountingEventContextActive = false;
  tAccountingEventContextPoisoned = true;
  // A rejected registration cannot attribute anything, so no owner is active
  // even if an earlier enter on this thread succeeded.
  gAccountingContextState.eventOwnerActive = false;
}

void poisonAccountingEventContext() {
  std::lock_guard<std::mutex> lock(gAccountingContextState.mutex);
  poisonAccountingEventContextLocked();
}

void clearAccountingEventContextTLS() {
  tAccountingEventContext = {};
  tAccountingEventContextActive = false;
  tAccountingEventContextPoisoned = false;
}

AccountingContextSnapshot readAccountingContext() {
  std::lock_guard<std::mutex> lock(gAccountingContextState.mutex);
  AccountingContextSnapshot snapshot;
  snapshot.registered = gAccountingContextState.registered;
  if (snapshot.registered)
    snapshot.context = gAccountingContextState.context;
  snapshot.eventRegistered = gAccountingContextState.eventRegistered;
  snapshot.eventOwnerActive = gAccountingContextState.eventOwnerActive;
  if (snapshot.eventRegistered)
    snapshot.eventContext = gAccountingContextState.eventContext;
  return snapshot;
}

AccountingContextSnapshot beginAccountingEvent(uint8_t kind) {
  std::lock_guard<std::mutex> lock(gAccountingContextState.mutex);
  gAccountingContextState.sawEvent = true;
  if (kind != kAccountingScopeEnterKind)
    gAccountingContextState.sawEventForEventContext = true;
  AccountingContextSnapshot snapshot;
  snapshot.registered = gAccountingContextState.registered;
  if (snapshot.registered)
    snapshot.context = gAccountingContextState.context;
  snapshot.eventRegistered = gAccountingContextState.eventRegistered;
  snapshot.eventOwnerActive = gAccountingContextState.eventOwnerActive;
  if (snapshot.eventRegistered)
    snapshot.eventContext = gAccountingContextState.eventContext;
  return snapshot;
}
#endif

// Environment variable check (once at startup)
static int getVTCMDebugLevel() {
  static int level = -1;
  if (level == -1) {
    const char *env = std::getenv("VTCM_DEBUG");
    level = env ? std::atoi(env) : 0;
  }
  return level;
}

// Debug logging wrapper
template <typename... Args> void vtcmDebugLog(int minLevel, Args &&...args) {
  if (getVTCMDebugLevel() >= minLevel) {
    (std::cout << ... << std::forward<Args>(args)) << std::endl;
  }
}

// Size formatting helpers
const char *fmtKB(size_t bytes) {
  static thread_local char buf[32];
  snprintf(buf, sizeof(buf), "%.1fKB", bytes / 1024.0);
  return buf;
}

const char *fmtMB(size_t bytes) {
  static thread_local char buf[32];
  snprintf(buf, sizeof(buf), "%.2fMB", bytes / (1024.0 * 1024.0));
  return buf;
}

const char *fmtPct(size_t part, size_t total) {
  static thread_local char buf[32];
  if (total == 0) {
    snprintf(buf, sizeof(buf), "0.0%%");
  } else {
    snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * part / total);
  }
  return buf;
}

// Legacy size rounding is deliberately independent of address alignment: keep
// the established 128-byte small / 2048-byte large charge quantum. Alignment
// padding below remains visible as free-list fragments and is never charged as
// payload. Returning zero on overflow makes the caller fail closed.
size_t alignSize(size_t nbytes) {
  const size_t alignment =
      (nbytes >= kLargeThreshold) ? kLargeAlignment : kSmallAlignment;
  if (nbytes > std::numeric_limits<size_t>::max() - (alignment - 1))
    return 0;
  return (nbytes + (alignment - 1)) & ~(alignment - 1);
}

uintptr_t alignAddressUp(uintptr_t address, size_t alignment) {
  const uintptr_t mask = static_cast<uintptr_t>(alignment - 1);
  if (address > std::numeric_limits<uintptr_t>::max() - mask)
    return 0;
  return (address + mask) & ~mask;
}

uintptr_t alignAddressDown(uintptr_t address, size_t alignment) {
  return address & ~(static_cast<uintptr_t>(alignment) - 1);
}

// Return an aligned allocation start inside one free block, or nullptr when the
// block has no aligned span large enough. The charged allocation itself remains
// exactly `nbytes`; any prefix/suffix stays in the free list.
char *alignedStart(const std::pair<char *, size_t> &block, size_t nbytes,
                   size_t alignment) {
  const uintptr_t blockAddress = reinterpret_cast<uintptr_t>(block.first);
  const uintptr_t startAddress = alignAddressUp(blockAddress, alignment);
  if (startAddress == 0 || startAddress < blockAddress)
    return nullptr;
  const size_t prefix = static_cast<size_t>(startAddress - blockAddress);
  if (prefix > block.second || nbytes > block.second - prefix)
    return nullptr;
  return reinterpret_cast<char *>(startAddress);
}

// Find the last free block (by address)
std::vector<std::pair<char *, size_t>>::iterator
findLastFreeBlock(std::vector<std::pair<char *, size_t>> &free) {
  if (free.empty())
    return free.end();
  auto last = free.end();
  last--;
  return last;
}

// Find the smallest free block with an aligned span large enough. For the
// historical 128/2048 requests this chooses the same block as the old
// size-only best fit; stricter alignments merely reject infeasible blocks.
std::vector<std::pair<char *, size_t>>::iterator
findBestFit(std::vector<std::pair<char *, size_t>> &free, size_t nbytes,
            size_t alignment) {
  auto bestFit = free.end();
  for (auto it = free.begin(); it != free.end(); it++) {
    if (alignedStart(*it, nbytes, alignment) == nullptr)
      continue;
    if (bestFit == free.end() || it->second < bestFit->second) {
      bestFit = it;
      if (bestFit->second == nbytes &&
          alignedStart(*bestFit, nbytes, alignment) == bestFit->first) {
        break; // Perfect fit
      }
    }
  }
  return bestFit;
}

// Calculate total allocated size
size_t
calcTotalAllocated(const std::vector<std::pair<char *, size_t>> &allocations) {
  size_t total = 0;
  for (const auto &alloc : allocations) {
    total += alloc.second;
  }
  return total;
}

// Find largest free block size
size_t
calcLargestFreeBlock(const std::vector<std::pair<char *, size_t>> &free) {
  size_t largest = 0;
  for (const auto &block : free) {
    if (block.second > largest) {
      largest = block.second;
    }
  }
  return largest;
}

// Memory map visualization
void printMemoryMap(const std::vector<std::pair<char *, size_t>> &allocations,
                    const std::vector<std::pair<char *, size_t>> &free,
                    void *baseAddr, size_t totalSize) {
  if (getVTCMDebugLevel() < 2)
    return;

  struct Block {
    char *addr;
    size_t size;
    bool isAlloc;
    bool operator<(const Block &other) const { return addr < other.addr; }
  };

  std::vector<Block> blocks;
  for (const auto &a : allocations) {
    blocks.push_back({a.first, a.second, true});
  }
  for (const auto &f : free) {
    blocks.push_back({f.first, f.second, false});
  }

  std::sort(blocks.begin(), blocks.end());

  size_t totalAlloc = 0;
  for (const auto &b : blocks) {
    if (b.isAlloc)
      totalAlloc += b.size;
  }

  std::cout << "[VTCM] Memory Map (used=" << fmtPct(totalAlloc, totalSize)
            << ", " << allocations.size() << " allocs, " << free.size()
            << " free blocks):" << std::endl;

  char *base = static_cast<char *>(baseAddr);
  for (const auto &block : blocks) {
    size_t startOffset = block.addr - base;
    size_t endOffset = startOffset + block.size;

    std::cout << "  [" << (block.isAlloc ? "ALLOC" : "FREE ") << "] " << "0x"
              << std::hex << std::setfill('0') << std::setw(8) << startOffset
              << "-0x" << std::setw(8) << endOffset << std::dec << " ("
              << (block.isAlloc ? fmtKB(block.size) : fmtMB(block.size)) << ")"
              << std::endl;
  }
}
} // namespace

bool VtcmPool::IsSupportedAlignment(size_t alignment) {
  return alignment != 0 && alignment <= kMaxAlignment &&
         (alignment & (alignment - 1)) == 0;
}

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
size_t VtcmPool::SizeAlignedCharge(size_t nbytes) { return alignSize(nbytes); }
#endif

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
static bool accountingContextsEqual(const VtcmPool::AccountingContext &lhs,
                                    const VtcmPool::AccountingContext &rhs) {
  return lhs.version == rhs.version && lhs.flags == rhs.flags &&
         lhs.buildId == rhs.buildId && lhs.functionId == rhs.functionId &&
         lhs.allocationSiteId == rhs.allocationSiteId &&
         lhs.scopeId == rhs.scopeId &&
         lhs.residentKeyDigest == rhs.residentKeyDigest;
}

bool VtcmPool::registerAccountingContext(const AccountingContext &context) {
  if (context.version != kAccountingContextVersion ||
      (context.flags & ~kAccountingContextHasResidentKeyDigest) != 0) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(gAccountingContextState.mutex);
    if (gAccountingContextState.registered) {
      if (!accountingContextsEqual(gAccountingContextState.context, context))
        return false;
    } else {
      if (gAccountingContextState.sawEvent)
        return false;
      gAccountingContextState.context = context;
      gAccountingContextState.registered = true;
    }
  }

  // Registration is resource-free, but if a pool already exists, materialize
  // its one scope event now. The pool-lifetime mutex keeps the pointer valid
  // across this diagnostic-only callback.
  {
    std::lock_guard<std::mutex> lock(gAccountingPoolMutex);
    if (gAccountingPool != nullptr)
      gAccountingPool->recordAccountingScopeEvent();
  }
  return true;
}

bool VtcmPool::hasAccountingContext() {
  return readAccountingContext().registered;
}

bool VtcmPool::registerAccountingEventContext(
    const AccountingEventContext &context) {
  const uint64_t threadOrdinal = currentAccountingThreadOrdinal();
  if (context.version != kAccountingEventContextVersion ||
      (context.flags & ~(kAccountingEventContextSingleInvocation |
                         kAccountingEventContextGridOne)) != 0 ||
      (context.flags & (kAccountingEventContextSingleInvocation |
                        kAccountingEventContextGridOne)) !=
          (kAccountingEventContextSingleInvocation |
           kAccountingEventContextGridOne) ||
      (context.token.low == 0 && context.token.high == 0) ||
      context.accountingScopeId == 0 || context.invocationId == 0 ||
      context.functionId == 0 || context.allocationSiteId == 0) {
    poisonAccountingEventContext();
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(gAccountingContextState.mutex);
    if (gAccountingContextState.sawEventForEventContext ||
        (gAccountingContextState.eventRegistered &&
         gAccountingContextState.eventThreadOrdinal != threadOrdinal)) {
      poisonAccountingEventContextLocked();
      return false;
    }
    if (gAccountingContextState.eventRegistered) {
      if (!(gAccountingContextState.eventContext == context)) {
        poisonAccountingEventContextLocked();
        return false;
      }
    } else {
      gAccountingContextState.eventContext = context;
      gAccountingContextState.eventThreadOrdinal = threadOrdinal;
      gAccountingContextState.eventRegistered = true;
    }
    gAccountingContextState.eventOwnerActive = true;
  }
  tAccountingEventContext = context;
  tAccountingEventContextActive = true;
  tAccountingEventContextPoisoned = false;
  return true;
}

void VtcmPool::clearAccountingEventContext() {
  clearAccountingEventContextTLS();
  // Keep the registration, the declared identity and every counter: leave ends
  // the owner's lifetime, it does not rewrite history. Only the "an owner is
  // active right now" fact is cleared, so the declaration can report a cleared
  // scope instead of a live token binding.
  std::lock_guard<std::mutex> lock(gAccountingContextState.mutex);
  gAccountingContextState.eventOwnerActive = false;
}

bool VtcmPool::hasAccountingEventContext() {
  return readAccountingContext().eventOwnerActive;
}

namespace {
enum AccountingEventKind : uint8_t {
  kAccountingAllocation = 1,
  kAccountingAllocationFailure = 2,
  kAccountingFree = 3,
  kAccountingResidentAllocation = 4,
  kAccountingResidentAllocationFailure = 5,
  kAccountingResidentReuse = 6,
  kAccountingResidentFree = 7,
  kAccountingScopeEnter = 8,
  kAccountingFreeCacheRetain = 9,
  kAccountingFreeCacheHit = 10,
  kAccountingFreeCacheDrop = 11,
  kAccountingFreeCacheEvict = 12,
};

const char *accountingEventKindName(uint8_t kind) {
  switch (kind) {
  case kAccountingAllocation:
    return "alloc";
  case kAccountingAllocationFailure:
    return "alloc_fail";
  case kAccountingFree:
    return "free";
  case kAccountingResidentAllocation:
    return "resident_alloc";
  case kAccountingResidentAllocationFailure:
    return "resident_alloc_fail";
  case kAccountingResidentReuse:
    return "resident_reuse";
  case kAccountingResidentFree:
    return "resident_free";
  case kAccountingScopeEnter:
    return "scope_enter";
  case kAccountingFreeCacheRetain:
    return "free_cache_retain";
  case kAccountingFreeCacheHit:
    return "free_cache_hit";
  case kAccountingFreeCacheDrop:
    return "free_cache_drop";
  case kAccountingFreeCacheEvict:
    return "free_cache_evict";
  default:
    return "unknown";
  }
}

const char *residentKindName(uint8_t kind) {
  switch (kind) {
  case static_cast<uint8_t>(VtcmPool::ResidentKind::kWorkspace):
    return "workspace";
  case static_cast<uint8_t>(VtcmPool::ResidentKind::kWeight):
    return "weight";
  default:
    return "none";
  }
}

const char *residentAbiName(uint8_t abi) {
  switch (abi) {
  case static_cast<uint8_t>(VtcmPool::ResidentAbi::kV2):
    return "v2";
  default:
    return "none";
  }
}

const char *coalesceCaseName(uint8_t value) {
  switch (static_cast<VtcmPool::AccountingCoalesceCase>(value)) {
  case VtcmPool::AccountingCoalesceCase::kNone:
    return "none";
  case VtcmPool::AccountingCoalesceCase::kPrevious:
    return "previous";
  case VtcmPool::AccountingCoalesceCase::kNext:
    return "next";
  case VtcmPool::AccountingCoalesceCase::kBoth:
    return "both";
  default:
    return "invalid";
  }
}

const char *residentKeyDigestStatus(uint8_t residentKind,
                                    const VtcmPool::AccountingContext &context,
                                    bool contextRegistered) {
  if (residentKind == 0)
    return "not-applicable";
  if (contextRegistered &&
      (context.flags & VtcmPool::kAccountingContextHasResidentKeyDigest) != 0)
    return "declared-process-only";
  return "unbound";
}

// Keep formatting out of the allocator's control flow. In particular, no
// pointer-valued formatting argument is accepted here, so the diagnostic report
// cannot accidentally disclose a VTCM address.
int appendAccountingText(char *buf, int cap, int *offset, const char *format,
                         ...) {
  if (buf == nullptr || cap <= 0 || offset == nullptr || *offset >= cap)
    return 0;

  const int remaining = cap - *offset;
  va_list args;
  va_start(args, format);
  const int written =
      vsnprintf(buf + *offset, static_cast<size_t>(remaining), format, args);
  va_end(args);
  if (written < 0)
    return -1;
  if (written >= remaining) {
    *offset = cap;
    return 1;
  }
  *offset += written;
  return 0;
}
} // namespace
#endif

// Allocation strategy:
// - Small allocations (< 2KB): at least 128-byte aligned, allocated from the
//   end of the last free block (by address) to reduce fragmentation in the main
//   pool area. Falls back to best-fit if end allocation fails.
// - Large allocations (>= 2KB): at least 2KB aligned (hardware requirement for
//   certain tensor units), use best-fit to minimize fragmentation.
// - A stricter supported power-of-two request (for example 256 or 512 bytes)
//   raises the effective placement boundary without changing the legacy size
//   rounding quantum.
//
// The previous strategy allocated unaligned blocks from the end, which created
// unusable fragments when freed. By maintaining alignment, we avoid this bug
// while preserving memory efficiency (0% overhead for 128-byte allocations).
//
// Debug logging:
// - VTCM_DEBUG=0 (default): No debug output
// - VTCM_DEBUG=1: Concise operation summaries
// - VTCM_DEBUG=2: Detailed memory maps after each operation
// - Errors always print memory map (even without VTCM_DEBUG)
//
// Example usage:
//   VtcmPool pool;
//   void* small = pool.Allocate(256);  // 128-byte aligned
//   void* large = pool.Allocate(4096); // 2KB aligned
//   pool.Free(small, 256);
//   pool.Free(large, 4096);

VtcmPool::VtcmPool() {
  compute_res_attr_t resInfo;
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_init(&resInfo));

  unsigned int availBlockSize;
  compute_res_vtcm_page_t totalBlockLayout;
  compute_res_vtcm_page_t availBlockLayout;
  unsigned int pageSize = 0;

  HEXAGON_SAFE_CALL(compute_resource_query_VTCM(
      /* application_id = */ 0, &vtcmDeviceSize_, &totalBlockLayout,
      &availBlockSize, &availBlockLayout));

  // Use the max page size from the page_list
  for (int i = 0; i < totalBlockLayout.page_list_len; i++) {
    if (pageSize < totalBlockLayout.page_list[i].page_size) {
      pageSize = totalBlockLayout.page_list[i].page_size;
    }
  }

  CHECK((availBlockSize >= (1024 * 1024)), "Less than 1MB VTCM available");

  // allocate nbytes of vtcm on a single page
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param_v2(
      &resInfo,
      /*vtcm_size = */ vtcmDeviceSize_,
      /*min_page_size = */ pageSize,
      /*min_vtcm_size = */ availBlockSize));

  // TODO(HWE): Investigate why a non-zero timeout results in hanging, both in
  // the simulator and on hardware.
  contextId_ = HAP_compute_res_acquire(&resInfo, /*timeout = */ 0);
  CHECK((contextId_),
        "HAP_compute_res_acquire failed to acquire requested VTCM resource");
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_get_vtcm_ptr_v2(&resInfo, &vtcmData_,
                                                         &vtcmAllocatedSize_));
  CHECK((vtcmData_ != nullptr),
        "HAP_compute_res_acquire returned nullptr when allocating VTCM");
  CHECK((vtcmAllocatedSize_ >= availBlockSize),
        "HAP_compute_res_acquire failed to allocate minimum amount of VTCM");

  vtcmAllocatedPtr_ = vtcmData_;

  vtcmDebugLog(1, "[VTCM] Init: device=", fmtMB(vtcmDeviceSize_),
               " allocated=", fmtMB(vtcmAllocatedSize_));

  HEXAGON_PRINT(HIGH,
                "[hexagon-mlir runtime] VTCM: device=%u bytes, available=%u "
                "bytes, allocated=%u bytes\n",
                vtcmDeviceSize_, availBlockSize, vtcmAllocatedSize_);

#if defined(__hexagon__)
  // Also print to stdout so it can be captured by run wrappers.
  std::printf("[hexagon-mlir runtime] VTCM: device=%u bytes, available=%u "
              "bytes, allocated=%u bytes\n",
              vtcmDeviceSize_, availBlockSize, vtcmAllocatedSize_);
#endif

  free_.emplace_back(std::pair<char *, size_t>(static_cast<char *>(vtcmData_),
                                               vtcmAllocatedSize_));

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  // Publish the pool only after its free list is initialized, then materialize
  // a scope event if context registration happened before construction. No
  // pointer is copied into the event.
  {
    std::lock_guard<std::mutex> lock(gAccountingPoolMutex);
    gAccountingPool = this;
  }
  recordAccountingScopeEvent();
#endif

  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}

VtcmPool::~VtcmPool() {
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  {
    std::lock_guard<std::mutex> lock(gAccountingPoolMutex);
    if (gAccountingPool == this)
      gAccountingPool = nullptr;
  }
#endif
  // Check memory accounting
  assert(checkMemoryAccountedFor());

  // Leak detection
  if (!allocations_.empty()) {
    size_t totalLeaked = calcTotalAllocated(allocations_);

    std::cerr << "[VTCM] WARNING: Leaked " << fmtKB(totalLeaked) << " in "
              << allocations_.size() << " allocations" << std::endl;

    printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
  }

  vtcmDebugLog(1, "[VTCM] Cleanup: released ", fmtMB(vtcmAllocatedSize_));

  HEXAGON_SAFE_CALL(HAP_compute_res_release(contextId_));
}

void *VtcmPool::Allocate(size_t nbytes) {
  return Allocate(nbytes, nbytes >= kLargeThreshold ? kLargeAlignment
                                                    : kSmallAlignment);
}

void *VtcmPool::Allocate(size_t nbytes, size_t alignment) {
  return Allocate(nbytes, alignment, nullptr);
}

void *VtcmPool::Allocate(size_t nbytes, size_t alignment,
                         size_t *chargedBytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (chargedBytes != nullptr)
    *chargedBytes = 0;
  void *ptr = nullptr;
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  ptr = allocateLocked(nbytes, alignment, false, 0);
#else
  ptr = allocateLocked(nbytes, alignment);
#endif
  if (ptr != nullptr && chargedBytes != nullptr)
    *chargedBytes = alignSize(nbytes);
  return ptr;
}

bool VtcmPool::residentDescriptorsMatch(const ResidentDescriptor &lhs,
                                        const ResidentDescriptor &rhs) {
  return lhs.kind == rhs.kind && lhs.key == rhs.key && lhs.bytes == rhs.bytes &&
         lhs.alignment == rhs.alignment && lhs.chargedBytes == rhs.chargedBytes;
}

void *VtcmPool::Resident(ResidentKind kind, uint64_t key, size_t nbytes,
                         size_t alignment, const void *src) {
  std::lock_guard<std::mutex> lock(mutex_);

  const size_t chargedBytes = nbytes == 0 ? 0 : alignSize(nbytes);
  const bool validSource =
      (kind == ResidentKind::kWorkspace && src == nullptr) ||
      (kind == ResidentKind::kWeight && src != nullptr &&
       reinterpret_cast<uintptr_t>(src) == static_cast<uintptr_t>(key));
  if (nbytes == 0 || chargedBytes == 0 || !IsSupportedAlignment(alignment) ||
      !validSource) {
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    recordAccountingAllocationLocked(
        kAccountingResidentAllocationFailure, nbytes, chargedBytes,
        chargedBytes, false, static_cast<uint8_t>(kind), alignment, 0);
#endif
    return nullptr;
  }

  const ResidentDescriptor descriptor{kind, key, nbytes, alignment,
                                      chargedBytes};

  // A key identifies one process-local resident object. Reuse is allowed only
  // for the exact descriptor; in particular, do not fall through and allocate a
  // second block when a same-key request disagrees.
  for (const ResidentBlock &block : resident_) {
    if (block.descriptor.key != key)
      continue;
    if (!residentDescriptorsMatch(block.descriptor, descriptor)) {
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
      recordAccountingAllocationLocked(
          kAccountingResidentAllocationFailure, nbytes, chargedBytes,
          chargedBytes, false, static_cast<uint8_t>(kind), alignment, alignment);
#endif
      FARF(ERROR, "VTCM resident descriptor mismatch for key 0x%llx",
           static_cast<unsigned long long>(key));
      return nullptr;
    }
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    recordAccountingReuseLocked(nbytes, chargedBytes, chargedBytes,
                                static_cast<uint8_t>(kind));
#endif
    return block.ptr;
  }

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  char *ptr = allocateLocked(nbytes, alignment, true,
                               static_cast<uint8_t>(kind));
#else
  char *ptr = allocateLocked(nbytes, alignment);
#endif
  if (ptr == nullptr)
    return nullptr;

  // A weight source address is an immutable, one-object-per-process key: the
  // first call copies that image, and exact reuse never consults or hashes its
  // contents. Workspace storage has no source and is refilled by each launch.
  if (kind == ResidentKind::kWeight)
    std::memcpy(ptr, src, nbytes);
  resident_.push_back(ResidentBlock{descriptor, ptr});
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  // The allocator event is recorded after the resident vector is updated so
  // the event's resident_bytes snapshot describes the completed operation.
  recordAccountingAllocationLocked(
      kAccountingResidentAllocation, nbytes, chargedBytes, chargedBytes, true,
      static_cast<uint8_t>(kind), alignment, alignment);
#endif

  vtcmDebugLog(1, "[VTCM] Resident: ", fmtKB(nbytes), " @ 0x", std::hex,
               ptr - static_cast<char *>(vtcmAllocatedPtr_), std::dec,
               " | resident=", fmtKB(getResidentBytesLocked()));
  return ptr;
}

bool VtcmPool::isResidentLocked(char *ptr) const {
  for (const ResidentBlock &block : resident_)
    if (block.ptr == ptr)
      return true;
  return false;
}

bool VtcmPool::IsResident(void *ptr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return isResidentLocked(static_cast<char *>(ptr));
}

size_t VtcmPool::getResidentBytesLocked() const {
  size_t total = 0;
  for (const ResidentBlock &block : resident_)
    total += block.descriptor.chargedBytes;
  return total;
}

size_t VtcmPool::getResidentBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return getResidentBytesLocked();
}

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
char *VtcmPool::allocateLocked(size_t nbytes, size_t alignment,
                               bool residentAllocation, uint8_t residentKind) {
#else
char *VtcmPool::allocateLocked(size_t nbytes, size_t alignment) {
#endif
  // Edge case: zero-size allocation
  if (nbytes == 0) {
    vtcmDebugLog(1, "[VTCM] WARNING: Zero-size allocation requested");
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    recordAccountingAllocationLocked(
        residentAllocation ? kAccountingResidentAllocationFailure
                           : kAccountingAllocationFailure,
        0, 0, 0, false, residentKind, alignment, 0);
#endif
    return nullptr;
  }

  // Keep the requested value for diagnostics and accounting; the allocator
  // charge is computed separately below.
  const size_t original_nbytes = nbytes;
  if (!IsSupportedAlignment(alignment)) {
    std::cerr << "[VTCM] ERROR: Unsupported alignment " << alignment
              << std::endl;
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    recordAccountingAllocationLocked(
        residentAllocation ? kAccountingResidentAllocationFailure
                           : kAccountingAllocationFailure,
        original_nbytes, 0, 0, false, residentKind, alignment, 0);
#endif
    return nullptr;
  }

  // Preserve the established placement floor while enforcing any stricter
  // requested boundary. Small requests remain at least 128-byte aligned; large
  // requests remain at least 2048-byte aligned.
  const size_t legacyAlignment =
      original_nbytes >= kLargeThreshold ? kLargeAlignment : kSmallAlignment;
  const size_t effectiveAlignment =
      alignment > legacyAlignment ? alignment : legacyAlignment;

  nbytes = alignSize(nbytes);

  // Edge case: allocation larger than total pool or size-rounding overflow.
  if (nbytes == 0 || nbytes > vtcmAllocatedSize_) {
    std::cerr << "[VTCM] ERROR: Requested " << fmtKB(original_nbytes)
              << " (charged " << nbytes << ") exceeds total pool size "
              << vtcmAllocatedSize_ << std::endl;
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    recordAccountingAllocationLocked(
        residentAllocation ? kAccountingResidentAllocationFailure
                           : kAccountingAllocationFailure,
        original_nbytes, nbytes, nbytes, false, residentKind, alignment,
        effectiveAlignment);
#endif
    return nullptr;
  }

#if HEXMLIR_RT_DIAG
  vtcmDebugLog(2, "[VTCM] Alloc request: ", fmtKB(original_nbytes),
               " → aligned: ", fmtKB(nbytes));
#endif

  char *ptr = nullptr;

  // Small allocation: try END of last free block
  if (nbytes < kLargeThreshold) {
    ptr = tryAllocateFromEnd(nbytes, effectiveAlignment);
#if HEXMLIR_RT_DIAG
    if (ptr != nullptr) {
      vtcmDebugLog(2, "[VTCM]   Allocated from end of last block");
    } else {
      vtcmDebugLog(2, "[VTCM]   End allocation failed, using best-fit");
    }
#endif
  }

  // Large allocation OR small allocation fallback: use best-fit
  if (ptr == nullptr) {
    ptr = allocateBestFit(nbytes, effectiveAlignment);
  }

  if (ptr == nullptr) {
    handleAllocationFailure(nbytes);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    recordAccountingAllocationLocked(
        residentAllocation ? kAccountingResidentAllocationFailure
                           : kAccountingAllocationFailure,
        original_nbytes, nbytes, nbytes, false, residentKind, alignment,
        effectiveAlignment);
#endif
    return nullptr;
  }

  // The free-list search is alignment-aware; this unconditional guard keeps a
  // future placement change from silently violating the allocation contract.
  CHECK((reinterpret_cast<uintptr_t>(ptr) &
         static_cast<uintptr_t>(effectiveAlignment - 1)) == 0,
        "VTCM allocator returned a misaligned block");
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  if (allocations_.size() > accountingMaxAllocationCount_)
    accountingMaxAllocationCount_ = allocations_.size();
  if (free_.size() > accountingMaxFreeBlockCount_)
    accountingMaxFreeBlockCount_ = free_.size();
#endif

#if HEXMLIR_RT_DIAG
  logAllocationSuccess(ptr, nbytes);

  validateInvariants();
#endif

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  // Retain the allocation's opaque owner before recording either ordinary or
  // resident events. The map is internal allocator state; no pointer is ever
  // serialized.
  const AccountingOwner owner = currentAccountingOwner();
  rememberAccountingOwnerLocked(ptr, owner);
  // Resident allocation is recorded by Resident after it publishes the block;
  // regular allocation has no such post-state transition.
  if (!residentAllocation)
    recordAccountingAllocationLocked(
        kAccountingAllocation, original_nbytes, nbytes, nbytes, true, 0,
        alignment, effectiveAlignment, &owner);
#endif

  return ptr;
}

// Try to allocate from end of last free block
char *VtcmPool::tryAllocateFromEnd(size_t nbytes, size_t alignment) {
  auto lastBlock = findLastFreeBlock(free_);

  if (lastBlock == free_.end() || lastBlock->second < nbytes)
    return nullptr;

  // Keep the historical end placement, but round the address down to the
  // requested boundary. Any prefix remains free and will coalesce normally.
  const uintptr_t blockAddress = reinterpret_cast<uintptr_t>(lastBlock->first);
  const uintptr_t blockEnd = blockAddress + lastBlock->second;
  const uintptr_t allocationAddress =
      alignAddressDown(blockEnd - nbytes, alignment);
  if (allocationAddress < blockAddress ||
      allocationAddress + nbytes != blockEnd)
    return nullptr;
  char *ptr = reinterpret_cast<char *>(allocationAddress);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  const size_t prefixBytes =
      static_cast<size_t>(allocationAddress - blockAddress);
  if (prefixBytes != 0)
    recordAccountingSplitLocked(prefixBytes, 0);
#endif

  allocations_.emplace_back(std::pair<char *, size_t>(ptr, nbytes));
  lastBlock->second = static_cast<size_t>(allocationAddress - blockAddress);
  if (lastBlock->second == 0)
    free_.erase(lastBlock);

  return ptr;
}

// Allocate using best-fit strategy
char *VtcmPool::allocateBestFit(size_t nbytes, size_t alignment) {
  auto bestFit = findBestFit(free_, nbytes, alignment);

  if (bestFit == free_.end())
    return nullptr;

  // Allocate from the first aligned address in the best-fit block. Preserve a
  // prefix and/or suffix as free-list entries so every later allocation sees
  // the exact remaining address ranges.
  char *ptr = alignedStart(*bestFit, nbytes, alignment);
  CHECK((ptr != nullptr), "best-fit search lost its aligned candidate");
  const size_t prefix = static_cast<size_t>(ptr - bestFit->first);
  const size_t suffix = bestFit->second - prefix - nbytes;
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  if (prefix != 0 || suffix != 0)
    recordAccountingSplitLocked(prefix, suffix);
#endif
  allocations_.emplace_back(std::pair<char *, size_t>(ptr, nbytes));

  if (prefix != 0) {
    bestFit->second = prefix;
    if (suffix != 0)
      free_.emplace(bestFit + 1,
                    std::pair<char *, size_t>(ptr + nbytes, suffix));
  } else {
    bestFit->first = ptr + nbytes;
    bestFit->second = suffix;
    if (suffix == 0)
      free_.erase(bestFit);
  }

  return ptr;
}

// Handle allocation failure
void VtcmPool::handleAllocationFailure(size_t nbytes) {
  size_t totalAllocated = calcTotalAllocated(allocations_);
  size_t largestFree = calcLargestFreeBlock(free_);
  float fragScore = getFragmentationScore();

  std::cerr << "[VTCM] ERROR: Allocation failed!" << std::endl;
  std::cerr << "  Requested: " << fmtKB(nbytes) << std::endl;
  std::cerr << "  Largest free block: " << fmtKB(largestFree) << std::endl;
  std::cerr << "  Total used: " << fmtPct(totalAllocated, vtcmAllocatedSize_)
            << " (" << fmtKB(totalAllocated) << ")" << std::endl;
  std::cerr << "  Free blocks: " << free_.size() << std::endl;
  std::cerr << "  Fragmentation: " << (int)(fragScore * 100) << "%"
            << std::endl;

  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}

// Log successful allocation
void VtcmPool::logAllocationSuccess(char *ptr, size_t nbytes) {
  size_t totalAllocated = calcTotalAllocated(allocations_);
  size_t offset = ptr - static_cast<char *>(vtcmAllocatedPtr_);

  vtcmDebugLog(1, "[VTCM] Alloc: ", fmtKB(nbytes), " @ 0x", std::hex, offset,
               std::dec, " | used=", fmtPct(totalAllocated, vtcmAllocatedSize_),
               " allocs=", allocations_.size(), " free_blocks=", free_.size());

  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}

bool VtcmPool::freeResidentLocked(char *ptr, size_t nbytes) {
  for (const ResidentBlock &block : resident_) {
    if (block.ptr != ptr)
      continue;
    const size_t chargedBytes = nbytes == 0 ? 0 : alignSize(nbytes);
    if (nbytes != block.descriptor.bytes ||
        chargedBytes != block.descriptor.chargedBytes) {
      FARF(ERROR, "VTCM resident free descriptor mismatch");
      return false;
    }
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
    const AccountingOwner owner = lookupAccountingOwnerLocked(ptr);
    recordAccountingFreeLocked(
        kAccountingResidentFree, nbytes, chargedBytes, chargedBytes,
        static_cast<uint8_t>(block.descriptor.kind), /*coalesceCase=*/0,
        &owner);
#endif
    return true;
  }
  return false;
}

bool VtcmPool::FreeResident(void *ptr, size_t nbytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (ptr == nullptr || nbytes == 0)
    return false;
  return freeResidentLocked(static_cast<char *>(ptr), nbytes);
}

void VtcmPool::Free(void *ptr, size_t nbytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Edge case: null pointer
  if (ptr == nullptr) {
    vtcmDebugLog(1, "[VTCM] WARNING: Attempted to free null pointer");
    return;
  }

  // Edge case: zero size
  if (nbytes == 0) {
    vtcmDebugLog(1, "[VTCM] WARNING: Attempted to free zero-size allocation");
    return;
  }

  // Keep the request size for the resident descriptor check and diagnostics.
  const size_t original_nbytes = nbytes;
  nbytes = alignSize(nbytes);

  // A resident buffer is not part of the per-launch lifetime: the kernel's
  // deallocation reaches here every launch, so swallowing it is what keeps the
  // weight pinned. A descriptor mismatch is logged and never falls through to
  // the ordinary free path.
  if (isResidentLocked(static_cast<char *>(ptr))) {
#if HEXMLIR_RT_DIAG
    vtcmDebugLog(1, "[VTCM] Free: keeping resident ", fmtKB(original_nbytes));
#endif
    (void)freeResidentLocked(static_cast<char *>(ptr), original_nbytes);
    return;
  }

#if HEXMLIR_RT_DIAG
  vtcmDebugLog(2, "[VTCM] Free request: ", fmtKB(original_nbytes),
               " → aligned: ", fmtKB(nbytes));
#endif

  // Validate and remove from allocations
  char *ptrToFree = static_cast<char *>(ptr);
  auto it = std::find_if(allocations_.begin(), allocations_.end(),
                         [&](auto entry) { return entry.first == ptrToFree; });

  CHECK((it != allocations_.end()),
        "Attempted to free a pointer that had not been allocated");
  CHECK((it->second == nbytes),
        "Attempted to free a different size than was allocated");

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  const AccountingOwner owner = lookupAccountingOwnerLocked(ptrToFree);
#endif
  allocations_.erase(it);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  accountingOwners_.erase(ptrToFree);
#endif

  // Coalesce and add to free list. The return (number of blocks merged) is only
  // read by the debug log below.
  uint8_t coalesceCase = static_cast<uint8_t>(AccountingCoalesceCase::kNone);
  size_t numCoalesced =
      coalesceAndAddToFreeList(ptrToFree, nbytes, &coalesceCase);
  (void)numCoalesced;

  // Log the free operation
#if HEXMLIR_RT_DIAG
  logFreeSuccess(ptrToFree, nbytes, numCoalesced);

  validateInvariants();
#endif

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  recordAccountingFreeLocked(kAccountingFree, original_nbytes, nbytes,
                             nbytes, 0, coalesceCase, &owner);
#endif
}

// Coalesce adjacent free blocks and add to free list
size_t VtcmPool::coalesceAndAddToFreeList(char *ptr, size_t nbytes,
                                          uint8_t *coalesceCase) {
  bool mergedPrevious = false;
  bool mergedNext = false;

  auto it = std::lower_bound(free_.begin(), free_.end(),
                             std::pair<char *, size_t>(ptr, nbytes),
                             [](auto p, auto q) { return p.first < q.first; });
  if (it == free_.end()) {
    it = free_.emplace(it, std::pair<char *, size_t>(ptr, nbytes));
  } else {
    CHECK((ptr != it->first),
          "Attempting to free a pointer that was already free");
    CHECK((ptr + nbytes <= it->first),
          "free_ is in an inconsistent state, freed block overlaps with next");
    if (ptr + nbytes == it->first) {
      it->first = ptr;
      it->second += nbytes;
      mergedNext = true;
    } else {
      it = free_.emplace(it, std::pair<char *, size_t>(ptr, nbytes));
    }
  }

  if (it != free_.begin()) {
    auto itPrev = it;
    --itPrev;
    CHECK((itPrev->first + itPrev->second <= ptr),
          "free_ is in an inconsistent state, freed block overlaps with "
          "previous");
    if (itPrev->first + itPrev->second == ptr) {
      itPrev->second += it->second;
      free_.erase(it);
      mergedPrevious = true;
    }
  }

  const uint8_t caseValue =
      mergedPrevious
          ? (mergedNext ? static_cast<uint8_t>(AccountingCoalesceCase::kBoth)
                        : static_cast<uint8_t>(AccountingCoalesceCase::kPrevious))
          : (mergedNext ? static_cast<uint8_t>(AccountingCoalesceCase::kNext)
                        : static_cast<uint8_t>(AccountingCoalesceCase::kNone));
  if (coalesceCase != nullptr)
    *coalesceCase = caseValue;

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  switch (static_cast<AccountingCoalesceCase>(caseValue)) {
  case AccountingCoalesceCase::kNone:
    ++accountingCoalesceNoneEvents_;
    break;
  case AccountingCoalesceCase::kPrevious:
    ++accountingCoalescePreviousEvents_;
    ++accountingCoalesceEvents_;
    accountingCoalescedBlocks_ += 1;
    break;
  case AccountingCoalesceCase::kNext:
    ++accountingCoalesceNextEvents_;
    ++accountingCoalesceEvents_;
    accountingCoalescedBlocks_ += 1;
    break;
  case AccountingCoalesceCase::kBoth:
    ++accountingCoalesceBothEvents_;
    ++accountingCoalesceEvents_;
    accountingCoalescedBlocks_ += 2;
    break;
  }
  if (free_.size() > accountingMaxFreeBlockCount_)
    accountingMaxFreeBlockCount_ = free_.size();
#endif

  // Preserve the debug/logging return convention (one inserted block plus the
  // number of explicit neighbor merges), without using list-length deltas for
  // accounting.
  return 1 + static_cast<size_t>(mergedPrevious) +
         static_cast<size_t>(mergedNext);
}

// Log successful free operation
void VtcmPool::logFreeSuccess(char *ptr, size_t nbytes, size_t numCoalesced) {
  size_t offset = ptr - static_cast<char *>(vtcmAllocatedPtr_);

  vtcmDebugLog(1, "[VTCM] Free: ", fmtKB(nbytes), " @ 0x", std::hex, offset,
               std::dec, " | coalesced ", numCoalesced, " blocks");

  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}

// Query methods
size_t VtcmPool::getTotalAllocated() const {
  return calcTotalAllocated(allocations_);
}

size_t VtcmPool::getTotalFree() const {
  size_t total = 0;
  for (const auto &block : free_) {
    total += block.second;
  }
  return total;
}

size_t VtcmPool::getLargestFreeBlock() const {
  return calcLargestFreeBlock(free_);
}

float VtcmPool::getFragmentationScore() const {
  if (free_.empty())
    return 0.0f;

  size_t totalFree = getTotalFree();
  size_t largestFree = getLargestFreeBlock();

  if (totalFree == 0)
    return 0.0f;

  // Fragmentation = 1 - (largest_free / total_free)
  // 0 = one big block (no fragmentation)
  // 1 = many tiny blocks (high fragmentation)
  return 1.0f - (static_cast<float>(largestFree) / totalFree);
}

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
uint32_t VtcmPool::fragmentationPpmLocked(size_t totalFree,
                                          size_t largestFree) {
  if (totalFree == 0)
    return 0;
  if (largestFree >= totalFree)
    return 0;
  constexpr uint64_t kScale = 1000000;
  const uint64_t fragmented = totalFree - largestFree;
  return static_cast<uint32_t>((fragmented * kScale) / totalFree);
}

VtcmPool::AccountingSnapshot VtcmPool::getAccountingSnapshotLocked() const {
  AccountingSnapshot snapshot{};
  snapshot.poolBytes = vtcmAllocatedSize_;
  snapshot.allocatedBytes = calcTotalAllocated(allocations_);
  snapshot.freeBytes = getTotalFree();
  snapshot.largestFreeBytes = getLargestFreeBlock();
  snapshot.allocationCount = allocations_.size();
  snapshot.freeBlockCount = free_.size();
  for (const ResidentBlock &block : resident_)
    snapshot.residentBytes += block.descriptor.chargedBytes;
  snapshot.highWaterAllocatedBytes = accountingHighWaterAllocated_;
  snapshot.fragmentationPpm =
      fragmentationPpmLocked(snapshot.freeBytes, snapshot.largestFreeBytes);
  snapshot.allocationEvents = accountingAllocationEvents_;
  snapshot.allocationFailures = accountingAllocationFailures_;
  snapshot.freeEvents = accountingFreeEvents_;
  snapshot.residentAllocationEvents = accountingResidentAllocationEvents_;
  snapshot.residentReuseEvents = accountingResidentReuseEvents_;
  snapshot.residentFreeEvents = accountingResidentFreeEvents_;
  snapshot.residentV2AllocationEvents = accountingResidentV2AllocationEvents_;
  snapshot.residentV2ReuseEvents = accountingResidentV2ReuseEvents_;
  snapshot.residentV2FreeEvents = accountingResidentV2FreeEvents_;
  snapshot.splitEvents = accountingSplitEvents_;
  snapshot.splitPrefixBytes = accountingSplitPrefixBytes_;
  snapshot.splitSuffixBytes = accountingSplitSuffixBytes_;
  snapshot.coalesceEvents = accountingCoalesceEvents_;
  snapshot.coalescedBlocks = accountingCoalescedBlocks_;
  snapshot.coalesceNoneEvents = accountingCoalesceNoneEvents_;
  snapshot.coalescePreviousEvents = accountingCoalescePreviousEvents_;
  snapshot.coalesceNextEvents = accountingCoalesceNextEvents_;
  snapshot.coalesceBothEvents = accountingCoalesceBothEvents_;
  snapshot.maxAllocationCount = accountingMaxAllocationCount_;
  snapshot.maxFreeBlockCount = accountingMaxFreeBlockCount_;
  snapshot.eventCount = accountingEventCount_;
  snapshot.eventCapacity = kAccountingEventCapacity;
  snapshot.eventsDropped = accountingEventsDropped_;
  snapshot.eventSequence = accountingEventSequence_;
  snapshot.blockAccountingComplete =
      snapshot.allocatedBytes <= snapshot.poolBytes &&
      snapshot.freeBytes <= snapshot.poolBytes - snapshot.allocatedBytes &&
      snapshot.allocatedBytes + snapshot.freeBytes == snapshot.poolBytes;
  if (snapshot.eventCount != 0) {
    snapshot.eventWindowStartSequence =
        accountingEvents_[accountingEventStart_].sequence;
    const size_t last = (accountingEventStart_ + snapshot.eventCount - 1) %
                        kAccountingEventCapacity;
    snapshot.eventWindowEndSequence = accountingEvents_[last].sequence;
  }

  const AccountingContextSnapshot contextSnapshot = readAccountingContext();
  snapshot.contextRegistered = contextSnapshot.registered;
  snapshot.context = contextSnapshot.context;
  if (snapshot.contextRegistered)
    snapshot.contextVersion = snapshot.context.version;
  snapshot.eventContextRegistered = contextSnapshot.eventRegistered;
  snapshot.eventContextOwnerActive = contextSnapshot.eventOwnerActive;
  snapshot.eventContext = contextSnapshot.eventContext;
  snapshot.eventContextBoundEvents = accountingEventContextBoundEvents_;
  snapshot.eventContextAggregateEvents = accountingEventContextAggregateEvents_;

  // Count the event classes in the bounded ring. This is deliberately derived
  // from the recorded event kind, never from a cache key or an address. These
  // are retained-window counts; the cumulative cache counters are serialized by
  // BufferManager separately.
  for (size_t logical = 0; logical < snapshot.eventCount; ++logical) {
    const size_t index =
        (accountingEventStart_ + logical) % kAccountingEventCapacity;
    switch (accountingEvents_[index].kind) {
    case kAccountingScopeEnter:
      ++snapshot.scopeEvents;
      break;
    case kAccountingFreeCacheRetain:
      ++snapshot.freeCacheRetainEvents;
      break;
    case kAccountingFreeCacheHit:
      ++snapshot.freeCacheHitEvents;
      break;
    case kAccountingFreeCacheDrop:
      ++snapshot.freeCacheDropEvents;
      break;
    case kAccountingFreeCacheEvict:
      ++snapshot.freeCacheEvictEvents;
      break;
    default:
      break;
    }
  }
  return snapshot;
}

VtcmPool::AccountingSnapshot VtcmPool::getAccountingSnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return getAccountingSnapshotLocked();
}

VtcmPool::AccountingOwner VtcmPool::currentAccountingOwner() const {
  AccountingOwner owner;
  if (!tAccountingEventContextActive || tAccountingEventContextPoisoned)
    return owner;
  owner.context = tAccountingEventContext;
  owner.threadOrdinal = currentAccountingThreadOrdinal();
  owner.bound = true;
  return owner;
}

void VtcmPool::rememberAccountingOwnerLocked(void *ptr,
                                             const AccountingOwner &owner) {
  if (owner.bound)
    accountingOwners_[ptr] = owner;
}

VtcmPool::AccountingOwner
VtcmPool::lookupAccountingOwnerLocked(void *ptr) const {
  auto found = accountingOwners_.find(ptr);
  return found == accountingOwners_.end() ? AccountingOwner() : found->second;
}

void VtcmPool::recordAccountingSplitLocked(size_t prefixBytes,
                                           size_t suffixBytes) {
  ++accountingSplitEvents_;
  if (prefixBytes != 0) {
    if (accountingSplitPrefixBytes_ >
        std::numeric_limits<uint64_t>::max() - prefixBytes)
      accountingSplitPrefixBytes_ = std::numeric_limits<uint64_t>::max();
    else
      accountingSplitPrefixBytes_ += prefixBytes;
  }
  if (suffixBytes != 0) {
    if (accountingSplitSuffixBytes_ >
        std::numeric_limits<uint64_t>::max() - suffixBytes)
      accountingSplitSuffixBytes_ = std::numeric_limits<uint64_t>::max();
    else
      accountingSplitSuffixBytes_ += suffixBytes;
  }
}

void VtcmPool::recordAccountingEventLocked(
    uint8_t kind, size_t requestedBytes, size_t allocatorInputBytes,
    size_t sizeAlignedBytes, size_t chargedBytes, uint8_t residentKind,
    ResidentAbi residentAbi, size_t requestedAlignment,
    size_t effectiveAlignment, uint8_t coalesceCase,
    uint32_t coalescedBlocks, size_t cachedBytes, size_t cachedBuffers,
    bool emitScopeEvent, const AccountingOwner *owner) {
  // A scope event is the first process-scoped event in a registered stream. It
  // is emitted lazily so registration remains resource-free and can precede
  // lazy HexagonAPI construction; it never carries a site join key.
  if (emitScopeEvent && !accountingScopeEventRecorded_ &&
      hasAccountingContext()) {
    accountingScopeEventRecorded_ = true;
    recordAccountingEventLocked(
        kAccountingScopeEnter, 0, 0, 0,
        0, 0, ResidentAbi::kV2, 0, 0, 0, 0, 0, 0, false);
  }

  const AccountingContextSnapshot contextSnapshot = beginAccountingEvent(kind);
  AccountingEvent event{};
  event.sequence = ++accountingEventSequence_;
  event.kind = kind;
  event.residentKind = residentKind;
  event.residentAbi = static_cast<uint8_t>(residentAbi);
  event.requestedBytes = requestedBytes;
  event.allocatorInputBytes = allocatorInputBytes;
  event.sizeAlignedBytes = sizeAlignedBytes;
  event.chargedBytes = chargedBytes;
  event.requestedAlignment = requestedAlignment;
  event.effectiveAlignment = effectiveAlignment;
  event.coalesceCase = coalesceCase;
  event.coalescedBlocks = coalescedBlocks;
  event.allocatedBytes = calcTotalAllocated(allocations_);
  event.freeBytes = getTotalFree();
  event.largestFreeBytes = getLargestFreeBlock();
  event.allocationCount = allocations_.size();
  event.freeBlockCount = free_.size();
  for (const ResidentBlock &block : resident_)
    event.residentBytes += block.descriptor.chargedBytes;
  event.cachedBytes = cachedBytes;
  event.cachedBuffers = cachedBuffers;
  event.fragmentationPpm =
      fragmentationPpmLocked(event.freeBytes, event.largestFreeBytes);
  event.contextRegistered = contextSnapshot.registered;
  event.context = contextSnapshot.context;
  // All v1 events are process-scoped. Keep the process build/scope identity,
  // but make the absent function/site join explicit in the stored event.
  event.context.functionId = 0;
  event.context.allocationSiteId = 0;
  if (residentKind == 0)
    event.context.residentKeyDigest = {};

  // Only an owner captured on this exact thread and matching the process
  // registration is bound. A delayed cache event deliberately passes no owner;
  // a free uses the owner retained with its allocation, so a different thread
  // or invocation cannot inherit a merely global context.
  const AccountingOwner activeOwner = currentAccountingOwner();
  event.eventContextBound =
      owner != nullptr && owner->bound && activeOwner.bound &&
      contextSnapshot.eventRegistered && owner->context ==
                                            contextSnapshot.eventContext &&
      owner->context == activeOwner.context &&
      owner->threadOrdinal == activeOwner.threadOrdinal;
  if (event.eventContextBound) {
    event.eventContext = owner->context;
    ++accountingEventContextBoundEvents_;
  } else {
    ++accountingEventContextAggregateEvents_;
  }

  size_t index = 0;
  if (accountingEventCount_ < kAccountingEventCapacity) {
    index = (accountingEventStart_ + accountingEventCount_) %
            kAccountingEventCapacity;
    ++accountingEventCount_;
  } else {
    index = accountingEventStart_;
    accountingEventStart_ =
        (accountingEventStart_ + 1) % kAccountingEventCapacity;
    ++accountingEventsDropped_;
  }
  accountingEvents_[index] = event;
}

void VtcmPool::recordAccountingAllocationLocked(
    uint8_t kind, size_t requestedBytes, size_t sizeAlignedBytes,
    size_t chargedBytes, bool success, uint8_t residentKind,
    size_t requestedAlignment, size_t effectiveAlignment,
    const AccountingOwner *owner) {
  const AccountingOwner currentOwner = currentAccountingOwner();
  const AccountingOwner *effectiveOwner = owner ? owner : &currentOwner;
  // Version 1 has no per-event site binding. The registered function/site
  // fields are declared context metadata only; all current allocator/resident
  // events remain process-scoped.
  if (success) {
    ++accountingAllocationEvents_;
    if (kind == kAccountingResidentAllocation) {
      ++accountingResidentAllocationEvents_;
      ++accountingResidentV2AllocationEvents_;
    }
    const size_t allocated = calcTotalAllocated(allocations_);
    if (allocated > accountingHighWaterAllocated_)
      accountingHighWaterAllocated_ = allocated;
  } else {
    ++accountingAllocationFailures_;
  }
  recordAccountingEventLocked(
      kind, requestedBytes,
      requestedBytes, sizeAlignedBytes, chargedBytes, residentKind,
      ResidentAbi::kV2, requestedAlignment, effectiveAlignment, 0, 0, 0, 0,
      /*emitScopeEvent=*/true, effectiveOwner);
}

void VtcmPool::recordAccountingReuseLocked(size_t requestedBytes,
                                           size_t sizeAlignedBytes,
                                           size_t chargedBytes,
                                           uint8_t residentKind,
                                           const AccountingOwner *owner) {
  const AccountingOwner currentOwner = currentAccountingOwner();
  const AccountingOwner *effectiveOwner = owner ? owner : &currentOwner;
  ++accountingResidentReuseEvents_;
  ++accountingResidentV2ReuseEvents_;
  recordAccountingEventLocked(
      kAccountingResidentReuse,
      requestedBytes, requestedBytes, sizeAlignedBytes, chargedBytes,
      residentKind, ResidentAbi::kV2, 0, 0, 0, 0, 0, 0,
      /*emitScopeEvent=*/true, effectiveOwner);
}

void VtcmPool::recordAccountingFreeLocked(
    uint8_t kind, size_t requestedBytes, size_t sizeAlignedBytes,
    size_t chargedBytes, uint8_t residentKind, uint8_t coalesceCase,
    const AccountingOwner *owner) {
  const uint32_t coalescedBlocks =
      coalesceCase == static_cast<uint8_t>(AccountingCoalesceCase::kBoth)
          ? 2u
          : (coalesceCase == static_cast<uint8_t>(AccountingCoalesceCase::kNone)
                 ? 0u
                 : 1u);
  if (kind == kAccountingResidentFree) {
    ++accountingResidentFreeEvents_;
    ++accountingResidentV2FreeEvents_;
  } else {
    ++accountingFreeEvents_;
  }
  // Deallocation is process-scoped in v1 as well: the runtime cannot prove
  // which site owned a delayed/cache-mediated release.
  recordAccountingEventLocked(
      kind, requestedBytes,
      requestedBytes, sizeAlignedBytes, chargedBytes, residentKind,
      ResidentAbi::kV2, 0, 0, coalesceCase, coalescedBlocks, 0, 0,
      /*emitScopeEvent=*/true, owner);
}

void VtcmPool::recordAccountingScopeEvent() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (accountingScopeEventRecorded_ || !hasAccountingContext())
    return;
  accountingScopeEventRecorded_ = true;
  recordAccountingEventLocked(
      kAccountingScopeEnter, 0, 0, 0, 0,
      0, ResidentAbi::kV2, 0, 0, 0, 0, 0, 0, false);
}

void VtcmPool::recordAccountingCacheEvent(
    AccountingCacheEventKind kind, size_t requestedBytes,
    size_t allocatorInputBytes, size_t sizeAlignedBytes, size_t chargedBytes,
    size_t requestedAlignment, size_t cachedBytes, size_t cachedBuffers) {
  uint8_t eventKind = 0;
  switch (kind) {
  case AccountingCacheEventKind::kRetain:
    eventKind = kAccountingFreeCacheRetain;
    break;
  case AccountingCacheEventKind::kHit:
    eventKind = kAccountingFreeCacheHit;
    break;
  case AccountingCacheEventKind::kDrop:
    eventKind = kAccountingFreeCacheDrop;
    break;
  case AccountingCacheEventKind::kEvict:
    eventKind = kAccountingFreeCacheEvict;
    break;
  default:
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  recordAccountingEventLocked(
      eventKind, requestedBytes,
      allocatorInputBytes, sizeAlignedBytes, chargedBytes, 0,
      ResidentAbi::kV2,
      requestedAlignment, 0, 0, 0, cachedBytes, cachedBuffers);
}

int VtcmPool::writeAccountingReportLocked(char *buf, int cap) const {
  if (buf == nullptr || cap <= 0)
    return 0;

  const AccountingSnapshot snapshot = getAccountingSnapshotLocked();
  int offset = 0;
  const char *eventStatus =
      snapshot.eventsDropped == 0 ? "complete" : "incomplete";
  const char *identityMode =
      snapshot.contextRegistered ? "context" : "process-aggregate";
  const char *eventScopePolicy = "process-only-v1";
  const char *blockAccountingStatus =
      snapshot.blockAccountingComplete ? "complete" : "incomplete";
  if (appendAccountingText(buf, cap, &offset,
                           "VTCM_ACCOUNTING schema=1 identity_schema=1 "
                           "event_scope_schema=1 identity_mode=%s "
                           "event_scope_policy=%s "
                           "scope=process-high-water "
                           "unit=charged-bytes "
                           "allocator_aligned_unit_status=not-proven "
                           "requested_bytes_basis=allocator-input "
                           "aligned_bytes_basis=size-rounding "
                           "address_alignment_status=not-proven "
                           "source=runtime-vtcm-pool "
                           "event_log_status=%s\n",
                           identityMode, eventScopePolicy, eventStatus) != 0)
    return kAccountingReportTruncated;

  // This is deliberately a separate contract record. The legacy
  // allocator-aligned/charged names above describe the size-rounding charge;
  // they do not constitute a complete header/split/grid model. Keep every
  // unresolved axis machine-readable instead of letting one aggregate status
  // imply that a process observation belongs to a function or site.
  if (appendAccountingText(
          buf, cap, &offset,
          "VTCM_EVIDENCE schema=1 layer=layer-b-runtime "
          "evidence_status=incomplete scope=process-high-water "
          "function_scope_status=not-bound-in-v1 "
          "site_scope_status=not-bound-in-v1 "
          "grid_scope_status=not-bound-in-v1 "
          "grid_instance_status=not-proven "
          "requested_bytes_basis=raw-buffer-request "
          "allocator_input_bytes_basis=vtcmpool-entry "
          "cache_scope=vtcm-only ddr_cache_accounting=excluded "
          "allocator_aligned_bytes_status=not-proven "
          "allocator_aligned_bytes_reason=size-rounding-only-no-header-or-split-model "
          "charged_bytes_basis=runtime-pool-block-length "
          "resident_bytes_basis=resident-block-charge "
          "observed_bytes_basis=pool-snapshot "
          "pool_high_water_scope=charged-vtcm-pool-only "
          "block_accounting_status=%s "
          "block_accounting_scope=runtime-vtcm-pool "
          "header_accounting_status=not-proven "
          "header_accounting_reason=metadata-overhead-not-in-vtcm-pool "
          "split_coalesce_status=observed-only "
          "split_coalesce_model_status=not-proven "
          "exact_peak_model_status=not-proven "
          "cold_warm_status=not-proven-runtime "
          "reuse_status=observed-event-kinds "
          "content_identity_status=not-proven "
          "event_join_status=not-proven "
          "performance_claimed=false\n",
          blockAccountingStatus) != 0)
    return kAccountingReportTruncated;

  // This line is a versioned diagnostic contract, not a second accounting
  // stream and not a manifest field. It records the smallest scope we could
  // honestly describe and refuses the joins that the v1 event ABI cannot carry.
  if (appendAccountingText(
          buf, cap, &offset,
          "VTCM_EVIDENCE_CONTEXT schema=1 context_abi=accounting-v1 "
          "mode=diagnostic-only context_scope=process-high-water "
          "observation_scope=one-immutable-principal-module-function-canonical-site-grid1-single-invocation "
          "observation_scope_status=not-proven "
          "frame_status=not-proven event_owner_status=aggregate "
          "delayed_cache_owner_status=aggregate "
          "context_registered=%u build_id_status=%s "
          "function_id_status=not-bound-in-v1 "
          "allocation_site_id_status=not-bound-in-v1 "
          "resident_scope_binding=not-proven "
          "object_descriptor_status=not-proven "
          "content_descriptor_status=not-proven "
          "workspace_overwrite_proof_status=not-proven "
          "global_symbol_identity_status=not-proven "
          "packed_weight_source_view_status=not-proven "
          "packed_weight_digest_status=not-proven "
          "content_identity_status=not-proven "
          "content_mutation_status=not-proven address_reuse_status=not-proven "
          "module_scope_status=not-proven process_scope_status=not-proven "
          "grid_scope_status=not-bound-in-v1 event_join_status=not-proven "
          "allocator_units=raw-requested,allocator-input,size-aligned,charged,resident,observed "
          "raw_requested_bytes_basis=raw-buffer-request "
          "allocator_input_bytes_basis=vtcmpool-entry "
          "size_aligned_bytes_basis=size-rounding "
          "charged_bytes_basis=runtime-pool-block-length "
          "resident_bytes_basis=resident-block-charge "
          "observed_bytes_basis=pool-snapshot "
          "pool_high_water_scope=charged-vtcm-pool-only "
          "pool_cache_combined_status=not-proven "
          "full_kernel_occupancy_status=not-proven "
          "performance_claimed=false\n",
          snapshot.contextRegistered ? 1u : 0u,
          snapshot.contextRegistered ? "declared-process-only" : "unbound") != 0)
    return kAccountingReportTruncated;

  if (snapshot.eventContextRegistered) {
    // The identity below is historical: it is what was registered, not a claim
    // that an owner still exists. After leave (or a rejected registration) the
    // declaration must not read "registered" with a live token binding, so the
    // status names the cleared state and owner_active is 0. The bound/aggregate
    // counters are cumulative and never rewind.
    const char *contextStatus = snapshot.eventContextOwnerActive
                                    ? "registered-active"
                                    : "registered-cleared";
    const char *ownerStatus = snapshot.eventContextOwnerActive
                                  ? "token-bound"
                                  : "cleared-historical";
    if (appendAccountingText(
            buf, cap, &offset,
            "VTCM_EVENT_CONTEXT schema=1 context_abi=accounting-event-v1 "
            "mode=diagnostic-only context_status=%s owner_active=%u "
            "token_bits=128 "
            "token_low=%llu token_high=%llu accounting_scope_id=%llu invocation_id=%llu "
            "function_id=%llu allocation_site_id=%llu grid_product=1 "
            "event_owner_status=%s delayed_cache_owner_status=aggregate "
            "grid_scope_status=not-proven event_bound_events=%llu "
            "event_aggregate_events=%llu performance_claimed=false\n",
            contextStatus, snapshot.eventContextOwnerActive ? 1u : 0u,
            static_cast<unsigned long long>(snapshot.eventContext.token.low),
            static_cast<unsigned long long>(snapshot.eventContext.token.high),
            static_cast<unsigned long long>(snapshot.eventContext.accountingScopeId),
            static_cast<unsigned long long>(snapshot.eventContext.invocationId),
            static_cast<unsigned long long>(snapshot.eventContext.functionId),
            static_cast<unsigned long long>(
                snapshot.eventContext.allocationSiteId),
            ownerStatus,
            static_cast<unsigned long long>(
                snapshot.eventContextBoundEvents),
            static_cast<unsigned long long>(
                snapshot.eventContextAggregateEvents)) != 0)
      return kAccountingReportTruncated;
  }

  // The identity scope below is the 64-bit process observation ID only. It
  // is deliberately not compared with the 128-bit resident-v2 scope pair;
  // resident_scope_binding=not-proven.
  if (snapshot.contextRegistered) {
    const AccountingContext context = snapshot.context;
    if (appendAccountingText(
            buf, cap, &offset,
            "VTCM_IDENTITY schema=1 version=%u registered=1 "
            "event_scope_schema=1 event_scope_policy=process-only-v1 "
            "site_event_status=not-bound-in-v1 "
            "function_scope_status=not-bound-in-v1 "
            "grid_scope_status=not-bound-in-v1 "
            "full_kernel_join=not-supported "
            "context_identity_scope=declared-only "
            "build_id_bits=128 build_id_low=%llu build_id_high=%llu "
            "function_id_status=declared-only function_id=%llu "
            "allocation_site_id_status=declared-only allocation_site_id=%llu "
            "scope_id=%llu "
            "resident_key_digest_status=%s\n",
            context.version,
            static_cast<unsigned long long>(context.buildId.low),
            static_cast<unsigned long long>(context.buildId.high),
            static_cast<unsigned long long>(context.functionId),
            static_cast<unsigned long long>(context.allocationSiteId),
            static_cast<unsigned long long>(context.scopeId),
            (context.flags & kAccountingContextHasResidentKeyDigest) != 0
                ? "declared-process-only"
                : "unbound") != 0)
      return kAccountingReportTruncated;
  } else if (appendAccountingText(
                 buf, cap, &offset,
                 "VTCM_IDENTITY schema=1 registered=0 "
                 "event_scope_schema=1 identity_mode=process-aggregate "
                 "event_scope_policy=process-only-v1 "
                 "function_scope_status=not-bound-in-v1 "
                 "site_scope_status=not-bound-in-v1 "
                 "grid_scope_status=not-bound-in-v1\n") != 0) {
    return kAccountingReportTruncated;
  }

  if (appendAccountingText(
          buf, cap, &offset,
          "VTCM_SNAPSHOT pool_bytes=%llu charged_allocated_bytes=%llu "
          "allocated_bytes=%llu observed_allocated_bytes=%llu "
          "free_bytes=%llu largest_free_bytes=%llu fragmentation_ppm=%u "
          "allocation_count=%llu free_block_count=%llu resident_bytes=%llu "
          "high_water_allocated_bytes=%llu high_water_scope=charged-vtcm-pool-only "
          "allocation_events=%llu "
          "allocation_failures=%llu free_events=%llu "
          "resident_allocation_events=%llu resident_reuse_events=%llu "
          "resident_free_events=%llu "
          "resident_v2_allocation_events=%llu resident_v2_reuse_events=%llu "
          "resident_v2_free_events=%llu split_events=%llu "
          "split_prefix_bytes=%llu split_suffix_bytes=%llu "
          "coalesce_events=%llu coalesced_blocks=%llu "
          "coalesce_none_events=%llu coalesce_previous_events=%llu "
          "coalesce_next_events=%llu coalesce_both_events=%llu "
          "max_allocation_count=%llu max_free_block_count=%llu "
          "block_accounting_complete=%u event_count=%llu event_capacity=%llu "
          "events_dropped=%llu event_sequence=%llu "
          "event_window_start_sequence=%llu event_window_end_sequence=%llu "
          "context_registered=%u context_version=%u scope_events=%llu "
          "free_cache_retain_events=%llu free_cache_hit_events=%llu "
          "free_cache_drop_events=%llu free_cache_evict_events=%llu "
          "event_counter_scope=retained-ring\n",
          static_cast<unsigned long long>(snapshot.poolBytes),
          static_cast<unsigned long long>(snapshot.allocatedBytes),
          static_cast<unsigned long long>(snapshot.allocatedBytes),
          static_cast<unsigned long long>(snapshot.allocatedBytes),
          static_cast<unsigned long long>(snapshot.freeBytes),
          static_cast<unsigned long long>(snapshot.largestFreeBytes),
          snapshot.fragmentationPpm,
          static_cast<unsigned long long>(snapshot.allocationCount),
          static_cast<unsigned long long>(snapshot.freeBlockCount),
          static_cast<unsigned long long>(snapshot.residentBytes),
          static_cast<unsigned long long>(snapshot.highWaterAllocatedBytes),
          static_cast<unsigned long long>(snapshot.allocationEvents),
          static_cast<unsigned long long>(snapshot.allocationFailures),
          static_cast<unsigned long long>(snapshot.freeEvents),
          static_cast<unsigned long long>(snapshot.residentAllocationEvents),
          static_cast<unsigned long long>(snapshot.residentReuseEvents),
          static_cast<unsigned long long>(snapshot.residentFreeEvents),
          static_cast<unsigned long long>(snapshot.residentV2AllocationEvents),
          static_cast<unsigned long long>(snapshot.residentV2ReuseEvents),
          static_cast<unsigned long long>(snapshot.residentV2FreeEvents),
          static_cast<unsigned long long>(snapshot.splitEvents),
          static_cast<unsigned long long>(snapshot.splitPrefixBytes),
          static_cast<unsigned long long>(snapshot.splitSuffixBytes),
          static_cast<unsigned long long>(snapshot.coalesceEvents),
          static_cast<unsigned long long>(snapshot.coalescedBlocks),
          static_cast<unsigned long long>(snapshot.coalesceNoneEvents),
          static_cast<unsigned long long>(snapshot.coalescePreviousEvents),
          static_cast<unsigned long long>(snapshot.coalesceNextEvents),
          static_cast<unsigned long long>(snapshot.coalesceBothEvents),
          static_cast<unsigned long long>(snapshot.maxAllocationCount),
          static_cast<unsigned long long>(snapshot.maxFreeBlockCount),
          snapshot.blockAccountingComplete ? 1u : 0u,
          static_cast<unsigned long long>(snapshot.eventCount),
          static_cast<unsigned long long>(snapshot.eventCapacity),
          static_cast<unsigned long long>(snapshot.eventsDropped),
          static_cast<unsigned long long>(snapshot.eventSequence),
          static_cast<unsigned long long>(snapshot.eventWindowStartSequence),
          static_cast<unsigned long long>(snapshot.eventWindowEndSequence),
          snapshot.contextRegistered ? 1u : 0u, snapshot.contextVersion,
          static_cast<unsigned long long>(snapshot.scopeEvents),
          static_cast<unsigned long long>(snapshot.freeCacheRetainEvents),
          static_cast<unsigned long long>(snapshot.freeCacheHitEvents),
          static_cast<unsigned long long>(snapshot.freeCacheDropEvents),
          static_cast<unsigned long long>(snapshot.freeCacheEvictEvents)) != 0)
    return kAccountingReportTruncated;

  for (size_t logical = 0; logical < snapshot.eventCount; ++logical) {
    const size_t index =
        (accountingEventStart_ + logical) % kAccountingEventCapacity;
    const AccountingEvent &event = accountingEvents_[index];
    const char *residentAbi =
        event.residentKind == 0 ? "none" : residentAbiName(event.residentAbi);
    const char *keyStatus =
        residentKeyDigestStatus(event.residentKind, event.context,
                                event.contextRegistered);
    int status = appendAccountingText(
        buf, cap, &offset,
        "VTCM_EVENT sequence=%llu event_sequence=%llu kind=%s "
        "requested_bytes=%llu allocator_input_bytes=%llu "
        "size_aligned_bytes=%llu charged_bytes=%llu "
        "allocator_aligned_status=not-proven "
        "requested_alignment=%llu effective_alignment=%llu "
        "allocated_bytes_after=%llu free_bytes_after=%llu "
        "largest_free_bytes_after=%llu allocation_count_after=%llu "
        "free_block_count_after=%llu resident_bytes_after=%llu "
        "cached_bytes_after=%llu cached_buffers_after=%llu "
        "cached_bytes_scope=vtcm-only "
        "coalesce_case=%s coalesced_blocks=%u fragmentation_ppm_after=%u "
        "context_registered=%u event_identity_scope=process "
        "identity_status=process function_id_status=not-bound "
        "allocation_site_id_status=not-bound site_scope_status=not-bound-in-v1 "
        "grid_scope_status=not-bound-in-v1 resident_kind=%s "
        "resident_abi=%s resident_key_digest_status=%s",
        static_cast<unsigned long long>(event.sequence),
        static_cast<unsigned long long>(event.sequence),
        accountingEventKindName(event.kind),
        static_cast<unsigned long long>(event.requestedBytes),
        static_cast<unsigned long long>(event.allocatorInputBytes),
        static_cast<unsigned long long>(event.sizeAlignedBytes),
        static_cast<unsigned long long>(event.chargedBytes),
        static_cast<unsigned long long>(event.requestedAlignment),
        static_cast<unsigned long long>(event.effectiveAlignment),
        static_cast<unsigned long long>(event.allocatedBytes),
        static_cast<unsigned long long>(event.freeBytes),
        static_cast<unsigned long long>(event.largestFreeBytes),
        static_cast<unsigned long long>(event.allocationCount),
        static_cast<unsigned long long>(event.freeBlockCount),
        static_cast<unsigned long long>(event.residentBytes),
        static_cast<unsigned long long>(event.cachedBytes),
        static_cast<unsigned long long>(event.cachedBuffers),
        coalesceCaseName(event.coalesceCase), event.coalescedBlocks,
        event.fragmentationPpm, event.contextRegistered ? 1u : 0u,
        residentKindName(event.residentKind), residentAbi, keyStatus);
    if (status != 0)
      return kAccountingReportTruncated;

    if (event.contextRegistered) {
      status = appendAccountingText(
          buf, cap, &offset,
          " context_identity_scope=declared-only build_id_low=%llu "
          "build_id_high=%llu scope_id=%llu",
          static_cast<unsigned long long>(event.context.buildId.low),
          static_cast<unsigned long long>(event.context.buildId.high),
          static_cast<unsigned long long>(event.context.scopeId));
      if (status != 0)
        return kAccountingReportTruncated;
      if (event.residentKind != 0 &&
          (event.context.flags & kAccountingContextHasResidentKeyDigest) != 0) {
        status = appendAccountingText(
            buf, cap, &offset,
            " resident_key_digest_bits=128 resident_key_digest_low=%llu "
            "resident_key_digest_high=%llu",
            static_cast<unsigned long long>(
                event.context.residentKeyDigest.low),
            static_cast<unsigned long long>(
                event.context.residentKeyDigest.high));
        if (status != 0)
          return kAccountingReportTruncated;
      }
    }
    if (event.eventContextBound) {
      status = appendAccountingText(
          buf, cap, &offset,
          " event_context_status=bound event_token_bits=128 "
          "event_token_low=%llu event_token_high=%llu event_accounting_scope_id=%llu "
          "event_invocation_id=%llu event_function_id=%llu "
          "event_allocation_site_id=%llu",
          static_cast<unsigned long long>(event.eventContext.token.low),
          static_cast<unsigned long long>(event.eventContext.token.high),
          static_cast<unsigned long long>(event.eventContext.accountingScopeId),
          static_cast<unsigned long long>(event.eventContext.invocationId),
          static_cast<unsigned long long>(event.eventContext.functionId),
          static_cast<unsigned long long>(
              event.eventContext.allocationSiteId));
      if (status != 0)
        return kAccountingReportTruncated;
    } else {
      status = appendAccountingText(
          buf, cap, &offset, " event_context_status=%s",
          snapshot.eventContextRegistered ? "aggregate" : "not-registered");
      if (status != 0)
        return kAccountingReportTruncated;
    }
    if (appendAccountingText(buf, cap, &offset, "\n") != 0)
      return kAccountingReportTruncated;
  }
  return offset;
}

int VtcmPool::writeAccountingReport(char *buf, int cap) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return writeAccountingReportLocked(buf, cap);
}
#endif

void VtcmPool::printState() const {
  std::cout << "VTCM Pool State:" << std::endl;
  std::cout << "  Total: " << fmtMB(vtcmAllocatedSize_) << std::endl;
  std::cout << "  Allocated: " << fmtKB(getTotalAllocated()) << " ("
            << fmtPct(getTotalAllocated(), vtcmAllocatedSize_) << ")"
            << std::endl;
  std::cout << "  Free: " << fmtKB(getTotalFree()) << std::endl;
  std::cout << "  Largest free: " << fmtKB(getLargestFreeBlock()) << std::endl;
  std::cout << "  Allocations: " << allocations_.size() << std::endl;
  std::cout << "  Free blocks: " << free_.size() << std::endl;
  std::cout << "  Fragmentation: " << (int)(getFragmentationScore() * 100)
            << "%" << std::endl;
}

// Validation methods (debug builds only)
void VtcmPool::validateInvariants() const {
#if HEXMLIR_RT_DIAG
  // Check that allocations don't overlap
  for (auto it1 = allocations_.begin(); it1 != allocations_.end(); ++it1) {
    for (auto it2 = std::next(it1); it2 != allocations_.end(); ++it2) {
      char *end1 = it1->first + it1->second;
      char *end2 = it2->first + it2->second;
      assert((end1 <= it2->first || end2 <= it1->first) &&
             "Allocations overlap!");
    }
  }

  // Check that free blocks don't overlap
  for (auto it1 = free_.begin(); it1 != free_.end(); ++it1) {
    for (auto it2 = std::next(it1); it2 != free_.end(); ++it2) {
      char *end1 = it1->first + it1->second;
      char *end2 = it2->first + it2->second;
      assert((end1 <= it2->first || end2 <= it1->first) &&
             "Free blocks overlap!");
    }
  }

  // Check that allocations and free blocks don't overlap
  for (const auto &alloc : allocations_) {
    for (const auto &freeBlock : free_) {
      char *allocEnd = alloc.first + alloc.second;
      char *freeEnd = freeBlock.first + freeBlock.second;
      assert((allocEnd <= freeBlock.first || freeEnd <= alloc.first) &&
             "Allocation and free block overlap!");
    }
  }
#endif
}

bool VtcmPool::checkMemoryAccountedFor() const {
  size_t totalAllocated = getTotalAllocated();
  size_t totalFree = getTotalFree();
  size_t accounted = totalAllocated + totalFree;

  if (accounted != vtcmAllocatedSize_) {
    std::cerr << "[VTCM] ERROR: Memory accounting mismatch!" << std::endl;
    std::cerr << "  Allocated: " << totalAllocated << std::endl;
    std::cerr << "  Free: " << totalFree << std::endl;
    std::cerr << "  Total: " << accounted << std::endl;
    std::cerr << "  Expected: " << vtcmAllocatedSize_ << std::endl;
    return false;
  }
  return true;
}

void VtcmPool::DebugDump() {
  printState();
  printMemoryMap(allocations_, free_, vtcmAllocatedPtr_, vtcmAllocatedSize_);
}
