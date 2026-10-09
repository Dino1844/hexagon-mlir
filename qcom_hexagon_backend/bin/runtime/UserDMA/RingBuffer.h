//===- RingBuffer.h - Ring buffer implementation for DMA transfers -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file describes the Ring buffer interface for multiple DMA transfers.
// The details of a DMA transfer is maintained in a DMA descriptor. The ring
// buffer holds 'capacity' elements of type 'T' (DMA descriptors, in the use
// case) and the buffer is initialized upon construction. The ring buffer's
// constructor takes as argument a function that returns true if a 'T' object is
// still 'live', i.e. the DMA transfer is in progress). The ring buffer's
// 'alloc' implementation (that reserves an entry in the buffer) blocks till a
// free entry becomes available, if the ring buffer is full. An entry can be
// freed if it ceases to be 'in-flight' i.e. DMA transfer gets completed.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBACKEND_BIN_RUNTIME_USERDMA_RING_BUFFER_H
#define HEXAGONBACKEND_BIN_RUNTIME_USERDMA_RING_BUFFER_H

#include <cassert>
#include <functional>
#include <mutex>

namespace hexagon {
namespace userdma {

static const uint32_t LARGEST_TOKEN = static_cast<uint32_t>(-1);

using InFlightFn = bool (*)(void *);

template <class T> class RingBuffer {
private:
  T *ringBuffPtr = nullptr; // Pointer to the ring buffer

  const uint32_t capacity = 0; // Size of the ring buffer in number of Ts

  std::function<bool(T *)>
      inFlight; // Function that determines whether a T is in flight

  uint32_t tail = 0; // next allocation index
  uint32_t head = 0; // next free index

  // Guards head/tail. The only mutation point is alloc(), which therefore
  // takes this lock for its whole body (reservation of `tail` + the free-scan
  // that advances `head`); size()/isFull() read head/tail and are only ever
  // called from inside that critical section (no external callers today).
  // Same primitive as the rest of the runtime (VTCMPool/BufferManager/
  // HexagonThreadPool): plain std::mutex + std::lock_guard; no new dependency.
  //
  // Lock granularity & deadlock argument: the lock covers *all* of alloc(),
  // including the busy-wait over inFlight() when the ring is full — head must
  // not be scanned by two threads at once, and a partially-updated head/tail
  // is exactly what lets two callers reserve the same token or spin forever in
  // isFull(). While held, alloc() calls only the inFlight callback, and that
  // callback polls DMA registers (dmpoll()) — it takes no lock and never
  // re-enters the ring. So this mutex is a leaf: a holder waits on hardware,
  // never on another software lock, and callers may hold their own outer lock
  // (UserDMA::enqueueMutex_) on entry — the order outer -> ring is the only
  // direction that exists, hence no cycle and no deadlock.
  std::mutex ringMutex_;

public:
  /*! Creates a ring buffer for storage items of type T
   *  ringBuffCapacity: Size of the ring buffer in number of Ts
   *  inFlightParam: Function that determines whether a T is in flight
   */
  RingBuffer(uint32_t ringBuffCapacity, InFlightFn inFlightParam)
      : capacity(ringBuffCapacity), inFlight(inFlightParam) {
    assert(ringBuffCapacity != 0);
    // We want capacity to be a power of 2 for efficient access
    assert(((ringBuffCapacity & (ringBuffCapacity - 1)) == 0) &&
           "ring buffer capacity is not a power of 2");
    size_t numBytes = sizeof(T) * ringBuffCapacity;
    int ret = posix_memalign(reinterpret_cast<void **>(&ringBuffPtr), sizeof(T),
                             numBytes);
    assert(ret == 0 && "unable to allocate ring buffer storage");
    assert(ringBuffPtr != nullptr);
    // Initialize descriptors to 0's
    memset(ringBuffPtr, 0, numBytes);
  }

  uint32_t size() {
    if (head <= tail)
      return (tail - head);
    // tail has wrapped so it is actually tail + 2^32
    // 2^32 - 1 + 1 + tail - head
    // i.e. 2^32 - 1 - head + tail + 1
    return (LARGEST_TOKEN - head + tail + 1);
  }

  //! returns true iff # of elements in queue == capacity
  bool isFull() { return size() == capacity; }

  //! allocates an entry in ring buffer and returns the entry
  //! token identifier for the allocated entry
  //! Thread-safe: the whole body (isFull/free-scan/`tail` reservation) runs
  //! under ringMutex_; see that member for granularity and deadlock reasoning.
  T *alloc(uint32_t &token) {
    std::lock_guard<std::mutex> lock(ringMutex_);
    while (isFull()) { // Loop as long as the queue is full
      uint32_t numFreed = 0;
      uint32_t sampledSize = size();
      while ((!inFlight(getDataPtr(head))) && (numFreed < sampledSize)) {
        ++head; // keep freeing entries that are done
        ++numFreed;
      }
      if (numFreed == sampledSize)
        assert(head == tail);
    }
    token = tail;
    ++tail; // update next allocation index
    return getDataPtr(token);
  }

  //! Returns the address of a T given its index
  T *getDataPtr(uint32_t token) const {
    // Since capacity is a power of 2, we can use '&' in place of '%'
    uint32_t ringBuffIndex = token & (capacity - 1);
    return ringBuffPtr + ringBuffIndex;
  }

  ~RingBuffer() { free(ringBuffPtr); }
};
}; // namespace userdma
}; // namespace hexagon

#endif
