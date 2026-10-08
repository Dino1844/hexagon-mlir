//===- HmxSpscRing.h - SPSC ring of batched tile-group descriptors --------===//
//
// THE RUNTIME HALF OF THE S3 THREAD-ROLE BRIDGE (ROADMAP1001 §2, §4.3)
// ------------------------------------------------------------------
// A dual-role kernel runs its HMX section on one dedicated thread (T_HMX,
// see HmxRoleExecutor.h) and everything else on the launching thread. The two
// sides hand tile groups across an SPSC ring: the producer (the launching
// thread, single producer by construction) publishes a descriptor once it has
// finished packing a group of rows; the consumer (T_HMX) takes the group,
// runs the engine section on it, and retires it. This header is the ring
// itself: the descriptor ABI, the two atomic indices, and nothing that needs
// an operating system. The park/wake policy, the thread, and the HMX lock all
// live in the executor (multithreading/HmxRoleExecutor.cpp); this file must
// stay compilable on any host so the host contract test can compile it
// verbatim (bin/runtime/test/test_hmx_spsc_ring_contract.py).
//
// THREE DECISIONS THAT SHAPE THIS RING, EACH WITH ITS PROVENANCE
// --------------------------------------------------------------
// 1. THE DESCRIPTOR IS A *BATCH* OF G ROWS, NOT ONE TILE.
//    A per-tile handoff is a measured net loss on this part: a 32-byte
//    descriptor handoff with wake costs ~900 ns end to end while one m-tile
//    of read-out work is 760 ns (exp/hmx/s2_handoff/probe2.cpp, 2026-10-02;
//    quoted in include/HmxVectorExecutor.h:12-13). The break-even is reached
//    by covering G rows per handoff, which is why `rowCount` -- G -- is a
//    field of the descriptor and not something the consumer re-derives.
//    The executor never interprets G; it belongs to the bound section.
//
// 2. THE INDICES ARE MONOTONE COUNTERS, NOT MASKED POSITIONS.
//    A modular ring-position equality wedges when the consumer retires more
//    than one batch between two of the producer's observations: the wrapping
//    cursor passes the target without ever equaling it, and both sides park
//    forever (observed twice on device; see the drainLocked comment in
//    multithreading/HmxVectorExecutor.cpp and exp/hmx/s2_handoff/probe3.cpp).
//    Monotone `head`/`tail` cannot step over one another: the drain barrier
//    is the exact equality `tail == head`, and the slot index is
//    `counter % capacity`, which works for ANY capacity >= 1 -- a power of
//    two is not required (llama.cpp's hmx-queue.h needs one because it masks;
//    this ring deliberately does not).
//
// 3. THE FENCE POLICY IS ACQUIRE/RELEASE ON THE TWO INDICES.
//    Producer: slot stores -> release store of `head`.
//    Consumer: acquire load of `head` -> slot reads; fn returned -> release
//    store of `tail`. Producer: acquire load of `tail` -> slot reuse.
//    This is the same policy as the tree's existing SPSC ring
//    (multithreading/HmxVectorExecutor.cpp, idxWrite/idxRead), which has run
//    on this device since the read-out split landed, and whose calibration
//    note records the lowering this toolchain produces: the acquire loads
//    become plain loads (confirmed against that file's own disassembly).
//    Compiling THIS ring for the device confirms the same for the store
//    side: hexagon-clang++ lowers the release store of `head` to a plain
//    `memw` -- the whole HmxRoleExecutor object contains zero barrier
//    instructions (verified on the 19.0.04 toolchain, v79). The policy
//    therefore rests on the Hexagon memory model making plain ordered
//    loads/stores sufficient for release-acquire handoff, the same
//    assumption the shipped read-out ring rests on -- not on any barrier
//    this file emits.
//
//    [未验证] WHETHER THAT ASSUMPTION HOLDS FOR *THIS* RING ON THE DEVICE.
//    The host contract test validates the protocol under the HOST memory
//    model, which is stronger -- that is a real limitation, stated here
//    rather than in a footnote: a green host test does not clear the
//    device. The device-side validation is the fence arm of
//    exp/hmx/s2_runtime_probes/ring_probe (ready-to-run scaffold; to be
//    signed in with the S3 device window): a two-thread torture that
//    publishes marker-stamped descriptors and validates them bit-exact on
//    the consumer side, plus a disasm check that the atomics still lower as
//    documented above.
//
// OWNERSHIP MODEL (what "retire" means)
// -------------------------------------
// A slot's tile data never moves: it lives in the compiler-allocated VTCM
// slot the descriptor names, and the ring transfers OWNERSHIP only. The
// producer owns a slot from `poke`-time until `publish` makes the descriptor
// visible; the consumer owns it from the publish it observes until `retire`.
// Two threads writing the same tile simultaneously is a protocol violation
// the ring prevents dynamically (the producer cannot lap into a slot the
// consumer has not retired: the full check), and the S3 static checker
// prevents structurally (def-use on the tile values). The host contract test
// additionally detects violations at the data level (double-write detection
// through the driver's shadow memory).
//
// MEMORY
// ------
// The ring itself is small (capacity x 24 bytes) and lives in DDR on the
// device (malloc; HmxRoleExecutor.cpp). The 64-bit `event` word follows the
// event-wire discipline of HmxToLLVMPass.cpp's readEventWord: opaque,
// unsigned, never reinterpreted by the runtime and never reconstructed from
// a pointer. Both sides of the ABI agree on the descriptor layout by
// construction -- they compile this same header -- and the static_assert
// below makes any field change a loud build break everywhere at once.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HMX_SPSC_RING_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_HMX_SPSC_RING_H

#include <atomic>
#include <cstdint>

/// One group of G m-rows, handed from the producer to the consumer.
///
/// 24 bytes on the target (32-bit pointers), pinned by the static_assert
/// below. The measured handoff cost this descriptor rides on was taken with
/// a 32-byte descriptor (exp/hmx/s2_handoff/probe2.cpp), so the batch-size
/// table derived from that figure stays conservative.
///
/// The fields are the minimum the RING protocol needs; per-kernel geometry
/// (strides, crouton grid) is static knowledge of the bound section and does
/// not travel through the ring.
struct HmxTileGroupDesc {
  uint32_t slot;     ///< VTCM slot the group's tiles live in (data does not move)
  uint32_t rowStart; ///< first m-row of the group
  uint32_t rowCount; ///< G: rows per handoff (see decision 1 in the file header)
  /// Opaque ownership word in the event-wire discipline (readEventWord,
  /// HmxToLLVMPass.cpp): unsigned, opaque, never reinterpreted here.
  uint64_t event;
};
static_assert(sizeof(HmxTileGroupDesc) == 24,
              "the descriptor ABI changed size; every side compiles this "
              "header, so this break is loud by construction -- update the "
              "compiler emission and the host contract test with it");

/// Single-producer / single-consumer ring of HmxTileGroupDesc.
///
/// Capacity is a RUNTIME PARAMETER, never a constant of this file: the
/// compiler side derives it from the tile-ring geometry of the kernel
/// (ROADMAP1001 §2 choice 4: "环深由 HmxPartitionPass 的 tile 环推导，不拍
/// 脑袋"). Valid capacities are [1, 2^31); 2^31 is the ceiling because the
/// counters below are 32-bit and the occupancy arithmetic is unsigned
/// difference (safe while occupancy < 2^31, which the full check guarantees:
/// occupancy never exceeds `capacity`).
///
/// All methods are inline and free of operating-system calls.
struct HmxSpscRing {
  /// Slot storage, caller-provided (DDR on the device). Not atomic: it is
  /// swapped only under the drain-before-rebind protocol documented on
  /// reset(), never while the consumer can be mid-pass.
  HmxTileGroupDesc *slots = nullptr;
  /// Maximum in-flight descriptors. Parameter, not a constant (see above).
  uint32_t capacity = 0;

  /// Producer's next publish counter. Monotone; wraps at 2^32, which the
  /// unsigned-difference arithmetic absorbs (decision 2 in the file header).
  std::atomic<uint32_t> head{0};
  /// Consumer's next retire counter. Monotone twin of `head` and the drain
  /// barrier's word: it advances only AFTER the bound section returned, so
  /// observing `tail == head` means every published group's side effects are
  /// complete. Also the word the producer parks on in drain (the device
  /// wrapper futexes on it; monotone, so the equality the waiter checks can
  /// never wrap past itself the way a masked index can).
  std::atomic<uint32_t> tail{0};
  /// Wake word, bumped once per accepted publish; the word the consumer
  /// parks on when idle (same shape as hmx-queue.h `seqn` and
  /// HmxVectorExecutor's `seqn`). Never reset, so a parked consumer's
  /// sampled value stays distinct from the live one across rebinds.
  std::atomic<uint32_t> seqn{0};

  /// Re-arm the ring for a new producer epoch. LEGAL ONLY while the consumer
  /// cannot be mid-pass: after a completed drain the consumer touches
  /// nothing but `seqn`/`stop` until the next publish, and the next publish
  /// cannot happen until the rebinding producer returns (single producer).
  /// The executor's rebind establishes exactly that window; the slots
  /// pointer/capacity may be swapped in the same window (plain stores are
  /// enough for the same reason).
  void reset() {
    head.store(0, std::memory_order_release);
    tail.store(0, std::memory_order_release);
  }

  /// Publish one descriptor. Returns false when the ring is full -- it then
  /// does NOT write the slot, and the caller owns the dropped work (submit
  /// never blocks the engine thread; see HmxRoleExecutor.h).
  ///
  /// Ordering: the slot stores are ordered before the release store of
  /// `head`, so a consumer that acquires `head` sees the whole descriptor;
  /// the acquire load of `tail` pairs with the consumer's release retire so
  /// a slot is never reused while its reads are still in flight.
  bool publish(const HmxTileGroupDesc &desc) {
    const uint32_t h = head.load(std::memory_order_relaxed);
    const uint32_t t = tail.load(std::memory_order_acquire);
    if (h - t >= capacity) {
      return false; // full: the consumer owns every slot
    }
    slots[h % capacity] = desc;
    head.store(h + 1, std::memory_order_release);
    seqn.fetch_add(1, std::memory_order_release);
    return true;
  }

  /// In-flight count as the producer sees it (producer-side diagnostic and
  /// space query; `head` is producer-private, hence relaxed).
  uint32_t occupied() const {
    return head.load(std::memory_order_relaxed) -
           tail.load(std::memory_order_acquire);
  }

  /// Whether work is available for a consumer whose private cursor is
  /// `consumed` (the consumer derives it from `tail` at pass entry; `tail`
  /// is consumer-private between retires, hence relaxed there).
  bool hasPending(uint32_t consumed) const {
    return consumed != head.load(std::memory_order_acquire);
  }

  /// The descriptor at the consumer's cursor. Legal only while
  /// hasPending(consumed) -- i.e. the slot holds a published, unretired group
  /// the consumer owns.
  const HmxTileGroupDesc &peek(uint32_t consumed) const {
    return slots[consumed % capacity];
  }

  /// Retire the group at the consumer's cursor: advance `tail` past it.
  /// MUST be called only after the bound section has returned for that group
  /// -- `tail` is the drain barrier's word, so retiring early would let a
  /// drain return before the work's side effects landed (the barrier comment
  /// in HmxVectorExecutor.cpp's drainLocked is the measured story of why
  /// this ordering is load-bearing).
  void retire(uint32_t consumed) {
    tail.store(consumed + 1, std::memory_order_release);
  }

  /// Sample the drain target once, at barrier entry. The producer calling
  /// drain publishes nothing while it waits (it IS the producer), so the
  /// sample names exactly the work in flight.
  uint32_t drainTarget() const { return head.load(std::memory_order_acquire); }

  /// Exact "all drained" predicate: `tail` never exceeds `head` (the consumer
  /// only retires published groups), so equality is reachable and complete.
  bool drainedTo(uint32_t target) const {
    return tail.load(std::memory_order_acquire) == target;
  }
};

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HMX_SPSC_RING_H
