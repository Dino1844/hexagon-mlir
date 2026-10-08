//===- HmxRoleExecutor.cpp - Run the HMX section on a resident thread ------===//
//
// The design, the measurements every decision rests on, and the hazard that
// makes the executor lazy are all in include/HmxRoleExecutor.h. This file is
// only the implementation of the five entries that header declares; the ABI
// header is the contract and is not edited here. The ring protocol itself
// lives in include/HmxSpscRing.h and is not restated below.
//
// THE FOUR THINGS THAT SHAPE EVERY DECISION BELOW
// ------------------------------------------------
//  1. The bound thread is RESIDENT and its lifetime is the PROCESS. Creating
//     a thread measured ~18 us on this device (exp/hmx/s2_handoff/probe2.cpp)
//     and the per-launch thread identity of the wrapper's ThreadManager is
//     what sank the first enableWorkspaceResident attempt (rotating thread
//     serials; see include/HmxRoleExecutor.h's provenance block). Nothing on
//     a per-launch path may create, join, or stop this thread.
//
//  2. The thread takes the HMX lock ONCE, at start, and holds it for life.
//     That is the first half of the lock-ownership migration (ROADMAP1001
//     §4.2); the per-kernel ensure/unlock pairing stays the legacy path and
//     this file does not touch it. The lock is never released -- not at
//     thread exit, not ever -- per the recorded decision; the unit goes back
//     when the process tears down.
//
//  3. Every wait is either bounded or blocking; there is no unbounded spin
//     anywhere, on purpose: the first attempt at measuring the vector-side
//     handoff spun on `while (flag != tok);` and hung the DSP for 13 minutes
//     while holding the device lock (exp/hmx/s2_handoff/probe2.cpp:17-21).
//     Both waiters spin kPollCount times and then park in qurt_futex_wait.
//
//  4. drain() IS a real correctness barrier, and it is monotone: the ring's
//     `tail` advances only after the bound section RETURNED, and never
//     exceeds `head`, so `tail == head` (sampled once at barrier entry) is
//     the exact "all executed" predicate and cannot wedge the way a modular
//     ring-position equality can (the wedge story is measured twice over in
//     HmxVectorExecutor.cpp's drainLocked comment).
//
//===----------------------------------------------------------------------===//

#include "HmxRoleExecutor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// The dispatch channel's shared suffix header (ROADMAP1001 section 4.6). The
// compiler emission includes the same file, so the symbol spellings agree by
// construction. Plain C on purpose -- see its file header.
#include "HmxRoleChannel.h"

// dlsym for the channel probe. On the device this is the DSP libc's dlfcn
// (target/hexagon/include/dlfcn.h in the SDK tools); on the host contract
// test it is glibc's. The entry never spells a handle constant itself -- the
// caller passes RTLD_SELF on the device, a dlopen handle on the host -- so
// the two platforms' different RTLD_* values never meet this file.
#include <dlfcn.h>

// HAP_farf.h uses `, ##__VA_ARGS__`, which -pedantic rejects and this target
// compiles with -Werror (clang). Scoped to the include, same as
// HmxVectorExecutor.cpp. Guarded on __clang__: the diagnostic name is a
// clang one, and the host contract test compiles this file with g++, where
// the bare pragma would only add -Wpragmas noise.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <HAP_farf.h>
#pragma clang diagnostic pop
#else
#include <HAP_farf.h>
#endif

#include <qurt_futex.h>
#include <qurt_thread.h>

// The engine-precondition entry the compiler emits per kernel
// (HmxToLLVMPass.cpp kHmxEnsureFn -> HexagonCAPI.cpp). Declared locally with
// the _dsp device spelling, exactly as the leaf-bw probe wrapper does
// (exp/hmx/leaf_bw_probe/wrapper_leafbw.cpp:88); it resolves against the
// runtime bitcode linked into the kernel module at .so link time. This
// archive member is pulled only when a kernel references this executor's
// entries, so the reference stays lazy with the rest of this file.
extern "C" void hexagon_runtime_hmx_ensure_dsp(void);

namespace {

//===----------------------------------------------------------------------===//
// Tunables (each with its provenance; none is a ring depth -- depth is a
// bind() parameter, see HmxSpscRing.h)
//===----------------------------------------------------------------------===//

/// Bounded spin count before parking. 2000 iterations ~= 10.5 us was
/// calibrated on this part for the vector executor's idle loop
/// (exp/hmx/t2_handoff/poll_probe.cpp; see the full calibration note in
/// HmxVectorExecutor.cpp). Borrowed here because the wait shape is the same
/// (acquire-load a word, compare, decrement) and the budget's job is the
/// same: cover the producer's inter-publish gap so the steady state catches
/// the next group by re-reading instead of parking. A future part should
/// recalibrate with the ring probe, not inherit this number.
constexpr unsigned kPollCount = 2000;

/// Bound-thread stack, from the same source as the vector executor's
/// (hmx-queue.c matches HMX_QUEUE_STACK_SIZE): a resident worker whose body
/// is one descriptor dispatch. malloc'd DDR, where a QuRT stack must live.
constexpr size_t kStackBytes = 16384;

/// Lowest priority the bound thread may be given (QURT_LOWEST_PRIO,
/// hmx-queue.c:15).
constexpr int kLowestPriority = 254;

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

/// All executor state. Constant-initialised so it lands in .bss with no
/// dynamic initialiser and no atexit hook: the runtime is linked into a
/// dlopen'd kernel module, and a static destructor running at dlclose would
/// race the thread this file keeps alive (same discipline as
/// HmxVectorExecutor.cpp's Executor).
struct RoleExecutor {
  /// The ring. slots/capacity are set by bind() before `started` is
  /// published and swapped only in the drain-before-rebind window;
  /// head/tail/seqn are the protocol words (HmxSpscRing.h).
  HmxSpscRing ring;

  /// The section the bound thread runs. Written by bind(), which drains
  /// first, so it never changes while a group is in flight.
  std::atomic<HmxSectionFn> fn{nullptr};

  /// Set by join(); the bound thread exits at its next idle check.
  std::atomic<bool> stop{false};
  /// Whether the bound thread exists. Written by the producer side only.
  std::atomic<bool> started{false};

  qurt_thread_t thread{};
  void *stack{nullptr};
};

RoleExecutor gRoleExecutor;

//===----------------------------------------------------------------------===//
// Producer side
//===----------------------------------------------------------------------===//

/// Block until every group published before this call has been executed.
///
/// Monotone barrier: sample `head` once, then wait for `tail` to reach it.
/// The caller of drain is the producer (single producer), so it publishes
/// nothing while this runs and the sample names exactly the work in flight;
/// `tail` never exceeds `head`, so equality is exact and reachable. The
/// bound thread bumps `tail` only after the section RETURNED, so a return
/// from here is never early -- the destinations the sections wrote are
/// visible to the caller.
///
/// Bounded spin, then park on the ring's `tail` word, with the re-check
/// before the park: any retire after `seen` was sampled changes the word, so
/// the wait returns immediately instead of sleeping through the last
/// completion (the lost-wakeup window the vector executor's device log
/// pinned down; probe3.cpp:223-244).
void drainLocked(RoleExecutor &e) {
  const uint32_t target = e.ring.drainTarget();

  unsigned spins = kPollCount;
  while (!e.ring.drainedTo(target)) {
    if (spins != 0u) {
      --spins;
      continue;
    }
    const uint32_t seen =
        e.ring.tail.load(std::memory_order_acquire);
    if (e.ring.drainedTo(target)) {
      break;
    }
    qurt_futex_wait(&e.ring.tail, static_cast<int>(seen));
  }
}

//===----------------------------------------------------------------------===//
// The granular barrier (wait_retired; HmxRoleExecutor.h's entry comment is
// the contract this implements)
//===----------------------------------------------------------------------===//

/// Half-window "at or past" on the ring's monotone 32-bit counters.
///
/// `tail` and any caller-supplied target are both monotone counters that wrap
/// at 2^32 (HmxSpscRing.h, decision 2), so a plain `>=` on the wrapped values
/// is wrong across the wrap and an exact equality wedges when the consumer
/// steps past the target between two observations -- the exact bug that made
/// the ring monotone in the first place. The unsigned difference
/// `word - target` is small (< 2^31) exactly when `word` is at or past
/// `target` within a half window, and the ring's own invariants bound every
/// REAL distance below 2^31: occupancy never exceeds `capacity` < 2^31 (the
/// full check), and a sane target satisfies target <= head. Distances of 2^31
/// or more can therefore only come from a target the producer never published,
/// which the unreachable check below catches and names rather than parking on.
static inline bool counterReached(uint32_t word, uint32_t target) {
  return (word - target) < 0x80000000u;
}

/// Block until the retire counter reaches `target` (see the ABI header for the
/// read-out coexistence this serves and the proof it carries).
///
/// Same park policy as drainLocked: bounded spin, re-check before the park,
/// futex on the ring's `tail` word (the consumer wakes it once per retire).
/// The one way it DIFFERS from drain is the unreachable-target path: drain's
/// target is sampled from `head` itself so it is satisfiable by construction,
/// while this target is a caller-supplied group count -- and waiting forever
/// on a count the ring can never reach is the unbounded wait this file's
/// constraint 3 forbids. The check is sound because the caller IS the producer
/// (single producer, HmxRoleExecutor.h): it publishes nothing while it waits,
/// so `head` is stable for the whole call and a target beyond it now is beyond
/// it forever.
static void waitRetiredLocked(RoleExecutor &e, uint32_t target) {
  // Nothing has ever been published that retire-count 0 would wait for; also
  // sidesteps the wrap edge where a large retired `tail` compares ambiguously
  // against a zero target.
  if (target == 0u)
    return;
  if (counterReached(e.ring.tail.load(std::memory_order_acquire), target))
    return; // already satisfied: no spin, no park, no wakeup owed
  if (!counterReached(e.ring.head.load(std::memory_order_acquire), target)) {
    // Beyond the published head: the ring can never retire it. A caller bug
    // (the producer mis-derived its group count); name it and return rather
    // than parking forever -- the loud-and-recoverable side of the choice,
    // same direction as submit's short return, opposite of a DSP hang.
    FARF(ERROR, "hmx-role: wait_retired target %u is beyond the published "
                "head %u; it can never retire -- returning without waiting "
                "(caller bug)",
         target, e.ring.head.load(std::memory_order_relaxed));
    return;
  }

  unsigned spins = kPollCount;
  while (!counterReached(e.ring.tail.load(std::memory_order_acquire), target)) {
    if (spins != 0u) {
      --spins;
      continue;
    }
    const uint32_t seen =
        e.ring.tail.load(std::memory_order_acquire);
    if (counterReached(seen, target)) {
      break;
    }
    qurt_futex_wait(&e.ring.tail, static_cast<int>(seen));
  }
}

//===----------------------------------------------------------------------===//
// Consumer side (the bound thread)
//===----------------------------------------------------------------------===//

/// Run every currently published group, in order, on this thread.
///
/// One section call per group with count == 1 (the calling discipline pinned
/// in HmxRoleExecutor.h; coalescing a run would make `count`'s meaning
/// depend on ring geometry). `tail` advances only AFTER the section
/// returned, per group, and each retire wakes any drain parked on it. The
/// consumer cursor is re-derived from `tail` at entry -- `tail` is
/// consumer-private between retires -- and the `head` snapshot bounds this
/// pass; anything published later waits for the next seqn bump.
void processAvailable(RoleExecutor &e) {
  const HmxSectionFn fn = e.fn.load(std::memory_order_acquire);
  if (fn == nullptr) {
    // Unreachable: submit() refuses everything before bind() stored a
    // section. A hard stop rather than a skip, same fail-closed choice as
    // the vector executor's null-fn branch: advancing `tail` without
    // running the section would hand drain a barrier that passed while the
    // work never happened.
    FARF(ERROR, "hmx-role: queued group with no section bound");
    std::abort();
  }

  uint32_t consumed = e.ring.tail.load(std::memory_order_relaxed);
  const uint32_t snap = e.ring.head.load(std::memory_order_acquire);
  while (consumed != snap) {
    fn(&e.ring.slots[consumed % e.ring.capacity], 1);
    // retire(consumed) advances tail to consumed+1 -- pass the group's OWN
    // cursor, not an incremented one: tail must never exceed head (the
    // drain barrier's exactness depends on it, HmxSpscRing.h).
    e.ring.retire(consumed);
    ++consumed;
    qurt_futex_wake(&e.ring.tail, 1);
  }
}

/// The resident bound thread: T_HMX.
///
/// Structure copied from the vector executor's vectorThreadEntry, which
/// copied hmx-queue.c:70-86 -- compare the sequence counter against what was
/// last seen, do the work when it moved, otherwise spin a BOUNDED number of
/// times before parking in qurt_futex_wait on the same counter.
void roleThreadEntry(void *arg) {
  RoleExecutor &e = *static_cast<RoleExecutor *>(arg);

  // THE LIFETIME LOCK (HmxRoleExecutor.h's second provenance bullet). Once,
  // at thread start, before this thread can run any section; never released.
  // This also constructs the runtime singleton (powers HMX up and acquires
  // it) if nothing else already did. Every section this thread runs from
  // here on executes with the engine precondition already satisfied, so the
  // compiler side will not need to emit a per-section ensure for work routed
  // here -- that is the entire point of the migration's first half.
  hexagon_runtime_hmx_ensure_dsp();

  // Starts at 0, which is also the ring's initial seqn, so the first
  // iteration takes the idle path (spin out the budget, then park). The
  // value is never reset across rebinds (HmxSpscRing.h, `seqn`).
  uint32_t prevSeqn = 0;
  unsigned pollCnt = kPollCount;

  for (;;) {
    const uint32_t seqn = e.ring.seqn.load(std::memory_order_acquire);

    if (seqn != prevSeqn) {
      prevSeqn = seqn;
      pollCnt = kPollCount;
      processAvailable(e);
      continue;
    }

    if (e.stop.load(std::memory_order_acquire)) {
      break;
    }

    if (pollCnt == 0u) {
      // RE-CHECK BEFORE PARKING: closes the window between the seqn load at
      // the top of the loop and this wait (the lost-wakeup the vector
      // executor's device log pinned; probe3.cpp:223-244). A group that
      // arrived during the spin is processed here without parking at all.
      if (e.ring.seqn.load(std::memory_order_acquire) != prevSeqn) {
        pollCnt = kPollCount;
        continue;
      }
      qurt_futex_wait(&e.ring.seqn, static_cast<int>(prevSeqn));
      pollCnt = kPollCount;
      continue;
    }

    // Bounded spin, then park on the next iteration.
    --pollCnt;
  }

  // No HMX unlock here, ever: the lock is held for the thread's lifetime
  // (HmxRoleExecutor.h, decision 2). The thread only reaches this point via
  // join(), which is session teardown; the unit returns to the system when
  // the process goes away.
}

//===----------------------------------------------------------------------===//
// Thread lifecycle
//===----------------------------------------------------------------------===//

/// Create the resident bound thread. Called from bind() only, once per
/// process (the `started` guard), never on a per-launch path.
///
/// Returns 0 on success, non-zero if the thread could not be created.
int startRoleThread(RoleExecutor &e) {
  // qurt_thread_attr_set_stack_addr requires 8-byte alignment
  // (qurt_thread.h:471); malloc satisfies it, but rounding up keeps that an
  // observation rather than an assumption (same note as the vector
  // executor's startVectorThread).
  constexpr size_t kAlign = 8;
  void *raw = std::malloc(kStackBytes + kAlign);
  if (raw == nullptr) {
    return -1;
  }
  const uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(raw) + (kAlign - 1)) & ~(kAlign - 1);
  void *stack = reinterpret_cast<void *>(aligned);

  // Match the calling thread's priority, clamped exactly as hmx-queue.c
  // does: a bound thread that outranks its producer would starve the
  // pipeline it is supposed to overlap with; one that is starved turns
  // drain() into a serialisation.
  int prio = qurt_thread_get_priority(qurt_thread_get_id());
  if (prio < 1) {
    prio = 1;
  }
  if (prio > kLowestPriority) {
    prio = kLowestPriority;
  }

  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "hmx-role");
  qurt_thread_attr_set_priority(&attr, static_cast<unsigned short>(prio));
  qurt_thread_attr_set_stack_size(&attr, static_cast<unsigned>(kStackBytes));
  qurt_thread_attr_set_stack_addr(&attr, stack);

  // Published before the create, not after: the thread starts the moment
  // qurt_thread_create returns, and join() frees `e.stack`. Recording it
  // first means the pointer is never lost in the window between the two.
  e.stack = raw;

  const int err = qurt_thread_create(&e.thread, &attr, roleThreadEntry, &e);
  if (err != 0) {
    FARF(ERROR, "hmx-role: qurt_thread_create failed (%d)", err);
    e.stack = nullptr;
    std::free(raw);
    return -1;
  }
  return 0;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public ABI
//===----------------------------------------------------------------------===//
//
// The enclosing library is compiled with -fvisibility=hidden (inherited from
// the parent Triton CMake scope), so the attribute is what puts these four
// in the archive's symbol table -- the same mechanism as the vector
// executor's entries (HmxVectorExecutor.cpp's ABI comment). Nothing else in
// this file is visible outside it.

extern "C" {

__attribute__((visibility("default"))) int32_t
hexagon_runtime_hmx_role_bind(HmxSectionFn fn, uint32_t depth) {
  if (fn == nullptr) {
    return HEXMLIR_HMX_ROLE_ERR_NULL_FN;
  }
  if (depth == 0u) {
    // Depth is a parameter the compiler side derives from the kernel's tile
    // ring (HmxSpscRing.h); there is no default to fall back to and inventing
    // one here would hide a compiler-side bug.
    return HEXMLIR_HMX_ROLE_ERR_DEPTH;
  }

  RoleExecutor &e = gRoleExecutor;

  if (!e.started.load(std::memory_order_acquire)) {
    // First bind: allocate the ring storage (DDR; the ring is small and the
    // tile data itself never moves through it), then start the thread. A
    // stop left set by an earlier join() is cleared BEFORE the create: the
    // thread starts running as soon as qurt_thread_create returns, so a
    // clear afterwards would be a race (same note as the vector executor's
    // configure).
    HmxTileGroupDesc *slots = static_cast<HmxTileGroupDesc *>(
        std::malloc(sizeof(HmxTileGroupDesc) *
                    static_cast<size_t>(depth)));
    if (slots == nullptr) {
      return HEXMLIR_HMX_ROLE_ERR_CREATE;
    }
    e.ring.slots = slots;
    e.ring.capacity = depth;
    e.ring.reset();
    e.stop.store(false, std::memory_order_release);
    if (startRoleThread(e) != 0) {
      std::free(slots);
      e.ring.slots = nullptr;
      e.ring.capacity = 0;
      return HEXMLIR_HMX_ROLE_ERR_CREATE;
    }
    e.fn.store(fn, std::memory_order_release);
    e.started.store(true, std::memory_order_release);
    return HEXMLIR_HMX_ROLE_OK;
  }

  // Rebind (a recompile in the same process). Quiesce first: drain waits
  // for every in-flight group from the previous epoch, so the section swap
  // can never run a previous kernel's section against new descriptors --
  // and the quiesced window is also the ONLY window in which the ring
  // storage may be reallocated (after the last retire the bound thread
  // touches nothing but seqn/stop until the next publish, and the next
  // publish cannot happen until this bind returns: single producer).
  drainLocked(e);
  e.fn.store(fn, std::memory_order_release);
  if (depth > e.ring.capacity) {
    HmxTileGroupDesc *slots = static_cast<HmxTileGroupDesc *>(
        std::malloc(sizeof(HmxTileGroupDesc) *
                    static_cast<size_t>(depth)));
    if (slots != nullptr) {
      // Grow-only: a smaller rebind keeps the larger ring above (a deeper
      // ring than requested is harmless; shrinking would buy nothing and
      // pay another realloc).
      HmxTileGroupDesc *old = e.ring.slots;
      e.ring.reset();
      e.ring.slots = slots;
      e.ring.capacity = depth;
      std::free(old);
    }
    // A failed grow keeps the old storage: the ring still works at the old
    // depth, and bind reports success -- the caller asked for "at least"
    // overlap depth, and refusing a whole recompile over a smaller ring
    // would trade a working kernel for nothing.
  }
  return HEXMLIR_HMX_ROLE_OK;
}

__attribute__((visibility("default"))) uint32_t
hexagon_runtime_hmx_role_submit(const HmxTileGroupDesc *groups, uint32_t count) {
  RoleExecutor &e = gRoleExecutor;
  if (groups == nullptr || count == 0u) {
    return 0;
  }

  if (!e.started.load(std::memory_order_acquire) ||
      e.fn.load(std::memory_order_acquire) == nullptr) {
    // Not bound: accept nothing rather than queue work the resident thread
    // would run against a stale section. Reported once so a kernel that
    // never binds is obvious in the log instead of silent.
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
      FARF(ERROR, "hmx-role: submit before bind; dropping %u group(s)", count);
    }
    return 0;
  }
  if (e.stop.load(std::memory_order_acquire)) {
    return 0;
  }

  uint32_t accepted = 0;
  while (accepted < count && e.ring.publish(groups[accepted])) {
    ++accepted;
  }
  if (accepted != 0u) {
    // Wake the consumer. The ring bumped its seqn wake word inside publish;
    // the WAKE call is platform policy and lives here, not in the ring --
    // same split as HmxVectorExecutor (its publish bumps seqn and wakes).
    // One wake per submit is enough: the consumer processes everything
    // pending per pass, and a consumer still spinning sees the seqn move on
    // its own.
    qurt_futex_wake(&e.ring.seqn, 1);
  }
  if (accepted < count) {
    // Ring full: the bound thread cannot keep up. Blocking here would stall
    // the producer for every group behind the waited one, so the tail is
    // dropped and the caller owns it (HmxRoleExecutor.h's submit contract).
    FARF(ERROR, "hmx-role: ring full, accepted %u of %u group(s); the caller "
                "must run the tail itself",
         accepted, count);
  }
  return accepted;
}

__attribute__((visibility("default"))) void hexagon_runtime_hmx_role_drain(void) {
  drainLocked(gRoleExecutor);
}

__attribute__((visibility("default"))) void
hexagon_runtime_hmx_role_wait_retired(uint32_t target) {
  waitRetiredLocked(gRoleExecutor, target);
}

__attribute__((visibility("default"))) void hexagon_runtime_hmx_role_join(void) {
  RoleExecutor &e = gRoleExecutor;
  if (!e.started.load(std::memory_order_acquire)) {
    return;
  }

  // Retire anything still queued before asking the thread to leave, so join
  // cannot race a section that is writing into caller memory. Session
  // teardown only -- never per launch (HmxRoleExecutor.h's first provenance
  // bullet).
  drainLocked(e);

  e.stop.store(true, std::memory_order_release);
  e.ring.seqn.fetch_add(1u, std::memory_order_release);
  qurt_futex_wake(&e.ring.seqn, 1);

  int status = 0;
  qurt_thread_join(e.thread, &status);

  // The thread has been reaped, so nothing walks the stack and the pages can
  // go back. The ring storage goes too -- the next bind allocates fresh
  // storage for its own depth (the grow-only rule lives inside one bound
  // epoch, not across join). `fn` stays until the next bind (a submit after
  // join is refused on `started`, not on a null section, so the refusal
  // reason stays honest).
  std::free(e.stack);
  e.stack = nullptr;
  e.thread = qurt_thread_t{};
  std::free(e.ring.slots);
  e.ring.slots = nullptr;
  e.ring.capacity = 0;
  e.ring.reset();
  e.started.store(false, std::memory_order_release);
  e.stop.store(false, std::memory_order_release);
}

//===----------------------------------------------------------------------===//
// The host->device dispatch channel (HmxRoleChannel.h, ROADMAP1001 4.6)
//===----------------------------------------------------------------------===//
//
// WHY THIS ENTRY LIVES IN THIS ARCHIVE MEMBER AND NOT ITS OWN FILE
// ---------------------------------------------------------------
// The launch side (the generated wrapper main) references this entry WEAKLY,
// exactly the way it references hexagon_runtime_hmx_exec_drain and
// hexagon_runtime_hmx_exec_dump (hexagon_launcher_base.py's headers block):
// a weak reference resolves to null when the definition is absent, which is
// what keeps a legacy kernel's wrapper a no-op without any launch ABI
// change. But a weak reference PULLS NOTHING from an archive: the member is
// only linked when some strong reference demands it. The strong references
// that exist are the compiler-emitted submit/drain calls of a dual-role
// kernel's body -- and they resolve against THIS member
// (hexagon_runtime_hmx_role_submit / _drain above). So this member, and only
// this member, is present in exactly those .so's that are dual-role, which
// is precisely the set for which the probe must exist. A separate channel
// translation unit would stay in the archive for every dual-role .so and
// the wrapper's weak reference would be null -- the probe would never run.
//
// THE PROBE
// ---------
// dlsym for `<entry>__hmx_section` (the section entry point) and
// `<entry>__hmx_role_depth` (the companion uint32_t depth object); both
// names are built from the suffixes in HmxRoleChannel.h, which the compiler
// emission includes too. Neither found: legacy, return 0, touch nothing.
// Exactly one found: a half-emitted channel -- refuse (the caller must not
// run the kernel; see the header's return-code block for why a refused
// launch is better than a bound-less dual-role kernel). Both found: bind.
//
// [未验证] WHETHER dlsym(RTLD_SELF, ...) SEARCHES THE CALLING .so ON THE
// DEVICE. The DSP libc's dlfcn.h documents RTLD_SELF as "search the caller
// itself", and this entry, the wrapper that calls it, and the section
// symbols all live in the same .so by the archive-pull argument above, so
// the handle names the right module. The host contract test proves the
// probe's logic against real dlopen/dlsym on stub modules; the device-side
// confirmation (and, if RTLD_SELF's scope differs there, the one-line handle
// change in the wrapper) belongs to the S3 device window, which is where
// this channel gets its first real launch.
__attribute__((visibility("default"))) int32_t
hexagon_runtime_hmx_role_channel_launch(void *handle, const char *kernel_entry) {
  if (kernel_entry == nullptr || handle == nullptr) {
    // A null handle is refused alongside a null name: passing 0 to dlsym is
    // RTLD_DEFAULT on some platforms (a global-scope search), which would
    // make the probe's answer depend on what else is loaded rather than on
    // the module being launched. The launch side always has a handle.
    FARF(ERROR, "hmx-role: channel probe needs a dlsym handle and a kernel "
                "entry name");
    return HMX_ROLE_CHANNEL_ERR_ARG;
  }

  // One buffer, reused: the two names share the prefix and neither outlives
  // its dlsym call. Long enough for a kernel entry name plus the suffix with
  // room to spare; a name that does not fit is refused rather than
  // truncated, because a truncated name probes a symbol nobody emitted.
  char name[256];
  const size_t entryLen = std::strlen(kernel_entry);

  auto buildName = [&](const char *suffix) -> bool {
    const size_t suffixLen = std::strlen(suffix);
    if (entryLen + suffixLen + 1 > sizeof(name)) {
      FARF(ERROR, "hmx-role: kernel entry name '%s' plus the channel suffix "
                  "does not fit the probe buffer",
           kernel_entry);
      return false;
    }
    std::memcpy(name, kernel_entry, entryLen);
    std::memcpy(name + entryLen, suffix, suffixLen + 1);
    return true;
  };

  if (!buildName(HMX_ROLE_SECTION_SUFFIX))
    return HMX_ROLE_CHANNEL_ERR_NAME;
  // POSIX requires dlsym's void* to convert to a function pointer; both the
  // DSP libc and the host libcs used here honor it.
  void *section = dlsym(handle, name);

  if (!buildName(HMX_ROLE_DEPTH_SUFFIX))
    return HMX_ROLE_CHANNEL_ERR_NAME;
  void *depthSym = dlsym(handle, name);

  if (section == nullptr && depthSym == nullptr)
    return HMX_ROLE_CHANNEL_LEGACY; // today's path: nothing bound, nothing touched

  if (section == nullptr || depthSym == nullptr) {
    // Half a channel: the compiler emits the section and the depth as one
    // unit (HmxToLLVMPass), so one without the other is a broken build, not
    // a legacy kernel. Binding anyway would need a guessed depth; treating
    // it as legacy would run a dual-role kernel unbound, whose submits drop
    // every group. Both are wrong answers, so the launch is refused.
    FARF(ERROR, "hmx-role: half-emitted channel for kernel '%s': section=%d "
                "depth=%d; refusing to run the kernel",
         kernel_entry, section != nullptr, depthSym != nullptr);
    return HMX_ROLE_CHANNEL_ERR_HALF;
  }

  const uint32_t depth = *static_cast<const uint32_t *>(depthSym);
  const int32_t rc = hexagon_runtime_hmx_role_bind(
      reinterpret_cast<HmxSectionFn>(section), depth);
  if (rc != HEXMLIR_HMX_ROLE_OK) {
    // bind already logged its own reason for create failures; the depth
    // refusals (null section cannot happen here, depth 0 can) are named
    // here so the log and the return agree.
    FARF(ERROR, "hmx-role: bind refused kernel '%s' (rc=%d, depth=%u)",
         kernel_entry, rc, depth);
    return HMX_ROLE_CHANNEL_ERR_BIND;
  }
  return HMX_ROLE_CHANNEL_BOUND;
}

} // extern "C"
