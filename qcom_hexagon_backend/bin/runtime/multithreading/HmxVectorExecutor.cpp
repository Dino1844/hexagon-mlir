//===- HmxVectorExecutor.cpp - Run HVX read-out on a second thread ---------===//
//
// The design, the measurements it rests on, and the split between what is
// copied from llama.cpp and what is deliberately not, are all in
// include/HmxVectorExecutor.h. This file is only the implementation of the four
// entries that header declares; the header is the ABI and is not edited here.
//
// THE FOUR THINGS THAT SHAPE EVERY DECISION BELOW
// ------------------------------------------------
//  1. The vector thread is RESIDENT. Creating a thread measured ~18 us on this
//     device; the whole read-out the split is trying to hide is 24.3 us. A
//     per-launch thread costs three quarters of the budget before it does any
//     work, so the thread is created once on the first `configure` and lives
//     until `shutdown`. See startVectorThread().
//
//  2. The ring is a single-producer/single-consumer lock-free ring shaped after
//     ggml/src/ggml-hexagon/htp/hmx-queue.h: `idxWrite` / `idxRead` atomics and
//     an `idx_mask` / `capacity` pair. It is the DATA path only. The barrier is
//     deliberately NOT expressed through the ring: drain compares two monotone
//     batch totals (`publishedTotal` vs `retireSeqn`), because an equality on a
//     wrapping ring position wedges when the consumer retires more than one
//     batch between the producer's observations -- which is the steady state
//     whenever drain is a live barrier. See drainLocked().
//
//  3. Every wait in this file is either bounded or blocking. There is no
//     unbounded spin anywhere, on purpose: the first attempt at measuring this
//     handoff spun on `while (flag != tok);` and hung the DSP for 13 minutes
//     while holding the device lock (exp/hmx/s2_handoff/probe2.cpp:17-21).
//     Both waiters spin `kPollCount` times and then park in qurt_futex_wait.
//
//  4. publish() NEVER blocks and drain() IS a real correctness barrier. Those
//     two are not in tension: the consumer bumps `retireSeqn` only AFTER the
//     read-out function has returned, and only once per batch publish()
//     accepted -- so `retireSeqn` never exceeds `publishedTotal` -- and drain()
//     waits for the two totals to equal. See drainLocked().
//
//===----------------------------------------------------------------------===//

#include "HmxVectorExecutor.h"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// HAP_farf.h uses `, ##__VA_ARGS__`, which -pedantic rejects and this target
// compiles with -Werror. The diagnostic is scoped to the include rather than to
// the file so it cannot mask a warning from anything written below.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <HAP_farf.h>
#pragma GCC diagnostic pop

#include <qurt_futex.h>
#include <qurt_hvx.h>
#include <HAP_perf.h>

#include <cstdio>
#include <qurt_thread.h>

namespace {

//===----------------------------------------------------------------------===//
// Tunables
//===----------------------------------------------------------------------===//

/// Ring depth. Power of two, and the producer always leaves one slot empty so
/// that `idxWrite == idxRead` means "empty" without a separate count
/// (hmx-queue.h:75). 16 matches HMX_QUEUE_CAPACITY in
/// ggml/src/ggml-hexagon/htp/main.c:39. Two slots is already enough for the
/// overlap to exist at pipeline-depth=2; the rest cost 32 bytes of .bss each and
/// only let the engine run further ahead than the reader, which is the property
/// that makes llama.cpp's wake latency land on an already-finished generation.
constexpr uint32_t kRingCapacity = 16;
constexpr uint32_t kRingMask = kRingCapacity - 1;
static_assert((kRingCapacity & kRingMask) == 0u, "ring capacity must be 2^n");

/// Depth of the two diagnostic rings in Executor (live counters, 2026-10-04).
/// Power of two, masked by kDiagMask. 256 batch entries at G=4 (8 batches per
/// iteration) is 32 iterations of steady state; the drain ring keeps the last
/// 256 drains. Everything older is only in the aggregate counters.
constexpr uint32_t kDiagCapacity = 256;
constexpr uint32_t kDiagMask = kDiagCapacity - 1;
static_assert((kDiagCapacity & kDiagMask) == 0u, "diag capacity must be 2^n");

/// Bounded spin count: 2000 iterations, ~= 10.5 us per idle transition.
///
/// CALIBRATION (measured 2026-10-04 on this part; probe:
/// exp/hmx/t2_handoff/poll_probe.cpp, log stem t2_poll_cost):
/// one idle iteration -- acquire-load `seqn`, acquire-load `stop`, compare,
/// budget test, decrement, the literal loop body below, confirmed against
/// this file's own disassembly (3 packets; the acquire loads lower to plain
/// loads, no barrier in the loop body) -- costs 11.12 pcyc = 5.27 ns at the
/// measured 2112 pcyc/us. The window the spin must cover is the engine's G
/// rows of mma work between two publishes: ~5 us at G=4 (8 publishes per
/// iteration) and ~10 us at G=8, and the largest observed last-drain wait
/// is 3.62 us (T6). 2000 x 5.27 ns ~= 10.5 us covers the G=8 gap and the
/// drain wait with margin, so in steady state the consumer catches the next
/// publish by re-reading `seqn` instead of parking, and the ~0.5 us
/// publish->consumer wake latency (T6 median 0.52 us) disappears. Still
/// BOUNDED: no path spins more than kPollCount iterations before it parks
/// (design constraint 3 in the file header).
///
/// WHY THIS DEVIATES FROM llama.cpp's v79 = 1
/// -------------------------------------------
/// ggml/src/ggml-hexagon/htp/hmx-queue.h:20-24 ships 2000 above v79 and 1
/// at v79. Its v79 value is a power-saving choice for a phone-side process
/// where spinning steals a core from other work. That trade is inverted
/// here, and both halves are measured on THIS part, not assumed
/// (exp/hmx/t5_hvx_contention):
///  * spinning does not steal the engine's core: two threads at 100% duty
///    each keep 1.969/2.0 throughput with BOTH windows at the full
///    2112 pcyc/us (true parallel units, not time-slicing);
///  * the alternative is not free: every park costs the HVX context pair
///    (372 pcyc, T5 Q2) plus the wake (T6: 0.52 us median handoff), and the
///    consumer parked on 7960 of 8008 batches at G=4 (T6).
/// The `__HVX_ARCH__` tiering the previous revision inherited from
/// hmx-queue.h is gone on purpose: this TU is compiled WITHOUT -mhvx
/// (multithreading/CMakeLists.txt adds no -m flags at all), so the macro is
/// undefined and the >79 branch was dead code. A future part should
/// recalibrate with the probe, not inherit this number.
constexpr unsigned kPollCount = 2000;

/// Vector thread stack. Matches HMX_QUEUE_STACK_SIZE in
/// ggml/src/ggml-hexagon/htp/main.c:40, which sizes it for the same job: a
/// resident worker whose body is one descriptor dispatch. Allocated from DDR by
/// malloc, which is where a QuRT thread stack has to live.
constexpr size_t kStackBytes = 16384;

/// Lowest priority the vector thread may be given (QURT_LOWEST_PRIO,
/// hmx-queue.c:15).
constexpr int kLowestPriority = 254;

/// This executor runs ONE vector thread. See the numThreads handling in
/// hexagon_runtime_hmx_exec_configure for why the ABI field is not a knob yet.
constexpr int32_t kVectorThreads = 1;

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

/// All executor state. Every member is constant-initialised so this lands in
/// .bss with no dynamic initialiser and no atexit hook: the runtime is linked
/// into a dlopen'd kernel module, and a static destructor running at dlclose
/// would have to race the thread this file keeps alive.
///
/// Single producer by construction: the engine thread is the kernel's own
/// thread, and one kernel launch is one producer. Two producer threads would
/// need an MPSC ring and are not supported.
struct Executor {
  /// The ring. Written by the producer only for slots below `idxWrite`, read by
  /// the consumer only for slots at or above `idxRead` and below `idxWrite`;
  /// `idxWrite`/`idxRead` publish those writes, so no slot is ever torn.
  HmxReadoutBatch slots[kRingCapacity];

  // --- diagnostics for "where do the cycles go" ---------------------------------
  // Counters only; they change no behaviour. They exist so the next device run can
  // attribute the per-iteration cost to a specific mechanism rather than to a
  // guess -- which is the only thing five refuted hypotheses have produced so far.
  std::atomic<uint64_t> nPark{0};      ///< qurt_futex_wait calls, consumer side
  std::atomic<uint64_t> nHvxLock{0};   ///< qurt_hvx_lock calls
  std::atomic<uint64_t> nHvxUnlock{0}; ///< qurt_hvx_unlock calls
  std::atomic<uint64_t> nPublish{0};   ///< hexagon_runtime_hmx_exec_publish entries

  /// Next slot the producer will write. Producer-private, so a relaxed load is
  /// enough, but it is an atomic because the consumer reads it.
  std::atomic<uint32_t> idxWrite{0};
  /// Next slot the consumer will read. Consumer-private.
  std::atomic<uint32_t> idxRead{0};
  /// Total batches accepted by publish(), monotonic, never reset. The drain
  /// barrier waits for `retireSeqn` to reach it: two monotone totals can never
  /// step over one another, unlike a ring-position equality, which the
  /// consumer's wrapping cursor can pass without ever equaling (the wedge
  /// observed twice on device; see drainLocked).
  std::atomic<uint32_t> publishedTotal{0};

  /// Diagnostic counters, added 2026-10-03. The read-out split measured as
  /// device-neutral (ON within +1.6% of OFF at 1 MB output, and 2-4% SLOWER at
  /// 2 MB), which has two very different explanations: the vector work never
  /// reached the vector thread, or it did and the engines contend. These
  /// separate them without needing another device experiment: if `batchesRun`
  /// is zero the rewrite is broken; if it is nonzero and `readoutPcyc` is a
  /// substantial fraction of the kernel, the work IS moving and the remaining
  /// question is hardware contention.
  std::atomic<uint32_t> batchesRun{0};
  std::atomic<uint64_t> readoutPcyc{0};

  // --- live-counter rings (2026-10-04) ----------------------------------------
  //
  // The read-back path for the diagnostics above. The counters were already
  // accumulating on every run, but hmxExecTrace's own file never came back
  // from a FastRPC-launched kernel (errata 12 section 5.1), so there was no
  // way to read any of them. These rings change that: the wrapper appends a
  // dump -- hexagon_runtime_hmx_exec_dump, at the bottom of this file -- to
  // perf.txt AFTER the timed region, from the same launch context that
  // demonstrably has working file I/O (the PerfPcycles line comes back every
  // time). Diagnostics only: nothing in this file reads any of these fields
  // to make a decision, and every write is a plain relaxed store.
  //
  // Both rings are indexed by the MONOTONIC batch/drain sequence number, not
  // by the wrapping data-ring position, so every dumped line carries an
  // absolute sequence number and the dump's consumer can align the batch and
  // drain timelines without knowing any ring geometry. Bounded: only the last
  // kDiagCapacity entries survive; everything older is aggregated in the
  // counters above.
  struct BatchDiag {
    uint64_t publishAcceptPcyc; ///< producer: the publish() that accepted it
    uint64_t consumerStartPcyc; ///< consumer: immediately before fn()
    uint64_t consumerEndPcyc;   ///< consumer: immediately after fn() returned
  };
  struct DrainDiag {
    uint64_t enterPcyc; ///< producer: drainLocked() entry
    uint64_t exitPcyc;  ///< producer: barrier passed
  };
  /// Per-batch ring. Entry for batch n lives at diag[n & kDiagMask]. The
  /// producer stamps publishAcceptPcyc; the consumer stamps the other two
  /// fields. No same-slot race is possible: the producer can only be kUsable
  /// (15) batches ahead of the consumer -- the data ring's backpressure -- and
  /// kDiagCapacity (256) is far above that, so the producer's write for batch
  /// n+256 can never overlap the consumer's writes for batch n.
  BatchDiag diag[kDiagCapacity]{};
  /// Per-drain ring, entry for the n-th drainLocked() call at
  /// drainDiag[n & kDiagMask]. Producer-only writer (every drainLocked caller
  /// is the engine thread), so plain stores suffice.
  DrainDiag drainDiag[kDiagCapacity]{};
  /// Monotonic count of drainLocked() calls (configure, drain, shutdown).
  std::atomic<uint64_t> nDrain{0};

  /// Futex word bumped once per publish; what the consumer sleeps on.
  std::atomic<uint32_t> seqn{0};
  /// Futex word bumped once per completed batch; what the producer sleeps on in
  /// drain. Separate from `idxRead` so the producer's wait does not need the
  /// ring indices to be lock-free words it is allowed to scribble on.
  std::atomic<uint32_t> retireSeqn{0};

  /// The read-out the consumer runs. Written by configure(), which drains
  /// first, so the consumer is either blocked or looking at an empty ring
  /// whenever this changes.
  std::atomic<HmxReadoutFn> fn{nullptr};

  /// Set by shutdown(); the consumer exits at its next idle check.
  std::atomic<bool> stop{false};
  /// Whether the consumer thread exists. Written by the producer only.
  std::atomic<bool> started{false};

  qurt_thread_t thread{};
  void *stack{nullptr};
};

Executor gExecutor;

//===----------------------------------------------------------------------===//
// Producer side
//===----------------------------------------------------------------------===//

/// Block until every batch published before this call has been read out.
///
/// This is the correctness barrier, and it is a real one:
///
///  * `target` is sampled once. Nothing the producer does afterwards can be
///    published while this runs, so `target` names exactly the work in flight.
///  * The consumer stores `idxRead` only after `fn` has RETURNED
///    (processAvailable), so observing `idxRead == target` means the read-out
///    for every slot below `target` has already written `dst`.
///  * The return value is therefore never early. That matters because `dst` is
///    caller memory: a barrier that returned before the stores landed would
///    hand the kernel its own uninitialised output, looking correct.
///
/// `retireSeqn` is sampled BEFORE re-checking `idxRead`, and the consumer bumps
/// `retireSeqn` AFTER storing `idxRead`. That ordering is what makes the wakeup
/// unmissable: if the consumer finishes between our two reads we return
/// without waiting, and if it finishes after them the futex word no longer
/// equals the value we sampled, so qurt_futex_wait returns immediately instead
/// of parking.
// Where the accounting lands. FARF is the natural place for a runtime message,
// but FARF output does not come back through `adb logcat` on this rig (checked:
// empty), so a FARF-only accounting line is indistinguishable from no output at
// all. A file under the user's home is what every probe today uses and is
// verified to survive the RPC round trip.
static const char *kReadoutAccountingPath =
    "/data/data/com.termux/files/home/csm/op/hmx_readout_acct.txt";

/// Whether the diagnostic file is written at all. OFF by default.
///
/// WHY THIS EXISTS -- THE ACCOUNTING WAS THE ENTIRE SLOWDOWN
/// --------------------------------------------------------
/// The counters themselves are free: `batchesRun` and `readoutPcyc` are two
/// relaxed atomics bumped on the consumer's own path. It is the FILE WRITE that
/// is ruinous, and it was measured rather than assumed
/// (exp/hmx/s2_handoff/probe3.cpp, arm E, 2026-10-03):
///
///     one fopen + fprintf + fclose append to this path: 684 us median
///
/// A launch performs THREE of them -- two in reportReadoutAccounting() below and
/// one in configure() -- but ONLY where the file I/O works. Measured 2026-10-03
/// 23:30: from a FastRPC-launched kernel (run_main_on_hexagon) the fopen to
/// kReadoutAccountingPath FAILS silently -- the same launch context that
/// swallows device-side printf -- so an ON arm with HMX_EXEC_ACCT=1 completes
/// with an EMPTY accounting file (g1, 47 us, correct) and the writes cost
/// nothing. The 684 us median was measured in a probe's own process, which has
/// a working file system. Two consequences: (1) the earlier claim that
/// "3 x 684 = 2052 us of accounting I/O explains the 2090 us ON-minus-OFF
/// regression" is wrong for launched kernels -- errata 11 §1's stale-cache
/// attribution stands on its own; (2) this file is NOT a usable diagnostic
/// channel for launched kernels; localize hangs with the barrier shape and
/// per-arm timing instead.
///
/// The accounting stays opt-in anyway (HMX_EXEC_ACCT=1 in the environment that
/// launches the kernel): the atomics accumulate either way, and a context that
/// does have file I/O -- a probe, or a future wrapper that mounts one -- gets
/// the numbers for free.
///
/// The read is done once, on the first call, through a function-local static.
/// getenv itself is not free and this sits on a per-launch path.
static bool acctEnabled() {
  static const bool on = [] {
    const char *v = std::getenv("HMX_EXEC_ACCT");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

void hmxExecTrace(const char *what, uint32_t n, uint64_t c, int rc) {
  if (!acctEnabled())
    return;
  FILE *f = fopen(kReadoutAccountingPath, "a");
  if (!f)
    return;
  fprintf(f, "%s batches=%lu pcyc=%llu rc=%d\n", what, (unsigned long)n,
          (unsigned long long)c, rc);
  fclose(f);
}

void reportReadoutAccounting(Executor &e) {
  const uint32_t n = e.batchesRun.load(std::memory_order_relaxed);
  const uint64_t c = e.readoutPcyc.load(std::memory_order_relaxed);
  hmxExecTrace("drain", n, c, 0);
  // Mechanism counters. These are the ones that decide whether the per-iteration
  // cost is the futex park path or the HVX lock/unlock churn: the consumer parks
  // only when its bounded spin budget runs out (kPollCount iterations, ~= 10.5
  // us -- sized to cover the engine's inter-publish gap, see kPollCount's
  // calibration note), and the HVX context is dropped only on the way into that
  // park, so a steady state that never exhausts the budget shows nPark ~ 0 and
  // nHvxLock ~ 1 per launch. One number per mechanism is all that is needed to
  // tell them apart, and no inference is needed.
  hmxExecTrace("mech", (uint32_t)e.nPublish.load(std::memory_order_relaxed),
               e.nPark.load(std::memory_order_relaxed), 0);
  hmxExecTrace("mech-hvxlock", (uint32_t)e.nHvxLock.load(std::memory_order_relaxed),
               e.nHvxUnlock.load(std::memory_order_relaxed), 0);
  // pcyc->ns needs the core clock; 2112 pcyc/ns was calibrated on this part
  // (exp/hmx/s2_handoff/probe2.cpp) but is deliberately NOT hardcoded as truth
  // here -- the raw pcyc count is what matters for the question being asked, and
  // the ratio against kernel pcyc is clock-independent.
  if (!acctEnabled())
    return;
  FILE *f = fopen(kReadoutAccountingPath, "a");
  if (!f)
    return;
  fprintf(f, "readout batches=%lu pcyc=%llu\n", (unsigned long)n, (unsigned long long)c);
  fclose(f);
}

void drainLocked(Executor &e) {
  // Monotone barrier: wait until the consumer has retired every batch
  // publish() has accepted. This loop's predecessor advanced a ring cursor one
  // slot at a time and waited for `idxRead == next` -- a MODULAR equality.
  // processAvailable retires back-to-back batches without parking, so whenever
  // the consumer retires more than one batch between two of the producer's
  // observations -- at G>=4, where drain is a live barrier, that is the steady
  // state, not an edge case -- its wrapping cursor passes `next` without ever
  // equaling it again, and both sides park forever: the producer is inside
  // drain so it issues no more publishes, and the consumer, fully caught up,
  // has nothing to retire. Observed twice on device (probe3.cpp:289-311),
  // which switched to this monotone form and completed.
  //
  // `retireSeqn` never exceeds `publishedTotal` (the consumer only retires
  // accepted batches), so equality is the exact "all drained" predicate, and
  // it is reachable for as long as the consumer is alive.
  const uint32_t target = e.publishedTotal.load(std::memory_order_acquire);

  // Live-counter ring, part 1: bracket the barrier wait. drainLocked is
  // entered only from the engine thread (configure, drain, shutdown), so the
  // drainDiag slot this writes is single-writer and needs no lock.
  const uint64_t drainEnterPcyc = HAP_perf_get_pcycles();

  // Bounded spin, then park. Same kPollCount and the same shape as the
  // consumer's idle wait. The futex value check closes the lost-wakeup
  // window: any retire after `seen` was sampled changes the word, so the
  // wait returns immediately instead of sleeping. The spin budget is per
  // drain, which is one barrier, so no path here can spin more than
  // kPollCount times before it blocks. The budget (~10.5 us) covers the
  // largest observed drain wait (3.62 us, T6: the consumer's tail batch),
  // so the producer typically spins through the tail instead of parking in
  // the futex.
  unsigned spins = kPollCount;
  while (e.retireSeqn.load(std::memory_order_acquire) != target) {
    if (spins != 0u) {
      --spins;
      continue;
    }
    const uint32_t seen = e.retireSeqn.load(std::memory_order_acquire);
    if (e.retireSeqn.load(std::memory_order_acquire) == target) {
      break;
    }
    qurt_futex_wait(&e.retireSeqn, static_cast<int>(seen));
  }

  // Live-counter ring, part 2. fetch_add returns this call's monotonic
  // sequence number, which is the dump-side index for the slot just written.
  const uint64_t drainSeq = e.nDrain.fetch_add(1u, std::memory_order_relaxed);
  e.drainDiag[drainSeq & kDiagMask].enterPcyc = drainEnterPcyc;
  e.drainDiag[drainSeq & kDiagMask].exitPcyc = HAP_perf_get_pcycles();
}

//===----------------------------------------------------------------------===//
// Consumer side
//===----------------------------------------------------------------------===//

/// Run every batch currently queued, in order.
///
/// One `fn` call per batch, with `count == 1`, even when several are queued.
/// The ring slots are contiguous so a run *could* be passed as one
/// `(first, n)` call, but that only holds while the run does not wrap, which
/// would make the meaning of `count` depend on the producer's write index -- a
/// contract the compiler pass would have to reproduce exactly. A call is a few
/// nanoseconds against a 760 ns read-out, so the simple form wins.
///
/// `holdsHvx` is this thread's HVX context state. It is taken around the whole
/// batch, not per batch, so a full ring costs one acquire, not sixteen.
void processAvailable(Executor &e, bool &holdsHvx) {
  uint32_t ir = e.idxRead.load(std::memory_order_relaxed);
  const uint32_t iw = e.idxWrite.load(std::memory_order_acquire);
  if (ir == iw) {
    return;
  }

  if (!holdsHvx) {
    // The read-out is HVX store code (`hmx.unpack_acc`), so it cannot run
    // without an HVX context. qurt_hvx_lock BLOCKS until a unit in the
    // requested mode is free; it does not spin, which is why it is safe to call
    // here even though everything else in this file is spin-bounded.
    //
    // A non-zero return means QuRT refused, which no bounded retry can fix and
    // which must NOT be papered over. Advancing `idxRead` without running the
    // read-out would let drain() return on time and hand the kernel an
    // unwritten destination, so this aborts instead. The same fail-closed
    // choice HEXAGON_SAFE_CALL makes in include/HexagonCommon.h:28-36.
    //
    // THE SECOND-UNIT ASSUMPTION -- SETTLED BY MEASUREMENT (2026-10-04)
    // ------------------------------------------------------------------
    // include/HmxVectorExecutor.h:33-35 says the engine thread "already holds"
    // the HVX context and that the idle side is therefore this thread. That
    // implies a SECOND 128-byte HVX unit exists: the engine thread holds one
    // implicitly by executing HVX leaves (nothing in this runtime calls
    // qurt_hvx_lock on its behalf) and this thread takes the other. llama.cpp
    // needs the same thing and additionally gives its context back when idle,
    // which is why vectorThreadEntry() releases on the way into a park (and,
    // since the spin-budget rework, holds through the bounded spin before
    // it -- see the park block there).
    //
    // This was once the single assumption in this file that no host-side
    // reasoning could confirm; it is now measured
    // (exp/hmx/t5_hvx_contention): two threads take one 128-byte unit each
    // via qurt_hvx_try_lock (both rc=0) and run at full duty concurrently at
    // 1.969/2.0 throughput with both windows at 2112 pcyc/us -- two real
    // units, not time-slicing. The "blocks forever on a single-unit part"
    // failure mode is therefore retired for this part; it would resurface
    // only on hardware with one unit, where the fix remains what it was: drop
    // the qurt_hvx_lock here and let the two threads share a unit unsafely,
    // which is only correct if the engine thread never runs HVX between a
    // publish and its drain -- a question for the compiler pass as much as
    // for this file.
    e.nHvxLock.fetch_add(1, std::memory_order_relaxed);
      if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) {
      FARF(ERROR, "hmx-exec: qurt_hvx_lock failed; cannot read out accumulators");
      std::abort();
    }
    holdsHvx = true;
  }

  const HmxReadoutFn fn = e.fn.load(std::memory_order_acquire);
  if (fn == nullptr) {
    // Unreachable: publish() refuses to accept anything before configure() has
    // stored a function. Kept as a hard stop rather than a skip, for the same
    // reason as the lock failure above.
    FARF(ERROR, "hmx-exec: queued batch with no read-out function configured");
    std::abort();
  }

  while (ir != iw) {
    const uint64_t fnStart = HAP_perf_get_pcycles();
    fn(&e.slots[ir], 1);
    const uint64_t fnEnd = HAP_perf_get_pcycles();
    e.readoutPcyc.fetch_add(fnEnd - fnStart, std::memory_order_relaxed);
    // Live-counter ring: the fetch_add returns this batch's monotonic
    // sequence number, which is the same index the producer stamped its
    // publishAcceptPcyc under (single consumer, in-order, one ring entry per
    // fn() call -- one per batch publish() accepted).
    const uint32_t seq = e.batchesRun.fetch_add(1u, std::memory_order_relaxed);
    e.diag[seq & kDiagMask].consumerStartPcyc = fnStart;
    e.diag[seq & kDiagMask].consumerEndPcyc = fnEnd;
    ir = (ir + 1u) & kRingMask;
    // Release: everything the read-out wrote to `dst` must be visible to the
    // producer before `idxRead` advertises the slot as finished.
    e.idxRead.store(ir, std::memory_order_release);
    e.retireSeqn.fetch_add(1u, std::memory_order_release);
    qurt_futex_wake(&e.retireSeqn, 1);
  }
}

/// The resident vector thread.
///
/// Structure copied from hmx-queue.c:70-86: compare the sequence counter against
/// what was last seen, do the work when it moved, and otherwise spin a BOUNDED
/// number of times before parking in qurt_futex_wait on the same counter.
void vectorThreadEntry(void *arg) {
  Executor &e = *static_cast<Executor *>(arg);

  // Starts at 0, which is also the initial value of `seqn`, so the first
  // iteration takes the idle path (spin out the budget, then park).
  // Matching hmx-queue.c:70.
  uint32_t prevSeqn = 0;
  unsigned pollCnt = kPollCount;
  bool holdsHvx = false;

  for (;;) {
    const uint32_t seqn = e.seqn.load(std::memory_order_acquire);

    if (seqn != prevSeqn) {
      prevSeqn = seqn;
      pollCnt = kPollCount;
      processAvailable(e, holdsHvx);
      continue;
    }

    if (e.stop.load(std::memory_order_acquire)) {
      break;
    }

    if (pollCnt == 0) {
      // RE-CHECK BEFORE RELEASING OR PARKING. The comment that used to be
      // here claimed the futex value check alone closes the window between
      // the seqn load at the top of the loop and this wait. The device log
      // says otherwise (probe3.cpp:223-244): the last wake of a publish
      // burst can land while this thread is between its seqn load and its
      // park -- consumed by nobody, after which the consumer sleeps on a
      // value the producer has already moved past, and drain (which has no
      // more publishes to offer) waits for a retire that never comes. The
      // explicit re-load closes the window regardless of what
      // qurt_futex_wait does internally. Doing it BEFORE the unlock below
      // also means a batch that arrived during the spin is processed with
      // the context still held -- no unlock/relock pair is paid at all.
      if (e.seqn.load(std::memory_order_acquire) != prevSeqn) {
        pollCnt = kPollCount;
        continue;
      }
      // Release the HVX context ONLY here, on the way into the park.
      //
      // Through the whole bounded spin above the context was HELD on
      // purpose: re-taking it after every wake cost 372 pcyc per park
      // (exp/hmx/t5_hvx_contention Q2), and with the spin budget now
      // covering the inter-publish gap the steady state never reaches this
      // line at all (the re-check above continues instead).
      //
      // HYGIENE vs include/HmxVectorExecutor.h:33-35 (the engine thread
      // already holds a context, so the idle side must be the one giving
      // it back): preserved where it matters -- while PARKED, which is an
      // unbounded idle state, the unit is free. What changed is only the
      // bounded window: during the spin (<= ~10.5 us, kPollCount above) a
      // 128-byte unit stays held by an idle thread.
      //
      // THIRD-PARTY HVX USERS: during that window a third qurt_hvx_lock
      // caller would block for at most the remainder of one spin budget.
      // Two units are proven on this part (T5: two threads, try_lock both
      // rc=0, 1.969/2.0 at full duty), so engine + this thread fit without
      // contending; three or more concurrent users are NOT measured (T5
      // left that open), which is why the hold is bounded and released at
      // the park rather than held for the process lifetime. Power/thermal
      // of the longer spin is likewise unmeasured (llama.cpp's v79 = 1 is
      // the power-saving end of this same trade; see kPollCount above).
      //
      // This is the one place this file departs from hmx-queue.c:76-77,
      // which guards its own qurt_hvx_unlock behind `poll_cnt > 1`. The
      // intent there -- drop the context while idle -- is preserved here
      // at the park, which is the only idle state that can outlive a
      // bounded spin.
      if (holdsHvx) {
        e.nHvxUnlock.fetch_add(1, std::memory_order_relaxed);
        if (qurt_hvx_unlock() != 0) {
          FARF(ERROR, "hmx-exec: qurt_hvx_unlock failed while idle");
        }
        holdsHvx = false;
      }
      e.nPark.fetch_add(1, std::memory_order_relaxed);
      qurt_futex_wait(&e.seqn, static_cast<int>(prevSeqn));
      pollCnt = kPollCount;
      continue;
    }

    // Bounded spin, then park on the next iteration. The budget is
    // kPollCount iterations ~= 10.5 us (see its calibration note): sized to
    // cover the engine's inter-publish gap, so the steady state catches the
    // next batch here instead of reaching the park path below.
    --pollCnt;
  }

  if (holdsHvx) {
    if (qurt_hvx_unlock() != 0) {
      FARF(ERROR, "hmx-exec: qurt_hvx_unlock failed at thread exit");
    }
  }
}

//===----------------------------------------------------------------------===//
// Thread lifecycle
//===----------------------------------------------------------------------===//

/// Create the resident vector thread. Called from `configure` only.
///
/// WHY THE THREAD IS RESIDENT, AND WHY THAT IS THE WHOLE POINT
/// -----------------------------------------------------------
/// Creating a thread measured ~18 us on this device
/// (exp/hmx/s2_handoff/probe2.cpp measures it; the number the split depends on
/// is quoted in include/HmxVectorExecutor.h:100-101). The read-out being hidden
/// is 24.3 us. So a thread created per launch spends roughly three quarters of
/// the entire budget on thread creation before reading out a single accumulator
/// row, and the split is a net loss no matter how cheap the handoff is.
/// Therefore: create once here, reuse for every subsequent launch, and let only
/// `shutdown` destroy it. Nothing in this file may be moved onto a per-launch
/// path.
///
/// Returns 0 on success, non-zero if the thread could not be created.
int startVectorThread(Executor &e) {
  // qurt_thread_attr_set_stack_addr requires an 8-byte aligned address
  // (qurt_thread.h:471). malloc satisfies it on this target, but allocating the
  // slack and rounding up keeps that an observation rather than an assumption.
  constexpr size_t kAlign = 8;
  void *raw = std::malloc(kStackBytes + kAlign);
  if (raw == nullptr) {
    return -1;
  }
  const uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(raw) + (kAlign - 1)) & ~(kAlign - 1);
  void *stack = reinterpret_cast<void *>(aligned);

  // Match the calling thread's priority, clamped exactly as hmx-queue.c:131-138
  // does. A vector thread that outranks the engine thread would steal the
  // pipeline-depth it is supposed to be hiding behind; one that is starved
  // turns drain() into a serialisation.
  int prio = qurt_thread_get_priority(qurt_thread_get_id());
  if (prio < 1) {
    prio = 1;
  }
  if (prio > kLowestPriority) {
    prio = kLowestPriority;
  }

  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "hmx-vec");
  qurt_thread_attr_set_priority(&attr, static_cast<unsigned short>(prio));
  qurt_thread_attr_set_stack_size(&attr, static_cast<unsigned>(kStackBytes));
  qurt_thread_attr_set_stack_addr(&attr, stack);

  // Published before the create, not after: qurt_thread_create starts the thread
  // the moment it returns, and `shutdown` frees `e.stack`. Recording it first
  // means the pointer is never lost in the window between the two. Both callers
  // are on the engine thread (single producer), so the residual window before
  // `started` is published is not reachable.
  e.stack = raw;

  const int err = qurt_thread_create(&e.thread, &attr, vectorThreadEntry, &e);
  if (err != 0) {
    FARF(ERROR, "hmx-exec: qurt_thread_create failed (%d)", err);
    // The stack belongs to us; no thread was created to have referenced it.
    e.stack = nullptr;
    std::free(raw);
    return -1;
  }

  e.started.store(true, std::memory_order_release);
  FARF(ALWAYS, "hmx-exec: resident vector thread up, prio %d", prio);
  return 0;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public ABI
//===----------------------------------------------------------------------===//
//
// These are the only symbols in this file that are visible outside it. The
// enclosing library is compiled with -fvisibility=hidden (it arrives from the
// parent Triton CMake scope, not from bin/runtime/multithreading/CMakeLists.txt),
// so the attribute is what puts them in the archive's symbol table -- the same
// mechanism multithreading/AsyncRuntime.h:33 and
// multithreading/HexagonCustomDefs.cpp:93 use. There is no export list, no
// registration table and no dlsym handshake in this runtime to register with.

extern "C" {

/// Every entry below carries __attribute__((visibility("default"))).
///
/// The enclosing library is compiled with -fvisibility=hidden, which arrives
/// from the parent Triton CMake scope and not from
/// multithreading/CMakeLists.txt, so without the attribute all four land in the
/// archive as GLOBAL HIDDEN and a kernel module that calls them does not link.
/// The attribute is the whole export mechanism here: this runtime has no export
/// list, no registration table and no dlsym handshake, and the existing entries
/// use exactly this escape hatch (multithreading/AsyncRuntime.h:33 and
/// multithreading/HexagonCustomDefs.cpp:93). It cannot go on the declarations in
/// include/HmxVectorExecutor.h without editing that header, and it does not need
/// to -- default visibility on the definition is what the linker records.
///
/// Nothing else in this file is visible outside it: everything below the
/// anonymous namespace, including the Executor state, is LOCAL in the object
/// file.

/// Return values:
///    0  the function is registered and the vector thread is running
///   -1  `fn` was null
///   -2  `numThreads` was outside [1, kVectorThreads]; one thread was used
///   -3  the thread could not be created
///
/// `numThreads` is honoured only up to one. The ABI carries it, but this
/// executor is a single-producer ring feeding a single resident thread, which
/// is what the two-thread split is defined as; a second vector thread would need
/// an MPSC ring and a work split the descriptor does not carry (each batch is
/// already a whole read-out). Asking for more logs once and continues with one,
/// rather than accepting a value this implementation cannot honour.
__attribute__((visibility("default"))) int32_t
hexagon_runtime_hmx_exec_configure(HmxReadoutFn fn, int32_t numThreads) {
  hmxExecTrace("configure", 0, 0, 0);
  if (fn == nullptr) {
    return -1;
  }

  int32_t status = 0;
  if (numThreads < 1) {
    numThreads = 1;
  }
  if (numThreads != kVectorThreads) {
    // ERROR level, not a warning: the request is not being honoured, and a
    // caller that assumed two vector threads would be silently wrong. Logged
    // once per process so a caller that passes a constant cannot flood the log.
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
      FARF(ERROR, "hmx-exec: numThreads=%d requested, using %d; this executor "
                  "runs a single vector thread",
           numThreads, kVectorThreads);
    }
    status = -2;
  }

  Executor &e = gExecutor;

  // Quiesce before swapping the function. A recompile re-enters here with a new
  // pointer while the resident thread is still alive, so the swap has to wait
  // for any in-flight read-out to finish: the consumer loads `fn` per queue
  // drain, and a pointer it read one instruction before the swap would run the
  // PREVIOUS kernel's read-out against this kernel's descriptors. drainLocked()
  // is exactly that wait, and it is a no-op before the first publish.
  drainLocked(e);

  // Idempotent: a repeated configure with the same function just re-stores it,
  // which the resident thread picks up on its next pass. The thread is started
  // only once per process lifetime; `started` is the guard, not `thread`, so a
  // recycled qurt_thread_t cannot be mistaken for a live thread.
  e.fn.store(fn, std::memory_order_release);

  if (!e.started.load(std::memory_order_acquire)) {
    // A shutdown earlier in this process left `stop` set, and a fresh thread
    // would read it on its first idle check and exit before doing any work. Clear
    // it before the create, not after: the thread starts running as soon as
    // qurt_thread_create returns, so a clear afterwards would be a race.
    e.stop.store(false, std::memory_order_release);
    if (startVectorThread(e) != 0) {
      e.fn.store(nullptr, std::memory_order_release);
      return -3;
    }
  }

  return status;
}

/// Hand `count` batches to the vector thread and report how many it took.
///
/// Never blocks and never waits on the consumer. The only thing on this path
/// that can cost time is the ring stores and one futex wake.
///
/// A short return means the ring was full, i.e. the vector side cannot keep up.
/// The tail is dropped -- the first `freeSlots` batches are the ones taken --
/// because blocking here would stall the engine thread and lose the overlap for
/// every batch behind the one being waited on.
///
/// THE CALLER OWNS THE DROPPED BATCHES. `dst` is caller memory that only the
/// read-out function writes, so a dropped batch is a destination region that is
/// never written and the kernel returns whatever was there before. That is not
/// a degraded result, it is a wrong one, so a caller that ignores the return
/// value has a latent silent-corruption bug whenever the ring fills. It is also
/// why the drop is logged at ERROR rather than folded away quietly: the ring is
/// 15 deep and the engine outruns the reader only if something is already
/// wrong, so this should be rare enough to see.
__attribute__((visibility("default"))) uint32_t
hexagon_runtime_hmx_exec_publish(const HmxReadoutBatch *batches, uint32_t count) {
  Executor &e = gExecutor;
  e.nPublish.fetch_add(1, std::memory_order_relaxed);
  if (static int once = 0; !once) { once = 1; hmxExecTrace("publish-first", 0, 0, (int)count); }
  if (batches == nullptr || count == 0u) {
    return 0;
  }

  if (!e.started.load(std::memory_order_acquire) ||
      e.fn.load(std::memory_order_acquire) == nullptr) {
    // Not configured: accept nothing rather than queue work the resident thread
    // would run against a stale function pointer. Reported once so a kernel that
    // never calls configure() is obvious in the log instead of silent.
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
      FARF(ERROR, "hmx-exec: publish before configure; dropping %u batch(es)",
           count);
    }
    return 0;
  }
  if (e.stop.load(std::memory_order_acquire)) {
    return 0;
  }

  // Producer-private, so relaxed is enough.
  const uint32_t iw = e.idxWrite.load(std::memory_order_relaxed);
  const uint32_t ir = e.idxRead.load(std::memory_order_acquire);

  // Occupancy is how far the writer is ahead of the reader. The producer holds
  // one slot back at all times so that "empty" and "full" are different values
  // of the same two indices (hmx-queue.h:75 tests `((iw + 1) & idx_mask) ==
  // ir`, which is exactly occupancy == capacity - 1). Without that the ring
  // cannot distinguish a full ring from an empty one and drain() would either
  // hang or return early.
  const uint32_t occupancy = (iw - ir) & kRingMask;
  constexpr uint32_t kUsable = kRingCapacity - 1u;
  if (occupancy >= kUsable) {
    FARF(ERROR, "hmx-exec: ring full, dropping %u batch(es)", count);
    return 0;
  }

  const uint32_t freeSlots = kUsable - occupancy;
  const uint32_t accepted = count < freeSlots ? count : freeSlots;
  for (uint32_t i = 0; i < accepted; ++i) {
    e.slots[(iw + i) & kRingMask] = batches[i];
  }
  // Live-counter ring: stamp every accepted batch with the publish time.
  // BEFORE the idxWrite release store below, so the consumer -- which can
  // only see a batch after acquiring idxWrite -- can never start a batch
  // whose publishAcceptPcyc is not yet in place. Wrap safety: see the
  // comment on Executor::diag. publishedTotal is loaded relaxed because the
  // producer is its only writer; the value here is the sequence number of
  // the first batch this call is accepting.
  if (accepted != 0u) {
    const uint64_t acceptPcyc = HAP_perf_get_pcycles();
    const uint32_t baseSeq = e.publishedTotal.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < accepted; ++i) {
      e.diag[(baseSeq + i) & kDiagMask].publishAcceptPcyc = acceptPcyc;
    }
  }
  // Release: the slot stores must be visible to the consumer before it can
  // observe the new write index.
  e.idxWrite.store((iw + accepted) & kRingMask, std::memory_order_release);
  // Monotone twin of the modular idxWrite above, in batch counts: what the
  // drain barrier waits for retireSeqn to reach. Release so the slot stores
  // are ordered before it for anyone who loads it acquire.
  e.publishedTotal.fetch_add(accepted, std::memory_order_release);

  // Bump then wake, so a consumer already awake re-reads `seqn` and finds it
  // moved rather than parking on a value that changed under it.
  e.seqn.fetch_add(1u, std::memory_order_release);
  qurt_futex_wake(&e.seqn, 1);

  if (accepted < count) {
    FARF(ERROR, "hmx-exec: ring full, accepted %u of %u batch(es); the caller "
                "must read out the tail itself",
         accepted, count);
  }
  return accepted;
}

/// Block until every published batch has been read out. See drainLocked().
__attribute__((visibility("default"))) void hexagon_runtime_hmx_exec_drain(void) {
  drainLocked(gExecutor);
  reportReadoutAccounting(gExecutor);
}

/// Release the vector thread. Session teardown only -- see the residency note
/// on startVectorThread() for why this must never run per launch.

__attribute__((visibility("default"))) void hexagon_runtime_hmx_exec_shutdown(void) {
  Executor &e = gExecutor;
  if (!e.started.load(std::memory_order_acquire)) {
    return;
  }

  // Retire anything still queued before asking the thread to leave, so shutdown
  // cannot race a read-out that is writing into caller memory.
  drainLocked(e);

  e.stop.store(true, std::memory_order_release);
  e.seqn.fetch_add(1u, std::memory_order_release);
  qurt_futex_wake(&e.seqn, 1);

  int status = 0;
  qurt_thread_join(e.thread, &status);

  // The thread has been reaped by join, so nothing can be walking the stack any
  // more and it is safe to hand the pages back.
  std::free(e.stack);
  e.stack = nullptr;
  e.thread = qurt_thread_t{};
  e.started.store(false, std::memory_order_release);

  // Left in place: idxWrite/idxRead are both 0 after the drain above, so a
  // later configure() reuses a clean ring without reinitialising anything, and
  // `fn` stays until the next configure so a publish after shutdown is still
  // refused on `stopped`, not on a null function.
}

/// Append the accumulated live counters and the two diagnostic rings to
/// `path`. This is the read-back path for the rings in Executor: the wrapper
/// calls it AFTER the timed region (see the benchmarking template in
/// backend/hexagon_launcher_base.py), never from inside a kernel's timed
/// path, so the file I/O cannot perturb Perf.
///
/// Not in the frozen ABI header, deliberately: the wrapper declares this
/// symbol WEAK, so kernels that do not link the async runtime's
/// HmxVectorExecutor object (every non-readout kernel) resolve it to null and
/// skip the call, while read-out kernels -- which reference the executor ABI
/// strongly from their LLVM object, pulling this member into their .so --
/// resolve it at static link time. Default visibility on the definition is
/// the whole export mechanism, exactly as for the four entries above.
///
/// The first block also adjudicates, in one shot, why the executor's own
/// accounting file came back empty from a FastRPC-launched kernel (errata 12
/// section 5.1): it reports getenv("HMX_EXEC_ACCT") -- whether the variable
/// the host set ever reached this DSP process -- AND probes fopen of the
/// accounting path from this same process, so "the environment never
/// arrived" and "fopen fails here" are separated by evidence rather than by
/// inference. An empty file with env="(null)" has nothing to do with fopen.
__attribute__((visibility("default"))) void
hexagon_runtime_hmx_exec_dump(const char *path) {
  if (path == nullptr) {
    return;
  }
  Executor &e = gExecutor;
  FILE *f = fopen(path, "a");
  if (!f) {
    return;
  }

  const uint64_t nBatches = e.batchesRun.load(std::memory_order_relaxed);
  const uint64_t nDrainTotal = e.nDrain.load(std::memory_order_relaxed);

  fprintf(f, "HMXEXECDUMP begin\n");
  // One-shot adjudication (see the function comment). The probe writes to
  // the accounting path itself, so a SUCCESS is independently verifiable by
  // pulling that file after the run.
  {
    const char *env = std::getenv("HMX_EXEC_ACCT");
    fprintf(f, "HMXEXECDUMP env HMX_EXEC_ACCT=%s\n",
            env != nullptr ? env : "(null)");
    FILE *probe = fopen(kReadoutAccountingPath, "a");
    if (probe != nullptr) {
      fprintf(probe, "dump probe: fopen of the accounting path succeeded\n");
      fclose(probe);
      fprintf(f, "HMXEXECDUMP acctpath fopen=ok\n");
    } else {
      fprintf(f, "HMXEXECDUMP acctpath fopen=FAIL errno=%d\n", errno);
    }
  }
  fprintf(f,
          "HMXEXECDUMP counters batchesRun=%llu readoutPcyc=%llu "
          "publishedTotal=%llu retireSeqn=%llu nPublish=%llu nPark=%llu "
          "nHvxLock=%llu nHvxUnlock=%llu nDrain=%llu\n",
          (unsigned long long)nBatches,
          (unsigned long long)e.readoutPcyc.load(std::memory_order_relaxed),
          (unsigned long long)e.publishedTotal.load(std::memory_order_relaxed),
          (unsigned long long)e.retireSeqn.load(std::memory_order_relaxed),
          (unsigned long long)e.nPublish.load(std::memory_order_relaxed),
          (unsigned long long)e.nPark.load(std::memory_order_relaxed),
          (unsigned long long)e.nHvxLock.load(std::memory_order_relaxed),
          (unsigned long long)e.nHvxUnlock.load(std::memory_order_relaxed),
          (unsigned long long)nDrainTotal);

  if (nBatches == 0u) {
    // The never-configured / never-published case (e.g. a kernel that does
    // not use the read-out split but still links this object).
    fprintf(f, "HMXEXECDUMP no batches\n");
  } else {
    const uint64_t first = nBatches > kDiagCapacity
                               ? nBatches - kDiagCapacity
                               : 0;
    for (uint64_t s = first; s < nBatches; ++s) {
      const Executor::BatchDiag &d = e.diag[s & kDiagMask];
      fprintf(f,
              "HMXEXECDUMP batch seq=%llu pub=%llu cstart=%llu cend=%llu\n",
              (unsigned long long)s,
              (unsigned long long)d.publishAcceptPcyc,
              (unsigned long long)d.consumerStartPcyc,
              (unsigned long long)d.consumerEndPcyc);
    }
  }
  if (nDrainTotal != 0u) {
    const uint64_t first = nDrainTotal > kDiagCapacity
                               ? nDrainTotal - kDiagCapacity
                               : 0;
    for (uint64_t s = first; s < nDrainTotal; ++s) {
      const Executor::DrainDiag &d = e.drainDiag[s & kDiagMask];
      fprintf(f, "HMXEXECDUMP drain seq=%llu enter=%llu exit=%llu\n",
              (unsigned long long)s, (unsigned long long)d.enterPcyc,
              (unsigned long long)d.exitPcyc);
    }
  }
  fprintf(f, "HMXEXECDUMP end\n");
  fclose(f);
}

} // extern "C"
