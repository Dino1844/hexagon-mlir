/* Test-only protocol driver: runs the REAL HmxRoleExecutor.cpp on the host.
 *
 * Consumed by: test_hmx_spsc_ring_contract.py, which compiles this file with
 * the stub SDK headers in this directory first on the include path. The
 * production file bin/runtime/multithreading/HmxRoleExecutor.cpp is pulled in
 * verbatim below -- its ring protocol, its thread loop, its futex parks, its
 * bind/submit/drain/join entries are the code under test; this driver
 * contains no executor logic whatsoever. Same pattern as
 * hmx_layout_host_emu/driver.c, which compiles the real HMXLayout.c.
 *
 * What the stubs cannot emulate (honest limits, each with its owner):
 *   - The HMX lock. The stub ensure() below records its calls and returns;
 *     it cannot model "a second ensure from another thread blocks forever".
 *     The single-holder property is the lock-lifetime device probe's job
 *     (exp/hmx/s2_runtime_probes/lock_lifetime).
 *   - Thread priority and the malloc'd DDR stack (see qurt_thread.h stub).
 *   - The 2^32 counter wrap of the ring's own monotone indices at speed; the
 *     driver can only PRESET the counters next to the wrap boundary (the
 *     `rpreset` command) and push a handful of descriptors across it.
 *
 * The wire protocol (one command per line, EXACTLY ONE reply line per
 * command, so the Python side never has to guess where a reply ends):
 *
 *   > fn <name>                     select the section for the next bind
 *   > bind <depth>                  -> BIND <rc>
 *   > submit <n> <slot> <rowStart> <rowCount> <eventLo> <eventHi> ...x n
 *                                   -> SUBMIT <accepted>
 *   > drain                         -> DRAIN ok
 *   > wait <target>                 -> WAIT ok           (granular barrier)
 *   > join                          -> JOIN ok
 *   > poke <slot> <value>           -> POKE ok | POKE VIOLATION
 *   > poke-force <slot> <value>     -> POKE-FORCE ok     (negative-control bypass)
 *   > shadow <slot>                 -> SHADOW <slot> <value>
 *   > probe                         -> PROBE k=v ...
 *   > fnlog                         -> FNLOG <n>|entry;entry;...
 *   > violations                    -> VIOLATIONS <n>|kind,slot,owner;...
 *   > ring <capacity>               -> RING ok           (direct ring battery)
 *   > rpush <slot> <rowStart> <rowCount> <eventLo> <eventHi> -> RPUSH <1|0>
 *   > rfree                         -> RFREE <n>
 *   > rpend                         -> RPEND <0|1>
 *   > rpeek                         -> RPEEK <slot> <rowStart> <rowCount> <evLo> <evHi>
 *   > rretire                       -> RRETIRE ok
 *   > rtarget                       -> RTARGET <head>
 *   > rdrained <target>             -> RDRAINED <0|1>
 *   > rreset                        -> RRESET ok
 *   > rpreset <value>               -> RPRESET ok        (test-only counter warp)
 *   > quit
 *
 * OWNERSHIP SHADOW (what "double-write detection" concretely means here):
 * the descriptor's `slot` names a driver-owned shadow word standing in for
 * the compiler-allocated VTCM slot. The producer writes it through `poke`,
 * the bound section writes it through the guarded helper; each write checks
 * the writer owns the slot at that moment (producer: slot not in flight;
 * consumer: slot is the descriptor's own). Violations are recorded, never
 * silently absorbed, and the battery's negative controls prove the guards
 * fire. `poke-force` bypasses the producer guard on purpose: it simulates a
 * protocol-violating second write so the section's read-validation detector
 * is proven non-vacuous.
 */
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "qurt_thread.h" /* stubs win the include race: -I<this dir> first */
#include "qurt_futex.h"
#include "HAP_farf.h"

// ---------------------------------------------------------------------------
// The HMX-lock recorder: the host stand-in for the runtime's ensure entry.
// The production TU declares this symbol extern and calls it once at bound
// thread start; here it only records (calls, thread, sequence) so the test
// can prove: called exactly once, on the thread that was created, before any
// section ran. It CANNOT model lock blocking -- see the file header.
// ---------------------------------------------------------------------------
static std::atomic<uint64_t> gSeq{0}; // global event order: ensure vs sections
static std::atomic<uint64_t> gEnsureCalls{0};
static std::thread::id gEnsureThread{};
static uint64_t gEnsureSeq = 0;

extern "C" void hexagon_runtime_hmx_ensure_dsp(void) {
  gEnsureThread = std::this_thread::get_id();
  gEnsureSeq = gSeq.fetch_add(1, std::memory_order_relaxed) + 1;
  gEnsureCalls.fetch_add(1, std::memory_order_relaxed);
}

// The production executor, verbatim. Everything above this line is test
// scaffolding; everything the tests exercise is below it.
#include "../../multithreading/HmxRoleExecutor.cpp"

// ---------------------------------------------------------------------------
// Ownership shadow + guards
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kShadowSlots = 256;
constexpr uint32_t kMarkerBase = 0xC0DE0000u;

struct Violation {
  std::string kind;
  uint32_t slot;
  uint32_t owner; // -1 encoded as 0xFFFFFFFF for "no owner"
};

struct ShadowState {
  std::mutex mu;
  uint32_t shadow[kShadowSlots] = {0};
  std::map<uint32_t, uint32_t> lastPoke;  // slot -> producer value (legal poke)
  std::map<uint32_t, uint32_t> expected;  // slot -> value at submit time
  std::map<uint32_t, bool> inflight;      // slot -> published, not retired
  std::vector<Violation> violations;

  void record(const char *kind, uint32_t slot, uint32_t owner) {
    violations.push_back(Violation{kind, slot, owner});
  }
};

ShadowState gShadow;

struct FnLogEntry {
  uint64_t seq;
  std::string thread;
  uint32_t slot;
  uint32_t rowStart;
  uint32_t rowCount;
  uint64_t event;
  uint32_t readValue;
  uint32_t wroteValue;
};

struct FnState {
  std::mutex mu;
  std::vector<FnLogEntry> log;
};

FnState gFnLog;

std::string threadName() {
  std::ostringstream os;
  os << std::this_thread::get_id();
  return os.str();
}

// Guarded consumer-side write: the section may write only the slot its own
// descriptor names. A cross-slot write is recorded and REFUSED (recording is
// the signal; writing anyway would corrupt the battery's accounting).
void consumerWrite(uint32_t slot, uint32_t value, uint32_t ownerSlot) {
  if (slot != ownerSlot) {
    std::lock_guard<std::mutex> lk(gShadow.mu);
    gShadow.record("cross-write", slot, ownerSlot);
    return;
  }
  if (slot >= kShadowSlots) {
    std::lock_guard<std::mutex> lk(gShadow.mu);
    gShadow.record("out-of-range", slot, ownerSlot);
    return;
  }
  std::lock_guard<std::mutex> lk(gShadow.mu);
  gShadow.shadow[slot] = value;
}

// The common section body: read the poked value (validating it against what
// the producer wrote at submit time), write the per-invocation marker, log.
void sectionCommon(const HmxTileGroupDesc *groups, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    const HmxTileGroupDesc &d = groups[i];
    if (d.slot >= kShadowSlots) {
      std::lock_guard<std::mutex> lk(gShadow.mu);
      gShadow.record("out-of-range", d.slot, 0xFFFFFFFFu);
      continue;
    }
    uint32_t readValue = 0;
    {
      std::lock_guard<std::mutex> lk(gShadow.mu);
      readValue = gShadow.shadow[d.slot];
      auto it = gShadow.expected.find(d.slot);
      if (it != gShadow.expected.end() && it->second != readValue) {
        // The producer wrote after publish (see poke-force): the value the
        // section reads is not the one it was handed.
        gShadow.record("read-mismatch", d.slot, 0xFFFFFFFFu);
      }
    }
    const uint64_t seq =
        gSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint32_t marker = kMarkerBase | static_cast<uint32_t>(seq & 0xFFFFu);
    consumerWrite(d.slot, marker, d.slot);
    {
      std::lock_guard<std::mutex> lk(gFnLog.mu);
      gFnLog.log.push_back(FnLogEntry{seq, threadName(), d.slot, d.rowStart,
                                      d.rowCount, d.event, readValue,
                                      marker});
    }
  }
}

void sectionProbe(const HmxTileGroupDesc *g, uint32_t n) {
  sectionCommon(g, n);
}

void sectionSlow(const HmxTileGroupDesc *g, uint32_t n) {
  sectionCommon(g, n);
  // 20 ms: the timing proofs below compare this sleep against the submit
  // loop (microseconds) and against a pipe round-trip (tens of
  // microseconds). 20 ms keeps both comparisons safe even when the host is
  // loaded and the main thread is preempted mid-loop: the consumer's first
  // retire cannot land inside the submit's publish window, and a poke-force
  // issued right after submit is always in place before a LATER group's
  // slot is read. Deliberately NOT held under any lock.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

void sectionRogue(const HmxTileGroupDesc *g, uint32_t n) {
  sectionCommon(g, n);
  // Negative control: deliberately write a slot this descriptor does not
  // own. The guard must record a cross-write violation; the battery asserts
  // it fired (a detector that never fires is vacuous).
  if (n > 0 && g[0].slot < kShadowSlots) {
    consumerWrite((g[0].slot + 1) % kShadowSlots, 0xBEEFu, g[0].slot);
  }
}

HmxSectionFn selectedFn(const std::string &name) {
  if (name == "probe")
    return sectionProbe;
  if (name == "slow")
    return sectionSlow;
  if (name == "rogue")
    return sectionRogue;
  return nullptr; // "none": bind must refuse with ERR_NULL_FN
}

// ---------------------------------------------------------------------------
// Direct ring battery state (main thread only; the bound thread never sees it)
// ---------------------------------------------------------------------------
constexpr uint32_t kRingMaxCapacity = 64;
HmxTileGroupDesc gRingSlots[kRingMaxCapacity];
HmxSpscRing gRing;
uint32_t gRingConsumed = 0;

// ---------------------------------------------------------------------------
// Protocol helpers
// ---------------------------------------------------------------------------
[[noreturn]] void protoError(const std::string &msg) {
  std::fprintf(stdout, "ERR %s\n", msg.c_str());
  std::fflush(stdout);
  std::exit(2);
}

uint64_t nextUint(const char *&p, const char *what) {
  while (*p == ' ')
    ++p;
  if (*p == '\0')
    protoError(std::string("missing ") + what);
  char *end = nullptr;
  const unsigned long long v = std::strtoull(p, &end, 10);
  if (end == p)
    protoError(std::string("bad ") + what);
  p = end;
  return v;
}

void markInflight(const HmxTileGroupDesc &d) {
  if (d.slot >= kShadowSlots)
    return;
  std::lock_guard<std::mutex> lk(gShadow.mu);
  gShadow.inflight[d.slot] = true;
  auto poke = gShadow.lastPoke.find(d.slot);
  gShadow.expected[d.slot] =
      poke != gShadow.lastPoke.end() ? poke->second : gShadow.shadow[d.slot];
}

void clearInflight() {
  std::lock_guard<std::mutex> lk(gShadow.mu);
  gShadow.inflight.clear();
  gShadow.expected.clear();
}

// The slots of the first `target` accepted groups, in submission (and hence
// retirement) order. The ring retires FIFO, so the front of the order vector
// is exactly the group the retire counter just passed -- this is the driver's
// model of what wait_retired's target means for slot ownership.
std::vector<uint32_t> gSubmitOrder;

void retireThrough(uint32_t target) {
  std::lock_guard<std::mutex> lk(gShadow.mu);
  size_t done = std::min<size_t>(gSubmitOrder.size(), target);
  for (size_t i = 0; i < done; ++i) {
    gShadow.inflight.erase(gSubmitOrder[i]);
    gShadow.expected.erase(gSubmitOrder[i]);
  }
  gSubmitOrder.erase(gSubmitOrder.begin(),
                     gSubmitOrder.begin() + static_cast<long>(done));
}

} // namespace

int main() {
  std::fprintf(stdout, "role-driver ready\n");
  std::fflush(stdout);

  std::string fnName = "none";
  char line[65536];
  while (std::fgets(line, sizeof(line), stdin)) {
    std::string reply;
    const char *p = line;
    while (*p == ' ')
      ++p;
    if (std::strncmp(p, "quit", 4) == 0) {
      // _Exit, not return: a session that never joined leaves the bound
      // thread running, and static destructors (the state mutexes below)
      // would race it during a normal exit. No destructor runs here.
      std::_Exit(0);
    } else if (std::strncmp(p, "fn ", 3) == 0) {
      p += 3;
      fnName = p;
      while (!fnName.empty() &&
             (fnName.back() == '\n' || fnName.back() == ' '))
        fnName.pop_back();
      reply = "FN " + fnName;
    } else if (std::strncmp(p, "bind ", 5) == 0) {
      p += 5;
      const uint64_t depth = nextUint(p, "depth");
      reply = "BIND " +
              std::to_string(hexagon_runtime_hmx_role_bind(
                  selectedFn(fnName), static_cast<uint32_t>(depth)));
    } else if (std::strncmp(p, "submit ", 7) == 0) {
      p += 7;
      const uint64_t n = nextUint(p, "count");
      if (n > 256)
        protoError("submit count too large");
      static HmxTileGroupDesc groups[256];
      for (uint64_t i = 0; i < n; ++i) {
        groups[i].slot = static_cast<uint32_t>(nextUint(p, "slot"));
        groups[i].rowStart = static_cast<uint32_t>(nextUint(p, "rowStart"));
        groups[i].rowCount = static_cast<uint32_t>(nextUint(p, "rowCount"));
        const uint64_t lo = nextUint(p, "eventLo");
        const uint64_t hi = nextUint(p, "eventHi");
        groups[i].event = (hi << 32) | (lo & 0xFFFFFFFFull);
      }
      const uint32_t accepted =
          hexagon_runtime_hmx_role_submit(groups, static_cast<uint32_t>(n));
      for (uint32_t i = 0; i < accepted; ++i) {
        markInflight(groups[i]);
        // Ownership order tracking for `wait`: the ring retires FIFO, so the
        // accepted prefix of this submit continues the retirement order.
        if (groups[i].slot < kShadowSlots)
          gSubmitOrder.push_back(groups[i].slot);
      }
      reply = "SUBMIT " + std::to_string(accepted);
    } else if (std::strncmp(p, "drain", 5) == 0) {
      hexagon_runtime_hmx_role_drain();
      clearInflight();
      gSubmitOrder.clear();
      reply = "DRAIN ok";
    } else if (std::strncmp(p, "wait ", 5) == 0) {
      p += 5;
      const uint64_t target = nextUint(p, "target");
      if (target > 0xFFFFFFFFull)
        protoError("wait target out of range");
      // Sample the ownership effect BEFORE the wait: the wait's own semantics
      // (retire counter >= target) are what make retireThrough's model true,
      // and the order here is the same order the runtime establishes.
      hexagon_runtime_hmx_role_wait_retired(
          static_cast<uint32_t>(target));
      retireThrough(static_cast<uint32_t>(target));
      reply = "WAIT ok";
    } else if (std::strncmp(p, "join", 4) == 0) {
      hexagon_runtime_hmx_role_join();
      clearInflight();
      gSubmitOrder.clear();
      reply = "JOIN ok";
    } else if (std::strncmp(p, "poke-force ", 11) == 0) {
      p += 11;
      const uint64_t slot = nextUint(p, "slot");
      const uint64_t value = nextUint(p, "value");
      if (slot >= kShadowSlots)
        protoError("poke slot out of range");
      // Bypasses the producer guard AND leaves lastPoke alone: the section's
      // read-validation must then fire (read-mismatch). Negative control.
      std::lock_guard<std::mutex> lk(gShadow.mu);
      gShadow.shadow[slot] = static_cast<uint32_t>(value);
      reply = "POKE-FORCE ok";
    } else if (std::strncmp(p, "poke ", 5) == 0) {
      p += 5;
      const uint64_t slot = nextUint(p, "slot");
      const uint64_t value = nextUint(p, "value");
      if (slot >= kShadowSlots)
        protoError("poke slot out of range");
      std::lock_guard<std::mutex> lk(gShadow.mu);
      if (!gShadow.inflight.empty() && gShadow.inflight.count(
                                            static_cast<uint32_t>(slot))) {
        gShadow.record("poke-inflight", static_cast<uint32_t>(slot),
                       0xFFFFFFFFu);
        reply = "POKE VIOLATION";
      } else {
        gShadow.shadow[slot] = static_cast<uint32_t>(value);
        gShadow.lastPoke[slot] = static_cast<uint32_t>(value);
        reply = "POKE ok";
      }
    } else if (std::strncmp(p, "shadow ", 7) == 0) {
      p += 7;
      const uint64_t slot = nextUint(p, "slot");
      if (slot >= kShadowSlots)
        protoError("shadow slot out of range");
      std::lock_guard<std::mutex> lk(gShadow.mu);
      reply = "SHADOW " + std::to_string(slot) + " " +
              std::to_string(gShadow.shadow[slot]);
    } else if (std::strncmp(p, "probe", 5) == 0) {
      const auto &created = qurt_emu_created_thread_ids();
      std::string bound = "-";
      if (!created.empty()) {
        std::ostringstream os;
        os << created.back();
        bound = os.str();
      }
      std::ostringstream os;
      os << "PROBE"
         << " ensureCalls=" << gEnsureCalls.load(std::memory_order_relaxed)
         << " ensureSeq=" << gEnsureSeq << " ensureThread=" << gEnsureThread
         << " boundThread=" << bound
         << " createdThreads=" << created.size()
         << " futexWaits=" << qurt_emu_futex_waits.load(std::memory_order_relaxed)
         << " futexWakes=" << qurt_emu_futex_wakes.load(std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lk(gShadow.mu);
        os << " violations=" << gShadow.violations.size();
      }
      reply = os.str();
    } else if (std::strncmp(p, "fnlog", 5) == 0) {
      std::ostringstream os;
      {
        std::lock_guard<std::mutex> lk(gFnLog.mu);
        os << "FNLOG " << gFnLog.log.size() << "|";
        for (size_t i = 0; i < gFnLog.log.size(); ++i) {
          const FnLogEntry &e = gFnLog.log[i];
          if (i != 0)
            os << ";";
          os << e.seq << "," << e.thread << "," << e.slot << ","
             << e.rowStart << "," << e.rowCount << "," << (e.event & 0xFFFFFFFFull)
             << "," << (e.event >> 32) << "," << e.readValue << ","
             << e.wroteValue;
        }
      }
      reply = os.str();
    } else if (std::strncmp(p, "violations", 10) == 0) {
      std::ostringstream os;
      {
        std::lock_guard<std::mutex> lk(gShadow.mu);
        os << "VIOLATIONS " << gShadow.violations.size() << "|";
        for (size_t i = 0; i < gShadow.violations.size(); ++i) {
          if (i != 0)
            os << ";";
          os << gShadow.violations[i].kind << ","
             << gShadow.violations[i].slot << ","
             << gShadow.violations[i].owner;
        }
      }
      reply = os.str();
    } else if (std::strncmp(p, "ring ", 5) == 0) {
      p += 5;
      const uint64_t cap = nextUint(p, "capacity");
      if (cap == 0 || cap > kRingMaxCapacity)
        protoError("ring capacity out of range");
      gRing.slots = gRingSlots;
      gRing.capacity = static_cast<uint32_t>(cap);
      gRing.reset();
      gRingConsumed = 0;
      reply = "RING ok";
    } else if (std::strncmp(p, "rpush ", 6) == 0) {
      p += 6;
      HmxTileGroupDesc d;
      d.slot = static_cast<uint32_t>(nextUint(p, "slot"));
      d.rowStart = static_cast<uint32_t>(nextUint(p, "rowStart"));
      d.rowCount = static_cast<uint32_t>(nextUint(p, "rowCount"));
      const uint64_t lo = nextUint(p, "eventLo");
      const uint64_t hi = nextUint(p, "eventHi");
      d.event = (hi << 32) | (lo & 0xFFFFFFFFull);
      reply = std::string("RPUSH ") + (gRing.publish(d) ? "1" : "0");
    } else if (std::strncmp(p, "rfree", 5) == 0) {
      reply = "RFREE " + std::to_string(gRing.capacity - gRing.occupied());
    } else if (std::strncmp(p, "rpend", 5) == 0) {
      reply = std::string("RPEND ") +
              (gRing.hasPending(gRingConsumed) ? "1" : "0");
    } else if (std::strncmp(p, "rpeek", 5) == 0) {
      if (!gRing.hasPending(gRingConsumed))
        protoError("rpeek with nothing pending");
      const HmxTileGroupDesc &d = gRing.peek(gRingConsumed);
      std::ostringstream os;
      os << "RPEEK " << d.slot << " " << d.rowStart << " " << d.rowCount
         << " " << (d.event & 0xFFFFFFFFull) << " " << (d.event >> 32);
      reply = os.str();
    } else if (std::strncmp(p, "rretire", 7) == 0) {
      if (!gRing.hasPending(gRingConsumed))
        protoError("rretire with nothing pending");
      gRing.retire(gRingConsumed);
      ++gRingConsumed;
      reply = "RRETIRE ok";
    } else if (std::strncmp(p, "rtarget", 7) == 0) {
      reply = "RTARGET " + std::to_string(gRing.drainTarget());
    } else if (std::strncmp(p, "rdrained ", 9) == 0) {
      p += 9;
      const uint64_t target = nextUint(p, "target");
      reply = std::string("RDRAINED ") +
              (gRing.drainedTo(static_cast<uint32_t>(target)) ? "1" : "0");
    } else if (std::strncmp(p, "rreset", 6) == 0) {
      gRing.reset();
      gRingConsumed = 0;
      reply = "RRESET ok";
    } else if (std::strncmp(p, "rpreset ", 8) == 0) {
      // Test-only: park both counters just below the 2^32 wrap so a handful
      // of pushes crosses it. Single-threaded battery state, so a direct
      // store is legal here (the executor's own rebind path never uses this).
      p += 8;
      const uint64_t v = nextUint(p, "value");
      if (v > 0xFFFFFFFFull)
        protoError("rpreset value out of range");
      gRing.head.store(static_cast<uint32_t>(v), std::memory_order_release);
      gRing.tail.store(static_cast<uint32_t>(v), std::memory_order_release);
      gRingConsumed = static_cast<uint32_t>(v);
      reply = "RPRESET ok";
    } else {
      protoError(std::string("unknown command: ") + p);
    }
    std::fprintf(stdout, "%s\n", reply.c_str());
    std::fflush(stdout);
  }
  std::_Exit(0);
}
