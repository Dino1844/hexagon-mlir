/* Test-only host emulation of the QuRT futex API surface HmxRoleExecutor.cpp
 * uses. NOT for the device build: there the real SDK header
 * (rtos/qurt/computev-<ver>/include/qurt/qurt_futex.h) wins the include race.
 *
 * The QuRT contract being emulated (qurt_futex.h, computev79):
 *
 *   qurt_futex_wait(lock, val): "Moves the caller thread into waiting state
 *     when a memory object address contains a value that is the same as a
 *     specified value." I.e. an atomic check-then-park on a 32-bit word.
 *   qurt_futex_wake(lock, n_to_wake): wakes at most n_to_wake waiters.
 *
 * How the emulation differs, and why each difference is safe for what the
 * host contract test measures:
 *
 *   - Implemented with one mutex + condition_variable per word address and a
 *     GENERATION counter. The generation is what closes the lost-wakeup
 *     window that a bare condvar has (a notify that lands between the
 *     waiter's value check and its park is invisible to a plain condvar
 *     wait): every wake bumps the generation under the slot's mutex, and the
 *     waiter only sleeps when the generation it sampled is still current --
 *     so a wake that arrived early still terminates the wait.
 *   - qurt_futex_wake's n_to_wake is ignored: the emulation wakes ALL
 *     waiters on the word. That is a conservative over-wake: futex semantics
 *     allow spurious wakes, every production waiter re-checks its predicate
 *     in a loop, and there is at most one waiter per word in this runtime
 *     (single producer parks on the ring's tail, single consumer on seqn).
 *   - The registry itself is guarded: lookups and insertions happen under
 *     the registry mutex, and slots are heap-allocated so their addresses
 *     stay stable while the registry grows. Slots are never freed (two words
 *     exist for this runtime's whole lifetime; the leak is two pointers).
 *
 * Counters (qurt_emu_futex_waits / qurt_emu_futex_wakes) exist so the test
 * can assert the PARK paths actually executed (bounded spin exhausted, park
 * taken) rather than inferring it from timing. Test-only, clearly named.
 */
#ifndef HMX_ROLE_HOST_EMU_QURT_FUTEX_H
#define HMX_ROLE_HOST_EMU_QURT_FUTEX_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <unordered_map>

inline std::atomic<uint64_t> qurt_emu_futex_waits{0};
inline std::atomic<uint64_t> qurt_emu_futex_wakes{0};

struct HmxEmuFutexSlot {
  std::mutex mu;
  std::condition_variable cv;
  uint64_t gen = 0;
};

inline HmxEmuFutexSlot *qurt_emu_futex_slot_for(void *lock) {
  static std::mutex registryMu;
  static std::unordered_map<void *, HmxEmuFutexSlot *> registry;
  std::lock_guard<std::mutex> lk(registryMu);
  auto it = registry.find(lock);
  if (it == registry.end()) {
    auto *slot = new HmxEmuFutexSlot();
    registry.emplace(lock, slot);
    return slot;
  }
  return it->second;
}

inline int qurt_futex_wait(void *lock, int val) {
  qurt_emu_futex_waits.fetch_add(1, std::memory_order_relaxed);
  HmxEmuFutexSlot *slot = qurt_emu_futex_slot_for(lock);
  std::atomic<uint32_t> *word = static_cast<std::atomic<uint32_t> *>(lock);
  std::unique_lock<std::mutex> lk(slot->mu);
  const uint64_t gen0 = slot->gen;
  if (word->load(std::memory_order_acquire) ==
      static_cast<uint32_t>(val)) {
    slot->cv.wait(lk, [&slot, gen0] { return slot->gen != gen0; });
  }
  return 0;
}

inline int qurt_futex_wake(void *lock, int n_to_wake) {
  (void)n_to_wake; /* over-wake is legal futex behavior; see header */
  qurt_emu_futex_wakes.fetch_add(1, std::memory_order_relaxed);
  HmxEmuFutexSlot *slot = qurt_emu_futex_slot_for(lock);
  {
    std::lock_guard<std::mutex> lk(slot->mu);
    ++slot->gen;
  }
  slot->cv.notify_all();
  return n_to_wake;
}

#endif /* HMX_ROLE_HOST_EMU_QURT_FUTEX_H */
