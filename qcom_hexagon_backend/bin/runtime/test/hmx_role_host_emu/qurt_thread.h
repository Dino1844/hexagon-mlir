/* Test-only host emulation of the QuRT thread API surface HmxRoleExecutor.cpp
 * uses. NOT for the device build: there the real SDK header
 * (rtos/qurt/computev-<ver>/include/qurt/qurt_thread.h) wins the include
 * race, and this file is never on its path.
 *
 * What is emulated, and how the emulation differs from QuRT -- each deviation
 * is a place the host test is weaker than the device, listed here rather than
 * left implicit:
 *
 *   qurt_thread_create: spawns a std::thread. The attr's stack address/size
 *     are IGNORED (std::thread provisions its own stack), so the device-only
 *     behavior of running on a malloc'd DDR stack of kStackBytes is not
 *     exercised. Stack sizing is not what the host contract test measures.
 *   qurt_thread_get_priority: returns a constant (100, the ThreadManager
 *     default the device path would clamp). Priority is a scheduling
 *     property; the host test asserts ordering only through the drain
 *     barrier, which does not depend on it.
 *   qurt_thread_get_id: returns a sentinel handle. The production code only
 *     uses it to fetch the creator's priority, so the sentinel is enough.
 *
 * Identity observability (test-only): every thread created through
 * qurt_thread_create records its std::thread::id, in creation order, so the
 * test can prove the bound section ran on the thread that was created (and
 * that the HMX ensure ran there too). Clearly named qurt_emu_* so it cannot
 * be mistaken for a QuRT API.
 */
#ifndef HMX_ROLE_HOST_EMU_QURT_THREAD_H
#define HMX_ROLE_HOST_EMU_QURT_THREAD_H

#include <thread>
#include <vector>

typedef struct {
  std::thread *handle;
} qurt_thread_t;

typedef struct {
  const char *name;
  unsigned short priority;
  unsigned int stack_size;
  void *stack_addr;
} qurt_thread_attr_t;

inline void qurt_thread_attr_init(qurt_thread_attr_t *attr) {
  attr->name = nullptr;
  attr->priority = 100;
  attr->stack_size = 0;
  attr->stack_addr = nullptr;
}
inline void qurt_thread_attr_set_name(qurt_thread_attr_t *attr,
                                      const char *name) {
  attr->name = name;
}
inline void qurt_thread_attr_set_priority(qurt_thread_attr_t *attr,
                                          unsigned short priority) {
  attr->priority = priority;
}
inline void qurt_thread_attr_set_stack_size(qurt_thread_attr_t *attr,
                                            unsigned int stack_size) {
  attr->stack_size = stack_size;
}
inline void qurt_thread_attr_set_stack_addr(qurt_thread_attr_t *attr,
                                            void *stack_addr) {
  attr->stack_addr = stack_addr;
}

/* Creation-order record of every thread this emulation started. The last
 * entry is the most recently created (== the bound thread, one at a time). */
inline std::vector<std::thread::id> &qurt_emu_created_thread_ids() {
  static std::vector<std::thread::id> ids;
  return ids;
}

inline int qurt_thread_create(qurt_thread_t *thread_id,
                              qurt_thread_attr_t *attr,
                              void (*entrypoint)(void *), void *args) {
  (void)attr; /* name/priority/stack are scheduling properties; see header */
  try {
    thread_id->handle = new std::thread(entrypoint, args);
  } catch (...) {
    thread_id->handle = nullptr;
    return -1;
  }
  qurt_emu_created_thread_ids().push_back(thread_id->handle->get_id());
  return 0;
}

inline int qurt_thread_join(qurt_thread_t thread_id, int *status) {
  if (thread_id.handle == nullptr) {
    return -1;
  }
  thread_id.handle->join();
  delete thread_id.handle;
  *status = 0;
  return 0;
}

inline qurt_thread_t qurt_thread_get_id(void) {
  qurt_thread_t t;
  t.handle = nullptr; /* sentinel: only fed to qurt_thread_get_priority */
  return t;
}

inline int qurt_thread_get_priority(qurt_thread_t thread_id) {
  (void)thread_id;
  return 100;
}

#endif /* HMX_ROLE_HOST_EMU_QURT_THREAD_H */
