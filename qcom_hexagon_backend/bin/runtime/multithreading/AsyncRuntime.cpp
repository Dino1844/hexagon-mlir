//===- AsyncRuntime.cpp - Async runtime reference implementation ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements basic Async runtime API for supporting Async dialect
// to LLVM dialect lowering.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

// Crash-triage progress marker (fa-rowmax plan §10/§15). This opens, writes
// and closes rt_trc.txt on every async runtime call, which otherwise dominates
// small kernels (fa-crash log R24); it is compiled out unless
// HEXMLIR_RUNTIME_TRACE is defined. Never enable it for measurements.
#ifdef HEXMLIR_RUNTIME_TRACE
static void _trc(const char *m) {
  FILE *_f = fopen("/data/data/com.termux/files/home/csm/op/rt_trc.txt", "a");
  if (_f) { fputs(m, _f); fputc('\n', _f); fclose(_f); }
}
#else
#define _trc(m) ((void)0)
#endif
#include "AsyncRuntime.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "HexagonThreadPool.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir::runtime;

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace runtime {
namespace {

// Forward declare classes defined below.
class RefCounted;
template <typename T> class PooledFreelist;

// -------------------------------------------------------------------------- //
// A mutex-guarded free list for recycling AsyncToken / AsyncGroup objects.
//
// The dispatch path allocates one token per async.execute on the main
// thread and destroys it on the worker that emplaces it -- a cross-thread
// malloc/free pair per execute (measured ~1.9K pcyc for the create+destroy
// round trip, plus the embedded awaiters vector's own 32-byte malloc per
// AddTokenToGroup; exp/hmx/fa_util/dispatch_cut, 2026-10-04). Recycling the
// objects removes both, and keeps the awaiters vector's heap storage warm
// across reuses. The list mutex is uncontended in steady state (main
// pushes/pops between group gaps, workers push on destroy) and costs
// ~0.17K pcyc per op vs ~1K+ for the malloc path.
//
// The intrusive link lives in RefCounted::poolNext -- never at offset 0,
// which is the vtable pointer (a freelist next stored there was caught
// crashing the virtual destroy() dispatch by the dispatch_cut probe,
// 2026-10-04).
// -------------------------------------------------------------------------- //

template <typename T> class PooledFreelist {
public:
  // Returns nullptr when the list is empty (caller falls back to new).
  T *pop() {
    std::lock_guard<std::mutex> lock(mu);
    if (!head)
      return nullptr;
    T *p = head;
    head = static_cast<T *>(p->poolNext);
    --count;
    return p;
  }

  // Returns false when the pool is at capacity (caller deletes instead).
  bool push(T *p) {
    std::lock_guard<std::mutex> lock(mu);
    if (count >= kCapacity)
      return false;
    p->poolNext = head;
    head = p;
    ++count;
    return true;
  }

private:
  static constexpr size_t kCapacity = 512;
  std::mutex mu;
  T *head = nullptr;
  size_t count = 0;
};
// -------------------------------------------------------------------------- //
// AsyncRuntime orchestrates all async operations and Async runtime API is built
// on top of the default runtime instance.
// -------------------------------------------------------------------------- //

class AsyncRuntime {
public:
  AsyncRuntime() : numRefCountedObjects(0) {}

  ~AsyncRuntime() {
    threadPool.wait(); // wait for the completion of all async tasks
    drainPools();      // pooled objects still count as alive until deleted
    assert(getNumRefCountedObjects() == 0 &&
           "all ref counted objects must be destroyed");
  }

  int64_t getNumRefCountedObjects() {
    return numRefCountedObjects.load(std::memory_order_relaxed);
  }

  HexagonThreadPool &getThreadPool() { return threadPool; }

  PooledFreelist<AsyncToken> &tokenPool() { return tokenFreelist; }
  PooledFreelist<AsyncGroup> &groupPool() { return groupFreelist; }

private:
  friend class RefCounted;

  void drainPools();

  // Count the total number of reference counted objects in this instance
  // of an AsyncRuntime. For debugging purposes only.
  void addNumRefCountedObjects() {
    numRefCountedObjects.fetch_add(1, std::memory_order_relaxed);
  }
  void dropNumRefCountedObjects() {
    numRefCountedObjects.fetch_sub(1, std::memory_order_relaxed);
  }

  std::atomic<int64_t> numRefCountedObjects;
  HexagonThreadPool threadPool;
  PooledFreelist<AsyncToken> tokenFreelist;
  PooledFreelist<AsyncGroup> groupFreelist;
};

// -------------------------------------------------------------------------- //
// A state of the async runtime value (token, value or group).
// -------------------------------------------------------------------------- //

class State {
public:
  enum StateEnum : int8_t {
    // The underlying value is not yet available for consumption.
    kUnavailable = 0,
    // The underlying value is available for consumption. This state can not
    // transition to any other state.
    kAvailable = 1,
    // This underlying value is available and contains an error. This state can
    // not transition to any other state.
    kError = 2,
  };

  /* implicit */ State(StateEnum s) : state(s) {}
  /* implicit */ operator StateEnum() { return state; }

  bool isUnavailable() const { return state == kUnavailable; }
  bool isAvailable() const { return state == kAvailable; }
  bool isError() const { return state == kError; }
  bool isAvailableOrError() const { return isAvailable() || isError(); }

  const char *debug() const {
    switch (state) {
    case kUnavailable:
      return "unavailable";
    case kAvailable:
      return "available";
    case kError:
      return "error";
    }
  }

private:
  StateEnum state;
};

// -------------------------------------------------------------------------- //
// A base class for all reference counted objects created by the async runtime.
// -------------------------------------------------------------------------- //

class RefCounted {
public:
  RefCounted(AsyncRuntime *runtime, int64_t refCount = 1)
      : runtime(runtime), refCount(refCount) {
    runtime->addNumRefCountedObjects();
  }

  virtual ~RefCounted() {
    assert(refCount.load() == 0 && "reference count must be zero");
    runtime->dropNumRefCountedObjects();
  }

  RefCounted(const RefCounted &) = delete;
  RefCounted &operator=(const RefCounted &) = delete;

  void addRef(int64_t count = 1) { refCount.fetch_add(count); }

  void dropRef(int64_t count = 1) {
    int64_t previous = refCount.fetch_sub(count);
    assert(previous >= count && "reference count should not go below zero");
    if (previous == count)
      destroy();
  }

protected:
  virtual void destroy() { delete this; }

  // Recycles this object through a pool instead of deleting it: the ref
  // count is reset for the next user. Pool-recycled objects must not have
  // outstanding references (same discipline as delete).
  void resetRefCount(int64_t count) { refCount.store(count); }

private:
  template <typename T> friend class PooledFreelist;

  AsyncRuntime *runtime;
  std::atomic<int64_t> refCount;

  // Intrusive free-list link used while the object sits in a pool (see
  // PooledFreelist). Guarded by the pool's mutex.
  void *poolNext = nullptr;
};


} // namespace

// Returns the default per-process instance of an async runtime.
static std::unique_ptr<AsyncRuntime> &getDefaultAsyncRuntimeInstance() {
  static auto runtime = std::make_unique<AsyncRuntime>();
  return runtime;
}

static void resetDefaultAsyncRuntime() {
  return getDefaultAsyncRuntimeInstance().reset();
}

static AsyncRuntime *getDefaultAsyncRuntime() {
  return getDefaultAsyncRuntimeInstance().get();
}

// Async token provides a mechanism to signal asynchronous operation completion.
struct AsyncToken : public RefCounted {
  // AsyncToken created with a reference count of 2 because it will be returned
  // to the `async.execute` caller and also will be later on emplaced by the
  // asynchronously executed task. If the caller immediately will drop its
  // reference we must ensure that the token will be alive until the
  // asynchronous operation is completed.
  AsyncToken(AsyncRuntime *runtime)
      : RefCounted(runtime, /*refCount=*/2), state(State::kUnavailable) {}

  // Pool recycling: reset the mutable state for the next user. The embedded
  // mutex / condition_variable are left as-is (unlocked, no waiters can
  // remain at refcount zero) and the awaiters vector keeps its heap storage
  // (cleared, so the next AddTokenToGroup emplace does not allocate).
  void reinit() {
    resetRefCount(2);
    state = State::kUnavailable;
    awaiters.clear();
  }

protected:
  void destroy() override {
    if (!getDefaultAsyncRuntime()->tokenPool().push(this))
      delete this;
  }

public:
  std::atomic<State::StateEnum> state;

  // Pending awaiters are guarded by a mutex.
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::function<void()>> awaiters;
};

// Async value provides a mechanism to access the result of asynchronous
// operations. It owns the storage that is used to store/load the value of the
// underlying type, and a flag to signal if the value is ready or not.
struct AsyncValue : public RefCounted {
  // AsyncValue similar to an AsyncToken created with a reference count of 2.
  AsyncValue(AsyncRuntime *runtime, int64_t size)
      : RefCounted(runtime, /*refCount=*/2), state(State::kUnavailable),
        storage(size) {}

  std::atomic<State::StateEnum> state;

  // Use vector of bytes to store async value payload.
  std::vector<std::byte> storage;

  // Pending awaiters are guarded by a mutex.
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::function<void()>> awaiters;
};

// Async group provides a mechanism to group together multiple async tokens or
// values to await on all of them together (wait for the completion of all
// tokens or values added to the group).
struct AsyncGroup : public RefCounted {
  AsyncGroup(AsyncRuntime *runtime, int64_t size)
      : RefCounted(runtime), pendingTokens(size), numErrors(0), rank(0) {}

  // Pool recycling: see AsyncToken::reinit.
  void reinit(int64_t size) {
    resetRefCount(1);
    pendingTokens = (int)size;
    numErrors = 0;
    rank = 0;
    awaiters.clear();
  }

protected:
  void destroy() override {
    if (!getDefaultAsyncRuntime()->groupPool().push(this))
      delete this;
  }

public:
  std::atomic<int> pendingTokens;
  std::atomic<int> numErrors;
  std::atomic<int> rank;

  // Pending awaiters are guarded by a mutex.
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::function<void()>> awaiters;
};

// Out of line: the pooled types are complete only here, below their
// definitions; deleting through them runs the RefCounted virtual
// destructor correctly.
inline void AsyncRuntime::drainPools() {
  while (auto *p = tokenFreelist.pop())
    delete p;
  while (auto *p = groupFreelist.pop())
    delete p;
}

// Adds references to reference counted runtime object.
extern "C" void mlirAsyncRuntimeAddRef(RefCountedObjPtr ptr, int64_t count) {
  RefCounted *refCounted = static_cast<RefCounted *>(ptr);
  refCounted->addRef(count);
}

// Drops references from reference counted runtime object.
extern "C" void mlirAsyncRuntimeDropRef(RefCountedObjPtr ptr, int64_t count) {
  RefCounted *refCounted = static_cast<RefCounted *>(ptr);
  refCounted->dropRef(count);
}

// Creates a new `async.token` in not-ready state.
extern "C" AsyncToken *mlirAsyncRuntimeCreateToken() { _trc("T0");
  AsyncRuntime *runtime = getDefaultAsyncRuntime();
  if (AsyncToken *token = runtime->tokenPool().pop()) {
    token->reinit();
    return token;
  }
  return new AsyncToken(runtime);
}

// Creates a new `async.value` in not-ready state.
extern "C" AsyncValue *mlirAsyncRuntimeCreateValue(int64_t size) {
  AsyncValue *value = new AsyncValue(getDefaultAsyncRuntime(), size);
  return value;
}

// Create a new `async.group` in empty state.
extern "C" AsyncGroup *mlirAsyncRuntimeCreateGroup(int64_t size) { _trc("G0");
  AsyncRuntime *runtime = getDefaultAsyncRuntime();
  if (AsyncGroup *group = runtime->groupPool().pop()) {
    group->reinit(size);
    return group;
  }
  return new AsyncGroup(runtime, size);
}

extern "C" int64_t mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token,
                                                    AsyncGroup *group) { _trc("G1");
  std::unique_lock<std::mutex> lockToken(token->mu);

  // The ready/not-ready decision must be made under token->mu: racing with
  // setTokenState (the worker that emplaces this token) otherwise appends
  // an awaiter to an already-ready token and the group's pendingTokens
  // never reaches zero (caught as an AwaitAllInGroup hang by the
  // dispatch_cut probe, 2026-10-04).

  if (State(token->state).isAvailableOrError()) {
    // The group lock is required on the ready path: onTokenReady changes
    // the group predicate and runs its awaiters, which must not race with
    // AwaitAllInGroup's sleep (lost wakeup).
    std::unique_lock<std::mutex> lockGroup(group->mu);

    // Get the rank of the token inside the group before we drop the reference.
    int rank = group->rank.fetch_add(1);

    auto onTokenReady = [group, token]() {
      // Increment the number of errors in the group.
      if (State(token->state).isError())
        group->numErrors.fetch_add(1);

      // If pending tokens go below zero it means that more tokens than the group
      // size were added to this group.
      assert(group->pendingTokens > 0 && "wrong group size");

      // Run all group awaiters if it was the last token in the group.
      if (group->pendingTokens.fetch_sub(1) == 1) {
        group->cv.notify_all();
        for (auto &awaiter : group->awaiters)
          awaiter();
      }
    };

    // Update group pending tokens immediately and maybe run awaiters.
    onTokenReady();

    return rank;
  }

  // Not-ready path: this thread touches no group state guarded by group->mu
  // (rank and addRef are atomics; the awaiter closure below takes group->mu
  // itself on the worker), so token->mu above is enough. One uncontended
  // mutex pair is ~0.33K pcyc (dispatch_cut probe); the async dispatch
  // always takes this path.

  // Get the rank of the token inside the group before we drop the reference.
  int rank = group->rank.fetch_add(1);

  // Update group pending tokens when token will become ready. Because this
  // will happen asynchronously we must ensure that `group` is alive until
  // then.
  group->addRef();

  token->awaiters.emplace_back([group, token]() {
    // Runs on the worker that emplaces the token. Make sure that `dropRef`
    // does not destroy the mutex owned by the lock.
    {
      std::unique_lock<std::mutex> lockGroup(group->mu);
      // Increment the number of errors in the group.
      if (State(token->state).isError())
        group->numErrors.fetch_add(1);

      assert(group->pendingTokens > 0 && "wrong group size");

      // Run all group awaiters if it was the last token in the group.
      if (group->pendingTokens.fetch_sub(1) == 1) {
        group->cv.notify_all();
        for (auto &awaiter : group->awaiters)
          awaiter();
      }
    }
    group->dropRef();
  });

  return rank;
}

// Switches `async.token` to available or error state (terminatl state) and runs
// all awaiters.
static void setTokenState(AsyncToken *token, State state) {
  assert(state.isAvailableOrError() && "must be terminal state");
  assert(State(token->state).isUnavailable() && "token must be unavailable");

  // Make sure that `dropRef` does not destroy the mutex owned by the lock.
  {
    std::unique_lock<std::mutex> lock(token->mu);
    token->state = state;
    token->cv.notify_all();
    for (auto &awaiter : token->awaiters)
      awaiter();
  }

  // Async tokens created with a ref count `2` to keep token alive until the
  // async task completes. Drop this reference explicitly when token emplaced.
  token->dropRef();
}

static void setValueState(AsyncValue *value, State state) {
  assert(state.isAvailableOrError() && "must be terminal state");
  assert(State(value->state).isUnavailable() && "value must be unavailable");

  // Make sure that `dropRef` does not destroy the mutex owned by the lock.
  {
    std::unique_lock<std::mutex> lock(value->mu);
    value->state = state;
    value->cv.notify_all();
    for (auto &awaiter : value->awaiters)
      awaiter();
  }

  // Async values created with a ref count `2` to keep value alive until the
  // async task completes. Drop this reference explicitly when value emplaced.
  value->dropRef();
}

extern "C" void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) { _trc("M0");
  setTokenState(token, State::kAvailable);
}

extern "C" void mlirAsyncRuntimeEmplaceValue(AsyncValue *value) {
  setValueState(value, State::kAvailable);
}

extern "C" void mlirAsyncRuntimeSetTokenError(AsyncToken *token) { _trc("SE");
  setTokenState(token, State::kError);
}

extern "C" void mlirAsyncRuntimeSetValueError(AsyncValue *value) {
  setValueState(value, State::kError);
}

extern "C" bool mlirAsyncRuntimeIsTokenError(AsyncToken *token) {
  return State(token->state).isError();
}

extern "C" bool mlirAsyncRuntimeIsValueError(AsyncValue *value) {
  return State(value->state).isError();
}

extern "C" bool mlirAsyncRuntimeIsGroupError(AsyncGroup *group) { _trc("E0");
  return group->numErrors.load() > 0;
}

extern "C" void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  std::unique_lock<std::mutex> lock(token->mu);
  if (!State(token->state).isAvailableOrError())
    token->cv.wait(
        lock, [token] { return State(token->state).isAvailableOrError(); });
}

extern "C" void mlirAsyncRuntimeAwaitValue(AsyncValue *value) {
  std::unique_lock<std::mutex> lock(value->mu);
  if (!State(value->state).isAvailableOrError())
    value->cv.wait(
        lock, [value] { return State(value->state).isAvailableOrError(); });
}

extern "C" void mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) { _trc("W0");
  std::unique_lock<std::mutex> lock(group->mu);
  if (group->pendingTokens != 0)
    group->cv.wait(lock, [group] { return group->pendingTokens == 0; });
}

// Returns a pointer to the storage owned by the async value.
extern "C" ValueStorage mlirAsyncRuntimeGetValueStorage(AsyncValue *value) {
  assert(!State(value->state).isError() && "unexpected error state");
  return value->storage.data();
}

extern "C" void mlirAsyncRuntimeExecute(CoroHandle handle, CoroResume resume) { _trc("X0");
  auto *runtime = getDefaultAsyncRuntime();
  runtime->getThreadPool().enqueueCoro(handle, resume);
}

extern "C" void mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token,
                                                     CoroHandle handle,
                                                     CoroResume resume) {
  auto execute = [handle, resume]() { (*resume)(handle); };
  std::unique_lock<std::mutex> lock(token->mu);
  if (State(token->state).isAvailableOrError()) {
    lock.unlock();
    execute();
  } else {
    token->awaiters.emplace_back([execute]() { execute(); });
  }
}

extern "C" void mlirAsyncRuntimeAwaitValueAndExecute(AsyncValue *value,
                                                     CoroHandle handle,
                                                     CoroResume resume) {
  auto execute = [handle, resume]() { (*resume)(handle); };
  std::unique_lock<std::mutex> lock(value->mu);
  if (State(value->state).isAvailableOrError()) {
    lock.unlock();
    execute();
  } else {
    value->awaiters.emplace_back([execute]() { execute(); });
  }
}

extern "C" void mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group,
                                                          CoroHandle handle,
                                                          CoroResume resume) {
  auto execute = [handle, resume]() { (*resume)(handle); };
  std::unique_lock<std::mutex> lock(group->mu);
  if (group->pendingTokens == 0) {
    lock.unlock();
    execute();
  } else {
    group->awaiters.emplace_back([execute]() { execute(); });
  }
}

extern "C" int64_t mlirAsyncRuntimGetNumWorkerThreads() {
  return getDefaultAsyncRuntime()->getThreadPool().getMaxConcurrency();
}

//===----------------------------------------------------------------------===//
// Small async runtime support library for testing.
//===----------------------------------------------------------------------===//

extern "C" void mlirAsyncRuntimePrintCurrentThreadId() {
  [[maybe_unused]] static thread_local std::thread::id thisId =
      std::this_thread::get_id();
  // std::cout << "Current thread id: " << thisId << '\n';
}

//===----------------------------------------------------------------------===//
// MLIR ExecutionEngine dynamic library integration.
//===----------------------------------------------------------------------===//

// Visual Studio had a bug that fails to compile nested generic lambdas
// inside an `extern "C"` function.
//   https://developercommunity.visualstudio.com/content/problem/475494/clexe-error-with-lambda-inside-function-templates.html
// The bug is fixed in VS2019 16.1. Separating the declaration and definition is
// a work around for older versions of Visual Studio.
// NOLINTNEXTLINE(*-identifier-naming): externally called.
extern "C" MLIR_ASYNC_RUNTIME_EXPORT void
__mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols);

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void __mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols) {
  auto exportSymbol = [&](llvm::StringRef name, auto ptr) {
    assert(exportSymbols.count(name) == 0 && "symbol already exists");
    exportSymbols[name] = reinterpret_cast<void *>(ptr);
  };

  exportSymbol("mlirAsyncRuntimeAddRef",
               &mlir::runtime::mlirAsyncRuntimeAddRef);
  exportSymbol("mlirAsyncRuntimeDropRef",
               &mlir::runtime::mlirAsyncRuntimeDropRef);
  exportSymbol("mlirAsyncRuntimeExecute",
               &mlir::runtime::mlirAsyncRuntimeExecute);
  exportSymbol("mlirAsyncRuntimeGetValueStorage",
               &mlir::runtime::mlirAsyncRuntimeGetValueStorage);
  exportSymbol("mlirAsyncRuntimeCreateToken",
               &mlir::runtime::mlirAsyncRuntimeCreateToken);
  exportSymbol("mlirAsyncRuntimeCreateValue",
               &mlir::runtime::mlirAsyncRuntimeCreateValue);
  exportSymbol("mlirAsyncRuntimeEmplaceToken",
               &mlir::runtime::mlirAsyncRuntimeEmplaceToken);
  exportSymbol("mlirAsyncRuntimeEmplaceValue",
               &mlir::runtime::mlirAsyncRuntimeEmplaceValue);
  exportSymbol("mlirAsyncRuntimeSetTokenError",
               &mlir::runtime::mlirAsyncRuntimeSetTokenError);
  exportSymbol("mlirAsyncRuntimeSetValueError",
               &mlir::runtime::mlirAsyncRuntimeSetValueError);
  exportSymbol("mlirAsyncRuntimeIsTokenError",
               &mlir::runtime::mlirAsyncRuntimeIsTokenError);
  exportSymbol("mlirAsyncRuntimeIsValueError",
               &mlir::runtime::mlirAsyncRuntimeIsValueError);
  exportSymbol("mlirAsyncRuntimeIsGroupError",
               &mlir::runtime::mlirAsyncRuntimeIsGroupError);
  exportSymbol("mlirAsyncRuntimeAwaitToken",
               &mlir::runtime::mlirAsyncRuntimeAwaitToken);
  exportSymbol("mlirAsyncRuntimeAwaitValue",
               &mlir::runtime::mlirAsyncRuntimeAwaitValue);
  exportSymbol("mlirAsyncRuntimeAwaitTokenAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitTokenAndExecute);
  exportSymbol("mlirAsyncRuntimeAwaitValueAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitValueAndExecute);
  exportSymbol("mlirAsyncRuntimeCreateGroup",
               &mlir::runtime::mlirAsyncRuntimeCreateGroup);
  exportSymbol("mlirAsyncRuntimeAddTokenToGroup",
               &mlir::runtime::mlirAsyncRuntimeAddTokenToGroup);
  exportSymbol("mlirAsyncRuntimeAwaitAllInGroup",
               &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroup);
  exportSymbol("mlirAsyncRuntimeAwaitAllInGroupAndExecute",
               &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroupAndExecute);
  exportSymbol("mlirAsyncRuntimGetNumWorkerThreads",
               &mlir::runtime::mlirAsyncRuntimGetNumWorkerThreads);
  exportSymbol("mlirAsyncRuntimePrintCurrentThreadId",
               &mlir::runtime::mlirAsyncRuntimePrintCurrentThreadId);
}

// NOLINTNEXTLINE(*-identifier-naming): externally called.
extern "C" MLIR_ASYNC_RUNTIME_EXPORT void __mlir_execution_engine_destroy() {
  resetDefaultAsyncRuntime();
}

} // namespace runtime
} // namespace mlir
