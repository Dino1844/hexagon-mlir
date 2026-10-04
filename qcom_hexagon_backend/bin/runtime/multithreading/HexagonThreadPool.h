//===- HexagonThreadPool.h - Hexagon multi-threading threadpool manager ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_MULTITHREADING_HEXAGON_THREAD_POOL_H
#define HEXAGON_BIN_RUNTIME_MULTITHREADING_HEXAGON_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

class HexagonThreadPool {
public:
  HexagonThreadPool(size_t numThreads = defaultThreadCount);
  ~HexagonThreadPool();

  // A coro task (handle + resume) is the only thing this pool ever runs:
  // AsyncRuntime's mlirAsyncRuntimeExecute is the sole producer. Storing the
  // pair as a 16-byte struct instead of a std::function removes a heap
  // allocation (the >SBO `[this, task]` callable) plus the double type
  // erasure per enqueue -- measured +1.7K pcyc per push+pop for the
  // std::function element (exp/hmx/fa_util/dispatch_cut, 2026-10-04), and
  // the malloc/free crossed threads (main enqueues, workers free).
  void enqueueCoro(void *handle, void (*resume)(void *));
  void wait();
  size_t getMaxConcurrency() const;

private:
  struct CoroTask {
    void *handle;
    void (*resume)(void *);
  };

  std::vector<std::thread> workers;
  std::deque<CoroTask> tasks;

  std::mutex queueMutex;
  std::condition_variable condition;
  bool stop;

  std::atomic<int> activeTasks;
  std::condition_variable allTasksDone;
  std::mutex allTasksDoneMutex;

  void workerThread();

  static constexpr size_t defaultThreadCount = 8; // Default number of threads
};

#endif // HEXAGON_BIN_RUNTIME_MULTITHREADING_HEXAGON_THREAD_POOL_H
