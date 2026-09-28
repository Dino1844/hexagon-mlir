//===- hexagon_benchmark.h ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

/* Header file used to time a pipeline.
 *  This is meant to be used in device-standalone mode. */

#ifndef HEXAGON_BENCHMARK_H
#define HEXAGON_BENCHMARK_H
#include "HAP_perf.h"
#include "hexagon_types.h"

// Returns the average time, in microseconds, taken to run
// op for the given number of iterations.
template <typename F> uint64_t benchmark_time_us(int iterations, F op) {
  uint64_t start_time = HAP_perf_get_time_us();

  for (int i = 0; i < iterations; ++i) {
    op();
  }

  uint64_t end_time = HAP_perf_get_time_us();
  return (uint64_t)((end_time - start_time) / iterations);
}

// Returns the average cycles taken to run
// op for the given number of iterations.
template <typename F> uint64_t benchmark_pcycles(int iterations, F op) {
  uint64_t start_cycle = HAP_perf_get_pcycles();

  for (int i = 0; i < iterations; ++i) {
    op();
  }

  uint64_t end_cycle = HAP_perf_get_pcycles();
  return (uint64_t)((end_cycle - start_cycle) / iterations);
}

// Returns BOTH averages from a SINGLE pass over `op`.
//
// Two separate calls (benchmark_time_us then benchmark_pcycles) would time two
// different runs, and on this part the run-to-run state is not constant: the
// processor cycle counter C15:14 stops while the DSP is clock-gated
// (HAP_perf.h:79) while the 19.2 MHz qtimer does not, so the ratio between them
// moves with how much of the run the core spends waiting. Measured on one build,
// the same shape reported 2.10-2.15 GHz in a tight leaf loop and 1.10-1.15 GHz
// inside a full matmul. Timing the two separately would therefore compare
// different regimes and produce a ratio that belongs to neither.
//
// One pass, both counters, is the only way the pair is comparable. See
// docs/hmx/hmx-perf-findings-2026-09-27.md 1.
template <typename F>
void benchmark_time_and_pcycles(int iterations, F op, uint64_t *us_out,
                                uint64_t *pcycles_out) {
  uint64_t start_time = HAP_perf_get_time_us();
  uint64_t start_cycle = HAP_perf_get_pcycles();

  for (int i = 0; i < iterations; ++i) {
    op();
  }

  uint64_t end_cycle = HAP_perf_get_pcycles();
  uint64_t end_time = HAP_perf_get_time_us();
  *us_out = (uint64_t)((end_time - start_time) / iterations);
  *pcycles_out = (uint64_t)((end_cycle - start_cycle) / iterations);
}
#endif
