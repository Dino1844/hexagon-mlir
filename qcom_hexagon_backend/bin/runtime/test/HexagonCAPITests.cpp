//===- HexagonCAPITests.cpp                                  --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "HexagonCAPI.h"
#include "HexagonResources.h"
#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

static_assert(
    std::is_same<decltype(&hexagon_runtime_workspace_resident_v2),
                 void *(*)(uint64_t, uint32_t, uint32_t)>::value,
    "workspace resident v2 ABI must be (i64 key, i32 bytes, i32 alignment)");
static_assert(
    std::is_same<decltype(&hexagon_runtime_weight_resident_v2),
                 void *(*)(uint64_t, uint32_t, uint32_t)>::value,
    "weight resident v2 ABI must be (i64 source, i32 bytes, i32 alignment)");
static_assert(std::is_same<decltype(&hexagon_runtime_resident_free_v2),
                           int32_t (*)(void *, uint32_t)>::value,
              "resident free v2 ABI must be (ptr, i32 bytes) -> i32");

static_assert(std::is_same<decltype(&hexagon_runtime_resident_scope_enter_v2),
                           int32_t (*)(uint64_t, uint64_t)>::value,
              "resident scope v2 ABI must be (i64 low, i64 high) -> i32");

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
static_assert(
    std::is_same<decltype(&hexagon_runtime_vtcm_accounting_context_enter_v1),
                 int32_t (*)(uint32_t, uint32_t, uint64_t, uint64_t, uint64_t,
                             uint64_t, uint64_t, uint64_t, uint64_t)>::value,
    "VTCM accounting context v1 ABI must carry explicit versioned IDs");
#endif

TEST(HexagonCAPItest, test_flow) {
  AllocateHexagonResources();
  size_t bytes{2048};
  int64_t shape2d[2]{256, 256};
  size_t alignment{128};
  size_t alignment_2d{2048};
  void *buf1 = hexagon_runtime_alloc_1d(bytes, alignment, false /* isVtcm */);
  void *buf2 = hexagon_runtime_alloc_1d(bytes, alignment, true /* isVtcm */);
  void *croutonBuf1 = hexagon_runtime_build_crouton(buf1, bytes);
  void *origBuf1 = hexagon_runtime_get_contiguous_memref(croutonBuf1);
  EXPECT_EQ(origBuf1, buf1);
  hexagon_runtime_free_1d(buf1);
  hexagon_runtime_free_1d(buf2);
  buf1 = hexagon_runtime_alloc_2d(shape2d[0], bytes, alignment_2d,
                                  false /* isVtcm */);
  buf2 = hexagon_runtime_alloc_2d(shape2d[0], bytes, alignment_2d,
                                  true /* isVtcm */);
  hexagon_runtime_free_2d(buf1);
  hexagon_runtime_free_2d(buf2);
  DeallocateHexagonResources();
}

TEST(HexagonCAPItest, resident_v2_scope_is_single_and_idempotent) {
  constexpr uint64_t low = 0x0123456789abcdefULL;
  constexpr uint64_t high = 0xfedcba9876543210ULL;
  EXPECT_EQ(hexagon_runtime_resident_scope_enter_v2(low, high), 0);
  EXPECT_EQ(hexagon_runtime_resident_scope_enter_v2(low, high), 0);
  EXPECT_NE(hexagon_runtime_resident_scope_enter_v2(low ^ 1, high), 0);
  EXPECT_EQ(hexagon_runtime_resident_scope_enter_v2(low, high), 0);
}
