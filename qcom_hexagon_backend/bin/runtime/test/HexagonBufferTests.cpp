//===- HexagonBufferTests.cpp - implmentation file.              -----------==//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "BufferManager.h"
#include "HexagonBuffer.h"
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
#include "HexagonAPI.h"
#endif
#include "HexagonResources.h"
#include <gtest/gtest.h>

#include <cstdint>
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
#include <string>
#include <vector>
#endif

class HexagonBufferTest : public ::testing::Test {
public:
  static void SetUpTestSuite() { AllocateHexagonResources(); }
  static void TearDownTestSuite() { DeallocateHexagonResources(); }
};

TEST_F(HexagonBufferTest, defaultScope) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, false);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);
}

TEST_F(HexagonBufferTest, ddrScope) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, false);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);
}

TEST_F(HexagonBufferTest, vtcmScope) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, true);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kVTCM);
}

TEST_F(HexagonBufferTest, vtcmScope2) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, true);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kVTCM);
}

TEST_F(HexagonBufferTest, vtcmCarriesRequested256Alignment) {
  HexagonBuffer hb(257 /* nbytes */, 256 /* alignment */, true);
  ASSERT_TRUE(hb.HasValidAllocation());
  EXPECT_EQ(reinterpret_cast<uintptr_t>(hb.GetPointer()) % 256, 0u);
}

TEST_F(HexagonBufferTest, cacheKeyIncludesDimensionality) {
  const HexagonBuffer::CacheKey oneD{1, 1, 257, 128, false};
  const HexagonBuffer::CacheKey oneBlock2D{2, 1, 257, 128, false};

  EXPECT_FALSE(oneD == oneBlock2D);
}

TEST_F(HexagonBufferTest, freeCacheSeparates1DFromOneBlock2D) {
  constexpr size_t bytes = 257;
  constexpr size_t alignment = 128;
  BufferManager manager;

  void *oneD = manager.AllocateHexagonBuffer(bytes, alignment, false);
  ASSERT_NE(oneD, nullptr);
  manager.FreeHexagonBuffer(oneD);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  const auto before = manager.getAccountingSnapshot();
#endif

  void *oneBlock2D =
      manager.AllocateHexagonBuffer(1, bytes, alignment, false);
  ASSERT_NE(oneBlock2D, nullptr);
  // A one-block 2-D request returns a pointer table, whereas a 1-D request
  // returns the allocation itself. Reusing the 1-D cache entry would expose
  // the former as the latter's raw pointer.
  EXPECT_NE(oneBlock2D, oneD);
  EXPECT_NE(static_cast<void **>(oneBlock2D)[0], nullptr);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  EXPECT_EQ(manager.getAccountingSnapshot().freeCacheHits,
            before.freeCacheHits);
#endif
  manager.FreeHexagonBuffer(oneBlock2D);
}

TEST_F(HexagonBufferTest, freeCacheSeparatesOneBlock2DFrom1D) {
  constexpr size_t bytes = 263;
  constexpr size_t alignment = 128;
  BufferManager manager;

  void *oneBlock2D =
      manager.AllocateHexagonBuffer(1, bytes, alignment, false);
  ASSERT_NE(oneBlock2D, nullptr);
  void *row = static_cast<void **>(oneBlock2D)[0];
  ASSERT_NE(row, nullptr);
  manager.FreeHexagonBuffer(oneBlock2D);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  const auto before = manager.getAccountingSnapshot();
#endif

  void *oneD = manager.AllocateHexagonBuffer(bytes, alignment, false);
  ASSERT_NE(oneD, nullptr);
  // A cache collision would hand the old 2-D object to the 1-D caller, whose
  // GetPointer() then returns the first row address instead of allocating a
  // fresh 1-D object.
  EXPECT_NE(oneD, row);
#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
  EXPECT_EQ(manager.getAccountingSnapshot().freeCacheHits,
            before.freeCacheHits);
#endif
  manager.FreeHexagonBuffer(oneD);
}

TEST_F(HexagonBufferTest, oneBlock2DPreservesPerBlockPadding) {
  constexpr size_t bytes = 257;
  constexpr size_t alignment = 128;
  HexagonBuffer oneD(bytes, alignment, false);
  HexagonBuffer oneBlock2D(1, bytes, alignment, false);

  EXPECT_EQ(oneD.GetAllocatedBytes(), bytes);
  EXPECT_EQ(oneBlock2D.GetAllocatedBytes(), 384);
}

#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE
TEST_F(HexagonBufferTest, oneBlock2DAccountingKeepsRawInputAlignedCharged) {
  constexpr size_t rawBytes = 4097;
  constexpr size_t alignment = 256;
  VtcmPool *pool = HexagonAPI::Global()->getVtcmPool();
  BufferManager manager(pool);

  void *buffer = manager.AllocateHexagonBuffer(1, rawBytes, alignment, true);
  ASSERT_NE(buffer, nullptr);
  manager.FreeHexagonBuffer(buffer);

  // The 2-D constructor pads the one block before entering VtcmPool. Keep all
  // four accounting units distinct in this regression: raw request, padded
  // allocator input, size-rounding quantum, and the actual pool charge.
  std::vector<char> reportBuffer(256 * 1024);
  const int reportLength = pool->writeAccountingReport(
      reportBuffer.data(), static_cast<int>(reportBuffer.size()));
  ASSERT_NE(reportLength, VtcmPool::kAccountingReportTruncated);
  ASSERT_GE(reportLength, 0);
  const std::string report(reportBuffer.data(),
                           static_cast<size_t>(reportLength));
  EXPECT_NE(report.find("kind=free_cache_retain requested_bytes=4097 "
                        "allocator_input_bytes=4352 size_aligned_bytes=6144 "
                        "charged_bytes=6144"),
            std::string::npos);
}

TEST_F(HexagonBufferTest, cacheAccountingSeparatesVTCMAndDDR) {
  BufferManager *manager = HexagonAPI::Global()->getBufferManager();
  const auto before = manager->getAccountingSnapshot();

  // DDR retention is real cache behavior, but it must not enter the VTCM
  // accounting counters or the VTCM event stream.
  void *ddr = manager->AllocateHexagonBuffer(257, 128, false);
  ASSERT_NE(ddr, nullptr);
  manager->FreeHexagonBuffer(ddr);
  const auto afterDdr = manager->getAccountingSnapshot();
  EXPECT_EQ(afterDdr.freeCacheRetains, before.freeCacheRetains);
  EXPECT_EQ(afterDdr.cachedBytes, before.cachedBytes + 257);
  EXPECT_EQ(afterDdr.cachedBuffers, before.cachedBuffers);
  EXPECT_EQ(afterDdr.cachedVtcmBytes, before.cachedVtcmBytes);

  void *vtcm = manager->AllocateHexagonBuffer(257, 128, true);
  ASSERT_NE(vtcm, nullptr);
  manager->FreeHexagonBuffer(vtcm);
  const auto afterRetain = manager->getAccountingSnapshot();
  EXPECT_EQ(afterRetain.freeCacheRetains, before.freeCacheRetains + 1);
  EXPECT_EQ(afterRetain.freeCacheRetainRequestedBytes,
            before.freeCacheRetainRequestedBytes + 257);
  EXPECT_EQ(afterRetain.freeCacheRetainedBytes,
            before.freeCacheRetainedBytes + 384);
  EXPECT_EQ(afterRetain.cachedBytes, afterDdr.cachedBytes + 384);
  EXPECT_EQ(afterRetain.cachedBuffers, before.cachedBuffers + 1);
  EXPECT_EQ(afterRetain.cachedVtcmBytes, before.cachedVtcmBytes + 384);

  void *hit = manager->AllocateHexagonBuffer(257, 128, true);
  ASSERT_EQ(hit, vtcm);
  const auto afterHit = manager->getAccountingSnapshot();
  EXPECT_EQ(afterHit.freeCacheHits, before.freeCacheHits + 1);
  EXPECT_EQ(afterHit.cachedBytes, afterDdr.cachedBytes);
  EXPECT_EQ(afterHit.cachedBuffers, before.cachedBuffers);
  EXPECT_EQ(afterHit.cachedVtcmBytes, before.cachedVtcmBytes);
  manager->FreeHexagonBuffer(hit);

  // A bounded-cache failure path must evict and account only VTCM entries.
  void *large = manager->AllocateHexagonBuffer((2u << 20) + 128, 128, true);
  ASSERT_NE(large, nullptr);
  manager->FreeHexagonBuffer(large);
  const auto afterDrop = manager->getAccountingSnapshot();
  EXPECT_EQ(afterDrop.freeCacheDrops, before.freeCacheDrops + 1);

  void *live = manager->AllocateHexagonBuffer(3u << 20, 128, true);
  ASSERT_NE(live, nullptr);
  void *failed = manager->AllocateHexagonBuffer(4u << 20, 128, true);
  if (failed != nullptr)
    manager->FreeHexagonBuffer(failed);
  manager->FreeHexagonBuffer(live);
  const auto afterEvict = manager->getAccountingSnapshot();
  EXPECT_GE(afterEvict.freeCacheEvictions,
            before.freeCacheEvictions + 1);
}
#endif

TEST_F(HexagonBufferTest, microCopiesCorrespondingRegions) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(16)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 16);

  std::vector<void *> destPtr{ptr(64), ptr(80)};
  BufferSet dest(destPtr.data(), destPtr.size(), 16);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(microCopies.size(), 2);
  for (size_t i = 0; i < microCopies.size(); i++) {
    EXPECT_EQ(microCopies[i].src, ptr(16 * i));
    EXPECT_EQ(microCopies[i].dest, ptr(64 + 16 * i));
    EXPECT_EQ(microCopies[i].numBytes, 16);
  }
}

TEST_F(HexagonBufferTest, microCopiesSrcBigger) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(16)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 16);

  std::vector<void *> destPtr{ptr(64), ptr(72), ptr(80), ptr(88)};
  BufferSet dest(destPtr.data(), destPtr.size(), 8);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(microCopies.size(), 4);
  for (size_t i = 0; i < microCopies.size(); i++) {
    EXPECT_EQ(microCopies[i].src, ptr(8 * i));
    EXPECT_EQ(microCopies[i].dest, ptr(64 + 8 * i));
    EXPECT_EQ(microCopies[i].numBytes, 8);
  }
}

TEST_F(HexagonBufferTest, microCopiesDestBigger) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(8), ptr(16), ptr(24)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 8);

  std::vector<void *> destPtr{ptr(64), ptr(80)};
  BufferSet dest(destPtr.data(), destPtr.size(), 16);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(microCopies.size(), 4);
  for (size_t i = 0; i < microCopies.size(); i++) {
    EXPECT_EQ(microCopies[i].src, ptr(8 * i));
    EXPECT_EQ(microCopies[i].dest, ptr(64 + 8 * i));
    EXPECT_EQ(microCopies[i].numBytes, 8);
  }
}

TEST_F(HexagonBufferTest, microCopies_src_overlaps_dest_region) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(16)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 16);

  std::vector<void *> destPtr{ptr(64), ptr(76)};
  BufferSet dest(destPtr.data(), destPtr.size(), 12);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 24);
  EXPECT_EQ(microCopies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(microCopies[0].src, ptr(0));
  EXPECT_EQ(microCopies[0].dest, ptr(64));
  EXPECT_EQ(microCopies[0].numBytes, 12);

  // First region of source, second region of dest
  EXPECT_EQ(microCopies[1].src, ptr(12));
  EXPECT_EQ(microCopies[1].dest, ptr(76));
  EXPECT_EQ(microCopies[1].numBytes, 4);

  // Second region of source, second region of dest
  EXPECT_EQ(microCopies[2].src, ptr(16));
  EXPECT_EQ(microCopies[2].dest, ptr(80));
  EXPECT_EQ(microCopies[2].numBytes, 8);
}

TEST_F(HexagonBufferTest, microCopies_dest_overlaps_src_region) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  std::vector<void *> srcPtr{ptr(0), ptr(12)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 12);

  std::vector<void *> destPtr{ptr(64), ptr(80)};
  BufferSet dest(destPtr.data(), destPtr.size(), 16);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 24);
  EXPECT_EQ(microCopies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(microCopies[0].src, ptr(0));
  EXPECT_EQ(microCopies[0].dest, ptr(64));
  EXPECT_EQ(microCopies[0].numBytes, 12);

  // Second region of source, first region of dest
  EXPECT_EQ(microCopies[1].src, ptr(12));
  EXPECT_EQ(microCopies[1].dest, ptr(76));
  EXPECT_EQ(microCopies[1].numBytes, 4);

  // Second region of source, second region of dest
  EXPECT_EQ(microCopies[2].src, ptr(16));
  EXPECT_EQ(microCopies[2].dest, ptr(80));
  EXPECT_EQ(microCopies[2].numBytes, 8);
}

TEST_F(HexagonBufferTest, microCopiesDiscontiguousRegions) {
  auto ptr = [](auto val) { return reinterpret_cast<void *>(val); };

  // Stride of 16, but only first 11 bytes in each region belong to
  // this buffer.
  std::vector<void *> srcPtr{ptr(0), ptr(16)};
  BufferSet src(srcPtr.data(), srcPtr.size(), 11);

  std::vector<void *> destPtr{ptr(64), ptr(80)};
  BufferSet dest(destPtr.data(), destPtr.size(), 13);

  auto microCopies = BufferSet::MemoryCopies(dest, src, 16);
  EXPECT_EQ(microCopies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(microCopies[0].src, ptr(0));
  EXPECT_EQ(microCopies[0].dest, ptr(64));
  EXPECT_EQ(microCopies[0].numBytes, 11);

  // Second region of source, first region of dest
  EXPECT_EQ(microCopies[1].src, ptr(16));
  EXPECT_EQ(microCopies[1].dest, ptr(75));
  EXPECT_EQ(microCopies[1].numBytes, 2);

  // Second region of source, second region of dest
  EXPECT_EQ(microCopies[2].src, ptr(18));
  EXPECT_EQ(microCopies[2].dest, ptr(80));
  EXPECT_EQ(microCopies[2].numBytes, 3);
}

TEST_F(HexagonBufferTest, copyFrom) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data.data(), data.size());

  uint8_t *ptr = static_cast<uint8_t *>(hb.GetPointer());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST_F(HexagonBufferTest, nd) {
  HexagonBuffer hb_default(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                           false /* isVTCM */);
  EXPECT_EQ(hb_default.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);

  HexagonBuffer hbGlobal(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                         false /* isVTCM */);
  EXPECT_EQ(hbGlobal.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);

  HexagonBuffer hbVtcm(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                       true /* isVTCM */);
  EXPECT_EQ(hbVtcm.GetStorageScope(), HexagonBuffer::StorageScope::kVTCM);
}

TEST_F(HexagonBufferTest, nd_copy_from) {
  HexagonBuffer hb(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                   false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data.data(), data.size());

  uint8_t **ptr = static_cast<uint8_t **>(hb.GetPointer());
  EXPECT_EQ(ptr[0][0], data[0]);
  EXPECT_EQ(ptr[0][1], data[1]);
  EXPECT_EQ(ptr[0][2], data[2]);
  EXPECT_EQ(ptr[0][3], data[3]);
  EXPECT_EQ(ptr[1][0], data[4]);
  EXPECT_EQ(ptr[1][1], data[5]);
  EXPECT_EQ(ptr[1][2], data[6]);
  EXPECT_EQ(ptr[1][3], data[7]);
}

TEST_F(HexagonBufferTest, 1dCopyFrom1d) {
  HexagonBuffer from(8 /* nbytes */, 8 /* alignment */, false /* isVTCM */);
  HexagonBuffer to(8 /* nbytes */, 8 /* alignment */, true /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  from.CopyFrom(data.data(), data.size());
  to.CopyFrom(from, 8);

  uint8_t *ptr = static_cast<uint8_t *>(to.GetPointer());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST_F(HexagonBufferTest, 2d_copy_from_1d) {
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, true /* isVTCM */);
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                     false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb1d.CopyFrom(data.data(), data.size());
  hb2d.CopyFrom(hb1d, 8);

  uint8_t **ptr = static_cast<uint8_t **>(hb2d.GetPointer());
  EXPECT_EQ(ptr[0][0], data[0]);
  EXPECT_EQ(ptr[0][1], data[1]);
  EXPECT_EQ(ptr[0][2], data[2]);
  EXPECT_EQ(ptr[0][3], data[3]);
  EXPECT_EQ(ptr[1][0], data[4]);
  EXPECT_EQ(ptr[1][1], data[5]);
  EXPECT_EQ(ptr[1][2], data[6]);
  EXPECT_EQ(ptr[1][3], data[7]);
}

TEST_F(HexagonBufferTest, 1dCopyFrom2d) {
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                     true /* isVTCM */);
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb2d.CopyFrom(data.data(), data.size());
  hb1d.CopyFrom(hb2d, 8);

  uint8_t *ptr = static_cast<uint8_t *>(hb1d.GetPointer());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST_F(HexagonBufferTest, mdCopyFromNd) {
  HexagonBuffer hb3d(3 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                     false /* isVTCM */);
  HexagonBuffer hb4d(4 /* ndim */, 3 /* nbytes */, 8 /* alignment */,
                     false /* isVTCM */);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  hb3d.CopyFrom(data.data(), data.size());
  hb4d.CopyFrom(hb3d, data.size());

  uint8_t **hb3dPtr = static_cast<uint8_t **>(hb3d.GetPointer());
  uint8_t **hb4dPtr = static_cast<uint8_t **>(hb4d.GetPointer());
  for (size_t i = 0; i < 12; i++) {
    EXPECT_EQ(hb3dPtr[i / 4][i % 4], hb4dPtr[i / 3][i % 3]);
  }
}

TEST_F(HexagonBufferTest, copyTo) {
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, false /* isVTCM */);

  std::vector<uint8_t> dataIn{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(dataIn.data(), dataIn.size());

  std::vector<uint8_t> dataOut{7, 6, 5, 4, 3, 2, 1, 0};
  hb.CopyTo(dataOut.data(), dataOut.size());

  for (size_t i = 0; i < dataIn.size(); ++i) {
    EXPECT_EQ(dataIn[i], dataOut[i]);
  }
}

TEST_F(HexagonBufferTest, ndCopyTo) {
  HexagonBuffer hb(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */,
                   false /* isVTCM */);

  std::vector<uint8_t> dataIn{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(dataIn.data(), dataIn.size());

  std::vector<uint8_t> dataOut{7, 6, 5, 4, 3, 2, 1, 0};
  hb.CopyTo(dataOut.data(), dataOut.size());

  for (size_t i = 0; i < dataIn.size(); ++i) {
    EXPECT_EQ(dataIn[i], dataOut[i]);
  }
}
