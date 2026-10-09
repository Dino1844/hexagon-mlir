//===- RuntimeDMA.h - Specification of DMA APIs for users ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_RUNTIME_DMA_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_RUNTIME_DMA_H

#include <cstdint>

namespace hexagon {
namespace userdma {

typedef enum { DMAFailure = -1, DMASuccess = 0 } DMAStatus;
typedef enum { DDR = 0, VTCM = 1 } AddrSpace;

/// Token returned by the start APIs when they refuse a transfer. A refusal
/// also sets `*status = DMAFailure`; no ring descriptor is enqueued, so there
/// is nothing to wait for and `hexagon_runtime_dma_wait` returns immediately
/// for this value. Real tokens are ring allocation counters starting at 0,
/// so a refusal cannot be encoded as 0. (A real token can equal this value
/// only after the 32-bit ring counter wraps at 2^32 starts; that one wait()
/// would return without polling -- accepted against the alternative this
/// replaces, a forged completed descriptor on every refused start.)
///
/// ABI contract, both sides: `status` must point at storage of its own. It
/// must not be memory that later holds the returned token, or the caller's
/// token store erases the failure report (the bug behind
/// roadmap/ARCH-REVIEW.md #2; the compiler side is pinned by
/// test/Conversion/DMAToLLVM/dma_status_vs_token.mlir).
static const uint32_t DMA_TOKEN_NONE = static_cast<uint32_t>(-1);

extern "C" uint32_t
hexagon_runtime_dma_start(void *src, AddrSpace srcAS, void *dst,
                          AddrSpace dstAS, uint32_t length, bool bypassCacheSrc,
                          bool bypassCacheDst, DMAStatus *status);

extern "C" uint32_t hexagon_runtime_dma2d_start(
    void *src, AddrSpace srcAS, void *dst, AddrSpace dstAS, uint32_t width,
    uint32_t height, uint32_t srcStride, uint32_t dstStride,
    bool bypassCacheSrc, bool bypassCacheDst, bool isOrdered,
    uint32_t cacheAllocationPolicy, DMAStatus *status);

extern "C" void hexagon_runtime_dma_wait(uint32_t token);

} // namespace userdma
} // namespace hexagon

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_RUNTIME_DMA_H
