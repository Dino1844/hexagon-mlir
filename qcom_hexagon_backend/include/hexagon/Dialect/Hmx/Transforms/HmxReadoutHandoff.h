//===-- HmxReadoutHandoff.h - compiler half of the vector-readout ABI --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// WHY THIS HEADER EXISTS
// ---------------------
// The vector read-out split has TWO compiler halves that run at DIFFERENT points
// of the pipeline and cannot see each other's IR:
//
//   * HmxVectorReadoutPass runs while kernels are still `func.func`. It builds the
//     `HmxReadoutBatch` descriptor, publishes it, drains, and outlines the
//     read-out into a private function.
//   * HmxToLLVMPass runs after `convert-func-to-llvm`. It is the only pass that
//     can take a function's ADDRESS, so it emits the `configure` call naming the
//     vector thread's work.
//
// Both halves have to agree, exactly, on the runtime symbols, on the descriptor's
// field order, and on how the one names the other. Spelling those in one header is
// what stops a one-sided rename from becoming a silent mismatch: a wrong symbol
// is an unresolved link, and a wrong field index is the executor reading a
// garbage pointer out of the middle of a tile descriptor -- neither of which any
// compiler check would report.
//
// This is the same arrangement HmxResidentContract.h uses for the diagnostic
// resident contract, for the same reason.
//
// THE FROZEN HALF
// ----------------
// The runtime's half of this ABI is bin/runtime/include/HmxVectorExecutor.h, which
// is frozen: `HmxReadoutBatch` (six 32-bit words, field order below),
// `HmxReadoutFn` (a two-argument function pointer) and the four entry points.
// Nothing below may restate a *meaning* from it -- only the compiler's spelling
// of the same facts.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXREADOUTHANDOFF_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXREADOUTHANDOFF_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace hmx {

/// Runtime entry points of the vector executor. Spelled once: the producer pass
/// declares them, the lowering calls two of them, and the linker resolves them
/// against bin/runtime/multithreading/HmxVectorExecutor.cpp.
inline constexpr StringLiteral kHmxExecConfigureFn =
    "hexagon_runtime_hmx_exec_configure";
inline constexpr StringLiteral kHmxExecPublishFn =
    "hexagon_runtime_hmx_exec_publish";
inline constexpr StringLiteral kHmxExecDrainFn = "hexagon_runtime_hmx_exec_drain";
inline constexpr StringLiteral kHmxExecShutdownFn =
    "hexagon_runtime_hmx_exec_shutdown";

/// `HmxReadoutBatch` as the DEVICE C compiler lays it out.
///
/// NOT-A-DECISION: the word count and the field order are the frozen ABI in
/// bin/runtime/include/HmxVectorExecutor.h, which declares
/// `uint32_t rowStart, rowCount, validRows, nCroutons; void *ar, *dst;`. A
/// pointer is 32 bits on this target -- every address in this backend crosses
/// into the runtime as an i32 (`asAddress`, HmxToLLVMPass.cpp) -- so the struct
/// is 24 bytes with no padding, and the runtime compiles that same header with
/// hexagon-clang++, which is the layout it will read. The header's "32 bytes" is
/// the host x86-64 layout and never applies to a device descriptor.
///
/// `validRows` and `nCroutons` are written but never read back by the outlined
/// read-out: they are properties of the LAST row of a batch, and per-row
/// validity cannot be a static attribute once rows are grouped (see
/// matchReadout). They exist because the struct is an interface fact, not
/// because the compiler half consumes them.
inline constexpr int64_t kHmxReadoutBatchWords = 6;
enum HmxReadoutBatchField : int64_t {
  kHmxReadoutRowStart = 0,
  kHmxReadoutRowCount = 1,
  kHmxReadoutValidRows = 2,
  kHmxReadoutNCroutons = 3,
  kHmxReadoutAr = 4,
  kHmxReadoutDst = 5,
};

/// Marker on the outlined read-out function itself. It says "this is the work
/// the vector thread runs" and nothing more, which is why the details travel in
/// the module sidecar below rather than here: a function attribute does not
/// survive every conversion in the pipeline (CollapseAddressSpace rebuilds
/// llvm.func and drops discardable attributes), while a module attribute is
/// never touched by any of them.
inline constexpr StringLiteral kHmxReadoutOutlinedAttr =
    "hmx.readout.outlined";

/// Module-level handoff record: an array of dictionaries, one per rewritten
/// kernel, each naming
///
///   engine : StringAttr, the kernel whose publish/drain the executor serves
///   work   : StringAttr, the outlined function `configure` is pointed at
///   ar     : TypeAttr (MemRefType), the accumulator array's type
///   dst    : TypeAttr (MemRefType), the destination's type
///
/// The two memref types are the reason this record exists rather than a search
/// for "the function named `__hmx_readout`". `convert-func-to-llvm` rewrites a
/// memref argument into its COMPONENTS (allocated, aligned, offset, sizes,
/// strides) and turns `index` into i64, so by the time anything can name the
/// function, the outlined signature no longer says which words are which
/// memref's. The producer is the only place that still knows.
inline constexpr StringLiteral kHmxReadoutHandoffsAttr = "hmx.readout.handoffs";
inline constexpr StringLiteral kHmxReadoutEngineField = "engine";
inline constexpr StringLiteral kHmxReadoutWorkField = "work";
inline constexpr StringLiteral kHmxReadoutArField = "ar";
inline constexpr StringLiteral kHmxReadoutDstField = "dst";

/// Suffix of the generated entry point: the outlined read-out keeps its own
/// four-argument shape (the vector thread cannot be given a `func.func`
/// signature, only a C one), and this names the function that adapts the
/// runtime's `HmxReadoutFn` onto it.
inline constexpr StringLiteral kHmxReadoutEntrySuffix = "_entry";

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXREADOUTHANDOFF_H