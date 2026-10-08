//===-- HmxRoleHandoff.h - compiler half of the thread-role channel ABI ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// WHY THIS HEADER EXISTS
// ---------------------
// The thread-role split has TWO compiler halves that run at DIFFERENT points
// of the pipeline and cannot see each other's IR, exactly the arrangement
// HmxReadoutHandoff.h documents for the vector read-out:
//
//   * ThreadRolePartition runs while kernels are still `func.func`, right
//     after hmx-partition built the tile loop. It outlines the engine
//     section into a private work function, rewrites the tile loop into the
//     producer side (pack + submit + drain), and records the handoff below.
//   * HmxToLLVMPass runs after `convert-func-to-llvm`. It is the only pass
//     that can build the C-ABI entry point and the exported depth object
//     the launch-side probe dlsym's, and the only one that can exempt the
//     work function from the per-kernel ensure/unlock pair (the bound
//     thread already holds the HMX lock for its lifetime).
//
// Both halves, plus the runtime probe, have to agree on the runtime symbol
// names, the sidecar schema, and the descriptor's word layout. The suffix
// spellings live in the runtime's own shared header
// (bin/runtime/include/HmxRoleChannel.h), which the probe and the
// HmxToLLVM emission both include; everything that is purely the compiler's
// business is spelled HERE, once.
//
// THE FROZEN HALF
// ----------------
// The runtime's halves of this ABI are bin/runtime/include/HmxRoleExecutor.h
// (the executor: HmxSectionFn, bind/submit/drain/join) and
// bin/runtime/include/HmxRoleChannel.h (the channel: the suffixes and the
// probe entry). Both are frozen contracts; nothing below may restate a
// *meaning* from them -- only the compiler's spelling of the same facts.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXROLEHANDOFF_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXROLEHANDOFF_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace hmx {

/// Runtime entry points of the role executor, spelled once for the producer
/// pass's declarations (the lowering calls three of them and checks nothing:
/// the short return is HANDLED in IR, not trapped -- see the emission).
inline constexpr StringLiteral kHmxRoleSubmitFn =
    "hexagon_runtime_hmx_role_submit";
inline constexpr StringLiteral kHmxRoleDrainFn =
    "hexagon_runtime_hmx_role_drain";
/// The granular barrier (the ABI entry's own comment in HmxRoleExecutor.h is
/// the contract): wait until the retire counter reaches the target. Spelled
/// here for the same reason submit/drain are -- the pass declares it, the
/// runtime defines it, and a one-sided rename must be an unresolved link
/// rather than a silently missing wait.
inline constexpr StringLiteral kHmxRoleWaitRetiredFn =
    "hexagon_runtime_hmx_role_wait_retired";

/// The region's thread role, as an attribute on the function that runs it
/// (ROADMAP1001 section 3.2: the carrier is an attribute, not a new op; the
/// value discriminates the role). A builtin StringAttr rather than a
/// dedicated attr class on purpose: the attr is read by this pass's own
/// checks and by the manifest story, and a builtin carrier needs no dialect
/// registration to survive in -- and be printed from -- the IR between the
/// outlining and the conversion. It does NOT survive every later conversion
/// (function attributes are discardable; CollapseAddressSpace rebuilds
/// llvm.func), which is the same reason the read-out's marker is a function
/// attr only where it is consumed and a MODULE attr wherever it has to
/// travel: the identity that has to reach HmxToLLVMPass travels in the
/// handoff record below, and the LLVM fn attribute that carries the thread
/// contract to the backend ("hexagon_hmx", upstream PR #222340's TTI hooks,
// backported in S0) is set by HmxToLLVMPass itself, on the llvm.func.
inline constexpr StringLiteral kHmxThreadRoleAttr = "hex.thread_role";
inline constexpr StringLiteral kHmxThreadRoleHmx = "hmx";

/// Prefix of the outlined engine-section work function. Numbered
/// probe-then-create (`__hmx_role_section`, `__hmx_role_section_1`, ...) the
/// same way the read-out's is, under the shared module-state mutex.
inline constexpr StringLiteral kHmxRoleWorkFnPrefix = "__hmx_role_section";

/// Module-level handoff record: an array of dictionaries, one per split
/// kernel, each naming
///
///   engine : StringAttr, the kernel whose producer side submits
///   work   : StringAttr, the outlined work function the entry point calls
///   depth  : IntegerAttr, the ring depth the launch side binds (also the
///            rotating scratch row count -- see the emission for why they are
///            the same number)
///   rows   : TypeAttr (MemRefType), the rotating scratch's type
///   wt     : TypeAttr (MemRefType), the weight crouton array's type
///   bias   : TypeAttr (MemRefType), the bias block's type
///   ar     : TypeAttr (MemRefType), the accumulator array's type
///
/// The four memref types are here for the same reason the read-out's are in
/// its record: by the time anything can take a function's address,
/// `convert-func-to-llvm` has exploded each memref argument into its
/// components, so the outlined signature no longer says which words belong
/// to which buffer. The producer pass is the only place that still knows.
///
/// `engine` is also the prefix of the two EXPORTED channel symbols
/// (`<engine>__hmx_section`, `<engine>__hmx_role_depth`): HmxToLLVMPass
/// builds both names from it plus the suffixes in HmxRoleChannel.h.
inline constexpr StringLiteral kHmxRoleHandoffsAttr = "hmx.role.handoffs";
inline constexpr StringLiteral kHmxRoleEngineField = "engine";
inline constexpr StringLiteral kHmxRoleWorkField = "work";
inline constexpr StringLiteral kHmxRoleDepthField = "depth";
inline constexpr StringLiteral kHmxRoleRowsField = "rows";
inline constexpr StringLiteral kHmxRoleWtField = "wt";
inline constexpr StringLiteral kHmxRoleBiasField = "bias";
inline constexpr StringLiteral kHmxRoleArField = "ar";

/// THE PER-LAUNCH BUFFER TABLE, and why it exists
/// ------------------------------------------------------------
/// The read-out handoff ships its buffers (ar, dst) INSIDE its descriptor:
/// HmxReadoutBatch has two pointer words. HmxTileGroupDesc does not -- the
/// frozen ABI is 24 bytes of slot/rowStart/rowCount/event, none of them a
/// buffer address -- so the four buffers the engine section closes over
/// (rows, wt, bias, ar) cannot ride the ring. They are per-LAUNCH values
/// (kernel arguments and an allocation), so they cannot be constants of the
/// outlined function either. The one channel left is a module-level table
/// the producer writes once per launch, before its first submit, and the
/// entry point reads per section call:
///
///   producer (kernel entry):  store rows/wt/bias/ar addresses into the
///                             four globals below
///   entry (per descriptor):   load the four addresses, rebuild the memref
///                             arguments (appendMemrefArgs), call the work
///                             function
///
/// This is the same division the readout uses -- the ring carries the WORK
/// ITEMS, the function parameters carry the buffers closed over -- with the
/// one mechanical difference the frozen descriptor forces: the buffers'
/// addresses travel beside the ring instead of inside the descriptor.
/// Soundness rests on the single-producer, one-launch-at-a-time contract
/// the executor already states (HmxRoleExecutor.h: one launch is one
/// producer; grid>1 is outside the contract): the drain before the kernel
/// returns means no section call reads the table after the launch that
/// wrote it ends.
inline constexpr StringLiteral kHmxRoleRowsGlobal = "__hmx_role_rows";
inline constexpr StringLiteral kHmxRoleWtGlobal = "__hmx_role_wt";
inline constexpr StringLiteral kHmxRoleBiasGlobal = "__hmx_role_bias";
inline constexpr StringLiteral kHmxRoleArGlobal = "__hmx_role_ar";

/// `HmxTileGroupDesc` as the DEVICE C compiler lays it out, in i32 words.
///
/// NOT-A-DECISION: the field order and the PADDING are the frozen ABI in
/// bin/runtime/include/HmxSpscRing.h, whose static_assert pins the struct at
/// 24 bytes on the target: three uint32 fields, one 4-byte alignment pad
/// (the uint64 event needs 8-byte alignment), then the event's two words.
/// The pad word is WRITTEN (zero) rather than left to whatever the stack
/// held: the runtime never reads it, and a deterministic producer keeps the
/// descriptor a pure function of the tile index.
///
/// The event word itself is 0 in this emission: the first form carries no
/// event to hand across (the slot field names the buffer, the ownership
/// transfer is the ring's own protocol), and the word is opaque to the
/// runtime by contract (readEventWord discipline: unsigned, never
/// reinterpreted, never reconstructed from a pointer). A future form that
/// wires a real event fills words 4/5 and changes nothing else.
inline constexpr int64_t kHmxRoleDescWords = 6;
enum HmxRoleDescField : int64_t {
  kHmxRoleDescSlot = 0,
  kHmxRoleDescRowStart = 1,
  kHmxRoleDescRowCount = 2,
  kHmxRoleDescPad = 3,
  kHmxRoleDescEventLo = 4,
  kHmxRoleDescEventHi = 5,
};

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXROLEHANDOFF_H
