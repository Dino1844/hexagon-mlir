//===-- HmxTarget.h - what the HMX engine can take ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The engine's contract, as a value the passes consult -- not as logic buried in
// a rewrite pattern that names one op class.
//
// Upstream Triton has the same shape: an `Accelerate*` pass per op class, each
// asking `TargetInfo` whether the *operation* fits (`getMMAVersionSafe` +
// `supportMMA`).
//
// Honest status (2026-09-23, review): the *table* is generic, but the only
// adapter that asks it is `matmul-to-hmx`, and its bridge/epilogue logic is
// written against `linalg.matmul` directly. A second op class will first need
// that extraction generalized (an op interface for extents/element types), not
// just a thin adapter -- update this comment when it happens instead of letting
// the claim run ahead of the code.
//
// See docs/history/hmx/hmx-generality.md for what is and is not general today.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXTARGET_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXTARGET_H

#include "hexagon/Dialect/Hmx/IR/HmxCroutonLayout.h"
#include "hexagon/Dialect/Hmx/IR/HmxDType.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlir {
namespace hmx {

/// The HMX engine capabilities known to this compiler.
///
/// The contract is compile-time constants -- one crouton is a 32x32 fp16 block,
/// the device VTCM is 8 MiB and the engine shares it with the rest of the kernel
/// -- so there is no runtime target description to parse. The VTCM budget is the
/// one field a pass may narrow (a test, or a future device); it defaults to the
/// device's.
struct HmxTarget {
  /// One crouton is a 32x32 fp16 block; every extent the engine touches is a
  /// multiple of it.
  // NOT-A-DECISION: one crouton edge, 32. This is the HMX hardware tile
  // geometry, not a tuning knob: the engine's crouton is 16x32x2 whatever we
  // would prefer. Aliased from layout::kTileEdge so the one spelling lives in
  // the layout contract (HmxCroutonLayout.h), cross-checked against
  // bin/runtime/hmx/include/HMXAPI.h.
  static constexpr int64_t tileEdge = layout::kTileEdge;

  /// The smallest logical M an HMX contraction may have.
  ///
  /// Sourced from the handwritten reference: llama.cpp draws the same line at
  /// `HTP_MM_HMX_MIN_NROWS = 4`
  /// (`llama.cpp/ggml/src/ggml-hexagon/htp/matmul-ops.h:20`, used at
  /// `ggml-hexagon.cpp:3871`).
  ///
  /// ⚠️ **The two sides state different mechanisms, and that is unresolved.**
  /// llama.cpp's own comment calls theirs "M alignment"; the comment here used
  /// to claim "the engine needs a few rows before a tile can be formed", which
  /// does not follow obviously — a crouton row is 32 elements, so needing only
  /// 4 rows is not self-evident. Treat the *value* as sourced from the
  /// reference (it is), and the *mechanism* as `[假设]`.
  ///
  /// ⚠️ **Reachable, but only for M in 1..4** — and that is a real gate, not a
  /// dead one. `queryContraction` checks `minRows` *before* it splits extents
  /// for alignment, so a contraction with `m = 1..4` is refused here as
  /// `MinRows` even though it would also fail alignment. (An earlier note in
  /// this file claimed it might be unreachable, on the reasoning that a
  /// tile-aligned M is a multiple of 32 and so never <= 4. That reasoning is
  /// wrong: alignment is checked *after* this gate, not before.)
  ///
  /// The open question is therefore not reachability but **whether M <= 4
  /// occurs in a real workload at all** — plausible for a skinny GEMV/outer
  /// product, unlikely for the attention/FFN matmuls this backend targets. That
  /// is a workload question, not a code question; see
  /// `docs/codegen/constants-traceability-2026-09-30.md` C4.
  //
  // TRACEABILITY: minRows
  //   mechanism: a capability floor. `queryContraction` refuses M <= minRows
  //     before it splits extents for alignment, so the M range 1..4 can never
  //     reach HMX no matter how the other extents look.
  //   measurement: **none of our own.** The value is sourced from the
  //     handwritten reference (see above), and the two sides state DIFFERENT
  //     mechanisms for it -- llama.cpp says "M alignment", this file previously
  //     said "rows needed to form a tile", and a 32-row crouton needing only 4
  //     rows does not follow obviously. **Mechanism: [假设].**
  //   shape set: n/a -- the constant is a threshold, not a tuned parameter, and
  //     no calibration set applies to it.
  //   workload representativeness: **NOT ESTABLISHED.** Open question is whether
  //     M <= 4 occurs in a real workload at all (plausible for a skinny
  //     GEMV/outer product, unlikely for attention/FFN matmuls). Nobody has
  //     looked, so the gate's real-world hit rate is unknown.
  static constexpr int64_t minRows = 4;

  // TRACEABILITY: defaultVtcmBudget
  //   mechanism: one shared on-chip VTCM pool. Every admission, block plan and
  //     ring decision in the bridge is weighed against this number, so it is a
  //     capacity, not a preference.
  //   measurement: a DEVICE QUERY, not a measurement of ours --
  //     `HAP_compute_res_query_VTCM` reports 8 MB total
  //     (docs/analysis/hmx-device-prerequisites.md:222). The same document's
  //     section 12 also records VTCM being unavailable or shared with system
  //     processes, so the *usable fraction* is unverified.
  //   shape set: n/a (a capacity, not a tuned threshold).
  //   workload representativeness: n/a, but see the caveat above: this is the
  //     TOTAL pool, and treating total as usable is the open question.
  //   failure mode: if the usable pool is smaller, the pass admits work the
  //     runtime cannot place. That is loud (allocation failure / DSP abort),
  //     not a silent slowdown, which is why it is ranked below the staging
  //     floor. Settling it needs no timing: print the queried usable bytes.
  static constexpr int64_t defaultVtcmBudget = 8 * 1024 * 1024;

  /// The budget every query is weighed against; defaults to the device's.
  int64_t vtcmBudget = defaultVtcmBudget;

  /// Whether the pipeline provides the allocator the crouton arrays need.
  ///
  /// The engine reads and writes VTCM only, and the crouton arrays are memory
  /// space 1, so a pipeline that does not lower space-1 allocations through the
  /// runtime VTCM pool (`enableConvertToHexagonmem` / the hexagonmem path) would
  /// leave them somewhere the engine cannot reach. Attributing there does not
  /// degrade -- it kills the DSP -- so it is refused instead.
  bool vtcmAllocator = true;

  /// The engine's one contraction contract: an f16 read-out (a wider result is
  /// that image widened afterwards, exactly what llama.cpp's f32 HMX matmul
  /// does), all three extents on the tile grid, and enough rows.
  ///
  /// The operands are the engine's fp16 croutons; a *source* may be fp32, in
  /// which case the pack that materialises the crouton quantises it (see
  /// `hmx.pack_act`). That is why a wider operand element type is admitted here
  /// rather than refused: the engine never sees it.
  ///
  /// Any op whose meaning reduces to such a contraction can ask this -- that is
  /// the whole of the engine's capability today. A second shape (a quantized
  /// operands contract, say) belongs here as another query, not as another pass.
  static bool isContractionOperand(Type elem) {
    return dtype::isAdmittedFloat(elem);
  }

  /// The shape-level plan selected by the capability query. `HMXTail` is a
  /// classification, not permission to lower: only the explicit diagnostic
  /// producer/partition bridge consumes it, and production attribution remains
  /// closed until kernel-wide workspace accounting is complete.
  enum class ContractionPlan { FullHMX, HMXTail, HVX };

  /// The single capability query behind both attribution and its diagnostics.
  /// Keeping the refusal here prevents the pass from growing a second copy of
  /// the engine contract merely to explain it.
  enum class ContractionRefusal {
    None,
    UnsupportedDType,
    MinRows,
    TileAlignment,
  };

  struct ContractionShape {
    int64_t m = 0, n = 0, k = 0;
    int64_t mp = 0, np = 0, kp = 0;
    int64_t mf = 0, nf = 0, kf = 0;
    int64_t mt = 0, nt = 0, kt = 0;
  };

  struct ContractionDecision {
    ContractionRefusal refusal = ContractionRefusal::None;
    ContractionPlan plan = ContractionPlan::HVX;
    ContractionShape shape;

    /// `supported()` remains the executable predicate for the current
    /// pipeline. A tail candidate deliberately has a TileAlignment refusal
    /// until the tail lowering contract is implemented.
    bool supported() const { return refusal == ContractionRefusal::None; }
    bool tailCandidate() const { return plan == ContractionPlan::HMXTail; }
    explicit operator bool() const { return supported(); }
  };

  static bool splitExtent(int64_t value, int64_t &full, int64_t &padded,
                          int64_t &tail) {
    if (value <= 0 ||
        value > std::numeric_limits<int64_t>::max() - (tileEdge - 1))
      return false;
    full = (value / tileEdge) * tileEdge;
    tail = value - full;
    padded = ((value + tileEdge - 1) / tileEdge) * tileEdge;
    return true;
  }

  ContractionDecision queryContraction(int64_t m, int64_t n, int64_t k,
                                      Type lhsElem, Type rhsElem,
                                      Type outElem) const {
    if (!isContractionOperand(lhsElem) || !isContractionOperand(rhsElem) ||
        (!dtype::isAdmittedFloat(outElem)))
      return {ContractionRefusal::UnsupportedDType, ContractionPlan::HVX, {}};
    if (m <= minRows)
      return {ContractionRefusal::MinRows, ContractionPlan::HVX, {}};

    ContractionShape shape;
    shape.m = m;
    shape.n = n;
    shape.k = k;
    if (!splitExtent(m, shape.mf, shape.mp, shape.mt) ||
        !splitExtent(n, shape.nf, shape.np, shape.nt) ||
        !splitExtent(k, shape.kf, shape.kp, shape.kt))
      return {ContractionRefusal::TileAlignment, ContractionPlan::HVX, {}};

    bool aligned = shape.mt == 0 && shape.nt == 0 && shape.kt == 0;
    return {aligned ? ContractionRefusal::None
                    : ContractionRefusal::TileAlignment,
            aligned ? ContractionPlan::FullHMX : ContractionPlan::HMXTail,
            shape};
  }

  /// How the crouton bridge pays for its residency: the M extent it walks in
  /// one block, and the crouton bytes that one block holds. A contraction too
  /// large to hold whole is walked in M blocks instead of being refused (see
  /// `planBridge`).
  struct BridgePlan {
    /// Rows of M one block covers: a multiple of the tile edge and a divisor of
    /// the contraction's M, so the blocks tile M exactly -- no partial block and
    /// no bounds guard. 0 means no bridge fits.
    int64_t blockM = 0;
    /// The bytes one block holds: its activation block, the whole weight and its
    /// fp16 read-out block. This is the peak VTCM the bridge asks for.
    int64_t bytes = 0;

    /// True when the contraction is walked in several M blocks.
    bool blocked(int64_t m) const { return blockM > 0 && blockM < m; }
    explicit operator bool() const { return blockM > 0; }
  };

  /// The crouton's element size in bytes: the engine's fp16, whatever the
  /// source's element type is (a wider source is quantised by the pack, its
  /// crouton array is still fp16).
  // NOT-A-DECISION: the crouton element is fp16, hence 2 bytes. A property of
  // the format, not a choice.
  static constexpr int64_t croutonElemBytes = 2;

  /// The crouton footprint of a contraction operated on `rows` rows: both
  /// operands and the engine's fp16 read-out, in bytes. The read-out is fp16
  /// whatever the result's element type (a wider result is widened after the
  /// unpack, outside VTCM).
  static int64_t croutonBytes(int64_t rows, int64_t n, int64_t k) {
    return (rows * k + k * n + rows * n) * croutonElemBytes;
  }

  /// The engine's second question, next to `queryContraction`: legality
  /// says the engine *can* take this contraction, this says how the crouton
  /// bridge fits what is left of the pool. One hard gate, in the terse shape of
  /// upstream's K threshold (`supportMMA`: `if (k < 256 / bitWidth)`).
  ///
  /// When the whole contraction fits, the plan is the whole M and the bridge is
  /// emitted once. When it does not, M is walked in the largest block that
  /// fits -- shrinking the activation and the read-out, the two arrays that
  /// scale with M -- while the weight stays whole. Only when not even one block
  /// fits is the contraction refused (an empty plan), and the IR is left
  /// untouched: never a half-built hmx region the verifier could see.
  ///
  /// Blocking M rather than N: the weight and the read-out both scale with N,
  /// so a block along N would shrink the read-out but leave the weight whole
  /// either way; a block along M shrinks both the activation and the read-out,
  /// which is where a large MxN product's bytes are. N (the weight) becomes the
  /// binding term at the other end and is blocked the same way when it does.
  BridgePlan planBridge(int64_t m, int64_t n, int64_t k,
                        int64_t vtcmUsed) const {
    int64_t room = vtcmBudget - vtcmUsed;
    // The whole contraction wins when it fits. The strict `<` is the same
    // boundary the pass's budget remark is written against (`footprint >= room`
    // refuses).
    int64_t whole = croutonBytes(m, n, k);
    if (whole < room)
      return BridgePlan{m, whole};

    // Blocking M shrinks the activation and the read-out but not the weight:
    //   F(b) = b * (k + n) * croutonElemBytes + k * n * croutonElemBytes.
    // F is increasing in b, so the largest fitting b is one division away.
    int64_t perRow = (k + n) * croutonElemBytes;
    int64_t weightBytes = k * n * croutonElemBytes;
    if (perRow <= 0)
      return BridgePlan{};
    int64_t maxRows = (room - weightBytes - 1) / perRow;
    int64_t maxBlock = (maxRows / tileEdge) * tileEdge;
    if (maxBlock < tileEdge)
      return BridgePlan{};
    // Round the block down to a divisor of the tile count so every block is the
    // same static shape. One tile always divides, so a block exists whenever
    // `maxBlock` did.
    int64_t maxTiles = maxBlock / tileEdge;
    int64_t mTiles = m / tileEdge;
    int64_t blockTiles = 0;
    for (int64_t d = std::min(maxTiles, mTiles); d >= 1; --d) {
      if (mTiles % d == 0) {
        blockTiles = d;
        break;
      }
    }
    if (blockTiles == 0)
      return BridgePlan{};
    int64_t blockM = blockTiles * tileEdge;
    // A block that spans the whole M would have fit the branch above; a block
    // that does not divide M cannot tile it (only reachable for an M off the
    // tile grid, which the engine's contract already rejects).
    if (blockM >= m || m % blockM != 0)
      return BridgePlan{};
    return BridgePlan{blockM, croutonBytes(blockM, n, k)};
  }
};

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXTARGET_H
