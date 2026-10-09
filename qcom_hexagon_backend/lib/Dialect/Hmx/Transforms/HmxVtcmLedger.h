//===-- HmxVtcmLedger.h - the one VTCM byte ledger ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// "How many bytes of VTCM does this function already hold?" was answered by
// five hand-written walks:
//
//   * `HmxPartitionPass::vtcmBytesCommitted`      (memref walk + declaration)
//   * `ThreadRolePartition::roleVtcmBytesCommitted`
//     (line-for-line the same walk + declaration; its own comment said the two
//      passes "run at the same pipeline point and must see the same number or
//      this pass's budget check is fiction")
//   * `WeightResidentPass::transientVtcmBytes`    (the walk without the
//     declaration, because that pass adds the resident bytes itself)
//   * `MatmulToHmxPass::vtcmBytesCommitted`       (the tensor-level image:
//     `hmx.alloc_crouton` before bufferization)
//   * `HmxVtcmAccountingPass`'s allocation-site census
//
// Those five were *copies*, not cross-checks. Every copy carried a comment
// asserting it "must see the same number" as the others, which is a description
// of drift risk rather than an independent derivation: nothing among them could
// disagree and be right. (ARCH-REVIEW verification 2026-10-09, section 3.1,
// argued this before the extraction was approved: "must see the same number" =
// copy, not check.) This is the opposite of the duplications this repository
// keeps on purpose -- the crouton geometry copies pinned against each other by
// `test_crouton_size_agreement.py`, for instance, where a disagreement is
// *possible* and is made loud by a test.
//
// A genuine cross-check derives the answer twice from different evidence, and
// those stay exactly where they are:
//
//   * `WeightResidentPass::verifyResidentByteAggregate` re-sums the resident
//     descriptors and rejects a module whose declaration does not close;
//   * `HmxVtcmAccountingPass` re-derives the footprint from the allocation
//     sites and marks its census `incomplete` when the two disagree.
//
// The accounting pass is therefore a *strict superset* of this ledger: it
// subsumes every population below, splits the buckets further (transient /
// workspace-resident / weight-resident, aligned charge, liveness peaks) and
// reads the declaration through `residentBytes`. It keeps its own site walk --
// it reports per-site facts this ledger does not have -- and it consumes the
// ledger rather than re-implementing the ledger's sums.
//
// Scope: this file owns *how the bytes are counted*. It deliberately does not
// own a budget (each pass has its own, see `HmxTarget::defaultVtcmBudget` and
// the `vtcm-budget` options), a capacity constant (four of those exist; see
// the traceability notes in `HmxTarget.h`), or a cost model
// (`HmxLeafCostTable.h` still has no budget consumer -- wiring it in would be
// a behaviour change, not an extraction, and is deliberately out of scope).
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXVTCMLEDGER_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXVTCMLEDGER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <cstdint>

namespace mlir {
class Operation;

namespace hmx {
namespace vtcm {

/// Which population of allocations a query reads. The two populations are the
/// two images of the *same* VTCM arrays at the two ends of bufferization, and
/// both existed before this file -- a caller picks by where it runs, not by
/// preference.
enum class Population {
  /// Tensor level (before bufferization): static `hmx.alloc_crouton` arrays.
  /// The population `matmul-to-hmx` attributes against. No resident term: the
  /// resident declaration is written by `weight-resident`, which runs after
  /// bufferization, and reading it here would double-count the weight, whose
  /// array is still an `hmx.alloc_crouton` in this population.
  Tensor,
  /// Memref level (after bufferization): static `memref.alloc`s in the VTCM
  /// space, plus the resident declaration. `hexagonmem.alloc` is *not* walked:
  /// a resident weight is a `hexagonmem.alloc` (or already lowered) and would
  /// then be counted twice -- once through the walk it is invisible to today
  /// and once through the declaration it is only visible through. Transient
  /// `hexagonmem.alloc`s are likewise not walked, exactly as they were not
  /// walked by the five callers before; widening the population would move
  /// every budget this side of the conversion and is not part of an
  /// extraction.
  Memref,
};

/// Bytes `within`'s function already holds in VTCM from `population`, without
/// the resident declaration. `within` may be the function itself or any
/// operation inside it; no enclosing function means 0.
int64_t transientBytes(Operation *within, Population population);

/// The module's declared resident footprint (`hmx.weight_resident_bytes`), 0
/// when absent. This is the *plain* read used by the budget readers; the
/// writer's guarded read (`WeightResidentPass::residentBytesDeclared`) rejects
/// a malformed declaration instead of returning it and stays a check of its
/// own. `scope` may be a module or anything inside one.
int64_t residentBytes(Operation *scope);

/// What the budget checks compare against: `transientBytes` plus, for the
/// memref population, `residentBytes`. The tensor population carries no
/// resident term (see `Population::Tensor`).
int64_t committedBytes(Operation *within, Population population);

/// `budget - committedBytes(within, Population::Memref) + released`, the room a
/// caller has once the buffers it is about to free (`released`, e.g. the
/// activation array a staging path retires) come back to the pool.
int64_t roomBytes(Operation *within, int64_t budget, int64_t released = 0);

// The function-shaped overloads the callers hold; they only fix up the anchor
// operation, so there is one implementation behind both spellings.
inline int64_t transientBytes(func::FuncOp func, Population population) {
  return transientBytes(func.getOperation(), population);
}
inline int64_t committedBytes(func::FuncOp func, Population population) {
  return committedBytes(func.getOperation(), population);
}
inline int64_t roomBytes(func::FuncOp func, int64_t budget,
                         int64_t released = 0) {
  return roomBytes(func.getOperation(), budget, released);
}

} // namespace vtcm
} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXVTCMLEDGER_H
