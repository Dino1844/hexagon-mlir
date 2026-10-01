//===- matmul-to-hmx-remark.mlir - why an f16 matmul was skipped ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// An f16 matmul that misses the engine's contract is a near miss, and the pass
// explains it. Every case here has operands the engine could take and an empty
// initialiser, so each one fails exactly one condition.
//
// Besides the per-op remark the pass aggregates one module-level warning per
// run: the production pipeline never shows remarks, so that warning is what a
// silently refused HMX path looks like there. It quotes the counts and the
// first refusal. Each module below is spelled explicitly so the warning has a
// line of its own to attach to (an implicit module gets a line-0 location).
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-budget=10000}))' -verify-diagnostics -split-input-file
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{vtcm-allocator=false}))' -split-input-file 2>&1 | FileCheck %s --check-prefix=ENV
//
// With the VTCM allocator off the environment refusal fires before any shape
// question and leads the warning with the switch that closed the whole path:
// ENV: HMX disabled: this pipeline has no VTCM allocator (enableConvertToHexagonmem is off)
//===----------------------------------------------------------------------===//

// expected-warning @+1 {{HMX: 1 matmul(s) skipped; first refusal: HMX not applied: needs 2D static shapes, M/N/K multiples of 32, M > 4 [candidate=hmx-tail, padded=(64, 64, 128), full=(64, 64, 96), tail=(0, 0, 4)] [reason=tile-alignment]; matmul M=64, N=64, K=100, lhsElem='f16', rhsElem='f16', outElem='f16'}}
module {
func.func @k_not_aligned(%a: tensor<64x100xf16>, %b: tensor<100x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  // expected-remark @+1 {{HMX not applied: needs 2D static shapes, M/N/K multiples of 32, M > 4 [candidate=hmx-tail, padded=(64, 64, 128), full=(64, 64, 96), tail=(0, 0, 4)] [reason=tile-alignment]}}
  %0 = linalg.matmul ins(%a, %b : tensor<64x100xf16>, tensor<100x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
}

// -----

// expected-warning @+1 {{first refusal: HMX not applied: needs 2D static shapes, M/N/K multiples of 32, M > 4 [candidate=hmx-tail, padded=(64, 64, 128), full=(64, 32, 128), tail=(0, 16, 0)] [reason=tile-alignment]; matmul M=64, N=48, K=128}}
module {
func.func @n_not_aligned(%a: tensor<64x128xf16>, %b: tensor<128x48xf16>) -> tensor<64x48xf16> {
  %c = tensor.empty() : tensor<64x48xf16>
  // expected-remark @+1 {{HMX not applied: needs 2D static shapes, M/N/K multiples of 32, M > 4 [candidate=hmx-tail, padded=(64, 64, 128), full=(64, 32, 128), tail=(0, 16, 0)] [reason=tile-alignment]}}
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x48xf16>) outs(%c : tensor<64x48xf16>) -> tensor<64x48xf16>
  return %0 : tensor<64x48xf16>
}
}

// -----

// Too few rows to form a tile.
// expected-warning @+1 {{first refusal: HMX not applied: needs 2D static shapes, M/N/K multiples of 32, M > 4 [reason=min-rows]; matmul M=2, N=64, K=64}}
module {
func.func @too_few_rows(%a: tensor<2x64xf16>, %b: tensor<64x64xf16>) -> tensor<2x64xf16> {
  %c = tensor.empty() : tensor<2x64xf16>
  // expected-remark @+1 {{M > 4 [reason=min-rows]}}
  %0 = linalg.matmul ins(%a, %b : tensor<2x64xf16>, tensor<64x64xf16>) outs(%c : tensor<2x64xf16>) -> tensor<2x64xf16>
  return %0 : tensor<2x64xf16>
}
}

// -----

// A bridge that does not fit the remaining VTCM budget is left alone, loudly:
// a 64x64x64 bridge is 24576 bytes and its smallest M block (one 32-row tile,
// 16384 bytes) is still over the budget below, so no block fits and the op is
// refused rather than blocked.
// expected-warning @+1 {{HMX: 1 matmul(s) skipped; first refusal: HMX not applied: matmul (M=64, N=64, K=64, lhsElem='f16', rhsElem='f16', outElem='f16', vtcmUsed=0) bridge footprint does not fit remaining VTCM (vtcmBudget=10000 bytes) [reason=vtcm-budget]; matmul M=64, N=64, K=64, lhsElem='f16', rhsElem='f16', outElem='f16'}}
module {
func.func @over_budget(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  // expected-remark @+1 {{HMX not applied: matmul (M=64, N=64, K=64, lhsElem='f16', rhsElem='f16', outElem='f16', vtcmUsed=0) bridge footprint does not fit remaining VTCM (vtcmBudget=10000 bytes) [reason=vtcm-budget]}}
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
}

// -----

// A batched contraction is refused, and says so. It used to be the one refusal
// with no diagnostic at all: `NonRank2` sat in the `llvm_unreachable` arm of
// `renderRefusal` and was absent from `reportsRefusal`, so the manifest record
// was the only channel -- and the manifest is machine-facing. That silence is
// what made scoping this limitation expensive; see the note in the pass.
//
// The reason text is the point of the case, not just the `[reason=non-rank-2]`
// tag: a bare "not applied" does not tell a user whether collapsing the batch is
// safe, and for `linalg.batch_matmul` it is not (the weight is per-batch).
// expected-warning @+1 {{first refusal: HMX not applied: not a rank-2 contraction. The HMX mma is rank-2, and a batch dimension is a scheduling fact of the host loop rather than of the mma. Collapsing the batch into M is only correct when the weight is shared across the batch; linalg.batch_matmul carries a per-batch weight, so that needs a batch loop instead, which in turn drops weight residency because a resident weight must be rank-2 [reason=non-rank-2]}}
module {
func.func @batched_is_refused_loudly(%a: tensor<2x64x128xf16>, %b: tensor<2x128x64xf16>) -> tensor<2x64x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %init = tensor.empty() : tensor<2x64x64xf16>
  %c = linalg.fill ins(%zero : f16) outs(%init : tensor<2x64x64xf16>) -> tensor<2x64x64xf16>
  // expected-remark @+1 {{HMX not applied: not a rank-2 contraction. The HMX mma is rank-2}}
  %0 = linalg.batch_matmul ins(%a, %b : tensor<2x64x128xf16>, tensor<2x128x64xf16>)
                                outs(%c : tensor<2x64x64xf16>) -> tensor<2x64x64xf16>
  return %0 : tensor<2x64x64xf16>
}
}

// -----

// A dtype the engine does not admit, refused loudly. This and the dynamic-shape
// case below were both silent until 2026-09-30: the manifest carried a correct
// record (`reason = "unsupported-dtype"`) and the compiler printed nothing, so a
// user who wrote `bf16` had no way to learn that the engine admits f16/f32 only.
// The rule is "a refusal is loud unless the party that made the decision is the
// party that already knows" -- and the user picked this dtype, not us.
//
// The message is dtype-only on purpose. It used to share a case with the shape
// refusals and so told a perfectly-shaped bf16 matmul that it needed "2D static
// shapes, M/N/K multiples of 32, M > 4" -- three conditions it already met.
// expected-warning @+1 {{first refusal: HMX not applied: needs f16/f32 inputs and an f16/f32 result; the engine's mma has no bf16 or f64 form; got lhs='bf16', rhs='bf16', out='bf16' [reason=unsupported-dtype]}}
module {
func.func @bf16_is_refused_loudly(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %c = tensor.empty() : tensor<64x64xbf16>
  // expected-remark @+1 {{HMX not applied: needs f16/f32 inputs and an f16/f32 result; the engine's mma has no bf16 or f64 form; got lhs='bf16', rhs='bf16', out='bf16' [reason=unsupported-dtype]}}
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%c : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}
}

// -----

// A dynamic shape is the user's shape, so it is loud for the same reason. Note
// the dynamic extent needs a real operand: `tensor.empty() : tensor<?x64xf16>`
// does not parse ("incorrect number of dynamic sizes"), and a broken test input
// reads exactly like a silent refusal, which is how this case first looked
// unaffected.
// expected-warning @+1 {{first refusal: HMX not applied: the shape is not static, so the bridge cannot pick a tile grid; specialise the extents (or pad to a multiple of 32) to reach HMX [reason=dynamic-shape]}}
module {
func.func @dynamic_is_refused_loudly(%a: tensor<?x64xf16>, %b: tensor<?x64xf16>, %d: index) -> tensor<?x64xf16> {
  %c = tensor.empty(%d) : tensor<?x64xf16>
  // expected-remark @+1 {{HMX not applied: the shape is not static, so the bridge cannot pick a tile grid; specialise the extents (or pad to a multiple of 32) to reach HMX [reason=dynamic-shape]}}
  %0 = linalg.matmul ins(%a, %b : tensor<?x64xf16>, tensor<?x64xf16>) outs(%c : tensor<?x64xf16>) -> tensor<?x64xf16>
  return %0 : tensor<?x64xf16>
}
}
