//===- hmx-partition-crontons-per-mma-reject.mlir - batch range is a domain --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// `croutons-per-mma` has a domain, {0} u [1, 32], and this file is what holds
// the boundary: a value outside it is an error, not a clamp.
//
// The clamp is the tempting alternative and it is wrong here for a reason worth
// writing down. 32 is a hardware bound -- the engine's K repeat field `Rt[dC]`
// is five bits (V81 PRM 4.2.1), which is also the verifier's `n_croutons <= 32`
// (HmxOps.cpp:410-411) -- so 64 is not a deeper request, it is an unencodable
// one. Clamping it to 32 would report success for a batch the caller never
// asked for, and the caller would read the run as evidence for the number they
// typed. That is the shape `AGENTS.md` §7 rule 8 calls "a knob that looks
// effective and is not", and it is worse than an error: the error is one line of
// the log, the clamp is invisible.
//
// 33 is here as well as 64 because 33 is the case a wrong bound hides: a domain
// written `< 32` instead of `<= 32` accepts 33, and nothing else in the tree
// would notice. That is the same reason `hmx-partition-deep-crontons.mlir`
// carries 31/32/33 rather than one round number.
//
// Named gap, so it is not mistaken for coverage: the pass also rejects a
// negative value (the domain is {0} u [1, 32], and a count has no meaning below
// zero), but there is no run line for it. No test anywhere in this tree passes a
// negative number to a pass option, so the `{croutons-per-mma=-1}` spelling may
// be a pipeline-parser error rather than reaching the pass, and a lit that
// guessed wrong would be red for a reason that has nothing to do with the
// domain. Unverified until someone runs it; the code guard is in `runOnOperation`.
//
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{croutons-per-mma=64}))' 2>&1 | FileCheck %s --check-prefix=TOOBIG
// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hmx-partition{croutons-per-mma=33}))' 2>&1 | FileCheck %s --check-prefix=OFFBYONE
//===----------------------------------------------------------------------===//

// Matched from the message, without the `error: ` prefix, the way the other
// negative tests here do: the prefix and the source location are the driver's
// formatting, the message is the contract.
// Matched from the message, without the `error: ` prefix and the source location,
// the way the other negative tests here do: the prefix and the location are the
// driver's formatting, the message is the contract.
// TOOBIG: hmx-partition croutons-per-mma must be 0 (the hardware maximum, 32) or in [1, 32], but got 64
// OFFBYONE: hmx-partition croutons-per-mma must be 0 (the hardware maximum, 32) or in [1, 32], but got 33

func.func @kt32() {
  %a = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  %w = memref.alloc() : memref<2x32x16x32x2xf16, 1>
  %r = memref.alloc() : memref<2x2x16x32x2xf16, 1>
  hmx.matmul ins(%a, %w : memref<2x32x16x32x2xf16, 1>, memref<2x32x16x32x2xf16, 1>)
             outs(%r : memref<2x2x16x32x2xf16, 1>)
  return
}
