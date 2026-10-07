//===- maxnum-legalize-pipeline.mlir - enable-maxnum-legalize wiring ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Pins the `enable-maxnum-legalize` wiring of the LinalgToLLVM pipeline
// (LinalgToLLVMPass.cpp: the nested hvx-maxnum-legalize pass is inserted only
// when the option is on). The pass reports itself with an always-on remark, so
// both arms observe insertion vs non-insertion directly. The rewrite semantics
// themselves are pinned by maxnum-legalize.mlir (unit, -hvx-maxnum-legalize).
//
// Two RUN lines, one per arm, like every other pipeline test in this tree: each
// RUN line is an independent lit test (test/lit.cfg.py:24 uses
// lit.formats.ShTest), so a failure names the arm that broke. An earlier version
// folded both arms into one `... && ...` line on the stated grounds that "the
// workspace's manual lit runner executes only the first RUN line of a file".
// That is false: tools/hexmlir/run_lit_manual.sh collects and runs every RUN
// line (its `collect_runs` + `for runline in` loop).
//
// RUN: linalg-hexagon-opt %s -linalg-to-llvm 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: linalg-hexagon-opt %s -linalg-to-llvm="enable-maxnum-legalize=true" 2>&1 | FileCheck %s --check-prefix=ON
//
// OFF (default): the pass must not run at all.
// OFF-NOT: hvx-maxnum-legalize
//
// With the option on, exactly one vector maxnumf candidate is seen. NB:
// FileCheck treats every occurrence of the ON prefix followed by a colon as a
// directive (that is how the remark line below is checked), so no prose line may
// spell that sequence out.
//
// EXPECTED-CHANGE (2026-09-30): this line used to end `skip=0 limit=-1`. The
// pass's three bisection knobs were removed, so the remark no longer reports
// them. The assertion's actual job -- "the pass ran and saw N sites" -- is
// unchanged, and `candidates=1` is what carries it.
// ON: remark: hvx-maxnum-legalize: candidates=1
func.func @wiring(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %r = arith.maxnumf %a, %b : vector<32xf32>
  return %r : vector<32xf32>
}
