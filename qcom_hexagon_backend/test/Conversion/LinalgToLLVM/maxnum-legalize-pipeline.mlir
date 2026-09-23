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
// One RUN line carries both arms on purpose: the workspace's manual lit runner
// executes only the first RUN line of a file.
//
// RUN: linalg-hexagon-opt %s -linalg-to-llvm 2>&1 | FileCheck %s --check-prefix=OFF && linalg-hexagon-opt %s -linalg-to-llvm="enable-maxnum-legalize=true" 2>&1 | FileCheck %s --check-prefix=ON
//
// OFF (default): the pass must not run at all.
// OFF-NOT: hvx-maxnum-legalize
//
// With the option on, exactly one vector maxnumf candidate is seen, at the
// default bisection settings (rewrite all: limit=-1, skip=0). NB: FileCheck
// treats every occurrence of the ON prefix followed by a colon as a directive
// (that is how the remark line below is checked), so no prose line may spell
// that sequence out.
// ON: remark: hvx-maxnum-legalize: candidates=1 skip=0 limit=-1
func.func @wiring(%a: vector<32xf32>, %b: vector<32xf32>) -> vector<32xf32> {
  %r = arith.maxnumf %a, %b : vector<32xf32>
  return %r : vector<32xf32>
}
