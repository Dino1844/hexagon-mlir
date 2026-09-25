//===-- HmxRecordV3Pass.cpp - finalize the record-only v3 document -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Finalization pass for `hex.hmx.kernel_manifest/v3`.
//
// It runs at module scope, right after the P1.5 VTCM accounting sidecars, and
// is inert unless the module carries the internal record-mode marker.  With the
// marker absent this pass does nothing at all, so the v2 manifest, its
// fingerprint and every launcher decision are untouched.
//
// Its whole job is to turn the compile-time record skeletons into a *published*
// document: fold in the requested-byte facts the P1.5 sidecars can prove,
// recompute each record fingerprint, and validate the closed contract.  A
// document it cannot prove is a failure, not a reduced document -- silently
// dropping a proof status would turn "not measured" into "not there".
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hmx/Transforms/HmxRecordV3.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_HMXRECORDV3
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

using namespace mlir;
using namespace mlir::hmx;

namespace {

struct HmxRecordV3Pass
    : public hmx::impl::HmxRecordV3Base<HmxRecordV3Pass> {
  void runOnOperation() override {
    // `finalizeHmxRecordV3` owns the marker check, the inert case, the
    // enrichment and the validation, so the pass cannot drift from the
    // publication boundary it feeds.
    if (failed(finalizeHmxRecordV3(cast<ModuleOp>(getOperation()))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hmx::createHmxRecordV3Pass() {
  return std::make_unique<HmxRecordV3Pass>();
}