//===-- HmxDiagnosticRecordPass.cpp - diagnostics entry point --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Independent entry point for the two HMX diagnostic passes.
//
// The VTCM allocation census (`hmx-vtcm-accounting`) and the record-only v3
// document (`hmx-v3-record`) used to be mounted unconditionally inside the
// production `linalg-to-llvm` pass. They are no longer: the production
// pipeline carries no diagnostic pass, and this library -- which holds them --
// is linked only into `linalg-hexagon-opt`, so no diagnostic pass code reaches
// the backend library or the device.
//
// What replaces the mount is this pass. It builds the exact production
// sequence (`hexagon::addLinalgToLLVMPasses`) and then appends the two
// diagnostic passes at the stage where they belong: module scope, after the
// placement rewrites, while VTCM allocations and the sidecars they produce are
// still explicit. The stage is the load-bearing part -- the census inspects
// `hexagonmem.alloc`s that no longer exist after `hexagonmem-to-llvm`, and the
// record folds in requested-byte facts the census sidecars prove -- which is
// why the diagnostics cannot simply be sequenced after `linalg-to-llvm` on a
// command line.
//
// The production options are forwarded verbatim in `linalg-to-llvm`'s own
// key=value spelling, e.g.
//
//   linalg-hexagon-opt %s -pass-pipeline='builtin.module(
//     hmx-diagnostic-record{production=enable-workspace-resident=false})'
//
// so a diagnostic run is the production run plus the two passes, and no option
// is duplicated as a second surface.
//
// `stop-after-diagnostic-stage` ends the production sequence right after the
// stage (census plus the marker-gated record document that follows it), so the
// run stops at the placement layer. That is the layered-lit entry: the same
// production order, one layer of it, with the allocations and hmx ops still
// explicit -- and it is reachable only from here, which is why no stop flag
// exists on `linalg-to-llvm` and none reaches the backend library.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/Hmx/Transforms/Passes.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_HMXDIAGNOSTICRECORD
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

using namespace mlir;
using namespace mlir::hmx;

namespace {

struct HmxDiagnosticRecordPass
    : public PassWrapper<HmxDiagnosticRecordPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HmxDiagnosticRecordPass)

  HmxDiagnosticRecordPass() = default;
  HmxDiagnosticRecordPass(const HmxDiagnosticRecordPass &other)
      : PassWrapper(other) {
    production = other.production;
    stopAfterDiagnosticStage = other.stopAfterDiagnosticStage;
  }

  StringRef getArgument() const final { return "hmx-diagnostic-record"; }

  StringRef getDescription() const final {
    return "Run the production pipeline with the HMX diagnostics attached";
  }

  // The nested pipeline below runs with the context already marked
  // multi-threaded, which is when a context refuses to load a dialect. So this
  // pass declares them and the *outer* manager loads them before anything runs.
  void getDependentDialects(DialectRegistry &registry) const override {
    hexagon::addLinalgToLLVMDependentDialects(registry);
  }

  void runOnOperation() override {
    auto module = cast<ModuleOp>(getOperation());

    // The census is handed to the production pass, not appended after it: it
    // inspects VTCM allocations and the record document folds in facts the census
    // sidecars prove, so both must run while those still exist. The production
    // sequence mounts the record pass right after this stage, so injecting the
    // census here puts it in front of the document it feeds.
    LogicalResult run = hexagon::runLinalgToLLVMPipeline(
        getContext(), module, this->production, [](PassManager &pm) {
          pm.addPass(mlir::hmx::createHmxVtcmAccountingPass());
        }, this->stopAfterDiagnosticStage);

    if (failed(run)) {
      module.emitError("hmx-diagnostic-record: cannot run the production "
                       "pipeline with `production=")
          << this->production << "`";
      signalPassFailure();
    }
  }

  /// The production options in `linalg-to-llvm`'s own key=value spelling.
  /// Empty means the production defaults.
  Option<std::string> production{
      *this, "production",
      llvm::cl::desc(
          "options forwarded to the production linalg-to-llvm pipeline, in "
          "linalg-to-llvm's own key=value spelling (empty = the defaults)"),
      llvm::cl::init("")};

  /// End the production sequence right after the diagnostic stage instead of
  /// lowering through translation. The stage is the layer boundary: everything
  /// before it is the placement layer (VTCM allocations, hmx ops, manifest and
  /// diagnostic facts still explicit), everything after it is lowering that
  /// erases them. This is what a layered lit wants -- the same production
  /// order, one layer of it, no second copy of the sequence -- and it is why
  /// the option lives on this opt-tool-only entry point: no stop flag exists
  /// on `linalg-to-llvm` itself, so nothing that ships to the device can
  /// request it.
  ///
  ///   linalg-hexagon-opt %s -pass-pipeline='builtin.module(
  ///     hmx-diagnostic-record{stop-after-diagnostic-stage})'
  Option<bool> stopAfterDiagnosticStage{
      *this, "stop-after-diagnostic-stage",
      llvm::cl::desc("stop the production sequence after the diagnostic "
                     "stage (the placement layer) instead of lowering"),
      llvm::cl::init(false)};
};

} // namespace

std::unique_ptr<Pass> mlir::hmx::createHmxDiagnosticRecordPass() {
  return std::make_unique<HmxDiagnosticRecordPass>();
}
