//===-- HmxVtcmLedger.cpp - the one VTCM byte ledger ----------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// See HmxVtcmLedger.h for why the walks live in one place. The bodies below
// are the five callers' original walks, unchanged: behaviour equivalence is
// the acceptance criterion for this file, so the arithmetic (no overflow
// checks, `bitWidth / 8` truncation and all) is preserved verbatim rather than
// improved. A stricter sum belongs in the accounting pass, which validates
// what it publishes.
//
//===----------------------------------------------------------------------===//

#include "HmxVtcmLedger.h"

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
namespace vtcm {

namespace {

/// The module a budget question is about: the declaration lives there, and the
/// anchor operation may be either the module itself or something inside it.
ModuleOp moduleOf(Operation *scope) {
  if (auto module = dyn_cast<ModuleOp>(scope))
    return module;
  return scope->getParentOfType<ModuleOp>();
}

/// The function a budget question is about, accepting the function itself as
/// well as an operation inside it (`getParentOfType` from the function would
/// answer null for the function).
func::FuncOp funcOf(Operation *within) {
  if (auto func = dyn_cast<func::FuncOp>(within))
    return func;
  return within->getParentOfType<func::FuncOp>();
}

/// `hmx.weight_resident_bytes`: the aggregate `weight-resident` publishes for
/// the whole kernel. Resident weights are `hexagonmem.alloc`s (or already
/// lowered), never `memref.alloc`s, so this declaration is the only place their
/// footprint is visible to the memref walk.
constexpr const char *kWeightResidentBytesAttr = "hmx.weight_resident_bytes";

/// Tensor-level population: the crouton arrays previous attributions in this
/// function already committed (`matmul-to-hmx` runs before bufferization, so
/// they are `hmx.alloc_crouton`s). Deliberately a query, not an allocator:
/// placement stays with the space-1 machinery.
int64_t tensorWalk(Operation *within) {
  func::FuncOp func = funcOf(within);
  if (!func)
    return 0;
  int64_t bytes = 0;
  func.walk([&](AllocCroutonOp alloc) {
    auto type = dyn_cast<RankedTensorType>(alloc.getResult().getType());
    if (!type || !type.hasStaticShape())
      return;
    bytes += type.getNumElements() * (type.getElementTypeBitWidth() / 8);
  });
  return bytes;
}

/// Memref-level population: the static space-1 arrays the pipeline stages,
/// which is what the budget checks after bufferization see.
int64_t memrefWalk(Operation *within) {
  func::FuncOp func = funcOf(within);
  if (!func)
    return 0;
  int64_t bytes = 0;
  func.walk([&](memref::AllocOp alloc) {
    auto type = dyn_cast<MemRefType>(alloc.getType());
    if (!type || !type.hasStaticShape() ||
        type.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
      return;
    Type elem = type.getElementType();
    if (!elem.isIntOrFloat())
      return;
    bytes += type.getNumElements() * (elem.getIntOrFloatBitWidth() / 8);
  });
  return bytes;
}

} // namespace

int64_t transientBytes(Operation *within, Population population) {
  return population == Population::Tensor ? tensorWalk(within)
                                           : memrefWalk(within);
}

int64_t residentBytes(Operation *scope) {
  ModuleOp module = moduleOf(scope);
  if (!module)
    return 0;
  if (auto resident =
          module->getAttrOfType<IntegerAttr>(kWeightResidentBytesAttr))
    return resident.getInt();
  return 0;
}

int64_t committedBytes(Operation *within, Population population) {
  int64_t bytes = transientBytes(within, population);
  // The declaration is a memref-level fact: it is written after bufferization
  // and, at tensor level, the weight it pays for is still an `hmx.alloc_crouton`
  // in the walk above -- adding it there would charge the same weight twice.
  if (population == Population::Memref)
    bytes += residentBytes(within);
  return bytes;
}

int64_t roomBytes(Operation *within, int64_t budget, int64_t released) {
  return budget - committedBytes(within, Population::Memref) + released;
}

} // namespace vtcm
} // namespace hmx
} // namespace mlir
