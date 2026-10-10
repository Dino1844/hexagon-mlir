//===-- HexagonMemOps.cpp - HexagonMem dialect ops ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the HexagonMem dialect operations.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::hexagonmem;

namespace {
constexpr uint64_t kMaxAllocationAlignment = 2048;

/// Keep the IR allocation contract identical to the runtime allocator's
/// address-placement contract.  Zero, non-powers-of-two, and values above the
/// 2048-byte hardware quantum cannot produce a valid aligned descriptor; they
/// must fail before lowering rather than reaching a CHECK/abort in the device
/// runtime.  This is a contract check, not a claim that an arbitrary runtime
/// pointer is statically known to be non-null.
bool isSupportedAllocationAlignment(uint64_t alignment) {
  return alignment != 0 && alignment <= kMaxAllocationAlignment &&
         (alignment & (alignment - 1)) == 0;
}
} // namespace

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void HexagonMemDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemOps.cpp.inc"
      >();
}

LogicalResult AllocOp::verify() {
  auto type = getBuffer().getType();
  if (auto memRefType = mlir::dyn_cast<MemRefType>(type)) {
    if (!isSupportedAllocationAlignment(getAlignment()))
      return emitOpError(
          "alignment must be a power of two in [1, 2048]");
    // A resident-weight allocation is static but carries its source address as
    // one extra operand: the runtime needs that address as the residency key,
    // and unlike a dynamic size it does not participate in the shape (the
    // lowering builds the descriptor from the static type). This is the only
    // op form that may name a dynamic-arity operand it does not consume.
    int64_t expectedDims = memRefType.getNumDynamicDims();
    if (auto resident =
            (*this)->getAttrOfType<DictionaryAttr>(hmx::kHmxWeightResidentAttr))
      if (resident.get("address"))
        expectedDims += 1;
    if (static_cast<int64_t>(getDynamicSizes().size()) != expectedDims)
      return emitOpError("dimension operand count does not equal memref "
                         "dynamic dimension count");
  }
  return success();
}


//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemOps.cpp.inc"
