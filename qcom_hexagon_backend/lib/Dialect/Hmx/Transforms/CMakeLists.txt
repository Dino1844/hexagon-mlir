add_mlir_dialect_library(HmxTransforms
  BufferizableOpInterfaceImpl.cpp
  # Two libraries share this directory (see HmxDiagnostics below), so neither
  # lists every source here. LLVM's source-list check accepts that when the
  # targets that omit files say so.
  PARTIAL_SOURCES_INTENDED
  HmxManifest.cpp
  HmxRecordV3.cpp
  HmxRecordV3Pass.cpp
  MatmulToHmxPass.cpp
  ThreadRolePartition.cpp
  HmxPartitionPass.cpp
  WeightResidentPass.cpp
  HmxVectorReadoutPass.cpp
  HmxWorkspaceResidentPass.cpp
  HmxVtcmLedger.cpp

  DEPENDS
  HmxTransformsPassIncGen

  LINK_LIBS PUBLIC
  HexagonCommon
  HexagonMem
  Hmx
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)

# The HMX diagnostic census and its independent entry point, in their own library
# so the production pipeline cannot link them.
#
# `linalg-to-llvm` used to mount `hmx-vtcm-accounting` unconditionally; it no
# longer mounts it at all. The census is now reachable only from
# `hmx-diagnostic-record`, which builds the production sequence and appends it at
# the stage it belongs. Because that entry lives here, and because this library is
# linked only into `linalg-hexagon-opt`, the backend library -- and therefore
# everything that ships to the device -- carries no census code.
#
# `hmx-v3-record` (the 60-line finaliser of the record document) stays in
# HmxTransforms on purpose: the document it finalises is published by
# `matmul-to-hmx`, so its library (`HmxRecordV3.cpp`) is already in the production
# library, and the host contract tests receive the finished document through
# `translate_linalg_to_obj`. It runs at the same stage the census used to run at,
# so a diagnostic run still gets the census sidecars first.
add_mlir_dialect_library(HmxDiagnostics
  HmxDiagnosticRecordPass.cpp
  HmxVtcmAccountingPass.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  HmxTransformsPassIncGen

  LINK_LIBS PUBLIC
  HexagonCommon
  HexagonMem
  Hmx
  LinalgToLLVM
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
