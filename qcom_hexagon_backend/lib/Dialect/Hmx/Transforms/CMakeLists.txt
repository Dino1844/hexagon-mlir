add_mlir_dialect_library(HmxTransforms
  BufferizableOpInterfaceImpl.cpp
  HmxManifest.cpp
  HmxRecordV3.cpp
  HmxRecordV3Pass.cpp
  MatmulToHmxPass.cpp
  HmxPartitionPass.cpp
  WeightResidentPass.cpp
  HmxWorkspaceResidentPass.cpp
  HmxVtcmAccountingPass.cpp

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
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
