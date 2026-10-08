# The channel header bin/runtime/include/HmxRoleChannel.h is included by
# HmxToLLVMPass.cpp: it names the exported section/depth symbols
# (<engine>__hmx_section, <engine>__hmx_role_depth) from the SAME single
# source the runtime's launch-side probe builds its dlsym names from, so the
# two spellings cannot drift (the tree's convention is compiler-side headers
# that RESTATE runtime facts -- HmxReadoutHandoff.h -- but a name the probe
# concatenates at runtime is exactly the fact that must live in ONE header,
# per ROADMAP1001 section 4.6's per-kernel-symbol decision).
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/runtime/include)

add_mlir_conversion_library(HmxToLLVM
  HmxToLLVMPass.cpp
  HmxExternalFnNames.cpp

  DEPENDS
  HmxToLLVMConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  Hmx
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
