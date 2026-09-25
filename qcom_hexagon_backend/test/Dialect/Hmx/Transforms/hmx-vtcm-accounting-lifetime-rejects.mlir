//===- hmx-vtcm-accounting-lifetime-rejects.mlir - census safety gates --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' -split-input-file -verify-diagnostics
//===----------------------------------------------------------------------===//

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @resident_deallocation() {
    %a = memref.alloc() {hmx.workspace_resident = {key = 1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @resident_escape() -> memref<64x64xf16, 1> {
    %a = memref.alloc() {hmx.workspace_resident = {key = 2 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    return %a : memref<64x64xf16, 1>
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @resident_loop_deallocation() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %a = memref.alloc() {hmx.workspace_resident = {key = 4 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%carry = %a) -> (memref<64x64xf16, 1>) {
      memref.dealloc %carry : memref<64x64xf16, 1>
      scf.yield %carry : memref<64x64xf16, 1>
    }
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @vtcm_alloca() {
    %a = memref.alloca() : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @pre_bufferization_vtcm_tensor() {
    %a = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<64x64xf16>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @external_scratch(%scratch: memref<64xi8, 1> {hexagon.scratch}) {
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.weight_resident_bytes = "bad"} {
  func.func @bad_weight_aggregate() {
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @missing_weight_aggregate() {
    %a = hexagonmem.alloc() {hmx.weight_resident = {global = @weight, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @dual_resident_tags() {
    %a = memref.alloc() {hmx.weight_resident = {global = @weight, bytes = 8192 : i64}, hmx.workspace_resident = {key = 3 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func private @opaque_sink(memref<64x64xf16, 1>)

  func.func @resident_call_escape() {
    %a = memref.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    func.call @opaque_sink(%a) : (memref<64x64xf16, 1>) -> ()
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @resident_cfg_escape() {
    %a = memref.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    cf.br ^next(%a : memref<64x64xf16, 1>)
  ^next(%b: memref<64x64xf16, 1>):
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @resident_tag_on_ddr() {
    %a = memref.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @resident_tag_on_ddr_hexagonmem() {
    %a = hexagonmem.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @custom_memory_space() {
    %a = memref.alloc() : memref<64x64xf16, "custom">
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.weight_resident_bytes = 8192 : i64} {
  func.func @unresolved_weight_global() {
    %a = hexagonmem.alloc() {hmx.weight_resident = {global = @missing_weight, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.weight_resident_bytes = 8192 : i64} {
  func.func @unproven_weight_address() {
    %address = arith.constant 0 : index
    %a = hexagonmem.alloc(%address) {hmx.weight_resident = {address, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @duplicate_workspace_keys() {
    %a = memref.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    %b = memref.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @noncanonical_workspace_key() {
    %a = memref.alloc() {hmx.workspace_resident = {key = 1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  llvm.func @sink(i64)

  func.func @resident_llvm_call_escape() {
    %a = memref.alloc() {hmx.workspace_resident = {key = -1 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    %address = memref.extract_aligned_pointer_as_index %a
        : memref<64x64xf16, 1> -> index
    %address64 = arith.index_cast %address : index to i64
    llvm.call @sink(%address64) : (i64) -> ()
    return
  }
}

// -----

// A resident allocation is a process-lifetime fact only when it is created
// unconditionally in the function entry block.  A conditional allocation
// cannot be counted as a floor for every invocation.
// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @conditional_resident() {
    %cond = arith.constant true
    scf.if %cond {
      %a = memref.alloc() {hmx.workspace_resident = {key = -5 : i64, bytes = 8192 : i64}}
          : memref<64x64xf16, 1>
    }
    return
  }
}

// -----

// Carrying a resident value through a loop makes ownership and process
// lifetime ambiguous even if the body never explicitly frees it.
// expected-error @+1 {{HMX VTCM accounting is incomplete: static allocation bytes or resident-byte provenance could not be proven}}
module attributes {hmx.diagnostic_vtcm_accounting} {
  func.func @resident_loop_carry() {
    %a = memref.alloc() {hmx.workspace_resident = {key = -6 : i64, bytes = 8192 : i64}}
        : memref<64x64xf16, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = scf.for %i = %c0 to %c2 step %c1
        iter_args(%carry = %a) -> (memref<64x64xf16, 1>) {
      scf.yield %carry : memref<64x64xf16, 1>
    }
    return
  }
}

// -----

// expected-error @+1 {{hmx.diagnostic_vtcm_accounting must be a unit attribute}}
module attributes {hmx.diagnostic_vtcm_accounting = "bad"} {
  func.func @bad_marker() {
    return
  }
}
