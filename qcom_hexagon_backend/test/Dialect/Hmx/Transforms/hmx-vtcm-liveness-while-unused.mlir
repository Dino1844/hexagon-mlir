//===- hmx-vtcm-liveness-while-unused.mlir - while carried-state fixpoint -===//
//
// A VTCM value carried through `scf.condition` remains live even when the
// after-region block argument is unused. The before-condition exit and the
// enclosing zero-trip state must be joined; dropping that state would make
// this leak look complete.
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' -verify-diagnostics
//===----------------------------------------------------------------------===//

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: ambiguous or aliased deallocation origin}}
module @while_unused attributes {hmx.diagnostic_vtcm_accounting,
                                  hmx.diagnostic_vtcm_liveness} {
  func.func @while_carried_vtcm_unused_after() {
    %dummy = memref.alloc() : memref<16x16xf16, 1>
    %true = arith.constant true
    %result = scf.while (%carry = %dummy) : (memref<16x16xf16, 1>) -> (memref<16x16xf16, 1>) {
      %leaked = memref.alloc() : memref<16x16xf16, 1> loc("while":4:7)
      scf.condition(%true) %leaked : memref<16x16xf16, 1>
    } do {
    ^bb0(%arg: memref<16x16xf16, 1>):
      // Deliberately do not use %arg. The yielded value is a different object;
      // the carried value must not disappear from the fixpoint.
      scf.yield %dummy : memref<16x16xf16, 1>
    }
    memref.dealloc %result : memref<16x16xf16, 1>
    return
  }
}
