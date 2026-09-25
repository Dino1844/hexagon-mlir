//===- hmx-vtcm-liveness-rejects.mlir - unsupported event shapes fail closed =//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hmx-vtcm-accounting)' -split-input-file -verify-diagnostics -allow-unregistered-dialect
//===----------------------------------------------------------------------===//

// expected-error @+1 {{hmx.diagnostic_vtcm_liveness requires hmx.diagnostic_vtcm_accounting}}
module attributes {hmx.diagnostic_vtcm_liveness} {
  func.func @liveness_requires_census() {
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: missing deallocation at function return}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @missing_deallocation() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: missing deallocation at function return}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  // A loop body that releases an allocation made *outside* the loop does not
  // form a balanced pair: a zero-trip loop never performs the release, so the
  // allocation is still live at the return.  The fixpoint joins the zero-trip
  // path with the body exit precisely so this cannot be reported as complete.
  func.func @loop_zero_trip_deallocation() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      memref.dealloc %a : memref<64x64xf16, 1>
      scf.yield
    }
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: allocation site is live across a back edge or a path merge}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  // A site allocated on a path that re-enters the allocating block is live
  // across a back edge, so its deallocations do not pair one-to-one with it.
  func.func @back_edge_carried_allocation() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    cf.br ^header
  ^header:
    %t = memref.alloc() : memref<8x8xf16, 1>
    %cond = arith.constant true
    cf.cond_br %cond, ^exit, ^header
  ^exit:
    memref.dealloc %t : memref<8x8xf16, 1>
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: call is unsupported}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func private @opaque()

  func.func @call_graph_unsupported() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    func.call @opaque() : () -> ()
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: asynchronous transfer is unsupported}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @async_stage_unsupported() {
    %src = memref.alloc() : memref<64x1024xf16>
    %row = arith.constant 0 : index
    %slot = memref.alloc() : memref<32x1024xf16, 1>
    %status = memref.alloc() : memref<1xi32>
    %token = hmx.stage ins(%src, %row : memref<64x1024xf16>)
        outs(%slot, %status : memref<32x1024xf16, 1>, memref<1xi32>) -> i32
    memref.dealloc %src : memref<64x1024xf16>
    memref.dealloc %slot : memref<32x1024xf16, 1>
    memref.dealloc %status : memref<1xi32>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: duplicate deallocation}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @double_deallocation() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    memref.dealloc %a : memref<64x64xf16, 1>
    memref.dealloc %a : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: ambiguous or aliased deallocation origin}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @ambiguous_deallocation_origin() {
    %a = memref.alloc() : memref<64x64xf16, 1>
    %view = memref.cast %a : memref<64x64xf16, 1> to memref<64x64xf16, 1>
    memref.dealloc %view : memref<64x64xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: ambiguous or aliased deallocation origin}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @memory_space_cast_deallocation() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %alias = memref.memory_space_cast %a
        : memref<16x16xf16, 1> to memref<16x16xf16>
    memref.dealloc %alias : memref<16x16xf16>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: function accepts an untracked VTCM value}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @external_vtcm_argument_unsupported(%arg: memref<64x64xf16, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = scf.for %i = %c0 to %c2 step %c1
        iter_args(%carry = %arg) -> (memref<64x64xf16, 1>) {
      scf.yield %carry : memref<64x64xf16, 1>
    }
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: region-carried value has ambiguous alias provenance}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  // The carried value is the tracked allocation on the taken path and a
  // different tracked allocation on the other, so no single site owns the
  // loop-carried handle and the ambiguity is reported instead of resolved.
  func.func @ambiguous_region_carry() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %b = memref.alloc() : memref<16x16xf16, 1>
    %cond = arith.constant true
    %result = scf.if %cond -> (memref<16x16xf16, 1>) {
      scf.yield %a : memref<16x16xf16, 1>
    } else {
      scf.yield %b : memref<16x16xf16, 1>
    }
    memref.dealloc %a : memref<16x16xf16, 1>
    memref.dealloc %b : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: call is unsupported}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  // A callee that hands back a VTCM memref is storage this analysis cannot
  // attribute to any site it tracks.  There is no reviewed call summary, so the
  // class is reported rather than the result being treated as a zero-cost
  // untracked handle.
  func.func private @vtcm_factory() -> memref<16x16xf16, 1>

  func.func @callee_returns_vtcm() {
    %produced = call @vtcm_factory() : () -> memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: an allocation site was not reached; allocation-site coverage incomplete}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  // Unreachable blocks are never evaluated, so their sites are never charged.
  // Coverage stays incomplete rather than the function being reported complete
  // with the unvisited bytes missing from the record.
  func.func @unreachable_allocation_site() {
    return
  ^unreachable:
    %a = memref.alloc() : memref<16x16xf16, 1>
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: VTCM allocation pointer extraction is unsupported}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @pointer_escape() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %pointer = memref.extract_aligned_pointer_as_index %a
        : memref<16x16xf16, 1> -> index
    %sink = memref.alloc() : memref<1xindex>
    %zero = arith.constant 0 : index
    memref.store %pointer, %sink[%zero] : memref<1xindex>
    memref.dealloc %a : memref<16x16xf16, 1>
    memref.dealloc %sink : memref<1xindex>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: VTCM allocation value is stored or aliased}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  memref.global "private" @handle_sink
      : memref<1xmemref<16x16xf16, 1>> = uninitialized
  func.func @global_memref_handle_escape() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %handles = memref.get_global @handle_sink
        : memref<1xmemref<16x16xf16, 1>>
    %zero = arith.constant 0 : index
    memref.store %a, %handles[%zero] : memref<1xmemref<16x16xf16, 1>>
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: VTCM global storage is unsupported}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  memref.global "private" @vtcm_global : memref<16x16xf16, 1> = uninitialized
  func.func @global_vtcm_source() {
    %global = memref.get_global @vtcm_global : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: ambiguous or aliased deallocation origin}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @view_deallocation() {
    %a = memref.alloc() : memref<512xi8, 1>
    %c0 = arith.constant 0 : index
    %view = memref.view %a[%c0][]
        : memref<512xi8, 1> to memref<256xi8, 1>
    memref.dealloc %view : memref<256xi8, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: allocation-derived value used after deallocation}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @view_use_after_deallocation() {
    %a = memref.alloc() : memref<512xi8, 1>
    %c0 = arith.constant 0 : index
    %view = memref.view %a[%c0][]
        : memref<512xi8, 1> to memref<256xi8, 1>
    memref.dealloc %a : memref<512xi8, 1>
    %value = memref.load %view[%c0] : memref<256xi8, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: ambiguous or aliased deallocation origin}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @subview_deallocation() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %subview = memref.subview %a[%c0, %c0][1, 1][1, 1]
        : memref<16x16xf16, 1> to
          memref<1x1xf16, strided<[16, 1], offset: ?>, 1>
    memref.dealloc %subview : memref<1x1xf16, strided<[16, 1], offset: ?>, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: asynchronous transfer is unsupported}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @async_await_unsupported() {
    %slot = memref.alloc() : memref<16x16xf16, 1>
    %token = arith.constant 0 : i32
    %ready = hmx.await ins(%token : i32)
        outs(%slot : memref<16x16xf16, 1>) -> memref<16x16xf16, 1>
    memref.dealloc %slot : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: asynchronous transfer is unsupported}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @dma_unsupported() {
    %src = memref.alloc() : memref<16x16xf16>
    %dst = memref.alloc() : memref<16x16xf16, 1>
    %tag = memref.alloc() : memref<1xi32>
    %zero = arith.constant 0 : index
    %bytes = arith.constant 256 : index
    memref.dma_start %src[%zero, %zero], %dst[%zero, %zero], %bytes, %tag[%zero]
        : memref<16x16xf16>, memref<16x16xf16, 1>, memref<1xi32>
    memref.dealloc %src : memref<16x16xf16>
    memref.dealloc %dst : memref<16x16xf16, 1>
    memref.dealloc %tag : memref<1xi32>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: unknown side effect or asynchronous operation}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @unknown_async_unsupported() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    "test.unknown_async"(%a) : (memref<16x16xf16, 1>) -> ()
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: missing deallocation at function return}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  // An execute_region that hands a *fresh* allocation outward keeps that site
  // live past the region, so it is not a balanced pair.  Handing outward an
  // alias of a site allocated outside is admitted; see
  // hmx-vtcm-liveness-cfg.mlir.
  func.func @execute_region_vtcm_result() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %result = scf.execute_region -> memref<16x16xf16, 1> {
      %inner = memref.alloc() : memref<16x16xf16, 1>
      scf.yield %inner : memref<16x16xf16, 1>
    }
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: call is unsupported}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @call_escape() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    func.call @vtcm_sink(%a) : (memref<16x16xf16, 1>) -> ()
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }

  func.func private @vtcm_sink(memref<16x16xf16, 1>)
}

// -----

// A duplicated release event is a dropped event: it must not be silently
// ignored, because ignoring it would let a function report a complete peak
// while an unaccounted release happened.
// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: duplicate deallocation}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @duplicated_deallocation_event() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %b = memref.alloc() : memref<16x16xf16, 1>
    memref.dealloc %a : memref<16x16xf16, 1>
    // `b` is still live here, and `a` has already been retired.
    memref.dealloc %b : memref<16x16xf16, 1>
    memref.dealloc %a : memref<16x16xf16, 1>
    return
  }
}

// -----

// A release on only one path is not a balanced pair; the merge keeps the site
// live and the return check reports it rather than assuming the other path ran.
// expected-error @+1 {{HMX VTCM structured liveness is not proven for this IR: missing deallocation at function return}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness} {
  func.func @conditional_deallocation() {
    %a = memref.alloc() : memref<16x16xf16, 1>
    %cond = arith.constant true
    scf.if %cond {
      memref.dealloc %a : memref<16x16xf16, 1>
    }
    return
  }
}

// -----

// An incomplete analysis context is rejected before any fact is published: the
// liveness probe is not a standalone switch.
// expected-error @+1 {{hmx.diagnostic_vtcm_liveness requires hmx.diagnostic_vtcm_accounting}}
module attributes {hmx.diagnostic_vtcm_liveness} {
  func.func @liveness_without_census() {
    return
  }
}

// -----

// A non-unit marker is malformed rather than absent, and must not be read as
// an enabled probe.
// expected-error @+1 {{hmx.diagnostic_vtcm_liveness must be a unit attribute}}
module attributes {hmx.diagnostic_vtcm_accounting,
                    hmx.diagnostic_vtcm_liveness = "yes"} {
  func.func @malformed_liveness_marker() {
    return
  }
}

// -----

// A dynamic extent with a proven constant bound is accounted exactly, so it is
// no longer a negative case; see hmx-vtcm-accounting-dynamic-extent.mlir.
