// The kernel-level thread-role verdict, and the IR witness for each one.
//
// The five codes are not five arbitrary strings: four of the five are separated
// by a single question -- is producer layout work (pack) co-located with engine
// work, and if not, is the layout work co-located at all? Each case below is
// built to flip exactly one of those answers, so a change that widens or narrows
// a predicate shows up here as a changed string rather than as a silent new
// classification.
//
// `topology` is a module-level manifest field, so each case is its own module
// (`-split-input-file`); two HMX kernels in one module are a compile error, and
// the last case holds that down.
//
// The pass records only. Nothing is outlined and no attribute guides a later
// lowering, so every case must also leave the function body untouched.

// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(thread-role-partition))' | FileCheck %s

//===----------------------------------------------------------------------===//
// 1. Engine only: no layout work anywhere. An HMX thread is required, and
//    nothing can be handed to another thread.
//===----------------------------------------------------------------------===//

// CHECK:      module attributes {{.*}}thread_role_regions = 1 : i64{{.*}}topology = "topology-single-role-hmx"
module {
  func.func @engine_only(%bias: memref<256xi8, 1>, %a: memref<2x2x16x32x2xf16, 1>,
                         %m: index, %n: index) {
    hmx.bias_init %bias : memref<256xi8, 1>
    hmx.acc_clear
    hmx.acc_read %bias, %a, %m, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// 2. Layout only: HVX code with no engine work, so the HMX thread would sit
//    idle. (This is also the verdict for a kernel with no HMX work at all,
//    except that one records nothing -- see case 6.)
//===----------------------------------------------------------------------===//

// CHECK:      module attributes {{.*}}thread_role_regions = 1 : i64{{.*}}topology = "topology-single-role-hvx"
module {
  func.func @vector_only(%src: memref<2x2x16x32x2xf16, 1>,
                         %dst: memref<64x64xf16>, %r: index, %c: index) {
    hmx.unpack_acc ins(%src, %r, %c : memref<2x2x16x32x2xf16, 1>)
        outs(%dst : memref<64x64xf16>)
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// 3. Pack in its own loop, engine work in a later one. The pack loop runs to
//    completion before the engine work starts, so there is no per-tile producer
//    stream to overlap -- the shape `pipeline-depth=1` produces.
//===----------------------------------------------------------------------===//

// CHECK:      module attributes {{.*}}thread_role_regions = 2 : i64{{.*}}topology = "role-split-nopack"
module {
  func.func @mixed_disjoint(%bias: memref<256xi8, 1>, %a: memref<2x2x16x32x2xf16, 1>,
                            %s: memref<64x64xf16>, %r: index, %c: index,
                            %n: index, %m: index) {
    scf.for %i = %c to %n step %c {
      hmx.pack_act ins(%s, %r, %c : memref<64x64xf16>)
          outs(%a : memref<2x2x16x32x2xf16, 1>)
    }
    hmx.bias_init %bias : memref<256xi8, 1>
    hmx.acc_clear
    hmx.acc_read %bias, %a, %m, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// 4. The same two kinds of work, but the pack is inside the loop that issues
//    HMX. That is a per-tile producer stream -- while tile i is in the matrix
//    engine, another thread can build tile i+1 -- which is the shape
//    `pipeline-depth=2` produces, and the only one where a thread split pays.
//===----------------------------------------------------------------------===//

// CHECK:      module attributes {{.*}}thread_role_regions = 1 : i64{{.*}}topology = "role-split-ok"
module {
  func.func @mixed_per_tile(%bias: memref<256xi8, 1>, %a: memref<2x2x16x32x2xf16, 1>,
                            %s: memref<64x64xf16>, %r: index, %c: index,
                            %n: index, %m: index) {
    scf.for %i = %c to %n step %c {
      hmx.pack_act ins(%s, %r, %c : memref<64x64xf16>)
          outs(%a : memref<2x2x16x32x2xf16, 1>)
      hmx.acc_clear
      hmx.acc_read %bias, %a, %m, %i {bias_set = 0 : i32} : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// 5. Layout work co-located with engine work, but the layout work only consumes
//    what the engine op just wrote. An unpack cannot be hoisted ahead of the
//    mma, so there is a dependency and no producer stream.
//===----------------------------------------------------------------------===//

// CHECK:      module attributes {{.*}}thread_role_regions = 1 : i64{{.*}}topology = "role-mixed-irreducible"
module {
  func.func @mixed_unpack_only(%bias: memref<256xi8, 1>, %a: memref<2x2x16x32x2xf16, 1>,
                               %dst: memref<64x64xf16>, %r: index, %c: index,
                               %n: index, %m: index) {
    scf.for %i = %c to %n step %c {
      hmx.acc_clear
      hmx.acc_read %bias, %a, %m, %i {bias_set = 0 : i32} : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
      hmx.unpack_acc ins(%a, %r, %c : memref<2x2x16x32x2xf16, 1>)
          outs(%dst : memref<64x64xf16>)
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// 6. No HMX work at all. There is no role decision to record, and writing
//    "single-role-hvx" would read as a placement that was chosen rather than
//    one that was unnecessary.
//===----------------------------------------------------------------------===//

// CHECK:      module
// CHECK-NOT:  topology
module {
  func.func @pure_vector(%a: memref<64xf16>) {
    return
  }
}
