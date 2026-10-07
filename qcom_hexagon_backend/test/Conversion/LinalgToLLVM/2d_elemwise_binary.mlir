// RUN: linalg-hexagon-opt %s -linalg-to-llvm | \
// RUN: linalg-hexagon-translate -emit=llvmir  | FileCheck %s

// This test checks the llvm-ir strictly to ensure that basic 2-D tensor sub is
// done efficiently no matter how other opts behave.
// Not all tests have to be this restricted.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @kernel(%x: memref<1024x256xf32>, %y: memref<1024x256xf32>, %z: memref<1024x256xf32>) {
   %t0 = bufferization.to_tensor %x restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>
   %t1 = bufferization.to_tensor %y restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>
   %t2 = bufferization.to_tensor %z restrict writable : memref<1024x256xf32> to tensor<1024x256xf32>

   %t3 = linalg.generic {
           indexing_maps = [#map, #map, #map],
           iterator_types = ["parallel", "parallel"]}
           ins(%t0, %t1 : tensor<1024x256xf32>, tensor<1024x256xf32>)
           outs(%t2 : tensor<1024x256xf32>) {
   ^bb0(%in: f32, %in_2: f32, %out: f32):

     %4 = arith.subf %in, %in_2 : f32
     linalg.yield %4 : f32
   } -> tensor<1024x256xf32>

   bufferization.materialize_in_destination %t3 in writable %z
      : (tensor<1024x256xf32>, memref<1024x256xf32>) -> ()
   return
 }
}
// This generic is pure parallel + identity, so VTCM staging is deliberately
// skipped (VTCMTiling.cpp: "streaming"): it must stream straight through DDR
// with vectorized loads/stores, and never pay the extra VTCM round trip.
//
// 2026-10-05 (C1): the streaming 2-D row slices now get L2 prefetch coverage
// (the l2-prefetch matcher generalized to row slices this day), so the
// kernel carries two l2fetch streams (x and y), targeting 8 KiB ahead:
// (row * 1024) + 8192 bytes from the base. The guard is the window-entry
// form -- one fire per 2 KiB block of stream advance, which for this
// 256-element (1 KiB) row is every second row: the window test folds to
// `row & 1 == 0`, and the room term folds to `row < 1015`
// ((262144 - 2560) / 256: the extent minus distance+block, over the row
// stride). Passing the pointers to the fetch intrinsic also costs the input
// pointers their inferred `readonly` attribute -- an effect the flat form
// has had since the pass landed (it was priced into the 2026-10-04 A/B), not
// a behavior change of this kernel.
// CHECK-LABEL: @kernel(ptr readnone captures(none) %0, ptr %1, i64 %2,
// CHECK-SAME:          i64 %3, i64 %4, i64 %5, i64 %6, ptr readnone captures(none) %7, ptr %8,
// CHECK-NOT:  hexagon_runtime_copy_dsp
// The guarded fetches: even rows only (the 2 KiB fire window is two rows
// wide), and room for one more fetch block 8 KiB ahead of the row.
// CHECK:      and i64 {{.+}}, 1
// CHECK:      icmp eq i64 {{.+}}, 0
// CHECK:      icmp {{.*}}ult {{.+}}, 1015
// CHECK:      tail call void @llvm.hexagon.Y5.l2fetch(ptr {{.+}}, i64 8796227239937)
// CHECK:      tail call void @llvm.hexagon.Y5.l2fetch(ptr {{.+}}, i64 8796227239937)
// CHECK:      [[GEP_X:%.+]] = getelementptr [4 x i8], ptr %1
// CHECK-NEXT: [[LOAD_X:%.+]] = load <32 x float>, ptr [[GEP_X]], align 4
// CHECK-NEXT: [[GEP_Y:%.+]] = getelementptr [4 x i8], ptr %8
// CHECK-NEXT: [[LOAD_Y:%.+]] = load <32 x float>, ptr [[GEP_Y]], align 4
// CHECK-NEXT: [[SUB:%.+]] = fsub fast <32 x float> [[LOAD_X]], [[LOAD_Y]]
// CHECK-NEXT: [[GEP_Z:%.+]]  = getelementptr [4 x i8], ptr
// CHECK-NEXT: store <32 x float> [[SUB]], ptr [[GEP_Z]]
