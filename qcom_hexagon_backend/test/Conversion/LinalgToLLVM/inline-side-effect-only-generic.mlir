// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion),func.func(inline-side-effect-only-generic),canonicalize)' | FileCheck %s --check-prefix=INLINED
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(inline-side-effect-only-generic))' | FileCheck %s --check-prefix=UNTOUCHED

// The pre-fusion shape of a keepdim reduce's [1]-store (ticket 34 pathology
// 1): the unstructured-to-memref lowering's store generic -- no outs, no
// results, body is the scalar store -- fed by two linalg.fills. Elementwise
// fusion folds the fills into the region and erases the operands they
// become, which leaves a zero-operand shell that no lowering accepts (the
// degenerate form cannot be written in text, only produced this way).
//
// Two arms, one input.
//   INLINED   -- fusion produces the zero-operand shell; the pass inlines the
//                shell's body into the function and erases the shell.
//   UNTOUCHED -- with the operands still attached (no fusion), the matcher
//                does not fire and the IR is untouched: the pass only repairs
//                the degenerate form.
#map2 = affine_map<() -> ()>

// INLINED-LABEL: func.func @store_of_filled_scalars
// INLINED-NOT: linalg.
// INLINED: memref.store
// UNTOUCHED-LABEL: func.func @store_of_filled_scalars
// UNTOUCHED: linalg.generic
// UNTOUCHED-SAME: ins(%{{.*}}, %{{.*}} : tensor<i32>, tensor<f32>)
// UNTOUCHED: arith.index_cast
// UNTOUCHED: memref.store
func.func @store_of_filled_scalars(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: tensor<f32>) {
  %cast = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
  %extracted = tensor.extract %arg3[] : tensor<f32>
  %0 = tensor.empty() : tensor<i32>
  %1 = linalg.fill ins(%arg2 : i32) outs(%0 : tensor<i32>) -> tensor<i32>
  %2 = tensor.empty() : tensor<f32>
  %3 = linalg.fill ins(%extracted : f32) outs(%2 : tensor<f32>) -> tensor<f32>
  linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%1, %3 : tensor<i32>, tensor<f32>) {
  ^bb0(%in: i32, %in_0: f32):
    %4 = arith.index_cast %in : i32 to index
    memref.store %in_0, %cast[%4] : memref<?xf32>
    linalg.yield
  }
  return
}
