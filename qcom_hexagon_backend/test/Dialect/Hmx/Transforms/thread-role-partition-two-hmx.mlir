// Two HMX kernels in one module is a compile error, not a last-one-wins record.
//
// `topology` is a module-level manifest field, so a module with two HMX kernels
// would publish whichever verdict the interface pass happened to write last --
// a confident answer about a kernel nobody asked about. This lives in its own
// file because `-split-input-file` fails the whole run on one bad chunk, which
// would take the six valid cases down with it.

// RUN: not linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(thread-role-partition))' 2>&1 | FileCheck %s

// CHECK:      error: thread-role-partition: the module's manifest topology field is module-level and cannot describe more than one HMX kernel
module {
  func.func @first(%bias: memref<256xi8, 1>, %a: memref<2x2x16x32x2xf16, 1>,
                   %m: index, %n: index) {
    hmx.acc_clear
    hmx.acc_read %bias, %a, %m, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
    return
  }
  func.func @second(%bias: memref<256xi8, 1>, %a: memref<2x2x16x32x2xf16, 1>,
                    %m: index, %n: index) {
    hmx.acc_clear
    hmx.acc_read %bias, %a, %m, %n {bias_set = 0 : i32} : memref<256xi8, 1>, memref<2x2x16x32x2xf16, 1>
    return
  }
}
