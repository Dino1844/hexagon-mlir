// RUN: linalg-hexagon-opt %s -expand-math-ops  | FileCheck %s 

// This test expands complex operations into simple operations. 
// It verifies if the operation is expanding into it's mathematical expression
// consisting of simple operations

// A trailing CHECK-NOT only guards the text *after* the preceding match, so the
// single directive used to be satisfiable by a math.tanh emitted *before* the
// exp. Bracketing the positive check closes both ends.
// CHECK-NOT: math.tanh
// CHECK: {{.*}}  = math.exp %{{.*}} : vector<32xf32>
// CHECK-NOT: math.tanh

func.func @tanh(%arg0: vector<32xf32>) -> vector<32xf32> {
  %0 = math.tanh %arg0 fastmath<fast>: vector<32xf32>
  return %0 : vector<32xf32>
}
