//===- row-reduce-group-store.mlir - the row result stays in a vector ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// The shape `HexagonVectorLowering` leaves for a vectorized 2-D last-dim row
// reduction: a row loop carrying a tensor<rows x T>, whose body is a column loop
// stepping by one whole 128 B HVX vector, whose body folds each chunk with
// `vector.reduction <maxnumf>` against the row's current value and writes the
// scalar result into a rank-0 slice of that same tensor. The last two ops are
// the lesion -- Hexagon HVX has no "vector lane 0 to GPR" instruction, so a
// reduction result that becomes an f32 costs a 128 B vector stack write plus a
// 4 B stack read back, per row per 128 B chunk, in a loop-carried chain.
//
// What this pins:
//   * the group carrier is a vector, not a tensor<lanes x T> (one-shot
//     bufferize leaves the former alone and materializes the latter);
//   * the butterfly's lane 0 is never extracted;
//   * 32 f32 rows / 64 f16 rows per group store (lanes = kHvxVectorBytes /
//     elemBytes, never written down as 32);
//   * the folds carry `fastmath<nnan>` without this pass waiting for
//     HexagonAddFastMath;
//   * every refusal says which gate, and leaves the IR untouched.
//
// One RUN line carries both arms on purpose: the workspace's manual lit runner
// executes only the first RUN line of a file.
//
// RUN: linalg-hexagon-opt %s -row-reduce-group-store -split-input-file -verify-diagnostics && linalg-hexagon-opt %s -row-reduce-group-store -split-input-file 2>&1 | FileCheck %s
//===----------------------------------------------------------------------===//

// -----
// f32: 128 rows, 64 KV columns => two chunks per row, 32 lanes, a 5-step
// butterfly (bytes 64..4). Two groups per 64-row tile.
//
// expected-remark @+1 {{row-reduce-group-store: candidates=1, rewritten=1}}
func.func @rowmax_f32(%S: tensor<128x64xf32>, %m: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.000000e+00 : f32
  %out = scf.forall (%t) = (0) to (128) step (64) shared_outs(%o = %m) -> (tensor<128xf32>) {
    %sTile = tensor.extract_slice %S[%t, 0] [64, 64] [1, 1] : tensor<128x64xf32> to tensor<64x64xf32>
    %mTile = tensor.extract_slice %o[%t] [64] [1] : tensor<128xf32> to tensor<64xf32>
    %rows = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %mTile) -> (tensor<64xf32>) {
      %chunks = scf.for %c = %c0 to %c64 step %c32 iter_args(%inner = %acc) -> (tensor<64xf32>) {
        %sChunk = tensor.extract_slice %sTile[%i, %c] [1, 32] [1, 1] : tensor<64x64xf32> to tensor<32xf32>
        %mRow = tensor.extract_slice %inner[%i] [1] [1] : tensor<64xf32> to tensor<f32>
        %v = vector.transfer_read %sChunk[%c0], %zero {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %init = tensor.extract %mRow[] : tensor<f32>
        %red = vector.reduction <maxnumf>, %v, %init : vector<32xf32> into f32
        %ins = tensor.insert %red into %mRow[] : tensor<f32>
        %insSlice = tensor.insert_slice %ins into %inner[%i] [1] [1] : tensor<f32> into tensor<64xf32>
        scf.yield %insSlice : tensor<64xf32>
      }
      scf.yield %chunks : tensor<64xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %rows into %o[%t] [64] [1] : tensor<64xf32> into tensor<128xf32>
    }
  }
  return %out : tensor<128xf32>
}

// These CHECK blocks were written before the pass existed and carried a
// PROVISIONAL note saying they had never been run. That note is stale and was
// removed on 2026-10-05: the file's single RUN uses -split-input-file, so
// FileCheck matches against the concatenated stdout of all 16 chunks, every
// CHECK-LABEL below resolves, and the flag spellings (notably the `fastmath<nnan>`
// on arith.maxnumf and the vror immediates) are verified on every run.
//
// CHECK-LABEL: func.func @rowmax_f32
// CHECK-NOT: vector.reduction
// CHECK-NOT: vector.extract
// CHECK-NOT: tensor.extract %
// CHECK-NOT: tensor.insert %
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<32xf32>)
// CHECK: vector.transfer_read {{.*}} : tensor<32xf32>, vector<32xf32>
// CHECK: arith.maxnumf {{.*}} fastmath<nnan> : vector<32xf32>
// CHECK: hvx.vror {{.*}}, 64 : vector<32xf32>
// CHECK: arith.maxnumf {{.*}} fastmath<nnan> : vector<32xf32>
// CHECK: hvx.vror {{.*}}, 32 : vector<32xf32>
// CHECK: arith.maxnumf {{.*}} fastmath<nnan> : vector<32xf32>
// CHECK: hvx.vror {{.*}}, 16 : vector<32xf32>
// CHECK: arith.maxnumf {{.*}} fastmath<nnan> : vector<32xf32>
// CHECK: hvx.vror {{.*}}, 8 : vector<32xf32>
// CHECK: arith.maxnumf {{.*}} fastmath<nnan> : vector<32xf32>
// CHECK: hvx.vror {{.*}}, 4 : vector<32xf32>
// CHECK: arith.maxnumf {{.*}} fastmath<nnan> : vector<32xf32>
// The merge fold folds the butterfly result into the group carried in, so its
// second operand IS the loop's block argument (%argN), not a numbered value.

// CHECK-NOT: hvx.vror
// CHECK: arith.index_cast {{.*}} : index to i32
// CHECK: vector.broadcast {{.*}} : i32 to vector<32xi32>
// CHECK: arith.cmpi eq, {{.*}}, {{.*}} : vector<32xi32>
// The order of the two value operands is load-bearing: the true arm must be the
// merged row result (the maxnumf above), the false arm the group carried in from
// the previous row (the loop's block argument).  A swap is numerically wrong --
// lane j would keep the stale value and the other 31 lanes would get the new row
// -- and it used to pass this file, because `%[[X:.*]], %[[Y:.*]]` matches in
// either order.  So the two slots are pinned by *shape*: a numbered SSA value
// that is the merge fold, and a loop block argument.
// Pinned by operand SHAPE, not by name: condition and true arm are numbered SSA
// values, the false arm is the loop's block argument.  A swap puts %argN in the
// true arm and stops matching, which is the bug this replaces.
// CHECK: arith.select {{%[a-z0-9]+}}, {{%[a-z0-9]+}}, {{%arg[0-9]+}} : vector<32xi1>, vector<32xf32>
// The merged value must leave the row loop and reach the group store -- that is
// what makes the 32-row fold actually accumulate. (The store is fed by the
// scf.for's result, so it is pinned through the yield rather than by name.)
// CHECK: scf.yield %{{[a-z0-9]+}} : vector<32xf32>
// CHECK: vector.transfer_write {{.*}} : vector<32xf32>, tensor<32xf32>
// CHECK: tensor.insert_slice {{.*}}[{{.*}}] [32] [1] : tensor<32xf32> into tensor<64xf32>
// -----

// -----
// f16: 64 lanes, so a SIX-step butterfly (bytes 64..2) and a rank-1 group of 64.
// Proves the lane count is derived, not written down as 32.
//
// expected-remark @+1 {{row-reduce-group-store: candidates=1, rewritten=1}}
func.func @rowmax_f16(%S: tensor<128x128xf16>, %m: tensor<128xf16>) -> tensor<128xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %zero = arith.constant 0.000000e+00 : f16
  %out = scf.forall (%t) = (0) to (128) step (64) shared_outs(%o = %m) -> (tensor<128xf16>) {
    %sTile = tensor.extract_slice %S[%t, 0] [64, 128] [1, 1] : tensor<128x128xf16> to tensor<64x128xf16>
    %mTile = tensor.extract_slice %o[%t] [64] [1] : tensor<128xf16> to tensor<64xf16>
    %rows = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %mTile) -> (tensor<64xf16>) {
      %chunks = scf.for %c = %c0 to %c128 step %c64 iter_args(%inner = %acc) -> (tensor<64xf16>) {
        %sChunk = tensor.extract_slice %sTile[%i, %c] [1, 64] [1, 1] : tensor<64x128xf16> to tensor<64xf16>
        %mRow = tensor.extract_slice %inner[%i] [1] [1] : tensor<64xf16> to tensor<f16>
        %v = vector.transfer_read %sChunk[%c0], %zero {in_bounds = [true]} : tensor<64xf16>, vector<64xf16>
        %init = tensor.extract %mRow[] : tensor<f16>
        %red = vector.reduction <maxnumf>, %v, %init : vector<64xf16> into f16
        %ins = tensor.insert %red into %mRow[] : tensor<f16>
        %insSlice = tensor.insert_slice %ins into %inner[%i] [1] [1] : tensor<f16> into tensor<64xf16>
        scf.yield %insSlice : tensor<64xf16>
      }
      scf.yield %chunks : tensor<64xf16>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %rows into %o[%t] [64] [1] : tensor<64xf16> into tensor<128xf16>
    }
  }
  return %out : tensor<128xf16>
}

// Flag spellings verified -- see the note above the f32 block.
//
// CHECK-LABEL: func.func @rowmax_f16
// CHECK-NOT: vector.reduction
// CHECK-NOT: vector.extract
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<64xf16>)
// CHECK: hvx.vror {{.*}}, 64 : vector<64xf16>
// CHECK: hvx.vror {{.*}}, 32 : vector<64xf16>
// CHECK: hvx.vror {{.*}}, 16 : vector<64xf16>
// CHECK: hvx.vror {{.*}}, 8 : vector<64xf16>
// CHECK: hvx.vror {{.*}}, 4 : vector<64xf16>
// CHECK: hvx.vror {{.*}}, 2 : vector<64xf16>
// CHECK-NOT: hvx.vror
// CHECK: arith.cmpi eq, {{.*}}, {{.*}} : vector<64xi32>
// CHECK: %[[SEL64:.*]] = arith.select %[[PRED64:.*]], %[[MERGED64:.*]], %[[GROUP64:.*]] : vector<64xi1>, vector<64xf16>
// CHECK: scf.yield %{{[a-z0-9]+}} : vector<64xf16>
// CHECK: vector.transfer_write {{.*}} : vector<64xf16>, tensor<64xf16>
// -----

// -----
// Rejection (G3, forall half): a forall tile's base is `lb + k*step`, not a
// constant, so what must be a multiple of the lane count is that pair. A tile
// starting at 48 puts global row 48 in lane 0, and 48 % 32 != 0, so the
// byte-neutral layout argument does not hold.
//
// CHECK-LABEL: func.func @forall_tile_base_not_multiple_of_lanes
// CHECK-NOT: hvx.vror
// CHECK: vector.reduction
// expected-remark @+1 {{row-reduce-group-store: candidates=1, rewritten=0}}
func.func @forall_tile_base_not_multiple_of_lanes(%S: tensor<128x32xf32>, %m: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0.000000e+00 : f32
// expected-remark @+1 {{row-reduce-group-store: the enclosing scf.forall tile [lb=48, ub=128) step 32 is not groupable by the 32-lane group}}
  %out = scf.forall (%t) = (48) to (128) step (32) shared_outs(%o = %m) -> (tensor<128xf32>) {
    %sTile = tensor.extract_slice %S[%t, 0] [32, 32] [1, 1] : tensor<128x32xf32> to tensor<32x32xf32>
    %mTile = tensor.extract_slice %o[%t] [32] [1] : tensor<128xf32> to tensor<32xf32>
    %rows = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc = %mTile) -> (tensor<32xf32>) {
      %chunks = scf.for %c = %c0 to %c32 step %c32 iter_args(%inner = %acc) -> (tensor<32xf32>) {
        %sChunk = tensor.extract_slice %sTile[%i, %c] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<32xf32>
        %mRow = tensor.extract_slice %inner[%i] [1] [1] : tensor<32xf32> to tensor<f32>
        %v = vector.transfer_read %sChunk[%c0], %zero {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %init = tensor.extract %mRow[] : tensor<f32>
        %red = vector.reduction <maxnumf>, %v, %init : vector<32xf32> into f32
        %ins = tensor.insert %red into %mRow[] : tensor<f32>
        %insSlice = tensor.insert_slice %ins into %inner[%i] [1] [1] : tensor<f32> into tensor<32xf32>
        scf.yield %insSlice : tensor<32xf32>
      }
      scf.yield %chunks : tensor<32xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %rows into %o[%t] [32] [1] : tensor<32xf32> into tensor<128xf32>
    }
  }
  return %out : tensor<128xf32>
}

// -----

// -----
// Rejection (G0, step): a column step of 48 is not the 32-lane group width. The
// horizontal butterfly is only defined on a whole 128 B register, so "rotate by
// half the register and fold" has no meaning on three quarters of one.
//
// CHECK-LABEL: func.func @column_step_not_lane_width
// CHECK-NOT: hvx.vror
// CHECK: vector.reduction
// expected-remark @+1 {{row-reduce-group-store: candidates=1, rewritten=0}}
func.func @column_step_not_lane_width(%S: tensor<32x96xf32>, %m: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c48 = arith.constant 48 : index
  %c96 = arith.constant 96 : index
  %zero = arith.constant 0.000000e+00 : f32
  %out = scf.forall (%t) = (0) to (32) step (32) shared_outs(%o = %m) -> (tensor<32xf32>) {
    %sTile = tensor.extract_slice %S[%t, 0] [32, 96] [1, 1] : tensor<32x96xf32> to tensor<32x96xf32>
    %mTile = tensor.extract_slice %o[%t] [32] [1] : tensor<32xf32> to tensor<32xf32>
    %rows = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc = %mTile) -> (tensor<32xf32>) {
// expected-remark @+1 {{row-reduce-group-store: the column step 48 is not the 32-lane group width}}
      %chunks = scf.for %c = %c0 to %c96 step %c48 iter_args(%inner = %acc) -> (tensor<32xf32>) {
        %sChunk = tensor.extract_slice %sTile[%i, %c] [1, 32] [1, 1] : tensor<32x96xf32> to tensor<32xf32>
        %mRow = tensor.extract_slice %inner[%i] [1] [1] : tensor<32xf32> to tensor<f32>
        %v = vector.transfer_read %sChunk[%c0], %zero {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %init = tensor.extract %mRow[] : tensor<f32>
        %red = vector.reduction <maxnumf>, %v, %init : vector<32xf32> into f32
        %ins = tensor.insert %red into %mRow[] : tensor<f32>
        %insSlice = tensor.insert_slice %ins into %inner[%i] [1] [1] : tensor<f32> into tensor<32xf32>
        scf.yield %insSlice : tensor<32xf32>
      }
      scf.yield %chunks : tensor<32xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %rows into %o[%t] [32] [1] : tensor<32xf32> into tensor<32xf32>
    }
  }
  return %out : tensor<32xf32>
}

// -----

// -----
// Rejection (G0, width): 48 columns is 192 B, not a multiple of the 128 B
// vector. The column loop does step by a whole lane here, so this is the width
// gate and not the step gate.
//
// CHECK-LABEL: func.func @column_width_not_whole_vector
// CHECK-NOT: hvx.vror
// CHECK: vector.reduction
// expected-remark @+1 {{row-reduce-group-store: candidates=1, rewritten=0}}
func.func @column_width_not_whole_vector(%S: tensor<32x48xf32>, %m: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c48 = arith.constant 48 : index
  %zero = arith.constant 0.000000e+00 : f32
  %out = scf.forall (%t) = (0) to (32) step (32) shared_outs(%o = %m) -> (tensor<32xf32>) {
    %sTile = tensor.extract_slice %S[%t, 0] [32, 48] [1, 1] : tensor<32x48xf32> to tensor<32x48xf32>
    %mTile = tensor.extract_slice %o[%t] [32] [1] : tensor<32xf32> to tensor<32xf32>
    %rows = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc = %mTile) -> (tensor<32xf32>) {
// expected-remark @+1 {{row-reduce-group-store: the column width 192 B is not a positive multiple of the 128 B vector}}
      %chunks = scf.for %c = %c0 to %c48 step %c32 iter_args(%inner = %acc) -> (tensor<32xf32>) {
        %sChunk = tensor.extract_slice %sTile[%i, %c] [1, 32] [1, 1] : tensor<32x48xf32> to tensor<32xf32>
        %mRow = tensor.extract_slice %inner[%i] [1] [1] : tensor<32xf32> to tensor<f32>
        %v = vector.transfer_read %sChunk[%c0], %zero {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %init = tensor.extract %mRow[] : tensor<f32>
        %red = vector.reduction <maxnumf>, %v, %init : vector<32xf32> into f32
        %ins = tensor.insert %red into %mRow[] : tensor<f32>
        %insSlice = tensor.insert_slice %ins into %inner[%i] [1] [1] : tensor<f32> into tensor<32xf32>
        scf.yield %insSlice : tensor<32xf32>
      }
      scf.yield %chunks : tensor<32xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %rows into %o[%t] [32] [1] : tensor<32xf32> into tensor<32xf32>
    }
  }
  return %out : tensor<32xf32>
}

// -----

// -----
// Rejection (G0, static): a dynamic column bound. The rewrite emits one
// transfer_read per chunk, so the chunk count has to be a compile-time number.
//
// CHECK-LABEL: func.func @column_bounds_not_static
// CHECK-NOT: hvx.vror
// CHECK: vector.reduction
// expected-remark @+1 {{row-reduce-group-store: candidates=1, rewritten=0}}
func.func @column_bounds_not_static(%S: tensor<32x64xf32>, %m: tensor<32xf32>, %ub: index) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0.000000e+00 : f32
  %out = scf.forall (%t) = (0) to (32) step (32) shared_outs(%o = %m) -> (tensor<32xf32>) {
    %sTile = tensor.extract_slice %S[%t, 0] [32, 64] [1, 1] : tensor<32x64xf32> to tensor<32x64xf32>
    %mTile = tensor.extract_slice %o[%t] [32] [1] : tensor<32xf32> to tensor<32xf32>
    %rows = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc = %mTile) -> (tensor<32xf32>) {
// expected-remark @+1 {{row-reduce-group-store: the column loop bounds are not static constants}}
      %chunks = scf.for %c = %c0 to %ub step %c32 iter_args(%inner = %acc) -> (tensor<32xf32>) {
        %sChunk = tensor.extract_slice %sTile[%i, %c] [1, 32] [1, 1] : tensor<32x64xf32> to tensor<32xf32>
        %mRow = tensor.extract_slice %inner[%i] [1] [1] : tensor<32xf32> to tensor<f32>
        %v = vector.transfer_read %sChunk[%c0], %zero {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %init = tensor.extract %mRow[] : tensor<f32>
        %red = vector.reduction <maxnumf>, %v, %init : vector<32xf32> into f32
        %ins = tensor.insert %red into %mRow[] : tensor<f32>
        %insSlice = tensor.insert_slice %ins into %inner[%i] [1] [1] : tensor<f32> into tensor<32xf32>
        scf.yield %insSlice : tensor<32xf32>
      }
      scf.yield %chunks : tensor<32xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %rows into %o[%t] [32] [1] : tensor<32xf32> into tensor<32xf32>
    }
  }
  return %out : tensor<32xf32>
}

// -----

// -----
// Rejection (G1): the row sum. Same shape, `add` fold -- and folding the chunks
// before the butterfly reassociates the sum, `max(max(reduce(c0), m),
// reduce(c1))` becoming `max(m, max(c0, c1))`. That is inside the pipeline's
// reassoc contract and still a numerical change nobody asked for, so the sum
// stays on the scalar path until a measurement asks for it.
//
// CHECK-LABEL: func.func @rowsum_addf_is_left_alone
// CHECK-NOT: hvx.vror
// CHECK: vector.reduction
// expected-remark @+1 {{row-reduce-group-store: candidates=1, rewritten=0}}
func.func @rowsum_addf_is_left_alone(%S: tensor<32x64xf32>, %m: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.000000e+00 : f32
  %out = scf.forall (%t) = (0) to (32) step (32) shared_outs(%o = %m) -> (tensor<32xf32>) {
    %sTile = tensor.extract_slice %S[%t, 0] [32, 64] [1, 1] : tensor<32x64xf32> to tensor<32x64xf32>
    %mTile = tensor.extract_slice %o[%t] [32] [1] : tensor<32xf32> to tensor<32xf32>
    %rows = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc = %mTile) -> (tensor<32xf32>) {
// expected-remark @+1 {{row-reduce-group-store: reduction kind add is not maxnumf}}
      %chunks = scf.for %c = %c0 to %c64 step %c32 iter_args(%inner = %acc) -> (tensor<32xf32>) {
        %sChunk = tensor.extract_slice %sTile[%i, %c] [1, 32] [1, 1] : tensor<32x64xf32> to tensor<32xf32>
        %mRow = tensor.extract_slice %inner[%i] [1] [1] : tensor<32xf32> to tensor<f32>
        %v = vector.transfer_read %sChunk[%c0], %zero {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %init = tensor.extract %mRow[] : tensor<f32>
        %red = vector.reduction <add>, %v, %init : vector<32xf32> into f32
        %ins = tensor.insert %red into %mRow[] : tensor<f32>
        %insSlice = tensor.insert_slice %ins into %inner[%i] [1] [1] : tensor<f32> into tensor<32xf32>
        scf.yield %insSlice : tensor<32xf32>
      }
      scf.yield %chunks : tensor<32xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %rows into %o[%t] [32] [1] : tensor<32xf32> into tensor<32xf32>
    }
  }
  return %out : tensor<32xf32>
}

// -----

// -----
// Not a candidate at all: a kernel with no row-reduction nest is left alone,
// and the census says zero rather than staying silent. Without that line, "the
// gate was too narrow" and "the kernel got faster" are the same observation.
//
// CHECK-LABEL: func.func @no_row_reduction_here
// CHECK-NOT: hvx.vror
// CHECK: tensor.empty
// expected-remark @+1 {{row-reduce-group-store: candidates=0, rewritten=0}}
func.func @no_row_reduction_here() -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %empty = tensor.empty() : tensor<64xf32>
  %out = scf.for %c = %c0 to %c64 step %c32 iter_args(%a = %empty) -> (tensor<64xf32>) {
    scf.yield %a : tensor<64xf32>
  }
  return %out : tensor<64xf32>
}