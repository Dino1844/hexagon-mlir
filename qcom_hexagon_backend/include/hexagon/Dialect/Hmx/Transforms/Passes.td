//===- Passes.td - HMX dialect passes ----------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The hmx dialect's own transform passes: engine attribution, the tile level,
// residency, and the internal VTCM accounting census. They live with the
// dialect so the HMX feature has one home; `hmx-to-llvm` is a conversion and
// stays under Conversion/HmxToLLVM.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_TD
#define HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def MatmulToHmx : InterfacePass<"matmul-to-hmx", "mlir::FunctionOpInterface"> {
  let summary = "Attribute and convert HMX-eligible linalg contractions to hmx.matmul";

  let description = [{
    Engine attribution for the HMX matrix engine, modelled on Triton's
    AccelerateMatmul: the entire decision is a legality predicate (`hmxEligible`,
    asking `HmxTarget`) followed by a residency check
    (`HmxTarget::planBridge`: the crouton arrays of one M block must fit VTCM; a
    contraction too large to hold whole is walked in M blocks, and only one that
    fits no block at all is left alone). There is no ordinary enable/disable
    switch -- only `vtcm-budget`, which caps the residency the bridge may
    claim, plus the metadata-only `record-only` mode used by pipelines that do
    not lower HMX.

    Version 1 (plain HMX) requires f16 or f32 inputs (an f32 source is quantised
    to the engine's fp16 by the bridge pack), an f16 or f32 output, 2D static
    shapes, M/N/K multiples of the 32-wide tile and M > 4. A matmul that does not
    qualify is left untouched -- the existing HVX and linalg paths keep handling
    it, because "this op is not for HMX" is an engine choice rather than a
    failure. The pass therefore never errors; it emits a remark, so a near-miss
    is explainable.

    Its operands live in VTCM, so the pass needs the pipeline to provide the
    VTCM allocator (`vtcm-allocator`); without it the attribution is refused
    rather than made on buffers the engine cannot read.

    Seeding and propagating the crouton layout is deliberately NOT this pass's
    job: that is standard `linalg.pack`/`unpack` (inner_dims_pos = [0], inner_tiles = [2]) work for
    the existing layout machinery.
  }];

  let options = [
    Option<"vtcmBudgetBytes", "vtcm-budget", "int64_t", /*default=*/"0",
           "VTCM budget in bytes the bridge must fit; 0 means the device "
           "default (HmxTarget::defaultVtcmBudget).">,
    Option<"vtcmAllocator", "vtcm-allocator", "bool", /*default=*/"true",
           "Whether the pipeline provides the VTCM allocator for the crouton "
           "arrays (the hexagonmem path that lowers space-1 allocations). The "
           "engine reads and writes VTCM only, so without it an attributed "
           "matmul would hand the leaves addresses the engine cannot use.">,
    Option<"recordOnly", "record-only", "bool", /*default=*/"false",
           "Record the attribution manifest without rewriting linalg.matmul or "
           "emitting HMX operations. The pass still evaluates the same "
           "eligibility and VTCM capability contract, but treats VTCM as "
           "unavailable in this mode so records carry the canonical "
           "vtcm-allocator-disabled reason; callers should set vtcm-allocator "
           "to false explicitly when constructing it.">,
    Option<"dropEncodings", "drop-encodings", "bool", /*default=*/"true",
           "Erase every #hmx.crouton tensor encoding once the pass is done, "
           "restoring the bare rank-5 physical types the rest of the pipeline "
           "knows (default bufferization drops the encoding anyway and emits "
           "no layout). Turn off to keep the encodings: the one-shot "
           "bufferization maps them to the identity-map "
           "#hmx.crouton_memref_layout through the dialect's own allocation op "
           "hmx.alloc_crouton (what this pass emits instead of "
           "bufferization.alloc_tensor), whose BufferizableOpInterface carries "
           "the layout into the memref types.">
  ];

  let constructor = "mlir::hmx::createMatmulToHmxPass()";
}

def ThreadRolePartition : InterfacePass<"thread-role-partition",
    "mlir::FunctionOpInterface"> {
  let summary = "Decide which thread each region belongs on, and record it";

  let description = [{
    Make "which thread runs this code" a compile-time fact instead of a runtime
    accident. Two decisions, kept orthogonal on purpose -- they are separate
    attributes, not two values of one enum, because a dual-role kernel
    legitimately holds both:

      region level, one per region: does it contain an op that must run where
        the HMX resource is held? Read off the positive `HmxEngineIns` marker.
      kernel level, one per kernel: single-role (and which engine), or mixed,
        and if mixed whether there is anything to hand across.

    The kernel verdict is one of:

      `topology-single-role-hmx`    engine only; an HMX thread is required
      `topology-single-role-hvx`    no engine work at all
      `role-split-ok`               a pack sits directly in a loop whose body
                                    holds engine work -- the engine ops may be
                                    one loop deeper, which is where every
                                    emitter puts them -- so there is a per-tile
                                    producer stream to overlap: while tile i is
                                    in the matrix engine, another thread builds
                                    tile i+1
      `role-mixed-irreducible`      layout work is co-located with engine work
                                    but is only `unpack_acc`/`unpack_acc_f32`.
                                    It consumes what the engine op just wrote,
                                    so there is no producer stream, only a
                                    dependency
      `role-split-nopack`           layout work exists but in disjoint regions:
                                    the pack loop ran to completion before the
                                    engine work began, so there is no per-tile
                                    granularity to stream at all

    Note what is NOT a criterion: "this kernel touches both engines". Nearly
    every one does, so that predicate splits nothing. Producer layout work
    co-located with engine work is the criterion, and `HmxLayoutHvx` cannot
    express it -- that one trait covers `pack_act`, `pack_weight`,
    `unpack_acc` and `unpack_acc_f32` alike, and only the first two can be
    hoisted onto the other thread ahead of time.

    `role-split-ok` is the one that decides real workloads, and it is decided
    by whether the kernel's activation pack rides the tile loop. Every STAGED
    arm puts the pack inside the m-tile loop, one pack per tile -- depth 2
    (pipelined) and depth 1 (the serial ring) alike, because both are the same
    source loop -- so both stage to `role-split-ok`. The plain serial arm
    (`pipeline-depth=3`, and every `auto` decline) leaves the whole-array pack
    loop in front of the tile loop, and the verdict is `role-split-nopack`.
    (2026-10-08, S2.5: this paragraph used to say "depth 1 -> nopack, depth 2
    -> ok", which misread the depth-1 arm -- it is the staged serial RING,
    whose pack is as per-tile as depth 2's; only the single-thread overlap is
    missing, and the overlap is what a thread split would add. The measured A/B
    that made the knob interesting is that depth-1 -> depth-2 overlap on one
    thread, worth 14% (1024x512x64) to 35% (256x64x2048) --
    docs/results/t1c-residual-and-thmx-real-ceiling-2026-10-02.md.)

    It runs AFTER hmx-partition, and that order is load-bearing. Before
    hmx-partition every HMX kernel looks the same: one `hmx.matmul` between two
    independent pack loops, so the predicate has one answer for all of them and
    the verdict carries no information. hmx-partition is what creates the tile
    loop and, at depth 2, what puts the pack inside it. Running after it is what
    lets the pass observe that difference instead of re-deriving it.

    It also runs AFTER hmx-vector-readout, and that order is load-bearing in
    the other direction (2026-10-09, the merged R2+R3 unit): the read-out
    split attaches while its completion proof (an in-body `acc_read`) is
    still in the tile loop and replaces the read-out with publishes; this
    pass then moves that `acc_read` onto the bound thread and re-establishes
    the proof per group with a `wait_retired` barrier ahead of each publish
    (the frozen executor's granular retire visibility). With the order
    reversed, this pass would first orphan the read-outs past its exit drain
    and the read-out split would find nothing to attach to. The split itself
    handles BOTH source forms -- the serial ring and the pipelined depth-2
    ring (whose peeled epilogue is outlined into the same engine section as
    the steady loop, the peeled tile riding the tail descriptor) -- and
    declines loudly on anything else.

    The manifest topology field is module-level, so it describes the kernel the
    backend emits -- one per module in this pipeline.
  }];
}


def HmxPartition : InterfacePass<"hmx-partition", "mlir::FunctionOpInterface"> {
  let summary = "Partition hmx.matmul into the tile loop";

  let description = [{
    The tile level of the HMX path (post-bufferization): an `hmx.matmul` on
    crouton arrays becomes a bias init, an (m, n) tile loop whose body is a K
    loop of mmas, and a read-out per tile.

    VTCM ownership (docs/hmx/hmx-system-design.md §10.12): the crouton arrays
    stay ordinary memory-space-1 buffers, and this pass deliberately does not
    carve a region out for them. The VTCM the kernel runs in is the one the
    runtime already owns, and the existing space-1 machinery is what places them
    there, together with every other VTCM buffer of the kernel.

    The software-pipelined form stages one 32-row activation tile at a time
    through a static ring of `hmx.stage` slots (`depth` 2 when the budget and
    the tile count allow, else 1): the pass emits the serial source loop
    (issue -> await -> compute per tile) and, at depth 2, hands it to the SCF
    software pipeliner (`scf::pipelineForLoop`) with the schedule "issue in
    stage 0, await and compute in stage 1"; the pipeliner generates the
    prologue, the steady kernel and the peeled epilogue and versions the
    cross-stage (token, slot) pair as its `iter_args`. Depth 1 is the same
    source loop left unpipelined. Keeping every stage row in range is by
    construction (the kernel stops one iteration short per stage and the
    epilogue stages nothing), so the transfer needs neither a guard nor a token
    sentinel. When staging is declined the plain tile loop is emitted unchanged
    and a remark names the reason.

    The staged path packs into one crouton-row scratch, not the whole activation
    array: the pack and its mmas are in the same iteration, so one row is live
    at a time. The array is retired whole, so a staged loop never carries a
    second copy of the activation.

    `pipeline-depth` selects the form: the staged ring at some depth, or the
    unstaged serial tile loop. The serial arm skips staging entirely -- no
    `hmx.stage`/`hmx.await`, the activation bridge and its array are kept -- so
    it is the same clear -> K mmas -> read nest the pass emits when the pipeline
    does not apply, reached by construction rather than by declining.

    `croutons-per-mma` sizes a batch instead of choosing one: it is how many K
    croutons a single `hmx.mma` walks, and it is the same integer in the staged
    arm, the unstaged arm and the peeled edge regions, so one value describes
    the whole kernel. The default (0) is the hardware maximum, so the knob
    cannot make the default path slower; its only purpose is to make "batch less
    than the maximum" expressible, which is what turns the batching into a
    measurable A/B rather than a rebuild.
  }];

  let options = [
    Option<"vtcmBudgetBytes", "vtcm-budget", "int64_t", /*default=*/"0",
           "VTCM budget in bytes the pipeline ring must fit; 0 means the device "
           "default (HmxTarget::defaultVtcmBudget).">,
    Option<"pipelineDepth", "pipeline-depth", "int64_t", /*default=*/"0",
           "Activation-staging ring depth: 0 = auto (the deepest ring the VTCM "
           "budget and the tile count allow), 1 = force the serial ring, 2 = "
           "ask for the double ring (narrowed to the deepest ring that fits, "
           "with a remark, when the budget cannot pay for it), 3 = skip the "
           "staging rewrite and emit the unstaged serial tile loop (no "
           "hmx.stage/hmx.await, the activation array is kept).">,
    Option<"stagedReadoutMTiles", "staged-readout-m-tiles", "int64_t",
           /*default=*/"0",
           "The m-tile floor for `auto` staging's read-out channel: at or "
           "above this many m-tiles, `auto` may stage a shallow-K shape too, "
           "because the staged loop is also what the read-out split's m-tile "
           "loop attaches to, and at least two read-out batches (m-tiles / "
           "batch) can overlap behind the engine. Honoured only for a matmul "
           "that no outer loop re-executes: the staged ring's fixed cost is "
           "per tile-loop execution, and a matmul inside an outer loop (flash "
           "attention's per-chunk Q@K and P@V) pays it per iteration. 0 keeps "
           "the channel closed (shallow-K `auto` always declines, the "
           "pre-readout behaviour).">,
    Option<"croutonsPerMma", "croutons-per-mma", "int64_t", /*default=*/"0",
           "How many K croutons one hmx.mma walks, i.e. the engine's repeat "
           "count. 0 = the hardware maximum (32), which is what 32 itself "
           "means -- so 'not passed' and 'passed 32' are the same request, the "
           "same way vtcm-budget's 0 means the device default. 1 reproduces the "
           "one-mma-per-cronton form (a Kt-trip software loop of single-cronton "
           "mmas) and is the A/B arm against the default; Kt smaller than the "
           "value still collapses to one mma of n_croutons = Kt. The domain is "
           "{0} u [1, 32] because 32 is a hardware bound (Rt[dC] is five bits, "
           "HmxOps.cpp:410-411), and a value outside it is an error rather "
           "than a clamp: silently narrowing '64' would make a request that "
           "cannot be encoded look like it took effect. See "
           "docs/hmx/ncroutons-k-fusion-2026-10-01.md.">
  ];

  let constructor = "mlir::hmx::createHmxPartitionPass()";
}

def HmxVectorReadout : InterfacePass<"hmx-vector-readout",
    "mlir::FunctionOpInterface"> {
  let summary = "Hand a lowered HMX read-out to a second thread in batches";

  let description = [{
    A lowered HMX matmul issues its accumulator read-out (`hmx.unpack_acc`) on
    the same thread as the matrix engine, so the two Hexagon engines take turns.
    At pipeline-depth 2 the read-out of a 1024x512x64 matmul is 39.79% of the
    kernel -- ~24.3 us of 61 us -- and it overlaps nothing. This pass moves it
    onto a second thread.

    What makes the move affordable is a layout fact, not a scheduling trick: the
    AR crouton array is allocated once *before* the m-tile loop and released
    *after* it (verified post-partition, `pp-1024x512x64.mlir` lines 11 and 95),
    so every AR row is live for the whole loop. Deferring the read-out of a
    group of G rows therefore costs no extra VTCM and needs no double buffering.
    And because the destination is a loop-invariant `memref.reinterpret_cast`
    with a zero offset -- addressing is done entirely by the `row` operand, whose
    lowering selects AR row `m` and destination rows `m*32` through `m*32+31` -- one
    descriptor can name a whole group of rows.

    The cost that makes per-tile deferral a loss is the handoff: measured
    0.9 us for a descriptor with a `qurt_signal` wake against 760 ns of read-out
    per m-tile. Batching divides the handoff by G:

        G=1 -> 28.8 us of handoff, loses.  G=2 -> 14.4 us, 1.19x.
        G=4 ->  7.2 us, 1.39x.            G=8 ->  3.6 us, 1.51x.

    which is why `batch` defaults to 4 rather than 1.

    The shape recognised is exactly one `scf.for` over m whose body holds exactly
    one `hmx.unpack_acc` reading an AR value that an `hmx.acc_read` in the same
    body writes, whose destination is defined outside the loop, and whose AR
    allocation dominates the loop and is released only after it. EVERY further
    requirement is a decline and never a repair: an unread row leaves the
    caller's output buffer holding stale bytes, which is a plausible-looking
    wrong answer and strictly worse than leaving the read-out where it was. A
    shape this pass does not fully understand comes out byte-identical.

    Row coverage is the argument the whole rewrite rests on, so it is made
    structurally rather than assumed. The loop must run from 0 to `ub` step 1 and
    its `row` operand must fold to exactly the induction variable, so the kernel
    publishes `G*floor(ub/G)` rows in groups of G and the tail batch publishes
    the rows from `G*floor(ub/G)` up to `Mt`. Those two ranges are adjacent and
    together they are every row from 0 through `Mt-1`, so each AR row is read once and
    once only. That is only a rewrite when the original IR covered the same rows:
    with a peeled epilogue read-out the kernel must stop at exactly `Mt-1` (one
    peeled row), and without one it must reach `Mt`. A kernel that leaves a gap is
    declined, not repaired -- reading a row no `acc_read` wrote would publish
    whatever the previous launch left in the array.

    Note that the tail is gated on the computed tail range, not on `Mt % G`: at
    depth 2 the kernel peels a row, so the loop stops at `Mt-1` and the tail is
    never empty even when `Mt` is a multiple of G (1024 rows at G=4: the loop
    covers rows 0..27 and the tail names 28..31, the peeled row included). A test
    on `Mt % G` alone would drop that row, which is the stale-bytes bug this pass
    exists to avoid.

    `drain` runs after the last publish and before the AR array is released, so
    the vector thread never reads freed VTCM and the caller never observes the
    destination before it is written.

    KNOWN INCOMPLETE: the three runtime entry points are called, but the
    `configure` call that hands the executor its function pointer is NOT emitted
    -- a function's address is not expressible before `convert-func-to-llvm`
    (`llvm.mlir.addressof` rejects a `func.func` symbol). The pass is therefore
    off by default and the kernel is not runnable until that call is emitted from
    a pass that runs after the func-to-llvm conversion. See the comments on
    `kReadoutFnPrefix` in the implementation.
  }];

  let options = [
    Option<"batch", "hmx-readout-batch", "int64_t", /*default=*/"4",
           "How many AR rows one handoff names (G). Must be >= 1; 0 or a "
           "negative value is an error rather than a clamp, because a clamped "
           "request would look honoured when it is not (measured: G=1 loses, "
           "G=4 is the default, G=8 is 1.51x).">,
    Option<"deferDrain", "defer-drain", "bool", /*default=*/"false",
           "Drop the kernel-exit drain and let the NEXT call's configure() "
           "barrier wait for the tail batch instead, so the tail read-out "
           "overlaps the function epilogue (AR release, return) rather than "
           "serialising behind it. The LAST call of a launch is drained by "
           "the wrapper, outside the timed region. Only takes effect when "
           "the function has exactly ONE readout loop: a second loop would "
           "publish into a ring whose only barrier this pass just removed, "
           "and a full ring drops batches, so a multi-loop function keeps "
           "the per-loop drain and is reported with a remark. Off by "
           "default: it changes the cross-call contract (a batch may still "
           "be in flight after the kernel returns), which is only sound for "
           "sequential single-instance launches.">
  ];

  let constructor = "mlir::hmx::createHmxVectorReadoutPass()";
}

def WeightResident : InterfacePass<"weight-resident", "mlir::FunctionOpInterface"> {
  let summary = "Make constant HMX weights a resident VTCM buffer";

  let description = [{
    A constant weight reaches the HMX tile level as a `memref.get_global` of a
    prepacked `memref.global` (the compile-time pack of `matmul-to-hmx`), which
    the engine cannot read: `hmx.mma` requires its weight in VTCM. This pass
    turns each such operand into a VTCM `hexagonmem.alloc` tagged
    `hmx.weight_resident`, recording the source symbol and byte count, and
    declares the total in the module attribute `hmx.weight_resident_bytes`.

    The buffer is not filled by the kernel. Its lowering emits one call to
    `hexagon_runtime_weight_resident_v2_dsp(key, bytes, alignment)`: the runtime
    allocates the buffer on the first launch, copies the weight in once, pins
    it (the per-launch deallocation is swallowed) and returns the same address
    on every later launch. So the weight is packed into VTCM exactly once per
    process.

    Must run after bufferization (the operand is a `memref.get_global`) and
    before `hmx-partition` (which needs the weight in VTCM).

    With `prepackRuntimeWeights` on, the same residency is applied to a runtime
    weight: the `hmx.pack_weight` bridge of a function argument is replaced by a
    resident buffer whose contents the runtime copies once, and the argument's
    slot and crouton layout are published on the module (`hmx.weight_prepack`)
    for the host pre-packer. The option is off by default because dropping the
    pack is only correct when the caller actually pre-packs; the default leaves
    the IR byte-identical to before.

    Process boundary (a real constraint, not an implementation detail): the
    residency key is the compile-time source address (a constant's
    `memref.global`) or the argument slot's aligned pointer (a prepacked
    runtime weight); the runtime copies on the FIRST hit and returns the pinned
    buffer on later hits without ever revalidating its contents. Within one
    process, the weight behind such a key must therefore keep its contents --
    reusing the same kernel with changed weight data in-process reads stale VTCM
    (the host pre-packer caches by content hash on its side; the device side
    never rechecks). A changed weight needs a fresh process.
  }];

  let options = [
    Option<"prepackRuntimeWeights", "prepack-runtime-weights", "bool",
           /*default=*/"false",
           "Replace a runtime weight's per-launch pack bridge with a resident "
           "buffer the host pre-packs (see hmx.weight_prepack).">,
  ];

  let constructor = "mlir::hmx::createWeightResidentPass()";
}

def HmxWorkspaceResident : InterfacePass<"hmx-workspace-resident",
                                         "mlir::FunctionOpInterface"> {
  let summary = "Make the per-launch HMX VTCM workspace a resident buffer";

  let description = [{
    Every HMX kernel's prologue allocates (and epilogue frees) its VTCM
    workspace: the crouton arrays of `matmul-to-hmx`, the `hmx-partition`
    conversion state, the activation-staging ring slots and statuses, and the
    crouton-row scratch. The runtime's allocator charges a size-independent
    best-fit scan plus bookkeeping for each, measured at ~6.3 us per alloc/free
    pair, so on a small shape the sequence costs more than the compute.

    (2026-10-07 correction: the "~6.3 us per pair" above and the
    "size-independent" qualifier are BOTH wrong and must not be quoted. The
    measured cost is 1.12 / 1.51 / 1.69 us per pair on S1/S2/S3, i.e. 4-6x
    lower, and the three points scatter by 1.5x, so size-independence has no
    evidence behind it -- three points cannot decide it, and the pair counts
    came from a 09-28 build. Source:
    docs/hmx/m3.2-device-result-2026-09-29.md §4.)

    A workspace has no compile-time contents -- the kernel refills it every
    launch -- so all it needs is a stable buffer. This pass tags each static
    VTCM `memref.alloc` with `hmx.workspace_resident` and drops its
    deallocation. The lowering emits one
    `hexagon_runtime_workspace_resident_v2_dsp(key, bytes, alignment)` call:
    the first launch allocates and pins the buffer, every later launch
    gets the same address back with no allocation, no free and no copy.

    The key is `(hash(function symbol) << 32) | allocation index` with the top
    bit set, so it is stable across launches, unique per (function, buffer) and
    disjoint from the weight path's address keys.

    Correctness boundary: a resident buffer is shared by every launch in the
    process. Sequential launches are safe (each launch overwrites the whole
    buffer before reading it); a grid>1 launch runs several instances in one
    process and they would clobber each other. The pass is therefore only added
    by the caller's opt-in (`enable-workspace-resident`), which promises
    single-instance execution.

    Runs after `hmx-partition` (so the partition pass's buffers exist) and
    before `convert-to-hexagonmem` (which carries the tag to the
    `hexagonmem.alloc` whose lowering reads it).
  }];

  let constructor = "mlir::hmx::createHmxWorkspaceResidentPass()";
}

def HmxVtcmAccounting : Pass<"hmx-vtcm-accounting"> {
  let summary = "Record a raw full-kernel VTCM allocation-site census";

  let description = [{
    This diagnostic pass runs at module scope after HMX partition and
    workspace residency, while VTCM allocations are still explicit. It is
    inert unless the module carries the internal
    `hmx.diagnostic_vtcm_accounting` unit marker. The result is the internal
    `hmx.kernel_vtcm_accounting` dictionary, not a manifest field and not a
    launcher contract.

    The first accounting slice is intentionally a raw allocation-site census:
    each distinct VTCM allocation site contributes its statically derivable
    requested bytes once, split into transient, workspace-resident, and
    weight-resident categories. It is not a live-range peak proof: it does not
    resolve loop/branch lifetimes, allocator alignment/fragmentation, or free
    cache retention. Dynamic, unsupported-backing, malformed-resident, and
    escaping-resident cases are reported as incomplete and fail closed. An
    optional second marker, `hmx.diagnostic_vtcm_liveness`, additionally emits
    a separate `hmx.kernel_vtcm_live_range` diagnostic for the supported
    single-invocation structured-SCF allocator-event subset. It is still not a
    runtime allocator, grid, fragmentation, or free-cache proof. Exact
    liveness and any future manifest schema remain separate work.
  }];

  let constructor = "mlir::hmx::createHmxVtcmAccountingPass()";
}

def HmxRecordV3 : Pass<"hmx-v3-record"> {
  let summary = "Finalize the record-only HMX v3 record document";

  let description = [{
    Finalizes the record-only `hex.hmx.kernel_manifest/v3` document that
    attribution published from compile-time facts. It runs at module scope right
    after the P1.5 VTCM accounting sidecars, so the requested-byte facts it can
    prove are still available, and it is inert unless the module carries the
    internal `hmx.diagnostic_v3_record` unit marker. That marker is deliberately
    not a backend option, so the production path cannot be switched to the v3
    envelope from a user-facing knob.

    Finalization folds in only the facts the P1.5 sidecars prove, recomputes each
    record fingerprint, and validates the closed v3 contract. Unknown capacity
    stays `null` and an unproven axis stays `not-proven`; the pass never writes a
    zero for something it did not measure. A document that cannot be proven is a
    failure rather than a reduced document.

    The result is a diagnostic record. It authorizes no HMX/tail plan, no
    full-kernel budget and no launch-grid constraint; the v2 manifest remains
    the execution authority.
  }];

  let constructor = "mlir::hmx::createHmxRecordV3Pass()";
}

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_TD
