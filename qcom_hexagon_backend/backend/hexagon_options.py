# ===- hexagon_options.py ---------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
from dataclasses import dataclass
from typing import Tuple
import hashlib


@dataclass(frozen=True)
class HexagonOptions:
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    arch_triple: str = "hexagon"
    arch_features: str = f'+hvxv{os.getenv("HEXAGON_ARCH_VERSION")},+hvx-length128b'
    device_type: str = "hexagon"
    vectorize: int = 1
    num_threads: int = 4
    data_layout: str = (
        "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:"
        "32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
    )
    # Part of the launch-option contract with Triton core, not tuning knobs of
    # this backend: JITFunction._pack_args (triton/python/triton/runtime/jit.py)
    # raises KeyError for any launch kwarg that is absent from
    # `options.__dict__`, and core injects these as launch defaults. Deleting
    # them here (they are never branched on locally) makes every launch die with
    # "Keyword argument <name> was specified but unrecognised" -- measured on
    # device: all 7 tests failed in <2s each. Keep them.
    num_warps: int = 1
    num_stages: int = 1
    num_ctas: int = 1
    shared: bool = False
    cluster_dims: tuple = (1, 1, 1)
    supported_fp8_dtypes: Tuple[str, ...] = ()
    sanitize_overflow: bool = True
    debug: bool = False
    # Fallback only: a real compilation overwrites metadata["name"] with the
    # function name extracted from the MLIR module (backend/compiler.py), and
    # that is the symbol pack_metadata() hands to the launcher.
    name: str = "Hexagon"
    instrumentation_mode: str = ""
    htp_kernel_gen: bool = False
    target_artifact: str = "o"
    iterations: int = 10  # Triton specific benchmarking iteration count

    # Hexagon Linalg Options
    fusion: bool = True
    fusionAllowRecompute: bool = False
    fusionDoMultiUse: bool = True

    enableBufferization: bool = True  # Used to disable for some dma testing
    enableCollapseAddressSpace: bool = True  # lower llvm.ptr<1> if hexagonmem did not
    enableConvTiling: bool = False
    enableDoubleBuffering: bool = False  # enable double buffering optimization
    convTileSizes: str = ""
    enableConvertToHexagonmem: bool = True  # rewrites memref.alloc/copy to hexagonmem.*
    enableHexagonmemCopyToDMA: bool = False  # rewrites hexmem.copy to memref.dma_*
    enableHexKL: bool = False  # use HexKL to lower matmul and convolutions
    # hexKLMode (str, "micro"/"macro") was REMOVED here on 2026-09-30. Every
    # branch that tested it sat behind enableHexKL, which LinalgToLLVMPass
    # rejects outright ("enableHexKL is incompatible with the HMX manifest
    # contract"), so the field was unreachable; the declared pass default is
    # "micro" (Passes.td), i.e. what this field always supplied. Verified in
    # docs/codegen/knob-fork-classification-2026-09-30.md §1.3.1-5. Re-adding a
    # Python field for a pass option that no reachable path reads is the
    # "adding a knob that cannot change anything" pattern; register a real
    # owner in ROADMAP §2.1 first if one is ever needed.
    enableMultiThreading: bool = (
        False  # linalg-generic based multi-threading (FormVirtualThreadsPass)
    )
    enableThreadedDispatch: bool = (
        False  # use tm.exec() (real qurt threads) for SPMD grid dispatch; implicitly enabled when enableMultiThreading=True; when False, tm.exec_serial() is used
    )
    enableSCFThreading: bool = False  # scf based multi-threading
    enableSplitReduction: bool = False  # split-reduction optimization
    enableSplitReduceGeneric: bool = False  # split-reduce-generic optimization
    enableVectorization: bool = True  # enable HVX vectorization.
    enableVTCMTiling: bool = (
        True  # tile linalg-generic and introduce vtcm address-space
    )
    # External VTCM scratch buffer size in bytes per program instance.
    # scratch=0 (default): disabled, each instance allocates VTCM internally.
    # scratch=N (N>0): enables external VTCM flow. The wrapper allocates
    # N * prod(grid) bytes total, passes each instance a memref<Nxi8, 1>
    # {hexagon.scratch} slice. The compiler tiles and plans within N bytes.
    scratch: int = 0
    enableHVXInlining: bool = False
    enableSCFLoopUnroll: bool = False
    enableConversionToFp16: bool = False

    # Runtime-weight residency (P2). When on, the compiler drops a runtime
    # weight's per-launch `hmx.pack_weight` bridge and instead reads it from a
    # resident VTCM buffer; the launcher pre-packs the weight once per process
    # using the module's `hmx.weight_prepack` metadata. On by default: the
    # generated launcher always pre-packs, and a weight the pass cannot prove
    # dense (a dynamic-offset N-split view) keeps its old bridge rather than
    # being mis-packed. Measured at steady state: S1 -8.5%, S2 -19.8%, S3 -25.0%.
    enableWeightResident: bool = True

    # 2-D last-dim linalg.reduce (f16/f32 max/add) lowered to per-vector
    # elementwise folding plus an hvx.vror butterfly (vector-row-reduce pass)
    # instead of the scalarized per-lane chain the Hexagon backend produces for
    # vector.reduce.fmax. Off by default: device A/B switch. NOTE 2026-09-23:
    # the pattern (bufferized 2-D linalg.reduce) does not occur in production
    # FA/softmax (fused/loopified earlier); S0 micro + lit only. See R19.
    enableVectorRowReduce: bool = False

    # Vector f16/f32 arith.maxnumf rewritten into arith.maximumf + an explicit
    # NaN correction (hvx-maxnum-legalize pass). The Hexagon HVX backend has
    # no lowering for FMAXNUM (only FMAXIMUMNUM is Legal), so a vector maxnumf
    # is expanded per lane into a ~130-instruction stack/scalar/build-vector
    # chain (or fmaxf libcalls). Host-side the rewrite cuts the FA kernel 55%;
    # but on device the R1-ON kernel aborted at launch on the hexmem path with
    # a >=[512,128] maxnum tile. Root cause found (2026-09-23): three
    # llvm_triton Hexagon backend RA bugs -- docs/history/hmx/llvm-hexagon-ra-bugs.md;
    # the PS_aligna one is cherry-picked (tools/hexmlir/llvm-hexagon-ps-aligna.patch),
    # the rest await upstream. Default OFF until then; the knob re-enables it
    # (FA_MAXNUM in exp/hmx/op_bench/fa_ablate.py -- workspace scaffold, not
    # in this repo).
    enableMaxnumLegalize: bool = False
    # A row reduction that lands its per-row result in a rank-0 slice of a
    # tensor<rows x T> keeps that result in the vector domain instead
    # (row-reduce-group-store pass): the row loop steps by a whole HVX vector of
    # rows, the group's running values stay a vector<lanes x T> loop-carried
    # value, the hvx.vror butterfly is not extracted, and an arith.cmpi +
    # arith.select places it in the group's lane. Off by default: device A/B
    # switch. Only the maxnumf fold is rewritten -- addf would reassociate the
    # row sum, which is inside the pipeline's reassoc contract but is a numerical
    # change nobody asked for.
    enableRowReduceGroupStore: bool = False
    # The pass's three bisection knobs (enableMaxnumLegalizeFixup / Sel / Skip)
    # are deliberately NOT exposed here. All three were removed 2026-09-30:
    #   - Fixup was a semantics switch, not a debug affordance. Turning it off
    #     emits bare `maximumf`, which DROPS strict maxnum NaN semantics -- a
    #     different, wrong answer. A correctness footgun does not belong on a
    #     user-facing option surface.
    #   - Sel/Skip bisected a device crash that has since been attributed to the
    #     LLVM Hexagon AP under-alignment (see docs/hmx/fa-crash-resolved.md and
    #     the cherry-picked tools/hexmlir/llvm-hexagon-ps-aligna.patch). The
    #     question R1 still owes an answer is "does it pay?", which is a
    #     whole-knob ON/OFF A/B, not a per-site bisection.
    # No lit test ever passed any of the three, so nothing regressed here.

    # HMX tile-level software-pipeline depth (hmx-partition). 0 = auto (the
    # deepest activation-staging ring the VTCM budget and the tile count allow),
    # 1 = force the serial ring, 2 = request the double ring (narrowed to the
    # deepest ring that fits, with a remark, when the budget cannot pay for it),
    # 3 = skip staging and emit the unstaged serial tile loop (the third A/B arm:
    # no hmx.stage/hmx.await, the activation bridge is kept).
    enableHmxPipelineDepth: int = 0

    # Compile-time thread-role split (thread-role-partition). Off by default: with
    # the option off the pass is not added to the pipeline and nothing about the
    # kernel changes. On, the pass decides each region's role and records the
    # decision in the manifest; it does not yet outline or move code, so the
    # recorded topology is what a later stage would act on.
    enableThreadRolePartition: bool = False

    # K croutons walked by one `hmx.mma` (hmx-partition's croutons-per-mma).
    # 0 = the hardware maximum, 32, which is also what 32 means -- so this field
    # left at its default emits exactly the code it emitted before the option
    # existed. 1 is the A/B arm: one mma per crouton, i.e. a Kt-trip software
    # loop, which is what the compiler emitted before the batching. The domain
    # is {0} u [1, 32]; anything else is an error from hmx-partition, not a
    # clamp, because 32 is a hardware bound (the engine's K repeat field is
    # five bits) and a clamped request would look honoured when it is not.
    # Why it is on the Python surface rather than CLI-only: the batching's
    # benefit is unmeasured, and the only way to measure it without rebuilding
    # (which invalidates the device anchor) is two arms in one build. See
    # docs/hmx/ncroutons-k-fusion-2026-10-01.md.
    hmxCroutonsPerMma: int = 0

    # Per-launch VTCM workspace residency (hmx-workspace-resident). When on, the
    # crouton arrays, conversion state, staging ring/scratch and statuses of an
    # HMX kernel are allocated once and reused by every launch instead of being
    # allocated/freed per launch (~6.3 us per alloc/free pair, size-independent:
    # on S3 128x128x128 that is more than half the launch, 14 -> 6 us; on S1
    # 1024x512x64 it is another -17% on top of the readout split; gap-table
    # 2026-10-04, iters=1000). On by default: the resident entry is keyed by
    # the caller's flat program id (VtcmPool::Resident's slot), so concurrent
    # instances of a grid>1 launch get separate buffers instead of a shared
    # clobbered one, and the same pid across launches reuses the same buffer.
    enableWorkspaceResident: bool = True

    # Move a lowered HMX matmul's accumulator read-out (`hmx.unpack_acc`) onto a
    # second thread, in batches (hmx-vector-readout). At pipeline-depth 2 that
    # read-out is 39.79% of a 1024x512x64 kernel (~24.3 us of 61 us) and it
    # overlaps nothing, because it shares a thread with the matrix engine.
    # Batching exists because a per-row handoff would LOSE: measured 0.9 us per
    # descriptor against 760 ns of read-out per m-tile. The accumulator array is
    # allocated before the m-tile loop and released after it, so a group of rows
    # can be deferred at no VTCM cost, which divides the handoff by the group
    # size (G=4 -> 1.39x, G=8 -> 1.51x).
    # On by default since the gap-table measurement (2026-10-04, three shapes,
    # iters=1000): OFF->G4 is -24% on S1 1024x512x64, and the pass declines
    # every non-matching structure with a remark (FA/KDA object census: zero
    # readout symbols), so the default cannot silently rewrite them. The pass
    # runs after hmx-partition created the m-tile loop; the `configure()`
    # handoff that gives the executor its function pointer is emitted by
    # HmxToLLVM after convert-func-to-llvm (wireVectorReadout).
    enableHmxVectorReadout: bool = True

    # AR rows one handoff names (G, `hmx-vector-readout`'s batch). 4 is the
    # measured best of {1, 2, 4, 8}: G=1 loses to the handoff cost. Must be >= 1;
    # 0 or negative is rejected by the pass rather than clamped, because a clamped
    # request would look honoured when it is not.
    hmxReadoutBatch: int = 4

    # Drop the read-out split's kernel-exit drain and let the next call's
    # configure() barrier wait for the tail batch instead, so the tail read-out
    # overlaps the function epilogue (AR release + return) instead of
    # serialising behind it. The wrapper drains the last call of a launch
    # outside the timed region (weak symbol, skipped by non-readout kernels).
    # Declined with a remark for a function holding more than one readout
    # loop (the per-loop drain is the only ring barrier such a function has).
    # Off by default, deliberately: its single contribution is -6.4% on top of
    # the readout split (47 -> 44 us, two consistent runs, iters=1000), below
    # the > max(3*CV, 15%) evidence gate -- and a grid>1 launch has no
    # barrier between the wrapper's program iterations, so the ring can
    # overflow and trap. Making it grid-safe needs a per-iteration wrapper
    # drain (the wrapper is the layer that knows the grid); until that lands,
    # the deferral stays an explicit opt-in.
    hmxReadoutDeferredDrain: bool = False

    # `enableSeedLayoutConversions` was removed from this surface 2026-09-30.
    # It is an UPSTREAM pass option and it still works when driven directly
    # (`-linalg-to-llvm="enable-seed-layout-conversions=true"`, and
    # test/Conversion/LinalgToLLVM/matmul_to_conv.mlir drives -matmul-to-conv
    # directly) -- none of that was touched. What was removed is only the
    # Python field, because from this backend the option alone was a no-op: it
    # needs `enableMatmulToConv` too, and that has no field here. It therefore
    # sat on the option surface as something a user could set that changed
    # nothing. `enableConvTiling` is a separate S-class option and was NOT
    # touched.

    # Upstream crouton/pack machinery. The pack frontier extension is on by
    # default upstream; the HVX croutonization pass is not, and it is what turns
    # a crouton layout conversion into HVX permutes instead of element-wise moves.
    extendPackUpperFrontier: bool = True
    extendPackLowerFrontier: bool = True
    forceHVXCroutonization: bool = False

    tileSizes: str = ""  # User defined tile sizes - for debugging purposes

    # Separate out constants (dense_resource) into separate shared objects
    lowerConstantsInSeparateSharedObjects: bool = False

    # light weight profiling using hardware instructions
    enableLWP: bool = False

    # By default, loops are instrumented along with the function body.
    # To turn off loop level instrumentation, set it to True.
    disableLWPLoop: bool = False
    # By default, delete all artifacts pushed to device for this kernel's execution after it runs.
    # This solely applies to execution on the standalone launcher.
    deviceCleanup: bool = True

    def __post_init__(self):
        # Validate target_artifact
        valid_artifacts = {"ttir", "ttsharedir", "llir", "o", "so"}
        if self.target_artifact not in valid_artifacts:
            raise ValueError(
                f"Invalid target_artifact '{self.target_artifact}'. "
                f"Must be one of: {', '.join(sorted(valid_artifacts))}"
            )

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()
