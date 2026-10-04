#!/usr/bin/env python3
"""The OBJECT-level gate for the HMX vector read-out split (host only, no device).

WHY THIS FILE EXISTS
--------------------
This feature had a build that linked, produced a numerically correct kernel on the
device, and provably never ran. Every level of checking that looked at IR was
green: lit passes FileCheck the rewritten module, pytest read the manifest, and the
Triton compile reported success. The rewrite was then discarded somewhere between
IR and the object file, and the kernel executed the original serial read-out.

This gate looks at the OBJECT, with `llvm-nm`, because that is the only level at
which "the code is in the binary" is a fact rather than an intention.

WHY IT NOW CHECKS TWO KERNEL SHAPES
-----------------------------------
One shape was not enough, and the one shape it had was the reason seven measurement
sweeps read as "no speedup" when the truth was "feature absent". `matmul_kernel`
tiles nothing: its block pointers start at `(0, 0)`, so `materialize_in_destination`
builds the destination memref in the entry block, hmx-partition can hoist the
read-out into the m-tile loop, and `hmx-vector-readout` has something to attach to.

`matmul_kernel_grid` is the same 1024x512x64 matmul with `offsets=(pid * BLOCK_M, 0)`.
The destination offset now depends on `tl.program_id`, so the destination memref is
built AFTER the matmul, the read-out stays a separate loop walking AR rows after the
tile loop has retired, and `hmx-vector-readout` declines -- silently, with the option
explicitly on. Measured 2026-10-03 on this build, before the fix: 5 read-out symbols
in the whole-matrix object, 0 in the grid object's. Same options, same binary, one
variable changed.

A gate that only asserted the whole-matrix object was therefore green against a
kernel that had none of the feature, which is the failure this file exists to catch
applied to the gate itself. So both shapes are asserted, and both are asserted with
the SAME predicate function (`_split_is_in_object`) so neither can be quietly weaker.

WHAT IS ASSERTED, AND WHY EACH ONE
----------------------------------
  * `__hmx_readout` is DEFINED (T/t) in the object. This is the read-out itself.
    Absent means the whole rewrite vanished.
  * `__hmx_readout_entry` is defined. It is the adapter the executor is handed;
    its absence means the `configure` handoff could not have been wired.
  * `hexagon_runtime_hmx_exec_{configure,publish,drain}` are present -- as an
    undefined reference (`U`) here, resolved at device link against
    `libhexagon_mlir_async_runtime.a`. Presence, not definedness: the kernel
    object is not supposed to carry the runtime's code.
  * NO `error:` line appears anywhere in the compile's stderr. This is the check
    that would have caught the original defect: the pass failed hard with
    "HMX bridge operation has no explicit hmx.decision_id" and the failure was
    only visible on stderr, never in a return value.
  * The manifest's bridge counts do not double count (see the last test).
  * The rows the engine thread publishes cover every AR row exactly once. This is
    the check for the failure that cannot be seen at all from outside: a wrong
    `rowStart` reads the wrong accumulator rows and the kernel returns plausible
    garbage. It is computed from the descriptor stores in the pass's own output,
    not from the object's symbol table, because `__hmx_readout` takes `row0` and
    `nrows` as arguments -- the numbers only exist in the caller.

WHY THE NEGATIVE ARM IS THE POINT
----------------------------------
At `enableHmxPipelineDepth=0` there is no m-tile loop for the read-out pass to
attach to, so the pass is a SILENT NO-OP and the object is byte-identical to the
option-off build. It compiles cleanly, it links, it runs, it is numerically
correct, and it contains none of the feature.

So an object-level gate that only asserted "an object was produced" would pass on
the exact failure it exists to catch. The gate's predicate is therefore asserted
against that no-op build and REQUIRED TO BE FALSE. That is what makes it
non-vacuous, and it costs nothing: it is the same binary and the same shape, only
a different option. The same negative arm runs on BOTH shapes.

Run:  source tools/hexmlir/env.sh
      .venv/bin/python hexagon-mlir/qcom_hexagon_backend/test/test_hmx_vector_readout_object_gate.py
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

# The measured shape: 1024x512x64 is the kernel the 39.79% / 24.3 us read-out
# figure is about, and at pipeline-depth 2 hmx-partition hoists the read-out into
# the m-tile loop, which is the shape the split needs. A smaller or depth-1 shape
# would not exercise the rewrite.
M, N, K = 1024, 512, 64

# The grid shape tiles M into BLOCK_M-row programs, so each program is a
# BLOCK_M x 512 x 64 matmul. BLOCK_M = 256 is 8 m-tiles of 32, which is above the
# depth-2 ring cap, so hmx-partition stages AND peels it -- the same loop shape the
# whole-matrix kernel has, reached through the `tl.program_id` offsets rather than
# through zero offsets. BLOCK_M = 64 is 2 m-tiles, which is at the cap, so it does
# not peel; both are checked because the two take different arms of the rewrite
# (epilogue-anchored tail vs loop-anchored tail) and the drain ordering differs
# between them.
BLOCK_M = 256
BLOCK_M_SMALL = 64

# The symbols whose absence means "the feature did not reach the object".
DEFINED_REQUIRED = ("__hmx_readout", "__hmx_readout_entry")
REFERENCED_REQUIRED = (
    "hexagon_runtime_hmx_exec_configure",
    "hexagon_runtime_hmx_exec_publish",
    "hexagon_runtime_hmx_exec_drain",
)

# The options the measurement used, so this test compiles the kernel a reader
# would actually be measuring rather than a stripped-down variant.
DEPTH_2 = dict(
    enableThreadRolePartition=True,
    enableHmxPipelineDepth=2,
)
READOUT_ON = dict(enableHmxVectorReadout=True)


@triton.jit
def matmul_kernel(A, B, C, N_ROWS: tl.constexpr, N_COLUMNS: tl.constexpr,
                  N_INNER: tl.constexpr):
    """The whole-block matmul, verbatim from tools/hexmlir/dump_codegen.py:89.

    Copied rather than re-expressed so this gate compiles the kernel a reader would
    actually be measuring. The Triton front end in this tree rejects a walrus
    expression in a `constexpr` block shape ("unsupported AST node type:
    NamedExpr"), so a tidier spelling of the same kernel does not compile at all --
    which is the whole reason to take the shipped one verbatim.
    """
    A_block_ptr = tl.make_block_ptr(base=A, shape=(N_ROWS, N_INNER), strides=(N_INNER, 1),
                                    offsets=(0, 0), block_shape=(N_ROWS, N_INNER), order=(1, 0))
    B_block_ptr = tl.make_block_ptr(base=B, shape=(N_INNER, N_COLUMNS), strides=(N_COLUMNS, 1),
                                    offsets=(0, 0), block_shape=(N_INNER, N_COLUMNS), order=(1, 0))
    C_block_ptr = tl.make_block_ptr(base=C, shape=(N_ROWS, N_COLUMNS), strides=(N_COLUMNS, 1),
                                    offsets=(0, 0), block_shape=(N_ROWS, N_COLUMNS), order=(1, 0))
    q = tl.load(A_block_ptr)
    k_t = tl.load(B_block_ptr)
    qk = tl.dot(q, k_t, out_dtype=C.type.element_ty)
    tl.store(C_block_ptr, qk)


@triton.jit
def two_matmul_kernel(A, B, C, D, E, N_ROWS: tl.constexpr,
                      N_COLUMNS: tl.constexpr, N_INNER: tl.constexpr):
    """Two whole-block matmuls in sequence, both at function top level.

    This is the multi-matmul reach of the split, distilled: two `hmx.matmul`,
    each naming its own AR, destination and decision id, with no outer loop
    re-executing either. hmx-vector-readout takes such a function as one GROUP
    per matmul (since 2026-10-04); before that it declined the whole function
    on the first disagreement between the matmuls. Both block bodies are
    `matmul_kernel` verbatim (same reason it is taken verbatim from
    dump_codegen.py: the shipped kernel is what a reader would measure).
    """
    A_block_ptr = tl.make_block_ptr(base=A, shape=(N_ROWS, N_INNER), strides=(N_INNER, 1),
                                    offsets=(0, 0), block_shape=(N_ROWS, N_INNER), order=(1, 0))
    B_block_ptr = tl.make_block_ptr(base=B, shape=(N_INNER, N_COLUMNS), strides=(N_COLUMNS, 1),
                                    offsets=(0, 0), block_shape=(N_INNER, N_COLUMNS), order=(1, 0))
    C_block_ptr = tl.make_block_ptr(base=C, shape=(N_ROWS, N_COLUMNS), strides=(N_COLUMNS, 1),
                                    offsets=(0, 0), block_shape=(N_ROWS, N_COLUMNS), order=(1, 0))
    q = tl.load(A_block_ptr)
    k_t = tl.load(B_block_ptr)
    qk = tl.dot(q, k_t, out_dtype=C.type.element_ty)
    tl.store(C_block_ptr, qk)
    D_block_ptr = tl.make_block_ptr(base=D, shape=(N_ROWS, N_INNER), strides=(N_INNER, 1),
                                    offsets=(0, 0), block_shape=(N_ROWS, N_INNER), order=(1, 0))
    E_block_ptr = tl.make_block_ptr(base=E, shape=(N_ROWS, N_COLUMNS), strides=(N_COLUMNS, 1),
                                    offsets=(0, 0), block_shape=(N_ROWS, N_COLUMNS), order=(1, 0))
    q2 = tl.load(D_block_ptr)
    qk2 = tl.dot(q2, k_t, out_dtype=E.type.element_ty)
    tl.store(E_block_ptr, qk2)


@triton.jit
def matmul_kernel_grid(A, B, C, N_ROWS: tl.constexpr, N_COLUMNS: tl.constexpr,
                       N_INNER: tl.constexpr, BLOCK_M: tl.constexpr):
    """The same matmul, M tiled by `tl.program_id` -- the shape real code uses.

    Identical to `matmul_kernel` in every respect except the two block pointers'
    `offsets`, which now carry `pid * BLOCK_M`. That is the whole variable: it is
    what makes the destination memref's offset dynamic, which is what moved the
    destination's `memref.reinterpret_cast` below the matmul, which is what stopped
    hmx-partition hoisting the read-out, which is what left `hmx-vector-readout`
    with nothing to rewrite.

    `N_ROWS` is still the full M: the block pointer's `shape` describes the whole
    matrix and only the starting offset moves, which is the ordinary Triton idiom.
    """
    pid = tl.program_id(0)
    A_block_ptr = tl.make_block_ptr(base=A, shape=(N_ROWS, N_INNER), strides=(N_INNER, 1),
                                    offsets=(pid * BLOCK_M, 0), block_shape=(BLOCK_M, N_INNER), order=(1, 0))
    B_block_ptr = tl.make_block_ptr(base=B, shape=(N_INNER, N_COLUMNS), strides=(N_COLUMNS, 1),
                                    offsets=(0, 0), block_shape=(N_INNER, N_COLUMNS), order=(1, 0))
    C_block_ptr = tl.make_block_ptr(base=C, shape=(N_ROWS, N_COLUMNS), strides=(N_COLUMNS, 1),
                                    offsets=(pid * BLOCK_M, 0), block_shape=(BLOCK_M, N_COLUMNS), order=(1, 0))
    q = tl.load(A_block_ptr)
    k_t = tl.load(B_block_ptr)
    qk = tl.dot(q, k_t, out_dtype=C.type.element_ty)
    tl.store(C_block_ptr, qk)


def _compile_obj(kernel, grid, constexpr, **options):
    """Compile one kernel shape to a device object, returning (bytes, stderr).

    `kernel.warmup` with `target_artifact="o"` compiles and does NOT launch: the
    difference between `warmup` and `kernel[grid](...)` is one character and the
    second one talks to the DSP. No device is touched here.
    """
    a = torch.zeros(M, K, dtype=torch.float16)
    b = torch.zeros(K, N, dtype=torch.float16)
    c = torch.zeros(M, N, dtype=torch.float16)

    # An isolated kernel cache, so this test never reads or writes the shared one
    # and never depends on a stale entry for the same options.
    with tempfile.TemporaryDirectory(prefix="hmx-readout-gate-") as tmp:
        previous_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
        # fd 2 is captured directly rather than through `contextlib.redirect_stderr`,
        # because the diagnostics come from C++ (llvm::errs()) inside libtriton and
        # would not be routed through sys.stderr.
        captured = os.dup(2)
        path = os.path.join(tmp, "stderr")
        sink = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.dup2(sink, 2)
            kernel_obj = kernel.warmup(
                a, b, c, grid=grid, target_artifact="o",
                N_ROWS=M, N_COLUMNS=N, N_INNER=K,
                # `{**a, **b}` so a later key WINS. `dict(a, **b, **c)`
                # raises when b and c share a key, which is exactly what
                # the depth-0 arm's `enableHmxPipelineDepth=0` does.
                **{**constexpr, **DEPTH_2, **options},
            )
        finally:
            os.dup2(captured, 2)
            os.close(sink)
            os.close(captured)
            if previous_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous_cache
        stderr = Path(path).read_text(errors="replace")
    obj = kernel_obj.asm["o"]
    return bytes(obj), stderr


# The two shapes, named once so the positive and negative arms cannot drift apart
# and so a failure message says which shape failed.
WHOLE = ("whole-matrix", matmul_kernel, (1,), {})
GRID = ("grid-tiled", matmul_kernel_grid, (M // BLOCK_M,), dict(BLOCK_M=BLOCK_M))
GRID_SMALL = ("grid-tiled-small", matmul_kernel_grid, (M // BLOCK_M_SMALL,),
              dict(BLOCK_M=BLOCK_M_SMALL))
SHAPES = (WHOLE, GRID, GRID_SMALL)


# ---------------------------------------------------------------------------
# The multi-matmul shape, both halves.
#
# The split's multi-matmul reach is gate-checked in two directions, because the
# two directions fail differently and each has failed silently once already:
#
#   * A TOP-LEVEL two-matmul function (`two_matmul_kernel` above) must carry
#     BOTH read-outs: two outlined read-outs, two entry adapters. Until
#     2026-10-04 hmx-vector-readout declined such a function whole, and the
#     object carried zero read-out symbols with the option on.
#   * Flash attention -- the production multi-matmul kernel, two `hmx.matmul`
#     per chunk of the attention loop -- must carry NONE, and that is the
#     correct answer, not a missing feature. The chain, each link measured or
#     read off the IR on 2026-10-04: its matmuls sit inside an outer loop, so
#     `auto` staging's read-out channel stays closed for them (the staged
#     ring's fixed cost is per tile-loop execution; flash attention pays it
#     twice per chunk -- 5466/5555 us staged vs 5230 unstaged, iters=1000),
#     so its tile loops take the unstaged serial form, where the m-tile loop
#     (mma, acc_read) and the read-out loop (unpack) are SEPARATE siblings --
#     and matchReadout declines that form, correctly: with all the engine
#     work done before the read-out starts and the softmax consumer right
#     after it, there is no engine work for a second thread to overlap
#     against. Moving the read-out would be pure handoff cost.
#
# The kernel is imported rather than copied (the dump_codegen.py precedent): a
# transcription slip would gate a kernel the device never ran.
def _import_fa_kernel():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    tests = os.path.normpath(os.path.join(here, "..", "..", "test",
                                          "python", "triton"))
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import test_flash_attention  # noqa: E402  (sys.path must be set first)
    return test_flash_attention


_FA = _import_fa_kernel()
FA_ABSENT_REQUIRED = ("__hmx_readout", "__hmx_readout_1",
                      "__hmx_readout_entry", "__hmx_readout_1_entry",
                      "hexagon_runtime_hmx_exec_configure",
                      "hexagon_runtime_hmx_exec_publish",
                      "hexagon_runtime_hmx_exec_drain")
TWO_MM_DEFINED_REQUIRED = ("__hmx_readout", "__hmx_readout_1",
                           "__hmx_readout_entry", "__hmx_readout_1_entry")


def _two_matmul_compile_obj(**options):
    """Compile `two_matmul_kernel` to an object, host only, like `_compile_obj`."""
    a = torch.zeros(M, K, dtype=torch.float16)
    b = torch.zeros(K, N, dtype=torch.float16)
    c = torch.zeros(M, N, dtype=torch.float16)
    d = torch.zeros(M, K, dtype=torch.float16)
    e = torch.zeros(M, N, dtype=torch.float16)
    with tempfile.TemporaryDirectory(prefix="hmx-readout-gate-2mm-") as tmp:
        previous_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
        captured = os.dup(2)
        path = os.path.join(tmp, "stderr")
        sink = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.dup2(sink, 2)
            kernel_obj = two_matmul_kernel.warmup(
                a, b, c, d, e, grid=(1,), target_artifact="o",
                N_ROWS=M, N_COLUMNS=N, N_INNER=K,
                **{**DEPTH_2, **options},
            )
        finally:
            os.dup2(captured, 2)
            os.close(sink)
            os.close(captured)
            if previous_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous_cache
        stderr = Path(path).read_text(errors="replace")
    return bytes(kernel_obj.asm["o"]), stderr


def _fa_compile_obj(**options):
    """Compile the FA kernel to an object with the FA test's own options.

    Host only (`warmup`), exactly like `_compile_obj` above; the FA options
    are the test's verbatim so the gate describes the kernel the device runs.
    The compile's stderr comes back alongside, for the same no-`error:`-line
    reason as every other arm.
    """
    z, h, n_ctx, d_head = 1, 1, 1024, 64
    q = torch.rand(z, h, n_ctx, d_head)
    k = torch.rand(z, h, n_ctx, d_head)
    v = torch.rand(z, h, n_ctx, d_head)
    out = torch.zeros_like(q)
    fa_opts = dict(
        enableVectorization=True, enableSplitReduceGeneric=True,
        enableHVXInlining=True, enableSCFLoopUnroll=True,
        enableMultiThreading=True, enableHexKL=False,
        enableVTCMTiling=False, enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=False,
        N_CTX=n_ctx, BLOCK_M=n_ctx, BLOCK_DMODEL=d_head, BLOCK_N=64,
        STAGE=1, stride_0=h * n_ctx * d_head, stride_1=n_ctx * d_head,
        stride_2=d_head, stride_3=1,
    )
    with tempfile.TemporaryDirectory(prefix="hmx-readout-gate-fa-") as tmp:
        previous_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
        captured = os.dup(2)
        path = os.path.join(tmp, "stderr")
        sink = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.dup2(sink, 2)
            kernel_obj = _FA.attention_fwd_kernel.warmup(
                0, q, k, v, 0.5, out, h, grid=(1,), target_artifact="o",
                **{**fa_opts, **options},
            )
        finally:
            os.dup2(captured, 2)
            os.close(sink)
            os.close(captured)
            if previous_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous_cache
        stderr = Path(path).read_text(errors="replace")
    return bytes(kernel_obj.asm["o"]), stderr


# `shutil.which` is resolved once: every arm below needs it, and a gate that skips
# itself three times over is not a gate.
_LLVM_NM = shutil.which("llvm-nm") or shutil.which("llvm-nm-14")


def _nm(obj):
    """Symbol table of a device object: {name: nm type letter}."""
    if _LLVM_NM is None:
        raise unittest.SkipTest("llvm-nm not found; this gate reads objects")
    with tempfile.TemporaryDirectory(prefix="hmx-readout-nm-") as tmp:
        path = Path(tmp) / "kernel.o"
        path.write_bytes(obj)
        # Hexagon objects need the right target for `llvm-nm` to parse them; the
        # host default reads ELF fine, so no -mtriple is passed here. If that ever
        # stops being true the gate fails loudly rather than silently seeing zero
        # symbols, because every assertion below is on a non-empty table.
        result = subprocess.run(
            [_LLVM_NM, str(path)], capture_output=True, text=True, check=True
        )
    symbols = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            symbols[parts[-1]] = parts[-2]
    return symbols


def _kernel_disassembly(obj, symbol="matmul_kernel"):
    """The kernel's own instructions, as text, with the objdump header removed.

    This is the deepest comparison in the file, and the only reproducible one.
    Two claims about it, both measured on this build:

      * The INSTRUCTIONS are deterministic: two compiles with byte-identical
        options produce identical disassembly.
      * The OBJECT BYTES are NOT. Those same two compiles produce different md5s,
        because the embedded runtime bitcode's sections and symbols are emitted in
        a different ORDER each time -- 505 symbols in both, identical sizes,
        permuted. Measured: two `dump_codegen.py matmul o` runs at
        pipeline-depth 0 with identical options gave
        8708bfd2dc1d74cbbdfd734004312502 and 8abbeadd9888948b95e0320e8a973956.

    So a byte-comparison of two objects from this toolchain proves nothing, and a
    gate built on one would be flaky in the worst way: passing or failing for a
    reason unrelated to the feature. That is why the no-op arm compares
    instructions instead of md5.
    """
    objdump = shutil.which("llvm-objdump") or shutil.which("llvm-objdump-14")
    if objdump is None:
        raise unittest.SkipTest("llvm-objdump not found; this gate reads objects")
    with tempfile.TemporaryDirectory(prefix="hmx-readout-dis-") as tmp:
        path = Path(tmp) / "kernel.o"
        path.write_bytes(obj)
        result = subprocess.run(
            [objdump, "-d", "--disassemble-symbols=" + symbol, str(path)],
            capture_output=True, text=True, check=True,
        )
    # objdump echoes its input path in the header line, which differs per temporary
    # directory; dropping it is what makes the rest comparable.
    return [ln for ln in result.stdout.splitlines() if "file format" not in ln]


def _manifest(kernel, grid, constexpr, **options):
    """The host-side HMX manifest for the same compile."""
    a = torch.zeros(M, K, dtype=torch.float16)
    b = torch.zeros(K, N, dtype=torch.float16)
    c = torch.zeros(M, N, dtype=torch.float16)
    with tempfile.TemporaryDirectory(prefix="hmx-readout-manifest-") as tmp:
        previous_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
        try:
            compiled = kernel.warmup(
                a, b, c, grid=grid, target_artifact="o",
                N_ROWS=M, N_COLUMNS=N, N_INNER=K,
                # `{**a, **b}` so a later key WINS. `dict(a, **b, **c)`
                # raises when b and c share a key, which is exactly what
                # the depth-0 arm's `enableHmxPipelineDepth=0` does.
                **{**constexpr, **DEPTH_2, **options},
            )
        finally:
            if previous_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous_cache
    return json.loads(compiled.packed_metadata["hmx_manifest"])


def _split_is_in_object(symbols):
    """The gate's whole predicate, in one place, so the negative arm tests it.

    Kept as a function rather than inlined into the assertions: the negative arms
    below call THIS on an object that compiled perfectly and contains none of the
    feature, which is only a meaningful check if it is the same code the positive
    arm trusts.
    """
    missing = []
    for name in DEFINED_REQUIRED:
        if symbols.get(name) not in ("T", "t"):
            missing.append(f"{name} is not defined (got {symbols.get(name)!r})")
    for name in REFERENCED_REQUIRED:
        if name not in symbols:
            missing.append(f"{name} is absent entirely")
    return missing


# ---------------------------------------------------------------------------
# Published-row coverage.
#
# `__hmx_readout` takes `(ar, dst, row0, nrows)` and walks `row0 .. row0+nrows-1`,
# so the per-handoff `(rowStart, rowCount)` is the ONLY thing that decides which
# accumulator rows reach the caller's output. A wrong rowStart reads the wrong rows
# and the kernel returns plausible garbage: no symbol check, no error line and no
# test in this file can see it. So the descriptor stores are read back out of the
# pass's own output and turned into a row set.
#
# WHY THIS READS AND DOES NOT RECOMPUTE
# --------------------------------------
# The previous version of this file derived the covered set from `Mt`, `upper` and
# `G` by re-deriving the coverage formula, and asserted the result was 0..Mt-1. That
# is a restatement of the pass's INTENT, so it agreed with the intent by
# construction and could not falsify it. It stayed green against a `tailStart` that
# was one row too high -- `HmxVectorReadoutPass.cpp:903` read
# `(upper / G) * G + 1`, which drops exactly one AR row on EVERY shape: row 28 of 32
# on the whole-matrix matmul, row 4 of 8 on grid BM=256, row 0 of 2 on grid BM=64.
# Thirty-two output rows of stale bytes per kernel, with a clean compile, no `error:`
# line, and read-out symbols present in the object. The reader below therefore reads
# each publish site's EMITTED `rowStart` and `rowCount` out of the descriptor stores
# and unions the intervals they name. It still folds `G` out of the in-loop boundary
# test, but only to place the group's end; the group's own start and row count come
# from what the pass wrote, so an off-by-one anywhere in the emission is visible here
# rather than being re-derived away.
# ---------------------------------------------------------------------------

def _readout_ir(kernel, grid, constexpr, **options):
    """The pass's own output IR for one shape, as text.

    Driven through the pipeline binary with `-mlir-print-ir-after-all` and sliced at
    `HmxVectorReadout`, because that is the only place the published rowStart and
    rowCount exist: they are stored into the descriptor, and the descriptor is a
    stack alloca the object no longer names.
    """
    opt = os.environ.get("TRITON_SHARED_OPT_PATH")
    if not opt:
        return None
    binary = opt.split("triton_shared/")[0] + "qcom_hexagon_backend/bin/linalg-hexagon-opt"
    if not os.path.exists(binary):
        return None

    a = torch.zeros(M, K, dtype=torch.float16)
    b = torch.zeros(K, N, dtype=torch.float16)
    c = torch.zeros(M, N, dtype=torch.float16)
    with tempfile.TemporaryDirectory(prefix="hmx-readout-ir-") as tmp:
        os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
        try:
            compiled = kernel.warmup(
                a, b, c, grid=grid, target_artifact="ttsharedir",
                N_ROWS=M, N_COLUMNS=N, N_INNER=K,
                # `{**a, **b}` so a later key WINS. `dict(a, **b, **c)`
                # raises when b and c share a key, which is exactly what
                # the depth-0 arm's `enableHmxPipelineDepth=0` does.
                **{**constexpr, **DEPTH_2, **options},
            )
        finally:
            os.environ.pop("TRITON_CACHE_DIR", None)
        src = Path(tmp) / "in.mlir"
        src.write_text(compiled.asm["ttsharedir"])

        out, err = Path(tmp) / "dump.txt", Path(tmp) / "dump.err"
        with open(out, "w") as f, open(err, "w") as e:
            # The same knob the production pipeline uses, so the IR this reads is
            # the IR the kernel was built from and not a reduced stand-in.
            subprocess.run(
                [binary, str(src),
                 "-linalg-to-llvm=enable-multi-threading=true "
                 "enable-vtcm-tiling=true enable-convert-to-hexagonmem=true "
                 "enable-hexagonmem-copy-to-dma=true enable-weight-resident=true "
                 "enable-hmx-pipeline-depth=2 enable-hmx-vector-readout=true",
                 "-mlir-print-ir-after-all", "-o", "/dev/null"],
                stdout=f, stderr=e, check=True,
            )
        lines = out.read_text(errors="replace").splitlines()
        if not any("IR Dump After HmxVectorReadout" in ln for ln in lines):
            # This build prints the IR dumps on stderr, not stdout.
            lines += err.read_text(errors="replace").splitlines()

    body, grabbing = [], False
    for line in lines:
        if "IR Dump After HmxVectorReadout" in line:
            grabbing = True
            continue
        if grabbing and "IR Dump After" in line:
            break
        if grabbing:
            body.append(line)
    return "\n".join(body) or None


def _published_rows(ir):
    """The rows the pass ACTUALLY published, read out of the emitted IR.

    Returns a dict, or None if the IR is not readable as a read-out handoff at all.

    Keys, all read from the IR rather than derived:

      `mt`     dim 0 of the VTCM crouton array the descriptor stores the AR address
               of -- the number of accumulator rows that exist.
      `group`  the divisor of the emitted in-loop boundary test `(iv+1) % G == 0`.
      `upper`  the tile loop's exclusive upper bound, for the induction variable the
               in-loop publish is built from.
      `tail`   the tail site's emitted `(rowStart, rowCount)`, or None.
      `sites`  how many `exec_publish` calls the pass emitted in total.
      `covered`, `dupes`, `missing`  the row set those sites name.

    A publish site is read as the `rowStart`/`rowCount` pair stored into the
    descriptor immediately before its `exec_publish` call. Descriptor field indices
    are the ABI's, not this file's: rowStart 0, rowCount 1 (HmxReadoutHandoff.h:81-82,
    matching `bin/runtime/include/HmxVectorExecutor.h`). The other four fields are
    stored at indices 2..5, so an index-pinned match cannot confuse them.

    A `rowStart` that folds to a constant is the tail. A `rowStart` built as
    `index_cast(subi %iv, %c)` is an in-loop group, which names `[iv-(G-1), iv]` for
    the `iv` values where `(iv+1) % G == 0`; the loop's own bounds turn that into a
    concrete interval set rather than an assumption about how far the loop runs.
    """
    import re

    consts = {m.group(1): int(m.group(2))
              for m in re.finditer(r"(%\S+) = arith\.constant (-?\d+) :", ir)}

    def fold(name, depth=0):
        """An index SSA name as a number, through arith.constant/subi/muli."""
        if name is None or depth > 8:
            return None
        if name in consts:
            return consts[name]
        # The name must be pinned: an unbound search would find the first op of
        # that kind anywhere in the function, not this value's definition.
        pinned = re.escape(name)
        for pattern, op in (
            (r" = arith\.subi (%\S+), (%\S+) : index", "-"),
            (r" = arith\.muli (%\S+), (%\S+) : index", "*"),
        ):
            m = re.search(pinned + pattern, ir)
            if m:
                a, b = fold(m.group(1), depth + 1), fold(m.group(2), depth + 1)
                if a is None or b is None:
                    return None
                return a - b if op == "-" else a * b
        return None

    # Mt: the crouton array the descriptor stores the AR base address of.
    mt_m = re.search(r": memref<(\d+)x16x16x32x2xf16, 1>", ir)
    if not mt_m:
        return None
    mt = int(mt_m.group(1))

    # G: the divisor of the emitted boundary test. Read, never assumed -- it is the
    # batch size the caller asked for, and assuming it here would hide a pass that
    # used a different one than it published with.
    remsi = re.search(r"arith\.remsi (%\S+), (%\S+) : index", ir)
    group = fold(remsi.group(2)) if remsi else None
    if not group or group < 1:
        return None

    # Every publish site, with the rowStart/rowCount stores feeding it.
    sites, iv, tail, covered = [], None, None, {}

    def take(covered, start, count):
        for row in range(start, start + count):
            covered[row] = covered.get(row, 0) + 1

    for call in re.finditer(r"call @hexagon_runtime_hmx_exec_publish", ir):
        head = ir[:call.start()]
        # The LAST rowStart stored at field 0 before this call, and the LAST rowCount
        # stored at field 1: a later store to the same field overrides an earlier one,
        # so "last wins" is what the pass's own semantics are.
        starts = re.findall(
            r"%\S+ = arith\.index_cast (\S+) : index to i32\s*\n"
            r"\s*%\S+ = arith\.constant 0 : index\s*\n"
            r"\s*memref\.store %\S+, %\S+\[%\S+\]", head)
        counts = re.findall(
            r"%\S+ = arith\.constant (\d+) : i32\s*\n"
            r"\s*%\S+ = arith\.constant 1 : index\s*\n"
            r"\s*memref\.store %\S+, %\S+\[%\S+\]", head)
        if not starts or not counts:
            return None
        row_count = int(counts[-1])
        source = starts[-1]

        # In-loop group? Then rowStart is `index_cast(subi %iv, %c)`.
        sub = re.search(re.escape(source) +
                        r" = arith\.subi (%\S+), (%\S+) : index", ir)
        if sub:
            iv = sub.group(1)
            sites.append(("group", row_count))
            continue
        row_start = fold(source)
        if row_start is None:
            return None
        tail = (row_start, row_count)
        sites.append(("tail", row_count))
        take(covered, row_start, row_count)

    lower = upper = None
    if iv:
        # The loop whose induction variable the in-loop publish is built from. Named
        # by the IV rather than found by position: the inner N loop has a header too,
        # and picking the wrong one is how a coverage check ends up confident about
        # an interval nobody published.
        hdr = re.search(r"scf\.for " + re.escape(iv) + r" = (%\S+) to (%\S+) step", ir)
        if not hdr:
            return None
        lower, upper = fold(hdr.group(1)), fold(hdr.group(2))
        if lower is None or upper is None:
            return None
        # Batch k fires at iv = k*G - 1, so it only runs while k*G - 1 < upper, i.e.
        # k = 1 .. floor(upper/G); it names rows [(k-1)G, kG-1]. The start and the
        # count come from the site; G and the bounds only decide how many times the
        # site is reached.
        for k in range(1, upper // group + 1):
            take(covered, k * group - group, row_count_of_last_group(sites, group))

    return {
        "mt": mt, "group": group, "lower": lower, "upper": upper, "iv": iv,
        "tail": tail, "sites": sites,
        "covered": sorted(covered),
        "dupes": sorted(r for r, n in covered.items() if n > 1),
        "missing": sorted(set(range(mt)) - set(covered)),
    }


def row_count_of_last_group(sites, group):
    """The rowCount the in-loop group sites published.

    Read from the sites rather than assumed to be `group`, so that a pass which
    published groups of a different size than it divides the loop by is reported as
    the wrong answer it is instead of being normalised into the expected one.
    """
    for kind, count in reversed(sites):
        if kind == "group":
            return count
    return group


class VectorReadoutReachesTheObject(unittest.TestCase):
    """The feature, asserted on the object a real Triton compile produced.

    Generated over every shape rather than written once per shape, so a shape cannot
    be added to the negative arms and quietly omitted from the positive one: both
    arms iterate the same `SHAPES` tuple.
    """

    @classmethod
    def setUpClass(cls):
        cls.compiled = {}
        for name, kernel, grid, constexpr in SHAPES:
            obj, stderr = _compile_obj(kernel, grid, constexpr, **READOUT_ON)
            cls.compiled[name] = (obj, stderr, _nm(obj))

    def test_the_compile_itself_is_clean(self):
        # The check that would have caught the original defect. The pass failed
        # with "HMX bridge operation has no explicit hmx.decision_id" and the
        # failure surfaced only on stderr.
        for name, (_obj, stderr, _syms) in self.compiled.items():
            errors = [ln for ln in stderr.splitlines() if "error:" in ln]
            self.assertEqual(
                [], errors,
                f"[{name}] compiling with enableHmxVectorReadout=True must produce "
                f"no error: diagnostics; got {errors}",
            )

    def test_the_object_is_not_empty(self):
        # Guards the gate itself: if llvm-nm ever stopped parsing Hexagon objects
        # and returned nothing, every assertion below would pass vacuously.
        for name, (_obj, _stderr, symbols) in self.compiled.items():
            self.assertGreater(
                len(symbols), 50,
                f"[{name}] llvm-nm returned only {len(symbols)} symbols; the gate "
                "would be reading nothing",
            )
            self.assertIn("hexagon_runtime_hmx_ensure_dsp", symbols)

    def test_the_outlined_readout_is_in_the_object(self):
        # The assertion whose absence is why seven sweeps read "no speedup".
        for name, (_obj, _stderr, symbols) in self.compiled.items():
            missing = _split_is_in_object(symbols)
            self.assertEqual([], missing, f"[{name}] " + "; ".join(missing))

    def test_the_runtime_entry_points_are_referenced(self):
        # Undefined here, resolved at device link. Asserted separately from the
        # definitions above because the two mean different things: `T` is code
        # this kernel carries, `U` is a demand it places on the runtime archive.
        for name, (_obj, _stderr, symbols) in self.compiled.items():
            for required in REFERENCED_REQUIRED:
                self.assertEqual(
                    "U", symbols.get(required),
                    f"[{name}] {required} should be an undefined reference resolved "
                    f"at device link, got {symbols.get(required)!r}",
                )

    def test_every_accumulator_row_is_published_exactly_once(self):
        """No row read twice, none dropped, on any shape.

        The one failure in this feature that is invisible from the outside: the
        outlined read-out walks `row0 .. row0+nrows-1`, so a wrong `rowStart` is a
        wrong answer that still links, still runs and still looks plausible. The
        row set is READ BACK OUT of the descriptor stores the pass emitted, not
        recomputed from the pass's intended formula and not taken from the object.

        The two ways this test can go wrong are both handled rather than skipped:

          * the IR is unreadable (no pipeline binary, an unexpected shape) -- that is
            environmental, so it SKIPS;
          * the IR is readable and the pass published NOTHING -- that is the pass
            declining on a shape it is supposed to handle, which is a FAILURE and the
            original defect. It is asserted before the coverage assertions below so
            the message says "published nothing" instead of "every row is missing".
        """
        for name, kernel, grid, constexpr in SHAPES:
            ir = _readout_ir(kernel, grid, constexpr, **READOUT_ON)
            if ir is None:
                self.skipTest(f"[{name}] the pipeline binary or its IR dump is "
                              "unavailable, so the published rows cannot be read")
            read = _published_rows(ir)
            self.assertIsNotNone(
                read,
                f"[{name}] could not read Mt / the batch size / the emitted rowStart "
                f"and rowCount out of the read-out pass's own IR, so the row-coverage "
                f"check would be a guess",
            )
            mt = read["mt"]
            group = read["group"]
            upper = read["upper"]
            tail = read["tail"]

            # Not a skip: a declined pass is a failed pass.
            self.assertTrue(
                read["sites"],
                f"[{name}] enableHmxVectorReadout=True at pipeline-depth {DEPTH_2['enableHmxPipelineDepth']} "
                f"emitted ZERO exec_publish calls: the read-out pass declined on this "
                f"shape, so the kernel carries none of the feature. Declining is "
                f"harmless but invisible -- this is the failure that made seven "
                f"measurement sweeps read as 'no speedup'.",
            )
            covered, dupes, missing = read["covered"], read["dupes"], read["missing"]
            self.assertEqual(
                [], dupes,
                f"[{name}] rows published more than once (Mt={mt} upper={upper} "
                f"G={group} tail={tail}): {dupes}",
            )
            self.assertEqual(
                [], missing,
                f"[{name}] accumulator rows never published, so those destination "
                f"rows keep whatever was there before the launch (Mt={mt} "
                f"upper={upper} G={group} emitted tail={tail}): {missing}",
            )
            self.assertEqual(
                list(range(mt)), covered,
                f"[{name}] the published rows must be exactly 0..Mt-1 "
                f"(Mt={mt} upper={upper} G={group} emitted tail={tail}), got {covered}",
            )


class TheGateIsNotVacuous(unittest.TestCase):
    """The gate must FAIL on a build where the pass is inert.

    Two independent ways the pass can be inert, both of which produce a perfectly
    good-looking object with none of the feature in it:

      * the option explicitly off, and
      * the option ON at pipeline-depth 0 on a shape the read-out channel
        declines (GRID_SMALL, 2 m-tiles), where the split has no loop it
        matches so the object is code-identical to the option-off build.

    The second is the trap: it is a build with `enableHmxVectorReadout=True`, so a
    gate that only checked "the option was on and an object came out" would pass.
    """

    def test_option_off_object_has_none_of_the_feature(self):
        # The option is EXPLICITLY off. It was the default once, and the arm
        # relied on that; the default flipped to on (2026-10-04), and this arm
        # silently became "the default build" -- which carries the feature --
        # until the explicit False was added. A negative arm that follows a
        # default is not a negative arm.
        for name, kernel, grid, constexpr in SHAPES:
            symbols = _nm(_compile_obj(kernel, grid, constexpr,
                                       enableHmxVectorReadout=False)[0])
            missing = _split_is_in_object(symbols)
            self.assertNotEqual(
                [], missing,
                f"[{name}] the gate predicate must reject the option-off object, or "
                "it cannot distinguish a working build from a dead one",
            )

    def test_depth_zero_is_a_silent_noop_and_the_gate_rejects_it(self):
        # The trap, re-scoped to the shapes where it still is one. At
        # pipeline-depth 0 (`auto`) the read-out channel opens only for a
        # matmul with at least `stagedReadoutMTiles` m-tiles (2 x batch = 8),
        # so the two big shapes stage and the split attaches; GRID_SMALL (2
        # m-tiles) stays on the unstaged serial form, the split declines, the
        # kernel runs the original serial read-out, and the build looks
        # entirely healthy: no error, an object, a correct answer.
        #
        # (When this arm was written, depth 0 declined all three shapes; the
        # channel wiring has since made depth 0 a real request for the shapes
        # the channel covers, so the arm now names the shape it depends on
        # rather than all of them.)
        for name, kernel, grid, constexpr in SHAPES:
            if name != "grid-tiled-small":
                continue
            symbol = kernel.fn.__name__ if hasattr(kernel, "fn") else "matmul_kernel"
            on_zero, _ = _compile_obj(kernel, grid, constexpr,
                                      enableHmxPipelineDepth=0, **READOUT_ON)
            off_zero, _ = _compile_obj(kernel, grid, constexpr,
                                       enableHmxPipelineDepth=0,
                                       enableHmxVectorReadout=False)

            # "No-op" is asserted as identical kernel CODE, not as identical bytes.
            # Byte equality is not available here: the object embeds runtime bitcode
            # whose section order is emitted non-deterministically (see
            # _kernel_disassembly), so two compiles of the SAME options differ in md5.
            self.assertEqual(
                _kernel_disassembly(on_zero, symbol), _kernel_disassembly(off_zero, symbol),
                f"[{name}] at pipeline-depth 0 the read-out pass has no m-tile loop to "
                "attach to, so the kernel code must be identical with the option on "
                "and off",
            )
            # And the gate's own predicate has to reject it. This is the assertion
            # that makes the gate non-vacuous: same binary, same shape, option
            # explicitly ON, and the feature is still absent.
            self.assertNotEqual(
                [], _split_is_in_object(_nm(on_zero)),
                f"[{name}] the gate predicate must reject the depth-0 option-ON "
                "object; otherwise it cannot tell a working build from a dead one",
            )


class TheManifestDoesNotDoubleCount(unittest.TestCase):
    """`pack_act_sites`/`unpack_sites` must not double count across the split.

    The outlined read-out is a NEW `hmx.unpack_acc` in a new function, so the
    bridge recount walks it. The recount attributes it to the ENGINE's record --
    resolved through `hmx.readout.handoffs`, which names the kernel each outlined
    function serves -- rather than skipping it, because skipping would report
    `unpack_sites = 0` for a record whose read-out is still in the module.

    The numbers that follow are what "no double count" means concretely:

      * pack_act_sites is UNCHANGED. The split never creates or destroys a pack, so
        any movement here would mean the recount is attributing an op twice.
      * unpack_sites goes DOWN, from 2 inline sites to 1 outlined site. Two sites
        replaced by one is the direction a double count cannot take; a double
        count would show up as 2 -> 4.

    The two grid shapes are covered alongside the whole-matrix one because the
    count they report differs for a structural reason worth pinning: a shape with no
    peeled epilogue (BLOCK_M = 64) has one inline site, not two. Asserting `2 -> 1`
    for that shape would be asserting a fact about the peel, not about the split, so
    the direction is what is checked and the numbers are reported in the message.
    """

    def test_bridge_counts(self):
        for name, kernel, grid, constexpr in SHAPES:
            off = _manifest(kernel, grid, constexpr)["matmuls"][0]["execution"]["bridge_counts"]
            on = _manifest(kernel, grid, constexpr, **READOUT_ON)["matmuls"][0]["execution"]["bridge_counts"]
            self.assertEqual(
                off["pack_act_sites"], on["pack_act_sites"],
                f"[{name}] the split must not change the pack count: {off} -> {on}",
            )
            self.assertEqual(
                1, on["unpack_sites"],
                f"[{name}] exactly one read-out bridge site must remain, counted "
                f"against the kernel's record: {off} -> {on}",
            )
            self.assertLessEqual(
                on["unpack_sites"], off["unpack_sites"],
                f"[{name}] the split must not INCREASE the read-out site count "
                f"({off} -> {on}); a rise is a bridge op attributed to two records",
            )


class TheMultiMatmulReach(unittest.TestCase):
    """One group per matmul -- present where it pays, declined where it cannot.

    Both assertions are object-level (see the module comment for why IR-level
    checks once passed on a build whose feature never ran), and the two are a
    pair: the presence arm proves the multi-matmul grouping works at all, the
    flash-attention arm proves the decline it takes instead is the one the
    measurements ordered -- not the feature silently absent. Either arm alone
    could pass on the other's failure.
    """

    def test_a_top_level_two_matmul_function_carries_both_readouts(self):
        obj, stderr = _two_matmul_compile_obj()
        errs = [l for l in stderr.splitlines() if "error" in l.lower()]
        self.assertEqual([], errs, "the two-matmul compile must be clean")
        symbols = _nm(obj)
        missing = []
        for name in TWO_MM_DEFINED_REQUIRED:
            if symbols.get(name) not in ("T", "t"):
                missing.append(f"{name} is not defined (got {symbols.get(name)!r})")
        for name in REFERENCED_REQUIRED:
            if name not in symbols:
                missing.append(f"{name} is absent entirely")
        self.assertEqual(
            [], missing,
            "a top-level function holding two matmuls must carry one read-out "
            "each; a missing symbol is the multi-matmul grouping silently "
            "absent, not a slower kernel",
        )

    def test_flash_attention_is_declined_and_that_is_the_measured_answer(self):
        # See the comment at `_import_fa_kernel` for the full chain. The short
        # form: flash attention's matmuls sit inside the attention loop, the
        # staged ring's fixed cost is per tile-loop execution and measurably
        # loses to the unstaged serial loop there, and the unstaged form has
        # no engine work overlapping the read-out -- so the correct object has
        # no read-out symbols at all, with the option on and the compile clean.
        obj, stderr = _fa_compile_obj()
        errs = [l for l in stderr.splitlines() if "error" in l.lower()]
        self.assertEqual([], errs, "the FA compile must be clean")
        symbols = _nm(obj)
        present = [name for name in FA_ABSENT_REQUIRED if name in symbols]
        self.assertEqual(
            [], present,
            "flash attention must take the unstaged serial form and be "
            "declined by the read-out split; read-out symbols here would mean "
            "the staging channel opened for a loop-nested matmul again (the "
            "measured +5%)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)