#!/usr/bin/env python3
"""The OBJECT-level gate for hexagon-l2-prefetch (host only, no device).

WHY THIS FILE EXISTS
--------------------
The read-out split taught the lesson this file is a copy of: a feature can be
green at every IR level and still be absent from the binary. So the gate for
the L2 prefetch looks at the OBJECT, and specifically for the `l2fetch`
INSTRUCTION in the disassembly -- there is no symbol to `llvm-nm` for, the
fetch is an instruction the pipeline has to carry from an MLIR-level
`llvm.call @llvm.hexagon.Y5.l2fetch` all the way through translation and
ISel.

WHAT IS ASSERTED
----------------
  * The compile itself is clean: no `error:` on stderr, in either arm. The
    fd-2 capture is direct (not contextlib) because the diagnostics come from
    C++ inside libtriton.
  * DEFAULT arm (no options): the kernel's disassembly contains the fetches.
    vec_add has two streams, so at least 2 `l2fetch` instructions. The
    default flipped ON 2026-10-05 after the device A/B passed (5.5x flat /
    1.62x nested / -33% near-L2, neutral elsewhere, numerics unchanged --
    docs/results/l2-prefetch-2026-10-04.md), so "the default build contains
    the feature" is now the gate's positive assertion.
  * OFF arm (`enableL2Prefetch=False`): ZERO `l2fetch` instructions. This is
    the negative arm and the point: the opt-out must actually opt out, and a
    default regression in either direction only ever shows up as an
    unexplained perf change in some other measurement.

WHY THREE SHAPES
---------------
The pass handles the three streaming forms the pipeline produces, and they
fail differently if the pass regresses:

  * `add_kernel` (BLOCK_SIZE=131072, grid=(1,)) is the FLAT form: the loop
    slices the function-argument views directly.
  * `silu_kernel` at grid=(4,), BLOCK_SIZE=4096 is the GRID-STRIDED form: the
    stream base is `pid * BLOCK_SIZE`, a runtime value, so the fetch address
    goes through extract_strided_metadata. If the matcher ever stops
    accepting a dynamic source offset, this arm goes to zero fetches while
    the flat arm stays green -- one shape would not catch that.
  * `rowslice_kernel` (one program walking [128, 128] f32 by whole rows
    through a block pointer) is the 2-D ROW-SLICE form the 2026-10-05 (C1)
    matcher generalization added: the loop's slices are 2-D subviews
    `[%iv, 0] [1, C]` of a dense rank-2 source, with the source stride
    (sigma = C) scaling the IV. If the matcher ever regresses to rank-1
    sources only, this arm goes to zero fetches while both other arms stay
    green.

The objdump is the SDK's `hexagon-llvm-objdump` (via HEXMLIR_SDK_BIN from
tools/hexmlir/env.sh), falling back to the pinned LLVM's llvm-objdump, which
also decodes l2fetch (verified on this build). The SYSTEM llvm-objdump does
not decode Hexagon packet encodings -- it prints `<unknown>` -- so it is not
a candidate.

Run:  source tools/hexmlir/env.sh
      .venv/bin/python hexagon-mlir/qcom_hexagon_backend/test/test_l2_prefetch_object_gate.py
"""

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

# The shapes dump_codegen compiles for these kernels, so this gate checks the
# objects a reader would actually be measuring.
VEC_ADD_N = 131072
SILU_N = 16384
SILU_BLOCK = SILU_N // 4
# The row-slice arm: [ROWS, COLS] f32 = 64 KiB, mapped as one 2-D block.
# The constants make the stream long enough to fire (16384 elements against
# the 2560-element distance+block) and the per-iteration advance a divisor
# of the 512-element fetch block, so the window-entry guard has a whole
# period (one fire per 512 elements = every 4th row).
ROWS, COLS = 128, 128

VEC_ADD_OPTS = dict(enableMultiThreading=True, enableVTCMTiling=True,
                    enableConvertToHexagonmem=True, enableHexagonmemCopyToDMA=True)
SILU_OPTS = dict(enableMultiThreading=False, enableVTCMTiling=False,
                 enableConvertToHexagonmem=False, enableHexagonmemCopyToDMA=False)
# The linattn bench configuration: the option set the kernel-attention arms
# are measured under, and the one the row-slice coverage was verified with.
ROWSLICE_OPTS = dict(enableMultiThreading=False, enableVTCMTiling=True,
                     enableConvertToHexagonmem=True,
                     enableHexagonmemCopyToDMA=True, enableHVXInlining=True,
                     enableSCFLoopUnroll=True, enableSplitReduceGeneric=True)
OFF = dict(enableL2Prefetch=False)


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    """Verbatim from tools/hexmlir/dump_codegen.py (copied, not re-expressed,
    so the gate compiles the kernel a reader would measure)."""
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(output_ptr + offsets, x + y)


@triton.jit
def silu_kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    """Verbatim from tools/hexmlir/dump_codegen.py: the `pid * BLOCK_SIZE`
    offsets are what make this the dynamic-base streaming form."""
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + (pid * BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    output = x / (1 + tl.exp((-x).to(tl.float32)).to(x.dtype))
    tl.store(output_ptr + offsets, output)


@triton.jit
def rowslice_kernel(x_ptr, output_ptr, R: tl.constexpr, C: tl.constexpr):
    """A whole-tensor 2-D elementwise map, the way the linear-attention
    kernels read their Q/K/V (bench_ops.py la_naive: `offs[:, None] * D +
    offs[None, :]`): one program, the whole [R, C] block as one 2-D load.
    The vectorizer lowers this to an outer row loop x inner column loop
    whose slices are 2-D subviews `[%row, %col] [1, w]` of the row-major
    source -- the form the C1 matcher generalization exists for (before it,
    a rank-2 source was declined outright)."""
    offs_r = tl.arange(0, R)
    offs_c = tl.arange(0, C)
    x = tl.load(x_ptr + offs_r[:, None] * C + offs_c[None, :])
    tl.store(output_ptr + offs_r[:, None] * C + offs_c[None, :], x + 1.0)


def _compile_obj(kernel, args, grid, **options):
    """Compile one kernel to a device object, returning (bytes, stderr).

    `kernel.warmup` with `target_artifact="o"` compiles and does NOT launch:
    the difference between `warmup` and `kernel[grid](...)` is one character
    and the second one talks to the DSP. No device is touched here.
    """
    with tempfile.TemporaryDirectory(prefix="l2pf-gate-") as tmp:
        # An isolated kernel cache, so this test never reads a stale entry
        # for the same options and never pollutes the shared one.
        previous_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
        # fd 2 is captured directly rather than through
        # contextlib.redirect_stderr: the diagnostics come from C++
        # (llvm::errs()) inside libtriton and are not routed through
        # sys.stderr.
        captured = os.dup(2)
        path = os.path.join(tmp, "stderr")
        sink = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.dup2(sink, 2)
            compiled = kernel.warmup(
                *args, grid=grid, target_artifact="o", **options
            )
        finally:
            os.dup2(captured, 2)
            os.close(sink)
            os.close(captured)
            if previous_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous_cache
        with open(path) as fh:
            stderr = fh.read()
    # `warmup` returns the CompiledKernel; the object itself rides in its
    # `asm` dictionary under "o" (same channel the ttsharedir/llir artifacts
    # come back through).
    return compiled.asm["o"], stderr


def _find_objdump():
    """An objdump that decodes Hexagon packet encodings.

    The SDK's tool first (the one tools/hexmlir/disasm.sh uses), the pinned
    LLVM's second. The system llvm-objdump prints `<unknown>` for Hexagon
    instructions and would make this gate vacuously red (or worse, green
    against a fetch it cannot name).
    """
    candidates = []
    sdk_bin = os.environ.get("HEXMLIR_SDK_BIN")
    if sdk_bin:
        candidates.append(Path(sdk_bin) / "hexagon-llvm-objdump")
    found = shutil.which("hexagon-llvm-objdump")
    if found:
        candidates.append(Path(found))
    # test/ -> qcom_hexagon_backend -> hexagon-mlir -> workspace root
    candidates.append(
        Path(__file__).resolve().parents[3] / "llvm_triton/build/bin/llvm-objdump")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _disassemble(obj):
    objdump = _find_objdump()
    if objdump is None:
        raise unittest.SkipTest(
            "no Hexagon-capable objdump (HEXMLIR_SDK_BIN or the pinned LLVM); "
            "this gate reads instructions")
    with tempfile.TemporaryDirectory(prefix="l2pf-dis-") as tmp:
        path = Path(tmp) / "kernel.o"
        path.write_bytes(obj)
        result = subprocess.run(
            [objdump, "-d", "--no-show-raw-insn", str(path)],
            capture_output=True, text=True, check=True)
    return result.stdout


def _l2fetch_count(disasm):
    return sum(1 for ln in disasm.splitlines() if "l2fetch" in ln)


class L2PrefetchObjectGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.objdump = _find_objdump()
        x = torch.rand(VEC_ADD_N)
        y = torch.rand(VEC_ADD_N)
        o = torch.empty(VEC_ADD_N)
        sx = torch.rand(SILU_N)
        so = torch.empty(SILU_N)
        rx = torch.rand(ROWS, COLS)
        ro = torch.empty(ROWS, COLS)
        # (bytes, stderr) per arm; the arms differ ONLY in the option.
        # The default arm passes no option at all: after the 2026-10-05 flip
        # the default must carry the fetches, and the explicit OFF arm is
        # the control that proves the difference is the option's.
        cls.vec_default = _compile_obj(
            add_kernel, (x, y, o), (1,), **VEC_ADD_OPTS,
            BLOCK_SIZE=VEC_ADD_N)
        cls.vec_off = _compile_obj(
            add_kernel, (x, y, o), (1,), **{**VEC_ADD_OPTS, **OFF,
                                            "BLOCK_SIZE": VEC_ADD_N})
        cls.silu_default = _compile_obj(
            silu_kernel, (sx, so), (4,), **SILU_OPTS,
            BLOCK_SIZE=SILU_BLOCK)
        cls.silu_off = _compile_obj(
            silu_kernel, (sx, so), (4,), **{**SILU_OPTS, **OFF,
                                            "BLOCK_SIZE": SILU_BLOCK})
        cls.row_default = _compile_obj(
            rowslice_kernel, (rx, ro), (1,), **ROWSLICE_OPTS,
            R=ROWS, C=COLS)
        cls.row_off = _compile_obj(
            rowslice_kernel, (rx, ro), (1,), **{**ROWSLICE_OPTS, **OFF,
                                                 "R": ROWS, "C": COLS})

    def test_the_compile_is_clean(self):
        # The check that would have caught the read-out's original defect:
        # a hard pass failure only ever showed on fd 2.
        for name, (_, stderr) in (("vec_add default", self.vec_default),
                                  ("vec_add off", self.vec_off),
                                  ("silu default", self.silu_default),
                                  ("silu off", self.silu_off),
                                  ("rowslice default", self.row_default),
                                  ("rowslice off", self.row_off)):
            self.assertNotIn("error:", stderr, f"[{name}] {stderr[-2000:]}")

    def test_the_flat_shape_carries_the_fetches(self):
        if self.objdump is None:
            self.skipTest("no Hexagon-capable objdump")
        disasm = _disassemble(self.vec_default[0])
        # Two streams, so at least two fetches; more would mean the async
        # chunking cloned the guarded fetch block, which is also correct.
        self.assertGreaterEqual(
            _l2fetch_count(disasm), 2,
            "the flat streaming loop's fetches did not reach the object "
            "through the default options")

    def test_the_dynamic_base_shape_carries_the_fetches(self):
        if self.objdump is None:
            self.skipTest("no Hexagon-capable objdump")
        disasm = _disassemble(self.silu_default[0])
        self.assertGreaterEqual(
            _l2fetch_count(disasm), 1,
            "the grid-strided (dynamic base) loop's fetches did not reach "
            "the object through the default options")

    def test_the_row_slice_shape_carries_the_fetches(self):
        if self.objdump is None:
            self.skipTest("no Hexagon-capable objdump")
        disasm = _disassemble(self.row_default[0])
        self.assertGreaterEqual(
            _l2fetch_count(disasm), 1,
            "the 2-D row-slice loop's fetches did not reach the object: the "
            "rank-2 source form regressed to a decline (before the 2026-10-05 "
            "generalization this shape was declined outright)")

    def test_the_off_arm_carries_none(self):
        # The negative arm, and the reason this gate can be trusted: the
        # opt-out must contain none of the feature, so a green default arm
        # is a difference the option made, not an artifact that was always
        # there.
        if self.objdump is None:
            self.skipTest("no Hexagon-capable objdump")
        for name, (obj, _) in (("vec_add", self.vec_off),
                               ("silu", self.silu_off),
                               ("rowslice", self.row_off)):
            count = _l2fetch_count(_disassemble(obj))
            self.assertEqual(
                count, 0,
                f"[{name}] l2fetch in the option-off object: the opt-out is "
                f"no longer honest ({count} instructions)")


if __name__ == "__main__":
    unittest.main()
