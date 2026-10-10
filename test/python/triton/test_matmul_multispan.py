#!/usr/bin/env python3
"""The multi-span matmul form, in the device gate (`hexmlir-all`).

WHY THIS FILE EXISTS
--------------------
The standard gate's matmul is a single tile with zero offsets, so it could
not see the span dimension at all: with one span the destination offset is
zero and any offset bug is invisible. The epilogue-fold defect had exactly
that signature -- every read-out batch published the bare aligned pointer
of C instead of its own span view's address, so all 16 spans of a 2048^3
matmul wrote the top-left 512x512 block and rel came out 0.96817 where
sqrt(15/16) = 0.96825 ("only 1/16 of the output is non-zero"). The gate
was green throughout (`docs/results/device-mega-validation-2026-10-10.md`
section 2; claim C48 in `docs/state/CLAIMS.md`).

The minimal form is M=N=2048 tiled BM=BN=512 with whole K: a 4x4 grid of
spans inside one program, sixteen independent 512x512 read-outs, sixteen
different destination offsets. It is the shape the defect was measured on.

WHAT IS ASSERTED (and why each one)
-----------------------------------
* engine: the manifest plans the contraction `full-hmx`.
* the read-out split is live: the outlined `__hmx_readout` and the executor
  handoff (`hexagon_runtime_hmx_exec_configure`) are in the artifact. If
  the pass declines, the store falls back to a serial write, the numbers
  stay right, and this file must not pretend it covered anything -- the
  same vacuity the host object gate guards against
  (`test_hmx_vector_readout_object_gate.py`).
* the span ABI: the unpack leaf is called with the OUTPUT's row stride
  (2048), not the tile width (512). That difference only exists when the
  output has more than one n-span, so the assertion is false for the
  single-span gate matmul and true only for the multi-span form.
* numerics on the DSP: rel against the f32 torch reference must stay in
  the f16-clean class (~2.2e-4 measured for this shape). The fold defect
  signed 9.7e-01; the gate is 2e-3, four orders of magnitude below it.

Run:  tools/run_tests.sh hexmlir test/python/triton/test_matmul_multispan.py
"""

import json
import re

import torch
import triton
import triton.language as tl

# The repo matmul test's launch options, verbatim.
OPTIONS = dict(
    enableMultiThreading=True,
    enableVTCMTiling=True,
    enableConvertToHexagonmem=True,
    enableHexagonmemCopyToDMA=True,
)

# 4x4 spans of 512x512 over a 2048x2048 output, whole K (BLOCK_K == K), so
# the K dimension contributes exactly one contraction per span and the
# span dimension is the only thing under test.
NUM_ROWS = 2048
NUM_COLUMNS = 2048
NUM_INNER = 2048
BLOCK_M = 512
BLOCK_N = 512
M_SPANS = NUM_ROWS // BLOCK_M
N_SPANS = NUM_COLUMNS // BLOCK_N
SPANS = M_SPANS * N_SPANS

# Measured f16-clean for this shape is 2.198e-04; the fold defect signed
# 9.682e-01. 2e-3 is ~9x the clean anchor and ~500x below the defect.
REL_TOL = 2e-3

# The read-out leaf writes (dst, src, rows, cols, dst_stride, row, count):
# a 512-wide tile into a 2048-stride output is the multi-span ABI. A
# single-span dense output would pass 512 for both. The first two operands
# are SSA names (`%27`, `%.unpack10`), hence the character class.
SPAN_READOUT = re.compile(
    r"call void @hmx_unpack_acc_f16_bulk\(i32 [\w%.]+, i32 [\w%.]+, "
    r"i32 512, i32 512, i32 2048,"
)


@triton.jit
def multispan_matmul(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # The tile loops live in the kernel (one program), the way the shape
    # that exposed the fold defect was written: each iteration is one span,
    # and each span's store is a different window of C.
    for m0 in range(0, M, BLOCK_M):
        for n0 in range(0, N, BLOCK_N):
            rows = m0 + tl.arange(0, BLOCK_M)
            cols = n0 + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, K)
            a = tl.load(A + rows[:, None] * K + offs_k[None, :])
            b = tl.load(B + offs_k[:, None] * N + cols[None, :])
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            acc = tl.dot(a, b, acc)
            tl.store(
                C + rows[:, None] * N + cols[None, :],
                acc.to(C.type.element_ty),
            )


def test_multispan_matmul_on_device():
    torch.manual_seed(0)
    mat_A = torch.rand((NUM_ROWS, NUM_INNER), dtype=torch.float16)
    mat_B = torch.rand((NUM_INNER, NUM_COLUMNS), dtype=torch.float16)
    mat_C = torch.zeros((NUM_ROWS, NUM_COLUMNS), dtype=torch.float16)

    # `warmup` compiles and never launches: the difference between it and
    # `multispan_matmul[(1,)](...)` below is one character, and the second
    # one talks to the DSP.
    compiled = multispan_matmul.warmup(
        mat_A,
        mat_B,
        mat_C,
        grid=(1,),
        target_artifact="llir",
        htp_kernel_gen=True,
        M=NUM_ROWS,
        N=NUM_COLUMNS,
        K=NUM_INNER,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        **OPTIONS,
    )
    llir = compiled.asm["llir"]
    manifest = json.loads(compiled.packed_metadata["hmx_manifest"])
    plans = [(rec.get("plan"), rec.get("reason")) for rec in manifest["matmuls"]]
    assert plans == [("full-hmx", "selected-aligned")], f"HMX plans: {plans}"

    # The epilogue-fold read-out split, without which this form silently
    # degrades to a serial store and covers nothing.
    assert "define void @__hmx_readout" in llir, (
        "the read-out split is not in the artifact: the spans were written "
        "serially, so the span path this file exists for did not run"
    )
    assert "@hexagon_runtime_hmx_exec_configure" in llir, (
        "the read-out executor handoff is missing"
    )
    match = SPAN_READOUT.search(llir)
    assert match, (
        "no read-out call writes a 512-wide tile into the 2048-stride "
        "output: the per-span destination ABI is not what this shape needs"
    )

    print(
        f"Multi-span matmul {NUM_ROWS}x{NUM_INNER}x{NUM_COLUMNS}, "
        f"tile {BLOCK_M}x{BLOCK_N}x{NUM_INNER} "
        f"({M_SPANS}x{N_SPANS} = {SPANS} spans), plans={plans}"
    )
    multispan_matmul[(1,)](
        mat_A,
        mat_B,
        mat_C,
        NUM_ROWS,
        NUM_COLUMNS,
        NUM_INNER,
        BLOCK_M,
        BLOCK_N,
        **OPTIONS,
    )

    reference = torch.matmul(mat_A.float(), mat_B.float())
    rel = ((mat_C.float() - reference).norm() / reference.norm()).item()
    print(f"rel={rel:.3e} (gate {REL_TOL:.0e})")
    assert rel < REL_TOL, (
        f"rel {rel:.3e} exceeds the f16-clean gate {REL_TOL:.0e}: a span "
        "regression (the epilogue-fold defect signed 9.7e-01: every span "
        "writing the top-left block)"
    )
