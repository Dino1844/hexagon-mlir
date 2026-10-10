#!/usr/bin/env python3
"""The K-looped matmul form, in the device gate (`hexmlir-all`).

WHY THIS FILE EXISTS
--------------------
The standard gate's matmul (`test_matmul.py`) is a whole-K single `tl.dot`,
so until now the gate contained **no K-looped matmul at all**. Two defect
classes live exactly in that shape and both sailed through a green
`hexmlir-all`:

* load-death: on the kchunk builds every K-looped form died in the device
  loader (`Error -2147482611: Failed to call main() on DSP`, 5/5 attempts)
  while every whole-K form on the same build ran fine
  (`docs/results/device-window-2026-10-10.md` section 2);
* numerics: while the 1D-DMA stage shear was in place, a two-segment K loop
  produced a uniform ~2% noise (rel 2.02e-02) while every k=1 form stayed at
  the f16 floor (`docs/analysis/a-kloop-rel-investigation-2026-10-09.md`
  section 1 -- the 2x2x2 factor table).

The minimal form is M=N=256, K=512 in two segments of BLOCK_K=256: one
m/n tile, one accumulator carried across the loop, two engine
contractions. It compiles and reaches the engine in about a second, so it
costs the gate almost nothing.

WHAT IS ASSERTED (and why each one)
-----------------------------------
* engine: the manifest plans the contraction `full-hmx`. A silent fallback
  to HVX would still be numerically right and must not count as coverage.
* K-chunk accumulator: the read-out is `hmx_unpack_acc_f32_bulk` (the
  engine's f16 read-out lands straight in the resident f32 accumulator,
  with the accumulator as residual) and there is **no** per-segment f16
  image. That is the mechanism whose VTCM growth is the load-death
  suspect, so a build that reverts to the per-segment round trip should
  fail here rather than in someone's device window.
* numerics on the DSP: rel against the f32 torch reference must stay in
  the f16-clean class. The gate is 2e-3: about 7x the measured clean
  anchor (3.0e-04 on a two-segment form) and about 10x below the shear
  signature (2.0e-02), so neither known defect class can pass.

Run:  tools/run_tests.sh hexmlir test/python/triton/test_matmul_kloop.py
"""

import json

import torch
import triton
import triton.language as tl

# The repo matmul test's launch options, verbatim: the gate should measure
# the configuration a user actually runs.
OPTIONS = dict(
    enableMultiThreading=True,
    enableVTCMTiling=True,
    enableConvertToHexagonmem=True,
    enableHexagonmemCopyToDMA=True,
)

# Two K segments: the minimal K-looped form. BLOCK_K == K/2, so the loop
# body runs exactly twice and the accumulator is carried once.
NUM_ROWS = 256
NUM_COLUMNS = 256
NUM_INNER = 512
BLOCK_K = 256
K_SEGMENTS = NUM_INNER // BLOCK_K

# f16-clean measured 2.2e-4..3.0e-4; the shear bug signed 2.0e-02 and the
# span bug 9.7e-01. 2e-3 sits an order of magnitude from either side.
REL_TOL = 2e-3


@triton.jit
def klooped_matmul(
    A,
    B,
    C,
    N_COLUMNS: tl.constexpr,
    N_INNER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * N_INNER + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N_COLUMNS + offs_n[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _k in range(0, N_INNER, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N_COLUMNS
    tl.store(
        C + offs_m[:, None] * N_COLUMNS + offs_n[None, :],
        acc.to(C.type.element_ty),
    )


def test_klooped_matmul_on_device():
    torch.manual_seed(0)
    mat_A = torch.rand((NUM_ROWS, NUM_INNER), dtype=torch.float16)
    mat_B = torch.rand((NUM_INNER, NUM_COLUMNS), dtype=torch.float16)
    mat_C = torch.zeros((NUM_ROWS, NUM_COLUMNS), dtype=torch.float16)

    # `warmup` compiles and never launches: the difference between it and
    # `klooped_matmul[(1,)](...)` below is one character, and the second
    # one talks to the DSP.
    compiled = klooped_matmul.warmup(
        mat_A,
        mat_B,
        mat_C,
        grid=(1,),
        target_artifact="llir",
        htp_kernel_gen=True,
        N_COLUMNS=NUM_COLUMNS,
        N_INNER=NUM_INNER,
        BLOCK_M=NUM_ROWS,
        BLOCK_N=NUM_COLUMNS,
        BLOCK_K=BLOCK_K,
        **OPTIONS,
    )
    llir = compiled.asm["llir"]
    manifest = json.loads(compiled.packed_metadata["hmx_manifest"])
    plans = [(rec.get("plan"), rec.get("reason")) for rec in manifest["matmuls"]]
    assert plans == [("full-hmx", "selected-aligned")], f"HMX plans: {plans}"

    # The engine is reached, and the accumulator stays resident across the
    # K segments: no per-segment f16 image, no widening of it.
    assert "@hmx_mma_f16" in llir, "K-loop contraction did not reach HMX"
    assert "@hmx_unpack_acc_f32_bulk" in llir, (
        "the K-chunk resident f32 accumulator is gone: the read-out is no "
        "longer written straight into the carried accumulator"
    )
    assert "@hmx_unpack_acc_f16_bulk" not in llir, (
        "a per-segment f16 image is back: one DDR round trip per K segment "
        "(the kchunk residency removed exactly this)"
    )

    print(
        f"K-loop matmul {NUM_ROWS}x{NUM_INNER}x{NUM_COLUMNS}, "
        f"tile {NUM_ROWS}x{NUM_COLUMNS}x{BLOCK_K} "
        f"({K_SEGMENTS} K segments), plans={plans}"
    )
    klooped_matmul[(1,)](
        mat_A,
        mat_B,
        mat_C,
        NUM_COLUMNS,
        NUM_INNER,
        NUM_ROWS,
        NUM_COLUMNS,
        BLOCK_K,
        **OPTIONS,
    )

    reference = torch.matmul(mat_A.float(), mat_B.float())
    rel = ((mat_C.float() - reference).norm() / reference.norm()).item()
    print(f"rel={rel:.3e} (gate {REL_TOL:.0e})")
    assert rel < REL_TOL, (
        f"rel {rel:.3e} exceeds the f16-clean gate {REL_TOL:.0e}: a K-loop "
        "numerics regression (the 1D-DMA shear signed 2.0e-02)"
    )
