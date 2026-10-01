#!/usr/bin/env python3
"""The Triton-level half of the K-looped matmul HMX coverage (host only).

Why this file exists
--------------------
`test/Conversion/LinalgToLLVM/hmx-klooped-matmul-pipeline.mlir` pins the
*linalg* pipeline (`scf.for` + `linalg.matmul` with a carried accumulator ->
residual add-back). It cannot see the Triton front end, which is where the
K-loop form is actually written and where the whole-block form dies
(`tl.arange` / block dims must be powers of two, `python/triton/_utils.py:68`).
This file compiles the Triton kernel the users write -- `acc = tl.dot(a, b,
acc)` inside `for _k in range(0, N_INNER, BLOCK_K)` -- and asserts the HMX
leaves appear, including the residual add-back that makes the carried
accumulator correct.

Placement: `hexagon-mlir/qcom_hexagon_backend/test/test_hmx_klooped_matmul_triton.py`.
It is host-only (no device, `target_artifact="llir"`) and is picked up by
`qcom_hexagon_backend/test/run_host_tests.py` automatically (glob `test_*.py`)
in both styles (script and pytest), i.e. gate 2 of
docs/state/STATE-OF-PLAY.md section 4.1.

Run:  source tools/hexmlir/env.sh
      .venv/bin/python hexagon-mlir/qcom_hexagon_backend/test/run_host_tests.py
"""

import json
import unittest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

# The repo matmul test's launch options, verbatim.
BASE = dict(
    enableMultiThreading=True, enableVTCMTiling=True,
    enableConvertToHexagonmem=True, enableHexagonmemCopyToDMA=True,
)

# LLaMA-3 8B FFN width: not a power of two, so the whole-block form cannot even
# be written (front end), while the K-loop form only sees BLOCK_K.
REAL_FFN_K = 14336


@triton.jit
def klooped_matmul(A, B, C, N_INNER: tl.constexpr, N_COLUMNS: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
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
    tl.store(C + offs_m[:, None] * N_COLUMNS + offs_n[None, :],
             acc.to(C.type.element_ty))


@triton.jit
def whole_block_matmul(A, B, C, N_INNER: tl.constexpr, N_COLUMNS: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, N_INNER)
    a = tl.load(A + offs_m[:, None] * N_INNER + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N_COLUMNS + offs_n[None, :])
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    tl.store(C + offs_m[:, None] * N_COLUMNS + offs_n[None, :],
             acc.to(C.type.element_ty))


def _compile(kernel, rows, inner, cols, **extra):
    a = torch.rand(rows, inner, dtype=torch.float16)
    b = torch.rand(inner, cols, dtype=torch.float16)
    c = torch.zeros(rows, cols, dtype=torch.float16)
    return kernel.warmup(a, b, c, grid=(1,), target_artifact="llir",
                         htp_kernel_gen=True, **dict(
                             BASE, N_INNER=inner, N_COLUMNS=cols,
                             BLOCK_M=rows, BLOCK_N=cols, **extra))


class KloopedMatmulReachesHmx(unittest.TestCase):

    def test_real_ffn_shape_reaches_hmx_with_residual_add_back(self):
        compiled = _compile(klooped_matmul, 256, REAL_FFN_K, 256, BLOCK_K=128)
        asm = compiled.asm["llir"]
        # The engine is reached: the four steps of one contraction.
        self.assertIn("@hmx_pack_act_f16_bulk", asm)
        self.assertIn("@hmx_mma_f16", asm)
        self.assertIn("@hmx_unpack_acc_f16_bulk", asm)
        # The carried accumulator is added back after the read-out. Without it
        # the K loop would silently keep only the last K block's product.
        self.assertRegex(asm, r"\bfadd\b")
        # Attribution agrees it is the per-block contraction that reached HMX.
        manifest = json.loads(compiled.packed_metadata["hmx_manifest"])
        plans = [(rec.get("plan"), rec.get("reason"))
                 for rec in manifest["matmuls"]]
        self.assertEqual(plans, [("full-hmx", "selected-aligned")],
                         f"unexpected HMX plan: {plans}")

    def test_whole_block_same_shape_is_refused_by_the_front_end(self):
        # The contrast that motivates the K-loop form at all. If this ever
        # stops raising, the front-end limits changed and this file (and
        # docs/hmx/wide-n-compile-limit.md) must be re-read.
        with self.assertRaises(Exception) as ctx:
            _compile(whole_block_matmul, 256, REAL_FFN_K, 256)
        self.assertIn("power of 2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
