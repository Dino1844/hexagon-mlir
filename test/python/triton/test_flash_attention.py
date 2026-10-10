# ===- test_flash_attention.py ----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import pytest

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

Z, H, N_CTX, D_HEAD = 1, 1, 1024, 64
BLOCK_N = 64
BLOCK_DMODEL = 64
STAGE = 1
NUM_THREADS = 1  # 2026-09-23: was 4. grid=4/BLOCK_M=256 was 2.2x slower than
# grid=1/BLOCK_M=1024 at N_CTX=1024 (17.4 ms vs 7.9 ms, same build 08ead6f2,
# 2 reps each, same options otherwise). Mechanism: every program re-reads the
# whole K/V, and the per-program penalty (~1.8-5.8 ms per extra program at
# BN=64) dominates; grid=1 pays it once. BLOCK_N stays 64. NOTE: BLOCK_M =
# N_CTX / NUM_THREADS, so this does not scale to long sequences -- for large
# N_CTX keep NUM_THREADS high enough that BLOCK_M fits.

assert N_CTX % NUM_THREADS == 0
BLOCK_M = N_CTX // NUM_THREADS

stride_0 = H * N_CTX * D_HEAD
stride_1 = N_CTX * D_HEAD
stride_2 = D_HEAD
stride_3 = 1

SCALE = 0.5


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    qk_scale,  #
    BLOCK_N: tl.constexpr,  #
    N_CTX: tl.constexpr,
):
    # range of values handled by this stage
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        # DO NOT "FIX" THIS BY DELETING `tl.trans` AND NOTHING ELSE.
        #
        # The old comment here said the explicit transpose was a workaround for
        # "block ptr transpose creation failure". That is obsolete: with the
        # pinned Triton (3.7.0 @ a9ced83) a transposed K block pointer
        # (shape=(BLOCK_DMODEL, N_CTX), strides=(stride_3, stride_2),
        # block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0,1)) lowers cleanly --
        # no `tt.trans`, no `linalg.transpose`, just a column-major
        # `memref.reinterpret_cast` with `strides: [1, 64]`. Measured, see
        # docs/hmx/fa-transpose-copy-design-2026-09-30.md.
        #
        # The blocker is ours, not Triton's: `hmx.pack_weight`'s source contract
        # is row-major only, and the column-major K view is rejected by
        # `HmxOps.cpp` verifyPackSource ("src row stride 1 is smaller than its
        # 64 columns"). Worse, a column-major RHS currently sails through
        # `MatmulToHmxPass` admission (`isRowMajorMatmul` compares indexing
        # maps, not strides) and only dies in that leaf verifier.
        #
        # ⚠️ THE TRAP, measured 2026-10-01: deleting ONLY the `tl.trans` line,
        # leaving the block pointer alone, compiles with ZERO diagnostics and
        # silently computes qkᵀ instead of qk. Its ttsharedir differs from the
        # correct transposed-block-pointer form in exactly two stride values
        # (`strides: [64, 1]` where it must be `[1, 64]`) and nothing else. It
        # type-checks only because BLOCK_N == BLOCK_DMODEL == 64.
        # The explicit transpose below is therefore load-bearing until the
        # pack leaf can read a K-contiguous source; see the same doc.
        qk = tl.dot(
            q, tl.trans(k)
        )
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        # p = p.to(tl.float32)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def attention_fwd_kernel(
    pid1,  ## Set to 0
    Q,
    K,
    V,
    sm_scale: tl.constexpr,  #
    Out,  #
    H,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    stride_0: tl.constexpr,  #
    stride_1: tl.constexpr,  #
    stride_2: tl.constexpr,  #
    stride_3: tl.constexpr,  #
):

    pid0 = tl.program_id(0)
    start_m = pid0
    off_hz = pid1
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int32) * stride_0 + off_h.to(tl.int32) * stride_1
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 99999999999999.0  # float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        qk_scale,  #
        BLOCK_N,  #
        N_CTX,  #
    )
    # epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def test_flash_attention():
    query = torch.rand(Z, H, N_CTX, D_HEAD)
    key = torch.rand(Z, H, N_CTX, D_HEAD)
    value = torch.rand(Z, H, N_CTX, D_HEAD)

    matrix_stride_0 = H * N_CTX * D_HEAD
    matrix_stride_1 = N_CTX * D_HEAD
    matrix_stride_2 = D_HEAD
    matrix_stride_3 = 1

    output = torch.zeros_like(query)

    grid = (NUM_THREADS,)
    attention_fwd_kernel[grid](
        pid1=0,
        Q=query,
        K=key,
        V=value,
        sm_scale=SCALE,
        Out=output,
        H=H,  #
        N_CTX=N_CTX,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_DMODEL=BLOCK_DMODEL,  #
        BLOCK_N=BLOCK_N,  #
        STAGE=STAGE,
        stride_0=stride_0,
        stride_1=stride_1,
        stride_2=stride_2,
        stride_3=stride_3,
        enableVectorization=True,
        enableSplitReduceGeneric=True,
        enableHVXInlining=True,
        enableSCFLoopUnroll=True,
        enableMultiThreading=True,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=True,  # was False: this is the only gate on the HMX path (vtcm-allocator), and leaving it off made this test measure a kernel with zero HMX leaves
        enableHexagonmemCopyToDMA=False,
    )

    reference = F.scaled_dot_product_attention(query, key, value, scale=SCALE)
    # The HMX engine's operands are fp16: an fp32 activation is quantised by the
    # pack leaf (accepted as the f32 activation ABI, 2026-09-19), so this is a
    # quantisation budget rather than a bit-exactness check. Measured on device
    # over repeated runs: relative Frobenius error 7.2e-05 .. 7.4e-05. The bound
    # is ~70x that noise and ~10x fp16 eps (4.9e-04), while a real correctness
    # bug (wrong crouton offset, missing pack, wrong residual) shows up at
    # rel ~ 1e-1..1 and still fails loudly.
    rel = float((output - reference).norm() / reference.norm())
    assert rel < 5e-3, f"relative error {rel:.3e} exceeds the fp16 quantisation budget"
