# ===- test_vec_add.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 131072


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)


def test_vec_add():
    x = torch.rand(BLOCK_SIZE)
    y = torch.rand(BLOCK_SIZE)
    output = torch.empty_like(x)

    # "best performance is with just MT on" is no longer something we can claim (or
    # need): for this shape the three staging flags below are currently no-ops.
    # Default vs staging-off: ttsharedir byte-identical (md5 865d25d1...) and the
    # kernel @async_execute_fn.resume is 34 lines with the same external symbols on
    # both -- the direct consequence of VTCMTiling's "skip pure-streaming" fix.
    # They stay ON to keep exercising the staging path, NOT for speed.
    # Re-check (host only, no device):
    #   bash tools/run_tests.sh codegen vec_add
    #   bash tools/run_tests.sh codegen vec_add enableVTCMTiling=False \
    #       enableConvertToHexagonmem=False enableHexagonmemCopyToDMA=False
    # Instrument matters: whole-llir/whole-.o md5 is NOT valid for "no artifact"
    # (AGENTS.md 7.9); use ttsharedir bytes + per-kernel disassembly.
    add_kernel[(1,)](
        x,
        y,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        enableMultiThreading=True,
        enableVTCMTiling=True,
        enableConvertToHexagonmem=True,
        enableHexagonmemCopyToDMA=True,
    )
    reference = x + y
    assert torch.allclose(output, reference)
