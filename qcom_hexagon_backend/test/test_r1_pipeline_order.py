#!/usr/bin/env python3
"""R1's position in the pipeline is load-bearing, and nothing was guarding it.

`hvx-maxnum-legalize` (R1) rewrites *vector* `arith.maxnumf` into `vmax` plus a
NaN fixup, because HVX only marks `FMAXIMUMNUM` legal -- a vector `maxnumf`
otherwise gets scalarised lane by lane. Its predicate
(`MaxnumLegalizePass.cpp:43-49`) requires a `VectorType` whose element is f32 or
f16.

That makes the pass's *position* a correctness property, not a style choice:

  * `HexagonVectorization` (`LinalgToLLVMPass.cpp:424`) is what CREATES the
    vector-typed maxnum ops. Measured on the FA kernel: 60 of them, all
    `vector<32xf32>` (= 128 B, exactly one HVX vector).
  * `HvxMaxnumLegalizePass` (`:583`) is what consumes them.

If the two are ever reordered, R1 sees only the 94 *scalar* `arith.maxnumf` that
exist beforehand, matches nothing, and **silently becomes a no-op** -- with the
pass still reporting success and still defaulting to a compile that looks fine.
Nothing in the tree would notice: no test asserts the order, and no comment on
either call site mentions it.

The R1 row in `ROADMAP.md` has been sitting on "needs an A/B" partly because its
preconditions were never checked. They have now been, and the order is one of
them.

This asserts the order from the source rather than from a comment, because a
comment asserting an order is exactly what fails silently when someone reorders
the code.

Run: ../../.venv/bin/python -m pytest -q test/test_r1_pipeline_order.py
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
PIPELINE_CPP = BACKEND / "lib" / "Conversion" / "LinalgToLLVM" / "LinalgToLLVMPass.cpp"
PASS_CPP = BACKEND / "lib" / "Transforms" / "MaxnumLegalizePass.cpp"


def call_site(source, needle):
    """1-indexed line number where `needle` is called, or None."""
    for i, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return i
    return None


def test_the_two_passes_are_both_still_in_the_pipeline():
    src = PIPELINE_CPP.read_text(errors="replace")
    for needle, what in (
        ("createHexagonVectorizationPass", "the pass that creates the vector maxnum"),
        ("createHvxMaxnumLegalizePass", "R1, the pass that consumes it"),
    ):
        assert call_site(src, needle) is not None, (
            f"{needle} is no longer called from LinalgToLLVMPass.cpp -- it is the {what}. "
            "If it moved or was renamed, re-derive whether R1 can still match anything."
        )


def test_r1_runs_after_the_vectorizer():
    """The order that makes R1 work at all.

    Asserted as line numbers because that is what the pipeline actually is. A
    comment saying "must run after HexagonVectorization" would not survive a
    well-meaning reorder; this does.
    """
    src = PIPELINE_CPP.read_text(errors="replace")
    vectorize = call_site(src, "createHexagonVectorizationPass")
    r1 = call_site(src, "createHvxMaxnumLegalizePass")
    assert vectorize is not None and r1 is not None, "a call site went missing"
    assert r1 > vectorize, (
        f"hvx-maxnum-legalize is now added at line {r1}, BEFORE HexagonVectorization at "
        f"line {vectorize}. R1's predicate requires a VectorType, and the vectorizer is "
        "what creates vector-typed arith.maxnumf (measured: 60 of them, vector<32xf32>, on "
        "the FA kernel). In this order R1 matches nothing and silently becomes a no-op."
    )


def test_r1_predicate_still_requires_a_vector():
    """If the predicate ever stops requiring a VectorType, the order stops mattering
    -- and the note above becomes stale. Catch that here rather than let it rot."""
    src = PASS_CPP.read_text(errors="replace")
    predicate = re.search(
        r"static bool isHvxVectorMaxnum\(arith::MaxNumFOp op\)\s*\{(.*?)\n\}", src, re.S
    )
    assert predicate, "isHvxVectorMaxnum is gone or renamed; re-derive R1's trigger"
    body = predicate.group(1)
    assert "VectorType" in body, (
        "isHvxVectorMaxnum no longer tests for a VectorType. If R1 now also matches "
        "scalars, the ordering assertion above is no longer the thing that makes R1 work "
        "-- update both, and re-check whether the 60-vs-94 measurement still applies."
    )
    assert re.search(r"isF32\(\)\s*\|\|\s*\w+\.isF16\(\)", body), (
        "isHvxVectorMaxnum no longer accepts f32/f16. The measured pattern is f32 "
        "(vector<32xf32>), so a narrowing here would make R1 a no-op on FA."
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
