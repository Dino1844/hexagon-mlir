#!/usr/bin/env python3
"""Make the object-gate preconditions fail loudly instead of skipping.

The problem
-----------
Two of the highest-value gates in this suite are *object-code* gates: they read
the rows an HVX/HMX pass actually published out of the pipeline binary's own IR
dump, and out of the emitted object. A file comment in
``test_hmx_vector_readout_object_gate.py`` records that the earlier version of
its main assertion was tautological -- it re-derived the published-row set from
the formula it was supposed to be checking -- and that rewriting it to read the
real rows made it the strongest single assertion in the file.

Both gates then acquired an environment precondition:

* ``test_hmx_vector_readout_object_gate.py`` needs ``TRITON_SHARED_OPT_PATH`` and
  a ``linalg-hexagon-opt`` next to it, else ``_readout_ir`` returns ``None`` and
  the test calls ``skipTest``.
* ``test_hmx_record_v3.py``'s ``GeneratedCodeInvarianceTest._opt`` raises
  ``SkipTest`` unless exactly one ``build/*/third_party/qcom_hexagon_backend/bin/
  linalg-hexagon-opt`` is found.

A skip is silent. ``pytest`` reports green, ``run_host_tests.py`` reports green,
and the strongest assertions in the suite quietly stop running -- which is worse
than them being absent, because the file still reads as coverage.

What this does
--------------
It asserts the preconditions themselves, as ordinary host checks. If the
environment stops satisfying them, *this* file fails and names the cause,
instead of two other files quietly turning into no-ops. It deliberately does not
run the object gates (they need a compile each); it only guarantees that they
are runnable.

Per the same discipline as ``test_object_bytes_are_not_a_valid_invariance_oracle``
in ``test_hmx_record_v3.py``: an environment sentinel is an environment
assertion, so do not count this file as evidence that a production change would
be caught.
"""
from pathlib import Path
import os
import unittest

BACKEND = Path(__file__).resolve().parents[1]
# hexagon-mlir/, i.e. the parent of this backend directory. Written out rather
# than walked upwards so the glob depth is visible and reviewable.
REPO_ROOT = BACKEND.parents[0]


def _opt_binaries() -> list[Path]:
    """Locate the pipeline binaries the way the build lays them out.

    This mirrors ``test_hmx_record_v3.py``'s discovery (``build/*/third_party/
    qcom_hexagon_backend/bin/linalg-hexagon-opt`` under the triton root) but is
    spelled as a path from this file instead of via ``import triton``.

    That is deliberate. Deriving it from ``triton.__file__``, the way the guarded
    gate does, makes this file fail with an opaque ``ImportError: Python version
    mismatch`` whenever it is run by an interpreter that cannot import the
    compiled triton -- and ``run_host_tests.py`` documents exactly that trap. A
    precondition check must be runnable by whatever runs the rest of the host
    suite, so it may not depend on importing the thing it is checking.
    """
    return sorted(
        REPO_ROOT.glob(
            "triton/build/*/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt"
        )
    )


class PipelineBinaryIsDiscoverable(unittest.TestCase):
    """Precondition of GeneratedCodeInvarianceTest in test_hmx_record_v3.py."""

    def test_exactly_one_pipeline_binary_is_reachable(self):
        found = _opt_binaries()
        self.assertEqual(
            len(found),
            1,
            "test_hmx_record_v3.py's GeneratedCodeInvarianceTest skips unless "
            f"exactly one linalg-hexagon-opt exists; found {len(found)}: {found}. "
            "Zero means the gate is not looking where this file looked; more "
            "than one makes the gate skip rather than run.",
        )

    def test_it_is_executable(self):
        found = _opt_binaries()
        self.assertEqual(len(found), 1, "see the test above for the diagnosis")
        self.assertTrue(os.access(found[0], os.X_OK), f"{found[0]} is not executable")


class ReadoutGateEnvironmentIsSet(unittest.TestCase):
    """Precondition of the row-coverage assertion in the readout object gate.

    That assertion is the one the file's own comment calls out as the fix for a
    tautological predecessor, so it is the last thing in the suite that should be
    allowed to go quietly missing.
    """

    def test_triton_shared_opt_path_is_set_and_resolves(self):
        opt = os.environ.get("TRITON_SHARED_OPT_PATH")
        self.assertTrue(
            opt,
            "TRITON_SHARED_OPT_PATH is unset, so _readout_ir() returns None and "
            "test_every_accumulator_row_is_published_exactly_once calls "
            "skipTest. Export it (tools/hexmlir/env.sh does) or that assertion "
            "is not running.",
        )
        self.assertIn("triton_shared", opt, f"unexpected TRITON_SHARED_OPT_PATH={opt}")
        binary = Path(opt.split("triton_shared/")[0]) / (
            "qcom_hexagon_backend/bin/linalg-hexagon-opt"
        )
        self.assertTrue(
            binary.exists(),
            f"TRITON_SHARED_OPT_PATH points at {opt} but {binary} does not exist",
        )

    def test_the_gate_file_still_refuses_to_skip_silently(self):
        # The skip is intentional and documented in the gate itself, so this test
        # does not forbid it; it pins that the skip still names its cause, which
        # is what makes this file's failure actionable.
        gate = (BACKEND / "test/test_hmx_vector_readout_object_gate.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("skipTest", gate)
        self.assertIn(
            "pipeline binary or its IR dump is",
            gate,
            "the skip message lost its cause text, so a skip would no longer say "
            "what was missing",
        )


if __name__ == "__main__":
    unittest.main()