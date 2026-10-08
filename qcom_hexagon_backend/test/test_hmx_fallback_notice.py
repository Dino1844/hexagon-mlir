#!/usr/bin/env python3
"""Host-only tests for the default-path HMX fallback notice.

Why this file exists
--------------------
Measured 2026-10-01 (docs/evidence/2026-10-01/
why-hmx-refusals-are-invisible-2026-10-01.md): 8 of 41 dot-bearing operators
fall back to HVX with zero diagnostics, and 6 of those 8 carry an exact reason
code in the compiled manifest.  The data was never missing; no default path
read it.  Route B of that report is the reader this file pins: a one-line
notice emitted at the compile-time read point in backend/compiler.py, built by
the pure reporter in backend/utils.py.

The three noise rules the notice must keep, each pinned here:

1. **Silent when nothing fell back.**  A kernel whose every contraction site
   reached the engine prints no line at all -- tested both on the pure
   reporter and through a real host compile of a kernel that reaches HMX.
2. **One line per kernel.**  N refusals produce one line with grouped,
   counted reasons, not N lines, and a pointer to the full manifest rather
   than a dump of it.
3. **No global diagnostic handler, no threshold games.**  The notice reads
   the manifest the compiler already produced; the `MLIR_ENABLE_DIAGNOSTICS`
   route is a measured no-op on this path (the report above, section 2) and
   is not touched here.

The fourth property is honesty about the 2-of-8 case: a site that fell back
*without* a usable reason code must be reported as `reason unavailable`, not
dropped and not crashed on.  The manifest validator refuses such a record, so
the only way one reaches the notice is a stale or hand-edited artifact -- and
the notice still has to speak, because "fell back, reason unknown" is a
different sentence from silence.

This file also pins the repaired opt-in verdict reader in backend/driver.py:
the launcher retains the manifest as the JSON *string* the launcher itself
consumes, and until 2026-10-08 the verdict reader handed that string to the
reporter un-parsed, so every real launch answered "nothing to summarize".
The reader is unreachable without a launch, which is exactly why no test had
ever caught it; the fix is pinned here by driving the real driver class with
a stub, no device.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "hexagon_backend_utils_fallback_notice", _BACKEND / "backend" / "utils.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "hexagon_backend_manifest_fallback_notice_fixtures",
    _HERE.parent / "test_hmx_manifest_metadata.py",
)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
_FIXTURES = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURES)

# Spelled from the validator's own vocabulary (see test_hmx_manifest_summary.py
# for the reason a hand-written code here would prove nothing).
_ON_ENGINE = "selected-aligned"
_REFUSED_A = "min-rows"
_REFUSED_B = "tile-alignment"


def _on_engine_site(record_id, *, function="chain_kernel"):
    """One site that reached the engine (plan full-hmx)."""
    return _FIXTURES._hmx_record(record_id=record_id, function=function)


def _refused_site(record_id, *, reason=_REFUSED_A, function="chain_kernel"):
    """One site that fell back to HVX, with the compiler's own reason."""
    return _FIXTURES._hvx_record(
        record_id=record_id,
        function=function,
        reason=reason,
        shape=(64, 64, 64),
    )


def _validated(records):
    """A manifest the real host gate accepts, like the compiler emits."""
    manifest = _FIXTURES._manifest(records)
    _UTILS.validate_hmx_manifest(manifest)
    return manifest


def _notice_lines(stdout):
    return [
        line
        for line in stdout.splitlines()
        if line.startswith("hmx: ")
    ]


class FallbackNoticeTest(unittest.TestCase):
    """The pure reporter: what the line says, and when there is no line."""

    def test_a_refusal_is_one_line_with_counts_reasons_and_a_pointer(self):
        manifest = _validated(
            [_on_engine_site(0), _on_engine_site(1), _refused_site(2)]
        )
        notice = _UTILS.hmx_fallback_notice(manifest, "chain_kernel")

        self.assertIsInstance(notice, str)
        # One line: N refusals must not become N lines.
        self.assertNotIn("\n", notice)
        self.assertTrue(notice.startswith("hmx: chain_kernel: "), notice)
        self.assertIn("1/3 matmuls fell back to HVX", notice)
        # The reason the user came for, spelled by the validator's vocabulary.
        self.assertIn(f"({_REFUSED_A} x1)", notice)
        # A pointer to the full manifest, not a dump of it.
        self.assertIn('kernel.packed_metadata["hmx_manifest"]', notice)

    def test_reasons_are_grouped_counted_and_deterministically_ordered(self):
        manifest = _validated(
            [
                _refused_site(0, reason=_REFUSED_B),
                _refused_site(1, reason=_REFUSED_B),
                _refused_site(2, reason=_REFUSED_A),
                _on_engine_site(3),
            ]
        )
        notice = _UTILS.hmx_fallback_notice(manifest)
        # Most frequent first; only reasons that occurred are named.
        self.assertIn(f"({_REFUSED_B} x2, {_REFUSED_A} x1)", notice)
        self.assertIn("3/4 matmuls fell back to HVX", notice)
        # No kernel name, no name colon: the prefix stays parseable.
        self.assertTrue(notice.startswith("hmx: 3/4 "), notice)

        # A count tie is broken alphabetically, so the line is deterministic.
        tie = _validated(
            [_refused_site(0, reason=_REFUSED_B), _refused_site(1, reason=_REFUSED_A)]
        )
        tie_notice = _UTILS.hmx_fallback_notice(tie)
        self.assertIn(f"({_REFUSED_A} x1, {_REFUSED_B} x1)", tie_notice)

    def test_no_fallback_means_no_line_at_all(self):
        """The noise rule: nothing fell back, nothing is printed."""
        all_on_engine = _validated(
            [_on_engine_site(0), _on_engine_site(1), _on_engine_site(2)]
        )
        self.assertIsNone(_UTILS.hmx_fallback_notice(all_on_engine, "k"))
        # A kernel with no contraction site never fell back either.
        empty = _validated([])
        self.assertIsNone(_UTILS.hmx_fallback_notice(empty, "k"))
        for not_a_manifest in (None, "junk", 5, {}, {"matmuls": []}):
            with self.subTest(value=not_a_manifest):
                self.assertIsNone(_UTILS.hmx_fallback_notice(not_a_manifest, "k"))

    def test_a_missing_reason_is_reported_not_dropped(self):
        """The 2-of-8 case: fell back, and no reason code to name.

        The validator refuses a reasonless hvx record, so this state can only
        reach the notice as a stale or hand-edited artifact.  The notice must
        neither crash nor go silent on it: "reason unavailable" is a different
        sentence from silence, and it is the honest one.
        """
        manifest = _validated([_refused_site(0), _on_engine_site(1)])
        for mutation in (
            {"reason": None},
            {"reason": "not-a-code"},
            {"reason": "vtcm-budget "},  # trailing space: unrecognized, unquotable
        ):
            with self.subTest(mutation=mutation):
                stale = copy.deepcopy(manifest)
                stale["matmuls"][0].update(mutation)
                notice = _UTILS.hmx_fallback_notice(stale, "k")
                self.assertIsInstance(notice, str)
                self.assertIn("(reason unavailable x1)", notice)
                # An unrecognized code is never quoted back as if the
                # compiler had emitted it.
                self.assertNotIn("vtcm-budget x1", notice)
                self.assertNotIn("not-a-code", notice)

    def test_a_missing_reason_key_is_the_same_sentence(self):
        manifest = _validated([_refused_site(0), _on_engine_site(1)])
        stale = copy.deepcopy(manifest)
        del stale["matmuls"][0]["reason"]
        notice = _UTILS.hmx_fallback_notice(stale, "k")
        self.assertIn("(reason unavailable x1)", notice)

    def test_garbage_never_raises(self):
        """A diagnostic that dies on its own input is worse than silence."""
        for garbage in (
            {"matmuls": "junk"},
            {"matmuls": [None, 5, {"plan": "hvx"}]},
            {"matmuls": [{"plan": "hvx"}]},
            {"matmuls": [{"plan": 7, "reason": 9}]},
        ):
            with self.subTest(garbage=garbage):
                notice = _UTILS.hmx_fallback_notice(garbage, "k")
                self.assertTrue(notice is None or isinstance(notice, str))

    def test_the_reporter_is_pure(self):
        """Reads the manifest, writes nothing: emission is the read point's job."""
        manifest = _validated([_refused_site(0), _on_engine_site(1)])
        before = copy.deepcopy(manifest)
        stdout, stderr = io.StringIO(), io.StringIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                notice = _UTILS.hmx_fallback_notice(manifest, "k")

        self.assertEqual(manifest, before)
        self.assertIsInstance(notice, str)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")


class CompileReadPointTest(unittest.TestCase):
    """The emission: a real host compile through the actual read point.

    Both arms drive ``ttsharedir_to_obj`` (the default pipeline's "o" stage)
    on the constant-weight fixture test_hmx_manifest_metadata.py already
    compiles host-side.  Host-only: the stage never constructs a launcher and
    never touches a device.
    """

    _FIXTURE = (
        _BACKEND / "test" / "Conversion" / "LinalgToLLVM"
        / "hmx-weight-resident-pipeline.mlir"
    )

    def _compile(self, options):
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            ttsharedir_to_obj,
        )

        metadata = {}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            obj = ttsharedir_to_obj(self._FIXTURE.read_text(), options, metadata)
        return metadata, stdout.getvalue(), obj

    def test_a_silent_fallback_is_announced_at_the_read_point(self):
        from triton.backends.qcom_hexagon_backend.hexagon_options import (  # noqa: PLC0415
            HexagonOptions,
        )

        # hexagonmem off is one of the 8 measured silent fallbacks: the crouton
        # arrays cannot get engine-readable memory, every matmul drops to HVX
        # with reason `vtcm-allocator-disabled`, and the compile still
        # succeeds -- which is exactly why nothing else ever said so.
        metadata, stdout, obj = self._compile(
            HexagonOptions(enableConvertToHexagonmem=False)
        )

        self.assertGreater(len(obj), 0)
        lines = _notice_lines(stdout)
        self.assertEqual(len(lines), 1, stdout)
        line = lines[0]
        # The kernel's own name, so a multi-kernel program can tell who fell.
        self.assertTrue(line.startswith("hmx: constant_weight: "), line)
        self.assertIn("1/1 matmuls fell back to HVX", line)
        self.assertIn("(vtcm-allocator-disabled x1)", line)
        # Read point and pure reporter agree on the same manifest: what was
        # printed is what the reporter returns for the manifest the stage
        # retained, not a second opinion computed from different data.
        manifest = json.loads(metadata["hmx_manifest"])
        self.assertEqual(
            _UTILS.hmx_fallback_notice(manifest, metadata.get("name")),
            line,
        )

    def test_a_kernel_that_reaches_hmx_prints_nothing(self):
        from triton.backends.qcom_hexagon_backend.hexagon_options import (  # noqa: PLC0415
            HexagonOptions,
        )

        metadata, stdout, obj = self._compile(HexagonOptions())

        self.assertGreater(len(obj), 0)
        self.assertEqual(_notice_lines(stdout), [])
        # Self-verifying arm: this kernel really did reach the engine, so the
        # silence is the notice's decision, not a compile that went somewhere
        # else.  Before this change the same compile was equally silent with
        # the option flipped, which was the whole defect.
        manifest = json.loads(metadata["hmx_manifest"])
        self.assertEqual(
            {record["plan"] for record in manifest["matmuls"]},
            {"full-hmx"},
        )
        self.assertIsNone(_UTILS.hmx_fallback_notice(manifest, metadata.get("name")))


class VerdictReaderTest(unittest.TestCase):
    """The repaired opt-in verdict reader, driven through the real driver."""

    def _launcher(self):
        from triton.backends.qcom_hexagon_backend.driver import (  # noqa: PLC0415
            HexagonDriver,
        )

        driver = HexagonDriver()
        src = SimpleNamespace(fn=SimpleNamespace(arg_names=[]), signature={})
        return driver.launcher_cls(src, None)

    def test_the_retained_json_string_is_parsed_into_a_real_verdict(self):
        manifest = _validated(
            [_on_engine_site(0), _on_engine_site(1), _refused_site(2)]
        )
        launcher = self._launcher()
        # What __call__ retains: the JSON string the launcher itself consumes.
        launcher.hmx_manifest = json.dumps(manifest)

        verdict = launcher.hmx_manifest_verdict()

        self.assertIsInstance(verdict, str)
        self.assertIn("2 of 3 on HMX", verdict)
        self.assertIn(_REFUSED_A, verdict)
        # The broken behaviour this replaces: a string manifest was answered
        # with "nothing to summarize" even though the kernel had one.
        self.assertNotIn("nothing to summarize", verdict)

    def test_no_manifest_still_means_none(self):
        launcher = self._launcher()
        self.assertIsNone(launcher.hmx_manifest_verdict())


class NoticeSurfaceTest(unittest.TestCase):
    """Structural, not a promise: where the notice is allowed to be mentioned.

    The notice is *supposed* to sit on one default path -- the compile-time
    read point -- and nowhere else.  A second caller under backend/ (a launch
    path, a per-call hook) would change the emission cadence without any test
    in this file noticing, so the mentions are pinned the same way the opt-in
    reporter's are in test_hmx_manifest_summary.py.
    """

    def test_the_notice_has_exactly_one_default_path_caller(self):
        backend = _BACKEND / "backend"
        allowed = {
            ("utils.py", "hmx_fallback_notice"): {
                "def hmx_fallback_notice(manifest, kernel_name=None):",
            },
            ("compiler.py", "hmx_fallback_notice"): {
                # The import entry, and the one call, at the read point.
                "hmx_fallback_notice,",
                "notice = hmx_fallback_notice(",
            },
        }
        for path in sorted(backend.glob("*.py")):
            mentions = {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if "hmx_fallback_notice" in line
            }
            self.assertEqual(
                mentions,
                allowed.get((path.name, "hmx_fallback_notice"), set()),
                f"{path.name}: unexpected hmx_fallback_notice mentions",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
