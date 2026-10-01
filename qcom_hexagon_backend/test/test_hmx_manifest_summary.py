#!/usr/bin/env python3
"""Host-only tests for summarize_hmx_manifest(), the opt-in v2 manifest reporter.

Why this file exists
--------------------
The v2 manifest has carried an exact reason code for every refusal since it
was introduced, and 8 of 41 dot-bearing operators still fall back to HVX
(measured 2026-10-01, docs/codegen/
why-hmx-refusals-are-invisible-2026-10-01.md).  The information is in the
artifact; what was missing was a way to ask for it.  `summarize_hmx_manifest`
in `backend/utils.py` is that reader, and this file pins the two properties
that make it worth having:

1. **It separates "reached the engine" from "reached the engine and used it
   well."**  A site can clear every admission gate, take an HMX plan, and still
   run one tile at a time (`HmxManifest.h:112-113` publishes `pipeline.selected`
   for exactly this).  A `plan`-only report calls that a success.  The test
   that matters here uses the *same reason code* on both sides -- `vtcm-budget`
   refusing a site and `vtcm-budget` serializing another -- because that is the
   only input on which a report can get the two backwards without looking wrong.

2. **It is opt-in.**  Nothing in the compile path or the launch path calls the
   reporter or the driver's `hmx_manifest_verdict()`, so asking is the only way
   to get output.  `test_the_reporter_is_not_on_any_default_path` asserts that
   structurally rather than trusting a comment.

It also asserts the reporter is silent -- no `warnings.warn`, nothing on stdout
or stderr -- because a reporter wired into a default path that starts printing
is a behaviour change wearing a diagnostic's clothes, and stage 2 of this work
(which may warn) has to be signed off separately.

Fixtures are borrowed from `test_hmx_manifest_metadata.py` and passed through
`validate_hmx_manifest()` before use, so every manifest here is one the real
host gate accepts; a fixture the validator rejects could not have come off a
compiler and would be testing nothing.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import unittest
import warnings

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "hexagon_backend_utils_summary", _BACKEND / "backend" / "utils.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "hexagon_backend_manifest_summary_fixtures",
    _HERE.parent / "test_hmx_manifest_metadata.py",
)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
_FIXTURES = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURES)


# The pipeline reason the two admitted sites below were serialized by.  Spelled
# from the validator's own vocabulary rather than typed here, for the reason
# test_hmx_manifest_field_agreement.py exists: this file and the C++ producer
# cannot see each other, so a hand-written code would be a plausible line that
# no validator has ever heard of.
_SERIAL_REASON = "vtcm-budget"
_REFUSED_REASON = "min-rows"


def _serial_site(record_id, *, function="chain_kernel", reason=_SERIAL_REASON, requested=2):
    """One site that took an HMX plan and then ran a serial tile loop."""
    record = _FIXTURES._hmx_record(record_id=record_id, function=function)
    record["execution"]["pipeline"] = {
        "requested": requested,
        "selected": "serial",
        "depth": 0,
        "reason": reason,
    }
    record["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(record)
    return record


def _staged_site(record_id, *, function="chain_kernel", depth=2):
    """One site that reached the engine and pipelined it."""
    record = _FIXTURES._hmx_record(record_id=record_id, function=function)
    record["execution"]["pipeline"] = {
        "requested": depth,
        "selected": "staged",
        "depth": depth,
    }
    record["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(record)
    return record


def _validated(records):
    manifest = _FIXTURES._manifest(records)
    # A fixture the host gate rejects could not have come off a compiler.
    _UTILS.validate_hmx_manifest(manifest)
    return manifest


def _two_of_three():
    """Three contraction sites: two admitted to HMX, one refused.

    The two admitted ones are serial for the VTCM budget, which is the state
    the reporter exists to make visible: two of the three sites are on the
    engine, and neither of the two is pipelined.
    """
    return _validated(
        [
            _serial_site(0),
            _serial_site(1),
            _FIXTURES._hvx_record(
                record_id=2,
                function="chain_kernel",
                reason=_REFUSED_REASON,
                shape=(4, 64, 64),
            ),
        ]
    )


def _same_code_both_ways():
    """One serial site and one refusal, both citing the same reason code.

    The adversarial input for requirement 1: `vtcm-budget` here means "refused
    this site" on one line and "would not stage this site" on the other.  A
    report that grouped by reason code would print one line for both.
    """
    return _validated(
        [
            _serial_site(0, function="budget_kernel"),
            _FIXTURES._hvx_record(
                record_id=1,
                function="budget_kernel",
                reason=_SERIAL_REASON,
                shape=(64, 64, 64),
            ),
        ]
    )


class HmxManifestSummary(unittest.TestCase):
    def test_two_of_three_names_the_count_and_both_budget_lines(self):
        report = _UTILS.summarize_hmx_manifest(_two_of_three())

        self.assertIn("2 of 3 on HMX", report)
        # The count and the reason the user came for, in one report.
        self.assertIn("2", report)
        self.assertIn(_SERIAL_REASON, report)
        # Exactly the two admitted sites, so the count is not an accident of a
        # headline that repeats a code the per-site lines never mention.
        citing = [line for line in report.splitlines() if _SERIAL_REASON in line]
        self.assertEqual(len(citing), 2, report)
        # The refusal carries its own, different code.
        self.assertIn(_REFUSED_REASON, report)
        # One line per site, plus the headline.
        self.assertEqual(len(report.splitlines()), 4, report)

    def test_a_refusal_and_a_serial_site_are_not_the_same_sentence(self):
        report = _UTILS.summarize_hmx_manifest(_same_code_both_ways())
        lines = report.splitlines()
        self.assertEqual(len(lines), 3, report)

        serial_line = next(line for line in lines if "SERIAL" in line)
        refused_line = next(line for line in lines if "DROPPED" in line)
        self.assertNotEqual(serial_line, refused_line)
        # Both cite the same code, so the code alone cannot be what separates
        # them; the wording has to.
        self.assertIn(_SERIAL_REASON, serial_line)
        self.assertIn(_SERIAL_REASON, refused_line)
        # The serial site is on the engine, and saying otherwise is the defect
        # this whole exercise exists to remove.
        self.assertIn("ON HMX", serial_line)
        self.assertNotIn("ON HMX", refused_line)

    def test_a_staged_site_reads_as_a_success_with_no_invented_reason(self):
        manifest = _validated([_staged_site(0), _staged_site(1), _staged_site(2)])
        report = _UTILS.summarize_hmx_manifest(manifest)

        self.assertIn("ALL 3 of 3 on HMX", report)
        self.assertIn("staged pipeline depth=2", report)
        # A staged run legitimately publishes no pipeline reason, so the report
        # must not dress the absence up as a defect.
        self.assertNotIn(_SERIAL_REASON, report)
        self.assertNotIn("unrecognized", report)

    def test_an_unrecognized_code_is_marked_rather_than_quoted_back(self):
        manifest = _two_of_three()
        site = manifest["matmuls"][0]
        site["execution"]["pipeline"]["reason"] = "vtcm-budget "  # trailing space
        site["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(site)

        report = _UTILS.summarize_hmx_manifest(manifest)
        self.assertIn("unrecognized reason", report)
        # A code this boundary does not know must never appear bare, because a
        # reader cannot tell it apart from a code the compiler really emitted.
        self.assertNotIn(f"staging declined: {_SERIAL_REASON}\n", report)

    def test_no_manifest_and_no_contraction_site_both_say_so(self):
        for value, expected in (
            (None, "nothing to summarize"),
            ({}, "no contraction site recorded"),
            ({"matmuls": []}, "no contraction site recorded"),
        ):
            with self.subTest(value=value):
                report = _UTILS.summarize_hmx_manifest(value)
                self.assertIn(expected, report)
                self.assertIsInstance(report, str)

    def test_an_unclassifiable_record_is_not_counted_as_a_refusal(self):
        manifest = _two_of_three()
        manifest["matmuls"].append({"id": 3, "function": "x", "plan": "smx"})
        report = _UTILS.summarize_hmx_manifest(manifest)

        # "2 of 4 on HMX" plus an explicit note, never a silent "1 refused".
        self.assertIn("2 of 4 on HMX", report)
        self.assertIn("1 record(s) carry an unrecognized plan", report)
        self.assertIn("'smx' is not an HMX plan", report)

    def test_the_reporter_reads_only(self):
        manifest = _two_of_three()
        before = copy.deepcopy(manifest)
        _UTILS.summarize_hmx_manifest(manifest)
        self.assertEqual(manifest, before)

    def test_the_reporter_is_silent(self):
        """No warning, no stdout, no stderr: stage 2 owns the warning."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                for manifest in (
                    _two_of_three(),
                    _same_code_both_ways(),
                    None,
                    {"matmuls": [{"plan": "full-hmx"}]},
                ):
                    _UTILS.summarize_hmx_manifest(manifest)

        self.assertEqual([str(w.message) for w in caught], [])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_the_reporter_is_not_on_any_default_path(self):
        """Structural, not a promise: name the callers and check the list.

        `summarize_hmx_manifest` is reachable from `driver.py` and from this
        test.  A third mention anywhere under `backend/` would mean something in
        the compile or launch path started reporting, which is the change this
        function must not make.  The same holds for the driver's opt-in reader:
        a second mention under `backend/` would be a caller.
        """
        backend = _BACKEND / "backend"
        allowed = {
            ("utils.py", "summarize_hmx_manifest"): {
                "def summarize_hmx_manifest(manifest):",
            },
            # A pointer, not a call: the two names are checked as exact lines so
            # a comment cannot be confused with a caller.
            ("utils.py", "hmx_manifest_verdict"): {
                "caller is `hmx_manifest_verdict()` in driver.py, which is a read-only",
            },
            ("driver.py", "summarize_hmx_manifest"): {
                "from triton.backends.qcom_hexagon_backend.utils import "
                "summarize_hmx_manifest",
                "return summarize_hmx_manifest(self.hmx_manifest)",
            },
            ("driver.py", "hmx_manifest_verdict"): {
                "def hmx_manifest_verdict(self):",
            },
        }
        for path in sorted(backend.glob("*.py")):
            for name in ("summarize_hmx_manifest", "hmx_manifest_verdict"):
                with self.subTest(file=path.name, name=name):
                    mentions = {
                        line.strip()
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if name in line
                    }
                    self.assertEqual(mentions, allowed.get((path.name, name), set()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
