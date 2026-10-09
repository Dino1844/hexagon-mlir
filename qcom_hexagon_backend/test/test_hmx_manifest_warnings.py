#!/usr/bin/env python3
"""Host-only tests for the launch-time HMX manifest warnings.

Why this file exists
--------------------
Three facts about a compiled kernel are recorded in ``hmx_manifest`` and shown
by no default path at the moment a user is watching a launch:

1. **REFUSAL** -- which matmul sites fell back to HVX, with the compiler's own
   reason. Measured 2026-10-01: 8 of 41 dot-bearing operators fall back and
   the reasons sit unread in the manifest (docs/evidence/2026-10-01/
   why-hmx-refusals-are-invisible-2026-10-01.md). The compile-time print
   (backend/compiler.py) is invisible on a cache hit and buried in compile
   output otherwise.
2. **PARTIAL** -- a function where some sites reached HMX and some did not
   reads as "I got HMX" in every plan-only check (the module-level PARTIAL
   verdict of tools/hexmlir/manifest_verdict.py).
3. **CONTRADICTION** -- ``plan=full-hmx`` with
   ``execution.bridge_counts`` pack_act/pack_weight/unpack all 0. The counts
   are recounted from the IR (``refreshHmxManifestBridgeCounts``), so zero
   bridge sites means the kernel contains no bridge op: hmx-partition emitted
   an empty HMX span while the manifest kept the admitted plan -- the false
   green measured on n-loop-only matmul structures.

The judgment is the pure reporter ``hmx_manifest_warnings`` in
backend/utils.py (and the grid sibling ``hmx_grid_notice``); this file drives
it with synthetic manifests the real semantic validator accepts, so no
compile and no device is involved.

Noise rules, each pinned below:

* a manifest with every site on the engine and self-consistent counts
  contributes **no** message at all;
* messages are deterministic -- nothing per-launch (reps, timestamps,
  addresses) -- which is what makes the interpreter's default warning filter
  deduplicate repeated launches of the same kernel;
* the reporter never raises on unreadable input: a diagnostic that dies on
  the manifest it exists to read is worse than silence.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest
import warnings

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "hexagon_backend_utils_launch_warnings", _BACKEND / "backend" / "utils.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)

_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "hexagon_backend_manifest_launch_warnings_fixtures",
    _HERE.parent / "test_hmx_manifest_metadata.py",
)
assert _FIXTURE_SPEC is not None and _FIXTURE_SPEC.loader is not None
_FIXTURES = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURES)

# Spelled from the validator's vocabulary via the fixtures, never hand-written
# (a hand-written reason would prove the reporter echoes whatever it is given).
_KERNEL = "chain_kernel"

_ZERO_COUNTS = {
    "pack_act_sites": 0,
    "pack_weight_sites": 0,
    "unpack_sites": 0,
    "count_semantics": "ir_sites",
}


def _on_engine_site(record_id, *, counts=None, function=_KERNEL):
    return _FIXTURES._hmx_record(
        record_id=record_id, function=function, counts=counts
    )


def _refused_site(record_id, *, reason="min-rows", shape=(64, 64, 64),
                  function=_KERNEL):
    return _FIXTURES._hvx_record(
        record_id=record_id, function=function, reason=reason, shape=shape
    )


def _validated(records):
    """A manifest the real host gate accepts, like the compiler emits."""
    manifest = _FIXTURES._manifest(records)
    _UTILS.validate_hmx_manifest(manifest)
    return manifest


class RefusalWarningTest(unittest.TestCase):
    """plan=hvx anywhere -> one message naming kernel, sites and reasons."""

    def test_the_warning_names_the_kernel_every_site_and_every_reason(self):
        manifest = _validated(
            [
                _refused_site(0, reason="min-rows", shape=(64, 64, 64)),
                _refused_site(1, reason="tile-alignment", shape=(32, 128, 64)),
            ]
        )
        messages = _UTILS.hmx_manifest_warnings(manifest, _KERNEL)

        self.assertEqual(len(messages), 1, messages)
        message = messages[0]
        self.assertTrue(message.startswith(f"hmx: {_KERNEL}: "), message)
        # The count, and the plain statement the task exists for.
        self.assertIn("HMX is not used for 2/2 matmul(s)", message)
        self.assertIn("HVX path", message)
        self.assertIn("not on the HMX engine", message)
        # Per site: id (the matmul ordinal), shape, and the compiler's reason.
        self.assertIn("#0", message)
        self.assertIn("#1", message)
        self.assertIn("64x64x64", message)
        self.assertIn("32x128x64", message)
        self.assertIn("reason=min-rows", message)
        self.assertIn("reason=tile-alignment", message)
        # A pointer to the full records, not a dump of them.
        self.assertIn('kernel.packed_metadata["hmx_manifest"]', message)

    def test_a_refusal_with_an_unusable_reason_is_still_reported(self):
        """Stale/hand-edited artifact: say `reason unavailable`, do not die.

        The semantic validator refuses a reasonless hvx record, so only a
        stale artifact can reach the reporter with one -- and silence would
        be the wrong answer to "why is this on HVX?".
        """
        manifest = _validated([_refused_site(0), _on_engine_site(1)])
        for mutation in ({"reason": None}, {"reason": "not-a-code"}):
            with self.subTest(mutation=mutation):
                stale = copy.deepcopy(manifest)
                stale["matmuls"][0].update(mutation)
                messages = _UTILS.hmx_manifest_warnings(stale, _KERNEL)
                # Mixed manifest: the refusal detail and the PARTIAL count
                # are both due; the reason lives in the refusal message.
                refusal = [m for m in messages if "HMX is not used" in m]
                self.assertEqual(len(refusal), 1, messages)
                self.assertTrue(
                    "reason unavailable" in refusal[0]
                    or "unrecognized reason" in refusal[0],
                    refusal[0],
                )
                # An unrecognized code is labelled, never echoed as if the
                # compiler had emitted it verbatim without qualification.
                if "not-a-code" in mutation.values():
                    self.assertIn("unrecognized reason", refusal[0])


class PartialWarningTest(unittest.TestCase):
    """Mixed full-hmx + hvx -> the k/n count, alongside the refusal detail."""

    def test_the_warning_states_how_many_matmuls_are_on_hmx(self):
        manifest = _validated(
            [_on_engine_site(0), _on_engine_site(1), _refused_site(2)]
        )
        messages = _UTILS.hmx_manifest_warnings(manifest, _KERNEL)

        partial = [m for m in messages if "PARTIAL" in m]
        self.assertEqual(len(partial), 1, messages)
        self.assertIn("2/3 matmul(s) on HMX", partial[0])
        self.assertIn("1 on HVX", partial[0])
        # The refusal message accompanies it, with the same denominators.
        refusal = [m for m in messages if "HMX is not used" in m]
        self.assertEqual(len(refusal), 1, messages)
        self.assertIn("1/3 matmul(s)", refusal[0])

    def test_all_on_engine_produces_no_partial_message(self):
        manifest = _validated([_on_engine_site(0), _on_engine_site(1)])
        self.assertEqual(_UTILS.hmx_manifest_warnings(manifest, _KERNEL), [])


class ContradictionWarningTest(unittest.TestCase):
    """full-hmx + bridge_counts all 0 -> the false green is named."""

    def test_zero_bridge_sites_on_a_full_hmx_record_are_flagged(self):
        manifest = _validated([_on_engine_site(0, counts=_ZERO_COUNTS)])
        # The premise: the semantic validator ACCEPTS this record -- wire
        # shape, arithmetic and fingerprint all hold. The contradiction is
        # semantic (plan vs. IR recount) and only this layer can see it.
        messages = _UTILS.hmx_manifest_warnings(manifest, _KERNEL)

        self.assertEqual(len(messages), 1, messages)
        message = messages[0]
        self.assertTrue(message.startswith(f"hmx: {_KERNEL}: "), message)
        self.assertIn("claims plan=full-hmx", message)
        self.assertIn("pack_act_sites=0", message)
        self.assertIn("pack_weight_sites=0", message)
        self.assertIn("unpack_sites=0", message)
        self.assertIn("hmx-partition produced no HMX bridge sites", message)
        self.assertIn("false green", message)
        self.assertIn("did not execute on HMX", message)
        self.assertIn("#0", message)

    def test_nonzero_or_partial_counts_are_not_flagged(self):
        healthy = _validated(
            [_on_engine_site(0), _on_engine_site(1, counts=None)]
        )
        self.assertEqual(_UTILS.hmx_manifest_warnings(healthy, _KERNEL), [])

        # weight-resident kernels legitimately zero pack_weight alone; a
        # single zeroed field with the others alive is execution, not absence.
        resident = _validated(
            [
                _on_engine_site(
                    0,
                    counts={
                        "pack_act_sites": 1,
                        "pack_weight_sites": 0,
                        "unpack_sites": 1,
                        "count_semantics": "ir_sites",
                    },
                )
            ]
        )
        self.assertEqual(_UTILS.hmx_manifest_warnings(resident, _KERNEL), [])

    def test_missing_bridge_keys_are_unreadable_not_contradictory(self):
        """A record that reads nothing must not be quoted as self-contradictory."""
        manifest = _validated([_on_engine_site(0)])
        for mutation in (
            lambda record: record["execution"].pop("bridge_counts"),
            lambda record: record["execution"]["bridge_counts"].pop(
                "pack_act_sites"
            ),
            lambda record: record.pop("execution"),
        ):
            with self.subTest(mutation=mutation):
                unreadable = copy.deepcopy(manifest)
                mutation(unreadable["matmuls"][0])
                self.assertEqual(
                    _UTILS.hmx_manifest_warnings(unreadable, _KERNEL), []
                )

    def test_only_full_hmx_claims_are_checked(self):
        """Scope pin: an hmx-tail record with zero counts is not yet a claim.

        No tail case with zero bridge counts has been measured, so the
        reporter must not invent one; widen this together with the
        measurement, not before it.
        """
        tail = _FIXTURES._hmx_record(
            record_id=0, plan="hmx-tail", counts=_ZERO_COUNTS, shape=(48, 64, 64)
        )
        manifest = _FIXTURES._manifest([tail])
        _UTILS.validate_hmx_manifest(manifest)
        self.assertEqual(_UTILS.hmx_manifest_warnings(manifest, _KERNEL), [])


class QuietAndToleranceTest(unittest.TestCase):
    """The noise rule, and never-raise on unreadable input."""

    def test_a_fully_green_manifest_is_silent(self):
        manifest = _validated(
            [_on_engine_site(0), _on_engine_site(1), _on_engine_site(2)]
        )
        self.assertEqual(_UTILS.hmx_manifest_warnings(manifest, _KERNEL), [])

    def test_no_matmuls_and_unreadable_inputs_yield_no_messages(self):
        for value in (
            None,
            "",
            "junk",
            5,
            [],
            {},
            {"matmuls": []},
            {"matmuls": "junk"},
            {"matmuls": [{"plan": 7, "reason": 9}]},
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    _UTILS.hmx_manifest_warnings(value, _KERNEL), []
                )

    def test_a_partially_readable_record_is_still_reported_not_crashed_on(self):
        """Tolerance: garbage around a readable hvx record keeps the report.

        `None` and `5` contribute nothing, but a record that plainly says
        plan=hvx is exactly the fact the warning exists for -- dropping it
        because its neighbours are unreadable would be silent by accident.
        """
        messages = _UTILS.hmx_manifest_warnings(
            {"matmuls": [None, 5, {"plan": "hvx"}]}, _KERNEL
        )
        self.assertEqual(len(messages), 1, messages)
        self.assertIn("HMX is not used for 1/3 matmul(s)", messages[0])
        # Readable parts are labelled, unreadable ones are `?`, never invented.
        self.assertIn("#? ? ?", messages[0])
        self.assertIn("reason=reason unavailable", messages[0])

    def test_the_json_string_form_matches_the_parsed_form(self):
        """The launch path holds the string; tests hold the dict; same answer."""
        manifest = _validated([_refused_site(0), _on_engine_site(1)])
        self.assertEqual(
            _UTILS.hmx_manifest_warnings(json.dumps(manifest), _KERNEL),
            _UTILS.hmx_manifest_warnings(manifest, _KERNEL),
        )

    def test_the_reporter_is_pure(self):
        """Reads the manifest, writes nothing: emission is the caller's job."""
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        manifest = _validated([_refused_site(0)])
        before = copy.deepcopy(manifest)
        stdout, stderr = io.StringIO(), io.StringIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                _UTILS.hmx_manifest_warnings(manifest, _KERNEL)

        self.assertEqual(manifest, before)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")


class DeduplicationTest(unittest.TestCase):
    """What makes warnings.warn the right emission (and print the wrong one)."""

    @staticmethod
    def _emit_like_the_driver(messages):
        """The driver's forwarding loop, reproduced at one fixed call site."""
        for message in messages:
            warnings.warn(message, stacklevel=2)

    def test_repeated_launches_warn_once_per_message(self):
        manifest = _validated([_refused_site(0), _on_engine_site(1)])
        messages = _UTILS.hmx_manifest_warnings(manifest, _KERNEL)
        self.assertGreater(len(messages), 0)

        with warnings.catch_warnings(record=True) as caught:
            # NOT "always": the default filter is exactly the mechanism under
            # test -- dedup by message text at the reported location.
            warnings.simplefilter("default")
            for _ in range(3):  # three launches of the same kernel
                self._emit_like_the_driver(messages)

        self.assertEqual(len(caught), len(messages), [str(w) for w in caught])
        self.assertEqual(
            [str(w.message) for w in caught],
            messages,
        )

    def test_messages_are_deterministic_and_free_of_launch_varying_content(self):
        """The dedup key is the message: it must not vary per launch."""
        manifest = _validated([_refused_site(0), _on_engine_site(1)])
        first = _UTILS.hmx_manifest_warnings(manifest, _KERNEL)
        second = _UTILS.hmx_manifest_warnings(manifest, _KERNEL)
        self.assertEqual(first, second)
        for message in first:
            for volatile in ("rep=", "stamp", "@0x", "0x"):
                self.assertNotIn(volatile, message)

    def test_two_different_kernels_get_two_different_messages(self):
        """Dedup must be per kernel: names differ, so both must survive."""
        refused_a = _refused_site(0, function="kernel_a")
        refused_b = _refused_site(0, function="kernel_b")
        messages = [
            *_UTILS.hmx_manifest_warnings(_validated([refused_a]), "kernel_a"),
            *_UTILS.hmx_manifest_warnings(_validated([refused_b]), "kernel_b"),
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(2):
                self._emit_like_the_driver(messages)
        self.assertEqual(len(caught), 2, [str(w) for w in caught])


class GridNoticeTest(unittest.TestCase):
    """grid>1 + HMX on the manifest -> the fact-only parallelism notice."""

    def _hmx_manifest(self):
        return _validated([_on_engine_site(0)])

    def test_grid_one_and_hvx_only_manifests_stay_silent(self):
        manifest = self._hmx_manifest()
        self.assertIsNone(
            _UTILS.hmx_grid_notice(manifest, (1, 1, 1), _KERNEL)
        )
        hvx_only = _validated([_refused_site(0)])
        self.assertIsNone(
            _UTILS.hmx_grid_notice(hvx_only, (1, 64, 1), _KERNEL)
        )
        self.assertIsNone(_UTILS.hmx_grid_notice(None, (1, 64, 1), _KERNEL))

    def test_grid_above_one_with_hmx_states_the_mechanism_and_the_advice(self):
        notice = _UTILS.hmx_grid_notice(
            self._hmx_manifest(), (1, 64, 1), _KERNEL, threaded_dispatch=True
        )

        self.assertIsInstance(notice, str)
        self.assertTrue(notice.startswith(f"hmx: {_KERNEL}: "), notice)
        self.assertIn("(1, 64, 1) = 64 program instance(s)", notice)
        # The load-bearing fact: more programs do not parallelize HMX.
        self.assertIn("single resident thread", notice)
        self.assertIn("holds the HMX lock for life", notice)
        self.assertIn("HmxRoleExecutor", notice)
        self.assertIn("do not run in parallel across programs", notice)
        # The requested structural recommendation, not a performance claim.
        self.assertIn("prefer grid=1", notice)
        self.assertIn("tiling loop inside the kernel", notice)
        # Facts only: no measured ratio exists for this combination, so the
        # message may not imply one.
        for claim in ("faster", "speedup", "%", "µs", "us per"):
            self.assertNotIn(claim, notice)

    def test_the_dispatch_mode_is_stated_not_guessed(self):
        manifest = self._hmx_manifest()
        threaded = _UTILS.hmx_grid_notice(
            manifest, (1, 4, 1), _KERNEL, threaded_dispatch=True
        )
        serial = _UTILS.hmx_grid_notice(
            manifest, (1, 4, 1), _KERNEL, threaded_dispatch=False
        )
        unknown = _UTILS.hmx_grid_notice(manifest, (1, 4, 1), _KERNEL)

        self.assertIn("fresh qurt threads", threaded)
        self.assertNotIn("tm.exec_serial", threaded)
        self.assertIn("tm.exec_serial", serial)
        self.assertNotIn("fresh qurt threads", serial)
        # Unknown mode: both facts, honestly labelled.
        self.assertIn("fresh qurt threads", unknown)
        self.assertIn("serial loop", unknown)

    def test_a_malformed_grid_never_raises(self):
        manifest = self._hmx_manifest()
        for grid in (None, (1,), (0, 1, 1), (1, -2, 1), "junk", (1.0, 1, 1)):
            with self.subTest(grid=grid):
                self.assertIsNone(
                    _UTILS.hmx_grid_notice(manifest, grid, _KERNEL)
                )


class WarningSurfaceTest(unittest.TestCase):
    """Structural: where each reporter is allowed to be mentioned.

    The reporters are supposed to sit at exactly one emission point each --
    driver.py for the manifest warnings, triton_hexagon_launcher.py for the
    grid notice -- so a second caller under backend/ would change the cadence
    without any test in this file noticing. Same shape as the mention pins in
    test_hmx_fallback_notice.py and test_hmx_manifest_summary.py.
    """

    _ALLOWED = {
        ("utils.py", "hmx_manifest_warnings"): {
            "def hmx_manifest_warnings(manifest, kernel_name=None):",
        },
        ("driver.py", "hmx_manifest_warnings"): {
            "from triton.backends.qcom_hexagon_backend.utils import "
            "hmx_manifest_warnings",
            "for _warning in hmx_manifest_warnings(",
        },
        ("utils.py", "hmx_grid_notice"): {
            "def hmx_grid_notice(manifest, launch_grid, kernel_name=None, *, "
            "threaded_dispatch=None):",
        },
        ("triton_hexagon_launcher.py", "hmx_grid_notice"): {
            "hmx_grid_notice,",
            "grid_notice = hmx_grid_notice(",
        },
    }

    def test_each_reporter_has_exactly_one_emission_path(self):
        backend = _BACKEND / "backend"
        for name in ("hmx_manifest_warnings", "hmx_grid_notice"):
            for path in sorted(backend.glob("*.py")):
                with self.subTest(file=path.name, name=name):
                    mentions = {
                        line.strip()
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if name in line
                    }
                    self.assertEqual(
                        mentions,
                        self._ALLOWED.get((path.name, name), set()),
                        f"{path.name}: unexpected {name} mentions",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
