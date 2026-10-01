#!/usr/bin/env python3
"""The minimum-rows rule is implemented twice, so it is checked twice.

`HmxManifest.cpp` refuses to *emit* a `full-hmx` or `hmx-tail` record whose
logical M is at or below `kMinimumHmxRows`; below that it emits an `hvx` record
with reason `min-rows` instead.  `backend/utils.py` is the host's last gate
before such a record reaches the launcher, and it used to carry no equivalent
rule -- so a stale or hand-edited cached artifact declaring an HMX plan the
compiler refuses to publish passed every host gate and only surfaced on device.

The two implementations cannot see each other, so this file asserts three
things agree: a literal written here, the C++ constant, and the *observed*
boundary of the Python consumer.  The literal is what makes this an invariant
rather than a consistency check -- editing both implementations in the same
commit still fails until the boundary is moved on purpose, here, where the
change is visible.

Asking the consumer where its boundary is, rather than reading its source, is
deliberate: it measures behaviour, so a rewrite that preserves the behaviour
passes and one that quietly widens it does not.  The accept-side cases are what
give that teeth -- a consumer that rejected everything would satisfy the
rejection cases alone.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest

_TEST_DIR = Path(__file__).resolve().parent
_BACKEND = _TEST_DIR.parent
_MANIFEST_SRC = _BACKEND / "lib" / "Dialect" / "Hmx" / "Transforms" / "HmxManifest.cpp"

# The frozen boundary, written out rather than derived, on purpose. See the
# module docstring.
MINIMUM_ROWS = 4

# The boundary used to have a second, private home in the manifest
# (`kMinimumHmxRows = 4`), which made it the fourth copy of one number. It now
# reads `HmxTarget::minRows`, so the checks below point there instead. Same
# guarantee, one fewer copy: the value is still pinned, and the manifest is now
# additionally required to *use* the shared constant rather than restate it.
_TARGET_SRC = (
    _BACKEND / "include" / "hexagon" / "Dialect" / "Hmx" / "Transforms" / "HmxTarget.h"
)

_TARGET_CONSTANT = re.compile(
    r"static\s+constexpr\s+int64_t\s+minRows\s*=\s*(\d+)\s*;", re.MULTILINE
)

# Any *other* spelling of the boundary inside the manifest is a reintroduced
# duplicate, which is the thing this change set out to remove. Comments are
# stripped first: the file documents this constant's history in prose, and
# naming the old spelling there is correct, not a duplicate.
_STRAY_BOUNDARY = re.compile(r"(?<!HmxTarget::)\b(?:kMinimumHmxRows|minRows)\b")
_LINE_COMMENT = re.compile(r"//[^\n]*")


def _code_only(source):
    return _LINE_COMMENT.sub("", source)


def _utils():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hexagon_backend_utils_minrows", _BACKEND / "backend" / "utils.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(record):
    from test_hmx_manifest_metadata import _manifest

    return _manifest([record])


def _tail(m: int):
    """An `hmx-tail` record at logical M = m. 32-aligned N/K, non-zero tail."""
    from test_hmx_manifest_metadata import _hmx_record

    return _hmx_record(plan="hmx-tail", reason="selected-tail", shape=(m, 47, 70))


class MinimumRowsAgreementTest(unittest.TestCase):
    def test_the_cpp_constant_is_the_frozen_value(self):
        source = _TARGET_SRC.read_text(encoding="utf-8")
        found = _TARGET_CONSTANT.search(source)
        self.assertIsNotNone(
            found,
            f"HmxTarget::minRows not found in {_TARGET_SRC}. If it was renamed, "
            "decide here whether the boundary moved or this regex went stale.",
        )
        self.assertEqual(
            int(found.group(1)),
            MINIMUM_ROWS,
            "the producer's minimum-rows boundary changed. That is a compiler "
            "contract change, not a test update: move MINIMUM_ROWS here "
            "deliberately, or revert the producer change.",
        )

    def test_the_manifest_uses_the_shared_constant(self):
        """The manifest must read the one home, not restate the number.

        This is the check that replaced "the private copy equals 4": a
        duplicate is worse than a stale value, because it can drift silently
        while both copies still look correct.
        """
        source = _MANIFEST_SRC.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r"HmxTarget::minRows",
            "the manifest no longer references HmxTarget::minRows; it must not "
            "carry its own minimum-rows threshold",
        )
        strays = [m.group(0) for m in _STRAY_BOUNDARY.finditer(_code_only(source))]
        self.assertEqual(
            strays, [], f"reintroduced a private minimum-rows name in {_MANIFEST_SRC}"
        )

    def test_the_python_constant_is_the_frozen_value(self):
        self.assertEqual(_utils().HMX_MINIMUM_ROWS, MINIMUM_ROWS)

    def test_an_hmx_plan_at_or_below_the_boundary_is_refused(self):
        """The hole this closes: `hmx-tail` with a tiny M used to be accepted.

        Only `hmx-tail` is exercised at M <= 4. `full-hmx` cannot reach the rule
        at those extents, because `padded.m` must be a multiple of 32 fires
        first -- so it is covered by the accept/reject pair below rather than
        pretending to test a path it cannot reach.
        """
        utils = _utils()
        for m in range(1, MINIMUM_ROWS + 1):
            with self.subTest(m=m):
                with self.assertRaisesRegex(ValueError, "logical M > "):
                    utils.validate_hmx_manifest(_manifest(_tail(m)))

    def test_an_hmx_plan_above_the_boundary_is_accepted(self):
        utils = _utils()
        for m in (MINIMUM_ROWS + 1, 32, 64, 129):
            with self.subTest(m=m):
                utils.validate_hmx_manifest(_manifest(_tail(m)))

    def test_the_hvx_min_rows_fallback_still_runs(self):
        """The control that gives the rule its meaning.

        Below the boundary the compiler does not refuse the matmul, it routes it
        to HVX with reason `min-rows`. If the new rule had swallowed that, small
        matmuls would stop launching -- so this is the case that must NOT be
        rejected, and it is the reason the rule is scoped to HMX plans.
        """
        from test_hmx_manifest_metadata import _hvx_record

        utils = _utils()
        for m in range(1, MINIMUM_ROWS + 1):
            with self.subTest(m=m):
                record = _hvx_record(reason="min-rows", shape=(m, 47, 70))
                utils.validate_hmx_manifest(_manifest(record))


if __name__ == "__main__":
    unittest.main()
