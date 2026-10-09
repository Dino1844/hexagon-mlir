#!/usr/bin/env python3
"""The `weight_prepack` wire format is written in three places, so it is pinned.

`hmx.weight_prepack` is a JSON contract with one producer and two consumers that
cannot see each other:

* the producer, `WeightResidentPass.cpp`, which builds each entry by string
  concatenation (`{"func",...,"slot",...,"shape",...,"crouton",...,"dtype",
  ...,"location"}`) plus the top-level `{"layout",...,"weights"}` object;
* a second C++ reader, `HmxManifest.cpp`, which re-spells `"func"`, `"slot"`
  and `"shape"` as inline literals to recover the logical N per resident slot;
* the consumer, `backend/utils.py::validate_weight_prepack`, which requires an
  exact field set and constrains `dtype` and the two list ranks.

Nothing connected them. If the producer learned a key the consumer did not, a
legitimate manifest compiled fine and then failed at
`triton_hexagon_launcher.py::WeightPrepack.from_metadata` on the *first launch*,
as `unknown field(s)`. If the consumer learned a key the producer did not, the
host would accept a shape the compiler never publishes. Both are silent until
someone runs the other side.

The existing round-trip coverage cannot catch this, and the reason is worth
stating: the only test that runs the real C++ producer and validates the result
with the Python consumer uses a fixture whose prepack is **empty**
(`{"layout": null, "weights": []}`), so it exercises the empty arm and no entry
schema at all. A key added to the producer would reach no assertion anywhere.

So this file asserts four things: a literal written here, the producer's
concatenated key list, the second C++ reader's keys, and the consumer's observed
accept-set. The literal is what makes it an invariant rather than a consistency
check -- editing producer and consumer in one commit still fails until the
boundary is moved on purpose, here, where the change is visible.

Asking the consumer where its boundary *is*, rather than reading its source, is
deliberate: it measures behaviour, so a rewrite that preserves the behaviour
passes and one that quietly widens it does not. The rejection cases give that
teeth, so a consumer that accepted everything would not pass.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import unittest

_TEST_DIR = Path(__file__).resolve().parent
_BACKEND = _TEST_DIR.parent
_PRODUCER_SRC = (
    _BACKEND / "lib" / "Dialect" / "Hmx" / "Transforms" / "WeightResidentPass.cpp"
)
_READER_SRC = _BACKEND / "lib" / "Dialect" / "Hmx" / "Transforms" / "HmxManifest.cpp"
_BINDING_SRC = _BACKEND / "python" / "triton_qcom_hexagon_backend_api.cc"

# ---------------------------------------------------------------------------
# The frozen boundary, written out rather than derived, on purpose.
# ---------------------------------------------------------------------------
ENTRY_FIELDS = ("func", "slot", "shape", "crouton", "dtype", "location")
TOP_LEVEL_FIELDS = ("layout", "weights")
#: Rank of `shape` (logical [N, K]) and of `crouton` ([Nt, Kt, 16, 32, 2]).
LOGICAL_RANK = 2
CROUTON_RANK = 5
#: The source dtypes the producer may emit and the consumer accepts. `dtype`
#: names the source argument's element type; the packed image is always the
#: fp16 crouton, and an f32 source is quantised by the host.
DTYPES = ("f16", "f32")
DTYPE = DTYPES[0]
#: The placements the producer may name: the VTCM pool, or the permanent DDR
#: mirror the compiler falls back to when the pool cannot hold the weight. The
#: consumer validates it (it is a fact the host contract carries) but does not
#: act on it: the image bytes are the same either way.
LOCATIONS = ("vtcm", "ddr")
LOCATION = LOCATIONS[0]

# A layout the consumer accepts, mirroring the producer's `prepackLayoutJson`
# coefficient map: physical (d0=n_tile, d1=k_tile, d2=j, d3=c, d4=h) maps to
# logical (tile*d1 + half*d2 + d4, tile*d0 + d3).
_LAYOUT = {
    "ndims": 5,
    "results": [[[1, 32], [2, 2], [4, 1]], [[0, 32], [3, 1]]],
}


def _utils():
    spec = importlib.util.spec_from_file_location(
        "hexagon_backend_utils_prepack", _BACKEND / "backend" / "utils.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _conforming_entry(**overrides):
    entry = {
        "func": "matmul_kernel",
        "slot": 0,
        "shape": [64, 32],
        "crouton": [2, 1, 16, 32, 2],
        "dtype": DTYPE,
        "location": LOCATION,
    }
    entry.update(overrides)
    return entry


def _prepack(*entries, layout=_LAYOUT):
    return {"layout": layout, "weights": list(entries)}


def _producer_entry_keys() -> list[str]:
    """Keys in the producer's concatenated entry literal, in source order.

    Matched on the `\\"key\\":` escapes the producer actually emits, so a rename
    shows up as a changed list rather than as a silently missing test.
    """
    source = _PRODUCER_SRC.read_text(encoding="utf-8")
    start = source.find('"{\\"func\\":\\""')
    assert start != -1, f"the entry literal was not found in {_PRODUCER_SRC}"
    window = source[start : start + 600]
    return re.findall(r'\\"([A-Za-z_][A-Za-z0-9_]*)\\":', window)


def _reader_entry_keys() -> set[str]:
    source = _READER_SRC.read_text(encoding="utf-8")
    anchor = source.find("hmx.weight_prepack entry must be an object")
    assert anchor != -1, f"the prepack reader was not found in {_READER_SRC}"
    window = source[max(0, anchor - 400) : anchor + 1600]
    return set(
        re.findall(
            r'(?:getString|getArray|getInteger)\("([A-Za-z_][A-Za-z0-9_]*)"\)', window
        )
    )


class PrepackFieldAgreementTest(unittest.TestCase):
    """The three declarations must agree, checked against the source of each."""

    def test_the_producer_emits_exactly_the_frozen_entry_fields(self):
        self.assertEqual(_producer_entry_keys(), list(ENTRY_FIELDS))

    def test_the_binding_emits_exactly_the_frozen_prepack_fields(self):
        # The `{"layout",...,"weights"}` wrapper is not built by the pass: the
        # pass appends entry JSON to a bare array, and the binding wraps it.
        source = _BINDING_SRC.read_text(encoding="utf-8")
        anchor = source.find('"weight_prepack\\":{\\"layout\\":')
        self.assertNotEqual(
            anchor,
            -1,
            f"the prepack wrapper literal was not found in {_BINDING_SRC}",
        )
        # Stop at the append that closes the prepack object (`json += "},...`);
        # past it the next envelope key is still in range and would read as ours.
        window = source[anchor : anchor + 400]
        end = window.find('json += "}')
        if end != -1:
            window = window[:end]
        self.assertEqual(
            re.findall(r'\\"([A-Za-z_][A-Za-z0-9_]*)\\":', window),
            list(TOP_LEVEL_FIELDS),
        )

    def test_the_second_cpp_reader_reads_only_produced_keys(self):
        # It reads only what it needs (function, slot, shape). A key it reads
        # that the producer never emits would be dead; a key it *stops* reading
        # is not a contract change, so this asserts subset, not equality.
        unknown = _reader_entry_keys() - set(ENTRY_FIELDS)
        self.assertEqual(
            unknown,
            set(),
            "HmxManifest.cpp reads prepack keys the producer never emits: "
            f"{sorted(unknown)}",
        )

    def test_the_consumer_required_tuple_is_the_frozen_one(self):
        # The source pin, so the literal cannot be edited on the consumer side
        # alone. Scoped to `validate_weight_prepack` because utils.py declares
        # several unrelated `required = (...)` tuples and a bare regex finds the
        # wrong one. The behavioural half lives in PrepackBehaviourTest below.
        source = (_BACKEND / "backend" / "utils.py").read_text(encoding="utf-8")
        start = source.index("def validate_weight_prepack(")
        body = source[start : start + 3000]
        found = re.search(r"required = \(([^)]*)\)", body)
        self.assertIsNotNone(found, "the consumer's required tuple is gone")
        self.assertEqual(
            tuple(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', found.group(1))),
            ENTRY_FIELDS,
        )


class PrepackBehaviourTest(unittest.TestCase):
    """Where the accepted set is measured, so the pins above mean something."""

    def _accepts(self, *entries, **kwargs):
        _utils().validate_weight_prepack(_prepack(*entries, **kwargs))

    def _rejects(self, prepack, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            _utils().validate_weight_prepack(prepack)

    def test_a_conforming_entry_is_accepted(self):
        self._accepts(_conforming_entry())

    def test_an_unknown_entry_field_is_refused(self):
        # The producer-side failure this file exists to catch: a key added to
        # the C++ literal is rejected here, so it would be rejected on device, at
        # the first launch, long after the compiler accepted it.
        self._rejects(_prepack(_conforming_entry(extra=1)), "unknown field")

    def test_a_missing_entry_field_is_refused(self):
        entry = _conforming_entry()
        del entry["crouton"]
        self._rejects(_prepack(entry), "missing required field")

    def test_an_unknown_top_level_field_is_refused(self):
        prepack = _prepack()
        prepack["surprise"] = True
        self._rejects(prepack, "unknown field")

    def test_both_admitted_source_dtypes_are_accepted(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                self._accepts(_conforming_entry(dtype=dtype))

    def test_a_wrong_dtype_is_refused(self):
        self._rejects(
            _prepack(_conforming_entry(dtype="bf16")), "must be 'f16' or 'f32'"
        )

    def test_both_placements_are_accepted_and_anything_else_refused(self):
        # `location` is new with the DDR mirror (docs/hmx/
        # pack-redundancy-fix-plan-2026-10-09.md P1): the producer emits it on
        # every entry, so a consumer that did not learn it would reject every
        # kernel compiled after the change -- and one that accepted any string
        # would let a typo become a placement nobody can find in the IR.
        for location in LOCATIONS:
            with self.subTest(location=location):
                self._accepts(_conforming_entry(location=location))
        self._rejects(
            _prepack(_conforming_entry(location="pool")),
            "location must be one of",
        )

    def test_the_two_ranks_are_refused_when_wrong(self):
        # Rank is the load-bearing part of the crouton contract, not its values:
        # a producer emitting a different rank would reshape every weight.
        self._rejects(_prepack(_conforming_entry(crouton=[2, 1, 16, 32])), "crouton")
        self._rejects(_prepack(_conforming_entry(shape=[64])), "shape")

    def test_weights_require_a_layout(self):
        # What the empty round-trip fixture never exercises: the producer only
        # ever emits a non-empty layout alongside a non-empty weights list.
        self._rejects(_prepack(_conforming_entry(), layout=None), "layout")
        self._rejects(_prepack(_conforming_entry(), layout={}), "layout")

    def test_duplicate_slots_are_refused(self):
        self._rejects(
            _prepack(_conforming_entry(slot=0), _conforming_entry(slot=0)), "duplicates"
        )

    def test_the_empty_prepack_is_accepted(self):
        # The arm the existing round-trip fixture covers, pinned here so the
        # behaviour tests above are known to be narrowing a set that was not
        # already empty.
        self._accepts(layout=None)

    def test_anti_vacuity_floor(self):
        # The three rejection probes that used to live here -- unknown field,
        # wrong dtype, wrong crouton rank -- were removed on 2026-10-05. They
        # were the only tautological test in this file: the loop incremented
        # `reached` unconditionally after `assertRaisesRegex` returned, and the
        # loop ran a fixed three times, so `assertEqual(reached, 3)` could not
        # fail. It read as a floor on "no failures means no calls" while
        # depending on nothing. The same three rejections are already asserted,
        # by pattern, in the tests above; keeping a second copy of them bought
        # no coverage. What remains is the part that was real: the frozen ranks
        # and tile edge are the ones the producer emits, not numbers chosen
        # independently here.
        utils = _utils()
        self.assertEqual(utils.HMX_TILE_EDGE, 32)
        self.assertEqual(CROUTON_RANK, 5)
        self.assertEqual(LOGICAL_RANK, 2)


if __name__ == "__main__":
    unittest.main()
