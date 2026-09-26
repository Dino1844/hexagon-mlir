#!/usr/bin/env python3
"""The manifest's allowed-field rule is implemented twice, so it is checked twice.

`hex.hmx.kernel_manifest/v2` is a frozen boundary: byte and behaviour invariant.
The rule that says which fields a plan record may carry is implemented in two
places that cannot see each other:

* the producer, `lib/Dialect/Hmx/Transforms/HmxManifest.cpp`, which refuses to
  emit a record carrying an unknown field;
* the consumer, `backend/utils.py::_validate_record`, which refuses to accept one.

Nothing in this repository connected them. If the producer learned a new field
the consumer did not, a legitimate manifest compiled fine and then failed at the
launcher with a message about a field the compiler had just emitted. If the
consumer learned a field the producer did not, the compiler would refuse to emit
something the backend had declared legal. Both are silent until someone runs the
other side.

So this test asserts three things agree per plan: a frozen literal written here,
the C++ `allowedFields` construction, and the Python `allowed` tuples. The literal
is what makes it an invariant rather than a consistency check -- editing both
implementations in the same commit still fails until the boundary is changed on
purpose, in this file, where the change is visible.

Adding a field to the frozen v2 boundary is not something this test can authorise.
`hex.hmx.kernel_manifest/v3` exists for that, and it is record-only. A change here
means either the boundary moved on purpose or one of the two implementations has
a bug; decide which before editing the literal.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest

_TEST_DIR = Path(__file__).resolve().parent
_BACKEND = _TEST_DIR.parent
_MANIFEST = (
    _BACKEND / "lib" / "Dialect" / "Hmx" / "Transforms" / "HmxManifest.cpp"
)
_UTILS = _BACKEND / "backend" / "utils.py"

MANIFEST_SCHEMA = "hex.hmx.kernel_manifest/v2"

# The frozen boundary, per plan. Written out rather than derived, on purpose.
BASE_FIELDS = frozenset(
    {
        "function",
        "id",
        "plan",
        "reason",
        "shape_state",
        "logical",
        "dtypes",
        "plan_fingerprint",
    }
)
HMX_FIELDS = frozenset(
    {
        "padded",
        "full",
        "tail",
        "layout",
        "workspace_class",
        "grid_policy",
        "vtcm_accounting",
        "vtcm_budget_bytes",
        "vtcm_before_bytes",
        "vtcm_bridge_peak_bytes",
        "execution",
        "weight_binding",
    }
)
TAIL_ONLY_FIELDS = frozenset({"tail_policy"})

EXPECTED = {
    "hvx": BASE_FIELDS,
    "full-hmx": BASE_FIELDS | HMX_FIELDS,
    "hmx-tail": BASE_FIELDS | HMX_FIELDS | TAIL_ONLY_FIELDS,
}

# The plan vocabulary is part of the same boundary: a fourth plan name would be a
# field the two validators do not agree about, because neither has a branch for
# it.
PLANS = frozenset({"full-hmx", "hmx-tail", "hvx"})


def _cpp_key_constants() -> dict[str, str]:
    source = _MANIFEST.read_text(encoding="utf-8")
    constants = dict(
        re.findall(r'constexpr StringLiteral (kKey\w+) = "([^"]+)";', source)
    )
    if not constants:
        raise AssertionError("no kKey constants found; the source moved")
    return constants


def _cpp_allowed_fields() -> dict[str, frozenset[str]]:
    """Read the producer's `allowedFields` construction out of the source.

    Resolved through the `kKey*` constants rather than matched as bare strings,
    so renaming a constant changes one place and this follows it, while renaming
    the *field* it denotes is still a boundary change and still fails.
    """

    source = _MANIFEST.read_text(encoding="utf-8")
    keys = _cpp_key_constants()
    start = source.index("SmallVector<StringRef, 24> allowedFields = {")
    end = source.index("if (failed(", start)
    block = source[start:end]

    def names(text: str) -> list[str]:
        return [keys[token] for token in re.findall(r"\b(kKey\w+)\b", text)]

    base = names(block[: block.index("};")])

    # One loop adds the fields every non-HVX plan carries; the tail plan adds one
    # more by a direct push. Both are gated, and the gates are part of what is
    # being checked, so they are read rather than assumed.
    loop = re.search(
        r"if \(\*plan != PlanKind::(\w+)\) \{\s*"
        r"for \(StringRef field\s*:\s*\{(.*?)\}\)\s*"
        r"allowedFields\.push_back\(field\);",
        block,
        re.S,
    )
    if loop is None:
        raise AssertionError("the non-HVX field list is no longer a gated loop")
    if loop.group(1) != "HVX":
        raise AssertionError(
            f"the non-HVX field list is gated on != {loop.group(1)}, not != HVX"
        )
    non_hvx = names(loop.group(2))

    tail = re.search(
        r"if \(\*plan == PlanKind::(\w+)\)\s*"
        r"allowedFields\.push_back\((kKey\w+)\);",
        block,
    )
    if tail is None:
        raise AssertionError("the tail-only field is no longer a gated push_back")
    if tail.group(1) != "HMXTail":
        raise AssertionError(
            f"the tail-only field is gated on == {tail.group(1)}, not == HMXTail"
        )
    tail_only = names(tail.group(2))
    return {
        "base": frozenset(base),
        "non_hvx": frozenset(non_hvx),
        "tail_only": frozenset(tail_only),
    }


def _python_allowed_fields() -> dict[str, frozenset[str]]:
    """Ask the consumer which fields it accepts, rather than reading its source.

    The consumer's rule is `_require_exact_fields(entry, path, allowed)`: a record
    is accepted only when its key set *equals* the allowed set. So one accepted
    record per plan already states the answer -- its keys are the allowed set, by
    the consumer's own definition rather than by this file's reading of it.

    Asking is the right tool and a source scan would be the wrong one: a scan
    would have to understand a three-branch if/elif/else whose arms are multi-line
    tuples, and a parser that clever is a second thing that can be wrong in a way
    the test does not notice. A negative control at the end confirms the rule is
    exact, so "accepted" cannot quietly mean "tolerated".
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hexagon_backend_utils_probe", _UTILS
    )
    assert spec is not None and spec.loader is not None
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    # The same record builders the manifest consumer tests use, so "a valid
    # record" means one thing across this repository.
    spec2 = importlib.util.spec_from_file_location(
        "hexagon_backend_manifest_tests", _TEST_DIR / "test_hmx_manifest_metadata.py"
    )
    assert spec2 is not None and spec2.loader is not None
    fixtures = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(fixtures)

    records = {
        "full-hmx": fixtures._hmx_record(plan="full-hmx", reason="selected-aligned"),
        "hmx-tail": fixtures._hmx_record(
            plan="hmx-tail", reason="selected-tail", shape=(65, 47, 70)
        ),
        "hvx": fixtures._hvx_record(),
    }
    if set(records) != set(PLANS):
        raise AssertionError(f"no valid record for every plan: {sorted(records)}")

    allowed: dict[str, frozenset[str]] = {}
    for plan, record in records.items():
        # Accepted, so its key set is the allowed set exactly.
        utils._validate_record(record, f"probe.{plan}")
        allowed[plan] = frozenset(record)

    # Negative control: the rule is exact, so an unrecognised key is refused. If
    # this ever passes silently the probe above would be measuring tolerance.
    widened = dict(records["full-hmx"])
    widened["unregistered_field"] = "probe"
    try:
        utils._validate_record(widened, "probe.widened")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "the consumer accepted an unrecognised field, so a record's key set "
            "is not its allowed set and this probe cannot be trusted"
        )
    return allowed


class ManifestFieldAgreement(unittest.TestCase):
    def test_producer_and_consumer_agree_with_the_frozen_literal(self) -> None:
        cpp = _cpp_allowed_fields()
        py = _python_allowed_fields()

        self.assertEqual(cpp["base"], BASE_FIELDS, "producer base drifted")
        self.assertEqual(cpp["non_hvx"], HMX_FIELDS, "producer HMX set drifted")
        self.assertEqual(cpp["tail_only"], TAIL_ONLY_FIELDS, "producer tail set drifted")

        self.assertEqual(py["hvx"], EXPECTED["hvx"], "consumer hvx drifted")
        self.assertEqual(py["full-hmx"], EXPECTED["full-hmx"], "consumer full-hmx drifted")
        self.assertEqual(py["hmx-tail"], EXPECTED["hmx-tail"], "consumer hmx-tail drifted")

        # The two implementations, compared to each other rather than only to the
        # literal, because that is the comparison whose absence this file exists
        # to fix. A field one side learns and the other does not is the failure
        # this catches: the manifest compiles and then fails at the launcher, or
        # the backend declares legal something the compiler refuses to emit.
        self.assertEqual(cpp["base"], py["hvx"], "base fields differ between the two")
        self.assertEqual(cpp["base"] | cpp["non_hvx"], py["full-hmx"])
        self.assertEqual(
            cpp["base"] | cpp["non_hvx"] | cpp["tail_only"], py["hmx-tail"]
        )

    def test_the_three_plans_partition_the_vocabulary(self) -> None:
        self.assertEqual(set(EXPECTED), set(PLANS))
        # `hvx` carries no HMX geometry: that is what makes it the cheap arm, and
        # a field leaking into it would widen the arm the cost model treats as
        # different.
        self.assertEqual(EXPECTED["hvx"], BASE_FIELDS)
        self.assertEqual(EXPECTED["hvx"] & HMX_FIELDS, frozenset())
        # `tail_policy` is the one field that distinguishes the tail plan, so it
        # must be in exactly one plan.
        carriers = {plan for plan, fields in EXPECTED.items() if TAIL_ONLY_FIELDS & fields}
        self.assertEqual(carriers, {"hmx-tail"})

    def test_the_schema_string_is_the_frozen_one(self) -> None:
        cpp = _MANIFEST.read_text(encoding="utf-8")
        utils = _UTILS.read_text(encoding="utf-8")
        self.assertIn(f'kManifestSchema = "{MANIFEST_SCHEMA}"', cpp)
        self.assertIn(f'HMX_MANIFEST_SCHEMA = "{MANIFEST_SCHEMA}"', utils)
        # v3 exists and is record-only. Its presence next to v2 is deliberate; a
        # v2 producer that started emitting v3 keys would break every consumer
        # that still validates v2, so the two schemas must not be conflated.
        self.assertIn('HMX_RECORD_SCHEMA = "hex.hmx.kernel_manifest/v3"', utils)


if __name__ == "__main__":
    unittest.main(verbosity=2)
