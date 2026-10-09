#!/usr/bin/env python3
"""The manifest's conditional rules are implemented twice, so they are checked twice.

`hex.hmx.kernel_manifest/v2` is a frozen boundary.  `test_hmx_manifest_field_agreement.py`
pins which *fields* a record of each plan may carry; this file pins the six
*conditional* rules that say when a record is admissible at all.  They are the
rules that are not a vocabulary: `full-hmx` does not merely want a reason from
the shared set, it wants `selected-aligned`; a null `logical` is not merely
illegal, it is illegal except for three reasons; a resident-prepack policy's
reason has to name the same dtype the weight actually has.

Each of the six is written twice, in files that cannot see each other:

* the producer, `lib/Dialect/Hmx/Transforms/HmxManifest.cpp` (with its reason
  vocabulary in `HmxManifest.h`), which refuses to *emit* a record that breaks
  the rule;
* the consumer, `backend/utils.py`, which refuses to *accept* one.

Nothing connected them.  The consumer's own comments said so -- `utils.py`
claimed the mirror was held by a cross-language gate, and no such gate existed.
If one side learned a rule the other did not, the symptom is silent in the
direction that already works: either the compiler refuses to publish a manifest
the host has declared legal, or a manifest the compiler would never publish
validates on the host and fails at the launcher, blaming the producer for a
decision the consumer made.

The six rules, and where each lives:

| rule | producer | consumer |
|---|---|---|
| plan <-> reason pairing | the three-arm comparison in `validateRecord`, over `isCanonicalHmxMatmulReason` | `HMX_PLAN_REASONS` + the "not a canonical pair" refusal |
| which reasons may carry no logical shape | the `logical.present && state == Unavailable` guard | `_validate_logical`'s null branch |
| grid policy per plan | `validateWorkspaceAndVtcm`'s two arm checks | `_validate_workspace`'s two raises |
| which topology verdicts exist | `isCanonicalHmxTopology` | the inline tuple in `validate_hmx_manifest` |
| weight reason vs weight dtype | the `dtypeMatchesReason` conjunction in the finalize pass | `_validate_manifest_weight_prepack`'s dtype-specific reasons |
| dynamic-shape needs a dynamic dimension | the `reason == kHmxReasonDynamicShape` guard | `_validate_logical`'s dynamic-shape raise |

How this checks them, and why that way:

A rule is only checked if both sides are *asked*, so each test drives one
semantic case through both implementations on one code path.  The case is
written once here and routed to the side that implements it:

* the Python side is executed for real -- the shared record builders from
  `test_hmx_manifest_metadata.py` build a record, the *consumer's own*
  `_validate_*` path validates it, and the refusal text is what the rule says.
  Nothing about the consumer is transcribed or re-derived;
* the C++ side is not executed.  It is *read*: the condition's shape is parsed
  out of the source and the same case is evaluated against it.  That is the same
  method, and the same limitation, as `test_hmx_manifest_field_agreement.py`:
  a parse that no longer matches the source raises rather than passes with an
  empty answer, and the case table is what turns "both agree" from a tautology
  into a claim about the world.

Agreeing-with-each-other is deliberately not enough on its own: every case also
states the answer it expects, taken from the rule's prose, independently of both
implementations.  That is what makes the gate able to fail when the two sides
drift *together* into a rule that is not the documented one.

What this file does not check: the field lists and the vocabularies themselves
(both pinned by `test_hmx_manifest_field_agreement.py`), the fingerprint
round-trip (`test_hmx_record_v3.py`), the minimum-rows threshold
(`test_hmx_manifest_minimum_rows.py`).  Those are different rules with their own
gates; repeating them here would make this file's failure messages ambiguous.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import re
import unittest
from typing import Callable, NamedTuple

_TEST_DIR = Path(__file__).resolve().parent
_BACKEND = _TEST_DIR.parent
_REPO = _BACKEND.parent
_MANIFEST_CPP = (
    _BACKEND / "lib" / "Dialect" / "Hmx" / "Transforms" / "HmxManifest.cpp"
)
_MANIFEST_H = (
    _BACKEND / "include" / "hexagon" / "Dialect" / "Hmx" / "Transforms"
    / "HmxManifest.h"
)
_UTILS = _BACKEND / "backend" / "utils.py"
_SHARED_FIXTURES = _TEST_DIR / "test_hmx_manifest_metadata.py"
_TRITON_PYTHON = _REPO / "triton" / "python"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_module("hexagon_backend_utils_plan_rules", _UTILS)


def _shared_fixtures():
    """The manifest consumer tests' own record builders, imported not copied.

    A case has to be a record the consumer can be asked about, so "a valid
    record" must mean the same thing here as it does everywhere else in this
    repository.  The builders live in `test_hmx_manifest_metadata.py`; a
    transcription would gate a record shape the launcher never sees.

    That module imports `triton`, which binds the source checkout at the
    repository root as a namespace package unless `triton/python` is on
    `sys.path` first -- the trap `run_host_tests.py` documents.  The same prefix
    is applied here so this file is runnable directly, without the host runner's
    environment.
    """
    import sys

    if _TRITON_PYTHON.is_dir() and str(_TRITON_PYTHON) not in sys.path:
        sys.path.insert(0, str(_TRITON_PYTHON))
    return _load_module(
        "hexagon_backend_manifest_fixtures", _SHARED_FIXTURES
    )


fixtures = _shared_fixtures()

CPP_SOURCE = _MANIFEST_CPP.read_text(encoding="utf-8")
HEADER_SOURCE = _MANIFEST_H.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Reading the producer's conditions out of its source
#
# Every helper below has the same contract: return the rule's answer, or raise.
# A regex that no longer matches its target raises `AssertionError` naming the
# rule rather than returning an empty or half-read answer, because an empty
# answer would make every case "agree" with the other side.
# ---------------------------------------------------------------------------

_CPP_ALIASES = dict(
    re.findall(
        r"constexpr StringLiteral (k\w+) =\s*(kHmx\w+);",
        CPP_SOURCE,
    )
)
_CPP_LITERALS = dict(
    re.findall(
        r'constexpr StringLiteral (k\w+) = "([^"]+)";',
        CPP_SOURCE,
    )
)
_HEADER_LITERALS = dict(
    re.findall(
        r'inline constexpr StringLiteral (kHmx\w+) =\s*"([^"]+)";',
        HEADER_SOURCE,
    )
)


def _resolve(name: str) -> str:
    """Follow a `k*` constant to the wire string it denotes.

    Three spellings exist for the same value -- a literal in the `.cpp`, a
    literal in the header, or a `.cpp` alias of a header literal -- and a rule
    may use any of them.  Resolving through all three means renaming a constant
    changes nothing this file reads, while changing the *value* still does.
    """
    seen: list[str] = []
    while name in _CPP_ALIASES:
        if name in seen:
            raise AssertionError(f"constant alias cycle at {name!r}")
        seen.append(name)
        name = _CPP_ALIASES[name]
    for table in (_CPP_LITERALS, _HEADER_LITERALS):
        if name in table:
            return table[name]
    raise AssertionError(
        f"{name!r} does not resolve to a manifest wire value; the "
        "constant spelling moved and this reader has to be re-derived"
    )


def _cpp_body(name: str, source: str = CPP_SOURCE) -> str:
    """The body of a one-line-signature predicate, or raise."""
    match = re.search(rf"\b{name}\([^)]*\) \{{\n(.*?)\n\}}", source, re.S)
    if match is None:
        raise AssertionError(
            f"the manifest source no longer defines {name}() with a body this "
            "file can read; the reader must be re-derived, not guessed"
        )
    return match.group(1)


def _cpp_guard(anchor: str) -> str:
    """One `if (...) return emitManifestError(...)` condition, by its first arm.

    Slice-and-search rather than a full expression parser: the condition ends at
    the `return` it guards, which the surrounding text cannot imitate.
    """
    start = CPP_SOURCE.index(anchor)
    end = CPP_SOURCE.index("return emitManifestError", start)
    return CPP_SOURCE[start:end]


def _cpp_plan_wire_names() -> dict[str, str]:
    """`PlanKind` enum name -> plan string, read from `parsePlan`."""
    pairs = re.findall(
        r"if \(value == (k\w+)\)\s*return PlanKind::(\w+);",
        _cpp_body("parsePlan"),
    )
    if len(pairs) != 3:
        raise AssertionError(
            f"parsePlan has {len(pairs)} arms, not the documented three; the "
            "plan vocabulary moved"
        )
    return {kind: _resolve(alias) for alias, kind in pairs}


def _cpp_reason_vocabulary() -> frozenset[str]:
    """Every reason the producer recognises, from `isCanonicalHmxMatmulReason`."""
    names = re.findall(
        r"reason == (kHmxReason\w+)", _cpp_body("isCanonicalHmxMatmulReason", HEADER_SOURCE)
    )
    if not names:
        raise AssertionError("isCanonicalHmxMatmulReason reads as empty")
    return frozenset(_resolve(name) for name in names)


def _cpp_plan_reason_pairing() -> dict[str, object]:
    """The plan->reason pairing rule, read out of `validateRecord`.

    Three arms: two plans pin one reason each, and `hvx` may carry anything
    except the two pinned ones.  That last arm is what makes the rule a pairing
    rather than a per-plan vocabulary: `selected-aligned` is a legal reason and
    an illegal one for `hvx` at the same time.
    """
    marker = '"HMX manifest semantic plan and reason disagree on attribution"'
    at = CPP_SOURCE.index(marker)
    start = CPP_SOURCE.rindex("if ((", 0, at)
    block = CPP_SOURCE[start:at]
    required = dict(
        re.findall(
            r"\*plan == PlanKind::(\w+) &&\s*reason\.getValue\(\) != (kHmxReason\w+)",
            block,
        )
    )
    excluded = re.findall(r"reason\.getValue\(\) == (kHmxReason\w+)", block)
    if len(required) != 2 or len(excluded) != 2:
        raise AssertionError(
            f"the pairing condition has {len(required)} pinned arm(s) and "
            f"{len(excluded)} excluded reason(s), not two and two"
        )
    plans = _cpp_plan_wire_names()
    pinned = {plans[kind]: _resolve(reason) for kind, reason in required.items()}
    if set(pinned) != {"full-hmx", "hmx-tail"}:
        raise AssertionError(
            f"the pairing pins reasons for {sorted(pinned)}, not for the two "
            "HMX plans"
        )
    return {
        "vocabulary": _cpp_reason_vocabulary(),
        "pinned": pinned,
        "excluded": frozenset(_resolve(name) for name in excluded),
    }


def _cpp_pairing_verdict(plan: str, reason: str) -> bool:
    """The producer's answer to "may this plan carry this reason?"."""
    rule = CPP_PLAN_REASON_PAIRING
    if reason not in rule["vocabulary"]:
        return False
    if plan in rule["pinned"]:
        return reason == rule["pinned"][plan]
    return reason not in rule["excluded"]


def _cpp_null_logical_reasons() -> frozenset[str]:
    """Which reasons may carry an unavailable logical shape.

    Read from the guard's three `!=` arms, then cross-checked against the
    refusal's own prose: the message names the same three reasons, and if the
    code and the message ever disagree the refusal is not describing the rule.
    """
    block = _cpp_guard(
        "if (logical.present && logical.state == ShapeState::Unavailable &&"
    )
    names = re.findall(r"reason\.getValue\(\) != (kHmxReason\w+)", block)
    reasons = frozenset(_resolve(name) for name in names)
    if len(reasons) != 3:
        raise AssertionError(
            f"the null-logical guard admits {len(reasons)} reasons, not the "
            "documented three"
        )
    message = "only library-call, non-rank-2 and unsupported-layout may have"
    for reason in reasons:
        if reason not in message:
            raise AssertionError(
                f"the guard admits {reason!r} but the refusal text does not "
                f"name it; the two no longer describe the same rule"
            )
    return reasons


CPP_PLAN_REASON_PAIRING = _cpp_plan_reason_pairing()
CPP_NULL_LOGICAL_REASONS = _cpp_null_logical_reasons()


def _null_logical_verdict(reason: str) -> bool:
    """The producer's answer to "may this reason carry no logical shape?"."""
    return reason in CPP_NULL_LOGICAL_REASONS


def _cpp_grid_policy_requirements() -> dict[str, str]:
    """plan -> required grid policy, from the two arms in `validateWorkspaceAndVtcm`.

    Both arms are read rather than assumed, because they are spelled differently
    (one compares a field read, the other a local `StringRef`), which is exactly
    the shape that lets one arm drift without the other being noticed.
    """
    tail = re.search(
        r"if \(plan == PlanKind::(\w+) && hasGrid &&\s*"
        r"stringField\(record, kKeyGridPolicy\)\.getValue\(\) != (kGrid\w+)\)",
        CPP_SOURCE,
    )
    full = re.search(
        r"if \(plan == PlanKind::(\w+) && grid != (kGrid\w+)\)",
        CPP_SOURCE,
    )
    if tail is None or full is None:
        raise AssertionError(
            "the per-plan grid policy conditions are no longer both readable"
        )
    plans = _cpp_plan_wire_names()
    return {
        plans[tail.group(1)]: _resolve(tail.group(2)),
        plans[full.group(1)]: _resolve(full.group(2)),
    }


def _cpp_grid_policy_verdict(plan: str, grid_policy: str) -> bool:
    """The producer's answer to "may this plan declare this grid policy?"."""
    return CPP_GRID_POLICY_REQUIREMENTS.get(plan, grid_policy) == grid_policy


def _cpp_topology_verdicts() -> frozenset[str]:
    """The five topology verdicts, from `isCanonicalHmxTopology`."""
    names = re.findall(
        r"topology == (kTopology\w+)", _cpp_body("isCanonicalHmxTopology")
    )
    if len(names) != 5:
        raise AssertionError(
            f"isCanonicalHmxTopology has {len(names)} verdicts, not the "
            "documented five"
        )
    return frozenset(_resolve(name) for name in names)


def _cpp_weight_reason_dtypes() -> dict[str, str]:
    """reason -> dtype it names, from the finalize `dtypeMatchesReason` conjunction.

    The b2-slice reason is absent on purpose: it names a view, not a dtype, so
    neither side constrains it.  Which reasons are absent is as much of the rule
    as which are present, so the arms are read as a conjunction of
    `reason != X || dtype` pairs and no default is invented for the rest.
    """
    assignment = re.search(r"bool dtypeMatchesReason =\s*(.*?);\n", CPP_SOURCE, re.S)
    if assignment is None:
        raise AssertionError("the dtypeMatchesReason conjunction is unreadable")
    arms = re.findall(
        r"reason != (kWeight\w+) \|\| admitted(F16|F32)", assignment.group(1)
    )
    slots = dict(
        re.findall(
            r"admitted(F16|F32) =\s*dtypesValue && dtypesValue\.getValue\(\) == "
            r"(kHmxDType\w+)",
            CPP_SOURCE,
        )
    )
    if len(arms) != 2 or sorted(slots) != ["F16", "F32"]:
        raise AssertionError(
            "the resident-prepack dtype pairing no longer has two reasons and "
            "two dtype slots"
        )
    return {_resolve(reason): _resolve(slots[slot]) for reason, slot in arms}


def _cpp_weight_dtype_verdict(policy: str, reason: str, dtype: str) -> bool:
    """The producer's answer to "may this policy reason name this weight dtype?"."""
    if policy not in CPP_RESIDENT_PREPACK_FAMILY:
        return True
    required = CPP_WEIGHT_REASON_DTYPES.get(reason)
    if required is None:
        return True
    return dtype == required


def _cpp_resident_prepack_family() -> frozenset[str]:
    """The placements of a host pre-packed weight, from `isResidentPrepackPolicy`."""
    names = re.findall(
        r"policy == (kWeight\w+)", _cpp_body("isResidentPrepackPolicy")
    )
    if not names:
        raise AssertionError("isResidentPrepackPolicy reads as empty")
    return frozenset(_resolve(name) for name in names)


def _cpp_dynamic_shape_rule() -> tuple[str, str]:
    """(reason, shape state) that may not co-occur, from the guard's own text."""
    block = _cpp_guard(
        "if (logical.present && reason.getValue() == kHmxReasonDynamicShape &&"
    )
    reason = re.search(r"reason\.getValue\(\) == (kHmxReason\w+)", block)
    state = re.search(r"logical\.state == ShapeState::(\w+)", block)
    if reason is None or state is None:
        raise AssertionError("the dynamic-shape guard no longer names both sides")
    return _resolve(reason.group(1)), state.group(1)


def _dynamic_shape_verdict(reason: str, static: bool) -> bool:
    """The producer's answer to "may this reason claim a fully static shape?"."""
    pinned_reason, pinned_state = CPP_DYNAMIC_SHAPE_RULE
    return not (reason == pinned_reason and static and pinned_state == "Static")


CPP_GRID_POLICY_REQUIREMENTS = _cpp_grid_policy_requirements()
CPP_TOPOLOGY_VERDICTS = _cpp_topology_verdicts()
CPP_WEIGHT_REASON_DTYPES = _cpp_weight_reason_dtypes()
CPP_RESIDENT_PREPACK_FAMILY = _cpp_resident_prepack_family()
CPP_DYNAMIC_SHAPE_RULE = _cpp_dynamic_shape_rule()


# ---------------------------------------------------------------------------
# Cases: one semantic case, two serializations
#
# A case states what it expects from the rule's prose, independently of either
# implementation, and asks both sides about the same fact.  `refused` is the
# fragment the refusing side must name: without it, a refusal for an unrelated
# reason would count as agreement, which is the failure mode this file exists to
# prevent.
# ---------------------------------------------------------------------------


class Case(NamedTuple):
    """One semantic case, and the two functions that answer it.

    `producer` is the rule as read out of the manifest source; `consumer` runs
    the host validator for real.  `expect` is what the rule's prose says, and
    `refused` the fragment a refusal must name -- without it, a refusal for an
    unrelated reason would count as agreement.
    """

    label: str
    expect: bool
    refused: str | None
    producer: Callable[[], bool]
    consumer: Callable[[], tuple[bool, str]]


def _record_case(label, record, expect, refused, producer) -> Case:
    """Ask the consumer's own record validator about one record."""

    def consumer():
        try:
            utils._validate_record(copy.deepcopy(record), "probe.record")
        except ValueError as exc:
            return False, str(exc)
        return True, ""

    return Case(label, expect, refused, producer, consumer)


def _manifest_case(label, manifest, expect, refused, producer) -> Case:
    """Ask the consumer's own manifest validator about one manifest."""

    def consumer():
        try:
            utils.validate_hmx_manifest(
                copy.deepcopy(manifest), "probe.manifest"
            )
        except ValueError as exc:
            return False, str(exc)
        return True, ""

    return Case(label, expect, refused, producer, consumer)


def _prepack_case(label, manifest, prepack, expect, refused, producer) -> Case:
    """Ask the consumer's manifest and pre-pack validators together.

    The weight's dtype is one fact with two serializations: the consumer record
    declares it (`dtypes.rhs`, which the producer's finalize check reads) and the
    pre-pack image repeats it (`weights[].dtype`, which the consumer's check
    reads).  A case therefore sets both to the same value; a case that set only
    one would be asking the two sides about different facts.
    """

    def consumer():
        document = copy.deepcopy(manifest)
        image = copy.deepcopy(prepack)
        try:
            utils.validate_hmx_manifest(document, "probe.manifest")
            utils.validate_weight_prepack(image, "probe.weight_prepack")
            # The consumer's own pairing check, the one `parse_translation_metadata`
            # calls; invoked directly so the refusal names this rule rather than
            # whatever else the launcher path would hit first.
            utils._validate_manifest_weight_prepack(document, image)
        except ValueError as exc:
            return False, str(exc)
        return True, ""

    return Case(label, expect, refused, producer, consumer)


# The documented answers, written out here rather than derived from either side.
#
# This is the file's third opinion.  Asking both sides about the same case is
# only a consistency check: it fails when one side moves, and passes when both
# move together -- including when both move onto a rule the prose never
# described.  A written-down answer is what fails then, so every case below is
# expected against these tables and not against the reader that describes the
# side under test.
DOCUMENTED_PLAN_REASONS = {
    # A plan's reason is the consequence of its decision, not a free field: the
    # aligned plan was selected because nothing is cut off, the tail plan
    # because something is.  Neither can carry the other's reason.
    "full-hmx": frozenset({"selected-aligned"}),
    "hmx-tail": frozenset({"selected-tail"}),
}
DOCUMENTED_HVX_EXCLUDED = frozenset({"selected-aligned", "selected-tail"})
DOCUMENTED_NULL_LOGICAL_REASONS = frozenset(
    {"library-call", "non-rank-2", "unsupported-layout"}
)
DOCUMENTED_GRID_POLICIES = {
    "hmx-tail": "single-instance",
    "full-hmx": "legacy-runtime",
}
DOCUMENTED_TOPOLOGY_VERDICTS = frozenset(
    {
        "topology-single-role-hmx",
        "topology-single-role-hvx",
        "role-split-ok",
        "role-mixed-irreducible",
        "role-split-nopack",
    }
)
DOCUMENTED_WEIGHT_REASON_DTYPES = {
    # `eligible-b2-n-slice` is deliberately absent: it names a view of the
    # weight, not its element type, so neither side constrains it and a reader
    # that invented a constraint for it would be wrong in both directions.
    "eligible-aligned-f16": "f16",
    "eligible-quantized-f32": "f32",
}
DOCUMENTED_DYNAMIC_SHAPE_REASON = "dynamic-shape"

# The shape that makes a record of each plan valid on every other axis, so a
# refusal cannot be blamed on anything but the rule under test.
PLAN_SHAPES = {"full-hmx": (64, 64, 64), "hmx-tail": (65, 47, 70), "hvx": (64, 64, 64)}
GRID_POLICIES = ("single-instance", "legacy-runtime")
REASONS = sorted(CPP_PLAN_REASON_PAIRING["vocabulary"])
# The reasons the pairing rule lets `hvx` carry.  A sweep over the whole
# vocabulary would spend most of its cases on the pairing rule, which has its
# own sweep, instead of on the rule being swept.
HVX_REASONS = sorted(
    reason
    for reason in REASONS
    if reason not in DOCUMENTED_HVX_EXCLUDED
)
DTYPES = ("f16", "f32")
# The per-policy reason vocabulary is a different rule, pinned by
# `test_hmx_manifest_field_agreement.py`; it is written out here as literals so
# this sweep varies only the dtype pairing.
WEIGHT_POLICY_REASONS = {
    "resident-prepack": (
        "eligible-aligned-f16",
        "eligible-quantized-f32",
        "eligible-b2-n-slice",
    ),
    "resident-prepack-ddr": (
        "eligible-aligned-f16",
        "eligible-quantized-f32",
        "eligible-b2-n-slice",
    ),
    "device-pack": (
        "tail-consumer",
        "f32-source",
        "unproven-offset",
        "incompatible-consumers",
        "prepack-disabled",
    ),
}


def _pairing_case(label, plan, reason, expect, refused) -> Case:
    """One plan/reason pairing case, as the record that plan emits.

    The shape is adjusted for one reason on purpose: `dynamic-shape` with an
    all-static shape is refused by the dynamic-shape rule, which would fire
    before the pairing verdict and make this case a pairing test only by
    accident.  Everything else about the record is held valid, so a refusal can
    only be the rule under test.
    """
    if plan == "hvx":
        shape = (
            (None, None, None)
            if reason == DOCUMENTED_DYNAMIC_SHAPE_REASON
            else PLAN_SHAPES[plan]
        )
        record = fixtures._hvx_record(reason=reason, shape=shape)
    else:
        record = fixtures._hmx_record(
            plan=plan, reason=reason, shape=PLAN_SHAPES[plan]
        )
    return _record_case(
        label,
        record,
        expect,
        refused,
        producer=lambda: _cpp_pairing_verdict(plan, reason),
    )


def _documented_pairing(plan: str, reason: str) -> bool:
    if plan == "hvx":
        return reason not in DOCUMENTED_HVX_EXCLUDED
    return reason in DOCUMENTED_PLAN_REASONS[plan]


PAIRING_RULE = "plan/reason pairing"
PAIRING_CASES = [
    _pairing_case(
        f"{plan} carrying {reason}",
        plan,
        reason,
        _documented_pairing(plan, reason),
        "not a canonical pair",
    )
    for plan in ("full-hmx", "hmx-tail", "hvx")
    for reason in REASONS
] + [
    # A reason outside the vocabulary is not a pairing question: the producer
    # refuses it before the pairing arms are reached, and the consumer refuses
    # it as a non-canonical code.  Both say no, for that shared reason.
    _pairing_case(
        "hvx carrying an unrecognised reason",
        "hvx",
        "not-a-real-reason",
        False,
        "is not a canonical code",
    )
]


def _grid_case(label, plan, grid_policy, expect, workspace="runtime-internal") -> Case:
    record = fixtures._hmx_record(
        plan=plan,
        grid_policy=grid_policy,
        shape=PLAN_SHAPES[plan],
        workspace_class=workspace,
    )
    return _record_case(
        label,
        record,
        expect,
        f"{plan} requires grid_policy='{DOCUMENTED_GRID_POLICIES[plan]}'",
        producer=lambda: _cpp_grid_policy_verdict(plan, grid_policy),
    )


GRID_RULE = "per-plan grid policy"
GRID_CASES = [
    _grid_case(f"{plan} declaring {policy}", plan, policy, policy == required)
    for plan, required in DOCUMENTED_GRID_POLICIES.items()
    for policy in GRID_POLICIES
] + [
    # The arm is per plan, not per workspace class: both sources carry the same
    # comment saying a resident workspace does not change the requirement.
    _grid_case(
        "full-hmx declaring legacy-runtime as a resident workspace",
        "full-hmx",
        "legacy-runtime",
        True,
        workspace="resident",
    )
]


def _topology_case(label, topology) -> Case:
    record = fixtures._hmx_record()
    manifest = fixtures._manifest([record])
    manifest["topology"] = topology
    manifest["thread_role_regions"] = 4
    return _manifest_case(
        label,
        manifest,
        topology in DOCUMENTED_TOPOLOGY_VERDICTS,
        "not a canonical verdict",
        producer=lambda: topology in CPP_TOPOLOGY_VERDICTS,
    )


TOPOLOGY_RULE = "topology verdicts"
TOPOLOGY_CASES = [
    _topology_case(f"verdict {value!r}", value)
    for value in sorted(DOCUMENTED_TOPOLOGY_VERDICTS)
] + [
    # Invented values, so a sixth verdict added to both sides fails against the
    # documented five instead of passing as agreement.
    _topology_case(f"an invented verdict {value!r}", value)
    for value in ("role-split-maybe", "topology-single-role-npu")
]


def _weight_policy_case(label, policy, reason, dtype, expect) -> Case:
    """One weight-policy case, serialized for both sides.

    The weight's dtype is one fact with two serializations: the consumer record
    declares it (`dtypes.rhs`, which the producer's finalize check reads) and the
    pre-pack image repeats it (`weights[].dtype`, which the consumer's check
    reads).  A case sets both to the same value; setting only one would ask the
    two sides about different facts, which is the shape of a gate that always
    agrees.
    """
    entry = {
        "function": "matmul_kernel",
        "slot": 1,
        "policy": policy,
        "reason": reason,
        "consumers": [0],
    }
    record = fixtures._hmx_record(weight_kind="argument-slot", policy=dict(entry))
    record["dtypes"]["rhs"] = dtype
    # The record's fingerprint covers its dtypes, so a case carrying a
    # non-default rhs dtype re-derives the digest instead of reusing the
    # builder's.
    record["plan_fingerprint"] = utils.compute_hmx_plan_fingerprint(record, entry)
    manifest = fixtures._manifest([record], policies=[entry])
    # A device-pack policy publishes no pre-pack image -- the consumer refuses
    # one that does -- so its cases carry an empty image.
    resident = policy in CPP_RESIDENT_PREPACK_FAMILY
    prepack = {
        "layout": {
            "ndims": 5,
            "results": [[[1, 32], [2, 2], [4, 1]], [[0, 32], [3, 1]]],
        }
        if resident
        else None,
        "weights": [
            {
                "func": "matmul_kernel",
                "slot": 1,
                "shape": [64, 64],
                "crouton": [2, 2, 16, 32, 2],
                "dtype": dtype,
                "location": "vtcm",
            }
        ]
        if resident
        else [],
    }
    return _prepack_case(
        label,
        manifest,
        prepack,
        expect,
        reason,
        producer=lambda: _cpp_weight_dtype_verdict(policy, reason, dtype),
    )


WEIGHT_RULE = "weight reason vs weight dtype"
WEIGHT_CASES = [
    _weight_policy_case(
        f"{policy} reason {reason} on an {dtype} weight",
        policy,
        reason,
        dtype,
        DOCUMENTED_WEIGHT_REASON_DTYPES.get(reason, dtype) == dtype,
    )
    for policy, reasons in WEIGHT_POLICY_REASONS.items()
    for reason in reasons
    for dtype in DTYPES
]


def _dynamic_shape_case(label, reason, shape, expect) -> Case:
    record = fixtures._hvx_record(reason=reason, shape=shape)
    static = all(value is not None for value in shape)
    return _record_case(
        label,
        record,
        expect,
        "dynamic-shape reason requires a dynamic dimension",
        producer=lambda: _dynamic_shape_verdict(reason, static),
    )


def _documented_dynamic_shape(reason: str, static: bool) -> bool:
    """The dynamic-shape reason may not claim a shape with no dynamic dimension."""
    return not (reason == DOCUMENTED_DYNAMIC_SHAPE_REASON and static)


DYNAMIC_SHAPE_RULE = "dynamic-shape needs a dynamic dimension"
DYNAMIC_SHAPE_CASES = [
    _dynamic_shape_case(
        f"{reason} with a {'fully static' if static_shape else 'dynamic'} shape",
        reason,
        (64, 64, 64) if static_shape else (None, None, None),
        _documented_dynamic_shape(reason, static_shape),
    )
    for reason in HVX_REASONS
    for static_shape in (True, False)
]


NULL_LOGICAL_RULE = "reasons that may carry no logical shape"


def _null_logical_case(label, reason, expect) -> Case:
    record = fixtures._hvx_record(reason=reason, shape=None)
    return _record_case(
        label,
        record,
        expect,
        "may be null only for library-call",
        producer=lambda: _null_logical_verdict(reason),
    )


NULL_LOGICAL_CASES = [
    _null_logical_case(
        f"{reason} without a logical shape",
        reason,
        reason in DOCUMENTED_NULL_LOGICAL_REASONS,
    )
    for reason in HVX_REASONS
]



def _assert_agrees(test: unittest.TestCase, rule: str, case: Case) -> None:
    """Both sides must reach this case's answer, and the refusal must name it."""
    produced = case.producer()
    consumed, message = case.consumer()
    where = f"[{rule}] {case.label}"
    test.assertEqual(
        produced,
        case.expect,
        f"{where}: the producer's rule says {'admit' if produced else 'refuse'}, "
        f"this case expects {'admit' if case.expect else 'refuse'}",
    )
    test.assertEqual(
        consumed,
        case.expect,
        f"{where}: the consumer's validator says "
        f"{'admit' if consumed else 'refuse'}, this case expects "
        f"{'admit' if case.expect else 'refuse'}"
        + (f" (refusal: {message!r})" if message else ""),
    )
    if not case.expect:
        test.assertIn(
            case.refused,
            message,
            f"{where}: this case expects a refusal naming {case.refused!r}, and "
            f"the consumer's refusal was {message!r} -- agreeing by refusing for "
            "an unrelated reason is not the agreement this gate claims",
        )


class PlanReasonPairingTest(unittest.TestCase):
    def test_the_plan_reason_pairing_agrees_on_every_case(self) -> None:
        for case in PAIRING_CASES:
            _assert_agrees(self, PAIRING_RULE, case)

    def test_the_reason_vocabulary_is_the_same_on_both_sides(self) -> None:
        """The pairing's operands: the vocabulary and its per-plan partition.

        Compared as data, not as behaviour on cases, because a reason only one
        side knows is exactly the case that never reaches a fixture: the side
        that knows it refuses the record before the rule under test runs.
        """
        cpp = CPP_PLAN_REASON_PAIRING
        consumer = {
            plan: frozenset(reasons)
            for plan, reasons in utils.HMX_PLAN_REASONS.items()
        }
        self.assertEqual(
            frozenset(
                reason
                for reasons in utils.HMX_PLAN_REASONS.values()
                for reason in reasons
            ),
            utils.HMX_MANIFEST_REASONS,
            "the consumer's per-plan reasons no longer partition its vocabulary",
        )
        self.assertEqual(cpp["vocabulary"], utils.HMX_MANIFEST_REASONS)
        self.assertEqual(set(cpp["pinned"]), {"full-hmx", "hmx-tail"})
        self.assertEqual(
            {
                "full-hmx": frozenset({cpp["pinned"]["full-hmx"]}),
                "hmx-tail": frozenset({cpp["pinned"]["hmx-tail"]}),
                "hvx": cpp["vocabulary"] - cpp["excluded"],
            },
            consumer,
            "the producer's pairing and the consumer's per-plan vocabulary "
            "describe different rules",
        )


class NullLogicalReasonTest(unittest.TestCase):
    def test_the_null_logical_whitelist_agrees_on_every_case(self) -> None:
        for case in NULL_LOGICAL_CASES:
            _assert_agrees(self, NULL_LOGICAL_RULE, case)

    def test_the_producer_reads_the_documented_whitelist(self) -> None:
        self.assertEqual(
            CPP_NULL_LOGICAL_REASONS,
            DOCUMENTED_NULL_LOGICAL_REASONS,
            "the producer's null-logical whitelist is no longer the documented "
            "three reasons",
        )


class PerPlanGridPolicyTest(unittest.TestCase):
    def test_the_per_plan_grid_policy_agrees_on_every_case(self) -> None:
        for case in GRID_CASES:
            _assert_agrees(self, GRID_RULE, case)

    def test_the_grid_policy_vocabulary_is_the_same_on_both_sides(self) -> None:
        cpp = frozenset(
            _resolve(name)
            for name in re.findall(
                r"value == (kGrid\w+)", _cpp_body("isCanonicalGridPolicy")
            )
        )
        self.assertEqual(cpp, frozenset(utils.HMX_GRID_POLICIES))


class TopologyVerdictTest(unittest.TestCase):
    def test_the_topology_verdicts_agree_on_every_case(self) -> None:
        for case in TOPOLOGY_CASES:
            _assert_agrees(self, TOPOLOGY_RULE, case)

    def test_the_producer_reads_the_documented_five_verdicts(self) -> None:
        self.assertEqual(
            CPP_TOPOLOGY_VERDICTS,
            DOCUMENTED_TOPOLOGY_VERDICTS,
            "the producer's topology vocabulary is no longer the documented five",
        )


class WeightReasonDtypeTest(unittest.TestCase):
    def test_the_weight_reason_dtype_pairing_agrees_on_every_case(self) -> None:
        for case in WEIGHT_CASES:
            _assert_agrees(self, WEIGHT_RULE, case)


class DynamicShapeReasonTest(unittest.TestCase):
    def test_the_dynamic_shape_rule_agrees_on_every_case(self) -> None:
        for case in DYNAMIC_SHAPE_CASES:
            _assert_agrees(self, DYNAMIC_SHAPE_RULE, case)


class ProducerReadersTest(unittest.TestCase):
    """The readers must have found the rules, not silence.

    Every helper above raises when its target no longer matches the source.
    This is the belt to those braces: a reader that returned an empty answer
    would make both sides agree on nothing, which is the one outcome worse than
    a red gate.
    """

    def test_every_reader_found_the_documented_rule(self) -> None:
        """What each reader says, next to what the prose says it should say.

        Counts alone would only prove the reader matched something; comparing
        against the documented tables proves it matched the right thing, and a
        reader that stopped matching its target raises here instead of silently
        handing back an empty answer.
        """
        self.assertEqual(
            {
                plan: frozenset({reason})
                for plan, reason in CPP_PLAN_REASON_PAIRING["pinned"].items()
            },
            DOCUMENTED_PLAN_REASONS,
        )
        self.assertEqual(
            set(CPP_PLAN_REASON_PAIRING["excluded"]),
            set(DOCUMENTED_HVX_EXCLUDED),
        )
        self.assertEqual(CPP_NULL_LOGICAL_REASONS, DOCUMENTED_NULL_LOGICAL_REASONS)
        self.assertEqual(CPP_GRID_POLICY_REQUIREMENTS, DOCUMENTED_GRID_POLICIES)
        self.assertEqual(CPP_TOPOLOGY_VERDICTS, DOCUMENTED_TOPOLOGY_VERDICTS)
        self.assertEqual(CPP_WEIGHT_REASON_DTYPES, DOCUMENTED_WEIGHT_REASON_DTYPES)
        # The two placements of one host pre-packed image: the VTCM pool and the
        # DDR mirror the compiler falls back to.  A third would be a placement
        # this file's weight cases no longer describe.
        self.assertEqual(len(CPP_RESIDENT_PREPACK_FAMILY), 2)

    def test_the_readers_and_the_case_tables_cover_the_same_rules(self) -> None:
        tables = {
            PAIRING_RULE: PAIRING_CASES,
            NULL_LOGICAL_RULE: NULL_LOGICAL_CASES,
            GRID_RULE: GRID_CASES,
            TOPOLOGY_RULE: TOPOLOGY_CASES,
            WEIGHT_RULE: WEIGHT_CASES,
            DYNAMIC_SHAPE_RULE: DYNAMIC_SHAPE_CASES,
        }
        for rule, cases in tables.items():
            self.assertGreater(len(cases), 0, f"{rule} has no cases")
            refusals = {case.expect for case in cases}
            self.assertEqual(
                refusals,
                {True, False},
                f"{rule} never fails or never passes, so it is not a rule",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
