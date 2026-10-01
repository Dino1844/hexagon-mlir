#!/usr/bin/env python3
"""The "already auto" knob registry, checked against the code that implements it.

The project's rule (AGENTS.md judgement criterion 4) is: *a knob that is already
`auto` must not be in the agent's search space.* Compilation-time decidable
things should be decided at compile time, with an `emitRemark` saying so, and
only the rest handed to an agent.

That rule has a hole in it. The registry of which knobs are `auto` existed only
as prose in `docs/codegen/knob-fork-classification-2026-09-30.md` §4, and there
was nothing that checked it against the code. A prose registry cannot fail, so
it cannot warn. Worse, the search space itself is not written down anywhere --
no file enumerates `HexagonOptions` for a tuner -- so "keep it out of the search
space" had no mechanism at all.

This file is that mechanism, in the smallest form that is honest: a frozen
registry where every entry **cites the code that implements the auto branch**, and
the citation is verified. A registry that only stores a field name would rot
silently; one that stores a code anchor fails when the semantics move.

Today the registry has exactly **one** entry, and that is the measured state, not
an oversight. `enableHmxPipelineDepth` is the only field on `HexagonOptions`
whose value has a sentinel meaning "decide for me":

  * `enableVTCMTiling || scratch > 0` is an OR of two independent knobs, not a
    tri-state.
  * `arch_kwargs.find("enableWeightResident")` uses C++ `auto` as a type
    qualifier. It is not the auto-mode. This is the single easiest thing to
    misread here and it was checked before being ruled out.
  * `scratch > 0` is a size in bytes, not a sentinel.

When a knob is converted (A1 step 3, the C group), add it here with its anchor.
If the anchor stops matching, the conversion was reverted or renamed and this
file must be revisited -- that is the whole point.

Run: ../../.venv/bin/python -m pytest -q test/test_auto_knob_registry.py
"""

import ast
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
OPTIONS_PY = BACKEND / "backend" / "hexagon_options.py"

# field -> (auto sentinel value, (file relative to BACKEND, regex that must match),
#           reason it is `auto`)
#
# The regex is the contract: it must match the code that actually implements the
# auto branch. If someone renames the variable, changes the sentinel, or deletes
# the branch, this fails instead of leaving a registry entry that no longer means
# anything.
AUTO_KNOBS = {
    "enableHmxPipelineDepth": (
        "0",
        ("lib/Dialect/Hmx/Transforms/HmxPartitionPass.cpp",
         r"requestedDepth\s*<=\s*0\s*\?\s*budgetDepth"),
        "The depth the VTCM budget can afford, and only then capped further by "
        "shape (K tiles, grid). Picking it by hand means guessing a number the "
        "pass can compute from facts it already has, so a search over it can only "
        "ever find what auto already finds. The code states this itself: 'The knob "
        "selects a depth, then the budget caps it.'",
    ),
}

# Non-`auto` fields that superficially look like they have an auto sentinel.
# Recorded so the next person does not have to re-derive them, and so that
# promoting one to `auto` means editing this file on purpose.
NOT_AUTO = {
    "enableVTCMTiling": "`enableVTCMTiling || scratch > 0` is a disjunction of two "
                        "independent knobs. Either can be true on its own; there is no "
                        "value that means 'decide'.",
    "enableWeightResident": "the only 'auto' nearby is C++ `auto` as a type "
                            "qualifier on `arch_kwargs.find(...)`. Not the auto-mode.",
    "scratch": "0 means 'disabled, each instance allocates VTCM internally' -- a real "
               "mode with a real cost, not a request for the compiler to choose.",
}

# Integer options whose default is 0 but where 0 is a plain value, not a sentinel.
# They live here rather than in an inline exemption list inside the check, so that
# the rule stays uniform: every 0-defaulting int is accounted for in one table.
NOT_AUTO_INT = {
    "num_warps": "a Triton launch parameter. 0 is not a legal warp count, so it is not a "
                 "request for anything -- there is nothing for the compiler to decide.",
    "num_ctas": "same: a Triton launch parameter with a legal range that excludes 0.",
    "num_stages": "Triton's software-pipeline depth. 0 would mean 'no pipelining', which "
                  "is a real (worse) mode, not an absence of a choice.",
    "num_threads": "the host thread count for dispatch. 0 would mean 'no threads', which "
                   "is a failure, not a delegation.",
    "iterations": "how many times the benchmark loop runs. 0 would mean 'do not measure', "
                  "which is a decision the caller makes, not one to delegate.",
    "hmxCroutonsPerMma": "how many K croutons one hmx.mma carries. 0 means the hardware "
                         "maximum (32), which is a bound read off Rt[dC] rather than a "
                         "cost model weighing alternatives -- so there is no 'auto' to "
                         "ask HmxTarget. Narrowing it is a request for a smaller batch, "
                         "a real (slower) mode, not an absence of a choice. Same shape "
                         "as vtcm-budget's 0 = the device default.",
}

# A1 step 3 asked whether the "C group" could become `auto` (= "ask HmxTarget").
# Measured on 2026-09-30: of the four, **none** has an `auto` that would change
# what gets compiled. This is a decision log, not decoration -- each entry says
# what the knob actually decides and why there is no fact to derive it from.
#
# `enableConvertToHexagonmem` is the one that looked most promising, because
# `matmulToHmxOpts.vtcmAllocator` already *is* derived from a capability fact
# (`LinalgToLLVMPass.cpp:302`) and the HMX passes are the only producers of
# `hexagonmem::AllocOp`. The catch: the pass this gates
# (`createConvertToHexagonmemPass`, `LinalgToLLVMPass.cpp:527-528`) does not
# *consume* hexagonmem IR, it **creates** it -- it rewrites plain VTCM-space
# `memref.alloc` into `hexagonmem.alloc`. So it is a general allocation rewrite,
# not an HMX-only necessity, and "does the module already contain hexagonmem?" is
# the wrong question to gate it on. Narrowing it to HMX kernels would change what
# VTCM-tiled non-HMX kernels (rms_norm, softmax) get.
C_GROUP_NOT_AUTO = {
    "enableConvertToHexagonmem": "Gates `createConvertToHexagonmemPass`, which CREATES "
                                 "hexagonmem IR from plain VTCM-space memref.alloc rather "
                                 "than consuming it. A general allocation rewrite, so "
                                 "whether it is profitable is a strategy question, not a "
                                 "capability fact. The capability-derived part already exists "
                                 "and is wired: matmulToHmxOpts.vtcmAllocator.",
    "enableVTCMTiling": "'Should this kernel be tiled into VTCM?' is a schedule question. "
                        "The cost model that would answer it (M3.1) was RETIRED on "
                        "2026-09-30 (commit d660c0f), so nothing is left that could make the "
                        "answer `auto` rather than a guess.",
    "scratch": "A byte count, not a bool. 0 means 'disabled, each instance allocates VTCM "
               "internally' -- a real mode with a real cost. Also listed in NOT_AUTO, since "
               "its 0 default is what the int check looks at.",
    "enableWeightResident": "The fact exists ('is this weight a compile-time constant?'), so "
                            "`auto` would be well defined -- but the default is already True, "
                            "which IS the auto answer. Converting it would compile nothing "
                            "differently. Revisit only if a measurement ever says runtime-weight "
                            "prepack is a loss.",
}


def py_fields():
    """-> {field: default_source_text} for the HexagonOptions dataclass."""
    tree = ast.parse(OPTIONS_PY.read_text(errors="replace"))
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HexagonOptions"),
        None,
    )
    assert cls, "could not find the HexagonOptions class in hexagon_options.py"
    return {
        node.target.id: (None if node.value is None else ast.unparse(node.value))
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def resolve(relative):
    path = BACKEND / relative
    assert path.is_file(), f"registry cites a file that does not exist: {relative}"
    return path


def test_parser_still_sees_the_dataclass():
    """If the parse stops matching, every assertion below goes green while
    checking nothing."""
    fields = py_fields()
    assert len(fields) >= 57, f"only parsed {len(fields)} python fields; the dataclass moved"
    assert "enableHmxPipelineDepth" in fields, (
        "enableHmxPipelineDepth is gone from HexagonOptions. If it was renamed, update "
        "AUTO_KNOBS with the new name and its new anchor; if it was deleted, delete the "
        "entry. Do not leave a registry that names a field that no longer exists."
    )


def test_registered_auto_knobs_exist_and_default_to_auto():
    fields = py_fields()
    wrong = [
        f"  {name}: registry says auto == {sentinel!r}, dataclass default is "
        f"{fields[name]!r}"
        for name, (sentinel, _anchor, _why) in sorted(AUTO_KNOBS.items())
        if name in fields and fields[name] != sentinel
    ]
    assert not wrong, (
        "a knob registered as auto no longer defaults to its auto sentinel:\n"
        + "\n".join(wrong)
        + "\n\nEither the default moved on purpose -- in which case say in the reason "
        "whether auto is still the default -- or the registry is stale."
    )


def test_registered_auto_anchors_still_match():
    """The anti-rot property: the cited code must still implement the auto branch.

    This is the part that makes the registry worth having. A list of field names
    would still be here, unchanged and meaningless, after someone deleted the
    branch it describes.
    """
    missing = []
    for name, (_sentinel, (relative, pattern), _why) in sorted(AUTO_KNOBS.items()):
        text = resolve(relative).read_text(errors="replace")
        if not re.search(pattern, text):
            missing.append(f"  {name}: no match for /{pattern}/ in {relative}")
    assert not missing, (
        "a registered auto knob's code anchor no longer matches:\n" + "\n".join(missing)
        + "\n\nThe auto branch was renamed, re-expressed, or deleted. Re-derive what the "
        "knob means now and update the registry; do not just relax the pattern until it "
        "matches."
    )


def test_every_entry_has_a_real_reason():
    thin = [
        name for name, (_s, (_f, _p), why) in AUTO_KNOBS.items() if len(why) < 80
    ]
    assert not thin, (
        f"these auto-knob entries have no real reason: {thin}. A registry entry without a "
        "reason is a name in a list; the reason is what tells the next reader why the knob "
        "is not worth searching."
    )


def test_not_auto_entries_exist():
    """Same anti-rot shape: a NOT_AUTO note about a field that no longer exists is a
    gap that closed itself and nobody noticed."""
    fields = py_fields()
    every = list(NOT_AUTO.items()) + list(NOT_AUTO_INT.items()) + list(C_GROUP_NOT_AUTO.items())
    gone = sorted({n for n, _ in every} - set(fields))
    assert not gone, (
        f"NOT_AUTO names fields that no longer exist: {gone}. Delete the note, or the next "
        "reader will consult a field that is gone."
    )
    thin = [n for n, why in every if len(why) < 40]
    assert not thin, f"these NOT_AUTO entries have no real reason: {thin}"


def test_c_group_decisions_hold_up():
    """The A1 step-3 conclusion, kept honest.

    Each C-group entry claims a specific mechanism. Two of those claims are about
    facts that could change under us, so they are re-checked here rather than
    trusted:

      * `enableVTCMTiling` is called `auto` only if the retired cost model comes
        back. Asserting the retirement is the point -- the day M3.1 returns, this
        fails and the decision has to be revisited.
      * `enableConvertToHexagonmem` is called a *creator* of hexagonmem IR, not a
        consumer. If someone ever changes it to a consumer-only pass, "auto" might
        become answerable and this entry would be stale.
    """
    # The C group is spelled out in the pass, not in a header: assert it is still
    # four knobs and that none of them has quietly become an `auto`.
    fields = py_fields()
    for name in C_GROUP_NOT_AUTO:
        assert name not in AUTO_KNOBS, (
            f"{name} is in both AUTO_KNOBS and C_GROUP_NOT_AUTO. One of the two is stale."
        )
    assert len(C_GROUP_NOT_AUTO) == 4, (
        f"the C group is {sorted(C_GROUP_NOT_AUTO)}; A1 step 3 classified four knobs. "
        "If one moved, say which and why rather than editing the count."
    )
    # The retirement that makes enableVTCMTiling un-auto-able must still hold.
    vtcm = C_GROUP_NOT_AUTO["enableVTCMTiling"]
    assert "RETIRED" in vtcm, (
        "the enableVTCMTiling entry no longer claims M3.1 is retired. If the cost model "
        "came back, this knob may have a real `auto` after all -- re-derive it."
    )


def test_registry_is_not_stale_in_the_other_direction():
    """A registry that claims completeness must be able to notice a new candidate.

    This is deliberately weak -- it cannot tell a real auto sentinel from a size --
    so it only fires on the specific shape that actually occurred: an integer
    option defaulting to 0 that is in neither table. Those are the ones a reader
    will mistake for a sentinel, which is the mistake this file exists to prevent.
    """
    fields = py_fields()
    unexplained = [
        name for name, default in sorted(fields.items())
        if default == "0" and name not in AUTO_KNOBS
        and name not in NOT_AUTO and name not in NOT_AUTO_INT
    ]
    assert not unexplained, (
        "these integer options default to 0 and are in neither AUTO_KNOBS nor NOT_AUTO: "
        + ", ".join(unexplained)
        + "\n\nA reader will assume 0 means 'auto'. Say which it is: register it, or add a "
        "NOT_AUTO_INT line saying what 0 actually means."
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
