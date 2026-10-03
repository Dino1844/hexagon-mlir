#!/usr/bin/env python3
"""The Python option surface and the TableGen option surface must agree.

Two defects motivated this, both found on 2026-09-30 while deleting dead
option fields:

1. **Thirteen `LinalgToLLVM` pass options have no Python field at all.** They
   silently take their `.td` default, so a user has no way to change them, while
   each one steers real code in `LinalgToLLVMPass.cpp`. Twelve are upstream; the
   thirteenth, `enableSeedLayoutConversions`, became an orphan the same day its
   Python field was removed as a proven no-op. That removal was right -- this
   file is where the value it now takes is pinned on the record.

2. **The two sides are coupled by name, and the failure mode is a thrown
   exception rather than a silent default.** `MLLVMIRTranslation.cpp` reads
   `arch_kwargs.at("...")` (29 sites) and `arch_kwargs` is built from
   `HexagonOptions().__dict__`. So deleting a Python field without deleting its
   `.at()` produced `IndexError: unordered_map::at` in sixteen tests.
   `test_arch_kwargs_contract.py` covers the other direction -- a C++ key must
   have a Python field -- and does not cover this one.

What this asserts, and why each part is here:

  * Every `Option<>` of `def LinalgToLLVM` is either wired to a Python field or
    listed in `PINNED_DEFAULTS` with a reason. This is the gate: adding a pass
    option without deciding its exposure fails here, by name, in the message.
  * Each pinned value still equals the `.td` default, so a TableGen default bump
    cannot slip through.
  * For every wired option the two defaults are compared -- **all of them, with
    no exemption list**. An earlier draft of this file could only compare 29 of
    35 because a regex could not read a parenthesised multi-line default or an
    escaped-quote string. Parsing `hexagon_options.py` with `ast` instead of a
    regex closed that gap completely, so there is no `NOT_COMPARED` table to
    drift out of date. **A check that is skipped is a check that does not
    exist**; if one is ever needed, it must be named with a reason here.

Scope limits. Each of these is a trap that was hit while writing this:

  * **Only `def LinalgToLLVM` is the Python surface.** That table declares seven
    passes; the other six take options set by pipeline lambdas
    (`upperFrontier`, `skipVectorRowReduce`, `vtcmBudget`, ...) which must NOT
    have Python fields. Matching every `Option<` in the file yields ~26 false
    positives. A test that cries wolf gets ignored, which is worse than no test.
  * **Key on the first argument, never the flag name.** `disableLWPLoop`'s flag
    is `loop`, `LWPloopDepth`'s is `depth`, `tileSizes`'s is
    `tile-sizes-override`, and `enableMatmulToConv` is camelCase even as a flag.
  * The `.td` default is a *C++ source literal*, so `device_type`'s default is
    the text `"hexagon"` while the Python default is the string `hexagon`.
    Comparing raw text reports a phantom divergence; `normalise` unquotes the
    C++ layer too.
  * The Hmx dialect's own `Passes.td` is a **third tier**: it is not bridged by
    name at all, because `LinalgToLLVMPass` fills those option structs from
    derived values (`recordOnly = !enableBufferization`, `pipelineDepth =
    enableHmxPipelineDepth`, ...) rather than from `HexagonOptions`. So the rule
    there is different: every option must be **set by the pipeline** or be a
    **deliberate CLI/test-only knob** named in `HMX_CLI_ONLY`, with the test that
    justifies it. See `test_every_hmx_option_is_covered` below.

Scope limits. Each of these is a trap that was hit while writing this:

  * **Only `def LinalgToLLVM` is the Python surface.** That table declares seven
    passes; the other six take options set by pipeline lambdas
    (`upperFrontier`, `skipVectorRowReduce`, `vtcmBudget`, ...) which must NOT
    have Python fields. Matching every `Option<` in the file yields ~26 false
    positives. A test that cries wolf gets ignored, which is worse than no test.
  * **Key on the first argument, never the flag name.** `disableLWPLoop`'s flag
    is `loop`, `LWPloopDepth`'s is `depth`, `tileSizes`'s is
    `tile-sizes-override`, and `enableMatmulToConv` is camelCase even as a flag.
  * **...but to find a knob in a lit test you need the flag, not the field.**
    Three tests set `vtcm-budget`, which is the *flag* of the field
    `vtcmBudgetBytes`. Searching for the field name finds nothing and would make a
    live, tested option look dead -- which is exactly the mistake that nearly
    deleted it.
  * The `.td` default is a *C++ source literal*, so `device_type`'s default is
    the text `"hexagon"` while the Python default is the string `hexagon`.
    Comparing raw text reports a phantom divergence; `normalise` unquotes the
    C++ layer too.
  * `vtcm-budget` is declared on **two different passes** (`VTCMTiling`, where it
    is live and set from `scratch`, and both HMX passes, where it is not). The
    same flag name on two passes is legal and is not checked here; the reason it
    is called out is that it is genuinely confusing.

Run: ../../.venv/bin/python -m pytest -q test/test_option_surface_agreement.py
"""

import ast
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PASSES_TD = BACKEND / "include" / "hexagon" / "Conversion" / "LinalgToLLVM" / "Passes.td"
OPTIONS_PY = BACKEND / "backend" / "hexagon_options.py"

PASS_BLOCK = re.compile(r"^def LinalgToLLVM\b.*?^\}", re.S | re.M)
# Four captured groups: field name, flag, type, default. The field is the C++
# struct member and is what the pipeline assigns; the flag is what a lit test
# writes. They are different strings and you need both -- searching for the field
# when you meant the flag (or the reverse) reports a live option as dead.
# The default is a C++ source literal and may itself contain escaped quotes,
# hence the inner character class.
OPTION = re.compile(
    r'Option<\s*"(\w+)"\s*,\s*"([^"]*)"\s*,\s*"([\w:<>]+)"\s*,'
    r'\s*/\*default=\*/\s*"((?:[^"\\]|\\.)*)"',
    re.S,
)

# --------------------------------------------------------------------------
# Frozen decisions. Changing a value here is a deliberate act: say what it is.
# --------------------------------------------------------------------------

# `LinalgToLLVM` pass options deliberately NOT exposed on the Python surface,
# with the `.td` default each one currently takes. Every entry needs a reason,
# and the test fails if the `.td` default moves without this file moving with it.
PINNED_DEFAULTS = {
    "puntBuffer": ("true", "upstream; no measurement of ours either way. This is a buffer-ownership "
                          "policy (alias the user buffer vs copy it local), not a profitability "
                          "knob, so exposing it would invite tuning noise."),
    "returnValueOptimization": ("true", "upstream; rewrites the IR into a shape the one-shot "
                                        "bufferizer folds. Turning it off breaks the bufferization "
                                        "contract rather than merely slowing things down, so it "
                                        "is not a knob."),
    "addFastMath": ("true", "upstream; fast-math flags are assertions about the input domain, not "
                            "hints. Worth noting this one has no Python field at all, so it is not "
                            "part of the audited option surface; it is pinned here because it is a "
                            "real decision point that happens to be invisible."),
    "splitTilingRange": ("true", "upstream tiling-range policy. No measurement of ours, and the "
                                   "Python surface already carries the tiling knobs that were "
                                   "measured, so this stays with its upstream value."),
    "expandBoolVec": ("true", "upstream lowering policy for i1 vectors. A lowering choice rather "
                              "than a schedule choice; exposing it would not buy a decision."),
    "enableHexagonRoutines": ("true", "upstream; controls linking the Hexagon builtin/routines "
                                          "library. This is a link-time fact of the target, not a "
                                          "tunable."),
    "extendPackParallelsOnly": ("true", "upstream. Note the asymmetry that makes this worth "
                                            "revisiting: the sibling knobs extendPackUpper and "
                                            "extendPackLowerFrontier ARE on the Python surface, so "
                                            "this one looks like an oversight rather than a "
                                            "decision."),
    "useInterchangeVector": ("false", "upstream; loop interchange for the vectorizer. Off upstream "
                                       "too, and the Python surface has no measured case for turning "
                                       "it on."),
    "enableSlicing": ("false", "upstream; opt-in slicing pass, off upstream as well. Its companion "
                               "slicingFactor is pinned below and is meaningless without it."),
    "slicingFactor": ("1", "only meaningful together with enableSlicing, which is off. Pinned so "
                           "that the pair cannot be half-changed."),
    "enableMatmulToConv": ("false", "upstream. This is the guard for the matmul-to-conv test path, "
                                     "and it has never had a Python field -- which is exactly why "
                                     "the matmul-to-conv + seed-layout block was unreachable from "
                                     "this backend."),
    "enableSeedLayoutConversions": ("false", "Became an orphan on 2026-09-30: the Python field was "
                                             "removed as a proven no-op, because it only does "
                                             "anything when enableMatmulToConv is on and that has no "
                                             "field either. The upstream pass option and the direct "
                                             "-matmul-to-conv path are untouched; this entry pins "
                                             "the value the pipeline now takes."),
    "LWPloopDepth": ("1", "upstream LWP loop depth. The Python surface exposes enableLWP and "
                           "disableLWPLoop but not the depth, which is an inconsistency rather than "
                           "a decision -- recorded as such so it is not mistaken for one."),
    "hexKLMode": (r'\"micro\"', "Became an orphan on 2026-09-30, the same shape as "
                              "enableSeedLayoutConversions above: the Python field was removed as a "
                              "proven no-op. Every `hexKLMode == \"macro\"` branch sits behind "
                              "enableHexKL, and LinalgToLLVMPass rejects that combination outright "
                              "(\"enableHexKL is incompatible with the HMX manifest contract\"), so "
                              "the branches were unreachable. Verified rather than argued: the only "
                              "caller that ever set the field, "
                              "test/python/torch-mlir/test_hexkl_macro_matmul.py, already fails on "
                              "exactly that error and is not in the host gate; and deleting the "
                              "field left the production FA kernel byte-identical (1131 "
                              "instructions, kernel md5 unchanged). The upstream pass option and the "
                              "direct -matmul-to-hexkl lit path are untouched. This entry pins the "
                              "value the pipeline now takes."),
}

# --------------------------------------------------------------------------
# The Hmx pass surface: a third tier, bridged by derived values, not by name.
# --------------------------------------------------------------------------

HMX_PASSES_TD = BACKEND / "include" / "hexagon" / "Dialect" / "Hmx" / "Transforms" / "Passes.td"
HMX_PIPELINE = BACKEND / "lib" / "Conversion" / "LinalgToLLVM" / "LinalgToLLVMPass.cpp"
HMX_TESTS = BACKEND / "test"

# Hmx pass options deliberately kept OFF the Triton pipeline, i.e. reachable only
# from the `linalg-hexagon-opt` command line. Each entry names the test that
# exercises it: if that test is deleted, the justification is gone and this file
# must be revisited, which `test_hmx_cli_only_entries_are_still_exercised` checks.
HMX_CLI_ONLY = {
    "vtcmBudgetBytes": ("0", "Not set by the Triton pipeline, on purpose: the device default is "
                             "what production wants, and WeightResidentPass.cpp already argues "
                             "that 'a third way to spell the pool size is one more thing that can "
                             "disagree with the device'. Kept because three lit tests use it to "
                             "pin budget-sensitive remarks and rejections. The name is the FIELD; "
                             "tests set the flag `vtcm-budget`.",
                        "matmul-to-hmx-budget.mlir"),
    "dropEncodings": ("true", "Not on the Python surface because the ON path is the "
                              "byte-identical default (it was split out precisely so the default "
                              "behaviour could not drift), and exposing a knob whose only "
                              "interesting value is 'off' invites regressions. Exercised by one "
                              "lit test.",
                              "matmul-to-hmx-keep-encoding.mlir"),
}


def hmx_options():
    """-> [(pass_name, field, flag, default)] for every Hmx pass option."""
    text = HMX_PASSES_TD.read_text(errors="replace")
    out = []
    for block in re.finditer(r"^def\s+(\w+)[^\n]*\n(.*?)^\}", text, re.S | re.M):
        pass_name, body = block.group(1), block.group(2)
        for field, flag, _typ, dflt in OPTION.findall(body):
            out.append((pass_name, field, flag, dflt))
    return out


def hmx_pipeline_sets(field):
    src = HMX_PIPELINE.read_text(errors="replace")
    return bool(re.search(r"[Oo]pts\." + field + r"\s*=", src))


def hmx_tests_setting(flag):
    hits = []
    for path in sorted(HMX_TESTS.rglob("*.mlir")):
        body = path.read_text(errors="replace")
        if re.search(r"\{[^}]*\b" + re.escape(flag) + r"\s*=", body):
            hits.append(path.name)
    return hits



def td_options():
    """-> {field: (type, default_source_text)} for `def LinalgToLLVM` only."""
    text = PASSES_TD.read_text(errors="replace")
    block = PASS_BLOCK.search(text)
    assert block, "could not find `def LinalgToLLVM` in Passes.td"
    return {name: (typ, dflt) for name, _flag, typ, dflt in OPTION.findall(block.group(0))}


def py_fields():
    """-> {field: default_source_text} for the HexagonOptions dataclass.

    Uses `ast`, not a regex: an earlier regex version could not read a
    parenthesised multi-line default, which cost six blind spots.
    """
    tree = ast.parse(OPTIONS_PY.read_text(errors="replace"))
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HexagonOptions"),
        None,
    )
    assert cls, "could not find the HexagonOptions class in hexagon_options.py"
    out = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = None if node.value is None else ast.unparse(node.value)
    return out


def normalise(text):
    """Reduce a C++ or Python literal to a comparable lower-case string.

    Two layers of quoting meet here: the TableGen default is C++ *source* (so
    `device_type`'s is the text `"hexagon"`), while the Python default is already
    a string value. Both are unwrapped.
    """
    if text is None:
        return None
    text = text.strip()
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text.lower()
    if isinstance(value, str):
        # A C++ literal may itself spell a quoted string; unwrap that layer too.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            return value[1:-1]
        return value
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).lower()


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_parsers_still_see_what_we_think_they_see():
    """Guard the parsers themselves.

    If either stops matching, every agreement test below goes green while
    checking nothing at all. These are floors with reasons, not decoration.
    """
    td = td_options()
    assert len(td) >= 49, f"only parsed {len(td)} LinalgToLLVM options; regex drifted"
    assert "upperFrontier" not in td, (
        "the pass block matched more than def LinalgToLLVM -- nested-pass options like "
        "upperFrontier are set by pipeline lambdas and must NOT be part of the Python surface"
    )
    py = py_fields()
    assert len(py) >= 58, f"only parsed {len(py)} python fields; the dataclass shrank or moved"


def test_every_option_is_wired_or_pinned():
    td = td_options()
    py = py_fields()
    undecided = sorted(set(td) - set(py) - set(PINNED_DEFAULTS))
    assert not undecided, (
        f"{len(undecided)} LinalgToLLVM option(s) have neither a Python field nor a pinned "
        f"entry:\n  " + "\n  ".join(undecided) +
        "\n\nDecide each one: expose it on HexagonOptions, or pin it in PINNED_DEFAULTS with a "
        "reason. An unpinned option silently takes its .td default, so the user cannot change it "
        "and nobody notices."
    )


def test_pinned_entries_all_exist():
    """A pinned name that is not an option is a stale entry and a silently useless
    one: the pinning stops guarding whatever it was written for."""
    td = td_options()
    stale = sorted(set(PINNED_DEFAULTS) - set(td))
    assert not stale, (
        f"PINNED_DEFAULTS names options that no longer exist: {stale}. Remove them, or the "
        "pinning stops guarding anything."
    )


def test_pinned_values_match_tablegen():
    td = td_options()
    wrong = [
        f"  {name}: pinned {PINNED_DEFAULTS[name][0]!r} but .td default is {td[name][1]!r}"
        for name in sorted(PINNED_DEFAULTS)
        if name in td and PINNED_DEFAULTS[name][0] != td[name][1]
    ]
    assert not wrong, (
        "a pinned value no longer matches its TableGen default:\n" + "\n".join(wrong) +
        "\n\nIf the .td default changed on purpose, update PINNED_DEFAULTS here and say so in its "
        "reason string. If it changed by accident, revert the .td change."
    )


def test_pinned_entries_have_reasons():
    thin = [name for name, (_v, reason) in PINNED_DEFAULTS.items() if len(reason) < 60]
    assert not thin, (
        f"these pinned entries have no real reason: {thin}. A pin without a reason is just a "
        "suppression; the reason is the part that keeps the next reader informed."
    )


def test_wired_defaults_agree():
    """Every wired option, with no exemption list.

    This is what found the stale claim in docs/codegen/switch-census.md §2, which
    asserted that `enableSplitReduction` and `enableConversionToFp16` default to
    true in TableGen and false in Python. Both `.td` defaults are `false`, so
    there is no divergence and the census was wrong.
    """
    td = td_options()
    py = py_fields()
    compared, diverged = 0, []
    for name in sorted(set(td) & set(py)):
        td_default = normalise('"' + td[name][1] + '"')
        py_default = normalise(py[name])
        assert py_default is not None, f"{name} has no python default to compare"
        compared += 1
        if td_default != py_default:
            diverged.append(f"  {name}: .td={td_default!r} python={py_default!r}")
    assert compared >= 36, (
        f"only compared {compared} of the wired options; the comparison is no longer covering "
        "everything and a blind spot is exactly what this file exists to remove"
    )
    assert not diverged, (
        "these wired options disagree between .td and the Python dataclass:\n"
        + "\n".join(diverged)
        + "\n\nOne of the two is lying about the effective default. Pick the intended value and "
        "change that side -- and if Python always passes the value explicitly, say so here, "
        "because that is a third state, not agreement."
    )


# --------------------------------------------------------------------------
# The Hmx pass surface
# --------------------------------------------------------------------------

def test_hmx_parser_sees_every_option():
    """Same reason as test_parsers_still_see_what_we_think_they_see: if the block
    regex stops matching, the coverage test below goes green while checking
    nothing."""
    opts = hmx_options()
    fields = {field for _p, field, _f, _d in opts}
    assert len(opts) >= 8, f"only found {len(opts)} Hmx options across {len(fields)} fields"
    # Seven are filled by the pipeline, two are CLI-only. If this number moves,
    # one of the two states changed and the classification below needs a human.
    assert len(fields) == 9, f"unexpected Hmx option field set: {sorted(fields)}"
    assert {p for p, _f, _fl, _d in opts} == {
        "MatmulToHmx", "HmxPartition", "WeightResident", "HmxVectorReadout",
    }, "the pass-block regex picked up a different set of passes"


def test_every_hmx_option_is_covered():
    """Every Hmx pass option is either filled by the pipeline or is a named
    CLI-only knob. Anything else is an option that silently takes its default,
    which is the failure this whole file exists to prevent.

    This is the check that nearly deleted a live option: `vtcmBudgetBytes` looks
    unset by the pipeline, and a search for the *field* name in the tests finds
    nothing. Searching by *flag* finds three tests that set `vtcm-budget`. It is
    a tested knob and it stays.
    """
    uncovered = []
    for _pass, field, _flag, _dflt in hmx_options():
        if hmx_pipeline_sets(field) or field in HMX_CLI_ONLY:
            continue
        uncovered.append(f"{field} (flag {_flag})")
    assert not uncovered, (
        "these Hmx pass options are neither set by the pipeline nor pinned as CLI-only:\n  "
        + "\n  ".join(uncovered)
        + "\n\nEach one silently takes its .td default. Either fill it in "
        "LinalgToLLVMPass.cpp, or add it to HMX_CLI_ONLY naming the test that justifies keeping "
        "it off the pipeline. If it is genuinely dead, delete the option instead -- do not leave "
        "a knob that cannot be turned."
    )


def test_hmx_pinned_values_match_tablegen():
    seen = {field: dflt for _p, field, _f, dflt in hmx_options()}
    wrong = [
        f"  {name}: pinned {HMX_CLI_ONLY[name][0]!r} but .td default is {seen[name]!r}"
        for name in sorted(HMX_CLI_ONLY)
        if name in seen and HMX_CLI_ONLY[name][0] != seen[name]
    ]
    assert not wrong, (
        "a pinned Hmx value no longer matches its TableGen default:\n" + "\n".join(wrong)
        + "\n\nIf the .td default moved on purpose, update HMX_CLI_ONLY and say so in the reason."
    )


def test_hmx_cli_only_entries_still_exist():
    seen = {field for _p, field, _f, _d in hmx_options()}
    stale = sorted(set(HMX_CLI_ONLY) - seen)
    assert not stale, (
        f"HMX_CLI_ONLY names options that no longer exist: {stale}. Remove them, or the pinning "
        "stops guarding anything."
    )
    thin = [n for n, (_v, reason, _t) in HMX_CLI_ONLY.items() if len(reason) < 60]
    assert not thin, f"these HMX_CLI_ONLY entries have no real reason: {thin}"


def test_hmx_cli_only_entries_are_still_exercised():
    """The anti-rot property: a CLI-only pin is justified by a test, so if that
    test stops setting the flag the pin has lost its reason and must be revisited.

    Without this, deleting the covering test would leave a knob nobody exercises
    and nobody can reach -- the exact state this file is about.
    """
    flag_of = {field: flag for _p, field, flag, _d in hmx_options()}
    unjustified = []
    for name, (_value, _reason, test_name) in sorted(HMX_CLI_ONLY.items()):
        if name not in flag_of:
            continue
        setting = hmx_tests_setting(flag_of[name])
        if test_name not in setting:
            unjustified.append(
                f"  {name}: claimed cover {test_name!r} does not set `{flag_of[name]}`. "
                f"Tests that do: {setting or 'NONE'}"
            )
    assert not unjustified, (
        "a CLI-only pin no longer has the test that justified it:\n" + "\n".join(unjustified)
        + "\n\nEither the test was renamed or removed. Re-derive the reason, or drop the option."
    )


def test_hmx_options_really_are_read_somewhere():
    """A pinned or pipeline-set option that nothing reads is a knob that changes
    nothing -- the cheapest dead code there is. This is a floor, not an
    exhaustive proof."""
    for _pass, field, _flag, _dflt in hmx_options():
        hits = 0
        for path in (BACKEND / "lib").rglob("*.cpp"):
            if re.search(r"\bthis->" + field + r"\b|\b" + field + r"\b", path.read_text(errors="replace")):
                hits += 1
        assert hits >= 1, f"{field} is declared but never mentioned in lib/**/*.cpp"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
