#!/usr/bin/env python3
"""Every steering constant in the Hmx backend must carry a traceability triple.

The project's rule (`hexagon-mlir/ROADMAP.md` section 2.1a) is that a constant
which can change a decision needs three things written down: the *mechanism*,
the *measurement*, and the *shape set* -- plus whether that shape set actually
represents a real workload. A constant that cannot supply a measurement must say
so and say why; that is a legitimate answer, silence is not.

This is what stops the category from quietly regrowing. Without it the rule is a
paragraph nobody re-reads; with it, adding a threshold without saying why is a
test failure.

Deliberate limits, each of which exists because the obvious alternative was tried
and was wrong:

  * It checks the *presence and shape* of the block, not the truth of its
    contents. A dishonest triple passes. Only review catches that, and
    pretending otherwise would be its own kind of lie.
  * Wire vocabulary (attribute names, reason strings) is out of scope *by
    construction*, not by exemption list. An earlier version matched any
    `constexpr` and demanded a profitability triple for `kResidentAttr` and
    friends; 206 of 207 hits were that noise. Widening the exemption list to
    absorb it would have treated the symptom.
  * `NOT-A-DECISION` is honoured only on a constant's OWN adjacent block, and
    only after the block is known to exist. A group block that happened to
    mention the marker must not silently exempt everything it names.
  * A group block must be adjacent to the first constant it names, and every
    name on a `TRACEABILITY:` line must be a constant that actually exists --
    otherwise renaming a constant turns its traceability block into a comment
    that guards nothing, silently.

Excluded on purpose: `bin/runtime/` (device-side C, mirrored constants pinned by
its own contract test) and the Triton/LLVM shims. This gate is about the
compiler's decisions.

Run: ../../.venv/bin/python -m pytest -q test_hmx_constant_traceability.py
"""

import re
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]

# Scope is by ROLE, not convenience: the code that decides whether HMX runs, and
# the header that holds those decisions.
#
# IN: the Hmx dialect's transforms, its LLVM conversion, and `HmxTarget.h`.
#     Those are where a profitability or capability decision is made.
# OUT, with reasons -- an earlier version scanned all of `lib/` and was wrong
#     twice over: it demanded triples from upstream files we do not own, and
#     from headers whose entire job is to be a wire contract.
EXCLUDED = {
    # The ABI wire vocabulary. These values are chosen for wire stability
    # across host/device and across builds, not for profitability, so a
    # "mechanism + calibration" triple would be nonsense for them.
    "include/hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h",
    # The v3 record schema: version numbers and field identities.
    "include/hexagon/Dialect/Hmx/Transforms/HmxRecordV3.h",
    # The device layout contract: one crouton is 16 pairs x 32 columns x 2
    # halves because that is what the hardware does, cross-checked against
    # bin/runtime/hmx/include/HMXAPI.h. Not tunable, so not a decision.
    "include/hexagon/Dialect/Hmx/IR/HmxCroutonLayout.h",
}
ROOTS = [
    BACKEND / "lib" / "Dialect" / "Hmx",
    BACKEND / "lib" / "Conversion" / "HmxToLLVM",
    BACKEND / "include" / "hexagon" / "Dialect" / "Hmx",
]
SUFFIXES = {".cpp", ".h"}


def _excluded(path):
    try:
        rel = str(path.relative_to(BACKEND))
    except ValueError:
        return True
    return rel in EXCLUDED


def sources():
    for root in ROOTS:
        for path in sorted(root.rglob("*")):
            if path.suffix in SUFFIXES and path.is_file() and not _excluded(path):
                yield path


def numeric_type(token):
    return bool(
        re.fullmatch(
            r"(?:u?int(?:8|16|32|64)_t|unsigned(?:\s+int)?|int|size_t|long"
            r"|long\s+long|float|double)",
            token,
        )
    )


DECL = re.compile(
    r"(?:static\s+|inline\s+|static\s+inline\s+|inline\s+static\s+|constexpr\s+)*"
    r"constexpr\s+(?P<type>\w+(?:\s+\w+)?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"(?:\{\s*|=\s*)",
)


def parse_decls(text):
    """[(name, start, type)] for every `constexpr <number> name =` in `text`."""
    out = []
    for m in DECL.finditer(text):
        if numeric_type(m.group("type").strip()):
            out.append((m.group("name"), m.start(), m.group("type").strip()))
    return out


def preceding_comment(text, pos):
    """The contiguous `//` comment block immediately above `pos`, else ''."""
    lines = text[:pos].split("\n")
    block = []
    for line in reversed(lines[:-1] if lines and not lines[-1].strip() else lines):
        stripped = line.strip()
        if stripped.startswith("//"):
            block.append(stripped)
        elif not stripped:
            continue
        else:
            break
    return "\n".join(reversed(block))


FIELDS = ("mechanism", "measurement", "shape set", "workload representativeness")
TRACE_LINE = re.compile(r"TRACEABILITY:\s*([A-Za-z_][\w\s]*)$", re.MULTILINE)
EXEMPT = "NOT-A-DECISION"

# A traceability block must introduce the constant, so the declaration
# immediately follows it with no blank line between.
BLOCK = re.compile(
    r"(?P<comment>(?:^[ \t]*//.*\n)+)"
    r"[ \t]*(?:static\s+|inline\s+|static\s+inline\s+|inline\s+static\s+)*"
    r"constexpr\s+(?P<type>\w+(?:\s+\w+)?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?:\{\s*|=\s*)",
    re.MULTILINE,
)


def scan():
    """(path, name, comment) for every numeric constexpr in scope."""
    out = []
    for path in sources():
        text = path.read_text(errors="replace")
        for name, start, _ in parse_decls(text):
            out.append((path, name, preceding_comment(text, start)))
    return out


CONSTANTS = scan()

# name -> (path, comment) for blocks that are adjacent to their first constant.
BLOCKS = []
for _path in sources():
    _text = _path.read_text(errors="replace")
    for _m in BLOCK.finditer(_text):
        if numeric_type(_m.group("type").strip()):
            BLOCKS.append((_path, _m.group("name"), _m.group("comment")))

BLOCK_BY_NAME = {name: (p, c) for p, name, c in BLOCKS}
KNOWN = {name for _, name, _ in CONSTANTS}
GROUP_OWNERS = {}  # member name -> (path, comment) of the group block
for _p, _owner, _c in BLOCKS:
    m = TRACE_LINE.search(_c)
    if m:
        for member in m.group(1).split():
            GROUP_OWNERS.setdefault(member, (_p, _c))


def comment_for(path, name, own):
    """The block that governs this constant: its own, or its group's."""
    if TRACE_LINE.search(own) or EXEMPT in own:
        return own
    return GROUP_OWNERS.get(name, ("", ""))[1]


def missing_fields(body):
    lowered = body.lower()
    return [f for f in FIELDS if f"{f}:" not in lowered]


def test_discovery_is_not_vacuous():
    """If the pattern stops matching, every other test here goes green while
    checking nothing. A gate that checks nothing is worse than no gate."""
    assert len(CONSTANTS) >= 15, (
        f"only found {len(CONSTANTS)} numeric constexprs in scope (expected >= 15); "
        "the DECL regex has probably stopped matching"
    )
    files = {p for p, _, _ in CONSTANTS}
    assert len(files) >= 7, f"only {len(files)} files scanned: {sorted(map(str, files))}"
    # Pin names, not just a count: a rename must fail loudly here rather than
    # quietly shrink coverage while the count still looks plausible.
    for expected in (
        "kStageMinKTiles",   # the profitability floor this whole exercise is about
        "kNodeBudget",       # a size guard
        "kMaxI64",           # a type bound
        "minRows",           # a capability threshold
        "defaultVtcmBudget",  # a capacity
        "tileEdge",          # a device fact
    ):
        assert expected in KNOWN, f"lost sight of {expected}; KNOWN={sorted(KNOWN)}"


# The declaration pattern is the whole gate, so it is tested directly rather
# than inferred from the corpus. An earlier version "verified" inline coverage
# by pointing at HmxManifest.h -- which has 100+ `inline constexpr` but every
# one is a StringLiteral, so it proved nothing while looking like a check.
# A regex should be tested where it is used: on the spellings it must accept
# and the ones it must not.
MUST_MATCH = [
    "constexpr int64_t kA = 8;",
    "static constexpr int64_t kB = 8;",
    "inline constexpr int64_t kC = 8;",
    "static inline constexpr int64_t kD = 8;",
    "inline static constexpr unsigned kE = 8;",
    "static constexpr int kF{8};",
    "constexpr size_t kG = 8;",
    "constexpr long long kH = 8;",
    "constexpr uint32_t kI = 8;",
    "constexpr double kJ = 8;",
]
MUST_NOT_MATCH = [
    # Wire vocabulary, not decisions. `StringLiteral` is not a numeric type.
    'inline constexpr StringLiteral kA = "x";',
    'constexpr StringLiteral kB = "y";',
    # Not a compile-time constant at all.
    "const int kC = 8;",
    "int kD = 8;",
    # A function, not a constant.
    "constexpr int64_t f(int64_t x) { return x; }",
]


@pytest.mark.parametrize("decl", MUST_MATCH)
def test_declaration_pattern_accepts_known_spellings(decl):
    found = parse_decls(decl)
    assert len(found) == 1, f"pattern missed a real spelling: {decl!r} -> {found}"


@pytest.mark.parametrize("decl", MUST_NOT_MATCH)
def test_declaration_pattern_rejects_non_decisions(decl):
    found = parse_decls(decl)
    assert not found, f"pattern matched something that is not a decision: {decl!r}"


def test_no_stale_traceability_names():
    """Every name on a TRACEABILITY line must be a real constant.

    Otherwise renaming a constant leaves behind a block that guards nothing
    and reads as if it still does.
    """
    stale = []
    for path, _owner, comment in BLOCKS:
        m = TRACE_LINE.search(comment)
        if not m:
            continue
        for member in m.group(1).split():
            if member not in KNOWN:
                stale.append(f"{path.relative_to(BACKEND)}: {member}")
    assert not stale, (
        "TRACEABILITY block(s) name constants that do not exist -- the block "
        f"now guards nothing:\n  " + "\n  ".join(stale)
    )


@pytest.mark.parametrize(
    "path,name,own",
    CONSTANTS,
    ids=[f"{p.relative_to(BACKEND)}:{n}" for p, n, _ in CONSTANTS],
)
def test_constant_is_traceable_or_explicitly_exempt(path, name, own):
    comment = comment_for(path, name, own)
    rel = path.relative_to(BACKEND)

    # A block must exist before an exemption can be considered, so a group
    # block that merely mentions the marker cannot exempt its members.
    if not TRACE_LINE.search(comment) and EXEMPT not in comment:
        pytest.fail(
            f"{rel}: `{name}` is a numeric constexpr with no traceability "
            f"triple.\n"
            f"  Put a comment block IMMEDIATELY above it (no blank line):\n"
            f"    // TRACEABILITY: {name}\n"
            f"    //   mechanism: ...\n"
            f"    //   measurement: ...   (or: none, and why none is owed)\n"
            f"    //   shape set: ...\n"
            f"    //   workload representativeness: ...\n"
            f"  A block may name several constants to document a group.\n"
            f"  If it is a hardware or interface fact rather than a decision,\n"
            f"  write `// {EXEMPT} <reason>` on its own block instead."
        )

    if EXEMPT in comment and not TRACE_LINE.search(comment):
        line = next(l for l in comment.splitlines() if EXEMPT in l)
        reason = line.split(EXEMPT, 1)[1].strip()
        assert reason, f"{rel}: `{name}` says {EXEMPT} with no reason"
        return

    missing = missing_fields(comment)
    assert not missing, (
        f"{rel}: traceability block for `{name}` is missing {missing}.\n"
        f"  'measurement: none' and 'shape set: n/a' are valid answers, but\n"
        f"  they must be written down -- silence is what this test exists for.\n"
        + "\n".join("    " + l for l in comment.splitlines()[:16])
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
