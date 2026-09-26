#!/usr/bin/env python3
"""Source contract: every runtime compile command must have a usable depfile.

Why this exists
---------------
The runtime headers carry the opt-in accounting structs *by value* between
translation units (`AccountingSnapshot`, `AccountingOwner`, the site-scope
state).  If one translation unit is not rebuilt after a header edit, the
process links two TUs that disagree about a struct layout.  Nothing crashes and
nothing warns: the capture just emits records that contradict each other.  That
was observed as a `VTCM_BUFFER_CACHE` ring count disagreeing with the
`VTCM_SNAPSHOT` from the same report, and it was initially misread as an
accounting-semantics bug.

The trap
--------
Adding `-MMD -MF <name>.d` and `DEPFILE <name>.d` is necessary but NOT
sufficient.  These rules run `cd`-ing into the runtime binary directory, so a
bare relative name makes the compiler write the depfile *next to the bitcode*,
while ninja resolves `depfile` against the *build root* and looks for a file
that is not there.  Ninja then reports `deps not found` for the edge and
silently drops every header dependency -- which reproduces the original bug
exactly, while the CMakeLists reads as though it were fixed.

So the contract is not "a DEPFILE exists" but "the `-MF` path and the `DEPFILE`
property are the same absolute path".  This checks that, per compile rule.
"""

from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
RUNTIME = HERE.parent
CMAKE = RUNTIME / "CMakeLists.txt"

# A rule that compiles something and therefore needs header dependency tracking.
COMPILES = re.compile(r"-emit-llvm|-c\s|hexagon-clang|\bAR\b\s+rcs")
# `-MF <path>`: where the compiler is told to write the depfile.
MF = re.compile(r"-MF\s+(\S+)")
# `DEPFILE <path>`: where the build system is told to read it from.
DEPFILE = re.compile(r"^\s*DEPFILE\s+(\S+)", re.MULTILINE)


def _custom_command_blocks(text: str) -> list[str]:
    """Return the body of every add_custom_command(...) call."""
    blocks: list[str] = []
    for match in re.finditer(r"add_custom_command\s*\(", text):
        depth = 0
        index = match.end() - 1
        while index < len(text):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        blocks.append(text[match.start() : index + 1])
    return blocks


def _expands_to_absolute(path: str, text: str) -> bool:
    """Resolve the ``${VAR}`` prefixes this file actually uses.

    A source-level check cannot just look for a leading slash: the HMX rule
    writes ``${HMX_OBJ}.d`` and that variable is *defined* as
    ``${CMAKE_CURRENT_BINARY_DIR}/...``, so it is already absolute.  What
    matters is the value after expansion, so expand the variables that appear
    in the path and require the result to be rooted.
    """
    seen: set[str] = set()

    def expand(value: str) -> str:
        # Two passes are enough for this file and keep the resolver honest
        # about what it understands rather than pretending to be CMake.
        for _ in range(4):
            match = re.search(r"\$\{(\w+)\}", value)
            if match is None:
                return value
            name = match.group(1)
            if name in seen:
                return value
            seen.add(name)
            if name == "CMAKE_CURRENT_BINARY_DIR":
                value = value.replace(match.group(0), "/abs/build")
                continue
            definition = re.search(
                rf"^\s*set\(\s*{name}\s+(\S+)", text, re.MULTILINE
            )
            if definition is None:
                return value
            value = value.replace(match.group(0), definition.group(1))
        return value

    return expand(path).startswith("/")


def test_every_compile_rule_has_a_matching_absolute_depfile() -> None:
    text = CMAKE.read_text(encoding="utf-8")
    checked = 0
    for block in _custom_command_blocks(text):
        if "OUTPUT" not in block or not COMPILES.search(block):
            continue
        checked += 1
        label = re.search(r"OUTPUT\s+(\S+)", block)
        name = label.group(1) if label else "<unknown>"

        mf = MF.search(block)
        assert mf is not None, f"{name}: compiles but passes no -MF depfile path"
        depfile = DEPFILE.search(block)
        assert depfile is not None, f"{name}: compiles but sets no DEPFILE"

        mf_path, depfile_path = mf.group(1), depfile.group(1)
        assert mf_path == depfile_path, (
            f"{name}: the compiler writes the depfile to {mf_path!r} but the "
            f"build system reads {depfile_path!r}; ninja resolves DEPFILE "
            f"against the build root, so a mismatched or relative path "
            f"silently drops every header dependency"
        )
        assert _expands_to_absolute(mf_path, text), (
            f"{name}: depfile path {mf_path!r} does not expand to an absolute "
            f"path, so it cannot be resolved from both the compiler's working "
            f"directory and the build root"
        )
    assert checked >= 2, f"expected the runtime bitcode and HMX rules, saw {checked}"


def test_the_bitcode_rule_roots_its_depfile_in_the_binary_dir() -> None:
    """The bitcode rule is the one that carried the live bug; pin it directly."""
    text = CMAKE.read_text(encoding="utf-8")
    bitcode = [
        block
        for block in _custom_command_blocks(text)
        if "OUTPUT ${BC}" in block
    ]
    assert len(bitcode) == 1, f"expected exactly one ${{BC}} rule, saw {len(bitcode)}"
    block = bitcode[0]
    assert "-MMD" in block, "the bitcode rule must ask for header dependencies"
    assert "-MF ${CMAKE_CURRENT_BINARY_DIR}/${BC}.d" in block, (
        "the bitcode depfile must be written through "
        "CMAKE_CURRENT_BINARY_DIR so the path is identical for the compiler and "
        "for ninja"
    )
    assert "DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${BC}.d" in block, (
        "the DEPFILE property must name the same absolute path as -MF"
    )


def test_the_stale_bitcode_bug_cannot_come_back() -> None:
    """Pin the exact shape that reproduced the bug, as a negative.

    A bare `${BC}.d` compiles and reconfigures cleanly, produces a depfile on
    disk, and still leaves ninja with no header dependencies.  Only the
    resolution path distinguishes it from the working form, so the check has to
    be on the spelling rather than on anything observable from the sources.
    """
    text = CMAKE.read_text(encoding="utf-8")
    assert not re.search(r"-MF\s+\$\{BC\}\.d", text), (
        "a bare ${BC}.d resolves against the rule's working directory while "
        "ninja resolves DEPFILE against the build root; this silently drops "
        "every header dependency and reintroduces the cross-TU ABI skew"
    )
    assert not re.search(r"DEPFILE\s+\$\{BC\}\.d", text), (
        "the same bare relative DEPFILE drops header dependencies"
    )


def main() -> int:
    tests = (
        test_every_compile_rule_has_a_matching_absolute_depfile,
        test_the_bitcode_rule_roots_its_depfile_in_the_binary_dir,
        test_the_stale_bitcode_bug_cannot_come_back,
    )
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"runtime depfile build contract: {len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
