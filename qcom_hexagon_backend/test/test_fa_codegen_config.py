#!/usr/bin/env python3
"""The host FA dump must describe the kernel the device actually runs.

`tools/hexmlir/dump_codegen.py` gained a `flash_attention` case on 2026-09-30 so
that the R1/R2 rowmax patterns could be checked on the host, without spending
device time to find out whether an A/B was worth running.

The *kernel* is imported from the test rather than copied, so it cannot drift.
The *shape constants* cannot be imported -- the test computes them at module
level from `NUM_THREADS` -- so they are mirrored by hand in `dump_codegen.py`.
That mirror is the one remaining copy, and this is what checks it.

The failure this prevents is specific and nasty: if the test is re-tuned (say
`NUM_THREADS` goes back to 4, which its own comment says was 2.2x slower) and the
host dump keeps the old constants, then every host number reported for FA
describes a kernel that does not exist. Nothing errors. The numbers are simply
about something else.

Run: ../../.venv/bin/python -m pytest -q test/test_fa_codegen_config.py
"""

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent              # .../hexagon-mlir/qcom_hexagon_backend
HEXMLIR = BACKEND.parent           # .../hexagon-mlir
WORKSPACE = HEXMLIR.parent         # .../hex-work
TOOL = WORKSPACE / "tools" / "hexmlir" / "dump_codegen.py"
FA_TEST = HEXMLIR / "test" / "python" / "triton" / "test_flash_attention.py"

# name -> the expression dump_codegen.py is expected to evaluate it to. Kept as
# source text so the check is "these two files say the same thing", not
# "import both and compare", which would not catch a stale literal in the tool.
MIRRORED = {
    "Z": "1", "H": "1", "N_CTX": "1024", "D_HEAD": "64",
    "BLOCK_N": "64", "BLOCK_DMODEL": "64", "STAGE": "1", "NUM_THREADS": "1",
}


def test_files_exist():
    assert TOOL.is_file(), f"missing {TOOL}"
    assert FA_TEST.is_file(), f"missing {FA_TEST}"


def test_fa_case_is_present_and_imports_the_kernel():
    """The whole point is that the kernel is not copied. If someone inlines it,
    this fails, because then there is a second source of truth again."""
    src = TOOL.read_text()
    assert 'if op == "flash_attention":' in src, "the FA case is gone from dump_codegen.py"
    assert "fa.attention_fwd_kernel" in src, (
        "the FA case no longer uses the imported kernel. A hand copy would be a second "
        "source of truth for the exact kernel under study."
    )
    assert "_import_fa_kernel" in src, "the import helper is gone"


def _int_assignments(path_or_src, tree=None):
    """-> {name: int} for module-level integer constants, both assignment forms.

    `X = 1` puts a Name in `targets[0]`; `X, Y = 1, 2` puts a single `Tuple`
    there. The FA test uses both (`Z, H, N_CTX, D_HEAD = 1, 1, 1024, 64` and
    `NUM_THREADS = 1`), so both have to be handled or the check silently sees
    only half the constants.
    """
    if tree is None:
        tree = ast.parse(Path(path_or_src).read_text())
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                names.extend(
                    e.id for e in target.elts if isinstance(e, ast.Name)
                )
        if isinstance(value, tuple):
            for name, item in zip(names, value):
                if isinstance(item, int):
                    out[name] = item
        elif isinstance(value, int) and len(names) == 1:
            out[names[0]] = value
    return out


def _module_constants(path):
    """Module-level integer constants from a Python file."""
    return _int_assignments(path)


def _fa_branch(tree):
    """The `if op == "flash_attention":` body, found structurally.

    Slicing the source text and re-parsing the slice is fragile -- it broke on
    indentation as soon as the branch grew a call. Walking the tree is simpler
    and stricter: a renamed op is simply not found.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "op"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "flash_attention"
        ):
            return node
    return None


def test_tool_constants_mirror_the_test():
    """The tool's mirrored literals must equal the test's module constants.

    `BLOCK_M` is derived (`N_CTX // NUM_THREADS`) in both files, so it is
    checked by deriving it rather than mirrored.
    """
    src = TOOL.read_text()
    tree = ast.parse(src)
    branch = _fa_branch(tree)
    assert branch is not None, (
        'no `if op == "flash_attention":` in dump_codegen.py -- the FA host dump is gone'
    )
    literals = _int_assignments(None, tree=branch)

    real = _module_constants(FA_TEST)
    wrong = []
    for name, expected in MIRRORED.items():
        if name not in real:
            wrong.append(f"{name}: the test no longer defines it at module level")
            continue
        if real[name] != int(expected):
            wrong.append(f"{name}: test has {real[name]!r}, dump_codegen.py mirrors {expected!r}")
        if literals.get(name) != int(expected):
            wrong.append(
                f"{name}: dump_codegen.py's FA branch has {literals.get(name)!r}, "
                f"expected {expected!r}"
            )
    assert not wrong, (
        "the host FA dump no longer describes the kernel under test:\n  "
        + "\n  ".join(wrong)
        + "\n\nHost numbers for FA would then describe a kernel the device never ran. "
        "Fix the constants in dump_codegen.py (or change this list if the test "
        "legitimately changed)."
    )


def test_block_m_is_derived_the_same_way():
    real = _module_constants(FA_TEST)
    expected_block_m = real["N_CTX"] // real["NUM_THREADS"]
    src = TOOL.read_text()
    assert "BLOCK_M = N_CTX // NUM_THREADS" in src, (
        "the FA branch must derive BLOCK_M the way the test does "
        f"(N_CTX // NUM_THREADS = {expected_block_m}), not mirror the number"
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
