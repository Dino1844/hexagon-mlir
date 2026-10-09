#!/usr/bin/env python3
"""The leaf signature table, HMXAPI.h and the emitted names must agree.

Three files describe the ABI of the 29 leaves `libhmxapi.a` exports, and
nothing in the compiler checks them against each other:

  * ``bin/runtime/hmx/include/HMXAPI.h`` -- what the runtime declares. This is
    the authority: the SDK's clang compiles it and the kernel links to it.
  * ``lib/Conversion/HmxToLLVM/HmxExternalFnNames.cpp`` -- the symbol the
    lowering emits for an hmx op.
  * ``include/hexagon/Conversion/HmxToLLVM/HmxLeafSignatures.h`` -- how many
    arguments the lowering passes and of what type.

`test_hmx_leaf_names_contract.py` already binds the first two **by name**. It
cannot see arity or types, and before this file existed neither could anything
else: every lowering site in ``HmxToLLVMPass.cpp`` hand-wrote its own
``SmallVector<Type>(9, i32Ty)``, so a signature edit in HMXAPI.h was a silent
ABI mismatch. It compiles, it links, it passes a name-only contract, and on
the device it reads arguments out of the wrong registers.

What this asserts, and why each part is here:

  * The table and HMXAPI.h list the same leaf names **with the same parameter
    types, position by position**. This is the gate; both directions are
    checked, and the message names the leaf and shows both spellings, because
    a check that reports "some set differs" costs a grep to act on.
  * The table and ``HmxExternalFnNames.cpp`` list the same names. The third
    leg: a row for a leaf nobody emits is dead ABI, and an emitted name with
    no row would make ``getVoidLeaf`` fail at lowering time instead of here.
  * Every C spelling in play is one ``HmxLeafSignature::argTys`` translates.
    The translation table is a two-line ``if`` in the C++, so this file keeps
    its own copy of the known set *and says where the other copy lives*; a new
    spelling that only reaches the table passes the checks above and then
    fails on the device, which is exactly the failure mode being removed.
  * Floors on what each parser saw. A parser that silently stops matching
    turns every agreement check green while comparing nothing.

Scope limits, each of them a trap:

  * **Only `void hmx_*` declarations count**, and a leaf may declare its empty
    parameter list as `(void)`. Parsing `(void)` as one parameter of type
    `void` would report a phantom divergence against the table's `""`.
  * **Parameter types are compared as spelled, not normalised.** `unsigned`
    and `unsigned int` are the same C type and different table entries; if one
    side ever spells the other, the right fix is to make the spelling match,
    because the whole value of this file is that it can be diffed against the
    header by eye.
  * **`HmxExternalFnNames.cpp` also carries the two DMA staging names**
    (``hmx.stage`` / ``hmx.await`` forward to ``hexagon_runtime_dma_*``).
    They are not in HMXAPI.h and not in the table -- DMAToLLVMPass owns their
    signature -- so this file only ever compares the ``hmx_``-prefixed subset.

Run: ../../.venv/bin/python -m pytest -q test/test_hmx_leaf_signature_agreement.py
"""

import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
TABLE = (
    BACKEND / "include" / "hexagon" / "Conversion" / "HmxToLLVM" / "HmxLeafSignatures.h"
)
DECLARATIONS = BACKEND / "bin" / "runtime" / "hmx" / "include" / "HMXAPI.h"
EMITTER = BACKEND / "lib" / "Conversion" / "HmxToLLVM" / "HmxExternalFnNames.cpp"

# One table row: a name literal, a comma, then one or more parameter-type
# literals. The type list is written as adjacent string literals when it does
# not fit on one line, so the type group is `(...)+` and the pieces are joined
# below -- a group that stopped at the first closing quote would compare half
# a signature and pass.
ROW = re.compile(r'\{\s*"(hmx_[A-Za-z0-9_]+)"\s*,\s*((?:\s*"[^"]*"\s*)+)\}')
LITERAL = re.compile(r'"([^"]*)"')

# `void hmx_pack_act_f16(unsigned dst_addr, ...)` -- possibly across lines,
# terminated by `);`. Names carry `[A-Z]` because of the `_T` twins.
DECL = re.compile(r"^void\s+(hmx_[A-Za-z0-9_]+)\s*\((.*?)\)\s*;", re.M | re.S)

#: The C spellings `HmxLeafSignature::argTys` (HmxLeafSignatures.h)
#: translates to an MLIR type. The other copy of this set is that function's
#: `if` chain; when it grows, this set grows in the same edit or the check
#: below fails by spelling.
ARG_TYS_SPELLINGS = {"unsigned"}

#: What every parameter of every leaf is today. Not a design assertion -- a
#: canary: HMXAPI.h's header comment promises "no pointer-typed or _Float16 *
#: arguments anywhere", and a new spelling is the signal to go read that
#: promise before widening the table.
FLOOR_LEAVES = 29


def table_rows() -> "dict[str, list[str]]":
    """The signature table: leaf name -> parameter type spellings, in order."""
    text = TABLE.read_text(encoding="utf-8")
    rows = {}
    for name, types_group in ROW.findall(text):
        assert name not in rows, f"duplicate row for {name}"
        spellings = "".join(LITERAL.findall(types_group)).split(",")
        rows[name] = [] if spellings == [""] else spellings
    return rows


def header_rows() -> "dict[str, list[str]]":
    """HMXAPI.h: leaf name -> parameter type spellings, in declaration order."""
    rows = {}
    for name, params in DECL.findall(DECLARATIONS.read_text(encoding="utf-8")):
        params = " ".join(params.split())
        # `(void)` is an empty parameter list, not one parameter of type void.
        spellings = [] if params in ("", "void") else params.split(",")
        # Each parameter is `<type> <name>`; the type may be multi-word, the
        # name never is, so everything but the last token is the type.
        rows[name] = [" ".join(p.split()[:-1]) for p in spellings]
    return rows


def emitted_names() -> "set[str]":
    """The leaf symbols HmxExternalFnNames.cpp returns (hmx_-prefixed only)."""
    return set(
        re.findall(r'"(hmx_[A-Za-z0-9_]+)"', EMITTER.read_text(encoding="utf-8"))
    )


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_parsers_still_see_what_we_think_they_see():
    """Guard the parsers themselves.

    If either regex stops matching, every agreement test below goes green
    while checking nothing at all. Floors with reasons, not decoration.
    """
    table = table_rows()
    header = header_rows()
    emitted = emitted_names()
    assert len(table) >= FLOOR_LEAVES, (
        f"parsed {len(table)} rows from HmxLeafSignatures.h; the ROW regex "
        "stopped matching and every check below is now vacuous"
    )
    assert len(header) >= FLOOR_LEAVES, (
        f"parsed {len(header)} declarations from HMXAPI.h; the DECL regex "
        "stopped matching (it needs `[A-Z]` in the name class for the `_T` "
        "twins) and every check below is now vacuous"
    )
    assert len(emitted) >= FLOOR_LEAVES, (
        f"parsed {len(emitted)} names from HmxExternalFnNames.cpp"
    )


def test_table_and_header_agree_parameter_by_parameter():
    """The gate: the transcription and the header say the same thing.

    Both directions, because either one alone misses half the drift -- a row
    for a deleted leaf is as dead as an undeclared leaf that still gets
    called.
    """
    table = table_rows()
    header = header_rows()

    only_in_table = sorted(set(table) - set(header))
    assert not only_in_table, (
        "HmxLeafSignatures.h declares leaves HMXAPI.h does not: "
        f"{only_in_table}. A row for a symbol the runtime does not export is "
        "dead ABI; delete the row, or add the declaration to HMXAPI.h."
    )
    only_in_header = sorted(set(header) - set(table))
    assert not only_in_header, (
        "HMXAPI.h declares leaves HmxLeafSignatures.h has no row for: "
        f"{only_in_header}. The lowering will hand-write this leaf's argument "
        "types again -- add the row (declaration order) instead."
    )

    diverged = [
        f"  {name}: table {table[name]!r} but HMXAPI.h {header[name]!r}"
        for name in sorted(table)
        if name in header and table[name] != header[name]
    ]
    assert not diverged, (
        "these leaves are transcribed with the wrong parameter types:\n"
        + "\n".join(diverged)
        + "\n\nHMXAPI.h is the authority (libhmxapi.a is compiled from it). "
        "If the header changed on purpose, change the row in the same edit; "
        "if the row changed on purpose, say why in HmxLeafSignatures.h -- "
        "the lowering passes whatever the row says, so a wrong row is a "
        "silently wrong call, not a compile error."
    )


def test_table_names_match_the_emitter():
    """The third leg: every row is a symbol the lowering can actually emit.

    Without this, a renamed getter in HmxExternalFnNames.cpp would leave the
    table describing a leaf that no longer exists, and `getVoidLeaf` would
    start failing at lowering time rather than at the gate.
    """
    table = table_rows()
    emitted = emitted_names()

    unemitted = sorted(set(table) - emitted)
    assert not unemitted, (
        "the table describes leaves HmxExternalFnNames.cpp never emits: "
        f"{unemitted}. Nothing will ever lower to them, so the row is dead "
        "and the name/arity pair it claims to pin is unverified."
    )
    missing = sorted(emitted - set(table))
    assert not missing, (
        "the lowering emits leaves the table has no row for: "
        f"{missing}. getVoidLeaf will fail at lowering time; add the row "
        "with the parameter types HMXAPI.h declares."
    )


def test_every_spelling_is_one_arg_tys_translates():
    """Every C type in play must be one `argTys` maps to an MLIR type.

    The translation lives in one `if` in HmxLeafSignatures.h. This check
    keeps the two copies of its known set together: a spelling that reaches
    the table but not the `if` fails every lowering with a bare `failure()`,
    and a spelling that reaches HMXAPI.h but not the table is caught above.
    """
    spellings = set()
    for types in table_rows().values():
        spellings |= set(types)
    for types in header_rows().values():
        spellings |= set(types)
    unknown = sorted(spellings - ARG_TYS_SPELLINGS)
    assert not unknown, (
        f"C parameter spellings neither table nor gate knows how to place: "
        f"{unknown}.\n\nExtend HmxLeafSignature::argTys "
        "(HmxLeafSignatures.h) with the MLIR type it means and add the "
        "spelling to ARG_TYS_SPELLINGS here. Until both are done the lowering "
        "cannot build this leaf's call at all."
    )
    assert spellings, "no spellings parsed; the comparison above was vacuous"


def test_table_is_in_header_declaration_order():
    """Rows appear in HMXAPI.h's declaration order.

    Order carries no semantics -- it is the property that makes the two files
    diffable by eye, which is how a human reviews a signature change before
    the gate ever runs. Losing it quietly turns this file into the only
    signal.
    """
    table_order = list(table_rows())
    header_order = list(header_rows())
    if table_order == header_order:
        return
    first = next(
        (i for i, (a, b) in enumerate(zip(table_order, header_order)) if a != b),
        min(len(table_order), len(header_order)),
    )
    raise AssertionError(
        "HmxLeafSignatures.h is not in HMXAPI.h's declaration order; the "
        f"first difference is at row {first}:\n"
        f"  table:  {table_order[first: first + 3]}\n"
        f"  header: {header_order[first: first + 3]}\n"
        "Reorder the rows -- the by-eye diff is part of the contract."
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
