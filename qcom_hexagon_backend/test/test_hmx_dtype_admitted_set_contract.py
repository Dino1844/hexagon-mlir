#!/usr/bin/env python3
"""The HMX admitted dtype set is spelled four times on this side of the ABI;
this binds the spellings.

Why this file exists
--------------------
The 2026-10-07 dual-track review found the engine's source-dtype admission set
({f16, f32} today) hand-written in five places with nothing binding them:

1. the ODS element-type constraints (HmxOps.td: the pack sources and the
   stage/await slots are `[F16, F32]`),
2. `dtype::isAdmittedFloat` (HmxDType.h),
3. the pack lowering's leaf dispatch -- a silent `srcIsF32 ? f32leaf :
   f16leaf` ternary scattered over four sites with no `else` guard,
4. the runtime leaf families (bin/runtime/hmx, one family per dtype),
5. the host prepack's dtype map (backend/hmx_weight_prepack.py).

HmxDType.h itself said "two hand-written copies, nothing binds them
mechanically" while claiming that extending to bf16 "only extends the
vocabulary". Following that instruction literally, a bf16 source would pass
the ODS constraint, reach the lowering, and be read by the **f16 leaf** --
wrong values, right exit code, no diagnostic. That is the same structural
trap as the stride-family bugs (pack `src_stride` / unpack `dst_stride`): the
leaf takes the number and walks.

The lowering is now a single-point dispatch (packActLeafFamily /
packWeightLeafFamily in HmxToLLVMPass.cpp) that fails the conversion on an
unadmitted type instead of falling back to f16. This file pins four
spellings so none of them can drift alone:

  * HmxOps.td: every multi-dtype element list (the pack sources, and the
    stage/await slots that carry the same rows across the DMA),
  * HmxDType.h: the dtypes `isAdmittedFloat` admits,
  * HmxToLLVMPass.cpp: the dtype branches of the two leaf-family tables,
  * hmx_weight_prepack.py: the keys of the host prepack's dtype map (the
    copy that spells the set in the contract's own "f16"/"f32" vocabulary).

It also pins the refactor itself: the dtype dispatch must stay single-point
(no dtype predicate outside the two tables, no `srcIsF32` marker, both
converters consulting both tables), and `test_simulated_drift_is_caught`
re-proves on every run that the comparison actually fires when one side is
drifted in memory.

Scope limits, each a trap hit or a decision made while writing this:

  * The runtime leaf families (spelling 4 in the list above) are NOT bound
    here: test_hmx_leaf_names_contract.py pins the emitter literals against
    the HMXAPI.h declarations, which is the binding that side can carry (one
    C function family per dtype; names are its set).
  * The unpack side needs no dispatch and is deliberately not checked: the op
    kind carries the dtype (hmx.unpack_acc reads out f16, hmx.unpack_acc_f32
    reads out f32; the ODS pins each op's destination to exactly one dtype),
    so there is no dtype -> leaf decision to make there.
  * Only multi-dtype lists count on the ODS side. Single-dtype lists
    (`[F16]` crouton arrays, `[F32]` f32 read-outs, `[I32]` status words) are
    different contracts -- the crouton element is `isCroutonElement`, not the
    admitted set -- so collecting them would compare the wrong thing.
  * The float vocabulary below (F16/BF16/F32/... and the Float8/4/6 ODS
    tokens) is what the parsers recognise. A dtype spelled outside it is
    invisible here -- but it is not invisible to the floors: an op that stops
    carrying a recognised multi-dtype list fails loudly, so a novel spelling
    cannot pass silently.

Run: python qcom_hexagon_backend/test/test_hmx_dtype_admitted_set_contract.py
     ../../.venv/bin/python -m pytest -q test/test_hmx_dtype_admitted_set_contract.py
"""

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
HMX_OPS_TD = BACKEND / "include" / "hexagon" / "Dialect" / "Hmx" / "IR" / "HmxOps.td"
HMX_DTYPE_H = BACKEND / "include" / "hexagon" / "Dialect" / "Hmx" / "IR" / "HmxDType.h"
HMX_TO_LLVM = BACKEND / "lib" / "Conversion" / "HmxToLLVM" / "HmxToLLVMPass.cpp"
PREPACK = BACKEND / "backend" / "hmx_weight_prepack.py"

# The ops whose operands spell the admitted set in ODS: the two pack sources
# and the two staging ops that carry the same source rows across the DMA. The
# set is exact on purpose -- a new op carrying a multi-dtype list must be
# brought here by a human who also checks its lowering dispatches it.
ADMITTED_OPS = frozenset({"PackAct", "PackWeight", "Stage", "Await"})

# ODS element-list containers: the inner bracket carries the dtype vocabulary
# (`Non0RankedMemRefOf<[F16, F32]>`, tensor+memref twins inside an AnyTypeOf).
ELEMENT_LIST = re.compile(r"\b(?:Non0Ranked)?(?:Tensor|MemRef)Of<\[([^\]]+)\]>")

# A TableGen float dtype token; canonicalised by lowercasing. The C++
# predicates below canonicalise to the same names.
ODS_FLOAT = re.compile(r"BF16|F16|F32|F64|F80|F128|Float[0-9][A-Za-z0-9]*")

# A Type predicate call in C++: `srcElem.isF16()`, `dtype::isF32(t)`. Used on
# dispatch bodies and on the whole pass file (the single-point check), so the
# spelling matters: a dtype decision written any other way is invisible to it
# -- the floors, not this regex, are the guard against that.
CPP_PREDICATE = re.compile(
    r"(?:\.|::)(isBF16|isBFloat16|isF16|isF32|isF64|isF80|isF128"
    r"|isFloat[0-9][A-Za-z0-9]*)\s*\("
)

ACT_SIG = "static FailureOr<PackActLeafFamily> packActLeafFamily(Operation *op,"
WT_SIG = "static FailureOr<PackWeightLeafFamily> packWeightLeafFamily(Operation *op,"


def canon_predicate(match: str) -> str:
    """`isF16` -> f16, `isBFloat16` -> bf16: the ODS tokens' lower-case names."""
    name = match.removeprefix("is").lower()
    return {"bfloat16": "bf16"}.get(name, name)


def _function_span(text: str, signature: str) -> tuple[int, int]:
    """The [start, end) span of the function with this definition signature.

    Brace-counted, so a body containing aggregate-initialiser braces (the
    family tables do) ends at its own closing brace, not the first one.
    """
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return start, index
    raise AssertionError(f"unterminated function body: {signature}")


def ods_admitted(text: str) -> set[str]:
    """The admitted set as ODS spells it, from the ops that carry it.

    Every multi-dtype element list in the file must (a) sit on one of the
    ADMITTED_OPS and (b) agree with every other one -- the pack sources and
    the stage/await slots are the same set, and one of them narrowing alone is
    exactly the drift this exists to catch.
    """
    by_op: dict[str, list[set[str]]] = {}
    for block in re.finditer(r"^def\s+Hmx_(\w+)Op\b(.*?)^\}", text, re.S | re.M):
        op, body = block.group(1), block.group(2)
        for inner in ELEMENT_LIST.finditer(body):
            tokens = [t.strip() for t in inner.group(1).split(",")]
            floats = {t.lower() for t in tokens if ODS_FLOAT.fullmatch(t)}
            if len(floats) >= 2:
                by_op.setdefault(op, []).append(floats)
    carriers = set(by_op)
    assert carriers == ADMITTED_OPS, (
        f"the ops carrying a multi-dtype element list are {sorted(carriers)}, "
        f"expected {sorted(ADMITTED_OPS)}. A new op joined the admitted set, "
        "or one left it: check its lowering dispatches the set, then update "
        "ADMITTED_OPS here"
    )
    lists = [s for sets in by_op.values() for s in sets]
    merged = set().union(*lists)
    assert all(s == merged for s in lists), (
        "the multi-dtype element lists disagree with each other: "
        f"{sorted(map(sorted, lists))} -- one operand narrowed or widened alone"
    )
    return merged


def dtype_admitted(text: str) -> set[str]:
    """The admitted set as `dtype::isAdmittedFloat` spells it."""
    match = re.search(r"inline bool isAdmittedFloat\(Type type\) \{([^{}]*)\}", text)
    assert match, "isAdmittedFloat not found (or grew nested braces) in HmxDType.h"
    found = {canon_predicate(m) for m in CPP_PREDICATE.findall(match.group(1))}
    assert found, "isAdmittedFloat admits nothing -- parser drift"
    return found


def dispatch_admitted(text: str) -> dict[str, set[str]]:
    """The admitted set as the two pack leaf-family tables spell it.

    Also floors the tables themselves: each must still name a full family
    (act: 3 forms x 2 dtypes; weight: 6 forms x 2 dtypes), so a gutted table
    cannot pass vacuously.
    """
    out: dict[str, set[str]] = {}
    for name, sig, min_names in (
        ("packActLeafFamily", ACT_SIG, 6),
        ("packWeightLeafFamily", WT_SIG, 12),
    ):
        start, end = _function_span(text, sig)
        body = text[start:end]
        found = {canon_predicate(m) for m in CPP_PREDICATE.findall(body)}
        assert found, f"{name} admits no dtype -- parser drift or a gutted table"
        names = re.findall(r"getPack\w+FnName", body)
        assert len(names) >= min_names, (
            f"{name} names only {len(names)} leaf functions (>= {min_names} "
            "expected): the family table lost a form"
        )
        out[name] = found
    return out


def prepack_admitted(text: str) -> set[str]:
    """The admitted set as the host prepack's dtype map keys spell it.

    The map is the copy that spells the set in the contract's own vocabulary
    ("f16"/"f32"); its keys must be exactly the admitted set, because the
    prepack builds the image the device leaf will read.
    """
    match = re.search(r"\{([^{}]+)\}\[desc\[\"dtype\"\]\]", text)
    assert match, (
        'the prepack dtype map ({...}[desc["dtype"]]) was not found in '
        "hmx_weight_prepack.py -- the lookup moved; update this parser with it"
    )
    keys = re.findall(r'"([a-z0-9]+)"\s*:', match.group(1))
    assert keys, "the prepack dtype map admits nothing -- parser drift"
    return set(keys)


def assert_agree(
    ods: set[str],
    dtype: set[str],
    dispatch: dict[str, set[str]],
    prepack: set[str],
) -> None:
    """All spellings of the admitted set are the same set."""
    sides = [
        ("HmxOps.td (pack/stage element lists)", ods),
        ("HmxDType.h (dtype::isAdmittedFloat)", dtype),
        ("hmx_weight_prepack.py (dtype map keys)", prepack),
    ] + [(f"HmxToLLVMPass.cpp ({name})", s) for name, s in sorted(dispatch.items())]
    reference = sides[0][1]
    assert all(s == reference for _n, s in sides), (
        "the admitted dtype set disagrees across its spellings:\n"
        + "\n".join(f"  {name}: {sorted(s)}" for name, s in sides)
        + "\n\nAll spellings must move together: dtype::isAdmittedFloat "
        "(HmxDType.h), the HmxOps.td source constraints, the "
        "packActLeafFamily/packWeightLeafFamily tables (HmxToLLVMPass.cpp), "
        "and the host prepack's dtype map (hmx_weight_prepack.py). The "
        "runtime leaf families (bin/runtime/hmx) are the one further "
        "spelling this test does not bind -- extend them in the same change."
    )


def check_agreement() -> None:
    assert_agree(
        ods_admitted(HMX_OPS_TD.read_text(encoding="utf-8")),
        dtype_admitted(HMX_DTYPE_H.read_text(encoding="utf-8")),
        dispatch_admitted(HMX_TO_LLVM.read_text(encoding="utf-8")),
        prepack_admitted(PREPACK.read_text(encoding="utf-8")),
    )


def check_dispatch_single_point() -> None:
    """The dtype -> leaf decision lives in exactly the two family tables.

    Every dtype predicate in the whole pass file must sit inside one of the
    two table bodies, the scattered ternary marker must stay gone, and both
    converters must still consult their table (definition + two call sites).
    A future dispatch written with a width comparison or a renamed marker is
    invisible to this -- floors, not proofs -- but the common regressions
    (reverting a call site, re-scattering the ternary) all fire.
    """
    text = HMX_TO_LLVM.read_text(encoding="utf-8")
    spans = [_function_span(text, ACT_SIG), _function_span(text, WT_SIG)]
    for m in CPP_PREDICATE.finditer(text):
        assert any(s <= m.start() < e for s, e in spans), (
            f"a dtype predicate ({m.group(0)}) appears outside the two pack "
            f"leaf-family tables at offset {m.start()}: the dtype -> leaf "
            "dispatch must stay single-point, because a second dispatch site "
            "is how the admitted set and the lowering drift apart silently"
        )
    assert "srcIsF32" not in text, (
        "the scattered srcIsF32 ternary is back: the pack dtype dispatch "
        "belongs in packActLeafFamily/packWeightLeafFamily, which fail loudly "
        "on unadmitted types instead of silently selecting the f16 leaf"
    )
    for name in ("packActLeafFamily", "packWeightLeafFamily"):
        count = text.count(name + "(")
        assert count >= 3, (
            f"{name} is referenced {count} times (>= 3 expected: its "
            "definition plus the tail and the bulk/single call sites) -- a "
            "converter stopped consulting the table"
        )


def _must_fire(what: str, fn) -> None:
    """`fn()` must raise AssertionError; anything else is a dead contract."""
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError(
        f"simulated drift did NOT fire the contract: {what}. Either the "
        "injection anchor moved (update it) or the comparison really lets "
        "this through (the contract is guarding nothing)"
    )


def check_simulated_drift_fires() -> None:
    """The self-proof, made permanent: each drift direction must fire.

    2026-10-07's fix instructions asked for a one-off demonstration that a
    bf16 added to one side alone turns the contract red. Doing it in memory
    (never on disk) on every run is strictly better: the proof cannot rot,
    and a parser change that accidentally blinds the comparison fails here
    instead of passing vacuously.
    """
    ods_text = HMX_OPS_TD.read_text(encoding="utf-8")
    dtype_text = HMX_DTYPE_H.read_text(encoding="utf-8")
    pass_text = HMX_TO_LLVM.read_text(encoding="utf-8")
    prepack_text = PREPACK.read_text(encoding="utf-8")
    ods_set = ods_admitted(ods_text)
    dtype_set = dtype_admitted(dtype_text)
    dispatch_sets = dispatch_admitted(pass_text)
    prepack_set = prepack_admitted(prepack_text)

    # ODS admits bf16; the dispatch does not follow.
    drifted = ods_text.replace("Of<[F16, F32]>", "Of<[F16, F32, BF16]>")
    assert drifted != ods_text, "the ODS injection anchor (Of<[F16, F32]>) moved"
    _must_fire(
        "ODS admits bf16, dispatch does not follow",
        lambda: assert_agree(ods_admitted(drifted), dtype_set, dispatch_sets,
                             prepack_set),
    )

    # HmxDType.h admits bf16; the ODS constraints do not follow.
    drifted = dtype_text.replace(
        "return type.isF16() || type.isF32();",
        "return type.isBF16() || type.isF16() || type.isF32();",
    )
    assert drifted != dtype_text, (
        "the isAdmittedFloat injection anchor (return type.isF16() || "
        "type.isF32();) moved"
    )
    _must_fire(
        "HmxDType.h admits bf16, ODS does not follow",
        lambda: assert_agree(ods_set, dtype_admitted(drifted), dispatch_sets,
                             prepack_set),
    )

    # The dispatch adds a bf16 branch; neither spelling follows.
    anchor = "packActLeafFamily(Operation *op,"
    opening = pass_text.index("{", pass_text.index(anchor))
    drifted = (
        pass_text[: opening + 1]
        + "\n  if (srcElem.isBF16())\n    return PackActLeafFamily{};"
        + pass_text[opening + 1 :]
    )
    _must_fire(
        "dispatch adds a bf16 branch, ODS does not follow",
        lambda: assert_agree(ods_set, dtype_set, dispatch_admitted(drifted),
                             prepack_set),
    )

    # The host prepack admits bf16; the compiler spellings do not follow.
    drifted = prepack_text.replace(
        '{"f16": np.float16, "f32": np.float32}',
        '{"bf16": None, "f16": np.float16, "f32": np.float32}',
    )
    assert drifted != prepack_text, (
        'the prepack injection anchor ({"f16": np.float16, "f32": '
        "np.float32}) moved"
    )
    _must_fire(
        "prepack admits bf16, compiler spellings do not follow",
        lambda: assert_agree(ods_set, dtype_set, dispatch_sets,
                             prepack_admitted(drifted)),
    )


def main() -> None:
    check_agreement()
    check_dispatch_single_point()
    check_simulated_drift_fires()
    print("HMX dtype admitted-set contract: PASS (four spellings agree)")


def test_admitted_set_agrees_across_four_sides() -> None:
    check_agreement()


def test_dispatch_is_single_point() -> None:
    check_dispatch_single_point()


def test_simulated_drift_is_caught() -> None:
    check_simulated_drift_fires()


if __name__ == "__main__":
    main()
