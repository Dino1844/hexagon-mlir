#!/usr/bin/env python3
"""The query facade is worth nothing if it answers a different number.

Phase 1 of the HmxTarget query facade (operator-parity ticket 15) moved the
planning facts the passes each read on their own -- the vtcm-budget
resolution, the crouton tile counts, the crouton byte terms, the row floor --
into named query methods on `HmxTarget.h`. It migrated no consumer: the
passes still spell the same arithmetic in place. That leaves two spellings
of each fact, and this file is what holds them at the same value until
Phase 2 retires one of them.

Three things are pinned, per query:

  * the query's own body, extracted from `HmxTarget.h` and evaluated;
  * the consuming pass's spelling, extracted from its source and evaluated;
  * the frozen values the IR side observes -- the CHECK literals of
    `test/Dialect/Hmx/Transforms/hmx-target-query-equality.mlir` and
    `hmx-target-query-refusals.mlir`, which run the real pass on this
    build. Those literals are read from the lit files, never restated here:
    a third copy of the same number is the disease this work set out to
    cure, and a test file is not exempt.

Why evaluate instead of comparing text: the two spellings are deliberately
different text (`contract.m * contract.k * inBytes` and
`activationBytes(rows, k)` name the same product), so an equality of
*values* over a table of inputs is the only thing that can actually fail.

One spelling deserves its own note. `MatmulToHmxPass` writes the read-out
bytes as `contract.m * contract.n * 2` with a bare `2`, where the facade
says `croutonElemBytes`. They are equal today because the engine's read-out
is fp16, which is exactly what `readoutBytes` documents. Pinning them binds
that coincidence to its reason: if the crouton element ever stopped being
2 bytes, this test fails and the site must be fixed rather than the number
adjusted. (The same treatment `test_crouton_size_agreement.py` gives the
layout header's bare literals.)
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
_LIB = _BACKEND / "lib" / "Dialect" / "Hmx" / "Transforms"
_INCLUDE = _BACKEND / "include" / "hexagon" / "Dialect" / "Hmx"

TARGET_HEADER = _INCLUDE / "Transforms" / "HmxTarget.h"
CRoUTON_LAYOUT = _INCLUDE / "IR" / "HmxCroutonLayout.h"
MATMUL_PASS = _LIB / "MatmulToHmxPass.cpp"
PARTITION_PASS = _LIB / "HmxPartitionPass.cpp"
WEIGHT_PASS = _LIB / "WeightResidentPass.cpp"
THREAD_PASS = _LIB / "ThreadRolePartition.cpp"
MANIFEST = _LIB / "HmxManifest.cpp"
EQUIVALENCE_LIT = _HERE / "Dialect" / "Hmx" / "Transforms" / "hmx-target-query-equality.mlir"
REFUSALS_LIT = _HERE / "Dialect" / "Hmx" / "Transforms" / "hmx-target-query-refusals.mlir"

#: The frozen facts the queries are expressed in. Written out rather than
#: derived, on purpose (the same contract `test_hmx_manifest_minimum_rows.py`
#: uses): editing the header and the consumer together still fails until the
#: boundary is moved here, where the change is visible.
TILE_EDGE = 32
CROUTON_ELEM_BYTES = 2
MIN_ROWS = 4
DEFAULT_VTCM_BUDGET = 8 * 1024 * 1024

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _code_only(source: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", source))


def _read(path: Path) -> str:
    """The file's code, comments stripped, as one line."""
    return re.sub(r"\s+", " ", _code_only(path.read_text())).strip()


def _method_body(source: str, name: str) -> str:
    """A `HmxTarget` method's body as an expression (`return`/`;` stripped).

    Braces are matched, not regexed to the first `}`: a body may contain a
    braced initialiser (a `BridgePlan{...}` return) and stopping there would
    silently pin half a method.
    """
    match = re.search(
        r"\b(?:static\s+)?(?:constexpr\s+)?[A-Za-z_]\w*\s+"
        + re.escape(name)
        + r"\s*\([^)]*\)\s*(?:const\s*)?\{",
        source,
    )
    if not match:
        raise AssertionError(f"no {name} query in HmxTarget.h")
    depth = 0
    for index in range(match.end() - 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                body = source[match.end():index].strip()
                body = re.sub(r"^return\s+", "", body)
                return body.rstrip(";").strip()
    raise AssertionError(f"unbalanced body for {name}")


#: A consumer's token to the name the query's body assumes, so both sides
#: evaluate under one set of bindings. `roomBeside` reads the member
#: `vtcmBudget`; the pin evaluates it as the second parameter it is.
_RENAMES = (
    (r"HmxTarget::", ""),
    (r"layout::", ""),
    (r"this->", ""),
    (r"\bcontract\.", ""),
    (r"\btarget\.", ""),
    (r"\badmission\.budget\b", "budget"),
    (r"\bvtcmBudgetBytes\b", "option"),
    (r"\boptionBytes\b", "option"),
    (r"\binBytes\b", "croutonElemBytes"),
    (r"\boutType\.getDimSize\(0\)", "dim0"),
    (r"\boutType\.getDimSize\(1\)", "dim1"),
    (r"\b\(\*logical\.staticShape\)\[0\]", "shape0"),
    (r"\broomBeside\(\s*vtcmUsed\s*\)", "roomBeside(vtcmUsed, vtcmBudget)"),
)


def _python(expression: str) -> str:
    """A C++ arithmetic/boolean expression, as a Python expression."""
    text = expression
    for pattern, replacement in _RENAMES:
        text = re.sub(pattern, replacement, text)
    # The one ternary in the family: `x > 0 ? x : default`.
    ternary = re.fullmatch(r"(\w+ > 0) \? (\w+) : (\w+)", text)
    if ternary:
        text = f"({ternary.group(2)} if {ternary.group(1)} else {ternary.group(3)})"
    text = text.replace("&&", " and ")
    text = re.sub(r"(?<![&|<>!=])=(?!=)", "==", text)
    # Integer division: the only division this family performs.
    text = re.sub(r"(?<![)/])/(?![/*])", "//", text)
    return text


#: The queries' parameter names, in dependency order: the byte terms are
#: built before the footprint that composes them. A query that reads the
#: narrowed `vtcmBudget` member takes it as a parameter instead, so one set
#: of bindings can hold both a default and a narrowed instance.
_SIGNATURES = {
    "resolveVtcmBudget": ("option",),
    "roomBeside": ("vtcmUsed", "vtcmBudget"),
    "tilesIn": ("extent",),
    "isTileAligned": ("extent",),
    "hasEnoughRows": ("m",),
    "activationBytes": ("m", "k"),
    "weightBytes": ("k", "n"),
    "readoutBytes": ("m", "n"),
    "croutonBytes": ("rows", "n", "k"),
}

_CONSTANTS = {
    "tileEdge": TILE_EDGE,
    "minRows": MIN_ROWS,
    "defaultVtcmBudget": DEFAULT_VTCM_BUDGET,
    "croutonElemBytes": CROUTON_ELEM_BYTES,
}


def _evaluate(expression: str, bindings: dict) -> object:
    """Evaluate a C++ expression (query body or consumer spelling) in Python.

    The queries are rebuilt per evaluation, with the row's bindings in their
    globals: a lambda built once would close over the first row it saw.
    """
    header = _read(TARGET_HEADER)
    namespace = dict(_CONSTANTS)
    namespace.update(bindings)
    for name, parameters in _SIGNATURES.items():
        body = _python(_method_body(header, name))
        namespace[name] = eval(  # noqa: S307
            f"lambda {', '.join(parameters)}: ({body})",
            {"__builtins__": {}, **namespace},
        )
    return eval(_python(expression), {"__builtins__": {}}, namespace)  # noqa: S307


class HmxTargetQueryFacadeTest(unittest.TestCase):
    def setUp(self):
        self.header = _read(TARGET_HEADER)
        self.layout = _read(CRoUTON_LAYOUT)
        self.matmul = _read(MATMUL_PASS)
        self.partition = _read(PARTITION_PASS)
        self.weight = _read(WEIGHT_PASS)
        self.thread = _read(THREAD_PASS)
        self.manifest = _read(MANIFEST)
        # One row per role the parameters play. `vtcmBudget` is what a
        # narrowed HmxTarget instance holds, `budget` what a pass resolved
        # from its option, and the rest the shapes the bridges walk.
        self.rows = [
            dict(rows=96, extent=96, m=96, n=64, k=160, cols=160, dim0=96,
                 dim1=64, vtcmUsed=0, vtcmBudget=8388608, option=1234567,
                 budget=8388608, shape0=2),
            dict(rows=64, extent=160, m=64, n=64, k=128, cols=64, dim0=64,
                 dim1=64, vtcmUsed=16384, vtcmBudget=4096, option=4096,
                 budget=4096, shape0=8),
            dict(rows=32, extent=32, m=32, n=32, k=32, cols=32, dim0=32,
                 dim1=31, vtcmUsed=4096, vtcmBudget=4096, option=0,
                 budget=8388608, shape0=4),
        ]

    # -- helpers: the consumer's spelling, pinned as value and as text ------
    def assertSameValue(self, query: str, consumer: str):
        for row in self.rows:
            self.assertEqual(
                _evaluate(query, row),
                _evaluate(consumer, row),
                f"{query} != {consumer} at {row}",
            )

    def assertSite(self, source, needle: str, what: str):
        # A boolean check, not assertIn: a failure must print the needle,
        # not the megabyte-sized file it was searched in.
        self.assertTrue(needle in source, what)

    # -- the budget family --------------------------------------------------
    def test_resolve_vtcm_budget_matches_matmul_option(self):
        # matmul-to-hmx narrows the target from its option, 0 leaving the
        # device default. Written as an if-statement there; canonicalised
        # here as the ternary it means, with the text pinned below.
        self.assertSameValue(
            "resolveVtcmBudget(option)",
            "option > 0 ? option : defaultVtcmBudget",
        )
        self.assertSite(
            self.matmul,
            "if (vtcmBudgetBytes > 0) target.vtcmBudget = vtcmBudgetBytes;",
            "matmul-to-hmx no longer resolves its vtcm-budget option here "
            "(the reading moved, or this pin is stale)",
        )

    def test_resolve_vtcm_budget_matches_partition_option(self):
        # hmx-partition spells the same resolution as a ternary; evaluate
        # the spelling it actually wrote.
        self.assertSameValue(
            "resolveVtcmBudget(option)",
            "option > 0 ? option : defaultVtcmBudget",
        )
        self.assertSite(
            self.partition,
            "const int64_t vtcmBudget = this->vtcmBudgetBytes > 0 "
            "? this->vtcmBudgetBytes : HmxTarget::defaultVtcmBudget;",
            "hmx-partition no longer resolves its vtcm-budget option here "
            "(the reading moved, or this pin is stale)",
        )

    def test_budget_less_passes_read_the_device_default(self):
        # weight-resident and thread-role-partition have no option, so what
        # they read is the default the query resolves 0 to: whatever a
        # narrowed instance would hold, these two see the device's pool.
        for row in self.rows:
            unset = dict(row, option=0)
            self.assertEqual(
                _evaluate("resolveVtcmBudget(option)", unset),
                DEFAULT_VTCM_BUDGET,
            )
        self.assertSite(
            self.weight,
            "admission.budget = HmxTarget::defaultVtcmBudget;",
            "weight-resident no longer reads the device default here",
        )
        self.assertSite(
            self.thread,
            "int64_t budget = HmxTarget().vtcmBudget;",
            "thread-role-partition no longer reads the default budget here",
        )

    def test_room_beside_matches_matmul_room(self):
        self.assertSameValue("roomBeside(vtcmUsed)", "vtcmBudget - vtcmUsed")
        self.assertSite(
            self.matmul,
            "int64_t room = target.vtcmBudget - vtcmUsed;",
            "matmul-to-hmx no longer computes the room beside the "
            "committed bytes here",
        )

    # -- the tile-edge family -----------------------------------------------
    def test_tiles_in_matches_pack_tile_counts(self):
        self.assertSameValue("tilesIn(extent)", "extent / tileEdge")
        self.assertSite(
            self.matmul,
            "int64_t tileCols = cols / HmxTarget::tileEdge; "
            "int64_t rowTiles = rows / HmxTarget::tileEdge;",
            "the pack bridge no longer divides its extents by the tile "
            "edge here",
        )

    def test_is_tile_aligned_matches_fused_tail_legality(self):
        self.assertSameValue("isTileAligned(extent)", "extent % tileEdge == 0")
        self.assertSite(
            self.matmul,
            "return outType.getDimSize(0) % HmxTarget::tileEdge == 0 "
            "&& outType.getDimSize(1) % HmxTarget::tileEdge == 0;",
            "fusedTailLegal no longer checks the tile grid in place",
        )

    # -- the row-floor family -------------------------------------------------
    def test_has_enough_rows_matches_manifest_boundary(self):
        self.assertSameValue(
            "hasEnoughRows(shape0)", "not (shape0 <= minRows)"
        )
        self.assertSite(
            self.manifest,
            "(*logical.staticShape)[0] <= HmxTarget::minRows)",
            "the manifest no longer refuses on the row floor here",
        )

    # -- the crouton bytes family ---------------------------------------------
    def test_byte_terms_match_the_bridge_readings(self):
        self.assertSameValue(
            "activationBytes(m, k)", "m * k * croutonElemBytes")
        self.assertSameValue("weightBytes(k, n)", "k * n * croutonElemBytes")
        self.assertSameValue("readoutBytes(m, n)", "m * n * croutonElemBytes")
        self.assertSite(
            self.matmul,
            "int64_t lhsBytes = contract.m * contract.k * inBytes;",
            "the activation term is no longer spelled in place",
        )
        self.assertSite(
            self.matmul,
            "int64_t rhsBytes = contract.k * contract.n * inBytes;",
            "the weight term is no longer spelled in place",
        )
        self.assertSite(
            self.matmul,
            "int64_t outBytes = contract.m * contract.n * 2;",
            "the read-out term is no longer spelled in place (its bare 2 "
            "is what this pin anchors to croutonElemBytes)",
        )
        self.assertSite(
            self.matmul,
            "int64_t arBytes = contract.m * contract.n * HmxTarget::croutonElemBytes;",
            "the read-out term is no longer spelled in place on the "
            "accumulator path",
        )

    def test_crouton_bytes_is_the_sum_of_its_terms(self):
        # One footprint, three named terms, no fourth spelling inside the
        # facade itself.
        self.assertSite(
            self.header,
            "return activationBytes(rows, k) + weightBytes(k, n) "
            "+ readoutBytes(rows, n);",
            "croutonBytes no longer composes the three term queries",
        )
        for row in self.rows:
            self.assertEqual(
                _evaluate("croutonBytes(rows, n, k)", row),
                _evaluate("activationBytes(m, k)", row)
                + _evaluate("weightBytes(k, n)", row)
                + _evaluate("readoutBytes(m, n)", row),
            )

    def test_plan_bridge_uses_the_named_weight_term(self):
        body = _method_body(self.header, "planBridge")
        self.assertIn("weight = weightBytes(k, n);", body)

    def test_frozen_constants_live_in_the_header(self):
        # The frozen values above are the ones the lit files check; bind them
        # to the header so a header edit cannot leave them behind.
        match = re.search(r"static constexpr int64_t minRows = (\d+);", self.header)
        self.assertTrue(match, "minRows is not spelled in HmxTarget.h")
        self.assertEqual(int(match.group(1)), MIN_ROWS)
        match = re.search(
            r"static constexpr int64_t croutonElemBytes = (\d+);", self.header)
        self.assertTrue(match, "croutonElemBytes is not spelled in HmxTarget.h")
        self.assertEqual(int(match.group(1)), CROUTON_ELEM_BYTES)
        match = re.search(
            r"static constexpr int64_t defaultVtcmBudget = ([^;]+);", self.header)
        self.assertTrue(match, "defaultVtcmBudget is not spelled in HmxTarget.h")
        self.assertEqual(eval(match.group(1)), DEFAULT_VTCM_BUDGET)  # noqa: S307

    def test_tile_edge_is_the_layout_alias(self):
        self.assertSite(
            self.header,
            "static constexpr int64_t tileEdge = layout::kTileEdge;",
            "tileEdge is no longer the layout constant's alias",
        )
        self.assertSite(
            self.layout, "constexpr int64_t kTileEdge = 32;", "kTileEdge moved"
        )

    # -- the frozen values the IR side checks ---------------------------------
    def test_frozen_values_the_lit_files_check(self):
        # The lit files run the real pass; their CHECK literals are the
        # consumer-observed values, read here instead of restated.
        equality = EQUIVALENCE_LIT.read_text()
        refusals = REFUSALS_LIT.read_text()
        self.assertIn("vtcm_budget_bytes = 8388608 : i64", equality)
        self.assertIn("vtcm_budget_bytes = 1234567 : i64", equality)
        self.assertIn("vtcm_budget_bytes = 8388608 : i64", equality)
        # 64x64x64 and 96x64x160: the two bridge peaks the manifest prints,
        # and the tile counts the pack bridge walked.
        self.assertIn("vtcm_bridge_peak_bytes = 24576 : i64", equality)
        self.assertEqual(
            _evaluate("croutonBytes(rows, n, k)",
                      dict(rows=64, n=64, k=64)),
            24576,
        )
        # The refused fixture's own footprint, the one the 4096 remark
        # weighs against.
        self.assertIn("40960", refusals)
        self.assertEqual(
            _evaluate("croutonBytes(rows, n, k)",
                      dict(rows=64, n=64, k=128)),
            40960,
        )
        self.assertIn("vtcm_bridge_peak_bytes = 63488 : i64", equality)
        self.assertEqual(
            _evaluate("croutonBytes(rows, n, k)",
                      dict(rows=96, n=64, k=160)),
            63488,
        )
        self.assertEqual(
            [_evaluate("tilesIn(extent)", dict(extent=e))
             for e in (96, 160, 64)],
            [3, 5, 2],
        )
        # A narrowing option survives to the remark; the row floor is
        # stated by value.
        self.assertIn("vtcmBudget=4096 bytes", refusals)
        self.assertIn("M > 4", refusals)


if __name__ == "__main__":
    unittest.main()
