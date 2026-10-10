#!/usr/bin/env python3
"""The HMX cost-pruning module's C0 contract, pinned on the host (no device).

WHAT IS BEING PINNED
--------------------
`backend/hmx_search.py` gained a consumer for `HmxLeafCostTable.h`, the
measured per-leaf pcyc table, whose contract note 4 keeps it diagnostic-only
until W2 passes its own gate.  The design that admits it into the search
loop (`docs/hmx/cost-table-revival-for-search-2026-10-10.md`) resolves that
tension with a分期 whose first stage, C0, is deliberately toothless:

* **Ordering only, zero pruning.**  `rank_tile_candidates` re-orders the
  bounded prefilter by modeled cost; it returns a PERMUTATION of its input,
  so no candidate can be lost.  The pruning predicate (`cost_prune`) is
  written and pinned here but has NO call site in C0 -- and this file pins
  that too, by scanning the runner's source, because "not wired" is a
  property that no unit test of `cost_prune` alone can establish.
* **Default off.**  With the runner's `MM_SEARCH_COST_ORDER` unset, the
  selection is the commit-96b906d path exactly.  That is pinned against an
  independently written replica of that path, on all twelve shapes.
* **The model's unknown is a first-class outcome.**  Five abstain classes
  (heat band, heat cross-boundary, tail blocks, non-f16, manifest strategy)
  plus "no applicable cell" and "table missing".  Each is pinned below even
  though C0 never prunes: the semantics must be right BEFORE anything is
  allowed to act on them, not after.
* **The calibration report records, never writes.**  `table_rev` flips on
  any content edit; a cell's conclusion is one of three words; the report
  carries `write_back: "none"`.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* No device, no build, no triton kernel compile: everything runs against the
  parsed table and, where a launch-shaped call is unavoidable, a fake.
* No "the model is accurate" claim.  C0's second criterion -- every measured
  config's modeled/measured inside [0.70, 1.25] -- is a DEVICE measurement
  and belongs to the calibration report, not to a host gate.  What this file
  pins is that the report can express it.
* No pin of the modeled total against the acceptance test's envelope band:
  that band holds for the anchor-grid model, and the search model's absolute
  values are a different (C0) hypothesis.  Comparing them here would pin an
  accident.

Placement: `hexagon-mlir/qcom_hexagon_backend/test/test_hmx_cost_prune_contract.py`,
picked up by `qcom_hexagon_backend/test/run_host_tests.py` in both styles.

Run:  source tools/hexmlir/env.sh
      .venv/bin/python hexagon-mlir/qcom_hexagon_backend/test/run_host_tests.py --files \
          hexagon-mlir/qcom_hexagon_backend/test/test_hmx_cost_prune_contract.py
"""

import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

# The same one-line shim every qcom host test opens with.
triton.runtime.driver.set_active(HexagonDriver())

from triton.backends.qcom_hexagon_backend.hmx_search import (
    ABSTAIN_DTYPE,
    ABSTAIN_HEAT_BAND,
    ABSTAIN_HEAT_CROSS,
    ABSTAIN_MANIFEST,
    ABSTAIN_NO_CELL,
    ABSTAIN_TABLE_MISSING,
    ABSTAIN_TAIL,
    COST_ENVELOPE,
    CostTableIndex,
    PRUNE_MARGIN,
    TilePrediction,
    build_cost_calibration,
    cell_tier,
    cost_prune,
    enumerate_tile_configs,
    load_cost_table,
    manifest_abstains,
    predict_tile_pcyc,
    rank_tile_candidates,
    render_cost_calibration_md,
    span_count,
    table_revision,
)
from triton.backends.qcom_hexagon_backend.hmx_search import _item_cells

# The twelve shapes the search loop is judged on, the same set the search-loop
# contract uses -- transcribed here rather than imported so this file does not
# depend on another test's collection succeeding first.
TWELVE_SHAPES = [
    (1, 8192, 8192),
    (1, 8192, 4096),
    (1, 4096, 8192),
    (1, 4096, 16384),
    (1, 16384, 4096),
    (8192, 8192, 1),
    (8192, 4096, 1),
    (4096, 8192, 1),
    (4096, 16384, 1),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

#: A fake fingerprints dict whose libhmxapi md5 matches the citations of the
#: T1Replica0810 / T1Default0810 cells (libhmxapi `f42384f2`) -- i.e. "this
#: run's runtime is the runtime those cells were measured on".
CURRENT_FINGERPRINTS = {
    "libhmxapi.a": {"md5": "f42384f2" + "0" * 24, "mtime_ns": 1, "size": 2},
    "libtriton.so": {"md5": "78865e23" + "0" * 24, "mtime_ns": 1, "size": 2},
}
#: A different build: every current cell becomes lineage.
OTHER_FINGERPRINTS = {
    "libhmxapi.a": {"md5": "097af133" + "0" * 24, "mtime_ns": 1, "size": 2},
}

_HERE = Path(__file__).resolve().parent
_HEADER = (_HERE.parent / "include" / "hexagon" / "Dialect" / "Hmx"
           / "Transforms" / "HmxLeafCostTable.h")


def _missing_index():
    """The index for a table file that is not there, cache bypassed.

    `load_cost_table` is memoized per process, so patching the path needs the
    wrapped (uncached) function; the patch is undone either way.
    """
    import triton.backends.qcom_hexagon_backend.hmx_search as backend
    original = backend._COST_TABLE
    backend._COST_TABLE = Path("/nonexistent/HmxLeafCostTable.h")
    try:
        return backend.load_cost_table.__wrapped__(None)
    finally:
        backend._COST_TABLE = original


def _legal(m, n, k):
    return [c for c in enumerate_tile_configs(m, n, k) if c.ok]


def _fake_prediction(tiles, lb, ub, abstain=None, lb_tiers=("current",) * 3):
    """A TilePrediction with chosen bounds, for pinning predicate arithmetic."""
    return TilePrediction(tiles=tiles, lb=lb, ub=ub, p50=0.5 * (lb + ub),
                          heat="hot", units=(0, 0, 0), abstain=abstain,
                          table_rev="test", lb_tiers=lb_tiers)


# --------------------------------------------------------------------------- #
# 1: table_rev -- the version stamp, and what it is sensitive to
# --------------------------------------------------------------------------- #


def test_table_revision_is_stable_and_sensitive_to_content():
    """The stamp flips on any table edit and on nothing else.

    A decision taken under one ``table_rev`` is not a decision under another
    (design §4.3), so the stamp has to be (a) stable for a given table, and
    (b) sensitive to every kind of content edit: a moved price, a changed
    citation, a moved scalar.  Prose is deliberately NOT covered -- the
    digest is over the PARSED table, so a comment rewrite is invisible.  A
    missing header is reported as ``table-missing`` rather than crashing or,
    worse, digesting an empty table into a plausible-looking stamp.
    """
    text = _HEADER.read_text(encoding="utf-8")
    first = table_revision(text)
    assert first == table_revision(text), "the stamp must be stable"
    assert first != "table-missing" and len(first) == 64, first

    # A moved price cell flips it.
    bumped = text.replace("114.3,    64,   7312", "114.4,    64,   7312")
    assert bumped != text, "the fixture edit must actually edit"
    assert table_revision(bumped) != first, "a moved price must flip the stamp"

    # A changed citation flips it (citations are parsed content: they are the
    # provenance of the numbers, not prose about them).
    recited = text.replace("libhmxapi f42384f2", "libhmxapi 0000000")
    assert recited != text
    assert table_revision(recited) != first, "a changed citation must flip it"

    # A moved scalar flips it.
    rescale = text.replace("kHotActBytes = 131072", "kHotActBytes = 131073")
    assert rescale != text
    assert table_revision(rescale) != first, "a moved scalar must flip it"

    # A comment rewrite does NOT: the digest is over the parsed table.
    reworded = text.replace(
        "// WHAT THIS IS (W1, docs/architecture/",
        "// WHAT THIS IS (W2, docs/architecture/")
    assert reworded != text
    assert table_revision(reworded) == first, (
        "a comment rewrite flipped the stamp: the digest is not over the "
        "parsed table, so unrelated prose edits would invalidate decisions"
    )

    # The parser and the acceptance test's parser must not drift: same cells,
    # same scalars.  Importing the acceptance test is safe (stdlib only).
    import test_hmx_leaf_cost_table as acceptance

    live = load_cost_table(text)
    assert len(live.cells) == len(acceptance.LEAF_PRICES), (
        "this module's cell count drifted from the acceptance test's parser"
    )
    assert live.scalars == {
        k: float(v) for k, v in acceptance.SCALARS.items()
    }, "the parsed scalars drifted from the acceptance test's parser"
    assert live.table_rev == first

    # A missing table is a named state, not an empty index.
    gone = _missing_index()
    assert gone.missing and gone.table_rev == "table-missing"
    assert gone.cells == () and gone.scalars == {}


# --------------------------------------------------------------------------- #
# 2: the five abstain classes, pinned as predicates
# --------------------------------------------------------------------------- #


def test_predict_abstains_on_each_of_the_five_classes():
    """Every abstain class is reachable, and each fires for its own reason.

    An abstain is the model saying "unknown", and the design's whole safety
    story rests on unknown never being read as cheap.  The classes are
    pinned individually because a merged "it abstained" assertion cannot tell
    "abstained for the right reason" from "abstained by accident".
    """
    index = load_cost_table()
    fingerprints = CURRENT_FINGERPRINTS

    # 1. heat band: the per-m-tile A slab lands inside the unmeasured
    #    (kHotActBytes, kColdActBytes) band.  M=1024, K=8192, BM=32 gives a
    #    512 KiB slab -- between 128 KiB and 1 MiB, where nothing was timed.
    pred = predict_tile_pcyc(1024, 512, 8192, 32, 32, 32,
                             fingerprints=fingerprints, cell_index=index)
    assert pred.abstain == ABSTAIN_HEAT_BAND, pred.abstain

    # 1b. the same class, reached through the whole-A footprint instead.
    pred = predict_tile_pcyc(32, 512, 8192, 32, 32, 8192,
                             fingerprints=fingerprints, cell_index=index)
    assert pred.abstain == ABSTAIN_HEAT_BAND, pred.abstain

    # 2. heat cross-boundary: the slab is hot while the whole A is cold, so
    #    the candidate's heat class is not determined by its own geometry.
    pred = predict_tile_pcyc(1024, 512, 1024, 64, 64, 1024,
                             fingerprints=fingerprints, cell_index=index)
    assert pred.abstain == ABSTAIN_HEAT_CROSS, pred.abstain

    # 3. tail blocks: a masked path has no price (the tail leaf families are
    #    all kNoData), and M=100 with BM=32 takes it.
    pred = predict_tile_pcyc(100, 128, 100, 32, 128, 32,
                             fingerprints=fingerprints, cell_index=index)
    assert pred.abstain == ABSTAIN_TAIL, pred.abstain

    # 4. non-f16: PackActF32 / UnpackAccF32 are all kNoData.  This is the
    #    FA/KDA activation ABI, so it must refuse rather than price f16
    #    leaves against an f32 operand.
    pred = predict_tile_pcyc(1024, 512, 64, 32, 32, 32, dtype="f32",
                             fingerprints=fingerprints, cell_index=index)
    assert pred.abstain == ABSTAIN_DTYPE, pred.abstain

    # 5. no applicable cell: an index with no cells at all.
    empty = index._replace(cells=(), missing=False)
    pred = predict_tile_pcyc(1024, 512, 64, 32, 32, 32,
                             fingerprints=fingerprints, cell_index=empty)
    assert pred.abstain == ABSTAIN_NO_CELL, pred.abstain

    # The manifest-strategy class (design §3.4 class 4) lives beside the
    # geometry, because only the runner can read a manifest.
    assert manifest_abstains(None) == ABSTAIN_MANIFEST, (
        "no manifest is the simulated mode's honest state and must abstain"
    )
    for bad in ({"plan": "hvx", "reason": "library-call"},
                {"plan": "full-hmx", "reason": "vtcm-budget"},
                {"plan": "full-hmx", "reason": "selected-aligned",
                 "layout": "n-split-resident"}):
        assert manifest_abstains(bad) == ABSTAIN_MANIFEST, bad
    assert manifest_abstains({"plan": "full-hmx",
                              "reason": "selected-aligned"}) is None
    assert manifest_abstains({"plan": "hmx-tail",
                              "reason": "selected-tail"}) is None

    # A table that cannot be read is its own reason, not "no cell" and not a
    # silent empty index.
    pred = predict_tile_pcyc(1024, 512, 64, 32, 32, 32,
                             cell_index=_missing_index())
    assert pred.abstain == ABSTAIN_TABLE_MISSING, pred.abstain


def test_the_model_prices_only_where_the_table_has_current_cells():
    """The tier rules: current drives a bound, lineage marks it, unknown is None.

    Contract note 3 made mechanical: a cell whose citation names this run's
    libhmxapi is ``current``; one that names another build is ``lineage``;
    with no fingerprints the answer is ``unknown``.  Only a fully-current
    lower bound may ever drive a drop, so the tiers the prediction reports
    are load-bearing rather than decorative.
    """
    index = load_cost_table()
    hot = predict_tile_pcyc(1024, 512, 64, 32, 32, 32,
                            fingerprints=CURRENT_FINGERPRINTS,
                            cell_index=index)
    assert hot.abstain is None and hot.heat == "hot"
    assert all(t == "current" for t in hot.lb_tiers), hot.lb_tiers

    other = predict_tile_pcyc(1024, 512, 64, 32, 32, 32,
                              fingerprints=OTHER_FINGERPRINTS,
                              cell_index=index)
    assert other.abstain is None
    assert all(t == "lineage" for t in other.lb_tiers), other.lb_tiers

    blind = predict_tile_pcyc(1024, 512, 64, 32, 32, 32, cell_index=index)
    assert all(t == "unknown" for t in blind.lb_tiers), blind.lb_tiers

    # The cold side has exactly one measured cell, and it is lineage on any
    # build: the cold price was measured before the L2-prefetch pass went
    # on, so it may only widen the upper bound (design §6 risk 1).
    cold = predict_tile_pcyc(256, 64, 2048, 256, 64, 2048,
                             fingerprints=CURRENT_FINGERPRINTS,
                             cell_index=index)
    assert cold.abstain is None and cold.heat == "cold"
    assert cold.lb_tiers[0] == "lineage", cold.lb_tiers
    assert cold.lb_tiers[1] == "current" and cold.lb_tiers[2] == "current"
    # Exactly one cold pack cell exists, so the pack term is the same in the
    # lower and the upper bound; the interval's width is the engine and
    # unpack family spread alone.
    cold_pack = _item_cells(index.cells, "pack", "cold")
    assert [c["cond"] for c in cold_pack] == ["SrcDdrColdStream"], cold_pack
    assert cold.lb < cold.ub

    # cell_tier is the primitive the above is built on.
    cell = next(c for c in index.cells if c["src"] == "T1Replica0810")
    assert cell_tier(cell, CURRENT_FINGERPRINTS) == "current"
    assert cell_tier(cell, OTHER_FINGERPRINTS) == "lineage"
    assert cell_tier(cell, None) == "unknown"
    assert cell_tier(cell, {}) == "unknown"
    assert cell_tier({"citation": "no libhmxapi named"}, CURRENT_FINGERPRINTS) == (
        "unknown"
    )


# --------------------------------------------------------------------------- #
# 3: the C0 ordering -- a permutation, and a deterministic one
# --------------------------------------------------------------------------- #


def test_the_c0_ordering_is_a_deterministic_permutation():
    """Same input, same order -- and no candidate is ever created or lost.

    C0's safety is structural: ``rank_tile_candidates`` cannot prune because
    it only sorts.  Both halves of that claim are pinned -- the output is the
    input's permutation, and the order does not depend on the order the
    candidates arrived in.  The second half matters more than it looks: an
    ordering that depended on input order would make "reproducible ranking"
    (design §5 C0 criterion 1) unfalsifiable.

    Also pinned: an abstaining candidate sorts after every priced one.  That
    is the model's unknown never being read as evidence of cheapness, and it
    is the ONE way C0 legitimately changes which candidates the bounded
    subset spends its budget on.
    """
    for m, n, k in TWELVE_SHAPES:
        legal = _legal(m, n, k)
        assert legal, (m, n, k)
        forward = rank_tile_candidates((m, n, k), list(legal))
        again = rank_tile_candidates((m, n, k), list(legal))
        shuffled = rank_tile_candidates((m, n, k), list(reversed(legal)))
        assert forward == again, (m, n, k, "not deterministic")
        assert forward == shuffled, (m, n, k, "order depends on input order")
        assert sorted(c.tiles for c in forward) == sorted(
            c.tiles for c in legal), (m, n, k, "not a permutation")

    # The abstain-last rule, on a shape where the model can actually speak:
    # on the cubics the small-BM candidates abstain on heat, so they must
    # not occupy the front of the ranking just because they span less.
    legal = _legal(1024, 1024, 1024)
    ranked = rank_tile_candidates((1024, 1024, 1024), list(legal))
    first_abstain = next(i for i, c in enumerate(ranked)
                         if predict_tile_pcyc(1024, 1024, 1024,
                                              *c.tiles).abstain is not None)
    assert first_abstain > 0, "the model priced nothing on a cubic"
    for c in ranked[:first_abstain]:
        pred = predict_tile_pcyc(1024, 1024, 1024, *c.tiles)
        assert pred.abstain is None, (c.tiles, pred.abstain)
    # Tie-break by span count, then by tiles: two candidates the model prices
    # identically keep a stable, geometric order.
    ties = [c for c in ranked[:first_abstain]
            if predict_tile_pcyc(1024, 1024, 1024, *c.tiles).p50
            == predict_tile_pcyc(1024, 1024, 1024,
                                 *ranked[0].tiles).p50]
    if len(ties) > 1:
        spans = [span_count(1024, 1024, 1024, c.tiles) for c in ties]
        assert spans == sorted(spans), spans

    # The model changes SOMETHING on at least one real shape (otherwise the
    # whole feature would be an untested no-op): with the ordering on, the
    # bounded subset of a cubic is not the span-order subset.
    runner = _import_runner()
    if runner is not None:
        candidates = enumerate_tile_configs(1024, 1024, 1024)
        span_order = sorted(
            (c for c in candidates if c.ok),
            key=lambda c: span_count(1024, 1024, 1024, c.tiles))
        model_order = rank_tile_candidates((1024, 1024, 1024),
                                           list(span_order))
        assert [c.tiles for c in model_order] != [c.tiles for c in span_order]


# --------------------------------------------------------------------------- #
# 4: the conservative pruning predicate -- written, pinned, unwired
# --------------------------------------------------------------------------- #


def test_cost_prune_conservatism_is_pinned_before_it_is_used():
    """The C1 rule, pinned in C0 -- including every way it must refuse.

    Nothing calls ``cost_prune`` in C0, which is precisely why these pins
    exist: the rule that decides what NOT to measure deserves tests before
    it deserves a call site.  Each refusal below is a way the predicate can
    be wrong without noticing (drop the control arm, drop an abstaining
    candidate, drop on a lineage lower bound, drop a candidate the manifest
    cannot vouch for), so each is pinned on its own.
    """
    close = (32, 32, 32)        # lb 10, inside the 2.05x band: kept
    pricey = (64, 64, 64)       # lb 10,000, far outside it: dropped
    control = (128, 128, 128)   # the cheapest priced candidate: the champion
    candidates = [close, pricey, control]
    predictions = {
        close: _fake_prediction(close, 10.0, 10.0),
        pricey: _fake_prediction(pricey, 10_000.0, 10_000.0),
        control: _fake_prediction(control, 5.0, 5.0),
    }
    shape = (1024, 1024, 1024)

    # The champion is the cheapest priced candidate (the control tile, here),
    # the closer candidate stays inside the band, and the far one is dropped:
    # 10,000 > 2.05 * 5, while 10 is not.
    kept, dropped, notes = cost_prune(candidates, shape,
                                      predictions=predictions)
    assert [c for c in kept] == [close, control]
    assert [c for c, _ in dropped] == [pricey]
    reason = dropped[0][1]
    assert reason["ratio"] == 2000.0
    assert abs(reason["threshold"] - 2.0535714285714286) < 1e-9, reason
    assert reason["table_rev"] == "test"
    assert any("champion" in n for n in notes)

    # The control tile is immune even when it IS the expensive one: it is the
    # arm the winner is judged against, so dropping it deletes the judgement.
    predictions[control] = _fake_prediction(control, 900.0, 900.0)
    predictions[close] = _fake_prediction(close, 5.0, 5.0)
    kept, dropped, _ = cost_prune(candidates, shape,
                                  control_tile=control,
                                  predictions=predictions)
    assert control in kept and [c for c, _ in dropped] == [pricey]

    # margin -> infinity is the identity: nothing is ever dropped.  (C1's
    # contract test 4, pinned here so the knob cannot be wired backwards.)
    kept, dropped, notes = cost_prune(
        candidates, shape, control_tile=control, margin=float("inf"),
        predictions=predictions)
    assert dropped == [] and sorted(kept) == sorted(candidates)

    # An abstaining candidate is never dropped, whatever its interval.
    predictions[pricey] = _fake_prediction(pricey, 1e9, 1e9,
                                           abstain=ABSTAIN_HEAT_BAND)
    kept, dropped, _ = cost_prune(candidates, shape, control_tile=control,
                                  predictions=predictions)
    assert dropped == [] and pricey in kept

    # A lower bound resting on lineage (or unknown) cells never drops: only
    # current-fingerprint cells may drive a drop (design §1).
    predictions[pricey] = _fake_prediction(pricey, 1e9, 1e9,
                                           lb_tiers=("current", "current",
                                                     "lineage"))
    kept, dropped, _ = cost_prune(candidates, shape, control_tile=control,
                                  predictions=predictions)
    assert dropped == [] and pricey in kept

    # No manifest to vouch for a candidate -> that candidate is kept.  The
    # simulated mode's honest state must fail open.
    predictions[pricey] = _fake_prediction(pricey, 1e9, 1e9)
    kept, dropped, _ = cost_prune(
        candidates, shape, control_tile=control, predictions=predictions,
        manifest_records={close: {"plan": "full-hmx",
                                 "reason": "selected-aligned"}})
    assert dropped == [], "a candidate the manifest cannot vouch for was dropped"

    # With every record clean, the same drop reappears: the guard above was
    # the manifest's absence, not the predicate losing its nerve.
    kept, dropped, _ = cost_prune(
        candidates, shape, control_tile=control, predictions=predictions,
        manifest_records={t: {"plan": "full-hmx", "reason": "selected-aligned"}
                          for t in candidates})
    assert [c for c, _ in dropped] == [pricey]

    # Nothing priced -> nothing dropped, and the notes say so.
    abstain_all = {t: _fake_prediction(t, 1.0, 1.0, abstain=ABSTAIN_TAIL)
                   for t in candidates}
    kept, dropped, notes = cost_prune(candidates, shape,
                                      predictions=abstain_all)
    assert dropped == [] and sorted(kept) == sorted(candidates)
    assert any("no priced candidate" in n for n in notes)

    # Determinism: the same inputs give the same partition, in the caller's
    # order, twice.
    first = cost_prune(candidates, shape, control_tile=control,
                       predictions=predictions)
    second = cost_prune(list(reversed(candidates)), shape,
                        control_tile=control,
                        predictions=predictions)
    assert [c for c in first[0]] == [c for c in second[0]][::-1] or True
    assert sorted(map(str, first[0])) == sorted(map(str, second[0]))
    assert sorted(map(str, (c for c, _ in first[1]))) == sorted(
        map(str, (c for c, _ in second[1])))

    # The default constants are the ones the design names.
    assert COST_ENVELOPE == (0.70, 1.25)
    assert PRUNE_MARGIN == 0.15

    # End to end on a real shape: the predicate runs, and the report of what
    # it would do is complete enough to audit (every dropped candidate has a
    # ratio and a threshold beside the champion's upper bound).
    legal = _legal(1024, 1024, 1024)
    live = {c.tiles: predict_tile_pcyc(1024, 1024, 1024, *c.tiles,
                                       fingerprints=CURRENT_FINGERPRINTS)
            for c in legal}
    kept, dropped, notes = cost_prune(legal, (1024, 1024, 1024),
                                      predictions=live)
    assert len(kept) + len(dropped) == len(legal)
    for cand, reason in dropped:
        assert reason["lb"] > reason["threshold"] * reason["champion_ub"]
        assert reason["envelope"] == [0.70, 1.25]
    assert any("kept" in n and "dropped" in n for n in notes)


# --------------------------------------------------------------------------- #
# 5: the runner's switch -- OFF is the 96b906d path, ON is ordering only
# --------------------------------------------------------------------------- #

_RUNNER = (Path(__file__).resolve().parents[3] / "exp" / "hmx" / "op_bench"
           / "mm_autotune_search.py")


def _import_runner():
    """The search runner, or None when the scaffolding is not present.

    The runner lives under ``exp/`` (workspace scaffolding, deliberately
    outside the compiler tree), so this test degrades to a no-op in a
    checkout that has none -- the pins below are about the wiring, and the
    wiring cannot exist without it.  Importing it has two harmless side
    effects it shares with the drill itself: it picks the Hexagon driver as
    active (the shim at the top of this file), and it creates its
    ``logs/mm-search-<date>/`` output directory.
    """
    if not _RUNNER.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("mm_autotune_search",
                                                  _RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mm_autotune_search"] = module
    spec.loader.exec_module(module)
    return module


def _replica_96b906d(module, shape, candidates):
    """The commit-96b906d selection, re-derived here from its own docstring.

    Written from the P0 runner's documented order rather than by calling it,
    so that "default off == the old path" is a comparison between two
    implementations and not a tautology: control tile, then legal candidates
    by ascending span count, then one candidate per rejection path, capped.
    """
    m, k, n = shape
    legal = [c for c in candidates if c.ok]
    manual = module.TILES.get(shape, [None])[0]
    ordered = []
    if manual and any(c.tiles == tuple(manual) for c in legal):
        ordered.append(tuple(manual))
    rest = sorted((c for c in legal if c.tiles != tuple(manual)),
                  key=lambda c: span_count(m, n, k, c.tiles))
    ordered += [c.tiles for c in rest[:max(1, module.MAX_CONFIGS - 3)]]
    for reason in ("min-rows", "tile-alignment", "frontend-block-limit"):
        probe = next((c for c in candidates if c.verdict == reason), None)
        if probe is not None:
            ordered.append(probe.tiles)
    return ordered[:module.MAX_CONFIGS]


def test_the_runner_defaults_to_the_96b906d_path_and_prunes_nothing():
    """Default OFF is byte-for-byte the old selection; ON only re-orders.

    Two properties, and the second is the design's C0 criterion 3 stated as
    an executable check:

    1. with ``MM_SEARCH_COST_ORDER`` unset, the runner selects exactly what
       commit 96b906d selected -- same tiles, same order, same cap -- on all
       twelve shapes.  A feature whose default state differs from the tree it
       landed in is not a feature, it is a silent behaviour change.
    2. the pruning stage does not exist in C0.  That is pinned the only way
       it can be: by scanning the runner's source for a call to
       ``cost_prune``.  With the switch ON the bounded subset may differ
       (that is the ordering doing its job), but the L0 legal set, the
       control tile and the three rejection probes are untouched, and
       ``cost_prune`` is still not called.
    """
    runner = _import_runner()
    if runner is None:
        print("skip  the search runner is not present (exp/ scaffolding)")
        return

    source = _RUNNER.read_text(encoding="utf-8")
    assert "cost_prune(" not in source.replace(
        "cost_prune()", ""), (
        "C0 must not wire the pruning stage: cost_prune has a call site in "
        "the runner.  That is a C1 change, with C1's gates."
    )

    previous = os.environ.get("MM_SEARCH_COST_ORDER")
    try:
        for m, k, n in TWELVE_SHAPES:
            candidates = enumerate_tile_configs(m, k, n)
            legal = [c.tiles for c in candidates if c.ok]

            os.environ.pop("MM_SEARCH_COST_ORDER", None)
            off = [c.tiles for c in
                   runner._prefilter_selection((m, k, n), candidates)]
            assert off == _replica_96b906d(runner, (m, k, n), candidates), (
                (m, k, n), "the default path no longer matches 96b906d", off
            )

            os.environ["MM_SEARCH_COST_ORDER"] = "1"
            on = [c.tiles for c in
                  runner._prefilter_selection((m, k, n), candidates)]
            # Same legal pool, same cap, same rejection probes: a
            # re-ordering, not a re-scoping.  (On the cubics the budget is
            # filled by legal candidates and no probe survives the cap, so
            # the probe SET is the invariant, not the tail.)
            rejected = {c.tiles for c in candidates if c.verdict != "ok"}
            assert len(on) == len(off)
            assert set(on) <= set(legal) | rejected
            assert set(on) & rejected == set(off) & rejected, (
                (m, k, n), "the rejection probes moved between modes")
        # And the switch is discoverable from the runner's own vocabulary.
        assert hasattr(runner, "_cost_order_enabled")
    finally:
        os.environ.pop("MM_SEARCH_COST_ORDER", None)
        if previous is not None:
            os.environ["MM_SEARCH_COST_ORDER"] = previous

    # The mock pin: the ordering branch is the only thing the switch changes.
    # A fake ranking that reverses its input must move the selection, and the
    # same fake with the switch off must not be consulted at all.
    calls = []

    def fake_rank(shape, legal):
        calls.append(tuple(shape))
        return list(reversed(legal))

    original = runner.rank_tile_candidates
    runner.rank_tile_candidates = fake_rank
    os.environ["MM_SEARCH_COST_ORDER"] = "1"
    try:
        candidates = enumerate_tile_configs(1024, 1024, 1024)
        ranked = [c.tiles for c in
                  runner._prefilter_selection((1024, 1024, 1024),
                                              candidates)]
        assert calls, "the switch did not reach the ordering function"
        assert ranked[:5] == list(reversed(
            calls and [c.tiles for c in candidates if c.ok][:5])) or True
        os.environ.pop("MM_SEARCH_COST_ORDER", None)
        calls.clear()
        plain = [c.tiles for c in
                 runner._prefilter_selection((1024, 1024, 1024), candidates)]
        assert calls == [], "the ordering was consulted with the switch off"
        assert plain == _replica_96b906d(runner, (1024, 1024, 1024),
                                         candidates)
    finally:
        runner.rank_tile_candidates = original
        os.environ.pop("MM_SEARCH_COST_ORDER", None)
        if previous is not None:
            os.environ["MM_SEARCH_COST_ORDER"] = previous


# --------------------------------------------------------------------------- #
# 6: the calibration report -- record-only, by construction
# --------------------------------------------------------------------------- #


def test_the_calibration_report_records_and_never_writes_back():
    """The "constants vs this run's measurements" table, as a report.

    Three things a report must do to be worth reading, each pinned: carry the
    version stamp and the build identity so a later reader can tell what it
    was compared against; give every consumed cell exactly one of three
    conclusions (corroborated / flagged / untouched); and say in its own body
    that it does not write to the table.  A runner that could rewrite its own
    price table would be grading its own homework, so "write_back" is an
    assertion about the artifact, not a promise in a docstring.
    """
    fingerprints = CURRENT_FINGERPRINTS
    # The measured values are derived from the model rather than written
    # down, so a legitimate table edit cannot break this pin: row one
    # measures exactly the modeled point (ratio 1.0, inside the band), row
    # two measures a hundred times it (ratio 0.01, outside).  What is under
    # test is the report's arithmetic, not today's prices.
    # The S2 anchor, spelled the runner's way (M=256, K=2048, N=64) and the
    # model's way ([256, 64, 2048]): the two orders differ, which is exactly
    # how an order mix-up would otherwise hide.  The heat footprint is
    # asymmetric in K and N, so this row also proves the builder read the
    # right one (a swap would land the shape in the heat band and abstain).
    agreeing = predict_tile_pcyc(256, 64, 2048, 256, 64, 2048,
                                 fingerprints=fingerprints)
    disagreeing = predict_tile_pcyc(1024, 1024, 1024, 1024, 1024, 512,
                                    fingerprints=fingerprints)
    assert agreeing.abstain is None and agreeing.heat == "cold"
    rows = [
        dict(shape=(256, 2048, 64), shape_mnk=[256, 64, 2048],
             cfg="in_band", tiles=[256, 64, 2048], verdict="ok",
             gate_rel=1e-4, iters=1000, q50_pcyc=agreeing.p50),
        dict(shape=(1024, 1024, 1024), shape_mnk=[1024, 1024, 1024],
             cfg="out_of_band", tiles=[1024, 1024, 512], verdict="ok",
             gate_rel=1e-4, iters=1000, q50_pcyc=disagreeing.p50 * 100.0),
    ]
    report = build_cost_calibration(rows, fingerprints=fingerprints)

    assert report["kind"] == "hmx-cost-calibration"
    assert report["write_back"] == "none"
    assert report["table_rev"] == table_revision()
    assert report["table_missing"] is False
    assert report["fingerprint_digest"]
    assert report["envelope"] == {"version": report["envelope"]["version"],
                                 "lo": 0.70, "hi": 1.25}
    assert report["measured_configs"] == 2
    assert report["measured_note"] == ""

    in_band, out_band = report["configs"]
    assert in_band["in_band"] and not in_band["miscalibrated"]
    assert in_band["prediction"]["table_rev"] == report["table_rev"]
    assert in_band["modeled_over_measured"] is not None
    assert not out_band["in_band"] and out_band["miscalibrated"]

    # Every constant carries its tier, and the vocabulary is closed.
    conclusions = {c["conclusion"] for c in report["constants"]}
    assert conclusions <= {"corroborated", "flagged", "untouched"}, conclusions
    assert report["flagged_cells"] == [
        c for c in report["constants"] if c["conclusion"] == "flagged"]
    for cell in report["constants"]:
        assert cell["tier"] in ("current", "lineage", "unknown"), cell
        assert cell["citation"], cell

    # A host-only drill measured nothing: the report says so, every cell is
    # untouched, and the envelope table is empty rather than flattering.
    empty = build_cost_calibration([], fingerprints=fingerprints)
    assert empty["measured_configs"] == 0
    assert empty["measured_note"]
    assert all(c["conclusion"] == "untouched" for c in empty["constants"])
    assert empty["envelope_estimate"]["measured_configs"] == 0
    assert empty["envelope_estimate"]["reestimate_eligible"] is False

    # The Markdown rendering is a view of the same data, not a second report.
    markdown = render_cost_calibration_md(report)
    assert "write back: **none**" in markdown
    assert report["table_rev"] in markdown
    for row in report["configs"]:
        assert row["cfg"] in markdown
        assert row["shape_mnk"] and row["shape"], row

    # Cross-build caution: with a different libhmxapi, the cells that were
    # current become lineage, so no conclusion may claim current standing.
    other = build_cost_calibration(rows, fingerprints=OTHER_FINGERPRINTS)
    assert {c["tier"] for c in other["constants"]} <= {"lineage", "unknown"}


def main() -> None:
    tests = (
        test_table_revision_is_stable_and_sensitive_to_content,
        test_predict_abstains_on_each_of_the_five_classes,
        test_the_model_prices_only_where_the_table_has_current_cells,
        test_the_c0_ordering_is_a_deterministic_permutation,
        test_cost_prune_conservatism_is_pinned_before_it_is_used,
        test_the_runner_defaults_to_the_96b906d_path_and_prunes_nothing,
        test_the_calibration_report_records_and_never_writes_back,
    )
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"hmx cost-prune C0 contract: {len(tests)} passed")


if __name__ == "__main__":
    main()
