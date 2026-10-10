#!/usr/bin/env python3
"""The champion criterion and the per-round canary, pinned on the host.

What is being pinned
--------------------
The search's runner promised a verdict it never implemented: "the winner is
judged against the manual default tile by `max(3xCV, 15%)`" with no CV
computed anywhere, and a champion decided by a 3-rep argmin (defect scan
`docs/analysis/defect-scan-2026-10-10.md` N1).  This file is the gate for
the mechanism that now answers it, in `backend/hmx_search.py`:

1. **the gate is the project's own** -- `max(3 x CV, 15%)`, inherited
   numbers (the same floor `PRUNE_MARGIN` names), and the tie goes to the
   incumbent: a challenger takes the crown only by being faster by MORE
   than the gate.
2. **the crown is defended, not argmin'd** -- the walk starts at the manual
   control tile and a challenger that beats the incumbent by less than the
   gate keeps the incumbent.  A 2% "improvement" on noise cannot take the
   crown; the changeover log says who beat whom, by how much, under which
   gate.
3. **a canary failure voids the round** -- a known-healthy minimal config,
   re-measured every round, separates "this config is slow" from "the
   device is not well".  Missing, unmeasured or drifted canary: the round is
   `invalid-canary`, the champion is still named (it is the walk's
   survivor) but is marked not significant, and the reason says which of
   the two facts it is.  Before this, "all configs are slow" and "the device
   is broken" were the same JSON.
4. **the ordering is monotone in the interval** -- `rank_tile_candidates`
   sorts by the modeled LOWER bound, not the interval midpoint: under
   overlapping intervals the midpoint order can invert (A = [100, 300]
   mid 200 loses to B = [150, 160] mid 155 although A could still be the
   cheaper config), while `lb` guarantees `ub(A) < lb(B) => A first`.  The
   invariant is pinned on injected intervals, on the twelve real shapes,
   and by the abstain-last rule that survives it.

Not pinned here, deliberately: anything device-shaped.  Every measurement
in this file is synthetic; no launch, no lock, no phone.  The canary's
reference bootstrapping (a first round establishes it) is pinned too,
because a sentinel that demanded a pre-existing reference could never
start.

Placement: `hexagon-mlir/qcom_hexagon_backend/test/test_hmx_champion_criterion_contract.py`,
picked up by `qcom_hexagon_backend/test/run_host_tests.py` in both styles.

Run:  source tools/hexmlir/env.sh
      .venv/bin/python hexagon-mlir/qcom_hexagon_backend/test/run_host_tests.py --files \
          hexagon-mlir/qcom_hexagon_backend/test/test_hmx_champion_criterion_contract.py
"""

import json

from triton.backends.qcom_hexagon_backend.hmx_search import (
    CANARY_DRIFT,
    CANARY_MISSING,
    CANARY_OK,
    CANARY_REFERENCE,
    CANARY_SHAPE,
    CANARY_TILES,
    CANARY_UNMEASURED,
    CHAMPION_CV_FACTOR,
    CHAMPION_MARGIN_FLOOR,
    REASON_OK,
    ROUND_INVALID_CANARY,
    ROUND_VALID,
    TileConfig,
    canary_config,
    canary_verdict,
    champion_gate,
    enumerate_tile_configs,
    judge_champion_round,
    predict_tile_pcyc,
    quantile_spread_cv,
    rank_tile_candidates,
    span_count,
    tile_verdict,
)

import triton.backends.qcom_hexagon_backend.hmx_search as hmx_search

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

INF = float("inf")


def _row(cfg, tiles, q50, spread=None):
    """One measured config, in the shape the sweep's timings mapping has.

    ``spread`` is the half-width of the quantile band around the median; the
    default 2% makes the interquantile CV 0.04, safely under the 15% floor so
    the geometric cases below test the FLOOR and not the CV term.
    """
    if spread is None:
        spread = 0.02 * q50
    return dict(cfg=cfg, tiles=list(tiles), q50_pcyc=q50,
                q20_pcyc=q50 - spread, q80_pcyc=q50 + spread)


def _canary(q50=5_000.0, spread=100.0):
    return _row("canary-32x32x32", CANARY_TILES, q50, spread=spread)


# --------------------------------------------------------------------------- #
# 1: the gate is the project's own, with the tie going to the incumbent
# --------------------------------------------------------------------------- #


def test_the_gate_is_the_projects_own_and_the_tie_goes_to_the_incumbent():
    """`max(3 x CV, 15%)`: both factors inherited, neither chosen here.

    The floor is the same 15% `PRUNE_MARGIN` names for the pruning
    predicate and the multiplier is the project verdict gate's own; a second
    pair of constants for the same rule is how two copies start to
    disagree.  The comparison is strict, so a delta exactly equal to the
    gate keeps the incumbent.
    """
    assert (CHAMPION_CV_FACTOR, CHAMPION_MARGIN_FLOOR) == (3.0, 0.15)
    assert champion_gate(0.0) == 0.15, "an unreadable CV leaves the floor"
    assert abs(champion_gate(0.05) - 0.15) < 1e-12, "3 x 5% == the floor"
    assert abs(champion_gate(0.10) - 0.30) < 1e-12
    assert abs(champion_gate(1.0) - 3.0) < 1e-12
    assert champion_gate(None) == 0.15, "an absent CV is not an open gate"

    # The CV the quantile triple carries is the interquantile spread over
    # the median -- a proxy that overstates sigma, which can only widen the
    # gate.  A missing band is 0, not an invented precision.
    assert quantile_spread_cv(980, 1000, 1020) == 0.04
    assert quantile_spread_cv(None, 1000, 1020) == 0.0
    assert quantile_spread_cv(980, INF, 1020) == 0.0
    assert quantile_spread_cv(0, 0, 0) == 0.0

    # Exactly at the gate: no takeover.  control 1000, challenger 850 is a
    # 15.0% delta -- equal, not greater.
    rows = [_row("ctrl", (32, 32, 32), 1000.0),
            _row("edge", (64, 32, 32), 850.0)]
    report = judge_champion_round(rows, control="ctrl",
                                  canary=_canary())
    assert report["champion"] == "ctrl", (
        "a delta exactly equal to the gate must not change the crown")
    assert report["changes"] == []
    assert report["comparisons"][0]["outcome"] == "kept"


# --------------------------------------------------------------------------- #
# 2: the crown is defended, not argmin'd
# --------------------------------------------------------------------------- #


def test_a_challenger_takes_the_crown_only_past_the_gate():
    """The walk starts at the control tile; sub-gate gains keep it.

    Control 1000.  A at 700 (30% faster) takes it.  B at 900 against the new
    incumbent A is faster than A but inside the gate -- it keeps A.  C at 640
    is the argmin of the round, yet 8.6% past A is not enough, and the crown
    stays with A.  A 3-rep argmin would have crowned C.
    """
    rows = [_row("ctrl", (32, 32, 32), 1000.0),
            _row("A", (64, 32, 32), 700.0),
            _row("B", (32, 64, 32), 900.0),
            _row("C", (128, 32, 32), 640.0)]
    report = judge_champion_round(rows, control="ctrl", iters=300,
                                  canary=_canary())

    assert report["round_verdict"] == ROUND_VALID
    assert report["champion"] == "A", report["champion"]
    assert report["champion_tiles"] == [64, 32, 32]
    assert [c["challenger"] for c in report["changes"]] == ["A"], (
        "only the past-the-gate challenger may appear in the changeover log")
    takeover = report["changes"][0]
    assert takeover["incumbent"] == "ctrl" and takeover["outcome"] == "takeover"
    assert abs(takeover["delta_pct"] - 30.0) < 1e-9
    assert abs(takeover["gate_pct"] - 15.0) < 1e-9
    assert takeover["gate_form"] == "max(3xCV, 15%)"
    # The walk visits every measured config, and the log records the ones
    # that failed to take it -- the audit trail, not just the winner.
    outcomes = {c["challenger"]: c["outcome"] for c in report["comparisons"]}
    assert outcomes == {"B": "kept", "A": "takeover", "C": "kept"}, outcomes

    # Beating the CONTROL tile is what significance means, and it is the
    # champion that is judged against it -- not the argmin.
    beats = report["beats_control"]
    assert beats["significant"] is True
    assert abs(beats["delta_pct"] - 30.0) < 1e-9
    assert report["champion_significant"] is True
    assert report["iters"] == 300, "a percentage rides with its iters_per_launch"
    assert report["walk_note"]

    # The noisier arm sets the bar: with a 10% CV the same 30% delta is only
    # equal to its 30% gate, so the crown does not move.
    noisy = [_row("ctrl", (32, 32, 32), 1000.0, spread=50.0),
             _row("A", (64, 32, 32), 700.0, spread=35.0)]
    assert abs(quantile_spread_cv(950, 1000, 1050) - 0.10) < 1e-12
    report = judge_champion_round(noisy, control="ctrl", canary=_canary())
    assert report["champion"] == "ctrl", (
        "a noisy arm must raise the gate, not lower it")
    assert abs(report["comparisons"][0]["gate_pct"] - 30.0) < 1e-9


def test_a_champion_without_a_control_arm_is_named_but_not_proven():
    """No control arm: the walk is deterministic, the verdict is not.

    The incumbent starts at the first enumerated candidate, so the order is
    reproducible; but a crown that never had to defend anything is the
    walk's survivor, and saying otherwise would be the same unfalsifiable
    claim the defect scan found.
    """
    rows = [_row("x", (32, 32, 32), 900.0), _row("y", (64, 32, 32), 500.0)]
    report = judge_champion_round(rows, canary=_canary())
    assert report["champion"] == "y"
    assert report["beats_control"] is None
    assert report["champion_significant"] is False
    assert "NOT-PROVEN" in report["significance_note"]
    assert report["significance_note"] != report["significance_note"].replace(
        "which is not the same as 'no effect'", "x"), (
        "the note must distinguish 'not proven' from 'no effect'")

    # A nominated control that never measured is its own state, distinct
    # from "no control was nominated".
    dead = rows + [_row("ctrl", (128, 128, 128), INF)]
    report = judge_champion_round(dead, control=(128, 128, 128),
                                  canary=_canary())
    assert report["control"] is None
    assert report["beats_control"] is None
    assert "never measured" in report["walk_note"]


def test_a_config_that_never_measured_is_a_per_config_fact():
    """inf is that config's failure; the round can still speak.

    This is the distinction the canary exists to keep: an unmeasured
    candidate is recorded as unmeasured and excluded from the walk -- it
    does not void the round, and it does not get to be "the champion" by
    being absent.
    """
    rows = [_row("ctrl", (32, 32, 32), 1000.0),
            _row("dead", (64, 64, 64), INF),
            _row("alive", (128, 32, 32), 700.0)]
    report = judge_champion_round(rows, control="ctrl", canary=_canary())
    assert report["unmeasured"] == ["dead"]
    assert report["champion"] == "alive"
    assert report["round_verdict"] == ROUND_VALID
    assert report["champion_significant"] is True

    empty = judge_champion_round([_row("only", (32, 32, 32), INF)],
                                 canary=_canary())
    assert empty["champion"] is None and empty["unmeasured"] == ["only"]
    assert empty["champion_significant"] is False


# --------------------------------------------------------------------------- #
# 3: the canary -- the round-level sentinel
# --------------------------------------------------------------------------- #


def test_the_canary_config_is_the_smallest_legal_probe():
    """A sentinel whose config could be illegal would indict the wrong thing.

    ``tile_verdict(32, 32, 32)`` is :data:`REASON_OK`, so a canary failure is
    never a legality question, and the shape is a CUBE so the (M, N, K) /
    (M, K, N) spelling difference between this module and the runner cannot
    alter its geometry -- the one place where a K/N mix-up would be
    undetectable in the numbers.
    """
    assert CANARY_TILES == (32, 32, 32)
    assert tile_verdict(*CANARY_TILES) == REASON_OK
    assert len(set(CANARY_SHAPE)) == 1, "the canary shape must be symmetric"
    assert list(CANARY_TILES) == canary_config()["tiles"]
    assert canary_config()["shape_mnk"] == list(CANARY_SHAPE)
    # One span of every loop: the cheapest honest probe there is.
    assert span_count(*CANARY_SHAPE, CANARY_TILES) == 1


def test_a_canary_failure_voids_the_round_instead_of_blaming_the_configs():
    """The whole point: "the device is not well" is not "all configs are slow".

    Three failure shapes, one healthy shape, and the bootstrap.  In every
    invalid case the champion is still NAMED -- it is the walk's survivor --
    but `champion_significant` is False and the note says the round is void,
    so a reader cannot mistake a broken device for a sweep result.
    """
    rows = [_row("ctrl", (32, 32, 32), 1000.0),
            _row("A", (64, 32, 32), 700.0)]
    good = judge_champion_round(rows, control="ctrl", canary=_canary())
    assert good["round_verdict"] == ROUND_VALID
    # No reference yet: the canary ESTABLISHES one, which is healthy -- a
    # sentinel that demanded a pre-existing reference could never start.
    assert good["canary"]["status"] == CANARY_REFERENCE
    assert good["canary"]["healthy"]
    assert good["champion_significant"] is True

    # With a reference in hand, the same canary is judged against it.
    judged = judge_champion_round(rows, control="ctrl", canary=_canary(),
                                  canary_reference=_canary())
    assert judged["canary"]["status"] == CANARY_OK
    assert judged["round_verdict"] == ROUND_VALID

    # Unmeasured: a known-healthy config that cannot run indicts the device.
    for label, canary in (("unmeasured", _row("canary", CANARY_TILES, INF)),
                          ("missing", None)):
        report = judge_champion_round(rows, control="ctrl", canary=canary)
        assert report["round_verdict"] == ROUND_INVALID_CANARY, label
        assert report["champion"] == "A", label
        assert report["champion_significant"] is False, label
        assert report["beats_control"]["significant"] is False, label
        assert "void" in report["significance_note"], label
        assert report["canary"]["healthy"] is False, label
    verdict = judge_champion_round(rows, control="ctrl", canary=None)
    assert verdict["canary"]["status"] == CANARY_MISSING
    assert judge_champion_round(
        rows, control="ctrl", canary=_row("canary", CANARY_TILES, INF)
    )["canary"]["status"] == CANARY_UNMEASURED

    # Drift: the same config moved more than the gate from its reference,
    # so the device state moved and every number rode with it.
    reference = _canary(5_000.0)
    drifted = _canary(8_000.0)  # +60%, gate 15%
    report = judge_champion_round(rows, control="ctrl", canary=drifted,
                                  canary_reference=reference)
    assert report["canary"]["status"] == CANARY_DRIFT
    assert report["round_verdict"] == ROUND_INVALID_CANARY
    assert report["champion_significant"] is False
    assert "device state moved" in report["canary"]["reason"]

    # ... and the same magnitude on the FAST side is also a drift: a canary
    # that got quicker means the clock state changed, not that the configs
    # improved.
    faster = _canary(3_000.0)
    verdict = canary_verdict(faster, reference)
    assert verdict["status"] == CANARY_DRIFT and not verdict["healthy"]

    # Inside the gate against a reference: healthy, with the magnitude
    # recorded so a reader can see how close it came.
    steady = _canary(5_400.0)  # +8%, inside the 15% gate
    verdict = canary_verdict(steady, reference)
    assert verdict["status"] == CANARY_OK and verdict["healthy"]
    assert abs(verdict["delta_pct"] - 8.0) < 1e-9

    # The bootstrap: with no reference the canary establishes one and the
    # round is valid -- a sentinel that demanded a pre-existing reference
    # could never start.
    verdict = canary_verdict(_canary(), None)
    assert verdict["status"] == CANARY_REFERENCE and verdict["healthy"]
    first = judge_champion_round(rows, control="ctrl", canary=_canary(),
                                 canary_reference=None)
    assert first["round_verdict"] == ROUND_VALID
    # An unmeasured reference is not a reference: a fresh one is established.
    verdict = canary_verdict(_canary(), _row("ref", CANARY_TILES, INF))
    assert verdict["status"] == CANARY_REFERENCE


def test_the_round_report_is_json_ready():
    """It goes into the search result JSON; JSON has no inf and no NaN.

    Both the valid and the invalid round serialize, with the void round's
    infinities recorded as null rather than as non-standard tokens a reader
    would have to special-case.
    """
    rows = [_row("ctrl", (32, 32, 32), 1000.0),
            _row("dead", (64, 64, 64), INF)]
    for canary, reference in ((_canary(), None),
                              (_row("canary", CANARY_TILES, INF), None),
                              (_canary(9_000.0), _canary())):
        report = judge_champion_round(rows, control="ctrl", canary=canary,
                                      canary_reference=reference, iters=42)
        text = json.dumps(report)  # raises on inf/NaN
        assert '"champion"' in text and '"canary"' in text
        assert report["kind"] == "hmx-search-champion"
        assert report["gate"]["form"] == "max(3xCV, 15%)"
        assert report["cvs"], "the per-config CVs ride with the round"


# --------------------------------------------------------------------------- #
# 4: the ordering is monotone in the interval (lb, not the midpoint)
# --------------------------------------------------------------------------- #


def _inject_predictions(intervals):
    """Pin `rank_tile_candidates`' key onto chosen [lb, ub] intervals.

    The real predictor needs the table and real geometry; what these pins
    are about is the KEY's behaviour under chosen interval shapes, so the
    predictions are injected by tiles.  Restored by the caller.
    """
    original = hmx_search.predict_tile_pcyc

    def fake(m, n, k, bm, bn, bk, **_kw):
        lb, ub, abstain = intervals[(bm, bn, bk)]
        return hmx_search.TilePrediction(
            tiles=(bm, bn, bk), lb=lb, ub=ub, p50=0.5 * (lb + ub),
            heat="hot", units=(0, 0, 0), abstain=abstain,
            table_rev="test", lb_tiers=())

    hmx_search.predict_tile_pcyc = fake
    return original


def test_the_ordering_is_monotone_in_the_interval():
    """`lb` guarantees what the midpoint cannot: disjoint order is order.

    The defect scan's own example: A = [100, 300] (midpoint 200) versus
    B = [150, 160] (midpoint 155).  Midpoint ordering puts B first and
    buries A, although A could still be the cheaper config; the lower
    bound puts A first.  The docstring's promise -- "an ordering that is
    monotone in the interval" -- is what this pins, so it cannot go back to
    being a comment nobody checks.
    """
    wide = (32, 32, 32)     # [100, 300] -- midpoint 200
    narrow = (64, 32, 32)   # [150, 160] -- midpoint 155
    far = (32, 64, 32)      # [1000, 1100] -- midpoint 1050
    intervals = {wide: (100.0, 300.0, None),
                 narrow: (150.0, 160.0, None),
                 far: (1000.0, 1100.0, None)}
    original = _inject_predictions(intervals)
    try:
        configs = [TileConfig(t, "ok") for t in (wide, narrow, far)]
        ranked = rank_tile_candidates((1024, 1024, 1024), list(configs))
        assert [c.tiles for c in ranked] == [wide, narrow, far], (
            "the order must follow the lower bound, not the midpoint")
        # Reversed input, same output: the order is a function of the SET.
        assert rank_tile_candidates((1024, 1024, 1024),
                                    list(reversed(configs))) == ranked
        # A permutation, always: nothing added, nothing lost.
        assert sorted(c.tiles for c in ranked) == sorted(intervals)
    finally:
        hmx_search.predict_tile_pcyc = original


def test_the_ordering_keeps_abstainers_last_and_ties_geometric():
    """Unknown is not evidence of cheapness, and ties stay reproducible.

    An abstaining candidate keeps its span order but sorts after every
    priced one; two candidates the model prices to the same lower bound are
    ordered by span count, then by tiles -- so "same input, same order"
    survives the tie-break too.
    """
    cheap = (32, 32, 32)      # lb 10
    tied_a = (64, 64, 64)     # lb 10, more spans
    tied_b = (128, 128, 128)  # lb 10, one span
    blind = (32, 64, 32)      # abstains, fewer spans than tied_a
    intervals = {cheap: (10.0, 20.0, None),
                 tied_a: (10.0, 40.0, None),
                 tied_b: (10.0, 15.0, None),
                 blind: (1.0, 1.0, hmx_search.ABSTAIN_HEAT_BAND)}
    original = _inject_predictions(intervals)
    try:
        shape = (1024, 1024, 1024)
        configs = [TileConfig(t, "ok") for t in intervals]
        ranked = rank_tile_candidates(shape, list(configs))
        assert [c.tiles for c in ranked][-1] == blind, (
            "an abstaining candidate must sort after every priced one")
        head = [c.tiles for c in ranked[:-1]]
        # Equal lb -> fewer spans first (tied_b, one span), then tiles.
        assert head == sorted(
            head, key=lambda t: (span_count(1024, 1024, 1024, t), t)), head
        # Determinism, and independence from the arrival order.
        assert rank_tile_candidates(shape, list(reversed(configs))) == ranked
    finally:
        hmx_search.predict_tile_pcyc = original


def test_the_lb_key_actually_moved_the_real_shape_ranking():
    """The fix is not a no-op: on real shapes lb and p50 rank differently.

    If the two keys agreed everywhere this change would be cosmetic, and
    the defect ("the promise was false in the implementation") would have
    been a doc bug only.  On the cubics they do not.
    """
    diverged = []
    for m, n, k in TWELVE_SHAPES:
        legal = [c for c in enumerate_tile_configs(m, n, k) if c.ok]
        by_lb = [c.tiles for c in rank_tile_candidates((m, n, k), legal)]
        by_p50 = sorted(legal, key=lambda c: (
            0 if predict_tile_pcyc(m, n, k, *c.tiles).abstain is None else 1,
            predict_tile_pcyc(m, n, k, *c.tiles).p50
            if predict_tile_pcyc(m, n, k, *c.tiles).abstain is None else 0.0,
            span_count(m, n, k, c.tiles), c.tiles))
        if by_lb != [c.tiles for c in by_p50]:
            diverged.append((m, n, k))
    assert diverged, (
        "the lb key ranks every real shape exactly as the midpoint did: "
        "either the ordering change is a no-op or the comparison is wrong")
    assert (1024, 1024, 1024) in diverged


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def main() -> None:
    tests = (
        test_the_gate_is_the_projects_own_and_the_tie_goes_to_the_incumbent,
        test_a_challenger_takes_the_crown_only_past_the_gate,
        test_a_champion_without_a_control_arm_is_named_but_not_proven,
        test_a_config_that_never_measured_is_a_per_config_fact,
        test_the_canary_config_is_the_smallest_legal_probe,
        test_a_canary_failure_voids_the_round_instead_of_blaming_the_configs,
        test_the_round_report_is_json_ready,
        test_the_ordering_is_monotone_in_the_interval,
        test_the_ordering_keeps_abstainers_last_and_ties_geometric,
        test_the_lb_key_actually_moved_the_real_shape_ranking,
    )
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"hmx champion-criterion contract: {len(tests)} passed")


if __name__ == "__main__":
    main()
