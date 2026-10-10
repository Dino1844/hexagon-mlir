#!/usr/bin/env python3
"""The HMX search loop's contract, pinned on the host (no device, ever).

What is being pinned
--------------------
`@triton.autotune` was unusable on this backend for one measured reason: the
default benchmarker crashes on the first config, and everything that a search
needs beyond that -- bad configs scoring inf instead of aborting, a correctness
gate that runs BEFORE timing, a cache key that rotates with the build -- is
mechanism the tree did not have.  `backend/hmx_search.py` supplies it and this
file is its gate.  Six properties, each checked against the real code:

1. the default benchmarker is still the broken one (the defect is nailed, not
   fixed -- passing `do_bench=` is what makes it irrelevant, and if someone
   "fixes" `get_benchmarker` this fails and the decision gets revisited);
2. `set_active(HexagonDriver())` is what makes the backend's own hooks
   reachable at all -- two active drivers on this host, and `driver.active`
   refuses to guess;
3. a bad config scores inf and does not stop the search (the VM-diagnostic
   RuntimeError, the launch-contract ValueError, and the device layer's
   SystemExit are each outside upstream's except tuple);
4. the rel gate short-circuits on the first rep, so a wrong config costs one
   launch rather than the whole rep budget;
5. the tile enumerator agrees with an independently written oracle on the
   twelve real shapes, and both HMX rejection paths (min-rows, tile-alignment)
   are inside that set;
6. the search space contains no knob the registry calls `auto`, and no option
   name at all -- the pins that keep a tuned launch's cache key identical to a
   plain launch's are checked against `HexagonOptions`' own defaults.

Plus the two crash-chain regressions from the investigation that motivated the
work: the `TypeError` chain (bypassed by the explicit `do_bench`) and the
`KeyError` chain (`maxnreg` / `ir_override`), and the disk-cache behaviour the
two-fingerprint key buys: a hit on the identical identity, a re-measure when the
identity moves.

Every `Autotuner` here runs against a fake launcher that prints the two lines a
real launch leaves in `perf.txt`; nothing in this file compiles a kernel for
the device, touches the DSP, or needs the phone.

Placement: `hexagon-mlir/qcom_hexagon_backend/test/test_hmx_search_loop_contract.py`,
picked up by `qcom_hexagon_backend/test/run_host_tests.py` in both styles.

Run:  source tools/hexmlir/env.sh
      .venv/bin/python hexagon-mlir/qcom_hexagon_backend/test/run_host_tests.py --files \
          hexagon-mlir/qcom_hexagon_backend/test/test_hmx_search_loop_contract.py
"""

import json
import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

# The same one-line shim every qcom host test uses.  Without it `driver.active`
# cannot be resolved at all on this host (two active drivers); with it, the
# backend's own `get_benchmarker` / `get_current_target` are the ones the
# autotuner sees.
triton.runtime.driver.set_active(HexagonDriver())

from triton.runtime.autotuner import Autotuner, Config

from triton.backends.qcom_hexagon_backend.hmx_search import (
    REASON_MIN_ROWS,
    REASON_OK,
    REASON_TILE_ALIGNMENT,
    REASON_VTCM_BUDGET,
    TileConfig,
    build_fingerprints,
    calibration_iters,
    crouton_bytes,
    enumerate_tile_configs,
    fingerprint_digest,
    hexagon_bench,
    parse_perf,
    pinned_launch_kwargs,
    tile_verdict,
    tuning_key_int,
)

from test_auto_knob_registry import (
    AUTO_KNOBS,
    C_GROUP_NOT_AUTO,
    NOT_AUTO,
    NOT_AUTO_INT,
    py_fields,
)

# The twelve shapes the user's reference tables print (table 1's nine
# bandwidth-bound GEMV-shaped cases and table 2's three cubics), transcribed in
# logs/mm-user-shapes-2026-10-09/reference-tables.md.  This is the calibration
# set the search loop is judged on: it contains M=1 and N=1 rows and the three
# cubic sizes, so both HMX refusal paths and the VTCM budget boundary all
# appear.  Not self-chosen convenient shapes.
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


# --------------------------------------------------------------------------- #
# A launcher that never reaches the device
# --------------------------------------------------------------------------- #


@triton.jit
def _identity_kernel(A, B, C, MM: tl.constexpr, NN: tl.constexpr,
                     KK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, BUILD_ID: tl.constexpr):
    """Never compiled, never launched: identity only.

    The `Autotuner` walks `fn.fn` down to a real `JITFunction` for
    `base_fn`/`cache_key`, and this supplies exactly that.  All execution is
    intercepted by `_FakeLauncher.run`.
    """
    pass


class _FakeLauncher:
    """Stands in for the JITFunction inside the Autotuner.

    Every call prints the two lines a real launch leaves in `perf.txt`, with
    timings keyed by the config so a sweep is deterministic.  Configs listed in
    `failures` raise what a real failure raises -- which is the point: the
    shapes a bad config takes are NOT the ones upstream's `_bench` catches.
    """

    def __init__(self, timings, failures=None):
        self.fn = _identity_kernel
        self._timings = dict(timings)
        self._failures = dict(failures or {})
        self.calls = []

    def run(self, *args, **kwargs):
        key = (kwargs.get("BM"), kwargs.get("BN"), kwargs.get("BK"))
        self.calls.append(key)
        if key in self._failures:
            raise self._failures[key]
        perf_us, pcyc = self._timings[key]
        print(f"Test_Info: fake\nPerf: {perf_us}\nPerfPcycles: {pcyc}")


def _configs(tiles, **pin):
    return [Config(dict(BM=bm, BN=bn, BK=bk), **pin) for bm, bn, bk in tiles]


def _autotuner(fake, configs, *, cache_results=False):
    return Autotuner(
        fake,
        ["A", "B", "C", "MM", "NN", "KK", "BM", "BN", "BK", "BUILD_ID"],
        configs,
        ["MM", "NN", "KK", "BUILD_ID"],
        None,
        None,
        do_bench=hexagon_bench,
        cache_results=cache_results,
    )


def _args(**overrides):
    base = dict(MM=256, NN=256, KK=256, BUILD_ID=7)
    base.update(overrides)
    return base


def _sweep(at, **kw):
    """One full autotuned run, with the launcher's chatter swallowed.

    The winner's final launch prints outside `hexagon_bench`'s capture, so a
    bare run would spray fake perf lines through the gate's output.
    """
    with redirect_stdout(StringIO()):
        at.run(*_three_tensors(), **_args(**kw))


# --------------------------------------------------------------------------- #
# 1 + 2: the crash chain and the shim, both nailed
# --------------------------------------------------------------------------- #


def test_the_default_benchmarker_is_still_the_broken_one():
    """The defect is pinned, not papered over.

    `HexagonDriver.get_benchmarker()` returns `triton.testing.do_bench`, whose
    first act is `torch.cpu.Event(enable_timing=True)` -- `torch.cpu.Event`
    takes no constructor arguments, so the first config dies with TypeError
    before any kernel runs.  If a future change repairs the default path (or a
    new torch grows the missing constructor), this fails and the "pass
    `do_bench=` explicitly" decision gets re-derived instead of silently
    rotting.
    """
    benchmarker = HexagonDriver().get_benchmarker()
    assert benchmarker is triton.testing.do_bench
    raised = None
    try:
        benchmarker(lambda: None, warmup=1, rep=1)
    except TypeError as exc:
        raised = exc
    assert raised is not None, (
        "the default benchmarker no longer raises TypeError: revisit whether "
        "the search loop still needs to pass do_bench= explicitly."
    )
    assert "Event" in str(raised), str(raised)


def test_the_active_driver_shim_is_what_makes_the_backend_reachable():
    """Two active drivers on this host; `driver.active` refuses to guess.

    `HexagonDriver.is_active()` is unconditionally true and the CUDA driver
    also claims activeness, so on this host `driver.active` raises "2 active
    drivers" until something picks one.  That is why every qcom host test opens
    with `set_active(HexagonDriver())`, and why the disk cache's
    `make_backend(driver.active.get_current_target()).hash()` -- which
    `check_disk_cache` calls -- is reachable at all.
    """
    assert isinstance(triton.runtime.driver.active, HexagonDriver)
    assert HexagonDriver().get_current_target().backend == "hexagon"

    saved = triton.runtime.driver.active
    try:
        # Undo the shim the way a process that never made it would look:
        # no active driver picked, so `.active` must resolve through
        # `_create_driver()`.
        triton.runtime.driver._active = None
        try:
            triton.runtime.driver.active
        except RuntimeError as exc:
            assert "active drivers" in str(exc), str(exc)
        else:
            raise AssertionError(
                "this host now has a single active driver: the shim is a "
                "no-op here and this test's premise must be re-derived."
            )
    finally:
        triton.runtime.driver.set_active(saved)
    assert isinstance(triton.runtime.driver.active, HexagonDriver)


# --------------------------------------------------------------------------- #
# 3: bad configs score inf
# --------------------------------------------------------------------------- #


def test_a_bad_config_scores_inf_instead_of_stopping_the_search():
    """Every failure shape a device run has actually taken becomes inf.

    Upstream's `_bench` catches only OutOfResources / CompileTimeAssertion-
    Failure / PTXASError.  What this backend raises instead: an MLIR diagnostic
    for a VTCM overflow (RuntimeError), `enforce_hmx_launch_contract`'s
    ValueError, and -- observed in the 2026-10-09 sweep -- the device layer's
    SystemExit(1), which is not even an `Exception`.  A launch whose report has
    no `PerfPcycles` line is equally unusable.  All of them must be inf.
    """
    good = (10.0, 20_000)
    pin = pinned_launch_kwargs()
    cases = {
        "vtcm overflow (MLIR diagnostic)":
            RuntimeError("bridge footprint does not fit remaining VTCM"),
        "launch contract": ValueError("grid>1 with grid_policy=single-instance"),
        "device layer gave up": SystemExit(1),
        "unrecognised kwarg": KeyError("maxnreg"),
    }
    for label, failure in cases.items():
        fake = _FakeLauncher({(64, 64, 64): good},
                             failures={(32, 32, 32): failure})
        at = _autotuner(fake, _configs([(32, 32, 32), (64, 64, 64)], **pin))
        _sweep(at)
        broken = Config(dict(BM=32, BN=32, BK=32), **pin)
        assert at.configs_timings[broken] == [INF, INF, INF], label
        assert at.best_config.kwargs == {"BM": 64, "BN": 64, "BK": 64}, label

    silent = _FakeLauncher({(32, 32, 32): good})
    silent.run = lambda *a, **k: print("Test_Info: fake\nPerf: 10.0")
    at = _autotuner(silent, _configs([(32, 32, 32), (64, 64, 64)], **pin))
    _sweep(at)
    broken = Config(dict(BM=32, BN=32, BK=32), **pin)
    assert at.configs_timings[broken] == [INF, INF, INF], (
        "a launch with no pcycles must score inf"
    )

    # And the counter-case: the same machinery with a well-behaved launch must
    # produce finite quantiles from the reps, computed in pcyc.
    fake = _FakeLauncher({(32, 32, 32): (10.0, 20_000),
                          (64, 64, 64): (50.0, 90_000)})
    at = _autotuner(fake, _configs([(32, 32, 32), (64, 64, 64)], **pin))
    _sweep(at)
    fine = Config(dict(BM=32, BN=32, BK=32), **pin)
    assert at.configs_timings[fine] == [20_000.0] * 3
    assert at.best_config.kwargs == fine.kwargs
    assert len(fake.calls) == 2 * 3 + 1, "reps=3 per config, plus the winner"


# --------------------------------------------------------------------------- #
# 4: the rel gate runs before the timing matters
# --------------------------------------------------------------------------- #


def test_the_rel_gate_short_circuits_before_the_reps_are_spent():
    """A wrong config costs one launch, not the whole rep budget.

    The correctness gate is inside the search, not after it: `verify` is
    consulted after the first rep, and a failure (or an unreadable output)
    returns inf immediately.  A search that kept timing a config producing
    garbage would be measuring how fast it is to be wrong.
    """
    seen = []

    def verify():
        seen.append(1)
        return False

    fake = _FakeLauncher({(32, 32, 32): (10.0, 20_000)})
    launch = lambda: fake.run(BM=32, BN=32, BK=32)  # noqa: E731
    out = hexagon_bench(launch, reps=3, verify=verify)
    assert out == [INF, INF, INF]
    assert len(fake.calls) == 1, "the gate must fire on the first rep"
    assert seen == [1]

    def verify_raises():
        raise FileNotFoundError("no output pulled")

    fake = _FakeLauncher({(32, 32, 32): (10.0, 20_000)})
    launch = lambda: fake.run(BM=32, BN=32, BK=32)  # noqa: E731
    out = hexagon_bench(launch, reps=3, verify=verify_raises)
    assert out == [INF, INF, INF], "an unreadable output is not a pass"

    # A passing gate still runs every rep and still reports that run's own
    # cycles: three distinct runs, three distinct numbers, quantiles in order.
    fake = _FakeLauncher({(32, 32, 32): (10.0, 20_000)})
    calls = {"n": 0}

    def drifting_launch():
        calls["n"] += 1
        print(f"Perf: 10.0\nPerfPcycles: {1000 * calls['n']}")

    out = hexagon_bench(drifting_launch, reps=3, verify=lambda: True)
    assert out == [2000.0, 1400.0, 2600.0], out


# --------------------------------------------------------------------------- #
# 5: the enumerator against an independent oracle
# --------------------------------------------------------------------------- #


def _oracle_verdict(bm, bn, bk):
    """The same rules, written differently on purpose.

    Independent formulation of the L0 predicate: one helper per rule, each
    returning the reason it refuses with, and a pipeline order constant that
    spells out which refusal wins.  If the enumerator and this oracle ever
    disagree, one of them misread `HmxTarget` -- and the search-loop drill
    (which compiles real kernels and reads the manifest's own reason strings)
    is the third opinion.
    """
    def power_of_two(x):
        return 2 ** int(round(__import__("math").log2(x))) == x

    def frontend_rules():
        for dim in (bm, bn, bk):
            if not power_of_two(dim):
                return "frontend-block-limit"
        if max(bm * bk, bk * bn, bm * bn) > 1048576:
            return "frontend-block-limit"
        return None

    order = (
        ("frontend-block-limit", frontend_rules),
        ("min-rows", lambda: "min-rows" if bm <= 4 else None),
        ("tile-alignment",
         lambda: "tile-alignment" if (bm % 32 or bn % 32 or bk % 32) else None),
        ("vtcm-budget",
         lambda: "vtcm-budget"
         if (bm * bk + bk * bn + bm * bn) * 2 >= 8 * 1024 * 1024 else None),
    )
    for _reason, rule in order:
        refusal = rule()
        if refusal is not None:
            return refusal
    return REASON_OK


def test_enumerator_matches_an_independent_oracle_on_the_twelve_shapes():
    for m, n, k in TWELVE_SHAPES:
        configs = enumerate_tile_configs(m, n, k)
        assert configs, (m, n, k)
        for cfg in configs:
            expected = _oracle_verdict(*cfg.tiles)
            assert cfg.verdict == expected, (
                f"shape {(m, n, k)} tile {cfg.tiles}: enumerator says "
                f"{cfg.verdict!r}, oracle says {expected!r}"
            )
        # Sorted, so two runs enumerate identically.
        assert configs == sorted(configs)

    verdicts = {
        cfg.verdict
        for m, n, k in TWELVE_SHAPES
        for cfg in enumerate_tile_configs(m, n, k)
    }
    # The design's requirement that both HMX refusal paths are inside the
    # twelve-shape set.  (The vtcm-budget path is deliberately NOT expected
    # here and cannot be -- see the next test.)
    assert REASON_MIN_ROWS in verdicts, "an M<=4 candidate must be enumerated"
    assert REASON_TILE_ALIGNMENT in verdicts, (
        "an off-grid candidate must be enumerated"
    )
    assert REASON_OK in verdicts

    # The degenerate whole-extent candidate is what puts the two refusal paths
    # in the space at all: M=1 offers BM=1 (min-rows) and N=1 offers BN=1
    # (tile-alignment).
    assert tile_verdict(1, 128, 8192) == REASON_MIN_ROWS
    assert tile_verdict(128, 1, 8192) == REASON_TILE_ALIGNMENT
    assert tile_verdict(32, 32, 32) == REASON_OK


def test_the_budget_gate_cannot_bite_on_an_empty_pool():
    """A measured property of the lattice, not an oversight.

    The front end caps every operand block at 2^20 elements, so for any tile it
    accepts, ``croutonBytes`` is at most 3 * 2^20 * 2 = 6 MiB -- below the 8 MiB
    budget ``planBridge`` starts from.  So the vtcm-budget refusal is
    unreachable through this enumerator with an empty pool, and it is NOT
    claimed to appear in the twelve-shape set.  What it does bite on is a
    partially-used pool, which is exactly how the real pass sees it: the
    enumerator's budget parameter is that ``vtcmUsed``, and the manifest read
    after a compile is authoritative for the real number.
    """
    assert tile_verdict(1024, 1024, 1024) == REASON_OK  # 6.0 MiB < 8 MiB
    assert tile_verdict(1024, 1024, 1024, vtcm_budget=4 * 1024 * 1024) == (
        REASON_VTCM_BUDGET
    )
    # The boundary is planBridge's strict <: exactly the budget is refused.
    assert tile_verdict(32, 32, 32, vtcm_budget=crouton_bytes(32, 32, 32)) == (
        REASON_VTCM_BUDGET
    )
    assert tile_verdict(32, 32, 32,
                        vtcm_budget=crouton_bytes(32, 32, 32) + 1) == REASON_OK


def test_the_calibration_box_time_boxes_a_launch():
    # The fixed-count failure this replaces: one slow config took seconds per
    # iteration and ended the whole sweep.  A 10 ms/iter shape gets the top of
    # the box; a 2.6 s/iter shape gets the floor; the floor is not zero.
    assert calibration_iters(10.0) == 1000
    assert calibration_iters(2_600_000.0) == 5
    assert calibration_iters(0.0) == 1000, "an unreadable ms must not divide"


# --------------------------------------------------------------------------- #
# 6: the search space excludes every auto knob
# --------------------------------------------------------------------------- #


def test_the_search_space_excludes_every_auto_knob():
    """The machine gate for "already-auto knobs are not search dimensions".

    `test_auto_knob_registry.py`'s registries are the single place that says
    which knobs the compiler decides; this asserts the search space and those
    registries are disjoint, so promoting a knob to `auto` and adding it to the
    search space cannot both happen without one of the two failing here.
    """
    tunables = set(TileConfig.tunables())
    registered = (set(AUTO_KNOBS) | set(NOT_AUTO) | set(NOT_AUTO_INT)
                  | set(C_GROUP_NOT_AUTO))
    assert tunables, "the search space must declare its variables"
    overlap = tunables & registered
    assert not overlap, (
        f"search dimensions {sorted(overlap)} are registered knobs: a knob the "
        "compiler can decide is not a search dimension."
    )
    # No option name anywhere in the config representation, not just the
    # declared tunables.
    option_fields = set(py_fields())
    assert not (tunables & option_fields)
    assert set(TileConfig._fields) == {"tiles", "verdict"}
    for m, n, k in TWELVE_SHAPES:
        for cfg in enumerate_tile_configs(m, n, k):
            assert cfg.verdict in (REASON_OK, REASON_MIN_ROWS,
                                   REASON_TILE_ALIGNMENT, REASON_VTCM_BUDGET,
                                   "frontend-block-limit")

    # The launch pins keep a tuned launch's options dict -- and therefore its
    # compile cache key -- identical to a plain launch's: they must be the
    # backend's own defaults, and they must not be search dimensions.
    pins = pinned_launch_kwargs()
    assert set(pins) == {"num_warps", "num_stages", "num_ctas"}
    assert not (set(pins) & tunables)
    from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions

    defaults = HexagonOptions().__dict__
    for name, value in pins.items():
        assert defaults[name] == value, (
            f"{name} pinned to {value} but the backend default is "
            f"{defaults[name]}: a tuned launch would compile differently "
            "from a plain one for no effect."
        )


# --------------------------------------------------------------------------- #
# 7: the crash-chain regressions from the investigation
# --------------------------------------------------------------------------- #


def test_an_unknown_option_keyword_still_dies_with_keyerror():
    """`maxnreg` / `ir_override` must KeyError -- and must never be emitted.

    `HexagonOptions` has no such fields, and `JITFunction._pack_args` raises
    KeyError for a keyword that is neither an option nor a signature name.  The
    trap is real and stays pinned; what the search loop does about it is emit
    neither.
    """
    import pytest

    small = torch.zeros(4, 4, dtype=torch.float16)
    for kwarg in ({"maxnreg": 64}, {"ir_override": "kernel.llir"}):
        with pytest.raises(KeyError):
            _identity_kernel.warmup(small, small, small, 4, 4, 4, 4, 4, 4, 1,
                                    grid=(1,), **kwarg)

    cfg = Config(dict(BM=32, BN=32, BK=32), **pinned_launch_kwargs())
    kwargs = cfg.all_kwargs()
    assert "maxnreg" not in kwargs and "ir_override" not in kwargs, kwargs
    assert kwargs["num_warps"] == 1 and kwargs["num_stages"] == 1
    assert kwargs["num_ctas"] == 1


def test_the_explicit_do_bench_bypasses_the_typeerror_chain():
    """The fix, and only the fix: `do_bench=` is honoured end to end.

    A full `Autotuner` run against the fake launcher benches every config
    through `hexagon_bench` and returns a finite winner -- no TypeError from
    the default path, because the default path is not taken.  The cached
    property is asserted too: a future `get_benchmarker` change cannot silently
    re-enter this sweep.
    """
    fake = _FakeLauncher({(32, 32, 32): (9.0, 20_000),
                          (64, 64, 64): (5.0, 90_000)})
    at = _autotuner(fake, _configs([(32, 32, 32), (64, 64, 64)],
                                   **pinned_launch_kwargs()))
    assert at.do_bench is hexagon_bench

    _sweep(at)
    # 2 configs x 3 reps, plus the winner's final launch.
    assert len(fake.calls) == 2 * 3 + 1, fake.calls
    assert at.best_config.kwargs == {"BM": 32, "BN": 32, "BK": 32}, (
        "the winner must be the faster pcyc config (20k < 90k)"
    )
    assert at.configs_timings, "the sweep must record every config's timing"


# --------------------------------------------------------------------------- #
# 8: the disk cache and the two-fingerprint key
# --------------------------------------------------------------------------- #


def test_the_tuning_key_rotates_on_build_and_protocol():
    """One identity, two inputs: the build and the iteration count.

    Neither is visible to upstream's cache key -- the backend hashes target and
    options only, and the runtime library is in neither -- so both ride in the
    integer the runner passes as a kernel constexpr and an autotune key member.
    Anything less and a rebuilt runtime would inherit the previous build's
    winner, or a re-run at a different iteration count would inherit a timing
    measured under a different protocol.
    """
    first = build_fingerprints()
    assert set(first) == {"libtriton.so", "libhmxapi.a"}, sorted(first)
    digest = fingerprint_digest(first)
    assert digest == fingerprint_digest(first), "the digest must be stable"
    assert digest != fingerprint_digest({}), "and not degenerate"

    same_build = tuning_key_int(first, iters=300)
    assert same_build == tuning_key_int(first, iters=300)
    assert same_build != tuning_key_int(first, iters=301), (
        "a different iteration count is a different experiment"
    )
    forged = dict(first)
    forged["libhmxapi.a"] = dict(forged["libhmxapi.a"])
    forged["libhmxapi.a"]["md5"] = "0" * 32
    assert same_build != tuning_key_int(forged, iters=300), (
        "a rebuilt runtime must not inherit the previous build's verdict"
    )


def test_disk_cache_hits_once_and_re_measures_when_the_identity_moves():
    """The cache the design reuses, with the identity check in front of it.

    Same shape, same configs, same identity: the second run reads the file and
    launches nothing.  Move the identity -- which is what a rebuilt runtime or
    a re-calibrated iteration count does -- and it re-measures instead of
    answering with the previous build's numbers.
    """
    timings = {(32, 32, 32): (9.0, 20_000), (64, 64, 64): (5.0, 90_000)}
    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = tmp
        try:
            fake = _FakeLauncher(timings)
            at = _autotuner(fake, _configs([(32, 32, 32), (64, 64, 64)],
                                           **pinned_launch_kwargs()),
                            cache_results=True)
            _sweep(at)
            first_calls = len(fake.calls)
            assert first_calls == 2 * 3 + 1
            wrote = [os.path.join(root, name)
                     for root, _dirs, files in os.walk(tmp) for name in files
                     if name.endswith(".autotune.json")]
            assert wrote, "the sweep must persist its timings"
            record = json.loads(open(wrote[0]).read())
            assert record["key"], "the persisted key must carry the identity"

            again = _FakeLauncher(timings)
            at2 = _autotuner(again, _configs([(32, 32, 32), (64, 64, 64)],
                                             **pinned_launch_kwargs()),
                             cache_results=True)
            _sweep(at2)
            # One call, not seven: the winner's final launch only -- zero
            # benchmark launches, which is what "hit the cache" means.
            assert len(again.calls) == 1, (
                "an identical identity must be a cache hit, not a re-measure"
            )
            assert at2.best_config.kwargs == at.best_config.kwargs

            moved = _FakeLauncher(timings)
            at3 = _autotuner(moved, _configs([(32, 32, 32), (64, 64, 64)],
                                             **pinned_launch_kwargs()),
                             cache_results=True)
            _sweep(at3, BUILD_ID=8)
            assert moved.calls, "a new identity must re-measure"
        finally:
            if previous is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _three_tensors():
    z = torch.zeros(8, 8, dtype=torch.float16)
    return (z, z, z)


def main() -> None:
    tests = (
        test_the_default_benchmarker_is_still_the_broken_one,
        test_the_active_driver_shim_is_what_makes_the_backend_reachable,
        test_a_bad_config_scores_inf_instead_of_stopping_the_search,
        test_the_rel_gate_short_circuits_before_the_reps_are_spent,
        test_enumerator_matches_an_independent_oracle_on_the_twelve_shapes,
        test_the_budget_gate_cannot_bite_on_an_empty_pool,
        test_the_calibration_box_time_boxes_a_launch,
        test_the_search_space_excludes_every_auto_knob,
        test_an_unknown_option_keyword_still_dies_with_keyerror,
        test_the_explicit_do_bench_bypasses_the_typeerror_chain,
        test_the_tuning_key_rotates_on_build_and_protocol,
        test_disk_cache_hits_once_and_re_measures_when_the_identity_moves,
    )
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"hmx search-loop contract: {len(tests)} passed")


if __name__ == "__main__":
    main()
