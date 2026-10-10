# ===- hmx_search.py -------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""The reusable halves of the HMX search loop: a device-pcyc benchmarker, a
tile-config enumerator, the cost table's C0 consumer, and the build/protocol
identity that keys its cache.

Why this module exists
----------------------
`@triton.autotune` was dead on arrival for this backend: its default benchmarker
is `HexagonDriver.get_benchmarker()` -> `triton.testing.do_bench`, whose first
line asks `torch.cpu.Event(enable_timing=True)` and dies with ``TypeError``.
`Autotuner._bench` only catches ``OutOfResources`` /
``CompileTimeAssertionFailure`` / ``PTXASError``, so that first config aborted
the whole sweep.  The crash chain is nailed (not fixed) by
``test/test_hmx_search_loop_contract.py``; this module supplies the two pieces
that make the explicit ``do_bench=`` path work instead:

* :func:`hexagon_bench` -- the benchmarker.  One ``kernel_call()`` is one
  ``JITFunction.run`` is one launch; the device wrapper's
  ``benchmark_time_and_pcycles`` already averages BOTH counters over the
  ``iterations`` loop in a single pass, and the launcher appends
  ``PerfPcycles:<n>`` to ``perf.txt``, which the executor prints whole.  So one
  launch yields that run's own per-iteration microseconds and processor cycles
  -- and a caller-supplied ``verify()`` can read the pulled output buffer from
  the same launch.  Clock constants are deliberately NOT used: C15:14 stops
  while the DSP is clock-gated while the qtimer behind ``Perf`` does not, so the
  ratio between them is a property of the run (measured 2.10-2.15 GHz in a
  tight leaf loop, 1.10-1.15 GHz inside a full matmul on one build).
* :func:`enumerate_tile_configs` -- the L0 legality predicate for matmul tiles,
  mirroring ``HmxTarget`` (``tileEdge``/``minRows``/``croutonBytes``/
  ``planBridge``'s budget) plus the Triton front-end's block limits.  Illegal
  configs are rejected on the host, before anything is pushed to the device.
* :func:`predict_tile_pcyc` / :func:`cost_prune` / :func:`rank_tile_candidates`
  -- the leaf cost table's C0 consumer (design:
  ``docs/hmx/cost-table-revival-for-search-2026-10-10.md``).  C0 ORDERS
  candidates by modeled LOWER bound and prunes NOTHING: the pruning
  predicate is written and pinned here but has no call site until C1, and
  the runner switch that enables even the ordering defaults to OFF.  The
  ordering sorts by ``lb``, not by the interval midpoint, so it is monotone
  in the interval -- if every cell of A is cheaper than every cell of B,
  A sorts first -- which is the property the midpoint cannot promise under
  overlapping intervals.  What C0 does produce every run is the calibration
  cross table ("constants vs this run's measurements", version-stamped by
  ``table_rev``) -- a report, never a table edit.
* :func:`judge_champion_round` / :func:`canary_verdict` -- the verdict the
  search's runner promised in its own docstring and never implemented: a
  challenger takes the crown only by beating the incumbent by more than
  this project's gate ``max(3xCV, 15%)``, and a known-healthy canary config
  re-measured every round decides whether the round's numbers may be read
  at all -- "the device is not well" is a different fact from "every
  config was slow", and without the sentinel the two are the same number.
  Both land in the runner's result JSON as the changeover log, the
  per-config CVs and the canary status.

What is deliberately NOT here
-----------------------------
* No new ``HexagonOptions`` field, no new pass, no new compile-time constant.
  Everything tunable is a ``tl.constexpr`` the kernel already has, so adding a
  search dimension costs one file, not four passes.
* No ``num_warps``/``num_stages``/``num_ctas`` in the search space: the backend
  never branches on them (they are launch-metadata fields), so they cannot be
  searched.  :func:`pinned_launch_kwargs` exists to keep ``Config``'s defaults
  (4/3/1) from diverging from the backend's (1/1/1) and splitting the compile
  cache key for no effect.
* Nothing device-touching.  This module never launches; the caller's
  ``kernel_call`` does.

Run:  ::

    from triton.backends.qcom_hexagon_backend.hmx_search import (
        hexagon_bench, enumerate_tile_configs, tuning_key_int,
        predict_tile_pcyc, cost_prune, rank_tile_candidates,
        build_cost_calibration, judge_champion_round, canary_verdict,
    )
"""

from __future__ import annotations

import functools
import hashlib
import io
import os
import re
import statistics
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence

__all__ = [
    "TILE_EDGE",
    "MIN_ROWS",
    "DEFAULT_VTCM_BUDGET",
    "CROUTON_ELEM_BYTES",
    "FRONTEND_MAX_BLOCK_NUMEL",
    "REASON_OK",
    "REASON_MIN_ROWS",
    "REASON_TILE_ALIGNMENT",
    "REASON_VTCM_BUDGET",
    "REASON_FRONTEND_BLOCK_LIMIT",
    "TileConfig",
    "crouton_bytes",
    "is_power_of_two",
    "next_power_of_two",
    "tile_ladder",
    "tile_verdict",
    "enumerate_tile_configs",
    "pinned_launch_kwargs",
    "build_fingerprints",
    "fingerprint_digest",
    "tuning_key_int",
    "parse_perf",
    "hexagon_bench",
    "calibration_iters",
    "span_count",
    "table_revision",
    "load_cost_table",
    "cell_tier",
    "CostTableIndex",
    "TilePrediction",
    "predict_tile_pcyc",
    "cost_prune",
    "rank_tile_candidates",
    "manifest_abstains",
    "build_cost_calibration",
    "render_cost_calibration_md",
    "COST_ENVELOPE",
    "ENVELOPE_VERSION",
    "PRUNE_MARGIN",
    "ABSTAIN_HEAT_BAND",
    "ABSTAIN_HEAT_CROSS",
    "ABSTAIN_TAIL",
    "ABSTAIN_DTYPE",
    "ABSTAIN_MANIFEST",
    "ABSTAIN_NO_CELL",
    "ABSTAIN_TABLE_MISSING",
    "CHAMPION_CV_FACTOR",
    "CHAMPION_MARGIN_FLOOR",
    "champion_gate",
    "quantile_spread_cv",
    "CANARY_MISSING",
    "CANARY_UNMEASURED",
    "CANARY_DRIFT",
    "CANARY_REFERENCE",
    "CANARY_OK",
    "ROUND_VALID",
    "ROUND_INVALID_CANARY",
    "CANARY_TILES",
    "CANARY_SHAPE",
    "canary_config",
    "canary_verdict",
    "judge_champion_round",
]

# --------------------------------------------------------------------------- #
# Constants.  Each one mirrors a single source of truth; none is a new number
# invented here.  The mirror is checked against the real compiler by the
# search-loop drill (exp/hmx/op_bench/mm_autotune_search.py) and against an
# independent oracle by the contract test.
# --------------------------------------------------------------------------- #

#: ``HmxTarget::tileEdge`` (``layout::kTileEdge``): the engine's crouton
#: geometry.  A contraction whose extent is not a multiple of it takes the tail
#: path; ``tl.arange`` additionally demands a power of two, so the legal tiles
#: are powers of two >= 32.
TILE_EDGE = 32

#: ``HmxTarget::minRows``: a capability floor. ``queryContraction`` checks it
#: BEFORE it splits extents, so an M of 4 or fewer is refused as ``min-rows``
#: and never reaches the alignment question.
MIN_ROWS = 4

#: ``HmxTarget::defaultVtcmBudget`` -- 8 MiB, queried from the device, not a
#: preference.  ``vtcmUsed`` is unknown to a pure predicate, so the enumerator
#: assumes an empty pool and the manifest (read after a real compile) is the
#: authoritative answer.
DEFAULT_VTCM_BUDGET = 8 * 1024 * 1024

#: ``HmxTarget::croutonElemBytes``: the crouton element is fp16 whatever the
#: source element type is.
CROUTON_ELEM_BYTES = 2

#: Triton's front-end block limit (``triton/_utils.py``'s
#: ``TRITON_MAX_TENSOR_NUMEL``, enforced by ``validate_block_shape`` on every
#: block-shaped tensor): each block dimension a power of two and the product
#: at most this.  A tile pair whose operand product exceeds it dies in the
#: Python tracer, before any compiler question is asked.
FRONTEND_MAX_BLOCK_NUMEL = 1048576

#: Canonical rejection vocabulary.  The first three are the manifest's own
#: reason strings (``HmxManifest.h``'s ``kHmxReason*``), so a host-side verdict
#: and a compile-side manifest record can be compared word for word.
REASON_OK = "ok"
REASON_MIN_ROWS = "min-rows"
REASON_TILE_ALIGNMENT = "tile-alignment"
REASON_VTCM_BUDGET = "vtcm-budget"
#: Deliberately NOT a manifest reason: the front end refuses before the
#: compiler emits a manifest, so no reason string exists to mirror.
REASON_FRONTEND_BLOCK_LIMIT = "frontend-block-limit"


# --------------------------------------------------------------------------- #
# The L0 predicate
# --------------------------------------------------------------------------- #


def crouton_bytes(rows: int, n: int, k: int) -> int:
    """``HmxTarget::croutonBytes``: the crouton footprint of one bridge block.

    Both operands plus the engine's fp16 read-out, in bytes.
    """
    return (rows * k + k * n + rows * n) * CROUTON_ELEM_BYTES


def span_count(m: int, n: int, k: int, tiles: tuple[int, int, int]) -> int:
    """ceil(M/BM)*ceil(N/BN)*ceil(K/BK): the engine spans one config costs.

    The search-loop design's L1 objective, used today only to ORDER a bounded
    selection.  It counts blocks, not bytes and not heat: a BK=2048 single
    span and a BK=64 32-span config differ 32x in spans while potentially
    differing by one "once per K segment" tax in cycles -- which is exactly
    the blind spot :func:`predict_tile_pcyc` was added to price (C0 orders by
    the modeled lower bound, with this as the tiebreak).
    """
    bm, bn, bk = tiles
    return ((m + bm - 1) // bm) * ((n + bn - 1) // bn) * ((k + bk - 1) // bk)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def tile_ladder(extent: int) -> tuple[int, ...]:
    """The legal tile values along one dimension, smallest first.

    Powers of two from the tile edge up to the padded extent -- a tile wider
    than the padded extent computes only masked lanes, so the lattice stops
    there -- plus the extent itself when it is a power of two.  That last
    candidate is the whole-block ("degenerate") form, and it is what puts the
    ``min-rows`` and ``tile-alignment`` rejection paths inside the enumerated
    space: a shape with M=1 offers BM=1, a shape with N=1 offers BN=1, and both
    are refused for reasons the manifest vocabulary already names.
    """
    extent = int(extent)
    top = max(TILE_EDGE, next_power_of_two(extent))
    values = []
    rung = TILE_EDGE
    while rung <= top:
        values.append(rung)
        rung *= 2
    if is_power_of_two(extent) and extent not in values:
        values.append(extent)
    return tuple(sorted(set(values)))


def tile_verdict(bm: int, bn: int, bk: int,
                 vtcm_budget: int = DEFAULT_VTCM_BUDGET) -> str:
    """One tile candidate's verdict: :data:`REASON_OK` or a rejection reason.

    The order is the pipeline's order, not an arbitrary one: the Python tracer
    refuses a block shape before the compiler exists, then
    ``HmxTarget::queryContraction`` checks ``minRows`` before it splits
    extents, and only then does ``planBridge`` weigh the VTCM budget (a whole
    contraction that fits wins; the strict ``<`` is the boundary the pass's own
    budget remark is written against).
    """
    if not all(is_power_of_two(d) for d in (bm, bn, bk)):
        return REASON_FRONTEND_BLOCK_LIMIT
    if (bm * bk > FRONTEND_MAX_BLOCK_NUMEL
            or bk * bn > FRONTEND_MAX_BLOCK_NUMEL
            or bm * bn > FRONTEND_MAX_BLOCK_NUMEL):
        return REASON_FRONTEND_BLOCK_LIMIT
    if bm <= MIN_ROWS:
        return REASON_MIN_ROWS
    if bm % TILE_EDGE or bn % TILE_EDGE or bk % TILE_EDGE:
        return REASON_TILE_ALIGNMENT
    if crouton_bytes(bm, bn, bk) >= vtcm_budget:
        return REASON_VTCM_BUDGET
    return REASON_OK


class TileConfig(NamedTuple):
    """One candidate: its tile, and why it is (in)eligible.

    ``tunables()`` is the search-space declaration the contract test checks
    against the auto-knob registry: the variables a search over these configs
    moves are exactly the tile dimensions -- nothing else.
    """

    tiles: tuple[int, int, int]
    verdict: str

    @property
    def ok(self) -> bool:
        return self.verdict == REASON_OK

    @classmethod
    def tunables(cls) -> tuple[str, ...]:
        return ("BM", "BN", "BK")


def enumerate_tile_configs(m: int, n: int, k: int,
                           vtcm_budget: int = DEFAULT_VTCM_BUDGET
                           ) -> list[TileConfig]:
    """Every legal-and-illegal tile candidate for one contraction shape.

    The result is the L0 layer of the design's three pruning layers: legality
    only.  It does not rank (that is the L1 span objective) and it does not
    price leaves (L2, whose table is diagnostic-only by design).  Sorted by
    tile so two runs enumerate identically.
    """
    configs = [
        TileConfig(tiles=(bm, bn, bk),
                   verdict=tile_verdict(bm, bn, bk, vtcm_budget))
        for bm in tile_ladder(m)
        for bn in tile_ladder(n)
        for bk in tile_ladder(k)
    ]
    return sorted(configs)


def pinned_launch_kwargs() -> dict[str, int]:
    """The launch parameters ``Config`` must carry to stay invisible.

    ``Config`` injects ``num_warps=4, num_stages=3, num_ctas=1`` by default,
    while the backend's own defaults are 1/1/1.  The backend never branches on
    any of them, so pinning them to the backend defaults is what keeps an
    autotuned launch's options dict -- and therefore its compile cache key --
    identical to a plain launch's.  ``maxnreg`` and ``ir_override`` are absent
    on purpose: ``HexagonOptions`` has no such fields and ``_pack_args`` raises
    ``KeyError`` for a keyword that is neither an option nor a signature name.
    """
    return {"num_warps": 1, "num_stages": 1, "num_ctas": 1}


# --------------------------------------------------------------------------- #
# Build/protocol identity
# --------------------------------------------------------------------------- #

#: The two artifacts whose identity a measurement claim depends on.  The
#: compiled backend is NOT part of any upstream cache key (``HexagonBackend``
#: hashes target + resolved options only), and the runtime static library is
#: not either -- so a rebuilt ``libhmxapi.a`` could otherwise reuse a stale
#: autotune verdict.  ``tools/hexmlir/env.sh`` splits ``TRITON_CACHE_DIR`` by
#: ``libtriton.so``'s (mtime, size); this is the same discipline, applied to
#: both files at once and inside the key itself.
_REPO = Path(__file__).resolve().parents[2]
_LIBTRITON = _REPO / "triton" / "python" / "triton" / "_C" / "libtriton.so"


def _libhmxapi() -> Optional[Path]:
    builds = sorted(
        (_REPO / "triton" / "build").glob(
            "cmake.linux-x86_64-cpython-*/third_party/qcom_hexagon_backend"
            "/bin/runtime/libhmxapi.a"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )
    return builds[-1] if builds else None


def build_fingerprints() -> dict[str, dict[str, object]]:
    """md5 + mtime_ns + size for the two build artifacts, as plain data.

    A missing artifact is recorded as MISSING rather than skipped: a
    fingerprint that silently drops half of its inputs would claim an identity
    it does not have.
    """
    paths = [("libtriton.so", _LIBTRITON)]
    hmx = _libhmxapi()
    if hmx is not None:
        paths.append(("libhmxapi.a", hmx))
    out: dict[str, dict[str, object]] = {}
    for label, path in paths:
        if not path.exists():
            out[label] = {"path": str(path), "md5": "MISSING",
                          "mtime_ns": 0, "size": 0}
            continue
        digest = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        st = path.stat()
        out[label] = {"path": str(path), "md5": digest.hexdigest(),
                      "mtime_ns": st.st_mtime_ns, "size": st.st_size}
    return out


def fingerprint_digest(fingerprints: Optional[dict] = None) -> str:
    """A stable digest over both fingerprints, or "no-fingerprints"."""
    if fingerprints is None:
        fingerprints = build_fingerprints()
    if not fingerprints:
        return "no-fingerprints"
    canonical = ";".join(
        f"{label}={fingerprints[label]['md5']}"
        f"@{fingerprints[label]['mtime_ns']}"
        f":{fingerprints[label]['size']}"
        for label in sorted(fingerprints)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tuning_key_int(fingerprints: Optional[dict] = None, *,
                   iters: Optional[int] = None) -> int:
    """The identity a search's cache key must carry, as one small integer.

    Two things rotate it, and both must:

    * **the build** -- a rebuilt runtime or backend makes every recorded
      timing cross-build, and cross-build numbers are not comparable (the
      reason ``env.sh`` splits the kernel cache at all); and
    * **the measurement protocol** -- the per-shape iteration count is a
      compile option, and a timing recorded at a different iteration count is
      a different experiment even on the same build.

    Upstream's own key cannot see either one: ``check_disk_cache`` mixes
    ``triton_key()``, the backend hash, ``fn.cache_key``, the cache-invalidating
    env vars, the tuning key and the config list, and neither
    ``TRITON_CACHE_AUTOTUNING`` (verified absent from this pin's
    ``CACHE_INVALIDATING_ENV_VARS`` in ``triton/include/triton/Tools/Sys/
    GetEnv.hpp``) nor the runtime library is among them.  So the search runner
    passes this integer as a kernel ``tl.constexpr`` that is also a member of
    ``@triton.autotune``'s ``key`` -- which puts it into both the tuning cache
    key and the compile cache key at once.  The comparison happens before a
    hit, not after: an unequal identity is a cache miss, so it is re-measured.
    """
    material = fingerprint_digest(fingerprints)
    if iters is not None:
        material = f"{material}|iters={int(iters)}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# The leaf cost table's C0 consumer: ORDERING only, zero pruning
# --------------------------------------------------------------------------- #

#: ``HmxLeafCostTable.h`` -- the measured per-leaf pcyc table.  It lives in
#: the compiler tree on purpose (its contract note 4 keeps it diagnostic-only
#: until W2 passes its own gate), and this module reads it exactly the way
#: the acceptance test does: by parsing the header's own entry grammar.  The
#: parsers below mirror ``test_hmx_leaf_cost_table.py``'s regexes on purpose
#: -- a reformatted row must fail loudly in both places instead of silently
#: dropping out of one of them.
_COST_TABLE = (
    _REPO / "qcom_hexagon_backend" / "include" / "hexagon" / "Dialect"
    / "Hmx" / "Transforms" / "HmxLeafCostTable.h"
)

#: The table's entry grammar.  Mirrored one-for-one from the acceptance test;
#: keep the two in step, because a row that one parser reads and the other
#: silently skips is a number nobody is checking.
_TABLE_STRINGS = r'(?:"(?:[^"\\]|\\.)*"\s*)+'
_SRC_ENTRY = re.compile(rf"\{{Src::(\w+),\s*({_TABLE_STRINGS})\}},")
_NO_DATA_ENTRY = re.compile(rf"\{{Leaf::(\w+),\s*({_TABLE_STRINGS})\}},")
_LEAF_ENTRY = re.compile(
    r"\{Leaf::(\w+),\s*Family::(\w+),\s*Cond::(\w+),\s*"
    r"(-?[\d.]+),\s*(\d+),\s*(\d+),\s*Src::(\w+)\},"
)
_FAMILY_ENTRY = re.compile(
    r"\{Family::(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(-?[\d.]+),\s*"
    r"(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(\d+),"
    r"\s*Src::(\w+)\},"
)
_SCALAR = re.compile(
    r"constexpr\s+(?:double|int64_t)\s+(\w+)\s*=\s*(-?[\d.]+);"
)

#: The libhmxapi md5 a citation carries, when it carries one.  Citations are
#: prose, so this is a prefix match by construction (``libhmxapi f42384f2``).
_CITATION_LIBHMXAPI = re.compile(r"libhmxapi(?:\.a)?\s+([0-9a-f]{7,32})")


def _join_strings(blob: str) -> str:
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', blob)
    return "".join(parts)


def table_revision(text: Optional[str] = None) -> str:
    """A content digest over the whole cost table: the ``table_rev`` stamp.

    Every parsed structure is represented -- the priced cells, the empty
    cells, the family bundles, every scalar and the provenance citations --
    so ANY edit to the table's content flips the digest.  That is the
    mechanical meaning of the version stamp: a decision taken under one
    ``table_rev`` is not a decision under another, and nobody has to remember
    that (design §4.3).

    Why not a git commit: the table rides the compiler tree, so a commit
    number conflates "a cell moved" with "the build moved" -- two facts with
    different consequences for a consumer.  A content digest separates them
    and does not depend on git state, which is unreliable under the local
    patch workflow.

    Prose is NOT covered: the digest is over the parsed table, so a comment
    rewrite leaves it alone while a changed citation flips it (citations are
    parsed content, and they are the provenance of the numbers).
    """
    return load_cost_table(text).table_rev


class CostTableIndex(NamedTuple):
    """The parsed table: every priced cell, the model scalars, the sources.

    ``missing`` is True when the header could not be read at all.  A missing
    table is a state the model must be able to name rather than paper over:
    predictions then abstain (:data:`ABSTAIN_TABLE_MISSING`) instead of
    pricing off an empty index, which would look like "everything abstains
    for a real reason" -- exactly the silent-failure shape this file avoids
    everywhere else.
    """

    table_rev: str
    cells: tuple[dict, ...]
    scalars: dict[str, float]
    sources: dict[str, str]
    missing: bool = False


def _canonical_table_text(text: str) -> str:
    """One normalized line per parsed entry, in a fixed order.

    Everything the parsers extract is represented, so any content edit flips
    the digest while a comment rewrite does not.
    """
    lines: list[str] = []
    for name, citation in sorted(
        (m.group(1), _join_strings(m.group(2)))
        for m in _SRC_ENTRY.finditer(text)
    ):
        lines.append(f"src {name} {citation}")
    for m in _LEAF_ENTRY.finditer(text):
        lines.append(
            "leaf {} {} {} {} {} {} {}".format(
                m.group(1), m.group(2), m.group(3), m.group(4),
                m.group(5), m.group(6), m.group(7))
        )
    for m in _NO_DATA_ENTRY.finditer(text):
        lines.append(f"nodata {m.group(1)} {_join_strings(m.group(2))}")
    for m in _FAMILY_ENTRY.finditer(text):
        lines.append(
            "family {} {} {} {} {} {} {} {} {} {} {}".format(
                *m.groups())
        )
    for m in _SCALAR.finditer(text):
        lines.append(f"scalar {m.group(1)} {m.group(2)}")
    return "\n".join(lines)


@functools.lru_cache(maxsize=None)
def load_cost_table(text: Optional[str] = None) -> CostTableIndex:
    """Parse the cost table into the index the model consumes.

    ``text`` is injectable for tests (a modified copy must flip the stamp
    without touching the tree); ``None`` reads the header.  Cached per input:
    the header does not change mid-process, and the runner asks once per
    candidate.
    """
    if text is None:
        try:
            text = _COST_TABLE.read_text(encoding="utf-8")
        except OSError:
            return CostTableIndex(table_rev="table-missing", cells=(),
                                  scalars={}, sources={}, missing=True)
    digest = hashlib.sha256(
        _canonical_table_text(text).encode("utf-8")).hexdigest()
    sources = {
        m.group(1): _join_strings(m.group(2))
        for m in _SRC_ENTRY.finditer(text)
    }
    cells = tuple(
        dict(leaf=m.group(1), family=m.group(2), cond=m.group(3),
             per_unit=float(m.group(4)), unit_count=int(m.group(5)),
             region_pcyc=int(m.group(6)), src=m.group(7),
             citation=sources.get(m.group(7), ""))
        for m in _LEAF_ENTRY.finditer(text)
    )
    scalars = {
        m.group(1): float(m.group(2)) for m in _SCALAR.finditer(text)
    }
    return CostTableIndex(table_rev=digest, cells=cells, scalars=scalars,
                          sources=sources, missing=False)


def cell_tier(cell: Mapping[str, Any],
              fingerprints: Optional[Mapping[str, Any]]) -> str:
    """Whether a cell's citation fingerprint matches this run's libhmxapi.

    ``current`` -- the cell was measured on the runtime library this run
    will execute, so its price is this build's price (the T1Replica0810 /
    T1Default0810 cells, citing libhmxapi ``f42384f2``); ``lineage`` -- the
    citation names a different build, so the number is the right shape at
    best; ``unknown`` -- the citation names no libhmxapi at all, or the run
    has no fingerprints.

    Contract note 3 made mechanical: this comparison is the trigger, not a
    judgement call.  Only ``current`` cells may enter the lower bound of the
    pruning interval (the side that can cause a drop); ``lineage`` cells may
    widen the upper bound, which only makes pruning more conservative.
    """
    if not fingerprints:
        return "unknown"
    current = fingerprints.get("libhmxapi.a") or {}
    md5 = str(current.get("md5", "") or "")
    if not md5 or md5 == "MISSING":
        return "unknown"
    cited = _CITATION_LIBHMXAPI.search(str(cell.get("citation", "")))
    if cited is None:
        return "unknown"
    return "current" if md5.startswith(cited.group(1)) else "lineage"


# --------------------------------------------------------------------------- #
# The tile cost model (C0: ORDERING only -- nothing here prunes a candidate)
# --------------------------------------------------------------------------- #

#: The three items the C0 model prices, and why exactly these three: they
#: mirror the table's own serial-arm model (``modeledSerialTotal`` =
#: pack_act + the blended engine + unpack).  Deliberately NOT priced:
#: ``pack_weight`` (the WR=0 arm only -- weight residency is the default),
#: the ring/DMA fixed overhead and the read-out protocol cost.  Those price
#: ARM decisions (staged vs serial, read-out on/off), and depth/readout are
#: not search dimensions, so a tile search has nothing to say about them
#: (design §2.1's out-of-scope classes C/D, locked again in §5 C2).
MODEL_ITEMS = ("pack", "engine", "unpack")

#: Per item: the leaf family it prices, and the cond values whose price
#: geometry this model cannot see.
#:
#: * ``Historical`` -- superseded build lineage the table keeps as evidence.
#: * ``SrcVtcm`` for pack -- the VTCM-source price belongs to the STAGED
#:   ring, and the model prices the serial arm (the table's own
#:   ``modeledSerialProducer`` uses only the hot/cold prices, same reading).
#: * the two strided conds -- whether a view strides is invisible in the tile
#:   geometry (the cross-step prices differ 4.8x), so they are never
#:   applicable here; the design makes them a mandatory abstain source.
#:
#: A cell with ``per_unit == 0`` is also never applicable: the table uses
#: that spelling for a reserved slot (T1Pending) and for a fused REGION
#: price (S2's staged ring), neither of which is a per-unit price.
_MODEL_LEAF_RULES = {
    "pack": ("PackActF16", {"Historical", "SrcVtcm", "SrcVtcmStrided",
                            "SrcDdrStridedCold"}),
    "engine": ("MmaF16", {"Historical"}),
    "unpack": ("UnpackAccF16", {"Historical"}),
}

#: The heat vocabulary, matching the table's ``Heat`` enum.  The band is the
#: interval the table explicitly leaves unmeasured: below ``kHotActBytes``
#: the source is L2-hot at the hot price, at/above ``kColdActBytes`` it
#: streams at the cold price, and between the two NOTHING was ever timed --
#: so the model abstains there instead of interpolating.
COST_HEAT_HOT = "hot"
COST_HEAT_COLD = "cold"
COST_HEAT_BAND = "band"


def _heat_class(footprint_bytes: int, hot_bytes: float,
                cold_bytes: float) -> str:
    """The table's ``activationHeat`` boundaries, applied to a footprint."""
    if footprint_bytes <= hot_bytes:
        return COST_HEAT_HOT
    if footprint_bytes >= cold_bytes:
        return COST_HEAT_COLD
    return COST_HEAT_BAND


def _item_cells(cells: Iterable[Mapping[str, Any]], item: str,
                heat: str) -> list[dict]:
    """The cells applicable to one model item under one heat class."""
    leaf, excluded = _MODEL_LEAF_RULES[item]
    out = []
    for cell in cells:
        if cell["leaf"] != leaf or cell["cond"] in excluded:
            continue
        if cell["per_unit"] <= 0:  # a reserved slot or a fused region
            continue
        if item == "pack":
            # Heat-conditioned: the serial ring's source-heat price is the
            # whole flip-1 mechanism.  In-kernel cells stand for the hot
            # side; the cold side has exactly one measured cell.
            if heat == COST_HEAT_COLD:
                if cell["cond"] != "SrcDdrColdStream":
                    continue
            elif cell["cond"] == "SrcDdrColdStream":
                continue
        out.append(cell)
    return out


# --------------------------------------------------------------------------- #
# The C0 vocabulary: abstain reasons, the manifest gate, the pruning constants
# --------------------------------------------------------------------------- #

#: The abstain reasons, one per design §3.4 class (plus ``table-missing``
#: for the unreadable header).  They go into the report and the JSON so a
#: reader can tell "the model said slow" from "the model declined to speak".
ABSTAIN_HEAT_BAND = "heat-band"
ABSTAIN_HEAT_CROSS = "heat-cross-boundary"
ABSTAIN_TAIL = "tail-blocks"
ABSTAIN_DTYPE = "dtype-no-cell"
ABSTAIN_MANIFEST = "manifest-strategy-unknown"
ABSTAIN_NO_CELL = "no-applicable-cell"
ABSTAIN_TABLE_MISSING = "table-missing"

#: The manifest vocabulary the model accepts, mirrored from
#: ``HmxManifest.h`` (``kHmxPlan*`` / ``kHmxReason*`` / the layout literal).
#: Anything else -- and, deliberately, NOTHING (no manifest at all, which is
#: what the simulated mode honestly has) -- abstains: a strategy the price
#: table cannot see (an N-split strided resident view, whose cross-step
#: prices differ 4.8x) must not be priced as if it were a dense bridge.
MANIFEST_PLANS_OK = ("full-hmx", "hmx-tail")
MANIFEST_REASONS_OK = ("selected-aligned", "selected-tail")
MANIFEST_LAYOUT_OK = "row-major-inner-contiguous"

#: The modeled/measured error band the acceptance test itself pinned for
#: these modeled totals, and its version.  Re-estimating it is a C1 gate
#: (>=8 same-libhmxapi configs, miscalibration <= 1/8); C0 only records.
COST_ENVELOPE = (0.70, 1.25)
ENVELOPE_VERSION = "env-1"

#: The decision margin of the pruning predicate: 15%, the floor of this
#: project's own verdict gate ``max(3xCV, 15%)``.  Below it a "gain" is not
#: a gain, so it is also not a reason to stop measuring a candidate.
PRUNE_MARGIN = 0.15


class TilePrediction(NamedTuple):
    """One candidate's modeled cost: an interval, its basis, its abstention.

    ``abstain`` is the model's "I don't know" and is a first-class outcome,
    not an error: an abstaining candidate keeps its place in the search (C0
    never drops anything) and simply sorts after the priced ones.  The five
    abstain classes are the design §3.4 vocabulary, plus ``table-missing``
    for the unreadable-header case.

    ``lb``/``ub`` are the interval the pruning predicate would test: the
    lower bound sums each item's CHEAPEST applicable cell, the upper bound
    the most expensive one, so the family spread (engine 22.0 vs 26.4,
    ~20% and unexplained; unpack 86.0 vs 191.9 across families) is absorbed
    by construction rather than argued about.  ``p50`` is the interval
    midpoint -- C0 has no basis for a better point estimate -- and it is
    what the calibration report compares against a measurement.  The
    ORDERING does not use it: :func:`rank_tile_candidates` sorts by ``lb``,
    because the midpoint is not monotone in the interval when intervals
    overlap (A = [100, 300] has midpoint 200 and loses to B = [150, 160]
    with midpoint 155, although A could still be the cheaper config), while
    the lower bound is: if every cell of A is cheaper than every cell of B
    (``ub(A) < lb(B)``), A sorts first, guaranteed.  Sorting by ``lb`` also
    puts the candidate that COULD be cheapest at the front of the bounded
    selection's dispatch budget, which is the right experiment to run
    first.
    """

    tiles: tuple[int, int, int]
    lb: float
    ub: float
    p50: float
    heat: str
    units: tuple[int, int, int]
    abstain: Optional[str]
    table_rev: str
    lb_tiers: tuple[str, ...] = ()


def predict_tile_pcyc(m: int, n: int, k: int, bm: int, bn: int, bk: int, *,
                      dtype: str = "f16", fingerprints: Optional[Mapping] = None,
                      cell_index: Optional[CostTableIndex] = None
                      ) -> TilePrediction:
    """Modeled pcyc interval for one tile candidate of one contraction.

    Host-only, device-free, triton-free: geometry in, interval out.  The
    unit counts follow the design (``pack Mt*Kt``,
    ``engine Mt*Nt*(3+ceil(Kt/32))``, ``unpack Mt*Nt``, with
    ``Mt=ceil(M/BM)`` etc.), the same shape as the table's own
    ``packActUnitCount``/``engineCallCount``/``unpackUnitCount``; the prices
    come from the cells the citation fingerprints qualify for.

    What the tile BUYS in this model is spelled out, because it is the C0
    hypothesis under test rather than a settled fact: a bigger tile means
    fewer pack sites and fewer engine calls, and a different source-heat
    class (its A slab is bigger).  At the anchor grids -- where every tile
    edge is 32, so ``Mt*Kt`` equals ``tiles(M)*tiles(K)`` -- the two
    readings of the unit count coincide; above 32 they do not, and the
    per-site count used here is the design's choice.  If the per-crouton
    reading turns out to be the true one, the calibration report shows it
    (``modeled/measured`` drifting systematically with ``BM``, cells going
    ``flagged``) -- which is exactly what C0 exists to find out.

    Abstains (design §3.4), in the order they are tested:

    1. **heat band** -- either footprint falls in the unmeasured
       (``kHotActBytes``, ``kColdActBytes``) band;
    2. **heat cross-boundary** -- the per-m-tile A slab and the whole A sit
       on opposite sides of a boundary, so the tile's heat class is not
       determined by its own geometry (the slab translation is the design's
       flagged assumption; the abstain is what covers it);
    3. **tail blocks** -- ``M%BM or N%BN or K%BK``: the tail leaf families
       are in ``kNoData``, a masked path has no price;
    4. **non-f16** -- ``PackActF32``/``UnpackAccF32`` are all ``kNoData``;
    5. **no applicable cell** -- some item has none (an empty injected
       index, a table without the needed cells).

    The manifest-strategy abstain (class 4 of the design) is NOT here: it
    needs the manifest, which only the runner reads.  It lives in
    :func:`manifest_abstains` and :func:`cost_prune`.
    """
    tiles = (int(bm), int(bn), int(bk))
    index = cell_index if cell_index is not None else load_cost_table()
    rev = index.table_rev

    def abstain(reason: str, heat: str = "unknown",
                units: tuple[int, int, int] = (0, 0, 0),
                tiers: tuple[str, ...] = ()) -> TilePrediction:
        nan = float("nan")
        return TilePrediction(tiles, nan, nan, nan, heat, units, reason,
                              rev, tiers)

    if dtype != "f16":
        return abstain(ABSTAIN_DTYPE)
    if index.missing:
        return abstain(ABSTAIN_TABLE_MISSING)
    if m % bm or n % bn or k % bk:
        return abstain(ABSTAIN_TAIL)
    hot = index.scalars.get("kHotActBytes")
    cold = index.scalars.get("kColdActBytes")
    if hot is None or cold is None:
        return abstain(ABSTAIN_TABLE_MISSING)

    # The two footprints the heat class is read from: the per-m-tile A slab
    # (this candidate's own reuse window) and the whole A (the serial ring's
    # source).  Disagreement between them is the abstain, not a coin flip.
    slab_heat = _heat_class(bm * k * CROUTON_ELEM_BYTES, hot, cold)
    whole_heat = _heat_class(m * k * CROUTON_ELEM_BYTES, hot, cold)
    if COST_HEAT_BAND in (slab_heat, whole_heat):
        return abstain(ABSTAIN_HEAT_BAND, heat=slab_heat)
    if slab_heat != whole_heat:
        return abstain(ABSTAIN_HEAT_CROSS, heat=slab_heat)
    heat = slab_heat

    mt = -(-m // bm)
    nt = -(-n // bn)
    kt = -(-k // bk)
    units = {
        "pack": mt * kt,
        "engine": mt * nt * (3 + (kt + 31) // 32),
        "unpack": mt * nt,
    }

    lb = 0.0
    ub = 0.0
    tiers: list[str] = []
    for item in MODEL_ITEMS:
        applicable = _item_cells(index.cells, item, heat)
        if not applicable:
            return abstain(ABSTAIN_NO_CELL, heat=heat,
                           units=(units["pack"], units["engine"],
                                  units["unpack"]))
        # The lower bound takes the cheapest CURRENT-fingerprint cell when
        # one exists: only cells measured on this run's runtime may drive a
        # drop (design §1).  A lineage cell may still price the candidate --
        # it just marks the bound as not-current, and cost_prune keeps such a
        # candidate.  With no fingerprints at all (pure ordering) the tier is
        # "unknown" and the cheapest applicable cell is used.
        current = [c for c in applicable
                   if fingerprints
                   and cell_tier(c, fingerprints) == "current"]
        cheapest = min(current or applicable,
                       key=lambda c: (c["per_unit"], c["src"], c["cond"]))
        lb += cheapest["per_unit"] * units[item]
        ub += max(c["per_unit"] for c in applicable) * units[item]
        tiers.append(cell_tier(cheapest, fingerprints))
    ordered_units = (units["pack"], units["engine"], units["unpack"])
    return TilePrediction(tiles, lb, ub, 0.5 * (lb + ub), heat, ordered_units,
                          None, rev, tuple(tiers))


# --------------------------------------------------------------------------- #
# The pruning predicate and the ordering (C0: the predicate has NO call site)
# --------------------------------------------------------------------------- #


def manifest_abstains(record: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The manifest-strategy abstain (design §3.4 class 4), or None.

    ``record`` is what the runner's compile prefilter reads out of a
    compiled kernel's manifest (``plan`` / ``reason`` / ``layout``) -- or
    ``None``, which is the simulated mode's honest state: with no compile
    there is no manifest, and the fail-open direction is to abstain rather
    than to price a strategy nobody looked at.
    """
    if record is None:
        return ABSTAIN_MANIFEST
    if record.get("plan") not in MANIFEST_PLANS_OK:
        return ABSTAIN_MANIFEST
    if record.get("reason") not in MANIFEST_REASONS_OK:
        return ABSTAIN_MANIFEST
    layout = record.get("layout")
    if layout is not None and layout != MANIFEST_LAYOUT_OK:
        return ABSTAIN_MANIFEST
    return None


def cost_prune(candidates, shape, *, control_tile=None,
               margin: float = PRUNE_MARGIN,
               envelope: tuple[float, float] = COST_ENVELOPE,
               predictions: Optional[Mapping[tuple, TilePrediction]] = None,
               manifest_records: Optional[Mapping[tuple, Mapping]] = None,
               fingerprints: Optional[Mapping] = None,
               cell_index: Optional[CostTableIndex] = None):
    """The conservative interval test -- written, pinned, and NOT CALLED.

    Nothing in C0 calls this.  It is here because the rule is the design's
    core safety property and it deserves tests before it deserves a call
    site: drop candidate ``B`` only when its most-optimistic modeled cost
    still exceeds the most-pessimistic modeled cost of the best candidate
    ``A`` by the margin composed with the error band::

        0.70 * lb(B)  >  (1 + margin) * 1.25 * ub(A)      # == lb(B) > 2.05 * ub(A)

    The composition is not decoration.  The table is UPPER-BOUND caliber
    (no overlap credit anywhere), and its own acceptance test pins
    modeled/measured to [0.70, 1.25]; a bare 15% would let a model that
    over-estimates A by 25% and under-estimates B by 30% prune a candidate
    that was never actually slower.  Candidates abstaining for ANY reason
    are kept -- the model's unknown is not evidence of cheapness -- and the
    manual control tile is immune: it is the arm the winner is judged
    against, so pruning it would delete the judgement itself.

    ``A`` is the priced candidate with the lowest ``p50`` (tie-broken by
    tiles, so the choice is deterministic).  If nothing is priced, nothing
    is dropped.  A candidate whose lower bound rests on ``lineage`` cells
    (measured on a different libhmxapi than this run) is also kept: only
    ``current``-fingerprint cells may drive a drop (design §1).

    ``shape`` is ``(M, N, K)``, the convention :func:`predict_tile_pcyc`
    takes -- note that the runner's own tuples are ``(M, K, N)``, the
    reference tables' spelling, so a caller must convert rather than pass
    its tuple through.

    Returns ``(kept, dropped, notes)``; ``kept`` preserves the caller's
    order, ``dropped`` carries one reason dict per candidate.
    """
    m, n, k = shape
    index = cell_index if cell_index is not None else load_cost_table()

    def tiles_of(candidate):
        """Accept both a TileConfig-like candidate and a bare tile tuple."""
        return tuple(candidate.tiles if hasattr(candidate, "tiles")
                     else candidate)

    if predictions is None:
        predictions = {
            tiles_of(c): predict_tile_pcyc(m, n, k, *tiles_of(c),
                                           fingerprints=fingerprints,
                                           cell_index=index)
            for c in candidates
        }
    control = tiles_of(control_tile) if control_tile is not None else None

    priced = [(c, predictions[tiles_of(c)]) for c in candidates
              if predictions[tiles_of(c)].abstain is None]
    if not priced:
        return list(candidates), [], [
            "cost_prune: no priced candidate -- nothing compared, nothing "
            "dropped"
        ]

    champion, champ = min(priced, key=lambda cp: (cp[1].p50, tiles_of(cp[0])))
    lo, hi = envelope
    threshold = (1.0 + float(margin)) * hi / lo
    kept, dropped, notes = [], [], [
        f"cost_prune: champion {tiles_of(champion)} p50={champ.p50:.0f} "
        f"lb={champ.lb:.0f} ub={champ.ub:.0f}; drop when "
        f"lb(B) > {threshold:.3f} * ub(A)"
    ]
    for cand in candidates:
        tiles = tiles_of(cand)
        pred = predictions[tiles]
        if control is not None and tiles == control:
            kept.append(cand)
            notes.append(f"cost_prune: {tiles} kept (control tile is "
                         "the arm the winner is judged against)")
            continue
        if pred.abstain is not None:
            kept.append(cand)
            notes.append(f"cost_prune: {tiles} kept (abstain "
                         f"{pred.abstain})")
            continue
        if manifest_records is not None:
            strategy = manifest_abstains(manifest_records.get(tiles))
            if strategy is not None:
                kept.append(cand)
                notes.append(f"cost_prune: {tiles} kept (abstain "
                             f"{strategy})")
                continue
        if any(tier != "current" for tier in pred.lb_tiers):
            kept.append(cand)
            notes.append(f"cost_prune: {tiles} kept (lower bound rests "
                         f"on {pred.lb_tiers} cells: only current-fingerprint "
                         "cells may drive a drop)")
            continue
        if pred.lb > threshold * champ.ub:
            dropped.append((cand, dict(
                tiles=list(tiles), lb=pred.lb, ub=pred.ub,
                champion_tiles=list(tiles_of(champion)),
                champion_ub=champ.ub,
                ratio=pred.lb / champ.ub, threshold=threshold,
                margin=margin, envelope=list(envelope),
                table_rev=pred.table_rev,
            )))
        else:
            kept.append(cand)
    notes.append(f"cost_prune: kept {len(kept)}, dropped {len(dropped)}")
    return kept, dropped, notes


def rank_tile_candidates(shape, legal, *, fingerprints=None,
                         cell_index=None):
    """C0's ordering: modeled lower bound first, span count as the tiebreak.

    This is the only way C0 touches the search, and it is safe by
    construction: the function returns a PERMUTATION of its input.  No
    candidate is added, none is removed, and the caller's bounded selection
    then spends its dispatch budget on the cheapest-looking candidates
    instead of the fewest-span ones.  span_count only counts blocks, so it
    is blind to block bytes and to source heat -- BK=2048 in one span and
    BK=64 in thirty-two spans is the blind spot this ordering repairs.

    The key is ``lb``, not the interval midpoint, and that is what makes the
    ordering's promise true: it is monotone in the interval (``ub(A) <
    lb(B)`` implies A first), which the midpoint cannot promise when
    intervals overlap -- see :class:`TilePrediction`.  ``lb`` is also the
    conservative choice for a bounded budget: the candidate whose
    OPTIMISTIC bound is lowest is the one that could be fastest, so it is
    the one worth measuring before the dispatch budget runs out.

    An abstaining candidate keeps its span order but sorts after every
    priced one: an abstain is the model saying "unknown", and unknown is
    not evidence of cheapness.  The sort key ends in ``tiles`` so the order
    is a function of the input SET, not of the order it arrived in -- which
    is what makes "same input, same order" a testable property.
    """
    m, n, k = shape
    index = cell_index if cell_index is not None else load_cost_table()

    def key(cfg):
        pred = predict_tile_pcyc(m, n, k, *cfg.tiles,
                                 fingerprints=fingerprints, cell_index=index)
        spans = span_count(m, n, k, cfg.tiles)
        if pred.abstain is not None:
            return (1, 0.0, spans, cfg.tiles)
        return (0, pred.lb, spans, cfg.tiles)

    return sorted(legal, key=key)


# --------------------------------------------------------------------------- #
# The benchmarker
# --------------------------------------------------------------------------- #

_PERF_RE = re.compile(r"^Perf:\s*([0-9]+(?:\.[0-9]+)?)", re.MULTILINE)
_PCYC_RE = re.compile(r"^PerfPcycles:\s*([0-9]+)", re.MULTILINE)


def parse_perf(text: str) -> Optional[dict[str, float]]:
    """The two numbers one launch's report carries, or ``None`` if absent.

    Both are per-iteration averages taken from the SAME pass over the loop
    (the device wrapper's ``benchmark_time_and_pcycles``), which is what makes
    the pair comparable at all.
    """
    perf = _PERF_RE.search(text)
    pcyc = _PCYC_RE.search(text)
    if perf is None and pcyc is None:
        return None
    return {
        "perf_us": float(perf.group(1)) if perf else float("nan"),
        "pcyc": float(pcyc.group(1)) if pcyc else float("nan"),
    }


def _quantile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an unsorted sample list."""
    if not samples:
        return float("inf")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def hexagon_bench(kernel_call, quantiles=(0.5, 0.2, 0.8), *, reps: int = 3,
                  verify=None) -> list[float]:
    """``triton.testing.do_bench``'s contract, answered by the device's cycles.

    ``kernel_call`` is one launch (the autotuner's closure over
    ``JITFunction.run``).  Each rep runs it, captures what the executor prints
    for that run, and keeps that run's OWN ``PerfPcycles`` -- never a clock
    constant, because the pcycles:qtimer ratio is a property of the run.  The
    return is ``[q50, q20, q80]`` over the reps, which the autotuner compares
    by first element exactly as it compares upstream's quantile list.

    Two properties this must have and upstream's ``do_bench`` does not:

    * **a bad config scores inf, it does not stop the search.**  What a bad
      config raises here is not ``OutOfResources``: a VTCM overflow is an MLIR
      diagnostic surfaced as ``RuntimeError``, and a grid/launch-contract
      violation is a ``ValueError`` from ``enforce_hmx_launch_contract`` --
      neither is in ``_bench``'s except tuple, so both would abort the sweep.
      The device layer has also been observed to end a run with
      ``SystemExit(1)``.  Everything except ``KeyboardInterrupt`` is caught
      here and becomes inf.
    * **a wrong config is wrong, not slow.**  ``verify`` is called after every
      rep with the output already pulled back (the pull happens anyway for the
      perf report); returning False, or raising, scores inf on the FIRST rep,
      so a broken config costs one launch instead of ``reps``.  A search that
      would pick a config producing garbage is not a slow search, it is a
      broken one.
    """
    samples: list[float] = []
    for _rep in range(max(1, int(reps))):
        spoken = io.StringIO()
        try:
            with redirect_stdout(spoken):
                kernel_call()
            metrics = parse_perf(spoken.getvalue())
        except (Exception, SystemExit):  # noqa: BLE001 - see the docstring
            return [float("inf")] * len(quantiles)
        if metrics is None or metrics["pcyc"] != metrics["pcyc"]:  # NaN check
            return [float("inf")] * len(quantiles)
        if verify is not None:
            try:
                accepted = verify()
            except (Exception, SystemExit):  # noqa: BLE001 - rel cannot be read
                return [float("inf")] * len(quantiles)
            if not accepted:
                return [float("inf")] * len(quantiles)
        samples.append(metrics["pcyc"])
    return [_quantile(samples, q) for q in quantiles]


def calibration_iters(perf_us: float, target_us: float = 2_000_000.0,
                      lo: int = 5, hi: int = 1000) -> int:
    """Iteration count that time-boxes one measured launch.

    Mirrors the calibration in ``exp/hmx/op_bench/mm_user_shapes.py``: a fixed
    count was measured to be fatal there, because one slow config took seconds
    per iteration and the device layer ended the whole run.  The box is
    ``[lo, hi]``; the once-term is already excluded device-side (the wrapper
    discards one warm call before the timed loop), so a per-iteration number
    stays valid at the low end of the box.
    """
    per_iter = max(float(perf_us), 1.0)
    return max(lo, min(hi, int(target_us / per_iter)))


# --------------------------------------------------------------------------- #
# The champion criterion and the per-round canary
# --------------------------------------------------------------------------- #

#: The project's verdict gate, ``max(3 x CV, 15%)`` (``AGENTS.md`` section 4),
#: written as its two factors.  Both numbers are inherited, not chosen here:
#: the multiplier is the project's own, and the floor is the same 15% that
#: :data:`PRUNE_MARGIN` names for the pruning predicate ("below it a gain is
#: not a gain").  A second pair of constants for the same gate is how two
#: copies of one rule start to disagree.
CHAMPION_CV_FACTOR = 3.0
CHAMPION_MARGIN_FLOOR = 0.15

#: The canary's statuses.  A canary is a known-healthy minimal config measured
#: once per round; its whole job is to separate "this config is slow" from
#: "the device is not well" -- two facts that produce the same per-config
#: numbers and need opposite responses.
CANARY_MISSING = "missing"
CANARY_UNMEASURED = "unmeasured"
CANARY_DRIFT = "drift"
CANARY_REFERENCE = "reference-established"
CANARY_OK = "ok"

#: What a round as a whole may say.  ``valid``: the canary was healthy, so the
#: configs' verdicts below it may be read.  ``invalid-canary``: they may NOT
#: -- the round is void, which is a different statement from "every config
#: was slow", and the one a reader could not otherwise make.
ROUND_VALID = "valid"
ROUND_INVALID_CANARY = "invalid-canary"

#: The canary config: the smallest tile the engine takes exactly, on the
#: contraction that exercises exactly it.  Legal by the enumerator
#: (``tile_verdict(32, 32, 32)`` is :data:`REASON_OK`), so a canary failure
#: is never a legality question, and one span of every loop, so it is the
#: cheapest honest probe there is.  The shape is a CUBE on purpose: the
#: predictions in this module take (M, N, K) while the runner's tuples are
#: (M, K, N), and a symmetric shape makes that spelling difference unable to
#: alter the canary's geometry -- the one place where a K/N mix-up would be
#: undetectable in the numbers.
CANARY_TILES = (32, 32, 32)
CANARY_SHAPE = (32, 32, 32)


def canary_config() -> dict:
    """The sentinel config as data: its tiles, its shape, and its label.

    Returned rather than written at each call site so the runner, the
    contract test and the report all name the SAME sentinel -- a canary
    whose shape quietly differed between rounds would compare this round's
    device against nobody.
    """
    return {"cfg": "canary-32x32x32", "tiles": list(CANARY_TILES),
            "shape_mnk": list(CANARY_SHAPE)}


def _is_finite(value) -> bool:
    """JSON-and-device-safe finiteness: present, not NaN, not infinite."""
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _record_q50(record) -> Optional[float]:
    """A measurement's median pcyc, or None when it has none.

    Accepts both a measurement dict and a bare number, so a previous
    round's canary record and a hand-written constant both work as a
    reference.
    """
    if isinstance(record, Mapping):
        value = record.get("q50_pcyc")
    else:
        value = record
    return float(value) if _is_finite(value) else None


def quantile_spread_cv(q20, q50, q80) -> float:
    """The CV a ``[q20, q50, q80]`` triple carries: ``(q80 - q20) / q50``.

    A spread proxy, not the sample stdev/mean: what one launch keeps is the
    quantile triple :func:`hexagon_bench` returns, not the per-rep samples
    (the autotuner's ``do_bench`` is called with exactly ``quantiles`` and
    nothing else, so the samples have nowhere to ride).  The proxy is
    deliberately on the conservative side -- for a normal sample the
    interquantile range is ``2 * 0.8416 * sigma``, so this overstates sigma
    by about 1.68x, and an overstated CV can only WIDEN the gate, never
    crown a config that is not there.  The paired-difference CV the
    criterion analysis prefers (``docs/analysis/
    criterion-paired-se-v2-2026-10-07.md``) needs the per-rep samples and a
    paired protocol, neither of which the search keeps; a per-arm spread
    overstates it for the same reason, which is the safe direction to be
    wrong in.  A missing or unreadable spread yields 0.0, which puts the
    gate on its 15% floor rather than inventing precision.
    """
    if not all(_is_finite(v) for v in (q20, q50, q80)):
        return 0.0
    median = float(q50)
    if median <= 0.0:
        return 0.0
    return max(0.0, (float(q80) - float(q20)) / median)


def champion_gate(cv) -> float:
    """``max(3 x CV, 15%)`` as a fraction: the margin a challenger must clear.

    The comparison is strict, so exactly-equal goes to the incumbent: the
    crown changes hands only when the challenger is faster by MORE than
    this.  An unreadable CV is treated as 0, which leaves the 15% floor
    standing rather than opening the gate.
    """
    value = float(cv) if _is_finite(cv) else 0.0
    return max(CHAMPION_CV_FACTOR * abs(value), CHAMPION_MARGIN_FLOOR)


def canary_verdict(metrics, reference=None) -> dict:
    """The per-round sentinel's verdict, as data.

    ``metrics`` is the canary config's own measurement this round -- the
    same ``{q50_pcyc, q20_pcyc, q80_pcyc}`` shape every config's is -- and
    ``reference`` is the previous round's canary record (or its median).
    Three ways it fails, and one way it cannot judge yet:

    * **missing** -- no canary rode with this round.  The round cannot tell
      "the device is broken" from "the configs are slow", so it may not
      publish a verdict at all.
    * **unmeasured** -- the canary failed or printed no cycles.  A
      known-healthy config that cannot run is the device, not the config.
    * **drift** -- the canary moved by more than the gate from its
      reference.  Same config, same build, same iteration count: a move
      that large is the clock/power state (C15:14 gates while the qtimer
      does not, and this project has measured that ratio move), and every
      other number in the round was taken under whatever state that was.
    * **reference-established** -- no healthy reference existed, so this
      run becomes the one.  The round is valid; the next round can judge.

    Only ``ok`` and ``reference-established`` are healthy.  The returned
    record carries enough state (its own median) to serve as the next
    round's ``reference``.
    """
    record = dict(status=CANARY_OK, healthy=True, q50_pcyc=None, cv=0.0,
                  reference_q50_pcyc=None, delta_pct=None, gate_pct=None,
                  reason="")
    if metrics is None:
        record.update(
            status=CANARY_MISSING, healthy=False,
            reason=("no canary measurement rode with this round: without "
                    "its sentinel the round cannot tell a broken device "
                    "from slow configs, so it publishes no verdict"))
        return record

    q50 = _record_q50(metrics)
    cv = quantile_spread_cv(metrics.get("q20_pcyc"), metrics.get("q50_pcyc"),
                            metrics.get("q80_pcyc")) if isinstance(
                                metrics, Mapping) else 0.0
    record["cv"] = cv
    record["q50_pcyc"] = _finite_or_none(q50)
    if q50 is None:
        record.update(
            status=CANARY_UNMEASURED, healthy=False,
            reason=("the canary config failed or printed no cycles: a "
                    "known-healthy config that cannot run indicts the "
                    "device or the launch path, not the config"))
        return record

    reference_q50 = _record_q50(reference)
    record["reference_q50_pcyc"] = _finite_or_none(reference_q50)
    if reference_q50 is None or reference_q50 <= 0.0:
        record.update(
            status=CANARY_REFERENCE,
            reason=("no healthy reference existed, so this run establishes "
                    "it; the round is valid and the next round can judge "
                    "against this median"))
        return record

    gate = champion_gate(cv)
    delta = (q50 - reference_q50) / reference_q50
    record["delta_pct"] = _finite_or_none(100.0 * delta)
    record["gate_pct"] = _finite_or_none(100.0 * gate)
    if abs(delta) > gate:
        record.update(
            status=CANARY_DRIFT, healthy=False,
            reason=(f"the canary moved {100.0 * delta:+.1f}% from its "
                    f"reference, more than the {100.0 * gate:.1f}% gate: "
                    "same config, same build, same iteration count, so the "
                    "device state moved and every other number in this "
                    "round was taken under it"))
        return record
    record["reason"] = (f"the canary stayed within {100.0 * abs(delta):.1f}% "
                        f"of its reference (gate {100.0 * gate:.1f}%)")
    return record


def judge_champion_round(measurements, *, control=None, canary=None,
                         canary_reference=None, iters=None) -> dict:
    """The search's verdict: who won, by how much, and whether to read it.

    ``measurements`` is one record per measured config -- the ``timings``
    mapping the sweep already produces (``cfg``, ``tiles``, ``q50_pcyc``,
    ``q20_pcyc``, ``q80_pcyc``; a missing or infinite median means that
    config was never measured).  ``control`` is the manual default tile the
    winner is judged against (the design's own control arm), given as a cfg
    label or a tiles tuple.  ``canary`` is the sentinel's measurement and
    ``canary_reference`` the previous round's canary record.

    The crown is DEFENDED, not merely argmin'd.  Candidates are visited in
    a deterministic order (tiles, then label -- never the measurement
    itself, so noise cannot pick its own bracket) starting from the control
    arm, and a challenger takes the crown only by beating the INCUMBENT by
    more than ``max(3 x CV, 15%)`` with the bar set by the noisier of the
    two arms.  A challenger inside the gate keeps the incumbent, so a 2%
    "improvement" cannot take the crown on its own noise.  With no control
    arm the incumbent starts at the first enumerated candidate: the walk is
    still deterministic, but the starting point is nominal, which is why the
    significance flag is then False no matter who survives.

    The per-config failures and the round-level failure stay separate.  A
    config that scored inf is recorded under ``unmeasured`` -- that config
    is bad, and the round can say so.  A canary that failed is recorded as
    ``invalid-canary`` -- the DEVICE is suspect, every config in the round
    is equally suspect, and the report's champion is still named (it is the
    walk's survivor) but marked not significant, because a round that
    cannot vouch for itself does not get to publish a winner.  That is the
    whole point of the sentinel: "all configs are slow" and "the device is
    broken" used to be the same JSON.

    ``iters`` rides along because a percentage without its
    ``iters_per_launch`` is not a citation (``AGENTS.md`` section 4).
    """
    canary_record = canary_verdict(canary, canary_reference)
    round_valid = bool(canary_record["healthy"])

    rows = []
    for entry in measurements:
        row = dict(entry)
        row["tiles_tuple"] = tuple(row.get("tiles") or ())
        rows.append(row)

    def q50_of(row):
        return _record_q50(row)

    measured = [r for r in rows if q50_of(r) is not None]
    unmeasured = [str(r.get("cfg", "")) for r in rows if q50_of(r) is None]

    def cv_of(row):
        return quantile_spread_cv(row.get("q20_pcyc"), row.get("q50_pcyc"),
                                  row.get("q80_pcyc"))

    def label_of(row):
        return str(row.get("cfg", ""))

    def as_control(row):
        """Match the control arm by label or by tiles."""
        if control is None:
            return False
        if str(control) == label_of(row):
            return True
        tiles = tuple(control) if not isinstance(control, str) else None
        return tiles is not None and tiles == row["tiles_tuple"]

    control_rows = [r for r in measured if as_control(r)]
    control_row = control_rows[0] if control_rows else None
    control_any = [r for r in rows if as_control(r)]
    ordered = sorted(measured,
                     key=lambda r: (r["tiles_tuple"], label_of(r)))

    comparisons: list[dict] = []
    changes: list[dict] = []

    def compare(challenger, incumbent, step):
        challenger_q50 = q50_of(challenger)
        incumbent_q50 = q50_of(incumbent)
        cv = max(cv_of(challenger), cv_of(incumbent))
        gate = champion_gate(cv)
        delta = (incumbent_q50 - challenger_q50) / incumbent_q50
        record = dict(
            step=step, challenger=label_of(challenger),
            incumbent=label_of(incumbent),
            challenger_q50_pcyc=_finite_or_none(challenger_q50),
            incumbent_q50_pcyc=_finite_or_none(incumbent_q50),
            delta_pct=_finite_or_none(100.0 * delta),
            cv=_finite_or_none(cv), gate_pct=_finite_or_none(100.0 * gate),
            gate_form="max(3xCV, 15%)",
            outcome="kept",
        )
        if delta > gate:
            record["outcome"] = "takeover"
            changes.append(dict(record))
        comparisons.append(record)
        return record["outcome"] == "takeover"

    incumbent = None
    walk_note = ""
    if control_row is not None:
        incumbent = control_row
        walk_note = ("the incumbent starts at the manual control tile: the "
                     "winner is judged against it, so it defends the crown "
                     "first")
    elif control_any:
        walk_note = ("the manual control tile was nominated but never "
                     "measured (it scored inf), so there is no arm to judge "
                     "a win against")
    elif control is not None:
        walk_note = ("the nominated control arm is not among this round's "
                     "measurements, so there is no arm to judge a win "
                     "against")
    elif ordered:
        incumbent = ordered[0]
        walk_note = ("no control arm was nominated: the incumbent starts at "
                     "the first enumerated candidate, so the walk is "
                     "deterministic but the starting point is nominal")

    if incumbent is not None:
        for step, challenger in enumerate(
                [r for r in ordered if r is not incumbent]):
            if compare(challenger, incumbent, step):
                incumbent = challenger

    champion_row = incumbent
    champion = label_of(champion_row) if champion_row is not None else None
    champion_tiles = (list(champion_row["tiles_tuple"])
                      if champion_row is not None else None)

    beats_control = None
    if control_row is not None and champion_row is not None:
        control_q50 = q50_of(control_row)
        champion_q50 = q50_of(champion_row)
        cv = max(cv_of(control_row), cv_of(champion_row))
        gate = champion_gate(cv)
        if champion_row is control_row:
            beats_control = dict(
                delta_pct=0.0, cv=_finite_or_none(cv),
                gate_pct=_finite_or_none(100.0 * gate),
                gate_form="max(3xCV, 15%)", significant=False,
                note=("the manual control tile kept the crown: no "
                      "candidate beat it by the gate"))
        else:
            delta = (control_q50 - champion_q50) / control_q50
            significant = bool(round_valid and delta > gate)
            beats_control = dict(
                delta_pct=_finite_or_none(100.0 * delta),
                cv=_finite_or_none(cv),
                gate_pct=_finite_or_none(100.0 * gate),
                gate_form="max(3xCV, 15%)", significant=significant,
                note=("the champion beat the manual control tile by "
                      f"{100.0 * delta:.1f}%, "
                      + ("more than" if delta > gate else "not more than")
                      + f" the {100.0 * gate:.1f}% gate"))

    significant = bool(round_valid and beats_control is not None
                       and beats_control["significant"])
    if not round_valid:
        significance_note = (
            "the round is void (" + canary_record["status"] + "): "
            + canary_record["reason"] + ". The champion below is the walk's "
            "survivor, not a verdict.")
    elif beats_control is None:
        significance_note = (
            "no control arm produced a measurement, so there is nothing to "
            "prove a win against: the champion is the walk's survivor "
            "(NOT-PROVEN, which is not the same as 'no effect')")
    elif beats_control["significant"]:
        significance_note = (
            f"the champion beat the manual control tile by more than the "
            f"gate at iters={iters}")
    else:
        significance_note = (
            f"the champion did not beat the manual control tile by more "
            f"than the gate at iters={iters}: NOT-PROVEN, which is not the "
            "same as 'no effect'")

    return dict(
        kind="hmx-search-champion",
        round_verdict=ROUND_VALID if round_valid else ROUND_INVALID_CANARY,
        iters=iters,
        gate=dict(form="max(3xCV, 15%)", cv_factor=CHAMPION_CV_FACTOR,
                  margin_floor=CHAMPION_MARGIN_FLOOR),
        canary=canary_record,
        walk_note=walk_note,
        measured_configs=len(measured),
        champion=champion,
        champion_tiles=champion_tiles,
        champion_q50_pcyc=(_finite_or_none(q50_of(champion_row))
                           if champion_row is not None else None),
        control=(label_of(control_row) if control_row is not None else None),
        control_tiles=(list(control_row["tiles_tuple"])
                       if control_row is not None else None),
        beats_control=beats_control,
        champion_significant=significant,
        significance_note=significance_note,
        changes=changes,
        comparisons=comparisons,
        unmeasured=unmeasured,
        cvs={label_of(r): _finite_or_none(cv_of(r)) for r in rows},
    )


# --------------------------------------------------------------------------- #
# The calibration cross table: constants vs this run's measurements
# --------------------------------------------------------------------------- #

#: The only three conclusions a cell may carry in the report (design §4.1).
#: "untouched" is the honest majority when a host-only drill measured
#: nothing: the run has no data, so the cell gets no verdict.
CALIBRATION_CORROBORATED = "corroborated"
CALIBRATION_FLAGGED = "flagged"
CALIBRATION_UNTOUCHED = "untouched"


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    """JSON has no infinity and no NaN; record those as None."""
    if value is None:
        return None
    return value if value == value and value not in (float("inf"),
                                                     float("-inf")) else None


def build_cost_calibration(rows, *, fingerprints, cell_index=None,
                           envelope=COST_ENVELOPE):
    """The "constants vs this run's measurements" cross table, as data.

    ``rows`` is one entry per MEASURED config -- what the runner's sweep
    produced.  Each carries BOTH spellings of the shape: ``shape`` is the
    runner's own (M, K, N) order (the reference tables'), recorded for the
    reader of the JSON, and ``shape_mnk`` is this module's (M, N, K) order,
    which the model is called with.  The two exist because the heat
    footprint is asymmetric in K and N, so an order mix-up would be a
    silent wrong-footprint class rather than an error.

    For each row the model is re-run, so the table records modeled lb/p50/ub
    beside the measured number, the ratio, and whether that ratio sits inside
    the envelope.  A host-only drill has no measured rows, and the report
    says so instead of inventing agreement.

    The report is a PROPOSAL, never an edit: nothing here writes to
    ``HmxLeafCostTable.h``, and ``write_back`` says so in the artifact
    itself.  Updating the table is a separate reviewed edit that adds a new
    row with a new citation (the T1 fill precedent) -- which flips
    ``table_rev``, which invalidates every decision taken under the old one.
    That is the whole write-back protocol, and it is deliberately not
    automated: a runner that could rewrite its own price table would be
    grading its own homework.

    Every search-measured number is a WHOLE-KERNEL pcyc, so it has no
    standing to move a per-unit leaf price (design §4.2): the most this
    report does to a Tier-2 cell is ``flag`` it for re-measurement.
    """
    index = cell_index if cell_index is not None else load_cost_table()
    lo, hi = envelope

    configs = []
    cell_judged: dict[tuple, int] = {}
    cell_out: dict[tuple, int] = {}
    for row in rows:
        m, n, k = row["shape_mnk"]
        bm, bn, bk = row["tiles"]
        pred = predict_tile_pcyc(m, n, k, bm, bn, bk,
                                 fingerprints=fingerprints,
                                 cell_index=index)
        measured = row["q50_pcyc"]
        ratio = (pred.p50 / measured) if (measured and pred.abstain is None
                                          and pred.p50 == pred.p50) else None
        in_band = ratio is not None and lo <= ratio <= hi
        configs.append(dict(
            shape=list(row["shape"]), shape_mnk=list(row["shape_mnk"]),
            cfg=row.get("cfg"),
            tiles=list(row["tiles"]), verdict=row.get("verdict"),
            gate_rel=row.get("gate_rel"), iters=row.get("iters"),
            q50_pcyc=_finite_or_none(measured),
            prediction=dict(
                lb=_finite_or_none(pred.lb), p50=_finite_or_none(pred.p50),
                ub=_finite_or_none(pred.ub), heat=pred.heat,
                units=list(pred.units), abstain=pred.abstain,
                lb_tiers=list(pred.lb_tiers), table_rev=pred.table_rev),
            modeled_over_measured=ratio,
            in_band=in_band,
            miscalibrated=bool(ratio is not None and not in_band),
        ))
        # Which cells this row's prediction consulted (the model's own
        # applicable set), counted so a cell's conclusion is traceable to
        # the rows that implicate it.
        if pred.abstain is None:
            for item in MODEL_ITEMS:
                for cell in _item_cells(index.cells, item, pred.heat):
                    key = (cell["leaf"], cell["family"], cell["cond"],
                           cell["per_unit"], cell["src"])
                    cell_judged[key] = cell_judged.get(key, 0) + 1
                    if not in_band:
                        cell_out[key] = cell_out.get(key, 0) + 1

    constants = []
    for cell in index.cells:
        key = (cell["leaf"], cell["family"], cell["cond"], cell["per_unit"],
               cell["src"])
        judged = cell_judged.get(key, 0)
        outside = cell_out.get(key, 0)
        if judged == 0:
            conclusion = CALIBRATION_UNTOUCHED
        elif outside:
            conclusion = CALIBRATION_FLAGGED
        else:
            conclusion = CALIBRATION_CORROBORATED
        constants.append(dict(
            leaf=cell["leaf"], family=cell["family"], cond=cell["cond"],
            per_unit=cell["per_unit"], src=cell["src"],
            citation=cell.get("citation", ""),
            tier=cell_tier(cell, fingerprints),
            judged_in=judged, outside_envelope=outside,
            conclusion=conclusion))

    flagged = [c for c in constants if c["conclusion"] == CALIBRATION_FLAGGED]
    ratios = [c["modeled_over_measured"] for c in configs
              if c["modeled_over_measured"] is not None]
    envelope_table = dict(
        version=ENVELOPE_VERSION, lo=lo, hi=hi,
        measured_configs=len(ratios),
        in_band=sum(1 for r in ratios if lo <= r <= hi),
        out_of_band=sum(1 for r in ratios if not lo <= r <= hi),
        min_ratio=min(ratios) if ratios else None,
        max_ratio=max(ratios) if ratios else None,
        # C1's gate, computed and recorded but NOT acted on in C0.
        reestimate_eligible=(
            len(ratios) >= 8
            and sum(1 for r in ratios if not lo <= r <= hi) <= 1),
        note=("C0 records only: re-estimating the envelope is a C1 gate "
              "(>=8 same-libhmxapi configs, miscalibration <= 1/8), and a "
              "re-estimate is a NEW VERSION, never an edit of this one."),
    )
    return dict(
        kind="hmx-cost-calibration",
        write_back="none",
        table_rev=index.table_rev,
        table_missing=index.missing,
        envelope=dict(version=ENVELOPE_VERSION, lo=lo, hi=hi),
        fingerprint_digest=fingerprint_digest(fingerprints),
        fingerprints=fingerprints,
        measured_configs=len(configs),
        measured_note=("host-only drill: no config was measured, so every "
                       "cell is 'untouched' and the envelope table is empty "
                       "-- the report exists to be filled by a device run."
                       if not configs else ""),
        configs=configs,
        constants=constants,
        flagged_cells=flagged,
        envelope_estimate=envelope_table,
    )


def render_cost_calibration_md(report) -> str:
    """The same report as Markdown, for a human reading the log directory."""
    lines = [
        "# HMX cost calibration -- constants vs this run's measurements",
        "",
        f"- table_rev: `{report['table_rev']}`",
        f"- build: `{report['fingerprint_digest']}`",
        f"- envelope: [{report['envelope']['lo']}, {report['envelope']['hi']}] "
        f"({report['envelope']['version']})",
        f"- write back: **{report['write_back']}** -- this report proposes; "
        "editing the table is a separate reviewed edit.",
        "",
        "## Constants vs measured",
        "",
    ]
    if report["measured_note"]:
        lines += [f"> {report['measured_configs']} measured configs. "
                  f"{report['measured_note']}", ""]
    lines += [
        "| cell | per_unit | tier | judged | outside | conclusion |",
        "|---|---|---|---|---|---|",
    ]
    for cell in report["constants"]:
        lines.append(
            f"| {cell['leaf']}/{cell['family']}/{cell['cond']} "
            f"({cell['src']}) | {cell['per_unit']} | {cell['tier']} "
            f"| {cell['judged_in']} | {cell['outside_envelope']} "
            f"| {cell['conclusion']} |")
    lines += ["", "## Measured configs", ""]
    if report["configs"]:
        lines += [
            "| config | measured q50 | modeled p50 | modeled ub | ratio | "
            "in band |", "|---|---|---|---|---|---|",
        ]
        for row in report["configs"]:
            pred = row["prediction"]
            ratio = row["modeled_over_measured"]
            lines.append(
                f"| {row['cfg']} | {row['q50_pcyc']} | {pred['p50']} "
                f"| {pred['ub']} | "
                f"{ratio:.3f} | {'yes' if row['in_band'] else 'NO'} |")
    else:
        lines.append("(none -- this run measured no config.)")
    est = report["envelope_estimate"]
    lines += [
        "", "## Envelope", "",
        f"- measured configs in band: {est['in_band']} / "
        f"{est['measured_configs']}",
        f"- re-estimate eligible (C1 gate): {est['reestimate_eligible']}",
        f"- {est['note']}",
        "",
    ]
    return "\n".join(lines)
