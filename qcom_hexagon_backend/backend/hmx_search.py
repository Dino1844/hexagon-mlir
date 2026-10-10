# ===- hmx_search.py -------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""The reusable halves of the HMX search loop: a device-pcyc benchmarker, a
tile-config enumerator, and the build/protocol identity that keys its cache.

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
    )
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import statistics
from contextlib import redirect_stdout
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

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
