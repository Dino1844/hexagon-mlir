#!/usr/bin/env python3
"""Host-runnable, byte-level lock on the HMX pack/unpack leaf permutations.

Why this file exists
--------------------
``bin/runtime/hmx/src/HMXLayout.c`` implements the crouton layout leaves --
the permutation that maps logical (row, col) coordinates into a 2 KB crouton's
byte slots (and back). Until now its *permutation behavior* had no test that
runs without a device: the only locks were the device-side numerical tests
(need the phone) and the tree-external oracle (``exp/hmx/oracle/``, which
project rules keep out of the public tree). The historical cost of that gap
is recorded in HMXLayout.c itself: ``hmx__pack_T1_f32`` once carried the
wrong offset table and was only caught on device as ``rel=0.25``.

How it works
------------
This compiles the **real HMXLayout.c, unmodified**, on the host:

  1. ``hmx_layout_host_emu/`` holds test-only stubs of ``<hexagon_types.h>``
     and ``<hvx_hexagon_protos.h>`` whose 15 intrinsic emulations are scalar
     transcriptions of the HVX builtins' documented semantics (each with its
     provenance written next to it -- the strongest being llama.cpp's own
     identical idioms; see the stub headers).
  2. ``hmx_layout_host_emu/driver.c`` includes the production file verbatim
     and speaks a tiny stdin/stdout protocol: Python fills buffers, calls one
     leaf, and reads the destination buffer back. It contains no layout
     knowledge. Buffers live below 2 GB because every leaf takes ``unsigned``
     (32-bit) addresses, exactly like the device.
  3. This file computes every case's **expected bytes from an independent
     closed-form model** -- the AH/WH/AR byte-offset formulas
     ``byte(r, c) = 128*(r>>1) + 4*c + 2*(r&1)`` with documented boundary
     semantics (out-of-range rows/columns read as zero on pack; only in-range
     rows/columns are written on unpack) -- and compares byte for byte, with
     poisoned margins so a write outside the predicted region is caught.

The expected bytes are never derived from HMXLayout.c's own tables, so a
wrong table, a wrong path, a wrong bound or a wrong intrinsic operand makes
the battery red. Two committed negative controls prove the gate can go red
by injecting known-wrong tables into scratch copies of the file: one
replays the historical ``hmx__pack_T1_f32`` bug exactly (its initializer
replaced by the concatenated-form ``hmx__pack_T1`` table, as documented in
the production file's comment).

What it locks
-------------
* The permutation math of every layout leaf in HMXLayout.c (pack act/weight,
  f16/f32, single/bulk, transposed-source T forms, bounds-safe tails; unpack
  f16/f32, single/bulk, tails with and without residual), against the
  closed-form layout: every output byte, including zero fills, plus "no
  writes outside the predicted region" (margin poison).
* The boundary semantics: strided sources, partial rows/columns, tiles past
  the edge, bulk step/clamp behavior, valid-extent staging of the tails.
* The physical constants, by parsing ``HMXAPI.h`` (same source-text lock
  pattern as ``test_hmx_leaf_abi_contract.py``) and asserting the engine's
  physical shape, so a tile-size change is a loud failure, not silent
  adaptation.

What it does NOT lock (residual risk, stated honestly)
------------------------------------------------------
* **The HVX intrinsic semantics themselves.** If an emulation in the stub
  disagrees with real HVX hardware in a way that exactly compensates a wrong
  production call, this test stays green while the device is wrong. The
  counterweights are: each emulation's independent provenance (llama.cpp
  idioms, transcribed intents), multiple call sites per intrinsic expecting
  different results, and the device-side numerical tests.
* **The SDK compiler's lowering** of HMXLayout.c to Hexagon code, the
  aligned/unaligned store instruction choice (VMEM vs VMEMU write the same
  bytes), and anything about performance.
* **The HMX engine's own expectation.** This locks the leaves against the
  closed-form layout; that the engine (mma/bias/acc leaves, which need HMX
  instructions and stay out of scope here) agrees with that same layout is
  only proven on device.
* Non-integer-32 dst_cols for the f16 unpack: HMXLayout.c documents that a
  column remainder is a multiple of 32 (the crouton width), so the battery
  stays inside that contract; the f32 unpack's predicated tail (any width)
  IS covered.

Run: python test_hmx_layout_permutation_contract.py
     python -m pytest -q test_hmx_layout_permutation_contract.py
"""
from __future__ import annotations

import itertools
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
EMU_DIR = TEST_DIR / "hmx_layout_host_emu"
DRIVER_C = EMU_DIR / "driver.c"
RUNTIME = TEST_DIR.parent  # bin/runtime
HMXAPI_H = RUNTIME / "hmx" / "include" / "HMXAPI.h"
LAYOUT_C = RUNTIME / "hmx" / "src" / "HMXLayout.c"

MARGIN = 256  # poisoned bytes on each side of the predicted write region
SRC_OFF = 4096
DST_OFF = 4096
RES_OFF = 4096

# ---------------------------------------------------------------------------
# The physical constants, locked to HMXAPI.h's own #defines (source-text
# lock, same pattern as test_hmx_leaf_abi_contract.py). The closed form below
# is written in terms of these, and the physical shape is asserted so a tile
# reshape is a loud failure requiring this model to be re-derived.
# ---------------------------------------------------------------------------


def c_macro(source: str, name: str) -> int:
    match = re.search(rf"^#define\s+{name}\s+([0-9]+)u?\b", source, re.M)
    assert match, f"#define {name} not found in HMXAPI.h (renamed?)"
    return int(match.group(1))


_API = HMXAPI_H.read_text(encoding="utf-8")
TILE_BYTES = c_macro(_API, "HMX_TILE_BYTES")
TILE_ROWS = c_macro(_API, "HMX_TILE_ROWS")
TILE_COLS = c_macro(_API, "HMX_TILE_COLS")
BLOCK_BYTES = c_macro(_API, "HMX_BLOCK_BYTES")
# HMX_BLOCK_PAIRS is spelled as the expression (HMX_TILE_ROWS / 2u); pin the
# literal text so an edit to it is loud rather than silently re-derived.
_bp = re.search(r"^#define\s+HMX_BLOCK_PAIRS\s+\(HMX_TILE_ROWS\s*/\s*2u\)", _API, re.M)
assert _bp, "HMX_BLOCK_PAIRS is no longer (HMX_TILE_ROWS / 2u); update this lock"
BLOCK_PAIRS = TILE_ROWS // 2

# The closed form's shape. AH and WH are the same permutation (HMXLayout.c's
# own header note), and AR is too; see the model below.
assert (TILE_ROWS, TILE_COLS, TILE_BYTES, BLOCK_BYTES, BLOCK_PAIRS) == (
    32, 32, 2048, 128, 16,
), "the engine tile changed shape: this closed-form model must be re-derived"


def lane_byte(r: int, c: int) -> int:
    """The AH/WH/AR closed form: byte slot of logical (row, col) inside one
    crouton, from the HMX PRM figures and llama.cpp's producers/consumers
    (two independent derivations agreeing; provenance recorded in the
    tree-external exp/hmx/oracle/layout_oracle.h, restated here so this test
    is self-contained in the public tree).

    AH: r = spatial row (M),  c = input channel (K)
    WH: r = input channel (K), c = output channel (N)
    AR: r = spatial row (M),  c = output channel (N)
    """
    return BLOCK_BYTES * (r >> 1) + 4 * c + 2 * (r & 1)


# ---------------------------------------------------------------------------
# Deterministic value generators. f16 pack values are moved as bits only, so
# any FINITE fp16 pattern is valid (finite keeps NaN-payload handling out of
# scope). Every value on an f32 path is a quarter-integer that is exact in
# fp16, fp32 and the emulated qf32, so no rounding mode is load-bearing.
# ---------------------------------------------------------------------------


class Seq:
    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF

    def next(self) -> int:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s


def h16(seq: Seq) -> int:
    """A finite fp16 bit pattern (exponent < 31), subnormals included."""
    x = seq.next()
    return (x & 0x7BFF) | ((seq.next() & 0x10000) << 15)


F16_PAD = 0x7BAD  # finite; fills logical padding, never legitimately read
F32_PAD = 1337.25  # exact in fp16; same role


def f32_exact(seq: Seq) -> float:
    return ((seq.next() % 4096) - 2048) / 4.0


def f32_to_h16(v: float) -> int:
    """Exact conversion (battery discipline makes inexact values impossible:
    struct raises OverflowError otherwise, which is the loud guard)."""
    return struct.unpack("<H", struct.pack("<e", v))[0]


def h16_bytes(v: int) -> bytes:
    return struct.pack("<H", v & 0xFFFF)


def h16_to_f32(v: int) -> float:
    return struct.unpack("<e", struct.pack("<H", v & 0xFFFF))[0]


def f32_bytes(v: float) -> bytes:
    return struct.pack("<f", v)


# ---------------------------------------------------------------------------
# The independent reference model. Everything below is derived from the
# documented leaf contracts in HMXAPI.h plus lane_byte(); nothing reads
# HMXLayout.c's tables or code paths.
# ---------------------------------------------------------------------------


def tile_from_get(get_elem, tr: int, tc: int) -> bytes:
    """One 2 KB crouton: get_elem(global_r, global_c) -> halfword (0 for
    out-of-range), placed at lane_byte(local_r, local_c)."""
    out = bytearray(TILE_BYTES)
    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            v = get_elem(tr * TILE_ROWS + r, tc * TILE_COLS + c) & 0xFFFF
            o = lane_byte(r, c)
            out[o] = v & 0xFF
            out[o + 1] = v >> 8
    return bytes(out)


def rowmajor_getter(rect: list, stride: int, rows: int, cols: int, f32: bool):
    def get(gr: int, gc: int) -> int:
        if gr >= rows or gc >= cols:
            return 0
        v = rect[gr * stride + gc]
        return f32_to_h16(v) if f32 else v

    return get


def transposed_getter(rect: list, stride: int, k_ext: int, n_ext: int, f32: bool):
    """Source is the row-major [n][k] transpose input: W[k, n] = src[n][k]."""

    def get(k: int, n: int) -> int:
        if k >= k_ext or n >= n_ext:
            return 0
        v = rect[n * stride + k]
        return f32_to_h16(v) if f32 else v

    return get


def tail_getter(rect: list, stride: int, rows: int, cols: int, f32: bool,
                vr: int, vc: int, tr: int, tc: int, transposed: bool):
    """Bounds-safe staging: element (r, c) of the tile at (tr, tc) is copied
    iff r < valid_rows and c < valid_cols AND the logical coordinate is
    inside the extents; everything else is zero (HMXAPI.h: 'the valid extents
    are in [1, 32]'; the staging copies only valid logical elements)."""

    def get(gr: int, gc: int) -> int:
        r, c = gr - tr * TILE_ROWS, gc - tc * TILE_COLS
        if r >= vr or c >= vc:
            return 0
        if transposed:
            # gr is the global K coordinate, gc the global N coordinate.
            if gr >= rows or gc >= cols:
                return 0
            v = rect[gc * stride + gr]
        else:
            if gr >= rows or gc >= cols:
                return 0
            v = rect[gr * stride + gc]
        return f32_to_h16(v) if f32 else v

    return get


def unpack_writes(croutons: bytes, dst_rows: int, dst_cols: int, dst_stride: int,
                  tile_row: int, first_block: int, n_pairs: int, f32: bool,
                  res=None, res_stride: int = 0):
    """(row, col) -> bytes for one unpack leaf call (single: n_pairs=1 at
    first_block; bulk: blocks 0..n_pairs-1). Out-of-range rows are skipped;
    only the first dst_cols columns are written (HMXAPI.h)."""
    writes: dict[tuple[int, int], bytes] = {}
    for j in range(n_pairs):
        bj = first_block + j
        for p in (0, 1):
            row = tile_row * TILE_ROWS + 2 * bj + p
            if row >= dst_rows:
                continue
            for col in range(dst_cols):
                t, c = divmod(col, TILE_COLS)
                off = t * TILE_BYTES + bj * BLOCK_BYTES + 4 * c + 2 * p
                half = croutons[off:off + 2]
                if not f32:
                    writes[(row, col)] = half
                    continue
                v = h16_to_f32(struct.unpack("<H", half)[0])
                if res is not None:
                    v += res[row * res_stride + col]
                writes[(row, col)] = f32_bytes(v)
    return writes


def bulk_pair_count(dst_rows: int, tile_row: int, n_pairs: int) -> int:
    """Both bulk unpacks cover pairs j whose first row tile_row*32 + 2j is
    inside dst_rows, capped at n_pairs (f16 clamps up front, f32 breaks at
    the first out-of-range pair; same set)."""
    base = tile_row * TILE_ROWS
    avail = (dst_rows - base + 1) // 2 if base < dst_rows else 0
    return min(n_pairs, avail)


def unpack_tail_writes(crouton: bytes, dst_rows: int, dst_cols: int,
                       dst_stride: int, tile_row: int, block_j: int,
                       valid_rows: int, valid_cols: int, f32: bool,
                       res=None, res_stride: int = 0):
    """(row, col) -> bytes for the bounds-safe unpack tails: the pair at
    block_j is unpacked to a scratch, then only the intersection with
    valid_rows/valid_cols and the logical extents is copied (HMXAPI.h)."""
    writes: dict[tuple[int, int], bytes] = {}
    first = 2 * block_j
    vr_eff = min(valid_rows - first, 2) if first < valid_rows else 0
    vc = min(valid_cols, TILE_COLS)
    for r in range(vr_eff):
        grow = tile_row * TILE_ROWS + first + r
        if grow >= dst_rows:
            break
        for c in range(vc):
            if c >= dst_cols:
                break
            off = block_j * BLOCK_BYTES + 4 * c + 2 * r
            half = crouton[off:off + 2]
            if not f32:
                writes[(grow, c)] = half
                continue
            v = h16_to_f32(struct.unpack("<H", half)[0])
            if res is not None:
                v += res[grow * res_stride + c]
            writes[(grow, c)] = f32_bytes(v)
    return writes


# ---------------------------------------------------------------------------
# Case/session harness
# ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    leaf: str
    args: list  # ints; buffer arguments are OFFSETS, resolved by the driver
    fills: list  # (buffer char, offset, bytes)
    dump_off: int
    dump_len: int
    poison: int  # the byte the driver pre-fills every buffer with
    expect: bytes
    context: dict = field(default_factory=dict)


_CASE_SEQ = itertools.count()


def next_poison() -> int:
    """A varying poison byte per case: unwritten bytes must equal it, which
    also catches stale data leaking across cases."""
    return 0xA0 | (next(_CASE_SEQ) & 0x0F)


def canvas(dump_off: int, dump_len: int, poison: int, writes: dict) -> bytes:
    out = bytearray([poison]) * dump_len
    for off, data in writes.items():
        i = off - dump_off
        assert 0 <= i and i + len(data) <= dump_len, f"write outside dump window at {off}"
        out[i:i + len(data)] = data
    return bytes(out)


def build_rect(rows: int, stride: int, cols: int, f32: bool, seq: Seq) -> tuple:
    """A [rows][stride] rectangle of values: c < cols -> generator values,
    c >= cols -> padding (never legitimately read; a leak mismatches the
    model). Returns (flat value list, fill bytes)."""
    vals: list = []
    fill = bytearray()
    for _r in range(rows):
        for c in range(stride):
            if c < cols:
                v = f32_exact(seq) if f32 else h16(seq)
                vals.append(v)
                fill += f32_bytes(v) if f32 else h16_bytes(v)
            else:
                vals.append(F32_PAD if f32 else F16_PAD)
                fill += f32_bytes(F32_PAD) if f32 else h16_bytes(F16_PAD)
    return vals, bytes(fill)


def build_crouton_row(n_ct: int, seq: Seq) -> bytes:
    """A crouton row for unpack sources: every halfword slot of every crouton
    gets a distinct finite pattern (all slots are real AR slots)."""
    fill = bytearray()
    for _ in range(n_ct):
        for _ in range(TILE_BYTES // 2):
            fill += h16_bytes(h16(seq))
    return bytes(fill)


def build_res_rect(rows: int, stride: int, seq: Seq) -> tuple:
    vals: list = []
    fill = bytearray()
    for _ in range(rows):
        for _ in range(stride):
            v = f32_exact(seq)
            vals.append(v)
            fill += f32_bytes(v)
    return vals, bytes(fill)


def pack_case(name: str, leaf: str, args: list, tiles: list, src_fill: bytes,
              res_fill: bytes | None, context: dict | None = None,
              src_off: int = SRC_OFF, dst_off: int = DST_OFF,
              res_off: int = RES_OFF) -> Case:
    poison = next_poison()
    region = TILE_BYTES * len(tiles)
    dump_off = dst_off - MARGIN
    dump_len = region + 2 * MARGIN
    writes: dict = {}
    for t, tile in enumerate(tiles):
        for i in range(0, TILE_BYTES, 2):
            writes[dst_off + t * TILE_BYTES + i] = tile[i:i + 2]
    fills = [("S", src_off, src_fill)]
    if res_fill is not None:
        fills.append(("R", res_off, res_fill))
    return Case(name, leaf, args, fills, dump_off, dump_len, poison,
                canvas(dump_off, dump_len, poison, writes), context or {})


def unpack_case(name: str, leaf: str, args: list, writes_rc: dict, dst_rows: int,
                dst_stride: int, esz: int, src_fill: bytes, res_fill: bytes | None,
                context: dict | None = None, src_off: int = SRC_OFF,
                dst_off: int = DST_OFF, res_off: int = RES_OFF) -> Case:
    poison = next_poison()
    span = dst_rows * dst_stride * esz
    dump_off = dst_off - MARGIN
    dump_len = span + 2 * MARGIN
    writes = {
        dst_off + (row * dst_stride + col) * esz: data
        for (row, col), data in writes_rc.items()
    }
    fills = [("S", src_off, src_fill)]
    if res_fill is not None:
        fills.append(("R", res_off, res_fill))
    return Case(name, leaf, args, fills, dump_off, dump_len, poison,
                canvas(dump_off, dump_len, poison, writes), context or {})


class Harness:
    """Compiles the driver (once per configuration) and runs case sessions."""

    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hmx_layout_emu_"))
        self._cache: dict = {}

    def close(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def compilers() -> list:
        cands = [os.environ.get("CC"), "cc", "gcc", "clang"]
        seen, out = set(), []
        for c in cands:
            path = shutil.which(c) if c else None
            if path and path not in seen:
                seen.add(path)
                out.append(path)
        assert out, (
            "no host C compiler found (tried $CC, cc, gcc, clang); this test "
            "compiles the real HMXLayout.c, so it needs one"
        )
        return out

    def compile(self, cc: str | None = None, opt: str = "-O2",
                layout_c: Path | None = None) -> Path:
        key = (cc, opt, str(layout_c))
        if key in self._cache:
            return self._cache[key]
        cc = cc or self.compilers()[0]
        tag = f"driver_{len(self._cache)}"
        out = self.tmp / tag
        cmd = [cc, "-std=gnu11", opt, "-fno-strict-aliasing", "-Wall",
               f"-I{EMU_DIR}", f"-I{RUNTIME / 'hmx' / 'include'}",
               str(DRIVER_C), "-o", str(out), "-pthread"]
        if layout_c is not None:
            cmd.insert(1, f'-DHMX_EMU_LAYOUT_C="{layout_c}"')
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if done.returncode != 0:
            raise AssertionError(
                f"driver compile failed ({' '.join(cmd)}):\n{done.stderr}"
            )
        self._cache[key] = out
        return out

    def run(self, cases: list, driver: Path | None = None) -> list:
        driver = driver or self.compile()
        cmds = []
        for i, case in enumerate(cases):
            cmds.append(f"case {i} {case.poison:x}")
            for buf, off, data in case.fills:
                cmds.append(f"fill {buf} {off} {data.hex()}")
            cmds.append("run " + case.leaf + " " + " ".join(str(a) for a in case.args))
            cmds.append(f"dump D {case.dump_off} {case.dump_len}")
        cmds.append("quit")
        try:
            done = subprocess.run([str(driver)], input="\n".join(cmds) + "\n",
                                  capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - loud failure
            raise AssertionError(f"driver session timed out: {exc}") from exc
        if done.returncode != 0:
            raise AssertionError(
                f"driver exited rc={done.returncode}\nstderr:\n{done.stderr[-4000:]}"
                f"\nstdout tail:\n{done.stdout[-4000:]}"
            )
        lines = done.stdout.splitlines()
        assert lines and lines[0].startswith("emu-driver ready"), lines[:1]
        addrs = [int(tok.split("=")[1]) for tok in lines[0].split()[2:]]
        assert all(a < (1 << 31) for a in addrs), f"buffers not below 2 GB: {addrs}"
        dumps = lines[1:]
        assert len(dumps) == len(cases), (
            f"expected {len(cases)} dump replies, got {len(dumps)}"
        )
        got_all = []
        for case, line in zip(cases, dumps):
            prefix = f"D {case.dump_off} "
            assert line.startswith(prefix), line[:80]
            got = bytes.fromhex(line[len(prefix):])
            assert len(got) == case.dump_len
            got_all.append(got)
        return got_all

    def check(self, cases: list, driver: Path | None = None) -> None:
        got_all = self.run(cases, driver)
        for case, got in zip(cases, got_all):
            if got == case.expect:
                continue
            bad = [i for i in range(len(got)) if got[i] != case.expect[i]]
            lines = [
                f"{case.name}: {len(bad)} of {len(got)} dump bytes mismatch "
                f"(leaf={case.leaf} args={case.args})",
            ]
            for ctx_key, ctx_val in case.context.items():
                lines.append(f"  {ctx_key}: {ctx_val}")
            for i in bad[:4]:
                a = case.dump_off + i
                lines.append(
                    f"  dump+{i} (D-buffer byte {a}): expected "
                    f"{case.expect[i:i + 4].hex()} got {got[i:i + 4].hex()}"
                )
            if len(bad) > 4:
                lines.append(f"  ... and {len(bad) - 4} more")
            raise AssertionError("\n".join(lines))


# ---------------------------------------------------------------------------
# Case families. Offsets passed to leaves are multiples of 2048 for crouton
# bases (the realistic VTCM layout, and the aligned-store paths need it).
# ---------------------------------------------------------------------------


def cases_pack_f16_rowmajor() -> list:
    cases = []
    seq = Seq(0xC0FFEE)

    def add(name, leaf, rows, cols, stride, tr, tc):
        vals, fill = build_rect(rows, stride, cols, False, seq)
        get = rowmajor_getter(vals, stride, rows, cols, f32=False)
        tile = tile_from_get(get, tr, tc)
        args = [DST_OFF, SRC_OFF, rows, cols, stride, tr, tc]
        cases.append(pack_case(name, leaf, args, [tile], fill, None,
                               context={"rows": rows, "cols": cols,
                                        "stride": stride, "tile": (tr, tc)}))

    # (rows, cols, stride, tile_row, tile_col): interior, edges, zero tiles.
    for i, (r, c, s, tr, tc) in enumerate([
        (64, 64, 64, 0, 0),   # interior
        (64, 64, 64, 1, 1),   # interior, nonzero tile coords
        (32, 32, 32, 0, 0),   # dense adjacent-rows fast path
        (40, 48, 128, 0, 0),  # strided view, full tile
        (40, 48, 128, 0, 1),  # partial columns (16)
        (21, 32, 64, 0, 0),   # partial rows
        (33, 32, 64, 1, 0),   # one valid row in tile 1
        (64, 20, 32, 0, 0),   # partial columns (20), stride 32
        (64, 20, 32, 0, 1),   # tile past the columns: zero crouton
        (32, 64, 64, 1, 0),   # tile past the rows: zero crouton
        (1, 1, 8, 0, 0),      # tiny: heavy reliance on padding reads
        (64, 96, 128, 1, 2),  # strided interior, far tile
    ]):
        add(f"act_f16_{i}", "pack_act_f16", r, c, s, tr, tc)
    for i, (k, n, s, kt, nt) in enumerate([
        (64, 64, 64, 0, 0),
        (37, 21, 64, 1, 0),   # partial k tile (5) + partial n (21)
        (37, 21, 64, 0, 1),   # n tile past n: zero crouton
        (32, 48, 128, 0, 1),  # strided N-split view
    ]):
        add(f"weight_f16_{i}", "pack_weight_f16", k, n, s, kt, nt)
    return cases


def cases_pack_f16_bulk() -> list:
    cases = []
    seq = Seq(0xBADC0DE)

    def add_act(name, rows, cols, stride, tr, start, n):
        vals, fill = build_rect(rows, stride, cols, False, seq)
        get = rowmajor_getter(vals, stride, rows, cols, f32=False)
        tiles = [tile_from_get(get, tr, start + t) for t in range(n)]
        args = [DST_OFF, SRC_OFF, rows, cols, stride, tr, start, n]
        cases.append(pack_case(name, "pack_act_f16_bulk", args, tiles, fill,
                               None, context={"rows": rows, "cols": cols,
                                              "stride": stride, "tr": tr,
                                              "start": start, "n": n}))

    def add_weight(name, k, n, stride, kts, nt, n_k):
        vals, fill = build_rect(k, stride, n, False, seq)
        get = rowmajor_getter(vals, stride, k, n, f32=False)
        tiles = [tile_from_get(get, kts + t, nt) for t in range(n_k)]
        args = [DST_OFF, SRC_OFF, k, n, stride, kts, nt, n_k]
        cases.append(pack_case(name, "pack_weight_f16_bulk", args, tiles, fill,
                               None, context={"k": k, "n": n, "stride": stride,
                                              "kts": kts, "nt": nt, "n_k": n_k}))

    add_act("act_bulk_pairs", 64, 192, 192, 0, 0, 6)   # three 2-tile passes
    add_act("act_bulk_pair_plus_singles", 64, 96, 96, 0, 0, 3)
    add_act("act_bulk_partial_rows", 40, 128, 128, 1, 0, 4)  # singles fallback
    add_act("act_bulk_odd_start", 32, 192, 192, 0, 1, 5)
    add_act("act_bulk_one_dense", 32, 32, 32, 0, 0, 1)      # adjacent-rows single
    add_act("act_bulk_edge_cols", 64, 100, 128, 0, 0, 4)    # last tile partial (4)
    add_weight("weight_bulk_run", 192, 64, 64, 0, 0, 6)
    add_weight("weight_bulk_partial_k", 37, 64, 64, 1, 0, 1)
    add_weight("weight_bulk_zero_n", 96, 32, 96, 0, 1, 3)   # n tile past n
    return cases


def cases_pack_f32() -> list:
    cases = []
    seq = Seq(0xFACEB00)

    def add(name, leaf, rows, cols, stride, tr, tc):
        vals, fill = build_rect(rows, stride, cols, True, seq)
        get = rowmajor_getter(vals, stride, rows, cols, f32=True)
        tile = tile_from_get(get, tr, tc)
        args = [DST_OFF, SRC_OFF, rows, cols, stride, tr, tc]
        cases.append(pack_case(name, leaf, args, [tile], fill, None,
                               context={"rows": rows, "cols": cols,
                                        "stride": stride, "tile": (tr, tc)}))

    for i, (r, c, s, tr, tc) in enumerate([
        (64, 64, 64, 0, 0),
        (40, 48, 128, 0, 1),  # partial columns
        (21, 32, 64, 0, 0),   # partial rows
        (64, 20, 32, 0, 0),   # partial columns, stride 32
        (64, 20, 32, 0, 1),   # zero crouton
        (1, 1, 8, 0, 0),      # tiny
        (32, 64, 64, 1, 0),   # zero crouton (rows past edge)
    ]):
        add(f"act_f32_{i}", "pack_act_f32", r, c, s, tr, tc)
    add("weight_f32_partial", "pack_weight_f32", 37, 21, 64, 1, 0)

    # bulks
    def add_bulk_act(name, rows, cols, stride, tr, start, n):
        vals, fill = build_rect(rows, stride, cols, True, seq)
        get = rowmajor_getter(vals, stride, rows, cols, f32=True)
        tiles = [tile_from_get(get, tr, start + t) for t in range(n)]
        args = [DST_OFF, SRC_OFF, rows, cols, stride, tr, start, n]
        cases.append(pack_case(name, "pack_act_f32_bulk", args, tiles, fill,
                               None))

    def add_bulk_weight(name, k, n, stride, kts, nt, n_k):
        vals, fill = build_rect(k, stride, n, True, seq)
        get = rowmajor_getter(vals, stride, k, n, f32=True)
        tiles = [tile_from_get(get, kts + t, nt) for t in range(n_k)]
        args = [DST_OFF, SRC_OFF, k, n, stride, kts, nt, n_k]
        cases.append(pack_case(name, "pack_weight_f32_bulk", args, tiles,
                               fill, None))

    add_bulk_act("act_f32_bulk_pairs", 64, 192, 192, 0, 0, 6)
    add_bulk_act("act_f32_bulk_partial", 40, 96, 96, 1, 0, 3)
    add_bulk_weight("weight_f32_bulk_run", 192, 64, 64, 0, 0, 6)
    add_bulk_weight("weight_f32_bulk_partial_k", 37, 64, 64, 1, 0, 1)
    return cases


def cases_pack_T() -> list:
    cases = []
    seq = Seq(0x5EED)

    def add(name, leaf, k, n, stride, kt, nt, f32):
        vals, fill = build_rect(n, stride, k, f32, seq)  # source is [n][stride]
        get = transposed_getter(vals, stride, k, n, f32=f32)
        tile = tile_from_get(get, kt, nt)
        args = [DST_OFF, SRC_OFF, k, n, stride, kt, nt]
        cases.append(pack_case(name, leaf, args, [tile], fill, None,
                               context={"k": k, "n": n, "stride": stride,
                                        "tile": (kt, nt), "f32": f32}))

    def add_bulk(name, leaf, k, n, stride, kts, nt, n_k, f32):
        vals, fill = build_rect(n, stride, k, f32, seq)
        get = transposed_getter(vals, stride, k, n, f32=f32)
        tiles = [tile_from_get(get, kts + t, nt) for t in range(n_k)]
        args = [DST_OFF, SRC_OFF, k, n, stride, kts, nt, n_k]
        cases.append(pack_case(name, leaf, args, tiles, fill, None,
                               context={"k": k, "n": n, "stride": stride,
                                        "kts": kts, "nt": nt, "n_k": n_k,
                                        "f32": f32}))

    # f16 singles
    add("T_f16_full", "pack_weight_f16_T", 64, 64, 64, 0, 0, False)
    add("T_f16_full_t1", "pack_weight_f16_T", 64, 64, 64, 1, 1, False)
    add("T_f16_partial_k", "pack_weight_f16_T", 37, 64, 64, 1, 0, False)
    add("T_f16_partial_n", "pack_weight_f16_T", 64, 21, 64, 0, 0, False)
    add("T_f16_strided", "pack_weight_f16_T", 20, 64, 128, 0, 0, False)
    add("T_f16_zero", "pack_weight_f16_T", 64, 64, 64, 2, 0, False)
    # f16 bulks (the 2-tile scatter lives here)
    add_bulk("T_f16_bulk_pairs", "pack_weight_f16_T_bulk", 128, 64, 64, 0, 0, 4, False)
    add_bulk("T_f16_bulk_mixed", "pack_weight_f16_T_bulk", 96, 64, 64, 0, 0, 3, False)
    add_bulk("T_f16_bulk_partial", "pack_weight_f16_T_bulk", 37, 21, 64, 1, 0, 1, False)
    add_bulk("T_f16_bulk_zero_n", "pack_weight_f16_T_bulk", 192, 32, 192, 0, 1, 6, False)
    # f32 singles and bulks -- home of the historical hmx__pack_T1_f32 bug
    add("T_f32_full", "pack_weight_f32_T", 64, 64, 64, 0, 0, True)
    add("T_f32_partial_k", "pack_weight_f32_T", 37, 21, 64, 1, 0, True)
    add("T_f32_partial_n", "pack_weight_f32_T", 64, 21, 64, 0, 0, True)
    add("T_f32_strided", "pack_weight_f32_T", 20, 64, 128, 0, 0, True)
    add_bulk("T_f32_bulk_pairs", "pack_weight_f32_T_bulk", 128, 64, 64, 0, 0, 4, True)
    add_bulk("T_f32_bulk_mixed", "pack_weight_f32_T_bulk", 96, 64, 64, 0, 0, 3, True)
    return cases


def cases_pack_tails() -> list:
    cases = []
    seq = Seq(0xD1CE)

    def add(name, leaf, rect_rows, rect_cols, stride, tr, tc, vr, vc, f32,
            transposed):
        # For transposed sources the rect is [n][stride] holding W^T.
        rows = rect_cols if transposed else rect_rows
        cols = rect_rows if transposed else rect_cols
        vals, fill = build_rect(rows, stride, cols, f32, seq)
        get = tail_getter(vals, stride, rect_rows, rect_cols, f32, vr, vc,
                          tr, tc, transposed)
        tile = tile_from_get(get, tr, tc)
        args = [DST_OFF, SRC_OFF, rect_rows, rect_cols, stride, tr, tc, vr, vc]
        cases.append(pack_case(name, leaf, args, [tile], fill, None,
                               context={"extents": (rect_rows, rect_cols),
                                        "tile": (tr, tc), "valid": (vr, vc),
                                        "f32": f32, "T": transposed}))

    # (leaf, rows, cols, stride, tr, tc, vr, vc, f32, T)
    specs = [
        ("act_f16_full", "pack_act_tail_f16", 64, 64, 64, 0, 0, 32, 32, False, False),
        ("act_f16_partial", "pack_act_tail_f16", 64, 64, 64, 0, 0, 5, 7, False, False),
        ("act_f16_rows_clip", "pack_act_tail_f16", 20, 64, 64, 0, 0, 32, 32, False, False),
        ("act_f16_tile1", "pack_act_tail_f16", 40, 64, 64, 1, 0, 8, 32, False, False),
        ("weight_f16_partial", "pack_weight_tail_f16", 37, 21, 64, 1, 0, 5, 32, False, False),
        ("act_f32_full", "pack_act_tail_f32", 64, 64, 64, 0, 0, 32, 32, True, False),
        ("act_f32_partial", "pack_act_tail_f32", 64, 64, 64, 0, 0, 5, 7, True, False),
        ("weight_f32_partial", "pack_weight_tail_f32", 37, 64, 64, 1, 0, 5, 32, True, False),
        ("T_f16_partial", "pack_weight_tail_f16_T", 64, 64, 64, 0, 0, 5, 7, False, True),
        ("T_f16_n_clip", "pack_weight_tail_f16_T", 64, 20, 64, 0, 0, 32, 32, False, True),
        ("T_f32_partial", "pack_weight_tail_f32_T", 37, 21, 64, 1, 0, 5, 5, True, True),
    ]
    for spec in specs:
        add(*spec)
    return cases


def cases_unpack_f16() -> list:
    cases = []
    seq = Seq(0xFEED)

    def add(name, rows, cols, stride, tr, bj, dst_off=DST_OFF):
        n_ct = (cols + TILE_COLS - 1) // TILE_COLS
        croutons = build_crouton_row(n_ct, seq)
        writes = unpack_writes(croutons, rows, cols, stride, tr, bj, 1, f32=False)
        args = [dst_off, SRC_OFF, rows, cols, stride, tr, bj]
        cases.append(unpack_case(name, "unpack_acc_f16", args, writes, rows,
                                 stride, 2, croutons, None,
                                 dst_off=dst_off,
                                 context={"rows": rows, "cols": cols,
                                          "stride": stride, "tr": tr,
                                          "block_j": bj}))

    add("full_chunk", 64, 64, 64, 0, 0)
    add("tail32_chunk", 64, 32, 64, 0, 5)
    add("chunk_plus_tail", 64, 96, 96, 0, 3)
    add("unrolled_256", 64, 256, 256, 0, 7)
    add("nsplit_aligned", 8, 64, 128, 0, 2)   # aligned arm: stride % 64 == 0
    add("strided_unaligned", 8, 64, 96, 0, 1)  # unaligned arm: stride 96
    add("base_unaligned", 8, 64, 64, 0, 1, dst_off=DST_OFF + 130)
    add("odd_rows_last", 33, 64, 64, 1, 0)
    add("last_block", 31, 64, 64, 0, 15)
    add("no_rows", 31, 64, 64, 1, 0)          # tile past dst_rows: nothing
    add("dense_small", 2, 32, 32, 0, 0)

    def add_bulk(name, rows, cols, stride, tr, n_pairs):
        n_ct = (cols + TILE_COLS - 1) // TILE_COLS
        croutons = build_crouton_row(n_ct, seq)
        n_eff = bulk_pair_count(rows, tr, n_pairs)
        writes = unpack_writes(croutons, rows, cols, stride, tr, 0, n_eff, f32=False)
        args = [DST_OFF, SRC_OFF, rows, cols, stride, tr, n_pairs]
        cases.append(unpack_case(name, "unpack_acc_f16_bulk", args, writes,
                                 rows, stride, 2, croutons, None,
                                 context={"rows": rows, "cols": cols,
                                          "stride": stride, "tr": tr,
                                          "n_pairs": n_pairs, "n_eff": n_eff}))

    add_bulk("bulk_full", 64, 128, 128, 0, 16)
    add_bulk("bulk_clamp", 5, 64, 64, 0, 16)
    add_bulk("bulk_clamp_tile1", 40, 64, 64, 1, 16)
    add_bulk("bulk_unaligned", 8, 64, 96, 0, 4)
    return cases


def cases_unpack_f32() -> list:
    cases = []
    seq = Seq(0xACE0)

    def add(name, rows, cols, stride, res_stride, tr, bj, has_res,
            n_pairs=None):
        n_ct = (cols + TILE_COLS - 1) // TILE_COLS
        croutons = build_crouton_row(n_ct, seq)
        res_vals, res_fill = build_res_rect(rows, res_stride, seq)
        res = res_vals if has_res else None
        if n_pairs is None:
            writes = unpack_writes(croutons, rows, cols, stride, tr, bj, 1,
                                   True, res=res, res_stride=res_stride)
            args = [DST_OFF, RES_OFF, 1 if has_res else 0, SRC_OFF, rows,
                    cols, stride, res_stride, tr, bj]
            leaf = "unpack_acc_f32"
        else:
            n_eff = bulk_pair_count(rows, tr, n_pairs)
            writes = unpack_writes(croutons, rows, cols, stride, tr, 0, n_eff,
                                   True, res=res, res_stride=res_stride)
            args = [DST_OFF, RES_OFF, 1 if has_res else 0, SRC_OFF, rows,
                    cols, stride, res_stride, tr, n_pairs]
            leaf = "unpack_acc_f32_bulk"
        cases.append(unpack_case(name, leaf, args, writes, rows, stride, 4,
                                 croutons, res_fill if has_res else None,
                                 context={"rows": rows, "cols": cols,
                                          "stride": stride,
                                          "res_stride": res_stride,
                                          "has_res": has_res}))

    add("no_res_aligned", 64, 64, 64, 64, 0, 0, False)
    add("res_aligned", 64, 64, 64, 64, 0, 0, True)
    add("res_unaligned", 8, 64, 96, 96, 0, 2, True)   # stride 96: unaligned arm
    add("res_pred_tail16", 8, 48, 64, 96, 0, 1, True)  # dst_cols 48: tail 16
    add("res_pred_tail18", 8, 50, 64, 128, 0, 0, True)  # dst_cols 50: tail 18
    add("res_odd_rows", 33, 96, 96, 96, 1, 0, True)
    add("res_last_block", 64, 32, 32, 32, 0, 15, True)
    add("res_bulk_full", 64, 128, 128, 128, 0, 0, True, n_pairs=16)
    add("res_bulk_clamp", 6, 64, 64, 64, 0, 0, True, n_pairs=16)
    add("res_bulk_nothing", 20, 64, 64, 64, 1, 0, True, n_pairs=16)  # past rows
    return cases


def cases_unpack_tails() -> list:
    cases = []
    seq = Seq(0xBEEF)

    def add(name, rows, cols, stride, res_stride, tr, bj, vr, vc, f32, has_res):
        crouton = build_crouton_row(1, seq)
        res_vals, res_fill = build_res_rect(rows, res_stride, seq) if has_res else (None, None)
        writes = unpack_tail_writes(crouton, rows, cols, stride, tr, bj, vr, vc,
                                    f32, res=res_vals, res_stride=res_stride)
        if f32:
            args = [DST_OFF, RES_OFF, 1 if has_res else 0, SRC_OFF, rows, cols,
                    stride, res_stride, tr, bj, vr, vc]
            leaf = "unpack_acc_tail_f32"
        else:
            args = [DST_OFF, SRC_OFF, rows, cols, stride, tr, bj, vr, vc]
            leaf = "unpack_acc_tail_f16"
        cases.append(unpack_case(name, leaf, args, writes, rows, stride,
                                 4 if f32 else 2, crouton,
                                 res_fill if has_res else None,
                                 context={"rows": rows, "cols": cols,
                                          "tile": (tr, bj), "valid": (vr, vc),
                                          "f32": f32, "res": has_res}))

    add("f16_full", 64, 64, 64, 0, 0, 0, 32, 32, False, False)
    add("f16_partial", 64, 64, 64, 0, 0, 3, 5, 7, False, False)
    add("f16_vr_spans_pair", 64, 64, 64, 0, 0, 1, 3, 32, False, False)
    add("f16_one_by_one", 64, 64, 64, 0, 0, 0, 1, 1, False, False)
    add("f16_global_clip", 20, 48, 64, 0, 0, 0, 32, 32, False, False)
    add("f32_partial_res", 64, 64, 64, 64, 0, 2, 5, 7, True, True)
    add("f32_full_res", 33, 96, 96, 96, 1, 0, 32, 32, True, True)
    return cases


def smoke_cases() -> list:
    return (
        cases_pack_f16_rowmajor()[:1]
        + cases_pack_f16_bulk()[:1]
        + cases_pack_f32()[:1]
        + cases_pack_T()[10:12]   # the f32 T singles (historical bug's home)
        + cases_pack_tails()[1:2]
        + cases_unpack_f16()[:1]
        + cases_unpack_f32()[3:4]  # predicated tail with residual
        + cases_unpack_tails()[1:2]
    )


# ---------------------------------------------------------------------------
# Negative controls: inject known-wrong tables into scratch copies and
# assert the battery goes red (and pinpointedly so).
# ---------------------------------------------------------------------------


def _table_values(source: str, name: str) -> list:
    m = re.search(rf"{name}\[64\][^=]*=\s*\{{(.*?)\}};", source, re.S)
    assert m, f"table {name} not found in HMXLayout.c"
    vals = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    assert len(vals) == 64, f"table {name}: expected 64 values, got {len(vals)}"
    return vals


def _rewrite_table(source: str, name: str, vals: list) -> str:
    m = re.search(rf"{name}\[64\][^=]*=\s*\{{(.*?)\}};", source, re.S)
    assert m, f"table {name} not found in HMXLayout.c"
    body = ", ".join(str(v) for v in vals)
    return source[:m.start(1)] + body + source[m.end(1):]


def corrupted_layout_c(harness: Harness, tag: str, transform) -> Path:
    """A scratch copy of HMXLayout.c with one table corrupted, its HMXAPI.h
    include redirected to the real header (the copy lives in a temp dir)."""
    src = LAYOUT_C.read_text(encoding="utf-8")
    src = transform(src)
    src = src.replace('#include "HMXAPI.h"', f'#include "{HMXAPI_H}"')
    path = harness.tmp / f"HMXLayout_{tag}.c"
    path.write_text(src, encoding="utf-8")
    return path


def replay_t1_f32_bug(src: str) -> str:
    """The historical bug, verbatim: hmx__pack_T1_f32 carrying the
    concatenated-form hmx__pack_T1 table while being fed the interleaved
    conversion output (HMXLayout.c documents it: 'Feeding the conversion's
    output to T1 itself scrambles the tile')."""
    return _rewrite_table(src, "hmx__pack_T1_f32", _table_values(src, "hmx__pack_T1"))


def corrupt_pack_offsets(src: str) -> str:
    """Swap two adjacent offsets in the base scatter table."""
    vals = _table_values(src, "hmx__pack_offsets")
    vals[1], vals[2] = vals[2], vals[1]
    return _rewrite_table(src, "hmx__pack_offsets", vals)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


class HmxLayoutPermutationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = Harness()
        assert LAYOUT_C.is_file(), f"missing production file: {LAYOUT_C}"
        assert DRIVER_C.is_file() and EMU_DIR.is_dir()

    @classmethod
    def tearDownClass(cls):
        cls.harness.close()

    def test_physical_constants_match_hmxapi_header(self):
        # The model's constants are parsed from the runtime header, and the
        # closed form's shape is asserted here (not silently adapted).
        self.assertEqual((TILE_ROWS, TILE_COLS), (32, 32))
        self.assertEqual(TILE_BYTES, TILE_ROWS * TILE_COLS * 2)
        self.assertEqual(BLOCK_PAIRS * BLOCK_BYTES, TILE_BYTES)

    def test_pack_f16_rowmajor_single(self):
        self.harness.check(cases_pack_f16_rowmajor())

    def test_pack_f16_bulk(self):
        self.harness.check(cases_pack_f16_bulk())

    def test_pack_f32_rowmajor_and_bulk(self):
        self.harness.check(cases_pack_f32())

    def test_pack_transposed_sources(self):
        self.harness.check(cases_pack_T())

    def test_pack_bounds_safe_tails(self):
        self.harness.check(cases_pack_tails())

    def test_unpack_f16_single_and_bulk(self):
        self.harness.check(cases_unpack_f16())

    def test_unpack_f32_single_and_bulk(self):
        self.harness.check(cases_unpack_f32())

    def test_unpack_bounds_safe_tails(self):
        self.harness.check(cases_unpack_tails())

    def test_negative_control_replays_the_historical_t1_f32_bug(self):
        # The exact bug class that motivated this file, replayed against a
        # scratch copy: the battery MUST go red on the f32 transposed packs,
        # and MUST stay green on the f16 packs (the corruption is pinpointed).
        path = corrupted_layout_c(self.harness, "t1f32_bug", replay_t1_f32_bug)
        driver = self.harness.compile(layout_c=path)
        with self.assertRaises(AssertionError) as caught:
            self.harness.check([c for c in cases_pack_T() if "T_f32" in c.name],
                               driver)
        self.assertIn("mismatch", str(caught.exception))
        self.harness.check([c for c in cases_pack_T() if "T_f16" in c.name], driver)

    def test_negative_control_corrupted_scatter_table(self):
        path = corrupted_layout_c(self.harness, "offsets_bug", corrupt_pack_offsets)
        driver = self.harness.compile(layout_c=path)
        with self.assertRaises(AssertionError) as caught:
            self.harness.check(cases_pack_f16_rowmajor(), driver)
        self.assertIn("mismatch", str(caught.exception))
        # unpack reads croutons through the same closed form and does not use
        # the pack offsets table, so it must stay green.
        self.harness.check(cases_unpack_f16()[:3], driver)

    def test_compiler_variation_smoke(self):
        # The same battery under other compilers / optimization levels, as a
        # cheap UB sentry: the production file's punning must not depend on
        # one compiler's mercy.
        compilers = self.harness.compilers()
        configs = [(compilers[0], "-O0")]
        if len(compilers) > 1:
            configs.append((compilers[1], "-O2"))
        for cc, opt in configs:
            driver = self.harness.compile(cc=cc, opt=opt)
            self.harness.check(smoke_cases(), driver)


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        HmxLayoutPermutationContract
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total = suite.countTestCases()
    print(
        f"HMX layout permutation contract: "
        f"{'PASS' if result.wasSuccessful() else 'FAIL'} ({total} tests)"
    )
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
