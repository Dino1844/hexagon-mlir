#!/usr/bin/env python3
"""Host-runnable contract test for the thread-role dispatch channel.

Why this file exists
--------------------
``hexagon_runtime_hmx_role_channel_launch`` (in
``bin/runtime/multithreading/HmxRoleExecutor.cpp``) is the S3 launch-side
probe of the per-kernel symbol channel (``bin/runtime/include/
HmxRoleChannel.h``, ROADMAP1001 section 4.6): it dlsym's ``<entry>__hmx_
section`` + ``<entry>__hmx_role_depth`` in the loaded module and binds what
it finds. Until now nothing could say whether that decision is right without
a device, because the decision is exactly about dynamically loaded modules.

How it works
------------
The sibling contract test (``test_hmx_spsc_ring_contract.py``) compiles the
real executor into an executable driver. This one compiles it into STUB
MODULES -- ``.so`` files standing in for compiled kernels -- because the
probe's input is a dlopen handle:

  * each variant compiles the REAL ``HmxRoleExecutor.cpp`` verbatim (via the
    same ``#include`` the spsc driver uses, so the production file is the
    code under test), plus a generated stub that plays the COMPILER side of
    the channel: it exports the section entry point and the depth object a
    dual-role kernel's LLVM object would carry, spelled literally;
  * the test dlopens the variant and calls the real probe through ctypes,
    with the module's own dlopen handle -- the same role the generated
    wrapper's ``RTLD_SELF`` plays on the device;
  * what each variant exports is the ONLY thing that varies, so the return
    code isolates the probe's decision, not the executor's (whose lifecycle
    the spsc test already locks).

The third-spelling guard
------------------------
The stub spells the symbol names LITERALLY (C cannot paste a string-literal
macro into an identifier), while the probe builds them from the header's
macros. That makes the stub an independent third spelling next to the
probe's (macros) and the compiler emission's (macros, via the same header):
if the suffix macros drift without the emission and the stub following, the
dual variants stop being found and this test fails. A Python-side regex
check on the header adds a readable failure message for the same drift.

What it locks
-------------
* The decision table: neither symbol -> ``HMX_ROLE_CHANNEL_LEGACY`` and
  nothing bound (a submit afterwards is refused, accepted == 0); both ->
  ``HMX_ROLE_CHANNEL_BOUND`` and the executor really holds the section (a
  submit + drain runs the stub section, and the ring's capacity equals the
  depth the companion object carried, proven by a submit of capacity+1
  groups accepting exactly capacity); exactly one symbol -> refused loudly;
  depth 0 -> refused (bind's own depth precondition).
* The half-emitted refusal: the channel must not guess a depth and must not
  fall back to legacy -- both would hand back a wrong answer, so the return
  is negative and nothing is bound.
* The suffix agreement (above) and the argument refusals (null handle,
  null name).
* The end-to-end handoff through the REAL ring: the stub section receives
  one call per descriptor with count == 1 (the per-item calling discipline
  pinned in HmxRoleExecutor.h) and reads the descriptor's fields bit-exact,
  and the bound thread took the HMX lock exactly once before any of it.

What it does NOT lock (residual risk, stated honestly)
------------------------------------------------------
* **The device dlsym scope.** The probe runs here under glibc, where a
  dlopen handle names the module exactly. On the device the wrapper passes
  ``RTLD_SELF`` inside the FastRPC-loaded .so; whether that handle's search
  covers the .so's own symbols is a device property, marked [未验证] in
  HmxRoleExecutor.cpp, and belongs to the S3 device window.
* **The compiler emission.** That ``HmxToLLVMPass`` really exports the two
  symbols with these names is locked where the emission is testable (the
  lit tests of the role split), not here -- this test proves the RUNTIME
  half of the handshake.
* The HMX lock's blocking semantics (same limit as the spsc test; the stub
  ensure records and returns).
* The 24-byte target layout of HmxTileGroupDesc: the host struct is larger
  (64-bit padding); the target size is pinned by the header's own
  static_assert, which every side compiles.

Run: python test_hmx_role_channel_contract.py
     python -m pytest -q test_hmx_role_channel_contract.py
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
EMU_DIR = TEST_DIR / "hmx_role_host_emu"
RUNTIME = TEST_DIR.parent  # bin/runtime
RUNTIME_INCLUDE = RUNTIME / "include"
CHANNEL_H = RUNTIME_INCLUDE / "HmxRoleChannel.h"
ROLE_EXECUTOR_CPP = RUNTIME / "multithreading" / "HmxRoleExecutor.cpp"

# The kernel entry name every variant stands in for. The section and depth
# symbols are this plus the suffix spellings -- literally in the stub, from
# the macros in the probe.
ENTRY = "emu_kernel"
SECTION_SYMBOL = ENTRY + "__hmx_section"
DEPTH_SYMBOL = ENTRY + "__hmx_role_depth"

# Return codes, restated from HmxRoleChannel.h so a failure reads as a name
# rather than a bare integer (the header remains the single source; if these
# drift the tests below fail and point here first).
RC_BOUND = 1
RC_LEGACY = 0
RC_ERR_ARG = -1
RC_ERR_HALF = -3
RC_ERR_BIND = -4

# The suffix spellings the stub and the Python pre-check assume. See
# "The third-spelling guard" in the module docstring.
EXPECTED_SECTION_SUFFIX = "__hmx_section"
EXPECTED_DEPTH_SUFFIX = "__hmx_role_depth"

# The one marker value the end-to-end arm threads through the descriptor's
# event word: distinctive in both halves, and never mistaken for a pointer.
EVENT_MARKER = 0x1122334455667788


class Group(ctypes.Structure):
    """HmxTileGroupDesc with the header's field ORDER.

    The HOST layout is not the target layout (the event word's alignment
    pads on a 64-bit host); the target's 24 bytes are pinned by the
    static_assert in HmxSpscRing.h, which every side of the ABI compiles.
    This mirror exists only to hand the probe real descriptors.
    """

    _fields_ = [
        ("slot", ctypes.c_uint32),
        ("rowStart", ctypes.c_uint32),
        ("rowCount", ctypes.c_uint32),
        ("event", ctypes.c_uint64),
    ]


def _compilers() -> list[str]:
    cands = [os.environ.get("CXX"), "c++", "g++", "clang++"]
    seen, out = set(), []
    for c in cands:
        path = shutil.which(c) if c else None
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    assert out, "no host C++ compiler found (tried $CXX, c++, g++, clang++)"
    return out


class Variant:
    """One stub module: the real executor plus a generated compiler-side stub.

    ``kind`` picks what the stub exports, which is the only variable the
    probe sees:
      "dual"    section entry point + depth object (a healthy dual-role kernel)
      "legacy"  neither (every kernel compiled today)
      "section" the entry point only (a half-emitted channel)
      "depth"   the depth object only (the other half)
      "zero"    both, but the depth object holds 0 (bind's refusal)
    """

    DEPTHS = {"dual": 4, "zero": 0}

    def __init__(self, harness: "Harness", kind: str):
        self.kind = kind
        self.depth = self.DEPTHS.get(kind)
        src = harness.tmp / f"stub_{kind}.cc"
        src.write_text(self._source())
        so = harness.tmp / f"lib{kind}.so"
        cmd = [
            harness.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra",
            "-Wno-unused-parameter", "-fPIC", "-shared",
            f"-I{EMU_DIR}", f"-I{RUNTIME_INCLUDE}",
            str(src), "-o", str(so), "-pthread",
        ]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        assert done.returncode == 0, (
            f"variant {kind} compile failed ({' '.join(cmd)}):\n{done.stderr}"
        )
        self.lib = ctypes.CDLL(str(so))
        self._so_path = so
        # The dlopen handle of THIS module: what the device wrapper spells
        # RTLD_SELF for. CDLL keeps it as an integer; the probe takes void*.
        self.handle = ctypes.c_void_p(self.lib._handle)
        self.launch = self.lib.hexagon_runtime_hmx_role_channel_launch
        self.launch.restype = ctypes.c_int32
        self.launch.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.submit = self.lib.hexagon_runtime_hmx_role_submit
        self.submit.restype = ctypes.c_uint32
        self.submit.argtypes = [ctypes.POINTER(Group), ctypes.c_uint32]
        self.drain = self.lib.hexagon_runtime_hmx_role_drain
        self.drain.restype = None
        self.join = self.lib.hexagon_runtime_hmx_role_join
        self.join.restype = None

    def _section_stub(self) -> str:
        return f"""
extern "C" {{
uint32_t emu_section_calls = 0;
uint32_t emu_section_violations = 0;
uint32_t emu_last_slot = 0xDEADBEEFu;
uint32_t emu_last_rowStart = 0xDEADBEEFu;
uint32_t emu_last_rowCount = 0xDEADBEEFu;
uint64_t emu_last_event = 0xDEADBEEFDEADBEEFull;
}}

// The section entry point a dual-role kernel's LLVM object exports. The name
// is spelled LITERALLY (C cannot paste a string-literal macro into an
// identifier): this is the independent third spelling the module docstring
// describes -- the probe builds the name from HmxRoleChannel.h's macros, so
// a suffix that drifts makes the dual variants stop being found and the
// test fail.
extern "C" void {SECTION_SYMBOL}(const HmxTileGroupDesc *groups,
                                 uint32_t count) {{
  if (count != 1u)
    emu_section_violations++;  // per-item discipline: one group per call
  for (uint32_t i = 0; i < count; ++i) {{
    emu_last_slot = groups[i].slot;
    emu_last_rowStart = groups[i].rowStart;
    emu_last_rowCount = groups[i].rowCount;
    emu_last_event = groups[i].event;
  }}
  emu_section_calls++;
}}

// The companion depth object: default-visibility uint32_t, exactly what the
// compiler emission exports for a dual-role kernel.
extern "C" uint32_t {DEPTH_SYMBOL} = {self.depth};
"""

    def _source(self) -> str:
        if self.kind in ("dual", "zero"):
            stub = self._section_stub()
        elif self.kind == "legacy":
            stub = """
// A legacy kernel module: no channel symbols at all. The probe must return
// HMX_ROLE_CHANNEL_LEGACY and bind nothing.
"""
        elif self.kind == "section":
            stub = f"""
extern "C" void {SECTION_SYMBOL}(const HmxTileGroupDesc *, uint32_t) {{}}
// Half a channel: the entry point without the depth companion. The probe
// must refuse (HMX_ROLE_CHANNEL_ERR_HALF), not guess and not fall back.
"""
        elif self.kind == "depth":
            stub = f"""
extern "C" uint32_t {DEPTH_SYMBOL} = 4;
// The other half: the depth companion without the entry point.
"""
        else:
            raise AssertionError(f"unknown variant kind {self.kind!r}")

        # The HMX-lock recorder, then the production executor verbatim --
        # same arrangement as hmx_role_host_emu/driver.cc, so the code under
        # test is the file that ships, unmodified.
        return f"""/* Generated by test_hmx_role_channel_contract.py -- do not edit. */
#include <atomic>
#include <cstdint>

#include "qurt_thread.h"
#include "qurt_futex.h"
#include "HAP_farf.h"

static std::atomic<uint64_t> gEnsureCalls{{0}};
extern "C" void hexagon_runtime_hmx_ensure_dsp(void) {{
  gEnsureCalls.fetch_add(1, std::memory_order_relaxed);
}}
extern "C" uint64_t emu_ensure_calls(void) {{
  return gEnsureCalls.load(std::memory_order_relaxed);
}}

#include "../../multithreading/HmxRoleExecutor.cpp"

{stub}
"""

    def ensure_calls(self) -> int:
        fn = self.lib.emu_ensure_calls
        fn.restype = ctypes.c_uint64
        return int(fn())

    def global_u32(self, name: str) -> int:
        return ctypes.c_uint32.in_dll(self.lib, name).value

    def global_u64(self, name: str) -> int:
        return ctypes.c_uint64.in_dll(self.lib, name).value

    def close(self) -> None:
        # Join first (drains, reaps the bound thread), then drop the CDLL.
        # Without the join the dlclose at GC could race the resident thread
        # on a loaded host -- the same discipline the spsc driver's quit
        # path exists for.
        try:
            self.join()
        except Exception:
            pass
        self.lib = None


class Harness:
    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hmx_role_channel_"))
        self.cxx = _compilers()[0]
        self._variants: dict[str, Variant] = {}

    def variant(self, kind: str) -> Variant:
        if kind not in self._variants:
            self._variants[kind] = Variant(self, kind)
        return self._variants[kind]

    def close(self) -> None:
        for v in self._variants.values():
            v.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


HARNESS: Harness | None = None


def setUpModule() -> None:
    global HARNESS
    HARNESS = Harness()


def tearDownModule() -> None:
    if HARNESS:
        HARNESS.close()


class SuffixSpellingTests(unittest.TestCase):
    """The header's macros spell what everyone else assumes they spell."""

    def test_suffix_macros(self) -> None:
        text = CHANNEL_H.read_text()
        section = re.search(
            r'#define\s+HMX_ROLE_SECTION_SUFFIX\s+"([^"]+)"', text)
        depth = re.search(
            r'#define\s+HMX_ROLE_DEPTH_SUFFIX\s+"([^"]+)"', text)
        self.assertIsNotNone(section, "HMX_ROLE_SECTION_SUFFIX not found")
        self.assertIsNotNone(depth, "HMX_ROLE_DEPTH_SUFFIX not found")
        assert section and depth
        self.assertEqual(section.group(1), EXPECTED_SECTION_SUFFIX)
        self.assertEqual(depth.group(1), EXPECTED_DEPTH_SUFFIX)


class ChannelDecisionTests(unittest.TestCase):
    """The probe's decision table, one variant per outcome."""

    def test_argument_refusals(self) -> None:
        v = HARNESS.variant("dual")
        null = ctypes.c_void_p(0)
        self.assertEqual(v.launch(null, ENTRY.encode()), RC_ERR_ARG)
        self.assertEqual(v.launch(v.handle, None), RC_ERR_ARG)

    def test_dual_kernel_binds_and_runs(self) -> None:
        v = HARNESS.variant("dual")
        depth = v.depth
        assert depth is not None

        # Before the probe: the executor is not bound, so a submit is
        # refused with zero accepted (the executor's own contract; this is
        # the "nothing was touched yet" baseline).
        groups = (Group * 1)()
        self.assertEqual(v.submit(groups, 1), 0)

        rc = v.launch(v.handle, ENTRY.encode())
        self.assertEqual(rc, RC_BOUND)

        # The ring capacity is the depth the companion object carried
        # (bind(fn, depth)): submitting capacity+1 groups accepts exactly
        # capacity -- the full check at work, and the proof the depth really
        # reached bind rather than a default.
        groups = (Group * (depth + 1))()
        for i in range(depth + 1):
            groups[i].slot = i
            groups[i].rowStart = 100 + i
            groups[i].rowCount = 1
            groups[i].event = EVENT_MARKER
        accepted = v.submit(groups, depth + 1)
        self.assertEqual(accepted, depth)
        v.drain()

        # One section call per accepted descriptor, count == 1 each, and
        # the last descriptor's fields arrived bit-exact.
        self.assertEqual(v.global_u32("emu_section_calls"), depth)
        self.assertEqual(v.global_u32("emu_section_violations"), 0)
        self.assertEqual(v.global_u32("emu_last_slot"), depth - 1)
        self.assertEqual(v.global_u32("emu_last_rowStart"), 100 + depth - 1)
        self.assertEqual(v.global_u32("emu_last_rowCount"), 1)
        self.assertEqual(v.global_u64("emu_last_event"), EVENT_MARKER)

        # The bound thread took the HMX lock exactly once, before any
        # section ran (drain has happened, so the thread has started; the
        # order ensure-before-section is roleThreadEntry's own program
        # order, and exactly-once is the lifetime-lock claim this channel
        # stands on).
        self.assertEqual(v.ensure_calls(), 1)

        # Idempotent per launch: a second probe for the same kernel binds
        # the same section again (rebind re-stores; no new thread, no new
        # ensure) and stays BOUND.
        rc2 = v.launch(v.handle, ENTRY.encode())
        self.assertEqual(rc2, RC_BOUND)
        self.assertEqual(v.ensure_calls(), 1)

    def test_half_emitted_depth_only_refused(self) -> None:
        v = HARNESS.variant("depth")
        self.assertEqual(v.launch(v.handle, ENTRY.encode()), RC_ERR_HALF)

    def test_half_emitted_section_only_refused(self) -> None:
        v = HARNESS.variant("section")
        self.assertEqual(v.launch(v.handle, ENTRY.encode()), RC_ERR_HALF)

    def test_legacy_kernel_returns_legacy_and_binds_nothing(self) -> None:
        v = HARNESS.variant("legacy")
        rc = v.launch(v.handle, ENTRY.encode())
        self.assertEqual(rc, RC_LEGACY)
        # Nothing was bound: a submit is still refused (the executor's
        # submit-before-bind path), which is what keeps the legacy path
        # byte-for-byte today's -- the probe touched nothing, and no thread
        # was ever created (the ensure recorder stayed at zero).
        groups = (Group * 1)()
        self.assertEqual(v.submit(groups, 1), 0)
        self.assertEqual(v.ensure_calls(), 0)

    def test_zero_depth_refused_by_bind(self) -> None:
        v = HARNESS.variant("zero")
        self.assertEqual(v.launch(v.handle, ENTRY.encode()), RC_ERR_BIND)


if __name__ == "__main__":
    unittest.main(verbosity=2)
