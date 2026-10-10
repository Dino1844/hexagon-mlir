#!/usr/bin/env python3
"""The OBJECT-level gate for the HMX diagnostics split (host only, no device).

WHY THIS FILE EXISTS
--------------------
Ticket 23 moved the VTCM accounting census out of the production pipeline. The
IR-level gates can only prove the production pipeline *schedules* no census
(`hmx-production-pipeline-no-diagnostics.mlir`); they cannot prove the census
code is not in the binary. A feature can be green at every IR level and still be
linked in, and the whole point of the split is that it is not.

So this file reads the OBJECT with `llvm-nm` and asserts the census pass is
absent from the backend library, and that the runtime VTCM accounting probe is
absent from the device runtime in the default build. Both are the same kind of
claim: "the diagnostic code does not ship".

WHAT IS ASSERTED
----------------
  * The census pass is NOT in `libtriton.so`: neither its `runOnOperation` nor
    its constructor symbol. The census is reachable only from
    `hmx-diagnostic-record`, which lives in the HmxDiagnostics library, linked
    into `linalg-hexagon-opt` only.
  * The census pass IS still present in the `libHmxDiagnostics.a` the opt tool
    links. Without this arm the first assertion would also pass if the census
    had been deleted outright -- a gate that cannot tell "split" from "gone" is
    not a gate.
  * The runtime VTCM accounting probe (`HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE`)
    is off by default and contributes no runtime module. The device bitcode is
    LLVM 19 (SDK clang) and the host tools are LLVM 18, so the probe's *symbols*
    cannot be read from here -- that check needs the SDK's own disassembler and is
    listed as a follow-up. What is assertable on the host is that the gate is
    closed: the CMake option defaults OFF, no probe-only translation unit is in
    `HEXAGON_RUNTIME_SRC`, and no module descriptor for it is in the archive. A
    `nm` arm over `libhexagon_runtime.a` pins the last one.
  * The record-only finaliser (`hmx-v3-record`) deliberately REMAINS reachable
    from the production path: the document it finalises is published by
    `matmul-to-hmx`, so its library is already in the production library, and the
    host contract tests receive the finished document through
    `translate_linalg_to_obj`. Only the census moved. This arm pins that too, so
    a future change that moves the record pass out does not silently change what
    this gate means.

WHAT IS NOT ASSERTED
--------------------
The probe's *logic* still lives beside the allocator it measures (VTCMPool.h /
VTCMPool.cpp). Moving it into its own translation unit is a much larger change
than the ticket modelled -- the probe declares ~2800 lines of `#ifdef`-gated
code across nine files and is part of the C ABI `HexagonCAPI` exposes. What is
asserted here is only that none of it is in the default build.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
_TBLD = os.environ.get("TRITON_BUILD_DIR") or str(
    _REPO
    / "triton"
    / "build"
    / f"cmake.linux-x86_64-cpython-{sys.version_info.major}.{sys.version_info.minor}"
)

_LIBTRITON = _REPO / "triton" / "python" / "triton" / "_C" / "libtriton.so"
_DIAGNOSTICS_LIB = (
    Path(_TBLD)
    / "third_party/qcom_hexagon_backend/lib/Dialect/Hmx/Transforms/libHmxDiagnostics.a"
)
_OPT_TOOL = (
    Path(_TBLD) / "third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt"
)
_RUNTIME_LIB = (
    Path(_TBLD)
    / "third_party/qcom_hexagon_backend/bin/runtime/libhexagon_runtime.a"
)

_LLVM_NM = shutil.which("llvm-nm") or shutil.which("llvm-nm-14")

# Symbols that exist only if the code is linked. `runOnOperation` is the one that
# cannot be dead-stripped away, because the pass manager calls it through a
# vtable.
_CENSUS_MARKERS = (
    "HmxVtcmAccountingPass",
    "createHmxVtcmAccountingPass",
)
_RECORD_MARKERS = (
    "HmxRecordV3Pass",
    "createHmxRecordV3Pass",
)
_PROBE_MARKERS = (
    "AccountingSnapshot",
    "AccountingContext",
    "writeAccountingReport",
    "recordAccountingCacheEvent",
)


def _nm(obj: Path, min_symbols: int = 100) -> str:
    """Demangled symbol table of one object.

    `min_symbols` guards against reading the wrong object and passing vacuously.
    The device runtime archive legitimately carries only one descriptor symbol
    per bitcode module it embeds, so its floor is much lower than a shared
    library's.
    """
    if not _LLVM_NM:
        raise unittest.SkipTest("llvm-nm not found; this gate reads objects")
    if not obj.exists():
        raise unittest.SkipTest(f"{obj} is not built; run the build first")
    out = subprocess.run(
        [_LLVM_NM, "-C", str(obj)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, f"llvm-nm failed on {obj}: {out.stderr[-400:]}"
    assert len(out.stdout.splitlines()) > min_symbols, (
        f"llvm-nm returned only {len(out.stdout.splitlines())} symbols for {obj}; "
        "the gate would pass vacuously"
    )
    return out.stdout


class HmxDiagnosticsSplitGate(unittest.TestCase):
    """The census does not ship; the probe does not ship; the record still does."""

    @classmethod
    def setUpClass(cls):
        cls.libtriton = _nm(_LIBTRITON)

    def test_census_is_not_in_the_backend_library(self):
        for marker in _CENSUS_MARKERS:
            self.assertNotIn(
                marker,
                self.libtriton,
                f"{marker} is linked into libtriton.so: the production pipeline "
                "must not carry the VTCM accounting census",
            )

    def test_census_is_reachable_from_the_opt_tool(self):
        """The split, not the deletion: the census still exists somewhere."""
        self.assertTrue(
            _OPT_TOOL.exists(),
            "linalg-hexagon-opt is not built; the opt tool is the only entry "
            "point for the census",
        )
        symbols = _nm(_DIAGNOSTICS_LIB)
        for marker in _CENSUS_MARKERS:
            self.assertIn(marker, symbols)

    def test_record_finaliser_is_still_reachable_from_production(self):
        for marker in _RECORD_MARKERS:
            self.assertIn(
                marker,
                self.libtriton,
                f"{marker} is gone from libtriton.so: the v3 record document is "
                "delivered through translate_linalg_to_obj, which uses the "
                "production pipeline",
            )

    def test_runtime_probe_gate_is_closed_by_default(self):
        cmake = (_BACKEND / "bin" / "runtime" / "CMakeLists.txt").read_text()
        self.assertRegex(
            cmake,
            r'option\(\s*HEXMLIR_VTCM_ACCOUNTING_PROBE\s+"[^"]*"\s+OFF\s*\)',
            "the runtime VTCM accounting probe must stay OFF by default",
        )
        self.assertNotIn(
            "VtcmAccountingProbe",
            cmake,
            "a probe-only translation unit in HEXAGON_RUNTIME_SRC would be the "
            "'measure the measured thing inside the measured thing' structure "
            "coming back as a separate file",
        )

    def test_runtime_probe_contributes_no_module(self):
        symbols = _nm(_RUNTIME_LIB, min_symbols=8)
        for marker in _PROBE_MARKERS:
            self.assertNotIn(
                marker,
                symbols,
                f"{marker} names a runtime module in libhexagon_runtime.a: "
                "HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE must stay off by default",
            )


if __name__ == "__main__":
    unittest.main()
