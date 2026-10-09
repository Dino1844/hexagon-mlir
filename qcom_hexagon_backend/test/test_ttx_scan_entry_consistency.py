#!/usr/bin/env python3
"""Host-only regression: the tt -> ttx dialect-boundary rewrite is entry-agnostic.

ARCH-REVIEW #7.  ``compiler.py`` rewrites ``tt.scan`` -> ``ttx.scan`` so the
TTX lowering (``lower-ttx``) can handle Triton's scan op -- ``TT_ScanOp`` has no
``assemblyFormat``, so the op prints in MLIR generic form as ``"tt.scan"``, and
its terminator prints bare as ``tt.scan.return``.  The rewrite used to live only
inside ``ttsharedir_to_obj``; the sibling entry ``ttsharedir_to_llir`` (the
``llir`` stage) never rewrote, so the *same* ``tt.scan`` module produced an
object through one entry and failed to lower through the other.  That is a
divergence between two compile entries over identical input.

Both entries now parse through the shared ``_parse_ttsharedir``, which applies
the rewrite exactly once.  What these tests pin:

1. **Both entries accept the same ``tt.scan`` input.**  Before the fix
   ``ttsharedir_to_llir`` raised (``Failed to convert Triton Linalg to LLVM
   MLIR``) while ``ttsharedir_to_obj`` succeeded -- the divergence itself.
2. **The rewrite is a pure dialect rename, not a semantic change.**  The LLVM
   IR of a ``tt.scan`` module is byte-identical to its ``ttx.scan`` twin, and so
   is the object code -- the positive path compiles to the same artifact.
3. **Comments and data strings are left alone.**  The old regex rewrote
   ``tt.scan`` wherever it appeared, including inside ``//`` comments and quoted
   attribute strings; the rewrite now moves only genuine op references.

Host-only: drives the real compile entry points, never constructs a launcher and
never touches a device.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import unittest

# Boot the top-level triton package before any per-test
# ``triton.backends.qcom_hexagon_backend...`` import.  Starting the first triton
# import at the backend sub-package cold-starts ``triton.backends`` directly,
# which pulls in ``triton.backends.amd`` -> ``triton.runtime`` -> back into the
# still-partially-initialized ``triton.backends`` (circular ImportError);
# ``import triton`` initializes in the safe order so the imports below resolve
# from an already-loaded package.  This is the same reason run_host_tests.py
# puts ``triton/python`` on PYTHONPATH.
import triton  # noqa: F401

# A tt.sharedir-level module carrying a Triton scan.  OPNAME becomes either the
# generic-form op name (quoted) or its terminator (bare), so one template
# exercises both spellings.  The leading comment mentions tt.scan on purpose:
# it is the case the old regex clobbered and the rewrite must keep.
_SCAN_TEMPLATE = """// a comment mentioning tt.scan must survive the rewrite
module {
  func.func @scan_kernel(%x: tensor<256xi32>) -> tensor<256xi32> {
    %result = "OPNAME"(%x) <{axis = 0 : i32, reverse = false}> ({
      ^bb0(%a: i32, %b: i32):
        %r = arith.addi %a, %b : i32
        OPNAME.return %r : i32
      }) : (tensor<256xi32>) -> tensor<256xi32>
    return %result : tensor<256xi32>
  }
}
"""


def _scan_module(op: str) -> str:
    return _SCAN_TEMPLATE.replace("OPNAME", op)


def _digest(result) -> str:
    raw = result if isinstance(result, bytes) else result.encode()
    return hashlib.sha256(raw).hexdigest()


def _run(fn, src: str):
    """Drive a real compile entry point, host-only, swallowing backend chatter."""
    from triton.backends.qcom_hexagon_backend.hexagon_options import (  # noqa: PLC0415
        HexagonOptions,
    )

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return fn(src, HexagonOptions(), {})


class BothEntriesAgreeTest(unittest.TestCase):
    """The divergence regression: one input, both entries, same accept/reject."""

    def test_both_entries_accept_the_same_tt_scan_input(self):
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            ttsharedir_to_llir,
            ttsharedir_to_obj,
        )

        src = _scan_module("tt.scan")
        # Pre-fix this raised "Failed to convert Triton Linalg to LLVM MLIR"
        # while ttsharedir_to_obj succeeded -- the divergence under repair.
        llir = _run(ttsharedir_to_llir, src)
        obj = _run(ttsharedir_to_obj, src)

        self.assertTrue(llir)
        self.assertGreater(len(obj), 0)

    def test_llir_agrees_with_the_already_lowered_ttx_form(self):
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            ttsharedir_to_llir,
        )

        # The rewrite must yield exactly the ttx.scan form, so compiling the
        # tt.scan input produces byte-identical LLVM IR to its ttx.scan twin.
        self.assertEqual(
            _digest(_run(ttsharedir_to_llir, _scan_module("tt.scan"))),
            _digest(_run(ttsharedir_to_llir, _scan_module("ttx.scan"))),
        )

    def test_the_object_artifact_is_unchanged_by_the_rewrite(self):
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            ttsharedir_to_obj,
        )

        # tt.scan must compile to the same object as its ttx.scan twin: the
        # rewrite renames the dialect, it does not change what is compiled.
        self.assertEqual(
            _digest(_run(ttsharedir_to_obj, _scan_module("tt.scan"))),
            _digest(_run(ttsharedir_to_obj, _scan_module("ttx.scan"))),
        )


class RewriteScopeTest(unittest.TestCase):
    """What the rewrite touches, and what it must leave alone (text level)."""

    def test_comments_and_data_strings_are_not_rewritten(self):
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            _apply_ttx_scan_rewrite,
        )

        text = (
            "// keep tt.scan in this comment\n"
            'module attributes {info = "see tt.scan here"} {\n'
            '  %0 = "tt.scan"(%x) : () -> ()\n'
            "  tt.scan.return %r : i32\n"
            "}\n"
        )
        out = _apply_ttx_scan_rewrite(text)

        # Comment untouched: the old regex rewrote it to ttx.scan.
        self.assertIn("// keep tt.scan in this comment", out)
        # Data string untouched: only a whole-content op name may move.
        self.assertIn('"see tt.scan here"', out)
        # Genuine op references moved: generic-form op name and bare terminator.
        self.assertIn('"ttx.scan"(%x)', out)
        self.assertIn("ttx.scan.return %r", out)
        self.assertNotIn("tt.scan.return", out)

    def test_rewriting_is_idempotent(self):
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            _apply_ttx_scan_rewrite,
        )

        once = _apply_ttx_scan_rewrite(_scan_module("tt.scan"))
        self.assertEqual(once, _apply_ttx_scan_rewrite(once))
        # An already-ttx module is a no-op.
        self.assertEqual(
            _apply_ttx_scan_rewrite(_scan_module("ttx.scan")),
            _scan_module("ttx.scan"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
