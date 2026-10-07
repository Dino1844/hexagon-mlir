#!/usr/bin/env python3
"""Source contract: a 2D DMA descriptor that does not fit must be refused.

Why this file exists
--------------------
``dma2DGeometryFits`` is the only guard between a caller's geometry and the
descriptor setters, and the setters only *mask*.  Its own comment says so:

    // the setters only mask, so an out-of-range value would be truncated
    // silently.

The only thing that ever exercised it was ``test/DMA2DDescLayoutTest.cpp``,
which is a standalone host ``main()`` that **no runner compiles**: it is absent
from ``test/CMakeLists.txt`` (whose comment explains why -- it trips undefined
symbols in ``run_main_on_hexagon``), and nothing else in the tree builds it.  So
the guard against silent descriptor truncation had *zero* live coverage, while
the file sat in the test directory looking like coverage.

It also mis-attributed its own oracle.  Its header claimed the expected words
came from ``exp/hmx/dma2d_probe/wrapper_dma2d.cpp`` and were confirmed on v79 in
``logs/dma2d_probe/REPORT.txt``; neither path exists in the workspace.  Per the
evidence discipline in ``AGENTS.md`` §7.9, a claim pointing at a missing artefact
is worse than no claim, because it reads as verified.

What this file does instead
---------------------------
It pins the same facts against the artefact that *is* the source of truth --
``UserDMA/UserDMADescriptors.h`` -- using literal comparisons, so a rename or a
coordinated wrong value fails loudly rather than passing by re-derivation.  The
numbers are the ones the dead C++ test pinned, cross-checked against the header
at the bottom of this file, so the two cannot drift apart unnoticed.

Scope: this covers the *reject* contract and the bit-field placement.  The
end-to-end descriptor words stay in the C++ file, which is still the right home
for them; that file is unbuilt and now says so.
"""
from pathlib import Path
import re
import unittest

RUNTIME = Path(__file__).resolve().parents[1]
DESCRIPTORS = RUNTIME / "UserDMA/UserDMADescriptors.h"
USER_DMA = RUNTIME / "UserDMA/UserDMA.cc"
DEAD_TEST = Path(__file__).resolve().parent / "DMA2DDescLayoutTest.cpp"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def define(source: str, name: str) -> int:
    match = re.search(rf"^#define\s+{name}\s+([0-9A-Fa-fx]+)u?\s*$", source, re.M)
    assert match, f"#define {name} not found in {DESCRIPTORS.name} (renamed?)"
    return int(match.group(1), 0)


class BoundsAreTheDocumentedBitWidths(unittest.TestCase):
    """The accept/reject boundary is the descriptor field width, not a tuning knob."""

    # Literals, not expressions derived from the header: these are the 24-bit
    # descriptor ABI widths. Changing one here is a deliberate ABI change.
    EXPECTED = {
        "DESC_ROWSIZE_MAX": 0x00FFFFFF,
        "DESC_NROWS_MAX": 0x0000FFFF,
        "DESC_STRIDE_MAX": 0x00FFFFFF,
    }

    def test_each_bound_equals_its_field_width(self):
        source = read(DESCRIPTORS)
        for name, want in self.EXPECTED.items():
            with self.subTest(name=name):
                self.assertEqual(define(source, name), want)

    def test_the_row_and_src_stride_widths_are_the_same_number(self):
        # Not a tautology: it records that row_size and src_stride are both
        # 24-bit while nrows is 16-bit split across two bytes. A future edit that
        # widened only one of the two 24-bit fields would break the descriptor
        # layout that DMA2DDescLayoutTest.cpp's words encode.
        source = read(DESCRIPTORS)
        self.assertEqual(define(source, "DESC_ROWSIZE_MAX"), define(source, "DESC_STRIDE_MAX"))
        self.assertNotEqual(define(source, "DESC_NROWS_MAX"), define(source, "DESC_ROWSIZE_MAX"))


class FitsIsAConjunctionOverExactlyThoseBounds(unittest.TestCase):
    """The guard must test all four inputs; dropping one re-opens silent truncation."""

    def _body(self) -> str:
        source = read(DESCRIPTORS)
        match = re.search(
            r"inline bool dma2DGeometryFits\([^)]*\)\s*\{(.*?)\n\}", source, re.S
        )
        assert match, "dma2DGeometryFits not found in UserDMADescriptors.h"
        return match.group(1)

    def test_all_four_inputs_are_bounded_by_their_own_max(self):
        body = self._body()
        for name in ("width", "height", "srcStride", "dstStride"):
            with self.subTest(param=name):
                bound = {
                    "width": "DESC_ROWSIZE_MAX",
                    "height": "DESC_NROWS_MAX",
                    "srcStride": "DESC_STRIDE_MAX",
                    "dstStride": "DESC_STRIDE_MAX",
                }[name]
                self.assertRegex(body, rf"{name}\s*<=\s*{bound}")

    def test_the_result_is_an_and_of_exactly_four_comparisons(self):
        # Counting the conjuncts catches both a dropped bound and a bound that
        # was added as a tautology (`x <= DESC_ROWSIZE_MAX || x <= ...`).
        body = self._body()
        conjuncts = [c for c in body.split("&&") if "<=" in c]
        self.assertEqual(len(conjuncts), 4, body)

    def test_no_bound_was_written_as_a_disjunction(self):
        self.assertNotIn("||", self._body())


class CallersCheckBeforeProgramming(unittest.TestCase):
    """The reject only helps if it runs before the masking setters.

    This is the ordering that makes the guard meaningful: `UserDMA.cc` copies a
    setter sequence from the test, and if the check ever moved after the first
    setter, an out-of-range value would already be truncated.
    """

    def test_the_caller_checks_before_the_first_setter(self):
        source = read(USER_DMA)
        # Scoped to copy2D's own body: dmaDescSet* appears in the 1D path earlier
        # in the file, and a whole-file `find` would compare against the wrong
        # function and make this assertion vacuously true.
        start = source.find("uint32_t UserDMA::copy2D(")
        assert start > 0, "UserDMA::copy2D not found"
        end = source.find("\nuint32_t UserDMA::", start + 1)
        body = source[start:end if end > 0 else len(source)]
        check = body.find("dma2DGeometryFits(")
        assert check > 0, "copy2D no longer calls dma2DGeometryFits"
        first_setter = body.find("dmaDescSet")
        assert first_setter > 0, "copy2D no longer programs a descriptor"
        self.assertLess(
            check,
            first_setter,
            "dma2DGeometryFits must be called before any dmaDescSet* setter, "
            "otherwise the out-of-range value is truncated before it is noticed",
        )


class BitFieldPlacementTest(unittest.TestCase):
    """Where each field lives in the eight words, as literals.

    These are the shifts and masks that make the words in
    DMA2DDescLayoutTest.cpp come out the way they do; that file pins the
    end-to-end bytes, this pins the layout that produces them.
    """

    EXPECTED = {
        "DESC_TYPE_MASK": 0x000000FF,
        "DESC_TYPE_SHIFT": 0,
        "DESC_TYPE_2D_24BIT": 9,
        "DESC_ROWSIZE_MASK": 0x00FFFFFF,
        "DESC_ROWSIZE_SHIFT": 0,
        "DESC_NROWSLO_MASK": 0xFF000000,
        "DESC_NROWSLO_SHIFT": 24,
        "DESC_NROWSHI_MASK": 0x000000FF,
        "DESC_NROWSHI_SHIFT": 0,
    }

    def test_field_placement_is_unchanged(self):
        source = read(DESCRIPTORS)
        for name, want in self.EXPECTED.items():
            with self.subTest(name=name):
                self.assertEqual(define(source, name), want)

    def test_nrows_low_byte_shares_a_word_with_row_size(self):
        # nrows_lo occupies bits 24..31 and row_size bits 0..23, i.e. they are
        # the two halves of descriptor word 5. This is the non-obvious part of
        # the layout and the reason word5 in the C++ test reads 0x02000040.
        source = read(DESCRIPTORS)
        rowsize = define(source, "DESC_ROWSIZE_MASK")
        nrowslo = define(source, "DESC_NROWSLO_MASK")
        self.assertEqual(rowsize, 0x00FFFFFF)
        self.assertEqual(nrowslo, 0xFF000000)
        self.assertEqual(rowsize & nrowslo, 0, "the two halves must not overlap")

    def test_src_stride_starts_at_bit_eight_of_the_next_word(self):
        # src stride is stored as (v << 8) into nrowsHiSrcStride, i.e. descriptor
        # word 6 bits 8..31, above the 8-bit nrows_hi. Asserted against the
        # header's own constants rather than by re-deriving from the setter, so
        # a shift change and a mask change cannot both slip through.
        source = read(DESCRIPTORS)
        self.assertEqual(define(source, "DESC_SRCSTRIDE_SHIFT"), 8)
        self.assertEqual(define(source, "DESC_SRCSTRIDE_MASK"), 0xFFFFFF00)
        # ... and it must not overlap nrows_hi, which is the other half.
        self.assertEqual(define(source, "DESC_NROWSHI_MASK"), 0x000000FF)
        self.assertEqual(
            define(source, "DESC_SRCSTRIDE_MASK") & define(source, "DESC_NROWSHI_MASK"),
            0,
            "src_stride and nrows_hi share word 6 and must not overlap",
        )
        # dst_stride, by contrast, starts at bit 0 of word 1.
        self.assertEqual(define(source, "DESC_DSTSTRIDE_SHIFT"), 0)
        self.assertEqual(define(source, "DESC_DSTSTRIDE_MASK"), 0x00FFFFFF)


class TheUnbuiltTestAgreesWithTheHeader(unittest.TestCase):
    """Cross-check, so the dead C++ test cannot rot against a header edit.

    DMA2DDescLayoutTest.cpp is not compiled (see the module docstring). It is
    kept because it is the only end-to-end pin of the descriptor words, and
    AGENTS.md §7.1 asks for a deletion to be proposed rather than for coverage
    to disappear quietly. What is *not* acceptable is for it to rot unnoticed,
    so its literal words are checked against the header here.
    """

    def test_the_max_words_still_follow_from_the_field_widths(self):
        dead = read(DEAD_TEST)
        source = read(DESCRIPTORS)
        desc_type = define(source, "DESC_TYPE_2D_24BIT")

        def word(n: int, value: int) -> str:
            return f'checkWord("max word{n}", desc[{n}], 0x{value:08X}u)'

        # word4 is the descriptor type, a single byte.
        self.assertIn(f'checkWord("geom word4", desc[4], 0x{desc_type:08X}u)', dead)
        # word5 packs row_size (bits 0..23) with nrows_lo (bits 24..31) and word6
        # packs nrows_hi (bits 0..7) with src_stride (bits 8..31). At the maximum
        # geometry both halves of each word saturate, which is why -- and only
        # why -- those two expected words are all ones.
        for n in (5, 6):
            with self.subTest(word=n):
                self.assertIn(word(n, 0xFFFFFFFF), dead)
        # word1 is dst_stride with desc_size in the top byte, and desc_size is 1
        # for a 2D descriptor, so this word is *not* all ones.
        self.assertIn(word(1, 0x01FFFFFF), dead)
        self.assertNotIn(word(1, 0xFFFFFFFF), dead)

    def test_it_now_names_an_oracle_that_exists(self):
        # The correction paragraph has to *name* the old missing paths in order to
        # explain itself, so the assertion is not "those strings are absent" but
        # "the file points at an artefact that is actually there".
        dead = read(DEAD_TEST)
        self.assertIn("UserDMA/UserDMADescriptors.h", dead)
        self.assertIn("test_userdma_2d_geometry_contract.py", dead)


if __name__ == "__main__":
    unittest.main()