#!/usr/bin/env python3
"""Source contract: every test source is either built or has a written reason.

Two silences this closes
------------------------
**A dropped ``.cpp`` loses its tests with no signal.** ``test/CMakeLists.txt``
spells out its source list literally. Removing a line from it does not fail any
build and does not fail any test: the file is still sitting in the directory, so
``ls test/`` still shows it, but its ``TEST``/``TEST_F`` bodies are never
compiled and never run. Nothing else in the tree reads that list, so the loss is
silent in both directions.

**The suite is off by default, and nothing says so where it matters.**
``HEXAGON_MLIR_BUILD_RUNTIME_TESTS`` defaults to ``OFF`` because the community
SDK ships no googletest (its own CMake comment says so, and ``AGENTS.md`` §5
agrees). That is a legitimate environment constraint, not a defect. But nothing
in the tree records the consequence: the device gtest suite reads as covered
while never having been compiled. So the OFF default is pinned here *together
with* the reason, which means flipping the default has to update this file too.

What this does not do
---------------------
It does not try to make the device gtests run on a host that cannot build them.
It makes the boundary explicit: this is the accounted-for set, and anything that
moves across it is a deliberate edit to both files.
"""
from pathlib import Path
import re
import unittest

RUNTIME = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
TEST_CMAKE = TEST_DIR / "CMakeLists.txt"
RUNTIME_CMAKE = RUNTIME / "CMakeLists.txt"

# Sources in this directory that are deliberately not in the device gtest list,
# each with the reason it is allowed to stay out. Adding an entry here is a
# decision to lose device-side coverage for those tests, so the reason is
# mandatory and is the thing a reviewer argues with.
EXEMPT = {
    "DMA2DDescLayoutTest.cpp": (
        "standalone host main() with its own build recipe; it trips undefined "
        "symbols in run_main_on_hexagon, so adding it would break the device "
        "suite. Its reject contract is covered automatically by "
        "test_userdma_2d_geometry_contract.py."
    ),
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def listed_sources() -> set[str]:
    """The basenames the device gtest library is actually built from."""
    cmake = read(TEST_CMAKE)
    block = re.search(
        r"set\(HEXAGON_RUNTIME_TEST_SRCS(.*?)\n\)", cmake, re.S
    )
    assert block, "HEXAGON_RUNTIME_TEST_SRCS not found in test/CMakeLists.txt"
    return set(re.findall(r"test/([\w.]+\.cpp)", block.group(1)))


class SourceListIsComplete(unittest.TestCase):
    def test_every_source_in_this_directory_is_accounted_for(self):
        on_disk = {p.name for p in TEST_DIR.glob("*.cpp")}
        accounted = listed_sources() | set(EXEMPT)
        self.assertEqual(
            on_disk - accounted,
            set(),
            "a .cpp in bin/runtime/test/ is neither in the device gtest source "
            "list nor in EXEMPT, so its tests would silently stop running",
        )

    def test_the_exemption_list_names_no_file_that_is_gone(self):
        on_disk = {p.name for p in TEST_DIR.glob("*.cpp")}
        self.assertEqual(
            set(EXEMPT) - on_disk,
            set(),
            "EXEMPT lists a file that no longer exists; remove the stale entry",
        )

    def test_every_exemption_states_a_reason(self):
        for name, reason in EXEMPT.items():
            with self.subTest(name=name):
                self.assertTrue(
                    reason.strip() and len(reason) > 40,
                    "an exemption without a real reason is just a silent exclusion",
                )

    def test_nothing_exempt_is_also_being_built(self):
        # An entry in both lists would mean the exemption reason is stale but the
        # file is in fact covered -- the comment would then understate reality.
        self.assertEqual(listed_sources() & set(EXEMPT), set())

    def test_the_built_sources_are_not_empty(self):
        # Guards against the list being emptied wholesale, which would also make
        # the completeness test above pass vacuously.
        self.assertGreaterEqual(len(listed_sources()), 8)


class OffByDefaultIsAccountedFor(unittest.TestCase):
    def test_the_runtime_test_option_exists_and_defaults_off(self):
        cmake = read(RUNTIME_CMAKE)
        match = re.search(
            r'option\(HEXAGON_MLIR_BUILD_RUNTIME_TESTS\s+"[^"]*"\s+(ON|OFF)\)',
            cmake,
        )
        assert match, "HEXAGON_MLIR_BUILD_RUNTIME_TESTS option not found"
        self.assertEqual(
            match.group(1),
            "OFF",
            "the device gtest default was flipped to ON; this file's OFF "
            "accounting is now wrong and needs revisiting",
        )

    def test_the_reason_for_the_default_is_stated_where_the_default_lives(self):
        # A bare `option(... OFF)` reads like an oversight. The comment above it
        # is what makes it a decision, so the comment is part of the contract.
        cmake = read(RUNTIME_CMAKE)
        idx = cmake.find("option(HEXAGON_MLIR_BUILD_RUNTIME_TESTS")
        assert idx > 0
        window = cmake[max(0, idx - 600) : idx]
        self.assertIn("googletest", window.lower())

    def test_nothing_in_the_tree_silently_turns_it_on(self):
        # If some script started passing -DHEXAGON_MLIR_BUILD_RUNTIME_TESTS=ON,
        # the OFF accounting above would no longer describe reality.
        root = RUNTIME.parents[2]  # hexagon-mlir/
        offenders = []
        for pattern in ("*.sh", "*.cmake", "CMakeLists.txt"):
            for path in root.rglob(pattern):
                if "build/" in str(path) or "/triton/" in str(path):
                    continue
                if re.search(
                    r"-DHEXAGON_MLIR_BUILD_RUNTIME_TESTS=ON", read(path)
                ):
                    offenders.append(str(path))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()