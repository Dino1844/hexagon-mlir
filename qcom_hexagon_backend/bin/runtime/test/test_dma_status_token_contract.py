#!/usr/bin/env python3
"""Source contract: a refused DMA start reports through status, not a forgery.

Why this file exists
--------------------
The runtime's device gtest suite is OFF by default (community SDK ships no
googletest -- ``test_runtime_test_suite_contract.py`` pins that), and this
workspace never runs on a device for a ticket gate. So the refusal path of
``UserDMA::copy`` / ``copy2D`` -- the path that must now report truthfully --
would otherwise have *zero* executable coverage.

This is the behaviour it pins, after the status/token split of
``roadmap/ARCH-REVIEW.md`` bug #2:

* A refusal writes ``*status = DMAFailure`` and returns ``DMA_TOKEN_NONE``.
* No descriptor is forged to make ``wait()`` return (``enqueueRejectedDesc``
  and its ``DESC_DONE_COMPLETE`` lie are gone); ``wait(DMA_TOKEN_NONE)``
  short-circuits instead.
* The caller's status word is documented as storage of its own -- never the
  tag that later holds the returned token.
* The enqueue lock from ticket 03 stays in place around every enqueue.

The compiler half of the same contract (the call's status argument is a
dedicated slot and the token is stored to the tag only) is pinned where it
belongs, against real emitted IR, by
``test/Conversion/DMAToLLVM/dma_status_vs_token.mlir`` -- this file pins the
runtime half only.
"""
from pathlib import Path
import re
import unittest

RUNTIME = Path(__file__).resolve().parents[1]
USER_DMA = RUNTIME / "UserDMA/UserDMA.cc"
USER_DMA_H = RUNTIME / "UserDMA/UserDMA.h"
PUBLIC_API = RUNTIME / "include/RuntimeDMA.h"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def function_body(source: str, signature: str, next_signature: str) -> str:
    """Slice one member function's body; scoping beats whole-file greps."""
    start = source.find(signature)
    assert start > 0, f"{signature} not found in UserDMA.cc (renamed?)"
    end = source.find(next_signature, start + len(signature))
    assert end > start, f"{next_signature} not found after {signature}"
    return source[start:end]


def copy_body(source: str) -> str:
    return function_body(
        source, "uint32_t UserDMA::copy(", "uint32_t UserDMA::copy2D("
    )


def returns_of(body: str) -> list[str]:
    """Statement-level `return` values (line-anchored, so prose comments
    mentioning `return` cannot smuggle a value into the scan)."""
    return re.findall(r"^[ \t]*return\s+([^;]+);", body, re.M)


def copy2d_body(source: str) -> str:
    return function_body(
        source, "uint32_t UserDMA::copy2D(", "void UserDMA::wait("
    )


def wait_body(source: str) -> str:
    start = source.find("void UserDMA::wait(")
    assert start > 0, "UserDMA::wait not found"
    end = source.find("\nUserDMA::UserDMA(", start)
    return source[start:end if end > 0 else len(source)]


class TheSentinelIsDefinedOnce(unittest.TestCase):
    def test_the_public_header_defines_dma_token_none_as_all_ones(self):
        api = read(PUBLIC_API)
        match = re.search(
            r"static const uint32_t DMA_TOKEN_NONE = ([^;]+);", api
        )
        self.assertIsNotNone(match, "DMA_TOKEN_NONE missing from RuntimeDMA.h")
        # The value is written as a cast, not a literal, so widening
        # uint32_t would be a visible edit rather than a silent one.
        self.assertEqual(match.group(1).strip(), "static_cast<uint32_t>(-1)")

    def test_both_start_entries_are_declared_against_one_status_contract(self):
        # The 2D entry used to exist only as a definition in RuntimeDMA.cc,
        # so the header could not document (or drift-check) its status
        # parameter. Both start signatures now sit under the same
        # status/token ABI note.
        api = read(PUBLIC_API)
        self.assertIn("hexagon_runtime_dma_start", api)
        self.assertIn("hexagon_runtime_dma2d_start", api)
        self.assertIn("must not be memory that later holds", api)

    def test_the_header_of_userdma_documents_the_split_for_each_call(self):
        header = read(USER_DMA_H)
        for decl in ("uint32_t copy(", "uint32_t copy2D("):
            with self.subTest(decl=decl):
                idx = header.find(decl)
                self.assertGreater(idx, 0, f"{decl} declaration not found")
                # The doc block sits above the declaration and the signature
                # carries the status parameter: span both.
                window = header[max(0, idx - 700) : idx + 400]
                self.assertIn("DMA_TOKEN_NONE", window)
                self.assertIn("status", window.lower())


class RefusalReportsInsteadOfForging(unittest.TestCase):
    """Every early return is a truthful report; no descriptor is invented."""

    def setUp(self):
        self.source = read(USER_DMA)

    def test_the_forged_descriptor_helper_is_gone(self):
        self.assertNotIn("enqueueRejectedDesc", self.source)

    def test_no_refusal_early_return_smuggles_a_bare_zero_token(self):
        # `return 0` was the 1D refusal: wait(0) then polls ring slot 0,
        # an unrelated (or never-used, and therefore spinning) descriptor.
        for body in (copy_body(self.source), copy2d_body(self.source)):
            with self.subTest(fn=body[:30]):
                returns = returns_of(body)
                self.assertTrue(returns, "no returns found -- wrong slice?")
                for value in returns:
                    self.assertIn(
                        value.strip(),
                        {"DMA_TOKEN_NONE", "token"},
                        "a return that is neither the refused-start sentinel "
                        "nor the enqueued token slipped in",
                    )
                self.assertNotIn("return 0;", body)

    def test_each_function_reports_failure_through_its_status_pointer(self):
        for body in (copy_body(self.source), copy2d_body(self.source)):
            with self.subTest(fn=body[:30]):
                self.assertGreaterEqual(
                    body.count("*status = DMAFailure"), 1,
                    "a refusal path no longer writes the status word",
                )
                self.assertIn("*status = DMASuccess", body)

    def test_no_refusal_mark_a_descriptor_done_to_fake_completion(self):
        # DESC_DONE_COMPLETE inside an enqueue body is the forgery's
        # fingerprint; reading it in inFlight() is legitimate and lives
        # outside both bodies.
        for body in (copy_body(self.source), copy2d_body(self.source)):
            with self.subTest(fn=body[:30]):
                self.assertNotIn("DESC_DONE_COMPLETE", body)

    def test_wait_short_circuits_on_the_sentinel_before_touching_the_ring(self):
        body = wait_body(self.source)
        guard = body.find("token == DMA_TOKEN_NONE")
        ring = body.find("getDataPtr")
        self.assertGreater(guard, 0, "wait() no longer recognises the sentinel")
        self.assertGreater(ring, 0, "wait() no longer polls the ring at all")
        self.assertLess(
            guard, ring,
            "the sentinel guard must run before getDataPtr: token 0xFFFFFFFF "
            "would index the last ring slot and wait on an unrelated transfer",
        )


class Ticket03LocksSurviveTheRework(unittest.TestCase):
    """The status/token split must not quietly undo the enqueue serialization."""

    def test_both_enqueue_paths_still_hold_the_enqueue_lock(self):
        source = read(USER_DMA)
        lock = "std::lock_guard<std::mutex> lock(enqueueMutex_);"
        for body in (copy_body(source), copy2d_body(source)):
            with self.subTest(fn=body[:30]):
                self.assertIn(lock, body)


if __name__ == "__main__":
    unittest.main()
