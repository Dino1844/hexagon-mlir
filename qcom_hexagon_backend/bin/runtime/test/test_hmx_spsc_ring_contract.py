#!/usr/bin/env python3
"""Host-runnable lock on the SPSC tile-group ring and the HMX role executor.

Why this file exists
--------------------
``bin/runtime/multithreading/HmxRoleExecutor.cpp`` is the S2 runtime floor for
thread-role execution (roadmap/ROADMAP1001.md §4): a resident bound thread
(T_HMX) that takes the HMX lock once and holds it for life, fed by the SPSC
ring in ``include/HmxSpscRing.h``. Until now nothing about it could run
without a device, and the two existing precedents each cover only half the
risk: the vector executor (HmxVectorExecutor.cpp) has no host test at all,
and the layout permutation test locks leaves but not threads.

How it works
------------
This compiles the **real HmxRoleExecutor.cpp, unmodified**, on the host, the
same way ``test_hmx_layout_permutation_contract.py`` compiles the real
HMXLayout.c:

  1. ``hmx_role_host_emu/`` holds test-only stubs of ``<qurt_thread.h>``,
     ``<qurt_futex.h>`` and ``<HAP_farf.h>`` whose emulations transcribe the
     documented QuRT/HAP contracts (each deviation is listed in the stub).
     The stub futex is a real mutex+condvar with a generation counter, so
     parks and wakes actually sleep and wake -- the park paths are exercised,
     not skipped.
  2. ``hmx_role_host_emu/driver.cc`` includes the production file verbatim
     and speaks a one-reply-per-command stdin/stdout protocol. It contains
     no executor logic; what it does own is the *ownership shadow*: a
     driver-owned stand-in for the compiler-allocated VTCM slots, with
     guards that record who wrote what when, so ownership violations are
     DETECTED, not assumed away.
  3. This file derives every expectation independently (submit order, poked
     values, marker values) and compares field by field.

What it locks
-------------
* The ring protocol, deterministically (no threads): empty/full boundaries at
  capacity 1, fill-to-full and slot reuse only after retire, wraparound at
  non-power-of-two capacities, the monotone drain barrier, and the 2^32
  counter wrap (via a preset just below the boundary).
* The descriptor ABI end to end: slot / rowStart / rowCount (G) / the 64-bit
  event word survive publish->peek and submit->section bit-exact, and the
  batch G travels with the descriptor (one section call per group, never
  coalesced or split).
* The executor lifecycle: bind refusals (null section, depth 0), submit
  before bind and after join refused, full-ring short return with the tail
  recoverable by the caller, drain as a real barrier (marker values visible
  in the shadow after drain), the park paths actually parking (futex
  counters), and rebind WITHOUT thread restart -- createdThreads and
  ensureCalls stay 1 across grow and shrink rebinds, which is the property
  the lifetime lock depends on (a restarted thread would have to re-ensure
  and block on its own predecessor's held lock).
* The lock migration's first half: the ensure entry is called exactly once,
  on the thread that was created, before any section ran (sequence numbers).
* Ownership transfer: a producer write to an in-flight slot is refused; a
  section writing a slot other than its descriptor's is recorded; and two
  committed negative controls (``poke-force`` bypassing the producer guard,
  the ``rogue`` section writing a foreign slot) prove both detectors fire
  rather than pass vacuously.

What it does NOT lock (residual risk, stated honestly)
------------------------------------------------------
* **The Hexagon memory model.** The protocol runs under the HOST memory
  model, which is stronger. A green battery does not clear the device; the
  device-side validation is the fence arm of
  ``exp/hmx/s2_runtime_probes/ring_probe`` (ready-to-run scaffold, to be
  signed in with the S3 device window). This limit is also stated in
  HmxSpscRing.h's fence-policy block.
* **The HMX lock's blocking semantics.** The stub ensure records and
  returns; it cannot model "a second ensure from another thread blocks
  forever" (the hazard documented in HmxRoleExecutor.h). The single-holder
  property is the lock-lifetime device probe's job.
* **Thread priority and the malloc'd DDR stack** (see the qurt_thread.h
  stub), and the 2^32 counter wrap at production speed (only the preset
  crossing is covered).
* **bind-after-join.** The lock is never released (the recorded decision),
  so whether a post-join re-ensure can succeed is a device property; the
  battery deliberately does not run that sequence.

Run: python test_hmx_spsc_ring_contract.py
     python -m pytest -q test_hmx_spsc_ring_contract.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
EMU_DIR = TEST_DIR / "hmx_role_host_emu"
DRIVER_CC = EMU_DIR / "driver.cc"
RUNTIME = TEST_DIR.parent  # bin/runtime
RUNTIME_INCLUDE = RUNTIME / "include"
ROLE_EXECUTOR_H = RUNTIME_INCLUDE / "HmxRoleExecutor.h"
SPSC_RING_H = RUNTIME_INCLUDE / "HmxSpscRing.h"
ROLE_EXECUTOR_CPP = RUNTIME / "multithreading" / "HmxRoleExecutor.cpp"


# ---------------------------------------------------------------------------
# Session harness: compile the driver once per configuration, run interactive
# sessions with paced send/receive (the park assertions need real sleeps
# between commands, which a batch pipe cannot give).
# ---------------------------------------------------------------------------


class Session:
    """One driver process. One reply line per command, with a watchdog.

    A wedged protocol would otherwise hang the suite forever (the production
    code under test is a thread park), so every reply read is bounded and a
    timeout kills the process and names the command that never answered.
    """

    def __init__(self, driver: Path):
        self.proc = subprocess.Popen(
            [str(driver)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._stderr: list[str] = []
        banner = self._read_reply("banner")
        if not banner.startswith("role-driver ready"):
            raise AssertionError(f"bad banner: {banner!r}")

    def _read_reply(self, what: str) -> str:
        import selectors

        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            events = sel.select(timeout=90)
            if not events:
                self.kill()
                raise AssertionError(
                    f"driver did not answer {what!r} within 90s "
                    "(wedged protocol or deadlocked production code)"
                )
        finally:
            sel.close()
        line = self.proc.stdout.readline()
        if not line:
            self.kill()
            raise AssertionError(f"driver closed stdout while awaiting {what!r}")
        return line.rstrip("\n")

    def cmd(self, line: str) -> str:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        reply = self._read_reply(line)
        if reply.startswith("ERR"):
            self.kill()
            raise AssertionError(f"driver protocol error for {line!r}: {reply}")
        return reply

    def close(self) -> str:
        """Terminate (quit) and return the accumulated stderr text."""
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.kill()
        assert self.proc.stderr is not None
        err = self.proc.stderr.read()
        return err or ""

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()


class Harness:
    """Compiles the driver (once per configuration) and runs sessions."""

    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hmx_role_emu_"))
        self._cache: dict = {}

    def close(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def compilers() -> list:
        cands = [os.environ.get("CXX"), "c++", "g++", "clang++"]
        seen, out = set(), []
        for c in cands:
            path = shutil.which(c) if c else None
            if path and path not in seen:
                seen.add(path)
                out.append(path)
        assert out, (
            "no host C++ compiler found (tried $CXX, c++, g++, clang++); this "
            "test compiles the real HmxRoleExecutor.cpp, so it needs one"
        )
        return out

    def compile(self, cxx: str | None = None, opt: str = "-O2") -> Path:
        key = (cxx, opt)
        if key in self._cache:
            return self._cache[key]
        cxx = cxx or self.compilers()[0]
        tag = f"driver_{len(self._cache)}"
        out = self.tmp / tag
        cmd = [
            cxx, "-std=c++17", opt, "-Wall", "-Wextra", "-Wno-unused-parameter",
            f"-I{EMU_DIR}", f"-I{RUNTIME_INCLUDE}",
            str(DRIVER_CC), "-o", str(out), "-pthread",
        ]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if done.returncode != 0:
            raise AssertionError(
                f"driver compile failed ({' '.join(cmd)}):\n{done.stderr}"
            )
        self._cache[key] = out
        return out


# ---------------------------------------------------------------------------
# Reply parsers
# ---------------------------------------------------------------------------


def parse_kv(reply: str) -> dict:
    """PROBE k=v ..."""
    head, *parts = reply.split()
    assert head == "PROBE", reply
    out = {}
    for part in parts:
        k, _, v = part.partition("=")
        out[k] = v
    return out


def parse_fnlog(reply: str) -> list:
    """FNLOG <n>|seq,thread,slot,rowStart,rowCount,eventLo,eventHi,read,wrote;..."""
    assert reply.startswith("FNLOG "), reply
    body = reply[len("FNLOG "):]
    count_s, _, payload = body.partition("|")
    assert int(count_s) == (len(payload.split(";")) if payload else 0), reply
    entries = []
    if not payload:
        return entries
    for item in payload.split(";"):
        f = item.split(",")
        assert len(f) == 9, item
        entries.append({
            "seq": int(f[0]), "thread": f[1], "slot": int(f[2]),
            "rowStart": int(f[3]), "rowCount": int(f[4]),
            "eventLo": int(f[5]), "eventHi": int(f[6]),
            "event": (int(f[6]) << 32) | int(f[5]),
            "read": int(f[7]), "wrote": int(f[8]),
        })
    return entries


def parse_violations(reply: str) -> list:
    assert reply.startswith("VIOLATIONS "), reply
    body = reply[len("VIOLATIONS "):]
    count_s, _, payload = body.partition("|")
    assert int(count_s) == (len(payload.split(";")) if payload else 0), reply
    if not payload:
        return []
    out = []
    for item in payload.split(";"):
        kind, slot, owner = item.split(",")
        out.append({"kind": kind, "slot": int(slot), "owner": int(owner)})
    return out


def submit_args(groups: list) -> str:
    """groups: list of (slot, rowStart, rowCount, event) tuples -> one line."""
    parts = [str(len(groups))]
    for slot, row_start, row_count, event in groups:
        parts += [
            str(slot), str(row_start), str(row_count),
            str(event & 0xFFFFFFFF), str(event >> 32),
        ]
    return "submit " + " ".join(parts)


# ---------------------------------------------------------------------------
# Test batteries
# ---------------------------------------------------------------------------


class RingProtocolTests(unittest.TestCase):
    """The ring protocol, deterministically, with no threads in the way.

    The driver keeps a local HmxSpscRing; these commands drive both sides of
    it (publish from the producer role, peek/retire from the consumer role)
    in a fixed order, so every boundary below is exact.
    """

    @classmethod
    def setUpClass(cls):
        cls.harness = Harness()
        cls.driver = cls.harness.compile()

    @classmethod
    def tearDownClass(cls):
        cls.harness.close()

    def session(self) -> Session:
        return Session(self.driver)

    def test_capacity_one_boundaries(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("ring 1"), "RING ok")
            self.assertEqual(s.cmd("rfree"), "RFREE 1")
            self.assertEqual(s.cmd("rpend"), "RPEND 0")
            self.assertEqual(s.cmd("rpush 3 0 4 11 22"), "RPUSH 1")
            self.assertEqual(s.cmd("rpend"), "RPEND 1")
            self.assertEqual(s.cmd("rfree"), "RFREE 0")
            # Full at capacity 1: the second publish must be refused, or the
            # producer would be writing a slot the consumer still owns.
            self.assertEqual(s.cmd("rpush 4 4 2 0 0"), "RPUSH 0")
            self.assertEqual(s.cmd("rpeek"),
                             "RPEEK 3 0 4 11 22")
            self.assertEqual(s.cmd("rretire"), "RRETIRE ok")
            self.assertEqual(s.cmd("rpend"), "RPEND 0")
            self.assertEqual(s.cmd("rfree"), "RFREE 1")
            # Reuse is legal only now, after the retire.
            self.assertEqual(s.cmd("rpush 5 4 1 33 44"), "RPUSH 1")
            self.assertEqual(s.cmd("rpeek"), "RPEEK 5 4 1 33 44")
        finally:
            s.close()

    def test_fill_full_reuse_and_wraparound(self):
        s = self.session()
        try:
            cap = 3
            self.assertEqual(s.cmd(f"ring {cap}"), "RING ok")

            def drain_n(published, n):
                # FIFO check at every retire: the ring must hand out the
                # oldest not-yet-consumed descriptor, whatever the wrapped
                # slot index happens to be.
                for _ in range(n):
                    self.assertEqual(s.cmd("rpend"), "RPEND 1")
                    head = published[0]
                    self.assertEqual(
                        s.cmd("rpeek"),
                        "RPEEK " + " ".join(map(str, head)))
                    self.assertEqual(s.cmd("rretire"), "RRETIRE ok")
                    published.pop(0)
                return published

            def push(nxt):
                self.assertEqual(
                    s.cmd("rpush " + " ".join(map(str, nxt))), "RPUSH 1")

            published = []
            # Fill to full, then verify reuse requires retire.
            for i in range(cap):
                nxt = (i, i * 8, 2, i, 0)
                push(nxt)
                published.append(nxt)
            self.assertEqual(s.cmd("rfree"), "RFREE 0")
            self.assertEqual(s.cmd("rpush 9 9 9 9 9"), "RPUSH 0")
            # Retire one, publish one: slot 0 is reused by counter 3.
            published = drain_n(published, 1)
            self.assertEqual(s.cmd("rfree"), "RFREE 1")
            nxt = (30, 24, 8, 7, 0)
            push(nxt)
            published.append(nxt)
            # Two more full laps: counters wrap modulo capacity (not a
            # power of two), and every peek must return the exact descriptor
            # its counter published, in order.
            for _ in range(2):
                published = drain_n(published, len(published))
                for k in range(cap):
                    nxt = (100 + k, 8 * k, 4, k, 1)
                    push(nxt)
                    published.append(nxt)
            published = drain_n(published, len(published))
            self.assertEqual(published, [])
            self.assertEqual(s.cmd("rpend"), "RPEND 0")
            target = int(s.cmd("rtarget").split()[1])
            self.assertEqual(s.cmd(f"rdrained {target}"), "RDRAINED 1")
        finally:
            s.close()

    def test_descriptor_roundtrip_bit_exact(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("ring 8"), "RING ok")
            events = [0, 1, 0xDEADBEEF, (1 << 63), (1 << 64) - 1, 0x7FFFFFFF]
            gs = [1, 2, 4, 8]
            for i, (event, g) in enumerate(
                    [(e, g) for e in events for g in gs][:16]):
                slot = i % 5
                row_start = i * 7
                self.assertEqual(
                    s.cmd(f"rpush {slot} {row_start} {g} "
                          f"{event & 0xFFFFFFFF} {event >> 32}"),
                    "RPUSH 1")
                self.assertEqual(
                    s.cmd("rpeek"),
                    f"RPEEK {slot} {row_start} {g} "
                    f"{event & 0xFFFFFFFF} {event >> 32}")
                self.assertEqual(s.cmd("rretire"), "RRETIRE ok")
        finally:
            s.close()

    def test_drain_barrier_words(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("ring 4"), "RING ok")
            self.assertEqual(s.cmd("rpush 0 0 1 0 0"), "RPUSH 1")
            self.assertEqual(s.cmd("rpush 1 1 1 0 0"), "RPUSH 1")
            target = int(s.cmd("rtarget").split()[1])
            self.assertEqual(target, 2)
            self.assertEqual(s.cmd(f"rdrained {target}"), "RDRAINED 0")
            self.assertEqual(s.cmd("rretire"), "RRETIRE ok")
            # tail(1) != head(2): the barrier must NOT be satisfied -- the
            # second group's section has not returned.
            self.assertEqual(s.cmd(f"rdrained {target}"), "RDRAINED 0")
            self.assertEqual(s.cmd("rretire"), "RRETIRE ok")
            self.assertEqual(s.cmd(f"rdrained {target}"), "RDRAINED 1")
        finally:
            s.close()

    def test_counter_wrap_across_2p32(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("ring 3"), "RING ok")
            # Park both counters 4 below the 32-bit wrap. This is the only
            # way to cover the wrap at test speed; the executor's own
            # counters pass through it only after 4G handoffs.
            self.assertEqual(s.cmd("rpreset 4294967292"), "RPRESET ok")
            for i in range(3):
                self.assertEqual(s.cmd(f"rpush {i} {i} 1 {i} 0"), "RPUSH 1")
            self.assertEqual(s.cmd("rpush 9 9 9 9 9"), "RPUSH 0")  # full
            self.assertEqual(s.cmd("rfree"), "RFREE 0")
            for _ in range(3):
                self.assertEqual(s.cmd("rretire"), "RRETIRE ok")
            # head wrapped: 0xFFFFFFFC + 5 == 1 (mod 2^32)
            self.assertEqual(s.cmd("rpush 5 5 2 5 0"), "RPUSH 1")
            self.assertEqual(s.cmd("rpush 6 6 2 6 0"), "RPUSH 1")
            self.assertEqual(s.cmd("rtarget"), "RTARGET 1")
            self.assertEqual(s.cmd("rdrained 1"), "RDRAINED 0")
            while s.cmd("rpend") == "RPEND 1":
                self.assertEqual(s.cmd("rretire"), "RRETIRE ok")
            self.assertEqual(s.cmd("rdrained 1"), "RDRAINED 1")
            self.assertEqual(s.cmd("rfree"), "RFREE 3")
        finally:
            s.close()

    def test_reset_starts_a_clean_epoch(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("ring 4"), "RING ok")
            self.assertEqual(s.cmd("rpush 0 0 1 0 0"), "RPUSH 1")
            self.assertEqual(s.cmd("rreset"), "RRESET ok")
            self.assertEqual(s.cmd("rpend"), "RPEND 0")
            self.assertEqual(s.cmd("rfree"), "RFREE 4")
            self.assertEqual(s.cmd("rtarget"), "RTARGET 0")
        finally:
            s.close()


class ExecutorLifecycleTests(unittest.TestCase):
    """The bind/submit/drain/join protocol against real host threads."""

    @classmethod
    def setUpClass(cls):
        cls.harness = Harness()
        cls.driver = cls.harness.compile()

    @classmethod
    def tearDownClass(cls):
        cls.harness.close()

    def session(self) -> Session:
        return Session(self.driver)

    def test_bind_refusals_and_submit_before_bind(self):
        s = self.session()
        try:
            # Null section: bind must refuse, not store a null pointer the
            # bound thread would later call.
            self.assertEqual(s.cmd("fn none"), "FN none")
            self.assertEqual(s.cmd("bind 4"), "BIND -1")
            # Depth 0: depth is a parameter with no default (HmxSpscRing.h);
            # a silent default would hide a compiler-side derivation bug.
            self.assertEqual(s.cmd("fn probe"), "FN probe")
            self.assertEqual(s.cmd("bind 0"), "BIND -2")
            # No thread was created by the refusals.
            probe = parse_kv(s.cmd("probe"))
            self.assertEqual(probe["createdThreads"], "0")
            self.assertEqual(probe["ensureCalls"], "0")
            # Submit before bind accepts nothing.
            self.assertEqual(s.cmd(submit_args([(1, 0, 1, 5)])), "SUBMIT 0")
            err = s.close()
        finally:
            s.kill()
        self.assertIn("submit before bind", err)

    def test_roundtrip_g_semantics_and_lock_first_half(self):
        s = self.session()
        try:
            groups = [
                (1, 0, 1, 0x1122334455667788),
                (2, 8, 2, 0),
                (3, 16, 4, (1 << 63) | 7),
                (4, 24, 8, (1 << 64) - 1),
            ]
            self.assertEqual(s.cmd("fn probe"), "FN probe")
            self.assertEqual(s.cmd("bind 4"), "BIND 0")

            # The thread is created inside bind, so this much is already
            # deterministic. The ensure COUNT is not probed here: thread
            # creation returns before the new thread runs its first
            # instruction, so ensureCalls is racy at this point. It is
            # asserted after the drain below, where the production ordering
            # guarantees it: ensure runs at thread entry, before the loop
            # that runs sections, and every section ran before drain
            # returned.
            probe = parse_kv(s.cmd("probe"))
            self.assertEqual(probe["createdThreads"], "1")

            # Producer writes its slots before handing them over.
            for slot, _, _, _ in groups:
                self.assertEqual(s.cmd(f"poke {slot} {100 + slot}"), "POKE ok")
            self.assertEqual(s.cmd(submit_args(groups)), "SUBMIT 4")
            self.assertEqual(s.cmd("drain"), "DRAIN ok")

            probe = parse_kv(s.cmd("probe"))
            # THE LOCK MIGRATION'S FIRST HALF: ensure called exactly once,
            # on the thread that was created (deterministic here -- see the
            # comment above).
            self.assertEqual(probe["ensureCalls"], "1")
            self.assertEqual(probe["ensureThread"], probe["boundThread"])

            log = parse_fnlog(s.cmd("fnlog"))
            self.assertEqual(len(log), 4)
            bound = probe["boundThread"]
            for entry, (slot, row_start, row_count, event) in zip(log, groups):
                # Execution happened on the bound thread, not the submitter.
                self.assertEqual(entry["thread"], bound)
                # The batch G travels with the descriptor: one section call
                # per group, never coalesced or split.
                self.assertEqual(entry["slot"], slot)
                self.assertEqual(entry["rowStart"], row_start)
                self.assertEqual(entry["rowCount"], row_count)
                self.assertEqual(entry["event"], event)
                # The section read exactly what the producer poked...
                self.assertEqual(entry["read"], 100 + slot)
                # ...and its marker write is visible through the barrier.
                self.assertEqual(entry["wrote"], 0xC0DE0000 | (entry["seq"] & 0xFFFF))
                self.assertEqual(
                    int(s.cmd(f"shadow {slot}").split()[2]), entry["wrote"])
            # ensure ran BEFORE any section (sequence numbers).
            self.assertTrue(all(e["seq"] > int(probe["ensureSeq"]) for e in log))
            self.assertEqual(parse_violations(s.cmd("violations")), [])

            self.assertEqual(s.cmd("join"), "JOIN ok")
            # Submit after join is refused (started gate), not queued.
            self.assertEqual(s.cmd(submit_args([(1, 0, 1, 1)])), "SUBMIT 0")
        finally:
            s.close()

    def test_full_ring_short_return_and_tail_recovery(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("fn slow"), "FN slow")
            self.assertEqual(s.cmd("bind 2"), "BIND 0")
            groups = [(i, i * 8, 2, i) for i in range(5)]
            for slot, _, _, _ in groups:
                self.assertEqual(s.cmd(f"poke {slot} {10 + slot}"), "POKE ok")
            # Capacity 2, nothing in flight, each section sleeps 300us while
            # the submit loop runs in well under a microsecond: exactly 2
            # are accepted. This is the never-block contract -- the caller
            # owns the dropped tail.
            self.assertEqual(s.cmd(submit_args(groups)), "SUBMIT 2")
            # The consumer owns slots 0 and 1 now; a producer poke at either
            # must be refused (checked in OwnershipTests too, but assert the
            # state here while the ring is provably full).
            self.assertEqual(s.cmd("poke 0 999"), "POKE VIOLATION")
            self.assertEqual(s.cmd("drain"), "DRAIN ok")
            # Recover the dropped tail: the caller re-submits it. One group
            # at a time, because a 3-group submit into a 2-deep ring would
            # itself be cut short -- that cut is the contract, not a bug.
            for g in groups[2:]:
                self.assertEqual(s.cmd(submit_args([g])), "SUBMIT 1")
                self.assertEqual(s.cmd("drain"), "DRAIN ok")

            log = parse_fnlog(s.cmd("fnlog"))
            self.assertEqual([e["slot"] for e in log], [0, 1, 2, 3, 4])
            for entry, (slot, _, _, _) in zip(log, groups):
                self.assertEqual(entry["read"], 10 + slot)
            # The one recorded violation is the deliberate in-flight probe
            # above (POKE VIOLATION); nothing else may be recorded.
            v = parse_violations(s.cmd("violations"))
            self.assertEqual(
                v, [{"kind": "poke-inflight", "slot": 0, "owner": 0xFFFFFFFF}])
            err = s.close()
        finally:
            s.kill()
        # The production full-ring path logged at ERROR; the stub FARF
        # lands on stderr. A silent drop would be un-debuggable.
        self.assertIn("ring full", err)

    def test_park_paths_actually_park(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("fn slow"), "FN slow")
            self.assertEqual(s.cmd("bind 2"), "BIND 0")
            self.assertEqual(s.cmd(submit_args([(0, 0, 1, 1), (1, 8, 1, 2)])),
                             "SUBMIT 2")
            # The producer's drain exhausts its bounded spin while the first
            # section sleeps and parks on the ring's tail word.
            self.assertEqual(s.cmd("drain"), "DRAIN ok")
            probe = parse_kv(s.cmd("probe"))
            self.assertGreaterEqual(int(probe["futexWaits"]), 1,
                                    "drain never parked: the bounded-spin-"
                                    "then-park path did not execute")
            # Let the consumer go idle: it spins out its budget and parks on
            # the seqn word.
            time.sleep(0.2)
            probe = parse_kv(s.cmd("probe"))
            self.assertGreaterEqual(int(probe["futexWaits"]), 2,
                                    "the consumer never parked: the idle "
                                    "path did not execute")
            self.assertGreaterEqual(int(probe["futexWakes"]), 2)
        finally:
            s.close()

    def test_wraparound_torture(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("fn probe"), "FN probe")
            # Non-power-of-two depth on purpose: the modulo arithmetic must
            # not lean on a mask.
            self.assertEqual(s.cmd("bind 3"), "BIND 0")
            expected = []
            for rnd in range(40):
                # Distinct slots within a round: in-flight groups occupy
                # distinct VTCM slots by construction (that is what the
                # ownership model hands over), and the shadow's expected
                # model assumes it. Slot reuse ACROSS rounds is fine -- the
                # drain retires everything before the next round pokes.
                groups = [((rnd * 5 + i) % 32, rnd * 5 + i, 1 + (i % 8),
                           (rnd << 32) | i) for i in range(5)]
                for slot, _, _, _ in groups:
                    self.assertEqual(
                        s.cmd(f"poke {slot} {1000 + rnd * 10 + slot}"), "POKE ok")
                # Capacity 3 against 5 groups: the submit may legitimately
                # come back short (never-block contract). The caller owns
                # the dropped tail, so it drains and re-submits until every
                # group landed -- which is exactly what the S3 compiler
                # side will have to do with the short return.
                done = 0
                while done < len(groups):
                    reply = s.cmd(submit_args(groups[done:]))
                    accepted = int(reply.split()[1])
                    self.assertGreaterEqual(accepted, 1,
                                            "an empty ring accepted nothing")
                    done += accepted
                    self.assertEqual(s.cmd("drain"), "DRAIN ok")
                expected.extend(groups)
            log = parse_fnlog(s.cmd("fnlog"))
            self.assertEqual(len(log), len(expected))
            for entry, (slot, row_start, row_count, event) in zip(log, expected):
                self.assertEqual(
                    (entry["slot"], entry["rowStart"], entry["rowCount"],
                     entry["event"]),
                    (slot, row_start, row_count, event))
            self.assertEqual(parse_violations(s.cmd("violations")), [])
            self.assertEqual(s.cmd("join"), "JOIN ok")
        finally:
            s.close()

    def test_rebind_never_restarts_the_thread(self):
        s = self.session()
        try:
            # Grow: 2 -> 5.
            self.assertEqual(s.cmd("fn slow"), "FN slow")
            self.assertEqual(s.cmd("bind 2"), "BIND 0")
            self.assertEqual(s.cmd(submit_args([(0, 0, 1, 1), (1, 1, 1, 2)])),
                             "SUBMIT 2")
            self.assertEqual(s.cmd("drain"), "DRAIN ok")
            self.assertEqual(s.cmd("bind 5"), "BIND 0")
            # The ring really grew: 5 fit in one submit with nothing in
            # flight (deterministic -- each section sleeps 300us).
            self.assertEqual(
                s.cmd(submit_args([(i, i, 1, i) for i in range(5)])),
                "SUBMIT 5")
            self.assertEqual(s.cmd("drain"), "DRAIN ok")
            # Shrink: 5 -> 3 keeps the larger ring (grow-only).
            self.assertEqual(s.cmd("fn probe"), "FN probe")
            self.assertEqual(s.cmd("bind 3"), "BIND 0")
            self.assertEqual(
                s.cmd(submit_args([(i, i, 1, i) for i in range(5)])),
                "SUBMIT 5")
            self.assertEqual(s.cmd("drain"), "DRAIN ok")

            # The property the lifetime lock depends on: ONE thread, ONE
            # ensure, across every rebind. A restarted thread would need a
            # second ensure and block on its own predecessor's held lock.
            probe = parse_kv(s.cmd("probe"))
            self.assertEqual(probe["createdThreads"], "1")
            self.assertEqual(probe["ensureCalls"], "1")
            self.assertEqual(parse_violations(s.cmd("violations")), [])
            self.assertEqual(s.cmd("join"), "JOIN ok")
        finally:
            s.close()

    def test_rebind_swaps_the_section_only_after_drain(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("fn slow"), "FN slow")
            self.assertEqual(s.cmd("bind 2"), "BIND 0")
            self.assertEqual(s.cmd(submit_args([(0, 0, 1, 1), (1, 1, 1, 2)])),
                             "SUBMIT 2")
            # Rebind while the two slow groups are in flight. The rebind's
            # drain must let them finish under the OLD section before the
            # new one is installed -- the fnlog order below proves the old
            # section ran them (both read the poked values), and no group is
            # lost or duplicated.
            self.assertEqual(s.cmd("fn probe"), "FN probe")
            self.assertEqual(s.cmd("bind 2"), "BIND 0")
            self.assertEqual(s.cmd("drain"), "DRAIN ok")
            log = parse_fnlog(s.cmd("fnlog"))
            self.assertEqual([e["slot"] for e in log], [0, 1])
            self.assertEqual(parse_violations(s.cmd("violations")), [])
            self.assertEqual(s.cmd("join"), "JOIN ok")
        finally:
            s.close()


class OwnershipTests(unittest.TestCase):
    """Ownership transfer is enforced at the data level, and the detectors
    are proven to fire by committed negative controls."""

    @classmethod
    def setUpClass(cls):
        cls.harness = Harness()
        cls.driver = cls.harness.compile()

    @classmethod
    def tearDownClass(cls):
        cls.harness.close()

    def session(self) -> Session:
        return Session(self.driver)

    def test_producer_cannot_write_an_inflight_slot(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("fn slow"), "FN slow")
            self.assertEqual(s.cmd("bind 2"), "BIND 0")
            self.assertEqual(s.cmd("poke 7 70"), "POKE ok")
            self.assertEqual(s.cmd(submit_args([(7, 0, 2, 1)])), "SUBMIT 1")
            # In flight: the consumer owns slot 7 until the drain retires it.
            self.assertEqual(s.cmd("poke 7 71"), "POKE VIOLATION")
            v = parse_violations(s.cmd("violations"))
            self.assertEqual(len(v), 1)
            self.assertEqual(v[0]["kind"], "poke-inflight")
            self.assertEqual(v[0]["slot"], 7)
            # After the barrier the slot is the producer's again.
            self.assertEqual(s.cmd("drain"), "DRAIN ok")
            self.assertEqual(s.cmd("poke 7 72"), "POKE ok")
            # The violation was recorded once, not twice.
            self.assertEqual(len(parse_violations(s.cmd("violations"))), 1)
            self.assertEqual(s.cmd("join"), "JOIN ok")
        finally:
            s.close()

    def test_negative_control_poke_force_fires_the_read_detector(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("fn slow"), "FN slow")
            self.assertEqual(s.cmd("bind 3"), "BIND 0")
            self.assertEqual(s.cmd("poke 9 111"), "POKE ok")
            self.assertEqual(s.cmd("poke 8 222"), "POKE ok")
            self.assertEqual(s.cmd("poke 7 333"), "POKE ok")
            # Three groups. Group 9 is read first -- its slot is NEVER
            # bypassed, so it must come back clean no matter how the wake
            # races. Groups 8 and 7 are read only after group 9's 20 ms
            # section sleep, so poke-forcing them microseconds after the
            # submit is deterministically in place before their reads.
            self.assertEqual(
                s.cmd(submit_args([(9, 0, 1, 1), (8, 1, 1, 2), (7, 2, 1, 3)])),
                "SUBMIT 3")
            # Negative control: bypass the producer guard on the LATER
            # groups' slots, exactly the write the guard exists for. The
            # sections' read-validation must then fire -- proving that
            # detector is not vacuous.
            self.assertEqual(s.cmd("poke-force 8 999"), "POKE-FORCE ok")
            self.assertEqual(s.cmd("poke-force 7 888"), "POKE-FORCE ok")
            self.assertEqual(s.cmd("drain"), "DRAIN ok")
            v = parse_violations(s.cmd("violations"))
            mismatches = [x for x in v if x["kind"] == "read-mismatch"]
            self.assertEqual(
                sorted(x["slot"] for x in mismatches), [7, 8],
                f"expected exactly the two bypassed slots, got {v}")
            # The first group's slot was untouched by the bypass.
            log = parse_fnlog(s.cmd("fnlog"))
            self.assertEqual(log[0]["read"], 111)
            self.assertEqual(s.cmd("join"), "JOIN ok")
        finally:
            s.close()

    def test_negative_control_rogue_section_cross_write_detected(self):
        s = self.session()
        try:
            self.assertEqual(s.cmd("fn rogue"), "FN rogue")
            self.assertEqual(s.cmd("bind 2"), "BIND 0")
            self.assertEqual(s.cmd("poke 10 101"), "POKE ok")
            self.assertEqual(s.cmd(submit_args([(10, 0, 1, 1)])), "SUBMIT 1")
            self.assertEqual(s.cmd("drain"), "DRAIN ok")
            v = parse_violations(s.cmd("violations"))
            self.assertEqual(len(v), 1)
            self.assertEqual(v[0]["kind"], "cross-write")
            self.assertEqual(v[0]["slot"], 11)
            self.assertEqual(v[0]["owner"], 10)
            # The legal part of the rogue section still ran to completion.
            log = parse_fnlog(s.cmd("fnlog"))
            self.assertEqual([e["slot"] for e in log], [10])
            self.assertEqual(log[0]["read"], 101)
            self.assertEqual(s.cmd("join"), "JOIN ok")
        finally:
            s.close()


class SourceContractTests(unittest.TestCase):
    """Source-text locks on the ABI and the decisions behind it (the same
    pattern as test_hmx_leaf_abi_contract.py: parse the header, assert the
    shape, so a change is a loud failure rather than silent adaptation)."""

    def test_abi_header_declares_the_four_entries(self):
        text = ROLE_EXECUTOR_H.read_text(encoding="utf-8")
        for entry in ("hexagon_runtime_hmx_role_bind",
                      "hexagon_runtime_hmx_role_submit",
                      "hexagon_runtime_hmx_role_drain",
                      "hexagon_runtime_hmx_role_join"):
            self.assertIn(entry, text, f"{entry} missing from the ABI header")
        self.assertIn("#define HEXMLIR_HMX_ROLE_EXECUTOR_ABI 1", text)
        for code in ("HEXMLIR_HMX_ROLE_OK", "HEXMLIR_HMX_ROLE_ERR_NULL_FN",
                     "HEXMLIR_HMX_ROLE_ERR_DEPTH",
                     "HEXMLIR_HMX_ROLE_ERR_CREATE"):
            self.assertIn(code, text)
        # bind takes the depth as a parameter -- no default in the signature.
        self.assertRegex(
            text,
            r"int32_t hexagon_runtime_hmx_role_bind\(HmxSectionFn fn, "
            r"uint32_t depth\)")

    def test_descriptor_shape_is_locked(self):
        text = SPSC_RING_H.read_text(encoding="utf-8")
        self.assertRegex(text, r"struct HmxTileGroupDesc \{")
        for field in ("uint32_t slot;", "uint32_t rowStart;",
                      "uint32_t rowCount;", "uint64_t event;"):
            self.assertIn(field, text, f"descriptor field {field} renamed?")
        # Both sides compile this header; the static_assert is what makes a
        # layout change a build break instead of silent ABI skew.
        self.assertIn('static_assert(sizeof(HmxTileGroupDesc) == 24', text)

    def test_lifetime_lock_decision_is_pinned(self):
        """The lock migration's first half, as a test: exactly one ensure
        call (the thread start), and no unlock anywhere in the executor."""
        text = ROLE_EXECUTOR_CPP.read_text(encoding="utf-8")
        n_ensure = len(re.findall(r"hexagon_runtime_hmx_ensure_dsp\(\);", text))
        self.assertEqual(
            n_ensure, 1,
            "T_HMX must call the HMX ensure exactly once, at thread start "
            "(lifetime hold; see HmxRoleExecutor.h decision 2)")
        self.assertEqual(
            len(re.findall(r"hmx_unlock", text)), 0,
            "the role executor must never unlock: the lock is held for the "
            "thread's lifetime by recorded decision")

    def test_no_hardcoded_ring_depth(self):
        """Depth is a bind() parameter derived by the compiler side; the
        production sources must not invent a numeric default."""
        for path in (SPSC_RING_H, ROLE_EXECUTOR_H, ROLE_EXECUTOR_CPP):
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                re.search(r"\b(?:capacity|depth)\s*=\s*[1-9][0-9]*\b", text),
                f"{path.name} assigns a literal ring depth; depth is a "
                "parameter (HmxSpscRing.h), not a constant to reinvent")

    def test_production_sources_do_not_reach_into_the_host_emu(self):
        for path in (SPSC_RING_H, ROLE_EXECUTOR_H, ROLE_EXECUTOR_CPP):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "hmx_role_host_emu", text,
                f"{path.name} references the host emulation directory; the "
                "stubs are test-only and must not leak into the device build")


class CompilerVariationSmoke(unittest.TestCase):
    """The core battery under other optimization levels. Atomics and the
    spin-then-park loops are exactly the code whose behavior a compiler is
    most tempted to transform; -O0 vs -O2 is a cheap cross-check."""

    def test_core_battery_under_opt_levels(self):
        for opt in ("-O0", "-O2"):
            harness = Harness()
            try:
                driver = harness.compile(opt=opt)
                s = Session(driver)
                try:
                    self.assertEqual(s.cmd("fn probe"), "FN probe")
                    self.assertEqual(s.cmd("bind 2"), "BIND 0")
                    groups = [(i, i * 4, 2 ** i, (i << 62) | i)
                              for i in range(2)]
                    for slot, _, _, _ in groups:
                        self.assertEqual(
                            s.cmd(f"poke {slot} {50 + slot}"), "POKE ok")
                    self.assertEqual(s.cmd(submit_args(groups)), "SUBMIT 2")
                    self.assertEqual(s.cmd("drain"), "DRAIN ok")
                    log = parse_fnlog(s.cmd("fnlog"))
                    self.assertEqual(len(log), 2)
                    for entry, (slot, _, _, event) in zip(log, groups):
                        self.assertEqual(entry["slot"], slot)
                        self.assertEqual(entry["event"], event)
                        self.assertEqual(entry["read"], 50 + slot)
                    self.assertEqual(parse_violations(s.cmd("violations")), [])
                    self.assertEqual(s.cmd("join"), "JOIN ok")
                finally:
                    s.close()
            finally:
                harness.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
