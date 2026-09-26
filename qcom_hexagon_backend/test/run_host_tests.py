#!/usr/bin/env python3
"""The single entry point for the host-only Python test suite.

Why this file exists
--------------------
Every host test under ``qcom_hexagon_backend/test/`` and
``qcom_hexagon_backend/bin/runtime/test/`` is runnable two ways: as a script
(``python <file>``) and under pytest (``pytest <file>``).  Those two ways used
to disagree, in both directions:

* A test file could pass as a script and collect **zero** tests under pytest,
  so a whole file was a silent no-op gate in one of the two styles.  Two files
  did exactly that.
* A test file could collect tests under pytest and run **nothing** as a script,
  because it had no ``__main__`` guard.  One file did exactly that.
* The whole suite was green or red depending on the *interpreter* and on one
  entry in ``sys.path`` -- see "The triton shadowing trap" below.  The two
  natural ways to ask the question disagreed, and nothing in the tree said
  which one was authoritative.

So this runner does three things:

1. Pins the conditions under which "green" is a meaningful word: it selects an
   interpreter that can import triton, and it puts ``triton/python`` on
   ``sys.path`` for every subprocess it starts.
2. Runs every discovered file **both** ways and fails if either is red.
3. Asserts the two structural invariants that made the disagreement possible:
   no file is silently empty in either style, and no file's collected test count
   depends on which other files are in the invocation.

It needs no device, no network and no build.  It runs no test itself; it only
supervises interpreters and compares results.

Usage
-----
    python qcom_hexagon_backend/test/run_host_tests.py

Exit status is 0 only if every discovered file is green in both styles and both
structural invariants hold.  Set ``HEXMLIR_TEST_PYTHON`` to force a particular
interpreter.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
BACKEND = REPO / "qcom_hexagon_backend"
#: The two directories whose ``test_*.py`` files make up the host suite.
TEST_ROOTS = (BACKEND / "test", BACKEND / "bin" / "runtime" / "test")
#: ``triton`` is a PEP 420 namespace package trap unless this is on ``sys.path``
#: -- see the module docstring of :func:`_probe_interpreter`.
TRITON_PYTHON = REPO / "triton" / "python"
#: The regular package we must actually get, as opposed to the shadow.
TRITON_INIT = TRITON_PYTHON / "triton" / "__init__.py"

# Exit codes from the interpreter probe, so the messages can be specific.
_PROBE_OK = 0
_PROBE_NO_PYTEST = 10
_PROBE_NO_TRITON = 11
_PROBE_SHADOWED = 12

_PROBE_SOURCE = f"""
import sys

try:
    import pytest
except Exception as exc:  # noqa: BLE001 - report, do not raise
    print("pytest:", exc)
    sys.exit({_PROBE_NO_PYTEST})
try:
    import triton
except Exception as exc:  # noqa: BLE001
    print("triton:", exc)
    sys.exit({_PROBE_NO_TRITON})
origin = getattr(triton, "__file__", None)
if origin is None:
    print("triton.__path__:", list(getattr(triton, "__path__", [])))
    sys.exit({_PROBE_SHADOWED})
print(origin)
sys.exit({_PROBE_OK})
"""

_COLLECTED = re.compile(r"^(?P<nodeid>\S+::\S+)\s*$")
_SUMMARY = re.compile(
    r"(?P<count>\d+) (?P<word>passed|failed|error|errors|skipped|xfailed|subtests?)"
)


@dataclass
class Outcome:
    """What one invocation of one file did."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    @property
    def quiet(self) -> bool:
        """True when the run said nothing at all.

        A script that exits 0 without printing has not demonstrated that it ran
        anything; that is the silent no-op this runner exists to catch.
        """
        return not self.output.strip()


@dataclass
class FileReport:
    """Both styles' results for one test file, plus its structural verdicts."""

    path: Path
    script: Outcome | None = None
    isolated: Outcome | None = None
    isolated_tests: int = 0
    batch_tests: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(REPO))

    @property
    def green(self) -> bool:
        return not self.problems


def _child_env() -> dict[str, str]:
    """The environment every child runs with.

    ``triton/python`` is prepended to ``PYTHONPATH`` rather than replacing it.
    Without it, ``import triton`` can bind the ``triton/`` source checkout in
    the repository root as a namespace package, and the suite's answer depends
    on the caller's shell.
    """
    env = dict(os.environ)
    parts = [str(TRITON_PYTHON)]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run(argv: list[str], cwd: Path) -> Outcome:
    """Run ``argv`` to completion, capturing both streams.

    ``cwd`` is passed explicitly and the child is started from ``REPO`` so that
    the result never depends on where the runner was invoked from.
    """
    done = subprocess.run(
        argv,
        cwd=str(cwd),
        env=_child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    return Outcome(done.returncode, done.stdout, done.stderr)


def _probe_interpreter(python: Path) -> tuple[int, str]:
    """Can ``python`` import pytest and the real triton package?"""
    done = subprocess.run(
        [str(python), "-c", _PROBE_SOURCE],
        cwd=str(REPO),
        env=_child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, (done.stdout + done.stderr).strip()


def _interpreter_candidates() -> list[Path]:
    """Interpreters to try, best first.

    ``HEXMLIR_TEST_PYTHON`` is not consulted here; :func:`select_interpreter`
    handles it so that an explicit request cannot be silently overridden.
    """
    found: list[Path] = [Path(sys.executable)]
    # The workspace virtualenv is a sibling of the repository, not a child of
    # it, so walk up from the repository root as well as looking inside it.
    for parent in (REPO, *REPO.parents):
        found.append(parent / ".venv" / "bin" / "python")
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        found.append(Path(venv) / "bin" / "python")

    unique: list[Path] = []
    for candidate in found:
        if candidate not in unique and candidate.is_file():
            unique.append(candidate)
    return unique


def select_interpreter() -> Path:
    """Return an interpreter that can import pytest and the real triton.

    Raises ``SystemExit`` with an explanation when none can.  This is the whole
    point of the runner: "which python" stops being the caller's problem.

    ``HEXMLIR_TEST_PYTHON`` is honoured strictly.  Silently falling back to a
    different interpreter than the one that was asked for would reintroduce
    exactly the class of bug this runner exists to remove: a green answer from
    a process the caller did not choose.
    """
    rejected: list[str] = []
    override = os.environ.get("HEXMLIR_TEST_PYTHON")
    candidates = _interpreter_candidates()
    if override:
        requested = Path(override)
        if not requested.is_file():
            print(
                f"error: HEXMLIR_TEST_PYTHON={override} is not an executable file",
                file=sys.stderr,
            )
            raise SystemExit(2)
        candidates = [requested]

    for candidate in candidates:
        code, detail = _probe_interpreter(candidate)
        if code == _PROBE_OK:
            return candidate
        if code == _PROBE_SHADOWED:
            rejected.append(
                f"{candidate}: 'triton' resolved to a namespace package "
                f"({detail or 'no __file__'}) instead of {TRITON_INIT}"
            )
        elif code == _PROBE_NO_PYTEST:
            rejected.append(f"{candidate}: no pytest ({detail})")
        elif code == _PROBE_NO_TRITON:
            rejected.append(f"{candidate}: cannot import triton ({detail})")
        else:
            rejected.append(f"{candidate}: probe failed rc={code} ({detail})")

    print("error: no interpreter can run the host suite.", file=sys.stderr)
    for line in rejected:
        print(f"  rejected {line}", file=sys.stderr)
    print(
        "\nThe host tests import triton's C extension, so the interpreter must be\n"
        "the one triton was built for.  Point at it explicitly with\n"
        "HEXMLIR_TEST_PYTHON=/path/to/python.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def discover() -> list[Path]:
    """Every host test file, in a stable order."""
    found: list[Path] = []
    for root in TEST_ROOTS:
        if not root.is_dir():
            print(f"error: missing test root {root}", file=sys.stderr)
            raise SystemExit(2)
        found.extend(sorted(root.glob("test_*.py")))
    if not found:
        print("error: discovered no test files", file=sys.stderr)
        raise SystemExit(2)
    return found


def _collected_counts(python: Path, files: list[Path]) -> dict[Path, int]:
    """Per-file collected test counts for one pytest invocation.

    Collection only: it is cheap, and comparing it across invocations is what
    detects a file whose tests appear or vanish depending on its neighbours.
    """
    outcome = _run(
        [str(python), "-m", "pytest", "--collect-only", "-q", *map(str, files)],
        REPO,
    )
    counts: dict[Path, int] = {path: 0 for path in files}
    for line in outcome.stdout.splitlines():
        match = _COLLECTED.match(line.strip())
        if not match:
            continue
        node = match.group("nodeid")
        for path in files:
            if node.endswith(path.name) or path.name in node:
                counts[path] += 1
                break
    return counts


def _parse_summary(output: str) -> str:
    """The trailing 'N passed' style clause, for the report."""
    tail = output.strip().splitlines()[-6:]
    for line in reversed(tail):
        if "passed" in line or "failed" in line or "error" in line:
            return line.strip()
    return ""


def main() -> int:
    python = select_interpreter()
    files = discover()

    print(f"interpreter : {python}")
    print(f"repository  : {REPO}")
    print(f"files       : {len(files)}")
    print()

    reports = {path: FileReport(path) for path in files}

    # Pass 1 -- each file as a script.  This is the style the runtime contracts
    # in bin/runtime/test/ are written for.
    print("== pass 1/3: as a script ==", flush=True)
    for path in files:
        report = reports[path]
        report.script = _run([str(python), str(path)], path.parent)
        state = "ok" if report.script.returncode == 0 else f"rc={report.script.returncode}"
        if report.script.quiet:
            state += " SILENT"
        print(f"  {state:>16}  {report.rel}", flush=True)

    # Pass 2 -- each file under pytest on its own.
    print("\n== pass 2/3: pytest, one file per invocation ==", flush=True)
    for path in files:
        report = reports[path]
        report.isolated = _run(
            [str(python), "-m", "pytest", "-q", str(path)], REPO
        )
        report.isolated_tests = _collected_counts(python, [path]).get(path, 0)
        state = "ok" if report.isolated.returncode == 0 else f"rc={report.isolated.returncode}"
        print(f"  {state:>16}  {report.rel}  ({report.isolated_tests} tests)", flush=True)

    # Pass 3 -- the whole suite in one pytest invocation, plus a collection-only
    # pass over the same file set for the order-dependence comparison.
    print("\n== pass 3/3: pytest, whole suite in one invocation ==", flush=True)
    batch = _run([str(python), "-m", "pytest", "-q", *map(str, files)], REPO)
    batch_counts = _collected_counts(python, files)
    for path in files:
        reports[path].batch_tests = batch_counts.get(path, 0)
    print(f"  {'ok' if batch.returncode == 0 else f'rc={batch.returncode}':>16}  whole suite")
    if batch.returncode != 0:
        print("\n--- whole-suite pytest output ---")
        print(batch.output.rstrip())
    print()

    # ---- structural invariants -------------------------------------------
    #
    # A file that runs nothing in one of the two styles is the same defect as a
    # gate that is never executed: it reports success without having checked
    # anything.  A file whose test count moves when its neighbours change means
    # the suite's result depends on collection order rather than on the code.
    for path in files:
        report = reports[path]
        if report.isolated_tests == 0:
            report.problems.append(
                "collects no tests under pytest: the file is a no-op in that style"
            )
        if report.script is not None and report.script.quiet:
            report.problems.append(
                "prints nothing when run as a script: the file is a no-op in that style"
            )
        if report.isolated_tests != report.batch_tests:
            report.problems.append(
                f"collected test count depends on invocation order: "
                f"{report.isolated_tests} alone vs {report.batch_tests} in the suite"
            )
        if report.script is not None and report.script.returncode != 0:
            report.problems.append(
                f"red as a script (rc={report.script.returncode})"
            )
        if report.isolated is not None and report.isolated.returncode != 0:
            report.problems.append(
                f"red under pytest (rc={report.isolated.returncode})"
            )

    # ---- report ----------------------------------------------------------
    width = max(len(report.rel) for report in reports.values())
    print(f"{'file'.ljust(width)}  {'script':>8}  {'pytest':>8}  {'tests':>5}  verdict")
    print("-" * (width + 40))
    failures = 0
    for path in files:
        report = reports[path]
        script = "ok" if report.script and report.script.returncode == 0 else "FAIL"
        pytest = "ok" if report.isolated and report.isolated.returncode == 0 else "FAIL"
        verdict = "ok" if report.green else "; ".join(report.problems)
        if not report.green:
            failures += 1
        print(
            f"{report.rel.ljust(width)}  {script:>8}  {pytest:>8}  "
            f"{report.isolated_tests:>5}  {verdict}"
        )
    print("-" * (width + 40))

    total_tests = sum(report.isolated_tests for report in reports.values())
    green = [r for r in reports.values() if r.green]
    print(f"{len(green)}/{len(files)} files green in both styles, {total_tests} tests")
    print(f"whole-suite pytest: {_parse_summary(batch.output) or f'rc={batch.returncode}'}")
    if failures:
        print(f"\nFAILED: {failures} file(s) are not green in both styles", file=sys.stderr)
        for report in reports.values():
            for problem in report.problems:
                print(f"  {report.rel}: {problem}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
