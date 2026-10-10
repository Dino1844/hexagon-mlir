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
   no file is silently empty in either style, and no file's collected test
   count depends on which other files are in the invocation.

It needs no device, no network and no build.  It runs no test itself; it only
supervises interpreters and compares results.

The torch-mlir model gate
-------------------------
``test/python/torch-mlir/`` -- five model-level tests (add_matmul_softmax,
concat_linear, gelu, layernorm, softmax) -- was covered by no gate at all
before 2026-10-10.  They cannot join ``TEST_ROOTS``: under pytest those files
run on the phone (``execute_and_compare`` -> ``execute_kernel`` ->
``HexagonExecutor.run``), so plain discovery would turn this runner, whose
entire premise is "no device, no phone", into a device gate.  What they *are*
good for is the part a host gate can check: the torch->linalg export, the
bytecode file, the MLIR->obj translation, the wrapper and the
``hexagon-clang++`` link.  A pure host bug anywhere in that chain used to be
invisible -- on 2026-10-01 every one of those models was red because
``parse_translation_metadata`` was unpacked with the old arity.

So ``--torch-mlir`` is a dedicated second entry point: it compiles each model
once and asserts only "the compile exited 0".  It is an *exit-code* gate and
must stay one -- identical sources do not compile to identical objects (the
``Softmax`` .o came out three different sizes in three back-to-back builds,
measured 2026-10-10; source:
``docs/analysis/torchmlir-gating-feasibility-2026-10-10.md``).  The parent run
includes it by default, so ``tools/run_tests.sh host`` carries the model-level
frontend regression gate with it.  Rationale and measurements live in that
same document.

Speed
-----
Passes 1 and 2 are independent subprocesses per file, so they run in parallel:
one worker per file (capped by ``--jobs``, default one per file up to the CPU
count).  The subprocesses of the SAME file stay sequential inside its worker --
a file must never race itself -- and the per-file pytest children run with the
cache provider disabled so parallel invocations cannot race on
``.pytest_cache``.  Pass 3 is a single pytest invocation over the whole suite
and stays serial: observing every file in one process is its entire purpose.

``--gate`` skips pass 3.  It is the fast regression gate for agent final
checks (both styles + the non-emptiness invariant, tens of seconds): it does
NOT check the order-independence invariant, which needs the whole suite in one
process.  Run the full mode when a test file is new or its structure changed.

Usage
-----
    python qcom_hexagon_backend/test/run_host_tests.py              # full: all three passes + model gate
    python qcom_hexagon_backend/test/run_host_tests.py --gate       # fast gate: passes 1+2 + model gate
    python qcom_hexagon_backend/test/run_host_tests.py --files A B  # full treatment on a subset
    python qcom_hexagon_backend/test/run_host_tests.py --jobs 8     # cap the parallelism
    python qcom_hexagon_backend/test/run_host_tests.py --torch-mlir # only the model gate (also what the parent invokes)

Exit status is 0 only if every discovered file is green in both styles and
both structural invariants hold (in ``--gate`` mode: both styles and the
non-emptiness invariant), and the torch-mlir model gate is green.  Set
``HEXMLIR_TEST_PYTHON`` to force a particular interpreter.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
BACKEND = REPO / "qcom_hexagon_backend"
#: The two directories whose ``test_*.py`` files make up the host suite.
TEST_ROOTS = (BACKEND / "test", BACKEND / "bin" / "runtime" / "test")
#: torch-mlir's model-level tests.  Deliberately *not* a ``TEST_ROOTS`` entry:
#: under pytest those files run on the phone, so discovery would turn this
#: runner into a device gate.  Covered instead by the ``--torch-mlir`` mode
#: below; see the module docstring.
TORCH_MLIR_DIR = REPO / "test" / "python" / "torch-mlir"
#: Where the model gate leaves the objects it compiles.  Set per run by
#: :func:`_torch_mlir_gate`; the tests themselves would otherwise rewrite the
#: ``*.mlirbc`` files that are tracked in that directory.
_TM_SCRATCH: Path | None = None
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
    seconds: float = 0.0

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(REPO))
        except ValueError:  # --files may point outside the repository
            return str(self.path)

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


# --------------------------------------------------------------------------- #
# The torch-mlir model gate (``--torch-mlir``)
# --------------------------------------------------------------------------- #
#
# The model tests live outside every gate, and their device half is not what a
# host runner should assert on: the numerical tolerances in that directory span
# five orders of magnitude and four of the five files pin no random seed, so
# "green on the phone" is not yet a meaningful claim about the code.  The host
# compile half is: it is stable enough for an exit code, it is the half that
# broke (every model red from one pure host bug) on 2026-10-01, and it costs
# about 40 s for the five models.


def _tm_scratch_dir(tag: str) -> Path:
    """A directory outside the repository for this run's artifacts.

    The tests write ``<Model>.mlirbc`` next to themselves and those files are
    tracked, so a gate that uses the tests' own path would dirty the tree it is
    gating -- and then "clean tree" would no longer mean "no side effects".
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"torchmlir-hostgate-{tag}-{stamp}-"))


def _tm_install_guard() -> list[str]:
    """Turn every device entry point into a loud error instead of an ssh.

    This is what makes "this gate never touches the DSP" a property of the gate
    rather than a promise about future code: if the compile path ever grows a
    call that reaches the device, the gate fails here instead of starting a
    session with the phone.
    """
    import triton.backends.qcom_hexagon_backend.hexagon_executor as hexec_mod
    import triton.backends.qcom_hexagon_backend.hexagon_launcher_base as hlb

    def _no_device(*_args, **_kwargs):
        raise AssertionError(
            "HOST GATE GUARD: a device/simulator execution path was reached"
        )

    patched: list[str] = []
    for name in ("run", "run_kernel_on_device", "run_kernel_on_simulator"):
        setattr(hexec_mod.HexagonExecutor, name, _no_device)
        patched.append(f"HexagonExecutor.{name}")
    hlb.HexagonLauncherBase.execute_kernel = _no_device
    patched.append("HexagonLauncherBase.execute_kernel")
    return patched


def _tm_host_only(utils) -> None:
    """Replace the half-steps of a torch-mlir test that must not run.

    ``execute_and_compare`` keeps everything up to and including the
    shared-object link and drops the launch; ``write_bytecode_to_file`` is
    redirected into the scratch directory; ``process_lwp`` becomes a no-op
    because it shells out to a script that reads ``/tmp/lwp.json`` -- an
    environment fact (it needs ``HEXAGON_MLIR_ROOT``), not a property of the
    code under test.
    """
    from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
    from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
    from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
        TorchMLIRHexagonLauncher,
    )

    def _write_bytecode_to_file(self):
        bytecode = self.mlir_module.operation.get_asm(binary=True)
        assert _TM_SCRATCH is not None
        self.linalg_filename = str(_TM_SCRATCH / (self.func_name + ".mlirbc"))
        with open(self.linalg_filename, "wb") as f:
            f.write(bytecode)

    def _execute_and_compare(
        self,
        rtol=1e-05,
        atol=1e-08,
        dump_outputs=False,
        iterations=1,
        options=None,
        enable_etm=False,
    ):
        # The x86 reference: the model must be constructible and runnable with
        # the inputs this test declares.  That is what catches a broken input
        # dimension, before the export is even attempted -- a symbolic trace
        # would happily record a layer norm over the wrong size.
        self.model(*self.inputs)
        del rtol, atol, dump_outputs, enable_etm  # nothing to compare against yet
        opts = dict(HexagonOptions().__dict__) if options is None else dict(options)
        hexec = HexagonExecutor(
            kernel_run_id="torch-mlir-hostgate",
            compile_only=True,
            enable_lwp=bool(opts.get("enableLWP", False)),
        )
        TorchMLIRHexagonLauncher().compile_torch_mlir(
            hexec,
            str(_TM_SCRATCH),
            str(self.linalg_filename),
            list(self.inputs),
            self.func_name,
            opts,
            iterations,
        )

    utils.ModelManager.write_bytecode_to_file = _write_bytecode_to_file
    utils.ModelManager.execute_and_compare = _execute_and_compare

    def _process_lwp():
        print("[host gate] process_lwp() skipped: no device run produced lwp.json")

    utils.process_lwp = _process_lwp


def _tm_first_param_set(fn) -> dict[str, object]:
    """The first configuration ``@pytest.mark.parametrize`` declares for a test.

    Reading it off the test keeps one source of truth for shapes and options:
    change the test file and the gate follows it, instead of compiling a stale
    copy of the model that lives in the runner.
    """
    params: dict[str, object] = {}
    for mark in getattr(fn, "pytestmark", []):
        if getattr(mark, "name", None) != "parametrize":
            continue
        names = mark.args[0]
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        values = mark.args[1][0]
        if not isinstance(values, tuple):
            values = (values,)
        for name, value in zip(names, values):
            params.setdefault(name, value)
    return params


def _tm_test_functions(path: Path) -> list[tuple[str, object]]:
    """The ``test_*`` callables defined in one torch-mlir test file."""
    module = importlib.import_module(path.stem)
    found: list[tuple[str, object]] = []
    for name, obj in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        found.append((name, obj))
    return found


class _Captured:
    """Everything a model compile writes, captured at the file-descriptor level.

    Rebinding ``sys.stdout`` is not enough: MLIR's pass diagnostics
    (``IR Dump After ...``) go through ``llvm::errs()``, i.e. file descriptor 2
    directly, so they survive a Python-level redirect and print five pages of IR
    per model.  Redirecting the descriptors catches both.  The text is kept for
    the report and shown only for a model that failed.
    """

    def __init__(self) -> None:
        self.text = ""
        self._sink: tempfile._TemporaryFileWrapper | None = None
        self._saved: tuple[int, int] | None = None

    def __enter__(self) -> "_Captured":
        self._sink = tempfile.TemporaryFile(mode="w+")
        self._saved = (os.dup(1), os.dup(2))
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._sink.fileno(), 1)
        os.dup2(self._sink.fileno(), 2)
        return self

    def __exit__(self, *_exc: object) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        assert self._saved is not None and self._sink is not None
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        os.close(self._saved[0])
        os.close(self._saved[1])
        self._sink.seek(0)
        self.text = self._sink.read()
        self._sink.close()


def _torch_mlir_gate() -> int:
    """The dedicated torch-mlir entry point: compile each model, exit 0 or not.

    Exit codes
        ``0``  every model compiled
        ``1``  at least one model failed
        ``2``  the gate cannot run at all (missing directory, missing imports)
    """
    global _TM_SCRATCH
    started = time.perf_counter()

    files = sorted(TORCH_MLIR_DIR.glob("test_*.py"))
    if not TORCH_MLIR_DIR.is_dir() or not files:
        print(
            f"error: no torch-mlir tests discovered in {TORCH_MLIR_DIR}",
            file=sys.stderr,
        )
        return 2

    _TM_SCRATCH = _tm_scratch_dir("run")
    print(f"models      : {len(files)} files in "
          f"{TORCH_MLIR_DIR.relative_to(REPO)}")
    print(f"scratch     : {_TM_SCRATCH}  (removed when the gate is green)")
    print("asserts     : export -> bytecode -> MLIR->obj -> wrapper -> .so, exit 0")
    print("does not    : launch anything, compare numbers, or diff object bytes")
    print()

    try:
        # The test files import `utils` as a top-level module, so their
        # directory must be importable before they are.
        if str(TORCH_MLIR_DIR) not in sys.path:
            sys.path.insert(0, str(TORCH_MLIR_DIR))
        import torch  # noqa: F401
        import torch_mlir.fx  # noqa: F401
        import utils  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        print(
            f"error: the torch-mlir host stack is not importable: {exc!r}",
            file=sys.stderr,
        )
        print(
            "This gate needs torch, torch_mlir and the triton backend on the "
            "interpreter it was started with.",
            file=sys.stderr,
        )
        return 2

    _tm_host_only(utils)
    guarded = _tm_install_guard()
    print(f"guard       : {len(guarded)} device entry points replaced by raisers")
    print()

    failures: list[str] = []
    for path in files:
        rel = str(path.relative_to(REPO))
        case_started = time.perf_counter()
        # Defaults to the file stem, so an import-time failure still names a file.
        name = path.stem
        failed: BaseException | None = None
        captured = _Captured()
        try:
            with captured:
                functions = _tm_test_functions(path)
                if not functions:
                    raise RuntimeError(f"{rel}: defines no test_* function")
                for name, fn in functions:
                    fn(**_tm_first_param_set(fn))
            verdict = "ok"
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - SystemExit on link failure too
            # Broad on purpose: a model gate's job is to name the model that
            # broke, whatever the shape of the breakage is.
            verdict = "FAIL"
            failed = exc
            failures.append(f"{rel}::{name}: {type(exc).__name__}: {exc}")
        seconds = time.perf_counter() - case_started
        print(f"  {verdict:>4}  {rel:<44} {name:<32} {seconds:>6.1f}s", flush=True)
        if failed is not None:
            print(f"        {type(failed).__name__}: {failed}")
            # Frames inside this repository are the actionable ones; the tail of
            # a torch exception is mostly torch internals.
            frames = [
                frame
                for frame in traceback.extract_tb(failed.__traceback__)
                if str(frame.filename).startswith(str(REPO))
            ]
            for frame in frames[-3:]:
                shown = Path(frame.filename).relative_to(REPO)
                print(f"        {shown}:{frame.lineno} in {frame.name}")
            for line in captured.text.strip().splitlines()[-4:]:
                print("        | " + line)
            print(f"      (full traceback still points at {rel}::{name})")

    print()
    green = len(files) - len(failures)
    print(f"{green}/{len(files)} torch-mlir models compiled on host, "
          f"{time.perf_counter() - started:.1f}s total")
    if failures:
        print("\nartifacts kept for inspection:", file=sys.stderr)
        print(f"  {_TM_SCRATCH}", file=sys.stderr)
        return 1
    print("artifacts removed (every model compiled)")
    shutil.rmtree(_TM_SCRATCH, ignore_errors=True)
    return 0


def _collected_counts(python: Path, files: list[Path]) -> dict[Path, int]:
    """Per-file collected test counts for one pytest invocation.

    Collection only: it is cheap, and comparing it across invocations is what
    detects a file whose tests appear or vanish depending on its neighbours.
    The cache provider is disabled: these run in parallel per file, and the
    collect-only cache has no reader.
    """
    outcome = _run(
        [
            str(python),
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *map(str, files),
        ],
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


def _run_one_file(python: Path, path: Path) -> FileReport:
    """One file's two styles plus its isolated collection count.

    The three subprocesses of ONE file run sequentially -- a file must never
    race itself (its script run and its pytest run can share in-process state
    conventions, and some tests mutate ``os.environ`` for their own duration).
    Different files run in parallel; see :func:`main`.
    """
    report = FileReport(path)
    file_started = time.perf_counter()
    report.script = _run([str(python), str(path)], path.parent)
    report.isolated = _run(
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            # Parallel per-file invocations would otherwise race on
            # .pytest_cache in the repository root; the per-file cache has no
            # reader (the whole-suite pass keeps its own).
            "-p",
            "no:cacheprovider",
            str(path),
        ],
        REPO,
    )
    report.isolated_tests = _collected_counts(python, [path]).get(path, 0)
    report.seconds = time.perf_counter() - file_started
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the host-only Python test suite both ways (script + pytest)."
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="fast regression gate: parallel passes 1+2, skip the whole-suite "
        "pass 3 (and with it the order-independence invariant -- use the full "
        "mode when a test file is new or its structure changed)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="run only these test files (full treatment on the subset)",
    )
    parser.add_argument(
        "--torch-mlir",
        action="store_true",
        help="run ONLY the torch-mlir model host-compile gate.  This is also "
        "the flag the parent process uses to supervise it as a child, so the "
        "gate never shares an interpreter with the rest of the suite.  The "
        "default (and --gate) runs include the model gate as an extra phase; "
        "--files is an explicit subset and does not",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="max files running in parallel in passes 1+2 "
        "(default: one per file, capped at the CPU count)",
    )
    return parser.parse_args()


def _parse_summary(output: str) -> str:
    """The trailing 'N passed' style clause, for the report."""
    tail = output.strip().splitlines()[-6:]
    for line in reversed(tail):
        if "passed" in line or "failed" in line or "error" in line:
            return line.strip()
    return ""


def main() -> int:
    started = time.perf_counter()
    args = _parse_args()

    if args.torch_mlir:
        # The dedicated entry point: compile the torch-mlir models and nothing
        # else.  The parent invokes this same flag in a child process, so the
        # gate's imports (torch, torch_mlir, the backend launcher) are never
        # loaded into the interpreter that supervises the contract suite.
        return _torch_mlir_gate()

    python = select_interpreter()

    if args.files:
        files: list[Path] = []
        for requested in args.files:
            path = requested.resolve()
            if not path.is_file():
                print(f"error: no such test file {requested}", file=sys.stderr)
                raise SystemExit(2)
            files.append(path)
    else:
        files = discover()

    jobs = min(len(files), max(1, args.jobs or (os.cpu_count() or 4)))
    total_passes = 2 if args.gate else 3

    print(f"interpreter : {python}")
    print(f"repository  : {REPO}")
    print(f"files       : {len(files)}")
    print(
        f"mode        : "
        f"{'gate (passes 1+2, parallel, no whole-suite pass)' if args.gate else 'full (all three passes)'}"
        f"  jobs={jobs}"
        + (", plus the torch-mlir model gate" if not args.files else "")
    )
    print()

    # Passes 1+2 -- one worker per file.  The subprocesses of a single file
    # stay sequential (see _run_one_file); different files have no shared
    # mutable state on disk (audited: tempdirs only, no fixed paths), so they
    # run concurrently.
    phase_started = time.perf_counter()
    reports: dict[Path, FileReport] = {path: FileReport(path) for path in files}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one_file, python, path): path for path in files}
        for future in as_completed(futures):
            path = futures[future]
            try:
                reports[path] = future.result()
            except Exception as exc:  # noqa: BLE001 - report, do not lose the file
                reports[path].problems.append(f"worker crashed: {exc!r}")
    phase_seconds = time.perf_counter() - phase_started

    print(f"== pass 1/{total_passes}: as a script ==", flush=True)
    for path in files:
        report = reports[path]
        assert report.script is not None
        state = "ok" if report.script.returncode == 0 else f"rc={report.script.returncode}"
        if report.script.quiet:
            state += " SILENT"
        print(f"  {state:>16}  {report.rel}")

    print(f"\n== pass 2/{total_passes}: pytest, one file per invocation ==", flush=True)
    for path in files:
        report = reports[path]
        assert report.isolated is not None
        state = "ok" if report.isolated.returncode == 0 else f"rc={report.isolated.returncode}"
        print(f"  {state:>16}  {report.rel}  ({report.isolated_tests} tests)")
    print(f"  (passes 1+2 wall: {phase_seconds:.1f}s, {jobs} parallel workers)")

    # Pass 3 -- the whole suite in one pytest invocation, plus a collection-only
    # pass over the same file set for the order-dependence comparison.  Skipped
    # in --gate mode: its whole point is every file in ONE process, which
    # neither parallelism nor a subset can substitute for.
    batch: Outcome | None = None
    if not args.gate:
        print(f"\n== pass 3/{total_passes}: pytest, whole suite in one invocation ==", flush=True)
        batch = _run(
            [str(python), "-m", "pytest", "-q", *map(str, files)], REPO
        )
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
        if batch is not None and report.isolated_tests != report.batch_tests:
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

    # ---- the whole-suite verdict is part of the gate ----------------------
    #
    # Pass 3 is the only pass that can observe cross-file interference, which is
    # the defect this runner exists to eliminate, so its result cannot be printed
    # and then discarded: a batch that is red while every file is green *alone* is
    # exactly the order-dependent failure, and it used to report PASS.  Recorded as
    # its own verdict rather than folded into a file, because in that case no single
    # file is at fault and blaming one would send the reader to the wrong place.
    batch_problems: list[str] = []
    if batch is not None and batch.returncode != 0:
        green_in_isolation = all(reports[p].green for p in files)
        batch_problems.append(
            f"whole-suite pytest is red (rc={batch.returncode})"
            + (
                " while every file is green in isolation: cross-file interference"
                " (module-level state, sys.path, or sys.modules shadowing)"
                if green_in_isolation
                else ""
            )
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
    slowest = sorted(reports.values(), key=lambda r: r.seconds, reverse=True)[:3]
    print(
        "slowest files: "
        + ", ".join(f"{r.rel} ({r.seconds:.0f}s)" for r in slowest)
    )
    if batch is not None:
        print(
            f"whole-suite pytest: {_parse_summary(batch.output) or f'rc={batch.returncode}'}"
        )
    else:
        print(
            "whole-suite pytest: SKIPPED (--gate; run the full mode for the "
            "cross-file interference check)"
        )
    # ---- the torch-mlir model gate ---------------------------------------
    #
    # Supervised the same way the per-file passes are: a child process, this
    # time the runner itself with ``--torch-mlir``.  It runs after the contract
    # suite rather than alongside it, because the model gate is the newest and
    # the slowest phase (~40 s) and the cheap structural complaints should reach
    # the reader first.  It is skipped for ``--files``: an explicit subset must
    # not be silently widened by the gate that happens to be on by default.
    model_gate: Outcome | None = None
    if not args.files:
        model_gate = _run([str(python), str(_HERE), "--torch-mlir"], REPO)
        print("\n== torch-mlir model gate: host compile only, never a device ==")
        print(model_gate.output.rstrip() or "(no output)")
        print()

    model_problems: list[str] = []
    if model_gate is not None and model_gate.returncode != 0:
        model_problems.append(
            f"torch-mlir model gate is red (rc={model_gate.returncode})"
        )

    if failures or batch_problems:
        if failures:
            print(
                f"\nFAILED: {failures} file(s) are not green in both styles",
                file=sys.stderr,
            )
            for report in reports.values():
                for problem in report.problems:
                    print(f"  {report.rel}: {problem}", file=sys.stderr)
        for problem in batch_problems:
            print(f"  FAIL  {problem}", file=sys.stderr)
        if batch_problems:
            print(
                f"\nFAILED: {len(batch_problems)} whole-suite problem(s)",
                file=sys.stderr,
            )
        return 1
    for problem in model_problems:
        print(f"  FAIL  {problem}", file=sys.stderr)
    if model_problems:
        print(
            f"\nFAILED: {len(model_problems)} model gate problem(s)",
            file=sys.stderr,
        )
        return 1
    print(f"PASS  (total wall: {time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
