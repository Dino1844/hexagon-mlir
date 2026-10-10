#!/usr/bin/env python3
"""The output-pull launch intent, as bare-assert pytest tests.

What is being pinned
--------------------
A launch pays for its output bytes twice as much as for anything else: the
tunnel runs at roughly 0.2-0.6 MB/s in *both* directions, and a steady-state
benchmark reads only the ``Perf`` line.  ``launch_intent.pull_outputs(False)``
lets such a caller skip the pull.

The number this file protects is not the saving, it is the **boundary**: the
skip must reach the pull command and nothing else.  Three properties, each
checked against the real code:

1. The default is "pull".  A caller that declares nothing must get today's
   behaviour, because every ``rel`` check and every value test in the tree
   depends on it.
2. The skip is total for the output pull and invisible to everything else:
   ``perf.txt`` is still pulled, the device-side dump still happens (the
   wrapper writes the files), the pre-run ``rm -rf`` still clears them.
3. The skipped pull is observable, not silent.  A launch that simply cost less
   with no reason printed is how a protocol claim goes unverifiable.

``main()`` below exists so that running this file as a script is a real gate
rather than a silent no-op; its siblings carry the same guard.

No device, no network, no SDK: the toolchain paths the executor checks are
synthesised in a temporary directory.
"""

import collections
import contextlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from triton.backends.qcom_hexagon_backend.hexagon_executor import (
    HexagonExecutor,
)
from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    HexagonLauncherBase,
    HexagonWrapperGenerator,
    WrapperGeneratorStrings,
)
from triton.backends.qcom_hexagon_backend.launch_intent import (
    pull_outputs,
    pull_outputs_enabled,
)
from triton.backends.qcom_hexagon_backend.utils import profile_triton_inputs

#: The manifest/weight fixtures the launch path validates against.  Reused
#: rather than re-declared so this file cannot drift from the contract the
#: driver already enforces.
from test_hmx_manifest_metadata import _MANIFEST, _WEIGHT

#: One value that means "this launch contract has no `arg_writes` field at
#: all".  It is distinct from ``None``, which is a *published* field whose
#: value is "unknown": both must produce the same launcher behaviour (return
#: every ranked input), but a fixture that cannot tell them apart could not
#: prove that, and a future field default that differs between them would pass
#: anyway.
_ABSENT = object()


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class _SubprocessRecorder:
    """Stands in for the ``subprocess`` module inside the executor.

    Every command the executor would have handed to a shell is recorded
    verbatim and answered with an empty success, which is enough: the executor
    parses nothing from a pull (the bytes land as files) and parses nothing
    from ``ls`` when there are no dumps.
    """

    def __init__(self):
        self.commands = []

    def check_output(self, command, **kwargs):
        self.commands.append(command)
        return b""

    def run(self, command, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def pulls(self):
        return [c for c in self.commands if " pull " in c]

    def CalledProcessError(self):  # pragma: no cover - never raised here
        raise AssertionError("unreachable")


class _ExecutorStub:
    """The half of ``HexagonExecutor`` the launcher touches."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_Executable_Path(self):
        return ""

    def generate_shared_object(self, *_args, **_kwargs):
        return "/tmp/pull-policy-stub.so"


class _GeneratorStub:
    def __init__(
        self,
        input_profs,
        iterations,
        func_name,
        output_profs,
        grid,
        options,
        scope,
        arg_writes=_ABSENT,
    ):
        self.input_profs = input_profs
        self.output_profs = output_profs
        # Only set when the launch contract published one.  The generator reads
        # the field with getattr(..., None), so an unset attribute is the
        # "no write set" arm and must stay observable as such: a fixture that
        # always defined it would hide the fail-closed default.
        if arg_writes is not _ABSENT:
            self.arg_writes = arg_writes

    def generate_cpp_wrapper(self, _file_name, _exec_dir):
        return "// stub wrapper"


class _WrapperStub:
    """The generator-side facts ``generate_input_output_paths`` reads."""

    input_profs = []
    output_profs = []
    weight_prepack = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fake_toolchain(root: Path) -> tuple[str, str]:
    """Create the two libraries ``run_kernel_on_device`` insists exist."""
    tools = root / "HEXAGON_Tools"
    libdir = tools / "target/hexagon/lib/v79/G0/pic"
    libdir.mkdir(parents=True)
    (libdir / "libc++.so.1").write_bytes(b"")
    (libdir / "libc++abi.so.1").write_bytes(b"")
    return str(tools), "79"


def _executor(root: Path):
    """A real ``HexagonExecutor`` with every external path synthesised."""
    tools, q6 = _fake_toolchain(root)
    executor = object.__new__(HexagonExecutor)
    executor.exec_mode = "device"
    executor.device_path = str(root / "device")
    executor.lib_path = str(root / "device/lib")
    executor.alt_perf_path = None
    executor.enable_lwp = False
    executor.enable_etm = False
    executor.final_result = "Pass"
    executor.cleanup_device_post_exec = True
    config = collections.namedtuple("config", ("env_vars", "HEX_TOOLS", "Q6_VERSION"))
    executor.config = config(
        env_vars={
            "ANDROID_HOST": "",
            "ANDROID_SERIAL": "pull-policy-serial",
            "HEXAGON_MLIR_ROOT": str(root),
            "HEXAGON_SDK_ROOT": str(root / "Hexagon_SDK"),
            "HEXAGON_TOOLS": tools,
        },
        HEX_TOOLS={},
        Q6_VERSION=q6,
    )
    return executor


def _launcher_module():
    import triton.backends.qcom_hexagon_backend.triton_hexagon_launcher as module

    return module


def _launcher_instance():
    """A real launcher with only ``__init__`` skipped, as the driver builds it."""
    import triton.backends.qcom_hexagon_backend.triton_hexagon_launcher as module

    return object.__new__(module.TritonHexagonLauncher)


def _launch(
    launcher,
    pull=True,
    *,
    inputs=None,
    arg_writes=_ABSENT,
    results=None,
    weight_prepack=None,
):
    """One ``_exec_kernel`` call with the surrounding pieces stubbed out.

    ``HexagonLauncherBase.execute_kernel`` and the executor classes are replaced
    by recorders, so what is observed is the *plumbing*: whether the intent
    reached the call and whether the result came back.

    ``inputs``/``results`` are the launch's tensors and the bytes the executor
    hands back; ``arg_writes`` is the launch contract's write set (``_ABSENT``
    for a metadata object that has no such field).
    """
    module = _launcher_module()
    original_executor = module.HexagonExecutor
    original_generator = module.TritonHexagonWrapperGenerator
    original_execute = HexagonLauncherBase.execute_kernel
    seen = {}
    inputs = [torch.zeros(4)] if inputs is None else inputs
    results = [torch.zeros(4) for _ in inputs] if results is None else results

    def execute(self, hexec, local_dir, filename, libs, wrapper_generator, pull_outputs=True):
        seen["pull_outputs"] = pull_outputs
        return [] if not pull_outputs else results

    module.HexagonExecutor = _ExecutorStub
    module.TritonHexagonWrapperGenerator = (
        lambda *args, **kwargs: _GeneratorStub(
            *args, arg_writes=arg_writes, **kwargs
        )
    )
    HexagonLauncherBase.execute_kernel = execute
    launcher.generate_and_dump_wrapper = lambda *_a, **_k: "/tmp/pull-policy-wrapper.cpp"
    # Artifacts (the .o the launcher writes) land in a scratch dir that is
    # removed with the test, not in /tmp.
    with tempfile.TemporaryDirectory() as scratch:
        previous_dump_dir = os.environ.get("HEXAGON_MLIR_DUMP_DIR")
        os.environ["HEXAGON_MLIR_DUMP_DIR"] = scratch
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                launch_results = launcher._exec_kernel(
                    b"kernel-object-bytes",
                    1,
                    "pull_policy_probe",
                    inputs,
                    [],
                    (1, 1, 1),
                    weight_prepack=(
                        json.dumps(_WEIGHT)
                        if weight_prepack is None
                        else json.dumps(weight_prepack)
                    ),
                    hmx_manifest=json.dumps(_MANIFEST),
                    arg_writes=None if arg_writes is _ABSENT else arg_writes,
                )
        finally:
            if previous_dump_dir is None:
                os.environ.pop("HEXAGON_MLIR_DUMP_DIR", None)
            else:
                os.environ["HEXAGON_MLIR_DUMP_DIR"] = previous_dump_dir
            module.HexagonExecutor = original_executor
            module.TritonHexagonWrapperGenerator = original_generator
            HexagonLauncherBase.execute_kernel = original_execute

    # The caller's tensors are the evidence for the copy-back half: each is
    # written only when the slot it covers came back.
    return seen, launch_results


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_the_intent_is_scoped_and_restored():
    assert pull_outputs_enabled() is True, "a caller that declares nothing must pull"
    with pull_outputs(False):
        assert pull_outputs_enabled() is False
    assert pull_outputs_enabled() is True, "the scope must close"
    # An exception must not leak the declaration: a failing benchmark is still
    # a benchmark, and the next launch in the same session is somebody's rel
    # check.
    try:
        with pull_outputs(False):
            raise RuntimeError("benchmark exploded")
    except RuntimeError:
        pass
    assert pull_outputs_enabled() is True, "an exception must not leave the intent set"


def test_the_executor_skips_the_output_pull_and_nothing_else():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        executor = _executor(root)
        so_path = root / "kernel.so"
        so_path.write_bytes(b"")
        outputs = [str(root / f"kernel_o{i}.raw") for i in range(2)]
        inputs = [str(root / "kernel_t0.raw")]
        for p in inputs + outputs:
            Path(p).write_bytes(b"")

        import triton.backends.qcom_hexagon_backend.hexagon_executor as hexec_mod

        def launch(pull_outputs):
            """One launch with every shell command recorded and stdout kept."""
            recorder = _SubprocessRecorder()
            spoken = io.StringIO()
            original = hexec_mod.subprocess
            hexec_mod.subprocess = recorder
            try:
                with contextlib.redirect_stdout(spoken):
                    hexec_mod.HexagonExecutor.run_kernel_on_device(
                        executor,
                        [str(so_path)],
                        inputs,
                        outputs,
                        generatePerf=True,
                        pull_outputs=pull_outputs,
                    )
            finally:
                hexec_mod.subprocess = original
            return recorder.commands, spoken.getvalue()

        commands, spoken = launch(True)
        pulls = [c for c in commands if " pull " in c]
        assert [c for c in pulls if "kernel_o0.raw" in c], pulls
        assert [c for c in pulls if "kernel_o1.raw" in c], "every output, not just the first"
        assert any("perf.txt" in c for c in pulls), "perf is not part of the skip"
        assert "output pull skipped" not in spoken
        # The staleness guard: each launch still clears the previous run's
        # outputs before running, so a later pull can never return them.
        assert [c for c in commands if "rm -rf" in c and "kernel_o0.raw" in c]

        commands, spoken = launch(False)
        pulls = [c for c in commands if " pull " in c]
        assert not [c for c in pulls if "kernel_o0.raw" in c], pulls
        assert not [c for c in pulls if "kernel_o1.raw" in c], pulls
        assert any("perf.txt" in c for c in pulls), "the timing must survive"
        assert [c for c in commands if "rm -rf" in c and "kernel_o0.raw" in c], (
            "the staleness guard must survive the skip"
        )
        assert any("kernel.so" in c for c in commands), "the library handling is untouched"
        assert "output pull skipped" in spoken, "the skip must be visible, not silent"


def test_a_skipped_pull_returns_nothing_rather_than_stale_bytes():
    """``run()`` must not read files it was told not to pull.

    The output paths here deliberately do not exist: the collecting half raises
    on them when it runs, and returns an empty list when it does not.  That
    asymmetry is the assertion -- a partial or stale tensor instead of an error
    is the failure this ordering prevents.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        executor = _executor(root)
        so_path = root / "kernel.so"
        so_path.write_bytes(b"")
        missing = [str(root / "kernel_o0.raw")]
        calls = []

        def fake_device(self, paths, inputs, outputs, generatePerf=False, pull_outputs=True):
            calls.append(pull_outputs)
            return []

        original = HexagonExecutor.run_kernel_on_device
        HexagonExecutor.run_kernel_on_device = fake_device
        try:
            try:
                executor.run([str(so_path)], [], missing, generatePerf=True)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError(
                    "the default still collects the outputs -- and these do not exist"
                )
            assert (
                executor.run(
                    [str(so_path)], [], missing, generatePerf=True, pull_outputs=False
                )
                == []
            )
        finally:
            HexagonExecutor.run_kernel_on_device = original
        assert calls == [True, False]


def test_execute_kernel_forwards_the_declaration():
    import triton.backends.qcom_hexagon_backend.triton_hexagon_launcher as module

    class _Executor:
        def __init__(self):
            self.seen = None

        def run(self, paths, inputs, outputs, generatePerf=False, pull_outputs=True):
            self.seen = pull_outputs
            return [] if not pull_outputs else [torch.zeros(1)]

    launcher = object.__new__(module.TritonHexagonLauncher)
    with tempfile.TemporaryDirectory() as tmp:
        for declared, expected in ((None, True), (False, False), (True, True)):
            executor = _Executor()
            kwargs = {} if declared is None else {"pull_outputs": declared}
            launcher.execute_kernel(
                executor, tmp, "kernel", ["/tmp/k.so"], _WrapperStub(), **kwargs
            )
            assert executor.seen is expected, declared


def test_the_launcher_reads_the_intent_and_skips_the_copy_back():
    import triton.backends.qcom_hexagon_backend.triton_hexagon_launcher as module

    launcher = object.__new__(module.TritonHexagonLauncher)

    seen, results = _launch(launcher)
    assert seen["pull_outputs"] is True
    assert len(results) == 1, "the default round-trips the tensors"

    seen, results = _launch(launcher)
    assert seen["pull_outputs"] is True, "the previous scope must not leak"

    with pull_outputs(False):
        seen, results = _launch(launcher)
        assert seen["pull_outputs"] is False, "the declaration must reach the call"
        assert results == [], "no tensors, so nothing to copy back"


# ---------------------------------------------------------------------------
# The argument write set (`arg_writes`)
# ---------------------------------------------------------------------------


def _wrapper(inputs, *, arg_writes=_ABSENT, weight_prepack=None):
    """A real generator over real input profiles, with the launch facts set.

    The generator is the one class that reads the write set for the device-side
    dump list, so a stub there would stub out the thing under test.
    """
    wrapper = HexagonWrapperGenerator(
        profile_triton_inputs(list(inputs)),
        1,
        "pull_policy_probe",
        [],
        WrapperGeneratorStrings(),
        {"enableLWP": False},
    )
    if arg_writes is not _ABSENT:
        wrapper.arg_writes = arg_writes
    wrapper.weight_prepack = weight_prepack
    return wrapper


def _dumped_slots(wrapper):
    """The `_o{i}.raw` files the generated wrapper writes, in order."""
    calls = wrapper.generate_tensor_write_to_file_calls("kernel", "/exec")
    return [int(name) for name in re.findall(r"_o(\d+)\.raw", calls)]


def _pull_paths(wrapper):
    """The host-side paths `generate_input_output_paths` hands the executor."""
    launcher = HexagonLauncherBase()
    with tempfile.TemporaryDirectory() as scratch:
        _, output_paths = launcher.generate_input_output_paths(
            scratch, "kernel", wrapper
        )
    return [os.path.basename(path) for path in output_paths]


def test_a_launch_without_a_write_set_returns_every_ranked_input():
    # Today's behaviour, and the fail-closed default: a metadata object that
    # has no `arg_writes` at all and one that publishes `null` must behave the
    # same, because both mean "the producer could not prove a write set".
    inputs = [torch.zeros(4) for _ in range(3)]
    for arg_writes in (_ABSENT, None):
        # Both arms mean "the producer could not prove a write set": the field
        # is missing, or its value is `null`.  They must behave identically.
        print(f"  arm arg_writes={arg_writes!r}")
        wrapper = _wrapper(inputs, arg_writes=arg_writes)
        assert _dumped_slots(wrapper) == [0, 1, 2], "every ranked input is dumped"
        assert _pull_paths(wrapper) == [
            "kernel_o0.raw",
            "kernel_o1.raw",
            "kernel_o2.raw",
        ]


def test_a_write_set_prunes_the_dump_the_pull_and_the_copy_back():
    # One kernel, three tensors, one written through its pointer: the two
    # unwritten tensors cost a dump, a pull and a copy-back today and nothing
    # after this.
    inputs = [torch.zeros(4) for _ in range(3)]
    wrapper = _wrapper(inputs, arg_writes=[1])
    assert _dumped_slots(wrapper) == [1], "the device writes only the written tensor"
    # The slot is kept as the file's own index, not renumbered: the host pulls
    # `_o1.raw`, and the wrapper reads it back into tensor 1.
    assert _pull_paths(wrapper) == ["kernel_o1.raw"]

    live = [t.clone() for t in inputs]
    _launch(
        _launcher_instance(),
        inputs=live,
        arg_writes=[1],
        results=[torch.full((4,), 7.0)],
    )
    # Only the written tensor changed; the other two are exactly what the host
    # pushed for them, which is the whole point of not pulling them.
    assert live[0].sum() == 0.0, "an unwritten input is left alone"
    assert live[1].sum() == 28.0, "the written tensor gets the pulled bytes"
    assert live[2].sum() == 0.0, "an unwritten input is left alone"


def test_a_skipped_slot_does_not_advance_the_result_index():
    # The alignment trap, in the form that matters: a write set that prunes a
    # slot must not let the copy-back walk consume that slot's position.  Here
    # the write set is {1, 2} with three tensors, so the first pulled result
    # belongs to tensor 1 and the second to tensor 2 -- and never to tensor 0.
    inputs = [torch.zeros(4) for _ in range(3)]
    live = [t.clone() for t in inputs]
    _launch(
        _launcher_instance(),
        inputs=live,
        arg_writes=[1, 2],
        results=[torch.full((4,), 1.0), torch.full((4,), 2.0)],
    )
    assert live[0].sum() == 0.0, "the pruned slot owns no result"
    assert live[1].sum() == 4.0, "the first pulled result is tensor 1's"
    assert live[2].sum() == 8.0, "the second pulled result is tensor 2's"


def test_a_pre_packed_slot_still_advances_the_result_index():
    # The two skip rules side by side, because they differ in exactly this one
    # respect and mixing them up writes one tensor's bytes into another:
    # a pre-packed slot is dumped and pulled (its crouton bytes come back) but
    # never copied back, so it DOES advance; a slot the kernel never writes is
    # neither dumped nor pulled, so it does not.
    #
    # Here slot 0 is pre-packed and written, slot 1 is written and ordinary.
    # One result comes back per pulled slot, in slot order, so the first
    # belongs to slot 0 -- and is discarded -- and the second to slot 1.
    weight = {
        "layout": {
            "ndims": 5,
            "results": [[[1, 32], [2, 2], [4, 1]], [[0, 32], [3, 1]]],
        },
        "weights": [
            {
                "func": "pull_policy_probe",
                "slot": 0,
                "shape": [4, 4],
                "crouton": [1, 1, 16, 32, 2],
                "dtype": "f16",
                "location": "vtcm",
            }
        ],
    }
    inputs = [torch.zeros(4), torch.zeros(4)]
    live = [t.clone() for t in inputs]
    _launch(
        _launcher_instance(),
        inputs=live,
        arg_writes=[0, 1],
        results=[torch.full((4,), 1.0), torch.full((4,), 2.0)],
        weight_prepack=weight,
    )
    # Slot 0's pulled bytes are the crouton image, so the copy-back is skipped
    # and the caller's row-major tensor survives; slot 1 is copied normally.
    assert live[0].sum() == 0.0, "a pre-packed slot is never copied back"
    assert live[1].sum() == 8.0, "the next slot gets the next result"

def main() -> None:
    tests = (
        test_the_intent_is_scoped_and_restored,
        test_the_executor_skips_the_output_pull_and_nothing_else,
        test_a_skipped_pull_returns_nothing_rather_than_stale_bytes,
        test_execute_kernel_forwards_the_declaration,
        test_the_launcher_reads_the_intent_and_skips_the_copy_back,
        test_a_launch_without_a_write_set_returns_every_ranked_input,
        test_a_write_set_prunes_the_dump_the_pull_and_the_copy_back,
        test_a_skipped_slot_does_not_advance_the_result_index,
        test_a_pre_packed_slot_still_advances_the_result_index,
    )
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"output-pull launch intent: {len(tests)} passed")


if __name__ == "__main__":
    main()
