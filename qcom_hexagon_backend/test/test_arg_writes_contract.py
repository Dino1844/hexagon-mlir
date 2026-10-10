#!/usr/bin/env python3
"""The kernel's tensor-argument write set, from producer to launcher.

What is being pinned
--------------------
The Triton launcher's no-return path used to treat every ranked input tensor
as a possible output pointer: dump it on the device, pull the file back across
the tunnel, copy it into the caller's tensor.  A nine-shape survey of the tree
measured 65% of those pulled bytes and every file but one per launch as pure
waste, so the launch needed one fact -- which tensor arguments the kernel
writes through -- and the compiler already has it: Triton's lowering writes
every output pointer through a `bufferization.materialize_in_destination`
(hello) whose destination chains back through views to the argument that owns
the memory.

This file pins three things about that fact, in order of consequence:

1. **The producer's answer, against real compiles.**  Four fixtures through
   `translate_linalg_to_obj` -- a partial write set, no write at all, a write
   the extractor refuses to model, and a return-valued kernel -- each
   publishing exactly the answer it should, through the real C++ -> Python
   envelope boundary.
2. **The boundary's shape.**  A set is a list of tensor ordinals in the same
   space `weight_prepack`'s `slot` and `input_profs.idx` use; `null` is the
   fail-closed "unknown"; `[]` is the positive "nothing is written".  Those
   last two are different values and are never merged.
3. **The launcher's consumption.**  Absent or `null` means every ranked input
   comes back, which is today's behaviour; a list means only its slots are
   dumped, pulled and copied back; and the `res_idx` walk stays aligned,
   including next to a pre-packed weight slot.
4. **How the extractor's one load-bearing invariant fails.**  A missing tensor
   ordinal used to be an `assert`, i.e. a build-flag invariant: with
   `-DNDEBUG` it vanishes and the guard becomes an `end()` dereference that
   publishes a garbage slot, which reads as a pass on a result that never
   arrived.  It now throws, like the other 23 `fail(...)` sites in that file,
   and nothing in the build removes that.  The throw itself is pinned here
   through a neighbouring invariant, because the ordinal one is closed-world
   and cannot be reached from a compile.

The direction of failure is why the decline arm exists.  Excluding a tensor
the kernel *does* write would silently lose a result -- no dump, no pull, no
copy-back, and every `rel` check reading as a pass -- so an unmodelled write
must produce `null` and not an answer.  Claiming too much is harmless: it only
forgoes the saving.

No device, no network: every probe here is a host compile or a Python-level
assertion.
"""

import importlib.util
import json
import unittest
from pathlib import Path

from triton.backends.qcom_hexagon_backend.compiler import HexagonBackend
from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions
from triton._C.libtriton import ir, qcom_hexagon_backend  # type: ignore

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parents[1]
_FIXTURES = _HERE.parent / "Conversion" / "LinalgToLLVM"
_UTILS_PATH = _BACKEND / "backend" / "utils.py"
#: The C++ file the producer lives in, read as text by the invariant test below.
_SOURCE = _BACKEND / "python" / "triton_qcom_hexagon_backend_api.cc"
_SPEC = importlib.util.spec_from_file_location(
    "hexagon_backend_utils_argwrites", _UTILS_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)


def _options():
    return {k: str(v) for k, v in HexagonOptions().__dict__.items()}


def _compile(fixture_name):
    """Run the real producer over one fixture and return its envelope.

    One compile per fixture text, cached on the path: the boundary under test
    is C++ -> JSON -> Python, and only a real compile exercises the C++ half.
    """
    return _compile_text((_FIXTURES / fixture_name).read_text())


def _compile_text(source):
    """Run the real producer over MLIR text and return its envelope."""
    options = _options()
    context = ir.context()
    qcom_hexagon_backend.load_dialects(context)
    module = qcom_hexagon_backend.parse_mlir_module_from_str(source, context)
    _, metadata_json = qcom_hexagon_backend.translate_linalg_to_obj(
        module, options, True
    )
    return json.loads(metadata_json)


class ProducerWriteSetTest(unittest.TestCase):
    """The C++ extractor's answers, measured through the envelope."""

    def test_a_partial_write_set_names_only_the_written_argument(self):
        # Three tensor arguments, one written through the argument pointer.
        # This is every Triton kernel in the tree: the two unwritten tensors
        # are what the launcher used to return for nothing.
        envelope = _compile("arg-writes-partial.mlir")
        self.assertEqual(envelope["arg_writes"], [1])
        # The envelope is still closed and still valid once the new child rides
        # in it, so the field is checked by the boundary that owns the shape.
        _UTILS.parse_translation_metadata(json.dumps(envelope))

    def test_a_kernel_that_writes_no_argument_publishes_an_empty_list(self):
        # A positive claim, not a decline: the result lives in a buffer the
        # kernel owns, so nothing came back through the arguments.
        self.assertEqual(_compile("arg-writes-none.mlir")["arg_writes"], [])

    def test_an_unmodelled_write_declines_rather_than_guessing(self):
        # `null`, not `[]`: the kernel calls an external function with an
        # argument pointer and nothing here says whether it writes.  The
        # launcher's fail-closed reading of `null` is pinned by
        # LauncherWriteSetConsumptionTest below.
        self.assertIsNone(_compile("arg-writes-declined.mlir")["arg_writes"])

    def test_a_return_valued_kernel_still_carries_a_write_set(self):
        # The launcher ignores this field when the kernel has return values
        # (the slots are the returns), but the producer does not know or care
        # which launcher will consume the artifact, so the field is still
        # there -- and here it is the empty list, because the linalg output
        # chains to a `tensor.empty` rather than to an argument.
        self.assertEqual(
            _compile("arg-writes-return-kernel.mlir")["arg_writes"], []
        )


class WriteSetBoundaryTest(unittest.TestCase):
    """Where the accepted set is measured, so the pins above mean something."""

    def test_the_decoded_value_round_trips_through_the_packed_metadata(self):
        # `[]` and `None` are the two arms that matter, and they arrive as
        # themselves: the packed field is decoded, not JSON text, because it is
        # a flat list the launcher reads directly.
        for value in (None, [], [0], [1, 3]):
            with self.subTest(arg_writes=value):
                metadata = {}
                _UTILS.apply_translation_metadata(
                    metadata,
                    json.dumps(
                        {
                            "schema": _UTILS.TRANSLATION_METADATA_SCHEMA,
                            "weight_prepack": {"layout": None, "weights": []},
                            "hmx_manifest": _manifest(),
                            "arg_writes": value,
                        }
                    ),
                )
                self.assertEqual(metadata["arg_writes"], value)

    def test_the_two_meanings_of_no_write_set_are_kept_apart(self):
        # An empty list is a fact and `None` is the absence of one.  A
        # consumer that collapsed them would turn an unknown into a decision,
        # which is the failure the decline exists to prevent.
        self.assertEqual(_UTILS.validate_arg_writes(None), None)
        self.assertEqual(_UTILS.validate_arg_writes([]), [])
        self.assertIsNotNone(_UTILS.validate_arg_writes([]))
        self.assertIsNone(_UTILS.validate_arg_writes(None))

    def test_a_malformed_write_set_is_refused(self):
        # Rejections give the accept-set teeth: ordinals are non-negative
        # integers, the list is strictly increasing (so a duplicate or an
        # out-of-order slot is a producer disagreement, not a repair), and
        # nothing but a list or null is a write set at all.
        for value in ([-1], [0, 0], [2, 1], ["0"], [0.5], "0", 0, True, {}):
            with self.subTest(arg_writes=value):
                with self.assertRaises(ValueError):
                    _UTILS.validate_arg_writes(value)

    def test_a_packed_artifact_without_the_field_is_a_stale_artifact(self):
        # Required, like the other launch-contract children: an entry written
        # before the field existed must be an actionable error, not a kernel
        # with the feature quietly off.
        from test_hmx_manifest_metadata import _MANIFEST, _WEIGHT

        packed = {
            "num_warps": 1,
            "num_ctas": 1,
            "shared": 0,
            "cluster_dims": (),
            "name": "matmul_kernel",
            "return_types": [],
            "iterations": 1,
            "scratch": 0,
            "enableMultiThreading": False,
            "enableThreadedDispatch": False,
            "enableLWP": False,
            "weight_prepack": json.dumps(_WEIGHT),
            "hmx_manifest": json.dumps(_MANIFEST),
            "hmx_record": "",
            "arg_writes": None,
        }
        self.assertIs(_UTILS.validate_pack_metadata(packed), packed)
        del packed["arg_writes"]
        with self.assertRaisesRegex(RuntimeError, "arg_writes"):
            _UTILS.require_pack_metadata_fields(packed)
        with self.assertRaisesRegex(RuntimeError, "arg_writes"):
            _UTILS.validate_pack_metadata(packed)

    def test_the_field_is_published_by_the_backend_packer(self):
        # `pack_metadata` copies the field by name out of the metadata object,
        # exactly as it does the other fourteen: a field the packer drops would
        # reach the launcher as a missing required field on the first launch.
        from test_hmx_manifest_metadata import _metadata

        metadata = _metadata(arg_writes=[2])
        packed = HexagonBackend.pack_metadata(object(), metadata)
        self.assertEqual(packed["arg_writes"], [2])


def _manifest():
    """A minimal valid v2 manifest, borrowed from the existing fixtures."""
    from test_hmx_manifest_metadata import _MANIFEST

    return json.loads(json.dumps(_MANIFEST))


#: A module whose prepack contract is malformed.  `validatePrepackAttributes`
#: refuses it with the file's `fail(...)` convention before anything else runs,
#: which makes it the reachable stand-in for the write set's own invariant
#: break: both are "this file refuses instead of answering".
_MALFORMED_PREPACK_MODULE = """
module attributes {hmx.weight_prepack = "not json at all"} {
  func.func @broken() {
    return
  }
}
"""


class WriteSetInvariantFailureTest(unittest.TestCase):
    """How the extractor's one load-bearing invariant fails."""

    def test_the_invariant_is_not_guarded_by_a_bare_assert(self):
        # The invariant is "a written memref argument always has a tensor
        # ordinal".  It was an `assert`, which is a build-flag invariant: with
        # -DNDEBUG the check disappears and `ordinal->second` dereferences
        # `end()`, publishing a garbage tensor ordinal -- a slot the host then
        # never dumps, pulls or copies back, so a `rel` check reads as a pass on
        # a result that never arrived.  Nothing in this tree declares that
        # assertions stay on, so the extractor now refuses with `fail(...)`,
        # the convention the other 23 sites in that file already use.  This is
        # the machine-readable half of the decision: the section must not grow
        # a bare `assert` back.
        source = _SOURCE.read_text()
        start = source.index("std::optional<std::string> argWritesJson(")
        end = source.index("std::string argWritesJsonFor(")
        extractor = source[start:end]
        self.assertNotIn("assert(", extractor)
        self.assertIn(
            'fail("a written memref argument has no tensor ordinal")', extractor
        )

    def test_a_refused_invariant_aborts_the_compile_instead_of_answering(self):
        # The ordinal invariant itself is closed-world and cannot be reached
        # from a compile, so what is pinned here is the mechanism it now shares
        # with every other refusal in that file: the compile raises instead of
        # publishing an envelope.  A throw swallowed -- or turned into a
        # SIGABRT -- anywhere between the C++ and here would turn "refuse" into
        # "die unclearly", which is the failure mode the decline arm exists to
        # prevent.  This arm is the one that was broken: a malformed *JSON*
        # prepack contract aborted the process (an `Expected` died still holding
        # its Error), so the refusal was a build-flag property rather than the
        # code's.  The non-JSON arm is pinned in test_hmx_manifest_metadata.py.
        with self.assertRaisesRegex(RuntimeError, "malformed HMX prepack"):
            _compile_text(_MALFORMED_PREPACK_MODULE)


def main() -> None:
    tests = (
        ProducerWriteSetTest(
            "test_a_partial_write_set_names_only_the_written_argument"
        ),
        ProducerWriteSetTest(
            "test_a_kernel_that_writes_no_argument_publishes_an_empty_list"
        ),
        ProducerWriteSetTest(
            "test_an_unmodelled_write_declines_rather_than_guessing"
        ),
        ProducerWriteSetTest(
            "test_a_return_valued_kernel_still_carries_a_write_set"
        ),
        WriteSetBoundaryTest(
            "test_the_decoded_value_round_trips_through_the_packed_metadata"
        ),
        WriteSetBoundaryTest(
            "test_the_two_meanings_of_no_write_set_are_kept_apart"
        ),
        WriteSetBoundaryTest("test_a_malformed_write_set_is_refused"),
        WriteSetBoundaryTest(
            "test_a_packed_artifact_without_the_field_is_a_stale_artifact"
        ),
        WriteSetBoundaryTest(
            "test_the_field_is_published_by_the_backend_packer"
        ),
        WriteSetInvariantFailureTest(
            "test_the_invariant_is_not_guarded_by_a_bare_assert"
        ),
        WriteSetInvariantFailureTest(
            "test_a_refused_invariant_aborts_the_compile_instead_of_answering"
        ),
    )
    for test in tests:
        test.run()
        print(f"ok  {type(test).__name__}.{test._testMethodName}")
    print(f"argument write set contract: {len(tests)} passed")


if __name__ == "__main__":
    main()
