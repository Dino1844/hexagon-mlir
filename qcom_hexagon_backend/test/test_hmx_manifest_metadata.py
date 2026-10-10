"""Host-only tests for the semantic HMX manifest consumer."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from triton.backends.qcom_hexagon_backend.compiler import HexagonBackend
from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions
from triton._C.libtriton import ir, qcom_hexagon_backend  # type: ignore


_HERE = Path(__file__).resolve()
_UTILS_PATH = _HERE.parents[1] / "backend" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("hexagon_backend_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)


def _dimension(value=None, *, symbol=None):
    if symbol is not None:
        return {"kind": "dynamic", "symbol": symbol}
    return {"kind": "static", "value": value}


def _logical(m=64, n=64, k=64):
    if m is None or n is None or k is None:
        return {
            "m": _dimension(symbol="m" if m is None else None, value=m),
            "n": _dimension(symbol="n" if n is None else None, value=n),
            "k": _dimension(symbol="k" if k is None else None, value=k),
        }
    return {"m": _dimension(m), "n": _dimension(n), "k": _dimension(k)}


def _execution(*, block_m=64, blocking=None, pipeline=None, counts=None):
    return {
        "blocking": blocking or ("whole" if block_m == 64 else "m_blocked"),
        "block_m": block_m,
        "pipeline": pipeline
        or {
            "requested": 0,
            "selected": "serial",
            "depth": 0,
            "reason": "shallow-k",
        },
        "bridge_counts": counts
        or {
            "pack_act_sites": 1,
            "pack_weight_sites": 0,
            "unpack_sites": 1,
            "count_semantics": "ir_sites",
        },
    }


def _hmx_record(
    *,
    record_id=0,
    function="matmul_kernel",
    plan="full-hmx",
    reason=None,
    shape=(64, 64, 64),
    weight_kind="compile-time-constant",
    policy=None,
    block_m=None,
    counts=None,
    workspace_class="runtime-internal",
    grid_policy=None,
):
    m, n, k = shape
    padded = {axis: ((value + 31) // 32) * 32 for axis, value in zip("mnk", shape)}
    full = {axis: (value // 32) * 32 for axis, value in zip("mnk", shape)}
    tail = {axis: value - full[axis] for axis, value in zip("mnk", shape)}
    if reason is None:
        reason = "selected-aligned" if plan == "full-hmx" else "selected-tail"
    if block_m is None:
        block_m = m
    if grid_policy is None:
        grid_policy = (
            "single-instance" if plan == "hmx-tail" else "legacy-runtime"
        )
    record = {
        "function": function,
        "id": record_id,
        "plan": plan,
        "reason": reason,
        "shape_state": "static",
        "logical": _logical(m, n, k),
        "dtypes": {
            "lhs": "f16",
            "rhs": "f16",
            "out": "f16",
            "crouton": "f16",
        },
        "padded": padded,
        "full": full,
        "tail": tail,
        "layout": "row-major-inner-contiguous",
        "workspace_class": workspace_class,
        "grid_policy": grid_policy,
        "vtcm_accounting": "bridge-only",
        "vtcm_budget_bytes": 8388608,
        "vtcm_before_bytes": 0,
        "vtcm_bridge_peak_bytes": 24576,
        "execution": _execution(
            block_m=block_m,
            blocking="whole" if block_m == m else "m_blocked",
            counts=counts,
        ),
        "weight_binding": {"kind": weight_kind},
    }
    if plan == "hmx-tail":
        record["tail_policy"] = {
            "k": "zero-pad-both-operands",
            "mn": "padded-edge-tile-bounded-store",
        }
    if weight_kind == "argument-slot":
        record["weight_binding"]["policy_ref"] = {
            "function": function,
            "slot": policy["slot"],
        }
    record["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(record, policy)
    return record


def _hvx_record(
    *,
    record_id=1,
    function="matmul_kernel",
    reason="tile-alignment",
    shape=(64, 64, 64),
    dtypes=None,
):
    if shape is None:
        logical = None
        shape_state = "unavailable"
    else:
        logical = _logical(*shape)
        static_count = sum(value is not None for value in shape)
        if static_count == 3:
            shape_state = "static"
        elif static_count == 0:
            shape_state = "dynamic"
        else:
            shape_state = "partially-dynamic"
    record = {
        "function": function,
        "id": record_id,
        "plan": "hvx",
        "reason": reason,
        "shape_state": shape_state,
        "logical": logical,
        "dtypes": dtypes
        or {"lhs": "f16", "rhs": "f16", "out": "f16"},
    }
    record["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(record)
    return record


def _manifest(records, policies=None):
    totals = {"pack_act_sites": 0, "pack_weight_sites": 0, "unpack_sites": 0}
    for record in records:
        if record["plan"] == "hvx":
            continue
        counts = record["execution"]["bridge_counts"]
        for field in totals:
            totals[field] += counts[field]
    return {
        "schema": _UTILS.HMX_MANIFEST_SCHEMA,
        "matmuls": copy.deepcopy(records),
        "weight_policies": copy.deepcopy(policies or []),
        **totals,
        "count_semantics": "ir_sites",
    }


_POLICY = {
    "function": "matmul_kernel",
    "slot": 1,
    "policy": "device-pack",
    "reason": "prepack-disabled",
    "consumers": [0],
}
_MANIFEST_RECORD = _hmx_record()
_MANIFEST = _manifest([_MANIFEST_RECORD])
_WEIGHT = {"layout": None, "weights": []}
_LAYOUT = {
    "ndims": 5,
    "results": [[[1, 32], [2, 2], [4, 1]], [[0, 32], [3, 1]]],
}
_VALID_WEIGHT = {
    "layout": _LAYOUT,
    "weights": [
        {
            "func": "matmul_kernel",
            "slot": 1,
            "shape": [64, 64],
            "crouton": [2, 2, 16, 32, 2],
            "dtype": "f16",
            # The placement fact the compiler publishes on every entry (P1 of
            # docs/hmx/pack-redundancy-fix-plan-2026-10-09.md): the image lives
            # in the VTCM pool here, in the permanent DDR mirror when the pool
            # cannot hold it. A fixture without it is a contract from before the
            # field existed, which is exactly what the consumer now refuses.
            "location": "vtcm",
        }
    ],
}


def _envelope(*, manifest=None, weight=None, arg_writes=None):
    return json.dumps(
        {
            "schema": _UTILS.TRANSLATION_METADATA_SCHEMA,
            "weight_prepack": copy.deepcopy(_WEIGHT if weight is None else weight),
            "hmx_manifest": copy.deepcopy(_MANIFEST if manifest is None else manifest),
            # The write-set child, `null` here as it is for a kernel whose
            # write set the producer declined to prove.
            "arg_writes": arg_writes,
        }
    )


def _packed_metadata(**overrides):
    values = {
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
        # A v1 envelope publishes no record child; the empty string is that one
        # spelling, and it is the only way to get an absent record.
        "hmx_record": "",
        # The launch contract's write set: None = the producer could not prove
        # one, which the launcher reads as "return every ranked input".
        "arg_writes": None,
    }
    values.update(overrides)
    return values


def _metadata(**overrides):
    return SimpleNamespace(**_packed_metadata(**overrides))


class TranslationMetadataTest(unittest.TestCase):
    def test_current_manifest_accepts_full_plan_and_nested_execution(self):
        self.assertIs(_UTILS.validate_hmx_manifest(_MANIFEST), _MANIFEST)
        self.assertEqual(_MANIFEST["matmuls"][0]["shape_state"], "static")
        self.assertNotIn("engine", _MANIFEST["matmuls"][0])
        self.assertEqual(
            _MANIFEST["matmuls"][0]["execution"]["bridge_counts"][
                "count_semantics"
            ],
            "ir_sites",
        )

    def test_launch_contract_gates_single_instance_records(self):
        tail = _manifest(
            [_hmx_record(plan="hmx-tail", shape=(33, 33, 33), block_m=33)]
        )
        tail_json = json.dumps(tail)
        self.assertEqual(
            _UTILS.enforce_hmx_launch_contract(tail_json, (1, 1, 1)), tail
        )
        with self.assertRaisesRegex(ValueError, "single program instance"):
            _UTILS.enforce_hmx_launch_contract(tail_json, (2, 1, 1))

        # A resident full-HMX workspace is keyed by the caller's flat program
        # id at runtime, so a grid>1 launch is sound and must pass the gate.
        resident = _manifest(
            [
                _hmx_record(
                    workspace_class="resident",
                )
            ]
        )
        self.assertEqual(
            _UTILS.enforce_hmx_launch_contract(
                json.dumps(resident), (1, 2, 1)
            ),
            resident,
        )

        # The legacy runtime-internal full plan deliberately retains the old
        # grid behavior; this gate must not silently turn it into single-instance.
        self.assertEqual(
            _UTILS.enforce_hmx_launch_contract(json.dumps(_MANIFEST), (2, 1, 1)),
            _MANIFEST,
        )

    def test_launch_contract_rejects_invalid_grid_shape(self):
        with self.assertRaisesRegex(ValueError, "three dimensions"):
            _UTILS.enforce_hmx_launch_contract(json.dumps(_MANIFEST), (1, 1))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            _UTILS.enforce_hmx_launch_contract(json.dumps(_MANIFEST), (1, 0, 1))

    def test_peeled_diagnostic_pipeline_reason_is_closed(self):
        record = _hmx_record(plan="hmx-tail", shape=(33, 33, 33), block_m=33)
        record["execution"]["pipeline"]["reason"] = "tail-peeled-edge"
        record["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(record)
        _UTILS.validate_hmx_manifest(_manifest([record]))

        bad = copy.deepcopy(record)
        bad["execution"]["pipeline"]["reason"] = "unknown-peeled-reason"
        with self.assertRaisesRegex(ValueError, "pipeline.reason"):
            _UTILS.validate_hmx_manifest(_manifest([bad]))

    def test_pipeline_budget_depth_is_optional_but_never_below_depth(self):
        # 2026-10-02: budget_depth records the depth the VTCM budget allowed,
        # which is what makes a clamped ring readable. It is optional so a
        # manifest written before the field stays valid, and it is never allowed
        # to sit below depth: a pass that chose deeper than the budget permitted
        # would be a contradiction, not a clamped run.
        #
        # Every variant re-stamps plan_fingerprint because
        # _plan_fingerprint_payload hashes the whole record minus the digest, so
        # editing pipeline without re-stamping fails on the fingerprint check
        # before the pipeline check is ever reached. The existing tests do the
        # same (see the peeled-reason test above).
        def with_pipeline(**updates):
            record = _hmx_record(plan="full-hmx", shape=(64, 64, 64), block_m=64)
            record["execution"]["pipeline"].update(updates)
            record["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(record)
            return record

        # Absent: a manifest written before the field is still valid.
        _UTILS.validate_hmx_manifest(_manifest([with_pipeline()]))

        # The clamped case this field exists for: staged, clamped from 2 to 1.
        _UTILS.validate_hmx_manifest(
            _manifest([with_pipeline(requested=2, selected="staged", depth=1, budget_depth=2)])
        )

        # A budget deeper than the selection is normal: the budget allowed more
        # than the schedule needed.
        _UTILS.validate_hmx_manifest(_manifest([with_pipeline(budget_depth=3)]))

        # A budget below the selected depth is a contradiction, not a clamp.
        with self.assertRaisesRegex(ValueError, "pipeline.budget_depth"):
            _UTILS.validate_hmx_manifest(
                _manifest([with_pipeline(requested=2, selected="staged", depth=2, budget_depth=1)])
            )

        with self.assertRaisesRegex(ValueError, "pipeline.budget_depth"):
            _UTILS.validate_hmx_manifest(_manifest([with_pipeline(budget_depth=-1)]))

    def test_tail_plan_arithmetic_is_checked(self):
        record = _hmx_record(plan="hmx-tail", shape=(65, 47, 70), block_m=65)
        manifest = _manifest([record])
        _UTILS.validate_hmx_manifest(manifest)

        bad = copy.deepcopy(manifest)
        bad["matmuls"][0]["padded"]["k"] = 64
        with self.assertRaisesRegex(ValueError, "padded.k"):
            _UTILS.validate_hmx_manifest(bad)

    def test_shape_state_is_derived_from_tagged_dimensions(self):
        partial = _hvx_record(reason="dynamic-shape", shape=(64, None, 64))
        dynamic = _hvx_record(record_id=2, reason="dynamic-shape", shape=(None, None, None))
        manifest = _manifest([partial, dynamic])
        _UTILS.validate_hmx_manifest(manifest)

        bad = copy.deepcopy(manifest)
        bad["matmuls"][0]["shape_state"] = "dynamic"
        with self.assertRaisesRegex(ValueError, "shape_state"):
            _UTILS.validate_hmx_manifest(bad)

        unavailable = _hvx_record(
            record_id=3, reason="library-call", shape=None,
            dtypes={"lhs": "unavailable", "rhs": "unavailable", "out": "unavailable"},
        )
        _UTILS.validate_hmx_manifest(_manifest([unavailable]))

    def test_plan_and_reason_matrix_is_closed(self):
        cases = [
            ("full-hmx", "selected-aligned", (64, 64, 64)),
            ("hmx-tail", "selected-tail", (65, 47, 70)),
            ("hvx", "library-call", None),
            ("hvx", "vtcm-allocator-disabled", (64, 64, 64)),
            ("hvx", "non-rank-2", None),
            ("hvx", "dynamic-shape", (None, 64, 64)),
            ("hvx", "unsupported-dtype", (64, 64, 64)),
            ("hvx", "min-rows", (4, 64, 64)),
            ("hvx", "tile-alignment", (65, 64, 64)),
            ("hvx", "unsupported-layout", (64, 64, 64)),
            # The same reason with no logical shape. HmxManifest.cpp:937-944
            # admits an unavailable shape for unsupported-layout as well as for
            # library-call and non-rank-2, and _validate_logical has to agree --
            # the case above alone cannot catch a mirror that drifts, because it
            # only exercises the non-null path. Found 2026-10-02: the C++ side
            # allowed it, backend/utils.py did not, so every contraction the
            # interface generalization refuses (linalg.contract, transposed-B
            # matmul, matvec, vecmat, mmt4d) was rejected at launch.
            ("hvx", "unsupported-layout", None),
            ("hvx", "vtcm-budget", (64, 64, 64)),
        ]
        for index, (plan, reason, shape) in enumerate(cases):
            with self.subTest(plan=plan, reason=reason):
                if plan == "hvx":
                    record = _hvx_record(record_id=index, reason=reason, shape=shape)
                else:
                    record = _hmx_record(
                        record_id=index, plan=plan, reason=reason, shape=shape
                    )
                _UTILS.validate_hmx_manifest(_manifest([record]))

        bad = copy.deepcopy(_MANIFEST)
        bad["matmuls"][0]["reason"] = "selected-tail"
        with self.assertRaisesRegex(ValueError, "canonical pair"):
            _UTILS.validate_hmx_manifest(bad)

    def test_unknown_and_legacy_fields_are_rejected(self):
        for field, value in (
            ("engine", "hmx"),
            ("workspace", {"bytes": 1}),
            ("tail_policy", {"k": "wrong", "mn": "wrong"}),
        ):
            with self.subTest(field=field):
                bad = copy.deepcopy(_MANIFEST)
                bad["matmuls"][0][field] = value
                with self.assertRaisesRegex(ValueError, "unknown field"):
                    _UTILS.validate_hmx_manifest(bad)

        missing = copy.deepcopy(_MANIFEST)
        del missing["weight_policies"]
        with self.assertRaisesRegex(ValueError, "weight_policies"):
            _UTILS.validate_hmx_manifest(missing)

        legacy = copy.deepcopy(_MANIFEST)
        legacy["schema"] = "hex.hmx.kernel_manifest/old"
        with self.assertRaisesRegex(ValueError, _UTILS.HMX_MANIFEST_SCHEMA):
            _UTILS.validate_hmx_manifest(legacy)

    def test_execution_bridge_and_vtcm_contracts_are_nested_and_strict(self):
        bad = copy.deepcopy(_MANIFEST)
        bad["matmuls"][0]["execution"]["bridge_counts"]["unpack_sites"] = -1
        with self.assertRaisesRegex(ValueError, "unpack_sites"):
            _UTILS.validate_hmx_manifest(bad)

        bad = copy.deepcopy(_MANIFEST)
        bad["matmuls"][0]["execution"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown field"):
            _UTILS.validate_hmx_manifest(bad)

        bad = copy.deepcopy(_MANIFEST)
        bad["matmuls"][0]["workspace"] = {"bytes": 1}
        with self.assertRaisesRegex(ValueError, "unknown field"):
            _UTILS.validate_hmx_manifest(bad)

        bad = copy.deepcopy(_MANIFEST)
        bad["matmuls"][0]["grid_policy"] = "single-instance"
        with self.assertRaisesRegex(ValueError, "grid_policy"):
            _UTILS.validate_hmx_manifest(bad)

        bad = copy.deepcopy(_MANIFEST)
        bad["matmuls"][0]["vtcm_accounting"] = "kernel-peak"
        with self.assertRaisesRegex(ValueError, "bridge-only"):
            _UTILS.validate_hmx_manifest(bad)

    def test_weight_policies_and_bindings_are_resolved(self):
        policy = copy.deepcopy(_POLICY)
        record = _hmx_record(weight_kind="argument-slot", policy=policy)
        manifest = _manifest([record], [policy])
        _UTILS.validate_hmx_manifest(manifest)

        missing = copy.deepcopy(manifest)
        missing["weight_policies"] = []
        with self.assertRaisesRegex(ValueError, "weight_policies"):
            _UTILS.validate_hmx_manifest(missing)

        bad = copy.deepcopy(manifest)
        bad["weight_policies"][0]["policy"] = "resident-prepack"
        with self.assertRaisesRegex(ValueError, "not valid for policy"):
            _UTILS.validate_hmx_manifest(bad)

        orphan = copy.deepcopy(manifest)
        orphan["weight_policies"][0]["consumers"] = []
        with self.assertRaisesRegex(ValueError, "consumer"):
            _UTILS.validate_hmx_manifest(orphan)

    def test_fingerprint_is_compact_sorted_json_and_covers_semantics(self):
        record = copy.deepcopy(_MANIFEST["matmuls"][0])
        expected_payload = {"schema": _UTILS.HMX_MANIFEST_SCHEMA}
        expected_payload.update(
            {
                key: value
                for key, value in record.items()
                if key != "plan_fingerprint"
            }
        )
        encoded = json.dumps(
            expected_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        self.assertEqual(
            record["plan_fingerprint"],
            "sha256:" + hashlib.sha256(encoded).hexdigest(),
        )
        reordered = dict(reversed(list(record.items())))
        reordered["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(reordered)
        self.assertEqual(reordered["plan_fingerprint"], record["plan_fingerprint"])

        changed = copy.deepcopy(_MANIFEST)
        changed["matmuls"][0]["weight_binding"] = {"kind": "internal-value"}
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            _UTILS.validate_hmx_manifest(changed)

    def test_apply_translation_metadata_publishes_current_objects(self):
        metadata = {}
        _UTILS.apply_translation_metadata(metadata, _envelope())
        self.assertEqual(json.loads(metadata["weight_prepack"]), _WEIGHT)
        self.assertEqual(json.loads(metadata["hmx_manifest"]), _MANIFEST)
        # A v1 envelope publishes no record, and never a default one.
        self.assertEqual(metadata["hmx_record"], "")
        # The write set passes through decoded: this fixture's envelope carries
        # `null`, so the published value is None -- the fail-closed spelling.
        self.assertIsNone(metadata["arg_writes"])

    def test_the_write_set_passes_through_unchanged(self):
        # One producer decision the launcher must not second-guess: a proved
        # list is published as that list, in order, and a declined one as None.
        # The launcher decides what None costs; this boundary decides nothing.
        # (`validate_pack_metadata` is not in play here: apply_translation_metadata
        # publishes the envelope children only, not the other eleven
        # packed-metadata fields -- the pack boundary is covered separately.)
        for value in (None, [], [0], [1, 3]):
            with self.subTest(arg_writes=value):
                metadata = {}
                _UTILS.apply_translation_metadata(
                    metadata, _envelope(arg_writes=value)
                )
                self.assertEqual(metadata["arg_writes"], value)

    def test_strict_weight_contract_accepts_a_valid_nonempty_entry(self):
        _UTILS.validate_weight_prepack(_VALID_WEIGHT)
        policy = copy.deepcopy(_POLICY)
        policy["policy"] = "resident-prepack"
        policy["reason"] = "eligible-aligned-f16"
        record = _hmx_record(weight_kind="argument-slot", policy=policy)
        manifest = _manifest([record], [policy])
        weight_json, _, record_json, _ = _UTILS.parse_translation_metadata(
            _envelope(manifest=manifest, weight=_VALID_WEIGHT)
        )
        self.assertEqual(json.loads(weight_json), _VALID_WEIGHT)
        self.assertIsNone(record_json)

    def test_the_weight_policy_reason_must_match_the_entry_dtype(self):
        # The Python mirror of the C++ pairing: the two dtype-specific reasons
        # bind to the contract's source dtype, while the B2 N-slice reason
        # names the view and is admitted for either.
        def fixture(reason, dtype):
            policy = copy.deepcopy(_POLICY)
            policy["policy"] = "resident-prepack"
            policy["reason"] = reason
            record = _hmx_record(weight_kind="argument-slot", policy=policy)
            record["dtypes"]["rhs"] = dtype
            record["plan_fingerprint"] = _UTILS.compute_hmx_plan_fingerprint(
                record, policy
            )
            entry = copy.deepcopy(_VALID_WEIGHT)
            entry["weights"][0]["dtype"] = dtype
            return entry, _manifest([record], [policy])

        for reason, dtype in (
            ("eligible-aligned-f16", "f16"),
            ("eligible-quantized-f32", "f32"),
            ("eligible-b2-n-slice", "f16"),
            ("eligible-b2-n-slice", "f32"),
        ):
            with self.subTest(reason=reason, dtype=dtype):
                entry, manifest = fixture(reason, dtype)
                _UTILS.parse_translation_metadata(
                    _envelope(manifest=manifest, weight=entry)
                )

        for reason, dtype in (
            ("eligible-aligned-f16", "f32"),
            ("eligible-quantized-f32", "f16"),
        ):
            with self.subTest(reason=reason, dtype=dtype):
                entry, manifest = fixture(reason, dtype)
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    _UTILS.parse_translation_metadata(
                        _envelope(manifest=manifest, weight=entry)
                    )

    def test_strict_weight_contract_rejects_malformed_entries(self):
        def contract_with(**changes):
            value = copy.deepcopy(_VALID_WEIGHT)
            value.update(changes)
            return value

        def entry_with(**changes):
            value = copy.deepcopy(_VALID_WEIGHT)
            value["weights"][0].update(changes)
            return value

        cases = [
            (contract_with(weights={}), "weights must be a list"),
            (contract_with(weights=[1]), r"weights\[0\] must be an object"),
            (contract_with(layout=None), "layout must be a non-empty object"),
            (contract_with(layout={}), "layout must be a non-empty object"),
            (entry_with(func=""), "func must be a non-empty string"),
            (entry_with(func=1), "func must be a non-empty string"),
            (entry_with(slot=True), "slot must be an int"),
            (entry_with(slot=-1), "slot must be >= 0"),
            (entry_with(shape=True), "shape must be a list"),
            (entry_with(shape=[64]), "shape must have 2 entries"),
            (entry_with(shape=[True, 64]), r"shape\[0\] must be an int"),
            (entry_with(shape=[64, 0]), r"shape\[1\] must be >= 1"),
            (entry_with(crouton=[2, 2, 16, 32]), "crouton must have 5 entries"),
            (entry_with(crouton=[2, 2, 16, 32, 0]), r"crouton\[4\] must be >= 1"),
            (entry_with(dtype="bf16"), "dtype must be 'f16' or 'f32'"),
        ]
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _UTILS.validate_weight_prepack(value)

        missing = copy.deepcopy(_VALID_WEIGHT)
        del missing["weights"][0]["func"]
        with self.assertRaisesRegex(ValueError, "missing required field.*func"):
            _UTILS.validate_weight_prepack(missing)

        duplicate = copy.deepcopy(_VALID_WEIGHT)
        duplicate["weights"].append(copy.deepcopy(duplicate["weights"][0]))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _UTILS.validate_weight_prepack(duplicate)

        multi_function = copy.deepcopy(_VALID_WEIGHT)
        second = copy.deepcopy(multi_function["weights"][0])
        second["func"] = "other_kernel"
        second["slot"] = 2
        multi_function["weights"].append(second)
        with self.assertRaisesRegex(ValueError, "multiple functions"):
            _UTILS.validate_weight_prepack(multi_function)

    def test_packed_metadata_fields_and_semantics_are_validated_at_pack_boundary(self):
        packed = HexagonBackend.pack_metadata(object(), _metadata())
        self.assertEqual(set(packed), set(_UTILS.PACK_METADATA_REQUIRED))
        self.assertIs(_UTILS.validate_pack_metadata(packed), packed)

        missing = _metadata()
        del missing.hmx_manifest
        with self.assertRaisesRegex(RuntimeError, "hmx_manifest"):
            HexagonBackend.pack_metadata(object(), missing)

        bad = _metadata(weight_prepack="{")
        with self.assertRaisesRegex(RuntimeError, "weight_prepack"):
            HexagonBackend.pack_metadata(object(), bad)
        bad_manifest = _metadata(hmx_manifest="[")
        with self.assertRaisesRegex(RuntimeError, "hmx_manifest"):
            HexagonBackend.pack_metadata(object(), bad_manifest)

        stale_manifest = copy.deepcopy(_MANIFEST)
        stale_manifest["schema"] = "hex.hmx.kernel_manifest/v1"
        with self.assertRaisesRegex(RuntimeError, "schema"):
            HexagonBackend.pack_metadata(
                object(), _metadata(hmx_manifest=json.dumps(stale_manifest))
            )

        # The write set is a required child like the others: a cached artifact
        # written before it existed is a stale artifact, not a kernel with the
        # feature off.
        stale = _metadata()
        del stale.arg_writes
        with self.assertRaisesRegex(RuntimeError, "arg_writes"):
            HexagonBackend.pack_metadata(object(), stale)
        for value in ("[", "[1,1]", '[1,0]', "[-1]", '"0"', "1"):
            with self.subTest(arg_writes=value):
                with self.assertRaisesRegex(RuntimeError, "arg_writes"):
                    HexagonBackend.pack_metadata(
                        object(), _metadata(arg_writes=value)
                    )

    def test_translation_weight_policy_and_prepack_are_cross_checked(self):
        policy = copy.deepcopy(_POLICY)
        policy["policy"] = "resident-prepack"
        policy["reason"] = "eligible-aligned-f16"
        record = _hmx_record(weight_kind="argument-slot", policy=policy)
        manifest = _manifest([record], [policy])
        with self.assertRaisesRegex(ValueError, "matching weight_prepack"):
            _UTILS.parse_translation_metadata(_envelope(manifest=manifest))

        weight_json, _, record_json, _ = _UTILS.parse_translation_metadata(
            _envelope(manifest=manifest, weight=_VALID_WEIGHT)
        )
        self.assertEqual(json.loads(weight_json), _VALID_WEIGHT)
        self.assertIsNone(record_json)

    def test_cpp_serialized_manifest_round_trips_through_python(self):
        fixture_root = _HERE.parent / "Conversion" / "LinalgToLLVM"
        options = {k: str(v) for k, v in HexagonOptions().__dict__.items()}
        options["enableWeightResident"] = "True"
        # The host packer's coefficient-map literal, loaded from the test that
        # actually packs with it, so the compiler's own emitted map is bound to
        # a Python literal instead of only to another copy of itself.
        spec = importlib.util.spec_from_file_location(
            "hexagon_backend_prepack_f32_tests",
            _HERE.parent / "test_hmx_weight_prepack_f32.py",
        )
        assert spec is not None and spec.loader is not None
        prepack_tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepack_tests)

        for fixture, reason, dtype in (
            (
                "hmx-weight-resident-runtime-pipeline.mlir",
                "eligible-aligned-f16",
                "f16",
            ),
            (
                "hmx-weight-resident-f32-pipeline.mlir",
                "eligible-quantized-f32",
                "f32",
            ),
        ):
            with self.subTest(fixture=fixture):
                context = ir.context()
                qcom_hexagon_backend.load_dialects(context)
                module = qcom_hexagon_backend.parse_mlir_module_from_str(
                    (fixture_root / fixture).read_text(), context
                )
                _, metadata_json = qcom_hexagon_backend.translate_linalg_to_obj(
                    module, options, True
                )
                envelope = json.loads(metadata_json)
                manifest = envelope["hmx_manifest"]
                weight = envelope["weight_prepack"]
                self.assertIs(_UTILS.validate_hmx_manifest(manifest), manifest)
                # The producer's own prepack contract joined to its policy by
                # the consumer. The f32 fixture is what puts the new reason and
                # the source dtype through the real C++ -> Python boundary.
                _UTILS._validate_manifest_weight_prepack(manifest, weight)
                self.assertEqual(
                    manifest["matmuls"][0]["plan_fingerprint"],
                    _UTILS.compute_hmx_plan_fingerprint(
                        manifest["matmuls"][0], manifest["weight_policies"][0]
                    ),
                )
                self.assertEqual(
                    manifest["weight_policies"][0]["policy"], "resident-prepack"
                )
                self.assertEqual(manifest["weight_policies"][0]["reason"], reason)
                self.assertEqual(weight["weights"][0]["dtype"], dtype)
                # The producer's layout map is the host packer's map; a change
                # to `prepackLayoutJson` must fail here, not in a launch.
                self.assertEqual(
                    weight["layout"]["results"],
                    prepack_tests.LAYOUT["results"],
                    "the compiler layout map drifted from the host literal",
                )

    def test_cpp_serialized_manifest_publishes_workspace_facts(self):
        fixture_root = _HERE.parent / "Conversion" / "LinalgToLLVM"
        options = {k: str(v) for k, v in HexagonOptions().__dict__.items()}
        options["enableWorkspaceResident"] = "True"

        context = ir.context()
        qcom_hexagon_backend.load_dialects(context)
        module = qcom_hexagon_backend.parse_mlir_module_from_str(
            (fixture_root / "hmx-workspace-resident-pipeline.mlir").read_text(),
            context,
        )
        _, metadata_json = qcom_hexagon_backend.translate_linalg_to_obj(
            module, options, True
        )
        manifest = json.loads(metadata_json)["hmx_manifest"]
        _UTILS.validate_hmx_manifest(manifest)
        self.assertEqual(
            manifest["matmuls"][0]["workspace_class"],
            "resident",
        )
        self.assertEqual(
            manifest["matmuls"][0]["grid_policy"], "legacy-runtime"
        )

    def test_cpp_api_rejects_unconsumed_prepack_without_meta(self):
        fixture_root = _HERE.parent / "Conversion" / "LinalgToLLVM"
        options = {k: str(v) for k, v in HexagonOptions().__dict__.items()}
        options["enableWeightResident"] = "True"

        def translate(name):
            context = ir.context()
            qcom_hexagon_backend.load_dialects(context)
            module = qcom_hexagon_backend.parse_mlir_module_from_str(
                (fixture_root / name).read_text(), context
            )
            return qcom_hexagon_backend.translate_linalg_to_obj(module, options, False)

        translate("hmx-weight-resident-pipeline.mlir")
        with self.assertRaisesRegex(RuntimeError, "runtime weight-prepack contract"):
            translate("hmx-weight-resident-runtime-pipeline.mlir")

    def test_cpp_api_rejects_malformed_prepack_attributes(self):
        for attr in ('hmx.weight_prepack = ""', 'hmx.weight_prepack = 1 : i32'):
            with self.subTest(attr=attr):
                context = ir.context()
                qcom_hexagon_backend.load_dialects(context)
                module = qcom_hexagon_backend.parse_mlir_module_from_str(
                    f"module attributes {{{attr}}} {{ func.func @empty() {{ return }} }}",
                    context,
                )
                with self.assertRaisesRegex(RuntimeError, "malformed HMX prepack"):
                    qcom_hexagon_backend.translate_linalg_to_obj(module, {})

    def test_driver_field_helper_does_not_reparse_payloads(self):
        packed = _packed_metadata(weight_prepack="{", hmx_manifest="[")
        self.assertIs(_UTILS.require_pack_metadata_fields(packed), packed)
        with self.assertRaisesRegex(RuntimeError, "weight_prepack"):
            _UTILS.validate_pack_metadata(packed)

        del packed["hmx_manifest"]
        with self.assertRaisesRegex(RuntimeError, "hmx_manifest"):
            _UTILS.require_pack_metadata_fields(packed)

        del packed["hmx_record"]
        with self.assertRaisesRegex(RuntimeError, "hmx_record"):
            _UTILS.require_pack_metadata_fields(packed)


if __name__ == "__main__":
    unittest.main()
