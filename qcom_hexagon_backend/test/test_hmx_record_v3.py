"""Host-only tests for the record-only HMX v3 consumer and envelope.

`hex.hmx.kernel_manifest/v3` is a diagnostic record, not an executable contract.
These tests pin the two properties that make it safe to publish next to a
production manifest: it is a *closed* schema that refuses anything it cannot
interpret, and it grants nothing.  The v2 boundary is checked for invariance
throughout, because the record must never become a v2 upgrade.
"""

import copy
import importlib.util
import json
import pathlib
from pathlib import Path
from types import SimpleNamespace
import unittest


_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parents[1] / "backend"
_SPEC = importlib.util.spec_from_file_location(
    "hexagon_backend_utils", _BACKEND / "utils.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _axis(value=None, *, symbol=None):
    if symbol is not None:
        return {"kind": "dynamic", "symbol": symbol}
    return {"kind": "static", "value": value}


def _logical(m=64, n=64, k=64):
    logical = {}
    for axis, value in (("m", m), ("n", n), ("k", k)):
        if value is None:
            logical[axis] = _axis(symbol=axis)
        else:
            logical[axis] = _axis(value)
    return logical


def _unknown_capacity(unit, basis, quantities):
    block = {"unit": unit, "basis": basis, "status": "not-proven"}
    block.update({quantity: None for quantity in quantities})
    return block


_REQUESTED = (
    "transient_requested_peak_bytes",
    "resident_requested_bytes",
    "modeled_requested_peak_bytes",
)
_ALIGNED = (
    "transient_aligned_peak_bytes",
    "resident_aligned_bytes",
    "modeled_aligned_peak_bytes",
)


def _resources():
    return {
        "requested": _unknown_capacity(
            "bytes", "compile-time-requested", _REQUESTED
        ),
        "allocator_aligned": _unknown_capacity("bytes", "allocator-model", _ALIGNED),
        "observed_high_water": {
            "unit": "bytes",
            "basis": "runtime-observation",
            "status": "not-proven",
            "value_bytes": None,
            "scope": "process-high-water",
            "source": None,
        },
    }


def _proof(status="not-proven", basis=None):
    return {"status": status, "basis": basis}


def _proofs():
    return {axis: _proof() for axis in ("liveness", "allocator", "grid", "resident")}


def _record(
    *,
    record_id=0,
    function="matmul_kernel",
    plan="hvx",
    shape=(64, 64, 64),
    resources=None,
    proofs=None,
):
    m, n, k = shape
    static_count = sum(value is not None for value in shape)
    state = (
        "static" if static_count == 3
        else "dynamic" if static_count == 0
        else "partially-dynamic"
    )
    record = {
        "function": function,
        "id": record_id,
        "plan": plan,
        "shape": {
            "state": state,
            "logical": _logical(m, n, k),
            "specialization": "upstream-static" if static_count == 3 else "upstream-only",
        },
        "scope": {
            "function": function,
            "invocations": 1,
            "grid": {"policy": "single-instance", "required_product": 1},
            "resident": "process-floor",
        },
        "resources": copy.deepcopy(_resources() if resources is None else resources),
        "proofs": copy.deepcopy(_proofs() if proofs is None else proofs),
        "fallback": dict(_UTILS.HMX_RECORD_FALLBACK),
    }
    _reseal(record)
    return record


def _reseal(record):
    record["record_fingerprint"] = _UTILS.compute_hmx_record_fingerprint(record)
    return record


def _document(*records):
    return {
        "schema": "hex.hmx.kernel_manifest/v3",
        "record_mode": "record-only",
        "admission": "not-authorized",
        "records": [copy.deepcopy(record) for record in records],
    }


_RECORD = _record()
_DOCUMENT = _document(_RECORD)


def _with(record, path, value):
    """Return a copy of ``record`` with one nested field replaced."""
    mutated = copy.deepcopy(record)
    node = mutated
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return _reseal(mutated)


def _bad(**changes):
    """A document whose single record is mutated without resealing the digest."""
    document = copy.deepcopy(_DOCUMENT)
    document["records"][0].update(changes)
    return document


def _reject(test, document, pattern):
    with test.assertRaisesRegex(ValueError, pattern):
        _UTILS.validate_hmx_record_document(document)


# ---------------------------------------------------------------------------
# The closed contract
# ---------------------------------------------------------------------------


class RecordDocumentTest(unittest.TestCase):
    def test_static_hvx_tail_and_dynamic_records_are_accepted(self):
        cases = [
            _record(plan="hvx", shape=(64, 64, 64)),
            _record(plan="full-hmx", shape=(64, 64, 64)),
            _record(plan="hmx-tail", shape=(33, 33, 33)),
            _record(plan="hvx", shape=(64, None, None)),
            _record(plan="hvx", shape=(None, None, None)),
        ]
        for record in cases:
            with self.subTest(plan=record["plan"], state=record["shape"]["state"]):
                document = _document(record)
                self.assertIs(_UTILS.validate_hmx_record_document(document), document)

    def test_record_mode_and_admission_are_fixed(self):
        _reject(self, _bad(record_mode="admission-authorized"), "record_mode")
        _reject(self, _bad(admission="authorized"), "admission")
        _reject(self, _bad(schema="hex.hmx.kernel_manifest/v2"), "schema")
        _reject(self, _bad(schema="hex.hmx.kernel_manifest/v4"), "schema")

    def test_unknown_fields_are_rejected_at_every_level(self):
        cases = [
            ({"budget_peak_bytes": 1}, "records\\[0\\] has unknown field"),
            ({"scope": {"function": "matmul_kernel", "invocations": 1,
                        "grid": {"policy": "single-instance", "required_product": 1},
                        "resident": "process-floor", "tenant": "x"}},
             "scope has unknown field"),
            ({"fallback": dict(_UTILS.HMX_RECORD_FALLBACK, admission="allow")},
             "fallback has unknown field"),
            # The two fields the decision doc's illustrative shape listed are
            # deliberately absent: a record-only document makes no such decision,
            # so a record that still carries them is refused rather than ignored.
            ({"fallback": dict(_UTILS.HMX_RECORD_FALLBACK,
                               on_unproven_proof="retain-v2-plan")},
             "fallback has unknown field"),
            ({"fallback": dict(_UTILS.HMX_RECORD_FALLBACK,
                               on_descriptor_mismatch="reject")},
             "fallback has unknown field"),
        ]
        for changes, pattern in cases:
            with self.subTest(pattern=pattern):
                _reject(self, _bad(**changes), pattern)

        # A resource block and a proof block are closed too, so a v4 capacity
        # fact cannot appear in a v3 record unnoticed.
        resources = _resources()
        resources["allocator_aligned"]["fragmentation_bytes"] = 0
        _reject(
            self,
            _document(_record(resources=resources)),
            "allocator_aligned has unknown field",
        )
        proofs = _proofs()
        proofs["liveness"]["observation_id"] = 3
        _reject(
            self,
            _document(_record(proofs=proofs)),
            "liveness has unknown field",
        )

    def test_missing_fields_are_rejected_rather_than_defaulted(self):
        for field in (
            "function",
            "id",
            "plan",
            "shape",
            "scope",
            "resources",
            "proofs",
            "fallback",
            "record_fingerprint",
        ):
            with self.subTest(field=field):
                document = copy.deepcopy(_DOCUMENT)
                del document["records"][0][field]
                _reject(self, document, "missing required field")

    def test_shape_state_and_specialization_are_derived_not_declared(self):
        # A partially dynamic record may not declare itself static.
        dynamic = _record(shape=(64, None, None))
        _reject(
            self,
            _document(_with(dynamic, ("shape", "state"), "static")),
            "the tagged axes derive",
        )
        _reject(
            self,
            _document(_with(_RECORD, ("shape", "specialization"), "upstream-only")),
            "specialization",
        )
        # An unknown declared state is refused outright.
        document = copy.deepcopy(_DOCUMENT)
        document["records"][0]["shape"]["state"] = "guessed"
        _reject(self, document, "state")
        # A dynamic axis names its own axis and carries no runtime guess.
        dynamic = _with(_record(), ("shape", "logical", "m"), {"kind": "dynamic", "symbol": "n"})
        _reject(self, _document(dynamic), "must name this axis")
        guessed = _with(_record(), ("shape", "logical", "m"),
                        {"kind": "dynamic", "symbol": "m", "value": 64})
        _reject(self, _document(guessed), "unknown field")
        both = _with(_record(), ("shape", "logical", "m"),
                     {"kind": "static", "value": 64, "symbol": "m"})
        _reject(self, _document(both), "unknown field")
        zero = _with(_record(), ("shape", "logical", "m"), {"kind": "static", "value": 0})
        _reject(self, _document(zero), "must be >= 1")

    def test_scope_is_fixed_to_the_first_revision(self):
        for path, value, pattern in (
            (("scope", "invocations"), 2, "invocations"),
            (("scope", "grid", "policy"), "legacy-runtime", "grid.policy"),
            (("scope", "grid", "required_product"), 4, "required_product"),
            (("scope", "resident"), "invocation-peak", "resident"),
            (("scope", "function"), "other_kernel", "must name the record"),
        ):
            with self.subTest(pattern=pattern):
                _reject(self, _document(_with(_RECORD, path, value)), pattern)

    def test_duplicate_function_local_ids_are_rejected(self):
        duplicate = _document(_record(record_id=0), _record(record_id=0))
        _reject(self, duplicate, "duplicates function-local id")
        # Distinct functions may reuse an id: the key is function-local.
        _UTILS.validate_hmx_record_document(
            _document(_record(function="alpha"), _record(function="beta"))
        )

    def test_records_may_be_empty_but_must_be_a_list(self):
        empty = _document()
        self.assertEqual(
            _UTILS.validate_hmx_record_document(empty)["records"], []
        )
        document = copy.deepcopy(_DOCUMENT)
        document["records"] = {}
        _reject(self, document, "records must be a list")


# ---------------------------------------------------------------------------
# Honesty: unknown is not zero, and the axes stay separate
# ---------------------------------------------------------------------------


class RecordHonestyTest(unittest.TestCase):
    def test_not_proven_never_carries_a_number(self):
        for quantity in _REQUESTED:
            with self.subTest(quantity=quantity):
                resources = _resources()
                resources["requested"][quantity] = 0
                _reject(
                    self,
                    _document(_record(resources=resources)),
                    f"requested.{quantity} is 0",
                )

    def test_proven_zero_is_a_real_measurement(self):
        # "Proven to be zero" is a fact; "not measured" is not the same fact, and
        # the schema must be able to say both.
        resources = _resources()
        resources["requested"]["status"] = "complete"
        resources["requested"].update(
            {
                "transient_requested_peak_bytes": 0,
                "resident_requested_bytes": 0,
                "modeled_requested_peak_bytes": 0,
            }
        )
        proofs = _proofs()
        proofs["liveness"] = _proof("complete", "structured-requested-upper-bound")
        _UTILS.validate_hmx_record_document(
            _document(_record(resources=resources, proofs=proofs))
        )

    def test_capacity_units_and_bases_are_pinned_per_block(self):
        cases = [
            (("requested", "unit"), "kilobytes", "requested.unit"),
            (("requested", "basis"), "allocator-model", "requested.basis"),
            (("allocator_aligned", "basis"), "compile-time-requested", "allocator_aligned.basis"),
            (("observed_high_water", "unit"), "kibibytes", "observed_high_water.unit"),
            (("observed_high_water", "basis"), "allocator-model", "observed_high_water.basis"),
        ]
        for path, value, pattern in cases:
            with self.subTest(pattern=pattern):
                resources = _resources()
                node = resources
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                _reject(
                    self,
                    _document(_record(resources=resources)),
                    pattern,
                )

    def test_observation_scope_stays_process_aggregate(self):
        # The captured device evidence is a process high-water mark with no
        # (function, allocation-site) join key.  Publishing it per record would
        # be a claim the evidence cannot support, so a narrower scope is refused.
        resources = _resources()
        resources["observed_high_water"]["scope"] = "per-function-high-water"
        _reject(
            self,
            _document(_record(resources=resources)),
            "observed_high_water.scope",
        )

    def test_observed_value_requires_a_source_and_a_proven_status(self):
        resources = _resources()
        resources["observed_high_water"]["value_bytes"] = 4096
        _reject(
            self,
            _document(_record(resources=resources)),
            "value_bytes is 4096",
        )

        resources = _resources()
        resources["observed_high_water"].update(
            {"status": "complete", "value_bytes": 4096, "source": "probe-run-1"}
        )
        _UTILS.validate_hmx_record_document(
            _document(_record(resources=resources))
        )

    def test_four_proof_axes_are_validated_independently(self):
        proofs = _proofs()
        proofs["liveness"] = _proof("complete", "structured-requested-upper-bound")
        proofs["grid"] = _proof("complete", "compile-time-grid")
        _UTILS.validate_hmx_record_document(
            _document(_record(proofs=proofs))
        )
        # A complete grid proof must not have promoted the resident axis.
        self.assertEqual(
            _DOCUMENT["records"][0]["proofs"]["resident"]["status"], "not-proven"
        )

    def test_complete_without_a_basis_is_rejected(self):
        for axis in ("liveness", "allocator", "grid", "resident"):
            with self.subTest(axis=axis):
                proofs = _proofs()
                proofs[axis] = _proof("complete", None)
                _reject(
                    self,
                    _document(_record(proofs=proofs)),
                    f"{axis} is complete but names no basis",
                )

    def test_unknown_proof_status_is_rejected(self):
        proofs = _proofs()
        proofs["allocator"] = _proof("probably-fine", "model")
        _reject(
            self,
            _document(_record(proofs=proofs)),
            "allocator.status is not a canonical proof status",
        )

    def test_fallback_is_protocol_not_policy(self):
        for field in _UTILS.HMX_RECORD_FALLBACK:
            with self.subTest(field=field):
                document = copy.deepcopy(_DOCUMENT)
                document["records"][0]["fallback"][field] = "allow"
                _reject(self, document, f"fallback.{field} must be")

    def test_fingerprint_is_canonical_and_content_sensitive(self):
        record = _RECORD
        reordered = dict(reversed(list(record.items())))
        self.assertEqual(
            _UTILS.compute_hmx_record_fingerprint(reordered),
            record["record_fingerprint"],
        )
        # The digest is prefixed by the schema, so a record cannot be replayed
        # under a different wire name.
        payload = {"schema": "hex.hmx.kernel_manifest/v2"}
        payload.update({k: v for k, v in record.items() if k != "record_fingerprint"})
        self.assertNotEqual(
            _UTILS.compute_hmx_record_fingerprint(record),
            _UTILS.canonical_json_sha256(payload),
        )
        # Any semantic change invalidates the digest.
        changed = copy.deepcopy(record)
        changed["plan"] = "full-hmx"
        _reject(self, _document(changed), "record_fingerprint does not match")
        # ... and a bad digest is rejected rather than recomputed away.
        broken = copy.deepcopy(record)
        broken["record_fingerprint"] = "sha256:" + "0" * 64
        _reject(self, _document(broken), "record_fingerprint does not match")
        broken = copy.deepcopy(record)
        broken["record_fingerprint"] = "sha256:NOTHEX"
        _reject(self, _document(broken), "lowercase sha256")


# ---------------------------------------------------------------------------
# Envelope and cache identity
# ---------------------------------------------------------------------------


class EnvelopeTest(unittest.TestCase):
    def _envelope(self, schema, **overrides):
        from test_hmx_manifest_metadata import _MANIFEST, _WEIGHT  # noqa: PLC0415

        envelope = {
            "schema": schema,
            "weight_prepack": copy.deepcopy(_WEIGHT),
            "hmx_manifest": copy.deepcopy(_MANIFEST),
        }
        envelope.update(copy.deepcopy(overrides))
        return json.dumps(envelope)

    def test_v1_envelope_publishes_no_record(self):
        weight, manifest, record = _UTILS.parse_translation_metadata(
            self._envelope("hex.hmx.translation/v1")
        )
        self.assertIsNone(record)
        self.assertEqual(json.loads(manifest)["schema"], _UTILS.HMX_MANIFEST_SCHEMA)
        self.assertEqual(json.loads(weight), {"layout": None, "weights": []})

    def test_v2_envelope_carries_both_children(self):
        _, manifest, record = _UTILS.parse_translation_metadata(
            self._envelope("hex.hmx.translation/v2", hmx_record=_DOCUMENT)
        )
        self.assertEqual(json.loads(record), _DOCUMENT)
        self.assertEqual(json.loads(manifest)["schema"], _UTILS.HMX_MANIFEST_SCHEMA)

    def test_a_v1_envelope_may_not_carry_a_record(self):
        with self.assertRaisesRegex(ValueError, "unknown field"):
            _UTILS.parse_translation_metadata(
                self._envelope("hex.hmx.translation/v1", hmx_record=_DOCUMENT)
            )

    def test_a_v2_envelope_must_carry_a_record(self):
        with self.assertRaisesRegex(ValueError, "missing required field"):
            _UTILS.parse_translation_metadata(
                self._envelope("hex.hmx.translation/v2")
            )

    def test_unknown_envelope_schema_is_rejected_not_guessed(self):
        for schema in ("hex.hmx.translation/v3", "hex.hmx.kernel_manifest/v3", None):
            with self.subTest(schema=schema):
                with self.assertRaisesRegex(ValueError, "schema must be one of"):
                    _UTILS.parse_translation_metadata(
                        self._envelope(schema)
                    )

    def test_a_malformed_record_child_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "record_fingerprint does not match"):
            _UTILS.parse_translation_metadata(
                self._envelope(
                    "hex.hmx.translation/v2",
                    hmx_record=_bad(plan="full-hmx"),
                )
            )

    def test_apply_publishes_an_empty_record_only_for_v1(self):
        metadata = {}
        _UTILS.apply_translation_metadata(
            metadata, self._envelope("hex.hmx.translation/v1")
        )
        self.assertEqual(metadata["hmx_record"], "")
        metadata = {}
        _UTILS.apply_translation_metadata(
            metadata,
            self._envelope("hex.hmx.translation/v2", hmx_record=_DOCUMENT),
        )
        self.assertEqual(json.loads(metadata["hmx_record"]), _DOCUMENT)

    def test_apply_reports_a_stale_envelope_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "rebuild the kernel"):
            _UTILS.apply_translation_metadata(
                {}, self._envelope("hex.hmx.translation/v2")
            )


class V2InvarianceTest(unittest.TestCase):
    """The v2 boundary must not learn to read a v3 record, or vice versa."""

    def test_v2_validator_rejects_a_v3_document(self):
        with self.assertRaisesRegex(ValueError, "schema must be"):
            _UTILS.validate_hmx_manifest(_DOCUMENT)

    def test_v3_validator_rejects_a_v2_document(self):
        from test_hmx_manifest_metadata import _MANIFEST  # noqa: PLC0415

        # The two shapes do not overlap: a v2 manifest has `matmuls` where a v3
        # document has `records`, so the closed field set refuses it before any
        # value is even looked at.
        with self.assertRaisesRegex(ValueError, "hmx_record is missing required field"):
            _UTILS.validate_hmx_record_document(copy.deepcopy(_MANIFEST))

    def test_a_v3_record_is_not_a_v2_matmul_entry(self):
        record = _RECORD
        # A v3 record has no `reason`, no `dtypes` and no `plan_fingerprint`, and
        # a v2 record has no `proofs`/`resources`/`record_fingerprint`.  Renaming
        # one into the other is not a conversion, and neither validator accepts
        # the result.
        v2_shaped = {
            "function": record["function"],
            "id": record["id"],
            "plan": record["plan"],
            "reason": "vtcm-allocator-disabled",
            "shape_state": "static",
            "logical": record["shape"]["logical"],
            "dtypes": {"lhs": "f16", "rhs": "f16", "out": "f16"},
            "plan_fingerprint": "sha256:" + "0" * 64,
        }
        with self.assertRaises(ValueError):
            _UTILS.validate_hmx_manifest(
                {
                    "schema": _UTILS.HMX_MANIFEST_SCHEMA,
                    "matmuls": [v2_shaped],
                    "weight_policies": [],
                    "pack_act_sites": 0,
                    "pack_weight_sites": 0,
                    "unpack_sites": 0,
                    "count_semantics": "ir_sites",
                }
            )


class CacheIdentityTest(unittest.TestCase):
    def test_every_component_is_separate(self):
        base = _UTILS.hmx_cache_identity(options_hash="opts")
        variants = {
            "envelope": dict(envelope_schema=_UTILS.HMX_TRANSLATION_V2_SCHEMA),
            "manifest": dict(manifest_schema=_UTILS.HMX_RECORD_SCHEMA),
            "accounting": dict(accounting_mode=_UTILS.HMX_RECORD_ACCOUNTING_MODE),
            "proof_abi": dict(proof_producer_abi="hmx-v3-proof-producer-abi-0000-00-v0"),
            "semantic_abi": dict(semantic_hmx_abi="shape-tail-abi-0000-00-v0"),
            "options": dict(options_hash="other"),
        }
        for name, override in variants.items():
            with self.subTest(component=name):
                self.assertNotEqual(_UTILS.hmx_cache_identity(**override), base)

    def test_production_and_record_only_modes_do_not_share_an_identity(self):
        production = _UTILS.hmx_cache_identity()
        record_only = _UTILS.hmx_cache_identity(
            envelope_schema=_UTILS.HMX_TRANSLATION_V2_SCHEMA,
            manifest_schema=_UTILS.HMX_RECORD_SCHEMA,
            accounting_mode=_UTILS.HMX_RECORD_ACCOUNTING_MODE,
        )
        self.assertNotEqual(production, record_only)
        for field in (production.split("|"), record_only.split("|")):
            self.assertEqual(len(field), 6)

    def test_unknown_schemas_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "envelope schema"):
            _UTILS.hmx_cache_identity(envelope_schema="hex.hmx.translation/v9")
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            _UTILS.hmx_cache_identity(manifest_schema="hex.hmx.kernel_manifest/v4")


class PackedMetadataTest(unittest.TestCase):
    def _packed(self, **overrides):
        from test_hmx_manifest_metadata import _MANIFEST, _WEIGHT  # noqa: PLC0415

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
            "hmx_record": "",
        }
        values.update(overrides)
        return values

    def test_record_field_is_required_and_optional_in_content(self):
        packed = self._packed()
        self.assertIs(_UTILS.validate_pack_metadata(packed), packed)
        del packed["hmx_record"]
        with self.assertRaisesRegex(RuntimeError, "hmx_record"):
            _UTILS.require_pack_metadata_fields(packed)

    def test_a_present_record_must_be_complete(self):
        with self.assertRaisesRegex(RuntimeError, "hmx_record"):
            _UTILS.validate_pack_metadata(self._packed(hmx_record="{"))
        with self.assertRaisesRegex(RuntimeError, "record_fingerprint"):
            _UTILS.validate_pack_metadata(
                self._packed(hmx_record=json.dumps(_bad(plan="full-hmx")))
            )

    def test_a_valid_record_survives_the_pack_boundary(self):
        packed = self._packed(hmx_record=json.dumps(_DOCUMENT))
        self.assertIs(_UTILS.validate_pack_metadata(packed), packed)


class CppRoundTripTest(unittest.TestCase):
    """The producer and the consumer must agree on the wire contract.

    Two separate implementations of the same canonical digest and the same closed
    field set exist on purpose: the C++ producer writes the document and the
    Python consumer re-derives it.  If they ever disagree, the disagreement has
    to show up here rather than as a record that validates only on the machine
    that produced it.
    """

    def _translate(self, source):
        from triton._C.libtriton import ir, qcom_hexagon_backend  # noqa: PLC0415
        from triton.backends.qcom_hexagon_backend.hexagon_options import (  # noqa: PLC0415
            HexagonOptions,
        )

        options = {k: str(v) for k, v in HexagonOptions().__dict__.items()}
        context = ir.context()
        qcom_hexagon_backend.load_dialects(context)
        module = qcom_hexagon_backend.parse_mlir_module_from_str(source, context)
        return qcom_hexagon_backend.translate_linalg_to_obj(module, options, True)[1]

    def _sources(self):
        fixture = (
            _HERE.parent
            / "Conversion"
            / "LinalgToLLVM"
            / "hmx-record-v3-pipeline.mlir"
        ).read_text()
        # The same module without the internal record-mode marker: it must keep
        # producing the v1 envelope and a byte-identical v2 execution child.
        return fixture, fixture.replace(
            "module attributes {hmx.diagnostic_v3_record} {", "module {"
        )

    def test_cpp_record_child_validates_and_its_digest_is_reproducible(self):
        marked, _ = self._sources()
        envelope = json.loads(self._translate(marked))
        self.assertEqual(envelope["schema"], _UTILS.HMX_TRANSLATION_V2_SCHEMA)
        self.assertEqual(
            sorted(envelope), ["hmx_manifest", "hmx_record", "schema", "weight_prepack"]
        )

        record = _UTILS.validate_hmx_record_document(envelope["hmx_record"])
        self.assertEqual(
            record["records"][0]["record_fingerprint"],
            _UTILS.compute_hmx_record_fingerprint(record["records"][0]),
        )
        self.assertEqual(record["records"][0]["plan"], "full-hmx")
        # The v2 execution manifest is still validated by its own validator.
        _UTILS.validate_hmx_manifest(envelope["hmx_manifest"])

        weight, manifest, published = _UTILS.parse_translation_metadata(
            json.dumps(envelope)
        )
        self.assertEqual(json.loads(manifest), envelope["hmx_manifest"])
        self.assertEqual(json.loads(published), envelope["hmx_record"])
        self.assertEqual(json.loads(weight), envelope["weight_prepack"])

    def test_the_v2_execution_child_is_identical_in_both_envelopes(self):
        """Marker-on/off A/B over the v2 execution children.

        What this asserts: for one module text, the ``hmx_manifest`` and
        ``weight_prepack`` children are equal whether or not the internal
        record-mode marker is present, and the envelope schema differs exactly as
        designed.  It does **not** assert anything about object bytes -- see
        ``GeneratedCodeInvarianceTest`` for why that is not assertable, and for
        the invariance claim that *is*.
        """
        marked, unmarked = self._sources()
        marked_envelope = json.loads(self._translate(marked))
        unmarked_envelope = json.loads(self._translate(unmarked))
        self.assertEqual(unmarked_envelope["schema"], _UTILS.HMX_TRANSLATION_V1_SCHEMA)
        self.assertEqual(marked_envelope["schema"], _UTILS.HMX_TRANSLATION_V2_SCHEMA)
        self.assertNotIn("hmx_record", unmarked_envelope)
        self.assertEqual(
            marked_envelope["hmx_manifest"], unmarked_envelope["hmx_manifest"]
        )
        self.assertEqual(
            marked_envelope["weight_prepack"], unmarked_envelope["weight_prepack"]
        )

    def test_the_v1_envelope_key_set_is_pinned_too(self):
        """The v2 arm is pinned at `sorted(envelope)` above; v1 was not.

        `_require_exact_fields` rejects any envelope carrying a key the host
        does not know, so a fourth key added to the v1 emitter would leave the
        host suite green and fail in production at `parse_translation_metadata`
        -- with a message that blames the publisher for a decision the host made.
        Pinning both arms keeps that asymmetry from coming back, and it is
        checked against a C++-produced envelope rather than a hand-written dict,
        which is the only way the producer's literal is actually exercised.
        """
        marked, unmarked = self._sources()
        v1_envelope = json.loads(self._translate(unmarked))
        v2_envelope = json.loads(self._translate(marked))
        self.assertEqual(
            sorted(v1_envelope), ["hmx_manifest", "schema", "weight_prepack"]
        )
        self.assertEqual(
            sorted(v2_envelope),
            ["hmx_manifest", "hmx_record", "schema", "weight_prepack"],
        )
        # And the closed shape is enforced, not merely documented: one extra key
        # on either arm is refused rather than ignored.
        for envelope in (v1_envelope, v2_envelope):
            polluted = dict(envelope, surprise=1)
            with self.assertRaisesRegex(ValueError, "surprise"):
                _UTILS.parse_translation_metadata(json.dumps(polluted))

    def test_every_v3_record_agrees_with_the_v2_plan_it_describes(self):
        """Plan agreement, cross-checked rather than co-located.

        The pipeline fixtures pin the same plan string in both schemas, but two
        CHECKs on one output line do not *relate* them -- each would still pass if
        the producer emitted a different plan in one schema and the fixture text
        happened to contain both.  So the relation is asserted here, where both
        documents are parsed: every v3 record must name a function-local id that
        the v2 manifest also owns, and the two must agree on the plan.

        This is also the check that would catch a v3 record describing a kernel
        that was not built -- the failure mode that matters most for a
        record-only schema, since nothing downstream acts on the record to
        surface it.
        """
        for name in (
            "hmx-record-v3-pipeline.mlir",
            "hmx-record-v3-evidence-pipeline.mlir",
            "hmx-record-v3-multi-pipeline.mlir",
        ):
            with self.subTest(fixture=name):
                envelope = json.loads(
                    self._translate(
                        (_HERE.parent / "Conversion" / "LinalgToLLVM" / name).read_text()
                    )
                )
                v2 = _UTILS.validate_hmx_manifest(envelope["hmx_manifest"])
                record = _UTILS.validate_hmx_record_document(envelope["hmx_record"])
                by_id = {(m["function"], m["id"]): m for m in v2["matmuls"]}
                self.assertTrue(record["records"], "the fixture published no records")
                for entry in record["records"]:
                    key = (entry["function"], entry["id"])
                    self.assertIn(
                        key, by_id, f"v3 record {key} has no v2 matmul it describes"
                    )
                    self.assertEqual(
                        entry["plan"],
                        by_id[key]["plan"],
                        f"v3 record {key} plans {entry['plan']!r} but the v2 "
                        f"execution child plans {by_id[key]['plan']!r}",
                    )
                # The record is a subset, never a superset: it may omit a matmul
                # it cannot describe, but it may not invent one.
                self.assertLessEqual(
                    {(e["function"], e["id"]) for e in record["records"]}, set(by_id)
                )

    def test_an_evidence_enriched_record_digest_agrees_across_the_boundary(self):
        """C++/Python digest parity on a record that actually carries numbers.

        An all-``null`` record would agree on its digest for uninteresting
        reasons -- there is nothing in it to canonicalize differently.  This runs
        the fixture whose census+liveness sidecars prove a requested-byte bound,
        and re-derives the digest in Python from the parsed document.
        """
        source = (
            _HERE.parent
            / "Conversion"
            / "LinalgToLLVM"
            / "hmx-record-v3-evidence-pipeline.mlir"
        ).read_text()
        envelope = json.loads(self._translate(source))
        record = _UTILS.validate_hmx_record_document(envelope["hmx_record"])
        entry = record["records"][0]

        # The join really did publish integers and a basis, otherwise this test
        # would silently degrade into the all-null case.
        self.assertEqual(entry["resources"]["requested"]["status"], "complete")
        self.assertEqual(
            entry["resources"]["requested"]["transient_requested_peak_bytes"], 8192
        )
        self.assertEqual(
            entry["resources"]["requested"]["resident_requested_bytes"], 2048
        )
        self.assertEqual(
            entry["resources"]["requested"]["modeled_requested_peak_bytes"], 10240
        )
        self.assertEqual(
            entry["proofs"]["liveness"],
            {"status": "complete", "basis": "structured-requested-upper-bound"},
        )
        # One proven axis promoted nothing.
        for axis in ("allocator", "grid", "resident"):
            self.assertEqual(entry["proofs"][axis]["status"], "not-proven")
        self.assertIsNone(entry["resources"]["allocator_aligned"]["transient_aligned_peak_bytes"])
        self.assertIsNone(entry["resources"]["observed_high_water"]["value_bytes"])

        self.assertEqual(
            entry["record_fingerprint"],
            _UTILS.compute_hmx_record_fingerprint(entry),
        )
        # And a single-field edit on either side is caught.
        mutated = copy.deepcopy(entry)
        mutated["resources"]["requested"]["resident_requested_bytes"] = 2049
        with self.assertRaisesRegex(ValueError, "record_fingerprint"):
            _UTILS.validate_hmx_record_document(_document(mutated))

    def test_a_two_matmul_function_publishes_two_digested_records(self):
        """The P0-1 regression, at the publication boundary.

        Per-function requested-byte evidence is attributable to a single record
        only, so a function with two matmuls must publish two records that both
        stay `not-proven` -- and both must still carry a digest, because a
        skeleton is not a publishable record.
        """
        source = (
            _HERE.parent
            / "Conversion"
            / "LinalgToLLVM"
            / "hmx-record-v3-multi-pipeline.mlir"
        ).read_text()
        envelope = json.loads(self._translate(source))
        record = _UTILS.validate_hmx_record_document(envelope["hmx_record"])
        self.assertEqual(len(record["records"]), 2)
        digests = set()
        for entry in record["records"]:
            self.assertEqual(
                entry["record_fingerprint"],
                _UTILS.compute_hmx_record_fingerprint(entry),
            )
            digests.add(entry["record_fingerprint"])
            # No capacity was claimed: the function-level bound cannot be split.
            self.assertEqual(entry["resources"]["requested"]["status"], "not-proven")
            for quantity in (
                "transient_requested_peak_bytes",
                "resident_requested_bytes",
                "modeled_requested_peak_bytes",
            ):
                self.assertIsNone(entry["resources"]["requested"][quantity])
            self.assertEqual(entry["proofs"]["liveness"]["status"], "not-proven")
        # Two records, two distinct identities -- not one record counted twice.
        self.assertEqual(len(digests), 2)
        # The v2 execution child still owns both.
        v2 = _UTILS.validate_hmx_manifest(envelope["hmx_manifest"])
        self.assertEqual([m["id"] for m in v2["matmuls"]], [0, 1])

    def test_v3_records_are_a_deliberate_subset_of_the_v2_matmuls(self):
        """A v3 record set may be shorter than the v2 matmul list, and that is fine.

        A v3 record needs a three-axis logical shape.  When the compile inputs
        carry none, no record is published for that matmul -- the omission is by
        construction, never a zip that silently drops a record.  The consumer
        must therefore not require a correspondence, because the two schemas have
        different jobs: v2 is the execution authority and owns every matmul, v3
        publishes only the facts it can state.
        """
        marked, _ = self._sources()
        envelope = json.loads(self._translate(marked))
        v2 = _UTILS.validate_hmx_manifest(envelope["hmx_manifest"])
        record = _UTILS.validate_hmx_record_document(envelope["hmx_record"])
        self.assertLessEqual(len(record["records"]), len(v2["matmuls"]))
        # A hand-built subset is a valid document; the validator never reaches for
        # the v2 manifest to complete it.
        subset = _document(*record["records"][:1])
        self.assertEqual(len(subset["records"]), 1)
        self.assertIs(_UTILS.validate_hmx_record_document(subset), subset)
        # An empty record set is also valid: "nothing publishable this run".
        self.assertEqual(_UTILS.validate_hmx_record_document(_document())["records"], [])

    def test_cpp_refuses_a_record_document_without_the_mode_marker(self):
        marked, _ = self._sources()
        source = marked.replace(
            'module attributes {hmx.diagnostic_v3_record} {',
            'module attributes {"hmx.kernel_record/v3" = {records = [], '
            'record_mode = "record-only", admission = "not-authorized", '
            'schema = "hex.hmx.kernel_manifest/v3"}} {',
        )
        # The finalizer catches it in the pipeline, before the envelope is built:
        # a document with no mode marker is a forged artifact, not a v1 kernel.
        # MLIR reports the specific diagnostic on stderr and the binding turns
        # the failed pass into this generic refusal, so the assertion is on the
        # refusal and the exact reason is asserted by the lit fixture
        # `hmx-record-v3-reject.mlir`.
        with self.assertRaisesRegex(RuntimeError, "Failed to convert Triton Linalg"):
            self._translate(source)


class GeneratedCodeInvarianceTest(unittest.TestCase):
    """Marker-on/off must not change a single line of generated code.

    Honest scope, stated up front.  The obvious assertion -- "the object bytes
    are identical" -- is **not** available: two runs of
    ``translate_linalg_to_obj`` on byte-identical input produce different object
    bytes and even different lengths, because the object carries
    nondeterministic content this change does not control.  Asserting object
    equality would therefore be a flaky test that proves nothing.

    So the assertion is made one level up, where it is deterministic and
    strictly stronger in the sense that matters: the *generated IR* -- everything
    ``linalg-to-llvm`` produces, which is exactly what the object is translated
    from -- is byte-identical with and without the marker, and the v2 execution
    children are equal.  What this does not cover is the LLVM backend's own
    object emission, which is unchanged input to unchanged lowering.
    """

    @staticmethod
    def _opt():
        import triton  # noqa: PLC0415

        root = pathlib.Path(triton.__file__).resolve().parents[2]
        found = sorted(
            root.glob("build/*/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt")
        )
        if len(found) != 1:
            raise unittest.SkipTest(
                f"expected exactly one linalg-hexagon-opt under {root}/build, got {found}"
            )
        return found[0]

    @staticmethod
    def _lower(opt, source):
        import subprocess  # noqa: PLC0415

        result = subprocess.run(
            [str(opt), "-pass-pipeline=builtin.module(linalg-to-llvm)", "-"],
            input=source,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    @staticmethod
    def _body(text):
        """The generated code, with the module attribute line removed.

        The record document lives in the module attribute dictionary, which is
        printed on the first line.  Comparing that line would compare the record
        with itself; everything after it is the code the object is translated
        from, and that is the claim.
        """
        header, separator, body = text.partition("\n")
        if not separator or not header.startswith("module"):
            raise AssertionError(f"unexpected lowering output header: {header[:80]!r}")
        return body

    def test_marked_and_unmarked_modules_lower_to_identical_code(self):
        opt = self._opt()
        fixture = (
            _HERE.parent
            / "Conversion"
            / "LinalgToLLVM"
            / "hmx-record-v3-pipeline.mlir"
        ).read_text()
        unmarked = fixture.replace(
            "module attributes {hmx.diagnostic_v3_record} {", "module {"
        )
        marked_text = self._lower(opt, fixture)
        marked_body = self._body(marked_text)
        self.assertNotEqual(marked_body, "", "the marked run produced no code")
        unmarked_body = self._body(self._lower(opt, unmarked))
        self.assertEqual(
            marked_body,
            unmarked_body,
            "the record-mode marker changed the generated code; it is metadata-only "
            "and must not reach codegen",
        )
        # The record document really is present in one run and absent in the
        # other, so the comparison above is not comparing two identical inputs.
        self.assertIn('"hmx.kernel_record/v3"', marked_text)
        self.assertNotIn("hmx.kernel_record/v3", self._lower(opt, unmarked))

    def test_object_bytes_are_not_a_valid_invariance_oracle(self):
        """Document the limitation with a test, so nobody re-adds the bad one.

        If the object ever becomes reproducible, this test fails and the
        docstring above should be narrowed to the stronger claim.  While it
        holds, it is the evidence for why the IR is the oracle.
        """
        from triton._C.libtriton import ir, qcom_hexagon_backend  # noqa: PLC0415
        from triton.backends.qcom_hexagon_backend.hexagon_options import (  # noqa: PLC0415
            HexagonOptions,
        )

        source = (
            _HERE.parent
            / "Conversion"
            / "LinalgToLLVM"
            / "hmx-record-v3-pipeline.mlir"
        ).read_text()
        options = {k: str(v) for k, v in HexagonOptions().__dict__.items()}
        runs = []
        for _ in range(2):
            context = ir.context()
            qcom_hexagon_backend.load_dialects(context)
            module = qcom_hexagon_backend.parse_mlir_module_from_str(source, context)
            objs, _ = qcom_hexagon_backend.translate_linalg_to_obj(
                module, options, True
            )
            runs.append(bytes(objs[0]))
        self.assertNotEqual(
            runs[0],
            runs[1],
            "object bytes are now reproducible; update "
            "GeneratedCodeInvarianceTest to assert them directly, which is a "
            "stronger invariance claim than the IR comparison it replaces",
        )


class CacheBoundaryTest(unittest.TestCase):
    """The production cache boundary, exercised through the real code paths.

    These tests deliberately drive ``HexagonBackend.hash()``, ``pack_metadata``
    and ``ttsharedir_to_obj`` rather than calling the identity helper with
    hand-written arguments.  A helper call with unused constants would pass no
    matter what the key actually contained.
    """

    def _backend(self, **options):
        from triton.backends.compiler import GPUTarget  # noqa: PLC0415
        from triton.backends.qcom_hexagon_backend.compiler import HexagonBackend  # noqa: PLC0415
        from triton.backends.qcom_hexagon_backend.hexagon_options import (  # noqa: PLC0415
            HexagonOptions,
        )

        backend = HexagonBackend(GPUTarget("hexagon", 0, 0))
        opts = dict(HexagonOptions().__dict__)
        opts.update(options)
        backend.parse_options(opts)
        return backend

    def test_the_key_really_is_the_digest_of_the_six_component_identity(self):
        import hashlib  # noqa: PLC0415

        from triton.backends.qcom_hexagon_backend.utils import (  # noqa: PLC0415
            HMX_ACCOUNTING_MODE,
            HMX_MANIFEST_SCHEMA,
            HMX_RECORD_PROOF_PRODUCER_ABI,
            HMX_SHAPE_TAIL_ABI_VERSION,
            TRANSLATION_METADATA_SCHEMA,
            hmx_cache_identity,
        )

        backend = self._backend()
        identity = hmx_cache_identity(
            envelope_schema=TRANSLATION_METADATA_SCHEMA,
            manifest_schema=HMX_MANIFEST_SCHEMA,
            accounting_mode=HMX_ACCOUNTING_MODE,
            proof_producer_abi=HMX_RECORD_PROOF_PRODUCER_ABI,
            semantic_hmx_abi=HMX_SHAPE_TAIL_ABI_VERSION,
            options_hash=backend._parsed_options.hash(),
        )
        # Every component is in the key by construction, not by convention: the
        # key is the digest of the joined identity, so dropping a field changes
        # it.
        self.assertEqual(len(identity.split("|")), 6)
        self.assertEqual(
            backend.hash(),
            hashlib.sha256(f"{identity}-{backend.target}".encode("utf-8")).hexdigest(),
        )

    def test_effective_options_change_the_production_key(self):
        baseline = self._backend()
        for options in (
            {"enableVTCMTiling": not baseline._parsed_options.enableVTCMTiling},
            {"enableConvertToHexagonmem": False},
            {"num_warps": 8},
        ):
            with self.subTest(options=options):
                self.assertNotEqual(baseline.hash(), self._backend(**options).hash())

    def test_the_key_is_stable_for_identical_options(self):
        self.assertEqual(self._backend().hash(), self._backend().hash())

    def test_the_record_only_envelope_is_refused_on_the_cacheable_path(self):
        """A marked module can never become -- or read -- a Triton cache entry.

        This is the P0-2 mechanism.  ``hash()`` has no per-source input, so the
        marker cannot be part of the key; a marked module that compiled here
        would silently lose ``hmx_record`` to a v1 cache hit.  Refusing it at the
        stage boundary is what makes the two modes non-shareable, and it fails
        closed with no option to turn it off.
        """
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            ttsharedir_to_llir,
            ttsharedir_to_obj,
        )
        from triton.backends.qcom_hexagon_backend.hexagon_options import (  # noqa: PLC0415
            HexagonOptions,
        )

        marked = (
            _HERE.parent
            / "Conversion"
            / "LinalgToLLVM"
            / "hmx-record-v3-pipeline.mlir"
        ).read_text()
        options = HexagonOptions()
        for stage in (ttsharedir_to_obj, ttsharedir_to_llir):
            with self.subTest(stage=stage.__name__):
                with self.assertRaisesRegex(RuntimeError, "Triton object cache"):
                    stage(marked, options)
        # The guard is marker-scoped, not a blanket rejection of the stage
        # functions: ordinary text passes it untouched.
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            _reject_cacheable_record_only_module,
        )

        _reject_cacheable_record_only_module("module { func.func @f() { return } }")

    def test_the_guard_is_scoped_to_the_marker(self):
        from triton.backends.qcom_hexagon_backend.compiler import (  # noqa: PLC0415
            _reject_cacheable_record_only_module,
        )

        marker = "hmx.diagnostic_v3_record"
        _reject_cacheable_record_only_module("module {}")
        _reject_cacheable_record_only_module("module { func.func @f() { return } }")
        with self.assertRaises(RuntimeError):
            _reject_cacheable_record_only_module(f"module attributes {{{marker}}} {{}}")
        # A mention in a string literal is not the module attribute; the guard is
        # a text screen, so the *producer* also checks the parsed attribute.  The
        # three calls above are the whole assertion -- the outer bound of this
        # check is the producer's parsed-attribute test, not anything here.

    def test_a_v1_envelope_packs_with_the_named_absent_spelling(self):
        from test_hmx_manifest_metadata import _MANIFEST, _WEIGHT  # noqa: PLC0415

        from triton.backends.qcom_hexagon_backend.compiler import HexagonBackend  # noqa: PLC0415
        from triton.backends.qcom_hexagon_backend.utils import (  # noqa: PLC0415
            HMX_RECORD_ABSENT,
        )

        metadata = SimpleNamespace(
            num_warps=1,
            num_ctas=1,
            shared=0,
            cluster_dims=(),
            name="matmul_kernel",
            return_types=[],
            iterations=1,
            scratch=0,
            enableMultiThreading=False,
            enableThreadedDispatch=False,
            enableLWP=False,
            weight_prepack=json.dumps(_WEIGHT),
            hmx_manifest=json.dumps(_MANIFEST),
            hmx_record=HMX_RECORD_ABSENT,
        )
        packed = HexagonBackend.pack_metadata(object(), metadata)
        self.assertEqual(packed["hmx_record"], HMX_RECORD_ABSENT)
        self.assertIsNotNone(packed["hmx_record"])
        # A stale entry that predates the field is a loud error, not a default.
        del metadata.hmx_record
        with self.assertRaisesRegex(RuntimeError, "hmx_record"):
            HexagonBackend.pack_metadata(object(), metadata)


class _RecordingLauncher:
    """Stands in for ``TritonHexagonLauncher`` and records how it was called."""

    def __init__(self):
        self.calls = []

    def _exec_kernel(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "launched"


class DriverRecordTest(unittest.TestCase):
    """The driver may read the record; the launcher must not."""

    @staticmethod
    def _launcher():
        """A real ``HexagonLauncher`` instance, built without ``__init__``.

        ``__init__`` needs a triton ``ASTSource`` and builds a
        ``TritonHexagonLauncher``; neither is available or wanted on the host.
        ``object.__new__`` gives a genuine instance, so the methods under test
        are reached as *bound* methods with a real ``self`` -- which is how the
        driver reaches them.  An earlier version of this class held the *class*
        and called the unbound function with ``self=None``; that only passed
        because ``_load_hmx_record`` never reads ``self``, so the launch path
        was untested and the ``None`` read as a two-argument signature.
        """
        from triton.backends.qcom_hexagon_backend.driver import (  # noqa: PLC0415
            getHexagonLauncherClass,
        )

        return object.__new__(getHexagonLauncherClass())

    @staticmethod
    def _packed(**overrides):
        from test_hmx_manifest_metadata import _MANIFEST, _WEIGHT  # noqa: PLC0415

        packed = {
            "return_types": [],
            "name": "matmul_kernel",
            "iterations": 1,
            "scratch": 0,
            "enableMultiThreading": False,
            "enableThreadedDispatch": False,
            "enableLWP": False,
            # Required by `require_pack_metadata_fields` on the launch path;
            # the driver's own structural contract is checked separately.
            "num_warps": 4,
            "num_ctas": 1,
            "shared": 0,
            "cluster_dims": [],
            "weight_prepack": json.dumps(_WEIGHT),
            "hmx_manifest": json.dumps(_MANIFEST),
            "hmx_record": "",
        }
        packed.update(overrides)
        return packed

    def _call_args(self, packed):
        """The positional tuple ``CompiledKernel.run`` hands to ``__call__``."""
        return (1, 1, 1, None, "kernel_llir", packed, None, None, None)

    def test_it_is_a_real_instance_not_a_class(self):
        # Guards the mistake this class used to make: a test that reaches the
        # method through the class never runs the code the driver runs.
        import inspect  # noqa: PLC0415

        launcher = self._launcher()
        self.assertFalse(inspect.isclass(launcher))
        self.assertIsNotNone(launcher._load_hmx_record.__self__)

    def test_a_v1_envelope_leaves_no_record(self):
        launcher = self._launcher()
        self.assertIsNone(launcher._load_hmx_record(self._packed()))

    def test_a_v2_envelope_record_is_validated_and_kept(self):
        launcher = self._launcher()
        packed = self._packed(hmx_record=json.dumps(_DOCUMENT))
        self.assertEqual(
            json.loads(launcher._load_hmx_record(packed)), _DOCUMENT
        )
        # A malformed record is an actionable error, not a silent "no record".
        packed["hmx_record"] = json.dumps(_bad(plan="full-hmx"))
        with self.assertRaisesRegex(RuntimeError, "hmx_record"):
            launcher._load_hmx_record(packed)

    def test_a_malformed_record_stops_the_launch(self):
        """Fail-closed has to happen on the launch path, not only in a helper."""
        launcher = self._launcher()
        launcher.launcher = _RecordingLauncher()
        launcher.input_type_list = {}
        packed = self._packed(hmx_record=json.dumps(_bad(plan="full-hmx")))
        with self.assertRaisesRegex(RuntimeError, "hmx_record"):
            launcher(*self._call_args(packed))
        # The launcher must not have been reached: a record is evidence, and
        # unreadable evidence is not a reason to run the kernel.
        self.assertEqual(launcher.launcher.calls, [])

    def test_the_launch_call_receives_the_manifest_but_never_the_record(self):
        """The boundary, checked by what the launcher is actually handed."""
        launcher = self._launcher()
        launcher.launcher = _RecordingLauncher()
        launcher.input_type_list = {}
        packed = self._packed(hmx_record=json.dumps(_DOCUMENT))
        launcher(*self._call_args(packed))
        self.assertEqual(len(launcher.launcher.calls), 1)
        _, kwargs = launcher.launcher.calls[0]
        self.assertIn("hmx_manifest", kwargs)
        self.assertNotIn("hmx_record", kwargs)
        # The driver kept the validated record for its own diagnostic read.
        self.assertEqual(launcher.hmx_record_diagnostic(), packed["hmx_record"])

    def test_the_launcher_api_cannot_carry_a_record(self):
        import inspect  # noqa: PLC0415

        from triton.backends.qcom_hexagon_backend import (  # noqa: PLC0415
            triton_hexagon_launcher,
        )

        # Even if a future caller tried, there is no parameter to carry it.
        parameters = inspect.signature(
            triton_hexagon_launcher.TritonHexagonLauncher._exec_kernel
        ).parameters
        self.assertNotIn("hmx_record", parameters)
        self.assertIn("hmx_manifest", parameters)


if __name__ == "__main__":
    unittest.main()
