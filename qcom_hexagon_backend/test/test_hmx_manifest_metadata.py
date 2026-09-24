"""Host-only tests for the HMX translation metadata envelope."""

import importlib.util
import json
import re
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


_MANIFEST = {
    "schema": "hex.hmx.kernel_manifest/v1",
    "matmuls": [
        {
            "function": "matmul_kernel",
            "id": 0,
            "engine": "hmx",
            "reason": "selected",
            "m": 64,
            "n": 64,
            "k": 64,
            "lhs_elem": "f16",
            "rhs_elem": "f16",
            "out_elem": "f16",
            "vtcm_budget": 8388608,
            "vtcm_before": 0,
            "vtcm_peak": 24576,
            "blocking": "whole",
            "block_m": 64,
            "pack_act_sites": 1,
            "pack_weight_sites": 0,
            "unpack_sites": 1,
            "count_semantics": "ir_sites",
        },
        {
            "function": "matmul_kernel",
            "id": 1,
            "engine": "hvx",
            "reason": "vtcm-budget",
        },
    ],
    "pack_act_sites": 1,
    "pack_weight_sites": 0,
    "unpack_sites": 1,
    "count_semantics": "ir_sites",
}
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
        }
    ],
}


def _envelope(*, manifest=_MANIFEST, weight=_WEIGHT):
    return json.dumps(
        {
            "schema": "hex.hmx.translation/v1",
            "weight_prepack": weight,
            "hmx_manifest": manifest,
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
    }
    values.update(overrides)
    return values


def _metadata(**overrides):
    return SimpleNamespace(**_packed_metadata(**overrides))


class TranslationMetadataTest(unittest.TestCase):
    def test_valid_envelope_unpacks_independent_objects(self):
        weight_json, manifest_json = _UTILS.parse_translation_metadata(_envelope())
        self.assertEqual(json.loads(weight_json), _WEIGHT)
        self.assertEqual(json.loads(manifest_json), _MANIFEST)
        self.assertNotIn("translation", weight_json)
        self.assertNotIn("translation", manifest_json)

    def test_missing_manifest_is_rejected(self):
        envelope = json.loads(_envelope())
        del envelope["hmx_manifest"]
        with self.assertRaisesRegex(ValueError, "hmx_manifest"):
            _UTILS.parse_translation_metadata(json.dumps(envelope))

    def test_bad_manifest_schema_is_rejected(self):
        manifest = dict(_MANIFEST, schema="hex.hmx.kernel_manifest/v0")
        with self.assertRaisesRegex(ValueError, "kernel_manifest/v1"):
            _UTILS.parse_translation_metadata(_envelope(manifest=manifest))

    def test_manifest_entry_missing_required_field_is_rejected(self):
        manifest = json.loads(json.dumps(_MANIFEST))
        del manifest["matmuls"][0]["reason"]
        with self.assertRaisesRegex(ValueError, "reason"):
            _UTILS.parse_translation_metadata(_envelope(manifest=manifest))

    def test_manifest_entry_shape_is_checked(self):
        bad = dict(_MANIFEST)
        bad["matmuls"] = [
            {
                "function": "matmul_kernel",
                "id": True,
                "engine": "hmx",
                "reason": "selected",
            }
        ]
        with self.assertRaisesRegex(ValueError, "id must be a non-negative int"):
            _UTILS.parse_translation_metadata(_envelope(manifest=bad))

    def test_hmx_entry_requires_final_bridge_counts(self):
        manifest = json.loads(json.dumps(_MANIFEST))
        del manifest["matmuls"][0]["unpack_sites"]
        with self.assertRaisesRegex(ValueError, "unpack_sites"):
            _UTILS.validate_hmx_manifest(manifest)

        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["matmuls"][0]["pack_act_sites"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _UTILS.validate_hmx_manifest(manifest)

        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["pack_act_sites"] = 2
        with self.assertRaisesRegex(ValueError, "per-matmul counts sum"):
            _UTILS.validate_hmx_manifest(manifest)

    def test_engine_and_reason_must_agree(self):
        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["matmuls"][0]["reason"] = "tile-alignment"
        with self.assertRaisesRegex(ValueError, "engine/reason disagree"):
            _UTILS.validate_hmx_manifest(manifest)

    def test_selected_contract_and_pipeline_relationships_are_checked(self):
        cases = [
            ("m", "partial matmul contract"),
            ("block_m", "must be >= 1"),
            ("vtcm_peak", "exceeds vtcm_budget"),
            ("blocking", "must be 'whole' or 'm_blocked'"),
        ]
        for field, message in cases:
            manifest = json.loads(json.dumps(_MANIFEST))
            if field == "m":
                del manifest["matmuls"][0][field]
            elif field == "block_m":
                manifest["matmuls"][0][field] = 0
            elif field == "vtcm_peak":
                manifest["matmuls"][0][field] = manifest["matmuls"][0]["vtcm_budget"] + 1
            else:
                manifest["matmuls"][0][field] = "none"
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    _UTILS.validate_hmx_manifest(manifest)

        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["matmuls"][0].update(
            pipeline_requested=2,
            pipeline_selected="staged",
            pipeline_depth=0,
        )
        with self.assertRaisesRegex(ValueError, "positive for staged"):
            _UTILS.validate_hmx_manifest(manifest)

        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["matmuls"][1]["pipeline_requested"] = 0
        manifest["matmuls"][1]["pipeline_selected"] = "staged"
        manifest["matmuls"][1]["pipeline_depth"] = 1
        with self.assertRaisesRegex(ValueError, "HMX-only fields"):
            _UTILS.validate_hmx_manifest(manifest)

        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["matmuls"].append(json.loads(json.dumps(manifest["matmuls"][0])))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _UTILS.validate_hmx_manifest(manifest)

    def test_apply_translation_metadata_publishes_both_fields(self):
        metadata = {}
        _UTILS.apply_translation_metadata(metadata, _envelope())
        self.assertEqual(json.loads(metadata["weight_prepack"]), _WEIGHT)
        self.assertEqual(json.loads(metadata["hmx_manifest"]), _MANIFEST)

    def test_strict_weight_contract_accepts_a_valid_nonempty_entry(self):
        _UTILS.validate_weight_prepack(_VALID_WEIGHT)
        weight_json, _ = _UTILS.parse_translation_metadata(
            _envelope(weight=_VALID_WEIGHT)
        )
        self.assertEqual(json.loads(weight_json), _VALID_WEIGHT)

    def test_strict_weight_contract_rejects_malformed_entries(self):
        def contract_with(**changes):
            value = json.loads(json.dumps(_VALID_WEIGHT))
            value.update(changes)
            return value

        def entry_with(**changes):
            value = json.loads(json.dumps(_VALID_WEIGHT))
            value["weights"][0].update(changes)
            return value

        cases = [
            (contract_with(weights={}), "weights must be a list"),
            (contract_with(weights=[1]), "weights\\[0\\] must be an object"),
            (contract_with(layout=None), "layout must be a non-empty object"),
            (contract_with(layout={}), "layout must be a non-empty object"),
            (entry_with(func=""), "func must be a non-empty string"),
            (entry_with(func=1), "func must be a non-empty string"),
            (entry_with(slot=True), "slot must be an int"),
            (entry_with(slot=-1), "slot must be >= 0"),
            (entry_with(shape=True), "shape must be a list"),
            (entry_with(shape=[64]), "shape must have 2 entries"),
            (entry_with(shape=[True, 64]), "shape\\[0\\] must be an int"),
            (entry_with(shape=[64, 0]), "shape\\[1\\] must be >= 1"),
            (entry_with(crouton=[2, 2, 16, 32]), "crouton must have 5 entries"),
            (entry_with(crouton=[2, 2, 16, 32, 0]), "crouton\\[4\\] must be >= 1"),
            (entry_with(dtype="f32"), "dtype must be 'f16'"),
        ]
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _UTILS.validate_weight_prepack(value)

        missing = json.loads(json.dumps(_VALID_WEIGHT))
        del missing["weights"][0]["func"]
        with self.assertRaisesRegex(ValueError, "missing required field.*func"):
            _UTILS.validate_weight_prepack(missing)

        duplicate = json.loads(json.dumps(_VALID_WEIGHT))
        duplicate["weights"].append(json.loads(json.dumps(duplicate["weights"][0])))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _UTILS.validate_weight_prepack(duplicate)

        multi_function = json.loads(json.dumps(_VALID_WEIGHT))
        second = json.loads(json.dumps(multi_function["weights"][0]))
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
        bad_manifest = _metadata(hmx_manifest="{")
        with self.assertRaisesRegex(RuntimeError, "hmx_manifest"):
            HexagonBackend.pack_metadata(object(), bad_manifest)

        invalid_weight = _metadata(
            weight_prepack=json.dumps(
                {
                    "layout": _LAYOUT,
                    "weights": [
                        {
                            "func": "matmul_kernel",
                            "slot": True,
                            "shape": [64, 64],
                            "crouton": [2, 2, 16, 32, 2],
                            "dtype": "f16",
                        }
                    ],
                }
            )
        )
        with self.assertRaisesRegex(RuntimeError, "slot"):
            HexagonBackend.pack_metadata(object(), invalid_weight)

    def test_unknown_manifest_reason_is_rejected(self):
        manifest = json.loads(json.dumps(_MANIFEST))
        manifest["matmuls"][1]["reason"] = "future-unknown-reason"
        with self.assertRaisesRegex(ValueError, "canonical HMX reason"):
            _UTILS.validate_hmx_manifest(manifest)

    def test_every_canonical_manifest_reason_is_accepted(self):
        for reason in _UTILS.HMX_MANIFEST_REASONS:
            with self.subTest(reason=reason):
                manifest = json.loads(json.dumps(_MANIFEST))
                if reason == "selected":
                    manifest["matmuls"][0]["reason"] = reason
                else:
                    manifest["matmuls"][1]["reason"] = reason
                _UTILS.validate_hmx_manifest(manifest)

    def test_python_reason_allowlist_matches_cpp_vocabulary(self):
        backend_root = _HERE.parents[1]
        header_path = backend_root / "include/hexagon/Dialect/Hmx/Transforms/HmxManifest.h"
        producer_path = backend_root / "lib/Dialect/Hmx/Transforms/MatmulToHmxPass.cpp"
        header_text = header_path.read_text()
        reason_constants = dict(
            re.findall(r"\b(kHmxReason\w+)\s*=\s*\"([^\"]+)\"", header_text)
        )
        canonical_body = header_text.split(
            "inline bool isCanonicalHmxMatmulReason", 1
        )[1].split("\n}", 1)[0]
        producer_text = producer_path.read_text()
        for name in reason_constants:
            with self.subTest(constant=name):
                self.assertIn(name, canonical_body)
                self.assertIn(f"return {name};", producer_text)
        self.assertEqual(set(reason_constants.values()), _UTILS.HMX_MANIFEST_REASONS)

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


if __name__ == "__main__":
    unittest.main()
