# ===- utils.py -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import json
import os, re
import torch
from collections import namedtuple
from enum import Enum


class MissingEnvironmentVariable(Exception):
    pass


def get_env_var(name, default=None):
    if default is not None:
        return os.environ.get(name, default)
    try:
        return os.environ[name]
    except KeyError:
        raise MissingEnvironmentVariable(f"{name} variable needed in the environment!")


def split_path(path):
    filename = os.path.basename(path)
    directory = os.path.dirname(path)
    filename_without_ext, extension = os.path.splitext(filename)
    return (directory, filename_without_ext, extension)


def to_torch_type(type_id):
    cpp_to_torch_map = {
        0: torch.bool,
        1: torch.float16,
        2: torch.float32,
        3: torch.float64,
        4: torch.int8,
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        8: torch.uint8,
    }
    # Old torch versions do not support torch.uint16, torch.uint32, torch.uint64
    try:
        cpp_to_torch_map[9] = torch.uint16
        cpp_to_torch_map[10] = torch.uint32
    except:
        pass
    return cpp_to_torch_map[type_id]


def replace_list_brackets(dimension_list):
    """Replaces a list surrounded by square brackets with curly braces"""
    dimension_list = str(dimension_list)
    if "[" in str(dimension_list) and "]" in dimension_list:
        return dimension_list.translate(str.maketrans({"[": "{", "]": "}"}))


def get_shape(shape):
    """Converts torch sizes to conventional list of sizes/strides"""
    sizes = [
        dimension for dimension in shape
    ]  # shape here will be of form torch.Size[...]
    # Now, we want the strides. NOTE: We can skip the first dimension and keep inserting and multiplying with val at first index
    strides = [1]
    for dim_shape in reversed(sizes[1:]):
        strides.insert(0, strides[0] * dim_shape)
    sizes = replace_list_brackets(sizes)
    strides = replace_list_brackets(strides)
    return sizes, strides


def get_ctype(dtype):
    """Maps a torch data type to hexagon-clang compatible data types"""
    mapping = {
        torch.double: "double",
        torch.float: "float",
        torch.float64: "double",
        torch.float32: "float",
        torch.float16: "_Float16",
        torch.long: "int32_t",
        torch.int: "int32_t",
        torch.int64: "int64_t",
        torch.int32: "int32_t",
        torch.int16: "int16_t",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.bool: "bool",
    }
    # Old torch versions do not support torch.uint16, torch.uint32 and
    # torch.uint64. Add entry into the map only if there are no errors
    try:
        mapping[torch.uint64] = "uint64_t"
        mapping[torch.uint32] = "uint32_t"
        mapping[torch.uint16] = "uint16_t"
    except:
        pass
    return mapping.get(
        dtype, "unknown_type"
    )  # returns value of key-value pair if found in mapping, else returns "unknown_type"


def get_mlir_to_ctype(dtype):
    """Maps a MLIR data type to hexagon-clang compatible data types"""
    mapping = {
        "f64": "double",
        "f32": "float",
        "f16": "_Float16",
        "i32": "int32_t",
        "i64": "int64_t",
        "i16": "int16_t",
        "i8": "int8_t",
        "b8": "bool",
    }

    return mapping.get(
        dtype, "unknown_type"
    )  # returns value of key-value pair if found in mapping, else returns "unknown_type"


def profile_triton_inputs(inputs):
    return profile_inputs(inputs)


def profile_torch_mlir_inputs(inputs):
    return profile_inputs(inputs)


# Given an list of input tensors, to return the profile inputs by categorizing the inputs
# into various inputs types and deriving related metadata.
def profile_inputs(inputs):
    Profiled_input = namedtuple(
        "InputProfile",
        (
            "idx",  # Index for tensor or scalar input
            "input_id",  # Corresponding index in the input list
            "input_type",  # Type of the input (tensor or scalar)
            "value",  # Actual value of the input
            "dtype",  # CPP data type of the input value
            "rank",  # Rank of the input (None for scalars)
            "shape",  # Shape of the input (None for scalars)
        ),
    )
    profiled = []
    tensor_index = 0
    scalar_index = 0
    for index, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            profiled.append(
                Profiled_input(
                    input_type="tensor",
                    idx=tensor_index,
                    input_id=index,
                    value=inp.data,
                    dtype=get_ctype(inp.dtype),
                    rank=inp.dim(),
                    shape=get_shape(inp.shape),
                )
            )
            tensor_index += 1
        elif isinstance(inp, (int, float)):
            profiled.append(
                Profiled_input(
                    input_type="scalar",
                    idx=scalar_index,
                    input_id=index,
                    value=inp,
                    dtype=type(inp).__name__,
                    rank=None,
                    shape=None,
                )
            )
            scalar_index += 1
        else:
            raise ValueError(f"Unsupported input type {type(inp.__name__)}")
    return profiled


# TODO: Parses the LLVM IR to ge the kernel_name. This is limited to use cases where there is only one function
# hence, need to directly pass the filename.
def parse_triton_llvm_kernel_signature(llir, num_tensors, num_inputs):

    pattern = r"define (.+) @(\w+)\((.+)\)"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1

    # Parse return type
    return_type = matches[0][0]
    assert return_type == "void"

    # Get kernel name
    kernel_name = matches[0][1]

    kernel_args = matches[0][2].split(",")
    # Assert that inputs tensors are unranked
    ptr_cnt = sum(1 for arg in kernel_args if "ptr" in arg)
    assert ptr_cnt == num_tensors

    return kernel_name


def make_profiled_return(iter_in):
    """Generate a profile for each return type"""
    profiled_return = namedtuple(
        "ReturnProfile",
        (
            "idx",  # Index of return type
            "dtype",  # CPP data type of the return type
            "rank",  # Rank of the return type (None for scalars)
        ),
    )
    return profiled_return(*iter_in)


def parse_return_types(return_type_list):

    return_types = []
    for idx, r in enumerate(return_type_list):
        rank = None if r[0] == 0 else r[0]
        dtype = str(r[1])

        return_types.append(make_profiled_return([idx, get_mlir_to_ctype(dtype), rank]))

    return return_types


def get_exec_mode():
    """
    Determines the expected execution target (device or simulator)
    which is required for generating the cpp wrapper and
    configuring the HexagonExecutor.
    Defaults to runnig the kernel on the device.
    """
    RUN_ON_SIM = get_env_var("RUN_ON_SIM", default="0")
    if RUN_ON_SIM == "0":
        return "device"
    elif RUN_ON_SIM == "1":
        return "simulator"
    else:
        raise ValueError(
            "Invalid value for RUN_ON_SIM. Set RUN_ON_SIM to 1 "
            "to run the kernel on the simulator and 0 to run it on the device."
        )


# ---------------------------------------------------------------------------
# Compiled-kernel metadata contract (backend/compiler.py -> backend/driver.py)
# ---------------------------------------------------------------------------
# Triton core treats this object as opaque: CompiledKernel stores
# backend.pack_metadata(...) as kernel.packed_metadata and passes it to run() as
# one positional argument (triton/python/triton/compiler/compiler.py), where
# this backend reads it back as args[5].  Nothing outside this backend indexes
# it, so it is a dict of field names rather than a positional tuple.  Every
# field is required: a cache entry without the translation envelope is stale,
# not a kernel with an optional feature disabled.
PACK_METADATA_REQUIRED = (
    "num_warps",
    "num_ctas",
    "shared",
    "cluster_dims",
    "name",
    "return_types",
    "iterations",
    "scratch",
    "enableMultiThreading",
    "enableThreadedDispatch",
    "enableLWP",
    "weight_prepack",
    "hmx_manifest",
)

TRANSLATION_METADATA_SCHEMA = "hex.hmx.translation/v1"
HMX_MANIFEST_SCHEMA = "hex.hmx.kernel_manifest/v1"
HMX_MANIFEST_REASONS = frozenset(
    {
        "selected",
        "library-call",
        "vtcm-allocator-disabled",
        "non-rank-2",
        "dynamic-shape",
        "unsupported-dtype",
        "min-rows",
        "tile-alignment",
        "vtcm-budget",
    }
)
HMX_PIPELINE_REASONS = frozenset(
    {
        "serial-requested",
        "no-row-major-bridge",
        "extra-activation-reader",
        "invalid-staging-geometry",
        "empty-staging-grid",
        "staging-grid-mismatch",
        "shallow-k",
        "vtcm-budget",
        "tile-count",
        "pipeliner-failed",
    }
)
_KEBAB_CASE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def _json_object(value, field_name):
    """Decode one JSON object and turn every failure into an actionable error."""
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a JSON string, got {type(value).__name__}"
        )
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(
            f"{field_name} must decode to an object, got {type(decoded).__name__}"
        )
    return decoded


def validate_hmx_manifest(manifest, field_name="hmx_manifest"):
    """Validate the versioned HMX kernel manifest object.

    This is deliberately a small schema check rather than a second decision
    engine.  The C++ HMX passes own the values; host code only ensures that a
    missing or malformed diagnostic cannot silently turn into a feature-off
    path.
    """
    if not isinstance(manifest, dict):
        raise ValueError(
            f"{field_name} must be an object, got {type(manifest).__name__}"
        )
    schema = manifest.get("schema")
    if schema != HMX_MANIFEST_SCHEMA:
        raise ValueError(
            f"{field_name}.schema must be {HMX_MANIFEST_SCHEMA!r}, got {schema!r}"
        )
    matmuls = manifest.get("matmuls")
    if not isinstance(matmuls, list):
        raise ValueError(
            f"{field_name}.matmuls must be a list, got {type(matmuls).__name__}"
        )
    for key in ("pack_act_sites", "pack_weight_sites", "unpack_sites"):
        value = manifest.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(
                f"{field_name}.{key} must be a non-negative int, got {value!r}"
            )
    if manifest.get("count_semantics") != "ir_sites":
        raise ValueError(
            f"{field_name}.count_semantics must be 'ir_sites', got "
            f"{manifest.get('count_semantics')!r}"
        )

    required = ("function", "id", "engine", "reason")
    totals = {"pack_act_sites": 0, "pack_weight_sites": 0, "unpack_sites": 0}
    seen_records = set()
    for index, entry in enumerate(matmuls):
        path = f"{field_name}.matmuls[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path} must be an object, got {type(entry).__name__}"
            )
        missing = [key for key in required if key not in entry]
        if missing:
            raise ValueError(f"{path} is missing required field(s): {missing}")

        function = entry["function"]
        if not isinstance(function, str) or not function.strip():
            raise ValueError(f"{path}.function must be a non-empty string")

        entry_id = entry["id"]
        if type(entry_id) is not int or entry_id < 0:
            raise ValueError(
                f"{path}.id must be a non-negative int, got {entry_id!r}"
            )
        record_key = (function, entry_id)
        if record_key in seen_records:
            raise ValueError(f"{path} duplicates function-local id {entry_id}")
        seen_records.add(record_key)

        engine = entry["engine"]
        if engine not in ("hmx", "hvx"):
            raise ValueError(
                f"{path}.engine must be 'hmx' or 'hvx', got {engine!r}"
            )

        reason = entry["reason"]
        if (
            not isinstance(reason, str)
            or not reason
            or _KEBAB_CASE_RE.fullmatch(reason) is None
        ):
            raise ValueError(
                f"{path}.reason must be a non-empty kebab-case code, got {reason!r}"
            )
        if reason not in HMX_MANIFEST_REASONS:
            raise ValueError(f"{path}.reason is not a canonical HMX reason: {reason!r}")

        if (engine == "hmx") != (reason == "selected"):
            raise ValueError(
                f"{path} engine/reason disagree on HMX attribution"
            )
        _validate_hmx_matmul_contract(entry, path, selected=engine == "hmx")
        _validate_hmx_pipeline(entry, path)
        if engine == "hvx":
            hmx_only = (
                "vtcm_budget",
                "vtcm_before",
                "vtcm_peak",
                "blocking",
                "block_m",
                "pipeline_requested",
                "pipeline_selected",
                "pipeline_depth",
                "pipeline_reason",
                "count_semantics",
                "pack_act_sites",
                "pack_weight_sites",
                "unpack_sites",
            )
            present_hmx_only = [key for key in hmx_only if key in entry]
            if present_hmx_only:
                raise ValueError(
                    f"{path} HVX record must not carry HMX-only fields: "
                    f"{present_hmx_only}"
                )
        if engine == "hmx":
            count_fields = (
                "pack_act_sites",
                "pack_weight_sites",
                "unpack_sites",
            )
            missing_counts = [key for key in count_fields if key not in entry]
            if missing_counts:
                raise ValueError(
                    f"{path} is missing bridge count field(s): {missing_counts}"
                )
            for key in count_fields:
                value = entry[key]
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"{path}.{key} must be a non-negative int, got {value!r}"
                    )
                totals[key] += value
            if entry.get("count_semantics") != "ir_sites":
                raise ValueError(
                    f"{path}.count_semantics must be 'ir_sites', got "
                    f"{entry.get('count_semantics')!r}"
                )
            for key in ("vtcm_budget", "vtcm_before", "vtcm_peak", "block_m"):
                if key not in entry:
                    raise ValueError(f"{path} is missing selected field {key}")
                _require_strict_int(
                    entry[key], f"{path}.{key}", minimum=1 if key == "block_m" else 0
                )
            if entry["vtcm_peak"] < entry["vtcm_before"]:
                raise ValueError(f"{path}.vtcm_peak is below vtcm_before")
            if entry["vtcm_peak"] > entry["vtcm_budget"]:
                raise ValueError(f"{path}.vtcm_peak exceeds vtcm_budget")
            blocking = entry.get("blocking")
            if blocking not in ("whole", "m_blocked"):
                raise ValueError(
                    f"{path}.blocking must be 'whole' or 'm_blocked', got {blocking!r}"
                )
            if blocking == "whole":
                if entry["block_m"] != entry["m"]:
                    raise ValueError(f"{path}.block_m must equal m for whole blocking")
            elif entry["block_m"] >= entry["m"] or entry["m"] % entry["block_m"]:
                raise ValueError(f"{path}.block_m is not a proper divisor for m_blocked")
    for key, total in totals.items():
        if manifest[key] != total:
            raise ValueError(
                f"{field_name}.{key} is {manifest[key]}, but per-matmul counts "
                f"sum to {total}"
            )
    return manifest


def validate_hmx_manifest_json(manifest_json, field_name="hmx_manifest"):
    """Validate the JSON text stored in packed kernel metadata."""
    return validate_hmx_manifest(_json_object(manifest_json, field_name), field_name)


def _require_strict_int(value, path, *, minimum=None):
    # bool is an int subclass, but accepting it here would turn malformed JSON
    # values into plausible argument slots and tile dimensions.
    if type(value) is not int:
        raise ValueError(f"{path} must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}, got {value}")
    return value


def _require_positive_int_list(value, path, length):
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{path} must have {length} entries, got {len(value)}")
    for index, item in enumerate(value):
        _require_strict_int(item, f"{path}[{index}]", minimum=1)
    return value


def _validate_hmx_matmul_contract(entry, path, *, selected):
    contract_fields = ("m", "n", "k", "lhs_elem", "rhs_elem", "out_elem")
    present = [field for field in contract_fields if field in entry]
    if present and len(present) != len(contract_fields):
        missing = [field for field in contract_fields if field not in entry]
        raise ValueError(f"{path} has a partial matmul contract; missing {missing}")
    if not present:
        if selected:
            raise ValueError(f"{path} is missing the matmul contract")
        return

    for field in ("m", "n", "k"):
        _require_strict_int(entry[field], f"{path}.{field}", minimum=1)
    for field in ("lhs_elem", "rhs_elem", "out_elem"):
        value = entry[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}.{field} must be a non-empty string")
        if selected and value not in ("f16", "f32"):
            raise ValueError(
                f"{path}.{field} must be 'f16' or 'f32' for HMX, got {value!r}"
            )


def _validate_hmx_pipeline(entry, path):
    fields = (
        "pipeline_requested",
        "pipeline_selected",
        "pipeline_depth",
        "pipeline_reason",
    )
    present = [field for field in fields if field in entry]
    if not present:
        return
    required = ("pipeline_requested", "pipeline_selected", "pipeline_depth")
    missing = [field for field in required if field not in entry]
    if missing:
        raise ValueError(f"{path} has a partial pipeline decision: {missing}")
    _require_strict_int(
        entry["pipeline_requested"], f"{path}.pipeline_requested", minimum=0
    )
    depth = _require_strict_int(
        entry["pipeline_depth"], f"{path}.pipeline_depth", minimum=0
    )
    selected = entry["pipeline_selected"]
    if selected not in ("serial", "staged"):
        raise ValueError(
            f"{path}.pipeline_selected must be 'serial' or 'staged', got {selected!r}"
        )
    if selected == "serial" and depth != 0:
        raise ValueError(f"{path}.pipeline_depth must be 0 for serial selection")
    if selected == "staged" and depth == 0:
        raise ValueError(f"{path}.pipeline_depth must be positive for staged selection")
    reason = entry.get("pipeline_reason")
    if reason is not None and (
        not isinstance(reason, str)
        or not reason
        or _KEBAB_CASE_RE.fullmatch(reason) is None
    ):
        raise ValueError(f"{path}.pipeline_reason must be a kebab-case code")
    if reason is not None and reason not in HMX_PIPELINE_REASONS:
        raise ValueError(
            f"{path}.pipeline_reason is not a canonical pipeline reason: {reason!r}"
        )


def validate_weight_prepack(weight_prepack, field_name="weight_prepack"):
    """Validate the versioned host-side weight pre-pack contract."""
    if not isinstance(weight_prepack, dict):
        raise ValueError(
            f"{field_name} must be an object, got {type(weight_prepack).__name__}"
        )
    missing = [key for key in ("layout", "weights") if key not in weight_prepack]
    if missing:
        raise ValueError(f"{field_name} is missing required field(s): {missing}")

    weights = weight_prepack["weights"]
    if not isinstance(weights, list):
        raise ValueError(
            f"{field_name}.weights must be a list, got {type(weights).__name__}"
        )
    layout = weight_prepack["layout"]
    if layout is not None and not isinstance(layout, dict):
        raise ValueError(
            f"{field_name}.layout must be an object or null, got "
            f"{type(layout).__name__}"
        )
    if weights and not layout:
        raise ValueError(
            f"{field_name}.layout must be a non-empty object when weights is "
            "non-empty"
        )

    required = ("func", "slot", "shape", "crouton", "dtype")
    seen_slots = set()
    functions = set()
    for index, entry in enumerate(weights):
        path = f"{field_name}.weights[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path} must be an object, got {type(entry).__name__}"
            )
        missing = [key for key in required if key not in entry]
        if missing:
            raise ValueError(f"{path} is missing required field(s): {missing}")

        function = entry["func"]
        if not isinstance(function, str) or not function.strip():
            raise ValueError(f"{path}.func must be a non-empty string")

        slot = _require_strict_int(entry["slot"], f"{path}.slot", minimum=0)
        functions.add(function)
        if slot in seen_slots:
            raise ValueError(f"{path}.slot duplicates an earlier slot")
        seen_slots.add(slot)

        _require_positive_int_list(entry["shape"], f"{path}.shape", 2)
        _require_positive_int_list(entry["crouton"], f"{path}.crouton", 5)
        if entry["dtype"] != "f16":
            raise ValueError(f"{path}.dtype must be 'f16', got {entry['dtype']!r}")
    if len(functions) > 1:
        raise ValueError(
            f"{field_name}.weights spans multiple functions {sorted(functions)}; "
            "resident prepack currently supports one function per module"
        )
    return weight_prepack


def _validate_weight_prepack_json(weight_prepack_json, field_name="weight_prepack"):
    return validate_weight_prepack(_json_object(weight_prepack_json, field_name), field_name)


def parse_translation_metadata(metadata_json):
    """Unpack the C++ translation envelope into launcher-facing JSON strings.

    The returned pair is ``(weight_prepack, hmx_manifest)``.  Both values stay
    JSON text because the existing launcher contract consumes the former as a
    string and the latter is consumed by host tooling/diagnostics.  No default
    object is manufactured for a missing field: a stale or malformed envelope
    is an actionable compilation error.
    """
    envelope = _json_object(metadata_json, "translation metadata")
    expected_schema = TRANSLATION_METADATA_SCHEMA
    actual_schema = envelope.get("schema")
    if actual_schema != expected_schema:
        raise ValueError(
            "translation metadata.schema must be "
            f"{expected_schema!r}, got {actual_schema!r}"
        )
    missing = [
        key for key in ("weight_prepack", "hmx_manifest") if key not in envelope
    ]
    if missing:
        raise ValueError(
            f"translation metadata is missing required field(s): {missing}"
        )

    weight_prepack = envelope["weight_prepack"]
    validate_weight_prepack(weight_prepack)
    hmx_manifest = envelope["hmx_manifest"]
    validate_hmx_manifest(hmx_manifest)

    # Re-encode only the two inner objects.  The envelope itself is an internal
    # C++/Python transport detail and must not leak into either consumer.
    return (
        json.dumps(weight_prepack, separators=(",", ":"), ensure_ascii=False),
        json.dumps(hmx_manifest, separators=(",", ":"), ensure_ascii=False),
    )


def apply_translation_metadata(metadata, metadata_json):
    """Validate an envelope and publish its two independent fields."""
    try:
        weight_prepack, hmx_manifest = parse_translation_metadata(metadata_json)
    except ValueError as exc:
        raise RuntimeError(
            "invalid HMX translation metadata returned by the backend: "
            f"{exc}; rebuild the kernel and check the HMX manifest publisher"
        ) from exc
    metadata["weight_prepack"] = weight_prepack
    metadata["hmx_manifest"] = hmx_manifest


def require_pack_metadata_fields(packed):
    """Check the packed metadata shape without decoding its JSON payloads.

    ``pack_metadata`` performs the full semantic validation once, including
    cache-hit construction.  The launcher still checks this small structural
    contract on every call so a manually supplied/stale value produces an
    actionable missing-field error without reparsing JSON on every launch.
    """
    if not isinstance(packed, dict):
        raise RuntimeError(
            "compiled kernel metadata must be a dict of field names, got "
            f"{type(packed).__name__}; the kernel was built by a different "
            "backend version - clear TRITON_CACHE_DIR (tools/hexmlir/env.sh)"
        )
    missing = [key for key in PACK_METADATA_REQUIRED if key not in packed]
    if missing:
        raise RuntimeError(
            f"compiled kernel metadata is missing {missing} (has {sorted(packed)}); "
            "this artifact was written by an older backend - clear "
            "TRITON_CACHE_DIR (tools/hexmlir/env.sh)"
        )
    return packed


def validate_pack_metadata(packed):
    """Validate packed metadata once at CompiledKernel construction."""
    require_pack_metadata_fields(packed)
    try:
        _validate_weight_prepack_json(packed["weight_prepack"])
        validate_hmx_manifest_json(packed["hmx_manifest"])
    except ValueError as exc:
        raise RuntimeError(
            f"invalid compiled kernel metadata: {exc}; clear "
            "TRITON_CACHE_DIR (tools/hexmlir/env.sh) and rebuild the kernel"
        ) from exc
    return packed
