# ===- utils.py -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import hashlib
import json
import os, re
from math import prod
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
# This is the wire identity of the current semantic manifest.  Keep the
# version marker in the protocol value, rather than in Python symbol names.
HMX_MANIFEST_SCHEMA = "hex.hmx.kernel_manifest/v2"
# Bump this semantic ABI token whenever the generated HMX plan/object contract
# changes. It is deliberately independent of the manifest wire spelling.
HMX_SHAPE_TAIL_ABI_VERSION = "shape-tail-abi-2026-09-25-resident-v2-aligned"

HMX_TILE_EDGE = 32
HMX_PLANS = frozenset({"full-hmx", "hmx-tail", "hvx"})
HMX_PLAN_REASONS = {
    "full-hmx": frozenset({"selected-aligned"}),
    "hmx-tail": frozenset({"selected-tail"}),
    "hvx": frozenset(
        {
            "library-call",
            "vtcm-allocator-disabled",
            "non-rank-2",
            "dynamic-shape",
            "unsupported-dtype",
            "min-rows",
            "tile-alignment",
            "unsupported-layout",
            "vtcm-budget",
        }
    ),
}
HMX_MANIFEST_REASONS = frozenset(
    reason for reasons in HMX_PLAN_REASONS.values() for reason in reasons
)
HMX_SHAPE_STATES = frozenset(
    {"static", "partially-dynamic", "dynamic", "unavailable"}
)
HMX_DIMENSION_KINDS = frozenset({"static", "dynamic"})
HMX_DIMENSION_SYMBOLS = frozenset({"m", "n", "k"})
HMX_TAIL_POLICIES = {
    "k": "zero-pad-both-operands",
    "mn": "padded-edge-tile-bounded-store",
}
HMX_WORKSPACE_CLASSES = frozenset({"runtime-internal", "resident-single-instance"})
HMX_GRID_POLICIES = frozenset({"single-instance", "legacy-runtime"})
HMX_VTCM_ACCOUNTING = frozenset({"bridge-only"})
HMX_WEIGHT_BINDING_KINDS = frozenset(
    {"argument-slot", "compile-time-constant", "internal-value"}
)
HMX_WEIGHT_POLICIES = frozenset({"resident-prepack", "device-pack"})
HMX_WEIGHT_POLICY_REASONS = {
    "resident-prepack": frozenset({"eligible-aligned-f16", "eligible-b2-n-slice"}),
    "device-pack": frozenset(
        {
            "tail-consumer",
            "f32-source",
            "unproven-offset",
            "incompatible-consumers",
            "prepack-disabled",
        }
    ),
}
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
        "tail-peeled-edge",
    }
)
_KEBAB_CASE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SHA256_FINGERPRINT_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_I64 = (1 << 63) - 1


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _json_object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _json_object(value, field_name):
    """Decode one strict JSON object and turn every failure into an error."""
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a JSON string, got {type(value).__name__}"
        )
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(
            f"{field_name} must decode to an object, got {type(decoded).__name__}"
        )
    return decoded


def _require_object(value, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object, got {type(value).__name__}")
    return value


def _require_exact_fields(value, path, required, optional=()):
    """Require a closed object shape; unknown fields are never ignored."""
    _require_object(value, path)
    required = tuple(required)
    allowed = set(required) | set(optional)
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{path} is missing required field(s): {missing}")
    unknown = [key for key in value if key not in allowed]
    unknown.sort(key=str)
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {unknown}")
    return value


def _require_string(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_strict_int(value, path, *, minimum=None):
    # bool is an int subclass, but accepting it here would turn malformed JSON
    # values into plausible argument slots and tile dimensions.
    if type(value) is not int:
        raise ValueError(f"{path} must be an int, got {type(value).__name__}")
    if value > _MAX_I64:
        raise ValueError(f"{path} exceeds the signed 64-bit range")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}, got {value}")
    return value


def _require_nonnegative_int(value, path):
    return _require_strict_int(value, path, minimum=0)


def _require_positive_int_list(value, path, length):
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list, got {type(value).__name__}")
    if len(value) != length:
        raise ValueError(f"{path} must have {length} entries, got {len(value)}")
    for index, item in enumerate(value):
        _require_strict_int(item, f"{path}[{index}]", minimum=1)
    return value


def _require_code(value, path, allowed):
    if (
        not isinstance(value, str)
        or not value
        or _KEBAB_CASE_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{path} must be a non-empty kebab-case code, got {value!r}")
    if value not in allowed:
        raise ValueError(f"{path} is not a canonical code: {value!r}")
    return value


def _validate_dimension(value, path, axis):
    _require_object(value, path)
    kind = value.get("kind")
    if kind == "static":
        _require_exact_fields(value, path, ("kind", "value"))
        _require_strict_int(value["value"], f"{path}.value", minimum=1)
    elif kind == "dynamic":
        _require_exact_fields(value, path, ("kind", "symbol"))
        symbol = value["symbol"]
        if not isinstance(symbol, str) or symbol not in HMX_DIMENSION_SYMBOLS or symbol != axis:
            raise ValueError(
                f"{path}.symbol must name this axis ({axis!r}), got {symbol!r}"
            )
    else:
        raise ValueError(
            f"{path}.kind must be 'static' or 'dynamic', got {kind!r}"
        )
    return kind


def _derive_shape_state(logical, path):
    if logical is None:
        return "unavailable"
    _require_object(logical, path)
    kinds = {
        axis: _validate_dimension(logical.get(axis), f"{path}.{axis}", axis)
        for axis in ("m", "n", "k")
    }
    static_count = sum(kind == "static" for kind in kinds.values())
    if static_count == 3:
        return "static"
    if static_count == 0:
        return "dynamic"
    return "partially-dynamic"


def _validate_logical(entry, path, reason, plan):
    shape_state = entry.get("shape_state")
    if not isinstance(shape_state, str) or shape_state not in HMX_SHAPE_STATES:
        raise ValueError(
            f"{path}.shape_state is not canonical: {shape_state!r}"
        )
    if "logical" not in entry:
        raise ValueError(f"{path} is missing required field(s): ['logical']")
    logical = entry["logical"]
    if logical is None:
        if reason not in ("library-call", "non-rank-2"):
            raise ValueError(
                f"{path}.logical may be null only for library-call or non-rank-2"
            )
        if shape_state != "unavailable":
            raise ValueError(
                f"{path}.shape_state must be 'unavailable' when logical is null"
            )
        return shape_state, None
    if not isinstance(logical, dict):
        raise ValueError(f"{path}.logical must be an object or null")
    _require_exact_fields(logical, f"{path}.logical", ("m", "n", "k"))
    derived = _derive_shape_state(logical, f"{path}.logical")
    if shape_state != derived:
        raise ValueError(
            f"{path}.shape_state {shape_state!r} disagrees with logical dimensions "
            f"(derived {derived!r})"
        )
    if plan != "hvx" and derived != "static":
        raise ValueError(f"{path} HMX plan requires a fully static logical shape")
    if reason == "dynamic-shape" and derived == "static":
        raise ValueError(f"{path} dynamic-shape reason requires a dynamic dimension")
    static_values = None
    if derived == "static":
        static_values = {
            axis: logical[axis]["value"] for axis in ("m", "n", "k")
        }
    return shape_state, static_values


def _validate_dtypes(value, path, *, hmx):
    _require_object(value, path)
    required = {"lhs", "rhs", "out"}
    if hmx:
        required.add("crouton")
    _require_exact_fields(value, path, tuple(sorted(required)))
    for field in sorted(required):
        _require_string(value[field], f"{path}.{field}")
    if hmx:
        for field in ("lhs", "rhs", "out"):
            if value[field] not in ("f16", "f32"):
                raise ValueError(
                    f"{path}.{field} must be 'f16' or 'f32' for HMX, "
                    f"got {value[field]!r}"
                )
        if value["crouton"] != "f16":
            raise ValueError(
                f"{path}.crouton must be 'f16' for HMX, got {value['crouton']!r}"
            )
    return value


def _validate_axis_values(value, path, *, positive):
    _require_exact_fields(value, path, ("m", "n", "k"))
    result = {}
    for axis in ("m", "n", "k"):
        result[axis] = _require_strict_int(
            value[axis], f"{path}.{axis}", minimum=1 if positive else 0
        )
    return result


def _validate_hmx_shape(entry, path, plan, logical_values):
    if plan == "hvx":
        return
    if logical_values is None:
        raise ValueError(f"{path} HMX shape requires static logical dimensions")
    padded = _validate_axis_values(entry["padded"], f"{path}.padded", positive=True)
    full = _validate_axis_values(entry["full"], f"{path}.full", positive=False)
    tail = _validate_axis_values(entry["tail"], f"{path}.tail", positive=False)
    for axis in ("m", "n", "k"):
        if padded[axis] % HMX_TILE_EDGE:
            raise ValueError(
                f"{path}.padded.{axis} must be a multiple of {HMX_TILE_EDGE}"
            )
        if full[axis] % HMX_TILE_EDGE:
            raise ValueError(
                f"{path}.full.{axis} must be a multiple of {HMX_TILE_EDGE}"
            )
        if tail[axis] >= HMX_TILE_EDGE:
            raise ValueError(
                f"{path}.tail.{axis} must be < {HMX_TILE_EDGE}"
            )
        expected_full = (
            logical_values[axis] // HMX_TILE_EDGE
        ) * HMX_TILE_EDGE
        expected_padded = (
            (logical_values[axis] + HMX_TILE_EDGE - 1) // HMX_TILE_EDGE
        ) * HMX_TILE_EDGE
        if full[axis] != expected_full:
            raise ValueError(
                f"{path}.full.{axis} disagrees with the logical extent"
            )
        if tail[axis] != logical_values[axis] - expected_full:
            raise ValueError(f"{path}.tail.{axis} disagrees with the logical extent")
        if padded[axis] != expected_padded:
            raise ValueError(
                f"{path}.padded.{axis} disagrees with align-up arithmetic"
            )
        if padded[axis] != full[axis] + tail[axis] + (
            HMX_TILE_EDGE - tail[axis] if tail[axis] else 0
        ):
            raise ValueError(f"{path}.padded.{axis} has inconsistent padding")

    if plan == "full-hmx":
        if any(tail.values()):
            raise ValueError(f"{path} full-hmx must have an all-zero tail")
        if padded != logical_values or full != logical_values:
            raise ValueError(
                f"{path} full-hmx padded/full extents must equal logical extents"
            )
    elif plan == "hmx-tail":
        if not any(tail.values()):
            raise ValueError(f"{path} hmx-tail must have at least one tail extent")
        _require_exact_fields(entry["tail_policy"], f"{path}.tail_policy", ("k", "mn"))
        for field, expected in HMX_TAIL_POLICIES.items():
            if entry["tail_policy"][field] != expected:
                raise ValueError(
                    f"{path}.tail_policy.{field} must be {expected!r}, "
                    f"got {entry['tail_policy'][field]!r}"
                )
    else:
        raise ValueError(f"{path}.plan is not an HMX plan: {plan!r}")


def _validate_workspace(entry, path, plan):
    workspace_class = entry.get("workspace_class")
    grid_policy = entry.get("grid_policy")
    if not isinstance(workspace_class, str) or workspace_class not in HMX_WORKSPACE_CLASSES:
        raise ValueError(
            f"{path}.workspace_class is not canonical: {workspace_class!r}"
        )
    if not isinstance(grid_policy, str) or grid_policy not in HMX_GRID_POLICIES:
        raise ValueError(f"{path}.grid_policy is not canonical: {grid_policy!r}")
    if plan == "hmx-tail" and grid_policy != "single-instance":
        raise ValueError(f"{path} hmx-tail requires grid_policy='single-instance'")
    if plan == "full-hmx":
        expected = (
            "single-instance"
            if workspace_class == "resident-single-instance"
            else "legacy-runtime"
        )
        if grid_policy != expected:
            raise ValueError(
                f"{path} grid_policy {grid_policy!r} disagrees with workspace_class "
                f"{workspace_class!r}"
            )
    vtcm_accounting = entry.get("vtcm_accounting")
    if not isinstance(vtcm_accounting, str) or vtcm_accounting not in HMX_VTCM_ACCOUNTING:
        raise ValueError(
            f"{path}.vtcm_accounting must be 'bridge-only', got "
            f"{entry.get('vtcm_accounting')!r}"
        )
    budget = _require_nonnegative_int(
        entry["vtcm_budget_bytes"], f"{path}.vtcm_budget_bytes"
    )
    before = _require_nonnegative_int(
        entry["vtcm_before_bytes"], f"{path}.vtcm_before_bytes"
    )
    peak = _require_nonnegative_int(
        entry["vtcm_bridge_peak_bytes"], f"{path}.vtcm_bridge_peak_bytes"
    )
    if peak < before:
        raise ValueError(f"{path}.vtcm_bridge_peak_bytes is below vtcm_before_bytes")
    if peak > budget:
        raise ValueError(f"{path}.vtcm_bridge_peak_bytes exceeds vtcm_budget_bytes")
    return budget, before, peak


def _validate_execution(entry, path, logical_values):
    execution = _require_object(entry["execution"], f"{path}.execution")
    _require_exact_fields(
        execution, f"{path}.execution", ("blocking", "block_m", "pipeline", "bridge_counts")
    )
    blocking = execution["blocking"]
    if not isinstance(blocking, str) or blocking not in ("whole", "m_blocked"):
        raise ValueError(
            f"{path}.execution.blocking must be 'whole' or 'm_blocked', "
            f"got {blocking!r}"
        )
    block_m = _require_strict_int(
        execution["block_m"], f"{path}.execution.block_m", minimum=1
    )
    m = logical_values["m"]
    if blocking == "whole":
        if block_m != m:
            raise ValueError(f"{path}.execution.block_m must equal logical M for whole")
    elif block_m >= m or m % block_m:
        raise ValueError(
            f"{path}.execution.block_m must be a proper divisor for m_blocked"
        )

    _validate_hmx_pipeline(execution, f"{path}.execution")
    bridge_counts = _require_object(
        execution["bridge_counts"], f"{path}.execution.bridge_counts"
    )
    _require_exact_fields(
        bridge_counts,
        f"{path}.execution.bridge_counts",
        ("pack_act_sites", "pack_weight_sites", "unpack_sites", "count_semantics"),
    )
    for field in ("pack_act_sites", "pack_weight_sites", "unpack_sites"):
        _require_nonnegative_int(bridge_counts[field], f"{path}.execution.bridge_counts.{field}")
    if bridge_counts["count_semantics"] != "ir_sites":
        raise ValueError(
            f"{path}.execution.bridge_counts.count_semantics must be 'ir_sites'"
        )
    return bridge_counts


def _validate_hmx_pipeline(entry, path):
    pipeline = _require_object(entry.get("pipeline"), f"{path}.pipeline")
    _require_exact_fields(
        pipeline, f"{path}.pipeline", ("requested", "selected", "depth"), ("reason",)
    )
    _require_nonnegative_int(pipeline["requested"], f"{path}.pipeline.requested")
    depth = _require_nonnegative_int(pipeline["depth"], f"{path}.pipeline.depth")
    selected = pipeline["selected"]
    if not isinstance(selected, str) or selected not in ("serial", "staged"):
        raise ValueError(
            f"{path}.pipeline.selected must be 'serial' or 'staged', got {selected!r}"
        )
    if selected == "serial" and depth != 0:
        raise ValueError(f"{path}.pipeline.depth must be 0 for serial selection")
    if selected == "staged" and depth == 0:
        raise ValueError(f"{path}.pipeline.depth must be positive for staged selection")
    if "reason" in pipeline:
        _require_code(pipeline["reason"], f"{path}.pipeline.reason", HMX_PIPELINE_REASONS)


def _validate_hmx_matmul_contract(entry, path, *, selected):
    """Validate the current tagged logical-shape and dtype contract."""
    plan = "full-hmx" if selected else entry.get("plan", "hvx")
    reason = entry.get("reason", "")
    _, logical_values = _validate_logical(entry, path, reason, plan)
    _validate_dtypes(entry.get("dtypes"), f"{path}.dtypes", hmx=selected)
    return logical_values


def _validate_weight_binding(value, path):
    _require_object(value, path)
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in HMX_WEIGHT_BINDING_KINDS:
        raise ValueError(f"{path}.kind is not canonical: {kind!r}")
    if kind == "argument-slot":
        _require_exact_fields(value, path, ("kind", "policy_ref"))
        ref = _require_object(value["policy_ref"], f"{path}.policy_ref")
        _require_exact_fields(ref, f"{path}.policy_ref", ("function", "slot"))
        _require_string(ref["function"], f"{path}.policy_ref.function")
        _require_nonnegative_int(ref["slot"], f"{path}.policy_ref.slot")
        return kind, (ref["function"], ref["slot"])
    _require_exact_fields(value, path, ("kind",))
    return kind, None


def _validate_weight_policies(value, path):
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list, got {type(value).__name__}")
    result = {}
    resident_functions = set()
    for index, policy in enumerate(value):
        policy_path = f"{path}[{index}]"
        _require_exact_fields(
            policy,
            policy_path,
            ("function", "slot", "policy", "reason", "consumers"),
        )
        function = _require_string(policy["function"], f"{policy_path}.function")
        slot = _require_nonnegative_int(policy["slot"], f"{policy_path}.slot")
        name = policy["policy"]
        if not isinstance(name, str) or name not in HMX_WEIGHT_POLICIES:
            raise ValueError(
                f"{policy_path}.policy is not canonical: {name!r}"
            )
        reason = policy["reason"]
        if not isinstance(reason, str) or reason not in HMX_WEIGHT_POLICY_REASONS[name]:
            raise ValueError(
                f"{policy_path}.reason {reason!r} is not valid for policy {name!r}"
            )
        consumers = policy["consumers"]
        if not isinstance(consumers, list):
            raise ValueError(
                f"{policy_path}.consumers must be a list, got {type(consumers).__name__}"
            )
        seen_consumers = set()
        for consumer_index, consumer in enumerate(consumers):
            consumer = _require_nonnegative_int(
                consumer, f"{policy_path}.consumers[{consumer_index}]"
            )
            if consumer in seen_consumers:
                raise ValueError(
                    f"{policy_path}.consumers contains duplicate id {consumer}"
                )
            seen_consumers.add(consumer)
        key = (function, slot)
        if key in result:
            raise ValueError(
                f"{policy_path} duplicates slot policy for function {function!r}, "
                f"slot {slot}"
            )
        if name == "resident-prepack":
            resident_functions.add(function)
        result[key] = policy
    if len(resident_functions) > 1:
        raise ValueError(
            f"{path} allows resident-prepack for only one function"
        )
    return result


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_sha256(value):
    """Return the canonical compact JSON SHA256 used by manifest plans."""
    encoded = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _plan_fingerprint_payload(record, weight_policy=None):
    # The C++ producer hashes the complete semantic record (minus the digest
    # itself), then adds the referenced slot policy.  Copying the validated
    # record here keeps derived fields and the wire contract in the identity.
    payload = {"schema": HMX_MANIFEST_SCHEMA}
    for key, value in record.items():
        if key != "plan_fingerprint":
            payload[key] = value
    if (
        record["plan"] != "hvx"
        and record["weight_binding"]["kind"] == "argument-slot"
        and weight_policy is not None
    ):
        payload["weight_policy"] = weight_policy
    return payload


def compute_hmx_plan_fingerprint(record, weight_policy=None):
    """Compute the stable fingerprint for one validated plan record."""
    return canonical_json_sha256(_plan_fingerprint_payload(record, weight_policy))


def _validate_record(entry, path):
    _require_object(entry, path)
    base_fields = (
        "function",
        "id",
        "plan",
        "reason",
        "shape_state",
        "logical",
        "dtypes",
        "plan_fingerprint",
    )
    plan = entry.get("plan")
    if not isinstance(plan, str) or plan not in HMX_PLANS:
        raise ValueError(f"{path}.plan is not canonical: {plan!r}")
    if plan == "hvx":
        allowed = base_fields
    elif plan == "hmx-tail":
        allowed = base_fields + (
            "padded",
            "full",
            "tail",
            "tail_policy",
            "layout",
            "workspace_class",
            "grid_policy",
            "vtcm_accounting",
            "vtcm_budget_bytes",
            "vtcm_before_bytes",
            "vtcm_bridge_peak_bytes",
            "execution",
            "weight_binding",
        )
    else:
        allowed = base_fields + (
            "padded",
            "full",
            "tail",
            "layout",
            "workspace_class",
            "grid_policy",
            "vtcm_accounting",
            "vtcm_budget_bytes",
            "vtcm_before_bytes",
            "vtcm_bridge_peak_bytes",
            "execution",
            "weight_binding",
        )
    _require_exact_fields(entry, path, allowed)

    function = _require_string(entry["function"], f"{path}.function")
    record_id = _require_nonnegative_int(entry["id"], f"{path}.id")
    reason = _require_code(entry["reason"], f"{path}.reason", HMX_MANIFEST_REASONS)
    if reason not in HMX_PLAN_REASONS[plan]:
        raise ValueError(
            f"{path} plan {plan!r} and reason {reason!r} are not a canonical pair"
        )
    shape_state, logical_values = _validate_logical(entry, path, reason, plan)
    _validate_dtypes(entry["dtypes"], f"{path}.dtypes", hmx=plan != "hvx")

    counts = None
    binding = None
    if plan != "hvx":
        _validate_hmx_shape(entry, path, plan, logical_values)
        if entry["layout"] != "row-major-inner-contiguous":
            raise ValueError(
                f"{path}.layout must be 'row-major-inner-contiguous', "
                f"got {entry['layout']!r}"
            )
        _validate_workspace(entry, path, plan)
        counts = _validate_execution(entry, path, logical_values)
        binding = _validate_weight_binding(entry["weight_binding"], f"{path}.weight_binding")
    fingerprint = entry["plan_fingerprint"]
    if not isinstance(fingerprint, str) or _SHA256_FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise ValueError(
            f"{path}.plan_fingerprint must be a lowercase sha256 digest"
        )
    return {
        "path": path,
        "entry": entry,
        "function": function,
        "id": record_id,
        "plan": plan,
        "reason": reason,
        "logical_values": logical_values,
        "counts": counts,
        "binding": binding,
    }


def validate_hmx_manifest(manifest, field_name="hmx_manifest"):
    """Validate the current semantic HMX manifest object.

    The manifest is an execution-plan contract, not a second attribution
    engine.  Its producer owns the decision; this boundary checks the closed
    wire shape, derived arithmetic, and the integrity fingerprint before a
    stale or malformed object can reach the launcher.
    """
    _require_object(manifest, field_name)
    schema = manifest.get("schema")
    if schema != HMX_MANIFEST_SCHEMA:
        raise ValueError(
            f"{field_name}.schema must be {HMX_MANIFEST_SCHEMA!r}, got {schema!r}"
        )
    _require_exact_fields(
        manifest,
        field_name,
        (
            "schema",
            "matmuls",
            "weight_policies",
            "pack_act_sites",
            "pack_weight_sites",
            "unpack_sites",
            "count_semantics",
        ),
    )
    matmuls = manifest["matmuls"]
    if not isinstance(matmuls, list):
        raise ValueError(
            f"{field_name}.matmuls must be a list, got {type(matmuls).__name__}"
        )
    policies = _validate_weight_policies(
        manifest["weight_policies"], f"{field_name}.weight_policies"
    )
    totals = {"pack_act_sites": 0, "pack_weight_sites": 0, "unpack_sites": 0}
    for field in totals:
        _require_nonnegative_int(manifest[field], f"{field_name}.{field}")
    if manifest["count_semantics"] != "ir_sites":
        raise ValueError(
            f"{field_name}.count_semantics must be 'ir_sites', got "
            f"{manifest['count_semantics']!r}"
        )

    records = []
    by_key = {}
    for index, entry in enumerate(matmuls):
        path = f"{field_name}.matmuls[{index}]"
        info = _validate_record(entry, path)
        key = (info["function"], info["id"])
        if key in by_key:
            raise ValueError(f"{path} duplicates function-local id {info['id']}")
        by_key[key] = info
        records.append(info)
        if info["plan"] != "hvx":
            for field in totals:
                totals[field] += info["counts"][field]

    for info in records:
        binding = info["binding"]
        if binding is None or binding[0] != "argument-slot":
            continue
        _, reference = binding
        if reference[0] != info["function"]:
            raise ValueError(
                f"{info['path']} weight policy reference must use the record function"
            )
        policy = policies.get(reference)
        if policy is None:
            raise ValueError(
                f"{field_name}.weight_policies has no entry for "
                f"function {reference[0]!r}, slot {reference[1]}"
            )
        if info["id"] not in policy["consumers"]:
            raise ValueError(
                f"{field_name}.weight_policies entry for slot {reference[1]} "
                f"does not list consumer {info['id']}"
            )
        info["weight_policy"] = policy

    for key, policy in policies.items():
        if not policy["consumers"]:
            raise ValueError(
                f"{field_name}.weight_policies entry for function {key[0]!r}, "
                f"slot {key[1]} has no consumers"
            )
        for consumer in policy["consumers"]:
            info = by_key.get((key[0], consumer))
            if info is None or info["plan"] == "hvx":
                raise ValueError(
                    f"{field_name}.weight_policies consumer {consumer} does not name "
                    "an HMX plan record"
                )
            if info["binding"] is None or info["binding"][0] != "argument-slot":
                raise ValueError(
                    f"{field_name}.weight_policies consumer {consumer} is not bound "
                    "to an argument slot"
                )
            if info["binding"][1] != key:
                raise ValueError(
                    f"{field_name}.weight_policies consumer {consumer} references a "
                    "different slot policy"
                )
    for info in records:
        policy = info.get("weight_policy")
        expected = compute_hmx_plan_fingerprint(info["entry"], policy)
        if info["entry"]["plan_fingerprint"] != expected:
            raise ValueError(
                f"{field_name}.matmuls plan_fingerprint does not match its "
                "canonical semantic plan"
            )

    for field, total in totals.items():
        if manifest[field] != total:
            raise ValueError(
                f"{field_name}.{field} is {manifest[field]}, but per-matmul counts "
                f"sum to {total}"
            )
    return manifest


def validate_hmx_manifest_json(manifest_json, field_name="hmx_manifest"):
    """Validate the JSON text stored in packed kernel metadata."""
    return validate_hmx_manifest(_json_object(manifest_json, field_name), field_name)


def enforce_hmx_launch_contract(
    manifest_json: str, launch_grid: tuple[int, int, int], field_name: str = "hmx_manifest"
):
    """Validate the manifest and enforce its v2 single-instance launch promise.

    v2 deliberately carries bridge-only VTCM facts, not a kernel-wide peak.  It
    can nevertheless prove the launch-shape part of the contract: a tail plan
    (and a resident full-HMX workspace) is single-instance, while the legacy
    runtime-internal full plan retains the launcher's existing grid behavior.
    The check lives beside the strict consumer so every launcher entry point can
    apply the same rule before it creates a wrapper or touches the device.
    """
    manifest = validate_hmx_manifest_json(manifest_json, field_name)
    if not isinstance(launch_grid, (tuple, list)) or len(launch_grid) != 3:
        raise ValueError(
            f"{field_name} launch grid must contain exactly three dimensions"
        )
    if any(type(size) is not int or size < 1 for size in launch_grid):
        raise ValueError(
            f"{field_name} launch grid dimensions must be positive integers"
        )
    if prod(launch_grid) != 1:
        single_instance = [
            record
            for record in manifest["matmuls"]
            if record["plan"] != "hvx"
            and record["grid_policy"] == "single-instance"
        ]
        if single_instance:
            plans = sorted({record["plan"] for record in single_instance})
            raise ValueError(
                f"{field_name} requires a single program instance for "
                f"{', '.join(plans)}; got launch grid {tuple(launch_grid)} "
                f"(product={prod(launch_grid)})"
            )
    return manifest


def validate_weight_prepack(weight_prepack, field_name="weight_prepack"):
    """Validate the existing host-side weight pre-pack contract."""
    _require_exact_fields(weight_prepack, field_name, ("layout", "weights"))

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
        _require_exact_fields(entry, path, required)
        function = _require_string(entry["func"], f"{path}.func")
        slot = _require_nonnegative_int(entry["slot"], f"{path}.slot")
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


def _validate_manifest_weight_prepack(manifest, weight_prepack):
    """Keep slot policies and the independent host image contract closed."""
    policies = {
        (policy["function"], policy["slot"]): policy
        for policy in manifest["weight_policies"]
    }
    entries = {
        (entry["func"], entry["slot"]): entry
        for entry in weight_prepack["weights"]
    }
    for key, policy in policies.items():
        entry = entries.get(key)
        if policy["policy"] == "resident-prepack":
            if entry is None:
                raise ValueError(
                    "resident-prepack policy has no matching weight_prepack entry"
                )
        elif entry is not None:
            raise ValueError(
                "device-pack policy must not publish a weight_prepack entry"
            )
    for key in entries:
        policy = policies.get(key)
        if policy is None or policy["policy"] != "resident-prepack":
            raise ValueError(
                "weight_prepack entry is not referenced by a resident-prepack policy"
            )


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
    _require_exact_fields(
        envelope, "translation metadata", ("schema", "weight_prepack", "hmx_manifest")
    )

    weight_prepack = envelope["weight_prepack"]
    validate_weight_prepack(weight_prepack)
    hmx_manifest = envelope["hmx_manifest"]
    validate_hmx_manifest(hmx_manifest)
    _validate_manifest_weight_prepack(hmx_manifest, weight_prepack)

    # Re-encode only the two inner objects.  The envelope itself is an internal
    # C++/Python transport detail and must not leak into either consumer.
    return (
        _canonical_json(weight_prepack),
        _canonical_json(hmx_manifest),
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
