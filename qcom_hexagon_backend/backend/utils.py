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
            raise ValueError(f"Unsupported input type {type(inp).__name__}")
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
    "hmx_record",
)

# The default translation envelope.  It carries the v2 execution manifest only;
# the record-only v3 child belongs to `hex.hmx.translation/v2` below.
TRANSLATION_METADATA_SCHEMA = "hex.hmx.translation/v1"
# This is the wire identity of the current semantic manifest.  Keep the
# version marker in the protocol value, rather than in Python symbol names.
HMX_MANIFEST_SCHEMA = "hex.hmx.kernel_manifest/v2"
# Bump this semantic ABI token whenever the generated HMX plan/object contract
# changes. It is deliberately independent of the manifest wire spelling.
HMX_SHAPE_TAIL_ABI_VERSION = "shape-tail-abi-2026-09-25-resident-v2-aligned"

# ---------------------------------------------------------------------------
# Record-only v3 contract (hex.hmx.kernel_manifest/v3)
# ---------------------------------------------------------------------------
# A second, independent wire schema.  It records analysis facts, evidence and
# four separate proof statuses; it is not an executable upgrade of v2 and it
# grants no HMX/tail admission.  There is deliberately no `v2_to_v3` path here:
# the C++ producer builds the document from the compile-time facts and the P1.5
# diagnostic sidecars, and this module only consumes it.
#
# `hex.hmx.translation/v1` (the default) carries the v2 manifest only.
# `hex.hmx.translation/v2` carries the byte-identical v2 execution manifest plus
# a separate `hmx_record` v3 child.  The two envelopes are closed identities: a
# v1 consumer rejects a v2 envelope and vice versa, and no code path "tries v2
# and then guesses v3".
HMX_RECORD_SCHEMA = "hex.hmx.kernel_manifest/v3"
# `TRANSLATION_METADATA_SCHEMA` below is the v1 envelope; it is the historical
# name of the default and stays the single spelling of that wire identity.
HMX_TRANSLATION_V1_SCHEMA = TRANSLATION_METADATA_SCHEMA
HMX_TRANSLATION_V2_SCHEMA = "hex.hmx.translation/v2"

# The one spelling of "this envelope carries no record child".  A v1 envelope has
# no `hmx_record` key at all, but packed metadata crosses a JSON boundary, so the
# published value needs a name of its own rather than a bare `""` at each use.
HMX_RECORD_ABSENT = ""

# Fixed protocol values, not runtime switches.
HMX_RECORD_MODE = "record-only"
HMX_RECORD_ADMISSION = "not-authorized"
HMX_RECORD_UNIT_BYTES = "bytes"
HMX_RECORD_BASIS_REQUESTED = "compile-time-requested"
HMX_RECORD_BASIS_ALLOCATOR = "allocator-model"
HMX_RECORD_BASIS_OBSERVED = "runtime-observation"
HMX_RECORD_OBSERVATION_SCOPE = "process-high-water"
HMX_RECORD_GRID_POLICY = "single-instance"
HMX_RECORD_GRID_REQUIRED_PRODUCT = 1
HMX_RECORD_INVOCATIONS = 1
HMX_RECORD_RESIDENT_SCOPE = "process-floor"
# The record's one declared failure behavior.  The decision doc's illustrative
# shape also listed `on_unproven_proof` and `on_descriptor_mismatch`; a
# record-only document makes no such decision -- it never re-selects a plan and
# never resolves a descriptor -- so publishing them would advertise a capability
# the record does not have.  See HmxRecordV3.h for the same reasoning.
HMX_RECORD_FALLBACK = {"on_malformed_record": "reject-v3-record"}
HMX_RECORD_SHAPE_STATES = frozenset({"static", "partially-dynamic", "dynamic"})
HMX_RECORD_SPECIALIZATIONS = frozenset({"upstream-static", "upstream-only"})
# One total status would hide exactly the gaps the four axes exist to show.
HMX_RECORD_PROOF_STATUSES = frozenset({"complete", "incomplete", "not-proven"})
HMX_RECORD_PROOF_AXES = ("liveness", "allocator", "grid", "resident")
# Bump when the C++ proof/observation producer's accepted evidence changes.  It is
# part of the cache identity, so a producer that reads different sidecar facts
# cannot reuse a previous build's objects.
HMX_RECORD_PROOF_PRODUCER_ABI = "hmx-v3-proof-producer-abi-2026-09-26-v1"
# The accounting mode of the default envelope.  v2 publishes bridge-only VTCM
# facts and nothing else; the record-only mode is a different mode, not a
# different spelling of this one.
HMX_ACCOUNTING_MODE = "bridge-only"
HMX_RECORD_ACCOUNTING_MODE = "v3-record-only"

HMX_TILE_EDGE = 32
HMX_PLANS = frozenset({"full-hmx", "hmx-tail", "hvx"})
# The host's counterpart of `kMinimumHmxRows` in
# lib/Dialect/Hmx/Transforms/HmxManifest.cpp.  The compiler refuses to *emit* an
# HMX plan with logical M <= this; the host used to accept one, so a stale or
# hand-edited cached artifact could declare a shape the compiler would never
# publish and still pass every host gate.  Pinned from both sides by
# test/test_hmx_manifest_minimum_rows.py.
HMX_MINIMUM_ROWS = 4
HMX_NON_HVX_PLANS = frozenset({"full-hmx", "hmx-tail"})
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
HMX_WORKSPACE_CLASSES = frozenset({"runtime-internal", "resident"})
HMX_GRID_POLICIES = frozenset({"single-instance", "legacy-runtime"})
HMX_VTCM_ACCOUNTING = frozenset({"bridge-only"})
HMX_WEIGHT_BINDING_KINDS = frozenset(
    {"argument-slot", "compile-time-constant", "internal-value"}
)
#: The two placements of a host pre-packed weight image: the VTCM pool, or the
#: permanent DDR mirror the compiler falls back to when the pool cannot hold
#: the weight (`hmx.weight_prepack` entry's `location` field, which the C++
#: producer writes and this validator closes).
HMX_WEIGHT_LOCATIONS = frozenset({"vtcm", "ddr"})
HMX_WEIGHT_POLICIES = frozenset(
    {"resident-prepack", "resident-prepack-ddr", "device-pack"}
)
#: What makes a weight packable does not depend on where its image ends up, so
#: the two resident policies share one reason vocabulary (mirrors the C++
#: `isResidentPrepackPolicy`).
HMX_WEIGHT_POLICY_REASONS = {
    "resident-prepack": frozenset(
        {"eligible-aligned-f16", "eligible-quantized-f32", "eligible-b2-n-slice"}
    ),
    "resident-prepack-ddr": frozenset(
        {"eligible-aligned-f16", "eligible-quantized-f32", "eligible-b2-n-slice"}
    ),
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
        # Must mirror HmxManifest.cpp:937-944 exactly. That check admits an
        # unavailable logical shape for these three reasons; a fourth added on
        # the C++ side without adding it here turns every such kernel into a
        # launch-time rejection, and the cross-language gate compares reason
        # vocabularies rather than conditional rules so it cannot catch it.
        if reason not in ("library-call", "non-rank-2", "unsupported-layout"):
            raise ValueError(
                f"{path}.logical may be null only for library-call, non-rank-2 "
                f"or unsupported-layout"
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

    if plan in HMX_NON_HVX_PLANS and logical_values["m"] <= HMX_MINIMUM_ROWS:
        # Mirrors kMinimumHmxRows on the producer side.  The host is the last
        # gate before a manifest reaches the launcher, so a record the compiler
        # could not have produced has to stop here rather than at a device.
        raise ValueError(
            f"{path} HMX plans require logical M > {HMX_MINIMUM_ROWS}, "
            f"got {logical_values['m']}"
        )
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
    # A full-HMX workspace keeps the plan's legacy-runtime grid policy
    # regardless of class: a resident workspace is keyed by the caller's flat
    # program id at runtime, so it is sound under grid>1 the same way the
    # runtime-internal one always was.
    if plan == "full-hmx" and grid_policy != "legacy-runtime":
        raise ValueError(
            f"{path} full-hmx requires grid_policy='legacy-runtime', got "
            f"{grid_policy!r}"
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
        pipeline,
        f"{path}.pipeline",
        ("requested", "selected", "depth"),
        ("reason", "budget_depth"),
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
    if "budget_depth" in pipeline:
        # 2026-10-02: the depth the VTCM budget allowed, which is what makes a
        # clamped ring readable. Optional because a manifest that predates the
        # field is still valid -- it just does not say why the depth is what it
        # is. A budget below the selected depth would mean the pass chose deeper
        # than the budget permitted, which cannot happen.
        budget_depth = _require_nonnegative_int(
            pipeline["budget_depth"], f"{path}.pipeline.budget_depth"
        )
        if budget_depth < depth:
            raise ValueError(
                f"{path}.pipeline.budget_depth {budget_depth} is below "
                f"{path}.pipeline.depth {depth}"
            )


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
        if name in ("resident-prepack", "resident-prepack-ddr"):
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
        optional=("topology", "thread_role_regions"),
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
    # The five canonical `topology` verdicts. This list and
    # isCanonicalHmxTopology (HmxManifest.cpp) are the same five by contract: a
    # value the C++ writer rejects must not be accepted here, or a kernel the
    # device could not compile would still validate on the host.
    if "topology" in manifest or "thread_role_regions" in manifest:
        topology = manifest.get("topology")
        if topology not in (
            "topology-single-role-hmx",
            "topology-single-role-hvx",
            "role-split-ok",
            "role-mixed-irreducible",
            "role-split-nopack",
        ):
            raise ValueError(
                f"{field_name}.topology is not a canonical verdict: {topology!r}"
            )
        _require_nonnegative_int(
            manifest["thread_role_regions"], f"{field_name}.thread_role_regions"
        )
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


# ---------------------------------------------------------------------------
# v2 manifest reporter (human-facing, opt-in)
# ---------------------------------------------------------------------------
# A pure reader, on the same terms as hmx_manifest_contract() in driver.py: it
# turns an already-decided manifest into text and changes nothing.  Nothing in
# the compile or launch path calls it, so this section is additive by
# construction -- adding a line here cannot move a default code path.
#
# Every reason spelling it prints comes from HMX_PLAN_REASONS /
# HMX_PIPELINE_REASONS above, never from a literal written here.  The C++
# producer and this file are separate boundaries (HmxManifest.h:79-81), so a
# hand-typed reason would print as a plausible-looking line that no validator
# has ever heard of; asking the vocabulary instead turns that into an explicit
# "unrecognized" marker.
#
# The one thing a `plan`-only report cannot express is that "on HMX" and
# "using the HMX engine" are different questions.  A site can clear every
# admission gate, take an HMX plan, and still run an unpipelined serial tile
# loop (HmxManifest.h:112-113, HmxManifest.h:240-246); a report that prints
# only the plan calls that a success.  So each site is labelled with which of
# the two it is.


def _hmx_summary_axis(logical, axis):
    """One `logical` axis as text, tolerating the unavailable form.

    `logical` is null for a site the compiler never had a shape for
    (`library-call`, `non-rank-2` -- _validate_logical above), so a reporter
    that assumed a dict would fail on exactly the sites whose reason matters
    most.
    """
    if not isinstance(logical, dict):
        return "?"
    entry = logical.get(axis)
    if not isinstance(entry, dict):
        return "?"
    value = entry.get("value")
    if value is None:
        return f"{axis}={entry.get('kind', '?')}"
    return str(value)


def _hmx_summary_shape(record):
    logical = record.get("logical")
    if not isinstance(logical, dict):
        return "?"
    return "x".join(_hmx_summary_axis(logical, axis) for axis in ("m", "n", "k"))


def _hmx_summary_known(code, allowed):
    """Render a reason code, marking one this boundary does not recognise.

    An unknown code is reported as unknown rather than passed through.  The
    failure it stands for is silent by nature: a manifest carrying a code no
    validator accepts would otherwise be quoted back to the user as if it were
    the compiler's own explanation.
    """
    if code in allowed:
        return str(code)
    return f"unrecognized reason {code!r}"


def _hmx_summary_site(record):
    """One line per contraction site: did it reach HMX, and what stopped it."""
    if not isinstance(record, dict):
        return f"  [?] unreadable manifest record: {record!r}"
    plan = record.get("plan")
    label = f"  [{record.get('id', '?')}] {record.get('function', '?')} " \
            f"{_hmx_summary_shape(record)} {plan}: "

    if plan in HMX_NON_HVX_PLANS:
        allowed_plan_reasons = HMX_PLAN_REASONS[plan]
        execution = record.get("execution")
        pipeline = execution.get("pipeline") if isinstance(execution, dict) else None
        pipeline = pipeline if isinstance(pipeline, dict) else {}
        selected = pipeline.get("selected")
        reason = record.get("reason")
        admitted = _hmx_summary_known(reason, allowed_plan_reasons)
        if selected == "staged":
            return (
                f"{label}ON HMX, staged pipeline depth="
                f"{pipeline.get('depth', '?')} (admitted: {admitted})"
            )
        if selected == "serial":
            # On the engine, running one tile at a time.  Not a refusal, and
            # deliberately not worded like one.
            pipeline_reason = pipeline.get("reason")
            if pipeline_reason is None:
                detail = "no pipeline reason published"
            else:
                detail = _hmx_summary_known(
                    pipeline_reason, HMX_PIPELINE_REASONS
                )
            return (
                f"{label}ON HMX but SERIAL tile loop, not pipelined "
                f"(admitted: {admitted}; staging declined: {detail})"
            )
        return (
            f"{label}ON HMX, pipeline selection unreadable "
            f"(selected={selected!r}; admitted: {admitted})"
        )

    if plan in HMX_PLANS - HMX_NON_HVX_PLANS:
        # Refused.  The reason is the compiler's own, and it is the whole
        # answer to "why did this not get HMX".
        return (
            f"{label}DROPPED to {plan} (reason: "
            f"{_hmx_summary_known(record.get('reason'), HMX_PLAN_REASONS[plan])})"
        )

    return f"{label}plan {plan!r} is not an HMX plan this boundary knows"


def summarize_hmx_manifest(manifest):
    """Return a human-readable verdict for one parsed v2 HMX manifest.

    The question this answers is the one a `plan`-only answer gets wrong: of
    the contraction sites in this kernel, how many reached the HMX engine, how
    many were refused, and which sites reached it but run serially.

    Properties this function is required to keep, and why:

    * **Pure.**  It reads the manifest and builds a string.  No printing, no
      warnings, no logging, no attribute writes, and no validation that could
      raise -- a reporter that refuses to report a malformed manifest is
      useless exactly when it is needed, so an unreadable value is labelled
      instead of raised on.
    * **No invented vocabulary.**  Plan and reason spellings are taken from
      HMX_PLAN_REASONS / HMX_PIPELINE_REASONS, and "reached HMX" is decided by
      HMX_NON_HVX_PLANS, the same set `enforce_hmx_launch_contract` uses.
    * **Opt-in.**  Nothing in the compile or launch path calls it.  The one
      caller is `hmx_manifest_verdict()` in driver.py, which is a read-only
      sibling of `hmx_manifest_contract()` and is likewise only reached when a
      caller asks for it.

    Accepts `None`, which is what `hmx_manifest_contract()` returns before the
    kernel's first launch.
    """
    if not isinstance(manifest, dict):
        return "no HMX manifest on this kernel: nothing to summarize"
    records = manifest.get("matmuls")
    if not isinstance(records, list) or not records:
        return "HMX manifest: no contraction site recorded in this kernel"

    on_hmx = sum(
        1
        for record in records
        if isinstance(record, dict) and record.get("plan") in HMX_NON_HVX_PLANS
    )
    refused = sum(
        1
        for record in records
        if isinstance(record, dict)
        and record.get("plan") in HMX_PLANS - HMX_NON_HVX_PLANS
    )
    total = len(records)

    if on_hmx == total:
        headline = f"HMX manifest: ALL {on_hmx} of {total} on HMX"
    elif on_hmx == 0:
        headline = (
            f"HMX manifest: NO site on HMX -- 0 of {total} on HMX; "
            f"all {refused} refused"
        )
    else:
        headline = (
            f"HMX manifest: PARTIAL -- {on_hmx} of {total} on HMX; "
            f"{refused} refused"
        )
    unreadable = total - on_hmx - refused
    if unreadable:
        # Never claim a plan for a record this boundary could not classify.
        headline += f"; {unreadable} record(s) carry an unrecognized plan"

    lines = [headline]
    lines.extend(_hmx_summary_site(record) for record in records)
    return "\n".join(lines)


# The default-path reader.  The opt-in reporter above answers "tell me
# everything about this kernel" for a caller who already knows to ask; this
# one answers the question a caller did not know to ask -- "did this kernel
# silently lose HMX, and why?" -- on the one default path every Triton
# compilation takes.  Measured 2026-10-01: 8 of 41 dot-bearing operators fall
# back to HVX with zero diagnostics, and 6 of those 8 carry an exact reason
# code in this very manifest (docs/evidence/2026-10-01/
# why-hmx-refusals-are-invisible-2026-10-01.md).  The data was never missing;
# the reader was.


def hmx_fallback_notice(manifest, kernel_name=None):
    """One line naming the contraction sites that fell back to HVX, and why.

    Returns None when nothing fell back, so a kernel whose every matmul
    reached the engine contributes no line at all -- the noise rule that
    leaves the default path exactly as quiet as it was for every kernel that
    did not fall back.  When at least one site fell back, the answer is one
    line:

        hmx: <kernel>: 2/5 matmuls fell back to HVX (tile-alignment x2,
        min-rows x1) -- per-site reasons: kernel.packed_metadata[...]

    Properties, each pinned by the host suite:

    * **Never raises.**  A record this boundary cannot read -- a missing,
      null or unrecognized reason, the state 2 of the 8 measured fallbacks
      were in -- contributes `reason unavailable` instead of an exception or
      silence.  A reporter that dies on the manifest it exists to read is
      worse than no reporter.
    * **No invented vocabulary.**  A refusal is a `plan` in HMX_PLANS minus
      HMX_NON_HVX_PLANS, the same sets the launch contract decides "on the
      engine" with, and a reason is only quoted when it is in HMX_PLAN_REASONS
      for that record's plan.  Anything else is `reason unavailable`: an
      unrecognized code is never quoted back as if the compiler had emitted
      it.
    * **One line per kernel, reasons grouped.**  N refusals produce one line
      with per-reason counts, not N lines, and only reasons that occurred are
      named.
    * **Pure and read-only.**  Builds a string and nothing else: no printing,
      no warnings, no mutation.  The emission is the compile-time read
      point's job (backend/compiler.py), not this function's, so the two can
      be tested separately.
    """
    if not isinstance(manifest, dict):
        return None
    records = manifest.get("matmuls")
    if not isinstance(records, list) or not records:
        return None

    refused = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("plan") in HMX_PLANS - HMX_NON_HVX_PLANS
    ]
    if not refused:
        return None

    unavailable = "reason unavailable"
    counts = {}
    for record in refused:
        reason = record.get("reason")
        allowed = HMX_PLAN_REASONS.get(record.get("plan"), frozenset())
        if not isinstance(reason, str) or reason not in allowed:
            reason = unavailable
        counts[reason] = counts.get(reason, 0) + 1
    # Most frequent first, then alphabetical, so the line is deterministic
    # and the reason a user is most likely acting on comes first.
    grouped = ", ".join(
        f"{reason} x{counts[reason]}"
        for reason in sorted(counts, key=lambda r: (-counts[r], r))
    )

    who = f"{kernel_name}: " if isinstance(kernel_name, str) and kernel_name else ""
    return (
        f"hmx: {who}{len(refused)}/{len(records)} matmuls fell back to HVX "
        f'({grouped}) -- per-site reasons: kernel.packed_metadata["hmx_manifest"]'
    )


# ---------------------------------------------------------------------------
# Launch-time readers
# ---------------------------------------------------------------------------
# The compile-time notice above answers "did this kernel silently lose HMX?"
# at the one point every compilation passes through. It cannot answer the
# questions a user asks at LAUNCH time, and each of the three below has already
# produced a wrong conclusion in this repository:
#
#   REFUSAL       -- the per-site reasons, at the moment the launch is being
#                    watched. The compile-time print runs only on a cache miss
#                    and is buried in compile output
#                    (docs/evidence/2026-10-01/
#                    why-hmx-refusals-are-invisible-2026-10-01.md).
#   PARTIAL       -- a function where some matmuls reached the engine and some
#                    did not reads as "I got HMX" in every plan-only check
#                    (the module-level PARTIAL verdict of
#                    tools/hexmlir/manifest_verdict.py).
#   CONTRADICTION -- a record claiming plan=full-hmx whose own
#                    execution.bridge_counts say pack_act/pack_weight/unpack
#                    are ALL 0. Those counts are recounted from the IR
#                    (refreshHmxManifestBridgeCounts), so zero bridge sites
#                    means no bridge op exists in the kernel: hmx-partition
#                    emitted an empty HMX span while the manifest kept the
#                    admitted plan -- the false green measured on n-loop-only
#                    matmul structures (the TILES comment in
#                    exp/hmx/op_bench/mm_user_shapes.py). The two halves of
#                    such a record cannot both be true.
#
# Both functions below are pure and never raise, on the same terms as the
# compile-time notice above: the judgment lives here so host tests can drive it
# with synthetic manifests, and the emission (warnings.warn from the driver's
# launch path) is the caller's job. The interpreter's default warning filter
# deduplicates by message text, so repeated launches of a kernel print once --
# that only holds if these messages carry nothing that varies per launch (no
# reps, timestamps or addresses), which is why nothing here formats either.


def _hmx_warning_name(kernel_name):
    return f"{kernel_name}: " if isinstance(kernel_name, str) and kernel_name else ""


def _hmx_warning_reason(record):
    """The compiler's own reason for one record, never an invented spelling.

    A missing reason is `reason unavailable` and an unrecognized one is
    labelled unrecognized (via _hmx_summary_known), matching the
    compile-time notice's vocabulary: a code this boundary does not know is
    never quoted back as if the compiler had emitted it.
    """
    reason = record.get("reason")
    if reason is None:
        return "reason unavailable"
    return _hmx_summary_known(
        reason, HMX_PLAN_REASONS.get(record.get("plan"), frozenset())
    )


def _hmx_warning_site(record):
    """`[#id function shape reason=...]` for one record in a launch warning."""
    return (
        f"[#{record.get('id', '?')} {record.get('function', '?')} "
        f"{_hmx_summary_shape(record)} "
        f"reason={_hmx_warning_reason(record)}]"
    )


def _hmx_parse_manifest(manifest):
    """The manifest as a dict, or None.

    The launch path holds the manifest as the JSON string the launcher itself
    consumes (driver.py); host tests hand in the parsed dict. Either spelling
    is accepted, and anything unreadable answers None rather than raising --
    a warning that crashes the launch it is warning about would be worse than
    no warning.
    """
    if isinstance(manifest, str):
        try:
            manifest = json.loads(manifest)
        except (TypeError, ValueError):
            return None
    return manifest if isinstance(manifest, dict) else None


def _hmx_bridge_sites_all_zero(record):
    """True only for a readable full record whose three bridge counts are 0.

    A missing execution/bridge_counts/count key is NOT a contradiction: it is
    an unreadable record, and this check exists to catch a record that reads
    clearly and disagrees with itself, not to guess at one that reads nothing.
    """
    execution = record.get("execution")
    if not isinstance(execution, dict):
        return False
    counts = execution.get("bridge_counts")
    if not isinstance(counts, dict):
        return False
    return all(
        isinstance(counts.get(field), int) and counts.get(field) == 0
        for field in ("pack_act_sites", "pack_weight_sites", "unpack_sites")
    )


def hmx_manifest_warnings(manifest, kernel_name=None):
    """Launch-time warnings for HMX facts the manifest records but no path shows.

    Returns a list of message strings, empty exactly when every contraction
    site in the manifest reached the HMX engine and the manifest is
    self-consistent about how. Three conditions, in this order:

    * any site with plan=hvx  -> one REFUSAL message naming the kernel, each
      refused site's id/function/shape and the compiler's own reason, and
      saying plainly that those matmuls run on the HVX path instead of HMX;
    * both hvx and non-hvx sites -> one PARTIAL message stating `k/n matmul(s)
      on HMX`, so a mixed function cannot read as "HMX is on";
    * every full-hmx record whose pack_act_sites, pack_weight_sites and
      unpack_sites are all 0 -> one CONTRADICTION message per record: the
      partition produced no HMX bridge sites (false green) and the matmul did
      not execute on HMX.

    Accepts the JSON string form or the parsed dict; an unreadable manifest,
    no matmuls, or records this boundary cannot classify yield no warning
    rather than an exception (a reporter that dies on its own input is worse
    than silence). Records with an unrecognized plan are counted by neither
    side: an unknown plan is not evidence of HMX use nor of refusal.
    """
    manifest = _hmx_parse_manifest(manifest)
    if manifest is None:
        return []
    records = manifest.get("matmuls")
    if not isinstance(records, list) or not records:
        return []
    refused = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("plan") in HMX_PLANS - HMX_NON_HVX_PLANS
    ]
    on_hmx = [
        record
        for record in records
        if isinstance(record, dict) and record.get("plan") in HMX_NON_HVX_PLANS
    ]
    total = len(records)
    who = _hmx_warning_name(kernel_name)
    messages = []

    if refused:
        sites = " ".join(_hmx_warning_site(record) for record in refused)
        messages.append(
            f"hmx: {who}HMX is not used for {len(refused)}/{total} matmul(s); "
            f"they run on the HVX path, not on the HMX engine: {sites} "
            f'-- per-site records: kernel.packed_metadata["hmx_manifest"]'
        )
    if refused and on_hmx:
        messages.append(
            f"hmx: {who}PARTIAL -- {len(on_hmx)}/{total} matmul(s) on HMX, "
            f"{len(refused)} on HVX; 'this kernel uses HMX' is true only for "
            f"the {len(on_hmx)} site(s) that reached the engine"
        )
    for record in on_hmx:
        # Scope, not special-casing: the all-zero bridge-count contradiction
        # has only ever been measured on plan=full-hmx records (the n-loop
        # empty-span defect). An hmx-tail record's counts are not pinned by
        # any measurement yet, so this checks exactly the claim that is known
        # to be falsifiable -- widen it when a tail case is measured.
        if record.get("plan") != "full-hmx":
            continue
        if not _hmx_bridge_sites_all_zero(record):
            continue
        messages.append(
            f"hmx: {who}matmul #{record.get('id', '?')} "
            f"{record.get('function', '?')} {_hmx_summary_shape(record)} "
            f"claims plan=full-hmx, but its execution.bridge_counts are "
            f"pack_act_sites=0, pack_weight_sites=0, unpack_sites=0: "
            f"hmx-partition produced no HMX bridge sites (false green) -- "
            f"no HMX work exists in this kernel, so this matmul did not "
            f"execute on HMX"
        )
    return messages


def hmx_grid_notice(manifest, launch_grid, kernel_name=None, *, threaded_dispatch=None):
    """The grid>1-with-HMX facts a Triton programmer otherwise gets wrong.

    Returns None unless BOTH hold: prod(launch_grid) > 1, and the manifest
    claims at least one HMX site (plan in HMX_NON_HVX_PLANS). Those are the
    conditions under which "more programs" is believed to mean "parallel HMX
    matmuls", which the runtime does not provide:

    * the HMX engine section runs on a single resident thread that holds the
      HMX lock for life (HmxRoleExecutor), so programs never execute their
      HMX matmuls in parallel, no matter how many there are;
    * `threaded_dispatch` (the wrapper's tm.exec vs tm.exec_serial choice,
      enableThreadedDispatch or enableMultiThreading) decides what the other
      programs do: fresh qurt threads created and joined on every launch
      (ThreadManager does not keep the thread pool alive), or one serial
      loop. None means the caller does not know which, so the message states
      both rather than guessing.

    Facts only: no timing, no speedup, no comparison -- nothing here has been
    measured as a ratio, so nothing here claims one. The message ends with
    the one structural recommendation the mechanism supports: for HMX
    matmuls, grid=1 with the tiling loop inside the kernel.
    """
    manifest = _hmx_parse_manifest(manifest)
    if manifest is None:
        return None
    records = manifest.get("matmuls")
    if not isinstance(records, list):
        return None
    on_hmx = [
        record
        for record in records
        if isinstance(record, dict) and record.get("plan") in HMX_NON_HVX_PLANS
    ]
    if not on_hmx:
        return None
    if (
        not isinstance(launch_grid, (tuple, list))
        or len(launch_grid) != 3
        or any(type(size) is not int or size < 1 for size in launch_grid)
    ):
        return None
    programs = prod(launch_grid)
    if programs <= 1:
        return None

    threads = (
        f"each launch creates and joins {programs} fresh qurt threads "
        "(ThreadManager does not keep the thread pool alive)"
    )
    serial = f"the {programs} instances run one after another in a serial loop"
    if threaded_dispatch is True:
        dispatch = threads
    elif threaded_dispatch is False:
        dispatch = f"{serial} (tm.exec_serial), not in parallel"
    else:
        dispatch = (
            f"with threaded dispatch {threads}; without it {serial} "
            f"(tm.exec_serial)"
        )

    who = _hmx_warning_name(kernel_name)
    return (
        f"hmx: {who}launch grid {tuple(launch_grid)} = {programs} program "
        f"instance(s), but their {len(on_hmx)} HMX matmul site(s) do not run "
        f"in parallel across programs: the HMX engine section executes on a "
        f"single resident thread that holds the HMX lock for life "
        f"(HmxRoleExecutor), and {dispatch}. For HMX matmuls prefer grid=1 "
        f"with the tiling loop inside the kernel"
    )


# ---------------------------------------------------------------------------
# Record-only v3 consumer
# ---------------------------------------------------------------------------
# Every check below is exact-field and closed-enum.  There is no "fill in a
# default" path and no silent skip: a v3 record with an unknown field, an
# unknown status, a wrong unit or a mismatched scope is rejected, because a
# record-only document that nobody can interpret is worse than no record at all.


def _require_bytes_or_null(value, path, *, allow_value):
    """A capacity quantity: a proven non-negative count, or explicit unknown.

    `None` is the only spelling of "unknown"; a missing key never stands in for
    it, because that is how a `not-proven` fact would quietly become a `0`.
    """
    if value is None:
        return
    if not allow_value:
        raise ValueError(
            f"{path} is {value!r}, but a 'not-proven' fact must stay unknown"
        )
    _require_nonnegative_int(value, path)


def _require_proof_status(value, path):
    if not isinstance(value, str) or value not in HMX_RECORD_PROOF_STATUSES:
        raise ValueError(f"{path} is not a canonical proof status: {value!r}")
    return value


def _validate_record_axis(value, path, axis):
    _require_object(value, path)
    if "kind" not in value:
        raise ValueError(f"{path} is missing required field(s): ['kind']")
    # Dispatch on `kind` first: the closed field set of an axis is a function of
    # its kind, so the exact-field check has to come after it.
    kind = value["kind"]
    if kind == "static":
        _require_exact_fields(value, path, ("kind", "value"))
        _require_strict_int(value["value"], f"{path}.value", minimum=1)
    elif kind == "dynamic":
        # A dynamic axis names itself and carries no runtime guess.
        _require_exact_fields(value, path, ("kind", "symbol"))
        if value["symbol"] != axis:
            raise ValueError(
                f"{path}.symbol must name this axis ({axis!r}), got {value['symbol']!r}"
            )
    else:
        raise ValueError(f"{path}.kind must be 'static' or 'dynamic', got {kind!r}")
    return kind


def _validate_record_shape_v3(value, path):
    _require_exact_fields(value, path, ("state", "logical", "specialization"))
    state = value["state"]
    if state not in HMX_RECORD_SHAPE_STATES:
        raise ValueError(f"{path}.state is not canonical: {state!r}")
    if value["specialization"] not in HMX_RECORD_SPECIALIZATIONS:
        raise ValueError(
            f"{path}.specialization is not canonical: {value['specialization']!r}"
        )
    logical = _require_object(value["logical"], f"{path}.logical")
    _require_exact_fields(logical, f"{path}.logical", ("m", "n", "k"))
    kinds = {
        axis: _validate_record_axis(logical[axis], f"{path}.logical.{axis}", axis)
        for axis in ("m", "n", "k")
    }
    static_count = sum(kind == "static" for kind in kinds.values())
    derived = (
        "static" if static_count == 3
        else "dynamic" if static_count == 0
        else "partially-dynamic"
    )
    # The declared state must be the unique derivation of the three axes, and
    # the specialization policy follows from the same three axes.
    if state != derived:
        raise ValueError(
            f"{path}.state is {state!r} but the tagged axes derive {derived!r}"
        )
    expected = "upstream-static" if static_count == 3 else "upstream-only"
    if value["specialization"] != expected:
        raise ValueError(
            f"{path}.specialization must be {expected!r} for a {derived!r} shape, "
            f"got {value['specialization']!r}"
        )
    return value


def _validate_record_scope_v3(value, path, function):
    _require_exact_fields(value, path, ("function", "invocations", "grid", "resident"))
    if value["function"] != function:
        raise ValueError(
            f"{path}.function must name the record's own function ({function!r}), "
            f"got {value['function']!r}"
        )
    _require_strict_int(value["invocations"], f"{path}.invocations", minimum=1)
    if value["invocations"] != HMX_RECORD_INVOCATIONS:
        raise ValueError(
            f"{path}.invocations must be {HMX_RECORD_INVOCATIONS}, got "
            f"{value['invocations']!r}"
        )
    grid = _require_object(value["grid"], f"{path}.grid")
    _require_exact_fields(grid, f"{path}.grid", ("policy", "required_product"))
    if grid["policy"] != HMX_RECORD_GRID_POLICY:
        raise ValueError(
            f"{path}.grid.policy must be {HMX_RECORD_GRID_POLICY!r}, got "
            f"{grid['policy']!r}"
        )
    _require_strict_int(grid["required_product"], f"{path}.grid.required_product", minimum=1)
    if grid["required_product"] != HMX_RECORD_GRID_REQUIRED_PRODUCT:
        raise ValueError(
            f"{path}.grid.required_product must be {HMX_RECORD_GRID_REQUIRED_PRODUCT}, "
            f"got {grid['required_product']!r}"
        )
    if value["resident"] != HMX_RECORD_RESIDENT_SCOPE:
        raise ValueError(
            f"{path}.resident must be {HMX_RECORD_RESIDENT_SCOPE!r}, got "
            f"{value['resident']!r}"
        )
    return value


def _validate_record_capacity_v3(value, path, unit, basis, quantities):
    """One capacity block: pinned unit and basis, closed status, honest values."""
    required = ("unit", "basis", "status") + tuple(quantities)
    _require_exact_fields(value, path, required)
    if value["unit"] != unit:
        raise ValueError(f"{path}.unit must be {unit!r}, got {value['unit']!r}")
    if value["basis"] != basis:
        raise ValueError(f"{path}.basis must be {basis!r}, got {value['basis']!r}")
    status = _require_proof_status(value["status"], f"{path}.status")
    # The machine form of "never turn not-proven into 0": an unproven block may
    # not carry a number at all, and a proven one may not leave a default.
    for quantity in quantities:
        _require_bytes_or_null(
            value[quantity], f"{path}.{quantity}", allow_value=status != "not-proven"
        )
    return value


def _validate_record_resources_v3(value, path):
    _require_exact_fields(
        value, path, ("requested", "allocator_aligned", "observed_high_water")
    )
    _validate_record_capacity_v3(
        value["requested"],
        f"{path}.requested",
        HMX_RECORD_UNIT_BYTES,
        HMX_RECORD_BASIS_REQUESTED,
        (
            "transient_requested_peak_bytes",
            "resident_requested_bytes",
            "modeled_requested_peak_bytes",
        ),
    )
    _validate_record_capacity_v3(
        value["allocator_aligned"],
        f"{path}.allocator_aligned",
        HMX_RECORD_UNIT_BYTES,
        HMX_RECORD_BASIS_ALLOCATOR,
        (
            "transient_aligned_peak_bytes",
            "resident_aligned_bytes",
            "modeled_aligned_peak_bytes",
        ),
    )
    observed = _require_object(value["observed_high_water"], f"{path}.observed_high_water")
    _require_exact_fields(
        observed,
        f"{path}.observed_high_water",
        ("unit", "basis", "status", "value_bytes", "scope", "source"),
    )
    if observed["unit"] != HMX_RECORD_UNIT_BYTES:
        raise ValueError(
            f"{path}.observed_high_water.unit must be {HMX_RECORD_UNIT_BYTES!r}"
        )
    if observed["basis"] != HMX_RECORD_BASIS_OBSERVED:
        raise ValueError(
            f"{path}.observed_high_water.basis must be {HMX_RECORD_BASIS_OBSERVED!r}"
        )
    status = _require_proof_status(
        observed["status"], f"{path}.observed_high_water.status"
    )
    # A per-record high-water mark has no accepted join key, so the only
    # permitted scope is the process aggregate.  Naming a narrower scope would
    # be a claim the schema cannot support.
    if observed["scope"] != HMX_RECORD_OBSERVATION_SCOPE:
        raise ValueError(
            f"{path}.observed_high_water.scope must be {HMX_RECORD_OBSERVATION_SCOPE!r}, "
            f"got {observed['scope']!r}"
        )
    _require_bytes_or_null(
        observed["value_bytes"],
        f"{path}.observed_high_water.value_bytes",
        allow_value=status != "not-proven",
    )
    if observed["source"] is not None:
        _require_string(observed["source"], f"{path}.observed_high_water.source")
    return value


def _validate_record_proof_v3(value, path):
    _require_exact_fields(value, path, ("status", "basis"))
    status = _require_proof_status(value["status"], f"{path}.status")
    if value["basis"] is None:
        # `complete` with no basis is exactly the "masks a gap" case the
        # four-axis contract exists to prevent.
        if status == "complete":
            raise ValueError(f"{path} is complete but names no basis")
        return value
    _require_string(value["basis"], f"{path}.basis")
    return value


def _validate_record_proofs_v3(value, path):
    _require_exact_fields(value, path, HMX_RECORD_PROOF_AXES)
    for axis in HMX_RECORD_PROOF_AXES:
        _validate_record_proof_v3(value[axis], f"{path}.{axis}")
    return value


def _record_fingerprint_payload(record):
    # The C++ producer hashes the complete record minus the digest, prefixed by
    # the schema so a record can never be replayed under a different wire name.
    payload = {"schema": HMX_RECORD_SCHEMA}
    for key, value in record.items():
        if key != "record_fingerprint":
            payload[key] = value
    return payload


def compute_hmx_record_fingerprint(record):
    """Compute the stable fingerprint for one validated v3 record."""
    return canonical_json_sha256(_record_fingerprint_payload(record))


def _validate_record_v3(entry, path):
    _require_exact_fields(
        entry,
        path,
        (
            "function",
            "id",
            "plan",
            "shape",
            "scope",
            "resources",
            "proofs",
            "fallback",
            "record_fingerprint",
        ),
    )
    function = _require_string(entry["function"], f"{path}.function")
    record_id = _require_nonnegative_int(entry["id"], f"{path}.id")
    if entry["plan"] not in HMX_PLANS:
        raise ValueError(f"{path}.plan is not canonical: {entry['plan']!r}")
    _validate_record_shape_v3(entry["shape"], f"{path}.shape")
    _validate_record_scope_v3(entry["scope"], f"{path}.scope", function)
    _validate_record_resources_v3(entry["resources"], f"{path}.resources")
    _validate_record_proofs_v3(entry["proofs"], f"{path}.proofs")
    # Read the one declared failure behavior rather than hardcoding it: a record
    # that promised something else must be refused, not acted on.
    _require_exact_fields(entry["fallback"], f"{path}.fallback", tuple(HMX_RECORD_FALLBACK))
    for key, expected in HMX_RECORD_FALLBACK.items():
        if entry["fallback"][key] != expected:
            raise ValueError(
                f"{path}.fallback.{key} must be {expected!r}, got "
                f"{entry['fallback'][key]!r}; it is a protocol value, not a policy "
                f"switch"
            )
    fingerprint = entry["record_fingerprint"]
    if not isinstance(fingerprint, str) or _SHA256_FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"{path}.record_fingerprint must be a lowercase sha256 digest")
    if fingerprint != compute_hmx_record_fingerprint(entry):
        raise ValueError(
            f"{path}.record_fingerprint does not match its canonical record content"
        )
    return {"path": path, "entry": entry, "function": function, "id": record_id}


def validate_hmx_record_document(document, field_name="hmx_record"):
    """Validate one record-only v3 document.

    The document is a diagnostic record, not an executable contract: this
    function proves that the record says what it means and that its digests are
    intact.  It deliberately grants nothing -- no plan, no budget, no grid
    constraint and no resident default can be read out of a record it accepted.
    """
    _require_object(document, field_name)
    _require_exact_fields(
        document, field_name, ("schema", "record_mode", "admission", "records")
    )
    if document["schema"] != HMX_RECORD_SCHEMA:
        raise ValueError(
            f"{field_name}.schema must be {HMX_RECORD_SCHEMA!r}, got "
            f"{document['schema']!r}"
        )
    # Neither value is a runtime switch, and a consumer that accepted anything
    # else would be accepting an admission-capable document.
    if document["record_mode"] != HMX_RECORD_MODE:
        raise ValueError(
            f"{field_name}.record_mode must be {HMX_RECORD_MODE!r}, got "
            f"{document['record_mode']!r}"
        )
    if document["admission"] != HMX_RECORD_ADMISSION:
        raise ValueError(
            f"{field_name}.admission must be {HMX_RECORD_ADMISSION!r}, got "
            f"{document['admission']!r}"
        )
    records = document["records"]
    if not isinstance(records, list):
        raise ValueError(
            f"{field_name}.records must be a list, got {type(records).__name__}"
        )
    seen = set()
    for index, entry in enumerate(records):
        info = _validate_record_v3(entry, f"{field_name}.records[{index}]")
        key = (info["function"], info["id"])
        if key in seen:
            raise ValueError(
                f"{field_name}.records[{index}] duplicates function-local id {info['id']}"
            )
        seen.add(key)
    return document


def validate_hmx_record_json(record_json, field_name="hmx_record"):
    """Validate the JSON text of a v3 record child."""
    return validate_hmx_record_document(
        _json_object(record_json, field_name), field_name
    )


def enforce_hmx_launch_contract(
    manifest_json: str, launch_grid: tuple[int, int, int], field_name: str = "hmx_manifest"
):
    """Validate the manifest and enforce its v2 single-instance launch promise.

    v2 deliberately carries bridge-only VTCM facts, not a kernel-wide peak.  It
    can nevertheless prove the launch-shape part of the contract: a tail plan
    is single-instance.  A resident full-HMX workspace used to be single-
    instance too; since 2026-10-04 its residency is keyed by the caller's flat
    program id at runtime (concurrent instances get separate buffers, the same
    pid reuses its buffer across launches), so it keeps the legacy runtime grid
    behavior like the runtime-internal full plan.  The check lives beside the
    strict consumer so every launcher entry point can apply the same rule
    before it creates a wrapper or touches the device.
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

    required = ("func", "slot", "shape", "crouton", "dtype", "location")
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
        # The source argument's element type; the packed image is always the
        # fp16 crouton. An f32 source is quantised by the host with the same
        # conversion the device pack runs.
        if entry["dtype"] not in ("f16", "f32"):
            raise ValueError(
                f"{path}.dtype must be 'f16' or 'f32', got {entry['dtype']!r}"
            )
        # Where the compiler put the image. The host writes the same bytes
        # either way -- the field is a fact about the kernel, not an
        # instruction -- so it is validated rather than acted on.
        if entry["location"] not in HMX_WEIGHT_LOCATIONS:
            raise ValueError(
                f"{path}.location must be one of {sorted(HMX_WEIGHT_LOCATIONS)}, "
                f"got {entry['location']!r}"
            )
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
        if policy["policy"] in ("resident-prepack", "resident-prepack-ddr"):
            if entry is None:
                raise ValueError(
                    "resident-prepack policy has no matching weight_prepack entry"
                )
            # Mirror the C++ pairing (`HmxManifest.cpp`, `isCanonicalWeightReason`
            # + the finalize check): a dtype-specific reason must agree with the
            # entry's source dtype. `eligible-b2-n-slice` names the view, not the
            # dtype, so it is admitted for either -- exactly as in C++.
            if policy["reason"] == "eligible-aligned-f16" and entry["dtype"] != "f16":
                raise ValueError(
                    "resident-prepack reason 'eligible-aligned-f16' disagrees "
                    f"with weight_prepack dtype {entry['dtype']!r}"
                )
            if (
                policy["reason"] == "eligible-quantized-f32"
                and entry["dtype"] != "f32"
            ):
                raise ValueError(
                    "resident-prepack reason 'eligible-quantized-f32' disagrees "
                    f"with weight_prepack dtype {entry['dtype']!r}"
                )
        elif entry is not None:
            raise ValueError(
                "device-pack policy must not publish a weight_prepack entry"
            )
    for key in entries:
        policy = policies.get(key)
        if policy is None or policy["policy"] not in (
            "resident-prepack",
            "resident-prepack-ddr",
        ):
            raise ValueError(
                "weight_prepack entry is not referenced by a resident-prepack policy"
            )


# The two envelopes are closed identities, and each one carries exactly the
# children its schema defines.  A v1 envelope has no record child; a v2 envelope
# must have one, because a v3-capable envelope that structurally omits the child
# is a malformed artifact rather than a "no record this run" signal.
TRANSLATION_ENVELOPE_CHILDREN = {
    HMX_TRANSLATION_V1_SCHEMA: ("schema", "weight_prepack", "hmx_manifest"),
    HMX_TRANSLATION_V2_SCHEMA: (
        "schema",
        "weight_prepack",
        "hmx_manifest",
        "hmx_record",
    ),
}


def parse_translation_metadata(metadata_json):
    """Unpack the C++ translation envelope into launcher-facing JSON strings.

    Returns ``(weight_prepack, hmx_manifest, hmx_record)``.  All three stay JSON
    text because the existing launcher contract consumes the first as a string
    and the other two are host-side consumers.  ``hmx_record`` is ``None`` for a
    v1 envelope, which is the only way to get ``None`` here: no default record
    object is manufactured for a missing child, because a stale or malformed
    envelope is an actionable compilation error.

    The envelope schema is dispatched on its declared value from a closed set.
    Nothing here probes one schema and falls back to another: a v2 envelope
    missing its v3 child, or a v1 envelope carrying one, is rejected outright.
    """
    envelope = _json_object(metadata_json, "translation metadata")
    schema = envelope.get("schema")
    if schema not in TRANSLATION_ENVELOPE_CHILDREN:
        raise ValueError(
            "translation metadata.schema must be one of "
            f"{sorted(TRANSLATION_ENVELOPE_CHILDREN)}, got {schema!r}"
        )
    _require_exact_fields(
        envelope, "translation metadata", TRANSLATION_ENVELOPE_CHILDREN[schema]
    )

    weight_prepack = envelope["weight_prepack"]
    validate_weight_prepack(weight_prepack)
    hmx_manifest = envelope["hmx_manifest"]
    validate_hmx_manifest(hmx_manifest)
    _validate_manifest_weight_prepack(hmx_manifest, weight_prepack)

    # The v2 execution manifest is the execution authority in both envelopes and
    # is re-encoded, not reinterpreted.  The v3 child is a sibling record: it is
    # validated against its own closed contract and never merged into, derived
    # from, or substituted for the v2 child.
    hmx_record = None
    if schema == HMX_TRANSLATION_V2_SCHEMA:
        hmx_record = _canonical_json(
            validate_hmx_record_document(envelope["hmx_record"])
        )

    # Re-encode only the inner objects.  The envelope itself is an internal
    # C++/Python transport detail and must not leak into any consumer.
    return (
        _canonical_json(weight_prepack),
        _canonical_json(hmx_manifest),
        hmx_record,
    )


def apply_translation_metadata(metadata, metadata_json):
    """Validate an envelope and publish its independent fields."""
    try:
        weight_prepack, hmx_manifest, hmx_record = parse_translation_metadata(
            metadata_json
        )
    except ValueError as exc:
        raise RuntimeError(
            "invalid HMX translation metadata returned by the backend: "
            f"{exc}; rebuild the kernel and check the HMX manifest publisher"
        ) from exc
    metadata["weight_prepack"] = weight_prepack
    metadata["hmx_manifest"] = hmx_manifest
    # `HMX_RECORD_ABSENT` for a v1 envelope, a validated JSON document for a v2
    # one.  It is a diagnostic record, so the launcher must not read it to pick a
    # plan, a budget, a resident default or a launch grid.
    metadata["hmx_record"] = (
        hmx_record if hmx_record is not None else HMX_RECORD_ABSENT
    )


def hmx_cache_identity(
    envelope_schema=TRANSLATION_METADATA_SCHEMA,
    manifest_schema=HMX_MANIFEST_SCHEMA,
    accounting_mode=HMX_ACCOUNTING_MODE,
    proof_producer_abi=HMX_RECORD_PROOF_PRODUCER_ABI,
    semantic_hmx_abi=HMX_SHAPE_TAIL_ABI_VERSION,
    options_hash="",
):
    """Return the wire-mode token mixed into the Triton object cache key.

    Cache identity separates the envelope schema, the manifest schema, the
    accounting mode, the proof/observation producer ABI, the semantic HMX ABI and
    the effective backend options.  Changing any one of them must miss the cache,
    so each is a separate field of the token rather than a concatenated blob a
    rename could silently collapse.  A build that changes what a cached artifact
    *means* therefore cannot reuse an entry written under the old meaning.

    What this token deliberately does **not** do is separate the record-only
    envelope from the production one.  Those two modes are chosen by an IR
    attribute, and every parameter here is a build-time or per-kernel-source
    input, so no value of this function can tell them apart.  Rather than let the
    parameters imply a separation the key cannot deliver, the record-only mode is
    kept out of the Triton object cache altogether -- see
    ``backend/compiler.py::_reject_cacheable_record_only_module`` for the
    mechanism and the reason.  The parameters exist so that a future mode which
    *is* build-time visible (a v4 admission contract, say) has a place to land.
    """
    if envelope_schema not in TRANSLATION_ENVELOPE_CHILDREN:
        raise ValueError(f"unknown translation envelope schema: {envelope_schema!r}")
    if manifest_schema not in (HMX_MANIFEST_SCHEMA, HMX_RECORD_SCHEMA):
        raise ValueError(f"unknown HMX manifest schema: {manifest_schema!r}")
    return "|".join(
        (
            envelope_schema,
            manifest_schema,
            accounting_mode,
            proof_producer_abi,
            semantic_hmx_abi,
            options_hash,
        )
    )


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
        # `HMX_RECORD_ABSENT` is the v1-envelope spelling of "no record child"; a
        # non-empty one must be a complete, self-consistent v3 document.
        if packed["hmx_record"] != HMX_RECORD_ABSENT:
            validate_hmx_record_json(packed["hmx_record"])
    except ValueError as exc:
        raise RuntimeError(
            f"invalid compiled kernel metadata: {exc}; clear "
            "TRITON_CACHE_DIR (tools/hexmlir/env.sh) and rebuild the kernel"
        ) from exc
    return packed
