# ===- compiler.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import hashlib
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path
import subprocess
from typing import Any, Dict, no_type_check
from types import ModuleType

from triton._C.libtriton import ir, passes, qcom_hexagon_backend  # type: ignore
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.qcom_hexagon_backend.utils import parse_return_types
from triton.backends.qcom_hexagon_backend.utils import (
    HMX_ACCOUNTING_MODE,
    HMX_MANIFEST_SCHEMA,
    HMX_SHAPE_TAIL_ABI_VERSION,
    PACK_METADATA_REQUIRED,
    TRANSLATION_METADATA_SCHEMA,
    apply_translation_metadata,
    hmx_cache_identity,
    validate_pack_metadata,
)

# Temporary measure to compile .so for HTP without calling the Triton driver
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from .hexagon_options import HexagonOptions

# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
# mypy compiler.py hexagon_executor.py hexagon_launcher_base.py torch_mlir_hexagon_launcher.py triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs


def _get_triton_shared_opt_path(device_type: str) -> str:
    path = os.getenv(
        "TRITON_SHARED_OPT_PATH",
        "",
    )
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")

    bin_path = Path(path).resolve()

    if not bin_path.exists() or not bin_path.is_file():
        raise FileNotFoundError(
            f"Could not find 'triton-shared-opt' at expected location: {bin_path}"
        )
    if not os.access(bin_path, os.X_OK):
        raise PermissionError(
            f"'triton-shared-opt' exists but is not executable: {bin_path}"
        )

    return str(bin_path)


def ttir_to_ttsharedir(mod: str, options):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        triton_shared_opt_path = _get_triton_shared_opt_path(options.device_type)
        subprocess.check_call(
            [
                triton_shared_opt_path,
                src_path,
                "--triton-to-linalg-experimental",
                "-o",
                dst_path,
            ]
        )

        return Path(dst_path).read_text()


# The internal, IR-borne request for the record-only v3 envelope.  It is a module
# attribute rather than a backend option on purpose: nothing a user can pass
# should be able to switch the wire envelope of a production compilation.
HMX_RECORD_V3_MARKER = "hmx.diagnostic_v3_record"


def _reject_cacheable_record_only_module(mod: str) -> None:
    """Refuse a record-only module on any path that uses Triton's object cache.

    The key fact, stated precisely because the rest of this depends on it.  The
    Triton object cache is keyed by ``HexagonBackend.hash()``, the effective
    options, and the Triton AST source.  The record-mode marker lives in the
    *ttsharedir* module, which ``ttir_to_ttsharedir`` produces **downstream** of
    every one of those inputs.  A downstream artifact is not a key input: no
    part of the key contract promises that a marked module hashes differently from
    its unmarked twin, and nothing here may rely on it doing so.

    So the current absence of a collision is incidental, not structural.  It
    happens to hold today because no production path can set the marker and the
    direct-binding callers bypass this cache entirely -- and that is a property of
    today's callers, not of the key.  If a marked module ever reached this
    function, the two modes would share a cache entry, and because the record
    child rides in the same packed metadata as the execution child, a v1 hit
    would drop ``hmx_record`` silently.  It could not be caught afterwards
    either: a cache hit does not re-run a stage, so no consumer ever sees the IR
    that chose the envelope.

    This refusal is what turns non-collision from an accident into a property, so
    it is not to be relaxed on the reasoning that marked modules "cannot get
    here".  They can: this check is the only thing saying so.

    A key-based separation would need the mode to be a key input, and the only
    Python-visible candidates are the target and the backend options -- an option
    being exactly the user-facing switch this migration must not have.  The
    record-only envelope therefore stays out of this cache, while remaining fully
    implemented and reachable through the direct backend binding, which the MLIR
    fixtures and host tests use.

    Fail-closed by construction: there is no fallback that drops the marker and
    quietly compiles a v1 kernel instead.
    """
    if not isinstance(mod, str) or HMX_RECORD_V3_MARKER not in mod:
        return
    raise RuntimeError(
        f"refusing to compile a {HMX_RECORD_V3_MARKER!r} module through the Triton "
        "object cache. The marker is applied to the ttsharedir module, downstream of "
        "every Triton cache-key input (HexagonBackend.hash(), the effective options and "
        "the Triton AST source), so it is not a key input: a marked module and its "
        "unmarked twin would share one cache entry, and a v1 hit would drop the record "
        "child without any error. Absence of that collision today is incidental to the "
        "callers, not a property of the key. There is no backend option that makes the "
        "key see it. Use the direct backend binding (translate_linalg_to_obj with "
        "with_meta=true) for record-only diagnostics."
    )


# NOTE: This func has significant overlap with ttsharedir_to_obj(). This is intentional.
# This is a stopgap solution to allow more control over compiling kernels for QNN custom ops.
# When we later have a better way to pass compilation flags to ttsharedir_to_obj(),
# this function can be deprecated- we'd use ttsharedir_to_obj() in all cases.
def ttsharedir_to_llir(mod: str, options, metadata=None):
    _reject_cacheable_record_only_module(mod)
    if metadata is None:
        metadata = {}
    context = ir.context()
    qcom_hexagon_backend.load_dialects(context)
    mlir_mod = qcom_hexagon_backend.parse_mlir_module_from_str(mod, context)

    metadata["name"] = qcom_hexagon_backend.extract_func_name_from_mlir_module(mlir_mod)
    return_values = qcom_hexagon_backend.get_return_list(mlir_mod, metadata["name"])
    metadata["return_types"] = parse_return_types(return_values)

    options_map = {k: str(v) for k, v in (options.__dict__).items()}
    llvmir, translation_metadata = (
        qcom_hexagon_backend.translate_linalg_to_llvmir(mlir_mod, options_map)
    )
    apply_translation_metadata(metadata, translation_metadata)
    return llvmir


def ttsharedir_to_obj(mod: str, options, metadata=None) -> bytes:
    _reject_cacheable_record_only_module(mod)
    if metadata is None:
        metadata = {}
    context = ir.context()
    qcom_hexagon_backend.load_dialects(context)
    # Temporary regex substitution to lower tt.scan
    mod = re.sub(r"\btt\.scan\b", "ttx.scan", mod)
    mlir_mod = qcom_hexagon_backend.parse_mlir_module_from_str(mod, context)

    # Parses kernel name and return signature from mlir module
    metadata["name"] = qcom_hexagon_backend.extract_func_name_from_mlir_module(mlir_mod)
    return_values = qcom_hexagon_backend.get_return_list(mlir_mod, metadata["name"])
    metadata["return_types"] = parse_return_types(return_values)

    options_map = {k: str(v) for k, v in (options.__dict__).items()}
    # TODO: Move setting benchmarking iterations when additional stage for shared object creation is part of compilation pipeline.
    metadata["iterations"] = options_map["iterations"]
    metadata["scratch"] = options_map["scratch"]
    metadata["enableMultiThreading"] = options_map["enableMultiThreading"]
    metadata["enableThreadedDispatch"] = options_map["enableThreadedDispatch"]
    metadata["enableLWP"] = options_map["enableLWP"]

    # TODO: The lowering pipeline needs to be refactored similar to other Triton backends to
    # have a dynamic pipeline filtered by options with each pass represented by a pybind function.
    # See make_ttgir() in nvidia backend as an example.
    # `with_meta=True` returns the versioned translation metadata envelope.  It
    # carries two independent JSON objects: the launcher-facing weight
    # pre-pack contract and the host-only HMX manifest.  Unpack and validate
    # both before any object code is returned; never turn a missing feature
    # into an empty/default object.
    mods_llvmir_bytes, translation_metadata = (
        qcom_hexagon_backend.translate_linalg_to_obj(mlir_mod, options_map, True)
    )
    apply_translation_metadata(metadata, translation_metadata)
    # Note: translate_linalg_to_obj() now returns a collection of object codes in general,
    # which in the case of the triton flow will only contain one element (i.e. one object code)
    # since there is no separation of constants for the triton flow.
    # So we need to get the first and unique one here
    principal_mod_bytes = mods_llvmir_bytes[0]
    return principal_mod_bytes


# Temporary measure for .so generation, only for Triton-HTP integration
def obj_to_so(mod, metadata={}) -> str:
    tmpdir = tempfile.mkdtemp()
    kernel_code = os.path.join(tmpdir, metadata["name"] + ".o")
    Path(kernel_code).write_bytes(mod)
    so_path = HexagonExecutor(compile_only=True).generate_shared_object(
        "", kernel_code, htp_kernel_gen=True
    )
    return so_path


class HexagonBackend(BaseBackend):
    # TODO: Setting the binary extension of the final executable as .o,
    # to be updated to .so if additional stages are added.
    binary_ext = "o"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.version_key = None
        # Effective backend options, set by parse_options() and consumed by
        # hash(). Held at the dataclass defaults until the first parse (the
        # autotuner hashes a freshly constructed backend that never parses).
        self._parsed_options = HexagonOptions()

    @no_type_check  # Forced to ignore typing since base class uses deprecated annotation
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "hexagon"

    def hash(self) -> str:
        """Backend identity for triton's kernel cache key (triton/runtime/cache.py).

        Covers the target plus every effective backend option, so changing an
        option re-keys the cache instead of reusing a kernel compiled under the
        old one. The compiled backend library itself is deliberately not part
        of this hash; the semantic HMX shape/tail ABI token is mixed in as
        well. tools/hexmlir/env.sh partitions TRITON_CACHE_DIR on the
        libtriton.so identity for the compiled library.

        The HMX wire identity is a first-class field of that key: the envelope
        schema, the manifest schema, the accounting mode and the
        proof/observation producer ABI all change what a cached artifact *means*,
        not just how it was produced, so a build that changes one of them misses
        the cache instead of reusing an entry written under the old meaning.

        What this key does *not* do is separate the record-only envelope from the
        production one.  Those modes are chosen by an IR attribute, and this
        method has no per-source input, so no value of it can tell them apart.
        The record-only mode is therefore kept out of this cache entirely -- see
        ``_reject_cacheable_record_only_module`` for the mechanism and the
        reason -- rather than pretending the key separates something it cannot.
        """
        cache_identity = "-".join(
            (
                hmx_cache_identity(
                    envelope_schema=TRANSLATION_METADATA_SCHEMA,
                    manifest_schema=HMX_MANIFEST_SCHEMA,
                    accounting_mode=HMX_ACCOUNTING_MODE,
                    semantic_hmx_abi=HMX_SHAPE_TAIL_ABI_VERSION,
                    options_hash=self._parsed_options.hash(),
                ),
                str(self.target),
            )
        )
        return hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()

    def parse_options(self, opts) -> Any:
        assert self.target.backend == "hexagon"
        args = {
            k: opts[k] for k in HexagonOptions.__dataclass_fields__.keys() if k in opts
        }
        hexagon_opts = HexagonOptions(**args)

        # When external VTCM scratch is enabled (scratch > 0), automatically
        # configure flags for correct SPMD behavior at compile time so that
        # both the compiled IR and the generated wrapper are consistent:
        # - Disable enableConvertToHexagonmem to prevent VTCMPool from
        #   concurrently trying to allocate VTCM alongside the external pool.
        # - Disable enableHexagonmemCopyToDMA to avoid DMA/thread conflicts.
        # - Enable enableThreadedDispatch so instances run in parallel on
        #   real qurt hardware threads (handled at wrapper generation time).
        # Users do not need to set these flags manually when scratch > 0.
        if hexagon_opts.scratch > 0:
            hexagon_opts = replace(
                hexagon_opts,
                enableMultiThreading=False,
                enableConvertToHexagonmem=False,
                enableVTCMTiling=True,
                enableThreadedDispatch=True,
            )

        self._parsed_options = hexagon_opts
        return hexagon_opts

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm, False)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ttir")
        return mod

    # May need to add num_warps
    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: ttir_to_ttsharedir(src, options)
        if options.htp_kernel_gen:
            if options.target_artifact == "llir":
                stages["llir"] = lambda src, metadata: ttsharedir_to_llir(
                    src, options, metadata
                )
                # A dummy "o" stage is necessary because Triton requires a stage corresponding to the default binary extention (".o")
                # There is no hook we can use to change the binary extension after HexagonBackend is instantiated but before
                # self.binary_ext is used. We cannot create the .o and the .llir file in the same pipeline, as the
                # output of one stage must directly feed into the next stage (and foregoing the llir stage leaves us
                # without a direct way to return the llir). The only way around this hack is to use the output of the llir
                # stage as the input of the .o stage, which we explicitly work around to support the torch-mlir workflow.
                stages["o"] = lambda src, metadata: b"__DUMMY_OBJ_FILE_PLACEHOLDER__"
            elif options.target_artifact == "so":
                stages["o"] = lambda src, metadata: ttsharedir_to_obj(
                    src, options, metadata
                )
                stages["so"] = lambda src, metadata: obj_to_so(src, metadata)
            else:  # target_artifact == "o"
                stages["o"] = lambda src, metadata: ttsharedir_to_obj(
                    src, options, metadata
                )
        else:  # Default compilation pipeline
            assert options.device_type == "hexagon"

            stages["o"] = lambda src, metadata: ttsharedir_to_obj(
                src, options, metadata
            )

    # Skipping those methods below for now

    def add_meta_info(self, ir, cur_module, next_module, metadata, asm):
        metadata["name"] = "hexagon_backend_name"
        metadata["shared"] = "0"

    def get_stream(self):
        return ""

    def get_current_device(self):
        return ""

    def set_current_device(self, device):
        pass

    def get_kernel_bin(self):
        return "tthir"

    def get_version_key(self):
        if self.version_key is None:
            self.version_key = compute_core_version_key()  # type: ignore[name-defined]
        return self.version_key

    # Loading dialects for each layer seperately
    def load_dialects(self, context):
        return

    def get_codegen_implementation(self, options):
        # We don't have limitations for minimum dot size for our architecture, so we just
        # add a placeholder with the minimum size values.
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        """Turn the compilation metadata into the dict the launcher reads.

        Field names live in backend/utils.py (PACK_METADATA_*): this object is
        handed to the launcher as one opaque argument and read back there by
        name, so the contract is the key set, not a positional order. A stale
        cached JSON (written before a field existed) raises here with the field
        names instead of silently shifting every index downstream.
        """
        packed_fields = sorted(
            getattr(metadata, "_fields", [k for k in dir(metadata) if not k.startswith("_")])
        )
        missing = [k for k in PACK_METADATA_REQUIRED if not hasattr(metadata, k)]
        if missing:
            raise RuntimeError(
                f"compiled kernel metadata is missing {missing} (has {packed_fields}); "
                "this cached artifact was written by an older backend - clear "
                "TRITON_CACHE_DIR (tools/hexmlir/env.sh)"
            )
        packed = {k: getattr(metadata, k) for k in PACK_METADATA_REQUIRED}
        return validate_pack_metadata(packed)

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.backends.qcom_hexagon_backend.hexagon_extern.hexagon import (
            libdevice,
        )

        return {"triton.language.extra.libdevice": libdevice}
