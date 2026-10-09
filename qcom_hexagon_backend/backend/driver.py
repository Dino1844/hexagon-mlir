# ===- driver.py ------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import json
import warnings

from math import prod
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.backends.qcom_hexagon_backend.triton_hexagon_launcher import (
    TritonHexagonLauncher,
    HexagonUtils,
)
from triton.backends.qcom_hexagon_backend.utils import make_profiled_return
from triton.backends.qcom_hexagon_backend.utils import hmx_manifest_warnings
from triton.backends.qcom_hexagon_backend.utils import require_pack_metadata_fields
from triton.backends.qcom_hexagon_backend.utils import summarize_hmx_manifest
from triton.backends.qcom_hexagon_backend.utils import validate_hmx_record_json


def getHexagonLauncherClass():
    class HexagonLauncher:
        def __init__(self, src, metadata):
            if not hasattr(self, "_initialized"):
                raise RuntimeError(
                    "HexagonLauncher can only be instantialized by HexagonDriver."
                )
            cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
            self.input_type_list = {
                cst_key(key): value for key, value in src.signature.items()
            }
            self.launcher = TritonHexagonLauncher()
            # The record-only v3 child of a `hex.hmx.translation/v2` envelope,
            # refreshed on every launch and readable through
            # `hmx_record_diagnostic()`.  It is deliberately *not* a launcher
            # input: a record authorizes no plan, no budget, no resident default
            # and no launch grid, so the launcher never sees it.  `None` means
            # the kernel was built with a v1 envelope.
            self.hmx_record = None
            # The v2 manifest, retained for the same reason and with the same
            # restriction as the record above: the launcher consumes it for the
            # launch contract (triton_hexagon_launcher.py:568) and then has no
            # use for it, so before this it was parsed out of packed_metadata,
            # handed to the launcher, and unreachable from host tooling. That
            # is why "why did my operator not get HMX" had no answer available
            # on the device side even though the reason code was sitting right
            # there in the metadata -- measured 2026-10-01, 8 of 41 dot-bearing
            # operators fall back to HVX and all 6 identifiable ones carry an
            # exact reason code (docs/codegen/
            # why-hmx-refusals-are-invisible-2026-10-01.md). Inspectable is not
            # the same as acted upon; see hmx_manifest_contract below.
            self.hmx_manifest = None

        def hmx_manifest_contract(self):
            """Return the compiled kernel's v2 HMX manifest, or None.

            A READ-ONLY reader, on the same terms as hmx_record_diagnostic: it
            exists so that host tooling can ask a compiled kernel what it
            decided and why. It is not a decision surface, nothing in the launch
            path consults it, and setting it would not change what runs.

            ⚠️ TIMING, measured 2026-10-01: this returns None until the kernel
            has actually been LAUNCHED once. It is assigned in __call__, from the
            same `pack_metadata["hmx_manifest"]` the launcher receives, so it
            shares its lifetime exactly -- including this limitation, which
            hmx_record_diagnostic() above has too. It is NOT populated by a
            compile-only `warmup()`: the knob's __init__ receives
            `(src, metadata)`, and that `metadata` is Triton's launch metadata,
            which does not carry `packed_metadata`. Making it available before
            the first launch needs plumbing through backend/compiler.py, and
            there is no consumer for that yet, so it was not done.

            So: useful for tests and for inspecting a kernel after a run; NOT a
            way to find out why an operator was refused without running it. For
            that, read the manifest the compiler already emitted --
            tools/hexmlir/manifest_verdict.py does exactly that over a codegen
            dump, and that is the supported route today.
            """
            return self.hmx_manifest

        def hmx_manifest_verdict(self):
            """Return the same manifest as text, or None if there is none.

            The opt-in reader: nothing in the compile path or the launch path
            calls this, so asking is the only way to get output and not asking
            leaves every default path byte-for-byte as it was.

            It shares hmx_manifest_contract()'s lifetime exactly -- same
            retained value, same "None until the first launch" limitation, same
            read-only terms.  The only difference is the form of the answer:
            the contract returns the document for a caller that wants to read
            the fields, this returns the question a user actually asks, which
            is how many contraction sites reached the HMX engine, how many were
            refused, and which ones reached it but run serially.

            The retained value is the JSON string the launcher itself consumes
            -- that is the whole packed-metadata contract -- so this parses it
            before summarizing.  Until 2026-10-08 it did not, and every real
            launch answered "nothing to summarize": the reader is unreachable
            without a launch, and no launch-path test could exercise it with
            the string form.  Pinned host-side by the fallback-notice tests.

            Measured 2026-10-01: 8 of 41 dot-bearing operators fall back to HVX
            and every identifiable one carries an exact reason code, so the
            information is present in the artifact; what was missing was a way
            to ask for it (docs/codegen/
            why-hmx-refusals-are-invisible-2026-10-01.md).
            """
            if self.hmx_manifest is None:
                return None
            return summarize_hmx_manifest(json.loads(self.hmx_manifest))

        def hmx_record_diagnostic(self):
            """Return the validated record-only v3 child, or None.

            The single reader of the retained record.  It exists so the record a
            kernel was compiled with is inspectable from host tooling; it is not
            a decision surface, and nothing in the launch path consults it.
            """
            return self.hmx_record

        def _load_hmx_record(self, packed):
            """Unpack and validate the v3 record child, if the envelope has one.

            A malformed record fails closed here rather than being downgraded to
            "no record": the driver is the only component that reads it, and a
            silent drop would leave a caller believing a kernel has no
            record-only evidence when in fact its evidence is unreadable.
            """
            record_json = packed["hmx_record"]
            if not record_json:
                return None
            try:
                validate_hmx_record_json(record_json)
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid record-only HMX v3 child in packed metadata: {exc}; "
                    "clear TRITON_CACHE_DIR (tools/hexmlir/env.sh) and rebuild "
                    "the kernel"
                ) from exc
            return record_json

        def __call__(self, *args, **kwargs):
            # args =  grid_0, grid_1, grid_2, stream, kernel.function,
            #         kernel.packed_metadata, launch_metadata,
            #         CompiledKernel.launch_enter_hook,
            #         self.CompiledKernel.launch_exit_hook, *args
            kernel_llir = args[4]
            # Validated once, then read by name: a positional tuple made every
            # appended field shift these indices silently, and `len(x) > i`
            # turned a stale cached JSON into "treat the feature as off".
            pack_metadata = require_pack_metadata_fields(args[5])
            unstructured_return_types = pack_metadata["return_types"]
            return_profs = [
                make_profiled_return(ret) for ret in unstructured_return_types
            ]
            # Extract the "name" field which has the function name.
            # Add "_mlir_ciface_" prefix if kernel has >0 returns (this changes the calling conv.)
            func_name = ("_mlir_ciface_" if len(return_profs) > 0 else "") + pack_metadata[
                "name"
            ]
            iterations = pack_metadata["iterations"]
            compiled_scratch = pack_metadata["scratch"]
            compiled_enable_multithreading = pack_metadata["enableMultiThreading"]
            compiled_enable_threaded_dispatch = pack_metadata["enableThreadedDispatch"]
            compiled_enable_lwp = pack_metadata["enableLWP"]
            weight_prepack = pack_metadata["weight_prepack"]
            hmx_manifest = pack_metadata["hmx_manifest"]
            # Retained for hmx_manifest_contract() above. Same value the
            # launcher already receives, so this cannot disagree with what ran.
            self.hmx_manifest = hmx_manifest
            # Validated for diagnostics, then deliberately not forwarded: the
            # launcher keeps using the v2 execution child alone.
            self.hmx_record = self._load_hmx_record(pack_metadata)
            # Launch-time visibility for what this kernel's manifest already
            # records but no default path showed: refused sites with the
            # compiler's own reasons, partial HMX coverage (`k/n on HMX`), and
            # the self-contradictory full-hmx record whose bridge_counts are
            # all zero -- hmx-partition emitted no HMX bridge sites, so the
            # claimed HMX execution does not exist (false green). The judgment
            # is the pure reporter in utils.py; this is only the emission.
            # warnings.warn rather than print: the interpreter's default
            # filter deduplicates by message text, so repeated launches of the
            # same kernel warn once and stay quiet for kernels with nothing to
            # report -- messages must therefore carry no per-launch content
            # (the reporter keeps them free of reps/timestamps/addresses).
            for _warning in hmx_manifest_warnings(
                hmx_manifest, pack_metadata["name"]
            ):
                warnings.warn(_warning, stacklevel=2)
            num_fixed_args = 9
            inputs_with_constants = list(args[num_fixed_args:])
            inputs = [
                inp
                for idx, inp in enumerate(inputs_with_constants)
                if not self.input_type_list[idx] == "constexpr"
            ]
            launch_grid = (args[0], args[1], args[2])
            if prod(launch_grid) < 1:
                raise ValueError(
                    """
                    Must set at least 1 thread in SPMD launch grid.
                    To launch singlethreaded, invoke kernel as follows:
                    your_kernel[(1,)](...)
                    """
                )
            self.launcher._exec_kernel(
                kernel_llir,
                iterations,
                func_name,
                inputs,
                return_profs,
                launch_grid,
                compiled_scratch=compiled_scratch,
                compiled_enable_multithreading=compiled_enable_multithreading,
                compiled_enable_threaded_dispatch=compiled_enable_threaded_dispatch,
                compiled_enable_lwp=compiled_enable_lwp,
                weight_prepack=weight_prepack,
                hmx_manifest=hmx_manifest,
                runtime_options=kwargs,
            )
            # TODO: There seems to be no way to propogate the call returns upward, because
            #    - The call result is not used by the caller
            #    - The packed_metadata field (or any input arg) is immutable, because the args are passed as a tuple
            #    - This HexagonLauncher class doesn't inherit from the caller (CompiledKernel)
            # To support direct returns, we'd need to make very simple changes to the upstream compiler.py

    return HexagonLauncher


def ty_to_cpp(ty):
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


class HexagonDriver(DriverBase):
    def __init__(self, device_type="dsp"):
        self.utils = HexagonUtils()
        self.backend = "HEXAGON"
        # Kept as driver state (upstream reads it); it is deliberately *not*
        # threaded into getHexagonLauncherClass, which never used it.
        self.device_type = device_type
        instance = getHexagonLauncherClass()
        instance._initialized = True
        self.launcher_cls = instance
        # Dummy executable/binary set, since there is no binary for our usecase.
        self.binary_ext = "hex"

    @staticmethod
    def is_active():
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_current_target(self):
        return GPUTarget("hexagon", 0, 0)

    def get_current_device(self):
        return GPUTarget("hexagon", 0, 0)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def get_current_stream(self, device):
        return None

    def get_device_interface(self):
        import torch

        return torch.cpu

    def get_empty_cache_for_benchmark(self):
        import torch

        device = "cpu"
        # 256MB cache
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device=device)

    def clear_cache(self, cache):
        cache.zero_()
