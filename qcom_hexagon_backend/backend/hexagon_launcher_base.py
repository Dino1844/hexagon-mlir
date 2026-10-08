# ===- hexagon_launcher_base.py ---------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

from pathlib import Path
import hashlib
import os, tempfile
from datetime import datetime
from typing import Optional
from torch import Tensor  # For type annotations
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from triton._C.libtriton import qcom_hexagon_backend  # type: ignore

# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
# mypy compiler.py hexagon_executor.py hexagon_launcher_base.py torch_mlir_hexagon_launcher.py triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs


def make_resident_scope_id(kernel_obj: bytes, func_name: str) -> tuple[int, int]:
    """Derive a stable 128-bit identity for one principal kernel object."""
    digest = hashlib.sha256()
    digest.update(len(kernel_obj).to_bytes(8, "little"))
    digest.update(kernel_obj)
    digest.update(func_name.encode("utf-8"))
    raw = digest.digest()[:16]
    return (
        int.from_bytes(raw[:8], "little"),
        int.from_bytes(raw[8:], "little"),
    )


class WrapperGeneratorStrings:
    def __init__(self):
        self.input_tensor_name = "T"
        self.memrefdesc_name = "dt"
        self.scalar_name = "S"
        self.output_tensor_name = "O"

        self.input_wrapper_name = "wrDt"
        self.input_wrapper_ptr_name = "pDt"

        self.tensor_init_template = """
    Tensor<{tensor_ctype}, {tensor_rank}> {input_tensor_prefix}{i}({{{sizes}, {strides}, 128}}, MemType::HEAP);\nMemRefDescriptor<{tensor_ctype}, {tensor_rank}> *{memrefdesc_name}{i} = {input_tensor_prefix}{i}.toMemRefDesc(); \n"""
        self.tensor_init_template_rank_0 = """
    Tensor<{tensor_ctype}, 0> {input_tensor_prefix}{i};\nMemRefDescriptor<{tensor_ctype}, 0> *{memrefdesc_name}{i} = {input_tensor_prefix}{i}.toMemRefDesc(); \n"""
        self.scalar_init_template = (
            """{scalar_ctype} {scalar_prefix}{i} = {scalar_val}; \n"""
        )

        self.result_tensor_init_template = """
    Tensor<{tensor_ctype}, {tensor_rank}> {output_tensor_prefix}{i}(r->r{i}, MemType::HEAP, 128);\n"""
        self.extern_llvm_func_no_return_args = (
            """int64_t, MemRefDescriptor<{tensor_ctype}, {tensor_rank}> *, """
        )
        self.extern_llvm_func_with_return_args = (
            """ MemRefDescriptor<{tensor_ctype}, {tensor_rank}> *, """
        )
        self.result_struct_init = """FuncResult *r = new FuncResult;\n"""
        self.result_struct_def = """
struct FuncResult
{{{result_fields}
}};
"""
        self.extern_llvm_func_args_tensor = """ {tensor_pointer_ctype} *, """
        self.extern_llvm_func_args_scalar = """{scalar_ctype}, """
        self.extern_llvm_func_defn = (
            """extern "C" void {kernel_name}({function_arg_string});"""
        )
        self.extern_llvm_func_call = """{func_name}({descriptor_string});\n"""
        self.load_from_file_string = """{input_tensor_prefix}{idx}.load_from_file("{path}/{file_name}_t{idx}.raw");\n"""
        self.dump_to_file_string = """{output_tensor_prefix}{idx}.dump_to_file("{path}/{file_name}_o{idx}.raw");\n"""
        self.call_lwp = """WriteLWPOutput("{path}/{fname}.json");\n"""

        self.func_call_and_benchmarking = """
uint64_t avg_time_us = 0, avg_pcycles = 0;

// Thread-role channel probe (S3): decide, before anything runs, whether this
// kernel is dual-role -- its HMX section outlined for the bound thread of
// HmxRoleExecutor.h -- and bind it if so. The channel is per-kernel SYMBOLS
// (bin/runtime/include/HmxRoleChannel.h): the compiler exports
// <func_name>__hmx_section + <func_name>__hmx_role_depth for a dual-role
// kernel and nothing for a legacy one, and the probe inside the runtime
// dlsym's them. The declaration below is WEAK for the same reason the
// read-out hooks further down are: a legacy kernel's LLVM object references
// none of the role executor ABI, so the archive member defining this entry
// is not linked into its .so and the reference resolves to null -- the
// guarded call is skipped and the legacy path is byte-for-byte today's. A
// dual-role kernel references hexagon_runtime_hmx_role_{{submit,drain}}
// strongly, so the member (and this entry with it) is present by
// construction.
//
// RTLD_SELF: search the calling module itself -- the wrapper, the channel
// entry and the section symbols all live in this .so. Negative return: a
// half-emitted channel or a refused bind; the kernel must NOT run (an
// unbound dual-role kernel submits into an executor that drops every group
// -- a wrong answer, not a slow one), so the launch fails loudly here.
// Return -1, not 0: the host treats a nonzero exit as a failed launch.
if (hexagon_runtime_hmx_role_channel_launch != nullptr) {{
  int32_t hmx_role_rc =
      hexagon_runtime_hmx_role_channel_launch(RTLD_SELF, "{func_name}");
  if (hmx_role_rc < 0) {{
    FARF(ERROR, "hmx-role channel refused kernel '{func_name}' (rc=%d)",
         (int)hmx_role_rc);
    return -1;
  }}
}}

// One discarded call before the timed loop. Adopted 2026-10-02.
//
// Measured on the phone (1024x512x64 f16, per-iteration pcycle trace): the first
// matmul_kernel call inside a launch costs about 2.5M cycles and 2.7 ms, every
// later call in the same launch about 112k cycles and 53 us -- a 22x cycle and
// 50x wall-clock gap. The gap is bimodal with no ramp: the first call implies
// 0.92-0.97 GHz against 2.08-2.14 GHz for all later ones, so the HMX clock is
// not at its final corner yet.
//
// Why the clock is low on that first call: HexagonAPI::AcquireResources() runs
// per launch, not per process -- its constructor calls it (HexagonAPI.h:52) and
// it calls initialize_and_acquire_hmx() (HexagonAPI.h:72), which asks
// HAP_power_set for HMX_v2 with set_clock = TRUE, target_corner =
// HAP_DCVS_EXP_VCORNER_MAX and perf_mode = HAP_CLK_PERF_HIGH
// (HexagonAPI.cpp:231, :234, :235, :238) and then blocks in
// HAP_compute_res_acquire with a 100 ms timeout (HexagonAPI.cpp:261). The
// power-up and clock ramp therefore happen inside every launch, and the
// kernel's first call pays for them.
//
// Two consequences that make this worth doing rather than documenting:
//   * Every Perf before this change divided that one-off cost by iterations, so
//     at N=10 a shape read 320 us instead of 52 us -- a 6x error that shrinks
//     as 1/N, which is exactly the t(N) = A + B/N shape that made B look like
//     a per-kernel property for so long. Historical Perf numbers are
//     cold-start-inclusive; see docs/state/CURRENT.md 3.11.
//   * It cannot be warmed away from outside. WARMAB_AB_WARMUP=1 runs a
//     discarded case in a SEPARATE launch (shape_pair.py:187-189) and leaves B
//     unchanged (2,880 us with it, 2,624 us without, measured 2026-10-02). Only
//     a call inside this launch can.
//
// After the change, across nine shapes, B falls from 2,624-2,880 us to -10..+20
// us -- inside the 30 us noise floor implied by a 3 us rep spread over a 0.099
// span -- and once_share at N=1000 drops from 4.7-5.2% (over the 2% threshold)
// to at most 0.106%. The steady-state value A moves only 53.4 -> 52.0 us, so
// ratio-type conclusions are untouched; what was wrong was the once term.
//
// The call is discarded, so the reported number is the warm steady state that
// the roadmap's comparisons intend. Reverting this line restores the
// cold-start-inclusive number.
{function_call}

benchmark_time_and_pcycles({iterations}, [&]() {{
    {function_call}
}}, &avg_time_us, &avg_pcycles);
TestReport tr("{func_name}", avg_time_us, "us", Result::Pass, "{save_path}");
tr.save();
// Deferred-drain barrier for the HMX read-out split (2026-10-04). When the
// kernel was compiled with hmx-readout-deferred-drain, the LAST call of the
// loop above returned with its tail batch still in flight; this is the drain
// that would otherwise have sat inside every iteration. It must run before
// anything reads this launch's outputs, and it is OUTSIDE the timed region
// by construction. Weak for the same reason as the dump hook below: kernels
// that do not reference the executor ABI strongly (every non-readout kernel)
// resolve it to null and skip the call; readout kernels already pull the
// HmxVectorExecutor object into their .so through configure/publish, so the
// symbol resolves at static link time. For a readout kernel compiled
// WITHOUT deferral this is a no-op: its own exit drain already emptied the
// ring, and drainLocked on an empty ring returns without parking.
if (hexagon_runtime_hmx_exec_drain != nullptr)
  hexagon_runtime_hmx_exec_drain();
// The processor-cycle average, from the SAME pass as the microsecond one above.
// APPENDED to the same report file rather than printed: the device's stdout is
// not captured by the executor, so a printf here is silently lost -- the
// Test_Info block that reaches the host comes from TestReport::save() writing
// this file, and hexagon_executor prints the file's whole contents. The line is
// additive and contains no "Result", so get_test_result() (which returns the
// first line containing "Result") is unaffected.
//
// Why it has to come from the same pass: C15:14 stops while the DSP is
// clock-gated (HAP_perf.h:79) while the 19.2 MHz qtimer behind Perf does not, so
// the ratio between the two is a property of the run, not a constant of the part
// -- measured 2.10-2.15 GHz in a tight leaf loop and 1.10-1.15 GHz inside a full
// matmul on one build. Any microsecond measurement that gets converted to cycles
// needs this number from its own run; see
// docs/hmx/hmx-perf-findings-2026-09-27.md 1.
{{
  FILE *pcf = fopen("{save_path}", "a");
  if (pcf) {{ fprintf(pcf, "PerfPcycles:%llu\\n", (unsigned long long)avg_pcycles); fclose(pcf); }}
}}
// Live-counter dump for the HMX read-out split diagnostics: appends the
// executor's accumulated counters and timing rings to this same report file,
// from the same launch context that demonstrably has working file I/O --
// unlike the executor's own accounting path, which never came back from a
// FastRPC-launched kernel (roadmap/ERRATA.md errata 12 section 5.1). Placed
// here, AFTER benchmark_time_and_pcycles returned, so the file I/O cannot
// perturb Perf. Null-guarded: the declaration in the headers block is weak,
// and kernels that do not link the async runtime leave it null.
if (hexagon_runtime_hmx_exec_dump != nullptr)
  hexagon_runtime_hmx_exec_dump("{save_path}");
"""

        # Codegen string for the Headers in the generated CPP file.
        self.code_headers = """
#define FARF_ALWAYS 1
#include <HAP_farf.h>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include "common.h"
#include "tensor.h"
#include "hexagon_benchmark.h"
#include "test_report.h"
#include "CRunnerUtils.cpp"
#include "debug.h"
#include "prof_utils.h"
extern "C" int hexagon_runtime_resident_scope_enter_v2_dsp(uint64_t low64,
                                                            uint64_t high64)
    __attribute__((weak));
// Deferred-drain barrier for the HMX read-out split (2026-10-04). Weak for
// the same reason as the declaration above and the dump hook below: kernels
// that do not reference the executor ABI strongly (every non-readout kernel)
// resolve it to null and the guarded call in the benchmarking template is
// skipped. Readout kernels reference hexagon_runtime_hmx_exec_{configure,
// publish} strongly from their LLVM object, so the archive member defining
// this symbol is already in their .so.
extern "C" void hexagon_runtime_hmx_exec_drain(void)
    __attribute__((weak));
// Live-counter dump hook for the HMX read-out split (2026-10-04). Weak for
// the same reason as the declaration above: kernels that do not reference the
// executor ABI strongly (every non-readout kernel) resolve it to null and the
// guarded call in the benchmarking template below is skipped. Read-out
// kernels reference hexagon_runtime_hmx_exec_* strongly from their LLVM
// object, so the archive member defining this symbol is already in their .so.
extern "C" void hexagon_runtime_hmx_exec_dump(const char *path)
    __attribute__((weak));
// Thread-role channel probe (S3, bin/runtime/include/HmxRoleChannel.h): the
// launch-side half of the dual-role dispatch. Weak for the same reason as
// the three declarations above: only a dual-role kernel's .so contains the
// archive member that defines it (the kernel's strong submit/drain
// references pull it in), so legacy kernels resolve it to null and the
// guarded call in the benchmarking block is skipped. dlfcn.h is included
// for RTLD_SELF, the handle that names this .so to the probe.
#include <dlfcn.h>
extern "C" int32_t hexagon_runtime_hmx_role_channel_launch(void *handle,
                                                           const char *entry)
    __attribute__((weak));
"""

        # Codegen string for the data structures and functions defined before the main body.
        self.code_define = """
{llvm_func_sign}
{input_wrapper_struct_def}
{result_struct_def}
"""

        # Codegen string for the contents of main body.
        self.code_body = """
int main() {{
{resident_scope_setup}
{tensor_definition_str}
{input_wrapper_structs_init}
{result_struct_init}
{read_from_file_calls}
{benchmarking_and_reporting}
{update_tensor}
{write_to_file_calls}
{lwp}
return 0;
}}
"""

        # Codegen string for the entire CPP file.
        self.code_string = """
{code_headers}
{code_define}
{code_body}
"""


class HexagonWrapperGenerator:
    def __init__(
        self,
        input_profs,
        iterations,
        func_name,
        output_profs,
        common_strings,
        options,
        resident_scope_id: tuple[int, int] | None = None,
    ):
        """
        Initialize the HexagonWrapperGenerator instance.

        Parameters:
            input_profs (list): A list of input tensors and scalars for the kernel.
            func_name (str): The name of the LLVM IR function to be defined/called.
            output_profs (list): A list of output profiles.
            common_strings(WrapperGeneratorStrings): Set of strings with placeholder to be updated.
        """
        self.input_profs = input_profs
        self.iterations = iterations
        self.func_name = func_name
        self.output_profs = output_profs
        self.common_strings = common_strings
        self.enable_lwp = options["enableLWP"]
        self.resident_scope_id = resident_scope_id

    def generate_resident_scope_setup(self) -> str:
        # The weak declaration is for ordinary kernels that do not pull the
        # runtime module at all.  Any kernel that actually uses a resident-v2
        # call has a strong reference to that ABI in its LLVM object, so the
        # linker still fails closed if the runtime implementation is missing.
        if self.resident_scope_id is None:
            return ""
        low64, high64 = self.resident_scope_id
        return f"""
if (hexagon_runtime_resident_scope_enter_v2_dsp != nullptr &&
    hexagon_runtime_resident_scope_enter_v2_dsp(0x{low64:x}ULL, 0x{high64:x}ULL) != 0) {{
  FARF(ERROR, "resident scope registration failed");
  return -1;
}}
"""

    def generate_input_declarations(self):
        """
        Generate declaration for each input
        for torch tensors, write them in Tensor class form,
        for scalars, declare and intialize a variable with scalar input's value.
        """
        formatted_tensor_string = ""

        for inp in self.input_profs:
            if inp.input_type == "tensor":
                if inp.rank == 0:
                    tensor_declaration = (
                        self.common_strings.tensor_init_template_rank_0.format(
                            tensor_ctype=inp.dtype,
                            i=inp.idx,
                            memrefdesc_name=self.common_strings.memrefdesc_name,
                            input_tensor_prefix=self.common_strings.input_tensor_name,
                        ).lstrip()
                    )
                else:
                    tensor_declaration = (
                        self.common_strings.tensor_init_template.format(
                            tensor_ctype=inp.dtype,
                            tensor_rank=inp.rank,
                            i=inp.idx,
                            sizes=inp.shape[0],
                            strides=inp.shape[1],
                            memrefdesc_name=self.common_strings.memrefdesc_name,
                            input_tensor_prefix=self.common_strings.input_tensor_name,
                        ).lstrip()
                    )
                formatted_tensor_string += tensor_declaration
            else:
                scalar_declaration = self.common_strings.scalar_init_template.format(
                    i=inp.idx,
                    scalar_ctype=inp.dtype,
                    scalar_val=inp.value,
                    scalar_prefix=self.common_strings.scalar_name,
                ).lstrip()
                formatted_tensor_string += scalar_declaration
        return formatted_tensor_string

    def generate_llvm_function_signature_arg_string(self):
        """Generate common function signature args"""
        function_arg_string = ""
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                if len(self.output_profs) > 0:
                    extern_args = (
                        self.common_strings.extern_llvm_func_with_return_args.format(
                            tensor_ctype=inp.dtype, tensor_rank=inp.rank
                        )
                    )
                else:
                    extern_args = (
                        self.common_strings.extern_llvm_func_no_return_args.format(
                            tensor_ctype=inp.dtype, tensor_rank=inp.rank
                        )
                    )
            else:
                extern_args = self.common_strings.extern_llvm_func_args_scalar.format(
                    scalar_ctype=inp.dtype
                )
            function_arg_string += extern_args
        return function_arg_string[:-2]

    def generate_llvm_function_signature(self):
        """Generate lowered LLVM function definition to be called from CPP launcher"""
        raise NotImplementedError("LLVM function signature generation not implemented.")

    def generate_llvm_function_call_arg_string(self):
        """Generate common function call args"""
        function_call_descriptor_string = ""
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                if len(self.output_profs) == 0:
                    function_call_descriptor_string += f"{inp.rank}, "
                function_call_descriptor_string += (
                    f"{self.common_strings.memrefdesc_name}{inp.idx}, "
                )
            else:
                function_call_descriptor_string += (
                    f"{self.common_strings.scalar_name}{inp.idx}, "
                )
        return function_call_descriptor_string[:-2]

    def generate_llvm_function_call(self):
        """Generates actual function call to lowered LLVM function call"""
        raise NotImplementedError("LLVM function call generation not implemented.")

    def generate_input_wrapper_struct_def(self):
        """Generates template for input wrapper structs"""
        raise NotImplementedError("Unimplemented- requires definition by frontend.")

    def generate_input_wrapper_structs_init(self):
        """Generates initializations for input wrapper structs"""
        raise NotImplementedError("Unimplemented- requires definition by frontend.")

    def generate_benchmarking_and_reporting(self, function_call, exec_dir):
        return self.common_strings.func_call_and_benchmarking.format(
            iterations=self.iterations,
            function_call=function_call,
            func_name=self.func_name,
            save_path=f"{exec_dir}/perf.txt",
        )

    def generate_update_tensor_calls(self):
        """Generates a function call to update the tensor with the result of the output"""

        formatted_result_string = ""

        # TODO: Currently skipping returning scalar values, since writing mechanism is not in place.
        # To be updated later with the usecase which returns scalar values.
        for idx, ret in enumerate(self.output_profs):
            if ret.rank:
                tensor_declaration = (
                    self.common_strings.result_tensor_init_template.format(
                        output_tensor_prefix=self.common_strings.output_tensor_name,
                        tensor_ctype=ret.dtype,
                        tensor_rank=ret.rank,
                        i=idx,
                    ).lstrip()
                )
                formatted_result_string += tensor_declaration
        return formatted_result_string

    def generate_result_struct(self):
        """Generates the Result string"""
        if len(self.output_profs) > 0:
            result_fields_string = ""
            for idx, res in enumerate(self.output_profs):
                if res.rank:
                    result_fields_string += (
                        f"\n\tMemRefDescriptor<{res.dtype},{res.rank}> r{idx};"
                    )
                else:
                    result_fields_string += f"{res.dtype} r{idx};\n"
            return self.common_strings.result_struct_def.format(
                result_fields=result_fields_string
            )
        return ""

    def generate_result_struct_init(self):
        if len(self.output_profs) > 0:
            return self.common_strings.result_struct_init
        return ""

    def generate_tensor_read_from_file_calls(self, file_name, exec_dir):
        """Generates calls to read tensors from file"""
        read_from_file_string = ""

        for inp in self.input_profs:
            if inp.rank:
                tensor_idx = inp.idx
                write_call = self.common_strings.load_from_file_string.format(
                    input_tensor_prefix=self.common_strings.input_tensor_name,
                    idx=tensor_idx,
                    path=exec_dir,
                    file_name=file_name,
                )
                read_from_file_string += write_call
        return read_from_file_string

    def generate_lwp_call(self):
        if self.enable_lwp:
            # Same base as HexagonExecutor.device_path; /data/local/tmp is not
            # writable on every target (e.g. a Termux app).
            return self.common_strings.call_lwp.format(
                path=os.getenv("HEXAGON_DEVICE_BASE", "/data/local/tmp"), fname="lwp"
            )
        return ""

    def generate_tensor_write_to_file_calls(self, fname, exec_dir):
        """Generates calls to dump tensors to file"""
        if len(self.output_profs) > 0:
            profs = self.output_profs
            prefix = self.common_strings.output_tensor_name
        else:
            profs = self.input_profs
            prefix = self.common_strings.input_tensor_name

        write_to_file_string = ""
        for out in profs:
            # TODO: add scalar/bool support
            if out.rank:
                write_to_file_string += self.common_strings.dump_to_file_string.format(
                    output_tensor_prefix=prefix,
                    idx=out.idx,
                    path=exec_dir,
                    file_name=fname,
                )
        return write_to_file_string

    def generate_cpp_code_headers(self) -> str:
        code_headers = self.common_strings.code_headers
        return code_headers

    def generate_cpp_code_define(self) -> str:
        code_define = self.common_strings.code_define.format(
            llvm_func_sign=self.generate_llvm_function_signature(),
            input_wrapper_struct_def=self.generate_input_wrapper_struct_def(),
            result_struct_def=self.generate_result_struct(),
        )
        return code_define

    def generate_cpp_code_body(self, file_name, exec_dir) -> str:
        code_body = self.common_strings.code_body.format(
            resident_scope_setup=self.generate_resident_scope_setup(),
            tensor_definition_str=self.generate_input_declarations(),
            input_wrapper_structs_init=self.generate_input_wrapper_structs_init(),
            result_struct_init=self.generate_result_struct_init(),
            read_from_file_calls=self.generate_tensor_read_from_file_calls(
                file_name, exec_dir
            ),
            benchmarking_and_reporting=self.generate_benchmarking_and_reporting(
                self.generate_llvm_function_call(), exec_dir
            ),
            update_tensor=self.generate_update_tensor_calls(),
            write_to_file_calls=self.generate_tensor_write_to_file_calls(
                file_name, exec_dir
            ),
            lwp=self.generate_lwp_call(),
        )
        return code_body

    def generate_cpp_wrapper(self, file_name, exec_dir) -> str:
        """
        Generates skeleton cpp file which calls a function.
        """
        code_headers = HexagonWrapperGenerator.generate_cpp_code_headers(self)
        code_define = HexagonWrapperGenerator.generate_cpp_code_define(self)
        code_body = HexagonWrapperGenerator.generate_cpp_code_body(
            self, file_name, exec_dir
        )

        return self.common_strings.code_string.format(
            code_headers=code_headers, code_define=code_define, code_body=code_body
        )


# Utility function to create a timestamped folder, which will be used to place all
# the build artifacts (cpp wrapper, .o and .so, etc) that are generated.
# The folder will be called model_name-%Y-%m-%d_%H-%M-%S and will be put under:
# possibility 1 - the user-privided `optional_base_dir` if it is not None
# possibility 2 - the value of the environment variable HEXAGON_MLIR_DUMP_DIR
# possibility 3 - the /tmp/ directory otherwise
# which will be check in this order
# Returns (full path, leaf dirname) to the folder created
def create_timestamped_folder(
    model_name: str, optional_base_dir: Optional[str] = None
) -> str:
    # --- Part 1: deciding the base directory ---
    # Possibility 1: the user has provided the base directory
    if optional_base_dir != None:
        base_dir = optional_base_dir
        reason = "(as requested by the --output-dir argument provided)"
    else:
        # Possibility 2: otherwise look at the env variable HEXAGON_MLIR_DUMP_DIR
        env_var_dump_ir = os.getenv("HEXAGON_MLIR_DUMP_DIR")
        if env_var_dump_ir != None:
            base_dir = env_var_dump_ir
            reason = "(as specified by the env variable HEXAGON_MLIR_DUMP_DIR)"
        else:
            # Possibility 3: otherwise use a predetermined base directory
            # To prevent clutter, we create the folder under /tmp
            base_dir = "/tmp"
            reason = "(as default location)"
    assert os.path.isdir(
        base_dir
    ), f"The base directory {base_dir} ({reason}) isn't a valid directory"
    # --- Part 2: naming and creating the subfolder that will contain all the artifacts ---
    # HEXAGON_FAST_LAUNCH=1 keeps the folder name stable (no timestamp) so the
    # device directory -- and every file pushed into it -- keeps its name across
    # launches, which is what lets the adb shim's md5 push-skip (gated by the
    # same env var) recognise unchanged files instead of re-pushing 3+ MB every
    # launch (~20 s measured, for kernels that run in ~50 us). Opt-in: only the
    # sweep in exp/hmx/t3_overlap_ab sets it. Safe against collisions because
    # device work is serialised by tools/run_tests.sh lock; a stale file from a
    # previous launch is either overwritten (same name) or never loaded
    # (run_main_on_hexagon loads libs by name from this run's push set).
    if os.getenv("HEXAGON_FAST_LAUNCH") == "1":
        folder_name = model_name
        full_path = os.path.join(base_dir, folder_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path, folder_name
    # Get the current date and time
    now = datetime.now()
    # Format the folder name
    folder_name = now.strftime(model_name + "-%Y-%m-%d_%H-%M-%S-%f")
    # Full path
    full_path = os.path.join(base_dir, folder_name)
    # Create the folder
    os.makedirs(full_path)
    print(
        f"==> Folder '{folder_name}' created successfully inside of", base_dir, reason
    )
    return full_path, folder_name


class HexagonLauncherBase:
    def generate_input_output_paths(
        self, directory: str, file_name: str, wrapper: HexagonWrapperGenerator
    ) -> tuple[list[str], list[str]]:
        input_paths = []
        output_paths = []

        # P2 host pre-pack: when the compiler published a weight-residency
        # contract, the kernel reads these arguments as crouton arrays and will
        # not pack them itself, so the bytes written here must already be in
        # crouton order. `WeightPrepack` is keyed by slot and cached by content.
        # The generic Torch-MLIR wrapper has no P2 contract and keeps raw inputs.
        prepack = getattr(wrapper, "weight_prepack", None)
        consumed_slots = set()
        if prepack is not None and prepack.function_name is not None:
            canonical_name = file_name.removeprefix("_mlir_ciface_")
            if prepack.function_name not in (file_name, canonical_name):
                raise RuntimeError(
                    f"weight_prepack function {prepack.function_name!r} does not "
                    f"match launcher function {file_name!r}"
                )

        for inp in wrapper.input_profs:
            if inp.input_type == "tensor":
                i = inp.idx
                data = inp.value
                input_path = os.path.join(directory, f"{file_name}_t{i}.raw")
                payload = None
                if prepack is not None and prepack.has_slot(i):
                    consumed_slots.add(i)
                    payload = prepack.pack(data, i)
                    if payload is None:
                        raise RuntimeError(
                            f"weight slot {i} has a prepack contract but no packed image"
                        )
                    # The image is the crouton (fp16), so its length is exact:
                    # an f16 source packs to the argument's own byte count, an
                    # f32 source to half of it. Anything else is a packer or
                    # contract defect, not something to pad over.
                    expected = prepack.image_bytes(i)
                    if len(payload) != expected or expected > data.nbytes:
                        raise RuntimeError(
                            f"weight slot {i} packed {len(payload)} bytes, "
                            f"expected the {expected}-byte crouton image "
                            f"(argument is {data.nbytes} bytes)"
                        )
                    if expected < data.nbytes:
                        # The wrapper loads the argument's own byte image
                        # (`elems * sizeof(T)`); the resident copy reads only
                        # the image's prefix, so zero-pad the tail it never
                        # reads.
                        payload = payload.ljust(data.nbytes, b"\0")
                if payload is None:
                    payload = data.numpy().tobytes()
                with open(input_path, "wb") as file:
                    file.write(payload)
                input_paths.append(input_path)

        if prepack is not None:
            missing = prepack.unconsumed_slots(consumed_slots)
            if missing:
                raise RuntimeError(
                    f"weight_prepack contract has unconsumed argument slots: "
                    f"{sorted(missing)}"
                )

        # The output path count is provided by the frontend. There are
        # several cases (writing back ptrs, return values, or both)
        # that must be handled accordingly
        output_tensor_path_count = self.get_output_tensor_path_count(wrapper)
        for i in range(output_tensor_path_count):
            output_path = os.path.join(directory, f"{file_name}_o{i}.raw")
            output_paths.append(output_path)

        return input_paths, output_paths

    @staticmethod
    def get_output_tensor_path_count(wrapper_generator: HexagonWrapperGenerator):
        raise NotImplementedError("Unimplemented- requires definition by frontend.")

    # Ask the HexagonWrapperGenerator `wrapper_generator` to generate the wrapper for `file_name`, which is then written to disk,
    # knowing that the execution folder (either on device or local) will be `exec_dir`
    def generate_and_dump_wrapper(
        self,
        wrapper_generator: HexagonWrapperGenerator,
        local_dir: str,
        file_name: str,
        exec_dir: str,
    ) -> str:
        """Generates and dumps CPP wrapper file using a given wrapper generator"""
        cpp_wrapper_path = os.path.join(local_dir, file_name + "_wrapper.cpp")
        with open(cpp_wrapper_path, "w") as file:
            cpp_wrapper = wrapper_generator.generate_cpp_wrapper(file_name, exec_dir)
            file.write(cpp_wrapper)
        return cpp_wrapper_path

    def execute_kernel(
        self,
        hexec: HexagonExecutor,
        local_dir: str,
        filename_without_ext: str,
        paths_to_shared_libs_generated: list[str],
        wrapper_generator: HexagonWrapperGenerator,
    ) -> list[Tensor]:
        """
        Generates the input and output paths and runs the kernel
        """
        # Generate input/output paths
        input_tensor_paths, output_tensor_paths = self.generate_input_output_paths(
            local_dir, filename_without_ext, wrapper_generator
        )
        # Run the kernel
        results = hexec.run(
            paths_to_shared_libs_generated,
            input_tensor_paths,
            output_tensor_paths,
            generatePerf=True,
        )
        return results
