#!/usr/bin/env python3
"""Focused source contract for the allocation-result fail-closed guard."""
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (RUNTIME / path).read_text(encoding="utf-8")


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function: {signature}")


def test_forced_null_allocation_result_fails_closed() -> None:
    capi = read("src/HexagonCAPI.cpp")
    guard = function_body(capi, "void *requireAllocationResult(")
    assert "CHECK(ptr != nullptr" in guard
    assert "return ptr" in guard


def main() -> None:
    capi = read("src/HexagonCAPI.cpp")
    test_forced_null_allocation_result_fails_closed()
    assert "return nullptr;" not in capi

    for signature, producer in (
        ("void *hexagon_runtime_alloc_1d_dsp(", "Alloc("),
        ("void *hexagon_runtime_alloc_2d_dsp(", "Alloc("),
        ("void *hexagon_runtime_workspace_resident_v2_dsp(", "WorkspaceResidentV2("),
        ("void *hexagon_runtime_weight_resident_v2_dsp(", "WeightResidentV2("),
        ("void *hexagon_runtime_build_crouton_dsp(", "CreateBufferAlias("),
        ("void *hexagon_runtime_get_contiguous_memref_dsp(", "GetOrigBufferFromAlias("),
    ):
        body = function_body(capi, signature)
        assert producer in body, signature
        assert "return requireAllocationResult(" in body, signature

    # The scope-missing branches must go through the same guard rather than
    # returning a null pointer to the compiler.
    for marker in (
        "VTCM workspace resident requested before scope entry",
        "VTCM weight resident requested before scope entry",
    ):
        index = capi.index(marker)
        branch = capi[index : capi.index("}", index)]
        assert "requireAllocationResult(nullptr)" in branch, marker

    compiler = read("../../lib/Conversion/HexagonMemToLLVM/HexagonMemToLLVMPass.cpp")
    assert "hexagon_runtime_alloc_1d_dsp" in compiler or "getAllocFnName" in compiler
    assert "non-null result contract" in compiler
    assert "createMemRefDescriptor" in compiler

    print("allocation-result guard source contract: PASS")


if __name__ == "__main__":
    main()
