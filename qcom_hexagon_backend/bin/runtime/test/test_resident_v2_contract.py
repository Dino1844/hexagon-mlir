#!/usr/bin/env python3
"""Source-level contract checks for the runtime resident-v2 ABI.

The device gtest suite is unavailable in the community SDK. These checks are
deliberately narrow: they verify the exported C signatures and the fail-closed
control flow that prevents a same-key descriptor mismatch from reaching the
allocator. They are not device correctness or performance evidence.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "bin" / "runtime"


def _read(relative: str) -> str:
    return (RUNTIME / relative).read_text(encoding="utf-8")


def _function_body(source: str, signature: str) -> str:
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


def test_v2_c_abi_signatures() -> None:
    header = _read("include/HexagonCAPI.h")
    implementation = _read("src/HexagonCAPI.cpp")

    resident_patterns = (
        r"void\s+\*\s*hexagon_runtime_workspace_resident_v2\s*\(\s*"
        r"uint64_t\s+key,\s*uint32_t\s+bytes,\s*uint32_t\s+alignment\s*\)",
        r"void\s+\*\s*hexagon_runtime_weight_resident_v2\s*\(\s*"
        r"uint64_t\s+src,\s*uint32_t\s+bytes,\s*uint32_t\s+alignment\s*\)",
    )
    for pattern in resident_patterns:
        assert re.search(pattern, header), f"missing public declaration: {pattern}"
        device_pattern = pattern.replace("_v2", "_v2_dsp")
        assert re.search(device_pattern, implementation), (
            f"missing device definition: {pattern}"
        )

    free_pattern = (
        r"int32_t\s+hexagon_runtime_resident_free_v2\s*\(\s*"
        r"void\s+\*ptr,\s*uint32_t\s+bytes\s*\)"
    )
    assert re.search(free_pattern, header), "missing public resident-free declaration"
    device_free_pattern = free_pattern.replace("_v2", "_v2_dsp")
    assert re.search(device_free_pattern, implementation), (
        "missing device resident-free definition"
    )

    scope_pattern = (
        r"int32_t\s+hexagon_runtime_resident_scope_enter_v2\s*\(\s*"
        r"uint64_t\s+scopeLow64,\s*uint64_t\s+scopeHigh64\s*\)"
    )
    assert re.search(scope_pattern, header)
    device_scope_pattern = scope_pattern.replace("_v2", "_v2_dsp")
    assert re.search(device_scope_pattern, implementation)

    # The unversioned resident ABI is deliberately absent. Keeping these
    # checks on declarations and definitions prevents a second, scope-free
    # resident entry point from reappearing.
    for legacy_symbol in (
        "hexagon_runtime_weight_resident",
        "hexagon_runtime_workspace_resident",
    ):
        assert not re.search(rf"\b{legacy_symbol}\s*\(", header)
        assert not re.search(rf"\b{legacy_symbol}_dsp\s*\(", implementation)
    assert "ResidentAbi abi" not in header
    assert "WeightResident(" not in header
    assert "WorkspaceResident(" not in header
    assert "ResidentAbi::kLegacy" not in implementation
    assert "kLegacy" not in header


def test_same_key_descriptor_mismatch_fails_before_allocation() -> None:
    header = _read("include/VTCMPool.h")
    source = _read("src/VTCMPool.cpp")
    body = _function_body(
        source,
        "void *VtcmPool::Resident(ResidentKind kind, uint64_t key, size_t nbytes,\n                         size_t alignment, const void *src) {",
    )

    for field in ("kind", "key", "bytes", "alignment", "chargedBytes"):
        assert re.search(rf"\b{field}\b", header), f"descriptor lacks {field}"
    for field in ("kind", "key", "bytes", "alignment", "chargedBytes"):
        assert f"lhs.{field}" in source and f"rhs.{field}" in source, (
            f"descriptor comparison omits {field}"
        )

    mismatch = body.index("!residentDescriptorsMatch")
    mismatch_return = body.index("return nullptr", mismatch)
    allocations = [
        match.start() for match in re.finditer(r"allocateLocked\(", body)
    ]
    assert allocations, "resident allocation path disappeared"
    assert all(mismatch_return < allocation for allocation in allocations), (
        "a same-key mismatch reaches the allocator"
    )


def test_resident_alignment_uses_the_vtcm_verifier() -> None:
    header = _read("include/VTCMPool.h")
    source = _read("src/VTCMPool.cpp")
    assert "powers of two through 2048" in header
    supported = _function_body(source, "bool VtcmPool::IsSupportedAlignment(")
    assert "alignment != 0 && alignment <= kMaxAlignment" in supported
    assert "(alignment & (alignment - 1)) == 0" in supported
    resident = _function_body(
        source,
        "void *VtcmPool::Resident(ResidentKind kind, uint64_t key, size_t nbytes,\n"
        "                         size_t alignment, const void *src) {",
    )
    assert "!IsSupportedAlignment(alignment)" in resident
    assert resident.index("!IsSupportedAlignment(alignment)") < resident.index(
        "allocateLocked("
    )


def test_v2_calls_are_scope_gated_before_runtime_construction() -> None:
    source = _read("src/HexagonCAPI.cpp")
    scope_body = _function_body(
        source, "int32_t hexagon_runtime_resident_scope_enter_v2_dsp("
    )
    assert "HexagonAPI::Global()" not in scope_body
    assert "gResidentScopeState.registered" in scope_body
    assert "return -1" in scope_body

    for signature in (
        "void *hexagon_runtime_workspace_resident_v2_dsp(",
        "void *hexagon_runtime_weight_resident_v2_dsp(",
    ):
        body = _function_body(source, signature)
        gate = body.index("residentScopeRegistered()")
        construction = body.index("HexagonAPI::Global()")
        assert gate < construction, f"{signature} constructs before scope gate"
        assert "ResidentV2(" in body
        legacy = (
            "WorkspaceResident("
            if "workspace" in signature
            else "WeightResident("
        )
        assert legacy not in body, f"{signature} falls back to the legacy ABI"

    free_body = _function_body(
        source, "int32_t hexagon_runtime_resident_free_v2_dsp("
    )
    assert free_body.index("residentScopeRegistered()") < free_body.index(
        "HexagonAPI::Global()"
    )
    assert "FreeResident(" in free_body


if __name__ == "__main__":
    test_v2_c_abi_signatures()
    test_same_key_descriptor_mismatch_fails_before_allocation()
    test_resident_alignment_uses_the_vtcm_verifier()
    test_v2_calls_are_scope_gated_before_runtime_construction()
    print("resident v2 source contract: PASS")
