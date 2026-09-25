#!/usr/bin/env python3
"""Source contract pinning the event-context ABI constants across the boundary.

The compiler half lives in one shared header, the runtime half in the VTCM pool
and the C ABI header, and the two are only ever checked against each other
statically. This test exists because nothing at link time compares them: a
silent divergence would let a kernel claim a scope the device never accepted.
"""
from pathlib import Path
import re

RUNTIME = Path(__file__).resolve().parents[1]
BACKEND = RUNTIME.parents[1]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def literal_after(source: str, declaration: str) -> str:
    """Return the string literal a `inline constexpr StringLiteral` carries."""
    match = re.search(
        declaration + r'\s*=\s*"([^"]+)"', source, re.MULTILINE)
    assert match, f"missing declaration: {declaration}"
    return match.group(1)


def value_after(source: str, declaration: str) -> str:
    # Enum members end in a comma, constexpr constants in a semicolon.
    match = re.search(
        declaration + r"\s*=\s*([^;,]+)[;,]", source, re.MULTILINE)
    assert match, f"missing declaration: {declaration}"
    return " ".join(match.group(1).split())


def main() -> None:
    contract = read(
        BACKEND / "include/hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h")
    pass_src = read(BACKEND / "lib/Conversion/HmxToLLVM/HmxToLLVMPass.cpp")
    producer = read(
        BACKEND / "lib/Dialect/Hmx/Transforms/HmxVtcmAccountingPass.cpp")
    pool_h = read(RUNTIME / "include/VTCMPool.h")
    capi_h = read(RUNTIME / "include/HexagonCAPI.h")
    capi = read(RUNTIME / "src/HexagonCAPI.cpp")

    # The shared header is the single compiler-side source of the ABI.
    enter = literal_after(
        contract, r"kHmxDiagnosticEventContextEnterFn")
    leave = literal_after(
        contract, r"kHmxDiagnosticEventContextLeaveFn")
    assert enter in capi, f"runtime does not define the enter symbol: {enter}"
    assert leave in capi, f"runtime does not define the leave symbol: {leave}"
    assert enter == enter.replace("_dsp", "") + "_dsp"
    assert leave == leave.replace("_dsp", "") + "_dsp"
    # The compiler consumes the shared constants; it must not restate them.
    assert "kHmxDiagnosticEventContextEnterFn" in pass_src
    assert "kHmxDiagnosticEventContextLeaveFn" in pass_src
    assert "kHmxDiagnosticEventContextAbiVersion" in pass_src
    assert "kHmxDiagnosticEventContextFlagSingleInvocation" in pass_src
    assert "kHmxDiagnosticEventContextFlagGridOne" in pass_src
    for local in (
        "kDiagnosticEventContextFn",
        "kDiagnosticEventContextLeaveFn",
        "kDiagnosticEventContextVersion",
        "kDiagnosticEventContextSingleInvocation",
        "kDiagnosticEventContextGridOne",
    ):
        assert not re.search(rf"\b{local}\b", pass_src), (
            f"the compiler restates the shared ABI constant: {local}")

    # Version and flag bits must be the same numbers on both sides.
    version = value_after(contract, r"kHmxDiagnosticEventContextAbiVersion")
    assert version == "1", version
    assert value_after(capi_h, r"HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_VERSION") == "1"
    assert value_after(pool_h, r"kAccountingEventContextVersion") == "1"
    single = value_after(
        contract, r"kHmxDiagnosticEventContextFlagSingleInvocation")
    grid = value_after(contract, r"kHmxDiagnosticEventContextFlagGridOne")
    assert single == value_after(
        capi_h, r"HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_SINGLE_INVOCATION")
    assert grid == value_after(
        capi_h, r"HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_GRID_ONE")
    assert single == value_after(pool_h, r"kAccountingEventContextSingleInvocation")
    assert grid == value_after(pool_h, r"kAccountingEventContextGridOne")

    # The token schema has exactly one compiler-side spelling.
    token_schema = literal_after(contract, r"kHmxDiagnosticEventTokenSchema")
    for source, name in ((pass_src, "HmxToLLVMPass.cpp"),
                         (producer, "HmxVtcmAccountingPass.cpp")):
        assert f'"{token_schema}"' not in source, (
            f"{name} restates the shared token schema literal")
    assert f'"{token_schema}"' in contract

    # The closed sidecar key sets live in the same shared header, and the
    # consumer uses them instead of re-listing field names.
    assert "hmxDiagnosticEventContextEligibleKeys" in contract
    assert "hmxDiagnosticEventContextIneligibleKeys" in contract
    assert "hasExactHmxDiagnosticEventContextKeys" in contract
    assert "kHmxDiagnosticEventContextKind" in contract
    assert "hasExactHmxDiagnosticEventContextKeys" in pass_src
    assert "kHmxDiagnosticEventContextKind" in pass_src
    eligible = re.search(
        r"hmxDiagnosticEventContextEligibleKeys\(\)\s*\{(.*?)\}", contract,
        re.S).group(1)
    ineligible = re.search(
        r"hmxDiagnosticEventContextIneligibleKeys\(\)\s*\{(.*?)\}", contract,
        re.S).group(1)
    eligible_keys = set(re.findall(r'"([^"]+)"', eligible))
    ineligible_keys = set(re.findall(r'"([^"]+)"', ineligible))
    assert "scope_id" not in eligible_keys, "the v1 process spelling leaked in"
    assert "promotable" not in eligible_keys | ineligible_keys
    assert eligible_keys - ineligible_keys == {
        "accounting_scope_id", "function_id", "allocation_site_id",
        "token_bits", "token_low", "token_high"}
    assert ineligible_keys - eligible_keys == {"reason"}

    print("VTCM event-context ABI constants source contract: PASS")


if __name__ == "__main__":
    main()
