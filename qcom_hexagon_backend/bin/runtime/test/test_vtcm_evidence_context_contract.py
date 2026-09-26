#!/usr/bin/env python3
"""Source/host contract checks for the integrated R-A evidence context.

These checks intentionally do not claim a device join.  They lock the
marker/version boundary, the smallest-scope refusal, resident prerequisites,
delayed-cache aggregate marking, and the separate pool/BufferManager high-water
ledgers.  They are positive and fail-closed source checks, not performance or
production-admission evidence.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "bin" / "runtime"


def read(relative: str) -> str:
    return (RUNTIME / relative).read_text(encoding="utf-8")


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


def test_marker_and_schema_are_explicit_and_non_wire() -> None:
    header = read("include/HexagonCAPI.h")
    accounting_header = (ROOT / "include/hexagon/Dialect/Hmx/Transforms/HmxVtcmAccounting.h").read_text(encoding="utf-8")
    pass_source = (ROOT / "lib/Dialect/Hmx/Transforms/HmxVtcmAccountingPass.cpp").read_text(encoding="utf-8")
    assert 'option(HEXMLIR_VTCM_ACCOUNTING_PROBE "Record VTCM allocation accounting diagnostics" OFF)' in read("CMakeLists.txt")
    assert "kHmxDiagnosticVtcmEvidenceContextAttr" in accounting_header
    assert 'hmx.diagnostic_vtcm_evidence_context' in accounting_header
    assert 'kHmxVtcmEvidenceContextAttr' in accounting_header
    assert 'hmx.kernel_vtcm_evidence_context' in accounting_header
    assert 'hmx.vtcm-evidence-context/v1' in accounting_header
    assert "evidenceContextMarker && !identityMarker" in pass_source
    assert "buildEvidenceContextAttr" in pass_source
    assert "kHmxVtcmEvidenceContextAttr" in pass_source
    assert "kEvidenceContextScope" in pass_source
    assert "event_owner_status" in pass_source
    assert "delayed_cache_owner_status" in pass_source
    assert "workspace_overwrite_proof_status" in pass_source
    assert "global_symbol_identity_status" in pass_source
    assert "packed_weight_source_view_status" in pass_source
    assert "packed_weight_digest_status" in pass_source
    assert "resident_scope_binding" in pass_source
    assert "pool_cache_combined_status" in pass_source
    assert "full_kernel_occupancy_status" in pass_source
    assert "hexagon_runtime_vtcm_accounting_event_context_enter_v1" in header
    assert "hexagon_runtime_vtcm_accounting_event_context_enter_v2" not in header


def test_runtime_context_line_refuses_unsound_joins() -> None:
    source = read("src/VTCMPool.cpp")
    report = function_body(source, "int VtcmPool::writeAccountingReportLocked(")
    for token in (
        "VTCM_EVIDENCE_CONTEXT schema=1",
        "observation_scope=one-immutable-principal-module-function-canonical-site-grid1-single-invocation",
        "observation_scope_status=not-proven",
        "frame_status=not-proven",
        "event_owner_status=aggregate",
        "delayed_cache_owner_status=aggregate",
        "function_id_status=not-bound-in-v1",
        "allocation_site_id_status=not-bound-in-v1",
        "resident_scope_binding=not-proven",
        "object_descriptor_status=not-proven",
        "content_descriptor_status=not-proven",
        "workspace_overwrite_proof_status=not-proven",
        "global_symbol_identity_status=not-proven",
        "packed_weight_source_view_status=not-proven",
        "packed_weight_digest_status=not-proven",
        "pool_cache_combined_status=not-proven",
        "full_kernel_occupancy_status=not-proven",
        "pool_high_water_scope=charged-vtcm-pool-only",
        "performance_claimed=false",
    ):
        assert token in report, token
    context_start = report.index('"VTCM_EVIDENCE_CONTEXT')
    context_end = report.index('"VTCM_IDENTITY', context_start)
    context_format = report[context_start:context_end]
    assert "0x" not in context_format
    assert "%p" not in context_format
    assert "identity_status=site-bound" not in context_format


def test_pool_and_cache_high_waters_are_separate_ledgers() -> None:
    header = read("include/BufferManager.h")
    capi = read("src/HexagonCAPI.cpp")
    for token in (
        "highWaterCachedVtcmBytes",
        "highWaterCachedBuffers",
        "highWaterCachedVtcmBytes_",
        "highWaterCachedBuffers_",
    ):
        assert token in header, token
    for token in (
        "cached_vtcm_high_water_bytes=%llu",
        "pool_high_water_charged_bytes=%llu",
        "combined_occupancy_lower_bound_bytes=%llu",
        "combined_occupancy_upper_bound_bytes=%llu",
        "combined_occupancy_status=not-proven",
        "combined_occupancy_basis=pool-charge-plus-buffer-manager-cache-no-double-count",
        "full_kernel_occupancy_status=not-proven",
    ):
        assert token in capi, token
    assert "pool.highWaterAllocatedBytes" in capi
    assert "cache.highWaterCachedVtcmBytes" in capi
    assert "full-kernel occupancy" in capi


# The probe contract check (wrapper / driver / evidence-context builder) used to
# live here. Every assertion in it was about experiment scaffolding under
# `exp/hmx/vtcm_accounting_probe/`, so keeping it here made the repository's test
# suite depend on a directory outside the repository -- a fresh clone failed before
# running anything, and the check would bit-rot whenever the probe changed shape.
# It now lives beside what it checks, in
# `exp/hmx/vtcm_accounting_probe/test_probe_contract.py`.


if __name__ == "__main__":
    tests = (
        test_marker_and_schema_are_explicit_and_non_wire,
        test_runtime_context_line_refuses_unsound_joins,
        test_pool_and_cache_high_waters_are_separate_ledgers,
    )
    for test in tests:
        test()
    print("VTCM evidence context source contract: PASS")
