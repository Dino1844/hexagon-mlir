#!/usr/bin/env python3
"""Host/source contract checks for the opt-in VTCM identity stream.

The community SDK used by this checkout does not ship the device googletest
runtime. These tests therefore verify the ABI and the fail-closed source
contract, not device execution or performance. They deliberately reject any
identity inference from an address or from a name: a future compiler/launcher
must provide the values through the versioned context ABI.
"""

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "bin" / "runtime"
# The claim ledger is part of the repository, not of the probe that checks it: a
# test in the shipped tree must not depend on a directory outside the shipped
# tree, or a fresh clone fails before it runs anything.
MATRIX = ROOT / "test" / "VTcmAccounting"


def _read(relative: str) -> str:
    return (RUNTIME / relative).read_text(encoding="utf-8")


def _matrix_cell(cell_id: str) -> dict:
    document = json.loads(
        (MATRIX / "boundary_matrix.json").read_text(encoding="utf-8")
    )
    return next(cell for cell in document["cells"] if cell["id"] == cell_id)


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


def test_context_abi_is_explicit_versioned_and_resource_free() -> None:
    header = _read("include/HexagonCAPI.h")
    implementation = _read("src/HexagonCAPI.cpp")
    signature = (
        r"int32_t\s+hexagon_runtime_vtcm_accounting_context_enter_v1\s*\(\s*"
        r"uint32_t\s+version,\s*uint32_t\s+flags,\s*"
        r"uint64_t\s+buildIdLow,\s*uint64_t\s+buildIdHigh,\s*"
        r"uint64_t\s+functionId,\s*uint64_t\s+allocationSiteId,\s*"
        r"uint64_t\s+scopeId,\s*uint64_t\s+residentKeyDigestLow,\s*"
        r"uint64_t\s+residentKeyDigestHigh\s*\)"
    )
    assert re.search(signature, header)
    assert re.search(signature.replace("_v1", "_v1_dsp"), implementation)
    assert "#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE" in header
    assert "#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE" in implementation
    assert "HEXAGON_RUNTIME_VTCM_ACCOUNTING_CONTEXT_VERSION = 1" in header
    assert "HEXAGON_RUNTIME_VTCM_ACCOUNTING_CONTEXT_HAS_RESIDENT_KEY_DIGEST" in header

    body = _function_body(
        implementation,
        "int32_t hexagon_runtime_vtcm_accounting_context_enter_v1_dsp(",
    )
    assert "HexagonAPI::Global()" not in body
    assert "VtcmPool::registerAccountingContext(context)" in body
    assert "context.version = version" in body
    assert "context.buildId = {buildIdLow, buildIdHigh}" in body
    assert "context.functionId = functionId" in body
    assert "context.allocationSiteId = allocationSiteId" in body
    assert "context.scopeId = scopeId" in body
    assert "context.residentKeyDigest = {residentKeyDigestLow, residentKeyDigestHigh}" in body
    assert "reinterpret_cast" not in body
    assert "std::hash" not in body


def test_event_context_abi_is_opaque_and_thread_guarded() -> None:
    header = _read("include/HexagonCAPI.h")
    implementation = _read("src/HexagonCAPI.cpp")
    pool_header = _read("include/VTCMPool.h")
    pool_source = _read("src/VTCMPool.cpp")
    signature = (
        r"void\s+hexagon_runtime_vtcm_accounting_event_context_enter_v1\s*\(\s*"
        r"uint32_t\s+version,\s*uint32_t\s+flags,\s*"
        r"uint64_t\s+tokenLow,\s*uint64_t\s+tokenHigh,\s*"
        r"uint64_t\s+accountingScopeId,\s*uint64_t\s+invocationId,\s*"
        r"uint64_t\s+functionId,\s*uint64_t\s+allocationSiteId,\s*"
        r"uint32_t\s+gridProduct\s*\)"
    )
    assert re.search(signature, header)
    assert re.search(signature.replace("_v1", "_v1_dsp"), implementation)
    assert "HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_VERSION = 1" in header
    assert "HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_SINGLE_INVOCATION" in header
    assert "HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_GRID_ONE" in header
    assert "registerAccountingEventContext" in pool_header
    assert "AccountingEventContext" in pool_header
    assert "currentAccountingOwner" in pool_header
    assert "accountingOwners_" in pool_header
    assert "threadOrdinal" in pool_header

    body = _function_body(
        implementation,
        "void hexagon_runtime_vtcm_accounting_event_context_enter_v1_dsp(",
    )
    assert "gridProduct != 1" in body
    assert "context.token = {tokenLow, tokenHigh}" in body
    assert "context.accountingScopeId = accountingScopeId" in body
    assert "registerAccountingEventContext(context)" in body
    assert "hexagon_runtime_vtcm_accounting_event_context_leave_v1" in header
    assert "hexagon_runtime_vtcm_accounting_event_context_leave_v1_dsp" in implementation
    assert "reinterpret_cast" not in body
    assert "std::hash" not in body

    register = _function_body(
        pool_source, "bool VtcmPool::registerAccountingEventContext("
    )
    assert "kAccountingEventContextVersion" in register
    assert "kAccountingEventContextSingleInvocation" in register
    assert "kAccountingEventContextGridOne" in register
    assert "gAccountingContextState.sawEventForEventContext" in register
    assert "gAccountingContextState.eventThreadOrdinal" in register
    assert "poisonAccountingEventContext" in register
    assert "eventRegistered" in register
    assert "threadOrdinal" in pool_source
    assert "owner->threadOrdinal == activeOwner.threadOrdinal" in pool_source
    assert "delayed cache event deliberately passes no owner" in pool_source
    assert "VTCM_EVENT_CONTEXT schema=1" in pool_source
    assert "accounting_scope_id=%llu" in pool_source
    assert "event_accounting_scope_id=%llu" in pool_source
    event_contract = pool_source.split("VTCM_EVENT_CONTEXT", 1)[1].split("\\n", 1)[0]
    assert "scope_id=%llu" not in event_contract.replace("accounting_scope_id", "")
    assert "event_context_status=bound" in pool_source
    assert "event_context_status=%s" in pool_source
    report = _function_body(pool_source, "int VtcmPool::writeAccountingReportLocked(")
    assert "0x" not in report
    assert "%p" not in report


def test_context_scope_is_distinct_from_resident_v2_scope() -> None:
    context_header = _read("include/HexagonCAPI.h")
    resident_header = _read("include/HexagonCAPI.h")
    pool_header = _read("include/VTCMPool.h")
    source = _read("src/HexagonCAPI.cpp")
    pool_source = _read("src/VTCMPool.cpp")

    # v1 deliberately carries one 64-bit process observation ID. Resident-v2
    # carries a separate 128-bit pair; neither width is silently substituted.
    assert re.search(r"uint64_t\s+scopeId\s*,", context_header)
    assert re.search(
        r"uint64_t\s+scopeLow64\s*,\s*uint64_t\s+scopeHigh64", resident_header
    )
    assert "distinct 64-bit process observation ID" in context_header
    assert "128-bit resident-v2 scope pair" in context_header
    assert "resident_scope_binding=not-proven" in context_header
    assert "resident_scope_binding=not-proven" in pool_header
    assert "resident_scope_binding=not-proven" in pool_source
    assert "gResidentScopeState" not in pool_source
    assert "scopeLow64" not in pool_source
    assert "scopeHigh64" not in pool_source

    context_entry = _function_body(
        source, "int32_t hexagon_runtime_vtcm_accounting_context_enter_v1_dsp("
    )
    assert "gResidentScopeState" not in context_entry
    assert "scopeLow64" not in context_entry
    assert "scopeHigh64" not in context_entry

    resident_entry = _function_body(
        source, "int32_t hexagon_runtime_resident_scope_enter_v2_dsp("
    )
    assert "gResidentScopeState" in resident_entry
    assert "scopeLow64" in resident_entry
    assert "scopeHigh64" in resident_entry


def test_registration_is_idempotent_and_changed_context_fails_closed() -> None:
    source = _read("src/VTCMPool.cpp")
    body = _function_body(source, "bool VtcmPool::registerAccountingContext(")
    assert "context.version != kAccountingContextVersion" in body
    assert "context.flags & ~kAccountingContextHasResidentKeyDigest" in body
    assert "gAccountingContextState.registered" in body
    assert "accountingContextsEqual" in body
    assert "gAccountingContextState.sawEvent" in body
    assert body.index("accountingContextsEqual") < body.index("sawEvent")
    assert body.index("if (gAccountingContextState.sawEvent)") < body.index(
        "gAccountingContextState.context = context"
    )
    header = _read("include/VTCMPool.h")
    assert "no mid-process context switch" in header
    assert "no per-event site" in header
    assert "All v1 runtime events are" in header


def test_event_contract_covers_scope_resident_and_free_cache() -> None:
    header = _read("include/VTCMPool.h")
    source = _read("src/VTCMPool.cpp")
    buffer_manager = _read("include/BufferManager.h")

    for field in (
        "buildId",
        "functionId",
        "allocationSiteId",
        "scopeId",
        "residentKeyDigest",
        "eventSequence",
    ):
        assert field in header, field

    for kind in (
        "kAccountingScopeEnter",
        "kAccountingFreeCacheRetain",
        "kAccountingFreeCacheHit",
        "kAccountingFreeCacheDrop",
        "kAccountingFreeCacheEvict",
        "kAccountingResidentAllocation",
        "kAccountingResidentReuse",
        "kAccountingResidentFree",
    ):
        assert kind in source, kind

    for kind in ("kRetain", "kHit", "kDrop", "kEvict"):
        assert f"AccountingCacheEventKind::{kind}" in buffer_manager, kind
    assert "recordAccountingCacheEvent" in buffer_manager
    assert "event.context = contextSnapshot.context" in source
    assert "beginAccountingEvent(kind)" in source
    record_body = _function_body(
        source, "void VtcmPool::recordAccountingEventLocked("
    )
    assert "event.context.functionId = 0" in record_body
    assert "event.context.allocationSiteId = 0" in record_body
    assert "event_identity_scope=process" in source
    assert "identity_status=process" in source
    assert "AccountingEventIdentityScope" not in header
    assert "AccountingEventIdentityScope" not in source
    assert "getAccountingContext" not in header
    assert "getAccountingContext" not in source
    assert "kAccountingScopeEnter" in source
    assert "coalesce_case=%s coalesced_blocks=%u" in source
    alloc_body = _function_body(
        source, "void VtcmPool::recordAccountingAllocationLocked("
    )
    assert "recordAccountingEventLocked(" in alloc_body
    cache_body = _function_body(
        source, "void VtcmPool::recordAccountingCacheEvent("
    )
    assert "recordAccountingEventLocked(" in cache_body
    scope_body = _function_body(source, "void VtcmPool::recordAccountingScopeEvent(")
    assert "recordAccountingEventLocked(" in scope_body
    reuse_body = _function_body(
        source, "void VtcmPool::recordAccountingReuseLocked("
    )
    assert "recordAccountingEventLocked(" in reuse_body
    free_body = _function_body(source, "void VtcmPool::recordAccountingFreeLocked(")
    assert "kAccountingResidentFree" in free_body
    assert "coalesceCase" in free_body
    api_header = _read("include/HexagonAPI.h")
    assert "std::make_unique<BufferManager>(runtimeVtcm.get())" in api_header

    report = _function_body(source, "int VtcmPool::writeAccountingReportLocked(")
    for field in (
        "identity_mode=%s",
        "event_scope_schema=1",
        "VTCM_IDENTITY",
        "build_id_bits=128",
        "build_id_low",
        "build_id_high",
        "function_id",
        "allocation_site_id",
        "scope_id",
        "resident_kind",
        "resident_key_digest_status",
        "event_identity_scope",
        "identity_status",
        "function_id_status",
        "allocation_site_id_status",
        "event_scope_policy=process-only-v1",
        "site_event_status=not-bound-in-v1",
        "full_kernel_join=not-supported",
        "context_identity_scope=declared-only",
        "event_sequence",
    ):
        assert field in report, field
    assert "residentKindName" in report
    assert "residentKeyDigestStatus" in report
    assert "identity_status=process" in report
    assert "identity_status=site-bound" not in report
    assert "resident_key_digest_status=%s" in report
    assert 'return "not-applicable";' in source
    assert 'return "declared-process-only";' in source
    assert "resident_key_digest_status=bound" not in report
    assert "function_id_status=not-bound" in report
    assert "allocation_site_id_status=not-bound" in report
    assert "site_scope_status=not-bound-in-v1" in report
    assert "0x" not in report
    assert "%p" not in report


def test_default_mode_and_truncation_are_explicit() -> None:
    source = _read("src/VTCMPool.cpp")
    implementation = _read("src/HexagonCAPI.cpp")
    header = _read("include/VTCMPool.h")
    record = _function_body(source, "void VtcmPool::recordAccountingEventLocked(")
    report = _function_body(source, "int VtcmPool::writeAccountingReportLocked(")
    public = _function_body(
        implementation, "int hexagon_runtime_vtcm_accounting_report("
    )

    assert "kAccountingEventCapacity = 256" in header
    assert "kAccountingReportTruncated = -2" in header
    assert "if (accountingEventCount_ < kAccountingEventCapacity)" in record
    assert "index = accountingEventStart_;" in record
    assert "(accountingEventStart_ + 1) % kAccountingEventCapacity" in record
    assert "++accountingEventsDropped_;" in record
    assert record.index("++accountingEventSequence_") < record.index(
        "accountingEvents_[index] = event"
    )
    assert "eventsDropped == 0 ? \"complete\" : \"incomplete\"" in report
    assert "event_log_status=%s" in report
    assert "events_dropped=%llu" in report
    assert "identity_mode=process-aggregate" in report
    assert "event_scope_policy=process-only-v1" in report
    assert "identity_mode=%s" in report
    assert report.count("return kAccountingReportTruncated") >= 2
    assert "appendAccountingText" in report
    assert "if (written >= remaining)" in source
    assert "if (offset == VtcmPool::kAccountingReportTruncated)" in public
    assert "if (written >= cap - offset)" in public
    assert "return VtcmPool::kAccountingReportTruncated" in public

    # No registration means the process event formatter still omits function
    # and site join fields rather than fabricating zeroes that look bound.
    assert "if (event.contextRegistered)" in report
    assert "context_registered=%u" in report
    assert "function_id_status=not-bound" in report
    assert "allocation_site_id_status=not-bound" in report
    assert "site_scope_status=not-bound-in-v1" in report
    assert "grid_scope_status=not-bound-in-v1" in report
    assert "VTCM_EVIDENCE" in report


def test_alignment_and_charged_units_are_explicit() -> None:
    source = _read("src/VTCMPool.cpp")
    capi = _read("include/HexagonCAPI.h")
    supported = _function_body(source, "bool VtcmPool::IsSupportedAlignment(")
    charge = _function_body(source, "size_t alignSize(size_t nbytes)")
    report = _function_body(source, "int VtcmPool::writeAccountingReportLocked(")

    assert "constexpr size_t kSmallAlignment = 128;" in source
    assert "constexpr size_t kLargeAlignment = 2048;" in source
    assert "constexpr size_t kMaxAlignment = 2048;" in source
    assert "alignment != 0 && alignment <= kMaxAlignment" in supported
    assert "(alignment & (alignment - 1)) == 0" in supported
    assert "powers of two through 2048" in capi
    assert "powers of two through 2048" in _read("include/VTCMPool.h")
    assert re.search(
        r"void\s+\*\s*hexagon_runtime_workspace_resident_v2\s*\(\s*"
        r"uint64_t\s+key,\s*uint32_t\s+bytes,\s*uint32_t\s+alignment\s*\)",
        capi,
    )
    assert re.search(
        r"void\s+\*\s*hexagon_runtime_weight_resident_v2\s*\(\s*"
        r"uint64_t\s+src,\s*uint32_t\s+bytes,\s*uint32_t\s+alignment\s*\)",
        capi,
    )

    assert "(nbytes >= kLargeThreshold) ? kLargeAlignment : kSmallAlignment" in charge
    assert "max() - (alignment - 1)" in charge
    assert "return 0" in charge
    assert "unit=charged-bytes" in report
    assert "allocator_aligned_unit_status=not-proven" in report
    assert "requested_bytes_basis=allocator-input" in report
    assert "aligned_bytes_basis=size-rounding" in report
    assert "address_alignment_status=not-proven" in report

    def charged(requested: int) -> int:
        quantum = 2048 if requested >= 2048 else 128
        return (requested + quantum - 1) // quantum * quantum

    assert {
        requested: charged(requested)
        for requested in (1, 127, 128, 129, 255, 256, 257, 2047, 2048, 2049)
    } == {
        1: 128,
        127: 128,
        128: 128,
        129: 256,
        255: 256,
        256: 256,
        257: 384,
        2047: 2048,
        2048: 2048,
        2049: 4096,
    }
    assert _matrix_cell("units.charged_bytes")["status"] == "pass"
    assert _matrix_cell("alignment.device_addresses")["status"] == "not-proven"


def test_resident_content_and_address_reuse_remain_not_proven() -> None:
    header = _read("include/VTCMPool.h")
    source = _read("src/VTCMPool.cpp")
    implementation = _read("src/HexagonCAPI.cpp")
    resident = _function_body(
        source,
        "void *VtcmPool::Resident(ResidentKind kind, uint64_t key, size_t nbytes,\n                         size_t alignment, const void *src) {",
    )
    descriptor = header[
        header.index("struct ResidentDescriptor") : header.index("struct ResidentBlock")
    ]

    for field in ("kind", "key", "bytes", "alignment", "chargedBytes"):
        assert field in descriptor
    assert "content" not in descriptor.lower()
    assert "std::memcmp" not in source
    assert "content hash" not in source.lower()
    assert resident.index("return block.ptr") < resident.index("std::memcpy")
    assert "Source-address reuse is valid only for the documented immutable" in implementation
    assert "runtime performs no content hash" in implementation

    cell = _matrix_cell("resident.content_address_reuse")
    assert cell["status"] == "not-proven"
    assert cell["claim_scope"] == "identity-join"
    assert cell["limitations"]
    assert cell["next_evidence"]


def test_multi_scope_conflicts_and_process_boundaries_fail_closed() -> None:
    pool = _read("src/VTCMPool.cpp")
    capi = _read("src/HexagonCAPI.cpp")
    context = _function_body(pool, "bool VtcmPool::registerAccountingContext(")
    scope = _function_body(
        capi, "int32_t hexagon_runtime_resident_scope_enter_v2_dsp("
    )

    assert "if (!accountingContextsEqual(gAccountingContextState.context, context))" in context
    assert "if (gAccountingContextState.sawEvent)" in context
    assert "gResidentScopeState" not in context
    assert "gResidentScopeState.low != scopeLow64" in scope
    assert "gResidentScopeState.high != scopeHigh64" in scope
    assert scope.index("gResidentScopeState.low = scopeLow64") > scope.index(
        "return -1"
    )
    assert "gAccountingContextState" not in scope
    assert "resident_scope_binding=not-proven" in pool

    assert _matrix_cell("scope.changed_context_and_resident_scope_conflicts")[
        "status"
    ] == "pass"
    for cell_id in (
        "scope.multi_function_same_process",
        "scope.multi_module_same_process",
        "scope.multi_process_isolation",
        "scope.resident_accounting_binding",
    ):
        assert _matrix_cell(cell_id)["status"] == "not-proven"


def test_pool_blocks_have_no_in_pool_header() -> None:
    """Positive source fact, recorded on its own axis.

    A VTCM pool block is a raw slice of one flat pool: the charged length is
    the size-rounded payload, the release size is an out-of-band argument that
    is verified against host-RAM bookkeeping before the block is returned, and
    that bookkeeping lives in ordinary host containers.  This is deliberately a
    *different* claim from ``header_accounting_status=not-proven``, which stays
    unproven because a charge still does not predict a general allocator peak.
    """
    header = _read("include/VTCMPool.h")
    pool = _read("src/VTCMPool.cpp")
    capi = _read("src/HexagonCAPI.cpp")
    manager = _read("include/BufferManager.h")

    # The charge is the size-rounded payload, with nothing reserved in front.
    align = _function_body(pool, "size_t alignSize(size_t nbytes)")
    assert "eader" not in align
    assert "nbytes = alignSize(nbytes);" in pool
    assert "const size_t chargedBytes = nbytes == 0 ? 0 : alignSize(nbytes);" in pool

    # Every release entry point takes the size as an out-of-band argument.
    assert "void Free(void *ptr, size_t nbytes);" in header
    assert "bool FreeResident(void *ptr, size_t nbytes);" in header
    assert "bool freeResidentLocked(char *ptr, size_t nbytes);" in header
    public_free = _function_body(
        capi, "void hexagon_runtime_free_1d_dsp(void *ptr) {"
    )
    assert "nbytes" not in public_free
    assert "HexagonAPI::Global()->Free(ptr)" in public_free

    # ... and it is verified against host-side bookkeeping before release.
    ordinary = _function_body(pool, "void VtcmPool::Free(void *ptr, size_t nbytes)")
    assert ordinary.index("nbytes = alignSize(nbytes);") < ordinary.index(
        "CHECK((it != allocations_.end())"
    )
    assert ordinary.index("CHECK((it != allocations_.end())") < ordinary.index(
        "CHECK((it->second == nbytes)"
    )
    resident = _function_body(pool, "bool VtcmPool::freeResidentLocked(")
    assert resident.index("nbytes != block.descriptor.bytes") < resident.index(
        "VTCM resident free descriptor mismatch"
    )
    assert "chargedBytes != block.descriptor.chargedBytes" in resident
    assert resident.index("VTCM resident free descriptor mismatch") < resident.index(
        "return true"
    )

    # The bookkeeping itself is host RAM, and the manager recovers the size
    # from it rather than from the block.
    for container in (
        "std::vector<std::pair<char *, size_t>> allocations_;",
        "std::vector<std::pair<char *, size_t>> free_;",
        "std::vector<ResidentBlock> resident_;",
    ):
        assert container in header
    release = _function_body(manager, "void FreeHexagonBuffer(void *ptr)")
    assert "bufferMap_.find(ptr)" in release
    assert "buf->GetAllocatedBytes()" in release

    # The only write the pool makes into a block is the caller's payload, at
    # offset zero: no leading header bytes are reserved or written.
    assert pool.count("std::memcpy") == 1
    assert "std::memset" not in pool
    fill = _function_body(
        pool,
        "void *VtcmPool::Resident(ResidentKind kind, uint64_t key, size_t nbytes,\n"
        "                         size_t alignment, const void *src) {",
    )
    assert "std::memcpy(ptr, src, nbytes);" in fill

    # The matrix records the fact on its own field and its own cell; the
    # header-inclusive allocator model keeps its own unproven status.
    document = json.loads(
        (MATRIX / "boundary_matrix.json").read_text(encoding="utf-8")
    )
    contract = document["contract"]
    assert (
        contract["in_pool_header_absence"]
        == "source-verified-absent-out-of-band-size-host-metadata"
    )
    assert contract["header_accounting_status"] == "not-proven"
    cell = next(
        cell
        for cell in document["cells"]
        if cell["id"] == "allocator.no_in_pool_header_source_contract"
    )
    assert cell["status"] == "pass"
    assert cell["claim_scope"] == "source-contract"
    assert cell["limitations"] == []
    assert cell["next_evidence"] == []
    assert _matrix_cell("allocator.header_split_exact_model")["status"] == "not-proven"


def test_boundary_matrix_declares_its_own_limits() -> None:
    """The ledger's declared contract, read directly from the ledger.

    This used to shell out to the probe's `matrix_runner.py` and assert on the
    runner's output. Two things were wrong with that. It made the repository's
    test suite depend on a file outside the repository, so a fresh clone failed
    before running anything; and it made a *declaration* verifiable only by
    running the *checker*, when the declaration is a file in this repository and
    can simply be read.

    What is left here is the declaration: the contract block, and the fact that
    every cell states its scope and carries no unearned evidence. The runner's
    own determinism and output shape are properties of the runner, so they are
    tested beside the runner.
    """

    document = json.loads(
        (MATRIX / "boundary_matrix.json").read_text(encoding="utf-8")
    )
    contract = document["contract"]
    assert contract["event_scope_policy"] == "process-only-v1"
    assert contract["function_attribution"] == "declared-only"
    assert contract["allocation_site_attribution"] == "not-bound-in-v1"
    assert contract["resident_accounting_scope_binding"] == "not-proven"
    assert contract["address_alignment_in_report"] == "not-proven"
    assert contract["header_accounting_status"] == "not-proven"
    assert (
        contract["in_pool_header_absence"]
        == "source-verified-absent-out-of-band-size-host-metadata"
    )
    assert contract["split_coalesce_model_status"] == "not-proven"
    assert contract["grid_attribution"] == "not-bound-in-v1"
    assert contract["backend_option"] is False
    assert contract["v3_producer"] is False
    assert contract["manifest_mutation"] is False
    assert contract["launcher_mutation"] is False
    assert contract["probe_default"] == "off"
    # A ledger that claimed completeness would be the failure mode, so the
    # unknown has to be visible in the file rather than in a reader's defaults.
    assert contract["status_vocabulary"] == ["pass", "fail", "not-proven"]
    assert {cell["status"] for cell in document["cells"]} <= {
        "pass",
        "fail",
        "not-proven",
    }
    assert any(cell["status"] == "not-proven" for cell in document["cells"])
    for cell in document["cells"]:
        assert cell["claim_scope"], cell["id"]
        assert "limitations" in cell and "next_evidence" in cell, cell["id"]
        # A cell that says `pass` with an empty limitations list and no next
        # evidence is a cell that has to mean it; the two narrower scopes are
        # where a claim is allowed to be complete.
        if cell["status"] == "pass" and cell["claim_scope"] != "source-contract":
            assert cell["limitations"] or cell["next_evidence"], cell["id"]


def test_per_event_address_padding_does_not_bump_the_event_abi() -> None:
    header = _read("include/VTCMPool.h")
    capi_header = _read("include/HexagonCAPI.h")
    source = _read("src/VTCMPool.cpp")

    # The per-event address padding is a report-level observation, so the
    # event-context C ABI is untouched: no new entry parameter, and therefore
    # no reason to reject a v1 producer that a v2 spelling would exclude.  The
    # version stays 1 on all three sides rather than being bumped for a field
    # the producer never sends.
    assert "kAccountingEventContextVersion = 1" in header
    assert "HEXAGON_RUNTIME_VTCM_EVENT_CONTEXT_VERSION = 1" in capi_header
    assert "kAccountingEventContextVersion" in source
    # The only event-context symbols are still the v1 pair: an "_v2" spelling
    # would mean a caller could pass something this build ignores.
    assert "hexagon_runtime_vtcm_accounting_event_context_enter_v2" not in capi_header
    assert "hexagon_runtime_vtcm_accounting_event_context_leave_v2" not in capi_header

    # The padding travels through the pool's own report, not the identity ABI.
    report = _function_body(
        source, "int VtcmPool::writeAccountingReportLocked("
    )
    for token in (
        "address_prefix_bytes=%llu",
        "address_suffix_bytes=%llu",
        "address_padding_scope=%s",
    ):
        assert token in report, token
    assert "AccountingPadding" in header
    assert "AccountingPadding" in source
    # A padding measurement is a placement fact, never an identity fact: the
    # event keeps its process-scoped v1 semantics and cannot be joined to a
    # function, site, or invocation by this field.
    assert "event_identity_scope=process" in report
    assert "function_id_status=not-bound" in report
    assert "allocation_site_id_status=not-bound" in report
    assert "event_scope_policy=process-only-v1" in report


def test_probe_is_opt_in_and_v2_abi_is_unchanged() -> None:
    cmake = (RUNTIME / "CMakeLists.txt").read_text(encoding="utf-8")
    capi = _read("include/HexagonCAPI.h")
    assert 'option(HEXMLIR_VTCM_ACCOUNTING_PROBE "Record VTCM allocation accounting diagnostics" OFF)' in cmake

    assert "ResidentAbi::kLegacy" not in _read("src/VTCMPool.cpp")
    assert "kLegacy" not in _read("include/VTCMPool.h")
    assert "hexagon_runtime_workspace_resident(" not in capi
    assert "hexagon_runtime_weight_resident(" not in capi
    assert "hexagon_runtime_workspace_resident_v2" in capi
    assert "hexagon_runtime_weight_resident_v2" in capi
    assert "hexagon_runtime_resident_free_v2" in capi
    assert "hexagon_runtime_vtcm_accounting_context_enter_v1" in capi
    assert "hexagon_runtime_vtcm_accounting_event_context_enter_v1" in capi
    assert "hexagon_runtime_vtcm_accounting_event_context_enter_v2" not in capi


if __name__ == "__main__":
    test_context_abi_is_explicit_versioned_and_resource_free()
    test_event_context_abi_is_opaque_and_thread_guarded()
    test_context_scope_is_distinct_from_resident_v2_scope()
    test_registration_is_idempotent_and_changed_context_fails_closed()
    test_event_contract_covers_scope_resident_and_free_cache()
    test_default_mode_and_truncation_are_explicit()
    test_alignment_and_charged_units_are_explicit()
    test_resident_content_and_address_reuse_remain_not_proven()
    test_pool_blocks_have_no_in_pool_header()
    test_multi_scope_conflicts_and_process_boundaries_fail_closed()
    test_boundary_matrix_declares_its_own_limits()
    test_per_event_address_padding_does_not_bump_the_event_abi()
    test_probe_is_opt_in_and_v2_abi_is_unchanged()
    print("VTCM accounting identity source contract: PASS")
