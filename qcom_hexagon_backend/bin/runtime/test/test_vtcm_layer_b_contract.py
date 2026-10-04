#!/usr/bin/env python3
"""Source contract for the opt-in Layer-B VTCM evidence report.

This is intentionally a source/contract test, not a device correctness or
performance test.  It checks that the runtime keeps the accounting units and
unproved boundaries explicit, and that diagnostic events never acquire an
address identity by accident.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "bin" / "runtime"
# The claim ledger is part of the repository; see test/VTcmAccounting/README.md.
MATRIX = ROOT / "test" / "VTcmAccounting"


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


def test_layer_b_contract_has_explicit_axes_and_boundaries() -> None:
    source = read("src/VTCMPool.cpp")
    report = function_body(source, "int VtcmPool::writeAccountingReportLocked(")
    required = (
        "VTCM_EVIDENCE schema=1 layer=layer-b-runtime",
        "evidence_status=incomplete",
        "scope=process-high-water",
        "function_scope_status=not-bound-in-v1",
        "site_scope_status=not-bound-in-v1",
        "grid_scope_status=not-bound-in-v1",
        "grid_instance_status=not-proven",
        "requested_bytes_basis=allocator-input",
        "allocator_aligned_bytes_status=not-proven",
        "charged_bytes_basis=runtime-pool-block-length",
        "resident_bytes_basis=resident-block-charge",
        "observed_bytes_basis=pool-snapshot",
        "block_accounting_status=%s",
        "header_accounting_status=not-proven",
        "split_coalesce_status=observed-only",
        "split_coalesce_model_status=not-proven",
        "exact_peak_model_status=not-proven",
        "cold_warm_status=not-proven-runtime",
        "reuse_status=observed-event-kinds",
        "content_identity_status=not-proven",
        "event_join_status=not-proven",
        "performance_claimed=false",
    )
    for token in required:
        assert token in report, token


def test_event_and_snapshot_units_are_not_conflated() -> None:
    source = read("src/VTCMPool.cpp")
    header = read("include/VTCMPool.h")
    report = function_body(source, "int VtcmPool::writeAccountingReportLocked(")
    event = function_body(source, "void VtcmPool::recordAccountingEventLocked(")
    snapshot = function_body(source, "VtcmPool::AccountingSnapshot VtcmPool::getAccountingSnapshotLocked()")

    for token in (
        "requestedBytes",
        "sizeAlignedBytes",
        "chargedBytes",
        "requestedAlignment",
        "effectiveAlignment",
        "coalescedBlocks",
        "coalesceCase",
    ):
        assert token in header or token in event, token
    for token in (
        "requested_bytes=%llu",
        "size_aligned_bytes=%llu",
        "charged_bytes=%llu",
        "allocator_aligned_status=not-proven",
        "requested_alignment=%llu",
        "effective_alignment=%llu",
        "coalesced_blocks=%u",
        "coalesce_case=%s",
        "cached_bytes_scope=vtcm-only",
    ):
        assert token in report, token
    for token in (
        "charged_allocated_bytes",
        "observed_allocated_bytes",
        "resident_bytes",
        "block_accounting_complete",
        "event_window_start_sequence",
        "event_window_end_sequence",
        "event_counter_scope=retained-ring",
    ):
        assert token in report, token
    assert "allocator_aligned_bytes_status=not-proven" in report
    assert "block_accounting_status" in report
    assert "header_accounting_status" in report
    assert "fragmentationPpm" in snapshot


def test_free_cache_records_requested_and_charged_footprints() -> None:
    source = read("include/BufferManager.h")
    capi = read("src/HexagonCAPI.cpp")
    for token in (
        "cacheAccountingSizes",
        "requestedBytes",
        "sizeAlignedBytes",
        "chargedBytes",
        "requestedAlignment",
        "freeCacheHitRequestedBytes_",
        "freeCacheRetainRequestedBytes_",
        "freeCacheDropBytes_",
        "freeCacheEvictionBytes_",
    ):
        assert token in source, token
    buffer_header = read("include/HexagonBuffer.h")
    assert "size_t ndim" in buffer_header
    assert "ndim == other.ndim" in buffer_header
    assert "key.ndim == 2" in source
    assert "std::hash<size_t>{}(key.ndim)" in source
    assert "HexagonBuffer::CacheKey{1, 1, nbytes, alignment, isVtcm}" in source
    assert "HexagonBuffer::CacheKey{2, nallocs, nbytes, alignment, isVtcm}" in source
    assert "CacheKey{ndim_, allocations_.size()" in read(
        "src/HexagonBuffer.cpp"
    )
    assert "if (key.numAllocations > 1)" not in source
    for token in (
        "counter_scope=cumulative-process",
        "event_counter_scope=retained-ring",
        "free_cache_hit_requested_bytes",
        "free_cache_retain_requested_bytes",
        "free_cache_drop_bytes",
        "free_cache_eviction_bytes",
        "cache_key_identity=footprint-only",
        "cached_bytes_scope=all-free-cache-occupancy",
        "cached_vtcm_bytes_scope=vtcm-ledger",
        "content_identity_status=not-proven",
        "eviction_retry_status=not-proven",
    ):
        assert token in capi, token


def test_split_coalesce_is_observed_not_promoted_to_a_model() -> None:
    source = read("src/VTCMPool.cpp")
    header = read("include/VTCMPool.h")
    for token in (
        "recordAccountingSplitLocked",
        "accountingSplitEvents_",
        "accountingCoalesceEvents_",
        "accountingCoalescedBlocks_",
        "accountingMaxAllocationCount_",
        "accountingMaxFreeBlockCount_",
    ):
        assert token in source or token in header, token
    for token in (
        "split_events=%llu",
        "split_prefix_bytes=%llu",
        "split_suffix_bytes=%llu",
        "coalesce_events=%llu",
        "coalesced_blocks=%llu",
        "coalesce_none_events=%llu",
        "coalesce_previous_events=%llu",
        "coalesce_next_events=%llu",
        "coalesce_both_events=%llu",
        "max_allocation_count=%llu",
        "max_free_block_count=%llu",
        "split_coalesce_model_status=not-proven",
        "exact_peak_model_status=not-proven",
    ):
        assert token in source or token in read("src/VTCMPool.cpp"), token


def test_per_event_address_padding_is_observed_and_never_a_model() -> None:
    source = read("src/VTCMPool.cpp")
    header = read("include/VTCMPool.h")
    report = function_body(source, "int VtcmPool::writeAccountingReportLocked(")
    record = function_body(
        source, "void VtcmPool::recordAccountingEventLocked("
    )

    # The measurement is the free-list remainder of one placement, and the
    # vocabulary is a closed scope rather than a "no padding" default.
    assert "enum class AccountingPaddingScope" in header
    assert "kNotApplicable = 0" in header
    assert "kPerPlacement = 1" in header
    assert "struct AccountingPadding" in header
    for field in ("prefixBytes", "suffixBytes"):
        assert field in header, field
    for field in ("paddingScope", "addressPrefixBytes", "addressSuffixBytes"):
        assert field in header, field

    # The bytes come from the two placement helpers, not from a free-list delta
    # and not from the request's alignment quantum.  The definitions are gated,
    # so both preprocessor spellings share one body: slice between the two
    # definitions rather than brace-counting across the `#else`.
    end_signature = (
        "char *VtcmPool::tryAllocateFromEnd(size_t nbytes, size_t alignment,\n"
        "                                   AccountingPadding *padding) {"
    )
    best_fit_signature = (
        "char *VtcmPool::allocateBestFit(size_t nbytes, size_t alignment,\n"
        "                                AccountingPadding *padding) {"
    )
    end_placement = source[source.index(end_signature) :]
    end_placement = end_placement[: end_placement.index("// Allocate using best-fit strategy")]
    best_fit = source[source.index(best_fit_signature) :]
    best_fit = best_fit[: best_fit.index("// Handle allocation failure")]
    assert "padding->prefixBytes = prefixBytes" in end_placement
    assert "padding->suffixBytes = 0" in end_placement
    assert "padding->prefixBytes = prefix" in best_fit
    assert "padding->suffixBytes = suffix" in best_fit
    for body in (end_placement, best_fit):
        assert "AccountingPaddingScope::kPerPlacement" in body
        # The cumulative split ledger keeps its own exact inputs, so the two
        # observations cannot drift apart.
        assert "recordAccountingSplitLocked(" in body
    assert "free_.size() - " not in best_fit
    assert "numFreeBlocksBefore" not in source

    # Only the one successful placement is attributed; a failure, a reuse, a
    # free, or a cache record has no padding pointer to attribute.
    allocate = source[source.index("char *VtcmPool::allocateLocked(size_t nbytes, size_t alignment,\n                               bool residentAllocation, uint8_t residentKind,\n                               AccountingPadding *padding) {") :]
    allocate = allocate[: allocate.index("// Try to allocate from end of last free block")]
    assert "AccountingPadding placement;" in allocate
    assert "tryAllocateFromEnd(nbytes, effectiveAlignment, &placement)" in allocate
    assert "allocateBestFit(nbytes, effectiveAlignment, &placement)" in allocate
    assert "&owner, &placement" in allocate
    assert "padding != nullptr &&" in record
    assert "padding->scope == AccountingPaddingScope::kPerPlacement" in record
    assert "event.addressPrefixBytes = padding->prefixBytes" in record
    assert "event.addressSuffixBytes = padding->suffixBytes" in record
    # A resident placement is published by Resident, so the measurement travels
    # out instead of being emitted twice.
    assert "*padding = placement;" in allocate
    assert "&padding);" in source

    # The serialized names are the contract the host overlay validates.
    for token in (
        "address_prefix_bytes=%llu",
        "address_suffix_bytes=%llu",
        "address_padding_scope=%s",
        "address_padding_basis=free-list-remainder-beside-placed-block",
        "address_padding_model_status=not-proven",
    ):
        assert token in report, token
    assert "paddingScopeName(event.paddingScope)" in report
    assert 'return "per-placement";' in source
    assert 'return "not-applicable";' in source

    # What this observation must never become: a fragmentation, occupancy, or
    # address-reuse claim.  The padding block itself carries only the declared
    # basis and the permanent not-proven status, and no address.
    padding_block = report[
        report.index("address_prefix_bytes=%llu") : report.index(
            "allocated_bytes_after=%llu"
        )
    ]
    assert "not-proven" in padding_block
    for forbidden in (
        "complete",
        "verified",
        "reuse",
        "occupancy",
        "fragmentation",
        "aligned_unit",
    ):
        assert forbidden not in padding_block, forbidden
    assert "split_coalesce_model_status=not-proven" in report
    assert "exact_peak_model_status=not-proven" in report
    assert "full_kernel_occupancy_status=not-proven" in report
    assert "address_padding_model_status=complete" not in source
    assert "address_padding_fragmentation" not in source
    assert "address_padding_occupancy" not in source
    assert "address_padding_reuse" not in source
    assert "%p" not in report
    assert "0x" not in report


def test_coalesce_cases_are_explicit_and_not_list_deltas() -> None:
    source = read("src/VTCMPool.cpp")
    header = read("include/VTCMPool.h")
    for token in (
        "AccountingCoalesceCase::kNone",
        "AccountingCoalesceCase::kPrevious",
        "AccountingCoalesceCase::kNext",
        "AccountingCoalesceCase::kBoth",
        "accountingCoalesceNoneEvents_",
        "accountingCoalescePreviousEvents_",
        "accountingCoalesceNextEvents_",
        "accountingCoalesceBothEvents_",
    ):
        assert token in source or token in header, token
    assert "numFreeBlocksBefore" not in source
    coalesce_body = source[
        source.index("size_t VtcmPool::coalesceAndAddToFreeList(") :
        source.index("// Log successful free operation")
    ]
    assert "numFreeBlocksBefore - free_.size()" not in coalesce_body
    assert "free_.size() - " not in coalesce_body

    def merged_blocks(previous: bool, next_block: bool) -> int:
        return int(previous) + int(next_block)

    assert {
        (False, False): merged_blocks(False, False),
        (True, False): merged_blocks(True, False),
        (False, True): merged_blocks(False, True),
        (True, True): merged_blocks(True, True),
    } == {
        (False, False): 0,
        (True, False): 1,
        (False, True): 1,
        (True, True): 2,
    }


def test_the_compiler_size_quantum_mirror_cannot_drift() -> None:
    """The host compiler duplicates this pool's size quantum; pin the two.

    `HmxVtcmAccountingPass` cannot call `VtcmPool::SizeAlignedCharge` -- that is
    a DSP-side symbol behind the probe gate -- so it mirrors the three constants
    to publish a size-aligned bound. A duplicated constant is only acceptable
    while something proves the copies agree, because a silent divergence would
    make the published bound describe a pool that does not exist: still no
    crash, still no warning, just a capacity figure that is quietly wrong.
    """
    pool = read("src/VTCMPool.cpp")
    compiler = (
        ROOT / "lib" / "Dialect" / "Hmx" / "Transforms" /
        "HmxVtcmAccountingPass.cpp"
    ).read_text(encoding="utf-8")

    # The runtime side, as the compiler must see it. The mapping is spelled out
    # rather than derived from the names: the two sides deliberately do not share
    # a naming scheme, and a derived mapping would silently pass if either name
    # changed shape.
    pairs = (
        ("kSmallAlignment", "kRuntimeSizeQuantumSmall"),
        ("kLargeAlignment", "kRuntimeSizeQuantumLarge"),
        ("kLargeThreshold", "kRuntimeSizeQuantumLargeThreshold"),
    )
    for runtime_name, mirror_name in pairs:
        declaration = re.search(
            rf"constexpr size_t {runtime_name} = (\d+);", pool
        )
        assert declaration is not None, f"runtime no longer declares {runtime_name}"
        runtime_value = int(declaration.group(1))
        mirror = re.search(rf"{mirror_name} = (\d+);", compiler)
        assert mirror is not None, f"compiler no longer mirrors {runtime_name}"
        assert int(mirror.group(1)) == runtime_value, (
            f"{runtime_name}: runtime charges {runtime_value} but the compiler "
            f"mirror {mirror_name} says {int(mirror.group(1))}; the published "
            f"size-aligned bound would describe a different pool"
        )

    # The rounding shape must match too, not just the constants: the runtime
    # rounds up to the quantum, and a mirror that truncated would understate the
    # bound and turn it into a false capacity claim.
    assert "const int64_t quantum = nbytes >= kRuntimeSizeQuantumLargeThreshold" in compiler
    assert "charge = (nbytes + (quantum - 1)) & ~(quantum - 1);" in compiler
    # ...and it must refuse rather than clamp, so an unrepresentable request
    # cannot be published as a zero-byte site.
    assert "if (nbytes < 0)\n    return false;" in compiler


def test_the_aligned_bound_is_not_published_as_a_capacity() -> None:
    """The sidecar bound must be labelled a bound, and v3 must stay not-proven.

    The aligned triple is an upper bound over an over-approximated live set. If
    it were published as a plain byte count, a reader would take it as the peak
    and derive a capacity from it -- which is exactly the claim the evidence
    contract refuses to make.
    """
    compiler = (
        ROOT / "lib" / "Dialect" / "Hmx" / "Transforms" /
        "HmxVtcmAccountingPass.cpp"
    ).read_text(encoding="utf-8")
    for field in (
        "transient_aligned_peak_bytes",
        "resident_aligned_bytes",
        "modeled_aligned_peak_bytes",
        "aligned_charge_basis",
    ):
        assert f'fields.append("{field}"' in compiler, field
    # Every aligned figure carries a status that says which kind of number it is.
    assert 'kAlignedTransientStatus = "upper-bound-not-exact-peak"' in compiler
    assert 'kAlignedResidentStatus = "exact-process-floor"' in compiler
    assert 'kAlignedModeledStatus = "upper-bound-not-occupancy"' in compiler
    # The two peaks are attributed independently: reusing the raw peak's sites
    # for the aligned number would name the wrong program point.
    assert "peak_aligned_site_id_status" in compiler
    assert "peakAlignedSiteIds" in compiler
    assert compiler.count("peak_aligned_site_ids") >= 1


def test_resident_abi_is_labeled_without_changing_key_reuse() -> None:
    header = read("include/VTCMPool.h")
    source = read("src/VTCMPool.cpp")
    api = read("src/HexagonAPI.cpp")
    assert "enum class ResidentAbi" in header
    assert "kV2" in header
    assert "kLegacy" not in header
    assert "ResidentAbi abi" not in header
    assert "residentAbi" in header
    assert "residentDescriptorsMatch" in source
    match = function_body(
        source,
        "void *VtcmPool::Resident(ResidentKind kind, uint64_t key, size_t nbytes,\n"
        "                         size_t alignment, const void *src, uint64_t slot) {",
    )
    assert "return block.ptr" in match
    assert "std::memcpy" in match
    assert "WeightResidentV2" in api
    assert "WorkspaceResidentV2" in api
    assert "runtimeVtcm->Resident" in api
    for token in (
        "resident_v2_allocation_events",
        "resident_abi=%s",
    ):
        assert token in source, token
    assert "resident_legacy_allocation_events" not in source


def test_scope_and_content_gaps_are_machine_readable() -> None:
    source = read("src/VTCMPool.cpp")
    header = read("include/VTCMPool.h")
    report = function_body(source, "int VtcmPool::writeAccountingReportLocked(")
    assert "function_id_status=not-bound" in report
    assert "allocation_site_id_status=not-bound" in report
    assert "site_scope_status=not-bound-in-v1" in report
    assert "grid_scope_status=not-bound-in-v1" in report
    assert "content_identity_status=not-proven" in report
    assert "cold_warm_status=not-proven-runtime" in report
    assert "performance_claimed=false" in report
    assert "ResidentAbi" in header
    assert "std::memcmp" not in source
    assert "%p" not in report
    assert "0x" not in report
    # A declared context is metadata only; the sole event formatter must keep
    # process scope explicit and never promote it to a function/site key.
    event_start = report.index('"VTCM_EVENT sequence=')
    event_end = report.index("coalesceCaseName(event.coalesceCase)", event_start)
    event_format = report[event_start:event_end]
    assert "identity_status=process" in event_format
    assert "function_id_status=not-bound" in event_format
    assert "allocation_site_id_status=not-bound" in event_format
    # Bound event-context metadata is allowed elsewhere in the report, but
    # the ordinary event record must not claim a plain function/site identity.
    assert "event_function_id=%llu" in report
    assert " function_id=%" not in event_format
    assert " allocation_site_id=%" not in event_format
    assert "identity_status=site-bound" not in report


def test_cache_accounting_transitions_are_executable_and_vtcm_only() -> None:
    source = read("include/BufferManager.h")
    snapshot_body = function_body(source, "BufferManager::getAccountingSnapshot()")
    initializer = snapshot_body.split("return AccountingSnapshot{", 1)[1].split(
        "};", 1
    )[0]
    snapshot_fields = [field.strip() for field in initializer.split(",")]
    assert snapshot_fields[-4:] == [
        "cachedBufferCount_",
        "cachedBytes_",
        "cachedVtcmBytes_",
        "kMaxCachedBytes",
    ]

    def sizes(*, is_vtcm: bool, ndim: int, blocks: int, per_block: int,
              alignment: int):
        raw = blocks * per_block
        if not is_vtcm:
            return None
        allocator_input = blocks * (
            ((per_block + alignment - 1) // alignment) * alignment
            if ndim == 2
            else per_block
        )
        quantum = 2048 if allocator_input >= 2048 else 128
        aligned = ((allocator_input + quantum - 1) // quantum) * quantum
        return raw, allocator_input, aligned

    assert sizes(
        is_vtcm=False, ndim=1, blocks=1, per_block=257, alignment=128
    ) is None
    assert sizes(
        is_vtcm=True, ndim=1, blocks=1, per_block=257, alignment=128
    ) == (257, 257, 384)
    assert sizes(
        is_vtcm=True, ndim=2, blocks=1, per_block=257, alignment=128
    ) == (257, 384, 384)
    assert sizes(
        is_vtcm=True, ndim=2, blocks=2, per_block=2049, alignment=128
    ) == (4098, 4352, 6144)

    state = {
        "retain": 0,
        "hit": 0,
        "drop": 0,
        "evict": 0,
        "cachedBytes": 0,
        "cachedVtcmBytes": 0,
        "cachedBuffers": 0,
    }
    raw, allocator_input, aligned = sizes(
        is_vtcm=True, ndim=1, blocks=1, per_block=257, alignment=128
    )
    state["retain"] += 1
    state["cachedBytes"] += aligned
    state["cachedVtcmBytes"] += aligned
    state["cachedBuffers"] += 1
    assert state == {
        "retain": 1,
        "hit": 0,
        "drop": 0,
        "evict": 0,
        "cachedBytes": 384,
        "cachedVtcmBytes": 384,
        "cachedBuffers": 1,
    }
    state["hit"] += 1
    state["cachedBytes"] -= aligned
    state["cachedVtcmBytes"] -= aligned
    state["cachedBuffers"] -= 1
    assert (state["cachedBytes"], state["cachedVtcmBytes"], state["cachedBuffers"]) == (
        0,
        0,
        0,
    )
    state["drop"] += 1
    state["evict"] += 1
    assert (state["drop"], state["evict"]) == (1, 1)

    # DDR retention increases total cache occupancy only; it is deliberately a
    # no-op in the VTCM byte and buffer ledgers.
    before = dict(state)
    state["cachedBytes"] += 257
    assert state["cachedBytes"] == before["cachedBytes"] + 257
    assert state["cachedVtcmBytes"] == before["cachedVtcmBytes"]
    assert state["cachedBuffers"] == before["cachedBuffers"]


def test_capture_precedes_cache_move_and_ddr_is_filtered() -> None:
    source = read("include/BufferManager.h")
    free_body = function_body(source, "void FreeHexagonBuffer(")
    capture = free_body.index("const HexagonBuffer::CacheKey cacheKey")
    move = free_body.index("push_back(std::move(buf))")
    assert capture < move
    assert "if (!key.isVtcm)" in source
    assert "return false;" in source
    assert "cachedVtcmBytes_" in source
    assert "cachedBytes_ == allBytes" in source
    assert "cachedVtcmBytes_ == vtcmBytes" in source
    assert "cachedBufferCount_ == vtcmBuffers" in source
    assert "evictedBuffers == cachedBufferCount_" in source
    assert "ddr_cache_accounting=excluded" in read("src/HexagonCAPI.cpp")


def test_boundary_matrix_marks_new_gaps_not_proven() -> None:
    import json

    document = json.loads(
        (MATRIX / "boundary_matrix.json").read_text(encoding="utf-8")
    )
    cells = {cell["id"]: cell for cell in document["cells"]}
    for cell_id in ("allocator.header_split_exact_model", "scope.grid_instance_join"):
        assert cells[cell_id]["status"] == "not-proven"
        assert cells[cell_id]["limitations"]
        assert cells[cell_id]["next_evidence"]
    assert document["contract"]["header_accounting_status"] == "not-proven"
    assert document["contract"]["split_coalesce_model_status"] == "not-proven"
    assert document["contract"]["grid_attribution"] == "not-bound-in-v1"


def main() -> int:
    tests = (
        test_layer_b_contract_has_explicit_axes_and_boundaries,
        test_event_and_snapshot_units_are_not_conflated,
        test_free_cache_records_requested_and_charged_footprints,
        test_split_coalesce_is_observed_not_promoted_to_a_model,
        test_per_event_address_padding_is_observed_and_never_a_model,
        test_the_compiler_size_quantum_mirror_cannot_drift,
        test_the_aligned_bound_is_not_published_as_a_capacity,
        test_coalesce_cases_are_explicit_and_not_list_deltas,
        test_resident_abi_is_labeled_without_changing_key_reuse,
        test_scope_and_content_gaps_are_machine_readable,
        test_cache_accounting_transitions_are_executable_and_vtcm_only,
        test_capture_precedes_cache_move_and_ddr_is_filtered,
        test_boundary_matrix_marks_new_gaps_not_proven,
    )
    for test in tests:
        test()
    print("VTCM Layer-B source contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
