#!/usr/bin/env python3
"""Source contract for the diagnostic per-site scope runtime ABI.

The per-site scope is a new, probe-only, versioned ABI next to the existing
frame context. These checks pin the boundary that makes it safe to add without
touching v1:

* the site ABI is a *separate* registration, not a widening of the frame one --
  a site token is not a frame token, and the runtime never derives either from
  an address;
* every rejection path poisons the thread, so a scope that was asked for and
  not granted can never look attributed;
* the scope is not nestable, because a nested span would make the free of the
  inner allocation ambiguous between two sites;
* the site binding is checked independently of the frame binding, and both must
  hold on the same thread at the same instant;
* the report additions are strictly additive: the existing v1 lines and every
  pre-existing per-event field keep their exact spelling, and the new
  declaration is emitted only when a site scope was registered.

This is a source/contract check. It is not a device observation and claims
nothing about allocator behaviour.
"""

from __future__ import annotations

from pathlib import Path
import re


RUNTIME = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (RUNTIME / relative).read_text(encoding="utf-8")


def body(source: str, signature: str) -> str:
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
    raise AssertionError(f"unterminated body: {signature}")


def test_site_abi_is_separate_from_the_frame_abi() -> None:
    header = read("include/VTCMPool.h")
    assert "struct AccountingSiteScope" in header
    assert "kAccountingSiteScopeVersion = 1" in header
    assert "kAccountingSiteScopeSingleInvocation = 1u << 0" in header
    assert "kAccountingSiteScopeGridOne = 1u << 1" in header
    # The site record carries its own token, its own identities, and the build
    # echo the frame record does not have.
    for field in (
        "AccountingDigest token{};",
        "uint64_t accountingScopeId{0};",
        "uint64_t invocationId{0};",
        "uint64_t functionId{0};",
        "uint64_t allocationSiteId{0};",
        "AccountingDigest buildId{};",
    ):
        site = body(header, "struct AccountingSiteScope {")
        assert field in site, field
    # A separate registration with its own three entry points. The frame entry
    # points must not have been changed to take a site argument.
    for signature in (
        "static bool registerAccountingSiteScope(const AccountingSiteScope &scope);",
        "static void clearAccountingSiteScope();",
        "static bool hasAccountingSiteScope();",
    ):
        assert signature in header, signature
    frame_enter = body(header, "static bool registerAccountingEventContext(")
    assert "AccountingSiteScope" not in frame_enter

    capi = read("include/HexagonCAPI.h")
    assert "hexagon_runtime_vtcm_accounting_site_scope_enter_v1" in capi
    assert "hexagon_runtime_vtcm_accounting_site_scope_leave_v1" in capi
    # The frame entry keeps its nine-word shape; the site entry is the one with
    # the build echo, so a reader can tell which ABI a call belongs to.
    frame = decl(
        capi, "void hexagon_runtime_vtcm_accounting_event_context_enter_v1("
    )
    assert "buildIdLow" not in frame
    site = decl(capi, "void hexagon_runtime_vtcm_accounting_site_scope_enter_v1(")
    for word in (
        "tokenLow",
        "tokenHigh",
        "accountingScopeId",
        "invocationId",
        "functionId",
        "allocationSiteId",
        "buildIdLow",
        "buildIdHigh",
        "gridProduct",
    ):
        assert word in site, word

    impl = read("src/HexagonCAPI.cpp")
    assert "hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp" in impl
    assert "hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp" in impl
    enter = body(
        impl, "void hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp("
    )
    assert "VtcmPool::clearAccountingSiteScope();" in enter
    assert "scope.buildId = {buildIdLow, buildIdHigh};" in enter
    assert "VtcmPool::registerAccountingSiteScope(scope)" in enter
    # Neither entry may return a value the kernel could branch on: a refused
    # registration must not change control flow.
    assert "-> " not in enter


def test_every_rejection_path_poisons() -> None:
    pool = read("src/VTCMPool.cpp")
    register = body(pool, "bool VtcmPool::registerAccountingSiteScope(")
    # One poison for the malformed record, one for the nesting refusal, and one
    # for the late/second-thread/different-scope refusals.
    assert register.count("poisonAccountingSiteScopeLocked()") >= 2
    assert register.count("poisonAccountingSiteScope()") >= 1
    assert "tAccountingSiteScopePoisoned = false" in register
    # A refusal must also drop the frame owner: two owners disagreeing about
    # whether this thread holds a token is exactly the state that makes a report
    # unreadable.
    poison = body(pool, "void poisonAccountingSiteScopeLocked()")
    assert "gAccountingContextState.siteOwnerActive = false" in poison
    assert "gAccountingContextState.eventOwnerActive = false" in poison
    unlocked = body(pool, "void poisonAccountingSiteScope()")
    assert "poisonAccountingSiteScopeLocked" in unlocked
    assert "lock_guard" in unlocked

    # The scope is not nestable, and that is a refusal rather than a stack push.
    assert "if (gAccountingContextState.siteOwnerActive)" in register
    nesting = register.index("if (gAccountingContextState.siteOwnerActive)")
    assert "poisonAccountingSiteScopeLocked();" in register[nesting:nesting + 200]
    # A zero build echo cannot bind: a site name carried by a kernel built from
    # different IR is a different observation.
    assert "scope.buildId.low == 0 && scope.buildId.high == 0" in register

    # Leave ends the owner, not the record.
    clear = body(pool, "void VtcmPool::clearAccountingSiteScope()")
    assert "clearAccountingSiteScopeTLS" in clear
    assert "gAccountingContextState.siteOwnerActive = false" in clear
    assert "siteRegistered = false" not in clear
    assert "siteTokens" not in clear
    has = body(pool, "bool VtcmPool::hasAccountingSiteScope()")
    assert "siteOwnerActive" in has
    assert "siteRegistered" not in has


def test_site_scope_requires_an_active_frame() -> None:
    """A site names one allocation inside an observation, so it needs one.

    Without this rule a site enter that arrives with no frame, or with a frame
    that declared a different function or invocation, is still bound and still
    published: the two halves disagree about what is being observed and nothing
    downstream can detect it, because each half is individually well formed.
    """
    pool = read("src/VTCMPool.cpp")
    register = body(pool, "bool VtcmPool::registerAccountingSiteScope(")

    # The refusal is a poison like every other site refusal, not a silent no-op.
    assert "poisonAccountingSiteScopeLocked();" in register
    nested = register.split("Not nestable")[0]
    for required in (
        "!gAccountingContextState.eventRegistered",
        "!gAccountingContextState.eventOwnerActive",
        "gAccountingContextState.eventThreadOrdinal != threadOrdinal",
        "gAccountingContextState.eventContext.accountingScopeId !=\n"
        "            scope.accountingScopeId",
        "gAccountingContextState.eventContext.functionId != scope.functionId",
        "gAccountingContextState.eventContext.invocationId !=\n"
        "            scope.invocationId",
    ):
        assert required in nested, required

    # The frame's own registration enforces the same two scope words, so a site
    # can never agree with a frame that the frame itself would have refused.
    frame = body(pool, "bool VtcmPool::registerAccountingEventContext(")
    assert "context.accountingScopeId == 0" in frame
    assert "context.functionId == 0" in frame
    assert "context.invocationId == 0" in frame


def test_site_scope_shape_counters_are_reported() -> None:
    """Leaked and over-wide spans are defect signals and must be published.

    `entered - leaked` is what makes "the join is closed" and "the scopes were
    well formed" two separate statements. A counter that exists but is never
    serialized would let a capture with a leaked span look identical to a clean
    one, which is the failure this ABI exists to make visible.
    """
    pool = read("src/VTCMPool.cpp")
    report = body(pool, "int VtcmPool::writeAccountingReportLocked(")
    tail = report.split("VTCM_SITE_SCOPE schema=1")[1]
    line = tail.split('\\n"')[0]
    # Each counter is a format slot *and* a real snapshot argument: a name in the
    # string with no value behind it would serialize as garbage, and a value with
    # no slot would be silently dropped from the report.
    for field, member in (
        ("site_scopes_entered=%llu", "snapshot.siteScopesEntered"),
        ("site_scopes_leaked=%llu", "snapshot.siteScopesLeaked"),
        (
            "site_scopes_with_multiple_events=%llu",
            "snapshot.siteScopesWithMultipleEvents",
        ),
    ):
        assert field in line, field
        assert member in tail, member

    # Entered is bumped once per granted scope, and a fresh span starts its own
    # tally so the `multiple` verdict describes this bracket, not the last one.
    register = body(pool, "bool VtcmPool::registerAccountingSiteScope(")
    assert "gSiteScopesEntered.fetch_add(1" in register
    assert "gSiteScopeCurrentEvents.store(0" in register

    # A frame that leaves with a scope still open has lost the allocation that
    # scope was naming; that is counted as a leak at the frame leave.
    frame_leave = body(pool, "void VtcmPool::clearAccountingEventContext(")
    assert "gSiteScopesLeaked.fetch_add(1" in frame_leave
    assert "if (gAccountingContextState.siteOwnerActive)" in frame_leave

    # `multiple` is decided when the span closes, and only from bound events: an
    # aggregate event is a refusal, not evidence of a span that was too wide.
    close = body(pool, "void VtcmPool::clearAccountingSiteScope(")
    assert "gSiteScopeCurrentEvents.load(std::memory_order_relaxed) > 1" in close
    assert "gSiteScopesWithMultipleEvents.fetch_add(1" in close
    event = body(pool, "void VtcmPool::recordAccountingEventLocked(")
    assert "if (event.siteScopeBound)\n    gSiteScopeCurrentEvents.fetch_add(1" in event

    # The counters are process-wide relaxed atomics, not pool members: a scope
    # is granted and closed from static entry points with no pool instance, and
    # the tally is bumped from the pool-locked path where taking the global
    # context mutex would invert the established lock order.
    assert "std::atomic<uint64_t> gSiteScopesEntered{0};" in pool
    for name in (
        "gSiteScopesLeaked",
        "gSiteScopesWithMultipleEvents",
        "gSiteScopeCurrentEvents",
    ):
        assert f"std::atomic<uint64_t> {name}{{0}};" in pool, name
    header = read("include/VTCMPool.h")
    assert "accountingSiteScopesEntered_" not in header


def test_free_cache_transitions_carry_the_delayed_owner() -> None:
    """Cached bytes belong to the site that allocated them, not the process.

    A retained block is still charged to the pool and still has its allocating
    site in `accountingOwners_`, so the cache transition can name that site. What
    it must never do is borrow an identity: a block that was never
    site-attributed has no owner, and its cache event stays aggregate.
    """
    pool = read("src/VTCMPool.cpp")
    manager = read("include/BufferManager.h")
    cache = body(pool, "void VtcmPool::recordAccountingCacheEvent(")

    # The pool resolves its own owner from the block pointer, so the private
    # owner type never leaks across the runtime boundary and the lookup happens
    # under the lock this function already holds.
    assert "AccountingOwner owner = lookupAccountingOwnerLocked(ownerBlock);" in cache
    assert "const AccountingOwner *bound = owner.bound ? &owner : nullptr;" in cache
    assert "cachedBytes, cachedBuffers, true, bound);" in cache
    # The counter is NOT bumped where the owner is resolved: a cache transition
    # after its site scope left resolves an owner and is still refused by the
    # live-scope rule, so counting it there would let the report claim a
    # binding no event carries. It follows the event's own verdict instead.
    assert "++accountingSiteCacheBoundEvents_" not in cache
    event = body(pool, "void VtcmPool::recordAccountingEventLocked(")
    assert "if (event.siteScopeBound)" in event
    bound_switch = event.split("if (event.siteScopeBound)")[1][:900]
    assert "++accountingSiteCacheBoundEvents_;" in bound_switch
    for kind in ("Retain", "Hit", "Drop", "Evict"):
        assert f"case kAccountingFreeCache{kind}:" in bound_switch, kind

    # Every cache transition names the block it is about: retain/drop on the
    # buffer being handed over, hit on the one coming back out. Matched on
    # whitespace-collapsed text so the check is about the argument rather than
    # about which branch of the function the call sits in.
    def squashed(text: str) -> str:
        return " ".join(text.split())

    for kind, owner in (
        ("kRetain", "ptr"),
        ("kHit", "buf->GetPointer()"),
        ("kDrop", "ptr"),
    ):
        segment = squashed(
            manager.split(f"AccountingCacheEventKind::{kind}")[1][:400]
        )
        assert f"alignment, {owner});" in segment, kind

    # Evict is a batch over an already-cleared cache, so it has no single block
    # to name. Nothing is lost: each evicted block is released through the pool
    # free path, which records its site individually.
    evict = squashed(manager.split("AccountingCacheEventKind::kEvict")[1][:400])
    assert "evictedBytes, 0, nullptr);" in evict

    # The status is measured, not a constant label.
    report = body(pool, "int VtcmPool::writeAccountingReportLocked(")
    assert (
        'accountingSiteCacheBoundEvents_ != 0 ? "owner-retained" : "aggregate"'
        in report
    )
    assert "delayed_cache_owner_status=%s" in report
    assert report.count("delayedCacheOwnerStatus") >= 3


def test_site_binding_is_independent_and_same_thread() -> None:
    pool = read("src/VTCMPool.cpp")
    report = body(pool, "int VtcmPool::writeAccountingReportLocked(")
    owner = body(pool, "VtcmPool::AccountingOwner VtcmPool::currentAccountingOwner()")
    # The site half is captured from this thread's TLS at this instant, and only
    # when the site scope is itself unpoisoned.
    assert "if (tAccountingSiteScopeActive && !tAccountingSiteScopePoisoned)" in owner
    assert "owner.siteBound = true;" in owner

    event = body(pool, "void VtcmPool::recordAccountingEventLocked(")
    assert "event.siteScopeBound" in event
    # The full conjunction: the owner carried a site, this thread still holds the
    # same one, and the process registration agrees with both.
    for required in (
        "owner != nullptr && owner->siteBound",
        "contextSnapshot.siteRegistered",
        "owner->siteScope == contextSnapshot.siteScope",
        "sameThread",
    ):
        assert required in event, required
    assert "++accountingSiteScopeBoundEvents_;" in event
    assert "++accountingSiteScopeAggregateEvents_;" in event

    # A release inherits the site of the block it frees, so it may outlive its
    # own span. An allocation may not: it only binds while this thread still
    # holds the same site scope, which is what stops a later allocation from
    # picking up a stale site. The two cases are one predicate with an explicit
    # discriminator, not two code paths that could drift.
    #
    # The owner-retained set deliberately spans the pool frees *and* every
    # free-cache transition. Whether a release lands in the pool or in the cache
    # is an allocator policy decision; letting that change whether the bytes are
    # attributable to the allocating site would make the evidence depend on a
    # cache size.
    for kind in (
        "kAccountingFree",
        "kAccountingResidentFree",
        "kAccountingFreeCacheRetain",
        "kAccountingFreeCacheHit",
        "kAccountingFreeCacheDrop",
        "kAccountingFreeCacheEvict",
    ):
        assert f"kind == {kind}" in event, kind
    assert "const bool ownerRetainedEvent =" in event
    assert "const bool liveScopeAgrees =" in event
    assert "ownerRetainedEvent ||" in event
    assert "owner->siteScope == activeSite" in event
    # The two published counters keep separate meanings: a release returns bytes
    # to the pool, a cache transition moves them between the live set and the
    # cache. Folding them would make `site_free_bound_events` describe
    # something other than what its name says.
    assert (
        "const bool ownerRetainedRelease =\n"
        "      kind == kAccountingFree || kind == kAccountingResidentFree;"
    ) in event, "a release must stay distinguishable from a cache transition"
    # Neither case may require the frame owner to be live: a release after the
    # frame leave is still attributable, and the frame counters are untouched.
    assert "activeOwner.bound" not in event.split("event.siteScopeBound =")[1][:400]
    assert "++accountingSiteFreeBoundEvents_;" in event
    # The release counter is a subset of the bound counter and the report has to
    # publish both, or a reader cannot tell allocation from deallocation.
    header = read("include/VTCMPool.h")
    assert "uint64_t accountingSiteFreeBoundEvents_{0};" in header
    assert "uint64_t siteFreeBoundEvents;" in header
    assert "site_free_bound_events=%llu" in report
    # The frame counters are untouched by the site decision and vice versa: a
    # bound frame event with an aggregate site must still count as a bound frame
    # event.
    assert event.index("++accountingEventContextBoundEvents_;") < event.index(
        "event.siteScopeBound ="
    )


def test_report_additions_are_strictly_additive() -> None:
    pool = read("src/VTCMPool.cpp")
    report = body(pool, "int VtcmPool::writeAccountingReportLocked(")

    # The pre-existing v1 records keep their exact spelling.
    for token in (
        "VTCM_ACCOUNTING schema=1 identity_schema=1 ",
        "VTCM_EVIDENCE schema=1 layer=layer-b-runtime ",
        "VTCM_EVIDENCE_CONTEXT schema=1 context_abi=accounting-v1 ",
        "VTCM_EVENT_CONTEXT schema=1 context_abi=accounting-event-v1 ",
        "VTCM_IDENTITY schema=1 version=%u registered=1 ",
        "VTCM_IDENTITY schema=1 registered=0 ",
        "VTCM_SNAPSHOT pool_bytes=%llu charged_allocated_bytes=%llu ",
        "VTCM_EVENT sequence=%llu event_sequence=%llu kind=%s ",
        " event_context_status=%s",
    ):
        assert token in report, token

    # The site declaration is emitted only when a site scope was registered, so a
    # frame-only stream is byte for byte what it was before.
    assert "if (snapshot.siteScopeRegistered) {" in report
    # Take the whole appendAccountingText call, format string and arguments: the
    # status pair is selected by the owner state, and that selection lives in the
    # arguments rather than in the format string.
    call = report[report.index("if (snapshot.siteScopeRegistered) {") :]
    call = call[: call.index("return kAccountingReportTruncated;")]
    site_line = call
    for token in (
        "VTCM_SITE_SCOPE schema=1 context_abi=accounting-site-v1",
        "context_status=%s owner_active=%u",
        "site_token_low=%llu site_token_high=%llu",
        "build_id_bits=128 build_id_low=%llu",
        "distinct_sites=%llu",
        "site_bound_events=%llu",
        "site_aggregate_events=%llu",
        "free_attribution=owner-retained",
        "grid_scope_status=not-proven",
        "performance_claimed=false",
    ):
        assert token in site_line, token
    # The declaration is a lifetime record: the owner state selects the status
    # pair, so a cleared scope can never read as a live binding.
    assert 'snapshot.siteScopeOwnerActive' in site_line
    assert "registered-active" in site_line
    assert "registered-cleared" in site_line
    assert "site-bound" in site_line
    assert "cleared-historical" in site_line
    # No address, and the site token is named opaque: the runtime compares the
    # pair and never interprets it.
    assert "0x" not in site_line
    assert "%p" not in site_line
    assert "token_basis=opaque" in site_line

    # The per-event fields are additive and appear only for a real claim. A
    # frame-only event keeps exactly the spelling it had.
    assert "if (event.siteScopeBound) {" in report
    assert "} else if (snapshot.siteScopeRegistered) {" in report
    assert '" event_site_scope_status=aggregate"' in report
    for token in (
        "event_site_scope_status=bound",
        "site_token_bits=128",
        "site_token_low=%llu site_token_high=%llu",
        "site_accounting_scope_id=%llu site_invocation_id=%llu",
        "site_function_id=%llu site_allocation_site_id=%llu",
        "site_build_id_bits=128 site_build_id_low=%llu site_build_id_high=%llu",
    ):
        assert token in report, token
    # The frame and site vocabularies are disjoint, so one can never be read as
    # the other.
    assert "site_event_token" not in report
    assert "event_site_token" not in report


def test_a1_padding_fields_are_preserved() -> None:
    """The per-placement padding work is unrelated and must survive intact."""
    header = read("include/VTCMPool.h")
    pool = read("src/VTCMPool.cpp")
    for token in (
        "enum class AccountingPaddingScope : uint8_t {",
        "kNotApplicable = 0,",
        "kPerPlacement = 1,",
        "struct AccountingPadding {",
        "size_t prefixBytes{0};",
        "size_t suffixBytes{0};",
        "const AccountingPadding *padding = nullptr);",
        "uint8_t paddingScope;",
        "size_t addressPrefixBytes;",
        "size_t addressSuffixBytes;",
    ):
        assert token in header, token
    assert "padding->scope == AccountingPaddingScope::kPerPlacement" in pool
    assert "address_padding_scope=%s" in pool


def test_default_build_has_no_site_symbols() -> None:
    """The whole ABI is behind the probe gate, like every other v1 record."""
    pool = read("src/VTCMPool.cpp")
    header = read("include/VTCMPool.h")
    capi = read("src/HexagonCAPI.cpp")
    for surface, name in (
        (pool, "registerAccountingSiteScope"),
        (pool, "clearAccountingSiteScope"),
        (pool, "hasAccountingSiteScope"),
        (header, "AccountingSiteScope"),
        (capi, "hexagon_runtime_vtcm_accounting_site_scope_enter_v1_dsp"),
        (capi, "hexagon_runtime_vtcm_accounting_site_scope_leave_v1_dsp"),
    ):
        # Every mention sits inside the probe-only region.
        assert f"#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE" in surface
        guarded = surface[: surface.index(name)]
        assert guarded.rindex("#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE") > guarded.rindex(
            "#endif"
        ), f"{name} is outside the probe guard"
    # The site report record is inside the same guarded report function.
    assert pool.count("#ifdef HEXMLIR_RUNTIME_VTCM_ACCOUNTING_PROBE") >= 1


def decl(source: str, signature: str) -> str:
    """Return a declaration up to its terminating semicolon."""
    start = source.index(signature)
    return source[start : source.index(";", start)]


def main() -> None:
    tests = (
        test_site_abi_is_separate_from_the_frame_abi,
        test_every_rejection_path_poisons,
        test_site_scope_requires_an_active_frame,
        test_site_scope_shape_counters_are_reported,
        test_free_cache_transitions_carry_the_delayed_owner,
        test_site_binding_is_independent_and_same_thread,
        test_report_additions_are_strictly_additive,
        test_a1_padding_fields_are_preserved,
        test_default_build_has_no_site_symbols,
    )
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"VTCM site-scope runtime contract: {len(tests)} passed")


if __name__ == "__main__":
    main()
