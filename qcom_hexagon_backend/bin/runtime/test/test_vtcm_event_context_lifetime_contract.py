#!/usr/bin/env python3
"""Focused source contract for the diagnostic event-context lifetime.

This deliberately does not run the exp parser or touch probe files. It checks
only the narrow R-A ABI/lifetime boundary implemented by the compiler and
runtime sources.
"""
from pathlib import Path
import re

RUNTIME = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (RUNTIME / path).read_text(encoding="utf-8")


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


def main() -> None:
    header = read("include/VTCMPool.h")
    capi_header = read("include/HexagonCAPI.h")
    pool = read("src/VTCMPool.cpp")
    capi = read("src/HexagonCAPI.cpp")
    contract = read(
        "../../include/hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h")
    compiler = read("../../lib/Conversion/HmxToLLVM/HmxToLLVMPass.cpp")

    assert "uint64_t accountingScopeId{0};" in header
    assert "clearAccountingEventContext" in header
    assert "eventThreadOrdinal" in pool
    assert "tAccountingEventContextPoisoned" in pool
    assert "clearAccountingEventContextTLS" in pool

    register = body(pool, "bool VtcmPool::registerAccountingEventContext(")
    assert "gAccountingContextState.sawEventForEventContext" in register
    assert "gAccountingContextState.eventThreadOrdinal != threadOrdinal" in register
    # Every rejection path poisons the thread owner, whether it is reached
    # before or while the context lock is held.
    assert register.count("poisonAccountingEventContextLocked()") >= 2
    assert register.count("poisonAccountingEventContext()") >= 1
    assert "tAccountingEventContextPoisoned = false" in register

    # A successful enter is the only thing that makes an owner active, and the
    # owner-active fact is separate from the historical registration.
    assert "gAccountingContextState.eventOwnerActive = true" in register
    poison_locked = body(pool, "void poisonAccountingEventContextLocked(")
    assert "gAccountingContextState.eventOwnerActive = false" in poison_locked
    poison = body(pool, "void poisonAccountingEventContext(")
    assert "poisonAccountingEventContextLocked" in poison
    assert "lock_guard" in poison, "the unlocked poison must take the lock"

    clear = body(pool, "void VtcmPool::clearAccountingEventContext()")
    assert "clearAccountingEventContextTLS" in clear
    # Leave ends the owner, not the record: the registration, the declared
    # identity and the monotonic counters survive.
    assert "gAccountingContextState.eventOwnerActive = false" in clear
    assert "eventRegistered = false" not in clear
    assert "BoundEvents" not in clear
    assert "AggregateEvents" not in clear
    has = body(pool, "bool VtcmPool::hasAccountingEventContext()")
    assert "eventOwnerActive" in has
    assert "eventRegistered" not in has
    assert "eventContextOwnerActive" in header

    assert "hexagon_runtime_vtcm_accounting_event_context_leave_v1" in capi_header
    assert "hexagon_runtime_vtcm_accounting_event_context_leave_v1_dsp" in capi
    assert "VtcmPool::clearAccountingEventContext" in capi

    # The compiler emits both halves of the lifetime. The leave is inserted
    # before every LLVM return, so a worker thread cannot reuse a stale owner.
    assert "kHmxDiagnosticEventContextLeaveFn" in contract
    assert "kHmxDiagnosticEventContextLeaveFn" in compiler
    assert "LLVM::ReturnOp" in compiler
    assert "event_context_leave_v1_dsp" in contract
    assert "accounting_scope_id" in compiler
    assert "getZExtValue()" in compiler
    assert not re.search(r'getAttr\("scope_id"\)', compiler)

    report = body(pool, "int VtcmPool::writeAccountingReportLocked(")
    assert "accounting_scope_id=%llu" in report
    assert "event_accounting_scope_id=%llu" in report
    # The declaration may not keep claiming a live token binding once the owner
    # has left: it names the cleared state and reports the owner explicitly.
    assert "context_status=%s" in report
    assert "owner_active=%u" in report
    assert "registered-active" in report
    assert "registered-cleared" in report
    assert "token-bound" in report
    assert "cleared-historical" in report
    assert "context_status=registered " not in report
    assert "event_owner_status=token-bound " not in report
    assert "0x" not in report
    assert "%p" not in report

    print("VTCM event-context lifetime source contract: PASS")


def test_event_context_lifetime_source_contract() -> None:
    """The assertions above, under pytest.

    Without this the file collects zero tests, so the whole contract is a
    no-op in the pytest style while still passing as a script.  Both styles must
    run the same assertions, so this delegates rather than restating them.
    """
    main()


if __name__ == "__main__":
    main()
