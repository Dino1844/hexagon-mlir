#!/usr/bin/env python3
"""Regression tests for the hmx-vtcm-event-join-pipeline attribute reader.

The reader is the only part of that checker that is not a direct equality check
against the compiler's own words, so it gets its own tests. They run without
linalg-hexagon-opt, without a device, and without importing MLIR or Triton.

The cases below are the two real defects this file exists for:

* whitespace between `=` and the value. Printed attributes always have some,
  but the amount is not a contract, so `a = 5`, `a =   5`, and `a =\\n\\t5`
  must all read the same.
* a token pattern with more than one participating group. The integer pattern
  matches both the digits and the `: i64` suffix, so selecting the alternative
  by "last matched group" silently reported the wrong alternative and every
  integer attribute was rejected. Each alternative is therefore selected by
  which group actually participated.

Strictness is asserted as well as acceptance: a missing type, a missing value,
a missing `=`, a duplicate field, an unquoted word, and an unterminated
dictionary must all stay rejected. A parser that had been loosened to make the
positive cases pass would fail this file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CHECKER = HERE / "hmx-vtcm-event-join-pipeline.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("event_join_pipeline", CHECKER)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {CHECKER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JOIN = load_checker()


def parse(fragment: str) -> dict:
    fields, _ = JOIN._read_attributes(fragment, 0)
    return fields


def test_whitespace_after_equals_is_not_a_contract() -> None:
    # The printed shape has one space; the amount is not part of the syntax.
    assert parse("{a = 5 : i64}") == {"a": 5}
    assert parse("{a =   5 : i64}") == {"a": 5}
    assert parse("{a =\n\t5 : i64}") == {"a": 5}
    assert parse("{a=5 : i64}") == {"a": 5}
    # Leading and trailing space, and space around the separators, too.
    assert parse("{ a = 5 : i64 , b = 6 : i64 }") == {"a": 5, "b": 6}


def test_multi_group_tokens_select_by_participation() -> None:
    # `integer` participates together with `type`; reading "lastgroup" would
    # name `type` and reject the value.
    assert parse("{a = -1 : i64}") == {"a": -1}
    assert parse("{a = 0 : i32}") == {"a": 0}
    assert parse("{a = 18446744073709551615 : i64}") == {"a": 18446744073709551615}
    # The textual-i64 floor is a legal printed value; the reader does not judge
    # the range, `to_unsigned` does, and it judges it as a typed i64.
    assert parse("{a = -9223372036854775808 : i64}") == {"a": -9223372036854775808}
    # An integer that is genuinely out of textual-i64 range is read, then
    # refused by the typed normalizer rather than wrapped by the reader.
    assert parse("{a = 18446744073709551616 : i64}") == {"a": 18446744073709551616}
    try:
        JOIN.to_unsigned(18446744073709551616, "a")
    except JOIN.JoinError:
        pass
    else:
        raise AssertionError("to_unsigned must refuse a value outside textual i64")


def test_composites_strings_and_booleans() -> None:
    assert parse("{s = \"vtcm-event-context\"}") == {"s": "vtcm-event-context"}
    # The delimiters are syntax: the value must not carry them.
    assert parse("{s = \"x\"}")["s"] == "x"
    assert parse("{b = true, c = false}") == {"b": True, "c": False}
    assert parse("{n = {q = 1 : i8}}") == {"n": {"q": 1}}
    assert parse("{l = [1 : i8, 2 : i8]}") == {"l": [1, 2]}
    assert parse("{}") == {}


def test_strictness_is_preserved() -> None:
    rejected = (
        ("{a = 5}", "integer without a type"),
        ("{a = }", "missing value"),
        ("{a}", "missing ="),
        ("{a = 5 : i64, a = 6 : i64}", "duplicate field"),
        ("{a = nope}", "unquoted word"),
        ("{a = 5 : i64", "unterminated dictionary"),
        ("{a = \"unterminated}", "unterminated string"),
        ("{a = [1 : i8}", "unterminated array"),
        ("{a = {b = 1 : i8}", "unterminated nested dictionary"),
    )
    for fragment, why in rejected:
        try:
            parse(fragment)
        except JOIN.JoinError:
            continue
        raise AssertionError(f"parser accepted {why}: {fragment}")


def test_module_split_ignores_the_declaration_preamble() -> None:
    # `linalg-hexagon-opt -split-input-file` prints a bare module of private
    # async-runtime declarations before the fixture's own modules. Module
    # selection must not depend on that preamble being absent, and must not
    # depend on position either.
    preamble = (
        'module attributes {llvm.target_triple = "hexagon"} {\n'
        "  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr, i64) attributes "
        '{sym_visibility = "private"}\n'
        "}\n"
        "// -----\n"
    )
    marked = (
        "module @target attributes {hmx.kernel_vtcm_event_context = "
        '{kind = "vtcm-event-context", eligible = false}} {\n'
        "  llvm.func @kernel() {\n"
        "    llvm.return\n"
        "  }\n"
        "}\n"
    )
    text = preamble + "// -----\n" + marked
    modules = dict(JOIN.split_modules(text))
    assert "" in modules, "the anonymous preamble module must still be recognised"
    assert "target" in modules
    # Selecting the named module ignores the preamble rather than reading it as
    # the first target.
    selected = JOIN.select_modules(text, ["target"])
    assert [symbol for symbol, _ in selected] == ["target"]
    # A named module that is not in the output is a failure, not an empty pass.
    try:
        JOIN.select_modules(text, ["absent"])
    except JOIN.JoinError:
        pass
    else:
        raise AssertionError("a missing expected module must fail")


TESTS = (
    test_whitespace_after_equals_is_not_a_contract,
    test_multi_group_tokens_select_by_participation,
    test_composites_strings_and_booleans,
    test_strictness_is_preserved,
    test_module_split_ignores_the_declaration_preamble,
)


def main() -> int:
    for test in TESTS:
        test()
        print(f"ok  {test.__name__}")
    print(f"event-join parser: {len(TESTS)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
