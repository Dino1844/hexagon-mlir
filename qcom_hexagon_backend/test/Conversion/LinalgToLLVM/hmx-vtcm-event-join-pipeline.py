#!/usr/bin/env python3
"""Prove the static-to-ABI per-site join on real `linalg-to-llvm` output.

The compiler half of the diagnostic event-context ABI is only meaningful if the
opaque token the static sidecar publishes is the token the lowered kernel
actually hands to the runtime.  A FileCheck pattern can show that a call exists,
but it cannot bind the nine call operands to the six sidecar words: the
constants are hoisted into unnamed-by-contract ``llvm.mlir.constant`` results and
two of them share one value.  This checker therefore reads the pipeline's own
output and requires exact equality.

What it checks, per module:

* ``hmx.kernel_vtcm_event_context`` is present and is either the closed eligible
  record or the closed ineligible one -- no third shape;
* the eligible record reconciles with ``hmx.kernel_vtcm_identity``: same
  ``accounting_scope_id``/``function``/``function_id``/``allocation_site_id`` and
  the same ``event_token_low``/``event_token_high`` the canonical site carries;
* exactly one ``enter_v1_dsp`` call exists, in the function the sidecar names,
  and its nine operands are exactly the ABI the runtime declares
  (version, flags, token low/high, accounting scope, invocation, function,
  site, grid product) with every 64-bit word equal to the sidecar after the
  usual textual-i64 to unsigned normalization;
* the enter call precedes the first VTCM runtime allocation, and every
  ``llvm.return`` is preceded by a ``leave_v1_dsp`` call, so the span cannot end
  before the allocation it is supposed to attribute;
* an ineligible record emits neither half of the ABI.

It reads only the printed module.  It does not run Triton, a launcher, or a
device, and it claims nothing about an observed allocation: the join proven here
is compiler-static to compiler-emitted, and the runtime half of the join is the
separate host transcript contract in ``exp/hmx/vtcm_accounting_probe``.
"""

from __future__ import annotations

import argparse
import re
import sys

INT64_FLOOR = -(1 << 63)
UINT64_MODULUS = 1 << 64

EVENT_CONTEXT_ATTR = "hmx.kernel_vtcm_event_context"
IDENTITY_ATTR = "hmx.kernel_vtcm_identity"
ENTER_FN = "hexagon_runtime_vtcm_accounting_event_context_enter_v1_dsp"
LEAVE_FN = "hexagon_runtime_vtcm_accounting_event_context_leave_v1_dsp"
VTCM_ALLOC_FN = "hexagon_runtime_alloc_1d_dsp"

# Mirrors hmxDiagnosticEventContextEligibleKeys() in
# include/hexagon/Dialect/Hmx/Transforms/HmxResidentContract.h.  The set is
# closed on purpose: an extra key such as a promotion field would otherwise be
# read as "the field I wanted was absent".
ELIGIBLE_KEYS = frozenset(
    {
        "kind",
        "schema",
        "status",
        "eligible",
        "mode",
        "function",
        "grid_product",
        "invocation_id",
        "immutable",
        "token_basis",
        "delayed_cache_owner_status",
        "grid_scope_status",
        "runtime_event_join_status",
        "performance_claimed",
        "accounting_scope_id",
        "function_id",
        "allocation_site_id",
        "token_bits",
        "token_low",
        "token_high",
    }
)
INELIGIBLE_KEYS = frozenset(
    {
        "kind",
        "schema",
        "status",
        "eligible",
        "mode",
        "function",
        "grid_product",
        "invocation_id",
        "immutable",
        "token_basis",
        "delayed_cache_owner_status",
        "grid_scope_status",
        "runtime_event_join_status",
        "performance_claimed",
        "reason",
    }
)
# The only reason a one-function, grid=1 module with a complete identity and
# complete accounting can still be ineligible.
ONE_SITE_REASON = "event context requires one function and one canonical site"

ABI_VERSION = 1
ABI_FLAG_SINGLE_INVOCATION = 1 << 0
ABI_FLAG_GRID_ONE = 1 << 1
ABI_FLAGS = ABI_FLAG_SINGLE_INVOCATION | ABI_FLAG_GRID_ONE


class JoinError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# A strict reader for the printed MLIR attribute subset the two sidecars use.
# Scalars are ``-?[0-9]+ : iN``, quoted strings, ``true``/``false``, and the two
# composites ``{}`` and ``[]``.  Anything else is an error rather than a guess.
# --------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
      (?P<space>\s+)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<integer>-?[0-9]+)(?P<type>\s*:\s*[a-zA-Z0-9_]+)?
    | (?P<open>[\{\[])
    | (?P<close>[\}\]])
    | (?P<equals>=)
    | (?P<comma>,)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_.$]*)
    """,
    re.VERBOSE,
)


def _read_attributes(text: str, start: int) -> tuple[dict[str, object], int]:
    """Parse one ``{...}`` attribute dictionary beginning at ``text[start]``."""

    def read_string(position: int) -> tuple[str, int]:
        match = _TOKEN.match(text, position)
        if match is None or match.group("string") is None:
            raise JoinError(f"expected a quoted string at offset {position}")
        # The token includes its delimiters. They are part of the syntax, not of
        # the value, so they are stripped here once: a comparison against a
        # schema word must not have to know how the printer spelled it. An
        # unterminated literal cannot match the pattern at all and is therefore
        # rejected by the match above rather than silently closed here.
        raw = match.group("string")[1:-1]
        return raw.encode("utf-8").decode("unicode_escape"), match.end()

    def read_value(position: int) -> tuple[object, int]:
        # Printed attributes separate `=` from their value, so skipping here is
        # the only thing that makes `= 3 : i32` readable.
        position = _skip(position)
        match = _TOKEN.match(text, position)
        if match is None:
            raise JoinError(f"unparsable attribute value at offset {position}")
        # `lastgroup` names the *last* matched group, and the integer pattern
        # matches both `integer` and `type`, so it would report "type" even for
        # a bare number. Each alternative is therefore selected by which group
        # actually participated.
        if match.group("open"):
            return read_composite(match.end(), match.group("open"))
        if match.group("integer") is not None:
            if match.group("type") is None:
                raise JoinError(
                    f"integer attribute at offset {position} has no type"
                )
            return int(match.group("integer")), match.end()
        if match.group("string") is not None:
            return read_string(position)[0], match.end()
        if match.group("ident"):
            token = match.group("ident")
            if token not in ("true", "false"):
                raise JoinError(f"unsupported attribute value: {token}")
            return token == "true", match.end()
        raise JoinError(f"unexpected token at offset {position}")

    def read_composite(position: int, opener: str) -> tuple[object, int]:
        closer = "}" if opener == "{" else "]"
        if closer == "}":
            fields: dict[str, object] = {}
            position = _skip(position)
            if text[position] == "}":
                return fields, position + 1
            while True:
                position = _skip(position)
                name_match = _TOKEN.match(text, position)
                if name_match is None or name_match.group("ident") is None:
                    raise JoinError(f"expected a field name at offset {position}")
                name = name_match.group("ident")
                position = _skip(name_match.end())
                equals = _TOKEN.match(text, position)
                if equals is None or equals.group("equals") is None:
                    raise JoinError(f"field {name} has no value at offset {position}")
                # `read_value` skips the whitespace between `=` and the value.
                value, position = read_value(equals.end())
                if name in fields:
                    raise JoinError(f"duplicate attribute field: {name}")
                fields[name] = value
                position = _skip(position)
                separator = _TOKEN.match(text, position)
                if separator is None:
                    raise JoinError(f"unterminated attribute after field {name}")
                if separator.group("comma"):
                    position = separator.end()
                    continue
                if separator.group("close"):
                    return fields, position + 1
                raise JoinError(f"unexpected token after field {name}")
        elements: list[object] = []
        position = _skip(position)
        if text[position] == "]":
            return elements, position + 1
        while True:
            value, position = read_value(position)
            elements.append(value)
            position = _skip(position)
            separator = _TOKEN.match(text, position)
            if separator is None:
                raise JoinError("unterminated attribute array")
            if separator.group("comma"):
                position = separator.end()
                continue
            if separator.group("close"):
                return elements, position + 1
            raise JoinError("unexpected token in attribute array")

    def _skip(position: int) -> int:
        match = _TOKEN.match(text, position)
        if match is not None and match.group("space"):
            return match.end()
        return position

    if text[start] != "{":
        raise JoinError("expected '{' at the start of an attribute dictionary")
    return read_composite(start + 1, "{")


def read_module_attribute(text: str, name: str) -> dict[str, object] | None:
    """Read one module-level attribute dictionary, or None when it is absent."""

    marker = re.search(rf"(?<![A-Za-z0-9_.]){re.escape(name)} = ", text)
    if marker is None:
        return None
    position = marker.end()
    while position < len(text) and text[position] in " \t":
        position += 1
    if position >= len(text) or text[position] != "{":
        raise JoinError(f"module attribute {name} is not a dictionary")
    fields, _ = _read_attributes(text, position)
    return fields


def to_unsigned(value: object, label: str) -> int:
    """Normalize a textual MLIR ``i64`` identifier to its unsigned value.

    MLIR prints ``i64`` attributes in signed decimal, so an identifier whose
    64-bit pattern has the top bit set arrives negative.  Two's complement is
    the only faithful reading of that text, and the runtime prints the same word
    as an unsigned decimal, so the join is made on the unsigned value.  A value
    that cannot be a textual i64 stays malformed instead of being wrapped.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise JoinError(f"{label} is not an integer")
    if value < INT64_FLOOR or value >= UINT64_MODULUS:
        raise JoinError(f"{label} is not a textual i64 value")
    return value & (UINT64_MODULUS - 1)


# --------------------------------------------------------------------------
# The emitted ABI.
# --------------------------------------------------------------------------

_FUNC_START = re.compile(r"^  (?:llvm\.func|func\.func) @([A-Za-z0-9_.$]+)")
_CONSTANT = re.compile(
    r"^(?P<reg>%[0-9]+) = llvm\.mlir\.constant\((?P<value>-?[0-9]+) : (?P<type>[a-z0-9]+)\)"
)
_CALL = re.compile(r"llvm\.call @([A-Za-z0-9_.$]+)\((?P<operands>[^)]*)\)")


def _function_bodies(text: str) -> list[tuple[str, str]]:
    """Return (name, body) for every function definition in the module."""

    bodies: list[tuple[str, str]] = []
    current_name: str | None = None
    current: list[str] = []
    for line in text.splitlines():
        header = _FUNC_START.match(line)
        if header is not None:
            if current_name is not None:
                bodies.append((current_name, "\n".join(current)))
            current_name = header.group(1)
            current = [line]
            continue
        if current_name is None:
            continue
        if line.startswith("}") and not line.startswith("  }"):
            bodies.append((current_name, "\n".join(current)))
            current_name = None
            current = []
            continue
        current.append(line)
    if current_name is not None:
        bodies.append((current_name, "\n".join(current)))
    return bodies


def read_emitted_abi(body: str) -> dict[str, int] | None:
    """Resolve the enter call's nine operands against the function's constants."""

    constants: dict[str, int] = {}
    enter_operands: list[str] | None = None
    for line in body.splitlines():
        constant = _CONSTANT.match(line.strip())
        if constant is not None:
            constants[constant.group("reg")] = int(constant.group("value"))
            continue
        call = _CALL.search(line)
        if call is not None and call.group(1) == ENTER_FN:
            if enter_operands is not None:
                raise JoinError("more than one event-context enter call")
            enter_operands = [operand.strip() for operand in call.group("operands").split(",")]
    if enter_operands is None:
        return None
    if len(enter_operands) != 9:
        raise JoinError(
            f"event-context enter call has {len(enter_operands)} operands, expected 9"
        )
    values: list[int] = []
    for operand in enter_operands:
        if operand not in constants:
            raise JoinError(f"event-context operand {operand} is not a constant")
        values.append(constants[operand])
    return dict(
        zip(
            (
                "version",
                "flags",
                "token_low",
                "token_high",
                "accounting_scope_id",
                "invocation_id",
                "function_id",
                "allocation_site_id",
                "grid_product",
            ),
            values,
        )
    )


def call_positions(body: str, callee: str) -> list[int]:
    positions = []
    for index, line in enumerate(body.splitlines()):
        call = _CALL.search(line)
        if call is not None and call.group(1) == callee:
            positions.append(index)
    return positions


# --------------------------------------------------------------------------
# The join itself.
# --------------------------------------------------------------------------

_MODULE_HEADER = re.compile(r"^module(?: @(?P<symbol>[A-Za-z0-9_.$]+))? attributes")
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def split_modules(text: str) -> list[tuple[str, str]]:
    """Split printed output into its top-level modules.

    The pipeline output is not one module per input chunk: with
    ``-split-input-file`` the async-runtime pass prepends a bare module holding
    its private runtime declarations, so positional indexing would read that
    preamble as the first fixture module.  Modules are therefore collected by
    brace depth, and the ``// -----`` separators are just line noise.  String
    literals are blanked before counting so a brace inside one cannot move the
    depth.
    """

    modules: list[tuple[str, str]] = []
    current: list[str] | None = None
    depth = 0
    for line in text.splitlines():
        if current is None:
            header = _MODULE_HEADER.match(line)
            if header is None:
                continue
            current = [line]
            depth = 0
        else:
            current.append(line)
        depth += _STRING.sub("", line).count("{") - _STRING.sub("", line).count("}")
        if current is not None and depth == 0 and len(current) > 1:
            header = _MODULE_HEADER.match(current[0])
            modules.append((header.group("symbol") or "", "\n".join(current)))
            current = None
    if current is not None:
        raise JoinError("unterminated module in the pipeline output")
    return modules


def select_modules(
    text: str, expected: list[str]
) -> list[tuple[str, str]]:
    """Select exactly the modules the caller named, and nothing else.

    A module that carries no event-context sidecar is ignored, which is what
    makes the runtime-declaration preamble irrelevant.  A module that *does*
    carry one but was not named is an error rather than a silent extra check:
    the fixture owns the transcript, so an unexpected marked module means the
    input and the expectations have drifted apart.
    """

    # `--expect @sym` is how the symbol is written in the fixture, but the
    # printed module header captures the bare name.
    expected = [symbol[1:] if symbol.startswith("@") else symbol for symbol in expected]
    if not expected:
        raise JoinError("no expected module symbol was given")
    found = {symbol: [] for symbol in expected}
    marked = []
    for symbol, body in split_modules(text):
        if read_module_attribute(body, EVENT_CONTEXT_ATTR) is not None:
            marked.append(symbol or "<anonymous>")
        if symbol in found:
            found[symbol].append(body)
    for symbol in expected:
        if not found[symbol]:
            raise JoinError(f"no module named {symbol} in the pipeline output")
        if len(found[symbol]) > 1:
            raise JoinError(f"module symbol {symbol} is ambiguous")
    unexpected = sorted(name for name in marked if name not in set(expected))
    if unexpected:
        raise JoinError(
            f"event-context sidecar in unexpected module(s): {unexpected}"
        )
    return [(symbol, found[symbol][0]) for symbol in expected]


def check_module(text: str) -> None:
    context = read_module_attribute(text, EVENT_CONTEXT_ATTR)
    if context is None:
        raise JoinError(f"module has no {EVENT_CONTEXT_ATTR}")
    eligible = context.get("eligible")
    if not isinstance(eligible, bool):
        raise JoinError("event-context sidecar has no BoolAttr 'eligible'")
    enter_calls = sum(
        len(call_positions(body, ENTER_FN)) for _, body in _function_bodies(text)
    )
    leave_calls = sum(
        len(call_positions(body, LEAVE_FN)) for _, body in _function_bodies(text)
    )
    if not eligible:
        if set(context) != INELIGIBLE_KEYS:
            raise JoinError(
                "ineligible sidecar is not the closed ineligible record: "
                f"{sorted(set(context) ^ INELIGIBLE_KEYS)}"
            )
        if enter_calls or leave_calls:
            raise JoinError(
                "an ineligible event-context sidecar must emit neither ABI half"
            )
        if context.get("status") != "not-proven":
            raise JoinError("ineligible sidecar must not claim a complete status")
        print(f"ok: ineligible sidecar refused ({context.get('reason')!r})")
        return

    if set(context) != ELIGIBLE_KEYS:
        raise JoinError(
            "eligible sidecar is not the closed eligible record: "
            f"{sorted(set(context) ^ ELIGIBLE_KEYS)}"
        )
    for key, value in (
        ("kind", "vtcm-event-context"),
        ("schema", "hmx.vtcm-event-context/v1"),
        ("status", "complete"),
        ("mode", "diagnostic-only"),
        ("token_basis", "hmx.vtcm-event-token/fnv1a128/v1"),
        ("delayed_cache_owner_status", "aggregate"),
        ("grid_scope_status", "not-proven"),
        ("runtime_event_join_status", "not-proven"),
    ):
        if context.get(key) != value:
            raise JoinError(f"sidecar {key}={context.get(key)!r}, expected {value!r}")
    if context.get("token_bits") != 128:
        raise JoinError("sidecar token_bits must be 128")
    if context.get("grid_product") != 1 or context.get("invocation_id") != 1:
        raise JoinError("sidecar must declare grid_product=1 and invocation_id=1")
    if context.get("immutable") != 1:
        raise JoinError("an eligible sidecar must declare immutable=1")
    if context.get("performance_claimed") != 0:
        raise JoinError("an eligible sidecar must not claim performance")
    function = context.get("function")
    if not isinstance(function, str) or not function:
        raise JoinError("sidecar names no function")

    scope = to_unsigned(context.get("accounting_scope_id"), "accounting_scope_id")
    function_id = to_unsigned(context.get("function_id"), "function_id")
    site_id = to_unsigned(context.get("allocation_site_id"), "allocation_site_id")
    token_low = to_unsigned(context.get("token_low"), "token_low")
    token_high = to_unsigned(context.get("token_high"), "token_high")

    # The token is derived from the canonical site, so the identity sidecar is
    # the independent record of what it was derived from.  Two sidecars that
    # disagree are a refusal, not a choice.
    identity = read_module_attribute(text, IDENTITY_ATTR)
    if identity is None:
        raise JoinError(f"module has no {IDENTITY_ATTR}")
    if to_unsigned(identity.get("scope_id"), "identity scope_id") != scope:
        raise JoinError("identity scope_id does not match the event context")
    if identity.get("status") != "complete":
        raise JoinError("identity sidecar is not complete")
    functions = identity.get("functions")
    if not isinstance(functions, list) or len(functions) != 1:
        raise JoinError("eligible scope must name exactly one identity function")
    claimed = functions[0]
    if not isinstance(claimed, dict):
        raise JoinError("identity function entry is not a dictionary")
    if claimed.get("function") != function:
        raise JoinError("identity function name does not match the event context")
    if to_unsigned(claimed.get("function_id"), "identity function_id") != function_id:
        raise JoinError("identity function_id does not match the event context")
    sites = claimed.get("sites")
    if not isinstance(sites, list) or len(sites) != 1:
        raise JoinError("eligible scope must name exactly one canonical site")
    site = sites[0]
    if not isinstance(site, dict):
        raise JoinError("identity site entry is not a dictionary")
    if site.get("identity_status") != "complete":
        raise JoinError("canonical site identity is not complete")
    if site.get("event_token_basis") != "hmx.vtcm-event-token/fnv1a128/v1":
        raise JoinError("canonical site does not carry the event-token basis")
    if site.get("event_token_bits") != 128:
        raise JoinError("canonical site event_token_bits must be 128")
    for label, printed, expected in (
        ("site_id", site.get("site_id"), site_id),
        ("function_id", site.get("function_id"), function_id),
        ("event_token_low", site.get("event_token_low"), token_low),
        ("event_token_high", site.get("event_token_high"), token_high),
    ):
        if to_unsigned(printed, f"identity {label}") != expected:
            raise JoinError(f"identity {label} does not match the event context")

    # The compiler half of the join: the lowered call must carry these words.
    if enter_calls != 1:
        raise JoinError(f"expected exactly one enter call, found {enter_calls}")
    bodies = {name: body for name, body in _function_bodies(text)}
    if function not in bodies:
        raise JoinError(f"the lowered module has no function named {function!r}")
    body = bodies[function]
    abi = read_emitted_abi(body)
    if abi is None:
        raise JoinError("the named function emits no event-context enter call")
    expected_abi = {
        "version": ABI_VERSION,
        "flags": ABI_FLAGS,
        "token_low": token_low,
        "token_high": token_high,
        "accounting_scope_id": scope,
        "invocation_id": 1,
        "function_id": function_id,
        "allocation_site_id": site_id,
        "grid_product": 1,
    }
    for key, expected in expected_abi.items():
        emitted = abi[key] & (UINT64_MODULUS - 1) if key in (
            "token_low",
            "token_high",
            "accounting_scope_id",
            "function_id",
            "allocation_site_id",
        ) else abi[key]
        if emitted != expected:
            raise JoinError(
                f"emitted {key}={emitted} does not match the sidecar value {expected}"
            )

    # The span must contain the allocation it attributes, and must not end
    # before it: enter first, then the VTCM allocation, and a leave before each
    # return so no thread can inherit a stale owner.
    lines = body.splitlines()
    enter_line = call_positions(body, ENTER_FN)[0]
    allocation_lines = call_positions(body, VTCM_ALLOC_FN)
    if not allocation_lines:
        raise JoinError("the named function makes no VTCM allocation to attribute")
    if enter_line > allocation_lines[0]:
        raise JoinError("the enter call does not precede the VTCM allocation")
    leave_lines = call_positions(body, LEAVE_FN)
    returns = [index for index, line in enumerate(lines) if "llvm.return" in line]
    if not returns:
        raise JoinError("the named function has no return to leave before")
    for index in returns:
        if not any(leave < index for leave in leave_lines):
            raise JoinError("a return is not preceded by a leave call")
    print(
        "ok: joined function={} scope={} function_id={} site_id={}".format(
            function, scope, function_id, site_id
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="SYMBOL",
        help="module symbol whose event-context sidecar must be checked; "
        "repeatable, and every named symbol must appear exactly once",
    )
    args = parser.parse_args()
    try:
        modules = select_modules(sys.stdin.read(), args.expect)
    except JoinError as error:
        print(f"event-join: FAIL: {error}", file=sys.stderr)
        return 1
    for symbol, body in modules:
        try:
            check_module(body)
        except JoinError as error:
            print(f"event-join: FAIL ({symbol}): {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
