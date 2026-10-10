#!/usr/bin/env python3
"""Drive the real tile notice with the manifests this pipeline really emits.

Why this file exists
--------------------
The launch-time tile recommendation (hmx_tile_notice in backend/utils.py) is
built from three published facts: a record's static logical shape, its
execution.block_m, and the VTCM budget/committed bytes beside it. This test
pins those facts to the compiler's output rather than to a host fixture: the
real pass pipeline runs over the sibling .mlir, and the real reporter reads
what came back.

So it checks the two halves that matter:

* the compile side -- a contraction the bridge must walk in M blocks publishes
  block_m below M, which is the only input that makes this notice speak;
* the reporter side -- that fact produces exactly one notice carrying the
  manifest's own numbers, while the two other shapes (a whole-block contraction
  and a budget refusal) produce none. Silence on the whole-block case is the
  property that keeps the notice off the default path.

The manifest is read out of the printed module attribute with a brace match
and a handful of field patterns, the same way the sibling VTCM-identity helper
reads its sidecar. A producer that stops publishing a field fails here, which
is the point: the reporter would otherwise fall silent for a reason no other
test could see.
"""

import importlib.util
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
UTILS = HERE.parents[4] / "backend" / "utils.py"

_SPEC = importlib.util.spec_from_file_location("hexagon_backend_utils", UTILS)
assert _SPEC is not None and _SPEC.loader is not None
UTILS_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(UTILS_MODULE)

_PLAN = re.compile(r'plan = "([^"]+)"')
_FUNCTION = re.compile(r'function = "([^"]+)"')
_BLOCK_M = re.compile(r"block_m = (-?\d+) : i64")
_BEFORE = re.compile(r"vtcm_before_bytes = (-?\d+) : i64")
_BUDGET = re.compile(r"vtcm_budget_bytes = (-?\d+) : i64")
_LOGICAL = re.compile(r"logical = \{")
_AXIS = re.compile(r'(m|n|k) = \{kind = "static", value = (-?\d+) : i64\}')
# Nested dictionaries that also name the function, so a record's own field is
# not confused with a reference to it.
_NESTED = ("weight_binding",)


def _cut_nested(text, keys):
    """Drop each key's balanced sub-dictionary, so only top-level fields remain."""
    for key in keys:
        header = f"{key} = {{"
        start = 0
        while True:
            begin = text.find(header, start)
            if begin < 0:
                break
            depth = 0
            for index in range(begin + len(header) - 1, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[:begin] + text[index + 1 :]
                        start = begin
                        break
            else:
                return text
    return text


def _attribute_bodies(text, header):
    """Each attribute's text, cut at its matching closing brace.

    The printer may wrap a long dictionary across lines, so a line-based read
    is not enough; cutting on the balanced brace keeps one module's facts out
    of the next module's.
    """
    bodies = []
    start = 0
    while True:
        begin = text.find(header, start)
        if begin < 0:
            return bodies
        open_brace = text.find("{", begin)
        if open_brace < 0:
            return bodies
        depth = 0
        for index in range(open_brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    bodies.append(text[open_brace : index + 1])
                    start = index + 1
                    break
        else:
            return bodies


def _records(text):
    """One manifest record per module the pipeline printed."""
    found = []
    for body in _attribute_bodies(text, "hmx.kernel_manifest = "):
        body = _cut_nested(body, _NESTED)
        functions = _FUNCTION.findall(body)
        if len(functions) != 1:
            raise SystemExit(
                f"expected exactly one matmul record per manifest, got {functions}"
            )
        plans = set(_PLAN.findall(body))
        if len(plans) != 1:
            raise SystemExit(f"expected one plan per record, got {plans}")
        logical = {}
        logical_begin = _LOGICAL.search(body)
        if not logical_begin:
            raise SystemExit(f"{functions[0]}: manifest has no logical shape")
        for axis, value in _AXIS.findall(body[logical_begin.end() :]):
            if axis in logical:
                break
            logical[axis] = {"kind": "static", "value": int(value)}
        if sorted(logical) != ["k", "m", "n"]:
            raise SystemExit(
                f"{functions[0]}: logical shape is not fully static: {logical}"
            )
        record = {
            "function": functions[0],
            "plan": plans.pop(),
            "logical": logical,
        }
        block_m = _BLOCK_M.search(body)
        if block_m:
            record["execution"] = {"block_m": int(block_m.group(1))}
        before = _BEFORE.search(body)
        budget = _BUDGET.search(body)
        # Only an attributed record publishes the budget facts, and the notice
        # needs them; a refused record publishes neither.
        if before and budget:
            record["vtcm_before_bytes"] = int(before.group(1))
            record["vtcm_budget_bytes"] = int(budget.group(1))
        elif block_m:
            raise SystemExit(f"{functions[0]}: attributed record has no VTCM facts")
        found.append(record)
    return found


def main():
    records = _records(sys.stdin.read())
    if not records:
        raise SystemExit("the pipeline printed no HMX manifest")

    by_function = {}
    for record in records:
        messages = UTILS_MODULE.hmx_tile_notice({"matmuls": [record]})
        by_function.setdefault(record["function"], []).extend(messages)

    # The blocked contraction: the notice speaks, and says the manifest's
    # numbers. 4096x4096x64 on the 8 MiB pool -> a 16-tile block, 8 spans.
    blocked = by_function.get("one_dot_for_a_4096x4096_output", [])
    if len(blocked) != 1:
        raise SystemExit(f"expected one tile notice, got {blocked}")
    message = blocked[0]
    for expected in (
        "4096x4096x64",
        "walked in 8 HMX span(s) of 512 rows",
        "one span holds 4784128 bytes",
        "all 4096 rows at once would need 34603008",
        "8388608 bytes the VTCM pool leaves",
        "524288-byte weight",
        "N fits one span up to 928 at this K",
        "80.9-100.3 us",
        "tiled-k-matmul-support-2026-09-30.md",
        "iters=300",
    ):
        if expected not in message:
            raise SystemExit(f"tile notice is missing {expected!r}: {message}")

    # The whole-block contraction: one span, and therefore no notice at all.
    if by_function.get("whole_block"):
        raise SystemExit(f"a single-span kernel produced a notice: {by_function}")

    # The budget refusal: no bridge plan is published, so the tile notice stays
    # silent and the refusal reporter owns that site.
    if by_function.get("weight_alone_fills_the_pool"):
        raise SystemExit(
            f"a refused contraction produced a tile notice: {by_function}"
        )

    print("hmx tile notice: PASS (1 blocked site reported, 2 sites silent)")


if __name__ == "__main__":
    main()
