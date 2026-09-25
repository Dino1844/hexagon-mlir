#!/usr/bin/env python3
"""Check that static site IDs follow source facts, not allocation order."""

import re
import sys


text = sys.stdin.read()
identity_lines = [line for line in text.splitlines() if "hmx.kernel_vtcm_identity" in line]
if len(identity_lines) != 2:
    raise SystemExit(f"expected two identity sidecars, found {len(identity_lines)}")

source_pattern = re.compile(r'source = "file:order:(10|11):1"')
site_pattern = re.compile(r"site_id = (-?[0-9]+) : i64")
function_pattern = re.compile(r"function_id = (-?[0-9]+) : i64")

seen = []
for line in identity_lines:
    sources = source_pattern.findall(line)
    site_ids = site_pattern.findall(line)
    function_ids = function_pattern.findall(line)
    if len(sources) != 2 or len(site_ids) != 2 or not function_ids:
        raise SystemExit("identity sidecar did not contain two source/site records")
    if any(int(site_id) == 0 for site_id in site_ids):
        raise SystemExit("zero was emitted as a site join key")
    if any(int(function_id) == 0 for function_id in function_ids):
        raise SystemExit("zero was emitted as a function join key")
    if len(set(function_ids)) != 1:
        raise SystemExit("function identity changed under reordering")
    # The pass sorts records by stable source facts.  Pairing in that order
    # makes this check independent of the input allocation order.
    seen.append(dict(zip(sources, site_ids)))

if seen[0] != seen[1]:
    raise SystemExit(f"source-to-site identity changed under reordering: {seen!r}")
if set(seen[0]) != {"10", "11"}:
    raise SystemExit(f"unexpected source set: {seen[0]!r}")

print("static VTCM site identity order: PASS")
