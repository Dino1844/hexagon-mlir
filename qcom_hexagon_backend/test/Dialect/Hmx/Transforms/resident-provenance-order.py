#!/usr/bin/env python3
"""Check that resident site identity does not depend on allocation order.

The runtime key is the production compatibility key (function symbol plus
allocation order) and may legitimately differ between orderings; the
order-independent contract lives on the provenance identity
(site_id/function_id).  Runtime keys only have to stay unique within the
function and carry the workspace tag bit.
"""

import re
import sys


def main() -> None:
    chunks = sys.stdin.read().split("// -----")
    maps = []
    for chunk in chunks:
        sites: dict[str, dict[str, str]] = {}
        for match in re.finditer(
            r'key = (-?\d+) : i64.*site = "file:order:(\d+):1".*'
            r'site_id = (-?\d+) : i64',
            chunk,
        ):
            sites.setdefault(match.group(2), {})["site_id"] = match.group(3)
            # Every workspace key carries the tag bit (1 << 63), so it prints
            # negative as i64; that bit namespace separates workspace keys
            # from weight-resident keys.
            assert int(match.group(1)) < 0
        function_ids = set(re.findall(r'function_id = (-?\d+) : i64', chunk))
        assert len(sites) == 2 and len(function_ids) == 1, (sites, function_ids)
        # The resident map is keyed by the runtime key, so the two sites in a
        # function must hold two distinct keys.  (?<![a-z_]) keeps this from
        # matching a hypothetical `identity_key = ...` field.
        keys = set(
            re.findall(r'(?<![a-z_])key = (-?\d+) : i64', chunk)
        )
        assert len(keys) == 2, keys
        for site, identity in sites.items():
            identity["function_id"] = next(iter(function_ids))
        maps.append(sites)
    # Reordering the same allocations must not change either site's identity.
    # The runtime key is allowed to track the order; the identity is not.
    assert len(maps) == 2 and maps[0] == maps[1], maps


if __name__ == "__main__":
    main()
