#!/usr/bin/env python3
"""Check that resident site keys do not depend on allocation order."""

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
            sites.setdefault(match.group(2), {})["key"] = match.group(1)
            sites[match.group(2)]["site_id"] = match.group(3)
            key = int(match.group(1))
            site_id = int(match.group(3))
            payload_mask = (1 << 63) - 1
            assert key < 0 and (key & payload_mask) == (site_id & payload_mask)
        function_ids = set(re.findall(r'function_id = (-?\d+) : i64', chunk))
        assert len(sites) == 2 and len(function_ids) == 1, (sites, function_ids)
        for site, identity in sites.items():
            identity["function_id"] = next(iter(function_ids))
        maps.append(sites)
    assert len(maps) == 2 and maps[0] == maps[1], maps


if __name__ == "__main__":
    main()
