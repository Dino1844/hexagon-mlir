#!/usr/bin/env python3
"""Source contract: the HMX leaf ABI is one physical shape, not two.

The engine's tile is spelled twice because it crosses two toolchains: the
runtime leaves (`bin/runtime/hmx/`) are built by the SDK clang, the compiler
constants live in `HmxCroutonLayout.h`, and the two cannot include each other
(C vs C++ namespaces, and the `.a`/`.so` build split). Until now nothing tied
them together, so a one-sided edit would silently walk the wrong stride.

This test reads both sides from source and asserts they agree. It is
deliberately a *literal* comparison against the compiler's single home, not a
re-derivation, so a rename or a coordinated wrong value fails loudly (the parse
asserts the names are present at all; a rename cannot pass vacuously).
"""
from pathlib import Path
import re

RUNTIME = Path(__file__).resolve().parents[1]
CROUTON_LAYOUT = RUNTIME / "../../include/hexagon/Dialect/Hmx/IR/HmxCroutonLayout.h"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def c_macro(source: str, name: str) -> int:
    match = re.search(rf"^#define\s+{name}\s+([0-9]+)u?\b", source, re.M)
    assert match, f"#define {name} not found (renamed?)"
    return int(match.group(1))


def cpp_constants(source: str) -> dict[str, int]:
    raw = dict(re.findall(r"constexpr\s+int64_t\s+(\w+)\s*=\s*([^;]+);", source))
    assert raw, "no constexpr int64_t constants found in the layout header"
    resolved: dict[str, int] = {}
    for _ in range(len(raw) + 1):
        for name, expr in raw.items():
            if name in resolved:
                continue
            try:
                resolved[name] = int(eval(expr, {"__builtins__": {}}, resolved))  # noqa: S307
            except NameError:
                pass
    unresolved = set(raw) - set(resolved)
    assert not unresolved, f"unresolved constants: {sorted(unresolved)}"
    return resolved


def main() -> None:
    constants = cpp_constants(read(CROUTON_LAYOUT))
    api = read(RUNTIME / "hmx/include/HMXAPI.h")

    # The compiler's single home must be the physical shape it claims to be, so a
    # coordinated-but-wrong pair cannot pass by agreeing with itself.
    assert (
        constants["kTileEdge"],
        constants["kCroutonPair"],
        constants["kCroutonCol"],
        constants["kCroutonHalf"],
    ) == (32, 16, 32, 2), constants
    assert constants["kCroutonBytes"] == 2048, constants
    assert constants["kConvStateBytes"] == 256, constants

    pairs = (
        ("HMX_TILE_BYTES", c_macro(api, "HMX_TILE_BYTES"), constants["kCroutonBytes"]),
        ("HMX_TILE_ROWS", c_macro(api, "HMX_TILE_ROWS"), constants["kTileEdge"]),
        ("HMX_TILE_COLS", c_macro(api, "HMX_TILE_COLS"), constants["kTileEdge"]),
        ("HMX_BIAS_BYTES", c_macro(api, "HMX_BIAS_BYTES"), constants["kConvStateBytes"]),
        # one 128 B scatter block = one crouton byte count / its row pairs
        (
            "HMX_BLOCK_BYTES",
            c_macro(api, "HMX_BLOCK_BYTES"),
            constants["kCroutonBytes"] // constants["kCroutonPair"],
        ),
    )
    for name, runtime_value, compiler_value in pairs:
        assert runtime_value == compiler_value, (
            f"{name}={runtime_value} disagrees with the compiler's "
            f"{compiler_value} (HmxCroutonLayout.h)"
        )

    print("HMX leaf ABI source contract: PASS")


def test_hmx_leaf_abi_matches_the_compiler() -> None:
    main()


if __name__ == "__main__":
    main()
