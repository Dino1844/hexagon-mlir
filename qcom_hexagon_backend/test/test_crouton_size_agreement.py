#!/usr/bin/env python3
"""Source contract: the 2 KiB crouton quantum is one number, not four spellings.

The crouton byte size is named once in `HmxCroutonLayout.h`, but three other
translation units re-spell the same quantum for their own reasons
(`HexagonMemOps` as an allocation alignment, `HexagonMemToLLVMPass` as the
default crouton size, the VTCM pool as its max alignment). They agree today;
nothing tied them. This binds them, without forcing HexagonMem to depend on the
Hmx dialect header.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent


def read(path: str) -> str:
    return (BACKEND / path).read_text(encoding="utf-8")


def cpp_int_const(source: str, name: str) -> int:
    match = re.search(rf"constexpr\s+\w+\s+{name}\s*=\s*([0-9]+)\s*;", source)
    assert match, f"constexpr {name} not found (renamed?)"
    return int(match.group(1))


def crouton_bytes() -> int:
    source = read("include/hexagon/Dialect/Hmx/IR/HmxCroutonLayout.h")
    raw = dict(re.findall(r"constexpr\s+int64_t\s+(\w+)\s*=\s*([^;]+);", source))
    assert raw, "no layout constants parsed"
    resolved: dict[str, int] = {}
    for _ in range(len(raw) + 1):
        for name, expr in raw.items():
            if name in resolved:
                continue
            try:
                resolved[name] = int(eval(expr, {"__builtins__": {}}, resolved))  # noqa: S307
            except NameError:
                pass
    assert "kCroutonBytes" in resolved, sorted(raw)
    return resolved["kCroutonBytes"]


def main() -> None:
    expected = crouton_bytes()
    assert expected == 2048, f"the layout header's crouton is not 2 KiB: {expected}"

    sites = {
        "HexagonMemOps::kMaxAllocationAlignment": cpp_int_const(
            read("lib/Dialect/HexagonMem/IR/HexagonMemOps.cpp"),
            "kMaxAllocationAlignment",
        ),
        "HexagonMemToLLVM::DEFAULT_CROUTON_SIZE": cpp_int_const(
            read("lib/Conversion/HexagonMemToLLVM/HexagonMemToLLVMPass.cpp"),
            "DEFAULT_CROUTON_SIZE",
        ),
        "VTCMPool::kMaxAlignment": cpp_int_const(
            read("bin/runtime/src/VTCMPool.cpp"), "kMaxAlignment"
        ),
    }
    for name, value in sites.items():
        assert value == expected, (
            f"{name}={value} disagrees with the crouton's {expected} "
            "(HmxCroutonLayout.h)"
        )

    print(f"crouton size agreement: PASS ({len(sites)} mirrors == {expected})")


def test_crouton_size_mirrors_agree() -> None:
    main()


if __name__ == "__main__":
    main()
