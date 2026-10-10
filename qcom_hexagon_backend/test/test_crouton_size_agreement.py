#!/usr/bin/env python3
"""Source contract: the crouton geometry is a few numbers, not many spellings.

The crouton byte size is named once in `HmxCroutonLayout.h`, but other
translation units re-spell the same quantum for their own reasons
(`HexagonMemOps` as an allocation alignment, the VTCM pool as its max
alignment). They agree today; nothing tied them. This binds them, without
forcing HexagonMem to depend on the Hmx dialect header.

A third mirror lived in `HexagonMemToLLVMPass` as `DEFAULT_CROUTON_SIZE`, for
the block size of a crouton allocation. The `hexagonmem` crouton conversion ops
that used it were deleted along with the crouton dialect, and with them the only
reason that constant existed, so it is gone rather than bound.

Two more geometries had the same disease, and worse: they are spelled on the
**Python host** side, where a divergence cannot fail to compile -- it just
produces wrong numbers on the device.

  * `kTileEdge` (32) vs `utils.HMX_TILE_EDGE` vs `hmx_weight_prepack._TILE`.
  * element bytes vs `HmxTarget::croutonElemBytes` vs
    `hmx_weight_prepack._CROUTON_ELEM_BYTES`.

This matters because `enableWeightResident` defaults **on**, so the host prepack
is on the default path. If the tile edge moved and only the C++ side learned
about it, the host would pack weights into the wrong shape: no compile error, no
failed assertion, silently wrong numerics.

A fourth spelling joined on 2026-10-09: `hexagon::F16_CROUTON_SHAPE`
(`include/hexagon/Common/Common.h`) is the HVX image side of the same crouton,
the block an (n, h, w, c) pack leaves innermost. It is bound to the engine
header by the one fact the two spellings share -- the block is exactly one
crouton's worth of elements -- and *not* dim by dim, for the reason above.

⚠️ **An equal number is not automatically the same fact.** The header's
`kCroutonHalf` is 2 ("two halves of a pair") and the element size is 2 ("bytes
per f16"). Binding those to each other would assert a coincidence, and would
then break for the wrong reason the day `kCroutonHalf` changes. So each mirror is
bound to the constant that actually states *its* fact.

Not covered here, and deliberately: `HmxCroutonLayout.h` still writes
`kCroutonBytes = kCroutonElements * 2` with a bare `2` for the element size.
That literal has no name to bind to, which is why the element-size check anchors
on `HmxTarget::croutonElemBytes` instead. Naming the header's literal is tracked
in ROADMAP; until then this file cannot assert the header and the host agree,
only that the two *named* copies do.

## The three Python copies are load-bearing. Do not "remove the duplication".

An investigation on 2026-09-30 concluded that two of the three Python copies
should be deleted, on the theory that the host receives the geometry from the
`hmx.weight_prepack` contract and should therefore read it from there. **Both
deletions would have been bugs**, and the reason is the same in each case: the
contract is what the host is *validating*, so a value read back out of it checks
nothing.

  * `utils.HMX_TILE_EDGE` drives an **independent recomputation** of the padded /
    full / tail split from the logical shape (`utils.py:645-674`). Its entire
    value is being independent of the compiler. Read the edge from the record and
    the check becomes a tautology that can never fail.
  * `hmx_weight_prepack._TILE` is not a size choice. It gates
    `rows != t1 * _TILE` (`hmx_weight_prepack.py:190`), i.e. it asserts the
    weight's logical shape is an exact number of tiles -- again a statement the
    host makes *about* the compiler's output, not a value copied from it. (The
    canary at `:227` does use it as a size, but that is not the only use.)
  * `hmx_weight_prepack._CROUTON_ELEM_BYTES` must **not** be derived from
    `desc["dtype"]`. That field is the *source* dtype (`"f16"` / `"f32"`), while
    the crouton element is **always** f16 (`dtype::croutonElementType`,
    `HmxAttrs.cpp:332`). For an f32 source, deriving from `desc["dtype"]` yields
    4 bytes per element and a buffer twice the correct size.

So the duplication is not a leak to be closed; it is a second, independent
statement of the same fact, and `test_host_copies_stay_independent` below is what
stops a future reader from tidying it away.
"""
import ast
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


def cpp_uint_vector(source: str, name: str) -> list[int]:
    """An `inline const SmallVector<unsigned> NAME = {...};` initializer."""
    match = re.search(
        rf"inline\s+const\s+SmallVector<unsigned>\s+{name}\s*=\s*\{{([^}}]*)\}}",
        source,
    )
    assert match, f"SmallVector {name} not found (renamed?)"
    return [int(value) for value in re.findall(r"\d+", match.group(1))]


def py_int_const(path: str, name: str) -> int:
    """A module-level `NAME = <int>` in a Python source file."""
    match = re.search(
        rf"^{re.escape(name)}\s*(?::[^=]+)?=\s*([0-9]+)\s*$", read(path), re.M
    )
    assert match, f"{name} = <int> not found in {path} (renamed?)"
    return int(match.group(1))


def layout_header() -> str:
    return read("include/hexagon/Dialect/Hmx/IR/HmxCroutonLayout.h")


def layout_consts() -> dict[str, int]:
    """Every `k*` constant in the layout header, expressions resolved."""
    source = layout_header()
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
    return resolved


def crouton_bytes() -> int:
    resolved = layout_consts()
    assert "kCroutonBytes" in resolved, sorted(resolved)
    return resolved["kCroutonBytes"]


def main() -> None:
    expected = crouton_bytes()
    assert expected == 2048, f"the layout header's crouton is not 2 KiB: {expected}"

    sites = {
        "HexagonMemOps::kMaxAllocationAlignment": cpp_int_const(
            read("lib/Dialect/HexagonMem/IR/HexagonMemOps.cpp"),
            "kMaxAllocationAlignment",
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

    # --- the tile edge, including the Python host -------------------------
    edge = cpp_int_const(layout_header(), "kTileEdge")
    edge_sites = {
        "utils.HMX_TILE_EDGE": py_int_const("backend/utils.py", "HMX_TILE_EDGE"),
        "hmx_weight_prepack._TILE": py_int_const(
            "backend/hmx_weight_prepack.py", "_TILE"
        ),
    }
    for name, value in edge_sites.items():
        assert value == edge, (
            f"{name}={value} disagrees with kTileEdge={edge} (HmxCroutonLayout.h). "
            "The host prepack would build weights in the wrong shape: no compile error, "
            "just wrong numbers on device."
        )

    # --- element bytes ---------------------------------------------------
    # NOT kCroutonHalf. That is "halves of a pair" and happens to be 2; the
    # element size is "bytes per f16" and is also 2. See the module docstring.
    elem_bytes = cpp_int_const(
        read("include/hexagon/Dialect/Hmx/Transforms/HmxTarget.h"), "croutonElemBytes"
    )
    elem_sites = {
        "hmx_weight_prepack._CROUTON_ELEM_BYTES": py_int_const(
            "backend/hmx_weight_prepack.py", "_CROUTON_ELEM_BYTES"
        ),
    }
    for name, value in elem_sites.items():
        assert value == elem_bytes, (
            f"{name}={value} disagrees with HmxTarget::croutonElemBytes={elem_bytes}. "
            "The host would size its crouton buffers from a different element size than "
            "the compiler used."
        )

    # --- the HVX image-side crouton block -----------------------------------
    # A fourth spelling of the same 2 KiB f16 crouton, from the HVX image side:
    # the block an (n, h, w, c) pack leaves innermost is 8 x 2 x 32 x 2
    # (`PackUnpackUtils.cpp` / `SeedLayoutConversions.cpp`). The engine header
    # calls itself "the one home for the engine's leaf physical constants" but
    # never mentions this table, so nothing tied the two decompositions of one
    # crouton together. Only the fact both spellings share is bound: the block
    # is exactly one crouton's worth of elements. The dim-by-dim correspondence
    # is deliberately *not* asserted -- the two tables decompose the same 1024
    # elements differently, which is the "equal number is not the same fact"
    # case from the module docstring.
    layout = layout_consts()
    hvx_block = cpp_uint_vector(
        read("include/hexagon/Common/Common.h"), "F16_CROUTON_SHAPE"
    )
    assert len(hvx_block) == 4, (
        f"F16_CROUTON_SHAPE is no longer the 4-d image block: {hvx_block}"
    )
    block_elements = 1
    for extent in hvx_block:
        block_elements *= extent
    hvx_sites = {
        "hexagon::F16_CROUTON_SHAPE (one HVX image block)": block_elements,
    }
    for name, value in hvx_sites.items():
        assert value == layout["kCroutonElements"], (
            f"{name}={value} covers a different number of f16 elements than "
            f"one crouton (kCroutonElements={layout['kCroutonElements']}, "
            "HmxCroutonLayout.h). The HVX image block and the engine crouton "
            "would no longer be the same 2 KiB: no compile error, wrong "
            "geometry on device."
        )

    total = len(sites) + len(edge_sites) + len(elem_sites) + len(hvx_sites)
    print(
        f"crouton geometry agreement: PASS ({total} mirrors; "
        f"crouton={expected} B, tile edge={edge}, elem bytes={elem_bytes}, "
        f"hvx block={block_elements} elems)"
    )


def test_crouton_size_mirrors_agree() -> None:
    main()


# The host copies, with the reason each one may not be derived from the contract.
HOST_COPIES = {
    "backend/utils.py": {
        "HMX_TILE_EDGE": "drives an independent recomputation of padded/full/tail",
    },
    "backend/hmx_weight_prepack.py": {
        "_TILE": "gates `rows != t1 * _TILE`, a claim about the compiler's output",
        "_CROUTON_ELEM_BYTES": "the crouton element is always f16, whatever the source",
    },
}


def test_host_copies_stay_independent() -> None:
    """Each host copy must be stated as a literal, not read back from the contract.

    This is the anti-"cleanup" gate. The host validates the `hmx.weight_prepack`
    contract, so a geometry value taken *from* that contract cannot disagree with
    it -- the check would be a tautology. The value has to be the host's own
    statement of what it believes the compiler does; the binding checks above are
    what make believing it dangerous-but-correct today.

    Each entry is required to be an integer literal for exactly this reason. The
    `_CROUTON_ELEM_BYTES` case is the sharp one: `desc["dtype"]` is the *source*
    dtype, and an f32 source must still pack to a 2-byte-element crouton.
    """
    offenders = []
    for path, names in HOST_COPIES.items():
        source = read(path)
        for name in names.items():
            key, why = name
            match = re.search(
                rf"^{re.escape(key)}\s*(?::[^=]+)?=\s*(.+?)\s*(?:#.*)?$", source, re.M
            )
            if not match:
                offenders.append(f"{path}: {key} is gone ({why})")
                continue
            rhs = match.group(1)
            try:
                value = ast.literal_eval(rhs)
            except (ValueError, SyntaxError):
                offenders.append(
                    f"{path}: {key} = {rhs!r} is no longer a plain integer literal.\n"
                    f"      It must stay one: {why}.\n"
                    f"      Reading it from the contract it is validating makes the check "
                    f"a tautology, which is how a compiler/host divergence goes silent."
                )
                continue
            if not isinstance(value, int):
                offenders.append(f"{path}: {key} = {value!r} is not an int ({why})")
    assert not offenders, (
        "a host geometry copy stopped being an independent statement:\n  "
        + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    main()
