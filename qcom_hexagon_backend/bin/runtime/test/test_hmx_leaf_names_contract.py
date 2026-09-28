#!/usr/bin/env python3
"""Source contract: the emitted HMX leaf names are the runtime's declarations.

`HmxExternalFnNames.cpp` turns an hmx op into a runtime call by name; the
runtime implements those names in `bin/runtime/hmx/include/HMXAPI.h`. The two
sides cannot share a header (the compiler is C++, the leaves are compiled as C
by the SDK clang), and only the device link ever noticed a drift -- a renamed
declaration is a silent rot until someone runs a kernel. This binds them.
"""
from pathlib import Path
import re

RUNTIME = Path(__file__).resolve().parents[1]
EMITTER = RUNTIME / "../../lib/Conversion/HmxToLLVM/HmxExternalFnNames.cpp"
DECLARATIONS = RUNTIME / "hmx/include/HMXAPI.h"


def main() -> None:
    emitted = set(
        re.findall(r'"(hmx_[a-z0-9_]+)"', EMITTER.read_text(encoding="utf-8"))
    )
    declared = set(
        re.findall(
            r"^void\s+(hmx_[a-z0-9_]+)\s*\(",
            DECLARATIONS.read_text(encoding="utf-8"),
            re.M,
        )
    )
    # A rename that drops every literal would otherwise pass vacuously.
    assert len(emitted) >= 20, f"parsed too few emitted names: {sorted(emitted)}"
    assert len(declared) >= 20, f"parsed too few declarations: {sorted(declared)}"

    missing_declaration = sorted(emitted - declared)
    assert not missing_declaration, (
        "the compiler emits leaf names the runtime does not declare: "
        f"{missing_declaration}"
    )
    unused_declaration = sorted(declared - emitted)
    assert not unused_declaration, (
        "the runtime declares leaf names the compiler never emits (dead ABI): "
        f"{unused_declaration}"
    )

    print(f"HMX leaf name contract: PASS ({len(emitted)} names)")


def test_hmx_leaf_names_match_the_runtime() -> None:
    main()


if __name__ == "__main__":
    main()
