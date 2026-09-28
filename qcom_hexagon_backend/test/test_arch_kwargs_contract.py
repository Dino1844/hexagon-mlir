#!/usr/bin/env python3
"""Source contract: every `arch_kwargs` key the C++ reads is a Python option field.

The backend serialises `HexagonOptions` to a string map (`arch_kwargs`) that the
C++ translation reads by key. The key names are hand-written on the C++ side and
hand-written as dataclass fields on the Python side; a rename on one side does
not fail to compile, it silently drops the option. This binds them.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
OPTIONS = BACKEND / "backend/hexagon_options.py"
TRANSLATION = BACKEND / "lib/Target/Linalg_MLLVMIR/MLLVMIRTranslation.cpp"


def main() -> None:
    fields = set(
        re.findall(
            r"^\s{4}([A-Za-z_]\w*)\s*:\s*(?:bool|int|str)",
            OPTIONS.read_text(encoding="utf-8"),
            re.M,
        )
    )
    # A rename that empties the parse must not pass vacuously.
    assert len(fields) >= 30, f"parsed too few option fields: {sorted(fields)}"

    source = TRANSLATION.read_text(encoding="utf-8")
    keys = set(re.findall(r'arch_kwargs\.(?:at|find)\("([^"]+)"\)', source))
    assert keys, "no arch_kwargs keys parsed from the translation"

    unknown = sorted(keys - fields)
    assert not unknown, (
        "the C++ reads arch_kwargs keys that are not HexagonOptions fields "
        f"(renamed on the Python side?): {unknown}"
    )

    # `.at()` throws when the key is absent, so those keys must be fields too.
    at_keys = set(re.findall(r'arch_kwargs\.at\("([^"]+)"\)', source))
    missing = sorted(at_keys - fields)
    assert not missing, f"required arch_kwargs keys not backed by a field: {missing}"

    print(f"arch_kwargs contract: PASS ({len(keys)} keys, {len(fields)} fields)")


def test_arch_kwargs_keys_are_option_fields() -> None:
    main()


if __name__ == "__main__":
    main()
