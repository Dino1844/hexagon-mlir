from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    HexagonWrapperGenerator,
    WrapperGeneratorStrings,
    make_resident_scope_id,
)


def test_resident_scope_id_is_stable_and_object_sensitive():
    object_a = b"kernel-a"
    object_b = b"kernel-b"
    scope_a = make_resident_scope_id(object_a, "entry")
    assert scope_a == make_resident_scope_id(object_a, "entry")
    assert scope_a != make_resident_scope_id(object_b, "entry")
    assert scope_a != make_resident_scope_id(object_a, "other_entry")


def test_generated_scope_setup_uses_device_symbol():
    generator = HexagonWrapperGenerator(
        [],
        1,
        "entry",
        [],
        WrapperGeneratorStrings(),
        {"enableLWP": False},
        (0x0123456789ABCDEF, 0xFEDCBA9876543210),
    )
    setup = generator.generate_resident_scope_setup()
    assert "hexagon_runtime_resident_scope_enter_v2_dsp" in setup
    assert "__attribute__((weak))" in generator.common_strings.code_headers
    assert "!= nullptr" in setup
    assert "0x123456789abcdefULL" in setup
    assert "0xfedcba9876543210ULL" in setup
    assert "return -1" in setup
