#!/usr/bin/env python3
"""An f32 weight source packs to the same fp16 crouton the device pack writes.

The compiler's ``hmx.weight_prepack`` contract may name an f32 source (``dtype``
``"f32"``): the host then has to quantise before permuting. The contract's
promise is that the resident bytes equal what the device-side f32 pack leaf
would have written -- ``hmx__f32_pair_to_block`` (a qf32 convert followed by
``Q6_Vhf_equals_Wqf32``, an IEEE round-to-nearest conversion) plus the same
block permutation as the f16 pack. This file pins the host half of that
agreement: the round-to-nearest cast and the permutation are checked element by
element against the compiler's own coefficient map (the map ``_verify_layout``
also uses as its canary), the image is the crouton's byte size and not the
argument's, and a source dtype the contract does not name is refused instead of
silently reinterpreted.

The device half -- that the leaf's conversion is the same round-to-nearest the
host applies -- is a device measurement, not a host one; it is covered by the
f32 A/B that compares host-prepacked and device-packed outputs bit for bit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    HexagonLauncherBase,
)

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
_SPEC = importlib.util.spec_from_file_location(
    "hmx_weight_prepack_f32", _BACKEND / "hmx_weight_prepack.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
WeightPrepack = _MOD.WeightPrepack

#: The compiler's own coefficient map (`prepackLayoutJson`): physical
#: (n_tile, k_tile, j, c, h) maps to logical
#: (32*k_tile + 2*j + h, 32*n_tile + c). The same values the pass publishes.
LAYOUT = {"ndims": 5, "results": [[[1, 32], [2, 2], [4, 1]], [[0, 32], [3, 1]]]}
_NT, _KT, _J, _C, _H = 2, 2, 16, 32, 2
_SHAPE = (_KT * 32, _NT * 32)
_CROUTON = (_NT, _KT, _J, _C, _H)


def _contract(dtype: str) -> str:
    return json.dumps(
        {
            "layout": LAYOUT,
            "weights": [
                {
                    "func": "k",
                    "slot": 1,
                    "shape": list(_SHAPE),
                    "crouton": list(_CROUTON),
                    "dtype": dtype,
                }
            ],
        }
    )


def _reference_image(arr: np.ndarray) -> bytes:
    """The leaf's sequence, then the compiler map, element by element.

    Deliberately written as the slow obvious form: the packer's fast path has
    to agree with *this*, so the reference cannot share code with it. An f32
    source goes through the leaf's `+0.0` add first (hexagon-sim:
    logs/f32-prepack-2026-09-29/sim_rounding_check.py), which is exact except
    that it canonicalises -0.0 to +0.0.
    """
    if arr.dtype == np.float32:
        arr = arr + np.float32(0.0)
    quant = arr.astype(np.float16)
    assert quant.shape == _SHAPE
    bits = quant.view(np.uint16)
    out = np.empty(_CROUTON, dtype=np.uint16)
    for phys in np.ndindex(*out.shape):
        row = 32 * phys[1] + 2 * phys[2] + phys[4]
        col = 32 * phys[0] + phys[3]
        out[phys] = bits[row, col]
    return out.tobytes()


def _patterned_f32() -> np.ndarray:
    """Values that stress the cast: ties, subnormals, overflow and negatives.

    Built from raw bit patterns so the halfway cases are exact: a tie is a
    mantissa whose discarded 13 bits are 0b1000000000000, and only an exact
    construction guarantees those appear.
    """
    rng = np.random.default_rng(20260929)
    arr = rng.standard_normal(_SHAPE).astype(np.float32)
    flat = arr.reshape(-1)
    ties = [
        0x3F800000 | 0x00001000,  # 1.0 + 2**-11: halfway, even low bit
        0x3F800000 | 0x00003000,  # halfway, odd low bit
        0x3F801000 | 0x00001000,  # halfway inside a normal mantissa
        0x00000001,               # smallest positive subnormal f32
        0x000003FF,               # subnormal, discarded bits all set
        0x00000400,               # halfway subnormal tie
        0x477FE000,               # largest finite f16 (65504)
        0xBF800000 | 0x00001000,  # negative tie
        0x80000000,               # negative zero
        0x7F800000,               # +inf (bit-equal, asserted by the pack test)
        0xFF800000,               # -inf
    ]
    for i, bits in enumerate(ties):
        flat[i] = np.array(bits, dtype=np.uint32).view(np.float32)
    return arr


class F32PrepackTest(unittest.TestCase):
    def test_f32_source_packs_to_the_quantised_crouton(self):
        arr = _patterned_f32()
        prepack = WeightPrepack.from_metadata(_contract("f32"))
        packed = prepack.pack(torch.from_numpy(arr), 1)
        self.assertIsNotNone(packed)
        # The image is the fp16 crouton: half the argument's byte count.
        self.assertEqual(len(packed), arr.size * 2)
        self.assertEqual(len(packed), arr.nbytes // 2)
        self.assertEqual(packed, _reference_image(arr))

    def test_f16_source_is_still_the_bit_permutation(self):
        arr = _patterned_f32().astype(np.float16)
        prepack = WeightPrepack.from_metadata(_contract("f16"))
        packed = prepack.pack(torch.from_numpy(arr), 1)
        self.assertIsNotNone(packed)
        self.assertEqual(len(packed), arr.size * 2)
        self.assertEqual(packed, _reference_image(arr))

    def test_a_source_the_contract_does_not_name_is_refused(self):
        # Both directions: the contract's dtype is the source's element type,
        # and a mismatch is an error rather than a reinterpretation.
        f32_contract = WeightPrepack.from_metadata(_contract("f32"))
        with self.assertRaisesRegex(ValueError, "expected float32"):
            f32_contract.pack(
                torch.zeros(_SHAPE, dtype=torch.float16), 1
            )
        f16_contract = WeightPrepack.from_metadata(_contract("f16"))
        with self.assertRaisesRegex(ValueError, "expected float16"):
            f16_contract.pack(torch.zeros(_SHAPE, dtype=torch.float32), 1)
        with self.assertRaisesRegex(ValueError, "expected float32"):
            f32_contract.pack(
                torch.zeros(_SHAPE, dtype=torch.float64), 1
            )

    def test_negative_zero_is_canonicalised_like_the_leaf(self):
        # The leaf converts `v + 0.0`; IEEE (-0) + (+0) = +0, so the device
        # image never holds a negative zero. The host must not either, or its
        # bytes would differ from the device pack's on a -0.0 weight.
        arr = np.zeros(_SHAPE, dtype=np.float32)
        arr[0, 0] = np.float32(-0.0)
        prepack = WeightPrepack.from_metadata(_contract("f32"))
        packed = prepack.pack(torch.from_numpy(arr), 1)
        self.assertEqual(packed, _reference_image(arr))
        # Source (0, 0) maps to crouton lane (0, 0, 0, 0, 0), i.e. byte 0.
        self.assertEqual(int(np.frombuffer(packed, dtype=np.uint16)[0]), 0x0000)

    def test_nan_inputs_are_nan_and_the_payload_is_numpys(self):
        # The one documented bit-level divergence from the device leaf: the
        # leaf canonicalises every NaN to its sign with an all-ones payload
        # (hexagon-sim: 0x7fc00000 -> 0x7fff, 0xffc00000 -> 0xffff, and both
        # sNaN forms likewise), while this packer keeps numpy's payload
        # (0x7e00 / 0xfe00). A NaN weight is a NaN either way, so the contract
        # asserted here is that NaN-ness survives and that the host image
        # stays the host's own reproducible bytes rather than a half-copied
        # device convention. The device half lives in
        # logs/f32-prepack-2026-09-29/sim_rounding_check.py.
        arr = _patterned_f32()
        for i, bits in enumerate(
            (0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001)
        ):
            arr.reshape(-1)[i] = np.array(bits, dtype=np.uint32).view(np.float32)
        prepack = WeightPrepack.from_metadata(_contract("f32"))
        packed = prepack.pack(torch.from_numpy(arr), 1)
        self.assertEqual(packed, _reference_image(arr))
        lanes = np.frombuffer(packed, dtype=np.uint16)
        # NaN, not inf: exponent all ones *and* a non-zero payload.
        nan_lanes = ((lanes & 0x7C00) == 0x7C00) & ((lanes & 0x03FF) != 0)
        self.assertEqual(int(nan_lanes.sum()), 4)
        # numpy's payload is the quiet bit alone (0x0200); the device leaf
        # writes all ones (0x03FF). The sign bit is the input's in both.
        self.assertTrue(np.all((lanes[nan_lanes] & 0x03FF) == 0x0200))

    def test_the_f32_image_is_content_addressed(self):
        prepack = WeightPrepack.from_metadata(_contract("f32"))
        arr = _patterned_f32()
        first = prepack.pack(torch.from_numpy(arr), 1)
        self.assertEqual(prepack.pack(torch.from_numpy(arr.copy()), 1), first)
        changed = arr.copy()
        changed[0, 0] = np.float32(1234.5678)
        self.assertNotEqual(prepack.pack(torch.from_numpy(changed), 1), first)


class _FakeWrapper:
    """The launcher reads exactly these attributes off the wrapper generator."""

    def __init__(self, profs, prepack):
        self.input_profs = profs
        self.weight_prepack = prepack


class LauncherArgumentImageTest(unittest.TestCase):
    """The `.raw` file is the argument's byte image, whatever the source dtype.

    The wrapper loads `elems * sizeof(T)` bytes from that file, so an f32
    source's half-size fp16 crouton image has to be padded to the argument's
    own byte count. What the kernel reads is the image's prefix (the resident
    copy takes the crouton's bytes); the pad is never read.
    """

    def _written_image(self, tensor, contract_json):
        prepack = WeightPrepack.from_metadata(contract_json)
        prof = SimpleNamespace(
            idx=1, input_id=0, input_type="tensor", value=tensor, rank=2
        )
        launcher = HexagonLauncherBase()
        launcher.get_output_tensor_path_count = lambda wrapper: 0
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = launcher.generate_input_output_paths(
                directory, "k", _FakeWrapper([prof], prepack)
            )
            return Path(paths[0]).read_bytes()

    def test_f32_image_is_padded_to_the_argument_bytes(self):
        arr = _patterned_f32()
        image = self._written_image(torch.from_numpy(arr), _contract("f32"))
        self.assertEqual(len(image), arr.nbytes)
        self.assertEqual(image[: arr.size * 2], _reference_image(arr))
        self.assertEqual(image[arr.size * 2:], b"\0" * (arr.size * 2))

    def test_f16_image_is_written_exactly(self):
        arr = _patterned_f32().astype(np.float16)
        image = self._written_image(torch.from_numpy(arr), _contract("f16"))
        self.assertEqual(len(image), arr.nbytes)
        self.assertEqual(image, _reference_image(arr))

    def test_a_short_image_is_refused_not_padded(self):
        # The image length is exact (f16: the argument's bytes; f32: half), so
        # a packer or contract defect must fail loudly instead of being
        # zero-padded into a plausible-looking image.
        arr = _patterned_f32()
        prepack = WeightPrepack.from_metadata(_contract("f32"))
        prepack.pack = lambda tensor, slot: b"\0" * 16
        prof = SimpleNamespace(
            idx=1, input_id=0, input_type="tensor",
            value=torch.from_numpy(arr), rank=2,
        )
        launcher = HexagonLauncherBase()
        launcher.get_output_tensor_path_count = lambda wrapper: 0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "expected the .* crouton image"):
                launcher.generate_input_output_paths(
                    directory, "k", _FakeWrapper([prof], prepack)
                )


if __name__ == "__main__":
    unittest.main()
