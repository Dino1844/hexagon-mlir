# ===- weight_prepack.py ----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
"""Host-side pre-pack for resident HMX weights (P2).

The compiler publishes a contract in the module attribute
``hmx.weight_prepack`` (a JSON object, handed to the launcher through the kernel
metadata). It names, for each runtime weight it made resident, the function
argument *slot*, the logical shape and the crouton shape, plus the crouton
permutation as an affine coefficient map.

When the contract is present, the launcher must hand the kernel that argument
already in crouton order: the kernel no longer emits ``hmx.pack_weight`` (it
copies the argument into VTCM once and pins it). This module is the host half of
that contract. It packs the permutation from the compiler's own coefficient map,
and it verifies its vectorised fast path against that map on a small canary, so
a future change to the compiler layout fails loudly instead of silently
corrupting the weight.

The image is always the engine's fp16 crouton. An f16 source is permuted; an
f32 source is first quantised to fp16 with the same conversion the device-side
f32 pack leaf runs (``hmx__f32_pair_to_block``: an exact ``+0.0`` in f32, then
round-to-nearest-even to fp16). For every non-NaN input the resident bytes are
therefore exactly what that pack would have written, including the ``-0.0``
canonicalisation the leaf's ``+0.0`` performs. NaN payloads are the one
exception: the leaf canonicalises a NaN to its sign with an all-ones payload
(observed on hexagon-sim, all four of qNaN/sNaN x +/-), while this packer keeps
numpy's (quiet bit alone). A NaN weight is a NaN either way, so the images are
numerically equivalent, not bit-equal. The contract's ``dtype`` names the
source, and ``pack`` refuses a tensor whose dtype does not match it.

Identity/caching: packing is keyed by ``(slot, shape, dtype, content hash)`` --
a content-addressed cache, never a tensor `data_ptr` cache -- so repeated
launches of the same weight reuse one packed image. The host contract accepts
one function per module; multi-function resident contracts are rejected before
launch rather than guessing which function a slot belongs to.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional, Tuple

import numpy as np

from triton.backends.qcom_hexagon_backend.utils import validate_weight_prepack

_TILE = 32
#: The crouton image's element is the engine's fp16, whatever the source dtype
#: is (a wider source is quantised by the pack).
_CROUTON_ELEM_BYTES = 2


class WeightPrepack:
    """Parsed ``hmx.weight_prepack`` contract plus a content-addressed cache."""

    def __init__(self, layout: dict, weights: list):
        contract = {"layout": layout, "weights": weights}
        validate_weight_prepack(contract)
        self._by_slot: Dict[int, dict] = {}
        for w in weights:
            self._by_slot[w["slot"]] = w
        self._function_name = weights[0]["func"] if weights else None
        self._layout = layout
        self._layout_verified = False
        self._cache: Dict[Tuple, bytes] = {}

    @classmethod
    def from_metadata(
        cls, weight_prepack_json: str
    ) -> Optional["WeightPrepack"]:
        """Parse a contract; only a valid empty contract means no prepacking."""
        if not isinstance(weight_prepack_json, str):
            raise ValueError(
                "weight_prepack must be a JSON string, got "
                f"{type(weight_prepack_json).__name__}"
            )
        if weight_prepack_json == "":
            raise ValueError("weight_prepack must contain a JSON object")
        try:
            doc = json.loads(weight_prepack_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"weight_prepack is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise ValueError(
                "weight_prepack must decode to an object, got "
                f"{type(doc).__name__}"
            )
        validate_weight_prepack(doc)
        if not doc["weights"]:
            return None
        return cls(doc["layout"], doc["weights"])

    @property
    def function_name(self) -> Optional[str]:
        return self._function_name

    def has_slot(self, slot: int) -> bool:
        return slot in self._by_slot

    def unconsumed_slots(self, consumed: set[int]) -> set[int]:
        return set(self._by_slot) - set(consumed)

    def image_bytes(self, slot: int) -> Optional[int]:
        """The crouton image's byte count for a contract slot.

        ``None`` means that the contract has no such slot. The image is the
        engine's fp16 crouton, so this is exact and independent of the source
        dtype: an f16 source packs to the argument's own byte count, an f32
        source to half of it. The launcher asserts ``pack`` against this
        instead of padding whatever it is handed.
        """
        desc = self._by_slot.get(slot)
        if desc is None:
            return None
        return int(np.prod(desc["crouton"], dtype=np.int64)) * _CROUTON_ELEM_BYTES

    def pack(self, tensor, slot: int) -> Optional[bytes]:
        """Return the fp16 crouton image for a contract slot.

        ``None`` means that the contract has no such slot.  Once a slot is
        present, a runtime shape or dtype mismatch is an error: the generated
        kernel consumes the resident image directly and has no safe raw-byte
        fallback.  An f32 source packs to an image half its byte size; the
        launcher writes it as the argument image (zero-padded to the
        argument's own byte count).
        """
        desc = self._by_slot.get(slot)
        if desc is None:
            return None
        try:
            arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                f"weight slot {slot} must be a tensor-like object"
            ) from exc
        shape = tuple(desc["shape"])
        if tuple(arr.shape) != shape:
            raise ValueError(
                f"weight slot {slot} has shape {tuple(arr.shape)}, expected {shape}"
            )
        # The contract spells dtypes in the compiler's vocabulary ("f16"/"f32").
        expected = {"f16": np.float16, "f32": np.float32}[desc["dtype"]]
        if arr.dtype != np.dtype(expected):
            raise ValueError(
                f"weight slot {slot} has dtype {arr.dtype}, "
                f"expected {np.dtype(expected).name}"
            )
        raw = arr.tobytes()
        key = (
            slot,
            shape,
            str(arr.dtype),
            hashlib.blake2b(raw, digest_size=16).hexdigest(),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        packed = self._pack_crouton(arr, shape, desc)
        self._cache[key] = packed
        return packed

    @staticmethod
    def _permute(bits: np.ndarray, t0: int, t1: int, j: int, c: int,
                 h: int) -> np.ndarray:
        """Permute a row-major ``[K, N]`` f16 image into weight crouton order.

        The compiler's weight grid is ``[Nt, Kt, 16, 32, 2]`` with logical
        ``[N, K]``, so ``t0 = Nt`` and ``t1 = Kt``. Flat row-major ``[K, N]``
        reshapes to ``(Kt, j, h, Nt, c)`` -- with ``k = ((kt) * j + jj) * h +
        hh`` and ``n = nt * c + cc`` -- and transposes to crouton order
        ``(Nt, Kt, j, c, h)`` = ``(n_tile, k_tile, j, c, h)``. Shared by the
        fast pack and the canary so the two cannot drift apart.
        """
        mid = bits.reshape(t1, j, h, t0, c)
        return np.ascontiguousarray(mid.transpose(3, 0, 1, 4, 2))

    def _pack_crouton(self, arr: np.ndarray, shape, desc) -> bytes:
        # The contract validator has already enforced strict positive ints;
        # do not coerce malformed metadata back into a plausible layout here.
        t0, t1, j, c, h = desc["crouton"]
        rows, cols = shape
        if rows != t1 * _TILE or cols != t0 * _TILE:
            raise ValueError(
                f"crouton grid {desc['crouton']} does not tile shape {shape}"
            )
        if not self._layout_verified:
            self._verify_layout(t0, t1, j, c, h)
            self._layout_verified = True
        # An f32 source is quantised to the crouton's fp16 first, with the
        # device leaf's own sequence (`hmx__f32_pair_to_block`: qf32 convert +
        # Q6_Vhf_equals_Wqf32). `astype` is IEEE round-to-nearest-even and the
        # leaf's conversion agrees with it on every *non-NaN* value, with one
        # deliberate addition: the leaf's `Q6_Vqf32_vadd_VsfVsf(v, zero)`
        # canonicalises -0.0 to +0.0 (IEEE: (-0) + (+0) = +0), which this add
        # reproduces. NaN payloads differ (the leaf writes sign + all-ones;
        # numpy keeps its payload) and are not reproduced -- see the module
        # docstring. Verified on hexagon-sim:
        # logs/f32-prepack-2026-09-29/sim_rounding_check.py.
        if arr.dtype == np.dtype(np.float32):
            arr = arr + np.float32(0.0)
            arr = arr.astype(np.float16)
        # f16 bits are preserved through the permutation.
        bits = arr.view(np.uint16)
        return self._permute(bits, t0, t1, j, c, h).tobytes()

    def _verify_layout(self, t0, t1, j, c, h) -> None:
        """Check the vectorised pack against the compiler's coefficient map.

        Uses a fixed small canary on both sides, so a mismatch means the host's
        permutation and the compiler's have drifted apart -- refuse to pack
        rather than feed the engine garbage.
        """
        if not self._layout or "results" not in self._layout:
            raise RuntimeError(
                "weight prepack metadata has no layout; refusing to guess the "
                "crouton permutation"
            )
        ct0 = ct1 = 2
        rows, cols = ct1 * _TILE, ct0 * _TILE
        ramp = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
        fast = self._permute(ramp, ct0, ct1, j, c, h)
        results = self._layout["results"]
        slow = np.empty(fast.shape, dtype=np.uint16)
        for phys in np.ndindex(*fast.shape):
            r = sum(coeff * phys[d] for d, coeff in results[0])
            col = sum(coeff * phys[d] for d, coeff in results[1])
            if not (0 <= r < rows and 0 <= col < cols):
                raise RuntimeError(
                    "host crouton pack disagrees with the compiler layout "
                    "map; refusing to pre-pack the weight"
                )
            slow[phys] = ramp[r, col]
        if not np.array_equal(fast, slow):
            raise RuntimeError(
                "host crouton pack disagrees with the compiler layout map; "
                "refusing to pre-pack the weight"
            )
