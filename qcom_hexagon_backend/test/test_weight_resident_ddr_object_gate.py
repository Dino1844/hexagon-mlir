#!/usr/bin/env python3
"""The OBJECT-level gate for DDR weight residency (host only, no device).

WHY THIS FILE EXISTS
--------------------
A resident weight's whole point is what the *binary* does not do: the
per-launch `hmx_pack_weight` bridge, whose device leaf is
`hmx_pack_weight_f16_bulk`, is supposed to be gone. Every cheaper level of
checking can say that and be wrong -- the pass erases the bridge, the manifest
recounts `pack_weight_sites` to 0, and then the rewrite is dropped somewhere
between IR and the object while the kernel keeps packing (this file's sibling,
`test_hmx_vector_readout_object_gate.py`, documents exactly that failure for
the read-out split). So this gate reads the object with `llvm-nm`.

WHICH INSTRUMENT, AND WHY NOT THE SYMBOL TABLE ALONE
----------------------------------------------------
The object links the *runtime* in, so `hexagon_runtime_weight_resident_v2_dsp`
and its DDR sibling are both DEFINED (`T`) in every object whether or not
anything calls them -- asserting their presence would pass on a kernel that
calls neither. The instrument for "the kernel calls this" is the object's
**relocation table**: a call to a local symbol becomes a relocation against it
(`R_HEX_B22_PCREL`), and the runtime never calls its own entries, so a
relocation to one is generated code calling it. The device leaves are the
opposite case: `hmx_pack_weight_f16_bulk` lives in `libhmxapi.a`, which is not
linked here, so there the ordinary symbol table (`U` when referenced, absent
when not) is already the call evidence -- and that is the assertion the plan
asked for.

WHAT IS ASSERTED, AND WHY EACH ARM EXISTS
-----------------------------------------
Three arms, one predicate (`_pack_leaf_is_in_object`) shared by all of them,
because a gate whose positive arm only proves "an object exists" passes on the
failure it exists to catch:

  * **DDR arm** (the weight does not fit the 8 MiB persistent VTCM pool):
    the mirror's runtime entry `hexagon_runtime_weight_resident_ddr_v2_dsp` is
    referenced, the VTCM entry is not, the pack leaf is ABSENT, the manifest
    says `pack_weight_sites = 0` with policy `resident-prepack-ddr`, and the
    host contract carries `location = "ddr"`. Those four must agree; a
    disagreement is a kernel that will be pre-packed for a home the compiler
    did not choose.
  * **Negative arm** (same kernel, `enableWeightResident=False`): the pack leaf
    is PRESENT. This is the non-vacuity half -- if the leaf were simply not
    linked into this toolchain, the DDR arm's absence would mean nothing. Same
    kernel, same shape, one option changed.
  * **VTCM arm** (a weight that fits): the *other* runtime entry
    `hexagon_runtime_weight_resident_v2_dsp` is referenced, the DDR entry is
    not, and the contract says `location = "vtcm"`. Placement has to be a
    decision, not a rename: a build in which every resident became a mirror
    would pass the DDR arm and still have broken the pool path.

The compile's stderr is checked for `error:` lines for the reason every other
gate in this directory does: a pass that failed hard reports it on stderr, and
the failure never reaches a return value.

WHAT IS *NOT* CLAIMED
---------------------
Presence in the object is not evidence that the mirror is correct on the
device: the DDR entry's copy, its lifetime (freed by `ReleaseResources`) and
the fetch's numerics are device questions. Those belong to the device A/B
(docs/hmx/pack-redundancy-fix-plan-2026-10-09.md §6), not to a host gate.

Run:  source tools/hexmlir/env.sh
      .venv/bin/python hexagon-mlir/qcom_hexagon_backend/test/test_weight_resident_ddr_object_gate.py
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

# The DDR arm's shape. The weight is 2048x2048 f16 = 8388608 bytes, which is
# the *whole* persistent budget, so `admitResidentVtcm`'s strict `<` refuses it
# no matter what else the kernel holds -- the refusal this arm is about cannot
# become "it fit after all" by moving another buffer. Both operand blocks are
# exactly 2**20 elements (BK*BN and BM*BK), the front end's per-block limit.
M, K, N = 512, 2048, 2048
BM, BN, BK = 512, 512, 2048

# The VTCM arm's shape: 256x256 f16 = 128 KiB of weight, four orders of
# magnitude under the budget, so it belongs in the pool and nowhere else.
SMALL = (256, 256, 256)
SMALL_TILES = (256, 256, 256)

PACK_LEAF = "hmx_pack_weight_f16_bulk"
VTCM_ENTRY = "hexagon_runtime_weight_resident_v2_dsp"
DDR_ENTRY = "hexagon_runtime_weight_resident_ddr_v2_dsp"


@triton.jit
def mm_user(A, B, C, MM: tl.constexpr, NN: tl.constexpr, KK: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Verbatim from `exp/hmx/op_bench/mm_user_shapes.py` (2026-10-09).

    Copied rather than re-expressed, and for the same reason this file's
    sibling copies its kernel: the gate has to compile the kernel the
    measurement is about. This is 2048^3-B, whose 16 device-side
    `pack_weight` sites are what the plan fixed -- and its `if NN - n0 < BN`
    tail guard is the shape that made the view matcher decline silently
    (P0-A1), so compiling it also exercises the matcher that authorises the
    residency in the first place.
    """
    for m0 in range(0, MM, BM):
        for n0 in range(0, NN, BN):
            rows = m0 + tl.arange(0, BM)
            cols = n0 + tl.arange(0, BN)
            pm = rows < MM
            pn = cols < NN
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            for k0 in range(0, KK, BK):
                ks = k0 + tl.arange(0, BK)
                pa = A + rows[:, None] * KK + ks[None, :]
                pb = B + ks[:, None] * NN + cols[None, :]
                if MM - m0 < BM:
                    a = tl.load(pa, mask=pm[:, None], other=0.0)
                else:
                    a = tl.load(pa)
                if NN - n0 < BN:
                    b = tl.load(pb, mask=pn[None, :], other=0.0)
                else:
                    b = tl.load(pb)
                acc = tl.dot(a, b, acc)
            pc = C + rows[:, None] * NN + cols[None, :]
            if MM - m0 < BM or NN - n0 < BN:
                tl.store(pc, acc.to(C.type.element_ty),
                         mask=pm[:, None] & pn[None, :])
            else:
                tl.store(pc, acc.to(C.type.element_ty))


_LLVM_NM = shutil.which("llvm-nm") or shutil.which("llvm-nm-14")


def _nm(obj):
    """Symbol table of a device object: {name: nm type letter}."""
    if _LLVM_NM is None:
        raise unittest.SkipTest("llvm-nm not found; this gate reads objects")
    with tempfile.TemporaryDirectory(prefix="hmx-ddr-gate-nm-") as tmp:
        path = Path(tmp) / "kernel.o"
        path.write_bytes(obj)
        result = subprocess.run(
            [_LLVM_NM, str(path)], capture_output=True, text=True, check=True
        )
    symbols = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            symbols[parts[-1]] = parts[-2]
    # A parse that silently produced nothing would make every "absent" assertion
    # below vacuously true, which is the failure mode this whole file is about.
    if len(symbols) < 100:
        raise AssertionError(
            f"llvm-nm returned only {len(symbols)} symbols; the gate would pass "
            "on an unreadable object"
        )
    return symbols


_LLVM_OBJDUMP = shutil.which("llvm-objdump") or shutil.which("llvm-objdump-14")


def _calls(obj):
    """Set of symbols the object's code calls, read off its relocations.

    A call to a symbol defined in the same object is a relocation against it,
    so this is the "the kernel calls X" predicate that the symbol table cannot
    give (see the note in the module docstring). The `+0xN` addendum some
    entries carry is the offset inside the callee and is not part of its name.
    """
    if _LLVM_OBJDUMP is None:
        raise unittest.SkipTest(
            "llvm-objdump not found; this gate reads the relocation table"
        )
    with tempfile.TemporaryDirectory(prefix="hmx-ddr-gate-rel-") as tmp:
        path = Path(tmp) / "kernel.o"
        path.write_bytes(obj)
        result = subprocess.run(
            [_LLVM_OBJDUMP, "-r", str(path)],
            capture_output=True, text=True, check=True,
        )
    targets = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        # `00000164 R_HEX_B22_PCREL          hexagon_runtime_weight_resident_ddr_v2_dsp`
        if len(parts) == 3 and parts[1].startswith("R_"):
            targets.add(parts[2].split("+", 1)[0])
    if not targets:
        raise AssertionError(
            "the relocation table read as empty; the gate would pass on an "
            "unreadable object"
        )
    return targets


def _compile(shape, tiles, **options):
    """Compile one kernel shape to an object, host only.

    `kernel.warmup` with `target_artifact="o"` compiles and does NOT launch:
    the difference between `warmup` and `kernel[grid](...)` is one character
    and the second one talks to the DSP. fd 2 is captured with `os.dup2`
    rather than `contextlib.redirect_stderr`, because the diagnostics come
    from C++ (llvm::errs()) inside libtriton and never reach sys.stderr.
    Returns (object bytes, packed metadata, stderr text).
    """
    m, k, n = shape
    a = torch.zeros(m, k, dtype=torch.float16)
    b = torch.zeros(k, n, dtype=torch.float16)
    c = torch.zeros(m, n, dtype=torch.float16)
    with tempfile.TemporaryDirectory(prefix="hmx-ddr-gate-") as tmp:
        previous_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp, "cache")
        captured = os.dup(2)
        path = os.path.join(tmp, "stderr")
        sink = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.dup2(sink, 2)
            compiled = mm_user.warmup(
                a, b, c, grid=(1,), target_artifact="o",
                MM=m, NN=n, KK=k, BM=tiles[0], BN=tiles[1], BK=tiles[2],
                **options,
            )
        finally:
            os.dup2(captured, 2)
            os.close(sink)
            os.close(captured)
            if previous_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous_cache
        stderr = Path(path).read_text(errors="replace")
        metadata = dict(compiled.packed_metadata or {})
    return bytes(compiled.asm["o"]), metadata, stderr


def _pack_leaf_is_in_object(symbols):
    """The one predicate every arm asserts against: the device pack is there.

    Definedness is deliberately not required: the leaf lives in the runtime's
    HMX library and the kernel object either references it (`U`) or carries it
    linked (`T`/`t`), depending on how far this build got before `nm`. What
    matters is that the *name* appears at all.
    """
    return PACK_LEAF in symbols


def _contract(metadata):
    return json.loads(metadata.get("weight_prepack") or "{}")


def _manifest(metadata):
    return json.loads(metadata.get("hmx_manifest") or "{}")


class DdrPlacementReachesTheObject(unittest.TestCase):
    """A weight the pool cannot hold: the mirror, no pack leaf, agreed facts."""

    obj = None
    metadata = None
    stderr = None

    @classmethod
    def setUpClass(cls):
        cls.obj, cls.metadata, cls.stderr = _compile((M, K, N), (BM, BN, BK))
        cls.symbols = _nm(cls.obj)

    def test_the_compile_is_clean(self):
        self.assertNotIn(
            "error:", self.stderr,
            "the compile reported an error on stderr; the object would be the "
            "product of a failed pass",
        )

    def test_the_pack_leaf_is_gone(self):
        self.assertFalse(
            _pack_leaf_is_in_object(self.symbols),
            "the resident kernel still carries the per-launch weight pack; "
            "either the bridge was not replaced or the rewrite was dropped "
            "between IR and the object",
        )

    def test_the_mirror_entry_is_called_and_the_pool_one_is_not(self):
        calls = _calls(self.obj)
        self.assertIn(
            DDR_ENTRY, calls,
            "the kernel does not call the runtime for a DDR mirror, so the "
            "placement decision did not reach the object",
        )
        self.assertNotIn(
            VTCM_ENTRY, calls,
            "a weight the pool cannot hold was still placed in the pool",
        )

    def test_the_fetch_is_present(self):
        # The mirror is only half of it: something has to move the block the
        # engine reads. Either lowering of the fetch is legal (the runtime
        # copy, or the DMA pair when `enable-hexagonmem-copy-to-dma` is on),
        # but *something* must read the mirror.
        calls = _calls(self.obj)
        self.assertTrue(
            "hexagon_runtime_copy_dsp" in calls
            or "hexagon_runtime_dma_start" in calls,
            "no copy of the mirror's block reaches the object; the engine "
            "would read a VTCM array nothing filled",
        )

    def test_the_manifest_and_the_contract_agree_on_the_placement(self):
        manifest = _manifest(self.metadata)
        self.assertEqual(manifest.get("pack_weight_sites"), 0)
        policies = {
            (p.get("function"), p.get("slot")): p.get("policy")
            for p in manifest.get("weight_policies", [])
        }
        self.assertIn(
            "resident-prepack-ddr", policies.values(),
            f"weight_policies does not name the mirror: {policies}",
        )
        weights = _contract(self.metadata).get("weights") or []
        self.assertEqual(
            [w.get("location") for w in weights], ["ddr"],
            "the host contract does not say where the image lives",
        )


class ThePackLeafArmIsNotVacuous(unittest.TestCase):
    """The same kernel with the feature off: the leaf MUST be there.

    Without this, `test_the_pack_leaf_is_gone` would pass on a toolchain that
    never linked the leaf at all, which is the shape of failure this file
    exists to refuse. Same kernel, same shape, one option.
    """

    @classmethod
    def setUpClass(cls):
        cls.obj, cls.metadata, cls.stderr = _compile(
            (M, K, N), (BM, BN, BK), enableWeightResident=False
        )
        cls.symbols = _nm(cls.obj)

    def test_the_compile_is_clean(self):
        self.assertNotIn("error:", self.stderr)

    def test_the_pack_leaf_is_present_without_residency(self):
        self.assertTrue(
            _pack_leaf_is_in_object(self.symbols),
            "the pack leaf is absent even with residency off, so the DDR "
            "arm's absence proves nothing about placement",
        )
        calls = _calls(self.obj)
        self.assertNotIn(DDR_ENTRY, calls)
        self.assertNotIn(VTCM_ENTRY, calls)


class AWeightThatFitsStaysInThePool(unittest.TestCase):
    """Placement is a decision: the small weight must take the VTCM entry.

    A change that sent *every* resident to the mirror would pass the DDR arm
    and still have broken the pool path (and the VTCM budget accounting with
    it), so the other arm of the decision is asserted with the same object
    discipline.
    """

    @classmethod
    def setUpClass(cls):
        cls.obj, cls.metadata, cls.stderr = _compile(SMALL, SMALL_TILES)
        cls.symbols = _nm(cls.obj)

    def test_the_compile_is_clean(self):
        self.assertNotIn("error:", self.stderr)

    def test_the_pool_entry_is_used_and_the_mirror_is_not(self):
        calls = _calls(self.obj)
        self.assertIn(
            VTCM_ENTRY, calls,
            "the weight that fits no longer reaches the pool's resident entry",
        )
        self.assertNotIn(
            DDR_ENTRY, calls,
            "a weight that fits the pool was placed in the DDR mirror",
        )
        self.assertFalse(_pack_leaf_is_in_object(self.symbols))

    def test_the_contract_says_vtcm(self):
        weights = _contract(self.metadata).get("weights") or []
        self.assertEqual([w.get("location") for w in weights], ["vtcm"])
        manifest = _manifest(self.metadata)
        policies = {p.get("policy") for p in manifest.get("weight_policies", [])}
        self.assertIn("resident-prepack", policies)
        self.assertNotIn("resident-prepack-ddr", policies)


if __name__ == "__main__":
    unittest.main(verbosity=2)
