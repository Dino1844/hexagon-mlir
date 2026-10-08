#!/usr/bin/env python3
"""The W1 acceptance test for the HMX leaf cost table (host only, no device).

WHY THIS FILE EXISTS
--------------------
W1 (docs/architecture/hmx-coscheduling-followups-2026-10-07.md §2) delivered
a measured per-leaf pcyc table whose ONLY permitted use today is diagnostic.
Its acceptance criterion is not "the numbers look plausible" but a hard one:
**the table, combined with the query model, must reproduce the two measured
flip points.** If it cannot, the table or the model is wrong, and this test
is where that gets said out loud:

  Flip 1 -- serial vs staged (the transfer channel), measured at
  docs/results/kstage-floor-pinned-2026-10-02.md (build ce26015e, N=1000,
  3 reps): S1 1024x512x64 (hot activation) serial 55 us vs staged 64 us;
  S3 128^3 (hot) 10 vs 16; S2 256x64x2048 (cold activation) 87 vs 41.

  Flip 2 -- serial ring vs staged ring + vector read-out (the read-out
  channel), measured at docs/results/auto-convergence-2026-10-04.md §3
  (build 35c61fe4 -> 123bd15f, iters=1000): S1 (Mt=32) 47 vs 37 us, the
  read-out wins; S3 (Mt=4) 3 vs 7 us, the serial ring wins.

The model these are reproduced through is the SOURCE-HEAT mechanism, not the
K-tile-count proxy: the serial ring packs straight from DDR (hot ~91-104
pcyc/unit, cold ~392), the staged ring DMA-copies into VTCM and packs from
VTCM (~99.6) behind the engine, and staging pays exactly when the heat gain
amortizes the ring's fixed overhead. Flip 2's mechanism is the batch count:
the first read-out batch cannot overlap anything, so Mt < 2*G is pure
handoff, and above that the hidden unpack must exceed the split's protocol
cost.

THE TWO STYLES, FOLLOWING THE HOUSE CONTRACT-TEST PATTERN
---------------------------------------------------------
  * TableContractTests is the SOURCE-AGREEMENT style
    (test_hmx_manifest_field_agreement.py precedent): it parses
    HmxLeafCostTable.h and pins it against frozen literals -- every priced
    cell cites a source, the empty cells are enumerated as no-data, the
    measured anchors match the logs, the traceability blocks exist, and the
    upper-bound contract notes are present.
  * FlipReproductionTests is the MODEL-BEHAVIOR style: it re-implements the
    header's query arithmetic in Python ON THE PARSED VALUES (a line-for-
    line mirror of the inline functions -- the header is the single source
    of the numbers) and asserts the flips.

  * DerivedEntryContractTests (the W1 close-out, 2026-10-09) pins the
    DERIVED rows the co-scheduling review added (docs/architecture/
    hmx-coscheduling-architecture-review-2026-10-09.md §2.2/§2.3): the S1
    default-face chain decomposition and the World A/B bounds. Pinned:
    existence, the "derived-inferred" caliber, the undecided/R1-adjudicator
    labels, the evidence chain in the citation, structural separation from
    the measured tables, and the ties that keep the derived arithmetic
    glued to the measured cells it was composed from. NOT pinned: the
    values -- their home is the citation, and a re-derivation updates the
    entries and these ties together, loudly.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
  * No binding-level exercise of the C++ query functions: this session is
    edit-only (no builds), so nothing has compiled the header. Post-battle,
    add a third style that binds `leafcost::stagedBeatsSerial` /
    `readoutBeatsSerialRing` / `modeledII` through the pybind surface and
    asserts they agree with this mirror -- until then the mirror is the only
    executable form, and a divergence between the two would be invisible.
  * No device, no triton import: the whole file is stdlib-only and host-only.
  * No verdict for anything the device never ran: the model must flag the
    Mt=5..31 band and the (128 KiB, 1 MiB) activation-heat band unverified,
    and the tests assert that flag rather than a winner. The K=1024 gate
    flip (256x256x1024 -> staged) was a SELECTION observation, not a timing
    one -- the model abstains there on purpose, exactly as
    HmxPartitionPass.cpp's own comments do for the bands it never measured.

Run:  python3 -m pytest -q test_hmx_leaf_cost_table.py
      (or: python3 test_hmx_leaf_cost_table.py)
"""

import re
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
HEADER = (
    _HERE.parent / "include" / "hexagon" / "Dialect" / "Hmx" / "Transforms"
    / "HmxLeafCostTable.h"
)
TEXT = HEADER.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Parsers. Every regex is anchored to the exact one-line entry grammar the
# header uses, so a reformatted entry fails loudly here instead of silently
# dropping out of the table.
# ---------------------------------------------------------------------------

_STRINGS = r'(?:"(?:[^"\\]|\\.)*"\s*)+'

_SRC_ENTRY = re.compile(rf"\{{Src::(\w+),\s*({_STRINGS})\}},")
_NO_DATA_ENTRY = re.compile(rf"\{{Leaf::(\w+),\s*({_STRINGS})\}},")
_LEAF_ENTRY = re.compile(
    r"\{Leaf::(\w+),\s*Family::(\w+),\s*Cond::(\w+),\s*"
    r"(-?[\d.]+),\s*(\d+),\s*(\d+),\s*Src::(\w+)\},"
)
_ARM_ENTRY = re.compile(
    r"\{Family::(\w+),\s*Arm::(\w+),\s*(\d+),\s*(\d+),\s*Src::(\w+)\},"
)
_PROTO_ENTRY = re.compile(
    r"\{Proto::(\w+),\s*(-?[\d.]+),\s*Src::(\w+)\},"
)
_FLIP_ENTRY = re.compile(
    r"\{Flip::(\w+),\s*Family::(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),"
    r"\s*Winner::(\w+),\s*(\d+),\s*(\d+),\s*Src::(\w+)\},"
)
_FAMILY_ENTRY = re.compile(
    r"\{Family::(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(-?[\d.]+),\s*"
    r"(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(\d+),"
    r"\s*Src::(\w+)\},"
)
_SCALAR = re.compile(
    r"constexpr\s+(?:double|int64_t)\s+(\w+)\s*=\s*(-?[\d.]+);"
)

_CHAIN_ROW = re.compile(
    r'\{"(\w+)",\s*(\d+),\s*(-?[\d.]+),\s*"([-\w]+)",\s*Src::(\w+)\},'
)
_WORLD_ROW = re.compile(
    rf'\{{"([AB])",\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*({_STRINGS}),'
    rf'\s*({_STRINGS}),\s*Src::(\w+)\}},'
)


def _join_strings(blob: str) -> str:
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', blob)
    return "".join(parts)


def sources():
    """{Src name: citation text} -- the provenance registry."""
    return {
        m.group(1): _join_strings(m.group(2))
        for m in _SRC_ENTRY.finditer(TEXT)
    }


def leaf_prices():
    """[{leaf, family, cond, per_unit, unit_count, region_pcyc, src}]"""
    return [
        dict(
            leaf=m.group(1),
            family=m.group(2),
            cond=m.group(3),
            per_unit=float(m.group(4)),
            unit_count=int(m.group(5)),
            region_pcyc=int(m.group(6)),
            src=m.group(7),
        )
        for m in _LEAF_ENTRY.finditer(TEXT)
    ]


def no_data():
    return {m.group(1): _join_strings(m.group(2)) for m in _NO_DATA_ENTRY.finditer(TEXT)}


def arm_anchors():
    return [
        dict(
            family=m.group(1),
            arm=m.group(2),
            us=int(m.group(3)),
            pcyc=int(m.group(4)),
            src=m.group(5),
        )
        for m in _ARM_ENTRY.finditer(TEXT)
    ]


def protocol_facts():
    return {m.group(1): float(m.group(2)) for m in _PROTO_ENTRY.finditer(TEXT)}


def flip_anchors():
    return [
        dict(
            flip=m.group(1),
            family=m.group(2),
            m=int(m.group(3)),
            n=int(m.group(4)),
            k=int(m.group(5)),
            batch=int(m.group(6)),
            measured=m.group(7),
            serial_us=int(m.group(8)),
            other_us=int(m.group(9)),
            src=m.group(10),
        )
        for m in _FLIP_ENTRY.finditer(TEXT)
    ]


def family_prices():
    """{family name: dict of the FamilyPrices fields}"""
    out = {}
    for m in _FAMILY_ENTRY.finditer(TEXT):
        out[m.group(1)] = dict(
            family=m.group(1),
            m=int(m.group(2)),
            n=int(m.group(3)),
            k=int(m.group(4)),
            hot=float(m.group(5)),
            vtcm=float(m.group(6)),
            cold=float(m.group(7)),
            engine=float(m.group(8)),
            unpack=float(m.group(9)),
            fused=int(m.group(10)),
            src=m.group(11),
        )
    return out


def scalars():
    return {m.group(1): float(m.group(2)) for m in _SCALAR.finditer(TEXT)}


def chain_rows():
    """[{component, pcyc, us, caliber, src}] -- kS1DefaultChain (DERIVED)."""
    return [
        dict(
            component=m.group(1),
            pcyc=int(m.group(2)),
            us=float(m.group(3)),
            caliber=m.group(4),
            src=m.group(5),
        )
        for m in _CHAIN_ROW.finditer(TEXT)
    ]


def world_rows():
    """[{world, baseline, lo, hi, carrier, caliber, src}] -- kWorldBounds
    (DERIVED)."""
    return [
        dict(
            world=m.group(1),
            baseline=float(m.group(2)),
            lo=float(m.group(3)),
            hi=float(m.group(4)),
            carrier=_join_strings(m.group(5)),
            caliber=_join_strings(m.group(6)),
            src=m.group(7),
        )
        for m in _WORLD_ROW.finditer(TEXT)
    ]


SOURCES = sources()
LEAF_PRICES = leaf_prices()
NO_DATA = no_data()
ARM_ANCHORS = arm_anchors()
PROTOCOL = protocol_facts()
FLIP_ANCHORS = flip_anchors()
FAMILIES = family_prices()
SCALARS = scalars()
CHAIN_ROWS = chain_rows()
WORLD_ROWS = world_rows()

# ---------------------------------------------------------------------------
# The frozen literals -- the measured endpoints, copied from the named logs.
# These are the acceptance anchors; a table that moves them breaks here.
# ---------------------------------------------------------------------------

# docs/results/kstage-floor-pinned-2026-10-02.md + depth_ab_N1000.log
FLIP1_MEASURED = {
    "S1": ("serial", 55, 64),
    "S2": ("staged", 87, 41),
    "S3": ("serial", 10, 16),
}
# docs/results/auto-convergence-2026-10-04.md §3 (def vs def+depth2)
FLIP2_MEASURED = {
    "S1": ("readout", 47, 37),
    "S3": ("serial", 3, 7),
}
# docs/results/t-hmx-staging-gate-dead-2026-10-02.md §4.2 -- SELECTION only
# (256x256, logical K fixed at 14336, only the BK step varies).
BK_SELECTION = {512: "serial", 1024: "staged", 2048: "staged"}

# The in-kernel region medians the per-unit prices were derived over
# (logs/t1-partition-2026-10-02/lwp_*.json; region pcyc / unit count).
REGION_CROSSCHECK = {
    ("S1", "PackActF16", "InKernelWR1"): (6628, 64),
    ("S1", "MmaF16", "InKernelWR1"): (45445, 2048),
    ("S1", "UnpackAccF16", "InKernelWR1"): (44045, 512),
    ("S3", "PackActF16", "InKernelWR1"): (1538, 16),
    ("S3", "MmaF16", "InKernelWR1"): (1692, 64),
    ("S3", "UnpackAccF16", "InKernelWR1"): (1987, 16),
    ("S2", "UnpackAccF16", "InKernelWR1"): (3071, 16),
}


# ---------------------------------------------------------------------------
# The model mirror. Line-for-line comparable to the header's inline queries:
# tiles / unit counts / heat / dmaCopyPcyc / the two arm producers / the two
# flip verdicts. The header is the single source of every NUMBER; this code
# only re-expresses the arithmetic so it can execute without a build.
# ---------------------------------------------------------------------------


class Grid:
    def __init__(self, m, n, k):
        self.m, self.n, self.k = m, n, k


def tiles(extent):
    return extent // 32


def pack_act_units(g):
    return tiles(g.m) * tiles(g.k)


def unpack_units(g):
    return tiles(g.m) * tiles(g.n)


def engine_calls(g):
    return tiles(g.m) * tiles(g.n) * (3 + (tiles(g.k) + 31) // 32)


def act_src_bytes(g):
    return g.m * g.k * 2


def dma_copy_pcyc(b):
    return int(2.0 * b / SCALARS["kDmaRdWrBytesPerPcyc"])


def activation_heat(g):
    if act_src_bytes(g) <= SCALARS["kHotActBytes"]:
        return "Hot"
    if act_src_bytes(g) >= SCALARS["kColdActBytes"]:
        return "Cold"
    return "Band"


def modeled_serial_producer(g, fp, heat):
    pack = fp["cold"] if heat == "Cold" else fp["hot"]
    return int(pack * pack_act_units(g) + fp["engine"] * engine_calls(g))


def modeled_staged_producer(g, fp):
    if fp["fused"] > 0:
        return fp["fused"]
    return modeled_staged_producer_formula(g, fp)


def modeled_staged_producer_formula(g, fp):
    """The formula branch of the header's modeledStagedProducer (what it
    computes when there is no measured fused region to stand in)."""
    per_tile = int(
        fp["vtcm"] * tiles(g.k)
        + fp["engine"] * (tiles(g.n) * (3 + (tiles(g.k) + 31) // 32))
    )
    dma = dma_copy_pcyc(32 * g.k * 2)
    return dma + tiles(g.m) * max(per_tile, dma)


def modeled_unpack(g, fp):
    return int(fp["unpack"] * unpack_units(g))


def is_measured_endpoint(flip, g, batch, wins):
    for a in FLIP_ANCHORS:
        if a["flip"] != flip or (a["m"], a["n"], a["k"]) != (g.m, g.n, g.k):
            continue
        if flip == "ReadoutVsSerialRing" and a["batch"] != batch:
            continue
        if (a["measured"] != "Serial") == wins:
            return True
    return False


def staged_beats_serial(g, fp):
    """Mirror of stagedBeatsSerial -> (wins, verified, note)."""
    heat = activation_heat(g)
    if heat == "Band":
        return False, False, "heat band"
    if heat == "Hot":
        return False, is_measured_endpoint("StagedVsSerial", g, 0, False), "hot"
    gain = int((fp["cold"] - fp["vtcm"]) * pack_act_units(g))
    if gain <= SCALARS["kRingFixedOverheadPcyc"]:
        return False, False, "cold but too small"
    return True, is_measured_endpoint("StagedVsSerial", g, 0, True), "cold"


def readout_beats_serial_ring(g, fp, batch):
    """Mirror of readoutBeatsSerialRing -> (wins, verified, note)."""
    batches = (tiles(g.m) + batch - 1) // batch
    if batches < 2:
        return (
            False,
            is_measured_endpoint("ReadoutVsSerialRing", g, batch, False),
            "single batch",
        )
    hidden = modeled_unpack(g, fp)
    if hidden <= SCALARS["kReadoutFixedPcyc"]:
        return False, False, "hidden unpack below protocol cost"
    return (
        True,
        is_measured_endpoint("ReadoutVsSerialRing", g, batch, True),
        "two or more batches",
    )


def family(name):
    return FAMILIES[name]


def grid_of(name):
    fp = FAMILIES[name]
    return Grid(fp["m"], fp["n"], fp["k"])


# ---------------------------------------------------------------------------
# Style 1: the source-agreement contract tests.
# ---------------------------------------------------------------------------


class TableContractTests(unittest.TestCase):
    """The table is pinned to its provenance, its empty cells, its anchors,
    and the four contract notes -- the field-agreement style."""

    def test_the_header_parses_and_every_table_is_nonempty(self):
        # If any regex stopped matching, every other assertion here goes
        # green while checking nothing -- so the discovery check comes first
        # (the same discipline as the traceability gate's own vacuity test).
        self.assertGreater(len(SOURCES), 10, "kSources did not parse")
        self.assertGreater(len(LEAF_PRICES), 20, "kLeafPrices did not parse")
        self.assertGreater(len(NO_DATA), 15, "kNoData did not parse")
        self.assertGreater(len(ARM_ANCHORS), 8, "kArmAnchors did not parse")
        self.assertGreater(len(PROTOCOL), 4, "kReadoutProtocol did not parse")
        self.assertEqual(len(FLIP_ANCHORS), 5, "kFlipAnchors did not parse")
        self.assertEqual(len(FAMILIES), 3, "kFamilyPrices did not parse")
        self.assertGreaterEqual(len(SCALARS), 7, "model scalars did not parse")

    def test_every_priced_cell_cites_a_registered_source(self):
        for entry in LEAF_PRICES:
            self.assertIn(
                entry["src"],
                SOURCES,
                f"{entry['leaf']}/{entry['family']}/{entry['cond']} cites "
                f"unregistered source {entry['src']}",
            )
        for anchor in ARM_ANCHORS + FLIP_ANCHORS:
            self.assertIn(anchor["src"], SOURCES)
        for fp in FAMILIES.values():
            self.assertIn(fp["src"], SOURCES)

    def test_no_priced_cell_is_a_silent_zero(self):
        # A zero price is only legal as a reserved T1 slot; everything else
        # must carry a per-unit price or a whole-region median.
        for entry in LEAF_PRICES:
            if entry["src"] == "T1Pending":
                self.assertEqual(
                    (entry["per_unit"], entry["region_pcyc"]),
                    (0.0, 0),
                    f"T1 slot {entry['leaf']}/{entry['family']} carries a "
                    "value -- pending slots are filled, not pre-filled",
                )
                continue
            self.assertTrue(
                entry["per_unit"] > 0 or entry["region_pcyc"] > 0,
                f"{entry['leaf']}/{entry['family']}/{entry['cond']} has no "
                "value; move it to kNoData instead of pricing it zero",
            )

    def test_in_kernel_region_and_per_unit_agree(self):
        # The per-unit price must be the region median over the unit count
        # it claims (the derivation the citation describes), within the
        # rounding of a one-decimal price.
        for (fam, leaf, cond), (region, units) in REGION_CROSSCHECK.items():
            entry = next(
                e
                for e in LEAF_PRICES
                if e["family"] == fam
                and e["leaf"] == leaf
                and e["cond"] == cond
                and e["src"] != "T1Pending"
            )
            self.assertEqual(entry["region_pcyc"], region, (fam, leaf, cond))
            self.assertEqual(entry["unit_count"], units, (fam, leaf, cond))
            self.assertAlmostEqual(
                entry["per_unit"],
                region / units,
                delta=0.06,
                msg=f"{fam}/{leaf}: per-unit price drifted from "
                f"{region}/{units}",
            )

    def test_the_empty_cells_are_enumerated_not_silent(self):
        # "无数据" is a legal state; a missing row is not. The families with
        # no measurement anywhere must appear in kNoData.
        for leaf in (
            "PackActF32",
            "PackWeightF32",
            "PackWeightF16T",
            "PackWeightF32T",
            "PackActTailF16",
            "UnpackAccF32",
            "UnpackAccTailF32",
            "BiasInitUnit",
            "BiasLoad",
            "AccClear",
            "AccRead",
            "MmaF16",  # standalone; the blended in-kernel price IS priced
            "DmaStage",
            "DmaAwait",
        ):
            self.assertIn(
                leaf, NO_DATA, f"{leaf} has neither a price nor a no-data row"
            )

    def test_measured_anchors_match_the_logs(self):
        # The frozen literals: (winner, serial-arm us, other-arm us) per the
        # named docs -- position 2 is ALWAYS the serial arm's time, position
        # 3 the staged / read-out arm's -- plus the readout protocol facts
        # per the gap table's live counters.
        by_key = {(a["family"], a["arm"]): a for a in ARM_ANCHORS}
        for fam, (winner, serial_us, other_us) in FLIP1_MEASURED.items():
            self.assertEqual(by_key[(fam, "SerialPlain")]["us"], serial_us, fam)
            self.assertEqual(
                by_key[(fam, "StagedNoReadout")]["us"], other_us, fam
            )
            self.assertLess(
                serial_us if winner == "serial" else other_us,
                other_us if winner == "serial" else serial_us,
                f"{fam}: the recorded winner does not have the smaller time",
            )
        for fam, (winner, serial_us, other_us) in FLIP2_MEASURED.items():
            self.assertEqual(by_key[(fam, "SerialRingDef")]["us"], serial_us, fam)
            self.assertEqual(by_key[(fam, "StagedReadout")]["us"], other_us, fam)
            self.assertLess(
                serial_us if winner == "serial" else other_us,
                other_us if winner == "serial" else serial_us,
                f"{fam}: the recorded winner does not have the smaller time",
            )
        self.assertAlmostEqual(
            PROTOCOL["ConsumerBusyPerBatchPcyc"], 5676, delta=1
        )
        self.assertAlmostEqual(
            PROTOCOL["LastDrainNonDeferredUs"], 2.54, delta=0.01
        )
        # The flip anchors themselves are the same endpoints, restated as
        # verdicts -- the queries' verified flags key off them.
        self.assertEqual(
            {(a["family"], a["flip"]) for a in FLIP_ANCHORS},
            {
                ("S1", "StagedVsSerial"),
                ("S2", "StagedVsSerial"),
                ("S3", "StagedVsSerial"),
                ("S1", "ReadoutVsSerialRing"),
                ("S3", "ReadoutVsSerialRing"),
            },
        )

    def test_t1_slots_filled_with_their_own_fingerprints(self):
        # The 2026-10-08 campaign's LWP partition rerun
        # (logs/t1-lwp-partition-2026-10-08/, frozen build libtriton 78865e23
        # / libhmxapi f42384f2, plus the post-f16a7b0 default-face backfill on
        # fa79e610) filled the reserved slots. The fill contract, pinned here:
        #   * every FILLED cell cites the arm it came from, and that source's
        #     citation carries that run's OWN build fingerprint;
        #   * the one slot neither arm can measure (S2 unpack: fused into the
        #     staged ring on the replica, outlined-and-LWP-skipped on the
        #     default face) stays T1Pending at zero;
        #   * the 2026-10-02 lineage cells are NOT replaced -- every
        #     REGION_CROSSCHECK key still resolves to its LwpT1T2_0210 cell
        #     with the frozen values.
        pending = [e for e in LEAF_PRICES if e["src"] == "T1Pending"]
        self.assertEqual(
            [(e["leaf"], e["family"], e["cond"]) for e in pending],
            [("UnpackAccF16", "S2", "InKernelWR1")],
            "the still-pending T1 slot set changed",
        )
        for e in pending:
            self.assertEqual(
                (e["per_unit"], e["region_pcyc"]),
                (0.0, 0),
                "a pending T1 slot carries a value -- fill it or re-reserve it",
            )
        # the filled replica cells carry the frozen build's fingerprint
        self.assertIn("78865e23", SOURCES["T1Replica0810"])
        replica = sorted(
            (e["leaf"], e["family"], e["cond"])
            for e in LEAF_PRICES
            if e["src"] == "T1Replica0810"
        )
        self.assertEqual(
            replica,
            [
                ("MmaF16", "S1", "InKernelWR1"),
                ("MmaF16", "S3", "InKernelWR1"),
                ("PackActF16", "S1", "InKernelWR1"),
                ("PackActF16", "S3", "InKernelWR1"),
                ("UnpackAccF16", "S1", "InKernelWR1"),
                ("UnpackAccF16", "S3", "InKernelWR1"),
            ],
            "the replica-filled T1 slot set changed",
        )
        # the filled default-face cell carries the post-fix build fingerprint
        self.assertIn("fa79e610", SOURCES["T1Default0810"])
        default = sorted(
            (e["leaf"], e["family"], e["cond"])
            for e in LEAF_PRICES
            if e["src"] == "T1Default0810"
        )
        self.assertEqual(
            default,
            [("PackActF16", "S2", "FusedStagedRing")],
            "the default-face-filled T1 slot set changed",
        )
        # the 2026-10-02 lineage cells are NOT replaced
        for (fam, leaf, cond), (region, units) in REGION_CROSSCHECK.items():
            lineage = [
                e
                for e in LEAF_PRICES
                if e["family"] == fam
                and e["leaf"] == leaf
                and e["cond"] == cond
                and e["src"] == "LwpT1T2_0210"
            ]
            self.assertTrue(
                lineage, f"{fam}/{leaf}/{cond}: the 2026-10-02 cell was replaced"
            )
            self.assertEqual(
                (lineage[0]["region_pcyc"], lineage[0]["unit_count"]),
                (region, units),
                f"{fam}/{leaf}/{cond}: the 2026-10-02 cell drifted",
            )
        # the pending source still names its campaign build
        self.assertIn("T1Pending", SOURCES)
        self.assertIn("78865e23", SOURCES["T1Pending"])

    def test_every_model_scalar_has_a_traceability_block(self):
        # The repo's own gate (test_hmx_constant_traceability.py) demands a
        # mechanism/measurement/shape-set/workload block per numeric
        # constexpr; this file is in that gate's scope, so hold it to the
        # same rule here rather than discovering it in the post-battle lit
        # run.
        for name in SCALARS:
            self.assertIn(
                f"TRACEABILITY: {name}",
                TEXT,
                f"scalar {name} has no traceability block",
            )
            block_start = TEXT.index(f"TRACEABILITY: {name}")
            decl = re.search(
                rf"constexpr\s+(?:double|int64_t)\s+{name}\s*=", TEXT[block_start:]
            )
            self.assertIsNotNone(decl, f"scalar {name} has no declaration")
            block = TEXT[block_start : block_start + decl.start()]
            for field in (
                "mechanism:",
                "measurement:",
                "shape set:",
                "workload representativeness:",
            ):
                self.assertIn(
                    field,
                    block,
                    f"scalar {name}'s block is missing {field}",
                )

    def test_the_upper_bound_contract_notes_are_present(self):
        # Contract note 1, in both languages, verbatim -- the note the W1
        # order itself required ("表是上界口径，跨线程收益预测必须以实测
        # treat/base 比值为准，禁止从表直接减出净增量").
        self.assertIn("禁止从表直接减出净增量", TEXT)
        self.assertIn("FORBIDDEN to subtract", TEXT)
        self.assertIn("UPPER-BOUND ACCOUNTING", TEXT)
        # ...and the API shape backs the note: there is no net-gain query.
        for forbidden in ("netGain", "NetGain", "net_gain", "netDelta"):
            self.assertNotIn(forbidden, TEXT)

    def test_the_header_stays_standalone(self):
        # Nothing includes this header yet (it is diagnostic-only until W2),
        # so nothing would compile it. Keeping it dependency-free means the
        # post-battle syntax check is one command. <cstdint> is the only
        # include it is allowed to grow. ([ \t], not \s: \s would let the
        # pattern eat the newline before the # and match across lines.)
        self.assertEqual(
            re.findall(r"^[ \t]*#[ \t]*include[ \t]*.+$", TEXT, re.M),
            ["#include <cstdint>"],
        )

    def test_the_manifest_wall_is_documented(self):
        # modeled_ii is a spec, not an implementation: the v2 boundary is a
        # frozen closed enumeration (field-agreement test) and v3 is frozen
        # by user ruling. The spec section must name both walls.
        self.assertIn("NOT IMPLEMENTED (contract wall)", TEXT)
        self.assertIn("frozen", TEXT.lower())
        self.assertIn("modeledII", TEXT)


# ---------------------------------------------------------------------------
# Style 2: the model-behavior tests -- the two flips and the unverified bands.
# ---------------------------------------------------------------------------


class FlipReproductionTests(unittest.TestCase):
    """The acceptance: from the table's own numbers, reproduce the two
    measured flip points, and refuse to predict where nothing was run."""

    def test_flip1_serial_beats_staged_on_the_hot_anchors(self):
        # S1 1024x512x64 (act 128 KiB, hot) measured serial 55 vs staged 64;
        # S3 128^3 (act 32 KiB, hot) measured 10 vs 16.
        for fam in ("S1", "S3"):
            with self.subTest(family=fam):
                g, fp = grid_of(fam), family(fam)
                self.assertEqual(activation_heat(g), "Hot", fam)
                wins, verified, _ = staged_beats_serial(g, fp)
                self.assertFalse(wins, f"{fam}: model picks staged, device said serial")
                self.assertTrue(verified, f"{fam}: hot endpoint must be verified")
                winner, serial_us, staged_us = FLIP1_MEASURED[fam]
                self.assertLess(serial_us, staged_us, fam)

    def test_flip1_staged_beats_serial_on_the_cold_anchor(self):
        # S2 256x64x2048 (act 1 MiB, cold) measured serial 87 vs staged 41.
        g, fp = grid_of("S2"), family("S2")
        self.assertEqual(activation_heat(g), "Cold")
        wins, verified, _ = staged_beats_serial(g, fp)
        self.assertTrue(wins, "model picks serial, device said staged")
        self.assertTrue(verified, "cold endpoint must be verified")

    def test_flip1_the_heat_gain_actually_explains_the_cold_win(self):
        # Not just the verdict: the mechanism's magnitudes. The serial arm's
        # modeled total (cold pack dominates) must exceed the measured serial
        # arm by no more than the upper-bound margin, and the staged
        # producer is the measured fused region. Cross-build, order of
        # magnitude only (contract note 2) -- this is a sanity band, not an
        # A/B.
        g, fp = grid_of("S2"), family("S2")
        serial_total = modeled_serial_producer(g, fp, "Cold") + modeled_unpack(g, fp)
        measured_pcyc = FLIP1_MEASURED["S2"][1] * SCALARS["kSteadyGhz"] * 1000.0
        ratio = serial_total / measured_pcyc
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 1.30, "modeled serial blew past the measurement")
        # ...and the standalone cold price is what makes the serial arm big:
        self.assertGreater(
            fp["cold"] * pack_act_units(g),
            0.9 * measured_pcyc,
            "the cold pack no longer dominates the serial arm -- the "
            "mechanism story broke",
        )

    def test_flip1_the_k1024_gate_boundary_lands_in_the_unverified_band(self):
        # The gate flips at K=1024 as a SELECTION matter (BK sweep, §4.2).
        # The model must not claim a timed winner there: 256x256x1024 has a
        # 512 KiB activation, inside the unmeasured (128 KiB, 1 MiB) heat
        # band. At BK=2048 the activation is 1 MiB (cold) and the model
        # agrees with the selection; at BK=512 and BK=1024 it is inside the
        # band and the model abstains. Asserting the abstention IS the
        # honesty check: a model that "reproduced" the boundary by timing it
        # does not exist.
        for bk, selection in BK_SELECTION.items():
            with self.subTest(bk=bk):
                g = Grid(256, 256, bk)
                # The BK-sweep shapes have no family of their own; the S2
                # bundle stands in. That is sound for THIS query: the Hot
                # branch uses no prices at all, and the Cold branch uses
                # only the cold/vtcm pack prices, which are the family-
                # independent standalone cells (391.85 / 99.57) in every
                # bundle.
                fp = family("S2")
                wins, verified, _ = staged_beats_serial(g, fp)
                act = act_src_bytes(g)
                if act < SCALARS["kColdActBytes"] and act > SCALARS["kHotActBytes"]:
                    self.assertFalse(verified, "heat band claimed a verdict")
                    # no timing exists at the boundary; the selection is
                    # recorded, the model abstains, and that is the answer
                else:
                    # BK=2048: cold, staged wins -- agrees with the sweep
                    self.assertEqual(
                        wins,
                        selection == "staged",
                        "model disagrees with a measured-side selection",
                    )

    def test_flip2_readout_wins_at_mt32_and_loses_at_mt4(self):
        # S1 (Mt=32, batch 4 -> 8 batches): measured 47 vs 37, readout wins.
        g, fp = grid_of("S1"), family("S1")
        wins, verified, _ = readout_beats_serial_ring(
            g, fp, int(SCALARS["kReadoutBatch"])
        )
        self.assertTrue(wins, "model picks the serial ring, device said readout")
        # S1 is the CALIBRATION endpoint of kReadoutFixedPcyc (its own
        # traceability says so); verified flags it as measured regardless.
        self.assertTrue(verified)
        # S3 (Mt=4 -> 1 batch): measured 3 vs 7, serial wins -- the
        # INDEPENDENT check: pure handoff, no constant involved.
        g3, fp3 = grid_of("S3"), family("S3")
        wins3, verified3, _ = readout_beats_serial_ring(
            g3, fp3, int(SCALARS["kReadoutBatch"])
        )
        self.assertFalse(wins3, "model picks readout, device said serial ring")
        self.assertTrue(verified3, "the Mt=4 endpoint must be verified")

    def test_flip2_the_mt5_to_mt31_band_is_flagged_unverified(self):
        # "No shape between Mt=5 and Mt=31 has been run" (the pass's own
        # read-out batch comment). Whatever the model predicts in that band,
        # it must not claim a measurement. This sweeps S1's geometry (the
        # only family with an anchor on the winning side), holding the
        # family prices and varying M.
        fp = family("S1")
        for mt in range(5, 32):
            with self.subTest(mt=mt):
                g = Grid(mt * 32, 512, 64)
                _, verified, _ = readout_beats_serial_ring(
                    g, fp, int(SCALARS["kReadoutBatch"])
                )
                self.assertFalse(
                    verified, f"Mt={mt} claimed a measured endpoint"
                )
        # The model's own crossover inside the band (hidden unpack
        # 86.0*16*Mt crossing kReadoutFixedPcyc) is a PREDICTION; assert it
        # only as model behavior, never as a device fact.
        loser = [mt for mt in range(5, 32)
                 if not readout_beats_serial_ring(
                     Grid(mt * 32, 512, 64), fp, int(SCALARS["kReadoutBatch"])
                 )[0]]
        winner = [mt for mt in range(5, 32)
                  if readout_beats_serial_ring(
                      Grid(mt * 32, 512, 64), fp, int(SCALARS["kReadoutBatch"])
                  )[0]]
        self.assertTrue(loser and winner, "the band crossover vanished")
        self.assertLess(max(loser), min(winner), "the band crossover is not monotone")

    def test_flip2_s2_readout_is_predicted_unmeasured(self):
        # S2 has two batches (Mt=8) but only ~3.1k pcyc of unpack to hide,
        # below the protocol cost -- the model says "no measured endpoint",
        # and indeed no clean S2 readout-only arm exists (the auto-
        # convergence off/def pair differs in WSR too).
        g, fp = grid_of("S2"), family("S2")
        wins, verified, _ = readout_beats_serial_ring(
            g, fp, int(SCALARS["kReadoutBatch"])
        )
        self.assertFalse(verified, "S2 readout claimed a measurement")
        self.assertFalse(wins)

    def test_modeled_serial_totals_stay_within_the_upper_bound_band(self):
        # Upper-bound accounting: the modeled serial region sum must sit
        # near the measured serial arm -- below it by no more than the
        # un-modeled residual (~10%), above it by no more than the overlap
        # discount allows (contract note 1; the real core ran >=37% below
        # the region sum once). Cross-build, order of magnitude only.
        for fam, arm_key, measured_us in (
            ("S1", "SerialRingDef", FLIP2_MEASURED["S1"][1]),
            ("S3", "SerialRingDef", FLIP2_MEASURED["S3"][1]),
            ("S2", "SerialPlain", FLIP1_MEASURED["S2"][1]),
        ):
            with self.subTest(family=fam):
                g, fp = grid_of(fam), family(fam)
                total = modeled_serial_producer(
                    g, fp, activation_heat(g)
                ) + modeled_unpack(g, fp)
                measured_pcyc = measured_us * SCALARS["kSteadyGhz"] * 1000.0
                ratio = total / measured_pcyc
                self.assertGreater(
                    ratio, 0.70, f"{fam}: modeled total collapsed ({ratio:.2f})"
                )
                self.assertLess(
                    ratio, 1.25, f"{fam}: modeled total blew up ({ratio:.2f})"
                )

    def test_the_s2_fused_region_is_consistent_with_the_staged_formula(self):
        # The staged-producer FORMULA (fill DMA + per-m-tile max(DMA, VTCM
        # pack + engine)) must land near S2's measured fused region
        # (50,834 pcyc) even though the arm models use the measured cell.
        # This is the only cross-check on kDmaRdWrBytesPerPcyc in situ; the
        # 15% gap is the DMA-rate calibration gap its traceability names.
        g, fp = grid_of("S2"), family("S2")
        self.assertGreater(fp["fused"], 0)
        ratio = modeled_staged_producer_formula(g, fp) / fp["fused"]
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 1.30, "the staged formula drifted from the fused region")

    def test_kreadout_fixed_pcyc_ties_to_its_derivation(self):
        # kReadoutFixedPcyc is DERIVED from the S1 positive endpoint (its
        # traceability says so): the arm's own recorded PerfPcycles (78,592,
        # carried in kArmAnchors) minus the modeled S1 staged producer.
        # Assert the arithmetic still closes -- if the prices or the formula
        # drift, this constant must be re-derived, not silently reused.
        g, fp = grid_of("S1"), family("S1")
        anchor = next(
            a for a in ARM_ANCHORS
            if a["family"] == "S1" and a["arm"] == "StagedReadout"
        )
        self.assertGreater(anchor["pcyc"], 0, "the pcyc anchor went missing")
        derived = anchor["pcyc"] - modeled_staged_producer_formula(g, fp)
        self.assertAlmostEqual(
            derived,
            SCALARS["kReadoutFixedPcyc"],
            delta=200,
            msg="kReadoutFixedPcyc no longer matches its derivation",
        )

    def test_modeled_ii_matches_the_unit_formulas(self):
        # The manifest spec's per-region value, checked against the
        # documented unit counts: pack Mt*Kt, engine Mt*Nt*(3+ceil(Kt/32))
        # -- the count the 2026-09-27 -> 2026-10-02 engine-region drop
        # (57,856 -> 45,445 pcyc when K-fusion halved the mma count)
        # validates -- and unpack Mt*Nt.
        g, fp = grid_of("S1"), family("S1")
        pack = int(fp["hot"] * pack_act_units(g))
        engine = int(fp["engine"] * engine_calls(g))
        unpack = modeled_unpack(g, fp)
        self.assertEqual((pack, engine, unpack), (6630, 45465, 44032))
        # the engine count formula is the post-K-fusion one
        self.assertEqual(engine_calls(g), 2048)
        self.assertEqual(engine_calls(Grid(256, 64, 2048)), 80)
        self.assertEqual(engine_calls(Grid(128, 128, 128)), 64)


# ---------------------------------------------------------------------------
# Style 3: the derived-entry contract tests (the W1 close-out, 2026-10-09).
# ---------------------------------------------------------------------------


class DerivedEntryContractTests(unittest.TestCase):
    """The review's DERIVED rows: the S1 default-face chain decomposition
    and the World A/B bounds (review §2.2/§2.3).

    These are DERIVED rows, not measured slots. What is pinned here: their
    EXISTENCE, their CERTAINTY LABELS ("derived-inferred"; the worlds
    additionally "undecided" + the R1 adjudicator), their CITATION (the
    review + T1 + both arms' build fingerprints), their STRUCTURAL
    SEPARATION from the measured tables, and the TIES that keep the
    derived arithmetic glued to the measured cells it was composed from.
    What is deliberately NOT pinned: the values as eternally correct --
    their home is the citation, and a re-derivation updates the entries
    and these ties together, loudly.
    """

    def test_the_derived_tables_parse(self):
        # The discovery check comes first (the same discipline as above):
        # if either regex stopped matching, every other assertion in this
        # class goes green while checking nothing.
        self.assertEqual(
            [r["component"] for r in CHAIN_ROWS],
            [
                "pack_act",
                "engine",
                "glue",
                "fused_total",
                "residual",
                "unpack_concurrent",
            ],
            "kS1DefaultChain did not parse (or a row changed)",
        )
        self.assertEqual(
            [w["world"] for w in WORLD_ROWS],
            ["A", "B"],
            "kWorldBounds did not parse (or a world changed)",
        )

    def test_every_derived_row_is_labeled_inferred_and_cites_the_review(self):
        for r in CHAIN_ROWS + WORLD_ROWS:
            name = r.get("component", r.get("world"))
            self.assertIn(
                "derived-inferred",
                r["caliber"],
                f"{name}: a derived row lost its inferred caliber",
            )
            self.assertEqual(
                r["src"],
                "ArchReview1009",
                f"{name}: a derived row cites something other than the review",
            )
        # the world bounds additionally name the undecided question and
        # the R1 adjudicator -- the honesty label the close-out requires
        for w in WORLD_ROWS:
            self.assertIn("undecided", w["caliber"], f"World {w['world']}")
            self.assertIn("R1", w["caliber"], f"World {w['world']}")

    def test_the_review_citation_carries_the_full_evidence_chain(self):
        # 出处纪律: the review doc + the T1 doc + BOTH arms' build
        # fingerprints + the default-face run's own clock + the DERIVED
        # caliber -- every needle a reader needs to re-derive the rows.
        cite = SOURCES["ArchReview1009"]
        for needle in (
            "hmx-coscheduling-architecture-review-2026-10-09.md",
            "t1-lwp-partition-2026-10-08.md",
            "78865e23",  # the replica arm's libtriton fingerprint
            "fa79e610",  # the default-face libtriton fingerprint
            "f42384f2",  # libhmxapi, unchanged across both arms
            "2.147 GHz",  # the default-face run's own implied clock
            "DERIVED",
        ):
            self.assertIn(needle, cite, f"the review citation lost {needle}")

    def test_no_derived_row_is_disguised_as_a_measured_slot(self):
        # Structural separation, stated directly: the measured tables must
        # not grow a review-cited row. (The frozen-set tests above pin
        # their exact membership; this is the rule itself, so a future
        # entry added to a measured table fails here first.)
        for e in LEAF_PRICES:
            self.assertNotEqual(
                e["src"],
                "ArchReview1009",
                f"{e['leaf']}/{e['family']}/{e['cond']}: a measured price "
                "cell cites the review -- derived rows live in their own "
                "arrays",
            )
        for a in ARM_ANCHORS + FLIP_ANCHORS:
            self.assertNotEqual(a["src"], "ArchReview1009")
        for fp in FAMILIES.values():
            self.assertNotEqual(fp["src"], "ArchReview1009")
        # ...and the derived section declares what it is
        self.assertIn("NOT measured slots", TEXT)

    def test_the_chain_rows_restate_the_replica_cells_exactly(self):
        # pack_act / engine / unpack_concurrent are the T1Replica0810
        # region medians restated in the review's decomposition. The tie
        # fails loudly if those cells are ever re-measured: re-derive the
        # decomposition, never let it drift.
        by_component = {r["component"]: r for r in CHAIN_ROWS}
        for leaf, comp in (
            ("PackActF16", "pack_act"),
            ("MmaF16", "engine"),
            ("UnpackAccF16", "unpack_concurrent"),
        ):
            cell = next(
                e
                for e in LEAF_PRICES
                if e["family"] == "S1"
                and e["leaf"] == leaf
                and e["cond"] == "InKernelWR1"
                and e["src"] == "T1Replica0810"
            )
            self.assertEqual(
                by_component[comp]["pcyc"],
                cell["region_pcyc"],
                f"{comp}: drifted from the replica cell it restates",
            )

    def test_the_decomposition_sums_and_converts_close(self):
        # Internal closure, on the parsed values (never frozen literals):
        # the three main-chain components sum to the fused total, and every
        # us figure is its pcyc at the citation's own clock.
        rows = {r["component"]: r for r in CHAIN_ROWS}
        self.assertEqual(
            rows["pack_act"]["pcyc"]
            + rows["engine"]["pcyc"]
            + rows["glue"]["pcyc"],
            rows["fused_total"]["pcyc"],
            "the main-chain components no longer sum to the fused total",
        )
        clock_ghz = float(
            re.search(r"([\d.]+) GHz", SOURCES["ArchReview1009"]).group(1)
        )
        for r in CHAIN_ROWS:
            self.assertAlmostEqual(
                r["us"],
                r["pcyc"] / (clock_ghz * 1000.0),
                delta=0.15,
                msg=f"{r['component']}: us is not its pcyc at the cited clock",
            )

    def test_the_reviews_closure_check_still_holds(self):
        # The review's own sanity check: fused_total + residual ~= the
        # measured default-face 38 us (T1 §7a, N=1000/20000 ladder 38/37;
        # the ~0.4 us gap is the independent engine residual region the
        # five-row table leaves outside the fused total). The 38 us
        # baseline is a MEASURED anchor the world rows carry -- frozen-
        # literal style: a new baseline is a re-derivation and must update
        # this pin consciously.
        rows = {r["component"]: r for r in CHAIN_ROWS}
        fused_plus_residual = (
            rows["fused_total"]["us"] + rows["residual"]["us"]
        )
        for w in WORLD_ROWS:
            self.assertEqual(
                w["baseline"],
                38.0,
                "the world rows' baseline left the measured 38 us anchor",
            )
            self.assertLess(
                fused_plus_residual,
                w["baseline"] + 1.0,
                "fused+residual overshot the measured default face",
            )
            self.assertGreater(
                fused_plus_residual,
                w["baseline"] - 2.0,
                "fused+residual collapsed away from the measured face",
            )

    def test_world_a_ties_to_its_mechanism(self):
        # World A's bound = engine + residual (pack+DMA+glue hidden inside
        # the engine window), and the premise -- the window is big enough
        # -- must still hold on the parsed rows.
        rows = {r["component"]: r for r in CHAIN_ROWS}
        a = next(w for w in WORLD_ROWS if w["world"] == "A")
        self.assertAlmostEqual(
            (a["lo"] + a["hi"]) / 2.0,
            rows["engine"]["us"] + rows["residual"]["us"],
            delta=0.3,
            msg="World A's bound drifted from engine + residual",
        )
        self.assertLess(
            rows["pack_act"]["us"] + rows["glue"]["us"],
            rows["engine"]["us"],
            "World A's premise broke: pack+glue no longer fit inside the "
            "engine window -- re-derive, do not reuse",
        )
        self.assertLess(a["hi"], a["baseline"])

    def test_world_b_ties_to_its_mechanism(self):
        # World B's lower bound = max(engine, concurrent unpack, the main
        # chain); the upper bound is the review's -30% endpoint of the
        # baseline. Both must sit below the baseline -- the whole point of
        # the topology.
        rows = {r["component"]: r for r in CHAIN_ROWS}
        b = next(w for w in WORLD_ROWS if w["world"] == "B")
        main_chain = (
            rows["pack_act"]["us"]
            + rows["glue"]["us"]
            + rows["residual"]["us"]
        )
        three_thread_max = max(
            rows["engine"]["us"], rows["unpack_concurrent"]["us"], main_chain
        )
        self.assertAlmostEqual(
            b["lo"],
            three_thread_max,
            delta=0.5,
            msg="World B's lower bound drifted from the three-thread max",
        )
        self.assertAlmostEqual(
            b["hi"],
            b["baseline"] * 0.70,
            delta=1.0,
            msg="World B's upper bound drifted from the -30% endpoint",
        )
        self.assertGreater(b["hi"], b["lo"])
        self.assertLess(b["hi"], b["baseline"])


if __name__ == "__main__":
    unittest.main()
