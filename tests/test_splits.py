"""Leakage tests. Non-negotiable: these run in CI and a failure blocks everything.

Leakage is the single most likely way this project produces a wrong result, so
the assertions here are deliberately blunt: no held-out allele string and no
held-out epitope string may appear anywhere in a training shard, under any
spelling.
"""

from __future__ import annotations

import pytest

from cla.data.splits import SplitConfig, normalize_allele, split_specs
from cla.layout.spec import Chain, ChainSpec, Role

CONFIG = SplitConfig.load()


def make_spec(
    example_id: str,
    *,
    epitope: str = "NLVPMVATV",
    allele: str = "HLA-A*02:01",
    beta: str = "CASSLGQAYEQYF",
    alpha: str = "CAVRDSNYQLIW",
) -> ChainSpec:
    return ChainSpec(
        chains=(
            Chain(Role.TCRA, alpha),
            Chain(Role.TCRB, beta),
            Chain(Role.PEPTIDE, epitope),
            Chain(Role.MHC_HEAVY, "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQ"),
            Chain(Role.B2M, "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSK"),
        ),
        example_id=example_id,
        label=1,
        epitope=epitope,
        mhc_allele=allele,
    )


@pytest.mark.parametrize(
    "written,expected",
    [
        ("HLA-A*02:01:01:01", "HLA-A*02:01"),
        ("hla-a*02:01", "HLA-A*02:01"),
        ("A*02:01", "HLA-A*02:01"),
        ("HLA-A*02:01", "HLA-A*02:01"),
    ],
)
def test_allele_spellings_collapse_to_one_key(written: str, expected: str) -> None:
    """A synonymous spelling must not let a held-out allele cross into training."""
    assert normalize_allele(written) == expected


def test_no_held_out_allele_or_epitope_reaches_training() -> None:
    # Every spec gets a distinct receptor. Sharing one would make the epitope
    # holdout propagate to the whole fixture and mask what this test is checking.
    def beta(tag: str) -> str:
        return "CASS" + "".join("ACDEFGHIKLMNPQRSTVWY"[ord(c) % 20] for c in tag) + "EQYF"

    specs = [
        make_spec(f"heldout-allele-{i}", allele=allele, beta=beta(f"a{i}"))
        for i, allele in enumerate(sorted(CONFIG.held_out_alleles))
    ]
    # Same alleles written at four-field resolution, the spelling most likely to slip through.
    specs += [
        make_spec(f"heldout-allele-long-{i}", allele=f"{allele}:01:01", beta=beta(f"b{i}"))
        for i, allele in enumerate(sorted(CONFIG.held_out_alleles))
    ]
    specs += [
        make_spec(f"heldout-epitope-{i}", epitope=epitope, beta=beta(f"c{i}"))
        for i, epitope in enumerate(sorted(CONFIG.held_out_epitopes))
    ]
    specs += [make_spec(f"train-{i}", beta=beta(f"t{i}")) for i in range(50)]

    shards = split_specs(specs, CONFIG)

    for spec in shards["train"] + shards["val"]:
        assert normalize_allele(spec.mhc_allele or "") not in CONFIG.held_out_alleles
        for chain in spec.chains:
            for epitope in CONFIG.held_out_epitopes:
                assert epitope not in chain.sequence

    assert shards["test_allele"], "held-out alleles produced an empty test shard"
    assert shards["test_epitope"], "held-out epitopes produced an empty test shard"


def test_tcr_seen_against_held_out_epitope_is_removed_everywhere() -> None:
    """A TCR that recognizes a held-out epitope leaks the answer if it also trains."""
    shared_beta = "CASSPRGGQETQYF"
    held = sorted(CONFIG.held_out_epitopes)[0]
    specs = [
        make_spec("against-held", epitope=held, beta=shared_beta),
        make_spec("same-tcr-other-epitope", epitope="NLVPMVATV", beta=shared_beta),
        make_spec("unrelated", epitope="NLVPMVATV", beta="CASSLGQAYEQYF"),
    ]

    shards = split_specs(specs, CONFIG)
    assigned = {s.example_id: shard for shard, group in shards.items() for s in group}

    assert assigned["same-tcr-other-epitope"] == "test_epitope"
    assert assigned["unrelated"] in {"train", "val"}


def test_split_assignment_is_stable_under_reordering() -> None:
    """Assignment must not depend on upstream row order, or results stop being comparable."""
    specs = [make_spec(f"ex-{i}") for i in range(200)]
    first = {s.example_id: k for k, g in split_specs(specs, CONFIG).items() for s in g}
    second = {s.example_id: k for k, g in split_specs(specs[::-1], CONFIG).items() for s in g}
    assert first == second
