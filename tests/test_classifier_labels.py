import numpy as np

from cla.layout.spec import Chain, ChainSpec, Role
from cla.train.classifier import labeled_set


def _spec(i: int, epitope: str, allele: str) -> ChainSpec:
    chains = (
        Chain(Role.TCRA, "CAVRDNYGQNFVF" + "A" * i),
        Chain(Role.TCRB, "CASSIVGGNEQFF" + "G" * i),
        Chain(Role.PEPTIDE, epitope),
        Chain(Role.MHC_HEAVY, "GSHSMRYF" + allele),
        Chain(Role.B2M, "IQRTPKIQ"),
    )
    # VDJdb rows carry label=1; the helper must not depend on that.
    return ChainSpec(chains, f"p{i}", label=None, epitope=epitope, mhc_allele=allele)


def test_labeled_set_marks_positives_one_and_decoys_zero_with_a_foreign_antigen():
    eps = ["GILGFVFTL", "NLVPMVATV", "YLQPRTFLL"]
    pos = [_spec(i, eps[i % 3], "AAA" if i % 2 else "CCC") for i in range(12)]
    out = labeled_set(pos, np.random.default_rng(0), pos)

    assert [s.label for s in out] == [1] * len(pos) + [0] * (len(out) - len(pos))
    own = {s.example_id: s.epitope for s in pos}
    decoys = out[len(pos) :]
    assert decoys
    for d in decoys:
        assert d.example_id.split("__neg")[0] in own
        assert d.epitope != own[d.example_id.split("__neg")[0]]
        assert d.by_role(Role.PEPTIDE).sequence == d.epitope
