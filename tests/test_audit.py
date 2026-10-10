import json

import numpy as np
import pytest

from cla.eval.audit import compute_metrics, evaluate_gate_g1, evaluate_gate_g1_nsi_c, run_audit
from cla.eval.negatives import make_mismatched_negatives
from cla.layout.spec import Chain, ChainSpec, Role

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def _seq(rng, n):
    return "".join(rng.choice(AA, n))


def synthetic_specs(n: int = 8) -> list[ChainSpec]:
    rng = np.random.default_rng(0)
    peptides = [_seq(rng, 9) for _ in range(4)]
    mhcs = [_seq(rng, 60) for _ in range(4)]
    b2m = _seq(rng, 30)
    specs = []
    for i in range(n):
        e = i % 4
        chains = (
            Chain(Role.TCRA, _seq(rng, 12 + i % 3)),
            Chain(Role.TCRB, _seq(rng, 11 + i % 5)),
            Chain(Role.PEPTIDE, peptides[e]),
            Chain(Role.MHC_HEAVY, mhcs[e]),
            Chain(Role.B2M, b2m),
        )
        specs.append(ChainSpec(chains, f"syn-{i}", 1, peptides[e], f"HLA-X*0{e}:01", "synthetic"))
    return specs


def _pairs(spec):
    return spec.by_role(Role.TCRA).sequence, spec.by_role(Role.TCRB).sequence, spec.epitope


def test_negatives_never_reproduce_a_positive_pair():
    specs = synthetic_specs(16)
    # A receptor seen against two epitopes: neither may appear in its decoys.
    shared = specs[0]
    extra = ChainSpec(
        shared.chains[:2] + specs[1].chains[2:], "syn-extra", 1, specs[1].epitope, "HLA-X*01:01"
    )
    outside = ChainSpec(
        shared.chains[:2] + specs[2].chains[2:], "syn-outside", 1, specs[2].epitope, "HLA-X*02:01"
    )
    pool = [*specs, extra]
    negs = make_mismatched_negatives(
        pool, np.random.default_rng(1), n_per_positive=3, known_positives=[*pool, outside]
    )
    positives = {_pairs(s) for s in [*pool, outside]}
    assert negs
    assert all(n.label == 0 for n in negs)
    assert not {_pairs(n) for n in negs} & positives
    assert all("__neg" in n.example_id for n in negs)
    # Beta-only guard: the shared receptor's beta never meets epitopes it was seen with.
    seen = {specs[0].epitope, specs[1].epitope, specs[2].epitope}
    beta = shared.by_role(Role.TCRB).sequence
    assert not any(n.by_role(Role.TCRB).sequence == beta and n.epitope in seen for n in negs)


def test_audit_smoke_end_to_end(tmp_path):
    config = {
        "run_name": "smoke",
        "model": "35M",
        "seed": 0,
        "n_examples": 8,
        "operators": ["P1", "P5"],
        "bootstrap": {"n_boot": 50, "alpha": 0.05},
        "max_length": 512,
        "separator_tokens": 1,
        "scoring": {"estimator": "span_pll", "targets": [["PEPTIDE"], ["TCRB"]]},
    }
    out = tmp_path / "res"
    result = run_audit(
        config, synthetic_specs(), out_dir=out, runs_root=tmp_path / "runs", device="cpu"
    )
    parsed = json.loads((out / "nsi.json").read_text())
    assert parsed["data"]["n_scored"] == 16
    assert set(parsed["targets"]) == {"PEPTIDE", "TCRB"}
    assert set(parsed["discrimination"]) == {"PEPTIDE", "TCRB"}
    for t in parsed["targets"].values():
        p1 = t["operators"]["P1"]["token_dropout"]
        assert set(p1) == {"true", "false"}
        assert all(p1[arm]["n_pairs"] == 16 for arm in p1)
        assert all(p1[arm]["nsi_c"]["n_paired"] == 16 for arm in p1)
        assert all(p1[arm]["nsi_c"]["value"] is not None for arm in p1)
        assert t["operators"]["P5"]["token_dropout"]["true"]["nsi_c"]["value"] is None
        assert t["discrimination"]["true"]["auroc"]["ci_lo"] is not None
        assert t["p1_arm_agreement"]["available"]
        assert t["gate_g1"]["nsi_c"]["verdict"] in {"PROCEED", "NOT_PROCEED", "UNDETERMINED"}
        diff = t["token_dropout_control"]["median_abs_original_score_diff_on_vs_off"]
        assert diff > 0
    assert result["targets"]["TCRB"]["target_roles"] == ["TCRB"]


def _m(nsi, rho=0.5, delta=0.0):
    return {"nsi": nsi, "spearman_rho": rho, "delta_auroc": delta}


def _c(nsi_c, rho=0.5):
    return {"nsi_c": nsi_c, "spearman_rho": rho}


@pytest.mark.parametrize(
    ("metrics", "verdict"),
    [
        # Two nulls clear both bars and the positive control moves most.
        ({"P1": _m(0.4), "P2": _m(0.3), "P4": _m(0.05), "P5": _m(1.2)}, "PROCEED"),
        # High NSI but ranks preserved (rho > 0.90) fails the pass criterion.
        ({"P1": _m(0.4, 0.95), "P2": _m(0.3, 0.97), "P5": _m(1.2)}, "AMBIGUOUS"),
        # All nulls quiet and AUROC unmoved.
        ({"P1": _m(0.02), "P2": _m(0.1, delta=-0.01), "P4": _m(0.05), "P5": _m(0.9)}, "STOP"),
        # Missing NSI cannot count toward STOP.
        ({"P1": _m(0.02), "P2": _m(None), "P5": _m(0.9)}, "AMBIGUOUS"),
        # No computable NSI at all: the rule is not evaluated.
        ({"P1": _m(None), "P2": _m(None), "P5": _m(None)}, None),
    ],
)
def test_gate_g1_verdicts(metrics, verdict):
    assert evaluate_gate_g1(metrics)["verdict"] == verdict


@pytest.mark.parametrize(
    ("metrics", "verdict"),
    [
        ({"P1": _c(0.4), "P2": _c(0.3), "P4": _c(0.05)}, "PROCEED"),
        # High rank correlation fails an operator even with NSI-C above threshold.
        ({"P1": _c(0.4, 0.95), "P2": _c(0.3, 0.97), "P4": _c(0.05)}, "NOT_PROCEED"),
        # P5 is the denominator and never counts as a null operator.
        ({"P1": _c(0.1), "P2": _c(0.1), "P4": _c(0.1), "P5": _c(5.0)}, "NOT_PROCEED"),
        ({"P1": _c(0.4), "P2": _c(None), "P4": _c(0.05)}, "UNDETERMINED"),
    ],
)
def test_gate_g1_nsi_c_verdicts(metrics, verdict):
    assert evaluate_gate_g1_nsi_c(metrics)["verdict"] == verdict


def test_nsi_withheld_when_binder_decoy_gap_is_unresolved():
    rng = np.random.default_rng(0)
    orig = rng.normal(size=60)
    labels = np.tile([0.0, 1.0], 30)
    kw = {"boot": {"n_boot": 50, "alpha": 0.05}, "bins": (0, 1000), "min_bin_n": 5, "seed": 0}
    shifted = orig + rng.normal(size=60)
    unresolved = orig.copy()
    unresolved[labels == 1] = np.median(orig[labels == 0]) + 1e-4  # gap far below the floor
    m = compute_metrics(unresolved, shifted, labels, np.full(60, 12.0), **kw)
    assert m["nsi"]["value"] is None
    assert m["nsi"]["reason"] == "denominator unresolved"
    assert m["nsi"]["separation"] < 1e-3
    resolved = orig.copy()
    resolved[labels == 1] += 5.0
    assert compute_metrics(resolved, shifted, labels, np.full(60, 12.0), **kw)["nsi"]["value"]
