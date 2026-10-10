import json

import numpy as np
import pytest

from cla.eval.audit import (
    compute_metrics,
    dose_response_shape,
    evaluate_gate_g1,
    evaluate_gate_gsi_model,
    run_audit,
)
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
        "operators": ["P1"],
        "geometry": {"operators": ["G1", "G2"], "gaps": [0, 5, 25], "out_of_distribution": [400]},
        "controls": {"c1_n_mutations": [20], "gsi_denominator_n": 20},
        "bootstrap": {"n_boot": 50, "alpha": 0.05},
        "max_length": 512,
        "separator_tokens": 1,
        "symmetric_ends": True,
        "scoring": {
            "targets": [["PEPTIDE"], ["TCRB"]],
            "estimators": {"PEPTIDE": "span_pll", "TCRB": "windowed_span_pll"},
            "windowed": {"window": 5, "stride": 5, "max_windows": 2},
        },
    }
    out = tmp_path / "res"
    run_audit(config, synthetic_specs(), out_dir=out, runs_root=tmp_path / "runs", device="cpu")
    parsed = json.loads((out / "nsi.json").read_text())
    assert parsed["baseline_rendering"] == {
        "separator_tokens": 1,
        "symmetric_ends": True,
        "max_length": 512,
    }
    assert parsed["data"]["n_scored"] == 16
    assert parsed["scoring"]["estimators"]["TCRB"]["name"] == "windowed_span_pll"
    for t in parsed["targets"].values():
        geo = t["geometry"]
        for op in ("G1", "G2"):
            gaps = geo["per_operator"][op]["gaps"]
            for arm in ("true", "false"):
                assert gaps["0"][arm]["gsi"]["value"] == 0.0
                assert gaps["25"][arm]["gsi"]["value"] > 0
                assert gaps["25"][arm]["n_pairs"] == 8
                e = gaps["25"][arm]
                assert e["median_abs_delta"] / e["c1_denominator"] == pytest.approx(
                    e["gsi"]["value"]
                )
            assert op in geo["out_of_distribution"]["per_operator"]
        assert geo["inertness_check"]["passed"]
        assert t["discrimination"]["true"]["auroc"]["ci_lo"] is not None
        assert t["token_dropout_control"]["median_abs_original_score_diff_on_vs_off"] > 0
    assert parsed["targets"]["TCRB"]["controls"]["C2"]["applicability"]["n_applied"] == 0
    assert "gsi" in parsed["gate_g1"]
    rows = (out / "summary.tsv").read_text().splitlines()
    assert rows[0].split("\t")[:3] == ["target", "operator", "gap"]
    assert len(rows) > 20


def test_gsi_gate_needs_two_gaps_and_ci_floor():
    def e(value, lo):
        return {"value": value, "ci_lo": lo}

    table = {
        "PEPTIDE": {"G1": {"5": e(1.4, 0.7), "25": e(1.1, 0.4), "100": e(2.0, 0.6)}},
        "MHC_HEAVY": {"G1": {"5": e(3.0, 2.0), "25": e(3.0, 2.0)}},
    }
    peptide_only = evaluate_gate_gsi_model(table, targets=["PEPTIDE"])
    assert peptide_only["qualifying_gaps"] == [5, 100]
    assert peptide_only["model_qualifies"]
    assert evaluate_gate_gsi_model(table)["qualifying_gaps"] == [5, 25, 100]
    one = evaluate_gate_gsi_model({"PEPTIDE": {"G1": {"5": e(1.4, 0.7), "25": e(0.9, 0.7)}}})
    assert not one["model_qualifies"]


def test_dose_response_distinguishes_step_from_graded():
    rng = np.random.default_rng(0)
    orig = np.zeros(400)
    idx = np.arange(400)
    noise = rng.uniform(0.9, 1.1, 400)
    boot = {"n_boot": 200, "alpha": 0.05}
    step = {g: (idx, 0.1 * noise) for g in (1, 2, 5, 25, 100)}
    graded = {g: (idx, 0.01 * g**0.6 * noise) for g in (1, 2, 5, 25, 100)}
    assert dose_response_shape(orig, step, boot=boot, seed=0)["shape"] == "step"
    assert dose_response_shape(orig, graded, boot=boot, seed=0)["shape"] == "graded"


def _m(nsi, rho=0.5, delta=0.0):
    return {"nsi": nsi, "spearman_rho": rho, "delta_auroc": delta}


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
