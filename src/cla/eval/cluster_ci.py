"""Cluster-bootstrap intervals for the MHC_HEAVY results.

    python -m cla.eval.cluster_ci --device mps

The audit bootstraps over examples, but VDJdb is dominated by a few HLA alleles, so many
examples share one MHC heavy-chain sequence and nearly one score. Resampling examples counts
those repeats as independent and gives intervals that are too narrow. Here the unit of
resampling is the MHC allele.

GSI is recomputed from the per-example scores stored in each run's `nsi.json`; no model is
run. Context gain is not stored per example, so it is rescored with the same sample and
renderings as `cla.eval.context`. The example-level interval is reported beside the cluster
interval so the width difference is visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cla.data.splits import split_specs
from cla.eval.audit import load_config, load_specs_jsonl, stratified_subsample
from cla.eval.context import TARGETS, score_conditions
from cla.eval.metrics import bootstrap_ci, cluster_bootstrap_ci, gsi
from cla.layout.spec import ChainSpec
from cla.model.backbone import load_esm, resolve_device
from cla.utils.seeding import set_all_seeds

TARGET = "MHC_HEAVY"
RUNS = {"35M": "audit-esm2-35m", "150M": "audit-esm2-150m", "650M": "audit-esm2-650m"}
OPERATORS = ("G1", "G2")
ARMS = {"true": "dropout_on", "false": "dropout_off"}
CONTEXT_QUANTITIES = (
    "context_gain",
    "context_gain_vs_padded_alone",
    "scrambled_minus_alone",
    "complex_minus_scrambled",
)


def cluster_diagnostics(labels: Sequence[str]) -> dict[str, Any]:
    """Cluster count, largest-cluster share, and Kish's effective sample size.

    ``n_eff = (sum n_i)^2 / sum(n_i^2)`` is the number of equally sized clusters that would
    give the same concentration, a ceiling on how many independent observations the sample
    can hold.
    """
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    order = np.argsort(-counts, kind="stable")
    n = int(counts.sum())
    return {
        "n_examples": n,
        "n_clusters": int(len(counts)),
        "largest_cluster": str(values[order[0]]),
        "largest_cluster_fraction": float(counts[order[0]] / n),
        "kish_n_eff": float(n**2 / np.sum(counts.astype(float) ** 2)),
        "top_clusters": {str(values[i]): int(counts[i]) for i in order[:5]},
    }


def _interval(fn, *arrays, clusters: np.ndarray, boot: dict, seed: int) -> dict[str, Any]:
    point, c_lo, c_hi = cluster_bootstrap_ci(
        fn, *arrays, clusters=clusters, n_boot=boot["n_boot"], alpha=boot["alpha"], seed=seed
    )
    _, e_lo, e_hi = bootstrap_ci(fn, *arrays, n_boot=boot["n_boot"], alpha=boot["alpha"], seed=seed)
    return {
        "point": point,
        "cluster_ci": [c_lo, c_hi],
        "example_ci": [e_lo, e_hi],
        "width_ratio": (c_hi - c_lo) / (e_hi - e_lo) if e_hi > e_lo else None,
    }


def _perturbed(scores: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    entry = scores["perturbed"][key]
    return np.asarray(entry["index"]), np.asarray(entry["scores"], dtype=float)


def audit_gsi(run: dict, allele_of: dict[str, str]) -> dict[str, Any]:
    """Cluster and example-level GSI intervals for one run's MHC_HEAVY target."""
    t = run["targets"][TARGET]
    boot = run["config"]["bootstrap"]
    seed = int(run["config"]["seed"])
    n_mut = t["geometry"]["denominator"]["n_mutations"]
    # Geometry and C1 are applied to positives only; decoys have no spec row and get no key.
    alleles = np.array([allele_of.get(i, "") for i in run["examples"]["example_id"]])
    gaps = list(t["geometry"]["gaps_in_distribution"])
    ood = list(t["geometry"].get("out_of_distribution", {}).get("gaps", []))

    out: dict[str, Any] = {}
    population = None
    for arm in ARMS:
        s = t["scores"][arm]
        orig = np.array([np.nan if v is None else v for v in s["original"]])
        c_idx, c_scores = _perturbed(s, f"C1:{n_mut}")
        for op in OPERATORS:
            for gap in gaps + ood:
                key = "G@0" if gap == 0 else f"{op}@{gap}"
                g_idx, g_scores = _perturbed(s, key)
                common, ia, ib = np.intersect1d(g_idx, c_idx, return_indices=True)
                population = common
                cell = _interval(
                    gsi,
                    orig[common],
                    g_scores[ia],
                    c_scores[ib],
                    clusters=alleles[common],
                    boot=boot,
                    seed=seed,
                )
                cell["n_pairs"] = int(common.size)
                cell["out_of_distribution"] = gap in ood
                cell["cluster_lower_bound_above_1"] = bool(cell["cluster_ci"][0] > 1.0)
                out.setdefault(arm, {}).setdefault(op, {})[str(gap)] = cell
    return {
        "usable": bool(t["estimator"]["sanity"]["usable"]),
        "geometry_population": cluster_diagnostics(alleles[population]),
        "gsi": out,
    }


def audit_context_gain(
    model_name: str, chosen: list[ChainSpec], allele_of: dict[str, str], *, device, config: dict
) -> dict[str, Any]:
    """Rescore the context-gain conditions and bootstrap their medians by allele."""
    model, tok = load_esm(model_name, device)
    boot = {
        "n_boot": int(config["bootstrap"]["n_boot"]),
        "alpha": float(config["bootstrap"]["alpha"]),
    }
    render_kwargs = {
        "separator_tokens": int(config.get("separator_tokens", 1)),
        "symmetric_ends": bool(config.get("symmetric_ends", True)),
        "max_length": int(config.get("max_length", 512)),
    }
    seed = int(config.get("seed", 0))
    kept, plls = score_conditions(
        model, tok, chosen, TARGET, render_kwargs=render_kwargs, seed=seed, **TARGETS[TARGET]
    )
    alleles = np.array([allele_of[i] for i in kept])
    arms: dict[str, Any] = {}
    for arm, pll in plls.items():
        diffs = {
            "context_gain": pll["complex"] - pll["alone"],
            "context_gain_vs_padded_alone": pll["complex"] - pll["alone_padded"],
            "scrambled_minus_alone": pll["scrambled"] - pll["alone"],
            "complex_minus_scrambled": pll["complex"] - pll["scrambled"],
        }
        arms[arm] = {
            name: _interval(np.median, v, clusters=alleles, boot=boot, seed=seed)
            for name, v in diffs.items()
        }
    return {"n": len(kept), "sample": cluster_diagnostics(alleles), "arms": arms}


def _tsv_rows(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    head = [
        "section",
        "model",
        "usable",
        "operator",
        "gap",
        "token_dropout",
        "quantity",
        "point",
        "cluster_lo",
        "cluster_hi",
        "example_lo",
        "example_hi",
        "width_ratio",
        "cluster_lo_above_1",
    ]
    rows = ["\t".join(head)]

    def line(section, model, usable, op, gap, arm, quantity, c, flag):
        ratio = "" if c["width_ratio"] is None else f"{c['width_ratio']:.2f}"
        vals = [
            section,
            model,
            str(usable).lower(),
            op,
            gap,
            ARMS[arm],
            quantity,
            f"{c['point']:.4f}",
            f"{c['cluster_ci'][0]:.4f}",
            f"{c['cluster_ci'][1]:.4f}",
            f"{c['example_ci'][0]:.4f}",
            f"{c['example_ci'][1]:.4f}",
            ratio,
            flag,
        ]
        rows.append("\t".join(vals))

    for model, m in result["models"].items():
        usable = m["audit"]["usable"]
        for arm, ops in m["audit"]["gsi"].items():
            for op, gaps in ops.items():
                for gap, c in gaps.items():
                    label = f"{gap} (ood)" if c["out_of_distribution"] else gap
                    flag = str(c["cluster_lower_bound_above_1"]).lower()
                    line("gsi", model, usable, op, label, arm, "gsi", c, flag)
        for arm, qs in (m.get("context_gain") or {}).get("arms", {}).items():
            for q, c in qs.items():
                line("context_gain", model, usable, "", "", arm, q, c, "")

    diag = [
        "\t".join(
            [
                "model",
                "population",
                "n_examples",
                "n_clusters",
                "largest_cluster",
                "largest_cluster_fraction",
                "kish_n_eff",
            ]
        )
    ]
    for model, m in result["models"].items():
        pops = {"audit_geometry": m["audit"]["geometry_population"]}
        if "context_gain" in m:
            pops["context_gain"] = m["context_gain"]["sample"]
        for name, d in pops.items():
            diag.append(
                "\t".join(
                    [
                        model,
                        name,
                        str(d["n_examples"]),
                        str(d["n_clusters"]),
                        d["largest_cluster"],
                        f"{d['largest_cluster_fraction']:.3f}",
                        f"{d['kish_n_eff']:.2f}",
                    ]
                )
            )
    return rows, diag


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(RUNS))
    ap.add_argument("--device", choices=["cpu", "mps"], default=None)
    ap.add_argument("--results", type=Path, default=Path("results/phase1"))
    ap.add_argument("--config", type=Path, default=Path("configs/audit/esm2_150m.yaml"))
    ap.add_argument("--n-context", type=int, default=300)
    ap.add_argument("--skip-context", action="store_true")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    specs_path = Path(config.get("data", {}).get("specs", "data/processed/vdjdb_specs.jsonl"))
    if not specs_path.exists():
        print(f"error: {specs_path} not found. Run `make data` first.", file=sys.stderr)
        return 2
    specs = load_specs_jsonl(specs_path)
    # The allele string is not stored in nsi.json, so it is looked up by example id.
    allele_of = {s.example_id: s.mhc_allele for s in specs if s.mhc_allele}

    chosen: list[ChainSpec] = []
    device = None
    if not args.skip_context:
        seed = int(config.get("seed", 0))
        set_all_seeds(seed)
        device = resolve_device(args.device)
        sample = stratified_subsample(
            split_specs(specs), args.n_context, np.random.default_rng([seed, 1])
        )
        chosen = [s for s, _ in sample]

    result: dict[str, Any] = {
        "cluster_key": "mhc_allele",
        "unit": "whole alleles resampled with replacement",
        "models": {},
    }
    for name in args.models:
        run = json.loads((args.results / RUNS[name] / "nsi.json").read_text())
        entry: dict[str, Any] = {"audit": audit_gsi(run, allele_of)}
        if not args.skip_context:
            entry["context_gain"] = audit_context_gain(
                name, chosen, allele_of, device=device, config=config
            )
        result["models"][name] = entry
        print(f"done {name}")

    out_json = args.results / "mhc_cluster_ci.json"
    out_json.write_text(json.dumps(result, indent=2))
    rows, diag = _tsv_rows(result)
    (args.results / "mhc_cluster_ci.tsv").write_text("\n".join(rows) + "\n")
    (args.results / "mhc_cluster_ci_diagnostics.tsv").write_text("\n".join(diag) + "\n")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
