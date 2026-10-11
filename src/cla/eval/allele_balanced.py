"""Allele-balanced confirmation of the MHC_HEAVY geometry sensitivity.

    python -m cla.eval.allele_balanced --device mps

The main audit sample is dominated by HLA-A*02:01 (Kish n_eff about 1.9), so a cluster
bootstrap over alleles has almost nothing to resample. Here each model is audited on a sample
with up to a fixed number of examples per distinct MHC allele (MHC_HEAVY, operator G1 only),
and GSI is reported pooled with a cluster bootstrap over alleles, with the example-level
interval beside it, and per allele. The per-allele table is the test of whether the effect
is general or specific to one allele; pooled intervals cannot distinguish the two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cla.eval.audit import load_config, load_specs_jsonl, run_audit
from cla.eval.cluster_ci import ARMS, TARGET, _interval, cluster_diagnostics
from cla.eval.metrics import bootstrap_ci, cluster_bootstrap_ci, gsi

MODELS = {"150M": "allele-balanced-esm2-150m", "650M": "allele-balanced-esm2-650m"}
CONCENTRATED = "mhc_cluster_ci.json"
OPERATOR = "G1"
REPORT_GAPS = (0, 1, 5, 25, 100)
PER_ALLELE_GAPS = (25, 100)
DOMINANT_ALLELE = "HLA-A*02:01"
# Below this many examples an allele's own GSI interval is not reported; its point value is
# still listed and flagged.
MIN_ALLELE_EXAMPLES_FOR_CI = 10


def _paired(run: dict, arm: str, gap: int) -> tuple[np.ndarray, ...]:
    """Example ids, original, geometry and C1 scores on the examples where both applied."""
    t = run["targets"][TARGET]
    n_mut = t["geometry"]["denominator"]["n_mutations"]
    s = t["scores"][arm]
    orig = np.array([np.nan if v is None else v for v in s["original"]])
    c = s["perturbed"][f"C1:{n_mut}"]
    g = s["perturbed"]["G@0" if gap == 0 else f"{OPERATOR}@{gap}"]
    common, ia, ib = np.intersect1d(g["index"], c["index"], return_indices=True)
    return (
        common,
        orig[common],
        np.asarray(g["scores"], dtype=float)[ia],
        np.asarray(c["scores"], dtype=float)[ib],
    )


def per_allele_gsi(
    alleles: np.ndarray, orig: np.ndarray, geo: np.ndarray, c1: np.ndarray, boot: dict, seed: int
) -> list[dict[str, Any]]:
    rows = []
    for allele in np.unique(alleles):
        m = alleles == allele
        row: dict[str, Any] = {"allele": str(allele), "n": int(m.sum()), "gsi": None, "ci": None}
        try:
            row["gsi"] = float(gsi(orig[m], geo[m], c1[m]))
            if m.sum() >= MIN_ALLELE_EXAMPLES_FOR_CI:
                _, lo, hi = bootstrap_ci(
                    gsi, orig[m], geo[m], c1[m], n_boot=boot["n_boot"], alpha=boot["alpha"],
                    seed=seed,
                )  # fmt: skip
                row["ci"] = [float(lo), float(hi)]
        except ValueError as e:
            row["reason"] = str(e)
        rows.append(row)
    return sorted(rows, key=lambda r: -r["n"])


def summarize_alleles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Heterogeneity summary over alleles with enough examples for a stable ratio."""
    solid = [r for r in rows if r["gsi"] is not None and r["n"] >= MIN_ALLELE_EXAMPLES_FOR_CI]
    vals = np.array([r["gsi"] for r in solid])
    out: dict[str, Any] = {
        "min_examples": MIN_ALLELE_EXAMPLES_FOR_CI,
        "n_alleles_listed": len(rows),
        "n_alleles_summarized": len(solid),
    }
    if not solid:
        return out
    out |= {
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "q25_q75": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
        "n_above_1": int((vals > 1).sum()),
        "n_ci_lower_above_1": sum(1 for r in solid if r["ci"] and r["ci"][0] > 1),
        "max_to_min_ratio": float(vals.max() / vals.min()) if vals.min() > 0 else None,
    }
    return out


def analyse_model(run: dict, concentrated: dict | None) -> dict[str, Any]:
    boot = run["config"]["bootstrap"]
    seed = int(run["config"]["seed"])
    ex_alleles = np.array([a or "" for a in run["examples"]["mhc_allele"]])
    out: dict[str, Any] = {"usable": bool(run["targets"][TARGET]["estimator"]["sanity"]["usable"])}
    pooled: dict[str, Any] = {}
    per_allele: dict[str, Any] = {}
    inertness: dict[str, Any] = {}
    population = None
    for arm, label in ARMS.items():
        for gap in REPORT_GAPS:
            idx, orig, geo, c1 = _paired(run, arm, gap)
            alleles = ex_alleles[idx]
            population = idx
            if gap == 0:
                inertness.setdefault(label, {})["max_abs_diff"] = float(np.max(np.abs(orig - geo)))
                continue
            cell = _interval(gsi, orig, geo, c1, clusters=alleles, boot=boot, seed=seed)
            cell["n_pairs"] = int(idx.size)
            keep = alleles != DOMINANT_ALLELE
            _, lo, hi = _interval_excluding(orig, geo, c1, alleles, keep, boot, seed)
            cell["without_dominant_allele"] = {
                "allele": DOMINANT_ALLELE,
                "n_pairs": int(keep.sum()),
                "point": float(gsi(orig[keep], geo[keep], c1[keep])),
                "cluster_ci": [lo, hi],
            }
            if concentrated is not None:
                prior = concentrated[arm]["G1"][str(gap)]
                cell["concentrated_sample"] = {
                    "point": prior["point"],
                    "cluster_ci": prior["cluster_ci"],
                    "balanced_point_inside_concentrated_ci": bool(
                        prior["cluster_ci"][0] <= cell["point"] <= prior["cluster_ci"][1]
                    ),
                    "concentrated_point_inside_balanced_ci": bool(
                        cell["cluster_ci"][0] <= prior["point"] <= cell["cluster_ci"][1]
                    ),
                    "balanced_over_concentrated": cell["point"] / prior["point"],
                }
            pooled.setdefault(label, {})[str(gap)] = cell
            if gap in PER_ALLELE_GAPS:
                rows = per_allele_gsi(alleles, orig, geo, c1, boot, seed)
                per_allele.setdefault(label, {})[str(gap)] = {
                    "alleles": rows,
                    "summary": summarize_alleles(rows),
                }
    out["geometry_population"] = cluster_diagnostics(ex_alleles[population])
    out["gap0_inertness"] = inertness
    out["gsi"] = pooled
    out["per_allele"] = per_allele
    return out


def _interval_excluding(orig, geo, c1, alleles, keep, boot, seed):
    return cluster_bootstrap_ci(
        gsi, orig[keep], geo[keep], c1[keep], clusters=alleles[keep],
        n_boot=boot["n_boot"], alpha=boot["alpha"], seed=seed,
    )  # fmt: skip


def _f(x, spec=".2f") -> str:
    return "" if x is None else format(x, spec)


def write_tables(result: dict[str, Any], out: Path) -> None:
    head = [
        "model", "token_dropout", "gap", "gsi", "cluster_lo", "cluster_hi", "example_lo",
        "example_hi", "n_pairs", "gsi_without_A0201", "without_A0201_lo", "without_A0201_hi",
        "concentrated_gsi", "concentrated_lo", "concentrated_hi",
    ]  # fmt: skip
    rows = ["\t".join(head)]
    for model, m in result["models"].items():
        for arm, gaps in m["gsi"].items():
            for gap, c in gaps.items():
                w, p = c["without_dominant_allele"], c.get("concentrated_sample") or {}
                pc = p.get("cluster_ci") or [None, None]
                vals = [
                    model, arm, gap, _f(c["point"]), _f(c["cluster_ci"][0]),
                    _f(c["cluster_ci"][1]), _f(c["example_ci"][0]), _f(c["example_ci"][1]),
                    str(c["n_pairs"]), _f(w["point"]), _f(w["cluster_ci"][0]),
                    _f(w["cluster_ci"][1]), _f(p.get("point")), _f(pc[0]), _f(pc[1]),
                ]  # fmt: skip
                rows.append("\t".join(vals))
    (out / "allele_balanced.tsv").write_text("\n".join(rows) + "\n")

    forest = ["\t".join(["model", "token_dropout", "gap", "allele", "n", "gsi", "ci_lo", "ci_hi"])]
    for model, m in result["models"].items():
        for arm, gaps in m["per_allele"].items():
            for gap, block in gaps.items():
                for r in block["alleles"]:
                    ci = r["ci"] or [None, None]
                    forest.append(
                        "\t".join(
                            [
                                model,
                                arm,
                                gap,
                                r["allele"],
                                str(r["n"]),
                                _f(r["gsi"]),
                                _f(ci[0]),
                                _f(ci[1]),
                            ]  # fmt: skip
                        )
                    )
    (out / "per_allele_gsi.tsv").write_text("\n".join(forest) + "\n")

    diag = ["\t".join(["model", "n_examples", "n_clusters", "largest_cluster",
                       "largest_cluster_fraction", "kish_n_eff", "per_allele_cap"])]  # fmt: skip
    for model, m in result["models"].items():
        d = m["geometry_population"]
        diag.append(
            "\t".join(
                [
                    model,
                    str(d["n_examples"]),
                    str(d["n_clusters"]),
                    d["largest_cluster"],
                    f"{d['largest_cluster_fraction']:.3f}",
                    f"{d['kish_n_eff']:.2f}",
                    str(m["sampling"].get("per_allele_cap")),
                ]  # fmt: skip
            )
        )
    (out / "allele_balanced_diagnostics.tsv").write_text("\n".join(diag) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--device", choices=["cpu", "mps"], default=None)
    ap.add_argument("--out", type=Path, default=Path("results/phase1/allele_balanced"))
    ap.add_argument("--concentrated", type=Path, default=Path("results/phase1") / CONCENTRATED)
    ap.add_argument("--reuse", action="store_true", help="reanalyse existing nsi.json files")
    args = ap.parse_args(argv)

    specs_path = Path("data/processed/vdjdb_specs.jsonl")
    if not specs_path.exists():
        print(f"error: {specs_path} not found. Run `make data` first.", file=sys.stderr)
        return 2
    specs = None if args.reuse else load_specs_jsonl(specs_path)
    prior = json.loads(args.concentrated.read_text()) if args.concentrated.exists() else None

    result: dict[str, Any] = {
        "target": TARGET,
        "operator": OPERATOR,
        "cluster_key": "mhc_allele",
        "models": {},
    }
    for name in args.models:
        run_name = MODELS[name]
        nsi_path = args.out / run_name / "nsi.json"
        if args.reuse:
            run = json.loads(nsi_path.read_text())
        else:
            config = load_config(Path(f"configs/audit/allele_balanced_{name.lower()}.yaml"))
            run = run_audit(
                config, specs, out_dir=args.out / run_name, runs_root=Path("runs"),
                device=args.device, data_files=[specs_path],
            )  # fmt: skip
            run = json.loads(nsi_path.read_text())
        entry = analyse_model(run, prior["models"][name]["audit"]["gsi"] if prior else None)
        entry["sampling"] = run["data"]["sampling"]
        result["models"][name] = entry
        print(f"done {name}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "allele_balanced.json").write_text(json.dumps(result, indent=2) + "\n")
    write_tables(result, args.out)
    print(f"wrote {args.out / 'allele_balanced.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
