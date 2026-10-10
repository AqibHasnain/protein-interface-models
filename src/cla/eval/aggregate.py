"""Cross-model Gate G1 (GSI rule), GSI table and dose-response shapes from audit results.

    python -m cla.eval.aggregate --results results/phase1

Reads every `<results>/*/nsi.json` and writes `gate_g1_gsi.json` and `gsi_table.tsv` next to
them. The per-model qualification is computed by the audit; this module applies the
"at least two models" half of the rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cla.eval.audit import GSI_MIN_MODELS


def _verdict(per_model: dict[str, bool]) -> dict:
    n = sum(per_model.values())
    if len(per_model) < GSI_MIN_MODELS:
        return {
            "verdict": None,
            "reason": f"only {len(per_model)} model(s) available; rule needs {GSI_MIN_MODELS}",
            "n_models_qualifying": n,
        }
    return {
        "verdict": "PROCEED" if n >= GSI_MIN_MODELS else "NOT_PROCEED",
        "n_models_qualifying": n,
    }


def aggregate(results_dir: Path) -> dict:
    runs = {}
    for f in sorted(results_dir.glob("*/nsi.json")):
        r = json.loads(f.read_text())
        runs[r["run_name"]] = r
    if not runs:
        raise SystemExit(f"no nsi.json found under {results_dir}")

    gate: dict = {"rule": next(iter(runs.values()))["gate_g1"]["gsi"]["rule"], "arms": {}}
    for arm_key in ("token_dropout_on", "token_dropout_off"):
        per_variant = {}
        for variant in ("headline", "all_targets"):
            models = {n: r["gate_g1"]["gsi"][arm_key][variant] for n, r in runs.items()}
            per_variant[variant] = {
                **_verdict({n: m["model_qualifies"] for n, m in models.items()}),
                "per_model": models,
            }
        gate["arms"][arm_key] = per_variant
    gate["preregistered"] = {
        n: {
            t: {"verdict": g["verdict"], "reason": g["reason"]}
            for t, g in r["gate_g1"]["preregistered"].items()
        }
        for n, r in runs.items()
    }
    gate["headline"] = {
        "arm": "token_dropout_on",
        "variant": "headline (PEPTIDE, TCRB)",
        "verdict": gate["arms"]["token_dropout_on"]["headline"]["verdict"],
    }

    shapes = []
    table = ["model\ttarget\toperator\tgap\ttoken_dropout\tmedian_abs_delta\tc1_denominator\t"
             "gsi\tgsi_ci_lo\tgsi_ci_hi\tn_pairs"]  # fmt: skip
    for name, r in runs.items():
        for tname, t in r["targets"].items():
            for op, entry in t["geometry"]["per_operator"].items():
                for arm in ("true", "false"):
                    d = entry["dose_response"][arm]
                    slope = d.get("loglog_slope") or {}
                    shapes.append(
                        {
                            "model": name,
                            "target": tname,
                            "operator": op,
                            "token_dropout": arm,
                            "shape": d["shape"],
                            "loglog_slope": slope.get("value"),
                            "slope_ci_lo": slope.get("ci_lo"),
                            "slope_ci_hi": slope.get("ci_hi"),
                            "fraction_of_reference": d.get("fraction_of_reference"),
                            "reason": d.get("reason"),
                        }
                    )
                for gap, per_arm in entry["gaps"].items():
                    for arm in ("true", "false"):
                        e = per_arm[arm]
                        g = e["gsi"]
                        vals = [e["median_abs_delta"], e["c1_denominator"], g["value"],
                                g["ci_lo"], g["ci_hi"]]  # fmt: skip
                        cells = ["" if v is None else f"{v:.5g}" for v in vals]
                        table.append(
                            "\t".join([name, tname, op, gap, arm, *cells, str(e["n_pairs"])])
                        )
    gate["dose_response_shapes"] = shapes
    (results_dir / "gate_g1_gsi.json").write_text(json.dumps(gate, indent=1) + "\n")
    (results_dir / "gsi_table.tsv").write_text("\n".join(table) + "\n")
    return gate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/phase1"))
    args = ap.parse_args(argv)
    gate = aggregate(args.results)
    print(json.dumps(gate["headline"], indent=1))
    for arm_key, variants in gate["arms"].items():
        for v, g in variants.items():
            print(arm_key, v, g["verdict"], g["n_models_qualifying"], g.get("reason", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
