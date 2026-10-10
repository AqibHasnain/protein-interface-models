"""Regenerate Phase 1 figures from the committed nsi.json files only.

    python -m cla.eval.figures --results results/phase1

Writes PNG and SVG per figure to `<results>/figures/`. The primary figure is the GSI
dose-response across models; per-run figures follow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito, distinguishable under common color-vision deficiencies.
BLUE, ORANGE, GREEN, VERMILION, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#555555"
ARM_STYLE = {"true": ("token dropout on (published)", BLUE), "false": ("token dropout off", ORANGE)}
OP_COLORS = {"P1": BLUE, "P2": ORANGE, "P3": GREEN, "P4": VERMILION, "P5": "#CC79A7"}
GATE_NSI = 0.25


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


TARGET_STYLE = {
    "PEPTIDE": (BLUE, "o", "-"),
    "TCRB": (ORANGE, "s", "--"),
    "MHC_HEAVY": (VERMILION, "^", ":"),
}
GSI_LABEL = "GSI (unitless)"
GAP_LABEL = "Position-index gap added (steps)"


def _gsi_panel(ax, result: dict, op: str, arm: str) -> None:
    """GSI against gap for one operator and arm, one line per target with a CI band."""
    for tname, target in result["targets"].items():
        color, marker, ls = TARGET_STYLE.get(tname, (GRAY, "o", "-"))
        gaps = target["geometry"]["per_operator"][op]["gaps"]
        xs = [int(g) for g in gaps]
        ys = [gaps[g][arm]["gsi"]["value"] for g in gaps]
        lo = [gaps[g][arm]["gsi"]["ci_lo"] for g in gaps]
        hi = [gaps[g][arm]["gsi"]["ci_hi"] for g in gaps]
        if any(v is None for v in ys):
            continue
        est = target["estimator"]["name"]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(xs, ys, color=color, marker=marker, ls=ls, ms=4, lw=1.4, label=f"{tname} ({est})")
    ax.axhline(1.0, color="black", lw=1, ls="-", alpha=0.6)
    ax.set_xscale("symlog", linthresh=1, linscale=0.4)
    ax.set_yscale("symlog", linthresh=1, linscale=0.8)
    gaps_all = [
        int(g) for g in next(iter(result["targets"].values()))["geometry"]["gaps_in_distribution"]
    ]
    ax.set_xticks(gaps_all)
    ax.set_xticklabels([str(g) for g in gaps_all], fontsize=7, rotation=60)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=-0.25)
    ax.set_xlabel(GAP_LABEL, fontsize=8)
    _style(ax)


def gsi_dose_response(result: dict):
    """Both operators and both token-dropout arms for one run."""
    ops = list(next(iter(result["targets"].values()))["geometry"]["per_operator"])
    fig, axes = plt.subplots(2, len(ops), figsize=(4.6 * len(ops), 7), squeeze=False)
    for j, op in enumerate(ops):
        for i, arm in enumerate(("true", "false")):
            ax = axes[i][j]
            _gsi_panel(ax, result, op, arm)
            ax.set_title(f"{op}, token dropout {'on' if arm == 'true' else 'off'}", fontsize=9)
            if j == 0:
                ax.set_ylabel(GSI_LABEL, fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{result['run_name']}: GSI dose-response (95% paired bootstrap CI; line at GSI = 1)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def gsi_dose_response_all_models(results: list[dict]):
    """Primary figure: published (token dropout on) arm, rows are operators, columns models."""
    ops = list(next(iter(results[0]["targets"].values()))["geometry"]["per_operator"])
    fig, axes = plt.subplots(
        len(ops), len(results), figsize=(4.4 * len(results), 3.6 * len(ops)), squeeze=False
    )
    for i, op in enumerate(ops):
        for j, res in enumerate(results):
            ax = axes[i][j]
            _gsi_panel(ax, res, op, "true")
            ax.set_title(f"{res['model']}, {op}", fontsize=9)
            if j == 0:
                ax.set_ylabel(GSI_LABEL, fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Geometry Sensitivity Index vs position gap, token dropout on "
        "(95% paired bootstrap CI)\n"
        "GSI = median |shift| from geometry / median |shift| from 20 substitutions; "
        "GSI >= 1 means the artifact exceeds that much real biology",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def length_nsi(result: dict, target: dict):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted = False
    for op, entry in target["operators"].items():
        for arm, ls in (("true", "-"), ("false", "--")):
            strat = entry["token_dropout"][arm].get("length_stratified_nsi")
            if not strat:
                continue
            labels = list(strat)
            ys = [strat[k]["nsi"] if strat[k]["nsi"] is not None else np.nan for k in labels]
            ax.plot(
                range(len(labels)),
                ys,
                ls=ls,
                marker="o",
                color=OP_COLORS.get(op, GRAY),
                label=f"{op}, dropout {'on' if arm == 'true' else 'off'}",
            )
            plotted = True
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"{k}\nn={strat[k]['n']}" for k in labels], fontsize=8)
    if not plotted:
        plt.close(fig)
        return None
    ax.axhline(GATE_NSI, color=VERMILION, ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("CDR3 beta length bucket (residues); buckets under the minimum n are omitted")
    ax.set_ylabel("NSI (unitless, log scale)")
    ax.set_title(f"{result['run_name']}: pre-registered NSI by CDR3 beta length", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _style(ax)
    return fig


def discrimination_auroc(result: dict):
    """Raw-score AUROC for binder vs decoy per target and dropout arm, with 95% CI."""
    names = list(result["targets"])
    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 2.5, 4.0))
    width = 0.38
    for j, arm in enumerate(("true", "false")):
        label, color = ARM_STYLE[arm]
        for i, name in enumerate(names):
            a = result["discrimination"][name][arm]["auroc"]
            if a["value"] is None:
                continue
            x = i + (j - 0.5) * width
            ax.errorbar(
                x,
                a["value"],
                yerr=[[a["value"] - a["ci_lo"]], [a["ci_hi"] - a["value"]]],
                fmt="o",
                color=color,
                capsize=3,
                label=label if i == 0 else None,
            )
    ax.axhline(0.5, color=GRAY, ls="--", lw=1)
    ax.set_ylim(0.3, 0.8)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("AUROC of raw span score, binder vs decoy")
    ax.set_title(f"{result['run_name']}: zero-shot discrimination (95% bootstrap CI)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/phase1"))
    args = ap.parse_args(argv)

    files = sorted(args.results.glob("*/nsi.json"))
    if not files:
        raise SystemExit(f"no nsi.json found under {args.results}; run the audit first")
    out = args.results / "figures"
    results = [json.loads(f.read_text()) for f in files]
    results.sort(
        key=lambda r: (
            ("35M", "150M", "650M").index(r["model"])
            if r["model"] in ("35M", "150M", "650M")
            else 9
        )
    )
    _save(gsi_dose_response_all_models(results), out, "gsi_dose_response_all_models")
    for result in results:
        name = result["run_name"]
        _save(gsi_dose_response(result), out, f"{name}_gsi_dose_response")
        _save(discrimination_auroc(result), out, f"{name}_discrimination_auroc")
        for tname, target in result["targets"].items():
            fig = length_nsi(result, target)
            if fig is not None:
                _save(fig, out, f"{name}_{tname.lower()}_nsi_by_cdr3b_length")
        print(f"figures for {name} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
