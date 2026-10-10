"""Regenerate Phase 1 figures from the committed nsi.json files only.

    python -m cla.eval.figures --results results/phase1

Writes PNG and SVG per figure to `<results>/figures/`, one set per run.
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
NSI_LABEL = "NSI (median |score shift| / binder-decoy median gap, unitless)"
GATE_NSI = 0.25


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def nsi_bars(result: dict):
    ops = list(result["operators"])
    fig, ax = plt.subplots(figsize=(1.6 * len(ops) + 2.5, 4.2))
    width = 0.38
    for j, arm in enumerate(("true", "false")):
        label, color = ARM_STYLE[arm]
        xs, vals, lo, hi = [], [], [], []
        for i, op in enumerate(ops):
            n = result["operators"][op]["token_dropout"][arm]["nsi"]
            x = i + (j - 0.5) * width
            if n["value"] is None:
                ax.text(x, 0, "n/a", ha="center", va="bottom", fontsize=8, color=GRAY)
                continue
            xs.append(x)
            vals.append(n["value"])
            # Bootstrap CIs on a ratio with a small denominator need not bracket the point.
            lo.append(max(n["value"] - n["ci_lo"], 0.0))
            hi.append(max(n["ci_hi"] - n["value"], 0.0))
        ax.bar(xs, vals, width, color=color, label=label)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1)
    ax.axhline(GATE_NSI, color=VERMILION, ls=":", lw=1.2)
    # Linear below 1 so the gate line stays readable, logarithmic above for large NSI.
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(bottom=0)
    ax.text(-0.55, 1.0, "NSI = 1", va="bottom", ha="left", fontsize=8, color=GRAY)
    ax.text(-0.55, GATE_NSI, "gate G1 = 0.25", va="bottom", ha="left", fontsize=8, color=VERMILION)
    ax.set_xlim(-0.6, len(ops) - 0.4)
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels(
        [f"{op}\n({result['operators'][op]['kind'].replace('_', ' ')})" for op in ops]
    )
    ax.set_ylabel(NSI_LABEL + "; symlog axis", fontsize=8)
    ax.set_title(f"{result['run_name']}: NSI by operator (95% bootstrap CI)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return fig


def length_nsi(result: dict):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for op, entry in result["operators"].items():
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
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"{k}\nn={strat[k]['n']}" for k in labels], fontsize=8)
    ax.axhline(GATE_NSI, color=VERMILION, ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("CDR3 beta length bucket (residues); buckets under the minimum n are omitted")
    ax.set_ylabel("NSI (unitless, log scale)")
    ax.set_title(f"{result['run_name']}: NSI by CDR3 beta length", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _style(ax)
    return fig


def worst_operator_scatter(result: dict):
    """Original vs perturbed score for the null operator with the highest NSI (dropout on)."""
    candidates = {
        op: e["token_dropout"]["true"]["nsi"]["value"]
        for op, e in result["operators"].items()
        if e["kind"] == "null" and e["token_dropout"]["true"]["nsi"]["value"] is not None
    }
    if not candidates:
        return None
    worst = max(candidates, key=candidates.get)
    pert = result["scores"]["true"]["perturbed"][worst]
    idx = np.asarray(pert["index"])
    orig = np.asarray(result["scores"]["true"]["original"])[idx]
    new = np.asarray(pert["scores"])
    labels = np.asarray(result["examples"]["label"])[idx]
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, name, color, marker in ((1, "binder", BLUE, "o"), (0, "decoy", VERMILION, "s")):
        m = labels == lab
        ax.scatter(orig[m], new[m], s=14, alpha=0.7, color=color, marker=marker, label=name)
    lim = [min(orig.min(), new.min()), max(orig.max(), new.max())]
    ax.plot(lim, lim, color=GRAY, lw=1, label="no change")
    ax.set_xlabel("Peptide log-likelihood, original layout (nats per residue)")
    ax.set_ylabel(f"Peptide log-likelihood, after {worst} (nats per residue)")
    ax.set_title(f"{result['run_name']}: worst null operator {worst}", fontsize=10)
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
    for f in files:
        result = json.loads(f.read_text())
        name = result["run_name"]
        _save(nsi_bars(result), out, f"{name}_nsi_by_operator")
        _save(length_nsi(result), out, f"{name}_nsi_by_cdr3b_length")
        scatter = worst_operator_scatter(result)
        if scatter is not None:
            _save(scatter, out, f"{name}_worst_operator_scatter")
        print(f"figures for {name} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
