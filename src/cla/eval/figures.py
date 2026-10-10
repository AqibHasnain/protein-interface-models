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
NSI_C_LABEL = "NSI-C (median |score shift| / median |P5 score shift|, unitless)"
GATE_NSI = 0.25


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def nsi_c_bars(result: dict, target: dict):
    ops = [op for op, e in target["operators"].items() if e["kind"] == "null"]
    fig, ax = plt.subplots(figsize=(1.6 * len(ops) + 2.5, 4.2))
    width = 0.38
    for j, arm in enumerate(("true", "false")):
        label, color = ARM_STYLE[arm]
        xs, vals, lo, hi = [], [], [], []
        for i, op in enumerate(ops):
            n = target["operators"][op]["token_dropout"][arm]["nsi_c"]
            x = i + (j - 0.5) * width
            if n["value"] is None:
                ax.text(x, 0, "n/a", ha="center", va="bottom", fontsize=8, color=GRAY)
                continue
            xs.append(x)
            vals.append(n["value"])
            lo.append(n["value"] - n["ci_lo"])
            hi.append(n["ci_hi"] - n["value"])
        ax.bar(xs, vals, width, color=color, label=label)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1)
    ax.axhline(GATE_NSI, color=VERMILION, ls=":", lw=1.2)
    ax.set_ylim(bottom=0)
    ax.text(-0.55, 1.0, "NSI-C = 1", va="bottom", ha="left", fontsize=8, color=GRAY)
    ax.text(-0.55, GATE_NSI, "gate G1 = 0.25", va="bottom", ha="left", fontsize=8, color=VERMILION)
    ax.set_xlim(-0.6, len(ops) - 0.4)
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels(ops)
    ax.set_ylabel(NSI_C_LABEL, fontsize=8)
    ax.set_title(
        f"{result['run_name']} [{'+'.join(target['target_roles'])}]: "
        "NSI-C by operator (95% bootstrap CI)",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
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


def worst_operator_scatter(result: dict, target: dict):
    """Original vs perturbed score for the null operator with the highest NSI-C (dropout on)."""
    candidates = {
        op: e["token_dropout"]["true"]["nsi_c"]["value"]
        for op, e in target["operators"].items()
        if e["kind"] == "null" and e["token_dropout"]["true"]["nsi_c"]["value"] is not None
    }
    if not candidates:
        return None
    worst = max(candidates, key=candidates.get)
    pert = target["scores"]["true"]["perturbed"][worst]
    idx = np.asarray(pert["index"])
    orig = np.asarray(target["scores"]["true"]["original"])[idx]
    new = np.asarray(pert["scores"])
    labels = np.asarray(result["examples"]["label"])[idx]
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, name, color, marker in ((1, "binder", BLUE, "o"), (0, "decoy", VERMILION, "s")):
        m = labels == lab
        ax.scatter(orig[m], new[m], s=14, alpha=0.7, color=color, marker=marker, label=name)
    lim = [min(orig.min(), new.min()), max(orig.max(), new.max())]
    ax.plot(lim, lim, color=GRAY, lw=1, label="no change")
    tname = "+".join(target["target_roles"])
    ax.set_xlabel(f"{tname} span log-likelihood, original layout (nats per residue)")
    ax.set_ylabel(f"{tname} span log-likelihood, after {worst} (nats per residue)")
    ax.set_title(f"{result['run_name']}: highest-NSI-C null operator {worst}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
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
    for f in files:
        result = json.loads(f.read_text())
        name = result["run_name"]
        _save(discrimination_auroc(result), out, f"{name}_discrimination_auroc")
        for tname, target in result["targets"].items():
            stem = f"{name}_{tname.lower()}"
            _save(nsi_c_bars(result, target), out, f"{stem}_nsi_c_by_operator")
            for fig, suffix in (
                (length_nsi(result, target), "nsi_by_cdr3b_length"),
                (worst_operator_scatter(result, target), "worst_operator_scatter"),
            ):
                if fig is not None:
                    _save(fig, out, f"{stem}_{suffix}")
        print(f"figures for {name} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
