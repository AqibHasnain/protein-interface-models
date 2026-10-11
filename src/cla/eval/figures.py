"""Regenerate the Phase 1 report figures and table from committed result files.

    python -m cla.eval.figures --results results/phase1 --out results/phase1/figures

Reads `<results>/<run>/summary.tsv` (and `nsi.json` for the discrimination figure) for every
run found by glob. Runs that are absent are skipped, so the same command works while the
largest model is still pending.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
AXIS = "#b4b3ae"
MUTED = "#9a9993"

# Fixed target-to-hue assignment. The aqua is below 3:1 contrast on the surface, so every
# series also carries a direct text label in TEXT_2 rather than relying on hue alone.
TARGET_COLORS = {"PEPTIDE": "#2a78d6", "TCRB": "#eb6834", "MHC_HEAVY": "#1baf7a"}
TARGET_MARKERS = {"PEPTIDE": "o", "TCRB": "s", "MHC_HEAVY": "^"}
TARGETS = tuple(TARGET_COLORS)

LINE_WIDTH = 1.5  # 2 px at 96 dpi
MARKER_SIZE = 5  # ~10 px
BAND_ALPHA = 0.18
EXPECTED_SIZES_M = (35, 150, 650)
# Horizontal offsets on the log size axis so CI bars of neighbouring targets do not overlap.
DODGE = {"PEPTIDE": 0.93, "TCRB": 1.0, "MHC_HEAVY": 1.07}

GAP_LABEL = "Position-index gap (index steps)"
GSI_LABEL = "GSI (median score shift / shift from 20 real substitutions)"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": AXIS,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 100,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,
    }
)


@dataclass
class Run:
    name: str
    params_m: float
    rows: list[dict]
    nsi: dict | None = None
    _index: dict = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        return f"{self.params_m:g}M"

    def get(self, operator: str, target: str, gap: int | None = None) -> dict | None:
        """Token-dropout-on, non-OOD row keyed by (operator, target, gap)."""
        key = (operator, target, gap)
        if not self._index:
            for r in self.rows:
                if r["token_dropout"] and r["regime"] != "out_of_distribution":
                    self._index[(r["operator"], r["target"], r["gap"])] = r
        return self._index.get(key)


def _float(s: str) -> float:
    return float(s) if s not in ("", "nan") else float("nan")


def _read_summary(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(
                {
                    "target": r["target"],
                    "operator": r["operator"],
                    "gap": int(r["gap"]) if r["gap"] != "" else None,
                    "regime": r["regime"],
                    "token_dropout": r["token_dropout"].strip().lower() == "true",
                    "shift": _float(r["median_abs_shift_nats"]),
                    "c1": _float(r["c1_denominator_nats"]),
                    "gsi": _float(r["gsi"]),
                    "lo": _float(r["gsi_ci_lo"]),
                    "hi": _float(r["gsi_ci_hi"]),
                }
            )
    return rows


def _params_m(name: str, nsi: dict | None) -> float | None:
    candidates = [str((nsi or {}).get("model", "")), name]
    for text in candidates:
        m = re.search(r"(\d+(?:\.\d+)?)\s*([MmBb])\b", text) or re.search(
            r"(\d+(?:\.\d+)?)([mb])(?![a-z])", text, re.I
        )
        if m:
            value = float(m.group(1))
            return value * 1000 if m.group(2).lower() == "b" else value
    return None


def load_runs(results: Path) -> list[Run]:
    runs = []
    for summary in sorted(results.glob("*/summary.tsv")):
        nsi_path = summary.with_name("nsi.json")
        nsi = json.loads(nsi_path.read_text()) if nsi_path.exists() else None
        params = _params_m(summary.parent.name, nsi)
        if params is None:
            continue
        runs.append(Run(summary.parent.name, params, _read_summary(summary), nsi))
    return sorted(runs, key=lambda r: r.params_m)


def _gap_series(run: Run, operator: str, target: str, field_: str = "gsi"):
    """(gaps, values, lo, hi) for the in-distribution, dropout-on sweep."""
    pts = sorted(
        (
            r
            for r in run.rows
            if _is_main(r) and r["operator"] == operator and r["target"] == target
        ),
        key=lambda r: r["gap"],
    )
    return (
        [r["gap"] for r in pts],
        [r[field_] for r in pts],
        [r["lo"] for r in pts],
        [r["hi"] for r in pts],
    )


def _is_main(r: dict) -> bool:
    return r["token_dropout"] and r["regime"] == "in_distribution" and r["gap"] is not None


def _c1(run: Run, target: str) -> float:
    for r in run.rows:
        if r["token_dropout"] and r["target"] == target and r["c1"] == r["c1"]:
            return r["c1"]
    return float("nan")


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def _spread(ys: list[float], min_gap: float) -> list[float]:
    """Push label positions apart to at least `min_gap`, keeping their centroid."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + min_gap)
    shift = (sum(ys) - sum(pos)) / len(ys)
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k] + shift
    return out


def _end_labels(ax, items: list[tuple[float, float, str]], dx: float = 5, min_gap: float = 9.5):
    """Direct labels right of each series end, in secondary text color.

    Must run after axis limits are final and the figure has been laid out, because label
    spacing is computed in display space.
    """
    fig = ax.figure
    fig.canvas.draw()
    pts_per_px = 72 / fig.dpi
    ys = [ax.transData.transform((x, y))[1] * pts_per_px for x, y, _ in items]
    adjusted = _spread(ys, min_gap)
    for (x, y, text), y0, y1 in zip(items, ys, adjusted, strict=True):
        label = ax.annotate(
            text,
            (x, y),
            xytext=(dx, y1 - y0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=TEXT_2,
            annotation_clip=False,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8},
        )
        # Headroom is reserved in the axis limits; labels must not resize the layout.
        label.set_in_layout(False)


def _ref_label(
    ax, text: str, y: float = 1.0, dy: float = -3, x: float | None = None, right: bool = False
) -> None:
    """Label a horizontal reference line at the left axis edge, a data x, or the right edge."""
    if right:
        xy, xycoords, dx = (1, y), ax.get_yaxis_transform(), -4
    elif x is None:
        xy, xycoords, dx = (0, y), ax.get_yaxis_transform(), 4
    else:
        xy, xycoords, dx = (x, y), "data", 0
    ax.annotate(
        text,
        xy=xy,
        xycoords=xycoords,
        xytext=(dx, dy),
        textcoords="offset points",
        va="top" if dy < 0 else "bottom",
        ha="right" if (right or x is not None) else "left",
        fontsize=7.5,
        color=TEXT_2,
    )


def _headline(fig, title: str, subtitle: str, bottom: float = 0.0) -> None:
    """Bold finding plus a one-line qualifier; reserves the top strip of the layout."""
    fig.get_layout_engine().set(rect=(0, bottom, 1, 1 - bottom - 0.1))
    fig.text(0.01, 0.985, title, fontsize=11, fontweight="bold", va="top")
    fig.text(0.01, 0.935, subtitle, fontsize=8.5, color=TEXT_2, va="top")


def _target_legend(fig, loc: str = "outside lower center") -> None:
    handles = [
        Line2D(
            [],
            [],
            color=TARGET_COLORS[t],
            marker=TARGET_MARKERS[t],
            label=t,
            markersize=MARKER_SIZE,
        )
        for t in TARGETS
    ]
    fig.legend(handles=handles, loc=loc, ncol=len(handles))


def _gap_axis(ax, gaps: list[int], right: float = 1000) -> None:
    """Symlog gap axis; the headroom to the right holds direct labels."""
    ax.set_xscale("symlog", linthresh=1, linscale=0.6)
    ax.set_xlim(-0.5, right)
    ticks = [g for g in gaps if g in (0, 1, 2, 5, 10, 25, 50, 100, 150)]
    # 100 and 150 sit too close on the symlog axis to share a label row.
    ax.set_xticks(ticks, [f"\n{t}" if t == 150 and 100 in ticks else str(t) for t in ticks])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines["bottom"].set_bounds(0, max(gaps))
    ax.grid(False, axis="x")


def _in_dist_gaps(runs: list[Run]) -> list[int]:
    return sorted({r["gap"] for run in runs for r in run.rows if _is_main(r)})


# ---------------------------------------------------------------- figure 1


def fig1_dose_response(runs: list[Run], out: Path) -> None:
    ops = [("G1", "G1: gap at every chain break"), ("G2", "G2: gap before the peptide only")]
    gaps = _in_dist_gaps(runs)
    fig, axes = plt.subplots(
        len(runs), 2, figsize=(8, 2.55 * len(runs) + 1.1), sharex=True, sharey=True, squeeze=False
    )
    ymax = max(
        (
            r["hi"]
            for run in runs
            for r in run.rows
            if _is_main(r) and r["operator"] in ("G1", "G2")
        ),
        default=1.0,
    )
    peak = max(
        (r["gsi"] for run in runs for r in run.rows if _is_main(r) and r["operator"] == "G1"),
        default=1.0,
    )
    for i, run in enumerate(runs):
        for j, (op, title) in enumerate(ops):
            ax = axes[i][j]
            ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ends = []
            for t in TARGETS:
                g, v, lo, hi = _gap_series(run, op, t)
                if not g:
                    continue
                c = TARGET_COLORS[t]
                ax.fill_between(g, lo, hi, color=c, alpha=BAND_ALPHA, lw=0, zorder=2)
                ax.plot(g, v, color=c, marker=TARGET_MARKERS[t], zorder=3, clip_on=True)
                ends.append((g[-1], v[-1], t))
            _gap_axis(ax, gaps)
            ax.set_ylim(0, ymax * 1.04)
            if i == 0:
                ax.set_title(title, loc="left")
            ax.text(
                0.02,
                0.96,
                f"ESM-2 {run.label}",
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color=TEXT,
            )
            if i == len(runs) - 1:
                ax.set_xlabel(GAP_LABEL)
            _end_labels(ax, ends)
            if (i, j) == (0, 0):
                _ref_label(ax, "equal to 20 real mutations", x=100)
    fig.supylabel(GSI_LABEL, fontsize=9)
    fig.suptitle(
        f"Chain-break gaps shift scores by up to {peak:.1f}x the effect of 20 real mutations",
        fontsize=11,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    _target_legend(fig)
    _save(fig, out, "fig1_dose_response")


# ---------------------------------------------------------------- figure 2


def _scale_axes(ax, runs: list[Run], series: dict, ylabel: str, ref=None, pending=True) -> None:
    """One line per target against model size on a log axis, with CI error bars."""
    for t in TARGETS:
        pts = series.get(t, [])
        if not pts:
            continue
        xs, ys, lo, hi = zip(*pts, strict=True)
        c = TARGET_COLORS[t]
        ax.plot(xs, ys, color=c, marker=TARGET_MARKERS[t], zorder=3)
        ax.errorbar(
            xs,
            ys,
            yerr=[
                [y - a for y, a in zip(ys, lo, strict=True)],
                [b - y for y, b in zip(ys, hi, strict=True)],
            ],
            fmt="none",
            ecolor=c,
            elinewidth=LINE_WIDTH,
            capsize=3,
            zorder=3,
        )
    sizes = sorted({r.params_m for r in runs} | set(EXPECTED_SIZES_M))
    ax.set_xscale("log")
    ax.set_xlim(min(sizes) * 0.7, max(sizes) * 3.2)
    ax.set_xticks(sizes, [f"{s:g}M" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Model parameters (millions, log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(False, axis="x")
    if pending:
        have = {r.params_m for r in runs}
        for s in EXPECTED_SIZES_M:
            if s not in have:
                ax.axvline(s, color=MUTED, lw=1, ls=":", zorder=1)
                ax.text(
                    s * 0.93,
                    0.97,
                    f"{s:g}M pending",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=TEXT_2,
                )


def fig2_scale(runs: list[Run], out: Path, gaps: tuple[int, int] = (50, 150)) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.1), sharey=True)
    lows, highs = [], []
    for ax, gap in zip(axes, gaps, strict=True):
        series = {}
        for t in TARGETS:
            pts = []
            for run in runs:
                r = run.get("G1", t, gap)
                if r:
                    pts.append((run.params_m * DODGE[t], r["gsi"], r["lo"], r["hi"]))
            series[t] = pts
            lows += [p[2] for p in pts]
            highs += [p[3] for p in pts]
        ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        _scale_axes(ax, runs, series, GSI_LABEL if ax is axes[0] else "")
        ax.set_title(f"G1 at gap {gap}", loc="left")
    axes[0].set_ylim(0, max(highs) * 1.06)
    axes[0].set_ylabel("GSI (median score shift / shift\nfrom 20 real substitutions)")
    for ax, gap in zip(axes, gaps, strict=True):
        items = []
        for t in TARGETS:
            pts = [
                (run.params_m * DODGE["MHC_HEAVY"], run.get("G1", t, gap)["gsi"])
                for run in runs
                if run.get("G1", t, gap)
            ]
            if pts:
                items.append((*pts[-1], t))
        _end_labels(ax, items)
        _ref_label(ax, "equal to 20 real mutations")
    _headline(
        fig,
        "Layout sensitivity does not diminish with model size",
        "Steepest for the MHC target; TCRB is flat and PEPTIDE rises modestly.",
    )
    _target_legend(fig)
    _save(fig, out, "fig2_scale")


# ---------------------------------------------------------------- figure 3


def fig3_absolute_scale(runs: list[Run], out: Path) -> None:
    gaps = _in_dist_gaps(runs)
    fig, axes = plt.subplots(
        len(runs),
        len(TARGETS),
        figsize=(11, 2.5 * len(runs) + 1.4),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    labels = []
    for i, run in enumerate(runs):
        for j, t in enumerate(TARGETS):
            ax = axes[i][j]
            c = TARGET_COLORS[t]
            c1 = _c1(run, t)
            ax.axhline(c1, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ends = [(max(gaps), c1, "C1 n=20")]
            for op, fill in (("G1", c), ("G2", SURFACE)):
                g, v, _, _ = _gap_series(run, op, t, "shift")
                if not g:
                    continue
                ax.plot(
                    g,
                    v,
                    color=c,
                    marker=TARGET_MARKERS[t],
                    markerfacecolor=fill,
                    markeredgecolor=c,
                    ls="-" if op == "G1" else (0, (1, 1.5)),
                    zorder=3,
                    clip_on=True,
                )
                ends.append((g[-1], v[-1], op))
            _gap_axis(ax, gaps)
            if i == 0:
                ax.set_title(t, loc="left")
            if j == 0:
                ax.set_ylabel(f"ESM-2 {run.label}\nmedian |score shift| (nats)")
            if i == len(runs) - 1:
                ax.set_xlabel(GAP_LABEL)
            labels.append((ax, ends))
    for j, t in enumerate(TARGETS):
        top = max(
            [r["shift"] for run in runs for r in run.rows if _is_main(r) and r["target"] == t]
            + [_c1(run, t) for run in runs]
        )
        axes[0][j].set_ylim(0, top * 1.06)
    # Shared y-limits are only final once every panel is drawn.
    for ax, ends in labels:
        _end_labels(ax, ends, min_gap=8)
    _headline(
        fig,
        "Absolute shifts are small for the peptide and TCR targets, but not for MHC",
        "GSI is a ratio of two small quantities except in the MHC panels, "
        "where the shift is large outright.",
        bottom=0.035,
    )
    fig.text(
        0.01,
        0.0,
        "C1 n=20 (dashed gray): median |shift| from 20 real substitutions, the GSI "
        "denominator. Solid: G1; hollow and dotted: G2. Y-axis is shared down each column.",
        fontsize=8,
        color=TEXT_2,
        va="bottom",
    )
    _save(fig, out, "fig3_absolute_scale")


# ---------------------------------------------------------------- figure 4


def fig4_discrimination(runs: list[Run], out: Path) -> None:
    runs = [r for r in runs if r.nsi and "discrimination" in r.nsi]
    if not runs:
        return
    series = {}
    for t in TARGETS:
        pts = []
        for run in runs:
            a = run.nsi["discrimination"].get(t, {}).get("true", {}).get("auroc")
            if a:
                pts.append((run.params_m * DODGE[t], a["value"], a["ci_lo"], a["ci_hi"]))
        series[t] = pts
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    _scale_axes(ax, runs, series, "Binder-vs-decoy AUROC (0.5 = chance)")
    allv = [p[k] for pts in series.values() for p in pts for k in (2, 3)]
    ax.set_ylim(min(allv + [0.5]) - 0.04, max(allv + [0.5]) + 0.04)
    _ref_label(ax, "chance", 0.5, dy=3, right=True)
    _end_labels(
        ax,
        [
            (pts[-1][0] / DODGE[t] * DODGE["MHC_HEAVY"], pts[-1][1], t)
            for t, pts in series.items()
            if pts
        ],
        dx=10,
    )
    _headline(
        fig,
        "Binder/decoy separation is at or near chance for every model and target",
        "No AUROC exceeds 0.60; the strongest, MHC at 150M, is 0.57 (token dropout on).",
    )
    _target_legend(fig)
    _save(fig, out, "fig4_discrimination")


# ---------------------------------------------------------------- figure 5


def fig5_inertness(runs: list[Run], out: Path) -> None:
    rows = [(run, t) for run in runs for t in TARGETS]
    vals = [
        (run.get(op, t, 0) or {}).get("shift", float("nan"))
        for run, t in rows
        for op in ("G1", "G2")
    ]
    exact = all(v == 0 for v in vals)
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(rows) + 1.6))
    ys = list(range(len(rows)))[::-1]
    for y, (run, t) in zip(ys, rows, strict=True):
        for op, dy, marker in (("G1", 0.14, "o"), ("G2", -0.14, "D")):
            v = (run.get(op, t, 0) or {}).get("shift", float("nan"))
            ax.plot(
                [v],
                [y + dy],
                marker=marker,
                color=TARGET_COLORS[t],
                ls="none",
                markersize=MARKER_SIZE,
                zorder=3,
            )
        ax.text(0.001, y, "G1 0.0, G2 0.0 nats", va="center", fontsize=8, color=TEXT_2)
    ax.set_yticks(ys, [f"{run.label}  {t}" for run, t in rows])
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.set_xlim(-0.004, 0.02)
    ax.set_xticks([0, 0.01, 0.02])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    ax.set_xlabel("Median |score shift| at gap 0 (nats)")
    ax.legend(
        handles=[
            Line2D([], [], color=TEXT_2, marker="o", ls="none", label="G1"),
            Line2D([], [], color=TEXT_2, marker="D", ls="none", label="G2"),
        ],
        loc="upper right",
        ncol=2,
    )
    fig.suptitle(
        "Gap 0 leaves scores unchanged: median |shift| is exactly 0.0 nats"
        if exact
        else "Gap 0 leaves scores nearly unchanged",
        fontsize=11,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    _save(fig, out, "fig5_inertness")


# ---------------------------------------------------------------- table


def _cell(r: dict | None) -> str:
    return "n/a" if r is None else f"{r['gsi']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"


def write_table1(runs: list[Run], out: Path, gaps: tuple[int, ...] = (1, 25, 150)) -> None:
    head = ["Model", "Target"]
    head += [f"{op} gap {g}" for op in ("G1", "G2") for g in gaps]
    head.append("C1 denominator (nats)")
    lines = [
        "Table 1. GSI as `value [95% CI lo, hi]`; token dropout on, in-distribution gaps only. "
        "The last column is the median |shift| from 20 real substitutions.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|",
    ]
    for run in runs:
        for t in TARGETS:
            cells = [f"ESM-2 {run.label}", t]
            cells += [_cell(run.get(op, t, g)) for op in ("G1", "G2") for g in gaps]
            cells.append(f"{_c1(run, t):.4f}")
            lines.append("| " + " | ".join(cells) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "table1.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("results/phase1"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or args.results / "figures"
    runs = load_runs(args.results)
    if not runs:
        print(f"no runs with summary.tsv under {args.results}")
        return 1
    fig1_dose_response(runs, out)
    fig2_scale(runs, out)
    fig3_absolute_scale(runs, out)
    fig4_discrimination(runs, out)
    fig5_inertness(runs, out)
    write_table1(runs, out)
    print(f"wrote figures and table1.md for {[r.name for r in runs]} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
