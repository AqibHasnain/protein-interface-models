"""Regenerate the Phase 1 report figures and table from committed result files.

    python -m cla.eval.figures --results results/phase1 --out results/phase1/figures

Reads `<results>/<run>/summary.tsv` and `nsi.json` for every run found by glob, plus
`<results>/mhc_cluster_ci.tsv` and `mhc_cluster_ci_diagnostics.tsv` when present. Runs that
are absent are skipped.

A (model, target) condition is admissible when `nsi.json` reports its estimator sanity check
as usable. Inadmissible conditions are drawn dashed with hollow markers in every figure.
MHC_HEAVY GSI bands use the cluster bootstrap over alleles, because the sample is dominated
by one allele and example-level intervals are far too narrow.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
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
    kish_n_eff: float | None = None
    _index: dict = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        return f"{self.params_m:g}M"

    def admissible(self, target: str) -> bool:
        """Estimator sanity check result; a missing nsi.json counts as not admissible."""
        sanity = ((self.nsi or {}).get("targets", {}).get(target, {}).get("estimator", {})).get(
            "sanity", {}
        )
        return bool(sanity.get("usable", False))

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


def _read_cluster_ci(path: Path) -> dict[tuple[str, str, int], tuple[float, float]]:
    """(model label, operator, gap) -> cluster-bootstrap (lo, hi) for the dropout-on GSI."""
    out = {}
    if not path.exists():
        return out
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["section"] != "gsi" or r["token_dropout"] != "dropout_on":
                continue
            if not r["gap"].isdigit():
                continue
            out[(r["model"], r["operator"], int(r["gap"]))] = (
                float(r["cluster_lo"]),
                float(r["cluster_hi"]),
            )
    return out


def _read_kish(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {r["model"]: float(r["kish_n_eff"]) for r in csv.DictReader(fh, delimiter="\t")}


def load_runs(results: Path) -> list[Run]:
    cluster = _read_cluster_ci(results / "mhc_cluster_ci.tsv")
    kish = _read_kish(results / "mhc_cluster_ci_diagnostics.tsv")
    runs = []
    for summary in sorted(results.glob("*/summary.tsv")):
        nsi_path = summary.with_name("nsi.json")
        nsi = json.loads(nsi_path.read_text()) if nsi_path.exists() else None
        params = _params_m(summary.parent.name, nsi)
        if params is None:
            continue
        run = Run(summary.parent.name, params, _read_summary(summary), nsi)
        run.kish_n_eff = kish.get(run.label)
        for r in run.rows:
            ci = cluster.get((run.label, r["operator"], r["gap"]))
            if ci and r["target"] == "MHC_HEAVY" and r["token_dropout"]:
                r["lo"], r["hi"] = ci
        runs.append(run)
    return sorted(runs, key=lambda r: r.params_m)


def _ci_caption(runs: list[Run]) -> str:
    """Records which interval estimator each target uses and the MHC effective sample size."""
    n_eff = {}
    for r in runs:
        if r.kish_n_eff is not None:
            n_eff.setdefault(f"{r.kish_n_eff:.2f}", []).append(r.label)
    eff = ", ".join(f"{v} at {' and '.join(models)}" for v, models in n_eff.items())
    mhc = "MHC_HEAVY intervals: cluster bootstrap over alleles"
    if eff:
        mhc += f", Kish n_eff = {eff}"
    return f"{mhc}. PEPTIDE and TCRB intervals: example-level bootstrap."


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


def _headline(fig, title: str, subtitle: str, caption: str = "", legend: bool = False) -> None:
    """Wrapped title and subtitle on top; optional legend and caption below.

    Strip heights are computed in inches so that long findings wrap instead of clipping.
    The layout rect is shrunk to leave room for them.
    """
    w, h = fig.get_size_inches()
    title = textwrap.fill(title, int(w * 10.5))
    subtitle = textwrap.fill(subtitle, int(w * 14.5))
    caption = (
        "\n".join(textwrap.fill(c, int(w * 17)) for c in caption.split("\n")) if caption else ""
    )
    top_in = 0.08 + 0.2 * (title.count("\n") + 1) + 0.17 * (subtitle.count("\n") + 1) + 0.12
    cap_in = (0.03 + 0.14 * (caption.count("\n") + 1)) if caption else 0.0
    leg_in = 0.36 if legend else 0.0
    bottom_in = cap_in + leg_in
    fig.get_layout_engine().set(rect=(0, bottom_in / h, 1, 1 - (top_in + bottom_in) / h))
    y = 1 - 0.06 / h
    fig.text(0.01, y, title, fontsize=11, fontweight="bold", va="top", linespacing=1.25)
    y -= (0.2 * (title.count("\n") + 1) + 0.04) / h
    fig.text(0.01, y, subtitle, fontsize=8.5, color=TEXT_2, va="top", linespacing=1.3)
    if caption:
        fig.text(0.01, 0.03 / h, caption, fontsize=7.5, color=TEXT_2, va="bottom", linespacing=1.4)
    if legend:
        _target_legend(fig, anchor=(0.5, cap_in / h))


DASH = (0, (4, 2.5))


def _marker_kw(color: str, marker: str, admissible: bool) -> dict:
    """Hollow marker for inadmissible conditions, filled for admissible ones."""
    return {
        "marker": marker,
        "markerfacecolor": color if admissible else SURFACE,
        "markeredgecolor": color,
        "markeredgewidth": 1.0 if admissible else 1.4,
    }


def _line_kw(admissible: bool) -> dict:
    return {"ls": "-" if admissible else DASH}


def _target_legend(fig, anchor: tuple[float, float]) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=TARGET_COLORS[t],
            label=t,
            markersize=MARKER_SIZE,
            **_marker_kw(TARGET_COLORS[t], TARGET_MARKERS[t], True),
        )
        for t in TARGETS
    ]
    handles += [
        Line2D(
            [],
            [],
            color=TEXT_2,
            label="passes sanity check",
            markersize=MARKER_SIZE,
            **_marker_kw(TEXT_2, "o", True),
        ),
        Line2D(
            [],
            [],
            color=TEXT_2,
            label="fails sanity check",
            markersize=MARKER_SIZE,
            **_line_kw(False),
            **_marker_kw(TEXT_2, "o", False),
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=anchor,
        ncol=len(handles),
        columnspacing=1.2,
    )


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
        len(runs), 2, figsize=(8.5, 2.55 * len(runs) + 1.9), sharex=True, sharey=True, squeeze=False
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
                adm = run.admissible(t)
                ax.fill_between(g, lo, hi, color=c, alpha=BAND_ALPHA, lw=0, zorder=2)
                ax.plot(
                    g,
                    v,
                    color=c,
                    zorder=3,
                    **_line_kw(adm),
                    **_marker_kw(c, TARGET_MARKERS[t], adm),
                )
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
                ax.text(
                    0.02,
                    0.84,
                    "dashed line: equal to 20 real mutations",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7.5,
                    color=TEXT_2,
                )
    fig.supylabel(GSI_LABEL, fontsize=9)
    _headline(
        fig,
        "A content-free change to position indices moves ESM-2's score several times "
        "more than 20 real substitutions",
        "Shown for MHC at 150M and 650M, the conditions whose estimator passes its sanity "
        "check; peptide and TCR spans are shown but not admissible.",
        caption=_ci_caption(runs) + " Shaded bands are 95% intervals.",
        legend=True,
    )
    _save(fig, out, "fig1_dose_response")


# ---------------------------------------------------------------- figure 2


def _scale_axes(ax, runs: list[Run], series: dict, ylabel: str) -> None:
    """One line per target against model size on a log axis, with CI error bars.

    `series[target]` is a list of (x, y, lo, hi, admissible). A segment is solid only when
    both of its endpoints are admissible.
    """
    for t in TARGETS:
        pts = series.get(t, [])
        if not pts:
            continue
        c = TARGET_COLORS[t]
        for a, b in zip(pts, pts[1:], strict=False):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=c, zorder=3, **_line_kw(a[4] and b[4]))
        for x, y, lo, hi, adm in pts:
            ax.errorbar(
                [x],
                [y],
                yerr=[[y - lo], [hi - y]],
                color=c,
                elinewidth=LINE_WIDTH,
                capsize=3,
                zorder=3,
                ls="none",
                **_marker_kw(c, TARGET_MARKERS[t], adm),
            )
    sizes = sorted({r.params_m for r in runs})
    ax.set_xscale("log")
    ax.set_xlim(min(sizes) * 0.7, max(sizes) * 3.2)
    ax.set_xticks(sizes, [f"{s:g}M" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Model parameters (millions, log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(False, axis="x")


def fig2_scale(runs: list[Run], out: Path, gaps: tuple[int, int] = (25, 100)) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True)
    highs = []
    for ax, gap in zip(axes, gaps, strict=True):
        series = {}
        for t in TARGETS:
            pts = []
            for run in runs:
                r = run.get("G1", t, gap)
                if r:
                    pts.append(
                        (run.params_m * DODGE[t], r["gsi"], r["lo"], r["hi"], run.admissible(t))
                    )
            series[t] = pts
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
    failing = sorted({r.label for r in runs if not r.admissible("MHC_HEAVY")})
    _headline(
        fig,
        "Layout sensitivity decreases with model size but does not disappear",
        "MHC at 150M and 650M, the admissible conditions; still about 5x the effect of 20 "
        "real substitutions at 650M.",
        caption=(
            _ci_caption(runs)
            + f"\nEstimator fails its sanity check for MHC at {', '.join(failing)} "
            "and for PEPTIDE and TCRB at every size."
        ),
        legend=True,
    )
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
            adm = run.admissible(t)
            for op in ("G1", "G2"):
                g, v, _, _ = _gap_series(run, op, t, "shift")
                if not g:
                    continue
                # G2 is drawn thinner and lighter so that dash and hollow stay free to mark
                # estimator admissibility.
                weak = op == "G2"
                ax.plot(
                    g,
                    v,
                    color=c,
                    lw=1.0 if weak else LINE_WIDTH,
                    alpha=0.6 if weak else 1.0,
                    zorder=3,
                    **_line_kw(adm),
                    **_marker_kw(c, TARGET_MARKERS[t], adm),
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
        _end_labels(ax, ends, min_gap=11)
    _headline(
        fig,
        "Absolute shifts are small for the peptide and TCR targets, but not for MHC",
        "GSI is a ratio of two small quantities except in the MHC panels, "
        "where the shift is large outright.",
        caption=(
            "C1 n=20 (dashed gray): median |shift| from 20 real substitutions, the GSI "
            "denominator. G1 full weight, G2 thinner and lighter.\n"
            "Dashed with hollow markers: estimator fails its sanity check (PEPTIDE and TCRB "
            "at every size, MHC at 35M). Y-axis is shared down each column."
        ),
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
                pts.append(
                    (run.params_m * DODGE[t], a["value"], a["ci_lo"], a["ci_hi"], run.admissible(t))
                )
        series[t] = pts
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
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
        caption="AUROC intervals are example-level bootstrap; for MHC_HEAVY they are too narrow "
        "(few distinct alleles).",
        legend=True,
    )
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
                color=TARGET_COLORS[t],
                ls="none",
                markersize=MARKER_SIZE,
                zorder=3,
                **_marker_kw(TARGET_COLORS[t], marker, run.admissible(t)),
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
            Line2D(
                [],
                [],
                ls="none",
                label="hollow: fails sanity check",
                **_marker_kw(TEXT_2, "o", False),
            ),
        ],
        loc="upper right",
        ncol=1,
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


MODEL_COLORS = {"150M": "#2a78d6", "650M": "#eb6834"}


def _read_balanced(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [r for r in csv.DictReader(fh, delimiter="\t") if r["token_dropout"] == "dropout_on"]


def fig6_allele_balanced(results: Path, out: Path) -> None:
    """GSI on the 51-allele balanced sample, with the per-allele spread behind it.

    The balanced sample is the strongest evidence for the headline, so it gets its own
    figure: the left panel is the aggregate, the right shows that the aggregate is not
    one allele carrying the rest.
    """
    root = results / "allele_balanced"
    rows = _read_balanced(root / "allele_balanced.tsv")
    per = _read_balanced(root / "per_allele_gsi.tsv")
    if not rows:
        return
    models = [m for m in MODEL_COLORS if any(r["model"] == m for r in rows)]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.2, 4.3), gridspec_kw={"width_ratios": (1.4, 1), "wspace": 0.22}
    )
    ends = []
    for model in models:
        series = sorted(
            (int(r["gap"]), float(r["gsi"]), float(r["cluster_lo"]), float(r["cluster_hi"]))
            for r in rows
            if r["model"] == model
        )
        gaps = [s[0] for s in series]
        color = MODEL_COLORS[model]
        ax.fill_between(
            gaps,
            [s[2] for s in series],
            [s[3] for s in series],
            color=color,
            alpha=BAND_ALPHA,
            lw=0,
        )
        ax.plot(
            gaps,
            [s[1] for s in series],
            color=color,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            zorder=3,
        )
        ends.append((gaps[-1], series[-1][1], f"ESM-2 {model}"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color=AXIS, lw=1, ls=DASH, zorder=1)
    _ref_label(ax, "GSI 1.0: a relayout moves the score as much as 20 real substitutions")
    _gap_axis(ax, sorted({int(r["gap"]) for r in rows}), right=175)
    ax.set_ylabel("GSI", fontsize=9)
    ax.set_xlabel(GAP_LABEL)

    # Per-allele points at the gap where both sizes are clearly above 1.0.
    gap = 25
    for i, model in enumerate(models):
        vals = sorted(
            float(r["gsi"])
            for r in per
            if r["model"] == model and int(r["gap"]) == gap and r["gsi"]
        )
        if not vals:
            continue
        rng = __import__("random").Random(0)
        xs = [i + rng.uniform(-0.17, 0.17) for _ in vals]
        ax2.plot(
            xs,
            vals,
            ls="none",
            marker="o",
            markersize=4,
            color=MODEL_COLORS[model],
            alpha=0.55,
            markeredgecolor="none",
        )
        med = vals[len(vals) // 2]
        ax2.plot([i - 0.28, i + 0.28], [med, med], color=MODEL_COLORS[model], lw=2.2, zorder=4)
        ax2.annotate(
            f"median {med:.1f}",
            (i, med),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT_2,
            ha="center",
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2},
        )
    ax2.set_yscale("log")
    ax2.axhline(1.0, color=AXIS, lw=1, ls=DASH, zorder=1)
    ax2.set_xticks(range(len(models)), [f"ESM-2 {m}" for m in models])
    ax2.set_xlim(-0.52, len(models) - 0.5)
    ax2.grid(False, axis="x")
    ax2.set_ylabel(f"Per-allele GSI at gap {gap}", fontsize=9)
    _ref_label(ax2, "GSI 1.0")

    fig.set_layout_engine("constrained")
    _headline(
        fig,
        "The layout artifact is present across 51 HLA alleles, not carried by HLA-A*02:01",
        f"MHC_HEAVY, operator G1, token dropout on. {len(models)} model sizes on an "
        "allele-balanced sample: 51 alleles, at most 27 examples each, n = 610, Kish "
        "effective sample size 26.86.",
        "GSI is the median absolute score shift from a relayout divided by the median shift\n"
        "from 20 real substitutions. Left: bands are 95% percentile bootstrap intervals\n"
        "resampling alleles.\n"
        "Right: one point per allele at n = 27, so individual alleles are poorly resolved; the "
        "spread is shown to make that visible rather than to support per-allele claims.",
    )
    _end_labels(ax, ends)
    _save(fig, out, "fig6_allele_balanced")


ARM_LABELS = {"A0": "A0 baseline", "A4": "A4 full module", "A5": "A4 + order augmentation"}


CLASSIFIER_ROOTS = (
    ("ESM-2 35M", Path("results/phase2/classifier")),
    ("ESM-2 650M", Path("results/phase2/classifier_650m")),
)


def _classifier_seeds(root: Path) -> tuple[dict, dict] | None:
    path = root / "summary.tsv"
    if not path.exists():
        return None
    with path.open() as fh:
        rows = {r["arm"]: r for r in csv.DictReader(fh, delimiter="\t")}
    seeds: dict[str, dict[str, list[float]]] = {}
    for arm in ARM_LABELS:
        per: dict[str, list[float]] = {}
        for s in range(3):
            m = root / "runs" / f"{arm}_s{s}" / "metrics.json"
            if not m.exists():
                continue
            ev = json.loads(m.read_text())["eval"]
            for shard in ("test_allele", "test_epitope"):
                per.setdefault(shard, []).append(ev[shard]["auroc"]["point"])
        if per and arm in rows:
            seeds[arm] = per
    return (rows, seeds) if seeds else None


def fig7_h2_classifier(out: Path, roots=CLASSIFIER_ROOTS) -> None:
    """Per-seed held-out AUROC against chance, the direct test of H2.

    Seeds are drawn individually rather than as a mean with an error bar: the seed spread is
    as large as the between-arm differences, and that is the finding.
    """
    loaded = [(label, *d) for label, root in roots if (d := _classifier_seeds(root))]
    if not loaded:
        return

    shards = [("test_allele", "Held-out alleles"), ("test_epitope", "Held-out epitopes")]
    nrow = len(loaded)
    fig, axes = plt.subplots(nrow, 2, figsize=(9.4, 1.0 + 3.0 * nrow), sharey=True, squeeze=False)
    for row, (size_label, rows, seeds) in enumerate(loaded):
        arms = list(seeds)
        for ax, (shard, shard_title) in zip(axes[row], shards, strict=True):
            for i, arm in enumerate(arms):
                color = TARGET_COLORS[TARGETS[i % len(TARGETS)]]
                ax.plot(
                    [i] * len(seeds[arm][shard]),
                    seeds[arm][shard],
                    ls="none",
                    marker="o",
                    markersize=7,
                    color=color,
                    alpha=0.65,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    zorder=3,
                )
                mean = float(rows[arm][f"{shard}_auroc_mean"])
                lo = float(rows[arm][f"{shard}_auroc_lo"])
                hi = float(rows[arm][f"{shard}_auroc_hi"])
                ax.plot([i - 0.26, i + 0.26], [mean, mean], color=color, lw=2.2, zorder=4)
                ax.plot([i, i], [lo, hi], color=color, lw=1.2, alpha=0.7, zorder=2)
            ax.axhline(0.5, color=AXIS, lw=1, ls=DASH, zorder=1)
            ax.set_xticks(range(len(arms)), [ARM_LABELS[a] for a in arms], fontsize=8.5)
            ax.set_xlim(-0.6, len(arms) - 0.4)
            ax.set_ylim(0.28, 0.75)
            ax.grid(False, axis="x")
            ax.set_title(f"{size_label}, {shard_title}", fontsize=9.5, color=TEXT_2, loc="left")
        axes[row][0].set_ylabel("Held-out binding AUROC", fontsize=9)
        _ref_label(axes[row][0], "chance", y=0.5)

    sizes = ", ".join(label for label, _, _ in loaded)
    fig.set_layout_engine("constrained")
    _headline(
        fig,
        "The chain-aware module does not improve held-out binding over a matched baseline",
        f"{sizes}. LoRA r=16, trainable parameters matched across arms, 300 steps, three "
        "seeds. Each dot is one seed; the bar is the seed mean, the thin line a cluster "
        "bootstrap over MHC alleles.",
        "No arm's interval excludes chance, so none is a usable binding predictor. A0 on\n"
        "held-out epitopes sits below chance at both sizes. Between-seed spread swamps the\n"
        "between-arm differences here; fig8 pairs the seeds, which is what the design supports.\n"
        "Negatives are mismatched decoys, so AUROC is a weak proxy for binding, not a bound.",
    )
    _save(fig, out, "fig7_h2_classifier")


SIZE_COLORS = {"35M": "#2a78d6", "650M": "#eb6834"}


def fig8_h2_paired(out: Path, results: Path = Path("results/phase2/h2_paired")) -> None:
    """Per-seed paired deltas against the matched baseline.

    Seeds are shared across arms, so the paired difference is the quantity the design
    produces. Drawing all six seeds of each contrast shows directly what the summary
    statistic claims: whether they agree in sign across two model sizes.
    """
    path = results / "h2_paired.tsv"
    if not path.exists():
        return
    with path.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        return

    shards = [("test_allele", "Held-out alleles"), ("test_epitope", "Held-out epitopes")]
    arms = sorted({r["arm"] for r in rows})
    sizes = sorted({r["size"] for r in rows}, key=lambda s: len(s))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, (shard, shard_title) in zip(axes, shards, strict=True):
        for i, arm in enumerate(arms):
            for j, size in enumerate(sizes):
                row = next(
                    (
                        r
                        for r in rows
                        if r["arm"] == arm and r["shard"] == shard and r["size"] == size
                    ),
                    None,
                )
                if row is None:
                    continue
                x = i + (j - (len(sizes) - 1) / 2) * 0.3
                vals = [float(v) for v in row["seed_deltas"].split(";")]
                color = SIZE_COLORS.get(size, MUTED)
                ax.plot(
                    [x] * len(vals),
                    vals,
                    ls="none",
                    marker="o",
                    markersize=7,
                    color=color,
                    alpha=0.7,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    zorder=3,
                )
                ax.plot(
                    [x - 0.11, x + 0.11],
                    [float(row["paired_delta"])] * 2,
                    color=color,
                    lw=2.2,
                    zorder=4,
                )
        ax.axhline(0, color=AXIS, lw=1, ls=DASH, zorder=1)
        ax.set_xticks(range(len(arms)), [ARM_LABELS[a] for a in arms], fontsize=8.5)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.grid(False, axis="x")
        ax.set_title(shard_title, fontsize=9.5, color=TEXT_2, loc="left")
    axes[0].set_ylabel("Paired AUROC delta vs A0 baseline", fontsize=9)
    _ref_label(axes[0], "no change", y=0)
    # Headroom so the extreme seed is not clipped by the legend below it.
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo - 0.04 * (hi - lo), hi)
    axes[1].legend(
        handles=[
            Line2D(
                [], [], color=SIZE_COLORS.get(s, MUTED), marker="o", ls="none", label=f"ESM-2 {s}"
            )
            for s in sizes
        ],
        loc="lower left",
        ncol=len(sizes),
        fontsize=8.5,
        frameon=False,
    )

    fig.set_layout_engine("constrained")
    _headline(
        fig,
        "Paired by seed, the module raises held-out-epitope AUROC in every seed at both sizes",
        "Each dot is one seed's delta against the baseline trained on the same subsample, "
        "initialization and batch order. The bar is the mean of the three.",
        "Three contrasts agree in sign across all six seeds: A4 down on alleles, A4 and A5 up\n"
        "on epitopes. A sign test over six seeds gives p = 0.031 before correcting for the four\n"
        "contrasts shown, and nothing survives that correction. The baseline is below chance on\n"
        "held-out epitopes, so an increase there moves toward 0.5 rather than above it. This\n"
        "pairing was chosen after seeing the unpaired result.",
    )
    _save(fig, out, "fig8_h2_paired")


# ---------------------------------------------------------------- table


def _cell(r: dict | None) -> str:
    return "n/a" if r is None else f"{r['gsi']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"


def write_table1(runs: list[Run], out: Path, gaps: tuple[int, ...] = (1, 25, 150)) -> None:
    head = ["Model", "Target", "Estimator passes"]
    head += [f"{op} gap {g}" for op in ("G1", "G2") for g in gaps]
    head.append("C1 denominator (nats)")
    lines = [
        "Table 1. GSI as `value [95% CI lo, hi]`; token dropout on, in-distribution gaps only. "
        "MHC_HEAVY intervals are cluster bootstrap over alleles, others example-level. "
        "Only rows with Estimator passes = yes support a claim. "
        "The last column is the median |shift| from 20 real substitutions.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * 3 + ["---:"] * (len(head) - 3)) + "|",
    ]
    for run in runs:
        for t in TARGETS:
            cells = [f"ESM-2 {run.label}", t, "yes" if run.admissible(t) else "no"]
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
    fig6_allele_balanced(args.results, out)
    fig7_h2_classifier(out)
    fig8_h2_paired(out)
    write_table1(runs, out)
    print(f"wrote figures and table1.md for {[r.name for r in runs]} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
