"""Phase 1 nuisance-sensitivity audit.

    python -m cla.eval.audit --config configs/audit/esm2_35m.yaml [--limit N] [--device cpu|mps]

Each example is scored once in its baseline layout, then again after every operator, under
two token-dropout arms (see docs/DECISIONS.md). Scores are the span log-likelihood of a
target role set given the rest of the complex; every entry of `scoring.targets` is audited
separately and reported under `targets`, each with the estimator configured for it.

The primary metric is the Geometry Sensitivity Index,

    GSI = median |s(x) - s(geo(x))| / median |s(x) - s(C1(x, n=20))|

where `geo` is a pure-geometry operator (G1, G2: only `position_ids` change) and C1 mutates
20 residues of the context chains. It measures how far a content-free position change moves
the score relative to twenty real amino-acid substitutions. Geometry operators and controls
run on the positive examples only; the content-coupled operators P1, P2, P4 run on positives
and decoys so the pre-registered binder/decoy quantities stay computable.

The pre-registered NSI divides by the binder/decoy gap, which is unresolvable for an
untrained backbone, so it is reported as null when that gap is below
`NSI_MIN_STANDARDIZED_SEPARATION` (in score SDs). Decoys are mismatched pairings from
`cla.eval.negatives`. A metric that cannot be computed is null with a reason.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cla.data.splits import split_specs
from cla.eval.metrics import (
    auroc,
    bootstrap_ci,
    decision_flip_rate,
    gsi,
    length_stratified_nsi,
    nsi,
    rank_stability,
)
from cla.eval.negatives import make_mismatched_negatives
from cla.layout.concat import RenderedLayout, render_naive
from cla.layout.geometry import GEOMETRY_OPERATORS
from cla.layout.perturb import CONTROLS, OPERATORS, OperatorNotApplicable
from cla.layout.spec import Chain, ChainSpec, Role
from cla.model.backbone import load_esm, resolve_device
from cla.model.scoring import score_many
from cla.utils.logging import JsonlLogger, get_logger, json_default
from cla.utils.manifest import build_manifest, write_manifest
from cla.utils.seeding import set_all_seeds

log = get_logger("cla.eval.audit")

SCHEMA_VERSION = 2
NULL_OPERATORS = ("P1", "P2", "P3", "P4")
ESTIMATORS = {"span_pll": "span", "pll_pseudo": "pseudo", "windowed_span_pll": "windowed"}
DEFAULT_LENGTH_BINS = (0, 12, 14, 16, 18, 1000)
ARMS = ("true", "false")

# ESM-2 trained on positions 0..1025. Rotary embeddings extrapolate silently past that, so
# in-distribution gaps are checked against it per example.
TRAINED_MAX_POSITION = 1025
UNIFORM_PRIOR_PLL = -math.log(20)
# A long-span estimator has to beat the uniform prior by this margin to count as usable.
ESTIMATOR_MIN_MARGIN = 0.25
OOD_LABEL = "out of distribution: position indices pass ESM-2's trained range of 1026"

# Gate G1 thresholds. Moving these after seeing data invalidates the gate.
G1_NSI_PROCEED = 0.25
G1_RHO_PROCEED = 0.90
G1_NSI_STOP = 0.15
G1_DELTA_AUROC_STOP = 0.02
G1_MIN_NULL_PASSES = 2
GSI_PROCEED = 1.0
GSI_CI_LO_PROCEED = 0.5
GSI_MIN_GAPS = 2
GSI_MIN_MODELS = 2

# A dose-response is called a step when the log-log slope of median |delta| against gap
# (gaps >= 1) has a CI entirely below this, graded when entirely above, mixed otherwise.
# 0.25 means a 150-fold gap range changes the shift by under 3.5x. Fixed before the sweep ran.
SHAPE_SLOPE_THRESHOLD = 0.25
SHAPE_REFERENCE_GAP = 150

# NSI is reported only when the binder/decoy median gap is at least this fraction of the
# standard deviation of the original scores (a standardized gap of 0.2 is the conventional
# "small" effect, AUROC of roughly 0.56). A 60-example pilot at 35M measured gaps of 0.0004
# nats/residue (0.003 SD) with token dropout on and 0.019 (0.12 SD) off. Dividing by those
# produced NSI values of 90 to 860 whose bootstrap intervals excluded the point estimate.
# Scaling by the score spread keeps the floor meaningful across targets and model sizes,
# whose score scales differ several-fold. The value is not pre-registered.
NSI_MIN_STANDARDIZED_SEPARATION = 0.2


# --------------------------------------------------------------------------- config / data


def load_config(path: Path) -> dict:
    """Load a YAML config, merging its `extends` parent first (child keys win)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(load_config((path.parent / parent).resolve()), raw)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def load_specs_jsonl(path: Path) -> list[ChainSpec]:
    specs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            d.pop("mhc_domains", None)  # assembly metadata, not a ChainSpec field
            d["chains"] = tuple(Chain(Role(c["role"]), c["sequence"]) for c in d["chains"])
            specs.append(ChainSpec(**d))
    return specs


def stratified_subsample(
    shards: dict[str, list[ChainSpec]], n: int, rng: np.random.Generator
) -> list[tuple[ChainSpec, str]]:
    """Draw `n` specs with equal quotas per non-empty shard, redistributing shortfalls.

    A uniform draw would be dominated by the train shard and leave the small held-out
    shards nearly empty; equal quotas keep every shard represented. Pools are sorted by
    example_id so the draw does not depend on file order.
    """
    pools = {k: sorted(v, key=lambda s: s.example_id) for k, v in sorted(shards.items()) if v}
    quota = dict.fromkeys(pools, 0)
    remaining, open_ = n, list(pools)
    while remaining > 0 and open_:
        share = max(remaining // len(open_), 1)
        for k in list(open_):
            take = min(share, len(pools[k]) - quota[k], remaining)
            quota[k] += take
            remaining -= take
            if quota[k] == len(pools[k]):
                open_.remove(k)
            if remaining == 0:
                break
    chosen: list[tuple[ChainSpec, str]] = []
    for k, pool in pools.items():
        idx = np.sort(rng.choice(len(pool), size=quota[k], replace=False))
        chosen.extend((pool[int(i)], k) for i in idx)
    return chosen


def kish_n_eff(counts: Any) -> float:
    """Kish effective sample size of cluster sizes: (sum n)^2 / sum n^2."""
    n = np.asarray(list(counts), dtype=float)
    return float(n.sum() ** 2 / np.sum(n**2))


def choose_allele_cap(
    counts: dict[str, int], total_range: tuple[int, int]
) -> tuple[int, dict[str, Any]]:
    """Per-allele cap that maximizes Kish n_eff with the sample total inside `total_range`.

    Capping at `k` takes min(count, k) from each allele, so n_eff falls as `k` grows while the
    total rises. The range bounds the total from below (enough pairs for a median ratio) and
    above (scoring cost), and the cap is picked inside it by n_eff alone.
    """
    lo, hi = total_range
    best: tuple[float, int, int] | None = None
    for k in range(1, max(counts.values()) + 1):
        sizes = [min(c, k) for c in counts.values()]
        total = sum(sizes)
        if lo <= total <= hi:
            cand = (kish_n_eff(sizes), -k, total)
            best = cand if best is None or cand > best else best
        if total > hi:
            break
    if best is None:
        raise ValueError(f"no per-allele cap gives a total in {total_range}")
    n_eff, neg_k, total = best
    return -neg_k, {"n_eff": n_eff, "total": total, "total_range": list(total_range)}


def allele_balanced_subsample(
    specs: list[ChainSpec], cap: int, rng: np.random.Generator
) -> list[ChainSpec]:
    """Draw up to `cap` specs per distinct MHC allele, uniformly within the allele.

    Proportional draws are dominated by HLA-A*02:01, which leaves the cluster bootstrap
    over alleles with an effective sample size near 2. Pools are sorted by example_id so the
    draw does not depend on file order.
    """
    by_allele: dict[str, list[ChainSpec]] = {}
    for s in sorted(specs, key=lambda s: s.example_id):
        if s.mhc_allele:
            by_allele.setdefault(s.mhc_allele, []).append(s)
    chosen: list[ChainSpec] = []
    for allele in sorted(by_allele):
        pool = by_allele[allele]
        idx = np.sort(rng.choice(len(pool), size=min(cap, len(pool)), replace=False))
        chosen.extend(pool[int(i)] for i in idx)
    return chosen


# --------------------------------------------------------------------------- token dropout


@contextlib.contextmanager
def token_dropout(model, enabled: bool):
    """Set token dropout for the duration of the block.

    HuggingFace copies `config.token_dropout` into `EsmEmbeddings.token_dropout` at
    construction and only the module attribute is read in `forward`, so flipping
    `config` alone has no effect. Both are set so they never disagree.
    """
    emb = model.esm.embeddings
    prev_attr, prev_cfg = emb.token_dropout, model.config.token_dropout
    emb.token_dropout = enabled
    model.config.token_dropout = enabled
    try:
        yield
    finally:
        emb.token_dropout, model.config.token_dropout = prev_attr, prev_cfg


# --------------------------------------------------------------------------- metrics


def _finite(name: str, value: float | None, reasons: dict[str, str]) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        reasons.setdefault(name, "non-finite value (constant input or undefined statistic)")
        return None
    return float(value)


def _boot_entry(fn, *arrays, boot: dict, seed: int) -> dict[str, Any]:
    """Bootstrap CI of `fn` as a `{value, ci_lo, ci_hi}` dict, or nulls with a reason."""
    try:
        value, lo, hi = bootstrap_ci(
            fn, *arrays, n_boot=boot["n_boot"], alpha=boot["alpha"], seed=seed
        )
    except ValueError as e:
        return {"value": None, "ci_lo": None, "ci_hi": None, "reason": str(e)}
    if not np.isfinite(value):
        return {"value": None, "ci_lo": None, "ci_hi": None, "reason": "non-finite value"}
    return {"value": value, "ci_lo": lo, "ci_hi": hi}


def compute_metrics(
    orig: np.ndarray,
    pert: np.ndarray,
    labels: np.ndarray,
    cdr3b_len: np.ndarray,
    *,
    boot: dict,
    bins: tuple[float, ...],
    min_bin_n: int,
    seed: int,
    min_standardized_separation: float = NSI_MIN_STANDARDIZED_SEPARATION,
    ref_orig: np.ndarray | None = None,
    ref_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pre-registered metric set for one (operator, arm). Uncomputable metrics are null with
    a reason.

    `ref_orig` and `ref_labels` are the baseline scores and labels of the whole population,
    used for the binder/decoy separation; they default to `orig` and `labels` when the
    perturbed examples already span both classes.
    """
    reasons: dict[str, str] = {}
    out: dict[str, Any] = {"n_pairs": int(len(orig))}
    ref_orig = orig if ref_orig is None else ref_orig
    ref_labels = labels if ref_labels is None else ref_labels

    def attempt(name: str, fn):
        try:
            return fn()
        except (ValueError, FloatingPointError) as e:
            reasons[name] = str(e)
            return None

    ref_binders, ref_nonbinders = ref_orig[ref_labels == 1], ref_orig[ref_labels == 0]
    out["binder_median"] = float(np.median(ref_binders)) if ref_binders.size else None
    out["nonbinder_median"] = float(np.median(ref_nonbinders)) if ref_nonbinders.size else None
    out["separation"] = (
        abs(out["binder_median"] - out["nonbinder_median"])
        if ref_binders.size and ref_nonbinders.size
        else None
    )
    out["median_abs_shift"] = float(np.median(np.abs(orig - pert))) if len(orig) else None

    def nsi_fn(a, b):
        return nsi(a, b, ref_binders, ref_nonbinders)

    spread = float(np.std(ref_orig)) if len(ref_orig) else 0.0
    out["standardized_separation"] = (
        out["separation"] / spread if out["separation"] is not None and spread > 0 else None
    )
    if (
        out["standardized_separation"] is None
        or out["standardized_separation"] < min_standardized_separation
    ):
        out["nsi"] = {
            "value": None,
            "ci_lo": None,
            "ci_hi": None,
            "reason": "denominator unresolved",
            "separation": out["separation"],
            "standardized_separation": out["standardized_separation"],
        }
    else:
        ci = attempt(
            "nsi",
            lambda: bootstrap_ci(
                nsi_fn,
                orig,
                pert,
                n_boot=boot["n_boot"],
                alpha=boot["alpha"],
                seed=seed,
            ),
        )
        out["nsi"] = {
            "value": _finite("nsi", ci[0], reasons) if ci else None,
            "ci_lo": ci[1] if ci else None,
            "ci_hi": ci[2] if ci else None,
        }

    rs = attempt("rank_stability", lambda: rank_stability(orig, pert)) or {}
    for key in ("spearman_rho", "spearman_p", "kendall_tau", "kendall_p"):
        out[key] = _finite(key, rs.get(key), reasons)
        if key not in rs:
            reasons.setdefault(key, reasons.get("rank_stability", "rank statistic unavailable"))
    reasons.pop("rank_stability", None)

    a0 = attempt("auroc_original", lambda: auroc(labels, orig))
    a1 = attempt("auroc_perturbed", lambda: auroc(labels, pert))
    out["auroc_original"], out["auroc_perturbed"] = a0, a1
    out["delta_auroc"] = a1 - a0 if a0 is not None and a1 is not None else None
    if out["delta_auroc"] is None:
        reasons["delta_auroc"] = "AUROC unavailable for original or perturbed scores"

    out["decision_flip_rate"] = attempt(
        "decision_flip_rate", lambda: decision_flip_rate(orig, pert)
    )
    if out["nsi"]["value"] is None and out["nsi"].get("reason"):
        out["length_stratified_nsi"] = None
        reasons["length_stratified_nsi"] = out["nsi"]["reason"]
    else:
        out["length_stratified_nsi"] = attempt(
            "length_stratified_nsi",
            lambda: length_stratified_nsi(
                orig, pert, cdr3b_len, ref_binders, ref_nonbinders, bins=bins, min_n=min_bin_n
            ),
        )
    out["null_reasons"] = reasons
    return out


def compute_geometry_entry(
    orig: np.ndarray,
    index: np.ndarray,
    geo: np.ndarray,
    c1_index: np.ndarray,
    c1: np.ndarray,
    *,
    boot: dict,
    seed: int,
) -> dict[str, Any]:
    """GSI and raw shift for one (geometry operator, gap, arm).

    Numerator and denominator are evaluated on the examples where both the geometry operator
    and C1 applied, so the paired bootstrap resamples them on one index draw and
    `median_abs_delta / c1_denominator` reproduces the GSI value.
    """
    common, ia, ib = np.intersect1d(index, c1_index, return_indices=True)
    if common.size == 0:
        reason = "no examples where both the geometry operator and C1 applied"
        return {
            "n_pairs": 0,
            "median_abs_delta": None,
            "median_abs_delta_ci": None,
            "c1_denominator": None,
            "gsi": {"value": None, "ci_lo": None, "ci_hi": None, "reason": reason},
        }
    a, g, c = orig[common], geo[ia], c1[ib]
    shift = _boot_entry(lambda x, y: float(np.median(np.abs(x - y))), a, g, boot=boot, seed=seed)
    return {
        "n_pairs": int(common.size),
        "median_abs_delta": float(np.median(np.abs(a - g))),
        "median_abs_delta_ci": [shift["ci_lo"], shift["ci_hi"]],
        "c1_denominator": float(np.median(np.abs(a - c))),
        "gsi": _boot_entry(gsi, a, g, c, boot=boot, seed=seed),
    }


def dose_response_shape(
    orig: np.ndarray,
    by_gap: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    boot: dict,
    seed: int,
) -> dict[str, Any]:
    """Step-versus-graded summary of median |delta| against gap, over gaps >= 1.

    `by_gap` maps gap -> (example index, scores). The fit uses the examples present at every
    gap, so one bootstrap draw resamples all gaps jointly. The slope is the OLS slope of
    log10 median |delta| on log10 gap. `fraction_of_reference` is each gap's median |delta|
    over that at `SHAPE_REFERENCE_GAP`.
    """
    gaps = sorted(g for g in by_gap if g >= 1)
    if len(gaps) < 3:
        return {"shape": None, "reason": "fewer than three positive gaps"}
    common = functools.reduce(np.intersect1d, [by_gap[g][0] for g in gaps])
    if common.size == 0:
        return {"shape": None, "reason": "no examples present at every gap"}
    deltas = np.stack(
        [np.abs(orig[common] - by_gap[g][1][np.searchsorted(by_gap[g][0], common)]) for g in gaps]
    )
    x = np.log10(np.asarray(gaps, dtype=float))

    def slope(cols: np.ndarray) -> float:
        med = np.median(deltas[:, cols], axis=1)
        if (med <= 0).any():
            raise ValueError("median |delta| is zero at a positive gap")
        return float(np.polyfit(x, np.log10(med), 1)[0])

    try:
        point = slope(np.arange(common.size))
    except ValueError as e:
        return {"shape": None, "reason": str(e), "n_pairs": int(common.size)}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(boot["n_boot"]):
        try:
            draws.append(slope(rng.integers(0, common.size, size=common.size)))
        except ValueError:
            continue
    if len(draws) < boot["n_boot"] / 2:
        return {"shape": None, "reason": "bootstrap slopes mostly undefined"}
    lo, hi = np.percentile(draws, [100 * boot["alpha"] / 2, 100 * (1 - boot["alpha"] / 2)])
    if hi < SHAPE_SLOPE_THRESHOLD:
        shape = "step"
    elif lo > SHAPE_SLOPE_THRESHOLD:
        shape = "graded"
    else:
        shape = "mixed"
    med = {g: float(np.median(deltas[i])) for i, g in enumerate(gaps)}
    ref = med.get(SHAPE_REFERENCE_GAP)
    return {
        "shape": shape,
        "loglog_slope": {"value": point, "ci_lo": float(lo), "ci_hi": float(hi)},
        "slope_threshold": SHAPE_SLOPE_THRESHOLD,
        "median_abs_delta": {str(g): v for g, v in med.items()},
        "fraction_of_reference": ({str(g): v / ref for g, v in med.items()} if ref else None),
        "reference_gap": SHAPE_REFERENCE_GAP,
        "n_pairs": int(common.size),
    }


def compute_discrimination(
    orig: np.ndarray, labels: np.ndarray, *, boot: dict, seed: int
) -> dict[str, Any]:
    """Binder vs decoy separation of the raw score, with a bootstrap CI on the AUROC."""
    binders, decoys = orig[labels == 1], orig[labels == 0]
    out: dict[str, Any] = {
        "n_binders": int(binders.size),
        "n_decoys": int(decoys.size),
        "binder_median": float(np.median(binders)) if binders.size else None,
        "decoy_median": float(np.median(decoys)) if decoys.size else None,
    }
    out["separation"] = (
        abs(out["binder_median"] - out["decoy_median"]) if binders.size and decoys.size else None
    )
    try:
        point, lo, hi = bootstrap_ci(
            auroc, labels, orig, n_boot=boot["n_boot"], alpha=boot["alpha"], seed=seed
        )
        out["auroc"] = {"value": point, "ci_lo": lo, "ci_hi": hi}
    except ValueError as e:
        out["auroc"] = {"value": None, "ci_lo": None, "ci_hi": None, "reason": str(e)}
    return out


# --------------------------------------------------------------------------- Gate G1


def _num(x) -> bool:
    return x is not None and np.isfinite(x)


def evaluate_gate_g1(metrics: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    """Pre-registered Gate G1 over `{op: {"nsi", "spearman_rho", "delta_auroc"}}`.

    PROCEED: at least two of P1-P4 have NSI >= 0.25 and Spearman rho <= 0.90, and P5's NSI
    exceeds that of every available null operator.
    STOP: every available null operator has NSI < 0.15 and |delta AUROC| < 0.02.
    AMBIGUOUS otherwise, including when a needed value is missing: unknown never counts
    toward STOP. When no null operator has a computable NSI the rule cannot be evaluated and
    the verdict is null with a reason.
    """
    nulls = {op: m for op, m in metrics.items() if op in NULL_OPERATORS}
    passing = [
        op
        for op, m in nulls.items()
        if _num(m.get("nsi"))
        and _num(m.get("spearman_rho"))
        and m["nsi"] >= G1_NSI_PROCEED
        and m["spearman_rho"] <= G1_RHO_PROCEED
    ]
    null_nsi = [m["nsi"] for m in nulls.values() if _num(m.get("nsi"))]
    p5_nsi = (metrics.get("P5") or {}).get("nsi")
    p5_moves_more = _num(p5_nsi) and bool(null_nsi) and p5_nsi > max(null_nsi)

    all_nsi_known = bool(nulls) and len(null_nsi) == len(nulls)
    deltas = [abs(m["delta_auroc"]) for m in nulls.values() if _num(m.get("delta_auroc"))]
    all_delta_known = bool(nulls) and len(deltas) == len(nulls)
    stop = (
        all_nsi_known
        and all_delta_known
        and all(v < G1_NSI_STOP for v in null_nsi)
        and all(d < G1_DELTA_AUROC_STOP for d in deltas)
    )

    reason = None
    if not null_nsi:
        verdict = None
        reason = "NSI not computable for any null operator (denominator unresolved)"
    elif len(passing) >= G1_MIN_NULL_PASSES and p5_moves_more:
        verdict = "PROCEED"
    elif stop:
        verdict = "STOP"
    else:
        verdict = "AMBIGUOUS"

    warnings = []
    if verdict == "STOP" and not p5_moves_more:
        warnings.append("P5 did not exceed the null operators; the audit may be insensitive")
    return {
        "verdict": verdict,
        "reason": reason,
        "null_operators_passing": passing,
        "n_null_passing": len(passing),
        "p5_nsi": float(p5_nsi) if _num(p5_nsi) else None,
        "max_null_nsi": float(max(null_nsi)) if null_nsi else None,
        "p5_moves_more_than_nulls": bool(p5_moves_more),
        "max_abs_delta_auroc_null": float(max(deltas)) if deltas else None,
        "thresholds": {
            "nsi_proceed": G1_NSI_PROCEED,
            "spearman_rho_proceed": G1_RHO_PROCEED,
            "min_null_passes": G1_MIN_NULL_PASSES,
            "nsi_stop": G1_NSI_STOP,
            "delta_auroc_stop": G1_DELTA_AUROC_STOP,
        },
        "per_operator": {op: dict(m) for op, m in metrics.items()},
        "warnings": warnings,
    }


def evaluate_gate_gsi_model(
    gsi_table: dict[str, dict[str, dict[str, dict[str, float | None]]]],
    *,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Per-model GSI rule over `{target: {op: {gap: {"value", "ci_lo"}}}}` (in-distribution gaps).

    A gap qualifies when, for at least one of `targets` and one geometry operator, GSI >= 1.0
    with a CI lower bound above 0.5. The model qualifies with at least two such gaps. `targets`
    defaults to every target in the table.
    """
    names = [t for t in gsi_table if targets is None or t in targets]
    by_gap: dict[int, list[dict[str, Any]]] = {}
    for tname in names:
        for op, gaps in gsi_table[tname].items():
            for gap, e in gaps.items():
                if _num(e.get("value")) and _num(e.get("ci_lo")):
                    if e["value"] >= GSI_PROCEED and e["ci_lo"] > GSI_CI_LO_PROCEED:
                        by_gap.setdefault(int(gap), []).append(
                            {"target": tname, "op": op, "gsi": e["value"], "ci_lo": e["ci_lo"]}
                        )
    gaps = sorted(by_gap)
    return {
        "targets_considered": names,
        "qualifying_gaps": gaps,
        "qualifying": {str(g): by_gap[g] for g in gaps},
        "model_qualifies": len(gaps) >= GSI_MIN_GAPS,
    }


def _gsi_table(geometry: dict, arm: str) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        op: {gap: per_arm[arm]["gsi"] for gap, per_arm in entry["gaps"].items()}
        for op, entry in geometry["per_operator"].items()
    }


def evaluate_gate_g1_gsi(targets: dict[str, dict], headline_targets: list[str]) -> dict[str, Any]:
    """Per-model GSI gate verdict inputs, on the published (token dropout on) arm.

    The headline variant considers `headline_targets` only; `all_targets` is reported next to
    it because MHC_HEAVY is diagnostic. The cross-model rule (at least two models qualify)
    is applied by `cla.eval.aggregate`.
    """
    out: dict[str, Any] = {
        "rule": (
            f"for at least {GSI_MIN_GAPS} in-distribution gaps on at least {GSI_MIN_MODELS} "
            f"models, GSI >= {GSI_PROCEED} for at least one target with CI lower bound "
            f"> {GSI_CI_LO_PROCEED}"
        ),
        "headline_arm": "token_dropout=true",
        "headline_targets": headline_targets,
    }
    for arm, key in (("true", "token_dropout_on"), ("false", "token_dropout_off")):
        table = {name: _gsi_table(t["geometry"], arm) for name, t in targets.items()}
        out[key] = {
            "headline": evaluate_gate_gsi_model(table, targets=headline_targets),
            "all_targets": evaluate_gate_gsi_model(table),
        }
    return out


# --------------------------------------------------------------------------- audit core

Job = dict[str, Any]  # {"index": list[int], "rendered": list[RenderedLayout], "reasons": Counter}


def _render(spec: ChainSpec, tok, kwargs: dict):
    try:
        return render_naive(spec, tok, **kwargs), None
    except ValueError as e:
        return None, str(e)


def _short(err: str) -> str:
    return err.split(":", 1)[-1].strip()[:60]


def build_spec_job(
    fn,
    examples: list[ChainSpec],
    population: list[int],
    tok,
    render_kwargs: dict,
    rng: np.random.Generator,
    *,
    overrides: dict | None = None,
) -> Job:
    """Apply a ChainSpec operator to `population` and render each result."""
    idx, rendered, reasons = [], [], Counter()
    for i in population:
        try:
            s2, over = fn(examples[i], rng)
        except OperatorNotApplicable as e:
            reasons[f"not_applicable: {e}"] += 1
            continue
        r, err = _render(s2, tok, {**render_kwargs, **over, **(overrides or {})})
        if r is None:
            reasons["render_failed: " + _short(err)] += 1
            continue
        idx.append(i)
        rendered.append(r)
    return {"index": idx, "rendered": rendered, "reasons": reasons}


def max_position(r: RenderedLayout) -> int:
    n = int(r.input_ids.shape[0])
    return n - 1 if r.position_ids is None else int(r.position_ids.max())


def build_geometry_job(
    op: str, gap: int, originals: list[RenderedLayout], population: list[int], *, check_range: bool
) -> Job:
    """Apply a geometry operator to the baseline layouts of `population`.

    With `check_range`, examples whose largest position index passes ESM-2's trained range
    are excluded and counted, so an in-distribution gap never measures extrapolation.
    """
    idx, rendered, reasons = [], [], Counter()
    for i in population:
        try:
            r = GEOMETRY_OPERATORS[op](originals[i], gap)
        except KeyError:
            reasons["not_applicable: layout has no PEPTIDE chain"] += 1
            continue
        if check_range and max_position(r) > TRAINED_MAX_POSITION:
            reasons["position_past_trained_range"] += 1
            continue
        idx.append(i)
        rendered.append(r)
    return {"index": idx, "rendered": rendered, "reasons": reasons}


def _restrict(job: Job, valid: set[int]) -> Job:
    keep = [k for k, i in enumerate(job["index"]) if i in valid]
    return {
        "index": [job["index"][k] for k in keep],
        "rendered": [job["rendered"][k] for k in keep],
        "reasons": job["reasons"],
    }


def target_name(roles: list[str]) -> str:
    return "+".join(roles)


def _estimator_for(scoring: dict, name: str) -> tuple[str, dict[str, Any]]:
    """Estimator name and its parameters for one target."""
    est = scoring.get("estimators", {}).get(name, scoring.get("estimator", "span_pll"))
    if est not in ESTIMATORS:
        raise ValueError(f"unknown estimator {est!r} for target {name}")
    params = dict(scoring.get("windowed", {})) if est == "windowed_span_pll" else {}
    return est, params


def _audit_target(
    model,
    tok,
    roles: list[str],
    *,
    estimator: tuple[str, dict[str, Any]],
    batch_size: int,
    examples: list[ChainSpec],
    originals: list[RenderedLayout],
    positives: list[int],
    op_jobs: dict[str, Job],
    sweep: dict[str, Job],
    geo_jobs: dict[tuple[str, int], Job],
    c2_job: Job,
    labels: np.ndarray,
    cdr3b_len: np.ndarray,
    boot: dict,
    bins: tuple[float, ...],
    min_bin_n: int,
    seed: int,
    config: dict,
    events: JsonlLogger,
    render_kwargs: dict,
) -> dict[str, Any]:
    """Score every layout under both token-dropout arms for one target role set."""
    name = target_name(roles)
    est_name, est_params = estimator
    method = ESTIMATORS[est_name]
    mkw = {"boot": boot, "bins": bins, "min_bin_n": min_bin_n, "seed": seed}
    valid_idx = [i for i, o in enumerate(originals) if all(r in o.chain_slices for r in roles)]
    valid = set(valid_idx)
    if len(roles) != 1:
        raise ValueError("C1 mutates the context of a single scored role; use single-role targets")
    gaps = [int(g) for g in config["geometry"]["gaps"]]
    ood_gaps = [int(g) for g in config["geometry"].get("out_of_distribution", [])]
    c1_ns = [int(n) for n in config["controls"]["c1_n_mutations"]]
    gsi_n = int(config["controls"]["gsi_denominator_n"])

    # C1 depends on the scored role (it mutates the other chains), so it is built per target.
    ctl_jobs: dict[str, Job] = {}
    for n in c1_ns:
        fn = functools.partial(CONTROLS["C1"], n_mutations=n, target_role=roles[0])
        ctl_jobs[f"C1:{n}"] = build_spec_job(
            fn,
            examples,
            [i for i in positives if i in valid],
            tok,
            render_kwargs,
            np.random.default_rng([seed, 300, n]),
        )
    if roles[0] == Role.TCRB.value:
        ctl_jobs["C2"] = {
            "index": [],
            "rendered": [],
            "reasons": Counter({"not_applicable: C2 swaps the scored TCRB chain": 1}),
        }
    else:
        ctl_jobs["C2"] = c2_job

    named: dict[str, Job] = {}
    for op, j in op_jobs.items():
        named[op] = _restrict(j, valid)
    for c, j in sweep.items():
        named[f"sweep:{c}"] = _restrict(j, valid)
    for key, j in ctl_jobs.items():
        named[key] = _restrict(j, valid)
    gap0_done = False
    for (op, gap), j in geo_jobs.items():
        if gap == 0:
            # G1 and G2 render the identical layout at gap 0; one scoring serves both.
            if not gap0_done:
                named["G@0"] = _restrict(j, valid)
                gap0_done = True
            continue
        named[f"{op}@{gap}"] = _restrict(j, valid)

    def score(rendered, desc):
        return score_many(
            model,
            tok,
            rendered,
            roles,
            method=method,
            batch_size=batch_size,
            desc=desc,
            **est_params,
        )

    raw: dict[str, dict] = {}
    for arm, enabled in (("true", True), ("false", False)):
        tag = f"{name} td={enabled}"
        with token_dropout(model, enabled):
            orig = np.full(len(originals), np.nan)
            # Positives and decoys are scored as separate calls so the positive baseline
            # shares batch composition with the geometry jobs. Accelerator kernels are not
            # bitwise batch-invariant, and the gap-0 check compares for exact equality.
            pos_set = set(positives)
            for part in (
                [i for i in valid_idx if i in pos_set],
                [i for i in valid_idx if i not in pos_set],
            ):
                if part:
                    orig[part] = score([originals[i] for i in part], f"{tag} orig")
            scored = {}
            for key, j in named.items():
                if j["index"]:
                    t0 = time.perf_counter()
                    log.info("start %s %s n=%d", tag, key, len(j["index"]))
                    scored[key] = (np.asarray(j["index"]), score(j["rendered"], f"{tag} {key}"))
                    log.info("done %s %s in %.0fs", tag, key, time.perf_counter() - t0)
                    events.log("scored", target=name, job=key, token_dropout=enabled)
        raw[arm] = {"orig": orig, "scored": scored}

    # Inertness: a gap of 0 must reproduce the baseline score bit for bit.
    inertness: dict[str, Any] = {}
    for arm in ARMS:
        idx, sc = raw[arm]["scored"].get("G@0", (np.array([], dtype=int), np.array([])))
        diff = float(np.max(np.abs(sc - raw[arm]["orig"][idx]))) if idx.size else None
        inertness[arm] = {"max_abs_diff": diff, "n": int(idx.size)}
        if diff is None or diff != 0.0:
            raise RuntimeError(
                f"{name}: gap-0 geometry changed the score (max |diff|={diff}, n={idx.size}); "
                "the geometry operators are not inert"
            )

    on_off = float(np.nanmedian(np.abs(raw["true"]["orig"] - raw["false"]["orig"])))
    if on_off == 0.0:
        raise RuntimeError("token_dropout toggle left every score unchanged; the control is inert")

    def sc_of(arm: str, key: str):
        return raw[arm]["scored"].get(key)

    def geo_key(op: str, gap: int) -> str:
        return "G@0" if gap == 0 else f"{op}@{gap}"

    def geometry_block(op: str, gap_list: list[int]) -> dict[str, Any]:
        per_gap: dict[str, dict] = {}
        for gap in gap_list:
            per_arm = {}
            for arm in ARMS:
                g, c1 = sc_of(arm, geo_key(op, gap)), sc_of(arm, f"C1:{gsi_n}")
                per_arm[arm] = (
                    compute_geometry_entry(
                        raw[arm]["orig"], g[0], g[1], c1[0], c1[1], boot=boot, seed=seed
                    )
                    if g is not None and c1 is not None
                    else _null_geometry("geometry or C1 job produced no examples")
                )
            per_gap[str(gap)] = per_arm
        return {
            "gaps": per_gap,
            "applicability": {
                str(gap): {
                    "n_applied": len(geo_jobs[(op, gap)]["index"]),
                    "n_valid_population": len(valid.intersection(geo_jobs[(op, gap)]["index"])),
                    "exclusion_reasons": dict(geo_jobs[(op, gap)]["reasons"]),
                }
                for gap in gap_list
            },
        }

    geo_ops = list(config["geometry"]["operators"])
    geometry: dict[str, Any] = {
        "population": "positives",
        "denominator": {"control": "C1", "n_mutations": gsi_n},
        "gaps_in_distribution": gaps,
        "per_operator": {},
    }
    for op in geo_ops:
        block = geometry_block(op, gaps)
        block["dose_response"] = {}
        for arm in ARMS:
            by_gap = {}
            for gap in gaps:
                s = sc_of(arm, geo_key(op, gap))
                if s is not None and gap >= 1:
                    by_gap[gap] = s
            block["dose_response"][arm] = dose_response_shape(
                raw[arm]["orig"], by_gap, boot=boot, seed=seed
            )
        geometry["per_operator"][op] = block
    geometry["out_of_distribution"] = {
        "label": OOD_LABEL,
        "gaps": ood_gaps,
        "excluded_from_headline": True,
        "per_operator": {op: geometry_block(op, ood_gaps) for op in geo_ops} if ood_gaps else {},
    }
    geometry["inertness_check"] = {"passed": True, "gap": 0, "per_arm": inertness}
    geometry["token_dropout_internal_control"] = (
        "Geometry operators leave input_ids, attention_mask and sequence length unchanged, so "
        "token dropout's rescale factor is identical before and after. Any difference between "
        "the on and off arms therefore reflects how the model's response to geometry depends on "
        "token dropout, with no length or content change confounding it."
    )

    def control_entry(arm: str, key: str) -> dict[str, Any]:
        s = sc_of(arm, key)
        if s is None:
            return {"n_pairs": 0, "median_abs_delta": None, "reason": "no applicable examples"}
        a = raw[arm]["orig"][s[0]]
        ci = _boot_entry(
            lambda x, y: float(np.median(np.abs(x - y))), a, s[1], boot=boot, seed=seed
        )
        return {
            "n_pairs": int(s[0].size),
            "median_abs_delta": float(np.median(np.abs(a - s[1]))),
            "median_abs_delta_ci": [ci["ci_lo"], ci["ci_hi"]],
        }

    controls = {
        "C1": {str(n): {arm: control_entry(arm, f"C1:{n}") for arm in ARMS} for n in c1_ns},
        "C2": {
            "applicability": {
                "n_applied": len(ctl_jobs["C2"]["index"]),
                "exclusion_reasons": dict(ctl_jobs["C2"]["reasons"]),
            },
            "token_dropout": {arm: control_entry(arm, "C2") for arm in ARMS},
        },
    }

    def block(arm: str, key: str) -> dict[str, Any]:
        idx, pert = raw[arm]["scored"][key]
        orig = raw[arm]["orig"]
        ref = np.asarray(valid_idx)
        return compute_metrics(
            orig[idx],
            pert,
            labels[idx],
            cdr3b_len[idx],
            ref_orig=orig[ref],
            ref_labels=labels[ref],
            **mkw,
        )

    operators: dict[str, dict] = {}
    for op in op_jobs:
        job = named[op]
        entry = {
            "kind": "null",
            "applicability": {
                "n_attempted": len(valid_idx),
                "n_applied": len(job["index"]),
                "n_excluded": len(valid_idx) - len(job["index"]),
                "exclusion_reasons": dict(job["reasons"]),
            },
            "token_dropout": {
                arm: block(arm, op) if op in raw[arm]["scored"] else _empty_block("no examples")
                for arm in ARMS
            },
        }
        if op == "P4":
            entry["spacer_length"] = config.get("spacer_length")
        operators[op] = entry
    p2_sweep = {
        c: {
            "applicability": {
                "n_applied": len(named[f"sweep:{c}"]["index"]),
                "exclusion_reasons": dict(named[f"sweep:{c}"]["reasons"]),
            },
            "token_dropout": {
                arm: block(arm, f"sweep:{c}")
                if f"sweep:{c}" in raw[arm]["scored"]
                else _empty_block("no examples")
                for arm in ARMS
            },
        }
        for c in sweep
    }

    def gate_inputs(arm: str) -> dict[str, dict]:
        return {
            op: {
                "nsi": e["token_dropout"][arm]["nsi"]["value"],
                "spearman_rho": e["token_dropout"][arm]["spearman_rho"],
                "delta_auroc": e["token_dropout"][arm]["delta_auroc"],
            }
            for op, e in operators.items()
        }

    pre = evaluate_gate_g1(gate_inputs("true"))
    pre["token_dropout_off"] = evaluate_gate_g1(gate_inputs("false"))
    pre["headline_arm"] = "token_dropout=true"

    base_med = float(np.median(raw["true"]["orig"][valid_idx]))
    sanity = {
        "median_baseline_pll_dropout_on": base_med,
        "uniform_prior_reference": UNIFORM_PRIOR_PLL,
        "margin_over_uniform_prior": base_med - UNIFORM_PRIOR_PLL,
        "required_margin": ESTIMATOR_MIN_MARGIN,
        "usable": bool(base_med - UNIFORM_PRIOR_PLL >= ESTIMATOR_MIN_MARGIN),
    }
    events.log("target_done", target=name, estimator=est_name, **sanity)

    scores = {
        arm: {
            "original": [None if np.isnan(v) else float(v) for v in raw[arm]["orig"]],
            "perturbed": {
                key: {"index": idx.tolist(), "scores": sc.tolist()}
                for key, (idx, sc) in raw[arm]["scored"].items()
            },
        }
        for arm in ARMS
    }
    return {
        "target_roles": roles,
        "estimator": {"name": est_name, "params": est_params, "sanity": sanity},
        "n_valid": len(valid_idx),
        "discrimination": {
            arm: compute_discrimination(
                raw[arm]["orig"][valid_idx], labels[valid_idx], boot=boot, seed=seed
            )
            for arm in ARMS
        },
        "token_dropout_control": {
            "mechanism": "model.esm.embeddings.token_dropout and model.config.token_dropout "
            "set together; module attribute is what forward reads",
            "median_abs_original_score_diff_on_vs_off": on_off,
        },
        "geometry": geometry,
        "controls": controls,
        "operators": operators,
        "p2_separator_sweep": p2_sweep,
        "scores": scores,
        "gate_g1_preregistered": pre,
    }


def _null_geometry(reason: str) -> dict[str, Any]:
    return {
        "n_pairs": 0,
        "median_abs_delta": None,
        "median_abs_delta_ci": None,
        "c1_denominator": None,
        "gsi": {"value": None, "ci_lo": None, "ci_hi": None, "reason": reason},
    }


def _empty_block(reason: str) -> dict[str, Any]:
    """Placeholder with the gate-facing fields set to null and the reason recorded."""
    return {
        "n_pairs": 0,
        "nsi": {"value": None, "ci_lo": None, "ci_hi": None, "reason": reason},
        "spearman_rho": None,
        "delta_auroc": None,
        "decision_flip_rate": None,
        "null_reasons": {"all": reason},
    }


def _p1_arm_agreement(targets_scores: dict, operators: dict) -> dict[str, Any]:
    """Agreement of P1's shifts between token-dropout arms; P1 changes no length."""
    if "P1" not in operators:
        return {"available": False, "reason": "P1 not in config operators"}
    arms = {}
    for arm in ARMS:
        p = targets_scores[arm]["perturbed"].get("P1")
        if not p or not p["index"]:
            return {"available": False, "reason": f"no P1 pairs in token_dropout={arm} arm"}
        idx = np.asarray(p["index"])
        orig = np.array([np.nan if v is None else v for v in targets_scores[arm]["original"]])
        arms[arm] = np.asarray(p["scores"]) - orig[idx]
    shift_rho = None
    if arms["true"].size > 2:
        r = float(rank_stability(arms["true"], arms["false"])["spearman_rho"])
        shift_rho = r if np.isfinite(r) else None
    return {
        "available": True,
        "shift_spearman_rho": shift_rho,
        "median_abs_diff_of_shifts": float(np.median(np.abs(arms["true"] - arms["false"]))),
        "median_abs_shift_dropout_on": float(np.median(np.abs(arms["true"]))),
        "median_abs_shift_dropout_off": float(np.median(np.abs(arms["false"]))),
        "n_pairs": int(arms["true"].size),
    }


def run_audit(
    config: dict,
    specs: list[ChainSpec],
    *,
    out_dir: Path,
    runs_root: Path,
    device: str | None,
    data_files: list[Path] = (),
) -> dict[str, Any]:
    """Run the audit on `specs` and write `out_dir/nsi.json`. Returns the result dict."""
    dev = resolve_device(None if device in (None, "auto") else device)
    set_all_seeds(config["seed"])
    seed = int(config["seed"])
    config = {**config, "device": str(dev)}

    manifest = build_manifest(config, data_files=data_files, device=str(dev))
    run_dir = runs_root / manifest.run_id
    write_manifest(manifest, run_dir)
    events = JsonlLogger(run_dir)
    events.log("run_start", run_id=manifest.run_id, device=str(dev))

    ops = list(config.get("operators", []))
    unknown = [o for o in ops if o not in OPERATORS or o == "P5"]
    if unknown:
        raise ValueError(f"unknown or unsupported operators in config: {unknown}; P5 is C2")
    geo_ops = list(config["geometry"]["operators"])
    unknown = [o for o in geo_ops if o not in GEOMETRY_OPERATORS]
    if unknown:
        raise ValueError(f"unknown geometry operators in config: {unknown}")
    gaps = [int(g) for g in config["geometry"]["gaps"]]
    ood_gaps = [int(g) for g in config["geometry"].get("out_of_distribution", [])]
    if 0 not in gaps:
        raise ValueError("geometry.gaps must include 0, the inertness check")
    if any(g > 150 for g in gaps):
        raise ValueError("in-distribution gaps are capped at 150; use geometry.out_of_distribution")
    gsi_n = int(config["controls"]["gsi_denominator_n"])
    if gsi_n not in [int(n) for n in config["controls"]["c1_n_mutations"]]:
        raise ValueError("controls.gsi_denominator_n must be one of controls.c1_n_mutations")

    sep_base = int(config.get("separator_tokens", 1))
    render_kwargs = {
        "separator_tokens": sep_base,
        "symmetric_ends": bool(config.get("symmetric_ends", True)),
        "max_length": int(config.get("max_length", 512)),
    }
    scoring = config.get("scoring", {})
    target_sets = [list(t) for t in scoring.get("targets", [["PEPTIDE"]])]
    if not target_sets or len({target_name(t) for t in target_sets}) != len(target_sets):
        raise ValueError(f"scoring.targets must be non-empty and distinct, got {target_sets}")
    batch_size = int(scoring.get("batch_size", 16))
    boot = {
        "n_boot": int(config["bootstrap"]["n_boot"]),
        "alpha": float(config["bootstrap"]["alpha"]),
    }
    bins = tuple(config.get("length_bins", DEFAULT_LENGTH_BINS))
    min_bin_n = int(config.get("min_bin_n", 20))
    headline_targets = list(config.get("headline_targets", [target_name(t) for t in target_sets]))

    # ---- sample
    shards = split_specs(specs)
    shard_sizes = {k: len(v) for k, v in shards.items()}
    sampling = config.get("sampling", {"mode": "stratified"})
    sampling_info: dict[str, Any] = {"mode": sampling["mode"]}
    if sampling["mode"] == "allele_balanced":
        counts = Counter(s.mhc_allele for s in specs if s.mhc_allele)
        cap, info = choose_allele_cap(dict(counts), tuple(sampling["total_range"]))
        shard_by_id = {s.example_id: k for k, v in shards.items() for s in v}
        drawn = allele_balanced_subsample(specs, cap, np.random.default_rng([seed, 1]))
        sample = [(s, shard_by_id[s.example_id]) for s in drawn]
        sampling_info |= {"per_allele_cap": cap, "n_alleles_available": len(counts), **info}
    elif sampling["mode"] == "stratified":
        sample = stratified_subsample(
            shards, int(config["n_examples"]), np.random.default_rng([seed, 1])
        )
    else:
        raise ValueError(f"unknown sampling mode {sampling['mode']!r}")
    positives_specs = [s for s, _ in sample]
    shard_of = {s.example_id: k for s, k in sample}
    negatives = make_mismatched_negatives(
        positives_specs,
        np.random.default_rng([seed, 2]),
        n_per_positive=int(config.get("negatives_per_positive", 1)),
        known_positives=specs,
    )
    for n in negatives:
        shard_of[n.example_id] = shard_of[n.example_id.rsplit("__neg", 1)[0]]
    candidates = positives_specs + negatives

    model, tok = load_esm(config["model"], dev)

    # ---- baseline layouts
    examples: list[ChainSpec] = []
    originals: list[RenderedLayout] = []
    dropped_orig = Counter()
    for s in candidates:
        r, err = _render(s, tok, render_kwargs)
        if r is None:
            dropped_orig[_short(err)] += 1
            continue
        examples.append(s)
        originals.append(r)
    labels = np.array([s.label for s in examples], dtype=float)
    cdr3b_len = np.array([len(s.by_role(Role.TCRB) or []) for s in examples], dtype=float)
    everyone = list(range(len(examples)))
    positives = [i for i in everyone if labels[i] == 1]
    events.log(
        "sampled",
        n_positive=len(positives_specs),
        n_negative=len(negatives),
        n_scored=len(examples),
    )

    # ---- operator layouts (built once, scored per target and arm)
    pool = sorted({c.sequence for s in specs if (c := s.by_role(Role.TCRB)) is not None})
    op_jobs: dict[str, Job] = {}
    for k, op in enumerate(ops):
        fn = functools.partial(OPERATORS[op], baseline=sep_base) if op == "P2" else OPERATORS[op]
        overrides = (
            {"spacer_before_peptide": int(config["spacer_length"])}
            if op == "P4" and config.get("spacer_length") is not None
            else None
        )
        op_jobs[op] = build_spec_job(
            fn,
            examples,
            everyone,
            tok,
            render_kwargs,
            np.random.default_rng([seed, 100 + k]),
            overrides=overrides,
        )

    sweep: dict[str, Job] = {}
    for c in config.get("separator_counts", []) if "P2" in ops else []:
        if int(c) == sep_base:
            continue  # renders the baseline exactly
        idx, rendered, reasons = [], [], Counter()
        for i in everyone:
            r, err = _render(examples[i], tok, {**render_kwargs, "separator_tokens": int(c)})
            if r is None:
                reasons["render_failed: " + _short(err)] += 1
                continue
            idx.append(i)
            rendered.append(r)
        sweep[str(c)] = {"index": idx, "rendered": rendered, "reasons": reasons}

    geo_jobs: dict[tuple[str, int], Job] = {}
    for op in geo_ops:
        for gap in gaps:
            geo_jobs[(op, gap)] = build_geometry_job(
                op, gap, originals, positives, check_range=True
            )
        for gap in ood_gaps:
            geo_jobs[(op, gap)] = build_geometry_job(
                op, gap, originals, positives, check_range=False
            )
    c2_job = build_spec_job(
        functools.partial(CONTROLS["C2"], pool=pool),
        examples,
        positives,
        tok,
        render_kwargs,
        np.random.default_rng([seed, 400]),
    )
    ood_positions = {
        f"{op}@{gap}": {
            "max_position_median": float(
                np.median([max_position(r) for r in geo_jobs[(op, gap)]["rendered"]])
            )
            if geo_jobs[(op, gap)]["rendered"]
            else None,
            "fraction_past_trained_range": float(
                np.mean(
                    [
                        max_position(r) > TRAINED_MAX_POSITION
                        for r in geo_jobs[(op, gap)]["rendered"]
                    ]
                )
            )
            if geo_jobs[(op, gap)]["rendered"]
            else None,
        }
        for op in geo_ops
        for gap in ood_gaps
    }

    # ---- scoring, once per target and token-dropout arm
    targets = {}
    for roles in target_sets:
        tname = target_name(roles)
        targets[tname] = _audit_target(
            model,
            tok,
            roles,
            estimator=_estimator_for(scoring, tname),
            batch_size=batch_size,
            examples=examples,
            originals=originals,
            positives=positives,
            op_jobs=op_jobs,
            sweep=sweep,
            geo_jobs=geo_jobs,
            c2_job=c2_job,
            labels=labels,
            cdr3b_len=cdr3b_len,
            boot=boot,
            bins=bins,
            min_bin_n=min_bin_n,
            seed=seed,
            config=config,
            events=events,
            render_kwargs=render_kwargs,
        )
        targets[tname]["geometry"]["out_of_distribution"]["max_positions"] = ood_positions
    for t in targets.values():
        t["p1_arm_agreement"] = _p1_arm_agreement(t["scores"], t["operators"])

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_name": config["run_name"],
        "model": config["model"],
        "manifest": asdict(manifest),
        "config": config,
        "baseline_rendering": render_kwargs,
        "reductions": config.get("reductions"),
        "scoring": {
            "targets": target_sets,
            "estimators": {n: t["estimator"] for n, t in targets.items()},
            "batch_size": batch_size,
            "nsi_min_standardized_separation": NSI_MIN_STANDARDIZED_SEPARATION,
        },
        "negatives": {
            "kind": "mismatched decoys, not validated non-binders",
            "n_per_positive": int(config.get("negatives_per_positive", 1)),
        },
        "data": {
            "sampling": sampling_info,
            "n_specs_total": len(specs),
            "shard_sizes": shard_sizes,
            "n_positive_sampled": len(positives_specs),
            "n_negative": len(negatives),
            "n_scored": len(examples),
            "n_positive_scored": len(positives),
            "dropped_original_render": dict(dropped_orig),
            "shard_counts_scored": dict(Counter(shard_of[s.example_id] for s in examples)),
        },
        "examples": {
            "example_id": [s.example_id for s in examples],
            "label": [int(s.label) for s in examples],
            "mhc_allele": [s.mhc_allele for s in examples],
            "shard": [shard_of[s.example_id] for s in examples],
            "cdr3b_length": cdr3b_len.astype(int).tolist(),
            "total_tokens": [int(r.input_ids.shape[0]) for r in originals],
        },
        "discrimination": {name: t["discrimination"] for name, t in targets.items()},
        "gate_g1": {
            "preregistered": {n: t["gate_g1_preregistered"] for n, t in targets.items()},
            "gsi": evaluate_gate_g1_gsi(targets, headline_targets),
        },
        "targets": targets,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nsi.json").write_text(
        json.dumps(result, indent=1, default=json_default, allow_nan=False) + "\n"
    )
    write_summary_tsv(result, out_dir / "summary.tsv")
    events.log(
        "gate_g1",
        preregistered={n: t["gate_g1_preregistered"]["verdict"] for n, t in targets.items()},
        gsi_model_qualifies=result["gate_g1"]["gsi"]["token_dropout_on"]["headline"][
            "model_qualifies"
        ],
    )
    events.close()
    return result


# --------------------------------------------------------------------------- reporting


SUMMARY_COLUMNS = (
    "target",
    "operator",
    "gap",
    "regime",
    "token_dropout",
    "median_abs_shift_nats",
    "c1_denominator_nats",
    "gsi",
    "gsi_ci_lo",
    "gsi_ci_hi",
    "n_pairs",
)


def summary_rows(result: dict[str, Any]) -> list[list[Any]]:
    """Flat rows, one per (target, operator, gap, token dropout arm).

    Geometry rows carry GSI. Control and content-operator rows carry the raw shift only,
    with empty GSI columns and an empty gap.
    """
    rows: list[list[Any]] = []
    for tname, t in result["targets"].items():
        geo = t["geometry"]
        c1 = t["controls"]["C1"][str(geo["denominator"]["n_mutations"])]
        groups = [
            ("in_distribution", geo["per_operator"]),
            ("out_of_distribution", geo["out_of_distribution"]["per_operator"]),
        ]
        for regime, per_op in groups:
            for op, entry in per_op.items():
                for gap, per_arm in entry["gaps"].items():
                    for arm in ARMS:
                        e = per_arm[arm]
                        g = e["gsi"]
                        rows.append(
                            [
                                tname,
                                op,
                                gap,
                                regime,
                                arm,
                                e["median_abs_delta"],
                                e["c1_denominator"],
                                g["value"],
                                g["ci_lo"],
                                g["ci_hi"],
                                e["n_pairs"],
                            ]
                        )
        for n, per_arm in t["controls"]["C1"].items():
            for arm in ARMS:
                e = per_arm[arm]
                rows.append(
                    [
                        tname,
                        f"C1(n={n})",
                        "",
                        "control",
                        arm,
                        e["median_abs_delta"],
                        c1[arm]["median_abs_delta"],
                        None,
                        None,
                        None,
                        e["n_pairs"],
                    ]
                )
        for arm in ARMS:
            e = t["controls"]["C2"]["token_dropout"][arm]
            rows.append(
                [
                    tname,
                    "C2",
                    "",
                    "control",
                    arm,
                    e["median_abs_delta"],
                    c1[arm]["median_abs_delta"],
                    None,
                    None,
                    None,
                    e["n_pairs"],
                ]
            )
        content = dict(t["operators"])
        content.update({f"P2(sep={c})": e for c, e in t["p2_separator_sweep"].items()})
        for op, entry in content.items():
            for arm in ARMS:
                m = entry["token_dropout"][arm]
                rows.append(
                    [
                        tname,
                        op,
                        "",
                        "content_coupled",
                        arm,
                        m.get("median_abs_shift"),
                        c1[arm]["median_abs_delta"],
                        None,
                        None,
                        None,
                        m["n_pairs"],
                    ]
                )
    return rows


def write_summary_tsv(result: dict[str, Any], path: Path) -> None:
    def cell(v) -> str:
        return "" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))

    lines = ["\t".join(SUMMARY_COLUMNS)]
    lines += ["\t".join(cell(v) for v in row) for row in summary_rows(result)]
    path.write_text("\n".join(lines) + "\n")


def _fmt(x, spec=".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def print_summary(result: dict[str, Any]) -> None:
    d = result["data"]
    print(f"\n== {result['run_name']} ({result['model']}) ==")
    print(
        f"examples scored: {d['n_scored']} "
        f"({d['n_positive_sampled']} positives, {d['n_negative']} decoys; "
        f"shards {d['shard_counts_scored']})"
    )
    for name, t in result["targets"].items():
        est = t["estimator"]
        print(
            f"\n-- target {name} (estimator {est['name']}, median pLL "
            f"{_fmt(est['sanity']['median_baseline_pll_dropout_on'])} vs prior "
            f"{_fmt(UNIFORM_PRIOR_PLL)}) --"
        )
        for arm in ARMS:
            dc = t["discrimination"][arm]
            a = dc["auroc"]
            print(
                f"discrimination dropout {'on' if arm == 'true' else 'off'}: "
                f"binder {_fmt(dc['binder_median'], '.4f')} "
                f"decoy {_fmt(dc['decoy_median'], '.4f')} "
                f"AUROC {_fmt(a['value'])} [{_fmt(a['ci_lo'])}, {_fmt(a['ci_hi'])}]"
            )
        denom = t["controls"]["C1"][str(t["geometry"]["denominator"]["n_mutations"])]["true"]
        print(f"C1 denominator (dropout on): {_fmt(denom['median_abs_delta'], '.4f')} nats")
        header = f"{'op':<5}{'gap':>5}{'|delta|':>10}{'GSI [95% CI] on':>26}{'GSI off':>10}"
        print(header)
        print("-" * len(header))
        for op, entry in t["geometry"]["per_operator"].items():
            for gap, per_arm in entry["gaps"].items():
                on, off = per_arm["true"], per_arm["false"]
                g_on = on["gsi"]
                ci = f"{_fmt(g_on['value'])} [{_fmt(g_on['ci_lo'])}, {_fmt(g_on['ci_hi'])}]"
                print(
                    f"{op:<5}{gap:>5}{_fmt(on['median_abs_delta'], '.4f'):>10}{ci:>26}"
                    f"{_fmt(off['gsi']['value']):>10}"
                )
            shape = entry["dose_response"]["true"]
            print(f"  {op} dose-response shape (dropout on): {shape['shape']}")
    g = result["gate_g1"]["gsi"]["token_dropout_on"]
    print(
        f"\nGSI gate (this model, dropout on): "
        f"headline qualifies={g['headline']['model_qualifies']} "
        f"gaps {g['headline']['qualifying_gaps']}; all targets qualifies="
        f"{g['all_targets']['model_qualifies']}"
    )
    for n, pre in result["gate_g1"]["preregistered"].items():
        print(f"Gate G1 pre-registered [{n}] (dropout on): {pre['verdict']} {pre['reason'] or ''}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap n_examples (smoke runs)")
    ap.add_argument("--device", choices=["cpu", "mps"], default=None)
    ap.add_argument("--out", type=Path, default=Path("results/phase1"))
    args = ap.parse_args(argv)

    config = load_config(args.config)
    if args.limit is not None:
        config["n_examples"] = min(int(config["n_examples"]), args.limit)
    specs_path = Path(config.get("data", {}).get("specs", "data/processed/vdjdb_specs.jsonl"))
    if not specs_path.exists():
        print(f"error: {specs_path} not found. Run `make data` first.", file=sys.stderr)
        return 2

    device = args.device or config.get("device")
    specs = load_specs_jsonl(specs_path)
    result = run_audit(
        config,
        specs,
        out_dir=args.out / config["run_name"],
        runs_root=Path("runs"),
        device=device,
        data_files=[specs_path],
    )
    print_summary(result)
    print(f"\nwrote {args.out / config['run_name'] / 'nsi.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
