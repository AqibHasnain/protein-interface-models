"""Phase 1 nuisance-sensitivity audit.

    python -m cla.eval.audit --config configs/audit/esm2_35m.yaml [--limit N] [--device cpu|mps]

Each example is scored once in its original layout, then again after every relayout
operator, under two token-dropout arms (see docs/DECISIONS.md). Scores are the span
log-likelihood of a target role set given the rest of the complex; every entry of
`scoring.targets` is audited separately and reported under `targets`.

NSI-C (normalized by the P5 positive control) is the primary metric. The pre-registered NSI
divides by the binder/decoy gap, which is unresolvable for an untrained backbone, so it is
reported as null when that gap is below `NSI_MIN_STANDARDIZED_SEPARATION` (in score SDs).
Decoys are mismatched pairings from `cla.eval.negatives`.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import sys
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
    length_stratified_nsi,
    nsi,
    nsi_control,
    rank_stability,
)
from cla.eval.negatives import make_mismatched_negatives
from cla.layout.concat import render_naive
from cla.layout.perturb import OPERATORS, OperatorNotApplicable
from cla.layout.spec import Chain, ChainSpec, Role
from cla.model.backbone import load_esm, resolve_device
from cla.model.scoring import score_many
from cla.utils.logging import JsonlLogger, get_logger, json_default
from cla.utils.manifest import build_manifest, write_manifest
from cla.utils.seeding import set_all_seeds

log = get_logger("cla.eval.audit")

SCHEMA_VERSION = 1
NULL_OPERATORS = ("P1", "P2", "P3", "P4")
POSITIVE_CONTROL = "P5"
ESTIMATORS = {"span_pll": "span", "pll_pseudo": "pseudo"}
DEFAULT_LENGTH_BINS = (0, 12, 14, 16, 18, 1000)

# Pre-registered Gate G1 thresholds. Moving these after seeing data invalidates the gate.
G1_NSI_PROCEED = 0.25
G1_RHO_PROCEED = 0.90
G1_NSI_STOP = 0.15
G1_DELTA_AUROC_STOP = 0.02
G1_MIN_NULL_PASSES = 2

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
) -> dict[str, Any]:
    """Full metric set for one (operator, arm). Uncomputable metrics are null with a reason."""
    reasons: dict[str, str] = {}
    out: dict[str, Any] = {"n_pairs": int(len(orig))}

    def attempt(name: str, fn):
        try:
            return fn()
        except (ValueError, FloatingPointError) as e:
            reasons[name] = str(e)
            return None

    binders, nonbinders = orig[labels == 1], orig[labels == 0]
    out["binder_median"] = float(np.median(binders)) if binders.size else None
    out["nonbinder_median"] = float(np.median(nonbinders)) if nonbinders.size else None
    out["separation"] = (
        abs(out["binder_median"] - out["nonbinder_median"])
        if binders.size and nonbinders.size
        else None
    )
    out["median_abs_shift"] = float(np.median(np.abs(orig - pert))) if len(orig) else None

    def nsi_fn(a, b, y):
        return nsi(a, b, a[y == 1], a[y == 0])

    spread = float(np.std(orig)) if len(orig) else 0.0
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
                labels,
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
                orig, pert, cdr3b_len, binders, nonbinders, bins=bins, min_n=min_bin_n
            ),
        )
    out["null_reasons"] = reasons
    return out


def compute_nsi_c(
    orig_all: np.ndarray,
    index: np.ndarray,
    pert: np.ndarray,
    control_index: np.ndarray,
    control_pert: np.ndarray,
    *,
    boot: dict,
    seed: int,
) -> dict[str, Any]:
    """NSI-C over the examples where both the operator and the P5 control applied.

    Restricting to the intersection keeps numerator and denominator on one example set, so
    the bootstrap resamples them jointly.
    """
    common, ia, ib = np.intersect1d(index, control_index, return_indices=True)
    n = int(common.size)
    if n == 0:
        return {
            "value": None,
            "reason": "no examples where both operator and P5 applied",
            "n_paired": 0,
        }
    try:
        value, lo, hi = bootstrap_ci(
            nsi_control,
            orig_all[common],
            pert[ia],
            control_pert[ib],
            n_boot=boot["n_boot"],
            alpha=boot["alpha"],
            seed=seed,
        )
    except ValueError as e:
        return {"value": None, "reason": str(e), "n_paired": n}
    if not np.isfinite(value):
        return {"value": None, "reason": "non-finite value", "n_paired": n}
    return {"value": value, "ci_lo": lo, "ci_hi": hi, "n_paired": n}


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
    p5_nsi = (metrics.get(POSITIVE_CONTROL) or {}).get("nsi")
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


def evaluate_gate_g1_nsi_c(metrics: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    """Gate G1 restated on NSI-C over `{op: {"nsi_c", "spearman_rho"}}`.

    PROCEED: at least two of P1, P2, P4 have NSI-C >= 0.25 and Spearman rho <= 0.90.
    UNDETERMINED: fewer than two pass but enough operators lack a value that the count could
    still reach two. NOT_PROCEED otherwise. The thresholds are the pre-registered ones.
    """
    nulls = {op: m for op, m in metrics.items() if op in NULL_OPERATORS}
    passing, failing, unknown = [], [], []
    for op, m in nulls.items():
        if not (_num(m.get("nsi_c")) and _num(m.get("spearman_rho"))):
            unknown.append(op)
        elif m["nsi_c"] >= G1_NSI_PROCEED and m["spearman_rho"] <= G1_RHO_PROCEED:
            passing.append(op)
        else:
            failing.append(op)
    if len(passing) >= G1_MIN_NULL_PASSES:
        verdict = "PROCEED"
    elif len(passing) + len(unknown) >= G1_MIN_NULL_PASSES:
        verdict = "UNDETERMINED"
    else:
        verdict = "NOT_PROCEED"
    return {
        "verdict": verdict,
        "null_operators_passing": passing,
        "null_operators_failing": failing,
        "null_operators_undetermined": unknown,
        "n_null_passing": len(passing),
        "thresholds": {
            "nsi_c_proceed": G1_NSI_PROCEED,
            "spearman_rho_proceed": G1_RHO_PROCEED,
            "min_null_passes": G1_MIN_NULL_PASSES,
        },
        "per_operator": {op: dict(m) for op, m in nulls.items()},
    }


def _gate_inputs(operators: dict[str, dict], arm: str) -> dict[str, dict[str, float | None]]:
    out = {}
    for op, entry in operators.items():
        m = entry["token_dropout"].get(arm)
        if m is None:
            continue
        out[op] = {
            "nsi": m["nsi"]["value"],
            "nsi_c": m["nsi_c"]["value"],
            "spearman_rho": m["spearman_rho"],
            "delta_auroc": m["delta_auroc"],
        }
    return out


# --------------------------------------------------------------------------- audit core


def _render(spec: ChainSpec, tok, kwargs: dict):
    try:
        return render_naive(spec, tok, **kwargs), None
    except ValueError as e:
        return None, str(e)


def _empty_block(reason: str) -> dict[str, Any]:
    """Placeholder with the gate-facing fields set to null and the reason recorded."""
    return {
        "n_pairs": 0,
        "nsi": {"value": None, "ci_lo": None, "ci_hi": None, "reason": reason},
        "nsi_c": {"value": None, "reason": reason, "n_paired": 0},
        "spearman_rho": None,
        "delta_auroc": None,
        "decision_flip_rate": None,
        "null_reasons": {"all": reason},
    }


def _p1_arm_agreement(scores: dict, operators: dict) -> dict[str, Any]:
    """Agreement of P1's shifts between token-dropout arms; P1 changes no length."""
    if "P1" not in operators:
        return {"available": False, "reason": "P1 not in config operators"}
    arms = {}
    for arm in ("true", "false"):
        p = scores[arm]["perturbed"].get("P1")
        if not p or not p["index"]:
            return {"available": False, "reason": f"no P1 pairs in token_dropout={arm} arm"}
        idx = np.asarray(p["index"])
        arms[arm] = np.asarray(p["scores"]) - np.asarray(scores[arm]["original"])[idx]
    n_on, n_off = (operators["P1"]["token_dropout"][a]["nsi_c"]["value"] for a in ("true", "false"))
    shift_rho = None
    if arms["true"].size > 2:
        r = float(rank_stability(arms["true"], arms["false"])["spearman_rho"])
        shift_rho = r if np.isfinite(r) else None
    return {
        "available": True,
        "nsi_c_dropout_on": n_on,
        "nsi_c_dropout_off": n_off,
        "nsi_c_abs_diff": abs(n_on - n_off) if n_on is not None and n_off is not None else None,
        "shift_spearman_rho": shift_rho,
        "median_abs_diff_of_shifts": float(np.median(np.abs(arms["true"] - arms["false"]))),
        "median_abs_shift_dropout_on": float(np.median(np.abs(arms["true"]))),
        "median_abs_shift_dropout_off": float(np.median(np.abs(arms["false"]))),
        "n_pairs": int(arms["true"].size),
    }


def target_name(roles: list[str]) -> str:
    return "+".join(roles)


def _audit_target(
    model,
    tok,
    roles: list[str],
    *,
    method: str,
    batch_size: int,
    originals: list,
    jobs: dict[str, dict[str, Any]],
    sweep: dict[str, dict[str, Any]],
    ops: list[str],
    labels: np.ndarray,
    cdr3b_len: np.ndarray,
    n_examples: int,
    boot: dict,
    bins: tuple[float, ...],
    min_bin_n: int,
    seed: int,
    config: dict,
    events: JsonlLogger,
) -> dict[str, Any]:
    """Score every layout under both token-dropout arms for one target role set."""
    name = target_name(roles)
    mkw = {"boot": boot, "bins": bins, "min_bin_n": min_bin_n, "seed": seed}
    raw: dict[str, dict] = {}  # arm -> {"orig", "ops": {op: scores}, "sweep": {c: scores}}

    def score(rendered, desc):
        return score_many(
            model, tok, rendered, roles, method=method, batch_size=batch_size, desc=desc
        )

    for arm_name, enabled in (("true", True), ("false", False)):
        tag = f"{name} td={enabled}"
        with token_dropout(model, enabled):
            arm = {"orig": score(originals, f"{tag} orig"), "ops": {}, "sweep": {}}
            for op in ops:
                if jobs[op]["index"]:
                    arm["ops"][op] = score(jobs[op]["rendered"], f"{tag} {op}")
                    events.log("operator_done", target=name, op=op, token_dropout=enabled)
            for c, job in sweep.items():
                if job["index"]:
                    arm["sweep"][c] = score(job["rendered"], f"{tag} P2 sep={c}")
        raw[arm_name] = arm

    def block(arm, orig, index, pert):
        ix = np.asarray(index)
        m = compute_metrics(orig[ix], pert, labels[ix], cdr3b_len[ix], **mkw)
        if POSITIVE_CONTROL not in arm["ops"]:
            m["nsi_c"] = {"value": None, "reason": "P5 not in config operators", "n_paired": 0}
        else:
            m["nsi_c"] = compute_nsi_c(
                orig,
                ix,
                pert,
                np.asarray(jobs[POSITIVE_CONTROL]["index"]),
                arm["ops"][POSITIVE_CONTROL],
                boot=boot,
                seed=seed,
            )
        return m

    metrics: dict[str, dict[str, dict]] = {}
    sweep_metrics: dict[str, dict[str, dict]] = {}
    for arm_name, arm in raw.items():
        metrics[arm_name], sweep_metrics[arm_name] = {}, {}
        for op in ops:
            if op not in arm["ops"]:
                metrics[arm_name][op] = _empty_block("no applicable examples")
                continue
            m = block(arm, arm["orig"], jobs[op]["index"], arm["ops"][op])
            if op == POSITIVE_CONTROL:
                m["nsi_c"] = {
                    "value": None,
                    "reason": "positive control is the NSI-C denominator",
                    "n_paired": m["n_pairs"],
                }
            metrics[arm_name][op] = m
        for c, job in sweep.items():
            sweep_metrics[arm_name][c] = (
                block(arm, arm["orig"], job["index"], arm["sweep"][c])
                if c in arm["sweep"]
                else _empty_block("no examples rendered")
            )

    scores = {
        arm_name: {
            "original": arm["orig"].tolist(),
            "perturbed": {
                op: {"index": jobs[op]["index"], "scores": arm["ops"][op].tolist()}
                for op in arm["ops"]
            },
        }
        for arm_name, arm in raw.items()
    }
    on, off = raw["true"]["orig"], raw["false"]["orig"]
    on_off_shift = float(np.median(np.abs(on - off)))
    if on_off_shift == 0.0:
        raise RuntimeError("token_dropout toggle left every score unchanged; the control is inert")

    operators: dict[str, dict] = {}
    for op in ops:
        job = jobs[op]
        operators[op] = {
            "kind": "positive_control" if op == POSITIVE_CONTROL else "null",
            "applicability": {
                "n_attempted": n_examples,
                "n_applied": len(job["index"]),
                "n_excluded": n_examples - len(job["index"]),
                "exclusion_reasons": dict(job["reasons"]),
            },
            "token_dropout": {arm: metrics[arm][op] for arm in ("true", "false")},
        }
        if op == "P4":
            operators[op]["spacer_length"] = config.get("spacer_length")

    p2_sweep = {
        c: {
            "applicability": {
                "n_applied": len(sweep[c]["index"]),
                "exclusion_reasons": dict(sweep[c]["reasons"]),
            },
            "token_dropout": {arm: sweep_metrics[arm][c] for arm in ("true", "false")},
        }
        for c in sweep
    }

    def gate(fn):
        g = fn(_gate_inputs(operators, "true"))
        g["token_dropout_off"] = fn(_gate_inputs(operators, "false"))
        g["headline_arm"] = "token_dropout=true"
        return g

    return {
        "target_roles": roles,
        "discrimination": {
            arm: compute_discrimination(raw[arm]["orig"], labels, boot=boot, seed=seed)
            for arm in ("true", "false")
        },
        "token_dropout_control": {
            "mechanism": "model.esm.embeddings.token_dropout and model.config.token_dropout "
            "set together; module attribute is what forward reads",
            "median_abs_original_score_diff_on_vs_off": on_off_shift,
        },
        "p1_arm_agreement": _p1_arm_agreement(scores, operators),
        "operators": operators,
        "p2_separator_sweep": p2_sweep,
        "scores": scores,
        "gate_g1": {
            "preregistered": gate(evaluate_gate_g1),
            "nsi_c": gate(evaluate_gate_g1_nsi_c),
        },
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

    ops = list(config["operators"])
    unknown = [o for o in ops if o not in OPERATORS]
    if unknown:
        raise ValueError(f"unknown operators in config: {unknown}")
    sep_base = int(config.get("separator_tokens", 1))
    render_kwargs = {"separator_tokens": sep_base, "max_length": int(config.get("max_length", 512))}
    scoring = config.get("scoring", {})
    estimator = scoring.get("estimator", "span_pll")
    method = ESTIMATORS[estimator]
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

    # ---- sample
    shards = split_specs(specs)
    shard_sizes = {k: len(v) for k, v in shards.items()}
    sample = stratified_subsample(
        shards, int(config["n_examples"]), np.random.default_rng([seed, 1])
    )
    positives = [s for s, _ in sample]
    shard_of = {s.example_id: k for s, k in sample}
    negatives = make_mismatched_negatives(
        positives,
        np.random.default_rng([seed, 2]),
        n_per_positive=int(config.get("negatives_per_positive", 1)),
        known_positives=specs,
    )
    for n in negatives:
        shard_of[n.example_id] = shard_of[n.example_id.rsplit("__neg", 1)[0]]
    candidates = positives + negatives

    model, tok = load_esm(config["model"], dev)

    # ---- originals
    examples: list[ChainSpec] = []
    originals = []
    dropped_orig = Counter()
    for s in candidates:
        r, err = _render(s, tok, render_kwargs)
        if r is None:
            dropped_orig[err.split(":", 1)[-1].strip()[:60]] += 1
            continue
        examples.append(s)
        originals.append(r)
    labels = np.array([s.label for s in examples], dtype=float)
    cdr3b_len = np.array([len(s.by_role(Role.TCRB) or []) for s in examples], dtype=float)
    events.log(
        "sampled", n_positive=len(positives), n_negative=len(negatives), n_scored=len(examples)
    )

    # ---- perturbed layouts (applied once, scored under both arms)
    pool = sorted({c.sequence for s in specs if (c := s.by_role(Role.TCRB)) is not None})
    jobs: dict[str, dict[str, Any]] = {}
    for k, op in enumerate(ops):
        fn = OPERATORS[op]
        if op == POSITIVE_CONTROL:
            fn = functools.partial(fn, pool=pool)
        rng = np.random.default_rng([seed, 100 + k])
        idx, rendered, reasons = [], [], Counter()
        for i, s in enumerate(examples):
            try:
                s2, over = fn(s, rng)
            except OperatorNotApplicable as e:
                reasons[f"not_applicable: {e}"] += 1
                continue
            if op == "P4" and config.get("spacer_length") is not None:
                over = {**over, "spacer_before_peptide": int(config["spacer_length"])}
            r, err = _render(s2, tok, {**render_kwargs, **over})
            if r is None:
                reasons["render_failed: " + err.split(":", 1)[-1].strip()[:60]] += 1
                continue
            idx.append(i)
            rendered.append(r)
        jobs[op] = {"index": idx, "rendered": rendered, "reasons": reasons}

    sweep: dict[str, dict[str, Any]] = {}
    for c in config.get("separator_counts", []) if "P2" in ops else []:
        idx, rendered, reasons = [], [], Counter()
        for i, s in enumerate(examples):
            r, err = _render(s, tok, {**render_kwargs, "separator_tokens": int(c)})
            if r is None:
                reasons["render_failed: " + err.split(":", 1)[-1].strip()[:60]] += 1
                continue
            idx.append(i)
            rendered.append(r)
        sweep[str(c)] = {"index": idx, "rendered": rendered, "reasons": reasons}

    # ---- scoring, once per target and token-dropout arm
    targets = {
        target_name(roles): _audit_target(
            model,
            tok,
            roles,
            method=method,
            batch_size=batch_size,
            originals=originals,
            jobs=jobs,
            sweep=sweep,
            ops=ops,
            labels=labels,
            cdr3b_len=cdr3b_len,
            n_examples=len(examples),
            boot=boot,
            bins=bins,
            min_bin_n=min_bin_n,
            seed=seed,
            config=config,
            events=events,
        )
        for roles in target_sets
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_name": config["run_name"],
        "model": config["model"],
        "manifest": asdict(manifest),
        "config": config,
        "scoring": {
            "estimator": estimator,
            "targets": target_sets,
            "batch_size": batch_size,
            "nsi_min_standardized_separation": NSI_MIN_STANDARDIZED_SEPARATION,
        },
        "negatives": {
            "kind": "mismatched decoys, not validated non-binders",
            "n_per_positive": int(config.get("negatives_per_positive", 1)),
        },
        "data": {
            "n_specs_total": len(specs),
            "shard_sizes": shard_sizes,
            "n_positive_sampled": len(positives),
            "n_negative": len(negatives),
            "n_scored": len(examples),
            "dropped_original_render": dict(dropped_orig),
            "shard_counts_scored": dict(Counter(shard_of[s.example_id] for s in examples)),
        },
        "examples": {
            "example_id": [s.example_id for s in examples],
            "label": [int(s.label) for s in examples],
            "shard": [shard_of[s.example_id] for s in examples],
            "cdr3b_length": cdr3b_len.astype(int).tolist(),
            "total_tokens": [int(r.input_ids.shape[0]) for r in originals],
        },
        "discrimination": {name: t["discrimination"] for name, t in targets.items()},
        "targets": targets,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nsi.json").write_text(
        json.dumps(result, indent=1, default=json_default, allow_nan=False) + "\n"
    )
    for name, t in targets.items():
        events.log(
            "gate_g1",
            target=name,
            preregistered=t["gate_g1"]["preregistered"]["verdict"],
            nsi_c=t["gate_g1"]["nsi_c"]["verdict"],
        )
    events.close()
    return result


# --------------------------------------------------------------------------- reporting


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
        print(f"\n-- target {name} --")
        for arm in ("true", "false"):
            dc = t["discrimination"][arm]
            a = dc["auroc"]
            print(
                f"discrimination dropout {'on' if arm == 'true' else 'off'}: "
                f"binder {_fmt(dc['binder_median'], '.4f')} "
                f"decoy {_fmt(dc['decoy_median'], '.4f')} "
                f"AUROC {_fmt(a['value'])} [{_fmt(a['ci_lo'])}, {_fmt(a['ci_hi'])}]"
            )
        header = (
            f"{'op':<10}{'dropout':<9}{'n':>6}{'NSI-C [95% CI]':>28}{'n_pair':>8}"
            f"{'rho':>8}{'shift':>9}{'flip':>8}"
        )
        print(header)
        print("-" * len(header))

        def row(label, arm, m):
            c = m["nsi_c"]
            ci = f"{_fmt(c['value'])} [{_fmt(c.get('ci_lo'))}, {_fmt(c.get('ci_hi'))}]"
            print(
                f"{label:<10}{arm:<9}{m['n_pairs']:>6}{ci:>28}{c.get('n_paired', 0):>8}"
                f"{_fmt(m['spearman_rho']):>8}{_fmt(m.get('median_abs_shift'), '.4f'):>9}"
                f"{_fmt(m.get('decision_flip_rate')):>8}"
            )

        for op, entry in t["operators"].items():
            for arm in ("true", "false"):
                row(op, "on" if arm == "true" else "off", entry["token_dropout"][arm])
            if entry["applicability"]["n_excluded"]:
                print(f"  {op}: excluded {entry['applicability']['exclusion_reasons']}")
        for c, entry in t["p2_separator_sweep"].items():
            for arm in ("true", "false"):
                row(f"P2 sep={c}", "on" if arm == "true" else "off", entry["token_dropout"][arm])
        g = t["gate_g1"]
        pre, nc = g["preregistered"], g["nsi_c"]
        print(
            f"Gate G1 pre-registered (dropout on): {pre['verdict']}"
            + (f" ({pre['reason']})" if pre["reason"] else "")
        )
        print(
            f"Gate G1 NSI-C (dropout on): {nc['verdict']} "
            f"[passing {nc['null_operators_passing']}, undetermined "
            f"{nc['null_operators_undetermined']}]"
        )
        print(f"Gate G1 NSI-C (dropout off): {nc['token_dropout_off']['verdict']}")


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
