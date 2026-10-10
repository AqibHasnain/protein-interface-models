"""Phase 1 nuisance-sensitivity audit.

    python -m cla.eval.audit --config configs/audit/esm2_35m.yaml [--limit N] [--device cpu|mps]

Each example is scored once in its original layout, then again after every relayout
operator, under two token-dropout arms (see docs/DECISIONS.md). Scores are the span
log-likelihood of the configured target roles given the rest of the complex.

Non-binders are mismatched-pairing decoys from `cla.eval.negatives`; they deflate NSI, so
the reported NSI is conservative.
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
    out["length_stratified_nsi"] = attempt(
        "length_stratified_nsi",
        lambda: length_stratified_nsi(
            orig, pert, cdr3b_len, binders, nonbinders, bins=bins, min_n=min_bin_n
        ),
    )
    out["null_reasons"] = reasons
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
    toward STOP.
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

    if len(passing) >= G1_MIN_NULL_PASSES and p5_moves_more:
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


def _gate_inputs(operators: dict[str, dict], arm: str) -> dict[str, dict[str, float | None]]:
    out = {}
    for op, entry in operators.items():
        m = entry["token_dropout"].get(arm)
        if m is None:
            continue
        out[op] = {
            "nsi": m["nsi"]["value"],
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
        "nsi": {"value": None, "ci_lo": None, "ci_hi": None},
        "spearman_rho": None,
        "delta_auroc": None,
        "decision_flip_rate": None,
        "null_reasons": {"all": reason},
    }


def _p1_arm_agreement(scores: dict, operators: dict) -> dict[str, Any]:
    if "P1" not in operators:
        return {"available": False, "reason": "P1 not in config operators"}
    arms = {}
    for arm in ("true", "false"):
        p = scores[arm]["perturbed"].get("P1")
        if not p or not p["index"]:
            return {"available": False, "reason": f"no P1 pairs in token_dropout={arm} arm"}
        idx = np.asarray(p["index"])
        arms[arm] = np.asarray(p["scores"]) - np.asarray(scores[arm]["original"])[idx]
    n_on, n_off = (operators["P1"]["token_dropout"][a]["nsi"]["value"] for a in ("true", "false"))
    shift_rho = None
    if arms["true"].size > 2:
        r = float(rank_stability(arms["true"], arms["false"])["spearman_rho"])
        shift_rho = r if np.isfinite(r) else None
    return {
        "available": True,
        "nsi_dropout_on": n_on,
        "nsi_dropout_off": n_off,
        "nsi_abs_diff": abs(n_on - n_off) if n_on is not None and n_off is not None else None,
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

    ops = list(config["operators"])
    unknown = [o for o in ops if o not in OPERATORS]
    if unknown:
        raise ValueError(f"unknown operators in config: {unknown}")
    sep_base = int(config.get("separator_tokens", 1))
    render_kwargs = {"separator_tokens": sep_base, "max_length": int(config.get("max_length", 512))}
    scoring = config.get("scoring", {})
    estimator = scoring.get("estimator", "span_pll")
    method = ESTIMATORS[estimator]
    roles = list(scoring.get("target_roles", ["PEPTIDE"]))
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

    # ---- scoring, once per token-dropout arm
    scores: dict[str, dict] = {}
    metrics: dict[str, dict[str, dict]] = {}  # arm -> key -> metrics
    sweep_metrics: dict[str, dict[str, dict]] = {}
    mkw = {"boot": boot, "bins": bins, "min_bin_n": min_bin_n, "seed": seed}

    def score(rendered, desc):
        return score_many(
            model, tok, rendered, roles, method=method, batch_size=batch_size, desc=desc
        )

    for arm_name, enabled in (("true", True), ("false", False)):
        with token_dropout(model, enabled):
            orig = score(originals, f"orig td={enabled}")
            scores[arm_name] = {"original": orig.tolist(), "perturbed": {}}
            metrics[arm_name], sweep_metrics[arm_name] = {}, {}
            for op in ops:
                job = jobs[op]
                if not job["index"]:
                    metrics[arm_name][op] = _empty_block("no applicable examples")
                    continue
                pert = score(job["rendered"], f"{op} td={enabled}")
                ix = np.asarray(job["index"])
                scores[arm_name]["perturbed"][op] = {
                    "index": job["index"],
                    "scores": pert.tolist(),
                }
                metrics[arm_name][op] = compute_metrics(
                    orig[ix], pert, labels[ix], cdr3b_len[ix], **mkw
                )
                events.log("operator_done", op=op, token_dropout=enabled)
            for c, job in sweep.items():
                if not job["index"]:
                    sweep_metrics[arm_name][c] = _empty_block("no examples rendered")
                    continue
                pert = score(job["rendered"], f"P2 sep={c} td={enabled}")
                ix = np.asarray(job["index"])
                sweep_metrics[arm_name][c] = compute_metrics(
                    orig[ix], pert, labels[ix], cdr3b_len[ix], **mkw
                )

    on, off = np.asarray(scores["true"]["original"]), np.asarray(scores["false"]["original"])
    on_off_shift = float(np.median(np.abs(on - off)))
    if on_off_shift == 0.0:
        raise RuntimeError("token_dropout toggle left every score unchanged; the control is inert")

    # ---- assemble
    operators: dict[str, dict] = {}
    for op in ops:
        job = jobs[op]
        operators[op] = {
            "kind": "positive_control" if op == POSITIVE_CONTROL else "null",
            "applicability": {
                "n_attempted": len(examples),
                "n_applied": len(job["index"]),
                "n_excluded": len(examples) - len(job["index"]),
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

    gate = evaluate_gate_g1(_gate_inputs(operators, "true"))
    gate["token_dropout_off"] = evaluate_gate_g1(_gate_inputs(operators, "false"))
    gate["headline_arm"] = "token_dropout=true"

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_name": config["run_name"],
        "model": config["model"],
        "manifest": asdict(manifest),
        "config": config,
        "scoring": {"estimator": estimator, "target_roles": roles, "batch_size": batch_size},
        "negatives": {
            "kind": "mismatched decoys, not validated non-binders",
            "bias_direction": "decoys inflate binder/non-binder separation, which deflates NSI",
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
        "token_dropout_control": {
            "mechanism": "model.esm.embeddings.token_dropout and model.config.token_dropout "
            "set together; module attribute is what forward reads",
            "median_abs_original_score_diff_on_vs_off": on_off_shift,
        },
        "p1_arm_agreement": _p1_arm_agreement(scores, operators),
        "operators": operators,
        "p2_separator_sweep": p2_sweep,
        "scores": scores,
        "gate_g1": gate,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nsi.json").write_text(
        json.dumps(result, indent=1, default=json_default, allow_nan=False) + "\n"
    )
    events.log("gate_g1", verdict=gate["verdict"])
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
    print(
        "NSI uses mismatched decoys as non-binders; decoys deflate NSI, so values are conservative."
    )
    header = (
        f"{'op':<10}{'dropout':<9}{'n':>6}{'NSI [95% CI]':>28}{'gap':>9}"
        f"{'rho':>8}{'dAUROC':>9}{'flip':>8}"
    )
    print(header)
    print("-" * len(header))

    def row(name, arm, m):
        ci = f"{_fmt(m['nsi']['value'])} [{_fmt(m['nsi']['ci_lo'])}, {_fmt(m['nsi']['ci_hi'])}]"
        print(
            f"{name:<10}{arm:<9}{m['n_pairs']:>6}{ci:>28}{_fmt(m.get('separation'), '.4f'):>9}"
            f"{_fmt(m['spearman_rho']):>8}"
            f"{_fmt(m['delta_auroc'], '+.3f'):>9}{_fmt(m.get('decision_flip_rate')):>8}"
        )

    for op, entry in result["operators"].items():
        for arm in ("true", "false"):
            row(op, "on" if arm == "true" else "off", entry["token_dropout"][arm])
        if entry["applicability"]["n_excluded"]:
            print(f"  {op}: excluded {entry['applicability']['exclusion_reasons']}")
    for c, entry in result["p2_separator_sweep"].items():
        for arm in ("true", "false"):
            m = entry["token_dropout"][arm]
            if m is not None:
                row(f"P2 sep={c}", "on" if arm == "true" else "off", m)
    agree = result["p1_arm_agreement"]
    print(
        "P1 arm agreement: "
        + (
            f"NSI on={_fmt(agree['nsi_dropout_on'])} off={_fmt(agree['nsi_dropout_off'])} "
            f"|diff|={_fmt(agree['nsi_abs_diff'])}"
            if agree["available"]
            else f"unavailable ({agree['reason']})"
        )
    )
    g = result["gate_g1"]
    print(
        f"\nGate G1 (token_dropout=on): {g['verdict']}  "
        f"[null ops passing {g['n_null_passing']} (need {G1_MIN_NULL_PASSES}) "
        f"{g['null_operators_passing']}; "
        f"P5 NSI {_fmt(g['p5_nsi'])} vs max null {_fmt(g['max_null_nsi'])}; "
        f"max |dAUROC| null {_fmt(g['max_abs_delta_auroc_null'])}]"
    )
    print(f"Gate G1 (token_dropout=off, informational): {g['token_dropout_off']['verdict']}")
    for w in g["warnings"]:
        print(f"WARNING: {w}")


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
