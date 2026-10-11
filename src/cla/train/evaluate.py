"""Held-out evaluation of a trained arm and the cross-arm summary table.

Everything here reads the `test_allele` and `test_epitope` shards, never `train`.

* Masked-LM loss with the training masking scheme (`cla.train.batching`) on held-out alleles and
  held-out epitopes. This is a language-modelling proxy for generalization. It is not binding
  AUROC, which the Phase 1 audit showed this backbone cannot produce.
* Median MHC_HEAVY windowed score and its margin over the uniform prior, so an arm that
  becomes a worse scorer is visible. MHC is never masked in training.
* Length sensitivity from `cla.eval.phase2_eval`: median |slope| of the MHC score against
  CDR3B length within matched sets of held-out-allele complexes sharing allele and peptide.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cla.data.splits import split_specs
from cla.eval.audit import UNIFORM_PRIOR_PLL
from cla.eval.metrics import cluster_bootstrap_ci
from cla.eval.phase2_eval import matched_sets, set_statistics
from cla.layout.concat import render_naive
from cla.layout.spec import ChainSpec, Role
from cla.model.scoring import score_many
from cla.train.batching import RENDER_KWARGS, collate

MLM_SHARDS = ("test_allele", "test_epitope")
MATCHED_MEMBER_CAP = 4
N_BOOT = 1000
SCORE_WINDOW = {"window": 15, "stride": 15, "max_windows": 4}
EVAL_BATCH = 16


class EvalData:
    def __init__(self, mlm: dict[str, list[ChainSpec]], sets: list[list[ChainSpec]]):
        self.mlm, self.sets = mlm, sets


def build_eval_data(specs: list[ChainSpec], seed: int, n_mlm: int) -> EvalData:
    shards = split_specs(specs)
    rng = np.random.default_rng([seed, 20])
    mlm = {}
    for name in MLM_SHARDS:
        pool = shards[name]
        pick = np.sort(rng.permutation(len(pool))[:n_mlm])
        mlm[name] = [pool[i] for i in pick]
    sets = matched_sets(
        shards["test_allele"], member_cap=MATCHED_MEMBER_CAP, rng=np.random.default_rng(0)
    )
    return EvalData(mlm, sets)


def _mlm(bound, tok, specs: list[ChainSpec], seed: int) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng([seed, 21])
    per_example: list[float] = []
    role_sum: dict[int, float] = defaultdict(float)
    role_n: dict[int, int] = defaultdict(int)
    for start in range(0, len(specs), EVAL_BATCH):
        batch = collate(specs[start : start + EVAL_BATCH], tok, bound.module, rng)
        batch = batch.to(next(bound.parameters()).device)
        logits = bound(
            batch.input_ids, batch.attention_mask, batch.position_ids, batch.chain_ids
        ).logits
        tok_loss = torch.nn.functional.cross_entropy(
            logits.float().transpose(1, 2), batch.target_ids, reduction="none"
        )
        masked = tok_loss * batch.loss_mask
        per_example.extend((masked.sum(1) / batch.loss_mask.sum(1)).tolist())
        for role_id in batch.chain_ids[batch.loss_mask].unique().tolist():
            sel = batch.loss_mask & (batch.chain_ids == role_id)
            role_sum[role_id] += float(tok_loss[sel].sum())
            role_n[role_id] += int(sel.sum())
    losses = np.array(per_example)
    roles = list(Role)
    return {
        "loss": float(losses.mean()),
        "se_examples": float(losses.std(ddof=1) / np.sqrt(len(losses))),
        "n": len(losses),
        "by_role": {roles[i - 1].value: role_sum[i] / role_n[i] for i in sorted(role_n)},
    }, losses


def evaluate(bound, tok, data: EvalData, *, step: int, seed: int):
    """Return (JSON-safe result, arrays for cross-arm paired comparisons)."""
    result: dict[str, Any] = {"step": step, "mlm": {}}
    arrays: dict[str, np.ndarray] = {}
    for name, specs in data.mlm.items():
        result["mlm"][name], arrays[f"mlm_{name}"] = _mlm(bound, tok, specs, seed)

    members = {m.example_id: m for ms in data.sets for m in ms}
    ids = list(members)
    layouts = [bound.module.apply(render_naive(members[e], tok, **RENDER_KWARGS)) for e in ids]
    scores = score_many(
        bound, tok, layouts, [Role.MHC_HEAVY.value], method="windowed", desc="eval",
        batch_size=EVAL_BATCH, **SCORE_WINDOW,
    )  # fmt: skip
    median = float(np.median(scores))
    result["mhc"] = {
        "n": len(ids),
        "median_score": median,
        "margin_over_uniform_prior": median - UNIFORM_PRIOR_PLL,
    }
    stats = set_statistics(data.sets, dict(zip(ids, scores, strict=True)), np.random.default_rng(4))
    alleles = np.array([ms[0].mhc_allele for ms in data.sets])
    med = lambda a: float(np.median(a))  # noqa: E731
    point, lo, hi = cluster_bootstrap_ci(
        med, stats["abs_slope"], clusters=alleles, n_boot=N_BOOT, seed=seed
    )
    excess = lambda a, n: med(a) - med(n)  # noqa: E731
    e_pt, e_lo, e_hi = cluster_bootstrap_ci(
        excess, stats["abs_slope"], stats["null"], clusters=alleles, n_boot=N_BOOT, seed=seed
    )
    result["length_sensitivity"] = {
        "n_sets": len(data.sets),
        "n_alleles": len(set(alleles)),
        "median_abs_slope": point,
        "ci": [lo, hi],
        "shuffled_null": med(stats["null"]),
        "excess_over_null": e_pt,
        "excess_ci": [e_lo, e_hi],
    }
    arrays |= {"abs_slope": stats["abs_slope"], "null": stats["null"], "set_alleles": alleles}
    return result, arrays


def save_eval_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


SUMMARY_COLUMNS = (
    "arm", "status", "steps", "trainable_params", "padding_params", "train_wall_clock_s",
    "allele_mlm_loss", "allele_mlm_se", "allele_delta_vs_a0", "allele_delta_se",
    "epitope_mlm_loss", "epitope_mlm_se", "epitope_delta_vs_a0", "epitope_delta_se",
    "mhc_median_score", "mhc_margin", "length_sens", "length_sens_lo", "length_sens_hi",
    "length_sens_excess", "length_sens_diff_vs_a0", "diff_lo", "diff_hi",
)  # fmt: skip


def summary_row(m: dict, a0: dict | None, arrays: dict, a0_arrays: dict | None) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in SUMMARY_COLUMNS}
    row |= {
        "arm": m["arm"],
        "status": m["status"],
        "steps": m["steps_run"],
        "trainable_params": m["trainable_params"]["total"],
        "padding_params": m["trainable_params"]["padding"],
        "train_wall_clock_s": round(m["train_wall_clock_s"]),
    }
    final = m["final"]
    if final is None:
        return row
    for key, tag in (("test_allele", "allele"), ("test_epitope", "epitope")):
        row[f"{tag}_mlm_loss"] = final["mlm"][key]["loss"]
        row[f"{tag}_mlm_se"] = final["mlm"][key]["se_examples"]
        if a0_arrays is not None and m["arm"] != "A0":
            d, se = _paired(arrays[f"mlm_{key}"], a0_arrays[f"mlm_{key}"])
            row[f"{tag}_delta_vs_a0"], row[f"{tag}_delta_se"] = d, se
    row["mhc_median_score"] = final["mhc"]["median_score"]
    row["mhc_margin"] = final["mhc"]["margin_over_uniform_prior"]
    ls = final["length_sensitivity"]
    row |= {
        "length_sens": ls["median_abs_slope"],
        "length_sens_lo": ls["ci"][0],
        "length_sens_hi": ls["ci"][1],
        "length_sens_excess": ls["excess_over_null"],
    }
    if a0_arrays is not None and m["arm"] != "A0":
        med = lambda a, b: float(np.median(a) - np.median(b))  # noqa: E731
        pt, lo, hi = cluster_bootstrap_ci(
            med, arrays["abs_slope"], a0_arrays["abs_slope"],
            clusters=arrays["set_alleles"], n_boot=N_BOOT, seed=0,
        )  # fmt: skip
        row |= {"length_sens_diff_vs_a0": pt, "diff_lo": lo, "diff_hi": hi}
    return row


def write_summary(root: Path) -> Path:
    import json

    metrics = {}
    for p in sorted(root.glob("A*/metrics.json")):
        m = json.loads(p.read_text())
        metrics[m["arm"]] = m
    arrays = {
        a: dict(np.load(root / a / "eval_arrays.npz"))
        for a, m in metrics.items()
        if m["final"] is not None
    }
    rows = [
        summary_row(m, metrics.get("A0"), arrays.get(a), arrays.get("A0"))
        for a, m in metrics.items()
    ]
    path = root / "summary.tsv"
    fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(SUMMARY_COLUMNS)
        for r in rows:
            w.writerow([fmt(r[c]) for c in SUMMARY_COLUMNS])
    return path
