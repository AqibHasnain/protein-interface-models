"""Zero-shot test of the position-only components of `ChainAwarePositions`.

    python -m cla.eval.zeroshot_positional --device mps

Reproduces the Phase 1 allele-balanced MHC_HEAVY audit (150M, windowed estimator, token
dropout on, G1 gap nuisance, C1 with 20 mutations as the GSI denominator) with the module
applied to every layout. Per-chain reset and chain-break offset are `position_ids` remaps, so
no training is needed. The chain embedding has no zero-shot meaning (it starts at zero) and
is not evaluated here.

Order of operations per arm: the nuisance gap is applied to the baseline layout first, then
the module. With reset enabled the module rebuilds every index from `chain_slices`, so the
incoming gap is discarded and the numerator is exactly zero. That is a property of the
construction, so the reset arms assert equal `position_ids` instead of rescoring. The
informative reset-arm quantities are the denominator and the baseline score shift. Arms
without reset keep the incoming positions, so the gap adds on top of the offset.

Examples whose largest position index passes ESM-2's trained range are excluded per cell, as
in the Phase 1 audit. Each cell reports GSI for the matched baseline (A0) on the same
examples so arms are compared on identical populations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cla.eval.audit import (
    TRAINED_MAX_POSITION,
    _render,
    allele_balanced_subsample,
    choose_allele_cap,
    load_config,
    load_specs_jsonl,
    max_position,
    token_dropout,
)
from cla.eval.cluster_ci import _interval
from cla.eval.metrics import gsi
from cla.layout.geometry import GEOMETRY_OPERATORS
from cla.layout.perturb import CONTROLS
from cla.layout.spec import Role
from cla.model.backbone import load_esm, resolve_device
from cla.model.positional import ChainAwarePositions
from cla.model.scoring import score_many
from cla.utils.seeding import set_all_seeds

TARGET = Role.MHC_HEAVY.value
GAPS = (1, 5, 25, 100)
OFFSETS = (25, 50, 100, 200)
ARMS: dict[str, dict[str, Any]] = {
    "A0": {"reset": False, "offset": None},
    "A2": {"reset": True, "offset": None},
    **{f"A3_o{o}": {"reset": False, "offset": o} for o in OFFSETS},
    **{f"A2+A3_o{o}": {"reset": True, "offset": o} for o in OFFSETS},
}


def _ids(r) -> torch.Tensor:
    return r.position_ids if r.position_ids is not None else torch.arange(len(r.input_ids))


def _gsi(orig, geo, c1, mask, alleles, boot, seed) -> dict[str, Any]:
    if mask.sum() < 2 or len(np.unique(alleles[mask])) < 2:
        return {"n_pairs": int(mask.sum()), "point": None}
    cell = _interval(
        gsi, orig[mask], geo[mask], c1[mask], clusters=alleles[mask], boot=boot, seed=seed
    )
    cell["n_pairs"] = int(mask.sum())
    cell["median_abs_shift"] = float(np.median(np.abs(orig[mask] - geo[mask])))
    cell["median_abs_c1_shift"] = float(np.median(np.abs(orig[mask] - c1[mask])))
    return cell


def run(config: dict, specs, device: str | None) -> dict[str, Any]:
    seed = int(config["seed"])
    set_all_seeds(seed)
    boot = {
        "n_boot": int(config["bootstrap"]["n_boot"]),
        "alpha": float(config["bootstrap"]["alpha"]),
    }
    model, tok = load_esm(
        config["model"], resolve_device(None if device in (None, "auto") else device)
    )
    counts = Counter(s.mhc_allele for s in specs if s.mhc_allele)
    cap, info = choose_allele_cap(dict(counts), tuple(config["sampling"]["total_range"]))
    sample = allele_balanced_subsample(specs, cap, np.random.default_rng([seed, 1]))
    render_kw = {
        "separator_tokens": int(config.get("separator_tokens", 1)),
        "symmetric_ends": bool(config.get("symmetric_ends", True)),
        "max_length": int(config.get("max_length", 512)),
    }
    est = config["scoring"]["windowed"]
    n_mut = int(config["controls"]["gsi_denominator_n"])

    c1_fn = CONTROLS["C1"]
    c1_rng = np.random.default_rng([seed, 300, n_mut])
    keep, originals, c1_layouts = [], [], []
    for s in sample:
        r, _ = _render(s, tok, render_kw)
        if r is None or TARGET not in r.chain_slices:
            continue
        s2, over = c1_fn(s, c1_rng, n_mutations=n_mut, target_role=TARGET)
        c1_r, _ = _render(s2, tok, {**render_kw, **over})
        if c1_r is not None:
            keep.append(s)
            originals.append(r)
            c1_layouts.append(c1_r)
    alleles = np.array([s.mhc_allele for s in keep])
    gap_layouts = {g: [GEOMETRY_OPERATORS["G1"](o, g) for o in originals] for g in GAPS}

    def score(layouts) -> np.ndarray:
        with token_dropout(model, True):
            return score_many(
                model, tok, layouts, [TARGET], method="windowed", desc="zeroshot",
                batch_size=int(config["scoring"].get("batch_size", 16)), **est,
            )  # fmt: skip

    def in_range(layouts) -> np.ndarray:
        return np.array([max_position(r) <= TRAINED_MAX_POSITION for r in layouts])

    raw: dict[str, dict[str, Any]] = {}
    for arm, flags in ARMS.items():
        mod = ChainAwarePositions(1, chain_embedding=False, **flags)
        base = [mod.apply(o) for o in originals]
        c1 = [mod.apply(r) for r in c1_layouts]
        ok = in_range(base) & in_range(c1)
        entry: dict[str, Any] = {"ok": ok, "orig": score(base), "c1": score(c1), "geo": {}}
        for g in GAPS:
            gapped = [mod.apply(r) for r in gap_layouts[g]]
            if flags["reset"]:
                assert all(
                    torch.equal(_ids(a), _ids(b)) for a, b in zip(gapped, base, strict=True)
                ), "reset must erase the nuisance gap"
                entry["geo"][g] = (entry["orig"].copy(), ok)
            else:
                entry["geo"][g] = (score(gapped), ok & in_range(gapped))
        raw[arm] = entry
        print(f"scored {arm}: n_in_range={int(ok.sum())}", flush=True)

    a0 = raw["A0"]
    out: dict[str, Any] = {
        "model": config["model"],
        "target": TARGET,
        "estimator": {"name": "windowed_span_pll", **est},
        "token_dropout": True,
        "sample": {"n": len(keep), "n_alleles": len(set(alleles)), "per_allele_cap": cap, **info},
        "gaps": list(GAPS),
        "arms": {},
    }
    for arm, e in raw.items():
        cells = {}
        for g in GAPS:
            geo, ok = e["geo"][g]
            mask = ok & a0["ok"] & a0["geo"][g][1]
            cell = _gsi(e["orig"], geo, e["c1"], mask, alleles, boot, seed)
            cell["matched_a0"] = _gsi(
                a0["orig"], a0["geo"][g][0], a0["c1"], mask, alleles, boot, seed
            )
            cells[str(g)] = cell
        m = e["ok"] & a0["ok"]
        out["arms"][arm] = {
            **ARMS[arm],
            "n_in_range": int(e["ok"].sum()),
            "median_baseline_score": float(np.median(e["orig"][e["ok"]]))
            if e["ok"].any()
            else None,
            "median_abs_score_change_vs_a0": (
                float(np.median(np.abs(e["orig"][m] - a0["orig"][m]))) if m.any() else None
            ),
            "gsi": cells,
        }
    return out


def write_tsv(result: dict[str, Any], path: Path) -> None:
    head = ["arm", "reset", "offset", "gap", "gsi", "ci_lo", "ci_hi", "n_pairs", "shift_nats",
            "c1_nats", "matched_a0_gsi", "matched_a0_lo", "matched_a0_hi"]  # fmt: skip
    f = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
    rows = ["\t".join(head)]
    for arm, a in result["arms"].items():
        for g, c in a["gsi"].items():
            m = c.get("matched_a0", {})
            ci, mci = c.get("cluster_ci") or [None, None], m.get("cluster_ci") or [None, None]
            rows.append("\t".join([
                arm, str(a["reset"]), str(a["offset"]), g, f(c["point"]), f(ci[0]), f(ci[1]),
                str(c["n_pairs"]), f(c.get("median_abs_shift")), f(c.get("median_abs_c1_shift")),
                f(m.get("point")), f(mci[0]), f(mci[1]),
            ]))  # fmt: skip
    path.write_text("\n".join(rows) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", choices=["cpu", "mps"], default=None)
    ap.add_argument("--out", type=Path, default=Path("results/phase2"))
    args = ap.parse_args(argv)
    specs_path = Path("data/processed/vdjdb_specs.jsonl")
    if not specs_path.exists():
        print(f"error: {specs_path} not found. Run `make data` first.", file=sys.stderr)
        return 2
    config = load_config(Path("configs/audit/allele_balanced_150m.yaml"))
    result = run(config, load_specs_jsonl(specs_path), args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "zeroshot_gsi.json").write_text(json.dumps(result, indent=2) + "\n")
    write_tsv(result, args.out / "summary.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
