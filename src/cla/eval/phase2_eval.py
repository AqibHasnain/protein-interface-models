"""Non-circular zero-shot evaluation of the chain-aware position module.

    python -m cla.eval.phase2_eval --device mps

Per-chain reset discards incoming position indices, so measuring it against an operator that
only edits indices (G1) returns zero by construction. The two operators here change real
sequence content or token order, which the module cannot undo:

* Operator L, native chain-length variation. Complexes that share MHC allele and peptide
  (hence an identical MHC_HEAVY chain) but differ in CDR3-beta length form a matched set. The
  MHC_HEAVY score is the admissible target, so its variation within a set comes from the
  receptor chains, and the part that tracks CDR3B length is the nuisance H1 is about. Under
  the baseline layout the CDR3B length shifts every downstream offset; per-chain reset removes
  that shift from the MHC indices. TCRA content also varies within a set and is not removed by
  any arm, so it appears as a floor common to all arms.
* Operator O, chain order permutation (P1). Token order really changes, so residual
  sensitivity from attention remains in every arm. An exact zero is a bug and raises.

Length sensitivity is the median over sets of |slope| of the MHC score against CDR3B length
within the set (nats per residue). Slope is preferred to range because the range grows with the
number of members, which differs across sets. Because |slope| is positive under pure noise, the
same statistic is recomputed with lengths shuffled within each set; the excess over that null
is the part attributable to length. Intervals are cluster bootstraps over alleles.

The primary matched sets come from the allele-balanced sample (cap 27 per allele). An extended
pool built from every spec with a per-set member cap is reported beside it because the balanced
sample leaves few sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cla.eval.audit import (
    TRAINED_MAX_POSITION,
    UNIFORM_PRIOR_PLL,
    _render,
    allele_balanced_subsample,
    choose_allele_cap,
    load_config,
    load_specs_jsonl,
    max_position,
    token_dropout,
)
from cla.eval.metrics import cluster_bootstrap_ci
from cla.layout.perturb import permute_chain_order
from cla.layout.spec import ChainSpec, Role
from cla.model.backbone import load_esm, resolve_device
from cla.model.positional import ChainAwarePositions
from cla.model.scoring import score_many
from cla.utils.seeding import set_all_seeds

TARGET = Role.MHC_HEAVY.value
USABLE_MARGIN = 0.25
EXTENDED_MEMBER_CAP = 8
N_NULL_SHUFFLES = 200
# Offsets 100 and 200 push five-chain layouts past ESM-2's 1026 positions, so they are not run.
ARMS: dict[str, dict[str, Any]] = {
    "A0": {"reset": False, "offset": None},
    "A2": {"reset": True, "offset": None},
    "A3_o25": {"reset": False, "offset": 25},
    "A3_o50": {"reset": False, "offset": 50},
    "A2+A3_o25": {"reset": True, "offset": 25},
    "A2+A3_o50": {"reset": True, "offset": 50},
}


def matched_sets(
    specs: list[ChainSpec], *, member_cap: int | None, rng: np.random.Generator
) -> list[list[ChainSpec]]:
    """Groups sharing allele and peptide with at least two distinct CDR3B lengths.

    With a `member_cap`, large groups are subsampled uniformly. A subsample that no longer
    spans two lengths is redrawn from the length-diverse members so the set stays valid.
    """
    groups: dict[tuple[str, str], list[ChainSpec]] = defaultdict(list)
    for s in sorted(specs, key=lambda s: s.example_id):
        if s.mhc_allele and s.epitope and s.by_role(Role.TCRB) and s.by_role(Role.MHC_HEAVY):
            groups[(s.mhc_allele, s.epitope)].append(s)
    out = []
    for key in sorted(groups):
        members = groups[key]
        if len({len(m.by_role(Role.TCRB)) for m in members}) < 2:
            continue
        if member_cap is not None and len(members) > member_cap:
            while True:
                pick = rng.choice(len(members), size=member_cap, replace=False)
                chosen = [members[int(i)] for i in sorted(pick)]
                if len({len(m.by_role(Role.TCRB)) for m in chosen}) >= 2:
                    break
            members = chosen
        out.append(members)
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc**2).sum())


def set_statistics(
    sets: list[list[ChainSpec]], score_of: dict[str, float], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Per-set |slope|, its length-shuffled null, and the score range across length bins."""
    abs_slope, null, spread = [], [], []
    for members in sets:
        x = np.array([len(m.by_role(Role.TCRB)) for m in members], dtype=float)
        y = np.array([score_of[m.example_id] for m in members])
        abs_slope.append(abs(_slope(x, y)))
        draws = []
        for _ in range(N_NULL_SHUFFLES):
            xs = rng.permutation(x)
            # A shuffle can collapse to a single length only if x does, which the set excludes.
            draws.append(abs(_slope(xs, y)) if np.ptp(xs) > 0 else 0.0)
        null.append(float(np.mean(draws)))
        bins = [y[x == v].mean() for v in np.unique(x)]
        spread.append(max(bins) - min(bins))
    return {
        "abs_slope": np.array(abs_slope),
        "null": np.array(null),
        "spread": np.array(spread),
    }


def _med(a: np.ndarray) -> float:
    return float(np.median(a))


def _ci(fn, *arrays, clusters, boot, seed) -> dict[str, Any]:
    point, lo, hi = cluster_bootstrap_ci(
        fn, *arrays, clusters=clusters, n_boot=boot["n_boot"], alpha=boot["alpha"], seed=seed
    )
    return {"point": point, "ci": [lo, hi]}


def length_block(
    stats: dict[str, dict[str, np.ndarray]], alleles: np.ndarray, boot: dict, seed: int
) -> dict[str, dict[str, Any]]:
    base = stats["A0"]
    out = {}
    for arm, st in stats.items():
        entry = {
            "length_sensitivity": _ci(
                _med, st["abs_slope"], clusters=alleles, boot=boot, seed=seed
            ),
            "shuffled_null": _ci(_med, st["null"], clusters=alleles, boot=boot, seed=seed),
            "excess_over_null": _ci(
                lambda a, n: _med(a) - _med(n), st["abs_slope"], st["null"],
                clusters=alleles, boot=boot, seed=seed,
            ),  # fmt: skip
            "median_spread_across_length_bins": _ci(
                _med, st["spread"], clusters=alleles, boot=boot, seed=seed
            ),
        }  # fmt: skip
        if arm != "A0":
            entry["diff_vs_a0"] = _ci(
                lambda a, b: _med(a) - _med(b), st["abs_slope"], base["abs_slope"],
                clusters=alleles, boot=boot, seed=seed,
            )  # fmt: skip
            entry["ratio_vs_a0"] = _ci(
                lambda a, b: _med(a) / _med(b), st["abs_slope"], base["abs_slope"],
                clusters=alleles, boot=boot, seed=seed,
            )  # fmt: skip
        out[arm] = entry
    return out


def run(config: dict, specs: list[ChainSpec], device: str | None) -> dict[str, Any]:
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

    set_sources = {
        "allele_balanced": matched_sets(sample, member_cap=None, rng=np.random.default_rng(0)),
        "extended_pool": matched_sets(
            specs, member_cap=EXTENDED_MEMBER_CAP, rng=np.random.default_rng([seed, 2])
        ),
    }

    # One render and one score per (arm, example); sets and baseline statistics index into it.
    needed: dict[str, ChainSpec] = {s.example_id: s for s in sample}
    for sets in set_sources.values():
        needed |= {m.example_id: m for members in sets for m in members}
    layouts, kept = {}, {}
    for eid, s in needed.items():
        r, _ = _render(s, tok, render_kw)
        if r is not None and TARGET in r.chain_slices:
            layouts[eid], kept[eid] = r, s
    # Members that failed to render drop out; a set must keep two lengths to stay valid.
    set_sources = {
        name: [ms for ms in ([m for m in members if m.example_id in kept] for members in sets)
               if len({len(m.by_role(Role.TCRB)) for m in ms}) >= 2]
        for name, sets in set_sources.items()
    }  # fmt: skip

    sample_ids = [s.example_id for s in sample if s.example_id in layouts]
    perm_rng = np.random.default_rng([seed, 3])
    perm_layouts = {}
    for eid in sample_ids:
        s2, over = permute_chain_order(kept[eid], perm_rng)
        r, _ = _render(s2, tok, {**render_kw, **over})
        if r is not None:
            perm_layouts[eid] = r
    perm_ids = [e for e in sample_ids if e in perm_layouts]
    perm_alleles = np.array([kept[e].mhc_allele for e in perm_ids])

    def score(rs) -> np.ndarray:
        with token_dropout(model, True):
            return score_many(
                model, tok, rs, [TARGET], method="windowed", desc="phase2",
                batch_size=int(config["scoring"].get("batch_size", 16)), **est,
            )  # fmt: skip

    ids = list(layouts)
    scores: dict[str, dict[str, Any]] = {}
    for arm, flags in ARMS.items():
        mod = ChainAwarePositions(1, chain_embedding=False, **flags)
        base = [mod.apply(layouts[e]) for e in ids]
        perm = [mod.apply(perm_layouts[e]) for e in perm_ids]
        # Offsets are capped at 50 and max_length at 512, so every layout stays in range.
        assert all(max_position(r) <= TRAINED_MAX_POSITION for r in base + perm)
        scores[arm] = {"base": dict(zip(ids, score(base), strict=True))}
        scores[arm]["perm"] = np.array(score(perm))
        print(f"scored {arm}", flush=True)

    sample_base = {arm: np.array([sc["base"][e] for e in sample_ids]) for arm, sc in scores.items()}
    perm_base = {arm: np.array([sc["base"][e] for e in perm_ids]) for arm, sc in scores.items()}

    out: dict[str, Any] = {
        "model": config["model"],
        "target": TARGET,
        "estimator": {"name": "windowed_span_pll", **est},
        "token_dropout": True,
        "uniform_prior": UNIFORM_PRIOR_PLL,
        "usable_margin": USABLE_MARGIN,
        "sample": {"n": len(sample_ids), "n_alleles": len({kept[e].mhc_allele for e in sample_ids}),
                   "per_allele_cap": cap, **info},
        "arms": {},
    }  # fmt: skip

    for arm, flags in ARMS.items():
        med = _med(sample_base[arm])
        margin = med - UNIFORM_PRIOR_PLL
        out["arms"][arm] = {
            **flags,
            "median_baseline_score": med,
            "margin_over_uniform_prior": margin,
            "usable": bool(margin >= USABLE_MARGIN),
        }

    out["operator_L"] = {}
    for name, sets in set_sources.items():
        alleles = np.array([m[0].mhc_allele for m in sets])
        rng_seed = [seed, 4]
        stats = {
            arm: set_statistics(sets, scores[arm]["base"], np.random.default_rng(rng_seed))
            for arm in ARMS
        }
        sizes = np.array([len(m) for m in sets])
        ranges = np.array([np.ptp([len(x.by_role(Role.TCRB)) for x in m]) for m in sets])
        out["operator_L"][name] = {
            "matched": {
                "n_sets": len(sets),
                "n_members": int(sizes.sum()),
                "n_alleles": len(set(alleles)),
                "members_per_set": {
                    "min": int(sizes.min()),
                    "median": _med(sizes),
                    "max": int(sizes.max()),
                },
                "cdr3b_length_range_per_set": {
                    "min": int(ranges.min()),
                    "median": _med(ranges),
                    "max": int(ranges.max()),
                },
            },  # fmt: skip
            "arms": length_block(stats, alleles, boot, seed),
        }

    base_abs = {
        arm: np.abs(perm_base[arm] - scores[arm]["perm"]) for arm in ARMS
    }  # score change under permutation
    out["operator_O"] = {"n": len(perm_ids), "n_alleles": len(set(perm_alleles)), "arms": {}}
    for arm, d in base_abs.items():
        if _med(d) == 0.0:
            raise RuntimeError(f"{arm}: median |score change| under permutation is exactly 0")
        entry = {
            "median_abs_change": _ci(_med, d, clusters=perm_alleles, boot=boot, seed=seed),
            "n_exact_zero": int((d == 0).sum()),
        }
        if arm != "A0":
            entry["diff_vs_a0"] = _ci(
                lambda a, b: _med(a) - _med(b), d, base_abs["A0"],
                clusters=perm_alleles, boot=boot, seed=seed,
            )  # fmt: skip
            entry["ratio_vs_a0"] = _ci(
                lambda a, b: _med(a) / _med(b), d, base_abs["A0"],
                clusters=perm_alleles, boot=boot, seed=seed,
            )  # fmt: skip
        out["operator_O"]["arms"][arm] = entry
    return out


def write_tsv(result: dict[str, Any], path: Path) -> None:
    head = ["operator", "matched_source", "arm", "usable", "median_baseline_score", "margin",
            "sensitivity", "ci_lo", "ci_hi", "null", "excess", "excess_lo", "excess_hi",
            "ratio_vs_a0", "ratio_lo", "ratio_hi", "n_sets_or_pairs", "n_alleles"]  # fmt: skip
    f = lambda x: "" if x is None else f"{x:.4f}"  # noqa: E731
    rows = ["\t".join(head)]

    def row(op, src, arm, main, n, n_al, null=None, excess=None):
        a = result["arms"][arm]
        ratio = main.get("ratio_vs_a0") or {}
        ex = excess or {}
        pt = main["length_sensitivity" if op == "L" else "median_abs_change"]
        rows.append("\t".join([
            op, src, arm, str(a["usable"]), f(a["median_baseline_score"]),
            f(a["margin_over_uniform_prior"]), f(pt["point"]), f(pt["ci"][0]), f(pt["ci"][1]),
            f((null or {}).get("point")), f(ex.get("point")),
            f((ex.get("ci") or [None, None])[0]), f((ex.get("ci") or [None, None])[1]),
            f(ratio.get("point")), f((ratio.get("ci") or [None, None])[0]),
            f((ratio.get("ci") or [None, None])[1]), str(n), str(n_al),
        ]))  # fmt: skip

    for src, blk in result["operator_L"].items():
        m = blk["matched"]
        for arm, e in blk["arms"].items():
            row("L", src, arm, e, m["n_sets"], m["n_alleles"], e["shuffled_null"],
                e["excess_over_null"])  # fmt: skip
    o = result["operator_O"]
    for arm, e in o["arms"].items():
        row("O", "allele_balanced", arm, e, o["n"], o["n_alleles"])
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
    (args.out / "noncircular_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    write_tsv(result, args.out / "noncircular_summary.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
