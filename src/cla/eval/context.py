"""Context gain: how much a span's pseudo-log-likelihood improves when the other chains are present.

    python -m cla.eval.context --models 35M 150M 650M --device mps

For each example the target span is scored under the audit's estimator in three renderings:

* `complex`: the full rendered complex, exactly as the audit renders it.
* `alone`: only the target chain, rendered `<cls> chain <eos>`.
* `scrambled`: the full complex with every other chain replaced by a residue shuffle of
  itself (same lengths and composition, no sequence information).

`context_gain = complex - alone` is positive only if the model extracts something from the
other chains or from their mere presence. `complex - scrambled` isolates the content: if it is
zero the model responds to context being there, not to what it says.

Token dropout rescales embeddings by the observed mask fraction, which differs between a
~320 token complex and a ~100 token chain. The dropout-off arm removes that difference, so it
is the cleaner attribution; both arms are always reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cla.data.splits import split_specs
from cla.eval.audit import load_config, load_specs_jsonl, stratified_subsample, token_dropout
from cla.eval.metrics import bootstrap_ci
from cla.layout.concat import render_naive
from cla.layout.spec import Chain, ChainSpec, Role
from cla.model.backbone import load_esm, resolve_device
from cla.model.scoring import score_many
from cla.utils.seeding import set_all_seeds

# Alone rendering is `<cls> chain <eos>` with the chain adjacent to both specials. The padded
# variant keeps the audit's separator tokens, to show the result does not hinge on that.
ALONE_KWARGS = {"separator_tokens": 0, "symmetric_ends": False}


def _only(spec: ChainSpec, role: str) -> ChainSpec:
    return ChainSpec(
        chains=tuple(c for c in spec.chains if c.role.value == role), example_id=spec.example_id
    )


def _scrambled(spec: ChainSpec, role: str, rng: np.random.Generator) -> ChainSpec:
    """Shuffle the residues of every chain except `role`, keeping lengths and composition."""
    chains = tuple(
        c if c.role.value == role else Chain(c.role, "".join(rng.permutation(list(c.sequence))))
        for c in spec.chains
    )
    return ChainSpec(chains=chains, example_id=spec.example_id)


def _summary(values: np.ndarray, boot: dict, seed: int) -> dict[str, Any]:
    median, lo, hi = bootstrap_ci(
        np.median, values, n_boot=boot["n_boot"], alpha=boot["alpha"], seed=seed
    )
    return {
        "median": median,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "fraction_positive": float(np.mean(values > 0)),
    }


def context_gain(
    model,
    tokenizer,
    specs: Sequence[ChainSpec],
    target_role: str,
    *,
    method: str = "span",
    window: int = 15,
    stride: int = 15,
    max_windows: int | None = None,
    batch_size: int = 16,
    render_kwargs: dict[str, Any] | None = None,
    dropout_arms: Sequence[bool] = (True, False),
    scramble: bool = True,
    boot: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Score `target_role` alone, in its complex, and in a scrambled complex, per dropout arm.

    Specs lacking the role, or whose complex does not render under `render_kwargs`, are
    dropped, so every condition is scored on the same examples. `method` and the window
    parameters are those of `cla.model.scoring.score_many`.
    """
    render_kwargs = {"separator_tokens": 1, "symmetric_ends": True, "max_length": 512} | (
        render_kwargs or {}
    )
    boot = boot or {"n_boot": 10000, "alpha": 0.05}
    conditions: dict[str, list] = {"complex": [], "alone": [], "alone_padded": []}
    if scramble:
        conditions["scrambled"] = []
    kept: list[str] = []
    for i, spec in enumerate(specs):
        if spec.by_role(Role(target_role)) is None:
            continue
        try:
            full = render_naive(spec, tokenizer, **render_kwargs)
        except ValueError:
            continue
        own = _only(spec, target_role)
        conditions["complex"].append(full)
        conditions["alone"].append(
            render_naive(own, tokenizer, max_length=render_kwargs["max_length"], **ALONE_KWARGS)
        )
        conditions["alone_padded"].append(render_naive(own, tokenizer, **render_kwargs))
        if scramble:
            shuffled = _scrambled(spec, target_role, np.random.default_rng([seed, 3, i]))
            conditions["scrambled"].append(render_naive(shuffled, tokenizer, **render_kwargs))
        kept.append(spec.example_id)
    if not kept:
        raise ValueError(f"no example has a renderable {target_role} chain")

    score_kw = {
        "method": method,
        "batch_size": batch_size,
        "window": window,
        "stride": stride,
        "max_windows": max_windows,
    }
    arms: dict[str, Any] = {}
    for enabled in dropout_arms:
        with token_dropout(model, enabled):
            pll = {
                name: score_many(
                    model, tokenizer, r, [target_role], desc=f"{target_role} {name}", **score_kw
                )
                for name, r in conditions.items()
            }
        entry: dict[str, Any] = {
            "median_pll": {k: float(np.median(v)) for k, v in pll.items()},
            "context_gain": _summary(pll["complex"] - pll["alone"], boot, seed),
            "context_gain_vs_padded_alone": _summary(
                pll["complex"] - pll["alone_padded"], boot, seed
            ),
        }
        if scramble:
            entry["scrambled_minus_alone"] = _summary(pll["scrambled"] - pll["alone"], boot, seed)
            entry["complex_minus_scrambled"] = _summary(
                pll["complex"] - pll["scrambled"], boot, seed
            )
        arms["true" if enabled else "false"] = entry
    return {"target": target_role, "n": len(kept), "arms": arms}


# Estimators matching the audit configs: windowed for the long MHC span, whole-span otherwise.
TARGETS: dict[str, dict[str, Any]] = {
    "MHC_HEAVY": {"method": "windowed", "window": 15, "stride": 15, "max_windows": 4},
    "PEPTIDE": {"method": "span"},
    "TCRB": {"method": "span"},
}


def run(
    models: Sequence[str],
    specs: list[ChainSpec],
    *,
    n: int,
    seed: int,
    device: str | None,
    config: dict,
) -> dict[str, Any]:
    dev = resolve_device(device)
    set_all_seeds(seed)
    # Same draw as the audit: equal quotas per split shard, rng seeded [seed, 1].
    sample = stratified_subsample(split_specs(specs), n, np.random.default_rng([seed, 1]))
    chosen = [s for s, _ in sample]
    render_kwargs = {
        "separator_tokens": int(config.get("separator_tokens", 1)),
        "symmetric_ends": bool(config.get("symmetric_ends", True)),
        "max_length": int(config.get("max_length", 512)),
    }
    boot = {
        "n_boot": int(config["bootstrap"]["n_boot"]),
        "alpha": float(config["bootstrap"]["alpha"]),
    }
    out: dict[str, Any] = {
        "n_requested": n,
        "seed": seed,
        "render_kwargs": render_kwargs,
        "alone_render": ALONE_KWARGS,
        "bootstrap": boot,
        "targets": TARGETS,
        "models": {},
    }
    for name in models:
        model, tok = load_esm(name, dev)
        out["models"][name] = {
            role: context_gain(
                model, tok, chosen, role, render_kwargs=render_kwargs, boot=boot, seed=seed, **kw
            )
            for role, kw in TARGETS.items()
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["35M", "150M", "650M"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "mps"], default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/audit/esm2_150m.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/phase1/context_gain.json"))
    args = ap.parse_args(argv)

    config = load_config(args.config)
    specs_path = Path(config.get("data", {}).get("specs", "data/processed/vdjdb_specs.jsonl"))
    if not specs_path.exists():
        print(f"error: {specs_path} not found. Run `make data` first.", file=sys.stderr)
        return 2
    result = run(
        args.models,
        load_specs_jsonl(specs_path),
        n=args.n,
        seed=args.seed,
        device=args.device,
        config=config,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
