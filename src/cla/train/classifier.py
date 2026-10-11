"""LoRA binding classifier: held-out binder-vs-decoy AUROC for each chain-aware arm.

    python -m cla.train.classifier --arm A0 --seed 0 --lr 5e-4 --device cpu
    python -m cla.train.classifier --summarize

DECOYS ARE NOT VALIDATED NON-BINDERS. Every negative is a real receptor paired with the peptide,
MHC and B2M of a different epitope (`cla.eval.negatives`). Some are false negatives from
cross-reactivity, and unrelated antigens are easier to separate than the borderline non-binders
seen in practice. An AUROC here is a weak proxy for real binding and is not an upper or lower
bound on it.

Setup. The ESM-2 backbone, LoRA adapters and arm modules come from `cla.train.lora.build_arm`, so
parameter matching and the padding of arms without a chain embedding are shared with the MLM
runs. A linear head reads the mean over residue tokens of the last hidden state. Mean pooling is
used because ESM-2's `<cls>` is never trained as a sequence summary, and restricting the mean to
chain residues keeps separators and padding out of the representation.

Training uses the `train` shard only, with one mismatched decoy per sampled positive drawn from
the same sampled shard. Each held-out shard builds its own decoys from its own members, so a
decoy peptide and MHC are held out as well. The evaluation sets are identical across seeds and
arms; seeds vary the training sample, initialization and batch order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from transformers import EsmForMaskedLM, EsmTokenizer, get_cosine_schedule_with_warmup

from cla.data.splits import DEFAULT_SPLITS_CONFIG, split_specs
from cla.eval.audit import load_specs_jsonl
from cla.eval.metrics import cluster_bootstrap_ci
from cla.eval.negatives import make_mismatched_negatives
from cla.layout.concat import render_naive
from cla.layout.spec import ChainSpec
from cla.model.backbone import resolve_checkpoint
from cla.model.positional import ChainAwarePositions
from cla.train.batching import RENDER_KWARGS
from cla.train.lora import ARMS, HPARAMS, SPECS_PATH, RoleBound, build_arm, trainable_counts
from cla.utils.logging import JsonlLogger, get_logger, json_default
from cla.utils.manifest import build_manifest, write_manifest
from cla.utils.seeding import set_all_seeds

log = get_logger("cla.train.classifier")

OUT_ROOT = Path("results/phase2/classifier")
EVAL_SHARDS = ("test_allele", "test_epitope")
EVAL_POSITIVES = {"test_allele": 1000, "test_epitope": 600}
TRAIN_POSITIVES = 2400
EVAL_SEED = 0
N_BOOT = 2000
EVAL_BATCH = 16
DECOY_NOTICE = (
    "Negatives are mismatched decoys, not validated non-binders: AUROC is a weak proxy for real "
    "binding, not a bound on it."
)
CHANCE = 0.5


class BindingClassifier(nn.Module):
    """`RoleBound` backbone plus a linear head on the mean-pooled residue representation."""

    def __init__(self, bound: RoleBound, hidden: int):
        super().__init__()
        self.bound = bound
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask, position_ids, chain_ids) -> torch.Tensor:
        esm = self.bound.model.base_model.model.esm
        with self.bound.module.bound(chain_ids):
            hidden = esm(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
            ).last_hidden_state
        residue = (chain_ids > 0).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * residue).sum(1) / residue.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def labeled_set(
    positives: list[ChainSpec],
    rng: np.random.Generator,
    known_positives: list[ChainSpec],
) -> list[ChainSpec]:
    """Positives plus one decoy each, with labels fixed to 1 and 0 here rather than trusted."""
    decoys = make_mismatched_negatives(
        positives, rng, known_positives=known_positives, n_per_positive=1
    )
    return [replace(s, label=1) for s in positives] + [replace(s, label=0) for s in decoys]


def build_data(specs: list[ChainSpec], seed: int) -> tuple[list[ChainSpec], dict]:
    shards = split_specs(specs)

    def sample(pool: list[ChainSpec], n: int, rng: np.random.Generator) -> list[ChainSpec]:
        pick = np.sort(rng.permutation(len(pool))[:n])
        return [pool[i] for i in pick]

    train = labeled_set(
        sample(shards["train"], TRAIN_POSITIVES, np.random.default_rng([seed, 30])),
        np.random.default_rng([seed, 31]),
        specs,
    )
    evals = {
        name: labeled_set(
            sample(shards[name], EVAL_POSITIVES[name], np.random.default_rng([EVAL_SEED, 40])),
            np.random.default_rng([EVAL_SEED, 41]),
            specs,
        )
        for name in EVAL_SHARDS
    }
    return train, evals


def collate_unmasked(
    specs: list[ChainSpec],
    tok,
    module: ChainAwarePositions,
    *,
    chain_order: tuple[int, ...] | None = None,
):
    rendered = []
    for spec in specs:
        if chain_order is not None:
            spec = spec.reordered(chain_order)
        rendered.append(module.apply(render_naive(spec, tok, **RENDER_KWARGS)))
    n, width = len(rendered), max(r.input_ids.shape[0] for r in rendered)
    ids = torch.full((n, width), tok.pad_token_id, dtype=torch.long)
    attn, pos, roles = (torch.zeros((n, width), dtype=torch.long) for _ in range(3))
    for i, r in enumerate(rendered):
        length = r.input_ids.shape[0]
        ids[i, :length], attn[i, :length], roles[i, :length] = (
            r.input_ids,
            r.attention_mask,
            r.chain_ids,
        )
        pos[i, :length] = r.position_ids if r.position_ids is not None else torch.arange(length)
    labels = torch.tensor([float(s.label) for s in specs])
    return ids, attn, pos, roles, labels


@torch.no_grad()
def predict(model: BindingClassifier, tok, specs: list[ChainSpec], device: str) -> np.ndarray:
    model.eval()
    # Length-sorted batches cut padding; scores are written back in the original order.
    order = np.argsort([s.total_residues for s in specs], kind="stable")
    out = np.zeros(len(specs))
    for start in range(0, len(specs), EVAL_BATCH):
        idx = order[start : start + EVAL_BATCH]
        ids, attn, pos, roles, _ = collate_unmasked(
            [specs[i] for i in idx], tok, model.bound.module
        )
        out[idx] = model(ids.to(device), attn.to(device), pos.to(device), roles.to(device)).cpu()
    model.train()
    return out


def eval_metrics(specs: list[ChainSpec], scores: np.ndarray, seed: int) -> dict:
    y = np.array([s.label for s in specs])
    alleles = np.array([s.mhc_allele for s in specs])
    res = {"n": len(y), "n_alleles": len(set(alleles))}
    for name, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
        pt, lo, hi = cluster_bootstrap_ci(fn, y, scores, clusters=alleles, n_boot=N_BOOT, seed=seed)
        res[name] = {"point": pt, "ci": [lo, hi]}
    return res


def train_run(arm: str, *, device: str, hp: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, seed = hp["steps"], hp["seed"]
    set_all_seeds(seed)
    torch.set_num_threads(hp["threads"])
    ckpt = resolve_checkpoint(hp["model"])
    tok = EsmTokenizer.from_pretrained(ckpt)
    mlm = EsmForMaskedLM.from_pretrained(ckpt, torch_dtype=torch.float32)
    model = BindingClassifier(build_arm(mlm, arm, hp), mlm.config.hidden_size).to(device)
    counts = trainable_counts(model.bound)
    counts["head"] = sum(p.numel() for p in model.head.parameters())
    counts["total"] += counts["head"]
    specs = load_specs_jsonl(SPECS_PATH)
    train, evals = build_data(specs, seed)

    config = {"arm": arm, "module": ARMS[arm], **hp, "device": device, "trainable": counts}
    write_manifest(
        build_manifest(config, data_files=[SPECS_PATH, DEFAULT_SPLITS_CONFIG], device=device),
        out_dir,
    )
    events = JsonlLogger(out_dir)
    events.log("start", **config, n_train=len(train), notice=DECOY_NOTICE)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(hp["warmup_frac"] * steps), steps)
    order_rng = np.random.default_rng([seed, 10])
    perm_rng = np.random.default_rng([seed, 12])
    bs = hp["batch_size"]
    perm, cursor = order_rng.permutation(len(train)), 0
    losses: list[float] = []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        if cursor + bs > len(perm):
            perm, cursor = order_rng.permutation(len(train)), 0
        chunk = [train[i] for i in perm[cursor : cursor + bs]]
        cursor += bs
        order = tuple(int(i) for i in perm_rng.permutation(5)) if ARMS[arm].get("augment") else None
        ids, attn, pos, roles, labels = collate_unmasked(
            chunk, tok, model.bound.module, chain_order=order
        )
        logits = model(ids.to(device), attn.to(device), pos.to(device), roles.to(device))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, hp["grad_clip"])
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == 1:
            events.log("train", step=step, loss=losses[-1], lr=sched.get_last_lr()[0])
        if not math.isfinite(losses[-1]):
            raise RuntimeError(f"{arm} seed {seed}: non-finite loss at step {step}")
    train_s = time.time() - t0

    t0 = time.time()
    arrays, results = {}, {}
    for name, ev in evals.items():
        scores = predict(model, tok, ev, device)
        results[name] = eval_metrics(ev, scores, seed=0)
        arrays |= {
            f"{name}_score": scores,
            f"{name}_label": np.array([s.label for s in ev]),
            f"{name}_allele": np.array([s.mhc_allele for s in ev]),
        }
    np.savez(out_dir / "eval_arrays.npz", **arrays)
    metrics = {
        "arm": arm,
        "seed": seed,
        "notice": DECOY_NOTICE,
        "hparams": hp,
        "module": ARMS[arm],
        "trainable_params": counts,
        "steps": len(losses),
        "train_wall_clock_s": train_s,
        "eval_wall_clock_s": time.time() - t0,
        "train_loss": {
            "first20": float(np.mean(losses[:20])),
            "last20": float(np.mean(losses[-20:])),
        },
        "eval": results,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=json_default) + "\n"
    )
    events.close()
    return metrics


def _load_runs(root: Path, pattern: str) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for p in sorted(root.glob(pattern)):
        m = json.loads(p.read_text())
        m["_dir"] = p.parent
        runs.setdefault(m["arm"], []).append(m)
    return runs


def _seed_mean_ci(arrays: list[dict], shard: str, fn, other: list[dict] | None = None) -> tuple:
    """Cluster-bootstrap CI of the seed-mean metric (or of the A-minus-other difference).

    Every seed scores the same examples, so one cluster resample is applied to all seeds at once.
    """
    y = arrays[0][f"{shard}_label"]
    stack = lambda arrs: np.stack([a[f"{shard}_score"] for a in arrs], axis=1)  # noqa: E731
    mean_metric = lambda yy, s: float(np.mean([fn(yy, s[:, k]) for k in range(s.shape[1])]))  # noqa: E731
    if other is None:
        return cluster_bootstrap_ci(
            mean_metric, y, stack(arrays), clusters=arrays[0][f"{shard}_allele"], n_boot=N_BOOT
        )
    diff = lambda yy, a, b: mean_metric(yy, a) - mean_metric(yy, b)  # noqa: E731
    return cluster_bootstrap_ci(
        diff, y, stack(arrays), stack(other), clusters=arrays[0][f"{shard}_allele"], n_boot=N_BOOT
    )


def summarize(root: Path = OUT_ROOT) -> Path:
    """Per-arm JSON, the A4-vs-A0 delta, and `summary.tsv` from the finished main runs."""
    runs = _load_runs(root, "runs/*/metrics.json")
    totals = {m["trainable_params"]["total"] for ms in runs.values() for m in ms}
    assert len(totals) == 1, f"trainable counts differ across arms: {sorted(totals)}"

    arrays = {
        a: [dict(np.load(m["_dir"] / "eval_arrays.npz")) for m in ms] for a, ms in runs.items()
    }
    rows, per_arm = [], {}
    for arm, ms in runs.items():
        entry = {"arm": arm, "seeds": [m["seed"] for m in ms], "notice": DECOY_NOTICE}
        row = {
            "arm": arm,
            "n_seeds": len(ms),
            "trainable_params": ms[0]["trainable_params"]["total"],
            "steps": ms[0]["steps"],
            "train_wall_clock_s_mean": float(np.mean([m["train_wall_clock_s"] for m in ms])),
        }
        for shard in EVAL_SHARDS:
            for metric, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
                pts = np.array([m["eval"][shard][metric]["point"] for m in ms])
                mean, lo, hi = _seed_mean_ci(arrays[arm], shard, fn)
                std = float(pts.std(ddof=1)) if len(pts) > 1 else None
                tag = f"{shard}_{metric}"
                entry[tag] = {"per_seed": pts.tolist(), "mean": mean, "std": std, "ci": [lo, hi]}
                row |= {f"{tag}_mean": mean, f"{tag}_std": std, f"{tag}_lo": lo, f"{tag}_hi": hi}
                if metric == "auroc":
                    row[f"{tag}_excludes_chance"] = bool(lo > CHANCE or hi < CHANCE)
        per_arm[arm] = entry
        rows.append(row)

    if "A0" in runs:
        for arm in runs:
            if arm == "A0":
                continue
            delta = {}
            for shard in EVAL_SHARDS:
                for metric, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
                    pt, lo, hi = _seed_mean_ci(arrays[arm], shard, fn, other=arrays["A0"])
                    delta[f"{shard}_{metric}"] = {"delta": pt, "ci": [lo, hi]}
                    for r in rows:
                        if r["arm"] == arm:
                            r |= {
                                f"{shard}_{metric}_delta_vs_a0": pt,
                                f"{shard}_{metric}_delta_lo": lo,
                                f"{shard}_{metric}_delta_hi": hi,
                            }
            per_arm[arm]["delta_vs_a0"] = delta
    for arm, entry in per_arm.items():
        (root / f"{arm}.json").write_text(json.dumps(entry, indent=2, default=json_default) + "\n")

    cols = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
    path = root / "summary.tsv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in cols])
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="35M")
    ap.add_argument("--steps", type=int, default=HPARAMS["steps"])
    ap.add_argument("--lr", type=float, default=HPARAMS["lr"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, help="run directory (default runs/<arm>_s<seed>)")
    ap.add_argument(
        "--root",
        type=Path,
        default=OUT_ROOT,
        help="results root, so a second model size can be swept without overwriting the first",
    )
    args = ap.parse_args(argv)
    if args.summarize:
        print(summarize(args.root))
        return 0
    if args.arm is None:
        ap.error("--arm or --summarize is required")
    hp = {
        **HPARAMS,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "model": args.model,
        "threads": args.threads,
    }
    out = args.out or args.root / "runs" / f"{args.arm}_s{args.seed}"
    m = train_run(args.arm, device=args.device, hp=hp, out_dir=out)
    log.info(
        "%s seed %d done in %.0fs: %s", args.arm, args.seed, m["train_wall_clock_s"], m["eval"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
