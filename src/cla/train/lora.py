"""LoRA fine-tuning of ESM-2 with the chain-aware position module, one process per arm.

    python -m cla.train.lora --arm A0 --device cpu
    python -m cla.train.lora --summarize

Arms follow PROJECT_PLAN.md section 7.2. Every arm trains the same LoRA adapters on the same
batches with the same masks and schedule. The only differences are the module flags and, for
A5, a chain order drawn once per batch.

Parameter matching. The chain-identity embedding adds `(len(Role) + 1) * hidden` trainable
parameters. Arms without it carry an unused embedding of the same shape in the optimizer, so
every arm has an identical trainable count. The padding never enters the forward pass.

The training set is the `train` shard of `cla.data.splits.split_specs`, subsampled. Held-out
alleles and epitopes only appear in `cla.train.evaluate`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import EsmForMaskedLM, EsmTokenizer, get_cosine_schedule_with_warmup

from cla.data.splits import DEFAULT_SPLITS_CONFIG, split_specs
from cla.eval.audit import load_specs_jsonl
from cla.layout.spec import ChainSpec
from cla.model.backbone import resolve_checkpoint
from cla.model.positional import ROLES, ChainAwarePositions
from cla.train.batching import collate
from cla.utils.logging import JsonlLogger, get_logger, json_default
from cla.utils.manifest import build_manifest, write_manifest
from cla.utils.seeding import set_all_seeds

log = get_logger("cla.train")

SPECS_PATH = Path("data/processed/vdjdb_specs.jsonl")
OUT_ROOT = Path("results/phase2/training")
# Offsets above 50 push five-chain layouts past ESM-2's 1026 positions.
BREAK_OFFSET = 50
ARMS: dict[str, dict[str, Any]] = {
    "A0": {"reset": False, "chain_embedding": False, "offset": None},
    "A1": {"reset": False, "chain_embedding": True, "offset": None},
    "A2": {"reset": True, "chain_embedding": False, "offset": None},
    "A3": {"reset": False, "chain_embedding": False, "offset": BREAK_OFFSET},
    "A4": {"reset": True, "chain_embedding": True, "offset": BREAK_OFFSET},
    "A5": {"reset": True, "chain_embedding": True, "offset": BREAK_OFFSET, "augment": True},
}
# Attention projections and both FFN projections of every layer; the LM head's dense layer
# is excluded.
LORA_TARGETS = (
    r"esm\.encoder\.layer\.\d+\."
    r"(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)"
)
HPARAMS = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lr": 5e-4,
    "weight_decay": 0.0,
    "warmup_frac": 0.05,
    "grad_clip": 1.0,
    "batch_size": 16,
    "steps": 300,
    "train_subsample": 4800,
    "seed": 0,
}


class RoleBound(nn.Module):
    """Model wrapper that binds per-token roles around each forward.

    `takes_chain_ids` tells `cla.model.scoring` to pass `chain_ids`, so the scoring path
    applies the chain-identity embedding too.
    """

    takes_chain_ids = True

    def __init__(self, model: nn.Module, module: ChainAwarePositions, padding: nn.Module | None):
        super().__init__()
        self.model, self.module, self.padding = model, module, padding

    def forward(self, input_ids, attention_mask, position_ids=None, chain_ids=None):
        if chain_ids is None:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
        with self.module.bound(chain_ids):
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
            )


def build_arm(model: EsmForMaskedLM, arm: str, hparams: dict) -> RoleBound:
    """Wrap `model` with LoRA and the arm's position module, padded for parameter matching."""
    flags = {k: v for k, v in ARMS[arm].items() if k != "augment"}
    hidden = model.config.hidden_size
    module = ChainAwarePositions(hidden, **flags)
    # The hook goes on the raw model before peft wraps it, so it survives the wrapping.
    module.attach(model)
    padding = None
    if module.role_embedding is None:
        padding = nn.Embedding(len(ROLES) + 1, hidden, padding_idx=0)
        nn.init.zeros_(padding.weight)
    cfg = LoraConfig(
        r=hparams["lora_r"],
        lora_alpha=hparams["lora_alpha"],
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
    )
    return RoleBound(get_peft_model(model, cfg), module, padding)


def trainable_counts(bound: RoleBound) -> dict[str, int]:
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)  # noqa: E731
    return {
        "total": count(bound),
        "lora": count(bound.model),
        "role_embedding": count(bound.module),
        "padding": count(bound.padding) if bound.padding is not None else 0,
    }


def _train_subsample(specs: list[ChainSpec], n: int, seed: int) -> list[ChainSpec]:
    train = split_specs(specs)["train"]
    order = np.random.default_rng(seed).permutation(len(train))[:n]
    return [train[i] for i in sorted(order)]


def _loss(logits: torch.Tensor, batch) -> torch.Tensor:
    mask = batch.loss_mask
    return nn.functional.cross_entropy(logits[mask].float(), batch.target_ids[mask])


def g2_verdict(train_losses: list[float], at0: dict, at10: dict, a0_at10: dict) -> dict[str, Any]:
    """Gate G2 at 10% of steps (PROJECT_PLAN.md 7.4), with every criterion fixed in advance.

    * loss: finite throughout and the mean training loss of the last third of the window is below
      the first third's.
    * no collapse: held-out-allele MLM loss is no higher than at step 0 and the held-out-allele
      MHC score still beats the uniform prior.
    * sensitivity: median |slope| of the MHC score against CDR3B length is below A0's at the same
      step. The point estimates decide; intervals are reported beside them.
    """
    third = max(1, len(train_losses) // 3)
    first, last = np.mean(train_losses[:third]), np.mean(train_losses[-third:])
    loss_ok = bool(np.isfinite(train_losses).all() and last < first)
    allele0, allele10 = at0["mlm"]["test_allele"]["loss"], at10["mlm"]["test_allele"]["loss"]
    margin = at10["mhc"]["margin_over_uniform_prior"]
    forgetting_ok = bool(allele10 <= allele0 and margin is not None and margin > 0)
    sens, base = at10["length_sensitivity"], a0_at10["length_sensitivity"]
    sens_ok = bool(sens["median_abs_slope"] < base["median_abs_slope"])
    return {
        "loss_decreasing_and_finite": loss_ok,
        "train_loss_first_third": float(first),
        "train_loss_last_third": float(last),
        "no_catastrophic_forgetting": forgetting_ok,
        "heldout_allele_mlm_loss_step0": allele0,
        "heldout_allele_mlm_loss_gate": allele10,
        "heldout_allele_mhc_margin_gate": margin,
        "sensitivity_lower_than_a0": sens_ok,
        "length_sensitivity_gate": sens["median_abs_slope"],
        "a0_length_sensitivity_same_step": base["median_abs_slope"],
        "passed": loss_ok and forgetting_ok and sens_ok,
    }


def train_arm(arm: str, *, device: str, hparams: dict, out_root: Path = OUT_ROOT) -> dict:
    from cla.train.evaluate import build_eval_data, evaluate, save_eval_arrays

    run_dir = out_root / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    hp = dict(hparams)
    steps = hp["steps"]
    gate_step = max(1, round(0.1 * steps))
    set_all_seeds(hp["seed"])
    torch.set_num_threads(hp.get("threads", 8))

    ckpt = resolve_checkpoint(hp["model"])
    tok = EsmTokenizer.from_pretrained(ckpt)
    bound = build_arm(EsmForMaskedLM.from_pretrained(ckpt, torch_dtype=torch.float32), arm, hp)
    bound.to(device)
    counts = trainable_counts(bound)
    specs = load_specs_jsonl(SPECS_PATH)
    train = _train_subsample(specs, hp["train_subsample"], hp["seed"])
    eval_data = build_eval_data(specs, hp["seed"], hp["eval_n_mlm"])

    config = {"arm": arm, "module": ARMS[arm], **hp, "device": device, "trainable": counts}
    write_manifest(
        build_manifest(config, data_files=[SPECS_PATH, DEFAULT_SPLITS_CONFIG], device=device),
        run_dir,
    )
    events = JsonlLogger(run_dir)
    events.log("start", **config, n_train=len(train), padding_note=_padding_note(counts))

    params = [p for p in bound.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(hp["warmup_frac"] * steps), steps)
    # Batch order and masks come from generators seeded identically in every arm, and A5's chain
    # order has its own generator, so arms differ only in the module and the permutation.
    order_rng = np.random.default_rng([hp["seed"], 10])
    mask_rng = np.random.default_rng([hp["seed"], 11])
    perm_rng = np.random.default_rng([hp["seed"], 12])
    bs = hp["batch_size"]

    evals: dict[str, dict] = {}
    t_eval = 0.0

    def run_eval(tag: str, step: int) -> dict:
        nonlocal t_eval
        t0 = time.time()
        bound.eval()
        with torch.no_grad():
            result, arrays = evaluate(bound, tok, eval_data, step=step, seed=hp["seed"])
        bound.train()
        t_eval += time.time() - t0
        evals[tag] = result
        events.log("eval", tag=tag, **result)
        if tag == "final":
            save_eval_arrays(run_dir / "eval_arrays.npz", arrays)
        return result

    run_eval("step0", 0)
    bound.train()
    losses: list[float] = []
    t_train = 0.0
    perm = order_rng.permutation(len(train))
    cursor = 0
    status, gate = "completed", None
    for step in range(1, steps + 1):
        t0 = time.time()
        if cursor + bs > len(perm):
            perm, cursor = order_rng.permutation(len(train)), 0
        chunk = [train[i] for i in perm[cursor : cursor + bs]]
        cursor += bs
        order = tuple(int(i) for i in perm_rng.permutation(5)) if ARMS[arm].get("augment") else None
        batch = collate(chunk, tok, bound.module, mask_rng, chain_order=order)
        batch = batch.to(device)
        out = bound(batch.input_ids, batch.attention_mask, batch.position_ids, batch.chain_ids)
        loss = _loss(out.logits, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, hp["grad_clip"])
        opt.step()
        sched.step()
        losses.append(float(loss))
        t_train += time.time() - t0
        if step % 10 == 0 or step == 1:
            events.log("train", step=step, loss=losses[-1], lr=sched.get_last_lr()[0])
        if not math.isfinite(losses[-1]):
            status = "diverged"
            events.log("diverged", step=step)
            break
        if step == gate_step:
            at10 = run_eval("gate", step)
            if arm == "A4":
                a0 = json.loads((out_root / "A0" / "metrics.json").read_text())["evals"]["gate"]
                gate = g2_verdict(losses, evals["step0"], at10, a0)
                events.log("g2", **gate)
                if not gate["passed"]:
                    status = "killed_g2"
                    break
    if status == "completed":
        run_eval("final", steps)

    final = evals.get("final")
    metrics = {
        "arm": arm,
        "status": status,
        "steps_run": len(losses),
        "steps_planned": steps,
        "hparams": hp,
        "module": ARMS[arm],
        "trainable_params": counts,
        "padding_note": _padding_note(counts),
        "train_wall_clock_s": t_train,
        "eval_wall_clock_s": t_eval,
        "train_loss": {
            "first10": float(np.mean(losses[:10])),
            "last10": float(np.mean(losses[-10:])),
        },
        "g2": gate,
        "evals": evals,
        "final": final,
        "final_null_reason": None if final else f"no final eval: run ended with status {status}",
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=json_default) + "\n"
    )
    events.close()
    return metrics


def _padding_note(counts: dict[str, int]) -> str:
    if counts["padding"]:
        return (
            f"{counts['padding']} unused embedding parameters added so the trainable count "
            "matches arms with the chain-identity embedding"
        )
    return "arm has the chain-identity embedding; no padding needed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="35M")
    ap.add_argument("--steps", type=int, default=HPARAMS["steps"])
    ap.add_argument("--lr", type=float, default=HPARAMS["lr"])
    ap.add_argument("--train-subsample", type=int, default=HPARAMS["train_subsample"])
    ap.add_argument("--eval-n-mlm", type=int, default=400)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args(argv)
    if args.summarize:
        from cla.train.evaluate import write_summary

        write_summary(args.out)
        return 0
    if args.arm is None:
        ap.error("--arm or --summarize is required")
    hp = {
        **HPARAMS,
        "steps": args.steps,
        "lr": args.lr,
        "train_subsample": args.train_subsample,
        "model": args.model,
        "eval_n_mlm": args.eval_n_mlm,
        "threads": args.threads,
    }
    metrics = train_arm(args.arm, device=args.device, hparams=hp, out_root=args.out)
    log.info("%s %s in %.0fs", args.arm, metrics["status"], metrics["train_wall_clock_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
