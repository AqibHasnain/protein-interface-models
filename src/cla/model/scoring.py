"""Span-level pseudo-log-likelihood under ESM-2.

Position handling in HuggingFace ESM-2 (transformers 5.x, `esm2_*` checkpoints have
`position_embedding_type="rotary"`):

* `EsmModel.forward` accepts `position_ids` and honors it. Positions enter only through
  `self.rotary_embeddings(inputs_embeds, position_ids)`, which builds cos/sin applied to
  queries and keys in every attention layer. `EsmEmbeddings` computes position ids too, but
  only adds learned absolute embeddings when `position_embedding_type == "absolute"`, which
  does not hold for ESM-2.
* Rotary attention depends only on position differences, so shifting all ids by a constant is
  a no-op while per-chain offsets (gaps between chains) change cross-chain attention.
* `chain_ids` has no entry point in the model; it is carried on the rendered layout for
  downstream interventions and ignored here.

Masked tokens are zeroed and embeddings rescaled when `config.token_dropout` is set (true for
ESM-2). Masking a whole span therefore changes the rescaling factor for every token in the
sequence, which is part of what `span_pll` measures.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class RenderedLike(Protocol):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    chain_slices: dict[str, slice]
    position_ids: torch.Tensor | None
    chain_ids: torch.Tensor | None


def _model_device(model: nn.Module) -> torch.device:
    return next(iter(model.parameters()), torch.empty(0)).device


def target_positions(rendered: RenderedLike, target_roles: Iterable[str]) -> torch.Tensor:
    """Sorted token indices covered by the slices of `target_roles`."""
    idx: list[int] = []
    for role in target_roles:
        key = getattr(role, "value", role)
        sl = rendered.chain_slices[key]
        idx.extend(range(*sl.indices(rendered.input_ids.shape[0])))
    if not idx:
        raise ValueError("target_roles selects no positions")
    return torch.tensor(sorted(set(idx)), dtype=torch.long)


def _masked_logprobs(
    model: nn.Module,
    tokenizer,
    jobs: Sequence[tuple[RenderedLike, torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> list[torch.Tensor]:
    """Run each job (rendered, positions to mask, positions to score) and return log-probs.

    Each job is one forward pass. Jobs are right-padded to the longest sequence in their batch
    and `attention_mask` excludes the padding. Returned tensors hold the true-token log-prob at
    the scored positions of each job.
    """
    device = _model_device(model)
    out: list[torch.Tensor] = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        n = len(batch)
        width = max(r.input_ids.shape[0] for r, _, _ in batch)
        x = torch.full((n, width), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((n, width), dtype=torch.long)
        use_pos = any(r.position_ids is not None for r, _, _ in batch)
        pos = torch.zeros((n, width), dtype=torch.long)
        for row, (r, mask_idx, _) in enumerate(batch):
            length = r.input_ids.shape[0]
            x[row, :length] = r.input_ids
            x[row, mask_idx] = tokenizer.mask_token_id
            attn[row, :length] = r.attention_mask
            pos[row, :length] = (
                r.position_ids if r.position_ids is not None else torch.arange(length)
            )
        kwargs = {"position_ids": pos.to(device)} if use_pos else {}
        with torch.inference_mode():
            logits = model(input_ids=x.to(device), attention_mask=attn.to(device), **kwargs).logits
        logp = torch.log_softmax(logits.float(), dim=-1).cpu()
        for row, (r, _, score_idx) in enumerate(batch):
            out.append(logp[row, score_idx, r.input_ids[score_idx]])
    return out


def span_pll(
    model: nn.Module,
    tokenizer,
    rendered: RenderedLike,
    target_roles: Iterable[str],
) -> float:
    """Mask every target span at once, forward once, and return the mean true-token log-prob."""
    pos = target_positions(rendered, target_roles)
    (lp,) = _masked_logprobs(model, tokenizer, [(rendered, pos, pos)], batch_size=1)
    return float(lp.mean())


def pll_pseudo(
    model: nn.Module,
    tokenizer,
    rendered: RenderedLike,
    target_roles: Iterable[str],
    *,
    chunk: int = 32,
) -> float:
    """Classic pseudo-log-likelihood: mask one target position per forward, average over positions.

    `chunk` bounds how many masked copies share a forward pass.
    """
    pos = target_positions(rendered, target_roles)
    jobs = [(rendered, p.reshape(1), p.reshape(1)) for p in pos]
    lps = _masked_logprobs(model, tokenizer, jobs, batch_size=chunk)
    return float(torch.cat(lps).mean())


def score_many(
    model: nn.Module,
    tokenizer,
    rendereds: Sequence[RenderedLike],
    target_roles: Iterable[str],
    *,
    method: str = "span",
    batch_size: int = 16,
    chunk: int = 32,
    desc: str = "scoring",
) -> np.ndarray:
    """Score each rendered layout; returns a float64 array aligned with `rendereds`.

    `batch_size` is the number of layouts per forward pass for the span method. The pseudo
    method already batches the L masked copies of one layout, `chunk` at a time.
    """
    roles = list(target_roles)
    if method == "pseudo":
        return np.array(
            [
                pll_pseudo(model, tokenizer, r, roles, chunk=chunk)
                for r in tqdm(rendereds, desc=desc)
            ],
            dtype=np.float64,
        )
    if method != "span":
        raise ValueError(f"unknown method {method!r}")
    scores: list[float] = []
    with tqdm(total=len(rendereds), desc=desc) as bar:
        for start in range(0, len(rendereds), batch_size):
            group = rendereds[start : start + batch_size]
            jobs = []
            for r in group:
                pos = target_positions(r, roles)
                jobs.append((r, pos, pos))
            scores.extend(
                float(lp.mean()) for lp in _masked_logprobs(model, tokenizer, jobs, len(jobs))
            )
            bar.update(len(group))
    return np.array(scores, dtype=np.float64)
