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

Long spans are scored with `windowed_span_pll`, which masks a short window at a time so local
context stays visible.

Masked tokens are zeroed and embeddings rescaled when `config.token_dropout` is set (true for
ESM-2). Masking a whole span therefore changes the rescaling factor for every token in the
sequence, which is part of what `span_pll` measures.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def _quiet() -> bool:
    """Progress bars repaint into log files, so they only show on a terminal."""
    return not sys.stderr.isatty()


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


def _window_starts(n: int, window: int, stride: int, max_windows: int | None) -> list[int]:
    """Start offsets of full-size windows over `n` span positions.

    A final window is anchored to the span end when the stride would leave a tail uncovered.
    A span shorter than `window` yields one window covering the whole span. When
    `max_windows` truncates, windows are chosen evenly across the span so every region of
    it stays represented, and the choice depends only on the span length.
    """
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    last = max(n - window, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    if max_windows is not None and len(starts) > max_windows:
        keep = np.unique(np.round(np.linspace(0, len(starts) - 1, max_windows)).astype(int))
        starts = [starts[i] for i in keep]
    return starts


def _window_jobs(
    rendered: RenderedLike,
    target_roles: Iterable[str],
    window: int,
    stride: int,
    max_windows: int | None,
) -> list[tuple[RenderedLike, torch.Tensor, torch.Tensor]]:
    pos = target_positions(rendered, target_roles)
    return [
        (rendered, pos[s : s + window], pos[s : s + window])
        for s in _window_starts(len(pos), window, stride, max_windows)
    ]


def windowed_span_pll(
    model: nn.Module,
    tokenizer,
    rendered: RenderedLike,
    target_roles: Iterable[str],
    *,
    window: int = 15,
    stride: int = 15,
    max_windows: int | None = None,
    batch_size: int = 16,
) -> float:
    """Mask `window` contiguous target residues per forward, slide by `stride`, and average.

    Masking a whole long span removes all local context, so the model predicts the true
    residue worse than a uniform prior. A window keeps the rest of the span visible. Each
    window contributes the mean true-token log-prob of its masked residues, and the result is
    the mean over windows. All windows of one layout are batched, `batch_size` forwards at a time.
    """
    jobs = _window_jobs(rendered, target_roles, window, stride, max_windows)
    lps = _masked_logprobs(model, tokenizer, jobs, batch_size=batch_size)
    return float(np.mean([float(lp.mean()) for lp in lps]))


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
    window: int = 15,
    stride: int = 15,
    max_windows: int | None = None,
) -> np.ndarray:
    """Score each rendered layout; returns a float64 array aligned with `rendereds`.

    `batch_size` is the number of layouts per forward pass for the span method. The pseudo
    method already batches the L masked copies of one layout, `chunk` at a time. The windowed
    method flattens the windows of `batch_size` layouts into one job list, so the forward count
    is layouts x windows but each forward is batched.
    """
    roles = list(target_roles)
    if method == "pseudo":
        return np.array(
            [
                pll_pseudo(model, tokenizer, r, roles, chunk=chunk)
                for r in tqdm(rendereds, desc=desc, disable=_quiet())
            ],
            dtype=np.float64,
        )
    if method == "windowed":
        out: list[float] = []
        with tqdm(total=len(rendereds), desc=desc, disable=_quiet()) as bar:
            for start in range(0, len(rendereds), batch_size):
                group = rendereds[start : start + batch_size]
                jobs = [_window_jobs(r, roles, window, stride, max_windows) for r in group]
                flat = [j for per in jobs for j in per]
                lps = _masked_logprobs(model, tokenizer, flat, batch_size=batch_size)
                at = 0
                for per in jobs:
                    out.append(float(np.mean([float(lp.mean()) for lp in lps[at : at + len(per)]])))
                    at += len(per)
                bar.update(len(group))
        return np.array(out, dtype=np.float64)
    if method != "span":
        raise ValueError(f"unknown method {method!r}")
    scores: list[float] = []
    with tqdm(total=len(rendereds), desc=desc, disable=_quiet()) as bar:
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
