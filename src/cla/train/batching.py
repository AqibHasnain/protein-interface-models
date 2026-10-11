"""Rendering, joint span masking and batching for stage-2 style masked-LM training.

Masking scheme. In every complex one contiguous span is masked in each of CDR3A, CDR3B and
PEPTIDE at once, covering `MASK_FRACTION` of the chain (at least one residue) at a uniformly
drawn start. Masking the three chains together makes the model predict each from the other two
plus the MHC, which is the joint CDR+peptide objective. The MHC heavy chain and B2M are never
masked, so their scores serve as a drift check on what the backbone already knew. Masked
positions all become `<mask>` with no random or kept replacements, matching how the scorer
masks, and the loss is cross-entropy over masked positions only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from cla.layout.concat import RenderedLayout, render_naive
from cla.layout.spec import ChainSpec, Role
from cla.model.positional import ChainAwarePositions

MASK_ROLES = (Role.TCRA, Role.TCRB, Role.PEPTIDE)
MASK_FRACTION = 0.4
# Same baseline serialization the Phase 1 audit scores under.
RENDER_KWARGS = {"separator_tokens": 1, "symmetric_ends": True, "max_length": 512}


@dataclass
class Batch:
    input_ids: torch.Tensor  # masked
    target_ids: torch.Tensor
    loss_mask: torch.Tensor  # bool, True where a masked residue is scored
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    chain_ids: torch.Tensor

    def to(self, device) -> Batch:
        return Batch(*(t.to(device) for t in vars(self).values()))


def span_mask(rendered: RenderedLayout, rng: np.random.Generator) -> torch.Tensor:
    mask = torch.zeros(rendered.input_ids.shape[0], dtype=torch.bool)
    for role in MASK_ROLES:
        sl = rendered.chain_slices[role.value]
        length = sl.stop - sl.start
        span = max(1, math.ceil(MASK_FRACTION * length))
        start = sl.start + int(rng.integers(0, length - span + 1))
        mask[start : start + span] = True
    return mask


def collate(
    specs: list[ChainSpec],
    tokenizer,
    module: ChainAwarePositions,
    rng: np.random.Generator,
    *,
    chain_order: tuple[int, ...] | None = None,
) -> Batch:
    """Render, mask and right-pad `specs`. `chain_order` permutes every complex's chains."""
    rendered = []
    for spec in specs:
        if chain_order is not None:
            spec = spec.reordered(chain_order)
        rendered.append(module.apply(render_naive(spec, tokenizer, **RENDER_KWARGS)))
    n, width = len(rendered), max(r.input_ids.shape[0] for r in rendered)
    target = torch.full((n, width), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    pos = torch.zeros((n, width), dtype=torch.long)
    roles = torch.zeros((n, width), dtype=torch.long)
    loss_mask = torch.zeros((n, width), dtype=torch.bool)
    for i, r in enumerate(rendered):
        length = r.input_ids.shape[0]
        target[i, :length] = r.input_ids
        attn[i, :length] = r.attention_mask
        pos[i, :length] = r.position_ids if r.position_ids is not None else torch.arange(length)
        roles[i, :length] = r.chain_ids
        loss_mask[i, :length] = span_mask(r, rng)
    inputs = target.masked_fill(loss_mask, tokenizer.mask_token_id)
    return Batch(inputs, target, loss_mask, attn, pos, roles)
