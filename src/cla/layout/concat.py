"""Baseline naive concatenation: the serialization under audit.

Layout is `<cls> chain_1 SEP^k chain_2 SEP^k ... chain_n <eos>`. The default k is 0:
chain boundaries are signalled through `position_ids`, not through inserted tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cla.layout.spec import ChainSpec, Role

# ESM-2 positions 0..1025 exist, with <cls> and <eos> consuming two.
DEFAULT_MAX_LENGTH = 1024


@dataclass(frozen=True)
class RenderedLayout:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    # Role value -> slice of input_ids covering exactly that chain's residues.
    chain_slices: dict[str, slice]
    # None means plain arange, which is what the model computes when given no position_ids.
    position_ids: torch.Tensor | None = None
    chain_ids: torch.Tensor | None = None


def normalize_position_ids(position_ids: torch.Tensor) -> torch.Tensor | None:
    """Return None when `position_ids` is plain arange, so callers can skip passing it."""
    if torch.equal(position_ids, torch.arange(position_ids.shape[0])):
        return None
    return position_ids


def render_naive(
    spec: ChainSpec,
    tokenizer,
    *,
    separator_tokens: int = 0,
    spacer_before_peptide: int = 0,
    chain_break_gap: int = 0,
    symmetric_ends: bool = False,
    max_length: int = DEFAULT_MAX_LENGTH,
    strict: bool = True,
) -> RenderedLayout:
    """Concatenate chains in `spec.chains` order into one sequence.

    Separator choice: `tokenizer.pad_token` repeated `separator_tokens` times between
    chains, with attention_mask=1 over them. The pad embedding is far from the residue
    embeddings (norm 5.83), so every separator is a real content token that all layers
    attend to. Varying the count therefore changes content as well as geometry; use
    `chain_break_gap` for a position-only boundary signal.

    `chain_break_gap` advances the position index by that many extra steps at the start
    of every chain after the first. ESM-2 positions enter only through rotary
    embeddings, so this changes cross-chain relative offsets without touching
    `input_ids`. With 0 the layout carries `position_ids=None`.

    `symmetric_ends` renders a separator run between `<cls>` and the first chain and
    between the last chain and `<eos>`, so no chain is adjacent to a special token.

    `spacer_before_peptide` inserts extra separator tokens immediately before the
    PEPTIDE chain (in addition to the normal separator).

    Truncation: if the full sequence exceeds `max_length` (specials included),
    `strict=True` raises ValueError. With `strict=False` the tail is cut and
    `<eos>` is kept as the last token; chain slices are clipped to the surviving
    tokens, so a fully cut chain gets an empty slice. Scoring code must treat
    clipped chains as invalid, which is why strict is the default.
    """
    if separator_tokens < 0 or spacer_before_peptide < 0 or chain_break_gap < 0:
        raise ValueError("separator_tokens, spacer_before_peptide and chain_break_gap must be >= 0")

    sep_id = tokenizer.pad_token_id
    ids: list[int] = [tokenizer.cls_token_id]
    slices: dict[str, slice] = {}
    # Token index -> extra position offset accumulated from chain_break_gap.
    offsets: list[int] = [0]
    offset = 0

    if symmetric_ends:
        ids.extend([sep_id] * separator_tokens)
        offsets.extend([0] * separator_tokens)

    for i, chain in enumerate(spec.chains):
        if chain.role.value in slices:
            raise ValueError(f"duplicate role {chain.role.value}; chain_slices is keyed by role")
        n_sep = separator_tokens if i > 0 else 0
        if chain.role is Role.PEPTIDE:
            n_sep += spacer_before_peptide
        ids.extend([sep_id] * n_sep)
        offsets.extend([offset] * n_sep)
        if i > 0:
            offset += chain_break_gap

        chain_ids = tokenizer(chain.sequence, add_special_tokens=False)["input_ids"]
        if len(chain_ids) != len(chain.sequence):
            raise ValueError(f"tokenizer is not one-token-per-residue for {chain.role.value}")
        slices[chain.role.value] = slice(len(ids), len(ids) + len(chain_ids))
        ids.extend(chain_ids)
        offsets.extend([offset] * len(chain_ids))

    if symmetric_ends:
        ids.extend([sep_id] * separator_tokens)
        offsets.extend([offset] * separator_tokens)
    ids.append(tokenizer.eos_token_id)
    offsets.append(offset)

    if len(ids) > max_length:
        if strict:
            raise ValueError(
                f"{spec.example_id}: rendered length {len(ids)} exceeds max_length {max_length}"
            )
        ids = ids[: max_length - 1] + [tokenizer.eos_token_id]
        limit = max_length - 1
        offsets = offsets[: max_length - 1] + [offsets[-1]]
        slices = {k: slice(min(s.start, limit), min(s.stop, limit)) for k, s in slices.items()}

    input_ids = torch.tensor(ids, dtype=torch.long)
    position_ids = normalize_position_ids(
        torch.arange(len(ids), dtype=torch.long) + torch.tensor(offsets, dtype=torch.long)
    )
    return RenderedLayout(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        chain_slices=slices,
        position_ids=position_ids,
        chain_ids=None,
    )
