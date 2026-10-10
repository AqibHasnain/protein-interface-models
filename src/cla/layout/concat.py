"""Baseline naive concatenation: the serialization under audit.

Layout is `<cls> chain_1 SEP^k chain_2 SEP^k ... chain_n <eos>`.
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
    position_ids: torch.Tensor | None = None
    chain_ids: torch.Tensor | None = None


def render_naive(
    spec: ChainSpec,
    tokenizer,
    *,
    separator_tokens: int = 1,
    spacer_before_peptide: int = 0,
    max_length: int = DEFAULT_MAX_LENGTH,
    strict: bool = True,
) -> RenderedLayout:
    """Concatenate chains in `spec.chains` order into one sequence.

    Separator choice: `tokenizer.pad_token` repeated `separator_tokens` times
    between chains, with attention_mask=1 over them. ESM-2 has no chain
    separator, and a glycine linker would add real residues that the model
    scores and conditions on. The audit measures sensitivity to position
    geometry, so separators must be non-residue tokens that still occupy
    positions; masking them out would leave the geometry unchanged for models
    using default 0..L-1 positions and defeat the purpose of varying their count.

    `spacer_before_peptide` inserts extra separator tokens immediately before the
    PEPTIDE chain (in addition to the normal separator), used by the spacer
    operator to shift downstream offsets without touching any chain.

    Truncation: if the full sequence exceeds `max_length` (specials included),
    `strict=True` raises ValueError. With `strict=False` the tail is cut and
    `<eos>` is kept as the last token; chain slices are clipped to the surviving
    tokens, so a fully cut chain gets an empty slice. Scoring code must treat
    clipped chains as invalid, which is why strict is the default.
    """
    if separator_tokens < 0 or spacer_before_peptide < 0:
        raise ValueError("separator_tokens and spacer_before_peptide must be >= 0")

    sep_id = tokenizer.pad_token_id
    ids: list[int] = [tokenizer.cls_token_id]
    slices: dict[str, slice] = {}

    for i, chain in enumerate(spec.chains):
        if chain.role.value in slices:
            raise ValueError(f"duplicate role {chain.role.value}; chain_slices is keyed by role")
        n_sep = separator_tokens if i > 0 else 0
        if chain.role is Role.PEPTIDE:
            n_sep += spacer_before_peptide
        ids.extend([sep_id] * n_sep)

        chain_ids = tokenizer(chain.sequence, add_special_tokens=False)["input_ids"]
        if len(chain_ids) != len(chain.sequence):
            raise ValueError(f"tokenizer is not one-token-per-residue for {chain.role.value}")
        slices[chain.role.value] = slice(len(ids), len(ids) + len(chain_ids))
        ids.extend(chain_ids)

    ids.append(tokenizer.eos_token_id)

    if len(ids) > max_length:
        if strict:
            raise ValueError(
                f"{spec.example_id}: rendered length {len(ids)} exceeds max_length {max_length}"
            )
        ids = ids[: max_length - 1] + [tokenizer.eos_token_id]
        limit = max_length - 1
        slices = {k: slice(min(s.start, limit), min(s.stop, limit)) for k, s in slices.items()}

    input_ids = torch.tensor(ids, dtype=torch.long)
    return RenderedLayout(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        chain_slices=slices,
        position_ids=None,
        chain_ids=None,
    )
