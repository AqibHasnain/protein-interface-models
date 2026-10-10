"""Pure-geometry operators on a rendered layout.

These take a `RenderedLayout` rather than a `ChainSpec` because their defining
property is that `input_ids`, `attention_mask` and `chain_slices` are untouched.
Only `position_ids` changes, so nullness is tensor equality that a test can assert.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch

from cla.layout.concat import RenderedLayout, normalize_position_ids
from cla.layout.spec import Role


def _with_jumps(rendered: RenderedLayout, starts: list[int], gap: int) -> RenderedLayout:
    """Add `gap` to every position at or after each token index in `starts`."""
    if gap < 0:
        raise ValueError("gap must be >= 0")
    n = rendered.input_ids.shape[0]
    base = (
        rendered.position_ids
        if rendered.position_ids is not None
        else torch.arange(n, dtype=torch.long)
    )
    idx = torch.arange(n)
    shift = sum((gap * (idx >= s).long() for s in starts), torch.zeros(n, dtype=torch.long))
    return replace(rendered, position_ids=normalize_position_ids(base + shift))


def gap_at_chain_breaks(rendered: RenderedLayout, gap: int) -> RenderedLayout:
    """Jump the position index by `gap` at the start of every chain after the first.

    `input_ids` is unchanged, so no content differs. Rotary attention depends only on
    index differences, so this changes exactly the cross-chain relative offsets and
    nothing else. Offsets between residues of the same chain are preserved.
    """
    starts = sorted(s.start for s in rendered.chain_slices.values())[1:]
    return _with_jumps(rendered, starts, gap)


def gap_before_peptide(rendered: RenderedLayout, gap: int) -> RenderedLayout:
    """Jump the position index by `gap` immediately before the PEPTIDE chain.

    `input_ids` is unchanged, so no content differs. Rotary attention depends only on
    index differences, so this changes exactly the offsets between the peptide (and
    everything after it) and the chains before it, and nothing else. When the peptide
    is the first chain the jump falls between `<cls>` and the peptide.
    """
    return _with_jumps(rendered, [rendered.chain_slices[Role.PEPTIDE.value].start], gap)


GEOMETRY_OPERATORS: dict[str, Callable[[RenderedLayout, int], RenderedLayout]] = {
    "G1": gap_at_chain_breaks,
    "G2": gap_before_peptide,
}
