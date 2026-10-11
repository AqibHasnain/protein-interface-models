"""Chain-aware positions for ESM-2: three independently switchable components.

* `reset`: position index restarts for every chain, so a chain's internal geometry no longer
  depends on how many residues precede it.
* `chain_embedding`: a learned vector per role added to the token embeddings.
* `offset`: a jump added to the index at each chain boundary (a fixed constant, or one
  learned scalar per ordered role pair when `learned_offset` is set).

Drop-in contract. ESM-2 positions enter only through rotary embeddings and `EsmModel.forward`
honors `position_ids`, so `reset` and `offset` are a remap of that one tensor: `apply` returns a
`RenderedLayout` whose `position_ids` the existing scoring path already passes to the model.
With everything off `apply` returns the layout unchanged (`position_ids=None`, i.e. arange).

The chain embedding is a forward hook on `model.esm.embeddings`. The hook adds to the output of
`EsmEmbeddings`, after token dropout has zeroed and rescaled masked tokens, so a masked residue
keeps its role. No ESM-2 code or weights change and `detach` removes it. The hook reads the
per-token role ids from `bound(...)`; the scoring path ignores `RenderedLayout.chain_ids`, so a
trainer that uses this component must wrap its forward in `bound`.

Each chain's residues take indices `RESET_BASE + 0..L-1`. The base of 1 keeps `<cls>` at index 0
from sharing an index with the first residue. Separator and `<eos>` tokens continue the index of
the chain before them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import torch
from torch import nn

from cla.layout.concat import RenderedLayout, normalize_position_ids
from cla.layout.spec import Role

ROLES = tuple(Role)
RESET_BASE = 1


class ChainAwarePositions(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        reset: bool = True,
        chain_embedding: bool = True,
        offset: int | None = 200,
        learned_offset: bool = False,
    ) -> None:
        super().__init__()
        if offset is not None and offset < 0:
            raise ValueError("offset must be >= 0 or None")
        if learned_offset and offset is None:
            raise ValueError("learned_offset needs an initial offset")
        self.reset, self.offset, self.learned = reset, offset, learned_offset
        # Index 0 is "not a chain residue" and stays zero. Zero init makes the component an
        # exact no-op at the start of training.
        self.role_embedding = (
            nn.Embedding(len(ROLES) + 1, hidden_size, padding_idx=0) if chain_embedding else None
        )
        if self.role_embedding is not None:
            nn.init.zeros_(self.role_embedding.weight)
        self.pair_offsets = (
            nn.Parameter(torch.full((len(ROLES) ** 2,), float(offset))) if learned_offset else None
        )
        self._bound: torch.Tensor | None = None
        self._handle = None

    def _ordered(self, rendered: RenderedLayout) -> list[tuple[int, int, Role]]:
        chains = [(s.start, s.stop, Role(r)) for r, s in rendered.chain_slices.items()]
        return sorted(c for c in chains if c[1] > c[0])

    def position_ids(self, rendered: RenderedLayout) -> torch.Tensor | None:
        """Position ids for `rendered`; None when they equal plain arange."""
        n = rendered.input_ids.shape[0]
        pos = (
            rendered.position_ids
            if rendered.position_ids is not None
            else torch.arange(n, dtype=torch.long)
        ).clone()
        chains = self._ordered(rendered)
        if self.reset and chains:
            # A chain's segment runs to the next chain's start, so trailing separators follow it.
            pos[: chains[0][0]] = torch.arange(chains[0][0])
            for k, (start, _, _) in enumerate(chains):
                end = chains[k + 1][0] if k + 1 < len(chains) else n
                pos[start:end] = RESET_BASE + torch.arange(end - start)
        if self.offset is not None:
            if self.learned:
                pos = pos.float()
            for k in range(1, len(chains)):
                pair = ROLES.index(chains[k - 1][2]) * len(ROLES) + ROLES.index(chains[k][2])
                jump = self.pair_offsets[pair] if self.learned else self.offset
                pos[chains[k][0] :] += jump
        return pos if self.learned else normalize_position_ids(pos)

    def chain_ids(self, rendered: RenderedLayout) -> torch.Tensor:
        """Per-token role index (1-based), 0 for `<cls>`, separators and `<eos>`."""
        ids = torch.zeros(rendered.input_ids.shape[0], dtype=torch.long)
        for start, stop, role in self._ordered(rendered):
            ids[start:stop] = ROLES.index(role) + 1
        return ids

    def apply(self, rendered: RenderedLayout) -> RenderedLayout:
        """Return `rendered` with `position_ids` and `chain_ids` set for the model."""
        return replace(
            rendered, position_ids=self.position_ids(rendered), chain_ids=self.chain_ids(rendered)
        )

    @contextmanager
    def bound(self, chain_ids: torch.Tensor):
        """Make `chain_ids` (batch, length; 0 for padding) visible to the embedding hook."""
        self._bound = chain_ids
        try:
            yield
        finally:
            self._bound = None

    def attach(self, model: nn.Module) -> None:
        if self.role_embedding is not None and self._handle is None:
            self._handle = model.esm.embeddings.register_forward_hook(self._add_roles)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _add_roles(self, _module, _args, out: torch.Tensor) -> torch.Tensor:
        if self._bound is None:
            return out
        return out + self.role_embedding(self._bound.to(out.device)).to(out.dtype)
