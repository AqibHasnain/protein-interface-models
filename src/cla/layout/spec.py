"""The chain-level representation every other module speaks.

A biological complex is a *set* of chains. Serialization into a single token
string is a representational choice made downstream, in `cla.layout.concat`.
Keeping the two separate is what makes the null-relayout operators in
`cla.layout.perturb` expressible as transformations that provably leave the
complex unchanged.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum

AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


class Role(StrEnum):
    """Functional identity of a chain within the complex."""

    TCRA = "TCRA"
    TCRB = "TCRB"
    PEPTIDE = "PEPTIDE"
    MHC_HEAVY = "MHC_HEAVY"
    B2M = "B2M"


@dataclass(frozen=True, slots=True)
class Chain:
    role: Role
    sequence: str

    def __post_init__(self) -> None:
        if not self.sequence:
            raise ValueError(f"{self.role} chain is empty")
        bad = sorted(set(self.sequence) - AA_ALPHABET)
        if bad:
            raise ValueError(f"{self.role} contains non-standard residues: {bad}")

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True, slots=True)
class ChainSpec:
    """An ordered list of chains plus the labels and provenance carried alongside.

    `chains` is ordered because serialization needs an order, but the biological
    content is the *multiset* of (role, sequence) pairs exposed by `identity`.
    """

    chains: tuple[Chain, ...]
    example_id: str
    label: int | None = None
    epitope: str | None = None
    mhc_allele: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.chains:
            raise ValueError("ChainSpec requires at least one chain")

    @property
    def identity(self) -> Counter[tuple[str, str]]:
        """The biology. Two specs with equal `identity` describe the same complex."""
        return Counter((c.role.value, c.sequence) for c in self.chains)

    @property
    def total_residues(self) -> int:
        return sum(len(c) for c in self.chains)

    def by_role(self, role: Role) -> Chain | None:
        return next((c for c in self.chains if c.role is role), None)

    def reordered(self, order: tuple[int, ...]) -> ChainSpec:
        if sorted(order) != list(range(len(self.chains))):
            raise ValueError(f"{order} is not a permutation of {len(self.chains)} chains")
        return replace(self, chains=tuple(self.chains[i] for i in order))

    def with_chains(self, chains: tuple[Chain, ...]) -> ChainSpec:
        return replace(self, chains=chains)
