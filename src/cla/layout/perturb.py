"""Relayout operators for the nuisance-sensitivity audit.

Design: an operator is `(spec, rng) -> (spec', render_overrides)`. The returned
spec carries every change to chain content or order. `render_overrides` carries
every change to serialization and is splatted into `render_naive(spec', tok,
**overrides)`. This keeps "what the complex is" separate from "how it is
written down", so the null claim for an operator reduces to a check on
`spec.identity` plus the knowledge that overrides touch only separators.

The geometry operators G1 and G2 live in `cla.layout.geometry` because they act on a
rendered layout. C1 is a content-changing positive control with controllable magnitude.

P5 needs a sequence pool, so it takes a keyword `pool`; bind it with
`functools.partial(resample_chain_length, pool=pool)` before calling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from cla.layout.spec import AA_ALPHABET, Chain, ChainSpec, Role

Overrides = dict[str, Any]
Operator = Callable[..., tuple[ChainSpec, Overrides]]

SEPARATOR_CHOICES = (0, 1, 5, 25)
PEPTIDE_SPACER_TOKENS = 10


class OperatorNotApplicable(Exception):
    """The operator cannot be applied honestly to this spec or data assembly."""


def permute_chain_order(spec: ChainSpec, rng: np.random.Generator) -> tuple[ChainSpec, Overrides]:
    """Reorder chains. Null because a complex is a set of chains.

    Chain order in a physical complex is undefined; any order describes the same
    molecule, so `identity` is unchanged. Rendering uses `symmetric_ends` so that no
    chain gains or loses adjacency to `<cls>` or `<eos>` by moving slots; this only
    has an effect when `separator_tokens > 0`. The permutation is never the identity
    permutation, so a model that is order-invariant and one that is not are
    distinguishable. Specs with identical chains in different slots can still
    render identically; those cases are unavoidable and harmless.
    """
    n = len(spec.chains)
    if n < 2:
        raise OperatorNotApplicable("need at least two chains to permute")
    identity = tuple(range(n))
    while True:
        order = tuple(int(i) for i in rng.permutation(n))
        if order != identity:
            break
    return spec.reordered(order), {"symmetric_ends": True}


def vary_separator_count(
    spec: ChainSpec, rng: np.random.Generator, *, baseline: int = 0
) -> tuple[ChainSpec, Overrides]:
    """Change the number of separator tokens between chains. Null with respect to the
    biology, but not to the model: separators are content tokens, so this measures
    token insertion, not position geometry (see `cla.layout.geometry` for that).

    The draw excludes `baseline`, the count used to render the original, so the
    result never reproduces the original layout. Choices are 0, 1, 5, 25.
    """
    choices = [c for c in SEPARATOR_CHOICES if c != baseline]
    return spec, {"separator_tokens": int(rng.choice(choices))}


def restore_native_flank(spec: ChainSpec, rng: np.random.Generator) -> tuple[ChainSpec, Overrides]:
    """Pad chains with their native flanking framework residues. Null with respect
    to the interface, since framework flanks lie outside the CDR3 contact region
    being scored.

    Reads `spec.native_flanks`, a mapping of role value to `(left, right)` strings.
    The current data assembly carries only CDR3 sequences and no flanks, and
    `ChainSpec` has no such field, so this operator raises
    `OperatorNotApplicable` and P3 is reported as not applicable. Substituting
    random or germline-guessed residues would change the biology and make the
    result uninterpretable, so refusing is the honest outcome. Unlike P1, P2 and
    P4 this operator changes `identity` when it does apply, so it cannot be
    checked by the multiset invariant.
    """
    flanks: dict[str, tuple[str, str]] | None = getattr(spec, "native_flanks", None)
    if not flanks:
        raise OperatorNotApplicable(
            "no native flank residues available; data assembly carries CDR3 only"
        )
    chains = []
    for c in spec.chains:
        left, right = flanks.get(c.role.value, ("", ""))
        chains.append(Chain(c.role, left + c.sequence + right))
    return spec.with_chains(tuple(chains)), {}


def insert_neutral_spacer(spec: ChainSpec, rng: np.random.Generator) -> tuple[ChainSpec, Overrides]:
    """Insert a fixed run of separator tokens immediately before the PEPTIDE chain.
    Null because no chain changes.

    The extra tokens are non-residue and only shift the peptide and anything after
    it relative to the chains before it, changing relative offsets while every
    (role, sequence) pair is preserved.
    """
    if spec.by_role(Role.PEPTIDE) is None:
        raise OperatorNotApplicable("spec has no PEPTIDE chain")
    return spec, {"spacer_before_peptide": PEPTIDE_SPACER_TOKENS}


def resample_chain_length(
    spec: ChainSpec,
    rng: np.random.Generator,
    *,
    pool: Sequence[str] | None = None,
) -> tuple[ChainSpec, Overrides]:
    """POSITIVE CONTROL, deliberately not null: swap the CDR3B for a different real
    CDR3B of a different length.

    This changes the biology (a different TCR beta chain, so the binding partner
    changes) and the sequence length. A score change is expected; an audit
    pipeline that reports no change here is insensitive and cannot be trusted
    on the null operators either. `pool` is a collection of real CDR3B
    sequences; without it the operator raises ValueError.
    """
    if not pool:
        raise ValueError("resample_chain_length requires a non-empty pool of CDR3B sequences")
    current = spec.by_role(Role.TCRB)
    if current is None:
        raise OperatorNotApplicable("spec has no TCRB chain")
    candidates = [s for s in pool if len(s) != len(current)]
    if not candidates:
        raise OperatorNotApplicable("pool has no CDR3B of a different length")
    new_seq = candidates[int(rng.integers(len(candidates)))]
    chains = tuple(Chain(c.role, new_seq) if c.role is Role.TCRB else c for c in spec.chains)
    return spec.with_chains(chains), {}


def mutate_context_residues(
    spec: ChainSpec,
    rng: np.random.Generator,
    *,
    n_mutations: int,
    target_role: Role | str,
) -> tuple[ChainSpec, Overrides]:
    """POSITIVE CONTROL: substitute `n_mutations` residues in chains other than the
    scored chain, each to a different amino acid.

    Positions are drawn uniformly without replacement over all residues of the
    non-target chains. The result is a real sequence change of controllable
    magnitude, which gives the audit a denominator for "how far does the score
    move when the context actually changes".
    """
    target = Role(target_role)
    others = [i for i, c in enumerate(spec.chains) if c.role is not target]
    sites = [(i, j) for i in others for j in range(len(spec.chains[i]))]
    if n_mutations > len(sites):
        raise OperatorNotApplicable(f"only {len(sites)} context residues for {n_mutations} edits")
    chosen = rng.choice(len(sites), size=n_mutations, replace=False)
    edits: dict[int, dict[int, str]] = {}
    for k in chosen:
        i, j = sites[int(k)]
        old = spec.chains[i].sequence[j]
        new = str(rng.choice(sorted(AA_ALPHABET - {old})))
        edits.setdefault(i, {})[j] = new
    chains = []
    for i, c in enumerate(spec.chains):
        seq = list(c.sequence)
        for j, aa in edits.get(i, {}).items():
            seq[j] = aa
        chains.append(Chain(c.role, "".join(seq)))
    return spec.with_chains(tuple(chains)), {}


OPERATORS: dict[str, Operator] = {
    "P1": permute_chain_order,
    "P2": vary_separator_count,
    "P3": restore_native_flank,
    "P4": insert_neutral_spacer,
    "P5": resample_chain_length,
}

CONTROLS: dict[str, Operator] = {
    "C1": mutate_context_residues,
    "C2": resample_chain_length,
}
