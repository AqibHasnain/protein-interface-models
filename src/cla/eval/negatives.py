"""Mismatched-pairing decoys for the audit's binder/non-binder denominator.

VDJdb holds positives only, but NSI divides by the separation between binder and
non-binder scores, so something has to play the non-binder role.

THESE ARE DECOYS, NOT VALIDATED NON-BINDERS. Each decoy pairs a real receptor with the
peptide and MHC of a different epitope. A cross-reactive TCR can bind such a pairing, and
absence from VDJdb is not evidence of non-binding, so some decoys are false negatives.

Direction of bias: unrelated peptide/MHC pairings are easier for a model to tell apart
from true pairings than the borderline non-binders seen in practice, so decoys tend to
inflate the binder/non-binder separation. A larger denominator DEFLATES NSI. Reported NSI
is therefore conservative: a high NSI measured against decoys would still be high against
harder, validated non-binders. Any claim that rests on NSI being low must not rely on this
set; the decoy policy requires reporting decoy-based results separately from any validated
negatives.

Exclusion rule: a decoy is never emitted for a (receptor, epitope) pair that occurs as a
positive anywhere in the known positives. Receptor identity is the (TCRA, TCRB) CDR3 pair,
and the beta CDR3 alone is also checked, since beta chains dominate epitope specificity and
the same beta seen against an epitope should not be paired with it as a negative.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cla.layout.spec import Chain, ChainSpec, Role

# Roles that describe the presented antigen and are swapped as a unit from the donor.
_ANTIGEN_ROLES = (Role.PEPTIDE, Role.MHC_HEAVY, Role.B2M)


def _epitope(spec: ChainSpec) -> str:
    if not spec.epitope:
        raise ValueError(f"{spec.example_id}: epitope is required to build negatives")
    return spec.epitope.strip().upper()


def _receptor_key(spec: ChainSpec) -> tuple[str, str]:
    alpha, beta = spec.by_role(Role.TCRA), spec.by_role(Role.TCRB)
    return (alpha.sequence if alpha else "", beta.sequence if beta else "")


def positive_pairs(specs: Sequence[ChainSpec]) -> tuple[set, set]:
    """(receptor, epitope) and (beta CDR3, epitope) pairs observed as positives."""
    full, beta_only = set(), set()
    for s in specs:
        if s.label == 0:
            continue
        key, ep = _receptor_key(s), _epitope(s)
        full.add((key, ep))
        beta_only.add((key[1], ep))
    return full, beta_only


def make_mismatched_negatives(
    specs: Sequence[ChainSpec],
    rng: np.random.Generator,
    n_per_positive: int = 1,
    *,
    known_positives: Sequence[ChainSpec] | None = None,
) -> list[ChainSpec]:
    """Pair each spec's receptor with the peptide/MHC of a donor from a different epitope.

    Donors are drawn from `specs`. `known_positives` (default `specs`) defines the
    forbidden (receptor, epitope) pairs; pass the full dataset when `specs` is a
    subsample so a decoy cannot collide with a positive outside the subsample.
    A spec with fewer than `n_per_positive` eligible donors yields as many as exist.
    Negatives carry `label=0` and `example_id` of the form `{orig}__neg{k}`.
    """
    if n_per_positive < 1:
        raise ValueError("n_per_positive must be >= 1")
    positives = [s for s in specs if s.label != 0]
    donors = [s for s in positives if s.by_role(Role.PEPTIDE) is not None]
    if len({_epitope(s) for s in donors}) < 2:
        raise ValueError("need at least two distinct epitopes to build mismatched negatives")

    full, beta_only = positive_pairs(known_positives if known_positives is not None else specs)
    donor_epitopes = np.array([_epitope(d) for d in donors])

    negatives: list[ChainSpec] = []
    for spec in positives:
        key, own = _receptor_key(spec), _epitope(spec)
        eligible = np.flatnonzero(
            [
                ep != own and (key, ep) not in full and (key[1], ep) not in beta_only
                for ep in donor_epitopes
            ]
        )
        take = min(n_per_positive, len(eligible))
        for k, di in enumerate(rng.choice(eligible, size=take, replace=False)):
            donor = donors[int(di)]
            donor_chains = {c.role: c for c in donor.chains}
            chains: list[Chain] = []
            for c in spec.chains:
                if c.role in _ANTIGEN_ROLES:
                    if c.role in donor_chains:
                        chains.append(donor_chains[c.role])
                else:
                    chains.append(c)
            negatives.append(
                ChainSpec(
                    chains=tuple(chains),
                    example_id=f"{spec.example_id}__neg{k}",
                    label=0,
                    epitope=donor.epitope,
                    mhc_allele=donor.mhc_allele,
                    source=spec.source,
                )
            )
    return negatives
