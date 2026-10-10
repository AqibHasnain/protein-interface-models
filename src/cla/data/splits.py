"""Deterministic train/val/test splitting with hard held-out alleles and epitopes.

Leakage is the single most likely way this project produces a wrong result, so
the split is defined by membership tests against `configs/splits.yaml` rather
than by a random partition that happens to separate things. A held-out epitope
removes every TCR observed against it, not just the exact row: a TCR that binds
YLQPRTFLL appearing in training with a different annotation would leak the
answer for the evaluation set.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from cla.layout.spec import ChainSpec, Role

DEFAULT_SPLITS_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "splits.yaml"


@dataclass(frozen=True, slots=True)
class SplitConfig:
    held_out_alleles: frozenset[str]
    held_out_epitopes: frozenset[str]
    seed: int
    val_fraction: float

    @classmethod
    def load(cls, path: Path | str = DEFAULT_SPLITS_CONFIG) -> SplitConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            held_out_alleles=frozenset(normalize_allele(a) for a in raw["held_out_alleles"]),
            held_out_epitopes=frozenset(e.strip().upper() for e in raw["held_out_epitopes"]),
            seed=int(raw.get("seed", 0)),
            val_fraction=float(raw.get("val_fraction", 0.1)),
        )


def normalize_allele(allele: str) -> str:
    """Collapse an IMGT allele name to two fields, the resolution the splits are defined at.

    `HLA-A*02:01:01:01` and `HLA-A*02:01` are the same protein for our purposes;
    treating them as distinct would let a synonymous spelling cross the split.
    """
    a = allele.strip().upper().replace("HLA-", "")
    if "*" not in a:
        return f"HLA-{a}"
    locus, fields = a.split("*", 1)
    parts = fields.split(":")
    return f"HLA-{locus}*{':'.join(parts[:2])}"


def _tcr_key(spec: ChainSpec) -> str | None:
    """Identity of the receptor, used to propagate an epitope holdout to its TCRs."""
    alpha = spec.by_role(Role.TCRA)
    beta = spec.by_role(Role.TCRB)
    if alpha is None and beta is None:
        return None
    return f"{alpha.sequence if alpha else ''}|{beta.sequence if beta else ''}"


def held_out_tcr_keys(specs: Iterable[ChainSpec], config: SplitConfig) -> set[str]:
    keys: set[str] = set()
    for spec in specs:
        if spec.epitope and spec.epitope.strip().upper() in config.held_out_epitopes:
            key = _tcr_key(spec)
            if key is not None:
                keys.add(key)
    return keys


def assign_split(
    spec: ChainSpec,
    config: SplitConfig,
    *,
    banned_tcrs: frozenset[str] = frozenset(),
) -> str:
    """Return one of 'test_allele', 'test_epitope', 'val' or 'train'.

    Epitope holdout is checked before allele holdout so an example matching both
    lands in a single, stable bucket.
    """
    epitope = (spec.epitope or "").strip().upper()
    if epitope and epitope in config.held_out_epitopes:
        return "test_epitope"

    key = _tcr_key(spec)
    if key is not None and key in banned_tcrs:
        return "test_epitope"

    if spec.mhc_allele and normalize_allele(spec.mhc_allele) in config.held_out_alleles:
        return "test_allele"

    if _hash_fraction(spec.example_id, config.seed) < config.val_fraction:
        return "val"
    return "train"


def _hash_fraction(example_id: str, seed: int) -> float:
    """Stable per-example uniform draw.

    Hashing the id rather than shuffling means the assignment does not change
    when rows are added, removed or reordered upstream.
    """
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_specs(
    specs: Sequence[ChainSpec], config: SplitConfig | None = None
) -> dict[str, list[ChainSpec]]:
    config = config or SplitConfig.load()
    banned = frozenset(held_out_tcr_keys(specs, config))
    out: dict[str, list[ChainSpec]] = {
        "train": [],
        "val": [],
        "test_allele": [],
        "test_epitope": [],
    }
    for spec in specs:
        out[assign_split(spec, config, banned_tcrs=banned)].append(spec)
    return out
