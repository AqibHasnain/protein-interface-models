"""Assemble TCR-pMHC ChainSpecs from VDJdb and IPD-IMGT/HLA.

Limitation: VDJdb stores CDR3 loops only, so the TCRA and TCRB chains in the
assembled specs are CDR3 sequences, not full-length receptor chains. Rebuilding
full chains from V/J gene calls is out of scope here.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd

from cla.data.sources import B2M_MATURE, ensure_all
from cla.layout.spec import Chain, ChainSpec, Role

log = logging.getLogger(__name__)

# Class I leaders are 24 residues for A/B/C (21 for E). The mature N-terminus is
# conserved, so the motif is the primary cut point and a fixed trim is the fallback.
MATURE_MOTIF = re.compile(r"[GC]SHS[ML][RK]Y")
MOTIF_SEARCH_WINDOW = 40
FALLBACK_SIGNAL_LEN = 24

# Matches "HLA-A*02:01:01:01", "A*02:01", "HLA-A*02", "HLA-A2". Fields past the
# second only distinguish synonymous or non-coding variants.
_ALLELE_RE = re.compile(r"^(?:HLA-)?([A-Z]+)\*?(\d{1,3})(?::(\d{2,3}))?(?::\d{2,3}){0,2}[NLSQCA]?$")
# Null/low/secreted/questionable/aberrant/cytoplasmic alleles are not canonical proteins.
_SUFFIXED_RE = re.compile(r"[NLSQCA]$")

SOURCE_TAG = "vdjdb"

MhcDomains = Literal["a1a2", "ectodomain", "full"]
# Last mature-chain residue kept. The alpha-1/alpha-2 groove contacts both peptide and TCR;
# the alpha-3 domain, transmembrane segment and tail add ~160 residues that contact neither
# and would push rendered complexes toward the 512-token limit once operators add tokens.
MHC_DOMAIN_END: dict[str, int | None] = {"a1a2": 180, "ectodomain": 275, "full": None}


def normalize_allele(name: str) -> str | None:
    """Reduce an allele string to 'GENE*GG' or 'GENE*GG:PP' (no HLA- prefix)."""
    m = _ALLELE_RE.match(name.strip())
    if m is None:
        return None
    gene, group, protein = m.groups()
    key = f"{gene}*{int(group):02d}"
    return f"{key}:{protein}" if protein else key


def trim_signal_peptide(seq: str) -> tuple[str, bool]:
    """Return (mature sequence, whether the motif was found)."""
    m = MATURE_MOTIF.search(seq[:MOTIF_SEARCH_WINDOW])
    if m is not None:
        return seq[m.start() :], True
    return seq[FALLBACK_SIGNAL_LEN:], False


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, list[str]]] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            fields = line.split()
            # IMGT headers look like ">HLA:HLA00001 A*01:01:01:01 365 bp".
            records.append((fields[1], []))
        elif line.strip() and records:
            records[-1][1].append(line.strip())
    return [(name, "".join(parts)) for name, parts in records]


class AlleleResolver:
    """Maps allele strings to mature alpha-chain sequences from hla_prot.fasta."""

    def __init__(self, records: list[tuple[str, str]]):
        self.alleles: dict[str, str] = {}
        self.fallback_trims: list[str] = []
        self.unresolved: Counter[str] = Counter()
        # Both indexes keep the first entry in file order. IMGT lists the lowest-numbered
        # allele of each group first, so a bare "A*02" resolves to A*02:01.
        two_field: dict[str, str] = {}
        one_field: dict[str, str] = {}
        for name, seq in records:
            if _SUFFIXED_RE.search(name):
                continue
            key = normalize_allele(name)
            if key is None or ":" not in key:
                continue
            two_field.setdefault(key, seq)
            group = key.split(":")[0]
            if group not in one_field:
                one_field[group] = key
        self._raw = two_field
        self._group_default = one_field

    @classmethod
    def from_fasta(cls, path: Path) -> AlleleResolver:
        return cls(_read_fasta(path))

    def canonical_name(self, allele: str) -> str | None:
        key = normalize_allele(allele)
        if key is None:
            return None
        if ":" not in key:
            key = self._group_default.get(key, key)
        return key if key in self._raw else None

    def resolve(self, allele: str) -> str | None:
        key = self.canonical_name(allele)
        if key is None:
            self.unresolved[allele] += 1
            return None
        if key not in self.alleles:
            mature, found = trim_signal_peptide(self._raw[key])
            if not found:
                self.fallback_trims.append(key)
                log.warning(
                    "%s: mature motif not found, trimmed %d residues", key, FALLBACK_SIGNAL_LEN
                )
            self.alleles[key] = mature
        return self.alleles[key]


def load_vdjdb(path: Path) -> pd.DataFrame:
    """Load vdjdb_full.txt and keep human MHC class I paired alpha/beta rows.

    Filters, applied in order (row counts after each are in `df.attrs["filters"]`):
      1. species == HomoSapiens
      2. mhc.class == MHCI
      3. cdr3.alpha and cdr3.beta both non-empty (paired receptors only)
      4. antigen.epitope non-empty
      5. mhc.a parses as an HLA allele (database lookup happens in build_chainspecs)
    `vdjdb.score` is kept as an int column.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    filters: list[tuple[str, int]] = [("raw", len(df))]

    steps = [
        ("species == HomoSapiens", df["species"] == "HomoSapiens"),
        ("mhc.class == MHCI", df["mhc.class"] == "MHCI"),
        (
            "cdr3.alpha and cdr3.beta non-empty",
            (df["cdr3.alpha"].str.strip() != "") & (df["cdr3.beta"].str.strip() != ""),
        ),
        ("antigen.epitope non-empty", df["antigen.epitope"].str.strip() != ""),
        (
            "mhc.a is an HLA allele",
            df["mhc.a"].str.startswith("HLA-")
            & df["mhc.a"].map(lambda a: normalize_allele(a) is not None),
        ),
    ]
    for label, mask in steps:
        # Masks were built on the unfiltered frame, so align them to the current one.
        df = df[mask.loc[df.index]]
        filters.append((label, len(df)))

    df = df.copy()
    df["vdjdb.score"] = df["vdjdb.score"].astype(int)
    df.attrs["filters"] = filters
    return df


class _InvalidChain(Exception):
    def __init__(self, role: Role):
        self.role = role


def _make_chain(role: Role, seq: str) -> Chain:
    try:
        return Chain(role, seq)
    except ValueError as e:
        raise _InvalidChain(role) from e


def trim_mhc(seq: str, mhc_domains: MhcDomains, allele: str = "") -> str:
    """Cut a mature heavy chain to the requested domains.

    A chain shorter than the cut point usually means the signal peptide was mis-cut
    for that allele, so it is kept whole and reported rather than dropped.
    """
    if mhc_domains not in MHC_DOMAIN_END:
        raise ValueError(f"mhc_domains must be in {sorted(MHC_DOMAIN_END)}, got {mhc_domains!r}")
    end = MHC_DOMAIN_END[mhc_domains]
    if end is None:
        return seq
    if len(seq) < end:
        log.warning("%s: mature chain has %d residues, expected >= %d", allele, len(seq), end)
    return seq[:end]


def build_chainspecs(
    df: pd.DataFrame,
    resolver: AlleleResolver,
    b2m: str = B2M_MATURE,
    mhc_domains: MhcDomains = "a1a2",
) -> tuple[list[ChainSpec], Counter[str]]:
    """Build one ChainSpec per valid row, plus a count of rows dropped per reason.

    Chains are ordered TCRA, TCRB, PEPTIDE, MHC_HEAVY, B2M. The TCR chains hold
    CDR3 loops only (see the module docstring). Every VDJdb row is a positive (label=1).
    The MHC chain is trimmed per `mhc_domains` (see `MHC_DOMAIN_END`).
    """
    specs: list[ChainSpec] = []
    dropped: Counter[str] = Counter()
    # Trimming is per allele, so each allele is trimmed (and warned about) once.
    trimmed: dict[str, str] = {}

    for idx, row in df.iterrows():
        allele = row["mhc.a"]
        mhc_seq = resolver.resolve(allele)
        if mhc_seq is None:
            dropped["unresolved_allele"] += 1
            continue
        if allele not in trimmed:
            trimmed[allele] = trim_mhc(mhc_seq, mhc_domains, str(resolver.canonical_name(allele)))

        parts = [
            (Role.TCRA, row["cdr3.alpha"].strip().upper()),
            (Role.TCRB, row["cdr3.beta"].strip().upper()),
            (Role.PEPTIDE, row["antigen.epitope"].strip().upper()),
            (Role.MHC_HEAVY, trimmed[allele]),
            (Role.B2M, b2m),
        ]
        try:
            chains = tuple(_make_chain(role, seq) for role, seq in parts)
        except _InvalidChain as e:
            dropped[f"invalid_{e.role.value.lower()}"] += 1
            continue

        specs.append(
            ChainSpec(
                chains=chains,
                example_id=f"{SOURCE_TAG}-{idx}",
                label=1,
                epitope=chains[2].sequence,
                mhc_allele=f"HLA-{resolver.canonical_name(allele)}",
                source=SOURCE_TAG,
            )
        )
    return specs, dropped


def dedupe_specs(
    specs: list[ChainSpec], scores: dict[str, int] | None = None
) -> tuple[list[ChainSpec], int]:
    """Collapse specs with equal `identity`; return (kept, number collapsed).

    VDJdb lists one row per clonotype per study and donor, so repeats would over-weight
    replicated complexes in medians and break the independence a bootstrap assumes.
    Each group keeps its highest-scoring member (the earliest on ties or without
    `scores`, keyed by example_id) at the position of the group's first occurrence.
    """
    scores = scores or {}
    best: dict[frozenset, int] = {}
    for i, s in enumerate(specs):
        key = frozenset(s.identity.items())
        j = best.get(key)
        if j is None or scores.get(s.example_id, 0) > scores.get(specs[j].example_id, 0):
            best[key] = i
    first_seen: dict[frozenset, int] = {}
    for i, s in enumerate(specs):
        first_seen.setdefault(frozenset(s.identity.items()), i)
    order = sorted(best, key=first_seen.__getitem__)
    return [specs[best[k]] for k in order], len(specs) - len(best)


def save_specs(specs: list[ChainSpec], path: Path, mhc_domains: MhcDomains | None = None) -> None:
    """Write one JSON record per spec; `mhc_domains` is stored on every record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in specs:
            rec = {
                "example_id": s.example_id,
                "label": s.label,
                "epitope": s.epitope,
                "mhc_allele": s.mhc_allele,
                "source": s.source,
                "mhc_domains": mhc_domains,
                "chains": [{"role": c.role.value, "sequence": c.sequence} for c in s.chains],
            }
            f.write(json.dumps(rec) + "\n")


def load_specs(path: Path) -> list[ChainSpec]:
    specs = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            rec.pop("mhc_domains", None)
            chains = tuple(Chain(Role(c["role"]), c["sequence"]) for c in rec.pop("chains"))
            specs.append(ChainSpec(chains=chains, **rec))
    return specs


def assemble(
    out_path: Path = Path("data/processed/vdjdb_specs.jsonl"),
    raw_root: Path = Path("data/raw"),
    mhc_domains: MhcDomains = "a1a2",
) -> dict:
    """Run download, filtering, assembly, deduplication and save. Returns summary counts."""
    paths = ensure_all(raw_root)
    df = load_vdjdb(paths["vdjdb"])
    resolver = AlleleResolver.from_fasta(paths["hla_prot"])
    specs, dropped = build_chainspecs(df, resolver, mhc_domains=mhc_domains)
    scores = {f"{SOURCE_TAG}-{i}": int(v) for i, v in df["vdjdb.score"].items()}
    kept, collapsed = dedupe_specs(specs, scores)
    save_specs(kept, out_path, mhc_domains)
    return {
        "filtered_rows": len(df),
        "dropped": dict(dropped),
        "assembled": len(specs),
        "duplicates_collapsed": collapsed,
        "saved": len(kept),
        "mhc_domains": mhc_domains,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mhc-domains", choices=sorted(MHC_DOMAIN_END), default="a1a2")
    ap.add_argument("--out", type=Path, default=Path("data/processed/vdjdb_specs.jsonl"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(assemble(args.out, mhc_domains=args.mhc_domains), indent=2))
