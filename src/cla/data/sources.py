"""Idempotent downloaders for the raw data behind the TCR-pMHC assembly.

Every source is either pinned (a SHA256 is committed below) or tracks a moving
upstream file. Moving files get their checksum recorded in
`OBSERVED_CHECKSUMS.json` on first download and are verified against it after.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests

log = logging.getLogger(__name__)

OBSERVED_FILE = "OBSERVED_CHECKSUMS.json"

VDJDB_TAG = "2026-06-03"
VDJDB_ZIP_URL = (
    f"https://github.com/antigenomics/vdjdb-db/releases/download/{VDJDB_TAG}/vdjdb-{VDJDB_TAG}.zip"
)
VDJDB_ZIP_SHA256 = "9ae7724ab2ce9fdb09b1fb82a90c7462f75269f14f669238f8c154a80239cfc1"
VDJDB_FULL_SHA256 = "55ee89fa1a0109f52f60fe30fd47c2f6842f94a0476d034b2d98b2511748bddc"
VDJDB_ZIP_MEMBER = f"vdjdb-{VDJDB_TAG}/vdjdb_full.txt"

# IPD-IMGT/HLA release 3.65.0 (2026-07-14). The raw URL is pinned to the release
# commit instead of the `Latest` branch so the checksum stays valid.
IMGTHLA_COMMIT = "37b324ba5d4fef45169c8b162c19bc851a1f8068"
HLA_PROT_URL = f"https://raw.githubusercontent.com/ANHIG/IMGTHLA/{IMGTHLA_COMMIT}/hla_prot.fasta"
HLA_PROT_SHA256 = "a6ff785f245b6cfc194c4b96176ed19e2bed21457ef98f86d7947b6606356db3"

B2M_URL = "https://rest.uniprot.org/uniprotkb/P61769.fasta"
# UniProt P61769 residues 21-119; residues 1-20 are the signal peptide.
B2M_MATURE = (
    "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRV"
    "NHVTLSQPKIVKWDRDM"
)
B2M_SIGNAL_LEN = 20


@dataclass(frozen=True)
class Source:
    """`sha256=None` marks a moving file whose checksum is recorded on first download."""

    name: str
    url: str
    filename: str
    sha256: str | None = None


SOURCES = {
    "hla_prot": Source("hla_prot", HLA_PROT_URL, "hla_prot.fasta", HLA_PROT_SHA256),
    "b2m": Source("b2m", B2M_URL, "b2m_P61769.fasta"),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name so an interrupted download is never mistaken for a valid file.
    part = dest.with_name(dest.name + ".part")
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    part.replace(dest)


def _read_observed(root: Path) -> dict:
    p = root / OBSERVED_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def _write_observed(root: Path, observed: dict) -> None:
    (root / OBSERVED_FILE).write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")


def _fetch_verified(src: Source, root: Path) -> Path:
    dest = root / src.filename
    observed = _read_observed(root)
    expected = src.sha256 or observed.get(src.filename, {}).get("sha256")

    if dest.exists():
        if expected is None or sha256_of(dest) == expected:
            if expected is None:
                observed[src.filename] = _observation(src, dest)
                _write_observed(root, observed)
            return dest
        log.warning("%s fails checksum, re-downloading", dest)
        dest.unlink()

    _download(src.url, dest)
    actual = sha256_of(dest)
    if src.sha256 is not None:
        if actual != src.sha256:
            dest.unlink()
            raise RuntimeError(f"{src.name}: sha256 {actual} != pinned {src.sha256}")
    elif expected is None:
        observed[src.filename] = _observation(src, dest)
        _write_observed(root, observed)
    elif actual != expected:
        dest.unlink()
        raise RuntimeError(
            f"{src.name}: upstream changed (sha256 {actual} != observed {expected}). "
            f"Delete its entry in {OBSERVED_FILE} to accept the new version."
        )
    return dest


def _observation(src: Source, path: Path) -> dict:
    return {"sha256": sha256_of(path), "url": src.url, "retrieved": date.today().isoformat()}


def ensure_vdjdb(root: Path) -> Path:
    full = root / "vdjdb_full.txt"
    if full.exists() and sha256_of(full) == VDJDB_FULL_SHA256:
        return full

    zip_path = root / f"vdjdb-{VDJDB_TAG}.zip"
    if not (zip_path.exists() and sha256_of(zip_path) == VDJDB_ZIP_SHA256):
        zip_path.unlink(missing_ok=True)
        _download(VDJDB_ZIP_URL, zip_path)
        actual = sha256_of(zip_path)
        if actual != VDJDB_ZIP_SHA256:
            zip_path.unlink()
            raise RuntimeError(f"vdjdb zip: sha256 {actual} != pinned {VDJDB_ZIP_SHA256}")

    # Read the member by exact name and write to a fixed path, so archive entry
    # names never influence where files land.
    with zipfile.ZipFile(zip_path) as zf, zf.open(VDJDB_ZIP_MEMBER) as src, full.open("wb") as out:
        for block in iter(lambda: src.read(1 << 20), b""):
            out.write(block)
    actual = sha256_of(full)
    if actual != VDJDB_FULL_SHA256:
        full.unlink()
        raise RuntimeError(f"vdjdb_full.txt: sha256 {actual} != pinned {VDJDB_FULL_SHA256}")
    return full


def ensure_b2m(root: Path) -> Path:
    path = _fetch_verified(SOURCES["b2m"], root)
    seq = "".join(line for line in path.read_text().splitlines() if not line.startswith(">"))
    if seq[B2M_SIGNAL_LEN:] != B2M_MATURE:
        raise RuntimeError("UniProt P61769 no longer matches the hardcoded mature B2M sequence")
    return path


def ensure_all(root: Path = Path("data/raw")) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "vdjdb": ensure_vdjdb(root),
        "hla_prot": _fetch_verified(SOURCES["hla_prot"], root),
        "b2m": ensure_b2m(root),
    }
