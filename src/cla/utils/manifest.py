from __future__ import annotations

import hashlib
import json
import platform
import secrets
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from cla.utils.logging import json_default

_CHUNK_SIZE = 1 << 20  # 1 MiB; data files can be ~100 MB, so never read them whole.
_GIT_TIMEOUT_S = 10
_PACKAGES = ("torch", "transformers", "numpy")


@dataclass
class RunManifest:
    run_id: str
    git_sha: str
    git_dirty: bool
    config: dict
    config_hash: str
    data_checksums: dict[str, str]
    package_versions: dict[str, str]
    device: str
    started_at: str


def config_hash(config: dict) -> str:
    # sort_keys makes the hash independent of insertion order, so two runs with the
    # same settings written in a different order still compare equal.
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
    # Runs against the current working directory. A missing git binary or a
    # non-repo directory yields None instead of raising, so manifests still get written.
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_state() -> tuple[str, bool]:
    head = _git("rev-parse", "HEAD")
    if head is None or head.returncode != 0:
        return "unknown", False
    status = _git("status", "--porcelain")
    dirty = status is not None and status.returncode == 0 and bool(status.stdout.strip())
    return head.stdout.strip(), dirty


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def build_manifest(
    config: dict,
    *,
    data_files: Iterable[Path] = (),
    device: str = "cpu",
) -> RunManifest:
    now = datetime.now(UTC)
    git_sha, git_dirty = _git_state()
    return RunManifest(
        # The random suffix keeps two runs started in the same second distinct.
        run_id=f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}",
        git_sha=git_sha,
        git_dirty=git_dirty,
        config=config,
        config_hash=config_hash(config),
        data_checksums={str(p): file_sha256(Path(p)) for p in data_files},
        package_versions=_package_versions(),
        device=device,
        started_at=now.isoformat(),
    )


def write_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    payload: dict[str, Any] = asdict(manifest)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    return path
