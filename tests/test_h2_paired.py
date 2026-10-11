"""The paired comparison must refuse to pair runs that are not actually paired.

A half-finished sweep is the realistic failure: if the baseline has three seeds and an arm
has one, zipping them silently compares arm seed 0 against baseline seed 0 and drops the
rest, or worse, reports a delta computed from a different seed. The delta is the headline
number for H2, so a wrong one is not recoverable by eye.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cla.eval.h2_paired import collect


def _write(root: Path, arm: str, seed: int, allele: float, epitope: float) -> None:
    d = root / "runs" / f"{arm}_s{seed}"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "eval": {
                    "test_allele": {"auroc": {"point": allele}},
                    "test_epitope": {"auroc": {"point": epitope}},
                }
            }
        )
    )


def test_partial_sweep_is_dropped_and_complete_pairs_keep_seed_order(tmp_path: Path) -> None:
    full, partial = tmp_path / "full", tmp_path / "partial"
    for seed, (base, treat) in enumerate([(0.50, 0.60), (0.52, 0.62), (0.54, 0.64)]):
        _write(full, "A0", seed, base, base)
        _write(full, "A4", seed, treat, treat)
        _write(partial, "A0", seed, base, base)
    _write(partial, "A4", 0, 0.99, 0.99)

    pairs = collect(sizes=(("full", full), ("partial", partial)))

    assert {p.size for p in pairs} == {"full"}, "an arm with fewer seeds must not be paired"
    allele = next(p for p in pairs if p.arm == "A4" and p.shard == "test_allele")
    assert allele.diffs == pytest.approx([0.10, 0.10, 0.10])
    assert allele.unanimous
