"""Paired-by-seed comparison of each chain-aware arm against the matched baseline.

    python -m cla.eval.h2_paired

Seeds are shared across arms: seed k fixes the training subsample, the initialization of
the shared parameters and the batch order, so arm A0 seed k and arm A4 seed k differ only
by the module. Comparing arm means ignores that pairing and buries a consistent shift
under between-seed variance, which is several times larger. The paired difference is the
quantity the experiment was designed to produce.

The sign test across seeds carries more weight here than the t-statistic. Three seeds give
a t-test almost no power and an unstable variance estimate, while "every seed moved the
same way, at two model sizes" is six independent draws agreeing.

This analysis was written after seeing the unpaired result, and the report says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

SIZES = (
    ("35M", Path("results/phase2/classifier")),
    ("650M", Path("results/phase2/classifier_650m")),
)
BASELINE = "A0"
ARMS = ("A4", "A5")
SHARDS = ("test_allele", "test_epitope")
OUT = Path("results/phase2/h2_paired")


@dataclass(frozen=True, slots=True)
class Paired:
    size: str
    arm: str
    shard: str
    baseline: list[float]
    treatment: list[float]

    @property
    def diffs(self) -> list[float]:
        return [t - b for t, b in zip(self.treatment, self.baseline, strict=True)]

    @property
    def unanimous(self) -> bool:
        """Every seed moved the same way. With n = 3 this is the load-bearing statistic."""
        return len({d > 0 for d in self.diffs}) == 1

    def row(self) -> dict:
        d = self.diffs
        t, p = stats.ttest_rel(self.treatment, self.baseline)
        return {
            "size": self.size,
            "arm": self.arm,
            "shard": self.shard,
            "n_seeds": len(d),
            "baseline_mean": round(st.mean(self.baseline), 4),
            "treatment_mean": round(st.mean(self.treatment), 4),
            "paired_delta": round(st.mean(d), 4),
            "paired_sd": round(st.stdev(d), 4) if len(d) > 1 else "",
            "unpaired_sd_baseline": round(st.stdev(self.baseline), 4) if len(d) > 1 else "",
            "t": round(float(t), 3),
            "p": round(float(p), 4),
            "unanimous": self.unanimous,
            "seed_deltas": ";".join(f"{x:+.3f}" for x in d),
        }


def _auroc(root: Path, arm: str, shard: str, seeds: range) -> list[float]:
    out = []
    for s in seeds:
        path = root / "runs" / f"{arm}_s{s}" / "metrics.json"
        if path.exists():
            out.append(json.loads(path.read_text())["eval"][shard]["auroc"]["point"])
    return out


def collect(sizes=SIZES, seeds: range = range(3)) -> list[Paired]:
    pairs = []
    for size, root in sizes:
        base = {sh: _auroc(root, BASELINE, sh, seeds) for sh in SHARDS}
        for arm in ARMS:
            for shard in SHARDS:
                treat = _auroc(root, arm, shard, seeds)
                # A partially finished sweep would otherwise pair seed 0 against seed 1.
                if len(treat) == len(base[shard]) and len(treat) > 1:
                    pairs.append(Paired(size, arm, shard, base[shard], treat))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    pairs = collect()
    if not pairs:
        print("no paired runs found")
        return 1
    rows = [p.row() for p in pairs]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "h2_paired.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    # A contrast replicates when both sizes move the same way in every seed.
    replicated = sorted(
        {
            (r["arm"], r["shard"])
            for r in rows
            if all(
                o["unanimous"] and (o["paired_delta"] > 0) == (r["paired_delta"] > 0)
                for o in rows
                if (o["arm"], o["shard"]) == (r["arm"], r["shard"])
            )
            and len({o["size"] for o in rows if (o["arm"], o["shard"]) == (r["arm"], r["shard"])})
            == len(SIZES)
        }
    )
    (args.out / "h2_paired.json").write_text(
        json.dumps({"rows": rows, "replicated_across_sizes": replicated}, indent=2) + "\n"
    )
    for r in rows:
        print(
            f"{r['size']:>5} {r['arm']} {r['shard']:<13} "
            f"{r['paired_delta']:+.3f} (sd {r['paired_sd']}) p={r['p']:.3f} "
            f"unanimous={r['unanimous']} {r['seed_deltas']}"
        )
    print(f"replicated across both sizes: {replicated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
