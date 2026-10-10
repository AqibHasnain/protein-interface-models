"""Metrics over per-example scores. Pure numpy/scipy/sklearn, no model or I/O."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

_MIN_DENOMINATOR = 1e-8
DEFAULT_MIN_BIN_N = 20


def _as_1d(x: object, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {arr.shape}")
    return arr


def _paired(original: object, perturbed: object) -> tuple[np.ndarray, np.ndarray]:
    a, b = _as_1d(original, "original"), _as_1d(perturbed, "perturbed")
    if a.shape != b.shape:
        raise ValueError(f"original and perturbed differ in shape: {a.shape} vs {b.shape}")
    return a, b


def _check_binary(labels: object) -> np.ndarray:
    y = _as_1d(labels, "labels")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0 and 1")
    if y.min() == y.max():
        raise ValueError("labels contain a single class; AUROC/AUPRC are undefined")
    return y


def auroc(labels: object, scores: object) -> float:
    """Area under the ROC curve. Raises ValueError if labels hold a single class."""
    y = _check_binary(labels)
    return float(roc_auc_score(y, _as_1d(scores, "scores")))


def auprc(labels: object, scores: object) -> float:
    """Average precision. Raises ValueError if labels hold a single class."""
    y = _check_binary(labels)
    return float(average_precision_score(y, _as_1d(scores, "scores")))


def nsi(
    original: object,
    perturbed: object,
    binder_scores: object,
    nonbinder_scores: object,
    min_denominator: float = _MIN_DENOMINATOR,
) -> float:
    """Nuisance Sensitivity Index.

    ``median_i |s(x_i) - s(perturb(x_i))| / |median(binders) - median(non-binders)|``

    NSI = 1.0 means a biologically meaningless relayout moves the score as much as the
    entire binder/non-binder signal. Values near 0 mean the score is invariant to the
    nuisance. The denominator is computed on the reference set the caller supplies.

    Raises
    ------
    ValueError
        If the binder/non-binder separation is below ``min_denominator``, since the
        ratio would be meaningless.
    """
    a, b = _paired(original, perturbed)
    sep = abs(
        float(np.median(_as_1d(binder_scores, "binder_scores")))
        - float(np.median(_as_1d(nonbinder_scores, "nonbinder_scores")))
    )
    if sep < min_denominator:
        raise ValueError(
            f"binder/non-binder median separation {sep:.3g} is below {min_denominator:g}; "
            "NSI is undefined"
        )
    return float(np.median(np.abs(a - b)) / sep)


def rank_stability(original: object, perturbed: object) -> dict[str, float]:
    """Spearman rho and Kendall tau (tau-b) between original and perturbed scores."""
    a, b = _paired(original, perturbed)
    rho = stats.spearmanr(a, b)
    tau = stats.kendalltau(a, b)
    return {
        "spearman_rho": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "kendall_tau": float(tau.statistic),
        "kendall_p": float(tau.pvalue),
    }


def delta_auroc(labels: object, original: object, perturbed: object) -> dict[str, float]:
    """AUROC on original and perturbed scores; ``delta = perturbed - original``."""
    a, b = _paired(original, perturbed)
    orig, pert = auroc(labels, a), auroc(labels, b)
    return {"auroc_original": orig, "auroc_perturbed": pert, "delta": pert - orig}


def decision_flip_rate(
    original: object, perturbed: object, threshold: float | None = None
) -> float:
    """Fraction of examples whose binder call changes under perturbation.

    An example is called positive iff its score is strictly greater than the threshold,
    so a score exactly on the threshold is negative. The default threshold is the median
    of ``original``, which makes the original calls roughly balanced (with an odd n and no
    other ties, the median example itself is negative).
    """
    a, b = _paired(original, perturbed)
    t = float(np.median(a)) if threshold is None else threshold
    return float(np.mean((a > t) != (b > t)))


def length_stratified_nsi(
    original: object,
    perturbed: object,
    lengths: object,
    binder_scores: object,
    nonbinder_scores: object,
    bins: Sequence[float],
    min_n: int = DEFAULT_MIN_BIN_N,
) -> dict[str, dict[str, float | int | None]]:
    """NSI per length bucket.

    ``bins`` are ascending edges; bucket k holds ``bins[k] <= length < bins[k+1]``, and
    examples outside the edges are dropped. The binder/non-binder denominator is shared
    across buckets so values are comparable. A bucket with fewer than ``min_n`` examples
    reports ``nsi=None`` because a median over a handful of examples is too noisy to
    compare across buckets.
    """
    a, b = _paired(original, perturbed)
    ln = _as_1d(lengths, "lengths")
    if ln.shape != a.shape:
        raise ValueError("lengths must match original in shape")
    edges = list(bins)
    if len(edges) < 2 or any(hi <= lo for lo, hi in zip(edges, edges[1:], strict=False)):
        raise ValueError("bins must be at least two strictly ascending edges")

    out: dict[str, dict[str, float | int | None]] = {}
    for lo, hi in zip(edges, edges[1:], strict=False):
        mask = (ln >= lo) & (ln < hi)
        n = int(mask.sum())
        value = nsi(a[mask], b[mask], binder_scores, nonbinder_scores) if n >= min_n else None
        out[f"[{lo:g},{hi:g})"] = {"n": n, "nsi": value}
    return out


def bootstrap_ci(
    fn: Callable[..., float],
    *arrays: object,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    paired: bool = True,
) -> tuple[float, float, float]:
    """Percentile bootstrap over examples. Returns ``(point, lo, hi)``.

    With ``paired=True`` every array is indexed by the same resampled indices, preserving
    the per-example pairing of original/perturbed scores. With ``paired=False`` each array
    gets independent indices (for arrays of unrelated populations, e.g. binders vs
    non-binders). The point estimate is ``fn`` on the full data.

    Resamples where ``fn`` raises ValueError (e.g. a single-class AUROC draw) are
    discarded; if more than half fail, the interval is not trustworthy and ValueError is
    raised.
    """
    if not arrays:
        raise ValueError("at least one array is required")
    data = [np.asarray(x) for x in arrays]
    if paired and len({len(d) for d in data}) != 1:
        raise ValueError("paired bootstrap requires arrays of equal length")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    point = float(fn(*data))
    rng = np.random.default_rng(seed)
    sizes = [len(d) for d in data]
    draws: list[float] = []
    for _ in range(n_boot):
        if paired:
            idx = rng.integers(0, sizes[0], size=sizes[0])
            sample = [d[idx] for d in data]
        else:
            sample = [d[rng.integers(0, n, size=n)] for d, n in zip(data, sizes, strict=True)]
        try:
            draws.append(float(fn(*sample)))
        except ValueError:
            continue
    if len(draws) < n_boot / 2:
        raise ValueError(f"{n_boot - len(draws)} of {n_boot} bootstrap resamples were invalid")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
