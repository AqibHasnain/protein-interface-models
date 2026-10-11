import numpy as np
import pytest

from cla.eval.metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    cluster_bootstrap_ci,
    decision_flip_rate,
    delta_auroc,
    length_stratified_nsi,
    nsi,
    nsi_control,
    rank_stability,
)


def test_nsi_worked_example():
    original = [-10.0, -20.0, -30.0, -40.0]
    perturbed = [-12.0, -19.0, -35.0, -41.0]
    # |diffs| = 2, 1, 5, 1 -> sorted 1, 1, 2, 5 -> median (1 + 2) / 2 = 1.5
    # binder median = -12, non-binder median = -24 -> separation 12
    # NSI = 1.5 / 12 = 0.125
    value = nsi(original, perturbed, [-10, -12, -14], [-20, -24, -28])
    assert value == pytest.approx(0.125)


def test_nsi_zero_separation_raises():
    with pytest.raises(ValueError, match="undefined"):
        nsi([1.0, 2.0], [1.5, 2.5], [-5, -5, -5], [-5, -5, -5])


def test_nsi_control_worked_example_and_paired_bootstrap():
    original = np.array([-10.0, -20.0, -30.0, -40.0])
    null = np.array([-11.0, -20.5, -31.0, -40.5])  # |diffs| 1, 0.5, 1, 0.5 -> median 0.75
    control = np.array([-13.0, -22.0, -37.0, -48.0])  # |diffs| 3, 2, 7, 8 -> median 5
    assert nsi_control(original, null, control) == pytest.approx(0.15)
    point, lo, hi = bootstrap_ci(nsi_control, original, null, control, n_boot=200)
    assert point == pytest.approx(0.15)
    assert lo <= hi


def test_nsi_control_zero_denominator_raises():
    with pytest.raises(ValueError, match="undefined"):
        nsi_control([1.0, 2.0], [1.5, 2.5], [1.0, 2.0])


def test_rank_stability_reversed_ranking():
    out = rank_stability([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert out["spearman_rho"] == pytest.approx(-1.0)
    assert out["kendall_tau"] == pytest.approx(-1.0)


def test_delta_auroc_hand_computed():
    labels = [0, 0, 1, 1]
    # Positives (0.35, 0.8) vs negatives (0.1, 0.4): 3 of 4 pairs ordered correctly.
    # The perturbed scores order every pair wrongly.
    out = delta_auroc(labels, [0.1, 0.4, 0.35, 0.8], [0.9, 0.8, 0.1, 0.2])
    assert out == pytest.approx({"auroc_original": 0.75, "auroc_perturbed": 0.0, "delta": -0.75})


@pytest.mark.parametrize("metric", [auroc, auprc])
def test_single_class_labels_raise_clear_error(metric):
    with pytest.raises(ValueError, match="single class"):
        metric([1, 1, 1], [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    ("perturbed", "expected"),
    [
        # Median of original is 3. Positive means strictly > 3, so the tie at 3 is negative.
        # Flips: index 0 (1 -> 4 crosses up) and index 3 (4 -> 2 crosses down);
        # index 2 sits on the threshold in both and does not flip.
        ([4.0, 1.0, 3.0, 2.0, 5.0], 2 / 5),
        # Index 2 moves from the threshold (negative) to 3.5 (positive): a third flip.
        ([4.0, 1.0, 3.5, 2.0, 5.0], 3 / 5),
    ],
)
def test_decision_flip_rate_ties_are_negative(perturbed, expected):
    assert decision_flip_rate([1.0, 2.0, 3.0, 4.0, 5.0], perturbed) == pytest.approx(expected)


def test_bootstrap_deterministic_and_brackets_point():
    x = np.random.default_rng(1).normal(size=40)
    first = bootstrap_ci(np.mean, x, n_boot=500, seed=3)
    assert first == bootstrap_ci(np.mean, x, n_boot=500, seed=3)
    assert first != bootstrap_ci(np.mean, x, n_boot=500, seed=4)
    point, lo, hi = first
    assert point == pytest.approx(x.mean())
    assert lo < point < hi


def test_bootstrap_constant_array_is_zero_width():
    assert bootstrap_ci(np.mean, np.full(20, 2.5), n_boot=200) == (2.5, 2.5, 2.5)


def test_bootstrap_resamples_examples_jointly_when_paired():
    a = np.array([0.0, 10.0, 20.0, 30.0])
    b = a.copy()

    def mean_abs_diff(x, y):
        return float(np.mean(np.abs(x - y)))

    # Identical pairs: any joint resample gives exactly 0. Independent resampling of a and b
    # breaks the pairing and yields positive differences.
    assert bootstrap_ci(mean_abs_diff, a, b, n_boot=200) == (0.0, 0.0, 0.0)
    _, _, hi_unpaired = bootstrap_ci(mean_abs_diff, a, b, n_boot=200, paired=False)
    assert hi_unpaired > 0


def test_bootstrap_discards_single_class_resamples():
    labels = np.array([0] * 9 + [1])
    scores = np.linspace(0, 1, 10)
    point, lo, hi = bootstrap_ci(auroc, labels, scores, n_boot=200)
    assert 0.0 <= lo <= hi <= 1.0
    assert point == pytest.approx(1.0)


def test_length_stratified_nsi_under_populated_bin_is_none():
    rng = np.random.default_rng(0)
    lengths = np.concatenate([np.full(30, 100.0), np.full(3, 300.0)])
    original = rng.normal(size=33)
    perturbed = original + 1.0
    out = length_stratified_nsi(
        original, perturbed, lengths, [0.0, 1.0], [-1.0, -3.0], bins=[0, 200, 400], min_n=10
    )
    # Every |diff| is 1; medians are 0.5 and -2, separation 2.5, so NSI = 1 / 2.5 = 0.4.
    assert out["[0,200)"] == {"n": 30, "nsi": pytest.approx(0.4)}
    assert out["[200,400)"] == {"n": 3, "nsi": None}


def test_cluster_bootstrap_is_wider_when_examples_are_duplicated_within_clusters():
    # Eight clusters of 100 identical examples: only eight independent values exist, but the
    # example-level bootstrap sees 800 and reports a far narrower interval.
    cluster_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    x = np.repeat(cluster_values, 100)
    clusters = np.repeat(np.arange(8), 100)
    point, lo, hi = cluster_bootstrap_ci(np.mean, x, clusters=clusters, n_boot=2000)
    ex_point, ex_lo, ex_hi = bootstrap_ci(np.mean, x, n_boot=2000)
    assert point == ex_point == pytest.approx(3.5)
    assert lo < point < hi
    assert (hi - lo) > 3 * (ex_hi - ex_lo)
    assert (point, lo, hi) == cluster_bootstrap_ci(np.mean, x, clusters=clusters, n_boot=2000)
