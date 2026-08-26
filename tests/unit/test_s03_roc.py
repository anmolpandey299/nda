"""A/B/C/G — the canonical fixed-FPR estimator, ties, direction, calibration threshold.

Expected values are derived by hand or from a closed form, never from the function under
test [AUTH: 02 §C1, §C2; 00 §16].
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.scoring.roc import (
    EstimatorError,
    realised_rate,
    roc_points,
    threshold_at_calibration_rate,
    tpr_at_fixed_fpr,
)

FloatArray = NDArray[np.float64]


def _labelled(member: list[float], non_member: list[float]) -> tuple[FloatArray, NDArray[np.int_]]:
    scores = np.array(member + non_member, dtype=np.float64)
    labels = np.array([1] * len(member) + [0] * len(non_member), dtype=np.int_)
    return scores, labels


# ------------------------------------------------------------------ A: estimator
def test_perfect_separation_reaches_full_power() -> None:
    """100 non-members below every member: at 1% FPR one non-member is admitted and all
    members already are, so TPR = 1 exactly."""
    scores, labels = _labelled([10.0] * 50, list(np.arange(100, dtype=float) / 100.0))
    result = tpr_at_fixed_fpr(scores, labels, 0.01)
    assert result.true_positive_rate == pytest.approx(1.0)
    assert result.read_off == "EXACT"


def test_hand_computed_operating_point() -> None:
    """Members 3,4,5; non-members 0,1,2. At FPR = 1/3 the threshold is 2, admitting all
    three members: TPR = 1. Worked out by hand from the definition."""
    scores, labels = _labelled([3.0, 4.0, 5.0], [0.0, 1.0, 2.0])
    result = tpr_at_fixed_fpr(scores, labels, 1.0 / 3.0)
    assert result.read_off == "EXACT"
    assert result.true_positive_rate == pytest.approx(1.0)


def test_interpolation_is_reported_as_interpolation() -> None:
    """With 4 non-members the achievable rates are 0, .25, .5, .75, 1. A target of 0.1 lies
    between (0,0) and (0.25, TPR); the estimator says so instead of pretending exactness."""
    scores, labels = _labelled([1.0, 2.0], [0.0, 0.5, 1.5, 3.0])
    result = tpr_at_fixed_fpr(scores, labels, 0.1)
    assert result.read_off == "INTERPOLATED"
    assert result.lower_point[0] == pytest.approx(0.0)
    assert result.upper_point[0] == pytest.approx(0.25)
    expected = result.lower_point[1] + 0.4 * (result.upper_point[1] - result.lower_point[1])
    assert result.true_positive_rate == pytest.approx(expected)


def test_analytic_gaussian_agreement() -> None:
    """Checked against the closed form, i.e. against mathematics rather than against itself."""
    from src.analysis.dryrun import analytic_true_positive_rate, gaussian_membership_scores

    scores, labels = gaussian_membership_scores(
        n_member=20000, n_non_member=20000, separation=2.0, seed=4242
    )
    observed = tpr_at_fixed_fpr(scores, labels, 0.01).true_positive_rate
    assert observed == pytest.approx(analytic_true_positive_rate(2.0, 0.01), abs=0.02)


# ------------------------------------------------------------------ B: ties
def test_tied_scores_are_admitted_together() -> None:
    """Every observation sits on one value. No threshold can separate the classes, so the
    only achievable points are (0,0) and (1,1) [AUTH: 02 §C2]."""
    scores, labels = _labelled([1.0] * 10, [1.0] * 10)
    curve = roc_points(scores, labels)
    assert curve.false_positive_rate.tolist() == [0.0, 1.0]
    assert curve.true_positive_rate.tolist() == [0.0, 1.0]
    # the interpolated read-off at 1% is 0.01, not the 1.0 an envelope over ties would give
    assert tpr_at_fixed_fpr(scores, labels, 0.01).true_positive_rate == pytest.approx(0.01)


def test_a_tie_group_cannot_manufacture_power() -> None:
    """Members and non-members share the value 5. Ordering members first inside the tie group
    would report TPR = 1 at FPR = 0; the tie-safe construction refuses that."""
    scores, labels = _labelled([5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 0.0])
    curve = roc_points(scores, labels)
    for rate, power in zip(curve.false_positive_rate, curve.true_positive_rate, strict=True):
        assert not (power == 1.0 and rate < 0.75), "an unachievable operating point appeared"


def test_roc_points_only_uses_distinct_values() -> None:
    scores, labels = _labelled([1.0, 1.0, 2.0], [1.0, 3.0, 3.0])
    curve = roc_points(scores, labels)
    assert curve.thresholds.tolist() == [np.inf, 3.0, 2.0, 1.0]


def test_tie_handling_is_order_independent() -> None:
    scores, labels = _labelled([1.0, 2.0, 2.0], [2.0, 0.0])
    first = tpr_at_fixed_fpr(scores, labels, 0.5)
    order = np.array([4, 0, 3, 1, 2])
    second = tpr_at_fixed_fpr(scores[order], labels[order], 0.5)
    assert first.true_positive_rate == second.true_positive_rate


# ------------------------------------------------------------------ C: direction
def test_score_direction_is_higher_equals_member() -> None:
    """Negating the scores must destroy the signal; a direction-agnostic estimator would not
    notice, which is the mutation this test kills."""
    scores, labels = _labelled([3.0, 4.0, 5.0], list(np.arange(100, dtype=float) / 100.0))
    correct = tpr_at_fixed_fpr(scores, labels, 0.01).true_positive_rate
    reversed_direction = tpr_at_fixed_fpr(-scores, labels, 0.01).true_positive_rate
    assert correct == pytest.approx(1.0)
    assert reversed_direction < 0.05


def test_membership_label_direction_is_explicit() -> None:
    scores, labels = _labelled([3.0, 4.0, 5.0], [0.0, 1.0, 2.0])
    swapped = 1 - labels
    assert tpr_at_fixed_fpr(scores, labels, 1 / 3).true_positive_rate == pytest.approx(1.0)
    assert tpr_at_fixed_fpr(scores, swapped, 1 / 3).true_positive_rate < 1.0


# ------------------------------------------------------------------ fail-closed
@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_target_is_refused(target: float) -> None:
    scores, labels = _labelled([1.0], [0.0])
    with pytest.raises(EstimatorError):
        tpr_at_fixed_fpr(scores, labels, target)


def test_one_class_only_is_refused() -> None:
    scores = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(EstimatorError, match="non-member"):
        tpr_at_fixed_fpr(scores, np.array([1, 1], dtype=np.int_), 0.01)
    with pytest.raises(EstimatorError, match="member"):
        tpr_at_fixed_fpr(scores, np.array([0, 0], dtype=np.int_), 0.01)


def test_non_finite_scores_are_refused() -> None:
    scores, labels = _labelled([np.nan], [0.0])
    with pytest.raises(EstimatorError, match="non-finite"):
        tpr_at_fixed_fpr(scores, labels, 0.01)


# ------------------------------------------------------------------ G: calibration threshold
def test_calibration_threshold_admits_at_most_the_target() -> None:
    """100 calibration non-members, target 1%: the threshold is the largest value, admitting
    exactly one, i.e. 1%."""
    calibration = np.arange(100, dtype=np.float64)
    threshold = threshold_at_calibration_rate(calibration, 0.01)
    assert threshold == 99.0
    assert realised_rate(calibration, threshold) == pytest.approx(0.01)


def test_calibration_threshold_never_overshoots_the_target() -> None:
    calibration = np.arange(50, dtype=np.float64)
    threshold = threshold_at_calibration_rate(calibration, 0.01)
    assert realised_rate(calibration, threshold) <= 0.01


def test_calibration_threshold_with_ties_stays_achievable() -> None:
    calibration = np.array([1.0] * 100, dtype=np.float64)
    threshold = threshold_at_calibration_rate(calibration, 0.01)
    assert threshold == np.inf
    assert realised_rate(calibration, threshold) == 0.0


# ------------------------------------------------------------------ B1: attainable point
def _bracketed_population() -> tuple[FloatArray, NDArray[np.int_]]:
    """500 non-members whose only achievable rates around 1% are 0.008 and 0.014.

    Four non-members sit above a tie group of three, so lowering the threshold past the tie
    jumps the false-positive rate from 4/500 to 7/500. Forty members sit exactly on the tie
    value, so the two bracketing points have different power.
    """
    non_member = [10.0, 9.0, 8.0, 7.0] + [6.0] * 3 + [0.0] * 493
    member = [12.0] * 50 + [6.5] * 100 + [6.0] * 40 + [0.0] * 10
    return _labelled(member, non_member)


def test_the_target_rate_is_genuinely_unachievable() -> None:
    scores, labels = _bracketed_population()
    curve = roc_points(scores, labels)
    assert 0.008 in curve.false_positive_rate.tolist()
    assert 0.014 in curve.false_positive_rate.tolist()
    assert 0.01 not in curve.false_positive_rate.tolist()


def test_the_reported_operating_point_is_a_real_threshold() -> None:
    """B1 invariants 4 and 6: the result names a threshold that exists, states the rate it
    actually realises, and the power it actually achieves."""
    scores, labels = _bracketed_population()
    result = tpr_at_fixed_fpr(scores, labels, 0.01)

    assert result.read_off == "INTERPOLATED"
    assert result.threshold in set(scores.tolist()), "the reported threshold does not exist"

    admitted = scores >= result.threshold
    direct_rate = np.count_nonzero(admitted & (labels == 0)) / 500
    direct_power = np.count_nonzero(admitted & (labels == 1)) / 200
    assert result.realised_false_positive_rate == pytest.approx(direct_rate)
    assert result.attainable_true_positive_rate == pytest.approx(direct_power)
    assert result.realised_false_positive_rate <= 0.01, "the attainable point overshot the target"
    # the most powerful threshold at or below 1%: 6.5, realising 4/500 with 150/200 members
    assert result.threshold == 6.5
    assert result.realised_false_positive_rate == pytest.approx(0.008)
    assert result.attainable_true_positive_rate == pytest.approx(0.75)


def test_the_interpolated_endpoint_is_bracketed_by_achievable_points() -> None:
    """The 00 §7.4(7) estimand is the interpolated value; it is labelled as one and its
    neighbours are stated, so it can never be mistaken for a threshold outcome."""
    scores, labels = _bracketed_population()
    result = tpr_at_fixed_fpr(scores, labels, 0.01)

    assert result.lower_point == pytest.approx((0.008, 0.75))
    assert result.upper_point == pytest.approx((0.014, 0.95))
    # hand-computed: 0.75 + (0.010 - 0.008) / (0.014 - 0.008) * (0.95 - 0.75)
    assert result.true_positive_rate == pytest.approx(0.75 + (1.0 / 3.0) * 0.20)
    assert result.lower_point[1] < result.true_positive_rate < result.upper_point[1]

    achievable_powers = set(roc_points(scores, labels).true_positive_rate.tolist())
    assert result.true_positive_rate not in achievable_powers


def test_an_exact_target_reports_itself_as_attainable() -> None:
    """When 1% is achievable the reported point is the threshold itself, not an interpolation."""
    scores, labels = _labelled([5.0] * 100, [4.0] + [0.0] * 99)
    result = tpr_at_fixed_fpr(scores, labels, 0.01)
    assert result.read_off == "EXACT"
    assert result.realised_false_positive_rate == pytest.approx(0.01)
    assert result.attainable_true_positive_rate == result.true_positive_rate
    admitted = scores >= result.threshold
    assert np.count_nonzero(admitted & (labels == 0)) / 100 == pytest.approx(0.01)


def test_a_large_tie_group_keeps_the_attainable_point_honest() -> None:
    """All non-members tied: no threshold admits a fraction between 0 and 1, so the only
    attainable point at or below 1% is the empty prediction set."""
    scores, labels = _labelled([1.0] * 10, [1.0] * 100)
    result = tpr_at_fixed_fpr(scores, labels, 0.01)
    assert result.threshold == np.inf
    assert result.realised_false_positive_rate == 0.0
    assert result.attainable_true_positive_rate == 0.0
    assert result.true_positive_rate == pytest.approx(0.01)


def test_the_attainable_point_never_exceeds_the_target(  # noqa: D103
) -> None:
    from src.analysis.dryrun import gaussian_membership_scores

    for seed in range(990000, 990010):
        scores, labels = gaussian_membership_scores(
            n_member=300, n_non_member=300, separation=1.5, seed=seed, round_to=0.5
        )
        result = tpr_at_fixed_fpr(scores, labels, 0.01)
        assert result.realised_false_positive_rate <= 0.01
        admitted = scores >= result.threshold
        assert np.count_nonzero(admitted & (labels == 0)) / 300 == pytest.approx(
            result.realised_false_positive_rate
        )
