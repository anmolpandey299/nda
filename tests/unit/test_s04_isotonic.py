"""Q/R/S — isotonic direction, e50 states, operator residuals [AUTH: 00 §29.1, §29.2]."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.isotonic import (
    BEYOND_CALIBRATION,
    CalibrationError,
    MonotoneCurve,
    crossing_point,
    delta_e50,
    fit_monotone_curve,
    operator_residuals,
    pool_adjacent_violators,
    residual_medians,
    rung_weights,
    seed_balanced_points,
    spearman,
)


# ------------------------------------------------------------------ Q: direction
def test_the_fit_is_non_increasing_in_error() -> None:
    """Error is the independent variable and recovery decreases with it. A fit increasing in
    e has the axes reversed, which is the mutation this kills [AUTH: 00 §29.1]."""
    e = np.linspace(0.0, 2.0, 21)
    r = 1.0 - 0.4 * e
    curve = fit_monotone_curve(e, r)
    assert curve.is_non_increasing()
    assert curve.recovery[0] > curve.recovery[-1]
    assert np.allclose(curve.recovery, r, atol=1e-12)


def test_reversed_axes_produce_a_different_object_entirely() -> None:
    """Feeding recovery as the independent variable yields a curve over the wrong domain, so
    a caller who swaps the axes cannot silently get a usable answer [AUTH: 00 §29.1]."""
    e = np.linspace(0.0, 2.0, 21)
    r = 1.0 - 0.4 * e
    correct = fit_monotone_curve(e, r)
    swapped = fit_monotone_curve(r, e)
    assert correct.support == pytest.approx((0.0, 2.0))
    assert swapped.support == pytest.approx((0.2, 1.0))
    assert correct.predict(1.5) == pytest.approx(0.4)
    with pytest.raises(CalibrationError, match=BEYOND_CALIBRATION):
        swapped.predict(1.5)


def test_pava_matches_an_independent_solver() -> None:
    """Checked against scikit-learn's solver, which is a structurally different implementation."""
    from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

    rng = np.random.Generator(np.random.PCG64(17))
    for _ in range(5):
        e = np.sort(rng.random(40))
        r = 1.0 - 0.5 * e + (rng.random(40) - 0.5) * 0.3
        mine = pool_adjacent_violators(r)
        theirs = IsotonicRegression(increasing=False).fit_transform(e, r)
        assert np.allclose(mine, theirs, atol=1e-12)


def test_pava_pools_a_violating_pair() -> None:
    """[3, 1, 2] must become [3, 1.5, 1.5]: the last two violate and pool to their mean."""
    assert pool_adjacent_violators(np.array([3.0, 1.0, 2.0])) == pytest.approx([3.0, 1.5, 1.5])


def test_spearman_sign_follows_the_planted_relationship() -> None:
    e = np.linspace(0.0, 2.0, 30)
    assert spearman(e, 1.0 - 0.4 * e) == pytest.approx(-1.0)
    assert spearman(e, 0.4 * e) == pytest.approx(1.0)
    assert abs(spearman(e, np.ones(30))) == pytest.approx(0.0)


# ------------------------------------------------------------------ support and prediction
def test_prediction_outside_the_support_is_refused() -> None:
    curve = fit_monotone_curve([0.5, 1.0, 1.5], [0.9, 0.6, 0.3])
    assert curve.predict(1.0) == pytest.approx(0.6)
    with pytest.raises(CalibrationError, match=BEYOND_CALIBRATION):
        curve.predict(2.5)
    with pytest.raises(CalibrationError, match=BEYOND_CALIBRATION):
        curve.predict(0.1)


def test_a_curve_needs_at_least_two_points() -> None:
    with pytest.raises(CalibrationError, match="at least two"):
        fit_monotone_curve([1.0], [0.5])


# ------------------------------------------------------------------ R: e50 states
def test_a_clean_crossing_is_estimable_and_hand_checkable() -> None:
    """Recovery runs 1.0 -> 0.0 linearly over e in [0, 2], so it reaches 0.5 at e = 1.0."""
    e = np.linspace(0.0, 2.0, 21)
    curve = fit_monotone_curve(e, 1.0 - 0.5 * e)
    result = crossing_point(curve, 0.5, min_support=4)
    assert result.state == "ESTIMABLE"
    assert result.value == pytest.approx(1.0, abs=1e-9)


def test_a_curve_that_never_falls_to_the_level_has_no_crossing() -> None:
    curve = fit_monotone_curve([0.0, 1.0, 2.0], [0.95, 0.9, 0.85])
    result = crossing_point(curve, 0.5, min_support=2)
    assert result.state == "NO_CROSSING"
    assert result.value is None


def test_a_curve_starting_below_the_level_has_no_crossing() -> None:
    curve = fit_monotone_curve([0.0, 1.0, 2.0], [0.3, 0.2, 0.1])
    assert crossing_point(curve, 0.5, min_support=2).state == "NO_CROSSING"


def test_too_few_points_is_insufficient_support() -> None:
    curve = fit_monotone_curve([0.0, 1.0], [0.9, 0.1])
    result = crossing_point(curve, 0.5, min_support=4)
    assert result.state == "INSUFFICIENT_SUPPORT"
    assert result.value is None


def test_no_e50_is_ever_extrapolated_beyond_the_support() -> None:
    """The curve stops at e = 1.0 while still above 0.5. A linear extrapolation would invent
    e50 ≈ 1.5; the frozen rule returns a state instead [AUTH: 00 §28A.7]."""
    curve = fit_monotone_curve(np.linspace(0.0, 1.0, 11), 1.0 - 0.4 * np.linspace(0.0, 1.0, 11))
    result = crossing_point(curve, 0.5, min_support=4)
    assert result.state == "NO_CROSSING"


def test_delta_e50_needs_both_sides_estimable() -> None:
    e = np.linspace(0.0, 2.0, 21)
    privacy = crossing_point(fit_monotone_curve(e, 1.0 - 0.5 * e), 0.5, min_support=4)
    functional = crossing_point(fit_monotone_curve(e, 1.0 - 0.2 * e), 0.5, min_support=4)
    assert privacy.state == "ESTIMABLE" and functional.state == "NO_CROSSING"
    combined = delta_e50(privacy, functional)
    assert combined.state == "NO_CROSSING"
    assert combined.value is None

    functional_ok = crossing_point(fit_monotone_curve(e, 1.0 - 0.3 * e), 0.5, min_support=4)
    both = delta_e50(privacy, functional_ok)
    assert both.state == "ESTIMABLE"
    assert both.value is not None and both.value < 0.0


# ------------------------------------------------------------------ S: operator residuals
def test_a_residual_is_observation_minus_the_seed_curve() -> None:
    curve = fit_monotone_curve([0.0, 1.0, 2.0], [1.0, 0.5, 0.0], seed=1)
    residuals = operator_residuals([("j", "O1", 1, 1.0, 0.8)], {1: curve})
    assert residuals[0].predicted == pytest.approx(0.5)
    assert residuals[0].value == pytest.approx(0.3)
    assert residuals[0].state == "ESTIMABLE"


def test_an_observation_beyond_calibration_yields_no_residual() -> None:
    """DRY-F: e > the calibrated support produces a label, never an extrapolated residual."""
    curve = fit_monotone_curve([0.0, 1.0, 2.0], [1.0, 0.5, 0.0], seed=1)
    residuals = operator_residuals([("j", "O1", 1, 2.5, 0.8)], {1: curve})
    assert residuals[0].state == BEYOND_CALIBRATION
    assert residuals[0].value is None
    assert residual_medians(residuals) == {}


def test_a_seed_without_a_curve_is_labelled_not_guessed() -> None:
    residuals = operator_residuals([("j", "O1", 9, 1.0, 0.8)], {})
    assert residuals[0].state == "NO_CALIBRATION"
    assert residuals[0].value is None


def test_residual_medians_group_by_operator() -> None:
    curve = MonotoneCurve(
        error=np.array([0.0, 1.0, 2.0]), recovery=np.array([1.0, 0.5, 0.0]), seed=1
    )
    residuals = operator_residuals(
        [
            ("a", "O1", 1, 1.0, 0.7),
            ("b", "O1", 1, 1.0, 0.9),
            ("c", "O2", 1, 1.0, 0.4),
        ],
        {1: curve},
    )
    medians = residual_medians(residuals)
    assert medians["O1"] == pytest.approx(0.3)
    assert medians["O2"] == pytest.approx(-0.1)


# ------------------------------------------------------------------ seed balance / rungs
def test_seed_balance_stops_one_seed_from_dominating() -> None:
    """Seed 1 contributes 20 draws at rung 0.5, seed 2 contributes 1. A plain mean would give
    ~0.95; the seed-balanced mean gives 0.5 [AUTH: 00 §29.1, §32.2]."""
    rows = [(1, 0.5, 1.0)] * 20 + [(2, 0.5, 0.0)]
    balanced = seed_balanced_points(rows)
    assert len(balanced) == 1
    assert balanced[0][0] == pytest.approx(0.5)
    assert balanced[0][1] == pytest.approx(0.5)


def test_aggregation_groups_by_exact_registered_rung() -> None:
    """Rungs are a registry, not a histogram: nothing is merged and nothing is invented
    [AUTH: 00 §28A.1, §28A.7]."""
    rungs = [0.01, 0.02, 0.05, 0.10]
    rows = [(seed, rung, 1.0 - 0.4 * rung) for rung in rungs for seed in (1, 2, 3)]
    balanced = seed_balanced_points(rows, required_seeds=(1, 2, 3))
    assert [point[0] for point in balanced] == rungs
    # 0.01 and 0.02 are distinct rungs and stay distinct
    assert balanced[0][0] != balanced[1][0]


def test_a_missing_required_seed_is_reported() -> None:
    rows = [(1, 0.1, 0.9), (2, 0.1, 0.8), (1, 0.2, 0.7), (2, 0.2, 0.6)]
    seed_balanced_points(rows, required_seeds=(1, 2))
    with pytest.raises(CalibrationError, match="missing required training seed"):
        seed_balanced_points(rows, required_seeds=(1, 2, 3))


def test_seed_balance_needs_rows() -> None:
    with pytest.raises(CalibrationError, match="no rows"):
        seed_balanced_points([])


def test_rung_weights_count_contributing_seeds() -> None:
    rows = [(1, 0.1, 0.9), (2, 0.1, 0.8), (1, 0.2, 0.7)]
    assert rung_weights(rows) == {0.1: 2, 0.2: 1}


# ------------------------------------------------------------------ B6: duplicate x
def test_duplicate_error_coordinates_are_collapsed_with_weights() -> None:
    """The reviewer counterexample. e=[0,0,1], r=[1,0.4,0] must average the duplicate rung to
    0.7 rather than letting arrival order decide, which used to read as e50 = 0."""
    curve = fit_monotone_curve([0.0, 0.0, 1.0], [1.0, 0.4, 0.0])
    assert curve.error.tolist() == [0.0, 1.0]
    assert curve.recovery.tolist() == pytest.approx([0.7, 0.0])

    crossing = crossing_point(curve, 0.5, min_support=2)
    assert crossing.state == "ESTIMABLE"
    assert crossing.value is not None
    assert crossing.value > 0.0, "the duplicate rung produced the old e50 = 0 failure"
    assert crossing.value == pytest.approx(2.0 / 7.0)


def test_weighted_pooling_respects_multiplicity() -> None:
    """Three draws at x=0 averaging 0.9 must outweigh one draw at x=0 of 0.3."""
    curve = fit_monotone_curve([0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.9, 0.8, 0.3, 0.0])
    assert curve.recovery[0] == pytest.approx((1.0 + 0.9 + 0.8 + 0.3) / 4.0)


def test_a_single_distinct_coordinate_fails_closed() -> None:
    with pytest.raises(CalibrationError, match="two distinct error coordinates"):
        fit_monotone_curve([0.5, 0.5, 0.5], [1.0, 0.5, 0.0])
