"""DRY-H4 — finite-sample calibration of the common-FPR estimator [AUTH: 02 §C3].

Synthetic score distributions only. The estimator is checked against a closed form, and then
against a trusted distinct-threshold reference under deliberately tied scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.analysis.dryrun import (
    analytic_true_positive_rate,
    calibrate_fixed_point_estimator,
    gaussian_membership_scores,
)
from src.analysis.settings import dry_run_settings
from src.scoring.roc import roc_points, tpr_at_fixed_fpr

FloatArray = NDArray[np.float64]

#: 02 §C3 fixes the sweep exactly; nothing here is shortened.
#: Every value below comes from the resolved dry-run config [AUTH: 01 §17; 02 §C3].
REPO_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS = dry_run_settings(REPO_ROOT)
TARGET = _SETTINGS.number("h4_target_fpr")
SEPARATION = _SETTINGS.number("h4_separation")
SAMPLE_SIZES = _SETTINGS.integers("h4_sample_sizes")
N_SIMULATIONS = _SETTINGS.integer("h4_simulations")


@pytest.fixture(scope="module")
def settings(repo_root: Path):  # type: ignore[no-untyped-def]
    return dry_run_settings(repo_root)


@pytest.fixture(scope="module")
def calibration(settings):  # type: ignore[no-untyped-def]
    return calibrate_fixed_point_estimator(
        sample_sizes=settings.integers("h4_sample_sizes"),
        n_simulations=settings.integer("h4_simulations"),
        separation=settings.number("h4_separation"),
        target=settings.number("h4_target_fpr"),
        seed_start=settings.integer("h4_seed_start"),
    )


def test_the_sweep_is_the_frozen_one(settings, calibration) -> None:  # type: ignore[no-untyped-def]
    """02 §C3: five sample sizes, 1,000 fixed simulation seeds each, matched member counts."""
    assert SAMPLE_SIZES == (200, 400, 800, 2048, 5000)
    assert N_SIMULATIONS == 1000
    assert tuple(row.n_non_member for row in calibration) == SAMPLE_SIZES
    assert all(row.n_simulations == N_SIMULATIONS for row in calibration)


def test_every_size_records_the_required_statistics(calibration) -> None:  # type: ignore[no-untyped-def]
    for row in calibration:
        assert row.signed_bias == row.signed_bias  # finite
        assert row.absolute_error >= abs(row.signed_bias)
        assert row.rmse >= row.absolute_error * 0.5
        low, median, high = row.percentiles
        assert low <= median <= high


# ------------------------------------------------------------------ H4A: continuous scores
def test_h4a_bias_shrinks_with_sample_size(calibration) -> None:  # type: ignore[no-untyped-def]
    """The estimator must not manufacture excess power, and its error must fall as n grows."""
    by_size = {row.n_non_member: row for row in calibration}
    assert by_size[5000].rmse < by_size[2048].rmse < by_size[200].rmse
    assert abs(by_size[5000].signed_bias) < abs(by_size[200].signed_bias)
    for row in calibration:
        assert abs(row.signed_bias) < 0.10, f"n={row.n_non_member} bias {row.signed_bias}"


def test_h4a_the_envelope_brackets_the_analytic_value(calibration) -> None:  # type: ignore[no-untyped-def]
    analytic = analytic_true_positive_rate(SEPARATION, TARGET)
    for row in calibration:
        low, _, high = row.percentiles
        assert low <= analytic <= high, f"n={row.n_non_member} envelope excludes the truth"


def test_h4a_a_null_separation_does_not_manufacture_power() -> None:
    """With no planted signal the estimator must sit at the null level, not above it."""
    estimates = []
    for index in range(60):
        scores, labels = gaussian_membership_scores(
            n_member=2048, n_non_member=2048, separation=0.0, seed=940000 + index
        )
        estimates.append(tpr_at_fixed_fpr(scores, labels, TARGET).true_positive_rate)
    assert float(np.mean(estimates)) == pytest.approx(TARGET, abs=0.006)


# ------------------------------------------------------------------ H4B: tied-score stress
def test_h4b_discretised_scores_agree_with_a_reference_construction() -> None:
    """Rounding to a coarse grid creates exact ties in both classes. The production estimator
    must agree with a trusted distinct-threshold reference built independently here."""
    scores, labels = gaussian_membership_scores(
        n_member=1024, n_non_member=1024, separation=SEPARATION, seed=950000, round_to=0.25
    )
    assert len(np.unique(scores)) < scores.size, "the stress case produced no ties"

    curve = roc_points(scores, labels)
    reference = _reference_curve(scores, labels)
    assert np.allclose(curve.false_positive_rate, reference[0])
    assert np.allclose(curve.true_positive_rate, reference[1])


def test_h4b_no_unachievable_point_is_produced() -> None:
    """Every reported point must be attainable by some threshold applied to all records."""
    scores, labels = gaussian_membership_scores(
        n_member=512, n_non_member=512, separation=1.5, seed=950001, round_to=0.5
    )
    n_member = int(np.count_nonzero(labels == 1))
    n_non_member = int(np.count_nonzero(labels == 0))
    curve = roc_points(scores, labels)
    for threshold, rate, power in zip(
        curve.thresholds, curve.false_positive_rate, curve.true_positive_rate, strict=True
    ):
        admitted = scores >= threshold
        assert np.count_nonzero(admitted & (labels == 0)) / n_non_member == pytest.approx(rate)
        assert np.count_nonzero(admitted & (labels == 1)) / n_member == pytest.approx(power)


def test_h4b_the_tied_read_off_is_reproducible() -> None:
    scores, labels = gaussian_membership_scores(
        n_member=512, n_non_member=512, separation=1.5, seed=950002, round_to=0.5
    )
    first = tpr_at_fixed_fpr(scores, labels, TARGET)
    order = np.argsort(scores, kind="stable")
    second = tpr_at_fixed_fpr(scores[order], labels[order], TARGET)
    assert first.true_positive_rate == second.true_positive_rate
    assert first.read_off == second.read_off


def _reference_curve(scores: FloatArray, labels: NDArray[np.int_]) -> tuple[FloatArray, FloatArray]:
    """A deliberately naive, independently written distinct-threshold sweep."""
    values = sorted(set(scores.tolist()), reverse=True)
    n_member = int(np.count_nonzero(labels == 1))
    n_non_member = int(np.count_nonzero(labels == 0))
    rates = [0.0]
    powers = [0.0]
    for value in values:
        admitted = scores >= value
        rates.append(float(np.count_nonzero(admitted & (labels == 0)) / n_non_member))
        powers.append(float(np.count_nonzero(admitted & (labels == 1)) / n_member))
    return np.array(rates), np.array(powers)
