"""DRY-A/B/C/E/F and the positive-control calibration [AUTH: 00 §34B.1, §34B.1A; 02 §C5].

DRY-D (spectral-tail) is not implementable in Block B: it needs O3 truncation ladders, i.e.
merge and recovery objects this block is forbidden to build. It belongs to the block that
owns those operators; the machinery it would use — residuals, Min-K parallel readout and the
crossing states — is exercised here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.dryrun import (
    COARSE_RUNGS,
    baseline_curves,
    calibrate_operator_positive_control,
    generate_calibration_ladder,
    generate_monotone_trial,
    generate_operator_observations,
    operator_detection_fraction,
    residuals_against_baseline,
    seed_curves,
)
from src.analysis.isotonic import (
    BEYOND_CALIBRATION,
    crossing_point,
    delta_e50,
    fit_monotone_curve,
    operator_residuals,
    residual_medians,
    seed_balanced_points,
    spearman,
)
from src.analysis.metrics import denominator_stability
from src.analysis.settings import analysis_constants, dry_run_settings, trial_design

#: Everything material comes from the resolved dry-run config, so a config change moves the
#: scenario rather than silently disagreeing with it [AUTH: 01 §17].
REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = dry_run_settings(REPO_ROOT)
ANALYSIS = analysis_constants(REPO_ROOT)
DESIGN = trial_design(REPO_ROOT)
SLOPE = SETTINGS.number("dry_a_slope")
PLANTED_EFFECT = SETTINGS.number("dry_b_operator_effect")


# ------------------------------------------------------------------ DRY-A
def test_dry_a_monotone_magnitude_relationship() -> None:
    """Planted R_priv = 1 - c*e with no operator residual [AUTH: 00 §34B.1A: rho < -0.90]."""
    rows = generate_monotone_trial(DESIGN, 850000, slope=SLOPE, operator_effect={})
    rho = spearman([r.error for r in rows], [r.recovery for r in rows])
    assert rho < -0.90, f"planted monotone relationship not recovered (rho={rho:.4f})"

    medians = residual_medians(
        operator_residuals(
            [(f"{r.cell}:{r.seed}", r.operator, r.seed, r.error, r.recovery) for r in rows],
            seed_curves(rows),
        )
    )
    assert max(abs(value) for value in medians.values()) < 0.05


# ------------------------------------------------------------------ DRY-B + positive control
@pytest.fixture(scope="module")
def positive_control():  # type: ignore[no-untyped-def]
    start = SETTINGS.integer("positive_control_seed_start")
    count = SETTINGS.integer("positive_control_trials")
    return calibrate_operator_positive_control(
        DESIGN,
        list(range(start, start + count)),
        slope=SLOPE,
        planted_effect=PLANTED_EFFECT,
        quantile=SETTINGS.number("positive_control_quantile"),
    )


def test_the_planted_operator_effect_is_not_absorbed_into_its_own_baseline() -> None:
    """The reviewer counterexample: fitting the magnitude curve on the same rows that carry
    the planted residual hides most of it. Independent calibration observations recover it
    [AUTH: 00 §29.1, §29.2]."""
    curves = baseline_curves(generate_calibration_ladder(DESIGN, 831000, slope=SLOPE))
    observations = generate_operator_observations(
        DESIGN, 831500, slope=SLOPE, operator_effect={"O1": PLANTED_EFFECT}
    )
    independent = residuals_against_baseline(observations, curves)["O1"]

    self_fitted = generate_monotone_trial(
        DESIGN, 831000, slope=SLOPE, operator_effect={"O1": PLANTED_EFFECT}
    )
    absorbed = residuals_against_baseline(self_fitted, seed_curves(self_fitted))["O1"]

    assert independent == pytest.approx(PLANTED_EFFECT, abs=0.06)
    assert independent > absorbed, "the independent baseline recovered less than the self-fit"
    assert absorbed < 0.6 * PLANTED_EFFECT, "the self-fit did not actually absorb the effect"


def test_dry_b_positive_control_is_powered(positive_control) -> None:  # type: ignore[no-untyped-def]
    """02 §C5: the threshold is the 2.5th percentile of 200 planted trials, and the power is
    measured on trials OUTSIDE that bank."""
    assert positive_control.n_trials == SETTINGS.integer("positive_control_trials")
    assert positive_control.threshold > 0.0
    # in-bank fraction is arithmetic, not evidence
    expected_in_bank = 1.0 - SETTINGS.number("positive_control_quantile") / 100.0
    assert positive_control.in_bank_fraction == pytest.approx(expected_in_bank, abs=0.01)
    holdout_start = SETTINGS.integer("positive_control_holdout_seed_start")
    holdout_count = SETTINGS.integer("positive_control_holdout_trials")
    out_of_bank = operator_detection_fraction(
        DESIGN,
        list(range(holdout_start, holdout_start + holdout_count)),
        slope=SLOPE,
        planted_effect=PLANTED_EFFECT,
        threshold=positive_control.threshold,
    )
    assert out_of_bank >= 0.95, f"out-of-bank detection was only {out_of_bank:.3f}"


def test_dry_b_a_planted_operator_residual_is_detected(positive_control) -> None:  # type: ignore[no-untyped-def]
    curves = baseline_curves(generate_calibration_ladder(DESIGN, 851000, slope=SLOPE))
    observations = generate_operator_observations(
        DESIGN, 851500, slope=SLOPE, operator_effect={"O1": PLANTED_EFFECT}
    )
    medians = residuals_against_baseline(observations, curves)
    assert medians["O1"] > positive_control.threshold
    assert medians["O1"] > max(abs(medians[o]) for o in ("O2", "O3"))


def test_dry_b_an_unplanted_operator_stays_below_the_threshold(positive_control) -> None:  # type: ignore[no-untyped-def]
    curves = baseline_curves(generate_calibration_ladder(DESIGN, 852000, slope=SLOPE))
    observations = generate_operator_observations(DESIGN, 852500, slope=SLOPE, operator_effect={})
    medians = residuals_against_baseline(observations, curves)
    assert max(abs(value) for value in medians.values()) < positive_control.threshold


# ------------------------------------------------------------------ DRY-C
def test_dry_c_function_privacy_decoupling() -> None:
    """Plant e50_priv != e50_func with the frozen POSITIVE sign and require the estimate to
    recover both the sign and the magnitude [AUTH: 00 §34B.1A: Delta e50 > 0.15]."""
    errors = np.array(COARSE_RUNGS, dtype=np.float64)
    # privacy decays more slowly, so it crosses 0.5 at a LARGER error: Delta e50 > 0
    privacy = fit_monotone_curve(errors, 1.0 - 0.30 * errors)
    functional = fit_monotone_curve(errors, 1.0 - 0.60 * errors)
    priv_point = crossing_point(privacy, ANALYSIS.e50_level, min_support=ANALYSIS.e50_min_support)
    func_point = crossing_point(
        functional, ANALYSIS.e50_level, min_support=ANALYSIS.e50_min_support
    )
    assert priv_point.state == "ESTIMABLE" and func_point.state == "ESTIMABLE"

    combined = delta_e50(priv_point, func_point)
    assert combined.state == "ESTIMABLE"
    assert combined.value is not None
    assert combined.value > 0.15, f"planted positive Delta e50 not recovered: {combined.value}"


def test_dry_c_the_reversed_construction_fails_the_frozen_assertion() -> None:
    """Mutation control: the previous negative-sign construction must NOT satisfy DRY-C."""
    errors = np.array(COARSE_RUNGS, dtype=np.float64)
    privacy = fit_monotone_curve(errors, 1.0 - 0.60 * errors)
    functional = fit_monotone_curve(errors, 1.0 - 0.30 * errors)
    combined = delta_e50(
        crossing_point(privacy, ANALYSIS.e50_level, min_support=ANALYSIS.e50_min_support),
        crossing_point(functional, ANALYSIS.e50_level, min_support=ANALYSIS.e50_min_support),
    )
    assert combined.value is not None
    assert not combined.value > 0.15, "the reversed sign still passed the frozen assertion"


# ------------------------------------------------------------------ DRY-E
def test_dry_e_denominator_instability_is_labelled() -> None:
    """Planted M(A) - M0 near zero disables normalised cross-cell ranking.

    00 §20.8 defines three bands and 00 §34B.1A's DRY-E line names the > 0.30 case
    "DENOMINATOR_UNSTABLE" in prose. The definitional section is followed: 0.20 < RSE <= 0.30
    is the labelled-but-usable band, and RSE > 0.30 removes G_L/G_R from inferential
    cross-cell ranking, which is what DRY-E's "normalized cross-cell ranking disabled"
    requires.
    """
    bands = {
        "unstable_at": ANALYSIS.denominator_unstable_above,
        "unusable_above": ANALYSIS.denominator_unusable_above,
    }
    assert denominator_stability(signal=0.10, standard_error=0.01, **bands) == "STABLE"
    assert (
        denominator_stability(signal=0.004, standard_error=0.001, **bands) == "DENOMINATOR_UNSTABLE"
    )
    # the DRY-E planted case: the denominator is near zero, so normalisation is unusable
    assert (
        denominator_stability(signal=0.004, standard_error=0.0016, **bands)
        == "NORMALISATION_UNUSABLE"
    )


# ------------------------------------------------------------------ DRY-F
def test_dry_f_beyond_calibration_creates_no_residual() -> None:
    """An operator observation at e > 2.0 is labelled and contributes nothing downstream."""
    rows = generate_monotone_trial(DESIGN, 853000, slope=SLOPE, operator_effect={})
    curves = seed_curves(rows)
    residuals = operator_residuals([("beyond", "O1", 101, 2.5, 0.4)], curves)
    assert residuals[0].state == BEYOND_CALIBRATION
    assert residuals[0].value is None
    assert residual_medians(residuals) == {}


# ------------------------------------------------------------------ seed balance
def test_all_thirteen_registered_rungs_survive_aggregation() -> None:
    """Aggregation groups by exact registered rung; it neither merges nor invents one
    [AUTH: 00 §28A.1, §28A.7]."""
    rows = generate_calibration_ladder(DESIGN, 854000, slope=SLOPE)
    balanced = seed_balanced_points(
        [(r.seed, r.error, r.recovery) for r in rows], required_seeds=DESIGN.seeds
    )
    assert len(balanced) == len(COARSE_RUNGS) == 13
    assert [point[0] for point in balanced] == list(COARSE_RUNGS)

    curve = fit_monotone_curve([p[0] for p in balanced], [p[1] for p in balanced])
    assert curve.is_non_increasing()
    assert curve.error.tolist() == list(COARSE_RUNGS)
    assert spearman([p[0] for p in balanced], [p[1] for p in balanced]) < -0.90


def test_a_missing_required_seed_is_an_error_not_a_lighter_rung() -> None:
    rows = [
        (seed, rung, 1.0 - SLOPE * rung)
        for rung in COARSE_RUNGS
        for seed in DESIGN.seeds
        if not (rung == COARSE_RUNGS[0] and seed == DESIGN.seeds[-1])
    ]
    from src.analysis.isotonic import CalibrationError

    with pytest.raises(CalibrationError, match="missing required training seed"):
        seed_balanced_points(rows, required_seeds=DESIGN.seeds)
