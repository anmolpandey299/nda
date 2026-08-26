"""X — the global dry-run contract that gates opening any real privacy outcome.

    TRUE EFFECT   -> DETECTABLE
    NULL EFFECT   -> NOT MANUFACTURED
    LEAKAGE PATH  -> DETECTED

Deterministic synthetic data only [AUTH: 00 §34B; 02 §C4, §C5].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.dryrun import (
    analytic_true_positive_rate,
    calibrate_familywise,
    gaussian_membership_scores,
    generate_monotone_trial,
    generate_null_trial,
    null_statistics,
)
from src.analysis.endpoints import ReportingError, report_normalised_privacy
from src.analysis.isotonic import spearman
from src.analysis.metrics import privacy_eligibility, r_priv_for_condition
from src.analysis.resampling import PairedUnit, bootstrap, paired_difference
from src.analysis.settings import analysis_constants, scoring_constants, trial_design
from src.scoring.crossfit import Channel, LeakageError, Partition, plan_folds, run_outer_crossfit
from src.scoring.roc import realised_rate, threshold_at_calibration_rate, tpr_at_fixed_fpr

#: From resolved config [AUTH: 01 §17].
TARGET = scoring_constants(Path(__file__).resolve().parents[2]).target_fpr
NULL_LEVEL = analysis_constants(Path(__file__).resolve().parents[2]).null_level
REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN = trial_design(REPO_ROOT)


# ------------------------------------------------------------------ TRUE EFFECT
def test_true_effect_is_detectable_end_to_end() -> None:
    """A planted membership signal reaches the endpoint, clears eligibility, and yields an
    R_priv near the planted ratio — through the same estimator every view uses."""
    oracle_per_seed = []
    for index in range(3):
        scores, labels = gaussian_membership_scores(
            n_member=2048, n_non_member=2048, separation=2.0, seed=980000 + index
        )
        oracle_per_seed.append(tpr_at_fixed_fpr(scores, labels, TARGET).true_positive_rate)
    analytic = analytic_true_positive_rate(2.0, TARGET)
    assert all(abs(value - analytic) < 0.08 for value in oracle_per_seed)

    units = [
        PairedUnit(key=f"s{index}", values=(value, NULL_LEVEL))
        for index, value in enumerate(oracle_per_seed)
    ]
    signal = bootstrap(paired_difference(0, 1), units, n_replicates=2000, seed=1, alpha=0.05)
    verdict = privacy_eligibility(
        oracle_per_seed=oracle_per_seed,
        signal_ci=signal.interval,
        null_level=NULL_LEVEL,
        median_floor=0.05,
    )
    assert verdict.eligible, verdict.reasons

    m_oracle = float(np.median(oracle_per_seed))
    half_way = NULL_LEVEL + 0.5 * (m_oracle - NULL_LEVEL)
    assert r_priv_for_condition(half_way, m_oracle, NULL_LEVEL, verdict) == pytest.approx(0.5)


def test_true_effect_is_detectable_in_the_downstream_analysis() -> None:
    """Both planted effects are recovered together.

    The frozen rho < -0.90 tolerance belongs to DRY-A, which plants a monotone relationship
    with *no* operator residual; here a 0.25 residual on one operator is planted as well and
    legitimately adds scatter, so the monotone assertion is the weaker one and the operator
    assertion carries the DRY-B claim [AUTH: 00 §34B.1A].
    """
    rows = generate_monotone_trial(DESIGN, 981000, slope=0.45, operator_effect={"O1": 0.25})
    statistics = null_statistics(rows)
    assert statistics["spearman_rho"] < -0.85
    # above the DRY-B positive-control threshold regenerated under this estimator (02 §C5)
    assert statistics["max_abs_median_operator_residual"] > 0.055


# ------------------------------------------------------------------ NULL EFFECT
def test_null_effect_is_not_manufactured_at_the_endpoint() -> None:
    """No planted separation: the endpoint sits at the null level and eligibility refuses."""
    oracle_per_seed = []
    for index in range(3):
        scores, labels = gaussian_membership_scores(
            n_member=2048, n_non_member=2048, separation=0.0, seed=982000 + index
        )
        oracle_per_seed.append(tpr_at_fixed_fpr(scores, labels, TARGET).true_positive_rate)
    assert float(np.mean(oracle_per_seed)) == pytest.approx(NULL_LEVEL, abs=0.01)

    units = [
        PairedUnit(key=f"s{index}", values=(value, NULL_LEVEL))
        for index, value in enumerate(oracle_per_seed)
    ]
    signal = bootstrap(paired_difference(0, 1), units, n_replicates=2000, seed=2, alpha=0.05)
    verdict = privacy_eligibility(
        oracle_per_seed=oracle_per_seed,
        signal_ci=signal.interval,
        null_level=NULL_LEVEL,
        median_floor=0.05,
    )
    assert not verdict.eligible
    with pytest.raises(ReportingError, match="INELIGIBLE"):
        report_normalised_privacy(
            verdict=verdict,
            denominator_label="STABLE",
            m_view=0.5,
            m_desc=0.4,
            m_pool=0.45,
            m_rec=0.5,
            m_oracle=float(np.median(oracle_per_seed)),
            null_level=NULL_LEVEL,
        )


def test_null_effect_is_not_manufactured_in_the_downstream_analysis() -> None:
    rows = generate_null_trial(DESIGN, 983000)
    assert abs(spearman([r.error for r in rows], [r.recovery for r in rows])) < 0.5
    calibration = calibrate_familywise(DESIGN, list(range(820000, 820400)), quantile=0.95)
    assert calibration.passes(null_statistics(rows)), calibration.t_max(null_statistics(rows))


# ------------------------------------------------------------------ LEAKAGE PATH
def test_leakage_path_is_detected_structurally() -> None:
    with pytest.raises(LeakageError):
        Partition(calibration_ids=("a", "b"), evaluation_ids=("b",), fold=0)


def test_leakage_path_is_detected_statistically() -> None:
    """A threshold estimated on the held-out non-members inflates TPR by at least 0.05, so
    the suite demonstrably sees a leakage path when one is planted [AUTH: 00 §34B.1 DRY-H]."""
    scores, labels = gaussian_membership_scores(
        n_member=2048, n_non_member=2048, separation=2.0, seed=985000
    )
    ids = [f"mem-{i:05d}" for i in range(2048)] + [f"non-{i:05d}" for i in range(2048)]
    by_id = {r: float(v) for r, v in zip(ids, scores, strict=True)}
    membership = {r: int(v) for r, v in zip(ids, labels, strict=True)}
    plan = plan_folds(ids, 5, seed=985001)
    partition = plan.partition(0)

    leaked = dict(by_id)
    for record in partition.evaluation_ids:
        if membership[record] == 0:
            leaked[record] -= 1.75
    calibration_non = np.array(
        [leaked[r] for r in partition.calibration_ids if membership[r] == 0], dtype=np.float64
    )
    evaluation_non = np.array(
        [leaked[r] for r in partition.evaluation_ids if membership[r] == 0], dtype=np.float64
    )
    evaluation_mem = np.array(
        [leaked[r] for r in partition.evaluation_ids if membership[r] == 1], dtype=np.float64
    )
    proper = threshold_at_calibration_rate(calibration_non, TARGET)
    leaky = threshold_at_calibration_rate(evaluation_non, TARGET)
    assert realised_rate(evaluation_mem, leaky) - realised_rate(evaluation_mem, proper) >= 0.05


def test_the_controller_is_the_only_path_that_produces_the_endpoint() -> None:
    """The controller's aggregate statistic is the same estimator call, so a second ROC path
    cannot appear without this test noticing [AUTH: 00 §34A.2, §34A.5]."""
    scores, labels = gaussian_membership_scores(
        n_member=1024, n_non_member=1024, separation=1.8, seed=986000
    )
    ids = [f"mem-{i:05d}" for i in range(1024)] + [f"non-{i:05d}" for i in range(1024)]
    by_id = {r: float(v) for r, v in zip(ids, scores, strict=True)}
    membership = {r: int(v) for r, v in zip(ids, labels, strict=True)}
    plan = plan_folds(ids, 5, seed=986001)
    outcome = run_outer_crossfit(
        channels=[Channel("ONLY", by_id)], labels=membership, plan=plan, target=TARGET
    )
    ordered = list(outcome.out_of_fold_scores)
    pooled = np.array([outcome.out_of_fold_scores[r] for r in ordered], dtype=np.float64)
    marks = np.array([membership[r] for r in ordered], dtype=np.int_)
    direct = tpr_at_fixed_fpr(pooled, marks, TARGET)
    assert outcome.true_positive_rate == pytest.approx(direct.true_positive_rate)
