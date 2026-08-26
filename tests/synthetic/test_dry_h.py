"""DRY-H — cross-fitting controller plus deliberate leakage control [AUTH: 00 §34B.1].

Per-record synthetic scores at approximately the real canary scale: 4,096 candidate records,
about half members, five outer folds, and two channels with known Gaussian separation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.dryrun import analytic_true_positive_rate, gaussian_membership_scores
from src.analysis.settings import scoring_constants
from src.scoring.crossfit import Channel, LeakageError, Partition, plan_folds, run_outer_crossfit
from src.scoring.roc import realised_rate, threshold_at_calibration_rate

N_RECORDS = 4096
STRONG_SEPARATION = 2.0
WEAK_SEPARATION = 0.25
#: From resolved config [AUTH: 01 §17].
TARGET = scoring_constants(Path(__file__).resolve().parents[2]).target_fpr


@pytest.fixture(scope="module")
def population() -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    half = N_RECORDS // 2
    strong_scores, labels = gaussian_membership_scores(
        n_member=half, n_non_member=half, separation=STRONG_SEPARATION, seed=910000
    )
    weak_scores, _ = gaussian_membership_scores(
        n_member=half, n_non_member=half, separation=WEAK_SEPARATION, seed=910001
    )
    ids = [f"mem-{i:05d}" for i in range(half)] + [f"non-{i:05d}" for i in range(half)]
    strong = {r: float(v) for r, v in zip(ids, strong_scores, strict=True)}
    weak = {r: float(v) for r, v in zip(ids, weak_scores, strict=True)}
    membership = {r: int(v) for r, v in zip(ids, labels, strict=True)}
    return strong, weak, membership


# ------------------------------------------------------------------ H1: clean IID controller
def test_h1_clean_controller(population) -> None:  # type: ignore[no-untyped-def]
    strong, weak, labels = population
    plan = plan_folds(list(strong), 5, seed=910100)
    outcome = run_outer_crossfit(
        channels=[Channel("STRONG", strong), Channel("WEAK", weak)],
        labels=labels,
        plan=plan,
        target=TARGET,
    )
    analytic = analytic_true_positive_rate(STRONG_SEPARATION, TARGET)

    assert 0.004 <= outcome.aggregate_realised_false_positive_rate <= 0.016
    assert abs(outcome.true_positive_rate - analytic) <= 0.08
    assert outcome.selection_counts["STRONG"] >= 4


def test_h1_every_fold_records_its_own_operating_point(population) -> None:  # type: ignore[no-untyped-def]
    strong, weak, labels = population
    plan = plan_folds(list(strong), 5, seed=910100)
    outcome = run_outer_crossfit(
        channels=[Channel("STRONG", strong), Channel("WEAK", weak)],
        labels=labels,
        plan=plan,
        target=TARGET,
    )
    assert len(outcome.per_fold) == 5
    for fold in outcome.per_fold:
        assert 0.0 <= fold.realised_false_positive_rate <= 0.05
        assert fold.operational_true_positive_rate > fold.realised_false_positive_rate


# ------------------------------------------------------------------ H2: deliberate leak
def test_h2_deliberate_held_out_calibration_leak(population) -> None:  # type: ignore[no-untyped-def]
    """A held-out non-member distribution shifted away from calibration. The leaky threshold
    must beat the proper one by at least 0.05, which is what proves this suite can see a
    threshold-leakage path when one is deliberately introduced."""
    strong, _, labels = population
    plan = plan_folds(list(strong), 5, seed=910100)
    partition = plan.partition(0)

    shifted = dict(strong)
    for record in partition.evaluation_ids:
        if labels[record] == 0:
            shifted[record] -= 1.75

    calibration_non = np.array(
        [shifted[r] for r in partition.calibration_ids if labels[r] == 0], dtype=np.float64
    )
    evaluation_non = np.array(
        [shifted[r] for r in partition.evaluation_ids if labels[r] == 0], dtype=np.float64
    )
    evaluation_mem = np.array(
        [shifted[r] for r in partition.evaluation_ids if labels[r] == 1], dtype=np.float64
    )
    proper = threshold_at_calibration_rate(calibration_non, TARGET)
    leaky = threshold_at_calibration_rate(evaluation_non, TARGET)
    gain = realised_rate(evaluation_mem, leaky) - realised_rate(evaluation_mem, proper)
    assert gain >= 0.05, f"the planted leakage was not detectable (gain={gain:.4f})"


# ------------------------------------------------------------------ H3: overlap guard
def test_h3_structural_overlap_guard(population) -> None:  # type: ignore[no-untyped-def]
    """A calibration/evaluation split sharing at least one record id must raise LeakageError
    before any metric is computed."""
    strong, _, _ = population
    ids = list(strong)
    with pytest.raises(LeakageError):
        Partition(calibration_ids=tuple(ids[:100]), evaluation_ids=tuple(ids[99:120]), fold=0)
