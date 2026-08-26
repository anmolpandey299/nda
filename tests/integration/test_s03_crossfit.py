"""H/I/J — cross-fitting boundaries, deliberate leakage, pooling separation.

Multiple modules, synthetic fixtures, no network [AUTH: 01 §22; 00 §7.4, §15, §34A.2].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.dryrun import gaussian_membership_scores
from src.analysis.settings import scoring_constants
from src.scoring.crossfit import (
    Channel,
    LeakageError,
    Partition,
    calibrate_operational_threshold,
    plan_folds,
    run_outer_crossfit,
    select_channel_on_calibration,
)
from src.scoring.pooled import (
    RIDGE_POOL,
    STOUFFER_MEAN,
    PoolingError,
    calibration_moments,
    select_pooled_method,
    stouffer_mean,
)

#: From resolved config [AUTH: 01 §17].
TARGET = scoring_constants(Path(__file__).resolve().parents[2]).target_fpr


def _population(n: int, separation: float, seed: int) -> tuple[dict[str, float], dict[str, int]]:
    scores, labels = gaussian_membership_scores(
        n_member=n, n_non_member=n, separation=separation, seed=seed
    )
    ids = [f"mem-{i:05d}" for i in range(n)] + [f"non-{i:05d}" for i in range(n)]
    return (
        {record: float(score) for record, score in zip(ids, scores, strict=True)},
        {record: int(label) for record, label in zip(ids, labels, strict=True)},
    )


# ------------------------------------------------------------------ H: overlap guards
def test_overlapping_partition_raises_before_any_metric() -> None:
    with pytest.raises(LeakageError, match="both calibration and evaluation"):
        Partition(calibration_ids=("a", "b"), evaluation_ids=("b", "c"), fold=0)


def test_duplicate_record_ids_are_refused() -> None:
    with pytest.raises(LeakageError, match="duplicate calibration"):
        Partition(calibration_ids=("a", "a"), evaluation_ids=("b",), fold=0)
    with pytest.raises(LeakageError, match="duplicate evaluation"):
        Partition(calibration_ids=("a",), evaluation_ids=("b", "b"), fold=0)


def test_fold_plan_partitions_are_disjoint_and_complete() -> None:
    ids = [f"r{i:04d}" for i in range(200)]
    plan = plan_folds(ids, 5, seed=7)
    seen: set[str] = set()
    for fold in range(5):
        partition = plan.partition(fold)
        assert not set(partition.calibration_ids) & set(partition.evaluation_ids)
        assert len(partition.calibration_ids) + len(partition.evaluation_ids) == len(ids)
        seen |= set(partition.evaluation_ids)
    assert seen == set(ids), "every record is held out exactly once"


def test_fold_plan_is_deterministic_and_refuses_duplicates() -> None:
    ids = [f"r{i:03d}" for i in range(50)]
    assert plan_folds(ids, 5, 3).assignment == plan_folds(ids, 5, 3).assignment
    assert plan_folds(ids, 5, 3).assignment != plan_folds(ids, 5, 4).assignment
    with pytest.raises(LeakageError, match="duplicates"):
        plan_folds([*ids, ids[0]], 5, 3)


# ------------------------------------------------------------------ I: deliberate leakage
def test_evaluation_ids_offered_to_selection_are_refused() -> None:
    """The planted leakage path: a caller hands an evaluation record to a calibration-only
    interface. The controller refuses rather than quietly using it [AUTH: 01 §23]."""
    scores, labels = _population(50, 1.5, seed=11)
    ids = list(scores)
    # bypass the constructor guard on purpose, so the *interfaces* are the thing under test
    partition = Partition.__new__(Partition)
    object.__setattr__(partition, "calibration_ids", tuple(ids[:60]))
    object.__setattr__(partition, "evaluation_ids", tuple(ids[50:60]))
    object.__setattr__(partition, "fold", 0)
    channel = Channel(name="only", scores=scores)
    with pytest.raises(LeakageError, match="calibration-only selection"):
        select_channel_on_calibration([channel], partition, labels, TARGET)
    with pytest.raises(LeakageError, match="threshold calibration"):
        calibrate_operational_threshold(channel, partition, labels, TARGET)


def test_a_leaky_threshold_beats_the_proper_one_and_is_detectable() -> None:
    """DRY-H's H2 in miniature: a threshold estimated on the held-out non-members inflates
    TPR relative to the calibration-only threshold, which is why the calibration-only path
    is enforced [AUTH: 00 §34B.1 DRY-H]."""
    from src.scoring.roc import realised_rate, threshold_at_calibration_rate

    scores, labels = _population(1000, 2.0, seed=21)
    members = [r for r in scores if labels[r] == 1]
    non_members = [r for r in scores if labels[r] == 0]
    calibration_ids = members[:500] + non_members[:500]
    evaluation_ids = members[500:] + non_members[500:]
    # deliberately shift the held-out non-members downward
    shifted = dict(scores)
    for record in evaluation_ids:
        if labels[record] == 0:
            shifted[record] -= 1.5

    calibration_non = np.array(
        [shifted[r] for r in calibration_ids if labels[r] == 0], dtype=np.float64
    )
    evaluation_non = np.array(
        [shifted[r] for r in evaluation_ids if labels[r] == 0], dtype=np.float64
    )
    evaluation_mem = np.array(
        [shifted[r] for r in evaluation_ids if labels[r] == 1], dtype=np.float64
    )
    proper = threshold_at_calibration_rate(calibration_non, TARGET)
    leaky = threshold_at_calibration_rate(evaluation_non, TARGET)
    assert realised_rate(evaluation_mem, leaky) - realised_rate(evaluation_mem, proper) >= 0.05


# ------------------------------------------------------------------ controller end to end
def test_the_controller_selects_the_strong_channel_and_holds_the_operating_point() -> None:
    strong, labels = _population(500, 2.0, seed=31)
    weak = {record: value * 0.05 for record, value in strong.items()}
    plan = plan_folds(list(strong), 5, seed=5)
    outcome = run_outer_crossfit(
        channels=[Channel("STRONG", strong), Channel("WEAK", weak)],
        labels=labels,
        plan=plan,
        target=TARGET,
    )
    assert outcome.selection_counts["STRONG"] >= 4
    assert 0.004 <= outcome.aggregate_realised_false_positive_rate <= 0.016
    assert outcome.true_positive_rate > 0.2


def test_no_held_out_record_influences_its_own_threshold() -> None:
    """Structural check: each fold's threshold is a function of calibration scores only, so
    perturbing a held-out record cannot move it."""
    scores, labels = _population(200, 1.5, seed=41)
    plan = plan_folds(list(scores), 5, seed=9)
    partition = plan.partition(0)
    channel = Channel("c", scores)
    before = calibrate_operational_threshold(channel, partition, labels, TARGET)
    perturbed = dict(scores)
    for record in partition.evaluation_ids:
        perturbed[record] += 50.0
    after = calibrate_operational_threshold(Channel("c", perturbed), partition, labels, TARGET)
    assert before == after


# ------------------------------------------------------------------ J: pooling separation
def test_pooling_moments_come_from_calibration_non_members_only() -> None:
    scores, labels = _population(100, 1.0, seed=51)
    ids = list(scores)
    moments = calibration_moments(scores, ids[:150], labels)
    expected = np.array([scores[r] for r in ids[:150] if labels[r] == 0], dtype=np.float64)
    assert moments.mean == pytest.approx(float(np.mean(expected)))
    assert moments.sd == pytest.approx(float(np.std(expected, ddof=1)))


def test_stouffer_is_the_frozen_root_k_mean() -> None:
    """z_mean = (1/sqrt(k)) * sum z_i, checked against the formula written out by hand."""
    z = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    assert stouffer_mean(z) == pytest.approx(np.array([9.0, 12.0]) / np.sqrt(3.0))


def test_pooled_selection_uses_calibration_performance_only() -> None:
    scores_a, labels = _population(300, 1.2, seed=61)
    scores_b = {r: v + 0.3 for r, v in scores_a.items()}
    ids = list(scores_a)
    plan = plan_folds(ids, 5, seed=13)
    partition = plan.partition(0)
    selection = select_pooled_method(
        descendant_scores=[scores_a, scores_b],
        partition=partition,
        labels=labels,
        target=TARGET,
        regularisation=1.0,
    )
    assert selection.method in {STOUFFER_MEAN, RIDGE_POOL}
    assert set(selection.scores) == set(partition.calibration_ids) | set(partition.evaluation_ids)


def test_pooling_refuses_an_overlapping_partition() -> None:
    scores, labels = _population(50, 1.0, seed=71)
    bad = Partition.__new__(Partition)
    object.__setattr__(bad, "calibration_ids", tuple(list(scores)[:60]))
    object.__setattr__(bad, "evaluation_ids", tuple(list(scores)[50:70]))
    object.__setattr__(bad, "fold", 0)
    with pytest.raises(LeakageError, match="overlap"):
        select_pooled_method(
            descendant_scores=[scores],
            partition=bad,
            labels=labels,
            target=TARGET,
            regularisation=1.0,
        )


def test_pooling_needs_descendants_and_a_usable_spread() -> None:
    with pytest.raises(PoolingError, match="no descendants"):
        stouffer_mean([])
    flat = {f"non-{i}": 1.0 for i in range(10)}
    labels = dict.fromkeys(flat, 0)
    with pytest.raises(PoolingError, match="sd is not positive"):
        calibration_moments(flat, list(flat), labels).standardise(np.array([1.0]))
