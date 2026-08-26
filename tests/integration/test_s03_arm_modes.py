"""B2/B3/B10 — natural permanent holdout, cross-fold common scale, count aggregation.

[AUTH: 00 §7.4, §15.1, §16, §33.1, §34A.2; 01 §23]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.dryrun import gaussian_membership_scores
from src.analysis.settings import scoring_constants
from src.scoring.crossfit import (
    Channel,
    FoldOutcome,
    LeakageError,
    NaturalPartitions,
    calibration_scale,
    plan_folds,
    run_natural_arm,
    run_outer_crossfit,
)

#: From resolved config [AUTH: 01 §17].
TARGET = scoring_constants(Path(__file__).resolve().parents[2]).target_fpr


def _population(n: int, separation: float, seed: int) -> tuple[dict[str, float], dict[str, int]]:
    scores, labels = gaussian_membership_scores(
        n_member=n, n_non_member=n, separation=separation, seed=seed
    )
    ids = [f"mem-{i:05d}" for i in range(n)] + [f"non-{i:05d}" for i in range(n)]
    return (
        {r: float(v) for r, v in zip(ids, scores, strict=True)},
        {r: int(v) for r, v in zip(ids, labels, strict=True)},
    )


def _natural(n: int, separation: float, seed: int):  # type: ignore[no-untyped-def]
    scores, labels = _population(n, separation, seed)
    members = [r for r in scores if labels[r] == 1]
    non_members = [r for r in scores if labels[r] == 0]
    half = n // 2
    partitions = NaturalPartitions(
        member_calibration=tuple(members[:half]),
        calibration_non_members=tuple(non_members[:half]),
        member_eval=tuple(members[half:]),
        eval_non_members=tuple(non_members[half:]),
    )
    return scores, labels, partitions


# ================================================================== B2: natural holdout
def test_the_four_natural_partitions_are_pairwise_disjoint() -> None:
    _, _, partitions = _natural(400, 1.5, 11)
    everything = (
        partitions.member_calibration
        + partitions.calibration_non_members
        + partitions.member_eval
        + partitions.eval_non_members
    )
    assert len(set(everything)) == len(everything)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("member_calibration", "member_eval"),
        ("calibration_non_members", "eval_non_members"),
        ("member_calibration", "eval_non_members"),
    ],
)
def test_any_overlap_between_natural_partitions_is_refused(left: str, right: str) -> None:
    _, _, base = _natural(400, 1.5, 12)
    fields = {
        "member_calibration": list(base.member_calibration),
        "calibration_non_members": list(base.calibration_non_members),
        "member_eval": list(base.member_eval),
        "eval_non_members": list(base.eval_non_members),
    }
    fields[right] = [fields[left][0], *fields[right]]
    with pytest.raises(LeakageError, match="permanent"):
        NaturalPartitions(
            member_calibration=tuple(fields["member_calibration"]),
            calibration_non_members=tuple(fields["calibration_non_members"]),
            member_eval=tuple(fields["member_eval"]),
            eval_non_members=tuple(fields["eval_non_members"]),
        )


def test_a_natural_evaluation_record_never_enters_fitting() -> None:
    """There are no folds to rotate into: the calibration view of the arm contains exactly
    the two calibration partitions [AUTH: 00 §7.4, §33.1]."""
    _, _, partitions = _natural(400, 1.5, 13)
    partition = partitions.as_partition()
    assert set(partition.calibration_ids) == set(
        partitions.member_calibration + partitions.calibration_non_members
    )
    assert not set(partition.calibration_ids) & set(partitions.evaluation_ids)


def test_perturbing_evaluation_records_cannot_move_the_selection_or_threshold() -> None:
    """The planted leakage test: shift every held-out record and nothing chosen may change."""
    scores, labels, partitions = _natural(600, 1.6, 14)
    weak = {r: v * 0.05 for r, v in scores.items()}
    channels = [Channel("STRONG", scores), Channel("WEAK", weak)]
    before = run_natural_arm(channels=channels, labels=labels, partitions=partitions, target=TARGET)

    perturbed = dict(scores)
    for record in partitions.evaluation_ids:
        perturbed[record] += 50.0
    after = run_natural_arm(
        channels=[Channel("STRONG", perturbed), Channel("WEAK", weak)],
        labels=labels,
        partitions=partitions,
        target=TARGET,
    )
    assert after.selected_channel == before.selected_channel
    assert after.threshold == before.threshold
    assert after.location == before.location
    assert after.scale == before.scale


def test_the_natural_arm_reports_the_required_pair() -> None:
    """Realised FPR on EVAL_NONMEMBERS, M_PRIMARY on held-out scores [AUTH: 00 §7.4, §16]."""
    scores, labels, partitions = _natural(800, 2.0, 15)
    outcome = run_natural_arm(
        channels=[Channel("ONLY", scores)], labels=labels, partitions=partitions, target=TARGET
    )
    assert outcome.n_non_member == len(partitions.eval_non_members)
    assert outcome.n_member == len(partitions.member_eval)
    direct = np.count_nonzero(
        np.array([scores[r] for r in partitions.eval_non_members]) >= outcome.threshold
    )
    assert outcome.false_positives == direct
    assert outcome.realised_false_positive_rate == pytest.approx(
        direct / len(partitions.eval_non_members)
    )
    assert 0.0 < outcome.true_positive_rate <= 1.0
    assert set(outcome.out_of_sample_scores) == set(partitions.evaluation_ids)


# ================================================================== B3: common score scale
def test_the_common_scale_comes_from_calibration_non_members_only() -> None:
    scores, labels = _population(300, 1.5, 21)
    plan = plan_folds(list(scores), 5, seed=21)
    partition = plan.partition(0)
    scale = calibration_scale(Channel("c", scores), partition, labels)
    expected = np.array(
        [scores[r] for r in partition.calibration_ids if labels[r] == 0], dtype=np.float64
    )
    assert scale.location == pytest.approx(float(np.mean(expected)))
    assert scale.scale == pytest.approx(float(np.std(expected, ddof=1)))


def test_a_degenerate_calibration_spread_fails_closed() -> None:
    ids = [f"non-{i:03d}" for i in range(20)] + [f"mem-{i:03d}" for i in range(20)]
    flat = {r: (1.0 if r.startswith("non-") else 2.0) for r in ids}
    labels = {r: (0 if r.startswith("non-") else 1) for r in ids}
    plan = plan_folds(ids, 4, seed=22)
    with pytest.raises(LeakageError, match="degenerate"):
        calibration_scale(Channel("c", flat), plan.partition(0), labels)


def test_an_affine_shift_in_one_fold_cannot_move_the_pooled_endpoint() -> None:
    """The reviewer counterexample: one fold's selected scores are rescaled and relocated but
    keep their ranking, so the information is identical. After calibration normalisation the
    pooled M_PRIMARY must be unchanged [AUTH: 00 §7.4(6), §15.1]."""
    scores, labels = _population(1000, 1.8, 23)
    plan = plan_folds(list(scores), 5, seed=23)
    baseline = run_outer_crossfit(
        channels=[Channel("ONLY", scores)], labels=labels, plan=plan, target=TARGET
    )

    shifted = dict(scores)
    for record in plan.partition(2).evaluation_ids:
        shifted[record] = shifted[record] * 7.0 + 13.0
    for record in plan.partition(2).calibration_ids:
        shifted[record] = shifted[record] * 7.0 + 13.0
    moved = run_outer_crossfit(
        channels=[Channel("ONLY", shifted)], labels=labels, plan=plan, target=TARGET
    )
    assert moved.true_positive_rate == pytest.approx(baseline.true_positive_rate, abs=1e-9)


def test_raw_concatenation_would_have_moved_the_endpoint() -> None:
    """Mutation control: without normalisation the same affine shift changes the answer, so
    the invariance above is doing real work."""
    scores, labels = _population(1000, 1.8, 23)
    plan = plan_folds(list(scores), 5, seed=23)
    ids = list(scores)
    shifted = dict(scores)
    for record in plan.partition(2).evaluation_ids:
        shifted[record] = shifted[record] * 7.0 + 13.0

    from src.scoring.roc import tpr_at_fixed_fpr

    marks = np.array([labels[r] for r in ids], dtype=np.int_)
    before = tpr_at_fixed_fpr(
        np.array([scores[r] for r in ids], dtype=np.float64), marks, TARGET
    ).true_positive_rate
    after = tpr_at_fixed_fpr(
        np.array([shifted[r] for r in ids], dtype=np.float64), marks, TARGET
    ).true_positive_rate
    assert after != pytest.approx(before, abs=1e-9)


# ================================================================== B10: count aggregation
def test_aggregate_rates_pool_counts_not_fold_rates() -> None:
    """negatives=[1,1,1,1,6], FP=[1,0,0,0,0] -> 1/10 = 0.10, not mean(rates) = 0.20."""
    folds = tuple(
        FoldOutcome(
            fold=index,
            selected_channel="ONLY",
            threshold=0.0,
            location=0.0,
            scale=1.0,
            n_member=n,
            n_non_member=n,
            false_positives=fp,
            true_positives=fp,
        )
        for index, (n, fp) in enumerate(zip([1, 1, 1, 1, 6], [1, 0, 0, 0, 0], strict=True))
    )
    from src.scoring.crossfit import CrossFitOutcome
    from src.scoring.roc import tpr_at_fixed_fpr

    point = tpr_at_fixed_fpr(
        np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float64),
        np.array([1, 1, 0, 0], dtype=np.int_),
        0.5,
    )
    outcome = CrossFitOutcome(
        per_fold=folds, selection_counts={"ONLY": 5}, fixed_point=point, out_of_fold_scores={}
    )
    assert outcome.aggregate_realised_false_positive_rate == pytest.approx(0.10)
    assert float(np.mean([f.realised_false_positive_rate for f in folds])) == pytest.approx(0.20)
    assert outcome.aggregate_operational_true_positive_rate == pytest.approx(0.10)


def test_the_controller_reports_both_operational_rates() -> None:
    scores, labels = _population(600, 2.0, 24)
    plan = plan_folds(list(scores), 5, seed=24)
    outcome = run_outer_crossfit(
        channels=[Channel("ONLY", scores)], labels=labels, plan=plan, target=TARGET
    )
    negatives = sum(f.n_non_member for f in outcome.per_fold)
    positives = sum(f.n_member for f in outcome.per_fold)
    assert negatives == 600 and positives == 600
    assert outcome.aggregate_realised_false_positive_rate == pytest.approx(
        sum(f.false_positives for f in outcome.per_fold) / negatives
    )
    assert outcome.aggregate_operational_true_positive_rate == pytest.approx(
        sum(f.true_positives for f in outcome.per_fold) / positives
    )
    assert outcome.true_positive_rate > outcome.aggregate_realised_false_positive_rate
