"""THE outer cross-fitting controller. Implemented once, used by every arm and view.

00 §7.4 and §34A.2: inside each outer fold, every calibration-only choice — candidate
selection, pooled-method selection, hyperparameters, the operational threshold — is made
from the calibration folds alone, and the held-out fold is then scored with the frozen
choice. No held-out record participates in choosing anything that scores it.

The controller fails closed rather than warning. Overlap between calibration and evaluation,
a duplicated record id, or evaluation labels handed to a calibration-only interface all
raise `LeakageError` before any metric is computed [AUTH: 01 §23; 00 §33; 02 §C1].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.scoring.roc import (
    FixedFprResult,
    threshold_at_calibration_rate,
    tpr_at_fixed_fpr,
)

FloatArray = NDArray[np.float64]
_MEMBER: Final = 1


class LeakageError(RuntimeError):
    """An evaluation record reached a calibration-only decision [AUTH: 01 §23; 00 §33]."""


@dataclass(frozen=True)
class Partition:
    """One outer fold's calibration/evaluation split, validated on construction."""

    calibration_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    fold: int

    def __post_init__(self) -> None:
        calibration = set(self.calibration_ids)
        evaluation = set(self.evaluation_ids)
        if len(calibration) != len(self.calibration_ids):
            raise LeakageError(f"fold {self.fold}: duplicate calibration record id")
        if len(evaluation) != len(self.evaluation_ids):
            raise LeakageError(f"fold {self.fold}: duplicate evaluation record id")
        overlap = sorted(calibration & evaluation)
        if overlap:
            raise LeakageError(
                f"fold {self.fold}: {len(overlap)} record(s) appear in both calibration and"
                f" evaluation, first {overlap[0]!r} [AUTH: 01 §23; 00 §33]"
            )
        if not calibration:
            raise LeakageError(f"fold {self.fold}: no calibration records")
        if not evaluation:
            raise LeakageError(f"fold {self.fold}: no evaluation records")


@dataclass(frozen=True)
class FoldPlan:
    """A fixed fold assignment, generated before scoring [AUTH: 00 §7.4]."""

    assignment: Mapping[str, int]
    n_folds: int

    def partition(self, fold: int) -> Partition:
        if not 0 <= fold < self.n_folds:
            raise LeakageError(f"fold {fold} is outside 0..{self.n_folds - 1}")
        calibration = tuple(r for r, f in self.assignment.items() if f != fold)
        evaluation = tuple(r for r, f in self.assignment.items() if f == fold)
        return Partition(calibration_ids=calibration, evaluation_ids=evaluation, fold=fold)


def plan_folds(record_ids: Sequence[str], n_folds: int, seed: int) -> FoldPlan:
    """Deterministic fixed fold assignment. `seed` and `n_folds` come from config."""
    if n_folds < 2:
        raise LeakageError("cross-fitting needs at least two folds")
    unique = list(dict.fromkeys(record_ids))
    if len(unique) != len(record_ids):
        raise LeakageError("record ids contain duplicates, so folds would be ambiguous")
    order = np.argsort(
        np.array([hash_record(record_id, seed) for record_id in unique]), kind="stable"
    )
    assignment = {unique[int(position)]: index % n_folds for index, position in enumerate(order)}
    return FoldPlan(assignment=assignment, n_folds=n_folds)


def hash_record(record_id: str, seed: int) -> int:
    """Stable per-record ordering key. Not a scientific constant; reproducible anywhere."""
    import hashlib

    digest = hashlib.sha256(f"{seed}|{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class Channel:
    """One selectable attack-score channel over the whole record set."""

    name: str
    scores: Mapping[str, float]


@dataclass(frozen=True)
class FoldOutcome:
    """One fold's calibration-only choices and the counts its held-out records produced."""

    fold: int
    selected_channel: str
    threshold: float
    location: float
    scale: float
    n_member: int
    n_non_member: int
    false_positives: int
    true_positives: int

    @property
    def realised_false_positive_rate(self) -> float:
        return self.false_positives / self.n_non_member

    @property
    def operational_true_positive_rate(self) -> float:
        return self.true_positives / self.n_member


@dataclass(frozen=True)
class CrossFitOutcome:
    """Out-of-fold aggregate plus the per-fold record of what was chosen."""

    per_fold: tuple[FoldOutcome, ...]
    selection_counts: Mapping[str, int]
    fixed_point: FixedFprResult
    out_of_fold_scores: Mapping[str, float]

    @property
    def true_positive_rate(self) -> float:
        """M(V): the common-FPR statistic every headline comparison uses [AUTH: 00 §16]."""
        return self.fixed_point.true_positive_rate

    @property
    def aggregate_realised_false_positive_rate(self) -> float:
        """Pooled counts, not a mean of fold rates.

        Folds are not equal in size, so averaging their rates weights a small fold as heavily
        as a large one and misstates the realised operating point [AUTH: 00 §7.4(7), §16].
        """
        negatives = sum(fold.n_non_member for fold in self.per_fold)
        if negatives == 0:
            raise LeakageError("no held-out non-members to aggregate")
        return sum(fold.false_positives for fold in self.per_fold) / negatives

    @property
    def aggregate_operational_true_positive_rate(self) -> float:
        positives = sum(fold.n_member for fold in self.per_fold)
        if positives == 0:
            raise LeakageError("no held-out members to aggregate")
        return sum(fold.true_positives for fold in self.per_fold) / positives


def _split_by_label(
    ids: Sequence[str], scores: Mapping[str, float], labels: Mapping[str, int]
) -> tuple[FloatArray, FloatArray]:
    member = np.array([scores[r] for r in ids if labels[r] == _MEMBER], dtype=np.float64)
    non_member = np.array([scores[r] for r in ids if labels[r] != _MEMBER], dtype=np.float64)
    return member, non_member


def select_channel_on_calibration(
    channels: Sequence[Channel],
    partition: Partition,
    labels: Mapping[str, int],
    target: float,
) -> str:
    """Calibration-only candidate selection [AUTH: 00 §7.4(2), §15.3].

    Fails closed if an evaluation id is present in the ids offered for selection: that is the
    planted leakage path this controller exists to refuse.
    """
    evaluation = set(partition.evaluation_ids)
    offered = set(partition.calibration_ids)
    intruders = sorted(offered & evaluation)
    if intruders:
        raise LeakageError(
            f"evaluation record(s) offered to calibration-only selection: {intruders[:3]}"
        )
    if not channels:
        raise LeakageError("no candidate channels offered")
    best_name, best_value = channels[0].name, -np.inf
    for channel in channels:
        member, non_member = _split_by_label(partition.calibration_ids, channel.scores, labels)
        if member.size == 0 or non_member.size == 0:
            continue
        scores = np.concatenate([member, non_member])
        marks = np.concatenate(
            [np.ones(member.size, dtype=np.int_), np.zeros(non_member.size, dtype=np.int_)]
        )
        value = tpr_at_fixed_fpr(scores, marks, target).true_positive_rate
        if value > best_value:
            best_name, best_value = channel.name, value
    return best_name


def calibrate_operational_threshold(
    channel: Channel, partition: Partition, labels: Mapping[str, int], target: float
) -> float:
    """Threshold from calibration non-members only [AUTH: 00 §7.4(3), §16]."""
    evaluation = set(partition.evaluation_ids)
    intruders = sorted(set(partition.calibration_ids) & evaluation)
    if intruders:
        raise LeakageError(f"evaluation record(s) reached threshold calibration: {intruders[:3]}")
    _, non_member = _split_by_label(partition.calibration_ids, channel.scores, labels)
    if non_member.size == 0:
        raise LeakageError("no calibration non-members to estimate a threshold from")
    return threshold_at_calibration_rate(non_member, target)


@dataclass(frozen=True)
class CalibrationScale:
    """Location and scale of the selected channel on the calibration folds [AUTH: 00 §15.1]."""

    location: float
    scale: float

    def apply(self, values: FloatArray) -> FloatArray:
        return (values - self.location) / self.scale


def calibration_scale(
    channel: Channel, partition: Partition, labels: Mapping[str, int]
) -> CalibrationScale:
    """Derive the common scale from CALIBRATION non-members only.

    Different folds may select different channels or pooling methods, whose raw scores can
    sit on different scales. Concatenating them would make the pooled ROC a mixture of
    incomparable units, so each fold's held-out scores are mapped onto a common scale first —
    using calibration moments only, never held-out ones [AUTH: 00 §7.4(6), §15.1, §34A.2].

    Degenerate calibration variance fails closed rather than silently dividing: a fold whose
    calibration non-members are all identical carries no scale to map onto.
    """
    _, non_member = _split_by_label(partition.calibration_ids, channel.scores, labels)
    if non_member.size < 2:
        raise LeakageError(
            f"fold {partition.fold}: fewer than two calibration non-members, so no common"
            " scale can be derived"
        )
    location = float(np.mean(non_member))
    scale = float(np.std(non_member, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise LeakageError(
            f"fold {partition.fold}: calibration non-member spread is degenerate"
            f" (sd={scale!r}); refusing to map held-out scores onto an undefined scale"
        )
    return CalibrationScale(location=location, scale=scale)


def run_outer_crossfit(
    *,
    channels: Sequence[Channel],
    labels: Mapping[str, int],
    plan: FoldPlan,
    target: float,
) -> CrossFitOutcome:
    """Fold construction -> calibration-only selection, scale and threshold -> held-out scoring.

    The aggregate fixed-FPR statistic is computed once over the pooled out-of-fold scores
    after each fold's scores are mapped onto the common calibration scale, which is what
    00 §7.4(6) means by aggregating out-of-fold predictions.
    """
    by_name = {channel.name: channel for channel in channels}
    outcomes: list[FoldOutcome] = []
    counts: dict[str, int] = {channel.name: 0 for channel in channels}
    out_of_fold: dict[str, float] = {}
    held_out_scores: list[float] = []
    held_out_labels: list[int] = []

    for fold in range(plan.n_folds):
        partition = plan.partition(fold)
        chosen = select_channel_on_calibration(channels, partition, labels, target)
        counts[chosen] += 1
        channel = by_name[chosen]
        scale = calibration_scale(channel, partition, labels)
        threshold = calibrate_operational_threshold(channel, partition, labels, target)

        member, non_member = _split_by_label(partition.evaluation_ids, channel.scores, labels)
        if non_member.size == 0 or member.size == 0:
            raise LeakageError(f"fold {fold}: held-out fold lacks one of the two classes")
        outcomes.append(
            FoldOutcome(
                fold=fold,
                selected_channel=chosen,
                threshold=threshold,
                location=scale.location,
                scale=scale.scale,
                n_member=int(member.size),
                n_non_member=int(non_member.size),
                false_positives=int(np.count_nonzero(non_member >= threshold)),
                true_positives=int(np.count_nonzero(member >= threshold)),
            )
        )
        for record_id in partition.evaluation_ids:
            normalised = float(
                scale.apply(np.array([channel.scores[record_id]], dtype=np.float64))[0]
            )
            out_of_fold[record_id] = normalised
            held_out_scores.append(normalised)
            held_out_labels.append(labels[record_id])

    fixed_point = tpr_at_fixed_fpr(
        np.array(held_out_scores, dtype=np.float64),
        np.array(held_out_labels, dtype=np.int_),
        target,
    )
    return CrossFitOutcome(
        per_fold=tuple(outcomes),
        selection_counts=counts,
        fixed_point=fixed_point,
        out_of_fold_scores=out_of_fold,
    )


# ----------------------------------------------------------------------------------------
# Natural arm — a permanent holdout, not a rotation [AUTH: 00 §7.4, §33.1]
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NaturalPartitions:
    """The four frozen natural-record partitions [AUTH: 00 §6.2, §7.4, §33.1].

    Unlike the canary arm there is no rotation: an evaluation record is held out permanently
    and never enters fitting, selection or thresholding for any fold, because there are no
    folds. Construction validates that the four sets are pairwise disjoint.
    """

    member_calibration: tuple[str, ...]
    calibration_non_members: tuple[str, ...]
    member_eval: tuple[str, ...]
    eval_non_members: tuple[str, ...]

    def __post_init__(self) -> None:
        named = {
            "NATURAL_MEMBER_CALIBRATION": self.member_calibration,
            "CALIBRATION_NONMEMBERS": self.calibration_non_members,
            "NATURAL_MEMBER_EVAL": self.member_eval,
            "EVAL_NONMEMBERS": self.eval_non_members,
        }
        for name, ids in named.items():
            if len(set(ids)) != len(ids):
                raise LeakageError(f"{name}: duplicate record id")
            if not ids:
                raise LeakageError(f"{name}: empty partition")
        names = list(named)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = sorted(set(named[left]) & set(named[right]))
                if overlap:
                    raise LeakageError(
                        f"{left} and {right} share {len(overlap)} record(s), first"
                        f" {overlap[0]!r}; the natural holdout is permanent"
                        " [AUTH: 00 §7.4, §33.1; 01 §23]"
                    )

    @property
    def calibration_ids(self) -> tuple[str, ...]:
        return self.member_calibration + self.calibration_non_members

    @property
    def evaluation_ids(self) -> tuple[str, ...]:
        return self.member_eval + self.eval_non_members

    def as_partition(self) -> Partition:
        """The calibration-only view the shared selection helpers consume."""
        return Partition(
            calibration_ids=self.calibration_ids,
            evaluation_ids=self.evaluation_ids,
            fold=-1,
        )


@dataclass(frozen=True)
class NaturalArmOutcome:
    selected_channel: str
    threshold: float
    location: float
    scale: float
    fixed_point: FixedFprResult
    n_member: int
    n_non_member: int
    false_positives: int
    true_positives: int
    out_of_sample_scores: Mapping[str, float]

    @property
    def true_positive_rate(self) -> float:
        return self.fixed_point.true_positive_rate

    @property
    def realised_false_positive_rate(self) -> float:
        """Realised FPR is checked on EVAL_NONMEMBERS [AUTH: 00 §7.4]."""
        return self.false_positives / self.n_non_member

    @property
    def operational_true_positive_rate(self) -> float:
        return self.true_positives / self.n_member


def run_natural_arm(
    *,
    channels: Sequence[Channel],
    labels: Mapping[str, int],
    partitions: NaturalPartitions,
    target: float,
) -> NaturalArmOutcome:
    """Fit, select and calibrate on the calibration partitions; evaluate on the eval ones.

    Every calibration-only decision sees `NATURAL_MEMBER_CALIBRATION` and
    `CALIBRATION_NONMEMBERS` and nothing else; the threshold comes from
    `CALIBRATION_NONMEMBERS`; the realised FPR is measured on `EVAL_NONMEMBERS`; and
    M_PRIMARY is the fixed-FPR statistic over the held-out evaluation scores
    [AUTH: 00 §7.4, §16, §33.1].
    """
    partition = partitions.as_partition()
    chosen = select_channel_on_calibration(channels, partition, labels, target)
    channel = {c.name: c for c in channels}[chosen]
    scale = calibration_scale(channel, partition, labels)
    threshold = calibrate_operational_threshold(channel, partition, labels, target)

    member = np.array([channel.scores[r] for r in partitions.member_eval], dtype=np.float64)
    non_member = np.array(
        [channel.scores[r] for r in partitions.eval_non_members], dtype=np.float64
    )
    evaluation_ids = list(partitions.evaluation_ids)
    normalised = scale.apply(
        np.array([channel.scores[r] for r in evaluation_ids], dtype=np.float64)
    )
    fixed_point = tpr_at_fixed_fpr(
        normalised,
        np.array([labels[r] for r in evaluation_ids], dtype=np.int_),
        target,
    )
    return NaturalArmOutcome(
        selected_channel=chosen,
        threshold=threshold,
        location=scale.location,
        scale=scale.scale,
        fixed_point=fixed_point,
        n_member=int(member.size),
        n_non_member=int(non_member.size),
        false_positives=int(np.count_nonzero(non_member >= threshold)),
        true_positives=int(np.count_nonzero(member >= threshold)),
        out_of_sample_scores={
            record: float(value) for record, value in zip(evaluation_ids, normalised, strict=True)
        },
    )
