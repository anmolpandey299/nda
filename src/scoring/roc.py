"""THE canonical tie-safe fixed-FPR estimator. One implementation for the whole repository.

02 §C2 prohibits the individual-record cumulative ROC plus an upper envelope over tied
scores: if equal scalar scores occur in both classes there is no threshold that admits one
member of the tie group and excludes another, so individual-record ordering can invent ROC
points no threshold can reach.

The frozen construction here is therefore:

* operating points exist only at **distinct score values**;
* every observation sharing an exact score is admitted simultaneously;
* the sweep keeps all intermediate points (`drop_intermediate = false`);
* the fixed-FPR read-off is exact when an achievable point sits on the target FPR, and is a
  linear interpolation between the two neighbouring achievable points otherwise.

Interpolation is the estimand, not a convenience. The FINAL CLOSED v1.9 measurement spec
defines the endpoint that way in the endpoint clause itself, and repeats it wherever the
controller step is listed:

    00 §7.4(7)   "the ROC-interpolated TPR at exactly 1% FPR computed as an evaluation
                  statistic from held-out scores"
    00 §7.4      "Cross-view headline comparisons use the common-FPR interpolated
                  TPR@1%FPR; operational TPR/FPR pairs are reported alongside it"
    00 §34A.2(8) "fixed-FPR ROC interpolation"
    00 §34B.1    DRY-H step 6, "ROC-interpolated TPR at exactly 1% FPR"
    02 §C2       "linear interpolation between valid neighbouring ROC points"

What 02 §C2 prohibits is a different thing: an upper envelope built from an arbitrary
ordering *inside* a tie group, which invents points no threshold can reach. That is excluded
by construction here, because thresholds are distinct score values and a tie group is
admitted whole.

So that nothing can attribute a TPR to a threshold that does not exist, every result also
carries the **attainable** operating point: a real threshold, the false-positive rate it
actually realises, and the true-positive rate it actually achieves. The interpolated value is
a curve statistic between two such points and is labelled as one.

Conventions, stated rather than assumed [AUTH: 02 §C1, §C2, §C8; 00 §16, §34A.2]:

MEMBERSHIP LABEL    1 = member (positive), 0 = non-member (negative).
SCORE DIRECTION     higher score = more member-like. A family whose raw score points the
                    other way is oriented once, by the score family, never here.
DECISION RULE       predict member iff score >= threshold.

Every view, arm, bootstrap replicate and dry-run assertion calls `tpr_at_fixed_fpr`. There
is deliberately no second, faster or more convenient variant [AUTH: 00 §34A.2, §34A.5].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Read-off provenance: was the reported point achievable exactly, or interpolated between
#: two achievable neighbours?
ReadOff = Literal["EXACT", "INTERPOLATED"]

_MEMBER: Final = 1


class EstimatorError(ValueError):
    """The inputs cannot produce a defined fixed-FPR operating point."""


@dataclass(frozen=True)
class RocCurve:
    """Achievable operating points, ordered by non-decreasing false-positive rate.

    `thresholds[i]` is the score value at which `false_positive_rate[i]` and
    `true_positive_rate[i]` are attained under `score >= threshold`. The leading point is
    the empty prediction set (0, 0) and carries `+inf`.
    """

    thresholds: FloatArray
    false_positive_rate: FloatArray
    true_positive_rate: FloatArray
    n_member: int
    n_non_member: int

    def __post_init__(self) -> None:
        if not (
            len(self.thresholds) == len(self.false_positive_rate) == len(self.true_positive_rate)
        ):
            raise EstimatorError("ROC arrays disagree in length")


@dataclass(frozen=True)
class FixedFprResult:
    """A fixed-FPR read-off, with the evidence for how it was obtained.

    `true_positive_rate` is the 00 §7.4(7) endpoint: the ROC-interpolated TPR at exactly the
    target false-positive rate. `read_off` says whether an achievable point sat on the target
    (`EXACT`) or the value lies between two of them (`INTERPOLATED`).

    `threshold`, `realised_false_positive_rate` and `attainable_true_positive_rate` describe a
    real operating point — the most powerful threshold whose realised false-positive rate does
    not exceed the target — so a caller always has an achievable pair beside the curve
    statistic and can never mistake the interpolated value for a threshold outcome.
    """

    target_false_positive_rate: float
    true_positive_rate: float
    read_off: ReadOff
    threshold: float
    realised_false_positive_rate: float
    attainable_true_positive_rate: float
    lower_point: tuple[float, float]
    upper_point: tuple[float, float]
    n_member: int
    n_non_member: int


def _validated(scores: FloatArray, labels: NDArray[np.int_]) -> tuple[FloatArray, FloatArray]:
    if scores.ndim != 1 or labels.ndim != 1:
        raise EstimatorError("scores and labels must be one-dimensional")
    if scores.shape != labels.shape:
        raise EstimatorError(f"scores {scores.shape} and labels {labels.shape} disagree")
    if scores.size == 0:
        raise EstimatorError("no observations")
    if not np.isfinite(scores).all():
        raise EstimatorError("scores contain non-finite values")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels <= {0, 1}:
        raise EstimatorError(f"labels must be 0/1, found {sorted(unique_labels)}")
    member = scores[labels == _MEMBER]
    non_member = scores[labels != _MEMBER]
    if member.size == 0:
        raise EstimatorError("no member observations")
    if non_member.size == 0:
        raise EstimatorError("no non-member observations")
    return member.astype(np.float64), non_member.astype(np.float64)


def roc_points(scores: FloatArray, labels: NDArray[np.int_]) -> RocCurve:
    """Tie-safe ROC over distinct score thresholds.

    Ties are structural, not incidental: a score value that occurs in both classes admits
    both classes at once, which is exactly what a real threshold does.
    """
    member, non_member = _validated(scores, labels)
    distinct = np.unique(np.concatenate([member, non_member]))[::-1]

    # counts of observations at or above each distinct threshold, computed by searching the
    # sorted class arrays rather than by walking a per-record ordering.
    member_sorted = np.sort(member)
    non_member_sorted = np.sort(non_member)
    member_at_or_above = member_sorted.size - np.searchsorted(member_sorted, distinct, side="left")
    non_member_at_or_above = non_member_sorted.size - np.searchsorted(
        non_member_sorted, distinct, side="left"
    )

    thresholds = np.concatenate([[np.inf], distinct])
    true_positive = np.concatenate([[0.0], member_at_or_above / member.size])
    false_positive = np.concatenate([[0.0], non_member_at_or_above / non_member.size])
    return RocCurve(
        thresholds=thresholds.astype(np.float64),
        false_positive_rate=false_positive.astype(np.float64),
        true_positive_rate=true_positive.astype(np.float64),
        n_member=int(member.size),
        n_non_member=int(non_member.size),
    )


def tpr_at_fixed_fpr(scores: FloatArray, labels: NDArray[np.int_], target: float) -> FixedFprResult:
    """THE fixed-FPR estimator [AUTH: 02 §C1, §C2; 00 §16].

    `target` is supplied by the caller from resolved config; it is never a source literal
    [AUTH: 01 §17].
    """
    if not 0.0 < target < 1.0:
        raise EstimatorError(f"target false-positive rate {target!r} is not in (0, 1)")
    curve = roc_points(scores, labels)
    rates = curve.false_positive_rate
    powers = curve.true_positive_rate

    # The attainable operating point: the most powerful real threshold whose realised
    # false-positive rate does not exceed the target. This exists for every input because the
    # empty prediction set is always an operating point.
    permitted = np.flatnonzero(rates <= target)
    attainable = int(permitted[np.argmax(powers[permitted])])
    attainable_point = (
        float(curve.thresholds[attainable]),
        float(rates[attainable]),
        float(powers[attainable]),
    )

    exact = np.flatnonzero(rates == target)
    if exact.size:
        # Consecutive achievable points can share a false-positive rate when lowering the
        # threshold admits members only. Every one of them is attainable by a real
        # threshold, so the operating point with the greatest power at that exact rate is
        # reported. This is not an envelope over an arbitrary ordering inside a tie group,
        # which 02 §C2 prohibits: it is a choice among genuinely achievable thresholds.
        index = int(exact[np.argmax(powers[exact])])
        value = float(powers[index])
        return FixedFprResult(
            target_false_positive_rate=target,
            true_positive_rate=value,
            read_off="EXACT",
            threshold=float(curve.thresholds[index]),
            realised_false_positive_rate=float(rates[index]),
            attainable_true_positive_rate=value,
            lower_point=(target, value),
            upper_point=(target, value),
            n_member=curve.n_member,
            n_non_member=curve.n_non_member,
        )

    above = np.flatnonzero(rates > target)
    if above.size == 0:
        raise EstimatorError(
            f"no achievable operating point reaches a false-positive rate of {target}"
        )
    upper = int(above[0])
    lower = upper - 1
    if lower < 0:
        raise EstimatorError("the ROC does not start at the empty prediction set")

    lower_rate, upper_rate = float(rates[lower]), float(rates[upper])
    lower_power, upper_power = float(powers[lower]), float(powers[upper])
    span = upper_rate - lower_rate
    weight = 0.0 if span == 0.0 else (target - lower_rate) / span
    value = lower_power + weight * (upper_power - lower_power)
    return FixedFprResult(
        target_false_positive_rate=target,
        true_positive_rate=float(value),
        read_off="INTERPOLATED",
        threshold=attainable_point[0],
        realised_false_positive_rate=attainable_point[1],
        attainable_true_positive_rate=attainable_point[2],
        lower_point=(lower_rate, lower_power),
        upper_point=(upper_rate, upper_power),
        n_member=curve.n_member,
        n_non_member=curve.n_non_member,
    )


def threshold_at_calibration_rate(non_member_scores: FloatArray, target: float) -> float:
    """The operational threshold, estimated from calibration non-members only.

    Returned as an achievable score value: the smallest calibration score whose admitted
    fraction does not exceed `target`. Evaluation records never enter this computation
    [AUTH: 00 §7.4, §16; 01 §23].
    """
    if not 0.0 < target < 1.0:
        raise EstimatorError(f"target false-positive rate {target!r} is not in (0, 1)")
    if non_member_scores.size == 0:
        raise EstimatorError("no calibration non-members")
    distinct = np.unique(non_member_scores)[::-1]
    ordered = np.sort(non_member_scores)
    admitted = ordered.size - np.searchsorted(ordered, distinct, side="left")
    rate = admitted / ordered.size
    permitted = np.flatnonzero(rate <= target)
    if permitted.size == 0:
        return float(np.inf)
    return float(distinct[int(permitted[-1])])


def realised_rate(scores: FloatArray, threshold: float) -> float:
    """Fraction of `scores` admitted by `score >= threshold`."""
    if scores.size == 0:
        raise EstimatorError("no observations to measure a realised rate on")
    return float(np.count_nonzero(scores >= threshold) / scores.size)
