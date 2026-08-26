"""Pooled-lineage representation. A transform over cached descendant scores, never a model.

00 §34A.3: the pooled view consumes the canonical ScoreTable and does not invoke the model
again. 00 §15.1 fixes arm-matched per-descendant normalisation using calibration moments
only; §15.2 fixes the Stouffer aggregator; §15.3 adds one L2-regularised logistic comparator
whose selection is decided on calibration performance alone.

Pooling is representation-level. Reconstruction is a different object, and the two gaps

    D_L = M_pool - M_desc        D_R = M_rec  - M_pool

only mean what they claim if the two stay separate [AUTH: 00 §20.4, §20.6].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.scoring.crossfit import LeakageError, Partition
from src.scoring.roc import tpr_at_fixed_fpr

FloatArray = NDArray[np.float64]

STOUFFER_MEAN: Final = "STOUFFER_MEAN"
RIDGE_POOL: Final = "RIDGE_POOL"
_MEMBER: Final = 1


class PoolingError(ValueError):
    """A pooled representation cannot be formed from the given descendant scores."""


@dataclass(frozen=True)
class ArmMoments:
    """Calibration mean/sd for one descendant within one statistical arm [AUTH: 00 §15.1]."""

    mean: float
    sd: float

    def standardise(self, values: FloatArray) -> FloatArray:
        if self.sd <= 0.0:
            raise PoolingError("calibration sd is not positive, so z-scores are undefined")
        return (values - self.mean) / self.sd


def calibration_moments(
    scores: Mapping[str, float], calibration_ids: Sequence[str], labels: Mapping[str, int]
) -> ArmMoments:
    """Moments from calibration NON-MEMBERS of the same arm, never from evaluation records.

    Canary scores are never standardised with natural moments and vice versa; the caller
    supplies one arm's ids [AUTH: 00 §15.1].
    """
    values = np.array(
        [scores[r] for r in calibration_ids if labels[r] != _MEMBER], dtype=np.float64
    )
    if values.size < 2:
        raise PoolingError("too few calibration non-members to estimate moments")
    return ArmMoments(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))


def stouffer_mean(z_by_descendant: Sequence[FloatArray]) -> FloatArray:
    """z_mean(x) = (1/sqrt(k)) * sum_i z_i(x) [AUTH: 00 §15.2]. Deterministic, pre-registered."""
    if not z_by_descendant:
        raise PoolingError("no descendants to pool")
    stacked = np.vstack(z_by_descendant)
    return np.asarray(stacked.sum(axis=0) / np.sqrt(stacked.shape[0]), dtype=np.float64)


def fit_ridge_pool(
    features: FloatArray, labels: NDArray[np.int_], regularisation: float
) -> NDArray[np.float64]:
    """L2-regularised logistic pooling, fitted on calibration rows only [AUTH: 00 §15.3].

    `regularisation` is the inverse-strength parameter supplied from resolved config. The
    solver is deterministic; no held-out label reaches this call.
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

    if features.ndim != 2:
        raise PoolingError("pooling features must be two-dimensional")
    if len(set(labels.tolist())) < 2:
        raise PoolingError("calibration rows carry only one class")
    # L2 is expressed as l1_ratio=0 on this scikit-learn line; `penalty="l2"` is deprecated.
    model = LogisticRegression(
        l1_ratio=0.0, C=regularisation, solver="lbfgs", max_iter=1000, random_state=None
    )
    model.fit(features, labels)
    return np.asarray(
        np.concatenate([model.coef_.ravel(), model.intercept_.ravel()]), dtype=np.float64
    )


def apply_ridge_pool(features: FloatArray, coefficients: NDArray[np.float64]) -> FloatArray:
    weights, intercept = coefficients[:-1], coefficients[-1]
    return np.asarray(features @ weights + intercept, dtype=np.float64)


@dataclass(frozen=True)
class PooledSelection:
    """Which aggregator won on calibration, and the pooled scores it produced."""

    method: str
    calibration_true_positive_rate: float
    scores: Mapping[str, float]


def select_pooled_method(
    *,
    descendant_scores: Sequence[Mapping[str, float]],
    partition: Partition,
    labels: Mapping[str, int],
    target: float,
    regularisation: float,
) -> PooledSelection:
    """Fixed Stouffer vs ridge, chosen on calibration performance only [AUTH: 00 §15.3].

    Evaluation records are standardised and scored with the calibration-fitted objects; they
    never contribute a moment, a coefficient or a selection decision.
    """
    if not descendant_scores:
        raise PoolingError("no descendants to pool")
    evaluation = set(partition.evaluation_ids)
    if set(partition.calibration_ids) & evaluation:
        raise LeakageError("calibration ids overlap evaluation ids in pooled selection")

    ordered = list(partition.calibration_ids) + list(partition.evaluation_ids)
    moments = [
        calibration_moments(scores, partition.calibration_ids, labels)
        for scores in descendant_scores
    ]
    z_all = [
        moment.standardise(np.array([scores[r] for r in ordered], dtype=np.float64))
        for scores, moment in zip(descendant_scores, moments, strict=True)
    ]
    n_calibration = len(partition.calibration_ids)

    stouffer_all = stouffer_mean(z_all)
    features_all = np.vstack(z_all).T
    calibration_labels = np.array([labels[r] for r in partition.calibration_ids], dtype=np.int_)
    coefficients = fit_ridge_pool(features_all[:n_calibration], calibration_labels, regularisation)
    ridge_all = apply_ridge_pool(features_all, coefficients)

    def calibration_value(values: FloatArray) -> float:
        return tpr_at_fixed_fpr(
            values[:n_calibration], calibration_labels, target
        ).true_positive_rate

    candidates = {STOUFFER_MEAN: stouffer_all, RIDGE_POOL: ridge_all}
    scored = {name: calibration_value(values) for name, values in candidates.items()}
    # Deterministic tie-break: the pre-registered fixed aggregator wins a tie [AUTH: 00 §15.2].
    method = STOUFFER_MEAN
    if scored[RIDGE_POOL] > scored[STOUFFER_MEAN]:
        method = RIDGE_POOL
    chosen = candidates[method]
    return PooledSelection(
        method=method,
        calibration_true_positive_rate=scored[method],
        scores={record: float(value) for record, value in zip(ordered, chosen, strict=True)},
    )
