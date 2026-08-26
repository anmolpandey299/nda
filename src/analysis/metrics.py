"""Privacy endpoint arithmetic: eligibility, R_priv, R_func, and the paired audit gaps.

Every quantity here consumes M(V) values produced by `src.scoring.roc.tpr_at_fixed_fpr`.
Nothing in this module re-derives an operating point [AUTH: 00 §34A.2; 02 §C2].

Frozen definitions [AUTH: 00 §17, §18, §19, §20.4-§20.8, §23A]:

    M0            null level, supplied from resolved config, never a source literal
    R_priv(V)     (M(V) - M0) / (M(A) - M0), UNCLAMPED
    D_L           M_pool - M_desc          primary within-cell lineage gap
    D_R           M_rec  - M_pool          primary within-cell routing gap
    G_L, G_R      the same differences over (M(A) - M0), secondary and cross-cell only
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

ELIGIBLE: Final = "PRIVACY_ENDPOINT_ELIGIBLE"
INELIGIBLE: Final = "PRIVACY_ENDPOINT_INELIGIBLE"

DenominatorLabel = Literal["STABLE", "DENOMINATOR_UNSTABLE", "NORMALISATION_UNUSABLE"]


class MetricError(ValueError):
    """A metric is undefined for the values given, and will not be approximated."""


@dataclass(frozen=True)
class EligibilityVerdict:
    """The 00 §18 rule, with the reason it fired."""

    status: str
    oracle_median: float
    signal_ci: tuple[float, float]
    reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return self.status == ELIGIBLE


def privacy_eligibility(
    *,
    oracle_per_seed: Sequence[float],
    signal_ci: tuple[float, float],
    null_level: float,
    median_floor: float,
) -> EligibilityVerdict:
    """PRIVACY_ENDPOINT_ELIGIBLE iff both 00 §18 conditions hold.

    1. the bootstrap 95% CI for M(A) - M0 excludes zero on the positive side;
    2. median M(A) across the required seeds is at least `median_floor`.

    `null_level` and `median_floor` come from resolved config [AUTH: 01 §17].
    """
    if not oracle_per_seed:
        raise MetricError("no per-seed oracle values")
    low, high = signal_ci
    if low > high:
        raise MetricError("confidence interval bounds are inverted")
    median = float(np.median(np.asarray(oracle_per_seed, dtype=np.float64)))

    reasons: list[str] = []
    if not low > 0.0:
        reasons.append(
            f"bootstrap 95% CI for M(A) - M0 does not exclude zero on the positive side"
            f" (low={low:.6g})"
        )
    if median < median_floor:
        reasons.append(f"median M(A) = {median:.6g} is below the {median_floor:.6g} floor")
    status = ELIGIBLE if not reasons else INELIGIBLE
    return EligibilityVerdict(
        status=status,
        oracle_median=median,
        signal_ci=(float(low), float(high)),
        reasons=tuple(reasons),
    )


def oracle_signal(m_oracle: float, null_level: float) -> float:
    """Delta_A = M(A) - M0 [AUTH: 00 §17]."""
    return float(m_oracle) - float(null_level)


def _r_priv(m_view: float, m_oracle: float, null_level: float) -> float:
    """PRIVATE algebra: R_priv(V) = (M(V) - M0) / (M(A) - M0), unclamped [AUTH: 00 §19].

    Not part of the public scientific surface: it answers arithmetic, not whether the
    condition may have a normalised result at all. Experiment-facing callers must use the
    guarded reporting APIs in `src.analysis.endpoints` [AUTH: 00 §18.1, §20.8].

    A zero denominator is not divided through and not softened: an ineligible or floor-level
    condition must be represented as such, never as a ratio [AUTH: 00 §18.1]."""
    denominator = oracle_signal(m_oracle, null_level)
    if denominator == 0.0:
        raise MetricError(
            "M(A) - M0 is exactly zero; R_priv is undefined and must not be manufactured"
        )
    return (float(m_view) - float(null_level)) / denominator


def r_priv_for_condition(
    m_view: float, m_oracle: float, null_level: float, verdict: EligibilityVerdict
) -> float:
    """R_priv, refused outright for an ineligible condition [AUTH: 00 §18.1, §19]."""
    if not verdict.eligible:
        raise MetricError(
            "condition is PRIVACY_ENDPOINT_INELIGIBLE; report M(A), its CI and the score"
            f" distributions instead of R_priv ({'; '.join(verdict.reasons)})"
        )
    return _r_priv(m_view, m_oracle, null_level)


def lineage_gap(m_pool: float, m_desc: float) -> float:
    """D_L = M_pool - M_desc — primary within-cell statistic [AUTH: 00 §20.4]."""
    return float(m_pool) - float(m_desc)


def routing_gap(m_rec: float, m_pool: float) -> float:
    """D_R = M_rec - M_pool — primary within-cell statistic [AUTH: 00 §20.6]."""
    return float(m_rec) - float(m_pool)


def _normalised_gap(gap: float, m_oracle: float, null_level: float) -> float:
    """PRIVATE algebra. G_L / G_R = D / (M(A) - M0) [AUTH: 00 §19, §20.5, §23A].

    Not part of the public scientific surface: it answers arithmetic, not whether the
    condition may have a normalised result at all. Experiment-facing callers must use the
    guarded reporting APIs in `src.analysis.endpoints` [AUTH: 00 §18.1, §20.8].

    G_L / G_R = D / (M(A) - M0) — secondary, cross-cell [AUTH: 00 §20.5, §20.7]."""
    denominator = oracle_signal(m_oracle, null_level)
    if denominator == 0.0:
        raise MetricError("M(A) - M0 is exactly zero; the normalised gap is undefined")
    return float(gap) / denominator


def denominator_stability(
    *, signal: float, standard_error: float, unstable_at: float, unusable_above: float
) -> DenominatorLabel:
    """The 00 §20.8 interpretation rule. Thresholds arrive from resolved config."""
    if signal == 0.0:
        return "NORMALISATION_UNUSABLE"
    relative = abs(float(standard_error) / float(signal))
    if relative > unusable_above:
        return "NORMALISATION_UNUSABLE"
    if relative > unstable_at:
        return "DENOMINATOR_UNSTABLE"
    return "STABLE"


@dataclass(frozen=True)
class FunctionalVerdict:
    status: str
    reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return self.status == "FUNCTIONAL_ENDPOINT_ELIGIBLE"


def functional_eligibility(
    *,
    loss_base: float,
    loss_oracle: float,
    signal_ci: tuple[float, float],
    relative_floor: float,
    max_relative_se: float,
    standard_error: float,
) -> FunctionalVerdict:
    """The three 00 §23A conditions for NLL-based functional normalisation."""
    delta = float(loss_base) - float(loss_oracle)
    low, _ = signal_ci
    reasons: list[str] = []
    if not low > 0.0:
        reasons.append("paired bootstrap 95% CI for Delta_L does not exclude zero")
    if loss_base == 0.0 or delta / float(loss_base) < relative_floor:
        reasons.append(f"relative improvement is below {relative_floor:.6g}")
    if delta == 0.0 or abs(standard_error / delta) > max_relative_se:
        reasons.append(f"denominator relative SE exceeds {max_relative_se:.6g}")
    status = "FUNCTIONAL_ENDPOINT_ELIGIBLE" if not reasons else "FUNCTIONAL_ENDPOINT_INELIGIBLE"
    return FunctionalVerdict(status=status, reasons=tuple(reasons))


def _r_func(loss_base: float, loss_view: float, loss_oracle: float) -> float:
    """PRIVATE algebra. R_func(V) = (L(θ0) - L(V)) / (L(θ0) - L(A)) [AUTH: 00 §19, §20.5, §23A].

    Not part of the public scientific surface: it answers arithmetic, not whether the
    condition may have a normalised result at all. Experiment-facing callers must use the
    guarded reporting APIs in `src.analysis.endpoints` [AUTH: 00 §18.1, §20.8].

    R_func(V) = (L(θ0) - L(V)) / (L(θ0) - L(A)) [AUTH: 00 §23A]."""
    denominator = float(loss_base) - float(loss_oracle)
    if denominator == 0.0:
        raise MetricError("L(θ0) - L(A) is exactly zero; R_func is undefined")
    return (float(loss_base) - float(loss_view)) / denominator


#: The public scientific surface of this module. The raw normalisation algebra is private:
#: `_r_priv`, `_normalised_gap` and `_r_func` answer arithmetic, not eligibility, and every
#: experiment-facing normalised result must come from `src.analysis.endpoints`
#: [AUTH: 00 §18.1, §19, §20.5, §20.8, §23A].
__all__ = [
    "ELIGIBLE",
    "INELIGIBLE",
    "EligibilityVerdict",
    "FunctionalVerdict",
    "MetricError",
    "denominator_stability",
    "functional_eligibility",
    "lineage_gap",
    "oracle_signal",
    "privacy_eligibility",
    "r_priv_for_condition",
    "routing_gap",
]
