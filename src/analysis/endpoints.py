"""The reporting/inference boundary: eligibility-gated results and endpoint-aware bootstrap.

Two things live here rather than in `metrics.py`, which stays pure algebra:

* **Endpoint-aware resampling.** D_L and D_R are differences of *fixed-FPR endpoints*, not
  differences of record-level means. A bootstrap over scalar means answers a different
  question and can disagree in sign, so every replicate re-derives M via the canonical
  estimator on the resampled records [AUTH: 00 §20.4, §20.6, §32.1, §32.3].

* **Eligibility gating.** 00 §18.1 forbids assigning a normalised recovery to an ineligible
  condition. A caller must not be able to reach R_priv, R_func, G_L or G_R without passing
  the verdict that authorises it, so the reporting API takes the verdict and refuses.

Nothing here implements an operating point. Every endpoint comes from
`src.scoring.roc.tpr_at_fixed_fpr` [AUTH: 00 §34A.2, §34A.5].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.analysis.metrics import (
    ELIGIBLE,
    EligibilityVerdict,
    FunctionalVerdict,
    MetricError,
    _normalised_gap,
    _r_func,
    _r_priv,
    lineage_gap,
    oracle_signal,
    routing_gap,
)
from src.analysis.resampling import BootstrapResult, bootstrap
from src.scoring.roc import tpr_at_fixed_fpr

FloatArray = NDArray[np.float64]


class ReportingError(RuntimeError):
    """A normalised scientific result was requested for a condition that may not have one."""


@dataclass(frozen=True)
class ViewRecord:
    """One record's score under every view, plus its membership label.

    All views travel in one unit, so a replicate can never pair the descendant's record i
    with the pooled view's record j [AUTH: 00 §32.3].
    """

    record_id: str
    membership_label: int
    scores: Mapping[str, float]


def endpoint(records: Sequence[ViewRecord], view: str, target: float) -> float:
    """M(view) over `records`, through the canonical estimator."""
    missing = [record.record_id for record in records if view not in record.scores]
    if missing:
        raise MetricError(f"{len(missing)} record(s) carry no {view!r} score")
    return tpr_at_fixed_fpr(
        np.array([record.scores[view] for record in records], dtype=np.float64),
        np.array([record.membership_label for record in records], dtype=np.int_),
        target,
    ).true_positive_rate


def endpoint_difference(
    records: Sequence[ViewRecord], *, left: str, right: str, target: float
) -> float:
    """M(left) - M(right), both re-derived on the same resampled records."""
    return endpoint(records, left, target) - endpoint(records, right, target)


def bootstrap_endpoint_gap(
    records: Sequence[ViewRecord],
    *,
    left: str,
    right: str,
    target: float,
    n_replicates: int,
    seed: int,
    alpha: float,
) -> BootstrapResult:
    """Paired bootstrap of a fixed-FPR endpoint gap [AUTH: 00 §20.4, §20.6, §32.1, §32.3]."""

    def statistic(sampled: Sequence[ViewRecord]) -> float:
        return endpoint_difference(sampled, left=left, right=right, target=target)

    return bootstrap(statistic, records, n_replicates=n_replicates, seed=seed, alpha=alpha)


def _bootstrap_normalised_gap_unguarded(
    records: Sequence[ViewRecord],
    *,
    left: str,
    right: str,
    oracle_view: str,
    target: float,
    null_level: float,
    n_replicates: int,
    seed: int,
    alpha: float,
) -> BootstrapResult:
    """PRIVATE arithmetic. Not an evidentiary endpoint; not exported.

    Every experiment-facing caller must go through `bootstrap_normalised_gap`, which requires
    the eligibility verdict and denominator state 00 §18.1 and §20.8 make mandatory. This
    exists only so the guarded function has one implementation to call.
    """

    def statistic(sampled: Sequence[ViewRecord]) -> float:
        gap = endpoint_difference(sampled, left=left, right=right, target=target)
        denominator = oracle_signal(endpoint(sampled, oracle_view, target), null_level)
        if denominator == 0.0:
            return float("nan")
        return gap / denominator

    return bootstrap(statistic, records, n_replicates=n_replicates, seed=seed, alpha=alpha)


def _require_normalisable(verdict: EligibilityVerdict, denominator_label: str) -> None:
    """The two frozen gates every normalised privacy output passes [AUTH: 00 §18.1, §20.8]."""
    if verdict.status != ELIGIBLE:
        raise ReportingError(
            "condition is PRIVACY_ENDPOINT_INELIGIBLE; report M(A), its CI and the score"
            f" distributions instead [AUTH: 00 §18.1] ({'; '.join(verdict.reasons)})"
        )
    if denominator_label == "NORMALISATION_UNUSABLE":
        raise ReportingError(
            "denominator relative SE exceeds the 00 §20.8 limit; use the raw paired D_L/D_R"
            " rather than normalised cross-cell values"
        )


def bootstrap_normalised_gap(
    records: Sequence[ViewRecord],
    *,
    verdict: EligibilityVerdict,
    denominator_label: str,
    left: str,
    right: str,
    oracle_view: str,
    target: float,
    null_level: float,
    n_replicates: int,
    seed: int,
    alpha: float,
) -> BootstrapResult:
    """G_L / G_R with a confidence interval — the ONLY public normalised-gap resampler.

    The eligibility verdict and denominator state are checked BEFORE any replicate runs, so
    no point estimate is computed and then withheld. Numerator and oracle denominator are
    resampled inside the same replicate, because separating them breaks the correlation
    between a cell's gap and its own oracle signal [AUTH: 00 §18.1, §20.5, §20.7, §20.8,
    §32.3].
    """
    _require_normalisable(verdict, denominator_label)
    return _bootstrap_normalised_gap_unguarded(
        records,
        left=left,
        right=right,
        oracle_view=oracle_view,
        target=target,
        null_level=null_level,
        n_replicates=n_replicates,
        seed=seed,
        alpha=alpha,
    )


# ----------------------------------------------------------------------------------------
# eligibility-gated reporting
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalisedPrivacyReport:
    """Normalised privacy outputs, which exist only for an eligible condition."""

    status: str
    r_priv: float
    lineage_gap: float
    routing_gap: float
    g_l: float
    g_r: float
    denominator_label: str


def report_normalised_privacy(
    *,
    verdict: EligibilityVerdict,
    denominator_label: str,
    m_view: float,
    m_desc: float,
    m_pool: float,
    m_rec: float,
    m_oracle: float,
    null_level: float,
) -> NormalisedPrivacyReport:
    """THE way downstream code obtains R_priv, G_L and G_R [AUTH: 00 §18.1, §19, §20.8].

    An ineligible condition gets no normalised result at all — not a clamped one, not a NaN.
    A `NORMALISATION_UNUSABLE` denominator withholds the normalised gaps as well, since
    00 §20.8 removes G_L/G_R from inferential use there; the raw paired differences remain
    available from `metrics` and are reported instead.
    """
    _require_normalisable(verdict, denominator_label)
    d_l = lineage_gap(m_pool, m_desc)
    d_r = routing_gap(m_rec, m_pool)
    return NormalisedPrivacyReport(
        status=verdict.status,
        r_priv=_r_priv(m_view, m_oracle, null_level),
        lineage_gap=d_l,
        routing_gap=d_r,
        g_l=_normalised_gap(d_l, m_oracle, null_level),
        g_r=_normalised_gap(d_r, m_oracle, null_level),
        denominator_label=denominator_label,
    )


def report_normalised_functional(
    *, verdict: FunctionalVerdict, loss_base: float, loss_view: float, loss_oracle: float
) -> float:
    """R_func, gated by the 00 §23A eligibility conditions."""
    if not verdict.eligible:
        raise ReportingError(
            "condition is FUNCTIONAL_ENDPOINT_INELIGIBLE; R_func is not defined here"
            f" [AUTH: 00 §23A] ({'; '.join(verdict.reasons)})"
        )
    return _r_func(loss_base, loss_view, loss_oracle)


#: The experiment-facing surface. Raw endpoints and their paired intervals are unguarded;
#: every normalised output requires the eligibility and denominator state [AUTH: 00 §18.1,
#: §20.8, §23A]. `_bootstrap_normalised_gap_unguarded` is deliberately absent.
__all__ = [
    "NormalisedPrivacyReport",
    "ReportingError",
    "ViewRecord",
    "bootstrap_endpoint_gap",
    "bootstrap_normalised_gap",
    "endpoint",
    "endpoint_difference",
    "report_normalised_functional",
    "report_normalised_privacy",
]
