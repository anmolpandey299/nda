"""M/N/O — eligibility, R_priv edge cases, paired audit gaps [AUTH: 00 §17-§20, §23A]."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.metrics import (
    ELIGIBLE,
    INELIGIBLE,
    EligibilityVerdict,
    MetricError,
    _normalised_gap,
    _r_func,
    _r_priv,
    denominator_stability,
    functional_eligibility,
    lineage_gap,
    oracle_signal,
    privacy_eligibility,
    r_priv_for_condition,
    routing_gap,
)
from src.analysis.settings import analysis_constants

#: From the resolved analysis config, not duplicated here [AUTH: 01 §17].
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = analysis_constants(REPO_ROOT)
NULL_LEVEL = ANALYSIS.null_level
FLOOR = ANALYSIS.eligibility_median_floor


def _verdict(oracle: list[float], ci: tuple[float, float]) -> EligibilityVerdict:
    return privacy_eligibility(
        oracle_per_seed=oracle, signal_ci=ci, null_level=NULL_LEVEL, median_floor=FLOOR
    )


# ------------------------------------------------------------------ M: eligibility
def test_ci_above_zero_and_median_above_floor_is_eligible() -> None:
    verdict = _verdict([0.08, 0.09, 0.10], (0.02, 0.09))
    assert verdict.status == ELIGIBLE
    assert verdict.reasons == ()
    assert verdict.oracle_median == pytest.approx(0.09)


def test_a_ci_containing_zero_is_ineligible() -> None:
    verdict = _verdict([0.08, 0.09, 0.10], (-0.01, 0.09))
    assert verdict.status == INELIGIBLE
    assert any("exclude zero" in reason for reason in verdict.reasons)


def test_a_ci_touching_zero_is_ineligible() -> None:
    """ "Excludes zero on the positive side" means strictly above, not "reaches" it."""
    assert _verdict([0.08, 0.09, 0.10], (0.0, 0.09)).status == INELIGIBLE


def test_a_median_below_the_floor_is_ineligible() -> None:
    verdict = _verdict([0.02, 0.03, 0.04], (0.01, 0.05))
    assert verdict.status == INELIGIBLE
    assert any("floor" in reason for reason in verdict.reasons)


def test_the_floor_is_inclusive() -> None:
    assert _verdict([0.05, 0.05, 0.05], (0.01, 0.05)).status == ELIGIBLE


def test_both_failures_are_both_reported() -> None:
    verdict = _verdict([0.01, 0.02, 0.03], (-0.02, 0.02))
    assert len(verdict.reasons) == 2


def test_eligibility_needs_per_seed_values() -> None:
    with pytest.raises(MetricError, match="no per-seed"):
        _verdict([], (0.01, 0.05))


def test_an_inverted_interval_is_refused() -> None:
    with pytest.raises(MetricError, match="inverted"):
        _verdict([0.09], (0.5, 0.1))


# ------------------------------------------------------------------ N: R_priv edges
def test_r_priv_endpoint_checks() -> None:
    """M(V) = M(A) -> 1 and M(V) = M0 -> 0, the two checks 00 §19 states explicitly."""
    assert _r_priv(0.20, 0.20, NULL_LEVEL) == pytest.approx(1.0)
    assert _r_priv(NULL_LEVEL, 0.20, NULL_LEVEL) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("m_view", "expected"),
    [(0.005, -0.05), (0.01, 0.0), (0.06, 0.5), (0.11, 1.0), (0.21, 2.0)],
)
def test_r_priv_is_unclamped_in_both_directions(m_view: float, expected: float) -> None:
    """Values below 0 and above 1 survive; a clipped implementation fails here [AUTH: 00 §19]."""
    assert _r_priv(m_view, 0.11, NULL_LEVEL) == pytest.approx(expected)


def test_a_zero_denominator_is_never_divided_through() -> None:
    with pytest.raises(MetricError, match="undefined and must not be manufactured"):
        _r_priv(0.5, NULL_LEVEL, NULL_LEVEL)


def test_an_ineligible_condition_yields_no_ratio() -> None:
    verdict = _verdict([0.02], (-0.01, 0.02))
    with pytest.raises(MetricError, match="INELIGIBLE"):
        r_priv_for_condition(0.05, 0.02, NULL_LEVEL, verdict)


def test_an_eligible_condition_yields_the_ratio() -> None:
    verdict = _verdict([0.11, 0.11, 0.11], (0.05, 0.15))
    assert r_priv_for_condition(0.06, 0.11, NULL_LEVEL, verdict) == pytest.approx(0.5)


def test_oracle_signal_is_the_plain_difference() -> None:
    assert oracle_signal(0.11, NULL_LEVEL) == pytest.approx(0.10)


# ------------------------------------------------------------------ O: paired gaps
def test_the_two_gaps_are_distinct_differences() -> None:
    """D_L = M_pool - M_desc and D_R = M_rec - M_pool; collapsing them would make both equal
    to M_rec - M_desc [AUTH: 00 §20.4, §20.6]."""
    m_desc, m_pool, m_rec = 0.10, 0.14, 0.19
    assert lineage_gap(m_pool, m_desc) == pytest.approx(0.04)
    assert routing_gap(m_rec, m_pool) == pytest.approx(0.05)
    assert lineage_gap(m_pool, m_desc) != routing_gap(m_rec, m_pool)
    assert lineage_gap(m_pool, m_desc) + routing_gap(m_rec, m_pool) == pytest.approx(m_rec - m_desc)


def test_normalised_gaps_share_the_oracle_denominator() -> None:
    assert _normalised_gap(0.04, 0.11, NULL_LEVEL) == pytest.approx(0.4)
    assert _normalised_gap(-0.04, 0.11, NULL_LEVEL) == pytest.approx(-0.4)
    with pytest.raises(MetricError, match="undefined"):
        _normalised_gap(0.04, NULL_LEVEL, NULL_LEVEL)


# ------------------------------------------------------------------ denominator stability
@pytest.mark.parametrize(
    ("signal", "se", "expected"),
    [
        (0.10, 0.01, "STABLE"),
        (0.10, 0.02, "STABLE"),
        (0.10, 0.025, "DENOMINATOR_UNSTABLE"),
        (0.10, 0.05, "NORMALISATION_UNUSABLE"),
        (0.0, 0.01, "NORMALISATION_UNUSABLE"),
    ],
)
def test_denominator_stability_bands(signal: float, se: float, expected: str) -> None:
    assert (
        denominator_stability(
            signal=signal, standard_error=se, unstable_at=0.20, unusable_above=0.30
        )
        == expected
    )


# ------------------------------------------------------------------ R_func
def test_r_func_endpoints() -> None:
    assert _r_func(2.0, 1.0, 1.0) == pytest.approx(1.0)
    assert _r_func(2.0, 2.0, 1.0) == pytest.approx(0.0)
    with pytest.raises(MetricError, match="undefined"):
        _r_func(2.0, 1.5, 2.0)


def test_functional_eligibility_requires_all_three_conditions() -> None:
    good = functional_eligibility(
        loss_base=2.0,
        loss_oracle=1.8,
        signal_ci=(0.05, 0.3),
        relative_floor=0.02,
        max_relative_se=0.2,
        standard_error=0.02,
    )
    assert good.eligible
    weak = functional_eligibility(
        loss_base=2.0,
        loss_oracle=1.99,
        signal_ci=(-0.01, 0.3),
        relative_floor=0.02,
        max_relative_se=0.2,
        standard_error=0.5,
    )
    assert not weak.eligible
    assert len(weak.reasons) == 3
