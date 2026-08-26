"""B7/B8 — endpoint-aware paired bootstrap and eligibility-gated reporting.

[AUTH: 00 §18.1, §19, §20.4-§20.8, §23A, §32.1, §32.3]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.endpoints import (
    ReportingError,
    ViewRecord,
    bootstrap_endpoint_gap,
    bootstrap_normalised_gap,
    endpoint,
    endpoint_difference,
    report_normalised_functional,
    report_normalised_privacy,
)
from src.analysis.metrics import (
    functional_eligibility,
    privacy_eligibility,
)
from src.analysis.settings import analysis_constants, scoring_constants

#: From resolved config [AUTH: 01 §17].
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = scoring_constants(REPO_ROOT).target_fpr
NULL_LEVEL = analysis_constants(REPO_ROOT).null_level


def _records(shift: dict[str, float], n: int = 400, seed: int = 5) -> list[ViewRecord]:
    """One record per unit, carrying every view's score for that same record."""
    from src.analysis.dryrun import gaussian_membership_scores

    base, labels = gaussian_membership_scores(n_member=n, n_non_member=n, separation=1.6, seed=seed)
    return [
        ViewRecord(
            record_id=f"r{index:05d}",
            membership_label=int(labels[index]),
            scores={
                view: float(base[index]) + (offset if labels[index] == 1 else 0.0)
                for view, offset in shift.items()
            },
        )
        for index in range(base.size)
    ]


# ================================================================== B7: endpoint bootstrap
def test_the_gap_is_a_difference_of_endpoints_not_of_means() -> None:
    """The reviewer counterexample: a monotone transform of one view leaves its record-level
    mean far away while the fixed-FPR endpoint is unchanged, so the two statistics disagree
    [AUTH: 00 §20.4]."""
    records = _records({"DESC": 0.0, "POOL": 0.6})
    endpoint_gap = endpoint_difference(records, left="POOL", right="DESC", target=TARGET)

    scaled = [
        ViewRecord(
            record_id=record.record_id,
            membership_label=record.membership_label,
            scores={**record.scores, "POOL": record.scores["POOL"] * 100.0},
        )
        for record in records
    ]
    scaled_endpoint_gap = endpoint_difference(scaled, left="POOL", right="DESC", target=TARGET)
    assert scaled_endpoint_gap == pytest.approx(endpoint_gap)

    mean_gap = float(np.mean([r.scores["POOL"] - r.scores["DESC"] for r in records]))
    scaled_mean_gap = float(np.mean([r.scores["POOL"] - r.scores["DESC"] for r in scaled]))
    assert scaled_mean_gap != pytest.approx(mean_gap)
    assert scaled_mean_gap != pytest.approx(scaled_endpoint_gap)


def test_every_replicate_recomputes_the_endpoint() -> None:
    """A replicate whose resampled records change the ROC must change the statistic; a scalar
    mean difference could not see it."""
    records = _records({"DESC": 0.0, "POOL": 0.6})
    result = bootstrap_endpoint_gap(
        records, left="POOL", right="DESC", target=TARGET, n_replicates=200, seed=1, alpha=0.05
    )
    assert result.point == pytest.approx(
        endpoint_difference(records, left="POOL", right="DESC", target=TARGET)
    )
    assert result.replicates.std() > 0.0, "the replicates never re-derived the endpoint"
    assert result.excludes_zero_positive()


def test_pairing_survives_the_endpoint_bootstrap() -> None:
    """Both views travel in one unit, so identical views give an exactly zero gap in every
    replicate [AUTH: 00 §32.3]."""
    records = _records({"DESC": 0.0, "POOL": 0.0})
    result = bootstrap_endpoint_gap(
        records, left="POOL", right="DESC", target=TARGET, n_replicates=150, seed=2, alpha=0.05
    )
    assert np.allclose(result.replicates, 0.0)
    assert result.contains_zero()


def test_the_two_gaps_stay_distinct_through_resampling() -> None:
    records = _records({"DESC": 0.0, "POOL": 0.4, "REC": 0.9})
    d_l = bootstrap_endpoint_gap(
        records, left="POOL", right="DESC", target=TARGET, n_replicates=120, seed=3, alpha=0.05
    )
    d_r = bootstrap_endpoint_gap(
        records, left="REC", right="POOL", target=TARGET, n_replicates=120, seed=3, alpha=0.05
    )
    assert d_l.point != pytest.approx(d_r.point)


def test_the_normalised_gap_shares_one_replicate_with_its_denominator() -> None:
    """G resamples numerator and oracle denominator inside the same replicate [00 §20.5]."""
    records = _records({"DESC": 0.0, "POOL": 0.4, "ORACLE": 1.2})
    verdict = privacy_eligibility(
        oracle_per_seed=[0.11, 0.11, 0.11],
        signal_ci=(0.05, 0.15),
        null_level=NULL_LEVEL,
        median_floor=0.05,
    )
    result = bootstrap_normalised_gap(
        records,
        verdict=verdict,
        denominator_label="STABLE",
        left="POOL",
        right="DESC",
        oracle_view="ORACLE",
        target=TARGET,
        null_level=NULL_LEVEL,
        n_replicates=200,
        seed=4,
        alpha=0.05,
    )
    gap = endpoint_difference(records, left="POOL", right="DESC", target=TARGET)
    denominator = endpoint(records, "ORACLE", TARGET) - NULL_LEVEL
    assert result.point == pytest.approx(gap / denominator)
    assert np.isfinite(result.replicates).all()


def test_a_missing_view_fails_closed() -> None:
    from src.analysis.metrics import MetricError

    records = _records({"DESC": 0.0})
    with pytest.raises(MetricError, match="carry no"):
        endpoint(records, "POOL", TARGET)


# ================================================================== B8: eligibility gate
def _verdict(oracle: list[float], ci: tuple[float, float]):  # type: ignore[no-untyped-def]
    return privacy_eligibility(
        oracle_per_seed=oracle, signal_ci=ci, null_level=NULL_LEVEL, median_floor=0.05
    )


def test_an_ineligible_condition_emits_no_normalised_result() -> None:
    """oracle = [.02, .02, .02] -> PRIVACY_ENDPOINT_INELIGIBLE -> no R_priv, no G [00 §18.1]."""
    verdict = _verdict([0.02, 0.02, 0.02], (-0.005, 0.02))
    assert not verdict.eligible
    with pytest.raises(ReportingError, match="INELIGIBLE"):
        report_normalised_privacy(
            verdict=verdict,
            denominator_label="STABLE",
            m_view=0.015,
            m_desc=0.012,
            m_pool=0.014,
            m_rec=0.016,
            m_oracle=0.02,
            null_level=NULL_LEVEL,
        )


def test_a_median_below_the_floor_alone_blocks_the_report() -> None:
    verdict = _verdict([0.02, 0.02, 0.02], (0.005, 0.02))
    assert not verdict.eligible
    with pytest.raises(ReportingError):
        report_normalised_privacy(
            verdict=verdict,
            denominator_label="STABLE",
            m_view=0.015,
            m_desc=0.012,
            m_pool=0.014,
            m_rec=0.016,
            m_oracle=0.02,
            null_level=NULL_LEVEL,
        )


def test_an_unusable_denominator_withholds_the_normalised_gaps() -> None:
    """00 §20.8 removes G_L/G_R from inferential use above the limit; the raw D_L/D_R remain."""
    verdict = _verdict([0.11, 0.11, 0.11], (0.05, 0.15))
    with pytest.raises(ReportingError, match="raw paired"):
        report_normalised_privacy(
            verdict=verdict,
            denominator_label="NORMALISATION_UNUSABLE",
            m_view=0.06,
            m_desc=0.05,
            m_pool=0.07,
            m_rec=0.09,
            m_oracle=0.11,
            null_level=NULL_LEVEL,
        )


def test_an_eligible_condition_reports_every_normalised_quantity() -> None:
    verdict = _verdict([0.11, 0.11, 0.11], (0.05, 0.15))
    report = report_normalised_privacy(
        verdict=verdict,
        denominator_label="STABLE",
        m_view=0.06,
        m_desc=0.05,
        m_pool=0.07,
        m_rec=0.09,
        m_oracle=0.11,
        null_level=NULL_LEVEL,
    )
    assert report.r_priv == pytest.approx(0.5)
    assert report.lineage_gap == pytest.approx(0.02)
    assert report.routing_gap == pytest.approx(0.02)
    assert report.g_l == pytest.approx(0.2)
    assert report.g_r == pytest.approx(0.2)


def test_functional_reporting_is_gated_too() -> None:
    ineligible = functional_eligibility(
        loss_base=2.0,
        loss_oracle=1.99,
        signal_ci=(-0.01, 0.3),
        relative_floor=0.02,
        max_relative_se=0.2,
        standard_error=0.5,
    )
    with pytest.raises(ReportingError, match="FUNCTIONAL_ENDPOINT_INELIGIBLE"):
        report_normalised_functional(
            verdict=ineligible, loss_base=2.0, loss_view=1.995, loss_oracle=1.99
        )
    eligible = functional_eligibility(
        loss_base=2.0,
        loss_oracle=1.8,
        signal_ci=(0.05, 0.3),
        relative_floor=0.02,
        max_relative_se=0.2,
        standard_error=0.02,
    )
    assert report_normalised_functional(
        verdict=eligible, loss_base=2.0, loss_view=1.9, loss_oracle=1.8
    ) == pytest.approx(0.5)


# ================================================================== B8 closure: no bypass
def _ineligible_population() -> tuple[list[ViewRecord], list[float]]:
    """100 members, 100 non-members, with M(A)=0.02, M_desc=0.01, M_pool=0.03 at 1% FPR.

    Constructed by hand: with 100 non-members the achievable rates are multiples of 0.01, so
    the exact 1% point admits the single highest-scoring non-member. Placing 2, 1 and 3
    members above it gives the three endpoints, and the remaining members are put BELOW the
    99 low non-members so no lower threshold can reach them at 1% FPR.
    """
    records: list[ViewRecord] = []
    above = {"ORACLE": 2, "DESC": 1, "POOL": 3}
    for index in range(100):
        records.append(
            ViewRecord(
                record_id=f"mem-{index:03d}",
                membership_label=1,
                scores={view: (10.0 if index < count else -30.0) for view, count in above.items()},
            )
        )
    for index in range(100):
        # exactly one non-member sits at 5.0, so the 1% threshold lands just above it
        value = 5.0 if index == 0 else -20.0
        records.append(
            ViewRecord(
                record_id=f"non-{index:03d}",
                membership_label=0,
                scores=dict.fromkeys(above, value),
            )
        )
    return records, [0.02, 0.02, 0.02]


def test_the_constructed_population_has_the_intended_endpoints() -> None:
    records, _ = _ineligible_population()
    assert endpoint(records, "ORACLE", TARGET) == pytest.approx(0.02)
    assert endpoint(records, "DESC", TARGET) == pytest.approx(0.01)
    assert endpoint(records, "POOL", TARGET) == pytest.approx(0.03)


def test_the_ineligible_condition_cannot_reach_a_normalised_bootstrap() -> None:
    """G_L would be (0.03 - 0.01) / (0.02 - 0.01) = 2.0. The public API must refuse it, and
    must refuse before any replicate runs [AUTH: 00 §18.1]."""
    records, oracle_per_seed = _ineligible_population()
    verdict = privacy_eligibility(
        oracle_per_seed=oracle_per_seed,
        signal_ci=(-0.004, 0.02),
        null_level=NULL_LEVEL,
        median_floor=0.05,
    )
    assert verdict.status == "PRIVACY_ENDPOINT_INELIGIBLE"

    with pytest.raises(ReportingError, match="INELIGIBLE"):
        bootstrap_normalised_gap(
            records,
            verdict=verdict,
            denominator_label="STABLE",
            left="POOL",
            right="DESC",
            oracle_view="ORACLE",
            target=TARGET,
            null_level=NULL_LEVEL,
            n_replicates=200,
            seed=1,
            alpha=0.05,
        )
    with pytest.raises(ReportingError, match="INELIGIBLE"):
        report_normalised_privacy(
            verdict=verdict,
            denominator_label="STABLE",
            m_view=0.02,
            m_desc=0.01,
            m_pool=0.03,
            m_rec=0.03,
            m_oracle=0.02,
            null_level=NULL_LEVEL,
        )


def test_the_unusable_denominator_withholds_the_normalised_point_and_interval() -> None:
    records, _ = _ineligible_population()
    verdict = privacy_eligibility(
        oracle_per_seed=[0.11, 0.11, 0.11],
        signal_ci=(0.05, 0.15),
        null_level=NULL_LEVEL,
        median_floor=0.05,
    )
    assert verdict.eligible
    with pytest.raises(ReportingError, match="raw paired"):
        bootstrap_normalised_gap(
            records,
            verdict=verdict,
            denominator_label="NORMALISATION_UNUSABLE",
            left="POOL",
            right="DESC",
            oracle_view="ORACLE",
            target=TARGET,
            null_level=NULL_LEVEL,
            n_replicates=200,
            seed=2,
            alpha=0.05,
        )


def test_the_raw_gap_and_its_interval_remain_available() -> None:
    """D_L = M_pool - M_desc = 0.02 is still reportable for the ineligible condition."""
    records, _ = _ineligible_population()
    raw = endpoint_difference(records, left="POOL", right="DESC", target=TARGET)
    assert raw == pytest.approx(0.02)
    interval = bootstrap_endpoint_gap(
        records,
        left="POOL",
        right="DESC",
        target=TARGET,
        n_replicates=200,
        seed=3,
        alpha=0.05,
    )
    assert interval.point == pytest.approx(0.02)
    assert interval.low <= interval.point <= interval.high


def test_no_public_export_returns_a_normalised_result_unguarded() -> None:
    """Every experiment-facing normalised entry point requires the eligibility state, and the
    private helper is not exported [AUTH: 00 §18.1, §20.8, §23A]."""
    import inspect

    from src.analysis import endpoints

    guarded = {
        "bootstrap_normalised_gap": {"verdict", "denominator_label"},
        "report_normalised_privacy": {"verdict", "denominator_label"},
        "report_normalised_functional": {"verdict"},
    }
    for name, required in guarded.items():
        assert name in endpoints.__all__, f"{name} is not part of the public surface"
        parameters = set(inspect.signature(getattr(endpoints, name)).parameters)
        assert required <= parameters, f"{name} does not require {sorted(required)}"

    assert "_bootstrap_normalised_gap_unguarded" not in endpoints.__all__
    for name in endpoints.__all__:
        source = inspect.getsource(getattr(endpoints, name))
        if "normalised" in name or "r_priv" in source:
            continue
    # nothing public mentions the private helper except the guarded wrapper
    callers = [
        name
        for name in endpoints.__all__
        if "_bootstrap_normalised_gap_unguarded" in inspect.getsource(getattr(endpoints, name))
    ]
    assert callers == ["bootstrap_normalised_gap"]


def test_no_other_module_reaches_the_unguarded_helper(repo_root: Path) -> None:
    offenders = [
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "src").rglob("*.py")
        if path.name != "endpoints.py"
        and "_bootstrap_normalised_gap_unguarded" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"{offenders} bypass the eligibility guard"
