"""DRY-G — complete null / false-positive control [AUTH: 02 §C4; 00 §34B.1].

The gate is the binding 02 §C4 family-wise procedure: a 2,000-trial bank on fixed master
seeds from resolved config, each statistic standardised from the bank alone, T_max the
maximum standardised deviation, and c_FWER = Q_0.95(T_max). Cell-cluster intervals are
computed and reported, and decide nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from src.analysis.dryrun import (
    NULL_STATISTIC_NAMES,
    calibrate_familywise,
    cell_cluster_intervals,
    evaluate_dry_g,
    generate_null_trial,
    marginal_diagnostic_tolerances,
    null_statistics,
)
from src.analysis.resampling import ClusterUnit, bootstrap, cluster_rows
from src.analysis.settings import dry_run_settings, trial_design

REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN = trial_design(REPO_ROOT)


@pytest.fixture(scope="module")
def settings(repo_root: Path):  # type: ignore[no-untyped-def]
    return dry_run_settings(repo_root)


@pytest.fixture(scope="module")
def calibration(settings):  # type: ignore[no-untyped-def]
    start = settings.integer("familywise_calibration_seed_start")
    count = settings.integer("familywise_calibration_trials")
    return calibrate_familywise(
        DESIGN, list(range(start, start + count)), quantile=settings.number("familywise_quantile")
    )


# ------------------------------------------------------------------ the C4 bank
def test_the_bank_uses_the_configured_identities(calibration, settings) -> None:  # type: ignore[no-untyped-def]
    """2,000 trials on fixed master seeds, taken from resolved config [AUTH: 02 §C4]."""
    start = settings.integer("familywise_calibration_seed_start")
    count = settings.integer("familywise_calibration_trials")
    assert count == 2000
    assert calibration.n_trials == count
    assert calibration.seeds == tuple(range(start, start + count))
    assert calibration.quantile == 0.95
    assert set(calibration.centre) == set(NULL_STATISTIC_NAMES)
    assert set(calibration.scale) == set(NULL_STATISTIC_NAMES)
    assert all(scale > 0.0 for scale in calibration.scale.values())
    assert calibration.c_fwer > 0.0


def test_standardisation_uses_the_bank_alone(calibration) -> None:  # type: ignore[no-untyped-def]
    """Two-sided statistics use |Z|; the one-sided Delta_CV uses the signed Z [AUTH: 02 §C4]."""
    centred = dict(calibration.centre)
    assert calibration.t_max(centred) == pytest.approx(0.0, abs=1e-12)

    shifted = dict(calibration.centre)
    shifted["spearman_rho"] = centred["spearman_rho"] - 3.0 * calibration.scale["spearman_rho"]
    assert calibration.standardise(shifted)["spearman_rho"] == pytest.approx(3.0)

    lowered = dict(calibration.centre)
    lowered["delta_cv"] = centred["delta_cv"] - 3.0 * calibration.scale["delta_cv"]
    # a NEGATIVE Delta_CV is not evidence against the null, so it must not raise T_max
    assert calibration.standardise(lowered)["delta_cv"] == pytest.approx(-3.0)
    assert calibration.t_max(lowered) < 3.0


def test_c_fwer_is_the_bank_quantile(calibration) -> None:  # type: ignore[no-untyped-def]
    """c_FWER is Q_0.95 of the bank's own T_max distribution, so about 5% of null trials
    exceed it by construction [AUTH: 02 §C4]."""
    values = np.array(
        [
            calibration.t_max(null_statistics(generate_null_trial(DESIGN, seed)))
            for seed in calibration.seeds[:400]
        ]
    )
    assert float(np.mean(values > calibration.c_fwer)) == pytest.approx(0.05, abs=0.04)


def test_the_calibration_bank_is_deterministic(settings) -> None:  # type: ignore[no-untyped-def]
    start = settings.integer("familywise_calibration_seed_start")
    first = calibrate_familywise(DESIGN, list(range(start, start + 50)), quantile=0.95)
    second = calibrate_familywise(DESIGN, list(range(start, start + 50)), quantile=0.95)
    assert first.c_fwer == second.c_fwer and first.centre == second.centre


def test_the_planted_null_is_actually_null() -> None:
    rows = [row for seed in range(820000, 820040) for row in generate_null_trial(DESIGN, seed)]
    from src.analysis.isotonic import spearman

    assert abs(spearman([r.error for r in rows], [r.recovery for r in rows])) < 0.05
    assert float(np.mean([r.d_l for r in rows])) == pytest.approx(0.0, abs=0.01)
    assert float(np.mean([r.d_r for r in rows])) == pytest.approx(0.0, abs=0.01)


def test_the_gate_rejects_a_planted_effect(calibration) -> None:  # type: ignore[no-untyped-def]
    """A calibration that never fires is worthless."""
    from src.analysis.dryrun import generate_monotone_trial

    rows = generate_monotone_trial(DESIGN, 860000, slope=0.45, operator_effect={"O1": 0.25})
    assert not calibration.passes(null_statistics(rows))


# ------------------------------------------------------------------ validation
def test_validation_seeds_are_outside_the_calibration_bank(calibration, settings) -> None:  # type: ignore[no-untyped-def]
    seeds = settings.integers("validation_seeds")
    assert not set(seeds) & set(calibration.seeds)


def test_a_validation_seed_inside_the_bank_is_refused(calibration) -> None:  # type: ignore[no-untyped-def]
    from src.analysis.dryrun import DryRunError

    with pytest.raises(DryRunError, match="inside the calibration bank"):
        evaluate_dry_g(DESIGN, calibration, [calibration.seeds[0]], n_replicates=50, alpha=0.05)


def test_dry_g_validation_majority_semantics(calibration, settings) -> None:  # type: ignore[no-untyped-def]
    """3/3 -> PASS, 2/3 -> PASS_WITH_INVESTIGATION, <=1/3 -> FAIL [AUTH: 00 §34B.1]."""
    verdict = evaluate_dry_g(
        DESIGN,
        calibration,
        settings.integers("validation_seeds"),
        n_replicates=settings.integer("cell_cluster_bootstrap_replicates"),
        alpha=settings.number("cell_cluster_alpha"),
    )
    assert len(verdict.trials) == 3
    assert verdict.status in {"PASS", "PASS_WITH_INVESTIGATION"}, verdict.investigation_note()
    assert verdict.n_passed >= 2, verdict.investigation_note()
    for trial in verdict.trials:
        assert trial.passed == (trial.t_max <= calibration.c_fwer)


def test_cell_cluster_intervals_are_diagnostics_only(calibration, settings) -> None:  # type: ignore[no-untyped-def]
    """02 §C4: the intervals are reported, and are not a second pass/fail criterion."""
    verdict = evaluate_dry_g(
        DESIGN, calibration, settings.integers("validation_seeds"), n_replicates=200, alpha=0.05
    )
    for trial in verdict.trials:
        assert trial.diagnostic_intervals, "the diagnostic intervals were not reported"
        excluding = [
            name
            for name, (low, high) in trial.diagnostic_intervals.items()
            if not (low <= 0.0 <= high)
        ]
        # the verdict depends on T_max alone, whatever the intervals happen to show
        assert trial.passed == (trial.t_max <= calibration.c_fwer), excluding


def test_cell_cluster_intervals_are_deterministic() -> None:
    rows = generate_null_trial(DESIGN, 810200)
    first = cell_cluster_intervals(rows, n_replicates=200, seed=5, alpha=0.05)
    assert first == cell_cluster_intervals(rows, n_replicates=200, seed=5, alpha=0.05)


def test_cluster_resampling_keeps_every_seed_row() -> None:
    rows = generate_null_trial(DESIGN, 810200)
    by_cell: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        by_cell.setdefault(row.cell, []).append((row.d_l, row.d_r))
    clusters = [ClusterUnit(cell=c, rows=tuple(v)) for c, v in sorted(by_cell.items())]

    def statistic(units: Sequence[ClusterUnit]) -> float:
        values = cluster_rows(units)
        assert len(values) == 3 * len(units), "a sampled cell lost seed rows"
        return float(np.mean([v[0] for v in values]))

    assert bootstrap(statistic, clusters, n_replicates=200, seed=7, alpha=0.05).contains_zero()


# ------------------------------------------------------------------ legacy diagnostic
def test_the_marginal_bank_is_non_gating(settings) -> None:  # type: ignore[no-untyped-def]
    """The 200-trial v1.9 marginal bank survives only as a labelled diagnostic [AUTH: 02 §C4]."""
    assert settings.text("legacy_marginal_status") == "NON_GATING_DIAGNOSTIC"
    start = settings.integer("legacy_marginal_seed_start")
    count = settings.integer("legacy_marginal_trials")
    values = marginal_diagnostic_tolerances(
        DESIGN,
        list(range(start, start + count)),
        quantile=settings.number("legacy_marginal_quantile"),
    )
    assert set(values) == set(NULL_STATISTIC_NAMES)
    # nothing in the gate reads it
    import inspect

    from src.analysis import dryrun

    assert "marginal_diagnostic_tolerances" not in inspect.getsource(dryrun.evaluate_dry_g)
