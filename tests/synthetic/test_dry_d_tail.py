"""DRY-D — the spectral-tail scenario, activated now that S07 and S08 exist.

DRY-D asks whether privacy keeps degrading in the spectral tail *beyond* what the rank-matched
parameter-error curve already predicts:

    Delta_tail_priv(s) = R_priv^trunc(s) - f_priv^rank(e_floor(s))          [AUTH: 00 §28B.4]

so a trial needs a real O3 truncation ladder (for e_floor(s)) and a real recovery on each rung.
S07 supplies the first, S08 the second; only the privacy responses are planted, exactly as
DRY-A and DRY-B plant theirs — a dry run has no model to score.

DRY-D1 plants a genuine tail effect in both readouts and must corroborate. DRY-D2 moves only
the reference readout and must come back BASE_GEOMETRY_SENSITIVE rather than corroborated
[AUTH: 00 §34B.1 DRY-D].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.settings import dry_run_settings

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.analysis.dryrun import (  # noqa: E402
    DRY_D_ANCHOR_RANKS,
    DRY_D_BASE_GEOMETRY,
    DRY_D_FAIL,
    DRY_D_INTERIOR_RANKS,
    DRY_D_PASS,
    DRY_D_RETAINED_RANKS,
    DRY_D_TRIAL_STATISTIC,
    SUPERSEDED_DRY_D1_MINK_MAX,
    SUPERSEDED_DRY_D1_REFERENCE_MAX,
    SUPERSEDED_THRESHOLD_STATUS,
    evaluate_dry_d1,
    evaluate_dry_d2,
    generate_tail_trial,
    tail_thresholds,
)

#: The frozen planted DRY-D1 effect, from config — the same value the calibration bank used.
TAIL_SHIFT = dry_run_settings(REPO_ROOT).number("dry_d1_planted_shift")
THRESHOLDS = tail_thresholds(REPO_ROOT)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


@pytest.fixture(scope="module")
def corroborated_trial() -> object:
    return generate_tail_trial(4242, reference_shift=TAIL_SHIFT, mink_shift=TAIL_SHIFT)


@pytest.fixture(scope="module")
def geometry_trial() -> object:
    return generate_tail_trial(99, reference_shift=TAIL_SHIFT, mink_shift=0.0)


@pytest.fixture(scope="module")
def null_trial() -> object:
    return generate_tail_trial(7, reference_shift=0.0, mink_shift=0.0)


# ------------------------------------------------------------------ the planted geometry
def test_the_truncation_ladder_is_the_complete_frozen_28b_schedule(corroborated_trial) -> None:  # type: ignore[no-untyped-def]
    """R9/R10: all TWELVE interior ranks, exactly, in the registered order."""
    assert DRY_D_INTERIOR_RANKS == (31, 28, 24, 20, 16, 12, 8, 6, 4, 3, 2, 1)
    assert len(DRY_D_INTERIOR_RANKS) == 12
    assert DRY_D_ANCHOR_RANKS == (32, 0)
    assert not set(DRY_D_INTERIOR_RANKS) & set(DRY_D_ANCHOR_RANKS)
    assert DRY_D_RETAINED_RANKS == DRY_D_INTERIOR_RANKS
    assert corroborated_trial.retained_ranks == DRY_D_INTERIOR_RANKS
    assert list(corroborated_trial.retained_ranks) == sorted(DRY_D_INTERIOR_RANKS, reverse=True)
    assert len(corroborated_trial.e_floor) == 12
    assert len(corroborated_trial.solver_error) == 12


@pytest.mark.parametrize("dropped", [31, 28])
def test_dropping_a_registered_interior_rank_changes_the_statistic(dropped: int) -> None:
    """R10: omitting 31 or 28 — the two the first implementation missed — cannot pass quietly.

    A shortened ladder is a different trial, and this test fails the moment the implementation
    stops running the registered schedule.
    """
    assert dropped in DRY_D_INTERIOR_RANKS
    short = tuple(rank for rank in DRY_D_INTERIOR_RANKS if rank != dropped)
    truncated = generate_tail_trial(
        4242, reference_shift=TAIL_SHIFT, mink_shift=TAIL_SHIFT, retained_ranks=short
    )
    full = generate_tail_trial(4242, reference_shift=TAIL_SHIFT, mink_shift=TAIL_SHIFT)
    assert truncated.retained_ranks != full.retained_ranks
    assert len(truncated.e_floor) == 11
    assert truncated.as_dict() != full.as_dict()


def test_rank_31_and_28_are_genuine_truncations_on_this_fixture(corroborated_trial) -> None:  # type: ignore[no-untyped-def]
    """A ladder point that clamps to full rank would contribute a duplicate, not a rung.

    The planted constituent has rank 32 on a 40x36 surface, so retaining 31 or 28 directions
    really does discard information — which is why the fixture surface is wide enough to carry
    the 01 §8B adapter rank the 00 §28B.1 ladder is written for.
    """
    floors = dict(zip(corroborated_trial.retained_ranks, corroborated_trial.e_floor, strict=True))
    assert floors[31] > 0.0
    assert floors[28] > floors[31]
    assert len(set(corroborated_trial.e_floor)) == 12


def test_e_floor_is_measured_from_real_truncations_not_assumed(corroborated_trial) -> None:  # type: ignore[no-untyped-def]
    """The floor rises monotonically as directions are discarded.

    The planted constituent is exactly rank 32, so every interior rung discards something and
    e_floor(s) increases all the way down the ladder. Nothing supplies that shape except a
    genuine SVD truncation, which is the S07 dependency DRY-D was waiting on.
    """
    floors = np.asarray(corroborated_trial.e_floor, dtype=float)
    assert np.all(floors > 0.0)
    assert np.all(np.diff(floors) > 0.0), f"floor is not increasing into the tail: {floors}"
    assert floors[0] < 0.05, "retaining 31 of 32 directions should lose very little"
    assert floors[-1] > 0.8, "retaining one direction should lose most of A"


def test_the_solver_error_axis_is_reported_but_not_consumed(corroborated_trial) -> None:  # type: ignore[no-untyped-def]
    """R11: e_solver is a reported diagnostic; Delta_tail_priv does not consume it.

    00 §28B.4 compares the truncated artifact's privacy response against the rank-matched
    calibration curve at e_floor(s). The solver error rides along for interpretation and is
    finite and non-negative everywhere, but no part of the statistic reads it.
    """
    errors = np.asarray(corroborated_trial.solver_error, dtype=float)
    assert np.all(np.isfinite(errors))
    assert np.all(errors >= 0.0)


def test_the_trial_statistic_is_recorded_with_its_per_rank_values(corroborated_trial) -> None:  # type: ignore[no-untyped-def]
    """00 §34B.1 wants one scalar per score; the per-rank values are reported, not discarded."""
    row = corroborated_trial.as_dict()
    assert row["statistic"] == DRY_D_TRIAL_STATISTIC
    assert len(row["reference_delta_tail_priv"]) == 12
    assert len(row["mink_delta_tail_priv"]) == 12
    assert row["mean_reference_delta_tail_priv"] == pytest.approx(
        float(np.mean(row["reference_delta_tail_priv"]))
    )
    assert set(row["retained_ranks"]) == set(DRY_D_INTERIOR_RANKS)


# ------------------------------------------------------------------ DRY-D1
def test_dry_d1_corroborates_a_planted_tail_effect(corroborated_trial) -> None:  # type: ignore[no-untyped-def]
    verdict = evaluate_dry_d1(corroborated_trial)
    assert verdict.scenario == "DRY-D1"
    assert verdict.status == DRY_D_PASS
    assert corroborated_trial.mean_reference_delta < THRESHOLDS.d1_reference
    assert corroborated_trial.mean_mink_delta < THRESHOLDS.d1_mink


def test_dry_d1_recovers_the_planted_shift(corroborated_trial) -> None:  # type: ignore[no-untyped-def]
    """Delta_tail_priv IS the planted deviation from the rank-matched curve, so it recovers it."""
    assert corroborated_trial.mean_reference_delta == pytest.approx(TAIL_SHIFT, abs=0.02)
    assert corroborated_trial.mean_mink_delta == pytest.approx(TAIL_SHIFT, abs=0.02)


def test_dry_d1_does_not_fire_without_a_planted_tail_effect(null_trial) -> None:  # type: ignore[no-untyped-def]
    assert evaluate_dry_d1(null_trial).status == DRY_D_FAIL


def test_dry_d1_requires_both_readouts(geometry_trial) -> None:  # type: ignore[no-untyped-def]
    """A reference-only movement is exactly what the corroboration rule must reject."""
    assert geometry_trial.mean_reference_delta < THRESHOLDS.d1_reference
    assert geometry_trial.mean_mink_delta > THRESHOLDS.d1_mink
    assert evaluate_dry_d1(geometry_trial).status == DRY_D_FAIL


# ------------------------------------------------------------------ DRY-D2
def test_dry_d2_flags_a_geometry_confounded_trial(geometry_trial) -> None:  # type: ignore[no-untyped-def]
    verdict = evaluate_dry_d2(geometry_trial)
    assert verdict.scenario == "DRY-D2"
    assert verdict.status == DRY_D_BASE_GEOMETRY
    assert geometry_trial.mean_reference_delta < THRESHOLDS.d2_reference
    assert abs(geometry_trial.mean_mink_delta) < THRESHOLDS.d2_mink_abs


def test_dry_d2_does_not_claim_geometry_when_both_readouts_move(corroborated_trial) -> None:  # type: ignore[no-untyped-def]
    """The two verdicts are mutually exclusive on the same trial, which is the point."""
    assert evaluate_dry_d2(corroborated_trial).status == DRY_D_FAIL
    assert evaluate_dry_d1(corroborated_trial).status == DRY_D_PASS


def test_dry_d2_does_not_fire_on_a_null_trial(null_trial) -> None:  # type: ignore[no-untyped-def]
    assert evaluate_dry_d2(null_trial).status == DRY_D_FAIL


# ------------------------------------------------------------------ frozen thresholds
def test_the_cutoffs_come_from_the_frozen_calibration_artifact() -> None:
    """One source of truth: the config validation reads IS the calibrated bank's output.

    Result-driven tuning of a planted-truth cutoff is prohibited [AUTH: 00 §34B.1A], and
    calibration is generated once pre-data rather than on every run [AUTH: 02 §C5].
    """
    artifact = json.loads(
        (REPO_ROOT / "artifacts/p0_pre/P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json").read_text(
            encoding="utf-8"
        )
    )
    assert THRESHOLDS.d1_reference == artifact["threshold_reference"]
    assert THRESHOLDS.d1_mink == artifact["threshold_mink"]
    assert artifact["scenario"] == "DRY_D1"
    assert artifact["generated_pre_real_data"] is True
    assert THRESHOLDS.d1_reference < THRESHOLDS.d2_reference < 0.0


def test_dry_d2_was_not_recalibrated() -> None:
    """02 §C5 covers positive-control power calibration; DRY-D2's pair is a spec constant."""
    assert (THRESHOLDS.d2_reference, THRESHOLDS.d2_mink_abs) == (-0.08, 0.04)


def test_the_superseded_dry_d1_cutoffs_are_recorded_and_never_consumed() -> None:
    """The ten-rank implementation's cutoffs are history, not a fallback [AUTH: 02 §C5]."""
    assert (SUPERSEDED_DRY_D1_REFERENCE_MAX, SUPERSEDED_DRY_D1_MINK_MAX) == (
        -0.136213,
        -0.135921,
    )
    assert SUPERSEDED_THRESHOLD_STATUS == "SUPERSEDED_PRE_DATA_BY_02_C5_RECALIBRATION"
    assert THRESHOLDS.d1_reference != SUPERSEDED_DRY_D1_REFERENCE_MAX
    assert THRESHOLDS.d1_mink != SUPERSEDED_DRY_D1_MINK_MAX
    #: the superseded pair is strictly looser, which is exactly why it had to be regenerated
    assert THRESHOLDS.d1_reference < SUPERSEDED_DRY_D1_REFERENCE_MAX
    assert THRESHOLDS.d1_mink < SUPERSEDED_DRY_D1_MINK_MAX


def test_a_trial_is_reproducible_from_its_seed() -> None:
    left = generate_tail_trial(2024, reference_shift=TAIL_SHIFT, mink_shift=0.0)
    right = generate_tail_trial(2024, reference_shift=TAIL_SHIFT, mink_shift=0.0)
    assert left.as_dict() == right.as_dict()
    assert evaluate_dry_d2(left).as_dict() == evaluate_dry_d2(right).as_dict()


def test_the_verdict_row_carries_the_whole_trial(geometry_trial) -> None:  # type: ignore[no-untyped-def]
    row = evaluate_dry_d2(geometry_trial).as_dict()
    assert row["scenario"] == "DRY-D2"
    assert row["status"] == DRY_D_BASE_GEOMETRY
    assert "e_floor" in row and "solver_error" in row
