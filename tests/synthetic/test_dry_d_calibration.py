"""F2-F13 — the regenerated DRY-D1 positive-control calibration [AUTH: 02 §C5; 00 §34B.1A].

02 §C5 binds a positive-control cutoff to the implementation that generated it and forbids
preserving an old tolerance merely because it passed. The DRY-D statistic's retained-rank
schedule was corrected from ten interior ranks to the twelve of 00 §28B.1, so the DRY-D1
cutoffs were regenerated pre-data over the frozen seeds 810000..810199.

These tests read the frozen artifact. They do NOT regenerate it: calibration happens once,
and a suite that recalibrated on every run would have no frozen tolerance at all.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.analysis.dryrun import (
    CALIBRATION_QUANTILE_METHOD,
    DRY_D_ANCHOR_RANKS,
    DRY_D_INTERIOR_RANKS,
    SUPERSEDED_DRY_D1_MINK_MAX,
    SUPERSEDED_DRY_D1_REFERENCE_MAX,
    SUPERSEDED_THRESHOLD_STATUS,
    calibrate_dry_d1_positive_control,
    evaluate_dry_d1,
    generate_tail_trial,
    tail_thresholds,
)
from src.analysis.settings import dry_run_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "artifacts/p0_pre/P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json"
SETTINGS = dry_run_settings(REPO_ROOT)
THRESHOLDS = tail_thresholds(REPO_ROOT)


@pytest.fixture(scope="module")
def bank() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return document


# ------------------------------------------------------------------ F6 the artifact
def test_the_canonical_artifact_exists_where_00_34b2_names_it() -> None:
    assert ARTIFACT.name == "P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json"
    assert ARTIFACT.exists()
    #: exactly one canonical calibration artifact, not two competing ones
    matches = sorted(p.name for p in (REPO_ROOT / "artifacts").rglob("*TOLERANCE_CALIBRATION*"))
    assert matches == ["P0_PRE_00_DRYRUN_TOLERANCE_CALIBRATION.json"]


def test_the_artifact_is_pre_data_evidence(bank: dict[str, Any]) -> None:
    assert bank["generated_pre_real_data"] is True
    assert bank["scenario"] == "DRY_D1"
    assert bank["status"] == "FROZEN"
    assert "02 §C5" in str(bank["authority"])


def test_the_artifact_carries_the_complete_bank_not_only_the_cutoffs(
    bank: dict[str, Any],
) -> None:
    """F6: the bank IS the provenance; a cutoff nobody can re-derive is not evidence."""
    reference = bank["reference_statistics"]
    mink = bank["mink_statistics"]
    assert isinstance(reference, list) and isinstance(mink, list)
    assert len(reference) == len(mink) == 200 == bank["n_trials"]
    assert all(isinstance(v, float) for v in reference + mink)
    assert len(set(reference)) == 200, "a degenerate bank would not be a distribution"


def test_the_frozen_seeds_are_the_00_34b1a_positive_calibration_family(
    bank: dict[str, Any],
) -> None:
    seeds = bank["master_seeds"]
    assert isinstance(seeds, list)
    assert seeds == list(range(810000, 810200))
    assert seeds[0] == SETTINGS.integer("dry_d1_calibration_seed_start")
    assert len(seeds) == SETTINGS.integer("dry_d1_calibration_trials") == 200


def test_all_two_hundred_seeds_completed(bank: dict[str, Any]) -> None:
    """F11: no seed was skipped, replaced, or swallowed by a caught exception."""
    reference = bank["reference_statistics"]
    seeds = bank["master_seeds"]
    assert isinstance(reference, list) and isinstance(seeds, list)
    assert len(reference) == len(seeds) == 200
    assert all(np.isfinite(v) for v in reference)
    assert all(np.isfinite(v) for v in bank["mink_statistics"])


# ------------------------------------------------------------------ F13 the schedule
def test_the_calibration_ran_the_complete_twelve_rank_schedule(bank: dict[str, Any]) -> None:
    schedule = bank["rank_schedule"]
    assert isinstance(schedule, dict)
    assert schedule["interior"] == [31, 28, 24, 20, 16, 12, 8, 6, 4, 3, 2, 1]
    assert schedule["interior_count"] == 12
    assert schedule["anchors"] == [32, 0]
    assert tuple(schedule["interior"]) == DRY_D_INTERIOR_RANKS
    assert tuple(schedule["anchors"]) == DRY_D_ANCHOR_RANKS


def test_the_fixture_can_express_rank_thirty_two(bank: dict[str, Any]) -> None:
    fixture = bank["fixture"]
    assert isinstance(fixture, dict)
    assert fixture["protected_structural_rank"] == 32
    for shape in fixture["surface"].values():
        assert min(shape) > 32, "a surface narrower than 32 cannot express the ladder"


def test_e_floor_at_thirty_one_and_twenty_eight_is_nontrivial_and_distinct() -> None:
    """F13: a rung that clamps to full rank would be a duplicate, not a truncation point."""
    trial = generate_tail_trial(
        810000, reference_shift=THRESHOLDS_SHIFT, mink_shift=THRESHOLDS_SHIFT
    )
    floors = dict(zip(trial.retained_ranks, trial.e_floor, strict=True))
    assert floors[31] > 0.0
    assert floors[28] > floors[31]
    assert len(set(trial.e_floor)) == 12


THRESHOLDS_SHIFT = SETTINGS.number("dry_d1_planted_shift")


# ------------------------------------------------------------------ F4/F5 thresholds and power
def test_both_arms_are_calibrated_independently(bank: dict[str, Any]) -> None:
    """F4: reference and Min-K% carry different noise, so one cutoff would mis-state one arm."""
    assert bank["threshold_reference"] != bank["threshold_mink"]
    assert bank["quantile"] == 97.5 == SETTINGS.number("dry_d1_quantile")
    assert bank["quantile_method"] == CALIBRATION_QUANTILE_METHOD


def test_each_cutoff_is_the_declared_percentile_of_its_own_bank(bank: dict[str, Any]) -> None:
    """The cutoff is re-derivable from the recorded bank alone."""
    for arm in ("reference", "mink"):
        values = np.asarray(bank[f"{arm}_statistics"], dtype=np.float64)
        assert float(np.percentile(values, 97.5)) == pytest.approx(
            bank[f"threshold_{arm}"], abs=1e-12
        )


def test_the_empirical_power_meets_the_02_c5_target(bank: dict[str, Any]) -> None:
    """F5: >= 0.95 synthetic detection probability, recorded rather than assumed."""
    for arm in ("reference", "mink"):
        values = np.asarray(bank[f"{arm}_statistics"], dtype=np.float64)
        power = float(np.mean(values < bank[f"threshold_{arm}"]))
        assert power == pytest.approx(bank[f"empirical_power_{arm}"])
        assert power >= 0.95, (arm, power)
    assert bank["required_power"] == 0.95


def test_the_planted_effect_is_negative_so_detection_is_a_lower_tail(
    bank: dict[str, Any],
) -> None:
    planted = bank["planted_effect"]
    assert isinstance(planted, dict)
    assert planted["sign"] == "NEGATIVE"
    assert planted["detection_rule"] == "statistic < threshold"
    assert planted["reference_shift"] == THRESHOLDS_SHIFT < 0.0


# ------------------------------------------------------------------ F7 superseded pair
def test_the_old_cutoffs_are_recorded_as_superseded_not_retained(
    bank: dict[str, Any],
) -> None:
    superseded = bank["superseded"]
    assert isinstance(superseded, dict)
    assert superseded["threshold_reference"] == SUPERSEDED_DRY_D1_REFERENCE_MAX == -0.136213
    assert superseded["threshold_mink"] == SUPERSEDED_DRY_D1_MINK_MAX == -0.135921
    assert superseded["status"] == SUPERSEDED_THRESHOLD_STATUS
    assert bank["threshold_reference"] != superseded["threshold_reference"]
    assert bank["threshold_mink"] != superseded["threshold_mink"]


# ------------------------------------------------------------------ F8 one source of truth
def test_validation_consumes_exactly_the_calibrated_values(bank: dict[str, Any]) -> None:
    assert THRESHOLDS.d1_reference == bank["threshold_reference"]
    assert THRESHOLDS.d1_mink == bank["threshold_mink"]
    assert SETTINGS.number("dry_d1_reference_threshold") == bank["threshold_reference"]
    assert SETTINGS.number("dry_d1_mink_threshold") == bank["threshold_mink"]


def test_no_source_literal_can_shadow_the_calibrated_cutoffs() -> None:
    """The only DRY-D1 numbers in source are the superseded pair, and nothing consumes them."""
    source = (REPO_ROOT / "src" / "analysis" / "dryrun.py").read_text(encoding="utf-8")
    for value in (THRESHOLDS.d1_reference, THRESHOLDS.d1_mink):
        assert repr(value) not in source
    verdict = inspect.getsource(evaluate_dry_d1)
    assert "tail_thresholds" in verdict
    assert "-0.13" not in verdict


def test_validation_does_not_regenerate_calibration() -> None:
    """F8: an ordinary run reads the frozen bank; it does not re-derive it."""
    for function in (evaluate_dry_d1, tail_thresholds):
        assert "calibrate_dry_d1_positive_control" not in inspect.getsource(function)


# ------------------------------------------------------------------ F12 traceability
def test_the_calibration_and_validation_share_one_canonical_statistic() -> None:
    """F12: only the master seed differs — there is no easier calibration-only path."""
    body = ast.parse(inspect.getsource(calibrate_dry_d1_positive_control))
    called = {
        node.func.id
        for node in ast.walk(body)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "generate_tail_trial" in called

    signature = inspect.signature(calibrate_dry_d1_positive_control)
    assert set(signature.parameters) == {"master_seeds", "planted_shift", "quantile"}
    #: nothing about the schedule, the fixture or the noise is a calibration parameter
    assert "retained_ranks" not in signature.parameters


def test_one_calibration_trial_reproduces_its_recorded_bank_value(
    bank: dict[str, Any],
) -> None:
    """The bank is re-derivable: one seed, re-run, lands on its recorded statistic."""
    seeds = bank["master_seeds"]
    reference = bank["reference_statistics"]
    mink = bank["mink_statistics"]
    assert isinstance(seeds, list) and isinstance(reference, list) and isinstance(mink, list)
    index = seeds.index(810074)
    trial = generate_tail_trial(
        810074, reference_shift=THRESHOLDS_SHIFT, mink_shift=THRESHOLDS_SHIFT
    )
    assert trial.mean_reference_delta == pytest.approx(reference[index], abs=1e-12)
    assert trial.mean_mink_delta == pytest.approx(mink[index], abs=1e-12)


# ------------------------------------------------------------------ F11 the SVD regression
def test_seed_810074_completes_deterministically() -> None:
    """LAPACK's divide-and-conquer driver fails to converge on one iterate of this seed.

    The input is finite and ordinary (max ~5.4, Frobenius ~52, condition ~1e8); the QR-iteration
    driver decomposes it without difficulty. The escalation is deterministic and never runs
    unless the primary raises, so no other seed's bytes moved [AUTH: 01 §12].
    """
    first = generate_tail_trial(
        810074, reference_shift=THRESHOLDS_SHIFT, mink_shift=THRESHOLDS_SHIFT
    )
    second = generate_tail_trial(
        810074, reference_shift=THRESHOLDS_SHIFT, mink_shift=THRESHOLDS_SHIFT
    )
    assert first.as_dict() == second.as_dict()
    assert all(np.isfinite(v) for v in first.e_floor)
    assert len(first.e_floor) == 12


def test_the_svd_escalation_is_deterministic_and_refuses_a_malformed_iterate() -> None:
    from src.recovery.solvers import RecoveryError, _svd

    rng = np.random.default_rng(0)
    ordinary = rng.normal(size=(40, 36))
    left, singular, right = _svd(ordinary)
    assert np.allclose((left * singular) @ right, ordinary)
    reference = np.linalg.svd(ordinary, full_matrices=False)[1]
    assert np.allclose(singular, reference), "the primary driver must still be what runs"

    broken = ordinary.copy()
    broken[0, 0] = np.nan
    with pytest.raises(RecoveryError, match="NaN or Inf"):
        _svd(broken)


# ------------------------------------------------------------------ F10 validation
def test_dry_d1_validation_passes_under_the_new_cutoffs() -> None:
    """The validation fixture is untouched; only the cutoff was recalibrated."""
    trial = generate_tail_trial(4242, reference_shift=-0.30, mink_shift=-0.30)
    assert trial.mean_reference_delta < THRESHOLDS.d1_reference
    assert trial.mean_mink_delta < THRESHOLDS.d1_mink
    assert evaluate_dry_d1(trial).status == "TAIL_EFFECT_CORROBORATED"
