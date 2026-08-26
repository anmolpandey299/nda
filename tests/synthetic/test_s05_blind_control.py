"""S05.7 — planted-truth blind split / deduplication leakage control [AUTH: 00 §6.4].

Two planted worlds: a split that is genuinely random (the control must call it clean) and a
split that leaks (the control must not). The second is what makes the first meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from blockc_fixtures import synthetic_text, tiny_corpus

from src.data.blind import (
    CONTROL_VERSION,
    BlindControlError,
    area_under_curve,
    duplicate_leakage_across_partitions,
    run_blind_control,
)
from src.data.corpus import build_master_corpus
from src.data.settings import corpus_settings
from src.materials import material_integer, material_number, material_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = corpus_settings(REPO_ROOT)
CV_SEED = material_integer(SETTINGS, "blind_control_cv_seed")
CONFIRMATORY_SEED = material_integer(SETTINGS, "blind_control_confirmatory_seed")
FOLDS = material_integer(SETTINGS, "blind_control_folds")
TOLERANCE = material_number(SETTINGS, "blind_control_clean_tolerance")
BAND = (0.47, 0.53)
C = material_number(SETTINGS, "blind_control_regularisation", allow_provisional=True)


def control(members: list[str], non_members: list[str], *, seed: int = CV_SEED) -> Any:
    return run_blind_control(
        member_texts=members,
        non_member_texts=non_members,
        cv_seed=seed,
        n_folds=FOLDS,
        clean_tolerance=TOLERANCE,
        investigation_band=BAND,
        regularisation=C,
    )


# ------------------------------------------------------------------ the estimator
def test_the_auroc_is_the_rank_identity_not_a_threshold_sweep() -> None:
    """A second ROC path would violate 00 §34A.2; the Mann-Whitney identity avoids one."""
    import numpy as np

    scores = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int_)
    assert area_under_curve(scores, labels) == pytest.approx(0.75)

    tied = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    assert area_under_curve(tied, labels) == pytest.approx(0.5)


def test_a_perfectly_separable_set_scores_one() -> None:
    import numpy as np

    scores = np.array([0.0, 0.1, 0.9, 1.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int_)
    assert area_under_curve(scores, labels) == pytest.approx(1.0)


# ------------------------------------------------------------------ planted clean split
def test_a_randomly_assigned_split_is_called_clean() -> None:
    records = tiny_corpus(400)
    corpus = build_master_corpus(
        records,
        sizes={
            "TRAIN_CANDIDATES": 120,
            "CALIBRATION_NONMEMBERS": 60,
            "EVAL_NONMEMBERS": 120,
            "RESERVE": 60,
        },
        split_seed=20260820,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    by_id = {record.record_id: record.text for record in corpus.retained}
    members = [by_id[rid] for rid in corpus.partitions["TRAIN_CANDIDATES"]]
    non_members = [by_id[rid] for rid in corpus.partitions["EVAL_NONMEMBERS"]]
    result = control(members, non_members)
    assert result.is_clean, result.auroc
    assert result.verdict == "CLEAN"
    assert result.version == CONTROL_VERSION


def test_the_confirmatory_seed_agrees_with_the_first() -> None:
    records = tiny_corpus(300)
    members = [r.text for r in records[:150]]
    non_members = [r.text for r in records[150:]]
    first = control(members, non_members, seed=CV_SEED)
    second = control(members, non_members, seed=CONFIRMATORY_SEED)
    assert first.is_clean and second.is_clean
    assert first.cv_seed != second.cv_seed


def test_the_control_is_deterministic_for_one_seed() -> None:
    records = tiny_corpus(200)
    members = [r.text for r in records[:100]]
    non_members = [r.text for r in records[100:]]
    assert control(members, non_members).auroc == control(members, non_members).auroc


# ------------------------------------------------------------------ planted leakage
def test_a_planted_preprocessing_artifact_is_detected() -> None:
    """The mutation this kills: a control that cannot see a split it should refuse."""
    records = tiny_corpus(300)
    members = [f"MEMBER-COHORT-MARKER {r.text}" for r in records[:150]]
    non_members = [r.text for r in records[150:]]
    result = control(members, non_members)
    assert not result.is_clean, result.auroc
    assert result.verdict == "BLIND_SPLIT_INVESTIGATION"


def test_a_planted_topic_imbalance_is_detected() -> None:
    members = [synthetic_text(i)[1] for i in range(150)]
    non_members = [
        "Editorial commentary on funding policy and peer review workload, item "
        f"{index}, containing none of the study vocabulary."
        for index in range(150)
    ]
    assert not control(members, non_members).is_clean


def test_the_clean_tolerance_is_what_decides_cleanliness() -> None:
    records = tiny_corpus(200)
    members = [f"MEMBER-COHORT-MARKER {r.text}" for r in records[:100]]
    non_members = [r.text for r in records[100:]]
    strict = control(members, non_members)
    loose = run_blind_control(
        member_texts=members,
        non_member_texts=non_members,
        cv_seed=CV_SEED,
        n_folds=FOLDS,
        clean_tolerance=0.5,
        investigation_band=(0.0, 1.0),
        regularisation=C,
    )
    assert not strict.is_clean and loose.is_clean


# ------------------------------------------------------------------ duplicate audit
def test_the_duplicate_audit_finds_a_record_shared_across_partitions() -> None:
    """The mutation this kills: the same normalised text in train and eval."""
    findings = duplicate_leakage_across_partitions(
        {
            "TRAIN_CANDIDATES": {"A1": "h1", "A2": "h2"},
            "EVAL_NONMEMBERS": {"B1": "h2", "B2": "h3"},
        }
    )
    assert findings == [("h2", "EVAL_NONMEMBERS", "B1", "A2")] or findings == [
        ("h2", "TRAIN_CANDIDATES", "A2", "B1")
    ]


def test_the_duplicate_audit_is_silent_on_a_disjoint_split() -> None:
    assert (
        duplicate_leakage_across_partitions(
            {"TRAIN_CANDIDATES": {"A1": "h1"}, "EVAL_NONMEMBERS": {"B1": "h2"}}
        )
        == []
    )


def test_a_repeated_hash_inside_one_partition_is_not_cross_partition_leakage() -> None:
    assert duplicate_leakage_across_partitions({"TRAIN_CANDIDATES": {"A1": "h1", "A2": "h1"}}) == []


# ------------------------------------------------------------------ fail-closed
def test_a_one_class_control_is_refused() -> None:
    with pytest.raises(BlindControlError):
        control([r.text for r in tiny_corpus(10)], [])


def test_fewer_than_two_folds_is_refused() -> None:
    records = tiny_corpus(20)
    with pytest.raises(BlindControlError):
        run_blind_control(
            member_texts=[r.text for r in records[:10]],
            non_member_texts=[r.text for r in records[10:]],
            cv_seed=CV_SEED,
            n_folds=1,
            clean_tolerance=TOLERANCE,
            investigation_band=BAND,
            regularisation=C,
        )


def test_the_investigation_band_comes_from_config() -> None:
    assert material_text(SETTINGS, "exact_duplicate_key") == "NORMALISED_TEXT_SHA256"
    from src.materials import material

    assert material(SETTINGS, "blind_control_investigation_band") == [0.47, 0.53]
