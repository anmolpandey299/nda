"""S01 experiment identity vs run identity [AUTH: 01 §15, §36].

The 01 §15 determinism requirement and the 01 §36 "a rerun gets a new RUN_ID" requirement
are satisfied at two different levels; these tests pin both halves and the boundary.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from check_repo_invariants import RUN_ID_INPUTS

from src.provenance.identity import (
    PROVENANCE_INPUTS,
    ProvenanceError,
    ScientificProvenance,
    attempt_directory,
    attempt_is_claimed,
    claimed_run_ids,
    experiment_id,
    next_attempt,
    run_id,
    run_manifest_path,
)

BASE = ScientificProvenance(
    git_commit_sha="a" * 40,
    spec_sha256="b" * 64,
    execution_lock_sha256="c" * 64,
    config_sha256="d" * 64,
    model_revision="e" * 40,
    data_manifest_sha256="f" * 64,
    environment_lock_sha256="0" * 64,
    training_seed=101,
)


def _with(**changes: Any) -> ScientificProvenance:
    return dataclasses.replace(BASE, **changes)


def test_provenance_vocabulary_matches_the_invariant_checker() -> None:
    """One vocabulary for the eight 01 §15 inputs; src/ and scripts/ cannot drift apart."""
    assert PROVENANCE_INPUTS == RUN_ID_INPUTS
    assert set(PROVENANCE_INPUTS) == {f.name for f in dataclasses.fields(ScientificProvenance)}


# ------------------------------------------------------------------ contract 11
def test_identical_provenance_gives_one_deterministic_identity() -> None:
    assert experiment_id(BASE) == experiment_id(_with())
    assert len(experiment_id(BASE)) == 64


# ------------------------------------------------------------------ contracts 6-10
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "9" * 40),
        ("data_manifest_sha256", "1" * 64),
        ("config_sha256", "2" * 64),
        ("environment_lock_sha256", "3" * 64),
        ("training_seed", 202),
        ("git_commit_sha", "4" * 40),
        ("spec_sha256", "5" * 64),
        ("execution_lock_sha256", "6" * 64),
    ],
)
def test_every_provenance_input_changes_the_identity(field: str, value: object) -> None:
    assert experiment_id(_with(**{field: value})) != experiment_id(BASE)


def test_environment_identity_is_the_full_lock_not_the_lockfile() -> None:
    """The RUN_ID input is ENVIRONMENT_LOCK_SHA256, which already covers uv.lock, image
    digest, CUDA, driver, torch, python and GPU model [AUTH: 01 §12(5)-(9), §15]."""
    with pytest.raises(ProvenanceError, match="environment_lock_sha256"):
        experiment_id(_with(environment_lock_sha256="TBD_REQUIRES_HARDWARE"))


# ------------------------------------------------------------------ fail-closed provenance
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("environment_lock_sha256", "TBD_REQUIRES_HARDWARE", "TBD"),
        ("spec_sha256", "short", "SHA256"),
        ("git_commit_sha", "abc", "commit SHA"),
        ("model_revision", "main", "floating branch"),
        ("model_revision", "  ", "empty"),
        ("training_seed", "101", "integer"),
        ("training_seed", True, "integer"),
    ],
)
def test_unusable_provenance_cannot_become_an_identity(
    field: str, value: object, match: str
) -> None:
    with pytest.raises(ProvenanceError, match=match):
        experiment_id(_with(**{field: value}))


# ------------------------------------------------------------------ run identity
def test_run_identity_is_deterministic_per_attempt() -> None:
    experiment = experiment_id(BASE)
    assert run_id(experiment, 1) == run_id(experiment, 1)
    assert run_id(experiment, 1) != run_id(experiment, 2)
    assert run_id(experiment, 1) != experiment


def test_two_experiments_never_share_a_run_identity() -> None:
    left = experiment_id(BASE)
    right = experiment_id(_with(training_seed=202))
    assert run_id(left, 1) != run_id(right, 1)


@pytest.mark.parametrize("attempt", [0, -1, "1", True])
def test_a_bad_attempt_index_fails_closed(attempt: object) -> None:
    with pytest.raises(ProvenanceError, match="attempt"):
        run_id(experiment_id(BASE), attempt)  # type: ignore[arg-type]


def test_run_identity_requires_a_real_experiment_identity() -> None:
    with pytest.raises(ProvenanceError, match="experiment id"):
        run_id("not-a-digest", 1)


def test_next_attempt_probes_upward_from_one(tmp_path: Path) -> None:
    experiment = experiment_id(BASE)
    assert next_attempt(tmp_path, experiment) == 1
    attempt_directory(tmp_path, experiment, 1).mkdir(parents=True)
    assert next_attempt(tmp_path, experiment) == 2
    attempt_directory(tmp_path, experiment, 2).mkdir(parents=True)
    assert next_attempt(tmp_path, experiment) == 3


def test_attempt_allocation_is_per_experiment(tmp_path: Path) -> None:
    first = experiment_id(BASE)
    second = experiment_id(_with(training_seed=202))
    attempt_directory(tmp_path, first, 1).mkdir(parents=True)
    assert next_attempt(tmp_path, first) == 2
    assert next_attempt(tmp_path, second) == 1


def test_attempt_probing_is_bounded(tmp_path: Path) -> None:
    experiment = experiment_id(BASE)
    attempt_directory(tmp_path, experiment, 1).mkdir(parents=True)
    with pytest.raises(ProvenanceError, match="more than 1 attempts"):
        next_attempt(tmp_path, experiment, limit=1)


# ------------------------------------------------------------------ F1: two-record occupancy
def _claim_manifest(root: Path, experiment: str, attempt: int) -> Path:
    path = run_manifest_path(root, run_id(experiment, attempt))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"status": "FAILED_IMPLEMENTATION"}', encoding="utf-8")
    return path


def test_a_surviving_manifest_alone_claims_the_attempt(tmp_path: Path) -> None:
    """Deleting a failed attempt's directory must not free its index [AUTH: 01 §15, §36]."""
    experiment = experiment_id(BASE)
    _claim_manifest(tmp_path, experiment, 1)
    assert attempt_is_claimed(tmp_path, experiment, 1) is True
    assert next_attempt(tmp_path, experiment) == 2


def test_a_surviving_directory_alone_claims_the_attempt(tmp_path: Path) -> None:
    experiment = experiment_id(BASE)
    attempt_directory(tmp_path, experiment, 1).mkdir(parents=True)
    assert attempt_is_claimed(tmp_path, experiment, 1) is True
    assert next_attempt(tmp_path, experiment) == 2


def test_occupancy_skips_a_gap_rather_than_stopping(tmp_path: Path) -> None:
    experiment = experiment_id(BASE)
    _claim_manifest(tmp_path, experiment, 1)
    _claim_manifest(tmp_path, experiment, 3)
    assert next_attempt(tmp_path, experiment) == 2
    assert attempt_is_claimed(tmp_path, experiment, 3) is True


def test_claimed_ids_read_both_namespaces(tmp_path: Path) -> None:
    experiment = experiment_id(BASE)
    _claim_manifest(tmp_path, experiment, 1)
    attempt_directory(tmp_path, experiment, 2).mkdir(parents=True)
    assert claimed_run_ids(tmp_path) == {run_id(experiment, 1), run_id(experiment, 2)}


def test_claimed_ids_on_an_empty_root(tmp_path: Path) -> None:
    assert claimed_run_ids(tmp_path) == set()
