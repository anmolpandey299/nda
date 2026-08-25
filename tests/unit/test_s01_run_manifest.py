"""S01 run-manifest contract: seeds, config identity, metrics binding, GPU identity.

These assert the manifest *contract* directly, so a hand-written or hand-edited manifest is
refused on its own terms, not only when it happens to pass through the run store
[AUTH: 01 §15, §16, §19, §30, §36].
"""

from __future__ import annotations

from typing import Any

import pytest
from check_repo_invariants import RUN_MANIFEST_REQUIRED_FIELDS as CHECKER_FIELDS

from src.provenance.config import resolved_config_sha256
from src.provenance.identity import ScientificProvenance, experiment_id, run_id
from src.provenance.run_manifest import (
    ALL_REQUIRED_FIELDS,
    MASTER_SEED_FIELD,
    NOT_APPLICABLE,
    RUN_MANIFEST_ADDITIONAL_REQUIRED,
    RUN_MANIFEST_REQUIRED_FIELDS,
    SEED_FIELDS,
    RunManifestError,
    build_run_manifest,
    validate_run_manifest,
)

CONFIG: dict[str, Any] = {"lora_rank": 32, "merge_alphas": {"h": 0.5}}
PROVENANCE = ScientificProvenance(
    git_commit_sha="a" * 40,
    spec_sha256="b" * 64,
    execution_lock_sha256="c" * 64,
    config_sha256=resolved_config_sha256(CONFIG),
    model_revision="e" * 40,
    data_manifest_sha256="f" * 64,
    environment_lock_sha256="0" * 64,
    training_seed=101,
)


def _manifest(**overrides: Any) -> dict[str, Any]:
    experiment = experiment_id(PROVENANCE)
    document = build_run_manifest(
        run_id=run_id(experiment, 1),
        experiment_id=experiment,
        attempt=1,
        provenance=PROVENANCE,
        resolved_config=CONFIG,
        seeds={"python_rng": 1, "numpy_rng": 2},
        precision="float64",
        tokenizer_hash="9" * 64,
        stdout_log_path="logs/x/stdout.log",
        stderr_log_path="logs/x/stderr.log",
        wall_clock_start="2026-08-25T09:00:00+00:00",
    )
    document.update(overrides)
    return document


def _succeeded(**overrides: Any) -> dict[str, Any]:
    return _manifest(
        status="SUCCESS",
        exit_code=0,
        wall_clock_end="2026-08-25T09:05:00+00:00",
        **overrides,
    )


# ------------------------------------------------------------------ field vocabulary
def test_the_contract_is_a_superset_of_the_s00_minimum() -> None:
    """01 §16 asks for the GPU UUID; the S00 checker's tuple omits it. This contract adds it
    without ever dropping a field the S00 gate requires."""
    assert RUN_MANIFEST_REQUIRED_FIELDS == CHECKER_FIELDS
    assert set(CHECKER_FIELDS) < set(ALL_REQUIRED_FIELDS)
    assert RUN_MANIFEST_ADDITIONAL_REQUIRED == ("gpu_uuid",)
    assert "gpu_uuid" not in CHECKER_FIELDS


def test_a_complete_running_manifest_validates() -> None:
    assert validate_run_manifest(_manifest()) == []


@pytest.mark.parametrize("field", ALL_REQUIRED_FIELDS)
def test_every_required_field_is_required(field: str) -> None:
    document = _manifest()
    document.pop(field)
    assert validate_run_manifest(document) != []


# ------------------------------------------------------------------ F2-E seeds
def test_the_master_seed_is_recorded_from_provenance() -> None:
    assert _manifest()["seeds"][MASTER_SEED_FIELD] == PROVENANCE.training_seed


def test_a_conflicting_master_seed_is_refused() -> None:
    experiment = experiment_id(PROVENANCE)
    with pytest.raises(RunManifestError, match="disagrees with the provenance"):
        build_run_manifest(
            run_id=run_id(experiment, 1),
            experiment_id=experiment,
            attempt=1,
            provenance=PROVENANCE,
            resolved_config=CONFIG,
            seeds={MASTER_SEED_FIELD: 999},
            precision="float64",
            tokenizer_hash="9" * 64,
            stdout_log_path="logs/x/stdout.log",
            stderr_log_path="logs/x/stderr.log",
            wall_clock_start="2026-08-25T09:00:00+00:00",
        )


def test_an_undocumented_seed_family_is_refused() -> None:
    """`one_global_seed` is exactly what 01 §30 forbids as a substitute for named families."""
    problems = validate_run_manifest(_manifest(seeds={"one_global_seed": 101}))
    assert any("undocumented seed field" in p for p in problems), problems


def test_an_evidentiary_run_must_record_the_master_seed() -> None:
    document = _manifest(seeds={"python_rng": 1})
    problems = validate_run_manifest(document)
    assert any(MASTER_SEED_FIELD in p for p in problems), problems


def test_a_non_integer_seed_is_refused() -> None:
    assert validate_run_manifest(_manifest(seeds={MASTER_SEED_FIELD: "101"})) != []


def test_empty_seeds_are_refused() -> None:
    assert validate_run_manifest(_manifest(seeds={})) != []


def test_the_seed_vocabulary_is_the_01_30_families() -> None:
    assert MASTER_SEED_FIELD in SEED_FIELDS
    for family in (
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
        "torch_cuda_rng",
        "data_order",
        "canary_inclusion",
        "lora_init",
        "dp_noise",
    ):
        assert family in SEED_FIELDS


# ------------------------------------------------------------------ F2-C config identity
def test_a_config_identity_that_does_not_hash_the_config_is_refused() -> None:
    document = _manifest()
    document["config"] = {**CONFIG, "lora_rank": 8}
    problems = validate_run_manifest(document)
    assert any("does not hash the stored config" in p for p in problems), problems


def test_a_malformed_config_identity_is_refused() -> None:
    assert validate_run_manifest(_manifest(config_sha256="nope")) != []


# ------------------------------------------------------------------ F5 metrics
def test_success_with_an_unhashed_metrics_path_is_refused() -> None:
    problems = validate_run_manifest(_succeeded(metrics_paths=["results/p0/table.json"]))
    assert any("unhashed metrics path" in p for p in problems), problems


def test_success_with_a_hash_bound_metrics_path_validates() -> None:
    document = _succeeded(
        metrics_paths=["results/p0/table.json"],
        artifact_paths=["results/p0/table.json"],
        artifact_hashes={"results/p0/table.json": "7" * 64},
    )
    assert validate_run_manifest(document) == []


# ------------------------------------------------------------------ F6 GPU identity
def test_a_cpu_run_needs_no_gpu_identity() -> None:
    document = _succeeded()
    assert document["gpu_model"] == NOT_APPLICABLE
    assert document["gpu_uuid"] == NOT_APPLICABLE
    assert validate_run_manifest(document) == []


def test_hardware_without_a_uuid_is_refused() -> None:
    problems = validate_run_manifest(_manifest(gpu_model="NVIDIA H100 80GB HBM3"))
    assert any("requires the GPU UUID" in p for p in problems), problems


@pytest.mark.parametrize("uuid", [NOT_APPLICABLE, "TBD_REQUIRES_HARDWARE"])
def test_hardware_with_a_placeholder_uuid_is_refused(uuid: str) -> None:
    assert validate_run_manifest(_manifest(gpu_model="NVIDIA H100", gpu_uuid=uuid)) != []


def test_a_cpu_run_may_not_claim_a_gpu_uuid() -> None:
    problems = validate_run_manifest(_manifest(gpu_uuid="GPU-abc"))
    assert any("no GPU to identify" in p for p in problems), problems


def test_hardware_with_a_real_uuid_validates() -> None:
    document = _succeeded(
        gpu_model="NVIDIA H100 80GB HBM3",
        gpu_uuid="GPU-1f2e3d4c-5b6a-7988-9a0b-1c2d3e4f5061",
        cuda="13.0",
        pytorch="2.13.0+cu130",
    )
    assert validate_run_manifest(document) == []


def test_an_empty_gpu_field_is_refused() -> None:
    assert validate_run_manifest(_manifest(gpu_uuid="   ")) != []
