"""Block A end to end: config -> derived provenance -> identity -> immutable run attempt.

Multiple modules, tiny fixtures, no network [AUTH: 01 §22 Integration, §15, §16, §35(3), §36].

The workspace here is a real, hermetic git repository carrying a captured environment
manifest, because `begin_run` no longer believes a caller about the commit, the dirty state,
the config identity or the environment: it derives them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from check_repo_invariants import RUN_MANIFEST_REQUIRED_FIELDS

from src.provenance.config import ConfigSchema, resolve_and_persist
from src.provenance.hashing import JSONValue, read_canonical_json, sha256_file
from src.provenance.identity import (
    ProvenanceError,
    ScientificProvenance,
    experiment_id,
    run_id,
)
from src.provenance.repository import (
    ENVIRONMENT_LOCK_COMPONENTS,
    environment_lock_sha256,
    selected_environment_identity,
)
from src.provenance.run_manifest import RUNNING, TERMINAL_STATES, RunManifestError
from src.provenance.runs import (
    DirtyTreeError,
    RunAttempt,
    RunDirectoryExists,
    RunFinalizedError,
    RunStoreError,
    UnverifiedProvenanceError,
    attempts_of,
    begin_run,
    load_attempt,
)

RUN_SCHEMA = ConfigSchema(
    name="block-a.run.v1",
    required=frozenset({"lora_rank", "training_seed", "merge_alphas"}),
)
#: 01 §30 seed families. The master seed is added from provenance by the manifest builder.
SEED_SET: dict[str, int] = {"python_rng": 101, "numpy_rng": 101, "data_order": 7, "lora_init": 11}
TRAINING_SEED = 101


@dataclass(frozen=True)
class Workspace:
    """A repository whose facts are real enough to be derived rather than asserted."""

    root: Path
    provenance: ScientificProvenance
    resolved_config: dict[str, JSONValue]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


#: A resolved capture with real-shaped hardware, so a run can be checked against it.
ACCEPTED_ENVIRONMENT: dict[str, str] = {
    "uv_lock_sha256": "1" * 64,
    "docker_image_digest": "sha256:" + "2" * 64,
    "cuda_runtime": "13.0",
    "cuda_driver": "580.126.09",
    "torch_version": "2.13.0",
    "torch_cuda_build": "cu130",
    "python_version": "3.13.2",
    "gpu_model": "NVIDIA H100 80GB HBM3",
}
ACCEPTED_GPU_UUID = "GPU-1f2e3d4c-5b6a-7988-9a0b-1c2d3e4f5061"


def _write_environment_manifest(root: Path, **overrides: str) -> str:
    """A fully resolved capture, hashed by S00's scheme [AUTH: 01 §12(5)-(9), §15]."""
    manifest: dict[str, object] = {**ACCEPTED_ENVIRONMENT, "gpu_uuid": ACCEPTED_GPU_UUID}
    manifest.update(overrides)
    assert set(ENVIRONMENT_LOCK_COMPONENTS) <= set(manifest)
    identity = environment_lock_sha256(manifest)
    manifest["environment_lock_sha256"] = identity
    directory = root / "manifests" / "environments"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{identity}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return identity


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "repo"
    (root / "configs" / "p0").mkdir(parents=True)
    (root / "configs" / "p0" / "block_a.json").write_text(
        json.dumps({"lora_rank": 32, "training_seed": TRAINING_SEED, "merge_alphas": {"h": 0.5}}),
        encoding="utf-8",
    )
    _git(root, "init", "-b", "stage/test", "-q")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "config", "user.name", "fixture")
    _git(root, "add", "-A")
    _git(root, "-c", "core.hooksPath=", "commit", "-q", "-m", "config")
    head = _git(root, "rev-parse", "HEAD").strip()

    environment = _write_environment_manifest(root)
    document, config_sha256 = resolve_and_persist(
        root / "configs" / "p0" / "block_a.json",
        root / "artifacts" / "resolved" / "block_a.json",
        config_root=root / "configs",
        schema=RUN_SCHEMA,
    )
    provenance = ScientificProvenance(
        git_commit_sha=head,
        spec_sha256="b" * 64,
        execution_lock_sha256="c" * 64,
        config_sha256=config_sha256,
        model_revision="e" * 40,
        data_manifest_sha256="f" * 64,
        environment_lock_sha256=environment,
        training_seed=TRAINING_SEED,
    )
    return Workspace(root=root, provenance=provenance, resolved_config=dict(document))


def _begin(workspace: Workspace, **kwargs: object) -> RunAttempt:
    provenance = kwargs.pop("provenance", workspace.provenance)
    assert isinstance(provenance, ScientificProvenance)
    config = kwargs.pop("resolved_config", workspace.resolved_config)
    assert isinstance(config, dict)
    return begin_run(
        workspace.root,
        provenance,
        resolved_config=config,
        seeds=SEED_SET,
        precision="float64",
        tokenizer_hash="9" * 64,
        **kwargs,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------ contract 12
def test_the_first_attempt_is_created_before_any_work(workspace: Workspace) -> None:
    run = _begin(workspace)
    assert run.attempt == 1
    assert run.directory.is_dir()
    assert run.manifest_path.is_file()
    assert run.log_directory.is_dir()

    manifest = run.read_manifest()
    assert manifest["status"] == RUNNING
    assert manifest["run_id"] == run.run_id == run_id(run.experiment_id, 1)
    assert manifest["experiment_id"] == experiment_id(workspace.provenance)
    assert manifest["wall_clock_end"] is None
    assert manifest["config"] == workspace.resolved_config
    assert manifest["seeds"] == {**SEED_SET, "training_seed": TRAINING_SEED}
    assert manifest["git_dirty"] is False
    assert manifest["git_dirty_source"] == "derived"
    assert [f for f in RUN_MANIFEST_REQUIRED_FIELDS if f not in manifest] == []
    assert "gpu_uuid" in manifest
    assert run.verify() == []


def test_the_manifest_lives_where_authority_requires(workspace: Workspace) -> None:
    run = _begin(workspace)
    assert run.manifest_path == workspace.root / "manifests" / "runs" / f"{run.run_id}.json"
    assert run.directory == workspace.root / "artifacts" / "runs" / run.run_id
    assert load_attempt(workspace.root, run.run_id) == run


# ------------------------------------------------------------------ contract 13
def test_an_existing_attempt_can_never_overwrite_itself(workspace: Workspace) -> None:
    run = _begin(workspace)
    (run.directory / "output.json").write_text('{"value": 1}', encoding="utf-8")
    before = sha256_file(run.manifest_path)

    with pytest.raises(RunDirectoryExists):
        _begin(workspace, attempt=1)

    assert sha256_file(run.manifest_path) == before
    assert (run.directory / "output.json").read_text(encoding="utf-8") == '{"value": 1}'


# ------------------------------------------------------------------ F1 regression
def test_a_surviving_manifest_alone_reserves_the_run_id(workspace: Workspace) -> None:
    """The exact reviewer counterexample: attempt 1 fails, its run directory disappears, and
    the next rerun must not be handed attempt 1 again [AUTH: 01 §15, §36]."""
    import shutil

    first = _begin(workspace)
    first.finalize(status="FAILED_IMPLEMENTATION", exit_code=1)
    failed_manifest = first.manifest_path.read_bytes()
    failed_digest = hashlib.sha256(failed_manifest).hexdigest()

    shutil.rmtree(first.directory)
    assert not first.directory.exists()
    assert first.manifest_path.is_file()

    second = _begin(workspace)

    assert second.attempt == 2, "attempt 1 was handed out again after its directory vanished"
    assert second.run_id != first.run_id
    assert hashlib.sha256(first.manifest_path.read_bytes()).hexdigest() == failed_digest
    assert json.loads(failed_manifest)["status"] == "FAILED_IMPLEMENTATION"
    with pytest.raises(RunDirectoryExists):
        _begin(workspace, attempt=1)


def test_a_surviving_directory_alone_reserves_the_run_id(workspace: Workspace) -> None:
    first = _begin(workspace)
    first.manifest_path.unlink()
    assert first.directory.is_dir()

    second = _begin(workspace)
    assert second.attempt == 2
    with pytest.raises(RunDirectoryExists):
        _begin(workspace, attempt=1)


def test_a_gap_does_not_hide_a_later_failed_attempt(workspace: Workspace) -> None:
    """attempts_of must not stop at the first missing index [AUTH: 01 §36]."""
    import shutil

    first = _begin(workspace)
    first.finalize(status="FAILED_RESOURCE", exit_code=2)
    second = _begin(workspace)
    second.finalize(status="FAILED_ENVIRONMENT", exit_code=3)
    third = _begin(workspace)
    assert [a.attempt for a in attempts_of(workspace.root, first.experiment_id)] == [1, 2, 3]

    shutil.rmtree(second.directory)
    second.manifest_path.unlink()

    remaining = [a.attempt for a in attempts_of(workspace.root, first.experiment_id)]
    assert remaining == [1, 3], "a gap hid the surviving attempts above it"
    assert third.status() == RUNNING


# ------------------------------------------------------------------ contracts 14, 15, 16
def test_a_failed_attempt_survives_its_rerun(workspace: Workspace) -> None:
    first = _begin(workspace)
    artifact_rel = f"artifacts/runs/{first.run_id}/partial.json"
    (workspace.root / artifact_rel).write_text('{"partial": true}', encoding="utf-8")
    record = first.add_artifact(artifact_rel)
    first.finalize(status="FAILED_IMPLEMENTATION", exit_code=1)

    assert first.status() == "FAILED_IMPLEMENTATION"
    assert first.is_finalized()
    failed_manifest = sha256_file(first.manifest_path)
    failed_artifact = sha256_file(workspace.root / artifact_rel)

    second = _begin(workspace)
    assert second.attempt == 2
    assert second.run_id != first.run_id
    assert second.experiment_id == first.experiment_id  # same science, different execution
    assert second.directory != first.directory

    assert sha256_file(first.manifest_path) == failed_manifest
    assert sha256_file(workspace.root / artifact_rel) == failed_artifact
    assert record.sha256 == failed_artifact
    assert first.verify() == []
    assert [a.attempt for a in attempts_of(workspace.root, first.experiment_id)] == [1, 2]


def test_every_terminal_state_is_accepted_and_nothing_else_is(workspace: Workspace) -> None:
    for index, status in enumerate(sorted(TERMINAL_STATES), start=1):
        run = _begin(workspace, attempt=index)
        run.finalize(status=status, exit_code=0 if status == "SUCCESS" else 3)
        assert run.status() == status
    run = _begin(workspace)
    with pytest.raises(RunStoreError, match="terminal state"):
        run.finalize(status="MOSTLY_FINE", exit_code=0)


# ------------------------------------------------------------------ F2: derived, not believed
def test_a_wrong_git_commit_cannot_start_an_evidentiary_run(workspace: Workspace) -> None:
    import dataclasses

    forged = dataclasses.replace(workspace.provenance, git_commit_sha="d" * 40)
    with pytest.raises(UnverifiedProvenanceError, match="HEAD is"):
        _begin(workspace, provenance=forged)
    assert not (workspace.root / "manifests" / "runs").exists()


def test_a_false_clean_tree_assertion_cannot_start_an_evidentiary_run(
    workspace: Workspace,
) -> None:
    (workspace.root / "configs" / "p0" / "sneaked.json").write_text("{}", encoding="utf-8")
    with pytest.raises((DirtyTreeError, UnverifiedProvenanceError), match="dirty"):
        _begin(workspace, git_dirty=False)
    assert not (workspace.root / "manifests" / "runs").exists()


def test_a_config_hash_that_does_not_hash_the_stored_config_is_refused(
    workspace: Workspace,
) -> None:
    other_config = {**workspace.resolved_config, "lora_rank": 8}
    with pytest.raises(UnverifiedProvenanceError, match="does not hash the config"):
        _begin(workspace, resolved_config=other_config)


def test_a_uv_lock_digest_cannot_impersonate_the_environment_identity(
    workspace: Workspace,
) -> None:
    """A 64-hex string is not an environment. The digest must name a captured manifest that
    recomputes to it [AUTH: 01 §12(5)-(9), §15, §32]."""
    import dataclasses

    uv_lock_only = hashlib.sha256(b"pretend uv.lock bytes").hexdigest()
    forged = dataclasses.replace(workspace.provenance, environment_lock_sha256=uv_lock_only)
    with pytest.raises(UnverifiedProvenanceError, match="not a full environment identity"):
        _begin(workspace, provenance=forged)


def test_an_undocumented_global_seed_cannot_start_an_evidentiary_run(
    workspace: Workspace,
) -> None:
    with pytest.raises(RunManifestError, match="undocumented seed field"):
        begin_run(
            workspace.root,
            workspace.provenance,
            resolved_config=workspace.resolved_config,
            seeds={"one_global_seed": 101},
            precision="float64",
            tokenizer_hash="9" * 64,
        )


def test_unusable_provenance_never_becomes_an_evidentiary_run(workspace: Workspace) -> None:
    import dataclasses

    unresolved = dataclasses.replace(
        workspace.provenance, environment_lock_sha256="TBD_REQUIRES_HARDWARE"
    )
    with pytest.raises(ProvenanceError):
        _begin(workspace, provenance=unresolved)
    assert not (workspace.root / "manifests" / "runs").exists()


def test_a_non_evidentiary_run_outside_a_repository_must_declare_its_state(
    tmp_path: Path, workspace: Workspace
) -> None:
    with pytest.raises(UnverifiedProvenanceError, match="not a repository"):
        begin_run(
            tmp_path / "not_a_repo",
            workspace.provenance,
            resolved_config=workspace.resolved_config,
            seeds=SEED_SET,
            precision="float64",
            tokenizer_hash="9" * 64,
            evidentiary=False,
        )


def test_a_dirty_run_cannot_finalise_as_success(workspace: Workspace) -> None:
    (workspace.root / "configs" / "p0" / "sneaked.json").write_text("{}", encoding="utf-8")
    run = _begin(workspace, evidentiary=False)
    assert run.read_manifest()["git_dirty"] is True
    with pytest.raises(RunManifestError, match="dirty tree"):
        run.finalize(status="SUCCESS", exit_code=0)
    assert run.status() == RUNNING  # the refusal did not half-write a terminal record
    run.finalize(status="FAILED_ENVIRONMENT", exit_code=2)
    assert run.status() == "FAILED_ENVIRONMENT"


def test_a_success_needs_exit_code_zero(workspace: Workspace) -> None:
    run = _begin(workspace)
    with pytest.raises(RunManifestError, match="exit_code 0"):
        run.finalize(status="SUCCESS", exit_code=7)


def test_an_unmeasured_hardware_field_cannot_survive_into_a_success(
    workspace: Workspace,
) -> None:
    """An unmeasured value is not a result. Checked on a non-evidentiary run so the terminal
    contract is what fails, not the environment binding [AUTH: 03 §8; 01 §16]."""
    run = _begin(
        workspace,
        evidentiary=False,
        gpu_model="TBD_REQUIRES_HARDWARE",
        gpu_uuid="GPU-fixture",
    )
    with pytest.raises(RunManifestError, match="TBD_REQUIRES_HARDWARE"):
        run.finalize(status="SUCCESS", exit_code=0)


# ------------------------------------------------------------------ F6
def _hardware(**overrides: str) -> dict[str, str]:
    """Runtime fields that match the accepted environment exactly."""
    fields = {
        "gpu_model": ACCEPTED_ENVIRONMENT["gpu_model"],
        "gpu_uuid": ACCEPTED_GPU_UUID,
        "cuda": ACCEPTED_ENVIRONMENT["cuda_runtime"],
        "pytorch": f"{ACCEPTED_ENVIRONMENT['torch_version']}+"
        f"{ACCEPTED_ENVIRONMENT['torch_cuda_build']}",
    }
    fields.update(overrides)
    return fields


def test_a_hardware_run_without_a_gpu_uuid_is_refused(workspace: Workspace) -> None:
    """Refused at the claim, not at the end: an evidentiary manifest that names hardware it
    cannot identify must never exist, not even as RUNNING [AUTH: 01 §16; 03 §8]."""
    with pytest.raises(UnverifiedProvenanceError, match="gpu_uuid"):
        _begin(workspace, **_hardware(gpu_uuid="NOT_APPLICABLE"))


def test_a_cpu_run_may_not_name_a_gpu_uuid(workspace: Workspace) -> None:
    with pytest.raises(RunManifestError, match="no GPU to identify"):
        _begin(workspace, gpu_uuid=ACCEPTED_GPU_UUID)


def test_a_hardware_run_with_a_gpu_uuid_is_accepted(workspace: Workspace) -> None:
    run = _begin(workspace, **_hardware())
    manifest = run.finalize(status="SUCCESS", exit_code=0)
    assert manifest["gpu_uuid"] == ACCEPTED_GPU_UUID
    assert run.verify() == []


def test_a_cpu_fixture_run_needs_no_fabricated_gpu_identity(workspace: Workspace) -> None:
    run = _begin(workspace)
    manifest = run.finalize(status="SUCCESS", exit_code=0)
    assert manifest["gpu_model"] == "NOT_APPLICABLE"
    assert manifest["gpu_uuid"] == "NOT_APPLICABLE"
    assert run.verify() == []


# ------------------------------------------------------------------ F5
def test_metrics_are_hash_bound_by_finalisation(workspace: Workspace) -> None:
    run = _begin(workspace)
    rel = f"artifacts/runs/{run.run_id}/metrics.json"
    (workspace.root / rel).write_text('{"score": 0.42}', encoding="utf-8")

    manifest = run.finalize(status="SUCCESS", exit_code=0, metrics_paths=[rel])
    hashes = manifest["artifact_hashes"]
    assert isinstance(hashes, dict) and rel in hashes
    assert run.verify() == []

    (workspace.root / rel).write_text('{"score": 0.91}', encoding="utf-8")
    problems = run.verify()
    assert problems and any("hash mismatch" in p for p in problems)


def test_a_metrics_path_that_cannot_be_hashed_blocks_success(workspace: Workspace) -> None:
    run = _begin(workspace)
    rel = f"artifacts/runs/{run.run_id}/never_written.json"
    with pytest.raises(Exception, match="absent"):
        run.finalize(status="SUCCESS", exit_code=0, metrics_paths=[rel])
    assert run.status() == RUNNING


def test_a_hand_written_success_with_unhashed_metrics_is_refused(workspace: Workspace) -> None:
    from src.provenance.run_manifest import validate_run_manifest

    run = _begin(workspace)
    manifest = dict(run.read_manifest())
    manifest.update(
        {
            "status": "SUCCESS",
            "exit_code": 0,
            "wall_clock_end": "2026-08-25T10:00:00+00:00",
            "metrics_paths": ["results/p0/table.json"],
        }
    )
    problems = validate_run_manifest(manifest)
    assert any("unhashed metrics path" in p for p in problems), problems


def test_an_output_changed_before_success_prevents_success(workspace: Workspace) -> None:
    run = _begin(workspace)
    rel = f"artifacts/runs/{run.run_id}/result.json"
    (workspace.root / rel).write_text('{"tpr": 0.5}', encoding="utf-8")
    run.add_artifact(rel)
    (workspace.root / rel).write_text('{"tpr": 0.9}', encoding="utf-8")

    with pytest.raises(Exception, match="changed after it was recorded"):
        run.finalize(status="SUCCESS", exit_code=0)
    assert run.status() == RUNNING


# ------------------------------------------------------------------ artifact immutability
def test_artifacts_cannot_change_after_finalisation(workspace: Workspace) -> None:
    run = _begin(workspace)
    rel = f"artifacts/runs/{run.run_id}/result.json"
    (workspace.root / rel).write_text('{"tpr": 0.5}', encoding="utf-8")
    run.add_artifact(rel)
    manifest = run.finalize(status="SUCCESS", exit_code=0, metrics_paths=[rel])
    assert manifest["artifact_paths"] == [rel]
    assert run.verify() == []

    with pytest.raises(RunFinalizedError):
        run.add_artifact(rel)
    with pytest.raises(RunFinalizedError):
        run.finalize(status="SUCCESS", exit_code=0)

    (workspace.root / rel).write_text('{"tpr": 0.9}', encoding="utf-8")
    problems = run.verify()
    assert problems and any("hash mismatch" in p for p in problems)


def test_recording_the_same_path_with_different_bytes_is_refused(workspace: Workspace) -> None:
    run = _begin(workspace)
    rel = f"artifacts/runs/{run.run_id}/result.json"
    (workspace.root / rel).write_text("one", encoding="utf-8")
    run.add_artifact(rel)
    run.add_artifact(rel)  # unchanged bytes are idempotent
    (workspace.root / rel).write_text("two", encoding="utf-8")
    with pytest.raises(RunStoreError, match="different hash"):
        run.add_artifact(rel)


# ------------------------------------------------------------------ F2-F: verified identity
def test_verify_recomputes_the_identity_relationships(workspace: Workspace) -> None:
    run = _begin(workspace)
    run.finalize(status="SUCCESS", exit_code=0)
    assert run.verify() == []

    manifest = dict(run.read_manifest())
    manifest["model_revision"] = "a" * 40  # a different experiment, same file
    run.manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    problems = run.verify()
    assert any("experiment_id" in p for p in problems), problems


def test_verify_detects_a_config_swapped_under_a_stored_identity(workspace: Workspace) -> None:
    run = _begin(workspace)
    run.finalize(status="SUCCESS", exit_code=0)
    manifest = dict(run.read_manifest())
    stored_config = manifest["config"]
    assert isinstance(stored_config, dict)
    manifest["config"] = {**stored_config, "lora_rank": 8}
    run.manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    problems = run.verify()
    assert any("does not hash the stored config" in p for p in problems), problems


def test_a_finalised_manifest_is_a_complete_evidentiary_record(workspace: Workspace) -> None:
    run = _begin(workspace)
    run.finalize(status="SUCCESS", exit_code=0)
    manifest = read_canonical_json(run.manifest_path)
    assert isinstance(manifest, dict)
    for field in RUN_MANIFEST_REQUIRED_FIELDS:
        assert field in manifest, field
    assert manifest["evidentiary"] is True
    assert isinstance(manifest["wall_clock_end"], str)
    assert manifest["exit_code"] == 0


# ==================================================================== FINAL GUARD PATCH
# ENV_SELECTED_NOT_MERELY_VALID
def test_env_selected_not_merely_valid(workspace: Workspace) -> None:
    """Two internally valid captures coexist; only the selected one may be claimed.

    The frozen S00 rule takes the first lock manifest in sorted filename order that
    recomputes and agrees with itself, so the lexicographically smaller digest wins
    [AUTH: 01 §12, §32; 02 §C6].
    """
    import dataclasses

    first = workspace.provenance.environment_lock_sha256
    second = _write_environment_manifest(workspace.root, python_version="3.13.9")
    assert second != first

    selected, problems = selected_environment_identity(workspace.root)
    assert problems == [], problems
    assert selected == min(first, second)
    unselected = max(first, second)

    rejected = dataclasses.replace(workspace.provenance, environment_lock_sha256=unselected)
    with pytest.raises(UnverifiedProvenanceError, match="not the selected environment"):
        _begin(workspace, provenance=rejected)
    assert not (workspace.root / "manifests" / "runs").exists()

    accepted = dataclasses.replace(workspace.provenance, environment_lock_sha256=selected)
    run = _begin(workspace, provenance=accepted)
    run.finalize(status="SUCCESS", exit_code=0)
    assert run.verify() == []


def _write_unselected_environment(workspace: Workspace) -> str:
    """A second internally valid capture whose digest deterministically sorts *after* the
    fixture's, so the fixture identity stays the selected one on every machine."""
    incumbent = workspace.provenance.environment_lock_sha256
    for minor in range(100):
        candidate = _write_environment_manifest(workspace.root, python_version=f"3.13.{minor}")
        if candidate > incumbent:
            return candidate
        (workspace.root / "manifests" / "environments" / f"{candidate}.json").unlink(
            missing_ok=True
        )
    raise AssertionError("no capture sorted after the fixture identity")


def test_env_selected_not_merely_valid_is_rechecked_by_verify(workspace: Workspace) -> None:
    """A stored run swapped from selected A to valid-but-unselected B cannot certify."""
    unselected = _write_unselected_environment(workspace)
    selected, _ = selected_environment_identity(workspace.root)
    assert selected == workspace.provenance.environment_lock_sha256
    assert unselected != selected

    run = _begin(workspace)
    run.finalize(status="SUCCESS", exit_code=0)
    assert run.verify() == []

    manifest = dict(run.read_manifest())
    manifest["environment_lock_sha256"] = unselected
    run.manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    problems = run.verify()
    assert any("not the selected environment" in p for p in problems), problems


# GPU_RUNTIME_MATCHES_ENVIRONMENT
def test_gpu_runtime_matches_environment_forged_uuid(workspace: Workspace) -> None:
    """accepted gpu_uuid = GPU-actual, run claims GPU-forged -> refused."""
    with pytest.raises(UnverifiedProvenanceError, match="gpu_uuid 'GPU-forged'"):
        _begin(workspace, **_hardware(gpu_uuid="GPU-forged"))
    assert not (workspace.root / "manifests" / "runs").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_model", "NVIDIA A100 80GB"),
        ("cuda", "12.4"),
        ("pytorch", "2.4.1+cu124"),
    ],
)
def test_gpu_runtime_matches_environment_other_fields(
    workspace: Workspace, field: str, value: str
) -> None:
    with pytest.raises(UnverifiedProvenanceError, match="accepted environment"):
        _begin(workspace, **_hardware(**{field: value}))


def test_gpu_runtime_matches_environment_accepts_the_bare_torch_version(
    workspace: Workspace,
) -> None:
    run = _begin(workspace, **_hardware(pytorch=ACCEPTED_ENVIRONMENT["torch_version"]))
    run.finalize(status="SUCCESS", exit_code=0)
    assert run.verify() == []


def test_gpu_runtime_matches_environment_is_rechecked_by_verify(workspace: Workspace) -> None:
    """verify() must re-check the relationship, not only begin_run()."""
    run = _begin(workspace, **_hardware())
    run.finalize(status="SUCCESS", exit_code=0)
    assert run.verify() == []

    manifest = dict(run.read_manifest())
    manifest["gpu_uuid"] = "GPU-forged"
    run.manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    problems = run.verify()
    assert any("disagrees with the accepted environment" in p for p in problems), problems


# MASTER_SEED_ALONE_REJECTED
@pytest.mark.parametrize(
    "seeds",
    [{}, {"training_seed": TRAINING_SEED}, {"one_global_seed": TRAINING_SEED}],
)
def test_master_seed_alone_rejected(workspace: Workspace, seeds: dict[str, int]) -> None:
    """The reviewer counterexample: seeds={} became {"training_seed": 101} and certified a
    run with no 01 §30 RNG family recorded at all."""
    with pytest.raises(RunManifestError):
        begin_run(
            workspace.root,
            workspace.provenance,
            resolved_config=workspace.resolved_config,
            seeds=seeds,
            precision="float64",
            tokenizer_hash="9" * 64,
        )
    assert not (workspace.root / "manifests" / "runs").exists()


def test_master_seed_plus_one_family_satisfies_block_a(workspace: Workspace) -> None:
    run = begin_run(
        workspace.root,
        workspace.provenance,
        resolved_config=workspace.resolved_config,
        seeds={"python_rng": 7},
        precision="float64",
        tokenizer_hash="9" * 64,
    )
    assert run.read_manifest()["seeds"] == {"python_rng": 7, "training_seed": TRAINING_SEED}
    run.finalize(status="SUCCESS", exit_code=0)
    assert run.verify() == []
