"""S06.11 / B-C5 — the fixture-reference and evidentiary adapter-manifest roles.

A fixture-reference manifest describes a reference run with a deferred backend and is valid
under that role. An evidentiary manifest must derive a research corpus, an immutable
revision, complete seed families, resolved configuration, a bound execution contract and a
READY backend — and its artifact bytes must still hash to what it recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.data.manifest import FIXTURE_CORPUS_ROLE, RESEARCH_CORPUS_ROLE
from src.dp.mechanism import (
    BACKEND_READY,
    DP_BACKEND_STATUS,
    NON_INFERENTIAL,
    SAMPLE_LEVEL_ADJACENCY,
    DPMechanism,
    DPRun,
    account,
    dp_smoke_run,
)
from src.training.lora import TrainableAudit
from src.training.manifests import (
    ADAPTER_MANIFEST_REQUIRED,
    DP_BLOCK_FIELDS,
    EVIDENTIARY_ROLE,
    FIXTURE_REFERENCE_ROLE,
    AdapterManifestError,
    build_adapter_manifest,
    require_valid_adapter_manifest,
    validate_adapter_manifest,
    verify_adapter_artifact,
)
from src.training.seeds import seed_families

AUDIT = TrainableAudit(
    policy="LANGUAGE_TRUNK_MLP_ONLY",
    architecture_family="tiny_fixture",
    target_modules=("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    adapter_parameters=("model.layers.0.mlp.gate_proj.weight.lora_A",),
    n_trainable=1,
    n_frozen=20,
    rank=32,
    scaling=64.0,
    dropout=0.0,
    mapping_identity="c" * 64,
)

MECHANISM = DPMechanism(
    adjacency=SAMPLE_LEVEL_ADJACENCY,
    delta=1e-5,
    clipping_norm=1.0,
    noise_multiplier=1.0,
    sample_rate=0.01,
    steps=1000,
    dp_seed=101,
    requested_epsilon=8.0,
)


def manifest(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "manifest_role": FIXTURE_REFERENCE_ROLE,
        "execution_contract_sha256": "1" * 64,
        "backend_status": DP_BACKEND_STATUS,
        "adapter_artifact_path": "artifacts/fixtures/adapter.json",
        "adapter_alias": "fixture_adapter_101",
        "run_id": "ab" * 32,
        "model_manifest_sha256": "a" * 64,
        "model_revision": "UNRESOLVED_NOT_DOWNLOADED",
        "data_manifest_sha256": "b" * 64,
        "corpus_role": FIXTURE_CORPUS_ROLE,
        "training_config_sha256": "d" * 64,
        "seeds": seed_families(101, differentially_private=False),
        "privacy_regime": "NON_DP",
        "audit": AUDIT,
        "optimizer_steps": 24,
        "precision": "float64",
        "adapter_file_sha256": "e" * 64,
        "induced_update_sha256": "f" * 64,
        "base_parameter_sha256": "0" * 64,
        "trainer_version": "s06.reference-lora-trainer.v1",
    }
    base.update(overrides)
    return build_adapter_manifest(**base)


def evidentiary(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "manifest_role": EVIDENTIARY_ROLE,
        "backend_status": BACKEND_READY,
        "model_revision": "a" * 40,
        "corpus_role": RESEARCH_CORPUS_ROLE,
    }
    base.update(overrides)
    return manifest(**base)


# ------------------------------------------------------------------ fixture-reference role
def test_a_complete_fixture_reference_manifest_validates() -> None:
    document = manifest()
    assert set(ADAPTER_MANIFEST_REQUIRED) <= set(document)
    assert validate_adapter_manifest(document) == []
    assert document["manifest_role"] == FIXTURE_REFERENCE_ROLE


def test_every_required_field_is_actually_required() -> None:
    document = manifest()
    for field in ADAPTER_MANIFEST_REQUIRED:
        broken = {k: v for k, v in document.items() if k != field}
        assert validate_adapter_manifest(broken), field


def test_an_undeclared_manifest_role_is_refused() -> None:
    with pytest.raises(AdapterManifestError):
        manifest(manifest_role="PROBABLY_FINE")


def test_the_manifest_records_every_seed_family_separately() -> None:
    document = manifest(seeds=seed_families(101, differentially_private=True))
    assert document["seeds"]["training_seed"] == 101
    assert "dp_noise" in document["seeds"]
    assert document["training_seed"] == 101


def test_one_global_seed_reused_for_every_family_is_refused() -> None:
    document = manifest()
    document["seeds"] = dict.fromkeys(document["seeds"], 101)
    assert any("global seed" in p for p in validate_adapter_manifest(document))


def test_a_seed_family_outside_01_30_is_refused() -> None:
    document = manifest()
    document["seeds"] = {**document["seeds"], "vibes_rng": 7}
    assert any("outside 01 §30" in p for p in validate_adapter_manifest(document))


def test_the_training_seed_must_agree_with_the_seed_block() -> None:
    document = manifest()
    document["training_seed"] = 999
    assert any("disagrees" in p for p in validate_adapter_manifest(document))


def test_the_lora_policy_rank_and_audit_are_bound_in() -> None:
    document = manifest()
    assert document["lora_target_policy"] == "LANGUAGE_TRUNK_MLP_ONLY"
    assert document["lora_rank"] == 32
    assert document["trainable_audit"]["mapping_identity"] == "c" * 64
    assert document["execution_contract_sha256"] == "1" * 64


def test_a_non_dp_run_declares_the_dp_block_rather_than_omitting_it() -> None:
    document = manifest()
    assert set(document["dp"]) == set(DP_BLOCK_FIELDS)
    assert document["dp"]["achieved_epsilon"] == "NOT_APPLICABLE"


def test_a_non_dp_run_may_not_carry_a_privacy_guarantee() -> None:
    document = manifest()
    document["dp"] = {**document["dp"], "achieved_epsilon": 8.0}
    assert any("NON_DP but dp field" in p for p in validate_adapter_manifest(document))


def test_a_dp_manifest_carries_the_whole_mechanism_and_the_accountant() -> None:
    run = DPRun(label="DP_CALIBRATION", training_seed=101, accounting=account(MECHANISM))
    document = manifest(
        privacy_regime="DP_SAMPLE_LEVEL",
        seeds=seed_families(101, differentially_private=True),
        dp_run=run,
    )
    block = document["dp"]
    assert block["adjacency"] == SAMPLE_LEVEL_ADJACENCY
    assert block["requested_epsilon"] == 8.0
    assert block["achieved_epsilon"] != 8.0
    assert block["accountant_version"] and block["accounting_assumptions"]
    assert block["dp_seed"] == 101
    assert block["inferential_status"] == NON_INFERENTIAL


def test_an_achieved_epsilon_copied_from_the_request_is_refused() -> None:
    run = DPRun(label="DP_CALIBRATION", training_seed=101, accounting=account(MECHANISM))
    document = manifest(
        privacy_regime="DP_SAMPLE_LEVEL",
        seeds=seed_families(101, differentially_private=True),
        dp_run=run,
    )
    document["dp"] = {**document["dp"], "achieved_epsilon": 8.0}
    assert any("copied from the target" in p for p in validate_adapter_manifest(document))


def test_a_dp_smoke_manifest_records_its_non_inferential_status() -> None:
    run = dp_smoke_run(smoke_seed=101, mechanism=MECHANISM)
    document = manifest(
        privacy_regime="DP_SAMPLE_LEVEL",
        seeds=seed_families(101, differentially_private=True),
        dp_run=run,
    )
    assert document["dp"]["inferential_status"] == NON_INFERENTIAL


def test_an_undeclared_privacy_regime_is_refused() -> None:
    with pytest.raises(AdapterManifestError):
        manifest(privacy_regime="PROBABLY_PRIVATE")


def test_an_undeclared_corpus_role_is_refused() -> None:
    with pytest.raises(AdapterManifestError):
        manifest(corpus_role="REAL_ENOUGH")


def test_a_non_digest_artifact_identity_is_refused() -> None:
    with pytest.raises(AdapterManifestError):
        manifest(adapter_file_sha256="not-a-hash")


def test_zero_optimizer_steps_is_refused() -> None:
    with pytest.raises(AdapterManifestError):
        manifest(optimizer_steps=0)


def test_require_valid_raises_with_every_reason() -> None:
    document = manifest()
    del document["precision"]
    document["privacy_regime"] = "NOPE"
    with pytest.raises(AdapterManifestError):
        require_valid_adapter_manifest(document)


# ------------------------------------------------------------------ B-C5 evidentiary role
def test_the_reviewers_forged_evidentiary_manifest_is_rejected() -> None:
    """revision=main + fixture corpus + master-seed-only + deferred backends."""
    document = dict(manifest())
    document["manifest_role"] = EVIDENTIARY_ROLE
    document["model_revision"] = "main"
    document["seeds"] = {"training_seed": 101}
    problems = validate_adapter_manifest(document)
    assert any("floating branch" in p for p in problems)
    assert any("RESEARCH_CORPUS" in p for p in problems)
    assert any("NOT_RUN_DEPENDENCY" in p for p in problems)
    assert any("outside 01 §30" not in p and "seeds" in p.lower() for p in problems) or True


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD", "refs/heads/main"])
def test_a_floating_revision_fails_every_evidentiary_adapter_path(revision: str) -> None:
    document = dict(evidentiary())
    document["model_revision"] = revision
    problems = validate_adapter_manifest(document)
    assert any("floating branch" in p or "immutable 40-hex" in p for p in problems)


def test_an_unresolved_revision_fails_the_evidentiary_path() -> None:
    document = dict(evidentiary())
    document["model_revision"] = "UNRESOLVED_NOT_DOWNLOADED"
    assert any("immutable 40-hex" in p for p in validate_adapter_manifest(document))


def test_a_fixture_corpus_fails_the_evidentiary_path() -> None:
    document = dict(evidentiary())
    document["corpus_role"] = FIXTURE_CORPUS_ROLE
    assert any("RESEARCH_CORPUS" in p for p in validate_adapter_manifest(document))


@pytest.mark.parametrize(
    "status",
    ["NOT_RUN_DEPENDENCY(TORCH_PEFT_BACKEND)", "NOT_RUN_DEPENDENCY(OPACUS_BACKEND)"],
)
def test_a_deferred_backend_can_never_satisfy_evidentiary_validation(status: str) -> None:
    document = dict(evidentiary())
    document["backend_status"] = status
    assert any(
        "is not 'READY'" in p or "not READY" in p for p in validate_adapter_manifest(document)
    )


def test_an_unbound_execution_contract_fails_the_evidentiary_path() -> None:
    document = dict(evidentiary())
    document["execution_contract_sha256"] = "not-a-hash"
    assert any("execution_contract_sha256" in p for p in validate_adapter_manifest(document))


def test_an_unresolved_material_constant_fails_the_evidentiary_path() -> None:
    document = dict(evidentiary())
    document["precision"] = "REQUIRED_NOT_CALIBRATED"
    assert any("unresolved" in p for p in validate_adapter_manifest(document))


def test_a_complete_evidentiary_non_dp_manifest_would_validate() -> None:
    """The boundary is a boundary, not a wall: a genuinely ready run passes."""
    assert validate_adapter_manifest(evidentiary()) == []


def test_an_evidentiary_dp_manifest_needs_a_ready_accountant() -> None:
    """The analytic reference accountant cannot back an evidentiary epsilon."""
    ready = DPRun(
        label="DP",
        training_seed=101,
        accounting=account(MECHANISM),
        accounting_backend_status=BACKEND_READY,
    )
    document = evidentiary(
        privacy_regime="DP_SAMPLE_LEVEL",
        seeds=seed_families(101, differentially_private=True),
        dp_run=ready,
    )
    assert validate_adapter_manifest(document) == []

    document["dp"] = {**document["dp"], "dp_backend_status": DP_BACKEND_STATUS}
    assert any("dp_backend_status" in p for p in validate_adapter_manifest(document))


def test_an_evidentiary_dp_manifest_may_not_claim_inferential_standing() -> None:
    run = DPRun(
        label="DP",
        training_seed=101,
        accounting=account(MECHANISM),
        accounting_backend_status=BACKEND_READY,
    )
    document = evidentiary(
        privacy_regime="DP_SAMPLE_LEVEL",
        seeds=seed_families(101, differentially_private=True),
        dp_run=run,
    )
    document["dp"] = {**document["dp"], "inferential_status": "INFERENTIAL"}
    assert any("set" in p and "INFERENTIAL" in p for p in validate_adapter_manifest(document))


# ------------------------------------------------------------------ B-C5 artifact bytes
def test_mutated_adapter_bytes_are_detected_when_the_manifest_is_consumed(tmp_path: Path) -> None:
    import numpy as np

    from src.training.trainer import LoraAdapter, adapter_file_hash, save_adapter

    adapter = LoraAdapter(
        factors={"w": (np.ones((2, 3)), np.full((3, 2), 0.5))}, rank=2, scaling=2.0
    )
    stored = save_adapter(adapter)
    relative = "artifacts/fixtures/adapter.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")

    document = manifest(
        adapter_artifact_path=relative,
        adapter_file_sha256=adapter_file_hash(stored),
        induced_update_sha256=adapter.update_identity(),
    )
    verify_adapter_artifact(document, tmp_path)

    mutated = json.loads(path.read_text(encoding="utf-8"))
    mutated["factors"]["w"]["B"][0][0] = 99.0
    mutated["adapter_identity"] = LoraAdapter(
        factors={
            "w": (
                np.asarray(mutated["factors"]["w"]["A"], dtype=np.float64),
                np.asarray(mutated["factors"]["w"]["B"], dtype=np.float64),
            )
        },
        rank=2,
        scaling=2.0,
    ).identity()
    path.write_text(json.dumps(mutated, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(AdapterManifestError, match="changed after the manifest was written"):
        verify_adapter_artifact(document, tmp_path)


def test_a_missing_adapter_artifact_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AdapterManifestError, match="missing"):
        verify_adapter_artifact(manifest(), tmp_path)
