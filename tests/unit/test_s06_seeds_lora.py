"""S06.1–S06.5 — model contract, target policy, trainable surface and seed families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.models.fixtures import build_base_model, load_fixture_spec, target_parameter_names
from src.provenance.config import resolve_config
from src.provenance.run_manifest import SEED_FIELDS
from src.training.lora import (
    FALLBACK_POLICY,
    FORBIDDEN_ROLES,
    TARGET_POLICY,
    LoraPolicyError,
    audit_trainable_surface,
    load_target_mapping,
    select_target_parameters,
)
from src.training.model_contract import (
    UNRESOLVED,
    BackendRuntime,
    ModelContractError,
    dp_primary,
    load_panel,
    revision_problems,
    validate_panel_member,
)
from src.training.seeds import (
    DERIVED_FAMILIES,
    DP_ONLY_FAMILIES,
    SeedError,
    derive,
    families_are_independent,
    seed_families,
)
from src.training.settings import panel_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL = panel_settings(REPO_ROOT)
MAPPING = load_target_mapping("tiny_fixture", PANEL.document)


def fixture_spec() -> Any:
    configs = REPO_ROOT / "configs"
    return load_fixture_spec(
        resolve_config(configs / "models/tiny_fixture.json", config_root=configs)
    )


# ------------------------------------------------------------------ 01 §8G revisions
def test_an_unresolved_revision_is_declarable_but_not_evidentiary() -> None:
    assert revision_problems(UNRESOLVED, evidentiary=False) == []
    assert revision_problems(UNRESOLVED, evidentiary=True)


@pytest.mark.parametrize("revision", ["main", "master", "HEAD", "refs/heads/main"])
def test_a_floating_revision_is_always_refused(revision: str) -> None:
    """The mutation this kills: accepting a branch name as a model revision."""
    assert revision_problems(revision, evidentiary=False)
    assert revision_problems(revision, evidentiary=True)


def test_only_a_full_commit_revision_pins_an_evidentiary_run() -> None:
    assert revision_problems("a" * 40, evidentiary=True) == []
    assert revision_problems("a" * 12, evidentiary=True)


def test_the_declared_panel_has_exactly_one_dp_primary() -> None:
    members = load_panel(PANEL.document)
    assert dp_primary(members).model_id == "meta-llama/Llama-3.2-3B"
    assert all(m.revision == UNRESOLVED for m in members), "a model was acquired"
    for member in members:
        assert validate_panel_member(member, evidentiary=False) == []
        assert validate_panel_member(member, evidentiary=True)


def test_two_dp_primaries_are_refused() -> None:
    document: dict[str, Any] = dict(PANEL.document)
    panel = [dict(entry) for entry in document["panel"]]
    panel[1]["dp_role"] = "DP_PRIMARY"
    document["panel"] = panel
    with pytest.raises(ModelContractError):
        dp_primary(load_panel(document))


# ------------------------------------------------------------------ 01 §8B target policy
def test_the_frozen_policy_selects_only_mlp_projections() -> None:
    spec = fixture_spec()
    names = list(build_base_model(spec))
    selected = select_target_parameters(names, MAPPING)
    assert set(selected) == set(target_parameter_names(spec))
    assert MAPPING.policy == TARGET_POLICY


@pytest.mark.parametrize("forbidden", ["model.embed_tokens.weight", "lm_head.weight"])
def test_embeddings_and_the_lm_head_are_never_selected(forbidden: str) -> None:
    spec = fixture_spec()
    assert forbidden not in select_target_parameters(list(build_base_model(spec)), MAPPING)


def test_a_forbidden_module_is_excluded_even_when_it_contains_a_target_name() -> None:
    names = ["model.layers.0.mlp.down_proj.weight", "lm_head.mlp.down_proj.weight"]
    assert select_target_parameters(names, MAPPING) == ["model.layers.0.mlp.down_proj.weight"]


def test_a_mapping_that_matches_nothing_fails_closed() -> None:
    with pytest.raises(LoraPolicyError):
        select_target_parameters(["model.norm.weight"], MAPPING)


def test_a_partial_mlp_mapping_is_refused() -> None:
    document = {
        "policy": TARGET_POLICY,
        "target_mappings": {"x": {"roles": {"gate": "mlp.gate_proj"}}},
    }
    with pytest.raises(LoraPolicyError, match="missing"):
        load_target_mapping("x", document)  # type: ignore[arg-type]


def test_an_unauthorised_policy_string_is_refused() -> None:
    other: dict[str, Any] = {"policy": "EVERYTHING", "target_mappings": {"x": {"roles": {}}}}
    with pytest.raises(LoraPolicyError, match="frozen primary policy"):
        load_target_mapping("x", other)
    assert FALLBACK_POLICY == "LANGUAGE_TRUNK_ALL_COMMON_LINEAR"


# ------------------------------------------------------------------ 01 §8B trainable audit
def audit(trainable: list[str], adapters: list[str] | None = None) -> Any:
    spec = fixture_spec()
    base = list(build_base_model(spec))
    targets = target_parameter_names(spec)
    adapters = (
        adapters
        if adapters is not None
        else [f"{name}.lora_{part}" for name in targets for part in ("A", "B")]
    )
    return audit_trainable_surface(
        base_parameter_names=base,
        adapter_parameter_names=adapters,
        trainable_parameter_names=trainable,
        mapping=MAPPING,
        rank=32,
        scaling=64.0,
        dropout=0.0,
    )


def test_an_adapter_only_surface_passes_the_audit() -> None:
    spec = fixture_spec()
    adapters = [f"{n}.lora_{p}" for n in target_parameter_names(spec) for p in ("A", "B")]
    result = audit(adapters)
    assert result.n_trainable == len(adapters)
    assert result.n_frozen == len(build_base_model(spec))
    assert result.rank == 32 and result.policy == TARGET_POLICY


def test_a_trainable_base_weight_is_refused() -> None:
    """The mutation this kills: a base parameter left trainable, breaking ΔW = BA."""
    spec = fixture_spec()
    adapters = [f"{n}.lora_{p}" for n in target_parameter_names(spec) for p in ("A", "B")]
    with pytest.raises(LoraPolicyError, match="base"):
        audit([*adapters, "model.layers.0.mlp.gate_proj.weight"])


def test_an_unauthorised_adapter_module_is_refused() -> None:
    """The mutation this kills: an adapter attached to a module the policy forbids."""
    spec = fixture_spec()
    adapters = [f"{n}.lora_{p}" for n in target_parameter_names(spec) for p in ("A", "B")]
    rogue = ["lm_head.weight.lora_A", "lm_head.weight.lora_B"]
    with pytest.raises(LoraPolicyError):
        audit([*adapters, *rogue], adapters=[*adapters, *rogue])


def test_a_frozen_authorised_adapter_is_refused() -> None:
    spec = fixture_spec()
    adapters = [f"{n}.lora_{p}" for n in target_parameter_names(spec) for p in ("A", "B")]
    with pytest.raises(LoraPolicyError):
        audit(adapters[:-1])


def test_the_forbidden_role_vocabulary_covers_every_01_8b_entry() -> None:
    for role in ("embed", "lm_head", "vision", "audio", "projector", "router", "expert"):
        assert any(role in entry for entry in FORBIDDEN_ROLES), role


# ------------------------------------------------------------------ 01 §30 seed families
def test_every_family_is_derived_separately_and_reproducibly() -> None:
    families = seed_families(101, differentially_private=True)
    assert set(families.values) == set(DERIVED_FAMILIES)
    assert families == seed_families(101, differentially_private=True)
    assert families_are_independent(101)


def test_a_non_dp_run_records_no_dp_noise_family() -> None:
    families = seed_families(101, differentially_private=False)
    assert not set(families.values) & set(DP_ONLY_FAMILIES)
    with pytest.raises(SeedError):
        families["dp_noise"]


def test_families_are_independent_of_each_other() -> None:
    """The mutation this kills: one global RNG standing in for every family."""
    families = seed_families(101, differentially_private=True)
    values = list(families.values.values())
    assert len(set(values)) == len(values)
    assert families.training_seed not in values


def test_changing_the_master_seed_moves_every_family() -> None:
    first = seed_families(101, differentially_private=True)
    second = seed_families(202, differentially_private=True)
    for family in DERIVED_FAMILIES:
        assert first[family] != second[family], family


def test_the_family_vocabulary_is_block_a_s() -> None:
    assert set(("training_seed", *DERIVED_FAMILIES)) == set(SEED_FIELDS)
    manifest_seeds = seed_families(101, differentially_private=True).as_manifest_seeds()
    assert set(manifest_seeds) == set(SEED_FIELDS)
    assert manifest_seeds["training_seed"] == 101


def test_an_undeclared_family_is_refused() -> None:
    with pytest.raises(SeedError):
        derive(101, "one_global_seed")


# ------------------------------------------------------------------ Block B handoff
def test_a_runtime_that_differs_from_config_is_reported_not_absorbed() -> None:
    """The mutation this kills: silently accepting a runtime precision the config never set."""
    runtime = BackendRuntime(
        precision="bfloat16",
        max_sequence_length=512,
        backend_identity="fixture",
        supports_gradients=True,
    )
    assert runtime.mismatch(precision="bfloat16", max_sequence_length=512) == []
    assert runtime.mismatch(precision="float32", max_sequence_length=512)
    assert runtime.mismatch(precision="bfloat16", max_sequence_length=128)
