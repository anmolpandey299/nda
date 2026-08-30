"""C1-C11 — revision policy, load policy and the four locked target mappings.

Every check here runs on plain data: a manifest document, a checkpoint's declared config, or a
list of parameter names. That is deliberate — the scientific decisions the backend makes are
decidable without a GPU, a download, or torch installed, so they are testable now rather than
only on the H100 image.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from backend_fixtures import (
    FAMILY_ALIAS,
    FAMILY_LAYOUTS,
    FIXTURE_MODEL_ID,
    FIXTURE_REVISION,
    fixture_panel_entry,
    trunk_parameter_names,
)

from src.backend.adapters import AdapterError, resolve_lora_specification
from src.backend.architecture import (
    CORE_FAMILIES,
    MULTIMODAL_MARKERS,
    ArchitectureError,
    is_multimodal,
    resolve_trunk_selection,
)
from src.backend.loader import (
    QUANTIZATION_REQUIRED,
    LoaderError,
    build_load_plan,
    quantization_refusals,
)
from src.backend.revision import (
    MODEL_REVISION_NOT_FROZEN,
    UNRESOLVED,
    ModelRevisionNotFrozenError,
    require_evidentiary_identity,
    resolve_model_identity,
    revision_problems,
)
from src.backend.settings import backend_settings
from src.materials import material, material_text
from src.provenance.config import resolve_config
from src.training.lora import PRIMARY_TARGET_ROLES

REPO_ROOT = Path(__file__).resolve().parents[2]


def panel_entries() -> list[dict[str, Any]]:
    """The frozen panel, narrowed to plain dicts for the type checker."""
    document = resolve_config(
        REPO_ROOT / "configs" / "models" / "panel.json", config_root=REPO_ROOT / "configs"
    )
    entries = document["panel"]
    assert isinstance(entries, list)
    return [dict(entry) for entry in entries if isinstance(entry, Mapping)]


PANEL_ENTRIES = panel_entries()


# ==================================================================== C1/C2 revision
def test_c1_an_exact_immutable_revision_is_required_for_an_evidentiary_call() -> None:
    identity = resolve_model_identity(fixture_panel_entry(), evidentiary=True)
    assert identity.revision == FIXTURE_REVISION
    assert identity.evidentiary is True
    assert require_evidentiary_identity(identity) is identity


@pytest.mark.parametrize(
    "revision", ["main", "master", "HEAD", "latest", "refs/heads/main", "v1.0", "", None]
)
def test_c2_a_floating_or_absent_revision_is_rejected(revision: object) -> None:
    problems = revision_problems(revision, evidentiary=True)
    assert problems and all(MODEL_REVISION_NOT_FROZEN in p for p in problems)
    with pytest.raises(ModelRevisionNotFrozenError, match=MODEL_REVISION_NOT_FROZEN):
        resolve_model_identity(fixture_panel_entry(revision=revision), evidentiary=True)


def test_the_unacquired_panel_declares_but_cannot_run() -> None:
    """Every core model is declared and none is runnable: nothing has been acquired."""
    for entry in PANEL_ENTRIES:
        declared = resolve_model_identity(entry, evidentiary=False)
        assert declared.revision == UNRESOLVED
        assert declared.evidentiary is False
        with pytest.raises(ModelRevisionNotFrozenError):
            resolve_model_identity(entry, evidentiary=True)
        with pytest.raises(ModelRevisionNotFrozenError):
            require_evidentiary_identity(declared)


def test_the_locked_core_panel_is_the_four_frozen_base_checkpoints() -> None:
    """B1: exactly these four, base weights only, no instruct and no substitute."""
    entries = {str(entry["alias"]): entry for entry in PANEL_ENTRIES}
    assert {str(e["model_id"]) for e in entries.values()} == {
        "Qwen/Qwen3.5-4B-Base",
        "google/gemma-4-E4B",
        "mistralai/Ministral-3-3B-Base-2512",
        "meta-llama/Llama-3.2-3B",
    }
    for entry in entries.values():
        assert entry["base_or_instruct"] == "base"
        assert "instruct" not in str(entry["model_id"]).lower()
    assert set(FAMILY_ALIAS.values()) == set(entries)


# ==================================================================== C3/C4 load policy
def _forbidden(root: Path) -> tuple[str, ...]:
    raw = material(backend_settings(root), "forbidden_load_options")
    assert isinstance(raw, list)
    return tuple(str(k) for k in raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("load_in_8bit", True),
        ("load_in_4bit", True),
        ("quantization_config", {"bits": 4}),
        ("gptq", True),
        ("awq", True),
    ],
)
def test_c3_a_quantized_checkpoint_config_is_refused(key: str, value: object) -> None:
    problems = quantization_refusals({key: value}, forbidden=_forbidden(REPO_ROOT))
    assert problems and QUANTIZATION_REQUIRED in problems[0]

    identity = resolve_model_identity(fixture_panel_entry(), evidentiary=True)
    plan = build_load_plan(REPO_ROOT, identity, model_config={key: value})
    assert not plan.loadable
    with pytest.raises(LoaderError, match=QUANTIZATION_REQUIRED):
        plan.from_pretrained_kwargs()


def test_a_checkpoint_declaring_no_quantization_is_accepted() -> None:
    identity = resolve_model_identity(fixture_panel_entry(), evidentiary=True)
    plan = build_load_plan(REPO_ROOT, identity, model_config={"load_in_8bit": False})
    assert plan.loadable and not plan.refusals


def test_c4_the_production_precision_is_explicit_bf16_and_never_quantized() -> None:
    settings = backend_settings(REPO_ROOT)
    assert material_text(settings, "training_precision") == "bfloat16"
    assert material_text(settings, "forward_precision") == "bfloat16"
    assert material_text(settings, "statistic_precision") == "float64"
    assert material(settings, "permitted_precisions") == ["bfloat16", "float32"]

    identity = resolve_model_identity(fixture_panel_entry(), evidentiary=True)
    plan = build_load_plan(REPO_ROOT, identity)
    assert plan.dtype == "bfloat16"
    assert "quantization_config" not in plan.from_pretrained_kwargs()
    assert plan.from_pretrained_kwargs()["dtype"] == "bfloat16"


def test_an_unpermitted_precision_is_refused() -> None:
    identity = resolve_model_identity(fixture_panel_entry(), evidentiary=True)
    plan = build_load_plan(REPO_ROOT, identity, precision="float16")
    assert not plan.loadable
    assert any("permitted production precision" in r for r in plan.refusals)


def test_the_load_plan_declares_no_silent_fallback() -> None:
    settings = backend_settings(REPO_ROOT)
    assert material_text(settings, "device_fallback_policy") == "EXPLICIT_NO_SILENT_FALLBACK"
    identity = resolve_model_identity(fixture_panel_entry(), evidentiary=True)
    plan = build_load_plan(REPO_ROOT, identity, device="cuda:0")
    assert plan.device == "cuda:0"
    assert plan.identity() != build_load_plan(REPO_ROOT, identity, device="cpu").identity()


def test_an_unacquired_model_cannot_produce_an_evidentiary_load_plan() -> None:
    entry = PANEL_ENTRIES[0]
    declared = resolve_model_identity(entry, evidentiary=False)
    with pytest.raises(ModelRevisionNotFrozenError):
        build_load_plan(REPO_ROOT, declared, evidentiary=True)


# ==================================================================== C5-C10 trunk + mapping
def test_c5_the_language_trunk_is_selected_and_towers_are_excluded() -> None:
    for family, names in FAMILY_LAYOUTS.items():
        if family == "tiny_fixture":
            continue
        selection = resolve_trunk_selection(
            REPO_ROOT, architecture_family=family, parameter_names=names
        )
        assert selection.target_parameters
        assert all(n.startswith(selection.trunk_prefix) for n in selection.target_parameters)
        assert not any(is_multimodal(n) for n in selection.target_parameters)


def test_c6_no_forbidden_or_multimodal_module_can_enter_the_lora_surface() -> None:
    for family, names in FAMILY_LAYOUTS.items():
        selection = resolve_trunk_selection(
            REPO_ROOT, architecture_family=family, parameter_names=names
        )
        for name in selection.target_parameters:
            assert "embed_tokens" not in name and "lm_head" not in name
            assert not any(marker in name for marker in MULTIMODAL_MARKERS)
        assert set(selection.excluded_forbidden) >= {"lm_head.weight"}


@pytest.mark.parametrize("family", CORE_FAMILIES)
def test_c7_to_c10_each_locked_family_maps_to_its_three_mlp_projections(family: str) -> None:
    """C7 Qwen · C8 Gemma · C9 Ministral · C10 Llama."""
    names = FAMILY_LAYOUTS[family]
    selection = resolve_trunk_selection(
        REPO_ROOT, architecture_family=family, parameter_names=names
    )
    assert selection.target_modules == ("gate_proj", "up_proj", "down_proj")
    assert len(selection.target_parameters) == 6, "two decoder layers x three projections"
    for role in PRIMARY_TARGET_ROLES:
        module = selection.role_to_module[role]
        assert sum(n.endswith(f".{module}.weight") for n in selection.target_parameters) == 2
    assert selection.identity() != ""


def test_the_mapping_is_suffix_exact_not_substring() -> None:
    """A forbidden module carrying a target substring must not be captured."""
    names = trunk_parameter_names("model.", layers=1) + [
        "lm_head.down_proj.weight",
        "vision_tower.layers.0.mlp.up_proj.weight",
    ]
    selection = resolve_trunk_selection(
        REPO_ROOT, architecture_family="llama_3_2", parameter_names=names
    )
    assert "lm_head.down_proj.weight" not in selection.target_parameters
    assert "vision_tower.layers.0.mlp.up_proj.weight" not in selection.target_parameters
    assert len(selection.target_parameters) == 3


def test_a_family_whose_projections_cannot_be_identified_fails_compatibility() -> None:
    names = [n for n in FAMILY_LAYOUTS["llama_3_2"] if "gate_proj" not in n]
    with pytest.raises(ArchitectureError, match="fails compatibility"):
        resolve_trunk_selection(REPO_ROOT, architecture_family="llama_3_2", parameter_names=names)


def test_a_model_with_no_decoder_stack_is_refused() -> None:
    with pytest.raises(ArchitectureError, match="no decoder-layer stack"):
        resolve_trunk_selection(
            REPO_ROOT,
            architecture_family="llama_3_2",
            parameter_names=["model.embed_tokens.weight", "lm_head.weight"],
        )


def test_a_purely_multimodal_model_has_no_language_trunk() -> None:
    with pytest.raises(ArchitectureError, match="non-language tower"):
        resolve_trunk_selection(
            REPO_ROOT,
            architecture_family="llama_3_2",
            parameter_names=["vision_tower.layers.0.mlp.up_proj.weight"],
        )


# ==================================================================== C11 rank
@pytest.mark.parametrize("family", CORE_FAMILIES)
def test_c11_the_lora_rank_is_exactly_thirty_two(family: str) -> None:
    selection = resolve_trunk_selection(
        REPO_ROOT, architecture_family=family, parameter_names=FAMILY_LAYOUTS[family]
    )
    specification = resolve_lora_specification(
        REPO_ROOT, architecture_family=family, target_modules=selection.target_modules
    )
    assert specification.rank == 32
    assert specification.peft_config_kwargs()["r"] == 32
    assert specification.target_policy == "LANGUAGE_TRUNK_MLP_ONLY"
    assert specification.bias == "none"
    assert specification.task_type == "CAUSAL_LM"
    assert specification.use_rslora is False


def test_the_fallback_target_policy_is_not_activated() -> None:
    """01 §8B.1: the fallback stays gated on the P0-A measurement-power condition."""
    settings = backend_settings(REPO_ROOT)
    assert material(settings, "fallback_target_policy_authorized") is False
    document = json.loads(
        (REPO_ROOT / "configs" / "training" / "lora.json").read_text(encoding="utf-8")
    )
    assert document["target_policy"]["value"] == "LANGUAGE_TRUNK_MLP_ONLY"
    for path in sorted((REPO_ROOT / "src" / "backend").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "LANGUAGE_TRUNK_ALL_COMMON_LINEAR" in source:
            assert "not activated" in source or "gated" in source, path.name


def test_a_non_primary_policy_is_refused_by_the_specification(tmp_path: Path) -> None:
    """Rewriting the resolved policy makes the specification refuse, not adapt."""
    import shutil

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs", dirs_exist_ok=True)
    path = root / "configs" / "training" / "lora.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["target_policy"]["value"] = "LANGUAGE_TRUNK_ALL_COMMON_LINEAR"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(AdapterError, match="not the frozen primary policy"):
        resolve_lora_specification(
            root, architecture_family="llama_3_2", target_modules=("gate_proj",)
        )


def test_a_rank_other_than_thirty_two_is_refused(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs", dirs_exist_ok=True)
    path = root / "configs" / "training" / "lora.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["adapter_rank"]["value"] = 16
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(AdapterError, match="frozen LoRA rank is 32"):
        resolve_lora_specification(
            root, architecture_family="llama_3_2", target_modules=("gate_proj",)
        )


def test_the_fixture_model_id_cannot_be_mistaken_for_a_research_subject() -> None:
    assert "fixture" in FIXTURE_MODEL_ID and "not-a-research-subject" in FIXTURE_MODEL_ID
    assert FIXTURE_MODEL_ID not in {str(entry["model_id"]) for entry in PANEL_ENTRIES}
