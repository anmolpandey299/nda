"""C12-C17 plus R2/R3/R4 — adapter persistence, frozen-spec authority and target surface.

The persistence path under test is the real PEFT save layout: `adapter_config.json` plus
`adapter_model.safetensors` with `base_model.model.<module>.lora_{A,B}.weight` naming. Nothing
is mocked — the fixture writes what `PeftModel.save_pretrained` writes and the production
reader reads it back.

Two things the reader must NOT trust are exercised throughout: the adapter's self-declared
LoRA configuration, and its self-declared target set. Both travel with the adapter and both
can be edited, so both are validated against authority that comes from elsewhere — the frozen
config, and the base model's own parameter names.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from backend_fixtures import (
    FIXTURE_MODEL_ID,
    FIXTURE_REVISION,
    corrupt_adapter_tensor,
    drop_adapter_tensor,
    lora_factors,
    peft_layout,
    read_fixture_adapter,
    rewrite_adapter_config,
    trunk_selection,
    write_fixture_adapter,
)

from src.backend.adapters import (
    AdapterError,
    LoraSpecification,
    PersistedAdapter,
    ScalingError,
    induced_update_from_factors,
    resolve_lora_specification,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The canonical model-side target set, derived from the fixture family's parameter names.
SELECTION = list(trunk_selection("tiny_fixture"))
MODULES = tuple(name.removesuffix(".weight") for name in SELECTION)


def specification() -> LoraSpecification:
    return resolve_lora_specification(
        REPO_ROOT,
        architecture_family="tiny_fixture",
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )


def written(
    tmp_path: Path, *, seed: int = 11, modules: tuple[str, ...] = MODULES
) -> PersistedAdapter:
    spec = specification()
    return write_fixture_adapter(
        tmp_path / "adapter",
        specification=spec,
        factors=lora_factors(modules, rank=spec.rank, seed=seed),
        expected_selection=SELECTION,
    )


def read_back(
    tmp_path: Path,
    *,
    expected_base_model_id: str | None = None,
    expected_base_revision: str | None = None,
) -> PersistedAdapter:
    return read_fixture_adapter(
        tmp_path / "adapter",
        specification=specification(),
        expected_selection=SELECTION,
        expected_base_model_id=expected_base_model_id,
        expected_base_revision=expected_base_revision,
    )


# ==================================================================== C12 round trip
def test_c12_an_adapter_round_trips_through_the_real_peft_layout(tmp_path: Path) -> None:
    _, weights, config = peft_layout()
    saved = written(tmp_path)
    assert sorted(p.name for p in (tmp_path / "adapter").iterdir()) == sorted([config, weights])

    reloaded = read_back(
        tmp_path,
        expected_base_model_id=FIXTURE_MODEL_ID,
        expected_base_revision=FIXTURE_REVISION,
    )
    assert reloaded.target_parameters == saved.target_parameters
    assert reloaded.update_identity() == saved.update_identity()
    for name in reloaded.target_parameters:
        assert np.array_equal(reloaded.induced_update(name), saved.induced_update(name))


def test_the_saved_config_names_the_base_and_the_mapping_version(tmp_path: Path) -> None:
    written(tmp_path)
    _, _, config = peft_layout()
    document = json.loads((tmp_path / "adapter" / config).read_text(encoding="utf-8"))
    assert document["base_model_name_or_path"] == FIXTURE_MODEL_ID
    assert document["revision"] == FIXTURE_REVISION
    assert document["peft_type"] == "LORA"
    assert document["r"] == 32
    assert document["lora_target_policy"] == "LANGUAGE_TRUNK_MLP_ONLY"
    assert document["architecture_mapping_version"]
    assert document["use_rslora"] is False


# ==================================================================== C13/C14 wrong base
def test_c13_an_adapter_whose_declared_base_differs_is_rejected(tmp_path: Path) -> None:
    written(tmp_path)
    with pytest.raises(AdapterError, match="declares base"):
        read_back(tmp_path, expected_base_model_id="meta-llama/Llama-3.2-3B")


def test_c14_an_adapter_whose_declared_revision_differs_is_rejected(tmp_path: Path) -> None:
    written(tmp_path)
    with pytest.raises(AdapterError, match="base revision"):
        read_back(tmp_path, expected_base_revision="e" * 40)


def test_an_adapter_relabelled_to_another_base_is_still_caught(tmp_path: Path) -> None:
    written(tmp_path)
    rewrite_adapter_config(tmp_path / "adapter", base_model_name_or_path="Qwen/Qwen3.5-4B-Base")
    with pytest.raises(AdapterError, match="declares base"):
        read_back(tmp_path, expected_base_model_id=FIXTURE_MODEL_ID)


# ==================================================================== R2 frozen spec authority
@pytest.mark.parametrize("alpha", [16, 32, 128, 256])
def test_r2_a_tampered_lora_alpha_is_rejected(tmp_path: Path, alpha: int) -> None:
    """The frozen alpha is 64; an edited adapter may not redefine the scaling."""
    written(tmp_path)
    rewrite_adapter_config(tmp_path / "adapter", lora_alpha=alpha)
    with pytest.raises(AdapterError, match="lora_alpha"):
        read_back(tmp_path)


def test_r2_no_tampered_adapter_can_change_the_scaling(tmp_path: Path) -> None:
    """rank 32 and alpha 64 give scaling 2; .5, 1, 4 and 8 are unreachable."""
    spec = specification()
    assert (spec.rank, spec.lora_alpha, spec.scaling) == (32, 64.0, 2.0)
    written(tmp_path)
    for alpha in (16.0, 32.0, 128.0, 256.0):
        rewrite_adapter_config(tmp_path / "adapter", lora_alpha=alpha)
        with pytest.raises(AdapterError):
            read_back(tmp_path)
    rewrite_adapter_config(tmp_path / "adapter", lora_alpha=64.0)
    assert read_back(tmp_path).specification.scaling == 2.0


def test_r2_a_tampered_target_policy_is_rejected(tmp_path: Path) -> None:
    written(tmp_path)
    rewrite_adapter_config(
        tmp_path / "adapter", lora_target_policy="LANGUAGE_TRUNK_ALL_COMMON_LINEAR"
    )
    with pytest.raises(AdapterError, match="lora_target_policy"):
        read_back(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("r", 16),
        ("bias", "all"),
        ("task_type", "SEQ_CLS"),
        ("lora_scaling_rule", "ALPHA_OVER_SQRT_RANK"),
        ("use_rslora", True),
        ("architecture_mapping_version", "s00.lora-target-mapping.v99"),
        ("lora_dropout", 0.1),
    ],
)
def test_r2_every_material_lora_field_is_validated(
    tmp_path: Path, field: str, value: object
) -> None:
    written(tmp_path)
    rewrite_adapter_config(tmp_path / "adapter", **{field: value})
    with pytest.raises(AdapterError, match=field):
        read_back(tmp_path)


def test_r2_the_frozen_specification_is_what_extraction_uses(tmp_path: Path) -> None:
    """Even a consistent-looking adapter cannot substitute its own semantics."""
    saved = written(tmp_path)
    reloaded = read_back(tmp_path)
    assert reloaded.specification == specification()
    assert reloaded.specification.scaling == 2.0
    name = reloaded.target_parameters[0]
    a, b = reloaded.factors[name]
    assert np.array_equal(reloaded.induced_update(name), 2.0 * (b @ a))
    assert reloaded.update_identity() == saved.update_identity()


# ==================================================================== R3 use_rslora fail closed
def test_r3_use_rslora_resolves_through_the_material_accessor() -> None:
    import inspect

    source = inspect.getsource(resolve_lora_specification)
    assert 'material(backend, "use_rslora")' in source
    assert ".document.get(" not in source, "raw document access bypasses the fail-closed path"
    assert specification().use_rslora is False


@pytest.mark.parametrize("value", [None, "false", 1, "REQUIRED_NOT_CALIBRATED"])
def test_r3_a_non_boolean_use_rslora_fails_closed(tmp_path: Path, value: object) -> None:
    import shutil

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs", dirs_exist_ok=True)
    path = root / "configs" / "backend" / "runtime.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if value is None:
        document["use_rslora"] = {"status": "REQUIRED_NOT_CALIBRATED", "note": "x"}
    else:
        document["use_rslora"]["value"] = value
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(Exception, match="use_rslora|REQUIRED_NOT_CALIBRATED"):
        resolve_lora_specification(
            root, architecture_family="tiny_fixture", target_modules=("gate_proj",)
        )


# ==================================================================== R4 target surface
@pytest.mark.parametrize(
    "intruder",
    [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.o_proj",
        "lm_head",
        "model.embed_tokens",
        "vision_tower.layers.0.mlp.gate_proj",
        "audio_tower.layers.0.mlp.down_proj",
        "multi_modal_projector.linear",
        "model.per_layer_embeddings",
        "model.layers.0.mlp.experts.0.gate_proj",
    ],
)
def test_r4_an_adapter_over_a_forbidden_module_is_rejected(tmp_path: Path, intruder: str) -> None:
    """The adapted tensor keys decide what enters ΔW, so an extra target is a refusal."""
    spec = specification()
    factors = lora_factors((*MODULES, intruder), rank=spec.rank)
    with pytest.raises(AdapterError, match="outside the language-trunk MLP surface"):
        write_fixture_adapter(
            tmp_path / "adapter",
            specification=spec,
            factors=factors,
            expected_selection=SELECTION,
        )


def test_r4_an_adapter_missing_an_expected_target_is_rejected(tmp_path: Path) -> None:
    spec = specification()
    with pytest.raises(AdapterError, match="omits expected target"):
        write_fixture_adapter(
            tmp_path / "adapter",
            specification=spec,
            factors=lora_factors(MODULES[:-1], rank=spec.rank),
            expected_selection=SELECTION,
        )


def test_r4_a_truthful_target_modules_claim_cannot_launder_forbidden_tensors(
    tmp_path: Path,
) -> None:
    """The adapter config may say the right thing while the tensors say another."""
    from safetensors.numpy import load_file, save_file

    prefix, weights, _ = peft_layout()
    written(tmp_path)
    path = tmp_path / "adapter" / weights
    tensors = dict(load_file(str(path)))
    donor = f"{prefix}{MODULES[0]}"
    for factor in ("A", "B"):
        tensors[f"{prefix}lm_head.lora_{factor}.weight"] = tensors[f"{donor}.lora_{factor}.weight"]
    save_file(tensors, str(path))

    _, _, config = peft_layout()
    document = json.loads((tmp_path / "adapter" / config).read_text(encoding="utf-8"))
    assert document["target_modules"] == ["gate_proj", "up_proj", "down_proj"]

    with pytest.raises(AdapterError, match="outside the language-trunk MLP surface"):
        read_back(tmp_path)


def test_r4_the_expected_selection_comes_from_the_model_not_the_adapter() -> None:
    """The canonical selection is derived from base parameter names under the frozen policy."""
    assert SELECTION == sorted(SELECTION)
    assert all(
        name.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"))
        for name in SELECTION
    )
    assert not any("self_attn" in name or "lm_head" in name for name in SELECTION)


# ==================================================================== C15/C16 extraction
def test_c15_the_induced_update_is_exactly_scaling_times_b_times_a() -> None:
    a = np.array([[1.0, 2.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [3.0, 1.0], [0.0, 2.0]])
    expected = np.array([[1.0, 2.0], [3.0, 7.0], [0.0, 2.0]])
    assert np.array_equal(b @ a, expected)
    assert np.array_equal(induced_update_from_factors(a, b, scaling=1.0), expected)
    assert np.array_equal(induced_update_from_factors(a, b, scaling=2.5), 2.5 * expected)


def test_c16_the_peft_scaling_is_applied_exactly_once(tmp_path: Path) -> None:
    spec = specification()
    assert spec.scaling == 2.0 and spec.scaling != 1.0
    saved = written(tmp_path)
    name = saved.target_parameters[0]
    a, b = saved.factors[name]

    once = saved.induced_update(name)
    assert np.array_equal(once, spec.scaling * (b @ a))
    assert not np.allclose(once, b @ a), "an unscaled BA must differ"
    assert not np.allclose(once, spec.scaling**2 * (b @ a)), "a doubled scaling must differ"


def test_the_scaling_rule_is_alpha_over_rank_and_rslora_is_refused() -> None:
    spec = specification()
    assert spec.scaling == spec.lora_alpha / spec.rank
    with pytest.raises(ScalingError, match="use_rslora"):
        _ = replace(spec, use_rslora=True).scaling
    with pytest.raises(ScalingError, match="scaling rule"):
        _ = replace(spec, scaling_rule="SOMETHING_ELSE").scaling


def test_raw_factors_are_never_the_scientific_object(tmp_path: Path) -> None:
    """00 §22: the factorisation is not unique, so (A, B) is not the parameter object."""
    saved = written(tmp_path)
    name = saved.target_parameters[0]
    a, b = saved.factors[name]
    scale = np.diag(np.full(a.shape[0], 3.0))
    rescaled = induced_update_from_factors(scale @ a, b @ np.linalg.inv(scale), scaling=2.0)
    assert np.allclose(rescaled, saved.induced_update(name))
    assert not np.allclose(scale @ a, a), "the factors differ while ΔW does not"


# ==================================================================== C17 refusals
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_c17_a_non_finite_adapter_factor_is_rejected(tmp_path: Path, value: float) -> None:
    prefix, _, _ = peft_layout()
    written(tmp_path)
    corrupt_adapter_tensor(tmp_path / "adapter", f"{prefix}{MODULES[0]}.lora_A.weight", value)
    with pytest.raises(AdapterError, match="NaN or Inf"):
        read_back(tmp_path)


def test_an_unpaired_factor_is_rejected(tmp_path: Path) -> None:
    prefix, _, _ = peft_layout()
    written(tmp_path)
    drop_adapter_tensor(tmp_path / "adapter", f"{prefix}{MODULES[0]}.lora_B.weight")
    with pytest.raises(AdapterError, match="missing lora_B"):
        read_back(tmp_path)


def test_an_unexpected_tensor_is_rejected(tmp_path: Path) -> None:
    from safetensors.numpy import load_file, save_file

    prefix, weights, _ = peft_layout()
    written(tmp_path)
    path = tmp_path / "adapter" / weights
    tensors = dict(load_file(str(path)))
    tensors[f"{prefix}{MODULES[0]}.lora_magnitude_vector"] = np.ones((4,), dtype=np.float32)
    save_file(tensors, str(path))
    with pytest.raises(AdapterError, match="not a LoRA factor tensor"):
        read_back(tmp_path)


def test_a_factor_of_the_wrong_rank_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="do not carry rank 32"):
        write_fixture_adapter(
            tmp_path / "bad",
            specification=specification(),
            factors={MODULES[0]: (np.ones((4, 6)), np.ones((8, 4)))},
            expected_selection=SELECTION,
        )


def test_a_missing_adapter_file_is_reported_not_guessed(tmp_path: Path) -> None:
    _, weights, _ = peft_layout()
    written(tmp_path)
    (tmp_path / "adapter" / weights).unlink()
    with pytest.raises(AdapterError, match=f"no {weights}"):
        read_back(tmp_path)


def test_factors_that_do_not_compose_are_refused() -> None:
    with pytest.raises(AdapterError, match="do not compose"):
        induced_update_from_factors(np.ones((3, 5)), np.ones((7, 4)), scaling=1.0)
