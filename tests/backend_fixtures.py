"""Deterministic backend fixtures. No model download, no network, no GPU, no training.

Everything here is a tiny local structure that emulates the *layout* the production backend
meets on a real checkpoint: the four families' parameter-name trees, a real PEFT adapter
directory, and a synthetic logits source. The production code path is the one under test;
these only supply inputs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.backend.adapters import (
    LoraSpecification,
    PersistedAdapter,
    read_adapter_directory,
    save_adapter_directory,
)
from src.backend.scoring_backend import LogitsSource, TokenBatch
from src.backend.settings import backend_settings
from src.backend.tokenization import TokenizerIdentity, resolve_tokenizer_identity
from src.materials import material_text
from src.provenance.model_manifest import FIXTURE_ROLE

Matrix = NDArray[Any]
REPO_ROOT = Path(__file__).resolve().parents[1]

#: A fixture checkpoint's provenance. B2 permits fixture provenance for tiny local models; the
#: revision is a valid 40-hex shape so the *policy* is exercised, and the model id says plainly
#: that this is not a research subject.
FIXTURE_MODEL_ID = "local/tiny-fixture-not-a-research-subject"
FIXTURE_REVISION = "f" * 40

ATTENTION_MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_MODULES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def peft_layout(root: Path = REPO_ROOT) -> tuple[str, str, str]:
    """The PEFT save-layout names, resolved from config."""
    settings = backend_settings(root)
    return (
        material_text(settings, "peft_module_prefix"),
        material_text(settings, "adapter_weight_filename"),
        material_text(settings, "adapter_config_filename"),
    )


def trunk_parameter_names(prefix: str, *, layers: int = 2, extra: Sequence[str] = ()) -> list[str]:
    """A decoder stack under `prefix`, plus embeddings, the LM head and any extra towers."""
    names = [f"{prefix}embed_tokens.weight", "lm_head.weight", f"{prefix}norm.weight"]
    for index in range(layers):
        for module in (*ATTENTION_MODULES, *MLP_MODULES):
            names.append(f"{prefix}layers.{index}.{module}.weight")
        names.append(f"{prefix}layers.{index}.input_layernorm.weight")
    names.extend(extra)
    return names


#: The four locked families' module layouts, including the multimodal components each exposes.
FAMILY_LAYOUTS: dict[str, list[str]] = {
    "qwen3_5": trunk_parameter_names(
        "model.",
        extra=[
            "visual.blocks.0.mlp.gate_proj.weight",
            "visual.blocks.0.mlp.up_proj.weight",
            "visual.patch_embed.proj.weight",
        ],
    ),
    "gemma_4": trunk_parameter_names(
        "model.language_model.",
        extra=[
            "vision_tower.encoder.layers.0.mlp.up_proj.weight",
            "audio_tower.layers.0.mlp.down_proj.weight",
            "multi_modal_projector.linear.weight",
            "model.per_layer_embeddings.weight",
        ],
    ),
    "ministral_3": trunk_parameter_names(
        "model.",
        extra=[
            "vision_encoder.layers.0.mlp.gate_proj.weight",
            "vision_encoder.layers.0.mlp.down_proj.weight",
        ],
    ),
    "llama_3_2": trunk_parameter_names("model."),
    "tiny_fixture": trunk_parameter_names("model.", layers=1),
}

#: Which panel alias each family belongs to.
FAMILY_ALIAS: dict[str, str] = {
    "qwen3_5": "qwen3_5_4b_base",
    "gemma_4": "gemma_4_e4b",
    "ministral_3": "ministral_3_3b_base",
    "llama_3_2": "llama_3_2_3b",
}


def fixture_panel_entry(*, revision: Any = FIXTURE_REVISION, **overrides: Any) -> dict[str, Any]:
    """A fully resolved fixture model identity, for exercising the evidentiary path."""
    entry: dict[str, Any] = {
        "alias": "tiny_fixture",
        "model_id": FIXTURE_MODEL_ID,
        "revision": revision,
        "architecture_family": "tiny_fixture",
        "base_or_instruct": "base",
        # The production role constant, not a copy of its text: the compatibility contract
        # keys its "never fetch a fixture from the Hub" guard on exactly this value, so the
        # two must not be able to drift apart.
        "role": FIXTURE_ROLE,
        "config_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "weight_file_sha256": "c" * 64,
        "dtype_on_disk": "bfloat16",
        "parameter_count": "1234",
        "lora_target_mapping": "tiny_fixture",
    }
    entry.update(overrides)
    return entry


def generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def lora_factors(
    modules: Sequence[str], *, rank: int, rows: int = 8, columns: int = 6, seed: int = 11
) -> dict[str, tuple[Matrix, Matrix]]:
    """One (A, B) pair per module, in PEFT's shapes: A is (r, in), B is (out, r)."""
    rng = generator(seed)
    return {
        module: (rng.normal(size=(rank, columns)), rng.normal(size=(rows, rank)))
        for module in modules
    }


def write_fixture_adapter(
    directory: Path,
    *,
    specification: LoraSpecification,
    factors: Mapping[str, tuple[Matrix, Matrix]],
    base_model_id: str = FIXTURE_MODEL_ID,
    base_revision: str = FIXTURE_REVISION,
    expected_selection: Sequence[str],
    root: Path = REPO_ROOT,
) -> PersistedAdapter:
    """Write a real PEFT adapter directory through the production persistence path."""
    prefix, weights, config = peft_layout(root)
    return save_adapter_directory(
        directory,
        specification=specification,
        factors=factors,
        base_model_id=base_model_id,
        base_revision=base_revision,
        module_prefix=prefix,
        weight_filename=weights,
        config_filename=config,
        expected_selection=expected_selection,
    )


def read_fixture_adapter(
    directory: Path,
    *,
    specification: LoraSpecification,
    expected_selection: Sequence[str],
    expected_base_model_id: str | None = None,
    expected_base_revision: str | None = None,
    root: Path = REPO_ROOT,
) -> PersistedAdapter:
    """Read one back through the validating production reader."""
    prefix, weights, config = peft_layout(root)
    return read_adapter_directory(
        directory,
        frozen_specification=specification,
        expected_selection=expected_selection,
        expected_base_model_id=expected_base_model_id,
        expected_base_revision=expected_base_revision,
        module_prefix=prefix,
        weight_filename=weights,
        config_filename=config,
    )


def trunk_selection(family: str, *, root: Path = REPO_ROOT) -> Sequence[str]:
    """The canonical model-side target set for one family's fixture layout."""
    from src.backend.architecture import resolve_trunk_selection

    return resolve_trunk_selection(
        root, architecture_family=family, parameter_names=FAMILY_LAYOUTS[family]
    ).target_parameters


def corrupt_adapter_tensor(directory: Path, tensor_name: str, value: float) -> None:
    """Replace one saved factor with a non-finite value, in place."""
    from safetensors.numpy import load_file, save_file

    _, weights, _ = peft_layout()
    path = directory / weights
    tensors = dict(load_file(str(path)))
    array = np.array(tensors[tensor_name], dtype=np.float32)
    array[0, 0] = value
    tensors[tensor_name] = array
    save_file(tensors, str(path))


def drop_adapter_tensor(directory: Path, tensor_name: str) -> None:
    """Remove one saved factor, leaving an unpaired A or B."""
    from safetensors.numpy import load_file, save_file

    _, weights, _ = peft_layout()
    path = directory / weights
    tensors = {k: v for k, v in load_file(str(path)).items() if k != tensor_name}
    save_file(tensors, str(path))


def rewrite_adapter_config(directory: Path, **overrides: Any) -> None:
    """Edit the saved `adapter_config.json`, e.g. to claim a different base."""
    _, _, config = peft_layout()
    path = directory / config
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(overrides)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tokenizer_identity(
    *,
    revision: str = "1" * 40,
    config_sha256: str = "d" * 64,
    special_tokens: dict[str, Any] | None = None,
    max_sequence_length: int = 128,
    root: Path = REPO_ROOT,
) -> TokenizerIdentity:
    return resolve_tokenizer_identity(
        root,
        tokenizer_revision=revision,
        tokenizer_config_sha256=config_sha256,
        special_tokens=special_tokens or {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
        max_sequence_length=max_sequence_length,
    )


class DeterministicLogitsSource(LogitsSource):
    """Reproducible logits keyed by (artifact, record). No model, no network."""

    def __init__(self, vocabulary: int = 17, *, salt: str = "fixture") -> None:
        self.vocabulary = vocabulary
        self.salt = salt

    def logits_for(self, *, artifact_id: str, batch: TokenBatch) -> np.ndarray:
        import hashlib

        seed = int.from_bytes(
            hashlib.sha256(f"{self.salt}|{artifact_id}|{batch.record_id}".encode()).digest()[:8],
            "big",
        )
        rng = np.random.Generator(np.random.PCG64(seed % (2**63)))
        return rng.normal(size=(len(batch.input_ids), self.vocabulary))


def token_batches(
    record_ids: Sequence[str], *, length: int = 9, seed: int = 5
) -> dict[str, TokenBatch]:
    """One tokenized record per id, with a short pad tail so masking is exercised."""
    rng = generator(seed)
    batches: dict[str, TokenBatch] = {}
    for record_id in record_ids:
        ids = [1, *(int(v) for v in rng.integers(3, 16, size=length - 3)), 2, 0]
        mask = [1] * (len(ids) - 1) + [0]
        batches[record_id] = TokenBatch(
            record_id=record_id, input_ids=tuple(ids), attention_mask=tuple(mask)
        )
    return batches


# ----------------------------------------------------------------------------------------
# a deterministic local root the real CLI can run against
# ----------------------------------------------------------------------------------------


def fixture_checkpoint_config(quantized: bool = False) -> dict[str, Any]:
    """A tiny local `config.json`, as an acquired snapshot would carry."""
    document: dict[str, Any] = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "torch_dtype": "bfloat16",
    }
    if quantized:
        document["quantization_config"] = {"quant_method": "gptq", "bits": 4}
    return document


def write_checkpoint_snapshot(directory: Path, *, quantized: bool = False) -> tuple[Path, str]:
    """Write a local snapshot `config.json` and return its path and sha256."""
    from src.provenance.hashing import sha256_file

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(
        json.dumps(fixture_checkpoint_config(quantized), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, sha256_file(path)


def build_cli_fixture_root(
    destination: Path, *, config_sha256: str, alias: str = "tiny_fixture"
) -> Path:
    """A self-contained root the real command can run against.

    Carries the repository's own `configs/`, `src/` and `uv.lock`, plus a panel entry for the
    fixture checkpoint, and is initialised as a git repository so the command resolves a
    genuine commit rather than being handed a placeholder [AUTH: 01 §15, §16].
    """
    import shutil
    import subprocess

    destination.mkdir(parents=True, exist_ok=True)
    for relative in ("configs", "src"):
        shutil.copytree(REPO_ROOT / relative, destination / relative, dirs_exist_ok=True)
    shutil.copy2(REPO_ROOT / "uv.lock", destination / "uv.lock")

    panel_path = destination / "configs" / "models" / "panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    entry = fixture_panel_entry(alias=alias, config_sha256=config_sha256)
    entry["dp_role"] = None
    entry["frozen_modules"] = ["embed_tokens", "lm_head"]
    panel["panel"].append(entry)
    panel_path.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")

    for command in (
        ["git", "init", "--quiet"],
        ["git", "config", "user.email", "fixture@example.invalid"],
        ["git", "config", "user.name", "fixture"],
        ["git", "add", "-A"],
        ["git", "commit", "--quiet", "-m", "fixture root"],
    ):
        subprocess.run(command, cwd=destination, check=True, capture_output=True)  # noqa: S603
    return destination
