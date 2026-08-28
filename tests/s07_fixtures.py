"""Deterministic task-vector fixtures for S07. No model, no data, no network.

Every matrix comes from a SHA256 counter stream keyed on a label, so the same label gives the
same numbers on any platform. Everything built here is a `FixtureTaskVector` and is
permanently non-evidentiary; `verified_adapter_fixture` builds a real S06 artifact on disk so
the verified path can be exercised without a research model.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.merge.context import MergeExecutionContext, resolve_merge_context
from src.merge.updates import FixtureTaskVector, fixture_induced_update

Matrix = NDArray[Any]
_MANTISSA = float(1 << 53)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A small multi-matrix surface with non-square shapes, matching the LoRA MLP triple.
SURFACE: dict[str, tuple[int, int]] = {
    "model.layers.0.mlp.gate_proj.weight": (12, 8),
    "model.layers.0.mlp.up_proj.weight": (12, 8),
    "model.layers.0.mlp.down_proj.weight": (8, 12),
}


def context() -> MergeExecutionContext:
    """The repository's resolved S07 execution context."""
    return resolve_merge_context(REPO_ROOT)


def stream(label: str, count: int) -> NDArray[np.float64]:
    words: list[int] = []
    for block in range((count + 3) // 4):
        digest = hashlib.sha256(f"{label}|{block}".encode()).digest()
        words.extend(struct.unpack(">4Q", digest))
    raw = np.array(words[:count], dtype=np.uint64)
    return (raw >> np.uint64(11)).astype(np.float64) / _MANTISSA


def matrix(label: str, shape: tuple[int, int], *, scale: float = 1.0) -> NDArray[np.float64]:
    values = 2.0 * stream(label, shape[0] * shape[1]) - 1.0
    return (scale * values).reshape(shape)


def surface_vector(
    label: str,
    *,
    shapes: Mapping[str, tuple[int, int]] | None = None,
    scale: float = 1.0,
    ctx: MergeExecutionContext | None = None,
) -> FixtureTaskVector:
    layout = dict(shapes or SURFACE)
    return fixture_induced_update(
        {name: matrix(f"{label}/{name}", shape, scale=scale) for name, shape in layout.items()},
        context=ctx or context(),
        origin=label,
    )


def low_rank_vector(
    label: str, *, rank: int, scale: float = 1.0, ctx: MergeExecutionContext | None = None
) -> FixtureTaskVector:
    """A task vector whose every matrix is exactly rank `rank` — an induced ΔW = BA."""
    tensors: dict[str, Matrix] = {}
    for name, (rows, columns) in SURFACE.items():
        left = matrix(f"{label}/{name}/B", (rows, rank), scale=scale)
        right = matrix(f"{label}/{name}/A", (rank, columns), scale=scale)
        tensors[name] = left @ right
    return fixture_induced_update(tensors, context=ctx or context(), origin=label)


def diagonal_vector(
    values: Sequence[float],
    *,
    name: str = "diag",
    size: int = 8,
    ctx: MergeExecutionContext | None = None,
) -> FixtureTaskVector:
    """One square matrix with a known spectrum, for exact truncated-SVD assertions."""
    array = np.zeros((size, size), dtype=np.float64)
    for index, value in enumerate(values[:size]):
        array[index, index] = float(value)
    return fixture_induced_update({name: array}, context=ctx or context(), origin="diagonal")


def shuffled(vector: FixtureTaskVector) -> FixtureTaskVector:
    """The same logical mapping, built in reverse insertion order."""
    return fixture_induced_update(
        {name: vector[name] for name in reversed(vector.names)},
        context=vector.context,
        origin=vector.origin,
    )


# ----------------------------------------------------------------------------------------
# a real S06 adapter artifact, so the verified path can be exercised offline
# ----------------------------------------------------------------------------------------


def write_adapter_fixture(
    root: Path, *, label: str = "A", alias: str = "fixture_adapter_101", rank: int = 3
) -> dict[str, Any]:
    """Build a genuine S06 adapter, save it, and return its validated fixture manifest.

    Nothing is faked: the artifact is written by the accepted `save_adapter`, and the manifest
    records the hashes the accepted `verify_adapter_artifact` recomputes from those bytes.
    """
    from src.training.lora import TrainableAudit
    from src.training.manifests import FIXTURE_REFERENCE_ROLE, build_adapter_manifest
    from src.training.seeds import seed_families
    from src.training.trainer import LoraAdapter, adapter_file_hash, save_adapter

    factors: dict[str, tuple[Matrix, Matrix]] = {}
    for name, (rows, columns) in SURFACE.items():
        factors[name] = (
            matrix(f"{label}/{name}/A", (rank, columns)),
            matrix(f"{label}/{name}/B", (rows, rank)),
        )
    adapter = LoraAdapter(factors=factors, rank=rank, scaling=2.0)
    document = save_adapter(adapter)
    relative = f"artifacts/fixtures/{alias}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    audit = TrainableAudit(
        policy="LANGUAGE_TRUNK_MLP_ONLY",
        architecture_family="tiny_fixture",
        target_modules=("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
        adapter_parameters=tuple(f"{n}.lora_A" for n in sorted(SURFACE)),
        n_trainable=len(SURFACE),
        n_frozen=len(SURFACE),
        rank=rank,
        scaling=2.0,
        dropout=0.0,
        mapping_identity="c" * 64,
    )
    return build_adapter_manifest(
        manifest_role=FIXTURE_REFERENCE_ROLE,
        execution_contract_sha256="1" * 64,
        backend_status="NOT_RUN_DEPENDENCY(TORCH_PEFT_BACKEND)",
        adapter_artifact_path=relative,
        adapter_alias=alias,
        run_id="ab" * 32,
        model_manifest_sha256="a" * 64,
        model_revision="UNRESOLVED_NOT_DOWNLOADED",
        data_manifest_sha256="b" * 64,
        corpus_role="TEST_FIXTURE_NOT_RESEARCH_DATA",
        training_config_sha256="d" * 64,
        seeds=seed_families(101, differentially_private=False),
        privacy_regime="NON_DP",
        audit=audit,
        optimizer_steps=8,
        precision="float64",
        adapter_file_sha256=adapter_file_hash(document),
        induced_update_sha256=adapter.update_identity(),
        base_parameter_sha256="0" * 64,
        trainer_version="s06.reference-lora-trainer.v1",
    )
