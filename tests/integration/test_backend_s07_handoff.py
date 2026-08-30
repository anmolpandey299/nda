"""C18 — the backend's adapter reaches S07 through the accepted verified path, unchanged.

There is no privileged route. The backend writes the S06 artifact and manifest, and S07's own
`verified_task_vector_from_adapter` re-validates the manifest, re-hashes the artifact bytes,
reloads the adapter and derives ΔW itself. This test checks that the two derivations agree and
that S07 still refuses everything it refused before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from backend_fixtures import (
    FIXTURE_REVISION,
    lora_factors,
    trunk_selection,
    write_fixture_adapter,
)

from src.backend.adapters import LoraSpecification, PersistedAdapter, resolve_lora_specification
from src.backend.handoff import (
    HandoffError,
    adapter_artifact_document,
    check_backend_and_s07_agree,
    require_no_parallel_trusted_path,
    write_adapter_artifact,
)
from src.merge.context import resolve_merge_context
from src.merge.updates import ProvenanceError
from src.provenance.hashing import JSONValue
from src.training.lora import TargetMapping, TrainableAudit, audit_trainable_surface
from src.training.manifests import (
    FIXTURE_REFERENCE_ROLE,
    build_adapter_manifest,
    validate_adapter_manifest,
)
from src.training.seeds import seed_families
from src.training.trainer import TRAINER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The canonical model-side selection; the adapter is validated against it, not against itself.
SELECTION = list(trunk_selection("tiny_fixture"))
MODULES = tuple(name.removesuffix(".weight") for name in SELECTION)


def _specification() -> LoraSpecification:
    return resolve_lora_specification(
        REPO_ROOT,
        architecture_family="tiny_fixture",
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )


def _adapter(tmp_path: Path, *, seed: int = 11, directory: str = "peft") -> PersistedAdapter:
    spec = _specification()
    return write_fixture_adapter(
        tmp_path / directory,
        specification=spec,
        factors=lora_factors(MODULES, rank=spec.rank, seed=seed),
        expected_selection=SELECTION,
    )


def _audit(spec: LoraSpecification) -> TrainableAudit:
    mapping = TargetMapping(
        architecture_family="tiny_fixture",
        policy=spec.target_policy,
        role_to_module={"gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"},
        forbidden_modules=("embed_tokens", "lm_head"),
    )
    base = [f"model.layers.0.{m}.weight" for m in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    adapters = [f"{name}.lora_A" for name in base] + [f"{name}.lora_B" for name in base]
    return audit_trainable_surface(
        base_parameter_names=base,
        adapter_parameter_names=adapters,
        trainable_parameter_names=adapters,
        mapping=mapping,
        rank=spec.rank,
        scaling=spec.scaling,
        dropout=spec.lora_dropout,
    )


def _manifest(root: Path, adapter: PersistedAdapter, relative: str) -> dict[str, JSONValue]:
    file_sha, update_sha = write_adapter_artifact(root, relative, adapter)
    spec = adapter.specification
    return build_adapter_manifest(
        manifest_role=FIXTURE_REFERENCE_ROLE,
        execution_contract_sha256="a" * 64,
        backend_status="NOT_RUN_DEPENDENCY(TORCH_PEFT_BACKEND)",
        adapter_artifact_path=relative,
        adapter_alias="backend_fixture",
        run_id="fixture-run",
        model_manifest_sha256="b" * 64,
        model_revision=FIXTURE_REVISION,
        data_manifest_sha256="c" * 64,
        corpus_role="TEST_FIXTURE_NOT_RESEARCH_DATA",
        training_config_sha256="d" * 64,
        seeds=seed_families(101, differentially_private=False),
        privacy_regime="NON_DP",
        audit=_audit(spec),
        optimizer_steps=1,
        precision="float32",
        adapter_file_sha256=file_sha,
        induced_update_sha256=update_sha,
        base_parameter_sha256="e" * 64,
        trainer_version=TRAINER_VERSION,
    )


def test_c18_the_extracted_update_is_accepted_by_the_s07_verified_handoff(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    manifest = _manifest(tmp_path, adapter, "artifacts/adapters/backend_fixture.json")
    assert validate_adapter_manifest(manifest) == []

    report = check_backend_and_s07_agree(
        adapter=adapter,
        adapter_manifest=manifest,
        root=tmp_path,
        context=resolve_merge_context(REPO_ROOT),
    )
    assert report["n_target_parameters"] == len(MODULES)
    assert report["induced_update_sha256"] == adapter.update_identity()
    assert report["provenance_class"] == "VERIFIED_INDUCED_UPDATE"


def test_s07_derives_the_same_induced_update_the_backend_did(tmp_path: Path) -> None:
    """The two derivations are independent; agreement is checked, not assumed."""
    from src.backend.handoff import verified_task_vector_from_backend_adapter

    adapter = _adapter(tmp_path)
    manifest = _manifest(tmp_path, adapter, "artifacts/adapters/agree.json")
    vector = verified_task_vector_from_backend_adapter(
        adapter_manifest=manifest, root=tmp_path, context=resolve_merge_context(REPO_ROOT)
    )
    for name in adapter.target_parameters:
        assert np.array_equal(
            np.asarray(vector[name], dtype=np.float32),
            np.asarray(adapter.induced_update(name), dtype=np.float32),
        )


def test_a_tampered_artifact_is_caught_by_s07_not_by_the_backend(tmp_path: Path) -> None:
    """S07 re-hashes the bytes, so editing the artifact after the manifest fails there."""
    from src.provenance.hashing import read_canonical_json, write_canonical_json

    adapter = _adapter(tmp_path)
    relative = "artifacts/adapters/tampered.json"
    manifest = _manifest(tmp_path, adapter, relative)

    stored = read_canonical_json(tmp_path / relative)
    assert isinstance(stored, dict)
    factors = stored["factors"]
    assert isinstance(factors, dict)
    first = sorted(factors)[0]
    entry = factors[first]
    assert isinstance(entry, dict)
    rows = entry["A"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, list)
    row[0] = float(str(row[0])) + 1.0
    write_canonical_json(tmp_path / relative, stored)

    with pytest.raises(ProvenanceError, match="does not match its manifest"):
        check_backend_and_s07_agree(
            adapter=adapter,
            adapter_manifest=manifest,
            root=tmp_path,
            context=resolve_merge_context(REPO_ROOT),
        )


def test_a_manifest_for_another_adapter_is_refused(tmp_path: Path) -> None:
    first = _adapter(tmp_path, seed=11)
    second = _adapter(tmp_path, seed=999, directory="other")
    manifest = _manifest(tmp_path, first, "artifacts/adapters/first.json")
    assert first.update_identity() != second.update_identity()
    with pytest.raises(HandoffError, match="not the backend adapter's|different induced"):
        check_backend_and_s07_agree(
            adapter=second,
            adapter_manifest=manifest,
            root=tmp_path,
            context=resolve_merge_context(REPO_ROOT),
        )


def test_no_backend_module_can_mint_a_verified_task_vector() -> None:
    """B9: the backend has no parallel trusted path into S07."""
    require_no_parallel_trusted_path(REPO_ROOT)


def test_the_artifact_document_is_the_accepted_s06_shape(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    document = adapter_artifact_document(adapter)
    assert set(document) == {"rank", "scaling", "adapter_identity", "factors"}
    assert document["rank"] == 32
    assert document["scaling"] == adapter.specification.scaling
    factors = document["factors"]
    assert isinstance(factors, dict)
    assert set(factors) == set(adapter.target_parameters)
    for entry in factors.values():
        assert isinstance(entry, dict)
        assert set(entry) == {"A", "B"}


def test_the_handoff_needs_a_factory_issued_merge_context(tmp_path: Path) -> None:
    from src.backend.handoff import verified_task_vector_from_backend_adapter

    adapter = _adapter(tmp_path)
    manifest = _manifest(tmp_path, adapter, "artifacts/adapters/ctx.json")
    with pytest.raises(HandoffError, match="factory-issued merge execution context"):
        verified_task_vector_from_backend_adapter(
            adapter_manifest=manifest,
            root=tmp_path,
            context=object(),  # type: ignore[arg-type]
        )


# ==================================================================== R5 after validation
def test_r5_a_tampered_alpha_never_reaches_an_s06_artifact(tmp_path: Path) -> None:
    """The refusal happens on read, so no scientific artifact is produced at all."""
    from backend_fixtures import read_fixture_adapter, rewrite_adapter_config

    from src.backend.adapters import AdapterError

    _adapter(tmp_path)
    rewrite_adapter_config(tmp_path / "peft", lora_alpha=128.0)
    with pytest.raises(AdapterError, match="lora_alpha"):
        read_fixture_adapter(
            tmp_path / "peft",
            specification=_specification(),
            expected_selection=SELECTION,
        )
    assert not (tmp_path / "artifacts").exists(), "no artifact is written for a refused adapter"


def test_r5_a_forbidden_target_never_reaches_an_s06_artifact(tmp_path: Path) -> None:
    from src.backend.adapters import AdapterError

    spec = _specification()
    with pytest.raises(AdapterError, match="outside the language-trunk MLP surface"):
        write_fixture_adapter(
            tmp_path / "intruder",
            specification=spec,
            factors=lora_factors((*MODULES, "lm_head"), rank=spec.rank),
            expected_selection=SELECTION,
        )
    assert not (tmp_path / "artifacts").exists()


def test_r5_the_validated_adapter_still_completes_the_whole_handoff(tmp_path: Path) -> None:
    """Validation added on read does not change what a legitimate adapter produces."""
    adapter = _adapter(tmp_path)
    manifest = _manifest(tmp_path, adapter, "artifacts/adapters/validated.json")
    report = check_backend_and_s07_agree(
        adapter=adapter,
        adapter_manifest=manifest,
        root=tmp_path,
        context=resolve_merge_context(REPO_ROOT),
    )
    assert report["provenance_class"] == "VERIFIED_INDUCED_UPDATE"
    assert report["n_target_parameters"] == len(SELECTION)
    assert adapter.specification.scaling == 2.0
