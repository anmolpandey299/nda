"""B2/B4/B16/B17 — release families end to end on fixtures, and the S09 provenance handoff.

Nothing here loads a research model, real data or a GPU. Every artifact is a fixture merge
and is permanently NON_EVIDENTIARY [AUTH: 01 §14, §20].
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s07_fixtures import context, surface_vector, write_adapter_fixture

from src.merge.diagnostics import (
    _relative_frobenius_error,
    dare_perturbation_from_results,
    protected_information_floor_from_o3,
    rank_report,
)
from src.merge.family import (
    NON_EVIDENTIARY,
    MergeFamilyError,
    MergeSpec,
    build_release_family,
)
from src.merge.operators import DARE, LINEAR, SVD_TRUNC
from src.merge.settings import descendant_counts, effective_rank_tolerance, merge_settings
from src.merge.updates import (
    VERIFIED_INDUCED_UPDATE,
    FixtureTaskVector,
    fixture_induced_update,
    verified_task_vector_from_adapter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = merge_settings(REPO_ROOT)
PERMITTED_K = descendant_counts(SETTINGS)
TOLERANCE = effective_rank_tolerance(SETTINGS)
CTX = context()
SEED = 70701


def task_vector(tensors: Mapping[str, Any], *, origin: str = "unlabelled") -> FixtureTaskVector:
    """Local alias: every task vector in this module is a fixture."""
    return fixture_induced_update(tensors, context=CTX, origin=origin)


TAU_A = surface_vector("tau_A")
PARTNERS = {f"B{index + 1}": surface_vector(f"tau_B{index}") for index in range(4)}


def linear_specs(k: int, alphas: list[float]) -> list[MergeSpec]:
    return [
        MergeSpec(
            descendant_id=f"C{i + 1}", operator=LINEAR, alpha=alphas[i], partner_id=f"B{i + 1}"
        )
        for i in range(k)
    ]


def dare_specs(k: int, *, p: float) -> list[MergeSpec]:
    return [
        MergeSpec(
            descendant_id=f"C{i + 1}",
            operator=DARE,
            alpha=0.5,
            partner_id=f"B{i + 1}",
            drop_probability=p,
            dare_merge_seed=SEED + i,
        )
        for i in range(k)
    ]


# ------------------------------------------------------------------ Block C handoff
def test_s07_consumes_a_verified_block_c_adapter_without_changing_s06(tmp_path: Path) -> None:
    """The verified path walks the accepted S06 proof chain: manifest -> bytes -> updates.

    `validate_adapter_manifest` and `verify_adapter_artifact` are Block C exactly as accepted;
    S07 adds a receipt around them rather than changing them.
    """
    manifest = write_adapter_fixture(tmp_path)
    vector = verified_task_vector_from_adapter(
        adapter_manifest=manifest, root=tmp_path, context=CTX
    )

    assert vector.provenance_class == VERIFIED_INDUCED_UPDATE
    assert not any(name.endswith((".lora_A", ".lora_B")) for name in vector.names)
    receipt = vector.source_evidence()
    assert receipt["induced_update_sha256"] == manifest["induced_update_sha256"]
    assert receipt["run_id"] == manifest["run_id"]
    assert receipt["manifest_role"] == manifest["manifest_role"]

    # the arithmetic dtype is S07's policy, and the receipt records what it came from
    assert vector.dtype == "float32"
    assert receipt["source_precision"] == "float64"

    from src.provenance.hashing import read_canonical_json
    from src.training.trainer import load_adapter

    stored = read_canonical_json(tmp_path / str(manifest["adapter_artifact_path"]))
    adapter = load_adapter(stored)  # type: ignore[arg-type]
    for name in vector.names:
        assert np.allclose(vector[name], adapter.induced_update(name), rtol=1e-6)


def test_a_verified_constituent_merges_and_is_recorded_as_verified(tmp_path: Path) -> None:
    manifest = write_adapter_fixture(tmp_path, label="A", alias="fixture_adapter_A")
    partner_manifest = write_adapter_fixture(tmp_path, label="B", alias="fixture_adapter_B")
    protected = verified_task_vector_from_adapter(
        adapter_manifest=manifest, root=tmp_path, context=CTX
    )
    partner = verified_task_vector_from_adapter(
        adapter_manifest=partner_manifest, root=tmp_path, context=CTX
    )

    family = build_release_family(
        protected=protected,
        partners={"B1": partner},
        specs=linear_specs(1, [0.5]),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    document = family.by_id("C1").as_dict()
    assert document["protected_provenance_class"] == VERIFIED_INDUCED_UPDATE
    assert document["partner_provenance_class"] == VERIFIED_INDUCED_UPDATE
    assert document["provenance_class"] == NON_EVIDENTIARY


# ------------------------------------------------------------------ family construction
@pytest.mark.parametrize("k", [1, 2, 4])
def test_a_linear_family_reuses_one_protected_constituent(k: int) -> None:
    alphas = [0.1, 0.3, 0.6, 0.9][:k]
    family = build_release_family(
        protected=TAU_A,
        partners={f"B{i + 1}": PARTNERS[f"B{i + 1}"] for i in range(k)},
        specs=linear_specs(k, alphas),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    assert family.k == k
    assert {r.protected_identity for r in family.descendants} == {TAU_A.content_identity()}
    assert len({r.result_identity for r in family.descendants}) == k
    assert {r.surface_identity for r in family.descendants} == {family.surface_identity}


def test_a_dare_family_gives_each_descendant_its_own_masks() -> None:
    """B17 item 4: distinct configured mask seeds must not collapse to one mask."""
    family = build_release_family(
        protected=TAU_A,
        partners=PARTNERS,
        specs=dare_specs(4, p=0.5),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    masks = family.mask_identities()
    assert len(masks) == 4
    protected_masks = [pair[0] for pair in masks.values()]
    partner_masks = [pair[1] for pair in masks.values()]
    assert len(set(protected_masks)) == 4
    assert len(set(partner_masks)) == 4
    assert not set(protected_masks) & set(partner_masks)


def test_a_surface_change_between_descendants_is_refused() -> None:
    """B17 item 8."""
    narrowed = task_vector(
        {name: PARTNERS["B2"][name] for name in PARTNERS["B2"].names[:-1]}, origin="narrow"
    )
    with pytest.raises(Exception, match="missing|absent"):
        build_release_family(
            protected=TAU_A,
            partners={"B1": PARTNERS["B1"], "B2": narrowed},
            specs=linear_specs(2, [0.5, 0.5]),
            permitted_k=PERMITTED_K,
            context=CTX,
        )


def test_a_family_mixing_operators_keeps_each_operator_identity() -> None:
    specs = [
        MergeSpec(descendant_id="C1", operator=LINEAR, alpha=0.5, partner_id="B1"),
        MergeSpec(
            descendant_id="C2",
            operator=DARE,
            alpha=0.5,
            partner_id="B2",
            drop_probability=0.5,
            dare_merge_seed=SEED,
        ),
    ]
    family = build_release_family(
        protected=TAU_A,
        partners={"B1": PARTNERS["B1"], "B2": PARTNERS["B2"]},
        specs=specs,
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    assert family.by_id("C1").as_dict()["operator"] == LINEAR
    assert family.by_id("C2").as_dict()["operator"] == DARE
    assert family.by_id("C1").protected_mask is None
    assert family.by_id("C2").protected_mask is not None


# ------------------------------------------------------------------ B16 provenance handoff
def test_every_result_exposes_the_identities_s09_will_need() -> None:
    family = build_release_family(
        protected=TAU_A,
        partners=PARTNERS,
        specs=dare_specs(4, p=0.75),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    document = family.as_dict()
    assert document["provenance_class"] == NON_EVIDENTIARY
    assert json.loads(json.dumps(document)) == document, "the handoff must be JSON-serialisable"
    for result in family.descendants:
        row = result.as_dict()
        for key in (
            "operator",
            "operator_version",
            "protected_update_sha256",
            "partner_update_sha256",
            "partner_id",
            "alpha",
            "drop_probability",
            "protected_mask_sha256",
            "partner_mask_sha256",
            "result_update_sha256",
            "parameter_surface_sha256",
            "dtype",
        ):
            assert row[key] is not None, key
        assert row["operator_version"] == CTX.operator_version
        assert row["execution_context_sha256"] == CTX.identity()


def test_a_fixture_merge_never_claims_to_be_evidentiary() -> None:
    family = build_release_family(
        protected=TAU_A,
        partners={"B1": PARTNERS["B1"]},
        specs=linear_specs(1, [0.5]),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    assert family.as_dict()["provenance_class"] == NON_EVIDENTIARY
    assert all(r.provenance_class == NON_EVIDENTIARY for r in family.descendants)
    assert "run_id" not in family.as_dict()
    assert "manifest_sha256" not in family.as_dict()


def test_the_family_identity_moves_with_every_material_input() -> None:
    def identity(**changes: Any) -> str:
        spec = MergeSpec(
            descendant_id=changes.get("descendant_id", "C1"),
            operator=DARE,
            alpha=changes.get("alpha", 0.5),
            partner_id="B1",
            drop_probability=changes.get("p", 0.5),
            dare_merge_seed=changes.get("seed", SEED),
        )
        family = build_release_family(
            protected=changes.get("protected", TAU_A),
            partners={"B1": PARTNERS["B1"]},
            specs=[spec],
            permitted_k=PERMITTED_K,
            context=CTX,
        )
        return family.by_id(spec.descendant_id).identity()

    base = identity()
    assert identity(alpha=0.4) != base
    assert identity(p=0.75) != base
    assert identity(seed=SEED + 1) != base
    assert identity(protected=surface_vector("other_A")) != base
    assert identity() == base


# ------------------------------------------------------------------ diagnostics on a family
def test_a_family_reports_perturbation_and_rank_without_deciding_anything() -> None:
    linear_family = build_release_family(
        protected=TAU_A,
        partners=PARTNERS,
        specs=linear_specs(4, [0.5] * 4),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    dare_family = build_release_family(
        protected=TAU_A,
        partners=PARTNERS,
        specs=dare_specs(4, p=0.5),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    for descendant_id in ("C1", "C2", "C3", "C4"):
        perturbed = dare_family.by_id(descendant_id).update
        assert (
            dare_perturbation_from_results(
                dare_family.by_id(descendant_id), linear_family.by_id(descendant_id)
            )
            > 0.0
        )
        reports = rank_report(perturbed, constituent_ranks=[8, 8], tolerance=TOLERANCE)
        assert len(reports) == len(perturbed.names)
        assert all(r.effective_rank >= 0 for r in reports)


def test_an_o3_family_exposes_the_information_floor_inputs() -> None:
    specs = [
        MergeSpec(
            descendant_id="C1", operator=SVD_TRUNC, alpha=0.5, partner_id="B1", retained_rank=2
        )
    ]
    family = build_release_family(
        protected=TAU_A,
        partners={"B1": PARTNERS["B1"]},
        specs=specs,
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    result = family.by_id("C1")
    assert result.protected_truncations is not None
    from src.merge.operators import truncate_vector

    floor = _relative_frobenius_error(truncate_vector(TAU_A, rank=2, context=CTX), TAU_A)
    bound = protected_information_floor_from_o3(result, TAU_A)
    assert bound == pytest.approx(floor, rel=1e-4)
    assert result.as_dict()["retained_rank"] == 2


def test_an_unknown_descendant_id_is_refused() -> None:
    family = build_release_family(
        protected=TAU_A,
        partners={"B1": PARTNERS["B1"]},
        specs=linear_specs(1, [0.5]),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    with pytest.raises(MergeFamilyError, match="not in this family"):
        family.by_id("C9")
