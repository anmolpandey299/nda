"""The twenty-eight final S07 boundary attacks [F20].

Each is a way a scientifically false S07 artifact could be produced through an ordinary
supported Python API. The defence is structural throughout: identity-bearing objects are
opaque, non-dataclass, factory-issued and backed by immutable bytes, so the attacks do not
fail a check — they have no API to run through.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s07_fixtures import SURFACE, context, matrix, surface_vector, write_adapter_fixture

from src.merge.context import (
    ExecutionContextError,
    MergeExecutionContext,
    load_merge_execution_context,
)
from src.merge.diagnostics import (
    DiagnosticError,
    dare_perturbation_from_results,
    dare_theoretical_perturbation,
    protected_information_floor_from_o3,
)
from src.merge.family import (
    NON_EVIDENTIARY,
    MergeFamilyError,
    MergeResult,
    MergeSpec,
    ReleaseFamily,
    build_dare_descendant,
    build_linear_descendant,
    build_o3_descendant,
    build_release_family,
)
from src.merge.masks import DareMask
from src.merge.operators import (
    DARE,
    LINEAR,
    SVD_TRUNC,
    DeferredOperatorError,
    Truncation,
    require_supported_operator,
    truncate_vector,
    truncations,
)
from src.merge.settings import descendant_counts, merge_settings
from src.merge.updates import (
    FIXTURE_INDUCED_UPDATE,
    VERIFIED_INDUCED_UPDATE,
    FixtureTaskVector,
    ProvenanceError,
    TaskVector,
    TaskVectorError,
    VerifiedTaskVector,
    fixture_induced_update,
    verified_task_vector_from_adapter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CTX = context()
PERMITTED_K = descendant_counts(merge_settings(REPO_ROOT))
TAU_A = surface_vector("tau_A")
TAU_B = surface_vector("tau_B")
SEED = 70701


def attempt_replace(obj: object, **changes: object) -> object:
    """Run `dataclasses.replace` through `Any`.

    mypy already knows the call is invalid on a non-dataclass — which is the point — but the
    attack must actually execute at runtime, so the reference is untyped here.
    """
    replace: Any = dataclasses.replace
    return replace(obj, **changes)


def attempt_copy_replace(obj: object) -> object:
    replace: Any = copy.replace
    return replace(obj)


def linear_spec(cid: str = "C1", *, alpha: float = 0.5, partner: str = "B1") -> MergeSpec:
    return MergeSpec(descendant_id=cid, operator=LINEAR, alpha=alpha, partner_id=partner)


def dare_spec(
    cid: str = "C1", *, alpha: float = 0.5, p: float = 0.5, seed: int = SEED
) -> MergeSpec:
    return MergeSpec(
        descendant_id=cid,
        operator=DARE,
        alpha=alpha,
        partner_id="B1",
        drop_probability=p,
        dare_merge_seed=seed,
    )


def o3_spec(cid: str = "C1", *, rank: int = 3) -> MergeSpec:
    return MergeSpec(
        descendant_id=cid, operator=SVD_TRUNC, alpha=0.5, partner_id="B1", retained_rank=rank
    )


# ============================================================= 1-4  verified boundary
def test_01_a_renamed_lora_a_factor_cannot_become_verified() -> None:
    renamed = fixture_induced_update(
        {"model.layers.0.mlp.gate_proj.weight": matrix("factor_A", (3, 8))},
        context=CTX,
        origin="pretending_to_be_delta_w",
    )
    assert renamed.provenance_class == FIXTURE_INDUCED_UPDATE
    assert not renamed.verified and not isinstance(renamed, VerifiedTaskVector)


def test_02_a_renamed_lora_b_factor_cannot_become_verified() -> None:
    renamed = fixture_induced_update(
        {"model.layers.0.mlp.down_proj.weight": matrix("factor_B", (8, 3))},
        context=CTX,
        origin="also_pretending",
    )
    assert renamed.provenance_class == FIXTURE_INDUCED_UPDATE
    assert not isinstance(renamed, VerifiedTaskVector)


def test_03_no_public_constructor_accepts_tensors_plus_genuine_proof(tmp_path: Path) -> None:
    """The verified factory derives ΔW itself; a caller never supplies its tensors."""
    manifest = write_adapter_fixture(tmp_path)
    genuine = verified_task_vector_from_adapter(
        adapter_manifest=manifest, root=tmp_path, context=CTX
    )
    assert genuine.provenance_class == VERIFIED_INDUCED_UPDATE

    with pytest.raises(TypeError, match="factory-issued"):
        VerifiedTaskVector(tensors={"w": np.ones((2, 2))}, context=CTX)
    with pytest.raises(TypeError, match="factory-issued"):
        VerifiedTaskVector()
    with pytest.raises(TypeError, match="factory-issued"):
        TaskVector(tensors={"w": np.ones((2, 2))}, context=CTX)

    import inspect

    parameters = set(inspect.signature(verified_task_vector_from_adapter).parameters)
    assert parameters == {"adapter_manifest", "root", "context"}
    assert "tensors" not in parameters


def test_04_a_fixture_cannot_promote_itself_to_verified() -> None:
    fixture = surface_vector("A")
    assert isinstance(fixture, FixtureTaskVector)
    with pytest.raises(TypeError):
        attempt_replace(fixture, provenance_class=VERIFIED_INDUCED_UPDATE)
    with pytest.raises(AttributeError):
        object.__setattr__  # noqa: B018 - the real attempt is below
        fixture.__setattr__("_source", {})
    assert fixture.provenance_class == FIXTURE_INDUCED_UPDATE


# ============================================================= 5-6  immutable storage
def test_05_item_assignment_leaves_the_authoritative_vector_unchanged() -> None:
    vector = surface_vector("A")
    name = vector.names[0]
    recorded = vector.content_identity()
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        vector[name][0, 0] = 99.0
    assert vector.content_identity() == recorded


def test_06_a_numpy_writeback_attack_cannot_reach_the_authoritative_bytes() -> None:
    vector = surface_vector("A")
    name = vector.names[0]
    recorded = vector.content_identity()
    view = np.asarray(vector[name])
    with pytest.raises(ValueError, match="WRITEABLE"):
        view.setflags(write=True)
    assert not view.flags.writeable
    # a detached copy may be written, and cannot touch the vector
    scratch = vector.copy_of(name)
    scratch[0, 0] = 12345.0
    assert scratch.flags.owndata
    assert vector.content_identity() == recorded
    assert vector[name][0, 0] != 12345.0


def test_the_authoritative_storage_is_immutable_bytes() -> None:
    vector = surface_vector("A")
    stored = object.__getattribute__(vector, "_bytes")
    assert all(isinstance(payload, bytes) for payload in stored.values())
    # every accessor hands out a fresh view, never the storage
    first, second = vector[vector.names[0]], vector[vector.names[0]]
    assert first is not second
    assert np.array_equal(first, second)


# ============================================================= 7-8  no forgeable proof
def test_07_no_public_receipt_hash_can_manufacture_verified_status() -> None:
    """There is no public receipt object at all: the class was removed, not hardened."""
    import src.merge.updates as updates

    assert not hasattr(updates, "AdapterReceipt")
    assert not hasattr(updates, "receipt_seal")
    assert not hasattr(updates, "verify_adapter_source")
    public = {name for name in dir(updates) if not name.startswith("_")}
    assert not {name for name in public if "receipt" in name.lower()}


def test_08_dataclasses_replace_is_unavailable_on_a_verified_vector(tmp_path: Path) -> None:
    manifest = write_adapter_fixture(tmp_path)
    verified = verified_task_vector_from_adapter(
        adapter_manifest=manifest, root=tmp_path, context=CTX
    )
    assert not dataclasses.is_dataclass(verified)
    with pytest.raises(TypeError):
        attempt_replace(verified, origin="other")
    with pytest.raises(TypeError):
        attempt_copy_replace(verified)
    with pytest.raises(AttributeError):
        verified.__setattr__("_source", {})


def test_a_tampered_adapter_artifact_cannot_produce_a_verified_vector(tmp_path: Path) -> None:
    manifest = write_adapter_fixture(tmp_path)
    path = tmp_path / str(manifest["adapter_artifact_path"])
    stored = json.loads(path.read_text(encoding="utf-8"))
    first = sorted(stored["factors"])[0]
    stored["factors"][first]["B"][0][0] += 5.0
    path.write_text(json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="does not match its manifest"):
        verified_task_vector_from_adapter(adapter_manifest=manifest, root=tmp_path, context=CTX)


def test_an_invalid_manifest_cannot_produce_a_verified_vector(tmp_path: Path) -> None:
    manifest = dict(write_adapter_fixture(tmp_path))
    del manifest["precision"]
    with pytest.raises(ProvenanceError, match="manifest is invalid"):
        verified_task_vector_from_adapter(adapter_manifest=manifest, root=tmp_path, context=CTX)


# ============================================================= 9-10  opaque results
def test_09_dataclasses_replace_is_unavailable_on_a_merge_result() -> None:
    built = build_linear_descendant(protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX)
    assert not dataclasses.is_dataclass(built)
    for field in ("operator", "alpha", "protected_identity", "surface_identity"):
        with pytest.raises(TypeError):
            attempt_replace(built, **{field: "forged"})
    with pytest.raises(AttributeError):
        built.__setattr__("_operator", DARE)
    assert built.operator == LINEAR


def test_10_direct_merge_result_construction_fails() -> None:
    with pytest.raises(TypeError, match="factory-issued"):
        MergeResult()
    with pytest.raises(TypeError, match="factory-issued"):
        MergeResult(operator=LINEAR, alpha=0.5)


def test_the_provenance_class_cannot_be_promoted() -> None:
    built = build_linear_descendant(protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX)
    assert built.provenance_class == NON_EVIDENTIARY == "NON_EVIDENTIARY_S07_MERGE"
    with pytest.raises(TypeError):
        attempt_replace(built, provenance_class="EVIDENTIARY_MERGE")
    assert built.as_dict()["provenance_class"] == NON_EVIDENTIARY


# ============================================================= 11-12  family and partner
def test_11_a_mixed_a_release_family_fails() -> None:
    """C1 from A1 and C2 from A2, with no API to claim A1 for both."""
    a1, a2 = surface_vector("A1"), surface_vector("A2")
    c1 = build_linear_descendant(protected=a1, partner=TAU_B, spec=linear_spec("C1"), context=CTX)
    c2 = build_linear_descendant(protected=a2, partner=TAU_B, spec=linear_spec("C2"), context=CTX)
    assert c1.protected_identity != c2.protected_identity

    with pytest.raises(TypeError, match="factory-issued"):
        ReleaseFamily(
            protected_identity=c1.protected_identity,
            surface_identity=c1.surface_identity,
            context=CTX,
            descendants=(c1, c2),
        )
    import inspect

    assert "protected_identity" not in inspect.signature(build_release_family).parameters


def test_a_family_derives_its_protected_identity_from_the_issued_results() -> None:
    partners = {f"B{i + 1}": surface_vector(f"p{i}") for i in range(2)}
    family = build_release_family(
        protected=TAU_A,
        partners=partners,
        specs=[linear_spec("C1", partner="B1"), linear_spec("C2", partner="B2")],
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    assert family.protected_identity == TAU_A.content_identity()
    assert not dataclasses.is_dataclass(family)
    with pytest.raises(AttributeError):
        family.__setattr__("_protected_identity", "f" * 64)


def test_12_a_partner_cannot_report_another_partners_content_identity() -> None:
    """The display label is free; the scientific content identity comes from the bytes."""
    b1, b2 = surface_vector("B1_bytes"), surface_vector("B2_bytes")
    built = build_linear_descendant(
        protected=TAU_A, partner=b2, spec=linear_spec(partner="B1"), context=CTX
    )
    assert built.partner_label == "B1"
    assert built.partner_identity == b2.content_identity()
    assert built.partner_identity != b1.content_identity()
    assert built.as_dict()["partner_update_sha256"] == b2.content_identity()


# ============================================================= 13-15  operator identity
def test_13_dare_bytes_cannot_report_linear() -> None:
    with pytest.raises(MergeFamilyError, match="builds .* only"):
        build_linear_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX)
    built = build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX)
    assert built.operator == DARE and built.as_dict()["operator"] == DARE


def test_14_o3_bytes_cannot_report_linear() -> None:
    with pytest.raises(MergeFamilyError, match="builds .* only"):
        build_linear_descendant(protected=TAU_A, partner=TAU_B, spec=o3_spec(), context=CTX)
    built = build_o3_descendant(protected=TAU_A, partner=TAU_B, spec=o3_spec(), context=CTX)
    assert built.operator == SVD_TRUNC


def test_15_linear_bytes_cannot_report_dare() -> None:
    with pytest.raises(MergeFamilyError, match="builds .* only"):
        build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX)
    with pytest.raises(MergeFamilyError, match="builds .* only"):
        build_o3_descendant(protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX)


# ============================================================= 16-17  seed and config
def test_16_seed_101_evidence_cannot_be_reported_as_seed_202() -> None:
    first = build_dare_descendant(
        protected=TAU_A, partner=TAU_B, spec=dare_spec(seed=101), context=CTX
    )
    second = build_dare_descendant(
        protected=TAU_A, partner=TAU_B, spec=dare_spec(seed=202), context=CTX
    )
    assert first.as_dict()["protected_mask_sha256"] != second.as_dict()["protected_mask_sha256"]
    assert first.as_dict()["dare_merge_seed"] == 101
    with pytest.raises(TypeError):
        attempt_replace(first, _dare_merge_seed=202)
    with pytest.raises(AttributeError):
        first.__setattr__("_dare_merge_seed", 202)
    # and no caller supplies a mask or a mask hash anywhere
    assert not {f.name for f in dataclasses.fields(MergeSpec)} & {
        "protected_mask",
        "partner_mask",
        "protected_mask_sha256",
        "partner_mask_sha256",
    }
    with pytest.raises(TypeError, match="factory-issued"):
        DareMask()


def test_17_an_arbitrary_config_sha_cannot_be_attached_to_a_result() -> None:
    built = build_linear_descendant(protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX)
    assert built.as_dict()["config_sha256"] == CTX.config_sha256
    assert built.as_dict()["execution_context_sha256"] == CTX.identity()
    with pytest.raises(TypeError):
        attempt_replace(built, _context=None)
    with pytest.raises(AttributeError):
        built.__setattr__("_context", None)
    with pytest.raises(TypeError, match="factory-issued"):
        MergeExecutionContext()


# ============================================================= 18-20  context fails closed
def rewritten_context(tmp_path: Path, **overrides: Any) -> MergeExecutionContext:
    document = json.loads(
        (REPO_ROOT / "configs" / "merge" / "operators.json").read_text(encoding="utf-8")
    )
    for key, value in overrides.items():
        document[key] = {"status": "FROZEN", "value": value, "note": "test"}
    assert tmp_path is not None
    return load_merge_execution_context(document)


def test_18_a_float64_scientific_context_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExecutionContextError, match="float32"):
        rewritten_context(tmp_path, arithmetic_dtype="float64")
    assert CTX.arithmetic_dtype == "float32"


def test_19_an_unsupported_operator_version_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExecutionContextError, match="not executable by this code"):
        rewritten_context(tmp_path, operator_version="s07.operators.v99")


def test_20_an_unsupported_mask_scheme_or_backend_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExecutionContextError, match="mask scheme"):
        rewritten_context(tmp_path, dare_mask_scheme="m99")
    with pytest.raises(ExecutionContextError, match="SVD backend"):
        rewritten_context(tmp_path, svd_backend="scipy.something")


def test_the_context_hash_covers_the_config_actually_consumed(tmp_path: Path) -> None:
    document = json.loads(
        (REPO_ROOT / "configs" / "merge" / "operators.json").read_text(encoding="utf-8")
    )
    first = load_merge_execution_context(document)
    assert first.identity() == CTX.identity()
    document["description"] = "a different snapshot"
    second = load_merge_execution_context(document)
    assert second.config_sha256 != first.config_sha256
    assert second.identity() != first.identity()


# ============================================================= 21-23  DARE diagnostic
def test_21_a_wrong_baseline_partner_is_rejected() -> None:
    dare = build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX)
    other = build_linear_descendant(
        protected=TAU_A, partner=surface_vector("tau_B_prime"), spec=linear_spec(), context=CTX
    )
    with pytest.raises(DiagnosticError, match="different partner"):
        dare_perturbation_from_results(dare, other)


def test_22_a_wrong_baseline_alpha_is_rejected() -> None:
    dare = build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX)
    other = build_linear_descendant(
        protected=TAU_A, partner=TAU_B, spec=linear_spec(alpha=0.25), context=CTX
    )
    with pytest.raises(DiagnosticError, match="different merge coefficient"):
        dare_perturbation_from_results(dare, other)


def test_23_a_wrong_baseline_protected_constituent_is_rejected() -> None:
    dare = build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX)
    other = build_linear_descendant(
        protected=surface_vector("tau_A_prime"), partner=TAU_B, spec=linear_spec(), context=CTX
    )
    with pytest.raises(DiagnosticError, match="different protected constituent"):
        dare_perturbation_from_results(dare, other)


def test_the_bound_dare_diagnostic_accepts_the_true_comparison() -> None:
    linear = build_linear_descendant(
        protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX
    )
    dare = build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX)
    assert linear.comparison_identity() == dare.comparison_identity()
    assert dare_perturbation_from_results(dare, linear) > 0.0
    assert dare_theoretical_perturbation(0.25) == pytest.approx(np.sqrt(1.0 / 3.0))


def test_a_non_dare_comparand_or_non_linear_baseline_is_rejected() -> None:
    linear = build_linear_descendant(
        protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX
    )
    o3 = build_o3_descendant(protected=TAU_A, partner=TAU_B, spec=o3_spec(), context=CTX)
    with pytest.raises(DiagnosticError, match=f"not {DARE}"):
        dare_perturbation_from_results(linear, linear)
    with pytest.raises(DiagnosticError, match=f"not {LINEAR}"):
        dare_perturbation_from_results(
            build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX),
            o3,
        )


# ============================================================= 24-26  O3 floor
def test_24_a_partner_truncation_cannot_be_supplied_to_the_protected_floor() -> None:
    """There is no `parts` parameter: the floor recomputes T_s(A) itself."""
    import inspect

    parameters = list(inspect.signature(protected_information_floor_from_o3).parameters)
    assert parameters == ["o3_result", "protected_update"]
    assert "parts" not in parameters

    import src.merge.diagnostics as diagnostics

    assert not hasattr(diagnostics, "ProtectedConstituent")
    assert not hasattr(diagnostics, "protected_constituent")


def test_25_an_unrelated_truncation_cannot_be_supplied() -> None:
    o3 = build_o3_descendant(protected=TAU_A, partner=TAU_B, spec=o3_spec(), context=CTX)
    unrelated = truncations(TAU_B, rank=3, context=CTX)
    assert unrelated  # a partner truncation exists, and has nowhere to be handed in
    floor: Any = protected_information_floor_from_o3
    with pytest.raises(TypeError, match="unexpected keyword"):
        floor(o3, TAU_A, parts=unrelated)
    with pytest.raises(TypeError, match="factory-issued"):
        Truncation()


def test_26_a_wrong_protected_a_with_an_o3_result_is_rejected() -> None:
    o3 = build_o3_descendant(protected=TAU_A, partner=TAU_B, spec=o3_spec(), context=CTX)
    with pytest.raises(DiagnosticError, match="not the protected constituent"):
        protected_information_floor_from_o3(o3, TAU_B)
    with pytest.raises(DiagnosticError, match="not the protected constituent"):
        protected_information_floor_from_o3(o3, surface_vector("tau_A_prime"))


def test_a_non_o3_result_has_no_information_floor() -> None:
    linear = build_linear_descendant(
        protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX
    )
    with pytest.raises(DiagnosticError, match=f"defined for an {SVD_TRUNC}"):
        protected_information_floor_from_o3(linear, TAU_A)


def test_the_floor_recomputes_the_protected_truncation_and_matches_the_direct_error() -> None:
    from src.merge.diagnostics import _relative_frobenius_error

    for rank in (1, 2, 3):
        o3 = build_o3_descendant(
            protected=TAU_A, partner=TAU_B, spec=o3_spec(rank=rank), context=CTX
        )
        bound = protected_information_floor_from_o3(o3, TAU_A)
        direct = _relative_frobenius_error(truncate_vector(TAU_A, rank=rank, context=CTX), TAU_A)
        assert bound == pytest.approx(direct, rel=1e-4)
        assert o3.retained_rank == rank
        assert o3.protected_truncation_identity is not None


# ============================================================= 27-28
def test_27_the_fp32_counterexample() -> None:
    """alpha = 1/3 over float32 ±1 follows FP32 arithmetic [AUTH: 01 §10; 00 §25]."""
    ones = fixture_induced_update({"w": np.ones((2, 2))}, context=CTX)
    minus = fixture_induced_update({"w": -np.ones((2, 2))}, context=CTX)
    alpha = 1.0 / 3.0
    result = build_linear_descendant(
        protected=ones, partner=minus, spec=linear_spec(alpha=alpha), context=CTX
    )
    value = float(result.update["w"][0, 0])
    expected = float(
        np.float32(np.float32(alpha) * np.float32(1.0) + np.float32(1.0 - alpha) * np.float32(-1.0))
    )
    assert value == expected
    assert value != alpha * 1.0 + (1.0 - alpha) * (-1.0)
    assert value == pytest.approx(-0.3333333432674408)
    assert result.update["w"].dtype == np.float32
    assert result.as_dict()["arithmetic_dtype"] == "float32"


def test_every_scientific_artifact_is_float32() -> None:
    for spec, factory in (
        (linear_spec(), build_linear_descendant),
        (dare_spec(), build_dare_descendant),
        (o3_spec(), build_o3_descendant),
    ):
        built = factory(protected=TAU_A, partner=TAU_B, spec=spec, context=CTX)
        assert built.update.dtype == "float32"
        for name in built.update.names:
            assert built.update[name].dtype == np.float32


def test_28_ties_remains_refused() -> None:
    with pytest.raises(DeferredOperatorError, match="deferred"):
        require_supported_operator("TIES")
    with pytest.raises(DeferredOperatorError):
        MergeSpec(descendant_id="C1", operator="TIES", alpha=0.5, partner_id="B1")


# ============================================================= F17 dataclass audit
@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(CTX, id="MergeExecutionContext"),
        pytest.param(TAU_A, id="FixtureTaskVector"),
        pytest.param(
            build_linear_descendant(
                protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX
            ),
            id="MergeResult",
        ),
        pytest.param(
            build_dare_descendant(
                protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX
            ).protected_mask,
            id="DareMask",
        ),
    ],
)
def test_no_identity_bearing_object_is_a_dataclass(obj: object) -> None:
    """F17: the chosen defence is opacity, so this must hold for every issued object."""
    assert not dataclasses.is_dataclass(obj)
    with pytest.raises(TypeError):
        attempt_replace(obj)


def test_the_result_identity_moves_with_every_material_field() -> None:
    """F5: operator parameters and the final artifact bytes both enter the identity."""
    base = build_dare_descendant(
        protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX
    ).identity()
    assert (
        build_dare_descendant(
            protected=TAU_A, partner=TAU_B, spec=dare_spec(alpha=0.25), context=CTX
        ).identity()
        != base
    )
    assert (
        build_dare_descendant(
            protected=TAU_A, partner=TAU_B, spec=dare_spec(p=0.75), context=CTX
        ).identity()
        != base
    )
    assert (
        build_dare_descendant(
            protected=TAU_A, partner=TAU_B, spec=dare_spec(seed=SEED + 1), context=CTX
        ).identity()
        != base
    )
    assert (
        build_dare_descendant(
            protected=surface_vector("other_A"), partner=TAU_B, spec=dare_spec(), context=CTX
        ).identity()
        != base
    )
    assert (
        build_dare_descendant(
            protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX
        ).identity()
        == base
    )


def test_the_o3_result_identity_includes_the_truncation_evidence() -> None:
    first = build_o3_descendant(protected=TAU_A, partner=TAU_B, spec=o3_spec(rank=2), context=CTX)
    second = build_o3_descendant(protected=TAU_A, partner=TAU_B, spec=o3_spec(rank=3), context=CTX)
    assert first.identity() != second.identity()
    assert first.protected_truncation_identity != second.protected_truncation_identity
    assert first.as_dict()["protected_truncation_sha256"] == first.protected_truncation_identity


def test_a_surface_or_shape_change_is_still_refused() -> None:
    narrow = fixture_induced_update(
        {name: TAU_B[name] for name in TAU_B.names[:-1]}, context=CTX, origin="narrow"
    )
    with pytest.raises(TaskVectorError):
        build_linear_descendant(protected=TAU_A, partner=narrow, spec=linear_spec(), context=CTX)
    assert set(SURFACE) == set(TAU_A.names)
