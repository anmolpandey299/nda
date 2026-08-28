"""O2 DARE: drop, rescale, and mask evidence that is derived rather than incidental.

The mask *scope* attacks (one merge seed, role substreams, replay protection) live in
`test_s07_repair_attacks.py`; this module owns the operator mathematics and the DARE
diagnostic surface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from s07_fixtures import SURFACE, context, matrix, shuffled, surface_vector

from src.merge.diagnostics import (
    DiagnosticError,
    dare_perturbation_from_results,
    dare_theoretical_perturbation,
)
from src.merge.family import (
    MergeFamilyError,
    MergeSpec,
    build_dare_descendant,
    build_linear_descendant,
)
from src.merge.masks import PROTECTED_ROLE, derive_mask
from src.merge.operators import (
    DARE,
    LINEAR,
    SVD_TRUNC,
    OperatorError,
    dare_merge,
    dare_transform,
    linear_merge,
)
from src.merge.settings import (
    dare_candidates,
    dare_mask_scheme,
    dare_minimum,
    fixture_dare_merge_seed,
    gate_statuses,
    merge_settings,
)
from src.merge.updates import fixture_induced_update

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = merge_settings(REPO_ROOT)
CTX = context()
TAU_A = surface_vector("tau_A")
TAU_B = surface_vector("tau_B")
SEED = fixture_dare_merge_seed(SETTINGS)


def transform(vector: object, p: float, *, seed: int = SEED) -> object:
    return dare_transform(
        vector,  # type: ignore[arg-type]
        drop_probability=p,
        merge_seed=seed,
        role=PROTECTED_ROLE,
        context=CTX,
    )


def linear_spec(alpha: float = 0.5) -> MergeSpec:
    return MergeSpec(descendant_id="C1", operator=LINEAR, alpha=alpha, partner_id="B1")


def dare_spec(alpha: float = 0.5, *, p: float = 0.5, seed: int = SEED) -> MergeSpec:
    return MergeSpec(
        descendant_id="C1",
        operator=DARE,
        alpha=alpha,
        partner_id="B1",
        drop_probability=p,
        dare_merge_seed=seed,
    )


# ------------------------------------------------------------------ drop and rescale
def test_p_zero_is_the_exact_identity_transformation() -> None:
    result = transform(TAU_A, 0.0)
    for name in TAU_A.names:
        assert np.array_equal(result.vector[name], TAU_A[name])  # type: ignore[attr-defined]
    assert result.vector.content_identity() == TAU_A.content_identity()  # type: ignore[attr-defined]
    assert result.mask.dropped == 0  # type: ignore[attr-defined]
    assert result.mask.retained_fraction == 1.0  # type: ignore[attr-defined]


@pytest.mark.parametrize("p", [1.0, 1.5, -0.1, -1.0])
def test_an_invalid_drop_probability_is_refused(p: float) -> None:
    with pytest.raises(OperatorError, match=r"outside \[0, 1\)"):
        transform(TAU_A, p)


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75, 0.9])
def test_survivors_are_rescaled_by_exactly_one_over_one_minus_p(p: float) -> None:
    result = transform(TAU_A, p)
    scale = np.float32(1.0 / (1.0 - p))
    for name in TAU_A.names:
        keep = result.mask.keep_array(name).reshape(TAU_A.shapes[name])  # type: ignore[attr-defined]
        assert np.array_equal(result.vector[name][keep], TAU_A[name][keep] * scale)  # type: ignore[attr-defined]
        assert np.count_nonzero(result.vector[name][~keep]) == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75, 0.9])
def test_the_retained_fraction_tracks_one_minus_p(p: float) -> None:
    """A TEST tolerance over a large deterministic bank, not a scientific result."""
    bank = fixture_induced_update(
        {f"w{index}": matrix(f"bank/{index}", (60, 60)) for index in range(4)},
        context=CTX,
        origin="bank",
    )
    result = transform(bank, p)
    assert result.mask.retained_fraction == pytest.approx(1.0 - p, abs=0.02)  # type: ignore[attr-defined]
    assert result.mask.total == 4 * 60 * 60  # type: ignore[attr-defined]


def test_dropped_coordinates_are_exactly_zero_not_merely_small() -> None:
    result = transform(TAU_A, 0.5)
    for name in TAU_A.names:
        dropped = result.vector[name][~result.mask.keep_array(name).reshape(TAU_A.shapes[name])]  # type: ignore[attr-defined]
        assert dropped.size > 0
        assert np.all(dropped == 0.0)


# ------------------------------------------------------------------ mask determinism
def test_the_same_vector_seed_and_p_give_a_bitwise_identical_mask() -> None:
    first, second = transform(TAU_A, 0.5), transform(TAU_A, 0.5)
    assert first.mask.identity() == second.mask.identity()  # type: ignore[attr-defined]
    assert first.vector.content_identity() == second.vector.content_identity()  # type: ignore[attr-defined]


def test_a_different_merge_seed_gives_a_different_mask() -> None:
    first, second = transform(TAU_A, 0.5), transform(TAU_A, 0.5, seed=SEED + 1)
    assert first.mask.identity() != second.mask.identity()  # type: ignore[attr-defined]
    assert first.vector.content_identity() != second.vector.content_identity()  # type: ignore[attr-defined]


def test_a_different_p_or_input_gives_a_different_mask_identity() -> None:
    base = transform(TAU_A, 0.5).mask.identity()  # type: ignore[attr-defined]
    assert transform(TAU_A, 0.25).mask.identity() != base  # type: ignore[attr-defined]
    assert transform(TAU_B, 0.5).mask.identity() != base  # type: ignore[attr-defined]


def test_tensor_order_permutation_leaves_every_per_tensor_mask_identical() -> None:
    first, second = transform(TAU_A, 0.5), transform(shuffled(TAU_A), 0.5)
    for name in TAU_A.names:
        assert np.array_equal(first.mask.keep_array(name), second.mask.keep_array(name))  # type: ignore[attr-defined]
    assert first.mask.identity() == second.mask.identity()  # type: ignore[attr-defined]


def test_the_mask_records_everything_needed_to_reproduce_it() -> None:
    document = transform(TAU_A, 0.5).mask.as_dict()  # type: ignore[attr-defined]
    assert document["scheme"] == CTX.mask_scheme == dare_mask_scheme(SETTINGS)
    assert document["dare_merge_seed"] == SEED
    assert document["constituent_role"] == PROTECTED_ROLE
    assert document["drop_probability"] == 0.5
    assert document["retained_coordinates"] and document["dropped_coordinates"]
    assert document["total_coordinates"] == sum(r * c for r, c in SURFACE.values())
    assert document["operator_version"] == CTX.operator_version


# ------------------------------------------------------------------ merge semantics
def test_a_dare_descendant_differs_from_the_linear_baseline() -> None:
    baseline = linear_merge(TAU_A, TAU_B, alpha=0.5, context=CTX)
    merged, _, _ = dare_merge(
        TAU_A, TAU_B, alpha=0.5, drop_probability=0.5, merge_seed=SEED, context=CTX
    )
    assert merged.content_identity() != baseline.content_identity()


def test_p_zero_dare_merge_reproduces_the_linear_baseline() -> None:
    baseline = linear_merge(TAU_A, TAU_B, alpha=0.4, context=CTX)
    merged, _, _ = dare_merge(
        TAU_A, TAU_B, alpha=0.4, drop_probability=0.0, merge_seed=SEED, context=CTX
    )
    assert merged.content_identity() == baseline.content_identity()


def test_the_transform_is_constituent_wise_not_post_merge() -> None:
    """alpha·DARE(A) + (1-alpha)·DARE(B), never DARE(alpha·A + (1-alpha)·B)."""
    merged, protected, partner = dare_merge(
        TAU_A, TAU_B, alpha=0.5, drop_probability=0.5, merge_seed=SEED, context=CTX
    )
    for name in merged.names:
        expected = np.float32(0.5) * protected.vector[name] + np.float32(0.5) * partner.vector[name]
        assert np.array_equal(merged[name], np.asarray(expected, dtype=np.float32))


def test_a_dare_spec_without_a_declared_merge_seed_is_refused() -> None:
    with pytest.raises(MergeFamilyError, match="declared merge seed"):
        MergeSpec(
            descendant_id="C1", operator=DARE, alpha=0.5, partner_id="B1", drop_probability=0.5
        )


def test_dare_parameters_on_a_non_dare_spec_are_refused() -> None:
    for operator in (LINEAR, SVD_TRUNC):
        with pytest.raises(MergeFamilyError):
            MergeSpec(
                descendant_id="C1",
                operator=operator,
                alpha=0.5,
                partner_id="B1",
                drop_probability=0.5,
                dare_merge_seed=SEED,
            )


def test_the_artifact_is_reconstructable_from_its_recorded_identity() -> None:
    first = build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX)
    second = build_dare_descendant(protected=TAU_A, partner=TAU_B, spec=dare_spec(), context=CTX)
    assert first.identity() == second.identity()
    document = first.as_dict()
    for key in (
        "operator",
        "operator_version",
        "execution_context_sha256",
        "protected_update_sha256",
        "partner_update_sha256",
        "alpha",
        "drop_probability",
        "dare_merge_seed",
        "mask_scheme",
        "protected_mask_sha256",
        "partner_mask_sha256",
        "result_update_sha256",
        "parameter_surface_sha256",
        "dtype",
    ):
        assert document[key] is not None, key


# ------------------------------------------------------------------ diagnostics
def test_the_theoretical_reference_matches_the_spec_value() -> None:
    assert dare_theoretical_perturbation(0.25) == pytest.approx(np.sqrt(1.0 / 3.0))
    assert dare_theoretical_perturbation(0.25) == pytest.approx(0.5773502691896257)
    assert dare_theoretical_perturbation(0.0) == 0.0
    assert dare_theoretical_perturbation(0.9) > dare_theoretical_perturbation(0.75)
    with pytest.raises(DiagnosticError):
        dare_theoretical_perturbation(1.0)


def test_the_realised_perturbation_is_zero_at_p_zero_and_grows_with_p() -> None:
    baseline = build_linear_descendant(
        protected=TAU_A, partner=TAU_B, spec=linear_spec(), context=CTX
    )
    values = []
    for p in (0.0, 0.25, 0.5, 0.75, 0.9):
        merged = build_dare_descendant(
            protected=TAU_A, partner=TAU_B, spec=dare_spec(p=p), context=CTX
        )
        values.append(dare_perturbation_from_results(merged, baseline))
    assert values[0] == 0.0
    assert values == sorted(values)


def test_a_zero_baseline_is_an_explicit_undefined_state_not_inf() -> None:
    zeros = fixture_induced_update(
        {name: np.zeros(shape) for name, shape in SURFACE.items()}, context=CTX
    )
    baseline = build_linear_descendant(
        protected=zeros, partner=zeros, spec=linear_spec(), context=CTX
    )
    merged = build_dare_descendant(protected=zeros, partner=zeros, spec=dare_spec(), context=CTX)
    with pytest.raises(DiagnosticError, match="UNDEFINED_ZERO_DENOMINATOR"):
        dare_perturbation_from_results(merged, baseline)


# ------------------------------------------------------------------ no gate result
def test_no_dare_gate_result_is_fabricated() -> None:
    assert gate_statuses(SETTINGS)["DARE_GATE_STATUS"] == "NOT_RUN"
    assert dare_candidates(SETTINGS) == (0.25, 0.5, 0.75, 0.9)
    assert dare_minimum(SETTINGS) == 0.25


def test_the_selected_drop_probability_cannot_be_read() -> None:
    from src.materials import UncalibratedConstantError, material

    with pytest.raises(UncalibratedConstantError, match="REQUIRED_NOT_CALIBRATED"):
        material(SETTINGS, "dare_selected_drop_probability")


def test_the_mask_seed_is_one_merge_level_value() -> None:
    """Architect adjudication: no two independently caller-selectable scientific seeds."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(MergeSpec)}
    assert "dare_merge_seed" in fields
    assert not fields & {"protected_mask_seed", "partner_mask_seed"}
    import inspect

    assert "merge_seed" in inspect.signature(derive_mask).parameters
    assert "seed" not in inspect.signature(derive_mask).parameters
