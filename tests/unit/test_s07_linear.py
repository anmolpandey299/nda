"""B3/B4 — O1 linear / task-arithmetic merge and the release family [AUTH: 00 §10.1, §3.4]."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s07_fixtures import SURFACE, context, shuffled, surface_vector

from src.merge.family import (
    MergeFamilyError,
    MergeSpec,
    ReleaseFamily,
    build_release_family,
)
from src.merge.operators import (
    DARE,
    LINEAR,
    SVD_TRUNC,
    DeferredOperatorError,
    OperatorError,
    linear_merge,
    require_supported_operator,
)
from src.merge.settings import descendant_counts, merge_settings
from src.merge.updates import (
    FixtureTaskVector,
    SurfaceMismatchError,
    TaskVector,
    fixture_induced_update,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PERMITTED_K = descendant_counts(merge_settings(REPO_ROOT))
CTX = context()


def task_vector(tensors: Mapping[str, Any], *, origin: str = "unlabelled") -> FixtureTaskVector:
    """Local alias: every task vector in this module is a fixture."""
    return fixture_induced_update(tensors, context=CTX, origin=origin)


TAU_A = surface_vector("tau_A")
TAU_B = surface_vector("tau_B")


# ------------------------------------------------------------------ exact algebra
def test_alpha_one_is_exactly_the_protected_constituent() -> None:
    merged = linear_merge(TAU_A, TAU_B, alpha=1.0, context=CTX)
    for name in merged.names:
        assert np.array_equal(merged[name], TAU_A[name])
    assert merged.content_identity() == TAU_A.content_identity()


def test_alpha_zero_is_exactly_the_partner() -> None:
    merged = linear_merge(TAU_A, TAU_B, alpha=0.0, context=CTX)
    for name in merged.names:
        assert np.array_equal(merged[name], TAU_B[name])
    assert merged.content_identity() == TAU_B.content_identity()


def test_alpha_one_half_is_the_exact_arithmetic_midpoint() -> None:
    merged = linear_merge(TAU_A, TAU_B, alpha=0.5, context=CTX)
    for name in merged.names:
        assert np.array_equal(merged[name], np.float32(0.5) * (TAU_A[name] + TAU_B[name]))


@pytest.mark.parametrize("alpha", [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5])
def test_the_tensorwise_formula_holds_for_general_alpha(alpha: float) -> None:
    merged = linear_merge(TAU_A, TAU_B, alpha=alpha, context=CTX)
    for name in merged.names:
        expected = np.float32(alpha) * TAU_A[name] + np.float32(1.0 - alpha) * TAU_B[name]
        assert np.array_equal(merged[name], np.asarray(expected, dtype=np.float32))


def test_task_arithmetic_is_not_clamped_to_the_unit_interval() -> None:
    """00 §10.1 is task arithmetic; the registered schedule is S09's, not the operator's."""
    merged = linear_merge(TAU_A, TAU_B, alpha=2.0, context=CTX)
    for name in merged.names:
        assert np.allclose(merged[name], 2.0 * TAU_A[name] - 1.0 * TAU_B[name], rtol=1e-6)


def test_zero_updates_merge_to_zero() -> None:
    zeros = task_vector({name: np.zeros(shape) for name, shape in SURFACE.items()})
    merged = linear_merge(zeros, zeros, alpha=0.3, context=CTX)
    assert all(np.count_nonzero(merged[name]) == 0 for name in merged.names)


def test_negative_values_and_non_square_matrices_behave() -> None:
    negative = task_vector({name: -surface_vector("A")[name] for name in SURFACE})
    merged = linear_merge(negative, TAU_B, alpha=0.5, context=CTX)
    assert merged.shapes == TAU_B.shapes
    assert any(shape[0] != shape[1] for shape in merged.shapes.values())


def test_the_merge_is_affine_in_the_protected_constituent() -> None:
    """merge(A1 + A2, B, α) = merge(A1, B, α) + merge(A2, B, α) - (1-α)·B."""
    a1, a2 = surface_vector("A1"), surface_vector("A2")
    summed = task_vector({name: a1[name] + a2[name] for name in a1.names})
    alpha = 0.375
    left = linear_merge(summed, TAU_B, alpha=alpha, context=CTX)
    right_1 = linear_merge(a1, TAU_B, alpha=alpha, context=CTX)
    right_2 = linear_merge(a2, TAU_B, alpha=alpha, context=CTX)
    for name in left.names:
        expected = right_1[name] + right_2[name] - np.float32(1.0 - alpha) * TAU_B[name]
        assert np.allclose(left[name], expected, rtol=1e-5, atol=1e-7)


def test_mapping_order_does_not_change_the_result() -> None:
    first = linear_merge(TAU_A, TAU_B, alpha=0.4, context=CTX)
    second = linear_merge(shuffled(TAU_A), shuffled(TAU_B), alpha=0.4, context=CTX)
    assert first.content_identity() == second.content_identity()


# ------------------------------------------------------------------ fail-closed
def test_a_non_finite_coefficient_is_refused() -> None:
    for alpha in (float("nan"), float("inf")):
        with pytest.raises(OperatorError, match="not finite"):
            linear_merge(TAU_A, TAU_B, alpha=alpha, context=CTX)


def test_an_incompatible_surface_is_refused() -> None:
    partial = task_vector({name: TAU_B[name] for name in TAU_B.names[:-1]})
    with pytest.raises(SurfaceMismatchError):
        linear_merge(TAU_A, partial, alpha=0.5, context=CTX)


def test_ties_is_refused_by_the_operator_factory() -> None:
    """B14: no placeholder pretends to support TIES [AUTH: 00 §10.4]."""
    with pytest.raises(DeferredOperatorError, match="deferred"):
        require_supported_operator("TIES")
    for name in ("SLERP", "O4_QUANTIZATION"):
        with pytest.raises(DeferredOperatorError):
            require_supported_operator(name)
    with pytest.raises(OperatorError):
        require_supported_operator("SOMETHING_NEW")
    for name in (LINEAR, DARE, SVD_TRUNC):
        assert require_supported_operator(name) == name


def test_a_ties_merge_spec_cannot_be_constructed() -> None:
    with pytest.raises(DeferredOperatorError):
        MergeSpec(descendant_id="C1", operator="TIES", alpha=0.5, partner_id="B1")


# ------------------------------------------------------------------ release family
def specs(count: int, *, alphas: list[float] | None = None) -> list[MergeSpec]:
    coefficients = alphas or [0.5] * count
    return [
        MergeSpec(
            descendant_id=f"C{index + 1}",
            operator=LINEAR,
            alpha=coefficients[index],
            partner_id=f"B{index + 1}",
        )
        for index in range(count)
    ]


def partners(count: int) -> dict[str, TaskVector]:
    return {f"B{index + 1}": surface_vector(f"partner_{index}") for index in range(count)}


@pytest.mark.parametrize("k", [1, 2, 4])
def test_the_registered_descendant_counts_are_supported(k: int) -> None:
    family = build_release_family(
        protected=TAU_A,
        partners=partners(k),
        specs=specs(k),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    assert family.k == k
    assert set(PERMITTED_K) == {1, 2, 4}


def test_k_eight_is_deferred_and_refused() -> None:
    with pytest.raises(MergeFamilyError, match="not a registered value"):
        build_release_family(
            protected=TAU_A,
            partners=partners(8),
            specs=specs(8),
            permitted_k=PERMITTED_K,
            context=CTX,
        )


def test_every_descendant_reuses_the_same_protected_constituent() -> None:
    """00 §3.4: the base training of A is not re-run per descendant."""
    family = build_release_family(
        protected=TAU_A,
        partners=partners(4),
        specs=specs(4),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    assert {r.protected_identity for r in family.descendants} == {TAU_A.content_identity()}
    assert family.protected_identity == TAU_A.content_identity()


def test_a_different_a_in_one_descendant_is_refused() -> None:
    """B17 item 1 / F7: descendant 2 built from a second draw of A.

    `build_release_family` derives the protected identity from the first issued result and
    then requires every other to match, and `ReleaseFamily` is factory-issued, so there is no
    parameter through which a caller can claim A1 for a descendant built from A2.
    """
    other_a = surface_vector("tau_A_prime")
    assert other_a.content_identity() != TAU_A.content_identity()

    with pytest.raises(TypeError, match="factory-issued"):
        ReleaseFamily(
            protected_identity=TAU_A.content_identity(),
            surface_identity="d" * 64,
            context=CTX,
            descendants=(),
        )

    import inspect

    assert "protected_identity" not in inspect.signature(build_release_family).parameters
    honest = build_release_family(
        protected=TAU_A,
        partners=partners(1),
        specs=specs(1),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    assert honest.protected_identity == TAU_A.content_identity()


def test_duplicate_descendant_ids_are_refused() -> None:
    """B17 item 3."""
    duplicated = [
        MergeSpec(descendant_id="C1", operator=LINEAR, alpha=0.5, partner_id="B1"),
        MergeSpec(descendant_id="C1", operator=LINEAR, alpha=0.5, partner_id="B2"),
    ]
    with pytest.raises(MergeFamilyError, match="duplicate descendant id"):
        build_release_family(
            protected=TAU_A,
            partners=partners(2),
            specs=duplicated,
            permitted_k=PERMITTED_K,
            context=CTX,
        )


def test_the_partner_coefficient_association_survives_reordering() -> None:
    """B17 item 2: coefficients are never sorted independently of their partners."""
    alphas = [0.1, 0.4, 0.6, 0.9]
    forward = specs(4, alphas=alphas)
    available = partners(4)
    first = build_release_family(
        protected=TAU_A,
        partners=available,
        specs=forward,
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    second = build_release_family(
        protected=TAU_A,
        partners=available,
        specs=list(reversed(forward)),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    for descendant_id in ("C1", "C2", "C3", "C4"):
        assert (
            first.by_id(descendant_id).result_identity
            == second.by_id(descendant_id).result_identity
        )
        assert first.by_id(descendant_id).alpha == second.by_id(descendant_id).alpha
    assert [r.descendant_id for r in first.descendants] != [
        r.descendant_id for r in second.descendants
    ]


def test_swapping_two_coefficients_changes_the_descendants() -> None:
    available = partners(2)
    straight = build_release_family(
        protected=TAU_A,
        partners=available,
        specs=specs(2, alphas=[0.2, 0.8]),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    swapped = build_release_family(
        protected=TAU_A,
        partners=available,
        specs=specs(2, alphas=[0.8, 0.2]),
        permitted_k=PERMITTED_K,
        context=CTX,
    )
    assert straight.by_id("C1").result_identity != swapped.by_id("C1").result_identity


def test_an_unsupplied_partner_is_refused() -> None:
    with pytest.raises(MergeFamilyError, match="no task vector supplied"):
        build_release_family(
            protected=TAU_A,
            partners={"B1": TAU_B},
            specs=specs(2),
            permitted_k=PERMITTED_K,
            context=CTX,
        )
