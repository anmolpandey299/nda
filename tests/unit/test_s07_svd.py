"""B9–B13 — O3 SVD-TRUNC-MERGE: truncation, spectra, rank diagnostics, conditionality."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s07_fixtures import (
    SURFACE,
    context,
    diagonal_vector,
    low_rank_vector,
    shuffled,
    surface_vector,
)

from src.merge.diagnostics import (
    DiagnosticError,
    _relative_frobenius_error,
    numerical_effective_rank,
    rank_report,
    singular_values,
    spectral_energy,
    stable_rank,
    theoretical_rank_bound,
)
from src.merge.family import MergeFamilyError, MergeSpec, build_o3_descendant
from src.merge.operators import (
    O3_GATE_STATUS,
    O3_STATUS,
    SVD_TRUNC,
    OperatorError,
    linear_merge,
    svd_trunc_merge,
    truncate,
    truncate_vector,
    truncations,
)
from src.merge.settings import (
    effective_rank_tolerance,
    gate_statuses,
    merge_settings,
    svd_backend,
    svd_candidate_ranks,
)
from src.merge.updates import FixtureTaskVector, fixture_induced_update

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = merge_settings(REPO_ROOT)
TOLERANCE = effective_rank_tolerance(SETTINGS)
CTX = context()
TAU_A = surface_vector("tau_A")
TAU_B = surface_vector("tau_B")


def task_vector(tensors: Mapping[str, Any], *, origin: str = "unlabelled") -> FixtureTaskVector:
    """Local alias: every task vector in this module is a fixture."""
    return fixture_induced_update(tensors, context=CTX, origin=origin)


# ------------------------------------------------------------------ truncation
@pytest.mark.parametrize("rank", [1, 2, 4, 6])
def test_the_truncated_matrix_has_rank_at_most_s(rank: int) -> None:
    """Rank is judged by the frozen 00 §24A.6 criterion, not an ad-hoc absolute tolerance.

    Casting T_s to the float32 artifact dtype leaves residual singular values around 1e-8 of
    the leading one — far below the frozen 1e-6 threshold, and far above a float64-era
    absolute tol of 1e-10. The spec's own criterion is the one that means something here.
    """
    for name in TAU_A.names:
        result = truncate(TAU_A[name], rank=rank, context=CTX)
        assert numerical_effective_rank(result.matrix, tolerance=TOLERANCE) <= rank
        assert result.retained_rank <= rank


def test_full_retention_reconstructs_the_input() -> None:
    for name in TAU_A.names:
        full = min(TAU_A.shapes[name])
        result = truncate(TAU_A[name], rank=full, context=CTX)
        assert np.allclose(result.matrix, TAU_A[name], atol=1e-12)
        assert result.discarded_energy == pytest.approx(0.0, abs=1e-18)


def test_a_rank_beyond_the_matrix_dimension_still_reconstructs() -> None:
    name = TAU_A.names[0]
    result = truncate(TAU_A[name], rank=10_000, context=CTX)
    assert np.allclose(result.matrix, TAU_A[name], atol=1e-12)
    assert result.retained_rank == min(TAU_A.shapes[name])
    assert result.requested_rank == 10_000


def test_a_zero_matrix_truncates_to_zero() -> None:
    result = truncate(np.zeros((6, 4)), rank=3, context=CTX)
    assert np.count_nonzero(result.matrix) == 0
    assert result.total_energy == pytest.approx(0.0)


def test_a_known_diagonal_spectrum_retains_exactly_the_top_s() -> None:
    spectrum = [9.0, 7.0, 5.0, 3.0, 1.0, 0.5]
    vector = diagonal_vector(spectrum)
    result = truncate(vector["diag"], rank=3, context=CTX)
    retained = np.sort(np.diag(result.matrix))[::-1][:3]
    assert np.allclose(retained, [9.0, 7.0, 5.0])
    assert np.allclose(np.sort(result.singular_values)[::-1][:6], spectrum)
    assert result.discarded_energy == pytest.approx(3.0**2 + 1.0**2 + 0.5**2)


@pytest.mark.parametrize("rank", [0, -1, -10])
def test_a_non_positive_rank_is_refused(rank: int) -> None:
    with pytest.raises(OperatorError, match="positive integer"):
        truncate(TAU_A[TAU_A.names[0]], rank=rank, context=CTX)


def test_a_non_finite_matrix_cannot_be_truncated() -> None:
    array = np.ones((4, 4))
    array[0, 0] = np.nan
    with pytest.raises(OperatorError, match="NaN or Inf"):
        truncate(array, rank=2, context=CTX)


# ------------------------------------------------------------------ B10 determinism
def test_truncation_is_deterministic_and_order_invariant() -> None:
    first = truncate_vector(TAU_A, rank=3, context=CTX)
    second = truncate_vector(shuffled(TAU_A), rank=3, context=CTX)
    assert first.content_identity() == second.content_identity()
    assert (
        truncate_vector(TAU_A, rank=3, context=CTX).content_identity() == first.content_identity()
    )


def test_identity_is_taken_over_reconstructions_never_over_singular_vectors() -> None:
    """A degenerate spectrum has no unique basis; the reconstruction is still determined."""
    degenerate = np.diag([2.0, 2.0, 2.0, 0.5])
    first = truncate(degenerate, rank=3, context=CTX)
    second = truncate(degenerate.copy(), rank=3, context=CTX)
    assert np.allclose(first.matrix, second.matrix)
    assert np.allclose(np.sort(first.singular_values)[::-1], [2.0, 2.0, 2.0, 0.5])
    assert np.allclose(first.matrix, np.diag([2.0, 2.0, 2.0, 0.0]))


def test_the_svd_backend_is_pinned_in_config() -> None:
    assert CTX.svd_backend == svd_backend(SETTINGS)
    assert "full_matrices=False" in CTX.svd_backend and "float64" in CTX.svd_backend


# ------------------------------------------------------------------ B11 energy
def test_discarded_frobenius_energy_matches_the_singular_tail() -> None:
    """Two tolerances, for two different reasons.

    `discarded_energy` comes from the float64 SVD workspace, so it matches the tail to
    round-off. The residual is measured on the float32 *artifact*, so it additionally carries
    the deterministic cast error — a real property of the artifact, not slack in the test.
    """
    for rank in (1, 2, 3):
        parts = truncations(TAU_A, rank=rank, context=CTX)
        for name, part in parts.items():
            tail = float(np.sum(singular_values(TAU_A[name].astype(np.float64))[rank:] ** 2))
            assert part.discarded_energy == pytest.approx(tail, rel=1e-10)
            residual = float(
                np.sum((part.matrix.astype(np.float64) - TAU_A[name].astype(np.float64)) ** 2)
            )
            assert residual == pytest.approx(tail, rel=1e-5)


def test_the_information_floor_is_the_relative_truncation_error() -> None:
    """e_floor(s) = ||T_s(ΔW_A) - ΔW_A||_F / ||ΔW_A||_F [AUTH: 00 §24A.1]."""
    for rank in (1, 2, 3):
        energy = spectral_energy(truncations(TAU_A, rank=rank, context=CTX))
        direct = _relative_frobenius_error(truncate_vector(TAU_A, rank=rank, context=CTX), TAU_A)
        assert energy.floor() == pytest.approx(direct, rel=1e-8)


def test_the_floor_decreases_as_more_rank_is_retained() -> None:
    floors = [
        spectral_energy(truncations(TAU_A, rank=r, context=CTX)).floor() for r in (1, 2, 3, 4)
    ]
    assert floors == sorted(floors, reverse=True)


def test_a_zero_energy_constituent_makes_the_floor_undefined() -> None:
    zeros = task_vector({name: np.zeros(shape) for name, shape in SURFACE.items()})
    with pytest.raises(DiagnosticError, match="UNDEFINED_ZERO_DENOMINATOR"):
        spectral_energy(truncations(zeros, rank=2, context=CTX)).floor()


# ------------------------------------------------------------------ B12 rank diagnostics
def test_the_effective_rank_tolerance_is_the_frozen_criterion() -> None:
    assert TOLERANCE == 1e-06


def test_the_effective_rank_counts_the_frozen_criterion() -> None:
    vector = diagonal_vector([1.0, 1e-3, 1e-7, 0.0])
    assert numerical_effective_rank(vector["diag"], tolerance=TOLERANCE) == 2
    assert numerical_effective_rank(vector["diag"], tolerance=1e-8) == 3


def test_a_zero_matrix_has_effective_rank_zero_and_stable_rank_zero() -> None:
    zero = np.zeros((5, 5))
    assert numerical_effective_rank(zero, tolerance=TOLERANCE) == 0
    assert stable_rank(zero) == 0.0


def test_the_stable_rank_is_the_frobenius_over_spectral_ratio() -> None:
    array = np.diag([3.0, 4.0])
    assert stable_rank(array) == pytest.approx((9.0 + 16.0) / 16.0)
    assert stable_rank(np.eye(5)) == pytest.approx(5.0)
    assert stable_rank(np.diag([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_an_out_of_range_tolerance_is_refused() -> None:
    for tolerance in (0.0, -1e-6, 2.0):
        with pytest.raises(DiagnosticError):
            numerical_effective_rank(np.eye(3), tolerance=tolerance)


def test_a_two_parent_linear_merge_is_bounded_by_twice_the_constituent_rank() -> None:
    """00 §24A.6: linear rank <= 2r, O3 rank <= 2s, before subspace overlap."""
    rank = 3
    protected = low_rank_vector("A", rank=rank)
    partner = low_rank_vector("B", rank=rank)
    merged = linear_merge(protected, partner, alpha=0.5, context=CTX)
    assert theoretical_rank_bound(constituent_ranks=[rank, rank]) == 2 * rank
    for report in rank_report(merged, constituent_ranks=[rank, rank], tolerance=TOLERANCE):
        assert report.effective_rank <= report.theoretical_bound
        assert report.theoretical_bound <= 2 * rank


def test_an_o3_merge_is_bounded_by_twice_the_retained_rank() -> None:
    retained = 2
    merged, _, _ = svd_trunc_merge(TAU_A, TAU_B, alpha=0.5, rank=retained, context=CTX)
    for report in rank_report(merged, constituent_ranks=[retained, retained], tolerance=TOLERANCE):
        assert report.effective_rank <= 2 * retained
        assert report.as_dict()["effective_rank_tolerance"] == TOLERANCE


def test_a_non_positive_constituent_rank_is_refused() -> None:
    with pytest.raises(DiagnosticError):
        theoretical_rank_bound(constituent_ranks=[0])
    with pytest.raises(DiagnosticError):
        theoretical_rank_bound(constituent_ranks=[])


# ------------------------------------------------------------------ merge composition
def test_the_o3_merge_is_the_truncated_affine_combination() -> None:
    alpha, rank = 0.375, 3
    merged, protected_parts, partner_parts = svd_trunc_merge(
        TAU_A, TAU_B, alpha=alpha, rank=rank, context=CTX
    )
    for name in merged.names:
        expected = alpha * protected_parts[name].matrix + (1.0 - alpha) * partner_parts[name].matrix
        assert np.allclose(merged[name], expected)


def test_full_rank_o3_reproduces_the_linear_merge() -> None:
    full = max(max(shape) for shape in SURFACE.values())
    merged, _, _ = svd_trunc_merge(TAU_A, TAU_B, alpha=0.5, rank=full, context=CTX)
    baseline = linear_merge(TAU_A, TAU_B, alpha=0.5, context=CTX)
    for name in merged.names:
        assert np.allclose(merged[name], baseline[name], atol=1e-12)


def test_an_o3_result_cannot_claim_linear_identity() -> None:
    """An O3 artifact exists only through the O3 factory [R2]."""
    from src.merge.family import build_linear_descendant

    spec = MergeSpec(
        descendant_id="C1", operator=SVD_TRUNC, alpha=0.5, partner_id="B1", retained_rank=3
    )
    built = build_o3_descendant(protected=TAU_A, partner=TAU_B, spec=spec, context=CTX)
    assert built.as_dict()["retained_rank"] == 3
    assert built.as_dict()["operator"] == SVD_TRUNC
    with pytest.raises(MergeFamilyError, match="builds .* only"):
        build_linear_descendant(protected=TAU_A, partner=TAU_B, spec=spec, context=CTX)


def test_an_o3_spec_without_a_rank_is_refused() -> None:
    with pytest.raises(MergeFamilyError, match="needs its rank"):
        MergeSpec(descendant_id="C1", operator=SVD_TRUNC, alpha=0.5, partner_id="B1")


# ------------------------------------------------------------------ B13 conditionality
def test_o3_is_implemented_but_not_activated() -> None:
    assert O3_STATUS == "PRE_REGISTERED_CONDITIONAL"
    assert O3_GATE_STATUS == "NOT_RUN"
    assert gate_statuses(SETTINGS) == {
        "O3_STATUS": "PRE_REGISTERED_CONDITIONAL",
        "DARE_GATE_STATUS": "NOT_RUN",
        "O3_GATE_STATUS": "NOT_RUN",
    }


def test_the_candidate_ranks_are_a_grid_not_a_selection() -> None:
    assert svd_candidate_ranks(SETTINGS) == (24, 16, 8, 4)


def test_neither_a_selected_rank_nor_a_primary_operator_can_be_read() -> None:
    """B13: both are gate outcomes, and reading either fails closed [AUTH: 00 §24A.7]."""
    from src.materials import UncalibratedConstantError, material

    for key in ("svd_selected_retained_rank", "primary_lossy_operator"):
        with pytest.raises(UncalibratedConstantError, match="REQUIRED_NOT_CALIBRATED"):
            material(SETTINGS, key)
