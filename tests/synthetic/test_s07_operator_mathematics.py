"""Planted-truth operator mathematics over a deterministic bank [AUTH: 00 §10, §24.1, §24A].

Every claim here is a property of the merge transformation on synthetic matrices. None of it
is a privacy result, a utility result or a gate verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s07_fixtures import context, matrix

from src.merge.diagnostics import (
    _relative_frobenius_error,
    dare_perturbation_from_results,
    dare_theoretical_perturbation,
    numerical_effective_rank,
    singular_values,
    spectral_energy,
    stable_rank,
)
from src.merge.family import MergeSpec, build_dare_descendant, build_linear_descendant
from src.merge.masks import PROTECTED_ROLE
from src.merge.operators import (
    dare_merge,
    dare_transform,
    linear_merge,
    svd_trunc_merge,
    truncate,
    truncate_vector,
    truncations,
)
from src.merge.settings import effective_rank_tolerance, merge_settings
from src.merge.updates import FixtureTaskVector, fixture_induced_update

REPO_ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = effective_rank_tolerance(merge_settings(REPO_ROOT))
CTX = context()
SEED = 70701


def task_vector(tensors: Mapping[str, Any], *, origin: str = "unlabelled") -> FixtureTaskVector:
    """Local alias: every task vector in this module is a fixture."""
    return fixture_induced_update(tensors, context=CTX, origin=origin)


#: A larger bank than the unit fixtures, so empirical rates are meaningful.
BANK_A = task_vector(
    {f"w{index}": matrix(f"bankA/{index}", (48, 40)) for index in range(6)}, origin="bank_A"
)
BANK_B = task_vector(
    {f"w{index}": matrix(f"bankB/{index}", (48, 40)) for index in range(6)}, origin="bank_B"
)

LINEAR_SPEC = MergeSpec(
    descendant_id="C1", operator="O1_LINEAR_TASK_ARITHMETIC", alpha=0.5, partner_id="B1"
)


def dare_spec(p: float) -> MergeSpec:
    return MergeSpec(
        descendant_id="C1",
        operator="O2_DARE",
        alpha=0.5,
        partner_id="B1",
        drop_probability=p,
        dare_merge_seed=SEED,
    )


# ------------------------------------------------------------------ DARE
@pytest.mark.parametrize("p", [0.25, 0.5, 0.75, 0.9])
def test_survivors_are_exactly_rescaled_and_dropped_are_exactly_zero(p: float) -> None:
    result = dare_transform(
        BANK_A, drop_probability=p, merge_seed=SEED, role=PROTECTED_ROLE, context=CTX
    )
    scale = 1.0 / (1.0 - p)
    for name in BANK_A.names:
        keep = result.mask.keep_array(name).reshape(BANK_A.shapes[name])
        assert np.array_equal(result.vector[name][keep], BANK_A[name][keep] * scale)
        assert np.all(result.vector[name][~keep] == 0.0)


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75, 0.9])
def test_the_empirical_retained_fraction_matches_one_minus_p(p: float) -> None:
    """Test tolerance over 11,520 coordinates. Not a scientific result [AUTH: 00 §24.1]."""
    result = dare_transform(
        BANK_A, drop_probability=p, merge_seed=SEED, role=PROTECTED_ROLE, context=CTX
    )
    assert result.mask.total == 6 * 48 * 40
    assert result.mask.retained_fraction == pytest.approx(1.0 - p, abs=0.015)


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
def test_the_transform_preserves_the_task_vector_in_expectation(p: float) -> None:
    """Drop-then-rescale is mean-preserving: that is why survivors are scaled by 1/(1-p).

    The tolerance is the estimator's own analytic standard error, not a hand-picked number.
    Each coordinate contributes x/(1-p) with probability 1-p and 0 otherwise, so the total
    has variance Sigma x^2 * p/(1-p); averaging `draws` independent seeds divides that by
    `draws`. Four standard errors is the band.
    """
    draws = 12
    total = 0.0
    for offset in range(draws):
        transformed = dare_transform(
            BANK_A,
            drop_probability=p,
            merge_seed=SEED + offset,
            role=PROTECTED_ROLE,
            context=CTX,
        ).vector
        total += float(
            sum(float(np.sum(transformed[n].astype(np.float64))) for n in transformed.names)
        )
    average = total / draws

    reference = float(sum(float(np.sum(BANK_A[n].astype(np.float64))) for n in BANK_A.names))
    energy = float(sum(float(np.sum(BANK_A[n].astype(np.float64) ** 2)) for n in BANK_A.names))
    standard_error = float(np.sqrt(energy * p / (1.0 - p) / draws))
    assert abs(average - reference) < 4.0 * standard_error, (average, reference, standard_error)


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75, 0.9])
def test_the_realised_perturbation_sits_near_the_theoretical_reference(p: float) -> None:
    """Reported beside the theory value, never gated on it [AUTH: 00 §24.1]."""
    baseline = build_linear_descendant(
        protected=BANK_A, partner=BANK_B, spec=LINEAR_SPEC, context=CTX
    )
    merged = build_dare_descendant(protected=BANK_A, partner=BANK_B, spec=dare_spec(p), context=CTX)
    realised = dare_perturbation_from_results(merged, baseline)
    theory = dare_theoretical_perturbation(p)
    assert realised > 0.0
    assert 0.3 * theory < realised < 3.0 * theory, (p, realised, theory)


def test_the_theory_helper_is_monotone_and_matches_its_spec_values() -> None:
    values = [dare_theoretical_perturbation(p) for p in (0.0, 0.25, 0.5, 0.75, 0.9)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[1] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert dare_theoretical_perturbation(0.5) == pytest.approx(1.0)


def test_a_single_transform_is_reproducible_across_repeated_calls() -> None:
    identities = {
        dare_transform(
            BANK_A, drop_probability=0.5, merge_seed=SEED, role=PROTECTED_ROLE, context=CTX
        ).vector.content_identity()
        for _ in range(4)
    }
    assert len(identities) == 1


def test_distinct_seeds_give_distinct_masks_across_a_seed_sweep() -> None:
    masks = {
        dare_transform(
            BANK_A, drop_probability=0.5, merge_seed=seed, role=PROTECTED_ROLE, context=CTX
        ).mask.identity()
        for seed in range(SEED, SEED + 16)
    }
    assert len(masks) == 16


# ------------------------------------------------------------------ O3
@pytest.mark.parametrize("rank", [24, 16, 8, 4])
def test_the_candidate_ranks_truncate_correctly_on_the_bank(rank: int) -> None:
    parts = truncations(BANK_A, rank=rank, context=CTX)
    for name, part in parts.items():
        assert numerical_effective_rank(part.matrix, tolerance=TOLERANCE) <= rank
        tail = float(np.sum(singular_values(BANK_A[name].astype(np.float64))[rank:] ** 2))
        assert part.discarded_energy == pytest.approx(tail, rel=1e-10)


def test_the_information_floor_is_monotone_in_the_retained_rank() -> None:
    floors = [
        spectral_energy(truncations(BANK_A, rank=r, context=CTX)).floor() for r in (4, 8, 16, 24)
    ]
    assert floors == sorted(floors, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in floors)


def test_the_floor_equals_the_relative_truncation_error_on_the_bank() -> None:
    for rank in (4, 16):
        energy = spectral_energy(truncations(BANK_A, rank=rank, context=CTX))
        assert energy.floor() == pytest.approx(
            _relative_frobenius_error(truncate_vector(BANK_A, rank=rank, context=CTX), BANK_A),
            rel=1e-9,
        )


def test_a_planted_spectrum_is_retained_exactly() -> None:
    spectrum = np.array([16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125])
    left = np.linalg.qr(matrix("planted/U", (12, 8)))[0]
    right = np.linalg.qr(matrix("planted/V", (8, 8)))[0]
    planted = (left * spectrum) @ right.T
    assert np.allclose(np.sort(singular_values(planted))[::-1], spectrum, atol=1e-10)

    for rank in (1, 3, 5):
        part = truncate(planted, rank=rank, context=CTX)
        assert np.allclose(
            np.sort(singular_values(part.matrix.astype(np.float64)))[::-1][:rank],
            spectrum[:rank],
            rtol=1e-5,
        )
        assert part.discarded_energy == pytest.approx(float(np.sum(spectrum[rank:] ** 2)))


def test_effective_and_stable_rank_on_a_planted_spectrum() -> None:
    spectrum = np.array([1.0, 0.1, 1e-4, 1e-9])
    planted = np.diag(spectrum)
    assert numerical_effective_rank(planted, tolerance=TOLERANCE) == 3
    expected = float(np.sum(spectrum**2) / spectrum[0] ** 2)
    assert stable_rank(planted) == pytest.approx(expected)


def test_o3_composition_matches_the_truncated_affine_formula_on_the_bank() -> None:
    alpha, rank = 0.625, 8
    merged, protected_parts, partner_parts = svd_trunc_merge(
        BANK_A, BANK_B, alpha=alpha, rank=rank, context=CTX
    )
    for name in merged.names:
        expected = alpha * protected_parts[name].matrix + (1.0 - alpha) * partner_parts[name].matrix
        assert np.array_equal(merged[name], np.asarray(expected, dtype=np.float32))


def test_o3_distorts_the_descendant_relative_to_the_linear_baseline() -> None:
    baseline = linear_merge(BANK_A, BANK_B, alpha=0.5, context=CTX)
    errors = []
    for rank in (4, 8, 16, 24):
        merged, _, _ = svd_trunc_merge(BANK_A, BANK_B, alpha=0.5, rank=rank, context=CTX)
        errors.append(_relative_frobenius_error(merged, baseline))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]


# ------------------------------------------------------------------ operator separation
def test_the_three_operators_produce_three_different_descendants() -> None:
    alpha = 0.5
    linear = linear_merge(BANK_A, BANK_B, alpha=alpha, context=CTX)
    dare, _, _ = dare_merge(
        BANK_A, BANK_B, alpha=alpha, drop_probability=0.5, merge_seed=SEED, context=CTX
    )
    svd, _, _ = svd_trunc_merge(BANK_A, BANK_B, alpha=alpha, rank=8, context=CTX)
    identities = {
        linear.content_identity(),
        dare.content_identity(),
        svd.content_identity(),
    }
    assert len(identities) == 3
