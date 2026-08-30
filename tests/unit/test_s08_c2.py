"""B35 — the C2 spectral shared-source core [AUTH: 00 §25 P0-C2]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from s08_fixtures import (
    FIXTURE_K,
    SURFACE,
    _specs,
    bind,
    linear_family,
    merge_context,
    number,
    protected_and_partners,
    recoverable_bank,
    recovery_context,
)

from src.materials import material_integer, material_text
from src.merge.family import build_release_family
from src.recovery.evaluation import parameter_recovery_metrics
from src.recovery.observations import (
    ObservationError,
    ObservedLineage,
    StructuralNAError,
    observe_incomplete_fixed,
)
from src.recovery.settings import recovery_settings
from src.recovery.solvers import (
    C2_NAIVE_SHARED_MEAN,
    C2_SPECTRAL_FIXED,
    _spectral_detuning_core,
    recover_c2_fixed_alpha,
    recover_c2_naive_shared_mean,
    truncate_to_rank,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = recovery_settings(REPO_ROOT)
MCTX = merge_context()
RCTX = recovery_context()


def fixed_alpha_observation(k: int, alpha: float = 0.5, *, seed: int = 4242) -> ObservedLineage:
    protected, partners = protected_and_partners(k=k, seed=seed)
    family = linear_family(protected, partners, [alpha] * k)
    return observe_incomplete_fixed(family=family, context=RCTX)


# ------------------------------------------------------------------ the frozen adjudication
def test_the_method_identity_and_iteration_count_are_frozen_in_config() -> None:
    assert material_text(SETTINGS, "c2_method") == "SPECTRAL_DETUNING_CORE_FIXED_RANK_REPRODUCTION"
    assert material_integer(SETTINGS, "c2_n_iters") == 1000
    assert material_text(SETTINGS, "c2_rank_scheduler") == "DISABLED_FIXED_STRUCTURAL_RANK"
    assert material_integer(SETTINGS, "linear_residual_rank") == 32
    assert "Spectral DeTuning" in material_text(SETTINGS, "reference_method")


def test_the_method_is_labelled_a_reproduction_not_a_novelty() -> None:
    observation = fixed_alpha_observation(2)
    result = recover_c2_fixed_alpha(observation)
    assert result.method == C2_SPECTRAL_FIXED
    assert result.method_role == "OCCUPIED_REGIME_REPRODUCTION"
    source = Path("src/recovery/solvers.py").read_text(encoding="utf-8")
    assert "not a novel algorithm" in source
    assert "implemented independently" in source


# ------------------------------------------------------------------ Algorithm-1 equations
def test_the_initial_source_is_the_mean_of_the_observations() -> None:
    """S^(0) = mean_i X_i: one iteration of a rank that keeps everything returns it."""
    rng = np.random.default_rng(3)
    stack = [rng.normal(size=(6, 5)).astype(np.float32) for _ in range(3)]
    zero_iters_context = recovery_context(n_iters=1, residual_rank=5)
    mean = np.mean(np.stack(stack), axis=0, dtype=np.float32)
    # with r >= min(shape) the M-step reconstructs the residual exactly, so S^(1) = S^(0)
    result = _spectral_detuning_core(stack, residual_rank=5, n_iters=1, context=zero_iters_context)
    assert np.allclose(result, mean, atol=1e-5)


def test_the_m_step_uses_a_rank_r_truncation() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(10, 8))
    for rank in (1, 2, 4):
        truncated = truncate_to_rank(matrix, rank=rank, context=RCTX)
        assert np.linalg.matrix_rank(truncated.astype(np.float64), tol=1e-5) <= rank
        assert truncated.dtype == np.float32


def test_the_w_step_is_the_mean_of_x_minus_l() -> None:
    """One explicit iteration, recomputed by hand against the implementation."""
    rng = np.random.default_rng(7)
    stack = [rng.normal(size=(8, 6)).astype(np.float32) for _ in range(3)]
    rank = 2
    source = np.mean(np.stack(stack), axis=0, dtype=np.float32)
    residuals = [
        truncate_to_rank(np.asarray(x - source, dtype=np.float32), rank=rank, context=RCTX)
        for x in stack
    ]
    expected = np.mean(
        np.stack(
            [np.asarray(x - low, dtype=np.float32) for x, low in zip(stack, residuals, strict=True)]
        ),
        axis=0,
        dtype=np.float32,
    )
    actual = _spectral_detuning_core(stack, residual_rank=rank, n_iters=1, context=RCTX)
    assert np.allclose(actual, expected, atol=1e-6)


def test_the_iteration_count_comes_from_the_context() -> None:
    observation = fixed_alpha_observation(4)
    for n_iters in (5, 40):
        context = recovery_context(n_iters=n_iters)
        rebuilt = observe_incomplete_fixed(
            family=linear_family(*protected_and_partners(k=4), [0.5] * 4), context=context
        )
        result = recover_c2_fixed_alpha(rebuilt)
        assert result.iterations == n_iters
    assert recover_c2_fixed_alpha(observation).iterations == RCTX.c2_n_iters


def test_the_residual_rank_comes_from_the_context_not_the_caller() -> None:
    import inspect

    assert set(inspect.signature(recover_c2_fixed_alpha).parameters) == {"observation"}
    context = recovery_context(residual_rank=5)
    observation = observe_incomplete_fixed(
        family=linear_family(*protected_and_partners(k=4), [0.5] * 4), context=context
    )
    assert recover_c2_fixed_alpha(observation).residual_rank == 5


# ------------------------------------------------------------------ determinism
def test_deterministic_replay_is_bitwise_identical() -> None:
    observation = fixed_alpha_observation(4)
    first = recover_c2_fixed_alpha(observation)
    second = recover_c2_fixed_alpha(observation)
    assert first.content_identity() == second.content_identity()
    assert first.identity() == second.identity()


def test_descendant_order_does_not_change_the_recovered_source() -> None:
    """R12: canonical order, so a permuted caller list is bitwise the same observation."""
    protected, partners = protected_and_partners(k=4)
    forward = linear_family(protected, partners, [0.5] * 4)
    backward = build_release_family(
        protected=protected,
        partners={f"B{i + 1}": partner for i, partner in enumerate(partners)},
        specs=list(reversed(_specs([0.5] * 4))),
        permitted_k=FIXTURE_K,
        context=MCTX,
    )
    first = observe_incomplete_fixed(family=forward, context=RCTX)
    second = observe_incomplete_fixed(family=backward, context=RCTX)
    assert first.identity() == second.identity()
    a, b = recover_c2_fixed_alpha(first), recover_c2_fixed_alpha(second)
    assert a.content_identity() == b.content_identity()


def test_float32_and_float64_iterations_differ_on_a_sensitive_bank() -> None:
    """B7: the artifact dtype is a scientific choice, so it must be visible in the bytes."""
    rng = np.random.default_rng(11)
    stack = [(rng.normal(size=(9, 7)) * 1e4).astype(np.float32) for _ in range(3)]
    narrow = _spectral_detuning_core(stack, residual_rank=2, n_iters=25, context=RCTX)
    wide_context = recovery_context()
    wide = [np.asarray(x, dtype=np.float64) for x in stack]
    source = np.mean(np.stack(wide), axis=0)
    for _ in range(25):
        residuals = []
        for x in wide:
            left, singular, right = np.linalg.svd(x - source, full_matrices=False)
            residuals.append((left[:, :2] * singular[:2]) @ right[:2, :])
        source = np.mean(
            np.stack([x - low for x, low in zip(wide, residuals, strict=True)]), axis=0
        )
    assert wide_context.arithmetic_dtype == "float32"
    assert not np.array_equal(narrow.astype(np.float64), source)


# ------------------------------------------------------------------ k
@pytest.mark.parametrize("k", [2, 4])
def test_the_registered_descendant_counts_are_accepted(k: int) -> None:
    result = recover_c2_fixed_alpha(fixed_alpha_observation(k))
    assert set(result.names) == set(SURFACE)
    assert all(np.all(np.isfinite(result[name])) for name in result.names)


def test_k_equals_one_is_structurally_not_applicable() -> None:
    """R1: refused at observation time, before any solver can be handed a k = 1 view."""
    protected, partners = protected_and_partners(k=2)
    single = linear_family(protected, partners[:1], [0.5])
    with pytest.raises(StructuralNAError, match="STRUCTURAL_NA"):
        observe_incomplete_fixed(family=single, context=RCTX)


def test_a_zero_returned_instead_of_a_refusal_would_be_a_defect() -> None:
    """B27: k = 1 refuses; it does not return a zero vector and call that recovery."""
    protected, partners = protected_and_partners(k=2)
    with pytest.raises(StructuralNAError):
        observe_incomplete_fixed(family=linear_family(protected, partners[:1], [0.5]), context=RCTX)


def test_an_unregistered_descendant_count_is_refused() -> None:
    """R1: k must be a registered 00 §11 value, so k = 3 cannot silently run."""
    protected, partners = protected_and_partners(k=3)
    with pytest.raises(ObservationError, match="not a registered value"):
        observe_incomplete_fixed(family=linear_family(protected, partners, [0.5] * 3), context=RCTX)


def test_a_mixed_alpha_family_is_refused_by_the_fixed_alpha_factory() -> None:
    protected, partners = protected_and_partners(k=2)
    with pytest.raises(ObservationError, match="one shared alpha"):
        observe_incomplete_fixed(
            family=linear_family(protected, partners, [0.4, 0.6]), context=RCTX
        )


# ------------------------------------------------------------------ output contract
def test_the_recovered_artifact_is_float32_finite_and_on_the_input_surface() -> None:
    observation = fixed_alpha_observation(4)
    result = recover_c2_fixed_alpha(observation)
    assert result.dtype == "float32"
    assert result.surface_identity() != ""
    for name in result.names:
        assert result[name].dtype == np.float32
        assert np.all(np.isfinite(result[name]))
        assert result.shapes[name] == SURFACE[name]


# ------------------------------------------------------------------ beats naive
def test_the_planted_recoverable_fixture_beats_the_naive_shared_mean() -> None:
    """B12: on a correctly specified low-rank residual model, spectral C2 wins clearly."""
    protected, partners = recoverable_bank(k=4)
    family = linear_family(protected, partners, [0.5] * 4)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    binding = bind(observation, family, protected)

    spectral = recover_c2_fixed_alpha(observation)
    naive = recover_c2_naive_shared_mean(observation)
    e_spectral = number(parameter_recovery_metrics(spectral, binding), "e_f")
    e_naive = number(parameter_recovery_metrics(naive, binding), "e_f")
    assert e_spectral < e_naive, (e_spectral, e_naive)
    assert e_spectral < 0.05, e_spectral
    assert naive.method == C2_NAIVE_SHARED_MEAN
    assert naive.method_role == "NAIVE_COMPARATOR"


def test_the_spectral_method_does_not_call_or_label_the_naive_one() -> None:
    source = Path("src/recovery/solvers.py").read_text(encoding="utf-8")
    spectral_body = source.split("def recover_c2_fixed_alpha(")[1].split("\ndef ")[0]
    assert "recover_c2_naive_shared_mean" not in spectral_body
    assert C2_NAIVE_SHARED_MEAN not in spectral_body
