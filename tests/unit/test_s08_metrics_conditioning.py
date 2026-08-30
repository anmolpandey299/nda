"""B39/B40 — parameter-recovery metrics and direction-matrix conditioning [00 §22, §26]."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s08_fixtures import (
    SURFACE,
    bind,
    layer_map,
    linear_family,
    merge_context,
    number,
    protected_and_partners,
    recovery_context,
    sequence,
)

from src.materials import material_number
from src.merge.updates import fixture_induced_update
from src.recovery.conditioning import (
    RANK_DEFICIENT,
    ConditioningError,
    conditioning_from_observations,
    conditioning_report,
    direction_matrix,
)
from src.recovery.evaluation import UNDEFINED, EvaluationError, parameter_recovery_metrics
from src.recovery.observations import observe_incomplete_fixed, observe_known_partner
from src.recovery.settings import recovery_settings
from src.recovery.solvers import recover_c1, recover_c2_fixed_alpha

REPO_ROOT = Path(__file__).resolve().parents[2]


def loose(truth: object) -> Any:
    """Route a loose task vector through `Any`: mypy rejects it, and so must the runtime."""
    return truth


MCTX = merge_context()
RCTX = recovery_context()
TOLERANCE = material_number(recovery_settings(REPO_ROOT), "conditioning_numerical_rank_tolerance")


# ======================================================================== metrics
def test_an_exact_recovery_has_e_f_zero() -> None:
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [1.0]).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=RCTX
    )
    binding = bind(observation, descendant, protected)
    metrics = parameter_recovery_metrics(recover_c1(observation), binding)
    assert metrics["e_f"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)


def test_a_zero_recovery_anchors_e_f_at_one_for_a_nonzero_target() -> None:
    """The zero-delta anchor: recovering nothing is exactly one unit of relative error."""
    from src.recovery.evaluation import relative_frobenius

    protected, _ = protected_and_partners(k=2)
    wide = np.dtype(np.float64)
    zeros = {name: np.zeros(shape) for name, shape in SURFACE.items()}
    target = {name: protected[name] for name in SURFACE}
    assert relative_frobenius(zeros, target, wide=wide) == pytest.approx(1.0)


def test_a_known_scaled_error_is_reported_exactly() -> None:
    from src.recovery.evaluation import relative_frobenius

    wide = np.dtype(np.float64)
    target = {"w": np.array([[3.0, 4.0]])}
    candidate = {"w": np.array([[3.0, 4.0]]) * 1.1}
    assert relative_frobenius(candidate, target, wide=wide) == pytest.approx(0.1)


def test_the_per_layer_cosine_spectral_and_concentration_metrics_are_reported() -> None:
    protected, partners = protected_and_partners(k=4)
    family = linear_family(protected, partners, [0.5] * 4)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    metrics = parameter_recovery_metrics(
        recover_c2_fixed_alpha(observation), bind(observation, family, protected)
    )
    assert set(layer_map(metrics, "per_layer_relative_error")) == set(SURFACE)
    assert isinstance(metrics["cosine_similarity"], float)
    assert set(layer_map(metrics, "spectral_error")) == set(SURFACE)
    assert isinstance(metrics["max_layer_error"], float)
    concentration = metrics["error_concentration"]
    assert isinstance(concentration, float)
    assert 0.5 <= concentration <= 1.0
    assert number(metrics, "max_layer_error") == pytest.approx(
        max(layer_map(metrics, "per_layer_relative_error").values())
    )


def test_the_spectral_error_is_the_singular_value_difference() -> None:
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [1.0]).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=RCTX
    )
    result = recover_c1(observation)
    metrics = parameter_recovery_metrics(result, bind(observation, descendant, protected))
    for name in result.names:
        left = np.linalg.svd(np.asarray(result[name], dtype=np.float64), compute_uv=False)
        right = np.linalg.svd(np.asarray(protected[name], dtype=np.float64), compute_uv=False)
        assert layer_map(metrics, "spectral_error")[name] == pytest.approx(
            float(np.linalg.norm(left - right)), abs=1e-4
        )


def test_a_zero_target_is_an_explicit_undefined_state() -> None:
    _, partners = protected_and_partners(k=2)
    zero = fixture_induced_update(
        {name: np.zeros(shape) for name, shape in SURFACE.items()}, context=MCTX, origin="zero"
    )
    family = linear_family(zero, partners, [0.5, 0.5])
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    with pytest.raises(EvaluationError, match=UNDEFINED):
        parameter_recovery_metrics(
            recover_c2_fixed_alpha(observation), bind(observation, family, zero)
        )


def test_a_truth_the_lineage_was_not_built_from_is_refused() -> None:
    """R3: an A_1 recovery has no API through which to be scored against an unrelated A_2."""
    from src.recovery.truth import TruthBindingError

    protected, partners = protected_and_partners(k=2)
    other, _ = protected_and_partners(k=2, seed=999)
    family = linear_family(protected, partners, [0.5, 0.5])
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    with pytest.raises(TruthBindingError, match="not the one this lineage"):
        bind(observation, family, other)
    with pytest.raises(EvaluationError, match="EvaluationTruthBinding"):
        parameter_recovery_metrics(recover_c2_fixed_alpha(observation), loose(other))


def test_evaluation_does_not_mutate_the_recovery_result() -> None:
    protected, partners = protected_and_partners(k=4)
    family = linear_family(protected, partners, [0.5] * 4)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    binding = bind(observation, family, protected)
    result = recover_c2_fixed_alpha(observation)
    before = (result.identity(), result.content_identity())
    parameter_recovery_metrics(result, binding)
    parameter_recovery_metrics(result, binding)
    assert (result.identity(), result.content_identity()) == before


def test_e_f_is_never_clamped() -> None:
    from src.recovery.evaluation import relative_frobenius

    wide = np.dtype(np.float64)
    target = {"w": np.array([[1.0]])}
    candidate = {"w": np.array([[100.0]])}
    assert relative_frobenius(candidate, target, wide=wide) == pytest.approx(99.0)


# ======================================================================== conditioning
def test_the_numerical_rank_tolerance_reuses_the_frozen_criterion() -> None:
    assert TOLERANCE == 1e-06
    assert RCTX.conditioning_rank_tolerance == 1e-06


def test_a_well_conditioned_k2_family() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    report = conditioning_report(matrix, context=RCTX)
    assert report["k"] == 2
    assert report["effective_numerical_rank"] == 2
    assert report["kappa"] == pytest.approx(1.0)


def test_a_nearly_collinear_k2_family_has_a_large_kappa() -> None:
    matrix = np.array([[1.0, 1.0], [0.0, 1e-3], [0.0, 0.0]])
    report = conditioning_report(matrix, context=RCTX)
    assert report["effective_numerical_rank"] == 2
    assert isinstance(report["kappa"], float) and report["kappa"] > 100.0


def test_a_rank_one_direction_family_is_rank_deficient() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    report = conditioning_report(matrix, context=RCTX)
    assert report["effective_numerical_rank"] == 1
    assert report["kappa"] == RANK_DEFICIENT


def test_a_k4_family_with_a_known_spectrum() -> None:
    spectrum = [8.0, 4.0, 2.0, 1.0]
    left = np.linalg.qr(np.random.default_rng(3).normal(size=(10, 4)))[0]
    right = np.linalg.qr(np.random.default_rng(4).normal(size=(4, 4)))[0]
    matrix = (left * spectrum) @ right.T
    report = conditioning_report(matrix, context=RCTX)
    assert report["k"] == 4
    assert np.allclose(sorted(sequence(report, "singular_values"), reverse=True), spectrum)
    assert report["effective_numerical_rank"] == 4
    assert report["kappa"] == pytest.approx(8.0)


def test_a_zero_direction_matrix_reports_rank_zero_and_no_kappa() -> None:
    report = conditioning_report(np.zeros((6, 2)), context=RCTX)
    assert report["effective_numerical_rank"] == 0
    assert report["kappa"] == RANK_DEFICIENT
    assert not any(isinstance(v, float) and not np.isfinite(v) for v in report.values())


@pytest.mark.parametrize("ratio", [1e-5, 1e-7])
def test_the_numerical_rank_threshold_boundary(ratio: float) -> None:
    matrix = np.diag([1.0, ratio, 0.0])
    report = conditioning_report(matrix, context=RCTX)
    expected = 2 if ratio >= TOLERANCE else 1
    assert report["effective_numerical_rank"] == expected


def test_column_order_does_not_change_the_spectrum() -> None:
    matrix = np.array([[1.0, 0.0, 2.0, 0.5], [0.0, 3.0, 1.0, 0.0], [1.0, 1.0, 0.0, 2.0]])
    first = conditioning_report(matrix, context=RCTX)
    second = conditioning_report(matrix[:, ::-1], context=RCTX)
    assert np.allclose(sequence(first, "singular_values"), sequence(second, "singular_values"))
    assert number(first, "kappa") == pytest.approx(number(second, "kappa"))


def test_no_ordinary_conditioning_result_is_inf_or_nan() -> None:
    for matrix in (np.zeros((4, 2)), np.eye(4)[:, :2], np.ones((4, 2))):
        report = conditioning_report(matrix, context=RCTX)
        kappa = report["kappa"]
        assert kappa == RANK_DEFICIENT or (isinstance(kappa, float) and np.isfinite(kappa))


def test_conditioning_from_known_partner_observations() -> None:
    """00 §26: D_i = vec(C_i - B_i), available only where B_i is known."""
    protected, partners = protected_and_partners(k=2)
    family = linear_family(protected, partners, [0.5, 0.5])
    observations = [
        observe_known_partner(descendant=d, partner_update=p, context=RCTX)
        for d, p in zip(family.descendants, partners, strict=True)
    ]
    matrix = direction_matrix(observations)
    assert matrix.shape == (sum(r * c for r, c in SURFACE.values()), 2)
    report = conditioning_from_observations(observations, context=RCTX)
    assert report["k"] == 2
    assert report["effective_numerical_rank"] in {1, 2}


def test_an_empty_direction_family_is_refused() -> None:
    with pytest.raises(ConditioningError, match="at least one descendant"):
        direction_matrix([])
