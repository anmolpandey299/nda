"""B36 — C2B known-varying-alpha rescaling [AUTH: 00 §25 P0-C2B]."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest
from s08_fixtures import (
    bind,
    linear_family,
    merge_context,
    number,
    protected_and_partners,
    recoverable_bank,
    recovery_context,
)

from src.materials import material_number
from src.recovery.evaluation import (
    PARTNER_NORM_MATCH_FAILED,
    EvaluationError,
    beta_spread,
    parameter_recovery_metrics,
    partner_norm_guard,
    residual_spread,
)
from src.recovery.observations import (
    NotAuthorizedError,
    ObservedLineage,
    observe_incomplete_fixed,
    observe_incomplete_varying,
)
from src.recovery.settings import coefficient_schedule, recovery_settings
from src.recovery.solvers import (
    C2B_RESCALED,
    NAIVE_UNRESCALED_FIXED_ALPHA,
    RecoveryError,
    _spectral_detuning_core,
    beta_for,
    recover_c2b_varying_alpha,
    recover_naive_unrescaled_fixed_alpha,
    rescale_observations,
)
from src.recovery.truth import EvaluationTruthBinding

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = recovery_settings(REPO_ROOT)
MCTX = merge_context()
RCTX = recovery_context()

K2 = coefficient_schedule(SETTINGS, "c2b_k2_reference_schedule")
K4_MATCHED = coefficient_schedule(SETTINGS, "c2b_k4_spread_matched_schedule")
K4_WIDE = coefficient_schedule(SETTINGS, "c2b_k4_wide_schedule")


def varying_observation(alphas: tuple[float, ...], *, seed: int = 4242) -> ObservedLineage:
    protected, partners = protected_and_partners(k=len(alphas), seed=seed)
    return observe_incomplete_varying(
        family=linear_family(protected, partners, list(alphas)), context=RCTX
    )


def varying_case(
    alphas: tuple[float, ...], *, seed: int = 90210
) -> tuple[ObservedLineage, EvaluationTruthBinding]:
    """A varying-alpha observation on the planted recoverable bank, plus its truth binding."""
    protected, partners = recoverable_bank(k=len(alphas), seed=seed)
    family = linear_family(protected, partners, list(alphas))
    observation = observe_incomplete_varying(family=family, context=RCTX)
    return observation, bind(observation, family, protected)


# ------------------------------------------------------------------ the registered schedules
def test_the_three_registered_schedules_are_the_frozen_ones() -> None:
    assert K2 == (0.35, 0.65)
    assert K4_MATCHED == (0.35, 0.45, 0.55, 0.65)
    assert K4_WIDE == (0.25, 0.4, 0.6, 0.75)


def test_the_beta_formula_is_exact() -> None:
    assert beta_for((0.35, 0.65)) == pytest.approx((1.8571428571428572, 0.5384615384615384))
    assert beta_for((0.5,)) == pytest.approx((1.0,))
    for alphas in (K2, K4_MATCHED, K4_WIDE):
        assert beta_for(alphas) == pytest.approx(tuple((1.0 - a) / a for a in alphas))


@pytest.mark.parametrize(
    ("alphas", "rho", "amplification"),
    [(K2, 3.45, 2.86), (K4_MATCHED, 3.45, 2.86), (K4_WIDE, 9.0, 4.0)],
)
def test_the_analytic_spreads_match_the_spec_values(
    alphas: tuple[float, ...], rho: float, amplification: float
) -> None:
    report = beta_spread(alphas)
    assert report["rho_beta"] == pytest.approx(rho, abs=0.01)
    assert report["a_alpha"] == pytest.approx(amplification, abs=0.01)


def test_the_k4_matched_schedule_matches_the_k2_beta_spread() -> None:
    assert beta_spread(K2)["rho_beta"] == pytest.approx(beta_spread(K4_MATCHED)["rho_beta"])


# ------------------------------------------------------------------ B14 rescaling identity
@pytest.mark.parametrize("alphas", [K2, K4_MATCHED, K4_WIDE])
def test_the_rescaling_identity_holds_to_float32_tolerance(alphas: tuple[float, ...]) -> None:
    """D_i / alpha_i == A + beta_i * B_i [AUTH: 00 §25 P0-C2B]."""
    protected, partners = protected_and_partners(k=len(alphas))
    observation = observe_incomplete_varying(
        family=linear_family(protected, partners, list(alphas)), context=RCTX
    )
    rescaled = rescale_observations(observation)
    betas = beta_for(alphas)
    for name, stack in rescaled.items():
        for index, (beta, partner) in enumerate(zip(betas, partners, strict=True)):
            expected = np.asarray(
                protected[name] + np.float32(beta) * partner[name], dtype=np.float32
            )
            assert np.allclose(stack[index], expected, rtol=1e-4, atol=1e-5), (name, index)


@pytest.mark.parametrize("alphas", [K2, K4_MATCHED, K4_WIDE])
def test_c2b_runs_on_each_registered_schedule(alphas: tuple[float, ...]) -> None:
    observation, binding = varying_case(alphas)
    result = recover_c2b_varying_alpha(observation)
    assert result.method == C2B_RESCALED
    assert result.dtype == "float32"
    assert np.isfinite(number(parameter_recovery_metrics(result, binding), "e_f"))


# ------------------------------------------------------------------ same solver, no optimizer
def test_c2b_uses_the_same_c2_solver_implementation() -> None:
    source = Path("src/recovery/solvers.py").read_text(encoding="utf-8")
    body = source.split("def recover_c2b_varying_alpha(")[1].split("\ndef ")[0]
    assert "_spectral_detuning_core(" in body
    # exactly one solver core exists in the module
    assert source.count("def _spectral_detuning_core(") == 1


def test_no_second_optimizer_was_introduced() -> None:
    """B13: C2B preconditions and reuses C2; it adds no optimiser of its own."""
    import ast

    tree = ast.parse(Path("src/recovery/solvers.py").read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not called & {"minimize", "least_squares", "curve_fit", "fmin", "grad", "backward"}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not {m for m in imported if m.startswith(("scipy.optimize", "torch"))}


def test_the_c2b_result_binds_the_same_solver_version_as_c2() -> None:
    from src.recovery.solvers import recover_c2_fixed_alpha

    protected, partners = protected_and_partners(k=4)
    fixed = recover_c2_fixed_alpha(
        observe_incomplete_fixed(family=linear_family(protected, partners, [0.5] * 4), context=RCTX)
    )
    varying = recover_c2b_varying_alpha(varying_observation(K4_MATCHED))
    assert fixed.as_dict()["c2_method_version"] == varying.as_dict()["c2_method_version"]
    assert varying.iterations == RCTX.c2_n_iters


def test_the_rescaled_core_is_literally_the_shared_function() -> None:
    assert inspect.getsource(_spectral_detuning_core).count("def _spectral_detuning_core") == 1


def test_the_raw_numerical_core_is_module_private() -> None:
    """R2: the public scientific API takes issued observations, never raw matrices."""
    import src.recovery.solvers as solvers

    assert not hasattr(solvers, "spectral_detuning_core")
    public = [name for name in dir(solvers) if name.startswith("recover_")]
    for name in public:
        parameters = set(inspect.signature(getattr(solvers, name)).parameters)
        assert "observations" not in parameters and "matrices" not in parameters, name


# ------------------------------------------------------------------ fail closed
def test_a_zero_coefficient_cannot_be_rescaled() -> None:
    with pytest.raises(RecoveryError, match="zero coefficient"):
        beta_for((0.5, 0.0))


def test_a_zero_coefficient_is_refused_at_observation_time() -> None:
    from src.recovery.observations import ObservationError

    protected, partners = protected_and_partners(k=2)
    with pytest.raises(ObservationError, match="zero coefficient"):
        observe_incomplete_varying(
            family=linear_family(protected, partners, [0.5, 0.0]), context=RCTX
        )


def test_k_equals_one_is_structurally_not_applicable_for_c2b() -> None:
    from src.recovery.observations import StructuralNAError

    protected, partners = protected_and_partners(k=2)
    with pytest.raises(StructuralNAError, match="STRUCTURAL_NA"):
        observe_incomplete_varying(
            family=linear_family(protected, partners[:1], [0.35]), context=RCTX
        )


def test_c2b_refuses_a_fixed_alpha_observation() -> None:
    protected, partners = protected_and_partners(k=4)
    observation = observe_incomplete_fixed(
        family=linear_family(protected, partners, [0.5] * 4), context=RCTX
    )
    with pytest.raises(NotAuthorizedError):
        recover_c2b_varying_alpha(observation)


def test_no_hidden_alpha_estimation_exists() -> None:
    source = Path("src/recovery/solvers.py").read_text(encoding="utf-8")
    for banned in ("estimate_alpha", "infer_alpha", "fit_alpha", "solve_alpha"):
        assert f"def {banned}" not in source


# ------------------------------------------------------------------ B16 naive unrescaled
def test_the_naive_unrescaled_comparator_has_no_default_assumed_alpha() -> None:
    signature = inspect.signature(recover_naive_unrescaled_fixed_alpha)
    parameter = signature.parameters["assumed_fixed_alpha"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_naive_unrescaled_comparator_shares_the_observation_identity() -> None:
    """B23: compared methods consume the same attacker-visible observation R."""
    observation = varying_observation(K4_WIDE)
    rescaled = recover_c2b_varying_alpha(observation)
    naive = recover_naive_unrescaled_fixed_alpha(observation, assumed_fixed_alpha=0.5)
    assert rescaled.observation_identity == naive.observation_identity
    assert naive.method == NAIVE_UNRESCALED_FIXED_ALPHA
    assert naive.method_role == "NAIVE_COMPARATOR"


def test_the_naive_unrescaled_comparator_does_not_rescale_per_descendant() -> None:
    observation = varying_observation(K4_WIDE)
    naive = recover_naive_unrescaled_fixed_alpha(observation, assumed_fixed_alpha=0.5)
    expected = {
        name: np.asarray(
            np.mean(
                np.stack([u[name] for u in observation.updates], dtype=np.float32),
                axis=0,
                dtype=np.float32,
            )
            / np.float32(0.5),
            dtype=np.float32,
        )
        for name in naive.names
    }
    for name in naive.names:
        assert np.allclose(naive[name], expected[name], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("alpha", [0.0, float("nan"), float("inf")])
def test_an_invalid_predeclared_comparator_alpha_is_refused(alpha: float) -> None:
    observation = varying_observation(K2)
    with pytest.raises(RecoveryError, match="predeclared comparator alpha"):
        recover_naive_unrescaled_fixed_alpha(observation, assumed_fixed_alpha=alpha)


def test_the_config_refuses_to_supply_a_default_comparator_alpha() -> None:
    from src.materials import UncalibratedConstantError, material

    with pytest.raises(UncalibratedConstantError, match="REQUIRED_NOT_CALIBRATED"):
        material(SETTINGS, "naive_unrescaled_assumed_alpha")


# ------------------------------------------------------------------ B15 evaluator-only
def test_rho_resid_needs_true_partners_and_is_evaluator_only() -> None:
    protected, partners = protected_and_partners(k=4)
    del protected
    report = residual_spread(K4_WIDE, partners)
    betas = beta_for(K4_WIDE)
    norms = [abs(b) * p.frobenius_norm() for b, p in zip(betas, partners, strict=True)]
    assert report["rho_resid"] == pytest.approx(max(norms) / min(norms))

    import ast

    tree = ast.parse(Path("src/recovery/solvers.py").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "src.recovery.evaluation" not in imported, sorted(imported)
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not defined & {"residual_spread", "partner_norm_guard"}


def test_the_partner_norm_guard_uses_the_frozen_limit() -> None:
    limit = material_number(SETTINGS, "partner_norm_match_limit")
    assert limit == 1.25
    _, partners = protected_and_partners(k=4)
    assert RCTX.partner_norm_match_limit == limit
    report = partner_norm_guard(partners, context=RCTX)
    norms = [p.frobenius_norm() for p in partners]
    assert report["norm_ratio"] == pytest.approx(max(norms) / min(norms))
    assert report["status"] in {"PARTNER_NORM_MATCHED", PARTNER_NORM_MATCH_FAILED}


def test_the_partner_norm_guard_reports_failure_explicitly() -> None:
    from s08_fixtures import SURFACE

    from src.merge.updates import fixture_induced_update

    small = fixture_induced_update(
        {name: np.ones(shape) * 0.01 for name, shape in SURFACE.items()}, context=MCTX
    )
    large = fixture_induced_update(
        {name: np.ones(shape) * 10.0 for name, shape in SURFACE.items()}, context=MCTX
    )
    report = partner_norm_guard([small, large], context=RCTX)
    assert report["status"] == PARTNER_NORM_MATCH_FAILED
    assert isinstance(report["norm_ratio"], float) and report["norm_ratio"] > 1.25


def test_a_zero_norm_partner_is_an_explicit_undefined_state() -> None:
    from s08_fixtures import SURFACE

    from src.merge.updates import fixture_induced_update

    zero = fixture_induced_update(
        {name: np.zeros(shape) for name, shape in SURFACE.items()}, context=MCTX
    )
    _, partners = protected_and_partners(k=2)
    with pytest.raises(EvaluationError, match="UNDEFINED_ZERO_DENOMINATOR"):
        partner_norm_guard([zero, partners[0]], context=RCTX)


def test_the_recovery_result_is_not_a_dataclass() -> None:
    result = recover_c2b_varying_alpha(varying_observation(K2))
    assert not dataclasses.is_dataclass(result)
    with pytest.raises(TypeError):
        dataclasses.replace(result)  # type: ignore[type-var]
