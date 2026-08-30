"""B34 — C1 exact linear inversion [AUTH: 00 §25 P0-C1]."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s08_fixtures import (
    SURFACE,
    bind,
    linear_family,
    merge_context,
    number,
    o3_family,
    protected_and_partners,
    recovery_context,
)

from src.materials import material_number
from src.merge.family import MergeSpec, build_dare_descendant
from src.merge.updates import fixture_induced_update
from src.recovery.evaluation import c1_pass, parameter_recovery_metrics
from src.recovery.observations import (
    NotAuthorizedError,
    ObservationError,
    observe_incomplete_fixed,
    observe_known_partner,
)
from src.recovery.settings import recovery_settings
from src.recovery.solvers import (
    C1_METHOD,
    RecoveryError,
    RecoveryResult,
    recover_c1,
    recover_c2_fixed_alpha,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MCTX = merge_context()
RCTX = recovery_context()
LIMIT = material_number(recovery_settings(REPO_ROOT), "c1_pass_criterion_ef")


def known_partner_case(alpha: float = 0.5, *, k: int = 1) -> tuple[Any, Any, Any, Any]:
    """A genuine issued S07 descendant, its observation, and the evaluator truth binding."""
    protected, partners = protected_and_partners(k=max(k, 2))
    descendant = linear_family(protected, partners[:1], [alpha]).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=RCTX
    )
    return protected, descendant, observation, bind(observation, descendant, protected)


# ------------------------------------------------------------------ exactness
@pytest.mark.parametrize("alpha", [0.25, 0.35, 0.5, 0.65, 0.9, 1.0])
def test_c1_recovers_the_protected_constituent_exactly(alpha: float) -> None:
    _, _, observation, binding = known_partner_case(alpha)
    result = recover_c1(observation)
    assert number(parameter_recovery_metrics(result, binding), "e_f") <= LIMIT, alpha


def test_the_frozen_c1_pass_criterion_is_one_e_minus_five() -> None:
    assert LIMIT == 1e-05
    _, _, observation, binding = known_partner_case()
    result = recover_c1(observation)
    assert result.context.c1_pass_criterion_ef == LIMIT
    assert c1_pass(result, binding)


def test_c1_recovers_every_matrix_on_the_surface() -> None:
    protected, _, observation, _ = known_partner_case()
    result = recover_c1(observation)
    assert set(result.names) == set(SURFACE)
    for name in result.names:
        assert np.allclose(result[name], protected[name], rtol=1e-5, atol=1e-6)


def test_the_recovered_artifact_is_float32() -> None:
    _, _, observation, _ = known_partner_case()
    result = recover_c1(observation)
    assert result.dtype == "float32"
    for name in result.names:
        assert result[name].dtype == np.float32


# ------------------------------------------------------------------ fail closed
def test_a_zero_coefficient_is_refused() -> None:
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [0.0]).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=RCTX
    )
    with pytest.raises(RecoveryError, match="alpha = 0"):
        recover_c1(observation)


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_coefficient_is_refused_before_a_descendant_exists(alpha: float) -> None:
    protected, partners = protected_and_partners(k=2)
    with pytest.raises(Exception, match="not finite"):
        linear_family(protected, partners[:1], [alpha])


def test_the_wrong_partner_is_refused_at_observation_time() -> None:
    """B9: a descendant built from B1 cannot be paired with B2."""
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    with pytest.raises(ObservationError, match="not the partner this descendant"):
        observe_known_partner(descendant=descendant, partner_update=partners[1], context=RCTX)


def test_a_dare_descendant_cannot_be_inverted_by_c1() -> None:
    protected, partners = protected_and_partners(k=2)
    dare = build_dare_descendant(
        protected=protected,
        partner=partners[0],
        spec=MergeSpec(
            descendant_id="C1",
            operator="O2_DARE",
            alpha=0.5,
            partner_id="B1",
            drop_probability=0.5,
            dare_merge_seed=7,
        ),
        context=MCTX,
    )
    observation = observe_known_partner(descendant=dare, partner_update=partners[0], context=RCTX)
    with pytest.raises(RecoveryError, match="inverts the linear merge only"):
        recover_c1(observation)


def test_an_o3_descendant_cannot_be_inverted_by_c1() -> None:
    protected, partners = protected_and_partners(k=2)
    descendant = o3_family(protected, partners[:1], [0.5], retained_rank=2).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=RCTX
    )
    with pytest.raises(RecoveryError, match="inverts the linear merge only"):
        recover_c1(observation)


def test_c1_refuses_an_incomplete_lineage_observation() -> None:
    protected, partners = protected_and_partners(k=2)
    family = linear_family(protected, partners, [0.5, 0.5])
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    with pytest.raises(NotAuthorizedError, match="known-partner"):
        recover_c1(observation)


def test_a_mismatched_surface_is_refused() -> None:
    protected, partners = protected_and_partners(k=2)
    narrow = fixture_induced_update({"w0": partners[0]["w0"]}, context=MCTX, origin="narrow")
    descendant = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    with pytest.raises(ObservationError):
        observe_known_partner(descendant=descendant, partner_update=narrow, context=RCTX)


def test_a_context_mismatch_moves_the_observation_identity() -> None:
    """A result carries the context it ran under; a different context is a different run."""
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    first = observe_known_partner(descendant=descendant, partner_update=partners[0], context=RCTX)
    other = observe_known_partner(
        descendant=descendant,
        partner_update=partners[0],
        context=recovery_context(n_iters=17, residual_rank=2),
    )
    assert first.identity() != other.identity()
    assert recover_c1(first).identity() != recover_c1(other).identity()


# ------------------------------------------------------------------ B9 forgery boundary
def test_the_c1_method_identity_cannot_be_forged() -> None:
    _, _, observation, _ = known_partner_case()
    result = recover_c1(observation)
    assert result.method == C1_METHOD
    assert result.method_role == "ORACLE_CONTROL"
    assert not dataclasses.is_dataclass(result)
    with pytest.raises(TypeError):
        RecoveryResult()
    with pytest.raises(TypeError):
        dataclasses.replace(result)  # type: ignore[type-var]
    with pytest.raises(AttributeError):
        result.__setattr__("_method", "C2_SPECTRAL_FIXED")


def test_another_recovered_tensor_cannot_be_relabelled_c1() -> None:
    protected, partners = protected_and_partners(k=2)
    family = linear_family(protected, partners, [0.5, 0.5])
    spectral = recover_c2_fixed_alpha(observe_incomplete_fixed(family=family, context=RCTX))
    assert spectral.method != C1_METHOD
    with pytest.raises(AttributeError):
        spectral.__setattr__("_method", C1_METHOD)


def test_the_recovered_bytes_are_immutable() -> None:
    _, _, observation, _ = known_partner_case()
    result = recover_c1(observation)
    name = result.names[0]
    recorded = result.content_identity()
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        result[name][0, 0] = 99.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        np.asarray(result[name]).setflags(write=True)
    scratch = result.copy_of(name)
    scratch[0, 0] = 99.0
    assert result.content_identity() == recorded
