"""B38 — the fourteen information-firewall attacks [AUTH: 00 §12, §25; 01 §29]."""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s08_fixtures import (
    FIXTURE_K,
    linear_family,
    merge_context,
    o3_family,
    protected_and_partners,
    recovery_context,
    rewritten_recovery_context,
)

from src.recovery.context import RecoveryContextError, RecoveryExecutionContext
from src.recovery.observations import (
    Z_HIGH,
    Z_INCOMPLETE_FIXED,
    Z_INCOMPLETE_VARYING_KNOWN,
    NotAuthorizedError,
    ObservationError,
    ObservedLineage,
    ObservedUpdate,
    observe_hidden_alpha,
    observe_incomplete_fixed,
    observe_incomplete_varying,
    observe_known_partner,
)
from src.recovery.solvers import (
    RecoveryResult,
    recover_c1,
    recover_c2_fixed_alpha,
    recover_c2_naive_shared_mean,
    recover_dare_incomplete,
    recover_hidden_alpha,
    recover_ties,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MCTX = merge_context()
RCTX = recovery_context()


def attempt_replace(obj: object, **changes: object) -> object:
    """Run `dataclasses.replace` through `Any` so the runtime attack actually executes."""
    replace: Any = dataclasses.replace
    return replace(obj, **changes)


def attempt_copy_replace(obj: object) -> object:
    """Same, for `copy.replace`, which 3.13 added and which also refuses non-dataclasses."""
    replace: Any = copy.replace
    return replace(obj)


def incomplete(k: int = 4, alpha: float = 0.5) -> tuple[Any, list[Any], ObservedLineage]:
    protected, partners = protected_and_partners(k=k)
    family = linear_family(protected, partners, [alpha] * k)
    return protected, partners, observe_incomplete_fixed(family=family, context=RCTX)


# ============================================================ 1-2 injection
def test_01_hidden_partner_bytes_cannot_be_injected_into_a_c2_observation() -> None:
    _, partners, observation = incomplete()
    assert observation.has_partner is False
    with pytest.raises(NotAuthorizedError, match="not authorised"):
        _ = observation.partner
    with pytest.raises(AttributeError):
        observation.__setattr__("_partner", partners[0])
    with pytest.raises(TypeError):
        attempt_replace(observation, _partner=partners[0])
    # no slot on the incomplete observation holds a partner identity
    stored = {slot: getattr(observation, slot, None) for slot in ObservedLineage.__slots__}
    assert stored["_partner"] is None


def test_02_the_true_protected_constituent_cannot_be_injected() -> None:
    protected, _, observation = incomplete()
    document = observation.as_dict()
    assert protected.content_identity() not in str(document)
    assert not {key for key in document if "protected" in key.lower()}
    with pytest.raises(AttributeError):
        observation.__setattr__("_protected", protected)


def test_the_observation_carries_no_hidden_identity_at_all() -> None:
    protected, partners, observation = incomplete()
    rendered = str(observation.as_dict())
    for hidden in [protected.content_identity(), *(p.content_identity() for p in partners)]:
        assert hidden not in rendered


# ============================================================ 3-4 regime and alpha
def test_03_the_lineage_regime_cannot_be_forged_to_z_high() -> None:
    _, _, observation = incomplete()
    assert observation.regime == Z_INCOMPLETE_FIXED
    with pytest.raises(AttributeError):
        observation.__setattr__("_regime", Z_HIGH)
    with pytest.raises(TypeError):
        attempt_replace(observation, _regime=Z_HIGH)
    with pytest.raises(TypeError, match="factory-issued"):
        ObservedLineage()
    # and a forged regime string could not gain a partner anyway
    with pytest.raises(NotAuthorizedError):
        recover_c1(observation)


def test_04_the_public_alpha_cannot_be_changed_after_issuance() -> None:
    _, _, observation = incomplete(alpha=0.5)
    assert observation.alphas == (0.5,) * 4
    recorded = observation.identity()
    with pytest.raises(AttributeError):
        observation.__setattr__("_alphas", (0.9,) * 4)
    with pytest.raises(TypeError):
        attempt_replace(observation, _alphas=(0.9,) * 4)
    assert observation.identity() == recorded


# ============================================================ 5-8 result integrity
def test_05_the_result_method_identity_cannot_be_replaced() -> None:
    _, _, observation = incomplete()
    result = recover_c2_fixed_alpha(observation)
    with pytest.raises(AttributeError):
        result.__setattr__("_method", "C1_EXACT_LINEAR_ORACLE")
    with pytest.raises(TypeError):
        attempt_replace(result, _method="C1_EXACT_LINEAR_ORACLE")
    with pytest.raises(TypeError, match="factory-issued"):
        RecoveryResult()
    assert not dataclasses.is_dataclass(result)
    with pytest.raises(TypeError):
        attempt_copy_replace(result)


def test_06_the_result_observation_identity_cannot_be_replaced() -> None:
    _, _, observation = incomplete()
    result = recover_c2_fixed_alpha(observation)
    assert result.observation_identity == observation.identity()
    with pytest.raises(AttributeError):
        result.__setattr__("_observation_identity", "0" * 64)
    with pytest.raises(TypeError):
        attempt_replace(result, _observation_identity="0" * 64)


def test_07_recovered_bytes_cannot_be_mutated() -> None:
    _, _, observation = incomplete()
    result = recover_c2_fixed_alpha(observation)
    name = result.names[0]
    recorded = result.content_identity()
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        result[name][0, 0] = 42.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        np.asarray(result[name]).setflags(write=True)
    scratch = result.copy_of(name)
    scratch[0, 0] = 42.0
    assert scratch.flags.owndata
    assert result.content_identity() == recorded
    assert result.identity() != ""


def test_08_no_ground_truth_hash_can_be_attached_to_a_result() -> None:
    protected, partners, observation = incomplete()
    result = recover_c2_fixed_alpha(observation)
    document = result.as_dict()
    rendered = str(document)
    assert protected.content_identity() not in rendered
    for partner in partners:
        assert partner.content_identity() not in rendered
    assert not {k for k in document if "true" in k.lower() or "ground" in k.lower()}
    assert "e_f" not in document and "success" not in document
    with pytest.raises(AttributeError):
        result.__setattr__("_truth", protected.content_identity())


def test_the_solver_module_never_imports_the_evaluator() -> None:
    """B3: no hyperparameter or stopping rule can depend on a truth-based metric."""
    tree = ast.parse(Path("src/recovery/solvers.py").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "src.recovery.evaluation" not in imported
    assert "src.recovery.conditioning" not in imported
    assert "src.recovery.truth" not in imported


def test_the_c2_core_has_no_stopping_rule() -> None:
    """B6: a fixed iteration count, so no convergence test can read an outcome."""
    from src.recovery.solvers import _spectral_detuning_core

    source = inspect.getsource(_spectral_detuning_core)
    assert "for _ in range(n_iters)" in source
    assert "break" not in source
    assert "tol" not in source and "converge" not in source.replace("convergence stopping", "")


# ============================================================ 9 same observation R
def test_09_compared_methods_must_share_one_observation_object() -> None:
    protected, partners, first = incomplete()
    spectral = recover_c2_fixed_alpha(first)
    naive = recover_c2_naive_shared_mean(first)
    assert spectral.observation_identity == naive.observation_identity == first.identity()

    # an independently reconstructed view over different partners is a different R
    other_protected, other_partners = protected_and_partners(k=4, seed=999)
    second = observe_incomplete_fixed(
        family=linear_family(other_protected, other_partners, [0.5] * 4), context=RCTX
    )
    assert second.identity() != first.identity()
    assert (
        recover_c2_naive_shared_mean(second).observation_identity != spectral.observation_identity
    )
    assert protected is not None and partners is not None


# ============================================================ 10-11 C1 and conditioning
def test_10_c1_with_the_wrong_partner_is_refused() -> None:
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    with pytest.raises(ObservationError, match="not the partner this descendant"):
        observe_known_partner(descendant=descendant, partner_update=partners[1], context=RCTX)


def test_11_conditioning_is_unreachable_from_an_incomplete_lineage() -> None:
    from src.recovery.conditioning import conditioning_from_observations, direction_matrix

    _, _, observation = incomplete()
    with pytest.raises(NotAuthorizedError, match="needs the partner"):
        direction_matrix([observation])
    with pytest.raises(NotAuthorizedError):
        conditioning_from_observations([observation], context=RCTX)

    tree = ast.parse(Path("src/recovery/solvers.py").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "src.recovery.conditioning" not in imported


# ============================================================ 12-14 refused methods
def test_12_the_hidden_alpha_solver_is_structurally_not_authorized() -> None:
    with pytest.raises(NotAuthorizedError, match="STRUCTURAL_NOT_AUTHORIZED"):
        recover_hidden_alpha()
    with pytest.raises(NotAuthorizedError, match="STRUCTURAL_NOT_AUTHORIZED"):
        observe_hidden_alpha()


def test_13_dare_incomplete_recovery_is_method_gated() -> None:
    with pytest.raises(NotAuthorizedError, match="METHOD_GATED"):
        recover_dare_incomplete()
    protected, partners = protected_and_partners(k=2)
    from src.merge.family import MergeSpec, build_release_family

    dare = build_release_family(
        protected=protected,
        partners={f"B{index + 1}": partner for index, partner in enumerate(partners)},
        specs=[
            MergeSpec(
                descendant_id=f"D{index}",
                operator="O2_DARE",
                alpha=0.5,
                partner_id=f"B{index + 1}",
                drop_probability=0.5,
                dare_merge_seed=11,
            )
            for index in range(2)
        ],
        permitted_k=FIXTURE_K,
        context=MCTX,
    )
    with pytest.raises(NotAuthorizedError, match="METHOD_GATED"):
        observe_incomplete_fixed(family=dare, context=RCTX)


def test_14_ties_recovery_is_not_implemented() -> None:
    with pytest.raises(NotAuthorizedError, match="deferred"):
        recover_ties()
    for path in sorted(Path("src/recovery").glob("*.py")):
        source = path.read_text(encoding="utf-8").lower()
        assert "def recover_ties_" not in source
        assert "trim_elect" not in source


# ============================================================ context fails closed
def test_the_recovery_context_is_factory_issued_and_opaque() -> None:
    assert not dataclasses.is_dataclass(RCTX)
    with pytest.raises(TypeError, match="factory-issued"):
        RecoveryExecutionContext()
    with pytest.raises(AttributeError):
        RCTX.__setattr__("_config_sha256", "0" * 64)
    with pytest.raises(TypeError):
        attempt_replace(RCTX)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("arithmetic_dtype", "float64", "float32"),
        ("recovery_version", "s08.recovery.v99", "not executable"),
        ("c2_method", "SOMETHING_ELSE", "C2 method"),
        ("c2_method_version", "s08.c2-core.v99", "C2 method version"),
        ("svd_backend", "scipy.svd", "SVD backend"),
        ("c2_rank_scheduler", "ADAPTIVE", "rank scheduler"),
    ],
)
def test_an_unsupported_recovery_configuration_fails_closed(
    key: str, value: str, match: str
) -> None:
    with pytest.raises(RecoveryContextError, match=match):
        rewritten_recovery_context(**{key: value})


def test_the_context_hash_covers_the_config_actually_consumed() -> None:
    first = recovery_context()
    same = recovery_context()
    other = recovery_context(n_iters=999)
    assert first.identity() == same.identity()
    assert other.identity() != first.identity()
    assert other.config_sha256 != first.config_sha256


def test_an_observed_update_is_immutable_and_factory_issued() -> None:
    _, _, observation = incomplete()
    update = observation.updates[0]
    with pytest.raises(TypeError, match="factory-issued"):
        ObservedUpdate()
    name = update.names[0]
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        update[name][0, 0] = 5.0
    with pytest.raises(AttributeError):
        update.__setattr__("_entries", ())


def test_the_o3_and_varying_regimes_are_derived_from_their_factories() -> None:
    protected, partners = protected_and_partners(k=2, protected_rank=6)
    fixed = observe_incomplete_fixed(
        family=o3_family(protected, partners, [0.5, 0.5], retained_rank=2), context=RCTX
    )
    varying = observe_incomplete_varying(
        family=linear_family(protected, partners, [0.35, 0.65]), context=RCTX
    )
    assert fixed.regime == Z_INCOMPLETE_FIXED
    assert varying.regime == Z_INCOMPLETE_VARYING_KNOWN
    assert not fixed.has_partner and not varying.has_partner
