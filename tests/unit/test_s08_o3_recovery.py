"""B37 — O3 conditional recovery and its four separate errors [AUTH: 00 §24A, §25]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from s08_fixtures import (
    bind,
    merge_context,
    o3_family,
    protected_and_partners,
    recovery_context,
)

from src.materials import material_text
from src.merge.family import MergeSpec, ReleaseFamily
from src.merge.updates import FixtureTaskVector
from src.recovery.evaluation import UNDEFINED, EvaluationError, o3_error_report
from src.recovery.observations import (
    ObservationError,
    ObservedLineage,
    observe_incomplete_fixed,
    observe_incomplete_varying,
)
from src.recovery.settings import recovery_settings
from src.recovery.solvers import (
    O3_C2B_RESCALED,
    O3_SPECTRAL_FIXED,
    ORIGINAL_PROTECTED_CONSTITUENT,
    TRUNCATED_PROTECTED_SOURCE,
    recover_c2_fixed_alpha,
    recover_c2b_varying_alpha,
    recovery_status,
    truncate_to_rank,
)
from src.recovery.truth import EvaluationTruthBinding

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = recovery_settings(REPO_ROOT)
MCTX = merge_context()
RCTX = recovery_context()
RETAINED = 3


def o3_fixed(
    retained: int = RETAINED, k: int = 4, alpha: float = 0.5
) -> tuple[FixtureTaskVector, ReleaseFamily, ObservedLineage, EvaluationTruthBinding]:
    protected, partners = protected_and_partners(k=k, protected_rank=8, partner_rank=3)
    family = o3_family(protected, partners, [alpha] * k, retained_rank=retained)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    return protected, family, observation, bind(observation, family, protected)


# ------------------------------------------------------------------ target and rank
def test_the_fixed_alpha_o3_target_is_the_truncated_protected_source() -> None:
    _, _, observation, _ = o3_fixed()
    result = recover_c2_fixed_alpha(observation)
    assert result.target_class == TRUNCATED_PROTECTED_SOURCE
    assert result.method == O3_SPECTRAL_FIXED
    assert result.operator == "O3_SVD_TRUNC_MERGE"


def test_the_varying_alpha_o3_target_is_also_the_truncated_protected_source() -> None:
    protected, partners = protected_and_partners(k=4, protected_rank=8, partner_rank=3)
    family = o3_family(protected, partners, [0.35, 0.45, 0.55, 0.65], retained_rank=RETAINED)
    observation = observe_incomplete_varying(family=family, context=RCTX)
    result = recover_c2b_varying_alpha(observation)
    assert result.target_class == TRUNCATED_PROTECTED_SOURCE
    assert result.method == O3_C2B_RESCALED


def test_the_solver_residual_rank_is_the_public_retained_rank() -> None:
    """The rank comes from the issued O3 metadata, never from a caller or from true A."""
    for retained in (2, 3, 5):
        _, _, observation, _ = o3_fixed(retained=retained)
        result = recover_c2_fixed_alpha(observation)
        assert result.residual_rank == retained
        assert result.retained_rank == retained
    # a linear lineage falls back to the configured LoRA rank instead
    from s08_fixtures import linear_family

    protected, partners = protected_and_partners(k=2)
    linear = observe_incomplete_fixed(
        family=linear_family(protected, partners, [0.5, 0.5]), context=RCTX
    )
    assert recover_c2_fixed_alpha(linear).residual_rank == RCTX.linear_residual_rank


def test_mixed_retained_ranks_in_one_family_are_refused() -> None:
    from s08_fixtures import FIXTURE_K

    from src.merge.family import build_release_family

    protected, partners = protected_and_partners(k=2, protected_rank=8)
    mixed = build_release_family(
        protected=protected,
        partners={f"B{index + 1}": partner for index, partner in enumerate(partners)},
        specs=[
            MergeSpec(
                descendant_id=f"O{index}",
                operator="O3_SVD_TRUNC_MERGE",
                alpha=0.5,
                partner_id=f"B{index + 1}",
                retained_rank=2 + index,
            )
            for index in range(2)
        ],
        permitted_k=FIXTURE_K,
        context=MCTX,
    )
    with pytest.raises(ObservationError, match="different retained ranks"):
        observe_incomplete_fixed(family=mixed, context=RCTX)


# ------------------------------------------------------------------ B20 error decomposition
def test_the_four_o3_errors_are_reported_separately_and_correctly() -> None:
    protected, _, observation, binding = o3_fixed()
    result = recover_c2_fixed_alpha(observation)
    report = o3_error_report(result, binding)

    wide = np.float64
    names = list(result.names)
    truncated = {n: truncate_to_rank(protected[n], rank=RETAINED, context=RCTX) for n in names}

    def frob(values: list[np.ndarray]) -> float:
        return float(np.sqrt(sum(float(np.sum(np.asarray(v, dtype=wide) ** 2)) for v in values)))

    original_norm = frob([protected[n] for n in names])
    truncated_norm = frob([truncated[n] for n in names])
    gap = frob([result[n].astype(wide) - truncated[n].astype(wide) for n in names])
    e_f = frob([result[n].astype(wide) - protected[n].astype(wide) for n in names]) / original_norm

    assert report["e_f"] == pytest.approx(e_f, rel=1e-9)
    assert report["e_solver_origscale"] == pytest.approx(gap / original_norm, rel=1e-9)
    assert report["e_solver_retained"] == pytest.approx(gap / truncated_norm, rel=1e-9)
    assert isinstance(report["e_floor"], float)
    assert report["retained_rank"] == RETAINED


def test_the_floor_is_reused_from_the_frozen_s07_implementation() -> None:
    from src.merge.diagnostics import protected_information_floor_from_o3

    protected, family, observation, binding = o3_fixed()
    result = recover_c2_fixed_alpha(observation)
    report = o3_error_report(result, binding, o3_source=family)
    assert report["e_floor_source"] == "s07.protected_information_floor_from_o3"
    assert report["e_floor"] == pytest.approx(
        protected_information_floor_from_o3(family.descendants[0], protected)
    )


def test_e_f_minus_e_floor_is_never_reported_as_solver_error() -> None:
    """B20: they are different quantities on different denominators."""
    _, _, observation, binding = o3_fixed()
    result = recover_c2_fixed_alpha(observation)
    report = o3_error_report(result, binding)

    difference = float(report["e_f"]) - float(report["e_floor"])  # type: ignore[arg-type]
    origscale = float(report["e_solver_origscale"])  # type: ignore[arg-type]
    retained = float(report["e_solver_retained"])  # type: ignore[arg-type]
    assert not np.isclose(difference, origscale, rtol=1e-3), (difference, origscale)
    assert not np.isclose(difference, retained, rtol=1e-3), (difference, retained)

    # and no production API exposes the subtraction
    source = Path("src/recovery/evaluation.py").read_text(encoding="utf-8")
    assert "e_f - e_floor" not in source.replace('"', "").replace("'", "")
    assert set(report) & {"e_f", "e_floor", "e_solver_origscale", "e_solver_retained"}
    assert not {key for key in report if "minus" in key or "difference" in key}


def test_a_non_o3_recovery_has_no_o3_decomposition() -> None:
    from s08_fixtures import linear_family

    protected, partners = protected_and_partners(k=2)
    family = linear_family(protected, partners, [0.5, 0.5])
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    result = recover_c2_fixed_alpha(observation)
    assert result.target_class == ORIGINAL_PROTECTED_CONSTITUENT
    with pytest.raises(EvaluationError, match=TRUNCATED_PROTECTED_SOURCE):
        o3_error_report(result, bind(observation, family, protected))


def test_a_zero_protected_constituent_is_an_explicit_undefined_state() -> None:
    from s08_fixtures import SURFACE

    from src.merge.updates import fixture_induced_update

    zero = fixture_induced_update(
        {name: np.zeros(shape) for name, shape in SURFACE.items()}, context=MCTX
    )
    _, partners = protected_and_partners(k=2)
    family = o3_family(zero, partners, [0.5, 0.5], retained_rank=2)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    report = o3_error_report(recover_c2_fixed_alpha(observation), bind(observation, family, zero))
    assert report["e_f"] == UNDEFINED
    assert report["e_solver_origscale"] == UNDEFINED


# ------------------------------------------------------------------ B18 authorization
def test_o3_recovery_is_implemented_but_not_authorized() -> None:
    status = recovery_status(RCTX)
    assert status["O3_RECOVERY_IMPLEMENTED"] is True
    assert status["O3_RECOVERY_AUTHORIZATION"] == "CONDITIONAL_NOT_RUN"
    assert material_text(SETTINGS, "o3_recovery_authorization") == "CONDITIONAL_NOT_RUN"


def test_s08_never_names_a_primary_lossy_operator_or_claims_a_gate() -> None:
    for path in sorted(Path("src/recovery").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "P1_PRIMARY_LOSSY_OPERATOR" not in source, path.name
        assert "P0-C4" not in source and "P0_C4" not in source, path.name
    from src.materials import UncalibratedConstantError, material
    from src.merge.settings import merge_settings

    with pytest.raises(UncalibratedConstantError):
        material(merge_settings(REPO_ROOT), "primary_lossy_operator")
    with pytest.raises(UncalibratedConstantError):
        material(merge_settings(REPO_ROOT), "svd_selected_retained_rank")


def test_dare_incomplete_recovery_remains_method_gated() -> None:
    status = recovery_status(RCTX)
    assert status["DARE_INCOMPLETE_RECOVERY"] == "METHOD_GATED"
    assert material_text(SETTINGS, "dare_incomplete_recovery_status") == "METHOD_GATED"
