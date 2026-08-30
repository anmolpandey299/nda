"""S07 release family -> S08 observation -> recovery -> evaluation, end to end.

This is the cross-stage seam: an S07 descendant is issued by the merge stage, an S08
observation strips it to what the lineage regime authorises, a solver runs on the stripped
object alone, and the evaluator — the only component holding truth — scores it. Each unit
tests its own half; this test checks that the halves compose without either the truth or the
hidden partner crossing the firewall [AUTH: 00 §24A, §25, §26; 01 §39 S07, S08].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from s08_fixtures import (
    SURFACE,
    bind,
    linear_family,
    number,
    o3_family,
    protected_and_partners,
    recoverable_bank,
    recovery_context,
)

from src.recovery.conditioning import conditioning_from_observations, direction_matrix
from src.recovery.context import resolve_recovery_context
from src.recovery.evaluation import (
    beta_spread,
    c1_pass,
    o3_error_report,
    parameter_recovery_metrics,
)
from src.recovery.observations import (
    Z_HIGH,
    Z_INCOMPLETE_FIXED,
    Z_INCOMPLETE_VARYING_KNOWN,
    NotAuthorizedError,
    observe_incomplete_fixed,
    observe_incomplete_varying,
    observe_known_partner,
)
from src.recovery.settings import coefficient_schedule, descendant_counts, recovery_settings
from src.recovery.solvers import (
    C1_METHOD,
    C2_NAIVE_SHARED_MEAN,
    C2_SPECTRAL_FIXED,
    C2B_RESCALED,
    RecoveryResult,
    recover_c1,
    recover_c2_fixed_alpha,
    recover_c2_naive_shared_mean,
    recover_c2b_varying_alpha,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RCTX = recovery_context()
SETTINGS = recovery_settings(REPO_ROOT)
C1_LIMIT = 1e-05


# ------------------------------------------------------------------ C1: the oracle control
@pytest.mark.parametrize("alpha", [0.35, 0.5, 0.65])
def test_c1_inverts_a_linear_descendant_end_to_end(alpha: float) -> None:
    protected, partners = protected_and_partners(k=1)
    descendant = linear_family(protected, partners, [alpha]).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=RCTX
    )
    binding = bind(observation, descendant, protected)
    assert observation.regime == Z_HIGH
    result = recover_c1(observation)
    assert result.method == C1_METHOD
    assert result.method_role == "ORACLE_CONTROL"
    assert c1_pass(result, binding)
    assert number(parameter_recovery_metrics(result, binding), "e_f") < C1_LIMIT


def test_the_c1_control_is_the_upper_bound_the_occupied_regime_is_measured_against() -> None:
    """C1 knows the partner; C2 does not. The gap between them is the S08 result."""
    protected, partners = recoverable_bank(k=4)
    family = linear_family(protected, partners, [0.5] * 4)
    single = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    high = observe_known_partner(descendant=single, partner_update=partners[0], context=RCTX)
    incomplete = observe_incomplete_fixed(family=family, context=RCTX)
    oracle = recover_c1(high)
    occupied = recover_c2_fixed_alpha(incomplete)
    oracle_error = number(parameter_recovery_metrics(oracle, bind(high, single, protected)), "e_f")
    occupied_error = number(
        parameter_recovery_metrics(occupied, bind(incomplete, family, protected)), "e_f"
    )
    assert oracle_error <= occupied_error
    assert oracle.method_role == "ORACLE_CONTROL"
    assert occupied.method_role == "OCCUPIED_REGIME_REPRODUCTION"


# ------------------------------------------------------------------ C2: shared source
def _spectral_and_naive(
    k: int, seed: int = 90210
) -> tuple[float, float, RecoveryResult, RecoveryResult]:
    protected, partners = recoverable_bank(k=k, seed=seed)
    family = linear_family(protected, partners, [0.5] * k)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    binding = bind(observation, family, protected)
    spectral = recover_c2_fixed_alpha(observation)
    naive = recover_c2_naive_shared_mean(observation)
    #: Both comparands read the SAME observation, so nothing but the method differs.
    assert spectral.observation_identity == naive.observation_identity
    return (
        number(parameter_recovery_metrics(spectral, binding), "e_f"),
        number(parameter_recovery_metrics(naive, binding), "e_f"),
        spectral,
        naive,
    )


def test_c2_beats_the_naive_comparator_on_the_planted_k4_fixture() -> None:
    """The comparator is the point: a mean of descendants is not a recovery.

    On this planted k = 4 fixture at residual rank 3 the spectral solve is exact and the naive
    mean is nowhere near it. This is the stage requirement: at least one planted recoverable
    fixture on which the published-style C2 method beats the naive baseline.
    """
    spectral_error, naive_error, spectral, naive = _spectral_and_naive(4)
    assert spectral.method == C2_SPECTRAL_FIXED
    assert naive.method == C2_NAIVE_SHARED_MEAN
    assert naive.method_role == "NAIVE_COMPARATOR"
    assert spectral_error < naive_error
    assert spectral_error < 1e-05


def test_k2_fixture_does_not_claim_exact_identification() -> None:
    """MEASURED, NOT TUNED — and scoped to this fixture.

    On THIS planted k = 2 fixture the fixed-rank C2 objective admits a solution the iteration
    settles on that is not the planted source, and the solve does not beat the naive shared
    mean. S08 claims nothing beyond that: no theorem about two-descendant families is asserted
    here, and none exists in the frozen authority. The observation is recorded rather than
    removed by re-picking a fixture or a seed [AUTH: 00 §34B.1A no result-driven tuning].
    """
    k2_spectral, k2_naive, _, _ = _spectral_and_naive(2)
    assert k2_spectral > k2_naive, "the recorded k=2 fixture observation has changed"
    k4_spectral, _, _, _ = _spectral_and_naive(4)
    assert k4_spectral < k2_spectral


def test_the_fixed_iteration_count_is_past_convergence_not_a_stopping_rule() -> None:
    """A fixed point is reached and then held: 200, 1000 and 5000 sweeps agree exactly.

    On this fixture the iterate is unchanged well before the configured count, so the count is
    not standing in for a tuning knob. The frozen design is still a fixed N with no adaptive
    stop [AUTH: 00 §25]; this test records that the count is not doing hidden work.
    """
    protected, partners = recoverable_bank(k=4)
    family = linear_family(protected, partners, [0.5] * 4)
    identities = set()
    for iterations in (200, 1000, 5000):
        context = recovery_context(n_iters=iterations, residual_rank=3)
        observation = observe_incomplete_fixed(family=family, context=context)
        identities.add(recover_c2_fixed_alpha(observation).content_identity())
    assert len(identities) == 1


def test_the_permitted_descendant_counts_both_run() -> None:
    counts = descendant_counts(SETTINGS)
    assert counts == (2, 4)
    for k in counts:
        protected, partners = recoverable_bank(k=k)
        observation = observe_incomplete_fixed(
            family=linear_family(protected, partners, [0.5] * k), context=RCTX
        )
        assert len(observation.updates) == k
        assert recover_c2_fixed_alpha(observation).regime == Z_INCOMPLETE_FIXED


def test_a_single_descendant_cannot_support_a_shared_source_solve() -> None:
    from src.recovery.observations import StructuralNAError

    protected, partners = recoverable_bank(k=1)
    with pytest.raises(StructuralNAError):
        observe_incomplete_fixed(family=linear_family(protected, partners, [0.5]), context=RCTX)


# ------------------------------------------------------------------ C2B: varying alpha
@pytest.mark.parametrize(
    "schedule_key", ["c2b_k2_reference_schedule", "c2b_k4_spread_matched_schedule"]
)
def test_c2b_runs_each_registered_coefficient_schedule(schedule_key: str) -> None:
    alphas = coefficient_schedule(SETTINGS, schedule_key)
    protected, partners = recoverable_bank(k=len(alphas))
    family = linear_family(protected, partners, list(alphas))
    observation = observe_incomplete_varying(family=family, context=RCTX)
    assert observation.regime == Z_INCOMPLETE_VARYING_KNOWN
    assert observation.alphas == pytest.approx(alphas)
    result = recover_c2b_varying_alpha(observation)
    assert result.method == C2B_RESCALED
    binding = bind(observation, family, protected)
    assert number(parameter_recovery_metrics(result, binding), "e_f") >= 0.0
    assert number(beta_spread(list(alphas)), "rho_beta") >= 1.0


def test_the_wide_schedule_is_more_spread_than_the_matched_one() -> None:
    """00 §25 P0-C2B.1 registers both so the spread axis is a designed contrast."""
    matched = beta_spread(list(coefficient_schedule(SETTINGS, "c2b_k4_spread_matched_schedule")))
    wide = beta_spread(list(coefficient_schedule(SETTINGS, "c2b_k4_wide_schedule")))
    assert number(wide, "rho_beta") > number(matched, "rho_beta")


# ------------------------------------------------------------------ O3: truncated source
def test_o3_recovery_reports_the_floor_and_the_solver_error_separately() -> None:
    """00 §28B: the truncation floor is not the solver's error, and is never subtracted."""
    protected, partners = recoverable_bank(k=4)
    family = o3_family(protected, partners, [0.5] * 4, retained_rank=4)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    result = recover_c2_fixed_alpha(observation)
    assert result.target_class == "TRUNCATED_PROTECTED_SOURCE"
    assert result.retained_rank == 4
    report = o3_error_report(result, bind(observation, family, protected), o3_source=family)
    for field in ("e_f", "e_floor", "e_solver_origscale", "e_solver_retained", "e_floor_source"):
        assert field in report
    assert number(report, "e_floor") >= 0.0
    #: A difference is deliberately never reported; a caller wanting one must justify it.
    assert not any("minus" in key or "excess" in key for key in report)


def test_a_linear_family_has_no_truncation_floor() -> None:
    protected, partners = recoverable_bank(k=4)
    observation = observe_incomplete_fixed(
        family=linear_family(protected, partners, [0.5] * 4), context=RCTX
    )
    result = recover_c2_fixed_alpha(observation)
    assert result.target_class == "ORIGINAL_PROTECTED_CONSTITUENT"
    assert result.retained_rank is None


# ------------------------------------------------------------------ conditioning
def test_conditioning_is_reachable_only_from_the_known_partner_regime() -> None:
    protected, partners = protected_and_partners(k=2)
    family = linear_family(protected, partners, [0.5, 0.5])
    high = [
        observe_known_partner(descendant=d, partner_update=p, context=RCTX)
        for d, p in zip(family.descendants, partners, strict=True)
    ]
    report = conditioning_from_observations(high, context=RCTX)
    assert report["k"] == 2
    assert direction_matrix(high).shape[1] == 2

    incomplete = observe_incomplete_fixed(family=family, context=RCTX)
    with pytest.raises(NotAuthorizedError):
        direction_matrix([incomplete])


# ------------------------------------------------------------------ the firewall, composed
def test_no_recovery_result_carries_truth_or_a_privacy_outcome() -> None:
    protected, partners = recoverable_bank(k=4)
    observation = observe_incomplete_fixed(
        family=linear_family(protected, partners, [0.5] * 4), context=RCTX
    )
    row = recover_c2_fixed_alpha(observation).as_dict()
    forbidden = ("truth", "e_f", "success", "tpr", "auc", "privacy", "alpha_true", "partner")
    for key in row:
        assert not any(marker in key.lower() for marker in forbidden), key
    #: A fixture solve is labelled as one; the scientific class is the unsuffixed string.
    assert row["provenance_class"] == "NON_EVIDENTIARY_S08_RECOVERY_FIXTURE"
    assert row["recovery_profile"] == "FIXTURE_ONLY_NOT_SCIENTIFIC"


def test_the_result_binds_the_exact_observation_and_context_it_ran_under() -> None:
    protected, partners = recoverable_bank(k=4)
    observation = observe_incomplete_fixed(
        family=linear_family(protected, partners, [0.5] * 4), context=RCTX
    )
    result = recover_c2_fixed_alpha(observation)
    row = result.as_dict()
    assert row["observation_sha256"] == observation.identity()
    assert row["recovery_context_sha256"] == RCTX.identity()
    assert row["iterations"] == RCTX.c2_n_iters
    assert row["residual_rank"] == RCTX.linear_residual_rank


def test_the_production_context_is_the_one_the_repo_config_resolves_to() -> None:
    production = resolve_recovery_context(REPO_ROOT)
    assert production.c2_n_iters == 1000
    assert production.linear_residual_rank == 32
    assert production.arithmetic_dtype == "float32"
    assert production.identity() != RCTX.identity(), "the fixture context must not be production"


def test_a_recovered_artifact_is_float32_and_read_only() -> None:
    protected, partners = recoverable_bank(k=4)
    observation = observe_incomplete_fixed(
        family=linear_family(protected, partners, [0.5] * 4), context=RCTX
    )
    result = recover_c2_fixed_alpha(observation)
    for name in result.names:
        view = result[name]
        assert view.dtype == np.float32
        assert view.shape == SURFACE[name]
        assert not view.flags.writeable
        with pytest.raises(ValueError):
            view[0, 0] = 1.0
        writable = result.copy_of(name)
        writable[0, 0] = 1.0
        assert result[name][0, 0] != writable[0, 0]
