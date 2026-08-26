"""S06.8–S06.10 — the DP mechanism contract, the accountant, and the smoke-arm label."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.endpoints import ReportingError, report_normalised_privacy
from src.analysis.metrics import ELIGIBLE, INELIGIBLE, EligibilityVerdict
from src.dp.mechanism import (
    ACCOUNTANT_VERSION,
    ACCOUNTING_ASSUMPTIONS,
    BACKEND_READY,
    DP_BACKEND_STATUS,
    INFERENTIAL,
    NON_INFERENTIAL,
    SAMPLE_LEVEL_ADJACENCY,
    SMOKE_ONLY,
    UNSUPPORTED_ADJACENCY,
    DPError,
    DPMechanism,
    DPRun,
    DPUsageError,
    account,
    assert_inferential,
    calibrate_noise_multiplier,
    clip_to_norm,
    clipped_sum,
    dp_smoke_run,
    gaussian_noise,
    inferential_dp_set,
    poisson_sample_rate,
    privacy_endpoint_verdict,
    privatise,
    steps_for,
)


def mechanism(**overrides: object) -> DPMechanism:
    base = {
        "adjacency": SAMPLE_LEVEL_ADJACENCY,
        "delta": 1e-5,
        "clipping_norm": 1.0,
        "noise_multiplier": 1.0,
        "sample_rate": 0.01,
        "steps": 1000,
        "dp_seed": 101,
    }
    base.update(overrides)
    return DPMechanism(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ 00 §8.2 contract
def test_the_declared_adjacency_is_sample_level() -> None:
    assert mechanism().adjacency == SAMPLE_LEVEL_ADJACENCY
    assert "USER_LEVEL" in UNSUPPORTED_ADJACENCY


@pytest.mark.parametrize("adjacency", UNSUPPORTED_ADJACENCY)
def test_an_unsupported_adjacency_claim_is_refused(adjacency: str) -> None:
    with pytest.raises(DPError, match="adjacency"):
        mechanism(adjacency=adjacency)


@pytest.mark.parametrize(
    "override",
    [
        {"delta": 0.0},
        {"delta": 1.0},
        {"clipping_norm": 0.0},
        {"noise_multiplier": 0.0},
        {"sample_rate": 0.0},
        {"sample_rate": 1.5},
        {"steps": 0},
    ],
)
def test_an_incoherent_mechanism_is_refused(override: dict[str, object]) -> None:
    with pytest.raises(DPError):
        mechanism(**override)


def test_sigma_zero_is_not_differential_privacy() -> None:
    with pytest.raises(DPError, match="sigma = 0 is not DP"):
        mechanism(noise_multiplier=0.0)


def test_the_assumptions_travel_with_the_mechanism() -> None:
    document = mechanism().as_dict()
    assert document["accounting_assumptions"] == list(ACCOUNTING_ASSUMPTIONS)
    assert any("POISSON" in a for a in ACCOUNTING_ASSUMPTIONS)
    assert any("NO_REPETITION" in a for a in ACCOUNTING_ASSUMPTIONS)


def test_the_sample_rate_and_step_count_are_derived_not_asserted() -> None:
    assert poisson_sample_rate(batch_size=8, dataset_size=800) == 0.01
    assert steps_for(epochs=3, dataset_size=100, batch_size=8) == 39
    with pytest.raises(DPError):
        poisson_sample_rate(batch_size=10, dataset_size=5)


# ------------------------------------------------------------------ 00 §8.2 accounting
def test_the_accountant_is_deterministic() -> None:
    first, second = account(mechanism()), account(mechanism())
    assert first.achieved_epsilon == second.achieved_epsilon
    assert first.optimal_order == second.optimal_order
    assert first.accountant_version == ACCOUNTANT_VERSION


def test_epsilon_rises_as_noise_falls() -> None:
    values = [account(mechanism(noise_multiplier=s)).achieved_epsilon for s in (2.0, 1.5, 1.0, 0.8)]
    assert values == sorted(values), values


def test_epsilon_rises_with_more_steps() -> None:
    values = [account(mechanism(steps=n)).achieved_epsilon for n in (100, 500, 1000, 5000)]
    assert values == sorted(values), values


def test_epsilon_rises_with_a_larger_sample_rate() -> None:
    values = [account(mechanism(sample_rate=q)).achieved_epsilon for q in (0.002, 0.01, 0.05)]
    assert values == sorted(values), values


def test_the_accountant_reproduces_the_published_sampled_gaussian_value() -> None:
    """sigma=1, q=0.01, 1000 steps, delta=1e-5 is the textbook DP-SGD reference point."""
    assert account(mechanism()).achieved_epsilon == pytest.approx(2.11, abs=0.05)


def test_requested_and_achieved_epsilon_are_distinct_fields() -> None:
    """The mutation this kills: reporting the target epsilon as the achieved guarantee."""
    calibrated = calibrate_noise_multiplier(
        target_epsilon=8.0,
        delta=1e-5,
        sample_rate=0.01,
        steps=1000,
        clipping_norm=1.0,
        dp_seed=101,
    )
    result = account(calibrated)
    assert result.requested_epsilon == 8.0
    assert result.achieved_epsilon != result.requested_epsilon
    assert result.achieved_epsilon <= 8.0
    assert result.agreement_gap == result.achieved_epsilon - 8.0
    document = result.as_dict()
    assert document["requested_epsilon"] == 8.0
    assert document["achieved_epsilon"] == result.achieved_epsilon


def test_calibration_returns_a_feasible_noise_multiplier() -> None:
    calibrated = calibrate_noise_multiplier(
        target_epsilon=3.0,
        delta=1e-5,
        sample_rate=0.01,
        steps=1000,
        clipping_norm=1.0,
        dp_seed=101,
    )
    assert account(calibrated).achieved_epsilon <= 3.0
    looser = calibrate_noise_multiplier(
        target_epsilon=8.0,
        delta=1e-5,
        sample_rate=0.01,
        steps=1000,
        clipping_norm=1.0,
        dp_seed=101,
    )
    assert looser.noise_multiplier < calibrated.noise_multiplier


def test_an_empty_order_grid_fails_closed() -> None:
    with pytest.raises(DPError):
        account(mechanism(), orders=[])


def test_the_dp_backend_is_recorded_as_deferred_not_claimed() -> None:
    assert DP_BACKEND_STATUS.startswith("NOT_RUN_DEPENDENCY")
    assert "OPACUS" in DP_BACKEND_STATUS


# ------------------------------------------------------------------ mechanism primitives
def test_clipping_bounds_one_records_contribution() -> None:
    large = np.array([30.0, 40.0])
    assert float(np.linalg.norm(clip_to_norm(large, 1.0))) == pytest.approx(1.0)
    small = np.array([0.3, 0.4])
    assert np.array_equal(clip_to_norm(small, 1.0), small)


def test_clipping_happens_before_the_sum() -> None:
    gradients = [np.array([100.0, 0.0]), np.array([0.0, 100.0])]
    total = clipped_sum(gradients, 1.0)
    assert float(np.linalg.norm(total)) == pytest.approx(np.sqrt(2.0))


def test_noise_is_deterministic_in_the_recorded_dp_seed() -> None:
    first = gaussian_noise((4, 3), scale=2.0, dp_seed=101, step=0)
    assert np.array_equal(first, gaussian_noise((4, 3), scale=2.0, dp_seed=101, step=0))
    assert not np.array_equal(first, gaussian_noise((4, 3), scale=2.0, dp_seed=202, step=0))
    assert not np.array_equal(first, gaussian_noise((4, 3), scale=2.0, dp_seed=101, step=1))


def test_noise_has_the_declared_scale() -> None:
    sample = gaussian_noise((20000,), scale=3.0, dp_seed=7, step=0)
    assert float(np.std(sample)) == pytest.approx(3.0, rel=0.05)
    assert float(np.mean(sample)) == pytest.approx(0.0, abs=0.1)


def test_a_privatised_step_differs_from_the_clean_average() -> None:
    gradients = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    clean = sum(gradients) / len(gradients)
    noisy = privatise(gradients, mechanism=mechanism(), step=0)
    assert not np.allclose(noisy, clean)
    assert np.array_equal(noisy, privatise(gradients, mechanism=mechanism(), step=0))


# ------------------------------------------------------------------ B-C6 set-level standing
def facts(**overrides: str) -> dict[str, str]:
    base = {
        "execution_contract_sha256": "c" * 64,
        "model_identity": "meta-llama/Llama-3.2-3B@" + "a" * 40,
        "data_manifest_sha256": "d" * 64,
        "training_config_sha256": "e" * 64,
        "lora_policy": "LANGUAGE_TRUNK_MLP_ONLY",
        "backend_status": BACKEND_READY,
        "accounting_backend_status": BACKEND_READY,
    }
    base.update(overrides)
    return base


def run_at(seed: int, **overrides: str) -> DPRun:
    return DPRun(
        label="DP_CALIBRATION",
        training_seed=seed,
        accounting=account(mechanism()),
        **facts(**overrides),
    )


def test_the_smoke_run_is_non_inferential_by_construction() -> None:
    run = dp_smoke_run(smoke_seed=101, mechanism=mechanism())
    assert run.label == SMOKE_ONLY
    assert run.inferential_status == NON_INFERENTIAL
    assert not run.may_enter_privacy_endpoints
    assert run.training_seed == 101


def test_no_single_run_can_declare_itself_inferential() -> None:
    """B-C6: `DPRun(training_seed=101, inferential_status="INFERENTIAL")` no longer exists."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(DPRun)}
    assert "inferential_status" not in fields
    with pytest.raises(TypeError):
        DPRun(  # type: ignore[call-arg]
            label="DP",
            training_seed=101,
            accounting=account(mechanism()),
            inferential_status=INFERENTIAL,
        )
    assert run_at(101).inferential_status == NON_INFERENTIAL
    assert not run_at(101).may_enter_privacy_endpoints


def test_a_label_string_cannot_override_the_derived_status() -> None:
    for label in ("INFERENTIAL", "DP_CONFIRMATORY", "PRIMARY"):
        assert (
            DPRun(
                label=label, training_seed=101, accounting=account(mechanism()), **facts()
            ).inferential_status
            == NON_INFERENTIAL
        )


def test_one_seed_alone_is_not_an_inferential_set() -> None:
    """B-C6 counterexample: 101 only, with an INFERENTIAL label."""
    dp_set = inferential_dp_set([run_at(101)], required_seeds=(101, 202, 303))
    assert not dp_set.eligible
    assert any("missing seed(s) [202, 303]" in p for p in dp_set.problems())
    assert dp_set.status == NON_INFERENTIAL


def test_two_of_three_seeds_is_not_an_inferential_set() -> None:
    dp_set = inferential_dp_set([run_at(101), run_at(202)], required_seeds=(101, 202, 303))
    assert not dp_set.eligible
    assert any("missing seed(s) [303]" in p for p in dp_set.problems())


def test_three_of_the_same_seed_is_still_one_seed() -> None:
    dp_set = inferential_dp_set(
        [run_at(101), run_at(101), run_at(101)], required_seeds=(101, 202, 303)
    )
    assert any("repeats a training seed" in p for p in dp_set.problems())


def test_a_consistent_evidentiary_triple_is_structurally_eligible() -> None:
    dp_set = inferential_dp_set(
        [run_at(101), run_at(202), run_at(303)], required_seeds=(101, 202, 303)
    )
    assert dp_set.problems() == ()
    assert dp_set.eligible and dp_set.status == INFERENTIAL


def test_mismatched_conditions_across_the_seeds_are_refused() -> None:
    """B-C6: three seeds of three different scientific conditions are not an across-seed set."""
    dp_set = inferential_dp_set(
        [
            run_at(101),
            run_at(202, data_manifest_sha256="0" * 64),
            run_at(303, lora_policy="LANGUAGE_TRUNK_ALL_COMMON_LINEAR"),
        ],
        required_seeds=(101, 202, 303),
    )
    assert any("different scientific conditions" in p for p in dp_set.problems())


def test_a_differing_privacy_mechanism_breaks_the_condition() -> None:
    other = DPRun(
        label="DP_CALIBRATION",
        training_seed=303,
        accounting=account(mechanism(noise_multiplier=2.0)),
        **facts(),
    )
    dp_set = inferential_dp_set([run_at(101), run_at(202), other], required_seeds=(101, 202, 303))
    assert any("different scientific conditions" in p for p in dp_set.problems())


def test_a_deferred_backend_cannot_support_inference() -> None:
    dp_set = inferential_dp_set(
        [run_at(101), run_at(202), run_at(303, backend_status=DP_BACKEND_STATUS)],
        required_seeds=(101, 202, 303),
    )
    assert not dp_set.eligible
    assert any("303" in p and "READY" in p for p in dp_set.problems())


def test_a_deferred_accountant_cannot_support_inference() -> None:
    dp_set = inferential_dp_set(
        [
            run_at(101),
            run_at(202),
            run_at(303, accounting_backend_status=DP_BACKEND_STATUS),
        ],
        required_seeds=(101, 202, 303),
    )
    assert not dp_set.eligible


def test_runs_that_do_not_bind_their_identity_cannot_be_checked() -> None:
    unbound = DPRun(label="DP", training_seed=101, accounting=account(mechanism()))
    dp_set = inferential_dp_set([unbound, run_at(202), run_at(303)], required_seeds=(101, 202, 303))
    assert any(
        "do not bind their model/data/config/contract identity" in p for p in dp_set.problems()
    )


def test_using_a_non_inferential_set_where_inference_is_required_raises() -> None:
    dp_set = inferential_dp_set([run_at(101)], required_seeds=(101, 202, 303))
    with pytest.raises(DPUsageError, match="may not enter R_priv"):
        assert_inferential(dp_set)


def eligible_verdict() -> EligibilityVerdict:
    return EligibilityVerdict(status=ELIGIBLE, oracle_median=0.4, signal_ci=(0.1, 0.3), reasons=())


def test_a_one_seed_dp_arm_downgrades_an_eligible_verdict() -> None:
    dp_set = inferential_dp_set([run_at(101)], required_seeds=(101, 202, 303))
    verdict = privacy_endpoint_verdict(eligible_verdict(), dp_set=dp_set)
    assert verdict.status == INELIGIBLE
    assert any("no validated inferential seed set" in reason for reason in verdict.reasons)


def test_a_one_seed_dp_arm_cannot_reach_r_priv_g_l_or_g_r() -> None:
    """The Block B reporting gate refuses it, with no Block B change [AUTH: 00 §8.4, §18.1]."""
    dp_set = inferential_dp_set(
        [dp_smoke_run(smoke_seed=101, mechanism=mechanism(), **facts())],
        required_seeds=(101, 202, 303),
    )
    verdict = privacy_endpoint_verdict(eligible_verdict(), dp_set=dp_set)
    with pytest.raises(ReportingError):
        report_normalised_privacy(
            verdict=verdict,
            denominator_label="USABLE",
            m_view=0.4,
            m_desc=0.3,
            m_pool=0.35,
            m_rec=0.5,
            m_oracle=0.6,
            null_level=0.01,
        )


def test_a_validated_set_leaves_the_verdict_alone() -> None:
    dp_set = inferential_dp_set(
        [run_at(101), run_at(202), run_at(303)], required_seeds=(101, 202, 303)
    )
    base = eligible_verdict()
    assert privacy_endpoint_verdict(base, dp_set=dp_set) is base
    assert_inferential(dp_set)


def test_a_non_dp_condition_passes_the_base_verdict_through() -> None:
    base = eligible_verdict()
    assert privacy_endpoint_verdict(base, dp_set=None) is base


def test_a_non_inferential_set_can_never_upgrade_an_ineligible_verdict() -> None:
    dp_set = inferential_dp_set([run_at(101)], required_seeds=(101, 202, 303))
    base = EligibilityVerdict(
        status=INELIGIBLE, oracle_median=0.0, signal_ci=(-0.1, 0.1), reasons=("weak signal",)
    )
    verdict = privacy_endpoint_verdict(base, dp_set=dp_set)
    assert verdict.status == INELIGIBLE
    assert "weak signal" in verdict.reasons


def test_an_empty_set_is_refused() -> None:
    assert inferential_dp_set([], required_seeds=(101, 202, 303)).problems()
