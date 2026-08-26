"""S06.6/S06.7 — adapter-only training under ONE execution contract, and the matched control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from blockc_fixtures import candidate_pool, fixture_contract, fixture_model, inclusion_for

from src.data.canaries import generate_canary_pool
from src.data.membership import canary_plan, no_canary_plan
from src.models.fixtures import build_base_model, load_fixture_spec, target_parameter_names
from src.provenance.config import resolve_config
from src.training.execution import (
    ExecutionContractError,
    authorise_training,
    matched_execution_problems,
    matched_update_budget,
    reference_backend,
)
from src.training.lora import load_target_mapping
from src.training.seeds import seed_families
from src.training.settings import panel_settings
from src.training.trainer import (
    PRODUCTION_BACKEND_STATUS,
    TrainingError,
    adapter_file_hash,
    data_order,
    initialise_adapter,
    load_adapter,
    save_adapter,
    train_reference_lora,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
SPEC = load_fixture_spec(resolve_config(CONFIGS / "models/tiny_fixture.json", config_root=CONFIGS))
MAPPING = load_target_mapping("tiny_fixture", panel_settings(REPO_ROOT).document)
TARGETS = target_parameter_names(SPEC)
BASE_NAMES = list(build_base_model(SPEC))
POOL = generate_canary_pool(generator_seed=20260820, pool_size=16, secret_length=12)
NATURAL = [f"T{index:04d}" for index in range(12)]
BUDGET = 12


def plan_for(
    seed: int = 101, *, with_canaries: bool = True, natural: list[str] | None = None
) -> Any:
    ids = NATURAL if natural is None else natural
    candidates = candidate_pool(ids, POOL)
    if not with_canaries:
        return no_canary_plan(
            label="NO_CANARY", training_seed=seed, candidates=candidates, natural_ids=ids
        )
    return canary_plan(
        label="CANARY",
        candidates=candidates,
        natural_ids=ids,
        pool=POOL,
        inclusion=inclusion_for(POOL, seed),
    )


def contract_for(seed: int = 101, *, with_canaries: bool = True, budget: int = BUDGET) -> Any:
    return fixture_contract(
        plan=plan_for(seed, with_canaries=with_canaries),
        base_parameter_names=BASE_NAMES,
        mapping=MAPPING,
        training_seed=seed,
        optimizer_step_budget=budget,
    )


def run(seed: int = 101, *, with_canaries: bool = True, budget: int = BUDGET) -> Any:
    base = build_base_model(SPEC)
    return base, train_reference_lora(
        contract_for(seed, with_canaries=with_canaries, budget=budget), base=base
    )


# ------------------------------------------------------------------ initialisation
def test_an_untrained_adapter_is_an_exact_no_op() -> None:
    shapes = {name: (4, 6) for name in ("a", "b")}
    adapter = initialise_adapter(shapes, rank=2, scaling=2.0, init_seed=7)
    for name in shapes:
        assert np.count_nonzero(adapter.induced_update(name)) == 0


def test_a_shape_that_cannot_carry_the_rank_fails_closed() -> None:
    with pytest.raises(TrainingError, match="rank"):
        initialise_adapter({"a": (4, 4)}, rank=8, scaling=1.0, init_seed=1)


def test_initialisation_is_deterministic_in_the_lora_init_seed() -> None:
    shapes = {"a": (8, 8)}
    first = initialise_adapter(shapes, rank=2, scaling=1.0, init_seed=7)
    second = initialise_adapter(shapes, rank=2, scaling=1.0, init_seed=7)
    third = initialise_adapter(shapes, rank=2, scaling=1.0, init_seed=8)
    assert first.identity() == second.identity() != third.identity()


# ------------------------------------------------------------------ training mechanics
def test_the_loss_moves_in_the_expected_direction() -> None:
    _, outcome = run()
    assert outcome.final_loss < outcome.initial_loss


def test_base_weights_are_bitwise_unchanged() -> None:
    base, outcome = run()
    for name, array in build_base_model(SPEC).items():
        assert np.array_equal(base[name], array), name
    assert outcome.base_identity


def test_adapter_weights_actually_change() -> None:
    _, outcome = run()
    for name in TARGETS:
        _, b = outcome.adapter.factors[name]
        assert np.count_nonzero(b) > 0, name


def test_the_run_executes_exactly_the_contracted_update_budget() -> None:
    from src.training.execution import batches_per_epoch

    required = batches_per_epoch(plan_for().n_training_records, 3)
    for budget in (required, required + 5, required * 3):
        _, outcome = run(budget=budget)
        assert outcome.optimizer_steps == budget


def test_a_budget_too_small_to_reach_every_canary_is_refused() -> None:
    """C4: a treated budget that cannot cover its own corpus leaves canaries unconsumed."""
    from src.training.execution import batches_per_epoch

    required = batches_per_epoch(plan_for().n_training_records, 3)
    with pytest.raises(ExecutionContractError, match="included canaries would never be consumed"):
        contract_for(budget=required - 1)


def test_the_same_contract_reproduces_the_adapter_bitwise() -> None:
    _, first = run()
    _, second = run()
    assert first.adapter.identity() == second.adapter.identity()
    assert first.adapter.update_identity() == second.adapter.update_identity()
    assert first.order == second.order
    assert first.contract.identity() == second.contract.identity()


def test_a_different_training_seed_produces_a_different_run() -> None:
    _, first = run(seed=101)
    _, second = run(seed=202)
    assert first.adapter.identity() != second.adapter.identity()
    assert first.order != second.order


def test_the_data_order_seed_alone_changes_the_order() -> None:
    ids = [f"T{i:04d}" for i in range(20)]
    first_seed = seed_families(101, differentially_private=False)["data_order"]
    second_seed = seed_families(202, differentially_private=False)["data_order"]
    assert data_order(ids, order_seed=first_seed) != data_order(ids, order_seed=second_seed)
    assert sorted(data_order(ids, order_seed=first_seed)) == sorted(ids)
    assert data_order(ids, order_seed=first_seed) == data_order(
        list(reversed(ids)), order_seed=first_seed
    )


def test_a_target_absent_from_the_base_model_fails_closed() -> None:
    contract = contract_for()
    partial = {k: v for k, v in build_base_model(SPEC).items() if k != TARGETS[0]}
    with pytest.raises(TrainingError, match="absent from the base model"):
        train_reference_lora(contract, base=partial)


# ------------------------------------------------------------------ B-C4 contract binding
def test_execution_takes_no_independent_rank_scaling_or_target_list() -> None:
    """There is exactly one place these can come from, so they cannot disagree."""
    import inspect

    signature = inspect.signature(train_reference_lora)
    assert set(signature.parameters) == {"contract", "base"}


def test_the_audit_is_derived_from_the_surface_execution_will_use() -> None:
    contract = contract_for()
    assert set(contract.audit.adapter_parameters) == {
        f"{name}.lora_{part}" for name in TARGETS for part in ("A", "B")
    }
    assert contract.audit.rank == contract.rank
    assert contract.audit.scaling == contract.scaling


def test_a_forbidden_module_cannot_be_authorised_for_execution() -> None:
    """B-C4: lm_head.weight training while the audit says MLP_ONLY."""
    with pytest.raises(Exception) as caught:
        authorise_training(
            plan=plan_for(),
            model=fixture_model(),
            mapping=MAPPING,
            base_parameter_names=["lm_head.weight", "model.embed_tokens.weight"],
            rank=8,
            scaling=2.0,
            dropout=0.0,
            learning_rate=0.5,
            batch_size=3,
            optimizer_step_budget=BUDGET,
            precision="float64",
            sequence_length=128,
            seeds=seed_families(101, differentially_private=False),
            privacy_regime="NON_DP",
            backend=reference_backend(),
            resolved_config_sha256="f" * 64,
        )
    assert "no parameter matched" in str(caught.value) or "forbidden" in str(caught.value)


def test_an_audit_that_disagrees_with_execution_rank_is_unrepresentable() -> None:
    good = contract_for()
    with pytest.raises(ExecutionContractError, match="audit rank"):
        type(good)(
            plan=good.plan,
            model=good.model,
            mapping=good.mapping,
            target_names=good.target_names,
            audit=good.audit,
            rank=good.rank + 8,
            scaling=good.scaling,
            dropout=good.dropout,
            learning_rate=good.learning_rate,
            batch_size=good.batch_size,
            optimizer_step_budget=good.optimizer_step_budget,
            precision=good.precision,
            sequence_length=good.sequence_length,
            seeds=good.seeds,
            privacy_regime=good.privacy_regime,
            backend=good.backend,
            resolved_config_sha256=good.resolved_config_sha256,
        )


def test_a_contract_whose_seeds_are_for_another_run_is_refused() -> None:
    good = contract_for(seed=101)
    with pytest.raises(ExecutionContractError, match="seed families are for training seed"):
        type(good)(
            plan=good.plan,
            model=good.model,
            mapping=good.mapping,
            target_names=good.target_names,
            audit=good.audit,
            rank=good.rank,
            scaling=good.scaling,
            dropout=good.dropout,
            learning_rate=good.learning_rate,
            batch_size=good.batch_size,
            optimizer_step_budget=good.optimizer_step_budget,
            precision=good.precision,
            sequence_length=good.sequence_length,
            seeds=seed_families(202, differentially_private=False),
            privacy_regime=good.privacy_regime,
            backend=good.backend,
            resolved_config_sha256=good.resolved_config_sha256,
        )


# ------------------------------------------------------------------ save / load
def test_a_reloaded_adapter_gives_an_identical_induced_update() -> None:
    _, outcome = run()
    reloaded = load_adapter(save_adapter(outcome.adapter))
    assert reloaded.identity() == outcome.adapter.identity()
    for name, delta in outcome.adapter.induced_updates().items():
        assert np.array_equal(reloaded.induced_update(name), delta)
    assert reloaded.update_identity() == outcome.adapter.update_identity()


def test_a_tampered_adapter_document_is_refused_on_load() -> None:
    _, outcome = run()
    document = save_adapter(outcome.adapter)
    name = sorted(outcome.adapter.factors)[0]
    factors: Any = document["factors"]
    factors[name]["B"][0][0] += 1.0
    with pytest.raises(TrainingError, match="recorded identity"):
        load_adapter(document)


def test_the_adapter_file_hash_changes_with_the_adapter() -> None:
    _, first = run(seed=101)
    _, second = run(seed=202)
    assert adapter_file_hash(save_adapter(first.adapter)) != adapter_file_hash(
        save_adapter(second.adapter)
    )


def test_the_induced_update_is_the_scaled_low_rank_product() -> None:
    _, outcome = run()
    for name in TARGETS:
        a, b = outcome.adapter.factors[name]
        assert np.allclose(outcome.adapter.induced_update(name), outcome.contract.scaling * (b @ a))
        assert np.linalg.matrix_rank(outcome.adapter.induced_update(name)) <= outcome.contract.rank


# ------------------------------------------------------------------ S06.7 matched control
def test_the_matched_control_trains_the_same_natural_records_without_canaries() -> None:
    _, control = run(with_canaries=False)
    assert len(control.order) == len(NATURAL)
    assert not any(order.startswith("canary-") for order in control.order)


def test_the_two_arms_execute_the_same_number_of_updates() -> None:
    """B-C4 counterexample: 10 updates against 8 when the arms hold different record counts."""
    treated_plan = plan_for(101, with_canaries=True)
    control_plan = plan_for(101, with_canaries=False)
    assert treated_plan.n_training_records != control_plan.n_training_records

    naive_treated = (treated_plan.n_training_records + 1) // 2
    naive_control = (control_plan.n_training_records + 1) // 2
    assert naive_treated != naive_control, "the fixture must expose the mismatch"

    budget = matched_update_budget(treated=treated_plan, epochs=1, batch_size=2)
    treated = fixture_contract(
        plan=treated_plan,
        base_parameter_names=BASE_NAMES,
        mapping=MAPPING,
        training_seed=101,
        optimizer_step_budget=budget,
        batch_size=2,
    )
    control = fixture_contract(
        plan=control_plan,
        base_parameter_names=BASE_NAMES,
        mapping=MAPPING,
        training_seed=101,
        optimizer_step_budget=budget,
        batch_size=2,
    )
    assert matched_execution_problems(treated, control) == []
    treated_run = train_reference_lora(treated, base=build_base_model(SPEC))
    control_run = train_reference_lora(control, base=build_base_model(SPEC))
    assert treated_run.optimizer_steps == control_run.optimizer_steps == budget


def test_unequal_update_budgets_are_reported_as_an_unmatched_execution() -> None:
    treated = contract_for(101, with_canaries=True, budget=10)
    control = contract_for(101, with_canaries=False, budget=8)
    problems = matched_execution_problems(treated, control)
    assert any("optimiser-update budgets: 10 against 8" in p for p in problems)


def test_the_production_backend_is_recorded_as_deferred() -> None:
    _, outcome = run()
    assert outcome.production_backend_status == PRODUCTION_BACKEND_STATUS
    assert PRODUCTION_BACKEND_STATUS.startswith("NOT_RUN_DEPENDENCY")
    assert outcome.contract.backend.status.startswith("NOT_RUN_DEPENDENCY")
    assert not outcome.contract.backend.ready
