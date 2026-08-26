"""C4-FINAL-1 — the TREATED schedule defines the matched update budget.

The defect: the common budget came from the smaller no-canary control, so a treated run with
20 natural records and 13 included canaries executed 5 updates and left 9 canaries unconsumed
while they still counted as trained members.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from blockc_fixtures import candidate_pool, fixture_contract, inclusion_for

from src.data.canaries import generate_canary_pool
from src.data.membership import canary_plan, no_canary_plan
from src.models.fixtures import build_base_model, load_fixture_spec
from src.provenance.config import resolve_config
from src.training.execution import (
    CANONICAL_TREATED_SCHEDULE,
    MATCHED_CONTROL_CYCLED_SCHEDULE,
    ExecutionContractError,
    batches_per_epoch,
    matched_execution_problems,
    matched_outcome_problems,
    matched_update_budget,
    treated_update_budget,
)
from src.training.lora import load_target_mapping
from src.training.settings import panel_settings
from src.training.trainer import TrainingError, build_schedule, train_reference_lora

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
SPEC = load_fixture_spec(resolve_config(CONFIGS / "models/tiny_fixture.json", config_root=CONFIGS))
MAPPING = load_target_mapping("tiny_fixture", panel_settings(REPO_ROOT).document)
BASE_NAMES = list(build_base_model(SPEC))

#: The pool size whose seed-101 Bernoulli(0.5) draw includes exactly 13 canaries. Found by
#: scanning pool sizes, never by moving the frozen inclusion probability or seed.
POOL_SIZE_FOR_THIRTEEN = 30
NATURAL = [f"T{index:04d}" for index in range(20)]


def arms(*, epochs: int = 1, batch_size: int = 4) -> tuple[Any, Any, int]:
    pool = generate_canary_pool(
        generator_seed=20260820, pool_size=POOL_SIZE_FOR_THIRTEEN, secret_length=12
    )
    inclusion = inclusion_for(pool, 101)
    candidates = candidate_pool(NATURAL, pool)
    treated_plan = canary_plan(
        label="CANARY",
        candidates=candidates,
        natural_ids=NATURAL,
        pool=pool,
        inclusion=inclusion,
    )
    control_plan = no_canary_plan(
        label="NO_CANARY", training_seed=101, candidates=candidates, natural_ids=NATURAL
    )
    budget = matched_update_budget(treated=treated_plan, epochs=epochs, batch_size=batch_size)

    def build(plan: Any) -> Any:
        return fixture_contract(
            plan=plan,
            base_parameter_names=BASE_NAMES,
            mapping=MAPPING,
            training_seed=101,
            optimizer_step_budget=budget,
            batch_size=batch_size,
        )

    return build(treated_plan), build(control_plan), budget


# ------------------------------------------------------------------ the required counterexample
def test_twenty_natural_thirteen_canaries_batch_four_epoch_one() -> None:
    treated, control, budget = arms()
    assert len(treated.plan.canaries) == 13
    assert treated.plan.n_training_records == 33
    assert control.plan.n_training_records == 20
    assert budget == 9, "ceil(33/4) = 9 updates, not the control's ceil(20/4) = 5"

    treated_run = train_reference_lora(treated, base=build_base_model(SPEC))
    control_run = train_reference_lora(control, base=build_base_model(SPEC))

    assert treated_run.optimizer_steps == control_run.optimizer_steps == 9
    assert len(treated_run.included_canary_ids) == 13
    assert len(treated_run.consumed_canary_ids) == 13
    assert treated_run.unconsumed_canary_ids == ()
    assert set(treated_run.consumed_canary_ids) == set(treated_run.included_canary_ids)
    assert control_run.included_canary_ids == ()
    assert matched_execution_problems(treated, control) == []
    assert matched_outcome_problems(treated_run, control_run) == []


def test_the_old_five_update_truncation_is_now_impossible() -> None:
    """Deriving the budget from the smaller control arm no longer type-checks or validates."""
    import inspect

    assert set(inspect.signature(matched_update_budget).parameters) == {
        "treated",
        "epochs",
        "batch_size",
    }
    treated, _, _ = arms()
    with pytest.raises(ExecutionContractError, match="included canaries would never be consumed"):
        fixture_contract(
            plan=treated.plan,
            base_parameter_names=BASE_NAMES,
            mapping=MAPPING,
            training_seed=101,
            optimizer_step_budget=5,
            batch_size=4,
        )


def test_the_budget_is_the_treated_schedule_not_the_smaller_arm() -> None:
    treated, control, budget = arms()
    assert budget == treated_update_budget(treated=treated.plan, epochs=1, batch_size=4)
    assert budget == batches_per_epoch(33, 4)
    assert budget > batches_per_epoch(control.plan.n_training_records, 4)


# ------------------------------------------------------------------ the control schedule
def test_the_control_cycles_its_natural_order_to_reach_the_budget() -> None:
    treated, control, budget = arms()
    assert treated.schedule_role == CANONICAL_TREATED_SCHEDULE
    assert control.schedule_role == MATCHED_CONTROL_CYCLED_SCHEDULE

    control_run = train_reference_lora(control, base=build_base_model(SPEC))
    assert control_run.optimizer_steps == budget
    assert len(control_run.schedule.batches) == batches_per_epoch(20, 4)
    assert control_run.schedule.cycles == budget / batches_per_epoch(20, 4)
    assert set(control_run.consumed_natural_ids) == set(NATURAL)


def test_the_control_never_gains_a_canary_or_loses_a_natural_record() -> None:
    _, control, _ = arms()
    control_run = train_reference_lora(control, base=build_base_model(SPEC))
    assert control_run.included_canary_ids == ()
    assert not any(record.startswith("canary-") for record in control_run.schedule.consumed_ids)
    assert set(control_run.natural_ids) == set(NATURAL)
    assert set(control_run.consumed_natural_ids) == set(NATURAL)
    assert control_run.schedule.consumed_ids == frozenset(NATURAL)


def test_the_control_schedule_is_deterministic_from_the_data_order_seed() -> None:
    _, control, _ = arms()
    first = train_reference_lora(control, base=build_base_model(SPEC)).schedule
    second = train_reference_lora(control, base=build_base_model(SPEC)).schedule
    assert first.order == second.order
    assert first.executed == second.executed
    assert first.batches == second.batches


def test_the_control_schedule_is_a_replay_of_its_own_order() -> None:
    """Cycling means batch k is batch (k mod n) of one epoch, not a fresh reshuffle."""
    _, control, budget = arms()
    run = train_reference_lora(control, base=build_base_model(SPEC))
    for step in range(budget):
        assert run.schedule.executed[step] == run.schedule.batches[step % len(run.schedule.batches)]


# ------------------------------------------------------------------ the consumption audit
def test_the_consumption_audit_is_available_before_privacy_evaluation() -> None:
    treated, _, _ = arms()
    audit = train_reference_lora(treated, base=build_base_model(SPEC)).consumption_audit()
    assert audit["schedule_role"] == CANONICAL_TREATED_SCHEDULE
    assert len(audit["included_canary_ids"]) == 13  # type: ignore[arg-type]
    assert len(audit["consumed_canary_ids"]) == 13  # type: ignore[arg-type]
    assert audit["unconsumed_canary_ids"] == []
    assert audit["n_consumed_natural"] == 20


def test_a_schedule_that_omits_a_canary_makes_execution_impossible() -> None:
    """Prefer refusing the run to recording a planned-but-never-consumed trained member."""
    treated, _, _ = arms()
    starved = build_schedule(
        treated.plan,
        order_seed=treated.seeds["data_order"],
        batch_size=4,
        budget=5,
        role=CANONICAL_TREATED_SCHEDULE,
    )
    unreached = set(c.canary_id for c in treated.plan.canaries) - starved.consumed_ids
    assert unreached, "the truncated schedule must leave canaries unreached"

    from src.training.execution import TrainingExecutionContract

    starved_contract = object.__new__(TrainingExecutionContract)
    for field, value in vars(treated).items():
        object.__setattr__(starved_contract, field, value)
    object.__setattr__(starved_contract, "optimizer_step_budget", 5)
    with pytest.raises(TrainingError, match="would never be consumed"):
        train_reference_lora(starved_contract, base=build_base_model(SPEC))


def test_the_outcome_audit_refuses_an_unmatched_pair() -> None:
    treated, control, _ = arms()
    treated_run = train_reference_lora(treated, base=build_base_model(SPEC))
    control_run = train_reference_lora(control, base=build_base_model(SPEC))
    assert matched_outcome_problems(treated_run, control_run) == []
    assert any(
        "different optimiser-update counts" in problem
        for problem in matched_outcome_problems(
            treated_run,
            treated_run.__class__(
                adapter=control_run.adapter,
                optimizer_steps=5,
                initial_loss=control_run.initial_loss,
                final_loss=control_run.final_loss,
                schedule=control_run.schedule,
                contract=control_run.contract,
                base_identity=control_run.base_identity,
            ),
        )
    )
    assert any(
        "carries synthetic canaries" in problem
        for problem in matched_outcome_problems(treated_run, treated_run)
    )


# ------------------------------------------------------------------ multi-epoch semantics
@pytest.mark.parametrize("epochs", [1, 2, 3])
def test_every_epoch_revisits_the_corpus_without_duplicating_a_canary(epochs: int) -> None:
    """One copy in the corpus; epoch iteration revisits it [AUTH: 00 §7.3]."""
    treated, control, budget = arms(epochs=epochs)
    assert budget == epochs * batches_per_epoch(33, 4)

    treated_run = train_reference_lora(treated, base=build_base_model(SPEC))
    control_run = train_reference_lora(control, base=build_base_model(SPEC))
    assert treated_run.optimizer_steps == control_run.optimizer_steps == budget
    assert treated_run.unconsumed_canary_ids == ()
    assert len(treated_run.consumed_canary_ids) == 13
    assert len(treated_run.schedule.order) == 33
    assert len(set(treated_run.schedule.order)) == 33, "the corpus holds one copy of each"
    assert treated_run.schedule.cycles == float(epochs)
    assert matched_outcome_problems(treated_run, control_run) == []


def test_a_larger_batch_still_consumes_every_canary() -> None:
    for batch_size in (2, 3, 4, 8, 16, 33):
        treated, control, budget = arms(batch_size=batch_size)
        assert budget == batches_per_epoch(33, batch_size)
        run = train_reference_lora(treated, base=build_base_model(SPEC))
        assert run.unconsumed_canary_ids == (), batch_size
        assert train_reference_lora(control, base=build_base_model(SPEC)).optimizer_steps == budget


def test_zero_or_negative_epochs_are_refused() -> None:
    treated, _, _ = arms()
    with pytest.raises(ExecutionContractError, match="epochs must be positive"):
        treated_update_budget(treated=treated.plan, epochs=0, batch_size=4)
    with pytest.raises(ExecutionContractError, match="batch size must be positive"):
        treated_update_budget(treated=treated.plan, epochs=1, batch_size=0)
