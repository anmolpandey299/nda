"""The full-sweep plan: derived, deterministic, and honest about what it cannot run.

Fixtures only. Nothing here trains, downloads, merges or scores [AUTH: 00 §35, §36].
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.experiments.gates import GateState
from src.experiments.registry import CALIBRATION_SEEDS, load_registry
from src.experiments.sweep import (
    BLOCKED_BY_UNCALIBRATED_CONSTANT,
    MISSING_EXECUTION_BINDING,
    READINESS_CLASSES,
    TRAINING_BINDING,
    build_sweep_plan,
    readiness_audit,
    uncalibrated_training_constants,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(name="unresolved")
def _unresolved() -> GateState:
    """No P0 outcome has been opened, which is the honest current state."""
    return GateState(provenance_class="NON_EVIDENTIARY_FIXTURE", records={})


def test_the_plan_is_deterministic(unresolved: GateState) -> None:
    first = build_sweep_plan(ROOT, state=unresolved)
    second = build_sweep_plan(ROOT, state=unresolved)
    assert first.identity() == second.identity()
    assert json.dumps(first.as_dict(), sort_keys=True) == json.dumps(
        second.as_dict(), sort_keys=True
    )


def test_every_task_carries_a_registered_readiness_class(unresolved: GateState) -> None:
    plan = build_sweep_plan(ROOT, state=unresolved)
    assert plan.tasks
    for task in plan.tasks:
        assert task.readiness in READINESS_CLASSES
        assert task.execution_binding
        assert task.resolved_config_sha256()


def test_task_ids_are_unique_and_dependencies_resolve(unresolved: GateState) -> None:
    plan = build_sweep_plan(ROOT, state=unresolved)
    identifiers = [task.task_id for task in plan.tasks]
    assert len(identifiers) == len(set(identifiers))
    known = set(identifiers)
    for task in plan.tasks:
        unknown = sorted(set(task.depends_on) - known)
        assert not unknown, f"{task.task_id} depends on unenumerated {unknown}"


def test_every_registered_cell_is_enumerated(unresolved: GateState) -> None:
    """A cell the sweep forgets is a cell the study silently never runs."""
    plan = build_sweep_plan(ROOT, state=unresolved)
    planned = {task.cell_id for task in plan.tasks if task.cell_id}
    registered = {cell.cell_id for cell in load_registry(ROOT).cells}
    assert planned == registered


def test_structural_na_cells_are_never_ready(unresolved: GateState) -> None:
    """C04/C07/C13/C16 are permanently non-executable [AUTH: 00 §36.1]."""
    plan = build_sweep_plan(ROOT, state=unresolved)
    registry = load_registry(ROOT)
    na = {c.cell_id for c in registry.cells if c.status == "STRUCTURAL_NA"}
    assert na
    for task in plan.tasks:
        if task.cell_id in na:
            assert task.readiness != "READY"


def test_both_canary_and_matched_control_arms_are_planned(unresolved: GateState) -> None:
    """00 §7.5 pairs every canary seed with a matched no-canary control."""
    plan = build_sweep_plan(ROOT, state=unresolved)
    for seed in CALIBRATION_SEEDS:
        ids = {task.task_id for task in plan.tasks}
        assert f"train.qwen3_5_4b_base.canary.{seed}" in ids
        assert f"train.qwen3_5_4b_base.control.{seed}" in ids


def test_every_execution_binding_now_exists(unresolved: GateState) -> None:
    """No task names a binding that has not been written [AUTH: 01 §16]."""
    import importlib

    plan = build_sweep_plan(ROOT, state=unresolved)
    assert plan.missing_bindings == ()
    assert plan.by_readiness(MISSING_EXECUTION_BINDING) == ()
    assert plan.ready

    for task in plan.tasks:
        if task.execution_binding.startswith("("):
            continue  # STRUCTURAL_NA carries no binding by design
        module_name, symbol = task.execution_binding.split(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), f"{task.task_id} names a binding that does not exist"


def test_the_production_trainer_is_the_bound_adapter_producer(unresolved: GateState) -> None:
    plan = build_sweep_plan(ROOT, state=unresolved)
    adapters = [t for t in plan.tasks if t.kind == "ADAPTER"]
    assert len(adapters) == 9, "6 canary/control + 3 DP jobs"
    assert {t.execution_binding for t in adapters} == {TRAINING_BINDING}


def test_uncalibrated_constants_block_execution_without_hiding_the_binding(
    unresolved: GateState,
) -> None:
    """A frozen-constant gap is reported as itself, not as missing code.

    While `configs/training/**` still carries REQUIRED_NOT_CALIBRATED values the study cannot
    start, but the reason is a scientific freeze, not absent execution code [AUTH: 01 §17].
    """
    plan = build_sweep_plan(ROOT, state=unresolved)
    blocked = plan.by_readiness(BLOCKED_BY_UNCALIBRATED_CONSTANT)
    unfrozen = uncalibrated_training_constants(ROOT)
    assert (plan.uncalibrated == unfrozen) and (bool(blocked) == bool(unfrozen))
    assert plan.executable is (not unfrozen)
    assert all(task.kind == "ADAPTER" for task in blocked)
    # The distinction the classes exist to preserve.
    assert plan.ready and not plan.executable if unfrozen else plan.executable


def test_the_audit_agrees_with_the_plan(unresolved: GateState) -> None:
    plan = build_sweep_plan(ROOT, state=unresolved)
    audit = readiness_audit(plan)
    assert audit["full_sweep_code_ready"] is plan.ready
    assert audit["missing_execution_bindings"] == list(plan.missing_bindings)


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_study.py"), "--root", str(ROOT), *arguments],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_execute_full_refuses_while_a_constant_is_uncalibrated() -> None:
    """The sweep will not start on defaulted science [AUTH: 01 §17; 00 §8.3]."""
    completed = _cli("execute-full")
    assert completed.returncode == 1
    assert "REQUIRED_NOT_CALIBRATED" in completed.stderr
    assert "Nothing was run" in completed.stderr
    for constant in uncalibrated_training_constants(ROOT):
        assert constant in completed.stderr, "the refusal names every unfrozen constant"


def test_plan_full_is_side_effect_free() -> None:
    """PLAN writes nothing and repeats byte-identically."""
    first = _cli("plan-full")
    second = _cli("plan-full")
    assert first.returncode == 0
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["n_tasks"] > 0


def test_the_cli_refuses_a_scientific_override() -> None:
    """A dimension typed at the command line would define a new experiment [AUTH: 00 §36]."""
    completed = _cli("plan-full", "--alpha", "0.9")
    assert completed.returncode == 1
    assert "unrecognised argument" in completed.stderr
