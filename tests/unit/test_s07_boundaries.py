"""The S07 stage boundary: what the operator layer may and may not do or depend on."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from s07_fixtures import context

from src.merge.operators import (
    DARE_GATE_STATUS,
    DEFERRED_OPERATORS,
    LINEAR,
    O3_GATE_STATUS,
    O3_STATUS,
    OPERATORS,
    operator_status,
)
from src.merge.updates import TaskVectorError, fixture_induced_update

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGE_SOURCES = sorted((REPO_ROOT / "src" / "merge").glob("*.py"))


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


# ------------------------------------------------------------------ no real dependencies
@pytest.mark.parametrize("path", MERGE_SOURCES, ids=lambda p: p.name)
def test_no_merge_module_reaches_a_model_dataset_or_network(path: Path) -> None:
    forbidden = {
        "torch",
        "transformers",
        "peft",
        "opacus",
        "requests",
        "urllib",
        "urllib.request",
        "httpx",
        "socket",
        "huggingface_hub",
    }
    assert not imports(path) & forbidden, path.name


@pytest.mark.parametrize("path", MERGE_SOURCES, ids=lambda p: p.name)
def test_the_operator_layer_does_not_depend_on_data_or_training_stages(path: Path) -> None:
    """S07 is weight-space engineering; it reads no corpus and runs no trainer."""
    reached = {module for module in imports(path) if module.startswith("src.")}
    # src.merge.updates reaches src.training only inside verify_adapter_source, to walk the
    # accepted S06 proof chain. That is a provenance check, not a training dependency.
    allowed = {"src.training.manifests", "src.training.trainer"}
    offending = {
        m for m in reached if m.startswith(("src.data", "src.dp", "src.training"))
    } - allowed
    assert not offending, (path.name, sorted(reached))


@pytest.mark.parametrize("path", MERGE_SOURCES, ids=lambda p: p.name)
def test_no_merge_module_reaches_the_scoring_or_recovery_stages(path: Path) -> None:
    """No privacy scoring at S07, and S08 owns recovery [AUTH: 01 §39]."""
    reached = {module for module in imports(path) if module.startswith("src.")}
    assert not {m for m in reached if m.startswith(("src.scoring", "src.recovery", "src.attacks"))}


def test_no_recovery_solver_was_implemented() -> None:
    """B18: S08 owns C1 inversion, spectral de-tuning, conditioning and reconstructed A-hat."""
    banned = (
        "oracle_invert",
        "spectral_detuning",
        "recover_constituent",
        "reconstruct_a",
        "conditioning_number",
        "solve_for_a",
    )
    for path in MERGE_SOURCES:
        source = path.read_text(encoding="utf-8").lower()
        for name in banned:
            assert f"def {name}" not in source, (path.name, name)
    assert not (REPO_ROOT / "src" / "recovery" / "solvers.py").exists()


# ------------------------------------------------------------------ operator vocabulary
def test_the_declared_operators_are_exactly_the_three_frozen_ones() -> None:
    assert OPERATORS == (
        "O1_LINEAR_TASK_ARITHMETIC",
        "O2_DARE",
        "O3_SVD_TRUNC_MERGE",
    )
    assert "TIES" in DEFERRED_OPERATORS


def test_no_module_defines_a_ties_implementation() -> None:
    for path in MERGE_SOURCES:
        source = path.read_text(encoding="utf-8")
        assert "def ties" not in source.lower()
        assert "trim_elect" not in source.lower()


def test_the_status_surface_asserts_nothing_empirical() -> None:
    status = operator_status(context())
    assert status["O3_STATUS"] == O3_STATUS == "PRE_REGISTERED_CONDITIONAL"
    assert status["DARE_GATE_STATUS"] == DARE_GATE_STATUS == "NOT_RUN"
    assert status["O3_GATE_STATUS"] == O3_GATE_STATUS == "NOT_RUN"
    assert "PRIMARY" not in str(status).upper().replace("PRIMARY_LOSSY", "")


def test_no_merge_module_names_a_primary_lossy_operator() -> None:
    """B13: 00 §24A.7's P1_PRIMARY_LOSSY_OPERATOR is set after the gates, not here."""
    for path in MERGE_SOURCES:
        source = path.read_text(encoding="utf-8")
        assert "P1_PRIMARY_LOSSY_OPERATOR =" not in source
        assert "PRIMARY_LOSSY_OPERATOR:" not in source


# ------------------------------------------------------------------ induced updates only
def test_an_adapter_factor_mapping_cannot_be_merged() -> None:
    """B17 item 9: (A, B) factor pairs are not a parameter object."""
    with pytest.raises(TaskVectorError):
        factors: Any = (np.ones((2, 3)), np.ones((3, 2)))
        fixture_induced_update({"model.layers.0.mlp.gate_proj.weight": factors}, context=context())


def test_the_linear_operator_name_states_its_family() -> None:
    assert LINEAR.startswith("O1_") and "TASK_ARITHMETIC" in LINEAR
