"""C1-C36 — the registry is authoritative data, and every row matches 00 §36 exactly.

The registry is transcribed from the spec, so these tests compare it against the spec's own
table rather than against a summary of it. A row that drifted would otherwise be invisible:
the launcher would happily resolve it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from s09_fixtures import gate_state, unresolved_state

from src.experiments.gates import (
    DARE,
    OPERATOR_AXIS_INSUFFICIENT,
    SVD_TRUNC_MERGE,
    UNRESOLVED,
    GateError,
    GateState,
    OperatorFrozenError,
    fixture_gate_state,
    load_gate_state,
    require_no_operator_reversal,
    resolve_primary_lossy_operator,
)
from src.experiments.registry import (
    CALIBRATION_SEEDS,
    DP_INFERENTIAL_SEEDS,
    EXPECTED_CALIBRATION_ROWS,
    EXPECTED_DP_ROWS,
    EXPECTED_STRUCTURAL_NA,
    EXPECTED_TOTAL_ROWS,
    REGISTERED_K,
    REGISTRY_STATUSES,
    Cell,
    RegistryError,
    UnregisteredCellError,
    load_registry,
    refuse_scientific_override,
)
from src.experiments.resolution import dp_seed_set_is_inferential, resolve_cell
from src.experiments.schedules import (
    FIXED_ALPHA,
    PRE_RESULT_DESIGN_COMPLETION,
    REGISTERED_SCHEDULES,
    ScheduleError,
    resolve_schedule,
    schedule_for_cell,
)
from src.experiments.settings import registry_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = load_registry(REPO_ROOT)

#: 00 §36.1, transcribed independently of the config so drift in either shows up here.
SPEC_CALIBRATION: tuple[tuple[str, str, int, str, str | None, str], ...] = (
    ("C01", "LINEAR", 1, "Z-HIGH", "fixed", "RUN_CONTROL"),
    ("C02", "LINEAR", 2, "Z-HIGH", "fixed", "RUN_CONTROL"),
    ("C03", "LINEAR", 4, "Z-HIGH", "fixed", "RUN_CONTROL"),
    ("C04", "LINEAR", 1, "Z-INCOMPLETE-FIXED", "fixed", "STRUCTURAL_NA"),
    ("C05", "LINEAR", 2, "Z-INCOMPLETE-FIXED", "fixed", "RUN_REPRODUCTION"),
    ("C06", "LINEAR", 4, "Z-INCOMPLETE-FIXED", "fixed", "RUN_REPRODUCTION"),
    ("C07", "LINEAR", 1, "Z-INCOMPLETE-VARYING-KNOWN", None, "STRUCTURAL_NA"),
    ("C08", "LINEAR", 2, "Z-INCOMPLETE-VARYING-KNOWN", "k2-reference", "RUN_REPRODUCTION"),
    ("C09M", "LINEAR", 4, "Z-INCOMPLETE-VARYING-KNOWN", "spread-matched", "RUN_REPRODUCTION"),
    ("C09W", "LINEAR", 4, "Z-INCOMPLETE-VARYING-KNOWN", "wide-spread", "RUN_REPRODUCTION"),
    ("C10", "LOSSY", 1, "Z-HIGH", "fixed", "OPERATOR_GATED"),
    ("C11", "LOSSY", 2, "Z-HIGH", "fixed", "OPERATOR_GATED"),
    ("C12", "LOSSY", 4, "Z-HIGH", "fixed", "OPERATOR_GATED"),
    ("C13", "LOSSY", 1, "Z-INCOMPLETE-FIXED", "fixed", "STRUCTURAL_NA"),
    ("C14", "LOSSY", 2, "Z-INCOMPLETE-FIXED", "fixed", "CONDITIONAL"),
    ("C15", "LOSSY", 4, "Z-INCOMPLETE-FIXED", "fixed", "CONDITIONAL"),
    ("C16", "LOSSY", 1, "Z-INCOMPLETE-VARYING-KNOWN", None, "STRUCTURAL_NA"),
    ("C17", "LOSSY", 2, "Z-INCOMPLETE-VARYING-KNOWN", "k2-reference", "CONDITIONAL"),
    ("C18", "LOSSY", 4, "Z-INCOMPLETE-VARYING-KNOWN", "spread-matched", "CONDITIONAL"),
)

#: 00 §36.2. Every row is DP_SMOKE_ONLY before the 3-seed set.
SPEC_DP: tuple[tuple[str, str, int, str, str, str], ...] = (
    ("D01", "LINEAR", 1, "Z-HIGH", "fixed", "RUN_CONTROL"),
    ("D02", "LINEAR", 2, "Z-HIGH", "fixed", "RUN_CONTROL"),
    ("D03", "LINEAR", 4, "Z-HIGH", "fixed", "RUN_CONTROL"),
    ("D04", "LINEAR", 2, "Z-INCOMPLETE-FIXED", "fixed", "RUN_REPRODUCTION"),
    ("D05", "LINEAR", 4, "Z-INCOMPLETE-FIXED", "fixed", "RUN_REPRODUCTION"),
    ("D06", "LINEAR", 2, "Z-INCOMPLETE-VARYING-KNOWN", "k2-reference", "RUN_REPRODUCTION"),
    ("D07", "LINEAR", 4, "Z-INCOMPLETE-VARYING-KNOWN", "spread-matched", "RUN_REPRODUCTION"),
    ("D08", "LOSSY", 1, "Z-HIGH", "fixed", "OPERATOR_GATED"),
    ("D09", "LOSSY", 2, "Z-HIGH", "fixed", "OPERATOR_GATED"),
    ("D10", "LOSSY", 4, "Z-HIGH", "fixed", "OPERATOR_GATED"),
)


# ==================================================================== C1-C13 structure
def test_c1_c2_c3_the_registry_has_exactly_nineteen_ten_and_twenty_nine_rows() -> None:
    assert len(REGISTRY.calibration()) == EXPECTED_CALIBRATION_ROWS == 19
    assert len(REGISTRY.representative_dp()) == EXPECTED_DP_ROWS == 10
    assert len(REGISTRY.cells) == EXPECTED_TOTAL_ROWS == 29


def test_c4_exactly_four_calibration_rows_are_structural_na() -> None:
    na = [c.cell_id for c in REGISTRY.calibration() if c.status == "STRUCTURAL_NA"]
    assert na == ["C04", "C07", "C13", "C16"]
    assert len(na) == EXPECTED_STRUCTURAL_NA == 4


def test_c5_cell_ids_are_unique() -> None:
    ids = list(REGISTRY.cell_ids)
    assert len(ids) == len(set(ids)) == 29


def test_c6_the_registry_hash_is_deterministic() -> None:
    assert registry_sha256(REPO_ROOT) == registry_sha256(REPO_ROOT)
    assert len(registry_sha256(REPO_ROOT)) == 64
    assert REGISTRY.identity() == registry_sha256(REPO_ROOT)


def test_c7_cell_definition_hashes_are_deterministic_and_distinct() -> None:
    first = {c.cell_id: c.identity() for c in REGISTRY.cells}
    second = {c.cell_id: c.identity() for c in load_registry(REPO_ROOT).cells}
    assert first == second
    assert len(set(first.values())) == 29, "no two rows may share a definition hash"


def test_c8_only_the_closed_status_vocabulary_appears() -> None:
    assert REGISTRY_STATUSES == frozenset(
        {
            "RUN_CONTROL",
            "RUN_REPRODUCTION",
            "RUN_MEASUREMENT",
            "CONDITIONAL",
            "METHOD_GATED",
            "OPERATOR_GATED",
            "STRUCTURAL_NA",
            "DP_SMOKE_ONLY",
        }
    )
    for cell in REGISTRY.cells:
        assert cell.status in REGISTRY_STATUSES
        if cell.status_after_dp_seed_set is not None:
            assert cell.status_after_dp_seed_set in REGISTRY_STATUSES


@pytest.mark.parametrize("generic", ["ACTIVE", "ENABLED", "TODO", "OPTIONAL", "AUTO"])
def test_a_generic_status_is_refused(tmp_path: Path, generic: str) -> None:
    root = _rewritten_registry(tmp_path, lambda d: _set_status(d, "C01", generic))
    with pytest.raises(RegistryError, match="closed 00 §36 vocabulary"):
        load_registry(root)


def test_c9_only_k_one_two_and_four_appear() -> None:
    assert REGISTERED_K == frozenset({1, 2, 4})
    assert {cell.k for cell in REGISTRY.cells} == {1, 2, 4}


@pytest.mark.parametrize(
    ("marker", "label"),
    [
        ("Z-INCOMPLETE-ALPHA-HIDDEN", "hidden-alpha lineage"),
        ("TIES", "TIES operator"),
        ("QUANTIZATION", "quantization operator"),
        ("SLERP", "SLERP operator"),
    ],
)
def test_c10_c12_c13_forbidden_axes_are_absent(marker: str, label: str) -> None:
    rendered = json.dumps([cell.as_dict() for cell in REGISTRY.cells])
    assert marker not in rendered, f"{label} must not appear in the registry"


def test_c11_k_eight_is_absent() -> None:
    assert 8 not in {cell.k for cell in REGISTRY.cells}
    assert '"k": 8' not in json.dumps([c.as_dict() for c in REGISTRY.cells])


# ==================================================================== C14-C17 exact rows
def test_c14_every_calibration_row_matches_00_36_1_exactly() -> None:
    actual = tuple(
        (c.cell_id, c.operator, c.k, c.lineage, c.schedule, c.status)
        for c in REGISTRY.calibration()
    )
    assert actual == SPEC_CALIBRATION


def test_c15_every_dp_row_matches_00_36_2_exactly() -> None:
    actual = tuple(
        (c.cell_id, c.operator, c.k, c.lineage, c.schedule, c.status_after_dp_seed_set)
        for c in REGISTRY.representative_dp()
    )
    assert actual == SPEC_DP
    assert all(c.status == "DP_SMOKE_ONLY" for c in REGISTRY.representative_dp())


def test_c16_there_is_no_lossy_wide_spread_duplicate_of_c09w() -> None:
    """00 §36.1: C09W is a linear stress condition and is not duplicated under LOSSY."""
    wide = [c for c in REGISTRY.cells if c.schedule == "wide-spread"]
    assert [c.cell_id for c in wide] == ["C09W"]
    assert wide[0].operator == "LINEAR"


def test_c17_there_are_no_incomplete_lineage_lossy_dp_rows() -> None:
    """00 §36.2: they are not silently added to the 10-row DP registry."""
    intruders = [
        c.cell_id
        for c in REGISTRY.representative_dp()
        if c.operator == "LOSSY" and c.lineage != "Z-HIGH"
    ]
    assert intruders == []


# ==================================================================== C18-C22 schedules
def test_c18_c19_c20_the_three_schedules_carry_their_exact_vectors() -> None:
    assert resolve_schedule(REPO_ROOT, "k2-reference").alpha == (0.35, 0.65)
    assert resolve_schedule(REPO_ROOT, "spread-matched").alpha == (0.35, 0.45, 0.55, 0.65)
    assert resolve_schedule(REPO_ROOT, "wide-spread").alpha == (0.25, 0.4, 0.6, 0.75)


def test_c21_every_schedule_length_equals_its_k() -> None:
    for name in ("k2-reference", "spread-matched", "wide-spread"):
        schedule = resolve_schedule(REPO_ROOT, name)
        assert len(schedule.alpha) == schedule.k
        assert all(0.0 < a <= 1.0 for a in schedule.alpha)


def test_a_row_resolves_to_exactly_its_registered_schedule() -> None:
    for cell in REGISTRY.cells:
        if cell.schedule in (None, "fixed"):
            continue
        schedule = schedule_for_cell(REPO_ROOT, cell)
        assert schedule is not None
        assert schedule.name == cell.schedule
        assert schedule.k == cell.k


def test_c22_no_caller_alpha_vector_can_override_a_row() -> None:
    """00 §36.3: the row names a schedule; there is no parameter that supplies coefficients."""
    import inspect

    for function in (resolve_schedule, schedule_for_cell):
        parameters = set(inspect.signature(function).parameters)
        assert "alpha" not in parameters and "coefficients" not in parameters
    with pytest.raises(RegistryError, match="comes from the registry row"):
        refuse_scientific_override(alpha=[0.2, 0.8])


def test_the_fixed_schedule_is_a_pre_result_design_completion() -> None:
    """00 §36 registers the lineage without a coefficient; 0.50 is completed here, pre-result.

    The config records that provenance explicitly, so nobody later reads 0.50 as a number the
    closed measurement spec assigned.
    """
    assert "fixed" in REGISTERED_SCHEDULES
    assert FIXED_ALPHA == 0.5
    entry = json.loads(
        (REPO_ROOT / "configs" / "experiments" / "schedules.json").read_text(encoding="utf-8")
    )["fixed"]
    assert entry["status"] == "FROZEN"
    assert entry["value"]["alpha"] == 0.5
    assert entry["value"]["provenance"] == PRE_RESULT_DESIGN_COMPLETION
    assert "not a value from the closed measurement spec" in entry["note"]


@pytest.mark.parametrize(
    ("cell_id", "k"), [("C01", 1), ("C02", 2), ("C03", 4), ("C05", 2), ("C06", 4)]
)
def test_a_fixed_cell_resolves_to_alpha_one_half_for_every_descendant(cell_id: str, k: int) -> None:
    schedule = schedule_for_cell(REPO_ROOT, REGISTRY.cell(cell_id))
    assert schedule is not None
    assert schedule.name == "fixed"
    assert schedule.k == k
    assert schedule.alpha == (0.5,) * k
    assert schedule.provenance == PRE_RESULT_DESIGN_COMPLETION


def test_the_fixed_schedule_needs_a_cell_k_to_expand() -> None:
    """It registers one scalar, not a vector, so it never guesses a descendant count."""
    with pytest.raises(ScheduleError, match="did not supply"):
        resolve_schedule(REPO_ROOT, "fixed")


def test_a_varying_schedule_refuses_a_k_it_does_not_register() -> None:
    with pytest.raises(ScheduleError, match="registers k = 2, not 4"):
        resolve_schedule(REPO_ROOT, "k2-reference", k=4)


def test_changing_the_fixed_alpha_moves_the_registry_identity(tmp_path: Path) -> None:
    """Registry, schedule and resolved-config identity all follow the coefficient."""
    import shutil

    from s09_fixtures import unresolved_state

    from src.experiments.launcher import plan_cell
    from src.provenance.config import resolved_config_sha256

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs", dirs_exist_ok=True)

    def plan_sha(where: Path) -> tuple[str, str, str]:
        plan = plan_cell(
            where, "C01", model_alias="llama_3_2_3b", seed=101, state=unresolved_state()
        )
        return (
            registry_sha256(where),
            plan.identity(),
            resolved_config_sha256(plan.resolved_config()),
        )

    before = plan_sha(root)
    path = root / "configs" / "experiments" / "schedules.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fixed"]["value"]["alpha"] = 0.6
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    after = plan_sha(root)

    assert after[0] != before[0], "the registry identity must move"
    assert after[1] != before[1], "the plan identity must move"
    assert after[2] != before[2], "the resolved-config identity must move"


def test_an_unregistered_schedule_is_refused() -> None:
    with pytest.raises(ScheduleError, match="not a registered schedule"):
        resolve_schedule(REPO_ROOT, "k8-wide")


# ==================================================================== C23-C32 gating
@pytest.mark.parametrize("cell_id", ["C04", "C07", "C13", "C16"])
def test_c23_structural_na_cells_are_never_runnable(cell_id: str) -> None:
    """No gate state, and no flag, makes a structural N/A cell executable."""
    for state in (
        unresolved_state(),
        gate_state(P0_D="PASS", P0_C2="PASS", P0_C4="PASS"),
        gate_state(P0_D="FAIL", P0_D2="PASS", P0_C2="PASS", P0_C4="PASS"),
    ):
        resolved = resolve_cell(
            REGISTRY.cell(cell_id), state, dp_seeds=CALIBRATION_SEEDS, schedule_resolved=True
        )
        assert resolved.runnable is False
        assert resolved.resolved_status == "STRUCTURAL_NA"


def test_no_allow_na_flag_exists() -> None:
    import inspect

    from src.experiments import launcher, resolution

    for module in (launcher, resolution):
        source = inspect.getsource(module)
        assert "allow_na" not in source and "allow-na" not in source


@pytest.mark.parametrize("cell_id", ["C08", "C09M", "C09W"])
def test_c24_varying_alpha_cells_are_gated_until_p0_c2(cell_id: str) -> None:
    before = resolve_cell(REGISTRY.cell(cell_id), unresolved_state(), schedule_resolved=True)
    assert before.runnable is False
    assert before.reason == "VARYING_ALPHA_AWAITS_P0_C2"

    after = resolve_cell(REGISTRY.cell(cell_id), gate_state(P0_C2="PASS"), schedule_resolved=True)
    assert after.runnable is True
    assert after.resolved_status == "RUN_REPRODUCTION"


def test_c25_dare_pass_selects_dare() -> None:
    assert resolve_primary_lossy_operator(gate_state(P0_D="PASS")) == DARE


def test_c26_dare_fail_and_o3_pass_selects_svd_trunc_merge() -> None:
    state = gate_state(P0_D="FAIL", P0_D2="PASS")
    assert resolve_primary_lossy_operator(state) == SVD_TRUNC_MERGE


def test_c27_both_gates_failing_is_operator_axis_insufficient() -> None:
    state = gate_state(P0_D="FAIL", P0_D2="FAIL")
    assert resolve_primary_lossy_operator(state) == OPERATOR_AXIS_INSUFFICIENT


def test_c28_an_unresolved_gate_leaves_every_lossy_cell_ungated() -> None:
    assert resolve_primary_lossy_operator(unresolved_state()) == UNRESOLVED
    for cell in REGISTRY.cells:
        if cell.operator != "LOSSY" or cell.status == "STRUCTURAL_NA":
            continue
        resolved = resolve_cell(
            cell, unresolved_state(), dp_seeds=CALIBRATION_SEEDS, schedule_resolved=True
        )
        assert resolved.runnable is False, cell.cell_id


@pytest.mark.parametrize("cell_id", ["C14", "C15", "C17", "C18"])
def test_c29_dare_selection_method_gates_the_conditional_cells(cell_id: str) -> None:
    state = gate_state(P0_D="PASS", P0_C2="PASS")
    resolved = resolve_cell(REGISTRY.cell(cell_id), state, schedule_resolved=True)
    assert resolved.resolved_status == "METHOD_GATED"
    assert resolved.runnable is False
    assert resolved.resolved_operator == DARE


@pytest.mark.parametrize("cell_id", ["C14", "C15", "C17", "C18"])
@pytest.mark.parametrize("c4", ["FAIL", None])
def test_c30_o3_without_a_passing_p0_c4_keeps_the_conditional_cells_gated(
    cell_id: str, c4: str | None
) -> None:
    statuses = {"P0_D": "FAIL", "P0_D2": "PASS", "P0_C2": "PASS"}
    if c4 is not None:
        statuses["P0_C4"] = c4
    resolved = resolve_cell(REGISTRY.cell(cell_id), gate_state(**statuses), schedule_resolved=True)
    assert resolved.runnable is False
    assert resolved.reason == "CONDITIONAL_O3_C4_NOT_PASSED"


@pytest.mark.parametrize("cell_id", ["C14", "C15", "C17", "C18"])
def test_c31_o3_with_a_passing_p0_c4_makes_them_reproductions(cell_id: str) -> None:
    state = gate_state(P0_D="FAIL", P0_D2="PASS", P0_C4="PASS", P0_C2="PASS")
    resolved = resolve_cell(REGISTRY.cell(cell_id), state, schedule_resolved=True)
    assert resolved.resolved_status == "RUN_REPRODUCTION"
    assert resolved.resolved_operator == SVD_TRUNC_MERGE
    assert resolved.runnable is True


def test_c32_a_frozen_primary_operator_cannot_be_replaced() -> None:
    dare = _frozen(DARE, P0_D="PASS")
    o3 = _frozen(SVD_TRUNC_MERGE, P0_D="FAIL", P0_D2="PASS")
    with pytest.raises(OperatorFrozenError, match="already frozen"):
        require_no_operator_reversal(dare, o3)
    with pytest.raises(OperatorFrozenError, match="already frozen"):
        require_no_operator_reversal(o3, dare)
    require_no_operator_reversal(dare, dare)


def test_a_freeze_that_contradicts_its_own_gates_is_refused() -> None:
    with pytest.raises(OperatorFrozenError, match="cannot be replaced|resolve to"):
        _frozen(SVD_TRUNC_MERGE, P0_D="PASS")
    with pytest.raises(OperatorFrozenError, match="resolve to"):
        _frozen(DARE)


def _frozen(operator: str, **statuses: str) -> GateState:
    from s09_fixtures import frozen_operator_state

    return frozen_operator_state(operator, **statuses)


def test_the_operator_resolver_reads_only_the_two_gate_statuses() -> None:
    """No effect size, recovery quality or preference has any representation in the rule.

    Checked on the executable body rather than the docstring, which names those things
    precisely to say they are excluded.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(resolve_primary_lossy_operator).lstrip())
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body[1:] if isinstance(function.body[0], ast.Expr) else function.body
    rendered = "\n".join(ast.unparse(node) for node in body).lower()
    for banned in ("effect", "largest", "best", "prefer", "interesting", "available"):
        assert banned not in rendered
    #: exactly the two gate statuses, and no other input
    reads = {
        ast.unparse(node.args[0])
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "status"
    }
    assert reads == {"'P0-D'", "'P0-D2'"}


# ==================================================================== C33-C36 DP seeds
def test_c33_a_single_seed_stays_smoke_only() -> None:
    resolved = resolve_cell(
        REGISTRY.cell("D01"), gate_state(), dp_seeds=[101], schedule_resolved=True
    )
    assert resolved.resolved_status == "DP_SMOKE_ONLY"
    assert resolved.runnable is False


def test_c34_the_exact_three_seed_set_allows_the_after_status() -> None:
    assert DP_INFERENTIAL_SEEDS == frozenset({101, 202, 303})
    resolved = resolve_cell(
        REGISTRY.cell("D01"), gate_state(), dp_seeds=[101, 202, 303], schedule_resolved=True
    )
    assert resolved.resolved_status == "RUN_CONTROL"
    assert resolved.runnable is True


@pytest.mark.parametrize("seeds", [[101, 202, 404], [101, 202, 505], [404, 505, 606], [101, 202]])
def test_c35_c36_a_wrong_or_incomplete_seed_set_is_not_inferential(seeds: list[int]) -> None:
    """00 §36.2: the exact registered set, never a count of three."""
    assert dp_seed_set_is_inferential(seeds) is False
    resolved = resolve_cell(
        REGISTRY.cell("D04"), gate_state(), dp_seeds=seeds, schedule_resolved=True
    )
    assert resolved.resolved_status == "DP_SMOKE_ONLY"


def test_a_duplicated_seed_set_is_refused() -> None:
    assert dp_seed_set_is_inferential([101, 101, 202]) is False
    assert dp_seed_set_is_inferential([101, 202, 303, 303]) is False


def test_the_confirmatory_seeds_are_not_registry_seeds() -> None:
    assert CALIBRATION_SEEDS == (101, 202, 303)
    assert 404 not in CALIBRATION_SEEDS and 505 not in CALIBRATION_SEEDS
    assert 404 not in DP_INFERENTIAL_SEEDS and 505 not in DP_INFERENTIAL_SEEDS


# ==================================================================== C37-C43 no invention
def test_c37_an_unknown_cell_id_is_refused() -> None:
    for unknown in ("C99", "C09X", "D11", "", "c05", "C20"):
        with pytest.raises(UnregisteredCellError, match="not a registered P1 cell"):
            REGISTRY.cell(unknown)


@pytest.mark.parametrize(
    "override",
    [
        {"k": 4},
        {"lineage": "Z-HIGH"},
        {"operator": "LOSSY"},
        {"alpha": [0.2, 0.8]},
        {"recovery_method": "C2B"},
        {"schedule": "wide-spread"},
    ],
)
def test_c38_to_c42_no_scientific_dimension_may_be_supplied(override: dict[str, object]) -> None:
    with pytest.raises(RegistryError, match="comes from the registry row"):
        refuse_scientific_override(**override)


def test_c43_a_cell_cannot_be_constructed_from_a_free_form_combination() -> None:
    """`Cell` is registry-issued; a constructible row would be a CLI-defined experiment."""
    import dataclasses

    with pytest.raises(TypeError, match="registry-issued"):
        Cell()
    assert not dataclasses.is_dataclass(Cell)
    cell = REGISTRY.cell("C05")
    with pytest.raises(AttributeError):
        cell.__setattr__("_k", 4)
    with pytest.raises(TypeError):
        dataclasses.replace(cell)  # type: ignore[type-var]


def test_there_is_no_add_cell_api() -> None:
    """00 §36: adding a row is not a launch-time operation."""
    import inspect

    from src.experiments import launcher, registry

    for module in (registry, launcher):
        source = inspect.getsource(module)
        for banned in ("def add_cell", "def register_cell", "def new_cell", "auto_expand"):
            assert banned not in source, (module.__name__, banned)


def test_a_registry_missing_a_row_is_refused(tmp_path: Path) -> None:
    root = _rewritten_registry(tmp_path, lambda d: d["calibration"].pop())
    with pytest.raises(RegistryError, match="18 rows, not 19"):
        load_registry(root)


def test_a_registry_with_an_extra_row_is_refused(tmp_path: Path) -> None:
    def add(document: dict) -> None:  # type: ignore[type-arg]
        extra = dict(document["calibration"][0])
        extra["cell_id"] = "C19"
        document["calibration"].append(extra)

    root = _rewritten_registry(tmp_path, add)
    with pytest.raises(RegistryError, match="20 rows, not 19"):
        load_registry(root)


def test_a_hidden_alpha_row_is_refused(tmp_path: Path) -> None:
    def corrupt(document: dict) -> None:  # type: ignore[type-arg]
        document["calibration"][0]["lineage"] = "Z-INCOMPLETE-ALPHA-HIDDEN"

    root = _rewritten_registry(tmp_path, corrupt)
    with pytest.raises(RegistryError, match="not part of P1"):
        load_registry(root)


def test_a_k8_row_is_refused(tmp_path: Path) -> None:
    def corrupt(document: dict) -> None:  # type: ignore[type-arg]
        document["calibration"][0]["k"] = 8

    root = _rewritten_registry(tmp_path, corrupt)
    with pytest.raises(RegistryError, match=r"k 8 is not in \[1, 2, 4\]"):
        load_registry(root)


def test_a_concrete_operator_in_a_row_is_refused(tmp_path: Path) -> None:
    """A row names LINEAR or the abstract LOSSY; the concrete operator is gate-resolved."""

    def corrupt(document: dict) -> None:  # type: ignore[type-arg]
        document["calibration"][10]["operator"] = "DARE"

    root = _rewritten_registry(tmp_path, corrupt)
    with pytest.raises(RegistryError, match="not a registered axis value"):
        load_registry(root)


# ==================================================================== gate documents
def test_a_gate_document_must_declare_whether_it_is_evidence() -> None:
    with pytest.raises(GateError, match="provenance_class"):
        load_gate_state({"gates": {}}, root=REPO_ROOT)


def test_a_handwritten_gate_verdict_is_not_authority() -> None:
    """`{"DARE": "PASS"}` does not become evidentiary merely because it parses."""
    with pytest.raises(GateError):
        load_gate_state(
            {"provenance_class": "EVIDENTIARY_P0_GATE", "gates": {"P0-D": "PASS"}},
            root=REPO_ROOT,
        )


def test_an_evidentiary_gate_record_must_bind_its_source() -> None:
    document = fixture_gate_state(
        statuses={"P0-D": "PASS"}, registry_sha256=registry_sha256(REPO_ROOT)
    )
    document["provenance_class"] = "EVIDENTIARY_P0_GATE"
    with pytest.raises(GateError, match="RUN_ID|source artifact"):
        load_gate_state(document, root=REPO_ROOT)


def test_a_fixture_gate_document_is_marked_non_evidentiary() -> None:
    state = gate_state(P0_D="PASS")
    assert state.provenance_class == "NON_EVIDENTIARY_FIXTURE"
    assert state.evidentiary is False


def test_an_unregistered_gate_name_is_refused() -> None:
    with pytest.raises(GateError, match="not a registered gate"):
        gate_state(P0_ZZ="PASS")


# ==================================================================== helpers
def _set_status(document: dict, cell_id: str, status: str) -> None:  # type: ignore[type-arg]
    for row in document["calibration"]:
        if row["cell_id"] == cell_id:
            row["status"] = status


def _rewritten_registry(tmp_path: Path, mutate) -> Path:  # type: ignore[no-untyped-def]
    """A copied config tree whose P1 registry has been altered."""
    import shutil

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs", dirs_exist_ok=True)
    path = root / "configs" / "experiments" / "p1_registry.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return root
