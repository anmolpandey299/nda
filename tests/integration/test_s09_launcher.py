"""C44-C68 — the launcher, the accepted S01 lifecycle, and the P0 stage machine.

Evidentiary behaviour is exercised in a disposable committed fixture repository, because the
S09 development tree is necessarily dirty and a real evidentiary run must never be produced
from it [AUTH: 01 §35(3), §36].
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from s09_fixtures import (
    FIXTURE_MODEL_ALIAS,
    SEED_SET,
    TRAINING_SEED,
    Workspace,
    build_workspace,
    failing_argv,
    gate_state,
    harmless_argv,
    provenance_for,
    unresolved_state,
)

from src.experiments.gates import SVD_TRUNC_MERGE, resolve_primary_lossy_operator
from src.experiments.launcher import (
    ExecutionRequest,
    LaunchError,
    PreconditionError,
    backend_compatibility_argv,
    execute_task,
    list_cells,
    plan_cell,
    plan_document,
    require_execution_preconditions,
)
from src.experiments.registry import UnregisteredCellError, load_registry
from src.experiments.resolution import ResolutionError
from src.experiments.settings import registry_sha256
from src.experiments.stages import (
    EXPECTED_STAGE_ORDER,
    conditional_cells_active,
    load_stage_registry,
    recovery_pipeline_blocked,
    stage_eligibility,
)
from src.provenance.hashing import read_canonical_json
from src.provenance.identity import experiment_id, run_id
from src.provenance.run_manifest import RUNNING, TERMINAL_STATES
from src.provenance.runs import attempts_of, load_attempt

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "launch_experiment.py"


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return build_workspace(tmp_path / "repo")


def _plan(root: Path, cell_id: str = "C05", **overrides: object):  # type: ignore[no-untyped-def]
    arguments: dict[str, object] = {
        "model_alias": FIXTURE_MODEL_ALIAS,
        "seed": TRAINING_SEED,
        "state": gate_state(P0_C2="PASS"),
    }
    arguments.update(overrides)
    return plan_cell(root, cell_id, **arguments)  # type: ignore[arg-type]


# ==================================================================== C44-C47 plan
def test_c44_plan_creates_no_run_manifest_or_directory(workspace: Workspace) -> None:
    plan = _plan(workspace.root)
    assert plan.cell.cell_id == "C05"
    assert not (workspace.root / "artifacts" / "runs").exists()
    assert not (workspace.root / "manifests" / "runs").exists()
    assert not (workspace.root / "logs").exists()


def test_c45_repeated_plan_is_identical(workspace: Workspace) -> None:
    """No wall clock enters plan identity, so repeating it is bitwise the same."""
    first = plan_document(_plan(workspace.root))
    second = plan_document(_plan(workspace.root))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["plan_sha256"] == second["plan_sha256"]
    rendered = json.dumps(first)
    for stamp in ("wall_clock", "timestamp", "started", "utc"):
        assert stamp not in rendered.lower()


def test_c46_the_registry_sha_is_bound_into_the_resolved_config(workspace: Workspace) -> None:
    plan = _plan(workspace.root)
    config = plan.resolved_config()
    assert config["registry_sha256"] == registry_sha256(workspace.root)
    assert config["cell_id"] == "C05"
    assert config["cell_definition_sha256"] == plan.cell.identity()


def test_c47_changing_the_registry_changes_the_resolved_config_identity(
    workspace: Workspace,
) -> None:
    """No evidentiary run under registry A may be attributed to registry B."""
    from src.provenance.config import resolved_config_sha256

    before = resolved_config_sha256(_plan(workspace.root).resolved_config())

    path = workspace.root / "configs" / "experiments" / "schedules.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schedule_version"]["value"] = "s09.coefficient-schedules.v2"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    after = resolved_config_sha256(_plan(workspace.root).resolved_config())
    assert after != before


# ==================================================================== C48-C51 provenance
def test_c48_c49_the_accepted_s01_identity_apis_are_reused(workspace: Workspace) -> None:
    """One RUN_ID algorithm. A second would mean two answers to 'which run is this'."""
    import inspect

    from src.experiments import launcher

    source = inspect.getsource(launcher)
    assert "begin_run" in source
    for banned in (
        "S09RunManifest",
        "S09RunID",
        "def run_id",
        "def experiment_id",
        "attempt_counter",
    ):
        assert banned not in source, banned

    plan = _plan(workspace.root, "C08")
    provenance = provenance_for(workspace, plan.resolved_config())
    result = execute_task(
        workspace.root,
        plan,
        ExecutionRequest(task_id="fixture-echo", argv=harmless_argv()),
        provenance=provenance,
        seeds=SEED_SET,
        precision="float64",
        tokenizer_hash="9" * 64,
    )
    expected_experiment = experiment_id(provenance)
    assert result["experiment_id"] == expected_experiment
    assert result["run_id"] == run_id(expected_experiment, 1)


def test_c50_c51_an_external_run_id_flows_into_the_backend_evidence(
    workspace: Workspace, tmp_path: Path
) -> None:
    """S09 supplies the RUN_ID the backend evidence layer requires and never mints."""
    from backend_fixtures import build_cli_fixture_root, write_checkpoint_snapshot

    from src.backend.evidence import RunBinding, verify_contract_evidence

    plan = _plan(workspace.root, "C08")
    provenance = provenance_for(workspace, plan.resolved_config())
    issued = run_id(experiment_id(provenance), 1)

    snapshot, digest = write_checkpoint_snapshot(tmp_path / "snapshot")
    backend_root = build_cli_fixture_root(tmp_path / "backend", config_sha256=digest)
    argv = backend_compatibility_argv(
        alias="tiny_fixture",
        run_identifier=issued,
        evidence_path="evidence.json",
        extra=("--root", str(backend_root), "--checkpoint-config", str(snapshot)),
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, *argv[1:]], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 1, completed.stderr

    document = json.loads((backend_root / "evidence.json").read_text(encoding="utf-8"))
    assert document["run_id"] == issued, "the evidence binds the S09-issued RUN_ID exactly"

    fields = (
        "run_id",
        "git_commit",
        "environment_lock_sha256",
        "model_manifest_sha256",
        "model_revision",
        "backend_code_sha256",
        "scoring_code_sha256",
        "resolved_config_sha256",
    )
    binding = RunBinding(**{k: document[k] for k in fields})
    assert verify_contract_evidence(document, binding=binding, root=backend_root) == []

    tampered = {**document, "run_id": "0" * 64}
    problems = verify_contract_evidence(tampered, binding=binding, root=backend_root)
    assert any("run_id" in p for p in problems)


# ==================================================================== C52-C62 launcher
def test_c52_c53_c55_c56_a_fixture_execution_claims_one_attempt_and_succeeds(
    workspace: Workspace,
) -> None:
    plan = _plan(workspace.root, "C08")
    provenance = provenance_for(workspace, plan.resolved_config())
    result = execute_task(
        workspace.root,
        plan,
        ExecutionRequest(task_id="fixture-echo", argv=harmless_argv("hello-s09")),
        provenance=provenance,
        seeds=SEED_SET,
        precision="float64",
        tokenizer_hash="9" * 64,
    )
    assert result["status"] == "SUCCESS"
    assert result["exit_code"] == 0
    assert result["attempt"] == 1

    attempt = load_attempt(workspace.root, str(result["run_id"]))
    manifest = attempt.read_manifest()
    assert manifest["status"] == "SUCCESS"
    assert manifest["stdout_log_path"] == attempt.stdout_path
    assert manifest["stderr_log_path"] == attempt.stderr_path
    assert "hello-s09" in (workspace.root / attempt.stdout_path).read_text(encoding="utf-8")

    hashes = manifest["artifact_hashes"]
    assert isinstance(hashes, dict)
    assert attempt.stdout_path in hashes and attempt.stderr_path in hashes
    assert attempt.verify() == []
    assert len(attempts_of(workspace.root, attempt.experiment_id)) == 1


def test_c54_a_failing_fixture_finalises_an_accepted_failure_status(
    workspace: Workspace,
) -> None:
    plan = _plan(workspace.root, "C08")
    provenance = provenance_for(workspace, plan.resolved_config())
    result = execute_task(
        workspace.root,
        plan,
        ExecutionRequest(task_id="fixture-fail", argv=failing_argv()),
        provenance=provenance,
        seeds=SEED_SET,
        precision="float64",
        tokenizer_hash="9" * 64,
    )
    assert result["status"] == "FAILED_IMPLEMENTATION"
    assert result["status"] in TERMINAL_STATES
    assert result["exit_code"] == 3
    attempt = load_attempt(workspace.root, str(result["run_id"]))
    assert "planned failure" in (workspace.root / attempt.stderr_path).read_text(encoding="utf-8")


def test_c57_c58_a_rerun_gets_a_new_run_id_and_the_failed_attempt_is_preserved(
    workspace: Workspace,
) -> None:
    plan = _plan(workspace.root, "C08")
    provenance = provenance_for(workspace, plan.resolved_config())
    common = {
        "provenance": provenance,
        "seeds": SEED_SET,
        "precision": "float64",
        "tokenizer_hash": "9" * 64,
    }
    first = execute_task(
        workspace.root,
        plan,
        ExecutionRequest(task_id="fixture-fail", argv=failing_argv()),
        **common,  # type: ignore[arg-type]
    )
    before = load_attempt(workspace.root, str(first["run_id"])).read_manifest()

    second = execute_task(
        workspace.root,
        plan,
        ExecutionRequest(task_id="fixture-echo", argv=harmless_argv()),
        **common,  # type: ignore[arg-type]
    )
    assert second["run_id"] != first["run_id"]
    assert second["attempt"] == 2 and first["attempt"] == 1
    assert second["experiment_id"] == first["experiment_id"]
    assert second["status"] == "SUCCESS"

    after = load_attempt(workspace.root, str(first["run_id"])).read_manifest()
    assert after == before, "a failed attempt is never rewritten"
    assert after["status"] == "FAILED_IMPLEMENTATION"


def test_c59_an_unregistered_or_non_runnable_cell_never_reaches_begin_run(
    workspace: Workspace,
) -> None:
    registry = load_registry(workspace.root)
    with pytest.raises(UnregisteredCellError):
        plan_cell(
            workspace.root,
            "C99",
            model_alias=FIXTURE_MODEL_ALIAS,
            seed=TRAINING_SEED,
            state=unresolved_state(),
            registry=registry,
        )
    blocked = _plan(workspace.root, "C04")
    with pytest.raises(ResolutionError, match="STRUCTURAL_NA"):
        execute_task(
            workspace.root,
            blocked,
            ExecutionRequest(task_id="fixture-echo", argv=harmless_argv()),
            provenance=provenance_for(workspace, blocked.resolved_config()),
            seeds=SEED_SET,
            precision="float64",
            tokenizer_hash="9" * 64,
        )
    assert not (workspace.root / "artifacts" / "runs").exists()


def test_c60_a_dirty_evidentiary_tree_is_refused(workspace: Workspace) -> None:
    (workspace.root / "configs" / "experiments" / "p1_registry.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(PreconditionError, match="dirty"):
        require_execution_preconditions(workspace.root, workspace.provenance, evidentiary=True)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_revision", "main", "floating branch"),
        ("model_revision", "", "model_revision is empty"),
        ("environment_lock_sha256", "TBD_REQUIRES_HARDWARE", "TBD_REQUIRES_HARDWARE"),
        ("data_manifest_sha256", "UNKNOWN", "not a SHA256"),
        ("config_sha256", "0", "not a SHA256"),
    ],
)
def test_c61_c62_unresolved_provenance_refuses_an_evidentiary_run(
    workspace: Workspace, field: str, value: str, match: str
) -> None:
    """Nothing is filled in with zeros, UNKNOWN, latest or a fixture value."""
    from dataclasses import replace
    from typing import Any

    swap: Any = replace
    provenance = swap(workspace.provenance, **{field: value})
    with pytest.raises(PreconditionError, match=match):
        require_execution_preconditions(workspace.root, provenance, evidentiary=True)


def test_the_launcher_runs_an_argument_array_and_never_a_shell(workspace: Workspace) -> None:
    import inspect

    from src.experiments import launcher

    source = inspect.getsource(launcher)
    assert "shell=True" not in source
    assert "os.system" not in source
    with pytest.raises(LaunchError, match="needs a command"):
        ExecutionRequest(task_id="empty", argv=())


def test_one_invocation_is_one_attempt(workspace: Workspace) -> None:
    """No batch scheduler, no next cell, no DAG traversal."""
    import ast
    import inspect

    from src.experiments import launcher

    tree = ast.parse(inspect.getsource(launcher.execute_task).lstrip())
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body[1:] if isinstance(function.body[0], ast.Expr) else function.body
    rendered = "\n".join(ast.unparse(node) for node in body)

    calls = [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("begin_run") == 1, "exactly one attempt is claimed"
    assert not any(isinstance(node, ast.While) for node in ast.walk(function))
    for banned in ("queue", "scheduler", "daemon", "recurse", "next_cell"):
        assert banned not in rendered.lower()


def test_an_unregistered_seed_is_refused(workspace: Workspace) -> None:
    for seed in (404, 505, 7):
        with pytest.raises(LaunchError, match="not registered"):
            _plan(workspace.root, "C05", seed=seed)


# ==================================================================== C63-C68 P0
def test_c63_the_p0_stage_sequence_matches_00_35() -> None:
    registry = load_stage_registry(REPO_ROOT)
    assert registry.stage_ids == EXPECTED_STAGE_ORDER
    assert len(registry.stages) == 13
    assert registry.stage("P0-D2").depends_on == ("P0-D",)
    assert registry.stage("P0-D2").requires_upstream_failure == "P0-D"
    assert registry.stage("P0-C4").requires_selected_operator == SVD_TRUNC_MERGE
    assert registry.stage("P0-D3").requires_selected_operator == SVD_TRUNC_MERGE
    assert registry.stage("P0-F").depends_on == ("P0-C1",)
    assert registry.stage("P0-A").on_failure == "STOP_PRIVACY_GRID"
    assert registry.stage("P0-C1").on_failure == "STOP_RECOVERY_PIPELINE"


def test_c64_a_failed_p0_c1_blocks_the_recovery_continuation() -> None:
    registry = load_stage_registry(REPO_ROOT)
    failed = gate_state(P0_A="PASS", P0_C1="FAIL")
    assert recovery_pipeline_blocked(failed) is True
    eligibility = stage_eligibility(registry, "P0-F", failed)
    assert eligibility.eligible is False
    assert eligibility.reason == "BLOCKED_BY_DEPENDENCY"


def test_c65_p0_d2_is_unavailable_before_dare_has_failed() -> None:
    registry = load_stage_registry(REPO_ROOT)
    passed = gate_state(P0_A="PASS", P0_C1="PASS", P0_F="PASS", P0_D="PASS")
    eligibility = stage_eligibility(registry, "P0-D2", passed)
    assert eligibility.eligible is False
    assert eligibility.reason == "BLOCKED_BY_UPSTREAM_PASS"

    unresolved = gate_state(P0_A="PASS", P0_C1="PASS", P0_F="PASS")
    assert stage_eligibility(registry, "P0-D2", unresolved).eligible is False


@pytest.mark.parametrize("stage_id", ["P0-D3", "P0-C4"])
def test_c66_p0_d3_and_c4_are_unavailable_unless_o3_is_selected(stage_id: str) -> None:
    registry = load_stage_registry(REPO_ROOT)
    dare = gate_state(P0_A="PASS", P0_C1="PASS", P0_F="PASS", P0_D="PASS", P0_D2="PASS")
    eligibility = stage_eligibility(registry, stage_id, dare)
    assert eligibility.eligible is False
    assert eligibility.reason in {"BLOCKED_BY_OPERATOR", "BLOCKED_BY_DEPENDENCY"}
    assert conditional_cells_active(dare) is False


def test_c67_post_mvrs_expansion_stages_stay_gated() -> None:
    registry = load_stage_registry(REPO_ROOT)
    assert registry.calendar.compressed is True
    assert registry.calendar.software_slip is True
    assert registry.calendar.p1_launch_deadline == "2026-09-03T18:00:00+01:00"

    ready = gate_state(P0_A="PASS", P0_C1="PASS", P0_F="PASS")
    for stage_id in ("P0-C2", "P0-D", "P0-E"):
        stage = registry.stage(stage_id)
        assert stage.calendar_class == "POST_MVRS_EXPANSION"
        eligibility = stage_eligibility(registry, stage_id, ready)
        assert eligibility.eligible is False
        assert eligibility.reason == "DEFERRED_POST_MVRS_EXPANSION"


def test_a_deferred_stage_stays_in_the_registry() -> None:
    """00 §34C.1A: deferred is gated, never deleted."""
    registry = load_stage_registry(REPO_ROOT)
    assert {"P0-C2", "P0-D", "P0-E"} <= set(registry.stage_ids)


def test_c68_no_cli_flag_can_self_assert_a_p0_gate() -> None:
    """Checked on the parser's actual options, not on prose that names them to exclude them."""
    import importlib.util

    specification = importlib.util.spec_from_file_location("s09_launcher", LAUNCHER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    registered: set[str] = set()
    parser = module.build_parser()
    for action in parser._actions:  # noqa: SLF001
        registered.update(action.option_strings)
    for subparsers in (a for a in parser._actions if hasattr(a, "choices")):  # noqa: SLF001
        for candidate in (subparsers.choices or {}).values():
            for action in getattr(candidate, "_actions", ()):
                registered.update(action.option_strings)

    for banned in ("--p0-c1-pass", "--dare-pass", "--o3-pass", "--p0-c4-pass", "--allow-na"):
        assert banned not in registered, banned
    assert "--gate-state" in registered, "gate state arrives as a validated document"
    completed = _cli(
        "plan", "--cell", "C05", "--model", FIXTURE_MODEL_ALIAS, "--seed", "101", "--dare-pass"
    )
    assert completed.returncode == 1
    assert "comes from the registry row" in completed.stderr


# ==================================================================== command surface
def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(LAUNCHER), "--root", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_list_command_prints_the_canonical_registry() -> None:
    completed = _cli("list", "--phase", "p1")
    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert document["count"] == 29
    assert document["registry_sha256"] == registry_sha256(REPO_ROOT)
    assert len(list_cells(REPO_ROOT, regime="CALIBRATION")) == 19


def test_the_list_command_reports_the_p0_stage_state() -> None:
    completed = _cli("list", "--phase", "p0")
    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert [s["stage_id"] for s in document["stages"]] == list(EXPECTED_STAGE_ORDER)
    assert document["selected_primary_lossy_operator"] == "UNRESOLVED"
    assert document["calendar"]["calendar_compressed_mvrs"] == "ACTIVE"


def test_the_plan_command_resolves_one_registered_cell() -> None:
    completed = _cli("plan", "--cell", "C01", "--model", FIXTURE_MODEL_ALIAS, "--seed", "101")
    document = json.loads(completed.stdout)
    assert document["cell"]["cell_id"] == "C01"
    assert document["resolution"]["runnable"] is True
    assert document["resolution"]["resolved_status"] == "RUN_CONTROL"
    assert document["schedule"]["alpha"] == [0.5]
    assert completed.returncode == 0


def test_the_plan_command_exits_non_zero_for_a_gated_cell() -> None:
    completed = _cli("plan", "--cell", "C10", "--model", FIXTURE_MODEL_ALIAS, "--seed", "101")
    document = json.loads(completed.stdout)
    assert document["resolution"]["runnable"] is False
    assert document["resolution"]["reason"] == "OPERATOR_UNRESOLVED"
    assert completed.returncode == 1


def test_the_plan_command_refuses_an_unknown_cell() -> None:
    completed = _cli("plan", "--cell", "C99", "--model", FIXTURE_MODEL_ALIAS, "--seed", "101")
    assert completed.returncode == 1
    assert "not a registered P1 cell" in completed.stderr


@pytest.mark.parametrize(
    "override",
    [("--k", "4"), ("--lineage", "Z-HIGH"), ("--operator", "LOSSY"), ("--alpha", "0.2")],
)
def test_the_plan_command_refuses_a_scientific_override(override: tuple[str, str]) -> None:
    completed = _cli(
        "plan", "--cell", "C05", "--model", FIXTURE_MODEL_ALIAS, "--seed", "101", *override
    )
    assert completed.returncode == 1
    assert "comes from the registry row" in completed.stderr


def test_the_execute_command_registers_no_scientific_task_yet() -> None:
    """S09 registers no task; the registered set arrives with S10 [AUTH: 01 §39]."""
    completed = _cli(
        "execute",
        "--cell",
        "C05",
        "--model",
        FIXTURE_MODEL_ALIAS,
        "--seed",
        "101",
        "--task",
        "anything",
    )
    assert completed.returncode == 1
    assert "no scientific task is registered" in completed.stderr


def test_there_is_no_generic_fixture_flag() -> None:
    """C2: a `--fixture` flag could accidentally enter evidentiary use."""
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--fixture" not in source
    completed = _cli(
        "plan", "--cell", "C05", "--model", FIXTURE_MODEL_ALIAS, "--seed", "101", "--fixture"
    )
    assert completed.returncode == 1


def test_the_plan_command_is_side_effect_free_on_the_real_tree() -> None:
    before = sorted(p.name for p in (REPO_ROOT / "manifests" / "runs").glob("*"))
    _cli("plan", "--cell", "C08", "--model", FIXTURE_MODEL_ALIAS, "--seed", "202")
    after = sorted(p.name for p in (REPO_ROOT / "manifests" / "runs").glob("*"))
    assert before == after


def test_a_gate_document_may_be_supplied_but_never_asserted(tmp_path: Path) -> None:
    from src.experiments.gates import fixture_gate_state

    path = tmp_path / "gates.json"
    path.write_text(
        json.dumps(
            fixture_gate_state(
                statuses={"P0-D": "FAIL", "P0-D2": "PASS"},
                registry_sha256=registry_sha256(REPO_ROOT),
            )
        ),
        encoding="utf-8",
    )
    completed = _cli("list", "--phase", "p0", "--gate-state", str(path))
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["selected_primary_lossy_operator"] == SVD_TRUNC_MERGE


def test_the_run_manifest_records_the_registry_and_cell_identity(
    workspace: Workspace,
) -> None:
    plan = _plan(workspace.root, "C08")
    provenance = provenance_for(workspace, plan.resolved_config())
    result = execute_task(
        workspace.root,
        plan,
        ExecutionRequest(task_id="fixture-echo", argv=harmless_argv()),
        provenance=provenance,
        seeds=SEED_SET,
        precision="float64",
        tokenizer_hash="9" * 64,
    )
    manifest = read_canonical_json(
        workspace.root / "manifests" / "runs" / f"{result['run_id']}.json"
    )
    assert isinstance(manifest, dict)
    config = manifest["config"]
    assert isinstance(config, dict)
    assert config["registry_sha256"] == registry_sha256(workspace.root)
    assert config["cell_id"] == "C08"
    assert config["cell_definition_sha256"] == plan.cell.identity()
    assert manifest["status"] != RUNNING


def test_an_unresolved_operator_leaves_the_lossy_cells_unplannable(
    workspace: Workspace,
) -> None:
    plan = _plan(workspace.root, "C10", state=unresolved_state())
    assert plan.resolved.runnable is False
    assert resolve_primary_lossy_operator(unresolved_state()) == "UNRESOLVED"
