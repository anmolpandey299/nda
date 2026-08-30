"""The lightweight orchestrator: LIST, PLAN, EXECUTE [AUTH: 00 §36; 01 §15, §16, §36, §39].

The smallest correct layer. There is no database, service, daemon, queue, scheduler, dashboard
or DAG walker here — one invocation resolves one registered cell and, at most, claims one run
attempt through the **already accepted** S01 lifecycle. `begin_run`, `RunAttempt.finalize`,
`experiment_id` and `run_id` are Block A's and are used as they stand: a second RUN_ID
algorithm would mean two answers to "which run is this".

The three modes differ in what they are allowed to touch:

* **LIST** reads the registry and reports static statuses. No provenance is required.
* **PLAN** resolves one cell against the gate state and emits a canonical plan document plus
  its hash. It is side-effect free by construction — it never claims an attempt, never writes a
  manifest, and carries no wall-clock field, so repeating it is bitwise identical.
* **EXECUTE** additionally requires complete provenance, claims exactly one attempt, runs one
  registered command, captures its streams, hash-binds its artifacts and finalises once.

The registry identity is bound into the resolved config, so a run made under one registry can
never be attributed to another [AUTH: 00 §36; 01 §15].
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src.experiments.gates import GateState
from src.experiments.registry import Cell, Registry, load_registry
from src.experiments.resolution import ResolvedCell, require_runnable, resolve_cell
from src.experiments.schedules import (
    ScheduleNotCalibratedError,
    schedule_for_cell,
)
from src.experiments.settings import registry_sha256
from src.provenance.hashing import JSONDocument, JSONValue, sha256_canonical
from src.provenance.identity import ScientificProvenance

PLAN_SCHEMA: Final = "s09.launch-plan.v1"

#: Terminal states are 01 §36's. S09 invents none [AUTH: 01 §36].
SUCCESS: Final = "SUCCESS"
FAILED_IMPLEMENTATION: Final = "FAILED_IMPLEMENTATION"
FAILED_ENVIRONMENT: Final = "FAILED_ENVIRONMENT"


class LaunchError(RuntimeError):
    """The launcher refused an operation."""


class PreconditionError(LaunchError):
    """A runtime precondition for an evidentiary run is unresolved [AUTH: 01 §16]."""


@dataclass(frozen=True)
class LaunchPlan:
    """One resolved cell, ready to run — or a statement of why it is not.

    Deliberately carries no timestamp: a plan that embedded wall-clock time would hash
    differently every call, and the whole point of a plan identity is that it does not.
    """

    cell: Cell
    resolved: ResolvedCell
    model_alias: str
    seed: int
    registry_sha256: str
    schedule: Mapping[str, JSONValue] | None

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": PLAN_SCHEMA,
            "cell": self.cell.as_dict(),
            "resolution": self.resolved.as_dict(),
            "model_alias": self.model_alias,
            "training_seed": self.seed,
            "registry_sha256": self.registry_sha256,
            "schedule": None if self.schedule is None else dict(self.schedule),
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())

    def resolved_config(self) -> dict[str, JSONValue]:
        """The document whose hash a run manifest binds [AUTH: 01 §15, §17].

        The registry identity and the cell definition identity are inside it, so a run made
        under registry A cannot be attributed to registry B: the config hash — and therefore
        the EXPERIMENT_ID and the RUN_ID — moves with the registry.
        """
        return {
            "schema": PLAN_SCHEMA,
            "registry_sha256": self.registry_sha256,
            "cell_id": self.cell.cell_id,
            "cell_definition_sha256": self.cell.identity(),
            "operator": self.resolved.resolved_operator,
            "k": self.cell.k,
            "lineage": self.cell.lineage,
            "schedule": None if self.schedule is None else dict(self.schedule),
            "model_alias": self.model_alias,
            "training_seed": self.seed,
        }


def list_cells(root: Path, *, regime: str | None = None) -> list[dict[str, JSONValue]]:
    """LIST: the canonical registry rows and their static statuses. No provenance needed."""
    registry = load_registry(root)
    cells = registry.cells
    if regime is not None:
        cells = tuple(c for c in cells if c.regime == regime)
    return [cell.as_dict() for cell in cells]


def plan_cell(
    root: Path,
    cell_id: str,
    *,
    model_alias: str,
    seed: int,
    state: GateState,
    dp_seeds: Sequence[int] = (),
    registry: Registry | None = None,
) -> LaunchPlan:
    """PLAN: resolve one registered cell. Side-effect free.

    Nothing here writes, claims or mutates anything — not a manifest, not an attempt
    directory, not a gate document. The plan is derived from the registry, the gate state and
    two registered execution dimensions, and that is all.
    """
    registry = registry or load_registry(root)
    cell = registry.cell(cell_id)
    _require_registered_seed(cell, seed)

    schedule: Mapping[str, JSONValue] | None = None
    schedule_resolved = True
    try:
        resolved_schedule = schedule_for_cell(root, cell)
        schedule = None if resolved_schedule is None else resolved_schedule.as_dict()
    except ScheduleNotCalibratedError:
        # Registered but carrying no frozen coefficients: the plan still resolves and reports
        # the dependency rather than assuming a value.
        schedule_resolved = False

    resolution = resolve_cell(cell, state, dp_seeds=dp_seeds, schedule_resolved=schedule_resolved)
    return LaunchPlan(
        cell=cell,
        resolved=resolution,
        model_alias=model_alias,
        seed=seed,
        registry_sha256=registry_sha256(root),
        schedule=schedule,
    )


def _require_registered_seed(cell: Cell, seed: int) -> None:
    from src.experiments.registry import registered_seeds

    permitted = registered_seeds(cell.regime)
    if seed not in permitted:
        raise LaunchError(
            f"training seed {seed} is not registered for {cell.regime}; 00 §9 registers"
            f" {list(permitted)}. 404 and 505 are confirmatory-only expansion seeds and are"
            " not part of the ordinary P0/P1 registry"
        )


@dataclass(frozen=True)
class ExecutionRequest:
    """One registered command to run under one claimed attempt.

    `argv` is an argument array, never a shell string: there is no shell, no interpolation and
    nothing for a cell id to inject into.
    """

    task_id: str
    argv: tuple[str, ...]
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.argv:
            raise LaunchError(f"{self.task_id}: a registered task needs a command")
        for token in self.argv:
            if not isinstance(token, str):
                raise LaunchError(f"{self.task_id}: every argv token must be a string")


def require_execution_preconditions(
    root: Path, provenance: ScientificProvenance, *, evidentiary: bool
) -> None:
    """Everything an evidentiary run needs, checked before an attempt is claimed.

    Failing here leaves no run directory behind. Nothing is filled in: an unresolved model
    revision, environment identity or data manifest is reported, never replaced with zeros,
    `UNKNOWN`, `latest` or a fixture value [AUTH: 01 §16; 03 §8].
    """
    problems = provenance.problems()
    if problems:
        raise PreconditionError(
            "the run provenance is incomplete, so no evidentiary attempt may be claimed: "
            + "; ".join(problems)
            + " [AUTH: 01 §15, §16]"
        )
    if not evidentiary:
        return
    from src.provenance.repository import production_tree_dirty

    dirty = production_tree_dirty(root)
    if dirty:
        raise PreconditionError(
            f"the production tree is dirty ({len(dirty)} path(s), first {dirty[0]!r});"
            " an evidentiary run must be reproducible from a committed state"
            " [AUTH: 01 §35(3), §36]"
        )


def execute_task(
    root: Path,
    plan: LaunchPlan,
    request: ExecutionRequest,
    *,
    provenance: ScientificProvenance,
    seeds: Mapping[str, int],
    precision: str,
    tokenizer_hash: str,
    evidentiary: bool = True,
    timeout_seconds: int = 300,
) -> dict[str, JSONValue]:
    """EXECUTE: claim exactly one attempt, run one command, finalise once.

    The order is the one 01 §16 requires: preconditions first, so a refusal leaves nothing
    behind; then the attempt is claimed and its manifest written before any work starts, so a
    crash leaves a visibly unfinished run rather than an invisible one.

    Exactly one command runs. There is no next cell, no queue and no recursion: one invocation
    is one attempt.
    """
    from src.provenance.runs import begin_run

    require_runnable(plan.resolved)
    require_execution_preconditions(root, provenance, evidentiary=evidentiary)

    run = begin_run(
        root,
        provenance,
        resolved_config=plan.resolved_config(),
        seeds=seeds,
        precision=precision,
        tokenizer_hash=tokenizer_hash,
        evidentiary=evidentiary,
    )

    try:
        completed = subprocess.run(  # noqa: S603
            list(request.argv),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        (root / run.stdout_path).write_text(completed.stdout, encoding="utf-8")
        (root / run.stderr_path).write_text(completed.stderr, encoding="utf-8")
        run.add_artifact(run.stdout_path)
        run.add_artifact(run.stderr_path)
        for relative in request.artifacts:
            run.add_artifact(relative)
        status = SUCCESS if completed.returncode == 0 else FAILED_IMPLEMENTATION
        manifest = run.finalize(status=status, exit_code=completed.returncode)
    except subprocess.TimeoutExpired:
        manifest = run.finalize(status=FAILED_ENVIRONMENT, exit_code=124)
    except Exception:
        # A claimed attempt is never left falsely RUNNING or falsely SUCCESS [AUTH: 01 §36].
        run.finalize(status=FAILED_IMPLEMENTATION, exit_code=1)
        raise
    return {
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "attempt": run.attempt,
        "status": str(manifest["status"]),
        "exit_code": manifest["exit_code"],
        "plan_sha256": plan.identity(),
        "registry_sha256": plan.registry_sha256,
        "cell_id": plan.cell.cell_id,
        "stdout_path": run.stdout_path,
        "stderr_path": run.stderr_path,
    }


def backend_compatibility_argv(
    *, alias: str, run_identifier: str, evidence_path: str, extra: Sequence[str] = ()
) -> tuple[str, ...]:
    """The argv for the accepted backend compatibility command [AUTH: PRE_S09 §15].

    That command requires an EXTERNALLY issued RUN_ID and mints none of its own. S09 is the
    layer that supplies it — the S01 RUN_ID for the claimed attempt, passed through unchanged
    so the evidence binds exactly the run that produced it.
    """
    return (
        "python",
        "scripts/model_compatibility_contract.py",
        "--alias",
        alias,
        "--run-id",
        run_identifier,
        "--evidence",
        evidence_path,
        *extra,
    )


def plan_document(plan: LaunchPlan) -> JSONDocument:
    """The canonical plan, for printing or hashing."""
    document: dict[str, Any] = dict(plan.as_dict())
    document["plan_sha256"] = plan.identity()
    document["resolved_config_sha256"] = sha256_canonical(plan.resolved_config())
    return document
