"""The pre-result full-sweep plan [AUTH: 00 §35, §36; 01 §16, §39].

This enumerates every task the already-frozen study requires, without executing any of them.
It is the proof that the sweep is wired: if a task cannot be named here, with its stage, its
dependencies, its gate and the artifact it produces, then the sweep is not ready no matter how
much library code exists.

Tasks are derived, never invented. The P0 stages come from `configs/experiments/p0_registry.json`
in the 00 §35 order, the cells from the 29-row P1 registry, the seeds and models from accepted
config. Nothing here adds a cell, a seed or a model.

The plan is deterministic and side-effect free: no wall clock enters it, nothing is written,
and no run attempt is claimed. Two runs of `build_sweep_plan` on an unchanged repository
produce byte-identical documents.

Each task carries an `execution_binding` — the production callable that would run it — and a
`readiness` class:

    READY                      the binding exists and is reachable
    GATED_BY_RESULT            the binding exists; a scientific gate decides whether it runs
    MISSING_EXECUTION_BINDING  no production code path exists yet

`MISSING_EXECUTION_BINDING` is a finding, not a placeholder: it means the sweep cannot start
until that binding is written.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from src.experiments.gates import GateState, resolve_primary_lossy_operator
from src.experiments.registry import Registry, load_registry
from src.experiments.resolution import resolve_cell
from src.experiments.schedules import ScheduleNotCalibratedError, schedule_for_cell
from src.experiments.stages import StageRegistry, load_stage_registry, stage_eligibility
from src.provenance.hashing import JSONValue, sha256_canonical

SWEEP_SCHEMA: Final = "s10.full-sweep-plan.v1"

READY: Final = "READY"
GATED_BY_RESULT: Final = "GATED_BY_RESULT"
MISSING_EXECUTION_BINDING: Final = "MISSING_EXECUTION_BINDING"

#: The execution code exists and is bound, but a material constant it consumes is still
#: `REQUIRED_NOT_CALIBRATED` in `configs/training/**`. This is a *scientific* gap, not a code
#: gap, and it is kept distinct from both of the others so neither can absorb it: calling it
#: MISSING_EXECUTION_BINDING would send someone to write code that already exists, and calling
#: it GATED_BY_RESULT would imply an experiment decides it [AUTH: 01 §17; 00 §8.3].
BLOCKED_BY_UNCALIBRATED_CONSTANT: Final = "BLOCKED_BY_UNCALIBRATED_CONSTANT"

READINESS_CLASSES: Final[tuple[str, ...]] = (
    READY,
    GATED_BY_RESULT,
    MISSING_EXECUTION_BINDING,
    BLOCKED_BY_UNCALIBRATED_CONSTANT,
)

#: What a task produces. Not a filename: the run lifecycle owns paths.
ARTIFACT_KINDS: Final[tuple[str, ...]] = (
    "GATE_EVIDENCE",
    "ADAPTER",
    "MERGE_FAMILY",
    "RECOVERY_RESULT",
    "SCORE_TABLE",
    "RESULT_TABLE",
    "BENCHMARK",
)

#: The production LoRA training loop. S06 froze the training contract and shipped a reference
#: trainer over numeric fixtures; this is the executor that runs it against real HF/PEFT
#: models under the accepted adapter-persistence and run-lifecycle paths.
TRAINING_BINDING: Final = "src.training.production:train_production_lora"


class SweepError(ValueError):
    """The sweep plan cannot be derived as specified."""


@dataclass(frozen=True)
class SweepTask:
    """One enumerated task. Named, bound, gated and hashed — but not run."""

    task_id: str
    stage: str
    kind: str
    execution_binding: str
    readiness: str
    depends_on: tuple[str, ...] = ()
    gate: str = ""
    model_alias: str | None = None
    seed: int | None = None
    cell_id: str | None = None
    detail: str = ""
    resolved_config: dict[str, JSONValue] = field(default_factory=dict)

    def resolved_config_sha256(self) -> str:
        return sha256_canonical(dict(self.resolved_config))

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "artifact_kind": self.kind,
            "execution_binding": self.execution_binding,
            "readiness": self.readiness,
            "depends_on": list(self.depends_on),
            "gate": self.gate,
            "model_alias": self.model_alias,
            "seed": self.seed,
            "cell_id": self.cell_id,
            "detail": self.detail,
            "resolved_config_sha256": self.resolved_config_sha256(),
        }


@dataclass(frozen=True)
class SweepPlan:
    """Every task the frozen study requires, in dependency order."""

    tasks: tuple[SweepTask, ...]
    registry_sha256: str
    selected_operator: str
    uncalibrated: tuple[str, ...] = ()

    def by_readiness(self, readiness: str) -> tuple[SweepTask, ...]:
        return tuple(task for task in self.tasks if task.readiness == readiness)

    @property
    def missing_bindings(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {task.execution_binding for task in self.by_readiness(MISSING_EXECUTION_BINDING)}
            )
        )

    @property
    def blocked_constants(self) -> tuple[str, ...]:
        """Material constants that exist in config but are not yet frozen."""
        return self.uncalibrated

    @property
    def ready(self) -> bool:
        """Code-ready: every task names an execution binding that exists.

        This is deliberately *not* "the sweep can run": a frozen-constant gap leaves the code
        complete and the study still unable to start. `executable` is the stronger property.
        """
        return not self.missing_bindings

    @property
    def executable(self) -> bool:
        """Runnable now: code-ready AND no material constant is still uncalibrated."""
        return self.ready and not self.uncalibrated

    def as_dict(self) -> dict[str, JSONValue]:
        counts: dict[str, JSONValue] = {
            readiness: len(self.by_readiness(readiness)) for readiness in READINESS_CLASSES
        }
        return {
            "schema": SWEEP_SCHEMA,
            "registry_sha256": self.registry_sha256,
            "selected_primary_lossy_operator": self.selected_operator,
            "n_tasks": len(self.tasks),
            "readiness_counts": counts,
            "full_sweep_code_ready": self.ready,
            "full_sweep_executable": self.executable,
            "missing_execution_bindings": list(self.missing_bindings),
            "uncalibrated_constants": list(self.uncalibrated),
            "tasks": [task.as_dict() for task in self.tasks],
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())


def uncalibrated_training_constants(root: Path) -> tuple[str, ...]:
    """Which material training constants are still `REQUIRED_NOT_CALIBRATED`.

    Read from the config documents themselves, so this cannot drift from what the trainer
    will actually refuse at run time [AUTH: 01 §17].
    """
    from src.materials import uncalibrated_keys
    from src.training.settings import dp_settings, lora_settings

    found = [f"lora.{key}" for key in uncalibrated_keys(lora_settings(root))]
    found += [f"dp.{key}" for key in uncalibrated_keys(dp_settings(root))]
    return tuple(sorted(found))


def _training_readiness(blocked: Sequence[str]) -> tuple[str, str]:
    """A training task's class, and the detail that explains it."""
    if blocked:
        return (
            BLOCKED_BY_UNCALIBRATED_CONSTANT,
            f"the execution binding exists; {len(blocked)} material constant(s) are still"
            f" REQUIRED_NOT_CALIBRATED, first {blocked[0]} [AUTH: 01 §17; 00 §8.3]",
        )
    return READY, "the trainer and every material constant it consumes are resolved"


def _training_tasks(
    models: Sequence[str],
    seeds: Sequence[int],
    dp_seeds: Sequence[int],
    blocked: Sequence[str],
) -> list[SweepTask]:
    """Training jobs: the canary arm, its matched no-canary control, and the DP arm.

    00 §7.5 pairs every canary-containing calibration seed with a matched no-canary control, so
    both are enumerated per seed rather than the control being assumed.
    """
    readiness, why = _training_readiness(blocked)
    tasks: list[SweepTask] = []
    for alias in models:
        for seed in seeds:
            for arm, note in (
                ("canary", "canary-containing calibration seed [AUTH: 00 §7.5]"),
                ("control", "matched no-canary control [AUTH: 00 §7.5]"),
            ):
                tasks.append(
                    SweepTask(
                        task_id=f"train.{alias}.{arm}.{seed}",
                        stage="S06-TRAIN",
                        kind="ADAPTER",
                        execution_binding=TRAINING_BINDING,
                        readiness=readiness,
                        depends_on=("stage.P0-PRE", "stage.P0-0"),
                        gate="P0-PRE",
                        model_alias=alias,
                        seed=seed,
                        detail=f"{note}; {why}",
                        resolved_config={"alias": alias, "arm": arm, "seed": seed},
                    )
                )
        for seed in dp_seeds:
            tasks.append(
                SweepTask(
                    task_id=f"train.dp.{alias}.{seed}",
                    stage="S06-TRAIN-DP",
                    kind="ADAPTER",
                    execution_binding=TRAINING_BINDING,
                    readiness=readiness,
                    depends_on=("stage.P0-PRE",),
                    gate="P0-PRE",
                    model_alias=alias,
                    seed=seed,
                    detail=(
                        f"representative DP arm; the accountant is src.dp.mechanism.account; {why}"
                    ),
                    resolved_config={"alias": alias, "arm": "dp", "seed": seed},
                )
            )
    return tasks


def _stage_tasks(stages: StageRegistry, state: GateState) -> list[SweepTask]:
    """One task per registered P0 stage, in the 00 §35 order."""
    tasks: list[SweepTask] = []
    for stage in stages.stages:
        eligibility = stage_eligibility(stages, stage.stage_id, state)
        tasks.append(
            SweepTask(
                task_id=f"stage.{stage.stage_id}",
                stage=stage.stage_id,
                kind="GATE_EVIDENCE",
                execution_binding="src.experiments.stages:stage_eligibility",
                readiness=READY if eligibility.eligible else GATED_BY_RESULT,
                depends_on=tuple(f"stage.{d}" for d in stage.depends_on),
                gate=stage.gate,
                detail=eligibility.detail,
                resolved_config={
                    "stage_id": stage.stage_id,
                    "calendar_class": stage.calendar_class,
                },
            )
        )
    return tasks


def _cell_tasks(
    root: Path, registry: Registry, state: GateState, models: Sequence[str]
) -> list[SweepTask]:
    """Merge, recovery and scoring jobs for every registered cell."""
    tasks: list[SweepTask] = []
    for cell in registry.cells:
        if cell.status == "STRUCTURAL_NA":
            tasks.append(
                SweepTask(
                    task_id=f"cell.{cell.cell_id}",
                    stage="P1",
                    kind="RECOVERY_RESULT",
                    execution_binding="(none: permanently non-executable)",
                    readiness=GATED_BY_RESULT,
                    gate="STRUCTURAL_NA",
                    cell_id=cell.cell_id,
                    detail="unknown partner, one mixture: skipped, never run [AUTH: 00 §36.1]",
                    resolved_config={"cell_id": cell.cell_id},
                )
            )
            continue
        try:
            schedule = schedule_for_cell(root, cell)
            schedule_resolved = True
        except ScheduleNotCalibratedError:
            # Registered but carrying no frozen coefficients: reported, never assumed.
            schedule, schedule_resolved = None, False
        resolved = resolve_cell(cell, state, dp_seeds=(), schedule_resolved=schedule_resolved)
        for alias in models:
            tasks.append(
                SweepTask(
                    task_id=f"cell.{cell.cell_id}.{alias}",
                    stage="P1",
                    kind="SCORE_TABLE",
                    execution_binding="src.scoring.engine:score_records",
                    readiness=READY if resolved.runnable else GATED_BY_RESULT,
                    depends_on=(f"train.{alias}.canary.101", "stage.P0-C1"),
                    gate=resolved.reason,
                    model_alias=alias,
                    cell_id=cell.cell_id,
                    detail=resolved.detail,
                    resolved_config={
                        "cell_id": cell.cell_id,
                        "operator": resolved.resolved_operator,
                        "k": cell.k,
                        "lineage": cell.lineage,
                        "alpha": None if schedule is None else list(schedule.alpha),
                        "model_alias": alias,
                    },
                )
            )
    return tasks


def _analysis_tasks() -> list[SweepTask]:
    """The result tables the study ends in."""
    return [
        SweepTask(
            task_id="analysis.rank_matched_ladder",
            stage="P0-F",
            kind="RESULT_TABLE",
            execution_binding="src.analysis.isotonic:fit_monotone_curve",
            readiness=GATED_BY_RESULT,
            depends_on=("stage.P0-F",),
            gate="P0-F",
            detail="primary rank-matched ladder [AUTH: 00 §28A]",
            resolved_config={"ladder": "rank_matched"},
        ),
        SweepTask(
            task_id="analysis.isotropic_ladder",
            stage="P0-F",
            kind="RESULT_TABLE",
            execution_binding="src.analysis.isotonic:fit_monotone_curve",
            readiness=GATED_BY_RESULT,
            depends_on=("stage.P0-F",),
            gate="P0-F",
            detail="secondary full-rank isotropic ladder [AUTH: 00 §28A]",
            resolved_config={"ladder": "isotropic"},
        ),
        SweepTask(
            task_id="analysis.privacy_eligibility",
            stage="P0-A",
            kind="RESULT_TABLE",
            execution_binding="src.analysis.endpoints:report_normalised_privacy",
            readiness=GATED_BY_RESULT,
            depends_on=("stage.P0-A",),
            gate="P0-A",
            detail="per-arm eligibility [AUTH: 00 §18]",
            resolved_config={"analysis": "privacy_eligibility"},
        ),
        SweepTask(
            task_id="analysis.bootstrap_intervals",
            stage="P1",
            kind="RESULT_TABLE",
            execution_binding="src.analysis.resampling:bootstrap",
            readiness=GATED_BY_RESULT,
            depends_on=("analysis.privacy_eligibility",),
            gate="P1",
            detail="cluster bootstrap over the registered cells",
            resolved_config={"analysis": "bootstrap"},
        ),
    ]


def build_sweep_plan(
    root: Path,
    *,
    state: GateState,
    models: Sequence[str] = ("qwen3_5_4b_base",),
) -> SweepPlan:
    """Enumerate the whole study. Deterministic, side-effect free, executes nothing."""
    from src.experiments.registry import CALIBRATION_SEEDS, DP_INFERENTIAL_SEEDS
    from src.experiments.settings import registry_sha256

    registry = load_registry(root)
    stages = load_stage_registry(root)
    tasks: list[SweepTask] = []
    tasks += _stage_tasks(stages, state)
    blocked = uncalibrated_training_constants(root)
    tasks += _training_tasks(models, CALIBRATION_SEEDS, sorted(DP_INFERENTIAL_SEEDS), blocked)
    tasks += _cell_tasks(root, registry, state, models)
    tasks += _analysis_tasks()

    identifiers = [task.task_id for task in tasks]
    duplicates = sorted({i for i in identifiers if identifiers.count(i) > 1})
    if duplicates:
        raise SweepError(f"duplicate task id(s): {duplicates}")

    return SweepPlan(
        tasks=tuple(tasks),
        registry_sha256=registry_sha256(root),
        selected_operator=resolve_primary_lossy_operator(state),
        uncalibrated=blocked,
    )


def readiness_audit(plan: SweepPlan) -> dict[str, JSONValue]:
    """The K audit, derived from the plan rather than asserted alongside it."""
    return {
        "schema": SWEEP_SCHEMA,
        "full_sweep_code_ready": plan.ready,
        "full_sweep_executable": plan.executable,
        "missing_execution_bindings": list(plan.missing_bindings),
        "uncalibrated_constants": list(plan.uncalibrated),
        "counts": {readiness: len(plan.by_readiness(readiness)) for readiness in READINESS_CLASSES},
        "missing_tasks": [task.task_id for task in plan.by_readiness(MISSING_EXECUTION_BINDING)][
            :20
        ],
        "blocked_tasks": [
            task.task_id for task in plan.by_readiness(BLOCKED_BY_UNCALIBRATED_CONSTANT)
        ][:20],
    }
