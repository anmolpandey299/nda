"""The P0 execution sequence as a state machine over data [AUTH: 00 §34C.1A, §35].

The stages, their dependencies, their gates and their calendar classes live in
`configs/experiments/p0_registry.json`. This module walks that document; it encodes no
branching of its own, because ad-hoc Python `if` chains are how an execution order drifts from
the one 00 §35 fixes.

Three orderings are enforced rather than assumed:

* a stage whose dependency has not passed is not eligible — `P0-F` waits on `P0-C1`, and a
  failed `P0-C1` stops the recovery pipeline outright;
* `P0-D2` requires `P0-D` to have **failed**. Running the fallback before the primary has
  failed would be choosing an operator by availability [AUTH: 00 §35];
* `P0-D3` and `P0-C4` require O3 to be the *selected* operator, which is the resolver's output
  and not a status anyone can assert.

Under `CALENDAR_COMPRESSED_MVRS`, `P0-C2`, `P0-D` and `P0-E` carry the `POST_MVRS_EXPANSION`
class and stay deferred. They remain in the registry: a deferred stage is gated, never deleted
[AUTH: 00 §34C.1A].
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from src.experiments.gates import (
    FAIL,
    PASS,
    SVD_TRUNC_MERGE,
    UNRESOLVED,
    GateState,
    resolve_primary_lossy_operator,
)
from src.provenance.hashing import JSONValue, sha256_canonical

STAGE_SCHEMA: Final = "s09.p0-stage.v1"

CRITICAL_PATH: Final = "CRITICAL_PATH"
POST_MVRS_EXPANSION: Final = "POST_MVRS_EXPANSION"
CALENDAR_CLASSES: Final[tuple[str, ...]] = (CRITICAL_PATH, POST_MVRS_EXPANSION)

#: What a stage may be, from S09's point of view. None of these is a scientific outcome.
ELIGIBLE: Final = "ELIGIBLE"
BLOCKED_BY_DEPENDENCY: Final = "BLOCKED_BY_DEPENDENCY"
BLOCKED_BY_UPSTREAM_PASS: Final = "BLOCKED_BY_UPSTREAM_PASS"
BLOCKED_BY_OPERATOR: Final = "BLOCKED_BY_OPERATOR"
DEFERRED_CALENDAR: Final = "DEFERRED_POST_MVRS_EXPANSION"
ALREADY_RESOLVED: Final = "ALREADY_RESOLVED"

#: The 00 §35 order, used to check that the registry document is the sequence it claims.
EXPECTED_STAGE_ORDER: Final[tuple[str, ...]] = (
    "P0-PRE",
    "P0-0",
    "P0-A0",
    "P0-A",
    "P0-B",
    "P0-C1",
    "P0-F",
    "P0-C2",
    "P0-D",
    "P0-D2",
    "P0-D3",
    "P0-C4",
    "P0-E",
)


class StageError(ValueError):
    """The P0 stage registry is malformed, or a stage was requested that it does not hold."""


@dataclass(frozen=True)
class Stage:
    """One registered P0 stage."""

    stage_id: str
    title: str
    depends_on: tuple[str, ...]
    calendar_class: str
    gate: str
    on_failure: str
    on_pass: str | None = None
    requires_upstream_failure: str | None = None
    requires_selected_operator: str | None = None
    activates: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": STAGE_SCHEMA,
            "stage_id": self.stage_id,
            "title": self.title,
            "depends_on": list(self.depends_on),
            "calendar_class": self.calendar_class,
            "gate": self.gate,
            "on_pass": self.on_pass,
            "on_failure": self.on_failure,
            "requires_upstream_failure": self.requires_upstream_failure,
            "requires_selected_operator": self.requires_selected_operator,
            "activates": list(self.activates),
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())


@dataclass(frozen=True)
class StageEligibility:
    """Why a stage may or may not run right now."""

    stage_id: str
    eligible: bool
    reason: str
    detail: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "stage_id": self.stage_id,
            "eligible": self.eligible,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Calendar:
    """The planning state 00 §34C.1A and §42 record. Never a reason to weaken a gate."""

    software_slip: bool
    calendar_compressed_mvrs: str
    p1_launch_deadline: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "software_slip": self.software_slip,
            "calendar_compressed_mvrs": self.calendar_compressed_mvrs,
            "p1_launch_deadline": self.p1_launch_deadline,
        }

    @property
    def compressed(self) -> bool:
        return self.calendar_compressed_mvrs == "ACTIVE"


@dataclass(frozen=True)
class StageRegistry:
    """The 00 §35 sequence."""

    stages: tuple[Stage, ...]
    calendar: Calendar
    version: str

    def stage(self, stage_id: str) -> Stage:
        for candidate in self.stages:
            if candidate.stage_id == stage_id:
                return candidate
        raise StageError(f"{stage_id!r} is not a registered P0 stage")

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)


def load_stage_registry(root: Path) -> StageRegistry:
    """Load and validate the P0 stage document [AUTH: 00 §35]."""
    from src.experiments.settings import p0_registry_settings

    document = p0_registry_settings(root).document
    raw = document.get("stages")
    if not isinstance(raw, list):
        raise StageError("the P0 registry defines no 'stages' list")

    problems: list[str] = []
    stages: list[Stage] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            problems.append("every P0 stage must be an object")
            continue
        stage_id = str(entry.get("stage_id"))
        if stage_id in seen:
            problems.append(f"duplicate stage {stage_id}")
        seen.add(stage_id)
        calendar_class = str(entry.get("calendar_class"))
        if calendar_class not in CALENDAR_CLASSES:
            problems.append(f"{stage_id}: calendar_class {calendar_class!r} is not registered")
        depends = entry.get("depends_on")
        if not isinstance(depends, list):
            problems.append(f"{stage_id}: depends_on must be a list")
            depends = []
        stages.append(
            Stage(
                stage_id=stage_id,
                title=str(entry.get("title", "")),
                depends_on=tuple(str(d) for d in depends),
                calendar_class=calendar_class,
                gate=str(entry.get("gate", "")),
                on_failure=str(entry.get("on_failure", "")),
                on_pass=None if entry.get("on_pass") is None else str(entry["on_pass"]),
                requires_upstream_failure=(
                    None
                    if entry.get("requires_upstream_failure") is None
                    else str(entry["requires_upstream_failure"])
                ),
                requires_selected_operator=(
                    None
                    if entry.get("requires_selected_operator") is None
                    else str(entry["requires_selected_operator"])
                ),
                activates=tuple(str(a) for a in entry.get("activates", []) or []),
            )
        )

    order = tuple(stage.stage_id for stage in stages)
    if order != EXPECTED_STAGE_ORDER:
        problems.append(
            f"the stage sequence {list(order)} is not the 00 §35 order {list(EXPECTED_STAGE_ORDER)}"
        )
    for stage in stages:
        unknown = sorted(set(stage.depends_on) - seen)
        if unknown:
            problems.append(f"{stage.stage_id}: depends on unregistered stage(s) {unknown}")

    calendar_raw = document.get("calendar")
    if not isinstance(calendar_raw, Mapping):
        problems.append("the P0 registry declares no calendar state")
        calendar_raw = {}
    if problems:
        raise StageError("; ".join(problems))

    return StageRegistry(
        stages=tuple(stages),
        calendar=Calendar(
            software_slip=bool(calendar_raw.get("software_slip")),
            calendar_compressed_mvrs=str(calendar_raw.get("calendar_compressed_mvrs")),
            p1_launch_deadline=str(calendar_raw.get("p1_launch_deadline")),
        ),
        version=str(document.get("registry_version", "")),
    )


def stage_eligibility(registry: StageRegistry, stage_id: str, state: GateState) -> StageEligibility:
    """Whether one P0 stage may run right now, and why not when it may not.

    `eligible` never means "this stage passed": it means the sequence permits attempting it.
    No empirical outcome is inferred here.
    """
    stage = registry.stage(stage_id)
    own = _status_of(state, stage_id)
    if own in (PASS, FAIL):
        return StageEligibility(
            stage_id=stage_id,
            eligible=False,
            reason=ALREADY_RESOLVED,
            detail=f"{stage_id} is already recorded as {own}",
        )

    for dependency in stage.depends_on:
        status = _status_of(state, dependency)
        if status != PASS:
            return StageEligibility(
                stage_id=stage_id,
                eligible=False,
                reason=BLOCKED_BY_DEPENDENCY,
                detail=f"{dependency} is {status}; 00 §35 runs it first",
            )

    upstream = stage.requires_upstream_failure
    if upstream is not None:
        status = _status_of(state, upstream)
        if status != FAIL:
            return StageEligibility(
                stage_id=stage_id,
                eligible=False,
                reason=BLOCKED_BY_UPSTREAM_PASS,
                detail=(
                    f"{stage_id} is the fallback for {upstream}, which is {status}. Running it"
                    " before the primary has failed would select an operator by availability"
                    " [AUTH: 00 §35]"
                ),
            )

    required = stage.requires_selected_operator
    if required is not None:
        selected = resolve_primary_lossy_operator(state)
        if selected != required:
            return StageEligibility(
                stage_id=stage_id,
                eligible=False,
                reason=BLOCKED_BY_OPERATOR,
                detail=(
                    f"{stage_id} applies only when the selected primary lossy operator is"
                    f" {required}; it currently resolves to {selected}"
                ),
            )

    if registry.calendar.compressed and stage.calendar_class == POST_MVRS_EXPANSION:
        return StageEligibility(
            stage_id=stage_id,
            eligible=False,
            reason=DEFERRED_CALENDAR,
            detail=(
                f"{stage_id} is POST_MVRS_EXPANSION and CALENDAR_COMPRESSED_MVRS is ACTIVE, so"
                " it is deferred unless separately authorised [AUTH: 00 §34C.1A]"
            ),
        )
    return StageEligibility(
        stage_id=stage_id, eligible=True, reason=ELIGIBLE, detail="the sequence permits this"
    )


def _status_of(state: GateState, stage_id: str) -> str:
    """A stage the gate vocabulary does not cover is UNRESOLVED, never assumed passed.

    Only some 00 §35 stages are registered gates; the rest have no recorded decision, which is
    exactly what UNRESOLVED means here.
    """
    from src.experiments.gates import REGISTERED_GATES

    if stage_id not in REGISTERED_GATES:
        return UNRESOLVED
    return state.status(stage_id)


def recovery_pipeline_blocked(state: GateState) -> bool:
    """00 §35: a failed P0-C1 stops the recovery pipeline for debugging."""
    return _status_of(state, "P0-C1") == FAIL


def conditional_cells_active(state: GateState) -> bool:
    """C14/C15/C17/C18 run only when O3 is selected AND P0-C4 passes [AUTH: 00 §25, §36]."""
    return (
        resolve_primary_lossy_operator(state) == SVD_TRUNC_MERGE
        and _status_of(state, "P0-C4") == PASS
    )


def varying_alpha_activated(state: GateState) -> bool:
    """00 §36: the known-varying-alpha lineage waits on the fixed P0-C2 baseline."""
    return _status_of(state, "P0-C2") == PASS


def eligibility_report(registry: StageRegistry, state: GateState) -> dict[str, Any]:
    """Every stage's current eligibility, for the LIST view."""
    return {
        "registry_version": registry.version,
        "calendar": registry.calendar.as_dict(),
        "stages": [
            stage_eligibility(registry, stage.stage_id, state).as_dict()
            for stage in registry.stages
        ],
    }
