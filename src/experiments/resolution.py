"""Resolving a registered cell's current status [AUTH: 00 §25, §36; 01 §17].

Resolution only ever *narrows*. A registry row's status is what 00 §36 registered; gate state
can make a row non-runnable, and can turn a `CONDITIONAL` row into the reproduction it was
always registered to be — but nothing here promotes a row into a status the registry never
gave it, and no row is created, deleted or rewritten because of an outcome.

Four rules, each straight from the spec:

* `STRUCTURAL_NA` is permanent. C04/C07/C13/C16 are unknown-partner single-mixture cells; there
  is no quantity to recover, so no flag makes them runnable [AUTH: 00 §36.1];
* an abstract `LOSSY` row resolves to the ONE selected primary lossy operator, or stays gated;
* `CONDITIONAL` rows run only under O3 *and* a passing P0-C4; under DARE they are
  `METHOD_GATED`, because no DARE-compatible incomplete-lineage recovery method is frozen;
* the known-varying-alpha lineage waits on the fixed P0-C2 baseline, and a DP row's inferential
  status waits on the exact three-seed set — not on a count of three.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from src.experiments.gates import (
    DARE,
    OPERATOR_AXIS_INSUFFICIENT,
    UNRESOLVED,
    GateState,
    resolve_primary_lossy_operator,
)
from src.experiments.registry import DP_INFERENTIAL_SEEDS, Cell
from src.experiments.stages import conditional_cells_active, varying_alpha_activated
from src.provenance.hashing import JSONValue

VARYING_LINEAGE: Final = "Z-INCOMPLETE-VARYING-KNOWN"

#: Why a registered row is not runnable right now. None of these is a scientific outcome.
NOT_RUNNABLE_STRUCTURAL: Final = "STRUCTURAL_NA"
NOT_RUNNABLE_OPERATOR_UNRESOLVED: Final = "OPERATOR_UNRESOLVED"
NOT_RUNNABLE_OPERATOR_INSUFFICIENT: Final = "OPERATOR_AXIS_INSUFFICIENT"
NOT_RUNNABLE_METHOD_GATED: Final = "METHOD_GATED"
NOT_RUNNABLE_CONDITIONAL_GATED: Final = "CONDITIONAL_O3_C4_NOT_PASSED"
NOT_RUNNABLE_VARYING_GATED: Final = "VARYING_ALPHA_AWAITS_P0_C2"
NOT_RUNNABLE_DP_SMOKE: Final = "DP_SMOKE_ONLY"
NOT_RUNNABLE_SCHEDULE: Final = "SCHEDULE_NOT_RESOLVABLE"
RUNNABLE: Final = "RUNNABLE"


class ResolutionError(ValueError):
    """A registered cell cannot be resolved as requested."""


@dataclass(frozen=True)
class ResolvedCell:
    """One registered row, resolved against the current gate state."""

    cell: Cell
    resolved_status: str
    resolved_operator: str
    runnable: bool
    reason: str
    detail: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "cell_id": self.cell.cell_id,
            "registered_status": self.cell.status,
            "resolved_status": self.resolved_status,
            "resolved_operator": self.resolved_operator,
            "runnable": self.runnable,
            "reason": self.reason,
            "detail": self.detail,
            "cell_definition_sha256": self.cell.identity(),
        }


def dp_seed_set_is_inferential(seeds: Sequence[int]) -> bool:
    """00 §36.2: the EXACT registered three-seed set, not any three seeds.

    `{101, 202, 404}` is three seeds and is not the DP set; a count is not a set.
    """
    return set(seeds) == set(DP_INFERENTIAL_SEEDS) and len(set(seeds)) == len(seeds)


def resolve_cell(
    cell: Cell, state: GateState, *, dp_seeds: Sequence[int] = (), schedule_resolved: bool = True
) -> ResolvedCell:
    """The current status of one registered row. Narrowing only."""
    selected = resolve_primary_lossy_operator(state)

    # 1 — STRUCTURAL_NA is permanent, and is checked before anything else could soften it.
    if cell.status == "STRUCTURAL_NA":
        return ResolvedCell(
            cell=cell,
            resolved_status="STRUCTURAL_NA",
            resolved_operator=cell.operator,
            runnable=False,
            reason=NOT_RUNNABLE_STRUCTURAL,
            detail=(
                f"{cell.cell_id} is an unknown-partner single-mixture cell: there is no"
                " quantity to recover, so it is permanently non-executable [AUTH: 00 §36.1]"
            ),
        )

    # 2 — an abstract LOSSY row needs the one selected operator.
    operator = cell.operator
    if operator == "LOSSY":
        if selected == OPERATOR_AXIS_INSUFFICIENT:
            return _gated(
                cell,
                "OPERATOR_GATED",
                operator,
                NOT_RUNNABLE_OPERATOR_INSUFFICIENT,
                "both operator gates failed; operator-specific claims are removed rather than"
                " rescued [AUTH: 00 §24A.7, §36]",
            )
        if selected == UNRESOLVED:
            return _gated(
                cell,
                "OPERATOR_GATED",
                operator,
                NOT_RUNNABLE_OPERATOR_UNRESOLVED,
                "the primary lossy operator is not yet selected; it is frozen before operator"
                " privacy outcomes are opened [AUTH: 00 §36]",
            )
        operator = selected

    # 3 — CONDITIONAL rows: O3 and a passing P0-C4, or METHOD_GATED under DARE.
    resolved_status = cell.status
    if cell.status == "CONDITIONAL":
        if selected == DARE:
            return _gated(
                cell,
                "METHOD_GATED",
                operator,
                NOT_RUNNABLE_METHOD_GATED,
                "DARE is the selected operator and no DARE-compatible incomplete-lineage"
                " recovery method is frozen [AUTH: 00 §25, §36]",
            )
        if not conditional_cells_active(state):
            return _gated(
                cell,
                "CONDITIONAL",
                operator,
                NOT_RUNNABLE_CONDITIONAL_GATED,
                "O3 is selected but P0-C4 has not passed; the truncated-source recovery model"
                " is not validated [AUTH: 00 §25, §36]",
            )
        resolved_status = "RUN_REPRODUCTION"

    # 4 — the known-varying-alpha lineage waits on the fixed P0-C2 baseline.
    if cell.lineage == VARYING_LINEAGE and not varying_alpha_activated(state):
        return _gated(
            cell,
            resolved_status,
            operator,
            NOT_RUNNABLE_VARYING_GATED,
            "known-varying-alpha cells activate only after the fixed common-source baseline"
            " P0-C2 is reproduced under the frozen implementation [AUTH: 00 §36]",
        )

    # 5 — a DP row is smoke-only until the EXACT registered seed set is available.
    if cell.regime == "REPRESENTATIVE_DP":
        if not dp_seed_set_is_inferential(dp_seeds):
            return _gated(
                cell,
                "DP_SMOKE_ONLY",
                operator,
                NOT_RUNNABLE_DP_SMOKE,
                f"the DP seed set {sorted(set(dp_seeds))} is not the registered"
                f" {sorted(DP_INFERENTIAL_SEEDS)}; a one-seed result never becomes"
                " inferential [AUTH: 00 §36.2]",
            )
        after = cell.status_after_dp_seed_set or "DP_SMOKE_ONLY"
        if after == "OPERATOR_GATED" and selected in (UNRESOLVED, OPERATOR_AXIS_INSUFFICIENT):
            return _gated(
                cell,
                "OPERATOR_GATED",
                operator,
                NOT_RUNNABLE_OPERATOR_UNRESOLVED,
                "the primary lossy operator is not yet selected [AUTH: 00 §36]",
            )
        #: The registered after-3-seed status, verbatim. Promoting an OPERATOR_GATED DP row
        #: to RUN_CONTROL would invent a status 00 §36.2 never gave it; the satisfied gate is
        #: reported through `runnable`, not by renaming the row.
        resolved_status = after

    # 6 — the row's registered schedule must resolve to actual coefficients.
    if cell.schedule is not None and not schedule_resolved:
        return _gated(
            cell,
            resolved_status,
            operator,
            NOT_RUNNABLE_SCHEDULE,
            f"the {cell.schedule!r} coefficient schedule did not resolve; a cell whose"
            " coefficients are unknown is not runnable [AUTH: 01 §17]",
        )

    return ResolvedCell(
        cell=cell,
        resolved_status=resolved_status,
        resolved_operator=operator,
        runnable=True,
        reason=RUNNABLE,
        detail="every registered precondition for this row is satisfied",
    )


def _gated(cell: Cell, status: str, operator: str, reason: str, detail: str) -> ResolvedCell:
    return ResolvedCell(
        cell=cell,
        resolved_status=status,
        resolved_operator=operator,
        runnable=False,
        reason=reason,
        detail=detail,
    )


def require_runnable(resolved: ResolvedCell) -> ResolvedCell:
    """The guard EXECUTE runs. A non-runnable row is refused, never forced."""
    if not resolved.runnable:
        raise ResolutionError(
            f"{resolved.cell.cell_id} is not runnable: {resolved.reason} — {resolved.detail}."
            " No flag overrides a registered gate"
        )
    return resolved
