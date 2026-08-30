"""Gate state and the pure operator resolver [AUTH: 00 §24, §24A, §24A.7, §35, §36].

Two rules decide everything here.

**Gate state is never self-asserted.** There is no `--dare-pass`, no `--p0-c1-pass`, and no
CLI boolean of any kind: a scientific truth a user can type is not evidence. A gate record
must name its source artifact, that artifact's SHA256, the gate, its status, the RUN_ID(s) it
came from and the registry identity it was decided under. A hand-written `{"DARE": "PASS"}`
does not become authority merely because it parses.

**The operator is chosen by a rule, not by a result.** The resolver is pure:

    DARE PASS                  -> DARE
    DARE FAIL and O3 PASS      -> SVD_TRUNC_MERGE
    DARE FAIL and O3 FAIL      -> OPERATOR_AXIS_INSUFFICIENT
    otherwise                  -> UNRESOLVED

Nothing about effect size, recovery quality, interest or availability enters it, and the
selected operator is frozen before operator privacy outcomes are opened. Once a gate document
records `P1_PRIMARY_LOSSY_OPERATOR` for an evidentiary phase, a conflicting replacement is
refused: outcome-driven flipping is exactly what the freeze exists to prevent.

This module implements the validation interface and its fixture form. The real P0 evidence
that will populate it is produced later, not here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src.provenance.hashing import JSONDocument, JSONValue, sha256_canonical, sha256_file

GATE_SCHEMA: Final = "s09.gate-state.v1"

#: A gate record that carries this class is a deterministic fixture, never P0 evidence.
FIXTURE_CLASS: Final = "NON_EVIDENTIARY_FIXTURE"
EVIDENTIARY_CLASS: Final = "EVIDENTIARY_P0_GATE"
GATE_CLASSES: Final[tuple[str, ...]] = (FIXTURE_CLASS, EVIDENTIARY_CLASS)

PASS: Final = "PASS"
FAIL: Final = "FAIL"
UNRESOLVED: Final = "UNRESOLVED"
GATE_STATUSES: Final[tuple[str, ...]] = (PASS, FAIL, UNRESOLVED)

DARE: Final = "DARE"
SVD_TRUNC_MERGE: Final = "SVD_TRUNC_MERGE"
OPERATOR_AXIS_INSUFFICIENT: Final = "OPERATOR_AXIS_INSUFFICIENT"

#: The gates S09 resolves against. Each is decided by a 00 §35 stage, never by this module.
REGISTERED_GATES: Final[tuple[str, ...]] = (
    "P0-A",
    "P0-C1",
    "P0-F",
    "P0-C2",
    "P0-D",
    "P0-D2",
    "P0-D3",
    "P0-C4",
)

#: What every gate record must bind, so a status cannot float free of what decided it.
REQUIRED_GATE_FIELDS: Final[tuple[str, ...]] = (
    "gate",
    "status",
    "source_artifact_path",
    "source_artifact_sha256",
    "source_run_ids",
    "registry_sha256",
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class GateError(ValueError):
    """A gate-state document is malformed, unbound, or contradicts a frozen decision."""


class OperatorFrozenError(GateError):
    """A frozen primary lossy operator cannot be replaced [AUTH: 00 §36]."""


@dataclass(frozen=True)
class GateRecord:
    """One gate's decision, and what decided it."""

    gate: str
    status: str
    source_artifact_path: str
    source_artifact_sha256: str
    source_run_ids: tuple[str, ...]
    registry_sha256: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "gate": self.gate,
            "status": self.status,
            "source_artifact_path": self.source_artifact_path,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_run_ids": list(self.source_run_ids),
            "registry_sha256": self.registry_sha256,
        }


@dataclass(frozen=True)
class GateState:
    """Every resolved gate, plus any frozen primary lossy operator."""

    provenance_class: str
    records: Mapping[str, GateRecord]
    frozen_primary_lossy_operator: str | None = None

    @property
    def evidentiary(self) -> bool:
        return self.provenance_class == EVIDENTIARY_CLASS

    def status(self, gate: str) -> str:
        """A gate nobody has decided is UNRESOLVED, not absent and not assumed."""
        if gate not in REGISTERED_GATES:
            raise GateError(f"{gate!r} is not a registered gate {list(REGISTERED_GATES)}")
        record = self.records.get(gate)
        return record.status if record is not None else UNRESOLVED

    def identity(self) -> str:
        return sha256_canonical(
            {
                "schema": GATE_SCHEMA,
                "provenance_class": self.provenance_class,
                "gates": {name: r.as_dict() for name, r in sorted(self.records.items())},
                "frozen_primary_lossy_operator": self.frozen_primary_lossy_operator,
            }
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": GATE_SCHEMA,
            "provenance_class": self.provenance_class,
            "gates": {name: r.as_dict() for name, r in sorted(self.records.items())},
            "frozen_primary_lossy_operator": self.frozen_primary_lossy_operator,
            "gate_state_sha256": self.identity(),
        }


def _record_problems(entry: Mapping[str, Any], *, evidentiary: bool, root: Path) -> list[str]:
    problems: list[str] = []
    missing = sorted(set(REQUIRED_GATE_FIELDS) - set(entry))
    if missing:
        return [f"a gate record omits {missing}; a status must name what decided it"]
    gate = str(entry["gate"])
    if gate not in REGISTERED_GATES:
        problems.append(f"{gate!r} is not a registered gate")
    if str(entry["status"]) not in GATE_STATUSES:
        problems.append(f"{gate}: status {entry['status']!r} is not {list(GATE_STATUSES)}")
    for field in ("source_artifact_sha256", "registry_sha256"):
        if not _SHA256_RE.match(str(entry[field])):
            problems.append(f"{gate}: {field} is not a sha256 digest")
    runs = entry["source_run_ids"]
    if not isinstance(runs, list):
        problems.append(f"{gate}: source_run_ids must be a list")
    elif evidentiary:
        if not runs:
            problems.append(f"{gate}: an evidentiary gate names the RUN_ID(s) that decided it")
        for identifier in runs:
            if not _SHA256_RE.match(str(identifier)):
                problems.append(f"{gate}: source_run_ids entry {identifier!r} is not a RUN_ID")
    if evidentiary:
        # The artifact's bytes are re-hashed: a digest-shaped string proves nothing about a
        # file on disk [AUTH: 01 §16, §36].
        path = root / str(entry["source_artifact_path"])
        if not path.is_file():
            problems.append(f"{gate}: the named source artifact is missing: {path.name}")
        elif sha256_file(path) != str(entry["source_artifact_sha256"]):
            problems.append(
                f"{gate}: the source artifact hashes differently than the record claims;"
                " this decision was made against other bytes [AUTH: 01 §16]"
            )
    return problems


def load_gate_state(document: JSONDocument, *, root: Path) -> GateState:
    """Validate a gate-state document and issue it, or refuse.

    A fixture document is accepted for planning and testing but is marked
    `NON_EVIDENTIARY_FIXTURE` and can never be mistaken for P0 evidence.
    """
    provenance_class = str(document.get("provenance_class", ""))
    if provenance_class not in GATE_CLASSES:
        raise GateError(
            f"provenance_class {provenance_class!r} is not one of {list(GATE_CLASSES)}; a gate"
            " document must say plainly whether it is evidence or a fixture"
        )
    evidentiary = provenance_class == EVIDENTIARY_CLASS
    raw = document.get("gates")
    if not isinstance(raw, Mapping):
        raise GateError("the gate document defines no 'gates' object")

    problems: list[str] = []
    records: dict[str, GateRecord] = {}
    for name, entry in raw.items():
        if not isinstance(entry, Mapping):
            problems.append(f"{name}: every gate record must be an object")
            continue
        if str(entry.get("gate")) != str(name):
            problems.append(f"{name}: the record names gate {entry.get('gate')!r}")
        found = _record_problems(entry, evidentiary=evidentiary, root=root)
        problems += found
        if found:
            continue
        records[str(name)] = GateRecord(
            gate=str(entry["gate"]),
            status=str(entry["status"]),
            source_artifact_path=str(entry["source_artifact_path"]),
            source_artifact_sha256=str(entry["source_artifact_sha256"]),
            source_run_ids=tuple(str(r) for r in entry["source_run_ids"]),  # type: ignore[union-attr]
            registry_sha256=str(entry["registry_sha256"]),
        )

    frozen = document.get("frozen_primary_lossy_operator")
    if frozen is not None and str(frozen) not in (DARE, SVD_TRUNC_MERGE):
        problems.append(
            f"frozen_primary_lossy_operator {frozen!r} is not {DARE} or {SVD_TRUNC_MERGE}"
        )
    if problems:
        raise GateError("; ".join(problems))

    state = GateState(
        provenance_class=provenance_class,
        records=records,
        frozen_primary_lossy_operator=None if frozen is None else str(frozen),
    )
    # A frozen operator must agree with what the gates themselves resolve to.
    require_operator_consistent(state)
    return state


def resolve_primary_lossy_operator(state: GateState) -> str:
    """00 §36's rule, and nothing else.

    Deliberately reads only the two gate statuses. No effect size, recovery quality or
    reviewer preference has any representation in this function.
    """
    dare = state.status("P0-D")
    o3 = state.status("P0-D2")
    if dare == PASS:
        return DARE
    if dare == FAIL and o3 == PASS:
        return SVD_TRUNC_MERGE
    if dare == FAIL and o3 == FAIL:
        return OPERATOR_AXIS_INSUFFICIENT
    return UNRESOLVED


def require_operator_consistent(state: GateState) -> None:
    """A frozen operator may not contradict the gates, and may not be replaced.

    Both directions matter: a document that freezes DARE while its gates resolve to O3 is
    describing a decision the rule did not make, and a later request to swap a frozen operator
    is outcome-driven flipping [AUTH: 00 §36].
    """
    frozen = state.frozen_primary_lossy_operator
    if frozen is None:
        return
    resolved = resolve_primary_lossy_operator(state)
    if resolved in (UNRESOLVED, OPERATOR_AXIS_INSUFFICIENT):
        raise OperatorFrozenError(
            f"the document freezes {frozen} while its gates resolve to {resolved}; the"
            " selected operator follows the 00 §36 rule, not a declaration"
        )
    if resolved != frozen:
        raise OperatorFrozenError(
            f"the document freezes {frozen} while its gates select {resolved}. A frozen"
            " primary lossy operator cannot be replaced after the fact [AUTH: 00 §36]"
        )


def require_no_operator_reversal(previous: GateState, proposed: GateState) -> None:
    """Refuse a replacement of an already-frozen primary lossy operator."""
    was = previous.frozen_primary_lossy_operator
    now = proposed.frozen_primary_lossy_operator
    if was is None or now is None or was == now:
        return
    raise OperatorFrozenError(
        f"{was} is already frozen as the primary lossy operator; replacing it with {now}"
        " would be an outcome-driven change [AUTH: 00 §36]"
    )


def fixture_gate_state(
    *,
    statuses: Mapping[str, str],
    registry_sha256: str,
    frozen_primary_lossy_operator: str | None = None,
) -> dict[str, JSONValue]:
    """A deterministic NON_EVIDENTIARY_FIXTURE gate document, for planning and tests.

    It carries the same bindings a real record must, so the validation interface is exercised
    rather than bypassed — but its class says plainly that it is not P0 evidence.
    """
    gates: dict[str, JSONValue] = {}
    for gate, status in statuses.items():
        gates[gate] = {
            "gate": gate,
            "status": status,
            "source_artifact_path": f"fixtures/gates/{gate}.json",
            "source_artifact_sha256": sha256_canonical({"fixture-gate": gate, "status": status}),
            "source_run_ids": [],
            "registry_sha256": registry_sha256,
        }
    return {
        "schema": GATE_SCHEMA,
        "provenance_class": FIXTURE_CLASS,
        "gates": gates,
        "frozen_primary_lossy_operator": frozen_primary_lossy_operator,
    }


def unresolved_gate_state(registry_sha256: str) -> GateState:
    """The current real state: nothing has been decided [AUTH: 03 §8]."""
    del registry_sha256
    return GateState(provenance_class=FIXTURE_CLASS, records={})


def registered_gate_names() -> Sequence[str]:
    return REGISTERED_GATES
