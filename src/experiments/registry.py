"""The P1 cell registry [AUTH: 00 §36, §36.1, §36.2; 01 §17, §39 S09].

S09's locked role is to encode the registry as data and resolve it. **The launcher does not
invent cells.** A caller names a registered cell id; the scientific combination — operator, k,
lineage, schedule — comes from the row and from nowhere else. There is no API through which a
new combination can be assembled, which is why `Cell` is opaque, factory-issued and has no
constructor a caller can reach: an object that could be built field by field would be a way to
define an experiment at the command line.

The status vocabulary is a closed enum. 00 §36 lists exactly eight values, and anything else —
`ACTIVE`, `ENABLED`, `TODO`, `AUTO` — is rejected at load, because a generic status is a
scientific claim nobody registered.

The registry is 29 rows: 19 calibration and 10 representative-DP. Adding one is not a
launch-time operation; it changes `registry_sha256` and requires a pre-result revision or a
`SPEC_DEVIATION` outside ordinary launcher use [AUTH: 00 §36].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from src.provenance.hashing import JSONValue, sha256_canonical

CELL_SCHEMA: Final = "s09.registry-cell.v1"

#: 00 §36's closed status vocabulary. Exactly these, and nothing else.
REGISTRY_STATUSES: Final[frozenset[str]] = frozenset(
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

#: 00 §36: k in {1, 2, 4}. k = 8 is deferred and is not runnable.
REGISTERED_K: Final[frozenset[int]] = frozenset({1, 2, 4})

#: 00 §36: the three authorised lineage regimes. Hidden alpha is not part of P1.
REGISTERED_LINEAGES: Final[tuple[str, ...]] = (
    "Z-HIGH",
    "Z-INCOMPLETE-FIXED",
    "Z-INCOMPLETE-VARYING-KNOWN",
)
FORBIDDEN_LINEAGES: Final[tuple[str, ...]] = ("Z-INCOMPLETE-ALPHA-HIDDEN",)

#: The abstract operator axis. `LOSSY` resolves to the ONE selected primary lossy operator.
REGISTERED_OPERATORS: Final[tuple[str, ...]] = ("LINEAR", "LOSSY")
FORBIDDEN_OPERATORS: Final[tuple[str, ...]] = ("TIES", "SLERP", "QUANTIZATION", "DARE", "O3")

REGIMES: Final[tuple[str, ...]] = ("CALIBRATION", "REPRESENTATIVE_DP")

#: The registered training seeds. 404 and 505 are confirmatory-only expansion seeds and are
#: deliberately not part of the ordinary P0/P1 registry [AUTH: 00 §9].
CALIBRATION_SEEDS: Final[tuple[int, ...]] = (101, 202, 303)
DP_SMOKE_SEED: Final = 101
DP_INFERENTIAL_SEEDS: Final[frozenset[int]] = frozenset({101, 202, 303})

EXPECTED_CALIBRATION_ROWS: Final = 19
EXPECTED_DP_ROWS: Final = 10
EXPECTED_TOTAL_ROWS: Final = 29
EXPECTED_STRUCTURAL_NA: Final = 4


class RegistryError(ValueError):
    """The registry cannot be loaded, or a cell was requested that it does not contain."""


class UnregisteredCellError(RegistryError):
    """A cell id that the frozen registry does not contain [AUTH: 00 §36]."""


class Cell:
    """One registered P1 cell. Opaque, factory-issued, immutable.

    `__init__` raises and this is not a dataclass, so no `dataclasses.replace` and no ordinary
    construction can mint a row with a different operator, k, lineage or schedule. The only
    way to obtain one is to look up a registered id.
    """

    __slots__ = (
        "_cell_id",
        "_identity",
        "_interpretation",
        "_k",
        "_lineage",
        "_operator",
        "_regime",
        "_schedule",
        "_status",
        "_status_after_dp_seed_set",
    )

    _cell_id: str
    _regime: str
    _operator: str
    _k: int
    _lineage: str
    _schedule: str | None
    _status: str
    _status_after_dp_seed_set: str | None
    _interpretation: str
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "Cell is registry-issued; look one up by its registered id. A constructible cell"
            " would be a way to define a new experiment at the command line [AUTH: 00 §36]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a registry cell is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a registry cell is immutable")

    # ---------------------------------------------------------------- projections
    @property
    def cell_id(self) -> str:
        return self._cell_id

    @property
    def regime(self) -> str:
        return self._regime

    @property
    def operator(self) -> str:
        """`LINEAR`, or the abstract `LOSSY` that a resolved gate state makes concrete."""
        return self._operator

    @property
    def k(self) -> int:
        return self._k

    @property
    def lineage(self) -> str:
        return self._lineage

    @property
    def schedule(self) -> str | None:
        return self._schedule

    @property
    def status(self) -> str:
        """The registry's static status. Gate resolution may narrow it, never widen it."""
        return self._status

    @property
    def status_after_dp_seed_set(self) -> str | None:
        """DP rows only: what the row becomes once the exact 3-seed DP set is available."""
        return self._status_after_dp_seed_set

    @property
    def interpretation(self) -> str:
        return self._interpretation

    def identity(self) -> str:
        """`cell_definition_sha256`, over the scientific definition alone."""
        return self._identity

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": CELL_SCHEMA,
            "cell_id": self._cell_id,
            "regime": self._regime,
            "operator": self._operator,
            "k": self._k,
            "lineage": self._lineage,
            "schedule": self._schedule,
            "status": self._status,
            "status_after_dp_seed_set": self._status_after_dp_seed_set,
            "interpretation": self._interpretation,
            "cell_definition_sha256": self._identity,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Cell({self._cell_id}, {self._operator}, k={self._k}, {self._lineage})"


def _issue_cell(row: Mapping[str, Any]) -> Cell:
    definition: dict[str, JSONValue] = {
        "schema": CELL_SCHEMA,
        "cell_id": str(row["cell_id"]),
        "regime": str(row["regime"]),
        "operator": str(row["operator"]),
        "k": int(row["k"]),
        "lineage": str(row["lineage"]),
        "schedule": None if row.get("schedule") is None else str(row["schedule"]),
        "status": str(row["status"]),
        "status_after_dp_seed_set": (
            None
            if row.get("status_after_dp_seed_set") is None
            else str(row["status_after_dp_seed_set"])
        ),
    }
    cell = object.__new__(Cell)
    object.__setattr__(cell, "_cell_id", definition["cell_id"])
    object.__setattr__(cell, "_regime", definition["regime"])
    object.__setattr__(cell, "_operator", definition["operator"])
    object.__setattr__(cell, "_k", definition["k"])
    object.__setattr__(cell, "_lineage", definition["lineage"])
    object.__setattr__(cell, "_schedule", definition["schedule"])
    object.__setattr__(cell, "_status", definition["status"])
    object.__setattr__(cell, "_status_after_dp_seed_set", definition["status_after_dp_seed_set"])
    object.__setattr__(cell, "_interpretation", str(row.get("interpretation", "")))
    object.__setattr__(cell, "_identity", sha256_canonical(definition))
    return cell


class Registry:
    """The 29 frozen rows, plus the identity that binds a plan to this exact document."""

    __slots__ = ("_cells", "_identity", "_version")

    _cells: tuple[Cell, ...]
    _version: str
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Registry is factory-issued; use load_registry()")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("the registry is immutable")

    @property
    def cells(self) -> tuple[Cell, ...]:
        return self._cells

    @property
    def version(self) -> str:
        return self._version

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(cell.cell_id for cell in self._cells)

    def identity(self) -> str:
        return self._identity

    def calibration(self) -> tuple[Cell, ...]:
        return tuple(c for c in self._cells if c.regime == "CALIBRATION")

    def representative_dp(self) -> tuple[Cell, ...]:
        return tuple(c for c in self._cells if c.regime == "REPRESENTATIVE_DP")

    def cell(self, cell_id: str) -> Cell:
        """The registered row, or a refusal. There is no fall-through that builds one."""
        for candidate in self._cells:
            if candidate.cell_id == cell_id:
                return candidate
        raise UnregisteredCellError(
            f"{cell_id!r} is not a registered P1 cell. The registry holds"
            f" {len(self._cells)} rows and the launcher does not construct new ones; adding a"
            " cell requires a pre-result registry revision or a SPEC_DEVIATION"
            " [AUTH: 00 §36]"
        )


def _row_problems(row: Mapping[str, Any], *, regime: str) -> list[str]:
    """Every reason this row is not a valid registered cell."""
    problems: list[str] = []
    cell_id = str(row.get("cell_id", ""))
    if not cell_id:
        return ["a registry row carries no cell_id"]
    if str(row.get("regime")) != regime:
        problems.append(f"{cell_id}: regime {row.get('regime')!r} is not {regime}")

    operator = str(row.get("operator"))
    if operator not in REGISTERED_OPERATORS:
        problems.append(
            f"{cell_id}: operator {operator!r} is not a registered axis value"
            f" {list(REGISTERED_OPERATORS)}; a concrete lossy operator is resolved from the"
            " gate state, never written into a row [AUTH: 00 §36]"
        )

    k = row.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k not in REGISTERED_K:
        problems.append(f"{cell_id}: k {k!r} is not in {sorted(REGISTERED_K)} [AUTH: 00 §11]")

    lineage = str(row.get("lineage"))
    if lineage in FORBIDDEN_LINEAGES:
        problems.append(
            f"{cell_id}: lineage {lineage!r} is not part of P1; hidden coefficients are"
            " excluded [AUTH: 00 §36.3]"
        )
    elif lineage not in REGISTERED_LINEAGES:
        problems.append(f"{cell_id}: lineage {lineage!r} is not registered")

    status = str(row.get("status"))
    if status not in REGISTRY_STATUSES:
        problems.append(
            f"{cell_id}: status {status!r} is not in the closed 00 §36 vocabulary"
            f" {sorted(REGISTRY_STATUSES)}"
        )
    after = row.get("status_after_dp_seed_set")
    if after is not None and str(after) not in REGISTRY_STATUSES:
        problems.append(f"{cell_id}: status_after_dp_seed_set {after!r} is not registered")
    if regime == "REPRESENTATIVE_DP":
        if status != "DP_SMOKE_ONLY":
            problems.append(
                f"{cell_id}: a DP row's pre-3-seed status is DP_SMOKE_ONLY [AUTH: 00 §36.2]"
            )
        if after is None:
            problems.append(f"{cell_id}: a DP row must declare its after-3-seed status")
    elif after is not None:
        problems.append(f"{cell_id}: only a DP row carries status_after_dp_seed_set")
    return problems


def load_registry(root: Path) -> Registry:
    """Load and validate the frozen registry. Structure is checked, never repaired."""
    from src.experiments.settings import p1_registry_settings, registry_sha256

    document = p1_registry_settings(root).document
    problems: list[str] = []
    rows: list[Mapping[str, Any]] = []
    for key, regime in (("calibration", "CALIBRATION"), ("representative_dp", "REPRESENTATIVE_DP")):
        entries = document.get(key)
        if not isinstance(entries, list):
            problems.append(f"the registry defines no {key!r} list")
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                problems.append(f"{key}: every registry row must be an object")
                continue
            problems += _row_problems(entry, regime=regime)
            rows.append(entry)

    calibration = [r for r in rows if str(r.get("regime")) == "CALIBRATION"]
    dp = [r for r in rows if str(r.get("regime")) == "REPRESENTATIVE_DP"]
    if len(calibration) != EXPECTED_CALIBRATION_ROWS:
        problems.append(
            f"the calibration registry has {len(calibration)} rows, not"
            f" {EXPECTED_CALIBRATION_ROWS} [AUTH: 00 §36.1]"
        )
    if len(dp) != EXPECTED_DP_ROWS:
        problems.append(
            f"the DP registry has {len(dp)} rows, not {EXPECTED_DP_ROWS} [AUTH: 00 §36.2]"
        )
    na = sum(1 for r in calibration if str(r.get("status")) == "STRUCTURAL_NA")
    if calibration and na != EXPECTED_STRUCTURAL_NA:
        problems.append(
            f"the calibration registry has {na} STRUCTURAL_NA rows, not"
            f" {EXPECTED_STRUCTURAL_NA} [AUTH: 00 §36.1]"
        )
    identifiers = [str(r.get("cell_id")) for r in rows]
    duplicates = sorted({i for i in identifiers if identifiers.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate cell id(s): {duplicates}")

    if problems:
        raise RegistryError("; ".join(problems))

    registry = object.__new__(Registry)
    object.__setattr__(registry, "_cells", tuple(_issue_cell(row) for row in rows))
    object.__setattr__(registry, "_version", str(document.get("registry_version", "")))
    object.__setattr__(registry, "_identity", registry_sha256(root))
    return registry


def refuse_scientific_override(**overrides: Any) -> None:
    """Refuse any attempt to supply a scientific dimension alongside a cell id.

    00 §36 fixes the combination; the caller chooses which registered combination to run, not
    what one is. Every value here would redefine the experiment rather than select it.
    """
    supplied = sorted(name for name, value in overrides.items() if value is not None)
    if supplied:
        raise RegistryError(
            f"the scientific combination comes from the registry row, not from the command"
            f" line: {supplied} may not be supplied alongside a cell id. Choose a registered"
            " cell id instead [AUTH: 00 §36]"
        )


def registered_seeds(regime: str) -> Sequence[int]:
    """The registered training seeds for one regime [AUTH: 00 §9]."""
    if regime == "CALIBRATION":
        return CALIBRATION_SEEDS
    if regime == "REPRESENTATIVE_DP":
        return tuple(sorted(DP_INFERENTIAL_SEEDS))
    raise RegistryError(f"{regime!r} is not a registered regime {list(REGIMES)}")
