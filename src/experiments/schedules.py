"""Registered coefficient schedules [AUTH: 00 §25, §36.3; 01 §17].

A registry row names a schedule; it never carries a coefficient vector, and no caller may
supply one. That separation is the point: `--alpha 0.2 0.8` on cell C08 would silently define a
fourth schedule nobody registered.

The `fixed` schedule is a **pre-result design completion**. 00 §36 registers the fixed-alpha
lineage without assigning a numeric coefficient, and `configs/merge/operators.json` states that
the registered coefficient schedule is S09's rather than S07's — so no accepted value existed
to bind. It is completed here as alpha = 0.50, before any real scientific outcome has been
opened: symmetric equal-parent weighting, which privileges neither the protected constituent
nor the partner and avoids the coefficient amplification an extreme alpha would introduce into
the rescaling identity. It is **not** a value from the closed measurement spec, and the config
records that provenance explicitly [AUTH: 00 §36.3; 01 §17].

`fixed` carries no k of its own: it resolves deterministically to alpha_i = 0.50 for every one
of a cell's k descendants, so one registered schedule serves k in {1, 2, 4}.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src.provenance.hashing import JSONValue, sha256_canonical

SCHEDULE_SCHEMA: Final = "s09.coefficient-schedule.v1"

FIXED: Final = "fixed"
K2_REFERENCE: Final = "k2-reference"
SPREAD_MATCHED: Final = "spread-matched"
WIDE_SPREAD: Final = "wide-spread"

REGISTERED_SCHEDULES: Final[tuple[str, ...]] = (FIXED, K2_REFERENCE, SPREAD_MATCHED, WIDE_SPREAD)

#: How the `fixed` coefficient came to have a value. Recorded so nobody later reads 0.50 as a
#: number the closed measurement spec assigned.
PRE_RESULT_DESIGN_COMPLETION: Final = "PRE_RESULT_DESIGN_COMPLETION"

#: The completed fixed coefficient, applied to every descendant of a fixed-schedule cell.
FIXED_ALPHA: Final = 0.5


class ScheduleError(ValueError):
    """A coefficient schedule cannot be resolved as registered."""


class ScheduleNotCalibratedError(ScheduleError):
    """A registered schedule carries no frozen coefficients [AUTH: 01 §17].

    No schedule is in this state today; the class stays so a future schedule added without a
    value fails closed rather than resolving to something convenient.
    """


@dataclass(frozen=True)
class Schedule:
    """One registered schedule: a name, a descendant count and its exact coefficients."""

    name: str
    k: int
    alpha: tuple[float, ...]
    provenance: str | None = None

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": SCHEDULE_SCHEMA,
            "name": self.name,
            "k": self.k,
            "alpha": list(self.alpha),
            "provenance": self.provenance,
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())


def _validate(name: str, k: int, alpha: tuple[float, ...]) -> None:
    import math

    if len(alpha) != k:
        raise ScheduleError(
            f"{name}: the schedule declares k = {k} but carries {len(alpha)} coefficients"
        )
    for value in alpha:
        if not math.isfinite(value):
            raise ScheduleError(f"{name}: coefficient {value!r} is not finite")
        if not 0.0 < value <= 1.0:
            raise ScheduleError(f"{name}: coefficient {value!r} is not in (0, 1]")


def resolve_schedule(root: Path, name: str, *, k: int | None = None) -> Schedule:
    """The registered schedule, or a refusal. Never a caller-supplied vector.

    `k` is required only for `fixed`, which registers one scalar coefficient rather than a
    vector and expands it across a cell's descendants. For the varying schedules the count is
    part of the registration, and a supplied `k` that disagrees with it is a refusal.
    """
    from src.experiments.settings import schedule_settings
    from src.materials import material, material_status

    if name not in REGISTERED_SCHEDULES:
        raise ScheduleError(f"{name!r} is not a registered schedule {list(REGISTERED_SCHEDULES)}")
    settings = schedule_settings(root)
    if material_status(settings, name) == "REQUIRED_NOT_CALIBRATED":
        raise ScheduleNotCalibratedError(
            f"the {name!r} schedule is registered but its coefficients are not frozen; S09"
            " reports the dependency rather than inventing a coefficient [AUTH: 01 §17]"
        )
    value = material(settings, name)
    if not isinstance(value, Mapping):
        raise ScheduleError(f"{name}: the registered schedule is not an object")
    provenance = None if value.get("provenance") is None else str(value["provenance"])

    if value.get("per_descendant"):
        # One scalar coefficient, applied identically to every descendant.
        if k is None:
            raise ScheduleError(
                f"{name}: a per-descendant schedule expands across a cell's k, which the"
                " caller did not supply"
            )
        alpha = (float(str(value.get("alpha"))),) * k
        _validate(name, k, alpha)
        return Schedule(name=name, k=k, alpha=alpha, provenance=provenance)

    raw = value.get("alpha")
    if not isinstance(raw, list):
        raise ScheduleError(f"{name}: the registered schedule carries no alpha vector")
    registered_k = int(str(value.get("k")))
    if k is not None and k != registered_k:
        raise ScheduleError(f"{name}: the schedule registers k = {registered_k}, not {k}")
    alpha = tuple(float(str(item)) for item in raw)
    _validate(name, registered_k, alpha)
    return Schedule(name=name, k=registered_k, alpha=alpha, provenance=provenance)


def schedule_for_cell(root: Path, cell: Any) -> Schedule | None:
    """The schedule a registered cell names, checked against the cell's own k.

    `None` when the row registers no schedule at all — the two k = 1 varying-alpha rows, which
    are STRUCTURAL_NA and have no coefficients to resolve [AUTH: 00 §36.1].
    """
    name = cell.schedule
    if name is None:
        return None
    schedule = resolve_schedule(root, name, k=cell.k)
    if schedule.k != cell.k:  # pragma: no cover - resolve_schedule already refused
        raise ScheduleError(
            f"{cell.cell_id}: the row declares k = {cell.k} but names the {name!r} schedule,"
            f" which is k = {schedule.k}"
        )
    return schedule
