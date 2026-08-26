"""Material constants with an explicit calibration status [AUTH: 01 §17; 00 §8.3].

01 §17 puts material constants in `configs/**`. 00 §8.3 additionally requires a set of LoRA
and DP values to be "written into the experiment configuration file and hashed" without
fixing them numerically anywhere. Those two facts together create a third state that a plain
JSON number cannot express: a constant that is *required*, is *not yet calibrated*, and must
not be silently substituted by a plausible default.

So a material value is a small object carrying its status:

```json
"lora_scaling": {"status": "PROVISIONAL_FIXTURE_ONLY", "value": 64.0, "note": "..."}
```

* `FROZEN` — fixed by an authority document; usable anywhere.
* `PROVISIONAL_FIXTURE_ONLY` — exists so tiny-fixture tests can run; a caller must ask for it
  explicitly, and an evidentiary path never gets it.
* `REQUIRED_NOT_CALIBRATED` — carries no value at all and always fails closed.

Resolution itself is Block A's. This module only interprets the resolved values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from src.analysis.settings import ResolvedSettings
from src.provenance.config import ConfigError
from src.provenance.hashing import JSONValue

FROZEN: Final = "FROZEN"
PROVISIONAL_FIXTURE_ONLY: Final = "PROVISIONAL_FIXTURE_ONLY"
REQUIRED_NOT_CALIBRATED: Final = "REQUIRED_NOT_CALIBRATED"

MATERIAL_STATUSES: Final[tuple[str, ...]] = (
    FROZEN,
    PROVISIONAL_FIXTURE_ONLY,
    REQUIRED_NOT_CALIBRATED,
)


class UncalibratedConstantError(ConfigError):
    """A constant required before the real run has not been calibrated yet."""


def material(settings: ResolvedSettings, key: str, *, allow_provisional: bool = False) -> JSONValue:
    """The value behind a material key, or a refusal explaining which state it is in."""
    entry = settings.document.get(key)
    if not isinstance(entry, Mapping):
        raise ConfigError(
            f"{settings.name}: {key!r} is not a material value object; every material"
            " constant must declare its calibration status [AUTH: 01 §17; 00 §8.3]"
        )
    status = entry.get("status")
    if status not in MATERIAL_STATUSES:
        raise ConfigError(f"{settings.name}: {key!r} declares unknown status {status!r}")
    if status == REQUIRED_NOT_CALIBRATED:
        raise UncalibratedConstantError(
            f"{settings.name}: {key!r} is REQUIRED_NOT_CALIBRATED — it must be frozen in"
            f" configs/{settings.name} before any run that depends on it"
            f" ({entry.get('note', 'no note')}) [AUTH: 00 §8.3]"
        )
    if status == PROVISIONAL_FIXTURE_ONLY and not allow_provisional:
        raise UncalibratedConstantError(
            f"{settings.name}: {key!r} is PROVISIONAL_FIXTURE_ONLY and may be used by"
            " fixture code only; an evidentiary path must wait for the frozen value"
        )
    if "value" not in entry:
        raise ConfigError(f"{settings.name}: {key!r} declares a status but carries no value")
    value = entry["value"]
    if isinstance(value, Mapping) and "status" in value:
        raise ConfigError(
            f"{settings.name}: {key!r} nests another material object; a status may not be"
            " declared twice for one constant"
        )
    return value


def material_status(settings: ResolvedSettings, key: str) -> str:
    entry = settings.document.get(key)
    if not isinstance(entry, Mapping) or entry.get("status") not in MATERIAL_STATUSES:
        raise ConfigError(f"{settings.name}: {key!r} declares no calibration status")
    return str(entry["status"])


def material_number(
    settings: ResolvedSettings, key: str, *, allow_provisional: bool = False
) -> float:
    value = material(settings, key, allow_provisional=allow_provisional)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{settings.name}: {key!r} is not a number")
    return float(value)


def material_integer(
    settings: ResolvedSettings, key: str, *, allow_provisional: bool = False
) -> int:
    value = material(settings, key, allow_provisional=allow_provisional)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{settings.name}: {key!r} is not an integer")
    return value


def material_text(settings: ResolvedSettings, key: str, *, allow_provisional: bool = False) -> str:
    value = material(settings, key, allow_provisional=allow_provisional)
    if not isinstance(value, str):
        raise ConfigError(f"{settings.name}: {key!r} is not a string")
    return value


def material_integers(
    settings: ResolvedSettings, key: str, *, allow_provisional: bool = False
) -> tuple[int, ...]:
    value = material(settings, key, allow_provisional=allow_provisional)
    if isinstance(value, str) or not isinstance(value, list):
        raise ConfigError(f"{settings.name}: {key!r} is not a list")
    return tuple(int(str(item)) for item in value)


def material_mapping(
    settings: ResolvedSettings, key: str, *, allow_provisional: bool = False
) -> dict[str, JSONValue]:
    value = material(settings, key, allow_provisional=allow_provisional)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{settings.name}: {key!r} is not an object")
    return dict(value)


def uncalibrated_keys(settings: ResolvedSettings) -> tuple[str, ...]:
    """Every key still waiting for a frozen value. Reported, never defaulted."""
    return tuple(
        sorted(
            key
            for key, entry in settings.document.items()
            if isinstance(entry, Mapping) and entry.get("status") == REQUIRED_NOT_CALIBRATED
        )
    )
