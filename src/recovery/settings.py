"""Resolved-config access for the S08 recovery engine [AUTH: 01 §17; 00 §25, §26].

The files are named here; their values are read through `src.materials`, which carries the
calibration-status contract Block C froze. One key is `REQUIRED_NOT_CALIBRATED` on purpose:
`naive_unrescaled_assumed_alpha`, because 00 §25 names the comparator without freezing any
coefficient-estimation rule. Reading it fails closed, so no S08 code can invent one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from src.analysis.settings import ResolvedSettings, load_settings
from src.provenance.config import resolved_config_sha256
from src.provenance.hashing import JSONValue

RECOVERY_CONFIG: Final = "recovery/methods.json"


def recovery_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, RECOVERY_CONFIG)


def recovery_document_settings(document: Mapping[str, JSONValue]) -> ResolvedSettings:
    """Wrap an already-resolved snapshot so the material accessors can read it.

    `load_recovery_execution_context` takes the snapshot rather than a path, so the context
    hash covers exactly the document consumed.
    """
    return ResolvedSettings(
        name=RECOVERY_CONFIG,
        document=dict(document),
        sha256=resolved_config_sha256(dict(document)),
    )


def coefficient_schedule(settings: ResolvedSettings, key: str) -> tuple[float, ...]:
    """One registered 00 §25 P0-C2B.1 coefficient schedule."""
    from src.materials import material

    value = material(settings, key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return tuple(float(str(item)) for item in value)


def descendant_counts(settings: ResolvedSettings) -> tuple[int, ...]:
    from src.materials import material_integers

    return material_integers(settings, "descendant_counts")
