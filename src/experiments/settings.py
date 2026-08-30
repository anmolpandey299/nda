"""Resolved-config access for the S09 registry [AUTH: 01 §17].

One path, Block A's. The registry is *data*: nothing here re-implements resolution, and no
registry value has a source default that could shadow the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.analysis.settings import ResolvedSettings, load_settings
from src.provenance.hashing import sha256_canonical

P1_REGISTRY_CONFIG: Final = "experiments/p1_registry.json"
P0_REGISTRY_CONFIG: Final = "experiments/p0_registry.json"
SCHEDULES_CONFIG: Final = "experiments/schedules.json"


def p1_registry_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, P1_REGISTRY_CONFIG)


def p0_registry_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, P0_REGISTRY_CONFIG)


def schedule_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, SCHEDULES_CONFIG)


def registry_sha256(root: Path) -> str:
    """One identity over every registry document the launcher consumes [AUTH: 00 §36].

    All three are hashed together because a plan depends on all three: the cell row, the
    coefficient schedule it names, and the P0 stage machine that gates it. A change to any of
    them is a different registry, and no run made under one may be attributed to another.
    """
    return sha256_canonical(
        {
            "p1_registry": dict(p1_registry_settings(root).document),
            "p0_registry": dict(p0_registry_settings(root).document),
            "schedules": dict(schedule_settings(root).document),
        }
    )
