"""Resolved-config access for the S05 corpus and canary constants [AUTH: 01 §17].

The files are named here; their values are read through `src.materials`, which is where the
calibration-status contract lives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.analysis.settings import ResolvedSettings, load_settings

CORPUS_CONFIG: Final = "data/corpus.json"
CANARY_CONFIG: Final = "data/canaries.json"


def corpus_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, CORPUS_CONFIG)


def canary_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, CANARY_CONFIG)
