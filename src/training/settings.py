"""Resolved-config access for the S06 training, DP and model-panel constants.

00 §8.3 requires the LoRA and DP values to be written into the experiment configuration and
hashed, without fixing them numerically. They therefore reach execution through
`src.materials`, which refuses an uncalibrated constant instead of defaulting it
[AUTH: 01 §17; 00 §8.3].
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.analysis.settings import ResolvedSettings, load_settings

LORA_CONFIG: Final = "training/lora.json"
DP_CONFIG: Final = "training/dp.json"
PANEL_CONFIG: Final = "models/panel.json"


def lora_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, LORA_CONFIG)


def dp_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, DP_CONFIG)


def panel_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, PANEL_CONFIG)
