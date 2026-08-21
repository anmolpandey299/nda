"""Shared fixtures. Tests never touch the network and never download a model
[AUTH: 01 §20, §39 S00]."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the lane marker implied by the directory [AUTH: 01 §22]."""
    lanes = {"unit", "integration", "synthetic", "golden", "backend_contract", "gpu_smoke"}
    for item in items:
        for part in Path(str(item.fspath)).parts:
            if part in lanes:
                item.add_marker(getattr(pytest.mark, part))
                break
