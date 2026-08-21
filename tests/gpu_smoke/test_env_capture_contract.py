"""Acceptance for the S00-B environment capture [AUTH: 01 §12(2)-(9), §15, §16; plan §5.6].

Skips with an explicit reason while no environment manifest exists, so the lane reports
NOT_RUN rather than a state it never evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from preflight import (
    TBD,
    environment_identity_status,
    environment_lock_sha256,
    validate_environment_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = REPO_ROOT / "manifests" / "environments"


def _captured() -> list[Path]:
    if not ENV_DIR.is_dir():
        return []
    return [p for p in sorted(ENV_DIR.glob("*.json")) if p.name != "AI_ENGINEERING_STACK_S00.json"]


requires_capture = pytest.mark.skipif(
    not _captured(),
    reason="NOT_RUN(NO_GPU): no environment manifest; run `make env-capture` on the H100 "
    "image [AUTH: 01 §12; 03 §8]",
)


def test_env_capture_lane_is_not_empty() -> None:
    assert ENV_DIR.parent.is_dir()


@requires_capture
def test_captured_manifest_is_schema_complete() -> None:
    for path in _captured():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert validate_environment_manifest(manifest) == [], path.name


@requires_capture
def test_identity_is_recomputed_not_trusted() -> None:
    """The declared identity must equal the identity recomputed from the components."""
    for path in _captured():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["environment_lock_sha256"] == environment_lock_sha256(manifest)


@requires_capture
def test_no_component_remains_unresolved() -> None:
    identity, problems = environment_identity_status(REPO_ROOT)
    assert identity != TBD, f"S00-B incomplete: {problems}"
