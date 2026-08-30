"""Deterministic S09 fixtures. No network, no model, no training, no H100.

Gate documents built here carry `NON_EVIDENTIARY_FIXTURE`, so nothing produced from them can
be mistaken for P0 evidence [AUTH: 00 §35; 03 §8].
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.gates import GateState, fixture_gate_state, load_gate_state
from src.experiments.settings import registry_sha256
from src.provenance.identity import ScientificProvenance
from src.provenance.repository import ENVIRONMENT_LOCK_COMPONENTS, environment_lock_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A model alias from the locked core panel; no revision is resolved anywhere here.
FIXTURE_MODEL_ALIAS = "llama_3_2_3b"
TRAINING_SEED = 101

ACCEPTED_ENVIRONMENT: dict[str, str] = {
    "uv_lock_sha256": "1" * 64,
    "docker_image_digest": "sha256:" + "2" * 64,
    "cuda_runtime": "13.0",
    "cuda_driver": "580.126.09",
    "torch_version": "2.13.0",
    "torch_cuda_build": "cu130",
    "python_version": "3.13.2",
    "gpu_model": "NVIDIA H100 80GB HBM3",
}
ACCEPTED_GPU_UUID = "GPU-1f2e3d4c-5b6a-7988-9a0b-1c2d3e4f5061"

SEED_SET: dict[str, int] = {"python_rng": 101, "numpy_rng": 101, "data_order": 7}


def gate_state(**statuses: str) -> GateState:
    """A validated NON_EVIDENTIARY_FIXTURE gate state."""
    document = fixture_gate_state(
        statuses={k.replace("_", "-"): v for k, v in statuses.items()},
        registry_sha256=registry_sha256(REPO_ROOT),
    )
    return load_gate_state(document, root=REPO_ROOT)


def frozen_operator_state(operator: str, **statuses: str) -> GateState:
    """A fixture state that also freezes the primary lossy operator."""
    document = fixture_gate_state(
        statuses={k.replace("_", "-"): v for k, v in statuses.items()},
        registry_sha256=registry_sha256(REPO_ROOT),
        frozen_primary_lossy_operator=operator,
    )
    return load_gate_state(document, root=REPO_ROOT)


def unresolved_state() -> GateState:
    """The honest current state: no P0 outcome has been opened."""
    return GateState(provenance_class="NON_EVIDENTIARY_FIXTURE", records={})


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _write_environment_manifest(root: Path, **overrides: str) -> str:
    manifest: dict[str, Any] = {**ACCEPTED_ENVIRONMENT, "gpu_uuid": ACCEPTED_GPU_UUID}
    manifest.update(overrides)
    assert set(ENVIRONMENT_LOCK_COMPONENTS) <= set(manifest)
    identity = environment_lock_sha256(manifest)
    manifest["environment_lock_sha256"] = identity
    directory = root / "manifests" / "environments"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{identity}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return identity


@dataclass(frozen=True)
class Workspace:
    """A disposable repository whose facts are real enough to be derived, not asserted.

    The S09 development tree is necessarily dirty, so evidentiary behaviour is exercised here
    instead of from the working branch [AUTH: 01 §35(3), §36].
    """

    root: Path
    provenance: ScientificProvenance
    environment_lock_sha256: str


def build_workspace(destination: Path, *, model_revision: str = "e" * 40) -> Workspace:
    """A committed fixture repo carrying the registry configs and an environment capture."""
    import shutil

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "configs", destination / "configs", dirs_exist_ok=True)
    _git(destination, "init", "-b", "stage/s09-fixture", "-q")
    _git(destination, "config", "user.email", "fixture@example.invalid")
    _git(destination, "config", "user.name", "fixture")
    _git(destination, "add", "-A")
    _git(destination, "-c", "core.hooksPath=", "commit", "-q", "-m", "s09 fixture root")
    head = _git(destination, "rev-parse", "HEAD").strip()

    environment = _write_environment_manifest(destination)
    provenance = ScientificProvenance(
        git_commit_sha=head,
        spec_sha256="b" * 64,
        execution_lock_sha256="c" * 64,
        config_sha256="d" * 64,
        model_revision=model_revision,
        data_manifest_sha256="f" * 64,
        environment_lock_sha256=environment,
        training_seed=TRAINING_SEED,
    )
    return Workspace(root=destination, provenance=provenance, environment_lock_sha256=environment)


def provenance_for(
    workspace: Workspace, resolved_config: Mapping[str, Any]
) -> ScientificProvenance:
    """The workspace provenance rebound to a plan's resolved-config identity."""
    from dataclasses import replace

    from src.provenance.config import resolved_config_sha256

    return replace(
        workspace.provenance, config_sha256=resolved_config_sha256(dict(resolved_config))
    )


def harmless_argv(message: str = "s09-fixture") -> tuple[str, ...]:
    """A deterministic local command. No shell, no model, no network."""
    import sys

    return (sys.executable, "-c", f"print({message!r})")


def failing_argv() -> tuple[str, ...]:
    import sys

    return (sys.executable, "-c", "import sys; sys.stderr.write('planned failure\\n'); sys.exit(3)")
