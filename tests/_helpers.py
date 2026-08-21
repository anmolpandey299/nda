"""Fixture builders for provenance-bound evidence. No network, no model download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from check_repo_invariants import ENVIRONMENT_LOCK_COMPONENTS, RUN_MANIFEST_REQUIRED_FIELDS
from preflight import EVIDENCE_GATE_REL, EVIDENCE_LANES_REL, environment_lock_sha256

RUN_ID = "a" * 64


def write_environment_manifest(root: Path) -> str:
    """A fully resolved environment manifest whose declared identity is the recomputed one."""
    manifest: dict[str, object] = {k: f"resolved-{k}" for k in ENVIRONMENT_LOCK_COMPONENTS}
    manifest.update(
        {
            "nvidia_smi_capture": "fixture",
            "gpu_uuid": "GPU-fixture",
            "gpu_count": "1",
            "dependency_versions": "fixture",
            "bf16_fp32_tolerance": "1e-2",
            "nondeterminism_sources": "none",
            "capture_timestamp_utc": "2026-08-21T00:00:00Z",
        }
    )
    identity = environment_lock_sha256(manifest)
    manifest["environment_lock_sha256"] = identity
    env_dir = root / "manifests" / "environments"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / f"{identity}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return identity


def write_run_manifest(
    root: Path, run_id: str, env_lock: str, *, drop: tuple[str, ...] = ()
) -> str:
    manifest: dict[str, object] = {f: f"fixture-{f}" for f in RUN_MANIFEST_REQUIRED_FIELDS}
    manifest["run_id"] = run_id
    manifest["environment_lock_sha256"] = env_lock
    for field in drop:
        manifest.pop(field, None)
    runs = root / "manifests" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    rel = f"manifests/runs/{run_id}.json"
    (root / rel).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return rel


def write_artifact(root: Path, rel: str, payload: bytes = b"artifact") -> tuple[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return rel, hashlib.sha256(payload).hexdigest()


def write_evidence(root: Path, namespace: str, key: str, record: object) -> Path:
    base = root / (EVIDENCE_LANES_REL if namespace == "lanes" else EVIDENCE_GATE_REL)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{key}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def valid_record(root: Path, env_lock: str, *, run_id: str = RUN_ID) -> dict[str, object]:
    """A record that passes every FIX-2 check. Used as the positive control so the negative
    fixtures prove rejection rather than blanket refusal."""
    manifest_rel = write_run_manifest(root, run_id, env_lock)
    art_rel, art_sha = write_artifact(root, "artifacts/p0_pre/fixture.bin")
    return {
        "status": "PASS",
        "run_id": run_id,
        "environment_lock_sha256": env_lock,
        "run_manifest": manifest_rel,
        "artifact_sha256": {art_rel: art_sha},
    }
