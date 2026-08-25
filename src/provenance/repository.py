"""Repository, config and environment facts DERIVED from the repository, never asserted.

An evidentiary run manifest is a claim about what actually executed. If the caller supplies
the commit, the dirty state, the config identity and the environment identity, then a run
that never touched those things can still finalise SUCCESS and verify clean: the record is
internally consistent and externally false [AUTH: 01 §15, §16, §35(3), §36].

This module derives those facts instead. It exists as its own module because `src/` cannot
import `scripts/`: the science image copies the repository to /repo and installs no project
(`uv sync --no-install-project`), so `src.*` resolves from the working directory while
`scripts/*` is importable only when a script is run directly. The S00 definitions in
`scripts/preflight.py` and `scripts/check_repo_invariants.py` therefore cannot be reused by
import, and are instead bound to these by equivalence tests, so the two cannot drift apart.

The environment-hash scheme is S00's, unchanged: SHA256 over `key=value` lines for the eight
01 §12(5)-(9) components, in the frozen order [AUTH: 01 §12, §15, §32].
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

#: Paths whose modification changes what a run means [AUTH: 01 §35(3), §16, §27, §36].
#: Bound by test to scripts/preflight.py's copy.
PRODUCTION_PATHS: Final[tuple[str, ...]] = (
    "src/",
    "configs/",
    "specs/",
    "pyproject.toml",
    "uv.lock",
    "Dockerfile",
)

#: Components of ENVIRONMENT_LOCK_SHA256, in the frozen order [AUTH: 01 §12(5)-(9), §15].
#: Bound by test to scripts/check_repo_invariants.py's copy.
ENVIRONMENT_LOCK_COMPONENTS: Final[tuple[str, ...]] = (
    "uv_lock_sha256",
    "docker_image_digest",
    "cuda_runtime",
    "cuda_driver",
    "torch_version",
    "torch_cuda_build",
    "python_version",
    "gpu_model",
)

TBD: Final = "TBD_REQUIRES_HARDWARE"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")


class RepositoryError(RuntimeError):
    """A repository fact could not be derived, so it may not be assumed."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RepositoryError(f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}")
    return result.stdout


def is_repository(root: Path) -> bool:
    try:
        _git(root, "rev-parse", "--git-dir")
    except RepositoryError:
        return False
    return True


def head_commit(root: Path) -> str:
    """The commit a run is actually executing, read from the repository [AUTH: 01 §15]."""
    value = _git(root, "rev-parse", "HEAD").strip()
    if not _GIT_SHA_RE.match(value):
        raise RepositoryError(f"HEAD is not a full 40-hex commit SHA: {value!r}")
    return value


def production_tree_dirty(root: Path) -> list[str]:
    """Staged, unstaged and untracked changes under production paths.

    Same predicate as the S00 gate: `git status --porcelain -uall` reports all three classes,
    so a run cannot be attributed to a clean commit while the scorer, configs or environment
    differ from it [AUTH: 01 §35(3), §16, §27, §36].
    """
    dirty: list[str] = []
    for line in _git(root, "status", "--porcelain=v1", "-uall").splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:].strip().strip('"')
        path = path.split(" -> ")[-1]
        if any(path == p.rstrip("/") or path.startswith(p) for p in PRODUCTION_PATHS):
            dirty.append(f"{status} {path}")
    return sorted(dirty)


def environment_lock_sha256(manifest: Mapping[str, object]) -> str:
    """S00's environment identity, recomputed. Unresolved components yield TBD, never a hash."""
    parts: list[str] = []
    for key in ENVIRONMENT_LOCK_COMPONENTS:
        value = manifest.get(key)
        if not isinstance(value, str) or value == TBD or not value:
            return TBD
        parts.append(f"{key}={value}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def environment_manifest_path(root: Path, identity: str) -> Path:
    """A capture writes `manifests/environments/<ENVIRONMENT_LOCK_SHA256>.json` [AUTH: 01 §12]."""
    return root / "manifests" / "environments" / f"{identity}.json"


def is_environment_lock_manifest(path: Path) -> bool:
    """A capture writes `<ENVIRONMENT_LOCK_SHA256>.json`; anything else there is metadata.

    Reproduces the frozen S00 predicate exactly [AUTH: 01 §12; 02 §C6].
    """
    return bool(_SHA256_RE.match(path.stem)) and path.suffix == ".json"


def selected_environment_identity(root: Path) -> tuple[str, list[str]]:
    """THE environment for this tree, under the frozen S00 selection semantics.

    Reproduces `scripts/preflight.py::environment_identity_status` exactly: the environment
    manifests are considered in sorted filename order and the first one that is a lock
    manifest, parses, recomputes from its components and agrees with its own declared
    identity is the selected environment. Everything after it is a historical capture, not
    the current one.

    This distinction is the whole point: proving a claimed manifest is *internally valid* is
    not proving it is *the environment this execution ran in*, and two internally valid
    captures can coexist [AUTH: 01 §12(5)-(9), §15, §32; 02 §C6].
    """
    problems: list[str] = []
    directory = root / "manifests" / "environments"
    if not directory.is_dir():
        return TBD, ["no manifests/environments directory"]
    for path in sorted(directory.glob("*.json")):
        if not is_environment_lock_manifest(path):
            continue
        name = path.name
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{name}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(manifest, Mapping):
            problems.append(f"{name}: not a JSON object")
            continue
        recomputed = environment_lock_sha256(manifest)
        if recomputed == TBD:
            problems.append(f"{name}: components unresolved")
            continue
        if manifest.get("environment_lock_sha256") != recomputed:
            problems.append(f"{name}: declared identity does not match the recomputed identity")
            continue
        return recomputed, problems
    return TBD, problems


def read_environment_manifest(root: Path, identity: str) -> Mapping[str, object] | None:
    path = environment_manifest_path(root, identity)
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return manifest if isinstance(manifest, Mapping) else None


def verify_environment_identity(root: Path, identity: str) -> list[str]:
    """Every reason `identity` is not the accepted environment of this tree.

    A 64-hex string is not evidence, and neither is an internally consistent manifest: the
    digest must name a captured manifest whose eight components recompute to exactly it AND
    be the identity the frozen S00 selection rule independently picks. Otherwise a run can
    attribute itself to a valid historical capture that is not the environment it ran in
    [AUTH: 01 §12(5)-(9), §15, §32; 02 §C6].
    """
    if identity == TBD:
        return [f"environment identity is still {TBD}"]
    if not _SHA256_RE.match(identity):
        return ["environment identity is not a SHA256 digest"]
    path = environment_manifest_path(root, identity)
    if not path.is_file():
        return [
            f"no captured environment manifest for {identity[:12]};"
            " a bare digest is not a full environment identity"
        ]
    manifest = read_environment_manifest(root, identity)
    if manifest is None:
        return [f"{path.name}: not a readable JSON object"]

    problems: list[str] = []
    recomputed = environment_lock_sha256(manifest)
    if recomputed == TBD:
        problems.append(f"{path.name}: components are unresolved")
    elif recomputed != identity:
        problems.append(
            f"{path.name}: components recompute to {recomputed[:12]}, not {identity[:12]}"
        )
    declared = manifest.get("environment_lock_sha256")
    if declared != identity:
        problems.append(f"{path.name}: declared identity {str(declared)[:12]} disagrees")

    selected, selection_problems = selected_environment_identity(root)
    if selected == TBD:
        problems.append(
            "no environment is selected under the frozen S00 rule"
            + (f": {'; '.join(selection_problems)}" if selection_problems else "")
        )
    elif selected != identity:
        problems.append(
            f"environment {identity[:12]} is internally valid but is not the selected"
            f" environment {selected[:12]} [AUTH: 01 §12, §32; 02 §C6]"
        )
    return problems


#: Run-manifest field -> the accepted environment manifest field that fixes it. Only fields
#: S00 actually defines are compared; `precision` is deliberately absent because the
#: environment contract does not fix it [AUTH: 01 §12(5)-(9), §16].
RUNTIME_ATTRIBUTION_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("gpu_model", "gpu_model"),
    ("gpu_uuid", "gpu_uuid"),
    ("cuda", "cuda_runtime"),
)


def verify_runtime_attribution(
    root: Path, identity: str, recorded: Mapping[str, object]
) -> list[str]:
    """Hardware a run claims must be the hardware the accepted environment records.

    Without this a run may record `GPU-forged` while the accepted capture says `GPU-actual`,
    finalise SUCCESS and verify clean — a false physical-hardware attribution. Only an
    evidentiary run that actually claims hardware is checked; a run declaring
    NOT_APPLICABLE asserts no hardware and is left alone [AUTH: 01 §12, §16, §32].
    """
    manifest = read_environment_manifest(root, identity)
    if manifest is None:
        return [f"no readable environment manifest for {identity[:12]}"]

    problems: list[str] = []
    for run_field, environment_field in RUNTIME_ATTRIBUTION_FIELDS:
        claimed = recorded.get(run_field)
        accepted = manifest.get(environment_field)
        if not isinstance(accepted, str) or not accepted or accepted == TBD:
            problems.append(
                f"the accepted environment does not fix {environment_field!r},"
                f" so {run_field!r} cannot be attributed"
            )
        elif claimed != accepted:
            problems.append(
                f"{run_field} {str(claimed)!r} disagrees with the accepted environment's"
                f" {environment_field} {accepted!r} [AUTH: 01 §12, §16]"
            )

    problems += _torch_attribution(recorded.get("pytorch"), manifest)
    return problems


def _torch_attribution(claimed: object, manifest: Mapping[str, object]) -> list[str]:
    """`pytorch` must name the accepted torch version, optionally with its CUDA build."""
    version = manifest.get("torch_version")
    build = manifest.get("torch_cuda_build")
    if not isinstance(version, str) or not version or version == TBD:
        return ["the accepted environment does not fix 'torch_version'"]
    permitted = {version}
    if isinstance(build, str) and build and build != TBD:
        permitted.add(f"{version}+{build}")
    if claimed not in permitted:
        return [
            f"pytorch {str(claimed)!r} is not the accepted environment's torch"
            f" ({', '.join(sorted(permitted))}) [AUTH: 01 §12, §16]"
        ]
    return []
