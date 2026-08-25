"""S01 derived repository, config and environment facts [AUTH: 01 §12, §15, §32, §35(3)].

`src/` cannot import `scripts/` (the science image installs no project, so `scripts/*` is
importable only when a script is run directly). These tests are the binding that keeps the
two definitions from drifting: same production paths, same environment components, same
environment hash, same dirty-tree verdict.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import check_repo_invariants
import preflight
import pytest

from src.provenance.repository import (
    ENVIRONMENT_LOCK_COMPONENTS,
    TBD,
    RepositoryError,
    environment_lock_sha256,
    head_commit,
    is_environment_lock_manifest,
    is_repository,
    production_tree_dirty,
    selected_environment_identity,
    verify_environment_identity,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "init", "-b", "stage/test", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "-c", "core.hooksPath=", "commit", "-q", "-m", "one")
    return root


# ------------------------------------------------------------------ one definition, two homes
def test_production_paths_match_the_s00_gate() -> None:
    from src.provenance.repository import PRODUCTION_PATHS

    assert PRODUCTION_PATHS == preflight.PRODUCTION_PATHS


def test_environment_components_match_the_invariant_checker() -> None:
    assert ENVIRONMENT_LOCK_COMPONENTS == check_repo_invariants.ENVIRONMENT_LOCK_COMPONENTS


def test_environment_hash_matches_the_s00_scheme() -> None:
    manifest: dict[str, object] = {key: f"resolved-{key}" for key in ENVIRONMENT_LOCK_COMPONENTS}
    assert environment_lock_sha256(manifest) == preflight.environment_lock_sha256(manifest)
    incomplete: dict[str, object] = {**manifest, "cuda_driver": TBD}
    assert environment_lock_sha256(incomplete) == TBD
    assert preflight.environment_lock_sha256(incomplete) == TBD


def test_dirty_verdict_matches_the_s00_gate(repo: Path) -> None:
    assert production_tree_dirty(repo) == preflight.production_tree_dirty(repo) == []
    (repo / "src" / "module.py").write_text("x = 2\n", encoding="utf-8")
    assert production_tree_dirty(repo) == preflight.production_tree_dirty(repo) != []


def test_the_live_repository_agrees_with_the_s00_gate(repo_root: Path) -> None:
    assert production_tree_dirty(repo_root) == preflight.production_tree_dirty(repo_root)


# ------------------------------------------------------------------ derivation
def test_head_commit_is_read_from_the_repository(repo: Path) -> None:
    assert head_commit(repo) == _git(repo, "rev-parse", "HEAD").strip()


def test_a_non_repository_cannot_supply_facts(tmp_path: Path) -> None:
    assert is_repository(tmp_path / "nowhere") is False
    with pytest.raises(RepositoryError):
        head_commit(tmp_path / "nowhere")


def test_untracked_production_files_count_as_dirty(repo: Path) -> None:
    (repo / "configs").mkdir()
    (repo / "configs" / "new.json").write_text("{}", encoding="utf-8")
    assert any("configs/new.json" in entry for entry in production_tree_dirty(repo))


def test_changes_outside_production_paths_do_not_count(repo: Path) -> None:
    (repo / "logs").mkdir()
    (repo / "logs" / "run.log").write_text("noise\n", encoding="utf-8")
    assert production_tree_dirty(repo) == []


# ------------------------------------------------------------------ environment identity
def _capture(root: Path, **overrides: str) -> str:
    manifest: dict[str, object] = {key: f"resolved-{key}" for key in ENVIRONMENT_LOCK_COMPONENTS}
    manifest.update(overrides)
    identity = environment_lock_sha256(manifest)
    manifest["environment_lock_sha256"] = identity
    directory = root / "manifests" / "environments"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{identity}.json").write_text(json.dumps(manifest), encoding="utf-8")
    return identity


def test_a_captured_identity_verifies(tmp_path: Path) -> None:
    identity = _capture(tmp_path)
    assert verify_environment_identity(tmp_path, identity) == []


def test_a_uv_lock_only_digest_is_not_an_environment_identity(tmp_path: Path) -> None:
    """The exact reviewer counterexample: a lockfile digest is 64 hex characters and nothing
    else [AUTH: 01 §12(5)-(9), §15]."""
    import hashlib

    _capture(tmp_path)
    uv_lock_digest = hashlib.sha256(b"uv.lock bytes").hexdigest()
    problems = verify_environment_identity(tmp_path, uv_lock_digest)
    assert any("not a full environment identity" in p for p in problems), problems


def test_a_tampered_capture_is_rejected(tmp_path: Path) -> None:
    identity = _capture(tmp_path)
    path = tmp_path / "manifests" / "environments" / f"{identity}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["torch_version"] = "2.99.0"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    problems = verify_environment_identity(tmp_path, identity)
    assert any("recompute" in p for p in problems), problems


def test_an_unresolved_capture_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "manifests" / "environments"
    directory.mkdir(parents=True)
    identity = "a" * 64
    manifest = {key: TBD for key in ENVIRONMENT_LOCK_COMPONENTS}
    manifest["environment_lock_sha256"] = identity
    (directory / f"{identity}.json").write_text(json.dumps(manifest), encoding="utf-8")
    problems = verify_environment_identity(tmp_path, identity)
    assert any("unresolved" in p for p in problems), problems


@pytest.mark.parametrize("value", ["", "not-hex", "abc", TBD])
def test_a_malformed_identity_is_rejected(tmp_path: Path, value: str) -> None:
    assert verify_environment_identity(tmp_path, value) != []


# ------------------------------------------------------------------ ENV_SELECTED_NOT_MERELY_VALID
def test_the_selection_rule_matches_the_frozen_s00_rule(tmp_path: Path) -> None:
    """`selected_environment_identity` must be `environment_identity_status`, exactly."""
    assert selected_environment_identity(tmp_path) == preflight.environment_identity_status(
        tmp_path
    )
    _capture(tmp_path)
    _capture(tmp_path, python_version="3.13.9")
    assert selected_environment_identity(tmp_path) == preflight.environment_identity_status(
        tmp_path
    )


def test_the_live_repository_selection_matches(repo_root: Path) -> None:
    assert selected_environment_identity(repo_root) == preflight.environment_identity_status(
        repo_root
    )


def test_the_lock_manifest_predicate_matches(tmp_path: Path) -> None:
    assert is_environment_lock_manifest(Path(f"{'a' * 64}.json"))
    assert not is_environment_lock_manifest(Path("S00B_HARDWARE_PROBE.json"))
    assert is_environment_lock_manifest(
        Path(f"{'a' * 64}.json")
    ) is preflight.is_environment_lock_manifest(Path(f"{'a' * 64}.json"))


def test_only_the_selected_environment_verifies(tmp_path: Path) -> None:
    """Two internally valid captures; the unselected one is refused [AUTH: 01 §12, §32]."""
    first = _capture(tmp_path)
    second = _capture(tmp_path, python_version="3.13.9")
    selected, problems = selected_environment_identity(tmp_path)
    assert problems == [] and selected == min(first, second)

    assert verify_environment_identity(tmp_path, selected) == []
    rejected = verify_environment_identity(tmp_path, max(first, second))
    assert any("not the selected environment" in p for p in rejected), rejected


def test_an_internally_valid_capture_alone_is_not_enough(tmp_path: Path) -> None:
    """The manifest recomputes and agrees with itself, yet is not this tree's environment."""
    identity = _capture(tmp_path, python_version="3.13.9")
    assert (
        environment_lock_sha256(
            json.loads(
                (tmp_path / "manifests" / "environments" / f"{identity}.json").read_text("utf-8")
            )
        )
        == identity
    )
    _capture(tmp_path, python_version="3.13.0")
    selected, _ = selected_environment_identity(tmp_path)
    if selected != identity:
        assert verify_environment_identity(tmp_path, identity) != []
