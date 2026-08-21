"""S00 acceptance tests A5, A6, A10.

Multiple modules with tiny fixtures, no network [AUTH: 01 §22 Integration; plan §14].
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from preflight import (
    PREFLIGHT_STEPS,
    DirtyProductionTreeError,
    GateOutcome,
    Step,
    StepResult,
    assert_clean_production_tree,
    compute_readiness,
    production_tree_dirty,
    run_steps,
)

MAKE_TARGETS = (
    "format",
    "lint",
    "typecheck",
    "unit",
    "integration",
    "synthetic",
    "backend-contract",
    "gpu-smoke",
    "env-capture",
    "preflight",
)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], repo)


# ------------------------------------------------------------------ A5
def test_a5_every_target_exists(repo_root: Path) -> None:
    for target in MAKE_TARGETS:
        res = _run(["make", "-n", target], repo_root)
        assert res.returncode == 0, f"make {target} is not defined: {res.stderr}"


def test_a5_synthetic_lane_reports_not_run_never_pass(repo_root: Path) -> None:
    res = _run(["make", "synthetic"], repo_root)
    assert res.returncode == 0
    assert "NOT_RUN(EMPTY_AT_S00)" in res.stdout, res.stdout


def test_a5_backend_contract_reports_not_run(repo_root: Path) -> None:
    res = _run(["make", "backend-contract"], repo_root)
    assert res.returncode == 0
    assert "NOT_RUN(BACKEND_NOT_INTEGRATED)" in res.stdout, res.stdout
    assert "passed" not in res.stdout.lower()


def test_a5_gpu_smoke_reports_not_run_without_hardware(repo_root: Path) -> None:
    if shutil.which("nvidia-smi"):
        pytest.skip("H100 present; the NOT_RUN(NO_GPU) branch is not the live one")
    res = _run(["make", "gpu-smoke"], repo_root)
    assert res.returncode == 0
    assert "NOT_RUN(NO_GPU)" in res.stdout, res.stdout


def test_a5_env_capture_fails_while_hardware_is_unresolved(repo_root: Path) -> None:
    """An unresolved environment must never look captured [AUTH: 03 §8; 01 §12]."""
    if shutil.which("nvidia-smi"):
        pytest.skip("H100 present")
    res = _run(
        [
            "bash",
            "scripts/capture_environment.sh",
            "--out",
            str(repo_root / "artifacts" / "p0_pre" / "_env_probe"),
        ],
        repo_root,
    )
    assert res.returncode != 0
    assert "TBD_REQUIRES_HARDWARE" in res.stderr
    shutil.rmtree(repo_root / "artifacts" / "p0_pre" / "_env_probe", ignore_errors=True)


# ------------------------------------------------------------------ A6
def test_a6_step_order_matches_authority() -> None:
    """unit -> integration -> synthetic/golden, after the §33 gate [AUTH: 01 §22, §33]."""
    names = [s.name for s in PREFLIGHT_STEPS]
    assert names == [
        "repo-invariants",
        "format",
        "lint",
        "typecheck",
        "unit",
        "integration",
        "synthetic",
        "verify-evidence",
        "write-readiness",
    ]
    assert [s.index for s in PREFLIGHT_STEPS] == list(range(9))


def test_a6_halts_at_first_failure() -> None:
    def runner(step: Step) -> StepResult:
        return StepResult(step, step.index != 3)

    outcome: GateOutcome = run_steps(PREFLIGHT_STEPS, runner)
    assert outcome.halted_at == 3
    assert not outcome.ok
    assert [r.step.index for r in outcome.results] == [0, 1, 2, 3]


def test_a6_preflight_emits_readiness_false(repo_root: Path) -> None:
    readiness = compute_readiness(repo_root)
    assert readiness["P0_PRE_READY"] is False
    assert readiness["BACKEND_INTEGRATED"] is False
    written = json.loads(
        (repo_root / "artifacts/p0_pre/P0_PRE_READINESS.json").read_text(encoding="utf-8")
    )
    assert written["P0_PRE_READY"] is False


def test_a6_environment_mismatch_cannot_raise_a_flag(tmp_path: Path) -> None:
    """Evidence bound to a different environment identity is NON_EVIDENTIARY
    [AUTH: 01 §16; 02 §C6; C-01/S00-CBR-003 closure]."""
    env_dir = tmp_path / "manifests" / "environments"
    env_dir.mkdir(parents=True)
    (env_dir / "e.json").write_text(
        json.dumps({"environment_lock_sha256": "c" * 64}), encoding="utf-8"
    )
    runs = tmp_path / "manifests" / "runs"
    runs.mkdir(parents=True)
    (runs / "r.json").write_text("{}", encoding="utf-8")
    art = tmp_path / "artifacts" / "p0_pre"
    (art / "evidence").mkdir(parents=True)
    (art / "x.bin").write_bytes(b"x")
    import hashlib

    digest = hashlib.sha256(b"x").hexdigest()
    record = {
        "status": "PASS",
        "run_id": "a" * 64,
        "environment_lock_sha256": "d" * 64,  # <- different environment
        "run_manifest": "manifests/runs/r.json",
        "artifact_sha256": {"artifacts/p0_pre/x.bin": digest},
    }
    (art / "evidence" / "backend_contract.json").write_text(json.dumps(record), encoding="utf-8")
    readiness = compute_readiness(tmp_path)
    assert readiness["BACKEND_INTEGRATED"] is False
    assert "mismatch" in str(readiness["evidence"])


# ------------------------------------------------------------------ A10
def _temp_repo(tmp_path: Path, repo_root: Path) -> Path:
    repo = tmp_path / "clone"
    (repo / "src").mkdir(parents=True)
    (repo / "reviews" / "S00" / "scratch").mkdir(parents=True)
    shutil.copytree(repo_root / ".githooks", repo / ".githooks")
    shutil.copy(repo_root / "scripts" / "install_git_hooks.sh", repo / "install_git_hooks.sh")
    _git(repo, "init", "-b", "stage/test", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "core.hooksPath=", "commit", "-q", "-m", "root")
    return repo


def test_a10_live_repo_is_a_pinned_worktree(repo_root: Path) -> None:
    """§8.1 precondition: the reviewed revision must be identifiable [AUTH: 03 §9; 01 §26]."""
    head = _git(repo_root, "rev-parse", "HEAD")
    assert head.returncode == 0, "repository is not a Git worktree"
    assert len(head.stdout.strip()) == 40


def test_a10_hook_installer_sets_hooks_path(tmp_path: Path, repo_root: Path) -> None:
    repo = _temp_repo(tmp_path, repo_root)
    _run(["bash", "install_git_hooks.sh"], repo)
    assert _git(repo, "config", "--get", "core.hooksPath").stdout.strip() == ".githooks"


def test_a10_protected_branch_commit_is_blocked(tmp_path: Path, repo_root: Path) -> None:
    repo = _temp_repo(tmp_path, repo_root)
    _run(["bash", "install_git_hooks.sh"], repo)
    _git(repo, "checkout", "-q", "-b", "main")
    (repo / "src" / "mod.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    res = _git(repo, "commit", "-m", "should fail")
    assert res.returncode != 0
    assert "protected" in (res.stdout + res.stderr)


def test_a10_reviewer_code_under_reviews_is_blocked(tmp_path: Path, repo_root: Path) -> None:
    repo = _temp_repo(tmp_path, repo_root)
    _run(["bash", "install_git_hooks.sh"], repo)
    (repo / "reviews" / "S00" / "diag.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    res = _git(repo, "commit", "-m", "reviewer code")
    assert res.returncode != 0
    assert "reviews/" in (res.stdout + res.stderr)


def test_a10_scratch_diagnostics_are_permitted(tmp_path: Path, repo_root: Path) -> None:
    repo = _temp_repo(tmp_path, repo_root)
    _run(["bash", "install_git_hooks.sh"], repo)
    (repo / "reviews" / "S00" / "scratch" / "diag.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-m", "scratch diagnostic").returncode == 0


def test_a10_hooks_are_bypassable_so_they_are_only_a_convenience(
    tmp_path: Path, repo_root: Path
) -> None:
    """C-09 closure: --no-verify and an unset core.hooksPath both bypass the guard, which is
    why branch/tag protection and I15 are the authority mechanisms [AUTH: 03 §9; 01 §27]."""
    repo = _temp_repo(tmp_path, repo_root)
    _run(["bash", "install_git_hooks.sh"], repo)
    _git(repo, "checkout", "-q", "-b", "main")

    (repo / "src" / "mod.py").write_text("x = 3\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "--no-verify", "-m", "bypass").returncode == 0

    _git(repo, "config", "--unset", "core.hooksPath")
    (repo / "src" / "mod.py").write_text("x = 4\n", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-m", "unhooked clone").returncode == 0


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores mode bits")
def test_a10_reviewer_write_to_production_path_fails(tmp_path: Path, repo_root: Path) -> None:
    """A reviewer worktree is read-only outside scratch/ [AUTH: 03 §9]."""
    repo = _temp_repo(tmp_path, repo_root)
    production = repo / "src"
    production.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            (production / "injected.py").write_text("x = 1\n", encoding="utf-8")
        (repo / "reviews" / "S00" / "scratch" / "note.md").write_text("ok", encoding="utf-8")
    finally:
        production.chmod(0o755)


# ------------------------------------------------------------------ I15 foundation
def test_i15_detects_staged_unstaged_and_untracked(tmp_path: Path, repo_root: Path) -> None:
    """S00-CBR-002 closure: all three dirty classes under production paths
    [AUTH: 01 §35(3), §16, §27, §36]."""
    repo = _temp_repo(tmp_path, repo_root)
    assert production_tree_dirty(repo) == []
    assert_clean_production_tree(repo)

    (repo / "src" / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    assert any(e.startswith("??") for e in production_tree_dirty(repo))

    (repo / "src" / "mod.py").write_text("x = 99\n", encoding="utf-8")
    assert any("src/mod.py" in e for e in production_tree_dirty(repo))

    _git(repo, "add", "src/mod.py")
    assert any(e.startswith("M ") for e in production_tree_dirty(repo))

    with pytest.raises(DirtyProductionTreeError):
        assert_clean_production_tree(repo)


def test_i15_ignores_non_production_paths(tmp_path: Path, repo_root: Path) -> None:
    repo = _temp_repo(tmp_path, repo_root)
    (repo / "reviews" / "S00" / "scratch" / "note.md").write_text("scratch", encoding="utf-8")
    assert production_tree_dirty(repo) == []
    assert_clean_production_tree(repo)


def test_i15_refuses_when_not_a_worktree(tmp_path: Path) -> None:
    with pytest.raises(DirtyProductionTreeError):
        production_tree_dirty(tmp_path)
