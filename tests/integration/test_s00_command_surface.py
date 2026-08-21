"""S00 acceptance tests A5, A6, A10.

Multiple modules with tiny fixtures, no network [AUTH: 01 §22 Integration; plan §14].
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import build_bundle
import pytest
from _helpers import valid_record, write_environment_manifest, write_evidence
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
    [AUTH: 01 §16; 02 §C6; C-01 / S00-CBR-003 closure]."""
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    record["environment_lock_sha256"] = "d" * 64  # a different environment
    write_evidence(tmp_path, "lanes", "backend_contract", record)
    readiness = compute_readiness(tmp_path)
    assert readiness["BACKEND_INTEGRATED"] is False
    assert "mismatch" in str(readiness["evidence"])


def test_a6_matching_environment_is_accepted(tmp_path: Path) -> None:
    """Positive control, so the mismatch case above proves rejection rather than refusal."""
    env = write_environment_manifest(tmp_path)
    write_evidence(tmp_path, "lanes", "backend_contract", valid_record(tmp_path, env))
    assert compute_readiness(tmp_path)["BACKEND_INTEGRATED"] is True


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


# ------------------------------------------------------------------ FIX 1: both branches
def _readiness_fixture(tmp_path: Path, backend_integrated: bool) -> Path:
    path = tmp_path / "readiness.json"
    path.write_text(
        json.dumps(
            {
                "BACKEND_INTEGRATED": backend_integrated,
                "SUITE_SCOPE": "INTEGRATED_PRODUCTION_STACK"
                if backend_integrated
                else "STATISTICAL_STACK_ONLY",
                "P0_PRE_READY": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_fix1_branch_a_not_run_when_backend_absent(repo_root: Path, tmp_path: Path) -> None:
    res = _run(
        ["make", "backend-contract", f"READINESS={_readiness_fixture(tmp_path, False)}"], repo_root
    )
    assert res.returncode == 0
    assert "NOT_RUN(BACKEND_NOT_INTEGRATED)" in res.stdout
    assert "passed" not in res.stdout.lower()


def test_fix1_branch_b_actually_invokes_pytest(repo_root: Path, tmp_path: Path) -> None:
    """The old recipe expanded `$$READINESS` to a PID-prefixed literal, so this branch was
    unreachable and every run reported NOT_RUN [FIX 1]."""
    res = _run(
        ["make", "backend-contract", f"READINESS={_readiness_fixture(tmp_path, True)}"], repo_root
    )
    combined = res.stdout + res.stderr
    assert "NOT_RUN(BACKEND_NOT_INTEGRATED)" not in combined
    assert "pytest" in combined or "skipped" in combined.lower(), combined
    assert "tests/backend_contract" in combined or "skipped" in combined.lower()


def test_fix1_branch_c_unreadable_readiness_is_a_hard_failure(
    repo_root: Path, tmp_path: Path
) -> None:
    """A missing readiness file must not be silently swallowed into a NOT_RUN [FIX 1]."""
    res = _run(["make", "backend-contract", f"READINESS={tmp_path / 'absent.json'}"], repo_root)
    assert res.returncode != 0
    combined = res.stdout + res.stderr
    assert "NOT_RUN(BACKEND_NOT_INTEGRATED)" not in combined
    assert "not found" in combined or "unreadable" in combined


# ------------------------------------------------------------------ FIX 8: bundle exactness
def test_fix8_committed_bundle_describes_the_current_code(repo_root: Path) -> None:
    problems = build_bundle.verify(repo_root, "S00")
    assert problems == [], problems


def test_fix8_bundle_verify_target_passes(repo_root: Path) -> None:
    assert _run(["make", "bundle-verify"], repo_root).returncode == 0


def test_fix8_drift_after_the_described_commit_is_detected(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    (repo / "stage_acceptance" / "S00").mkdir(parents=True)
    _git(repo, "init", "-b", "stage/test", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "core.hooksPath=", "commit", "-q", "-m", "one")
    build_bundle.generate(repo, "S00")
    assert build_bundle.verify(repo, "S00") == []

    (repo / "src" / "mod.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "core.hooksPath=", "commit", "-q", "-m", "code drift")
    problems = build_bundle.verify(repo, "S00")
    assert any("stale" in p for p in problems), problems


def test_fix8_manifest_hash_tampering_is_detected(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init", "-b", "stage/test", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "core.hooksPath=", "commit", "-q", "-m", "one")
    build_bundle.generate(repo, "S00")
    manifest_path = repo / "stage_acceptance" / "S00" / "05_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["src/mod.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    assert any("hash mismatch" in p for p in build_bundle.verify(repo, "S00"))


# ------------------------------------------------------------------ FIX 9: reviewer worktree
def test_fix9_review_worktree_is_pinned_and_read_only(repo_root: Path, tmp_path: Path) -> None:
    """A blind reviewer gets a separate pinned checkout, writable only under scratch/
    [AUTH: 03 §9]. The primary worktree is never claimed to be read-only."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores mode bits")
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    dest = tmp_path / "review"
    created = _run(["bash", "scripts/make_review_worktree.sh", head, str(dest), "S00"], repo_root)
    try:
        assert created.returncode == 0, created.stderr
        assert head in created.stdout
        assert _git(dest, "rev-parse", "HEAD").stdout.strip() == head

        with pytest.raises(PermissionError):
            (dest / "src" / "injected.py").write_text("x = 1\n", encoding="utf-8")
        with pytest.raises(PermissionError):
            (dest / "Makefile").write_text("tampered\n", encoding="utf-8")

        note = dest / "reviews" / "S00" / "scratch" / "finding.md"
        note.write_text("reviewer diagnostic", encoding="utf-8")
        assert note.read_text(encoding="utf-8") == "reviewer diagnostic"
    finally:
        _run(["bash", "scripts/make_review_worktree.sh", "--remove", str(dest)], repo_root)
    assert not dest.exists()
