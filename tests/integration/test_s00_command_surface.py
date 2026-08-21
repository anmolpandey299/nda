"""S00 acceptance tests A5, A6, A10.

Multiple modules with tiny fixtures, no network [AUTH: 01 §22 Integration; plan §14].
"""

from __future__ import annotations

import ast
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
    TBD,
    DirtyProductionTreeError,
    GateOutcome,
    Step,
    StepResult,
    assert_clean_production_tree,
    compute_readiness,
    production_tree_dirty,
    record_lane_evidence,
    run_steps,
    write_readiness,
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
    """Immutable half only. The live repository may be mid-run: capture has produced a new
    environment identity and readiness is not re-derived until preflight step 8, so asserting
    the runtime half here is a circular ordering dependency. The runtime half is covered by
    controlled fixtures below and by the bootstrap's final gate."""
    problems = build_bundle.verify_source(repo_root, "S00")
    assert problems == [], problems


def test_fix8_bundle_verify_source_target_passes(repo_root: Path) -> None:
    assert _run(["make", "bundle-verify-source"], repo_root).returncode == 0


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
    manifest["source_artifacts"]["src/mod.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    assert any("source artifact hash mismatch" in p for p in build_bundle.verify(repo, "S00"))


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


# ------------------------------------------------------------------ S00-B bootstrap
def test_s00b_bootstrap_fails_closed_on_a_wrong_commit(repo_root: Path) -> None:
    """The first gate is the exact required commit; everything after it must not run."""
    res = _run(["bash", "scripts/bootstrap_runpod_s00b.sh", "0" * 40], repo_root)
    combined = res.stdout + res.stderr
    assert res.returncode != 0
    assert "S00B_BOOTSTRAP = FAIL" in combined
    assert "!= required" in combined
    assert "S00B_BOOTSTRAP = PASS" not in combined


def test_s00b_bootstrap_requires_an_argument(repo_root: Path) -> None:
    res = _run(["bash", "scripts/bootstrap_runpod_s00b.sh"], repo_root)
    assert res.returncode != 0
    assert "usage" in (res.stdout + res.stderr)


def test_s00b_bootstrap_never_reaches_capture_off_hardware(repo_root: Path) -> None:
    """Off-hardware the run must stop at a gate and never reach env-capture, whichever gate
    fires first (a dirty tree during development, or the absent H100)."""
    if shutil.which("nvidia-smi"):
        pytest.skip("H100 present; the no-GPU branch is not the live one")
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    res = _run(["bash", "scripts/bootstrap_runpod_s00b.sh", head], repo_root)
    combined = res.stdout + res.stderr
    assert res.returncode != 0
    assert "S00B_BOOTSTRAP = FAIL" in combined
    assert "S00B_BOOTSTRAP = PASS" not in combined
    assert "== 8. environment capture" not in combined
    assert any(gate in combined for gate in ("working tree is dirty", "nvidia-smi absent"))


# ------------------------------------------------- S00-B closure: runtime evidence lifecycle
def _prehardware_repo(tmp_path: Path) -> Path:
    """A repo in exactly the state S00-B starts from: committed bundle, no hardware yet."""
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    (repo / "artifacts" / "p0_pre").mkdir(parents=True)
    (repo / "manifests" / "environments").mkdir(parents=True)
    (repo / "stage_acceptance" / "S00").mkdir(parents=True)
    _git(repo, "init", "-b", "stage/test", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    write_readiness(repo, compute_readiness(repo, computed_by="DECLARED_AT_S00"))
    _git(repo, "add", "-A")
    _git(repo, "-c", "core.hooksPath=", "commit", "-q", "-m", "pre-hardware")
    build_bundle.generate(repo, "S00")
    return repo


def test_closure_prehardware_bundle_is_valid_before_the_h100_run(tmp_path: Path) -> None:
    repo = _prehardware_repo(tmp_path)
    assert build_bundle.verify(repo, "S00") == []
    readiness = json.loads((repo / "artifacts/p0_pre/P0_PRE_READINESS.json").read_text("utf-8"))
    assert readiness["environment_lock_sha256"] == TBD


def test_closure_correct_hardware_run_does_not_invalidate_its_own_bundle(
    tmp_path: Path,
) -> None:
    """The exact reported failure: capture + preflight legitimately rewrite readiness, and
    the pre-hardware bundle then reported `manifest hash mismatch`, making a correct
    bootstrap structurally incapable of passing."""
    repo = _prehardware_repo(tmp_path)
    before = (repo / "artifacts/p0_pre/P0_PRE_READINESS.json").read_bytes()

    identity = write_environment_manifest(repo)  # step 8, capture
    write_readiness(repo, compute_readiness(repo))  # preflight step 8

    after = (repo / "artifacts/p0_pre/P0_PRE_READINESS.json").read_bytes()
    assert after != before, "the run must genuinely change readiness, else this proves nothing"
    assert json.loads(after)["environment_lock_sha256"] == identity

    assert build_bundle.verify(repo, "S00") == [], "a correct hardware run invalidated itself"


def test_closure_source_drift_is_still_caught_after_a_hardware_run(tmp_path: Path) -> None:
    repo = _prehardware_repo(tmp_path)
    write_environment_manifest(repo)
    write_readiness(repo, compute_readiness(repo))
    assert build_bundle.verify(repo, "S00") == []

    (repo / "src" / "mod.py").write_text("x = 2\n", encoding="utf-8")
    problems = build_bundle.verify(repo, "S00")
    assert any("src/mod.py" in p for p in problems), problems


def test_closure_forged_readiness_is_caught_by_rederivation(tmp_path: Path) -> None:
    """Readiness keeps its fail-closed semantics: it is verified by re-derivation, which a
    hand-edited file cannot survive [AUTH: 02 §C6]."""
    repo = _prehardware_repo(tmp_path)
    write_environment_manifest(repo)
    write_readiness(repo, compute_readiness(repo))
    assert build_bundle.verify(repo, "S00") == []

    path = repo / "artifacts/p0_pre/P0_PRE_READINESS.json"
    forged = json.loads(path.read_text(encoding="utf-8"))
    forged["P0_PRE_READY"] = True
    forged["SUITE_SCOPE"] = "INTEGRATED_PRODUCTION_STACK"
    path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    problems = build_bundle.verify(repo, "S00")
    assert any("re-derivation" in p for p in problems), problems


def test_closure_tampered_environment_manifest_is_caught(tmp_path: Path) -> None:
    repo = _prehardware_repo(tmp_path)
    identity = write_environment_manifest(repo)
    write_readiness(repo, compute_readiness(repo))
    manifest = repo / "manifests" / "environments" / f"{identity}.json"
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    obj["cuda_driver"] = "999.99"
    manifest.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    problems = build_bundle.verify(repo, "S00")
    assert any("recomputed identity" in p for p in problems), problems


def test_closure_record_captures_the_hardware_run(tmp_path: Path) -> None:
    repo = _prehardware_repo(tmp_path)
    assert build_bundle.closure_record(repo, "S00")["s00b_complete"] is False

    identity = write_environment_manifest(repo)
    write_readiness(repo, compute_readiness(repo))
    record = build_bundle.closure_record(repo, "S00")
    assert record["s00b_complete"] is True
    assert record["environment_lock_sha256"] == identity
    assert record["environment_manifests"] == [f"{identity}.json"]
    produced = record["produced_artifact_sha256"]
    assert isinstance(produced, dict)
    assert f"manifests/environments/{identity}.json" in produced
    assert "artifacts/p0_pre/P0_PRE_READINESS.json" in produced
    readiness = record["readiness"]
    assert isinstance(readiness, dict)
    assert readiness["P0_PRE_READY"] is False, "no evidence records, so still not ready"


# ------------------------------------------------- S00-B ordering: verification vs the gate
def test_ordering_integration_must_not_require_final_readiness(tmp_path: Path) -> None:
    """The exact real-H100 failure, reproduced.

    capture publishes a new environment identity -> preflight starts -> integration (step 5)
    ran FULL bundle verification -> the verifier re-derived readiness against the new
    environment while readiness still held the pre-run identity -> step 5 failed -> step 8
    write-readiness was never reached -> readiness could never become consistent.
    """
    repo = _prehardware_repo(tmp_path)
    assert build_bundle.verify(repo, "S00") == []

    identity = write_environment_manifest(repo)  # step 8 of the bootstrap
    stored = json.loads(
        (repo / "artifacts/p0_pre/P0_PRE_READINESS.json").read_text(encoding="utf-8")
    )
    assert stored["environment_lock_sha256"] == TBD, "readiness is still the pre-run one"

    # This is the state preflight step 5 runs in. The immutable half must pass...
    assert build_bundle.verify_source(repo, "S00") == [], "source half must be evaluable now"
    # ...while the runtime half is legitimately inconsistent until step 8 has run.
    runtime_problems = build_bundle.verify_runtime(repo, "S00")
    assert any("re-derivation" in p for p in runtime_problems), runtime_problems

    write_readiness(repo, compute_readiness(repo))  # preflight step 8
    assert (
        json.loads((repo / "artifacts/p0_pre/P0_PRE_READINESS.json").read_text(encoding="utf-8"))[
            "environment_lock_sha256"
        ]
        == identity
    )
    assert build_bundle.verify(repo, "S00") == [], "the final gate must pass after step 8"


def test_ordering_bootstrap_runs_the_full_verifier_after_preflight(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "bootstrap_runpod_s00b.sh").read_text(encoding="utf-8")
    order = [
        script.index("make env-capture"),
        script.index("make gpu-smoke"),
        script.index("make preflight"),
        script.index("make bundle-verify"),
        script.index("make closure"),
    ]
    assert order == sorted(order), "the final verifier must follow capture, GPU and preflight"
    assert "make bundle-verify-source" not in script, "the bootstrap gate is the full one"


def test_ordering_preflight_step_five_does_not_run_the_full_verifier(repo_root: Path) -> None:
    """No integration test may run the FULL verifier against the live repository.

    Checked structurally rather than by grepping, because a grep for the forbidden call
    matches the assertion that forbids it.
    """
    module = ast.parse(
        (repo_root / "tests" / "integration" / "test_s00_command_surface.py").read_text(
            encoding="utf-8"
        )
    )
    offenders: list[str] = []
    for function in [n for n in module.body if isinstance(n, ast.FunctionDef)]:
        uses_live_repo = any(
            isinstance(arg, ast.arg) and arg.arg == "repo_root" for arg in function.args.args
        )
        if not uses_live_repo:
            continue
        for call in [n for n in ast.walk(function) if isinstance(n, ast.Call)]:
            func = call.func
            is_full_verify = (
                isinstance(func, ast.Attribute)
                and func.attr == "verify"
                and isinstance(func.value, ast.Name)
                and func.value.id == "build_bundle"
            )
            names = {a.id for a in call.args if isinstance(a, ast.Name)}
            if is_full_verify and "repo_root" in names:
                offenders.append(f"{function.name}: build_bundle.verify(repo_root)")
            if (
                isinstance(func, ast.Name)
                and func.id == "_run"
                and "bundle-verify" in ast.dump(call)
                and "bundle-verify-source" not in ast.dump(call)
                and "repo_root" in names
            ):
                offenders.append(f"{function.name}: make bundle-verify on the live repo")
    assert offenders == [], offenders


# ------------------------------------------------- S00-B gpu-smoke evidence persistence
def test_gpu_smoke_success_is_persisted_as_lane_evidence(tmp_path: Path) -> None:
    """A lane that ran 8 real GPU tests must not still read NOT_RUN(NO_GPU) [AUTH: 02 §C6]."""
    identity = write_environment_manifest(tmp_path)
    record_lane_evidence(tmp_path, "gpu_smoke", "PASS", "8 passed, 0 skipped in 3.1s")

    readiness = compute_readiness(tmp_path)
    lanes = readiness["evidence"]
    assert isinstance(lanes, dict)
    entry = lanes["gpu_smoke"]
    assert isinstance(entry, dict)
    assert entry["status"] != "NOT_RUN(NO_GPU)"
    observed = entry["observed"]
    assert isinstance(observed, dict)
    assert observed["outcome"] == "PASS"
    assert "8 passed" in observed["detail"]

    # Correct S00 state is unchanged: the record is observed, not yet evidentiary.
    assert readiness["BACKEND_INTEGRATED"] is False
    assert readiness["SUITE_SCOPE"] == "STATISTICAL_STACK_ONLY"
    assert readiness["P0_PRE_READY"] is False
    assert identity == readiness["environment_lock_sha256"]


def test_gpu_smoke_evidence_is_not_accepted_without_a_run_manifest(tmp_path: Path) -> None:
    """Recording a lane must not become a way to forge readiness [AUTH: 01 §16; 02 §C6]."""
    write_environment_manifest(tmp_path)
    record_lane_evidence(tmp_path, "gpu_smoke", "PASS", "8 passed")
    lanes = compute_readiness(tmp_path)["evidence"]
    assert isinstance(lanes, dict)
    entry = lanes["gpu_smoke"]
    assert isinstance(entry, dict)
    assert entry["status"] == "NON_EVIDENTIARY"
    assert "run_id" in str(entry["reason"])


def test_gpu_smoke_recorder_rejects_an_unknown_lane(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown lane"):
        record_lane_evidence(tmp_path, "not_a_lane", "PASS")


def test_gpu_smoke_recipe_records_only_on_success(repo_root: Path) -> None:
    recipe = (repo_root / "Makefile").read_text(encoding="utf-8")
    lane = recipe.split("gpu-smoke:", 1)[1].split("\n\n", 1)[0]
    assert "--record-lane gpu_smoke" in lane
    record_at = lane.index("--record-lane")
    assert lane.index("exit $$s") < record_at, "a failing lane must not record PASS"
    assert lane.index("zero tests collected") < record_at
