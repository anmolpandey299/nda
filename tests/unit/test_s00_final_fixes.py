"""Regression tests for the S00 final implementation-fix pass.

Every test below is a reviewer counterexample turned into a fixture. Nothing here relaxes an
existing assertion.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from _helpers import (
    RUN_ID,
    valid_record,
    write_artifact,
    write_environment_manifest,
    write_evidence,
    write_run_manifest,
)
from check_repo_invariants import (
    MATERIAL_CONSTANTS,
    RUN_MANIFEST_REQUIRED_FIELDS,
    check_material_constants,
    check_single_owner,
)
from preflight import (
    BACKEND_INTEGRATED_FALSE,
    BACKEND_INTEGRATED_TRUE,
    BACKEND_READINESS_UNREADABLE,
    EVIDENCE_GATE_REL,
    EVIDENCE_LANES_REL,
    TBD,
    backend_integrated_exit_code,
    compute_readiness,
    environment_identity_status,
    verify_evidence_record,
)


def _src(root: Path, rel: str, body: str) -> Path:
    path = root / "src" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    (root / "specs" / "deviations").mkdir(parents=True, exist_ok=True)
    (root / "specs" / "deviations" / "SCORER_EXCEPTIONS.md").write_text(
        "# register\n## Granted exceptions\n", encoding="utf-8"
    )
    return path


# ====================================================================== FIX 1
def test_fix1_backend_gate_reports_false(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"BACKEND_INTEGRATED": False}), encoding="utf-8")
    assert backend_integrated_exit_code(p)[0] == BACKEND_INTEGRATED_FALSE


def test_fix1_backend_gate_reports_true(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"BACKEND_INTEGRATED": True}), encoding="utf-8")
    assert backend_integrated_exit_code(p)[0] == BACKEND_INTEGRATED_TRUE


@pytest.mark.parametrize(
    "payload", [None, "not json at all", json.dumps({"other": 1}), json.dumps([1, 2])]
)
def test_fix1_unreadable_readiness_is_never_silently_not_run(
    tmp_path: Path, payload: str | None
) -> None:
    """A missing or malformed readiness file is a hard failure, not NOT_RUN [AUTH: 02 §C6]."""
    p = tmp_path / "r.json"
    if payload is not None:
        p.write_text(payload, encoding="utf-8")
    code, message = backend_integrated_exit_code(p)
    assert code == BACKEND_READINESS_UNREADABLE
    assert message


def test_fix1_makefile_uses_the_readiness_variable(repo_root: Path) -> None:
    """The recipe must expand $(READINESS); the old `$$READINESS` form was mangled by the
    shell into a PID-prefixed literal, so branch B was unreachable."""
    recipe = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert "$(READINESS)" in recipe
    assert "READINESS ?=" in recipe
    assert "2>/dev/null" not in recipe.split("backend-contract:")[1].split("\n\n")[0]


# ====================================================================== FIX 2
def test_fix2_positive_control_is_accepted(tmp_path: Path) -> None:
    """Guards against the hardened checks rejecting everything unconditionally."""
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert verdict.accepted, verdict.reason


def test_fix2_shaped_forgery_pointing_at_readme_is_rejected(tmp_path: Path) -> None:
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    (tmp_path / "README.md").write_text("# not a run manifest\n", encoding="utf-8")
    record["run_manifest"] = "README.md"
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert not verdict.accepted
    assert "manifests/runs" in verdict.reason


def test_fix2_empty_json_run_manifest_is_rejected(tmp_path: Path) -> None:
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    (tmp_path / str(record["run_manifest"])).write_text("{}", encoding="utf-8")
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert not verdict.accepted
    assert "01 §16 fields" in verdict.reason


def test_fix2_unparseable_run_manifest_is_rejected(tmp_path: Path) -> None:
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    (tmp_path / str(record["run_manifest"])).write_text("{not json", encoding="utf-8")
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert not verdict.accepted and "not valid JSON" in verdict.reason


def test_fix2_run_id_must_match_the_manifest(tmp_path: Path) -> None:
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    write_run_manifest(tmp_path, RUN_ID, env)
    manifest_path = tmp_path / str(record["run_manifest"])
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj["run_id"] = "b" * 64
    manifest_path.write_text(json.dumps(obj), encoding="utf-8")
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert not verdict.accepted and "RUN_ID" in verdict.reason


def test_fix2_manifest_environment_must_match_the_record(tmp_path: Path) -> None:
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    manifest_path = tmp_path / str(record["run_manifest"])
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj["environment_lock_sha256"] = "c" * 64
    manifest_path.write_text(json.dumps(obj), encoding="utf-8")
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert not verdict.accepted and "environment identity" in verdict.reason


@pytest.mark.parametrize("field", RUN_MANIFEST_REQUIRED_FIELDS[:6])
def test_fix2_every_required_run_manifest_field_is_enforced(tmp_path: Path, field: str) -> None:
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    write_run_manifest(tmp_path, RUN_ID, env, drop=(field,))
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert not verdict.accepted, f"missing {field} was accepted"


def test_fix2_artifact_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    write_artifact(tmp_path, "artifacts/p0_pre/fixture.bin", b"tampered")
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert not verdict.accepted and "hash mismatch" in verdict.reason


def test_fix2_path_traversal_is_rejected(tmp_path: Path) -> None:
    env = write_environment_manifest(tmp_path)
    record = valid_record(tmp_path, env)
    record["artifact_sha256"] = {"../outside.bin": "0" * 64}
    verdict = verify_evidence_record("backend_contract", record, tmp_path, env)
    assert not verdict.accepted


def test_fix2_identity_is_recomputed_not_trusted(tmp_path: Path) -> None:
    """A hand-edited environment_lock_sha256 must not authorise anything [AUTH: 01 §12, §32]."""
    env = write_environment_manifest(tmp_path)
    manifest_path = next((tmp_path / "manifests" / "environments").glob("*.json"))
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj["environment_lock_sha256"] = "d" * 64
    manifest_path.write_text(json.dumps(obj), encoding="utf-8")
    identity, problems = environment_identity_status(tmp_path)
    assert identity == TBD
    assert any("recomputed" in p for p in problems)
    assert env != TBD


def test_fix2_declared_identity_without_components_is_unusable(tmp_path: Path) -> None:
    env_dir = tmp_path / "manifests" / "environments"
    env_dir.mkdir(parents=True)
    (env_dir / "e.json").write_text(
        json.dumps({"environment_lock_sha256": "e" * 64}), encoding="utf-8"
    )
    assert environment_identity_status(tmp_path)[0] == TBD


# ====================================================================== FIX 3
def test_fix3_reviewer_counterexample_compute(tmp_path: Path) -> None:
    """def compute(...): return max(tpr[fpr <= 0.01]) — a ROC implementation by another name."""
    _src(
        tmp_path,
        "analysis/metrics.py",
        "def compute(tpr: list[float], fpr: list[float]) -> float:\n"
        "    return max(tpr[fpr <= 0.01])\n",
    )
    assert any(v.invariant == "I13" for v in check_single_owner(tmp_path))


@pytest.mark.parametrize(
    "stmt",
    [
        "from sklearn.metrics import roc_curve",
        "from sklearn.metrics import roc_curve as rc",
        "import sklearn.metrics as skm",
        "from sklearn import metrics",
        "from sklearn.metrics import roc_auc_score as auc",
    ],
)
def test_fix3_aliased_roc_toolkit_imports_are_blocked(tmp_path: Path, stmt: str) -> None:
    _src(tmp_path, "analysis/m.py", stmt + "\n")
    assert any(v.invariant == "I13" for v in check_single_owner(tmp_path)), stmt


def test_fix3_class_methods_are_covered(tmp_path: Path) -> None:
    _src(
        tmp_path,
        "attacks/scorer.py",
        "class Attack:\n    def tpr_at_fpr(self, x: float) -> float:\n        return x\n",
    )
    assert any(v.invariant == "I13" for v in check_single_owner(tmp_path))


def test_fix3_module_scope_implementation_is_covered(tmp_path: Path) -> None:
    _src(tmp_path, "analysis/m.py", "tpr = [0.1]\nfpr = [0.01]\nvalue = max(tpr)\n")
    assert any(v.invariant == "I13" for v in check_single_owner(tmp_path))


def test_fix3_operating_point_selection_alone_is_blocked(tmp_path: Path) -> None:
    _src(
        tmp_path,
        "analysis/m.py",
        "def pick(fpr: list[float]) -> list[bool]:\n    return [f <= 0.01 for f in fpr]\n",
    )
    assert any(v.invariant == "I13" for v in check_single_owner(tmp_path))


def test_fix3_scoring_package_itself_is_allowed(tmp_path: Path) -> None:
    """The sole owner must remain able to implement the estimator [AUTH: 00 §34A.2]."""
    _src(
        tmp_path,
        "scoring/roc.py",
        "from sklearn.metrics import roc_curve\n"
        "def tpr_at_fpr(tpr: list[float], fpr: list[float]) -> float:\n"
        "    return max(t for t, f in zip(tpr, fpr, strict=True) if f <= 0.01)\n",
    )
    assert check_single_owner(tmp_path) == []


def test_fix3_consuming_the_scoring_api_is_allowed(tmp_path: Path) -> None:
    _src(
        tmp_path,
        "analysis/report.py",
        "from src.scoring.roc import tpr_at_fpr\n"
        "def build(scores: list[float]) -> float:\n    return tpr_at_fpr(scores)\n",
    )
    assert check_single_owner(tmp_path) == []


# ====================================================================== FIX 4
@pytest.mark.parametrize(
    "line",
    [
        "target_fpr = 0.01",
        "TARGET_FPR = 0.01",
        "Target_Fpr = 0.01",
        "min_k = 0.2",
        "MIN_K = 0.2",
        "min_k_fraction = 0.2",
        "seeds = [101, 202, 303]",
        "SEEDS = [101, 202, 303]",
        "alpha = 0.5",
        "k = 4",
        "dare_p = 0.9",
        "svd_rank = 32",
        "bootstrap_replicates = 2000",
        "n_bootstrap = 2000",
        "random_seed = 101",
    ],
)
def test_fix4_material_constants_are_case_insensitive(tmp_path: Path, line: str) -> None:
    _src(tmp_path, "analysis/metrics.py", line + "\n")
    violations = check_material_constants(tmp_path)
    assert violations and violations[0].invariant == "I14", f"{line!r} not detected"


def test_fix4_annotated_assignment_is_covered(tmp_path: Path) -> None:
    _src(tmp_path, "analysis/metrics.py", "target_fpr: float = 0.01\n")
    assert check_material_constants(tmp_path)


def test_fix4_config_resolution_still_allowed(tmp_path: Path) -> None:
    _src(
        tmp_path,
        "analysis/metrics.py",
        "from cfg import load\ntarget_fpr = load('attacks')['target_fpr']\n",
    )
    assert check_material_constants(tmp_path) == []


def test_fix4_patterns_are_compiled_case_insensitively() -> None:
    assert all(p.flags & __import__("re").IGNORECASE for p in MATERIAL_CONSTANTS)


# ====================================================================== FIX 5
def test_fix5_suite_scope_requires_the_software_gate(tmp_path: Path) -> None:
    """All five lanes valid but the software gate absent -> not the integrated stack."""
    env = write_environment_manifest(tmp_path)
    for key in (
        "backend_contract",
        "synthetic_suite",
        "cache_assertions",
        "gpu_smoke",
        "benchmark_1000_seq",
    ):
        write_evidence(tmp_path, "lanes", key, valid_record(tmp_path, env))
    readiness = compute_readiness(tmp_path)
    assert readiness["BACKEND_INTEGRATED"] is True
    assert readiness["SUITE_SCOPE"] == "STATISTICAL_STACK_ONLY"
    assert readiness["P0_PRE_READY"] is False


def test_fix5_full_conjunction_flips_every_flag(tmp_path: Path) -> None:
    """Positive control: the gate is reachable, so the negative cases mean something."""
    env = write_environment_manifest(tmp_path)
    for key in (
        "backend_contract",
        "synthetic_suite",
        "cache_assertions",
        "gpu_smoke",
        "benchmark_1000_seq",
    ):
        write_evidence(tmp_path, "lanes", key, valid_record(tmp_path, env))
    for key in (
        "SCORER_ENGINE",
        "ANALYSIS_DRY_RUN",
        "CROSSFIT_NEGATIVE_CONTROL",
        "CACHE_ASSERTIONS",
        "P0_00_COMPUTE_BUDGET",
    ):
        write_evidence(tmp_path, "software_gate", key, valid_record(tmp_path, env))
    readiness = compute_readiness(tmp_path)
    assert readiness["SUITE_SCOPE"] == "INTEGRATED_PRODUCTION_STACK"
    assert readiness["P0_PRE_READY"] is True


def test_fix5_namespaces_are_disjoint_directories() -> None:
    assert EVIDENCE_LANES_REL != EVIDENCE_GATE_REL
    assert not EVIDENCE_LANES_REL.startswith(EVIDENCE_GATE_REL)
    assert not EVIDENCE_GATE_REL.startswith(EVIDENCE_LANES_REL)


def test_fix5_lane_evidence_cannot_satisfy_a_gate_slot(tmp_path: Path) -> None:
    """On a case-insensitive filesystem `cache_assertions.json` must not also satisfy
    `CACHE_ASSERTIONS` [AUTH: 02 §C6]."""
    env = write_environment_manifest(tmp_path)
    write_evidence(tmp_path, "lanes", "cache_assertions", valid_record(tmp_path, env))
    readiness = compute_readiness(tmp_path)
    gate = readiness["software_gate"]
    assert isinstance(gate, dict)
    assert gate["CACHE_ASSERTIONS"]["status"] == "NOT_RUN"


def test_fix5_case_variant_filename_does_not_satisfy_a_slot(tmp_path: Path) -> None:
    env = write_environment_manifest(tmp_path)
    base = tmp_path / EVIDENCE_GATE_REL
    base.mkdir(parents=True)
    (base / "cache_assertions.json").write_text(
        json.dumps(valid_record(tmp_path, env)), encoding="utf-8"
    )
    gate = compute_readiness(tmp_path)["software_gate"]
    assert isinstance(gate, dict)
    assert gate["CACHE_ASSERTIONS"]["status"] == "NOT_RUN"


# ====================================================================== FIX 6 / FIX 7
def test_fix6_docker_bases_are_digest_pinned(repo_root: Path) -> None:
    text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    froms = [ln.split()[1] for ln in text.splitlines() if ln.startswith("FROM ")]
    assert froms
    for ref in froms:
        assert ref.startswith("${") or "@sha256:" in ref, f"{ref} is a mutable reference"
        assert ":latest" not in ref, f"{ref} is a floating tag"
    assert ":latest" not in text, "a floating tag is referenced somewhere in the Dockerfile"


def test_fix6_capture_does_not_trust_caller_supplied_identity(repo_root: Path) -> None:
    """The capture body now lives in capture_environment.py; the guarantee is unchanged."""
    module = (repo_root / "scripts" / "capture_environment.py").read_text(encoding="utf-8")
    assert "/etc/pmm-image.json" in module
    assert "CALLER_SUPPLIED_UNVERIFIED" in module
    wrapper = (repo_root / "scripts" / "capture_environment.sh").read_text(encoding="utf-8")
    assert "capture_environment.py" in wrapper


def test_fix6_gpu_and_backend_lanes_collect_tests(repo_root: Path) -> None:
    """A lane that collects nothing cannot report a state [AUTH: 02 §C6; 00 §34B.3]."""
    for lane in ("gpu_smoke", "backend_contract"):
        files = sorted((repo_root / "tests" / lane).glob("test_*.py"))
        assert files, f"{lane} lane would collect zero tests"
        for f in files:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            tests = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            ]
            assert tests, f"{f} defines no tests"


def test_fix7_ci_uses_the_frozen_lock(repo_root: Path) -> None:
    text = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv lock --check" in text
    assert "uv sync --frozen" in text
