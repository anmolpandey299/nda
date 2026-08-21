"""S00 acceptance tests A1-A4, A7-A9, A11-A18.

Every test here runs on CPU with no network, no model download and no H100
[AUTH: 01 §20, §39 S00; 03 §8; plan §14].
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from check_repo_invariants import (
    ENVIRONMENT_LOCK_COMPONENTS,
    FORBIDDEN_PATHS,
    REQUIRED_PATHS,
    RUN_ID_INPUTS,
    SPEC_FILES,
    check_material_constants,
    check_prohibited_estimator,
    check_single_owner,
    run_all,
)
from preflight import (
    EVIDENCE_KEYS,
    SOFTWARE_GATE_KEYS,
    TBD,
    compute_readiness,
    environment_lock_sha256,
    validate_environment_manifest,
)

UNAVAILABLE = "UNAVAILABLE_NOT_EXPOSED"

#: A2 — asserted as literals so regenerating SPEC_HASHES.json alone cannot clear the gate
#: [AUTH: 01 §15; plan I4, C-06 closure].
FROZEN_SPEC_SHA256: dict[str, str] = dict(
    zip(
        SPEC_FILES,
        (
            "eb7ab9136279d9a3c189c635d34fbd3a84f8b370640d640da53bdbbe330e9deb",
            "82a9a93c0e1ad05fd5e64958f7daa75e6dbe99b23fb1b65ff750e6451fb8abc5",
            "1ecba4e6b0a9c5fb60ace3ba85335b803fe72d7b0b639b0dfb80dacaf2952cb6",
            "7bf2048067d09a2b72333f381c07bb4863baae3fd70f03e153e07af5d99691bc",
        ),
        strict=True,
    )
)


def _skeleton(root: Path, *, exceptions: str = "") -> Path:
    """Minimal tree for injected-violation cases."""
    (root / "src" / "scoring").mkdir(parents=True)
    (root / "src" / "analysis").mkdir(parents=True)
    (root / "specs" / "deviations").mkdir(parents=True)
    (root / "specs" / "deviations" / "SCORER_EXCEPTIONS.md").write_text(
        "# register\n## Granted exceptions\n" + exceptions, encoding="utf-8"
    )
    return root


# ------------------------------------------------------------------ A1
def test_a1_repo_structure(repo_root: Path) -> None:
    missing = [p for p in REQUIRED_PATHS if not (repo_root / p).exists()]
    present = [p for p in FORBIDDEN_PATHS if (repo_root / p).exists()]
    assert missing == [], f"required paths missing: {missing}"
    assert present == [], f"forbidden paths present: {present}"


def test_a1_no_violations_in_live_repo(repo_root: Path) -> None:
    assert [v.render() for v in run_all(repo_root)] == []


# ------------------------------------------------------------------ A2
def test_a2_spec_integrity(repo_root: Path) -> None:
    recorded = json.loads((repo_root / "specs" / "SPEC_HASHES.json").read_text(encoding="utf-8"))
    assert set(recorded) == set(SPEC_FILES)
    for name, frozen in FROZEN_SPEC_SHA256.items():
        actual = hashlib.sha256((repo_root / "specs" / name).read_bytes()).hexdigest()
        assert actual == frozen, f"{name} content changed"
        assert recorded[name]["sha256"] == frozen, f"{name} hash record diverged"


# ------------------------------------------------------------------ A3
def test_a3_plan_length(repo_root: Path) -> None:
    plans = sorted((repo_root / "stage_acceptance").glob("*/01_PLAN.md"))
    assert plans, "canonical stage-plan glob is empty [AUTH: 03 §13; plan D10]"
    for plan in plans:
        n = len(plan.read_text(encoding="utf-8").splitlines())
        assert n <= 1200, f"PLAN_TOO_LARGE: {plan} has {n} lines"


# ------------------------------------------------------------------ A4
def test_a4_readiness_initial_state(repo_root: Path) -> None:
    obj = json.loads((repo_root / "artifacts/p0_pre/P0_PRE_READINESS.json").read_text("utf-8"))
    assert obj["BACKEND_INTEGRATED"] is False
    assert obj["SUITE_SCOPE"] == "STATISTICAL_STACK_ONLY"
    assert obj["P0_PRE_READY"] is False
    assert set(obj["evidence"]) == set(EVIDENCE_KEYS)
    assert set(obj["software_gate"]) == set(SOFTWARE_GATE_KEYS)
    for entry in obj["evidence"].values():
        assert entry["status"].startswith("NOT_RUN")


def test_a4_no_source_path_raises_readiness(repo_root: Path) -> None:
    pat = re.compile("P0_PRE" + "_READY" + r'"?\s*[:=]\s*[Tt]rue')
    offenders = [
        p.relative_to(repo_root).as_posix()
        for p in list((repo_root / "src").rglob("*.py"))
        + list((repo_root / "scripts").rglob("*.py"))
        if p.name != "preflight.py" and pat.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_a4_forged_evidence_cannot_raise_flags(tmp_path: Path) -> None:
    """Hand-authored PASS files leave every flag unchanged [AUTH: 02 §C6; 01 §16]."""
    ev = tmp_path / "artifacts" / "p0_pre" / "evidence"
    ev.mkdir(parents=True)
    for key in EVIDENCE_KEYS + SOFTWARE_GATE_KEYS:
        (ev / f"{key}.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    readiness = compute_readiness(tmp_path)
    assert readiness["BACKEND_INTEGRATED"] is False
    assert readiness["SUITE_SCOPE"] == "STATISTICAL_STACK_ONLY"
    assert readiness["P0_PRE_READY"] is False


def test_a4_evidence_needs_full_provenance_binding(tmp_path: Path) -> None:
    """Even a run-id-bearing record fails without a run manifest and artifact hashes."""
    ev = tmp_path / "artifacts" / "p0_pre" / "evidence"
    ev.mkdir(parents=True)
    (ev / "backend_contract.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_id": "a" * 64,
                "environment_lock_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert compute_readiness(tmp_path)["BACKEND_INTEGRATED"] is False


# ------------------------------------------------------------------ A7 / A16
def test_a7_reserved_api_confined_to_scoring(tmp_path: Path) -> None:
    root = _skeleton(tmp_path)
    (root / "src" / "scoring" / "roc.py").write_text("def f() -> None: ...\n", encoding="utf-8")
    assert check_single_owner(root) == []
    (root / "src" / "analysis" / "roc.py").write_text("def f() -> None: ...\n", encoding="utf-8")
    v = check_single_owner(root)
    assert any(x.invariant == "I1" for x in v), "duplicate reserved module not detected"


def test_a7_scorer_exception_permits_a_named_duplicate(tmp_path: Path) -> None:
    root = _skeleton(tmp_path, exceptions="| X | | | src/analysis/roc.py | why | tests | claims |")
    (root / "src" / "scoring" / "roc.py").write_text("def f() -> None: ...\n", encoding="utf-8")
    (root / "src" / "analysis" / "roc.py").write_text("def f() -> None: ...\n", encoding="utf-8")
    assert [x for x in check_single_owner(root) if x.invariant == "I1"] == []


def test_a7_import_boundary_blocks_a_second_estimator(tmp_path: Path) -> None:
    """Codex S00-CBR-004's exact counterexample: a differently-named analysis module."""
    root = _skeleton(tmp_path)
    (root / "src" / "analysis" / "privacy_metrics.py").write_text(
        "def tpr_at_fpr(scores: list[float], target: float) -> float:\n    return 0.0\n",
        encoding="utf-8",
    )
    v = check_single_owner(root)
    assert any(x.invariant == "I13" for x in v), "second fixed-FPR estimator not detected"


@pytest.mark.parametrize("name", ["make_folds", "calibrate_threshold", "cross_fit_scores"])
def test_a7_import_boundary_covers_fold_and_threshold_logic(tmp_path: Path, name: str) -> None:
    root = _skeleton(tmp_path)
    (root / "src" / "analysis" / "m.py").write_text(
        f"def {name}() -> None: ...\n", encoding="utf-8"
    )
    assert any(x.invariant == "I13" for x in check_single_owner(root))


def test_a16_prohibited_estimator_absent(repo_root: Path) -> None:
    assert check_prohibited_estimator(repo_root) == []


def test_a16_prohibited_estimator_is_detected(tmp_path: Path) -> None:
    root = _skeleton(tmp_path)
    (root / "src" / "scoring" / "engine.py").write_text(
        "# take the max TPR at repeated FPR\n", encoding="utf-8"
    )
    assert check_prohibited_estimator(root) != []


def test_a16_m_primary_is_declared(repo_root: Path) -> None:
    """M_PRIMARY = common-FPR TPR@1%FPR, declared and prohibition recorded
    [AUTH: 02 §C1, §C2, §C8; plan §13.1]."""
    register = (repo_root / "specs/deviations/SCORER_EXCEPTIONS.md").read_text("utf-8")
    assert "M_PRIMARY" in register and "FPR = 0.01" in register
    assert "PROHIBITED" in register
    claude_md = (repo_root / "CLAUDE.md").read_text("utf-8")
    assert "common-FPR TPR@1%FPR" in claude_md
    plan = (repo_root / "stage_acceptance/S00/01_PLAN.md").read_text("utf-8")
    assert "M_PRIMARY(V) = TPR_OOF/eval( V | FPR = 0.01 )" in plan


def test_a16_at_most_one_fixed_fpr_owner_in_live_repo(repo_root: Path) -> None:
    """S00 ships zero estimators; the invariant is the constraint, not the existence."""
    owners = [p for p in (repo_root / "src").rglob("*.py") if p.stem == "roc"]
    assert len(owners) <= 1
    assert all(p.parent.name == "scoring" for p in owners)


# ------------------------------------------------------------------ A8 / A9
def test_a8_no_notebook_execution_path(repo_root: Path) -> None:
    assert list((repo_root / "src").rglob("*.ipynb")) == []
    assert ".ipynb" not in (repo_root / "Makefile").read_text("utf-8")


def test_a9_reviewer_isolation(repo_root: Path) -> None:
    rev = repo_root / "reviews"
    stray = [p for p in rev.rglob("*.py") if "scratch" not in p.relative_to(rev).parts]
    assert stray == []
    assert 'packages = ["src"]' in (repo_root / "pyproject.toml").read_text("utf-8")


# ------------------------------------------------------------------ A11
def test_a11_ci_workflow_contract(repo_root: Path) -> None:
    text = (repo_root / ".github/workflows/ci.yml").read_text("utf-8")
    assert "make preflight" in text
    assert "make backend-contract" not in text
    assert "make gpu-smoke" not in text
    assert re.search(r"(?i)runs-on:.*(gpu|cuda|h100)", text) is None


# ------------------------------------------------------------------ A12
def test_a12_run_id_input_list_is_frozen() -> None:
    """Exactly the eight 01 §15 inputs, no more [AUTH: 01 §15; plan §7.1]."""
    assert RUN_ID_INPUTS == (
        "git_commit_sha",
        "spec_sha256",
        "execution_lock_sha256",
        "config_sha256",
        "model_revision",
        "data_manifest_sha256",
        "environment_lock_sha256",
        "training_seed",
    )
    assert len(RUN_ID_INPUTS) == 8


def test_a12_provenance_paths_reserved(repo_root: Path) -> None:
    for rel in (
        "manifests/models",
        "manifests/data",
        "manifests/environments",
        "manifests/runs",
        "artifacts/p0_pre",
        "artifacts/runs",
        "artifacts/cache",
        "logs",
        "results/p0",
        "results/p1",
    ):
        assert (repo_root / rel).is_dir(), rel


# ------------------------------------------------------------------ A13
def test_a13_ai_stack_record(repo_root: Path) -> None:
    obj = json.loads(
        (repo_root / "manifests/environments/AI_ENGINEERING_STACK_S00.json").read_text("utf-8")
    )
    for field in (
        "claude_code_client_version",
        "exact_exposed_model_name",
        "datetime_utc",
        "permission_mode",
        "repository_commit",
    ):
        assert obj["claude"][field], f"claude.{field} missing [AUTH: 01 §2.1]"
    for field in (
        "codex_client_product",
        "client_version",
        "exact_model_name",
        "review_datetime_utc",
        "prompt_sha256",
        "repository_commit",
        "review_output_sha256",
    ):
        assert field in obj["codex"], f"codex.{field} missing [AUTH: 01 §4, §48]"
    for value in list(obj["claude"].values()) + list(obj["codex"].values()):
        if isinstance(value, str) and value.startswith("UNAVAILABLE"):
            assert value == UNAVAILABLE, "unavailable fields use one exact sentinel"
    assert re.fullmatch(r"[0-9a-f]{40}", obj["claude"]["repository_commit"]), (
        "repository_commit must be a real commit SHA [AUTH: 01 §2.1]"
    )


# ------------------------------------------------------------------ A14 / A15
def test_a14_no_fabricated_hardware_values(repo_root: Path) -> None:
    readiness = json.loads(
        (repo_root / "artifacts/p0_pre/P0_PRE_READINESS.json").read_text("utf-8")
    )
    env = readiness["environment_lock_sha256"]
    assert env == TBD or re.fullmatch(r"[0-9a-f]{64}", env)
    for manifest in (repo_root / "manifests/environments").glob("*.json"):
        if manifest.name == "AI_ENGINEERING_STACK_S00.json":
            continue
        obj = json.loads(manifest.read_text("utf-8"))
        for key in ENVIRONMENT_LOCK_COMPONENTS:
            value = obj.get(key, "")
            assert not (isinstance(value, str) and value.startswith("TBD") and value != TBD), (
                f"{manifest.name}:{key} is a malformed placeholder"
            )


def test_a15_env_manifest_schema_accepts_a_complete_capture() -> None:
    manifest: dict[str, object] = {k: f"value-{k}" for k in ENVIRONMENT_LOCK_COMPONENTS}
    manifest.update(
        {
            "nvidia_smi_capture": "...",
            "gpu_uuid": "GPU-x",
            "gpu_count": "1",
            "dependency_versions": "...",
            "bf16_fp32_tolerance": "1e-2",
            "nondeterminism_sources": "none",
            "capture_timestamp_utc": "2026-08-21T00:00:00Z",
            "docker_image_tag": "tag",
        }
    )
    manifest["environment_lock_sha256"] = environment_lock_sha256(manifest)
    assert validate_environment_manifest(manifest) == []
    assert re.fullmatch(r"[0-9a-f]{64}", str(manifest["environment_lock_sha256"]))


def test_a15_env_manifest_schema_rejects_a_partial_capture() -> None:
    assert "cuda_driver" in validate_environment_manifest({"uv_lock_sha256": "x"})


def test_a15_unresolved_environment_has_no_identity() -> None:
    """An unresolved component must not yield a well-formed identity [AUTH: 03 §8]."""
    manifest: dict[str, object] = {k: f"v-{k}" for k in ENVIRONMENT_LOCK_COMPONENTS}
    manifest["cuda_driver"] = TBD
    assert environment_lock_sha256(manifest) == TBD


def test_a15_environment_identity_covers_every_component() -> None:
    """C-01 / S00-CBR-003: two images sharing uv.lock but differing elsewhere must differ."""
    base: dict[str, object] = {k: f"v-{k}" for k in ENVIRONMENT_LOCK_COMPONENTS}
    reference = environment_lock_sha256(base)
    for key in ENVIRONMENT_LOCK_COMPONENTS:
        altered = dict(base)
        altered[key] = "CHANGED"
        assert environment_lock_sha256(altered) != reference, f"identity ignores {key}"


# ------------------------------------------------------------------ A17
def test_a17_material_constants_absent_from_src(repo_root: Path) -> None:
    assert check_material_constants(repo_root) == []


@pytest.mark.parametrize(
    "line",
    [
        "TARGET_FPR = 0.01",
        "MIN_K = 0.2",
        "SEEDS = [101, 202, 303]",
        "DARE_P = 0.9",
        "SVD_RANK = 32",
        "N_BOOTSTRAP = 2000",
        "ALPHA = 0.5",
    ],
)
def test_a17_injected_source_constant_is_rejected(tmp_path: Path, line: str) -> None:
    root = _skeleton(tmp_path)
    (root / "src" / "analysis" / "metrics.py").write_text(line + "\n", encoding="utf-8")
    v = check_material_constants(root)
    assert v and v[0].invariant == "I14", f"{line!r} not detected [AUTH: 01 §17]"


def test_a17_config_resolved_constants_are_allowed(tmp_path: Path) -> None:
    """A value read from config is not a source constant."""
    root = _skeleton(tmp_path)
    (root / "src" / "analysis" / "metrics.py").write_text(
        "from cfg import load\nTARGET_FPR = load('attacks')['target_fpr']\n", encoding="utf-8"
    )
    assert check_material_constants(root) == []


# ------------------------------------------------------------------ A18
def test_a18_acceptance_bundle_completeness(repo_root: Path) -> None:
    """All twenty 01 §26 fields present in the 01 §45 layout [AUTH: 01 §26, §45; plan §8.3]."""
    bundle = repo_root / "stage_acceptance" / "S00"
    for slot in (
        "00_INDEX.md",
        "01_PLAN.md",
        "02_DIFF.patch",
        "03_TEST_COMMANDS.txt",
        "03A_RESOLVED_CONFIGS",
        "04_TEST_OUTPUTS",
        "05_ARTIFACT_MANIFEST.json",
        "06_IMPLEMENTER_REPORT.md",
        "07_CLAUDE_REVIEW.md",
        "08_CODEX_REVIEW.md",
        "09_UNRESOLVED.md",
        "10_REPRODUCE.md",
        "11_DATA_INSPECTION_STATEMENT.md",
    ):
        assert (bundle / slot).exists(), f"bundle slot missing: {slot}"
    index = (bundle / "00_INDEX.md").read_text("utf-8")
    for field in (
        "STAGE_ID",
        "bounded objective",
        "controlling spec sections",
        "base git commit",
        "head git commit",
        "requested verdict",
    ):
        assert field in index, f"00_INDEX.md lacks 01 §26 field: {field}"
    statement = (bundle / "11_DATA_INSPECTION_STATEMENT.md").read_text("utf-8")
    assert "real scientific data" in statement
