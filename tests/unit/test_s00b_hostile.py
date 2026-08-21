"""Hostile regression matrix for the reconciled S00-B findings.

Every case here is a reviewer counterexample. Nothing is asserted about real hardware.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from _helpers import write_environment_manifest
from preflight import (
    EVIDENCE_LANES_REL,
    LANE_EVIDENCE_SCHEMA,
    current_lane_pass,
    record_lane_evidence,
    validate_lane_evidence,
)

LANE = "gpu_smoke"


def _lane(root: Path, count: int = 8) -> None:
    directory = root / "tests" / LANE
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test_lane.py").write_text(
        "".join(f"def test_case_{i}() -> None: ...\n" for i in range(count)), encoding="utf-8"
    )


def _junit(root: Path, *, tests: int = 8, skipped: int = 0) -> Path:
    path = root / "junit.xml"
    path.write_text(
        f'<?xml version="1.0"?><testsuites><testsuite tests="{tests}" failures="0" '
        f'errors="0" skipped="{skipped}"/></testsuites>',
        encoding="utf-8",
    )
    return path


def _published(root: Path) -> dict[str, object]:
    _lane(root)
    write_environment_manifest(root)
    record_lane_evidence(root, LANE, junit_xml=_junit(root), pytest_status=0)
    path = root / EVIDENCE_LANES_REL / f"{LANE}.json"
    published = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(published, dict)
    return published


def _write(root: Path, record: object) -> None:
    base = root / EVIDENCE_LANES_REL
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{LANE}.json").write_text(json.dumps(record), encoding="utf-8")


# ============================================================== F1 / A, B, C
def test_f1_1_valid_current_record_is_accepted(tmp_path: Path) -> None:
    _published(tmp_path)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert problems == []
    assert record is not None
    assert record["schema"] == LANE_EVIDENCE_SCHEMA


def test_f1_2_a_minimal_fake_pass_is_rejected(tmp_path: Path) -> None:
    """Codex's exact counterexample: environment id plus observed PASS and nothing else."""
    identity = write_environment_manifest(tmp_path)
    _write(tmp_path, {"environment_lock_sha256": identity, "observed": {"outcome": "PASS"}})
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None
    assert current_lane_pass(tmp_path, LANE) is None
    assert any("schema marker" in p for p in problems)


def test_f1_3_missing_status_is_rejected(tmp_path: Path) -> None:
    published = _published(tmp_path)
    del published["status"]
    _write(tmp_path, published)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None and any("status" in p for p in problems)


@pytest.mark.parametrize(
    "field", ["tests", "passed", "failures", "errors", "skipped", "expected_tests"]
)
def test_f1_4_missing_counts_are_rejected(tmp_path: Path, field: str) -> None:
    published = _published(tmp_path)
    observed = published["observed"]
    assert isinstance(observed, dict)
    del observed[field]
    _write(tmp_path, published)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None, f"missing observed.{field} was accepted"
    assert any(field in p for p in problems)


def test_f1_5_wrong_lane_is_rejected(tmp_path: Path) -> None:
    published = _published(tmp_path)
    published["lane"] = "synthetic_suite"
    _write(tmp_path, published)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None and any("declares lane" in p for p in problems)


def test_f1_6_nonzero_skipped_is_rejected(tmp_path: Path) -> None:
    published = _published(tmp_path)
    observed = published["observed"]
    assert isinstance(observed, dict)
    observed["skipped"] = 1
    _write(tmp_path, published)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None and any("skipped" in p for p in problems)


def test_f1_7_mismatched_environment_is_rejected(tmp_path: Path) -> None:
    published = _published(tmp_path)
    published["environment_lock_sha256"] = "e" * 64
    _write(tmp_path, published)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None and any("current environment" in p for p in problems)


def test_f1_partial_collection_is_rejected(tmp_path: Path) -> None:
    published = _published(tmp_path)
    observed = published["observed"]
    assert isinstance(observed, dict)
    observed["tests"] = 2
    observed["passed"] = 2
    _write(tmp_path, published)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None and any("required tests" in p for p in problems)


def test_f1_passed_must_account_for_every_collected_test(tmp_path: Path) -> None:
    published = _published(tmp_path)
    observed = published["observed"]
    assert isinstance(observed, dict)
    observed["passed"] = 3
    _write(tmp_path, published)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None and any("does not account" in p for p in problems)


def test_f1_unparseable_record_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / EVIDENCE_LANES_REL
    base.mkdir(parents=True)
    (base / f"{LANE}.json").write_text("{not json", encoding="utf-8")
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None and any("not valid JSON" in p for p in problems)


def test_f1_one_validator_serves_every_consumer() -> None:
    """No consumer may hold a weaker private predicate."""
    import build_bundle
    import preflight

    source = Path(preflight.__file__).read_text(encoding="utf-8")
    assert source.count("def validate_lane_evidence(") == 1
    assert "return validate_lane_evidence(root, key)[0]" in source
    bundle_source = Path(build_bundle.__file__).read_text(encoding="utf-8")
    assert "validate_lane_evidence" in bundle_source


# ============================================================== F5 / N
def test_f5_runbook_uses_the_build_commit_for_every_executable_step(
    repo_root: Path,
) -> None:
    text = (repo_root / "stage_acceptance/S00/10_REPRODUCE.md").read_text(encoding="utf-8")
    assert "<candidate-commit>" not in text, "ambiguous placeholder in an executable step"
    assert "SCIENCE_DESCRIBED_COMMIT" in text
    assert "FINAL_BUNDLE_BUILD_COMMIT" in text
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        for line in block.splitlines():
            if "bootstrap_runpod_s00b.sh" in line or "build_science_image.sh" in line:
                assert "candidate" not in line, line


def test_f5_closure_commit_lists_every_consumed_artifact(repo_root: Path) -> None:
    """A fresh clone must contain the evidence verification needs [finding 5, case M]."""
    text = (repo_root / "stage_acceptance/S00/10_REPRODUCE.md").read_text(encoding="utf-8")
    closure_block = text.split("git add ", 1)[1].split("git commit", 1)[0]
    for required in (
        "manifests/environments/<ENVIRONMENT_LOCK_SHA256>.json",
        "manifests/environments/S00B_IMAGE_RECORD.json",
        "artifacts/p0_pre/P0_PRE_READINESS.json",
        "artifacts/p0_pre/evidence/lanes/gpu_smoke.json",
        "stage_acceptance/S00/12_S00B_CLOSURE.json",
    ):
        assert required in closure_block, f"closure commit omits {required}"
