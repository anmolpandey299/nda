"""Fail-closed GPU lane evidence and commit-identity hygiene.

Covers reconciled findings BLOCKER 1 and BLOCKER 3 at the unit level; the lifecycle and
closure cases that need a real repository live in the integration lane.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _helpers import write_environment_manifest
from preflight import (
    TBD,
    CaptureFailure,
    current_lane_pass,
    expected_lane_test_count,
    invalidate_lane_evidence,
    main,
    parse_junit,
    record_lane_evidence,
)

LANE = "gpu_smoke"


def _fake_lane(root: Path, count: int = 8) -> None:
    directory = root / "tests" / LANE
    directory.mkdir(parents=True, exist_ok=True)
    body = "".join(f"def test_case_{i}() -> None: ...\n" for i in range(count))
    (directory / "test_lane.py").write_text(body, encoding="utf-8")


def _junit(root: Path, *, tests: int, failures: int = 0, errors: int = 0, skipped: int = 0) -> Path:
    path = root / "report.xml"
    path.write_text(
        f'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="{tests}" '
        f'failures="{failures}" errors="{errors}" skipped="{skipped}"/></testsuites>',
        encoding="utf-8",
    )
    return path


def _ready(root: Path, count: int = 8) -> str:
    _fake_lane(root, count)
    return write_environment_manifest(root)


# ============================================================== 3 — a genuine PASS
def test_3_eight_passed_zero_skipped_is_an_eligible_pass(tmp_path: Path) -> None:
    identity = _ready(tmp_path)
    written = record_lane_evidence(
        tmp_path,
        LANE,
        junit_xml=_junit(tmp_path, tests=8),
        pytest_status=0,
        detail="8 passed, 0 skipped in 3.1s",
    )
    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    assert record["observed"]["outcome"] == "PASS"
    assert record["observed"]["tests"] == 8
    assert record["observed"]["skipped"] == 0
    assert record["environment_lock_sha256"] == identity
    assert current_lane_pass(tmp_path, LANE) is not None


# ============================================================== 4, 5 — skips are not a PASS
def test_4_two_passed_six_skipped_is_not_a_pass(tmp_path: Path) -> None:
    _ready(tmp_path)
    with pytest.raises(CaptureFailure, match="skipped"):
        record_lane_evidence(
            tmp_path, LANE, junit_xml=_junit(tmp_path, tests=8, skipped=6), pytest_status=0
        )
    assert current_lane_pass(tmp_path, LANE) is None


def test_4_partial_collection_is_not_a_pass(tmp_path: Path) -> None:
    _ready(tmp_path, count=8)
    with pytest.raises(CaptureFailure, match="required suite did not run"):
        record_lane_evidence(tmp_path, LANE, junit_xml=_junit(tmp_path, tests=2), pytest_status=0)
    assert current_lane_pass(tmp_path, LANE) is None


def test_5_unresolved_environment_is_not_a_current_environment_pass(tmp_path: Path) -> None:
    """CUDA unavailable means capture never published an identity, so no PASS can bind."""
    _fake_lane(tmp_path)
    with pytest.raises(CaptureFailure, match="environment identity is unresolved"):
        record_lane_evidence(tmp_path, LANE, junit_xml=_junit(tmp_path, tests=8), pytest_status=0)
    assert current_lane_pass(tmp_path, LANE) is None


def test_5_pass_bound_to_another_environment_is_not_current(tmp_path: Path) -> None:
    _ready(tmp_path)
    record_lane_evidence(tmp_path, LANE, junit_xml=_junit(tmp_path, tests=8), pytest_status=0)
    path = tmp_path / "artifacts/p0_pre/evidence/lanes/gpu_smoke.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["environment_lock_sha256"] = "f" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    assert current_lane_pass(tmp_path, LANE) is None


# ============================================================== 6, 8 — no stale PASS survives
def test_6_pytest_failure_records_nothing(tmp_path: Path) -> None:
    _ready(tmp_path)
    with pytest.raises(CaptureFailure, match="pytest exited"):
        record_lane_evidence(
            tmp_path, LANE, junit_xml=_junit(tmp_path, tests=8, failures=1), pytest_status=1
        )
    assert current_lane_pass(tmp_path, LANE) is None


def test_8_previous_pass_cannot_survive_a_failed_rerun(tmp_path: Path) -> None:
    _ready(tmp_path)
    record_lane_evidence(tmp_path, LANE, junit_xml=_junit(tmp_path, tests=8), pytest_status=0)
    assert current_lane_pass(tmp_path, LANE) is not None

    with pytest.raises(CaptureFailure):
        record_lane_evidence(
            tmp_path, LANE, junit_xml=_junit(tmp_path, tests=8, failures=2), pytest_status=1
        )
    assert current_lane_pass(tmp_path, LANE) is None, "a stale PASS survived a failed rerun"


def test_8_invalidation_happens_before_the_attempt(tmp_path: Path) -> None:
    _ready(tmp_path)
    record_lane_evidence(tmp_path, LANE, junit_xml=_junit(tmp_path, tests=8), pytest_status=0)
    invalidate_lane_evidence(tmp_path, LANE)
    assert not (tmp_path / "artifacts/p0_pre/evidence/lanes/gpu_smoke.json").exists()


def test_6_missing_or_unparseable_junit_records_nothing(tmp_path: Path) -> None:
    _ready(tmp_path)
    with pytest.raises(CaptureFailure, match="no junit report"):
        record_lane_evidence(tmp_path, LANE, junit_xml=tmp_path / "absent.xml", pytest_status=0)
    broken = tmp_path / "broken.xml"
    broken.write_text("<not-xml", encoding="utf-8")
    with pytest.raises(CaptureFailure, match="unparseable"):
        record_lane_evidence(tmp_path, LANE, junit_xml=broken, pytest_status=0)
    assert current_lane_pass(tmp_path, LANE) is None


# ============================================================== 7 — recorder failure fails
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_7_recorder_write_failure_fails_the_command(tmp_path: Path) -> None:
    _ready(tmp_path)
    lanes = tmp_path / "artifacts" / "p0_pre" / "evidence" / "lanes"
    lanes.mkdir(parents=True)
    lanes.chmod(0o555)
    try:
        code = main(
            [
                "--root",
                str(tmp_path),
                "--record-lane",
                LANE,
                "--junit-xml",
                str(_junit(tmp_path, tests=8)),
                "--pytest-status",
                "0",
            ]
        )
        assert code != 0, "an unwritable evidence namespace must fail the command"
        assert not (lanes / "gpu_smoke.json").exists()
    finally:
        lanes.chmod(0o755)


def test_7_cli_refuses_to_record_without_a_report(tmp_path: Path) -> None:
    _ready(tmp_path)
    assert main(["--root", str(tmp_path), "--record-lane", LANE]) == 2
    assert current_lane_pass(tmp_path, LANE) is None


def test_7_cli_returns_nonzero_on_a_skipped_suite(tmp_path: Path) -> None:
    _ready(tmp_path)
    code = main(
        [
            "--root",
            str(tmp_path),
            "--record-lane",
            LANE,
            "--junit-xml",
            str(_junit(tmp_path, tests=8, skipped=1)),
            "--pytest-status",
            "0",
        ]
    )
    assert code == 1
    assert current_lane_pass(tmp_path, LANE) is None


# ============================================================== helpers behave
def test_expected_lane_count_reads_the_real_suite(repo_root: Path) -> None:
    assert expected_lane_test_count(repo_root, LANE) >= 8


def test_junit_parser_sums_multiple_suites(tmp_path: Path) -> None:
    path = tmp_path / "r.xml"
    path.write_text(
        '<?xml version="1.0"?><testsuites>'
        '<testsuite tests="3" failures="0" errors="0" skipped="1"/>'
        '<testsuite tests="5" failures="1" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )
    assert parse_junit(path) == {"tests": 8, "failures": 1, "errors": 0, "skipped": 1}


def test_unknown_lane_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown lane"):
        invalidate_lane_evidence(tmp_path, "not_a_lane")


# ============================================================== 1 — SHA hygiene
def test_1_bootstrap_rejects_a_malformed_or_absent_commit(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "bootstrap_runpod_s00b.sh").read_text(encoding="utf-8")
    assert "full 40-hex SHA obtained from git rev-parse" in script
    assert "git cat-file -e" in script
    assert "is not a commit in this repository" in script


def test_1_no_document_cites_the_fabricated_sha(repo_root: Path) -> None:
    """A previously reported SHA was an abbreviation expanded by hand. It does not exist and
    must never appear anywhere."""
    fabricated = "0e6f16f2c8a5d419b7e3f06c9a2d5b8e4c1f7a30"
    for path in repo_root.rglob("*"):
        if ".git/" in path.as_posix() or ".venv" in path.parts:
            continue
        if path.is_file() and path.suffix in {".md", ".py", ".sh", ".json", ".txt", ".yml"}:
            if path.name == "test_s00b_reconciled.py":
                continue
            assert fabricated not in path.read_text(encoding="utf-8", errors="ignore"), path


def test_1_environment_identities_are_not_hardcoded(repo_root: Path) -> None:
    for identity in (
        "aa64d5f2f87854f1527b79d2a11009cdf7f183b3cc8977e6a0db4a8ad90dbe9e",
        "b7356e1c0a1c323f035f3884d012cb63c93d7798d8e9b064d810b298748718dc",
    ):
        for directory in ("src", "scripts", "configs"):
            base = repo_root / directory
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if path.is_file():
                    assert identity not in path.read_text(encoding="utf-8", errors="ignore")


def test_10_s00b_pass_is_separate_from_p0_evidence_eligibility(tmp_path: Path) -> None:
    """A genuine S00-B hardware PASS may still be NON_EVIDENTIARY for P0 readiness."""
    from preflight import compute_readiness

    _ready(tmp_path)
    record_lane_evidence(
        tmp_path,
        LANE,
        junit_xml=_junit(tmp_path, tests=8),
        pytest_status=0,
        detail="8 passed, 0 skipped",
    )
    assert current_lane_pass(tmp_path, LANE) is not None  # S00-B: satisfied

    readiness = compute_readiness(tmp_path)
    lanes = readiness["evidence"]
    assert isinstance(lanes, dict)
    entry = lanes[LANE]
    assert isinstance(entry, dict)
    assert entry["status"] == "NON_EVIDENTIARY"  # P0: not yet eligible
    assert entry["observed"]["outcome"] == "PASS"
    assert readiness["BACKEND_INTEGRATED"] is False
    assert readiness["SUITE_SCOPE"] == "STATISTICAL_STACK_ONLY"
    assert readiness["P0_PRE_READY"] is False
    assert readiness["environment_lock_sha256"] != TBD
