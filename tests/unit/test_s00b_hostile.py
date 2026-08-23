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
    assert record is None and any("authoritative suite" in p for p in problems)


def test_f1_passed_must_account_for_every_collected_test(tmp_path: Path) -> None:
    published = _published(tmp_path)
    observed = published["observed"]
    assert isinstance(observed, dict)
    observed["passed"] = 3
    _write(tmp_path, published)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None and any("observed.passed" in p for p in problems)


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
    # The executable `git add` derives the list, so the artifacts are named in the
    # explanatory listing, with the environment manifest in template form.
    closure_block = text.split("The closure commit must contain exactly these artifacts:", 1)[1]
    closure_block = closure_block.split("```", 2)[1]
    for required in (
        "manifests/environments/<ENVIRONMENT_LOCK_SHA256>.json",
        "manifests/environments/S00B_IMAGE_RECORD.json",
        "artifacts/p0_pre/P0_PRE_READINESS.json",
        "artifacts/p0_pre/evidence/lanes/gpu_smoke.json",
        "stage_acceptance/S00/12_S00B_CLOSURE.json",
    ):
        assert required in closure_block, f"closure commit omits {required}"


# ============================================================== F01 — authoritative count
def test_f01_understated_expected_count_is_rejected(tmp_path: Path) -> None:
    """The suite defines 8; a record claiming it only needed 2 must authorise nothing."""
    _lane(tmp_path, count=8)
    identity = write_environment_manifest(tmp_path)
    _write(
        tmp_path,
        {
            "schema": LANE_EVIDENCE_SCHEMA,
            "status": "PASS",
            "lane": LANE,
            "environment_lock_sha256": identity,
            "observed": {
                "outcome": "PASS",
                "recorded_utc": "2026-08-21T00:00:00+00:00",
                "detail": "2/2",
                "tests": 2,
                "passed": 2,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "expected_tests": 2,
            },
        },
    )
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert record is None, "an understated required count was accepted"
    assert current_lane_pass(tmp_path, LANE) is None
    assert any("authoritative suite requires 8" in p for p in problems), problems


def test_f01_complete_authoritative_count_is_accepted(tmp_path: Path) -> None:
    _published(tmp_path)
    record, problems = validate_lane_evidence(tmp_path, LANE)
    assert problems == [] and record is not None
    observed = record["observed"]
    assert isinstance(observed, dict)
    assert observed["tests"] == observed["passed"] == observed["expected_tests"] == 8


def test_f01_recorder_refuses_a_partial_run(tmp_path: Path) -> None:
    from preflight import CaptureFailure

    _lane(tmp_path, count=8)
    write_environment_manifest(tmp_path)
    with pytest.raises(CaptureFailure, match="requires exactly 8"):
        record_lane_evidence(tmp_path, LANE, junit_xml=_junit(tmp_path, tests=2), pytest_status=0)


def test_f01_parametrised_lane_cannot_derive_a_mandatory_count(tmp_path: Path) -> None:
    """A parametrised case would make the collected count exceed the definition count."""
    from preflight import CaptureFailure, expected_lane_test_count

    directory = tmp_path / "tests" / LANE
    directory.mkdir(parents=True)
    (directory / "test_lane.py").write_text(
        "import pytest\n"
        '@pytest.mark.parametrize("x", [1, 2])\n'
        "def test_case(x: int) -> None: ...\n",
        encoding="utf-8",
    )
    with pytest.raises(CaptureFailure, match="parametrised"):
        expected_lane_test_count(tmp_path, LANE)


def test_f01_live_lane_count_matches_the_real_suite(repo_root: Path) -> None:
    from preflight import expected_lane_test_count

    assert expected_lane_test_count(repo_root, LANE) == 8


# ============================================================== F05 — one artifact list
#: The environment manifest is named after an identity capture computes on the H100, so it
#: cannot appear literally in a static document. Documentation carries the template.
ENV_MANIFEST_TEMPLATE = "manifests/environments/<ENVIRONMENT_LOCK_SHA256>.json"
CONCRETE_ENV_MANIFEST = re.compile(r"^manifests/environments/[0-9a-f]{64}\.json$")


def _documented_form(rel: str) -> str:
    """A concrete runtime manifest path documents as its template; everything else is
    literal."""
    return ENV_MANIFEST_TEMPLATE if CONCRETE_ENV_MANIFEST.match(rel) else rel


def test_f05_bootstrap_and_runbook_share_one_authority(repo_root: Path) -> None:
    """Both DERIVE the list from build_bundle, so they cannot disagree."""
    bootstrap = (repo_root / "scripts/bootstrap_runpod_s00b.sh").read_text(encoding="utf-8")
    runbook = (repo_root / "stage_acceptance/S00/10_REPRODUCE.md").read_text(encoding="utf-8")
    assert "--closure-artifacts" in bootstrap, "bootstrap must derive, not retype, the list"
    executable = runbook.split("git add ", 1)[1].split("git commit", 1)[0]
    assert "--closure-artifacts" in executable, (
        "the runbook's git add must derive the list, not hardcode paths that include a "
        "runtime identity"
    )


def test_f05_documented_artifacts_match_the_canonical_list(repo_root: Path) -> None:
    """The explanatory listing must name the same five artifacts, with the environment
    manifest in template form because its identity does not exist until capture runs."""
    import build_bundle

    canonical = build_bundle.closure_artifact_paths(repo_root, "S00")
    runbook = (repo_root / "stage_acceptance/S00/10_REPRODUCE.md").read_text(encoding="utf-8")
    listing = runbook.split("The closure commit must contain exactly these artifacts:", 1)[1]
    listing = listing.split("```", 2)[1]
    for rel in canonical:
        assert _documented_form(rel) in listing, f"the runbook omits {rel}"
    assert ENV_MANIFEST_TEMPLATE in listing


def test_f05_template_is_never_mistaken_for_a_concrete_path() -> None:
    concrete = "manifests/environments/" + "a" * 64 + ".json"
    assert _documented_form(concrete) == ENV_MANIFEST_TEMPLATE
    assert CONCRETE_ENV_MANIFEST.match(ENV_MANIFEST_TEMPLATE) is None
    for rel in (
        "manifests/environments/S00B_IMAGE_RECORD.json",
        "artifacts/p0_pre/P0_PRE_READINESS.json",
        "artifacts/p0_pre/evidence/lanes/gpu_smoke.json",
        "stage_acceptance/S00/12_S00B_CLOSURE.json",
    ):
        assert _documented_form(rel) == rel, "only the environment manifest is a template"


def test_f05_concrete_manifest_is_still_required_after_capture(tmp_path: Path) -> None:
    """The real requirement is untouched: once capture has run, the concrete manifest is in
    the canonical list that the closure commit adds."""
    import build_bundle

    identity = write_environment_manifest(tmp_path)
    canonical = build_bundle.closure_artifact_paths(tmp_path, "S00")
    assert f"manifests/environments/{identity}.json" in canonical
    assert ENV_MANIFEST_TEMPLATE not in canonical, (
        "after capture the list must resolve, not stay a template"
    )


def test_f05_canonical_list_covers_every_verifier(repo_root: Path) -> None:
    """A fresh clone with these artifacts can run all three verifiers."""
    import build_bundle

    canonical = build_bundle.closure_artifact_paths(repo_root, "S00")
    assert any("manifests/environments/" in p and "IMAGE_RECORD" not in p for p in canonical)
    assert "manifests/environments/S00B_IMAGE_RECORD.json" in canonical
    assert "artifacts/p0_pre/P0_PRE_READINESS.json" in canonical
    assert "artifacts/p0_pre/evidence/lanes/gpu_smoke.json" in canonical
    assert "stage_acceptance/S00/12_S00B_CLOSURE.json" in canonical


# ============================================================== image-record lifecycle
def test_image_record_is_derived_from_the_running_image(tmp_path: Path) -> None:
    """The record describes the image capture ran inside, not one carried in from a build."""
    import capture_environment as cap

    manifest: dict[str, object] = {
        "docker_image_digest": "sha256:" + "a" * 64,
        "image_source_git_commit": "b" * 40,
        "docker_image_tag": "registry/repo",
        "base_image_digest": "sha256:" + "c" * 64,
        "image_lane": "science",
        "environment_lock_sha256": "d" * 64,
        "image_identity_source": "BAKED_INTO_IMAGE",
    }
    record = cap.image_record_from_environment(manifest)
    assert record["sealed_image_digest"] == manifest["docker_image_digest"]
    assert record["source_git_commit"] == manifest["image_source_git_commit"]

    written = cap.write_image_record(tmp_path, manifest)
    assert written == tmp_path / "manifests/environments/S00B_IMAGE_RECORD.json"
    assert json.loads(written.read_text(encoding="utf-8")) == record


@pytest.mark.parametrize("field", ["docker_image_digest", "image_source_git_commit"])
def test_image_record_refuses_an_unresolved_identity(tmp_path: Path, field: str) -> None:
    import capture_environment as cap

    manifest: dict[str, object] = {
        "docker_image_digest": "sha256:" + "a" * 64,
        "image_source_git_commit": "b" * 40,
    }
    manifest[field] = "TBD_REQUIRES_HARDWARE"
    with pytest.raises(cap.CaptureError, match="cannot materialise"):
        cap.image_record_from_environment(manifest)


def test_build_script_does_not_write_the_record_into_tracked_source(
    repo_root: Path,
) -> None:
    """Writing a completed record for image A into source bakes A into the next build
    commit, which is the cycle this fix removes."""
    script = (repo_root / "scripts/build_science_image.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))
    assert "S00B_IMAGE_RECORD" not in code
    assert "manifests/environments" not in code


def test_capture_materialises_the_record_after_the_environment(repo_root: Path) -> None:
    source = (repo_root / "scripts/capture_environment.py").read_text(encoding="utf-8")
    publish = source.split("def publish(", 1)[1]
    assert "write_image_record(root, finalised)" in publish
    assert publish.index("os.replace(temporary") < publish.index("write_image_record")


def test_bootstrap_requires_the_materialised_record(repo_root: Path) -> None:
    script = (repo_root / "scripts/bootstrap_runpod_s00b.sh").read_text(encoding="utf-8")
    assert "did not materialise the image record" in script
