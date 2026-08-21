#!/usr/bin/env python3
"""`make preflight` — the ordered code-quality gate plus readiness evaluation.

Implements plan §10.1 steps 0-8, and carries the S00 foundations for:

* environment identity        ENVIRONMENT_LOCK_SHA256          [AUTH: 01 §12, §15, §32]
* provenance-bound readiness  P0_PRE_READY conjunction         [AUTH: 02 §C6; 01 §16]
* evidentiary-run clean-state guard (I15)                      [AUTH: 01 §35(3), §16, §36]

Steps 1-6 are CPU-only, so this file is exactly what CI runs. Step 7 never executes the
backend, GPU or benchmark lanes; it verifies their provenance-bound evidence records.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_repo_invariants import (  # noqa: E402
    ENVIRONMENT_LOCK_COMPONENTS,
    RUN_ID_INPUTS,
    RUN_MANIFEST_REQUIRED_FIELDS,
    Violation,
    run_all,
)

TBD = "TBD_REQUIRES_HARDWARE"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Readiness evidence, exactly the plan §11 keys [AUTH: 02 §C6].
EVIDENCE_KEYS: tuple[str, ...] = (
    "backend_contract",
    "synthetic_suite",
    "cache_assertions",
    "gpu_smoke",
    "benchmark_1000_seq",
)
#: The 00 §35 / §34C.1 software-gate conjunction, required alongside the above [plan §11.1].
SOFTWARE_GATE_KEYS: tuple[str, ...] = (
    "SCORER_ENGINE",
    "ANALYSIS_DRY_RUN",
    "CROSSFIT_NEGATIVE_CONTROL",
    "CACHE_ASSERTIONS",
    "P0_00_COMPUTE_BUDGET",
)

#: Paths whose modification changes what a run means [AUTH: 01 §35(3), §16, §27, §36].
PRODUCTION_PATHS: tuple[str, ...] = (
    "src/",
    "configs/",
    "specs/",
    "pyproject.toml",
    "uv.lock",
    "Dockerfile",
)

READINESS_REL = "artifacts/p0_pre/P0_PRE_READINESS.json"

#: Disjoint directories, not one directory keyed by name. On a case-insensitive filesystem a
#: single directory would let `cache_assertions.json` also satisfy `CACHE_ASSERTIONS`, so a
#: lane result would silently satisfy a software-gate slot [AUTH: 02 §C6; 00 §34C.1].
EVIDENCE_LANES_REL = "artifacts/p0_pre/evidence/lanes"
EVIDENCE_GATE_REL = "artifacts/p0_pre/evidence/software_gate"

#: Path under which a run manifest must live for a record to be evidentiary [AUTH: 01 §15].
RUN_MANIFEST_DIR_REL = "manifests/runs"


# --------------------------------------------------------------------------------------
# environment identity  [AUTH: 01 §12(5)-(9), §15, §16, §30, §32; plan §5.6]
# --------------------------------------------------------------------------------------


def environment_lock_sha256(manifest: dict[str, object]) -> str:
    """SHA256 over the complete frozen execution environment, never the lockfile alone.

    Returns TBD_REQUIRES_HARDWARE while any component is unresolved, so an unresolved
    environment cannot silently produce a well-formed identity.
    """
    parts: list[str] = []
    for key in ENVIRONMENT_LOCK_COMPONENTS:
        value = manifest.get(key)
        if not isinstance(value, str) or value == TBD or not value:
            return TBD
        parts.append(f"{key}={value}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def validate_environment_manifest(manifest: dict[str, object]) -> list[str]:
    """Return the list of §5.6 fields that are missing. Empty list means schema-complete."""
    required = list(ENVIRONMENT_LOCK_COMPONENTS) + [
        "nvidia_smi_capture",
        "gpu_uuid",
        "gpu_count",
        "dependency_versions",
        "bf16_fp32_tolerance",
        "nondeterminism_sources",
        "capture_timestamp_utc",
        "docker_image_tag",
        "environment_lock_sha256",
    ]
    return [k for k in required if k not in manifest]


def is_environment_lock_manifest(path: Path) -> bool:
    """A capture writes `<ENVIRONMENT_LOCK_SHA256>.json`. Anything else in that directory is
    metadata (an AI-stack record, a hardware probe) and is not an environment lock."""
    return bool(SHA256_RE.match(path.stem)) and path.suffix == ".json"


def environment_identity_status(root: Path) -> tuple[str, list[str]]:
    """Recompute the current environment identity from the frozen components.

    The declared `environment_lock_sha256` field is never trusted on its own: it is
    recomputed from the components and a manifest whose declared value disagrees is
    unusable, so a hand-edited identity cannot authorise anything
    [AUTH: 01 §12(5)-(9), §15, §32; 02 §C6].
    """
    problems: list[str] = []
    env_dir = root / "manifests" / "environments"
    if not env_dir.is_dir():
        return TBD, ["no manifests/environments directory"]
    for path in sorted(env_dir.glob("*.json")):
        if not is_environment_lock_manifest(path):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{rel}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(obj, dict):
            problems.append(f"{rel}: not a JSON object")
            continue
        recomputed = environment_lock_sha256(obj)
        if recomputed == TBD:
            problems.append(f"{rel}: components unresolved")
            continue
        declared = obj.get("environment_lock_sha256")
        if declared != recomputed:
            problems.append(f"{rel}: declared identity does not match the recomputed identity")
            continue
        return recomputed, problems
    return TBD, problems


def current_environment_lock(root: Path) -> str:
    return environment_identity_status(root)[0]


# --------------------------------------------------------------------------------------
# evidentiary-run clean-state guard  (I15 foundation)
# --------------------------------------------------------------------------------------


class CaptureFailure(RuntimeError):
    """A lane could not be recorded as a genuine current-environment PASS."""


class DirtyProductionTreeError(RuntimeError):
    """Raised before a run manifest is written when production paths are not clean."""


def production_tree_dirty(root: Path) -> list[str]:
    """Staged, unstaged and untracked changes under production paths.

    `git status --porcelain -uall` reports all three classes, so a run cannot be attributed
    to a clean commit while the scorer, configs or environment differ from it.
    """
    res = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "-uall"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise DirtyProductionTreeError(f"not a git worktree: {root}")
    dirty: list[str] = []
    for line in res.stdout.splitlines():
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:].strip().strip('"')
        path = path.split(" -> ")[-1]
        if any(path == p.rstrip("/") or path.startswith(p) for p in PRODUCTION_PATHS):
            dirty.append(f"{status} {path}")
    return sorted(dirty)


def assert_clean_production_tree(root: Path) -> None:
    """Gate every evidentiary run. S09's launcher calls this before writing a run manifest."""
    dirty = production_tree_dirty(root)
    if dirty:
        raise DirtyProductionTreeError(
            "refusing evidentiary execution; production tree is dirty [AUTH: 01 §35(3), §36]:\n"
            + "\n".join(dirty)
        )


# --------------------------------------------------------------------------------------
# step 7 — provenance-bound evidence verification  [AUTH: 01 §16; 02 §C6; 00 §34B.3]
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceVerdict:
    key: str
    accepted: bool
    status: str
    reason: str = ""


def _contained(root: Path, rel: object) -> Path | None:
    """Resolve `rel` under `root`, refusing traversal escapes."""
    if not isinstance(rel, str) or not rel:
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def verify_evidence_record(key: str, record: object, root: Path, env_lock: str) -> EvidenceVerdict:
    """Accept a record only if it is bound, end to end, to a real run.

    A complete-shaped forgery that points at README.md or an empty JSON object stays
    NON_EVIDENTIARY: the manifest must live under manifests/runs/, parse, carry the 01 §16
    minimum fields, and agree with the record on both RUN_ID and environment identity
    [AUTH: 01 §16; 02 §C6; 00 §34B.3].
    """

    def no(reason: str) -> EvidenceVerdict:
        return EvidenceVerdict(key, False, "NON_EVIDENTIARY", reason)

    if not isinstance(record, dict):
        return EvidenceVerdict(key, False, "NOT_RUN", "record is not an object")
    status = record.get("status")
    if not isinstance(status, str):
        return EvidenceVerdict(key, False, "NOT_RUN", "missing status")
    if status.startswith("NOT_RUN"):
        return EvidenceVerdict(key, False, status)
    if status != "PASS":
        return EvidenceVerdict(key, False, status, "status is not PASS")

    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not SHA256_RE.match(run_id):
        return no("missing or malformed run_id")

    recorded_env = record.get("environment_lock_sha256")
    if not isinstance(recorded_env, str) or not SHA256_RE.match(recorded_env):
        return no("missing or malformed environment_lock_sha256")
    if env_lock == TBD:
        return no("current environment identity is unresolved")
    if recorded_env != env_lock:
        return no(
            f"environment identity mismatch: record {recorded_env[:12]} "
            f"!= recomputed current {env_lock[:12]}"
        )

    manifest_rel = record.get("run_manifest")
    if not isinstance(manifest_rel, str) or not manifest_rel.startswith(RUN_MANIFEST_DIR_REL + "/"):
        return no(f"run_manifest is not under {RUN_MANIFEST_DIR_REL}/")
    manifest_path = _contained(root, manifest_rel)
    if manifest_path is None or not manifest_path.is_file():
        return no("run manifest absent")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return no(f"run manifest is not valid JSON ({exc.msg})")
    if not isinstance(manifest, dict):
        return no("run manifest is not a JSON object")

    missing = [f for f in RUN_MANIFEST_REQUIRED_FIELDS if f not in manifest]
    if missing:
        return no(
            f"run manifest missing 01 §16 fields: {', '.join(missing[:4])}"
            + (" ..." if len(missing) > 4 else "")
        )
    if manifest.get("run_id") != run_id:
        return no("run manifest RUN_ID does not match the evidence record")
    if manifest.get("environment_lock_sha256") != recorded_env:
        return no("run manifest environment identity does not match the evidence record")

    artifacts = record.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        return no("no artifact hashes")
    for rel, digest in artifacts.items():
        path = _contained(root, rel)
        if path is None or not path.is_file():
            return no(f"artifact absent: {rel}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return no(f"artifact hash mismatch: {rel}")
    return EvidenceVerdict(key, True, "PASS")


def _load_namespace(base: Path, keys: tuple[str, ...]) -> dict[str, object]:
    out: dict[str, object] = {}
    if not base.is_dir():
        return out
    for key in keys:
        path = base / f"{key}.json"
        if not path.is_file():
            continue
        # A case-insensitive filesystem resolves a differently-cased sibling; require the
        # on-disk name to match exactly so namespaces stay disjoint [AUTH: 02 §C6].
        if not any(entry.name == f"{key}.json" for entry in base.iterdir()):
            continue
        try:
            out[key] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out[key] = {"status": "PASS", "_malformed": True}
    return out


#: The producer whose output readiness re-derivation is allowed to trust [AUTH: 02 §C6].
READINESS_PRODUCER = "PREFLIGHT_STEP_8"

#: Publication marker for the lane evidence schema.
LANE_EVIDENCE_SCHEMA = "s00b.lane-evidence.v1"


def expected_lane_test_count(root: Path, lane: str) -> int:
    """How many tests the lane's authoritative suite defines.

    Recomputed from the suite on disk. A record that understates its own required count must
    not be able to authorise anything, so this value is never read from the evidence.

    Parametrised cases would make the collected count exceed the function count, so they are
    refused here rather than allowed to make the mandatory count silently wrong.
    """
    directory = root / "tests" / lane
    total = 0
    if not directory.is_dir():
        return 0
    for path in sorted(directory.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            for decorator in node.decorator_list:
                if "parametrize" in ast.dump(decorator):
                    raise CaptureFailure(
                        f"{lane}: {path.name}::{node.name} is parametrised, so the mandatory"
                        " test count cannot be derived from the suite definition"
                    )
            total += 1
    return total


def parse_junit(path: Path) -> dict[str, int]:
    """Machine-readable outcome. Text scraping cannot distinguish passed from skipped."""
    root_element = ElementTree.parse(path).getroot()
    suites = [root_element] if root_element.tag == "testsuite" else list(root_element)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, "0"))
    return counts


def invalidate_lane_evidence(root: Path, key: str) -> None:
    """Remove any previous record BEFORE a new attempt, so a stale PASS cannot survive it."""
    if key not in EVIDENCE_KEYS:
        raise ValueError(f"unknown lane {key!r}; expected one of {', '.join(EVIDENCE_KEYS)}")
    (root / EVIDENCE_LANES_REL / f"{key}.json").unlink(missing_ok=True)


def record_lane_evidence(
    root: Path,
    key: str,
    *,
    junit_xml: Path,
    pytest_status: int,
    detail: str = "",
    now: str | None = None,
) -> Path:
    """Persist a lane PASS, and only a real one. Every condition fails closed.

    A lane that executed and passed must not still read `NOT_RUN` in readiness. But a lane
    that skipped, partially collected, failed, or ran against an unresolved environment must
    never leave a PASS behind either — including a PASS left by an earlier successful attempt
    [AUTH: 02 §C6; 01 §16, §21].

    Acceptance as *readiness* evidence is a separate and stricter question, still decided by
    verify_evidence_record, which requires a run manifest that S01 introduces.
    """
    invalidate_lane_evidence(root, key)

    if pytest_status != 0:
        raise CaptureFailure(f"{key}: pytest exited {pytest_status}")
    if not junit_xml.is_file():
        raise CaptureFailure(f"{key}: no junit report at {junit_xml}")
    try:
        counts = parse_junit(junit_xml)
    except ElementTree.ParseError as exc:
        raise CaptureFailure(f"{key}: junit report is unparseable ({exc})") from exc

    expected = expected_lane_test_count(root, key)
    if expected == 0:
        raise CaptureFailure(f"{key}: the lane defines no tests, so it cannot report a state")
    if counts["tests"] != expected:
        raise CaptureFailure(
            f"{key}: collected {counts['tests']} tests but the suite requires exactly"
            f" {expected}; the mandatory suite did not run"
        )
    if counts["failures"] or counts["errors"]:
        raise CaptureFailure(f"{key}: {counts['failures']} failed, {counts['errors']} errored")
    if counts["skipped"]:
        raise CaptureFailure(
            f"{key}: {counts['skipped']} of {counts['tests']} tests skipped; a skipped"
            " hardware assertion is not a PASS"
        )

    identity = current_environment_lock(root)
    if identity == TBD:
        raise CaptureFailure(
            f"{key}: the environment identity is unresolved, so this cannot be a"
            " current-environment PASS; run `make env-capture` first"
        )

    stamp = now or datetime.datetime.now(datetime.UTC).isoformat()
    record = {
        "status": "PASS",
        "lane": key,
        "environment_lock_sha256": identity,
        "schema": LANE_EVIDENCE_SCHEMA,
        "observed": {
            "outcome": "PASS",
            "recorded_utc": stamp,
            "detail": detail.strip() or f"{counts['tests']} passed, 0 skipped",
            "tests": counts["tests"],
            "passed": counts["tests"] - counts["skipped"] - counts["failures"] - counts["errors"],
            "failures": counts["failures"],
            "errors": counts["errors"],
            "skipped": counts["skipped"],
            "expected_tests": expected,
        },
        "note": (
            "Lane execution record bound to the current environment. Acceptance as readiness"
            " evidence additionally requires a run manifest under manifests/runs/, which S01"
            " introduces [AUTH: 01 §16, §39 S01]."
        ),
    }
    base = root / EVIDENCE_LANES_REL
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{key}.json"
    handle, temporary = tempfile.mkstemp(dir=str(base), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


#: Fields the publication schema writes, and therefore the fields consumption must require.
LANE_OBSERVED_INT_FIELDS: tuple[str, ...] = (
    "tests",
    "passed",
    "failures",
    "errors",
    "skipped",
    "expected_tests",
)


def validate_lane_evidence(
    root: Path, key: str, *, record: object | None = None
) -> tuple[dict[str, object] | None, list[str]]:
    """THE canonical lane-evidence validator. One predicate, three consumers.

    Readiness derivation, runtime bundle verification and closure eligibility all call this,
    so a record cannot satisfy one and fail another. A truncated record carrying only an
    environment id and `observed.outcome = PASS` is rejected: consumption validates every
    field the publication schema writes [AUTH: 02 §C6; 01 §16, §21].
    """
    problems: list[str] = []
    if record is None:
        path = root / EVIDENCE_LANES_REL / f"{key}.json"
        if not path.is_file():
            return None, [f"{key}: no lane evidence recorded"]
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, [f"{key}: evidence is not valid JSON ({exc.msg})"]
    if not isinstance(record, dict):
        return None, [f"{key}: evidence is not a JSON object"]

    if record.get("schema") != LANE_EVIDENCE_SCHEMA:
        problems.append(
            f"{key}: schema marker is {record.get('schema')!r}, expected {LANE_EVIDENCE_SCHEMA!r}"
        )
    if record.get("status") != "PASS":
        problems.append(f"{key}: status is {record.get('status')!r}, expected 'PASS'")
    if record.get("lane") != key:
        problems.append(f"{key}: record declares lane {record.get('lane')!r}")

    identity = current_environment_lock(root)
    if identity == TBD:
        problems.append(f"{key}: no current environment identity to bind the evidence to")
    elif record.get("environment_lock_sha256") != identity:
        problems.append(
            f"{key}: evidence is bound to {str(record.get('environment_lock_sha256'))[:12]},"
            f" not to the current environment {identity[:12]}"
        )

    observed = record.get("observed")
    if not isinstance(observed, dict):
        problems.append(f"{key}: evidence carries no observed block")
        return None, problems
    if observed.get("outcome") != "PASS":
        problems.append(f"{key}: observed outcome is {observed.get('outcome')!r}")
    for name in LANE_OBSERVED_INT_FIELDS:
        value = observed.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"{key}: observed.{name} is missing or not an integer")
    if not isinstance(observed.get("recorded_utc"), str):
        problems.append(f"{key}: observed.recorded_utc is missing")
    if problems:
        return None, problems

    # The mandatory count comes from the authoritative suite, NOT from the record. A record
    # claiming expected_tests = 2 cannot authorise anything when the suite defines 8.
    try:
        required = expected_lane_test_count(root, key)
    except CaptureFailure as exc:
        return None, [str(exc)]
    if required <= 0:
        problems.append(f"{key}: the authoritative suite defines no tests")
    for name in ("expected_tests", "tests", "passed"):
        if int(observed[name]) != required:
            problems.append(
                f"{key}: observed.{name} = {observed[name]} but the authoritative suite"
                f" requires {required}"
            )
    for name in ("failures", "errors", "skipped"):
        if int(observed[name]) != 0:
            problems.append(f"{key}: observed.{name} = {observed[name]}, expected 0")
    return (None, problems) if problems else (record, [])


def current_lane_pass(root: Path, key: str) -> dict[str, object] | None:
    """Thin wrapper so no consumer can accidentally use a weaker predicate."""
    return validate_lane_evidence(root, key)[0]


def load_lane_evidence(root: Path) -> dict[str, object]:
    return _load_namespace(root / EVIDENCE_LANES_REL, EVIDENCE_KEYS)


def load_gate_evidence(root: Path) -> dict[str, object]:
    return _load_namespace(root / EVIDENCE_GATE_REL, SOFTWARE_GATE_KEYS)


# --------------------------------------------------------------------------------------
# step 8 — readiness  [AUTH: 02 §C6; plan §11.1]
# --------------------------------------------------------------------------------------


def compute_readiness(root: Path, computed_by: str = "PREFLIGHT_STEP_8") -> dict[str, object]:
    env_lock, env_problems = environment_identity_status(root)
    lane_records = load_lane_evidence(root)
    gate_records = load_gate_evidence(root)

    default_status = {
        "backend_contract": "NOT_RUN(BACKEND_NOT_INTEGRATED)",
        "synthetic_suite": "NOT_RUN(EMPTY_AT_S00)",
        "cache_assertions": "NOT_RUN(EMPTY_AT_S00)",
        "gpu_smoke": "NOT_RUN(NO_GPU)",
        "benchmark_1000_seq": "NOT_RUN",
    }

    evidence: dict[str, object] = {}
    accepted: dict[str, bool] = {}
    for key in EVIDENCE_KEYS:
        if key in lane_records:
            verdict = verify_evidence_record(key, lane_records[key], root, env_lock)
        else:
            verdict = EvidenceVerdict(key, False, default_status[key])
        accepted[key] = verdict.accepted
        entry: dict[str, object] = {
            "status": verdict.status
            if verdict.accepted
            else (verdict.status or default_status[key])
        }
        if verdict.reason:
            entry["reason"] = verdict.reason
        # A lane that ran must say so, even when its record is not yet accepted as evidence.
        validated, _ = validate_lane_evidence(root, key, record=lane_records.get(key))
        if validated is not None:
            entry["observed"] = validated["observed"]
        evidence[key] = entry

    software_gate: dict[str, object] = {}
    gate_ok = True
    for key in SOFTWARE_GATE_KEYS:
        if key in gate_records:
            verdict = verify_evidence_record(key, gate_records[key], root, env_lock)
            entry = {"status": "VERIFIED" if verdict.accepted else "NOT_RUN"}
            if verdict.reason:
                entry["reason"] = verdict.reason
            software_gate[key] = entry
            gate_ok = gate_ok and verdict.accepted
        else:
            software_gate[key] = {"status": "NOT_RUN"}
            gate_ok = False

    backend_integrated = accepted["backend_contract"]
    all_evidence = all(accepted.values())
    environment_resolved = env_lock != TBD
    #: SUITE_SCOPE is part of the same conjunction as P0_PRE_READY: the suite may only be
    #: called the integrated production stack when the whole software/backend gate passes
    #: [AUTH: 02 §C6; 00 §35, §34C.1].
    fully_integrated = bool(
        backend_integrated and all_evidence and gate_ok and environment_resolved
    )

    return {
        "BACKEND_INTEGRATED": backend_integrated,
        "SUITE_SCOPE": (
            "INTEGRATED_PRODUCTION_STACK" if fully_integrated else "STATISTICAL_STACK_ONLY"
        ),
        "P0_PRE_READY": fully_integrated,
        "computed_by": computed_by,
        "environment_lock_sha256": env_lock,
        "environment_problems": env_problems,
        "evidence": evidence,
        "evidence_namespaces": {
            "lanes": EVIDENCE_LANES_REL,
            "software_gate": EVIDENCE_GATE_REL,
        },
        "software_gate": software_gate,
        "authority": "02 §C6; 00 §35, §34C.1",
    }


def write_readiness(root: Path, readiness: dict[str, object]) -> Path:
    path = root / READINESS_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# the ordered gate  [AUTH: 01 §22, §33; plan §10.1]
# --------------------------------------------------------------------------------------


@dataclass
class Step:
    index: int
    name: str
    command: tuple[str, ...] = ()
    authority: str = ""


PREFLIGHT_STEPS: tuple[Step, ...] = (
    Step(0, "repo-invariants", (), "01 §15, §18; 03 §9, §13"),
    Step(1, "format", ("make", "format", "CHECK=1"), "01 §33"),
    Step(2, "lint", ("make", "lint"), "01 §33"),
    Step(3, "typecheck", ("make", "typecheck"), "01 §33"),
    Step(4, "unit", ("make", "unit"), "01 §22, §33"),
    Step(5, "integration", ("make", "integration"), "01 §22, §33"),
    Step(6, "synthetic", ("make", "synthetic"), "01 §22, §33"),
    Step(7, "verify-evidence", (), "02 §C6; 01 §16; 00 §34C.1"),
    Step(8, "write-readiness", (), "02 §C6"),
)


@dataclass
class StepResult:
    step: Step
    ok: bool
    detail: str = ""


@dataclass
class GateOutcome:
    results: list[StepResult] = field(default_factory=list)
    halted_at: int | None = None

    @property
    def ok(self) -> bool:
        return self.halted_at is None


def run_steps(steps: tuple[Step, ...], runner: Callable[[Step], StepResult]) -> GateOutcome:
    """Run steps in order, halting at the first failure [AUTH: 01 §22]."""
    outcome = GateOutcome()
    for step in steps:
        result = runner(step)
        outcome.results.append(result)
        if not result.ok:
            outcome.halted_at = step.index
            break
    return outcome


def _make_runner(root: Path, verbose: bool) -> Callable[[Step], StepResult]:
    def runner(step: Step) -> StepResult:
        if step.index == 0:
            violations: list[Violation] = run_all(root)
            for v in violations:
                print(v.render())
            return StepResult(step, not violations, f"{len(violations)} violation(s)")
        if step.index == 7:
            readiness = compute_readiness(root)
            return StepResult(step, True, json.dumps(readiness["evidence"], sort_keys=True))
        if step.index == 8:
            readiness = compute_readiness(root)
            path = write_readiness(root, readiness)
            return StepResult(
                step, True, f"{path.relative_to(root)} P0_PRE_READY={readiness['P0_PRE_READY']}"
            )
        res = subprocess.run(
            list(step.command), cwd=root, capture_output=not verbose, text=True, check=False
        )
        detail = "" if verbose else (res.stdout or "") + (res.stderr or "")
        return StepResult(step, res.returncode == 0, detail.strip())

    return runner


BACKEND_INTEGRATED_TRUE = 0
BACKEND_INTEGRATED_FALSE = 1
BACKEND_READINESS_UNREADABLE = 2


def backend_integrated_exit_code(readiness_path: Path) -> tuple[int, str]:
    """Decide the `make backend-contract` branch. Never swallows an I/O error.

    0 -> run the lane; 1 -> NOT_RUN(BACKEND_NOT_INTEGRATED); 2 -> the readiness file is
    missing or unreadable, which is a hard failure rather than a NOT_RUN [AUTH: 02 §C6].
    """
    try:
        obj = json.loads(readiness_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return BACKEND_READINESS_UNREADABLE, f"readiness file not found: {readiness_path}"
    except json.JSONDecodeError as exc:
        return BACKEND_READINESS_UNREADABLE, f"readiness file is not valid JSON: {exc.msg}"
    except OSError as exc:  # pragma: no cover - surfaced verbatim
        return BACKEND_READINESS_UNREADABLE, f"readiness file unreadable: {exc}"
    if not isinstance(obj, dict) or "BACKEND_INTEGRATED" not in obj:
        return BACKEND_READINESS_UNREADABLE, "readiness file has no BACKEND_INTEGRATED field"
    value = obj["BACKEND_INTEGRATED"]
    if value is True:
        return BACKEND_INTEGRATED_TRUE, "BACKEND_INTEGRATED is true"
    return BACKEND_INTEGRATED_FALSE, "BACKEND_INTEGRATED is false"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ordered §33 gate + readiness evaluation")
    ap.add_argument("--root", default=".")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--record-lane",
        metavar="LANE",
        help="persist a lane execution record into the evidence namespace",
    )
    ap.add_argument("--junit-xml", dest="junit_xml", default=None)
    ap.add_argument("--pytest-status", dest="pytest_status", type=int, default=None)
    ap.add_argument("--detail", default="")
    ap.add_argument("--invalidate-lane", dest="invalidate_lane", default=None)
    ap.add_argument(
        "--backend-integrated",
        metavar="READINESS_JSON",
        help="exit 0 if BACKEND_INTEGRATED, 1 if not, 2 if unreadable",
    )
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.invalidate_lane:
        invalidate_lane_evidence(root, args.invalidate_lane)
        print(f"invalidated any previous {args.invalidate_lane} evidence")
        return 0

    if args.record_lane:
        if args.junit_xml is None or args.pytest_status is None:
            print("--record-lane requires --junit-xml and --pytest-status", file=sys.stderr)
            return 2
        try:
            written = record_lane_evidence(
                root,
                args.record_lane,
                junit_xml=Path(args.junit_xml),
                pytest_status=args.pytest_status,
                detail=args.detail,
            )
        except (CaptureFailure, ValueError, OSError) as exc:
            print(f"{args.record_lane}: NOT RECORDED AS PASS - {exc}", file=sys.stderr)
            return 1
        print(f"recorded {args.record_lane} = PASS in {written.relative_to(root)}")
        return 0

    if args.backend_integrated:
        code, message = backend_integrated_exit_code(Path(args.backend_integrated))
        if code == BACKEND_READINESS_UNREADABLE:
            print(f"backend-contract gate: {message}", file=sys.stderr)
        return code

    print(f"preflight: RUN_ID inputs frozen at {len(RUN_ID_INPUTS)} [AUTH: 01 §15]")
    outcome = run_steps(PREFLIGHT_STEPS, _make_runner(root, args.verbose))
    for result in outcome.results:
        mark = "ok  " if result.ok else "FAIL"
        print(f"  step {result.step.index} {mark} {result.step.name}")
        if not result.ok and result.detail:
            print(result.detail)
    if outcome.halted_at is not None:
        print(f"preflight HALTED at step {outcome.halted_at} [AUTH: 01 §22 ordering]")
        return 1
    print("preflight PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
