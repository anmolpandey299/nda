"""Backend-contract evidence that cannot be hand-authored [AUTH: 01 §15, §16, §36; 02 §C6].

`{"status": "PASS"}` is not evidence. A backend-contract record is accepted only when every
identity it carries still matches the state it claims to describe: the run, the commit, the
environment lock, the model manifest and revision, the backend and scoring code, the resolved
config, and the hashes of the artifacts the run actually produced.

Verification recomputes each binding against the *current* repository and run state rather
than comparing the file to itself, so a forged record fails on the first identity that does
not belong to it. Timestamps and the exit code are recorded too: a record claiming PASS with
a non-zero exit, or with an end before its start, describes no run that happened.

Nothing here sets `P0_PRE_READY`. It builds and checks the evidence that a later readiness
conjunction will consume [AUTH: 02 §C6].
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.provenance.hashing import (
    JSONDocument,
    JSONValue,
    sha256_canonical,
    sha256_file,
    write_canonical_json,
)

EVIDENCE_SCHEMA: Final = "backend.contract-evidence.v1"

PASS: Final = "PASS"
FAIL: Final = "FAIL"
NOT_RUN: Final = "NOT_RUN"

#: The stable causes a check may report. Collapsing them to one string would lose the only
#: information a NOT_RUN row carries [AUTH: 02 §C6].
MODEL_REVISION_NOT_FROZEN: Final = "MODEL_REVISION_NOT_FROZEN"
BACKEND_NOT_INSTALLED: Final = "BACKEND_NOT_INSTALLED"
CHECKPOINT_NOT_ACQUIRED: Final = "CHECKPOINT_NOT_ACQUIRED"
NO_H100: Final = "NO_H100"
NO_PARAMETER_NAMES: Final = "NO_PARAMETER_NAMES"
OPACUS_BACKEND_NOT_INSTALLED: Final = "OPACUS_BACKEND_NOT_INSTALLED"
STRUCTURAL_CHECK_FAILED: Final = "STRUCTURAL_CHECK_FAILED"

NOT_RUN_REASONS: Final[tuple[str, ...]] = (
    MODEL_REVISION_NOT_FROZEN,
    BACKEND_NOT_INSTALLED,
    CHECKPOINT_NOT_ACQUIRED,
    NO_H100,
    NO_PARAMETER_NAMES,
    OPACUS_BACKEND_NOT_INSTALLED,
    STRUCTURAL_CHECK_FAILED,
)

#: Every binding a real backend-contract record must carry [AUTH: 01 §16].
REQUIRED_BINDINGS: Final[tuple[str, ...]] = (
    "schema",
    "status",
    "run_id",
    "git_commit",
    "environment_lock_sha256",
    "model_manifest_sha256",
    "model_revision",
    "backend_code_sha256",
    "scoring_code_sha256",
    "resolved_config_sha256",
    "attempt_id",
    "artifact_hashes",
    "started_utc",
    "ended_utc",
    "exit_code",
    "checks",
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_TIMESTAMP_RE: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class EvidenceError(ValueError):
    """A backend-contract evidence record is malformed or does not bind its run."""


@dataclass(frozen=True)
class RunBinding:
    """The current state a record is verified against. Recomputed, never read from the file.

    `run_id` is **supplied**, never derived here. 01 §15/§16 run identity belongs to the S09
    run-manifest layer; a backend that hashed its own timestamp and called the result a RUN_ID
    would be minting scientific provenance it has no authority to issue. The local,
    non-evidentiary identifier this command generates for its own bookkeeping is
    `attempt_id`, which is deliberately a different field with a different name.
    """

    run_id: str
    git_commit: str
    environment_lock_sha256: str
    model_manifest_sha256: str
    model_revision: str
    backend_code_sha256: str
    scoring_code_sha256: str
    resolved_config_sha256: str
    attempt_id: str = ""

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "environment_lock_sha256": self.environment_lock_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "model_revision": self.model_revision,
            "backend_code_sha256": self.backend_code_sha256,
            "scoring_code_sha256": self.scoring_code_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
        }

    def with_attempt(self) -> dict[str, JSONValue]:
        """The bindings plus the local attempt identifier, for serialisation."""
        return {**self.as_dict(), "attempt_id": self.attempt_id}


def aggregate_status(checks: Sequence[Mapping[str, JSONValue]]) -> str:
    """The frozen aggregation: any FAIL -> FAIL, else any NOT_RUN -> NOT_RUN, else PASS.

    Stated once and enforced by the verifier independently of whoever wrote the record, so a
    hand-authored `NOT_RUN` overall status cannot sit above a failing child check.
    """
    if not checks:
        raise EvidenceError("a backend-contract record must list the checks it ran")
    statuses = [str(check.get("status")) for check in checks]
    unknown = sorted({s for s in statuses} - {PASS, FAIL, NOT_RUN})
    if unknown:
        raise EvidenceError(f"check status(es) {unknown} are not PASS, FAIL or NOT_RUN")
    if FAIL in statuses:
        return FAIL
    if NOT_RUN in statuses:
        return NOT_RUN
    return PASS


def build_contract_evidence(
    *,
    binding: RunBinding,
    status: str,
    checks: Sequence[Mapping[str, JSONValue]],
    artifact_hashes: Mapping[str, str],
    started_utc: str,
    ended_utc: str,
    exit_code: int,
) -> dict[str, JSONValue]:
    """Assemble a record and refuse to return one that is internally inconsistent."""
    if status not in (PASS, FAIL, NOT_RUN):
        raise EvidenceError(f"{status!r} is not a backend-contract status")
    document: dict[str, JSONValue] = {
        "schema": EVIDENCE_SCHEMA,
        "status": status,
        **binding.with_attempt(),
        "artifact_hashes": dict(artifact_hashes),
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "exit_code": exit_code,
        "checks": [dict(check) for check in checks],
    }
    problems = structural_problems(document)
    if problems:
        raise EvidenceError("; ".join(problems))
    return document


def structural_problems(document: JSONDocument) -> list[str]:
    """Every reason this record is not a well-formed backend-contract evidence file."""
    problems: list[str] = []
    missing = sorted(set(REQUIRED_BINDINGS) - set(document))
    if missing:
        return [f"the record omits required binding(s): {', '.join(missing)}"]
    if document.get("schema") != EVIDENCE_SCHEMA:
        problems.append(f"unknown schema {document.get('schema')!r}")
    status = str(document.get("status"))
    if status not in (PASS, FAIL, NOT_RUN):
        problems.append(f"{status!r} is not a backend-contract status")

    for field in (
        "run_id",
        "environment_lock_sha256",
        "model_manifest_sha256",
        "backend_code_sha256",
        "scoring_code_sha256",
        "resolved_config_sha256",
    ):
        if not _SHA256_RE.match(str(document.get(field, ""))):
            problems.append(f"{field} is not a sha256 digest")
    if not _COMMIT_RE.match(str(document.get("git_commit", ""))):
        problems.append("git_commit is not a 40-hex commit sha")
    elif str(document.get("git_commit")) == "0" * 40:
        problems.append(
            "git_commit is the all-zero sha, which is a placeholder rather than a resolved"
            " commit; evidence is not issued when provenance cannot be resolved"
            " [AUTH: 01 §15, §16]"
        )
    if not _COMMIT_RE.match(str(document.get("model_revision", ""))):
        problems.append(
            "model_revision is not an immutable 40-hex revision; a backend-contract result"
            " on a floating revision cannot be re-derived [AUTH: 01 §8G]"
        )
    for field in ("started_utc", "ended_utc"):
        if not _TIMESTAMP_RE.match(str(document.get(field, ""))):
            problems.append(f"{field} is not an ISO-8601 timestamp")
    if str(document.get("started_utc", "")) > str(document.get("ended_utc", "")):
        problems.append("the record ends before it starts")

    exit_code = document.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        problems.append("exit_code must be an integer")
    elif (exit_code == 0) != (status == PASS):
        # The biconditional, not just one direction. Checking only "PASS implies exit 0" let a
        # NOT_RUN or FAIL record carry exit 0, which reads to any downstream consumer as a
        # command that succeeded [AUTH: 01 §15, §16; 02 §C6].
        problems.append(
            f"status {status} and exit_code {exit_code} disagree; a zero exit means"
            f" {PASS} and nothing else"
        )

    checks = document.get("checks")
    if not isinstance(checks, list) or not checks:
        problems.append("a backend-contract record must list the checks it ran")
    else:
        rows = [check for check in checks if isinstance(check, Mapping)]
        if len(rows) != len(checks):
            problems.append("every check row must be an object")
        else:
            problems.extend(_check_row_problems(rows))
            try:
                expected = aggregate_status(rows)
            except EvidenceError as exc:
                problems.append(str(exc))
            else:
                if status != expected:
                    problems.append(
                        f"the record claims {status} but its checks aggregate to {expected};"
                        " any FAIL makes the whole contract FAIL, and any remaining NOT_RUN"
                        " makes it NOT_RUN [AUTH: 02 §C6]"
                    )

    artifacts = document.get("artifact_hashes")
    if not isinstance(artifacts, Mapping):
        problems.append("artifact_hashes must be an object")
    else:
        for name, digest in artifacts.items():
            if not _SHA256_RE.match(str(digest)):
                problems.append(f"artifact_hashes[{name}] is not a sha256 digest")
    return problems


def _check_row_problems(rows: Sequence[Mapping[str, JSONValue]]) -> list[str]:
    """Each row must name its check, its status and — when NOT_RUN — its specific cause."""
    problems: list[str] = []
    for index, row in enumerate(rows):
        label = str(row.get("name", f"#{index}"))
        if not str(row.get("name", "")).strip():
            problems.append(f"check {label} has no name")
        status = str(row.get("status"))
        if status not in (PASS, FAIL, NOT_RUN):
            problems.append(f"check {label} has status {status!r}")
            continue
        if status != NOT_RUN:
            continue
        reason = str(row.get("reason", ""))
        if reason not in NOT_RUN_REASONS:
            problems.append(
                f"check {label} is NOT_RUN without a declared reason; the specific cause is"
                f" the only information a NOT_RUN row carries (expected one of"
                f" {list(NOT_RUN_REASONS)})"
            )
    return problems


def verify_contract_evidence(
    document: JSONDocument, *, binding: RunBinding, root: Path
) -> list[str]:
    """Every reason this record does not describe the current run state.

    Each identity is compared against a value recomputed now, so a hand-authored PASS carrying
    plausible-looking digests fails on the first one that is not this run's.
    """
    problems = structural_problems(document)
    if problems:
        return problems
    for field, expected in binding.as_dict().items():
        actual = str(document.get(field, ""))
        if actual != expected:
            problems.append(
                f"{field} records {actual[:12]} but the current run state is"
                f" {str(expected)[:12]}; this record does not describe this run"
                " [AUTH: 01 §15, §16]"
            )
    artifacts = document.get("artifact_hashes")
    if isinstance(artifacts, Mapping):
        for relative, digest in artifacts.items():
            path = root / str(relative)
            if not path.is_file():
                problems.append(f"the record names a missing artifact: {relative}")
                continue
            recomputed = sha256_file(path)
            if recomputed != str(digest):
                problems.append(
                    f"{relative} hashes to {recomputed[:12]} but the record says"
                    f" {str(digest)[:12]}; the artifact changed after the record was written"
                    " [AUTH: 01 §16, §36]"
                )
    return problems


def evidence_identity(document: JSONDocument) -> str:
    return sha256_canonical(dict(document))


def write_contract_evidence(root: Path, relative: str, document: JSONDocument) -> str:
    problems = structural_problems(document)
    if problems:
        raise EvidenceError("; ".join(problems))
    return write_canonical_json(root / relative, dict(document))
