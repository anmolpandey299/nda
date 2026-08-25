"""Experiment identity and run identity.

01 §15 requires a *deterministic* identity derived from eight provenance inputs. 01 §36
requires that a failed run is never overwritten and that a rerun receives a new RUN_ID.
Taken literally at one level those two requirements contradict each other: identical
provenance hashes to one value, yet a rerun of identical provenance must be distinguishable.

They are separated here into two levels:

    EXPERIMENT_ID = SHA256(canonical JSON of the eight 01 §15 provenance inputs)
    RUN_ID        = SHA256(canonical JSON of EXPERIMENT_ID + explicit attempt index)

EXPERIMENT_ID answers "which scientific configuration is this?" and is stable: the same
provenance always yields the same value, on any machine, in any order of dictionary keys.
RUN_ID answers "which execution of it is this?" and is distinct per attempt, so attempt 1
failing and attempt 2 succeeding produce two immutable directories and two manifests.

Attempt indices are allocated by probing, not by a mutable counter file: attempt n's
directory name is a pure function of (EXPERIMENT_ID, n), so the first free n is discoverable
without shared state, and claiming it is an atomic `mkdir` that fails if someone else won
the race [AUTH: 01 §15, §36].

Wall-clock time is recorded in the run manifest but is never an identity input: an identity
that moved with the clock could not be recomputed from provenance [AUTH: 01 §15].
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from src.provenance.hashing import JSONValue, sha256_canonical

#: The eight RUN_ID inputs, verbatim [AUTH: 01 §15]. Bound by test to the repository
#: invariant checker's copy so the two vocabularies cannot drift apart.
PROVENANCE_INPUTS: Final[tuple[str, ...]] = (
    "git_commit_sha",
    "spec_sha256",
    "execution_lock_sha256",
    "config_sha256",
    "model_revision",
    "data_manifest_sha256",
    "environment_lock_sha256",
    "training_seed",
)

#: Domain separators, so a run identity can never collide with an experiment identity even
#: if their payloads were otherwise identical.
EXPERIMENT_DOMAIN: Final = "privacy-model-merging/experiment-identity/v1"
RUN_DOMAIN: Final = "privacy-model-merging/run-identity/v1"

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_TBD: Final = "TBD_REQUIRES_HARDWARE"

#: Fields that must be a full SHA256 before an identity means anything.
_SHA256_FIELDS: Final[tuple[str, ...]] = (
    "spec_sha256",
    "execution_lock_sha256",
    "config_sha256",
    "data_manifest_sha256",
    "environment_lock_sha256",
)


class ProvenanceError(ValueError):
    """Provenance is incomplete or malformed, so no identity may be derived from it."""


@dataclass(frozen=True)
class ScientificProvenance:
    """The eight 01 §15 inputs. Every field is evidentiary; none has a default."""

    git_commit_sha: str
    spec_sha256: str
    execution_lock_sha256: str
    config_sha256: str
    model_revision: str
    data_manifest_sha256: str
    environment_lock_sha256: str
    training_seed: int

    def as_dict(self) -> dict[str, JSONValue]:
        return dict(asdict(self))

    def problems(self) -> list[str]:
        """Every reason this provenance may not become an identity."""
        found: list[str] = []
        for field_name in _SHA256_FIELDS:
            value = getattr(self, field_name)
            if value == _TBD:
                found.append(f"{field_name} is still {_TBD}")
            elif not isinstance(value, str) or not _SHA256_RE.match(value):
                found.append(f"{field_name} is not a SHA256 digest")
        if not _GIT_SHA_RE.match(self.git_commit_sha):
            found.append("git_commit_sha is not a full 40-hex commit SHA")
        revision = self.model_revision
        if not isinstance(revision, str) or not revision.strip():
            found.append("model_revision is empty")
        elif revision.strip().lower() in {"main", "master", "head"}:
            # A branch name is not a revision: it moves [AUTH: 01 §8G].
            found.append(f"model_revision {revision!r} is a floating branch, not a pinned revision")
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            found.append("training_seed is not an integer")
        return found

    def require_valid(self) -> None:
        found = self.problems()
        if found:
            raise ProvenanceError("; ".join(found))


def experiment_id(provenance: ScientificProvenance) -> str:
    """Deterministic scientific identity. Same provenance anywhere -> same value."""
    provenance.require_valid()
    payload: dict[str, JSONValue] = {
        "domain": EXPERIMENT_DOMAIN,
        "inputs": provenance.as_dict(),
    }
    return sha256_canonical(payload)


def run_id(experiment: str, attempt: int) -> str:
    """Identity of one execution attempt of `experiment`."""
    if not _SHA256_RE.match(experiment):
        raise ProvenanceError("experiment id is not a SHA256 digest")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ProvenanceError("attempt must be an integer >= 1")
    payload: dict[str, JSONValue] = {
        "domain": RUN_DOMAIN,
        "experiment_id": experiment,
        "attempt": attempt,
    }
    return sha256_canonical(payload)


def runs_root(root: Path) -> Path:
    """`artifacts/runs/` — one immutable directory per attempt [AUTH: 01 §36]."""
    return root / "artifacts" / "runs"


def manifests_root(root: Path) -> Path:
    """`manifests/runs/` — one manifest per attempt [AUTH: 01 §15, §16]."""
    return root / "manifests" / "runs"


def attempt_directory(root: Path, experiment: str, attempt: int) -> Path:
    return runs_root(root) / run_id(experiment, attempt)


def run_manifest_path(root: Path, identifier: str) -> Path:
    return manifests_root(root) / f"{identifier}.json"


def attempt_is_claimed(root: Path, experiment: str, attempt: int) -> bool:
    """An attempt index is spent if EITHER of its two records survives.

    Checking only the run directory was a way to reuse a RUN_ID: delete the directory of a
    failed attempt and its index is handed out again, and the next `begin_run` overwrites the
    failed attempt's manifest with a fresh RUNNING one. Either record alone permanently
    reserves the index, so a surviving manifest is enough to keep the failure visible
    [AUTH: 01 §15, §36].
    """
    identifier = run_id(experiment, attempt)
    return (
        attempt_directory(root, experiment, attempt).exists()
        or run_manifest_path(root, identifier).exists()
    )


def claimed_run_ids(root: Path) -> set[str]:
    """Every RUN_ID with a surviving record, from either namespace."""
    found: set[str] = set()
    manifests = manifests_root(root)
    if manifests.is_dir():
        found.update(path.stem for path in manifests.glob("*.json"))
    runs = runs_root(root)
    if runs.is_dir():
        found.update(entry.name for entry in runs.iterdir() if entry.is_dir())
    return found


def next_attempt(root: Path, experiment: str, *, limit: int = 10_000) -> int:
    """The lowest attempt index that is not already claimed by either record."""
    for attempt in range(1, limit + 1):
        if not attempt_is_claimed(root, experiment, attempt):
            return attempt
    raise ProvenanceError(f"experiment {experiment[:12]} has more than {limit} attempts")
