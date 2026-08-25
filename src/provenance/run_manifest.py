"""Run manifest — the document without which a result is NON_EVIDENTIARY [AUTH: 01 §16].

The manifest is materialised *before* the work it describes begins, carrying the identity,
the provenance it was derived from and a non-terminal `RUNNING` status. Finalisation adds
the terminal status, the end time, the exit code and the artifact hashes. A run that dies
mid-flight therefore leaves a manifest that says so, rather than no manifest at all.

`RUNNING` is deliberately not one of the 01 §36 states: §36 enumerates the states a run
*ends* in, and a manifest that still says RUNNING has not ended.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping
from typing import Final

from src.provenance.config import resolved_config_sha256
from src.provenance.hashing import JSONDocument, JSONValue
from src.provenance.identity import ScientificProvenance

#: The 01 §16 minimum fields, verbatim. Bound by test to the repository invariant checker's
#: copy so a field cannot be dropped from one without the other noticing.
RUN_MANIFEST_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "git_commit",
    "git_dirty",
    "spec_hash",
    "execution_lock_hash",
    "model_revision",
    "tokenizer_hash",
    "data_manifest_hash",
    "environment_lock_sha256",
    "gpu_model",
    "cuda",
    "pytorch",
    "precision",
    "seeds",
    "config",
    "wall_clock_start",
    "wall_clock_end",
    "exit_code",
    "artifact_paths",
    "artifact_hashes",
    "metrics_paths",
    "stdout_log_path",
    "stderr_log_path",
)

#: 01 §16 lists "GPU UUID where available" but scripts/check_repo_invariants.py's minimum
#: tuple omits it. This contract is a superset of that tuple and never a subset: the S00
#: gate stays satisfied, and a run manifest that names no GPU identity is refused here
#: [AUTH: 01 §16, §30].
RUN_MANIFEST_ADDITIONAL_REQUIRED: Final[tuple[str, ...]] = ("gpu_uuid",)

#: Everything a run manifest must carry.
ALL_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    *RUN_MANIFEST_REQUIRED_FIELDS,
    *RUN_MANIFEST_ADDITIONAL_REQUIRED,
)

#: Identity and lifecycle fields this repository adds beyond the 01 §16 minimum.
RUN_MANIFEST_EXTRA_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "experiment_id",
    "attempt",
    "status",
    "evidentiary",
    "config_sha256",
    "git_dirty_source",
)

#: Exactly the 01 §36 terminal states.
TERMINAL_STATES: Final[frozenset[str]] = frozenset(
    {
        "SUCCESS",
        "FAILED_IMPLEMENTATION",
        "FAILED_ENVIRONMENT",
        "FAILED_RESOURCE",
        "FAILED_SCIENTIFIC_GATE",
        "CANCELLED",
    }
)

#: Non-terminal marker written before the work starts.
RUNNING: Final = "RUNNING"

#: Honest value for a field that cannot apply, e.g. a GPU field on a CPU-only fixture run.
#: Distinct from TBD_REQUIRES_HARDWARE, which means "not measured yet" and must never
#: survive into a SUCCESS record [AUTH: 03 §8].
NOT_APPLICABLE: Final = "NOT_APPLICABLE"
_TBD: Final = "TBD_REQUIRES_HARDWARE"

#: 01 §30 requires each seed family to be recorded separately, never one global seed. This
#: is the closed vocabulary: a key outside it — `one_global_seed` being the case the review
#: raised — is refused rather than silently accepted as "some seed was recorded".
SEED_FIELDS: Final[tuple[str, ...]] = (
    "training_seed",
    "python_rng",
    "numpy_rng",
    "torch_cpu_rng",
    "torch_cuda_rng",
    "data_order",
    "canary_inclusion",
    "lora_init",
    "dp_noise",
)

#: The master seed, which is also a RUN_ID input, so an evidentiary run must record it and
#: it must agree with the provenance it was hashed from [AUTH: 01 §15, §30].
MASTER_SEED_FIELD: Final = "training_seed"

RUN_MANIFEST_SCHEMA: Final = "s01.run-manifest.v1"

_HARDWARE_FIELDS: Final[tuple[str, ...]] = ("gpu_model", "gpu_uuid", "cuda", "pytorch")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")


class RunManifestError(ValueError):
    """A run manifest is incomplete or would attribute a result to unusable provenance."""


def utc_now() -> str:
    """Wall-clock stamp. Recorded in the manifest, never an identity input [AUTH: 01 §15]."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def build_run_manifest(
    *,
    run_id: str,
    experiment_id: str,
    attempt: int,
    provenance: ScientificProvenance,
    resolved_config: JSONDocument,
    seeds: Mapping[str, int],
    precision: str,
    tokenizer_hash: str,
    stdout_log_path: str,
    stderr_log_path: str,
    wall_clock_start: str,
    gpu_model: str = NOT_APPLICABLE,
    gpu_uuid: str = NOT_APPLICABLE,
    cuda: str = NOT_APPLICABLE,
    pytorch: str = NOT_APPLICABLE,
    evidentiary: bool = True,
    git_dirty: bool = False,
    git_dirty_source: str = "derived",
) -> dict[str, JSONValue]:
    """The pre-execution manifest: every field present, terminal fields still open."""
    provenance.require_valid()
    recorded_seeds = dict(seeds)
    declared = recorded_seeds.get(MASTER_SEED_FIELD)
    if declared is not None and declared != provenance.training_seed:
        raise RunManifestError(
            f"recorded {MASTER_SEED_FIELD} {declared!r} disagrees with the provenance"
            f" {provenance.training_seed!r} the identity was derived from [AUTH: 01 §15, §30]"
        )
    recorded_seeds[MASTER_SEED_FIELD] = provenance.training_seed
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "attempt": attempt,
        "status": RUNNING,
        "evidentiary": evidentiary,
        "git_commit": provenance.git_commit_sha,
        "git_dirty": git_dirty,
        "git_dirty_source": git_dirty_source,
        "spec_hash": provenance.spec_sha256,
        "execution_lock_hash": provenance.execution_lock_sha256,
        "config_sha256": provenance.config_sha256,
        "model_revision": provenance.model_revision,
        "tokenizer_hash": tokenizer_hash,
        "data_manifest_hash": provenance.data_manifest_sha256,
        "environment_lock_sha256": provenance.environment_lock_sha256,
        "gpu_model": gpu_model,
        "gpu_uuid": gpu_uuid,
        "cuda": cuda,
        "pytorch": pytorch,
        "precision": precision,
        "seeds": recorded_seeds,
        "config": dict(resolved_config),
        "wall_clock_start": wall_clock_start,
        "wall_clock_end": None,
        "exit_code": None,
        "artifact_paths": [],
        "artifact_hashes": {},
        "metrics_paths": [],
        "stdout_log_path": stdout_log_path,
        "stderr_log_path": stderr_log_path,
    }


def validate_run_manifest(document: JSONDocument) -> list[str]:
    """Every reason this manifest cannot attribute a scientific result.

    Identity relationships are recomputed, not merely looked at: the stored config must hash
    to the stored config identity, so a manifest cannot claim hash(A) while carrying config
    B [AUTH: 01 §15, §16, §17].
    """
    problems: list[str] = []
    for name in ALL_REQUIRED_FIELDS:
        if name not in document:
            problems.append(f"missing required field {name!r}")

    status = document.get("status")
    if status != RUNNING and status not in TERMINAL_STATES:
        problems.append(f"status {status!r} is neither RUNNING nor an 01 §36 terminal state")

    run_id = document.get("run_id")
    if not isinstance(run_id, str) or not _SHA256_RE.match(run_id):
        problems.append("run_id is not a SHA256 digest")

    evidentiary = document.get("evidentiary") is True
    problems += _validate_seeds(document, evidentiary=evidentiary)
    problems += _validate_config_identity(document)
    problems += _validate_gpu_identity(document)

    if status in TERMINAL_STATES:
        problems += _validate_terminal(document, status)
    elif document.get("wall_clock_end") is not None:
        problems.append("a RUNNING manifest must not declare wall_clock_end")
    return problems


def _validate_seeds(document: JSONDocument, *, evidentiary: bool) -> list[str]:
    """01 §30: distinct seed families, from a documented vocabulary, never one global seed."""
    problems: list[str] = []
    seeds = document.get("seeds")
    if not isinstance(seeds, Mapping) or not seeds:
        return ["seeds must be a non-empty object; one global seed is not enough"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in seeds.values()):
        problems.append("every recorded seed must be an integer [AUTH: 01 §30]")
    undocumented = sorted(set(seeds) - set(SEED_FIELDS))
    if undocumented:
        problems.append(
            f"undocumented seed field(s) {', '.join(undocumented)};"
            f" 01 §30 requires the named families {', '.join(SEED_FIELDS)}"
        )
    if evidentiary:
        if MASTER_SEED_FIELD not in seeds:
            problems.append(
                f"an evidentiary run must record {MASTER_SEED_FIELD!r} [AUTH: 01 §15, §30]"
            )
        # The master seed is a RUN_ID input, not an RNG-family record. On its own it is not
        # a reproducibility record at all, and the builder injects it, so its presence alone
        # must not satisfy 01 §30. Which families apply to which training path is S06's
        # decision; Block A only refuses "master seed == the whole record".
        if not set(seeds) - {MASTER_SEED_FIELD}:
            problems.append(
                "an evidentiary run must record at least one explicit RNG-family seed"
                f" besides {MASTER_SEED_FIELD!r} [AUTH: 01 §30]"
            )
    return problems


def _validate_config_identity(document: JSONDocument) -> list[str]:
    """The stored config must hash to the stored identity, which is a RUN_ID input."""
    config = document.get("config")
    if not isinstance(config, Mapping):
        return ["config must be the resolved config object [AUTH: 01 §17]"]
    declared = document.get("config_sha256")
    if not isinstance(declared, str) or not _SHA256_RE.match(declared):
        return ["config_sha256 is not a SHA256 digest"]
    try:
        recomputed = resolved_config_sha256(config)
    except ValueError as exc:
        return [f"the stored config is not canonicalisable: {exc}"]
    if recomputed != declared:
        return [
            f"config_sha256 {declared[:12]} does not hash the stored config"
            f" ({recomputed[:12]}) [AUTH: 01 §15, §17]"
        ]
    return []


def _validate_gpu_identity(document: JSONDocument) -> list[str]:
    """01 §16 wants GPU UUID where available; neither field may be fabricated [AUTH: 03 §8]."""
    model = document.get("gpu_model")
    uuid = document.get("gpu_uuid")
    if not isinstance(model, str) or not model.strip():
        return ["gpu_model is missing"]
    if not isinstance(uuid, str) or not uuid.strip():
        return ["gpu_uuid is missing"]
    if model == NOT_APPLICABLE:
        if uuid != NOT_APPLICABLE:
            return [f"a CPU run names gpu_uuid {uuid!r}; it has no GPU to identify"]
        return []
    if uuid in (NOT_APPLICABLE, _TBD):
        return [
            f"gpu_model {model!r} names hardware but gpu_uuid is {uuid!r};"
            " 01 §16 requires the GPU UUID where available"
        ]
    return []


def _validate_terminal(document: JSONDocument, status: object) -> list[str]:
    problems: list[str] = []
    if not isinstance(document.get("wall_clock_end"), str):
        problems.append("a terminal manifest must record wall_clock_end")
    exit_code = document.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        problems.append("a terminal manifest must record an integer exit_code")

    paths = document.get("artifact_paths")
    hashes = document.get("artifact_hashes")
    if not isinstance(paths, list):
        problems.append("artifact_paths must be a list")
    elif not isinstance(hashes, Mapping):
        problems.append("artifact_hashes must be an object")
    elif sorted(str(p) for p in paths) != sorted(hashes):
        problems.append("artifact_paths and artifact_hashes disagree")

    if status == "SUCCESS":
        if document.get("git_dirty") is not False:
            problems.append("a SUCCESS run may not be attributed to a dirty tree [AUTH: 01 §35(3)]")
        if exit_code != 0:
            problems.append("a SUCCESS run must record exit_code 0")
        for name in _HARDWARE_FIELDS:
            value = document.get(name)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{name} is missing")
            elif value == _TBD:
                problems.append(f"{name} is still {_TBD}; an unmeasured value is not a result")
        problems += _validate_metrics_are_hash_bound(document)
    return problems


def _validate_metrics_are_hash_bound(document: JSONDocument) -> list[str]:
    """A SUCCESS run may not carry a metrics path that nothing hashes.

    A metrics table is the result. Listing its path without binding its bytes leaves the
    number free to change after the run ended, with the manifest still reading SUCCESS
    [AUTH: 01 §16, §19, §36].
    """
    metrics = document.get("metrics_paths")
    hashes = document.get("artifact_hashes")
    if not isinstance(metrics, list):
        return ["metrics_paths must be a list"]
    if not isinstance(hashes, Mapping):
        return ["artifact_hashes must be an object"]
    unbound = sorted(str(path) for path in metrics if str(path) not in hashes)
    if unbound:
        return [
            f"SUCCESS with unhashed metrics path(s): {', '.join(unbound)};"
            " every metrics path must carry an artifact hash [AUTH: 01 §16, §19]"
        ]
    return []


def require_valid_run_manifest(document: JSONDocument) -> None:
    problems = validate_run_manifest(document)
    if problems:
        raise RunManifestError("; ".join(problems))
