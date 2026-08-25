"""Immutable run attempts [AUTH: 01 §15, §16, §35(3), §36].

One attempt owns one directory, `artifacts/runs/<RUN_ID>/`, and one manifest,
`manifests/runs/<RUN_ID>.json`. The manifest is the single source of truth: the attempt is
finalised exactly when its manifest carries an 01 §36 terminal state.

The three guarantees, and how each is actually enforced:

* an attempt is never silently overwritten — the directory is claimed with a non-exist_ok
  `mkdir`, so a second `begin` on the same RUN_ID raises rather than truncating;
* a failed attempt stays intact — nothing deletes or rewrites a finalised attempt, and a
  rerun probes for the next free attempt index, which yields a different RUN_ID;
* artifacts cannot be silently replaced — they are hash-bound when recorded, recording is
  refused after finalisation, and `verify()` recomputes every digest.

Immutability is enforced by refusal and by hashing, not by filesystem permissions: a mode
bit is advisory, root ignores it, and it would not survive an archive round trip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from src.provenance.config import resolved_config_sha256
from src.provenance.hashing import (
    ArtifactRecord,
    ArtifactVerificationError,
    JSONDocument,
    JSONValue,
    read_canonical_json,
    record_artifact,
    verify_artifact,
    write_canonical_json,
)
from src.provenance.identity import (
    ProvenanceError,
    ScientificProvenance,
    attempt_directory,
    claimed_run_ids,
    experiment_id,
    next_attempt,
    run_id,
    run_manifest_path,
)
from src.provenance.repository import (
    RepositoryError,
    head_commit,
    is_repository,
    production_tree_dirty,
    verify_environment_identity,
    verify_runtime_attribution,
)
from src.provenance.run_manifest import (
    MASTER_SEED_FIELD,
    NOT_APPLICABLE,
    RUNNING,
    TERMINAL_STATES,
    build_run_manifest,
    require_valid_run_manifest,
    utc_now,
)


class RunStoreError(RuntimeError):
    """The run store refused an operation that would corrupt or lose evidence."""


class RunDirectoryExists(RunStoreError):
    """This attempt already exists; overwriting it is prohibited [AUTH: 01 §36]."""


class RunFinalizedError(RunStoreError):
    """This attempt has already ended; its record is immutable [AUTH: 01 §36]."""


class DirtyTreeError(RunStoreError):
    """An evidentiary run was requested from a dirty tree [AUTH: 01 §35(3), §36]."""


class UnverifiedProvenanceError(RunStoreError):
    """A caller asserted a repository, config or environment fact the repository contradicts.

    An evidentiary manifest is a claim about what executed. Believing the caller lets a run
    that never touched this commit, this config or this environment finalise SUCCESS and
    verify clean [AUTH: 01 §15, §16, §35(3)].
    """


@dataclass(frozen=True)
class RunAttempt:
    """A claimed, in-progress or finalised execution attempt."""

    root: Path
    experiment_id: str
    attempt: int
    run_id: str

    @property
    def directory(self) -> Path:
        return attempt_directory(self.root, self.experiment_id, self.attempt)

    @property
    def manifest_path(self) -> Path:
        return run_manifest_path(self.root, self.run_id)

    @property
    def log_directory(self) -> Path:
        return self.root / "logs" / self.run_id

    @property
    def stdout_path(self) -> str:
        return f"logs/{self.run_id}/stdout.log"

    @property
    def stderr_path(self) -> str:
        return f"logs/{self.run_id}/stderr.log"

    def read_manifest(self) -> dict[str, JSONValue]:
        document = read_canonical_json(self.manifest_path)
        if not isinstance(document, Mapping):
            raise RunStoreError(f"{self.manifest_path}: run manifest is not a JSON object")
        return dict(document)

    def status(self) -> str:
        return str(self.read_manifest().get("status"))

    def is_finalized(self) -> bool:
        return self.status() in TERMINAL_STATES

    def artifact_records(self) -> list[ArtifactRecord]:
        manifest = self.read_manifest()
        hashes = manifest.get("artifact_hashes")
        sizes = manifest.get("artifact_sizes")
        if not isinstance(hashes, Mapping):
            return []
        size_map = sizes if isinstance(sizes, Mapping) else {}
        return [
            ArtifactRecord(
                path=str(path),
                sha256=str(digest),
                size_bytes=int(str(size_map.get(path, 0))),
            )
            for path, digest in sorted(hashes.items())
        ]

    def _bind_artifact(self, manifest: dict[str, JSONValue], relative_path: str) -> ArtifactRecord:
        """Hash `relative_path` and record it in `manifest`, without writing."""
        record = record_artifact(self.root, relative_path)
        raw_hashes = manifest.get("artifact_hashes")
        raw_sizes = manifest.get("artifact_sizes")
        hashes: dict[str, JSONValue] = dict(raw_hashes) if isinstance(raw_hashes, Mapping) else {}
        sizes: dict[str, JSONValue] = dict(raw_sizes) if isinstance(raw_sizes, Mapping) else {}
        if relative_path in hashes and hashes[relative_path] != record.sha256:
            raise RunStoreError(
                f"artifact {relative_path} was already recorded with a different hash"
            )
        hashes[relative_path] = record.sha256
        sizes[relative_path] = record.size_bytes
        manifest["artifact_hashes"] = hashes
        manifest["artifact_sizes"] = sizes
        manifest["artifact_paths"] = sorted(hashes)
        return record

    def add_artifact(self, relative_path: str) -> ArtifactRecord:
        """Hash-bind one produced artifact. Refused once the attempt has ended."""
        manifest = self.read_manifest()
        if manifest.get("status") in TERMINAL_STATES:
            raise RunFinalizedError(
                f"attempt {self.run_id[:12]} is finalised; its artifact set cannot change"
            )
        record = self._bind_artifact(manifest, relative_path)
        write_canonical_json(self.manifest_path, manifest)
        return record

    def finalize(
        self,
        *,
        status: str,
        exit_code: int,
        metrics_paths: Sequence[str] = (),
        gpu_model: str | None = None,
        gpu_uuid: str | None = None,
        cuda: str | None = None,
        pytorch: str | None = None,
        wall_clock_end: str | None = None,
    ) -> dict[str, JSONValue]:
        """Write the terminal record exactly once.

        Before SUCCESS, every metrics path is hash-bound through the artifact mechanism and
        every already-registered artifact is re-verified. A result table that changed between
        being recorded and the run ending must not be able to end as SUCCESS
        [AUTH: 01 §16, §19, §36].
        """
        if status not in TERMINAL_STATES:
            raise RunStoreError(
                f"{status!r} is not an 01 §36 terminal state: {', '.join(sorted(TERMINAL_STATES))}"
            )
        manifest = self.read_manifest()
        if manifest.get("status") in TERMINAL_STATES:
            raise RunFinalizedError(
                f"attempt {self.run_id[:12]} already ended as {manifest['status']}"
            )

        if status == "SUCCESS":
            for relative_path in metrics_paths:
                self._bind_artifact(manifest, relative_path)
            stale = self._changed_artifacts(manifest)
            if stale:
                raise ArtifactVerificationError(
                    "refusing SUCCESS; registered output changed after it was recorded: "
                    + "; ".join(stale)
                )

        manifest["status"] = status
        manifest["exit_code"] = exit_code
        manifest["wall_clock_end"] = wall_clock_end or utc_now()
        manifest["metrics_paths"] = list(metrics_paths)
        for name, value in (
            ("gpu_model", gpu_model),
            ("gpu_uuid", gpu_uuid),
            ("cuda", cuda),
            ("pytorch", pytorch),
        ):
            if value is not None:
                manifest[name] = value
        require_valid_run_manifest(manifest)
        write_canonical_json(self.manifest_path, manifest)
        return manifest

    def _changed_artifacts(self, manifest: Mapping[str, JSONValue]) -> list[str]:
        raw = manifest.get("artifact_hashes")
        if not isinstance(raw, Mapping):
            return []
        problems: list[str] = []
        for path, digest in sorted(raw.items()):
            record = ArtifactRecord(path=str(path), sha256=str(digest), size_bytes=0)
            try:
                verify_artifact(self.root, record)
            except ArtifactVerificationError as exc:
                problems.append(str(exc))
        return problems

    def verify(self) -> list[str]:
        """Re-derive the identity relationships, then recompute every artifact digest.

        Field presence is not attribution. This recomputes the experiment identity from the
        provenance the manifest records, the RUN_ID from that identity and the attempt index,
        and — for an evidentiary run — the environment identity against the captured
        manifest, so a record whose parts do not actually belong together is rejected
        [AUTH: 01 §15, §16, §32].
        """
        problems = list(validate_stored_manifest(self.manifest_path))
        if not self.manifest_path.is_file():
            return problems
        manifest = self.read_manifest()
        problems += self._identity_problems(manifest)
        for record in self.artifact_records():
            try:
                verify_artifact(self.root, record)
            except ArtifactVerificationError as exc:
                problems.append(str(exc))
        return problems

    def _identity_problems(self, manifest: Mapping[str, JSONValue]) -> list[str]:
        seeds = manifest.get("seeds")
        master = seeds.get(MASTER_SEED_FIELD) if isinstance(seeds, Mapping) else None
        if isinstance(master, bool) or not isinstance(master, int):
            return [f"seeds.{MASTER_SEED_FIELD} is missing, so no identity can be re-derived"]
        try:
            provenance = ScientificProvenance(
                git_commit_sha=str(manifest["git_commit"]),
                spec_sha256=str(manifest["spec_hash"]),
                execution_lock_sha256=str(manifest["execution_lock_hash"]),
                config_sha256=str(manifest["config_sha256"]),
                model_revision=str(manifest["model_revision"]),
                data_manifest_sha256=str(manifest["data_manifest_hash"]),
                environment_lock_sha256=str(manifest["environment_lock_sha256"]),
                training_seed=master,
            )
            expected_experiment = experiment_id(provenance)
        except (KeyError, ProvenanceError) as exc:
            return [f"the recorded provenance cannot produce an identity: {exc}"]

        problems: list[str] = []
        if manifest.get("experiment_id") != expected_experiment:
            problems.append(
                f"experiment_id {str(manifest.get('experiment_id'))[:12]} is not what this"
                f" provenance derives ({expected_experiment[:12]}) [AUTH: 01 §15]"
            )
        attempt = manifest.get("attempt")
        if isinstance(attempt, int) and not isinstance(attempt, bool):
            expected_run = run_id(expected_experiment, attempt)
            if manifest.get("run_id") != expected_run:
                problems.append("run_id is not derived from this experiment identity and attempt")
        else:
            problems.append("attempt is missing or not an integer")
        if manifest.get("evidentiary") is True:
            identity = str(manifest.get("environment_lock_sha256"))
            problems += verify_environment_identity(self.root, identity)
            if str(manifest.get("gpu_model")) != NOT_APPLICABLE:
                problems += verify_runtime_attribution(self.root, identity, manifest)
        return problems


def validate_stored_manifest(path: Path) -> list[str]:
    if not path.is_file():
        return [f"run manifest absent: {path}"]
    document = read_canonical_json(path)
    if not isinstance(document, Mapping):
        return [f"{path}: run manifest is not a JSON object"]
    from src.provenance.run_manifest import validate_run_manifest

    return validate_run_manifest(dict(document))


def _derive_repository_facts(
    root: Path,
    provenance: ScientificProvenance,
    resolved_config: JSONDocument,
    *,
    evidentiary: bool,
    git_dirty: bool | None,
) -> tuple[bool, str]:
    """Check the caller's provenance against the repository. Returns (git_dirty, source).

    For an evidentiary run nothing here is taken on trust: the commit comes from the
    repository, the dirty state comes from the same production-path predicate the S00 gate
    uses, the config identity is recomputed from the config that will be stored, and the
    environment identity must name a captured manifest that recomputes to it
    [AUTH: 01 §12, §15, §17, §32, §35(3), §36].
    """
    recomputed_config = resolved_config_sha256(resolved_config)
    if recomputed_config != provenance.config_sha256:
        raise UnverifiedProvenanceError(
            f"config_sha256 {provenance.config_sha256[:12]} does not hash the config this run"
            f" will store ({recomputed_config[:12]}) [AUTH: 01 §15, §17]"
        )

    if not evidentiary:
        if is_repository(root):
            return bool(production_tree_dirty(root)), "derived"
        if git_dirty is None:
            raise UnverifiedProvenanceError(
                f"{root} is not a repository, so git_dirty cannot be derived; a"
                " non-evidentiary run must declare it explicitly"
            )
        return git_dirty, "declared"

    try:
        head = head_commit(root)
    except RepositoryError as exc:
        raise UnverifiedProvenanceError(
            f"an evidentiary run cannot derive its commit: {exc} [AUTH: 01 §15]"
        ) from exc
    if provenance.git_commit_sha != head:
        raise UnverifiedProvenanceError(
            f"provenance claims commit {provenance.git_commit_sha[:12]} but HEAD is"
            f" {head[:12]} [AUTH: 01 §15]"
        )

    problems = verify_environment_identity(root, provenance.environment_lock_sha256)
    if problems:
        raise UnverifiedProvenanceError(
            "environment identity is not an accepted S00 environment: " + "; ".join(problems)
        )

    dirty = production_tree_dirty(root)
    if git_dirty is False and dirty:
        raise UnverifiedProvenanceError(
            "the caller declared a clean tree but the production tree is dirty: "
            + ", ".join(dirty[:4])
        )
    if dirty:
        raise DirtyTreeError(
            "refusing an evidentiary run from a dirty production tree [AUTH: 01 §35(3), §36]: "
            + ", ".join(dirty[:4])
        )
    return False, "derived"


def begin_run(
    root: Path,
    provenance: ScientificProvenance,
    *,
    resolved_config: JSONDocument,
    seeds: Mapping[str, int],
    precision: str,
    tokenizer_hash: str,
    attempt: int | None = None,
    evidentiary: bool = True,
    git_dirty: bool | None = None,
    gpu_model: str = NOT_APPLICABLE,
    gpu_uuid: str = NOT_APPLICABLE,
    cuda: str = NOT_APPLICABLE,
    pytorch: str = NOT_APPLICABLE,
    wall_clock_start: str | None = None,
) -> RunAttempt:
    """Claim an attempt and materialise its manifest BEFORE any scientific work begins.

    Order matters: the attempt is claimed first, so two concurrent callers cannot both
    believe they own it, and the manifest is written immediately afterwards, so a crash
    leaves a run that is visibly unfinished rather than invisible [AUTH: 01 §15, §16].
    """
    provenance.require_valid()
    derived_dirty, dirty_source = _derive_repository_facts(
        root, provenance, resolved_config, evidentiary=evidentiary, git_dirty=git_dirty
    )
    if evidentiary and gpu_model != NOT_APPLICABLE:
        # Checked before the attempt is claimed, so a false hardware attribution never
        # leaves a run directory behind [AUTH: 01 §12, §16, §36].
        attribution = verify_runtime_attribution(
            root,
            provenance.environment_lock_sha256,
            {"gpu_model": gpu_model, "gpu_uuid": gpu_uuid, "cuda": cuda, "pytorch": pytorch},
        )
        if attribution:
            raise UnverifiedProvenanceError(
                "runtime identity does not match the accepted environment: "
                + "; ".join(attribution)
            )
    experiment = experiment_id(provenance)
    index = next_attempt(root, experiment) if attempt is None else attempt
    identifier = run_id(experiment, index)
    directory = attempt_directory(root, experiment, index)
    manifest_target = run_manifest_path(root, identifier)

    if manifest_target.exists():
        raise RunDirectoryExists(
            f"attempt {index} of this experiment already has a manifest and must never be"
            f" overwritten: {manifest_target}"
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError as exc:
        raise RunDirectoryExists(
            f"run directory already exists and must never be overwritten: {directory}"
        ) from exc
    # Re-checked after the claim: between the probe and the mkdir another writer may have
    # taken the manifest, and a surviving manifest outranks a fresh directory [AUTH: 01 §36].
    if manifest_target.exists():
        with suppress(OSError):
            directory.rmdir()
        raise RunDirectoryExists(
            f"attempt {index} was claimed concurrently; its manifest exists: {manifest_target}"
        )

    run = RunAttempt(root=root, experiment_id=experiment, attempt=index, run_id=identifier)
    run.log_directory.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        run_id=identifier,
        experiment_id=experiment,
        attempt=index,
        provenance=provenance,
        resolved_config=resolved_config,
        seeds=seeds,
        precision=precision,
        tokenizer_hash=tokenizer_hash,
        stdout_log_path=run.stdout_path,
        stderr_log_path=run.stderr_path,
        wall_clock_start=wall_clock_start or utc_now(),
        gpu_model=gpu_model,
        gpu_uuid=gpu_uuid,
        cuda=cuda,
        pytorch=pytorch,
        evidentiary=evidentiary,
        git_dirty=derived_dirty,
        git_dirty_source=dirty_source,
    )
    manifest["artifact_sizes"] = {}
    require_valid_run_manifest(manifest)
    write_canonical_json(run.manifest_path, manifest)
    return run


def load_attempt(root: Path, identifier: str) -> RunAttempt:
    """Rehydrate an attempt from its manifest."""
    path = root / "manifests" / "runs" / f"{identifier}.json"
    document = read_canonical_json(path)
    if not isinstance(document, Mapping):
        raise RunStoreError(f"{path}: run manifest is not a JSON object")
    return RunAttempt(
        root=root,
        experiment_id=str(document["experiment_id"]),
        attempt=int(str(document["attempt"])),
        run_id=identifier,
    )


def attempts_of(root: Path, experiment: str, *, limit: int = 1_000) -> list[RunAttempt]:
    """Every claimed attempt of one experiment, in order.

    Stopping at the first gap hid attempts: delete attempt 2's records and attempt 3, which
    may be a preserved failure, becomes invisible. This instead reads the two record
    namespaces once and matches derived RUN_IDs against them, so a gap is skipped rather than
    treated as the end. The scan stops as soon as every surviving record has been matched
    [AUTH: 01 §36].
    """
    claimed = claimed_run_ids(root)
    found: list[RunAttempt] = []
    for index in range(1, limit + 1):
        if len(found) == len(claimed):
            break
        identifier = run_id(experiment, index)
        if identifier in claimed:
            found.append(
                RunAttempt(root=root, experiment_id=experiment, attempt=index, run_id=identifier)
            )
    return found


__all__ = [
    "RUNNING",
    "DirtyTreeError",
    "RunAttempt",
    "RunDirectoryExists",
    "RunFinalizedError",
    "RunStoreError",
    "UnverifiedProvenanceError",
    "attempts_of",
    "begin_run",
    "load_attempt",
    "validate_stored_manifest",
]
