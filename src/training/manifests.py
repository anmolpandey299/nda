"""The adapter manifest: what a trained adapter is, and what it may be attributed to.

01 §13 requires an artifact's provenance to be recorded at the moment it is produced. An
adapter is the first artifact in this project that carries a research claim, so it must bind,
in one document, the run that made it, the model it modifies, the data it saw, the
configuration and seeds that determined it, the privacy regime it was trained under, and the
bytes it actually consists of.

There are TWO manifest roles, and the difference is structural rather than advisory:

* `FIXTURE_REFERENCE_ADAPTER_MANIFEST` describes a reference run over generated fixtures with
  a dependency-deferred backend. It is valid, and it is permanently non-evidentiary.
* `EVIDENTIARY_ADAPTER_MANIFEST` describes a run that may attribute a scientific result. It
  requires a validated RESEARCH_CORPUS, an immutable 40-hex model revision, the complete
  applicable seed families, every REQUIRED material constant resolved, an execution-contract
  identity, a READY production backend, and an artifact whose bytes are recomputed and
  compared — not merely a digest-shaped string.

A `NOT_RUN_DEPENDENCY(...)` backend cannot satisfy evidentiary validation: the check is on
the status value, so no combination of otherwise-valid fields lets a deferred run through.
A non-DP run declares the DP block rather than omitting it, because an absent field and a
declared "no DP" are different claims.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from src.data.manifest import CORPUS_ROLES, RESEARCH_CORPUS_ROLE
from src.dp.mechanism import INFERENTIAL, NON_INFERENTIAL, DPRun
from src.provenance.hashing import JSONDocument, JSONValue, sha256_canonical
from src.provenance.run_manifest import SEED_FIELDS
from src.training.lora import TrainableAudit
from src.training.seeds import SeedFamilies

ADAPTER_MANIFEST_SCHEMA: Final = "s06.adapter-manifest.v2"

#: The two manifest roles. A role is recorded in the document, so a reader never infers it.
FIXTURE_REFERENCE_ROLE: Final = "FIXTURE_REFERENCE_ADAPTER_MANIFEST"
EVIDENTIARY_ROLE: Final = "EVIDENTIARY_ADAPTER_MANIFEST"
MANIFEST_ROLES: Final[tuple[str, ...]] = (FIXTURE_REFERENCE_ROLE, EVIDENTIARY_ROLE)

#: A backend that actually ran.
BACKEND_READY: Final = "READY"
#: The prefix every dependency-deferred status carries.
DEPENDENCY_DEFERRED_PREFIX: Final = "NOT_RUN_DEPENDENCY"

_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_FLOATING_REVISIONS: Final = frozenset({"main", "master", "head", "latest"})

#: The privacy regimes 00 §8 defines. `NON_DP` is a declaration, not an omission.
PRIVACY_REGIMES: Final[tuple[str, ...]] = ("NON_DP", "DP_SAMPLE_LEVEL")

#: Every field an adapter manifest must carry [AUTH: 01 §13, §16, §30; 00 §8.2, §8.3].
ADAPTER_MANIFEST_REQUIRED: Final[tuple[str, ...]] = (
    "schema",
    "manifest_role",
    "execution_contract_sha256",
    "backend_status",
    "adapter_artifact_path",
    "adapter_alias",
    "run_id",
    "model_manifest_sha256",
    "model_revision",
    "data_manifest_sha256",
    "corpus_role",
    "training_config_sha256",
    "training_seed",
    "seeds",
    "privacy_regime",
    "lora_target_policy",
    "lora_rank",
    "lora_scaling",
    "trainable_audit",
    "optimizer_steps",
    "precision",
    "adapter_file_sha256",
    "induced_update_sha256",
    "base_parameter_sha256",
    "trainer_version",
    "dp",
)

#: The DP block's fields. Present under every regime; populated only under DP.
DP_BLOCK_FIELDS: Final[tuple[str, ...]] = (
    "adjacency",
    "delta",
    "clipping_norm",
    "noise_multiplier",
    "sample_rate",
    "steps",
    "dp_seed",
    "requested_epsilon",
    "achieved_epsilon",
    "accountant_version",
    "accounting_assumptions",
    "inferential_status",
    "dp_backend_status",
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_NOT_APPLICABLE: Final = "NOT_APPLICABLE"

#: Fields that must carry a real resolved value on an evidentiary manifest.
_RESOLVED_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("precision", "lora_target_policy")


class AdapterManifestError(ValueError):
    """An adapter manifest cannot attribute the artifact it describes."""


def _audit_block(audit: TrainableAudit) -> dict[str, JSONValue]:
    return {
        "policy": audit.policy,
        "architecture_family": audit.architecture_family,
        "target_modules": list(audit.target_modules),
        "adapter_parameters": list(audit.adapter_parameters),
        "n_trainable": audit.n_trainable,
        "n_frozen": audit.n_frozen,
        "mapping_identity": audit.mapping_identity,
    }


def _dp_block(run: DPRun | None) -> dict[str, JSONValue]:
    """The DP block. A non-DP run declares each field NOT_APPLICABLE rather than dropping it."""
    if run is None:
        block: dict[str, JSONValue] = dict.fromkeys(DP_BLOCK_FIELDS, _NOT_APPLICABLE)
        # A non-DP run took no DP steps, so it has no DP standing to declare either way.
        block["inferential_status"] = _NOT_APPLICABLE
        block["dp_backend_status"] = _NOT_APPLICABLE
        return block
    mechanism = run.accounting.mechanism
    return {
        "adjacency": mechanism.adjacency,
        "delta": mechanism.delta,
        "clipping_norm": mechanism.clipping_norm,
        "noise_multiplier": mechanism.noise_multiplier,
        "sample_rate": mechanism.sample_rate,
        "steps": mechanism.steps,
        "dp_seed": mechanism.dp_seed,
        "requested_epsilon": mechanism.requested_epsilon,
        "achieved_epsilon": run.accounting.achieved_epsilon,
        "accountant_version": run.accounting.accountant_version,
        "accounting_assumptions": list(mechanism.assumptions),
        "inferential_status": run.inferential_status,
        "dp_backend_status": run.accounting_backend_status,
    }


def build_adapter_manifest(
    *,
    manifest_role: str,
    execution_contract_sha256: str,
    backend_status: str,
    adapter_artifact_path: str,
    adapter_alias: str,
    run_id: str,
    model_manifest_sha256: str,
    model_revision: str,
    data_manifest_sha256: str,
    corpus_role: str,
    training_config_sha256: str,
    seeds: SeedFamilies,
    privacy_regime: str,
    audit: TrainableAudit,
    optimizer_steps: int,
    precision: str,
    adapter_file_sha256: str,
    induced_update_sha256: str,
    base_parameter_sha256: str,
    trainer_version: str,
    dp_run: DPRun | None = None,
) -> dict[str, JSONValue]:
    """Assemble the manifest and refuse to return one invalid for its declared role."""
    if manifest_role not in MANIFEST_ROLES:
        raise AdapterManifestError(f"{manifest_role!r} is not a declared manifest role")
    document: dict[str, JSONValue] = {
        "schema": ADAPTER_MANIFEST_SCHEMA,
        "manifest_role": manifest_role,
        "execution_contract_sha256": execution_contract_sha256,
        "backend_status": backend_status,
        "adapter_artifact_path": adapter_artifact_path,
        "adapter_alias": adapter_alias,
        "run_id": run_id,
        "model_manifest_sha256": model_manifest_sha256,
        "model_revision": model_revision,
        "data_manifest_sha256": data_manifest_sha256,
        "corpus_role": corpus_role,
        "training_config_sha256": training_config_sha256,
        "training_seed": seeds.training_seed,
        "seeds": seeds.as_manifest_seeds(),
        "privacy_regime": privacy_regime,
        "lora_target_policy": audit.policy,
        "lora_rank": audit.rank,
        "lora_scaling": audit.scaling,
        "trainable_audit": _audit_block(audit),
        "optimizer_steps": optimizer_steps,
        "precision": precision,
        "adapter_file_sha256": adapter_file_sha256,
        "induced_update_sha256": induced_update_sha256,
        "base_parameter_sha256": base_parameter_sha256,
        "trainer_version": trainer_version,
        "dp": _dp_block(dp_run),
    }
    require_valid_adapter_manifest(document)
    return document


def validate_adapter_manifest(document: JSONDocument) -> list[str]:
    """Every reason this adapter manifest is invalid FOR ITS DECLARED ROLE.

    A fixture-reference manifest is checked for internal consistency. An evidentiary one is
    additionally required to satisfy `_validate_evidentiary`, which is where the research
    corpus, the immutable revision, the complete seed families, the resolved configuration
    and the READY backend are demanded.
    """
    problems: list[str] = []
    for name in ADAPTER_MANIFEST_REQUIRED:
        if name not in document:
            problems.append(f"missing required field {name!r}")
        elif document[name] is None:
            problems.append(f"field {name!r} is null")
    if problems:
        return problems

    if document["schema"] != ADAPTER_MANIFEST_SCHEMA:
        problems.append(f"unknown adapter manifest schema {document['schema']!r}")
    for name in (
        "model_manifest_sha256",
        "data_manifest_sha256",
        "training_config_sha256",
        "adapter_file_sha256",
        "induced_update_sha256",
        "base_parameter_sha256",
    ):
        value = document[name]
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            problems.append(f"{name} is not a SHA256 digest")
    if document["corpus_role"] not in CORPUS_ROLES:
        problems.append(f"{document['corpus_role']!r} is not a declared corpus role")
    regime = document["privacy_regime"]
    if regime not in PRIVACY_REGIMES:
        problems.append(f"{regime!r} is not a declared privacy regime [AUTH: 00 §8]")
    steps = document["optimizer_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        problems.append("optimizer_steps must be a positive integer")

    role = document["manifest_role"]
    if role not in MANIFEST_ROLES:
        problems.append(f"{role!r} is not a declared manifest role")
    problems += _validate_seeds(document)
    problems += _validate_dp(document, regime)
    if role == EVIDENTIARY_ROLE:
        problems += _validate_evidentiary(document, regime)
    return problems


def _validate_evidentiary(document: JSONDocument, regime: object) -> list[str]:
    """The additional demands an evidentiary adapter manifest must meet.

    Every one of these is a way the reviewer's forged manifest passed: a floating revision, a
    fixture corpus, a master-seed-only seed block, and a deferred backend.
    """
    problems: list[str] = []

    revision = str(document.get("model_revision", ""))
    if revision.lower() in _FLOATING_REVISIONS or revision.startswith("refs/"):
        problems.append(
            f"model_revision {revision!r} is a floating branch or ref; an evidentiary adapter"
            " is pinned to an immutable revision [AUTH: 01 §8G]"
        )
    elif not _COMMIT_RE.match(revision):
        problems.append(
            f"model_revision {revision!r} is not an immutable 40-hex revision; the model has"
            " not been acquired [AUTH: 01 §8G; 03 §8]"
        )

    if document.get("corpus_role") != RESEARCH_CORPUS_ROLE:
        problems.append(
            f"corpus_role {document.get('corpus_role')!r} is not {RESEARCH_CORPUS_ROLE}; a"
            " generated corpus cannot attribute a scientific result [AUTH: 01 §14]"
        )

    backend = str(document.get("backend_status", ""))
    if backend != BACKEND_READY:
        problems.append(
            f"backend_status {backend!r} is not {BACKEND_READY}; a dependency-deferred"
            " backend cannot produce evidence [AUTH: 01 §12, §16]"
        )

    contract = document.get("execution_contract_sha256")
    if not isinstance(contract, str) or not _SHA256_RE.match(contract):
        problems.append("execution_contract_sha256 must bind the authorised execution contract")

    path = document.get("adapter_artifact_path")
    if not isinstance(path, str) or not path.strip():
        problems.append("an evidentiary adapter must name the artifact its hash was taken over")

    run_id = document.get("run_id")
    if not isinstance(run_id, str) or not _SHA256_RE.match(run_id):
        problems.append("run_id must be the Block A run identity [AUTH: 01 §15, §16]")

    unresolved = sorted(
        name
        for name in _RESOLVED_REQUIRED_FIELDS
        if not isinstance(document.get(name), str)
        or not str(document.get(name)).strip()
        or str(document.get(name)).startswith(("REQUIRED_", "UNRESOLVED", "NOT_APPLICABLE"))
    )
    if unresolved:
        problems.append(
            f"material configuration field(s) {', '.join(unresolved)} are unresolved; every"
            " REQUIRED constant must be frozen before an evidentiary run [AUTH: 01 §17]"
        )

    if regime == "DP_SAMPLE_LEVEL":
        problems += _validate_evidentiary_dp(document)
    return problems


def _validate_evidentiary_dp(document: JSONDocument) -> list[str]:
    """00 §8.2: an evidentiary DP claim needs a real accountant on a real backend."""
    block = document.get("dp")
    if not isinstance(block, Mapping):
        return ["dp must be an object"]
    problems: list[str] = []
    if block.get("dp_backend_status") != BACKEND_READY:
        problems.append(
            f"dp.dp_backend_status {block.get('dp_backend_status')!r} is not {BACKEND_READY};"
            " a deferred DP accountant/backend cannot support an evidentiary epsilon"
        )
    achieved, requested = block.get("achieved_epsilon"), block.get("requested_epsilon")
    if not isinstance(achieved, int | float) or isinstance(achieved, bool):
        problems.append("an evidentiary DP manifest must record a computed achieved_epsilon")
    if requested is None:
        problems.append("an evidentiary DP manifest must record the requested epsilon target")
    if achieved is not None and requested == achieved:
        problems.append("achieved_epsilon must be computed, not copied from the request")
    for name in ("delta", "clipping_norm", "noise_multiplier", "sample_rate", "steps", "dp_seed"):
        if block.get(name) in {None, _NOT_APPLICABLE}:
            problems.append(f"an evidentiary DP manifest must record dp.{name}")
    if block.get("inferential_status") != NON_INFERENTIAL:
        # A single manifest never carries inferential standing; that is a set-level property
        # validated by src.dp.mechanism.InferentialDPSet [AUTH: 00 §8.4].
        problems.append(
            "a single adapter manifest may not claim INFERENTIAL standing; inferential DP"
            " status is a property of the validated {101, 202, 303} set [AUTH: 00 §8.2, §8.4]"
        )
    return problems


def _validate_seeds(document: JSONDocument) -> list[str]:
    """01 §30: the families are recorded separately, and agree with the master seed."""
    seeds = document.get("seeds")
    if not isinstance(seeds, Mapping):
        return ["seeds must be an object of family -> integer"]
    problems: list[str] = []
    unknown = sorted(set(seeds) - set(SEED_FIELDS))
    if unknown:
        problems.append(f"seeds declares family/families outside 01 §30: {', '.join(unknown)}")
    for name, value in seeds.items():
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"seeds[{name!r}] is not an integer")
    if seeds.get("training_seed") != document.get("training_seed"):
        problems.append("seeds['training_seed'] disagrees with the recorded training_seed")
    derived = {name: value for name, value in seeds.items() if name != "training_seed"}
    if derived and len(set(derived.values())) != len(derived):
        problems.append(
            "two seed families share a value; a single global seed is not a seed family"
            " record [AUTH: 01 §30]"
        )
    return problems


def _validate_dp(document: JSONDocument, regime: object) -> list[str]:
    """The DP block must match the declared regime in both directions."""
    block = document.get("dp")
    if not isinstance(block, Mapping):
        return ["dp must be an object"]
    problems: list[str] = []
    missing = [name for name in DP_BLOCK_FIELDS if name not in block]
    if missing:
        return [f"dp is missing field(s): {', '.join(missing)}"]

    if regime == "NON_DP":
        populated = [
            name
            for name in DP_BLOCK_FIELDS
            if name not in {"inferential_status", "dp_backend_status"}
            and block[name] != _NOT_APPLICABLE
        ]
        if populated:
            problems.append(
                f"privacy_regime is NON_DP but dp field(s) {populated} are populated;"
                " a DP guarantee may not be recorded for a run that took no DP steps"
            )
        return problems

    if block["adjacency"] == _NOT_APPLICABLE:
        problems.append("a DP run must declare its adjacency relation [AUTH: 00 §8.2]")
    achieved, requested = block["achieved_epsilon"], block["requested_epsilon"]
    if not isinstance(achieved, int | float) or isinstance(achieved, bool):
        problems.append("achieved_epsilon must be a number computed by the accountant")
    if block["accountant_version"] == _NOT_APPLICABLE:
        problems.append("a DP run must record the accountant version that produced epsilon")
    if requested is not None and requested == achieved:
        # Not proof of fraud, but the overwhelmingly common way a target gets reported as a
        # result is assignment; a real accountant landing exactly on the target is not
        # expected at float precision, so it is refused and must be justified explicitly.
        problems.append(
            "achieved_epsilon is bitwise equal to requested_epsilon; the achieved guarantee"
            " must be computed from the mechanism actually used, not copied from the target"
        )
    if block["inferential_status"] not in {INFERENTIAL, NON_INFERENTIAL}:
        problems.append("dp.inferential_status is not a declared status")
    if not isinstance(block["accounting_assumptions"], list) or not block["accounting_assumptions"]:
        problems.append("a DP epsilon must record the assumptions it is valid under")
    return problems


def require_valid_adapter_manifest(document: JSONDocument) -> None:
    problems = validate_adapter_manifest(document)
    if problems:
        raise AdapterManifestError("; ".join(problems))


def adapter_manifest_sha256(document: JSONDocument) -> str:
    return sha256_canonical(dict(document))


def verify_adapter_artifact(document: JSONDocument, root: Path) -> None:
    """Recompute the adapter's bytes and compare them to what the manifest recorded.

    A digest-shaped string proves nothing about the file on disk. This reads the artifact the
    manifest names, recomputes both the file hash and the induced-update identity, and refuses
    any disagreement — which is how bytes changed after the manifest was written get caught at
    the moment the manifest is consumed [AUTH: 01 §16, §36].
    """
    from src.provenance.hashing import read_canonical_json
    from src.training.trainer import adapter_file_hash, load_adapter

    relative = document.get("adapter_artifact_path")
    if not isinstance(relative, str) or not relative.strip():
        raise AdapterManifestError("the manifest names no adapter artifact to verify against")
    path = root / relative
    if not path.is_file():
        raise AdapterManifestError(f"the adapter artifact is missing: {relative}")

    stored = read_canonical_json(path)
    if not isinstance(stored, Mapping):
        raise AdapterManifestError(f"{relative}: the adapter artifact is not a JSON object")
    recomputed_file = adapter_file_hash(stored)
    if recomputed_file != document.get("adapter_file_sha256"):
        raise AdapterManifestError(
            f"{relative}: the adapter bytes hash to {recomputed_file[:12]} but the manifest"
            f" records {str(document.get('adapter_file_sha256'))[:12]}; the artifact changed"
            " after the manifest was written [AUTH: 01 §16, §36]"
        )
    recomputed_update = load_adapter(stored).update_identity()
    if recomputed_update != document.get("induced_update_sha256"):
        raise AdapterManifestError(
            f"{relative}: the induced update hashes to {recomputed_update[:12]} but the"
            f" manifest records {str(document.get('induced_update_sha256'))[:12]}"
        )
