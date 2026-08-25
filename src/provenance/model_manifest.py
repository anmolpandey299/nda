"""Model provenance manifest contract [AUTH: 01 §13, §8G].

`manifests/models/<model_alias>.json` records what a model *was* when a run used it. The
nine 01 §13 fields are mandatory for every model. The 01 §8G research-model fields are
accepted now and become mandatory when the model is a research subject rather than a local
test fixture, so an S02 fixture manifest and a real four-family manifest share one contract
and one validator.

No model is downloaded here. Block A builds and validates the contract; acquisition belongs
to the stage that needs a real checkpoint.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from src.provenance.hashing import (
    JSONDocument,
    JSONValue,
    canonical_json_bytes,
    read_canonical_json,
    sha256_canonical,
    write_canonical_json,
)

#: Mandatory for every model manifest [AUTH: 01 §13].
MODEL_MANIFEST_REQUIRED: Final[tuple[str, ...]] = (
    "model_id",
    "revision",
    "config_sha256",
    "tokenizer_sha256",
    "license_snapshot_sha256",
    "download_timestamp_utc",
    "parameter_count",
    "dtype_on_disk",
    "role",
)

#: Additionally mandatory once the model is a research subject [AUTH: 01 §8G].
RESEARCH_MODEL_REQUIRED: Final[tuple[str, ...]] = (
    "architecture_family",
    "base_or_instruct",
    "weight_file_sha256",
    "lora_target_mapping",
    "frozen_modules",
    "license",
)

#: Optional §8G fields; absent is not a defect, a fabricated value would be.
RESEARCH_MODEL_OPTIONAL: Final[tuple[str, ...]] = (
    "effective_parameter_count",
    "training_data_cutoff",
)

#: A local deterministic fixture is explicitly not a research subject [AUTH: 01 §20].
FIXTURE_ROLE: Final = "TEST_FIXTURE_NOT_A_RESEARCH_SUBJECT"

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_TIMESTAMP_RE: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)
_FLOATING: Final = frozenset({"main", "master", "head"})

#: 01 §8G records a "weight-file SHA256/Xet identifier". Either exact form is accepted; a
#: free-text label is not an identity.
_WEIGHT_IDENTITY_RE: Final = re.compile(r"^(?:[0-9a-f]{64}|xet:[A-Za-z0-9._:-]{8,128})$")
_SHA256_FIELDS: Final[tuple[str, ...]] = (
    "config_sha256",
    "tokenizer_sha256",
    "license_snapshot_sha256",
)

#: Fields whose contract is a string. Checked for type before any syntax rule runs, so a
#: non-string identity cannot slip past an `isinstance`-guarded check [AUTH: 01 §13, §8G].
#: `parameter_count` is deliberately absent: its schema is legitimately numeric.
_STRING_FIELDS: Final[tuple[str, ...]] = (
    "model_id",
    "revision",
    "dtype_on_disk",
    "role",
    "architecture_family",
    "base_or_instruct",
    "license",
    "weight_file_sha256",
    "download_timestamp_utc",
    *_SHA256_FIELDS,
)


class ModelManifestError(ValueError):
    """A model manifest is missing evidentiary fields or carries an unusable value."""


class ManifestExistsError(ModelManifestError):
    """A different manifest already occupies this alias; provenance is never overwritten."""


def _syntax(
    document: JSONDocument, field: str, pattern: re.Pattern[str], description: str
) -> list[str]:
    """Type first, then syntax. An `isinstance`-guarded syntax check is fail-open."""
    if field not in document:
        return []
    value = document[field]
    if not isinstance(value, str):
        return [f"field {field!r} must be a string, got {type(value).__name__}"]
    if not pattern.match(value):
        return [f"field {field!r} is not {description}"]
    return []


def is_research_subject(document: JSONDocument) -> bool:
    """Derived from the manifest, never from a caller's default.

    A caller who omits a `research_subject=True` flag could otherwise have a manifest that
    declares itself a research subject validated against the weaker fixture contract, which
    is the whole strictness of 01 §8G bypassed by a default argument [AUTH: 01 §8G, §13].
    """
    return str(document.get("role", "")).strip() != FIXTURE_ROLE


def validate_model_manifest(document: JSONDocument) -> list[str]:
    """Every reason this manifest may not attribute a scientific result.

    The contract applied is derived from the manifest's own `role`: anything that is not the
    local test fixture is a research subject and receives the full 01 §8G contract, including
    an immutable commit revision and a real weight identity. Fails closed: a missing required
    field, an empty string, a placeholder, a floating revision or a non-positive parameter
    count are all defects, not warnings.
    """
    problems: list[str] = []
    research_subject = is_research_subject(document)
    required = list(MODEL_MANIFEST_REQUIRED)
    if research_subject:
        required += list(RESEARCH_MODEL_REQUIRED)
    for name in required:
        if name not in document:
            problems.append(f"missing required field {name!r}")
        elif document[name] is None:
            problems.append(f"field {name!r} is null")
        elif isinstance(document[name], str) and not str(document[name]).strip():
            problems.append(f"field {name!r} is empty")

    for name in _STRING_FIELDS:
        if name in document and not isinstance(document[name], str):
            problems.append(f"field {name!r} must be a string, got {type(document[name]).__name__}")
    for name in _SHA256_FIELDS:
        problems += _syntax(document, name, _SHA256_RE, "a SHA256 digest")

    revision = document.get("revision")
    if isinstance(revision, str):
        if revision.strip().lower() in _FLOATING:
            problems.append(f"revision {revision!r} is a floating branch [AUTH: 01 §8G]")
        elif research_subject and not _GIT_SHA_RE.match(revision.strip()):
            # 01 §8G: "exact revision SHA". A tag or branch name moves, so it cannot pin an
            # evidentiary research model even when it is not literally `main`.
            problems.append(
                f"revision {revision!r} is not an immutable 40-hex commit revision [AUTH: 01 §8G]"
            )

    if research_subject:
        problems += _syntax(
            document,
            "weight_file_sha256",
            _WEIGHT_IDENTITY_RE,
            "a SHA256 digest nor a Xet identifier [AUTH: 01 §8G]",
        )

    problems += _syntax(
        document, "download_timestamp_utc", _TIMESTAMP_RE, "an ISO-8601 UTC timestamp"
    )

    count = document.get("parameter_count")
    if isinstance(count, bool) or not isinstance(count, int):
        problems.append("parameter_count is not an integer")
    elif count <= 0:
        problems.append("parameter_count is not positive")

    unknown = sorted(
        set(document)
        - set(MODEL_MANIFEST_REQUIRED)
        - set(RESEARCH_MODEL_REQUIRED)
        - set(RESEARCH_MODEL_OPTIONAL)
    )
    if unknown:
        problems.append(f"unknown field(s): {', '.join(unknown)}")
    return problems


def require_valid_model_manifest(document: JSONDocument) -> None:
    problems = validate_model_manifest(document)
    if problems:
        raise ModelManifestError("; ".join(problems))


def model_manifest_sha256(document: JSONDocument) -> str:
    return sha256_canonical(dict(document))


def model_manifest_path(root: Path, alias: str) -> Path:
    if not alias or "/" in alias or alias.startswith("."):
        raise ModelManifestError(f"invalid model alias: {alias!r}")
    return root / "manifests" / "models" / f"{alias}.json"


def write_model_manifest(root: Path, alias: str, document: JSONDocument) -> tuple[Path, str]:
    """Validate then persist, and never overwrite a different manifest.

    Re-writing identical bytes is idempotent; writing different bytes under an alias that
    already exists is refused, because a model manifest is the provenance a past result was
    attributed to [AUTH: 01 §13, §27, §36].
    """
    require_valid_model_manifest(document)
    path = model_manifest_path(root, alias)
    payload = canonical_json_bytes(dict(document))
    if path.is_file():
        if path.read_bytes() != payload:
            raise ManifestExistsError(
                f"model manifest {alias!r} already exists with different content;"
                " provenance is never overwritten [AUTH: 01 §13, §27]"
            )
        return path, sha256_canonical(dict(document))
    digest = write_canonical_json(path, dict(document))
    return path, digest


def read_model_manifest(path: Path) -> dict[str, JSONValue]:
    document = read_canonical_json(path)
    if not isinstance(document, Mapping):
        raise ModelManifestError(f"{path}: manifest is not a JSON object")
    return dict(document)
