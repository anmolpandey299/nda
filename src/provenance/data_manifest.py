"""Data provenance manifest contract [AUTH: 01 §14; 00 §6.2, §33].

Records what a dataset version *was*: where it came from, how it was preprocessed, how it
was split, and what each partition hashed to. `data_manifest_sha256` is a RUN_ID input, so
this document's identity is part of every result's identity [AUTH: 01 §15].

Two properties are enforced here rather than left to convention, because both are ways a
privacy result silently stops meaning what it claims:

* partitions are disjoint — no record id appears in two partitions, so calibration and
  evaluation cannot overlap [AUTH: 01 §23; 00 §33; 02 §C1];
* the declared per-partition hash matches the recorded ids, so a manifest cannot claim a
  split it does not contain.

No data is acquired here. Block A builds and validates the contract; the real corpus belongs
to the stage that acquires it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
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

#: Every 01 §14 line, as an explicit field name.
DATA_MANIFEST_REQUIRED: Final[tuple[str, ...]] = (
    "source_query",
    "source_version",
    "download_timestamp_utc",
    "raw_data_sha256",
    "preprocessing_code_commit",
    "preprocessing_config_sha256",
    "split_rng_seed",
    "split_ids",
    "deduplication_method",
    "deduplication_version",
    "final_split_sha256",
)

DATA_MANIFEST_OPTIONAL: Final[tuple[str, ...]] = ("dataset_alias", "notes", "record_count")

#: The self-referential identity field, excluded from its own computation.
MANIFEST_SHA256_FIELD: Final = "manifest_sha256"

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_TIMESTAMP_RE: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)


class DataManifestError(ValueError):
    """A data manifest is incomplete, inconsistent, or describes overlapping partitions."""


class DataManifestExistsError(DataManifestError):
    """A different manifest already occupies this alias; provenance is never overwritten."""


def _syntax(
    document: JSONDocument, field: str, pattern: re.Pattern[str], description: str
) -> list[str]:
    """Type first, then syntax.

    Guarding a syntax check with `isinstance(value, str)` is fail-open: an integer skips the
    check entirely and a non-string identity passes validation [AUTH: 01 §14, §15].
    """
    if field not in document:
        return []
    value = document[field]
    if not isinstance(value, str):
        return [f"{field} must be a string, got {type(value).__name__}"]
    if not pattern.match(value):
        return [f"{field} is not {description}"]
    return []


def partition_sha256(record_ids: Sequence[str]) -> str:
    """Identity of one partition: the canonical hash of its id list, order preserved.

    Order is preserved rather than sorted because record order is part of what the split
    was; two manifests with the same ids in a different order describe different splits.
    """
    return sha256_canonical(list(record_ids))


def validate_data_manifest(document: JSONDocument) -> list[str]:
    """Every reason this manifest may not attribute a scientific result."""
    problems: list[str] = []
    for name in DATA_MANIFEST_REQUIRED:
        if name not in document:
            problems.append(f"missing required field {name!r}")
        elif document[name] is None:
            problems.append(f"field {name!r} is null")

    problems += _syntax(document, "raw_data_sha256", _SHA256_RE, "a SHA256 digest")
    problems += _syntax(document, "preprocessing_config_sha256", _SHA256_RE, "a SHA256 digest")
    problems += _syntax(
        document, "preprocessing_code_commit", _GIT_SHA_RE, "a full 40-hex commit SHA"
    )
    problems += _syntax(
        document, "download_timestamp_utc", _TIMESTAMP_RE, "an ISO-8601 UTC timestamp"
    )
    seed = document.get("split_rng_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.append("split_rng_seed is not an integer")

    problems += _validate_splits(document)
    problems += _validate_stamped_identity(document)

    unknown = sorted(
        set(document)
        - set(DATA_MANIFEST_REQUIRED)
        - set(DATA_MANIFEST_OPTIONAL)
        - {MANIFEST_SHA256_FIELD}
    )
    if unknown:
        problems.append(f"unknown field(s): {', '.join(unknown)}")
    return problems


def _validate_stamped_identity(document: JSONDocument) -> list[str]:
    """A stored `manifest_sha256` must still hash the manifest's identity-bearing content.

    Otherwise the stamp goes stale silently: mutate `raw_data_sha256`, keep the old stamp,
    and the manifest still claims the identity a past result was attributed to. The stamp is
    computed over the document with its own field removed, so it is not self-referential
    [AUTH: 01 §14, §15].
    """
    stamped = document.get(MANIFEST_SHA256_FIELD)
    if stamped is None:
        return []
    if not isinstance(stamped, str) or not _SHA256_RE.match(stamped):
        return [f"{MANIFEST_SHA256_FIELD} is not a SHA256 digest"]
    try:
        recomputed = data_manifest_sha256(document)
    except ValueError as exc:
        return [f"the manifest is not canonicalisable: {exc}"]
    if recomputed != stamped:
        return [
            f"{MANIFEST_SHA256_FIELD} {stamped[:12]} no longer hashes this manifest"
            f" ({recomputed[:12]}); the stamp is stale [AUTH: 01 §14, §15]"
        ]
    return []


def _validate_splits(document: JSONDocument) -> list[str]:
    problems: list[str] = []
    splits = document.get("split_ids")
    declared = document.get("final_split_sha256")
    if not isinstance(splits, Mapping) or not splits:
        return ["split_ids must be a non-empty object of partition -> list of record ids"]
    if not isinstance(declared, Mapping):
        return ["final_split_sha256 must be an object of partition -> SHA256"]

    seen: dict[str, str] = {}
    for partition, ids in splits.items():
        if not isinstance(ids, Sequence) or isinstance(ids, str):
            problems.append(f"split_ids[{partition!r}] is not a list")
            continue
        if not all(isinstance(record, str) and record for record in ids):
            problems.append(f"split_ids[{partition!r}] contains a non-string or empty id")
            continue
        record_ids = [str(record) for record in ids]
        if len(set(record_ids)) != len(record_ids):
            problems.append(f"split_ids[{partition!r}] repeats a record id")
        for record in record_ids:
            other = seen.setdefault(record, partition)
            if other != partition:
                problems.append(
                    f"record {record!r} appears in both {other!r} and {partition!r};"
                    " partitions must be disjoint [AUTH: 01 §23; 00 §33]"
                )
        expected = partition_sha256(record_ids)
        if partition not in declared:
            problems.append(f"final_split_sha256 has no entry for {partition!r}")
        elif not isinstance(declared[partition], str) or not _SHA256_RE.match(
            str(declared[partition])
        ):
            problems.append(f"final_split_sha256[{partition!r}] is not a SHA256 digest")
        elif declared[partition] != expected:
            problems.append(f"final_split_sha256[{partition!r}] does not match its recorded ids")

    extra = sorted(set(declared) - set(splits))
    if extra:
        problems.append(f"final_split_sha256 declares unknown partition(s): {', '.join(extra)}")
    return problems


def require_valid_data_manifest(document: JSONDocument) -> None:
    problems = validate_data_manifest(document)
    if problems:
        raise DataManifestError("; ".join(problems))


def data_manifest_sha256(document: JSONDocument) -> str:
    """DATA_MANIFEST_SHA256 — a RUN_ID input [AUTH: 01 §14, §15].

    Computed over the document with its own `manifest_sha256` field removed, so stamping the
    identity into the document cannot change the identity.
    """
    payload = {k: v for k, v in document.items() if k != MANIFEST_SHA256_FIELD}
    return sha256_canonical(payload)


def stamp_manifest_sha256(document: JSONDocument) -> dict[str, JSONValue]:
    """Return the manifest carrying its own identity."""
    stamped = {k: v for k, v in document.items() if k != MANIFEST_SHA256_FIELD}
    stamped[MANIFEST_SHA256_FIELD] = data_manifest_sha256(document)
    return stamped


def data_manifest_path(root: Path, alias: str) -> Path:
    if not alias or "/" in alias or alias.startswith("."):
        raise DataManifestError(f"invalid dataset alias: {alias!r}")
    return root / "manifests" / "data" / f"{alias}.json"


def write_data_manifest(root: Path, alias: str, document: JSONDocument) -> tuple[Path, str]:
    """Validate, stamp the identity, persist, and never overwrite a different manifest.

    Re-writing identical bytes is idempotent. Writing a different split, a different dedup
    version or different raw data under an alias that already exists is refused: a data
    manifest identity is a RUN_ID input, so replacing it in place would silently re-attribute
    every result that cited it [AUTH: 01 §14, §15, §27, §36; 00 §33.3].
    """
    require_valid_data_manifest(document)
    stamped = stamp_manifest_sha256(document)
    identity = stamped[MANIFEST_SHA256_FIELD]
    assert isinstance(identity, str)
    path = data_manifest_path(root, alias)
    payload = canonical_json_bytes(stamped)
    if path.is_file():
        if path.read_bytes() != payload:
            raise DataManifestExistsError(
                f"data manifest {alias!r} already exists with different content;"
                " provenance is never overwritten [AUTH: 01 §14, §27]"
            )
        return path, identity
    write_canonical_json(path, stamped)
    return path, identity


def read_data_manifest(path: Path) -> dict[str, JSONValue]:
    document = read_canonical_json(path)
    if not isinstance(document, Mapping):
        raise DataManifestError(f"{path}: manifest is not a JSON object")
    return dict(document)
