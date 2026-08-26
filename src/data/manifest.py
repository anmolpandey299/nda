"""S05 corpus -> 01 §14 data manifest, with the corpus role stated rather than assumed.

A data manifest is what a scientific result is attributed to, so a manifest built from
synthetic fixtures must be impossible to mistake for one built from PubMed. Block A's
`data_manifest` module owns the field vocabulary and validation; this module owns the
translation from a `MasterCorpus` into that vocabulary, and refuses to emit research-data
provenance for records nobody downloaded.

The role marker follows the precedent Block A set for models, where a fixture carries
`TEST_FIXTURE_NOT_A_RESEARCH_SUBJECT` [AUTH: 01 §13, §14; 00 §6.1].
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from src.data.acquisition import EVIDENTIARY_RESEARCH, AcquisitionReceipt
from src.data.corpus import AcquisitionSpec, CorpusError, MasterCorpus
from src.data.dedup import EXACT_ALGORITHM, NEAR_DUPLICATE_ALGORITHM
from src.data.normalise import NORMALISATION_VERSION
from src.provenance.data_manifest import (
    data_manifest_sha256,
    partition_sha256,
    require_valid_data_manifest,
    stamp_manifest_sha256,
)
from src.provenance.hashing import JSONDocument, JSONValue, sha256_canonical

#: A corpus of generated records. Never research data, whatever else it resembles.
FIXTURE_CORPUS_ROLE: Final = "TEST_FIXTURE_NOT_RESEARCH_DATA"
#: A corpus of real acquired records.
RESEARCH_CORPUS_ROLE: Final = "RESEARCH_CORPUS"

CORPUS_ROLES: Final[tuple[str, ...]] = (FIXTURE_CORPUS_ROLE, RESEARCH_CORPUS_ROLE)

#: What `source_query` says when there was no query, because nothing was fetched.
NO_QUERY: Final = "SYNTHETIC_FIXTURE_NO_QUERY"
#: What `source_version` says for a generated corpus.
NO_SOURCE_VERSION: Final = "SYNTHETIC_FIXTURE_NO_SOURCE_VERSION"

_ROLE_PREFIX: Final = "corpus_role="
_FIXTURE_ALIAS_PREFIX: Final = "fixture_"
_TIMESTAMP_RE: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)


class CorpusManifestError(CorpusError):
    """A data manifest was requested that would misdescribe where the records came from."""


def corpus_role(document: JSONDocument) -> str:
    """The declared role, read back from the manifest. Absent is not 'research'."""
    notes = document.get("notes")
    if not isinstance(notes, str):
        raise CorpusManifestError("the manifest declares no corpus role")
    for part in notes.split(";"):
        stripped = part.strip()
        if stripped.startswith(_ROLE_PREFIX):
            role = stripped[len(_ROLE_PREFIX) :]
            if role not in CORPUS_ROLES:
                raise CorpusManifestError(f"{role!r} is not a declared corpus role")
            return role
    raise CorpusManifestError("the manifest declares no corpus role")


def is_fixture_corpus(document: JSONDocument) -> bool:
    return corpus_role(document) == FIXTURE_CORPUS_ROLE


def require_research_corpus(document: JSONDocument) -> None:
    """Gate for anything evidentiary. A fixture manifest fails it, loudly."""
    role = corpus_role(document)
    if role != RESEARCH_CORPUS_ROLE:
        raise CorpusManifestError(
            f"this corpus is {role}; it may not attribute a scientific result [AUTH: 01 §14, §16]"
        )


def assert_manifest_unchanged(document: JSONDocument, expected_sha256: str) -> None:
    """A real manifest is immutable once privacy outcomes are opened [AUTH: 01 §14, §27].

    The check recomputes the identity rather than reading the stored stamp, so editing the
    manifest and restamping it is caught too: the recomputed hash no longer matches the value
    the earlier result was attributed to.
    """
    recomputed = data_manifest_sha256(dict(document))
    if recomputed != expected_sha256:
        raise CorpusManifestError(
            f"the data manifest has changed since {expected_sha256[:12]} (now"
            f" {recomputed[:12]}); a manifest a result was attributed to is immutable"
            " [AUTH: 01 §14, §27]"
        )


def preprocessing_config_identity(
    spec: AcquisitionSpec,
    *,
    shingle_size: int,
    near_duplicate_threshold: float,
    sizes: Mapping[str, int],
) -> str:
    """Hash of everything that decided which records survived and where they went."""
    return sha256_canonical(
        {
            "acquisition": {
                "query": spec.query,
                "publication_date_floor": spec.publication_date_floor,
                "required_source_fields": list(spec.required_source_fields),
                "version": spec.version,
            },
            "normalisation_version": NORMALISATION_VERSION,
            "exact_algorithm": EXACT_ALGORITHM,
            "near_duplicate_algorithm": NEAR_DUPLICATE_ALGORITHM,
            "shingle_size": shingle_size,
            "near_duplicate_threshold": near_duplicate_threshold,
            "partition_sizes": dict(sizes),
        }
    )


def raw_data_identity(records: Sequence[str]) -> str:
    """Identity of the acquired bytes, as the hashes of what was actually read."""
    return sha256_canonical(sorted(records))


def fixture_corpus_manifest(
    corpus: MasterCorpus,
    *,
    spec: AcquisitionSpec,
    raw_data_sha256: str,
    preprocessing_code_commit: str,
    preprocessing_config_sha256: str,
    download_timestamp_utc: str,
    dataset_alias: str,
) -> dict[str, JSONValue]:
    """The manifest for a GENERATED corpus. Its role is fixed and cannot be raised.

    `role` is deliberately not a parameter. The only way to obtain a `RESEARCH_CORPUS`
    manifest is `research_corpus_manifest`, which takes an `AcquisitionReceipt` and derives
    every provenance field from the acquisition walk [AUTH: 01 §14].
    """
    return _build_corpus_manifest(
        corpus,
        role=FIXTURE_CORPUS_ROLE,
        spec=spec,
        raw_data_sha256=raw_data_sha256,
        preprocessing_code_commit=preprocessing_code_commit,
        preprocessing_config_sha256=preprocessing_config_sha256,
        download_timestamp_utc=download_timestamp_utc,
        dataset_alias=dataset_alias,
    )


def research_corpus_manifest(
    receipt: AcquisitionReceipt, *, preprocessing_code_commit: str, dataset_alias: str
) -> dict[str, JSONValue]:
    """The 01 §14 document for a corpus that walked the canonical research path.

    Nothing here is a caller assertion. The query, the publication-date rule, the raw-data
    identity, the preprocessing identity, the split seed and the split ids are all read off
    the receipt, and the receipt exists only because `acquire_research_corpus` issued it.
    """
    if receipt.provenance_class != EVIDENTIARY_RESEARCH:
        raise CorpusManifestError(
            f"receipt is {receipt.provenance_class}; it cannot carry research provenance"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", preprocessing_code_commit):
        raise CorpusManifestError("preprocessing_code_commit must be a 40-hex commit SHA")
    document = _build_corpus_manifest(
        receipt.corpus,
        role=RESEARCH_CORPUS_ROLE,
        spec=receipt.spec,
        raw_data_sha256=receipt.raw_data_sha256,
        preprocessing_code_commit=preprocessing_code_commit,
        preprocessing_config_sha256=receipt.preprocessing_config_sha256,
        download_timestamp_utc=receipt.download_timestamp_utc,
        dataset_alias=dataset_alias,
    )
    _check_publication_evidence(receipt, document)
    return document


def _check_publication_evidence(
    receipt: AcquisitionReceipt, document: Mapping[str, JSONValue]
) -> None:
    """Every retained id must carry publication evidence from the walk [AUTH: 01 §8F]."""
    floor = receipt.spec.publication_date_floor
    placed = {record_id for ids in receipt.corpus.partitions.values() for record_id in ids}
    missing = sorted(placed - set(receipt.publication_evidence))
    if missing:
        raise CorpusManifestError(
            f"{len(missing)} split record(s) carry no publication evidence, first"
            f" {missing[0]!r}; a research corpus records the date each record qualified on"
        )
    early = sorted(
        record_id for record_id in placed if receipt.publication_evidence[record_id] < floor
    )
    if early:
        raise CorpusManifestError(
            f"{len(early)} split record(s) publish before the {floor} floor, first"
            f" {early[0]!r} [AUTH: 01 §8F]"
        )


def _build_corpus_manifest(
    corpus: MasterCorpus,
    *,
    role: str,
    spec: AcquisitionSpec,
    raw_data_sha256: str,
    preprocessing_code_commit: str,
    preprocessing_config_sha256: str,
    download_timestamp_utc: str,
    dataset_alias: str,
) -> dict[str, JSONValue]:
    """Shared body. PRIVATE: `role` is a parameter here and nowhere a caller can reach."""
    if role not in CORPUS_ROLES:
        raise CorpusManifestError(f"{role!r} is not a declared corpus role")
    if not _TIMESTAMP_RE.match(download_timestamp_utc):
        raise CorpusManifestError("download_timestamp_utc must be an ISO-8601 UTC timestamp")

    if role == FIXTURE_CORPUS_ROLE:
        if not dataset_alias.startswith(_FIXTURE_ALIAS_PREFIX):
            raise CorpusManifestError(
                f"a fixture corpus alias must start with {_FIXTURE_ALIAS_PREFIX!r}"
            )
        source_query, source_version = NO_QUERY, NO_SOURCE_VERSION
    else:
        if dataset_alias.startswith(_FIXTURE_ALIAS_PREFIX):
            raise CorpusManifestError("a research corpus may not carry a fixture alias")
        if spec.query in {NO_QUERY, ""}:
            raise CorpusManifestError(
                "a research corpus manifest requires the real acquisition query"
            )
        source_query, source_version = spec.query, spec.publication_date_floor

    split_ids = {name: list(ids) for name, ids in corpus.partitions.items()}
    document: dict[str, JSONValue] = {
        "dataset_alias": dataset_alias,
        "notes": f"{_ROLE_PREFIX}{role}; normalisation={NORMALISATION_VERSION}",
        "record_count": sum(len(ids) for ids in split_ids.values()),
        "source_query": source_query,
        "source_version": source_version,
        "download_timestamp_utc": download_timestamp_utc,
        "raw_data_sha256": raw_data_sha256,
        "preprocessing_code_commit": preprocessing_code_commit,
        "preprocessing_config_sha256": preprocessing_config_sha256,
        "split_rng_seed": corpus.split_seed,
        "split_ids": split_ids,
        "deduplication_method": f"{EXACT_ALGORITHM}+{NEAR_DUPLICATE_ALGORITHM}",
        "deduplication_version": NORMALISATION_VERSION,
        "final_split_sha256": {name: partition_sha256(ids) for name, ids in split_ids.items()},
    }
    stamped = stamp_manifest_sha256(document)
    require_valid_data_manifest(stamped)
    return stamped
