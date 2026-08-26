"""Acquisition contract, publication-date eligibility, and the immutable master corpus.

01 §8F sets the recency lock for the current core panel:

    publication_date >= 2026-04-01

and states plainly that **indexing date is not accepted as a substitute**. Both dates are
therefore carried separately on every record, and only `publication_date` is ever tested. This
supersedes 00 §6.1's 2025 target, which was written for the earlier single-model panel.

Partitions are created once, after deduplication, from the frozen split seed, and do not move
across training seeds [AUTH: 00 §6.2].

No network access lives here. Acquisition is a *specification* — the query, the retrieval
timestamp and the per-record source fields that the production fetcher must fill and freeze
into the data manifest [AUTH: 01 §14, §8F].
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.data.dedup import DedupResult, deduplicate
from src.data.normalise import NormalisedRecord
from src.provenance.hashing import JSONValue

#: The frozen partition names, in the order the splitter fills them [AUTH: 00 §6.2].
PARTITION_ORDER: Final[tuple[str, ...]] = (
    "TRAIN_CANDIDATES",
    "CALIBRATION_NONMEMBERS",
    "EVAL_NONMEMBERS",
    "RESERVE",
)

#: Fields the production acquisition path must freeze per record [AUTH: 01 §8F, §14].
REQUIRED_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "pmid",
    "publication_date",
    "indexing_date",
    "source_identity",
)

ACQUISITION_SPEC_VERSION: Final = "s05.acquisition.v1"


class CorpusError(ValueError):
    """The corpus cannot be built as specified."""


@dataclass(frozen=True)
class SourceRecord:
    """One acquired record, before normalisation [AUTH: 01 §8F, §14]."""

    record_id: str
    pmid: str
    publication_date: str
    indexing_date: str
    source_identity: str
    fields: Mapping[str, str]

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "record_id": self.record_id,
            "pmid": self.pmid,
            "publication_date": self.publication_date,
            "indexing_date": self.indexing_date,
            "source_identity": self.source_identity,
        }


def _parse_date(value: str, *, field: str, record_id: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise CorpusError(f"{record_id}: {field} {value!r} is not an ISO date") from exc


def is_eligible(record: SourceRecord, *, floor: str) -> bool:
    """Publication date at or after the floor. Indexing date is never consulted.

    A record indexed long after a late publication, or published before the floor but indexed
    after it, is decided on `publication_date` alone [AUTH: 01 §8F].
    """
    published = _parse_date(
        record.publication_date, field="publication_date", record_id=record.record_id
    )
    # The indexing date is parsed so a malformed one is still caught, and is then deliberately
    # discarded: it may not influence the decision [AUTH: 01 §8F].
    _parse_date(record.indexing_date, field="indexing_date", record_id=record.record_id)
    return published >= _dt.date.fromisoformat(floor)


def filter_eligible(
    records: Sequence[SourceRecord], *, floor: str
) -> tuple[list[SourceRecord], list[str]]:
    """Split the acquisition into eligible records and the ids refused, with the floor stated."""
    eligible: list[SourceRecord] = []
    refused: list[str] = []
    for record in records:
        if is_eligible(record, floor=floor):
            eligible.append(record)
        else:
            refused.append(record.record_id)
    return eligible, refused


@dataclass(frozen=True)
class AcquisitionSpec:
    """What the production fetcher must freeze. It performs no retrieval itself."""

    query: str
    publication_date_floor: str
    required_source_fields: tuple[str, ...] = REQUIRED_SOURCE_FIELDS
    version: str = ACQUISITION_SPEC_VERSION

    def validate_record(self, record: SourceRecord) -> list[str]:
        problems: list[str] = []
        for name in self.required_source_fields:
            value = getattr(record, name, None)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{record.record_id}: missing acquisition field {name!r}")
        return problems


@dataclass(frozen=True)
class MasterCorpus:
    """The immutable partitioned corpus [AUTH: 00 §6.2]."""

    partitions: Mapping[str, tuple[str, ...]]
    retained: tuple[NormalisedRecord, ...]
    dedup: DedupResult
    split_seed: int
    surplus: tuple[str, ...]

    def partition_of(self, record_id: str) -> str | None:
        for name, ids in self.partitions.items():
            if record_id in ids:
                return name
        return None

    def text_hashes(self) -> Mapping[str, str]:
        return {record.record_id: record.text_sha256 for record in self.retained}


def _ordering_key(record: NormalisedRecord, seed: int) -> str:
    """Deterministic shuffle key: SHA256(seed | text hash).

    Keyed on the normalised text hash rather than the record id, so two acquisitions of the
    same logical corpus with different id schemes still partition identically.
    """
    return hashlib.sha256(f"{seed}|{record.text_sha256}".encode()).hexdigest()


def build_master_corpus(
    records: Sequence[NormalisedRecord],
    *,
    sizes: Mapping[str, int],
    split_seed: int,
    shingle_size: int,
    near_duplicate_threshold: float,
) -> MasterCorpus:
    """Deduplicate, then split. Never the other way round [AUTH: 00 §6.2].

    Sizes are enforced exactly: too few eligible records fails closed rather than quietly
    producing a smaller evaluation set. Surplus records beyond the requested totals are
    retained under an explicit `surplus` list — the documented rule is that the deterministic
    order fills the partitions in `PARTITION_ORDER` and everything after them is surplus.
    """
    missing = [name for name in PARTITION_ORDER if name not in sizes]
    if missing:
        raise CorpusError(f"split sizes missing for {', '.join(missing)}")
    unknown = sorted(set(sizes) - set(PARTITION_ORDER))
    if unknown:
        raise CorpusError(
            f"unknown partition(s) {', '.join(unknown)}; the master corpus has exactly the"
            f" 00 §6.2 partitions {PARTITION_ORDER}"
        )
    negative = sorted(name for name in PARTITION_ORDER if sizes[name] < 0)
    if negative:
        raise CorpusError(f"partition size(s) {', '.join(negative)} are negative")

    result = deduplicate(records, shingle_size=shingle_size, threshold=near_duplicate_threshold)
    required = sum(sizes[name] for name in PARTITION_ORDER)
    if len(result.retained) < required:
        raise CorpusError(
            f"{len(result.retained)} deduplicated records available but {required} required;"
            " refusing to build undersized partitions [AUTH: 00 §6.2]"
        )

    ordered = sorted(result.retained, key=lambda record: _ordering_key(record, split_seed))
    partitions: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for name in PARTITION_ORDER:
        take = sizes[name]
        partitions[name] = tuple(record.record_id for record in ordered[cursor : cursor + take])
        cursor += take
    surplus = tuple(record.record_id for record in ordered[cursor:])

    _check_disjoint(partitions, {r.record_id: r.text_sha256 for r in result.retained})
    return MasterCorpus(
        partitions=partitions,
        retained=result.retained,
        dedup=result,
        split_seed=split_seed,
        surplus=surplus,
    )


def _check_disjoint(partitions: Mapping[str, tuple[str, ...]], hashes: Mapping[str, str]) -> None:
    """Partitions must be disjoint by record id AND by normalised text hash [AUTH: 00 §6.2]."""
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    for name, ids in partitions.items():
        for record_id in ids:
            other = seen_ids.setdefault(record_id, name)
            if other != name:
                raise CorpusError(f"record {record_id!r} appears in both {other} and {name}")
            digest = hashes[record_id]
            owner = seen_hashes.setdefault(digest, name)
            if owner != name:
                raise CorpusError(
                    f"normalised text hash {digest[:12]} appears in both {owner} and {name};"
                    " deduplication did not run before the split [AUTH: 00 §6.2]"
                )
