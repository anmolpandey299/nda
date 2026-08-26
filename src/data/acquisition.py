"""The evidentiary research-acquisition path: provenance is DERIVED, never asserted.

The fixture builders in `corpus.py` and `manifest.py` take a caller's word for the query, the
raw-data identity and the preprocessing identity. That is correct for a synthetic fixture and
catastrophic for a research corpus: it lets any generated `MasterCorpus` be relabelled
`RESEARCH_CORPUS` by supplying three strings.

So the two paths are separated here. A research corpus exists only as an `AcquisitionReceipt`,
and a receipt is issued only by `acquire_research_corpus`, which walks the canonical path:

    resolved acquisition config
      -> source records
      -> publication_date >= floor (indexing date never rescues)
      -> normalisation
      -> exact dedup
      -> near dedup
      -> split
      -> derived research manifest

Every field the manifest needs is computed from what actually went through that path. The
role, the query, the raw identity, the publication evidence, the preprocessing identity and
the split identity are outputs of the walk, not inputs to it [AUTH: 01 §8F, §14; 00 §6.1-§6.2].
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from src.analysis.settings import ResolvedSettings
from src.data.corpus import (
    AcquisitionSpec,
    CorpusError,
    MasterCorpus,
    SourceRecord,
    build_master_corpus,
    is_eligible,
)
from src.data.dedup import EXACT_ALGORITHM, NEAR_DUPLICATE_ALGORITHM
from src.data.normalise import NORMALISATION_VERSION, normalise_record
from src.materials import material, material_mapping
from src.provenance.hashing import JSONValue, sha256_canonical

RECEIPT_VERSION: Final = "s05.acquisition-receipt.v1"

#: A corpus that walked the canonical research path.
EVIDENTIARY_RESEARCH: Final = "EVIDENTIARY_RESEARCH"
#: A generated corpus. Permanently non-evidentiary, whatever else it resembles.
NON_EVIDENTIARY_FIXTURE: Final = "NON_EVIDENTIARY_FIXTURE"

#: Query strings that mean "nothing was fetched". A research acquisition may not use one.
NON_QUERIES: Final[frozenset[str]] = frozenset(
    {"", "SYNTHETIC_FIXTURE_NO_QUERY", "UNRESOLVED_NOT_DOWNLOADED", "REQUIRED_NOT_CALIBRATED"}
)

_TIMESTAMP_RE: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)

#: The capability token. A receipt refuses to exist without it, and it is issued in exactly
#: one place, so `AcquisitionReceipt(...)` cannot be hand-built with asserted values.
_ISSUER: Final = object()


class AcquisitionError(CorpusError):
    """A research acquisition cannot be completed, so no receipt is issued."""


@dataclass(frozen=True)
class AcquisitionReceipt:
    """Proof that one corpus walked the canonical research path, and what it produced.

    Construct only through `acquire_research_corpus`. The `issuer` field is a capability
    token; a caller holding an arbitrary `MasterCorpus` cannot mint one.
    """

    issuer: object = field(repr=False, compare=False)
    spec: AcquisitionSpec
    corpus: MasterCorpus
    raw_data_sha256: str
    preprocessing_config_sha256: str
    publication_evidence: Mapping[str, str]
    refused_ids: tuple[str, ...]
    download_timestamp_utc: str
    partition_sizes: Mapping[str, int]
    shingle_size: int
    near_duplicate_threshold: float
    provenance_class: str = EVIDENTIARY_RESEARCH
    receipt_version: str = RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.issuer is not _ISSUER:
            raise AcquisitionError(
                "an AcquisitionReceipt may only be issued by acquire_research_corpus();"
                " research provenance is derived from the acquisition walk and cannot be"
                " asserted by a caller [AUTH: 01 §14]"
            )

    def acquisition_identity(self) -> str:
        """One hash over everything the walk fixed."""
        return sha256_canonical(
            {
                "receipt_version": self.receipt_version,
                "provenance_class": self.provenance_class,
                "query": self.spec.query,
                "publication_date_floor": self.spec.publication_date_floor,
                "raw_data_sha256": self.raw_data_sha256,
                "preprocessing_config_sha256": self.preprocessing_config_sha256,
                "split_seed": self.corpus.split_seed,
                "download_timestamp_utc": self.download_timestamp_utc,
            }
        )


def raw_source_identity(records: Sequence[SourceRecord]) -> str:
    """Hash of the acquired records themselves, in canonical id order.

    Derived from the bytes that were read, so a caller cannot hand a manifest a raw-data
    hash unrelated to the records it describes [AUTH: 01 §14].
    """
    payload: list[JSONValue] = [
        {
            "record_id": record.record_id,
            "pmid": record.pmid,
            "publication_date": record.publication_date,
            "indexing_date": record.indexing_date,
            "source_identity": record.source_identity,
            "fields": dict(sorted(record.fields.items())),
        }
        for record in sorted(records, key=lambda item: item.record_id)
    ]
    return sha256_canonical(payload)


def preprocessing_identity(
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


def research_acquisition_spec(settings: ResolvedSettings) -> AcquisitionSpec:
    """Build the spec from resolved config alone [AUTH: 01 §17].

    Every value is read with `allow_provisional=False`, so a constant that is still
    `REQUIRED_NOT_CALIBRATED` or `PROVISIONAL_FIXTURE_ONLY` refuses the research path here
    rather than being defaulted into a scientific choice.
    """
    return AcquisitionSpec(
        query=str(material(settings, "source_query")),
        publication_date_floor=str(material(settings, "publication_date_floor")),
    )


def acquire_research_corpus(
    *,
    settings: ResolvedSettings,
    source_records: Sequence[SourceRecord],
    download_timestamp_utc: str,
) -> AcquisitionReceipt:
    """Walk the canonical research path and issue a receipt, or refuse.

    Configuration is read here, not passed in: the query, the publication-date floor, the
    partition sizes, the split seed and both dedup parameters come from the resolved corpus
    config, so no caller can substitute one of them for this acquisition.

    `source_records` are raw acquired records, not normalised ones — normalisation happens
    inside the walk, so text that never passed the publication-date gate cannot enter.
    """
    spec = research_acquisition_spec(settings)
    partition_sizes = {
        name: int(str(count))
        for name, count in material_mapping(settings, "partition_sizes").items()
    }
    split_seed = int(str(material(settings, "split_seed")))
    shingle_size = int(str(material(settings, "near_duplicate_shingle_size")))
    near_duplicate_threshold = float(str(material(settings, "near_duplicate_threshold")))

    if spec.query.strip() in NON_QUERIES:
        raise AcquisitionError(
            f"a research acquisition requires the real query; {spec.query!r} means nothing"
            " was fetched [AUTH: 01 §8F, §14]"
        )
    if not _TIMESTAMP_RE.match(download_timestamp_utc):
        raise AcquisitionError("download_timestamp_utc must be an ISO-8601 UTC timestamp")
    if not source_records:
        raise AcquisitionError("no source records were acquired")

    try:
        floor = _dt.date.fromisoformat(spec.publication_date_floor)
    except ValueError as exc:
        raise AcquisitionError(
            f"publication_date_floor {spec.publication_date_floor!r} is not an ISO date"
        ) from exc

    field_problems: list[str] = []
    for record in source_records:
        field_problems.extend(spec.validate_record(record))
    if field_problems:
        raise AcquisitionError(
            f"{len(field_problems)} acquisition field problem(s), first: {field_problems[0]}"
        )

    seen: set[str] = set()
    eligible: list[SourceRecord] = []
    refused: list[str] = []
    for record in source_records:
        if record.record_id in seen:
            raise AcquisitionError(f"the acquisition repeats record id {record.record_id!r}")
        seen.add(record.record_id)
        # is_eligible parses BOTH dates, so a malformed indexing date is still refused, and
        # then decides on publication_date alone [AUTH: 01 §8F].
        if is_eligible(record, floor=spec.publication_date_floor):
            eligible.append(record)
        else:
            refused.append(record.record_id)
    if not eligible:
        raise AcquisitionError(
            f"no acquired record satisfies publication_date >= {floor.isoformat()}"
        )

    normalised = [normalise_record(record.record_id, record.fields) for record in eligible]
    corpus = build_master_corpus(
        normalised,
        sizes=partition_sizes,
        split_seed=split_seed,
        shingle_size=shingle_size,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    return AcquisitionReceipt(
        issuer=_ISSUER,
        spec=spec,
        corpus=corpus,
        raw_data_sha256=raw_source_identity(source_records),
        preprocessing_config_sha256=preprocessing_identity(
            spec,
            shingle_size=shingle_size,
            near_duplicate_threshold=near_duplicate_threshold,
            sizes=partition_sizes,
        ),
        publication_evidence={
            record.record_id: record.publication_date
            for record in eligible
            if record.record_id in {r.record_id for r in corpus.retained}
        },
        refused_ids=tuple(refused),
        download_timestamp_utc=download_timestamp_utc,
        partition_sizes=dict(partition_sizes),
        shingle_size=shingle_size,
        near_duplicate_threshold=near_duplicate_threshold,
    )
