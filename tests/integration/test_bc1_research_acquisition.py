"""B-C1 — research corpus provenance is DERIVED from the acquisition walk, never asserted.

The exploit these tests close: an arbitrary fixture `MasterCorpus` relabelled
`RESEARCH_CORPUS` by supplying a caller-chosen query, raw hash and preprocessing hash, over
records that never passed the publication-date gate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from blockc_fixtures import source_records, tiny_corpus

from src.data.acquisition import (
    EVIDENTIARY_RESEARCH,
    NON_EVIDENTIARY_FIXTURE,
    AcquisitionError,
    AcquisitionReceipt,
    acquire_research_corpus,
    raw_source_identity,
    research_acquisition_spec,
)
from src.data.corpus import AcquisitionSpec, build_master_corpus
from src.data.manifest import (
    FIXTURE_CORPUS_ROLE,
    RESEARCH_CORPUS_ROLE,
    CorpusManifestError,
    corpus_role,
    fixture_corpus_manifest,
    require_research_corpus,
    research_corpus_manifest,
)
from src.data.normalise import normalise_record
from src.data.settings import corpus_settings
from src.materials import UncalibratedConstantError
from src.provenance.data_manifest import require_valid_data_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP = "2026-08-25T00:00:00Z"
COMMIT = "0" * 40

FROZEN_QUERY = "(fixture[Title/Abstract]) AND 2026/04/01:3000[dp]"
SIZES = {
    "TRAIN_CANDIDATES": 40,
    "CALIBRATION_NONMEMBERS": 15,
    "EVAL_NONMEMBERS": 15,
    "RESERVE": 10,
}


def frozen_config_root(tmp_path: Path, **overrides: Any) -> Path:
    """A config tree where the acquisition constants are frozen, so the walk can be tested.

    The repository's own `configs/data/corpus.json` deliberately leaves `source_query` and the
    dedup parameters uncalibrated, which is exactly why the real research path refuses today.
    """
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    path = root / "configs" / "data" / "corpus.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["source_query"] = {"status": "FROZEN", "value": FROZEN_QUERY, "note": "test"}
    document["near_duplicate_shingle_size"]["status"] = "FROZEN"
    document["near_duplicate_threshold"]["status"] = "FROZEN"
    document["partition_sizes"] = {"status": "FROZEN", "value": SIZES, "note": "test"}
    for key, value in overrides.items():
        document[key] = value
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return root


def acquire(root: Path, records: Any = None) -> AcquisitionReceipt:
    return acquire_research_corpus(
        settings=corpus_settings(root),
        source_records=records
        if records is not None
        else source_records(120, publication_date="2026-05-14"),
        download_timestamp_utc=STAMP,
    )


# ------------------------------------------------------------------ the real path refuses
def test_the_repository_config_refuses_a_research_acquisition_today() -> None:
    """`source_query` is REQUIRED_NOT_CALIBRATED, so no real acquisition can be walked."""
    with pytest.raises(UncalibratedConstantError, match="source_query"):
        research_acquisition_spec(corpus_settings(REPO_ROOT))
    with pytest.raises(UncalibratedConstantError):
        acquire_research_corpus(
            settings=corpus_settings(REPO_ROOT),
            source_records=source_records(10),
            download_timestamp_utc=STAMP,
        )


# ------------------------------------------------------------------ the forged manifest
def test_a_fixture_corpus_cannot_be_relabelled_research(tmp_path: Path) -> None:
    """The Codex counterexample, as an executable regression."""
    corpus = build_master_corpus(
        tiny_corpus(120),
        sizes=SIZES,
        split_seed=20260820,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    # 1. the fixture builder has no `role` parameter to raise.
    with pytest.raises(TypeError):
        fixture_corpus_manifest(  # type: ignore[call-arg]
            corpus,
            role=RESEARCH_CORPUS_ROLE,
            spec=AcquisitionSpec(query=FROZEN_QUERY, publication_date_floor="2026-04-01"),
            raw_data_sha256="a" * 64,
            preprocessing_code_commit=COMMIT,
            preprocessing_config_sha256="b" * 64,
            download_timestamp_utc=STAMP,
            dataset_alias="pubmed_2026",
        )
    # 2. whatever it is given, the manifest it produces is a fixture manifest.
    document = fixture_corpus_manifest(
        corpus,
        spec=AcquisitionSpec(query=FROZEN_QUERY, publication_date_floor="2026-04-01"),
        raw_data_sha256="a" * 64,
        preprocessing_code_commit=COMMIT,
        preprocessing_config_sha256="b" * 64,
        download_timestamp_utc=STAMP,
        dataset_alias="fixture_pubmed",
    )
    assert corpus_role(document) == FIXTURE_CORPUS_ROLE
    assert document["source_query"] == "SYNTHETIC_FIXTURE_NO_QUERY"
    with pytest.raises(CorpusManifestError, match="may not attribute"):
        require_research_corpus(document)


def test_a_receipt_cannot_be_hand_constructed(tmp_path: Path) -> None:
    """Research provenance needs the capability token, which only the walk issues."""
    corpus = build_master_corpus(
        tiny_corpus(120),
        sizes=SIZES,
        split_seed=20260820,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    with pytest.raises(AcquisitionError, match="only be issued by"):
        AcquisitionReceipt(
            issuer=object(),
            spec=AcquisitionSpec(query=FROZEN_QUERY, publication_date_floor="2026-04-01"),
            corpus=corpus,
            raw_data_sha256="a" * 64,
            preprocessing_config_sha256="b" * 64,
            publication_evidence={},
            refused_ids=(),
            download_timestamp_utc=STAMP,
            partition_sizes=SIZES,
            shingle_size=5,
            near_duplicate_threshold=0.8,
        )


def test_the_research_manifest_takes_only_a_receipt(tmp_path: Path) -> None:
    import inspect

    signature = inspect.signature(research_corpus_manifest)
    assert set(signature.parameters) == {"receipt", "preprocessing_code_commit", "dataset_alias"}


# ------------------------------------------------------------------ the walk
def test_the_walk_produces_a_valid_research_manifest(tmp_path: Path) -> None:
    receipt = acquire(frozen_config_root(tmp_path))
    assert receipt.provenance_class == EVIDENTIARY_RESEARCH
    document = research_corpus_manifest(
        receipt, preprocessing_code_commit=COMMIT, dataset_alias="pubmed_2026_core"
    )
    require_valid_data_manifest(document)
    require_research_corpus(document)
    assert corpus_role(document) == RESEARCH_CORPUS_ROLE
    assert document["source_query"] == FROZEN_QUERY
    assert document["split_rng_seed"] == 20260820


def test_the_raw_data_hash_is_derived_from_the_source_records(tmp_path: Path) -> None:
    """A caller cannot supply a raw hash inconsistent with the records it describes."""
    records = source_records(120, publication_date="2026-05-14")
    receipt = acquire(frozen_config_root(tmp_path), records)
    assert receipt.raw_data_sha256 == raw_source_identity(records)

    other = source_records(120, start=5000, publication_date="2026-05-14")
    assert raw_source_identity(other) != receipt.raw_data_sha256

    document = research_corpus_manifest(
        receipt, preprocessing_code_commit=COMMIT, dataset_alias="pubmed_2026_core"
    )
    assert document["raw_data_sha256"] == raw_source_identity(records)


def test_the_preprocessing_identity_is_derived_from_the_resolved_config(tmp_path: Path) -> None:
    first = acquire(frozen_config_root(tmp_path / "a"))
    changed = frozen_config_root(
        tmp_path / "b",
        near_duplicate_threshold={"status": "FROZEN", "value": 0.95, "note": "test"},
    )
    second = acquire(changed)
    assert first.preprocessing_config_sha256 != second.preprocessing_config_sha256


def test_a_research_acquisition_without_a_query_fails(tmp_path: Path) -> None:
    root = frozen_config_root(
        tmp_path,
        source_query={"status": "FROZEN", "value": "SYNTHETIC_FIXTURE_NO_QUERY", "note": "t"},
    )
    with pytest.raises(AcquisitionError, match="requires the real query"):
        acquire(root)


# ------------------------------------------------------------------ publication-date gate
def test_pre_floor_records_are_excluded_from_the_research_corpus(tmp_path: Path) -> None:
    root = frozen_config_root(tmp_path)
    records = [
        *source_records(120, publication_date="2026-05-14"),
        *source_records(20, start=900, publication_date="2026-01-05"),
    ]
    receipt = acquire(root, records)
    assert len(receipt.refused_ids) == 20
    placed = {rid for ids in receipt.corpus.partitions.values() for rid in ids}
    assert not placed & set(receipt.refused_ids)
    for record_id in placed:
        assert receipt.publication_evidence[record_id] >= "2026-04-01"


def test_a_late_indexing_date_cannot_rescue_an_early_publication(tmp_path: Path) -> None:
    root = frozen_config_root(tmp_path)
    records = [
        *source_records(120, publication_date="2026-05-14"),
        *source_records(20, start=900, publication_date="2026-01-05", indexing_date="2026-12-31"),
    ]
    receipt = acquire(root, records)
    assert len(receipt.refused_ids) == 20


def test_a_malformed_publication_date_fails_the_walk(tmp_path: Path) -> None:
    root = frozen_config_root(tmp_path)
    records = [
        *source_records(120, publication_date="2026-05-14"),
        *source_records(1, start=900, publication_date="not-a-date"),
    ]
    with pytest.raises(Exception, match="not an ISO date"):
        acquire(root, records)


def test_a_malformed_indexing_date_still_fails_the_walk(tmp_path: Path) -> None:
    """The indexing date decides nothing, but a malformed one is still a corrupt record."""
    root = frozen_config_root(tmp_path)
    records = [
        *source_records(120, publication_date="2026-05-14"),
        *source_records(1, start=900, indexing_date="soon"),
    ]
    with pytest.raises(Exception, match="not an ISO date"):
        acquire(root, records)


def test_a_missing_acquisition_field_fails_the_walk(tmp_path: Path) -> None:
    root = frozen_config_root(tmp_path)
    records = list(source_records(120, publication_date="2026-05-14"))
    broken = records[0]
    records[0] = type(broken)(
        record_id=broken.record_id,
        pmid="",
        publication_date=broken.publication_date,
        indexing_date=broken.indexing_date,
        source_identity=broken.source_identity,
        fields=broken.fields,
    )
    with pytest.raises(AcquisitionError, match="acquisition field"):
        acquire(root, records)


def test_no_eligible_record_fails_closed(tmp_path: Path) -> None:
    root = frozen_config_root(tmp_path)
    with pytest.raises(AcquisitionError, match="no acquired record satisfies"):
        acquire(root, source_records(20, publication_date="2025-01-01"))


def test_a_repeated_source_record_id_fails_the_walk(tmp_path: Path) -> None:
    root = frozen_config_root(tmp_path)
    records = list(source_records(120, publication_date="2026-05-14"))
    with pytest.raises(AcquisitionError, match="repeats record id"):
        acquire(root, [*records, records[0]])


def test_the_walk_normalises_its_own_records(tmp_path: Path) -> None:
    """Already-normalised text cannot be injected: the walk takes raw SourceRecords."""
    import inspect

    signature = inspect.signature(acquire_research_corpus)
    assert set(signature.parameters) == {
        "settings",
        "source_records",
        "download_timestamp_utc",
    }
    root = frozen_config_root(tmp_path)
    normalised = normalise_record("X1", {"title": "t", "abstract": "a"})
    with pytest.raises(AcquisitionError, match="acquisition field"):
        acquire_research_corpus(
            settings=corpus_settings(root),
            source_records=[normalised],  # type: ignore[list-item]
            download_timestamp_utc=STAMP,
        )


def test_a_malformed_download_timestamp_fails_the_walk(tmp_path: Path) -> None:
    root = frozen_config_root(tmp_path)
    with pytest.raises(AcquisitionError, match="ISO-8601"):
        acquire_research_corpus(
            settings=corpus_settings(root),
            source_records=source_records(120, publication_date="2026-05-14"),
            download_timestamp_utc="yesterday",
        )


def test_the_acquisition_identity_moves_with_every_input(tmp_path: Path) -> None:
    first = acquire(frozen_config_root(tmp_path / "a"))
    second = acquire(
        frozen_config_root(tmp_path / "b"),
        source_records(120, start=7000, publication_date="2026-05-14"),
    )
    assert first.acquisition_identity() != second.acquisition_identity()
    assert (
        first.acquisition_identity()
        == acquire(frozen_config_root(tmp_path / "c")).acquisition_identity()
    )


def test_the_fixture_provenance_class_is_permanently_non_evidentiary() -> None:
    assert len({NON_EVIDENTIARY_FIXTURE, EVIDENTIARY_RESEARCH}) == 2
    assert (
        corpus_role(
            fixture_corpus_manifest(
                build_master_corpus(
                    tiny_corpus(120),
                    sizes=SIZES,
                    split_seed=20260820,
                    shingle_size=5,
                    near_duplicate_threshold=0.8,
                ),
                spec=AcquisitionSpec(query="x", publication_date_floor="2026-04-01"),
                raw_data_sha256="a" * 64,
                preprocessing_code_commit=COMMIT,
                preprocessing_config_sha256="b" * 64,
                download_timestamp_utc=STAMP,
                dataset_alias="fixture_x",
            )
        )
        == FIXTURE_CORPUS_ROLE
    )
