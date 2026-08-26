"""S05.3 — eligibility, dedup-before-split, disjointness, and failing closed on scarcity."""

from __future__ import annotations

import pytest
from blockc_fixtures import normalised, source_records, tiny_corpus

from src.data.corpus import (
    PARTITION_ORDER,
    AcquisitionSpec,
    CorpusError,
    build_master_corpus,
    filter_eligible,
    is_eligible,
)
from src.data.normalise import normalise_record

FLOOR = "2026-04-01"
SIZES = {
    "TRAIN_CANDIDATES": 30,
    "CALIBRATION_NONMEMBERS": 10,
    "EVAL_NONMEMBERS": 10,
    "RESERVE": 10,
}


# ------------------------------------------------------------------ 01 §8F eligibility
def test_publication_date_decides_eligibility() -> None:
    late = source_records(1, publication_date="2026-05-14")[0]
    early = source_records(1, start=1, publication_date="2026-03-31")[0]
    assert is_eligible(late, floor=FLOOR)
    assert not is_eligible(early, floor=FLOOR)


def test_the_indexing_date_cannot_rescue_an_early_publication_date() -> None:
    """01 §8F: 'indexing date is not accepted as a substitute'."""
    record = source_records(1, publication_date="2026-01-05", indexing_date="2026-09-30")[0]
    assert not is_eligible(record, floor=FLOOR)


def test_a_late_indexing_date_cannot_disqualify_an_eligible_record() -> None:
    record = source_records(1, publication_date="2026-06-01", indexing_date="2026-02-01")[0]
    assert is_eligible(record, floor=FLOOR)


def test_the_boundary_date_is_inclusive() -> None:
    assert is_eligible(source_records(1, publication_date=FLOOR)[0], floor=FLOOR)


def test_a_malformed_date_fails_closed() -> None:
    record = source_records(1, publication_date="not-a-date")[0]
    with pytest.raises(CorpusError):
        is_eligible(record, floor=FLOOR)


def test_filter_eligible_partitions_records_and_reasons() -> None:
    records = [
        *source_records(4, publication_date="2026-05-14"),
        *source_records(3, start=100, publication_date="2026-01-14"),
    ]
    eligible, refused = filter_eligible(records, floor=FLOOR)
    assert [r.record_id for r in eligible] == [f"FIX{i:06d}" for i in range(4)]
    assert len(refused) == 3


def test_required_acquisition_fields_are_enforced() -> None:
    spec = AcquisitionSpec(query="fixture", publication_date_floor=FLOOR)
    good = source_records(1)[0]
    assert spec.validate_record(good) == []
    bad = source_records(1)[0].__class__(
        record_id="X",
        pmid="",
        publication_date="2026-05-01",
        indexing_date="2026-05-02",
        source_identity="s",
        fields={"title": "t"},
    )
    assert spec.validate_record(bad)


# ------------------------------------------------------------------ 00 §6.2 construction
def test_partitions_have_the_configured_sizes_and_order() -> None:
    corpus = build_master_corpus(
        tiny_corpus(120),
        sizes=SIZES,
        split_seed=20260820,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    assert tuple(corpus.partitions) == PARTITION_ORDER
    assert {name: len(ids) for name, ids in corpus.partitions.items()} == SIZES


def test_partitions_are_disjoint_by_record_id_and_by_text_hash() -> None:
    corpus = build_master_corpus(
        tiny_corpus(120),
        sizes=SIZES,
        split_seed=20260820,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    every = [rid for ids in corpus.partitions.values() for rid in ids]
    assert len(set(every)) == len(every)
    by_id = {r.record_id: r.text_sha256 for r in corpus.retained}
    hashes = [by_id[rid] for rid in every]
    assert len(set(hashes)) == len(hashes)


def test_a_duplicate_cannot_land_in_train_and_eval_because_dedup_runs_first() -> None:
    """The mutation this kills: splitting before deduplicating."""
    base = tiny_corpus(120)
    twins = [
        normalise_record(f"TWIN{index:04d}", {"title": record.text, "abstract": ""})
        for index, record in enumerate(base[:40])
    ]
    corpus = build_master_corpus(
        [*base, *twins],
        sizes=SIZES,
        split_seed=20260820,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    by_id = {r.record_id: r.text_sha256 for r in corpus.retained}
    placed = [by_id[rid] for ids in corpus.partitions.values() for rid in ids]
    assert len(set(placed)) == len(placed)
    assert corpus.dedup.decisions, "the twins were never detected"


def test_the_split_seed_decides_the_assignment() -> None:
    records = tiny_corpus(120)
    first = build_master_corpus(
        records, sizes=SIZES, split_seed=20260820, shingle_size=5, near_duplicate_threshold=0.8
    )
    second = build_master_corpus(
        records, sizes=SIZES, split_seed=20260821, shingle_size=5, near_duplicate_threshold=0.8
    )
    assert first.partitions != second.partitions


def test_the_split_is_reproducible_and_order_independent() -> None:
    records = tiny_corpus(120)
    first = build_master_corpus(
        records, sizes=SIZES, split_seed=20260820, shingle_size=5, near_duplicate_threshold=0.8
    )
    shuffled = [records[i] for i in (*range(60, 120), *range(60))]
    second = build_master_corpus(
        shuffled, sizes=SIZES, split_seed=20260820, shingle_size=5, near_duplicate_threshold=0.8
    )
    assert first.partitions == second.partitions


def test_too_few_eligible_records_fails_closed_rather_than_shrinking_a_partition() -> None:
    with pytest.raises(CorpusError, match="refusing to build undersized"):
        build_master_corpus(
            tiny_corpus(40),
            sizes=SIZES,
            split_seed=20260820,
            shingle_size=5,
            near_duplicate_threshold=0.8,
        )


def test_surplus_records_are_reported_not_silently_discarded() -> None:
    corpus = build_master_corpus(
        tiny_corpus(150),
        sizes=SIZES,
        split_seed=20260820,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    placed = {rid for ids in corpus.partitions.values() for rid in ids}
    assert len(corpus.surplus) == len(corpus.retained) - len(placed)
    assert not set(corpus.surplus) & placed


def test_an_unknown_partition_name_is_refused() -> None:
    with pytest.raises(CorpusError):
        build_master_corpus(
            tiny_corpus(120),
            sizes={**SIZES, "SOMETHING_ELSE": 3},
            split_seed=1,
            shingle_size=5,
            near_duplicate_threshold=0.8,
        )


def test_eligible_records_survive_normalisation_unchanged() -> None:
    records = source_records(10)
    assert len(normalised(records)) == 10
