"""S05.1/S05.2 — one normalisation, and deduplication that cannot be decided by input order."""

from __future__ import annotations

import pytest
from blockc_fixtures import source_records, tiny_corpus

from src.data.dedup import (
    EXACT_ALGORITHM,
    NEAR_DUPLICATE_ALGORITHM,
    DedupError,
    deduplicate,
    jaccard,
    shingles,
)
from src.data.normalise import (
    NORMALISATION_VERSION,
    NormalisationError,
    normalise_record,
    normalise_text,
)


# ------------------------------------------------------------------ normalisation
def test_normalisation_is_idempotent() -> None:
    for record in tiny_corpus(25):
        assert normalise_text(record.text) == record.text


@pytest.mark.parametrize(
    "raw",
    [
        "Alpha beta",
        "Alpha  beta",
        "Alpha\tbeta",
        "Alpha beta   ",
        "\ufeffAlpha beta",
        "Alpha beta\r\n",
    ],
)
def test_cosmetic_variants_collapse_to_one_text(raw: str) -> None:
    assert normalise_text(raw) == "Alpha beta"


def test_nfkc_folds_compatibility_forms() -> None:
    assert normalise_text("ﬁbrosis") == "fibrosis"
    assert normalise_text("Ⅳ stage") == "IV stage"


def test_zero_width_characters_are_removed_not_spaced() -> None:
    assert normalise_text("Alpha\u200bbeta") == "Alphabeta"


def test_newline_runs_collapse_to_one_newline() -> None:
    """A blank line is not part of a record's identity; a single line break is."""
    assert normalise_text("a\n\n\n\nb") == "a\nb"
    assert normalise_text("a\nb") == "a\nb"


def test_an_empty_record_fails_closed() -> None:
    with pytest.raises(NormalisationError):
        normalise_record("R1", {"title": "  ", "abstract": "​"})


def test_field_order_is_part_of_the_identity() -> None:
    first = normalise_record("R1", {"title": "A", "abstract": "B"})
    second = normalise_record("R1", {"title": "B", "abstract": "A"})
    assert first.text_sha256 != second.text_sha256
    assert first.normalisation_version == NORMALISATION_VERSION


def test_identity_is_the_hash_of_the_normalised_text() -> None:
    import hashlib

    record = normalise_record("R1", {"title": "A  b", "abstract": "c"})
    assert record.text_sha256 == hashlib.sha256(record.text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ deduplication
def test_exact_duplicates_are_removed_by_normalised_hash() -> None:
    records = tiny_corpus(10)
    twins = [
        normalise_record(f"TWIN{i:03d}", {"title": r.text, "abstract": ""})
        for i, r in enumerate(records)
    ]
    result = deduplicate([*records, *twins], shingle_size=5, threshold=0.8)
    assert len(result.retained) == 10
    assert {d.algorithm for d in result.decisions} == {EXACT_ALGORITHM}


def test_a_cosmetic_variant_is_an_exact_duplicate_after_normalisation() -> None:
    original = normalise_record("R1", {"title": "Hepatic perfusion", "abstract": "Body text."})
    variant = normalise_record(
        "R2", {"title": "Hepatic  perfusion", "abstract": "Body   text.\r\n"}
    )
    result = deduplicate([original, variant], shingle_size=5, threshold=0.8)
    assert len(result.retained) == 1
    assert result.decisions[0].algorithm == EXACT_ALGORITHM


def test_input_order_cannot_decide_which_duplicate_survives() -> None:
    records = tiny_corpus(12)
    duplicate = normalise_record("ZZZ999", {"title": "x", "abstract": records[3].text})
    candidates = [*records, duplicate]
    forward = deduplicate(candidates, shingle_size=5, threshold=0.8)
    backward = deduplicate(list(reversed(candidates)), shingle_size=5, threshold=0.8)
    assert [r.record_id for r in forward.retained] == [r.record_id for r in backward.retained]


def test_near_duplicates_are_removed_before_any_split_can_see_them() -> None:
    base = tiny_corpus(1)[0]
    near = normalise_record("NEAR1", {"title": base.text, "abstract": "One extra clause."})
    result = deduplicate([base, near], shingle_size=5, threshold=0.6)
    assert len(result.retained) == 1
    assert result.decisions[0].algorithm == NEAR_DUPLICATE_ALGORITHM
    assert result.decisions[0].similarity is not None


def test_the_near_duplicate_threshold_actually_controls_the_outcome() -> None:
    base = tiny_corpus(1)[0]
    near = normalise_record("NEAR1", {"title": base.text, "abstract": "One extra clause."})
    loose = deduplicate([base, near], shingle_size=5, threshold=0.4)
    strict = deduplicate([base, near], shingle_size=5, threshold=0.99)
    assert len(loose.retained) == 1
    assert len(strict.retained) == 2


def test_shingle_size_controls_the_similarity_measure() -> None:
    text = "abcdefgh"
    assert shingles(text, 3) != shingles(text, 5)
    assert jaccard(shingles("abcdef", 3), shingles("abcdef", 3)) == 1.0
    assert jaccard(shingles("abcdef", 3), shingles("uvwxyz", 3)) == 0.0


def test_a_corpus_that_repeats_a_record_id_is_refused() -> None:
    records = tiny_corpus(4)
    with pytest.raises(DedupError, match="repeats record id"):
        deduplicate([*records, records[0]], shingle_size=5, threshold=0.8)


def test_every_dropped_record_has_a_recorded_reason() -> None:
    records = tiny_corpus(8)
    twins = [
        normalise_record(f"TWIN{i:03d}", {"title": r.text, "abstract": ""})
        for i, r in enumerate(records)
    ]
    result = deduplicate([*records, *twins], shingle_size=5, threshold=0.8)
    dropped = {d.dropped_id for d in result.decisions}
    retained = {r.record_id for r in result.retained}
    assert dropped and not dropped & retained
    for decision in result.decisions:
        assert decision.retained_id in retained
        assert decision.config


def test_deduplication_is_deterministic() -> None:
    records = tiny_corpus(30)
    first = deduplicate(records, shingle_size=5, threshold=0.8)
    second = deduplicate(list(reversed(records)), shingle_size=5, threshold=0.8)
    assert [r.record_id for r in first.retained] == [r.record_id for r in second.retained]
    assert first.decisions == second.decisions


def test_fixture_source_records_are_all_distinct() -> None:
    records = source_records(200)
    assert len({r.fields["abstract"] for r in records}) == 200
