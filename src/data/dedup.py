"""Exact and near-duplicate detection, run BEFORE split assignment [AUTH: 00 §6.2].

Order matters: deduplicating after the split would leave a record's twin on the other side of
the member/non-member boundary, which is the exact leakage the blind control in 00 §6.4 later
looks for. So the corpus is reduced first and split second.

Near-duplicate detection is a frozen, deterministic MinHash-free construction: character
5-gram shingles over the normalised text, compared by exact Jaccard similarity. It is O(n^2)
in the corpus size for the fixture scale used here, and the production path can swap in a
banded index behind the same interface without changing the decision rule. The threshold
comes from config and is never chosen by looking at an outcome [AUTH: 00 §34B.1A; 01 §3.2].

Every decision is auditable: the retained record, the dropped record, the similarity and the
algorithm identity are all recorded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from src.data.normalise import NormalisedRecord

#: Bumped whenever the shingling or the comparison changes.
NEAR_DUPLICATE_ALGORITHM: Final = "s05.jaccard-char-shingle.v1"
EXACT_ALGORITHM: Final = "s05.normalised-sha256.v1"

DecisionKind = Literal["EXACT_DUPLICATE", "NEAR_DUPLICATE"]


class DedupError(ValueError):
    """Deduplication cannot be performed as configured."""


@dataclass(frozen=True)
class DuplicateDecision:
    """One dropped record and the retained record it duplicates."""

    kind: DecisionKind
    retained_id: str
    dropped_id: str
    similarity: float
    algorithm: str
    config: str

    def as_row(self) -> dict[str, str | float]:
        return {
            "kind": self.kind,
            "retained_id": self.retained_id,
            "dropped_id": self.dropped_id,
            "similarity": self.similarity,
            "algorithm": self.algorithm,
            "config": self.config,
        }


@dataclass(frozen=True)
class DedupResult:
    retained: tuple[NormalisedRecord, ...]
    decisions: tuple[DuplicateDecision, ...]

    @property
    def retained_ids(self) -> tuple[str, ...]:
        return tuple(record.record_id for record in self.retained)

    @property
    def dropped_ids(self) -> tuple[str, ...]:
        return tuple(decision.dropped_id for decision in self.decisions)


def shingles(text: str, size: int) -> frozenset[str]:
    """Character n-gram shingles. Short texts yield the whole string as one shingle."""
    if size < 1:
        raise DedupError("shingle size must be positive")
    if len(text) <= size:
        return frozenset({text})
    return frozenset(text[index : index + size] for index in range(len(text) - size + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return 0.0 if union == 0 else len(left & right) / union


def _canonical_order(records: Sequence[NormalisedRecord]) -> list[NormalisedRecord]:
    """Sort by (text hash, record id).

    Input order must not decide which twin survives, or the same logical corpus loaded in a
    different order would produce different retained ids and therefore different splits.
    """
    return sorted(records, key=lambda record: (record.text_sha256, record.record_id))


def deduplicate(
    records: Sequence[NormalisedRecord], *, shingle_size: int, threshold: float
) -> DedupResult:
    """Exact-then-near deduplication over a canonically ordered corpus.

    The guaranteed post-condition is pairwise, not component-wise: **no two retained records
    are at or above the threshold**, because a record is dropped if it is a near-duplicate of
    any already-retained record. A chain whose every consecutive pair exceeds the threshold
    therefore collapses to one survivor, while a chain whose ends fall below it keeps both
    ends — which is the right outcome, since only an above-threshold pair split across
    partitions is the leakage 00 §6.4 later looks for. Component-wise collapse would delete
    records that are not duplicates of anything retained.
    """
    if not 0.0 < threshold <= 1.0:
        raise DedupError(f"near-duplicate threshold {threshold!r} is not in (0, 1]")
    config = f"shingle_size={shingle_size};threshold={threshold}"
    ids = [record.record_id for record in records]
    if len(set(ids)) != len(ids):
        repeated = sorted({name for name in ids if ids.count(name) > 1})
        raise DedupError(
            f"the corpus repeats record id(s) {repeated[:3]}; two records sharing an id cannot"
            " be told apart in a split, a manifest or a decision log [AUTH: 01 §14]"
        )
    ordered = _canonical_order(records)

    decisions: list[DuplicateDecision] = []
    by_hash: dict[str, NormalisedRecord] = {}
    unique: list[NormalisedRecord] = []
    for record in ordered:
        first = by_hash.get(record.text_sha256)
        if first is not None:
            decisions.append(
                DuplicateDecision(
                    kind="EXACT_DUPLICATE",
                    retained_id=first.record_id,
                    dropped_id=record.record_id,
                    similarity=1.0,
                    algorithm=EXACT_ALGORITHM,
                    config="normalised text sha256",
                )
            )
            continue
        by_hash[record.text_sha256] = record
        unique.append(record)

    retained: list[NormalisedRecord] = []
    retained_shingles: list[frozenset[str]] = []
    for record in unique:
        current = shingles(record.text, shingle_size)
        duplicate_of: tuple[str, float] | None = None
        for kept, kept_shingles in zip(retained, retained_shingles, strict=True):
            similarity = jaccard(current, kept_shingles)
            if similarity >= threshold:
                duplicate_of = (kept.record_id, similarity)
                break
        if duplicate_of is None:
            retained.append(record)
            retained_shingles.append(current)
            continue
        decisions.append(
            DuplicateDecision(
                kind="NEAR_DUPLICATE",
                retained_id=duplicate_of[0],
                dropped_id=record.record_id,
                similarity=float(duplicate_of[1]),
                algorithm=NEAR_DUPLICATE_ALGORITHM,
                config=config,
            )
        )
    return DedupResult(retained=tuple(retained), decisions=tuple(decisions))
