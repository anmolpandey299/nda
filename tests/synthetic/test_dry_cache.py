"""DRY-CACHE — content-addressed cache correctness through the real engine [AUTH: 00 §34B.1]."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.provenance.hashing import JSONValue, sha256_canonical
from src.scoring.backends import ConstantBackend, SyntheticBackend
from src.scoring.cache import ScoreCache, scoring_code_hash
from src.scoring.engine import Arm, ScoreFamily, ScoringContext, View, score_records

RECORDS = [f"mem-{i:03d}" for i in range(20)] + [f"non-{i:03d}" for i in range(20)]
MEMBERSHIP = {r: (1 if r.startswith("mem-") else 0) for r in RECORDS}
#: A resolved scoring snapshot. precision and max_sequence_length live here now, because the
#: context derives them from the snapshot rather than accepting them beside it.
CONFIG: dict[str, JSONValue] = {
    "min_k_fraction": 0.2,
    "target_fpr": 0.01,
    "precision": "float64",
    "max_sequence_length": 128,
}


def _context(repo_root: Path, **overrides: object) -> ScoringContext:
    base = {
        "root": repo_root,
        "backend": SyntheticBackend(label="dry-cache", tokens_per_record=16, member_shift=0.8),
        "reference_backend": ConstantBackend(label="ref", tokens_per_record=16),
        "resolved_config": CONFIG,
        "model_revision": "a" * 40,
        "tokenizer_sha256": "1" * 64,
        "environment_lock_sha256": "6" * 64,
    }
    base.update(overrides)
    return ScoringContext(**base)  # type: ignore[arg-type]


def _score(context: ScoringContext, cache: ScoreCache, artifact: str = "artifact-A"):  # type: ignore[no-untyped-def]
    return score_records(
        arm=Arm.CANARY,
        score=ScoreFamily.REFERENCE_LOSS,
        view=View.SYNTHETIC,
        artifact=artifact,
        seed=101,
        fold=0,
        record_ids=RECORDS,
        membership=MEMBERSHIP,
        context=context,
        cache=cache,
    )


# ------------------------------------------------------------------ stable hit
def test_stable_hit_returns_bitwise_identical_scores(repo_root: Path, tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    context = _context(repo_root)
    first = _score(context, cache)
    second = _score(context, cache)

    assert first.cache_hit is False and second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert [row.scalar_score for row in first.rows] == [row.scalar_score for row in second.rows]


def test_the_row_schema_is_the_frozen_one(repo_root: Path, tmp_path: Path) -> None:
    from src.scoring.engine import SCORE_ROW_FIELDS

    table = _score(_context(repo_root), ScoreCache(tmp_path))
    for row in table.rows:
        assert set(SCORE_ROW_FIELDS) <= set(row.as_dict())
    assert len(table) == len(RECORDS)


# ------------------------------------------------------------------ version invalidation
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "b" * 40),
        ("tokenizer_sha256", "9" * 64),
        ("environment_lock_sha256", "7" * 64),
    ],
)
def test_a_changed_identity_component_misses(
    repo_root: Path, tmp_path: Path, field: str, value: object
) -> None:
    cache = ScoreCache(tmp_path)
    context = _context(repo_root)
    first = _score(context, cache)
    mutated = _score(dataclasses.replace(context, **{field: value}), cache)  # type: ignore[arg-type]
    assert mutated.cache_key != first.cache_key
    assert mutated.cache_hit is False


@pytest.mark.parametrize(
    ("key", "value"), [("precision", "bfloat16"), ("max_sequence_length", 256)]
)
def test_a_changed_snapshot_runtime_field_misses(
    repo_root: Path, tmp_path: Path, key: str, value: object
) -> None:
    """precision and max_sequence_length now move with the config snapshot."""
    cache = ScoreCache(tmp_path)
    first = _score(_context(repo_root), cache)
    mutated = _score(_context(repo_root, resolved_config={**CONFIG, key: value}), cache)
    assert mutated.cache_key != first.cache_key
    assert mutated.cache_hit is False


def test_a_changed_record_set_misses(repo_root: Path, tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    context = _context(repo_root)
    _score(context, cache)
    fewer = score_records(
        arm=Arm.CANARY,
        score=ScoreFamily.REFERENCE_LOSS,
        view=View.SYNTHETIC,
        artifact="artifact-A",
        seed=101,
        fold=0,
        record_ids=RECORDS[:-1],
        membership=MEMBERSHIP,
        context=context,
        cache=cache,
    )
    assert fewer.cache_hit is False


def test_record_order_is_part_of_the_identity(repo_root: Path, tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    context = _context(repo_root)
    first = _score(context, cache)
    reordered = score_records(
        arm=Arm.CANARY,
        score=ScoreFamily.REFERENCE_LOSS,
        view=View.SYNTHETIC,
        artifact="artifact-A",
        seed=101,
        fold=0,
        record_ids=list(reversed(RECORDS)),
        membership=MEMBERSHIP,
        context=context,
        cache=cache,
    )
    assert reordered.cache_key != first.cache_key


def test_a_changed_artifact_misses(repo_root: Path, tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    context = _context(repo_root)
    first = _score(context, cache)
    other = _score(context, cache, artifact="artifact-B")
    assert other.cache_key != first.cache_key and other.cache_hit is False


def test_a_changed_backend_misses(repo_root: Path, tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    context = _context(repo_root)
    first = _score(context, cache)
    swapped = dataclasses.replace(
        context,
        backend=SyntheticBackend(label="other", tokens_per_record=16, member_shift=0.8),
    )
    assert _score(swapped, cache).cache_key != first.cache_key


def test_a_changed_scoring_config_misses(repo_root: Path, tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    context = _context(repo_root)
    first = _score(context, cache)
    other_config = {**CONFIG, "min_k_fraction": 0.3}
    swapped = dataclasses.replace(context, resolved_config=other_config)
    # the identity is derived, so it moved with the body and nothing had to be declared
    assert swapped.resolved_config_sha256 == sha256_canonical(other_config)
    assert _score(swapped, cache).cache_key != first.cache_key


def test_the_scoring_code_hash_is_part_of_the_key(repo_root: Path, tmp_path: Path) -> None:
    """02 §C7: a manually typed scorer_version alone is not sufficient."""
    from src.scoring.cache import CacheIdentity

    identity = CacheIdentity(
        artifact_id="a",
        model_revision="a" * 40,
        tokenizer_sha256="1" * 64,
        record_set_sha256="2" * 64,
        precision="float64",
        max_sequence_length=128,
        scoring_code_hash=scoring_code_hash(repo_root, CONFIG),
        backend_identity="4" * 64,
        reference_backend_identity="5" * 64,
        score_family="REFERENCE_LOSS",
        resolved_config_sha256=sha256_canonical(CONFIG),
        environment_lock_sha256="6" * 64,
    )
    mutated = dataclasses.replace(identity, scoring_code_hash="0" * 64)
    assert identity.key() != mutated.key()
