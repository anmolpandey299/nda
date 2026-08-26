"""K/L + DRY-CACHE — cache identity and corruption [AUTH: 02 §C7; 00 §34A.4, §34B.1]."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.provenance.hashing import JSONValue
from src.scoring.cache import (
    SCORING_SOURCE_FILES,
    CacheError,
    CacheIdentity,
    ScoreCache,
    scoring_code_hash,
)

BASE = CacheIdentity(
    artifact_id="artifact-A",
    model_revision="a" * 40,
    tokenizer_sha256="1" * 64,
    record_set_sha256="2" * 64,
    precision="float64",
    max_sequence_length=128,
    scoring_code_hash="3" * 64,
    backend_identity="4" * 64,
    reference_backend_identity="8" * 64,
    score_family="REFERENCE_LOSS",
    resolved_config_sha256="5" * 64,
    environment_lock_sha256="6" * 64,
)
ROWS: list[dict[str, JSONValue]] = [
    {"record_id": "r1", "scalar_score": 0.5, "token_count": 12},
    {"record_id": "r2", "scalar_score": -0.25, "token_count": 12},
]


# ------------------------------------------------------------------ K: identity
def test_stable_hit_for_an_identical_identity(tmp_path: Path) -> None:
    """DRY-CACHE stable hit: same identity -> same key, hit, bitwise-identical scores."""
    cache = ScoreCache(tmp_path)
    first = BASE.key()
    cache.store(BASE, ROWS)
    second = dataclasses.replace(BASE).key()
    assert first == second
    loaded = cache.load(BASE)
    assert loaded == ROWS


@pytest.mark.parametrize(
    "field",
    [
        "artifact_id",
        "model_revision",
        "tokenizer_sha256",
        "record_set_sha256",
        "precision",
        "scoring_code_hash",
        "backend_identity",
        "reference_backend_identity",
        "score_family",
        "resolved_config_sha256",
        "environment_lock_sha256",
    ],
)
def test_every_identity_component_invalidates(tmp_path: Path, field: str) -> None:
    """DRY-CACHE version invalidation, generalised to every 02 §C7 component."""
    cache = ScoreCache(tmp_path)
    cache.store(BASE, ROWS)
    mutated = dataclasses.replace(BASE, **{field: "z" * 40})  # type: ignore[arg-type]
    assert mutated.key() != BASE.key()
    assert cache.load(mutated) is None, f"{field} did not invalidate the cache"


def test_max_sequence_length_invalidates(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    cache.store(BASE, ROWS)
    mutated = dataclasses.replace(BASE, max_sequence_length=256)
    assert mutated.key() != BASE.key()
    assert cache.load(mutated) is None


def test_a_miss_is_a_miss_not_an_error(tmp_path: Path) -> None:
    assert ScoreCache(tmp_path).load(BASE) is None


# ------------------------------------------------------------------ scoring code hash
def test_scoring_code_hash_covers_the_scoring_source(repo_root: Path) -> None:
    """A manually typed version string is not sufficient; the source bytes are the identity."""
    config = {"min_k_fraction": 0.2}
    first = scoring_code_hash(repo_root, config)
    assert first == scoring_code_hash(repo_root, config)
    assert first != scoring_code_hash(repo_root, {"min_k_fraction": 0.3})
    for relative in SCORING_SOURCE_FILES:
        assert (repo_root / relative).is_file(), relative


def test_a_changed_scoring_source_changes_the_hash(repo_root: Path, tmp_path: Path) -> None:
    """Copy the scoring package, change one byte of the reducer, and the identity moves."""
    import shutil

    shutil.copytree(repo_root / "src", tmp_path / "src")
    config = {"min_k_fraction": 0.2}
    before = scoring_code_hash(tmp_path, config)
    target = tmp_path / "src" / "scoring" / "reference.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert scoring_code_hash(tmp_path, config) != before


def test_a_missing_scoring_source_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match="scoring source missing"):
        scoring_code_hash(tmp_path, {"min_k_fraction": 0.2})


# ------------------------------------------------------------------ L: corruption
def test_a_corrupt_payload_is_rejected_not_served(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    cache.store(BASE, ROWS)
    path = cache.path_for(BASE)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["rows"][0]["scalar_score"] = 999.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CacheError, match="corrupt"):
        cache.load(BASE)


def test_a_swapped_identity_under_the_same_filename_is_rejected(tmp_path: Path) -> None:
    """The stored identity must be the one requested, not merely share a filename."""
    cache = ScoreCache(tmp_path)
    cache.store(BASE, ROWS)
    path = cache.path_for(BASE)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["identity"]["precision"] = "bfloat16"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CacheError, match="stored identity"):
        cache.load(BASE)


def test_an_unknown_schema_is_rejected(tmp_path: Path) -> None:
    cache = ScoreCache(tmp_path)
    cache.store(BASE, ROWS)
    path = cache.path_for(BASE)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema"] = "something-else"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CacheError, match="schema"):
        cache.load(BASE)


# ------------------------------------------------------------------ B1: semantic identity
def test_the_score_family_is_part_of_the_identity(tmp_path: Path) -> None:
    """REFERENCE_LOSS and MIN_K reduce the same evidence differently; they are different
    measurements and must not share an entry [AUTH: 00 §14.1, §14.2; 02 §C7]."""
    cache = ScoreCache(tmp_path)
    cache.store(BASE, ROWS)
    other = dataclasses.replace(BASE, score_family="MIN_K")
    assert other.key() != BASE.key()
    assert cache.load(other) is None


def test_the_reference_backend_is_part_of_the_identity(tmp_path: Path) -> None:
    """s_ref is a difference against a specific reference model."""
    cache = ScoreCache(tmp_path)
    cache.store(BASE, ROWS)
    other = dataclasses.replace(BASE, reference_backend_identity="9" * 64)
    assert other.key() != BASE.key()
    assert cache.load(other) is None


def test_attribution_fields_are_refused_by_the_cache(tmp_path: Path) -> None:
    """A cached row carrying seed/fold/arm could hand a later request another request's
    attribution, so the payload is restricted to scoring output [AUTH: 00 §34A.1]."""
    cache = ScoreCache(tmp_path)
    for field in ("seed", "fold", "arm", "view", "membership_label", "artifact_id"):
        rows: list[dict[str, JSONValue]] = [
            {"record_id": "r1", "scalar_score": 0.5, "token_count": 12, field: 101}
        ]
        with pytest.raises(CacheError, match="may not be cached"):
            cache.store(BASE, rows)
