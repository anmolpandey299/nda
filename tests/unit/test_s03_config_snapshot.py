"""C1 — the canonical immutable execution snapshot [AUTH: 01 §15, §17; 02 §C7].

A `ScoringContext` reads the caller's configuration exactly once, canonicalises it through
Block A's serialisation, deserialises it into plain built-ins, and derives BOTH the identity
and every executable constant from that one frozen document.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from src.provenance.hashing import JSONValue, canonical_json_bytes, sha256_bytes
from src.scoring.backends import ConstantBackend, SyntheticBackend
from src.scoring.cache import ScoreCache
from src.scoring.engine import Arm, EngineError, ScoreFamily, ScoringContext, View, score_records

CONFIG: dict[str, JSONValue] = {
    "min_k_fraction": 0.2,
    "target_fpr": 0.01,
    "precision": "float64",
    "max_sequence_length": 128,
}
RECORDS = [f"mem-{i:03d}" for i in range(8)] + [f"non-{i:03d}" for i in range(8)]
MEMBERSHIP = {r: (1 if r.startswith("mem-") else 0) for r in RECORDS}


def _context(root: Path, config: Mapping[str, JSONValue], **overrides: object) -> ScoringContext:
    base: dict[str, object] = {
        "root": root,
        "backend": SyntheticBackend(label="c1", tokens_per_record=16, member_shift=0.7),
        "reference_backend": ConstantBackend(label="ref", tokens_per_record=16),
        "resolved_config": config,
        "model_revision": "a" * 40,
        "tokenizer_sha256": "1" * 64,
        "environment_lock_sha256": "6" * 64,
    }
    base.update(overrides)
    return ScoringContext(**base)  # type: ignore[arg-type]


class LyingMapping(Mapping[str, Any]):
    """Its serialised body says one thing; its `.get()` answers another.

    This is the executed counterexample: a caller object that would let a run hash config B
    while executing config A, if anything read it after the snapshot was taken.
    """

    def __init__(self, stored: dict[str, Any], lies: dict[str, Any]) -> None:
        self._stored = stored
        self._lies = lies
        self.get_calls: list[str] = []

    def __getitem__(self, key: str) -> Any:
        return self._stored[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._stored)

    def __len__(self) -> int:
        return len(self._stored)

    def get(self, key: str, default: Any = None) -> Any:
        self.get_calls.append(key)
        return self._lies.get(key, self._stored.get(key, default))


# ------------------------------------------------------------------ malicious mapping
def test_a_lying_mapping_cannot_hash_one_config_and_execute_another(repo_root: Path) -> None:
    """Stored body says min_k_fraction = 1.0; `.get()` answers 0.2. Execution must use 1.0,
    the value the identity was computed from, and `.get()` must never be consulted."""
    stored = {**CONFIG, "min_k_fraction": 1.0}
    lying = LyingMapping(stored, {"min_k_fraction": 0.2})
    context = _context(repo_root, lying)

    assert context.min_k_fraction() == 1.0
    assert context.resolved_config_sha256 == sha256_bytes(canonical_json_bytes(stored))
    assert lying.get_calls == [], (
        f"the caller object was read after the snapshot: {lying.get_calls}"
    )


def test_the_lying_mapping_cannot_be_paired_with_the_other_hash(repo_root: Path) -> None:
    """Declaring SHA(min_k=0.2) while the body says 1.0 is refused outright."""
    stored = {**CONFIG, "min_k_fraction": 1.0}
    other = {**CONFIG, "min_k_fraction": 0.2}
    with pytest.raises(EngineError, match="does not hash the configuration"):
        _context(
            repo_root,
            LyingMapping(stored, {"min_k_fraction": 0.2}),
            expected_config_sha256=sha256_bytes(canonical_json_bytes(other)),
        )


def test_the_executed_score_follows_the_snapshot_not_the_lie(
    repo_root: Path, tmp_path: Path
) -> None:
    """End to end: the Min-K reduction actually performed is the snapshot's."""

    def run(config: Mapping[str, JSONValue]):  # type: ignore[no-untyped-def]
        return score_records(
            arm=Arm.CANARY,
            score=ScoreFamily.MIN_K,
            view=View.SYNTHETIC,
            artifact="a",
            seed=101,
            fold=0,
            record_ids=RECORDS,
            membership=MEMBERSHIP,
            context=_context(repo_root, config),
            cache=ScoreCache(tmp_path),
        )

    honest_full = run({**CONFIG, "min_k_fraction": 1.0})
    honest_fifth = run({**CONFIG, "min_k_fraction": 0.2})
    assert [r.scalar_score for r in honest_full.rows] != [r.scalar_score for r in honest_fifth.rows]

    lying = run(LyingMapping({**CONFIG, "min_k_fraction": 1.0}, {"min_k_fraction": 0.2}))
    assert [r.scalar_score for r in lying.rows] == [r.scalar_score for r in honest_full.rows]
    assert lying.cache_key == honest_full.cache_key


# ------------------------------------------------------------------ post-construction mutation
def test_mutating_the_caller_object_afterwards_changes_nothing(repo_root: Path) -> None:
    mutable = dict(CONFIG)
    context = _context(repo_root, mutable)
    before_digest = context.resolved_config_sha256
    before_fraction = context.min_k_fraction()
    before_snapshot = context.snapshot

    mutable["min_k_fraction"] = 0.99
    mutable["precision"] = "bfloat16"
    mutable["max_sequence_length"] = 4096

    assert context.resolved_config_sha256 == before_digest
    assert context.min_k_fraction() == before_fraction
    assert context.snapshot == before_snapshot
    assert context.precision == "float64"
    assert context.max_sequence_length == 128


def test_the_snapshot_is_not_the_caller_object(repo_root: Path) -> None:
    mutable = dict(CONFIG)
    context = _context(repo_root, mutable)
    assert context.snapshot is not mutable
    with pytest.raises(TypeError):
        context.resolved_config["min_k_fraction"] = 0.5  # type: ignore[index]


# ------------------------------------------------------------------ runtime binding
def test_the_precision_comes_from_the_snapshot(repo_root: Path) -> None:
    assert _context(repo_root, CONFIG).precision == "float64"
    assert _context(repo_root, {**CONFIG, "precision": "bfloat16"}).precision == "bfloat16"


def test_a_runtime_precision_mismatch_is_rejected(repo_root: Path) -> None:
    with pytest.raises(EngineError, match="runtime precision"):
        _context(repo_root, CONFIG, runtime_precision="float32")
    assert _context(repo_root, CONFIG, runtime_precision="float64").precision == "float64"


def test_a_runtime_max_sequence_mismatch_is_rejected(repo_root: Path) -> None:
    with pytest.raises(EngineError, match="runtime max_sequence_length"):
        _context(repo_root, CONFIG, runtime_max_sequence_length=256)
    assert _context(repo_root, CONFIG, runtime_max_sequence_length=128).max_sequence_length == 128


def test_a_missing_material_setting_fails_closed(repo_root: Path) -> None:
    for key in ("precision", "max_sequence_length", "min_k_fraction", "target_fpr"):
        stripped = {k: v for k, v in CONFIG.items() if k != key}
        context = _context(repo_root, stripped)
        with pytest.raises(EngineError, match=key):
            getattr(context, key)() if key in {"min_k_fraction", "target_fpr"} else getattr(
                context, key
            )


def test_a_non_canonicalisable_config_is_refused(repo_root: Path) -> None:
    with pytest.raises(EngineError, match="not canonicalisable"):
        _context(repo_root, {**CONFIG, "bad": {1, 2}})  # type: ignore[dict-item]
