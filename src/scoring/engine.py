"""THE parameterised scoring engine. One engine, not a script per arm/view/attack.

00 §34A.1 fixes the public call and the row schema; 00 §34A.5 forbids a second scoring path.
The engine reduces token evidence to one scalar per record, binds the result to a cache
identity, and returns the canonical ScoreTable every downstream view consumes.

It never trains and never merges. The backend is injected, so the block that integrates the
real HF/PEFT path plugs in without this file changing shape.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.provenance.hashing import (
    CanonicalisationError,
    JSONDocument,
    JSONValue,
    canonical_json_bytes,
    sha256_bytes,
)
from src.scoring.backends import ScoringBackend, record_set_identity
from src.scoring.cache import CacheIdentity, ScoreCache, scoring_code_hash
from src.scoring.reference import (
    ScoreFamilyError,
    TokenEvidence,
    min_k_percent_score,
    reference_calibrated_score,
)

FloatArray = NDArray[np.float64]


class Arm(StrEnum):
    CANARY = "CANARY"
    NATURAL = "NATURAL"


class ScoreFamily(StrEnum):
    REFERENCE_LOSS = "REFERENCE_LOSS"
    MIN_K = "MIN_K"


class View(StrEnum):
    ORACLE = "ORACLE"
    DESCENDANT = "DESCENDANT"
    POOLED = "POOLED"
    RECOVERED = "RECOVERED"
    SYNTHETIC = "SYNTHETIC"


#: The 00 §34A.1 row schema. Additional view metadata may ride alongside; these do not move.
SCORE_ROW_FIELDS: Final[tuple[str, ...]] = (
    "record_id",
    "seed",
    "fold",
    "arm",
    "score_family",
    "view",
    "artifact_id",
    "membership_label",
    "scalar_score",
    "token_count",
)


class EngineError(RuntimeError):
    """The engine refused to produce or return a score table."""


@dataclass(frozen=True)
class ScoreRow:
    record_id: str
    seed: int
    fold: int | None
    arm: str
    score_family: str
    view: str
    artifact_id: str
    membership_label: int
    scalar_score: float
    token_count: int

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "record_id": self.record_id,
            "seed": self.seed,
            "fold": self.fold,
            "arm": self.arm,
            "score_family": self.score_family,
            "view": self.view,
            "artifact_id": self.artifact_id,
            "membership_label": self.membership_label,
            "scalar_score": self.scalar_score,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class ScoreTable:
    """The canonical table. Every view, pooled transform and metric reads this shape."""

    rows: tuple[ScoreRow, ...]
    cache_key: str
    cache_hit: bool

    def __len__(self) -> int:
        return len(self.rows)

    def scores(self) -> FloatArray:
        return np.array([row.scalar_score for row in self.rows], dtype=np.float64)

    def labels(self) -> NDArray[np.int_]:
        return np.array([row.membership_label for row in self.rows], dtype=np.int_)

    def record_ids(self) -> tuple[str, ...]:
        return tuple(row.record_id for row in self.rows)

    def subset(self, record_ids: Sequence[str]) -> ScoreTable:
        wanted = set(record_ids)
        return ScoreTable(
            rows=tuple(row for row in self.rows if row.record_id in wanted),
            cache_key=self.cache_key,
            cache_hit=self.cache_hit,
        )


@dataclass(frozen=True)
class ScoringContext:
    """Everything the engine needs that is not per-call [AUTH: 02 §C7; 00 §34A.4].

    The caller's configuration object is used exactly once, at construction, to build a
    canonical execution snapshot:

        caller mapping
            -> Block A canonical serialisation (bytes)
            -> json.loads: a document of plain built-ins
            -> frozen snapshot, the ONE source of both the hash and the constants

    Nothing is read from the caller's object afterwards. A mapping whose serialised body says
    `min_k_fraction = 1.0` while a custom `.get()` answers `0.2` therefore cannot execute 0.2:
    `.get()` is never consulted, and hashing and execution read the same snapshot, so
    "hash B, execute A" has no representation [AUTH: 01 §15, §17; 02 §C7].

    Mutating the caller's object after construction changes nothing here, because the
    snapshot is a separate deserialised document.

    `precision` and `max_sequence_length` come from that snapshot too. They are not
    independently caller-controlled: a runtime that actually differs must be declared and is
    then checked against the configured value, which fails closed on a mismatch.
    """

    root: Path
    backend: ScoringBackend
    reference_backend: ScoringBackend
    resolved_config: JSONDocument
    model_revision: str
    tokenizer_sha256: str
    environment_lock_sha256: str
    expected_config_sha256: str | None = None
    runtime_precision: str | None = None
    runtime_max_sequence_length: int | None = None
    #: Set in __post_init__ from the canonical snapshot; never supplied by a caller.
    _snapshot: dict[str, JSONValue] = field(default_factory=dict, repr=False, compare=False)
    _config_sha256: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            payload = canonical_json_bytes(dict(self.resolved_config))
        except CanonicalisationError as exc:
            raise EngineError(f"the scoring config is not canonicalisable: {exc}") from exc
        snapshot = json.loads(payload.decode("utf-8"))
        if not isinstance(snapshot, dict):
            raise EngineError("the scoring config must canonicalise to a JSON object")
        # The snapshot replaces the caller's object outright: one source, no second read.
        object.__setattr__(self, "resolved_config", MappingProxyType(snapshot))
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_config_sha256", sha256_bytes(payload))

        if (
            self.expected_config_sha256 is not None
            and self.expected_config_sha256 != self._config_sha256
        ):
            raise EngineError(
                f"declared resolved_config_sha256 {self.expected_config_sha256[:12]} does not"
                f" hash the configuration this run will execute ({self._config_sha256[:12]});"
                " refusing before any cache lookup or scoring [AUTH: 01 §15, §17; 02 §C7]"
            )
        self._check_runtime_agreement()

    def _check_runtime_agreement(self) -> None:
        if self.runtime_precision is not None and self.runtime_precision != self.precision:
            raise EngineError(
                f"runtime precision {self.runtime_precision!r} differs from the configured"
                f" {self.precision!r}; the cache identity would describe a computation that"
                " did not happen [AUTH: 01 §10; 02 §C7]"
            )
        if (
            self.runtime_max_sequence_length is not None
            and self.runtime_max_sequence_length != self.max_sequence_length
        ):
            raise EngineError(
                f"runtime max_sequence_length {self.runtime_max_sequence_length} differs from"
                f" the configured {self.max_sequence_length} [AUTH: 00 §34A.4; 02 §C7]"
            )

    @property
    def snapshot(self) -> dict[str, JSONValue]:
        """The canonical execution document. Hash and constants both come from here."""
        return dict(self._snapshot)

    @property
    def resolved_config_sha256(self) -> str:
        """The identity of the snapshot actually being executed."""
        return self._config_sha256

    def _setting(self, key: str) -> JSONValue:
        if key not in self._snapshot:
            raise EngineError(f"the resolved scoring config does not define {key!r}")
        return self._snapshot[key]

    def _number(self, key: str) -> float:
        value = self._setting(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EngineError(f"scoring config {key!r} is not a number")
        return float(value)

    def _integer(self, key: str) -> int:
        value = self._setting(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise EngineError(f"scoring config {key!r} is not an integer")
        return value

    @property
    def precision(self) -> str:
        value = self._setting("precision")
        if not isinstance(value, str):
            raise EngineError("scoring config 'precision' is not a string")
        return value

    @property
    def max_sequence_length(self) -> int:
        return self._integer("max_sequence_length")

    def target_fpr(self) -> float:
        return self._number("target_fpr")

    def min_k_fraction(self) -> float:
        return self._number("min_k_fraction")


def _reduce(
    family: ScoreFamily,
    target: TokenEvidence,
    reference: TokenEvidence,
    fraction: float,
) -> float:
    if family is ScoreFamily.REFERENCE_LOSS:
        return reference_calibrated_score(target, reference)
    return min_k_percent_score(target, fraction)


def score_records(
    *,
    arm: Arm,
    score: ScoreFamily,
    view: View,
    artifact: str,
    seed: int,
    fold: int | None,
    record_ids: Sequence[str],
    membership: Mapping[str, int],
    context: ScoringContext,
    cache: ScoreCache | None = None,
) -> ScoreTable:
    """The 00 §34A.1 call. One code path for every arm, family and view."""
    if not record_ids:
        raise EngineError("no records to score")
    missing = [record for record in record_ids if record not in membership]
    if missing:
        raise EngineError(f"membership label missing for {len(missing)} record(s)")

    identity = CacheIdentity(
        artifact_id=artifact,
        model_revision=context.model_revision,
        tokenizer_sha256=context.tokenizer_sha256,
        record_set_sha256=record_set_identity(record_ids),
        precision=context.precision,
        max_sequence_length=context.max_sequence_length,
        scoring_code_hash=scoring_code_hash(context.root, context.resolved_config),
        backend_identity=context.backend.identity,
        reference_backend_identity=context.reference_backend.identity,
        score_family=str(score),
        resolved_config_sha256=context.resolved_config_sha256,
        environment_lock_sha256=context.environment_lock_sha256,
    )
    key = identity.key()

    cached = cache.load(identity) if cache is not None else None
    if cached is not None:
        # Attribution is rebuilt from THIS request. The cache supplies only the scalar and
        # the token count, so a previous call's seed, fold, arm, view or membership label
        # cannot travel into this one [AUTH: 00 §34A.1].
        payload = {str(row["record_id"]): row for row in cached}
        missing = [record for record in record_ids if record not in payload]
        if missing:
            raise EngineError(f"cache entry is missing {len(missing)} record(s)")
        rows = tuple(
            ScoreRow(
                record_id=record_id,
                seed=seed,
                fold=fold,
                arm=str(arm),
                score_family=str(score),
                view=str(view),
                artifact_id=artifact,
                membership_label=int(membership[record_id]),
                scalar_score=float(str(payload[record_id]["scalar_score"])),
                token_count=int(str(payload[record_id]["token_count"])),
            )
            for record_id in record_ids
        )
        return ScoreTable(rows=rows, cache_key=key, cache_hit=True)

    fraction = context.min_k_fraction()
    produced: list[ScoreRow] = []
    for record_id in record_ids:
        target = context.backend.token_evidence(artifact_id=artifact, record_id=record_id)
        reference = context.reference_backend.token_evidence(
            artifact_id=artifact, record_id=record_id
        )
        try:
            scalar = _reduce(score, target, reference, fraction)
        except ScoreFamilyError as exc:
            raise EngineError(f"{record_id}: {exc}") from exc
        produced.append(
            ScoreRow(
                record_id=record_id,
                seed=seed,
                fold=fold,
                arm=str(arm),
                score_family=str(score),
                view=str(view),
                artifact_id=artifact,
                membership_label=int(membership[record_id]),
                scalar_score=scalar,
                token_count=target.token_count,
            )
        )
    if cache is not None:
        cache.store(
            identity,
            [
                {
                    "record_id": row.record_id,
                    "scalar_score": row.scalar_score,
                    "token_count": row.token_count,
                }
                for row in produced
            ],
        )
    return ScoreTable(rows=tuple(produced), cache_key=key, cache_hit=False)
