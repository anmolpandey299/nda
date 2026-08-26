"""Scoring backends. Block B ships the deterministic synthetic fixture only.

The real HF/PEFT backend arrives with the block that integrates it; until then
`BACKEND_INTEGRATED = FALSE` [AUTH: 02 §C6]. The protocol exists so that block can be
plugged in without the engine, cache or controller changing shape [AUTH: 00 §34A.1].
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np

from src.scoring.reference import TokenEvidence

_MANTISSA_DIVISOR: Final = float(1 << 53)


class ScoringBackend(Protocol):
    """What the engine needs from any backend, real or synthetic."""

    @property
    def identity(self) -> str:
        """Stable hash of the backend adapter; a cache-key input [AUTH: 02 §C7]."""

    def token_evidence(self, *, artifact_id: str, record_id: str) -> TokenEvidence:
        """Per-token log-probabilities for one record under one artifact."""


def _stream(label: str, count: int) -> np.ndarray:
    """The Block A SHA256 counter stream, reused so fixtures stay reproducible anywhere."""
    words: list[int] = []
    for index in range((count + 3) // 4):
        digest = hashlib.sha256(f"{label}|{index}".encode()).digest()
        words.extend(struct.unpack(">4Q", digest))
    raw = np.array(words[:count], dtype=np.uint64)
    return (raw >> np.uint64(11)).astype(np.float64) / _MANTISSA_DIVISOR


@dataclass(frozen=True)
class SyntheticBackend:
    """A deterministic stand-in with a planted member/non-member separation.

    Members are drawn with a shifted mean, so the planted effect is known analytically and
    the estimator can be checked against it rather than against itself. No model, no
    network, no GPU [AUTH: 01 §20; 00 §34B].
    """

    label: str
    tokens_per_record: int
    member_shift: float
    base_log_prob: float = -2.0
    spread: float = 0.5

    @property
    def identity(self) -> str:
        payload = (
            f"{self.label}|{self.tokens_per_record}|{self.member_shift}"
            f"|{self.base_log_prob}|{self.spread}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def token_evidence(self, *, artifact_id: str, record_id: str) -> TokenEvidence:
        raw = _stream(f"{self.label}|{artifact_id}|{record_id}", self.tokens_per_record)
        # inverse-CDF-free symmetric jitter; the planted shift is what the tests rely on
        jitter = (2.0 * raw - 1.0) * self.spread
        shift = self.member_shift if record_id.startswith("mem-") else 0.0
        log_probs = self.base_log_prob + shift + jitter
        valid = np.ones(self.tokens_per_record, dtype=np.bool_)
        return TokenEvidence(log_probs=log_probs.astype(np.float64), valid=valid)


@dataclass(frozen=True)
class ConstantBackend:
    """A reference stand-in: identical evidence for every record and artifact."""

    label: str
    tokens_per_record: int
    log_prob: float = -2.0

    @property
    def identity(self) -> str:
        payload = f"{self.label}|{self.tokens_per_record}|{self.log_prob}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def token_evidence(self, *, artifact_id: str, record_id: str) -> TokenEvidence:
        return TokenEvidence(
            log_probs=np.full(self.tokens_per_record, self.log_prob, dtype=np.float64),
            valid=np.ones(self.tokens_per_record, dtype=np.bool_),
        )


def record_set_identity(record_ids: Sequence[str]) -> str:
    """Order-sensitive identity of a record set; a cache-key input [AUTH: 00 §34A.4]."""
    digest = hashlib.sha256()
    for record_id in record_ids:
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
