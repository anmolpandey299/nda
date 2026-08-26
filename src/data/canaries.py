"""Synthetic canary pool and per-seed inclusion [AUTH: 00 §7.1, §7.2, §7.3].

Pool: N_CANARY_CANDIDATES unique synthetic records, each carrying a high-entropy secret from
a fixed template family, generated before training and fixed across views. They contain no
real PII: every field is drawn from a keyed SHA256 stream over a declared alphabet.

Inclusion: for each training seed s, I[j,s] ~ Bernoulli(0.5) drawn INDEPENDENTLY per canary.
The draw is keyed on (inclusion seed family, canary id), so a different training seed
redraws every decision — a pool whose membership is identical across seeds would make the
canary arm a single observation repeated three times.

Repetition: an included canary appears EXACTLY ONCE. 00 §7.3 forbids repeated insertion in
the sample-level DP arm, so repetition is not offered here at all; the separated
NON_DP_MEMORIZATION_CALIBRATION arm would have to ask for it explicitly.

The canary identifier is a positional index. It deliberately encodes nothing about
membership: membership lives only in the per-seed inclusion map.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.data.normalise import NormalisedRecord, normalise_record

CANARY_GENERATOR_VERSION: Final = "s05.canary.v1"

#: Unambiguous alphabet for the secret: no 0/O or 1/I/l confusions.
_SECRET_ALPHABET: Final = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

#: The fixed template family [AUTH: 00 §7.1].
_TEMPLATE: Final = (
    "Internal record verification notice. Reference designation {designation}. "
    "Verification secret {secret}. This notice is synthetic and contains no patient data."
)


class CanaryError(ValueError):
    """A canary pool or inclusion map cannot be produced as specified."""


@dataclass(frozen=True)
class Canary:
    """One synthetic canary. `record` is its normalised, hashable form."""

    canary_id: str
    designation: str
    secret: str
    record: NormalisedRecord

    @property
    def text_sha256(self) -> str:
        return self.record.text_sha256


def _stream(label: str, count: int) -> list[int]:
    words: list[int] = []
    for index in range((count + 3) // 4):
        digest = hashlib.sha256(f"{label}|{index}".encode()).digest()
        words.extend(struct.unpack(">4Q", digest))
    return words[:count]


def _secret(label: str, length: int) -> str:
    alphabet = _SECRET_ALPHABET
    return "".join(alphabet[word % len(alphabet)] for word in _stream(label, length))


def generate_canary_pool(
    *, generator_seed: int, pool_size: int, secret_length: int
) -> list[Canary]:
    """The frozen pool. Deterministic from `generator_seed` alone [AUTH: 00 §7.1].

    Uniqueness is enforced on the normalised text hash, not merely on the secret, so two
    canaries cannot collide through the template.
    """
    if pool_size < 1:
        raise CanaryError("pool size must be positive")
    if secret_length < 8:
        raise CanaryError("the secret must carry real entropy; use at least 8 characters")

    pool: list[Canary] = []
    seen: set[str] = set()
    for index in range(pool_size):
        canary_id = f"canary-{index:05d}"
        designation = _secret(f"{generator_seed}|designation|{canary_id}", 8)
        secret = _secret(f"{generator_seed}|secret|{canary_id}", secret_length)
        text = _TEMPLATE.format(designation=designation, secret=secret)
        record = normalise_record(canary_id, {"abstract": text})
        if record.text_sha256 in seen:
            raise CanaryError(f"{canary_id}: canary text collides with an earlier canary")
        seen.add(record.text_sha256)
        pool.append(
            Canary(canary_id=canary_id, designation=designation, secret=secret, record=record)
        )
    return pool


def assert_disjoint_from_natural(
    pool: Sequence[Canary], natural: Sequence[NormalisedRecord]
) -> None:
    """A canary must never share a normalised text hash with a natural record."""
    natural_hashes = {record.text_sha256: record.record_id for record in natural}
    for canary in pool:
        clash = natural_hashes.get(canary.text_sha256)
        if clash is not None:
            raise CanaryError(
                f"{canary.canary_id} collides with natural record {clash}; the canary pool must"
                " be disjoint from the corpus [AUTH: 00 §7.1]"
            )


def inclusion_map(
    pool: Sequence[Canary], *, inclusion_seed: int, probability: float
) -> dict[str, bool]:
    """I[j,s] ~ Bernoulli(probability), drawn independently per canary [AUTH: 00 §7.2].

    The draw is a keyed hash of (inclusion seed, canary id) mapped to [0, 1), so it is
    reproducible from the recorded seed family and independent across canaries.
    """
    if not 0.0 <= probability <= 1.0:
        raise CanaryError(f"inclusion probability {probability!r} is not in [0, 1]")
    decisions: dict[str, bool] = {}
    for canary in pool:
        digest = hashlib.sha256(f"{inclusion_seed}|{canary.canary_id}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        decisions[canary.canary_id] = draw < probability
    return decisions


def included_canaries(pool: Sequence[Canary], decisions: Mapping[str, bool]) -> list[Canary]:
    """The members for one seed. Each appears exactly once [AUTH: 00 §7.2, §7.3]."""
    missing = [c.canary_id for c in pool if c.canary_id not in decisions]
    if missing:
        raise CanaryError(f"no inclusion decision for {len(missing)} canary/canaries")
    return [canary for canary in pool if decisions[canary.canary_id]]


def excluded_canaries(pool: Sequence[Canary], decisions: Mapping[str, bool]) -> list[Canary]:
    """The non-member controls for one seed: I[j,s] = 0 [AUTH: 00 §7.4]."""
    return [canary for canary in pool if not decisions[canary.canary_id]]
