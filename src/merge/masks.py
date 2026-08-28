"""Deterministic DARE mask identity [AUTH: 00 §10.2, §24.1; 01 §17, §30].

A DARE mask decides which coordinates of a task vector survive into a public descendant, and
under some later lineage conditions it may itself be released metadata. So it is derived,
recorded and reproducible, never a side effect of whichever RNG the process reached.

**Seed scope (architect adjudication).** Each DARE *descendant* has exactly one merge-level
seed. The protected and partner masks are independent domain-separated substreams of it:

    protected stream = derive(merge_seed, "protected", protected_identity, tensor, p, ...)
    partner   stream = derive(merge_seed, "partner",   partner_identity,   tensor, p, ...)

There are deliberately not two independently caller-selectable scientific seeds. One seed per
descendant is what a registry can record and a reader can reproduce; two invite a pairing
nobody declared.

**What the per-tensor keying does and does not guarantee.** Each tensor draws from its own
labelled stream, keyed on the whole-vector content identity among other things. So:

* reordering or re-iterating the SAME logical task vector cannot move any mask — dict
  iteration order, PYTHONHASHSEED, threads and filesystem order are all irrelevant;
* a task vector with a different tensor set is a *different task vector* with a different
  content identity, so its masks differ throughout. That is correct — the scientific object
  changed — and it is not a claim that existing tensors keep their old masks.

The merge seed is materially part of every stream label. S07 records the mask; it does not
decide who may see it, which is S09's lineage-condition question.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from src.merge.context import MergeExecutionContext
from src.merge.updates import TaskVector
from src.provenance.hashing import JSONValue

BoolMatrix = NDArray[np.bool_]

#: The two constituent roles a merge-level seed is split into.
PROTECTED_ROLE: Final = "protected"
PARTNER_ROLE: Final = "partner"
CONSTITUENT_ROLES: Final[tuple[str, ...]] = (PROTECTED_ROLE, PARTNER_ROLE)

_MANTISSA: Final = float(1 << 53)


class MaskError(ValueError):
    """A DARE mask cannot be derived as specified."""


def _uniform_stream(label: str, count: int) -> NDArray[np.float64]:
    """`count` values in [0, 1) from a SHA256 counter stream keyed on `label`.

    Counter-mode SHA256 rather than a library generator, so the same declared seed reproduces
    the same mask on any platform and any NumPy version. Drawn in float64 because a mask
    threshold comparison is a decision, not artifact arithmetic.
    """
    words: list[int] = []
    for block in range((count + 3) // 4):
        digest = hashlib.sha256(f"{label}|{block}".encode()).digest()
        words.extend(struct.unpack(">4Q", digest))
    raw = np.array(words[:count], dtype=np.uint64)
    return (raw >> np.uint64(11)).astype(np.float64) / _MANTISSA


def tensor_stream_label(
    *,
    scheme: str,
    operator_version: str,
    merge_seed: int,
    role: str,
    constituent_identity: str,
    drop_probability: float,
    name: str,
) -> str:
    """The domain separator for one constituent's one tensor. Every material input appears."""
    return (
        f"{scheme}|{operator_version}|merge_seed={merge_seed}|role={role}"
        f"|constituent={constituent_identity}|p={drop_probability!r}|tensor={name}"
    )


class DareMask:
    """Which coordinates survive, and everything needed to reproduce that decision.

    Opaque and factory-issued: masks are drawn by `derive_mask` from the merge seed's role
    substream, and no caller supplies mask bytes or a mask hash.
    """

    __slots__ = (
        "_constituent_identity",
        "_drop_probability",
        "_identity",
        "_keep",
        "_merge_seed",
        "_operator_version",
        "_role",
        "_scheme",
    )

    _keep: Mapping[str, bytes]
    _merge_seed: int
    _role: str
    _drop_probability: float
    _constituent_identity: str
    _scheme: str
    _operator_version: str
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("DareMask is factory-issued; use derive_mask()")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a DARE mask is immutable")

    @property
    def merge_seed(self) -> int:
        return self._merge_seed

    @property
    def role(self) -> str:
        return self._role

    @property
    def drop_probability(self) -> float:
        return self._drop_probability

    @property
    def constituent_identity(self) -> str:
        return self._constituent_identity

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def operator_version(self) -> str:
        return self._operator_version

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._keep))

    def keep_array(self, name: str) -> BoolMatrix:
        """A read-only boolean view of one tensor's survival pattern."""
        if name not in self._keep:
            raise MaskError(f"{name!r} is not covered by this mask")
        return np.frombuffer(self._keep[name], dtype=np.bool_)

    @property
    def retained(self) -> int:
        return int(sum(int(np.count_nonzero(self.keep_array(n))) for n in self.names))

    @property
    def total(self) -> int:
        return int(sum(int(self.keep_array(n).size) for n in self.names))

    @property
    def dropped(self) -> int:
        return self.total - self.retained

    @property
    def retained_fraction(self) -> float:
        return self.retained / self.total if self.total else 0.0

    def identity(self) -> str:
        """Hash of the actual mask bytes, plus what produced them."""
        return self._identity

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "scheme": self._scheme,
            "operator_version": self._operator_version,
            "dare_merge_seed": self._merge_seed,
            "constituent_role": self._role,
            "drop_probability": self._drop_probability,
            "constituent_identity": self._constituent_identity,
            "mask_sha256": self._identity,
            "retained_coordinates": self.retained,
            "dropped_coordinates": self.dropped,
            "total_coordinates": self.total,
            "retained_fraction": self.retained_fraction,
        }


def _issue_mask(
    *,
    keep: Mapping[str, np.ndarray[Any, Any]],
    merge_seed: int,
    role: str,
    drop_probability: float,
    constituent_identity: str,
    scheme: str,
    operator_version: str,
) -> DareMask:
    payload = {
        name: np.ascontiguousarray(array, dtype=np.bool_).tobytes("C")
        for name, array in keep.items()
    }
    digest = hashlib.sha256(
        f"{scheme}|{operator_version}|{merge_seed}|{role}"
        f"|{drop_probability!r}|{constituent_identity}".encode()
    )
    for name in sorted(payload):
        digest.update(name.encode("utf-8"))
        digest.update(payload[name])

    mask = object.__new__(DareMask)
    object.__setattr__(mask, "_keep", payload)
    object.__setattr__(mask, "_merge_seed", merge_seed)
    object.__setattr__(mask, "_role", role)
    object.__setattr__(mask, "_drop_probability", drop_probability)
    object.__setattr__(mask, "_constituent_identity", constituent_identity)
    object.__setattr__(mask, "_scheme", scheme)
    object.__setattr__(mask, "_operator_version", operator_version)
    object.__setattr__(mask, "_identity", digest.hexdigest())
    return mask


def derive_mask(
    vector: TaskVector,
    *,
    drop_probability: float,
    merge_seed: int,
    role: str,
    context: MergeExecutionContext,
) -> DareMask:
    """Draw one constituent's survival mask from its substream of the merge seed.

    A coordinate survives when its stream draw is at or above `drop_probability`, so p = 0
    retains everything exactly and larger p drops more [AUTH: 00 §10.2].
    """
    if role not in CONSTITUENT_ROLES:
        raise MaskError(f"{role!r} is not a declared constituent role {CONSTITUENT_ROLES}")
    if not 0.0 <= drop_probability < 1.0:
        raise MaskError(
            f"drop probability {drop_probability!r} is outside [0, 1); p = 1 would drop the"
            " whole task vector and is refused [AUTH: 00 §10.2]"
        )
    if isinstance(merge_seed, bool) or not isinstance(merge_seed, int):
        raise MaskError("the DARE merge seed must be an integer and must be recorded")
    identity = vector.content_identity()
    keep: dict[str, BoolMatrix] = {}
    for name in vector.names:
        shape = vector.shapes[name]
        draws = _uniform_stream(
            tensor_stream_label(
                scheme=context.mask_scheme,
                operator_version=context.operator_version,
                merge_seed=merge_seed,
                role=role,
                constituent_identity=identity,
                drop_probability=drop_probability,
                name=name,
            ),
            shape[0] * shape[1],
        )
        keep[name] = (draws >= drop_probability).reshape(shape)
    return _issue_mask(
        keep=keep,
        merge_seed=merge_seed,
        role=role,
        drop_probability=drop_probability,
        constituent_identity=identity,
        scheme=context.mask_scheme,
        operator_version=context.operator_version,
    )
