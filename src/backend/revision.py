"""Model revision policy: an evidentiary backend call needs an immutable revision [01 §8G].

A floating ref moves, so a result produced under `main` cannot be re-derived and is not
evidence. The policy is enforced where the backend actually loads, not merely recorded in a
manifest, because a manifest that says one thing while the loader accepts another is exactly
the drift 01 §8G exists to prevent.

An unacquired model carries the explicit sentinel `UNRESOLVED_NOT_DOWNLOADED`. That is a
valid *declaration* and an invalid *evidentiary* value: it can be written down now and cannot
be run on later. No revision SHA is invented here — the real ones are captured during the
acquisition and compatibility step [AUTH: 03 §8].
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

from src.provenance.hashing import JSONValue, sha256_canonical

IDENTITY_SCHEMA: Final = "backend.model-identity.v1"

#: The status an evidentiary backend call fails with when no immutable revision exists.
MODEL_REVISION_NOT_FROZEN: Final = "MODEL_REVISION_NOT_FROZEN"

#: What an unacquired model records instead of a guess [AUTH: 03 §8].
UNRESOLVED: Final = "UNRESOLVED_NOT_DOWNLOADED"

#: Every 01 §8G identity a research model must carry before it can be run on.
REQUIRED_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "model_id",
    "revision",
    "architecture_family",
    "base_or_instruct",
    "config_sha256",
    "tokenizer_sha256",
    "weight_file_sha256",
    "dtype_on_disk",
    "parameter_count",
    "lora_target_mapping",
)

_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_FLOATING: Final = frozenset({"main", "master", "head", "latest", "dev", "default"})


class RevisionError(ValueError):
    """A model identity cannot pin an evidentiary backend call."""


class ModelRevisionNotFrozenError(RevisionError):
    """The requested model has no immutable revision. Fails closed [AUTH: 01 §8G]."""


def canonical_revision(revision: object) -> str:
    """The one normalised form: stripped and lower-cased.

    Validation and identity must operate on the SAME value. Validating `revision.strip()`
    while storing the raw string would let `" abc… "` and `"abc…"` validate identically and
    then hash differently, so an identity could carry surrounding whitespace [AUTH: 01 §8G].
    """
    return revision.strip().lower() if isinstance(revision, str) else ""


def revision_problems(revision: object, *, evidentiary: bool) -> list[str]:
    """Every reason this revision may not pin a call of the requested kind.

    Reads the canonical form, which is also what `resolve_model_identity` stores.
    """
    if revision is None:
        return [f"{MODEL_REVISION_NOT_FROZEN}: revision is None, which is a floating default"]
    if not isinstance(revision, str) or not revision.strip():
        return [f"{MODEL_REVISION_NOT_FROZEN}: revision is empty"]
    value = canonical_revision(revision)
    if value == UNRESOLVED.lower():
        value = UNRESOLVED
    if value == UNRESOLVED:
        return (
            [f"{MODEL_REVISION_NOT_FROZEN}: the model has not been acquired [AUTH: 03 §8]"]
            if evidentiary
            else []
        )
    if value.lower() in _FLOATING or value.startswith("refs/"):
        return [
            f"{MODEL_REVISION_NOT_FROZEN}: revision {revision!r} is a floating branch or ref;"
            " it moves, so a result produced under it cannot be re-derived [AUTH: 01 §8G]"
        ]
    if not _COMMIT_RE.match(value):
        return [
            f"{MODEL_REVISION_NOT_FROZEN}: revision {revision!r} is not an immutable 40-hex"
            " commit revision [AUTH: 01 §8G]"
        ]
    return []


class ResolvedModelIdentity:
    """One model, pinned. Opaque, factory-issued, immutable.

    `__init__` raises and this is not a dataclass, so there is no ordinary API that mints an
    identity claiming an arbitrary revision or artifact hash: `resolve_model_identity` derives
    every field from a validated manifest document and refuses anything unpinned.
    """

    __slots__ = (
        "_architecture_family",
        "_base_or_instruct",
        "_config_sha256",
        "_dtype_on_disk",
        "_evidentiary",
        "_identity",
        "_lora_target_mapping",
        "_model_id",
        "_parameter_count",
        "_revision",
        "_tokenizer_sha256",
        "_weight_file_sha256",
    )

    _model_id: str
    _revision: str
    _architecture_family: str
    _base_or_instruct: str
    _config_sha256: str
    _tokenizer_sha256: str
    _weight_file_sha256: str
    _dtype_on_disk: str
    _parameter_count: str
    _lora_target_mapping: str
    _evidentiary: bool
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "ResolvedModelIdentity is factory-issued; use resolve_model_identity() so the"
            " revision policy runs before anything can claim a pinned model [AUTH: 01 §8G]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a resolved model identity is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a resolved model identity is immutable")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def architecture_family(self) -> str:
        return self._architecture_family

    @property
    def base_or_instruct(self) -> str:
        return self._base_or_instruct

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @property
    def tokenizer_sha256(self) -> str:
        return self._tokenizer_sha256

    @property
    def weight_file_sha256(self) -> str:
        return self._weight_file_sha256

    @property
    def dtype_on_disk(self) -> str:
        return self._dtype_on_disk

    @property
    def lora_target_mapping(self) -> str:
        return self._lora_target_mapping

    @property
    def evidentiary(self) -> bool:
        """True only when every 01 §8G identity is resolved and the revision is immutable."""
        return self._evidentiary

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": IDENTITY_SCHEMA,
            "model_id": self._model_id,
            "revision": self._revision,
            "architecture_family": self._architecture_family,
            "base_or_instruct": self._base_or_instruct,
            "config_sha256": self._config_sha256,
            "tokenizer_sha256": self._tokenizer_sha256,
            "weight_file_sha256": self._weight_file_sha256,
            "dtype_on_disk": self._dtype_on_disk,
            "parameter_count": self._parameter_count,
            "lora_target_mapping": self._lora_target_mapping,
            "evidentiary": self._evidentiary,
        }

    def identity(self) -> str:
        return self._identity

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ResolvedModelIdentity({self._model_id}@{self._revision[:12]})"


def unresolved_fields(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Which 01 §8G identities still have no value. Reported, never filled in."""
    return tuple(
        name
        for name in REQUIRED_IDENTITY_FIELDS
        if str(document.get(name, UNRESOLVED)) == UNRESOLVED
    )


def resolve_model_identity(
    document: Mapping[str, Any], *, evidentiary: bool
) -> ResolvedModelIdentity:
    """Issue a pinned identity from a model manifest/panel entry, or fail closed.

    `evidentiary=False` is the declaration path: an unacquired model resolves, carries
    `evidentiary = False`, and every loader refuses it. `evidentiary=True` is the run path and
    requires the whole 01 §8G identity set plus an immutable revision.
    """
    missing = sorted(set(REQUIRED_IDENTITY_FIELDS) - set(document))
    if missing:
        raise RevisionError(f"the model identity omits {missing}")
    revision = document["revision"]
    problems = revision_problems(revision, evidentiary=evidentiary)
    if problems:
        raise ModelRevisionNotFrozenError("; ".join(problems))

    outstanding = unresolved_fields(document)
    if evidentiary and outstanding:
        raise ModelRevisionNotFrozenError(
            f"{MODEL_REVISION_NOT_FROZEN}: 01 §8G field(s) {', '.join(outstanding)} are still"
            f" {UNRESOLVED}; the model has not been acquired [AUTH: 01 §8G; 03 §8]"
        )
    if evidentiary:
        for field in ("config_sha256", "tokenizer_sha256"):
            if not _SHA256_RE.match(str(document[field])):
                raise RevisionError(f"{field} is not a sha256 digest")

    values = {name: str(document[name]) for name in REQUIRED_IDENTITY_FIELDS}
    #: The identity carries the canonical revision, so no stored identity holds whitespace
    #: or mixed case that validation never saw [AUTH: 01 §8G].
    canonical = canonical_revision(values["revision"])
    values["revision"] = UNRESOLVED if canonical == UNRESOLVED.lower() else canonical
    resolved = object.__new__(ResolvedModelIdentity)
    for name, value in values.items():
        object.__setattr__(resolved, f"_{name}", value)
    object.__setattr__(resolved, "_evidentiary", bool(evidentiary) and not outstanding)
    object.__setattr__(
        resolved, "_identity", sha256_canonical({"schema": IDENTITY_SCHEMA, **values})
    )
    return resolved


def require_evidentiary_identity(identity: ResolvedModelIdentity) -> ResolvedModelIdentity:
    """The guard every evidentiary backend entry point runs [AUTH: 01 §8G; 02 §C6]."""
    if not identity.evidentiary:
        raise ModelRevisionNotFrozenError(
            f"{MODEL_REVISION_NOT_FROZEN}: {identity.model_id} is declared but not acquired;"
            " an evidentiary backend call needs an immutable revision and resolved artifact"
            " identities [AUTH: 01 §8G]"
        )
    return identity
