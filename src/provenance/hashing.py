"""Canonical serialisation and SHA256 — the single hashing path for the whole repository.

Every provenance hash in this project (config identity, manifest identity, artifact
identity, experiment identity, RUN_ID) is a SHA256 over either raw file bytes or the
canonical JSON encoding defined here. There is deliberately one implementation: a second
canonicaliser would let two modules disagree about what "the same config" means, and a
result would then be attributed to provenance it does not have [AUTH: 01 §13-§16, §19].

Canonical JSON is:

    sorted keys, no insignificant whitespace, UTF-8, exactly one trailing newline

and it refuses NaN/Infinity and non-string object keys, so a document that cannot be
reproduced byte-for-byte cannot acquire an identity at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

type JSONValue = str | int | float | bool | None | Mapping[str, JSONValue] | Sequence[JSONValue]
type JSONDocument = Mapping[str, JSONValue]

#: Read size for streamed file hashing. Not a scientific constant.
_CHUNK: Final = 1 << 20

_SEPARATORS: Final = (",", ":")


class CanonicalisationError(ValueError):
    """A value cannot be encoded reproducibly, so it may not receive an identity."""


class ArtifactVerificationError(RuntimeError):
    """A recorded artifact is missing or no longer hashes to its recorded digest."""


def _check_canonical(value: JSONValue, path: str = "$") -> None:
    """Reject anything whose JSON encoding would not be reproducible.

    `json.dumps` would silently coerce integer keys to strings and emit the non-standard
    `NaN`/`Infinity` tokens. Both would produce an identity that another reader cannot
    reconstruct, so both fail closed here rather than at read time.
    """
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise CanonicalisationError(f"{path}: non-finite float is not canonicalisable")
        return
    if isinstance(value, str):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalisationError(f"{path}: object key {key!r} is not a string")
            _check_canonical(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _check_canonical(item, f"{path}[{index}]")
        return
    raise CanonicalisationError(f"{path}: {type(value).__name__} is not a JSON type")


def canonical_json_bytes(value: JSONValue) -> bytes:
    """The exact bytes that both the hash and the on-disk file are made of."""
    _check_canonical(value)
    text = json.dumps(
        value, sort_keys=True, separators=_SEPARATORS, ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8") + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Streamed so a multi-gigabyte adapter or weight file never has to be resident."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical(value: JSONValue) -> str:
    """Identity of structured data. Key order and whitespace cannot change it."""
    return sha256_bytes(canonical_json_bytes(value))


def write_canonical_json(path: Path, value: JSONValue) -> str:
    """Write canonically and atomically; return the SHA256 of the bytes written.

    Atomic because a run manifest or resolved config that is half-written during a crash
    must not be readable as a complete one [AUTH: 01 §16, §36].
    """
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return sha256_bytes(payload)


def read_canonical_json(path: Path) -> JSONValue:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise CanonicalisationError(f"{path}: not valid JSON ({exc.msg})") from exc


@dataclass(frozen=True)
class ArtifactRecord:
    """One scientific artifact, bound to the bytes it had when it was recorded."""

    path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, JSONValue]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


def record_artifact(root: Path, relative_path: str) -> ArtifactRecord:
    """Hash an artifact that already exists on disk [AUTH: 01 §16, §19]."""
    target = _contained(root, relative_path)
    if not target.is_file():
        raise ArtifactVerificationError(f"artifact absent: {relative_path}")
    return ArtifactRecord(
        path=relative_path, sha256=sha256_file(target), size_bytes=target.stat().st_size
    )


def verify_artifact(root: Path, record: ArtifactRecord) -> None:
    """Recompute and compare. Mutation, truncation and deletion all raise."""
    target = _contained(root, record.path)
    if not target.is_file():
        raise ArtifactVerificationError(f"artifact absent: {record.path}")
    actual = sha256_file(target)
    if actual != record.sha256:
        raise ArtifactVerificationError(
            f"artifact hash mismatch: {record.path}"
            f" (recorded {record.sha256[:12]}, found {actual[:12]})"
        )


def verify_artifacts(root: Path, records: Sequence[ArtifactRecord]) -> list[str]:
    """Non-raising bulk form: every problem, not just the first."""
    problems: list[str] = []
    for record in records:
        try:
            verify_artifact(root, record)
        except ArtifactVerificationError as exc:
            problems.append(str(exc))
    return problems


def _contained(root: Path, relative_path: str) -> Path:
    """Resolve under `root`, refusing absolute paths and `..` escapes."""
    if not relative_path or Path(relative_path).is_absolute():
        raise ArtifactVerificationError(f"artifact path must be repo-relative: {relative_path!r}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactVerificationError(
            f"artifact path escapes the run root: {relative_path!r}"
        ) from exc
    return candidate
