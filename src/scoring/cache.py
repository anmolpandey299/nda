"""Content-addressed scoring cache. Filesystem only — no service, no database.

02 §C7: the cache identity must change whenever executable scoring semantics change, and a
hand-typed `scorer_version` string is not sufficient. SCORING_CODE_HASH is therefore
computed from the scoring source itself plus the resolved scoring config, and the full
Block A environment identity is bound in as well, so a score produced under a different
environment or a different implementation can never be served [AUTH: 02 §C7; 00 §34A.4].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from src.provenance.hashing import (
    JSONDocument,
    JSONValue,
    canonical_json_bytes,
    read_canonical_json,
    sha256_bytes,
    sha256_canonical,
    sha256_file,
    write_canonical_json,
)

#: Scoring source whose bytes define executable scoring semantics [AUTH: 02 §C7].
SCORING_SOURCE_FILES: Final[tuple[str, ...]] = (
    "src/scoring/roc.py",
    "src/scoring/reference.py",
    "src/scoring/engine.py",
    "src/scoring/crossfit.py",
    "src/scoring/pooled.py",
    "src/scoring/cache.py",
    "src/scoring/backends.py",
)

CACHE_SCHEMA: Final = "s03.score-cache.v2"

#: The only fields a cache entry may carry. Everything else is attribution belonging to the
#: request, and is rebuilt per call so it can never be inherited [AUTH: 00 §34A.1].
CACHE_PAYLOAD_FIELDS: Final[tuple[str, ...]] = ("record_id", "scalar_score", "token_count")


class CacheError(RuntimeError):
    """The cache refused to serve or store an entry."""


def scoring_code_hash(root: Path, resolved_scoring_config: JSONDocument) -> str:
    """SHA256 over the scoring source plus the resolved scoring config [AUTH: 02 §C7]."""
    parts: list[str] = []
    for relative in SCORING_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise CacheError(f"scoring source missing, so its identity is unknown: {relative}")
        parts.append(f"{relative}={sha256_file(path)}")
    parts.append(f"config={sha256_canonical(dict(resolved_scoring_config))}")
    return sha256_bytes("\n".join(parts).encode("utf-8"))


@dataclass(frozen=True)
class CacheIdentity:
    """Every input that changes what a cached score *means* [AUTH: 02 §C7; 00 §34A.4].

    `score_family` and `reference_backend_identity` are part of the identity because they
    change the number: REFERENCE_LOSS and MIN_K reduce the same token evidence differently,
    and the reference-calibrated score is a difference against a specific reference model.
    Two requests that differ only in those inputs are different measurements, not a hit.

    Attribution — arm, view, seed, fold, membership — is deliberately NOT here. It does not
    change the scalar, so binding it would fragment the cache; instead the payload carries no
    attribution at all and every row is rebuilt from the current request.
    """

    artifact_id: str
    model_revision: str
    tokenizer_sha256: str
    record_set_sha256: str
    precision: str
    max_sequence_length: int
    scoring_code_hash: str
    backend_identity: str
    reference_backend_identity: str
    score_family: str
    resolved_config_sha256: str
    environment_lock_sha256: str

    def as_dict(self) -> dict[str, JSONValue]:
        return dict(asdict(self))

    def key(self) -> str:
        return sha256_canonical({"schema": CACHE_SCHEMA, "identity": self.as_dict()})


@dataclass(frozen=True)
class ScoreCache:
    """One directory of content-addressed entries. Regenerable, never evidence."""

    root: Path

    @property
    def directory(self) -> Path:
        return self.root / "artifacts" / "cache"

    def path_for(self, identity: CacheIdentity) -> Path:
        return self.directory / f"{identity.key()}.json"

    def store(self, identity: CacheIdentity, rows: Sequence[Mapping[str, JSONValue]]) -> str:
        """Persist the canonical score payload. Attribution fields are refused.

        A cached row that carried `seed` or `fold` could hand a later request another
        request's attribution, so the payload is restricted to what the scoring semantics
        actually produced [AUTH: 00 §34A.1; 02 §C7].
        """
        for row in rows:
            intruders = sorted(set(row) - set(CACHE_PAYLOAD_FIELDS))
            if intruders:
                raise CacheError(
                    f"attribution field(s) {intruders} may not be cached; the cache stores"
                    " scoring output only"
                )
        payload: list[JSONValue] = [dict(row) for row in rows]
        document: dict[str, JSONValue] = {
            "schema": CACHE_SCHEMA,
            "identity": identity.as_dict(),
            "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
            "rows": payload,
        }
        write_canonical_json(self.path_for(identity), document)
        return identity.key()

    def load(self, identity: CacheIdentity) -> list[dict[str, JSONValue]] | None:
        """Return the cached rows, or None on a miss. Corruption raises rather than serves."""
        path = self.path_for(identity)
        if not path.is_file():
            return None
        document = read_canonical_json(path)
        if not isinstance(document, Mapping):
            raise CacheError(f"{path.name}: cache entry is not a JSON object")
        if document.get("schema") != CACHE_SCHEMA:
            raise CacheError(f"{path.name}: unknown cache schema {document.get('schema')!r}")
        if document.get("identity") != identity.as_dict():
            # A key collision or a hand-edited entry: the stored identity must be the one
            # asked for, not merely hash to the same filename.
            raise CacheError(f"{path.name}: stored identity does not match the requested one")
        rows = document.get("rows")
        if not isinstance(rows, list):
            raise CacheError(f"{path.name}: cache entry carries no rows")
        materialised = [dict(row) for row in rows]
        recomputed = sha256_bytes(canonical_json_bytes(materialised))
        if recomputed != document.get("payload_sha256"):
            raise CacheError(f"{path.name}: cached payload is corrupt")
        return materialised
