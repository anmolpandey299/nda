"""ONE deterministic text normalisation. Its output is what every hash is taken over.

The normalised text is the identity of a record: exact deduplication, the split hashes and
the data manifest all hash this string [AUTH: 00 §6.2; 01 §14]. So the transform is
conservative and fully enumerated — it repairs encoding and whitespace, and it does nothing
semantic. Aggressive rewriting (case folding, punctuation stripping, stemming) would silently
merge records that are not duplicates, which is a leakage path, not a cleanup.

The version string is part of the data manifest. Changing any step here requires bumping it,
because a record's identity would otherwise move without its provenance moving.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.provenance.hashing import sha256_bytes

#: Bumped whenever any step below changes [AUTH: 01 §14 "preprocessing-code commit"].
NORMALISATION_VERSION: Final = "s05.normalise.v1"

#: The frozen field order used when a record is assembled from parts. Order is part of the
#: identity, so it is declared rather than left to dict iteration.
FIELD_ORDER: Final[tuple[str, ...]] = ("title", "abstract")

_ZERO_WIDTH: Final = re.compile(r"[​‌‍﻿]")
_HORIZONTAL_WS: Final = re.compile(r"[^\S\r\n]+")
_NEWLINES: Final = re.compile(r"\r\n|\r")
_BLANK_RUN: Final = re.compile(r"\n{2,}")


class NormalisationError(ValueError):
    """A record cannot be normalised into an identity-bearing string."""


@dataclass(frozen=True)
class NormalisedRecord:
    """One record reduced to its canonical text and the hash taken over it."""

    record_id: str
    text: str
    text_sha256: str
    normalisation_version: str = NORMALISATION_VERSION

    @property
    def is_empty(self) -> bool:
        return self.text == ""


def normalise_text(value: str) -> str:
    """The frozen transform. Steps are listed in the order they are applied.

    1. Unicode NFKC, so visually identical strings share one encoding;
    2. zero-width and BOM characters removed — they are invisible and would defeat exact
       deduplication;
    3. CRLF/CR collapsed to LF;
    4. runs of horizontal whitespace collapsed to a single space, per line;
    5. trailing/leading whitespace stripped from every line;
    6. runs of newlines collapsed to a single newline, so a blank line is not part of a
       record's identity;
    7. the whole string stripped.

    Case, punctuation, digits and word forms are left exactly as published.
    """
    if not isinstance(value, str):
        raise NormalisationError(f"expected text, got {type(value).__name__}")
    text = unicodedata.normalize("NFKC", value)
    text = _ZERO_WIDTH.sub("", text)
    text = _NEWLINES.sub("\n", text)
    lines = [_HORIZONTAL_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_RUN.sub("\n", text)
    return text.strip()


def combine_fields(record: Mapping[str, str]) -> str:
    """Assemble the declared fields in the frozen order, skipping absent ones."""
    parts = [normalise_text(record[name]) for name in FIELD_ORDER if name in record]
    return "\n".join(part for part in parts if part)


def normalise_record(record_id: str, record: Mapping[str, str]) -> NormalisedRecord:
    """Normalise one record and take its identity hash.

    An empty result fails closed: a record that normalises to nothing has no identity, and
    letting it through would make every such record an exact duplicate of every other.
    """
    if not record_id:
        raise NormalisationError("a record needs an id")
    text = combine_fields(record)
    if text == "":
        raise NormalisationError(f"{record_id}: normalises to empty text, so it has no identity")
    return NormalisedRecord(
        record_id=record_id, text=text, text_sha256=sha256_bytes(text.encode("utf-8"))
    )


def normalise_corpus(
    records: Sequence[tuple[str, Mapping[str, str]]],
) -> list[NormalisedRecord]:
    """Normalise a whole corpus, refusing duplicate record ids."""
    seen: set[str] = set()
    out: list[NormalisedRecord] = []
    for record_id, record in records:
        if record_id in seen:
            raise NormalisationError(f"duplicate record id: {record_id}")
        seen.add(record_id)
        out.append(normalise_record(record_id, record))
    return out
