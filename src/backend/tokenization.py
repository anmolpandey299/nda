"""Tokenizer identity and the frozen token semantics [AUTH: 02 §C7; 00 §14.1, §14.2].

A tokenizer is not an implementation detail of scoring: it decides which tokens exist, so a
different tokenizer revision or special-token configuration produces different per-token
log-probabilities for the same record. 02 §C7 therefore requires it in the cache identity,
and a silent default drift here would serve one model's scores for another's tokenization.

Every material token decision is resolved from config and hashed together with the tokenizer's
own revision and config digest, so changing any of them moves the identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.provenance.hashing import JSONValue, sha256_canonical

TOKENIZATION_SCHEMA: Final = "backend.tokenization.v1"


class TokenizationError(ValueError):
    """The token semantics are not resolvable or not the frozen ones."""


@dataclass(frozen=True)
class TokenizerIdentity:
    """Everything that changes what a token sequence is [AUTH: 02 §C7]."""

    tokenizer_revision: str
    tokenizer_config_sha256: str
    special_tokens_sha256: str
    max_sequence_length: int
    truncation_policy: str
    padding_policy: str
    loss_mask_policy: str
    causal_shift: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": TOKENIZATION_SCHEMA,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "special_tokens_sha256": self.special_tokens_sha256,
            "max_sequence_length": self.max_sequence_length,
            "truncation_policy": self.truncation_policy,
            "padding_policy": self.padding_policy,
            "loss_mask_policy": self.loss_mask_policy,
            "causal_shift": self.causal_shift,
        }

    def identity(self) -> str:
        """The `tokenizer_sha256` the scoring cache identity binds [AUTH: 02 §C7]."""
        return sha256_canonical(self.as_dict())


def resolve_tokenizer_identity(
    root: Path,
    *,
    tokenizer_revision: str,
    tokenizer_config_sha256: str,
    special_tokens: dict[str, JSONValue],
    max_sequence_length: int,
) -> TokenizerIdentity:
    """Bind the resolved token policy to one tokenizer's own identity."""
    from src.backend.settings import backend_settings
    from src.materials import material_text

    if max_sequence_length < 1:
        raise TokenizationError("max_sequence_length must be positive")
    backend = backend_settings(root)
    return TokenizerIdentity(
        tokenizer_revision=tokenizer_revision,
        tokenizer_config_sha256=tokenizer_config_sha256,
        special_tokens_sha256=sha256_canonical(dict(special_tokens)),
        max_sequence_length=max_sequence_length,
        truncation_policy=material_text(backend, "truncation_policy"),
        padding_policy=material_text(backend, "padding_policy"),
        loss_mask_policy=material_text(backend, "loss_mask_policy"),
        causal_shift=material_text(backend, "causal_shift"),
    )
