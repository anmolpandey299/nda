"""The production scoring backend [AUTH: 00 §34A.1; 02 §C6, §C7].

There is one scorer. `src.scoring.engine` owns cross-fitting and attribution,
`src.scoring.reference` owns the reference-loss and Min-K% reductions, and neither is
duplicated here — this module implements only the `ScoringBackend` protocol those components
already consume: given an artifact and a record, return correctly aligned per-token
log-probabilities and a validity mask.

That boundary is what keeps the estimand single. A backend that reduced tokens itself would
be a second Min-K implementation and a second reference-loss implementation, which 00 §34A.2
prohibits outright, and no test of the accepted scorer would cover it.

The backend's `identity` is a content hash over the executable backend semantics — its own
source, the model revision, the tokenizer identity and the resolved precision — so 02 §C7's
cache requirement is met by construction rather than by a hand-typed version string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.backend.forward import token_evidence_from_logits
from src.backend.tokenization import TokenizerIdentity
from src.provenance.hashing import JSONValue, sha256_canonical, sha256_file
from src.scoring.reference import TokenEvidence

BACKEND_SCHEMA: Final = "backend.scoring.v1"

#: The one truncation policy this backend implements [AUTH: 02 §C7].
RIGHT_TRUNCATE: Final = "RIGHT_TRUNCATE_TO_MAX_SEQUENCE_LENGTH"

#: Backend source whose bytes define executable scoring semantics [AUTH: 02 §C7].
BACKEND_SOURCE_FILES: Final[tuple[str, ...]] = (
    "src/backend/forward.py",
    "src/backend/loader.py",
    "src/backend/scoring_backend.py",
    "src/backend/tokenization.py",
    "src/backend/adapters.py",
)


class ScoringBackendError(RuntimeError):
    """The production scoring backend cannot serve token evidence."""


def backend_code_hash(root: Path) -> str:
    """SHA256 over the backend source AND the resolved runtime config [AUTH: 02 §C7].

    A manually typed version string is explicitly insufficient, so the identity is taken over
    the bytes: editing the causal shift, the mask policy or the loader changes it whether or
    not anyone remembers to bump a constant.

    The resolved runtime document is hashed in alongside, because a material setting change —
    the attention implementation, the forward precision, `trust_remote_code`, the LoRA scaling
    rule, `use_rslora` — changes what the backend computes without touching a single source
    byte. Covering only the source would let exactly those changes serve a stale score.
    """
    from src.backend.settings import backend_runtime_config_sha256

    parts: list[str] = []
    for relative in BACKEND_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ScoringBackendError(
                f"backend source missing, so its identity is unknown: {relative}"
            )
        parts.append(f"{relative}={sha256_file(path)}")
    return sha256_canonical(
        {
            "schema": BACKEND_SCHEMA,
            "sources": parts,
            "runtime_config_sha256": backend_runtime_config_sha256(root),
        }
    )


@dataclass(frozen=True)
class TokenBatch:
    """One record's tokenization, as the backend hands it to a forward pass."""

    record_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    prompt_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.attention_mask):
            raise ScoringBackendError(f"{self.record_id}: ids and attention mask disagree")
        if self.prompt_mask is not None and len(self.prompt_mask) != len(self.input_ids):
            raise ScoringBackendError(f"{self.record_id}: the prompt mask does not cover the ids")

    def right_truncated(self, limit: int) -> TokenBatch:
        """RIGHT_TRUNCATE_TO_MAX_SEQUENCE_LENGTH: keep the left prefix, drop the right tail.

        Every aligned structure is cut at the same index — ids, attention mask and prompt
        mask — so no downstream reduction can be handed a mask that describes a different
        sequence than the one scored [AUTH: 00 §14.2; 02 §C7].
        """
        if limit < 1:
            raise ScoringBackendError(f"{self.record_id}: max_sequence_length must be positive")
        if len(self.input_ids) <= limit:
            return self
        return TokenBatch(
            record_id=self.record_id,
            input_ids=self.input_ids[:limit],
            attention_mask=self.attention_mask[:limit],
            prompt_mask=None if self.prompt_mask is None else self.prompt_mask[:limit],
        )


class LogitsSource:
    """What the backend needs from a model: logits for one tokenized record.

    A protocol rather than a concrete class so the H100 path and a deterministic CPU fixture
    are the same code from the scorer's point of view. The fixture is not a mock of the
    reduction — the reduction is the accepted scorer's, unchanged either way.
    """

    def logits_for(self, *, artifact_id: str, batch: TokenBatch) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover - implementations override


@dataclass(frozen=True)
class HuggingFaceScoringBackend:
    """The production `ScoringBackend`. Token execution only, no reduction."""

    root: Path
    model_revision: str
    tokenizer: TokenizerIdentity
    precision: str
    source: LogitsSource
    batches: Mapping[str, TokenBatch]

    def truncate(self, batch: TokenBatch) -> TokenBatch:
        """Apply the resolved truncation policy. An unknown policy fails closed."""
        policy = self.tokenizer.truncation_policy
        if policy != RIGHT_TRUNCATE:
            raise ScoringBackendError(
                f"truncation policy {policy!r} is not implemented by this backend; the"
                f" resolved policy must be one the backend executes (expected {RIGHT_TRUNCATE})"
                " [AUTH: 02 §C7]"
            )
        return batch.right_truncated(self.tokenizer.max_sequence_length)

    @property
    def identity(self) -> str:
        """A cache-key input covering the executable backend semantics [AUTH: 02 §C7].

        `backend_code_sha256` already spans both the backend source and the resolved runtime
        document, so a change to either moves this identity.
        """
        return sha256_canonical(
            {
                "schema": BACKEND_SCHEMA,
                "backend_code_sha256": backend_code_hash(self.root),
                "model_revision": self.model_revision,
                "tokenizer_sha256": self.tokenizer.identity(),
                "precision": self.precision,
            }
        )

    def token_evidence(self, *, artifact_id: str, record_id: str) -> TokenEvidence:
        """Aligned per-token log-probabilities. The scorer does the reducing.

        The resolved truncation policy is EXECUTED here, not merely recorded: a config that
        declares `RIGHT_TRUNCATE_TO_MAX_SEQUENCE_LENGTH` while the backend refused over-length
        input would be provenance describing something that never happens [AUTH: 02 §C7].
        """
        batch = self.batches.get(record_id)
        if batch is None:
            raise ScoringBackendError(f"{record_id!r} was not tokenized for this run")
        batch = self.truncate(batch)
        logits = self.source.logits_for(artifact_id=artifact_id, batch=batch)
        return token_evidence_from_logits(
            np.asarray(logits, dtype=np.float64),
            np.asarray(batch.input_ids, dtype=np.int64),
            attention_mask=np.asarray(batch.attention_mask, dtype=np.int64),
            prompt_mask=None
            if batch.prompt_mask is None
            else np.asarray(batch.prompt_mask, dtype=bool),
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": BACKEND_SCHEMA,
            "backend_identity": self.identity,
            "backend_code_sha256": backend_code_hash(self.root),
            "model_revision": self.model_revision,
            "tokenizer_sha256": self.tokenizer.identity(),
            "precision": self.precision,
            "n_records": len(self.batches),
        }


@dataclass(frozen=True)
class TorchLogitsSource(LogitsSource):
    """Real HF causal-LM execution. The only torch-touching path in the scoring lane."""

    model: Any
    device: str

    def logits_for(self, *, artifact_id: str, batch: TokenBatch) -> np.ndarray:
        del artifact_id
        import torch  # noqa: PLC0415

        with torch.no_grad():
            ids = torch.tensor([batch.input_ids], dtype=torch.long, device=self.device)
            mask = torch.tensor([batch.attention_mask], dtype=torch.long, device=self.device)
            output = self.model(input_ids=ids, attention_mask=mask)
        # float64 on the host: the reduction is a statistic, not artifact bytes [01 §10]
        return np.asarray(output.logits[0].to(torch.float32).cpu().numpy(), dtype=np.float64)


def scoring_source_identity(root: Path, relatives: Sequence[str]) -> str:
    """Helper for the cache: a stable digest over a named set of source files."""
    return sha256_canonical(
        {"sources": [f"{name}={sha256_file(root / name)}" for name in relatives]}
    )
