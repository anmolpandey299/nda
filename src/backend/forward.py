"""Causal-LM forward semantics: alignment, masking and token-level log-probabilities.

The backend's whole job for scoring is to hand the accepted S03 scorer correctly aligned
per-token log-probabilities and a validity mask. It computes **no score**: the reference-loss
reduction and the Min-K% reduction both live in `src.scoring.reference` and stay the only
implementations of those semantics [AUTH: 00 §34A.1, §34A.2; 02 §C6].

Three things are easy to get wrong and are therefore done in one place, on plain arrays, so a
CPU fixture exercises exactly the code a GPU run does:

* **the causal shift** — position t's logits predict token t+1, so log-probabilities are read
  at `logits[:-1]` against `labels[1:]`. An unshifted read scores each token against itself
  and inflates every likelihood;
* **the loss mask** — padding never enters a reduction, and neither does the unshiftable first
  position [AUTH: 00 §14.2]. A prompt mask, when present, excludes prompt tokens so no prompt
  likelihood leaks into a continuation score;
* **the reduction dtype** — log-softmax accumulates in float64 regardless of the forward
  precision, because a derived statistic is not artifact bytes [AUTH: 01 §10].
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from src.scoring.reference import TokenEvidence

FloatArray = NDArray[Any]
IntArray = NDArray[Any]
BoolArray = NDArray[Any]

FORWARD_VERSION: Final = "backend.causal-forward.v1"

#: The label value that marks a position excluded from every reduction, as HF uses it.
IGNORE_INDEX: Final = -100


class ForwardError(ValueError):
    """A forward pass cannot be reduced to token evidence."""


def log_softmax(logits: FloatArray) -> FloatArray:
    """Numerically stable log-softmax over the vocabulary axis, in float64."""
    wide = np.asarray(logits, dtype=np.float64)
    if wide.ndim != 2:
        raise ForwardError("logits must be (positions, vocabulary)")
    shifted = wide - np.max(wide, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def token_evidence_from_logits(
    logits: FloatArray,
    labels: IntArray,
    *,
    attention_mask: IntArray | None = None,
    prompt_mask: BoolArray | None = None,
) -> TokenEvidence:
    """Per-token log-probabilities for one sequence, causally shifted and masked.

    `labels` are the *unshifted* input ids. `attention_mask` marks real (non-padding) tokens.
    `prompt_mask` marks positions belonging to a prompt, which are excluded so a continuation
    score carries no prompt likelihood [AUTH: 00 §14.1].
    """
    values = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(labels)
    if values.ndim != 2:
        raise ForwardError("logits must be (positions, vocabulary)")
    if ids.ndim != 1:
        raise ForwardError("labels must be one-dimensional")
    if values.shape[0] != ids.shape[0]:
        raise ForwardError(
            f"logits cover {values.shape[0]} positions but labels cover {ids.shape[0]}"
        )
    if values.shape[0] < 2:
        raise ForwardError(
            "a sequence shorter than two tokens has no predicted position; the causal shift"
            " leaves nothing to score"
        )
    if not np.all(np.isfinite(values)):
        raise ForwardError("the forward pass produced non-finite logits")

    # position t predicts token t+1
    predicted = log_softmax(values[:-1, :])
    targets = ids[1:].astype(np.int64)
    if np.any(targets >= predicted.shape[1]) or np.any((targets < 0) & (targets != IGNORE_INDEX)):
        raise ForwardError("a label id lies outside the vocabulary")

    safe = np.where(targets == IGNORE_INDEX, 0, targets)
    log_probs = predicted[np.arange(predicted.shape[0]), safe]

    valid = targets != IGNORE_INDEX
    if attention_mask is not None:
        mask = np.asarray(attention_mask).astype(bool)
        if mask.shape != ids.shape:
            raise ForwardError("the attention mask does not cover the label positions")
        valid &= mask[1:]
    if prompt_mask is not None:
        prompt = np.asarray(prompt_mask).astype(bool)
        if prompt.shape != ids.shape:
            raise ForwardError("the prompt mask does not cover the label positions")
        valid &= ~prompt[1:]

    log_probs = np.where(valid, log_probs, 0.0)
    return TokenEvidence(log_probs=log_probs.astype(np.float64), valid=valid.astype(np.bool_))


def evidence_identity(evidence: TokenEvidence) -> str:
    """A content hash over the token evidence, for cross-run determinism checks."""
    import hashlib

    digest = hashlib.sha256(FORWARD_VERSION.encode())
    digest.update(np.ascontiguousarray(evidence.log_probs, dtype="<f8").tobytes("C"))
    digest.update(np.ascontiguousarray(evidence.valid, dtype=np.uint8).tobytes("C"))
    return digest.hexdigest()
