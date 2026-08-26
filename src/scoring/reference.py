"""Score families: canonical masking/reduction, reference-calibrated loss, Min-K%.

Both families read the *same* token-validity mask and the same reduction, so a difference
between them is a difference in the statistic and never in what was measured
[AUTH: 00 §14.1, §14.2, §34A.1].

Directions, stated once:

REFERENCE_LOSS   s_ref(x; V) = l_ref(x) - l_V(x). A member is fitted better by V, so l_V is
                 smaller and the score is larger. Higher = more member-like.
MIN_K            mean of the lowest-K% valid token log-probabilities. A member's least
                 likely tokens are still comparatively likely, so the mean is larger.
                 Higher = more member-like.

Both therefore feed `src/scoring/roc.py` without re-orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class ScoreFamilyError(ValueError):
    """A record cannot be scored under the frozen masking/reduction convention."""


@dataclass(frozen=True)
class TokenEvidence:
    """Per-token log-probabilities for one record, with the frozen validity mask.

    `valid` excludes padding and every ignored position. Nothing else may be excluded, and
    nothing excluded here may re-enter a reduction [AUTH: 00 §14.2].
    """

    log_probs: FloatArray
    valid: BoolArray

    def __post_init__(self) -> None:
        if self.log_probs.ndim != 1 or self.valid.ndim != 1:
            raise ScoreFamilyError("token evidence must be one-dimensional")
        if self.log_probs.shape != self.valid.shape:
            raise ScoreFamilyError(
                f"log_probs {self.log_probs.shape} and valid {self.valid.shape} disagree"
            )

    @property
    def token_count(self) -> int:
        return int(np.count_nonzero(self.valid))

    def valid_log_probs(self) -> FloatArray:
        """Fails closed on an empty valid set: a mean over nothing is not a measurement."""
        selected = self.log_probs[self.valid]
        if selected.size == 0:
            raise ScoreFamilyError("record has no valid tokens after masking")
        if not np.isfinite(selected).all():
            raise ScoreFamilyError("record has non-finite token log-probabilities")
        return selected.astype(np.float64)


def negative_log_likelihood(evidence: TokenEvidence) -> float:
    """Token-level average NLL over valid tokens — the frozen reduction [AUTH: 00 §14.1]."""
    return float(-np.mean(evidence.valid_log_probs()))


def reference_calibrated_score(target: TokenEvidence, reference: TokenEvidence) -> float:
    """s_ref = l_ref - l_target [AUTH: 00 §14.1].

    The two evidences must cover the same token positions. Averaging a target over one token
    set and a reference over another silently changes what the difference means, so a
    mismatch fails closed rather than being averaged away.
    """
    if target.valid.shape != reference.valid.shape:
        raise ScoreFamilyError(
            "target and reference cover different token counts; the record identity or"
            " tokenization disagrees"
        )
    if not np.array_equal(target.valid, reference.valid):
        raise ScoreFamilyError(
            "target and reference disagree about which tokens are valid; refusing to average"
            " incompatible token sets [AUTH: 00 §14.1]"
        )
    return negative_log_likelihood(reference) - negative_log_likelihood(target)


def min_k_percent_score(evidence: TokenEvidence, fraction: float) -> float:
    """Mean of the lowest-`fraction` valid token log-probabilities [AUTH: 00 §14.2].

    `fraction` comes from resolved config, never from a source literal [AUTH: 01 §17].
    Selection uses a stable ascending sort, so tied log-probabilities resolve in token order
    and the result is reproducible. At least one token is always selected.

    Only this scalar is returned; per-token arrays are reduced inline and never persisted
    [AUTH: 00 §14.2 implementation/storage rule].
    """
    if not 0.0 < fraction <= 1.0:
        raise ScoreFamilyError(f"Min-K fraction {fraction!r} is not in (0, 1]")
    selected = evidence.valid_log_probs()
    count = max(1, math.ceil(fraction * selected.size))
    order = np.argsort(selected, kind="stable")
    return float(np.mean(selected[order[:count]]))
