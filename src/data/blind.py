"""Blind split / deduplication leakage control [AUTH: 00 §6.4].

Before any model-based natural-record privacy result is interpreted, this asks whether the
fixed member/non-member split is distinguishable **from text alone**. It sees no model, no
score and no privacy outcome — only the corpus text and the split assignment.

The frozen construction is 00 §6.4's: TF-IDF over word 1-2 grams, L2-regularised logistic
regression, balanced labels, 5-fold stratified CV, AUROC, first seed 6104 and confirmatory
seed 6105, with the clean criterion |AUROC - 0.5| <= 0.03.

AUROC is computed from the Mann-Whitney rank identity, not from a threshold sweep. That is
deliberate: 00 §34A.2 and 02 §C2 give `src/scoring/roc.py` sole ownership of operating-point
selection, and this diagnostic must not become a second one. A rank statistic chooses no
threshold and reports no TPR at any operating point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

CONTROL_VERSION: Final = "s05.blind-split.v1"

Verdict = Literal["CLEAN", "BLIND_SPLIT_INVESTIGATION"]


class BlindControlError(ValueError):
    """The blind control cannot be run on the inputs given."""


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        mask = inverse == index
        ranks[mask] = ranks[mask].mean()
    return ranks


def area_under_curve(scores: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    """AUROC by the Mann-Whitney identity, tie-corrected.

        AUROC = (R+ - n+(n+ + 1)/2) / (n+ * n-)

    where R+ is the sum of the average ranks of the positive class. No threshold is chosen and
    no operating point is reported [AUTH: 00 §16 "AUC as secondary"; 02 §C2].
    """
    if scores.shape != labels.shape:
        raise BlindControlError("scores and labels disagree in length")
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise BlindControlError("the blind control needs both classes")
    ranks = _average_ranks(scores.astype(np.float64))
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


@dataclass(frozen=True)
class BlindControlResult:
    """One CV seed's text-only separability, and the frozen verdict it implies."""

    cv_seed: int
    auroc: float
    n_member: int
    n_non_member: int
    clean_tolerance: float
    investigation_band: tuple[float, float]
    version: str = CONTROL_VERSION

    @property
    def is_clean(self) -> bool:
        return abs(self.auroc - 0.5) <= self.clean_tolerance

    @property
    def verdict(self) -> Verdict:
        low, high = self.investigation_band
        return "CLEAN" if low <= self.auroc <= high else "BLIND_SPLIT_INVESTIGATION"


def run_blind_control(
    *,
    member_texts: Sequence[str],
    non_member_texts: Sequence[str],
    cv_seed: int,
    n_folds: int,
    clean_tolerance: float,
    investigation_band: tuple[float, float],
    regularisation: float,
    ngram_range: tuple[int, int] = (1, 2),
) -> BlindControlResult:
    """Stratified out-of-fold text-only AUROC [AUTH: 00 §6.4].

    Every fold fits the vectoriser and the classifier on its training folds only, so the
    held-out documents contribute no vocabulary or IDF weight to the model that scores them.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

    if not member_texts or not non_member_texts:
        raise BlindControlError("the blind control needs both classes")
    if n_folds < 2:
        raise BlindControlError("stratified CV needs at least two folds")

    texts = list(member_texts) + list(non_member_texts)
    labels = np.array([1] * len(member_texts) + [0] * len(non_member_texts), dtype=np.int_)
    folds = _stratified_folds(labels, n_folds=n_folds, seed=cv_seed)

    out_of_fold = np.zeros(labels.size, dtype=np.float64)
    for fold in range(n_folds):
        test = folds == fold
        train = ~test
        if len(set(labels[train].tolist())) < 2:
            raise BlindControlError(f"fold {fold}: a training split carries one class only")
        vectoriser = TfidfVectorizer(ngram_range=ngram_range, lowercase=True)
        features = vectoriser.fit_transform([texts[i] for i in np.flatnonzero(train)])
        model = LogisticRegression(l1_ratio=0.0, C=regularisation, solver="lbfgs", max_iter=1000)
        model.fit(features, labels[train])
        held = vectoriser.transform([texts[i] for i in np.flatnonzero(test)])
        out_of_fold[test] = model.decision_function(held)

    return BlindControlResult(
        cv_seed=cv_seed,
        auroc=area_under_curve(out_of_fold, labels),
        n_member=len(member_texts),
        n_non_member=len(non_member_texts),
        clean_tolerance=clean_tolerance,
        investigation_band=investigation_band,
    )


def _stratified_folds(labels: NDArray[np.int_], *, n_folds: int, seed: int) -> NDArray[np.int_]:
    """Balanced fold assignment, deterministic from `seed`."""
    assignment = np.empty(labels.size, dtype=np.int_)
    rng = np.random.Generator(np.random.PCG64(seed))
    for value in (0, 1):
        indices = np.flatnonzero(labels == value)
        order = indices[np.argsort(rng.random(indices.size), kind="stable")]
        assignment[order] = np.arange(order.size) % n_folds
    return assignment


def duplicate_leakage_across_partitions(
    partition_hashes: dict[str, dict[str, str]],
) -> list[tuple[str, str, str, str]]:
    """Exact-hash collisions across partitions: (hash, left, left_id, right_id).

    This is the audit 00 §6.4's investigation step 2 calls for, and it is a text-only check —
    it never touches a model or a score.
    """
    seen: dict[str, tuple[str, str]] = {}
    findings: list[tuple[str, str, str, str]] = []
    for partition in sorted(partition_hashes):
        for record_id, digest in sorted(partition_hashes[partition].items()):
            owner = seen.get(digest)
            if owner is None:
                seen[digest] = (partition, record_id)
                continue
            if owner[0] != partition:
                findings.append((digest, owner[0], owner[1], record_id))
    return findings
