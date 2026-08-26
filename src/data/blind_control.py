"""The authoritative blind split / deduplication leakage decision [AUTH: 00 §6.4].

`blind.py` owns the estimator — TF-IDF 1-2 grams, L2 logistic regression, stratified CV,
Mann-Whitney AUROC. This module owns the *decision*, which 00 §6.4 specifies in full and
which is not derivable from a point estimate alone.

The rule, verbatim in structure:

* the initial run is 5-fold stratified CV at seed 6104 over NATURAL_MEMBER_EVAL against
  EVAL_NONMEMBERS, with balanced labels;
* INVESTIGATION triggers if the point estimate leaves [0.47, 0.53] **or** the 95% CI excludes
  0.5 — either one is sufficient, so a tight interval away from 0.5 is not clean;
* investigation runs all five steps: the confirmatory CV at seed 6105, the cross-partition
  exact-hash audit, the frozen near-duplicate audit, the record-length comparison and the
  publication-month comparison;
* STOP_NATURAL_PRIVACY_ARM only when a concrete leakage path is found, or when BOTH fixed CV
  runs land outside the band on the same side. Anything else retains the arm and reports the
  investigation.

00 §6.4 does not fix a CI estimator, so its algorithm and parameters are frozen in
`configs/data/corpus.json` before any real corpus is opened. The interval is a TWO-SAMPLE
record bootstrap: members and non-members are unpaired populations and are resampled
independently from domain-separated streams. Block B's `resampling.bootstrap` resamples one
unit list under one index draw per replicate, which cannot express that design, so the draw
is written here — it is not a second path for any Block B estimator, and the percentile
convention is bound to Block B's by test.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.data.blind import (
    BlindControlError,
    BlindControlResult,
    area_under_curve,
    run_blind_control,
)
from src.data.dedup import jaccard, shingles
from src.data.normalise import NormalisedRecord
from src.provenance.hashing import JSONValue

CONTROLLER_VERSION: Final = "s05.blind-control-decision.v2"

#: The keyed-stream scheme for the two-sample CI, and the two class domain separators.
CI_STREAM_SCHEME: Final = "s05.blind-ci.two-sample.v1"
MEMBER_DOMAIN: Final = "blind_ci_member"
NON_MEMBER_DOMAIN: Final = "blind_ci_nonmember"

CLEAN: Final = "CLEAN"
INVESTIGATION: Final = "BLIND_SPLIT_INVESTIGATION"
STOP: Final = "STOP_NATURAL_PRIVACY_ARM"

#: The one frozen rule for unequal label counts. 00 §6.4 requires balanced labels, and
#: NATURAL_MEMBER_EVAL and EVAL_NONMEMBERS are equal by construction, so an unequal input
#: means something upstream is wrong. The controller refuses rather than quietly evaluating
#: an imbalanced problem; `balance_for_control` is the explicit, recorded alternative.
LABEL_BALANCE_RULE: Final = "REFUSE_UNEQUAL_LABEL_COUNTS"

#: The five investigation steps 00 §6.4 lists, in order.
INVESTIGATION_STEPS: Final[tuple[str, ...]] = (
    "CONFIRMATORY_CV",
    "CROSS_PARTITION_EXACT_HASH_AUDIT",
    "NEAR_DUPLICATE_AUDIT",
    "RECORD_LENGTH_COMPARISON",
    "PUBLICATION_MONTH_COMPARISON",
)


@dataclass(frozen=True)
class LabelledRecord:
    """One record as the controller sees it: text, identity, and its acquisition metadata."""

    record_id: str
    text: str
    text_sha256: str
    publication_date: str


@dataclass(frozen=True)
class BlindControlDecision:
    """The 00 §6.4 verdict, every input that produced it, and why it fired."""

    status: str
    initial: BlindControlResult
    confidence_interval: tuple[float, float]
    ci_method: str
    ci_replicates: int
    ci_seed: int
    band: tuple[float, float]
    triggers: tuple[str, ...]
    confirmatory: BlindControlResult | None
    investigation: Mapping[str, JSONValue]
    label_balance_rule: str = LABEL_BALANCE_RULE
    controller_version: str = CONTROLLER_VERSION

    @property
    def ci_excludes_half(self) -> bool:
        low, high = self.confidence_interval
        return low > 0.5 or high < 0.5

    @property
    def outside_band(self) -> bool:
        low, high = self.band
        return not low <= self.initial.auroc <= high

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "status": self.status,
            "controller_version": self.controller_version,
            "auroc": self.initial.auroc,
            "cv_seed": self.initial.cv_seed,
            "confidence_interval": list(self.confidence_interval),
            "ci_method": self.ci_method,
            "ci_replicates": self.ci_replicates,
            "ci_seed": self.ci_seed,
            "band": list(self.band),
            "triggers": list(self.triggers),
            "label_balance_rule": self.label_balance_rule,
            "confirmatory_auroc": None if self.confirmatory is None else self.confirmatory.auroc,
            "investigation": dict(self.investigation),
        }


def balance_for_control(
    members: Sequence[LabelledRecord], non_members: Sequence[LabelledRecord], *, seed: int
) -> tuple[list[LabelledRecord], list[LabelledRecord], int]:
    """Deterministically trim the larger class, and say how many records were dropped.

    Explicit and recorded, never automatic: `blind_control_decision` refuses unequal input,
    so a caller who balances has to do it here and carry the count into the report.
    """
    import hashlib

    def trim(records: Sequence[LabelledRecord], size: int) -> list[LabelledRecord]:
        keyed = sorted(
            records,
            key=lambda record: hashlib.sha256(f"{seed}|{record.record_id}".encode()).hexdigest(),
        )
        return keyed[:size]

    size = min(len(members), len(non_members))
    dropped = abs(len(members) - len(non_members))
    return trim(members, size), trim(non_members, size), dropped


def _class_stream(seed: int, domain: str) -> np.random.Generator:
    """A per-class RNG whose stream is keyed on (ci_seed, domain separator).

    Domain separation rather than one shared generator: it makes accidental recoupling of the
    two classes impossible to write, and it puts the separation in provenance where a reader
    can see it.
    """
    key = hashlib.sha256(f"{CI_STREAM_SCHEME}|{seed}|{domain}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(key[:8], "big")))


def _draw(rng: np.random.Generator, n: int) -> NDArray[np.int_]:
    """`n` indices with replacement, using Block B's index convention [AUTH: 00 §32.1].

    The formula is `src.analysis.resampling._indices`, replicated rather than imported
    because that name is private to accepted Block B code. `test_the_index_draw_matches_the
    _block_b_convention` binds the two.
    """
    return np.floor(rng.random(n) * n).astype(np.int_).clip(0, n - 1)


def auroc_confidence_interval(
    member_scores: Sequence[float],
    non_member_scores: Sequence[float],
    *,
    n_replicates: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the blind-control AUROC — a TWO-SAMPLE design.

    Members and non-members are unpaired populations, so each replicate draws them
    independently: `n_member` members with replacement from the members, `n_non_member`
    non-members with replacement from the non-members, from two domain-separated streams.
    Reusing one index vector across both classes would invent a pairing that does not exist
    and would make the interval depend on the order the rows happened to arrive in.

    Each class is canonicalised to sorted order before resampling. AUROC is a rank statistic
    over score multisets, so sorting discards nothing, and it is what makes the interval
    invariant to row order: the same two multisets give the same interval under the same
    frozen CI seed, however the callers happened to list them.
    """
    members = np.sort(np.asarray(member_scores, dtype=np.float64))
    non_members = np.sort(np.asarray(non_member_scores, dtype=np.float64))
    if members.size == 0 or non_members.size == 0:
        raise BlindControlError("a confidence interval needs both classes")
    if n_replicates < 1:
        raise BlindControlError("n_replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise BlindControlError(f"alpha {alpha!r} is not in (0, 1)")

    member_rng = _class_stream(seed, MEMBER_DOMAIN)
    non_member_rng = _class_stream(seed, NON_MEMBER_DOMAIN)
    labels = np.concatenate(
        [
            np.ones(members.size, dtype=np.int_),
            np.zeros(non_members.size, dtype=np.int_),
        ]
    )
    draws = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        member_index = _draw(member_rng, members.size)
        non_member_index = _draw(non_member_rng, non_members.size)
        scores = np.concatenate([members[member_index], non_members[non_member_index]])
        draws[replicate] = area_under_curve(scores, labels)

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise BlindControlError("every bootstrap replicate was non-finite")
    return (
        float(np.percentile(finite, 100.0 * alpha / 2.0)),
        float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0))),
    )


def cross_partition_exact_hash_audit(
    partitions: Mapping[str, Sequence[LabelledRecord]],
) -> list[dict[str, JSONValue]]:
    """00 §6.4 investigation step 2: the same normalised text in two partitions."""
    seen: dict[str, tuple[str, str]] = {}
    findings: list[dict[str, JSONValue]] = []
    for partition in sorted(partitions):
        for record in sorted(partitions[partition], key=lambda item: item.record_id):
            owner = seen.get(record.text_sha256)
            if owner is None:
                seen[record.text_sha256] = (partition, record.record_id)
            elif owner[0] != partition:
                findings.append(
                    {
                        "text_sha256": record.text_sha256,
                        "left_partition": owner[0],
                        "left_id": owner[1],
                        "right_partition": partition,
                        "right_id": record.record_id,
                    }
                )
    return findings


def near_duplicate_audit(
    members: Sequence[LabelledRecord],
    non_members: Sequence[LabelledRecord],
    *,
    shingle_size: int,
    threshold: float,
) -> list[dict[str, JSONValue]]:
    """00 §6.4 investigation step 3, using the frozen near-duplicate rule from `dedup.py`."""
    member_shingles = [(record, shingles(record.text, shingle_size)) for record in members]
    findings: list[dict[str, JSONValue]] = []
    for other in non_members:
        other_shingles = shingles(other.text, shingle_size)
        for record, current in member_shingles:
            similarity = jaccard(current, other_shingles)
            if similarity >= threshold:
                findings.append(
                    {
                        "member_id": record.record_id,
                        "non_member_id": other.record_id,
                        "similarity": similarity,
                    }
                )
    return findings


def record_length_comparison(
    members: Sequence[LabelledRecord], non_members: Sequence[LabelledRecord]
) -> dict[str, JSONValue]:
    """00 §6.4 investigation step 4. Reported, never a pass/fail criterion on its own."""
    member_lengths = np.array([len(record.text) for record in members], dtype=np.float64)
    non_member_lengths = np.array([len(record.text) for record in non_members], dtype=np.float64)
    return {
        "member_median_chars": float(np.median(member_lengths)),
        "non_member_median_chars": float(np.median(non_member_lengths)),
        "median_difference": float(np.median(member_lengths) - np.median(non_member_lengths)),
    }


def publication_month_comparison(
    members: Sequence[LabelledRecord], non_members: Sequence[LabelledRecord]
) -> dict[str, JSONValue]:
    """00 §6.4 investigation step 5: month-level composition on each side."""

    def months(records: Sequence[LabelledRecord]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            try:
                month = _dt.date.fromisoformat(record.publication_date).strftime("%Y-%m")
            except ValueError as exc:
                raise BlindControlError(
                    f"{record.record_id}: publication_date {record.publication_date!r} is not"
                    " an ISO date"
                ) from exc
            counts[month] = counts.get(month, 0) + 1
        return dict(sorted(counts.items()))

    member_months, non_member_months = months(members), months(non_members)
    every = sorted(set(member_months) | set(non_member_months))
    return {
        "member_months": member_months,
        "non_member_months": non_member_months,
        "max_absolute_share_difference": max(
            (
                abs(
                    member_months.get(month, 0) / max(len(members), 1)
                    - non_member_months.get(month, 0) / max(len(non_members), 1)
                )
                for month in every
            ),
            default=0.0,
        ),
    }


def _same_direction(first: float, second: float, band: tuple[float, float]) -> bool:
    low, high = band
    return (first > high and second > high) or (first < low and second < low)


def stop_required(
    *,
    initial_auroc: float,
    confirmatory_auroc: float,
    band: tuple[float, float],
    concrete_leakage: bool,
) -> bool:
    """The 00 §6.4 STOP rule, as a pure function.

    STOP_NATURAL_PRIVACY_ARM only when a concrete duplicate/preprocessing leakage path was
    found, OR both fixed CV runs land outside the band on the same side. A single run outside
    the band, or two runs outside it in opposite directions, is investigated and reported —
    the arm is retained.
    """
    return concrete_leakage or _same_direction(initial_auroc, confirmatory_auroc, band)


def blind_control_decision(
    *,
    members: Sequence[LabelledRecord],
    non_members: Sequence[LabelledRecord],
    other_partitions: Mapping[str, Sequence[LabelledRecord]] | None = None,
    cv_seed: int,
    confirmatory_seed: int,
    n_folds: int,
    clean_tolerance: float,
    band: tuple[float, float],
    regularisation: float,
    ngram_range: tuple[int, int],
    ci_method: str,
    ci_replicates: int,
    ci_seed: int,
    ci_alpha: float,
    shingle_size: int,
    near_duplicate_threshold: float,
) -> BlindControlDecision:
    """The complete 00 §6.4 decision. The only function that may declare the split CLEAN."""
    if len(members) != len(non_members):
        raise BlindControlError(
            f"{LABEL_BALANCE_RULE}: the blind control requires balanced labels and received"
            f" {len(members)} member(s) against {len(non_members)} non-member(s);"
            " call balance_for_control explicitly and record what it dropped [AUTH: 00 §6.4]"
        )
    if ci_method != "PERCENTILE_BOOTSTRAP_OVER_RECORDS":
        raise BlindControlError(
            f"{ci_method!r} is not the frozen blind-control CI method; the estimator and its"
            " parameters are frozen in config before any real corpus is opened"
        )

    member_texts = [record.text for record in members]
    non_member_texts = [record.text for record in non_members]
    initial = run_blind_control(
        member_texts=member_texts,
        non_member_texts=non_member_texts,
        cv_seed=cv_seed,
        n_folds=n_folds,
        clean_tolerance=clean_tolerance,
        investigation_band=band,
        regularisation=regularisation,
        ngram_range=ngram_range,
    )
    member_scores, non_member_scores = _out_of_fold_scores(
        member_texts,
        non_member_texts,
        cv_seed=cv_seed,
        n_folds=n_folds,
        regularisation=regularisation,
        ngram_range=ngram_range,
    )
    interval = auroc_confidence_interval(
        member_scores,
        non_member_scores,
        n_replicates=ci_replicates,
        seed=ci_seed,
        alpha=ci_alpha,
    )

    triggers: list[str] = []
    low, high = band
    if not low <= initial.auroc <= high:
        triggers.append(f"point estimate {initial.auroc:.4f} is outside {band}")
    if interval[0] > 0.5 or interval[1] < 0.5:
        triggers.append(f"95% CI {interval[0]:.4f}-{interval[1]:.4f} excludes 0.5")

    if not triggers:
        return BlindControlDecision(
            status=CLEAN,
            initial=initial,
            confidence_interval=interval,
            ci_method=ci_method,
            ci_replicates=ci_replicates,
            ci_seed=ci_seed,
            band=band,
            triggers=(),
            confirmatory=None,
            investigation={},
        )

    confirmatory = run_blind_control(
        member_texts=member_texts,
        non_member_texts=non_member_texts,
        cv_seed=confirmatory_seed,
        n_folds=n_folds,
        clean_tolerance=clean_tolerance,
        investigation_band=band,
        regularisation=regularisation,
        ngram_range=ngram_range,
    )
    partitions: dict[str, Sequence[LabelledRecord]] = {
        "NATURAL_MEMBER_EVAL": members,
        "EVAL_NONMEMBERS": non_members,
        **dict(other_partitions or {}),
    }
    exact = cross_partition_exact_hash_audit(partitions)
    near = near_duplicate_audit(
        members, non_members, shingle_size=shingle_size, threshold=near_duplicate_threshold
    )
    investigation: dict[str, JSONValue] = {
        "steps": list(INVESTIGATION_STEPS),
        "CONFIRMATORY_CV": {"cv_seed": confirmatory.cv_seed, "auroc": confirmatory.auroc},
        "CROSS_PARTITION_EXACT_HASH_AUDIT": list(exact),
        "NEAR_DUPLICATE_AUDIT": list(near),
        "RECORD_LENGTH_COMPARISON": record_length_comparison(members, non_members),
        "PUBLICATION_MONTH_COMPARISON": publication_month_comparison(members, non_members),
    }

    concrete_leakage = bool(exact) or bool(near)
    stop = stop_required(
        initial_auroc=initial.auroc,
        confirmatory_auroc=confirmatory.auroc,
        band=band,
        concrete_leakage=concrete_leakage,
    )
    if concrete_leakage:
        investigation["stop_reason"] = "CONCRETE_DUPLICATE_OR_PREPROCESSING_LEAKAGE_PATH"
    elif stop:
        investigation["stop_reason"] = "BOTH_FIXED_CV_RUNS_OUTSIDE_BAND_IN_THE_SAME_DIRECTION"
    status = STOP if stop else INVESTIGATION
    return BlindControlDecision(
        status=status,
        initial=initial,
        confidence_interval=interval,
        ci_method=ci_method,
        ci_replicates=ci_replicates,
        ci_seed=ci_seed,
        band=band,
        triggers=tuple(triggers),
        confirmatory=confirmatory,
        investigation=investigation,
    )


def _out_of_fold_scores(
    member_texts: Sequence[str],
    non_member_texts: Sequence[str],
    *,
    cv_seed: int,
    n_folds: int,
    regularisation: float,
    ngram_range: tuple[int, int],
) -> tuple[list[float], list[float]]:
    """The same out-of-fold scores `run_blind_control` derives its AUROC from.

    Recomputed here rather than returned from `blind.py`, because `blind.py` is the accepted
    estimator surface and this module must not change it. Both use the identical fold
    assignment, so the AUROC of these scores is the AUROC the result reports — bound by test.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

    from src.data.blind import _stratified_folds

    texts = list(member_texts) + list(non_member_texts)
    labels = np.array([1] * len(member_texts) + [0] * len(non_member_texts), dtype=np.int_)
    folds = _stratified_folds(labels, n_folds=n_folds, seed=cv_seed)
    out_of_fold = np.zeros(labels.size, dtype=np.float64)
    for fold in range(n_folds):
        test = folds == fold
        train = ~test
        vectoriser = TfidfVectorizer(ngram_range=ngram_range, lowercase=True)
        features = vectoriser.fit_transform([texts[i] for i in np.flatnonzero(train)])
        model = LogisticRegression(l1_ratio=0.0, C=regularisation, solver="lbfgs", max_iter=1000)
        model.fit(features, labels[train])
        held = vectoriser.transform([texts[i] for i in np.flatnonzero(test)])
        out_of_fold[test] = model.decision_function(held)
    return (
        [float(v) for v in out_of_fold[: len(member_texts)]],
        [float(v) for v in out_of_fold[len(member_texts) :]],
    )


def labelled(records: Sequence[NormalisedRecord], *, publication_date: str) -> list[LabelledRecord]:
    """Adapter for fixtures and for corpora whose dates travel separately."""
    return [
        LabelledRecord(
            record_id=record.record_id,
            text=record.text,
            text_sha256=record.text_sha256,
            publication_date=publication_date,
        )
        for record in records
    ]
