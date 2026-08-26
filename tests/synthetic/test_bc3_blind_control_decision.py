"""B-C3 — the complete 00 §6.4 blind split / deduplication decision.

Two triggers, five investigation steps, and a STOP rule that is narrower than either.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from blockc_fixtures import synthetic_text, tiny_corpus

from src.data.blind import BlindControlError
from src.data.blind_control import (
    CLEAN,
    INVESTIGATION,
    INVESTIGATION_STEPS,
    LABEL_BALANCE_RULE,
    STOP,
    BlindControlDecision,
    LabelledRecord,
    auroc_confidence_interval,
    balance_for_control,
    blind_control_decision,
    cross_partition_exact_hash_audit,
    labelled,
    near_duplicate_audit,
    publication_month_comparison,
    record_length_comparison,
    stop_required,
)
from src.data.corpus import build_master_corpus
from src.data.settings import corpus_settings
from src.materials import material, material_integer, material_number, material_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = corpus_settings(REPO_ROOT)
BAND = (0.47, 0.53)


def decide(
    members: Sequence[LabelledRecord],
    non_members: Sequence[LabelledRecord],
    *,
    ci_replicates: int = 400,
    **overrides: Any,
) -> BlindControlDecision:
    kwargs: dict[str, Any] = {
        "members": list(members),
        "non_members": list(non_members),
        "cv_seed": material_integer(SETTINGS, "blind_control_cv_seed"),
        "confirmatory_seed": material_integer(SETTINGS, "blind_control_confirmatory_seed"),
        "n_folds": material_integer(SETTINGS, "blind_control_folds"),
        "clean_tolerance": material_number(SETTINGS, "blind_control_clean_tolerance"),
        "band": BAND,
        "regularisation": material_number(
            SETTINGS, "blind_control_regularisation", allow_provisional=True
        ),
        "ngram_range": (1, 2),
        "ci_method": material_text(SETTINGS, "blind_control_ci_method"),
        "ci_replicates": ci_replicates,
        "ci_seed": material_integer(SETTINGS, "blind_control_ci_seed"),
        "ci_alpha": material_number(SETTINGS, "blind_control_ci_alpha"),
        "shingle_size": material_integer(
            SETTINGS, "near_duplicate_shingle_size", allow_provisional=True
        ),
        "near_duplicate_threshold": material_number(
            SETTINGS, "near_duplicate_threshold", allow_provisional=True
        ),
    }
    kwargs.update(overrides)
    return blind_control_decision(**kwargs)


@functools.lru_cache(maxsize=4)
def clean_pair(count: int = 400) -> tuple[tuple[LabelledRecord, ...], tuple[LabelledRecord, ...]]:
    """A deduplicated, randomly assigned split — what the real pipeline hands the control.

    The records go through `build_master_corpus` rather than being sliced directly, so the
    pair carries no near-duplicate the frozen rule would have removed upstream. The default
    size is 400 a side because the 00 §6.4 band is ±0.03, which is narrower than the sampling
    noise of a much smaller split: a genuinely random 150-a-side split lands outside the band
    on noise alone. The real control runs at 5,000 a side.
    """
    corpus = build_master_corpus(
        tiny_corpus(3 * count),
        sizes={
            "TRAIN_CANDIDATES": count,
            "CALIBRATION_NONMEMBERS": count // 2,
            "EVAL_NONMEMBERS": count,
            "RESERVE": count // 4,
        },
        split_seed=20260820,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    by_id = {record.record_id: record for record in corpus.retained}
    members = [by_id[rid] for rid in corpus.partitions["TRAIN_CANDIDATES"]]
    non_members = [by_id[rid] for rid in corpus.partitions["EVAL_NONMEMBERS"]]
    return (
        tuple(labelled(members, publication_date="2026-05-14")),
        tuple(labelled(non_members, publication_date="2026-05-20")),
    )


# ------------------------------------------------------------------ the frozen CI estimator
def test_the_ci_algorithm_and_parameters_are_frozen_in_config() -> None:
    assert material_text(SETTINGS, "blind_control_ci_method") == "PERCENTILE_BOOTSTRAP_OVER_RECORDS"
    assert material_integer(SETTINGS, "blind_control_ci_replicates") == 2000
    assert material_integer(SETTINGS, "blind_control_ci_seed") == 6106
    assert material_number(SETTINGS, "blind_control_ci_alpha") == 0.05
    assert material(SETTINGS, "blind_control_label_balance_rule") == LABEL_BALANCE_RULE


def test_an_unfrozen_ci_method_is_refused() -> None:
    members, non_members = clean_pair(40)
    with pytest.raises(BlindControlError, match="frozen blind-control CI method"):
        decide(members, non_members, ci_method="EYEBALL")


def test_the_interval_is_deterministic_and_brackets_the_estimate() -> None:
    members = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    non_members = [0.4, 0.3, 0.2, 0.1, 0.0, -0.1]
    first = auroc_confidence_interval(members, non_members, n_replicates=500, seed=1, alpha=0.05)
    assert first == auroc_confidence_interval(
        members, non_members, n_replicates=500, seed=1, alpha=0.05
    )
    assert first[0] <= first[1]


def test_the_ci_preserves_each_class_sample_size() -> None:
    """C3: independent resampling handles unequal classes; the balance rule is the controller's."""
    low, high = auroc_confidence_interval(
        [0.9, 0.8, 0.7], [0.3, 0.2], n_replicates=200, seed=1, alpha=0.05
    )
    assert 0.0 <= low <= high <= 1.0


def test_the_ci_still_needs_both_classes() -> None:
    with pytest.raises(BlindControlError, match="needs both classes"):
        auroc_confidence_interval([0.1, 0.2], [], n_replicates=100, seed=1, alpha=0.05)


# ------------------------------------------------------------------ the two triggers
def test_a_tight_interval_away_from_one_half_investigates_even_inside_the_band() -> None:
    """Codex counterexample: AUROC = .52 with CI = (.515, .525) must never be CLEAN."""
    auroc, interval = 0.52, (0.515, 0.525)
    assert BAND[0] <= auroc <= BAND[1], "the point estimate is inside the band"

    class _Result:
        cv_seed = 6104

    decision = BlindControlDecision(
        status=INVESTIGATION,
        initial=_Result(),  # type: ignore[arg-type]
        confidence_interval=interval,
        ci_method="PERCENTILE_BOOTSTRAP_OVER_RECORDS",
        ci_replicates=2000,
        ci_seed=6106,
        band=BAND,
        triggers=("95% CI excludes 0.5",),
        confirmatory=None,
        investigation={},
    )
    assert decision.ci_excludes_half
    assert decision.status != CLEAN


def test_the_controller_fires_on_the_ci_alone() -> None:
    """The same case, driven through the real controller on a planted separable split."""
    members = [
        LabelledRecord(
            record_id=f"M{index}",
            text=f"{synthetic_text(index)[1]} cohort marker {index % 3}",
            text_sha256=f"m{index}",
            publication_date="2026-05-14",
        )
        for index in range(120)
    ]
    non_members = [
        LabelledRecord(
            record_id=f"N{index}",
            text=synthetic_text(1000 + index)[1],
            text_sha256=f"n{index}",
            publication_date="2026-05-20",
        )
        for index in range(120)
    ]
    decision = decide(list(members), list(non_members))
    assert decision.status in {INVESTIGATION, STOP}
    assert decision.triggers
    assert decision.confirmatory is not None


def test_a_random_split_is_called_clean() -> None:
    members, non_members = clean_pair()
    decision = decide(list(members), list(non_members))
    assert decision.status == CLEAN, (decision.initial.auroc, decision.confidence_interval)
    assert decision.triggers == ()
    assert decision.confirmatory is None
    assert not decision.ci_excludes_half


def test_only_the_controller_may_declare_a_split_clean() -> None:
    members, non_members = clean_pair()
    decision = decide(list(members), list(non_members))
    assert decision.as_dict()["status"] == CLEAN
    assert decision.as_dict()["ci_method"] == "PERCENTILE_BOOTSTRAP_OVER_RECORDS"


# ------------------------------------------------------------------ balanced labels
def test_unequal_label_counts_are_refused_by_the_frozen_rule() -> None:
    members, non_members = clean_pair(60)
    with pytest.raises(BlindControlError, match=LABEL_BALANCE_RULE):
        decide(list(members), list(non_members[:50]))


def test_balancing_is_explicit_deterministic_and_reports_what_it_dropped() -> None:
    members, non_members = clean_pair(60)
    trimmed_members, trimmed_non, dropped = balance_for_control(
        list(members), list(non_members[:50]), seed=20260820
    )
    assert len(trimmed_members) == len(trimmed_non) == 50
    assert dropped == 10
    again = balance_for_control(list(members), list(non_members[:50]), seed=20260820)
    assert [r.record_id for r in again[0]] == [r.record_id for r in trimmed_members]
    decide(trimmed_members, trimmed_non)


# ------------------------------------------------------------------ the five steps
def test_every_investigation_step_runs_when_one_fires() -> None:
    members = [
        LabelledRecord(
            record_id=f"M{index}",
            text=f"MEMBER-COHORT-MARKER {synthetic_text(index)[1]}",
            text_sha256=f"m{index}",
            publication_date="2026-05-14",
        )
        for index in range(100)
    ]
    non_members = [
        LabelledRecord(
            record_id=f"N{index}",
            text=synthetic_text(2000 + index)[1],
            text_sha256=f"n{index}",
            publication_date="2026-06-14",
        )
        for index in range(100)
    ]
    decision = decide(list(members), list(non_members))
    assert decision.status in {INVESTIGATION, STOP}
    for step in INVESTIGATION_STEPS:
        assert step in decision.investigation, step
    assert decision.investigation["steps"] == list(INVESTIGATION_STEPS)


def test_the_exact_hash_audit_finds_a_shared_record() -> None:
    findings = cross_partition_exact_hash_audit(
        {
            "TRAIN": [LabelledRecord("A1", "a", "h1", "2026-05-01")],
            "EVAL": [LabelledRecord("B1", "a", "h1", "2026-05-01")],
        }
    )
    assert len(findings) == 1 and findings[0]["text_sha256"] == "h1"


def test_the_near_duplicate_audit_uses_the_frozen_rule() -> None:
    text = tiny_corpus(1)[0].text
    findings = near_duplicate_audit(
        [LabelledRecord("M1", text, "h1", "2026-05-01")],
        [LabelledRecord("N1", text + " one extra clause", "h2", "2026-05-01")],
        shingle_size=5,
        threshold=0.6,
    )
    assert len(findings) == 1 and findings[0]["member_id"] == "M1"


def test_the_length_and_month_comparisons_are_reported() -> None:
    members = [LabelledRecord("M1", "short", "h1", "2026-05-01")]
    non_members = [LabelledRecord("N1", "a much longer record body", "h2", "2026-07-01")]
    lengths = record_length_comparison(members, non_members)
    assert float(str(lengths["member_median_chars"])) < float(
        str(lengths["non_member_median_chars"])
    )
    months = publication_month_comparison(members, non_members)
    assert months["member_months"] == {"2026-05": 1}
    assert months["non_member_months"] == {"2026-07": 1}
    assert months["max_absolute_share_difference"] == 1.0


def test_a_malformed_publication_date_fails_the_month_comparison() -> None:
    with pytest.raises(BlindControlError, match="not an ISO date"):
        publication_month_comparison(
            [LabelledRecord("M1", "x", "h1", "whenever")],
            [LabelledRecord("N1", "y", "h2", "2026-07-01")],
        )


# ------------------------------------------------------------------ the STOP rule
def test_a_concrete_duplicate_leakage_path_stops_the_arm() -> None:
    shared = tiny_corpus(1)[0].text
    members = [
        LabelledRecord(f"M{i}", f"MEMBER-MARK {synthetic_text(i)[1]}", f"h{i}", "2026-05-14")
        for i in range(60)
    ]
    non_members = [
        LabelledRecord(f"N{i}", synthetic_text(3000 + i)[1], f"n{i}", "2026-05-20")
        for i in range(59)
    ]
    members[0] = LabelledRecord("M0", shared, "shared", "2026-05-14")
    non_members.append(LabelledRecord("N59", shared, "shared", "2026-05-20"))
    decision = decide(list(members), list(non_members))
    assert decision.status == STOP
    assert decision.investigation["stop_reason"] == (
        "CONCRETE_DUPLICATE_OR_PREPROCESSING_LEAKAGE_PATH"
    )
    assert decision.investigation["CROSS_PARTITION_EXACT_HASH_AUDIT"]


def test_both_runs_outside_the_band_in_the_same_direction_stops_the_arm() -> None:
    members = [
        LabelledRecord(
            f"M{i}", f"MEMBER-COHORT-MARKER {synthetic_text(i)[1]}", f"h{i}", "2026-05-14"
        )
        for i in range(100)
    ]
    non_members = [
        LabelledRecord(f"N{i}", synthetic_text(4000 + i)[1], f"n{i}", "2026-05-20")
        for i in range(100)
    ]
    decision = decide(list(members), list(non_members))
    assert decision.status == STOP
    assert decision.initial.auroc > BAND[1]
    assert decision.confirmatory is not None and decision.confirmatory.auroc > BAND[1]
    assert decision.investigation["stop_reason"] == (
        "BOTH_FIXED_CV_RUNS_OUTSIDE_BAND_IN_THE_SAME_DIRECTION"
    )


@pytest.mark.parametrize(
    ("initial", "confirmatory", "leakage", "expected"),
    [
        (0.60, 0.61, False, True),  # both high -> STOP
        (0.40, 0.39, False, True),  # both low  -> STOP
        (0.60, 0.40, False, False),  # opposite directions -> retain and investigate
        (0.60, 0.50, False, False),  # only one outside -> retain and investigate
        (0.50, 0.50, True, True),  # a concrete path stops regardless of the estimates
        (0.50, 0.50, False, False),  # neither -> retain
    ],
)
def test_the_stop_rule_is_narrower_than_the_investigation_trigger(
    initial: float, confirmatory: float, leakage: bool, expected: bool
) -> None:
    """00 §6.4: INVESTIGATION is not STOP. Uncertainty is reported; the arm survives."""
    assert (
        stop_required(
            initial_auroc=initial,
            confirmatory_auroc=confirmatory,
            band=BAND,
            concrete_leakage=leakage,
        )
        is expected
    )


def test_an_investigated_but_retained_split_records_no_stop_reason() -> None:
    members, non_members = clean_pair()
    decision = decide(list(members), list(non_members), band=(0.499, 0.501))
    assert decision.triggers
    assert not decision.investigation["CROSS_PARTITION_EXACT_HASH_AUDIT"]
    assert not decision.investigation["NEAR_DUPLICATE_AUDIT"]
    if decision.status == INVESTIGATION:
        assert "stop_reason" not in decision.investigation
    else:
        assert decision.investigation["stop_reason"] == (
            "BOTH_FIXED_CV_RUNS_OUTSIDE_BAND_IN_THE_SAME_DIRECTION"
        )
