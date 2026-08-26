"""D/E/F — masking, reference-calibrated loss, Min-K% [AUTH: 00 §14.1, §14.2]."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring.reference import (
    ScoreFamilyError,
    TokenEvidence,
    min_k_percent_score,
    negative_log_likelihood,
    reference_calibrated_score,
)


def _evidence(values: list[float], valid: list[bool] | None = None) -> TokenEvidence:
    return TokenEvidence(
        log_probs=np.array(values, dtype=np.float64),
        valid=np.array([True] * len(values) if valid is None else valid, dtype=np.bool_),
    )


# ------------------------------------------------------------------ D: masking
def test_masked_tokens_are_excluded_from_the_reduction() -> None:
    """Valid tokens are -1 and -3; padding is -100. The mean NLL is 2, by hand."""
    evidence = _evidence([-1.0, -100.0, -3.0], [True, False, True])
    assert evidence.token_count == 2
    assert negative_log_likelihood(evidence) == pytest.approx(2.0)


def test_an_empty_valid_set_fails_closed() -> None:
    evidence = _evidence([-1.0, -2.0], [False, False])
    with pytest.raises(ScoreFamilyError, match="no valid tokens"):
        negative_log_likelihood(evidence)


def test_a_non_finite_token_fails_closed() -> None:
    with pytest.raises(ScoreFamilyError, match="non-finite"):
        negative_log_likelihood(_evidence([-1.0, float("-inf")]))


def test_shape_disagreement_fails_closed() -> None:
    with pytest.raises(ScoreFamilyError, match="disagree"):
        TokenEvidence(log_probs=np.array([-1.0, -2.0]), valid=np.array([True], dtype=np.bool_))


# ------------------------------------------------------------------ E: reference loss
def test_reference_score_is_reference_minus_target() -> None:
    """l_ref = 4, l_target = 1, so s_ref = 3 [AUTH: 00 §14.1]."""
    target = _evidence([-1.0, -1.0])
    reference = _evidence([-4.0, -4.0])
    assert reference_calibrated_score(target, reference) == pytest.approx(3.0)


def test_a_member_scores_higher_than_a_non_member() -> None:
    """Direction check: the model fits the member better, so the member's score is larger."""
    reference = _evidence([-4.0, -4.0])
    member = reference_calibrated_score(_evidence([-1.0, -1.0]), reference)
    non_member = reference_calibrated_score(_evidence([-3.9, -3.9]), reference)
    assert member > non_member


def test_incompatible_token_sets_are_never_averaged() -> None:
    """Different valid masks mean the two averages cover different tokens; the difference
    would be meaningless, so it fails closed [AUTH: 00 §14.1]."""
    target = _evidence([-1.0, -1.0, -1.0], [True, True, False])
    reference = _evidence([-4.0, -4.0, -4.0], [True, False, True])
    with pytest.raises(ScoreFamilyError, match="incompatible token sets"):
        reference_calibrated_score(target, reference)


def test_different_token_counts_are_refused() -> None:
    with pytest.raises(ScoreFamilyError, match="different token counts"):
        reference_calibrated_score(_evidence([-1.0]), _evidence([-1.0, -1.0]))


# ------------------------------------------------------------------ F: Min-K%
def test_min_k_averages_the_lowest_fraction() -> None:
    """Ten tokens 0..-9; the bottom 20% are -9 and -8, mean -8.5, computed by hand."""
    evidence = _evidence([-float(i) for i in range(10)])
    assert min_k_percent_score(evidence, 0.2) == pytest.approx(-8.5)


def test_min_k_selects_at_least_one_token() -> None:
    evidence = _evidence([-1.0, -2.0, -3.0])
    assert min_k_percent_score(evidence, 0.01) == pytest.approx(-3.0)


def test_min_k_respects_the_mask() -> None:
    evidence = _evidence([-1.0, -99.0, -2.0], [True, False, True])
    assert min_k_percent_score(evidence, 1.0) == pytest.approx(-1.5)


def test_min_k_direction_favours_members() -> None:
    """A member's least likely tokens are still comparatively likely, so its score is larger."""
    member = _evidence([-1.0, -1.2, -1.4, -1.6, -1.8])
    non_member = _evidence([-3.0, -3.2, -3.4, -3.6, -3.8])
    assert min_k_percent_score(member, 0.4) > min_k_percent_score(non_member, 0.4)


def test_min_k_is_deterministic_under_ties() -> None:
    evidence = _evidence([-2.0, -2.0, -2.0, -1.0])
    assert min_k_percent_score(evidence, 0.5) == min_k_percent_score(evidence, 0.5)
    assert min_k_percent_score(evidence, 0.5) == pytest.approx(-2.0)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_an_out_of_range_fraction_is_refused(fraction: float) -> None:
    with pytest.raises(ScoreFamilyError, match="Min-K fraction"):
        min_k_percent_score(_evidence([-1.0]), fraction)


def test_full_fraction_is_the_plain_mean() -> None:
    values = [-1.0, -2.0, -3.0, -4.0]
    evidence = _evidence(values)
    assert min_k_percent_score(evidence, 1.0) == pytest.approx(float(np.mean(values)))
