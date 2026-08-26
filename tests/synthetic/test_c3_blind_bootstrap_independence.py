"""C3-FINAL-1 — the blind-control CI is a TWO-SAMPLE bootstrap over unpaired classes.

The defect: one index vector applied to both member and non-member scores. That invents a
pairing between two independent populations and makes the interval depend on the order the
rows arrived in, which moved the 00 §6.4 investigation trigger.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from src.data.blind import BlindControlError, area_under_curve
from src.data.blind_control import (
    CI_STREAM_SCHEME,
    MEMBER_DOMAIN,
    NON_MEMBER_DOMAIN,
    _class_stream,
    _draw,
    auroc_confidence_interval,
)

CI_SEED = 6106
REPLICATES = 800


def population(n_member: int = 150, n_non_member: int = 150) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(20260826)
    return (
        [float(v) for v in rng.normal(0.06, 1.0, n_member)],
        [float(v) for v in rng.normal(0.0, 1.0, n_non_member)],
    )


def point_auroc(members: Sequence[float], non_members: Sequence[float]) -> float:
    return area_under_curve(
        np.array([*members, *non_members], dtype=np.float64),
        np.array([1] * len(members) + [0] * len(non_members), dtype=np.int_),
    )


def interval(members: Sequence[float], non_members: Sequence[float]) -> tuple[float, float]:
    return auroc_confidence_interval(
        list(members), list(non_members), n_replicates=REPLICATES, seed=CI_SEED, alpha=0.05
    )


# ------------------------------------------------------------------ regression 1: order
def test_reversing_the_non_member_rows_changes_nothing() -> None:
    """The executed counterexample from review, as a regression."""
    members, non_members = population()
    assert point_auroc(members, non_members) == point_auroc(members, list(reversed(non_members)))
    assert interval(members, non_members) == interval(members, list(reversed(non_members)))


def test_reversing_the_member_rows_changes_nothing() -> None:
    members, non_members = population()
    assert point_auroc(members, non_members) == point_auroc(list(reversed(members)), non_members)
    assert interval(members, non_members) == interval(list(reversed(members)), non_members)


def test_a_random_permutation_of_either_class_changes_nothing() -> None:
    members, non_members = population()
    rng = np.random.default_rng(11)
    for _ in range(3):
        shuffled_members = [members[i] for i in rng.permutation(len(members))]
        shuffled_non = [non_members[i] for i in rng.permutation(len(non_members))]
        assert interval(shuffled_members, shuffled_non) == interval(members, non_members)
        assert point_auroc(shuffled_members, shuffled_non) == point_auroc(members, non_members)


def test_the_investigation_decision_is_order_invariant() -> None:
    """The trigger, not just the numbers: the same split cannot flip on row order."""
    members, non_members = population()
    for candidate in (non_members, list(reversed(non_members))):
        low, high = interval(members, candidate)
        assert (low > 0.5 or high < 0.5) is (
            interval(members, non_members)[0] > 0.5 or interval(members, non_members)[1] < 0.5
        )


# ------------------------------------------------------------------ regression 2: mutation
def _coupled_interval(
    members: Sequence[float], non_members: Sequence[float], *, seed: int, n_replicates: int
) -> tuple[float, float]:
    """The PREVIOUS defect, reintroduced deliberately: one index vector for both classes."""
    member_array = np.asarray(members, dtype=np.float64)
    non_member_array = np.asarray(non_members, dtype=np.float64)
    labels = np.concatenate(
        [
            np.ones(member_array.size, dtype=np.int_),
            np.zeros(non_member_array.size, dtype=np.int_),
        ]
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        shared = np.floor(rng.random(member_array.size) * member_array.size).astype(np.int_)
        scores = np.concatenate([member_array[shared], non_member_array[shared]])
        draws[replicate] = area_under_curve(scores, labels)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def test_the_coupled_index_mutation_is_order_dependent() -> None:
    """Kill the mutation: with one shared index vector, row order moves the interval."""
    members, non_members = population()
    forward = _coupled_interval(members, non_members, seed=CI_SEED, n_replicates=REPLICATES)
    reversed_ = _coupled_interval(
        members, list(reversed(non_members)), seed=CI_SEED, n_replicates=REPLICATES
    )
    assert forward != reversed_, "the coupled mutation should have been order dependent"

    fixed_forward = interval(members, non_members)
    fixed_reversed = interval(members, list(reversed(non_members)))
    assert fixed_forward == fixed_reversed


def test_the_coupled_index_mutation_gives_a_different_interval() -> None:
    members, non_members = population()
    assert _coupled_interval(
        members, non_members, seed=CI_SEED, n_replicates=REPLICATES
    ) != interval(members, non_members)


def test_the_two_class_streams_are_independent() -> None:
    """Domain separation: the two draws are different sequences, by construction."""
    assert len({MEMBER_DOMAIN, NON_MEMBER_DOMAIN}) == 2
    member = _draw(_class_stream(CI_SEED, MEMBER_DOMAIN), 64)
    non_member = _draw(_class_stream(CI_SEED, NON_MEMBER_DOMAIN), 64)
    assert not np.array_equal(member, non_member)
    assert np.array_equal(member, _draw(_class_stream(CI_SEED, MEMBER_DOMAIN), 64))
    assert CI_STREAM_SCHEME


def test_a_different_ci_seed_moves_both_streams() -> None:
    for domain in (MEMBER_DOMAIN, NON_MEMBER_DOMAIN):
        assert not np.array_equal(
            _draw(_class_stream(CI_SEED, domain), 64),
            _draw(_class_stream(CI_SEED + 1, domain), 64),
        )


def test_the_index_draw_matches_the_block_b_convention() -> None:
    """The formula is Block B's `resampling._indices`, replicated rather than imported."""
    from src.analysis.resampling import _indices

    for size in (7, 64, 501):
        mine = _draw(np.random.Generator(np.random.PCG64(3)), size)
        theirs = _indices(np.random.Generator(np.random.PCG64(3)), size, size)
        assert np.array_equal(mine, theirs)


def test_each_class_keeps_its_own_sample_size() -> None:
    members, non_members = population(n_member=90, n_non_member=140)
    low, high = auroc_confidence_interval(
        members, non_members, n_replicates=300, seed=CI_SEED, alpha=0.05
    )
    assert 0.0 <= low <= high <= 1.0
    assert (low, high) == auroc_confidence_interval(
        list(reversed(members)),
        list(reversed(non_members)),
        n_replicates=300,
        seed=CI_SEED,
        alpha=0.05,
    )


# ------------------------------------------------------------------ regression 3/4 unchanged
def test_the_interval_is_deterministic() -> None:
    members, non_members = population()
    assert interval(members, non_members) == interval(members, non_members)


def test_the_interval_brackets_a_separable_and_a_null_case() -> None:
    separable_low, separable_high = auroc_confidence_interval(
        [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        n_replicates=400,
        seed=CI_SEED,
        alpha=0.05,
    )
    assert separable_low > 0.5

    identical = [1.0] * 40
    null_low, null_high = auroc_confidence_interval(
        identical, identical, n_replicates=400, seed=CI_SEED, alpha=0.05
    )
    assert null_low <= 0.5 <= null_high


def test_a_degenerate_replicate_count_or_alpha_is_refused() -> None:
    members, non_members = population(20, 20)
    with pytest.raises(BlindControlError):
        auroc_confidence_interval(members, non_members, n_replicates=0, seed=1, alpha=0.05)
    with pytest.raises(BlindControlError):
        auroc_confidence_interval(members, non_members, n_replicates=10, seed=1, alpha=1.0)
