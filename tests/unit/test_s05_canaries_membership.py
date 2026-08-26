"""S05.4-S05.6 — canary pool, derived inclusion, structural member subsets, matched controls.

B-C2: membership state is structural. A canary cannot be presented as a natural candidate,
and a training seed, an inclusion seed and a mask cannot be asserted independently.
"""

from __future__ import annotations

from typing import Any

import pytest
from blockc_fixtures import tiny_corpus

from src.data.canaries import (
    CANARY_GENERATOR_VERSION,
    CanaryError,
    assert_disjoint_from_natural,
    excluded_canaries,
    generate_canary_pool,
    included_canaries,
    inclusion_map,
)
from src.data.membership import (
    CALIBRATION_DOMAIN,
    EVAL_DOMAIN,
    CanaryInclusion,
    MembershipError,
    NaturalMemberSubsets,
    TrainingPlan,
    canary_plan,
    canary_pool_identity,
    derive_canary_inclusion,
    matched_control_problems,
    natural_candidate_pool,
    natural_member_subsets,
    no_canary_plan,
    replay_canary_inclusion,
    sample_without_replacement,
    selection_seed,
)
from src.training.seeds import seed_families

POOL = generate_canary_pool(generator_seed=20260820, pool_size=64, secret_length=12)
STUDY_SEED = 20260820


def pool_of(ids: list[str]) -> Any:
    return natural_candidate_pool(ids, POOL, study_seed=STUDY_SEED)


def inclusion(seed: int, *, probability: float = 0.5) -> Any:
    return derive_canary_inclusion(
        POOL, seeds=seed_families(seed, differentially_private=False), probability=probability
    )


TRAINED = [f"T{index:04d}" for index in range(200)]


# ------------------------------------------------------------------ 00 §7.1 pool
def test_the_pool_is_deterministic_and_unique() -> None:
    again = generate_canary_pool(generator_seed=20260820, pool_size=64, secret_length=12)
    assert [c.canary_id for c in again] == [c.canary_id for c in POOL]
    assert [c.secret for c in again] == [c.secret for c in POOL]
    assert len({c.record.text_sha256 for c in POOL}) == len(POOL)
    assert CANARY_GENERATOR_VERSION


def test_a_different_generator_seed_gives_a_different_pool() -> None:
    other = generate_canary_pool(generator_seed=20260821, pool_size=64, secret_length=12)
    assert {c.secret for c in other} != {c.secret for c in POOL}
    assert canary_pool_identity(other) != canary_pool_identity(POOL)


def test_the_secret_alphabet_excludes_confusable_characters() -> None:
    assert not set("".join(c.secret for c in POOL)) & set("0O1Il")
    assert all(len(c.secret) == 12 for c in POOL)


def test_canaries_are_disjoint_from_the_natural_corpus() -> None:
    assert_disjoint_from_natural(POOL, tiny_corpus(60))


def test_a_canary_inside_the_natural_member_set_is_refused() -> None:
    natural = [*tiny_corpus(20), POOL[3].record]
    with pytest.raises(CanaryError):
        assert_disjoint_from_natural(POOL, natural)


# ------------------------------------------------------------------ B-C2 structural pool
def test_a_canary_presented_as_a_natural_candidate_is_rejected() -> None:
    """Codex counterexample: ["N1", "canary-00000", "N2"] reaching NATURAL_MEMBER_EVAL."""
    with pytest.raises(MembershipError, match="presented as natural"):
        natural_candidate_pool(["N1", POOL[0].canary_id, "N2"], POOL, study_seed=STUDY_SEED)


def test_the_subset_draw_requires_a_validated_pool_not_raw_strings() -> None:
    with pytest.raises((MembershipError, AttributeError, TypeError)):
        natural_member_subsets(
            ["N1", "N2", "N3"],  # type: ignore[arg-type]
            calibration_size=1,
            evaluation_size=1,
            training_seed=101,
        )


def test_a_repeated_natural_candidate_is_refused() -> None:
    with pytest.raises(MembershipError, match="repeats a record id"):
        natural_candidate_pool(["N1", "N1"], POOL, study_seed=STUDY_SEED)


def test_the_pool_identity_binds_the_canary_pool() -> None:
    other = generate_canary_pool(generator_seed=1, pool_size=8, secret_length=12)
    first = natural_candidate_pool(TRAINED, POOL, study_seed=STUDY_SEED)
    second = natural_candidate_pool(TRAINED, other, study_seed=STUDY_SEED)
    assert first.identity() != second.identity()


# ------------------------------------------------------------------ 00 §7.2 inclusion
def test_inclusion_is_derived_from_the_recorded_seed_family() -> None:
    families = seed_families(101, differentially_private=False)
    derived = inclusion(101)
    assert derived.training_seed == 101
    assert derived.inclusion_seed == families["canary_inclusion"]
    assert derived.mask == inclusion_map(
        POOL, inclusion_seed=families["canary_inclusion"], probability=0.5
    )


def test_inclusion_is_redrawn_independently_per_seed() -> None:
    masks = [inclusion(seed).mask_sha256() for seed in (101, 202, 303)]
    assert len(set(masks)) == 3


def test_inclusion_is_reproducible_within_a_seed() -> None:
    assert inclusion(101).mask_sha256() == inclusion(101).mask_sha256()


def test_probability_one_and_zero_are_honoured() -> None:
    assert all(inclusion(101, probability=1.0).mask.values())
    assert not any(inclusion(101, probability=0.0).mask.values())


def test_included_and_excluded_partition_the_pool() -> None:
    decisions = inclusion(202).mask
    inside = included_canaries(POOL, decisions)
    outside = excluded_canaries(POOL, decisions)
    assert len(inside) + len(outside) == len(POOL)
    assert not {c.canary_id for c in inside} & {c.canary_id for c in outside}


def test_a_missing_decision_fails_closed() -> None:
    decisions = dict(inclusion(202).mask)
    del decisions[POOL[0].canary_id]
    with pytest.raises(CanaryError):
        included_canaries(POOL, decisions)


# ------------------------------------------------------------------ B-C2 replay guards
def test_a_mask_from_another_seed_cannot_be_replayed_as_this_one() -> None:
    """Codex counterexample: seed 111's mask reused for seed 202 with a different recorded seed."""
    foreign = inclusion(111)
    with pytest.raises(MembershipError, match="seed family value|training seed"):
        replay_canary_inclusion(
            POOL,
            seeds=seed_families(202, differentially_private=False),
            probability=0.5,
            stored=foreign,
        )


def test_an_altered_stored_mask_is_rejected() -> None:
    stored = inclusion(101)
    tampered = CanaryInclusion(
        canary_pool_sha256=stored.canary_pool_sha256,
        training_seed=stored.training_seed,
        inclusion_seed=stored.inclusion_seed,
        probability=stored.probability,
        mask={key: not value for key, value in stored.mask.items()},
    )
    with pytest.raises(MembershipError, match="does not match the mask"):
        replay_canary_inclusion(
            POOL,
            seeds=seed_families(101, differentially_private=False),
            probability=0.5,
            stored=tampered,
        )


def test_one_mask_cannot_be_substituted_across_101_202_303() -> None:
    """The same mask claimed for every seed is refused at replay for two of the three."""
    shared = inclusion(101)
    accepted = []
    for seed in (101, 202, 303):
        try:
            replay_canary_inclusion(
                POOL,
                seeds=seed_families(seed, differentially_private=False),
                probability=0.5,
                stored=shared,
            )
            accepted.append(seed)
        except MembershipError:
            pass
    assert accepted == [101]


def test_a_replayed_mask_under_a_different_probability_is_rejected() -> None:
    stored = inclusion(101, probability=0.5)
    with pytest.raises(MembershipError, match="probability"):
        replay_canary_inclusion(
            POOL,
            seeds=seed_families(101, differentially_private=False),
            probability=0.25,
            stored=stored,
        )


def test_a_mask_drawn_over_another_pool_is_rejected() -> None:
    other = generate_canary_pool(generator_seed=7, pool_size=64, secret_length=12)
    stored = derive_canary_inclusion(
        other, seeds=seed_families(101, differentially_private=False), probability=0.5
    )
    with pytest.raises(MembershipError, match="different canary pool"):
        replay_canary_inclusion(
            POOL,
            seeds=seed_families(101, differentially_private=False),
            probability=0.5,
            stored=stored,
        )


# ------------------------------------------------------------------ 00 §6.3 subsets
def test_member_subsets_are_disjoint_and_correctly_sized() -> None:
    subsets = natural_member_subsets(
        pool_of(TRAINED), calibration_size=20, evaluation_size=50, training_seed=101
    )
    assert len(subsets.calibration) == 20 and len(subsets.evaluation) == 50
    assert not set(subsets.calibration) & set(subsets.evaluation)


def test_member_subsets_are_order_independent() -> None:
    first = natural_member_subsets(
        pool_of(TRAINED), calibration_size=20, evaluation_size=50, training_seed=101
    )
    second = natural_member_subsets(
        pool_of(list(reversed(TRAINED))),
        calibration_size=20,
        evaluation_size=50,
        training_seed=101,
    )
    assert first.calibration == second.calibration
    assert first.evaluation == second.evaluation


def test_the_selection_streams_are_domain_separated() -> None:
    assert selection_seed(101, EVAL_DOMAIN) != selection_seed(101, CALIBRATION_DOMAIN)
    assert selection_seed(STUDY_SEED, EVAL_DOMAIN) == selection_seed(STUDY_SEED, EVAL_DOMAIN)


def test_overlapping_calibration_and_evaluation_is_refused_at_construction() -> None:
    with pytest.raises(MembershipError):
        NaturalMemberSubsets(
            calibration=("A", "B"),
            evaluation=("B", "C"),
            seed=1,
            evaluation_seed=1,
            calibration_seed=2,
            candidate_pool_identity="x" * 64,
        )


def test_asking_for_more_records_than_exist_fails_closed() -> None:
    with pytest.raises(MembershipError):
        natural_member_subsets(
            pool_of([f"T{i}" for i in range(5)]),
            calibration_size=3,
            evaluation_size=3,
            training_seed=101,
        )
    with pytest.raises(MembershipError):
        sample_without_replacement([f"T{i}" for i in range(5)], count=6, seed=1)


def test_sampling_is_without_replacement() -> None:
    drawn = sample_without_replacement([f"T{i}" for i in range(50)], count=20, seed=7)
    assert len(set(drawn)) == 20


# ------------------------------------------------------------------ 00 §7.5 controls
def test_a_canary_plan_records_its_derived_inclusion() -> None:
    derived = inclusion(101)
    plan = canary_plan(
        label="CANARY",
        candidates=pool_of(TRAINED),
        natural_ids=TRAINED[:10],
        pool=POOL,
        inclusion=derived,
    )
    assert plan.inclusion_seed == derived.inclusion_seed
    assert plan.training_seed == 101
    assert {c.canary_id for c in plan.canaries} == set(derived.included_ids())
    assert plan.as_dict()["inclusion_mask_sha256"] == derived.mask_sha256()


def test_a_plan_cannot_carry_canaries_the_mask_did_not_select() -> None:
    derived = inclusion(101)
    excluded = [c for c in POOL if c.canary_id not in set(derived.included_ids())]
    with pytest.raises(MembershipError, match="not the ones the recorded"):
        TrainingPlan(
            label="CANARY",
            training_seed=101,
            natural_ids=("T0000",),
            canaries=tuple(excluded[:2]),
            canaries_permitted=True,
            inclusion=derived,
        )


def test_a_plan_whose_inclusion_is_for_another_seed_is_refused() -> None:
    derived = inclusion(202)
    with pytest.raises(MembershipError, match="inclusion draw is for training seed"):
        TrainingPlan(
            label="CANARY",
            training_seed=101,
            natural_ids=("T0000",),
            canaries=tuple(c for c in POOL if c.canary_id in set(derived.included_ids())),
            canaries_permitted=True,
            inclusion=derived,
        )


def test_the_matched_control_contains_no_canary() -> None:
    control = no_canary_plan(
        label="NO_CANARY", training_seed=101, candidates=pool_of(TRAINED), natural_ids=TRAINED[:10]
    )
    assert control.canaries == () and control.inclusion is None
    assert not control.canaries_permitted


def test_a_no_canary_plan_that_carries_a_canary_is_refused() -> None:
    with pytest.raises(MembershipError):
        TrainingPlan(
            label="NO_CANARY",
            training_seed=101,
            natural_ids=("T0000",),
            canaries=(POOL[0],),
            canaries_permitted=False,
            inclusion=None,
        )


def test_a_training_id_outside_the_validated_pool_is_refused() -> None:
    with pytest.raises(MembershipError, match="not in the validated natural candidate pool"):
        no_canary_plan(
            label="NO_CANARY",
            training_seed=101,
            candidates=pool_of(TRAINED),
            natural_ids=["NOT_A_CANDIDATE"],
        )


def test_the_matched_control_must_share_records_seed_and_candidate_pool() -> None:
    candidates = pool_of(TRAINED)
    treated = canary_plan(
        label="CANARY",
        candidates=candidates,
        natural_ids=TRAINED[:10],
        pool=POOL,
        inclusion=inclusion(101),
    )
    good = no_canary_plan(
        label="NO_CANARY", training_seed=101, candidates=candidates, natural_ids=TRAINED[:10]
    )
    assert matched_control_problems(treated, good) == []

    wrong_records = no_canary_plan(
        label="NO_CANARY", training_seed=101, candidates=candidates, natural_ids=TRAINED[:9]
    )
    assert matched_control_problems(treated, wrong_records)

    wrong_seed = no_canary_plan(
        label="NO_CANARY", training_seed=202, candidates=candidates, natural_ids=TRAINED[:10]
    )
    assert matched_control_problems(treated, wrong_seed)


def test_a_plan_that_repeats_a_canary_is_refused() -> None:
    derived = inclusion(101)
    first = next(c for c in POOL if c.canary_id in set(derived.included_ids()))
    with pytest.raises(MembershipError):
        TrainingPlan(
            label="CANARY",
            training_seed=101,
            natural_ids=("T0000",),
            canaries=(first, first),
            canaries_permitted=True,
            inclusion=derived,
        )
