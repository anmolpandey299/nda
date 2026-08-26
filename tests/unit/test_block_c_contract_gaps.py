"""The explicitly enumerated Block C test-contract items not covered elsewhere.

These are the cases the S05/S06 requirement text names by hand: transitive duplicate
components, a canary id that must not encode its own membership, ~50% inclusion at scale,
per-family seed mutation, natural-evaluation identity reuse within a seed, the 01 §8G
model provenance fields, and the Block B precision / max-sequence handoff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.analysis.settings import scoring_settings
from src.data.canaries import generate_canary_pool, inclusion_map
from src.data.dedup import deduplicate, jaccard, shingles
from src.data.membership import natural_member_subsets
from src.data.normalise import normalise_record
from src.materials import material
from src.models.fixtures import build_base_model, load_fixture_spec, target_parameter_names
from src.provenance.config import resolve_config
from src.training.lora import load_target_mapping
from src.training.model_contract import (
    RESEARCH_PROVENANCE_FIELDS,
    UNRESOLVED,
    BackendRuntime,
    ModelContractError,
    load_panel,
    require_scoring_agreement,
)
from src.training.seeds import DERIVED_FAMILIES, seed_families
from src.training.settings import lora_settings, panel_settings
from src.training.trainer import data_order, initialise_adapter, train_reference_lora

REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_FOR_SUBSETS = generate_canary_pool(generator_seed=20260820, pool_size=32, secret_length=12)
CONFIGS = REPO_ROOT / "configs"
SPEC = load_fixture_spec(resolve_config(CONFIGS / "models/tiny_fixture.json", config_root=CONFIGS))
PANEL = panel_settings(REPO_ROOT)
MAPPING = load_target_mapping("tiny_fixture", PANEL.document)
TARGETS = target_parameter_names(SPEC)


# ------------------------------------------------------------------ S05.2 transitive
def test_a_fully_connected_duplicate_component_collapses_to_one_record() -> None:
    """A chain whose every pair exceeds the threshold leaves exactly one survivor."""
    stem = "The hepatic microvascular perfusion index was measured in a simulated cohort"
    chain = [
        normalise_record("C1", {"title": stem, "abstract": "alpha beta gamma delta"}),
        normalise_record("C2", {"title": stem, "abstract": "alpha beta gamma epsilon"}),
        normalise_record("C3", {"title": stem, "abstract": "alpha beta zeta epsilon"}),
    ]
    result = deduplicate(chain, shingle_size=5, threshold=0.7)
    assert len(result.retained) == 1
    survivor = result.retained[0].record_id
    assert {d.dropped_id for d in result.decisions} == {"C1", "C2", "C3"} - {survivor}
    assert {d.retained_id for d in result.decisions} == {survivor}


def test_no_two_retained_records_are_at_or_above_the_threshold() -> None:
    """The guaranteed post-condition, on a chain whose ends fall below the threshold."""
    stem = "The hepatic microvascular perfusion index was measured in a simulated cohort"
    chain = [
        normalise_record("C1", {"title": stem, "abstract": "alpha beta gamma delta"}),
        normalise_record("C2", {"title": stem, "abstract": "alpha beta gamma epsilon"}),
        normalise_record("C3", {"title": stem, "abstract": "alpha beta zeta epsilon"}),
    ]
    result = deduplicate(chain, shingle_size=5, threshold=0.8)
    retained = list(result.retained)
    assert len(retained) == 2, "the middle record of the chain should have been dropped"
    for left in range(len(retained)):
        for right in range(left + 1, len(retained)):
            similarity = jaccard(
                shingles(retained[left].text, 5), shingles(retained[right].text, 5)
            )
            assert similarity < 0.8, (retained[left].record_id, retained[right].record_id)


def test_a_clearly_distinct_pair_is_left_alone() -> None:
    left = normalise_record("D1", {"title": "Renal tubular acidosis", "abstract": "alpha"})
    right = normalise_record("D2", {"title": "Atrial conduction", "abstract": "omega"})
    result = deduplicate([left, right], shingle_size=5, threshold=0.8)
    assert len(result.retained) == 2 and result.decisions == ()


# ------------------------------------------------------------------ S05.5 canary identity
def test_a_canary_id_does_not_encode_its_membership_label() -> None:
    """The mutation this kills: a canary whose id reveals whether it was trained on."""
    pool = generate_canary_pool(generator_seed=20260820, pool_size=256, secret_length=12)
    ids_by_seed = {}
    for seed in (101, 202, 303):
        decisions = inclusion_map(pool, inclusion_seed=seed, probability=0.5)
        ids_by_seed[seed] = {c.canary_id: decisions[c.canary_id] for c in pool}
    # the same id carries a different label under a different seed, so the id cannot encode it
    flipped = [
        canary_id
        for canary_id in ids_by_seed[101]
        if ids_by_seed[101][canary_id] != ids_by_seed[202][canary_id]
    ]
    assert flipped, "no canary changed membership between seeds"
    # and the ids themselves are label-free: a positional index only
    assert all(canary.canary_id.startswith("canary-") for canary in pool)
    assert all(canary.canary_id.removeprefix("canary-").isdigit() for canary in pool)
    assert not any(
        token in canary.canary_id.upper()
        for canary in pool
        for token in ("MEMBER", "TRAIN", "TRUE", "FALSE")
    )


def test_inclusion_is_about_one_half_over_a_large_pool() -> None:
    pool = generate_canary_pool(generator_seed=20260820, pool_size=1024, secret_length=12)
    for seed in (101, 202, 303):
        rate = sum(inclusion_map(pool, inclusion_seed=seed, probability=0.5).values()) / len(pool)
        assert 0.45 <= rate <= 0.55, (seed, rate)


def test_the_declared_pool_size_is_the_00_7_1_value() -> None:
    from src.data.settings import canary_settings

    assert material(canary_settings(REPO_ROOT), "pool_size") == 4096


# ------------------------------------------------------------------ S05.4 identity reuse
TRAINED = [f"T{index:05d}" for index in range(400)]


def _subsets(training_seed: int) -> Any:
    from blockc_fixtures import candidate_pool

    return natural_member_subsets(
        candidate_pool(TRAINED, POOL_FOR_SUBSETS),
        calibration_size=30,
        evaluation_size=60,
        training_seed=training_seed,
    )


def test_natural_subset_identities_are_reused_across_views_within_a_seed() -> None:
    """00 §6.3: the same candidate identities are reused across every audit view in a seed."""
    views = [_subsets(101) for _ in range(4)]
    assert len({view.evaluation for view in views}) == 1
    assert len({view.calibration for view in views}) == 1
    assert len({view.evaluation_identity() for view in views}) == 1
    assert len({view.calibration_identity() for view in views}) == 1


def test_natural_member_eval_is_one_fixed_set_across_101_202_303() -> None:
    """Architect adjudication: ONE evaluation set, reused across the training seeds."""
    per_seed = {seed: _subsets(seed) for seed in (101, 202, 303)}
    assert len({view.evaluation for view in per_seed.values()}) == 1
    assert len({view.evaluation_identity() for view in per_seed.values()}) == 1
    assert len({view.evaluation_seed for view in per_seed.values()}) == 1


def test_natural_member_calibration_may_vary_per_training_seed() -> None:
    per_seed = {seed: _subsets(seed) for seed in (101, 202, 303)}
    assert len({view.calibration for view in per_seed.values()}) == 3
    assert len({view.calibration_seed for view in per_seed.values()}) == 3
    for view in per_seed.values():
        assert not set(view.calibration) & set(view.evaluation)


def test_the_fixed_evaluation_identity_carries_no_training_seed() -> None:
    """Persisted and hashed, so the reused set can be checked later [adjudication]."""
    identity = _subsets(101).evaluation_identity()
    assert len(identity) == 64
    assert identity == _subsets(303).evaluation_identity()


def test_canary_inclusion_masks_stay_independently_redrawn() -> None:
    from blockc_fixtures import inclusion_for

    masks = {seed: inclusion_for(POOL_FOR_SUBSETS, seed).mask_sha256() for seed in (101, 202, 303)}
    assert len(set(masks.values())) == 3


# ------------------------------------------------------------------ S06.5 per-family mutation
def _adapter_identity(init_seed: int) -> str:
    return initialise_adapter(
        {name: (4, 6) for name in ("a", "b")}, rank=2, scaling=2.0, init_seed=init_seed
    ).identity()


@pytest.mark.parametrize("family", DERIVED_FAMILIES)
def test_each_family_moves_only_when_its_own_value_moves(family: str) -> None:
    """The mutation this kills: a family that is recorded but drives nothing."""
    first = seed_families(101, differentially_private=True)
    changed = dict(first.values)
    changed[family] = changed[family] + 1
    ids = [f"T{i}" for i in range(24)]

    order_before = data_order(ids, order_seed=first["data_order"])
    order_after = data_order(ids, order_seed=changed["data_order"])
    init_before, init_after = (
        _adapter_identity(first["lora_init"]),
        _adapter_identity(changed["lora_init"]),
    )

    if family == "data_order":
        assert order_before != order_after
        assert init_before == init_after
    elif family == "lora_init":
        assert init_before != init_after
        assert order_before == order_after
    else:
        assert order_before == order_after
        assert init_before == init_after


def test_the_canary_inclusion_family_drives_inclusion_alone() -> None:
    pool = generate_canary_pool(generator_seed=20260820, pool_size=64, secret_length=12)
    first = seed_families(101, differentially_private=True)
    second = seed_families(202, differentially_private=True)
    assert inclusion_map(pool, inclusion_seed=first["canary_inclusion"], probability=0.5) != (
        inclusion_map(pool, inclusion_seed=second["canary_inclusion"], probability=0.5)
    )
    assert inclusion_map(pool, inclusion_seed=first["canary_inclusion"], probability=0.5) == (
        inclusion_map(pool, inclusion_seed=first["canary_inclusion"], probability=0.5)
    )


def test_the_dp_noise_family_drives_noise_alone() -> None:
    import numpy as np

    from src.dp.mechanism import gaussian_noise

    first = seed_families(101, differentially_private=True)
    second = seed_families(202, differentially_private=True)
    assert not np.array_equal(
        gaussian_noise((8,), scale=1.0, dp_seed=first["dp_noise"], step=0),
        gaussian_noise((8,), scale=1.0, dp_seed=second["dp_noise"], step=0),
    )


def test_a_run_reproduces_bitwise_from_the_complete_seed_family_alone() -> None:
    from blockc_fixtures import candidate_pool, fixture_contract

    from src.data.membership import no_canary_plan

    ids = [f"T{index}" for index in range(10)]
    plan = no_canary_plan(
        label="NO_CANARY",
        training_seed=101,
        candidates=candidate_pool(ids, POOL_FOR_SUBSETS),
        natural_ids=ids,
    )
    contract = fixture_contract(
        plan=plan,
        base_parameter_names=list(build_base_model(SPEC)),
        mapping=MAPPING,
        training_seed=101,
        optimizer_step_budget=8,
    )

    def once() -> Any:
        return train_reference_lora(contract, base=build_base_model(SPEC))

    assert once().adapter.identity() == once().adapter.identity()


# ------------------------------------------------------------------ S06.1 provenance fields
def test_every_01_8g_provenance_field_is_declared_for_every_panel_member() -> None:
    for member in load_panel(PANEL.document):
        document = member.as_dict()
        for field in RESEARCH_PROVENANCE_FIELDS:
            assert field in document, f"{member.alias}:{field}"


def test_the_unacquired_fields_are_explicit_sentinels_not_guesses() -> None:
    for member in load_panel(PANEL.document):
        unresolved = set(member.unresolved_fields())
        assert {
            "revision",
            "parameter_count",
            "dtype_on_disk",
            "config_sha256",
            "tokenizer_sha256",
            "weight_file_sha256",
        } <= unresolved, member.alias
        # published model-card facts are recorded; artifact-derived identities are not
        assert member.license != UNRESOLVED
        assert member.frozen_modules


def test_an_evidentiary_run_refuses_an_unacquired_model() -> None:
    from src.training.model_contract import validate_panel_member

    for member in load_panel(PANEL.document):
        problems = validate_panel_member(member, evidentiary=True)
        assert any("01 §8G field" in p for p in problems), member.alias


# ------------------------------------------------------------------ Block B handoff
def test_the_backend_runtime_agrees_with_the_resolved_scoring_config() -> None:
    scoring = scoring_settings(REPO_ROOT).document
    runtime = BackendRuntime(
        precision=str(scoring["precision"]),
        max_sequence_length=int(str(scoring["max_sequence_length"])),
        backend_identity="fixture-numpy",
        supports_gradients=True,
    )
    require_scoring_agreement(runtime, scoring)


@pytest.mark.parametrize("field", ["precision", "max_sequence_length"])
def test_a_silent_runtime_mismatch_is_refused(field: str) -> None:
    """The mutation this kills: a backend quietly running at a different precision/length."""
    scoring = dict(scoring_settings(REPO_ROOT).document)
    runtime = BackendRuntime(
        precision="bfloat16" if field == "precision" else str(scoring["precision"]),
        max_sequence_length=(
            999 if field == "max_sequence_length" else int(str(scoring["max_sequence_length"]))
        ),
        backend_identity="fixture-numpy",
        supports_gradients=True,
    )
    with pytest.raises(ModelContractError, match=field):
        require_scoring_agreement(runtime, scoring)


def test_the_training_sequence_length_is_a_required_pre_run_field() -> None:
    """It must agree with the scorer's max_sequence_length, so it cannot default silently."""
    from src.materials import UncalibratedConstantError

    with pytest.raises(UncalibratedConstantError):
        material(lora_settings(REPO_ROOT), "sequence_length")
