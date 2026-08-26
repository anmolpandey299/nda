"""Block C end to end on tiny fixtures: corpus -> canaries -> plans -> execution -> manifests.

Nothing here fetches PubMed, downloads a model or touches a GPU. Every corpus is generated,
every manifest is a fixture-reference manifest, and the evidentiary paths are exercised only
by proving that they refuse [AUTH: 01 §20].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from blockc_fixtures import candidate_pool, fixture_contract, inclusion_for, source_records

from src.data.canaries import assert_disjoint_from_natural, generate_canary_pool
from src.data.corpus import AcquisitionSpec, build_master_corpus, filter_eligible
from src.data.manifest import (
    FIXTURE_CORPUS_ROLE,
    CorpusManifestError,
    corpus_role,
    fixture_corpus_manifest,
    is_fixture_corpus,
    require_research_corpus,
)
from src.data.membership import (
    canary_plan,
    matched_control_problems,
    natural_member_subsets,
    no_canary_plan,
)
from src.data.normalise import normalise_record
from src.dp.mechanism import (
    SAMPLE_LEVEL_ADJACENCY,
    calibrate_noise_multiplier,
    dp_smoke_run,
    poisson_sample_rate,
    steps_for,
)
from src.models.fixtures import build_base_model, load_fixture_spec
from src.provenance.config import resolve_config
from src.provenance.data_manifest import data_manifest_sha256, require_valid_data_manifest
from src.training.execution import matched_execution_problems, matched_update_budget
from src.training.lora import load_target_mapping
from src.training.manifests import (
    EVIDENTIARY_ROLE,
    FIXTURE_REFERENCE_ROLE,
    build_adapter_manifest,
    validate_adapter_manifest,
)
from src.training.seeds import seed_families
from src.training.settings import panel_settings
from src.training.trainer import adapter_file_hash, save_adapter, train_reference_lora

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
SPEC = load_fixture_spec(resolve_config(CONFIGS / "models/tiny_fixture.json", config_root=CONFIGS))
MAPPING = load_target_mapping("tiny_fixture", panel_settings(REPO_ROOT).document)
SIZES = {
    "TRAIN_CANDIDATES": 60,
    "CALIBRATION_NONMEMBERS": 20,
    "EVAL_NONMEMBERS": 20,
    "RESERVE": 20,
}
FLOOR = "2026-04-01"
STAMP = "2026-08-25T00:00:00Z"
BASE_NAMES = list(build_base_model(SPEC))


@pytest.fixture(scope="module")
def corpus() -> Any:
    records = [
        *source_records(150, publication_date="2026-05-14"),
        *source_records(10, start=900, publication_date="2026-02-01"),
    ]
    eligible, refused = filter_eligible(records, floor=FLOOR)
    assert len(refused) == 10
    normalised = [normalise_record(r.record_id, r.fields) for r in eligible]
    return build_master_corpus(
        normalised, sizes=SIZES, split_seed=20260820, shingle_size=5, near_duplicate_threshold=0.8
    )


@pytest.fixture(scope="module")
def pool() -> Any:
    return generate_canary_pool(generator_seed=20260820, pool_size=32, secret_length=12)


# ------------------------------------------------------------------ S05 assembly
def test_the_pipeline_produces_disjoint_partitions_and_canaries(corpus: Any, pool: Any) -> None:
    assert_disjoint_from_natural(pool, corpus.retained)
    every = [rid for ids in corpus.partitions.values() for rid in ids]
    assert len(set(every)) == len(every)
    assert not set(every) & {c.canary_id for c in pool}


def test_member_subsets_never_reach_the_evaluation_side(corpus: Any, pool: Any) -> None:
    trained = corpus.partitions["TRAIN_CANDIDATES"]
    subsets = natural_member_subsets(
        candidate_pool(trained, pool), calibration_size=15, evaluation_size=30, training_seed=101
    )
    assert not set(subsets.calibration) & set(subsets.evaluation)
    assert set(subsets.calibration) <= set(trained)
    assert not set(subsets.evaluation) & set(corpus.partitions["EVAL_NONMEMBERS"])


def test_a_canary_never_enters_the_natural_member_subsets(corpus: Any, pool: Any) -> None:
    trained = corpus.partitions["TRAIN_CANDIDATES"]
    subsets = natural_member_subsets(
        candidate_pool(trained, pool), calibration_size=15, evaluation_size=30, training_seed=101
    )
    ids = {c.canary_id for c in pool}
    assert not ids & set(subsets.calibration)
    assert not ids & set(subsets.evaluation)


def test_canary_inclusion_is_redrawn_per_training_seed(pool: Any) -> None:
    masks = {seed: inclusion_for(pool, seed).mask_sha256() for seed in (101, 202, 303)}
    assert len(set(masks.values())) == 3


# ------------------------------------------------------------------ S05.8 data manifest
def fixture_manifest(corpus: Any) -> dict[str, Any]:
    spec = AcquisitionSpec(query="SYNTHETIC_FIXTURE_NO_QUERY", publication_date_floor=FLOOR)
    return fixture_corpus_manifest(
        corpus,
        spec=spec,
        raw_data_sha256="a" * 64,
        preprocessing_code_commit="0" * 40,
        preprocessing_config_sha256="b" * 64,
        download_timestamp_utc=STAMP,
        dataset_alias="fixture_blockc_corpus",
    )


def test_the_fixture_corpus_manifest_is_valid_and_declares_its_role(corpus: Any) -> None:
    document = fixture_manifest(corpus)
    require_valid_data_manifest(document)
    assert corpus_role(document) == FIXTURE_CORPUS_ROLE
    assert is_fixture_corpus(document)
    assert document["source_query"] == "SYNTHETIC_FIXTURE_NO_QUERY"
    assert document["split_rng_seed"] == 20260820


def test_a_fixture_manifest_may_not_attribute_a_scientific_result(corpus: Any) -> None:
    with pytest.raises(CorpusManifestError, match="may not attribute"):
        require_research_corpus(fixture_manifest(corpus))


def test_the_manifest_identity_moves_when_the_split_moves(corpus: Any) -> None:
    other = build_master_corpus(
        corpus.retained,
        sizes=SIZES,
        split_seed=20260821,
        shingle_size=5,
        near_duplicate_threshold=0.8,
    )
    assert fixture_manifest(corpus)["manifest_sha256"] != fixture_manifest(other)["manifest_sha256"]


def test_a_manifest_a_result_was_attributed_to_is_immutable(corpus: Any) -> None:
    from src.data.manifest import assert_manifest_unchanged
    from src.provenance.data_manifest import stamp_manifest_sha256

    document = fixture_manifest(corpus)
    identity = data_manifest_sha256(document)
    assert_manifest_unchanged(document, identity)

    edited = dict(document)
    edited["record_count"] = 1
    with pytest.raises(CorpusManifestError, match="immutable"):
        assert_manifest_unchanged(edited, identity)
    with pytest.raises(CorpusManifestError, match="immutable"):
        assert_manifest_unchanged(stamp_manifest_sha256(edited), identity)


# ------------------------------------------------------------------ S06 execution
def arms(corpus: Any, pool: Any, seed: int, *, epochs: int = 2, batch_size: int = 4) -> Any:
    """One treated/control pair sharing a single matched optimiser-update budget."""
    trained = list(corpus.partitions["TRAIN_CANDIDATES"])[:20]
    candidates = candidate_pool(trained, pool)
    treated_plan = canary_plan(
        label="CANARY",
        candidates=candidates,
        natural_ids=trained,
        pool=pool,
        inclusion=inclusion_for(pool, seed),
    )
    control_plan = no_canary_plan(
        label="NO_CANARY", training_seed=seed, candidates=candidates, natural_ids=trained
    )
    budget = matched_update_budget(treated=treated_plan, epochs=epochs, batch_size=batch_size)

    def build(plan: Any) -> Any:
        return fixture_contract(
            plan=plan,
            base_parameter_names=BASE_NAMES,
            mapping=MAPPING,
            training_seed=seed,
            optimizer_step_budget=budget,
            batch_size=batch_size,
        )

    return build(treated_plan), build(control_plan)


def test_the_three_calibration_seeds_produce_three_distinct_adapters(
    corpus: Any, pool: Any
) -> None:
    identities = set()
    for seed in (101, 202, 303):
        treated, _ = arms(corpus, pool, seed)
        identities.add(
            train_reference_lora(treated, base=build_base_model(SPEC)).adapter.identity()
        )
    assert len(identities) == 3


def test_the_matched_control_executes_the_same_update_budget(corpus: Any, pool: Any) -> None:
    """B-C4: 20 natural + k canaries against 20 natural must not be 10 updates against 8."""
    treated, control = arms(corpus, pool, 101)
    assert matched_control_problems(treated.plan, control.plan) == []
    assert matched_execution_problems(treated, control) == []
    treated_run = train_reference_lora(treated, base=build_base_model(SPEC))
    control_run = train_reference_lora(control, base=build_base_model(SPEC))
    assert treated_run.optimizer_steps == control_run.optimizer_steps
    assert treated.plan.n_training_records != control.plan.n_training_records


def test_the_base_model_survives_the_whole_pipeline_bitwise(corpus: Any, pool: Any) -> None:
    treated, _ = arms(corpus, pool, 101)
    base = build_base_model(SPEC)
    train_reference_lora(treated, base=base)
    for name, array in build_base_model(SPEC).items():
        assert np.array_equal(base[name], array), name


# ------------------------------------------------------------------ S06.11 adapter manifest
def adapter_manifest(corpus: Any, pool: Any, seed: int, *, dp_run: Any = None) -> dict[str, Any]:
    treated, _ = arms(corpus, pool, seed)
    outcome = train_reference_lora(treated, base=build_base_model(SPEC))
    data_document = fixture_manifest(corpus)
    return build_adapter_manifest(
        manifest_role=FIXTURE_REFERENCE_ROLE,
        execution_contract_sha256=treated.identity(),
        backend_status=treated.backend.status,
        adapter_artifact_path=f"artifacts/fixtures/adapter_{seed}.json",
        adapter_alias=f"fixture_adapter_{seed}",
        run_id="r" * 64,
        model_manifest_sha256="a" * 64,
        model_revision="UNRESOLVED_NOT_DOWNLOADED",
        data_manifest_sha256=data_manifest_sha256(data_document),
        corpus_role=corpus_role(data_document),
        training_config_sha256="d" * 64,
        seeds=seed_families(seed, differentially_private=dp_run is not None),
        privacy_regime="DP_SAMPLE_LEVEL" if dp_run else "NON_DP",
        audit=treated.audit,
        optimizer_steps=outcome.optimizer_steps,
        precision=treated.precision,
        adapter_file_sha256=adapter_file_hash(save_adapter(outcome.adapter)),
        induced_update_sha256=outcome.adapter.update_identity(),
        base_parameter_sha256=outcome.base_identity,
        trainer_version=outcome.trainer_version,
        dp_run=dp_run,
    )


def test_the_adapter_manifest_binds_the_run_the_data_and_the_seeds(corpus: Any, pool: Any) -> None:
    document = adapter_manifest(corpus, pool, 101)
    assert validate_adapter_manifest(document) == []
    assert document["manifest_role"] == FIXTURE_REFERENCE_ROLE
    assert document["corpus_role"] == FIXTURE_CORPUS_ROLE
    assert document["execution_contract_sha256"]
    assert document["seeds"]["data_order"] != document["seeds"]["lora_init"]


def test_a_fixture_reference_manifest_cannot_be_relabelled_evidentiary(
    corpus: Any, pool: Any
) -> None:
    """B-C5: the exact forged combination the reviewer built."""
    document = dict(adapter_manifest(corpus, pool, 101))
    document["manifest_role"] = EVIDENTIARY_ROLE
    document["model_revision"] = "main"
    problems = validate_adapter_manifest(document)
    assert any("floating branch" in p for p in problems)
    assert any("RESEARCH_CORPUS" in p for p in problems)
    assert any("NOT_RUN_DEPENDENCY" in p or "not READY" in p for p in problems)


def test_two_seeds_produce_two_different_adapter_identities(corpus: Any, pool: Any) -> None:
    first = adapter_manifest(corpus, pool, 101)
    second = adapter_manifest(corpus, pool, 202)
    assert first["adapter_file_sha256"] != second["adapter_file_sha256"]
    assert first["induced_update_sha256"] != second["induced_update_sha256"]
    assert first["base_parameter_sha256"] == second["base_parameter_sha256"]


# ------------------------------------------------------------------ the DP smoke arm
def test_the_dp_smoke_manifest_is_labelled_non_inferential(corpus: Any, pool: Any) -> None:
    sample_rate = poisson_sample_rate(batch_size=4, dataset_size=400)
    steps = steps_for(epochs=2, dataset_size=400, batch_size=4)
    mechanism = calibrate_noise_multiplier(
        target_epsilon=8.0,
        delta=1e-5,
        sample_rate=sample_rate,
        steps=steps,
        clipping_norm=1.0,
        dp_seed=seed_families(101, differentially_private=True)["dp_noise"],
    )
    run = dp_smoke_run(smoke_seed=101, mechanism=mechanism)
    document = adapter_manifest(corpus, pool, 101, dp_run=run)
    assert validate_adapter_manifest(document) == []
    block = document["dp"]
    assert block["inferential_status"] == "NON_INFERENTIAL"
    assert block["adjacency"] == SAMPLE_LEVEL_ADJACENCY
    assert block["requested_epsilon"] == 8.0
    assert 0.0 < block["achieved_epsilon"] <= 8.0
    assert block["achieved_epsilon"] != block["requested_epsilon"]


def test_the_dp_seed_family_drives_the_dp_noise_not_the_master_seed() -> None:
    families = seed_families(101, differentially_private=True)
    assert families["dp_noise"] != families.training_seed
    assert families["dp_noise"] != families["data_order"]
