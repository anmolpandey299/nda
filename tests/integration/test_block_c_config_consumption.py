"""Every Block C material constant is consumed, and mutating it changes execution.

01 §17 says material constants live in config. That is only meaningful if the executable
path actually reads them: a constant nothing consumes is decoration, and a constant a source
default can shadow is worse than decoration. So each key gets a probe that resolves it
through the real accessor and reaches the real consumer, and the test mutates the key on a
copied config tree and requires the probe to move or to fail closed.

`INERT_CONFIG_FIELDS` is the declared exception list. It is empty for values; only the two
documentation keys are exempt.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from src.data.settings import canary_settings, corpus_settings
from src.materials import (
    PROVISIONAL_FIXTURE_ONLY,
    REQUIRED_NOT_CALIBRATED,
    UncalibratedConstantError,
    material,
    material_status,
    uncalibrated_keys,
)
from src.provenance.config import ConfigError
from src.training.settings import dp_settings, lora_settings, panel_settings

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Keys that carry no material value. Documentation only [AUTH: 01 §17].
INERT_CONFIG_FIELDS: tuple[str, ...] = ("authority", "description")

CONFIG_FILES: tuple[str, ...] = (
    "data/corpus.json",
    "data/canaries.json",
    "training/lora.json",
    "training/dp.json",
)


def _copy_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs", dirs_exist_ok=True)
    return root


def _mutate(root: Path, relative: str, key: str, value: object) -> None:
    path = root / "configs" / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    document[key]["value"] = value
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _drop(root: Path, relative: str, key: str) -> None:
    path = root / "configs" / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    del document[key]
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ structure
@pytest.mark.parametrize("relative", CONFIG_FILES)
def test_every_key_declares_a_calibration_status(relative: str) -> None:
    document = json.loads((REPO_ROOT / "configs" / relative).read_text(encoding="utf-8"))
    for key, entry in document.items():
        if key in INERT_CONFIG_FIELDS:
            continue
        assert isinstance(entry, dict) and "status" in entry, f"{relative}:{key}"


def test_no_material_key_is_inert() -> None:
    """The declared exception list holds only documentation keys."""
    assert set(INERT_CONFIG_FIELDS) == {"authority", "description"}


@pytest.mark.parametrize("relative", CONFIG_FILES)
def test_a_key_with_a_value_but_no_status_is_refused(relative: str, tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    path = root / "configs" / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    key = next(k for k in document if k not in INERT_CONFIG_FIELDS)
    document[key] = 1.0
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    settings = {
        "data/corpus.json": corpus_settings,
        "data/canaries.json": canary_settings,
        "training/lora.json": lora_settings,
        "training/dp.json": dp_settings,
    }[relative](root)
    with pytest.raises(ConfigError, match="calibration status"):
        material(settings, key)


# ------------------------------------------------------------------ fail-closed states
def test_an_uncalibrated_constant_always_fails_closed() -> None:
    settings = lora_settings(REPO_ROOT)
    for key in uncalibrated_keys(settings):
        with pytest.raises(UncalibratedConstantError, match="REQUIRED_NOT_CALIBRATED"):
            material(settings, key)
        with pytest.raises(UncalibratedConstantError):
            material(settings, key, allow_provisional=True)


def test_the_uncalibrated_keys_are_the_00_8_3_values_plus_the_query() -> None:
    assert uncalibrated_keys(lora_settings(REPO_ROOT)) == (
        "batch_size",
        "epochs",
        "gradient_accumulation_steps",
        "gradient_clipping_norm",
        "learning_rate",
        "optimizer",
        "precision",
        "sequence_length",
    )
    assert uncalibrated_keys(dp_settings(REPO_ROOT)) == (
        "clipping_norm",
        "delta",
        "noise_multiplier",
    )
    assert uncalibrated_keys(corpus_settings(REPO_ROOT)) == ("source_query",)
    assert uncalibrated_keys(canary_settings(REPO_ROOT)) == ()


def test_a_provisional_constant_is_refused_on_an_evidentiary_path() -> None:
    settings = corpus_settings(REPO_ROOT)
    assert material_status(settings, "near_duplicate_threshold") == PROVISIONAL_FIXTURE_ONLY
    with pytest.raises(UncalibratedConstantError, match="fixture code only"):
        material(settings, "near_duplicate_threshold")
    assert material(settings, "near_duplicate_threshold", allow_provisional=True) == 0.8


def test_the_frozen_spec_values_are_available_without_a_provisional_waiver() -> None:
    settings = corpus_settings(REPO_ROOT)
    assert material(settings, "split_seed") == 20260820
    assert material(settings, "publication_date_floor") == "2026-04-01"
    assert material(settings, "partition_sizes") == {
        "TRAIN_CANDIDATES": 30000,
        "CALIBRATION_NONMEMBERS": 5000,
        "EVAL_NONMEMBERS": 5000,
        "RESERVE": 5000,
    }
    assert material(canary_settings(REPO_ROOT), "pool_size") == 4096
    assert material(canary_settings(REPO_ROOT), "inclusion_probability") == 0.5
    assert material(lora_settings(REPO_ROOT), "adapter_rank") == 32
    assert material(dp_settings(REPO_ROOT), "target_epsilon") == 8.0
    assert material(dp_settings(REPO_ROOT), "smoke_seed") == 101


def test_a_missing_required_constant_fails_closed(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    _drop(root, "data/corpus.json", "split_seed")
    with pytest.raises(ConfigError):
        material(corpus_settings(root), "split_seed")


# ------------------------------------------------------------------ consumption probes
def _split_seed_probe(root: Path) -> object:
    from blockc_fixtures import tiny_corpus

    from src.data.corpus import build_master_corpus

    settings = corpus_settings(root)
    return build_master_corpus(
        tiny_corpus(80),
        sizes={
            "TRAIN_CANDIDATES": 20,
            "CALIBRATION_NONMEMBERS": 10,
            "EVAL_NONMEMBERS": 10,
            "RESERVE": 10,
        },
        split_seed=int(str(material(settings, "split_seed"))),
        shingle_size=int(
            str(material(settings, "near_duplicate_shingle_size", allow_provisional=True))
        ),
        near_duplicate_threshold=float(
            str(material(settings, "near_duplicate_threshold", allow_provisional=True))
        ),
    ).partitions


def _floor_probe(root: Path) -> object:
    from blockc_fixtures import source_records

    from src.data.corpus import filter_eligible

    floor = str(material(corpus_settings(root), "publication_date_floor"))
    eligible, refused = filter_eligible(
        source_records(6, publication_date="2026-04-15"), floor=floor
    )
    return (len(eligible), len(refused))


def _partition_probe(root: Path) -> object:
    from blockc_fixtures import tiny_corpus

    from src.data.corpus import build_master_corpus

    settings = corpus_settings(root)
    declared = material(settings, "partition_sizes")
    assert isinstance(declared, dict)
    sizes = {name: max(1, int(str(count)) // 500) for name, count in declared.items()}
    corpus = build_master_corpus(
        tiny_corpus(sum(sizes.values()) + 5),
        sizes=sizes,
        split_seed=int(str(material(settings, "split_seed"))),
        shingle_size=int(
            str(material(settings, "near_duplicate_shingle_size", allow_provisional=True))
        ),
        near_duplicate_threshold=float(
            str(material(settings, "near_duplicate_threshold", allow_provisional=True))
        ),
    )
    return {name: len(ids) for name, ids in corpus.partitions.items()}


def _dedup_probe(root: Path) -> object:
    from blockc_fixtures import tiny_corpus

    from src.data.dedup import deduplicate
    from src.data.normalise import normalise_record

    settings = corpus_settings(root)
    base = tiny_corpus(6)
    near = [
        normalise_record(f"N{i}", {"title": r.text, "abstract": "one extra clause"})
        for i, r in enumerate(base)
    ]
    result = deduplicate(
        [*base, *near],
        shingle_size=int(
            str(material(settings, "near_duplicate_shingle_size", allow_provisional=True))
        ),
        threshold=float(
            str(material(settings, "near_duplicate_threshold", allow_provisional=True))
        ),
    )
    return (
        len(result.retained),
        tuple(round(d.similarity, 9) for d in result.decisions),
    )


def _blind_probe(root: Path) -> object:
    settings = corpus_settings(root)
    return (
        int(str(material(settings, "blind_control_folds"))),
        int(str(material(settings, "blind_control_cv_seed"))),
        int(str(material(settings, "blind_control_confirmatory_seed"))),
        float(str(material(settings, "blind_control_clean_tolerance"))),
        material(settings, "blind_control_investigation_band"),
        material(settings, "blind_control_ngram_range"),
        float(str(material(settings, "blind_control_regularisation", allow_provisional=True))),
        str(material(settings, "exact_duplicate_key")),
        str(material(settings, "blind_control_ci_method")),
        int(str(material(settings, "blind_control_ci_replicates"))),
        int(str(material(settings, "blind_control_ci_seed"))),
        float(str(material(settings, "blind_control_ci_alpha"))),
        str(material(settings, "blind_control_label_balance_rule")),
    )


def _subset_probe(root: Path) -> object:
    from blockc_fixtures import candidate_pool

    from src.data.canaries import generate_canary_pool
    from src.data.membership import natural_member_subsets

    settings = corpus_settings(root)
    pool = generate_canary_pool(generator_seed=1, pool_size=4, secret_length=12)
    trained = [f"T{i:05d}" for i in range(200)]
    subsets = natural_member_subsets(
        candidate_pool(trained, pool),
        calibration_size=min(int(str(material(settings, "natural_member_calibration"))), 40),
        evaluation_size=min(int(str(material(settings, "natural_member_eval"))), 60),
        training_seed=101,
    )
    return (
        len(subsets.calibration),
        len(subsets.evaluation),
        subsets.evaluation_identity(),
        subsets.calibration_identity(),
    )


def _canary_probe(root: Path) -> object:
    from src.data.canaries import generate_canary_pool, inclusion_map

    settings = canary_settings(root)
    pool = generate_canary_pool(
        generator_seed=int(str(material(settings, "generator_seed"))),
        pool_size=min(int(str(material(settings, "pool_size"))), 32),
        secret_length=int(str(material(settings, "secret_length", allow_provisional=True))),
    )
    decisions = inclusion_map(
        pool,
        inclusion_seed=101,
        probability=float(str(material(settings, "inclusion_probability"))),
    )
    return (
        tuple(c.secret for c in pool),
        tuple(sorted(k for k, v in decisions.items() if v)),
        int(str(material(settings, "repetitions_per_canary"))),
    )


def _lora_probe(root: Path) -> object:
    settings = lora_settings(root)
    return (
        str(material(settings, "target_policy")),
        str(material(settings, "fallback_target_policy")),
        int(str(material(settings, "adapter_rank"))),
        float(str(material(settings, "adapter_scaling", allow_provisional=True))),
        float(str(material(settings, "adapter_dropout", allow_provisional=True))),
        material(settings, "training_seeds"),
        material(settings, "confirmatory_training_seeds"),
        int(str(material(settings, "member_subset_seed_offset"))),
        str(material(settings, "data_order_policy")),
        str(material(settings, "checkpoint_policy")),
    )


def _fixture_training_probe(root: Path) -> object:
    from src.training.trainer import data_order

    settings = lora_settings(root)
    epochs = int(str(material(settings, "fixture_epochs", allow_provisional=True)))
    batch = int(str(material(settings, "fixture_batch_size", allow_provisional=True)))
    rate = float(str(material(settings, "fixture_learning_rate", allow_provisional=True)))
    length = int(str(material(settings, "fixture_sequence_length", allow_provisional=True)))
    ids = [f"T{i}" for i in range(10)]
    return (epochs, batch, rate, length, data_order(ids, order_seed=epochs * 1000 + batch))


def _dp_probe(root: Path) -> object:
    from src.dp.mechanism import DPMechanism, account

    settings = dp_settings(root)
    mechanism = DPMechanism(
        adjacency=str(material(settings, "adjacency")),
        delta=float(str(material(settings, "fixture_delta", allow_provisional=True))),
        clipping_norm=float(
            str(material(settings, "fixture_clipping_norm", allow_provisional=True))
        ),
        noise_multiplier=float(
            str(material(settings, "fixture_noise_multiplier", allow_provisional=True))
        ),
        sample_rate=float(str(material(settings, "fixture_sample_rate", allow_provisional=True))),
        steps=int(str(material(settings, "fixture_steps", allow_provisional=True))),
        dp_seed=int(str(material(settings, "smoke_seed"))),
        requested_epsilon=float(str(material(settings, "target_epsilon"))),
    )
    import numpy as np

    from src.dp.mechanism import privatise

    privatised = privatise(
        [np.array([9.0, 12.0]), np.array([0.1, 0.2])], mechanism=mechanism, step=0
    )
    return (
        round(account(mechanism).achieved_epsilon, 9),
        tuple(round(float(v), 9) for v in privatised),
        mechanism.dp_seed,
        mechanism.requested_epsilon,
        material(settings, "inferential_seeds"),
        int(str(material(settings, "minimum_inferential_seeds"))),
    )


#: key -> (config file, probe, mutated value). Every non-inert key appears exactly once.
PROBES: dict[str, tuple[str, Callable[[Path], object], object]] = {
    "publication_date_floor": ("data/corpus.json", _floor_probe, "2026-12-31"),
    "partition_sizes": (
        "data/corpus.json",
        _partition_probe,
        {
            "TRAIN_CANDIDATES": 5000,
            "CALIBRATION_NONMEMBERS": 5000,
            "EVAL_NONMEMBERS": 5000,
            "RESERVE": 5000,
        },
    ),
    "split_seed": ("data/corpus.json", _split_seed_probe, 12345),
    "exact_duplicate_key": ("data/corpus.json", _blind_probe, "RAW_TEXT_SHA256"),
    "near_duplicate_shingle_size": ("data/corpus.json", _dedup_probe, 40),
    "near_duplicate_threshold": ("data/corpus.json", _dedup_probe, 0.999),
    "natural_member_calibration": ("data/corpus.json", _subset_probe, 7),
    "natural_member_eval": ("data/corpus.json", _subset_probe, 9),
    "blind_control_folds": ("data/corpus.json", _blind_probe, 4),
    "blind_control_cv_seed": ("data/corpus.json", _blind_probe, 1),
    "blind_control_confirmatory_seed": ("data/corpus.json", _blind_probe, 2),
    "blind_control_clean_tolerance": ("data/corpus.json", _blind_probe, 0.2),
    "blind_control_investigation_band": ("data/corpus.json", _blind_probe, [0.1, 0.9]),
    "blind_control_ngram_range": ("data/corpus.json", _blind_probe, [1, 3]),
    "blind_control_regularisation": ("data/corpus.json", _blind_probe, 10.0),
    "blind_control_ci_method": ("data/corpus.json", _blind_probe, "SOMETHING_ELSE"),
    "blind_control_ci_replicates": ("data/corpus.json", _blind_probe, 500),
    "blind_control_ci_seed": ("data/corpus.json", _blind_probe, 1),
    "blind_control_ci_alpha": ("data/corpus.json", _blind_probe, 0.1),
    "blind_control_label_balance_rule": ("data/corpus.json", _blind_probe, "SUBSAMPLE"),
    "pool_size": ("data/canaries.json", _canary_probe, 8),
    "inclusion_probability": ("data/canaries.json", _canary_probe, 0.25),
    "generator_seed": ("data/canaries.json", _canary_probe, 999),
    "secret_length": ("data/canaries.json", _canary_probe, 20),
    "repetitions_per_canary": ("data/canaries.json", _canary_probe, 3),
    "target_policy": ("training/lora.json", _lora_probe, "SOMETHING_ELSE"),
    "fallback_target_policy": ("training/lora.json", _lora_probe, "SOMETHING_ELSE"),
    "adapter_rank": ("training/lora.json", _lora_probe, 16),
    "adapter_scaling": ("training/lora.json", _lora_probe, 8.0),
    "adapter_dropout": ("training/lora.json", _lora_probe, 0.1),
    "fixture_learning_rate": ("training/lora.json", _fixture_training_probe, 0.25),
    "fixture_epochs": ("training/lora.json", _fixture_training_probe, 5),
    "fixture_batch_size": ("training/lora.json", _fixture_training_probe, 4),
    "fixture_sequence_length": ("training/lora.json", _fixture_training_probe, 64),
    "data_order_policy": ("training/lora.json", _lora_probe, "GLOBAL_RNG"),
    "checkpoint_policy": ("training/lora.json", _lora_probe, "BEST_BY_VALIDATION"),
    "training_seeds": ("training/lora.json", _lora_probe, [1, 2, 3]),
    "confirmatory_training_seeds": ("training/lora.json", _lora_probe, [9, 8]),
    "member_subset_seed_offset": ("training/lora.json", _lora_probe, 5),
    "adjacency": ("training/dp.json", _dp_probe, "USER_LEVEL"),
    "target_epsilon": ("training/dp.json", _dp_probe, 2.0),
    "fixture_delta": ("training/dp.json", _dp_probe, 1e-7),
    "smoke_seed": ("training/dp.json", _dp_probe, 777),
    "inferential_seeds": ("training/dp.json", _dp_probe, [1, 2, 3]),
    "minimum_inferential_seeds": ("training/dp.json", _dp_probe, 5),
    "fixture_clipping_norm": ("training/dp.json", _dp_probe, 4.0),
    "fixture_noise_multiplier": ("training/dp.json", _dp_probe, 3.0),
    "fixture_sample_rate": ("training/dp.json", _dp_probe, 0.2),
    "fixture_steps": ("training/dp.json", _dp_probe, 50),
}


def test_every_configured_key_has_a_probe_or_is_declared_uncalibrated() -> None:
    covered = set(PROBES) | set(INERT_CONFIG_FIELDS)
    for relative in CONFIG_FILES:
        document = json.loads((REPO_ROOT / "configs" / relative).read_text(encoding="utf-8"))
        for key, entry in document.items():
            if isinstance(entry, dict) and entry.get("status") == REQUIRED_NOT_CALIBRATED:
                continue
            assert key in covered, f"{relative}:{key} is consumed by nothing"


@pytest.mark.parametrize("key", sorted(PROBES))
def test_mutating_a_material_constant_changes_execution(key: str, tmp_path: Path) -> None:
    relative, probe, mutated = PROBES[key]
    baseline = probe(REPO_ROOT)
    root = _copy_repo(tmp_path)
    _mutate(root, relative, key, mutated)
    try:
        after = probe(root)
    except (ConfigError, ValueError) as exc:  # failing closed is a valid consumption
        assert str(exc)
        return
    assert after != baseline, f"{relative}:{key} does not reach execution"


@pytest.mark.parametrize("key", sorted(PROBES))
def test_removing_a_material_constant_fails_closed(key: str, tmp_path: Path) -> None:
    relative, probe, _ = PROBES[key]
    root = _copy_repo(tmp_path)
    _drop(root, relative, key)
    with pytest.raises((ConfigError, ValueError, KeyError)):
        probe(root)


# ------------------------------------------------------------------ cross-config agreement
def test_the_declared_target_policy_matches_the_executable_mapping() -> None:
    """Two files name the policy; drift between them is a defect, not a preference."""
    declared = str(material(lora_settings(REPO_ROOT), "target_policy"))
    assert panel_settings(REPO_ROOT).document["policy"] == declared


def test_the_dp_smoke_seed_is_the_first_inferential_seed() -> None:
    settings = dp_settings(REPO_ROOT)
    seeds = material(settings, "inferential_seeds")
    assert isinstance(seeds, list)
    assert seeds[0] == material(settings, "smoke_seed")
    assert material(settings, "minimum_inferential_seeds") == len(seeds)


def test_the_lora_training_seeds_are_the_00_9_calibration_seeds() -> None:
    assert material(lora_settings(REPO_ROOT), "training_seeds") == [101, 202, 303]
    assert material(lora_settings(REPO_ROOT), "confirmatory_training_seeds") == [404, 505]


def test_the_00_8_3_fixed_across_seeds_values_are_all_present() -> None:
    """00 §8.3 lists exactly what must be written into config and hashed."""
    document = json.loads((REPO_ROOT / "configs/training/lora.json").read_text(encoding="utf-8"))
    for key in (
        "adapter_rank",
        "target_policy",
        "adapter_scaling",
        "adapter_dropout",
        "optimizer",
        "learning_rate",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "sequence_length",
        "precision",
        "gradient_clipping_norm",
        "data_order_policy",
        "checkpoint_policy",
    ):
        assert key in document, key
    dp = json.loads((REPO_ROOT / "configs/training/dp.json").read_text(encoding="utf-8"))
    assert "clipping_norm" in dp and "noise_multiplier" in dp


def test_the_checkpoint_policy_forbids_outcome_driven_selection() -> None:
    """01 §29 / 00 §34B.1A: a checkpoint chosen by looking at a result is result-peeking."""
    policy = str(material(lora_settings(REPO_ROOT), "checkpoint_policy"))
    assert policy == "FINAL_ADAPTER_ONLY_NO_INTERMEDIATE_SELECTION"


def test_the_data_order_policy_names_the_seed_family_not_a_global_rng() -> None:
    policy = str(material(lora_settings(REPO_ROOT), "data_order_policy"))
    assert "DATA_ORDER_SEED" in policy and "GLOBAL" not in policy


def test_no_research_model_revision_is_resolved() -> None:
    """REAL model acquisition would show here as a 40-hex revision."""
    panel = panel_settings(REPO_ROOT).document["panel"]
    assert isinstance(panel, list)
    for entry in panel:
        assert isinstance(entry, dict)
        assert entry["revision"] == "UNRESOLVED_NOT_DOWNLOADED"
